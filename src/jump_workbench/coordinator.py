"""Authenticated coordinator state machine for the general workbench."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, MutableMapping

from jump_contracts.experiments import comparison_records, validate_experiment_plan, validate_experiment_run

from .workflow import (
    FrozenModel, WorkbenchError, confirm_and_predict, finalize_run, prepare_plan,
    validate_user_intent,
)

PLAN_RESPONSE_VERSION = "jump.experiment-plan-response/v1"
CONFIRMATION_VERSION = "jump.experiment-confirmation/v1"
RUN_RESPONSE_VERSION = "jump.experiment-run-response/v1"
QUESTION_FIELDS = frozenset({"schema_version", "request_id", "session_id", "intent", "seed", "repetitions"})
CONFIRMATION_FIELDS = frozenset({"schema_version", "plan_id", "plan_sha256", "confirmed"})


class CoordinatorError(ValueError):
    """Raised when an HTTP action cannot advance the sealed workflow."""


class GeneralCoordinator:
    def __init__(
        self,
        *,
        state: MutableMapping[str, Any],
        model: FrozenModel,
        transformers_revision: str,
        model_generate: Callable[[str, dict[str, Any]], dict[str, Any]],
        simulate: Callable[[dict[str, Any], str, dict[str, Any], dict[str, Any]], dict[str, Any]],
        code_version: str,
    ) -> None:
        model.validate()
        self.state = state
        self.model = model
        self.transformers_revision = transformers_revision
        self.model_generate = model_generate
        self.simulate = simulate
        self.code_version = code_version

    @property
    def model_identity(self) -> dict[str, Any]:
        return {
            "repo_id": self.model.model_id,
            "revision": self.model.revision,
            "transformers_revision": self.transformers_revision,
            "frozen": True,
            "adapter_id": None,
        }

    def plan(self, body: Any) -> dict[str, Any]:
        request = _validate_question(body)
        try:
            generated = self.model_generate("plan", request)
            planned = prepare_plan(
                request["intent"],
                session_id=request["session_id"],
                request_id=request["request_id"],
                seed=request["seed"],
                repetitions=request["repetitions"],
                model=self.model,
                planner=lambda _request: generated,
            )
            plan = validate_experiment_plan(planned["plan"])
        except (TypeError, ValueError, WorkbenchError) as exc:
            raise CoordinatorError(f"planner output rejected: {exc}") from exc
        self.state[plan["plan_id"]] = planned
        return {
            "schema_version": PLAN_RESPONSE_VERSION,
            "status": "awaiting_confirmation",
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "plan": plan,
            "model": self.model_identity,
            "confirmation": {
                "schema_version": CONFIRMATION_VERSION,
                "plan_id": plan["plan_id"],
                "plan_sha256": plan["plan_sha256"],
                "confirmed": False,
            },
        }

    def confirm(self, body: Any) -> dict[str, Any]:
        confirmation = _validate_confirmation(body)
        planned = self.state.get(confirmation["plan_id"])
        if not isinstance(planned, dict) or planned.get("state") != "awaiting_confirmation":
            raise CoordinatorError("plan is unknown or no longer awaiting confirmation")
        plan = validate_experiment_plan(planned["plan"])
        if confirmation["plan_sha256"] != plan["plan_sha256"]:
            raise CoordinatorError("confirmation does not match the sealed plan")
        try:
            prediction = self.model_generate("predict", {"plan": plan})
            prepared = confirm_and_predict(
                planned,
                confirmed=True,
                model=self.model,
                predictor=lambda _plan: prediction,
            )
            self.state[plan["plan_id"]] = {
                **planned,
                "state": "prediction_ready",
                "prediction": prepared.prediction,
                "prediction_recorded_at": prepared.prediction_recorded_at,
            }
            execution = self.simulate(
                prepared.plan,
                prepared.source,
                prepared.confirmation,
                prepared.prediction,
            )
            if not isinstance(execution, dict) or set(execution) != {
                "modal_call_id", "started_at", "completed_at", "result"
            }:
                raise CoordinatorError("restricted simulator returned an invalid execution record")
            revision = self.model_generate(
                "review",
                {
                    "plan": prepared.plan,
                    "prediction": prepared.prediction,
                    "measurements": execution["result"]["measurements"],
                    "comparisons": comparison_records(
                        prepared.plan, execution["result"]["measurements"]
                    ),
                },
            )
            run = finalize_run(
                prepared,
                execution["result"],
                modal_call_id=execution["modal_call_id"],
                code_version=self.code_version,
                started_at=_as_datetime(execution["started_at"]),
                completed_at=_as_datetime(execution["completed_at"]),
                model=self.model,
                reviewer=lambda _review: revision,
            )
            validate_experiment_run(
                run,
                prepared.plan,
                verified_run_result=_normative_result(run),
                artifact_bytes={},
                sealed_result=_sealed_result(run),
            )
        except CoordinatorError:
            raise
        except (TypeError, ValueError, WorkbenchError) as exc:
            raise CoordinatorError(f"confirmed experiment failed closed: {exc}") from exc
        request = planned["request"]
        self.state[plan["plan_id"]] = {
            "state": "completed",
            "request": request,
            "plan": plan,
            "run": run,
        }
        return {
            "schema_version": RUN_RESPONSE_VERSION,
            "status": "completed",
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "plan": plan,
            "run": run,
            "model": self.model_identity,
        }


def _validate_question(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != QUESTION_FIELDS:
        raise CoordinatorError(f"question request must contain exactly {sorted(QUESTION_FIELDS)}")
    if value["schema_version"] != "jump.experiment-question/v1":
        raise CoordinatorError("unsupported question schema_version")
    for key in ("request_id", "session_id"):
        if not isinstance(value[key], str) or not value[key] or len(value[key]) > 128:
            raise CoordinatorError(f"{key} must be bounded nonempty text")
    try:
        value = dict(value)
        value["intent"] = validate_user_intent(value["intent"])
    except WorkbenchError as exc:
        raise CoordinatorError(str(exc)) from exc
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool) or not 0 <= value["seed"] <= 2_147_483_647:
        raise CoordinatorError("seed must be an integer from 0 through 2147483647")
    if not isinstance(value["repetitions"], int) or isinstance(value["repetitions"], bool) or not 1 <= value["repetitions"] <= 100:
        raise CoordinatorError("repetitions must be an integer from 1 through 100")
    return value


def _validate_confirmation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CONFIRMATION_FIELDS:
        raise CoordinatorError(f"confirmation must contain exactly {sorted(CONFIRMATION_FIELDS)}")
    if value["schema_version"] != CONFIRMATION_VERSION or value["confirmed"] is not True:
        raise CoordinatorError("confirmation must explicitly confirm jump.experiment-confirmation/v1")
    if not isinstance(value["plan_id"], str) or not isinstance(value["plan_sha256"], str):
        raise CoordinatorError("confirmation plan identities must be text")
    return dict(value)


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed
    raise CoordinatorError("execution timestamps must be timezone-aware")


def _normative_result(run: dict[str, Any]) -> dict[str, Any]:
    execution = run["execution"]
    return {
        "schema_version": "jump.run-result/v1",
        "status": run["status"],
        "metrics": [{"name": item["id"], "value": item["estimate"]} for item in run["comparisons"]],
        "artifacts": [],
        "provenance": {
            "manifest_sha256": run["plan_sha256"],
            "run_id": execution["modal_call_id"],
            "code_version": execution["code_version"],
            "source_sha256": execution["source_sha256"],
            "policy_sha256": execution["policy_sha256"],
        },
        "plan_sha256": run["plan_sha256"],
        "measurements": run["measurements"],
        "comparisons": run["comparisons"],
    }


def _sealed_result(run: dict[str, Any]) -> dict[str, Any]:
    from jump_contracts.evidence import seal_result_envelope

    execution = run["execution"]
    payload = {
        "plan_sha256": run["plan_sha256"],
        "prediction": execution["prediction"],
        "measurements": run["measurements"],
        "comparisons": run["comparisons"],
        "revision": run["revision"],
    }
    return seal_result_envelope(
        payload,
        source="live",
        manifest_sha256=run["plan_sha256"],
        run_id=execution["modal_call_id"],
        code_version=execution["code_version"],
        checkpoint_id=run["plan_id"],
    )


__all__ = [
    "PLAN_RESPONSE_VERSION", "CONFIRMATION_VERSION", "RUN_RESPONSE_VERSION",
    "CoordinatorError", "GeneralCoordinator",
]
