"""State machine for the declarative visual thought-experiment compiler."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, MutableMapping

from jump_contracts.thought_experiments import (
    CONFIRMATION_VERSION,
    ENGINE_ID,
    QUESTION_VERSION,
    RUN_RESPONSE_VERSION,
    SPEC_RESPONSE_VERSION,
    ThoughtExperimentContractError,
    build_experiment_spec,
    build_thought_experiment_run,
    canonical_json,
    validate_experiment_spec,
)

from .workflow import FrozenModel, WorkbenchError, validate_user_intent

QUESTION_FIELDS = frozenset({"schema_version", "request_id", "session_id", "intent", "seed", "repetitions"})
CONFIRMATION_FIELDS = frozenset({"schema_version", "spec_id", "spec_sha256", "confirmed"})


class VisualCoordinatorError(ValueError):
    pass


class VisualCoordinator:
    def __init__(
        self,
        *,
        state: MutableMapping[str, Any],
        model: FrozenModel,
        transformers_revision: str,
        model_generate: Callable[[str, dict[str, Any]], dict[str, Any]],
        simulate: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
        code_version: str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        model.validate()
        self.state = state
        self.model = model
        self.transformers_revision = transformers_revision
        self.model_generate = model_generate
        self.simulate = simulate
        self.code_version = code_version
        self.now = now

    @property
    def model_identity(self) -> dict[str, Any]:
        return {
            "repo_id": self.model.model_id,
            "revision": self.model.revision,
            "transformers_revision": self.transformers_revision,
            "frozen": True,
            "adapter_id": None,
        }

    def compile(self, body: Any) -> dict[str, Any]:
        request = _validate_question(body)
        try:
            generated = self.model_generate("visual_spec", request)
            if not isinstance(generated, dict):
                raise VisualCoordinatorError("compiler must return a JSON object")
            if set(generated) == {"unsupported"}:
                raise VisualCoordinatorError(f"unsupported thought experiment: {generated['unsupported']}")
            expected = {
                "question", "hypothesis", "world", "dynamics", "conditions",
                "schedule", "measurements", "visualization",
            }
            if set(generated) != expected:
                raise VisualCoordinatorError("compiler output fields do not match ExperimentSpec v2")
            generated = _complete_nullable_spec_fields(generated)
            schedule = dict(generated["schedule"])
            schedule["seed"] = request["seed"]
            schedule["repetitions"] = request["repetitions"]
            spec = build_experiment_spec(
                intent=request["intent"],
                question=generated["question"],
                hypothesis=generated["hypothesis"],
                world=generated["world"],
                dynamics=generated["dynamics"],
                conditions=generated["conditions"],
                schedule=schedule,
                measurements=generated["measurements"],
                visualization=generated["visualization"],
            )
        except (TypeError, ValueError, WorkbenchError, ThoughtExperimentContractError) as exc:
            raise VisualCoordinatorError(f"visual compiler output rejected: {exc}") from exc
        planned = {"state": "awaiting_confirmation", "request": request, "spec": spec}
        self.state[spec["spec_id"]] = planned
        return {
            "schema_version": SPEC_RESPONSE_VERSION,
            "status": "awaiting_confirmation",
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "spec": spec,
            "model": self.model_identity,
            "confirmation": {
                "schema_version": CONFIRMATION_VERSION,
                "spec_id": spec["spec_id"],
                "spec_sha256": spec["spec_sha256"],
                "confirmed": False,
            },
        }

    def confirm(self, body: Any) -> dict[str, Any]:
        confirmation = _validate_confirmation(body)
        planned = self.state.get(confirmation["spec_id"])
        if not isinstance(planned, dict) or planned.get("state") != "awaiting_confirmation":
            raise VisualCoordinatorError("spec is unknown or no longer awaiting confirmation")
        spec = validate_experiment_spec(planned["spec"])
        if confirmation["spec_sha256"] != spec["spec_sha256"]:
            raise VisualCoordinatorError("confirmation does not bind the sealed ExperimentSpec")
        try:
            prediction = _validate_prediction(self.model_generate("visual_predict", {"spec": spec}), spec)
            prediction_recorded_at = _timestamp(self.now())
            self.state[spec["spec_id"]] = {
                **planned,
                "state": "prediction_ready",
                "prediction": prediction,
                "prediction_recorded_at": prediction_recorded_at,
            }
            execution = self.simulate(spec, prediction, prediction_recorded_at)
            if not isinstance(execution, dict) or set(execution) != {
                "modal_call_id", "started_at", "completed_at", "result"
            }:
                raise VisualCoordinatorError("visual engine returned an invalid execution record")
            result = execution["result"]
            if not isinstance(result, dict) or set(result) != {"conditions", "comparisons"}:
                raise VisualCoordinatorError("visual engine result fields are invalid")
            revision = self.model_generate("visual_review", {
                "prediction": prediction,
                "comparisons": result["comparisons"],
            })
            revision = _validate_revision(revision)
            evidence = {
                "spec_sha256": spec["spec_sha256"],
                "engine_id": ENGINE_ID,
                "code_version": self.code_version,
                "modal_call_id": execution["modal_call_id"],
                "result_sha256": "0" * 64,
                "sealed_payload_sha256": "0" * 64,
            }
            run = build_thought_experiment_run(
                spec,
                spec_id=spec["spec_id"],
                spec_sha256=spec["spec_sha256"],
                status="completed",
                execution={
                    "engine_id": ENGINE_ID,
                    "code_version": self.code_version,
                    "modal_call_id": execution["modal_call_id"],
                    "prediction": prediction,
                    "prediction_recorded_at": prediction_recorded_at,
                    "started_at": _timestamp(_as_datetime(execution["started_at"])),
                    "completed_at": _timestamp(_as_datetime(execution["completed_at"])),
                    "error": None,
                },
                conditions=result["conditions"],
                comparisons=result["comparisons"],
                revision=revision,
                evidence=evidence,
            )
        except VisualCoordinatorError:
            raise
        except (TypeError, ValueError, ThoughtExperimentContractError) as exc:
            raise VisualCoordinatorError(f"confirmed visual experiment failed closed: {exc}") from exc
        request = planned["request"]
        self.state[spec["spec_id"]] = {
            "state": "completed", "request": request, "spec": spec, "run": run,
        }
        return {
            "schema_version": RUN_RESPONSE_VERSION,
            "status": "completed",
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "spec": spec,
            "run": run,
            "model": self.model_identity,
        }


def _validate_question(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != QUESTION_FIELDS:
        raise VisualCoordinatorError(f"v2 question must contain exactly {sorted(QUESTION_FIELDS)}")
    if value["schema_version"] != QUESTION_VERSION:
        raise VisualCoordinatorError("unsupported v2 question schema_version")
    result = dict(value)
    try:
        result["intent"] = validate_user_intent(result["intent"])
    except WorkbenchError as exc:
        raise VisualCoordinatorError(str(exc)) from exc
    for key in ("request_id", "session_id"):
        if not isinstance(result[key], str) or not result[key] or len(result[key]) > 128:
            raise VisualCoordinatorError(f"{key} must be bounded nonempty text")
    if not isinstance(result["seed"], int) or isinstance(result["seed"], bool) or not 0 <= result["seed"] <= 2147483647:
        raise VisualCoordinatorError("seed must be an integer from 0 through 2147483647")
    if not isinstance(result["repetitions"], int) or isinstance(result["repetitions"], bool) or not 1 <= result["repetitions"] <= 20:
        raise VisualCoordinatorError("repetitions must be an integer from 1 through 20")
    return result


def _validate_confirmation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CONFIRMATION_FIELDS:
        raise VisualCoordinatorError(f"v2 confirmation must contain exactly {sorted(CONFIRMATION_FIELDS)}")
    if value["schema_version"] != CONFIRMATION_VERSION or value["confirmed"] is not True:
        raise VisualCoordinatorError("v2 confirmation must explicitly confirm the exact spec")
    if not isinstance(value["spec_id"], str) or not isinstance(value["spec_sha256"], str):
        raise VisualCoordinatorError("v2 confirmation identities must be text")
    return dict(value)


def _validate_prediction(value: Any, spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"summary", "expected_direction", "measurement_id"}:
        raise VisualCoordinatorError("visual prediction fields are invalid")
    if not isinstance(value["summary"], str) or not value["summary"].strip() or len(value["summary"]) > 500:
        raise VisualCoordinatorError("visual prediction summary must contain 1 through 500 characters")
    if value["expected_direction"] not in {"increase", "decrease", "change", "no_change"}:
        raise VisualCoordinatorError("visual prediction direction is invalid")
    if value["measurement_id"] not in {item["id"] for item in spec["measurements"]}:
        raise VisualCoordinatorError("visual prediction measurement is not declared")
    return {**value, "summary": value["summary"].strip()}


def _validate_revision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"disposition", "interpretation"}:
        raise VisualCoordinatorError("visual revision fields are invalid")
    if value["disposition"] not in {"retain", "revise", "reject"}:
        raise VisualCoordinatorError("visual revision disposition is invalid")
    if not isinstance(value["interpretation"], str) or not value["interpretation"].strip() or len(value["interpretation"]) > 500:
        raise VisualCoordinatorError("visual interpretation must contain 1 through 500 characters")
    return {**value, "interpretation": value["interpretation"].strip()}


def _complete_nullable_spec_fields(generated: dict[str, Any]) -> dict[str, Any]:
    """Materialize required nullable keys; never infer an operation or value."""
    value = dict(generated)
    world = dict(value["world"])
    world["entities"] = [
        {
            **item,
            "initial_state": {
                "numeric": item["initial_state"].get("numeric", {}),
                "categorical": item["initial_state"].get("categorical", {}),
            },
        }
        for item in world["entities"]
    ]
    value["world"] = world
    dynamics = dict(value["dynamics"])
    dynamics["rules"] = [
        {**item, "target_type": item.get("target_type")}
        for item in dynamics["rules"]
    ]
    value["dynamics"] = dynamics
    value["measurements"] = [
        {
            **item,
            "entity_type": item.get("entity_type"),
            "state": item.get("state"),
            "category": item.get("category"),
        }
        for item in value["measurements"]
    ]
    return value


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise VisualCoordinatorError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed
    raise VisualCoordinatorError("visual engine timestamps must be timezone-aware")


__all__ = ["VisualCoordinator", "VisualCoordinatorError"]
