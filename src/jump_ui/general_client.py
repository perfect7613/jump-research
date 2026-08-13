"""Fail-closed client for the authenticated general experiment coordinator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping
import urllib.error
import urllib.request

from jump_contracts.experiments import validate_experiment_plan, validate_experiment_run
from jump_contracts.evidence import seal_result_envelope

QUESTION_VERSION = "jump.experiment-question/v1"
PLAN_RESPONSE_VERSION = "jump.experiment-plan-response/v1"
CONFIRMATION_VERSION = "jump.experiment-confirmation/v1"
RUN_RESPONSE_VERSION = "jump.experiment-run-response/v1"


class GeneralCoordinatorError(RuntimeError):
    """The coordinator request or response could not be safely accepted."""


@dataclass(frozen=True)
class GeneralCoordinatorClient:
    base_url: str
    token: str
    timeout_seconds: float = 300.0

    @classmethod
    def from_environment(cls) -> "GeneralCoordinatorClient":
        base_url = os.environ.get("JUMP_GENERAL_COORDINATOR_URL", "").strip()
        token = os.environ.get("JUMP_MODAL_TOKEN", "").strip()
        if not base_url or not token:
            raise GeneralCoordinatorError(
                "The general coordinator is not configured. No result was substituted."
            )
        return cls(base_url=base_url, token=token)

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = {"schema_version", "request_id", "session_id", "intent", "seed", "repetitions"}
        if set(request) != expected or request.get("schema_version") != QUESTION_VERSION:
            raise GeneralCoordinatorError("The experiment question fields do not match the coordinator API.")
        payload = self._post("/v1/experiments/plan", dict(request))
        if set(payload) != {"schema_version", "status", "request_id", "session_id", "plan", "model", "confirmation"}:
            raise GeneralCoordinatorError("The plan response contains unexpected fields.")
        if payload["schema_version"] != PLAN_RESPONSE_VERSION or payload["status"] != "awaiting_confirmation":
            raise GeneralCoordinatorError("The plan response state is invalid.")
        if payload["request_id"] != request["request_id"] or payload["session_id"] != request["session_id"]:
            raise GeneralCoordinatorError("The plan response does not bind the request.")
        payload["plan"] = _validated_plan(payload["plan"])
        _validate_model(payload["model"])
        _validate_confirmation(payload["confirmation"], payload["plan"], confirmed=False)
        return payload

    def confirm(self, planned: Mapping[str, Any]) -> dict[str, Any]:
        if planned.get("schema_version") != PLAN_RESPONSE_VERSION or planned.get("status") != "awaiting_confirmation":
            raise GeneralCoordinatorError("Only an awaiting-confirmation plan can run.")
        plan = _validated_plan(planned.get("plan", {}))
        _validate_model(planned.get("model"))
        confirmation = dict(planned.get("confirmation", {}))
        _validate_confirmation(confirmation, plan, confirmed=False)
        confirmation["confirmed"] = True
        payload = self._post("/v1/experiments/confirm", confirmation)
        if set(payload) != {"schema_version", "status", "request_id", "session_id", "plan", "run", "model"}:
            raise GeneralCoordinatorError("The run response contains unexpected fields.")
        if payload["schema_version"] != RUN_RESPONSE_VERSION or payload["status"] != "completed":
            raise GeneralCoordinatorError("The run response state is invalid.")
        if payload["request_id"] != planned["request_id"] or payload["session_id"] != planned["session_id"]:
            raise GeneralCoordinatorError("The run response does not bind the request.")
        returned_plan = _validated_plan(payload["plan"])
        if returned_plan != plan:
            raise GeneralCoordinatorError("The run response changed the confirmed plan.")
        if payload["model"] != planned["model"]:
            raise GeneralCoordinatorError("The run response changed the frozen model identity.")
        _validate_model(payload["model"])
        try:
            payload["run"] = validate_run_response(payload["run"], returned_plan)
        except (KeyError, TypeError, ValueError) as exc:
            raise GeneralCoordinatorError(f"The run response failed direct validation: {exc}") from exc
        return payload

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _http_detail(exc)
            raise GeneralCoordinatorError(f"Coordinator request failed (HTTP {exc.code}): {detail}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GeneralCoordinatorError(
                f"Coordinator request failed closed ({type(exc).__name__}). No result was substituted."
            ) from exc
        if not isinstance(value, dict):
            raise GeneralCoordinatorError("The coordinator response must be a JSON object.")
        return value


def validate_run_response(run: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Directly validate a response run against its returned confirmed plan."""
    execution = run.get("execution", {})
    comparisons = run.get("comparisons", [])
    run_result = {
        "schema_version": "jump.run-result/v1",
        "status": run.get("status"),
        "metrics": [
            {"name": comparison["id"], "value": comparison["estimate"]}
            for comparison in comparisons
        ],
        "artifacts": [],
        "provenance": {
            "manifest_sha256": run.get("plan_sha256"),
            "run_id": execution.get("modal_call_id"),
            "code_version": execution.get("code_version"),
            "source_sha256": execution.get("source_sha256"),
            "policy_sha256": execution.get("policy_sha256"),
        },
        "plan_sha256": run.get("plan_sha256"),
        "measurements": run.get("measurements"),
        "comparisons": comparisons,
    }
    sealed_result = seal_result_envelope(
        {
            "plan_sha256": run.get("plan_sha256"),
            "prediction": execution.get("prediction"),
            "measurements": run.get("measurements"),
            "comparisons": comparisons,
            "revision": run.get("revision"),
        },
        source="live",
        manifest_sha256=run.get("plan_sha256"),
        run_id=execution.get("modal_call_id"),
        code_version=execution.get("code_version"),
        checkpoint_id=run.get("plan_id"),
    )
    return validate_experiment_run(
        run,
        plan,
        verified_run_result=run_result,
        artifact_bytes={},
        sealed_result=sealed_result,
    )


def _validate_model(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "repo_id", "revision", "transformers_revision", "frozen", "adapter_id"
    }:
        raise GeneralCoordinatorError("The frozen model identity fields are invalid.")
    if not all(isinstance(value[key], str) and value[key] for key in ("repo_id", "revision", "transformers_revision")):
        raise GeneralCoordinatorError("The frozen model identity is incomplete.")
    if value["frozen"] is not True or value["adapter_id"] is not None:
        raise GeneralCoordinatorError("The coordinator must use a frozen base model without an adapter.")


def _validated_plan(value: Any) -> dict[str, Any]:
    try:
        return validate_experiment_plan(value)
    except (TypeError, ValueError) as exc:
        raise GeneralCoordinatorError(f"The plan response failed direct validation: {exc}") from exc


def _validate_confirmation(value: Any, plan: Mapping[str, Any], *, confirmed: bool) -> None:
    if not isinstance(value, dict) or set(value) != {"schema_version", "plan_id", "plan_sha256", "confirmed"}:
        raise GeneralCoordinatorError("The confirmation fields are invalid.")
    if value != {
        "schema_version": CONFIRMATION_VERSION,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "confirmed": confirmed,
    }:
        raise GeneralCoordinatorError("The confirmation does not bind the exact plan.")


def _http_detail(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "request rejected"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) and detail else "request rejected"
