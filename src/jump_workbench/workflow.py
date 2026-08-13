"""State transitions for confirmed, prediction-first computational experiments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from jump_contracts.experiments import (
    ExperimentContractError,
    build_experiment_plan,
    build_experiment_run,
    canonical_json,
    comparison_records,
    validate_experiment_plan,
)
from jump_contracts.evidence import seal_result_envelope

from .safety import sandbox_declaration

QUESTION_VERSION = "jump.experiment-question/v1"
MAX_INTENT_CHARS = 2_000

_URL_OR_PATH = re.compile(r"(?:https?://|www\.|file://|(?:^|\s)[~/][^\s]+|[A-Za-z]:\\)", re.I)
_CODE = re.compile(r"(?:```|\b(?:import|from)\s+[A-Za-z_]|\bdef\s+[A-Za-z_]|\b(?:eval|exec|open)\s*\()", re.I)
_REAL_WORLD = re.compile(
    r"\b(?:scrape|crawl|download|upload|send an? (?:email|message)|place an? (?:order|trade)|"
    r"control (?:a )?(?:device|robot|vehicle)|collect (?:live|personal|patient|sensor) data|"
    r"experiment on (?:people|patients|animals)|clinical trial|wet[- ]lab)\b",
    re.I,
)


class WorkbenchError(ValueError):
    """Raised for an invalid workbench state transition."""


class ConfirmationRequired(WorkbenchError):
    """Raised when execution is attempted before explicit confirmation."""


@dataclass(frozen=True)
class FrozenModel:
    """Identity for a frozen base Gemma planner, predictor, or reviewer."""

    model_id: str
    revision: str
    family: str = "gemma"
    frozen: bool = True
    adapter_id: None = None

    def validate(self) -> None:
        if self.family != "gemma" or not self.frozen or self.adapter_id is not None:
            raise WorkbenchError("general planning uses a frozen base Gemma checkpoint without a research adapter")
        if not self.model_id.startswith("google/gemma"):
            raise WorkbenchError("general planning requires an identified Gemma base checkpoint")
        if re.fullmatch(r"[0-9a-f]{40,64}", self.revision) is None:
            raise WorkbenchError("frozen model revision must be an immutable commit SHA")


@dataclass(frozen=True)
class PreparedExecution:
    state: str
    plan: dict[str, Any]
    source: str
    prediction: dict[str, Any]
    prediction_recorded_at: str
    confirmation: dict[str, Any]
    model: FrozenModel


Planner = Callable[[dict[str, Any]], dict[str, Any]]
Predictor = Callable[[dict[str, Any]], dict[str, Any]]
Reviewer = Callable[[dict[str, Any]], dict[str, Any]]


def validate_user_intent(intent: str) -> str:
    """Keep the user's question inert and inside the simulation-only scope."""
    if not isinstance(intent, str):
        raise WorkbenchError("intent must be text")
    normalized = " ".join(intent.split())
    if not normalized or len(normalized) > MAX_INTENT_CHARS:
        raise WorkbenchError("intent must contain 1 to 2000 characters")
    if _URL_OR_PATH.search(normalized):
        raise WorkbenchError("URLs and file paths are outside the computational experiment scope")
    if _CODE.search(normalized):
        raise WorkbenchError("user-supplied code is not accepted")
    if _REAL_WORLD.search(normalized):
        raise WorkbenchError("only toy computational simulations are supported")
    return normalized


def prepare_plan(
    intent: str,
    *,
    session_id: str,
    request_id: str,
    seed: int,
    repetitions: int,
    model: FrozenModel,
    planner: Planner,
) -> dict[str, Any]:
    """Ask a frozen model for a plan, then replace all server-owned fields."""
    normalized = validate_user_intent(intent)
    model.validate()
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2_147_483_647:
        raise WorkbenchError("seed must be an integer from 0 through 2147483647")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or not 1 <= repetitions <= 100:
        raise WorkbenchError("repetitions must be an integer from 1 through 100")
    request = {
        "schema_version": QUESTION_VERSION,
        "request_id": request_id,
        "session_id": session_id,
        "intent": normalized,
        "seed": seed,
        "repetitions": repetitions,
    }
    generated = planner(request)
    if not isinstance(generated, dict) or set(generated) != {"plan", "source"}:
        raise WorkbenchError("planner must return exactly plan and source")
    if not isinstance(generated["plan"], dict) or not isinstance(generated["source"], str):
        raise WorkbenchError("planner returned invalid plan/source types")
    forbidden = {"schema_version", "plan_id", "plan_sha256", "sandbox", "intent", "sampling"}
    if forbidden & generated["plan"].keys():
        raise WorkbenchError("planner attempted to control a server-owned plan field")
    fields = dict(generated["plan"])
    fields.update(
        {
            "intent": normalized,
            "sampling": {
                "seed": seed,
                "repetitions": repetitions,
                "design": fields.pop("sampling_design", "paired_common_random_numbers"),
            },
            "sandbox": sandbox_declaration(generated["source"]),
        }
    )
    plan = build_experiment_plan(**fields)
    return {
        "state": "awaiting_confirmation",
        "request": request,
        "plan": plan,
        "source": generated["source"],
        "model": model,
    }


def confirm_and_predict(
    planned: Mapping[str, Any],
    *,
    confirmed: bool,
    model: FrozenModel,
    predictor: Predictor,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> PreparedExecution:
    """Record an explicit confirmation, then obtain a prediction before execution."""
    if planned.get("state") != "awaiting_confirmation":
        raise WorkbenchError("plan is not awaiting confirmation")
    if confirmed is not True:
        raise ConfirmationRequired("the user must confirm the exact sealed plan")
    model.validate()
    if planned.get("model") != model:
        raise WorkbenchError("predictor model must match the frozen planner model")
    plan = validate_experiment_plan(planned.get("plan", {}))
    source = planned.get("source")
    if not isinstance(source, str) or sandbox_declaration(source) != plan["sandbox"]:
        raise WorkbenchError("source does not match the confirmed plan")
    prediction = predictor(plan)
    _validate_prediction(prediction, plan)
    prediction_sha = hashlib.sha256(canonical_json(prediction)).hexdigest()
    recorded_at = _timestamp(now())
    return PreparedExecution(
        state="prediction_ready",
        plan=plan,
        source=source,
        prediction=prediction,
        prediction_recorded_at=recorded_at,
        confirmation={
            "confirmed": True,
            "plan_sha256": plan["plan_sha256"],
            "prediction_sha256": prediction_sha,
        },
        model=model,
    )


def finalize_run(
    prepared: PreparedExecution,
    modal_result: Mapping[str, Any],
    *,
    modal_call_id: str,
    code_version: str,
    started_at: datetime,
    completed_at: datetime,
    model: FrozenModel,
    reviewer: Reviewer,
) -> dict[str, Any]:
    """Compare measurements with the frozen prediction and seal one immutable run."""
    if prepared.state != "prediction_ready":
        raise ConfirmationRequired("execution requires a confirmed plan and recorded prediction")
    model.validate()
    if prepared.model != model:
        raise WorkbenchError("reviewer model must match the frozen planner and predictor model")
    if not isinstance(modal_result, Mapping) or set(modal_result) != {"measurements", "stdout"}:
        raise WorkbenchError("restricted Modal result has invalid fields")
    # Seal the same JSON-decoded numeric values that a remote validator will
    # receive before deriving comparisons or evidence from them.
    measurements = json.loads(canonical_json(list(modal_result["measurements"])))
    comparisons = comparison_records(prepared.plan, measurements)
    review_input = {
        "plan": prepared.plan,
        "prediction": prepared.prediction,
        "measurements": measurements,
        "comparisons": comparisons,
    }
    revision = reviewer(review_input)
    _validate_revision(revision)
    revision = {"plan_sha256": prepared.plan["plan_sha256"], **revision}
    run_result = {
        "schema_version": "jump.run-result/v1",
        "status": "completed",
        "metrics": [
            {"name": comparison["id"], "value": comparison["estimate"]}
            for comparison in comparisons
        ],
        "artifacts": [],
        "provenance": {
            "manifest_sha256": prepared.plan["plan_sha256"],
            "run_id": modal_call_id,
            "code_version": code_version,
            "source_sha256": prepared.plan["sandbox"]["source"]["sha256"],
            "policy_sha256": prepared.plan["sandbox"]["policy_sha256"],
        },
        "plan_sha256": prepared.plan["plan_sha256"],
        "measurements": measurements,
        "comparisons": comparisons,
    }
    payload = {
        "plan_sha256": prepared.plan["plan_sha256"],
        "prediction": prepared.prediction,
        "measurements": measurements,
        "comparisons": comparisons,
        "revision": revision,
    }
    sealed_result = seal_result_envelope(
        payload,
        source="live",
        manifest_sha256=prepared.plan["plan_sha256"],
        run_id=modal_call_id,
        code_version=code_version,
        checkpoint_id=prepared.plan["plan_id"],
    )
    prediction_sha = hashlib.sha256(canonical_json(prepared.prediction)).hexdigest()
    execution = {
        "plan_sha256": prepared.plan["plan_sha256"],
        "prediction": prepared.prediction,
        "prediction_sha256": prediction_sha,
        "prediction_recorded_at": prepared.prediction_recorded_at,
        "started_at": _timestamp(started_at),
        "completed_at": _timestamp(completed_at),
        "source_sha256": prepared.plan["sandbox"]["source"]["sha256"],
        "policy_sha256": prepared.plan["sandbox"]["policy_sha256"],
        "code_version": code_version,
        "modal_call_id": modal_call_id,
        "error": None,
    }
    evidence = {
        "plan_sha256": prepared.plan["plan_sha256"],
        "source_sha256": execution["source_sha256"],
        "policy_sha256": execution["policy_sha256"],
        "code_version": code_version,
        "modal_call_id": modal_call_id,
        "run_result_sha256": "0" * 64,
        "artifact_inventory_sha256": "0" * 64,
        "sealed_payload_sha256": "0" * 64,
    }
    try:
        return build_experiment_run(
            prepared.plan,
            verified_run_result=run_result,
            artifact_bytes={},
            sealed_result=sealed_result,
            plan_id=prepared.plan["plan_id"],
            plan_sha256=prepared.plan["plan_sha256"],
            status="completed",
            execution=execution,
            measurements=measurements,
            comparisons=comparisons,
            revision=revision,
            evidence=evidence,
        )
    except ExperimentContractError as exc:
        raise WorkbenchError(str(exc)) from exc


def _validate_prediction(value: Any, plan: Mapping[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"summary", "claims"}:
        raise WorkbenchError("predictor must return exactly summary and claims")
    if not isinstance(value["summary"], str) or not value["summary"]:
        raise WorkbenchError("prediction summary must be nonempty text")
    claims = value["claims"]
    if not isinstance(claims, list) or any(not isinstance(claim, dict) for claim in claims):
        raise WorkbenchError("prediction claims must be an array of objects")
    expected_ids = {item["id"] for item in plan["prediction_before_run"]["targets"]}
    actual_ids = {item.get("target_id") for item in claims}
    if actual_ids != expected_ids or len(claims) != len(expected_ids):
        raise WorkbenchError("prediction must cover each frozen target exactly once")
    for claim in claims:
        if set(claim) != {"target_id", "expected_relation", "rationale", "expected_value"}:
            raise WorkbenchError("prediction claim fields do not match the contract")
        if claim["expected_relation"] not in {"greater", "less", "equal", "different"}:
            raise WorkbenchError("prediction relation is invalid")
        if not isinstance(claim["rationale"], str) or not claim["rationale"]:
            raise WorkbenchError("prediction rationale must be nonempty text")
        expected_value = claim["expected_value"]
        if expected_value is not None and (not isinstance(expected_value, (int, float)) or isinstance(expected_value, bool)):
            raise WorkbenchError("prediction expected_value must be numeric or null")


def _validate_revision(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"disposition", "interpretation", "next_plan_sha256"}:
        raise WorkbenchError("reviewer must return disposition, interpretation, and next_plan_sha256")
    if value["disposition"] not in {"retain", "revise", "reject"}:
        raise WorkbenchError("revision disposition is invalid")
    if not isinstance(value["interpretation"], str) or not value["interpretation"]:
        raise WorkbenchError("revision interpretation must be nonempty text")
    next_sha = value["next_plan_sha256"]
    if next_sha is not None and (not isinstance(next_sha, str) or re.fullmatch(r"[0-9a-f]{64}", next_sha) is None):
        raise WorkbenchError("next_plan_sha256 must be null or a lowercase SHA-256")


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise WorkbenchError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "QUESTION_VERSION", "MAX_INTENT_CHARS", "WorkbenchError", "ConfirmationRequired", "FrozenModel",
    "PreparedExecution", "validate_user_intent", "prepare_plan", "confirm_and_predict", "finalize_run",
]
