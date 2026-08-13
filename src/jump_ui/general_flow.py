"""Non-live general workbench fixture using the corrected direct workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from jump_contracts.experiments import (
    DEPENDENCY_LOCK_SHA256,
    EXPERIMENT_PLAN_SCHEMA_SHA256,
    EXPERIMENT_RUN_SCHEMA_SHA256,
    RESTRICTED_POLICY_SHA256,
    validate_experiment_plan,
)
from .general_client import validate_run_response
from jump_workbench.workflow import (
    ConfirmationRequired,
    FrozenModel,
    PreparedExecution,
    WorkbenchError,
    confirm_and_predict,
    finalize_run,
    prepare_plan,
)

PLAN_SCHEMA_SHA256 = "4cadd0c9859add72c5c71c4cca0e71e3f0a7b5deb398948ae205a238035ffc08"
RUN_SCHEMA_SHA256 = "5e6be146224d6ae7661be9f911a0bce47df4bb5d145ee83967bc446c7ffc97ca"
POLICY_SHA256 = "3b11f25c4df2e8db5554943f64b1df7d6e26ecdd40bb85d41c6c9a47b09bcda0"
LOCK_SHA256 = "7de15d556cc846ab7127ffeb130bafe13278ef8b771ec5a7b830d813506fcf36"
FIXTURE_LABEL = "NON-LIVE GENERAL WORKBENCH FIXTURE"

SOURCE = '''def simulate(plan):
    rows = []
    for condition in plan["conditions"]:
        for repetition in range(plan["sampling"]["repetitions"]):
            rows.append({"condition_id": condition["id"], "repetition": repetition,
                         "pairing_key": "rep-" + str(repetition),
                         "values": {"outcome": condition["assignments"]["change"]}})
    return {"measurements": rows}
'''


class GeneralUIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Template:
    hypothesis: str
    change: str
    baseline_label: str
    intervention_label: str
    measure: str
    unit: str
    assumption: str
    relation: str
    prediction: str
    baseline: float
    intervention: float


TEMPLATES = {
    "traffic": Template(
        "Increasing service capacity will reduce average queue length.", "Queue service capacity",
        "Capacity 4", "Capacity 6", "Average queue length", "agents",
        "Both conditions use the same arrival pattern.", "less",
        "Higher service capacity should produce a shorter average queue.", 18.4, 12.1,
    ),
    "monty": Template(
        "Switching doors will win more often than staying.", "Door strategy", "Stay", "Switch",
        "Win rate", "proportion", "The host always reveals a losing door and offers a switch.",
        "greater", "Switching should win about twice as often as staying.", 0.34, 0.67,
    ),
    "bernoulli": Template(
        "Raising the success probability will increase the observed success rate.", "Success probability",
        "Probability 0.5", "Probability 0.7", "Observed success rate", "proportion",
        "Each draw is an independent Bernoulli trial.", "greater",
        "The higher probability should produce a higher observed success rate.", 0.5, 0.7,
    ),
}
MODEL = FrozenModel(model_id="google/gemma-fixture", revision="a" * 40)


def _template(intent: str) -> Template:
    text = intent.casefold()
    if "monty" in text or "door" in text:
        return TEMPLATES["monty"]
    if "bernoulli" in text or "probability" in text or "success rate" in text:
        return TEMPLATES["bernoulli"]
    if "traffic" in text or "queue" in text or "capacity" in text:
        return TEMPLATES["traffic"]
    raise GeneralUIError("The non-live fixture supports the three examples only. No result was substituted.")


def _planner(request: dict[str, Any]) -> dict[str, Any]:
    item = _template(request["intent"])
    return {
        "source": SOURCE,
        "plan": {
            "hypothesis": item.hypothesis,
            "variables": {
                "independent": [{"id": "change", "label": item.change, "levels": [0, 1]}],
                "dependent": [{"id": "outcome", "label": item.measure, "unit": item.unit}],
                "controlled": [],
            },
            "assumptions": [item.assumption],
            "conditions": [
                {"id": "baseline", "label": item.baseline_label, "kind": "baseline", "assignments": {"change": 0}},
                {"id": "intervention", "label": item.intervention_label, "kind": "intervention", "assignments": {"change": 1}},
            ],
            "sampling_design": "paired_common_random_numbers",
            "prediction_before_run": {"required": True, "targets": [{
                "id": "primary_comparison", "measurement_id": "outcome",
                "baseline_condition_id": "baseline", "intervention_condition_id": "intervention",
            }]},
            "measurements": [{"id": "outcome", "label": item.measure, "unit": item.unit, "aggregation": "mean", "display": "bar"}],
            "comparisons": [{
                "id": "primary_comparison", "measurement_id": "outcome",
                "baseline_condition_id": "baseline", "intervention_condition_id": "intervention",
                "statistic": "mean_difference", "pairing": "paired_by_repetition",
            }],
        },
    }


def prepare_fixture(intent: str, *, request_id: str, repetitions: int = 8) -> dict[str, Any]:
    try:
        return prepare_plan(
            intent, session_id="local-session", request_id=request_id, seed=7613,
            repetitions=repetitions, model=MODEL, planner=_planner,
        )
    except (WorkbenchError, ValueError) as exc:
        raise GeneralUIError(str(exc)) from exc


def confirm_fixture(planned: Mapping[str, Any], *, confirmed: bool) -> PreparedExecution:
    def predict(plan: dict[str, Any]) -> dict[str, Any]:
        item = _template(plan["intent"])
        return {"summary": item.prediction, "claims": [{
            "target_id": "primary_comparison", "expected_relation": item.relation,
            "rationale": item.hypothesis, "expected_value": None,
        }]}

    try:
        return confirm_and_predict(
            planned, confirmed=confirmed, model=MODEL, predictor=predict,
            now=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
        )
    except (ConfirmationRequired, WorkbenchError, ValueError) as exc:
        raise GeneralUIError(str(exc)) from exc


def execute_fixture(prepared: PreparedExecution) -> dict[str, Any]:
    item = _template(prepared.plan["intent"])
    rows = []
    for repetition in range(prepared.plan["sampling"]["repetitions"]):
        key = f"rep-{repetition}"
        rows += [
            {"plan_sha256": prepared.plan["plan_sha256"], "condition_id": "baseline", "repetition": repetition, "pairing_key": key, "values": {"outcome": item.baseline}},
            {"plan_sha256": prepared.plan["plan_sha256"], "condition_id": "intervention", "repetition": repetition, "pairing_key": key, "values": {"outcome": item.intervention}},
        ]
    try:
        return finalize_run(
            prepared, {"measurements": rows, "stdout": ""}, modal_call_id="fixture-call",
            code_version="general-fixture-v1",
            started_at=datetime(2026, 8, 13, 12, 1, tzinfo=timezone.utc),
            completed_at=datetime(2026, 8, 13, 12, 2, tzinfo=timezone.utc), model=MODEL,
            reviewer=lambda _value: {
                "disposition": "retain",
                "interpretation": "The simulated direction matched the recorded prediction in this non-live fixture.",
                "next_plan_sha256": None,
            },
        )
    except (WorkbenchError, ValueError) as exc:
        raise GeneralUIError(str(exc)) from exc


def plan_rows(plan: Mapping[str, Any]) -> dict[str, str]:
    checked = validate_experiment_plan(plan)
    variable, measure = checked["variables"]["independent"][0], checked["measurements"][0]
    return {
        "Question": checked["intent"],
        "Hypothesis": checked["hypothesis"],
        "Change": f"{variable['label']}: {checked['conditions'][0]['label']} → {checked['conditions'][1]['label']}",
        "Measure": f"{measure['label']} ({measure['unit']})",
        "Assumptions": "; ".join(checked["assumptions"]),
        "Repetitions": str(checked["sampling"]["repetitions"]),
    }


def result_rows(run: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, str]:
    checked = validate_experiment_plan(plan)
    checked_run = validate_run_response(run, checked)
    if checked_run["plan_id"] != checked["plan_id"] or checked_run["plan_sha256"] != checked["plan_sha256"]:
        raise GeneralUIError("run does not bind the confirmed plan")
    relation = checked_run["execution"]["prediction"]["claims"][0]["expected_relation"]
    estimate = checked_run["comparisons"][0]["estimate"]
    matched = (relation == "less" and estimate < 0) or (relation == "greater" and estimate > 0)
    prediction = checked_run["execution"]["prediction"]
    claim_text = "; ".join(
        f"{claim['target_id']}: expected {claim['expected_relation']} — {claim['rationale']}"
        + (f" (expected value {claim['expected_value']:.3g})" if claim["expected_value"] is not None else "")
        for claim in prediction["claims"]
    )
    conditions = {condition["id"]: condition["label"] for condition in checked["conditions"]}
    measurement = checked["measurements"][0]
    measured_averages = []
    for condition_id, label in conditions.items():
        values = [
            row["values"][measurement["id"]]
            for row in checked_run["measurements"]
            if row["condition_id"] == condition_id
        ]
        measured_averages.append(f"{label}: {sum(values) / len(values):.3g}")
    return {
        "Prediction": f"{prediction['summary']} {claim_text}",
        "Simulation": (
            f"Measured {measurement['label'].lower()} averages — {'; '.join(measured_averages)}. "
            f"The intervention-minus-baseline comparison was {estimate:.3g}."
        ),
        "Was the prediction right?": "Yes—the simulated direction matched the prediction." if matched else "No—the simulated direction differed from the prediction.",
        "What the model changed its mind about": (
            f"Disposition: {checked_run['revision']['disposition']}. "
            f"{checked_run['revision']['interpretation']}"
        ),
        "Evidence": f"Plan {checked_run['plan_id']} and run {checked_run['run_id']} passed the corrected evidence contracts.",
    }


assert EXPERIMENT_PLAN_SCHEMA_SHA256 == PLAN_SCHEMA_SHA256
assert EXPERIMENT_RUN_SCHEMA_SHA256 == RUN_SCHEMA_SHA256
assert RESTRICTED_POLICY_SHA256 == POLICY_SHA256
assert DEPENDENCY_LOCK_SHA256 == LOCK_SHA256
