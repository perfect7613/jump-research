from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jump_workbench.coordinator import CoordinatorError, GeneralCoordinator
from jump_workbench.gemma_planner import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION
from jump_workbench.workflow import FrozenModel


SOURCE = """\
import random

def simulate(plan):
    rows = []
    for condition in plan["conditions"]:
        for repetition in range(plan["sampling"]["repetitions"]):
            rows.append({"condition_id": condition["id"], "repetition": repetition,
                         "pairing_key": "rep-" + str(repetition),
                         "values": {"win_rate": float(condition["assignments"]["switch"])}})
    return {"measurements": rows}
"""

PLAN_FIELDS = {
    "hypothesis": "Switching doors increases the win rate in this toy Monty Hall simulation.",
    "variables": {
        "independent": [{"id": "switch", "label": "Switch doors", "levels": [0, 1]}],
        "dependent": [{"id": "win_rate", "label": "Win rate", "unit": None}],
        "controlled": [{"id": "doors", "label": "Door count", "value": 3}],
    },
    "assumptions": ["The prize door and initial choice are uniform."],
    "conditions": [
        {"id": "stay", "label": "Stay", "kind": "baseline", "assignments": {"switch": 0}},
        {"id": "switch", "label": "Switch", "kind": "intervention", "assignments": {"switch": 1}},
    ],
    "sampling_design": "paired_common_random_numbers",
    "prediction_before_run": {
        "required": True,
        "targets": [{"id": "switch_effect", "measurement_id": "win_rate", "baseline_condition_id": "stay", "intervention_condition_id": "switch"}],
    },
    "measurements": [{"id": "win_rate", "label": "Win rate", "unit": None, "aggregation": "mean", "display": "bar"}],
    "comparisons": [{"id": "switch_effect", "measurement_id": "win_rate", "baseline_condition_id": "stay", "intervention_condition_id": "switch", "statistic": "mean_difference", "pairing": "paired_by_repetition"}],
}


def _question(**extra):
    return {
        "schema_version": "jump.experiment-question/v1",
        "request_id": "request-1",
        "session_id": "session-1",
        "intent": "In a toy Monty Hall simulation, does switching improve win rate?",
        "seed": 7613,
        "repetitions": 2,
        **extra,
    }


def _coordinator(events):
    def model_generate(action, _payload):
        events.append(action)
        if action == "plan":
            return {"plan": PLAN_FIELDS, "source": SOURCE}
        if action == "predict":
            return {"summary": "Switching should win more often.", "claims": [{"target_id": "switch_effect", "expected_relation": "greater", "rationale": "The host removes a losing door.", "expected_value": 1 / 3}]}
        return {"disposition": "retain", "interpretation": "The simulated switch condition had the higher measured win rate.", "next_plan_sha256": None}

    def simulate(plan, _source, _confirmation, _prediction):
        events.append("simulate")
        started = datetime.now(timezone.utc) + timedelta(seconds=1)
        return {
            "modal_call_id": "fc-coordinator-test",
            "started_at": started,
            "completed_at": started + timedelta(seconds=1),
            "result": {
                "stdout": "",
                "measurements": [
                    {"plan_sha256": plan["plan_sha256"], "condition_id": "stay", "repetition": 0, "pairing_key": "rep-0", "values": {"win_rate": 0.0}},
                    {"plan_sha256": plan["plan_sha256"], "condition_id": "switch", "repetition": 0, "pairing_key": "rep-0", "values": {"win_rate": 1.0}},
                    {"plan_sha256": plan["plan_sha256"], "condition_id": "stay", "repetition": 1, "pairing_key": "rep-1", "values": {"win_rate": 0.0}},
                    {"plan_sha256": plan["plan_sha256"], "condition_id": "switch", "repetition": 1, "pairing_key": "rep-1", "values": {"win_rate": 1.0}},
                ],
            },
        }

    return GeneralCoordinator(
        state={},
        model=FrozenModel(model_id=BASE_REPO_ID, revision=BASE_REVISION),
        transformers_revision=TRANSFORMERS_REVISION,
        model_generate=model_generate,
        simulate=simulate,
        code_version="c" * 40,
    )


def test_question_ingress_rejects_source_before_model_call():
    events = []
    coordinator = _coordinator(events)
    with pytest.raises(CoordinatorError, match="exactly"):
        coordinator.plan(_question(source="def simulate(plan): pass"))
    assert events == []


def test_confirmation_records_prediction_before_restricted_execution():
    events = []
    coordinator = _coordinator(events)
    plan_response = coordinator.plan(_question())
    assert set(plan_response) == {
        "schema_version", "status", "request_id", "session_id", "plan", "model", "confirmation"
    }
    assert SOURCE not in str(plan_response)
    confirmation = {**plan_response["confirmation"], "confirmed": True}
    run_response = coordinator.confirm(confirmation)
    run = run_response["run"]
    assert events == ["plan", "predict", "simulate", "review"]
    assert run["execution"]["prediction_recorded_at"] < run["execution"]["started_at"]
    assert run["execution"]["modal_call_id"] == "fc-coordinator-test"
    assert run_response["model"]["adapter_id"] is None


def test_confirmation_must_match_the_human_visible_plan():
    coordinator = _coordinator([])
    response = coordinator.plan(_question())
    confirmation = {**response["confirmation"], "confirmed": True, "plan_sha256": "0" * 64}
    with pytest.raises(CoordinatorError, match="does not match"):
        coordinator.confirm(confirmation)
