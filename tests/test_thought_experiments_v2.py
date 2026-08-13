from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from jump_contracts.thought_experiments import (
    CONFIRMATION_VERSION,
    EXPERIMENT_SPEC_SCHEMA_SHA256,
    THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256,
    ThoughtExperimentContractError,
    build_experiment_spec,
    validate_thought_experiment_run,
)
from jump_workbench.gemma_planner import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION
from jump_workbench.visual_coordinator import VisualCoordinator, VisualCoordinatorError
from jump_workbench.visual_engine import execute_visual_spec
from jump_workbench.workflow import FrozenModel


def particle_fields():
    return {
        "question": "How does reversing attraction change a particle system?",
        "hypothesis": "Reversing attraction to repulsion increases mean squared distance from the center.",
        "world": {
            "bounds": {"width": 100.0, "height": 100.0, "boundary": "reflect"},
            "entities": [{
                "id": "particle", "label": "Particle", "count": 12,
                "appearance": {"shape": "circle", "color": "#4f46e5", "size": 2.0},
                "initial_state": {"numeric": {"vx": 0.0, "vy": 0.0}, "categorical": {"kind": "matter"}},
                "initial_layout": {"kind": "ring", "center": [50.0, 50.0], "spread": 15.0},
            }],
            "graph": {"kind": "none", "edge_probability": 0.0, "directed": False},
        },
        "dynamics": {"rules": [
            {"id": "force", "op": "pairwise_force_2d", "target_type": "particle", "parameters": {"strength": 8.0, "exponent": 2.0, "softening": 2.0}},
            {"id": "motion", "op": "move_2d", "target_type": "particle", "parameters": {"damping": 0.98, "max_speed": 5.0}},
        ]},
        "conditions": [
            {"id": "baseline", "label": "Attraction", "kind": "baseline", "interventions": []},
            {"id": "counterfactual", "label": "Repulsion after step 5", "kind": "counterfactual", "interventions": [{"time": 5, "operation": "scale_rule_parameter", "target": "force", "field": "strength", "value": -1.0}]},
        ],
        "schedule": {"duration_steps": 20, "dt": 0.2, "seed": 7613, "repetitions": 2},
        "measurements": [{"id": "mean_speed", "label": "Mean x velocity", "op": "mean_state", "entity_type": "particle", "state": "vx", "category": None}],
        "visualization": {"kind": "animated_2d", "frame_stride": 4, "max_frames": 6, "chart_measurement_ids": ["mean_speed"]},
    }


def built_spec():
    return build_experiment_spec(intent="Reverse the force after step five.", **particle_fields())


def test_contract_hashes_are_frozen_and_engine_is_not_template_or_code_driven():
    assert EXPERIMENT_SPEC_SCHEMA_SHA256 == "fa7674dc3c5f759dc74ff723cef7a194edc4186069496e631e65b4d0ebd84ab5"
    assert THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256 == "55d1fd3fdef215abfb1a148080cc01aea3fff118ba1e779e02e6841f43941166"
    spec = built_spec()
    assert "template_id" not in str(spec)
    bad = deepcopy(spec)
    bad["dynamics"]["rules"][0]["parameters"]["source"] = "import os"
    with pytest.raises(ThoughtExperimentContractError):
        # hash mismatch and closed parameter allowlist both fail closed
        from jump_contracts.thought_experiments import validate_experiment_spec
        validate_experiment_spec(bad)


def test_same_visual_dsl_executes_particle_and_graph_contagion_families():
    particle = built_spec()
    particle_result = execute_visual_spec(particle)
    assert {item["condition_id"] for item in particle_result["conditions"]} == {"baseline", "counterfactual"}
    assert all(len(item["frames"]) == 6 for item in particle_result["conditions"])

    fields = particle_fields()
    fields["question"] = "Does lowering transmission slow contagion on a ring?"
    fields["hypothesis"] = "Lower transmission produces fewer infected agents."
    fields["world"]["graph"] = {"kind": "ring", "edge_probability": 0.0, "directed": False}
    fields["world"]["entities"][0]["initial_state"] = {"numeric": {}, "categorical": {"health": "susceptible"}}
    fields["dynamics"] = {"rules": [{"id": "spread", "op": "graph_contagion", "target_type": None, "parameters": {
        "state": "health", "susceptible": "susceptible", "infected": "infected", "recovered": "recovered",
        "transmission_probability": 0.5, "recovery_probability": 0.1,
    }}]}
    fields["conditions"][1]["interventions"] = [{"time": 1, "operation": "set_rule_parameter", "target": "spread", "field": "transmission_probability", "value": 0.1}]
    fields["measurements"] = [{"id": "infected_count", "label": "Infected", "op": "count_category", "entity_type": "particle", "state": "health", "category": "infected"}]
    fields["visualization"]["chart_measurement_ids"] = ["infected_count"]
    contagion = build_experiment_spec(intent="Lower transmission in a graph contagion.", **fields)
    result = execute_visual_spec(contagion)
    assert len(result["comparisons"]) == 1


def test_visual_coordinator_requires_confirmation_and_records_prediction_first():
    events = []
    state = {}
    fields = particle_fields()

    def generate(action, _payload):
        events.append(action)
        if action == "visual_spec":
            return fields
        if action == "visual_predict":
            return {"summary": "The intervention should change mean speed.", "expected_direction": "change", "measurement_id": "mean_speed"}
        return {"disposition": "retain", "interpretation": "The simulated comparison is consistent with the prediction."}

    def simulate(spec, _prediction, recorded):
        events.append("simulate")
        started = datetime.fromisoformat(recorded.replace("Z", "+00:00")) + timedelta(milliseconds=1)
        return {"modal_call_id": "fc-visual-test", "started_at": started, "completed_at": started + timedelta(seconds=1), "result": execute_visual_spec(spec)}

    coordinator = VisualCoordinator(
        state=state,
        model=FrozenModel(model_id=BASE_REPO_ID, revision=BASE_REVISION),
        transformers_revision=TRANSFORMERS_REVISION,
        model_generate=generate,
        simulate=simulate,
        code_version="c" * 40,
    )
    request = {"schema_version": "jump.thought-experiment-question/v2", "request_id": "req-v2", "session_id": "session-v2", "intent": "Reverse attraction after five steps.", "seed": 7613, "repetitions": 2}
    planned = coordinator.compile(request)
    assert planned["status"] == "awaiting_confirmation"
    with pytest.raises(VisualCoordinatorError):
        coordinator.confirm({**planned["confirmation"], "confirmed": False})
    completed = coordinator.confirm({**planned["confirmation"], "confirmed": True})
    run = completed["run"]
    assert events == ["visual_spec", "visual_predict", "simulate", "visual_review"]
    assert run["execution"]["prediction_recorded_at"] < run["execution"]["started_at"]
    assert validate_thought_experiment_run(run, completed["spec"]) == run
    assert "image" not in str(run).lower() and "learned" not in str(run).lower()
