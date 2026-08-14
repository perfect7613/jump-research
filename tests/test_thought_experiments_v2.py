from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from jump_contracts.thought_experiments import (
    CONFIRMATION_VERSION,
    EXPERIMENT_SPEC_SCHEMA_SHA256,
    THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256,
    ThoughtExperimentContractError,
    build_experiment_spec,
    build_thought_experiment_run,
    validate_thought_experiment_run,
)
from jump_workbench.gemma_planner import (
    BASE_REPO_ID,
    BASE_REVISION,
    TRANSFORMERS_REVISION,
    _complete_json_object_end,
    _extract_json_object,
    _generate_visual_prediction,
    _normalize_visual_prediction,
    _validate_visual_review,
)
from jump_workbench.visual_coordinator import VisualCoordinator, VisualCoordinatorError
from jump_workbench.visual_engine import _apply_boundary, _graph_contagion, execute_visual_spec
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
    fields["world"]["entities"][0]["count"] = 11
    fields["world"]["entities"][0]["initial_state"] = {"numeric": {}, "categorical": {"health": "susceptible"}}
    fields["world"]["entities"].append({
        "id": "seed", "label": "Initially infected agent", "count": 1,
        "appearance": {"shape": "square", "color": "#dc2626", "size": 2.0},
        "initial_state": {"numeric": {}, "categorical": {"health": "infected"}},
        "initial_layout": {"kind": "line", "center": [50.0, 50.0], "spread": 0.0},
    })
    fields["dynamics"] = {"rules": [{"id": "spread", "op": "graph_contagion", "target_type": None, "parameters": {
        "state": "health", "susceptible": "susceptible", "infected": "infected", "recovered": "recovered",
        "transmission_probability": 1.0, "recovery_probability": 0.0,
    }}]}
    fields["conditions"][1]["interventions"] = [{"time": 0, "operation": "set_rule_parameter", "target": "spread", "field": "transmission_probability", "value": 0.0}]
    fields["measurements"] = [{"id": "infected_count", "label": "Infected", "op": "count_category", "entity_type": "particle", "state": "health", "category": "infected"}]
    fields["visualization"]["chart_measurement_ids"] = ["infected_count"]
    contagion = build_experiment_spec(intent="Lower transmission in a graph contagion.", **fields)
    result = execute_visual_spec(contagion)
    assert len(result["comparisons"]) == 1
    comparison = result["comparisons"][0]
    assert comparison["baseline_final"] > 0.0
    assert comparison["baseline_final"] > comparison["counterfactual_final"]


@pytest.mark.parametrize("operation,value", [
    ("set_rule_parameter", "not-a-number"),
    ("set_rule_parameter", 1001.0),
    ("scale_rule_parameter", 1001.0),
])
def test_rule_interventions_preserve_parameter_types_and_bounds(operation, value):
    fields = particle_fields()
    fields["conditions"][1]["interventions"] = [{
        "time": 5,
        "operation": operation,
        "target": "force",
        "field": "strength",
        "value": value,
    }]
    with pytest.raises(ThoughtExperimentContractError):
        build_experiment_spec(intent="Use an invalid force intervention.", **fields)


def test_visual_coordinator_requires_confirmation_and_records_prediction_first():
    events = []
    state = {}
    fields = particle_fields()
    replay = {}

    def generate(action, _payload):
        events.append(action)
        if action == "visual_spec":
            return fields
        if action == "visual_predict":
            if replay:
                with pytest.raises(VisualCoordinatorError, match="no longer awaiting"):
                    replay["coordinator"].confirm(replay["confirmation"])
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
    replay.update({
        "coordinator": coordinator,
        "confirmation": {**planned["confirmation"], "confirmed": True},
    })
    assert planned["status"] == "awaiting_confirmation"
    with pytest.raises(VisualCoordinatorError):
        coordinator.confirm({**planned["confirmation"], "confirmed": False})
    with pytest.raises(VisualCoordinatorError, match="originating request and session"):
        coordinator.confirm({**planned["confirmation"], "session_id": "other-session", "confirmed": True})
    completed = coordinator.confirm({**planned["confirmation"], "confirmed": True})
    run = completed["run"]
    assert events == ["visual_spec", "visual_predict", "simulate", "visual_review"]
    assert run["execution"]["prediction_recorded_at"] < run["execution"]["started_at"]
    assert validate_thought_experiment_run(run, completed["spec"]) == run
    assert "image" not in str(run).lower() and "learned" not in str(run).lower()
    with pytest.raises(VisualCoordinatorError, match="no longer awaiting"):
        coordinator.confirm({**planned["confirmation"], "confirmed": True})
    assert state == {}


def test_malformed_or_unsupported_model_spec_fails_closed_and_stale_state_is_pruned():
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    state = {"legacy": {"state": "awaiting_confirmation"}}
    outputs = [
        {**particle_fields(), "world": []},
        {"unsupported": "operation is outside the visual DSL"},
        particle_fields(),
    ]

    coordinator = VisualCoordinator(
        state=state,
        model=FrozenModel(model_id=BASE_REPO_ID, revision=BASE_REVISION),
        transformers_revision=TRANSFORMERS_REVISION,
        model_generate=lambda _action, _payload: outputs.pop(0),
        simulate=lambda *_args: {},
        code_version="c" * 40,
        now=lambda: now,
    )
    request = {
        "schema_version": "jump.thought-experiment-question/v2",
        "request_id": "req-fail-closed",
        "session_id": "session-fail-closed",
        "intent": "Compile a bounded toy experiment.",
        "seed": 7613,
        "repetitions": 1,
    }
    with pytest.raises(VisualCoordinatorError, match="compiler output rejected"):
        coordinator.compile(request)
    with pytest.raises(VisualCoordinatorError, match="unsupported thought experiment"):
        coordinator.compile(request)
    planned = coordinator.compile(request)
    assert "legacy" not in state
    assert list(state) == [planned["confirmation"]["confirmation_token"]]


def test_run_builder_reports_contract_error_for_missing_fields_and_parses_timezones():
    spec = built_spec()
    with pytest.raises(ThoughtExperimentContractError, match="missing required fields"):
        build_thought_experiment_run(
            spec,
            spec_id=spec["spec_id"],
            spec_sha256=spec["spec_sha256"],
        )

    result = execute_visual_spec(spec)
    base = {
        "spec_id": spec["spec_id"],
        "spec_sha256": spec["spec_sha256"],
        "status": "completed",
        "execution": {
            "engine_id": "jump.declarative-visual-engine/v2",
            "code_version": "c" * 40,
            "modal_call_id": "fc-timezone-test",
            "prediction": {
                "summary": "The intervention changes mean speed.",
                "expected_direction": "change",
                "measurement_id": "mean_speed",
            },
            "prediction_recorded_at": "2026-08-14T10:00:00+02:00",
            "started_at": "2026-08-14T09:00:00+00:00",
            "completed_at": "2026-08-14T09:01:00+00:00",
            "error": None,
        },
        "conditions": result["conditions"],
        "comparisons": result["comparisons"],
        "revision": {"disposition": "retain", "interpretation": "Bounded toy result."},
        "evidence": {
            "spec_sha256": spec["spec_sha256"],
            "engine_id": "jump.declarative-visual-engine/v2",
            "code_version": "c" * 40,
            "modal_call_id": "fc-timezone-test",
            "result_sha256": "0" * 64,
            "sealed_payload_sha256": "0" * 64,
        },
    }
    assert build_thought_experiment_run(spec, **base)["status"] == "completed"
    base["execution"]["prediction_recorded_at"] = "2026-08-14T08:30:00-01:00"
    with pytest.raises(ThoughtExperimentContractError, match="prediction must be recorded"):
        build_thought_experiment_run(spec, **base)


def test_dead_graph_entities_do_not_interact_and_reflection_does_not_declare_velocity():
    entities = [
        {"alive": False, "x": -5.0, "y": -5.0, "numeric": {}, "categorical": {"health": "infected"}},
        {"alive": True, "x": -1.0, "y": 12.0, "numeric": {}, "categorical": {"health": "susceptible"}},
    ]
    rule = {"parameters": {
        "state": "health",
        "susceptible": "susceptible",
        "infected": "infected",
        "recovered": "recovered",
        "transmission_probability": 1.0,
        "recovery_probability": 0.0,
    }}
    import random

    _graph_contagion(rule, entities, [(0, 1)], None, random.Random(7613))
    assert entities[1]["categorical"]["health"] == "susceptible"
    _apply_boundary(entities, {"width": 10.0, "height": 10.0, "boundary": "reflect"})
    assert (entities[0]["x"], entities[0]["y"]) == (-5.0, -5.0)
    assert (entities[1]["x"], entities[1]["y"]) == (0.0, 10.0)
    assert entities[1]["numeric"] == {}


def test_visual_result_decompression_is_bounded_during_expansion():
    import zlib

    from jump_workbench.modal_app import _open_visual_result

    assert _open_visual_result(zlib.compress(b"{}")) == {}
    with pytest.raises(ValueError, match="canonical JSON cap"):
        _open_visual_result(zlib.compress(b"x" * 1_000_001))


def test_visual_model_narratives_reject_overlong_output_instead_of_truncating():
    with pytest.raises(ValueError, match="1 through 500"):
        _normalize_visual_prediction({
            "summary": "x" * 501,
            "expected_direction": "change",
            "measurement_id": "mean_speed",
        })
    with pytest.raises(ValueError, match="1 through 500"):
        _validate_visual_review({
            "disposition": "retain",
            "interpretation": "x" * 501,
        })


def test_visual_prediction_rejects_duplicate_or_trailing_json_and_repairs_once(monkeypatch):
    valid = {
        "summary": "Reversing the force should change mean speed.",
        "expected_direction": "change",
        "measurement_id": "mean_speed",
    }
    encoded = json.dumps(valid)
    with pytest.raises(ValueError, match="exactly one JSON object"):
        _extract_json_object(f"{encoded} thought {encoded}", strict_single_object=True)
    with pytest.raises(ValueError, match="text outside"):
        _extract_json_object(f"{encoded} thought", strict_single_object=True)
    assert _complete_json_object_end(encoded + " thought " + encoded) == len(encoded)
    quoted_brace = '{"summary":"a } inside a string","nested":{"value":1}} trailing'
    assert _complete_json_object_end(quoted_brace) == quoted_brace.index(" trailing")
    assert _complete_json_object_end('{"summary":"unterminated"') is None

    from jump_workbench import gemma_planner

    prompts = []
    responses = iter([ValueError("duplicate model output"), valid])

    def fake_generate(
        _runtime,
        prompt,
        *,
        max_new_tokens,
        deterministic=False,
        strict_single_object=False,
    ):
        prompts.append((prompt, max_new_tokens, deterministic, strict_single_object))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(gemma_planner, "_generate_json", fake_generate)
    prediction = _generate_visual_prediction({}, {"spec": built_spec()})
    assert prediction == valid
    assert len(prompts) == 2
    assert prompts[0][1:] == (320, True, True)
    assert prompts[1][1:] == (240, True, True)
    assert "mean_speed" in prompts[1][0]
    assert "duplicate model output" not in prompts[1][0]

    responses = iter([
        ValueError("duplicate model output"),
        {**valid, "measurement_id": "undeclared_measurement"},
    ])
    with pytest.raises(ValueError, match="not declared"):
        _generate_visual_prediction({}, {"spec": built_spec()})
