import copy
import json

import pytest

from jump_benchmark.experiment_spec import (
    EXPERIMENT_SPEC_CONTRACT_SHA256,
    INTENT_SCHEMA_VERSION,
    build_planned_run,
    compile_experiment_intent,
    experiment_spec_contract,
    materialize_experiment,
    validate_experiment_spec,
)


def request(intent: str, *, seed=7, max_steps=4):
    return {
        "schema_version": INTENT_SCHEMA_VERSION,
        "intent": intent,
        "session_id": "space-session",
        "seed": seed,
        "max_steps": max_steps,
    }


def test_approved_examples_compile_to_wireframe_rows_without_raw_intent():
    expected = ["future-prediction", "hidden-law-discovery", "falsified-prior", "world-swap"]
    plans = [compile_experiment_intent(request(intent)) for intent in experiment_spec_contract()["examples"]]
    assert [plan["template_id"] for plan in plans] == expected
    for intent, plan in zip(experiment_spec_contract()["examples"], plans):
        assert plan["world_count"] == 6
        assert intent not in json.dumps(plan)
        compiled = materialize_experiment(plan)
        assert 1 <= len(compiled["worlds"]) <= 2
    assert set(experiment_spec_contract()["plan"]["confirmation_rows"]) == {"World", "Observe", "Change", "Predict"}


@pytest.mark.parametrize(
    "intent",
    [
        "open https://example.com and simulate",
        "read ./weights/model.bin",
        "import os; predict the future",
        "predict\u0000motion",
        "swap the world and falsify the prior",
    ],
)
def test_unsafe_or_ambiguous_intent_fails_closed(intent):
    with pytest.raises(ValueError):
        compile_experiment_intent(request(intent))


def test_plan_is_deterministic_tamper_evident_and_requires_four_frames():
    value = request("Predict the future trajectory.", seed=None, max_steps=4)
    first = compile_experiment_intent(value)
    assert first == compile_experiment_intent(value)
    changed = copy.deepcopy(first)
    changed["prediction_horizon"] = 2
    with pytest.raises(ValueError):
        validate_experiment_spec(changed)
    with pytest.raises(ValueError, match="at least four"):
        compile_experiment_intent(request("Predict motion", max_steps=3))


def test_planned_run_is_live_but_does_not_claim_execution():
    plan = compile_experiment_intent(request("Swap the learned latent between matched worlds."))
    run = build_planned_run(request_id="req-1", plan=plan)
    assert run == {
        "schema_version": "jump.experiment-run/v1",
        "status": "planned",
        "live": True,
        "request_id": "req-1",
        "plan": plan,
        "result": None,
        "error": None,
    }
    assert len(EXPERIMENT_SPEC_CONTRACT_SHA256) == 64
