import copy
import base64
import hashlib
import json
import struct

import pytest

from jump_contracts import build_learned_latent_evidence, learned_decoder_identity, seal_learned_latent_result
from jump_benchmark.canonical import sha256_json

from jump_benchmark.experiment_spec import (
    EXPERIMENT_SPEC_CONTRACT_SHA256,
    INTENT_SCHEMA_VERSION,
    build_planned_run,
    compile_experiment_intent,
    experiment_spec_contract,
    materialize_experiment,
    validate_experiment_run,
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
        assert plan["object_count"] == 6
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
    assert EXPERIMENT_SPEC_CONTRACT_SHA256 == "6c35ea47f3c3ad614cfb053c37b1670764a9584bbe7e2706b793f5d8f5635ad6"


def _completed_run(plan):
    raw = struct.pack("<4f", 0.25, -1.5, 2.0, 0.0)
    svg = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    answer = {
        "prediction": "bounded",
        "producer_bindings": {
            "experiment_id": plan["experiment_id"],
            "experiment_spec_sha256": sha256_json(plan),
        },
    }
    evidence = build_learned_latent_evidence(
        encoder_output=raw,
        decoder_input=raw,
        injection_input=raw,
        encoder_observation=b"observation",
        encoder_observation_artifact_name="observation.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[4],
        order="C",
        tensor_artifact_name="world-latent.f32le.bin",
        recipient_world_id="world-a",
        world_pair_id="pair-a",
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder.safetensors",
            artifact_sha256="d" * 64,
            training_manifest_sha256="e" * 64,
            code_version="0" * 40,
            architecture="test-decoder",
        ),
        decoded_image=svg,
        decoded_image_media_type="image/svg+xml",
        answer=answer,
    )
    sealed = seal_learned_latent_result(
        evidence,
        source="live",
        manifest_sha256="a" * 64,
        run_id="run-1",
        code_version="0" * 40,
        checkpoint_id="d" * 64,
    )
    return {
        "schema_version": "jump.experiment-run/v1",
        "status": "completed",
        "live": True,
        "request_id": "req-completed",
        "plan": plan,
        "result": {
            "experiment_id": plan["experiment_id"],
            "experiment_spec_sha256": sha256_json(plan),
            "sealed_result": sealed,
            "decoded_image": {
                "artifact_name": "predicted-from-z.svg",
                "media_type": "image/svg+xml",
                "encoding": "base64",
                "data": base64.b64encode(svg).decode(),
                "sha256": hashlib.sha256(svg).hexdigest(),
            },
            "presentation": {
                "world_built": "six objects",
                "model_prediction": answer,
                "what_changed": "nothing",
                "correctness": {
                    "format_valid": True,
                    "exact_correct": False,
                    "partition_correct": False,
                    "law_correct": False,
                    "adequacy_correct": False,
                    "force_score": None,
                    "notes": "engineering test",
                },
            },
        },
        "error": None,
    }


def test_completed_result_cannot_be_mixed_with_another_valid_plan():
    plan_a = compile_experiment_intent(request("Predict the future trajectory.", seed=7))
    run = _completed_run(plan_a)
    assert validate_experiment_run(run) == run
    plan_b = compile_experiment_intent(request("Predict the future trajectory.", seed=8))
    mixed = copy.deepcopy(run)
    mixed["plan"] = plan_b
    with pytest.raises(ValueError, match="does not match"):
        validate_experiment_run(mixed)
