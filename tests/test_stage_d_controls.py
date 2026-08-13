from __future__ import annotations

import copy
import hashlib
import struct
from dataclasses import replace

import pytest

from jump_contracts import (
    build_learned_latent_evidence,
    learned_decoder_identity,
    seal_learned_latent_result,
    seal_result_envelope,
)
from jump_mechanistic.stage_d import (
    LATENT_PERMUTATION_VERSION,
    STAGE_D_ARMS,
    STAGE_D_CONTROL_VERSION,
    StageDArmInput,
    StageDControlSpec,
    build_stage_d_control_result,
    execute_stage_d_control_set,
    identity_injection,
    no_z_injection,
    scrambled_injection,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _answer(world: str) -> dict:
    is_a = world == "world-a"
    return {
        "partition": [0, 0, 0, 1, 1, 1] if is_a else [0, 0, 1, 0, 1, 1],
        "replacement_law": {
            "same": "attract" if is_a else "repel",
            "different": "repel" if is_a else "attract",
            "exponent": 2 if is_a else 3,
        },
        "adequate": is_a,
    }


def _latent(recipient: str, donor: str | None, raw: bytes, suffix: str) -> dict:
    evidence = build_learned_latent_evidence(
        encoder_output=raw,
        decoder_input=raw,
        injection_input=raw,
        encoder_observation=f"observation-{suffix}".encode(),
        encoder_observation_artifact_name=f"observation-{suffix}.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[1, 16],
        order="C",
        tensor_artifact_name=f"latent-{suffix}.f32le.bin",
        recipient_world_id=recipient,
        donor_world_id=donor,
        world_pair_id="pair-a-b",
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder.safetensors",
            artifact_sha256=_digest("decoder"),
            training_manifest_sha256=_digest("decoder-training"),
            code_version="decoder-code",
            architecture="latent-observation-v1",
        ),
        decoded_image=f"decoded-{suffix}".encode(),
        decoded_image_media_type="image/png",
        answer=_answer(donor or recipient),
    )
    return seal_learned_latent_result(
        evidence,
        source="live",
        manifest_sha256=_digest("manifest"),
        run_id=f"latent-{suffix}",
        code_version="stage-d-test",
        checkpoint_id="checkpoint-1",
    )


def _matrix() -> tuple[StageDControlSpec, dict[str, StageDArmInput]]:
    raw_a = struct.pack("<16f", *[float(i) for i in range(16)])
    raw_b = struct.pack("<16f", *[float(i + 20) for i in range(16)])
    raw_wrong = struct.pack("<16f", *[float(i + 40) for i in range(16)])
    own = _latent("world-a", None, raw_a, "own-a")
    wrong = _latent("world-a", "world-wrong", raw_wrong, "wrong")
    a_to_b = _latent("world-b", "world-a", raw_a, "a-to-b")
    b_to_a = _latent("world-a", "world-b", raw_b, "b-to-a")
    permutation = list(reversed(range(16)))
    scrambled_binding, scrambled_raw = scrambled_injection(
        own,
        raw_a,
        tensor_artifact_name="latent-own-a.scrambled.f32le.bin",
        seed=17,
        indices=permutation,
    )
    lineages = {
        "own_z": ("world-a", "world-a", own, raw_a, raw_a, identity_injection(own, raw_a)),
        "no_z": ("world-a", None, None, None, None, no_z_injection()),
        "scrambled_z": (
            "world-a",
            "world-a",
            own,
            raw_a,
            scrambled_raw,
            scrambled_binding,
        ),
        "wrong_world_z": (
            "world-a",
            "world-wrong",
            wrong,
            raw_wrong,
            raw_wrong,
            identity_injection(wrong, raw_wrong),
        ),
        "swap_a_to_b": (
            "world-b",
            "world-a",
            a_to_b,
            raw_a,
            raw_a,
            identity_injection(a_to_b, raw_a),
        ),
        "swap_b_to_a": (
            "world-a",
            "world-b",
            b_to_a,
            raw_b,
            raw_b,
            identity_injection(b_to_a, raw_b),
        ),
    }
    result_world = {
        "own_z": "world-a",
        "no_z": "world-b",
        "scrambled_z": "world-b",
        "wrong_world_z": "world-b",
        "swap_a_to_b": "world-a",
        "swap_b_to_a": "world-b",
    }
    arms = {}
    for arm_id, arm_material in lineages.items():
        recipient, source, envelope, source_raw, injected_raw, injection = arm_material
        payload = build_stage_d_control_result(
            arm_id=arm_id,
            checkpoint_id="checkpoint-1",
            manifest_sha256=_digest("manifest"),
            pair_id="pair-a-b",
            recipient_world_id=recipient,
            source_world_id=source,
            answer=_answer(result_world[arm_id]),
            injection=injection,
        )
        arms[arm_id] = StageDArmInput(
            result_envelope=seal_result_envelope(
                payload,
                source="live",
                manifest_sha256=_digest("manifest"),
                run_id=f"control-{arm_id}",
                code_version="stage-d-test",
                checkpoint_id="checkpoint-1",
            ),
            expected_source="live",
            learned_latent_envelope=envelope,
            source_tensor_bytes=source_raw,
            injected_tensor_bytes=injected_raw,
        )
    spec = StageDControlSpec(
        checkpoint_id="checkpoint-1",
        manifest_sha256=_digest("manifest"),
        pair_id="pair-a-b",
        cluster_id="cluster-1",
        world_a_id="world-a",
        world_b_id="world-b",
        wrong_world_id="world-wrong",
        world_a_target=_answer("world-a"),
        world_b_target=_answer("world-b"),
    )
    return spec, arms


def test_exact_six_arm_matrix_uses_shared_envelopes_and_derived_scores():
    spec, arms = _matrix()
    evidence = execute_stage_d_control_set(spec, arms)
    assert tuple(arm.arm_id for arm in evidence.arms) == STAGE_D_ARMS
    assert evidence.arms[0].world_a_score == 1.0
    assert evidence.arms[0].moved_toward_source is True
    assert evidence.arms[4].moved_toward_source is True
    assert evidence.arms[5].moved_toward_source is True
    assert evidence.arms[1].world_latent_sha256 is None
    evidence.verify()

    own_payload = arms["own_z"].result_envelope["payload"]
    assert own_payload["schema_version"] == STAGE_D_CONTROL_VERSION
    assert set(own_payload) == {
        "schema_version",
        "arm_id",
        "control_kind",
        "checkpoint_id",
        "manifest_sha256",
        "pair_id",
        "recipient_world_id",
        "source_world_id",
        "answer",
        "answer_sha256",
        "execution_contract_sha256",
        "injection",
    }


def test_scrambled_arm_is_float32_element_permutation_and_tampering_fails():
    spec, arms = _matrix()
    payload = arms["scrambled_z"].result_envelope["payload"]
    permutation = payload["injection"]["permutation"]
    assert permutation["schema_version"] == LATENT_PERMUTATION_VERSION
    assert permutation["unit"] == "float32_element"
    assert len(set(permutation["indices"])) == 16
    source = arms["scrambled_z"].source_tensor_bytes
    injected = arms["scrambled_z"].injected_tensor_bytes
    assert struct.unpack("<16f", injected) == tuple(reversed(struct.unpack("<16f", source)))

    corrupt = copy.deepcopy(arms)
    record = corrupt["scrambled_z"]
    corrupt["scrambled_z"] = replace(
        record, injected_tensor_bytes=injected[:-4] + source[4:8]
    )
    with pytest.raises(ValueError, match="permutation"):
        execute_stage_d_control_set(spec, corrupt)


def test_partial_malformed_or_spoofed_controls_fail_closed():
    spec, arms = _matrix()
    partial = dict(arms)
    partial.pop("no_z")
    with pytest.raises(ValueError, match="exactly"):
        execute_stage_d_control_set(spec, partial)

    with pytest.raises(ValueError, match="parsed answer"):
        build_stage_d_control_result(
            arm_id="no_z",
            checkpoint_id="checkpoint-1",
            manifest_sha256=_digest("manifest"),
            pair_id="pair-a-b",
            recipient_world_id="world-a",
            source_world_id=None,
            answer={"raw_model_output": "not parsed"},
            injection=no_z_injection(),
        )

    spoofed = dict(arms)
    original = spoofed["swap_b_to_a"]
    spoofed_payload = copy.deepcopy(original.result_envelope["payload"])
    spoofed_payload["moved_toward_source"] = True
    spoofed["swap_b_to_a"] = replace(
        original,
        result_envelope=seal_result_envelope(
            spoofed_payload,
            source="live",
            manifest_sha256=_digest("manifest"),
            run_id="spoofed-control",
            code_version="stage-d-test",
            checkpoint_id="checkpoint-1",
        ),
    )
    with pytest.raises(ValueError, match="exactly"):
        execute_stage_d_control_set(spec, spoofed)
