from __future__ import annotations

import copy
import struct

import pytest

from jump_contracts import (
    EvidenceError,
    build_learned_latent_evidence,
    learned_decoder_identity,
    open_result_envelope,
    seal_learned_latent_result,
    tensor_bytes_sha256,
    validate_learned_latent_evidence,
    verify_decoded_image_bytes,
    verify_encoder_observation_bytes,
    verify_latent_tensor_bytes,
)


def _z(values: tuple[float, ...] = (0.25, -1.5, 2.0, 0.0)) -> bytes:
    return struct.pack("<4f", *values)


def _decoder() -> dict[str, str]:
    return learned_decoder_identity(
        artifact_name="learned-observation-decoder.safetensors",
        artifact_sha256="d" * 64,
        training_manifest_sha256="e" * 64,
        code_version="decoder-training-commit",
        architecture="latent-to-six-object-observation-v1",
    )


def _evidence(*, donor: str | None = None) -> dict:
    raw = _z()
    return build_learned_latent_evidence(
        encoder_output=raw,
        decoder_input=raw,
        injection_input=raw,
        encoder_observation=b"observation-only rendered prefix",
        encoder_observation_artifact_name="observed-prefix.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[1, 4],
        order="C",
        tensor_artifact_name="world-latent.f32le.bin",
        recipient_world_id="world-b",
        donor_world_id=donor,
        world_pair_id="pair-a-b",
        learned_decoder=_decoder(),
        decoded_image=b"learned decoder image bytes",
        decoded_image_media_type="image/png",
        answer={"partition": [0, 0, 1, 1, 0, 1], "adequate": False},
    )


def test_tensor_hash_preimage_includes_dtype_shape_order_and_raw_bytes_not_prompts():
    raw = _z()
    digest = tensor_bytes_sha256(raw, dtype="float32-le", shape=[1, 4], order="C")
    assert digest == tensor_bytes_sha256(raw, dtype="float32-le", shape=[1, 4], order="C")
    assert digest != tensor_bytes_sha256(raw, dtype="float16-le", shape=[2, 4], order="C")
    assert digest != tensor_bytes_sha256(raw, dtype="float32-le", shape=[4, 1], order="C")
    assert digest != tensor_bytes_sha256(raw, dtype="float32-le", shape=[1, 4], order="F")
    assert digest != tensor_bytes_sha256(_z((0.25, -1.5, 2.0, 1.0)), dtype="float32-le", shape=[1, 4], order="C")
    # There is deliberately no prompt/text parameter in the interface.
    with pytest.raises(TypeError):
        tensor_bytes_sha256(raw, dtype="float32-le", shape=[1, 4], order="C", prompt="z")


def test_tensor_hash_format_has_a_frozen_cross_package_vector():
    assert tensor_bytes_sha256(
        bytes(range(64)), dtype="float32-le", shape=[16], order="C"
    ) == "72a507373e1d8b984f28cd6c2258a5a09ef5fb24d3b39122c065cf046db49d36"


@pytest.mark.parametrize("source", ["cached", "live"])
def test_one_learned_z_is_identical_across_encoder_decoder_injection_answer_and_envelope(source):
    evidence = _evidence()
    tensor = evidence["tensor"]
    assert {
        tensor["world_latent_sha256"],
        tensor["encoder_output_sha256"],
        tensor["decoder_input_sha256"],
        tensor["injection_input_sha256"],
        evidence["answer_binding"]["world_latent_sha256"],
        evidence["answer_binding"]["injection_input_sha256"],
        evidence["decoded_observation"]["world_latent_sha256"],
    } == {tensor["world_latent_sha256"]}
    verify_latent_tensor_bytes(evidence, _z())
    verify_encoder_observation_bytes(evidence, b"observation-only rendered prefix")
    verify_decoded_image_bytes(evidence, b"learned decoder image bytes")

    envelope = seal_learned_latent_result(
        evidence,
        source=source,
        manifest_sha256="a" * 64,
        run_id="live-request-001",
        code_version="inference-commit",
        checkpoint_id="gemma-pinned-revision",
    )
    assert open_result_envelope(
        envelope,
        expected_source=source,
        expected_manifest_sha256="a" * 64,
        expected_checkpoint_id="gemma-pinned-revision",
    ) == evidence


def test_mismatched_decoder_or_injection_bytes_fail_closed():
    raw, changed = _z(), _z((0.25, -1.5, 2.0, 1.0))
    kwargs = dict(
        encoder_output=raw,
        decoder_input=raw,
        injection_input=raw,
        encoder_observation=b"observation-only rendered prefix",
        encoder_observation_artifact_name="observed-prefix.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[1, 4],
        order="C",
        tensor_artifact_name="world-latent.f32le.bin",
        recipient_world_id="world-b",
        world_pair_id="pair-a-b",
        learned_decoder=_decoder(),
        decoded_image=b"learned",
        decoded_image_media_type="image/png",
        answer={"adequate": False},
    )
    with pytest.raises(EvidenceError, match="tensor bytes must be identical"):
        build_learned_latent_evidence(**(kwargs | {"decoder_input": changed}))
    with pytest.raises(EvidenceError, match="tensor bytes must be identical"):
        build_learned_latent_evidence(**(kwargs | {"injection_input": changed}))


def test_donor_swap_lineage_is_literal_and_distinct():
    evidence = _evidence(donor="world-a")
    assert evidence["swap_lineage"] == {
        "mode": "donor_swap",
        "world_pair_id": "pair-a-b",
        "recipient_world_id": "world-b",
        "source_world_id": "world-a",
        "donor_world_id": "world-a",
        "direction": "world-a->world-b",
    }
    tampered = copy.deepcopy(evidence)
    tampered["swap_lineage"]["recipient_world_id"] = "world-a"
    with pytest.raises(EvidenceError, match="distinct donor"):
        validate_learned_latent_evidence(tampered)


def test_learned_decoder_identity_and_output_bytes_are_required():
    evidence = _evidence()
    tampered = copy.deepcopy(evidence)
    tampered["learned_decoder"]["artifact_sha256"] = "f" * 64
    with pytest.raises(EvidenceError, match="not bound to the learned decoder artifact"):
        validate_learned_latent_evidence(tampered)
    with pytest.raises(EvidenceError, match="displayed image bytes"):
        verify_decoded_image_bytes(evidence, b"different image")
    with pytest.raises(EvidenceError, match="raw latent tensor artifact"):
        verify_latent_tensor_bytes(evidence, _z((0.25, -1.5, 2.0, 1.0)))
    with pytest.raises(EvidenceError, match="encoder observation bytes"):
        verify_encoder_observation_bytes(evidence, b"ground truth instead")


def test_encoder_input_must_be_observation_only_and_ground_truth_free():
    evidence = _evidence()
    evidence["encoder_input"]["kind"] = "target_partition"
    with pytest.raises(EvidenceError, match="observation_only"):
        validate_learned_latent_evidence(evidence)
    evidence = _evidence()
    evidence["encoder_input"]["ground_truth_fields_present"] = True
    with pytest.raises(EvidenceError, match="must not contain ground-truth"):
        validate_learned_latent_evidence(evidence)


@pytest.mark.parametrize(
    "origin",
    ["ground_truth_renderer", "simulator_render", "deterministic_fixture"],
)
def test_ground_truth_rendered_images_cannot_be_presented_as_decoded(origin):
    evidence = _evidence()
    evidence["decoded_observation"]["origin"] = origin
    with pytest.raises(EvidenceError, match="ground-truth/simulator renders"):
        validate_learned_latent_evidence(evidence)


def test_serialized_hash_aliases_and_answer_drift_fail_closed():
    evidence = _evidence()
    for field in ("decoder_input_sha256", "injection_input_sha256"):
        tampered = copy.deepcopy(evidence)
        tampered["tensor"][field] = "0" * 64
        with pytest.raises(EvidenceError, match="must be identical"):
            validate_learned_latent_evidence(tampered)
    tampered = copy.deepcopy(evidence)
    tampered["answer"]["adequate"] = True
    with pytest.raises(EvidenceError, match="answer hash"):
        validate_learned_latent_evidence(tampered)
