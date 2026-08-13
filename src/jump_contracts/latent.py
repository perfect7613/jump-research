"""Fail-closed evidence for the authentic Track H learned-latent slice.

This module binds tensor bytes, rather than prompts or decoded descriptions, to
the encoder, learned decoder, Gemma injection, answer, cache, and live result.
It intentionally contains no simulator, model, or UI implementation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .evidence import (
    EvidenceError,
    canonical_json,
    is_sha256,
    normalize_json_object,
    seal_result_envelope,
)


LATENT_EVIDENCE_VERSION = "jump.learned-latent-evidence/v1"
TENSOR_PREIMAGE_VERSION = "jump.tensor-preimage/v1"
_DTYPE_BYTES = {
    "float16-le": 2,
    "bfloat16-le": 2,
    "float32-le": 4,
    "float64-le": 8,
    "int8": 1,
    "int16-le": 2,
    "int32-le": 4,
    "int64-le": 8,
}
_ORDERS = frozenset({"C", "F"})
_DECODED_IMAGE_ORIGIN = "learned_decoder_output"


def tensor_bytes_sha256(
    raw: bytes | bytearray | memoryview,
    *,
    dtype: str,
    shape: Sequence[int],
    order: str = "C",
) -> str:
    """Hash a tensor's explicit metadata and exact contiguous raw bytes.

    Dtypes include byte order so a producer cannot hash a platform-native
    tensor ambiguously. The preimage is canonical metadata, a NUL separator,
    and the raw bytes. Prompt strings and rendered values are never inputs.
    """
    content, descriptor = _tensor_input(raw, dtype=dtype, shape=shape, order=order)
    return hashlib.sha256(canonical_json(descriptor) + b"\0" + content).hexdigest()


def learned_decoder_identity(
    *,
    artifact_name: str,
    artifact_sha256: str,
    training_manifest_sha256: str,
    code_version: str,
    architecture: str,
) -> dict[str, str]:
    """Build the immutable identity of the learned image decoder artifact."""
    value = {
        "artifact_name": artifact_name,
        "artifact_sha256": artifact_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "code_version": code_version,
        "architecture": architecture,
    }
    _validate_decoder_identity(value)
    return value


def build_learned_latent_evidence(
    *,
    encoder_output: bytes | bytearray | memoryview,
    decoder_input: bytes | bytearray | memoryview,
    injection_input: bytes | bytearray | memoryview,
    encoder_observation: bytes | bytearray | memoryview,
    encoder_observation_artifact_name: str,
    encoder_observation_media_type: str,
    dtype: str,
    shape: Sequence[int],
    order: str,
    recipient_world_id: str,
    world_pair_id: str,
    learned_decoder: Mapping[str, Any],
    decoded_image: bytes | bytearray | memoryview,
    decoded_image_media_type: str,
    answer: Mapping[str, Any],
    tensor_artifact_name: str,
    donor_world_id: str | None = None,
) -> dict[str, Any]:
    """Bind one learned tensor to all consumers and its resulting outputs.

    ``encoder_output`` is the authoritative world latent. The other two byte
    sequences are independently hashed with the same dtype/shape/order and must
    be byte-identical. Donor swaps change lineage, never the tensor contract.
    """
    encoder_bytes, tensor = _tensor_input(
        encoder_output, dtype=dtype, shape=shape, order=order
    )
    decoder_bytes, _ = _tensor_input(decoder_input, dtype=dtype, shape=shape, order=order)
    injection_bytes, _ = _tensor_input(
        injection_input, dtype=dtype, shape=shape, order=order
    )
    world_hash = tensor_bytes_sha256(encoder_bytes, dtype=dtype, shape=shape, order=order)
    decoder_hash = tensor_bytes_sha256(decoder_bytes, dtype=dtype, shape=shape, order=order)
    injection_hash = tensor_bytes_sha256(
        injection_bytes, dtype=dtype, shape=shape, order=order
    )
    if len({world_hash, decoder_hash, injection_hash}) != 1:
        raise EvidenceError(
            "encoder output, decoder input, and injection input tensor bytes must be identical"
        )

    decoder = dict(learned_decoder)
    _validate_decoder_identity(decoder)
    image_bytes = _bytes(decoded_image, "decoded image")
    if not isinstance(decoded_image_media_type, str) or not decoded_image_media_type.startswith(
        "image/"
    ):
        raise EvidenceError("decoded image media type must be image/*")
    normalized_answer = normalize_json_object(answer, "answer")
    observation_bytes = _bytes(encoder_observation, "encoder observation")
    if not encoder_observation_artifact_name:
        raise EvidenceError("encoder observation artifact name must be nonempty")
    if not encoder_observation_media_type:
        raise EvidenceError("encoder observation media type must be nonempty")
    lineage = _swap_lineage(
        recipient_world_id=recipient_world_id,
        donor_world_id=donor_world_id,
        world_pair_id=world_pair_id,
    )
    evidence = {
        "schema_version": LATENT_EVIDENCE_VERSION,
        "encoder_input": {
            "kind": "observation_only",
            "artifact_name": encoder_observation_artifact_name,
            "media_type": encoder_observation_media_type,
            "observation_sha256": hashlib.sha256(observation_bytes).hexdigest(),
            "ground_truth_fields_present": False,
        },
        "tensor": {
            **tensor,
            "artifact_name": tensor_artifact_name,
            "raw_bytes_sha256": hashlib.sha256(encoder_bytes).hexdigest(),
            "world_latent_sha256": world_hash,
            "encoder_output_sha256": world_hash,
            "decoder_input_sha256": decoder_hash,
            "injection_input_sha256": injection_hash,
        },
        "swap_lineage": lineage,
        "learned_decoder": decoder,
        "decoded_observation": {
            "origin": _DECODED_IMAGE_ORIGIN,
            "media_type": decoded_image_media_type,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "world_latent_sha256": world_hash,
            "decoder_artifact_sha256": decoder["artifact_sha256"],
        },
        "answer": normalized_answer,
        "answer_binding": {
            "answer_sha256": hashlib.sha256(canonical_json(normalized_answer)).hexdigest(),
            "world_latent_sha256": world_hash,
            "injection_input_sha256": injection_hash,
        },
    }
    return validate_learned_latent_evidence(evidence)


def validate_learned_latent_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate latent metadata before decoding, caching, or display.

    This validates the serialized record, not separately transported tensor
    bytes. Call :func:`verify_latent_tensor_bytes` with the declared dtype,
    shape, and order before interpreting or caching any raw tensor artifact.
    """
    evidence = normalize_json_object(value, "learned latent evidence")
    required = {
        "schema_version",
        "encoder_input",
        "tensor",
        "swap_lineage",
        "learned_decoder",
        "decoded_observation",
        "answer",
        "answer_binding",
    }
    if set(evidence) != required:
        raise EvidenceError(f"learned latent evidence must contain exactly {sorted(required)}")
    if evidence["schema_version"] != LATENT_EVIDENCE_VERSION:
        raise EvidenceError(f"schema_version must be {LATENT_EVIDENCE_VERSION}")

    encoder_input = evidence["encoder_input"]
    encoder_fields = {
        "kind",
        "artifact_name",
        "media_type",
        "observation_sha256",
        "ground_truth_fields_present",
    }
    if not isinstance(encoder_input, dict) or set(encoder_input) != encoder_fields:
        raise EvidenceError(f"encoder input must contain exactly {sorted(encoder_fields)}")
    if encoder_input["kind"] != "observation_only":
        raise EvidenceError("world encoder input must be observation_only")
    for key in ("artifact_name", "media_type"):
        if not isinstance(encoder_input[key], str) or not encoder_input[key]:
            raise EvidenceError(f"encoder input {key} must be a nonempty string")
    if not is_sha256(encoder_input["observation_sha256"]):
        raise EvidenceError("encoder input observation_sha256 must be a SHA-256")
    if encoder_input["ground_truth_fields_present"] is not False:
        raise EvidenceError("world encoder input must not contain ground-truth fields")

    tensor = evidence["tensor"]
    tensor_fields = {
        "preimage_schema_version",
        "artifact_name",
        "dtype",
        "shape",
        "order",
        "byte_length",
        "raw_bytes_sha256",
        "world_latent_sha256",
        "encoder_output_sha256",
        "decoder_input_sha256",
        "injection_input_sha256",
    }
    if not isinstance(tensor, dict) or set(tensor) != tensor_fields:
        raise EvidenceError(f"latent tensor must contain exactly {sorted(tensor_fields)}")
    _validate_tensor_descriptor(tensor)
    if not isinstance(tensor["artifact_name"], str) or not tensor["artifact_name"]:
        raise EvidenceError("latent tensor artifact_name must be a nonempty string")
    if not is_sha256(tensor["raw_bytes_sha256"]):
        raise EvidenceError("latent tensor raw_bytes_sha256 must be a SHA-256")
    hashes = [
        tensor[name]
        for name in (
            "world_latent_sha256",
            "encoder_output_sha256",
            "decoder_input_sha256",
            "injection_input_sha256",
        )
    ]
    if any(not is_sha256(digest) for digest in hashes) or len(set(hashes)) != 1:
        raise EvidenceError(
            "world_latent_sha256, encoder_output_sha256, decoder_input_sha256, "
            "and injection_input_sha256 must be identical SHA-256 digests"
        )

    _validate_swap_lineage(evidence["swap_lineage"])
    _validate_decoder_identity(evidence["learned_decoder"])
    observation = evidence["decoded_observation"]
    observation_fields = {
        "origin",
        "media_type",
        "image_sha256",
        "world_latent_sha256",
        "decoder_artifact_sha256",
    }
    if not isinstance(observation, dict) or set(observation) != observation_fields:
        raise EvidenceError(
            f"decoded observation must contain exactly {sorted(observation_fields)}"
        )
    if observation["origin"] != _DECODED_IMAGE_ORIGIN:
        raise EvidenceError(
            "decoded observation must originate from the declared learned decoder; "
            "ground-truth/simulator renders cannot be presented as decoded"
        )
    if not isinstance(observation["media_type"], str) or not observation["media_type"].startswith(
        "image/"
    ):
        raise EvidenceError("decoded observation media type must be image/*")
    if not is_sha256(observation["image_sha256"]):
        raise EvidenceError("decoded observation image_sha256 must be a SHA-256")
    if observation["world_latent_sha256"] != tensor["world_latent_sha256"]:
        raise EvidenceError("decoded observation is not bound to the world latent")
    if observation["decoder_artifact_sha256"] != evidence["learned_decoder"]["artifact_sha256"]:
        raise EvidenceError("decoded observation is not bound to the learned decoder artifact")

    answer = normalize_json_object(evidence["answer"], "answer")
    binding = evidence["answer_binding"]
    binding_fields = {"answer_sha256", "world_latent_sha256", "injection_input_sha256"}
    if not isinstance(binding, dict) or set(binding) != binding_fields:
        raise EvidenceError(f"answer binding must contain exactly {sorted(binding_fields)}")
    expected_answer = hashlib.sha256(canonical_json(answer)).hexdigest()
    if binding["answer_sha256"] != expected_answer:
        raise EvidenceError("answer hash does not match the bound answer")
    if binding["world_latent_sha256"] != tensor["world_latent_sha256"]:
        raise EvidenceError("answer is not bound to the world latent")
    if binding["injection_input_sha256"] != tensor["injection_input_sha256"]:
        raise EvidenceError("answer is not bound to the Gemma injection input")
    return evidence


def seal_learned_latent_result(
    evidence: Mapping[str, Any],
    *,
    source: str,
    manifest_sha256: str,
    run_id: str,
    code_version: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Validate the vertical slice, then seal it for cache or live delivery."""
    payload = validate_learned_latent_evidence(evidence)
    return seal_result_envelope(
        payload,
        source=source,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        code_version=code_version,
        checkpoint_id=checkpoint_id,
    )


def verify_decoded_image_bytes(
    evidence: Mapping[str, Any], image: bytes | bytearray | memoryview
) -> None:
    """Verify the exact image bytes immediately before display or caching."""
    value = validate_learned_latent_evidence(evidence)
    if hashlib.sha256(_bytes(image, "decoded image")).hexdigest() != value[
        "decoded_observation"
    ]["image_sha256"]:
        raise EvidenceError("displayed image bytes do not match the learned decoder output")


def verify_latent_tensor_bytes(
    evidence: Mapping[str, Any], tensor: bytes | bytearray | memoryview
) -> None:
    """Verify a stored/cache tensor before decoder or Gemma consumption."""
    value = validate_learned_latent_evidence(evidence)
    descriptor = value["tensor"]
    raw = _bytes(tensor, "tensor")
    if hashlib.sha256(raw).hexdigest() != descriptor["raw_bytes_sha256"]:
        raise EvidenceError("raw latent tensor artifact hash mismatch")
    digest = tensor_bytes_sha256(
        raw,
        dtype=descriptor["dtype"],
        shape=descriptor["shape"],
        order=descriptor["order"],
    )
    if digest != descriptor["world_latent_sha256"]:
        raise EvidenceError("raw latent tensor does not match world_latent_sha256")


def verify_encoder_observation_bytes(
    evidence: Mapping[str, Any], observation: bytes | bytearray | memoryview
) -> None:
    """Verify the exact observation-only encoder input artifact bytes."""
    value = validate_learned_latent_evidence(evidence)
    digest = hashlib.sha256(_bytes(observation, "encoder observation")).hexdigest()
    if digest != value["encoder_input"]["observation_sha256"]:
        raise EvidenceError("encoder observation bytes do not match observation_sha256")


def _tensor_input(
    raw: bytes | bytearray | memoryview, *, dtype: str, shape: Sequence[int], order: str
) -> tuple[bytes, dict[str, Any]]:
    content = _bytes(raw, "tensor")
    descriptor = {
        "preimage_schema_version": TENSOR_PREIMAGE_VERSION,
        "dtype": dtype,
        "shape": list(shape) if isinstance(shape, Sequence) else shape,
        "order": order,
        "byte_length": len(content),
    }
    _validate_tensor_descriptor(descriptor)
    return content, descriptor


def _validate_tensor_descriptor(value: Mapping[str, Any]) -> None:
    descriptor_fields = {
        "preimage_schema_version",
        "dtype",
        "shape",
        "order",
        "byte_length",
    }
    descriptor = {key: value.get(key) for key in descriptor_fields}
    if descriptor["preimage_schema_version"] != TENSOR_PREIMAGE_VERSION:
        raise EvidenceError(f"tensor preimage schema must be {TENSOR_PREIMAGE_VERSION}")
    if descriptor["dtype"] not in _DTYPE_BYTES:
        raise EvidenceError(f"tensor dtype must be one of {sorted(_DTYPE_BYTES)}")
    shape = descriptor["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise EvidenceError("tensor shape must be a nonempty array of positive integers")
    if descriptor["order"] not in _ORDERS:
        raise EvidenceError("tensor order must be C or F")
    expected = math.prod(shape) * _DTYPE_BYTES[descriptor["dtype"]]
    if descriptor["byte_length"] != expected:
        raise EvidenceError(
            f"tensor byte length does not match dtype/shape: {descriptor['byte_length']} != {expected}"
        )


def _swap_lineage(
    *, recipient_world_id: str, donor_world_id: str | None, world_pair_id: str
) -> dict[str, Any]:
    if donor_world_id is None:
        value = {
            "mode": "own",
            "world_pair_id": world_pair_id,
            "recipient_world_id": recipient_world_id,
            "source_world_id": recipient_world_id,
            "donor_world_id": None,
            "direction": "own",
        }
    else:
        value = {
            "mode": "donor_swap",
            "world_pair_id": world_pair_id,
            "recipient_world_id": recipient_world_id,
            "source_world_id": donor_world_id,
            "donor_world_id": donor_world_id,
            "direction": f"{donor_world_id}->{recipient_world_id}",
        }
    _validate_swap_lineage(value)
    return value


def _validate_swap_lineage(value: Any) -> None:
    fields = {
        "mode",
        "world_pair_id",
        "recipient_world_id",
        "source_world_id",
        "donor_world_id",
        "direction",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"swap lineage must contain exactly {sorted(fields)}")
    for key in ("world_pair_id", "recipient_world_id", "source_world_id", "direction"):
        if not isinstance(value[key], str) or not value[key]:
            raise EvidenceError(f"swap lineage {key} must be a nonempty string")
    if value["mode"] == "own":
        if (
            value["donor_world_id"] is not None
            or value["source_world_id"] != value["recipient_world_id"]
            or value["direction"] != "own"
        ):
            raise EvidenceError("own-latent lineage must source the recipient world")
    elif value["mode"] == "donor_swap":
        donor = value["donor_world_id"]
        if (
            not isinstance(donor, str)
            or not donor
            or donor == value["recipient_world_id"]
            or value["source_world_id"] != donor
            or value["direction"] != f"{donor}->{value['recipient_world_id']}"
        ):
            raise EvidenceError("donor-swap lineage must bind a distinct donor to the recipient")
    else:
        raise EvidenceError("swap lineage mode must be own or donor_swap")


def _validate_decoder_identity(value: Any) -> None:
    fields = {
        "artifact_name",
        "artifact_sha256",
        "training_manifest_sha256",
        "code_version",
        "architecture",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"learned decoder identity must contain exactly {sorted(fields)}")
    for key in ("artifact_name", "code_version", "architecture"):
        if not isinstance(value[key], str) or not value[key]:
            raise EvidenceError(f"learned decoder {key} must be a nonempty string")
    for key in ("artifact_sha256", "training_manifest_sha256"):
        if not is_sha256(value[key]):
            raise EvidenceError(f"learned decoder {key} must be a SHA-256")


def _bytes(value: bytes | bytearray | memoryview, where: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise EvidenceError(f"{where} must be raw bytes")
    return bytes(value)
