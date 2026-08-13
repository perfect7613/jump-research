"""Fail-closed Stage D controls over externally executed, sealed results.

This module never loads or runs a model. A producer supplies one sealed result
for each frozen control arm and, when a latent is injected, the independently
sealed learned-latent envelope and exact tensor bytes. The executor verifies
the complete matrix and derives scores from parsed answers; caller-authored
outcome booleans are not part of the contract.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import InitVar, asdict, dataclass
from typing import Any, Mapping

from jump_contracts import (
    canonical_json,
    is_sha256,
    normalize_json_object,
    open_result_envelope,
    tensor_bytes_sha256,
    validate_learned_latent_evidence,
    verify_latent_tensor_bytes,
)

from .gates import SWAP_SCORING_CONTRACT_SHA256
from .scoring import score_episode

STAGE_D_CONTROL_VERSION = "jump.track-h-stage-d-control/v1"
LATENT_PERMUTATION_VERSION = "jump.latent-byte-permutation/v1"
STAGE_D_ARMS = (
    "own_z",
    "no_z",
    "scrambled_z",
    "wrong_world_z",
    "swap_a_to_b",
    "swap_b_to_a",
)
STAGE_D_EXECUTION_CONTRACT = {
    "schema_version": "jump.track-h-stage-d-execution-contract/v1",
    "arms": list(STAGE_D_ARMS),
    "model_execution": "external_producer_only",
    "forward_mode": "teacher_forced_single_forward",
    "answer": "exact_parsed_answer_only",
    "outcome_source": "sealed_answer_scored_by_locked_exact_scorer",
    "no_generation": True,
}
STAGE_D_EXECUTION_CONTRACT_SHA256 = hashlib.sha256(
    canonical_json(STAGE_D_EXECUTION_CONTRACT)
).hexdigest()
_EVIDENCE_TOKEN = object()
_INJECTION_FIELDS = {
    "present",
    "learned_latent_envelope_sha256",
    "tensor_artifact_name",
    "dtype",
    "shape",
    "order",
    "raw_bytes_sha256",
    "world_latent_sha256",
    "permutation",
}


@dataclass(frozen=True)
class StageDControlSpec:
    checkpoint_id: str
    manifest_sha256: str
    pair_id: str
    cluster_id: str
    world_a_id: str
    world_b_id: str
    wrong_world_id: str
    world_a_target: Mapping[str, Any]
    world_b_target: Mapping[str, Any]
    scoring_contract_sha256: str = SWAP_SCORING_CONTRACT_SHA256
    execution_contract_sha256: str = STAGE_D_EXECUTION_CONTRACT_SHA256

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_id",
            "pair_id",
            "cluster_id",
            "world_a_id",
            "world_b_id",
            "wrong_world_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Stage D {name} must be nonempty")
        if len({self.world_a_id, self.world_b_id, self.wrong_world_id}) != 3:
            raise ValueError("Stage D World A, World B, and wrong world must be distinct")
        if not is_sha256(self.manifest_sha256):
            raise ValueError("Stage D manifest_sha256 must be a SHA-256")
        if self.scoring_contract_sha256 != SWAP_SCORING_CONTRACT_SHA256:
            raise ValueError("Stage D scoring contract does not match the locked scorer")
        if self.execution_contract_sha256 != STAGE_D_EXECUTION_CONTRACT_SHA256:
            raise ValueError("Stage D execution contract does not match the locked matrix")
        score_episode(dict(self.world_a_target), dict(self.world_a_target))
        score_episode(dict(self.world_b_target), dict(self.world_b_target))


@dataclass(frozen=True)
class StageDArmInput:
    result_envelope: Mapping[str, Any]
    expected_source: str
    learned_latent_envelope: Mapping[str, Any] | None = None
    source_tensor_bytes: bytes | None = None
    injected_tensor_bytes: bytes | None = None


@dataclass(frozen=True)
class StageDArmEvidence:
    arm_id: str
    control_kind: str
    recipient_world_id: str
    source_world_id: str | None
    result_payload_sha256: str
    learned_latent_envelope_sha256: str | None
    world_latent_sha256: str | None
    raw_bytes_sha256: str | None
    answer_sha256: str
    world_a_score: float
    world_b_score: float
    moved_toward_source: bool | None
    injection_sha256: str


@dataclass(frozen=True)
class StageDControlEvidence:
    schema_version: str
    checkpoint_id: str
    manifest_sha256: str
    pair_id: str
    cluster_id: str
    world_a_id: str
    world_b_id: str
    scoring_contract_sha256: str
    execution_contract_sha256: str
    arms: tuple[StageDArmEvidence, ...]
    content_sha256: str
    _construction_token: InitVar[object]

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _EVIDENCE_TOKEN:
            raise ValueError("Stage D evidence must be built by execute_stage_d_control_set")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "manifest_sha256": self.manifest_sha256,
            "pair_id": self.pair_id,
            "cluster_id": self.cluster_id,
            "world_a_id": self.world_a_id,
            "world_b_id": self.world_b_id,
            "scoring_contract_sha256": self.scoring_contract_sha256,
            "execution_contract_sha256": self.execution_contract_sha256,
            "arms": [asdict(arm) for arm in self.arms],
        }

    def verify(self) -> None:
        if tuple(arm.arm_id for arm in self.arms) != STAGE_D_ARMS:
            raise ValueError("Stage D evidence arm order/coverage drifted")
        if _sha(self.unsigned_dict()) != self.content_sha256:
            raise ValueError("Stage D evidence content hash mismatch")


def build_stage_d_control_result(
    *,
    arm_id: str,
    checkpoint_id: str,
    manifest_sha256: str,
    pair_id: str,
    recipient_world_id: str,
    source_world_id: str | None,
    answer: Mapping[str, Any],
    injection: Mapping[str, Any],
    execution_contract_sha256: str = STAGE_D_EXECUTION_CONTRACT_SHA256,
) -> dict[str, Any]:
    """Build the exact v1 arm payload that an external producer seals.

    Raw model text is deliberately absent. Production must parse the answer
    exactly first; malformed output raises here rather than becoming evidence.
    """
    normalized_answer = _validate_parsed_answer(answer)
    value = {
        "schema_version": STAGE_D_CONTROL_VERSION,
        "arm_id": arm_id,
        "control_kind": arm_id,
        "checkpoint_id": checkpoint_id,
        "manifest_sha256": manifest_sha256,
        "pair_id": pair_id,
        "recipient_world_id": recipient_world_id,
        "source_world_id": source_world_id,
        "answer": normalized_answer,
        "answer_sha256": _sha(normalized_answer),
        "execution_contract_sha256": execution_contract_sha256,
        "injection": dict(injection),
    }
    _validate_control_payload(value)
    return value


def execute_stage_d_control_set(
    spec: StageDControlSpec,
    arms: Mapping[str, StageDArmInput],
) -> StageDControlEvidence:
    """Validate exactly six arms and return immutable, derived score evidence."""
    if not isinstance(spec, StageDControlSpec):
        raise ValueError("Stage D spec must be a StageDControlSpec")
    if not isinstance(arms, Mapping) or set(arms) != set(STAGE_D_ARMS):
        raise ValueError(f"Stage D controls require exactly {list(STAGE_D_ARMS)}")
    evidence = tuple(_execute_arm(spec, arm_id, arms[arm_id]) for arm_id in STAGE_D_ARMS)
    body = {
        "schema_version": "jump.track-h-stage-d-control-evidence/v1",
        "checkpoint_id": spec.checkpoint_id,
        "manifest_sha256": spec.manifest_sha256,
        "pair_id": spec.pair_id,
        "cluster_id": spec.cluster_id,
        "world_a_id": spec.world_a_id,
        "world_b_id": spec.world_b_id,
        "scoring_contract_sha256": spec.scoring_contract_sha256,
        "execution_contract_sha256": spec.execution_contract_sha256,
        "arms": [asdict(arm) for arm in evidence],
    }
    result = StageDControlEvidence(
        **{key: value for key, value in body.items() if key != "arms"},
        arms=evidence,
        content_sha256=_sha(body),
        _construction_token=_EVIDENCE_TOKEN,
    )
    result.verify()
    return result


def _execute_arm(
    spec: StageDControlSpec, arm_id: str, arm: StageDArmInput
) -> StageDArmEvidence:
    if not isinstance(arm, StageDArmInput):
        raise ValueError(f"Stage D {arm_id} must be a StageDArmInput")
    payload = open_result_envelope(
        arm.result_envelope,
        expected_source=arm.expected_source,
        expected_manifest_sha256=spec.manifest_sha256,
        expected_checkpoint_id=spec.checkpoint_id,
    )
    _validate_control_payload(payload)
    fixed_identity = {
        "arm_id": arm_id,
        "control_kind": arm_id,
        "checkpoint_id": spec.checkpoint_id,
        "manifest_sha256": spec.manifest_sha256,
        "pair_id": spec.pair_id,
        "execution_contract_sha256": spec.execution_contract_sha256,
    }
    if any(payload[key] != value for key, value in fixed_identity.items()):
        raise ValueError(f"Stage D {arm_id} result identity does not match its spec")
    recipient, source = _expected_lineage(spec, arm_id)
    if (payload["recipient_world_id"], payload["source_world_id"]) != (recipient, source):
        raise ValueError(f"Stage D {arm_id} recipient/source lineage drifted")

    injection = payload["injection"]
    learned_envelope_sha: str | None = None
    world_latent_sha: str | None = None
    raw_bytes_sha: str | None = None
    if arm_id == "no_z":
        if any(
            value is not None
            for value in (
                arm.learned_latent_envelope,
                arm.source_tensor_bytes,
                arm.injected_tensor_bytes,
            )
        ):
            raise ValueError("Stage D no_z cannot carry latent evidence or tensor bytes")
        if injection != no_z_injection():
            raise ValueError("Stage D no_z injection binding must be exactly absent")
    else:
        if (
            arm.learned_latent_envelope is None
            or arm.source_tensor_bytes is None
            or arm.injected_tensor_bytes is None
        ):
            raise ValueError(f"Stage D {arm_id} requires learned evidence and tensor bytes")
        learned_payload = open_result_envelope(
            arm.learned_latent_envelope,
            expected_source=arm.expected_source,
            expected_manifest_sha256=spec.manifest_sha256,
            expected_checkpoint_id=spec.checkpoint_id,
        )
        learned = validate_learned_latent_evidence(learned_payload)
        verify_latent_tensor_bytes(learned, arm.source_tensor_bytes)
        _require_latent_lineage(learned["swap_lineage"], arm_id, recipient, source, spec.pair_id)
        learned_envelope_sha = _sha(arm.learned_latent_envelope)
        descriptor = learned["tensor"]
        world_latent_sha, raw_bytes_sha = _verify_injection(
            arm_id,
            injection,
            learned_envelope_sha,
            descriptor,
            arm.source_tensor_bytes,
            arm.injected_tensor_bytes,
        )

    answer = payload["answer"]
    score_a = _joint_score(answer, spec.world_a_target)
    score_b = _joint_score(answer, spec.world_b_target)
    moved = None
    if source == spec.world_a_id:
        moved = score_a > score_b
    elif source == spec.world_b_id:
        moved = score_b > score_a
    return StageDArmEvidence(
        arm_id=arm_id,
        control_kind=payload["control_kind"],
        recipient_world_id=recipient,
        source_world_id=source,
        result_payload_sha256=arm.result_envelope["payload_sha256"],
        learned_latent_envelope_sha256=learned_envelope_sha,
        world_latent_sha256=world_latent_sha,
        raw_bytes_sha256=raw_bytes_sha,
        answer_sha256=payload["answer_sha256"],
        world_a_score=score_a,
        world_b_score=score_b,
        moved_toward_source=moved,
        injection_sha256=_sha(injection),
    )


def no_z_injection() -> dict[str, Any]:
    """Return the one valid absent-injection object for the no-z arm."""
    return {
        "present": False,
        "learned_latent_envelope_sha256": None,
        "tensor_artifact_name": None,
        "dtype": None,
        "shape": None,
        "order": None,
        "raw_bytes_sha256": None,
        "world_latent_sha256": None,
        "permutation": None,
    }


def identity_injection(
    learned_latent_envelope: Mapping[str, Any],
    source_tensor_bytes: bytes,
) -> dict[str, Any]:
    """Derive a non-scrambled injection binding from verified shared evidence."""
    learned = validate_learned_latent_evidence(
        open_result_envelope(learned_latent_envelope)
    )
    verify_latent_tensor_bytes(learned, source_tensor_bytes)
    tensor = learned["tensor"]
    return {
        "present": True,
        "learned_latent_envelope_sha256": _sha(learned_latent_envelope),
        "tensor_artifact_name": tensor["artifact_name"],
        "dtype": tensor["dtype"],
        "shape": tensor["shape"],
        "order": tensor["order"],
        "raw_bytes_sha256": tensor["raw_bytes_sha256"],
        "world_latent_sha256": tensor["world_latent_sha256"],
        "permutation": None,
    }


def scrambled_injection(
    learned_latent_envelope: Mapping[str, Any],
    source_tensor_bytes: bytes,
    *,
    tensor_artifact_name: str,
    seed: int,
    indices: list[int],
) -> tuple[dict[str, Any], bytes]:
    """Derive the frozen 16-float permutation and its exact injection binding."""
    learned = validate_learned_latent_evidence(
        open_result_envelope(learned_latent_envelope)
    )
    verify_latent_tensor_bytes(learned, source_tensor_bytes)
    tensor = learned["tensor"]
    if tensor["dtype"] != "float32-le" or math.prod(tensor["shape"]) != 16:
        raise ValueError("Stage D scrambled_z requires exactly 16 little-endian float32 elements")
    _validate_seed_and_indices(seed, indices)
    if indices == list(range(16)):
        raise ValueError("Stage D scrambled_z permutation must not be identity")
    permuted = b"".join(
        source_tensor_bytes[index * 4 : (index + 1) * 4] for index in indices
    )
    if permuted == source_tensor_bytes:
        raise ValueError("Stage D scrambled_z permuted bytes must differ from source")
    permuted_world_sha = tensor_bytes_sha256(
        permuted, dtype="float32-le", shape=tensor["shape"], order=tensor["order"]
    )
    permutation = {
        "schema_version": LATENT_PERMUTATION_VERSION,
        "unit": "float32_element",
        "seed": seed,
        "indices": list(indices),
        "indices_sha256": _sha(indices),
        "source_world_latent_sha256": tensor["world_latent_sha256"],
        "permuted_world_latent_sha256": permuted_world_sha,
    }
    injection = {
        "present": True,
        "learned_latent_envelope_sha256": _sha(learned_latent_envelope),
        "tensor_artifact_name": tensor_artifact_name,
        "dtype": "float32-le",
        "shape": tensor["shape"],
        "order": tensor["order"],
        "raw_bytes_sha256": hashlib.sha256(permuted).hexdigest(),
        "world_latent_sha256": permuted_world_sha,
        "permutation": permutation,
    }
    return injection, permuted


def _validate_control_payload(value: Mapping[str, Any]) -> None:
    required = {
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
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"Stage D control result must contain exactly {sorted(required)}")
    if value["schema_version"] != STAGE_D_CONTROL_VERSION:
        raise ValueError("Stage D control result schema_version drifted")
    if value["arm_id"] not in STAGE_D_ARMS or value["control_kind"] != value["arm_id"]:
        raise ValueError("Stage D control result has an unknown or mismatched control kind")
    for name in ("checkpoint_id", "pair_id", "recipient_world_id"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"Stage D control {name} must be nonempty")
    if not is_sha256(value["manifest_sha256"]):
        raise ValueError("Stage D control manifest_sha256 must be a SHA-256")
    if value["source_world_id"] is not None and (
        not isinstance(value["source_world_id"], str) or not value["source_world_id"]
    ):
        raise ValueError("Stage D source_world_id must be null or nonempty")
    answer = _validate_parsed_answer(value["answer"])
    if _sha(answer) != value["answer_sha256"]:
        raise ValueError("Stage D answer hash mismatch")
    if value["execution_contract_sha256"] != STAGE_D_EXECUTION_CONTRACT_SHA256:
        raise ValueError("Stage D execution contract hash mismatch")
    injection = value["injection"]
    if not isinstance(injection, Mapping) or set(injection) != _INJECTION_FIELDS:
        raise ValueError(f"Stage D injection must contain exactly {sorted(_INJECTION_FIELDS)}")
    if value["arm_id"] == "no_z":
        if value["source_world_id"] is not None or dict(injection) != no_z_injection():
            raise ValueError("Stage D no_z must have null source and exactly absent injection")
    else:
        if value["source_world_id"] is None or injection["present"] is not True:
            raise ValueError("Stage D injected controls require a source and present injection")
        for name in (
            "learned_latent_envelope_sha256",
            "raw_bytes_sha256",
            "world_latent_sha256",
        ):
            if not is_sha256(injection[name]):
                raise ValueError(f"Stage D injection {name} must be a SHA-256")
        for name in ("tensor_artifact_name", "dtype", "order"):
            if not isinstance(injection[name], str) or not injection[name]:
                raise ValueError(f"Stage D injection {name} must be nonempty")
        if (
            not isinstance(injection["shape"], list)
            or not injection["shape"]
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in injection["shape"]
            )
        ):
            raise ValueError("Stage D injection shape must be a nonempty positive integer array")
        if value["arm_id"] == "scrambled_z" and not isinstance(
            injection["permutation"], Mapping
        ):
            raise ValueError("Stage D scrambled_z requires permutation evidence")
        if value["arm_id"] != "scrambled_z" and injection["permutation"] is not None:
            raise ValueError(f"Stage D {value['arm_id']} cannot declare a permutation")


def _validate_parsed_answer(answer: Mapping[str, Any]) -> dict[str, Any]:
    value = normalize_json_object(answer, "Stage D parsed answer")
    required = {"partition", "replacement_law", "adequate"}
    if set(value) != required:
        raise ValueError(f"Stage D parsed answer must contain exactly {sorted(required)}")
    score_episode(value, value)
    return value


def _expected_lineage(spec: StageDControlSpec, arm_id: str) -> tuple[str, str | None]:
    return {
        "own_z": (spec.world_a_id, spec.world_a_id),
        "no_z": (spec.world_a_id, None),
        "scrambled_z": (spec.world_a_id, spec.world_a_id),
        "wrong_world_z": (spec.world_a_id, spec.wrong_world_id),
        "swap_a_to_b": (spec.world_b_id, spec.world_a_id),
        "swap_b_to_a": (spec.world_a_id, spec.world_b_id),
    }[arm_id]


def _require_latent_lineage(
    lineage: Mapping[str, Any], arm_id: str, recipient: str, source: str | None, pair_id: str
) -> None:
    expected_mode = "own" if arm_id in {"own_z", "scrambled_z"} else "donor_swap"
    if (
        lineage["mode"] != expected_mode
        or lineage["recipient_world_id"] != recipient
        or lineage["source_world_id"] != source
        or lineage["donor_world_id"] != (None if expected_mode == "own" else source)
        or lineage["world_pair_id"] != pair_id
    ):
        raise ValueError(f"Stage D {arm_id} learned-latent lineage drifted")


def _verify_injection(
    arm_id: str,
    injection: Mapping[str, Any],
    learned_envelope_sha: str,
    descriptor: Mapping[str, Any],
    source_raw: bytes,
    injected_raw: bytes,
) -> tuple[str, str]:
    if injection["present"] is not True:
        raise ValueError(f"Stage D {arm_id} injection must be present")
    if injection["learned_latent_envelope_sha256"] != learned_envelope_sha:
        raise ValueError(f"Stage D {arm_id} learned envelope hash mismatch")
    if arm_id == "scrambled_z":
        return _verify_scrambled_injection(injection, descriptor, source_raw, injected_raw)

    expected = {
        "tensor_artifact_name": descriptor["artifact_name"],
        "dtype": descriptor["dtype"],
        "shape": descriptor["shape"],
        "order": descriptor["order"],
        "raw_bytes_sha256": descriptor["raw_bytes_sha256"],
        "world_latent_sha256": descriptor["world_latent_sha256"],
        "permutation": None,
    }
    if any(injection[key] != value for key, value in expected.items()):
        raise ValueError(f"Stage D {arm_id} injection binding drifted")
    if injected_raw != source_raw:
        raise ValueError(f"Stage D {arm_id} must inject the verified source tensor unchanged")
    return descriptor["world_latent_sha256"], descriptor["raw_bytes_sha256"]


def _verify_scrambled_injection(
    injection: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    source_raw: bytes,
    injected_raw: bytes,
) -> tuple[str, str]:
    if descriptor["dtype"] != "float32-le" or math.prod(descriptor["shape"]) != 16:
        raise ValueError("Stage D scrambled_z requires exactly 16 little-endian float32 elements")
    if not isinstance(injection["tensor_artifact_name"], str) or not injection[
        "tensor_artifact_name"
    ]:
        raise ValueError("Stage D scrambled_z tensor_artifact_name must be nonempty")
    if (
        injection["dtype"] != "float32-le"
        or injection["shape"] != descriptor["shape"]
        or injection["order"] != descriptor["order"]
    ):
        raise ValueError("Stage D scrambled_z tensor descriptor drifted")
    permutation = injection["permutation"]
    required = {
        "schema_version",
        "unit",
        "seed",
        "indices",
        "indices_sha256",
        "source_world_latent_sha256",
        "permuted_world_latent_sha256",
    }
    if not isinstance(permutation, Mapping) or set(permutation) != required:
        raise ValueError(f"Stage D permutation must contain exactly {sorted(required)}")
    if (
        permutation["schema_version"] != LATENT_PERMUTATION_VERSION
        or permutation["unit"] != "float32_element"
    ):
        raise ValueError("Stage D permutation contract drifted")
    indices = permutation["indices"]
    _validate_seed_and_indices(permutation["seed"], indices)
    if indices == list(range(16)):
        raise ValueError("Stage D scrambled_z permutation must not be identity")
    if permutation["indices_sha256"] != _sha(indices):
        raise ValueError("Stage D scrambled_z indices hash mismatch")
    if permutation["source_world_latent_sha256"] != descriptor["world_latent_sha256"]:
        raise ValueError("Stage D scrambled_z source latent hash mismatch")
    expected_bytes = b"".join(source_raw[index * 4 : (index + 1) * 4] for index in indices)
    if expected_bytes == source_raw or expected_bytes != injected_raw:
        raise ValueError("Stage D scrambled_z bytes do not match a non-identity permutation")
    raw_sha = hashlib.sha256(injected_raw).hexdigest()
    world_sha = tensor_bytes_sha256(
        injected_raw,
        dtype="float32-le",
        shape=descriptor["shape"],
        order=descriptor["order"],
    )
    if (
        injection["raw_bytes_sha256"] != raw_sha
        or injection["world_latent_sha256"] != world_sha
        or permutation["permuted_world_latent_sha256"] != world_sha
        or world_sha == descriptor["world_latent_sha256"]
    ):
        raise ValueError("Stage D scrambled_z permuted tensor hash mismatch")
    return world_sha, raw_sha


def _validate_seed_and_indices(seed: Any, indices: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Stage D permutation seed must be a uint32")
    if (
        not isinstance(indices, list)
        or len(indices) != 16
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or sorted(indices) != list(range(16))
    ):
        raise ValueError("Stage D permutation indices must be 16 unique values 0..15")


def _joint_score(answer: Mapping[str, Any], target: Mapping[str, Any]) -> float:
    scores = score_episode(dict(answer), dict(target))
    return (scores["joint_theory_accuracy"] + scores["adequacy_correct"]) / 2.0


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
