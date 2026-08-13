"""Computed, content-bound research gates for Track R.

Gate decisions are derived from per-episode records. No API accepts a caller's
``g*_passed`` Boolean as evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import InitVar, asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from jump_contracts import (
    canonical_json,
    open_result_envelope,
    validate_learned_latent_evidence,
    verify_latent_tensor_bytes,
)

from .metrics import CheckpointIdentity, paired_effect
from .scoring import score_episode

_REQUIRED_HASH = 64
_MIN_CONFIRMATORY_CLUSTERS = 20
_MIN_REGIME_RECORDS = 200
_G3_CONDITIONS = ("E", "G", "W", "I", "C_prime", "T1c", "T2c")
_G5_CONDITIONS = (
    "target", "baseline", "matched_norm", "orthogonal", "generic_error", "sham", "prompt_length"
)
_MEDIATORS = ("primary", "matched_norm", "orthogonal", "generic_error", "sham", "prompt_length")
_STAGES = ("inadequacy", "promotion")
_ARMS = ("control_natural", "treated_natural", "treated_control_clamp")
_SWAP_RECORD_TOKEN = object()
_GATE_DECISION_TOKEN = object()
_SWAP_COMPARISON_TOKEN = object()
SWAP_SCORING_CONTRACT = {
    "schema_version": "jump.swap-scoring/v1",
    "scorer": "jump_mechanistic.scoring.score_episode",
    "score_fields": ["joint_theory_accuracy", "adequacy_correct"],
    "decision": "donor_score > recipient_score",
    "reference_requirements": [
        "recipient_baseline_prefers_recipient_target",
        "donor_reference_prefers_donor_target",
    ],
}
SWAP_SCORING_CONTRACT_SHA256 = hashlib.sha256(
    canonical_json(SWAP_SCORING_CONTRACT)
).hexdigest()


@dataclass(frozen=True)
class GateSubcondition:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    gate: str
    passed: bool
    subconditions: tuple[GateSubcondition, ...]
    evidence_sha256: str
    calculation_sha256: str
    _construction_token: InitVar[object]

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _GATE_DECISION_TOKEN:
            raise ValueError("gate decisions must be computed from raw evidence records")

    @property
    def reasons(self) -> list[str]:
        return [item.detail for item in self.subconditions if not item.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "subconditions": [asdict(item) for item in self.subconditions],
            "evidence_sha256": self.evidence_sha256,
            "calculation_sha256": self.calculation_sha256,
        }

    def verify(self) -> None:
        expected = _hash(
            {
                "gate": self.gate,
                "passed": self.passed,
                "subconditions": [asdict(item) for item in self.subconditions],
                "evidence_sha256": self.evidence_sha256,
            }
        )
        if expected != self.calculation_sha256:
            raise ValueError(f"{self.gate} calculation hash mismatch")
        if self.passed != all(item.passed for item in self.subconditions):
            raise ValueError(f"{self.gate} pass flag is inconsistent with subconditions")


@dataclass(frozen=True)
class RegimeRecord:
    episode_id: str
    parsed: bool
    joint_correct: bool


@dataclass(frozen=True)
class BehaviorConditionRecord:
    checkpoint_id: str
    episode_id: str
    cluster_id: str
    condition: str
    joint_correct: bool
    prompt_token_count: int
    decoding_sha256: str


@dataclass(frozen=True)
class SwapComparisonBinding:
    direction: str
    recipient_baseline_answer_sha256: str
    donor_reference_answer_sha256: str
    swapped_answer_sha256: str
    recipient_target_sha256: str
    donor_target_sha256: str
    scoring_contract_sha256: str
    recipient_score: float
    donor_score: float
    moved_toward_donor: bool
    content_sha256: str
    _construction_token: InitVar[object]

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _SWAP_COMPARISON_TOKEN:
            raise ValueError("swap comparisons must be computed by the locked scoring factory")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "jump.swap-comparison/v1",
            "direction": self.direction,
            "recipient_baseline_answer_sha256": self.recipient_baseline_answer_sha256,
            "donor_reference_answer_sha256": self.donor_reference_answer_sha256,
            "swapped_answer_sha256": self.swapped_answer_sha256,
            "recipient_target_sha256": self.recipient_target_sha256,
            "donor_target_sha256": self.donor_target_sha256,
            "scoring_contract_sha256": self.scoring_contract_sha256,
            "recipient_score": self.recipient_score,
            "donor_score": self.donor_score,
            "moved_toward_donor": self.moved_toward_donor,
        }

    def verify(self) -> None:
        if self.scoring_contract_sha256 != SWAP_SCORING_CONTRACT_SHA256:
            raise ValueError("swap comparison scoring contract drifted")
        for name in (
            "recipient_baseline_answer_sha256",
            "donor_reference_answer_sha256",
            "swapped_answer_sha256",
            "recipient_target_sha256",
            "donor_target_sha256",
            "scoring_contract_sha256",
        ):
            _digest(getattr(self, name), f"swap comparison {name}")
        if self.moved_toward_donor != (self.donor_score > self.recipient_score):
            raise ValueError("swap outcome is inconsistent with locked comparison scores")
        if _hash(self.unsigned_dict()) != self.content_sha256:
            raise ValueError("swap comparison content hash mismatch")


@dataclass(frozen=True)
class SwapOutcomeRecord:
    checkpoint_id: str
    pair_id: str
    cluster_id: str
    direction: str
    moved_toward_donor: bool
    recipient_prompt_token_count: int
    donor_prompt_token_count: int
    evidence_namespace: str
    world_latent_sha256: str
    decoder_input_sha256: str
    injection_input_sha256: str
    answer_world_latent_sha256: str
    delivered_world_latent_sha256: str
    answer_sha256: str
    envelope_payload_sha256: str
    donor_world_id: str
    recipient_world_id: str
    world_a_id: str
    world_b_id: str
    comparison: SwapComparisonBinding | None
    _construction_token: InitVar[object]

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _SWAP_RECORD_TOKEN:
            raise ValueError(
                "swap records must be built through fixture_nonclaim() or "
                "from_learned_latent_envelope()"
            )

    @classmethod
    def fixture_nonclaim(
        cls,
        *,
        checkpoint_id: str,
        pair_id: str,
        cluster_id: str,
        direction: str,
        moved_toward_donor: bool,
        recipient_prompt_token_count: int,
        donor_prompt_token_count: int,
        latent_sha256: str,
        answer_sha256: str,
        donor_world_id: str,
        recipient_world_id: str,
    ) -> "SwapOutcomeRecord":
        """Build a permanently non-claim fixture record with separated hashes."""
        return cls(
            checkpoint_id=checkpoint_id,
            pair_id=pair_id,
            cluster_id=cluster_id,
            direction=direction,
            moved_toward_donor=moved_toward_donor,
            recipient_prompt_token_count=recipient_prompt_token_count,
            donor_prompt_token_count=donor_prompt_token_count,
            evidence_namespace="synthetic_fixture_nonclaim",
            world_latent_sha256=latent_sha256,
            decoder_input_sha256=latent_sha256,
            injection_input_sha256=latent_sha256,
            answer_world_latent_sha256=latent_sha256,
            delivered_world_latent_sha256=latent_sha256,
            answer_sha256=answer_sha256,
            envelope_payload_sha256=_hash(
                {"fixture": pair_id, "direction": direction, "latent": latent_sha256}
            ),
            donor_world_id=donor_world_id,
            recipient_world_id=recipient_world_id,
            world_a_id=donor_world_id if direction == "a_to_b" else recipient_world_id,
            world_b_id=recipient_world_id if direction == "a_to_b" else donor_world_id,
            comparison=None,
            _construction_token=_SWAP_RECORD_TOKEN,
        )

    @classmethod
    def from_learned_latent_envelope(
        cls,
        envelope: Mapping[str, Any],
        *,
        latent_tensor_bytes: bytes,
        checkpoint_id: str,
        pair_id: str,
        cluster_id: str,
        recipient_prompt_token_count: int,
        donor_prompt_token_count: int,
        world_a_id: str,
        world_b_id: str,
        recipient_baseline_answer: Mapping[str, Any],
        donor_reference_answer: Mapping[str, Any],
        recipient_target: Mapping[str, Any],
        donor_target: Mapping[str, Any],
        scoring_contract_sha256: str,
        expected_source: str,
        expected_manifest_sha256: str,
    ) -> "SwapOutcomeRecord":
        """Build authentic swap evidence only after shared envelope/z validation."""
        payload = open_result_envelope(
            envelope,
            expected_source=expected_source,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_checkpoint_id=checkpoint_id,
        )
        evidence = validate_learned_latent_evidence(payload)
        verify_latent_tensor_bytes(evidence, latent_tensor_bytes)
        lineage = evidence["swap_lineage"]
        donor_world_id = lineage["donor_world_id"]
        recipient_world_id = lineage["recipient_world_id"]
        if (donor_world_id, recipient_world_id) == (world_a_id, world_b_id):
            direction = "a_to_b"
        elif (donor_world_id, recipient_world_id) == (world_b_id, world_a_id):
            direction = "b_to_a"
        else:
            raise ValueError("authentic swap lineage does not match the declared World A/B pair")
        if (
            lineage["mode"] != "donor_swap"
            or lineage["world_pair_id"] != pair_id
            or lineage["source_world_id"] != donor_world_id
            or lineage["direction"] != f"{donor_world_id}->{recipient_world_id}"
        ):
            raise ValueError("authentic swap envelope donor lineage does not match the requested pair")
        tensor = evidence["tensor"]
        binding = evidence["answer_binding"]
        comparison = _build_swap_comparison(
            direction=direction,
            recipient_baseline_answer=recipient_baseline_answer,
            donor_reference_answer=donor_reference_answer,
            swapped_answer=evidence["answer"],
            recipient_target=recipient_target,
            donor_target=donor_target,
            scoring_contract_sha256=scoring_contract_sha256,
        )
        if comparison.swapped_answer_sha256 != binding["answer_sha256"]:
            raise ValueError("swap comparison is not bound to the sealed answer")
        world_hash = tensor["world_latent_sha256"]
        return cls(
            checkpoint_id=checkpoint_id,
            pair_id=pair_id,
            cluster_id=cluster_id,
            direction=direction,
            moved_toward_donor=comparison.moved_toward_donor,
            recipient_prompt_token_count=recipient_prompt_token_count,
            donor_prompt_token_count=donor_prompt_token_count,
            evidence_namespace="authentic_learned_latent",
            world_latent_sha256=world_hash,
            decoder_input_sha256=tensor["decoder_input_sha256"],
            injection_input_sha256=tensor["injection_input_sha256"],
            answer_world_latent_sha256=binding["world_latent_sha256"],
            delivered_world_latent_sha256=payload["tensor"]["world_latent_sha256"],
            answer_sha256=binding["answer_sha256"],
            envelope_payload_sha256=envelope["payload_sha256"],
            donor_world_id=donor_world_id,
            recipient_world_id=recipient_world_id,
            world_a_id=world_a_id,
            world_b_id=world_b_id,
            comparison=comparison,
            _construction_token=_SWAP_RECORD_TOKEN,
        )


@dataclass(frozen=True)
class InterventionOutcomeRecord:
    checkpoint_id: str
    episode_id: str
    cluster_id: str
    intervention_kind: str
    condition: str
    outcome: bool
    parse_failed: bool
    site_id: str
    intervention_sha256: str


@dataclass(frozen=True)
class MediationArmRecord:
    checkpoint_id: str
    episode_id: str
    cluster_id: str
    stage: str
    mediator: str
    arm: str
    outcome: float
    site_id: str
    intervention_kind: str | None
    intervention_id: str | None
    source_activation_sha256: str | None
    result_sha256: str


@dataclass(frozen=True)
class PromotionAblationRecord:
    checkpoint_id: str
    episode_id: str
    cluster_id: str
    natural_promotion: float
    inadequacy_ablated_promotion: float
    intervention_id: str
    result_sha256: str


@dataclass(frozen=True)
class ComputedCheckpointEvidence:
    identity: CheckpointIdentity
    g3_evaluation: GateDecision
    g5_evaluation: GateDecision
    g6_evaluation: GateDecision
    ood_effect_ci_low: float
    ood_retention: float
    provenance_hash_match_rate: float

    @classmethod
    def from_computed(
        cls,
        *,
        identity: Mapping[str, Any],
        g3_evaluation: GateDecision,
        g5_evaluation: GateDecision,
        g6_evaluation: GateDecision,
        ood_effect_ci_low: float,
        ood_retention: float,
        provenance_hash_match_rate: float,
    ) -> "ComputedCheckpointEvidence":
        for gate, evaluation in (("g3", g3_evaluation), ("g5", g5_evaluation), ("g6", g6_evaluation)):
            if not isinstance(evaluation, GateDecision) or evaluation.gate != gate:
                raise ValueError(f"{gate} must be a computed GateDecision")
            evaluation.verify()
        numeric = (ood_effect_ci_low, ood_retention, provenance_hash_match_rate)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric):
            raise ValueError("checkpoint evidence metrics must be finite")
        return cls(
            identity=CheckpointIdentity.from_dict(dict(identity)),
            g3_evaluation=g3_evaluation,
            g5_evaluation=g5_evaluation,
            g6_evaluation=g6_evaluation,
            ood_effect_ci_low=float(ood_effect_ci_low),
            ood_retention=float(ood_retention),
            provenance_hash_match_rate=float(provenance_hash_match_rate),
        )


def evaluate_checkpoint_gates(
    primary: ComputedCheckpointEvidence | None,
    replication: ComputedCheckpointEvidence | None,
) -> dict[str, dict[str, Any]]:
    """Evaluate G3/G5/G6 plus OOD/replication only from computed evidence."""
    if primary is None:
        reason = "primary checkpoint evidence is missing"
        return {gate: {"passed": False, "reasons": [reason]} for gate in ("g3", "g5", "g6", "g7", "g8")}
    if not isinstance(primary, ComputedCheckpointEvidence):
        raise ValueError("primary checkpoint evidence must be computed from raw records")
    for decision in (primary.g3_evaluation, primary.g5_evaluation, primary.g6_evaluation):
        decision.verify()
    g3_reasons = list(primary.g3_evaluation.reasons)
    g5_reasons = list(primary.g5_evaluation.reasons)
    g6_reasons = list(primary.g6_evaluation.reasons)
    if g3_reasons:
        g6_reasons.append("primary checkpoint did not pass computed G3 prerequisite")
    if g5_reasons:
        g6_reasons.append("primary checkpoint did not pass computed G5 prerequisite")
    g7_reasons = _ood_reasons(primary)
    if g6_reasons:
        g7_reasons.append("primary checkpoint did not pass G6 prerequisite")
    g8_reasons: list[str] = []
    if replication is None:
        g8_reasons.append("second checkpoint evidence is missing")
    elif not isinstance(replication, ComputedCheckpointEvidence):
        raise ValueError("replication checkpoint evidence must be computed from raw records")
    else:
        for decision in (replication.g3_evaluation, replication.g5_evaluation, replication.g6_evaluation):
            decision.verify()
        if not _meaningfully_distinct(primary.identity, replication.identity):
            g8_reasons.append("second checkpoint is not an immutable, independent revision")
        if not replication.g3_evaluation.passed:
            g8_reasons.append("second checkpoint did not pass G3")
        if not replication.g5_evaluation.passed:
            g8_reasons.append("second checkpoint did not pass G5")
        g8_reasons.extend(f"second checkpoint: {reason}" for reason in replication.g6_evaluation.reasons)
        g8_reasons.extend(f"second checkpoint: {reason}" for reason in _ood_reasons(replication))
    if g6_reasons:
        g8_reasons.append("primary checkpoint did not pass G6")
    if g7_reasons:
        g8_reasons.append("primary checkpoint did not pass G7")
    return {
        "g3": {"passed": not g3_reasons, "reasons": g3_reasons},
        "g5": {"passed": not g5_reasons, "reasons": g5_reasons},
        "g6": {"passed": not g6_reasons, "reasons": g6_reasons},
        "g7": {"passed": not g7_reasons, "reasons": g7_reasons},
        "g8": {"passed": not g8_reasons, "reasons": g8_reasons},
    }


def evaluate_g1(records: Sequence[RegimeRecord]) -> GateDecision:
    _unique(records, lambda row: row.episode_id, "G1 episode_id")
    if not records:
        return _decision("g1", records, [GateSubcondition("evidence", False, "G1 evidence is empty")])
    for row in records:
        _nonempty(row.episode_id, "G1 episode_id")
        if not isinstance(row.parsed, bool) or not isinstance(row.joint_correct, bool):
            raise ValueError("G1 parsed/joint_correct must be Boolean raw observations")
    parse_success = sum(row.parsed for row in records) / len(records)
    joint_accuracy = sum(row.parsed and row.joint_correct for row in records) / len(records)
    conditions = [
        GateSubcondition(
            "minimum_records", len(records) >= _MIN_REGIME_RECORDS,
            f"record count {len(records)} {'meets' if len(records) >= _MIN_REGIME_RECORDS else 'is below'} G1 pilot minimum {_MIN_REGIME_RECORDS}",
        ),
        GateSubcondition(
            "joint_accuracy_range", 0.20 <= joint_accuracy <= 0.70,
            f"frozen joint accuracy {joint_accuracy:.6f} must be within [0.20, 0.70]",
        ),
        GateSubcondition(
            "parse_success", parse_success >= 0.95,
            f"parse success {parse_success:.6f} must be >= 0.95",
        ),
    ]
    return _decision("g1", records, conditions)


def evaluate_g3(
    condition_records: Sequence[BehaviorConditionRecord],
    swap_records: Sequence[SwapOutcomeRecord],
    *,
    seed: int = 0,
) -> GateDecision:
    if not condition_records or not swap_records:
        return _decision(
            "g3", [*condition_records, *swap_records],
            [GateSubcondition("evidence", False, "G3 condition and swap evidence are both required")],
        )
    _validate_g3_records(condition_records, swap_records)
    by_episode = _matrix(condition_records, lambda r: r.episode_id, lambda r: r.condition, _G3_CONDITIONS, "G3")
    ordered = sorted(by_episode)
    clusters = [by_episode[key]["E"].cluster_id for key in ordered]
    conditions: list[GateSubcondition] = [
        GateSubcondition(
            "minimum_clusters", len(set(clusters)) >= _MIN_CONFIRMATORY_CLUSTERS,
            f"cluster count {len(set(clusters))} must be >= {_MIN_CONFIRMATORY_CLUSTERS}",
        )
    ]
    for control in ("G", "W", "I", "T1c", "T2c", "C_prime"):
        effect = paired_effect(
            [float(by_episode[key]["E"].joint_correct) for key in ordered],
            [float(by_episode[key][control].joint_correct) for key in ordered],
            cluster_ids=clusters,
            seed=seed,
        )
        threshold = 0.03 if control == "C_prime" else 0.05
        passed = effect["ate"] >= threshold and effect["ci_low"] > 0
        conditions.append(
            GateSubcondition(
                f"E_vs_{control}", passed,
                f"E-{control} ATE {effect['ate']:.6f} (95% CI {effect['ci_low']:.6f}, {effect['ci_high']:.6f}) "
                f"must be >= {threshold:.2f} with lower CI > 0",
            )
        )
    length_failures = [
        episode for episode, rows in by_episode.items()
        if len({row.prompt_token_count for row in rows.values()}) != 1
    ]
    conditions.append(
        GateSubcondition(
            "matched_prompt_lengths", not length_failures,
            "all condition prompt token lengths match per episode" if not length_failures else f"prompt lengths differ for {len(length_failures)} episodes",
        )
    )
    decoding_failures = [
        episode for episode, rows in by_episode.items()
        if len({row.decoding_sha256 for row in rows.values()}) != 1
    ]
    conditions.append(
        GateSubcondition(
            "matched_decoding", not decoding_failures,
            "all decoding configurations match per episode" if not decoding_failures else f"decoding configuration differs for {len(decoding_failures)} episodes",
        )
    )

    successes = [float(record.moved_toward_donor) for record in swap_records]
    swap_effect = paired_effect(
        successes,
        [0.0] * len(successes),
        cluster_ids=[record.cluster_id for record in swap_records],
        seed=seed,
    )
    conditions.append(
        GateSubcondition(
            "swap_success", swap_effect["ate"] >= 0.65 and swap_effect["ci_low"] > 0.50,
            f"swap success {swap_effect['ate']:.6f} (95% CI {swap_effect['ci_low']:.6f}, {swap_effect['ci_high']:.6f}) "
            "must be >= 0.65 with lower CI > 0.50",
        )
    )
    provenance = sum(
        len(
            {
                r.world_latent_sha256,
                r.decoder_input_sha256,
                r.injection_input_sha256,
                r.answer_world_latent_sha256,
                r.delivered_world_latent_sha256,
            }
        )
        == 1
        for r in swap_records
    ) / len(swap_records)
    conditions.append(
        GateSubcondition(
            "provenance", provenance == 1.0,
            f"five-way learned-z provenance match rate {provenance:.6f} must equal 1.0; "
            "answer hashes are validated separately",
        )
    )
    authentic_rate = sum(
        record.evidence_namespace == "authentic_learned_latent" for record in swap_records
    ) / len(swap_records)
    conditions.append(
        GateSubcondition(
            "claim_eligible_swap_evidence",
            authentic_rate == 1.0,
            f"authentic shared learned-latent evidence rate {authentic_rate:.6f} must equal 1.0",
        )
    )
    return _decision("g3", [*condition_records, *swap_records], conditions)


def evaluate_g5(records: Sequence[InterventionOutcomeRecord], *, seed: int = 0) -> GateDecision:
    """Evaluate strict bidirectional evidence, not directional claim language.

    Any one-direction pivot wording remains a reporting-policy decision outside
    this evaluator and cannot alter its required ablation/injection matrix.
    """
    if not records:
        return _decision("g5", records, [GateSubcondition("evidence", False, "G5 evidence is empty")])
    if len({row.checkpoint_id for row in records}) != 1:
        raise ValueError("G5 records must belong to exactly one checkpoint")
    for row in records:
        if row.intervention_kind not in {"ablation", "injection"}:
            raise ValueError("G5 intervention_kind must be ablation or injection")
        if row.condition not in _G5_CONDITIONS:
            raise ValueError(f"G5 unknown control condition: {row.condition}")
        if not isinstance(row.outcome, bool) or not isinstance(row.parse_failed, bool):
            raise ValueError("G5 outcomes must be raw Boolean observations")
        for value, name in ((row.checkpoint_id, "checkpoint_id"), (row.episode_id, "episode_id"), (row.cluster_id, "cluster_id"), (row.site_id, "site_id")):
            _nonempty(value, f"G5 {name}")
        _digest(row.intervention_sha256, "G5 intervention_sha256")
    _unique(records, lambda r: (r.checkpoint_id, r.episode_id, r.intervention_kind, r.condition), "G5 node")
    if len({row.site_id for row in records}) != 1:
        raise ValueError("G5 records must use one manifest-fixed intervention site")
    conditions: list[GateSubcondition] = []
    for kind in ("ablation", "injection"):
        kind_rows = [row for row in records if row.intervention_kind == kind]
        matrix = _matrix(kind_rows, lambda r: r.episode_id, lambda r: r.condition, _G5_CONDITIONS, f"G5 {kind}")
        ordered = sorted(matrix)
        clusters = [matrix[key]["target"].cluster_id for key in ordered]
        conditions.append(
            GateSubcondition(
                f"{kind}_minimum_clusters", len(set(clusters)) >= _MIN_CONFIRMATORY_CLUSTERS,
                f"{kind} cluster count {len(set(clusters))} must be >= {_MIN_CONFIRMATORY_CLUSTERS}",
            )
        )
        baseline = paired_effect(
            [float(matrix[key]["target"].outcome) for key in ordered],
            [float(matrix[key]["baseline"].outcome) for key in ordered],
            cluster_ids=clusters, seed=seed,
        )
        conditions.append(
            GateSubcondition(
                f"{kind}_effect", baseline["ate"] >= 0.05 and baseline["ci_low"] > 0,
                f"{kind} ATE {baseline['ate']:.6f} (95% CI {baseline['ci_low']:.6f}, {baseline['ci_high']:.6f}) "
                "must be >= 0.05 with lower CI > 0",
            )
        )
        for control in _G5_CONDITIONS[2:]:
            effect = paired_effect(
                [float(matrix[key]["target"].outcome) for key in ordered],
                [float(matrix[key][control].outcome) for key in ordered],
                cluster_ids=clusters, seed=seed,
            )
            conditions.append(
                GateSubcondition(
                    f"{kind}_beats_{control}", effect["ci_low"] > 0,
                    f"{kind} target-{control} ATE {effect['ate']:.6f} "
                    f"(95% CI {effect['ci_low']:.6f}, {effect['ci_high']:.6f}) must have lower CI > 0",
                )
            )
        parse_delta = sum(
            float(matrix[key]["target"].parse_failed) - float(matrix[key]["baseline"].parse_failed)
            for key in ordered
        ) / len(ordered)
        conditions.append(
            GateSubcondition(
                f"{kind}_parse_failure", parse_delta < 0.02,
                f"{kind} added parse-failure rate {parse_delta:.6f} must be < 0.02",
            )
        )
    return _decision("g5", records, conditions)


def evaluate_g6(
    records: Sequence[MediationArmRecord],
    promotion_ablation: Sequence[PromotionAblationRecord],
    *,
    seed: int = 0,
) -> GateDecision:
    """Compute G6 only from explicit clamp/patch potential-outcome arms."""
    if not records or not promotion_ablation:
        return _decision(
            "g6", [*records, *promotion_ablation],
            [GateSubcondition("interventions", False, "G6 requires clamp/patch arms and inadequacy-ablation chain evidence")],
        )
    _validate_mediation_records(records, promotion_ablation)
    conditions: list[GateSubcondition] = []
    proportions: list[float] = []
    primary_te: float | None = None
    for stage in _STAGES:
        stage_rows = [row for row in records if row.stage == stage]
        matrix = _matrix(
            stage_rows, lambda r: r.episode_id, lambda r: f"{r.mediator}:{r.arm}",
            tuple(f"{mediator}:{arm}" for mediator in _MEDIATORS for arm in _ARMS), f"G6 {stage}",
        )
        ordered = sorted(matrix)
        clusters = [matrix[key]["primary:treated_natural"].cluster_id for key in ordered]
        conditions.append(
            GateSubcondition(
                f"{stage}_minimum_clusters", len(set(clusters)) >= _MIN_CONFIRMATORY_CLUSTERS,
                f"{stage} cluster count {len(set(clusters))} must be >= {_MIN_CONFIRMATORY_CLUSTERS}",
            )
        )
        treated = [matrix[key]["primary:treated_natural"].outcome for key in ordered]
        control = [matrix[key]["primary:control_natural"].outcome for key in ordered]
        clamped = [matrix[key]["primary:treated_control_clamp"].outcome for key in ordered]
        te = paired_effect(treated, control, cluster_ids=clusters, seed=seed)
        nde = paired_effect(clamped, control, cluster_ids=clusters, seed=seed)
        nie = paired_effect(treated, clamped, cluster_ids=clusters, seed=seed)
        if primary_te is None:
            primary_te = float(te["ate"])
        conditions.append(
            GateSubcondition(
                f"{stage}_te", te["ci_low"] > 0,
                f"{stage} TE {te['ate']:.6f} (95% CI {te['ci_low']:.6f}, {te['ci_high']:.6f}) must have lower CI > 0",
            )
        )
        conditions.append(
            GateSubcondition(
                f"{stage}_nie", nie["ci_low"] > 0,
                f"{stage} clamp/patch NIE {nie['ate']:.6f} (95% CI {nie['ci_low']:.6f}, {nie['ci_high']:.6f}) "
                f"must have lower CI > 0; NDE={nde['ate']:.6f}",
            )
        )
        proportions.append(float(nie["ate"]) / float(te["ate"]) if te["ate"] > 0 and te["ci_low"] > 0 else float("-inf"))
        for mediator in _MEDIATORS[1:]:
            control_nie = paired_effect(
                [matrix[key][f"{mediator}:treated_natural"].outcome for key in ordered],
                [matrix[key][f"{mediator}:treated_control_clamp"].outcome for key in ordered],
                cluster_ids=clusters, seed=seed,
            )
            conditions.append(
                GateSubcondition(
                    f"{stage}_{mediator}_specificity",
                    abs(float(control_nie["ate"])) < abs(float(nie["ate"])),
                    f"{stage} {mediator} |NIE| {abs(float(control_nie['ate'])):.6f} must be smaller than primary "
                    f"|NIE| {abs(float(nie['ate'])):.6f}",
                )
            )
    mediated = min(proportions)
    conditions.append(
        GateSubcondition(
            "mediated_proportion", math.isfinite(mediated) and mediated >= 0.20,
            f"minimum ordered mediated proportion {mediated:.6f} must be >= 0.20",
        )
    )
    ordered_ablation = sorted(promotion_ablation, key=lambda row: row.episode_id)
    ablation_effect = paired_effect(
        [row.natural_promotion for row in ordered_ablation],
        [row.inadequacy_ablated_promotion for row in ordered_ablation],
        cluster_ids=[row.cluster_id for row in ordered_ablation], seed=seed,
    )
    conditions.append(
        GateSubcondition(
            "inadequacy_ablation_reduces_promotion", ablation_effect["ci_low"] > 0,
            f"promotion reduction after inadequacy ablation {ablation_effect['ate']:.6f} "
            f"(95% CI {ablation_effect['ci_low']:.6f}, {ablation_effect['ci_high']:.6f}) must have lower CI > 0",
        )
    )
    return _decision("g6", [*records, *promotion_ablation], conditions)


def _validate_g3_records(condition_records: Sequence[BehaviorConditionRecord], swap_records: Sequence[SwapOutcomeRecord]) -> None:
    checkpoint_ids = {row.checkpoint_id for row in condition_records} | {row.checkpoint_id for row in swap_records}
    if len(checkpoint_ids) != 1:
        raise ValueError("G3 records must belong to exactly one checkpoint")
    for row in condition_records:
        if row.condition not in _G3_CONDITIONS:
            raise ValueError(f"G3 unknown condition: {row.condition}")
        if not isinstance(row.joint_correct, bool) or isinstance(row.prompt_token_count, bool) or row.prompt_token_count <= 0:
            raise ValueError("G3 joint_correct must be Boolean and prompt_token_count positive")
        for value, name in ((row.checkpoint_id, "checkpoint_id"), (row.episode_id, "episode_id"), (row.cluster_id, "cluster_id")):
            _nonempty(value, f"G3 {name}")
        _digest(row.decoding_sha256, "G3 decoding_sha256")
    _unique(condition_records, lambda r: (r.checkpoint_id, r.episode_id, r.condition), "G3 condition")
    for row in swap_records:
        if row.direction not in {"a_to_b", "b_to_a"} or not isinstance(row.moved_toward_donor, bool):
            raise ValueError("G3 swaps require a_to_b/b_to_a raw Boolean outcomes")
        if (
            isinstance(row.recipient_prompt_token_count, bool)
            or isinstance(row.donor_prompt_token_count, bool)
            or not isinstance(row.recipient_prompt_token_count, int)
            or not isinstance(row.donor_prompt_token_count, int)
            or row.recipient_prompt_token_count <= 0
            or row.recipient_prompt_token_count != row.donor_prompt_token_count
        ):
            raise ValueError("G3 donor/recipient swap prompt lengths must be equal positive integers")
        if row.evidence_namespace not in {
            "authentic_learned_latent",
            "synthetic_fixture_nonclaim",
        }:
            raise ValueError("G3 swap evidence namespace is unknown")
        if row.donor_world_id == row.recipient_world_id:
            raise ValueError("G3 swap donor and recipient worlds must be distinct")
        expected_direction = (
            "a_to_b"
            if (row.donor_world_id, row.recipient_world_id) == (row.world_a_id, row.world_b_id)
            else "b_to_a"
            if (row.donor_world_id, row.recipient_world_id) == (row.world_b_id, row.world_a_id)
            else None
        )
        if expected_direction != row.direction:
            raise ValueError("G3 swap direction does not match donor/recipient World A/B lineage")
        if row.evidence_namespace == "authentic_learned_latent":
            if not isinstance(row.comparison, SwapComparisonBinding):
                raise ValueError("authentic G3 swaps require a locked scoring comparison")
            row.comparison.verify()
            if (
                row.comparison.direction != row.direction
                or row.comparison.swapped_answer_sha256 != row.answer_sha256
                or row.comparison.moved_toward_donor != row.moved_toward_donor
            ):
                raise ValueError("authentic G3 outcome is not bound to the sealed answer and lineage")
        elif row.comparison is not None:
            raise ValueError("synthetic fixture swaps cannot masquerade as authentic comparisons")
        for field in (
            "world_latent_sha256",
            "decoder_input_sha256",
            "injection_input_sha256",
            "answer_world_latent_sha256",
            "delivered_world_latent_sha256",
            "answer_sha256",
            "envelope_payload_sha256",
        ):
            _digest(getattr(row, field), f"G3 {field}")
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in swap_records:
        grouped.setdefault((row.checkpoint_id, row.pair_id), set()).add(row.direction)
    if any(directions != {"a_to_b", "b_to_a"} for directions in grouped.values()):
        raise ValueError("G3 swaps require both directions for every pair")
    authentic_envelopes: dict[tuple[str, str], set[str]] = {}
    for row in swap_records:
        if row.evidence_namespace == "authentic_learned_latent":
            authentic_envelopes.setdefault((row.checkpoint_id, row.pair_id), set()).add(
                row.envelope_payload_sha256
            )
    if any(len(hashes) != 2 for hashes in authentic_envelopes.values()):
        raise ValueError("authentic G3 pairs require two distinct mirrored sealed envelopes")
    _unique(swap_records, lambda r: (r.checkpoint_id, r.pair_id, r.direction), "G3 swap direction")


def _build_swap_comparison(
    *,
    direction: str,
    recipient_baseline_answer: Mapping[str, Any],
    donor_reference_answer: Mapping[str, Any],
    swapped_answer: Mapping[str, Any],
    recipient_target: Mapping[str, Any],
    donor_target: Mapping[str, Any],
    scoring_contract_sha256: str,
) -> SwapComparisonBinding:
    if scoring_contract_sha256 != SWAP_SCORING_CONTRACT_SHA256:
        raise ValueError("swap scoring contract hash does not match the locked scorer")
    try:
        answers = {
            "recipient_baseline": dict(recipient_baseline_answer),
            "donor_reference": dict(donor_reference_answer),
            "swapped": dict(swapped_answer),
        }
        targets = {"recipient": dict(recipient_target), "donor": dict(donor_target)}
    except (TypeError, ValueError) as exc:
        raise ValueError("swap comparison answers and targets must be mappings") from exc
    recipient_baseline_scores = score_episode(answers["recipient_baseline"], targets["recipient"])
    donor_reference_scores = score_episode(answers["donor_reference"], targets["donor"])
    if (
        recipient_baseline_scores["joint_theory_accuracy"] != 1.0
        or donor_reference_scores["joint_theory_accuracy"] != 1.0
    ):
        raise ValueError("swap reference answers must exactly score against their own targets")
    swapped_recipient = score_episode(answers["swapped"], targets["recipient"])
    swapped_donor = score_episode(answers["swapped"], targets["donor"])
    recipient_score = (
        swapped_recipient["joint_theory_accuracy"]
        + swapped_recipient["adequacy_correct"]
    ) / 2.0
    donor_score = (
        swapped_donor["joint_theory_accuracy"] + swapped_donor["adequacy_correct"]
    ) / 2.0
    body = {
        "schema_version": "jump.swap-comparison/v1",
        "direction": direction,
        "recipient_baseline_answer_sha256": hashlib.sha256(
            canonical_json(answers["recipient_baseline"])
        ).hexdigest(),
        "donor_reference_answer_sha256": hashlib.sha256(
            canonical_json(answers["donor_reference"])
        ).hexdigest(),
        "swapped_answer_sha256": hashlib.sha256(
            canonical_json(answers["swapped"])
        ).hexdigest(),
        "recipient_target_sha256": hashlib.sha256(
            canonical_json(targets["recipient"])
        ).hexdigest(),
        "donor_target_sha256": hashlib.sha256(canonical_json(targets["donor"])).hexdigest(),
        "scoring_contract_sha256": scoring_contract_sha256,
        "recipient_score": recipient_score,
        "donor_score": donor_score,
        "moved_toward_donor": donor_score > recipient_score,
    }
    result = SwapComparisonBinding(
        **{key: value for key, value in body.items() if key != "schema_version"},
        content_sha256=_hash(body),
        _construction_token=_SWAP_COMPARISON_TOKEN,
    )
    result.verify()
    return result


def _ood_reasons(evidence: ComputedCheckpointEvidence) -> list[str]:
    reasons = []
    if evidence.ood_effect_ci_low <= 0:
        reasons.append("OOD causal-effect lower CI is not positive")
    if evidence.ood_retention < 0.50:
        reasons.append("OOD retention is below 50%")
    if evidence.provenance_hash_match_rate != 1.0:
        reasons.append("OOD provenance is not 100%")
    return reasons


def _meaningfully_distinct(first: CheckpointIdentity, second: CheckpointIdentity) -> bool:
    return (
        first.checkpoint_id != second.checkpoint_id
        and first.model_revision != second.model_revision
        and first.training_lineage_id != second.training_lineage_id
        and first.checkpoint_sha256 != second.checkpoint_sha256
    )


def _validate_mediation_records(records: Sequence[MediationArmRecord], ablations: Sequence[PromotionAblationRecord]) -> None:
    checkpoint_ids = {row.checkpoint_id for row in records} | {row.checkpoint_id for row in ablations}
    if len(checkpoint_ids) != 1:
        raise ValueError("G6 records must belong to exactly one checkpoint")
    for row in records:
        if row.stage not in _STAGES or row.mediator not in _MEDIATORS or row.arm not in _ARMS:
            raise ValueError("G6 record has an unknown stage, mediator, or potential-outcome arm")
        if not math.isfinite(row.outcome):
            raise ValueError("G6 outcomes must be finite")
        for value, name in ((row.checkpoint_id, "checkpoint_id"), (row.episode_id, "episode_id"), (row.cluster_id, "cluster_id"), (row.site_id, "site_id")):
            _nonempty(value, f"G6 {name}")
        _digest(row.result_sha256, "G6 result_sha256")
        if row.arm == "treated_control_clamp":
            if row.intervention_kind not in {"activation_clamp", "activation_patch"}:
                raise ValueError(
                    "G6 treated_control_clamp requires a typed activation_clamp/activation_patch intervention"
                )
            _nonempty(row.intervention_id, "G6 clamp intervention_id")
            _digest(row.source_activation_sha256, "G6 clamp source_activation_sha256")
        elif (
            row.intervention_kind is not None
            or row.intervention_id is not None
            or row.source_activation_sha256 is not None
        ):
            raise ValueError("G6 natural arms cannot masquerade as clamp interventions")
    _unique(records, lambda r: (r.checkpoint_id, r.episode_id, r.stage, r.mediator, r.arm), "G6 arm")
    for stage in _STAGES:
        if len({row.site_id for row in records if row.stage == stage}) != 1:
            raise ValueError(f"G6 {stage} records must use one manifest-fixed site")
    for row in ablations:
        _nonempty(row.intervention_id, "G6 ablation intervention_id")
        _digest(row.result_sha256, "G6 ablation result_sha256")
        if not math.isfinite(row.natural_promotion) or not math.isfinite(row.inadequacy_ablated_promotion):
            raise ValueError("G6 promotion outcomes must be finite")
    _unique(ablations, lambda r: (r.checkpoint_id, r.episode_id), "G6 promotion ablation")


def _matrix(rows: Sequence[Any], row_key: Any, column_key: Any, required: tuple[str, ...], name: str) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for row in rows:
        matrix.setdefault(row_key(row), {})[column_key(row)] = row
    for key, columns in matrix.items():
        if set(columns) != set(required):
            raise ValueError(f"{name} evidence for {key!r} must contain exactly {list(required)}")
        clusters = {row.cluster_id for row in columns.values()}
        checkpoints = {row.checkpoint_id for row in columns.values()}
        if len(clusters) != 1 or len(checkpoints) != 1:
            raise ValueError(f"{name} evidence identity drift for {key!r}")
    return matrix


def _decision(gate: str, records: Iterable[Any], conditions: list[GateSubcondition]) -> GateDecision:
    evidence_sha = _hash([asdict(record) for record in records])
    passed = all(condition.passed for condition in conditions)
    payload = {
        "gate": gate, "passed": passed,
        "subconditions": [asdict(item) for item in conditions],
        "evidence_sha256": evidence_sha,
    }
    result = GateDecision(
        gate,
        passed,
        tuple(conditions),
        evidence_sha,
        _hash(payload),
        _construction_token=_GATE_DECISION_TOKEN,
    )
    result.verify()
    return result


def _unique(rows: Sequence[Any], key: Any, name: str) -> None:
    values = [key(row) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name} evidence")


def _digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != _REQUIRED_HASH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
