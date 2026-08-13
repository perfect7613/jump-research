import copy
import hashlib
import struct
import unittest

from jump_contracts import (
    EvidenceError,
    build_learned_latent_evidence,
    learned_decoder_identity,
    seal_learned_latent_result,
)
from jump_mechanistic.gates import (
    BehaviorConditionRecord, GateDecision, InterventionOutcomeRecord, MediationArmRecord,
    PromotionAblationRecord, RegimeRecord, SwapOutcomeRecord,
    SWAP_SCORING_CONTRACT_SHA256, evaluate_g1, evaluate_g3, evaluate_g5, evaluate_g6,
)
from jump_mechanistic.metrics import ConfirmatoryEvidence, evaluate_confirmatory_gates, mediation_analysis


def digest(label): return hashlib.sha256(label.encode()).hexdigest()


def clusters(n=40): return [f"world-{i}" for i in range(n)]


def answer(adequate):
    return {
        "partition": [0, 0, 0, 1, 1, 1] if adequate else [0, 0, 1, 0, 1, 1],
        "replacement_law": {
            "same": "attract" if adequate else "repel",
            "different": "repel" if adequate else "attract",
            "exponent": 2 if adequate else 3,
        },
        "adequate": adequate,
    }


def learned_swap_material(i, direction):
    donor = f"world-{i}-a" if direction == "a_to_b" else f"world-{i}-b"
    recipient = f"world-{i}-b" if direction == "a_to_b" else f"world-{i}-a"
    raw = struct.pack("<4f", float(i), 1.0 if direction == "a_to_b" else -1.0, 0.5, 2.0)
    evidence = build_learned_latent_evidence(
        encoder_output=raw,
        decoder_input=bytes(raw),
        injection_input=memoryview(raw),
        encoder_observation=f"observation-{i}".encode(),
        encoder_observation_artifact_name=f"observation-{i}.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[1, 4],
        order="C",
        tensor_artifact_name=f"latent-{i}-{direction}.bin",
        recipient_world_id=recipient,
        donor_world_id=donor,
        world_pair_id=f"p-{i}",
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder.safetensors",
            artifact_sha256=digest("decoder"),
            training_manifest_sha256=digest("decoder-training"),
            code_version="decoder-code",
            architecture="latent-observation-v1",
        ),
        decoded_image=f"decoded-{i}".encode(),
        decoded_image_media_type="image/png",
        answer=answer(direction == "a_to_b"),
    )
    source = "cached" if i % 2 == 0 else "live"
    envelope = seal_learned_latent_result(
        evidence,
        source=source,
        manifest_sha256=digest("manifest"),
        run_id=f"swap-{i}-{direction}",
        code_version="test-code",
        checkpoint_id="primary",
    )
    return envelope, raw, donor, recipient, source


def authentic_swap(i, cluster, direction):
    envelope, raw, donor, recipient, source = learned_swap_material(i, direction)
    world_a, world_b = f"world-{i}-a", f"world-{i}-b"
    recipient_answer = answer(recipient == world_a)
    donor_answer = answer(donor == world_a)
    return SwapOutcomeRecord.from_learned_latent_envelope(
        envelope,
        latent_tensor_bytes=raw,
        checkpoint_id="primary",
        pair_id=f"p-{i}",
        cluster_id=cluster,
        recipient_prompt_token_count=64,
        donor_prompt_token_count=64,
        world_a_id=world_a,
        world_b_id=world_b,
        recipient_baseline_answer=recipient_answer,
        donor_reference_answer=donor_answer,
        recipient_target=recipient_answer,
        donor_target=donor_answer,
        scoring_contract_sha256=SWAP_SCORING_CONTRACT_SHA256,
        expected_source=source,
        expected_manifest_sha256=digest("manifest"),
    )


def g3_records(n=40):
    rows = []
    conditions = ("E", "G", "W", "I", "C_prime", "T1c", "T2c")
    for i, cluster in enumerate(clusters(n)):
        for condition in conditions:
            # E always succeeds; controls alternate, yielding a large paired effect.
            rows.append(BehaviorConditionRecord("primary", f"e-{i}", cluster, condition,
                                                condition == "E" or (condition != "E" and i % 4 == 0),
                                                64, digest("decoding")))
    swaps = []
    for i, cluster in enumerate(clusters(n)):
        for direction in ("a_to_b", "b_to_a"):
            swaps.append(authentic_swap(i, cluster, direction))
    return rows, swaps


def g5_records(n=40):
    rows = []
    controls = ("target", "baseline", "matched_norm", "orthogonal", "generic_error", "sham", "prompt_length")
    for kind in ("ablation", "injection"):
        for i, cluster in enumerate(clusters(n)):
            for condition in controls:
                outcome = condition == "target" or (condition != "target" and i % 4 == 0)
                rows.append(InterventionOutcomeRecord("primary", f"e-{i}", cluster, kind, condition,
                                                      outcome, False, "L7/T3", digest(f"{kind}-{condition}-{i}")))
    return rows


def g6_records(n=40):
    rows = []
    mediators = ("primary", "matched_norm", "orthogonal", "generic_error", "sham", "prompt_length")
    for stage in ("inadequacy", "promotion"):
        for i, cluster in enumerate(clusters(n)):
            for mediator in mediators:
                for arm, outcome in (("control_natural", 0.0), ("treated_natural", 1.0),
                                     ("treated_control_clamp", 0.2 if mediator == "primary" else 0.9)):
                    clamp = arm == "treated_control_clamp"
                    rows.append(MediationArmRecord("primary", f"e-{i}", cluster, stage, mediator, arm,
                                                   outcome, f"L7/{stage}",
                                                   "activation_clamp" if clamp else None,
                                                   f"clamp-{i}" if clamp else None,
                                                   digest(f"activation-{stage}-{mediator}-{i}") if clamp else None,
                                                   digest(f"result-{stage}-{mediator}-{arm}-{i}")))
    ablations = [PromotionAblationRecord("primary", f"e-{i}", cluster, 1.0, 0.2,
                                         f"ablate-{i}", digest(f"ablation-{i}"))
                 for i, cluster in enumerate(clusters(n))]
    return rows, ablations


class GateTests(unittest.TestCase):
    def test_g1_is_computed_and_partial_evidence_fails(self):
        records = [RegimeRecord(f"e-{i}", True, i < 100) for i in range(200)]
        result = evaluate_g1(records)
        self.assertTrue(result.passed)
        self.assertTrue(all(item.detail for item in result.subconditions))
        self.assertFalse(evaluate_g1(records[:10]).passed)
        corrupt = list(records) + [records[0]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_g1(corrupt)

    def test_g3_computes_every_comparison_provenance_and_sham_independent(self):
        conditions, swaps = g3_records()
        result = evaluate_g3(conditions, swaps, seed=17)
        self.assertTrue(result.passed, result.reasons)
        bad = list(swaps)
        bad[0] = copy.copy(bad[0])
        object.__setattr__(bad[0], "delivered_world_latent_sha256", digest("corruption"))
        failed = evaluate_g3(conditions, bad, seed=17)
        self.assertFalse(failed.passed)
        self.assertIn("provenance", [c.name for c in failed.subconditions if not c.passed])
        self.assertNotEqual(swaps[0].answer_sha256, swaps[0].world_latent_sha256)
        with self.assertRaisesRegex(ValueError, "exactly"):
            evaluate_g3(conditions[:-1], swaps, seed=17)

    def test_authentic_swaps_require_shared_envelope_tensor_and_donor_validation(self):
        record = authentic_swap(0, "world-0", "a_to_b")
        self.assertEqual(record.evidence_namespace, "authentic_learned_latent")
        envelope, raw, donor, recipient, source = learned_swap_material(0, "a_to_b")
        with self.assertRaisesRegex(EvidenceError, "raw latent tensor artifact hash mismatch"):
            SwapOutcomeRecord.from_learned_latent_envelope(
                envelope, latent_tensor_bytes=b"0" * len(raw), checkpoint_id="primary",
                pair_id="p-0", cluster_id="world-0", recipient_prompt_token_count=64,
                donor_prompt_token_count=64, world_a_id="world-0-a", world_b_id="world-0-b",
                recipient_baseline_answer=answer(False), donor_reference_answer=answer(True),
                recipient_target=answer(False), donor_target=answer(True),
                scoring_contract_sha256=SWAP_SCORING_CONTRACT_SHA256, expected_source=source,
                expected_manifest_sha256=digest("manifest"),
            )
        with self.assertRaisesRegex(ValueError, "World A/B pair"):
            SwapOutcomeRecord.from_learned_latent_envelope(
                envelope, latent_tensor_bytes=raw, checkpoint_id="primary",
                pair_id="p-0", cluster_id="world-0", recipient_prompt_token_count=64,
                donor_prompt_token_count=64, world_a_id="wrong-donor", world_b_id=recipient,
                recipient_baseline_answer=answer(False), donor_reference_answer=answer(True),
                recipient_target=answer(False), donor_target=answer(True),
                scoring_contract_sha256=SWAP_SCORING_CONTRACT_SHA256, expected_source=source,
                expected_manifest_sha256=digest("manifest"),
            )
        with self.assertRaisesRegex(ValueError, "must be built through"):
            SwapOutcomeRecord(
                checkpoint_id="primary", pair_id="p", cluster_id="c", direction="a_to_b",
                moved_toward_donor=True, recipient_prompt_token_count=1,
                donor_prompt_token_count=1, evidence_namespace="authentic_learned_latent",
                world_latent_sha256=digest("z"), decoder_input_sha256=digest("z"),
                injection_input_sha256=digest("z"), answer_world_latent_sha256=digest("z"),
                delivered_world_latent_sha256=digest("z"), answer_sha256=digest("answer"),
                envelope_payload_sha256=digest("payload"), donor_world_id="a",
                recipient_world_id="b", world_a_id="a", world_b_id="b",
                comparison=None, _construction_token=None,
            )

    def test_authentic_swaps_reject_duplicate_envelope_flipped_label_and_outcome_spoof(self):
        first = authentic_swap(0, "world-0", "a_to_b")
        duplicated = copy.copy(first)
        object.__setattr__(duplicated, "direction", "b_to_a")
        object.__setattr__(duplicated, "donor_world_id", first.recipient_world_id)
        object.__setattr__(duplicated, "recipient_world_id", first.donor_world_id)
        with self.assertRaisesRegex(ValueError, "sealed answer and lineage|distinct mirrored"):
            evaluate_g3(g3_records()[0], [first, duplicated], seed=17)

        flipped = copy.copy(first)
        object.__setattr__(flipped, "direction", "b_to_a")
        with self.assertRaisesRegex(ValueError, "direction does not match"):
            evaluate_g3(g3_records()[0], [flipped], seed=17)

        spoofed = copy.copy(first)
        object.__setattr__(spoofed, "moved_toward_donor", False)
        with self.assertRaisesRegex(ValueError, "sealed answer and lineage"):
            evaluate_g3(g3_records()[0], [spoofed], seed=17)

        envelope, raw, _donor, _recipient, source = learned_swap_material(0, "a_to_b")
        with self.assertRaisesRegex(ValueError, "scoring contract hash"):
            SwapOutcomeRecord.from_learned_latent_envelope(
                envelope, latent_tensor_bytes=raw, checkpoint_id="primary", pair_id="p-0",
                cluster_id="world-0", recipient_prompt_token_count=64,
                donor_prompt_token_count=64, world_a_id="world-0-a", world_b_id="world-0-b",
                recipient_baseline_answer=answer(False), donor_reference_answer=answer(True),
                recipient_target=answer(False), donor_target=answer(True),
                scoring_contract_sha256=digest("spoofed-scorer"), expected_source=source,
                expected_manifest_sha256=digest("manifest"),
            )

    def test_authentic_swaps_reject_distinct_envelopes_with_non_mirrored_worlds(self):
        conditions, swaps = g3_records()
        reverse_index = next(
            index
            for index, row in enumerate(swaps)
            if row.pair_id == "p-0" and row.direction == "b_to_a"
        )
        unrelated = copy.copy(swaps[reverse_index])
        object.__setattr__(unrelated, "world_a_id", "unrelated-a")
        object.__setattr__(unrelated, "world_b_id", "unrelated-b")
        object.__setattr__(unrelated, "donor_world_id", "unrelated-b")
        object.__setattr__(unrelated, "recipient_world_id", "unrelated-a")
        corrupted = list(swaps)
        corrupted[reverse_index] = unrelated
        with self.assertRaisesRegex(ValueError, "one ordered World A/B identity"):
            evaluate_g3(conditions, corrupted, seed=17)

    def test_g5_requires_all_controls_both_directions_and_sham(self):
        records = g5_records()
        result = evaluate_g5(records, seed=17)
        self.assertTrue(result.passed, result.reasons)
        missing_sham = [r for r in records if not (r.intervention_kind == "ablation" and r.condition == "sham")]
        with self.assertRaisesRegex(ValueError, "exactly"):
            evaluate_g5(missing_sham, seed=17)
        drift = list(records)
        drift[0] = copy.copy(drift[0]); object.__setattr__(drift[0], "intervention_sha256", "x")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evaluate_g5(drift, seed=17)

    def test_g6_requires_typed_interventions_and_ols_is_ineligible(self):
        records, ablations = g6_records()
        result = evaluate_g6(records, ablations, seed=17)
        self.assertTrue(result.passed, result.reasons)
        missing = [r for r in records if not (r.stage == "promotion" and r.mediator == "primary" and r.arm == "treated_control_clamp")]
        with self.assertRaisesRegex(ValueError, "exactly"):
            evaluate_g6(missing, ablations, seed=17)
        fake = copy.copy(records[2]); object.__setattr__(fake, "intervention_id", None)
        broken = list(records); broken[2] = fake
        with self.assertRaisesRegex(ValueError, "intervention_id"):
            evaluate_g6(broken, ablations, seed=17)
        untyped = copy.copy(records[2])
        object.__setattr__(untyped, "intervention_kind", "observational_ols")
        broken = list(records); broken[2] = untyped
        with self.assertRaisesRegex(ValueError, "typed activation_clamp/activation_patch"):
            evaluate_g6(broken, ablations, seed=17)
        descriptive = mediation_analysis(
            [i % 2 for i in range(40)],
            [(i % 2) + (i % 7) * 0.03 for i in range(40)],
            [1.2 * (i % 2) + (i % 5) * 0.02 for i in range(40)],
            cluster_ids=[f"c-{i // 2}" for i in range(40)], seed=17,
        )
        self.assertEqual(descriptive["estimator_type"], "observational_ols_descriptive_only")
        with self.assertRaisesRegex(ValueError, "trusted confirmatory"):
            ConfirmatoryEvidence.from_dict({"g3_passed": True})

    def test_gate_decision_hash_is_immutable(self):
        result = evaluate_g5(g5_records(), seed=17)
        tampered = copy.copy(result)
        object.__setattr__(tampered, "passed", False)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            tampered.verify()
        with self.assertRaisesRegex(ValueError, "computed from raw"):
            GateDecision("g5", True, (), digest("evidence"), digest("calculation"), None)

    def test_single_checkpoint_alias_cannot_satisfy_replication(self):
        g3_conditions, swaps = g3_records()
        computed_g3 = evaluate_g3(g3_conditions, swaps, seed=17)
        computed_g5 = evaluate_g5(g5_records(), seed=17)
        arms, ablations = g6_records()
        computed_g6 = evaluate_g6(arms, ablations, seed=17)
        identity = {
            "checkpoint_id": "primary", "model_revision": "rev-a",
            "tokenizer_revision": "tok-a", "training_lineage_id": "lineage-a",
            "checkpoint_sha256": digest("checkpoint-a"),
        }
        primary = ConfirmatoryEvidence.from_computed(
            identity=identity, g3_evaluation=computed_g3, g5_evaluation=computed_g5,
            g6_evaluation=computed_g6, ood_effect_ci_low=0.1, ood_retention=0.6,
            provenance_hash_match_rate=1.0,
        )
        alias = ConfirmatoryEvidence.from_computed(
            identity={**identity, "checkpoint_id": "alias"},
            g3_evaluation=computed_g3, g5_evaluation=computed_g5, g6_evaluation=computed_g6,
            ood_effect_ci_low=0.1, ood_retention=0.6, provenance_hash_match_rate=1.0,
        )
        gates = evaluate_confirmatory_gates(primary, alias)
        self.assertTrue(gates["g6"]["passed"])
        self.assertFalse(gates["g8"]["passed"])
        self.assertTrue(any("not an immutable, independent revision" in reason for reason in gates["g8"]["reasons"]))


if __name__ == "__main__": unittest.main()
