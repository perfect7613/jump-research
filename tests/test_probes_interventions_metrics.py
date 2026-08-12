import math
import unittest

from jump_mechanistic.interventions import (
    MatchedWorldPair,
    ablate,
    build_control_directions,
    evaluate_latent_swap,
    generic_error_control,
    inject,
)
from jump_mechanistic.metrics import (
    BOOTSTRAP_RESAMPLES,
    ConfirmatoryEvidence,
    evaluate_confirmatory_gates,
    mediation_analysis,
    paired_effect,
)
from jump_mechanistic.probes import (
    ProbeSample,
    heldout_probe_evaluation,
    ood_law_family_evaluation,
)
from jump_mechanistic.synthetic import load_fixture
from jump_mechanistic.vectors import dot, norm


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.samples = [ProbeSample(**row) for row in load_fixture()["probe_samples"]]

    def test_heldout_groups_do_not_leak(self):
        evaluation = heldout_probe_evaluation(self.samples, seed=17)
        self.assertTrue(set(evaluation["train_groups"]).isdisjoint(evaluation["test_groups"]))
        self.assertGreater(evaluation["metrics"]["auc"], 0.95)
        self.assertIn("balanced_accuracy", evaluation["metrics"])
        self.assertNotIn("accuracy", evaluation["metrics"])

    def test_ood_law_family(self):
        result = ood_law_family_evaluation(self.samples, heldout_family="spring")
        self.assertNotIn("spring", result["train_families"])
        self.assertTrue(set(result["train_groups"]).isdisjoint(result["test_groups"]))
        self.assertGreater(result["metrics"]["auc"], 0.95)

    def test_duplicate_sample_and_cross_family_group_leakage_are_rejected(self):
        duplicate = [*self.samples, self.samples[0]]
        with self.assertRaisesRegex(ValueError, "sample_id"):
            ood_law_family_evaluation(duplicate, heldout_family="spring")
        leaked = list(self.samples)
        spring_index = next(i for i, sample in enumerate(leaked) if sample.law_family == "spring")
        leaked[spring_index] = ProbeSample(
            sample_id=leaked[spring_index].sample_id,
            group_id=next(sample.group_id for sample in leaked if sample.law_family != "spring"),
            features=leaked[spring_index].features,
            label=leaked[spring_index].label,
            law_family="spring",
        )
        with self.assertRaisesRegex(ValueError, "multiple law families"):
            ood_law_family_evaluation(leaked, heldout_family="spring")


class InterventionTests(unittest.TestCase):
    def test_controls_match_norm_and_orthogonality(self):
        target = [1.0, 2.0, -1.0]
        controls = build_control_directions(target, seed=7, generic_error=[0.2, -0.1, 0.4])
        self.assertAlmostEqual(norm(controls["matched_norm"]), norm(target), places=10)
        self.assertAlmostEqual(norm(controls["orthogonal"]), norm(target), places=10)
        self.assertAlmostEqual(norm(controls["generic_error"]), norm(target), places=10)
        self.assertAlmostEqual(dot(target, controls["orthogonal"]), 0.0, places=10)
        removed = ablate([2.0, 4.0, -2.0], target)
        self.assertAlmostEqual(dot(removed, target), 0.0, places=10)
        self.assertNotEqual(inject([0.0, 0.0, 0.0], target, magnitude=1.0), [0.0, 0.0, 0.0])
        generic = generic_error_control([[2.0, 1.0], [4.0, 2.0]], [[1.0, 1.0], [2.0, 1.0]])
        self.assertEqual(generic, [1.5, 0.5])

    def test_matched_world_validation_and_directional_swap(self):
        pair = MatchedWorldPair("p", "a", "b", [-1.0, 0.0], [1.0, 0.0], {"camera": 1}, {"camera": 1}, 0, 1)
        result = evaluate_latent_swap(pair, lambda _, latent: latent[0])
        self.assertEqual(result["swap_effect"], 2.0)
        bad = MatchedWorldPair("p", "a", "b", [-1.0], [1.0], {"camera": 1}, {"camera": 2}, 0, 1)
        with self.assertRaises(ValueError):
            bad.validate()


class MetricTests(unittest.TestCase):
    def test_paired_effect_uses_locked_clustered_percentile_bootstrap(self):
        result = paired_effect(
            [1.1, 1.3, 1.2, 1.5, 1.8, 2.2],
            [0.1, 0.3, 0.2, 0.5, 0.2, 0.4],
            cluster_ids=["world-a", "world-a", "world-b", "world-b", "world-c", "world-c"],
            seed=17,
        )
        self.assertEqual(result["bootstrap_resamples"], BOOTSTRAP_RESAMPLES)
        self.assertEqual(result["bootstrap_seed"], 17)
        self.assertEqual(result["cluster_count"], 3)
        self.assertAlmostEqual(result["ate"], 1.2333333333333334)
        # Frozen numerical regression for clustered, not row-wise, percentile resampling.
        self.assertAlmostEqual(result["ci_low"], 1.0)
        self.assertAlmostEqual(result["ci_high"], 1.7)
        with self.assertRaisesRegex(ValueError, "exactly 10000"):
            paired_effect([1, 2], [0, 0], cluster_ids=["a", "b"], bootstrap_samples=9999)

    def test_mediation_recovers_positive_indirect_path(self):
        values = load_fixture()["mediation"]["checkpoint-primary"]
        result = mediation_analysis(
            values["treatment"],
            values["mediator"],
            values["outcome"],
            cluster_ids=values["cluster_ids"],
            seed=17,
        )
        self.assertGreater(result["indirect_effect"], 0.8)
        self.assertGreater(result["indirect_ci_low"], 0)
        self.assertTrue(math.isfinite(result["mediation_proportion"]))
        self.assertEqual(result["bootstrap_resamples"], BOOTSTRAP_RESAMPLES)
        self.assertEqual(result["bootstrap_seed"], 17)
        self.assertAlmostEqual(
            result["indirect_effect"], result["total_effect"] - result["direct_effect"]
        )

    def test_g6_g7_g8_require_complete_distinct_checkpoint_evidence(self):
        fixture = load_fixture()
        def record(key):
            return {
                "identity": fixture["checkpoint_identities"][key],
                "g3_passed": True,
                "g5_passed": True,
                "total_ci_low": 0.4,
                "ordered_nie_ci_lows": [0.2, 0.1],
                "mediated_proportion": 0.3,
                "specificity_passed": True,
                "ood_effect_ci_low": 0.1,
                "ood_retention": 0.6,
                "provenance_hash_match_rate": 1.0,
            }

        primary = ConfirmatoryEvidence.from_dict(record("checkpoint-primary"))
        replication = ConfirmatoryEvidence.from_dict(record("checkpoint-replication"))
        gates = evaluate_confirmatory_gates(primary, replication)
        self.assertTrue(all(gates[name]["passed"] for name in ("g6", "g7", "g8")))
        self.assertFalse(evaluate_confirmatory_gates(None, None)["g8"]["passed"])
        self.assertFalse(evaluate_confirmatory_gates(primary, None)["g8"]["passed"])

        alias = record("checkpoint-primary")
        alias["identity"] = {**alias["identity"], "checkpoint_id": "an-alias"}
        self.assertFalse(
            evaluate_confirmatory_gates(primary, ConfirmatoryEvidence.from_dict(alias))["g8"]["passed"]
        )
        g6_failure = record("checkpoint-primary")
        g6_failure["specificity_passed"] = False
        self.assertFalse(
            evaluate_confirmatory_gates(ConfirmatoryEvidence.from_dict(g6_failure), replication)["g6"]["passed"]
        )
        self.assertFalse(
            evaluate_confirmatory_gates(ConfirmatoryEvidence.from_dict(g6_failure), replication)["g8"]["passed"]
        )
        g7_failure = record("checkpoint-replication")
        g7_failure["ood_retention"] = 0.49
        self.assertFalse(
            evaluate_confirmatory_gates(primary, ConfirmatoryEvidence.from_dict(g7_failure))["g8"]["passed"]
        )
        incomplete = record("checkpoint-primary")
        incomplete.pop("mediated_proportion")
        with self.assertRaisesRegex(ValueError, "exactly"):
            ConfirmatoryEvidence.from_dict(incomplete)


if __name__ == "__main__":
    unittest.main()
