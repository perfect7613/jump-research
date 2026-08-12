import tempfile
import unittest
from pathlib import Path

from jump_mechanistic.capture import ActivationCapture, CapturePolicy
from jump_mechanistic.scoring import Law, partition_correct, score_dataset


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class CaptureTests(unittest.TestCase):
    def test_allowlists_and_tensor_flattening(self):
        capture = ActivationCapture(CapturePolicy.from_allowlists(["layer.4"], ["T2"]))
        record = capture.capture(
            episode_id="e1",
            checkpoint_id="c1",
            layer="layer.4",
            timepoint="T2",
            activation=FakeTensor([[1, 2], [3, 4]]),
        )
        self.assertEqual(record.values, [1.0, 2.0, 3.0, 4.0])
        hook = capture.make_hook(
            episode_id="e-hook",
            checkpoint_id="c1",
            layer="layer.4",
            timepoint="T2",
            selector=lambda output: output[-1],
        )
        hook(None, None, [[0, 0], [5, 6]])
        self.assertEqual(capture.records[-1].values, [5.0, 6.0])
        with self.assertRaises(PermissionError):
            capture.capture(
                episode_id="e2", checkpoint_id="c1", layer="layer.5", timepoint="T2", activation=[1]
            )
        with self.assertRaises(PermissionError):
            capture.capture(
                episode_id="e2", checkpoint_id="c1", layer="layer.4", timepoint="T3", activation=[1]
            )
        with self.assertRaises(ValueError):
            capture.capture(
                episode_id="e1", checkpoint_id="c1", layer="layer.4", timepoint="T2", activation=[1]
            )


class ScoringTests(unittest.TestCase):
    def test_partition_is_label_swap_invariant_but_nontrivial(self):
        expected = [0, 0, 1, 1, 0, 1]
        self.assertTrue(partition_correct(expected, expected))
        self.assertTrue(partition_correct([1 - x for x in expected], expected))
        self.assertFalse(partition_correct([0] * 6, expected))
        self.assertFalse(partition_correct([0, 1], expected))

    def test_law_schema_is_exact(self):
        self.assertEqual(Law.parse({"same": "attract", "different": "repel", "exponent": 2}).exponent, 2)
        with self.assertRaises(ValueError):
            Law.parse({"same": "attract", "different": "repel", "exponent": 2, "explanation": "x"})

    def test_dataset_scores_without_text_judge(self):
        target = {
            "partition": [0, 0, 1, 1, 0, 1],
            "replacement_law": {"same": "attract", "different": "repel", "exponent": 2},
            "adequate": False,
            "forces": [[1.0, 0.0]],
            "control_no_hidden_types": True,
        }
        positive = {**target, "adequate": True, "control_no_hidden_types": False}
        rows = [
            {"prediction": {**target, "confidence": 1.0, "posits_hidden_types": False}, "target": target},
            {"prediction": {**positive, "confidence": 1.0}, "target": positive},
        ]
        scores = score_dataset(rows, allowed_exponents=[1, 2, 3])
        self.assertEqual(scores["partition_accuracy"], 1.0)
        self.assertEqual(scores["full_law_accuracy"], 1.0)
        self.assertEqual(scores["sign_accuracy"], 1.0)
        self.assertEqual(scores["joint_theory_accuracy"], 1.0)
        self.assertEqual(scores["adequacy_balanced_accuracy"], 1.0)
        self.assertEqual(scores["false_abduction"], 0.0)
        self.assertEqual(scores["confidence_brier"], 0.0)
        self.assertEqual(scores["partition_chance"], 1 / 31)
        self.assertEqual(scores["sign_chance"], 1 / 4)
        self.assertEqual(scores["full_law_chance"], 1 / 12)
        self.assertEqual(scores["joint_theory_chance"], 1 / (31 * 12))

    def test_six_object_contract_and_balanced_adequacy_are_fail_closed(self):
        target = {
            "partition": [0, 0, 1, 1, 0, 1],
            "replacement_law": {"same": "attract", "different": "repel", "exponent": 2},
            "adequate": True,
        }
        rows = []
        for index in range(10):
            gold = {**target, "adequate": index < 9}
            # Nine positive targets correct, the sole negative target missed.
            prediction = {**gold, "adequate": True}
            rows.append({"prediction": prediction, "target": gold})
        scores = score_dataset(rows, allowed_exponents=[2])
        self.assertEqual(scores["adequacy_balanced_accuracy"], 0.5)
        bad = [{"prediction": {**target, "partition": [0, 0, 1, 1]}, "target": {**target, "partition": [0, 0, 1, 1]}}]
        with self.assertRaisesRegex(ValueError, "six-object"):
            score_dataset(bad, allowed_exponents=[2])
        with self.assertRaisesRegex(ValueError, "outside"):
            score_dataset(rows, allowed_exponents=[1, 3])


if __name__ == "__main__":
    unittest.main()
