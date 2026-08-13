import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jump_mechanistic.runner import execute_run, execute_task_file


ROOT = Path(__file__).parents[1]


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "examples" / "mechanistic-synthetic.manifest.json").read_text())

    def test_end_to_end_result_and_artifact_hashes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            result_a = execute_run(
                self.manifest,
                phase_id="mechanistic-smoke",
                run_id="synthetic-suite",
                output_dir=first,
            )
            result_b = execute_run(
                self.manifest,
                phase_id="mechanistic-smoke",
                run_id="synthetic-suite",
                output_dir=second,
            )
            self.assertEqual(result_a, result_b)
            self.assertEqual(result_a["schema_version"], "jump.run-result/v1")
            self.assertEqual(result_a["status"], "completed")
            names = {metric["name"] for metric in result_a["metrics"]}
            required = {
                "activation_capture.count",
                "behavior.partition_accuracy",
                "behavior.adequacy_balanced_accuracy",
                "behavior.full_law_accuracy",
                "behavior.joint_theory_accuracy",
                "probe.heldout.auc",
                "probe.ood.auc",
                "swap.swap_effect",
                "causal.ate",
                "causal.target_vs_control.ate",
                "mediation.indirect_effect",
                "mediation.specificity_passed",
                "gates.g6_passed",
                "gates.g7_passed",
                "gates.g8_passed",
            }
            self.assertTrue(required <= names)
            for gate in ("gates.g6_passed", "gates.g7_passed", "gates.g8_passed"):
                self.assertEqual(
                    [metric["value"] for metric in result_a["metrics"] if metric["name"] == gate],
                    [1.0],
                )
            bootstrap_metrics = [
                metric for metric in result_a["metrics"] if metric["name"].endswith("bootstrap_resamples")
            ]
            self.assertTrue(bootstrap_metrics)
            self.assertTrue(all(metric["value"] == 10_000 for metric in bootstrap_metrics))
            capture_dimensions = {
                (metric.get("layer"), metric.get("timepoint"))
                for metric in result_a["metrics"]
                if metric["name"] == "activation_capture.count" and metric.get("layer")
            }
            self.assertEqual(
                capture_dimensions,
                {
                    (layer, timepoint)
                    for layer in self.manifest["preregistration"]["layer_allowlist"]
                    for timepoint in self.manifest["preregistration"]["timepoint_allowlist"]
                },
            )
            for artifact in result_a["artifacts"]:
                contents = (Path(first) / artifact["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(contents).hexdigest(), artifact["sha256"])
            self.assertEqual(json.loads((Path(first) / "result.json").read_text()), result_a)

    def test_run_cannot_expand_preregistered_allowlists(self):
        self.manifest["phases"][0]["runs"][0]["selection"]["layers"].append("model.layers.47")
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(PermissionError):
                execute_run(self.manifest, phase_id="mechanistic-smoke", run_id="synthetic-suite", output_dir=output)

    def test_run_cannot_alias_or_omit_preregistered_checkpoint_revision(self):
        parameters = self.manifest["phases"][0]["runs"][0]["task"]["parameters"]
        parameters["replication_revision"] = parameters["checkpoint_revision"]
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "replication checkpoint identity"):
                execute_run(
                    self.manifest,
                    phase_id="mechanistic-smoke",
                    run_id="synthetic-suite",
                    output_dir=output,
                )

    def test_shared_runner_subprocess_file_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            attempt = root / "attempts" / "0001"
            attempt.mkdir(parents=True)
            config = {
                "preregistration": self.manifest["preregistration"],
                "selection": self.manifest["phases"][0]["runs"][0]["selection"],
            }
            (root / "config.json").write_text(json.dumps(config))
            parameter_path = attempt / "task-parameters.json"
            parameter_path.write_text(
                json.dumps(self.manifest["phases"][0]["runs"][0]["task"]["parameters"])
            )
            output = attempt / "work"
            checkpoint = attempt / "checkpoint"
            result = execute_task_file(
                parameter_path,
                output_dir=output,
                checkpoint_dir=checkpoint,
            )
            self.assertEqual(result["schema_version"], "jump.task-evidence/v1")
            self.assertIn("metrics", result)
            self.assertNotIn("status", result)
            self.assertEqual(
                {artifact["media_type"] for artifact in result["artifacts"]},
                {"application/json", "application/x-ndjson"},
            )
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertTrue((output / "artifacts" / "activations.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
