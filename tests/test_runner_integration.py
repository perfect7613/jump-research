import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jump_contracts import EvidenceError, load_task_evidence, promote_task_artifacts
from jump_mechanistic.runner import (
    MECHANISTIC_ARTIFACT_CONTRACT,
    execute_run,
    execute_task_file,
    validate_mechanistic_task_evidence,
)


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
                "observational_mediation.indirect_effect",
                "gates.g1_passed",
                "gates.g3_passed",
                "gates.g5_passed",
                "gates.g6_passed",
                "gates.g7_passed",
                "gates.g8_passed",
            }
            self.assertTrue(required <= names)
            # The fixture exercises contracts only: it is below the confirmatory
            # cluster floor and must never emit a positive mechanistic gate.
            for gate in ("gates.g1_passed", "gates.g3_passed", "gates.g5_passed", "gates.g6_passed", "gates.g7_passed", "gates.g8_passed"):
                self.assertEqual(
                    [metric["value"] for metric in result_a["metrics"] if metric["name"] == gate],
                    [0.0],
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
            gate_diagnostics = json.loads((Path(first) / "artifacts" / "computed_gates.json").read_text())
            self.assertFalse(gate_diagnostics["claim_eligible"])
            self.assertEqual(gate_diagnostics["evidence_namespace"], "synthetic_fixture_nonclaim")
            self.assertTrue(gate_diagnostics["raw_records"]["g1_regime"])
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

    def test_trusted_gate_boolean_parameters_are_rejected(self):
        self.manifest["phases"][0]["runs"][0]["task"]["parameters"]["g3_passed"] = True
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "trusted gate booleans are forbidden"):
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
            self.assertEqual(result["evidence_namespace"], "synthetic_fixture_nonclaim")
            self.assertFalse(result["claim_eligible"])
            self.assertIn("metrics", result)
            self.assertNotIn("status", result)
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertTrue((output / "artifacts" / "activations.jsonl").is_file())
            self.assertEqual(
                {
                    record["name"]: (record["role"], record["media_type"])
                    for record in result["artifacts"]
                },
                MECHANISTIC_ARTIFACT_CONTRACT,
            )
            self.assertTrue(
                all(
                    record["evidence_namespace"] == "synthetic_fixture_nonclaim"
                    and record["claim_eligible"] is False
                    for record in result["artifacts"]
                )
            )
            promoted_dir = attempt / "promoted"
            promoted = promote_task_artifacts(
                output, promoted_dir, "attempts/0001/artifacts", result
            )
            self.assertEqual(
                {
                    record["name"]: (
                        record["role"], record["media_type"], record["sha256"]
                    )
                    for record in promoted
                },
                {
                    record["name"]: (
                        record["role"], record["media_type"], record["sha256"]
                    )
                    for record in result["artifacts"]
                },
            )
            with self.assertRaisesRegex(EvidenceError, "destination already exists"):
                promote_task_artifacts(
                    output, promoted_dir, "attempts/0001/artifacts", result
                )
            (output / "artifacts" / "computed_gates.json").write_text("tampered")
            with self.assertRaisesRegex(EvidenceError, "artifact hash mismatch"):
                load_task_evidence(output / "result.json")

    def test_mechanistic_publication_rejects_legacy_discovered_artifacts(self):
        with self.assertRaisesRegex(EvidenceError, "jump.task-evidence/v1"):
            validate_mechanistic_task_evidence({"metrics": []})
        with self.assertRaisesRegex(EvidenceError, "stable artifact set"):
            validate_mechanistic_task_evidence(
                {
                    "schema_version": "jump.task-evidence/v1",
                    "metrics": [],
                    "artifacts": [],
                    "evidence_namespace": "synthetic_fixture_nonclaim",
                    "claim_eligible": False,
                }
            )


if __name__ == "__main__":
    unittest.main()
