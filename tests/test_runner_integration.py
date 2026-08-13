import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jump_contracts import EvidenceError, load_task_evidence, promote_task_artifacts
from jump_mechanistic.runner import (
    MECHANISTIC_ARTIFACT_CONTRACT,
    execute_task_file,
    validate_mechanistic_task_evidence,
)


ROOT = Path(__file__).parents[1]


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "examples" / "mechanistic-synthetic.manifest.json").read_text())

    def _execute_task(self, directory, *, parameter_changes=None, config_changes=None):
        root = Path(directory) / "run"
        attempt = root / "attempts" / "0001"
        attempt.mkdir(parents=True)
        config = {
            "preregistration": self.manifest["preregistration"],
            "selection": self.manifest["phases"][0]["runs"][0]["selection"],
        }
        if config_changes:
            config_changes(config)
        (root / "config.json").write_text(json.dumps(config))
        parameters = dict(self.manifest["phases"][0]["runs"][0]["task"]["parameters"])
        if parameter_changes:
            parameters.update(parameter_changes)
        parameter_path = attempt / "task-parameters.json"
        parameter_path.write_text(json.dumps(parameters))
        output = attempt / "work"
        return output, execute_task_file(parameter_path, output_dir=output)

    def test_end_to_end_task_evidence_and_artifact_hashes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_output, result_a = self._execute_task(first)
            _second_output, result_b = self._execute_task(second)
            self.assertEqual(result_a, result_b)
            self.assertEqual(result_a["schema_version"], "jump.task-evidence/v1")
            self.assertFalse(result_a["claim_eligible"])
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
                contents = (first_output / artifact["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(contents).hexdigest(), artifact["sha256"])
            gate_diagnostics = json.loads((first_output / "artifacts" / "computed_gates.json").read_text())
            self.assertFalse(gate_diagnostics["claim_eligible"])
            self.assertEqual(gate_diagnostics["evidence_namespace"], "synthetic_fixture_nonclaim")
            self.assertTrue(gate_diagnostics["raw_records"]["g1_regime"])
            self.assertEqual(json.loads((first_output / "result.json").read_text()), result_a)

    def test_run_cannot_expand_preregistered_allowlists(self):
        def expand(config):
            config["selection"] = {**config["selection"], "layers": [*config["selection"]["layers"], "model.layers.47"]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                self._execute_task(directory, config_changes=expand)

    def test_run_cannot_alias_or_omit_preregistered_checkpoint_revision(self):
        parameters = self.manifest["phases"][0]["runs"][0]["task"]["parameters"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "replication checkpoint identity"):
                self._execute_task(
                    directory,
                    parameter_changes={"replication_revision": parameters["checkpoint_revision"]},
                )

    def test_trusted_gate_boolean_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "trusted gate booleans are forbidden"):
                self._execute_task(directory, parameter_changes={"g3_passed": True})

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

    def test_repeat_task_write_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            output, first = self._execute_task(directory)
            first_bytes = (output / "result.json").read_bytes()
            artifact_bytes = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            parameter_path = Path(directory) / "run" / "attempts" / "0001" / "task-parameters.json"
            parameters = json.loads(parameter_path.read_text())
            parameter_path.write_text(json.dumps({**parameters, "seed": 999}))
            with self.assertRaisesRegex(EvidenceError, "immutable evidence already exists"):
                execute_task_file(parameter_path, output_dir=output)
            self.assertEqual((output / "result.json").read_bytes(), first_bytes)
            self.assertEqual(json.loads(first_bytes), first)
            self.assertEqual(
                {
                    path.relative_to(output): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                },
                artifact_bytes,
            )

    def test_legacy_manifest_cli_bypass_is_removed(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "jump_mechanistic.runner",
                "--manifest",
                str(ROOT / "examples" / "mechanistic-synthetic.manifest.json"),
                "--output-dir",
                str(ROOT),
            ],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("required: --parameters", completed.stderr)

    def test_runner_parameters_cannot_select_absolute_traversal_symlink_or_hash_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            symlink = root / "fixture-link.json"
            symlink.symlink_to(ROOT / "src" / "jump_mechanistic" / "fixtures" / "synthetic_experiment.json")
            for value in (
                str(ROOT / "src" / "jump_mechanistic" / "fixtures" / "synthetic_experiment.json"),
                "../synthetic_experiment.json",
                str(symlink),
            ):
                attempt = root / hashlib.sha256(value.encode()).hexdigest()
                with self.assertRaisesRegex(ValueError, "fixture selection is runner-owned"):
                    self._execute_task(attempt, parameter_changes={"fixture_path": value})
            with self.assertRaisesRegex(ValueError, "fixture selection is runner-owned"):
                self._execute_task(
                    root / "hash",
                    parameter_changes={"fixture_sha256": "0" * 64},
                )


if __name__ == "__main__":
    unittest.main()
