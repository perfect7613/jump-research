import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jump_mechanistic.hardware import ProfileEvidence, materialize_h100_manifest, select_hardware


ROOT = Path(__file__).parents[1]
CODE_VERSION = "review-head-deadbeef"


class HardwareSelectionTests(unittest.TestCase):
    def setUp(self):
        self._new_profile_tree()

    def tearDown(self):
        self.temporary.cleanup()

    def _new_profile_tree(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_manifest = {
            "schema_version": "jump.experiments/v1",
            "experiment_id": "hardware-profile-fixture",
            "phases": [{
                "id": "gpu-memory-profile",
                "runs": [],
                "gates": [
                    {
                        "id": "hook-integrity",
                        "metric": "activation_hash_match_rate",
                        "aggregation": "min",
                        "operator": "==",
                        "threshold": 1,
                    },
                    {
                        "id": "profile-complete",
                        "metric": "hardware_profile_complete",
                        "aggregation": "min",
                        "operator": "==",
                        "threshold": 1,
                    },
                ],
            }],
        }
        self.l40s = self._runner_result(
            "l40s-one-percent-hook-smoke", "L40S", measured_runtime=600.0
        )
        self.a100 = self._runner_result(
            "a100-throughput-profile", "A100-80GB", measured_runtime=500.0
        )
        self._write_phase_result()

    def setUp_clean_profiles(self):
        self.temporary.cleanup()
        self._new_profile_tree()

    def _runner_result(self, run_id, gpu, *, measured_runtime, code_version=CODE_VERSION):
        root = self.root / "phases" / "gpu-memory-profile" / "runs" / run_id
        artifact = root / "attempts" / "0001" / "artifacts" / "hardware-profile.json"
        artifact.parent.mkdir(parents=True)
        profile = {
            "schema_version": "jump.hardware-profile/v1",
            "run_id": run_id,
            "gpu": gpu,
            "workload": "activation_capture",
            "checkpoint_id": "gemma-4-12b-research",
            "checkpoint_revision": "gemma-4-12b-research@immutable-revision",
            "measured_peak_memory_gb": 60.0,
            "measured_runtime_seconds": measured_runtime,
            "projected_l40s_runtime_seconds": 3600.0,
            "projected_a100_runtime_seconds": 2400.0,
            "projected_h100_runtime_seconds": 1500.0,
            "runtime_ceiling_seconds": 1800.0,
            "remaining_budget_usd": 2.0,
        }
        artifact.write_text(json.dumps(profile, sort_keys=True, allow_nan=False))
        task = {
            "parameters": {
                "workload": profile["workload"],
                "checkpoint_id": profile["checkpoint_id"],
                "checkpoint_revision": profile["checkpoint_revision"],
            }
        }
        config = {
            "schema_version": "jump.run-config/v1",
            "manifest_sha256": "0" * 64,
            "phase_id": "gpu-memory-profile",
            "run_id": run_id,
            "code_version": code_version,
            "resources": {"gpu": gpu},
            "task": task,
        }
        result = {
            "schema_version": "jump.run-result/v1",
            "status": "completed",
            "metrics": [],
            "artifacts": [
                {
                    "name": "hardware-profile.json",
                    "path": "attempts/0001/artifacts/hardware-profile.json",
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "media_type": "application/json",
                }
            ],
            "provenance": {
                "manifest_sha256": "0" * 64,
                "run_id": run_id,
                "code_version": code_version,
            },
        }
        (root / "config.json").write_text(json.dumps(config, sort_keys=True))
        result_path = root / "result.json"
        result_path.write_text(json.dumps(result, sort_keys=True))
        self.source_manifest["phases"][0]["runs"].append(
            {"id": run_id, "resources": {"gpu": gpu}, "task": task}
        )
        self._rebind_all()
        return result_path

    def _rehash(self, result_path):
        root = result_path.parent
        artifact = root / "attempts" / "0001" / "artifacts" / "hardware-profile.json"
        bound_paths = [root / "config.json", result_path, artifact]
        (root / "hashes.sha256").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
                for path in bound_paths
            )
        )

    def _rebind_all(self):
        manifest_bytes = (
            json.dumps(
                self.source_manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        (self.root / "manifest.json").write_bytes(manifest_bytes)
        self.manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        for result_path in self.root.glob("phases/gpu-memory-profile/runs/*/result.json"):
            config_path = result_path.parent / "config.json"
            config = json.loads(config_path.read_text())
            result = json.loads(result_path.read_text())
            config["manifest_sha256"] = self.manifest_sha
            result["provenance"]["manifest_sha256"] = self.manifest_sha
            config_path.write_text(json.dumps(config, sort_keys=True))
            result_path.write_text(json.dumps(result, sort_keys=True))
            self._rehash(result_path)

    def _write_phase_result(self, *, status="passed", gate_overrides=None):
        declared = self.source_manifest["phases"][0]["gates"]
        gates = [
            {
                "id": gate["id"],
                "metric": gate["metric"],
                "aggregation": gate["aggregation"],
                "filters": {},
                "operator": gate["operator"],
                "threshold": gate["threshold"],
                "actual": 1,
                "samples": 2,
                "passed": True,
            }
            for gate in declared
        ]
        if gate_overrides is not None:
            gates = gate_overrides
        phase_result = {
            "schema_version": "jump.phase-result/v1",
            "phase_id": "gpu-memory-profile",
            "status": status,
            "runtime_seconds": 1,
            "estimated_cost_usd": 0,
            "gates": gates,
        }
        path = self.root / "phases" / "gpu-memory-profile" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(phase_result, sort_keys=True))
        self.phase_result = path

    def evidence(self):
        return ProfileEvidence.from_runner_results(self.l40s, self.a100)

    def materialize(self):
        template = json.loads(
            (ROOT / "examples" / "mechanistic-h100-escalation.manifest.template.json").read_text()
        )
        return materialize_h100_manifest(
            template,
            l40s_result_path=self.l40s,
            a100_result_path=self.a100,
        )

    def update_profiles(self, **overrides):
        for result_path in (self.l40s, self.a100):
            artifact = (
                result_path.parent
                / "attempts"
                / "0001"
                / "artifacts"
                / "hardware-profile.json"
            )
            profile = json.loads(artifact.read_text())
            profile.update(overrides)
            artifact.write_text(json.dumps(profile, sort_keys=True, allow_nan=False))
            result = json.loads(result_path.read_text())
            result["artifacts"][0]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result, sort_keys=True))
            self._rehash(result_path)
        return self.evidence()

    def test_checked_in_profile_manifest_has_no_h100_and_is_locked(self):
        manifest = json.loads(
            (ROOT / "examples" / "mechanistic-gpu-profile.manifest.json").read_text()
        )
        self.assertFalse(manifest["launch_policy"]["allow_full_matrix"])
        self.assertFalse(manifest["launch_policy"]["allow_h100"])
        resources = {
            run["resources"]["gpu"] for phase in manifest["phases"] for run in phase["runs"]
        }
        self.assertNotIn("H100", resources)

    def test_evidence_is_bound_to_runner_manifest_artifacts_and_config(self):
        evidence = self.evidence()
        self.assertEqual(evidence.l40s_run_id, "l40s-one-percent-hook-smoke")
        self.assertEqual(evidence.a100_run_id, "a100-throughput-profile")
        self.assertEqual(evidence.manifest_sha256, self.manifest_sha)
        self.assertRegex(evidence.l40s_artifact_sha256, r"^[0-9a-f]{64}$")
        fabricated = dataclasses.replace(evidence, remaining_budget_usd=999.0)
        with self.assertRaisesRegex(ValueError, "profile evidence"):
            select_hardware(fabricated, h100_forecast_cost_usd=1.0)
        object.__setattr__(evidence, "manifest_sha256", "f" * 64)
        with self.assertRaisesRegex(ValueError, "changed after"):
            select_hardware(evidence, h100_forecast_cost_usd=1.0)

    def test_materialization_reloads_paths_and_rejects_replaced_evidence(self):
        evidence = self.evidence()
        forged = dataclasses.replace(
            evidence,
            remaining_budget_usd=999.0,
            manifest_sha256="f" * 64,
            projected_h100_runtime_seconds=1.0,
        )
        template = json.loads(
            (ROOT / "examples" / "mechanistic-h100-escalation.manifest.template.json").read_text()
        )
        with self.assertRaises(TypeError):
            materialize_h100_manifest(template, forged)
        manifest, _ = self.materialize()
        justification = manifest["phases"][-1]["h100_justification"]
        self.assertEqual(justification["remaining_budget_usd"], 2.0)
        self.assertEqual(justification["manifest_sha256"], self.manifest_sha)
        self.assertNotEqual(justification["manifest_sha256"], forged.manifest_sha256)

    def test_post_verification_artifact_mutation_is_rechecked_at_materialization(self):
        self.evidence()
        artifact = self.a100.parent / "attempts" / "0001" / "artifacts" / "hardware-profile.json"
        profile = json.loads(artifact.read_text())
        profile["remaining_budget_usd"] = 999.0
        artifact.write_text(json.dumps(profile, sort_keys=True))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.materialize()

    def test_missing_tampered_and_mismatched_artifacts_are_rejected(self):
        artifact = self.l40s.parent / "attempts" / "0001" / "artifacts" / "hardware-profile.json"
        artifact.write_text(artifact.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.evidence()

        self.setUp_clean_profiles()
        result = json.loads(self.a100.read_text())
        result["artifacts"] = []
        self.a100.write_text(json.dumps(result))
        self._rehash(self.a100)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.evidence()

        self.setUp_clean_profiles()
        config = json.loads((self.a100.parent / "config.json").read_text())
        config["resources"]["gpu"] = "L40S"
        (self.a100.parent / "config.json").write_text(json.dumps(config))
        self._rehash(self.a100)
        with self.assertRaisesRegex(ValueError, "A100-80GB"):
            self.evidence()

    def test_missing_or_tampered_source_manifest_is_rejected(self):
        (self.root / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "source manifest"):
            self.evidence()
        self.setUp_clean_profiles()
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["experiment_id"] = "tampered"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            self.evidence()

    def test_profile_phase_and_every_declared_gate_must_pass(self):
        self.phase_result.unlink()
        with self.assertRaisesRegex(ValueError, "passed profile phase result"):
            self.materialize()
        self._write_phase_result(status="failed")
        with self.assertRaisesRegex(ValueError, "must be a passed"):
            self.materialize()
        self._write_phase_result(gate_overrides=[])
        with self.assertRaisesRegex(ValueError, "every declared gate"):
            self.materialize()
        self._write_phase_result()
        result = json.loads(self.phase_result.read_text())
        result["gates"][0]["passed"] = False
        self.phase_result.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.materialize()

    def test_cross_result_code_and_workload_mismatch_are_rejected(self):
        config_path = self.a100.parent / "config.json"
        config = json.loads(config_path.read_text())
        result = json.loads(self.a100.read_text())
        config["code_version"] = result["provenance"]["code_version"] = "different-code"
        config_path.write_text(json.dumps(config))
        self.a100.write_text(json.dumps(result))
        self._rehash(self.a100)
        with self.assertRaisesRegex(ValueError, "mismatch for code_version"):
            self.evidence()

        self.setUp_clean_profiles()
        artifact = self.a100.parent / "attempts" / "0001" / "artifacts" / "hardware-profile.json"
        profile = json.loads(artifact.read_text())
        profile["workload"] = "adapter_training"
        artifact.write_text(json.dumps(profile, sort_keys=True))
        result = json.loads(self.a100.read_text())
        result["artifacts"][0]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        self.a100.write_text(json.dumps(result))
        config = json.loads((self.a100.parent / "config.json").read_text())
        config["task"]["parameters"]["workload"] = "adapter_training"
        (self.a100.parent / "config.json").write_text(json.dumps(config))
        a100_run = next(
            run
            for run in self.source_manifest["phases"][0]["runs"]
            if run["id"] == "a100-throughput-profile"
        )
        a100_run["task"]["parameters"]["workload"] = "adapter_training"
        self._rebind_all()
        with self.assertRaisesRegex(ValueError, "mismatch for workload"):
            self.evidence()

    def test_uses_cheapest_tier_and_h100_requires_speedup_budget(self):
        evidence = self.evidence()
        self.assertEqual(
            select_hardware(evidence, h100_forecast_cost_usd=1.9746).selected_gpu, "H100"
        )
        self.setUp_clean_profiles()
        l40s = self.update_profiles(
            measured_peak_memory_gb=40, projected_l40s_runtime_seconds=1700
        )
        self.assertEqual(select_hardware(l40s, h100_forecast_cost_usd=1.9).selected_gpu, "L40S")
        self.setUp_clean_profiles()
        a100 = self.update_profiles(projected_a100_runtime_seconds=1700)
        self.assertEqual(select_hardware(a100, h100_forecast_cost_usd=1.9).selected_gpu, "A100-80GB")
        self.setUp_clean_profiles()
        slow = self.update_profiles(
            projected_a100_runtime_seconds=2000, projected_h100_runtime_seconds=1700
        )
        self.assertFalse(select_hardware(slow, h100_forecast_cost_usd=1.9746).passed)
        self.setUp_clean_profiles()
        over_budget = self.update_profiles(remaining_budget_usd=1)
        self.assertFalse(select_hardware(over_budget, h100_forecast_cost_usd=1.9746).passed)

    def test_h100_does_not_solve_eighty_gb_oom(self):
        too_large = self.update_profiles(measured_peak_memory_gb=73)
        decision = select_hardware(too_large, h100_forecast_cost_usd=1.9746)
        self.assertIsNone(decision.selected_gpu)
        self.assertIn("shard", decision.reason)

    def test_materializer_adds_bound_evidence_but_preserves_launch_locks(self):
        manifest, decision = self.materialize()
        self.assertEqual(decision.selected_gpu, "H100")
        self.assertFalse(manifest["launch_policy"]["allow_full_matrix"])
        self.assertFalse(manifest["launch_policy"]["allow_h100"])
        justification = manifest["phases"][-1]["h100_justification"]
        self.assertEqual(justification["forecast_cost_usd"], 1.9746)
        self.assertEqual(justification["manifest_sha256"], self.manifest_sha)
        self.assertEqual(justification["code_version"], CODE_VERSION)
        self.assertEqual(justification["l40s_run_id"], "l40s-one-percent-hook-smoke")
        self.assertEqual(justification["a100_run_id"], "a100-throughput-profile")
        self.assertEqual(
            justification["profile_phase_result_sha256"],
            hashlib.sha256(self.phase_result.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
