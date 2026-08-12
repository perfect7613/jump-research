from __future__ import annotations

import json
from pathlib import Path

import pytest

from jump_runner.executor import run_manifest, run_root
from jump_runner.manifest import manifest_hash


def test_smoke_writes_immutable_evidence_and_skips_completed(manifest, tmp_path):
    first = run_manifest(manifest, tmp_path, smoke=True)
    assert first["status"] == "completed"
    root = run_root(tmp_path, manifest, "smoke")
    run_dir = root / "phases/pilot/runs/pilot-1"
    result_before = (run_dir / "result.json").read_bytes()
    config_before = (run_dir / "config.json").read_bytes()
    attempts_before = sorted((run_dir / "attempts").iterdir())

    second = run_manifest(manifest, tmp_path, smoke=True)
    assert second["status"] == "completed"
    assert (run_dir / "result.json").read_bytes() == result_before
    assert (run_dir / "config.json").read_bytes() == config_before
    assert sorted((run_dir / "attempts").iterdir()) == attempts_before
    result = json.loads(result_before)
    assert result["provenance"]["manifest_sha256"] == manifest_hash(manifest)
    assert result["artifacts"][0]["sha256"]
    assert (attempts_before[0] / "stdout.log").exists()
    assert (attempts_before[0] / "stderr.log").exists()
    assert (attempts_before[0] / "hashes.sha256").exists()


def test_retry_creates_new_immutable_attempt_and_uses_checkpoint(manifest, tmp_path):
    run = manifest["phases"][0]["runs"][0]
    run["retry"] = {"max_attempts": 2, "retry_on_exit_codes": [7]}
    run["task"]["parameters"]["fail_until_attempt"] = 1
    manifest["phases"][0]["budget"]["max_runtime_seconds"] = 10
    result = run_manifest(manifest, tmp_path, smoke=True)
    assert result["status"] == "completed"
    run_dir = run_root(tmp_path, manifest, "smoke") / "phases/pilot/runs/pilot-1"
    attempts = sorted((run_dir / "attempts").iterdir())
    assert len(attempts) == 2
    assert json.loads((attempts[0] / "result.json").read_text())["status"] == "failed"
    started = json.loads((attempts[1] / "started.json").read_text())
    assert started["resume_from"].endswith("0001/checkpoint")


def test_crash_after_completed_attempt_finalizes_without_rerun(manifest, tmp_path):
    run_manifest(manifest, tmp_path, smoke=True)
    run_dir = run_root(tmp_path, manifest, "smoke") / "phases/pilot/runs/pilot-1"
    # Model the exact crash window: attempt result is durable, terminal result and
    # root hash manifests were not committed.
    (run_dir / "result.json").unlink()
    (run_dir / "hashes.sha256").unlink()
    (run_dir / "artifact_hashes.json").unlink()
    (run_dir.parent.parent / "result.json").unlink()
    root = run_root(tmp_path, manifest, "smoke")
    (root / "status.json").unlink()
    resumed = run_manifest(manifest, tmp_path, smoke=True)
    assert resumed["status"] == "completed"
    assert [path.name for path in (run_dir / "attempts").iterdir()] == ["0001"]


def test_incomplete_attempt_consumes_max_attempt_budget(manifest, tmp_path):
    from jump_runner.executor import execute_local_run
    from jump_runner.io import write_json_immutable

    phase = dict(manifest["phases"][0])
    phase["_preregistration"] = manifest["preregistration"]
    run = manifest["phases"][0]["runs"][0] | manifest["defaults"]
    run["resources"] = manifest["defaults"]["resources"]
    run["retry"] = manifest["defaults"]["retry"]
    run_dir = tmp_path / "interrupted"
    attempt = run_dir / "attempts/0001"
    attempt.mkdir(parents=True)
    write_json_immutable(
        attempt / "started.json",
        {"attempt": 1, "started_at": "2026-01-01T00:00:00+00:00", "manifest_sha256": "a" * 64},
    )
    result = execute_local_run(phase, run, run_dir, "a" * 64)
    assert result["status"] == "failed"
    assert result["retryable"] is False
    assert not (run_dir / "attempts/0002").exists()


@pytest.mark.parametrize("secret", ["abc", "sk-super-secret-value"])
def test_logs_redact_every_declared_value_length(manifest, tmp_path, monkeypatch, secret):
    monkeypatch.setenv("X", secret)
    monkeypatch.setenv("UNDECLARED_SECRET", "must-not-be-inherited")
    run = manifest["phases"][0]["runs"][0]
    run["task"]["parameters"].update(
        {
            "echo_env": ["X", "UNDECLARED_SECRET"],
            "stderr_text": f"generic={secret}",
            "traceback_secret": secret,
        }
    )
    from jump_runner.executor import execute_local_run
    from jump_runner.manifest import resolve_run

    phase = dict(manifest["phases"][0])
    phase["_preregistration"] = manifest["preregistration"]
    phase["_secret_keys"] = ["X"]
    resolved = resolve_run(manifest["defaults"], run)
    run_dir = tmp_path / "secret-run"
    result = execute_local_run(phase, resolved, run_dir, "b" * 64)
    assert result["status"] == "completed"
    stdout = (run_dir / "attempts/0001/stdout.log").read_text()
    stderr = (run_dir / "attempts/0001/stderr.log").read_text()
    assert secret not in stdout + stderr
    assert "[REDACTED]" in stdout
    assert "[REDACTED]" in stderr
    assert "must-not-be-inherited" not in stdout + stderr


def test_failed_gate_stops_dependent_phase(manifest, tmp_path):
    manifest["phases"][0]["gates"][0]["threshold"] = 0.9
    manifest["phases"].append(
        {
            "id": "mechanistic",
            "depends_on": ["pilot"],
            "budget": {
                "allowed_gpu_types": ["T4"],
                "max_concurrent_gpus": 1,
                "max_runtime_seconds": 5,
                "max_cost_usd": 1,
                "gpu_hourly_cost_usd": {"T4": 0.5},
            },
            "runs": [
                {
                    "id": "must-not-run",
                    "smoke_test": True,
                    "task": {"module": "jump_runner.mock_task", "parameters": {"layers": [4], "timepoints": ["T4"]}},
                    "resources": {"gpu": "T4", "timeout_seconds": 5},
                }
            ],
        }
    )
    result = run_manifest(manifest, tmp_path, smoke=True)
    assert result["status"] == "stopped"
    assert result["stopped_reason"] == "gate pilot-pass failed"
    root = run_root(tmp_path, manifest, "smoke")
    assert not (root / "phases/mechanistic").exists()


def test_result_layer_is_revalidated(manifest, tmp_path):
    # A malicious/buggy task cannot escape the preregistration through output.
    manifest["phases"][0]["runs"][0]["task"]["parameters"]["result_layer"] = 999
    result = run_manifest(manifest, tmp_path, smoke=True)
    assert result["status"] == "stopped"
    assert result["stopped_reason"].startswith("run pilot-1 failed")
