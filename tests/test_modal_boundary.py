from __future__ import annotations

import inspect
import threading
import time

import pytest

from jump_runner.errors import ManifestError, RunnerError


class MemoryLeaseStore:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def put(self, key, value, *, skip_if_exists=False):
        with self.lock:
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    def get(self, key):
        with self.lock:
            return self.values.get(key)

    def pop(self, key):
        with self.lock:
            return self.values.pop(key)


def test_controller_is_globally_single_container(monkeypatch):
    monkeypatch.setenv("JUMP_CODE_VERSION", "test-revision")
    from jump_runner import modal_app

    assert modal_app.CONTROLLER_MAX_CONTAINERS == 1
    assert modal_app.CODE_VERSION == "test-revision"
    source = inspect.getsource(modal_app)
    assert "max_containers=CONTROLLER_MAX_CONTAINERS" in source
    assert "@modal.concurrent(max_inputs=1)\ndef orchestrate" in source
    assert '"max_containers": 1' in source
    assert "execute = modal.concurrent(max_inputs=1)(execute)" in source
    from jump_benchmark.authentic_stage_c import (
        STAGE_C_MANIFEST_SHA256,
        authorize_stage_c_launch,
    )

    authorization = {
        "expected_manifest_sha256": STAGE_C_MANIFEST_SHA256,
        "expected_code_sha": "0" * 40,
        "actual_code_sha": "0" * 40,
    }
    with pytest.raises(PermissionError, match="literal confirm_paid"):
        authorize_stage_c_launch(**authorization, confirm_paid=False, confirm_h100=True)
    with pytest.raises(PermissionError, match="literal confirm_paid"):
        authorize_stage_c_launch(**authorization, confirm_paid=True, confirm_h100=False)
    authorize_stage_c_launch(**authorization, confirm_paid=True, confirm_h100=True)
    assert source.count("authorize_stage_c_launch(") == 2  # local and remote seams


def test_direct_worker_call_repeats_smoke_authorization(monkeypatch, manifest, tmp_path):
    monkeypatch.setenv("JUMP_CODE_VERSION", "test-revision")
    from jump_runner import modal_app

    run = manifest["phases"][0]["runs"][0]
    run["resources"] = {"gpu": "L40S", "timeout_seconds": 5}
    manifest["phases"][0]["budget"].update(
        {"allowed_gpu_types": ["L40S"], "gpu_hourly_cost_usd": {"L40S": 1.0}}
    )
    with pytest.raises(ManifestError, match="smoke resources are limited"):
        modal_app._worker(
            "L40S", None, manifest, "pilot", "pilot-1", True, False, False
        )


def test_duplicate_authorized_worker_uses_one_canonical_attempt(monkeypatch, manifest, tmp_path):
    monkeypatch.setenv("JUMP_CODE_VERSION", "test-revision")
    from jump_runner import modal_app
    from jump_runner.executor import run_root

    leases = MemoryLeaseStore()
    first = modal_app._execute_authorized_worker(
        "T4", None, manifest, "pilot", "pilot-1", True, False, False, tmp_path, leases
    )
    second = modal_app._execute_authorized_worker(
        "T4", None, manifest, "pilot", "pilot-1", True, False, False, tmp_path, leases
    )
    assert first["status"] == second["status"] == "completed"
    canonical = run_root(tmp_path, manifest, "smoke") / "phases/pilot/runs/pilot-1"
    assert [path.name for path in (canonical / "attempts").iterdir()] == ["0001"]
    assert (canonical / "result.json").exists()


def test_cross_worker_dispatch_is_globally_leased(monkeypatch, manifest, tmp_path):
    monkeypatch.setenv("JUMP_CODE_VERSION", "test-revision")
    from jump_runner import modal_app

    first = manifest["phases"][0]["runs"][0]
    first["task"]["parameters"]["sleep_seconds"] = 0.3
    second = {
        **first,
        "id": "pilot-2",
        "task": {
            "module": "jump_runner.mock_task",
            "parameters": {**first["task"]["parameters"], "sleep_seconds": 0},
        },
        "resources": {"gpu": "L4", "timeout_seconds": 5},
    }
    manifest["phases"][0]["runs"].append(second)
    manifest["phases"][0]["budget"].update(
        {
            "allowed_gpu_types": ["T4", "L4"],
            "max_runtime_seconds": 10,
            "gpu_hourly_cost_usd": {"T4": 0.5, "L4": 0.5},
        }
    )
    failures: list[Exception] = []
    leases = MemoryLeaseStore()

    def run_first():
        try:
            modal_app._execute_authorized_worker(
                "T4", None, manifest, "pilot", "pilot-1", True, False, False, tmp_path, leases
            )
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    for _ in range(100):
        if leases.get("global") is not None:
            break
        time.sleep(0.01)
    with pytest.raises(RunnerError, match="global dispatch lease"):
        modal_app._execute_authorized_worker(
            "L4", None, manifest, "pilot", "pilot-2", True, False, False, tmp_path, leases
        )
    thread.join()
    assert failures == []
