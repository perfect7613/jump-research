from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from jump_contracts.evidence import load_task_evidence, promote_task_artifacts

from .errors import GateFailed, ManifestError, RunnerError
from .io import atomic_write, canonical_json, sha256_file, write_json_immutable
from .manifest import (
    manifest_hash,
    resolve_run,
    selected_layers,
    selected_timepoints,
    validate_json_schema,
    validate_manifest,
)

RunExecutor = Callable[[dict[str, Any], dict[str, Any], Path, str], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_version() -> str:
    return os.environ.get("JUMP_CODE_VERSION", "unknown")


def run_root(runs_dir: Path, manifest: dict[str, Any], mode: str = "full") -> Path:
    return runs_dir / manifest["experiment_id"] / manifest_hash(manifest) / mode


def _attempt_number(root: Path) -> int:
    attempts = root / "attempts"
    existing = [int(path.name) for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()] if attempts.exists() else []
    return max(existing, default=0) + 1


def _redact_bytes(content: bytes, secret_values: list[str]) -> bytes:
    text = content.decode("utf-8", errors="replace")
    for value in sorted({value for value in secret_values if value}, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    # Defense in depth for common credential formats in tracebacks/tool output.
    patterns = (
        r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*)[^\s,;]+",
        r"\b(?:sk|hf|ghp|github_pat|ak)-[A-Za-z0-9_-]{8,}\b",
    )
    for pattern in patterns:
        text = re.sub(pattern, lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    return text.encode("utf-8")


def _persist_redacted_log(source: Path, target: Path, secret_values: list[str]) -> None:
    from .io import write_immutable

    content = source.read_bytes() if source.exists() else b""
    write_immutable(target, _redact_bytes(content, secret_values))
    source.unlink(missing_ok=True)


def _recover_attempt(root: Path, run: dict[str, Any], secret_values: list[str]) -> dict[str, Any] | None:
    """Reconcile the last durable attempt before allocating another attempt."""
    attempts_dir = root / "attempts"
    attempts = sorted(
        (path for path in attempts_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ) if attempts_dir.exists() else []
    if not attempts:
        return None
    last = attempts[-1]
    number = int(last.name)
    result_path = last / "result.json"
    max_attempts = run.get("retry", {}).get("max_attempts", 1)
    for stream in ("stdout", "stderr"):
        raw = last / f".{stream}.raw"
        if raw.exists():
            target = last / f"{stream}.log"
            if target.exists():
                raw.unlink()
            else:
                _persist_redacted_log(raw, target, secret_values)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") == "completed" or not result.get("retryable", False) or number >= max_attempts:
            terminal = dict(result)
            terminal["retryable"] = False
            write_json_immutable(root / "result.json", terminal)
            _write_hash_manifest(root)
            return terminal
        return None

    # A started attempt with no result consumed an attempt budget. Preserve an
    # explicit crash record; never reuse or silently skip its number.
    started_path = last / "started.json"
    started = json.loads(started_path.read_text()) if started_path.exists() else {}
    retryable = number < max_attempts
    recovered = {
        "schema_version": "jump.run-result/v1",
        "status": "failed",
        "attempt": number,
        "started_at": started.get("started_at"),
        "finished_at": utc_now(),
        "duration_seconds": 0,
        "exit_code": None,
        "error": "interrupted before durable attempt finalization",
        "metrics": [],
        "artifacts": [],
        "provenance": {
            "manifest_sha256": started.get("manifest_sha256", "0" * 64),
            "run_id": run["id"],
            "code_version": code_version(),
        },
        "retryable": retryable,
    }
    write_json_immutable(result_path, recovered)
    _write_hash_manifest(last)
    if not retryable:
        write_json_immutable(root / "result.json", recovered)
        _write_hash_manifest(root)
        return recovered
    return None


def _hash_tree(root: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "hashes.sha256":
            records.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return records


def _write_hash_manifest(root: Path) -> None:
    lines = [f"{record['sha256']}  {record['path']}" for record in _hash_tree(root)]
    write_json_immutable(root / "artifact_hashes.json", _hash_tree(root))
    # Compute the traditional checksum file last; it intentionally excludes itself and
    # artifact_hashes.json to avoid a recursive digest.
    digest_lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "artifact_hashes.json"}:
            digest_lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    content = ("\n".join(digest_lines) + "\n").encode()
    from .io import write_immutable

    write_immutable(root / "hashes.sha256", content)


def execute_local_run(
    phase: dict[str, Any], run: dict[str, Any], root: Path, manifest_sha256: str
) -> dict[str, Any]:
    """Execute exactly one run via ``python -m``, used locally and inside GPU workers."""
    final_path = root / "result.json"
    if final_path.exists():
        return json.loads(final_path.read_text())

    run_root_path = root
    run_root_path.mkdir(parents=True, exist_ok=True)
    secret_keys = phase.get("_secret_keys", [])
    secret_values = [os.environ[key] for key in secret_keys if key in os.environ]
    recovered = _recover_attempt(run_root_path, run, secret_values)
    if recovered is not None:
        return recovered
    config = {
        "schema_version": "jump.run-config/v1",
        "manifest_sha256": manifest_sha256,
        "phase_id": phase["id"],
        "run_id": run["id"],
        "task": run["task"],
        "resources": run["resources"],
        "selection": {
            "layers": sorted(selected_layers(run), key=str),
            "timepoints": sorted(selected_timepoints(run), key=str),
        },
        "preregistration": phase["_preregistration"],
        "code_version": code_version(),
    }
    write_json_immutable(run_root_path / "config.json", config)

    number = _attempt_number(run_root_path)
    attempt = run_root_path / "attempts" / f"{number:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    work_dir = attempt / "work"
    work_dir.mkdir()
    checkpoint_dir = attempt / "checkpoint"
    checkpoint_dir.mkdir()
    previous = attempt.parent / f"{number - 1:04d}" / "checkpoint" if number > 1 else None
    if number > run.get("retry", {}).get("max_attempts", 1):
        raise RunnerError(f"attempt budget exhausted for {run['id']}")
    started = {
        "attempt": number,
        "started_at": utc_now(),
        "resume_from": str(previous) if previous and previous.exists() else None,
        "manifest_sha256": manifest_sha256,
    }
    write_json_immutable(attempt / "started.json", started)
    task_config_path = attempt / "task-parameters.json"
    write_json_immutable(task_config_path, run["task"].get("parameters", {}))

    inherited_keys = ("PATH", "PYTHONPATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
    env = {key: os.environ[key] for key in inherited_keys if key in os.environ}
    env.update({key: os.environ[key] for key in secret_keys if key in os.environ})
    env.update(
        {
            "JUMP_RUN_ID": run["id"],
            "JUMP_PHASE_ID": phase["id"],
            "JUMP_PARAMETERS_PATH": str(task_config_path.resolve()),
            "JUMP_OUTPUT_DIR": str(work_dir.resolve()),
            "JUMP_CHECKPOINT_DIR": str(checkpoint_dir.resolve()),
            "JUMP_RESUME_FROM": str(previous.resolve()) if previous and previous.exists() else "",
        }
    )
    args = [
        sys.executable,
        "-m",
        run["task"]["module"],
        "--parameters",
        str(task_config_path.resolve()),
        "--output-dir",
        str(work_dir.resolve()),
        "--checkpoint-dir",
        str(checkpoint_dir.resolve()),
    ]
    started_monotonic = time.monotonic()
    exit_code: int | None = None
    error: str | None = None
    stdout_raw = attempt / ".stdout.raw"
    stderr_raw = attempt / ".stderr.raw"
    with stdout_raw.open("xb") as stdout, stderr_raw.open("xb") as stderr:
        try:
            completed = subprocess.run(
                args,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=run["resources"]["timeout_seconds"],
                check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            error = f"timed out after {run['resources']['timeout_seconds']} seconds"
        except Exception as exc:  # captured into the immutable attempt record
            error = f"{type(exc).__name__}: {exc}"
    duration = time.monotonic() - started_monotonic
    _persist_redacted_log(stdout_raw, attempt / "stdout.log", secret_values)
    _persist_redacted_log(stderr_raw, attempt / "stderr.log", secret_values)

    task_result_path = work_dir / "result.json"
    try:
        if error is None and exit_code == 0:
            if not task_result_path.is_file():
                raise RunnerError("task exited successfully but did not write result.json")
            task_result = load_task_evidence(
                task_result_path,
                allowed_layers=config["preregistration"]["layer_allowlist"],
                allowed_timepoints=config["preregistration"]["timepoint_allowlist"],
            )
            artifacts = promote_task_artifacts(
                work_dir,
                attempt / "artifacts",
                f"attempts/{number:04d}/artifacts",
                task_result,
            )
            status = "completed"
        else:
            task_result = {"metrics": []}
            artifacts = promote_task_artifacts(
                work_dir, attempt / "artifacts", f"attempts/{number:04d}/artifacts"
            )
            status = "failed"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        task_result = {"metrics": []}
        artifacts = (
            promote_task_artifacts(
                work_dir, attempt / "artifacts", f"attempts/{number:04d}/artifacts"
            )
            if not (attempt / "artifacts").exists()
            else []
        )

    result = {
        "schema_version": "jump.run-result/v1",
        "status": status,
        "attempt": number,
        "started_at": started["started_at"],
        "finished_at": utc_now(),
        "duration_seconds": round(duration, 6),
        "exit_code": exit_code,
        "error": error,
        "metrics": task_result.get("metrics", []),
        "artifacts": artifacts,
        "provenance": {
            "manifest_sha256": manifest_sha256,
            "run_id": run["id"],
            "code_version": code_version(),
        },
    }
    for key, value in task_result.items():
        if key not in result and key not in {"artifacts", "provenance", "status"}:
            result[key] = value
    retry = run.get("retry", {})
    retry_codes = retry.get("retry_on_exit_codes")
    may_retry_code = retry_codes is None or exit_code in retry_codes
    result["retryable"] = status != "completed" and number < retry.get("max_attempts", 1) and may_retry_code
    validate_json_schema(result, "run-result-v1.schema.json")
    write_json_immutable(attempt / "result.json", result)
    shutil.rmtree(work_dir)
    _write_hash_manifest(attempt)

    if status == "completed" or number >= retry.get("max_attempts", 1) or not may_retry_code:
        write_json_immutable(final_path, result)
        _write_hash_manifest(run_root_path)
    return result


def _gate_value(gate: dict[str, Any], results: list[dict[str, Any]]) -> tuple[float | None, int]:
    values: list[float] = []
    filters = gate.get("filters", {})
    for result in results:
        for metric in result.get("metrics", []):
            if metric.get("name") != gate.get("metric"):
                continue
            if any(metric.get(key) != value for key, value in filters.items()):
                continue
            values.append(float(metric["value"]))
    if not values:
        return None, 0
    aggregation = gate.get("aggregation", "mean")
    if aggregation == "mean":
        return sum(values) / len(values), len(values)
    if aggregation == "min":
        return min(values), len(values)
    if aggregation == "max":
        return max(values), len(values)
    raise ManifestError(f"unsupported gate aggregation: {aggregation}")


def _evaluate_gate(gate: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    actual, samples = _gate_value(gate, results)
    operator = gate["operator"]
    threshold = gate["threshold"]
    operations = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    passed = actual is not None and operations[operator](actual, threshold)
    return {
        "id": gate["id"],
        "metric": gate["metric"],
        "aggregation": gate.get("aggregation", "mean"),
        "filters": gate.get("filters", {}),
        "operator": operator,
        "threshold": threshold,
        "actual": actual,
        "samples": samples,
        "passed": passed,
        "evaluated_at": utc_now(),
    }


def run_manifest(
    manifest: dict[str, Any], runs_dir: str | Path, *, smoke: bool = False, executor: RunExecutor = execute_local_run
) -> dict[str, Any]:
    """Run phases and runs serially. Completed runs are always skipped."""
    validate_manifest(manifest)
    mode = "smoke" if smoke else "full"
    root = run_root(Path(runs_dir), manifest, mode)
    root.mkdir(parents=True, exist_ok=True)
    write_json_immutable(root / "manifest.json", manifest)
    manifest_sha = manifest_hash(manifest)
    phase_states: dict[str, str] = {}
    stopped_reason: str | None = None

    for phase in manifest["phases"]:
        selected = [resolve_run(manifest.get("defaults", {}), run) for run in phase["runs"]]
        if smoke:
            selected = [run for run in selected if run.get("smoke_test", False)]
        if not selected:
            continue
        existing_phase_result = root / "phases" / phase["id"] / "result.json"
        if existing_phase_result.exists():
            prior = json.loads(existing_phase_result.read_text())
            phase_states[phase["id"]] = prior["status"]
            if prior["status"] != "passed":
                stopped_reason = f"phase {phase['id']} previously failed"
                break
            continue
        if any(phase_states.get(dependency) != "passed" for dependency in phase.get("depends_on", [])):
            stopped_reason = f"dependency gate blocked phase {phase['id']}"
            break
        phase_dir = root / "phases" / phase["id"]
        phase_dir.mkdir(parents=True, exist_ok=True)
        phase_results: list[dict[str, Any]] = []
        elapsed = 0.0
        actual_cost = 0.0
        rates = phase["budget"]["gpu_hourly_cost_usd"]
        failed = False
        for run in selected:
            run_path = phase_dir / "runs" / run["id"]
            existing = run_path / "result.json"
            if existing.exists():
                result = json.loads(existing.read_text())
                attempt_results = [result]
            else:
                remaining = phase["budget"]["max_runtime_seconds"] - elapsed
                if run["resources"]["timeout_seconds"] > remaining:
                    stopped_reason = f"runtime ceiling reached before {run['id']}"
                    failed = True
                    break
                enriched_phase = dict(phase)
                enriched_phase["_preregistration"] = manifest["preregistration"]
                result = executor(enriched_phase, run, run_path, manifest_sha)
                attempt_results = [result]
                while result.get("retryable"):
                    result = executor(enriched_phase, run, run_path, manifest_sha)
                    attempt_results.append(result)
            new_duration = sum(float(item.get("duration_seconds", 0)) for item in attempt_results)
            elapsed += new_duration
            actual_cost += new_duration / 3600 * rates[run["resources"]["gpu"]]
            phase_results.append(result)
            if actual_cost > phase["budget"]["max_cost_usd"] + 1e-9:
                stopped_reason = f"cost ceiling reached in phase {phase['id']}"
                failed = True
                break
            if result.get("status") != "completed":
                stopped_reason = f"run {run['id']} failed after {result.get('attempt')} attempt(s)"
                failed = True
                break
        gate_results = [] if failed else [_evaluate_gate(gate, phase_results) for gate in phase.get("gates", [])]
        phase_state = "passed" if not failed and all(gate["passed"] for gate in gate_results) else "failed"
        phase_states[phase["id"]] = phase_state
        summary = {
            "schema_version": "jump.phase-result/v1",
            "phase_id": phase["id"],
            "status": phase_state,
            "runtime_seconds": round(elapsed, 6),
            "estimated_cost_usd": round(actual_cost, 6),
            "gates": gate_results,
        }
        write_json_immutable(phase_dir / "result.json", summary)
        if phase_state == "failed":
            failed_gate = next((gate["id"] for gate in gate_results if not gate["passed"]), None)
            stopped_reason = stopped_reason or f"gate {failed_gate} failed"
            break

    final = {
        "schema_version": "jump.experiment-status/v1",
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": manifest_sha,
        "mode": mode,
        "phase_states": phase_states,
        "status": "stopped" if stopped_reason else "completed",
        "stopped_reason": stopped_reason,
        "updated_at": utc_now(),
    }
    # Status is a mutable checkpoint by design; immutable evidence remains below phases/runs.
    atomic_write(root / "status.json", canonical_json(final))
    return final


def read_status(manifest: dict[str, Any], runs_dir: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    root = run_root(Path(runs_dir), manifest, "smoke" if smoke else "full")
    status_path = root / "status.json"
    if status_path.exists():
        return json.loads(status_path.read_text())
    completed = 0
    failed = 0
    for result_path in root.glob("phases/*/runs/*/result.json"):
        result = json.loads(result_path.read_text())
        completed += result.get("status") == "completed"
        failed += result.get("status") == "failed"
    return {"status": "not_started", "completed_runs": completed, "failed_runs": failed, "path": str(root)}
