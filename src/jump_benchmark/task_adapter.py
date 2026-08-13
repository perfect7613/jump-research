"""Canonical task-evidence adapter for bounded Track H producers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jump_contracts import artifact_declaration, write_task_evidence

from .canonical import sha256_json
from .experiment_spec import validate_experiment_spec


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def _role(path: Path) -> str:
    name = path.name
    if name == "terminal.json" or name == "aggregate-terminal.json":
        return "terminal-result"
    if name == "result.json":
        return "task-evidence"
    if name == "encoder.safetensors":
        return "world-encoder"
    if name == "decoder.safetensors":
        return "learned-decoder"
    if name == "world-latent.f32le.bin":
        return "world-latent"
    if name == "encoder-observation.f32le.bin":
        return "encoder-observation"
    if name == "predicted-from-z.svg":
        return "learned-decoder-output"
    if name in {"learned-latent-evidence.json", "sealed-result.json"}:
        return "learned-latent-evidence"
    if name == "SHA256SUMS.json":
        return "checksum-manifest"
    if "probe" in name:
        return "posthoc-probe-evidence"
    if "swap" in name:
        return "latent-swap-evidence"
    if "manifest" in name:
        return "training-manifest"
    if "dataset" in name:
        return "generated-dataset"
    return "benchmark-evidence"


def write_track_h_task_evidence(
    output_dir: str | Path,
    *,
    metrics: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any],
    experiment_spec: Mapping[str, Any],
    terminal_name: str = "terminal.json",
    track_h: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a bound terminal artifact and canonical ``jump.task-evidence/v1``.

    The adapter owns the ExperimentSpec binding, checksum manifest, artifact
    declarations, roles, and final result file.  Domain producers may create
    artifacts first but must not hand-write ``result.json``.
    """
    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError("Track H evidence root must already exist")
    if terminal_name not in {"terminal.json", "aggregate-terminal.json"}:
        raise ValueError("terminal_name is outside the bounded Track H allowlist")
    if (root / "result.json").exists():
        raise FileExistsError("canonical task result already exists")
    plan = validate_experiment_spec(dict(experiment_spec))
    experiment_spec_sha256 = sha256_json(plan)
    if "experiment_id" in terminal or "experiment_spec_sha256" in terminal:
        raise ValueError("ExperimentSpec binding fields are owned by the task adapter")
    bound_terminal = {
        **dict(terminal),
        "experiment_id": plan["experiment_id"],
        "experiment_spec_sha256": experiment_spec_sha256,
    }
    _write_new_json(root / terminal_name, bound_terminal)
    checksum_path = root / "SHA256SUMS.json"
    if checksum_path.exists():
        raise FileExistsError("checksum manifest already exists")
    checksums = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"result.json", "SHA256SUMS.json"}
    }
    _write_new_json(checksum_path, checksums)
    artifacts = [
        artifact_declaration(path, root, role=_role(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "result.json"
    ]
    return write_task_evidence(
        root,
        metrics=metrics,
        artifacts=artifacts,
        experiment_id=plan["experiment_id"],
        experiment_spec_sha256=experiment_spec_sha256,
        terminal_artifact=terminal_name,
        track_h=dict(track_h or {}),
    )
