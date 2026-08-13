"""Content-addressed evidence shared by tasks, runners, and read-only clients.

Tasks own files only while they are in an attempt's mutable work directory.  A
task-evidence declaration binds names and media types to those exact bytes.  An
executor verifies the declaration before promoting the files to immutable run
storage.  Cached clients use the same hashes when reading the resulting
``jump.run-result/v1`` object.

The old ``{"metrics": [...]}`` task result remains supported.  It is treated as
legacy, and the executor discovers all files with an opaque media type.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

TASK_EVIDENCE_VERSION = "jump.task-evidence/v1"
RUN_RESULT_VERSION = "jump.run-result/v1"
SEALED_RESULT_VERSION = "jump.sealed-result/v1"
_REQUIRED_ARTIFACT_FIELDS = frozenset({"name", "path", "sha256", "media_type"})
_RESULT_SOURCES = frozenset({"cached", "live"})


class EvidenceError(ValueError):
    """Raised when evidence cannot be trusted without weakening the contract."""


def artifact_declaration(
    path: str | Path,
    root: str | Path,
    *,
    name: str | None = None,
    media_type: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Describe an existing artifact and bind the declaration to its bytes."""
    if _REQUIRED_ARTIFACT_FIELDS & set(metadata):
        raise EvidenceError("artifact metadata cannot replace contract fields")
    root_path = Path(root).resolve()
    candidate = Path(path)
    if candidate.is_symlink():
        raise EvidenceError("artifact must not be a symlink")
    artifact_path = candidate.resolve()
    try:
        relative = artifact_path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise EvidenceError("artifact must be inside the evidence root") from exc
    relative = _relative_artifact_path(relative)
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise EvidenceError(f"artifact must be a regular non-symlink file: {relative}")
    record: dict[str, Any] = {
        "name": name or relative,
        "path": relative,
        "sha256": sha256_file(artifact_path),
        "media_type": media_type or mimetypes.guess_type(relative)[0] or "application/octet-stream",
    }
    record.update(metadata)
    _validate_artifact_record(record, where="artifact declaration")
    return record


def write_task_evidence(
    output_dir: str | Path,
    *,
    metrics: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]] = (),
    **fields: Any,
) -> dict[str, Any]:
    """Validate and immutably write a versioned task ``result.json``.

    All files in ``output_dir`` other than ``result.json`` must be declared.
    This exact-coverage rule prevents a producer from accidentally creating an
    unlabelled cache, latent, or answer that the executor silently publishes.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if "schema_version" in fields or "metrics" in fields or "artifacts" in fields:
        raise EvidenceError("schema_version, metrics, and artifacts are owned by the evidence module")
    result = {
        "schema_version": TASK_EVIDENCE_VERSION,
        "metrics": [dict(metric) for metric in metrics],
        "artifacts": [dict(artifact) for artifact in artifacts],
        **fields,
    }
    _validate_metrics(result["metrics"])
    _validate_declared_artifacts(root, result["artifacts"], require_exact_coverage=True)
    _write_immutable(root / "result.json", _canonical_json(result) + b"\n")
    return result


def load_task_evidence(
    result_path: str | Path,
    *,
    allowed_layers: Iterable[Any] | None = None,
    allowed_timepoints: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Load task output, preserving the legacy metrics-only interface."""
    path = Path(result_path)
    result = _read_json_object(path)
    version = result.get("schema_version")
    if version not in (None, TASK_EVIDENCE_VERSION):
        raise EvidenceError(f"unsupported task evidence schema_version: {version!r}")
    metrics = result.get("metrics", [])
    _validate_metrics(
        metrics,
        allowed_layers=set(allowed_layers) if allowed_layers is not None else None,
        allowed_timepoints=set(allowed_timepoints) if allowed_timepoints is not None else None,
    )
    if version == TASK_EVIDENCE_VERSION:
        if "artifacts" not in result:
            raise EvidenceError("versioned task evidence requires artifacts")
        _validate_declared_artifacts(path.parent, result["artifacts"], require_exact_coverage=True)
    elif "artifacts" in result:
        raise EvidenceError("artifact declarations require jump.task-evidence/v1")
    return result


def promote_task_artifacts(
    work_dir: str | Path,
    artifact_dir: str | Path,
    path_prefix: str,
    task_evidence: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Verify and copy task files into a new immutable attempt directory."""
    source_root = Path(work_dir)
    target_root = Path(artifact_dir)
    prefix = _relative_artifact_path(f"{path_prefix.rstrip('/')}/artifact-placeholder")
    prefix = str(PurePosixPath(prefix).parent)
    if target_root.exists():
        raise EvidenceError(f"artifact destination already exists: {target_root}")

    declared: dict[str, dict[str, Any]] | None = None
    if task_evidence and task_evidence.get("schema_version") == TASK_EVIDENCE_VERSION:
        records = task_evidence.get("artifacts")
        _validate_declared_artifacts(source_root, records, require_exact_coverage=True)
        declared = {record["path"]: dict(record) for record in records}

    promoted: list[dict[str, Any]] = []
    target_root.mkdir(parents=True)
    try:
        for source, relative in _artifact_files(source_root):
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=False)
            if declared is None:
                record = {
                    "name": relative,
                    "path": relative,
                    "sha256": sha256_file(target),
                    "media_type": "application/octet-stream",
                }
            else:
                record = dict(declared[relative])
                if sha256_file(target) != record["sha256"]:
                    raise EvidenceError(f"artifact changed while being promoted: {relative}")
            record["path"] = f"{prefix}/{relative}"
            promoted.append(record)
        return promoted
    except Exception:
        shutil.rmtree(target_root)
        raise


def load_verified_run_evidence(
    result_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
    require_completed: bool = True,
) -> dict[str, Any]:
    """Read a run result only after its provenance and artifact bytes verify.

    The default artifact root is the directory containing the terminal
    ``result.json``.  Callers reading an attempt-level result can pass the run
    root explicitly because its artifact paths are run-root relative.
    """
    path = Path(result_path)
    result = _read_json_object(path)
    if result.get("schema_version") != RUN_RESULT_VERSION:
        raise EvidenceError(f"schema_version must be {RUN_RESULT_VERSION}")
    if require_completed and result.get("status") != "completed":
        raise EvidenceError("run evidence is not completed")
    _validate_metrics(result.get("metrics"))
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise EvidenceError("run evidence requires provenance")
    manifest_sha = provenance.get("manifest_sha256")
    if not _is_sha256(manifest_sha):
        raise EvidenceError("provenance.manifest_sha256 must be a lowercase SHA-256")
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise EvidenceError("run evidence does not match the expected manifest")
    for key in ("run_id", "code_version"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise EvidenceError(f"provenance.{key} must be a nonempty string")

    records = result.get("artifacts")
    if not isinstance(records, list):
        raise EvidenceError("run evidence artifacts must be an array")
    root = Path(artifact_root) if artifact_root is not None else path.parent
    _validate_declared_artifacts(root, records, require_exact_coverage=False)
    return result


def read_verified_artifact(
    result_path: str | Path,
    name: str,
    *,
    artifact_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Return one named artifact only if the result and returned bytes verify."""
    result_path = Path(result_path)
    result = load_verified_run_evidence(
        result_path,
        artifact_root=artifact_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    matches = [record for record in result["artifacts"] if record["name"] == name]
    if len(matches) != 1:
        raise EvidenceError(f"artifact name must select exactly one record: {name!r}")
    record = matches[0]
    root = Path(artifact_root) if artifact_root is not None else result_path.parent
    source = _safe_artifact(root, record["path"])
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise EvidenceError(f"artifact hash mismatch while reading: {record['path']}")
    return dict(record), content


def load_cached_result_envelope(
    result_path: str | Path,
    name: str,
    *,
    checkpoint_id: str,
    artifact_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Turn one verified JSON run artifact into the shared result envelope."""
    result_path = Path(result_path)
    result = load_verified_run_evidence(
        result_path,
        artifact_root=artifact_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    matches = [record for record in result["artifacts"] if record["name"] == name]
    if len(matches) != 1:
        raise EvidenceError(f"artifact name must select exactly one record: {name!r}")
    record = matches[0]
    if record["media_type"] not in {"application/json", "application/vnd.api+json"}:
        raise EvidenceError("cached result artifact must declare a JSON media type")
    root = Path(artifact_root) if artifact_root is not None else result_path.parent
    source = _safe_artifact(root, record["path"])
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise EvidenceError(f"artifact hash mismatch while reading: {record['path']}")
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cached result artifact is not valid JSON: {exc}") from exc
    provenance = result["provenance"]
    return seal_result_envelope(
        payload,
        source="cached",
        manifest_sha256=provenance["manifest_sha256"],
        run_id=provenance["run_id"],
        code_version=provenance["code_version"],
        checkpoint_id=checkpoint_id,
    )


def seal_result_envelope(
    payload: Mapping[str, Any],
    *,
    source: str,
    manifest_sha256: str,
    run_id: str,
    code_version: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Seal one JSON result for identical cached and live consumption.

    The payload owns domain invariants such as latent/decoder equality and
    answer/image hashes. This outer envelope binds those bytes to an execution
    source and immutable manifest, run, code, and checkpoint identities.
    """
    normalized_payload = _normalize_json_object(payload, "sealed result payload")
    envelope = {
        "schema_version": SEALED_RESULT_VERSION,
        "source": source,
        "payload": normalized_payload,
        "payload_sha256": _sha256_bytes(_canonical_json(normalized_payload)),
        "provenance": {
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "code_version": code_version,
            "checkpoint_id": checkpoint_id,
        },
    }
    return _verify_result_envelope(envelope)


def open_result_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_source: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Verify an in-memory cached/live envelope and return a payload copy."""
    verified = _verify_result_envelope(envelope)
    if expected_source is not None and verified["source"] != expected_source:
        raise EvidenceError("sealed result source does not match the expected source")
    provenance = verified["provenance"]
    if (
        expected_manifest_sha256 is not None
        and provenance["manifest_sha256"] != expected_manifest_sha256
    ):
        raise EvidenceError("sealed result does not match the expected manifest")
    if expected_checkpoint_id is not None and provenance["checkpoint_id"] != expected_checkpoint_id:
        raise EvidenceError("sealed result does not match the expected checkpoint")
    return dict(verified["payload"])


def _validate_metrics(
    metrics: Any,
    *,
    allowed_layers: set[Any] | None = None,
    allowed_timepoints: set[Any] | None = None,
) -> None:
    if not isinstance(metrics, list):
        raise EvidenceError("task result.metrics must be a list")
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict) or not isinstance(metric.get("name"), str) or not metric["name"]:
            raise EvidenceError(f"task result metric {index} needs a name")
        value = metric.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvidenceError(f"task result metric {metric['name']} needs a numeric value")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise EvidenceError(f"task result metric {metric['name']} must be finite")
        if allowed_layers is not None and "layer" in metric and metric["layer"] not in allowed_layers:
            raise EvidenceError(f"result emitted non-preregistered layer {metric['layer']!r}")
        if (
            allowed_timepoints is not None
            and "timepoint" in metric
            and metric["timepoint"] not in allowed_timepoints
        ):
            raise EvidenceError(f"result emitted non-preregistered timepoint {metric['timepoint']!r}")


def _verify_result_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    value = _normalize_json_object(envelope, "sealed result envelope")
    required = {"schema_version", "source", "payload", "payload_sha256", "provenance"}
    if set(value) != required:
        raise EvidenceError(f"sealed result envelope must contain exactly {sorted(required)}")
    if value["schema_version"] != SEALED_RESULT_VERSION:
        raise EvidenceError(f"schema_version must be {SEALED_RESULT_VERSION}")
    if value["source"] not in _RESULT_SOURCES:
        raise EvidenceError(f"sealed result source must be one of {sorted(_RESULT_SOURCES)}")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise EvidenceError("sealed result payload must be an object")
    if not _is_sha256(value["payload_sha256"]):
        raise EvidenceError("payload_sha256 must be a lowercase SHA-256")
    if _sha256_bytes(_canonical_json(payload)) != value["payload_sha256"]:
        raise EvidenceError("sealed result payload hash mismatch")
    provenance = value["provenance"]
    provenance_fields = {"manifest_sha256", "run_id", "code_version", "checkpoint_id"}
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        raise EvidenceError(
            f"sealed result provenance must contain exactly {sorted(provenance_fields)}"
        )
    if not _is_sha256(provenance["manifest_sha256"]):
        raise EvidenceError("sealed result manifest_sha256 must be a lowercase SHA-256")
    for key in ("run_id", "code_version", "checkpoint_id"):
        if not isinstance(provenance[key], str) or not provenance[key]:
            raise EvidenceError(f"sealed result {key} must be a nonempty string")
    return value


def _validate_declared_artifacts(
    root: Path, records: Any, *, require_exact_coverage: bool
) -> None:
    if not isinstance(records, list):
        raise EvidenceError("artifacts must be an array")
    paths: set[str] = set()
    names: set[str] = set()
    for index, record in enumerate(records):
        _validate_artifact_record(record, where=f"artifact {index}")
        relative = _relative_artifact_path(record["path"])
        if relative in paths:
            raise EvidenceError(f"duplicate artifact path: {relative}")
        if record["name"] in names:
            raise EvidenceError(f"duplicate artifact name: {record['name']}")
        paths.add(relative)
        names.add(record["name"])
        source = _safe_artifact(root, relative)
        if not source.is_file() or source.is_symlink():
            raise EvidenceError(f"declared artifact is missing or unsafe: {relative}")
        if sha256_file(source) != record["sha256"]:
            raise EvidenceError(f"artifact hash mismatch: {relative}")
    if require_exact_coverage:
        actual = {relative for _path, relative in _artifact_files(root)}
        undeclared = sorted(actual - paths)
        missing = sorted(paths - actual)
        if undeclared:
            raise EvidenceError(f"undeclared artifact files: {', '.join(undeclared)}")
        if missing:
            raise EvidenceError(f"declared artifact files are missing: {', '.join(missing)}")


def _validate_artifact_record(record: Any, *, where: str) -> None:
    if not isinstance(record, dict) or not _REQUIRED_ARTIFACT_FIELDS <= set(record):
        raise EvidenceError(f"{where} requires name, path, sha256, and media_type")
    if not isinstance(record["name"], str) or not record["name"]:
        raise EvidenceError(f"{where} name must be a nonempty string")
    _relative_artifact_path(record["path"])
    if not _is_sha256(record["sha256"]):
        raise EvidenceError(f"{where} sha256 must be a lowercase SHA-256")
    if not isinstance(record["media_type"], str) or not record["media_type"]:
        raise EvidenceError(f"{where} media_type must be a nonempty string")


def _relative_artifact_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceError("artifact path must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in ("", ".", "..") for part in path.parts):
        raise EvidenceError(f"artifact path is not canonical and relative: {value!r}")
    if value == "result.json":
        raise EvidenceError("result.json cannot declare itself as an artifact")
    return value


def _artifact_files(root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    if not root.exists():
        return files
    for source in sorted(root.rglob("*")):
        relative = source.relative_to(root).as_posix()
        if source.is_symlink():
            raise EvidenceError(f"task artifacts may not be symlinks: {relative}")
        if source.is_file() and relative != "result.json":
            files.append((source, relative))
    return files


def _safe_artifact(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    source = root_resolved / relative
    if source.is_symlink():
        raise EvidenceError(f"declared artifact is a symlink: {relative}")
    try:
        source.resolve().relative_to(root_resolved)
    except ValueError as exc:
        raise EvidenceError(f"declared artifact escapes its root: {relative}") from exc
    cursor = source.parent
    while cursor != root_resolved and cursor != cursor.parent:
        if cursor.is_symlink():
            raise EvidenceError(f"declared artifact has a symlink parent: {relative}")
        cursor = cursor.parent
    return source


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read evidence JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError("evidence JSON must be an object")
    return value


def _normalize_json_object(value: Mapping[str, Any], where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{where} must be an object")
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{where} must be finite canonical JSON: {exc}") from exc
    if not isinstance(normalized, dict):  # defensive; the input is already a mapping
        raise EvidenceError(f"{where} must be an object")
    return normalized


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise EvidenceError(f"immutable evidence already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
