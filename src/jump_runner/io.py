from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .errors import ImmutableOutputError, ManifestError


def canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"value is not strict RFC JSON: {exc}") from exc
    return (encoded + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_immutable(path: Path, content: bytes) -> None:
    """Create an immutable-by-contract file, accepting an identical replay."""
    if path.exists():
        if path.read_bytes() == content:
            return
        raise ImmutableOutputError(f"refusing to overwrite immutable file: {path}")
    atomic_write(path, content)


def write_json_immutable(path: Path, value: Any) -> None:
    write_immutable(path, canonical_json(value))


def load_data(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {source}: {exc}") from exc
    try:
        if source.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    except Exception as exc:
        raise ManifestError(f"cannot parse manifest {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be an object")
    return data
