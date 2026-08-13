"""Deterministic synthetic inputs used for CPU integration tests and runner smoke tests."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

PACKAGED_FIXTURE_SHA256 = "3e8b78a5ddf0a628a40e904682c7e65c1db2a78d5c7997bada3a6f6255588339"


def load_fixture() -> dict[str, Any]:
    """Load only the packaged, content-addressed CPU fixture."""
    path = Path(__file__).parent / "fixtures" / "synthetic_experiment.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("packaged synthetic fixture must be a regular non-symlink file")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != PACKAGED_FIXTURE_SHA256:
        raise ValueError("packaged synthetic fixture content hash mismatch")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("packaged synthetic fixture must be a JSON object")
    return value
