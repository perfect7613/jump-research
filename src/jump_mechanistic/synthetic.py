"""Deterministic synthetic inputs used for CPU integration tests and runner smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_fixture(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "fixtures" / "synthetic_experiment.json"
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
