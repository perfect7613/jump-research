"""Canonical task adapter for the general visual world-model pilot."""
from pathlib import Path
from typing import Any

from .general_world_model import train_and_evaluate


def run(parameters: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if set(parameters) != {"expected_manifest_sha256", "expected_code_sha"}:
        raise ValueError("general world-model parameters mismatch")
    return train_and_evaluate(output_dir=output_dir, **parameters)
