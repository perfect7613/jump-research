"""Canonical task adapter for the general visual world-model pilot."""
import argparse
import json
import os
from pathlib import Path
from typing import Any

from jump_contracts import write_task_evidence

from .general_world_model import MANIFEST_SHA256, cpu_preflight, train_and_evaluate


def run(parameters: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if set(parameters) != {"expected_manifest_sha256", "expected_code_sha"}:
        raise ValueError("general world-model parameters mismatch")
    if os.environ.get("JUMP_GENERAL_WORLD_MODEL_TASK_PREFLIGHT") == "1":
        if parameters != {
            "expected_manifest_sha256": MANIFEST_SHA256,
            "expected_code_sha": os.environ.get("JUMP_CODE_VERSION"),
        }:
            raise ValueError("general world-model dry task identity mismatch")
        seam = cpu_preflight()
        return write_task_evidence(
            output_dir,
            metrics=[{"name": "tiny_overfit_improved", "value": float(seam["tiny_overfit_improved"])}],
            artifacts=[],
            general_world_model={"dry_run": True, "seam": seam},
        )
    return train_and_evaluate(output_root=output_dir, **parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    parameters = json.loads(Path(args.parameters).read_text())
    output_dir = Path(args.output_dir)
    run(parameters, output_dir)


if __name__ == "__main__":
    main()
