"""Allowlisted executor task for the frozen authentic Track H Stage C run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .authentic_stage_c import run_stage_c, stage_c_launch_spec


def run_task(parameters: dict[str, Any], output_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    if set(parameters) != {"expected_manifest_sha256", "expected_code_sha", "dry_run"}:
        raise ValueError("Stage C task parameters do not match the frozen allowlist")
    return run_stage_c(
        output_root=output_dir,
        checkpoint_root=checkpoint_dir,
        expected_manifest_sha256=parameters["expected_manifest_sha256"],
        expected_code_sha=parameters["expected_code_sha"],
        experiment_spec=stage_c_launch_spec(),
        device="cpu" if parameters["dry_run"] is True else "cuda",
        dry_run=parameters["dry_run"] is True,
        precreated_empty_output_root=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    parameters = json.loads(Path(args.parameters).read_text())
    run_task(parameters, Path(args.output_dir), Path(args.checkpoint_dir))


if __name__ == "__main__":
    main()
