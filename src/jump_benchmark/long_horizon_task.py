"""Canonical subprocess entrypoint for longer-horizon Phase A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .long_horizon import run_long_horizon


def run_task(parameters: dict[str, Any], output_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    if set(parameters) != {"mode", "expected_manifest_sha256", "expected_code_sha", "dry_run"}:
        raise ValueError("long-horizon task parameters do not match allowlist")
    return run_long_horizon(
        mode=parameters["mode"],
        output_root=output_dir,
        expected_manifest_sha256=parameters["expected_manifest_sha256"],
        expected_code_sha=parameters["expected_code_sha"],
        device="cpu" if parameters["dry_run"] else "cuda",
        dry_run=parameters["dry_run"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    run_task(json.loads(Path(args.parameters).read_text()), Path(args.output_dir), Path(args.checkpoint_dir))


if __name__ == "__main__":
    main()
