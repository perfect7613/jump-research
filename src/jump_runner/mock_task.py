"""Tiny deterministic task implementing the runner subprocess protocol."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text())
    if parameters.get("sleep_seconds"):
        time.sleep(float(parameters["sleep_seconds"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fail_until = int(parameters.get("fail_until_attempt", 0))
    actual_attempt = len(list(args.checkpoint_dir.parent.parent.glob("*/started.json")))
    (args.checkpoint_dir / "progress.json").write_text(json.dumps({"step": 1}) + "\n")
    if actual_attempt <= fail_until:
        print(f"intentional mock failure on attempt {actual_attempt}")
        return 7
    for key in parameters.get("echo_env", []):
        print(f"{key}={os.environ.get(key, '')}")
    if parameters.get("stderr_text"):
        print(parameters["stderr_text"], file=sys.stderr)
    if parameters.get("traceback_secret"):
        try:
            raise RuntimeError(parameters["traceback_secret"])
        except RuntimeError:
            traceback.print_exc()
    metric = {
        "name": parameters.get("metric_name", "accuracy"),
        "value": parameters.get("metric_value", 1.0),
        "split": parameters.get("split", "smoke"),
        "condition": parameters.get("condition", "mock"),
        "checkpoint_id": parameters.get("checkpoint_id", "mock-cpu"),
    }
    if parameters.get("layers") or "result_layer" in parameters:
        metric["layer"] = parameters.get("result_layer", parameters["layers"][0])
    if parameters.get("timepoints") or "result_timepoint" in parameters:
        metric["timepoint"] = parameters.get("result_timepoint", parameters["timepoints"][0])
    (args.output_dir / "evidence.txt").write_text("deterministic CPU smoke artifact\n")
    (args.output_dir / "result.json").write_text(json.dumps({"metrics": [metric]}) + "\n")
    print("mock task completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
