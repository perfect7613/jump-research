from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from .long_horizon_stage_b import SOURCE_RELATIVE_ROOT,train_and_evaluate

def run_task(parameters:dict[str,Any],output_dir:Path,checkpoint_dir:Path):
    if set(parameters)!={"expected_manifest_sha256","expected_code_sha"}:raise ValueError("Phase B task parameters mismatch")
    return train_and_evaluate(source_root=Path("/jump-runs")/SOURCE_RELATIVE_ROOT,output_root=output_dir,expected_manifest_sha256=parameters["expected_manifest_sha256"],expected_code_sha=parameters["expected_code_sha"])

def main():
    p=argparse.ArgumentParser();p.add_argument("--parameters",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--checkpoint-dir",required=True);a=p.parse_args();run_task(json.loads(Path(a.parameters).read_text()),Path(a.output_dir),Path(a.checkpoint_dir))
if __name__=="__main__":main()
