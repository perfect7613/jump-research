from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from jump_contracts import artifact_declaration, write_task_evidence
from .long_horizon_stage_b import SOURCE_RELATIVE_ROOT,prepare_executor_output_root,train_and_evaluate

def run_task(parameters:dict[str,Any],output_dir:Path,checkpoint_dir:Path):
    allowed={"expected_manifest_sha256","expected_code_sha"}
    if set(parameters) not in (allowed, allowed|{"dry_run"}):raise ValueError("Phase B task parameters mismatch")
    if parameters.get("dry_run") is True:
        prepare_executor_output_root(output_dir,expected_manifest_sha256=parameters["expected_manifest_sha256"],expected_code_sha=parameters["expected_code_sha"])
        marker=output_dir/"cpu-preflight.json"
        marker.write_text(json.dumps({"code_sha":parameters["expected_code_sha"],"manifest_sha256":parameters["expected_manifest_sha256"],"runner_work_root_verified":True},sort_keys=True,separators=(",",":"))+"\n")
        return write_task_evidence(output_dir,metrics=[{"name":"cpu_preflight","value":1.0}],artifacts=[artifact_declaration(marker,output_dir,role="cpu-preflight")],track_h={"phase":"B","scientific_evidence":False})
    return train_and_evaluate(source_root=Path("/jump-runs")/SOURCE_RELATIVE_ROOT,output_root=output_dir,expected_manifest_sha256=parameters["expected_manifest_sha256"],expected_code_sha=parameters["expected_code_sha"])

def main():
    p=argparse.ArgumentParser();p.add_argument("--parameters",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--checkpoint-dir",required=True);a=p.parse_args();run_task(json.loads(Path(a.parameters).read_text()),Path(a.output_dir),Path(a.checkpoint_dir))
if __name__=="__main__":main()
