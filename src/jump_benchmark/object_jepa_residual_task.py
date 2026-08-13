from __future__ import annotations
import argparse,json
from pathlib import Path
from .object_jepa_residual import train_and_evaluate
def run_task(parameters,output_dir:Path,checkpoint_dir:Path):
    if set(parameters)!={"expected_manifest_sha256","expected_code_sha"}:raise ValueError("residual JEPA parameters mismatch")
    return train_and_evaluate(output_dir,parameters["expected_manifest_sha256"],parameters["expected_code_sha"])
def main():
    p=argparse.ArgumentParser();p.add_argument("--parameters",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--checkpoint-dir",required=True);a=p.parse_args();run_task(json.loads(Path(a.parameters).read_text()),Path(a.output_dir),Path(a.checkpoint_dir))
if __name__=="__main__":main()
