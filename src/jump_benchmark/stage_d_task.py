"""Canonical runner task for the frozen authentic Stage D pilot."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from typing import Any

from .authentic_stage_d import (
    STAGE_D_MANIFEST_SHA256,
    WORLD_DECODER_SHA256,
    WORLD_ENCODER_SHA256,
    WORLD_SOURCE_REPO_ID,
    WORLD_SOURCE_REVISION,
    train_stage_d,
)


def run_task(parameters: dict[str, Any], output_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    if set(parameters) != {"expected_manifest_sha256", "expected_code_sha"}:
        raise ValueError("Stage D task parameters do not match frozen allowlist")
    if parameters["expected_manifest_sha256"] != STAGE_D_MANIFEST_SHA256:
        raise ValueError("Stage D task manifest mismatch")
    from huggingface_hub import hf_hub_download
    source = checkpoint_dir / "verified-world-source"
    source.mkdir(parents=True, exist_ok=False)
    for name in ("encoder.safetensors", "decoder.safetensors"):
        resolved = hf_hub_download(
            WORLD_SOURCE_REPO_ID,
            f"stage-b/{name}",
            revision=WORLD_SOURCE_REVISION,
        )
        shutil.copyfile(resolved, source / name)
    return train_stage_d(
        world_component_root=source,
        world_binding={"encoder_sha256": WORLD_ENCODER_SHA256, "decoder_sha256": WORLD_DECODER_SHA256},
        output_root=output_dir,
        expected_manifest_sha256=parameters["expected_manifest_sha256"],
        expected_code_sha=parameters["expected_code_sha"],
    )


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--parameters",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--checkpoint-dir",required=True)
    args=parser.parse_args(); run_task(json.loads(Path(args.parameters).read_text()),Path(args.output_dir),Path(args.checkpoint_dir))

if __name__ == "__main__": main()
