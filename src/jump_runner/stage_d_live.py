"""Secure live transport for the authentic Stage D engineering pipeline."""

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

CANONICAL_REPO_ID = "Perfect7613/jump-world-model"
CANONICAL_REPO_REVISION = "d197b3825a37e95dfa7d50144fab3c18b6a7fd39"
COMPONENT_MANIFEST_SHA256 = "04f4b4ea6c7f4e6d517cd5a27925ed948bcde1d5d744f84edcf1ee8cdbe890bb"
CHECKPOINT_ID = "stage-d-13c3d963b9ec7171f5d138a9e737b4b6294d542d0887dfbf9a52c2efba422071"
LIVE_SCHEMA_HEAD = "jump.experiment-run/v1"
MAX_BODY_BYTES = 2048
CLAIM_LABEL = (
    "engineering-only Stage D live run; Stage C pivot and Stage D null: "
    "own-z equaled no-z and donor shift was zero; no informative-z, behavioral, causal, or mechanistic claim"
)


def _materialized_component_root(cache_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            CANONICAL_REPO_ID,
            revision=CANONICAL_REPO_REVISION,
            cache_dir=cache_root / "hub",
            local_dir=cache_root / "distribution" / CANONICAL_REPO_REVISION,
        )
    )
    return root


def _verified_distribution(cache_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from jump_contracts import (
        build_world_model_load_record,
        validate_world_model_component_manifest,
        verify_world_model_component_files,
    )

    root = _materialized_component_root(cache_root)
    manifest = validate_world_model_component_manifest(
        json.loads((root / "components.json").read_text())
    )
    if manifest["manifest_sha256"] != COMPONENT_MANIFEST_SHA256:
        raise RuntimeError("live component manifest mismatch")
    verify_world_model_component_files(manifest, root)
    load_record = build_world_model_load_record(
        manifest,
        root,
        expected_repository_revision=CANONICAL_REPO_REVISION,
        resolved_repository_revision=CANONICAL_REPO_REVISION,
        mode="gated_gemma",
    )
    return root, manifest, load_record


def _load_runtime(cache_root: Path, commit_cache: Callable[[], None]) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    from jump_benchmark.authentic import build_gated_residual_projector, build_world_modules
    from jump_benchmark.authentic_stage_d import (
        BASE_REPO_ID,
        BASE_REVISION,
        assert_prompt_identity,
        freeze_base,
        verify_projector_compatibility,
    )
    root, manifest, load_record = _verified_distribution(cache_root)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_REPO_ID, revision=BASE_REVISION, trust_remote_code=False
    )
    assert_prompt_identity(tokenizer)
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_REPO_ID,
        revision=BASE_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to("cuda")
    freeze_base(model)
    model.eval()
    model.config.use_cache = True
    hidden_size = int(model.config.text_config.hidden_size)

    encoder, decoder = build_world_modules()
    encoder.load_state_dict(load_file(root / "encoder/model.safetensors"), strict=True)
    decoder.load_state_dict(load_file(root / "decoder/model.safetensors"), strict=True)
    encoder.to("cuda").eval()
    decoder.to("cuda").eval()
    for parameter in list(encoder.parameters()) + list(decoder.parameters()):
        parameter.requires_grad_(False)

    projector = build_gated_residual_projector(hidden_size).to(device="cuda", dtype=torch.bfloat16)
    projector.projector.load_state_dict(
        {"weight": load_file(root / "future_projector/model.safetensors")["projector.weight"]},
        strict=True,
    )
    gate = load_file(root / "gemma_adapter/adapter_model.safetensors")["gate"]
    with torch.no_grad():
        projector.gate.copy_(gate.to(device="cuda", dtype=torch.bfloat16))
    verify_projector_compatibility(projector, hidden_size=hidden_size)
    projector.eval()
    commit_cache()
    return {
        "root": root,
        "manifest": manifest,
        "load_record": load_record,
        "tokenizer": tokenizer,
        "model": model,
        "encoder": encoder,
        "decoder": decoder,
        "projector": projector,
    }


def _live_result(runtime: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    import torch

    from jump_benchmark.authentic import (
        deserialize_latent_tensor,
        render_predicted_state_svg,
        serialize_latent_tensor,
    )
    from jump_benchmark.authentic_stage_d import (
        STAGE_D_MANIFEST_SHA256,
        WORLD_DECODER_CODE_VERSION,
        WORLD_TRAINING_MANIFEST_SHA256,
        _encode_episode,
        _generate,
        _stage_d_target,
    )
    from jump_benchmark.canonical import sha256_json
    from jump_benchmark.experiment_spec import (
        compile_experiment_intent,
        materialize_experiment,
        validate_experiment_run,
    )
    from jump_contracts import (
        build_learned_latent_evidence,
        canonical_json,
        learned_decoder_identity,
        seal_learned_latent_result,
    )
    from jump_mechanistic.scoring import score_episode

    started = time.time()
    plan = compile_experiment_intent(request)
    compiled = materialize_experiment(plan)
    worlds = compiled["worlds"]
    if plan["template_id"] == "world-swap":
        source, recipient = worlds[0], worlds[1]
        swap_direction = "A_to_B"
    else:
        source = recipient = worlds[0]
        swap_direction = "own"

    z, observation = _encode_episode(runtime["encoder"], source, "cuda")
    serialized = serialize_latent_tensor(z[0])
    exact_z = deserialize_latent_tensor(serialized, device="cuda")
    answer = _generate(
        runtime["model"], runtime["tokenizer"], runtime["projector"], exact_z.unsqueeze(0), enabled=True
    )
    with torch.no_grad():
        predicted = runtime["decoder"](exact_z.unsqueeze(0))[0]
    svg = render_predicted_state_svg(
        predicted.detach().cpu().tolist(), serialized.sha256
    ).encode("utf-8")
    run_id = "live-" + uuid.uuid4().hex
    spec_sha = sha256_json(plan)
    # Targets remain sealed server-side and are consulted only after generation.
    structured_score = score_episode(answer, _stage_d_target(recipient["target"]))
    producer_bindings = {
        "experiment_id": plan["experiment_id"],
        "experiment_spec_sha256": spec_sha,
        "encoder_artifact_sha256": runtime["manifest"]["components"]["encoder"]["weights"]["sha256"],
        "encoder_training_manifest_sha256": WORLD_TRAINING_MANIFEST_SHA256,
        "encoder_architecture": runtime["manifest"]["components"]["encoder"]["architecture"],
        "source_observation_sha256": observation.sha256(),
        "canonical_repo_id": CANONICAL_REPO_ID,
        "canonical_repo_revision": CANONICAL_REPO_REVISION,
        "component_manifest_sha256": COMPONENT_MANIFEST_SHA256,
        "swap_direction": swap_direction,
        "claim_label": CLAIM_LABEL,
        "engineering_only": True,
        "live_ready": False,
    }
    evidence = build_learned_latent_evidence(
        encoder_output=serialized.data,
        decoder_input=bytes(serialized.data),
        injection_input=memoryview(serialized.data),
        encoder_observation=observation.bytes(),
        encoder_observation_artifact_name=f"{run_id}-encoder-observation.f32le.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[16],
        order="C",
        recipient_world_id=recipient["episode_id"],
        donor_world_id=source["episode_id"] if source is not recipient else None,
        world_pair_id=compiled["experiment_id"],
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder/model.safetensors",
            artifact_sha256=runtime["manifest"]["components"]["decoder"]["weights"]["sha256"],
            training_manifest_sha256=WORLD_TRAINING_MANIFEST_SHA256,
            code_version=WORLD_DECODER_CODE_VERSION,
            architecture="same-z-16d-to-six-object-next-position-v1",
        ),
        decoded_image=svg,
        decoded_image_media_type="image/svg+xml",
        answer={"structured_answer": answer, "producer_bindings": producer_bindings},
        tensor_artifact_name=f"{run_id}-world-latent.f32le.bin",
    )
    sealed = seal_learned_latent_result(
        evidence,
        source="live",
        manifest_sha256=STAGE_D_MANIFEST_SHA256,
        run_id=run_id,
        code_version=os.environ["JUMP_CODE_VERSION"],
        checkpoint_id=CHECKPOINT_ID,
    )
    result = {
        "experiment_id": plan["experiment_id"],
        "experiment_spec_sha256": spec_sha,
        "sealed_result": sealed,
        "decoded_image": {
            "artifact_name": "predicted-from-z.svg",
            "media_type": "image/svg+xml",
            "encoding": "base64",
            "data": base64.b64encode(svg).decode("ascii"),
            "sha256": hashlib.sha256(svg).hexdigest(),
        },
        "presentation": {
            "world_built": f"A deterministic six-object {plan['template_id']} world was generated from the sealed server seed.",
            "model_prediction": evidence["answer"],
            "what_changed": (
                "Donor z was injected unchanged into the matched recipient world."
                if source is not recipient
                else "The observation-only z from this world was injected non-textually."
            ),
            "correctness": {
                "format_valid": True,
                "exact_correct": bool(structured_score["joint_theory_accuracy"] and structured_score["adequacy_correct"]),
                "partition_correct": bool(structured_score["partition_accuracy"]),
                "law_correct": bool(structured_score["full_law_accuracy"]),
                "adequacy_correct": bool(structured_score["adequacy_correct"]),
                "force_score": None,
                "notes": CLAIM_LABEL,
            },
        },
    }
    response = {
        "schema_version": LIVE_SCHEMA_HEAD,
        "status": "completed",
        "live": True,
        "request_id": run_id,
        "plan": plan,
        "result": result,
        "error": None,
    }
    validate_experiment_run(response)
    # Enforce a bounded wire payload before returning it.
    encoded = canonical_json(response)
    if len(encoded) > 256_000:
        raise RuntimeError("live response exceeded the sealed output cap")
    response["result"]["presentation"]["correctness"]["notes"] += (
        f" Elapsed {int((time.time() - started) * 1000)} ms; exact scorer used sealed ground truth after generation."
    )
    return response


def build_live_app(*, cache_root: Path, commit_cache: Callable[[], None]):
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    state: dict[str, Any] = {}
    request_count = {"value": 0}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "available",
            "schema_version": LIVE_SCHEMA_HEAD,
            "manifest_sha256": COMPONENT_MANIFEST_SHA256,
            "checkpoint_id": CHECKPOINT_ID,
            "repository_revision": CANONICAL_REPO_REVISION,
            "engineering_only": True,
            "live_ready": False,
        }

    @app.post("/v1/experiment")
    async def experiment(request: Request) -> dict[str, Any]:
        expected = os.environ.get("JUMP_MODAL_TOKEN", "")
        supplied = request.headers.get("authorization", "")
        if not expected or not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:], expected):
            raise HTTPException(status_code=401, detail="unauthorized")
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            raise HTTPException(status_code=413, detail="request body exceeds cap")
        if request_count["value"] >= 12:
            raise HTTPException(status_code=429, detail="container spend guard reached")
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("request must be a JSON object")
            if not state:
                state.update(_load_runtime(cache_root, commit_cache))
            request_count["value"] += 1
            return _live_result(state, body)
        except HTTPException:
            raise
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def http_boundary_preflight(cache_root: Path, commit_cache: Callable[[], None]) -> dict[str, Any]:
    """Exercise the actual ASGI request boundary without loading weights."""
    from fastapi.testclient import TestClient

    app = build_live_app(
        cache_root=Path("/tmp/preflight-cache"),
        commit_cache=lambda: (_ for _ in ()).throw(RuntimeError("weights must not load")),
    )
    response = TestClient(app).post(
        "/v1/experiment",
        json={
            "schema_version": "jump.experiment-intent/v1",
            "intent": "Predict where the six objects move next.",
            "session_id": "preflight",
            "seed": 1,
            "max_steps": 4,
        },
    )
    if response.status_code != 401:
        raise RuntimeError(f"live HTTP boundary did not reject unauthenticated input: {response.status_code}")
    root, manifest, load_record = _verified_distribution(cache_root)
    commit_cache()
    return {
        "status": "passed",
        "http_status": 401,
        "weights_loaded": False,
        "gpu_allocated": False,
        "component_root_is_symlink": root.is_symlink(),
        "component_manifest_sha256": manifest["manifest_sha256"],
        "load_record_sha256": load_record["load_record_sha256"],
    }
