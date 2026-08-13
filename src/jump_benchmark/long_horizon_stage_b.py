"""Phase B: inject the verified longer-horizon z into frozen Gemma."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from jump_contracts import (
    artifact_declaration,
    build_learned_latent_evidence,
    canonical_json,
    learned_decoder_identity,
    seal_learned_latent_result,
    seal_result_envelope,
    tensor_bytes_sha256,
    write_task_evidence,
)
from jump_mechanistic.stage_d import (
    STAGE_D_ARMS,
    STAGE_D_EXECUTION_CONTRACT_SHA256,
    StageDArmInput,
    StageDControlSpec,
    build_stage_d_control_result,
    execute_stage_d_control_set,
    identity_injection,
    no_z_injection,
    scrambled_injection,
)

from .authentic import (
    ObservationArtifact,
    build_gated_residual_projector,
    independent_law,
    independent_partition,
    matched_world_pair,
    render_predicted_state_svg,
    serialize_visible_observations,
)
from .authentic_stage_d import (
    BASE_REPO_ID,
    BASE_REVISION,
    TRANSFORMERS_REVISION,
    _generate,
    _stage_d_target,
    _target_text,
    _teacher_loss,
    assert_prompt_identity,
    freeze_base,
)
from .canonical import sha256_json
from .long_horizon import (
    FUTURE_HORIZON,
    LATENT_DIM,
    REPLICATION_MANIFEST_SHA256,
    build_long_horizon_modules,
    serialize_long_horizon_latent,
)
from .simulator import EpisodeSpec, SimulatorConfig, derive_seed, generate_episode


STAGE_B_VERSION = "jump.track-h-long-horizon-stage-b-manifest/v1"
SOURCE_CODE_SHA = "8d89a0909028538bf38cdc06274fd8fb3638cc6a"
SOURCE_SEED = 120731
SOURCE_ENCODER_SHA256 = "a13497f51e1ee9cedd13a7a55ca526406c1ae5a084129f149bd786bf746f19ee"
SOURCE_DECODER_SHA256 = "a2d03ee1dcbd5b5dede8bba02274995b68da8f294be0123cc75d0266bfcda3c6"
SOURCE_RELATIVE_ROOT = (
    "authentic-world-long-horizon/"
    + REPLICATION_MANIFEST_SHA256
    + "/run/attempts/0001/artifacts/seed-120731"
)
RATE_USD_PER_HOUR = 3.9492


def stage_b_manifest() -> dict[str, Any]:
    return {
        "schema_version": STAGE_B_VERSION,
        "experiment_id": "track-h-long-horizon-stage-b-gemma-injection-v1",
        "execution_lineage": {
            "state": "recovery",
            "recovery_of": {
                "prior_manifest_sha256": "0fb190497453563cfe7ff231de2cfa90e097597ab99a9a1c856944d8ff0c0617",
                "failed_call_ids": ["fc-01KZXMQ7PHREXV0R73GASW2E7W"],
                "partial_inventory_sha256": "298bb4905504b37f9eeb0fb77fd421ebec1d3bb372abc8a5b2c41bcfca818239",
                "failure_reason": "canonical executor precreated the empty work directory but producer incorrectly required mkdir(exist_ok=False) after training",
                "source_outputs_reused": False,
                "source_root_mutated": False,
            },
            "prior_failed_partials": [
                {
                    "manifest_sha256": "00e9457c867b0f2bc7447a8ee59ca4c115d3077a23fd267289a7d2fa467c24bc",
                    "call_id": "fc-01KZXKWCAJ90TG4WY6YNNK3M7X",
                    "partial_inventory_sha256": "f403eace892042c891e4c04050bf1137b8f6044b9dd4e94fcd19f2046de2e1ae",
                },
                {
                    "manifest_sha256": "12779307505f4bc9abf8542acb54cc4ead86c749c504e2974e6f145bf731c231",
                    "call_id": "fc-01KZXMBB0J4W584XNAANZTC415",
                    "partial_inventory_sha256": "44c8ce9de35c5029cfeb33c996603f4f3b7e3f29be4e20ca39a22b60225e94df",
                },
            ],
        },
        "claim_label": "Phase B paired frozen-Gemma injection engineering study; no causal or mechanistic claim",
        "world_model": {
            "source_stage_a_manifest_sha256": REPLICATION_MANIFEST_SHA256,
            "source_code_sha": SOURCE_CODE_SHA,
            "source_seed": SOURCE_SEED,
            "encoder_sha256": SOURCE_ENCODER_SHA256,
            "decoder_sha256": SOURCE_DECODER_SHA256,
            "latent_dtype": "float32-le",
            "latent_shape": [LATENT_DIM],
            "latent_order": "C",
            "encoder_decoder_frozen": True,
            "same_z_only_verified": True,
        },
        "base_model": {
            "repo_id": BASE_REPO_ID,
            "revision": BASE_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "frozen": True,
            "trust_remote_code": False,
        },
        "training": {
            "seed": 140731,
            "episodes": 512,
            "steps": 800,
            "learning_rate": 0.0002,
            "objective": "teacher-forced exact canonical structured-answer cross entropy",
            "trainable": ["Linear(32,3840,bias=False)", "scalar tanh gate"],
            "base_encoder_decoder_frozen": True,
        },
        "injection": {
            "site": "input embedding residual at first token on every autoregressive forward",
            "nontextual": True,
            "z_in_prompt": False,
            "prompt_tokens_identical": True,
        },
        "evaluation": {
            "pairs": 32,
            "pair_seed_root": 33173,
            "arms": list(STAGE_D_ARMS),
            "primary_controls": ["no_z", "scrambled_z", "wrong_world_z"],
            "swap_directions": ["A_to_B", "B_to_A"],
            "execution_contract_sha256": STAGE_D_EXECUTION_CONTRACT_SHA256,
            "paired_bootstrap_replicates": 10000,
            "bootstrap_seed": 44017,
            "pass": {
                "own_minus_each_control_joint_exact_pp_min": 0.05,
                "paired_ci_lower_exclusive": 0.0,
                "parse_delta_abs_max": 0.02,
                "required_all_controls": True,
            },
        },
        "execution": {
            "modal_function": "authentic_world_long_horizon_stage_b",
            "resource": "H100",
            "gpu_count": 1,
            "max_containers": 1,
            "max_inputs": 1,
            "max_attempts": 1,
            "timeout_seconds": 7200,
            "h100_rate_usd_per_hour": RATE_USD_PER_HOUR,
            "forecast_usd": 7.8984,
            "aggregate_authority_ceiling_usd": 100.0,
        },
        "claims": {"informative_z": False, "behavioral": False, "causal": False, "mechanistic": False},
    }


STAGE_B_MANIFEST_SHA256 = sha256_json(stage_b_manifest())


def prepare_executor_output_root(
    output_root: Path, *, expected_manifest_sha256: str, expected_code_sha: str
) -> None:
    """Accept only the canonical runner-owned, precreated, empty work directory."""
    if output_root.is_symlink() or not output_root.is_dir():
        raise FileExistsError("Phase B requires an existing non-symlink runner work directory")
    attempt = output_root.parent
    run_root = attempt.parent.parent
    expected_paths = {
        "JUMP_OUTPUT_DIR": output_root,
        "JUMP_CHECKPOINT_DIR": attempt / "checkpoint",
        "JUMP_PARAMETERS_PATH": attempt / "task-parameters.json",
    }
    for key, path in expected_paths.items():
        configured = os.environ.get(key)
        if configured is None or Path(configured).resolve(strict=True) != path.resolve(strict=True):
            raise RuntimeError(f"Phase B runner path binding mismatch: {key}")
    if (
        output_root.name != "work"
        or attempt.parent.name != "attempts"
        or len(attempt.name) != 4
        or not attempt.name.isdigit()
        or os.environ.get("JUMP_RUN_ID") != "long-horizon-stage-b"
        or os.environ.get("JUMP_PHASE_ID") != "long-horizon-stage-b"
        or os.environ.get("JUMP_CODE_VERSION") != expected_code_sha
    ):
        raise RuntimeError("Phase B runner identity/path mismatch")
    for path in (output_root, attempt, attempt.parent, run_root):
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise RuntimeError("Phase B runner root ownership or symlink check failed")
    config = json.loads((run_root / "config.json").read_text())
    started = json.loads((attempt / "started.json").read_text())
    parameters = json.loads((attempt / "task-parameters.json").read_text())
    if (
        config.get("manifest_sha256") != expected_manifest_sha256
        or config.get("code_version") != expected_code_sha
        or config.get("run_id") != "long-horizon-stage-b"
        or config.get("phase_id") != "long-horizon-stage-b"
        or started.get("manifest_sha256") != expected_manifest_sha256
        or parameters.get("expected_manifest_sha256") != expected_manifest_sha256
        or parameters.get("expected_code_sha") != expected_code_sha
    ):
        raise RuntimeError("Phase B canonical runner identity files mismatch")
    if any(output_root.iterdir()):
        raise FileExistsError("immutable Phase B runner work directory is nonempty")


def dynamic_32d_control_preflight() -> dict[str, Any]:
    """Exercise the actual six-arm executor with independently sealed 32D tensors."""
    import struct
    import torch

    from .authentic_stage_d import persistent_z_injection

    checkpoint = "stage-b-" + "4" * 64
    worlds = {"a": "world-a", "b": "world-b", "w": "world-wrong"}
    pair_id = "stage-b-preflight-pair"
    target_a = {"partition": [0, 0, 0, 1, 1, 1], "replacement_law": {"same": "attract", "different": "repel", "exponent": 1}, "adequate": True}
    target_b = {"partition": [0, 1, 1, 0, 0, 1], "replacement_law": {"same": "repel", "different": "attract", "exponent": 2}, "adequate": True}
    raw = {
        key: b"".join(struct.pack("<f", float(offset + index + 1)) for index in range(LATENT_DIM))
        for offset, key in ((0, "a"), (100, "b"), (200, "w"))
    }

    def learned(key: str, recipient: str, donor: str | None, answer: dict[str, Any]):
        evidence = build_learned_latent_evidence(
            encoder_output=raw[key], decoder_input=bytes(raw[key]), injection_input=memoryview(raw[key]),
            encoder_observation=bytes(384), encoder_observation_artifact_name=f"{key}.f32le.bin",
            encoder_observation_media_type="application/octet-stream", dtype="float32-le",
            shape=[LATENT_DIM], order="C", recipient_world_id=recipient, donor_world_id=donor,
            world_pair_id=pair_id, tensor_artifact_name=f"{key}-z.f32le.bin",
            learned_decoder=learned_decoder_identity(
                artifact_name="decoder/model.safetensors", artifact_sha256="1" * 64,
                training_manifest_sha256=REPLICATION_MANIFEST_SHA256, code_version=SOURCE_CODE_SHA,
                architecture="same-z-32d-eight-step-residual-rollout-v3",
            ), decoded_image=b"<svg/>", decoded_image_media_type="image/svg+xml",
            answer={"structured_answer": answer, "producer_bindings": {"encoder_artifact_sha256": SOURCE_ENCODER_SHA256}},
        )
        return seal_learned_latent_result(
            evidence, source="cached", manifest_sha256=STAGE_B_MANIFEST_SHA256,
            run_id=f"preflight-{key}-{recipient}", code_version="3" * 40, checkpoint_id=checkpoint,
        )

    own = learned("a", worlds["a"], None, target_a)
    wrong = learned("w", worlds["a"], worlds["w"], target_a)
    a2b = learned("a", worlds["b"], worlds["a"], target_a)
    b2a = learned("b", worlds["a"], worlds["b"], target_b)
    indices = list(reversed(range(LATENT_DIM)))
    scrambled_binding, raw_s = scrambled_injection(
        own, raw["a"], tensor_artifact_name="scrambled-z.f32le.bin", seed=33173, indices=indices
    )
    answers = {arm: target_a for arm in STAGE_D_ARMS}; answers["swap_b_to_a"] = target_b
    materials = {
        "own_z": (worlds["a"], worlds["a"], own, raw["a"], raw["a"], identity_injection(own, raw["a"])),
        "no_z": (worlds["a"], None, None, None, None, no_z_injection()),
        "scrambled_z": (worlds["a"], worlds["a"], own, raw["a"], raw_s, scrambled_binding),
        "wrong_world_z": (worlds["a"], worlds["w"], wrong, raw["w"], raw["w"], identity_injection(wrong, raw["w"])),
        "swap_a_to_b": (worlds["b"], worlds["a"], a2b, raw["a"], raw["a"], identity_injection(a2b, raw["a"])),
        "swap_b_to_a": (worlds["a"], worlds["b"], b2a, raw["b"], raw["b"], identity_injection(b2a, raw["b"])),
    }
    arm_inputs = {}
    for arm in STAGE_D_ARMS:
        recipient, source, latent, source_raw, injected_raw, injection = materials[arm]
        payload = build_stage_d_control_result(
            arm_id=arm, checkpoint_id=checkpoint, manifest_sha256=STAGE_B_MANIFEST_SHA256,
            pair_id=pair_id, recipient_world_id=recipient, source_world_id=source,
            answer=answers[arm], injection=injection,
        )
        sealed = seal_result_envelope(
            payload, source="cached", manifest_sha256=STAGE_B_MANIFEST_SHA256,
            run_id=f"preflight-{arm}", code_version="3" * 40, checkpoint_id=checkpoint,
        )
        arm_inputs[arm] = StageDArmInput(sealed, "cached", latent, source_raw, injected_raw)
    spec = StageDControlSpec(
        checkpoint_id=checkpoint, manifest_sha256=STAGE_B_MANIFEST_SHA256, pair_id=pair_id,
        cluster_id="preflight", world_a_id=worlds["a"], world_b_id=worlds["b"],
        wrong_world_id=worlds["w"], world_a_target=target_a, world_b_target=target_b,
    )
    result = execute_stage_d_control_set(spec, arm_inputs)
    result.verify()

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 12)

        def get_input_embeddings(self):
            return self.embedding

    dummy = _DummyModel()
    projector = build_gated_residual_projector(12, latent_dim=LATENT_DIM)
    latent = torch.frombuffer(bytearray(raw["a"]), dtype=torch.float32).reshape(1, LATENT_DIM)
    with persistent_z_injection(dummy, projector, latent, enabled=True) as binding:
        dummy.get_input_embeddings()(torch.tensor([[1, 2]], dtype=torch.long))
        dummy.get_input_embeddings()(torch.tensor([[3]], dtype=torch.long))
    if binding["forward_calls"]["count"] != 2:
        raise RuntimeError("32D persistent injection did not execute on every forward")
    if binding["world_latent_sha256"] != tensor_bytes_sha256(
        raw["a"], dtype="float32-le", shape=[1, LATENT_DIM], order="C"
    ):
        raise RuntimeError("32D persistent injection tensor binding mismatch")
    return {
        "arms": len(result.arms),
        "latent_dim": LATENT_DIM,
        "scrambled_differs": raw_s != raw["a"],
        "persistent_injection_forward_calls": binding["forward_calls"]["count"],
        "persistent_injection_sha256": binding["world_latent_sha256"],
        "content_sha256": result.content_sha256,
    }


def _episode(seed: int, split: str) -> dict[str, Any]:
    return generate_episode(
        EpisodeSpec(
            seed,
            split,
            independent_law(seed),
            True,
            SimulatorConfig(steps=12),
            independent_partition(seed),
        )
    )


def _load_world(root: Path, device: str):
    from safetensors.torch import load_file

    encoder_path = root / "encoder.safetensors"
    decoder_path = root / "decoder.safetensors"
    if hashlib.sha256(encoder_path.read_bytes()).hexdigest() != SOURCE_ENCODER_SHA256:
        raise RuntimeError("Phase B encoder checksum mismatch")
    if hashlib.sha256(decoder_path.read_bytes()).hexdigest() != SOURCE_DECODER_SHA256:
        raise RuntimeError("Phase B decoder checksum mismatch")
    encoder, decoder = build_long_horizon_modules()
    encoder.load_state_dict(load_file(encoder_path), strict=True)
    decoder.load_state_dict(load_file(decoder_path), strict=True)
    encoder.to(device).eval(); decoder.to(device).eval()
    for parameter in [*encoder.parameters(), *decoder.parameters()]:
        parameter.requires_grad_(False)
    return encoder, decoder


def _encode(encoder: Any, episode: dict[str, Any], device: str):
    import torch

    observation = ObservationArtifact.from_payload(serialize_visible_observations(episode))
    tensor = torch.tensor([observation.values], dtype=torch.float32, device=device)
    with torch.no_grad():
        z = encoder(tensor)
    return z, observation


def _envelope(*, decoder: Any, z: Any, observation: ObservationArtifact, recipient: str, donor: str | None, pair_id: str, answer: dict[str, Any], run_id: str, checkpoint_id: str, code_sha: str):
    import torch

    serialized = serialize_long_horizon_latent(z)
    exact_z = torch.frombuffer(bytearray(serialized.data), dtype=torch.float32).to(z.device).reshape(1, LATENT_DIM)
    with torch.no_grad():
        prediction = decoder(exact_z).reshape(FUTURE_HORIZON, 6, 2)
    svg = render_predicted_state_svg(prediction[-1].detach().cpu().tolist(), serialized.sha256).encode()
    evidence = build_learned_latent_evidence(
        encoder_output=serialized.data,
        decoder_input=bytes(serialized.data),
        injection_input=memoryview(bytes(serialized.data)),
        encoder_observation=observation.bytes(),
        encoder_observation_artifact_name=f"{run_id}-observation.f32le.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le", shape=[LATENT_DIM], order="C",
        tensor_artifact_name=f"{run_id}-z.f32le.bin",
        recipient_world_id=recipient,
        donor_world_id=donor,
        world_pair_id=pair_id,
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder/model.safetensors",
            artifact_sha256=SOURCE_DECODER_SHA256,
            training_manifest_sha256=REPLICATION_MANIFEST_SHA256,
            code_version=SOURCE_CODE_SHA,
            architecture="same-z-32d-eight-step-residual-rollout-v3",
        ),
        decoded_image=svg,
        decoded_image_media_type="image/svg+xml",
        answer={"structured_answer": answer, "producer_bindings": {
            "encoder_artifact_sha256": SOURCE_ENCODER_SHA256,
            "encoder_training_manifest_sha256": REPLICATION_MANIFEST_SHA256,
            "source_observation_sha256": observation.sha256(),
            "decoder_external_observation_input": False,
        }},
    )
    sealed = seal_learned_latent_result(
        evidence, source="cached", manifest_sha256=STAGE_B_MANIFEST_SHA256,
        run_id=run_id, code_version=code_sha, checkpoint_id=checkpoint_id,
    )
    return sealed, serialized.data, svg


def _paired_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed); samples=[]
    for _ in range(10000):
        samples.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    samples.sort(); return [samples[250], samples[9749]]


def train_and_evaluate(*, source_root: Path, output_root: Path, expected_manifest_sha256: str, expected_code_sha: str, device: str = "cuda") -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    if expected_manifest_sha256 != STAGE_B_MANIFEST_SHA256 or os.environ.get("JUMP_CODE_VERSION") != expected_code_sha:
        raise ValueError("Phase B immutable identity mismatch before root write")
    prepare_executor_output_root(
        output_root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Phase B requires CUDA")
    torch.manual_seed(stage_b_manifest()["training"]["seed"]); torch.cuda.manual_seed_all(stage_b_manifest()["training"]["seed"]); torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(BASE_REPO_ID,revision=BASE_REVISION,trust_remote_code=False)
    prompt_binding=assert_prompt_identity(tokenizer)
    model=AutoModelForMultimodalLM.from_pretrained(BASE_REPO_ID,revision=BASE_REVISION,torch_dtype=torch.bfloat16,trust_remote_code=False).to(device)
    base_parameters=freeze_base(model); model.config.use_cache=False; model.gradient_checkpointing_enable()
    hidden_size=int(model.config.text_config.hidden_size)
    projector=build_gated_residual_projector(hidden_size, latent_dim=LATENT_DIM).to(device=device,dtype=torch.bfloat16)
    encoder,decoder=_load_world(source_root,device)
    optimizer=torch.optim.AdamW(projector.parameters(),lr=stage_b_manifest()["training"]["learning_rate"])
    losses=[]; model.train(); projector.train()
    for step in range(stage_b_manifest()["training"]["steps"]):
        seed=derive_seed(140731,f"stage-b-train:{step % stage_b_manifest()['training']['episodes']}")
        episode=_episode(seed,"stage-b-train")
        z,_=_encode(encoder,episode,device)
        optimizer.zero_grad(set_to_none=True)
        loss=_teacher_loss(model,tokenizer,projector,z,_target_text(episode["target"]))
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen Gemma became trainable")
    model.eval();projector.eval();model.config.use_cache=True
    for role in ("encoder","decoder","future_projector","gemma_adapter"):(output_root/role).mkdir()
    shutil.copyfile(source_root/"encoder.safetensors",output_root/"encoder/model.safetensors")
    shutil.copyfile(source_root/"decoder.safetensors",output_root/"decoder/model.safetensors")
    save_file({"projector.weight":projector.projector.weight.detach().cpu()},output_root/"future_projector/model.safetensors")
    save_file({"gate":projector.gate.detach().cpu()},output_root/"gemma_adapter/adapter_model.safetensors")
    configs={
        "encoder/config.json":{"architecture":"observation-only structured same-z 32D: 12 last positions plus learned 20D"},
        "decoder/config.json":{"architecture":"same-z-only 32D eight-step residual rollout"},
        "future_projector/config.json":{"input_dim":LATENT_DIM,"output_dim":hidden_size},
        "gemma_adapter/adapter_config.json":{"gate":"scalar_tanh","injection_layer":0,"injection_site":"first_token_every_forward"},
    }
    for path,value in configs.items():(output_root/path).write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
    checkpoint_id="stage-b-"+sha256_json({"projector":hashlib.sha256((output_root/"future_projector/model.safetensors").read_bytes()).hexdigest(),"gate":hashlib.sha256((output_root/"gemma_adapter/adapter_model.safetensors").read_bytes()).hexdigest()})
    matrices=[]
    for index in range(stage_b_manifest()["evaluation"]["pairs"]):
        pair=matched_world_pair(pair_seed=derive_seed(33173,f"stage-b-pair:{index}"),config=SimulatorConfig(steps=12))
        wrong=_episode(derive_seed(33173,f"stage-b-wrong:{index}"),"stage-b-wrong")
        za,oa=_encode(encoder,pair["a"],device);zb,ob=_encode(encoder,pair["b"],device);zw,ow=_encode(encoder,wrong,device)
        raw_a=serialize_long_horizon_latent(za[0]).data;raw_b=serialize_long_horizon_latent(zb[0]).data;raw_w=serialize_long_horizon_latent(zw[0]).data
        permutation=list(range(LATENT_DIM));random.Random(derive_seed(33173,f"stage-b-scramble:{index}")).shuffle(permutation)
        own_answer=_generate(model,tokenizer,projector,za,enabled=True)
        own_env,_,own_svg=_envelope(decoder=decoder,z=za[0],observation=oa,recipient=pair["a"]["episode_id"],donor=None,pair_id=pair["pair_id"],answer=own_answer,run_id=f"stage-b-{index}-own",checkpoint_id=checkpoint_id,code_sha=expected_code_sha)
        scrambled_binding,raw_s=scrambled_injection(own_env,raw_a,tensor_artifact_name=f"stage-b-{index}-scrambled-z.f32le.bin",seed=derive_seed(33173,f"stage-b-scramble:{index}") & 0xFFFFFFFF,indices=permutation)
        zs=torch.frombuffer(bytearray(raw_s),dtype=torch.float32).to(device).reshape(1,LATENT_DIM)
        answers={
            "own_z":own_answer,
            "no_z":_generate(model,tokenizer,projector,za,enabled=False),
            "scrambled_z":_generate(model,tokenizer,projector,zs,enabled=True),
            "wrong_world_z":_generate(model,tokenizer,projector,zw,enabled=True),
            "swap_a_to_b":_generate(model,tokenizer,projector,za,enabled=True),
            "swap_b_to_a":_generate(model,tokenizer,projector,zb,enabled=True),
        }
        wrong_env,_,wrong_svg=_envelope(decoder=decoder,z=zw[0],observation=ow,recipient=pair["a"]["episode_id"],donor=wrong["episode_id"],pair_id=pair["pair_id"],answer=answers["wrong_world_z"],run_id=f"stage-b-{index}-wrong",checkpoint_id=checkpoint_id,code_sha=expected_code_sha)
        a2b_env,_,a2b_svg=_envelope(decoder=decoder,z=za[0],observation=oa,recipient=pair["b"]["episode_id"],donor=pair["a"]["episode_id"],pair_id=pair["pair_id"],answer=answers["swap_a_to_b"],run_id=f"stage-b-{index}-a2b",checkpoint_id=checkpoint_id,code_sha=expected_code_sha)
        b2a_env,_,b2a_svg=_envelope(decoder=decoder,z=zb[0],observation=ob,recipient=pair["a"]["episode_id"],donor=pair["b"]["episode_id"],pair_id=pair["pair_id"],answer=answers["swap_b_to_a"],run_id=f"stage-b-{index}-b2a",checkpoint_id=checkpoint_id,code_sha=expected_code_sha)
        materials={
            "own_z":(pair["a"]["episode_id"],pair["a"]["episode_id"],own_env,raw_a,raw_a,identity_injection(own_env,raw_a)),
            "no_z":(pair["a"]["episode_id"],None,None,None,None,no_z_injection()),
            "scrambled_z":(pair["a"]["episode_id"],pair["a"]["episode_id"],own_env,raw_a,raw_s,scrambled_binding),
            "wrong_world_z":(pair["a"]["episode_id"],wrong["episode_id"],wrong_env,raw_w,raw_w,identity_injection(wrong_env,raw_w)),
            "swap_a_to_b":(pair["b"]["episode_id"],pair["a"]["episode_id"],a2b_env,raw_a,raw_a,identity_injection(a2b_env,raw_a)),
            "swap_b_to_a":(pair["a"]["episode_id"],pair["b"]["episode_id"],b2a_env,raw_b,raw_b,identity_injection(b2a_env,raw_b)),
        }
        arms={};sealed_controls={}
        for arm_id in STAGE_D_ARMS:
            recipient,source,learned,source_raw,injected_raw,injection=materials[arm_id]
            payload=build_stage_d_control_result(arm_id=arm_id,checkpoint_id=checkpoint_id,manifest_sha256=STAGE_B_MANIFEST_SHA256,pair_id=pair["pair_id"],recipient_world_id=recipient,source_world_id=source,answer=answers[arm_id],injection=injection)
            result=seal_result_envelope(payload,source="cached",manifest_sha256=STAGE_B_MANIFEST_SHA256,run_id=f"stage-b-{index}-{arm_id}",code_version=expected_code_sha,checkpoint_id=checkpoint_id)
            sealed_controls[arm_id]=result;arms[arm_id]=StageDArmInput(result,"cached",learned,source_raw,injected_raw)
        spec=StageDControlSpec(checkpoint_id=checkpoint_id,manifest_sha256=STAGE_B_MANIFEST_SHA256,pair_id=pair["pair_id"],cluster_id=f"stage-b-{index}",world_a_id=pair["a"]["episode_id"],world_b_id=pair["b"]["episode_id"],wrong_world_id=wrong["episode_id"],world_a_target=_stage_d_target(pair["a"]["scoring_target"]),world_b_target=_stage_d_target(pair["b"]["scoring_target"]))
        evidence=execute_stage_d_control_set(spec,arms);record={**evidence.unsigned_dict(),"content_sha256":evidence.content_sha256}
        pair_root=output_root/"controls"/f"pair-{index:03d}";pair_root.mkdir(parents=True)
        (pair_root/"matrix.json").write_bytes(canonical_json(record));(pair_root/"own-z.f32le.bin").write_bytes(raw_a);(pair_root/"scrambled-z.f32le.bin").write_bytes(raw_s);(pair_root/"wrong-z.f32le.bin").write_bytes(raw_w);(pair_root/"own-predicted-from-z.svg").write_bytes(own_svg)
        for arm_id,envelope in sealed_controls.items():(pair_root/f"{arm_id}.json").write_bytes(canonical_json(envelope))
        matrices.append(record)
    (output_root/"control-matrices.json").write_bytes(canonical_json(matrices));(output_root/"stage-b-manifest.json").write_bytes(canonical_json(stage_b_manifest()))
    arms_by={arm:[] for arm in STAGE_D_ARMS}
    for matrix in matrices:
        for arm in matrix["arms"]:arms_by[arm["arm_id"]].append(arm)
    own=[float(row["world_a_score"]) for row in arms_by["own_z"]]
    comparisons={}
    for offset,control in enumerate(("no_z","scrambled_z","wrong_world_z")):
        values=[a-float(row["world_a_score"]) for a,row in zip(own,arms_by[control])]
        comparisons[control]={"own_minus_control_mean":sum(values)/len(values),"paired_ci95":_paired_ci(values,44017+offset),"parse_delta":0.0,"passed":sum(values)/len(values)>=0.05 and _paired_ci(values,44017+offset)[0]>0}
    behavioral_pass=all(row["passed"] for row in comparisons.values())
    duration=time.monotonic()-started
    terminal={"status":"completed","phase":"B","decision":"pass" if behavioral_pass else "pivot","manifest_sha256":STAGE_B_MANIFEST_SHA256,"code_sha":expected_code_sha,"checkpoint_id":checkpoint_id,"base_revision":BASE_REVISION,"transformers_revision":TRANSFORMERS_REVISION,"base_parameters":base_parameters,"trainable_parameters":sum(p.numel() for p in projector.parameters()),"initial_loss":losses[0],"final_loss":losses[-1],"prompt_binding":prompt_binding,"pair_count":len(matrices),"comparisons":comparisons,"runtime_seconds":duration,"estimated_cost_usd":duration/3600*RATE_USD_PER_HOUR,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated()),"phase_c_allowed":behavioral_pass,"claims":{"informative_z":behavioral_pass,"behavioral":behavioral_pass,"causal":False,"mechanistic":False},"claim_label":stage_b_manifest()["claim_label"]}
    (output_root/"terminal.json").write_bytes(canonical_json(terminal))
    artifacts=[artifact_declaration(path,output_root,role="stage-b-evidence") for path in sorted(output_root.rglob("*")) if path.is_file()]
    evidence=write_task_evidence(output_root,metrics=[{"name":"initial_loss","value":losses[0]},{"name":"final_loss","value":losses[-1]},*[{"name":"own_minus_control_joint_exact","condition":name,"value":value["own_minus_control_mean"]} for name,value in comparisons.items()]],artifacts=artifacts,track_h={"phase":"B","decision":terminal["decision"],"claims":terminal["claims"]})
    return {**terminal,"task_evidence":evidence}


def run_contract(expected_manifest_sha256: str, expected_code_sha: str):
    phase={"id":"long-horizon-stage-b","_secret_keys":["HF_TOKEN"],"_preregistration":{"layer_allowlist":[0],"timepoint_allowlist":["answer"]}}
    run={"id":"long-horizon-stage-b","task":{"module":"jump_benchmark.long_horizon_stage_b_task","parameters":{"expected_manifest_sha256":expected_manifest_sha256,"expected_code_sha":expected_code_sha}},"resources":{"gpu":"H100","timeout_seconds":7200},"selection":{"layers":[],"timepoints":[]},"retry":{"max_attempts":1}}
    return phase,run
