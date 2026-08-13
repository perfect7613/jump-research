"""Authentic Stage D: non-textual learned-z injection into frozen Gemma."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jump_contracts import (
    build_learned_latent_evidence,
    build_world_model_component_manifest,
    canonical_json as contract_canonical_json,
    component_identity,
    learned_decoder_identity,
    seal_learned_latent_result,
    seal_result_envelope,
    verify_world_model_component_files,
)

from .authentic import (
    LATENT_DIM,
    ObservationArtifact,
    SerializedLatent,
    bind_latent_uses,
    build_gated_residual_projector,
    build_world_modules,
    deserialize_latent_tensor,
    literal_donor_swap,
    matched_injection_prompt,
    matched_world_pair,
    module_content_sha256,
    render_predicted_state_svg,
    serialize_latent_tensor,
)
from .canonical import sha256_json
from jump_mechanistic.scoring import score_episode
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
from .simulator import derive_seed

STAGE_D_SCHEMA_VERSION = "jump.track-h-authentic-stage-d-manifest/v1"
BASE_REPO_ID = "google/gemma-4-12B-it"
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
TRANSFORMERS_REVISION = "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"
WORLD_SOURCE_REPO_ID = "Perfect7613/jump-authentic-world-model"
WORLD_SOURCE_REVISION = "d64d757df1a964cd231a0bdfcf5222b714910c74"
WORLD_ENCODER_SHA256 = "8d57a3106f70783c4617a1d80a39e02ee3d248b4e32f9e3d33bf7e379b583969"
WORLD_DECODER_SHA256 = "1fc74e4a0934f31452e44bca5d1665a0558d7289dbca894fa5989c7c5619453f"
WORLD_TRAINING_MANIFEST_SHA256 = "17f0118898ff37f5debb915921480d055fc850e76f5a0c5045a8e673c2097e94"
WORLD_DECODER_CODE_VERSION = "d910b736e80b4d6717def84a6f4f99de3567f19e"
CONDITIONS = ("own_z", "no_z", "scrambled_z", "wrong_world_z", "swap_a_to_b", "swap_b_to_a")


def stage_d_manifest() -> dict[str, Any]:
    return {
        "schema_version": STAGE_D_SCHEMA_VERSION,
        "experiment_id": "track-h-authentic-stage-d-projector-pilot-v1",
        "claim_label": "Stage D non-textual learned-z engineering pipeline after Stage C null/pivot; no informative-z, causal, behavioral, or mechanistic claim",
        "source_world_model": {
            "policy": "verified canonical component artifacts only",
            "repo_id": WORLD_SOURCE_REPO_ID,
            "revision": WORLD_SOURCE_REVISION,
            "encoder_sha256": WORLD_ENCODER_SHA256,
            "decoder_sha256": WORLD_DECODER_SHA256,
            "training_manifest_sha256": WORLD_TRAINING_MANIFEST_SHA256,
            "decoder_code_version": WORLD_DECODER_CODE_VERSION,
            "required_roles": ["encoder", "decoder"],
            "encoder_decoder_frozen": True,
            "scientific_status": "engineering input only; three-seed Stage C did not beat persistence",
        },
        "base_model": {
            "repo_id": BASE_REPO_ID,
            "revision": BASE_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "frozen": True,
            "trust_remote_code": False,
        },
        "injection": {
            "latent_dtype": "float32-le",
            "latent_shape": [LATENT_DIM],
            "latent_order": "C",
            "projector": "Linear(16,hidden_size,bias=False)",
            "gate": "scalar tanh initialized zero",
            "layer": 0,
            "site": "input_embedding_residual_at_first_token_each_forward",
            "autoregressive_persistence": "embedding hook applies on prompt prefill and every cached decode forward",
            "z_in_prompt": False,
            "prompt_identity_required": True,
        },
        "training": {
            "seed": 77231,
            "episodes": 200,
            "steps": 200,
            "learning_rate": 0.0002,
            "trainable": ["latent_projector", "scalar_gate"],
            "objective": "teacher-forced canonical structured-answer cross entropy",
            "gradient_checkpointing": True,
        },
        "evaluation": {
            "conditions": list(CONDITIONS),
            "execution_contract_sha256": STAGE_D_EXECUTION_CONTRACT_SHA256,
            "decode": {
                "method": "deterministic_constrained_autoregressive_generation",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 128,
                "stop_condition": "grammar_complete",
                "grammar": "jump.track-h-exact-answer-grammar/v1",
                "target_prefix": False,
                "answer_prefix": False,
            },
            "swap_pairs": 8,
            "swap_directions": ["A_to_B", "B_to_A"],
            "exact_scorer": "jump_benchmark.scoring.score_answer",
            "image_source": "learned_decoder_prediction",
        },
        "stop": {
            "nonfinite_loss": True,
            "prompt_token_mismatch": True,
            "latent_hash_mismatch": True,
            "base_parameter_trainable": True,
            "artifact_or_distribution_verification_failure": True,
        },
        "execution": {
            "modal_function": "authentic_world_stage_d",
            "resource": "H100",
            "gpu_count": 1,
            "max_containers": 1,
            "max_inputs": 1,
            "max_attempts": 1,
            "timeout_seconds": 7200,
            "h100_rate_usd_per_hour": 3.9492,
            "retry_aware_forecast_usd": 7.8984,
            "hard_ceiling_usd": 20.0,
            "stage_r_auto_launch": False,
        },
        "claims": {
            "informative_z": False,
            "behavioral": False,
            "causal": False,
            "mechanistic": False,
            "track_r": False,
        },
    }


STAGE_D_MANIFEST_SHA256 = sha256_json(stage_d_manifest())


def prompt_tensors(tokenizer: Any, batch_size: int = 1) -> dict[str, Any]:
    """Materialize the one prompt used by own-z and every control."""
    import torch

    prompt = matched_injection_prompt()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"]
    mask = encoded.get("attention_mask", torch.ones_like(ids))
    if batch_size != ids.shape[0]:
        ids = ids.expand(batch_size, -1).clone()
        mask = mask.expand(batch_size, -1).clone()
    return {"input_ids": ids, "attention_mask": mask}


def assert_prompt_identity(tokenizer: Any) -> dict[str, Any]:
    own = prompt_tensors(tokenizer)
    no_z = prompt_tensors(tokenizer)
    if not own["input_ids"].equal(no_z["input_ids"]) or not own["attention_mask"].equal(no_z["attention_mask"]):
        raise RuntimeError("own-z and no-z prompt tensors differ")
    raw = own["input_ids"].to(device="cpu").contiguous().numpy().tobytes()
    return {"equal": True, "token_sha256": hashlib.sha256(raw).hexdigest(), "z_in_prompt": False}


def inject_exact_z(model: Any, projector: Any, input_ids: Any, attention_mask: Any, latent: Any, *, enabled: bool) -> tuple[Any, dict[str, Any]]:
    """Inject z only through embeddings; text tensors are never modified."""
    import torch

    captured = serialize_latent_tensor(latent)
    exact_z = deserialize_latent_tensor(captured, device=str(input_ids.device))
    embeddings = model.get_input_embeddings()(input_ids)
    if enabled:
        projected_z = exact_z.to(dtype=projector.projector.weight.dtype)
        embeddings = projector(embeddings, projected_z)
    first = attention_mask.to(torch.int64).argmax(dim=1)
    binding = bind_latent_uses(latent, exact_z, exact_z)
    return embeddings, {
        **binding,
        "enabled": enabled,
        "injection_site": "input_embedding_residual_at_first_token_each_forward",
        "injection_layer": 0,
        "first_nonpadding_token": first.tolist(),
    }


@contextmanager
def persistent_z_injection(model: Any, projector: Any, latent: Any, *, enabled: bool):
    """Apply the same byte-roundtripped z on prefill and every cached decode call."""
    captured = serialize_latent_tensor(latent)
    exact_z = deserialize_latent_tensor(captured, device=str(latent.device))
    binding = bind_latent_uses(latent, exact_z, exact_z)
    if not enabled:
        yield {**binding, "enabled": False, "forward_calls": 0}
        return
    calls = {"count": 0}

    def hook(_module: Any, _inputs: Any, output: Any):
        calls["count"] += 1
        return projector(output, exact_z.to(dtype=projector.projector.weight.dtype))

    handle = model.get_input_embeddings().register_forward_hook(hook)
    try:
        yield {**binding, "enabled": True, "forward_calls": calls}
    finally:
        handle.remove()


def controlled_latents(own: Any, wrong: Any, *, seed: int) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(own.shape[-1], generator=generator).to(own.device)
    return {
        "own_z": own,
        "no_z": own,
        "scrambled_z": own[..., permutation],
        "wrong_world_z": wrong,
    }


def freeze_base(model: Any) -> int:
    total = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        total += parameter.numel()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Gemma base was not frozen")
    return total


def verify_projector_compatibility(projector: Any, *, hidden_size: int) -> dict[str, Any]:
    weight = projector.projector.weight
    if tuple(weight.shape) != (hidden_size, LATENT_DIM) or projector.gate.numel() != 1:
        raise ValueError("projector/gate shape mismatch")
    return {
        "input_dim": LATENT_DIM,
        "output_dim": hidden_size,
        "gate_shape": list(projector.gate.shape),
        "state_sha256": module_content_sha256(projector),
    }


def swap_lineage(pair: dict[str, Any], latent_a: Any, latent_b: Any) -> list[dict[str, Any]]:
    a = serialize_latent_tensor(latent_a)
    b = serialize_latent_tensor(latent_b)
    return [
        {"direction": "A_to_B", **literal_donor_swap(pair["b"]["episode_id"], pair["a"]["episode_id"], a)},
        {"direction": "B_to_A", **literal_donor_swap(pair["a"]["episode_id"], pair["b"]["episode_id"], b)},
    ]


def stage_d_cpu_preflight(tokenizer: Any, hidden_size: int = 32) -> dict[str, Any]:
    import torch

    identity = assert_prompt_identity(tokenizer)
    projector = build_gated_residual_projector(hidden_size)
    compatibility = verify_projector_compatibility(projector, hidden_size=hidden_size)
    own = torch.arange(LATENT_DIM, dtype=torch.float32).unsqueeze(0)
    wrong = torch.flip(own, dims=[1])
    controls = controlled_latents(own, wrong, seed=77231)
    grammar = ExactAnswerGrammar(tokenizer)
    pair = matched_world_pair(pair_seed=77231)
    swaps = swap_lineage(pair, own[0], wrong[0])
    if swaps[0]["injected_world_latent_sha256"] != serialize_latent_tensor(own[0]).sha256:
        raise RuntimeError("literal A-to-B donor bytes changed")
    return {
        "status": "passed",
        "manifest_sha256": STAGE_D_MANIFEST_SHA256,
        "prompt": identity,
        "projector": compatibility,
        "control_shapes": {name: list(value.shape) for name, value in controls.items()},
        "grammar": {
            "schema_version": grammar.schema_version,
            "candidate_count": len(grammar.candidates),
            "target_values_inspected": False,
        },
        "swap": swaps,
        "gpu_allocated": False,
    }


def build_distribution_manifest(root: Path, *, hidden_size: int) -> dict[str, Any]:
    def identity(role: str, architecture: str) -> dict[str, Any]:
        directory = root / role
        weights = directory / ("adapter_model.safetensors" if role == "gemma_adapter" else "model.safetensors")
        config = directory / ("adapter_config.json" if role == "gemma_adapter" else "config.json")
        return component_identity(
            directory=role,
            weights_path=weights.relative_to(root).as_posix(),
            weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            config_path=config.relative_to(root).as_posix(),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            architecture=architecture,
        )
    manifest = build_world_model_component_manifest(
        base_model_repo_id=BASE_REPO_ID,
        base_model_revision=BASE_REVISION,
        transformers_revision=TRANSFORMERS_REVISION,
        latent_dtype="float32-le", latent_shape=[LATENT_DIM], latent_order="C",
        projector_input_dim=LATENT_DIM, projector_output_dim=hidden_size,
        projector_gate="scalar_tanh", injection_layer=0,
        injection_site="input_embedding_residual_at_first_token_each_forward",
        encoder=identity("encoder", "observation-only-mlp-96-64-16"),
        decoder=identity("decoder", "same-z-mlp-16-64-12"),
        future_projector=identity("future_projector", "linear-16-hidden-no-bias"),
        gemma_adapter=identity("gemma_adapter", "scalar-tanh-gate"),
        artifact_only_ready=True, end_to_end_injection=True, live_ready=False,
        claim_guards={
            "engineering_only": True,
            "behavioral_claim_allowed": False,
            "mechanistic_claim_allowed": False,
            "causal_claim_allowed": False,
            "benchmark_law_accuracy_claim_allowed": False,
            "track_r_claim_allowed": False,
        },
        claim_label=stage_d_manifest()["claim_label"],
    )
    verify_world_model_component_files(manifest, root)
    return manifest


def _target_text(target: dict[str, Any]) -> str:
    bounded = {
        "partition": target["partition"],
        "replacement_law": target["replacement_law"],
        "adequate": target["adequacy"],
    }
    return json.dumps(bounded, sort_keys=True, separators=(",", ":"))


def _stage_d_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "partition": target["partition"],
        "replacement_law": target["replacement_law"],
        "adequate": target["adequacy"],
    }


class ExactAnswerGrammar:
    """A label-independent finite grammar over every valid Stage D answer."""

    schema_version = "jump.track-h-exact-answer-grammar/v1"

    def __init__(self, tokenizer: Any):
        candidates: dict[tuple[int, ...], dict[str, Any]] = {}
        for mask in range(1, 32):
            partition = [0, *[(mask >> index) & 1 for index in range(5)]]
            for same in ("attract", "repel"):
                for different in ("attract", "repel"):
                    for exponent in (1, 2):
                        for adequate in (False, True):
                            answer = {
                                "partition": partition,
                                "replacement_law": {
                                    "same": same,
                                    "different": different,
                                    "exponent": exponent,
                                },
                                "adequate": adequate,
                            }
                            text = json.dumps(answer, sort_keys=True, separators=(",", ":"))
                            ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
                            if not ids or len(ids) > 128 or ids in candidates:
                                raise RuntimeError("Stage D grammar tokenization is not finite and unique")
                            candidates[ids] = answer
        self.candidates = candidates

    def allowed(self, generated: tuple[int, ...]) -> list[int]:
        allowed = {
            sequence[len(generated)]
            for sequence in self.candidates
            if len(sequence) > len(generated) and sequence[: len(generated)] == generated
        }
        if not allowed:
            raise RuntimeError("Stage D generation left the frozen answer grammar")
        return sorted(allowed)

    def completed(self, generated: tuple[int, ...]) -> bool:
        return generated in self.candidates

    def parse(self, generated: tuple[int, ...]) -> dict[str, Any]:
        if generated not in self.candidates:
            raise ValueError("Stage D output did not complete the exact answer grammar")
        answer = self.candidates[generated]
        score_episode(answer, answer)
        return answer

    def admits(self, answer: dict[str, Any], tokenizer: Any) -> bool:
        text = json.dumps(answer, sort_keys=True, separators=(",", ":"))
        ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
        return self.candidates.get(ids) == answer


def _episode(seed: int, split: str) -> dict[str, Any]:
    from .authentic import independent_law, independent_partition
    from .simulator import EpisodeSpec, SimulatorConfig, generate_episode

    return generate_episode(
        EpisodeSpec(seed, split, independent_law(seed), True, SimulatorConfig(steps=6), independent_partition(seed))
    )


def _load_world_components(root: Path, binding: dict[str, str], device: str):
    from safetensors.torch import load_file

    if set(binding) != {"encoder_sha256", "decoder_sha256"}:
        raise ValueError("world component binding requires exact encoder/decoder hashes")
    encoder_path, decoder_path = root / "encoder.safetensors", root / "decoder.safetensors"
    for path, key in ((encoder_path, "encoder_sha256"), (decoder_path, "decoder_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding[key]:
            raise ValueError(f"verified world component mismatch: {path.name}")
    encoder, decoder = build_world_modules()
    encoder.load_state_dict(load_file(encoder_path)); decoder.load_state_dict(load_file(decoder_path))
    encoder.to(device).eval(); decoder.to(device).eval()
    for parameter in list(encoder.parameters()) + list(decoder.parameters()):
        parameter.requires_grad_(False)
    return encoder, decoder


def _encode_episode(encoder: Any, episode: dict[str, Any], device: str):
    import torch
    from .authentic import ObservationArtifact, serialize_visible_observations

    payload = episode.get("encoder_input") or serialize_visible_observations(episode)
    observation = ObservationArtifact.from_payload(payload)
    tensor = torch.tensor([observation.values], dtype=torch.float32, device=device)
    with torch.no_grad():
        z = encoder(tensor)
    return z, observation


def _learned_envelope(
    *,
    decoder: Any,
    latent: Any,
    observation: ObservationArtifact,
    recipient_world_id: str,
    donor_world_id: str | None,
    pair_id: str,
    answer: dict[str, Any],
    run_id: str,
    checkpoint_id: str,
    code_sha: str,
) -> tuple[dict[str, Any], bytes, bytes]:
    import torch

    serialized = serialize_latent_tensor(latent)
    exact = deserialize_latent_tensor(serialized, device=str(latent.device))
    with torch.no_grad():
        predicted = decoder(exact.unsqueeze(0))[0]
    svg = render_predicted_state_svg(predicted.detach().cpu().tolist(), serialized.sha256).encode()
    evidence = build_learned_latent_evidence(
        encoder_output=serialized.data,
        decoder_input=bytes(serialized.data),
        injection_input=memoryview(serialized.data),
        encoder_observation=observation.bytes(),
        encoder_observation_artifact_name=f"{run_id}-encoder-observation.f32le.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[LATENT_DIM],
        order="C",
        recipient_world_id=recipient_world_id,
        donor_world_id=donor_world_id,
        world_pair_id=pair_id,
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder/model.safetensors",
            artifact_sha256=WORLD_DECODER_SHA256,
            training_manifest_sha256=WORLD_TRAINING_MANIFEST_SHA256,
            code_version=WORLD_DECODER_CODE_VERSION,
            architecture="same-z-16d-to-six-object-next-position-v1",
        ),
        decoded_image=svg,
        decoded_image_media_type="image/svg+xml",
        answer={
            "structured_answer": answer,
            "producer_bindings": {
                "encoder_artifact_sha256": WORLD_ENCODER_SHA256,
                "encoder_training_manifest_sha256": WORLD_TRAINING_MANIFEST_SHA256,
                "encoder_architecture": "observation-only-mlp-96-64-16",
                "source_observation_sha256": observation.sha256(),
            },
        },
        tensor_artifact_name=f"{run_id}-world-latent.f32le.bin",
    )
    envelope = seal_learned_latent_result(
        evidence,
        source="live",
        manifest_sha256=STAGE_D_MANIFEST_SHA256,
        run_id=run_id,
        code_version=code_sha,
        checkpoint_id=checkpoint_id,
    )
    return envelope, serialized.data, svg


def _stage_d_control_matrix(
    *,
    pair_index: int,
    pair: dict[str, Any],
    wrong_episode: dict[str, Any],
    encoder: Any,
    decoder: Any,
    model: Any,
    tokenizer: Any,
    projector: Any,
    checkpoint_id: str,
    code_sha: str,
    device: str,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    import torch

    za, observation_a = _encode_episode(encoder, pair["a"], device)
    zb, observation_b = _encode_episode(encoder, pair["b"], device)
    zwrong, observation_wrong = _encode_episode(encoder, wrong_episode, device)
    raw_a = serialize_latent_tensor(za[0]).data
    raw_b = serialize_latent_tensor(zb[0]).data
    raw_wrong = serialize_latent_tensor(zwrong[0]).data
    permutation_seed = derive_seed(77231, f"stage-d-scramble:{pair_index}") & 0xFFFFFFFF
    permutation = list(range(LATENT_DIM))
    random.Random(permutation_seed).shuffle(permutation)
    if permutation == list(range(LATENT_DIM)):
        permutation = list(reversed(permutation))

    answer_own = _generate(model, tokenizer, projector, za, enabled=True)
    answer_no_z = _generate(model, tokenizer, projector, za, enabled=False)
    own_envelope, _, own_svg = _learned_envelope(
        decoder=decoder, latent=za[0], observation=observation_a,
        recipient_world_id=pair["a"]["episode_id"], donor_world_id=None,
        pair_id=pair["pair_id"], answer=answer_own,
        run_id=f"stage-d-{pair_index}-own", checkpoint_id=checkpoint_id, code_sha=code_sha,
    )
    scrambled_binding, raw_scrambled = scrambled_injection(
        own_envelope,
        raw_a,
        tensor_artifact_name=f"stage-d-{pair_index}-scrambled-world-latent.f32le.bin",
        seed=permutation_seed,
        indices=permutation,
    )
    zscrambled = deserialize_latent_tensor(
        SerializedLatent(
            "float32-le", (LATENT_DIM,), raw_scrambled,
            scrambled_binding["world_latent_sha256"],
        ),
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        scrambled_positions = decoder(zscrambled)[0]
    scrambled_svg = render_predicted_state_svg(
        scrambled_positions.detach().cpu().tolist(),
        scrambled_binding["world_latent_sha256"],
    ).encode()
    answers = {
        "own_z": answer_own,
        "no_z": answer_no_z,
        "scrambled_z": _generate(model, tokenizer, projector, zscrambled, enabled=True),
        "wrong_world_z": _generate(model, tokenizer, projector, zwrong, enabled=True),
        "swap_a_to_b": _generate(model, tokenizer, projector, za, enabled=True),
        "swap_b_to_a": _generate(model, tokenizer, projector, zb, enabled=True),
    }
    wrong_envelope, _, wrong_svg = _learned_envelope(
        decoder=decoder, latent=zwrong[0], observation=observation_wrong,
        recipient_world_id=pair["a"]["episode_id"], donor_world_id=wrong_episode["episode_id"],
        pair_id=pair["pair_id"], answer=answers["wrong_world_z"],
        run_id=f"stage-d-{pair_index}-wrong", checkpoint_id=checkpoint_id, code_sha=code_sha,
    )
    a_to_b_envelope, _, a_to_b_svg = _learned_envelope(
        decoder=decoder, latent=za[0], observation=observation_a,
        recipient_world_id=pair["b"]["episode_id"], donor_world_id=pair["a"]["episode_id"],
        pair_id=pair["pair_id"], answer=answers["swap_a_to_b"],
        run_id=f"stage-d-{pair_index}-a-to-b", checkpoint_id=checkpoint_id, code_sha=code_sha,
    )
    b_to_a_envelope, _, b_to_a_svg = _learned_envelope(
        decoder=decoder, latent=zb[0], observation=observation_b,
        recipient_world_id=pair["a"]["episode_id"], donor_world_id=pair["b"]["episode_id"],
        pair_id=pair["pair_id"], answer=answers["swap_b_to_a"],
        run_id=f"stage-d-{pair_index}-b-to-a", checkpoint_id=checkpoint_id, code_sha=code_sha,
    )
    materials = {
        "own_z": (pair["a"]["episode_id"], pair["a"]["episode_id"], own_envelope, raw_a, raw_a, identity_injection(own_envelope, raw_a)),
        "no_z": (pair["a"]["episode_id"], None, None, None, None, no_z_injection()),
        "scrambled_z": (pair["a"]["episode_id"], pair["a"]["episode_id"], own_envelope, raw_a, raw_scrambled, scrambled_binding),
        "wrong_world_z": (pair["a"]["episode_id"], wrong_episode["episode_id"], wrong_envelope, raw_wrong, raw_wrong, identity_injection(wrong_envelope, raw_wrong)),
        "swap_a_to_b": (pair["b"]["episode_id"], pair["a"]["episode_id"], a_to_b_envelope, raw_a, raw_a, identity_injection(a_to_b_envelope, raw_a)),
        "swap_b_to_a": (pair["a"]["episode_id"], pair["b"]["episode_id"], b_to_a_envelope, raw_b, raw_b, identity_injection(b_to_a_envelope, raw_b)),
    }
    arm_inputs: dict[str, StageDArmInput] = {}
    result_envelopes: dict[str, Any] = {}
    for arm_id in STAGE_D_ARMS:
        recipient, source, learned, source_raw, injected_raw, injection = materials[arm_id]
        payload = build_stage_d_control_result(
            arm_id=arm_id,
            checkpoint_id=checkpoint_id,
            manifest_sha256=STAGE_D_MANIFEST_SHA256,
            pair_id=pair["pair_id"],
            recipient_world_id=recipient,
            source_world_id=source,
            answer=answers[arm_id],
            injection=injection,
        )
        result = seal_result_envelope(
            payload,
            source="live",
            manifest_sha256=STAGE_D_MANIFEST_SHA256,
            run_id=f"stage-d-{pair_index}-{arm_id}",
            code_version=code_sha,
            checkpoint_id=checkpoint_id,
        )
        result_envelopes[arm_id] = result
        arm_inputs[arm_id] = StageDArmInput(
            result_envelope=result,
            expected_source="live",
            learned_latent_envelope=learned,
            source_tensor_bytes=source_raw,
            injected_tensor_bytes=injected_raw,
        )
    spec = StageDControlSpec(
        checkpoint_id=checkpoint_id,
        manifest_sha256=STAGE_D_MANIFEST_SHA256,
        pair_id=pair["pair_id"],
        cluster_id=f"stage-d-pair-{pair_index}",
        world_a_id=pair["a"]["episode_id"],
        world_b_id=pair["b"]["episode_id"],
        wrong_world_id=wrong_episode["episode_id"],
        world_a_target=_stage_d_target(pair["a"]["scoring_target"]),
        world_b_target=_stage_d_target(pair["b"]["scoring_target"]),
    )
    evidence = execute_stage_d_control_set(spec, arm_inputs)
    evidence_dict = {**evidence.unsigned_dict(), "content_sha256": evidence.content_sha256}
    binary = {
        f"stage-d-{pair_index}-own-world-latent.f32le.bin": raw_a,
        f"stage-d-{pair_index}-own-encoder-observation.f32le.bin": observation_a.bytes(),
        f"stage-d-{pair_index}-wrong-world-latent.f32le.bin": raw_wrong,
        f"stage-d-{pair_index}-wrong-encoder-observation.f32le.bin": observation_wrong.bytes(),
        f"stage-d-{pair_index}-a-to-b-world-latent.f32le.bin": raw_a,
        f"stage-d-{pair_index}-a-to-b-encoder-observation.f32le.bin": observation_a.bytes(),
        f"stage-d-{pair_index}-b-to-a-world-latent.f32le.bin": raw_b,
        f"stage-d-{pair_index}-b-to-a-encoder-observation.f32le.bin": observation_b.bytes(),
        f"stage-d-{pair_index}-scrambled-world-latent.f32le.bin": raw_scrambled,
        "world-a-predicted-from-z.svg": own_svg,
        "wrong-world-predicted-from-z.svg": wrong_svg,
        "scrambled-predicted-from-z.svg": scrambled_svg,
        "a-to-b-predicted-from-z.svg": a_to_b_svg,
        "b-to-a-predicted-from-z.svg": b_to_a_svg,
    }
    envelopes = {
        "learned": {
            "own_z": own_envelope,
            "wrong_world_z": wrong_envelope,
            "swap_a_to_b": a_to_b_envelope,
            "swap_b_to_a": b_to_a_envelope,
        },
        "controls": result_envelopes,
    }
    return evidence_dict, binary, envelopes


def _teacher_loss(model: Any, tokenizer: Any, projector: Any, z: Any, target_text: str):
    import torch

    prompt = prompt_tensors(tokenizer)
    target = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=target.dtype)
    target = torch.cat([target, eos], dim=1)
    input_ids = torch.cat([prompt["input_ids"], target], dim=1).to(z.device)
    attention = torch.ones_like(input_ids)
    labels = torch.cat([torch.full_like(prompt["input_ids"], -100), target], dim=1).to(z.device)
    with persistent_z_injection(model, projector, z, enabled=True):
        output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
    if not torch.isfinite(output.loss):
        raise RuntimeError("non-finite Stage D projector loss")
    return output.loss


def _generate(model: Any, tokenizer: Any, projector: Any, z: Any, *, enabled: bool) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    prompt = prompt_tensors(tokenizer)
    input_ids = prompt["input_ids"].to(z.device)
    attention = prompt["attention_mask"].to(z.device)
    grammar = ExactAnswerGrammar(tokenizer)
    prompt_length = input_ids.shape[1]

    class GrammarComplete(StoppingCriteria):
        def __call__(self, current_ids: Any, scores: Any, **kwargs: Any) -> torch.BoolTensor:
            suffix = tuple(int(item) for item in current_ids[0, prompt_length:].tolist())
            return torch.tensor([grammar.completed(suffix)], device=current_ids.device)

    def allowed(_batch_id: int, current_ids: Any) -> list[int]:
        suffix = tuple(int(item) for item in current_ids[prompt_length:].tolist())
        return grammar.allowed(suffix)

    with persistent_z_injection(model, projector, z, enabled=enabled):
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            do_sample=False,
            num_beams=1,
            max_new_tokens=stage_d_manifest()["evaluation"]["decode"]["max_new_tokens"],
            prefix_allowed_tokens_fn=allowed,
            stopping_criteria=StoppingCriteriaList([GrammarComplete()]),
        )
    generated = tuple(int(item) for item in output[0, prompt_length:].tolist())
    return grammar.parse(generated)


def train_stage_d(
    *,
    world_component_root: Path,
    world_binding: dict[str, str],
    output_root: Path,
    expected_manifest_sha256: str,
    expected_code_sha: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train projector/gate only, execute frozen controls, and seal artifacts."""
    import os
    import shutil
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    from jump_contracts import artifact_declaration, write_task_evidence

    if expected_manifest_sha256 != STAGE_D_MANIFEST_SHA256:
        raise ValueError("Stage D manifest mismatch before output root")
    if os.environ.get("JUMP_CODE_VERSION") != expected_code_sha:
        raise ValueError("Stage D requires explicit immutable code identity")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable Stage D output root exists")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage D requires CUDA")
    started = time.monotonic()
    torch.manual_seed(stage_d_manifest()["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_REPO_ID, revision=BASE_REVISION, trust_remote_code=False
    )
    assert_prompt_identity(tokenizer)
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_REPO_ID,
        revision=BASE_REVISION,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to(device)
    base_parameters = freeze_base(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    hidden_size = int(model.config.text_config.hidden_size)
    projector = build_gated_residual_projector(hidden_size)
    verify_projector_compatibility(projector, hidden_size=hidden_size)
    projector = projector.to(device=device, dtype=torch.bfloat16)
    encoder, decoder = _load_world_components(world_component_root, world_binding, device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=stage_d_manifest()["training"]["learning_rate"])
    train_count = stage_d_manifest()["training"]["episodes"]
    steps = stage_d_manifest()["training"]["steps"]
    losses = []
    model.train()
    projector.train()
    for step in range(steps):
        episode = _episode(77231 + (step % train_count), "stage-d-train")
        z, _ = _encode_episode(encoder, episode, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _teacher_loss(model, tokenizer, projector, z, _target_text(episode["target"]))
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Gemma base became trainable")
    model.eval()
    projector.eval()

    output_root.mkdir(parents=True, exist_ok=True)
    for role in ("encoder", "decoder", "future_projector", "gemma_adapter"):
        (output_root / role).mkdir()
    shutil.copyfile(world_component_root / "encoder.safetensors", output_root / "encoder/model.safetensors")
    shutil.copyfile(world_component_root / "decoder.safetensors", output_root / "decoder/model.safetensors")
    save_file(
        {"projector.weight": projector.projector.weight.detach().cpu()},
        output_root / "future_projector/model.safetensors",
    )
    save_file(
        {"gate": projector.gate.detach().cpu()},
        output_root / "gemma_adapter/adapter_model.safetensors",
    )
    checkpoint_id = "stage-d-" + sha256_json({
        "projector": hashlib.sha256((output_root / "future_projector/model.safetensors").read_bytes()).hexdigest(),
        "gate": hashlib.sha256((output_root / "gemma_adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
    })
    configs = {
        "encoder/config.json": {"architecture": "observation-only-mlp-96-64-16"},
        "decoder/config.json": {"architecture": "same-z-mlp-16-64-12"},
        "future_projector/config.json": {"input_dim": LATENT_DIM, "output_dim": hidden_size},
        "gemma_adapter/adapter_config.json": {"gate": "scalar_tanh", "injection_layer": 0, "injection_site": "input_embedding_residual_at_first_token_each_forward"},
    }
    for name, value in configs.items():
        (output_root / name).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    distribution = build_distribution_manifest(output_root, hidden_size=hidden_size)
    (output_root / "components.json").write_text(json.dumps(distribution, sort_keys=True, separators=(",", ":")) + "\n")
    (output_root / "stage-d-manifest.json").write_text(json.dumps(stage_d_manifest(), sort_keys=True, separators=(",", ":")) + "\n")
    control_evidence = []
    for pair_index in range(stage_d_manifest()["evaluation"]["swap_pairs"]):
        pair = matched_world_pair(pair_seed=99231 + pair_index)
        wrong = _episode(109231 + pair_index, "stage-d-wrong-world")
        matrix, binary, envelopes = _stage_d_control_matrix(
            pair_index=pair_index,
            pair=pair,
            wrong_episode=wrong,
            encoder=encoder,
            decoder=decoder,
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            checkpoint_id=checkpoint_id,
            code_sha=expected_code_sha,
            device=device,
        )
        pair_root = output_root / "controls" / f"pair-{pair_index:02d}"
        pair_root.mkdir(parents=True)
        for name, content in binary.items():
            (pair_root / name).write_bytes(content)
        (pair_root / "stage-d-control-evidence.json").write_bytes(contract_canonical_json(matrix))
        for group, records in envelopes.items():
            group_root = pair_root / group
            group_root.mkdir()
            for arm_id, envelope in records.items():
                (group_root / f"{arm_id}.json").write_bytes(contract_canonical_json(envelope))
        control_evidence.append(matrix)
    (output_root / "control-matrices.json").write_bytes(contract_canonical_json(control_evidence))
    duration_seconds = time.monotonic() - started
    metrics = [
        {"name": "initial_loss", "value": losses[0]},
        {"name": "final_loss", "value": losses[-1]},
    ]
    for arm_id in STAGE_D_ARMS:
        arms = [next(arm for arm in matrix["arms"] if arm["arm_id"] == arm_id) for matrix in control_evidence]
        for score_name in ("world_a_score", "world_b_score"):
            metrics.append({
                "name": score_name,
                "condition": arm_id,
                "value": sum(float(arm[score_name]) for arm in arms) / len(arms),
            })
        shifted = [arm["moved_toward_source"] for arm in arms if arm["moved_toward_source"] is not None]
        if shifted:
            metrics.append({
                "name": "moved_toward_source_rate",
                "condition": arm_id,
                "value": sum(float(value) for value in shifted) / len(shifted),
            })
    terminal = {
        "status": "completed", "stage": "D", "manifest_sha256": STAGE_D_MANIFEST_SHA256,
        "code_sha": expected_code_sha, "base_model": {"repo_id": BASE_REPO_ID, "revision": BASE_REVISION},
        "transformers_revision": TRANSFORMERS_REVISION, "base_parameters": base_parameters,
        "trainable_parameters": sum(p.numel() for p in projector.parameters()),
        "checkpoint_id": checkpoint_id,
        "component_manifest_sha256": distribution["manifest_sha256"],
        "execution_contract_sha256": STAGE_D_EXECUTION_CONTRACT_SHA256,
        "control_matrix_count": len(control_evidence),
        "initial_loss": losses[0], "final_loss": losses[-1], "conditions": list(CONDITIONS),
        "swap_directions": ["A_to_B", "B_to_A"], "image_source": "learned_decoder_prediction",
        "duration_seconds": duration_seconds,
        "estimated_cost_usd": duration_seconds / 3600 * stage_d_manifest()["execution"]["h100_rate_usd_per_hour"],
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        "claim_label": stage_d_manifest()["claim_label"], "mechanistic_evidence": False,
    }
    (output_root / "terminal.json").write_text(json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n")
    artifacts = [artifact_declaration(p, output_root, role="stage-d-evidence") for p in sorted(output_root.rglob("*")) if p.is_file()]
    evidence = write_task_evidence(
        output_root,
        metrics=metrics,
        artifacts=artifacts,
        track_h={"stage": "D", "manifest_sha256": STAGE_D_MANIFEST_SHA256, "mechanistic_evidence": False},
    )
    return {**terminal, "task_evidence": evidence}
