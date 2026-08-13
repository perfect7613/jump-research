"""Observation-only longer-horizon predictive world model.

This is the Phase-A pivot after the one-step Stage-C null.  The encoder sees
only the sealed four-frame kinematic tensor.  It receives no partition, law,
seed, identifier, or future target field.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jump_contracts import (
    build_learned_latent_evidence,
    learned_decoder_identity,
    seal_learned_latent_result,
    tensor_bytes_sha256,
    verify_decoded_image_bytes,
    verify_encoder_observation_bytes,
    verify_latent_tensor_bytes,
)

from .authentic import (
    AUTHENTIC_SCHEMA_VERSION,
    EVIDENCE_FRAMES,
    HOLDOUT_LAW_FAMILY,
    ObservationArtifact,
    component_stream_seeds,
    independent_law,
    independent_partition,
    law_family,
    module_content_sha256,
    render_predicted_state_svg,
    serialize_visible_observations,
)
from .canonical import sha256_json
from .experiment_spec import compile_experiment_intent
from .simulator import EpisodeSpec, SimulatorConfig, derive_seed, generate_episode
from .task_adapter import write_track_h_task_evidence


LONG_HORIZON_SCHEMA_VERSION = "jump.track-h-long-horizon-manifest/v3"
LONG_HORIZON_RESULT_VERSION = "jump.track-h-long-horizon-result/v1"
FUTURE_HORIZON = 8
LATENT_DIM = 32
RATE_USD_PER_HOUR = 3.9492


@dataclass(frozen=True)
class LongHorizonLatent:
    data: bytes
    sha256: str


def serialize_long_horizon_latent(tensor: Any) -> LongHorizonLatent:
    import torch

    raw = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
    if len(raw) != LATENT_DIM * 4:
        raise ValueError("long-horizon latent byte length mismatch")
    return LongHorizonLatent(
        raw,
        tensor_bytes_sha256(raw, dtype="float32-le", shape=[LATENT_DIM], order="C"),
    )


def long_horizon_launch_spec() -> dict[str, Any]:
    return compile_experiment_intent(
        {
            "schema_version": "jump.experiment-intent/v1",
            "intent": "Predict eight future steps from four observed motion frames.",
            "session_id": "track-h-long-horizon",
            "seed": 120731,
            "max_steps": 12,
        }
    )


def long_horizon_manifest(mode: str) -> dict[str, Any]:
    if mode not in {"pilot", "replication"}:
        raise ValueError("long-horizon mode must be pilot or replication")
    pilot = mode == "pilot"
    seeds = [120731] if pilot else [120731, 120732, 120733]
    return {
        "schema_version": LONG_HORIZON_SCHEMA_VERSION,
        "experiment_id": f"track-h-long-horizon-same-z-residual-{mode}-v3",
        "phase": "A",
        "mode": mode,
        "architecture_attempt": 2,
        "recovery_of": {
            "architecture_attempt": 1,
            "pilot_manifest_sha256": "0f0fa9cb2428d1ae909ed4a2cffd7b2c0d988ec003058e8c2abc8196d496530b",
            "call_id": "fc-01KZXJKT35S00JR52EZR72WFS1",
            "decision": "pivot",
            "diagnosis": "absolute-state decoder underperformed copy-last persistence on both untouched splits",
            "source_outputs_reused": False,
            "source_root_mutated": False,
            "predictive_only_intermediate": {
                "pilot_call_id": "fc-01KZXJS8BN7XQ842P5V4F4479P",
                "replication_call_id": "fc-01KZXJYK7G94QK70M8NHN5WG39",
                "disposition": "predictive metrics retained; not eligible for Phase B because last visible state bypassed serialized z",
            },
        },
        "claim_label": "observation-only predictive engineering study; no behavioral, causal, or mechanistic claim",
        "input": {
            "frames": EVIDENCE_FRAMES,
            "shape": [EVIDENCE_FRAMES, 6, 4],
            "features": ["position_x", "position_y", "velocity_x", "velocity_y"],
            "forbidden": [
                "partition", "law", "replacement_law", "adequacy", "target",
                "future", "forces", "seed", "world_seed", "episode_id", "index",
            ],
        },
        "generator": {
            "schema_version": AUTHENTIC_SCHEMA_VERSION,
            "simulator_steps": EVIDENCE_FRAMES + FUTURE_HORIZON,
            "future_horizon": FUTURE_HORIZON,
            "independent_components": [
                "same-sign", "different-sign", "exponent", "partition",
                "appearance", "initial-state", "record-order",
            ],
            "heldout_law_family": list(HOLDOUT_LAW_FAMILY),
            "world_seed_disjoint": True,
            "counts": (
                {"train": 1024, "id_validation": 256, "id_test": 256, "heldout_law_ood": 256}
                if pilot
                else {"train": 4096, "id_validation": 512, "id_test": 512, "heldout_law_ood": 512}
            ),
        },
        "model": {
            "encoder": "observation-only structured 32D z: 12 exact last-visible position coordinates plus learned Linear(96,128)->GELU->Linear(128,64)->GELU->Linear(64,20)",
            "decoder": "same-z-only zero-initialized residual: repeat z[0:12] across horizon plus Linear(32,128)->GELU->Linear(128,256)->GELU->Linear(256,8x6x2)",
            "latent_dim": LATENT_DIM,
            "objective": "future-position MSE from decoder(serialized z) only, plus 1e-5 mean-squared learned-z tail penalty",
            "relation_or_law_supervision": False,
            "decoder_external_observation_input": False,
            "decoder_output_is_function_of_z_and_frozen_weights_only": True,
            "steps": 1800 if pilot else 3200,
            "batch_size": 128,
            "learning_rate": 0.0005,
        },
        "evaluation": {
            "primary": "8-step rollout NRMSE against copy-last persistence under identical target standard-deviation normalization",
            "untouched_splits": ["id_test", "heldout_law_ood"],
            "paired_world_bootstrap_replicates": 10000,
            "bootstrap_seed": 80317,
            "pass": {
                "relative_improvement_min": 0.20,
                "paired_ci_lower_exclusive": 0.0,
                "required_splits": ["id_test", "heldout_law_ood"],
                "required_all_seeds": True,
            },
            "id_validation_used_for_selection": False,
            "id_test_and_ood_materialized_after_training": True,
        },
        "initialization": {"from_scratch": True, "seeds": seeds},
        "execution": {
            "modal_function": "authentic_world_long_horizon",
            "resource": "H100",
            "gpu_count": 1,
            "max_containers": 1,
            "max_inputs": 1,
            "serial": True,
            "max_attempts": 1,
            "timeout_seconds": 3600 if pilot else 7200,
            "h100_rate_usd_per_hour": RATE_USD_PER_HOUR,
            "forecast_usd": RATE_USD_PER_HOUR * (1 if pilot else 2),
            "aggregate_authority_ceiling_usd": 100.0,
        },
        "claims": {"g2": False, "g3": False, "behavioral": False, "causal": False, "mechanistic": False},
    }


PILOT_MANIFEST_SHA256 = sha256_json(long_horizon_manifest("pilot"))
REPLICATION_MANIFEST_SHA256 = sha256_json(long_horizon_manifest("replication"))


def manifest_sha256(mode: str) -> str:
    return PILOT_MANIFEST_SHA256 if mode == "pilot" else REPLICATION_MANIFEST_SHA256


def build_long_horizon_modules():
    import torch

    class ObservationEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.learned = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(96, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, LATENT_DIM - 12),
            )

        def forward(self, observations):
            visible_last_positions = observations[:, -1, :, :2].reshape(observations.shape[0], 12)
            return torch.cat([visible_last_positions, self.learned(observations)], dim=-1)

    class SameZResidualDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residual = torch.nn.Sequential(
                torch.nn.Linear(LATENT_DIM, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, FUTURE_HORIZON * 6 * 2),
            )
            torch.nn.init.zeros_(self.residual[-1].weight)
            torch.nn.init.zeros_(self.residual[-1].bias)

        def forward(self, z):
            baseline = z[:, :12].reshape(-1, 1, 6, 2).expand(-1, FUTURE_HORIZON, -1, -1)
            residual = self.residual(z).reshape(-1, FUTURE_HORIZON, 6, 2)
            return (baseline + residual).reshape(-1, FUTURE_HORIZON * 6 * 2)

    encoder = ObservationEncoder()
    decoder = SameZResidualDecoder()
    return encoder, decoder


def _predict(encoder: Any, decoder: Any, inputs: Any):
    z = encoder(inputs)
    prediction = decoder(z).reshape(-1, FUTURE_HORIZON, 6, 2)
    persistence = inputs[:, -1, :, :2].unsqueeze(1).expand_as(prediction)
    return z, prediction, persistence


def _record(episode: dict[str, Any], split: str) -> dict[str, Any]:
    payload = serialize_visible_observations(episode, EVIDENCE_FRAMES)
    observation = ObservationArtifact.from_payload(payload)
    timeline = [episode["initial_state"], *episode["observations"]]
    future = [frame["positions"] for frame in timeline[EVIDENCE_FRAMES:EVIDENCE_FRAMES + FUTURE_HORIZON]]
    if len(future) != FUTURE_HORIZON:
        raise RuntimeError("simulator did not materialize the frozen future horizon")
    return {
        "split": split,
        "world_seed": episode["world_seed"],
        "world_id": episode["episode_id"],
        "encoder_input": observation.payload(),
        "encoder_input_sha256": observation.sha256(),
        "future_positions": future,
    }


def long_horizon_dataset(root_seed: int, counts: dict[str, int], splits: tuple[str, ...]) -> dict[str, Any]:
    allowed = {"train", "id_validation", "id_test", "heldout_law_ood"}
    if not splits or not set(splits) <= allowed:
        raise ValueError("invalid long-horizon split selection")
    config = SimulatorConfig(steps=EVIDENCE_FRAMES + FUTURE_HORIZON)
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for split in splits:
        accepted: list[dict[str, Any]] = []
        candidate = 0
        while len(accepted) < counts[split]:
            seed = derive_seed(root_seed, f"long-horizon:{split}:{candidate}")
            candidate += 1
            law = independent_law(seed)
            is_ood = law_family(law) == HOLDOUT_LAW_FAMILY
            if (split == "heldout_law_ood") != is_ood:
                continue
            if seed in seen:
                raise RuntimeError("world seed collision")
            seen.add(seed)
            partition = independent_partition(seed)
            episode = generate_episode(EpisodeSpec(seed, split, law, True, config, partition))
            row = _record(episode, split)
            row["generator_provenance"] = {
                "component_stream_seeds": component_stream_seeds(seed),
                "same_sign": law.same,
                "different_sign": law.different,
                "exponent": law.exponent,
                "partition": list(partition),
                "ordering_index_serialized_to_encoder": False,
            }
            accepted.append(row)
        random.Random(derive_seed(root_seed, f"long-horizon:record-order:{split}")).shuffle(accepted)
        records.extend(accepted)
    return {
        "schema_version": "jump.track-h-long-horizon-dataset/v1",
        "root_seed": root_seed,
        "counts": {split: counts[split] for split in splits},
        "materialized_splits": list(splits),
        "heldout_law_family": list(HOLDOUT_LAW_FAMILY),
        "records": records,
    }


def _tensors(dataset: dict[str, Any], split: str, device: str):
    import torch

    rows = [row for row in dataset["records"] if row["split"] == split]
    inputs = torch.tensor([row["encoder_input"]["values"] for row in rows], dtype=torch.float32, device=device)
    targets = torch.tensor([row["future_positions"] for row in rows], dtype=torch.float32, device=device)
    return rows, inputs, targets


def _bootstrap_ci(values: list[float], *, seed: int, replicates: int) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    return [means[int(0.025 * replicates)], means[min(replicates - 1, int(0.975 * replicates))]]


def evaluate_rollout(encoder: Any, decoder: Any, dataset: dict[str, Any], split: str, device: str, *, bootstrap_replicates: int = 10000) -> dict[str, Any]:
    import torch

    rows, inputs, targets = _tensors(dataset, split, device)
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        z, prediction, persistence = _predict(encoder, decoder, inputs)
    scale = targets.std(unbiased=False).clamp_min(1e-8)
    model_rmse = ((prediction - targets).square().mean(dim=(1, 2, 3)).sqrt() / scale)
    persistence_rmse = ((persistence - targets).square().mean(dim=(1, 2, 3)).sqrt() / scale)
    relative = 1.0 - model_rmse / persistence_rmse.clamp_min(1e-8)
    values = [float(value) for value in relative.detach().cpu()]
    return {
        "split": split,
        "worlds": len(rows),
        "rollout_horizon": FUTURE_HORIZON,
        "target_std": float(scale.detach().cpu()),
        "model_nrmse": float(model_rmse.mean().detach().cpu()),
        "persistence_nrmse": float(persistence_rmse.mean().detach().cpu()),
        "paired_relative_improvement_mean": sum(values) / len(values),
        "paired_relative_improvement_ci95": _bootstrap_ci(values, seed=80317 + len(split), replicates=bootstrap_replicates),
        "per_world_relative_improvement_sha256": hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest(),
        "first_prediction": prediction[0].detach().cpu().tolist(),
        "first_latent": z[0].detach().cpu(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _run_seed(*, seed: int, manifest: dict[str, Any], output_root: Path, code_sha: str, device: str, dry_run: bool) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    counts = manifest["generator"]["counts"] if not dry_run else {"train": 32, "id_validation": 8, "id_test": 8, "heldout_law_ood": 8}
    steps = manifest["model"]["steps"] if not dry_run else 4
    bootstrap = manifest["evaluation"]["paired_world_bootstrap_replicates"] if not dry_run else 50
    phase1 = long_horizon_dataset(seed, counts, ("train", "id_validation"))
    phase1_sha = sha256_json(phase1)
    encoder, decoder = build_long_horizon_modules()
    encoder.to(device); decoder.to(device)
    _, train_x, train_y = _tensors(phase1, "train", device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=manifest["model"]["learning_rate"])
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = min(manifest["model"]["batch_size"], train_x.shape[0])
    started = time.monotonic()
    initial_loss = None
    final_loss = None
    for step in range(steps):
        indices = torch.randint(train_x.shape[0], (batch_size,), generator=generator, device=device)
        z, prediction, _ = _predict(encoder, decoder, train_x[indices])
        loss = torch.nn.functional.mse_loss(prediction, train_y[indices]) + 1e-5 * z[:, 12:].square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite longer-horizon predictive loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        final_loss = float(loss.detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    for parameter in [*encoder.parameters(), *decoder.parameters()]:
        parameter.requires_grad_(False); parameter.grad = None
    model_hashes = {"encoder_state_sha256": module_content_sha256(encoder), "decoder_state_sha256": module_content_sha256(decoder)}
    validation = evaluate_rollout(encoder, decoder, phase1, "id_validation", device, bootstrap_replicates=bootstrap)
    final = long_horizon_dataset(seed, counts, ("id_test", "heldout_law_ood"))
    final_sha = sha256_json(final)
    metrics = {
        "id_validation": validation,
        "id_test": evaluate_rollout(encoder, decoder, final, "id_test", device, bootstrap_replicates=bootstrap),
        "heldout_law_ood": evaluate_rollout(encoder, decoder, final, "heldout_law_ood", device, bootstrap_replicates=bootstrap),
    }
    passed = all(
        metrics[split]["paired_relative_improvement_mean"] >= 0.20
        and metrics[split]["paired_relative_improvement_ci95"][0] > 0.0
        for split in ("id_test", "heldout_law_ood")
    )
    output_root.mkdir(parents=True, exist_ok=False)
    save_file({name: value.detach().cpu() for name, value in encoder.state_dict().items()}, output_root / "encoder.safetensors")
    save_file({name: value.detach().cpu() for name, value in decoder.state_dict().items()}, output_root / "decoder.safetensors")
    _write_json(output_root / "phase1-dataset.json", phase1)
    _write_json(output_root / "sealed-final-dataset.json", final)
    _write_json(output_root / "model-config.json", manifest["model"])
    encoder_weights_sha = hashlib.sha256((output_root / "encoder.safetensors").read_bytes()).hexdigest()
    decoder_weights_sha = hashlib.sha256((output_root / "decoder.safetensors").read_bytes()).hexdigest()
    row = next(item for item in final["records"] if item["split"] == "heldout_law_ood")
    observation = ObservationArtifact.from_payload(row["encoder_input"])
    latent = serialize_long_horizon_latent(metrics["heldout_law_ood"]["first_latent"])
    latent_copy_1 = bytes(latent.data); latent_copy_2 = memoryview(bytes(latent.data))
    exact_z = torch.frombuffer(bytearray(latent.data), dtype=torch.float32).to(device).reshape(1, LATENT_DIM)
    with torch.no_grad():
        exact_prediction = decoder(exact_z).reshape(FUTURE_HORIZON, 6, 2).detach().cpu().tolist()
    predicted_final = exact_prediction[-1]
    svg = render_predicted_state_svg(predicted_final, latent.sha256).encode()
    (output_root / "encoder-observation.f32le.bin").write_bytes(observation.bytes())
    (output_root / "world-latent.f32le.bin").write_bytes(latent.data)
    (output_root / "predicted-from-z.svg").write_bytes(svg)
    evidence = build_learned_latent_evidence(
        encoder_output=latent.data,
        decoder_input=latent_copy_1,
        injection_input=latent_copy_2,
        encoder_observation=observation.bytes(),
        encoder_observation_artifact_name="encoder-observation.f32le.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le", shape=[LATENT_DIM], order="C",
        tensor_artifact_name="world-latent.f32le.bin",
        recipient_world_id=row["world_id"],
        world_pair_id=f"long-horizon-{seed}",
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder.safetensors",
            artifact_sha256=decoder_weights_sha,
            training_manifest_sha256=manifest_sha256(manifest["mode"]),
            code_version=code_sha,
            architecture="same-z-32d-to-eight-step-six-object-rollout-v1",
        ),
        decoded_image=svg,
        decoded_image_media_type="image/svg+xml",
        answer={
            "predicted_future_positions": exact_prediction,
            "producer_bindings": {
                "encoder_artifact_sha256": encoder_weights_sha,
                "encoder_training_manifest_sha256": manifest_sha256(manifest["mode"]),
                "source_observation_sha256": observation.sha256(),
                "decoder_external_observation_input": False,
                "decoder_input_world_latent_sha256": latent.sha256,
                "code_sha": code_sha,
            },
        },
    )
    sealed = seal_learned_latent_result(
        evidence, source="cached", manifest_sha256=manifest_sha256(manifest["mode"]),
        run_id=f"long-horizon-{manifest['mode']}-{seed}", code_version=code_sha,
        checkpoint_id=decoder_weights_sha,
    )
    _write_json(output_root / "learned-latent-evidence.json", evidence)
    _write_json(output_root / "sealed-result.json", sealed)
    for split_metrics in metrics.values():
        split_metrics.pop("first_latent", None)
        split_metrics.pop("first_prediction", None)
    terminal = {
        "schema_version": LONG_HORIZON_RESULT_VERSION,
        "status": "completed",
        "phase": "A",
        "mode": manifest["mode"],
        "seed": seed,
        "decision": "dry_run_not_evaluated" if dry_run else ("pass" if passed else "pivot"),
        "manifest_sha256": manifest_sha256(manifest["mode"]),
        "phase1_dataset_sha256": phase1_sha,
        "sealed_final_dataset_sha256": final_sha,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "metrics": metrics,
        **model_hashes,
        "encoder_weights_sha256": encoder_weights_sha,
        "decoder_weights_sha256": decoder_weights_sha,
        "world_latent_sha256": evidence["tensor"]["world_latent_sha256"],
        "training_seconds": training_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "claims": manifest["claims"],
        "claim_label": manifest["claim_label"],
        "artifact_verification": "pending",
    }
    check_encoder, check_decoder = build_long_horizon_modules()
    check_encoder.load_state_dict(load_file(output_root / "encoder.safetensors"), strict=True)
    check_decoder.load_state_dict(load_file(output_root / "decoder.safetensors"), strict=True)
    if module_content_sha256(check_encoder) != model_hashes["encoder_state_sha256"] or module_content_sha256(check_decoder) != model_hashes["decoder_state_sha256"]:
        raise RuntimeError("saved longer-horizon safetensors mismatch")
    verify_encoder_observation_bytes(evidence, observation.bytes())
    verify_latent_tensor_bytes(evidence, latent.data)
    verify_decoded_image_bytes(evidence, svg)
    terminal["artifact_verification"] = "passed"
    _write_json(output_root / "seed-terminal.json", terminal)
    return terminal


def run_long_horizon(*, mode: str, output_root: Path, expected_manifest_sha256: str, expected_code_sha: str, device: str = "cuda", dry_run: bool = False) -> dict[str, Any]:
    manifest = long_horizon_manifest(mode)
    if expected_manifest_sha256 != manifest_sha256(mode):
        raise ValueError("long-horizon manifest mismatch before output write")
    if os.environ.get("JUMP_CODE_VERSION") != expected_code_sha:
        raise ValueError("long-horizon explicit code identity mismatch")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable long-horizon output root exists")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "training-manifest.json", manifest)
    started = time.monotonic()
    results = [
        _run_seed(seed=seed, manifest=manifest, output_root=output_root / f"seed-{seed}", code_sha=expected_code_sha, device=device, dry_run=dry_run)
        for seed in manifest["initialization"]["seeds"]
    ]
    aggregate_pass = all(result["decision"] == "pass" for result in results)
    duration = time.monotonic() - started
    terminal = {
        "schema_version": LONG_HORIZON_RESULT_VERSION,
        "status": "completed",
        "phase": "A",
        "mode": mode,
        "decision": "dry_run_not_evaluated" if dry_run else ("pass" if aggregate_pass else "pivot"),
        "manifest_sha256": expected_manifest_sha256,
        "code_sha": expected_code_sha,
        "seeds": results,
        "runtime_seconds": duration,
        "estimated_cost_usd": 0.0 if dry_run else duration / 3600 * RATE_USD_PER_HOUR,
        "claims": manifest["claims"],
        "claim_label": manifest["claim_label"],
        "phase_b_allowed": bool(not dry_run and aggregate_pass),
    }
    metrics = []
    for result in results:
        for split in ("id_test", "heldout_law_ood"):
            row = result["metrics"][split]
            metrics.extend(
                [
                    {"name": "rollout_nrmse", "value": row["model_nrmse"], "seed": result["seed"], "split": split},
                    {"name": "persistence_nrmse", "value": row["persistence_nrmse"], "seed": result["seed"], "split": split},
                    {"name": "paired_relative_improvement", "value": row["paired_relative_improvement_mean"], "seed": result["seed"], "split": split},
                ]
            )
    return write_track_h_task_evidence(
        output_root,
        metrics=metrics,
        terminal=terminal,
        experiment_spec=long_horizon_launch_spec(),
        terminal_name="aggregate-terminal.json",
        track_h={"phase": "A", "mode": mode, "decision": terminal["decision"], "claims": manifest["claims"]},
    )


def run_contract(*, mode: str, expected_manifest_sha256: str, expected_code_sha: str, dry_run: bool = False):
    phase = {"id": f"long-horizon-{mode}", "_secret_keys": [], "_preregistration": {"layer_allowlist": ["world-latent"], "timepoint_allowlist": ["future-steps-1-8"]}}
    run = {
        "id": f"long-horizon-{mode}{'-dry' if dry_run else ''}",
        "task": {"module": "jump_benchmark.long_horizon_task", "parameters": {"mode": mode, "expected_manifest_sha256": expected_manifest_sha256, "expected_code_sha": expected_code_sha, "dry_run": dry_run}},
        "resources": {"gpu": "cpu" if dry_run else "H100", "timeout_seconds": 300 if dry_run else long_horizon_manifest(mode)["execution"]["timeout_seconds"]},
        "selection": {"layers": [], "timepoints": []},
        "retry": {"max_attempts": 1},
    }
    return phase, run
