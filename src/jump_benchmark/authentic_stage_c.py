"""Frozen Stage C predictive-world pilot and post-hoc probe protocol.

This module does not import Gemma.  Diagnostic labels are available only after
the observation encoder and learned decoder have been trained and frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable

from jump_contracts import (
    build_learned_latent_evidence,
    learned_decoder_identity,
    seal_learned_latent_result,
    verify_decoded_image_bytes,
    verify_encoder_observation_bytes,
    verify_latent_tensor_bytes,
)

from .authentic import (
    AUTHENTIC_SCHEMA_VERSION,
    HOLDOUT_LAW_FAMILY,
    LATENT_DIM,
    LAW_FAMILIES,
    ObservationArtifact,
    _record,
    bind_source_observation,
    build_world_modules,
    component_stream_seeds,
    dataset_tensors,
    independent_law,
    independent_partition,
    law_family,
    matched_world_pair,
    module_content_sha256,
    render_predicted_state_svg,
    serialize_latent_tensor,
)
from .canonical import sha256_json
from .simulator import EpisodeSpec, SimulatorConfig, derive_seed, generate_episode


STAGE_C_SCHEMA_VERSION = "jump.track-h-authentic-stage-c-manifest/v1"
STAGE_C_RESULT_VERSION = "jump.track-h-authentic-stage-c-result/v1"


def stage_c_manifest() -> dict[str, Any]:
    """Return the preregistered one-seed Stage C engineering plan."""
    return {
        "schema_version": STAGE_C_SCHEMA_VERSION,
        "experiment_id": "track-h-authentic-stage-c-99173",
        "claim_label": (
            "one-seed predictive learned-latent engineering pilot; post-hoc probes are "
            "descriptive and do not establish causal or mechanistic evidence"
        ),
        "initialization": {
            "policy": "from_scratch",
            "parameter_seed": 99173,
            "stage_b_weights_loaded": False,
        },
        "dataset": {
            "generator_schema": AUTHENTIC_SCHEMA_VERSION,
            "root_seed": 99173,
            "counts": {
                "train": 1024,
                "id_validation": 256,
                "id_test": 256,
                "heldout_law_ood": 256,
            },
            "world_seed_disjoint": True,
            "heldout_law_family": list(HOLDOUT_LAW_FAMILY),
            "independent_rng_domains": [
                "same-sign",
                "different-sign",
                "exponent",
                "partition",
                "appearance",
                "initial-state",
                "record-order",
            ],
            "id_test_used_for_model_selection": False,
            "heldout_ood_used_for_model_selection": False,
        },
        "world_model": {
            "encoder": "observation-only MLP 96->64->16 GELU",
            "decoder": "same-z MLP 16->64 GELU -> 12 next-position coordinates",
            "objective": "next-position MSE + 1e-4 mean squared z",
            "label_supervision": "none",
            "steps": 2000,
            "learning_rate": 0.0005,
            "checkpoint_every_steps": 100,
            "selection": "final preregistered step; ID validation is diagnostic only",
        },
        "posthoc_probes": {
            "encoder_frozen_before_fit": True,
            "targets": ["15 pair relations", "5 canonical partition bits", "7 seen law families"],
            "model": "ridge linear probe with intercept",
            "regularization_grid": [0.001, 0.01, 0.1, 1.0],
            "outer_folds": 3,
            "inner_folds": 3,
            "fold_group": "world_seed",
            "final_alpha": "median outer-fold selected alpha with lower-value tie break",
            "clustered_ci": {"unit": "world_seed", "bootstrap_replicates": 1000, "level": 0.95, "seed": 44123},
            "multiplicity": "Benjamini-Hochberg FDR across declared ID/OOD probe hypotheses",
            "unseen_law_policy": "heldout-law OOD law accuracy is recorded but not interpreted as an unseen-class test",
            "encoder_gradient_from_probe": False,
        },
        "decoder_evaluation": {
            "rollout_horizon_steps": 1,
            "metrics": ["future_position_nrmse", "position_rmse", "position_mae", "render_coordinate_pixel_rmse"],
            "image_source": "learned_decoder_prediction",
            "ground_truth_used_for_render": False,
        },
        "partition_only_swap": {
            "pair_seed_root": 33173,
            "pairs": 64,
            "matched": ["initial visible frame", "nuisance", "candidate law", "correct law"],
            "varied": "hidden partition and noncoincident future consequence only",
            "decoder_conditions": ["own_z", "literal_donor_z", "deterministically_scrambled_z"],
            "directions": ["A_to_B", "B_to_A"],
            "donor_policy": "exact donor float32-le bytes; no re-encoding or recipient normalization",
            "gemma_used": False,
        },
        "persistence_and_probe_gates": {
            "manifest_verified_before_root_write": True,
            "immutable_output_root_and_one_attempt": True,
            "checkpoint_binds_manifest_dataset_step_optimizer_and_rng": True,
            "checkpoint_atomic_replace_and_volume_commit": True,
            "terminal_result_after_exact_artifact_and_seal_verification": True,
            "encoder_decoder_hashes_frozen_before_and_after_probes": True,
            "probe_folds_hyperparameters_predictions_ci_pvalues_and_bh_retained": True,
        },
        "decision": {
            "pass": {
                "finite": True,
                "loss_ratio_max": 0.2,
                "id_test_nrmse_max": 0.20,
                "heldout_ood_nrmse_max": 0.30,
                "donor_mse_exceeds_own_mse_in_both_directions": True,
                "all_artifact_and_latent_bindings_verified": True,
            },
            "pivot": "finite verified run that misses any predictive or two-direction swap pass threshold; do not proceed to Stage D",
            "kill": "non-finite values, observation leakage, manifest/dataset/checkpoint/hash mismatch, target-derived rendering, OOD NRMSE above 0.50, or budget violation",
            "probe_metrics_are_not_pass_gates": True,
            "g2": False,
            "g3": False,
            "mechanistic_evidence": False,
        },
        "execution": {
            "modal_function": "authentic_world_stage_c",
            "resource": "H100",
            "gpu_count": 1,
            "max_containers": 1,
            "max_inputs": 1,
            "max_attempts": 1,
            "timeout_seconds": 3600,
            "h100_rate_usd_per_hour": 3.9492,
            "retry_aware_forecast_usd": 3.9492,
            "hard_ceiling_usd": 20.0,
            "serial": True,
            "stage_d_auto_launch": False,
        },
    }


STAGE_C_MANIFEST_SHA256 = sha256_json(stage_c_manifest())


def _canonical_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def stage_c_dataset(*, root_seed: int, counts: dict[str, int]) -> dict[str, Any]:
    expected = {"train", "id_validation", "id_test", "heldout_law_ood"}
    if set(counts) != expected or any(isinstance(value, bool) or value <= 0 for value in counts.values()):
        raise ValueError("Stage C counts must contain four positive frozen splits")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    config = SimulatorConfig(steps=6)
    for split in ("train", "id_validation", "id_test", "heldout_law_ood"):
        accepted: list[dict[str, Any]] = []
        candidate = 0
        while len(accepted) < counts[split]:
            seed = derive_seed(root_seed, f"stage-c:{split}:{candidate}")
            candidate += 1
            if seed in seen:
                raise RuntimeError("Stage C world seed collision")
            law = independent_law(seed)
            heldout = law_family(law) == HOLDOUT_LAW_FAMILY
            if (split == "heldout_law_ood") != heldout:
                continue
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
            seen.add(seed)
            accepted.append(row)
        random.Random(derive_seed(root_seed, f"stage-c:record-order:{split}")).shuffle(accepted)
        records.extend(accepted)
    return {
        "schema_version": "jump.track-h-authentic-stage-c-dataset/v1",
        "root_seed": root_seed,
        "counts": {name: counts[name] for name in ("train", "id_validation", "id_test", "heldout_law_ood")},
        "heldout_law_family": list(HOLDOUT_LAW_FAMILY),
        "split_policy": "world-seed-disjoint; ID selection/test separate; heldout family only in heldout_law_ood",
        "records": records,
    }


def _extract_latents(encoder: Any, dataset: dict[str, Any], split: str, device: str):
    import torch

    rows, inputs, targets, relations, laws = dataset_tensors(dataset, split, device)
    encoder.eval()
    with torch.no_grad():
        z = encoder(inputs).detach().to(device="cpu", dtype=torch.float64)
    partitions = torch.tensor(
        [row["sealed_target"]["partition"][1:] for row in rows], dtype=torch.float64
    )
    return rows, z, targets.detach().cpu().to(torch.float64), relations.detach().cpu().to(torch.float64), partitions, laws.detach().cpu()


def _ridge_fit(x: Any, y: Any, alpha: float):
    import torch

    ones = torch.ones((x.shape[0], 1), dtype=torch.float64)
    design = torch.cat([ones, x], dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * alpha
    penalty[0, 0] = 0
    return torch.linalg.solve(design.T @ design + penalty, design.T @ y)


def _ridge_predict(x: Any, weights: Any):
    import torch

    return torch.cat([torch.ones((x.shape[0], 1), dtype=torch.float64), x], dim=1) @ weights


def _accuracy(prediction: Any, target: Any, kind: str) -> float:
    if kind == "multiclass":
        return float((prediction.argmax(dim=1) == target).double().mean())
    return float(((prediction >= 0.5) == (target >= 0.5)).double().mean())


def _targets(rows: list[dict[str, Any]], relations: Any, partitions: Any, laws: Any, kind: str):
    import torch

    if kind == "relations":
        return relations
    if kind == "partition":
        return partitions
    if kind == "law":
        return torch.nn.functional.one_hot(laws, num_classes=len(LAW_FAMILIES)).to(torch.float64)
    raise ValueError(kind)


def _nested_probe(x: Any, y: Any, world_seeds: list[int], *, kind: str, alphas: list[float]):
    import torch

    multiclass = kind == "law"
    outer = [derive_seed(int(seed), f"probe:{kind}:outer") % 3 for seed in world_seeds]
    selected: list[float] = []
    outer_scores: list[float] = []
    fold_records: list[dict[str, Any]] = []
    for fold in range(3):
        outer_train = torch.tensor([value != fold for value in outer], dtype=torch.bool)
        outer_test = ~outer_train
        inner = [derive_seed(int(seed), f"probe:{kind}:inner:{fold}") % 3 for seed in world_seeds]
        candidates = []
        for alpha in alphas:
            scores = []
            for inner_fold in range(3):
                fit_mask = outer_train & torch.tensor([value != inner_fold for value in inner], dtype=torch.bool)
                val_mask = outer_train & torch.tensor([value == inner_fold for value in inner], dtype=torch.bool)
                if not fit_mask.any() or not val_mask.any():
                    raise RuntimeError("empty grouped inner probe fold")
                weights = _ridge_fit(x[fit_mask], y[fit_mask], alpha)
                scores.append(_accuracy(_ridge_predict(x[val_mask], weights), y[val_mask].argmax(1) if multiclass else y[val_mask], "multiclass" if multiclass else "binary"))
            candidates.append((sum(scores) / len(scores), -alpha, alpha))
        alpha = max(candidates)[2]
        selected.append(alpha)
        weights = _ridge_fit(x[outer_train], y[outer_train], alpha)
        score = _accuracy(_ridge_predict(x[outer_test], weights), y[outer_test].argmax(1) if multiclass else y[outer_test], "multiclass" if multiclass else "binary")
        outer_scores.append(score)
        fold_records.append({"outer_fold": fold, "selected_alpha": alpha, "outer_score": score, "train_groups": int(outer_train.sum()), "test_groups": int(outer_test.sum())})
    final_alpha = sorted(selected)[len(selected) // 2]
    weights = _ridge_fit(x, y, final_alpha)
    return weights, {"kind": kind, "outer_folds": fold_records, "outer_mean_score": sum(outer_scores) / len(outer_scores), "selected_alphas": selected, "final_alpha": final_alpha}


def _clustered_ci(values: list[float], seeds: list[int], *, bootstrap_seed: int, replicates: int) -> list[float]:
    if len(values) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("world-seed clusters must be unique and aligned")
    rng = random.Random(bootstrap_seed)
    estimates = []
    for _ in range(replicates):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    return [estimates[int(0.025 * replicates)], estimates[min(replicates - 1, int(0.975 * replicates))]]


def _sign_test_p(values: list[float], chance: float) -> float:
    wins = sum(value > chance for value in values)
    losses = sum(value < chance for value in values)
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def _bh_fdr(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        name, pvalue = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, pvalue * len(ordered) / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def _probe_eval(prediction: Any, target: Any, *, multiclass: bool, seeds: list[int], ci_seed: int, reps: int, chance: float) -> dict[str, Any]:
    if multiclass:
        correct = (prediction.argmax(1) == target).double()
    else:
        correct = ((prediction >= 0.5) == (target >= 0.5)).double().mean(dim=1)
    values = [float(value) for value in correct]
    return {
        "accuracy": sum(values) / len(values),
        "clustered_ci95": _clustered_ci(values, seeds, bootstrap_seed=ci_seed, replicates=reps),
        "cluster_unit": "world_seed",
        "cluster_count": len(values),
        "chance": chance,
        "one_sided_cluster_sign_p": _sign_test_p(values, chance),
        "predictions": prediction.tolist(),
    }


def fit_posthoc_probes(encoder: Any, dataset: dict[str, Any], *, device: str, bootstrap_replicates: int) -> dict[str, Any]:
    import torch

    frozen_hash = module_content_sha256(encoder)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("encoder must be frozen before post-hoc probe fitting")
    train_rows, train_z, _, relations, partitions, laws = _extract_latents(encoder, dataset, "train", device)
    mean, std = train_z.mean(0), train_z.std(0).clamp_min(1e-8)
    train_x = (train_z - mean) / std
    alphas = [0.001, 0.01, 0.1, 1.0]
    trained: dict[str, tuple[Any, dict[str, Any]]] = {}
    for kind in ("relations", "partition", "law"):
        y = _targets(train_rows, relations, partitions, laws, kind)
        trained[kind] = _nested_probe(train_x, y, [row["world_seed"] for row in train_rows], kind=kind, alphas=alphas)
    evaluations: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for split in ("id_validation", "id_test", "heldout_law_ood"):
        rows, z, _, split_relations, split_partitions, split_laws = _extract_latents(encoder, dataset, split, device)
        x = (z - mean) / std
        seeds = [row["world_seed"] for row in rows]
        evaluations[split] = {}
        for offset, kind in enumerate(("relations", "partition", "law")):
            weights, cv = trained[kind]
            prediction = _ridge_predict(x, weights)
            multiclass = kind == "law"
            target = split_laws if multiclass else _targets(rows, split_relations, split_partitions, split_laws, kind)
            chance = 1 / 7 if multiclass else 0.5
            value = _probe_eval(prediction, target, multiclass=multiclass, seeds=seeds, ci_seed=44123 + offset + 10 * len(evaluations), reps=bootstrap_replicates, chance=chance)
            value["nested_cv"] = cv
            if split == "heldout_law_ood" and kind == "law":
                value["interpretation"] = "not_interpreted: heldout law is an unseen class"
            else:
                key = f"{split}:{kind}"
                pvalues[key] = value["one_sided_cluster_sign_p"]
            evaluations[split][kind] = value
    adjusted = _bh_fdr(pvalues)
    for key, qvalue in adjusted.items():
        split, kind = key.split(":")
        evaluations[split][kind]["bh_fdr_q"] = qvalue
    if module_content_sha256(encoder) != frozen_hash or any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("post-hoc probes changed or backpropagated into encoder")
    return {
        "schema_version": "jump.track-h-stage-c-posthoc-probes/v1",
        "encoder_frozen_sha256": frozen_hash,
        "standardization": {"mean": mean.tolist(), "std": std.tolist()},
        "bootstrap_replicates": bootstrap_replicates,
        "evaluations": evaluations,
        "bh_fdr_family": sorted(pvalues),
        "encoder_gradient_from_probe": False,
    }


def _prediction_metrics(encoder: Any, decoder: Any, dataset: dict[str, Any], split: str, device: str) -> dict[str, Any]:
    import torch

    rows, inputs, targets, _, _ = dataset_tensors(dataset, split, device)
    with torch.no_grad():
        z = encoder(inputs)
        predicted = decoder(z)
    error = predicted - targets
    rmse = torch.sqrt(torch.mean(error**2))
    scale = torch.std(targets).clamp_min(1e-8)
    pixel_scale = (512 - 64) / 6.0
    norms = torch.linalg.vector_norm(z, dim=1)
    return {
        "examples": len(rows),
        "rollout_horizon_steps": 1,
        "future_position_nrmse": float((rmse / scale).cpu()),
        "position_rmse": float(rmse.cpu()),
        "position_mae": float(torch.mean(torch.abs(error)).cpu()),
        "render_coordinate_pixel_rmse": float((rmse * pixel_scale).cpu()),
        "latent_norm_min": float(norms.min().cpu()),
        "latent_norm_mean": float(norms.mean().cpu()),
        "latent_norm_max": float(norms.max().cpu()),
        "first_prediction": predicted[0].detach().cpu().tolist(),
    }


def evaluate_partition_swaps(encoder: Any, decoder: Any, *, device: str, pairs: int) -> dict[str, Any]:
    import numpy as np
    import torch

    directions: list[dict[str, Any]] = []
    for pair_index in range(pairs):
        pair = matched_world_pair(pair_seed=derive_seed(33173, f"stage-c-pair:{pair_index}"))
        records = [pair["a"], pair["b"]]
        artifacts = [ObservationArtifact.from_payload(row["encoder_input"]) for row in records]
        observations = torch.tensor([item.values for item in artifacts], dtype=torch.float32, device=device)
        targets = torch.tensor([row["decoder_target"]["next_positions"] for row in records], dtype=torch.float32, device=device)
        with torch.no_grad():
            z = encoder(observations)
        own = [serialize_latent_tensor(z[index]) for index in range(2)]
        for recipient, donor, direction in ((0, 1, "B_to_A"), (1, 0, "A_to_B")):
            own_tensor = torch.from_numpy(np.frombuffer(own[recipient].data, dtype="<f4").copy()).to(device)
            donor_tensor = torch.from_numpy(np.frombuffer(own[donor].data, dtype="<f4").copy()).to(device)
            permutation = list(range(LATENT_DIM))
            random.Random(derive_seed(33173, f"scramble:{pair_index}:{direction}")).shuffle(permutation)
            scrambled_tensor = own_tensor[permutation]
            with torch.no_grad():
                own_prediction = decoder(own_tensor.unsqueeze(0))[0]
                donor_prediction = decoder(donor_tensor.unsqueeze(0))[0]
                scrambled_prediction = decoder(scrambled_tensor.unsqueeze(0))[0]
            own_mse = float(torch.mean((own_prediction - targets[recipient]) ** 2).cpu())
            donor_mse = float(torch.mean((donor_prediction - targets[recipient]) ** 2).cpu())
            scrambled_mse = float(torch.mean((scrambled_prediction - targets[recipient]) ** 2).cpu())
            injected = serialize_latent_tensor(donor_tensor)
            if injected.data != own[donor].data or injected.sha256 != own[donor].sha256:
                raise RuntimeError("literal donor swap changed donor z bytes")
            directions.append({
                "pair_id": pair["pair_id"], "direction": direction,
                "recipient_world_id": records[recipient]["episode_id"],
                "donor_world_id": records[donor]["episode_id"],
                "recipient_own_latent_sha256": own[recipient].sha256,
                "donor_latent_sha256": own[donor].sha256,
                "injected_latent_sha256": injected.sha256,
                "literal_donor_bytes_equal": True,
                "own_mse": own_mse, "donor_mse": donor_mse, "scrambled_mse": scrambled_mse,
            })
    aggregate = {}
    for direction in ("A_to_B", "B_to_A"):
        rows = [row for row in directions if row["direction"] == direction]
        aggregate[direction] = {
            "pairs": len(rows),
            "own_mse": sum(row["own_mse"] for row in rows) / len(rows),
            "donor_mse": sum(row["donor_mse"] for row in rows) / len(rows),
            "scrambled_mse": sum(row["scrambled_mse"] for row in rows) / len(rows),
            "donor_minus_own_mse": sum(row["donor_mse"] - row["own_mse"] for row in rows) / len(rows),
        }
    return {
        "schema_version": "jump.track-h-stage-c-partition-swap/v1",
        "gemma_used": False,
        "pair_definition": "partition-only with matched initial frame, nuisance, and law; noncoincident consequence",
        "directions": directions,
        "aggregate": aggregate,
    }


def _atomic_checkpoint(path: Path, value: Any, commit: Callable[[], None] | None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    if commit is not None:
        commit()


def run_stage_c(
    *,
    output_root: Path,
    checkpoint_root: Path,
    expected_manifest_sha256: str,
    device: str = "cuda",
    checkpoint_commit: Callable[[], None] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Stage C once; the caller owns the immutable root and paid gate."""
    import torch
    from safetensors.torch import load_file, save_file

    if expected_manifest_sha256 != STAGE_C_MANIFEST_SHA256:
        raise ValueError("Stage C manifest hash mismatch before output-root creation")
    if output_root.exists():
        raise FileExistsError("immutable Stage C output root already exists")
    manifest = stage_c_manifest()
    settings = manifest["dataset"]["counts"] if not dry_run else {"train": 24, "id_validation": 8, "id_test": 8, "heldout_law_ood": 8}
    steps = manifest["world_model"]["steps"] if not dry_run else 12
    pairs = manifest["partition_only_swap"]["pairs"] if not dry_run else 4
    bootstrap = manifest["posthoc_probes"]["clustered_ci"]["bootstrap_replicates"] if not dry_run else 50
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote Stage C")
    torch.manual_seed(manifest["initialization"]["parameter_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manifest["initialization"]["parameter_seed"])
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    dataset = stage_c_dataset(root_seed=manifest["dataset"]["root_seed"], counts=settings)
    dataset_sha = sha256_json(dataset)
    encoder, decoder = build_world_modules()
    encoder.to(device); decoder.to(device)
    _, train_inputs, train_targets, _, _ = dataset_tensors(dataset, "train", device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=manifest["world_model"]["learning_rate"])
    checkpoint_path = checkpoint_root / "latest.pt"
    start_step = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        binding = (checkpoint.get("manifest_sha256"), checkpoint.get("dataset_sha256"))
        if binding != (STAGE_C_MANIFEST_SHA256, dataset_sha):
            raise RuntimeError("Stage C checkpoint binding mismatch")
        encoder.load_state_dict(checkpoint["encoder"]); decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"]); torch.set_rng_state(checkpoint["cpu_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_step = int(checkpoint["step"])
    started = time.monotonic()
    with torch.no_grad():
        initial_z = encoder(train_inputs); initial_prediction = decoder(initial_z)
        initial_loss = torch.nn.functional.mse_loss(initial_prediction, train_targets) + 1e-4 * torch.mean(initial_z**2)
    loss = initial_loss
    for step in range(start_step + 1, steps + 1):
        encoder.train(); decoder.train(); optimizer.zero_grad(set_to_none=True)
        z = encoder(train_inputs); prediction = decoder(z)
        loss = torch.nn.functional.mse_loss(prediction, train_targets) + 1e-4 * torch.mean(z**2)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Stage C predictive loss")
        loss.backward(); optimizer.step()
        if step % manifest["world_model"]["checkpoint_every_steps"] == 0 or step == steps:
            _atomic_checkpoint(checkpoint_path, {
                "schema_version": "jump.track-h-stage-c-checkpoint/v1",
                "manifest_sha256": STAGE_C_MANIFEST_SHA256,
                "dataset_sha256": dataset_sha,
                "step": step,
                "encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, checkpoint_commit)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    for parameter in list(encoder.parameters()) + list(decoder.parameters()):
        parameter.requires_grad_(False); parameter.grad = None
    encoder_hash, decoder_hash = module_content_sha256(encoder), module_content_sha256(decoder)
    metrics = {split: _prediction_metrics(encoder, decoder, dataset, split, device) for split in ("train", "id_validation", "id_test", "heldout_law_ood")}
    probes = fit_posthoc_probes(encoder, dataset, device=device, bootstrap_replicates=bootstrap)
    swaps = evaluate_partition_swaps(encoder, decoder, device=device, pairs=pairs)
    if (encoder_hash, decoder_hash) != (module_content_sha256(encoder), module_content_sha256(decoder)):
        raise RuntimeError("Stage C weights changed after freeze/evaluation/probes")

    output_root.mkdir(parents=True, exist_ok=False)
    save_file({name: value.detach().cpu() for name, value in encoder.state_dict().items()}, output_root / "encoder.safetensors")
    save_file({name: value.detach().cpu() for name, value in decoder.state_dict().items()}, output_root / "decoder.safetensors")
    _canonical_write(output_root / "training-manifest.json", manifest)
    _canonical_write(output_root / "dataset.json", dataset)
    _canonical_write(output_root / "probe-results.json", probes)
    _canonical_write(output_root / "swap-results.json", swaps)
    _canonical_write(output_root / "model-config.json", {"encoder": "96->64->16", "decoder": "16->64->12", "initialization": manifest["initialization"]})

    ood_rows, ood_inputs, _, _, _ = dataset_tensors(dataset, "heldout_law_ood", device)
    with torch.no_grad():
        encoder_z = encoder(ood_inputs[:1]).squeeze(0)
        encoder_capture = serialize_latent_tensor(encoder_z)
        decoder_capture = serialize_latent_tensor(torch.frombuffer(bytearray(encoder_capture.data), dtype=torch.float32).to(device))
        injection_capture = serialize_latent_tensor(torch.frombuffer(bytearray(encoder_capture.data), dtype=torch.float32).to(device))
    observation = ObservationArtifact.from_payload(ood_rows[0]["encoder_input"])
    observation_bytes = observation.bytes()
    bind_source_observation(source_record=ood_rows[0], observation_bytes=observation_bytes, source_world_id=ood_rows[0]["episode_id"])
    svg = render_predicted_state_svg(metrics["heldout_law_ood"].pop("first_prediction"), encoder_capture.sha256)
    for split in ("train", "id_validation", "id_test"):
        metrics[split].pop("first_prediction")
    svg_bytes = svg.encode()
    (output_root / "predicted-from-z.svg").write_bytes(svg_bytes)
    (output_root / "encoder-observation.f32le.bin").write_bytes(observation_bytes)
    _canonical_write(output_root / "encoder-observation-metadata.json", observation.descriptor())
    (output_root / "world-latent.f32le.bin").write_bytes(encoder_capture.data)
    encoder_artifact_sha = hashlib.sha256((output_root / "encoder.safetensors").read_bytes()).hexdigest()
    decoder_artifact_sha = hashlib.sha256((output_root / "decoder.safetensors").read_bytes()).hexdigest()
    answer = {
        "predicted_next_positions": json.loads(json.dumps(metrics["heldout_law_ood"].get("retained_prediction", []))),
        "producer_bindings": {
            "encoder_identity": {"artifact_name": "encoder.safetensors", "artifact_sha256": encoder_artifact_sha, "training_manifest_sha256": STAGE_C_MANIFEST_SHA256, "architecture_config_sha256": sha256_json(manifest["world_model"]), "architecture": manifest["world_model"]["encoder"]},
            "source_observation": {**ood_rows[0]["observation_binding"], **observation.descriptor()},
            "stage_c_manifest_sha256": STAGE_C_MANIFEST_SHA256,
        },
    }
    # Recompute the retained prediction directly from the exact serialized z;
    # no ground-truth state enters the image or answer.
    with torch.no_grad():
        exact_z = torch.frombuffer(bytearray(encoder_capture.data), dtype=torch.float32).to(device)
        retained_prediction = decoder(exact_z.unsqueeze(0))[0].cpu().tolist()
    answer["predicted_next_positions"] = retained_prediction
    evidence = build_learned_latent_evidence(
        encoder_output=encoder_capture.data, decoder_input=decoder_capture.data, injection_input=injection_capture.data,
        encoder_observation=observation_bytes, encoder_observation_artifact_name="encoder-observation.f32le.bin", encoder_observation_media_type="application/octet-stream",
        dtype="float32-le", shape=[LATENT_DIM], order="C", tensor_artifact_name="world-latent.f32le.bin",
        recipient_world_id=ood_rows[0]["episode_id"], world_pair_id="stage-c-heldout-singleton",
        learned_decoder=learned_decoder_identity(artifact_name="decoder.safetensors", artifact_sha256=decoder_artifact_sha, training_manifest_sha256=STAGE_C_MANIFEST_SHA256, code_version=os.environ.get("JUMP_CODE_VERSION", "local-dry-run"), architecture="same-z-16d-to-six-object-next-position-v1"),
        decoded_image=svg_bytes, decoded_image_media_type="image/svg+xml", answer=answer,
    )
    sealed = seal_learned_latent_result(evidence, source="cached", manifest_sha256=STAGE_C_MANIFEST_SHA256, run_id="authentic-stage-c-99173", code_version=os.environ.get("JUMP_CODE_VERSION", "local-dry-run"), checkpoint_id=decoder_artifact_sha)
    _canonical_write(output_root / "learned-latent-evidence.json", evidence)
    _canonical_write(output_root / "sealed-result.json", sealed)

    loss_ratio = float(loss.detach().cpu()) / float(initial_loss.detach().cpu())
    donor_pass = all(value["donor_minus_own_mse"] > 0 for value in swaps["aggregate"].values())
    pass_gate = loss_ratio <= 0.2 and metrics["id_test"]["future_position_nrmse"] <= 0.20 and metrics["heldout_law_ood"]["future_position_nrmse"] <= 0.30 and donor_pass
    kill = metrics["heldout_law_ood"]["future_position_nrmse"] > 0.50
    # The reduced CPU dry run verifies only data/model/artifact plumbing.  Its
    # 12-step metrics are not the frozen 2,000-step experiment and therefore
    # must not trigger or masquerade as a scientific pass/pivot/kill decision.
    decision = "dry_run_not_evaluated" if dry_run else ("kill" if kill else ("pass" if pass_gate else "pivot"))
    result = {
        "schema_version": STAGE_C_RESULT_VERSION,
        "status": "completed", "stage": "C", "decision": decision,
        "claim_label": manifest["claim_label"], "mechanistic_evidence": False, "g2": False, "g3": False,
        "manifest_sha256": STAGE_C_MANIFEST_SHA256, "dataset_sha256": dataset_sha,
        "initial_loss": float(initial_loss.detach().cpu()), "final_loss": float(loss.detach().cpu()), "loss_ratio": loss_ratio,
        "metrics": metrics, "probe_results_sha256": hashlib.sha256((output_root / "probe-results.json").read_bytes()).hexdigest(),
        "swap_results_sha256": hashlib.sha256((output_root / "swap-results.json").read_bytes()).hexdigest(),
        "encoder_state_sha256": encoder_hash, "decoder_state_sha256": decoder_hash,
        "encoder_weights_sha256": encoder_artifact_sha, "decoder_weights_sha256": decoder_artifact_sha,
        "weights_frozen_before_evaluation_and_probes": True,
        "training_seconds": training_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "world_latent_sha256": evidence["tensor"]["world_latent_sha256"],
        "image_source": "learned_decoder_prediction", "dry_run": dry_run,
    }
    _canonical_write(output_root / "result.json", result)

    # Fail closed before terminal completion: independently open safetensors,
    # validate the shared evidence, and require exact declared artifact coverage.
    check_encoder, check_decoder = build_world_modules()
    check_encoder.load_state_dict(load_file(output_root / "encoder.safetensors")); check_decoder.load_state_dict(load_file(output_root / "decoder.safetensors"))
    if (module_content_sha256(check_encoder), module_content_sha256(check_decoder)) != (encoder_hash, decoder_hash):
        raise RuntimeError("saved Stage C safetensors do not match frozen modules")
    verify_encoder_observation_bytes(evidence, observation_bytes)
    verify_latent_tensor_bytes(evidence, encoder_capture.data)
    verify_decoded_image_bytes(evidence, svg_bytes)
    hashes = {str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output_root.rglob("*")) if path.is_file()}
    _canonical_write(output_root / "SHA256SUMS.json", hashes)
    return {**result, "artifact_hashes": hashes, "artifact_root": str(output_root), "artifact_verification": "passed"}
