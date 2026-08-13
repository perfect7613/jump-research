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
from .experiment_spec import compile_experiment_intent, validate_experiment_spec
from .simulator import EpisodeSpec, SimulatorConfig, derive_seed, generate_episode
from .task_adapter import write_track_h_task_evidence


STAGE_C_SCHEMA_VERSION = "jump.track-h-authentic-stage-c-manifest/v1"
STAGE_C_RESULT_VERSION = "jump.track-h-authentic-stage-c-result/v1"


def stage_c_launch_spec() -> dict[str, Any]:
    """Return the only ExperimentSpec authorized to label Stage C training."""
    return compile_experiment_intent(
        {
            "schema_version": "jump.experiment-intent/v1",
            "intent": "Predict the future trajectory from observed motion.",
            "session_id": "track-h-stage-c",
            "seed": 99173,
            "max_steps": 4,
        }
    )


STAGE_C_LAUNCH_SPEC_SHA256 = sha256_json(stage_c_launch_spec())


def authorize_stage_c_launch(
    *,
    expected_manifest_sha256: str,
    expected_code_sha: str,
    actual_code_sha: str,
    confirm_paid: bool,
    confirm_h100: bool,
) -> dict[str, Any]:
    """Fail closed on the one frozen paid H100 Stage C launch contract."""
    if confirm_paid is not True or confirm_h100 is not True:
        raise PermissionError("Stage C requires literal confirm_paid=true and confirm_h100=true")
    if expected_manifest_sha256 != STAGE_C_MANIFEST_SHA256:
        raise ValueError("Stage C manifest hash mismatch")
    if (
        not isinstance(expected_code_sha, str)
        or len(expected_code_sha) != 40
        or any(char not in "0123456789abcdef" for char in expected_code_sha)
        or actual_code_sha != expected_code_sha
    ):
        raise ValueError("Stage C code revision mismatch")
    execution = stage_c_manifest()["execution"]
    forecast = (
        execution["timeout_seconds"]
        * execution["max_attempts"]
        / 3600
        * execution["h100_rate_usd_per_hour"]
    )
    if (
        execution["modal_function"] != "authentic_world_stage_c"
        or execution["resource"] != "H100"
        or execution["gpu_count"] != 1
        or execution["max_containers"] != 1
        or execution["max_inputs"] != 1
        or execution["max_attempts"] != 1
        or execution["serial"] is not True
        or not math.isclose(forecast, execution["retry_aware_forecast_usd"], abs_tol=1e-9)
        or forecast > execution["hard_ceiling_usd"]
    ):
        raise ValueError("Stage C resource or spend contract is invalid")
    plan = stage_c_launch_spec()
    if sha256_json(plan) != STAGE_C_LAUNCH_SPEC_SHA256:
        raise ValueError("Stage C canonical launch spec hash mismatch")
    return plan


def stage_c_run_contract(
    *, expected_manifest_sha256: str, expected_code_sha: str, dry_run: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the fixed allowlisted runner phase/run used for canonical promotion."""
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    phase = {
        "id": "stage-c",
        "_secret_keys": [],
        "_preregistration": {
            "layer_allowlist": ["world-latent"],
            "timepoint_allowlist": ["future-step-1"],
        },
    }
    run = {
        "id": "authentic-world-stage-c-dry" if dry_run else "authentic-world-stage-c",
        "task": {
            "module": "jump_benchmark.stage_c_task",
            "parameters": {
                "expected_manifest_sha256": expected_manifest_sha256,
                "expected_code_sha": expected_code_sha,
                "dry_run": dry_run,
            },
        },
        "resources": {
            "gpu": "cpu" if dry_run else "H100",
            "timeout_seconds": 300 if dry_run else 10_800,
        },
        "selection": {"layers": [], "timepoints": []},
        "retry": {"max_attempts": 1},
    }
    return phase, run


def stage_c_manifest() -> dict[str, Any]:
    """Return the preregistered three-seed Stage C engineering plan."""
    return {
        "schema_version": STAGE_C_SCHEMA_VERSION,
        "experiment_id": "track-h-authentic-stage-c-three-seed-v1",
        "claim_label": (
            "three-seed predictive learned-latent engineering pilot; post-hoc probes are "
            "descriptive and do not establish causal or mechanistic evidence"
        ),
        "execution_lineage": {
            "state": "recovery",
            "recovery_of": {
                "prior_manifest_sha256": "a9cf6370d9bed04ecf3c3af8ec20948e8fa1748624f08d1c0d9992ccfbab63fd",
                "failed_call_ids": [
                    "fc-01KZX71R5MWG5MMPWTFD1XTQZQ",
                    "fc-01KZX8FV6W0EFNWA0E3GGMZ8YM",
                ],
                "partial_inventory_sha256": "f896b888e94491921a080ed61cc682e52523ac79f195bfe6a5d78ece7f83baf2",
                "failure_reason": "Stage C task subprocess omitted JUMP_CODE_VERSION and attempted unavailable git fallback",
                "source_outputs_reused": False,
                "source_root_mutated": False,
            },
        },
        "launch_spec": {
            "policy": "exact_canonical_training_spec_only",
            "sha256": STAGE_C_LAUNCH_SPEC_SHA256,
            "inference_experiment_specs_accepted": False,
        },
        "initialization": {
            "policy": "from_scratch",
            "seeds": [
                {"seed_id": "seed-99173", "parameter_seed": 99173, "dataset_root_seed": 99173},
                {"seed_id": "seed-99174", "parameter_seed": 99174, "dataset_root_seed": 99174},
                {"seed_id": "seed-99175", "parameter_seed": 99175, "dataset_root_seed": 99175},
            ],
            "independent_parameter_and_dataset_replicates": True,
            "stage_b_weights_loaded": False,
        },
        "dataset": {
            "generator_schema": AUTHENTIC_SCHEMA_VERSION,
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
            "evaluation_seal": {
                "phase_1": "train model and nested-CV probes using train split only; evaluate ID validation and freeze probe weights and thresholds",
                "phase_2": "after phase-1 artifact hash is frozen, unseal ID test and heldout-law OOD exactly once",
            },
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
            "clustered_ci": {"unit": "world_seed", "bootstrap_replicates": 10000, "level": 0.95, "seed": 44123},
            "relation_metrics": ["roc_auc", "balanced_accuracy", "roc_auc_clustered_ci95"],
            "relation_pairs_per_world": 15,
            "relation_pairs_cofolded_and_clustered_by_world": True,
            "threshold_selection": "ID-validation grid maximizes balanced accuracy; lower threshold wins ties; frozen before phase 2",
            "threshold_grid": [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75],
            "multiplicity_families": {
                "id_validation": ["relations:roc_auc", "partition:balanced_accuracy", "law:accuracy"],
                "final_ood": ["relations:roc_auc", "partition:balanced_accuracy"],
            },
            "id_and_ood_hypotheses_never_mixed": True,
            "unseen_law_policy": "heldout-law OOD law accuracy is recorded but not interpreted as an unseen-class test",
            "encoder_gradient_from_probe": False,
        },
        "decoder_evaluation": {
            "rollout_horizon_steps": 1,
            "metrics": ["future_position_nrmse", "persistence_position_nrmse", "persistence_relative_improvement", "position_rmse", "position_mae", "render_coordinate_pixel_rmse"],
            "persistence_predictor": "copy the last observed positions under identical targets and target-standard-deviation NRMSE normalization",
            "paired_improvement_min": 0.20,
            "paired_improvement_required_splits": ["id_test", "heldout_law_ood"],
            "paired_improvement_required_all_seeds": True,
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
            "paired_inference": {"bootstrap_replicates": 10000, "ci_level": 0.95, "seed": 55123},
            "success": {"donor_minus_own_mse_mean_min_exclusive": 0.0, "paired_ci_lower_min_exclusive": 0.0, "per_pair_success_rate_min": 0.75, "required_both_directions": True},
            "gemma_used": False,
        },
        "persistence_and_probe_gates": {
            "manifest_verified_before_root_write": True,
            "immutable_output_root_and_one_attempt": True,
            "checkpoint_binds_manifest_phase1_dataset_code_seed_step_optimizer_and_rng": True,
            "checkpoint_atomic_replace_and_volume_commit": True,
            "terminal_result_after_exact_artifact_and_seal_verification": True,
            "encoder_decoder_hashes_frozen_before_and_after_probes": True,
            "probe_folds_hyperparameters_predictions_ci_pvalues_and_bh_retained": True,
            "probe_weights_thresholds_and_id_bh_frozen_before_final_dataset_materialization": True,
        },
        "decision": {
            "pass": {
                "finite": True,
                "loss_ratio_max": 0.2,
                "id_test_nrmse_max": 0.20,
                "heldout_ood_nrmse_max": 0.30,
                "persistence_relative_improvement_min_all_seeds": 0.20,
                "relation_roc_auc_min_all_seeds": 0.75,
                "relation_roc_auc_ci_lower_min_exclusive_all_seeds": 0.50,
                "donor_swap_paired_success_all_seeds_and_directions": True,
                "all_artifact_and_latent_bindings_verified": True,
            },
            "pivot": "finite verified run that misses any predictive or two-direction swap pass threshold; do not proceed to Stage D",
            "kill": "non-finite values, observation leakage, manifest/dataset/checkpoint/hash mismatch, target-derived rendering, OOD NRMSE above 0.50, or budget violation",
            "probe_metrics_are_pass_gates_only_at_final_ood": True,
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
            "timeout_seconds": 10800,
            "h100_rate_usd_per_hour": 3.9492,
            "retry_aware_forecast_usd": 11.8476,
            "hard_ceiling_usd": 20.0,
            "serial": True,
            "code_sha_required_and_verified_against_git_head": True,
            "dependency_lock": {
                "jsonschema": "4.26.0",
                "numpy": "2.3.2",
                "safetensors": "0.8.0",
                "torch": "2.11.0",
            },
            "stage_d_auto_launch": False,
        },
    }


STAGE_C_MANIFEST_SHA256 = sha256_json(stage_c_manifest())


def _canonical_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def stage_c_dataset(
    *,
    root_seed: int,
    counts: dict[str, int],
    splits: tuple[str, ...] = ("train", "id_validation", "id_test", "heldout_law_ood"),
) -> dict[str, Any]:
    expected = {"train", "id_validation", "id_test", "heldout_law_ood"}
    if set(counts) != expected or any(isinstance(value, bool) or value <= 0 for value in counts.values()):
        raise ValueError("Stage C counts must contain four positive frozen splits")
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= expected:
        raise ValueError("Stage C splits must be a unique nonempty subset of the frozen splits")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    config = SimulatorConfig(steps=6)
    for split in splits:
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
        "counts": {name: counts[name] for name in splits},
        "materialized_splits": list(splits),
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


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("ROC AUC scores and labels must be nonempty and aligned")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both relation classes")
    ordered = sorted(enumerate(scores), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(scores)
    left = 0
    while left < len(ordered):
        right = left + 1
        while right < len(ordered) and ordered[right][1] == ordered[left][1]:
            right += 1
        average_rank = (left + 1 + right) / 2.0
        for offset in range(left, right):
            ranks[ordered[offset][0]] = average_rank
        left = right
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _balanced_accuracy(scores: list[float], labels: list[int], threshold: float) -> float:
    positives = [score >= threshold for score, label in zip(scores, labels) if label == 1]
    negatives = [score < threshold for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("balanced accuracy requires both classes")
    return 0.5 * (sum(positives) / len(positives) + sum(negatives) / len(negatives))


def _targets(rows: list[dict[str, Any]], relations: Any, partitions: Any, laws: Any, kind: str):
    import torch

    if kind == "relations":
        return relations
    if kind == "partition":
        return partitions
    if kind == "law":
        return torch.nn.functional.one_hot(laws, num_classes=len(LAW_FAMILIES)).to(torch.float64)
    raise ValueError(kind)


def _probe_score(prediction: Any, target: Any, kind: str) -> float:
    if kind == "law":
        return float((prediction.argmax(dim=1) == target.argmax(dim=1)).double().mean())
    if kind == "relations":
        return _binary_auc(
            [float(value) for value in prediction.reshape(-1)],
            [int(value) for value in target.reshape(-1)],
        )
    return _balanced_accuracy(
        [float(value) for value in prediction.reshape(-1)],
        [int(value) for value in target.reshape(-1)],
        0.5,
    )


def _nested_probe(x: Any, y: Any, world_seeds: list[int], *, kind: str, alphas: list[float]):
    import torch

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
                scores.append(_probe_score(_ridge_predict(x[val_mask], weights), y[val_mask], kind))
            candidates.append((sum(scores) / len(scores), -alpha, alpha))
        alpha = max(candidates)[2]
        selected.append(alpha)
        weights = _ridge_fit(x[outer_train], y[outer_train], alpha)
        score = _probe_score(_ridge_predict(x[outer_test], weights), y[outer_test], kind)
        outer_scores.append(score)
        fold_records.append({
            "outer_fold": fold,
            "selected_alpha": alpha,
            "outer_score": score,
            "score_metric": "roc_auc" if kind == "relations" else ("accuracy" if kind == "law" else "balanced_accuracy"),
            "train_worlds": int(outer_train.sum()),
            "test_worlds": int(outer_test.sum()),
            "relations_per_world_cofolded": 15 if kind == "relations" else None,
        })
    final_alpha = sorted(selected)[len(selected) // 2]
    weights = _ridge_fit(x, y, final_alpha)
    return weights, {
        "kind": kind,
        "outer_folds": fold_records,
        "outer_mean_score": sum(outer_scores) / len(outer_scores),
        "selected_alphas": selected,
        "final_alpha": final_alpha,
        "fold_group": "world_seed",
    }


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


def _relation_cluster_bootstrap(
    prediction: Any,
    target: Any,
    seeds: list[int],
    *,
    threshold: float,
    bootstrap_seed: int,
    replicates: int,
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 15:
        raise ValueError("relation probes require exactly 15 co-clustered pairs per world")
    if prediction.shape[0] != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("relation world clusters must be unique and aligned")
    score_rows = [[float(value) for value in row] for row in prediction]
    label_rows = [[int(value) for value in row] for row in target]
    scores = [value for row in score_rows for value in row]
    labels = [value for row in label_rows for value in row]
    auc = _binary_auc(scores, labels)
    balanced = _balanced_accuracy(scores, labels, threshold)
    rng = random.Random(bootstrap_seed)
    auc_samples: list[float] = []
    balanced_samples: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(len(seeds)) for _ in seeds]
        sampled_scores = [value for index in indices for value in score_rows[index]]
        sampled_labels = [value for index in indices for value in label_rows[index]]
        auc_samples.append(_binary_auc(sampled_scores, sampled_labels))
        balanced_samples.append(_balanced_accuracy(sampled_scores, sampled_labels, threshold))
    auc_samples.sort(); balanced_samples.sort()
    lower = int(0.025 * replicates); upper = min(replicates - 1, int(0.975 * replicates))
    return {
        "roc_auc": auc,
        "balanced_accuracy": balanced,
        "threshold": threshold,
        "roc_auc_clustered_ci95": [auc_samples[lower], auc_samples[upper]],
        "balanced_accuracy_clustered_ci95": [balanced_samples[lower], balanced_samples[upper]],
        "roc_auc_one_sided_bootstrap_p": (1 + sum(value <= 0.5 for value in auc_samples)) / (replicates + 1),
        "cluster_unit": "world_seed",
        "cluster_count": len(seeds),
        "relations_per_world": 15,
        "relations_cofolded_and_clustered": True,
        "predictions": prediction.tolist(),
    }


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


def _generic_probe_eval(
    prediction: Any,
    target: Any,
    *,
    kind: str,
    threshold: float | None,
    seeds: list[int],
    ci_seed: int,
    reps: int,
) -> dict[str, Any]:
    if kind == "law":
        correct = (prediction.argmax(1) == target).double()
        metric_name = "accuracy"
        chance = 1 / 7
    else:
        binary = ((prediction >= float(threshold)) == (target >= 0.5)).double()
        correct = binary.mean(dim=1)
        metric_name = "balanced_accuracy"
        chance = 0.5
    values = [float(value) for value in correct]
    return {
        metric_name: sum(values) / len(values),
        "threshold": threshold,
        f"{metric_name}_clustered_ci95": _clustered_ci(values, seeds, bootstrap_seed=ci_seed, replicates=reps),
        "cluster_unit": "world_seed",
        "cluster_count": len(values),
        "chance": chance,
        "one_sided_cluster_sign_p": _sign_test_p(values, chance),
        "predictions": prediction.tolist(),
    }


def _select_threshold(prediction: Any, target: Any, grid: list[float]) -> float:
    scores = [float(value) for value in prediction.reshape(-1)]
    labels = [int(value) for value in target.reshape(-1)]
    return max((_balanced_accuracy(scores, labels, threshold), -threshold, threshold) for threshold in grid)[2]


def _dataset_view(dataset: dict[str, Any], allowed_splits: set[str]) -> dict[str, Any]:
    records = [row for row in dataset["records"] if row["split"] in allowed_splits]
    if {row["split"] for row in records} != allowed_splits:
        raise ValueError("dataset view is missing a required sealed split")
    return {**{key: value for key, value in dataset.items() if key != "records"}, "records": records}


def fit_and_freeze_posthoc_probes(
    encoder: Any,
    train_dataset: dict[str, Any],
    id_validation_dataset: dict[str, Any],
    *,
    device: str,
    bootstrap_replicates: int,
    threshold_grid: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_hash = module_content_sha256(encoder)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("encoder must be frozen before post-hoc probe fitting")
    if {row["split"] for row in train_dataset["records"]} != {"train"}:
        raise RuntimeError("nested-CV probe fitting must receive train worlds only")
    if {row["split"] for row in id_validation_dataset["records"]} != {"id_validation"}:
        raise RuntimeError("phase-1 probe evaluation must receive ID validation worlds only")
    train_rows, train_z, _, relations, partitions, laws = _extract_latents(encoder, train_dataset, "train", device)
    mean, std = train_z.mean(0), train_z.std(0).clamp_min(1e-8)
    train_x = (train_z - mean) / std
    alphas = [0.001, 0.01, 0.1, 1.0]
    trained: dict[str, tuple[Any, dict[str, Any]]] = {}
    for kind in ("relations", "partition", "law"):
        y = _targets(train_rows, relations, partitions, laws, kind)
        trained[kind] = _nested_probe(
            train_x, y, [row["world_seed"] for row in train_rows], kind=kind, alphas=alphas
        )
    rows, z, _, id_relations, id_partitions, id_laws = _extract_latents(
        encoder, id_validation_dataset, "id_validation", device
    )
    x = (z - mean) / std
    seeds = [row["world_seed"] for row in rows]
    predictions = {kind: _ridge_predict(x, trained[kind][0]) for kind in trained}
    thresholds = {
        "relations": _select_threshold(predictions["relations"], id_relations, threshold_grid),
        "partition": _select_threshold(predictions["partition"], id_partitions, threshold_grid),
        "law": None,
    }
    evaluations = {
        "relations": _relation_cluster_bootstrap(
            predictions["relations"], id_relations, seeds, threshold=thresholds["relations"],
            bootstrap_seed=44123, replicates=bootstrap_replicates,
        ),
        "partition": _generic_probe_eval(
            predictions["partition"], id_partitions, kind="partition", threshold=thresholds["partition"],
            seeds=seeds, ci_seed=44124, reps=bootstrap_replicates,
        ),
        "law": _generic_probe_eval(
            predictions["law"], id_laws, kind="law", threshold=None, seeds=seeds, ci_seed=44125,
            reps=bootstrap_replicates,
        ),
    }
    pvalues = {
        "relations:roc_auc": evaluations["relations"]["roc_auc_one_sided_bootstrap_p"],
        "partition:balanced_accuracy": evaluations["partition"]["one_sided_cluster_sign_p"],
        "law:accuracy": evaluations["law"]["one_sided_cluster_sign_p"],
    }
    qvalues = _bh_fdr(pvalues)
    for name, qvalue in qvalues.items():
        kind = name.split(":", 1)[0]
        evaluations[kind]["bh_fdr_q"] = qvalue
    state = {
        "mean": mean,
        "std": std,
        "models": trained,
        "thresholds": thresholds,
        "encoder_frozen_sha256": frozen_hash,
    }
    artifact = {
        "schema_version": "jump.track-h-stage-c-probe-freeze/v1",
        "phase": "phase_1_id_validation_frozen_before_final_unseal",
        "encoder_frozen_sha256": frozen_hash,
        "standardization": {"mean": mean.tolist(), "std": std.tolist()},
        "thresholds": thresholds,
        "bootstrap_replicates": bootstrap_replicates,
        "nested_cv": {kind: value[1] for kind, value in trained.items()},
        "weights": {kind: value[0].tolist() for kind, value in trained.items()},
        "id_validation": evaluations,
        "bh_fdr_family": sorted(pvalues),
        "bh_fdr_scope": "id_validation_only",
        "relation_pairs_per_world": 15,
        "relation_pairs_cofolded_and_clustered": True,
        "encoder_gradient_from_probe": False,
    }
    artifact["probe_freeze_sha256"] = sha256_json(artifact)
    if module_content_sha256(encoder) != frozen_hash or any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("post-hoc probes changed or backpropagated into encoder")
    return artifact, state


def evaluate_frozen_probes(
    encoder: Any,
    state: dict[str, Any],
    dataset: dict[str, Any],
    *,
    split: str,
    device: str,
    bootstrap_replicates: int,
    probe_freeze_sha256: str,
) -> dict[str, Any]:
    if split not in {"heldout_law_ood"}:
        raise ValueError("phase-2 frozen probes may only unseal heldout-law OOD")
    if {row["split"] for row in dataset["records"]} != {split}:
        raise RuntimeError("phase-2 probe evaluation received mixed or unsealed splits")
    if module_content_sha256(encoder) != state["encoder_frozen_sha256"]:
        raise RuntimeError("encoder changed after phase-1 probe freeze")
    rows, z, _, relations, partitions, laws = _extract_latents(encoder, dataset, split, device)
    x = (z - state["mean"]) / state["std"]
    seeds = [row["world_seed"] for row in rows]
    predictions = {kind: _ridge_predict(x, state["models"][kind][0]) for kind in state["models"]}
    evaluations = {
        "relations": _relation_cluster_bootstrap(
            predictions["relations"], relations, seeds, threshold=state["thresholds"]["relations"],
            bootstrap_seed=45123, replicates=bootstrap_replicates,
        ),
        "partition": _generic_probe_eval(
            predictions["partition"], partitions, kind="partition", threshold=state["thresholds"]["partition"],
            seeds=seeds, ci_seed=45124, reps=bootstrap_replicates,
        ),
        "law": _generic_probe_eval(
            predictions["law"], laws, kind="law", threshold=None, seeds=seeds, ci_seed=45125,
            reps=bootstrap_replicates,
        ),
    }
    evaluations["law"]["interpretation"] = "descriptive_only: heldout law is an unseen class"
    pvalues = {
        "relations:roc_auc": evaluations["relations"]["roc_auc_one_sided_bootstrap_p"],
        "partition:balanced_accuracy": evaluations["partition"]["one_sided_cluster_sign_p"],
    }
    for name, qvalue in _bh_fdr(pvalues).items():
        evaluations[name.split(":", 1)[0]]["bh_fdr_q"] = qvalue
    return {
        "schema_version": "jump.track-h-stage-c-final-ood-probes/v1",
        "phase": "phase_2_final_ood_unsealed_after_probe_freeze",
        "probe_freeze_sha256": probe_freeze_sha256,
        "bootstrap_replicates": bootstrap_replicates,
        "evaluations": evaluations,
        "bh_fdr_family": sorted(pvalues),
        "bh_fdr_scope": "final_ood_only_never_mixed_with_id",
        "ood_used_for_selection": False,
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
    last_observed_positions = inputs[:, -1, :, :2]
    persistence_error = last_observed_positions - targets
    persistence_rmse = torch.sqrt(torch.mean(persistence_error**2))
    model_nrmse = rmse / scale
    persistence_nrmse = persistence_rmse / scale
    pixel_scale = (512 - 64) / 6.0
    norms = torch.linalg.vector_norm(z, dim=1)
    return {
        "examples": len(rows),
        "rollout_horizon_steps": 1,
        "future_position_nrmse": float(model_nrmse.cpu()),
        "persistence_position_nrmse": float(persistence_nrmse.cpu()),
        "persistence_relative_improvement": float(((persistence_nrmse - model_nrmse) / persistence_nrmse.clamp_min(1e-8)).cpu()),
        "nrmse_normalizer": "target_standard_deviation_shared_by_model_and_persistence",
        "position_rmse": float(rmse.cpu()),
        "position_mae": float(torch.mean(torch.abs(error)).cpu()),
        "render_coordinate_pixel_rmse": float((rmse * pixel_scale).cpu()),
        "latent_norm_min": float(norms.min().cpu()),
        "latent_norm_mean": float(norms.mean().cpu()),
        "latent_norm_max": float(norms.max().cpu()),
        "first_prediction": predicted[0].detach().cpu().tolist(),
    }


def evaluate_partition_swaps(
    encoder: Any,
    decoder: Any,
    *,
    device: str,
    pairs: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
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
        paired_differences = [row["donor_mse"] - row["own_mse"] for row in rows]
        paired_ci = _clustered_ci(
            paired_differences,
            list(range(len(paired_differences))),
            bootstrap_seed=55123 + (0 if direction == "A_to_B" else 1),
            replicates=bootstrap_replicates,
        )
        aggregate[direction] = {
            "pairs": len(rows),
            "own_mse": sum(row["own_mse"] for row in rows) / len(rows),
            "donor_mse": sum(row["donor_mse"] for row in rows) / len(rows),
            "scrambled_mse": sum(row["scrambled_mse"] for row in rows) / len(rows),
            "donor_minus_own_mse": sum(row["donor_mse"] - row["own_mse"] for row in rows) / len(rows),
            "donor_minus_own_mse_paired_ci95": paired_ci,
            "per_pair_success_rate": sum(value > 0 for value in paired_differences) / len(paired_differences),
            "success": paired_ci[0] > 0 and sum(value > 0 for value in paired_differences) / len(paired_differences) >= 0.75,
        }
    return {
        "schema_version": "jump.track-h-stage-c-partition-swap/v1",
        "gemma_used": False,
        "pair_definition": "partition-only with matched initial frame, nuisance, and law; noncoincident consequence",
        "directions": directions,
        "aggregate": aggregate,
        "paired_bootstrap_replicates": bootstrap_replicates,
        "success_threshold": "paired CI lower > 0 and >=75% donor-minus-own MSE values > 0 in both directions",
    }


def _atomic_checkpoint(path: Path, value: Any, commit: Callable[[], None] | None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    if commit is not None:
        commit()


def _load_safe_checkpoint(path: Path, *, device: str) -> dict[str, Any]:
    """Load tensor/primitive checkpoint data without permitting pickle globals."""
    import torch

    value = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(value, dict):
        raise RuntimeError("Stage C checkpoint must be a tensor/primitive mapping")
    return value


def _run_stage_c_seed(
    *,
    output_root: Path,
    checkpoint_root: Path,
    expected_manifest_sha256: str,
    seed_config: dict[str, Any],
    code_sha: str,
    experiment_id: str,
    experiment_spec_sha256: str,
    device: str = "cuda",
    checkpoint_commit: Callable[[], None] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one frozen Stage C seed under the aggregate caller-owned root."""
    import torch
    from safetensors.torch import load_file, save_file

    if expected_manifest_sha256 != STAGE_C_MANIFEST_SHA256:
        raise ValueError("Stage C manifest hash mismatch before output-root creation")
    if not isinstance(code_sha, str) or len(code_sha) != 40 or any(ch not in "0123456789abcdef" for ch in code_sha):
        raise ValueError("Stage C code_sha must be an exact lowercase 40-hex revision")
    if output_root.exists():
        raise FileExistsError("immutable Stage C output root already exists")
    manifest = stage_c_manifest()
    settings = manifest["dataset"]["counts"] if not dry_run else {"train": 24, "id_validation": 8, "id_test": 8, "heldout_law_ood": 8}
    steps = manifest["world_model"]["steps"] if not dry_run else 12
    pairs = manifest["partition_only_swap"]["pairs"] if not dry_run else 4
    bootstrap = manifest["posthoc_probes"]["clustered_ci"]["bootstrap_replicates"] if not dry_run else 50
    swap_bootstrap = manifest["partition_only_swap"]["paired_inference"]["bootstrap_replicates"] if not dry_run else 50
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote Stage C")
    torch.manual_seed(seed_config["parameter_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_config["parameter_seed"])
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    phase1_dataset = stage_c_dataset(
        root_seed=seed_config["dataset_root_seed"],
        counts=settings,
        splits=("train", "id_validation"),
    )
    phase1_dataset_sha = sha256_json(phase1_dataset)
    encoder, decoder = build_world_modules()
    encoder.to(device); decoder.to(device)
    _, train_inputs, train_targets, _, _ = dataset_tensors(phase1_dataset, "train", device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=manifest["world_model"]["learning_rate"])
    checkpoint_path = checkpoint_root / "latest.pt"
    start_step = 0
    if checkpoint_path.exists():
        checkpoint = _load_safe_checkpoint(checkpoint_path, device=device)
        binding = (checkpoint.get("manifest_sha256"), checkpoint.get("phase1_dataset_sha256"), checkpoint.get("code_sha"), checkpoint.get("seed_id"))
        if binding != (STAGE_C_MANIFEST_SHA256, phase1_dataset_sha, code_sha, seed_config["seed_id"]):
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
                "phase1_dataset_sha256": phase1_dataset_sha,
                "code_sha": code_sha,
                "seed_id": seed_config["seed_id"],
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
    # Phase 1 sees train and ID-validation only. Probe weights, normalization,
    # thresholds, and the ID-only BH family are sealed before any final split is
    # materialized for evaluation.
    metrics = {
        split: _prediction_metrics(encoder, decoder, _dataset_view(phase1_dataset, {split}), split, device)
        for split in ("train", "id_validation")
    }
    probe_freeze, probe_state = fit_and_freeze_posthoc_probes(
        encoder,
        _dataset_view(phase1_dataset, {"train"}),
        _dataset_view(phase1_dataset, {"id_validation"}),
        device=device,
        bootstrap_replicates=bootstrap,
        threshold_grid=manifest["posthoc_probes"]["threshold_grid"],
    )
    probe_freeze_sha256 = probe_freeze["probe_freeze_sha256"]
    # Phase 2 unseals final ID test and heldout-law OOD exactly once after the
    # phase-1 probe artifact is immutable by content hash.
    final_dataset = stage_c_dataset(
        root_seed=seed_config["dataset_root_seed"],
        counts=settings,
        splits=("id_test", "heldout_law_ood"),
    )
    final_dataset_sha = sha256_json(final_dataset)
    dataset_sha = sha256_json(
        {
            "schema_version": "jump.track-h-stage-c-two-phase-dataset-binding/v1",
            "phase1_dataset_sha256": phase1_dataset_sha,
            "probe_freeze_sha256": probe_freeze_sha256,
            "final_dataset_sha256": final_dataset_sha,
        }
    )
    for split in ("id_test", "heldout_law_ood"):
        metrics[split] = _prediction_metrics(
            encoder, decoder, _dataset_view(final_dataset, {split}), split, device
        )
    final_ood_probes = evaluate_frozen_probes(
        encoder, probe_state, _dataset_view(final_dataset, {"heldout_law_ood"}),
        split="heldout_law_ood", device=device, bootstrap_replicates=bootstrap,
        probe_freeze_sha256=probe_freeze_sha256,
    )
    swaps = evaluate_partition_swaps(
        encoder, decoder, device=device, pairs=pairs, bootstrap_replicates=swap_bootstrap
    )
    if (encoder_hash, decoder_hash) != (module_content_sha256(encoder), module_content_sha256(decoder)):
        raise RuntimeError("Stage C weights changed after freeze/evaluation/probes")

    output_root.mkdir(parents=True, exist_ok=False)
    save_file({name: value.detach().cpu() for name, value in encoder.state_dict().items()}, output_root / "encoder.safetensors")
    save_file({name: value.detach().cpu() for name, value in decoder.state_dict().items()}, output_root / "decoder.safetensors")
    _canonical_write(output_root / "training-manifest.json", manifest)
    _canonical_write(output_root / "phase1-train-id-validation-dataset.json", phase1_dataset)
    _canonical_write(output_root / "phase2-final-id-ood-dataset.json", final_dataset)
    _canonical_write(output_root / "probe-freeze.json", probe_freeze)
    _canonical_write(output_root / "final-ood-probes.json", final_ood_probes)
    _canonical_write(output_root / "swap-results.json", swaps)
    _canonical_write(output_root / "model-config.json", {"encoder": "96->64->16", "decoder": "16->64->12", "initialization": seed_config})

    ood_rows, ood_inputs, _, _, _ = dataset_tensors(final_dataset, "heldout_law_ood", device)
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
            "experiment_id": experiment_id,
            "experiment_spec_sha256": experiment_spec_sha256,
            "code_sha": code_sha,
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
        learned_decoder=learned_decoder_identity(artifact_name="decoder.safetensors", artifact_sha256=decoder_artifact_sha, training_manifest_sha256=STAGE_C_MANIFEST_SHA256, code_version=code_sha, architecture="same-z-16d-to-six-object-next-position-v1"),
        decoded_image=svg_bytes, decoded_image_media_type="image/svg+xml", answer=answer,
    )
    sealed = seal_learned_latent_result(evidence, source="cached", manifest_sha256=STAGE_C_MANIFEST_SHA256, run_id=f"authentic-stage-c-{seed_config['seed_id']}", code_version=code_sha, checkpoint_id=decoder_artifact_sha)
    _canonical_write(output_root / "learned-latent-evidence.json", evidence)
    _canonical_write(output_root / "sealed-result.json", sealed)

    loss_ratio = float(loss.detach().cpu()) / float(initial_loss.detach().cpu())
    donor_pass = all(value["success"] for value in swaps["aggregate"].values())
    persistence_pass = all(
        metrics[split]["persistence_relative_improvement"] >= 0.20
        for split in ("id_test", "heldout_law_ood")
    )
    relation = final_ood_probes["evaluations"]["relations"]
    relation_pass = relation["roc_auc"] >= 0.75 and relation["roc_auc_clustered_ci95"][0] > 0.50
    pass_gate = (
        loss_ratio <= 0.2
        and metrics["id_test"]["future_position_nrmse"] <= 0.20
        and metrics["heldout_law_ood"]["future_position_nrmse"] <= 0.30
        and persistence_pass
        and relation_pass
        and donor_pass
    )
    kill = metrics["heldout_law_ood"]["future_position_nrmse"] > 0.50
    # The reduced CPU dry run verifies only data/model/artifact plumbing.  Its
    # 12-step metrics are not the frozen 2,000-step experiment and therefore
    # must not trigger or masquerade as a scientific pass/pivot/kill decision.
    decision = "dry_run_not_evaluated" if dry_run else ("kill" if kill else ("pass" if pass_gate else "pivot"))
    result = {
        "schema_version": STAGE_C_RESULT_VERSION,
        "status": "completed", "stage": "C", "decision": decision,
        "claim_label": manifest["claim_label"], "mechanistic_evidence": False, "g2": False, "g3": False,
        "manifest_sha256": STAGE_C_MANIFEST_SHA256,
        "dataset_sha256": dataset_sha,
        "phase1_dataset_sha256": phase1_dataset_sha,
        "final_dataset_sha256": final_dataset_sha,
        "initial_loss": float(initial_loss.detach().cpu()), "final_loss": float(loss.detach().cpu()), "loss_ratio": loss_ratio,
        "metrics": metrics,
        "probe_freeze_sha256": probe_freeze_sha256,
        "final_ood_probes_sha256": hashlib.sha256((output_root / "final-ood-probes.json").read_bytes()).hexdigest(),
        "swap_results_sha256": hashlib.sha256((output_root / "swap-results.json").read_bytes()).hexdigest(),
        "encoder_state_sha256": encoder_hash, "decoder_state_sha256": decoder_hash,
        "encoder_weights_sha256": encoder_artifact_sha, "decoder_weights_sha256": decoder_artifact_sha,
        "weights_frozen_before_evaluation_and_probes": True,
        "evaluation_protocol": "phase_1_train_and_id_validation_freeze_then_phase_2_final_id_and_ood_unseal",
        "persistence_gate_passed": persistence_pass,
        "relation_probe_gate_passed": relation_pass,
        "final_ood_relation": {
            "roc_auc": relation["roc_auc"],
            "balanced_accuracy": relation["balanced_accuracy"],
            "roc_auc_clustered_ci95": relation["roc_auc_clustered_ci95"],
        },
        "donor_swap_gate_passed": donor_pass,
        "donor_swap_summary": swaps["aggregate"],
        "code_sha": code_sha,
        "seed_id": seed_config["seed_id"],
        "experiment_id": experiment_id,
        "experiment_spec_sha256": experiment_spec_sha256,
        "training_seconds": training_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "world_latent_sha256": evidence["tensor"]["world_latent_sha256"],
        "image_source": "learned_decoder_prediction", "dry_run": dry_run,
    }
    _canonical_write(output_root / "seed-terminal.json", result)

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


def run_stage_c(
    *,
    output_root: Path,
    checkpoint_root: Path,
    expected_manifest_sha256: str,
    expected_code_sha: str,
    experiment_spec: dict[str, Any],
    device: str = "cuda",
    checkpoint_commit: Callable[[], None] | None = None,
    dry_run: bool = False,
    precreated_empty_output_root: bool = False,
) -> dict[str, Any]:
    """Run all three frozen Stage C seeds serially and write canonical evidence."""
    if expected_manifest_sha256 != STAGE_C_MANIFEST_SHA256:
        raise ValueError("Stage C manifest hash mismatch before output-root creation")
    if not isinstance(expected_code_sha, str) or len(expected_code_sha) != 40 or any(
        char not in "0123456789abcdef" for char in expected_code_sha
    ):
        raise ValueError("expected_code_sha must be an exact lowercase 40-hex revision")
    plan = validate_experiment_spec(experiment_spec)
    experiment_spec_sha256 = sha256_json(plan)
    if experiment_spec_sha256 != STAGE_C_LAUNCH_SPEC_SHA256:
        raise ValueError("Stage C requires the exact frozen canonical launch spec")
    if output_root.exists():
        if not precreated_empty_output_root or any(output_root.iterdir()):
            raise FileExistsError("immutable Stage C aggregate output root already exists")
    actual_code_sha = os.environ.get("JUMP_CODE_VERSION")
    if not dry_run and actual_code_sha != expected_code_sha:
        raise RuntimeError("deployed Stage C requires explicit matching JUMP_CODE_VERSION")
    manifest = stage_c_manifest()
    seeds = manifest["initialization"]["seeds"]
    if len(seeds) != 3 or len({item["parameter_seed"] for item in seeds}) != 3 or len(
        {item["dataset_root_seed"] for item in seeds}
    ) != 3:
        raise RuntimeError("Stage C requires exactly three independent frozen seeds")
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    aggregate_started = time.monotonic()
    seed_results: list[dict[str, Any]] = []
    for seed_config in seeds:
        seed_result = _run_stage_c_seed(
            output_root=output_root / seed_config["seed_id"],
            checkpoint_root=checkpoint_root / seed_config["seed_id"],
            expected_manifest_sha256=expected_manifest_sha256,
            seed_config=seed_config,
            code_sha=expected_code_sha,
            experiment_id=plan["experiment_id"],
            experiment_spec_sha256=experiment_spec_sha256,
            device=device,
            checkpoint_commit=checkpoint_commit,
            dry_run=dry_run,
        )
        seed_results.append(seed_result)

    runtime_seconds = time.monotonic() - aggregate_started
    estimated_cost_usd = (
        0.0
        if dry_run
        else runtime_seconds / 3600 * manifest["execution"]["h100_rate_usd_per_hour"]
    )
    aggregate_pass = all(result["decision"] == "pass" for result in seed_results)
    aggregate_decision = (
        "dry_run_not_evaluated"
        if dry_run
        else ("pass" if aggregate_pass else ("kill" if any(result["decision"] == "kill" for result in seed_results) else "pivot"))
    )
    terminal = {
        "schema_version": STAGE_C_RESULT_VERSION,
        "status": "completed",
        "stage": "C",
        "decision": aggregate_decision,
        "claim_label": manifest["claim_label"],
        "manifest_sha256": STAGE_C_MANIFEST_SHA256,
        "code_sha": expected_code_sha,
        "modal_function": manifest["execution"]["modal_function"],
        "runtime_seconds": runtime_seconds,
        "peak_cuda_memory_bytes": max(result["peak_cuda_memory_bytes"] for result in seed_results),
        "estimated_cost_usd": estimated_cost_usd,
        "forecast_cost_usd": manifest["execution"]["retry_aware_forecast_usd"],
        "hard_ceiling_usd": manifest["execution"]["hard_ceiling_usd"],
        "three_seed_complete": len(seed_results) == 3,
        "seed_ids": [result["seed_id"] for result in seed_results],
        "all_seed_persistence_gate_passed": all(result["persistence_gate_passed"] for result in seed_results),
        "all_seed_relation_probe_gate_passed": all(result["relation_probe_gate_passed"] for result in seed_results),
        "all_seed_donor_swap_gate_passed": all(result["donor_swap_gate_passed"] for result in seed_results),
        "mechanistic_evidence": False,
        "g2": False,
        "g3": False,
        "dry_run": dry_run,
        "seed_results": [
            {
                "seed_id": result["seed_id"],
                "terminal_path": f"{result['seed_id']}/seed-terminal.json",
                "terminal_sha256": hashlib.sha256(
                    (output_root / result["seed_id"] / "seed-terminal.json").read_bytes()
                ).hexdigest(),
                "dataset_sha256": result["dataset_sha256"],
                "decision": result["decision"],
            }
            for result in seed_results
        ],
    }
    task_metrics: list[dict[str, Any]] = []
    for result in seed_results:
        seed = result["seed_id"]
        task_metrics.extend(
            [
                {"name": f"{seed}.id_test.future_position_nrmse", "value": result["metrics"]["id_test"]["future_position_nrmse"]},
                {"name": f"{seed}.heldout_ood.future_position_nrmse", "value": result["metrics"]["heldout_law_ood"]["future_position_nrmse"]},
                {"name": f"{seed}.id_test.persistence_relative_improvement", "value": result["metrics"]["id_test"]["persistence_relative_improvement"]},
                {"name": f"{seed}.heldout_ood.persistence_relative_improvement", "value": result["metrics"]["heldout_law_ood"]["persistence_relative_improvement"]},
                {"name": f"{seed}.heldout_ood.relation_roc_auc", "value": result["final_ood_relation"]["roc_auc"]},
                {"name": f"{seed}.heldout_ood.relation_roc_auc_ci_lower", "value": result["final_ood_relation"]["roc_auc_clustered_ci95"][0]},
            ]
        )
    evidence = write_track_h_task_evidence(
        output_root,
        metrics=task_metrics,
        terminal=terminal,
        experiment_spec=plan,
        terminal_name="aggregate-terminal.json",
        track_h={
            "stage": "C",
            "manifest_sha256": STAGE_C_MANIFEST_SHA256,
            "code_sha": expected_code_sha,
            "runtime_seconds": runtime_seconds,
            "estimated_cost_usd": estimated_cost_usd,
            "mechanistic_evidence": False,
            "dry_run": dry_run,
        },
    )
    return {
        **terminal,
        "experiment_id": plan["experiment_id"],
        "experiment_spec_sha256": experiment_spec_sha256,
        "task_evidence": evidence,
        "artifact_root": str(output_root),
        "artifact_verification": "passed",
    }
