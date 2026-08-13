"""Versioned, single-run interface for a Modal (or local) orchestrator.

This module deliberately does not schedule phases, retries, or GPUs. It accepts
one run selected from the shared manifest and returns/writes one immutable-style
result object. The Modal runner owns orchestration and run-directory lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

from jump_contracts.evidence import artifact_declaration, write_task_evidence

from .capture import ActivationCapture, CapturePolicy
from .interventions import (
    MatchedWorldPair,
    ablate,
    build_control_directions,
    evaluate_latent_swap,
    inject,
)
from .metrics import (
    ConfirmatoryEvidence,
    contrast_effects,
    evaluate_confirmatory_gates,
    mediation_analysis,
    mediation_specificity,
    paired_effect,
)
from .probes import ProbeSample, heldout_probe_evaluation, ood_law_family_evaluation
from .scoring import score_dataset
from .synthetic import load_fixture
from .vectors import dot, norm


MANIFEST_SCHEMA_VERSION = "jump.experiments/v1"
RESULT_SCHEMA_VERSION = "jump.run-result/v1"
SUPPORTED_TASKS = {"jump_mechanistic.runner", "mechanistic_suite.synthetic"}


def execute_run(
    manifest: dict[str, Any], *, phase_id: str, run_id: str, output_dir: str | Path
) -> dict[str, Any]:
    """Validate and execute exactly one manifest run.

    Parameters are read from ``run.task.parameters``. Layer/timepoint requests
    may be on the run itself or inherited from ``manifest.preregistration``;
    either way they must be subsets of the preregistered allowlists.
    """
    validate_manifest_header(manifest)
    phase, run = _select_run(manifest, phase_id, run_id)
    preregistration = manifest.get("preregistration", {})
    allowed_layers = _string_list(preregistration, "layer_allowlist")
    allowed_timepoints = _string_list(preregistration, "timepoint_allowlist")
    selection = run.get("selection", {})
    requested_layers = selection.get("layers", allowed_layers)
    requested_timepoints = selection.get("timepoints", allowed_timepoints)
    if not set(requested_layers) <= set(allowed_layers):
        raise PermissionError("run requests a layer outside preregistration.layer_allowlist")
    if not set(requested_timepoints) <= set(allowed_timepoints):
        raise PermissionError("run requests a timepoint outside preregistration.timepoint_allowlist")
    policy = CapturePolicy.from_allowlists(list(requested_layers), list(requested_timepoints))

    task = run.get("task", {})
    module = task.get("module") or task.get("command")
    if module not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task module: {module!r}")
    parameters = dict(task.get("parameters", {}))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = synthetic_suite_task(parameters, policy=policy, output_dir=target)
    manifest_sha = _sha256_bytes(_canonical_json(manifest))
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "phase_id": phase["id"],
        "run_id": run["id"],
        "status": "completed",
        "metrics": payload["metrics"],
        "artifacts": payload["artifacts"],
        "provenance": {
            "manifest_sha256": manifest_sha,
            "run_id": run["id"],
            "code_version": _code_version(),
            "python": platform.python_version(),
        },
    }
    _write_json(target / "result.json", result)
    return result


def synthetic_suite_task(
    parameters: dict[str, Any], *, policy: CapturePolicy, output_dir: Path
) -> dict[str, Any]:
    fixture = load_fixture(parameters.get("fixture_path"))
    seed = int(parameters.get("seed", 17))
    checkpoint_id = str(parameters.get("checkpoint_id", "synthetic-primary"))
    replication_id = str(parameters.get("replication_id", "fixture-replication"))
    _validate_checkpoint_preregistration(parameters, fixture)
    metrics: list[dict[str, Any]] = []

    capture = ActivationCapture(policy)
    for row in fixture["activations"]:
        if row["layer"] in policy.layers and row["timepoint"] in {p.value for p in policy.timepoints}:
            capture.capture(
                episode_id=row["episode_id"],
                checkpoint_id=row.get("checkpoint_id", checkpoint_id),
                layer=row["layer"],
                timepoint=row["timepoint"],
                activation=row["values"],
                labels=row.get("labels"),
            )
    if not capture.records:
        raise ValueError("fixture has no activations in the requested allowlists")
    activation_path = capture.write_jsonl(output_dir / "artifacts" / "activations.jsonl")
    _metric(metrics, "activation_capture.count", len(capture.records), split="all", checkpoint_id=checkpoint_id)
    for layer in sorted(policy.layers):
        for timepoint in sorted(policy.timepoints, key=lambda point: point.value):
            count = sum(
                record.layer == layer and record.timepoint == timepoint
                for record in capture.records
            )
            _metric(
                metrics,
                "activation_capture.count",
                count,
                split="all",
                checkpoint_id=checkpoint_id,
                layer=layer,
                timepoint=timepoint.value,
            )

    allowed_exponents = parameters.get("allowed_exponents")
    if allowed_exponents != fixture["allowed_exponents"]:
        raise ValueError("manifest allowed_exponents must exactly match the frozen fixture")
    behavior_scores = score_dataset(fixture["behavior"], allowed_exponents=allowed_exponents)
    for name, value in behavior_scores.items():
        _metric(metrics, f"behavior.{name}", value, split="test", checkpoint_id=checkpoint_id)

    samples = [ProbeSample(**sample) for sample in fixture["probe_samples"]]
    heldout = heldout_probe_evaluation(samples, seed=seed)
    for name, value in heldout["metrics"].items():
        _metric(metrics, f"probe.heldout.{name}", value, split="heldout", checkpoint_id=checkpoint_id)
    for family in sorted({sample.law_family for sample in samples}):
        ood = ood_law_family_evaluation(samples, heldout_family=family)
        for name, value in ood["metrics"].items():
            _metric(
                metrics,
                f"probe.ood.{name}",
                value,
                split="ood",
                condition=f"law_family={family}",
                checkpoint_id=checkpoint_id,
            )

    swap_rows: list[dict[str, Any]] = []
    for item in fixture["world_pairs"]:
        pair = MatchedWorldPair(**item)
        # Synthetic readout isolates the swapped latent while preserving the world input.
        swap = evaluate_latent_swap(pair, lambda _world_id, latent: float(latent[0]))
        swap_rows.append(swap)
        for name in ("swap_effect", "a_to_b_effect", "b_to_a_effect"):
            _metric(
                metrics,
                f"swap.{name}",
                swap[name],
                split="paired",
                condition="world_a_b_swap",
                checkpoint_id=checkpoint_id,
                world_pair_id=pair.pair_id,
            )

    directions = fixture["directions"]
    controls = build_control_directions(
        directions["target"], seed=seed, generic_error=directions["generic_error"]
    )
    example = directions["activation"]
    intervention_diagnostics = {
        "target_norm": norm(controls["target"]),
        "matched_norm": norm(controls["matched_norm"]),
        "orthogonal_norm": norm(controls["orthogonal"]),
        "orthogonal_dot": dot(controls["target"], controls["orthogonal"]),
        "generic_error_norm": norm(controls["generic_error"]),
        "ablated": ablate(example, controls["target"]),
        "injected": inject(example, controls["target"], magnitude=norm(controls["target"])),
    }
    _write_json(output_dir / "artifacts" / "interventions.json", intervention_diagnostics)

    intervention_clusters = fixture["intervention_cluster_ids"]
    effects = contrast_effects(
        fixture["intervention_outcomes"], cluster_ids=intervention_clusters, seed=seed
    )
    for condition, estimate in effects.items():
        for name in (
            "ate", "standard_error", "ci_low", "ci_high", "paired_standardized_effect",
            "cluster_count", "bootstrap_resamples", "bootstrap_seed",
        ):
            _metric(
                metrics,
                f"causal.{name}",
                estimate[name],
                split="paired",
                condition=condition,
                checkpoint_id=checkpoint_id,
            )
    target_outcomes = fixture["intervention_outcomes"]["target"]
    for control_name in ("matched_norm", "orthogonal", "generic_error"):
        comparison = paired_effect(
            target_outcomes,
            fixture["intervention_outcomes"][control_name],
            cluster_ids=intervention_clusters,
            seed=seed,
        )
        for name in (
            "ate", "standard_error", "ci_low", "ci_high", "paired_standardized_effect",
            "cluster_count", "bootstrap_resamples", "bootstrap_seed",
        ):
            _metric(
                metrics,
                f"causal.target_vs_control.{name}",
                comparison[name],
                split="paired",
                condition=control_name,
                checkpoint_id=checkpoint_id,
            )

    mediation_details: dict[str, Any] = {}
    checkpoint_mediation: dict[str, dict[str, Any]] = {}
    for check_id, values in fixture["mediation"].items():
        primary = mediation_analysis(
            values["treatment"],
            values["mediator"],
            values["outcome"],
            cluster_ids=values["cluster_ids"],
            seed=seed,
        )
        promotion = mediation_analysis(
            values["treatment"],
            values["promotion_mediator"],
            values["outcome"],
            cluster_ids=values["cluster_ids"],
            seed=seed,
        )
        mediator_controls = {
            name: mediation_analysis(
                values["treatment"],
                mediator,
                values["outcome"],
                cluster_ids=values["cluster_ids"],
                seed=seed,
            )
            for name, mediator in values["specificity_controls"].items()
        }
        specificity = mediation_specificity(primary, mediator_controls)
        mediation_details[check_id] = {
            "ordered": {"inadequacy": primary, "promotion": promotion},
            "controls": mediator_controls,
            "specificity": specificity,
        }
        checkpoint_mediation[check_id] = {
            "primary": primary,
            "promotion": promotion,
            "specificity": specificity,
        }
        for name in (
            "indirect_effect", "indirect_ci_low", "indirect_ci_high", "direct_effect",
            "direct_ci_low", "direct_ci_high", "total_effect", "total_ci_low", "total_ci_high",
            "mediation_proportion", "cluster_count", "bootstrap_resamples", "bootstrap_seed",
        ):
            _metric(metrics, f"mediation.{name}", primary[name], split="paired", checkpoint_id=check_id)
        _metric(
            metrics,
            "mediation.promotion_indirect_ci_low",
            promotion["indirect_ci_low"],
            split="paired",
            checkpoint_id=check_id,
        )
        _metric(
            metrics,
            "mediation.specificity_passed",
            float(specificity["passed"]),
            split="paired",
            checkpoint_id=check_id,
        )
    causal_replicates = {
        check_id: paired_effect(
            values["treated"],
            values["control"],
            cluster_ids=values["cluster_ids"],
            seed=seed,
        )
        for check_id, values in fixture["checkpoint_effects"].items()
    }
    ood_replicates = {
        check_id: paired_effect(
            values["treated"],
            values["control"],
            cluster_ids=values["cluster_ids"],
            seed=seed,
        )
        for check_id, values in fixture["ood_checkpoint_effects"].items()
    }
    confirmatory: dict[str, ConfirmatoryEvidence] = {}
    for check_id in sorted(checkpoint_mediation):
        mediation = checkpoint_mediation[check_id]
        causal = causal_replicates[check_id]
        ood_causal = ood_replicates[check_id]
        retention = (
            ood_causal["ate"] / causal["ate"]
            if causal["ate"] is not None and causal["ate"] > 0
            else 0.0
        )
        gate_input = fixture["checkpoint_gate_inputs"][check_id]
        confirmatory[check_id] = ConfirmatoryEvidence.from_dict(
            {
                "identity": fixture["checkpoint_identities"][check_id],
                "g3_passed": gate_input["g3_passed"],
                "g5_passed": gate_input["g5_passed"] and causal["ci_low"] > 0,
                "total_ci_low": mediation["primary"]["total_ci_low"],
                "ordered_nie_ci_lows": [
                    mediation["primary"]["indirect_ci_low"],
                    mediation["promotion"]["indirect_ci_low"],
                ],
                "mediated_proportion": mediation["primary"]["mediation_proportion"],
                "specificity_passed": mediation["specificity"]["passed"],
                "ood_effect_ci_low": ood_causal["ci_low"],
                "ood_retention": retention,
                "provenance_hash_match_rate": gate_input["provenance_hash_match_rate"],
            }
        )
    primary_key = checkpoint_id
    replication_key = replication_id
    gates = evaluate_confirmatory_gates(
        confirmatory.get(primary_key), confirmatory.get(replication_key)
    )
    for gate_name in ("g6", "g7", "g8"):
        _metric(
            metrics,
            f"gates.{gate_name}_passed",
            float(gates[gate_name]["passed"]),
            split="replication" if gate_name == "g8" else "confirmatory",
            condition=replication_id if gate_name == "g8" else None,
            checkpoint_id="all" if gate_name == "g8" else primary_key,
        )
    _write_json(
        output_dir / "artifacts" / "mediation.json",
        {
            "checkpoints": mediation_details,
            "causal_effects": causal_replicates,
            "ood_causal_effects": ood_replicates,
            "gates": gates,
        },
    )

    summary_path = output_dir / "artifacts" / "suite_summary.json"
    _write_json(
        summary_path,
        {
            "heldout_probe": {k: v for k, v in heldout.items() if k != "probe"},
            "swap_rows": swap_rows,
            "confirmatory_gates": gates,
        },
    )
    paths = [
        activation_path,
        output_dir / "artifacts" / "interventions.json",
        output_dir / "artifacts" / "mediation.json",
        summary_path,
    ]
    return {"metrics": metrics, "artifacts": [_artifact(path, output_dir) for path in paths]}


def run_manifest_file(
    manifest_path: str | Path, *, phase_id: str, run_id: str, output_dir: str | Path
) -> dict[str, Any]:
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return execute_run(manifest, phase_id=phase_id, run_id=run_id, output_dir=output_dir)


def execute_task_file(
    parameters_path: str | Path,
    *,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Implement the shared runner's ``python -m`` subprocess protocol.

    The sequential runner writes ``config.json`` two directories above its
    per-attempt parameter file. Reading selection and preregistration from that
    immutable config prevents task parameters from expanding capture scope.
    """
    parameter_file = Path(parameters_path).resolve()
    with parameter_file.open(encoding="utf-8") as handle:
        parameters = json.load(handle)
    if not isinstance(parameters, dict):
        raise ValueError("task parameters must be a JSON object")
    config_path = parameter_file.parents[2] / "config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        preregistration = config.get("preregistration", {})
        selection = config.get("selection", {})
        allowed_layers = _string_list(preregistration, "layer_allowlist")
        allowed_timepoints = _string_list(preregistration, "timepoint_allowlist")
        selected_layers = _string_list(selection, "layers")
        selected_timepoints = _string_list(selection, "timepoints")
        if not set(selected_layers) <= set(allowed_layers):
            raise PermissionError("runner config selects a non-preregistered layer")
        if not set(selected_timepoints) <= set(allowed_timepoints):
            raise PermissionError("runner config selects a non-preregistered timepoint")
    else:
        # Standalone task smoke tests may embed the already validated selection.
        selected_layers = _string_list(parameters, "layers")
        selected_timepoints = _string_list(parameters, "timepoints")
    policy = CapturePolicy.from_allowlists(selected_layers, selected_timepoints)
    task_name = parameters.get("task", "mechanistic_suite.synthetic")
    if task_name != "mechanistic_suite.synthetic":
        raise ValueError(f"unsupported mechanistic task: {task_name!r}")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    payload = synthetic_suite_task(parameters, policy=policy, output_dir=target)
    return write_task_evidence(
        target,
        metrics=payload["metrics"],
        artifacts=payload["artifacts"],
    )


def validate_manifest_header(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    if not isinstance(manifest.get("experiment_id"), str) or not manifest["experiment_id"]:
        raise ValueError("experiment_id is required")
    if not isinstance(manifest.get("phases"), list) or not manifest["phases"]:
        raise ValueError("phases must be a nonempty array")
    preregistration = manifest.get("preregistration")
    if not isinstance(preregistration, dict):
        raise ValueError("preregistration is required")
    CapturePolicy.from_allowlists(
        _string_list(preregistration, "layer_allowlist"),
        _string_list(preregistration, "timepoint_allowlist"),
    )


def _select_run(
    manifest: dict[str, Any], phase_id: str, run_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    phases = [phase for phase in manifest["phases"] if phase.get("id") == phase_id]
    if len(phases) != 1:
        raise ValueError(f"phase id must select exactly one phase: {phase_id}")
    runs = [run for run in phases[0].get("runs", []) if run.get("id") == run_id]
    if len(runs) != 1:
        raise ValueError(f"run id must select exactly one run: {run_id}")
    return phases[0], runs[0]


def _metric(
    metrics: list[dict[str, Any]],
    name: str,
    value: Any,
    *,
    split: str,
    checkpoint_id: str,
    **dimensions: Any,
) -> None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {name} must be a finite numeric value")
    if not math.isfinite(float(value)):
        raise ValueError(f"metric {name} must be finite")
    record = {"name": name, "value": value, "split": split, "checkpoint_id": checkpoint_id}
    record.update({key: value for key, value in dimensions.items() if value is not None})
    metrics.append(record)


def _artifact(path: Path, output_dir: Path) -> dict[str, Any]:
    return artifact_declaration(
        path,
        output_dir,
        name=path.stem,
        media_type="application/x-ndjson" if path.suffix == ".jsonl" else "application/json",
    )


def _string_list(mapping: dict[str, Any], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a nonempty string array")
    return value


def _validate_checkpoint_preregistration(
    parameters: dict[str, Any], fixture: dict[str, Any]
) -> None:
    pairs = {
        "checkpoint": {
            "checkpoint_id": parameters.get("checkpoint_id"),
            "model_revision": parameters.get("checkpoint_revision"),
            "tokenizer_revision": parameters.get("checkpoint_tokenizer_revision"),
            "training_lineage_id": parameters.get("checkpoint_training_lineage_id"),
            "checkpoint_sha256": parameters.get("checkpoint_sha256"),
        },
        "replication": {
            "checkpoint_id": parameters.get("replication_id"),
            "model_revision": parameters.get("replication_revision"),
            "tokenizer_revision": parameters.get("replication_tokenizer_revision"),
            "training_lineage_id": parameters.get("replication_training_lineage_id"),
            "checkpoint_sha256": parameters.get("replication_sha256"),
        },
    }
    identities = fixture.get("checkpoint_identities", {})
    for role, identity in pairs.items():
        checkpoint_key = identity["checkpoint_id"]
        if not isinstance(checkpoint_key, str) or identities.get(checkpoint_key) != identity:
            raise ValueError(f"{role} checkpoint identity must exactly match the frozen fixture")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    os.replace(temporary, path)


def _code_version() -> str:
    return os.environ.get("JUMP_CODE_VERSION", "unknown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a JUMP mechanistic task")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest")
    mode.add_argument("--parameters")
    parser.add_argument("--phase-id")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args(argv)
    try:
        if args.parameters:
            result = execute_task_file(
                args.parameters,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
            )
        else:
            if not args.phase_id or not args.run_id:
                parser.error("--manifest requires --phase-id and --run-id")
            result = run_manifest_file(
                args.manifest,
                phase_id=args.phase_id,
                run_id=args.run_id,
                output_dir=args.output_dir,
            )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
