"""Canonical task-evidence producer for the shared experiment runner.

This module does not interpret manifests, schedule phases, retry runs, write
``jump.run-result/v1``, or launch GPUs. The shared runner owns those boundaries;
this process accepts only its immutable per-attempt parameters/config and emits
one versioned task-evidence result for verified promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jump_contracts import TASK_EVIDENCE_VERSION, EvidenceError, artifact_declaration, write_task_evidence

from .capture import ActivationCapture, CapturePolicy
from .interventions import (
    MatchedWorldPair,
    ablate,
    build_control_directions,
    evaluate_latent_swap,
    inject,
)
from .gates import (
    BehaviorConditionRecord,
    InterventionOutcomeRecord,
    MediationArmRecord,
    PromotionAblationRecord,
    RegimeRecord,
    SwapOutcomeRecord,
    evaluate_g1,
    evaluate_g3,
    evaluate_g5,
    evaluate_g6,
)
from .metrics import (
    contrast_effects,
    observational_mediation_analysis,
    paired_effect,
)
from .probes import ProbeSample, heldout_probe_evaluation, ood_law_family_evaluation
from .scoring import score_dataset, score_episode
from .synthetic import load_fixture
from .vectors import dot, norm


MECHANISTIC_ARTIFACT_CONTRACT = {
    "activations": ("activation-evidence", "application/x-ndjson"),
    "interventions": ("intervention-control-evidence", "application/json"),
    "mediation": ("observational-mediation-evidence", "application/json"),
    "computed_gates": ("mechanistic-gate-evidence", "application/json"),
    "suite_summary": ("mechanistic-suite-summary", "application/json"),
}


def synthetic_suite_task(
    parameters: dict[str, Any], *, policy: CapturePolicy, output_dir: Path
) -> dict[str, Any]:
    forbidden_gate_flags = sorted(set(parameters) & {"g1_passed", "g3_passed", "g5_passed", "g6_passed"})
    if forbidden_gate_flags:
        raise ValueError(
            "trusted gate booleans are forbidden; provide raw content-addressed evidence records: "
            + ", ".join(forbidden_gate_flags)
        )
    forbidden_input_paths = sorted(
        set(parameters) & {"fixture_path", "fixture_sha256", "input_root"}
    )
    if forbidden_input_paths:
        raise ValueError(
            "mechanistic fixture selection is runner-owned; external paths and hashes are forbidden: "
            + ", ".join(forbidden_input_paths)
        )
    fixture = load_fixture()
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
    regime_records = [
        RegimeRecord(
            episode_id=f"fixture-behavior-{index}",
            parsed=isinstance(row.get("prediction"), dict),
            joint_correct=(
                score_episode(row["prediction"], row["target"])["joint_theory_accuracy"]
                == 1.0
            ),
        )
        for index, row in enumerate(fixture["behavior"])
    ]
    g1 = evaluate_g1(regime_records)
    _metric(metrics, "gates.g1_passed", 0.0, split="fixture_nonclaim", checkpoint_id=checkpoint_id)

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

    # The legacy fixture's mediator arrays remain available only as explicitly
    # observational descriptions. They are never supplied to G6.
    mediation_details: dict[str, Any] = {}
    for check_id, values in fixture["mediation"].items():
        primary = observational_mediation_analysis(
            values["treatment"],
            values["mediator"],
            values["outcome"],
            cluster_ids=values["cluster_ids"],
            seed=seed,
        )
        promotion = observational_mediation_analysis(
            values["treatment"],
            values["promotion_mediator"],
            values["outcome"],
            cluster_ids=values["cluster_ids"],
            seed=seed,
        )
        mediator_controls = {
            name: observational_mediation_analysis(
                values["treatment"],
                mediator,
                values["outcome"],
                cluster_ids=values["cluster_ids"],
                seed=seed,
            )
            for name, mediator in values["specificity_controls"].items()
        }
        mediation_details[check_id] = {
            "ordered": {"inadequacy": primary, "promotion": promotion},
            "controls": mediator_controls,
            "claim_eligibility": "exploratory_descriptive_only_not_g6",
        }
        for name in (
            "indirect_effect", "indirect_ci_low", "indirect_ci_high", "direct_effect",
            "direct_ci_low", "direct_ci_high", "total_effect", "total_ci_low", "total_ci_high",
            "mediation_proportion", "cluster_count", "bootstrap_resamples", "bootstrap_seed",
        ):
            _metric(metrics, f"observational_mediation.{name}", primary[name], split="paired", checkpoint_id=check_id)
        _metric(
            metrics,
            "observational_mediation.promotion_indirect_ci_low",
            promotion["indirect_ci_low"],
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
    raw_gate_artifacts: dict[str, Any] = {
        "schema_version": "jump.mechanistic-gate-diagnostics/v1",
        "evidence_namespace": "synthetic_fixture_nonclaim",
        "claim_eligible": False,
        "warning": "fixture outputs test contracts only and cannot set scientific gates",
        "g1": g1.to_dict(),
        "raw_records": {"g1_regime": [asdict(row) for row in regime_records]},
        "checkpoints": {},
    }
    for check_id in sorted(fixture["checkpoint_identities"]):
        causal = causal_replicates[check_id]
        ood_causal = ood_replicates[check_id]
        retention = (
            ood_causal["ate"] / causal["ate"]
            if causal["ate"] is not None and causal["ate"] > 0
            else 0.0
        )
        g3_records, swap_gate_records = _fixture_g3_records(check_id, fixture)
        g5_records = _fixture_g5_records(check_id, fixture)
        g6_records, ablation_records = _fixture_g6_records(check_id, fixture)
        g3 = evaluate_g3(g3_records, swap_gate_records, seed=seed)
        g5 = evaluate_g5(g5_records, seed=seed)
        g6 = evaluate_g6(g6_records, ablation_records, seed=seed)
        provenance_rate = sum(
            len(
                {
                    r.world_latent_sha256,
                    r.decoder_input_sha256,
                    r.injection_input_sha256,
                    r.answer_world_latent_sha256,
                    r.delivered_world_latent_sha256,
                }
            )
            == 1
            for r in swap_gate_records
        ) / len(swap_gate_records)
        raw_gate_artifacts["checkpoints"][check_id] = {
            "raw_records": {
                "g3_conditions": [asdict(row) for row in g3_records],
                "g3_swaps": [asdict(row) for row in swap_gate_records],
                "g5_interventions": [asdict(row) for row in g5_records],
                "g6_clamp_patch_arms": [asdict(row) for row in g6_records],
                "g6_promotion_ablation": [asdict(row) for row in ablation_records],
            },
            "g3": g3.to_dict(), "g5": g5.to_dict(), "g6": g6.to_dict(),
            "ood_diagnostic": {
                "effect_ci_low": ood_causal["ci_low"],
                "retention": retention,
                "provenance_hash_match_rate": provenance_rate,
            },
        }
    primary_key = checkpoint_id
    replication_key = replication_id
    gates = {
        gate: {
            "passed": False,
            "reasons": ["synthetic fixture evidence is not claim-eligible"],
        }
        for gate in ("g3", "g5", "g6", "g7", "g8")
    }
    for gate_name in ("g3", "g5", "g6", "g7", "g8"):
        _metric(
            metrics,
            f"gates.{gate_name}_passed",
            float(gates[gate_name]["passed"]),
            split="replication" if gate_name == "g8" else "confirmatory",
            condition=replication_id if gate_name == "g8" else None,
            checkpoint_id="all" if gate_name == "g8" else primary_key,
        )
    raw_gate_artifacts["confirmatory"] = gates
    _write_json(
        output_dir / "artifacts" / "mediation.json",
        {
            "checkpoints": mediation_details,
            "causal_effects": causal_replicates,
            "ood_causal_effects": ood_replicates,
            "gates": gates,
            "warning": "fixture outputs are contract tests, not mechanistic evidence",
        },
    )
    _write_json(output_dir / "artifacts" / "computed_gates.json", raw_gate_artifacts)

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
        output_dir / "artifacts" / "computed_gates.json",
        summary_path,
    ]
    return {"metrics": metrics, "artifacts": [_artifact(path, output_dir) for path in paths]}


def _fixture_g3_records(
    checkpoint_id: str, fixture: dict[str, Any]
) -> tuple[list[BehaviorConditionRecord], list[SwapOutcomeRecord]]:
    """Materialize raw CPU-fixture records; never a claim-bearing shortcut."""
    source = fixture["intervention_outcomes"]
    condition_source = {
        "E": "target",
        "G": "matched_norm",
        "W": "baseline",
        "I": "orthogonal",
        "C_prime": "generic_error",
        "T1c": "generic_error",
        "T2c": "baseline",
    }
    condition_records: list[BehaviorConditionRecord] = []
    for index in range(len(source["target"])):
        for condition, key in condition_source.items():
            condition_records.append(
                BehaviorConditionRecord(
                    checkpoint_id=checkpoint_id,
                    episode_id=f"{checkpoint_id}-g3-{index}",
                    cluster_id=fixture["intervention_cluster_ids"][index],
                    condition=condition,
                    joint_correct=float(source[key][index]) >= 0.5,
                    prompt_token_count=64,
                    decoding_sha256=_sha256_bytes(f"fixture-decoding:{checkpoint_id}".encode()),
                )
            )
    swaps: list[SwapOutcomeRecord] = []
    for pair in fixture["world_pairs"]:
        latent_hash = _sha256_bytes(_canonical_json(pair["latent_b"]))
        for direction in ("a_to_b", "b_to_a"):
            swaps.append(
                SwapOutcomeRecord.fixture_nonclaim(
                    checkpoint_id=checkpoint_id,
                    pair_id=pair["pair_id"],
                    cluster_id=f"{checkpoint_id}-{pair['pair_id']}",
                    direction=direction,
                    moved_toward_donor=True,
                    recipient_prompt_token_count=64,
                    donor_prompt_token_count=64,
                    latent_sha256=latent_hash,
                    answer_sha256=_sha256_bytes(
                        _canonical_json(
                            {"fixture_answer": pair["pair_id"], "direction": direction}
                        )
                    ),
                    donor_world_id=(
                        pair["world_a_id"] if direction == "a_to_b" else pair["world_b_id"]
                    ),
                    recipient_world_id=(
                        pair["world_b_id"] if direction == "a_to_b" else pair["world_a_id"]
                    ),
                )
            )
    return condition_records, swaps


def _fixture_g5_records(
    checkpoint_id: str, fixture: dict[str, Any]
) -> list[InterventionOutcomeRecord]:
    source = fixture["intervention_outcomes"]
    control_source = {
        "target": "target",
        "baseline": "baseline",
        "matched_norm": "matched_norm",
        "orthogonal": "orthogonal",
        "generic_error": "generic_error",
        "sham": "baseline",
        "prompt_length": "baseline",
    }
    records: list[InterventionOutcomeRecord] = []
    for kind in ("ablation", "injection"):
        for index in range(len(source["target"])):
            for condition, key in control_source.items():
                records.append(
                    InterventionOutcomeRecord(
                        checkpoint_id=checkpoint_id,
                        episode_id=f"{checkpoint_id}-g5-{index}",
                        cluster_id=fixture["intervention_cluster_ids"][index],
                        intervention_kind=kind,
                        condition=condition,
                        outcome=float(source[key][index]) >= 0.5,
                        parse_failed=False,
                        site_id="fixture.site/T3",
                        intervention_sha256=_sha256_bytes(
                            f"fixture:{checkpoint_id}:{kind}:{condition}:{index}".encode()
                        ),
                    )
                )
    return records


def _fixture_g6_records(
    checkpoint_id: str, fixture: dict[str, Any]
) -> tuple[list[MediationArmRecord], list[PromotionAblationRecord]]:
    effects = fixture["checkpoint_effects"][checkpoint_id]
    records: list[MediationArmRecord] = []
    ablations: list[PromotionAblationRecord] = []
    for index, (treated, control, cluster) in enumerate(
        zip(effects["treated"], effects["control"], effects["cluster_ids"])
    ):
        episode_id = f"{checkpoint_id}-g6-{index}"
        for stage in ("inadequacy", "promotion"):
            for mediator in ("primary", "matched_norm", "orthogonal", "generic_error", "sham", "prompt_length"):
                clamp = control + 0.2 * (treated - control) if mediator == "primary" else treated
                for arm, outcome in (
                    ("control_natural", control),
                    ("treated_natural", treated),
                    ("treated_control_clamp", clamp),
                ):
                    is_clamp = arm == "treated_control_clamp"
                    records.append(
                        MediationArmRecord(
                            checkpoint_id=checkpoint_id,
                            episode_id=episode_id,
                            cluster_id=cluster,
                            stage=stage,
                            mediator=mediator,
                            arm=arm,
                            outcome=float(outcome),
                            site_id=f"fixture.site/{stage}",
                            intervention_kind=("activation_clamp" if is_clamp else None),
                            intervention_id=(f"fixture-clamp:{stage}:{mediator}:{index}" if is_clamp else None),
                            source_activation_sha256=(
                                _sha256_bytes(f"fixture-activation:{stage}:{mediator}:{index}".encode())
                                if is_clamp else None
                            ),
                            result_sha256=_sha256_bytes(
                                f"fixture-result:{checkpoint_id}:{stage}:{mediator}:{arm}:{index}".encode()
                            ),
                        )
                    )
        ablations.append(
            PromotionAblationRecord(
                checkpoint_id=checkpoint_id,
                episode_id=episode_id,
                cluster_id=cluster,
                natural_promotion=float(treated),
                inadequacy_ablated_promotion=float(control),
                intervention_id=f"fixture-inadequacy-ablation:{index}",
                result_sha256=_sha256_bytes(
                    f"fixture-ablation-result:{checkpoint_id}:{index}".encode()
                ),
            )
        )
    return records, ablations


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
    if (target / "result.json").exists():
        raise EvidenceError(
            f"immutable evidence already exists: {target / 'result.json'}"
        )
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    payload = synthetic_suite_task(parameters, policy=policy, output_dir=target)
    result = write_task_evidence(
        target,
        metrics=payload["metrics"],
        artifacts=payload["artifacts"],
        evidence_namespace="synthetic_fixture_nonclaim",
        claim_eligible=False,
    )
    validate_mechanistic_task_evidence(result)
    return result


def validate_mechanistic_task_evidence(result: dict[str, Any]) -> None:
    """Reject the shared contract's lossy legacy fallback for this producer."""
    if result.get("schema_version") != TASK_EVIDENCE_VERSION:
        raise EvidenceError("mechanistic publication requires jump.task-evidence/v1")
    if result.get("evidence_namespace") != "synthetic_fixture_nonclaim":
        raise EvidenceError("mechanistic fixture evidence namespace is missing or incorrect")
    if result.get("claim_eligible") is not False:
        raise EvidenceError("mechanistic fixture evidence must be explicitly non-claim")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvidenceError("mechanistic task evidence requires declared artifacts")
    by_name = {record.get("name"): record for record in artifacts if isinstance(record, dict)}
    if set(by_name) != set(MECHANISTIC_ARTIFACT_CONTRACT) or len(artifacts) != len(by_name):
        raise EvidenceError("mechanistic task evidence must declare the exact stable artifact set")
    for name, (role, media_type) in MECHANISTIC_ARTIFACT_CONTRACT.items():
        record = by_name[name]
        if (
            record.get("role") != role
            or record.get("media_type") != media_type
            or record.get("evidence_namespace") != "synthetic_fixture_nonclaim"
            or record.get("claim_eligible") is not False
        ):
            raise EvidenceError(f"mechanistic artifact metadata drifted for {name}")


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
    if path.stem not in MECHANISTIC_ARTIFACT_CONTRACT:
        raise ValueError(f"mechanistic artifact has no stable evidence role: {path.name}")
    role, media_type = MECHANISTIC_ARTIFACT_CONTRACT[path.stem]
    return artifact_declaration(
        path,
        output_dir,
        name=path.stem,
        media_type=media_type,
        role=role,
        evidence_namespace="synthetic_fixture_nonclaim",
        claim_eligible=False,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a JUMP mechanistic task")
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args(argv)
    try:
        result = execute_task_file(
            args.parameters,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
