"""Locked clustered causal estimates, mediation, and fail-closed gates."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
_SHA256 = re.compile(r"[0-9a-f]{64}")


def paired_effect(
    treated: list[float],
    control: list[float],
    *,
    cluster_ids: list[str],
    seed: int = 0,
    bootstrap_samples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, float | None]:
    """Paired ATE with the PRD-locked clustered percentile bootstrap."""
    _validate_bootstrap(bootstrap_samples, confidence, seed)
    if len(treated) != len(control) or len(treated) < 2:
        raise ValueError("paired effects require equal-length arrays with at least two pairs")
    _validate_clusters(cluster_ids, len(treated))
    differences = [float(a) - float(b) for a, b in zip(treated, control)]
    _require_finite(differences, "paired outcomes")
    estimate = _mean(differences)
    bootstrap = _cluster_bootstrap(
        cluster_ids,
        bootstrap_samples,
        seed,
        lambda indices: _mean([differences[index] for index in indices]),
    )
    variance = sum((value - estimate) ** 2 for value in differences) / (len(differences) - 1)
    control_mean = _mean(control)
    control_variance = sum((float(value) - control_mean) ** 2 for value in control) / (len(control) - 1)
    low, high = _percentile_interval(bootstrap, confidence)
    return {
        "ate": estimate,
        "standard_error": _sample_sd(bootstrap),
        "ci_low": low,
        "ci_high": high,
        "paired_standardized_effect": estimate / math.sqrt(variance) if variance > 0 else None,
        "n_pairs": float(len(differences)),
        "cluster_count": float(len(set(cluster_ids))),
        "control_sd": math.sqrt(control_variance),
        "bootstrap_resamples": float(bootstrap_samples),
        "bootstrap_seed": float(seed),
    }


def contrast_effects(
    outcomes: dict[str, list[float]],
    *,
    cluster_ids: list[str],
    target: str = "target",
    baseline: str = "baseline",
    seed: int = 0,
    bootstrap_samples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, dict[str, float | None]]:
    if target not in outcomes or baseline not in outcomes:
        raise ValueError("target and baseline conditions are required")
    return {
        condition: paired_effect(
            values,
            outcomes[baseline],
            cluster_ids=cluster_ids,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        for condition, values in sorted(outcomes.items())
        if condition != baseline
    }


def mediation_analysis(
    treatment: list[float],
    mediator: list[float],
    outcome: list[float],
    *,
    cluster_ids: list[str],
    seed: int = 0,
    bootstrap_samples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, float | None]:
    """OLS clamp/patch primitive with NIE=TE-NDE and clustered percentile CIs."""
    _validate_bootstrap(bootstrap_samples, confidence, seed)
    if not (len(treatment) == len(mediator) == len(outcome)) or len(outcome) < 4:
        raise ValueError("mediation arrays must have equal length >= 4")
    _validate_clusters(cluster_ids, len(outcome))
    _require_finite([*treatment, *mediator, *outcome], "mediation arrays")

    path_a, total, direct, path_b = _mediation_estimate(treatment, mediator, outcome)

    def statistic(indices: list[int]) -> tuple[float, float, float]:
        tx = [treatment[index] for index in indices]
        md = [mediator[index] for index in indices]
        oy = [outcome[index] for index in indices]
        _, boot_total, boot_direct, _ = _mediation_estimate(tx, md, oy)
        return boot_total, boot_direct, boot_total - boot_direct

    bootstrap = _cluster_bootstrap(cluster_ids, bootstrap_samples, seed, statistic)
    total_bootstrap = [value[0] for value in bootstrap]
    direct_bootstrap = [value[1] for value in bootstrap]
    indirect_bootstrap = [value[2] for value in bootstrap]
    total_low, total_high = _percentile_interval(total_bootstrap, confidence)
    direct_low, direct_high = _percentile_interval(direct_bootstrap, confidence)
    indirect_low, indirect_high = _percentile_interval(indirect_bootstrap, confidence)
    indirect = total - direct
    # The PRD forbids publishing NIE/TE unless TE is positive and its CI excludes zero.
    proportion = indirect / total if total > 0 and total_low > 0 else None
    return {
        "path_a": path_a,
        "path_b": path_b,
        "total_effect": total,
        "total_ci_low": total_low,
        "total_ci_high": total_high,
        "direct_effect": direct,
        "direct_ci_low": direct_low,
        "direct_ci_high": direct_high,
        "indirect_effect": indirect,
        "indirect_ci_low": indirect_low,
        "indirect_ci_high": indirect_high,
        "mediation_proportion": proportion,
        "bootstrap_resamples": float(bootstrap_samples),
        "bootstrap_seed": float(seed),
        "cluster_count": float(len(set(cluster_ids))),
    }


def mediation_specificity(
    primary: dict[str, float], controls: dict[str, dict[str, float]], *, margin: float = 0.0
) -> dict[str, Any]:
    required = {"surprise", "uncertainty", "negation"}
    missing = required - set(controls)
    if missing:
        raise ValueError(f"missing specificity mediators: {sorted(missing)}")
    primary_effect = abs(primary["indirect_effect"])
    control_effects = {name: abs(value["indirect_effect"]) for name, value in controls.items()}
    return {
        "passed": all(primary_effect > effect + margin for effect in control_effects.values()),
        "primary_abs_indirect": primary_effect,
        "control_abs_indirect": control_effects,
    }


@dataclass(frozen=True)
class CheckpointIdentity:
    checkpoint_id: str
    model_revision: str
    tokenizer_revision: str
    training_lineage_id: str
    checkpoint_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CheckpointIdentity":
        required = {field for field in cls.__dataclass_fields__}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"checkpoint identity must contain exactly {sorted(required)}")
        identity = cls(**value)
        if any(not isinstance(getattr(identity, field), str) or not getattr(identity, field) for field in required):
            raise ValueError("checkpoint identity fields must be nonempty strings")
        if not _SHA256.fullmatch(identity.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        return identity


@dataclass(frozen=True)
class ConfirmatoryEvidence:
    identity: CheckpointIdentity
    g3_passed: bool
    g5_passed: bool
    total_ci_low: float
    ordered_nie_ci_lows: tuple[float, float]
    mediated_proportion: float | None
    specificity_passed: bool
    ood_effect_ci_low: float
    ood_retention: float
    provenance_hash_match_rate: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConfirmatoryEvidence":
        required = {
            "identity", "g3_passed", "g5_passed", "total_ci_low",
            "ordered_nie_ci_lows", "mediated_proportion", "specificity_passed",
            "ood_effect_ci_low", "ood_retention", "provenance_hash_match_rate",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"confirmatory evidence must contain exactly {sorted(required)}")
        ordered = value["ordered_nie_ci_lows"]
        if not isinstance(ordered, list) or len(ordered) != 2:
            raise ValueError("ordered_nie_ci_lows must contain exactly inadequacy and promotion")
        for flag in ("g3_passed", "g5_passed", "specificity_passed"):
            if not isinstance(value[flag], bool):
                raise ValueError(f"{flag} must be Boolean")
        numeric = [value["total_ci_low"], *ordered, value["ood_effect_ci_low"], value["ood_retention"], value["provenance_hash_match_rate"]]
        if value["mediated_proportion"] is not None:
            numeric.append(value["mediated_proportion"])
        _require_finite(numeric, "confirmatory evidence")
        return cls(
            identity=CheckpointIdentity.from_dict(value["identity"]),
            g3_passed=value["g3_passed"],
            g5_passed=value["g5_passed"],
            total_ci_low=float(value["total_ci_low"]),
            ordered_nie_ci_lows=(float(ordered[0]), float(ordered[1])),
            mediated_proportion=(
                None if value["mediated_proportion"] is None else float(value["mediated_proportion"])
            ),
            specificity_passed=value["specificity_passed"],
            ood_effect_ci_low=float(value["ood_effect_ci_low"]),
            ood_retention=float(value["ood_retention"]),
            provenance_hash_match_rate=float(value["provenance_hash_match_rate"]),
        )


def evaluate_confirmatory_gates(
    primary: ConfirmatoryEvidence | None,
    replication: ConfirmatoryEvidence | None,
) -> dict[str, dict[str, Any]]:
    """Evaluate locked G6-G8; missing, incomplete, or aliased evidence fails closed."""
    if primary is None:
        reason = "primary checkpoint evidence is missing"
        return {gate: {"passed": False, "reasons": [reason]} for gate in ("g6", "g7", "g8")}
    g6_reasons = _g6_reasons(primary)
    g7_reasons = _g7_reasons(primary)
    g8_reasons: list[str] = []
    if replication is None:
        g8_reasons.append("second checkpoint evidence is missing")
    else:
        if not _meaningfully_distinct(primary.identity, replication.identity):
            g8_reasons.append("second checkpoint is not an immutable, independent revision")
        if not replication.g3_passed:
            g8_reasons.append("second checkpoint did not pass G3")
        if not replication.g5_passed:
            g8_reasons.append("second checkpoint did not pass G5")
        g8_reasons.extend(f"second checkpoint: {reason}" for reason in _g6_reasons(replication))
        g8_reasons.extend(f"second checkpoint: {reason}" for reason in _g7_reasons(replication))
    if g6_reasons:
        g8_reasons.append("primary checkpoint did not pass G6")
    if g7_reasons:
        g8_reasons.append("primary checkpoint did not pass G7")
    return {
        "g6": {"passed": not g6_reasons, "reasons": g6_reasons},
        "g7": {"passed": not g7_reasons, "reasons": g7_reasons},
        "g8": {"passed": not g8_reasons, "reasons": g8_reasons},
    }


def _g6_reasons(evidence: ConfirmatoryEvidence) -> list[str]:
    reasons = []
    if evidence.total_ci_low <= 0:
        reasons.append("TE lower CI is not positive")
    if any(value <= 0 for value in evidence.ordered_nie_ci_lows):
        reasons.append("both ordered NIE lower CIs must be positive")
    if evidence.mediated_proportion is None or evidence.mediated_proportion < 0.20:
        reasons.append("mediated proportion is missing or below 20%")
    if not evidence.specificity_passed:
        reasons.append("specificity controls are not null or smaller")
    return reasons


def _g7_reasons(evidence: ConfirmatoryEvidence) -> list[str]:
    reasons = []
    if evidence.ood_effect_ci_low <= 0:
        reasons.append("OOD causal-effect lower CI is not positive")
    if evidence.ood_retention < 0.50:
        reasons.append("OOD retention is below 50%")
    if evidence.provenance_hash_match_rate != 1.0:
        reasons.append("OOD provenance is not 100%")
    return reasons


def _meaningfully_distinct(first: CheckpointIdentity, second: CheckpointIdentity) -> bool:
    return (
        first.checkpoint_id != second.checkpoint_id
        and first.model_revision != second.model_revision
        and first.training_lineage_id != second.training_lineage_id
        and first.checkpoint_sha256 != second.checkpoint_sha256
    )


def _mediation_estimate(
    treatment: list[float], mediator: list[float], outcome: list[float]
) -> tuple[float, float, float, float]:
    path_a = _simple_slope(treatment, mediator)
    total = _simple_slope(treatment, outcome)
    direct, path_b = _two_predictor_slopes(treatment, mediator, outcome)
    return path_a, total, direct, path_b


def _cluster_bootstrap(
    cluster_ids: list[str],
    samples: int,
    seed: int,
    statistic: Callable[[list[int]], Any],
) -> list[Any]:
    groups: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        groups.setdefault(cluster_id, []).append(index)
    ordered = sorted(groups)
    rng = random.Random(seed)
    results = []
    for _ in range(samples):
        indices: list[int] = []
        for _ in ordered:
            indices.extend(groups[ordered[rng.randrange(len(ordered))]])
        try:
            results.append(statistic(indices))
        except ValueError as error:
            raise ValueError("singular clustered bootstrap resample") from error
    return results


def _validate_clusters(cluster_ids: list[str], expected_length: int) -> None:
    if len(cluster_ids) != expected_length:
        raise ValueError("cluster_ids must align one-to-one with rows")
    if any(not isinstance(value, str) or not value for value in cluster_ids):
        raise ValueError("cluster_ids must be nonempty strings")
    if len(set(cluster_ids)) < 2:
        raise ValueError("at least two episode/world-seed clusters are required")


def _validate_bootstrap(samples: int, confidence: float, seed: int) -> None:
    if not isinstance(samples, int) or isinstance(samples, bool) or samples != BOOTSTRAP_RESAMPLES:
        raise ValueError(f"confirmatory bootstrap must use exactly {BOOTSTRAP_RESAMPLES} resamples")
    if confidence != BOOTSTRAP_CONFIDENCE:
        raise ValueError("confirmatory bootstrap confidence must be exactly 0.95")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("bootstrap seed must be an integer")


def _require_finite(values: list[Any], name: str) -> None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be finite numeric values")


def _simple_slope(x: list[float], y: list[float]) -> float:
    mean_x, mean_y = _mean(x), _mean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator <= 1e-12:
        raise ValueError("singular predictor")
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator


def _two_predictor_slopes(x1: list[float], x2: list[float], y: list[float]) -> tuple[float, float]:
    m1, m2, my = _mean(x1), _mean(x2), _mean(y)
    a11 = sum((value - m1) ** 2 for value in x1)
    a22 = sum((value - m2) ** 2 for value in x2)
    a12 = sum((value1 - m1) * (value2 - m2) for value1, value2 in zip(x1, x2))
    b1 = sum((value1 - m1) * (value_y - my) for value1, value_y in zip(x1, y))
    b2 = sum((value2 - m2) * (value_y - my) for value2, value_y in zip(x2, y))
    determinant = a11 * a22 - a12 * a12
    if abs(determinant) <= 1e-12:
        raise ValueError("collinear mediation predictors")
    return ((b1 * a22 - b2 * a12) / determinant, (b2 * a11 - b1 * a12) / determinant)


def _percentile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    ordered = sorted(values)
    tail = (1.0 - confidence) / 2.0
    return _quantile(ordered, tail), _quantile(ordered, 1.0 - tail)


def _sample_sd(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _mean(values: list[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)
