"""Leakage-resistant held-out linear probes and OOD law-family evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .vectors import Vector, dot


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    group_id: str
    features: Vector
    label: int
    law_family: str


@dataclass
class LinearProbe:
    weights: Vector
    bias: float
    center: Vector
    spread: Vector

    def probability(self, features: Vector) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature dimension does not match probe")
        normalized = [(x - c) / s for x, c, s in zip(features, self.center, self.spread)]
        value = max(-40.0, min(40.0, dot(self.weights, normalized) + self.bias))
        return 1.0 / (1.0 + math.exp(-value))


def heldout_group_split(
    samples: list[ProbeSample], *, test_fraction: float = 0.25, seed: int = 0
) -> tuple[list[ProbeSample], list[ProbeSample]]:
    """Split by episode/world group, never by activation row."""
    _validate_samples(samples)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    groups = sorted({sample.group_id for sample in samples})
    if len(groups) < 2:
        raise ValueError("at least two groups are required")
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest())
    count = max(1, min(len(groups) - 1, round(len(groups) * test_fraction)))
    test_groups = set(ranked[:count])
    return (
        [sample for sample in samples if sample.group_id not in test_groups],
        [sample for sample in samples if sample.group_id in test_groups],
    )


def train_probe(
    samples: list[ProbeSample], *, steps: int = 600, learning_rate: float = 0.15, l2: float = 1e-3
) -> LinearProbe:
    _validate_samples(samples)
    if len({sample.label for sample in samples}) < 2:
        raise ValueError("probe training requires both classes")
    width = len(samples[0].features)
    if width == 0 or any(len(sample.features) != width for sample in samples):
        raise ValueError("feature dimensions must match")
    center = [sum(sample.features[j] for sample in samples) / len(samples) for j in range(width)]
    spread = []
    for j in range(width):
        variance = sum((sample.features[j] - center[j]) ** 2 for sample in samples) / len(samples)
        spread.append(max(math.sqrt(variance), 1e-8))
    rows = [[(x - c) / s for x, c, s in zip(sample.features, center, spread)] for sample in samples]
    weights = [0.0] * width
    bias = 0.0
    for _ in range(steps):
        grad = [0.0] * width
        bias_grad = 0.0
        for row, sample in zip(rows, samples):
            score = max(-40.0, min(40.0, dot(weights, row) + bias))
            error = 1.0 / (1.0 + math.exp(-score)) - sample.label
            for j, value in enumerate(row):
                grad[j] += error * value
            bias_grad += error
        for j in range(width):
            weights[j] -= learning_rate * (grad[j] / len(rows) + l2 * weights[j])
        bias -= learning_rate * bias_grad / len(rows)
    return LinearProbe(weights, bias, center, spread)


def roc_auc(probabilities: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    wins = 0.0
    for i, label_i in enumerate(labels):
        if label_i != 1:
            continue
        for j, label_j in enumerate(labels):
            if label_j == 0:
                wins += float(probabilities[i] > probabilities[j]) + 0.5 * float(probabilities[i] == probabilities[j])
    return wins / (positives * negatives)


def evaluate_probe(probe: LinearProbe, samples: list[ProbeSample]) -> dict[str, float]:
    _validate_samples(samples)
    probabilities = [probe.probability(sample.features) for sample in samples]
    labels = [sample.label for sample in samples]
    predictions = [int(value >= 0.5) for value in probabilities]
    if set(labels) != {0, 1}:
        raise ValueError("probe evaluation requires both classes")
    recalls = [
        sum(predictions[index] == label for index, target in enumerate(labels) if target == label)
        / sum(target == label for target in labels)
        for label in (0, 1)
    ]
    return {
        "balanced_accuracy": sum(recalls) / 2.0,
        "auc": roc_auc(probabilities, labels),
        "n": float(len(labels)),
    }


def heldout_probe_evaluation(samples: list[ProbeSample], *, seed: int = 0) -> dict[str, Any]:
    train, test = heldout_group_split(samples, seed=seed)
    probe = train_probe(train)
    return {
        "metrics": evaluate_probe(probe, test),
        "train_groups": sorted({sample.group_id for sample in train}),
        "test_groups": sorted({sample.group_id for sample in test}),
        "probe": probe,
    }


def ood_law_family_evaluation(
    samples: list[ProbeSample], *, heldout_family: str
) -> dict[str, Any]:
    _validate_samples(samples)
    train = [sample for sample in samples if sample.law_family != heldout_family]
    test = [sample for sample in samples if sample.law_family == heldout_family]
    if not train or not test:
        raise ValueError("OOD evaluation requires in-domain training and held-out-family samples")
    train_groups = {sample.group_id for sample in train}
    test_groups = {sample.group_id for sample in test}
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(f"OOD group leakage across law families: {sorted(overlap)}")
    probe = train_probe(train)
    return {
        "heldout_family": heldout_family,
        "train_families": sorted({sample.law_family for sample in train}),
        "train_groups": sorted(train_groups),
        "test_groups": sorted(test_groups),
        "metrics": evaluate_probe(probe, test),
        "probe": probe,
    }


def _validate_samples(samples: list[ProbeSample]) -> None:
    if not samples:
        raise ValueError("probe samples must be nonempty")
    sample_ids: set[str] = set()
    group_families: dict[str, str] = {}
    width = len(samples[0].features)
    if width == 0:
        raise ValueError("probe features must be nonempty")
    for sample in samples:
        if not sample.sample_id or sample.sample_id in sample_ids:
            raise ValueError(f"probe sample_id must be nonempty and unique: {sample.sample_id!r}")
        sample_ids.add(sample.sample_id)
        if not sample.group_id or not sample.law_family:
            raise ValueError("probe group_id and law_family must be nonempty")
        prior = group_families.setdefault(sample.group_id, sample.law_family)
        if prior != sample.law_family:
            raise ValueError(
                f"probe group {sample.group_id!r} belongs to multiple law families: "
                f"{prior!r}, {sample.law_family!r}"
            )
        if sample.label not in (0, 1) or isinstance(sample.label, bool):
            raise ValueError("probe labels must be binary integers")
        if len(sample.features) != width or any(not math.isfinite(float(value)) for value in sample.features):
            raise ValueError("probe feature dimensions must match and be finite")
