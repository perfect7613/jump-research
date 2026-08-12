"""Exact, judge-free behavioral scoring for the JUMP benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_FORCES = frozenset({"attract", "repel"})
OBJECT_COUNT = 6
PARTITION_CHANCE = 1.0 / (2 ** (OBJECT_COUNT - 1) - 1)
SIGN_CHANCE = 1.0 / 4.0


@dataclass(frozen=True)
class Law:
    same: str
    different: str
    exponent: int

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "Law":
        if not isinstance(value, dict) or set(value) != {"same", "different", "exponent"}:
            raise ValueError("law must contain exactly same, different, exponent")
        if value["same"] not in VALID_FORCES or value["different"] not in VALID_FORCES:
            raise ValueError("same/different must be attract or repel")
        exponent = value["exponent"]
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise ValueError("exponent must be an integer")
        return cls(value["same"], value["different"], exponent)


def partition_correct(predicted: list[int], expected: list[int]) -> bool:
    """Score the locked six-object partition, invariant to swapping type names."""
    if not isinstance(predicted, list) or not isinstance(expected, list):
        return False
    if len(predicted) != OBJECT_COUNT or len(expected) != OBJECT_COUNT:
        return False
    if any(isinstance(value, bool) or value not in (0, 1) for value in predicted + expected):
        return False
    if len(set(predicted)) != 2 or len(set(expected)) != 2:
        return False
    return predicted == expected or [1 - value for value in predicted] == expected


def law_correct(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    try:
        return Law.parse(predicted) == Law.parse(expected)
    except (TypeError, ValueError):
        return False


def sign_correct(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    try:
        prediction, target = Law.parse(predicted), Law.parse(expected)
    except (TypeError, ValueError):
        return False
    return (prediction.same, prediction.different) == (target.same, target.different)


def force_mse(predicted: list[list[float]], expected: list[list[float]]) -> float:
    if not predicted or len(predicted) != len(expected):
        raise ValueError("force arrays must be nonempty and have matching rows")
    errors: list[float] = []
    for pred, gold in zip(predicted, expected):
        if len(pred) != len(gold) or not pred:
            raise ValueError("force rows must have matching nonzero dimensions")
        errors.extend((float(x) - float(y)) ** 2 for x, y in zip(pred, gold))
    return sum(errors) / len(errors)


def brier_score(probabilities: list[float], labels: list[int]) -> float:
    _validate_probabilities(probabilities, labels)
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)


def expected_calibration_error(
    probabilities: list[float], labels: list[int], bins: int = 10
) -> float:
    _validate_probabilities(probabilities, labels)
    if bins <= 0:
        raise ValueError("bins must be positive")
    total = len(labels)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            i
            for i, probability in enumerate(probabilities)
            if lower <= probability <= upper and (probability < upper or index == bins - 1)
        ]
        if selected:
            confidence = sum(probabilities[i] for i in selected) / len(selected)
            accuracy = sum(labels[i] for i in selected) / len(selected)
            result += len(selected) / total * abs(confidence - accuracy)
    return result


def score_episode(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    """Score typed fields only; malformed predictions score zero, targets fail closed."""
    _validate_target(target)
    partition = partition_correct(prediction.get("partition", []), target["partition"])
    full_law = law_correct(prediction.get("replacement_law", {}), target["replacement_law"])
    signs = sign_correct(prediction.get("replacement_law", {}), target["replacement_law"])
    result = {
        "partition_accuracy": float(partition),
        "sign_accuracy": float(signs),
        "full_law_accuracy": float(full_law),
        "joint_theory_accuracy": float(partition and full_law),
        "adequacy_correct": float(
            isinstance(prediction.get("adequate"), bool)
            and prediction["adequate"] is target["adequate"]
        ),
    }
    if "forces" in prediction:
        result["force_mse"] = force_mse(prediction["forces"], target["forces"])
    if target.get("control_no_hidden_types") is True:
        result["false_abduction"] = float(prediction.get("posits_hidden_types") is True)
    return result


def score_dataset(
    rows: list[dict[str, Any]], *, allowed_exponents: list[int]
) -> dict[str, float]:
    """Return metrics whose names and chance levels match the locked PRD exactly."""
    if not rows:
        raise ValueError("cannot score an empty dataset")
    if (
        not isinstance(allowed_exponents, list)
        or not allowed_exponents
        or any(isinstance(value, bool) or not isinstance(value, int) for value in allowed_exponents)
        or len(set(allowed_exponents)) != len(allowed_exponents)
    ):
        raise ValueError("allowed_exponents must be a nonempty unique integer array")
    allowed = set(allowed_exponents)
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("prediction"), dict)
            or not isinstance(row.get("target"), dict)
        ):
            raise ValueError("each behavior row must contain prediction and target objects")
        target_law = Law.parse(row["target"].get("replacement_law", {}))
        if target_law.exponent not in allowed:
            raise ValueError("target exponent is outside the preregistered allowlist")

    per_episode = [score_episode(row["prediction"], row["target"]) for row in rows]
    summary: dict[str, float] = {}
    for key in sorted({key for result in per_episode for key in result} - {"adequacy_correct"}):
        values = [result[key] for result in per_episode if key in result]
        summary[key] = sum(values) / len(values)

    adequacy_targets = [int(row["target"]["adequate"]) for row in rows]
    if set(adequacy_targets) != {0, 1}:
        raise ValueError("adequacy balanced accuracy requires both target classes")
    recalls = []
    for label in (0, 1):
        indices = [index for index, target in enumerate(adequacy_targets) if target == label]
        recalls.append(sum(per_episode[index]["adequacy_correct"] for index in indices) / len(indices))
    summary["adequacy_balanced_accuracy"] = sum(recalls) / 2.0

    exponent_count = len(allowed_exponents)
    summary.update(
        {
            "partition_chance": PARTITION_CHANCE,
            "sign_chance": SIGN_CHANCE,
            "full_law_chance": 1.0 / (4.0 * exponent_count),
            "joint_theory_chance": 1.0 / (31.0 * 4.0 * exponent_count),
        }
    )
    if all("confidence" in row["prediction"] for row in rows):
        probabilities = [float(row["prediction"]["confidence"]) for row in rows]
        labels = [int(result["full_law_accuracy"]) for result in per_episode]
        summary["confidence_brier"] = brier_score(probabilities, labels)
        summary["confidence_ece"] = expected_calibration_error(probabilities, labels)
    return summary


def _validate_target(target: dict[str, Any]) -> None:
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    partition = target.get("partition")
    if not partition_correct(partition, partition):
        raise ValueError("target partition must be a nontrivial six-object binary partition")
    Law.parse(target.get("replacement_law", {}))
    if not isinstance(target.get("adequate"), bool):
        raise ValueError("target adequate must be Boolean")


def _validate_probabilities(probabilities: list[float], labels: list[int]) -> None:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be nonempty and equal length")
    if any(not 0.0 <= p <= 1.0 for p in probabilities):
        raise ValueError("probabilities must be in [0, 1]")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary")
