"""Exact structured scoring for Track H; no free-text or LLM judge path exists."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from jump_mechanistic.scoring import expected_calibration_error

OBJECT_COUNT = 6
ANSWER_FIELDS = frozenset(
    {"partition", "replacement_law", "adequacy", "force_prediction", "confidence"}
)


@dataclass(frozen=True)
class ParsedAnswer:
    partition: tuple[int, ...]
    replacement_law: tuple[str, str, int]
    adequacy: bool
    force_prediction: dict[int, tuple[tuple[float, float], ...]]
    confidence: float


def _partition(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != OBJECT_COUNT:
        raise ValueError("partition must be a six-element array")
    if any(isinstance(item, bool) or item not in (0, 1) for item in value) or len(set(value)) != 2:
        raise ValueError("partition must be a non-trivial binary assignment")
    result = tuple(value)
    return tuple(1 - item for item in result) if result[0] else result


def _law(value: Any, allowed_exponents: set[int]) -> tuple[str, str, int]:
    if not isinstance(value, dict) or set(value) != {"same", "different", "exponent"}:
        raise ValueError("replacement_law must contain exactly same, different, exponent")
    if value["same"] not in {"attract", "repel"} or value["different"] not in {"attract", "repel"}:
        raise ValueError("law signs must be attract or repel")
    exponent = value["exponent"]
    if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent not in allowed_exponents:
        raise ValueError("law exponent is outside the declared allowlist")
    return value["same"], value["different"], exponent


def _forces(value: Any, horizons: set[int]) -> dict[int, tuple[tuple[float, float], ...]]:
    if not isinstance(value, dict) or set(value) != {str(horizon) for horizon in horizons}:
        raise ValueError("force_prediction horizon keys must exactly match the target")
    result: dict[int, tuple[tuple[float, float], ...]] = {}
    for horizon in sorted(horizons):
        vectors = value[str(horizon)]
        if not isinstance(vectors, list) or len(vectors) != OBJECT_COUNT:
            raise ValueError("every force horizon must contain six vectors")
        parsed = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != 2:
                raise ValueError("force vectors must be two-element arrays")
            if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in vector):
                raise ValueError("force components must be finite numbers")
            parsed.append((float(vector[0]), float(vector[1])))
        result[horizon] = tuple(parsed)
    return result


def parse_answer(value: Any, *, allowed_exponents: list[int], target_horizons: set[int]) -> ParsedAnswer:
    if (
        not isinstance(allowed_exponents, list)
        or not allowed_exponents
        or len(set(allowed_exponents)) != len(allowed_exponents)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in allowed_exponents)
    ):
        raise ValueError("allowed_exponents must be unique positive integers")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("answer is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != ANSWER_FIELDS:
        raise ValueError(f"answer must contain exactly {sorted(ANSWER_FIELDS)}")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        raise ValueError("confidence must be a finite number")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not isinstance(value["adequacy"], bool):
        raise ValueError("adequacy must be Boolean")
    exponents = set(allowed_exponents)
    return ParsedAnswer(
        partition=_partition(value["partition"]),
        replacement_law=_law(value["replacement_law"], exponents),
        adequacy=value["adequacy"],
        force_prediction=_forces(value["force_prediction"], target_horizons),
        confidence=confidence,
    )


def _target(value: Any, allowed_exponents: list[int]) -> ParsedAnswer:
    if not isinstance(value, dict):
        raise ValueError("target must be an object")
    required = {"partition", "replacement_law", "adequacy", "force_prediction"}
    if set(value) != required:
        raise ValueError(f"target must contain exactly {sorted(required)}")
    horizons = _target_horizons(value)
    return parse_answer(
        {**value, "confidence": 1.0},
        allowed_exponents=allowed_exponents,
        target_horizons=horizons,
    )


def _target_horizons(target: dict[str, Any]) -> set[int]:
    force_prediction = target.get("force_prediction")
    if not isinstance(force_prediction, dict) or not force_prediction:
        raise ValueError("target force_prediction must be a nonempty object")
    horizons: set[int] = set()
    for key in force_prediction:
        if not isinstance(key, str) or not key.isdigit() or int(key) <= 0 or str(int(key)) != key:
            raise ValueError("target force horizons must be canonical positive integer strings")
        horizons.add(int(key))
    return horizons


def _nrmse(
    predicted: tuple[tuple[float, float], ...],
    expected: tuple[tuple[float, float], ...],
    epsilon: float,
) -> float:
    errors = [(x - y) ** 2 for p, t in zip(predicted, expected) for x, y in zip(p, t)]
    targets = [value**2 for vector in expected for value in vector]
    rmse = math.sqrt(sum(errors) / len(errors))
    target_rms = math.sqrt(sum(targets) / len(targets))
    return rmse / max(target_rms, epsilon)


def score_answer(
    prediction: Any, target: dict[str, Any], *, allowed_exponents: list[int], epsilon: float = 1e-8
) -> dict[str, Any]:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    gold = _target(target, allowed_exponents)
    horizons = _target_horizons(target)
    try:
        answer = parse_answer(
            prediction, allowed_exponents=allowed_exponents, target_horizons=horizons
        )
    except (TypeError, ValueError) as exc:
        return {
            "parse_success": 0.0,
            "parse_error": str(exc),
            "partition_accuracy": 0.0,
            "sign_accuracy": 0.0,
            "full_law_accuracy": 0.0,
            "joint_theory_accuracy": 0.0,
            "adequacy_accuracy": 0.0,
            "confidence": None,
            "force_nrmse": {},
        }
    partition_ok = answer.partition == gold.partition
    sign_ok = answer.replacement_law[:2] == gold.replacement_law[:2]
    law_ok = answer.replacement_law == gold.replacement_law
    horizon_errors = {
        str(horizon): _nrmse(
            answer.force_prediction[horizon], gold.force_prediction[horizon], epsilon
        )
        for horizon in sorted(horizons)
    }
    ordered = sorted((int(horizon), value) for horizon, value in horizon_errors.items())
    if len(ordered) == 1:
        horizon_auc = ordered[0][1]
    else:
        area = sum(
            (right_h - left_h) * (left_v + right_v) / 2
            for (left_h, left_v), (right_h, right_v) in zip(ordered, ordered[1:])
        )
        horizon_auc = area / (ordered[-1][0] - ordered[0][0])
    return {
        "parse_success": 1.0,
        "parse_error": None,
        "partition_accuracy": float(partition_ok),
        "sign_accuracy": float(sign_ok),
        "full_law_accuracy": float(law_ok),
        "joint_theory_accuracy": float(partition_ok and law_ok),
        "adequacy_accuracy": float(answer.adequacy is gold.adequacy),
        "confidence": answer.confidence,
        "force_nrmse": horizon_errors,
        "force_nrmse_horizon_auc": horizon_auc,
    }


def score_rows(
    rows: list[dict[str, Any]], *, allowed_exponents: list[int], epsilon: float = 1e-8
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("cannot score an empty baseline result")
    if (
        not isinstance(allowed_exponents, list)
        or not allowed_exponents
        or len(set(allowed_exponents)) != len(allowed_exponents)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in allowed_exponents)
    ):
        raise ValueError("allowed_exponents must be unique positive integers")
    details: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("target"), dict):
            raise ValueError("every row must contain a target")
        score = score_answer(
            row.get("prediction"), row["target"], allowed_exponents=allowed_exponents, epsilon=epsilon
        )
        details.append({**row, "score": score})
    metrics: dict[str, float] = {}
    for name in (
        "parse_success",
        "partition_accuracy",
        "sign_accuracy",
        "full_law_accuracy",
        "joint_theory_accuracy",
    ):
        metrics[name] = sum(item["score"][name] for item in details) / len(details)
    adequacy_labels = [bool(item["target"]["adequacy"]) for item in details]
    if set(adequacy_labels) != {False, True}:
        raise ValueError("adequacy balanced accuracy requires both target classes")
    recalls = []
    for label in (False, True):
        selected = [item for item in details if item["target"]["adequacy"] is label]
        recalls.append(sum(item["score"]["adequacy_accuracy"] for item in selected) / len(selected))
    metrics["adequacy_balanced_accuracy"] = sum(recalls) / 2
    metrics["partition_chance"] = 1 / 31
    metrics["sign_chance"] = 1 / 4
    metrics["full_law_chance"] = 1 / (4 * len(allowed_exponents))
    metrics["joint_theory_chance"] = 1 / (31 * 4 * len(allowed_exponents))
    parsed = [item for item in details if item["score"]["parse_success"] == 1.0]
    metrics["force_prediction_coverage"] = len(parsed) / len(details)
    if parsed:
        horizons = sorted(parsed[0]["score"]["force_nrmse"], key=int)
        for horizon in horizons:
            metrics[f"force_nrmse.h{horizon}"] = sum(
                item["score"]["force_nrmse"][horizon] for item in parsed
            ) / len(parsed)
        metrics["force_nrmse.horizon_auc"] = sum(
            item["score"]["force_nrmse_horizon_auc"] for item in parsed
        ) / len(parsed)
        probabilities = [item["score"]["confidence"] for item in parsed]
        labels = [int(item["score"]["joint_theory_accuracy"]) for item in parsed]
        metrics["confidence_brier"] = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
        metrics["confidence_ece"] = expected_calibration_error(probabilities, labels, bins=10)
    return metrics, details
