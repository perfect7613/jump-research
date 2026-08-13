"""Private runtime used only by the restricted Modal function."""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import math
from copy import deepcopy
from typing import Any

from jump_contracts.experiments import canonical_json, validate_experiment_plan

from .safety import ALLOWED_BUILTINS, ALLOWED_IMPORTS, POLICY_SHA256, SafetyError, validate_simulation_source


class _BoundedStdout(io.StringIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, value: str) -> int:
        if self.tell() + len(value.encode("utf-8")) > self.limit:
            raise SafetyError("simulation stdout exceeds the byte limit")
        return super().write(value)


def _safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if level or name not in ALLOWED_IMPORTS:
        raise SafetyError("runtime import is not in the fixed allowlist")
    return builtins.__import__(name, globals, locals, fromlist, 0)


def _execute_validated_source(source: str, plan_value: dict[str, Any]) -> dict[str, Any]:
    """Execute source after remote revalidation; never expose this as a host API."""
    plan = validate_experiment_plan(plan_value)
    encoded = source.encode("utf-8")
    sandbox = plan["sandbox"]
    if hashlib.sha256(encoded).hexdigest() != sandbox["source"]["sha256"] or len(encoded) != sandbox["source"]["byte_length"]:
        raise SafetyError("source bytes do not match the confirmed plan")
    if sandbox["policy_sha256"] != POLICY_SHA256:
        raise SafetyError("plan policy does not match the deployed runtime policy")
    tree = validate_simulation_source(source)

    safe_builtins = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
    safe_builtins["__import__"] = _safe_import
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    stdout = _BoundedStdout(sandbox["limits"]["stdout_bytes"])
    with contextlib.redirect_stdout(stdout):
        exec(compile(tree, "<sealed-simulation>", "exec"), namespace, namespace)
        raw = namespace["simulate"](deepcopy(plan))
    return _validate_result(raw, plan, stdout.getvalue())


def _validate_result(raw: Any, plan: dict[str, Any], stdout: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"measurements"} or not isinstance(raw["measurements"], list):
        raise SafetyError("simulate must return exactly {'measurements': [...]} ")
    rows = raw["measurements"]
    limits = plan["sandbox"]["limits"]
    if len(rows) > limits["max_rows"]:
        raise SafetyError("simulation returned too many rows")
    measurement_ids = {item["id"] for item in plan["measurements"]}
    condition_ids = {item["id"] for item in plan["conditions"]}
    expected = {
        (condition_id, repetition)
        for condition_id in condition_ids
        for repetition in range(plan["sampling"]["repetitions"])
    }
    seen: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"condition_id", "repetition", "pairing_key", "values"}:
            raise SafetyError("each measurement row has invalid fields")
        condition_id, repetition = row["condition_id"], row["repetition"]
        if condition_id not in condition_ids or not isinstance(repetition, int) or isinstance(repetition, bool):
            raise SafetyError("measurement row references an unknown condition or repetition")
        if (condition_id, repetition) in seen:
            raise SafetyError("measurement condition/repetition pairs must be unique")
        seen.add((condition_id, repetition))
        values = row["values"]
        if not isinstance(values, dict) or set(values) != measurement_ids or len(values) > limits["max_columns"]:
            raise SafetyError("measurement values must cover the frozen measurements")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in values.values()):
            raise SafetyError("measurement values must be finite numbers")
        pairing_key = row["pairing_key"]
        if not isinstance(pairing_key, str) or not pairing_key or len(pairing_key) > 100:
            raise SafetyError("pairing_key must be bounded nonempty text")
        if plan["sampling"]["design"] == "paired_common_random_numbers" and pairing_key != f"rep-{repetition}":
            raise SafetyError("paired design requires the same repetition pairing key across conditions")
        if plan["sampling"]["design"] == "independent_repetitions" and pairing_key != f"{condition_id}:rep-{repetition}":
            raise SafetyError("independent design requires a condition-specific repetition key")
        normalized.append({
            "plan_sha256": plan["plan_sha256"], "condition_id": condition_id, "repetition": repetition,
            "pairing_key": pairing_key, "values": {key: float(value) for key, value in values.items()},
        })
    if seen != expected:
        raise SafetyError("simulation must return one row per condition and repetition")
    result = {"measurements": normalized, "stdout": stdout}
    if len(canonical_json(result)) > limits["result_bytes"]:
        raise SafetyError("simulation result exceeds the JSON byte limit")
    return result


__all__: list[str] = []
