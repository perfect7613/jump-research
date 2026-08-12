from __future__ import annotations

import copy
import json
import math
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

from .errors import BudgetError, ManifestError
from .io import canonical_json, load_data, sha256_bytes

SCHEMA_VERSION = "jump.experiments/v1"
SUPPORTED_GPUS = {"cpu", "T4", "L4", "A10", "L40S", "A100-80GB", "H100"}
H100_PROFILE_GPUS = {"T4", "L4", "L40S"}
SMOKE_MAX_RUNTIME_SECONDS = 60
SMOKE_MAX_COST_USD = 0.02
SMOKE_ALLOWED_RESOURCES = {"cpu", "T4", "L4"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LAYER_KEYS = {"layer", "layers", "extraction_layers"}
TIMEPOINT_KEYS = {"timepoint", "timepoints", "extraction_timepoints"}
SENSITIVE_PARAMETER_KEYS = {"api_key", "access_token", "password", "secret", "token", "credentials"}


def _required(mapping: dict[str, Any], key: str, where: str, kind: type | tuple[type, ...]) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind):
        names = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise ManifestError(f"{where}.{key} must be {names}")
    return value


def _valid_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ManifestError(f"{where} must match {SAFE_ID.pattern}")
    return value


def _collect_named_values(value: Any, keys: set[str]) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                if isinstance(item, list):
                    found.update(item)
                else:
                    found.add(item)
            else:
                found.update(_collect_named_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_named_values(item, keys))
    return found


def _find_sensitive_keys(value: Any, prefix: str = "parameters") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in SENSITIVE_PARAMETER_KEYS:
                found.append(f"{prefix}.{key}")
            found.extend(_find_sensitive_keys(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_sensitive_keys(item, f"{prefix}[{index}]"))
    return found


def selected_layers(run: dict[str, Any]) -> set[Any]:
    explicit = run.get("selection", {}).get("layers", [])
    return set(explicit) | _collect_named_values(run.get("task", {}).get("parameters", {}), LAYER_KEYS)


def selected_timepoints(run: dict[str, Any]) -> set[Any]:
    explicit = run.get("selection", {}).get("timepoints", [])
    return set(explicit) | _collect_named_values(run.get("task", {}).get("parameters", {}), TIMEPOINT_KEYS)


def manifest_hash(manifest: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(manifest))


def validate_json_schema(instance: dict[str, Any], schema_name: str) -> None:
    _reject_nonfinite(instance)
    try:
        import jsonschema
    except ImportError as exc:
        raise ManifestError("jsonschema is required for normative contract validation") from exc
    try:
        schema = json.loads(files("jump_runner").joinpath("schemas", schema_name).read_text())
        jsonschema.Draft202012Validator(schema).validate(instance)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "manifest"
        raise ManifestError(f"JSON Schema violation at {location}: {exc.message}") from exc


def _reject_nonfinite(value: Any, path: str = "manifest") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ManifestError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(key, f"{path}.<key>")
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _finite_number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ManifestError(f"{where} must be a finite number")
    if positive and value <= 0:
        raise ManifestError(f"{where} must be positive")
    return float(value)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_data(path)
    validate_manifest(manifest)
    return manifest


def iter_runs(manifest: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for phase in manifest["phases"]:
        for run in phase["runs"]:
            yield phase, run


def validate_manifest(manifest: dict[str, Any]) -> None:
    validate_json_schema(manifest, "experiment-manifest-v1.schema.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    _valid_id(manifest.get("experiment_id"), "experiment_id")
    prereg = _required(manifest, "preregistration", "manifest", dict)
    layers = _required(prereg, "layer_allowlist", "preregistration", list)
    timepoints = _required(prereg, "timepoint_allowlist", "preregistration", list)
    if not layers or not timepoints:
        raise ManifestError("preregistration allowlists cannot be empty")
    if len(set(map(str, layers))) != len(layers) or len(set(map(str, timepoints))) != len(timepoints):
        raise ManifestError("preregistration allowlists cannot contain duplicates")
    secret_refs = manifest.get("secrets", [])
    if not isinstance(secret_refs, list):
        raise ManifestError("manifest.secrets must be a list of Modal Secret references")
    for index, reference in enumerate(secret_refs):
        if not isinstance(reference, dict) or set(reference) - {"name", "required_keys"}:
            raise ManifestError(f"secrets[{index}] may contain only name and required_keys")
        _valid_id(reference.get("name"), f"secrets[{index}].name")
        required_keys = reference.get("required_keys", [])
        if not isinstance(required_keys, list) or not required_keys or not all(
            isinstance(key, str) and key for key in required_keys
        ):
            raise ManifestError(f"secrets[{index}].required_keys must be a nonempty string list")
    secret_names = {reference["name"] for reference in secret_refs}

    phases = _required(manifest, "phases", "manifest", list)
    if not phases:
        raise ManifestError("manifest.phases cannot be empty")
    phase_ids: set[str] = set()
    run_ids: set[str] = set()
    seen_phases: set[str] = set()
    defaults = manifest.get("defaults", {})
    if defaults.get("resources", {}).get("gpu") == "H100":
        raise ManifestError("H100 must be explicitly opted into per run; it cannot be the manifest default")
    phases_by_id: dict[str, dict[str, Any]] = {}

    for pidx, phase in enumerate(phases):
        where = f"phases[{pidx}]"
        _required(phase, "id", where, str)
        phase_id = _valid_id(phase["id"], f"{where}.id")
        if phase_id in phase_ids:
            raise ManifestError(f"duplicate phase id: {phase_id}")
        phase_ids.add(phase_id)
        dependencies = phase.get("depends_on", [])
        if not isinstance(dependencies, list) or any(dep not in seen_phases for dep in dependencies):
            raise ManifestError(f"{where}.depends_on must reference earlier phases")
        seen_phases.add(phase_id)
        phases_by_id[phase_id] = phase

        budget = _required(phase, "budget", where, dict)
        allowed_gpus = _required(budget, "allowed_gpu_types", f"{where}.budget", list)
        if not allowed_gpus or not set(allowed_gpus) <= SUPPORTED_GPUS:
            raise ManifestError(f"{where}.budget.allowed_gpu_types must use {sorted(SUPPORTED_GPUS)}")
        max_seconds = budget.get("max_runtime_seconds")
        max_cost = budget.get("max_cost_usd")
        rates = budget.get("gpu_hourly_cost_usd", {})
        expected_gpu_ceiling = 0 if set(allowed_gpus) == {"cpu"} else 1
        if budget.get("max_concurrent_gpus") != expected_gpu_ceiling:
            raise ManifestError(
                f"{where}.budget.max_concurrent_gpus must be {expected_gpu_ceiling} "
                f"for allowed resources {allowed_gpus}"
            )
        _finite_number(max_seconds, f"{where}.budget.max_runtime_seconds", positive=True)
        _finite_number(max_cost, f"{where}.budget.max_cost_usd")
        if max_cost < 0:
            raise ManifestError(f"{where}.budget.max_cost_usd must be nonnegative")
        for gpu in allowed_gpus:
            rate = _finite_number(rates.get(gpu), f"{where}.budget.gpu_hourly_cost_usd.{gpu}")
            if rate < 0:
                raise ManifestError(f"{where}.budget.gpu_hourly_cost_usd.{gpu} must be nonnegative")

        runs = _required(phase, "runs", where, list)
        planned_seconds = 0.0
        planned_cost = 0.0
        planned_h100_cost = 0.0
        for ridx, original in enumerate(runs):
            rwhere = f"{where}.runs[{ridx}]"
            run = resolve_run(defaults, original)
            run_id = _valid_id(run.get("id"), f"{rwhere}.id")
            if run_id in run_ids:
                raise ManifestError(f"duplicate run id: {run_id}")
            run_ids.add(run_id)
            task = _required(run, "task", rwhere, dict)
            if run.get("secret") is not None and run["secret"] not in secret_names:
                raise ManifestError(f"{rwhere}.secret must reference manifest.secrets")
            module = _required(task, "module", f"{rwhere}.task", str)
            if not module or module.startswith("-") or any(part in {"", ".", ".."} for part in module.split(".")):
                raise ManifestError(f"{rwhere}.task.module is not a safe Python module")
            if not isinstance(task.get("parameters", {}), dict):
                raise ManifestError(f"{rwhere}.task.parameters must be an object")
            sensitive = _find_sensitive_keys(task.get("parameters", {}))
            if sensitive:
                raise ManifestError(
                    f"{rwhere} contains secret-like parameter keys {sensitive}; use Modal Secret references and environment variables"
                )
            resources = _required(run, "resources", rwhere, dict)
            gpu = resources.get("gpu")
            timeout = resources.get("timeout_seconds")
            if gpu not in allowed_gpus:
                raise BudgetError(f"{rwhere} requests GPU {gpu!r} outside phase allowlist")
            if not isinstance(timeout, int) or timeout <= 0:
                raise ManifestError(f"{rwhere}.resources.timeout_seconds must be a positive integer")
            if timeout > 3600:
                raise ManifestError(f"{rwhere}.resources.timeout_seconds cannot exceed the 3600s worker ceiling")
            forbidden_layers = selected_layers(run) - set(layers)
            forbidden_timepoints = selected_timepoints(run) - set(timepoints)
            if forbidden_layers:
                raise ManifestError(f"{rwhere} requests non-preregistered layers: {sorted(forbidden_layers, key=str)}")
            if forbidden_timepoints:
                raise ManifestError(f"{rwhere} requests non-preregistered timepoints: {sorted(forbidden_timepoints, key=str)}")
            retry = run.get("retry", {})
            if not isinstance(retry.get("max_attempts", 1), int) or retry.get("max_attempts", 1) < 1:
                raise ManifestError(f"{rwhere}.retry.max_attempts must be >= 1")
            max_attempts = retry.get("max_attempts", 1)
            planned_seconds += timeout * max_attempts
            planned_cost += timeout * max_attempts / 3600 * rates[gpu]
            if gpu == "H100":
                if original.get("resources", {}).get("gpu") != "H100":
                    raise ManifestError(f"{rwhere} must explicitly set resources.gpu=H100")
                planned_h100_cost += timeout * max_attempts / 3600 * rates[gpu]

        if planned_seconds > max_seconds:
            raise BudgetError(f"phase {phase_id} plans {planned_seconds:g}s over {max_seconds:g}s ceiling")
        if planned_cost > max_cost + 1e-9:
            raise BudgetError(f"phase {phase_id} forecasts ${planned_cost:.4f} over ${max_cost:.4f} ceiling")
        if "H100" in allowed_gpus:
            _validate_h100_phase(phase, phases_by_id, defaults, planned_h100_cost, where)
        for gate in phase.get("gates", []):
            _valid_id(gate.get("id"), f"{where}.gates.id")
            if gate.get("operator") not in {">", ">=", "<", "<=", "==", "!="}:
                raise ManifestError(f"gate {gate.get('id')} has unsupported operator")
            _finite_number(gate.get("threshold"), f"gate {gate.get('id')} threshold")


def _validate_h100_phase(
    phase: dict[str, Any],
    phases_by_id: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    planned_h100_cost: float,
    where: str,
) -> None:
    h100 = phase.get("h100_justification")
    if not isinstance(h100, dict) or h100.get("opt_in") is not True:
        raise ManifestError(f"{where}.h100_justification.opt_in must be true")
    profile_phase_id = h100.get("profile_phase_id")
    if profile_phase_id not in phase.get("depends_on", []):
        raise ManifestError(f"{where} must depend_on its H100 profile_phase_id")
    profile_phase = phases_by_id.get(profile_phase_id)
    if profile_phase is None or profile_phase is phase:
        raise ManifestError(f"{where}.h100_justification.profile_phase_id must reference an earlier phase")
    profile_runs = profile_phase.get("runs", [])
    has_profile_smoke = any(
        run.get("smoke_test") is True
        and resolve_run(defaults, run).get("resources", {}).get("gpu") in H100_PROFILE_GPUS
        for run in profile_runs
    )
    if not has_profile_smoke:
        raise ManifestError(
            f"{where} requires an earlier smoke_test run on one of {sorted(H100_PROFILE_GPUS)}"
        )
    if not profile_phase.get("gates"):
        raise ManifestError(f"H100 profile phase {profile_phase_id} must have a stop-on-failure gate")
    if not phase.get("gates"):
        raise ManifestError(f"{where} must have a stop-on-failure result gate")
    for field in ("measured_peak_memory_gb", "measured_runtime_seconds", "forecast_cost_usd", "remaining_budget_usd"):
        value = _finite_number(h100.get(field), f"{where}.h100_justification.{field}")
        if value < 0:
            raise ManifestError(f"{where}.h100_justification.{field} must be nonnegative")
    if h100["measured_peak_memory_gb"] <= 0 or h100["measured_runtime_seconds"] <= 0:
        raise ManifestError(f"{where} requires positive measured memory and runtime")
    if not isinstance(h100.get("why_lower_gpu_insufficient"), str) or not h100["why_lower_gpu_insufficient"].strip():
        raise ManifestError(f"{where} requires why_lower_gpu_insufficient")
    if abs(float(h100["forecast_cost_usd"]) - planned_h100_cost) > 1e-6:
        raise BudgetError(
            f"{where} H100 forecast must equal retry-aware planned cost ${planned_h100_cost:.6f}"
        )
    if h100["forecast_cost_usd"] > h100["remaining_budget_usd"] + 1e-9:
        raise BudgetError(f"{where} H100 forecast exceeds declared remaining budget")


def uses_h100(manifest: dict[str, Any], *, smoke: bool = False) -> bool:
    for phase, original in iter_runs(manifest):
        if smoke and not original.get("smoke_test", False):
            continue
        run = resolve_run(manifest.get("defaults", {}), original)
        if run.get("resources", {}).get("gpu") == "H100":
            return True
    return False


def authorize_launch(
    manifest: dict[str, Any], *, smoke: bool, confirm_paid: bool, confirm_h100: bool
) -> None:
    """Authoritative launch policy, called by both local and remote boundaries."""
    validate_manifest(manifest)
    plan = make_plan(manifest, smoke=smoke)
    if smoke:
        selected_runs = [
            resolve_run(manifest.get("defaults", {}), run)
            for phase in manifest["phases"]
            for run in phase["runs"]
            if run.get("smoke_test") is True
        ]
        if not selected_runs:
            raise ManifestError("smoke submission selected no explicitly marked runs")
        if any(run["resources"]["gpu"] not in SMOKE_ALLOWED_RESOURCES for run in selected_runs):
            raise ManifestError(f"smoke resources are limited to {sorted(SMOKE_ALLOWED_RESOURCES)}")
        if sum(item["compute_seconds"] for item in plan["phases"]) > SMOKE_MAX_RUNTIME_SECONDS:
            raise BudgetError(f"smoke plan exceeds {SMOKE_MAX_RUNTIME_SECONDS}s hard runtime cap")
        if plan["total_forecast_cost_usd"] > SMOKE_MAX_COST_USD:
            raise BudgetError(f"smoke plan exceeds ${SMOKE_MAX_COST_USD:.2f} hard forecast cap")
        return
    if not (confirm_paid and manifest.get("launch_policy", {}).get("allow_full_matrix") is True):
        raise ManifestError(
            "full matrix submission is locked; require both launch_policy.allow_full_matrix=true and --confirm-paid"
        )
    if uses_h100(manifest) and not (
        confirm_h100 and manifest.get("launch_policy", {}).get("allow_h100") is True
    ):
        raise ManifestError(
            "H100 submission is locked; require launch_policy.allow_h100=true and --confirm-h100"
        )


def resolve_run(defaults: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(run)
    for section in ("resources", "retry"):
        combined = copy.deepcopy(defaults.get(section, {}))
        combined.update(resolved.get(section, {}))
        resolved[section] = combined
    return resolved


def make_plan(manifest: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for phase in manifest["phases"]:
        phase_runs = [resolve_run(manifest.get("defaults", {}), run) for run in phase["runs"]]
        if smoke:
            phase_runs = [run for run in phase_runs if run.get("smoke_test", False)]
        if not phase_runs:
            continue
        rates = phase["budget"]["gpu_hourly_cost_usd"]
        selected.append(
            {
                "phase_id": phase["id"],
                "depends_on": phase.get("depends_on", []),
                "runs": [run["id"] for run in phase_runs],
                "compute_seconds": sum(
                    run["resources"]["timeout_seconds"] * run.get("retry", {}).get("max_attempts", 1)
                    for run in phase_runs
                ),
                "gpu_seconds": sum(
                    run["resources"]["timeout_seconds"] * run.get("retry", {}).get("max_attempts", 1)
                    for run in phase_runs
                    if run["resources"]["gpu"] != "cpu"
                ),
                "forecast_cost_usd": round(
                    sum(
                        run["resources"]["timeout_seconds"]
                        * run.get("retry", {}).get("max_attempts", 1)
                        / 3600
                        * rates[run["resources"]["gpu"]]
                        for run in phase_runs
                    ),
                    6,
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": manifest_hash(manifest),
        "mode": "smoke" if smoke else "full",
        "phases": selected,
        "total_forecast_cost_usd": round(sum(item["forecast_cost_usd"] for item in selected), 6),
    }
