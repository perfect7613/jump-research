"""Closed, content-addressed contracts for general computational experiments.

The plan contains declarations and hashes, never executable source bytes or
planner-controlled runtime capabilities.  The run binds every result back to
the exact plan that was confirmed before prediction and execution.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .evidence import EvidenceError, open_result_envelope, validate_run_evidence_object

EXPERIMENT_PLAN_VERSION = "jump.experiment-plan/v1"
EXPERIMENT_RUN_VERSION = "jump.experiment-run/v1"
RESTRICTED_ADAPTER_ID = "modal.restricted-python-simulation/v1"

_SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_ID = {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$"}
_SCALAR = {"type": ["string", "number", "integer", "boolean", "null"]}
_SCALAR_OR_ARRAY = {"oneOf": [_SCALAR, {"type": "array", "items": _SCALAR, "maxItems": 100}]}

RESTRICTED_POLICY = {
    "schema_version": "jump.restricted-python-policy/v1",
    "entrypoint": "simulate",
    "source_bytes": 16_384,
    "ast_nodes": 1_200,
    "allowed_imports": ["collections", "heapq", "math", "random", "statistics"],
    "allowed_builtins": [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
        "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    ],
    "allowed_attributes": [
        "Counter", "Random", "append", "ceil", "choice", "choices", "copy", "cos", "count",
        "exp", "expovariate", "extend", "floor", "gauss", "get", "heappop", "heappush", "index",
        "isfinite", "items", "keys", "log", "mean", "median", "popleft", "pop", "pow", "pstdev",
        "randint", "random", "randrange", "sample", "shuffle", "sin", "sort", "sqrt", "uniform",
        "update", "values",
    ],
    "banned_names": [
        "breakpoint", "compile", "delattr", "eval", "exec", "getattr", "globals", "help", "input",
        "locals", "memoryview", "open", "setattr", "type", "vars", "__import__",
    ],
    "banned_modules": [
        "asyncio", "builtins", "ctypes", "ftplib", "http", "importlib", "inspect", "marshal",
        "multiprocessing", "os", "pathlib", "pickle", "pip", "requests", "shelve", "shutil", "socket",
        "subprocess", "sys", "tempfile", "threading", "urllib",
    ],
    "filesystem": False,
    "network": False,
    "subprocesses": False,
    "dynamic_code": False,
}
RESTRICTED_POLICY_SHA256 = hashlib.sha256(
    json.dumps(RESTRICTED_POLICY, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
).hexdigest()
DEPENDENCY_LOCK = {"python": "3.11", "distributions": []}
DEPENDENCY_LOCK_SHA256 = hashlib.sha256(
    json.dumps(DEPENDENCY_LOCK, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
).hexdigest()

FIXED_SANDBOX = {
    "adapter_id": RESTRICTED_ADAPTER_ID,
    "policy_sha256": RESTRICTED_POLICY_SHA256,
    "python": {"version": "3.11", "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256},
    "limits": {
        "cpu_cores": 1.0,
        "memory_mb": 512,
        "timeout_seconds": 30,
        "stdout_bytes": 8192,
        "result_bytes": 65536,
        "max_rows": 200,
        "max_columns": 20,
    },
    "capabilities": {
        "network": False,
        "modal_access": False,
        "single_use": True,
        "secrets": [],
        "volumes": [],
        "filesystem": False,
        "subprocesses": False,
    },
}


def _closed_object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


EXPERIMENT_PLAN_SCHEMA: dict[str, Any] = _closed_object(
    {
        "schema_version": {"const": EXPERIMENT_PLAN_VERSION},
        "plan_id": {"type": "string", "pattern": "^plan-[0-9a-f]{24}$"},
        "intent": {"type": "string", "minLength": 1, "maxLength": 2000},
        "hypothesis": {"type": "string", "minLength": 1, "maxLength": 1000},
        "variables": _closed_object(
            {
                "independent": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": _closed_object(
                        {"id": _ID, "label": {"type": "string", "minLength": 1, "maxLength": 120},
                         "levels": {"type": "array", "minItems": 2, "maxItems": 20, "items": _SCALAR}},
                        ["id", "label", "levels"],
                    ),
                },
                "dependent": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": _closed_object(
                        {"id": _ID, "label": {"type": "string", "minLength": 1, "maxLength": 120},
                         "unit": {"type": ["string", "null"], "maxLength": 40}},
                        ["id", "label", "unit"],
                    ),
                },
                "controlled": {
                    "type": "array",
                    "maxItems": 16,
                    "items": _closed_object(
                        {"id": _ID, "label": {"type": "string", "minLength": 1, "maxLength": 120},
                         "value": _SCALAR_OR_ARRAY},
                        ["id", "label", "value"],
                    ),
                },
            },
            ["independent", "dependent", "controlled"],
        ),
        "assumptions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
        "conditions": {
            "type": "array",
            "minItems": 2,
            "maxItems": 20,
            "items": _closed_object(
                {
                    "id": _ID,
                    "label": {"type": "string", "minLength": 1, "maxLength": 120},
                    "kind": {"enum": ["baseline", "intervention"]},
                    "assignments": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 16,
                        "additionalProperties": _SCALAR_OR_ARRAY,
                        "propertyNames": _ID,
                    },
                },
                ["id", "label", "kind", "assignments"],
            ),
        },
        "sampling": _closed_object(
            {
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "repetitions": {"type": "integer", "minimum": 1, "maximum": 100},
                "design": {"enum": ["paired_common_random_numbers", "independent_repetitions"]},
            },
            ["seed", "repetitions", "design"],
        ),
        "prediction_before_run": _closed_object(
            {
                "required": {"const": True},
                "targets": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": _closed_object(
                        {
                            "id": _ID,
                            "measurement_id": _ID,
                            "baseline_condition_id": _ID,
                            "intervention_condition_id": _ID,
                        },
                        ["id", "measurement_id", "baseline_condition_id", "intervention_condition_id"],
                    ),
                },
            },
            ["required", "targets"],
        ),
        "sandbox": _closed_object(
            {
                "adapter_id": {"const": RESTRICTED_ADAPTER_ID},
                "source": _closed_object(
                    {
                        "sha256": _SHA256,
                        "byte_length": {"type": "integer", "minimum": 1, "maximum": 16384},
                        "media_type": {"const": "text/x-python"},
                    },
                    ["sha256", "byte_length", "media_type"],
                ),
                "policy_sha256": _SHA256,
                "python": _closed_object(
                    {"version": {"const": "3.11"}, "dependency_lock_sha256": _SHA256},
                    ["version", "dependency_lock_sha256"],
                ),
                "limits": _closed_object(
                    {
                        "cpu_cores": {"const": 1.0}, "memory_mb": {"const": 512},
                        "timeout_seconds": {"const": 30}, "stdout_bytes": {"const": 8192},
                        "result_bytes": {"const": 65536}, "max_rows": {"const": 200},
                        "max_columns": {"const": 20},
                    },
                    ["cpu_cores", "memory_mb", "timeout_seconds", "stdout_bytes", "result_bytes", "max_rows", "max_columns"],
                ),
                "capabilities": _closed_object(
                    {
                        "network": {"const": False}, "modal_access": {"const": False},
                        "single_use": {"const": True}, "secrets": {"const": []},
                        "volumes": {"const": []}, "filesystem": {"const": False},
                        "subprocesses": {"const": False},
                    },
                    ["network", "modal_access", "single_use", "secrets", "volumes", "filesystem", "subprocesses"],
                ),
            },
            ["adapter_id", "source", "policy_sha256", "python", "limits", "capabilities"],
        ),
        "measurements": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": _closed_object(
                {"id": _ID, "label": {"type": "string", "minLength": 1, "maxLength": 120},
                 "unit": {"type": ["string", "null"], "maxLength": 40},
                 "aggregation": {"const": "mean"},
                 "display": {"enum": ["table", "line", "bar", "histogram"]}},
                ["id", "label", "unit", "aggregation", "display"],
            ),
        },
        "comparisons": {
            "type": "array", "minItems": 1, "maxItems": 20,
            "items": _closed_object(
                {"id": _ID, "measurement_id": _ID, "baseline_condition_id": _ID,
                 "intervention_condition_id": _ID, "statistic": {"const": "mean_difference"},
                 "pairing": {"enum": ["paired_by_repetition", "independent_samples"]}},
                ["id", "measurement_id", "baseline_condition_id", "intervention_condition_id", "statistic", "pairing"],
            ),
        },
        "plan_sha256": _SHA256,
    },
    ["schema_version", "plan_id", "intent", "hypothesis", "variables", "assumptions", "conditions",
     "sampling", "prediction_before_run", "sandbox", "measurements", "comparisons", "plan_sha256"],
)

_PREDICTION = _closed_object(
    {
        "summary": {"type": "string", "minLength": 1, "maxLength": 1000},
        "claims": {
            "type": "array", "minItems": 1, "maxItems": 20,
            "items": _closed_object(
                {"target_id": _ID, "expected_relation": {"enum": ["greater", "less", "equal", "different"]},
                 "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
                 "expected_value": {"type": ["number", "null"]}},
                ["target_id", "expected_relation", "rationale", "expected_value"],
            ),
        },
    },
    ["summary", "claims"],
)

EXPERIMENT_RUN_SCHEMA: dict[str, Any] = _closed_object(
    {
        "schema_version": {"const": EXPERIMENT_RUN_VERSION},
        "run_id": {"type": "string", "pattern": "^run-[0-9a-f]{24}$"},
        "plan_id": {"type": "string", "pattern": "^plan-[0-9a-f]{24}$"},
        "plan_sha256": _SHA256,
        "status": {"enum": ["completed", "failed"]},
        "execution": _closed_object(
            {
                "plan_sha256": _SHA256,
                "prediction": _PREDICTION,
                "prediction_sha256": _SHA256,
                "prediction_recorded_at": {"type": "string", "format": "date-time"},
                "started_at": {"type": ["string", "null"], "format": "date-time"},
                "completed_at": {"type": ["string", "null"], "format": "date-time"},
                "source_sha256": _SHA256,
                "policy_sha256": _SHA256,
                "code_version": {"type": "string", "minLength": 1, "maxLength": 200},
                "modal_call_id": {"type": ["string", "null"], "maxLength": 200},
                "error": {"type": ["string", "null"], "maxLength": 1000},
            },
            ["plan_sha256", "prediction", "prediction_sha256", "prediction_recorded_at", "started_at",
             "completed_at", "source_sha256", "policy_sha256", "code_version", "modal_call_id", "error"],
        ),
        "measurements": {
            "type": "array", "maxItems": 200,
            "items": _closed_object(
                {"plan_sha256": _SHA256, "condition_id": _ID,
                 "repetition": {"type": "integer", "minimum": 0, "maximum": 99},
                 "pairing_key": {"type": "string", "minLength": 1, "maxLength": 100},
                 "values": {"type": "object", "minProperties": 1, "maxProperties": 8,
                            "propertyNames": _ID, "additionalProperties": {"type": "number"}}},
                ["plan_sha256", "condition_id", "repetition", "pairing_key", "values"],
            ),
        },
        "comparisons": {
            "type": "array", "maxItems": 20,
            "items": _closed_object(
                {"id": _ID, "plan_sha256": _SHA256, "measurement_id": _ID,
                 "baseline_condition_id": _ID, "intervention_condition_id": _ID,
                 "pairing_keys": {"type": "array", "minItems": 1, "maxItems": 100,
                                  "items": {"type": "string", "minLength": 1, "maxLength": 100}},
                 "pair_set_sha256": _SHA256, "estimate": {"type": "number"}},
                ["id", "plan_sha256", "measurement_id", "baseline_condition_id", "intervention_condition_id",
                 "pairing_keys", "pair_set_sha256", "estimate"],
            ),
        },
        "revision": _closed_object(
            {"plan_sha256": _SHA256, "disposition": {"enum": ["retain", "revise", "reject"]},
             "interpretation": {"type": "string", "minLength": 1, "maxLength": 2000},
             "next_plan_sha256": {"oneOf": [_SHA256, {"type": "null"}]}},
            ["plan_sha256", "disposition", "interpretation", "next_plan_sha256"],
        ),
        "evidence": _closed_object(
            {"plan_sha256": _SHA256, "source_sha256": _SHA256, "policy_sha256": _SHA256,
             "code_version": {"type": "string", "minLength": 1, "maxLength": 200},
             "modal_call_id": {"type": ["string", "null"], "maxLength": 200},
             "run_result_sha256": _SHA256, "artifact_inventory_sha256": _SHA256,
             "sealed_payload_sha256": _SHA256},
            ["plan_sha256", "source_sha256", "policy_sha256", "code_version", "modal_call_id",
             "run_result_sha256", "artifact_inventory_sha256", "sealed_payload_sha256"],
        ),
        "run_sha256": _SHA256,
    },
    ["schema_version", "run_id", "plan_id", "plan_sha256", "status", "execution", "measurements",
     "comparisons", "revision", "evidence", "run_sha256"],
)


class ExperimentContractError(ValueError):
    """Raised when a plan or run weakens the public contract."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not canonical JSON: {exc}") from exc


def schema_sha256(schema: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(schema)).hexdigest()


EXPERIMENT_PLAN_SCHEMA_SHA256 = schema_sha256(EXPERIMENT_PLAN_SCHEMA)
EXPERIMENT_RUN_SCHEMA_SHA256 = schema_sha256(EXPERIMENT_RUN_SCHEMA)


def _content_hash(value: Mapping[str, Any], *, id_key: str, hash_key: str) -> str:
    preimage = {key: item for key, item in value.items() if key not in {id_key, hash_key}}
    return hashlib.sha256(canonical_json(preimage)).hexdigest()


def build_experiment_plan(**fields: Any) -> dict[str, Any]:
    """Build and validate a plan; server code must supply the fixed sandbox."""
    if {"schema_version", "plan_id", "plan_sha256"} & fields.keys():
        raise ExperimentContractError("schema_version, plan_id, and plan_sha256 are derived")
    plan = {"schema_version": EXPERIMENT_PLAN_VERSION, **deepcopy(fields)}
    digest = _content_hash(plan, id_key="plan_id", hash_key="plan_sha256")
    plan["plan_id"] = f"plan-{digest[:24]}"
    plan["plan_sha256"] = digest
    validate_experiment_plan(plan)
    return plan


def validate_experiment_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(value))
    _validate_schema(EXPERIMENT_PLAN_SCHEMA, plan, "ExperimentPlan")
    digest = _content_hash(plan, id_key="plan_id", hash_key="plan_sha256")
    if plan["plan_sha256"] != digest or plan["plan_id"] != f"plan-{digest[:24]}":
        raise ExperimentContractError("ExperimentPlan content hash or derived plan_id does not match")
    _validate_plan_references(plan)
    if {key: plan["sandbox"][key] for key in FIXED_SANDBOX} != FIXED_SANDBOX:
        raise ExperimentContractError("sandbox policy is server-owned and must match the fixed policy")
    return plan


def build_experiment_run(
    plan: Mapping[str, Any],
    *,
    verified_run_result: Mapping[str, Any] | bytes,
    artifact_bytes: Mapping[str, bytes],
    sealed_result: Mapping[str, Any] | bytes,
    **fields: Any,
) -> dict[str, Any]:
    """Build a run from an exact plan and independently supplied evidence bytes."""
    checked_plan = validate_experiment_plan(plan)
    if {"schema_version", "run_id", "run_sha256"} & fields.keys():
        raise ExperimentContractError("schema_version, run_id, and run_sha256 are derived")
    run = {"schema_version": EXPERIMENT_RUN_VERSION, **deepcopy(fields)}
    if run.get("plan_id") != checked_plan["plan_id"] or run.get("plan_sha256") != checked_plan["plan_sha256"]:
        raise ExperimentContractError("run fields must bind the required ExperimentPlan")
    if not isinstance(run.get("evidence"), dict):
        raise ExperimentContractError("run requires an evidence record")
    evidence_hashes = _verify_evidence(run, checked_plan, verified_run_result, artifact_bytes, sealed_result)
    run["evidence"].update(evidence_hashes)
    digest = _content_hash(run, id_key="run_id", hash_key="run_sha256")
    run["run_id"] = f"run-{digest[:24]}"
    run["run_sha256"] = digest
    validate_experiment_run(
        run,
        checked_plan,
        verified_run_result=verified_run_result,
        artifact_bytes=artifact_bytes,
        sealed_result=sealed_result,
    )
    return run


def validate_experiment_run(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    verified_run_result: Mapping[str, Any] | bytes,
    artifact_bytes: Mapping[str, bytes],
    sealed_result: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    run = deepcopy(dict(value))
    checked_plan = validate_experiment_plan(plan)
    _validate_schema(EXPERIMENT_RUN_SCHEMA, run, "ExperimentRun")
    digest = _content_hash(run, id_key="run_id", hash_key="run_sha256")
    if run["run_sha256"] != digest or run["run_id"] != f"run-{digest[:24]}":
        raise ExperimentContractError("ExperimentRun content hash or derived run_id does not match")
    plan_sha = run["plan_sha256"]
    bound = [run["execution"], *run["measurements"], *run["comparisons"], run["revision"], run["evidence"]]
    if any(record["plan_sha256"] != plan_sha for record in bound):
        raise ExperimentContractError("every run record must repeat the top-level plan_sha256")
    if run["execution"]["prediction_sha256"] != hashlib.sha256(canonical_json(run["execution"]["prediction"])).hexdigest():
        raise ExperimentContractError("prediction_sha256 does not bind the prediction")
    _validate_prediction_order(run["execution"])
    verified_hashes = _verify_evidence(
        run, checked_plan, verified_run_result, artifact_bytes, sealed_result
    )
    if {
        key: run["evidence"][key]
        for key in ("run_result_sha256", "artifact_inventory_sha256", "sealed_payload_sha256")
    } != verified_hashes:
        raise ExperimentContractError("evidence hashes do not match the verified evidence objects")
    _validate_run_against_plan(run, checked_plan)
    return run


def _verify_evidence(
    run: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_result: Mapping[str, Any] | bytes,
    artifact_bytes: Mapping[str, bytes],
    sealed_result: Mapping[str, Any] | bytes,
) -> dict[str, str]:
    execution = run.get("execution")
    if not isinstance(execution, Mapping):
        raise ExperimentContractError("run execution is required before evidence can be verified")
    modal_call_id = execution.get("modal_call_id")
    if not isinstance(modal_call_id, str) or not modal_call_id:
        raise ExperimentContractError("verified evidence requires a Modal call identity")
    run_result_value, run_result_bytes = _verified_json(run_result, Mapping, "run result")
    sealed_value, _ = _verified_json(sealed_result, Mapping, "sealed result")
    try:
        normative_result = validate_run_evidence_object(
            run_result_value,
            artifact_bytes=artifact_bytes,
            expected_manifest_sha256=plan["plan_sha256"],
            expected_run_id=modal_call_id,
            expected_code_version=execution["code_version"],
            require_completed=run.get("status") == "completed",
        )
        payload = open_result_envelope(
            sealed_value,
            expected_source="live",
            expected_manifest_sha256=plan["plan_sha256"],
            expected_checkpoint_id=plan["plan_id"],
        )
    except EvidenceError as exc:
        raise ExperimentContractError(f"canonical evidence verification failed: {exc}") from exc
    provenance = normative_result["provenance"]
    if normative_result["status"] != run["status"]:
        raise ExperimentContractError("run evidence status does not match the ExperimentRun")
    if provenance.get("source_sha256") != execution["source_sha256"]:
        raise ExperimentContractError("run evidence source does not match execution")
    if provenance.get("policy_sha256") != execution["policy_sha256"]:
        raise ExperimentContractError("run evidence policy does not match execution")
    if normative_result.get("plan_sha256") != plan["plan_sha256"]:
        raise ExperimentContractError("run evidence does not repeat the required plan")
    if normative_result.get("measurements") != run["measurements"]:
        raise ExperimentContractError("run evidence measurements do not match the ExperimentRun")
    if normative_result.get("comparisons") != run["comparisons"]:
        raise ExperimentContractError("run evidence comparisons do not match the ExperimentRun")
    expected_metrics = [
        {"name": comparison["id"], "value": comparison["estimate"]}
        for comparison in run["comparisons"]
    ]
    if normative_result["metrics"] != expected_metrics:
        raise ExperimentContractError("run evidence metrics do not match the frozen comparisons")
    sealed_provenance = sealed_value["provenance"]
    if sealed_provenance["run_id"] != modal_call_id:
        raise ExperimentContractError("sealed result does not match the Modal call identity")
    if sealed_provenance["code_version"] != execution["code_version"]:
        raise ExperimentContractError("sealed result does not match the execution code version")
    expected_payload = {
        "plan_sha256": run["plan_sha256"],
        "prediction": run["execution"]["prediction"],
        "measurements": run["measurements"],
        "comparisons": run["comparisons"],
        "revision": run["revision"],
    }
    if payload != expected_payload:
        raise ExperimentContractError("sealed result payload does not match the ExperimentRun")
    inventory_bytes = canonical_json(normative_result["artifacts"])
    return {
        "run_result_sha256": hashlib.sha256(run_result_bytes).hexdigest(),
        "artifact_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "sealed_payload_sha256": sealed_value["payload_sha256"],
    }


def _verified_json(value: Any, expected_type: type, label: str) -> tuple[Any, bytes]:
    if isinstance(value, bytes):
        try:
            decoded = json.loads(value)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ExperimentContractError(f"verified {label} bytes must be canonical JSON") from exc
        encoded = canonical_json(decoded)
        if value != encoded:
            raise ExperimentContractError(f"verified {label} bytes are not in canonical form")
        value = decoded
    if not isinstance(value, expected_type):
        raise ExperimentContractError(f"verified {label} has the wrong JSON type")
    return value, canonical_json(value)


def _validate_schema(schema: Mapping[str, Any], value: Any, label: str) -> None:
    errors = sorted(Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        error = errors[0]
        where = ".".join(str(part) for part in error.path) or "<root>"
        raise ExperimentContractError(f"{label} {where}: {error.message}")


def _unique_ids(records: list[dict[str, Any]], label: str) -> set[str]:
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ExperimentContractError(f"{label} IDs must be unique")
    return set(ids)


def _validate_plan_references(plan: dict[str, Any]) -> None:
    independent = _unique_ids(plan["variables"]["independent"], "independent variable")
    dependent = _unique_ids(plan["variables"]["dependent"], "dependent variable")
    controlled = _unique_ids(plan["variables"]["controlled"], "controlled variable")
    if independent & dependent or independent & controlled or dependent & controlled:
        raise ExperimentContractError("variable IDs must be unique across roles")
    conditions = plan["conditions"]
    condition_ids = _unique_ids(conditions, "condition")
    baselines = [item for item in conditions if item["kind"] == "baseline"]
    interventions = [item for item in conditions if item["kind"] == "intervention"]
    if len(baselines) != 1 or not interventions:
        raise ExperimentContractError("conditions require exactly one baseline and at least one intervention")
    if any(set(item["assignments"]) != independent for item in conditions):
        raise ExperimentContractError("every condition must assign each independent variable and no controlled variable")
    levels = {item["id"]: item["levels"] for item in plan["variables"]["independent"]}
    for condition in conditions:
        for variable_id, assigned in condition["assignments"].items():
            assigned_values = assigned if isinstance(assigned, list) else [assigned]
            if any(value not in levels[variable_id] for value in assigned_values):
                raise ExperimentContractError("condition assignment is outside the independent variable levels")
    if len(conditions) * plan["sampling"]["repetitions"] > plan["sandbox"]["limits"]["max_rows"]:
        raise ExperimentContractError("condition/repetition design exceeds the fixed result row limit")
    measurement_ids = _unique_ids(plan["measurements"], "measurement")
    if measurement_ids != dependent:
        raise ExperimentContractError("measurements must cover each dependent variable exactly once")
    targets = plan["prediction_before_run"]["targets"]
    target_ids = _unique_ids(targets, "prediction target")
    comparison_ids = _unique_ids(plan["comparisons"], "comparison")
    if target_ids != comparison_ids:
        raise ExperimentContractError("prediction target IDs must match comparison IDs")
    baseline_id = baselines[0]["id"]
    for item in [*targets, *plan["comparisons"]]:
        if item["measurement_id"] not in measurement_ids:
            raise ExperimentContractError("prediction/comparison references an unknown measurement")
        if item["baseline_condition_id"] != baseline_id:
            raise ExperimentContractError("prediction/comparison must reference the sole baseline")
        if item["intervention_condition_id"] not in condition_ids or item["intervention_condition_id"] == baseline_id:
            raise ExperimentContractError("prediction/comparison references an unknown intervention")
    target_by_id = {item["id"]: item for item in targets}
    for comparison in plan["comparisons"]:
        target = target_by_id[comparison["id"]]
        for key in ("measurement_id", "baseline_condition_id", "intervention_condition_id"):
            if comparison[key] != target[key]:
                raise ExperimentContractError("prediction target and comparison with the same ID must bind the same test")
    expected_pairing = "paired_by_repetition" if plan["sampling"]["design"] == "paired_common_random_numbers" else "independent_samples"
    if any(item["pairing"] != expected_pairing for item in plan["comparisons"]):
        raise ExperimentContractError("comparison pairing must match the sampling design")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_prediction_order(execution: dict[str, Any]) -> None:
    predicted = _parse_time(execution["prediction_recorded_at"])
    started = execution["started_at"]
    completed = execution["completed_at"]
    if started is not None and predicted >= _parse_time(started):
        raise ExperimentContractError("prediction must be recorded before execution starts")
    if started is not None and completed is not None and _parse_time(started) > _parse_time(completed):
        raise ExperimentContractError("execution cannot complete before it starts")


def _validate_run_against_plan(run: dict[str, Any], plan: dict[str, Any]) -> None:
    if run["plan_id"] != plan["plan_id"] or run["plan_sha256"] != plan["plan_sha256"]:
        raise ExperimentContractError("run does not bind the supplied plan")
    execution = run["execution"]
    evidence = run["evidence"]
    if execution["source_sha256"] != plan["sandbox"]["source"]["sha256"]:
        raise ExperimentContractError("execution source does not match the plan")
    if execution["policy_sha256"] != plan["sandbox"]["policy_sha256"]:
        raise ExperimentContractError("execution policy does not match the plan")
    for key in ("source_sha256", "policy_sha256", "code_version", "modal_call_id"):
        if evidence[key] != execution[key]:
            raise ExperimentContractError(f"evidence.{key} does not match execution.{key}")
    target_ids = {item["id"] for item in plan["prediction_before_run"]["targets"]}
    if {item["target_id"] for item in execution["prediction"]["claims"]} != target_ids:
        raise ExperimentContractError("prediction claims must cover each frozen prediction target")
    measurement_ids = {item["id"] for item in plan["measurements"]}
    condition_ids = {item["id"] for item in plan["conditions"]}
    expected_rows = {
        (condition_id, repetition)
        for condition_id in condition_ids
        for repetition in range(plan["sampling"]["repetitions"])
    }
    seen_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for record in run["measurements"]:
        if record["condition_id"] not in condition_ids or set(record["values"]) != measurement_ids:
            raise ExperimentContractError("measurement record does not cover the frozen plan")
        if record["repetition"] >= plan["sampling"]["repetitions"]:
            raise ExperimentContractError("measurement repetition exceeds the frozen sampling plan")
        row_key = (record["condition_id"], record["repetition"])
        if row_key in seen_rows:
            raise ExperimentContractError("measurement condition/repetition rows must be unique")
        expected_pairing_key = (
            f"rep-{record['repetition']}"
            if plan["sampling"]["design"] == "paired_common_random_numbers"
            else f"{record['condition_id']}:rep-{record['repetition']}"
        )
        if record["pairing_key"] != expected_pairing_key:
            raise ExperimentContractError("measurement pairing_key must be server-derived from the sampling plan")
        seen_rows[row_key] = record
    if run["status"] == "completed" and set(seen_rows) != expected_rows:
        raise ExperimentContractError("completed runs require every condition/repetition measurement exactly once")
    if run["status"] == "failed" and run["comparisons"]:
        raise ExperimentContractError("failed runs cannot contain comparisons")
    if run["status"] == "completed" and not execution["modal_call_id"]:
        raise ExperimentContractError("completed runs require a Modal call ID")
    planned_comparisons = {item["id"]: item for item in plan["comparisons"]}
    if run["status"] == "completed" and {item["id"] for item in run["comparisons"]} != set(planned_comparisons):
        raise ExperimentContractError("run comparisons must cover the frozen comparison specs")
    for record in run["comparisons"]:
        spec = planned_comparisons[record["id"]]
        for key in ("measurement_id", "baseline_condition_id", "intervention_condition_id"):
            if record[key] != spec[key]:
                raise ExperimentContractError("run comparison does not match its frozen spec")
        expected_pairing_keys, pair_set, estimate = _comparison_evidence(spec, plan, seen_rows)
        if record["pairing_keys"] != expected_pairing_keys:
            raise ExperimentContractError("comparison pairing_keys do not bind the measured rows")
        pair_hash = hashlib.sha256(canonical_json(pair_set)).hexdigest()
        if record["pair_set_sha256"] != pair_hash:
            raise ExperimentContractError("pair_set_sha256 does not bind the paired row hashes and values")
        if record["estimate"] != estimate:
            raise ExperimentContractError("comparison estimate must be recomputed from measured rows")


def comparison_records(plan: Mapping[str, Any], measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute all frozen mean differences and row-bound pair-set hashes."""
    checked = validate_experiment_plan(plan)
    rows = {(row["condition_id"], row["repetition"]): row for row in measurements}
    records: list[dict[str, Any]] = []
    for spec in checked["comparisons"]:
        pairing_keys, pair_set, estimate = _comparison_evidence(spec, checked, rows)
        records.append(
            {
                "id": spec["id"],
                "plan_sha256": checked["plan_sha256"],
                "measurement_id": spec["measurement_id"],
                "baseline_condition_id": spec["baseline_condition_id"],
                "intervention_condition_id": spec["intervention_condition_id"],
                "pairing_keys": pairing_keys,
                "pair_set_sha256": hashlib.sha256(canonical_json(pair_set)).hexdigest(),
                "estimate": estimate,
            }
        )
    return records


def _comparison_evidence(
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[str], Any, float]:
    baseline_rows: list[Mapping[str, Any]] = []
    intervention_rows: list[Mapping[str, Any]] = []
    for repetition in range(plan["sampling"]["repetitions"]):
        try:
            baseline_rows.append(rows[(spec["baseline_condition_id"], repetition)])
            intervention_rows.append(rows[(spec["intervention_condition_id"], repetition)])
        except KeyError as exc:
            raise ExperimentContractError("comparison is missing a required measured row") from exc
    measurement_id = spec["measurement_id"]
    estimate = float(
        sum(row["values"][measurement_id] for row in intervention_rows) / len(intervention_rows)
        - sum(row["values"][measurement_id] for row in baseline_rows) / len(baseline_rows)
    )
    if plan["sampling"]["design"] == "paired_common_random_numbers":
        pair_set = []
        pairing_keys = []
        for repetition, (baseline, intervention) in enumerate(zip(baseline_rows, intervention_rows)):
            pair = {
                "repetition": repetition,
                "pairing_key": f"rep-{repetition}",
                "baseline_row_sha256": hashlib.sha256(canonical_json(baseline)).hexdigest(),
                "intervention_row_sha256": hashlib.sha256(canonical_json(intervention)).hexdigest(),
            }
            pair_set.append(pair)
            pairing_keys.append(hashlib.sha256(canonical_json(pair)).hexdigest())
    else:
        pair_set = {
            "design": "independent_repetitions",
            "baseline_row_sha256s": [hashlib.sha256(canonical_json(row)).hexdigest() for row in baseline_rows],
            "intervention_row_sha256s": [hashlib.sha256(canonical_json(row)).hexdigest() for row in intervention_rows],
        }
        pairing_keys = [
            hashlib.sha256(canonical_json({"baseline": baseline, "intervention": intervention})).hexdigest()
            for baseline, intervention in zip(
                pair_set["baseline_row_sha256s"], pair_set["intervention_row_sha256s"]
            )
        ]
    return pairing_keys, pair_set, estimate


__all__ = [
    "EXPERIMENT_PLAN_VERSION", "EXPERIMENT_RUN_VERSION", "RESTRICTED_ADAPTER_ID",
    "RESTRICTED_POLICY", "RESTRICTED_POLICY_SHA256", "DEPENDENCY_LOCK", "DEPENDENCY_LOCK_SHA256",
    "EXPERIMENT_PLAN_SCHEMA", "EXPERIMENT_RUN_SCHEMA", "EXPERIMENT_PLAN_SCHEMA_SHA256",
    "EXPERIMENT_RUN_SCHEMA_SHA256", "FIXED_SANDBOX", "ExperimentContractError",
    "canonical_json", "schema_sha256", "build_experiment_plan", "validate_experiment_plan",
    "build_experiment_run", "validate_experiment_run", "comparison_records",
]
