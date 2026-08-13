"""Closed contracts for declarative, visual thought experiments.

ExperimentSpec v2 contains data only. It cannot name source code, modules,
commands, paths, URLs, imports, packages, secrets, or runtime resources.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from fractions import Fraction
from typing import Any, Mapping

from jsonschema import Draft202012Validator

QUESTION_VERSION = "jump.thought-experiment-question/v2"
SPEC_VERSION = "jump.thought-experiment-spec/v2"
CONFIRMATION_VERSION = "jump.thought-experiment-confirmation/v2"
SPEC_RESPONSE_VERSION = "jump.thought-experiment-spec-response/v2"
RUN_VERSION = "jump.thought-experiment-run/v2"
RUN_RESPONSE_VERSION = "jump.thought-experiment-run-response/v2"
ENGINE_ID = "jump.declarative-visual-engine/v2"

_ID = {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$"}
_SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_NUMBER = {"type": "number", "minimum": -1000000, "maximum": 1000000}
_COLOR = {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"}


def _closed(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


ENTITY_SCHEMA = _closed(
    {
        "id": _ID,
        "label": {"type": "string", "minLength": 1, "maxLength": 80},
        "count": {"type": "integer", "minimum": 1, "maximum": 32},
        "appearance": _closed(
            {
                "shape": {"enum": ["circle", "square", "triangle"]},
                "color": _COLOR,
                "size": {"type": "number", "exclusiveMinimum": 0, "maximum": 20},
            },
            ["shape", "color", "size"],
        ),
        "initial_state": _closed(
            {
                "numeric": {
                    "type": "object", "maxProperties": 12,
                    "propertyNames": _ID, "additionalProperties": _NUMBER,
                },
                "categorical": {
                    "type": "object", "maxProperties": 8,
                    "propertyNames": _ID,
                    "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 40},
                },
            },
            ["numeric", "categorical"],
        ),
        "initial_layout": _closed(
            {
                "kind": {"enum": ["uniform", "grid", "ring", "line"]},
                "center": {"type": "array", "prefixItems": [_NUMBER, _NUMBER], "items": False},
                "spread": {"type": "number", "minimum": 0, "maximum": 1000},
            },
            ["kind", "center", "spread"],
        ),
    },
    ["id", "label", "count", "appearance", "initial_state", "initial_layout"],
)

RULE_OPS = {
    "move_2d": {"damping", "max_speed"},
    "random_walk_2d": {"step_scale"},
    "pairwise_force_2d": {"strength", "exponent", "softening"},
    "graph_diffusion": {"state", "rate"},
    "graph_contagion": {
        "state", "susceptible", "infected", "recovered",
        "transmission_probability", "recovery_probability",
    },
    "predator_prey_2d": {
        "prey_type", "predator_type", "prey_growth", "predation_rate",
        "predator_efficiency", "predator_decay", "interaction_radius",
    },
    "lane_traffic_2d": {"speed_state", "desired_speed", "headway", "road_y"},
    "queue_agents_2d": {
        "arrival_probability", "service_capacity", "queue_x", "service_x",
    },
}

RULE_SCHEMA = _closed(
    {
        "id": _ID,
        "op": {"enum": sorted(RULE_OPS)},
        "target_type": {"oneOf": [_ID, {"type": "null"}]},
        "parameters": {
            "type": "object", "minProperties": 1, "maxProperties": 12,
            "propertyNames": _ID,
            "additionalProperties": {
                "type": ["string", "number", "integer", "boolean", "null"]
            },
        },
    },
    ["id", "op", "target_type", "parameters"],
)

INTERVENTION_SCHEMA = _closed(
    {
        "time": {"type": "integer", "minimum": 0, "maximum": 500},
        "operation": {
            "enum": [
                "set_rule_parameter", "scale_rule_parameter",
                "set_numeric_state", "set_categorical_state",
            ]
        },
        "target": _ID,
        "field": _ID,
        "value": {"type": ["string", "number", "integer", "boolean"]},
    },
    ["time", "operation", "target", "field", "value"],
)

MEASUREMENT_SCHEMA = _closed(
    {
        "id": _ID,
        "label": {"type": "string", "minLength": 1, "maxLength": 100},
        "op": {"enum": ["mean_state", "sum_state", "variance_state", "count_category", "population_count"]},
        "entity_type": {"oneOf": [_ID, {"type": "null"}]},
        "state": {"oneOf": [_ID, {"type": "null"}]},
        "category": {"oneOf": [{"type": "string", "minLength": 1, "maxLength": 40}, {"type": "null"}]},
    },
    ["id", "label", "op", "entity_type", "state", "category"],
)

EXPERIMENT_SPEC_SCHEMA = _closed(
    {
        "schema_version": {"const": SPEC_VERSION},
        "spec_id": {"type": "string", "pattern": "^spec-[0-9a-f]{24}$"},
        "intent": {"type": "string", "minLength": 1, "maxLength": 2000},
        "question": {"type": "string", "minLength": 1, "maxLength": 500},
        "hypothesis": {"type": "string", "minLength": 1, "maxLength": 500},
        "world": _closed(
            {
                "bounds": _closed(
                    {
                        "width": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
                        "height": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
                        "boundary": {"enum": ["wrap", "reflect", "clamp"]},
                    },
                    ["width", "height", "boundary"],
                ),
                "entities": {"type": "array", "minItems": 1, "maxItems": 8, "items": ENTITY_SCHEMA},
                "graph": _closed(
                    {
                        "kind": {"enum": ["none", "ring", "grid", "random"]},
                        "edge_probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "directed": {"type": "boolean"},
                    },
                    ["kind", "edge_probability", "directed"],
                ),
            },
            ["bounds", "entities", "graph"],
        ),
        "dynamics": _closed(
            {
                "rules": {"type": "array", "minItems": 1, "maxItems": 12, "items": RULE_SCHEMA},
            },
            ["rules"],
        ),
        "conditions": {
            "type": "array", "minItems": 2, "maxItems": 4,
            "items": _closed(
                {
                    "id": _ID,
                    "label": {"type": "string", "minLength": 1, "maxLength": 80},
                    "kind": {"enum": ["baseline", "counterfactual"]},
                    "interventions": {"type": "array", "maxItems": 8, "items": INTERVENTION_SCHEMA},
                },
                ["id", "label", "kind", "interventions"],
            ),
        },
        "schedule": _closed(
            {
                "duration_steps": {"type": "integer", "minimum": 2, "maximum": 500},
                "dt": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "repetitions": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            ["duration_steps", "dt", "seed", "repetitions"],
        ),
        "measurements": {"type": "array", "minItems": 1, "maxItems": 12, "items": MEASUREMENT_SCHEMA},
        "visualization": _closed(
            {
                "kind": {"const": "animated_2d"},
                "frame_stride": {"type": "integer", "minimum": 1, "maximum": 100},
                "max_frames": {"type": "integer", "minimum": 2, "maximum": 40},
                "chart_measurement_ids": {"type": "array", "minItems": 1, "maxItems": 12, "items": _ID},
            },
            ["kind", "frame_stride", "max_frames", "chart_measurement_ids"],
        ),
        "spec_sha256": _SHA,
    },
    [
        "schema_version", "spec_id", "intent", "question", "hypothesis", "world",
        "dynamics", "conditions", "schedule", "measurements", "visualization", "spec_sha256",
    ],
)

POINT_SCHEMA = _closed(
    {
        "entity_id": _ID,
        "type_id": _ID,
        "x": _NUMBER,
        "y": _NUMBER,
        "shape": {"enum": ["circle", "square", "triangle"]},
        "color": _COLOR,
        "size": {"type": "number", "exclusiveMinimum": 0, "maximum": 20},
        "category": {"oneOf": [{"type": "string", "maxLength": 40}, {"type": "null"}]},
    },
    ["entity_id", "type_id", "x", "y", "shape", "color", "size", "category"],
)

THOUGHT_EXPERIMENT_RUN_SCHEMA = _closed(
    {
        "schema_version": {"const": RUN_VERSION},
        "run_id": {"type": "string", "pattern": "^visual-run-[0-9a-f]{24}$"},
        "spec_id": {"type": "string", "pattern": "^spec-[0-9a-f]{24}$"},
        "spec_sha256": _SHA,
        "status": {"enum": ["completed", "failed"]},
        "execution": _closed(
            {
                "engine_id": {"const": ENGINE_ID},
                "code_version": {"type": "string", "minLength": 1, "maxLength": 80},
                "modal_call_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "prediction": _closed(
                    {
                        "summary": {"type": "string", "minLength": 1, "maxLength": 500},
                        "expected_direction": {"enum": ["increase", "decrease", "change", "no_change"]},
                        "measurement_id": _ID,
                    },
                    ["summary", "expected_direction", "measurement_id"],
                ),
                "prediction_recorded_at": {"type": "string", "format": "date-time"},
                "started_at": {"type": "string", "format": "date-time"},
                "completed_at": {"type": "string", "format": "date-time"},
                "error": {"oneOf": [{"type": "string", "maxLength": 500}, {"type": "null"}]},
            },
            [
                "engine_id", "code_version", "modal_call_id", "prediction",
                "prediction_recorded_at", "started_at", "completed_at", "error",
            ],
        ),
        "conditions": {
            "type": "array", "minItems": 2, "maxItems": 4,
            "items": _closed(
                {
                    "condition_id": _ID,
                    "frames": {
                        "type": "array", "minItems": 2, "maxItems": 40,
                        "items": _closed(
                            {
                                "step": {"type": "integer", "minimum": 0, "maximum": 500},
                                "points": {"type": "array", "maxItems": 256, "items": POINT_SCHEMA},
                            },
                            ["step", "points"],
                        ),
                    },
                    "series": {
                        "type": "array", "minItems": 1, "maxItems": 12,
                        "items": _closed(
                            {
                                "measurement_id": _ID,
                                "values": {
                                    "type": "array", "minItems": 2, "maxItems": 501,
                                    "items": _closed({"step": {"type": "integer"}, "value": _NUMBER}, ["step", "value"]),
                                },
                            },
                            ["measurement_id", "values"],
                        ),
                    },
                    "summary": {
                        "type": "object", "minProperties": 1, "maxProperties": 12,
                        "propertyNames": _ID, "additionalProperties": _NUMBER,
                    },
                },
                ["condition_id", "frames", "series", "summary"],
            ),
        },
        "comparisons": {
            "type": "array", "minItems": 1, "maxItems": 36,
            "items": _closed(
                {
                    "measurement_id": _ID,
                    "baseline_condition_id": _ID,
                    "counterfactual_condition_id": _ID,
                    "baseline_final": _NUMBER,
                    "counterfactual_final": _NUMBER,
                    "difference": _NUMBER,
                },
                [
                    "measurement_id", "baseline_condition_id", "counterfactual_condition_id",
                    "baseline_final", "counterfactual_final", "difference",
                ],
            ),
        },
        "revision": _closed(
            {
                "disposition": {"enum": ["retain", "revise", "reject"]},
                "interpretation": {"type": "string", "minLength": 1, "maxLength": 500},
            },
            ["disposition", "interpretation"],
        ),
        "evidence": _closed(
            {
                "spec_sha256": _SHA,
                "engine_id": {"const": ENGINE_ID},
                "code_version": {"type": "string", "minLength": 1, "maxLength": 80},
                "modal_call_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "result_sha256": _SHA,
                "sealed_payload_sha256": _SHA,
            },
            ["spec_sha256", "engine_id", "code_version", "modal_call_id", "result_sha256", "sealed_payload_sha256"],
        ),
        "run_sha256": _SHA,
    },
    [
        "schema_version", "run_id", "spec_id", "spec_sha256", "status", "execution",
        "conditions", "comparisons", "revision", "evidence", "run_sha256",
    ],
)


class ThoughtExperimentContractError(ValueError):
    """Raised when declarative visual experiment data violates the closed contract."""


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ThoughtExperimentContractError(f"value is not canonical JSON: {exc}") from exc


def schema_sha256(schema: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(schema)).hexdigest()


EXPERIMENT_SPEC_SCHEMA_SHA256 = schema_sha256(EXPERIMENT_SPEC_SCHEMA)
THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256 = schema_sha256(THOUGHT_EXPERIMENT_RUN_SCHEMA)


def _content_hash(value: Mapping[str, Any], *excluded: str) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in value.items() if k not in excluded})).hexdigest()


def build_experiment_spec(**fields: Any) -> dict[str, Any]:
    if {"schema_version", "spec_id", "spec_sha256"} & fields.keys():
        raise ThoughtExperimentContractError("schema_version, spec_id, and spec_sha256 are derived")
    spec = {"schema_version": SPEC_VERSION, **deepcopy(fields)}
    digest = _content_hash(spec, "spec_id", "spec_sha256")
    spec["spec_id"] = f"spec-{digest[:24]}"
    spec["spec_sha256"] = digest
    return validate_experiment_spec(spec)


def validate_experiment_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = deepcopy(dict(value))
    _validate_schema(EXPERIMENT_SPEC_SCHEMA, spec, "ExperimentSpec")
    digest = _content_hash(spec, "spec_id", "spec_sha256")
    if spec["spec_sha256"] != digest or spec["spec_id"] != f"spec-{digest[:24]}":
        raise ThoughtExperimentContractError("ExperimentSpec content hash or spec_id does not match")
    _validate_spec_semantics(spec)
    return spec


def build_thought_experiment_run(spec: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    checked = validate_experiment_spec(spec)
    if {"schema_version", "run_id", "run_sha256"} & fields.keys():
        raise ThoughtExperimentContractError("schema_version, run_id, and run_sha256 are derived")
    run = {"schema_version": RUN_VERSION, **deepcopy(fields)}
    if run.get("spec_id") != checked["spec_id"] or run.get("spec_sha256") != checked["spec_sha256"]:
        raise ThoughtExperimentContractError("run does not bind the required ExperimentSpec")
    payload = {k: run[k] for k in ("spec_id", "spec_sha256", "status", "execution", "conditions", "comparisons", "revision")}
    result_sha = hashlib.sha256(canonical_json({"conditions": run["conditions"], "comparisons": run["comparisons"]})).hexdigest()
    sealed_sha = hashlib.sha256(canonical_json(payload)).hexdigest()
    evidence = run.get("evidence")
    if not isinstance(evidence, dict):
        raise ThoughtExperimentContractError("run evidence is required")
    evidence["result_sha256"] = result_sha
    evidence["sealed_payload_sha256"] = sealed_sha
    digest = _content_hash(run, "run_id", "run_sha256")
    run["run_id"] = f"visual-run-{digest[:24]}"
    run["run_sha256"] = digest
    return validate_thought_experiment_run(run, checked)


def validate_thought_experiment_run(value: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    run = deepcopy(dict(value))
    checked = validate_experiment_spec(spec)
    _validate_schema(THOUGHT_EXPERIMENT_RUN_SCHEMA, run, "ThoughtExperimentRun")
    digest = _content_hash(run, "run_id", "run_sha256")
    if run["run_sha256"] != digest or run["run_id"] != f"visual-run-{digest[:24]}":
        raise ThoughtExperimentContractError("ThoughtExperimentRun content hash or run_id does not match")
    if run["spec_id"] != checked["spec_id"] or run["spec_sha256"] != checked["spec_sha256"]:
        raise ThoughtExperimentContractError("run does not bind the supplied ExperimentSpec")
    evidence = run["evidence"]
    execution = run["execution"]
    for key in ("spec_sha256", "engine_id", "code_version", "modal_call_id"):
        expected = run["spec_sha256"] if key == "spec_sha256" else execution[key]
        if evidence[key] != expected:
            raise ThoughtExperimentContractError(f"evidence.{key} does not bind execution")
    payload = {k: run[k] for k in ("spec_id", "spec_sha256", "status", "execution", "conditions", "comparisons", "revision")}
    if evidence["result_sha256"] != hashlib.sha256(canonical_json({"conditions": run["conditions"], "comparisons": run["comparisons"]})).hexdigest():
        raise ThoughtExperimentContractError("result_sha256 does not bind visual results")
    if evidence["sealed_payload_sha256"] != hashlib.sha256(canonical_json(payload)).hexdigest():
        raise ThoughtExperimentContractError("sealed_payload_sha256 does not bind the run payload")
    _validate_run_semantics(run, checked)
    return run


def _validate_spec_semantics(spec: dict[str, Any]) -> None:
    entities = spec["world"]["entities"]
    entity_ids = _unique((item["id"] for item in entities), "entity type")
    if sum(item["count"] for item in entities) > 64:
        raise ThoughtExperimentContractError("total entity count exceeds 64")
    rules = spec["dynamics"]["rules"]
    rule_ids = _unique((item["id"] for item in rules), "rule")
    entity_by_id = {item["id"]: item for item in entities}
    rule_by_id = {item["id"]: item for item in rules}
    for rule in rules:
        if set(rule["parameters"]) != RULE_OPS[rule["op"]]:
            raise ThoughtExperimentContractError(f"parameters for {rule['op']} do not match the allowlist")
        if rule["target_type"] is not None and rule["target_type"] not in entity_ids:
            raise ThoughtExperimentContractError("rule target_type is unknown")
        _finite_tree(rule["parameters"], "rule parameters")
        _validate_rule_semantics(rule, entity_by_id, spec["world"])
    conditions = spec["conditions"]
    condition_ids = _unique((item["id"] for item in conditions), "condition")
    baselines = [item for item in conditions if item["kind"] == "baseline"]
    counterfactuals = [item for item in conditions if item["kind"] == "counterfactual"]
    if len(baselines) != 1 or baselines[0]["interventions"] or not counterfactuals:
        raise ThoughtExperimentContractError("conditions require one empty baseline and at least one counterfactual")
    duration = spec["schedule"]["duration_steps"]
    for condition in counterfactuals:
        if not condition["interventions"]:
            raise ThoughtExperimentContractError("each counterfactual requires an intervention")
        for change in condition["interventions"]:
            if change["time"] > duration:
                raise ThoughtExperimentContractError("intervention time exceeds duration")
            operation, target, field, value = change["operation"], change["target"], change["field"], change["value"]
            if operation in {"set_rule_parameter", "scale_rule_parameter"}:
                if target not in rule_ids or field not in rule_by_id[target]["parameters"] or not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ThoughtExperimentContractError("rule intervention target/value is invalid")
            elif target not in entity_ids:
                raise ThoughtExperimentContractError("state intervention target is unknown")
            elif operation == "set_numeric_state" and (field not in entity_by_id[target]["initial_state"]["numeric"] or not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise ThoughtExperimentContractError("numeric-state intervention is invalid")
            elif operation == "set_categorical_state" and (field not in entity_by_id[target]["initial_state"]["categorical"] or not isinstance(value, str)):
                raise ThoughtExperimentContractError("categorical-state intervention is invalid")
    measurement_ids = _unique((item["id"] for item in spec["measurements"]), "measurement")
    for measurement in spec["measurements"]:
        entity_type = measurement["entity_type"]
        if entity_type is not None and entity_type not in entity_ids:
            raise ThoughtExperimentContractError("measurement entity_type is unknown")
        if measurement["op"] in {"mean_state", "sum_state", "variance_state"}:
            if entity_type is None or measurement["state"] not in entity_by_id[entity_type]["initial_state"]["numeric"] or measurement["category"] is not None:
                raise ThoughtExperimentContractError("numeric measurement reference is invalid")
        elif measurement["op"] == "count_category":
            if entity_type is None or measurement["state"] not in entity_by_id[entity_type]["initial_state"]["categorical"] or measurement["category"] is None:
                raise ThoughtExperimentContractError("categorical measurement reference is invalid")
        elif measurement["state"] is not None or measurement["category"] is not None:
            raise ThoughtExperimentContractError("population_count cannot name state/category")
    charts = spec["visualization"]["chart_measurement_ids"]
    if len(charts) != len(set(charts)) or not set(charts).issubset(measurement_ids):
        raise ThoughtExperimentContractError("visualization chart references are invalid")
    stride = spec["visualization"]["frame_stride"]
    required_frames = (duration + stride - 1) // stride + 1
    if required_frames > spec["visualization"]["max_frames"]:
        raise ThoughtExperimentContractError("frame_stride/max_frames cannot represent the requested duration")
    if len(condition_ids) * sum(item["count"] for item in entities) * required_frames > 10000:
        raise ThoughtExperimentContractError("visual result exceeds the fixed point budget")


def _validate_run_semantics(run: dict[str, Any], spec: dict[str, Any]) -> None:
    if run["status"] != "completed" or run["execution"]["error"] is not None:
        raise ThoughtExperimentContractError("v2 currently accepts completed visual runs only")
    if run["execution"]["prediction"]["measurement_id"] not in {item["id"] for item in spec["measurements"]}:
        raise ThoughtExperimentContractError("prediction measurement is not declared")
    if run["execution"]["prediction_recorded_at"] >= run["execution"]["started_at"]:
        raise ThoughtExperimentContractError("prediction must be recorded before execution")
    expected_conditions = {item["id"] for item in spec["conditions"]}
    if {item["condition_id"] for item in run["conditions"]} != expected_conditions or len(run["conditions"]) != len(expected_conditions):
        raise ThoughtExperimentContractError("visual results must cover each condition exactly once")
    measurement_ids = {item["id"] for item in spec["measurements"]}
    for result in run["conditions"]:
        if {item["measurement_id"] for item in result["series"]} != measurement_ids or set(result["summary"]) != measurement_ids:
            raise ThoughtExperimentContractError("condition result does not cover every measurement")
        if result["frames"][0]["step"] != 0 or result["frames"][-1]["step"] != spec["schedule"]["duration_steps"]:
            raise ThoughtExperimentContractError("frame sequence must include initial and final state")
    baseline_id = next(item["id"] for item in spec["conditions"] if item["kind"] == "baseline")
    counterfactual_ids = {item["id"] for item in spec["conditions"] if item["kind"] == "counterfactual"}
    results = {item["condition_id"]: item for item in run["conditions"]}
    expected = {(measurement, counterfactual) for measurement in measurement_ids for counterfactual in counterfactual_ids}
    seen: set[tuple[str, str]] = set()
    for comparison in run["comparisons"]:
        key = (comparison["measurement_id"], comparison["counterfactual_condition_id"])
        if key in seen or key not in expected or comparison["baseline_condition_id"] != baseline_id:
            raise ThoughtExperimentContractError("comparison references are invalid or duplicated")
        seen.add(key)
        baseline = results[baseline_id]["summary"][comparison["measurement_id"]]
        counterfactual = results[key[1]]["summary"][comparison["measurement_id"]]
        difference = float(Fraction(str(counterfactual)) - Fraction(str(baseline)))
        if comparison["baseline_final"] != baseline or comparison["counterfactual_final"] != counterfactual or comparison["difference"] != difference:
            raise ThoughtExperimentContractError("comparison must be recomputed from condition summaries")
    if seen != expected:
        raise ThoughtExperimentContractError("comparisons must cover each measurement/counterfactual pair")


def _validate_rule_semantics(rule: dict[str, Any], entities: dict[str, dict[str, Any]], world: dict[str, Any]) -> None:
    op, target, parameters = rule["op"], rule["target_type"], rule["parameters"]
    numeric = lambda name: isinstance(parameters[name], (int, float)) and not isinstance(parameters[name], bool)
    probability = lambda name: numeric(name) and 0 <= parameters[name] <= 1
    if op == "move_2d":
        if target is None or not {"vx", "vy"}.issubset(entities[target]["initial_state"]["numeric"]):
            raise ThoughtExperimentContractError("move_2d requires target numeric vx and vy states")
        if not probability("damping") or not numeric("max_speed") or not 0 <= parameters["max_speed"] <= 100:
            raise ThoughtExperimentContractError("move_2d parameters are outside fixed bounds")
    elif op == "random_walk_2d":
        if target is None or not numeric("step_scale") or not 0 <= parameters["step_scale"] <= 100:
            raise ThoughtExperimentContractError("random_walk_2d parameters are invalid")
    elif op == "pairwise_force_2d":
        selected = entities.values() if target is None else [entities[target]]
        if any(not {"vx", "vy"}.issubset(item["initial_state"]["numeric"]) for item in selected):
            raise ThoughtExperimentContractError("pairwise_force_2d requires numeric vx and vy states")
        if not all(numeric(name) for name in ("strength", "exponent", "softening")) or not -1000 <= parameters["strength"] <= 1000 or not 0 <= parameters["exponent"] <= 4 or not 0 < parameters["softening"] <= 100:
            raise ThoughtExperimentContractError("pairwise_force_2d parameters are outside fixed bounds")
    elif op == "graph_diffusion":
        state = parameters["state"]
        if target is not None or world["graph"]["kind"] == "none" or world["graph"]["directed"] or not isinstance(state, str) or any(state not in item["initial_state"]["numeric"] for item in entities.values()):
            raise ThoughtExperimentContractError("graph_diffusion requires one undirected graph and a shared numeric state")
        if not numeric("rate") or not 0 <= parameters["rate"] <= 0.5:
            raise ThoughtExperimentContractError("graph_diffusion rate is outside the stable bound")
    elif op == "graph_contagion":
        state = parameters["state"]
        labels = (parameters["susceptible"], parameters["infected"], parameters["recovered"])
        if target is not None or world["graph"]["kind"] == "none" or world["graph"]["directed"] or not all(isinstance(item, str) and item for item in (state, *labels)) or any(state not in item["initial_state"]["categorical"] for item in entities.values()):
            raise ThoughtExperimentContractError("graph_contagion requires one undirected graph and a shared categorical state")
        if not probability("transmission_probability") or not probability("recovery_probability"):
            raise ThoughtExperimentContractError("graph_contagion probabilities must lie in [0,1]")
    elif op == "predator_prey_2d":
        if target is not None or parameters["prey_type"] not in entities or parameters["predator_type"] not in entities or parameters["prey_type"] == parameters["predator_type"]:
            raise ThoughtExperimentContractError("predator_prey_2d type references are invalid")
        if not all(probability(name) for name in ("prey_growth", "predation_rate", "predator_efficiency", "predator_decay")) or not numeric("interaction_radius") or not 0 < parameters["interaction_radius"] <= 1000:
            raise ThoughtExperimentContractError("predator_prey_2d parameters are outside fixed bounds")
    elif op == "lane_traffic_2d":
        state = parameters["speed_state"]
        if target is None or not isinstance(state, str) or state not in entities[target]["initial_state"]["numeric"]:
            raise ThoughtExperimentContractError("lane_traffic_2d requires a target numeric speed state")
        if not all(numeric(name) for name in ("desired_speed", "headway", "road_y")) or not 0 <= parameters["desired_speed"] <= 100 or not 0 <= parameters["headway"] <= world["bounds"]["width"] or not 0 <= parameters["road_y"] <= world["bounds"]["height"]:
            raise ThoughtExperimentContractError("lane_traffic_2d parameters are outside world bounds")
    elif op == "queue_agents_2d":
        if target is None or entities[target]["initial_state"]["categorical"].get("queue_status") not in {"waiting", "queued", "served"}:
            raise ThoughtExperimentContractError("queue_agents_2d requires categorical queue_status")
        if not probability("arrival_probability") or not numeric("service_capacity") or int(parameters["service_capacity"]) != parameters["service_capacity"] or not 0 <= parameters["service_capacity"] <= 32 or not numeric("queue_x") or not numeric("service_x"):
            raise ThoughtExperimentContractError("queue_agents_2d parameters are outside fixed bounds")


def _unique(values: Any, label: str) -> set[str]:
    items = list(values)
    if len(items) != len(set(items)):
        raise ThoughtExperimentContractError(f"{label} IDs must be unique")
    return set(items)


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ThoughtExperimentContractError(f"{label} must be finite")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item, label)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item, label)


def _validate_schema(schema: Mapping[str, Any], value: Any, label: str) -> None:
    errors = sorted(Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        error = errors[0]
        where = ".".join(str(part) for part in error.path) or "<root>"
        raise ThoughtExperimentContractError(f"{label} {where}: {error.message}")


__all__ = [
    "QUESTION_VERSION", "SPEC_VERSION", "CONFIRMATION_VERSION", "SPEC_RESPONSE_VERSION",
    "RUN_VERSION", "RUN_RESPONSE_VERSION", "ENGINE_ID", "RULE_OPS",
    "EXPERIMENT_SPEC_SCHEMA", "EXPERIMENT_SPEC_SCHEMA_SHA256",
    "THOUGHT_EXPERIMENT_RUN_SCHEMA", "THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256",
    "ThoughtExperimentContractError", "canonical_json", "build_experiment_spec",
    "validate_experiment_spec", "build_thought_experiment_run", "validate_thought_experiment_run",
]
