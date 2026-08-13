"""Bounded free-text planner for deterministic six-object experiments.

User intent is classified as inert data.  It is never returned in the compiled
specification and is never concatenated into a model or system prompt.  The
only executable output is a closed, validated experiment description whose
templates and limits are owned by this module.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
import re
import unicodedata
from typing import Any

from jump_contracts import open_result_envelope, validate_learned_latent_evidence

from .authentic import independent_law, independent_partition, matched_world_pair
from .canonical import sha256_json
from .simulator import EpisodeSpec, SimulatorConfig, generate_episode


INTENT_SCHEMA_VERSION = "jump.experiment-intent/v1"
EXPERIMENT_SPEC_VERSION = "jump.experiment-spec/v1"
RUN_SCHEMA_VERSION = "jump.experiment-run/v1"
COMPILED_EXPERIMENT_VERSION = "jump.compiled-six-object-experiment/v1"

INTENT_FIELDS = frozenset({"schema_version", "intent", "session_id", "seed", "max_steps"})
SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "normalized_intent_sha256",
        "summary",
        "template_id",
        "world_count",
        "seed",
        "observed_steps",
        "prediction_horizon",
        "intervention",
        "requested_outputs",
        "limits",
        "claim_label",
    }
)
RUN_FIELDS = frozenset({"schema_version", "status", "live", "request_id", "plan", "result", "error"})
RESULT_FIELDS = frozenset({"sealed_result", "decoded_image", "presentation"})
DECODED_IMAGE_FIELDS = frozenset({"artifact_name", "media_type", "encoding", "data", "sha256"})
PRESENTATION_FIELDS = frozenset({"world_built", "model_prediction", "what_changed", "correctness"})
CORRECTNESS_FIELDS = frozenset(
    {
        "format_valid",
        "exact_correct",
        "partition_correct",
        "law_correct",
        "adequacy_correct",
        "force_score",
        "notes",
    }
)
INTERVENTION_FIELDS = frozenset({"kind", "recipient_world_id", "donor_world_id"})
LIMIT_FIELDS = frozenset(
    {
        "max_intent_chars",
        "max_worlds",
        "max_objects_per_world",
        "max_observed_steps",
        "max_prediction_horizon",
    }
)

MAX_INTENT_CHARS = 600
MAX_OBSERVED_STEPS = 32
MAX_PREDICTION_HORIZON = 1
MAX_WORLDS = 2
OBJECT_COUNT = 6
OUTPUT_ORDER = ("world", "model_prediction", "change_summary", "score", "evidence")
TEMPLATES = (
    "future-prediction",
    "hidden-law-discovery",
    "falsified-prior",
    "world-swap",
)
_CLASSIFICATION_PRIORITY = (
    "world-swap",
    "falsified-prior",
    "hidden-law-discovery",
    "future-prediction",
)
INTERVENTIONS = ("none", "change_force_law", "swap_hidden_types", "swap_learned_latent")
CLAIM_LABEL = "bounded synthetic Track H engineering experiment; not mechanistic or causal evidence"
CONTRACT_SCHEMA_VERSION = "jump.experiment-spec-contract/v1"

_SUMMARY = {
    "future-prediction": "Predict the next visible state of one deterministic six-object world.",
    "hidden-law-discovery": "Evaluate a structured hidden-law answer in one deterministic six-object world.",
    "falsified-prior": "Evaluate a deliberately incorrect prior law against one deterministic six-object world.",
    "world-swap": "Compare both directions of a matched two-world hidden-state intervention.",
}
_KEYWORDS = {
    "world-swap": frozenset({"swap", "donor", "exchange", "counterfactual"}),
    "falsified-prior": frozenset({"prior", "falsify", "falsified", "wrong", "incorrect", "adequacy"}),
    "hidden-law-discovery": frozenset({"law", "hidden", "interaction", "attract", "repel", "force"}),
    "future-prediction": frozenset({"predict", "future", "trajectory", "motion", "position", "next"}),
}
_URI = re.compile(r"(?i)(?:[a-z][a-z0-9+.-]*://|www\.)")
_PATH = re.compile(r"[/\\]")
_CODE = re.compile(
    r"(?i)(?:```|<script\b|</script>|\b(?:eval|exec|subprocess|system)\s*\(|"
    r"\bimport\s+[A-Za-z_]|\bfrom\s+[A-Za-z_]\S*\s+import\b|__\w+__|\$\(|&&|\|\||[;{}])"
)
_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_WORD = re.compile(r"[a-z]+")


def _uint32(value: Any, name: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be a uint32" + (" or null" if nullable else ""))
    return value


def _positive_int(value: Any, name: str, maximum: int, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]" + (" or null" if nullable else ""))
    return value


def _normalized_intent(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("intent must be a string")
    intent = unicodedata.normalize("NFKC", value).strip()
    if not 1 <= len(intent) <= MAX_INTENT_CHARS:
        raise ValueError(f"intent must contain 1..{MAX_INTENT_CHARS} characters after trim")
    if any(unicodedata.category(char).startswith("C") for char in intent):
        raise ValueError("intent must not contain control or formatting characters")
    if "\n" in intent or "\r" in intent:
        raise ValueError("intent must be one line")
    if _URI.search(intent):
        raise ValueError("intent must not contain URLs or URI schemes")
    if _PATH.search(intent):
        raise ValueError("intent must not contain file or path syntax")
    if _CODE.search(intent):
        raise ValueError("intent must not contain code or shell markers")
    return intent


def validate_experiment_intent(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != INTENT_FIELDS:
        raise ValueError(f"experiment intent must contain exactly {sorted(INTENT_FIELDS)}")
    if value["schema_version"] != INTENT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {INTENT_SCHEMA_VERSION}")
    intent = _normalized_intent(value["intent"])
    session_id = value["session_id"]
    if not isinstance(session_id, str) or _SESSION.fullmatch(session_id) is None:
        raise ValueError("session_id must be 1..64 safe identifier characters")
    return {
        "schema_version": INTENT_SCHEMA_VERSION,
        "intent": intent,
        "session_id": session_id,
        "seed": _uint32(value["seed"], "seed", nullable=True),
        "max_steps": _positive_int(value["max_steps"], "max_steps", MAX_OBSERVED_STEPS, nullable=True),
    }


def _classify(intent: str) -> tuple[str, frozenset[str]]:
    words = frozenset(_WORD.findall(intent.casefold()))
    matches = {template: words & terms for template, terms in _KEYWORDS.items()}
    # Swap and prior modification are mutually exclusive operations.  Other
    # overlap is natural language (for example, "wrong force law") and is
    # resolved by this frozen most-specific-to-general priority.
    if matches["world-swap"] and matches["falsified-prior"]:
        raise ValueError("intent ambiguously requests both a world swap and a prior-law change")
    for template in _CLASSIFICATION_PRIORITY:
        if matches[template]:
            return template, frozenset(matches[template])
    return "future-prediction", frozenset()


def _derived_seed(intent_hash: str, session_id: str) -> int:
    content = f"jump-experiment-spec-v1:{intent_hash}:{session_id}".encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:4], "big")


def _world_ids(template: str, seed: int, observed_steps: int, intervention_kind: str) -> tuple[str | None, str | None]:
    if template == "world-swap":
        pair = matched_world_pair(pair_seed=seed, config=SimulatorConfig(steps=observed_steps + 1))
        return pair["a"]["episode_id"], pair["b"]["episode_id"]
    if intervention_kind == "change_force_law":
        episode = generate_episode(
            EpisodeSpec(
                seed=seed,
                split="experiment",
                law=independent_law(seed),
                adequate_prior=False,
                config=SimulatorConfig(steps=observed_steps + 1),
                partition=independent_partition(seed),
            )
        )
        return episode["episode_id"], None
    return None, None


def _intervention(template: str, intent: str) -> str:
    if template == "falsified-prior":
        return "change_force_law"
    if template == "world-swap":
        return "swap_learned_latent" if "latent" in _WORD.findall(intent.casefold()) else "swap_hidden_types"
    return "none"


def _limits() -> dict[str, int]:
    return {
        "max_intent_chars": MAX_INTENT_CHARS,
        "max_worlds": MAX_WORLDS,
        "max_objects_per_world": OBJECT_COUNT,
        "max_observed_steps": MAX_OBSERVED_STEPS,
        "max_prediction_horizon": MAX_PREDICTION_HORIZON,
    }


def _experiment_id(spec_without_id: dict[str, Any]) -> str:
    return "exp-" + sha256_json(spec_without_id)[:24]


def compile_experiment_intent(value: Any) -> dict[str, Any]:
    """Compile inert free text into one closed, validated experiment spec."""
    request = validate_experiment_intent(value)
    intent = request["intent"]
    intent_hash = hashlib.sha256(intent.encode("utf-8")).hexdigest()
    template, _ = _classify(intent)
    seed = request["seed"] if request["seed"] is not None else _derived_seed(intent_hash, request["session_id"])
    observed_steps = request["max_steps"] if request["max_steps"] is not None else 4
    if observed_steps < 4:
        raise ValueError("the observation-only learned-z model requires at least four observed steps")
    kind = _intervention(template, intent)
    recipient, donor = _world_ids(template, seed, observed_steps, kind)
    spec: dict[str, Any] = {
        "schema_version": EXPERIMENT_SPEC_VERSION,
        "normalized_intent_sha256": intent_hash,
        "summary": _SUMMARY[template],
        "template_id": template,
        "world_count": OBJECT_COUNT,
        "seed": seed,
        "observed_steps": observed_steps,
        "prediction_horizon": 1,
        "intervention": {
            "kind": kind,
            "recipient_world_id": recipient,
            "donor_world_id": donor,
        },
        "requested_outputs": list(OUTPUT_ORDER),
        "limits": _limits(),
        "claim_label": CLAIM_LABEL,
    }
    spec["experiment_id"] = _experiment_id(spec)
    return validate_experiment_spec(spec)


def validate_experiment_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != SPEC_FIELDS:
        raise ValueError(f"experiment spec must contain exactly {sorted(SPEC_FIELDS)}")
    spec = dict(value)
    if spec["schema_version"] != EXPERIMENT_SPEC_VERSION:
        raise ValueError(f"schema_version must be {EXPERIMENT_SPEC_VERSION}")
    digest = spec["normalized_intent_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("normalized_intent_sha256 must be a lowercase SHA-256")
    template = spec["template_id"]
    if template not in TEMPLATES or spec["summary"] != _SUMMARY.get(template):
        raise ValueError("template_id and summary must be server-owned allowlisted values")
    if spec["world_count"] != OBJECT_COUNT:
        raise ValueError("world_count must be exactly six")
    _uint32(spec["seed"], "seed")
    _positive_int(spec["observed_steps"], "observed_steps", MAX_OBSERVED_STEPS)
    _positive_int(spec["prediction_horizon"], "prediction_horizon", MAX_PREDICTION_HORIZON)
    if spec["limits"] != _limits():
        raise ValueError("limits must equal the server-owned ExperimentSpec v1 limits")
    outputs = spec["requested_outputs"]
    if not isinstance(outputs, list) or not outputs or len(outputs) != len(set(outputs)):
        raise ValueError("requested_outputs must be a nonempty unique array")
    if outputs != [name for name in OUTPUT_ORDER if name in outputs]:
        raise ValueError("requested_outputs must be an ordered allowlisted subset")
    if spec["claim_label"] != CLAIM_LABEL:
        raise ValueError("claim_label must be the bounded server-owned label")
    intervention = spec["intervention"]
    if not isinstance(intervention, dict) or set(intervention) != INTERVENTION_FIELDS:
        raise ValueError(f"intervention must contain exactly {sorted(INTERVENTION_FIELDS)}")
    kind = intervention["kind"]
    recipient, donor = intervention["recipient_world_id"], intervention["donor_world_id"]
    if kind not in INTERVENTIONS:
        raise ValueError("intervention kind is outside the allowlist")
    for name, item in (("recipient_world_id", recipient), ("donor_world_id", donor)):
        if item is not None and (not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{20}", item) is None):
            raise ValueError(f"{name} must be null or a generated world ID")
    if kind == "none" and (recipient is not None or donor is not None):
        raise ValueError("none intervention requires null world IDs")
    if kind == "change_force_law" and (recipient is None or donor is not None):
        raise ValueError("change_force_law requires a recipient and no donor")
    if kind.startswith("swap_") and (recipient is None or donor is None or recipient == donor):
        raise ValueError("swap intervention requires distinct recipient and donor worlds")
    if template == "world-swap" and not kind.startswith("swap_"):
        raise ValueError("world-swap template requires a swap intervention")
    if template != "world-swap" and kind.startswith("swap_"):
        raise ValueError("swap intervention requires the world-swap template")
    identifier = spec.pop("experiment_id")
    if identifier != _experiment_id(spec):
        raise ValueError("experiment_id does not bind the normalized server-owned plan")
    spec["experiment_id"] = identifier
    return spec


def materialize_experiment(spec: Any) -> dict[str, Any]:
    """Generate bounded backend worlds; sealed targets never enter the UI plan."""
    plan = validate_experiment_spec(spec)
    steps = plan["observed_steps"] + 1
    if plan["template_id"] == "world-swap":
        pair = matched_world_pair(pair_seed=plan["seed"], config=SimulatorConfig(steps=steps))
        worlds = [pair["a"], pair["b"]]
        if [worlds[0]["episode_id"], worlds[1]["episode_id"]] != [
            plan["intervention"]["recipient_world_id"],
            plan["intervention"]["donor_world_id"],
        ]:
            raise RuntimeError("compiled swap world IDs changed during materialization")
    else:
        episode = generate_episode(
            EpisodeSpec(
                seed=plan["seed"],
                split="experiment",
                law=independent_law(plan["seed"]),
                adequate_prior=plan["template_id"] != "falsified-prior",
                config=SimulatorConfig(steps=steps),
                partition=independent_partition(plan["seed"]),
            )
        )
        worlds = [episode]
        recipient = plan["intervention"]["recipient_world_id"]
        if recipient is not None and recipient != episode["episode_id"]:
            raise RuntimeError("compiled recipient world ID changed during materialization")
    def object_count(world: dict[str, Any]) -> int | None:
        if "object_count" in world:
            return world["object_count"]
        shape = world.get("encoder_input", {}).get("shape")
        return shape[1] if isinstance(shape, list) and len(shape) == 3 else None

    if any(object_count(world) != OBJECT_COUNT for world in worlds):
        raise RuntimeError("planner materialized a non-six-object world")
    return {
        "schema_version": COMPILED_EXPERIMENT_VERSION,
        "experiment_spec_sha256": sha256_json(plan),
        "experiment_id": plan["experiment_id"],
        "template_id": plan["template_id"],
        "worlds": worlds,
    }


def build_planned_run(*, request_id: str, plan: Any) -> dict[str, Any]:
    if not isinstance(request_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", request_id) is None:
        raise ValueError("request_id must be a safe 1..64 character identifier")
    return validate_experiment_run(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "planned",
            "live": True,
            "request_id": request_id,
            "plan": validate_experiment_spec(plan),
            "result": None,
            "error": None,
        }
    )


def _validated_completed_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS:
        raise ValueError(f"completed result must contain exactly {sorted(RESULT_FIELDS)}")
    payload = open_result_envelope(value["sealed_result"], expected_source="live")
    evidence = validate_learned_latent_evidence(payload)
    image = value["decoded_image"]
    if not isinstance(image, dict) or set(image) != DECODED_IMAGE_FIELDS:
        raise ValueError(f"decoded_image must contain exactly {sorted(DECODED_IMAGE_FIELDS)}")
    if image["artifact_name"] != evidence["decoded_observation"].get("artifact_name", "predicted-from-z.svg"):
        # v1 learned-latent evidence names the image through transport rather
        # than its payload.  The one allowed transport name stays fixed.
        if image["artifact_name"] != "predicted-from-z.svg":
            raise ValueError("decoded_image artifact_name must be predicted-from-z.svg")
    if image["media_type"] != evidence["decoded_observation"]["media_type"] or image["media_type"] != "image/svg+xml":
        raise ValueError("decoded_image must be the learned decoder SVG")
    if image["encoding"] != "base64" or not isinstance(image["data"], str):
        raise ValueError("decoded_image must use base64 transport")
    try:
        raw_image = base64.b64decode(image["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("decoded_image data is not valid base64") from exc
    expected_image_sha = evidence["decoded_observation"]["image_sha256"]
    if image["sha256"] != expected_image_sha or hashlib.sha256(raw_image).hexdigest() != expected_image_sha:
        raise ValueError("decoded_image bytes do not match the sealed learned decoder output")
    presentation = value["presentation"]
    if not isinstance(presentation, dict) or set(presentation) != PRESENTATION_FIELDS:
        raise ValueError(f"presentation must contain exactly {sorted(PRESENTATION_FIELDS)}")
    for field in ("world_built", "what_changed"):
        text = presentation[field]
        if not isinstance(text, str) or not 1 <= len(text) <= 600:
            raise ValueError(f"presentation {field} must be a bounded nonempty string")
    if presentation["model_prediction"] != evidence["answer"]:
        raise ValueError("presentation model_prediction must equal the sealed structured answer")
    correctness = presentation["correctness"]
    if not isinstance(correctness, dict) or set(correctness) != CORRECTNESS_FIELDS:
        raise ValueError(f"correctness must contain exactly {sorted(CORRECTNESS_FIELDS)}")
    for field in ("format_valid", "exact_correct", "partition_correct", "law_correct", "adequacy_correct"):
        if not isinstance(correctness[field], bool):
            raise ValueError(f"correctness {field} must be Boolean")
    force_score = correctness["force_score"]
    if force_score is not None and (
        isinstance(force_score, bool) or not isinstance(force_score, (int, float)) or not math.isfinite(force_score)
    ):
        raise ValueError("correctness force_score must be finite or null")
    if not isinstance(correctness["notes"], str) or len(correctness["notes"]) > 600:
        raise ValueError("correctness notes must be a bounded string")
    return dict(value)


def validate_experiment_run(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUN_FIELDS:
        raise ValueError(f"experiment run must contain exactly {sorted(RUN_FIELDS)}")
    if value["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {RUN_SCHEMA_VERSION}")
    if value["status"] not in {"planned", "running", "completed", "failed"}:
        raise ValueError("status is outside the ExperimentSpec v1 run allowlist")
    if value["live"] is not True:
        raise ValueError("accepted experiment runs must declare live=true")
    if not isinstance(value["request_id"], str) or _SESSION.fullmatch(value["request_id"]) is None:
        raise ValueError("request_id must be a safe 1..64 character identifier")
    validate_experiment_spec(value["plan"])
    if value["status"] in {"planned", "running"} and (value["result"] is not None or value["error"] is not None):
        raise ValueError("planned/running responses cannot contain a result or error")
    if value["status"] == "completed":
        if value["error"] is not None:
            raise ValueError("completed responses require a null error")
        _validated_completed_result(value["result"])
    if value["status"] == "failed" and (value["result"] is not None or not isinstance(value["error"], str) or not value["error"]):
        raise ValueError("failed responses require a nonempty error and null result")
    return dict(value)


def experiment_spec_contract() -> dict[str, Any]:
    """Return the exact shared planner/execution descriptor hashed by v1."""
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "canonicalization": "jump_benchmark.canonical.sha256_json: sorted compact UTF-8 JSON plus newline",
        "intent": {
            "schema_version": INTENT_SCHEMA_VERSION,
            "exact_keys": sorted(INTENT_FIELDS),
            "intent_chars": [1, MAX_INTENT_CHARS],
            "session_id_chars": [1, 64],
            "seed": "null_or_uint32",
            "max_steps": [1, MAX_OBSERVED_STEPS],
            "learned_z_min_observed_steps": 4,
            "rejected": ["control_or_format_characters", "urls_or_uri_schemes", "file_or_path_syntax", "code_or_shell_markers"],
            "handling": "inert planner data; raw intent absent from compiled spec and all model/system prompts",
        },
        "plan": {
            "schema_version": EXPERIMENT_SPEC_VERSION,
            "exact_keys": sorted(SPEC_FIELDS),
            "template_ids": list(TEMPLATES),
            "classification_priority": list(_CLASSIFICATION_PRIORITY),
            "classification_keywords": {
                template: sorted(_KEYWORDS[template]) for template in sorted(_KEYWORDS)
            },
            "default_template": "future-prediction",
            "world_count": OBJECT_COUNT,
            "intervention_exact_keys": sorted(INTERVENTION_FIELDS),
            "intervention_kinds": list(INTERVENTIONS),
            "requested_output_order": list(OUTPUT_ORDER),
            "limits_exact_keys": sorted(LIMIT_FIELDS),
            "limits": _limits(),
            "confirmation_rows": {
                "World": ["template_id", "world_count", "seed"],
                "Observe": ["observed_steps"],
                "Change": ["intervention"],
                "Predict": ["prediction_horizon", "requested_outputs"],
            },
        },
        "run": {
            "schema_version": RUN_SCHEMA_VERSION,
            "exact_keys": sorted(RUN_FIELDS),
            "statuses": ["planned", "running", "completed", "failed"],
            "accepted_live": True,
            "completed_result_exact_keys": sorted(RESULT_FIELDS),
            "decoded_image_exact_keys": sorted(DECODED_IMAGE_FIELDS),
            "presentation_exact_keys": sorted(PRESENTATION_FIELDS),
            "correctness_exact_keys": sorted(CORRECTNESS_FIELDS),
            "format_valid_distinct_from_exact_correct": True,
            "sealed_result": "jump.sealed-result/v1 containing jump.learned-latent-evidence/v1",
        },
        "examples": [
            "Predict where the six objects move next.",
            "Discover the hidden interaction law from the observed motion.",
            "Test whether the stated prior force law is wrong.",
            "Swap the learned latent between two matched worlds.",
        ],
    }


EXPERIMENT_SPEC_CONTRACT_SHA256 = sha256_json(experiment_spec_contract())
