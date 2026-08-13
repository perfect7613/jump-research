"""Narrow real-Gemma causal tracing over verified Stage D latent arms.

The GPU/model loader remains owned by :mod:`jump_runner.stage_d_live`.  This
module accepts that verified runtime, refuses architecture or identity drift,
and runs the frozen C/D causal-tracing protocol.  It does not train, download,
or launch compute.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jump_contracts import (
    canonical_json,
    open_result_envelope,
    tensor_bytes_sha256,
    validate_learned_latent_evidence,
    validate_world_model_component_manifest,
    validate_world_model_load_record,
    verify_latent_tensor_bytes,
    verify_world_model_component_files,
)

from jump_benchmark.authentic import matched_world_pair
from jump_benchmark.authentic_stage_d import (
    BASE_REPO_ID,
    BASE_REVISION,
    ExactAnswerGrammar,
    TRANSFORMERS_REVISION,
    matched_injection_prompt,
    persistent_z_injection,
    prompt_tensors,
    _stage_d_target,
)
from jump_benchmark.canonical import sha256_json
from jump_benchmark.task_adapter import write_track_h_task_evidence
from jump_benchmark.simulator import derive_seed

from .scoring import score_episode
from .stage_d import (
    LATENT_PERMUTATION_VERSION,
    STAGE_D_CONTROL_VERSION,
    STAGE_D_EXECUTION_CONTRACT_SHA256,
)
from .metrics import paired_effect

REAL_GEMMA_TRACE_VERSION = "jump.real-gemma-causal-trace/v1"
REAL_GEMMA_RECORD_VERSION = "jump.real-gemma-causal-record/v1"
REAL_GEMMA_GRID_VERSION = "jump.real-gemma-hook-grid/v1"
MODEL_CLASS = "Gemma4UnifiedForConditionalGeneration"
LAYER_CLASS = "Gemma4UnifiedTextDecoderLayer"
LAYER_PREFIX = "model.language_model.layers"
HIDDEN_SIZE = 3840
LAYER_COUNT = 48
LAYERS = (7, 15, 23, 31, 39, 47)
SITE_MODULE_TYPES = {
    "residual_pre": LAYER_CLASS,
    "attention_output": "Gemma4UnifiedTextAttention",
    "residual_mid": "Gemma4UnifiedRMSNorm",
    "mlp_output": "Gemma4UnifiedTextMLP",
    "residual_post": LAYER_CLASS,
}
SITES = (
    ("T0", "residual_pre"),
    ("T1", "attention_output"),
    ("T2", "residual_mid"),
    ("T3", "mlp_output"),
    ("T4", "residual_post"),
)
PHASE_COUNTS = {"exploratory": 32, "confirmatory_minimum": 200}
CONTROL_NAMES = ("zero_sham", "matched_norm_random", "orthogonal", "prompt_token")
_ARM_FIELDS = {
    "schema_version",
    "arm_id",
    "control_kind",
    "checkpoint_id",
    "manifest_sha256",
    "pair_id",
    "recipient_world_id",
    "source_world_id",
    "answer",
    "answer_sha256",
    "execution_contract_sha256",
    "injection",
}
_INJECTION_FIELDS = {
    "present",
    "learned_latent_envelope_sha256",
    "tensor_artifact_name",
    "dtype",
    "shape",
    "order",
    "raw_bytes_sha256",
    "world_latent_sha256",
    "permutation",
}


@dataclass(frozen=True)
class RealGemmaLatentArm:
    """Transport material for one already sealed Stage D latent arm."""

    control_envelope: Mapping[str, Any]
    learned_latent_envelope: Mapping[str, Any]
    source_tensor_bytes: bytes
    injected_tensor_bytes: bytes
    expected_source: str = "live"


@dataclass(frozen=True)
class RealGemmaCausalPair:
    pair_index: int
    pair_id: str
    own_z: RealGemmaLatentArm
    scrambled_z: RealGemmaLatentArm
    wrong_world_z: RealGemmaLatentArm
    target_answer: Mapping[str, Any]
    alternative_answer: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.pair_index, bool) or not isinstance(self.pair_index, int) or self.pair_index < 0:
            raise ValueError("causal pair_index must be a nonnegative integer")
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("causal pair_id must be nonempty")


@dataclass(frozen=True, order=True)
class RealGemmaNode:
    layer_index: int
    timepoint: str
    site: str

    @property
    def node_id(self) -> str:
        return f"L{self.layer_index}/{self.timepoint}:{self.site}"


def frozen_hook_grid() -> tuple[RealGemmaNode, ...]:
    return tuple(
        RealGemmaNode(layer, timepoint, site)
        for layer in LAYERS
        for timepoint, site in SITES
    )


def validate_real_gemma_runtime(
    runtime: Mapping[str, Any],
    *,
    checkpoint_id: str,
    stage_d_manifest_sha256: str,
    component_manifest_sha256: str,
    transformers_revision: str,
) -> dict[str, Any]:
    """Bind the supplied loaded runtime to the exact Gemma and component IDs."""
    required = {"model", "tokenizer", "encoder", "decoder", "projector", "manifest", "load_record", "root"}
    if not isinstance(runtime, Mapping) or set(runtime) != required:
        raise ValueError(f"real Gemma runtime must contain exactly {sorted(required)}")
    if not checkpoint_id or not _is_sha256(checkpoint_id.removeprefix("stage-d-")):
        raise ValueError("real Gemma checkpoint_id must be an immutable stage-d SHA identity")
    if transformers_revision != TRANSFORMERS_REVISION:
        raise ValueError("Transformers source revision drifted")
    manifest = validate_world_model_component_manifest(runtime["manifest"])
    load_record = validate_world_model_load_record(
        runtime["load_record"], manifest, expected_mode="gated_gemma"
    )
    if manifest["manifest_sha256"] != component_manifest_sha256:
        raise ValueError("component manifest identity drifted")
    expected_checkpoint_id = "stage-d-" + sha256_json(
        {
            "projector": manifest["components"]["future_projector"]["weights"]["sha256"],
            "gate": manifest["components"]["gemma_adapter"]["weights"]["sha256"],
        }
    )
    if checkpoint_id != expected_checkpoint_id:
        raise ValueError("checkpoint_id does not bind the loaded projector/gate component bytes")
    if manifest["base_model"] != {"repo_id": BASE_REPO_ID, "revision": BASE_REVISION}:
        raise ValueError("pinned Gemma base identity drifted")
    if manifest["compatibility"]["transformers_revision"] != TRANSFORMERS_REVISION:
        raise ValueError("component manifest Transformers revision drifted")
    component_paths = verify_world_model_component_files(
        manifest,
        runtime["root"],
        roles=["future_projector", "gemma_adapter"],
    )
    from safetensors.torch import load_file

    projector_state = load_file(component_paths["future_projector"]["weights"])
    adapter_state = load_file(component_paths["gemma_adapter"]["weights"])
    projector = runtime["projector"]
    if set(projector_state) != {"projector.weight"} or set(adapter_state) != {"gate"}:
        raise RuntimeError("projector/adapter safetensors keys drifted")
    torch = _torch()
    if not torch.equal(
        projector.projector.weight.detach().cpu(), projector_state["projector.weight"]
    ) or not torch.equal(projector.gate.detach().cpu(), adapter_state["gate"]):
        raise RuntimeError("in-memory projector/gate do not match verified component bytes")
    model, tokenizer = runtime["model"], runtime["tokenizer"]
    if type(model).__name__ != MODEL_CLASS:
        raise RuntimeError(f"Gemma model class drift: expected {MODEL_CLASS}, got {type(model).__name__}")
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if (
        text_config is None
        or int(getattr(text_config, "hidden_size", -1)) != HIDDEN_SIZE
        or int(getattr(text_config, "num_hidden_layers", -1)) != LAYER_COUNT
    ):
        raise RuntimeError("Gemma text configuration drifted from 48x3840")
    model_commit = getattr(config, "_commit_hash", None)
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if model_commit != BASE_REVISION or tokenizer_commit != BASE_REVISION:
        raise RuntimeError("loaded model/tokenizer revision is not the exact pinned commit")
    if getattr(config, "_name_or_path", None) != BASE_REPO_ID or getattr(
        tokenizer, "name_or_path", None
    ) != BASE_REPO_ID:
        raise RuntimeError("loaded model/tokenizer repository ID drifted")
    if getattr(model, "training", True):
        raise RuntimeError("real Gemma causal tracing requires model.eval()")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("real Gemma causal tracing requires a frozen base model")
    modules = dict(model.named_modules())
    for layer in range(LAYER_COUNT):
        name = f"{LAYER_PREFIX}.{layer}"
        module = modules.get(name)
        if module is None or type(module).__name__ != LAYER_CLASS:
            actual = None if module is None else type(module).__name__
            raise RuntimeError(f"Gemma hook map drift at {name}: {actual}")
    for node in frozen_hook_grid():
        _resolve_node_module(modules, node)
    return {
        "schema_version": "jump.real-gemma-runtime-binding/v1",
        "checkpoint_id": checkpoint_id,
        "base_model_repo_id": BASE_REPO_ID,
        "base_model_revision": BASE_REVISION,
        "tokenizer_revision": BASE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "component_manifest_sha256": manifest["manifest_sha256"],
        "load_record_sha256": load_record["load_record_sha256"],
        "gemma_adapter_identity_sha256": manifest["components"]["gemma_adapter"][
            "identity_sha256"
        ],
        "encoder_artifact_sha256": manifest["components"]["encoder"]["weights"][
            "sha256"
        ],
        "decoder_artifact_sha256": manifest["components"]["decoder"]["weights"][
            "sha256"
        ],
        "model_class": MODEL_CLASS,
        "layer_class": LAYER_CLASS,
        "layer_prefix": LAYER_PREFIX,
        "layer_count": LAYER_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "site_module_types": dict(SITE_MODULE_TYPES),
        "grid_sha256": _sha([node.node_id for node in frozen_hook_grid()]),
    }


def execute_real_gemma_causal_trace(
    runtime: Mapping[str, Any],
    pairs: Sequence[RealGemmaCausalPair],
    *,
    phase: str,
    checkpoint_id: str,
    component_manifest_sha256: str,
    transformers_revision: str,
    frozen_node_id: str | None = None,
    random_seed: int = 33173,
    sham_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Execute C smoke, D exploratory n=32, or frozen-site D confirmation.

    Exploratory execution evaluates all 30 nodes using the locked logit
    contrast and returns the deterministic selected site. Confirmatory
    execution accepts that site explicitly and adds autonomous exact-answer
    generation and all frozen controls. No scientific gate or claim is emitted.
    """
    if phase not in {"smoke", "exploratory", "confirmatory"}:
        raise ValueError("phase must be smoke, exploratory, or confirmatory")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or not 0 <= random_seed <= 0xFFFFFFFF:
        raise ValueError("random_seed must be uint32")
    if not _is_sha256(stage_d_manifest_sha256):
        raise ValueError("stage_d_manifest_sha256 must be an immutable SHA-256")
    if not isinstance(sham_tolerance, (int, float)) or isinstance(sham_tolerance, bool) or sham_tolerance < 0:
        raise ValueError("sham_tolerance must be nonnegative")
    expected_count = {"smoke": 1, "exploratory": 32}.get(phase)
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(f"{phase} requires exactly {expected_count} causal pair(s)")
    if phase == "confirmatory" and len(pairs) < PHASE_COUNTS["confirmatory_minimum"]:
        raise ValueError("confirmatory causal tracing requires at least 200 separate pairs")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("causal trace pair IDs must be unique")
    if [pair.pair_index for pair in pairs] != list(range(len(pairs))):
        raise ValueError("causal trace pairs must cover ordered indices 0..n-1 exactly")
    grid = frozen_hook_grid()
    if phase == "confirmatory":
        nodes = tuple(node for node in grid if node.node_id == frozen_node_id)
        if len(nodes) != 1:
            raise ValueError("confirmatory run requires exactly one frozen 6x5-grid node")
    else:
        if frozen_node_id is not None:
            raise ValueError("smoke/exploratory runs cannot accept a post-selection frozen node")
        nodes = grid
    binding = validate_real_gemma_runtime(
        runtime,
        checkpoint_id=checkpoint_id,
        component_manifest_sha256=component_manifest_sha256,
        transformers_revision=transformers_revision,
    )
    prompt = _validated_prompt(runtime["tokenizer"])
    records: list[dict[str, Any]] = []
    smoke: dict[str, Any] | None = None
    for pair_index, pair in enumerate(pairs):
        materials = _validate_pair(
            pair,
            phase,
            checkpoint_id,
            stage_d_manifest_sha256,
            runtime["manifest"],
        )
        material = materials["shared"]
        clean_z = _latent_tensor(material["clean_raw"], material["clean_world_sha"], runtime)
        divergence = _divergence(runtime["tokenizer"], pair.target_answer, pair.alternative_answer)
        if phase == "smoke":
            smoke = _hook_smoke(runtime, prompt, clean_z, sham_tolerance)
            continue
        margin_inputs = _margin_inputs(prompt, divergence, clean_z.device)
        if phase == "exploratory":
            clean_output, clean_captures = _forward_capture_grid(
                runtime, margin_inputs, clean_z
            )
        else:
            clean_output, clean_capture = _forward_capture(
                runtime, margin_inputs, clean_z, nodes[0]
            )
            clean_captures = {nodes[0].node_id: clean_capture}
        for corruption_kind in ("scrambled_z", "wrong_world_z"):
            corruption = materials[corruption_kind]
            corrupt_z = _latent_tensor(
                corruption["corrupt_raw"], corruption["corrupt_world_sha"], runtime
            )
            if phase == "exploratory":
                corrupt_output, corrupt_captures = _forward_capture_grid(
                    runtime, margin_inputs, corrupt_z
                )
            else:
                corrupt_output, corrupt_capture = _forward_capture(
                    runtime, margin_inputs, corrupt_z, nodes[0]
                )
                corrupt_captures = {nodes[0].node_id: corrupt_capture}
            for node in nodes:
                record = _logit_record(
                    runtime,
                    pair,
                    {**material, **corruption},
                    prompt,
                    clean_z,
                    corrupt_z,
                    divergence,
                    node,
                    random_seed + pair_index,
                    sham_tolerance,
                    clean_baseline=(clean_output, clean_captures[node.node_id]),
                    corrupt_baseline=(corrupt_output, corrupt_captures[node.node_id]),
                    include_controls=phase == "confirmatory",
                )
                if phase == "confirmatory":
                    record["generation"] = _generation_record(
                        runtime,
                        pair,
                        prompt,
                        clean_z,
                        corrupt_z,
                        node,
                    )
                    record["content_sha256"] = _sha(
                        {key: value for key, value in record.items() if key != "content_sha256"}
                    )
                records.append(record)
    selected = select_exploratory_site(records) if phase == "exploratory" else None
    confirmation_summary = (
        _confirmation_summary(records, random_seed) if phase == "confirmatory" else None
    )
    body = {
        "schema_version": REAL_GEMMA_TRACE_VERSION,
        "phase": phase,
        "claim_eligible": False,
        "claim_label": "real Gemma causal-tracing execution evidence; no mechanistic claim",
        "runtime_binding": binding,
        "stage_d_manifest_sha256": stage_d_manifest_sha256,
        "protocol": {
            "exploratory_pair_count": 32,
            "confirmatory_minimum_pair_count": 200,
            "selection": "max min(mean_denoising_restoration,mean_noising_damage); tie earliest T then shallowest L",
            "primary_logit": "target_token_minus_alternative_at_first_canonical_divergence",
            "logit_prefix_use": "shared_canonical_prefix_only_in_scoring_forward",
            "autonomous_generation_target_prefix": False,
            "autonomous_generation_answer_prefix": False,
            "confirmatory_controls": list(CONTROL_NAMES),
            "exploratory_controls": [],
            "no_ols": True,
        },
        "prompt": prompt,
        "textual_boundary_sentinel_resolution_claimed": False,
        "smoke": smoke,
        "records": records,
        "selected_node": selected,
        "confirmation_summary": confirmation_summary,
    }
    return {**body, "content_sha256": _sha(body)}


def select_exploratory_site(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen n=32 selection rule without accepting a chosen site."""
    expected = PHASE_COUNTS["exploratory"] * 2 * len(frozen_hook_grid())
    if len(records) != expected:
        raise ValueError("site selection requires complete n=32 pairs x 2 corruptions x 30 nodes")
    seen = set()
    expected_pairs: set[str] | None = None
    by_node: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("schema_version") != REAL_GEMMA_RECORD_VERSION:
            raise ValueError("site selection received a non-causal-trace record")
        unsigned = {key: value for key, value in record.items() if key != "content_sha256"}
        if record.get("content_sha256") != _sha(unsigned):
            raise ValueError("site selection record immutable hash mismatch")
        key = (record["node_id"], record["pair_id"], record["corruption_kind"])
        if key in seen:
            raise ValueError("site selection contains duplicate node/pair/corruption evidence")
        seen.add(key)
        by_node.setdefault(record["node_id"], []).append(record)
    if set(by_node) != {node.node_id for node in frozen_hook_grid()} or any(
        len(rows) != 64 for rows in by_node.values()
    ):
        raise ValueError("exploratory records do not cover every pair at every frozen node")
    for rows in by_node.values():
        pairs = {row["pair_id"] for row in rows}
        if len(pairs) != 32 or any(
            {row["corruption_kind"] for row in rows if row["pair_id"] == pair_id}
            != {"scrambled_z", "wrong_world_z"}
            for pair_id in pairs
        ):
            raise ValueError("each exploratory node requires both corruptions for the same 32 pairs")
        if expected_pairs is None:
            expected_pairs = pairs
        elif pairs != expected_pairs:
            raise ValueError("exploratory node pair coverage drifted")
    ranked = []
    site_order = {timepoint: index for index, (timepoint, _site) in enumerate(SITES)}
    for node in frozen_hook_grid():
        rows = by_node[node.node_id]
        if {row["corruption_kind"] for row in rows} != {"scrambled_z", "wrong_world_z"}:
            raise ValueError("site selection requires both corruption kinds at every node")
        denoise = sum(float(row["denoising"]["normalized_effect"]) for row in rows) / 64
        noise = sum(float(row["noising"]["normalized_effect"]) for row in rows) / 64
        objective = min(denoise, noise)
        ranked.append((objective, site_order[node.timepoint], node.layer_index, node, denoise, noise))
    objective, _, _, selected, denoise, noise = sorted(
        ranked, key=lambda row: (-row[0], row[1], row[2])
    )[0]
    body = {
        "schema_version": "jump.real-gemma-site-selection/v1",
        "exploratory_only": True,
        "pair_count": 32,
        "node_id": selected.node_id,
        "mean_denoising_restoration": denoise,
        "mean_noising_damage": noise,
        "objective": objective,
        "nondegenerate": bool(math.isfinite(objective) and objective > 0.0),
        "eligible_for_separate_confirmation": bool(
            math.isfinite(objective) and objective > 0.0
        ),
        "tie_break": "earliest_timepoint_then_shallowest_layer",
    }
    return {**body, "content_sha256": _sha(body)}


def _confirmation_summary(
    records: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, Any]:
    summaries = {}
    for corruption in ("scrambled_z", "wrong_world_z"):
        rows = [row for row in records if row["corruption_kind"] == corruption]
        if len(rows) < PHASE_COUNTS["confirmatory_minimum"]:
            raise ValueError("confirmation summary requires >=200 rows per corruption")
        clusters = [row["pair_id"] for row in rows]
        if len(set(clusters)) != len(rows):
            raise ValueError("confirmation requires one row per pair and corruption at frozen node")
        zero = [0.0] * len(rows)
        directions = {}
        for direction in ("denoising", "noising"):
            primary = [float(row[direction]["normalized_effect"]) for row in rows]
            controls = {}
            for control in CONTROL_NAMES:
                control_values = [
                    float(row["controls"][control][direction]["normalized_effect"])
                    for row in rows
                ]
                controls[control] = paired_effect(
                    primary, control_values, cluster_ids=clusters, seed=seed
                )
            baseline = "corrupt" if direction == "denoising" else "clean"
            effect_name = "denoising" if direction == "denoising" else "noising"
            exact_treated = [
                float(row["generation"]["runs"][effect_name]["exact_target"])
                for row in rows
            ]
            exact_control = [
                float(row["generation"]["runs"][baseline]["exact_target"])
                for row in rows
            ]
            if direction == "noising":
                exact_treated, exact_control = exact_control, exact_treated
            directions[direction] = {
                "primary_vs_zero": paired_effect(
                    primary, zero, cluster_ids=clusters, seed=seed
                ),
                "primary_minus_controls": controls,
                "exact_answer_effect": paired_effect(
                    exact_treated, exact_control, cluster_ids=clusters, seed=seed
                ),
                "parse_failure_delta": 0.0,
                "parse_policy": "any malformed generation aborts evidence production",
            }
        summaries[corruption] = directions
    body = {
        "schema_version": "jump.real-gemma-confirmation-summary/v1",
        "paired_cluster_bootstrap_resamples": 10_000,
        "corruptions": summaries,
        "thresholds_for_terminal_audit": {
            "primary_ci_low": 0.0,
            "control_contrast_ci_low": 0.0,
            "exact_answer_effect_minimum": 0.05,
            "parse_failure_delta_maximum_exclusive": 0.02,
        },
        "computed_gate_claim": False,
    }
    return {**body, "content_sha256": _sha(body)}


def write_real_gemma_causal_trace_evidence(
    output_dir: Any,
    *,
    terminal: Mapping[str, Any],
    experiment_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Delegate immutable publication to the shared Track H evidence writer."""
    if terminal.get("schema_version") != REAL_GEMMA_TRACE_VERSION:
        raise ValueError("terminal must be verified real-Gemma causal-trace evidence")
    unsigned = {key: value for key, value in terminal.items() if key != "content_sha256"}
    if terminal.get("content_sha256") != _sha(unsigned):
        raise ValueError("real-Gemma terminal evidence hash mismatch")
    if terminal.get("claim_eligible") is not False:
        raise ValueError("causal trace pilot evidence cannot be claim eligible")
    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError("causal trace evidence root must already exist")
    trace_path = root / "causal-trace.json"
    with trace_path.open("xb") as handle:
        handle.write(canonical_json(terminal) + b"\n")
    metrics = _terminal_metrics(terminal)
    return write_track_h_task_evidence(
        output_dir,
        metrics=metrics,
        terminal=terminal,
        experiment_spec=experiment_spec,
        track_h={"stage": "R-C/D", "mechanistic_evidence": False},
    )


def _validate_pair(
    pair: RealGemmaCausalPair,
    phase: str,
    checkpoint_id: str,
    manifest_sha256: str,
    component_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(pair, RealGemmaCausalPair):
        raise ValueError("causal trace inputs must be RealGemmaCausalPair records")
    clean = _validate_arm(
        pair.own_z,
        "own_z",
        checkpoint_id,
        manifest_sha256,
        component_manifest,
    )
    corruptions = {
        "scrambled_z": _validate_arm(
            pair.scrambled_z,
            "scrambled_z",
            checkpoint_id,
            manifest_sha256,
            component_manifest,
        ),
        "wrong_world_z": _validate_arm(
            pair.wrong_world_z,
            "wrong_world_z",
            checkpoint_id,
            manifest_sha256,
            component_manifest,
        ),
    }
    domain = (
        "real-gemma-causal-pilot"
        if phase in {"smoke", "exploratory"}
        else "real-gemma-causal-confirmation"
    )
    pair_seed = derive_seed(33173, f"{domain}:{pair.pair_index}")
    expected_pair = matched_world_pair(pair_seed=pair_seed)
    expected_a = expected_pair["a"]["episode_id"]
    expected_b = expected_pair["b"]["episode_id"]
    if pair.pair_id != expected_pair["pair_id"]:
        raise ValueError("causal pair_id does not match the frozen root/domain/index derivation")
    clean_control = clean["control"]
    if clean_control["pair_id"] != pair.pair_id or any(
        corrupt["control"]["pair_id"] != pair.pair_id for corrupt in corruptions.values()
    ):
        raise ValueError("causal pair identity does not match its three sealed arms")
    if clean_control["source_world_id"] != clean_control["recipient_world_id"]:
        raise ValueError("own_z must use its recipient's own learned latent")
    if (
        clean_control["recipient_world_id"] != expected_a
        or clean_control["source_world_id"] != expected_a
    ):
        raise ValueError("own_z recipient/source do not match frozen matched-pair World A")
    for kind, corrupt in corruptions.items():
        corrupt_control = corrupt["control"]
        if clean_control["recipient_world_id"] != corrupt_control["recipient_world_id"]:
            raise ValueError("clean/corrupt arms require the same recipient world")
        if kind == "scrambled_z":
            if corrupt_control["source_world_id"] != clean_control["source_world_id"]:
                raise ValueError("scrambled_z must preserve the original own-z source")
            if (
                corrupt["learned_envelope_sha256"] != clean["learned_envelope_sha256"]
                or corrupt["source_raw"] != clean["source_raw"]
                or corrupt["injected_raw"] == clean["injected_raw"]
            ):
                raise ValueError("scrambled_z must permute the exact own-z envelope bytes")
        elif (
            corrupt_control["recipient_world_id"] != expected_a
            or corrupt_control["source_world_id"] != expected_b
        ):
            raise ValueError("wrong_world_z must transport frozen matched-pair World B into A")
    score_episode(dict(pair.target_answer), dict(pair.target_answer))
    score_episode(dict(pair.alternative_answer), dict(pair.alternative_answer))
    if canonical_json(pair.target_answer) == canonical_json(pair.alternative_answer):
        raise ValueError("target and counterfactual answers must differ")
    expected_target = _stage_d_target(expected_pair["a"]["scoring_target"])
    expected_alternative = _stage_d_target(expected_pair["b"]["scoring_target"])
    if dict(pair.target_answer) != expected_target or dict(pair.alternative_answer) != expected_alternative:
        raise ValueError("causal target/alternative do not match the frozen matched-pair answers")
    shared = {
        "clean_raw": clean["injected_raw"],
        "clean_world_sha": clean["injected_world_sha"],
        "clean_control_payload_sha256": pair.own_z.control_envelope["payload_sha256"],
        "clean_learned_envelope_sha256": clean["learned_envelope_sha256"],
    }
    result = {"shared": shared}
    for kind, arm in (("scrambled_z", pair.scrambled_z), ("wrong_world_z", pair.wrong_world_z)):
        corrupt = corruptions[kind]
        result[kind] = {
            "corruption_kind": kind,
            "corrupt_raw": corrupt["injected_raw"],
            "corrupt_world_sha": corrupt["injected_world_sha"],
            "corrupt_control_payload_sha256": arm.control_envelope["payload_sha256"],
            "corrupt_learned_envelope_sha256": corrupt["learned_envelope_sha256"],
        }
    return result


def _validate_arm(
    arm: RealGemmaLatentArm,
    expected_arm_id: str,
    checkpoint_id: str,
    manifest_sha256: str,
    component_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(arm, RealGemmaLatentArm):
        raise ValueError("causal trace arm material must be RealGemmaLatentArm")
    control = open_result_envelope(
        arm.control_envelope,
        expected_source=arm.expected_source,
        expected_manifest_sha256=manifest_sha256,
        expected_checkpoint_id=checkpoint_id,
    )
    if set(control) != _ARM_FIELDS or control["schema_version"] != STAGE_D_CONTROL_VERSION:
        raise ValueError("causal arm is not an exact Stage D v1 control payload")
    if (
        control["arm_id"] != expected_arm_id
        or control["control_kind"] != expected_arm_id
        or control["checkpoint_id"] != checkpoint_id
        or control["manifest_sha256"] != manifest_sha256
        or control["answer_sha256"] != _sha(control["answer"])
        or control["execution_contract_sha256"] != STAGE_D_EXECUTION_CONTRACT_SHA256
    ):
        raise ValueError("causal arm identity or answer binding drifted")
    score_episode(control["answer"], control["answer"])
    learned_payload = open_result_envelope(
        arm.learned_latent_envelope,
        expected_source=arm.expected_source,
        expected_manifest_sha256=manifest_sha256,
        expected_checkpoint_id=checkpoint_id,
    )
    learned = validate_learned_latent_evidence(learned_payload)
    components = validate_world_model_component_manifest(component_manifest)["components"]
    if learned["learned_decoder"]["artifact_sha256"] != components["decoder"]["weights"][
        "sha256"
    ]:
        raise ValueError("learned-latent evidence decoder does not match loaded distribution")
    learned_answer = learned["answer"]
    producer = learned_answer.get("producer_bindings") if isinstance(learned_answer, dict) else None
    if not isinstance(producer, dict) or producer.get("encoder_artifact_sha256") != components[
        "encoder"
    ]["weights"]["sha256"]:
        raise ValueError("learned-latent evidence encoder does not match loaded distribution")
    verify_latent_tensor_bytes(learned, arm.source_tensor_bytes)
    injection = control["injection"]
    if not isinstance(injection, dict) or set(injection) != _INJECTION_FIELDS:
        raise ValueError("causal arm injection fields drifted from Stage D v1")
    if injection.get("present") is not True or injection.get(
        "learned_latent_envelope_sha256"
    ) != _sha(arm.learned_latent_envelope):
        raise ValueError("causal arm injection does not bind the learned envelope")
    tensor = learned["tensor"]
    expected_shape = component_manifest["compatibility"]["latent"]["shape"]
    if (
        tensor["dtype"] != "float32-le"
        or tensor["shape"] != expected_shape
        or len(expected_shape) != 1
        or expected_shape[0] < 2
        or tensor["order"] != "C"
    ):
        raise ValueError("causal tracing latent descriptor does not match the verified 1-D float32 distribution")
    if injection.get("dtype") != "float32-le" or injection.get("shape") != expected_shape or injection.get("order") != "C":
        raise ValueError("causal arm injection tensor descriptor drifted")
    injected_raw_sha = hashlib.sha256(arm.injected_tensor_bytes).hexdigest()
    injected_world_sha = tensor_bytes_sha256(
        arm.injected_tensor_bytes, dtype="float32-le", shape=expected_shape, order="C"
    )
    if (
        injection.get("raw_bytes_sha256") != injected_raw_sha
        or injection.get("world_latent_sha256") != injected_world_sha
    ):
        raise ValueError("causal arm injected tensor bytes do not match sealed hashes")
    lineage = learned["swap_lineage"]
    if (
        lineage["world_pair_id"] != control["pair_id"]
        or lineage["recipient_world_id"] != control["recipient_world_id"]
        or lineage["source_world_id"] != control["source_world_id"]
    ):
        raise ValueError("causal arm learned-latent lineage drifted")
    if expected_arm_id == "scrambled_z":
        _validate_permutation(injection["permutation"], arm.source_tensor_bytes, arm.injected_tensor_bytes, tensor)
    else:
        if injection.get("permutation") is not None or arm.injected_tensor_bytes != arm.source_tensor_bytes:
            raise ValueError(f"{expected_arm_id} must inject the verified learned tensor unchanged")
        if injected_world_sha != tensor["world_latent_sha256"]:
            raise ValueError("identity arm latent hash drifted")
    return {
        "control": control,
        "source_raw": bytes(arm.source_tensor_bytes),
        "injected_raw": bytes(arm.injected_tensor_bytes),
        "injected_world_sha": injected_world_sha,
        "learned_envelope_sha256": _sha(arm.learned_latent_envelope),
    }


def _validate_permutation(
    value: Any,
    source_raw: bytes,
    injected_raw: bytes,
    tensor: Mapping[str, Any],
) -> None:
    fields = {
        "schema_version",
        "unit",
        "seed",
        "indices",
        "indices_sha256",
        "source_world_latent_sha256",
        "permuted_world_latent_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("scrambled causal arm requires exact permutation evidence")
    indices = value["indices"]
    element_count = math.prod(tensor["shape"])
    if (
        value["schema_version"] != LATENT_PERMUTATION_VERSION
        or value["unit"] != "float32_element"
        or isinstance(value["seed"], bool)
        or not isinstance(value["seed"], int)
        or not 0 <= value["seed"] <= 0xFFFFFFFF
        or not isinstance(indices, list)
        or sorted(indices) != list(range(element_count))
        or indices == list(range(element_count))
        or value["indices_sha256"] != _sha(indices)
        or value["source_world_latent_sha256"] != tensor["world_latent_sha256"]
    ):
        raise ValueError("scrambled causal arm permutation metadata drifted")
    expected = b"".join(source_raw[index * 4 : (index + 1) * 4] for index in indices)
    permuted_sha = tensor_bytes_sha256(expected, dtype="float32-le", shape=tensor["shape"], order="C")
    if expected == source_raw or expected != injected_raw or value[
        "permuted_world_latent_sha256"
    ] != permuted_sha:
        raise ValueError("scrambled causal arm is not the declared float32-element permutation")


def _validated_prompt(tokenizer: Any) -> dict[str, Any]:
    prompt = matched_injection_prompt()
    tensors = prompt_tensors(tokenizer)
    ids = tuple(int(item) for item in tensors["input_ids"][0].tolist())
    mask = tuple(int(item) for item in tensors["attention_mask"][0].tolist())
    repeated = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    if tuple(int(item) for item in repeated["input_ids"][0].tolist()) != ids:
        raise RuntimeError("pinned Stage D prompt retokenized between calls")
    if not ids or len(ids) != len(mask) or any(item != 1 for item in mask):
        raise RuntimeError("Stage D causal prompt must be one unmasked nonempty sequence")
    return {
        "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "tokenizer_revision": BASE_REVISION,
        "token_ids": list(ids),
        "attention_mask": list(mask),
        "token_ids_sha256": _sha(list(ids)),
        "z_in_prompt": False,
    }


def _latent_tensor(raw: bytes, world_sha: str, runtime: Mapping[str, Any]) -> Any:
    torch = _torch()
    model = runtime["model"]
    device = next(model.parameters()).device
    shape = runtime["manifest"]["compatibility"]["latent"]["shape"]
    if len(shape) != 1 or shape[0] < 2 or len(raw) != shape[0] * 4:
        raise ValueError("causal latent byte length does not match verified component shape")
    if tensor_bytes_sha256(raw, dtype="float32-le", shape=shape, order="C") != world_sha:
        raise ValueError("causal latent tensor hash mismatch")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).to(device).reshape(1, shape[0])


def _divergence(tokenizer: Any, target: Mapping[str, Any], alternative: Mapping[str, Any]) -> dict[str, Any]:
    grammar = ExactAnswerGrammar(tokenizer)
    target_text = json.dumps(dict(target), sort_keys=True, separators=(",", ":"))
    alt_text = json.dumps(dict(alternative), sort_keys=True, separators=(",", ":"))
    target_ids = tuple(tokenizer(target_text, add_special_tokens=False)["input_ids"])
    alt_ids = tuple(tokenizer(alt_text, add_special_tokens=False)["input_ids"])
    if not grammar.admits(dict(target), tokenizer) or not grammar.admits(dict(alternative), tokenizer):
        raise ValueError("causal target/alternative must belong to the frozen exact-answer grammar")
    index = next(
        (i for i, (left, right) in enumerate(zip(target_ids, alt_ids)) if left != right),
        None,
    )
    if index is None:
        raise ValueError("canonical target and alternative have no shared-prefix divergence")
    return {
        "common_prefix_ids": target_ids[:index],
        "target_token_id": target_ids[index],
        "alternative_token_id": alt_ids[index],
        "divergence_index": index,
        "target_answer_sha256": _sha(target),
        "alternative_answer_sha256": _sha(alternative),
    }


def _hook_smoke(
    runtime: Mapping[str, Any],
    prompt: Mapping[str, Any],
    clean_z: Any,
    sham_tolerance: float,
) -> dict[str, Any]:
    torch = _torch()
    model = runtime["model"]
    modules = dict(model.named_modules())
    inputs = {
        "input_ids": torch.tensor(
            [prompt["token_ids"]], dtype=torch.long, device=clean_z.device
        ),
        "attention_mask": torch.tensor(
            [prompt["attention_mask"]], dtype=torch.long, device=clean_z.device
        ),
        "use_cache": False,
    }
    calls: dict[str, list[dict[str, Any]]] = {node.node_id: [] for node in frozen_hook_grid()}
    handles = []
    try:
        for node in frozen_hook_grid():
            handles.append(_register_capture(modules, node, calls[node.node_id]))
        with torch.inference_mode(), persistent_z_injection(
            model, runtime["projector"], clean_z, enabled=True
        ):
            baseline_output = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(rows) != 1 for rows in calls.values()):
        raise RuntimeError("6x5 smoke requires every hook to fire exactly once per forward")
    baseline_logits = baseline_output.logits[0, -1].float()
    sham_deltas: dict[str, float] = {}
    for node in frozen_hook_grid():
        source = calls[node.node_id][0]["last"]
        patched_output, observed = _forward_with_patch(
            runtime,
            inputs,
            clean_z,
            node,
            source,
            strategy="source",
            scale=0.0,
            seed=0,
        )
        if observed != 1:
            raise RuntimeError("scale-zero smoke patch did not fire exactly once")
        delta = float(
            (patched_output.logits[0, -1].float() - baseline_logits).abs().max().item()
        )
        if delta > sham_tolerance:
            raise RuntimeError(
                f"scale-zero sham drift at {node.node_id}: {delta} > {sham_tolerance}"
            )
        sham_deltas[node.node_id] = delta
    nodes = []
    for node in frozen_hook_grid():
        row = calls[node.node_id][0]
        nodes.append(
            {
                "node_id": node.node_id,
                "module_name": row["module_name"],
                "module_type": row["module_type"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "device": row["device"],
                "single_fire": True,
                "activation_sha256": row["activation_sha256"],
                "zero_sham_logit_delta": sham_deltas[node.node_id],
            }
        )
    body = {
        "schema_version": "jump.real-gemma-hook-smoke/v1",
        "grid_version": REAL_GEMMA_GRID_VERSION,
        "node_count": len(nodes),
        "semantic_hook_sentinels": [
            {"timepoint": timepoint, "site": site} for timepoint, site in SITES
        ],
        "textual_boundary_sentinel_resolution_claimed": False,
        "audited_prompt_tokens_unchanged": True,
        "all_modules_present": True,
        "all_shapes_valid": True,
        "all_single_fire": True,
        "zero_sham_tolerance": sham_tolerance,
        "zero_sham_within_tolerance": True,
        "nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}


def _logit_record(
    runtime: Mapping[str, Any],
    pair: RealGemmaCausalPair,
    material: Mapping[str, Any],
    prompt: Mapping[str, Any],
    clean_z: Any,
    corrupt_z: Any,
    divergence: Mapping[str, Any],
    node: RealGemmaNode,
    seed: int,
    sham_tolerance: float,
    clean_baseline: tuple[Any, Mapping[str, Any]],
    corrupt_baseline: tuple[Any, Mapping[str, Any]],
    include_controls: bool,
) -> dict[str, Any]:
    inputs = _margin_inputs(prompt, divergence, clean_z.device)
    clean_output, clean_capture = clean_baseline
    corrupt_output, corrupt_capture = corrupt_baseline
    clean_margin = _margin(clean_output.logits[0, -1], divergence)
    corrupt_margin = _margin(corrupt_output.logits[0, -1], divergence)
    denominator = clean_margin - corrupt_margin
    if denominator <= 1e-6:
        raise RuntimeError(
            f"invalid causal comparison at {pair.pair_id}/{node.node_id}: "
            "clean target-logit ceiling does not exceed corrupt floor"
        )
    denoised, denoise_fires = _forward_with_patch(
        runtime,
        inputs,
        corrupt_z,
        node,
        clean_capture["last"],
        strategy="source",
        scale=1.0,
        seed=seed,
    )
    noised, noise_fires = _forward_with_patch(
        runtime,
        inputs,
        clean_z,
        node,
        corrupt_capture["last"],
        strategy="source",
        scale=1.0,
        seed=seed,
    )
    if denoise_fires != 1 or noise_fires != 1:
        raise RuntimeError("causal patch hook did not fire exactly once")
    denoised_margin = _margin(denoised.logits[0, -1], divergence)
    noised_margin = _margin(noised.logits[0, -1], divergence)
    denoising_effect = _finite_ratio(
        denoised_margin - corrupt_margin, denominator, "denoising restoration"
    )
    noising_effect = _finite_ratio(
        clean_margin - noised_margin, denominator, "noising damage"
    )
    controls: dict[str, dict[str, Any]] | None = {} if include_controls else None
    for control, strategy, position in (() if not include_controls else (
        ("zero_sham", "source", "last"),
        ("matched_norm_random", "random", "last"),
        ("orthogonal", "orthogonal", "last"),
        ("prompt_token", "source", "first"),
    )):
        scale = 0.0 if control == "zero_sham" else 1.0
        directions = {}
        for direction, target_z, source_capture, baseline in (
            ("denoising", corrupt_z, clean_capture, corrupt_margin),
            ("noising", clean_z, corrupt_capture, clean_margin),
        ):
            source = source_capture[position]
            output, fires = _forward_with_patch(
                runtime,
                inputs,
                target_z,
                node,
                source,
                strategy=strategy,
                scale=scale,
                seed=seed,
                position=position,
            )
            if fires != 1:
                raise RuntimeError(f"{direction}/{control} hook did not fire exactly once")
            value = _margin(output.logits[0, -1], divergence)
            if control == "zero_sham" and abs(value - baseline) > sham_tolerance:
                raise RuntimeError(
                    f"{direction} zero-sham target logit drift exceeded frozen tolerance"
                )
            effect = _finite_ratio(
                value - corrupt_margin
                if direction == "denoising"
                else clean_margin - value,
                denominator,
                f"{direction}/{control}",
            )
            directions[direction] = {
                "patched_logit_difference": value,
                "normalized_effect": effect,
            }
        if controls is None:
            raise RuntimeError("confirmatory control collection was not initialized")
        controls[control] = directions
    body = {
        "schema_version": REAL_GEMMA_RECORD_VERSION,
        "pair_id": pair.pair_id,
        "corruption_kind": material["corruption_kind"],
        "node_id": node.node_id,
        "layer_index": node.layer_index,
        "timepoint": node.timepoint,
        "site": node.site,
        "clean_control_payload_sha256": material["clean_control_payload_sha256"],
        "corrupt_control_payload_sha256": material["corrupt_control_payload_sha256"],
        "clean_learned_envelope_sha256": material["clean_learned_envelope_sha256"],
        "corrupt_learned_envelope_sha256": material["corrupt_learned_envelope_sha256"],
        "clean_world_latent_sha256": material["clean_world_sha"],
        "corrupt_world_latent_sha256": material["corrupt_world_sha"],
        "target_answer_sha256": divergence["target_answer_sha256"],
        "alternative_answer_sha256": divergence["alternative_answer_sha256"],
        "divergence_index": divergence["divergence_index"],
        "target_token_id": divergence["target_token_id"],
        "alternative_token_id": divergence["alternative_token_id"],
        "clean_activation_sha256": clean_capture["activation_sha256"],
        "corrupt_activation_sha256": corrupt_capture["activation_sha256"],
        "clean_prompt_token_activation_sha256": clean_capture[
            "first_activation_sha256"
        ],
        "corrupt_prompt_token_activation_sha256": corrupt_capture[
            "first_activation_sha256"
        ],
        "raw_floor_ceiling": {
            "clean_logit_difference": clean_margin,
            "corrupt_logit_difference": corrupt_margin,
            "clean_minus_corrupt": denominator,
        },
        "denoising": {
            "patched_logit_difference": denoised_margin,
            "normalized_effect": denoising_effect,
        },
        "noising": {
            "patched_logit_difference": noised_margin,
            "normalized_effect": noising_effect,
        },
        "controls": controls,
    }
    return {**body, "content_sha256": _sha(body)}


def _generation_record(
    runtime: Mapping[str, Any],
    pair: RealGemmaCausalPair,
    prompt: Mapping[str, Any],
    clean_z: Any,
    corrupt_z: Any,
    node: RealGemmaNode,
) -> dict[str, Any]:
    clean = _generate_with_hook(runtime, prompt, clean_z, node=node, capture=True)
    corrupt = _generate_with_hook(runtime, prompt, corrupt_z, node=node, capture=True)
    runs = {"clean": clean["output"], "corrupt": corrupt["output"]}
    patch_specs = (
        ("denoising", corrupt_z, clean["trace"], "source", 1.0, "last"),
        ("noising", clean_z, corrupt["trace"], "source", 1.0, "last"),
    )
    for name, latent, trace, strategy, scale, position in patch_specs:
        runs[name] = _generate_with_hook(
            runtime,
            prompt,
            latent,
            node=node,
            patch_trace=trace,
            patch_strategy=strategy,
            scale=scale,
            seed=0,
            patch_position=position,
        )["output"]
    target = dict(pair.target_answer)
    for output in runs.values():
        output["exact_target"] = output["answer"] == target
        output["exact_scores"] = score_episode(output["answer"], target)
        output["answer"] = dict(output["answer"])
    return {
        "schema_version": "jump.real-gemma-generation-effects/v1",
        "parse_policy": "malformed_output_rejects_record",
        "parse_success": {name: True for name in runs},
        "clean_activation_trace_sha256": _sha(
            [
                [row["activation_sha256"], row["first_activation_sha256"]]
                for row in clean["trace"]
            ]
        ),
        "corrupt_activation_trace_sha256": _sha(
            [
                [row["activation_sha256"], row["first_activation_sha256"]]
                for row in corrupt["trace"]
            ]
        ),
        "runs": runs,
        "denoising_exact_answer_effect": int(runs["denoising"]["exact_target"]) - int(
            runs["corrupt"]["exact_target"]
        ),
        "noising_exact_answer_damage": int(runs["clean"]["exact_target"]) - int(
            runs["noising"]["exact_target"]
        ),
    }


def _forward_capture(
    runtime: Mapping[str, Any], inputs: Mapping[str, Any], latent: Any, node: RealGemmaNode
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    rows: list[dict[str, Any]] = []
    modules = dict(runtime["model"].named_modules())
    handle = _register_capture(modules, node, rows)
    try:
        with torch.inference_mode(), persistent_z_injection(
            runtime["model"], runtime["projector"], latent, enabled=True
        ):
            output = runtime["model"](**dict(inputs))
    finally:
        handle.remove()
    if len(rows) != 1:
        raise RuntimeError(f"{node.node_id} fired {len(rows)} times in one logit forward")
    return output, rows[0]


def _forward_capture_grid(
    runtime: Mapping[str, Any], inputs: Mapping[str, Any], latent: Any
) -> tuple[Any, dict[str, Mapping[str, Any]]]:
    """Capture all 30 nodes in one forward for the cheap exploratory phase."""
    torch = _torch()
    modules = dict(runtime["model"].named_modules())
    rows: dict[str, list[dict[str, Any]]] = {
        node.node_id: [] for node in frozen_hook_grid()
    }
    handles = [
        _register_capture(modules, node, rows[node.node_id]) for node in frozen_hook_grid()
    ]
    try:
        with torch.inference_mode(), persistent_z_injection(
            runtime["model"], runtime["projector"], latent, enabled=True
        ):
            output = runtime["model"](**dict(inputs))
    finally:
        for handle in handles:
            handle.remove()
    if any(len(items) != 1 for items in rows.values()):
        raise RuntimeError("exploratory 6x5 capture requires every hook to fire exactly once")
    return output, {node_id: items[0] for node_id, items in rows.items()}


def _forward_with_patch(
    runtime: Mapping[str, Any],
    inputs: Mapping[str, Any],
    latent: Any,
    node: RealGemmaNode,
    source: Any,
    *,
    strategy: str,
    scale: float,
    seed: int,
    position: str = "last",
) -> tuple[Any, int]:
    torch = _torch()
    modules = dict(runtime["model"].named_modules())
    count = {"value": 0}
    handle = _register_patch(
        modules,
        node,
        source_provider=lambda _call: source,
        strategy=strategy,
        scale=scale,
        seed=seed,
        position=position,
        count=count,
    )
    try:
        with torch.inference_mode(), persistent_z_injection(
            runtime["model"], runtime["projector"], latent, enabled=True
        ):
            output = runtime["model"](**dict(inputs))
    finally:
        handle.remove()
    return output, count["value"]


def _generate_with_hook(
    runtime: Mapping[str, Any],
    prompt: Mapping[str, Any],
    latent: Any,
    *,
    node: RealGemmaNode,
    capture: bool = False,
    patch_trace: Sequence[Mapping[str, Any]] | None = None,
    patch_strategy: str = "source",
    scale: float = 1.0,
    seed: int = 0,
    patch_position: str = "last",
) -> dict[str, Any]:
    torch = _torch()
    from transformers import StoppingCriteria, StoppingCriteriaList

    model, tokenizer = runtime["model"], runtime["tokenizer"]
    input_ids = torch.tensor([prompt["token_ids"]], dtype=torch.long, device=latent.device)
    attention_mask = torch.tensor(
        [prompt["attention_mask"]], dtype=torch.long, device=latent.device
    )
    grammar = ExactAnswerGrammar(tokenizer)
    prompt_length = input_ids.shape[1]

    class GrammarComplete(StoppingCriteria):
        def __call__(self, current_ids: Any, scores: Any, **kwargs: Any) -> Any:
            suffix = tuple(int(item) for item in current_ids[0, prompt_length:].tolist())
            return torch.tensor([grammar.completed(suffix)], device=current_ids.device)

    def allowed(_batch_id: int, current_ids: Any) -> list[int]:
        suffix = tuple(int(item) for item in current_ids[prompt_length:].tolist())
        return grammar.allowed(suffix)

    modules = dict(model.named_modules())
    trace: list[dict[str, Any]] = []
    count = {"value": 0}
    handle = None
    if capture:
        handle = _register_capture(modules, node, trace)
    elif patch_trace is not None:
        if not patch_trace:
            raise ValueError("persistent generation patch requires a nonempty source trace")

        def provider(call: int) -> Any:
            if call >= len(patch_trace):
                raise RuntimeError("patched generation exceeded the source activation trace")
            source_position = "first" if patch_position == "first_prefill" else "last"
            return patch_trace[call][source_position]

        handle = _register_patch(
            modules,
            node,
            source_provider=provider,
            strategy=patch_strategy,
            scale=scale,
            seed=seed,
            position=patch_position,
            count=count,
        )
    try:
        with torch.inference_mode(), persistent_z_injection(
            model, runtime["projector"], latent, enabled=True
        ) as latent_binding:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=128,
                prefix_allowed_tokens_fn=allowed,
                stopping_criteria=StoppingCriteriaList([GrammarComplete()]),
            )
    finally:
        if handle is not None:
            handle.remove()
    generated = tuple(int(item) for item in output[0, prompt_length:].tolist())
    answer = grammar.parse(generated)
    calls = len(trace) if capture else count["value"]
    if calls != len(generated) or calls < 1:
        raise RuntimeError(
            "persistent generation hook must fire exactly once for every generated token"
        )
    z_calls = latent_binding["forward_calls"]["count"]
    if z_calls != len(generated):
        raise RuntimeError("nontextual z injection did not persist across every generation step")
    if patch_position == "first_prefill" and calls < 1:
        raise RuntimeError("prompt-token control missed prefill")
    result = {
        "answer": answer,
        "answer_sha256": _sha(answer),
        "generated_token_count": len(generated),
        "generated_token_ids_sha256": _sha(list(generated)),
        "forward_calls": calls,
        "persistent_z_forward_calls": z_calls,
    }
    return {"output": result, "trace": trace}


def _register_capture(
    modules: Mapping[str, Any], node: RealGemmaNode, sink: list[dict[str, Any]]
) -> Any:
    module, pre, module_name = _resolve_node_module(modules, node)

    def record(tensor: Any) -> None:
        _validate_hook_tensor(tensor, node)
        detached = tensor.detach().clone()
        last = detached[:, -1, :]
        first = detached[:, 0, :]
        sink.append(
            {
                "last": last,
                "first": first,
                "module_name": module_name,
                "module_type": type(module).__name__,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "activation_sha256": _tensor_sha(last),
                "first_activation_sha256": _tensor_sha(first),
            }
        )

    if pre:
        def pre_hook(_module: Any, args: Any) -> None:
            record(_first_tensor(args))

        return module.register_forward_pre_hook(pre_hook)

    def post_hook(_module: Any, _args: Any, output: Any) -> None:
        record(_first_tensor(output))

    return module.register_forward_hook(post_hook)


def _register_patch(
    modules: Mapping[str, Any],
    node: RealGemmaNode,
    *,
    source_provider: Callable[[int], Any],
    strategy: str,
    scale: float,
    seed: int,
    position: str,
    count: dict[str, int],
) -> Any:
    module, pre, _module_name = _resolve_node_module(modules, node)

    def patch_tensor(tensor: Any) -> Any:
        _validate_hook_tensor(tensor, node)
        call = count["value"]
        count["value"] += 1
        if position == "first_prefill" and call > 0:
            return tensor
        index = 0 if position in {"first", "first_prefill"} else tensor.shape[-2] - 1
        source = source_provider(call).to(device=tensor.device, dtype=tensor.dtype)
        if tuple(source.shape) != (1, HIDDEN_SIZE):
            raise RuntimeError("source activation shape drifted")
        current = tensor[:, index, :]
        replacement = _replacement(current, source, strategy, seed + call)
        patched = tensor.clone()
        patched[:, index, :] = current + (replacement - current) * float(scale)
        return patched

    if pre:
        def pre_hook(_module: Any, args: Any) -> Any:
            if not isinstance(args, tuple) or not args:
                raise RuntimeError("Gemma pre-hook arguments drifted")
            values = list(args)
            values[0] = patch_tensor(_first_tensor(args))
            return tuple(values)

        return module.register_forward_pre_hook(pre_hook)

    def post_hook(_module: Any, _args: Any, output: Any) -> Any:
        patched = patch_tensor(_first_tensor(output))
        return _replace_first_tensor(output, patched)

    return module.register_forward_hook(post_hook)


def _resolve_node_module(
    modules: Mapping[str, Any], node: RealGemmaNode
) -> tuple[Any, bool, str]:
    layer_name = f"{LAYER_PREFIX}.{node.layer_index}"
    target = {
        "residual_pre": (layer_name, True),
        "attention_output": (f"{layer_name}.self_attn", False),
        "residual_mid": (f"{layer_name}.pre_feedforward_layernorm", True),
        "mlp_output": (f"{layer_name}.mlp", False),
        "residual_post": (layer_name, False),
    }.get(node.site)
    if target is None:
        raise ValueError(f"unknown frozen Gemma site: {node.site}")
    module_name, pre = target
    module = modules.get(module_name)
    if module is None:
        raise RuntimeError(f"Gemma hook map drift: missing {module_name}")
    expected_type = SITE_MODULE_TYPES[node.site]
    if type(module).__name__ != expected_type:
        raise RuntimeError(
            f"Gemma hook type drift at {module_name}: expected {expected_type}, "
            f"got {type(module).__name__}"
        )
    return module, pre, module_name


def _replacement(current: Any, source: Any, strategy: str, seed: int) -> Any:
    torch = _torch()
    if strategy == "source":
        return source
    delta = source - current
    norm = torch.linalg.vector_norm(delta.float(), dim=-1, keepdim=True)
    if not torch.all(torch.isfinite(norm)) or torch.any(norm <= 1e-12):
        raise RuntimeError("matched causal control requires a finite nonzero patch norm")
    generator = torch.Generator(device=current.device).manual_seed(seed)
    random = torch.randn(current.shape, generator=generator, device=current.device).float()
    if strategy == "orthogonal":
        unit = delta.float() / norm
        random = random - (random * unit).sum(dim=-1, keepdim=True) * unit
    elif strategy != "random":
        raise ValueError(f"unknown patch strategy: {strategy}")
    random_norm = torch.linalg.vector_norm(random, dim=-1, keepdim=True)
    if torch.any(random_norm <= 1e-12):
        raise RuntimeError("degenerate matched causal control vector")
    return current + (random / random_norm * norm).to(dtype=current.dtype)


def _margin_inputs(
    prompt: Mapping[str, Any], divergence: Mapping[str, Any], device: Any
) -> dict[str, Any]:
    torch = _torch()
    ids = [*prompt["token_ids"], *divergence["common_prefix_ids"]]
    if not divergence["common_prefix_ids"]:
        raise ValueError("target/alternative divergence must follow a nonempty common answer prefix")
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        "use_cache": False,
    }


def _margin(logits: Any, divergence: Mapping[str, Any]) -> float:
    value = float(
        (
            logits[divergence["target_token_id"]]
            - logits[divergence["alternative_token_id"]]
        )
        .float()
        .item()
    )
    if not math.isfinite(value):
        raise RuntimeError("target-minus-alternative logit difference is nonfinite")
    return value


def _finite_ratio(numerator: float, denominator: float, name: str) -> float:
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite normalized causal effect: {name}")
    return value


def _validate_hook_tensor(tensor: Any, node: RealGemmaNode) -> None:
    shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
    if len(shape) != 3 or shape[0] != 1 or shape[-1] != HIDDEN_SIZE or shape[-2] < 1:
        raise RuntimeError(
            f"Gemma hook shape drift at {node.node_id}: expected [1,seq,{HIDDEN_SIZE}], got {shape}"
        )
    if not hasattr(tensor, "dtype") or not hasattr(tensor, "device"):
        raise RuntimeError("Gemma hook output lacks dtype/device")


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, (tuple, list)) and value and hasattr(value[0], "shape"):
        return value[0]
    raise RuntimeError("Gemma hook input/output no longer begins with a tensor")


def _replace_first_tensor(value: Any, tensor: Any) -> Any:
    if hasattr(value, "shape"):
        return tensor
    if not isinstance(value, (tuple, list)) or not value:
        raise RuntimeError("Gemma hook output structure drifted")
    items = list(value)
    items[0] = tensor
    return tuple(items) if isinstance(value, tuple) else items


def _tensor_sha(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu", dtype=_torch().float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _terminal_metrics(terminal: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    smoke = terminal.get("smoke")
    if smoke is not None:
        metrics.extend(
            [
                {"name": "real_gemma_hook_node_count", "value": float(smoke["node_count"])},
                {
                    "name": "real_gemma_zero_sham_passed",
                    "value": float(smoke["zero_sham_within_tolerance"]),
                },
            ]
        )
    records = terminal.get("records", [])
    if records:
        metrics.extend(
            [
                {
                    "name": "mean_denoising_restoration",
                    "value": sum(row["denoising"]["normalized_effect"] for row in records)
                    / len(records),
                },
                {
                    "name": "mean_noising_damage",
                    "value": sum(row["noising"]["normalized_effect"] for row in records)
                    / len(records),
                },
            ]
        )
    return metrics


def _torch() -> Any:
    import torch

    return torch


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
