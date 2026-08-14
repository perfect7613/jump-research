"""Bounded general world-model pilot for declarative visual ExperimentSpec v2.

The simulator is used only to construct training/evaluation tensors. Runtime
model APIs accept sealed tensors/spec embeddings and never call the simulator.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from jump_contracts import (
    artifact_declaration,
    build_learned_latent_evidence,
    canonical_json,
    learned_decoder_identity,
    seal_learned_latent_result,
    tensor_bytes_sha256,
    write_task_evidence,
)
from jump_contracts.thought_experiments import (
    EXPERIMENT_SPEC_SCHEMA_SHA256,
    build_experiment_spec,
)

from .authentic_stage_d import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION, freeze_base
from .canonical import sha256_json
from .simulator import derive_seed


SCHEMA_VERSION = "jump.general-world-model-pilot/v1"
ENGINE_ID = "jump.declarative-visual-engine/v2"
FAMILIES = ("force", "diffusion_contagion", "predator_prey", "traffic_queue")
TRAIN_VARIANTS = ("force_attract", "diffusion", "contagion", "predator_dense")
OOD_VARIANTS = ("traffic", "queue")
VARIANT_FAMILY = {
    "force_attract": "force", "force_repulse": "force",
    "diffusion": "diffusion_contagion", "contagion": "diffusion_contagion",
    "predator_dense": "predator_prey", "predator_sparse": "predator_prey",
    "traffic": "traffic_queue", "queue": "traffic_queue",
}
OBSERVED_STEPS = 4
HORIZONS = (1, 2, 4)
MAX_ENTITIES = 8
ENTITY_FEATURES = 8
CONTEXT_DIM = 20
ACTION_DIM = 14
SLOT_DIM = 16
LATENT_SHAPE = (MAX_ENTITIES + 1, SLOT_DIM)
RATE_USD_PER_HOUR = 3.9492
PILOT_SEED = 140826


def manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "general-declarative-visual-world-model-pilot-v1",
        "claim_label": "general visual world-model engineering pilot; no behavioral, causal, or mechanistic claim",
        "contract": {
            "spec_version": "jump.thought-experiment-spec/v2",
            "spec_schema_sha256": EXPERIMENT_SPEC_SCHEMA_SHA256,
            "engine_id": ENGINE_ID,
        },
        "data": {
            "families": list(FAMILIES),
            "train_variants": list(TRAIN_VARIANTS),
            "train_family_ids": sorted({VARIANT_FAMILY[item] for item in TRAIN_VARIANTS}),
            "family_ood_variants": list(OOD_VARIANTS),
            "family_ood_family_ids": sorted({VARIANT_FAMILY[item] for item in OOD_VARIANTS}),
            "train_records": 256,
            "id_records": 48,
            "family_ood_records": 48,
            "train_seed_root": PILOT_SEED,
            "world_seed_disjoint": True,
            "split_before_windows": True,
            "train_only_normalization": True,
            "simulator_role": "offline teacher/ground truth only",
        },
        "input": {
            "allowed": ["visible frames through cutoff", "visible measurement prefix", "validated rule embedding", "pre-outcome intervention embedding"],
            "forbidden": ["future frames", "future measurements", "answer labels", "target-answer prefixes", "simulator runtime handle", "run result"],
        },
        "architecture": {
            "encoder": "shared temporal entity slots plus all-pairs graph message and spec/rule global slot",
            "target": "stop-gradient EMA encoder",
            "predictor": "action/rule-conditioned residual multi-step latent transition",
            "state_decoder": "strict z-only entity/state decoder",
            "raster_decoder": "strict z-only learned 64x64 RGB decoder",
            "gemma_bridge": "nontextual token-conditioned latent memory at frozen layers 7/23/39",
            "latent": {"dtype": "float32-le", "shape": list(LATENT_SHAPE), "order": "C"},
            "same_z_consumers": ["predictor_output", "state_decoder", "raster_decoder", "Gemma_bridge", "cache", "evidence"],
        },
        "training": {
            "world_steps": 320,
            "world_batch_size": 32,
            "world_learning_rate": 0.0007,
            "ema_decay": 0.99,
            "bridge_steps": 36,
            "bridge_learning_rate": 0.0003,
            "base_frozen": True,
            "trainable": ["world encoder", "latent predictor", "z-only decoders", "latent bridge/gates"],
        },
        "evaluation": {
            "splits": ["id", "family_ood"],
            "world_model_gates_per_family": {
                "latent_vs_persistence_improvement_min": 0.20,
                "state_vs_copy_last_improvement_min": 0.20,
                "correct_action_vs_zero_shuffled_wrong_min": 0.10,
            },
            "behavior_gate": "heldout own-z margin and exact answer beat no-z, scrambled-z, wrong-world-z with paired CI lower > 0",
            "all_gates_required_for_product_exposure": True,
        },
        "execution": {
            "modal_function": "general_visual_world_model_pilot",
            "python_version": "3.11",
            "dependency_image": "stage_d_image with exact pinned torch/transformers/jsonschema",
            "resource": "H100",
            "gpu_count": 1,
            "max_containers": 1,
            "max_inputs": 1,
            "max_attempts": 1,
            "timeout_seconds": 5400,
            "h100_rate_usd_per_hour": RATE_USD_PER_HOUR,
            "forecast_usd": 5.9238,
            "aggregate_authority_ceiling_usd": 100.0,
        },
        "claims": {"general_latent": False, "behavioral": False, "causal": False, "mechanistic": False},
    }


MANIFEST_SHA256 = sha256_json(manifest())


def _entity(entity_id: str, label: str, count: int, *, numeric: dict[str, float] | None = None,
            categorical: dict[str, str] | None = None, shape: str = "circle", color: str = "#4f46e5",
            layout: str = "ring", center: tuple[float, float] = (50.0, 50.0), spread: float = 20.0) -> dict[str, Any]:
    return {
        "id": entity_id, "label": label, "count": count,
        "appearance": {"shape": shape, "color": color, "size": 3.0},
        "initial_state": {"numeric": numeric or {}, "categorical": categorical or {}},
        "initial_layout": {"kind": layout, "center": list(center), "spread": spread},
    }


def _base_spec_fields(seed: int, family: str, entities: list[dict[str, Any]], rules: list[dict[str, Any]],
                      intervention: dict[str, Any], measurement: dict[str, Any], graph: str = "none") -> dict[str, Any]:
    return {
        "intent": f"Procedural {family} visual dynamics.",
        "question": "How does the declared intervention change the future visible state?",
        "hypothesis": "The counterfactual changes the declared visible measurement.",
        "world": {
            "bounds": {"width": 100.0, "height": 100.0, "boundary": "wrap"},
            "entities": entities,
            "graph": {"kind": graph, "edge_probability": 0.35 if graph == "random" else 0.0, "directed": False},
        },
        "dynamics": {"rules": rules},
        "conditions": [
            {"id": "baseline", "label": "Baseline", "kind": "baseline", "interventions": []},
            {"id": "changed", "label": "Intervention", "kind": "counterfactual", "interventions": [intervention]},
        ],
        "schedule": {"duration_steps": 12, "dt": 0.25, "seed": seed, "repetitions": 1},
        "measurements": [measurement],
        "visualization": {"kind": "animated_2d", "frame_stride": 1, "max_frames": 13, "chart_measurement_ids": [measurement["id"]]},
    }


def procedural_spec(seed: int, variant: str) -> dict[str, Any]:
    rng = random.Random(derive_seed(seed, f"general-world:{variant}"))
    if variant.startswith("force_"):
        strength = -22.0 if variant == "force_attract" else 18.0
        entities = [_entity("particle", "Particles", 6, numeric={"vx": rng.uniform(-1, 1), "vy": rng.uniform(-1, 1)})]
        rules = [
            {"id": "force", "op": "pairwise_force_2d", "target_type": "particle", "parameters": {"strength": strength, "exponent": 1.5, "softening": 2.0}},
            {"id": "move", "op": "move_2d", "target_type": "particle", "parameters": {"damping": 0.98, "max_speed": 12.0}},
        ]
        intervention = {"time": 4, "operation": "scale_rule_parameter", "target": "force", "field": "strength", "value": -1.0}
        measurement = {"id": "speed", "label": "Mean horizontal speed", "op": "mean_state", "entity_type": "particle", "state": "vx", "category": None}
        fields = _base_spec_fields(seed, "force", entities, rules, intervention, measurement)
    elif variant == "diffusion":
        entities = [
            _entity("hot", "Hot nodes", 4, numeric={"heat": 1.0}, color="#ef4444", layout="line", center=(35, 50), spread=7),
            _entity("cold", "Cold nodes", 4, numeric={"heat": 0.0}, color="#3b82f6", layout="line", center=(65, 50), spread=7),
        ]
        rules = [{"id": "spread", "op": "graph_diffusion", "target_type": None, "parameters": {"state": "heat", "rate": 0.12}}]
        intervention = {"time": 4, "operation": "scale_rule_parameter", "target": "spread", "field": "rate", "value": 2.0}
        measurement = {"id": "heat", "label": "Mean hot-node heat", "op": "mean_state", "entity_type": "hot", "state": "heat", "category": None}
        fields = _base_spec_fields(seed, "diffusion_contagion", entities, rules, intervention, measurement, graph="ring")
    elif variant == "contagion":
        entities = [
            _entity("susceptible", "Susceptible", 7, categorical={"health": "susceptible"}, color="#22c55e"),
            _entity("infected", "Infected", 1, categorical={"health": "infected"}, color="#ef4444"),
        ]
        rules = [{"id": "spread", "op": "graph_contagion", "target_type": None, "parameters": {"state": "health", "susceptible": "susceptible", "infected": "infected", "recovered": "recovered", "transmission_probability": 0.25, "recovery_probability": 0.05}}]
        intervention = {"time": 4, "operation": "set_rule_parameter", "target": "spread", "field": "transmission_probability", "value": 0.7}
        measurement = {"id": "infected", "label": "New infections", "op": "count_category", "entity_type": "susceptible", "state": "health", "category": "infected"}
        fields = _base_spec_fields(seed, "diffusion_contagion", entities, rules, intervention, measurement, graph="ring")
    elif variant.startswith("predator_"):
        radius = 25.0 if variant == "predator_dense" else 10.0
        entities = [
            _entity("prey", "Prey", 6, color="#22c55e", spread=24),
            _entity("predator", "Predators", 2, shape="triangle", color="#ef4444", spread=18),
        ]
        rules = [{"id": "ecology", "op": "predator_prey_2d", "target_type": None, "parameters": {"prey_type": "prey", "predator_type": "predator", "prey_growth": 0.10, "predation_rate": 0.30, "predator_efficiency": 0.10, "predator_decay": 0.03, "interaction_radius": radius}}]
        intervention = {"time": 4, "operation": "set_rule_parameter", "target": "ecology", "field": "predation_rate", "value": 0.75}
        measurement = {"id": "prey_count", "label": "Prey population", "op": "population_count", "entity_type": "prey", "state": None, "category": None}
        fields = _base_spec_fields(seed, "predator_prey", entities, rules, intervention, measurement)
    elif variant == "traffic":
        entities = [_entity("car", "Cars", 8, numeric={"speed": 0.0}, shape="square", color="#f59e0b", layout="line", center=(50, 50), spread=8)]
        rules = [{"id": "traffic", "op": "lane_traffic_2d", "target_type": "car", "parameters": {"speed_state": "speed", "desired_speed": 8.0, "headway": 7.0, "road_y": 50.0}}]
        intervention = {"time": 4, "operation": "set_rule_parameter", "target": "traffic", "field": "desired_speed", "value": 15.0}
        measurement = {"id": "speed", "label": "Mean traffic speed", "op": "mean_state", "entity_type": "car", "state": "speed", "category": None}
        fields = _base_spec_fields(seed, "traffic_queue", entities, rules, intervention, measurement)
    elif variant == "queue":
        entities = [_entity("agent", "Queue agents", 8, categorical={"queue_status": "waiting"}, shape="square", color="#8b5cf6", layout="line", center=(40, 50), spread=4)]
        rules = [{"id": "queue", "op": "queue_agents_2d", "target_type": "agent", "parameters": {"arrival_probability": 0.45, "service_capacity": 1, "queue_x": 60.0, "service_x": 90.0}}]
        intervention = {"time": 4, "operation": "set_rule_parameter", "target": "queue", "field": "service_capacity", "value": 3}
        measurement = {"id": "queued", "label": "Queued agents", "op": "count_category", "entity_type": "agent", "state": "queue_status", "category": "queued"}
        fields = _base_spec_fields(seed, "traffic_queue", entities, rules, intervention, measurement)
    else:
        raise ValueError("unsupported procedural visual family variant")
    return build_experiment_spec(**fields)


_RULE_INDEX = {name: index for index, name in enumerate(("move_2d", "pairwise_force_2d", "graph_diffusion", "graph_contagion", "predator_prey_2d", "lane_traffic_2d", "queue_agents_2d", "random_walk_2d"))}
_CATEGORY = {None: 0.0, "susceptible": 0.15, "infected": 0.35, "recovered": 0.55, "waiting": 0.65, "queued": 0.8, "served": 1.0}


def _context(spec: dict[str, Any], family: str) -> list[float]:
    value = [0.0] * CONTEXT_DIM
    value[FAMILIES.index(family)] = 1.0
    for rule in spec["dynamics"]["rules"]:
        value[4 + _RULE_INDEX[rule["op"]]] = 1.0
    graph_index = {"none": 0, "ring": 1, "grid": 2, "random": 3}[spec["world"]["graph"]["kind"]]
    value[12 + graph_index] = 1.0
    numeric = [float(v) for rule in spec["dynamics"]["rules"] for v in rule["parameters"].values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for index, number in enumerate(numeric[:4]):
        value[16 + index] = math.tanh(number / 20.0)
    return value


def _action(spec: dict[str, Any], condition_id: str) -> list[float]:
    value = [0.0] * ACTION_DIM
    if condition_id == "baseline":
        return value
    change = next(item for item in spec["conditions"] if item["id"] == condition_id)["interventions"][0]
    op_index = {"set_rule_parameter": 0, "scale_rule_parameter": 1, "set_numeric_state": 2, "set_categorical_state": 3}[change["operation"]]
    value[op_index] = 1.0
    value[4] = change["time"] / spec["schedule"]["duration_steps"]
    value[5] = math.tanh(float(change["value"]) / 10.0) if isinstance(change["value"], (int, float)) else 0.5
    rule = next((item for item in spec["dynamics"]["rules"] if item["id"] == change["target"]), None)
    if rule is not None:
        value[6 + _RULE_INDEX[rule["op"]]] = 1.0
    return value


def _frame_tensor(spec: dict[str, Any], frame: dict[str, Any], previous: dict[str, Any] | None) -> tuple[list[list[float]], list[float]]:
    current = {item["entity_id"]: item for item in frame["points"]}
    prior = {item["entity_id"]: item for item in previous["points"]} if previous else current
    declared = [f"{entity['id']}-{index}" for entity in spec["world"]["entities"] for index in range(entity["count"])]
    slots: list[list[float]] = []
    mask: list[float] = []
    for slot, entity_id in enumerate(declared[:MAX_ENTITIES]):
        point = current.get(entity_id)
        old = prior.get(entity_id, point)
        if point is None:
            slots.append([0.0] * ENTITY_FEATURES); mask.append(0.0); continue
        width = spec["world"]["bounds"]["width"]
        height = spec["world"]["bounds"]["height"]
        type_ids = [item["id"] for item in spec["world"]["entities"]]
        slots.append([
            point["x"] / width, point["y"] / height,
            (point["x"] - old["x"]) / width, (point["y"] - old["y"]) / height,
            1.0, _CATEGORY.get(point["category"], 0.05),
            type_ids.index(point["type_id"]) / max(1, len(type_ids) - 1), point["size"] / 20.0,
        ]); mask.append(1.0)
    while len(slots) < MAX_ENTITIES:
        slots.append([0.0] * ENTITY_FEATURES); mask.append(0.0)
    return slots, mask


def _record(seed: int, variant: str, split: str, changed: bool) -> dict[str, Any]:
    from jump_workbench.visual_engine import execute_visual_spec

    spec = procedural_spec(seed, variant)
    executed = execute_visual_spec(spec)
    condition_id = "changed" if changed else "baseline"
    result = next(item for item in executed["conditions"] if item["condition_id"] == condition_id)
    family = VARIANT_FAMILY[variant]
    frames = result["frames"]
    tensors, masks = [], []
    for index, frame in enumerate(frames):
        state, mask = _frame_tensor(spec, frame, frames[index - 1] if index else None)
        tensors.append(state); masks.append(mask)
    series = [math.tanh(item["value"] / 20.0) for item in result["series"][0]["values"]]
    cutoff = OBSERVED_STEPS - 1
    target_indices = [cutoff + horizon for horizon in HORIZONS]
    windows = [tensors[max(0, index - OBSERVED_STEPS + 1):index + 1] for index in target_indices]
    window_masks = [masks[max(0, index - OBSERVED_STEPS + 1):index + 1] for index in target_indices]
    for index in range(len(windows)):
        while len(windows[index]) < OBSERVED_STEPS:
            windows[index].insert(0, windows[index][0]); window_masks[index].insert(0, window_masks[index][0])
    comparison = next(item for item in executed["comparisons"] if item["counterfactual_condition_id"] == "changed")
    context = _context(spec, family)
    context[-2] = series[cutoff]
    context[-1] = series[cutoff] - series[max(0, cutoff - 1)]
    world_manifest = {
        "schema_version": "jump.visual-world-manifest/v1", "family_id": family,
        "world": spec["world"], "dynamics": spec["dynamics"],
        "duration_steps": spec["schedule"]["duration_steps"], "dt": spec["schedule"]["dt"],
        "seed": spec["schedule"]["seed"],
    }
    return {
        "split": split, "family": family, "variant": variant,
        "spec_sha256": spec["spec_sha256"], "world_sha256": hashlib.sha256(canonical_json(world_manifest)).hexdigest(),
        "observed": tensors[:OBSERVED_STEPS], "observed_mask": masks[:OBSERVED_STEPS],
        "context": context, "action": _action(spec, condition_id),
        "target_windows": windows, "target_window_masks": window_masks,
        "target_states": [tensors[index] for index in target_indices],
        "target_masks": [masks[index] for index in target_indices],
        "target_series": [[series[index]] for index in target_indices],
        "last_state": tensors[cutoff],
        "answer_higher": bool(comparison["difference"] > 0),
    }


def dataset(split: str, count: int) -> dict[str, Any]:
    if split not in {"train", "id", "family_ood"}:
        raise ValueError("invalid general world-model split")
    variants = OOD_VARIANTS if split == "family_ood" else TRAIN_VARIANTS
    roots = {"train": PILOT_SEED, "id": PILOT_SEED + 1, "family_ood": PILOT_SEED + 2}
    seeds = []
    records = []
    for index in range(count):
        variant = variants[index % len(variants)]; family = VARIANT_FAMILY[variant]
        preimage = {"schema_version": "jump.visual-world-seed/v1", "root_seed": roots[split], "family_id": family, "world_index": index}
        digest = hashlib.sha256(canonical_json(preimage)).digest()
        seed = int.from_bytes(digest[:8], "big") % 2147483648
        seeds.append(seed); records.append(_record(seed, variant, split, bool(index % 2)))
    split_manifest = {"schema_version": "jump.visual-world-split/v1", "split": split, "family_ids": sorted({record["family"] for record in records}), "world_sha256": sorted(record["world_sha256"] for record in records)}
    return {"records": records, "seed_set_sha256": sha256_json(seeds), "dataset_sha256": sha256_json(records), "split_manifest": split_manifest, "split_manifest_sha256": sha256_json(split_manifest)}


def _tensorize(records: list[dict[str, Any]], device: str) -> dict[str, Any]:
    import torch
    keys = ("observed", "observed_mask", "context", "action", "target_windows", "target_window_masks", "target_states", "target_masks", "target_series", "last_state")
    value = {key: torch.tensor([record[key] for record in records], dtype=torch.float32, device=device) for key in keys}
    value["family"] = torch.tensor([FAMILIES.index(record["family"]) for record in records], dtype=torch.long, device=device)
    value["answer_higher"] = torch.tensor([record["answer_higher"] for record in records], dtype=torch.long, device=device)
    return value


def build_modules():
    import torch

    class GeneralEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = torch.nn.Sequential(torch.nn.Linear(OBSERVED_STEPS * (ENTITY_FEATURES + 1), 64), torch.nn.GELU(), torch.nn.Linear(64, SLOT_DIM))
            self.message = torch.nn.Sequential(torch.nn.Linear(SLOT_DIM * 2 + 4, 48), torch.nn.GELU(), torch.nn.Linear(48, SLOT_DIM))
            self.update = torch.nn.Sequential(torch.nn.Linear(SLOT_DIM * 2, 48), torch.nn.GELU(), torch.nn.Linear(48, SLOT_DIM))
            self.global_slot = torch.nn.Sequential(torch.nn.Linear(CONTEXT_DIM + SLOT_DIM, 48), torch.nn.GELU(), torch.nn.Linear(48, SLOT_DIM))
        def forward(self, observed, mask, context):
            masked = observed * mask[..., None]
            temporal = torch.cat([masked, mask[..., None]], -1).permute(0, 2, 1, 3).reshape(observed.shape[0], MAX_ENTITIES, -1)
            slots = self.temporal(temporal)
            latest = observed[:, -1]
            left = slots[:, :, None].expand(-1, -1, MAX_ENTITIES, -1)
            right = slots[:, None, :].expand(-1, MAX_ENTITIES, -1, -1)
            rel = latest[:, None, :, :4] - latest[:, :, None, :4]
            messages = self.message(torch.cat([left, right, rel], -1))
            valid = mask[:, -1]
            pair_mask = valid[:, :, None] * valid[:, None, :]
            eye = torch.eye(MAX_ENTITIES, device=observed.device, dtype=observed.dtype)[None]
            aggregate = (messages * pair_mask[..., None] * (1 - eye[..., None])).sum(2)
            slots = slots + self.update(torch.cat([slots, aggregate], -1))
            pooled = (slots * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            global_slot = self.global_slot(torch.cat([context, pooled], -1))[:, None]
            return torch.cat([slots, global_slot], 1)

    class LatentPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action = torch.nn.Sequential(torch.nn.Linear(ACTION_DIM + CONTEXT_DIM + len(HORIZONS), 64), torch.nn.GELU(), torch.nn.Linear(64, SLOT_DIM * 2))
            self.transition = torch.nn.Sequential(torch.nn.Linear(SLOT_DIM * 3, 64), torch.nn.GELU(), torch.nn.Linear(64, SLOT_DIM))
        def forward(self, z, action, context, horizon_index):
            onehot = torch.nn.functional.one_hot(horizon_index, len(HORIZONS)).to(z.dtype)
            film = self.action(torch.cat([action, context, onehot], -1))
            scale, shift = film.chunk(2, -1)
            global_slot = z[:, -1:].expand(-1, z.shape[1], -1)
            conditioned = z * (1 + 0.1 * torch.tanh(scale[:, None])) + shift[:, None]
            delta = self.transition(torch.cat([z, conditioned, global_slot], -1))
            return z + delta

    class StateDecoder(torch.nn.Module):
        """Strict z-only state decoder."""
        def __init__(self):
            super().__init__()
            self.entity = torch.nn.Sequential(torch.nn.Linear(SLOT_DIM * 2, 64), torch.nn.GELU(), torch.nn.Linear(64, ENTITY_FEATURES))
            self.series = torch.nn.Sequential(torch.nn.Linear(SLOT_DIM, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
        def forward(self, z):
            global_slot = z[:, -1:].expand(-1, MAX_ENTITIES, -1)
            return self.entity(torch.cat([z[:, :MAX_ENTITIES], global_slot], -1)), self.series(z[:, -1])

    class RasterDecoder(torch.nn.Module):
        """Strict z-only learned raster decoder; no observation/simulator argument."""
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(math.prod(LATENT_SHAPE), 128 * 4 * 4)
            self.up = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 96, 4, 2, 1), torch.nn.GELU(),
                torch.nn.ConvTranspose2d(96, 64, 4, 2, 1), torch.nn.GELU(),
                torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), torch.nn.GELU(),
                torch.nn.ConvTranspose2d(32, 3, 4, 2, 1), torch.nn.Sigmoid(),
            )
        def forward(self, z):
            return self.up(self.fc(z.reshape(z.shape[0], -1)).reshape(-1, 128, 4, 4))
    return GeneralEncoder(), LatentPredictor(), StateDecoder(), RasterDecoder()


def build_latent_bridge(hidden_size: int):
    import torch
    layers = (7, 23, 39)
    class Bridge(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory = torch.nn.Linear(SLOT_DIM, 16)
            self.q = torch.nn.ModuleDict({str(layer): torch.nn.Linear(hidden_size, 16, bias=False) for layer in layers})
            self.k = torch.nn.ModuleDict({str(layer): torch.nn.Linear(16, 16, bias=False) for layer in layers})
            self.v = torch.nn.ModuleDict({str(layer): torch.nn.Linear(16, 16, bias=False) for layer in layers})
            self.out = torch.nn.ModuleDict({str(layer): torch.nn.Linear(16, hidden_size, bias=False) for layer in layers})
            self.gates = torch.nn.ParameterDict({str(layer): torch.nn.Parameter(torch.tensor(-2.0)) for layer in layers})
        def inject(self, layer, hidden, z):
            memory = self.memory(z.to(next(self.parameters()).dtype)).to(hidden.dtype)
            attention = torch.softmax(torch.einsum("btd,bmd->btm", self.q[str(layer)](hidden), self.k[str(layer)](memory)) / 4.0, -1)
            residual = self.out[str(layer)](torch.einsum("btm,bmd->btd", attention, self.v[str(layer)](memory)))
            return hidden + torch.sigmoid(self.gates[str(layer)]) * residual
    return Bridge()


def _exact_z(z: Any) -> tuple[Any, bytes, str]:
    import numpy as np
    import torch
    raw = z.detach().to("cpu", torch.float32).contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
    roundtrip = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(z.shape)).to(z.device)
    exact = z + (roundtrip - z).detach()
    return exact, raw, tensor_bytes_sha256(raw, dtype="float32-le", shape=list(z.shape), order="C")


def _raster(states: Any) -> Any:
    import torch
    grid = torch.linspace(0, 1, 64, device=states.device, dtype=states.dtype)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    pos = states[..., :2].clamp(0, 1)
    alive = states[..., 4].clamp(0, 1)
    dx = xx[None, None] - pos[..., 0, None, None]
    dy = yy[None, None] - pos[..., 1, None, None]
    blobs = torch.exp(-(dx * dx + dy * dy) / (2 * 0.025**2)) * alive[..., None, None]
    type_code = states[..., 6].clamp(0, 1)
    colors = torch.stack([type_code, 1 - type_code, 0.35 + 0.4 * states[..., 5].clamp(0, 1)], -1)
    return (blobs[..., None, :, :] * colors[..., :, None, None]).sum(1).clamp(0, 1)


def _png(frame: Any) -> bytes:
    from PIL import Image
    array = (frame.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().to(dtype=__import__("torch").uint8).numpy()
    stream = io.BytesIO(); Image.fromarray(array, "RGB").save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _ci(values: list[float], seed: int, draws: int = 2000) -> list[float]:
    rng = random.Random(seed); samples = []
    for _ in range(draws):
        samples.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    samples.sort(); return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def _train_world(train: dict[str, Any], device: str, steps: int):
    import torch
    import torch.nn.functional as F
    online, predictor, state_decoder, raster_decoder = build_modules()
    online.to(device); predictor.to(device); state_decoder.to(device); raster_decoder.to(device)
    target = deepcopy(online).to(device).eval()
    for parameter in target.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()) + list(state_decoder.parameters()) + list(raster_decoder.parameters()), lr=manifest()["training"]["world_learning_rate"])
    losses = []; count = train["observed"].shape[0]; batch_size = min(manifest()["training"]["world_batch_size"], count)
    for step in range(steps):
        index = torch.arange(step * batch_size, (step + 1) * batch_size, device=device) % count
        z0 = online(train["observed"][index], train["observed_mask"][index], train["context"][index])
        predictions = []
        for horizon_index in range(len(HORIZONS)):
            predictions.append(predictor(z0, train["action"][index], train["context"][index], torch.full((batch_size,), horizon_index, device=device)))
        predicted = torch.stack(predictions, 1)
        windows = train["target_windows"][index].reshape(batch_size * len(HORIZONS), OBSERVED_STEPS, MAX_ENTITIES, ENTITY_FEATURES)
        window_masks = train["target_window_masks"][index].reshape(batch_size * len(HORIZONS), OBSERVED_STEPS, MAX_ENTITIES)
        context = train["context"][index][:, None].expand(-1, len(HORIZONS), -1).reshape(batch_size * len(HORIZONS), -1)
        with torch.no_grad(): target_z = target(windows, window_masks, context).reshape(batch_size, len(HORIZONS), *LATENT_SHAPE)
        latent_loss = F.smooth_l1_loss(predicted, target_z)
        flat_z = predicted.detach().reshape(batch_size * len(HORIZONS), *LATENT_SHAPE)
        state, series = state_decoder(flat_z)
        target_state = train["target_states"][index].reshape(batch_size * len(HORIZONS), MAX_ENTITIES, ENTITY_FEATURES)
        target_series = train["target_series"][index].reshape(batch_size * len(HORIZONS), 1)
        state_loss = F.smooth_l1_loss(state, target_state) + 0.25 * F.smooth_l1_loss(series, target_series)
        raster_loss = F.smooth_l1_loss(raster_decoder(flat_z), _raster(target_state))
        learned = z0.reshape(batch_size, -1)
        variance = torch.relu(0.25 - learned.std(0)).mean()
        loss = latent_loss + 0.1 * variance + 0.25 * state_loss + 0.05 * raster_loss
        if not torch.isfinite(loss): raise RuntimeError("non-finite general world-model loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
                target_parameter.mul_(0.99).add_(online_parameter, alpha=0.01)
    return (online.eval(), predictor.eval(), state_decoder.eval(), raster_decoder.eval(), target.eval()), losses


def _evaluate_world(modules: tuple[Any, ...], data: dict[str, Any], records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    import torch
    online, predictor, state_decoder, raster_decoder, target = modules
    count = data["observed"].shape[0]
    with torch.no_grad():
        z0 = online(data["observed"], data["observed_mask"], data["context"])
        predictions = lambda action: torch.stack([predictor(z0, action, data["context"], torch.full((count,), index, device=z0.device)) for index in range(len(HORIZONS))], 1)
        correct = predictions(data["action"]); zero = predictions(torch.zeros_like(data["action"])); shuffled = predictions(data["action"].roll(1, 0)); wrong = predictions(-data["action"])
        windows = data["target_windows"].reshape(count * len(HORIZONS), OBSERVED_STEPS, MAX_ENTITIES, ENTITY_FEATURES)
        masks = data["target_window_masks"].reshape(count * len(HORIZONS), OBSERVED_STEPS, MAX_ENTITIES)
        context = data["context"][:, None].expand(-1, len(HORIZONS), -1).reshape(count * len(HORIZONS), -1)
        target_z = target(windows, masks, context).reshape(count, len(HORIZONS), *LATENT_SHAPE)
        scale = target_z.std((0, 1)).clamp_min(1e-5)
        latent_error = lambda z: (((z - target_z) / scale).pow(2).mean((2, 3))).sqrt().mean(1)
        errors = {"model": latent_error(correct), "persistence": latent_error(z0[:, None].expand_as(target_z)), "zero": latent_error(zero), "shuffled": latent_error(shuffled), "wrong": latent_error(wrong)}
        exact_z, raw, z_hash = _exact_z(correct.reshape(count * len(HORIZONS), *LATENT_SHAPE))
        decoded, decoded_series = state_decoder(exact_z)
        target_states = data["target_states"].reshape(count * len(HORIZONS), MAX_ENTITIES, ENTITY_FEATURES)
        state_scale = target_states.std((0, 1)).clamp_min(1e-5)
        state_error = (((decoded - target_states) / state_scale).pow(2).mean((1, 2))).sqrt().reshape(count, len(HORIZONS)).mean(1)
        copy = data["last_state"][:, None].expand_as(data["target_states"]).reshape_as(target_states)
        copy_error = (((copy - target_states) / state_scale).pow(2).mean((1, 2))).sqrt().reshape(count, len(HORIZONS)).mean(1)
        raster = raster_decoder(exact_z); raster_target = _raster(target_states)
        pixel_l1 = (raster - raster_target).abs().mean((1, 2, 3)).reshape(count, len(HORIZONS)).mean(0)
        pixel_mse = (raster - raster_target).pow(2).mean((1, 2, 3)).reshape(count, len(HORIZONS)).mean(0)
        psnr = 10 * torch.log10(1 / pixel_mse.clamp_min(1e-12))
    by_family = {}
    present_families = sorted({record["family"] for record in records})
    for family_index, family in enumerate(present_families):
        selected = [index for index, record in enumerate(records) if record["family"] == family]
        latent_gain = [float((errors["persistence"][i] - errors["model"][i]) / errors["persistence"][i].clamp_min(1e-8)) for i in selected]
        state_gain = [float((copy_error[i] - state_error[i]) / copy_error[i].clamp_min(1e-8)) for i in selected]
        controls = {}
        for offset, name in enumerate(("zero", "shuffled", "wrong")):
            gain = [float((errors[name][i] - errors["model"][i]) / errors[name][i].clamp_min(1e-8)) for i in selected]
            controls[name] = {"mean": sum(gain) / len(gain), "ci95": _ci(gain, 40000 + family_index * 10 + offset)}
        by_family[family] = {
            "n": len(selected),
            "latent_vs_persistence": {"mean": sum(latent_gain) / len(latent_gain), "ci95": _ci(latent_gain, 41000 + family_index)},
            "state_vs_copy_last": {"mean": sum(state_gain) / len(state_gain), "ci95": _ci(state_gain, 42000 + family_index)},
            "action": controls,
        }
    sample_raw = raw[: math.prod(LATENT_SHAPE) * 4]
    sample_hash = tensor_bytes_sha256(sample_raw, dtype="float32-le", shape=list(LATENT_SHAPE), order="C")
    sample_raster = raster[0].cpu()
    return {
        "split": split, "n": count, "families": by_family,
        "pixel": {"horizons": list(HORIZONS), "l1": [float(v) for v in pixel_l1.cpu()], "psnr": [float(v) for v in psnr.cpu()]},
        "sample_z_raw": sample_raw, "sample_z_sha256": sample_hash, "batch_z_sha256": z_hash,
        "sample_raster": sample_raster, "sample_state": decoded[0].cpu(),
    }


def _world_gates(evaluation: dict[str, Any]) -> bool:
    for split in ("id", "family_ood"):
        for value in evaluation[split]["families"].values():
            if value["latent_vs_persistence"]["mean"] < 0.20 or value["latent_vs_persistence"]["ci95"][0] <= 0: return False
            if value["state_vs_copy_last"]["mean"] < 0.20 or value["state_vs_copy_last"]["ci95"][0] <= 0: return False
            if any(item["mean"] < 0.10 or item["ci95"][0] <= 0 for item in value["action"].values()): return False
    return True


def _injection(model: Any, bridge: Any, z: Any, enabled: bool):
    from contextlib import contextmanager
    @contextmanager
    def manager():
        if not enabled:
            yield; return
        modules = dict(model.named_modules()); handles = []
        for layer in (7, 23, 39):
            name = f"model.language_model.layers.{layer}"
            if name not in modules: raise RuntimeError(f"missing frozen Gemma layer {name}")
            def hook(_module, args, kwargs, layer=layer):
                hidden = args[0] if args else kwargs.get("hidden_states")
                changed = bridge.inject(layer, hidden, z.expand(hidden.shape[0], -1, -1).to(hidden.device))
                return ((changed, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": changed})
            handles.append(modules[name].register_forward_pre_hook(hook, with_kwargs=True))
        try: yield
        finally:
            for handle in handles: handle.remove()
    return manager()


def _choice_ids(tokenizer: Any) -> tuple[int, int]:
    values = []
    for choice in ("A", "B"):
        ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
        if len(ids) != 1: raise RuntimeError("structured choice is not one token")
        values.append(ids[0])
    return values[0], values[1]


def _student_logits(model: Any, tokenizer: Any, bridge: Any, z: Any, prompt: str, enabled: bool) -> Any:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"].to(z.device); mask = encoded["attention_mask"].to(z.device)
    with _injection(model, bridge, z, enabled): return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1]


def _behavior_prompt(family: str) -> str:
    return f"A validated {family} visual experiment compares a baseline with a declared intervention. Will its declared final measurement be higher under the intervention? Answer A for yes or B for no."


def _teacher_prompt(family: str, state: Any, series: Any) -> str:
    summary = {"predicted_mean_x": round(float(state[..., 0].mean()), 6), "predicted_mean_y": round(float(state[..., 1].mean()), 6), "predicted_alive": round(float(state[..., 4].sum()), 6), "predicted_measurement": round(float(series.mean()), 6)}
    return "World-model predicted rollout (positions/motion only):" + json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n" + _behavior_prompt(family)


def _train_bridge(modules: tuple[Any, ...], train: dict[str, Any], train_records: list[dict[str, Any]], heldout: dict[str, Any], heldout_records: list[dict[str, Any]], device: str):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    online, predictor, state_decoder, _raster_decoder, _target = modules
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION, torch_dtype=torch.bfloat16, trust_remote_code=False).to(device)
    base_parameters = freeze_base(model); model.eval(); model.config.use_cache = False
    bridge = build_latent_bridge(int(model.config.text_config.hidden_size)).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=manifest()["training"]["bridge_learning_rate"])
    choice = torch.tensor(_choice_ids(tokenizer), device=device)
    losses = []
    for step in range(manifest()["training"]["bridge_steps"]):
        index = step % train["observed"].shape[0]
        with torch.no_grad():
            z0 = online(train["observed"][index:index+1], train["observed_mask"][index:index+1], train["context"][index:index+1])
            predicted = predictor(z0, train["action"][index:index+1], train["context"][index:index+1], torch.tensor([len(HORIZONS)-1], device=device))
            exact, _, _ = _exact_z(predicted)
            state, series = state_decoder(exact)
            encoded = tokenizer(_teacher_prompt(train_records[index]["family"], state, series), return_tensors="pt", add_special_tokens=True)
            teacher = model(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device), use_cache=False).logits[:, -1, choice].float()
        student = _student_logits(model, tokenizer, bridge, exact, _behavior_prompt(train_records[index]["family"]), True)[:, choice].float()
        loss = F.kl_div(F.log_softmax(student / 2, -1), F.softmax(teacher / 2, -1), reduction="batchmean") * 4 + 0.25 * F.mse_loss(student[:, 0] - student[:, 1], teacher[:, 0] - teacher[:, 1])
        if not torch.isfinite(loss): raise RuntimeError("non-finite general latent bridge loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("frozen Gemma became trainable")
    records = []; bridge.eval()
    with torch.no_grad():
        z0 = online(heldout["observed"], heldout["observed_mask"], heldout["context"])
        own = predictor(z0, heldout["action"], heldout["context"], torch.full((z0.shape[0],), len(HORIZONS)-1, device=device))
    for index, record in enumerate(heldout_records):
        own_z, raw, digest = _exact_z(own[index:index + 1])
        generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(PILOT_SEED, f"bridge-scramble:{index}")
        )
        permutation = torch.randperm(own_z.shape[1], generator=generator).to(device)
        scrambled = own_z[:, permutation]
        wrong_index = (index + 4) % own.shape[0]
        wrong, _, _ = _exact_z(own[wrong_index:wrong_index + 1])
        arms = {"own_z": (own_z, True), "no_z": (own_z, False), "scrambled_z": (scrambled, True), "wrong_world_z": (wrong, True)}
        truth = int(record["answer_higher"]); correct_choice = 0 if truth else 1
        values = {}
        for name, (latent, enabled) in arms.items():
            logits = _student_logits(model, tokenizer, bridge, latent, _behavior_prompt(record["family"]), enabled)[:, choice].float()[0]
            values[name] = {"margin": float((logits[correct_choice] - logits[1-correct_choice]).cpu()), "exact": float(int(logits.argmax().item()) == correct_choice), "parse": 1.0}
        records.append({"family": record["family"], "world_latent_sha256": digest, "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(), "arms": values})
    comparisons = {}
    for offset, control in enumerate(("no_z", "scrambled_z", "wrong_world_z")):
        margin = [item["arms"]["own_z"]["margin"] - item["arms"][control]["margin"] for item in records]
        exact = [item["arms"]["own_z"]["exact"] - item["arms"][control]["exact"] for item in records]
        comparisons[control] = {"margin_mean": sum(margin)/len(margin), "margin_ci95": _ci(margin, 50000+offset), "exact_mean": sum(exact)/len(exact), "exact_ci95": _ci(exact, 51000+offset), "parse_delta": 0.0}
    passed = all(value["margin_ci95"][0] > 0 and value["exact_ci95"][0] > 0 and abs(value["parse_delta"]) < 0.02 for value in comparisons.values())
    prompt_hashes = {
        family: hashlib.sha256(canonical_json(tokenizer(_behavior_prompt(family), add_special_tokens=True)["input_ids"])).hexdigest()
        for family in FAMILIES
    }
    return bridge, {"initial_loss": losses[0], "final_loss": losses[-1], "comparisons": comparisons, "passed": passed, "records": records, "base_parameters": base_parameters, "trainable_parameters": sum(parameter.numel() for parameter in bridge.parameters()), "prompt_token_hashes": prompt_hashes, "prompt_tokens_identical_across_arms": True}


def cpu_preflight() -> dict[str, Any]:
    import torch
    first = dataset("train", 4)
    batch = _tensorize(first["records"], "cpu")
    modules, losses = _train_world(batch, "cpu", 12)
    online, predictor, state_decoder, raster_decoder, _target = modules
    with torch.no_grad():
        z0 = online(batch["observed"], batch["observed_mask"], batch["context"])
        predicted = predictor(z0, batch["action"], batch["context"], torch.zeros(4, dtype=torch.long))
        exact, raw, digest = _exact_z(predicted[0])
        exact_batch = exact.unsqueeze(0)
        state_decoder(exact_batch); first_image = raster_decoder(exact_batch); second_image = raster_decoder(exact_batch.clone())
    try:
        state_decoder(exact_batch, observation=batch["observed"]); raise AssertionError("z-only state decoder accepted side channel")
    except TypeError: pass
    try:
        raster_decoder(exact_batch, simulator="forbidden"); raise AssertionError("z-only raster decoder accepted side channel")
    except TypeError: pass
    if not torch.equal(first_image, second_image): raise RuntimeError("same z did not reproduce identical raster")
    return {"status": "passed", "spec_schema_sha256": EXPERIMENT_SPEC_SCHEMA_SHA256, "dataset_sha256": first["dataset_sha256"], "split_manifest_sha256": first["split_manifest_sha256"], "families": sorted({record["family"] for record in first["records"]}), "initial_loss": losses[0], "final_loss": losses[-1], "tiny_overfit_improved": losses[-1] < losses[0], "latent_shape": list(exact.shape), "world_latent_sha256": digest, "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(), "same_z_raster_reproducible": True, "z_only_decoders": True}


def train_and_evaluate(output_root: Path, expected_manifest_sha256: str, expected_code_sha: str, device: str = "cuda") -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    if expected_manifest_sha256 != MANIFEST_SHA256 or os.environ.get("JUMP_CODE_VERSION") != expected_code_sha: raise ValueError("general world-model identity mismatch")
    if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()): raise FileExistsError("general world-model requires empty canonical workdir")
    if not torch.cuda.is_available(): raise RuntimeError("general world-model pilot requires CUDA")
    torch.manual_seed(PILOT_SEED); torch.cuda.manual_seed_all(PILOT_SEED); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    train_data = dataset("train", manifest()["data"]["train_records"]); id_data = dataset("id", manifest()["data"]["id_records"]); ood_data = dataset("family_ood", manifest()["data"]["family_ood_records"])
    train = _tensorize(train_data["records"], device); id_batch = _tensorize(id_data["records"], device); ood_batch = _tensorize(ood_data["records"], device)
    modules, world_losses = _train_world(train, device, manifest()["training"]["world_steps"])
    id_eval = _evaluate_world(modules, id_batch, id_data["records"], "id"); ood_eval = _evaluate_world(modules, ood_batch, ood_data["records"], "family_ood")
    world_passed = _world_gates({"id": id_eval, "family_ood": ood_eval})
    bridge, behavior = _train_bridge(modules, train, train_data["records"], id_batch, id_data["records"], device)
    product_ready = world_passed and behavior["passed"]
    online, predictor, state_decoder, raster_decoder, _target = modules
    for role, module in (("encoder", online), ("latent_predictor", predictor), ("state_decoder", state_decoder), ("raster_decoder", raster_decoder), ("gemma_bridge", bridge)):
        root = output_root / role; root.mkdir()
        save_file({key: value.detach().cpu() for key, value in module.state_dict().items()}, root / "model.safetensors")
        (root / "config.json").write_bytes(canonical_json({"architecture": role, "latent_shape": list(LATENT_SHAPE), "strict_z_only": role in {"state_decoder", "raster_decoder"}, "engineering_only": True}))
    (output_root / "manifest.json").write_bytes(canonical_json(manifest()))
    sample_raw = id_eval.pop("sample_z_raw"); sample_raster = id_eval.pop("sample_raster"); id_eval.pop("sample_state")
    ood_eval.pop("sample_z_raw"); ood_eval.pop("sample_raster"); ood_eval.pop("sample_state")
    (output_root / "sample-world-latent.f32le.bin").write_bytes(sample_raw)
    image = _png(sample_raster); (output_root / "sample-learned-prediction.png").write_bytes(image)
    evaluation = {"world_model": {"id": id_eval, "family_ood": ood_eval, "passed": world_passed}, "Gemma_behavior": {key: value for key, value in behavior.items() if key != "records"}}
    (output_root / "evaluation.json").write_bytes(canonical_json(evaluation))
    (output_root / "behavior-records.json").write_bytes(canonical_json(behavior["records"]))
    encoder_weights = output_root / "encoder/model.safetensors"; decoder_weights = output_root / "raster_decoder/model.safetensors"; bridge_weights = output_root / "gemma_bridge/model.safetensors"
    observation_bytes = torch.tensor(id_data["records"][0]["observed"], dtype=torch.float32).numpy().astype("<f4").tobytes()
    (output_root / "sample-observation.f32le.bin").write_bytes(observation_bytes)
    computed_claims = {"general_latent": product_ready, "behavioral": product_ready, "causal": False, "mechanistic": False}
    evidence = build_learned_latent_evidence(
        encoder_output=sample_raw, decoder_input=bytes(sample_raw), injection_input=memoryview(sample_raw),
        encoder_observation=observation_bytes,
        encoder_observation_artifact_name="sample-observation.f32le.bin", encoder_observation_media_type="application/octet-stream",
        dtype="float32-le", shape=list(LATENT_SHAPE), order="C", recipient_world_id="heldout-id-0", world_pair_id="general-world-heldout-0",
        learned_decoder=learned_decoder_identity(artifact_name="raster_decoder/model.safetensors", artifact_sha256=hashlib.sha256(decoder_weights.read_bytes()).hexdigest(), training_manifest_sha256=MANIFEST_SHA256, code_version=expected_code_sha, architecture="general-v2-z-only-raster-decoder-v1"),
        decoded_image=image, decoded_image_media_type="image/png",
        answer={"engineering_only": True, "product_ready": product_ready, "claims": computed_claims, "behavior": behavior["comparisons"]},
        tensor_artifact_name="sample-world-latent.f32le.bin",
    )
    checkpoint_id = hashlib.sha256(encoder_weights.read_bytes() + decoder_weights.read_bytes() + bridge_weights.read_bytes()).hexdigest()
    sealed = seal_learned_latent_result(evidence, source="cached", manifest_sha256=MANIFEST_SHA256, run_id="general-world-model-pilot", code_version=expected_code_sha, checkpoint_id=checkpoint_id)
    (output_root / "sample-sealed-result.json").write_bytes(canonical_json(sealed))
    duration = time.monotonic() - started
    terminal = {"status": "completed", "decision": "pass" if product_ready else "pivot", "product_exposure_allowed": product_ready, "manifest_sha256": MANIFEST_SHA256, "code_sha": expected_code_sha, "checkpoint_id": checkpoint_id, "dataset_hashes": {split: value["dataset_sha256"] for split, value in (("train", train_data), ("id", id_data), ("family_ood", ood_data))}, "seed_set_hashes": {split: value["seed_set_sha256"] for split, value in (("train", train_data), ("id", id_data), ("family_ood", ood_data))}, "split_manifest_hashes": {split: value["split_manifest_sha256"] for split, value in (("train", train_data), ("id", id_data), ("family_ood", ood_data))}, "world_initial_loss": world_losses[0], "world_final_loss": world_losses[-1], "world_model_passed": world_passed, "behavior_passed": behavior["passed"], "runtime_seconds": duration, "estimated_cost_usd": duration / 3600 * RATE_USD_PER_HOUR, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()), "claims": computed_claims, "claim_label": manifest()["claim_label"]}
    (output_root / "terminal.json").write_bytes(canonical_json(terminal))
    artifacts = [artifact_declaration(path, output_root, role="general-world-model-engineering-evidence") for path in sorted(output_root.rglob("*")) if path.is_file()]
    task_evidence = write_task_evidence(output_root, metrics=[{"name": "world_initial_loss", "value": world_losses[0]}, {"name": "world_final_loss", "value": world_losses[-1]}, {"name": "world_model_passed", "value": float(world_passed)}, {"name": "behavior_passed", "value": float(behavior["passed"])}], artifacts=artifacts, track_h={"phase": "general-world-model-pilot", "decision": terminal["decision"], "claims": terminal["claims"]})
    return {**terminal, "task_evidence": task_evidence}


def run_contract(expected_manifest_sha256: str, expected_code_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"id": "general-world-model-pilot", "_secret_keys": ["HF_TOKEN"], "_preregistration": {"layer_allowlist": [7, 23, 39], "timepoint_allowlist": ["all_tokens"]}},
        {"id": "general-world-model-pilot", "task": {"module": "jump_benchmark.general_world_model_task", "parameters": {"expected_manifest_sha256": expected_manifest_sha256, "expected_code_sha": expected_code_sha}}, "resources": {"gpu": "H100", "timeout_seconds": 5400}, "selection": {"layers": [7, 23, 39], "timepoints": ["all_tokens"]}, "retry": {"max_attempts": 1}},
    )


__all__ = ["MANIFEST_SHA256", "manifest", "procedural_spec", "dataset", "build_modules", "build_latent_bridge", "cpu_preflight", "train_and_evaluate", "run_contract"]
