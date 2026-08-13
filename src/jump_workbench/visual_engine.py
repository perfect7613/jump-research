"""Server-owned interpreter for ExperimentSpec v2 data.

No generated code is accepted or executed. Every operation is dispatched from
an allowlist after the complete spec has passed the shared validator.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from fractions import Fraction
from typing import Any

from jump_contracts.thought_experiments import validate_experiment_spec


class VisualEngineError(ValueError):
    pass


def execute_visual_spec(spec_value: dict[str, Any]) -> dict[str, Any]:
    spec = validate_experiment_spec(spec_value)
    results = [_execute_condition(spec, condition) for condition in spec["conditions"]]
    by_id = {item["condition_id"]: item for item in results}
    baseline = next(item["id"] for item in spec["conditions"] if item["kind"] == "baseline")
    comparisons = []
    for condition in spec["conditions"]:
        if condition["kind"] != "counterfactual":
            continue
        for measurement in spec["measurements"]:
            base_value = by_id[baseline]["summary"][measurement["id"]]
            changed_value = by_id[condition["id"]]["summary"][measurement["id"]]
            difference = float(Fraction(str(changed_value)) - Fraction(str(base_value)))
            comparisons.append({
                "measurement_id": measurement["id"],
                "baseline_condition_id": baseline,
                "counterfactual_condition_id": condition["id"],
                "baseline_final": base_value,
                "counterfactual_final": changed_value,
                "difference": difference,
            })
    return {"conditions": results, "comparisons": comparisons}


def _execute_condition(spec: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    repetitions = spec["schedule"]["repetitions"]
    all_series: list[dict[str, list[float]]] = []
    frames: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        rng = random.Random(spec["schedule"]["seed"] + repetition)
        entities = _initial_entities(spec, rng)
        rules = deepcopy(spec["dynamics"]["rules"])
        series = {item["id"]: [] for item in spec["measurements"]}
        captured: list[dict[str, Any]] = []
        duration = spec["schedule"]["duration_steps"]
        stride = spec["visualization"]["frame_stride"]
        for step in range(duration + 1):
            _apply_interventions(entities, rules, condition["interventions"], step)
            if step:
                graph = _graph_edges(spec, entities, rng)
                for rule in rules:
                    _apply_rule(rule, entities, graph, spec, rng)
                _apply_boundary(entities, spec["world"]["bounds"])
            for measurement in spec["measurements"]:
                series[measurement["id"]].append(_measure(measurement, entities))
            if repetition == 0 and (step % stride == 0 or step == duration):
                captured.append({"step": step, "points": _points(entities)})
        all_series.append(series)
        if repetition == 0:
            frames = captured
    averaged = []
    summary = {}
    duration = spec["schedule"]["duration_steps"]
    for measurement in spec["measurements"]:
        measurement_id = measurement["id"]
        values = []
        for step in range(duration + 1):
            value = sum(item[measurement_id][step] for item in all_series) / repetitions
            values.append({"step": step, "value": float(value)})
        averaged.append({"measurement_id": measurement_id, "values": values})
        summary[measurement_id] = values[-1]["value"]
    return {"condition_id": condition["id"], "frames": frames, "series": averaged, "summary": summary}


def _initial_entities(spec: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    bounds = spec["world"]["bounds"]
    entities = []
    for declaration in spec["world"]["entities"]:
        center_x, center_y = declaration["initial_layout"]["center"]
        spread = declaration["initial_layout"]["spread"]
        count = declaration["count"]
        for index in range(count):
            x, y = _layout_position(declaration["initial_layout"]["kind"], index, count, center_x, center_y, spread, rng)
            entities.append({
                "id": f"{declaration['id']}-{index}",
                "type_id": declaration["id"],
                "x": x,
                "y": y,
                "numeric": deepcopy(declaration["initial_state"]["numeric"]),
                "categorical": deepcopy(declaration["initial_state"]["categorical"]),
                "appearance": deepcopy(declaration["appearance"]),
                "alive": True,
            })
    return entities


def _layout_position(kind: str, index: int, count: int, cx: float, cy: float, spread: float, rng: random.Random) -> tuple[float, float]:
    if kind == "uniform":
        return cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread)
    if kind == "ring":
        angle = 2 * math.pi * index / max(count, 1)
        return cx + spread * math.cos(angle), cy + spread * math.sin(angle)
    if kind == "line":
        return cx + spread * (index - (count - 1) / 2), cy
    side = max(1, math.ceil(math.sqrt(count)))
    return cx + spread * (index % side - (side - 1) / 2), cy + spread * (index // side - (side - 1) / 2)


def _graph_edges(spec: dict[str, Any], entities: list[dict[str, Any]], rng: random.Random) -> list[tuple[int, int]]:
    graph = spec["world"]["graph"]
    count = len(entities)
    if graph["kind"] == "none" or count < 2:
        return []
    if graph["kind"] == "ring":
        return [(index, (index + 1) % count) for index in range(count)]
    if graph["kind"] == "grid":
        side = max(1, math.ceil(math.sqrt(count)))
        edges = []
        for index in range(count):
            if index + 1 < count and index % side != side - 1:
                edges.append((index, index + 1))
            if index + side < count:
                edges.append((index, index + side))
        return edges
    graph_rng = random.Random(spec["schedule"]["seed"] + 99173)
    edges = []
    for left in range(count):
        for right in range(left + 1, count):
            if graph_rng.random() < graph["edge_probability"]:
                edges.append((left, right))
    return edges


def _apply_interventions(entities: list[dict[str, Any]], rules: list[dict[str, Any]], interventions: list[dict[str, Any]], step: int) -> None:
    rule_by_id = {item["id"]: item for item in rules}
    for change in interventions:
        if change["time"] != step:
            continue
        if change["operation"] == "set_rule_parameter":
            rule_by_id[change["target"]]["parameters"][change["field"]] = change["value"]
        elif change["operation"] == "scale_rule_parameter":
            rule_by_id[change["target"]]["parameters"][change["field"]] *= change["value"]
        else:
            state_kind = "numeric" if change["operation"] == "set_numeric_state" else "categorical"
            for entity in entities:
                if entity["type_id"] == change["target"]:
                    entity[state_kind][change["field"]] = change["value"]


def _apply_rule(rule: dict[str, Any], entities: list[dict[str, Any]], graph: list[tuple[int, int]], spec: dict[str, Any], rng: random.Random) -> None:
    dispatch = {
        "move_2d": _move,
        "random_walk_2d": _random_walk,
        "pairwise_force_2d": _pairwise_force,
        "graph_diffusion": _graph_diffusion,
        "graph_contagion": _graph_contagion,
        "predator_prey_2d": _predator_prey,
        "lane_traffic_2d": _lane_traffic,
        "queue_agents_2d": _queue_agents,
    }
    dispatch[rule["op"]](rule, entities, graph, spec, rng)


def _targets(rule: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in entities if item["alive"] and (rule["target_type"] is None or item["type_id"] == rule["target_type"])]


def _move(rule: dict[str, Any], entities: list[dict[str, Any]], _graph: Any, spec: dict[str, Any], _rng: Any) -> None:
    dt = spec["schedule"]["dt"]
    damping, max_speed = rule["parameters"]["damping"], rule["parameters"]["max_speed"]
    for entity in _targets(rule, entities):
        vx, vy = entity["numeric"].get("vx", 0.0), entity["numeric"].get("vy", 0.0)
        speed = math.hypot(vx, vy)
        if speed > max_speed > 0:
            vx, vy = vx * max_speed / speed, vy * max_speed / speed
        entity["x"] += vx * dt
        entity["y"] += vy * dt
        entity["numeric"]["vx"], entity["numeric"]["vy"] = vx * damping, vy * damping


def _random_walk(rule: dict[str, Any], entities: list[dict[str, Any]], _graph: Any, _spec: Any, rng: random.Random) -> None:
    scale = rule["parameters"]["step_scale"]
    for entity in _targets(rule, entities):
        entity["x"] += rng.gauss(0, scale)
        entity["y"] += rng.gauss(0, scale)


def _pairwise_force(rule: dict[str, Any], entities: list[dict[str, Any]], _graph: Any, spec: dict[str, Any], _rng: Any) -> None:
    selected = _targets(rule, entities)
    strength = rule["parameters"]["strength"]
    exponent = rule["parameters"]["exponent"]
    softening = max(rule["parameters"]["softening"], 1e-6)
    dt = spec["schedule"]["dt"]
    accelerations = {item["id"]: [0.0, 0.0] for item in selected}
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            dx, dy = right["x"] - left["x"], right["y"] - left["y"]
            distance = math.sqrt(dx * dx + dy * dy + softening * softening)
            force = strength / (distance ** exponent)
            fx, fy = force * dx / distance, force * dy / distance
            accelerations[left["id"]][0] += fx
            accelerations[left["id"]][1] += fy
            accelerations[right["id"]][0] -= fx
            accelerations[right["id"]][1] -= fy
    for entity in selected:
        entity["numeric"]["vx"] = entity["numeric"].get("vx", 0.0) + accelerations[entity["id"]][0] * dt
        entity["numeric"]["vy"] = entity["numeric"].get("vy", 0.0) + accelerations[entity["id"]][1] * dt


def _graph_diffusion(rule: dict[str, Any], entities: list[dict[str, Any]], graph: list[tuple[int, int]], _spec: Any, _rng: Any) -> None:
    state, rate = rule["parameters"]["state"], rule["parameters"]["rate"]
    changes = [0.0] * len(entities)
    for left, right in graph:
        if not entities[left]["alive"] or not entities[right]["alive"]:
            continue
        delta = rate * (entities[right]["numeric"][state] - entities[left]["numeric"][state])
        changes[left] += delta
        changes[right] -= delta
    for index, value in enumerate(changes):
        if entities[index]["alive"]:
            entities[index]["numeric"][state] += value


def _graph_contagion(rule: dict[str, Any], entities: list[dict[str, Any]], graph: list[tuple[int, int]], _spec: Any, rng: random.Random) -> None:
    p = rule["parameters"]
    infected = {
        index for index, item in enumerate(entities)
        if item["alive"] and item["categorical"].get(p["state"]) == p["infected"]
    }
    newly = set()
    for left, right in graph:
        if not entities[left]["alive"] or not entities[right]["alive"]:
            continue
        if left in infected and entities[right]["categorical"].get(p["state"]) == p["susceptible"] and rng.random() < p["transmission_probability"]:
            newly.add(right)
        if right in infected and entities[left]["categorical"].get(p["state"]) == p["susceptible"] and rng.random() < p["transmission_probability"]:
            newly.add(left)
    for index in newly:
        entities[index]["categorical"][p["state"]] = p["infected"]
    for index in infected:
        if rng.random() < p["recovery_probability"]:
            entities[index]["categorical"][p["state"]] = p["recovered"]


def _predator_prey(rule: dict[str, Any], entities: list[dict[str, Any]], _graph: Any, _spec: Any, rng: random.Random) -> None:
    p = rule["parameters"]
    prey = [item for item in entities if item["alive"] and item["type_id"] == p["prey_type"]]
    predators = [item for item in entities if item["alive"] and item["type_id"] == p["predator_type"]]
    for predator in predators:
        nearby = [item for item in prey if item["alive"] and math.hypot(item["x"] - predator["x"], item["y"] - predator["y"]) <= p["interaction_radius"]]
        if nearby and rng.random() < p["predation_rate"]:
            nearby[0]["alive"] = False
    for predator in predators:
        if rng.random() < p["predator_decay"]:
            predator["alive"] = False
    inactive_prey = [item for item in entities if not item["alive"] and item["type_id"] == p["prey_type"]]
    active_prey = [item for item in entities if item["alive"] and item["type_id"] == p["prey_type"]]
    for item in inactive_prey[:sum(1 for _ in active_prey if rng.random() < p["prey_growth"])]:
        item["alive"] = True
    inactive_predators = [item for item in entities if not item["alive"] and item["type_id"] == p["predator_type"]]
    if inactive_predators and any(not item["alive"] for item in prey) and rng.random() < p["predator_efficiency"]:
        inactive_predators[0]["alive"] = True


def _lane_traffic(rule: dict[str, Any], entities: list[dict[str, Any]], _graph: Any, spec: dict[str, Any], _rng: Any) -> None:
    p = rule["parameters"]
    width = spec["world"]["bounds"]["width"]
    cars = sorted(_targets(rule, entities), key=lambda item: item["x"])
    for index, car in enumerate(cars):
        ahead = cars[(index + 1) % len(cars)]
        gap = (ahead["x"] - car["x"]) % width
        speed = min(p["desired_speed"], max(0.0, gap - p["headway"]))
        car["numeric"][p["speed_state"]] = speed
        car["x"] += speed * spec["schedule"]["dt"]
        car["y"] = p["road_y"]


def _queue_agents(rule: dict[str, Any], entities: list[dict[str, Any]], _graph: Any, _spec: Any, rng: random.Random) -> None:
    p = rule["parameters"]
    agents = [item for item in _targets(rule, entities)]
    queued = [item for item in agents if item["categorical"].get("queue_status") == "queued"]
    waiting = [item for item in agents if item["categorical"].get("queue_status") != "queued"]
    for item in waiting:
        if rng.random() < p["arrival_probability"]:
            item["categorical"]["queue_status"] = "queued"; queued.append(item)
    for item in queued[:int(p["service_capacity"])]:
        item["categorical"]["queue_status"] = "served"
    queued = [item for item in agents if item["categorical"].get("queue_status") == "queued"]
    for index, item in enumerate(queued):
        item["x"], item["y"] = p["queue_x"] - index * 2, 0.0
    for item in agents:
        if item["categorical"].get("queue_status") == "served":
            item["x"] = p["service_x"]


def _apply_boundary(entities: list[dict[str, Any]], bounds: dict[str, Any]) -> None:
    width, height, boundary = bounds["width"], bounds["height"], bounds["boundary"]
    for item in entities:
        if not item["alive"]:
            continue
        if boundary == "wrap":
            item["x"] %= width
            item["y"] %= height
        elif boundary == "clamp":
            item["x"] = min(width, max(0.0, item["x"]))
            item["y"] = min(height, max(0.0, item["y"]))
        else:
            if item["x"] < 0 or item["x"] > width:
                if "vx" in item["numeric"]:
                    item["numeric"]["vx"] = -item["numeric"]["vx"]
                item["x"] = min(width, max(0.0, item["x"]))
            if item["y"] < 0 or item["y"] > height:
                if "vy" in item["numeric"]:
                    item["numeric"]["vy"] = -item["numeric"]["vy"]
                item["y"] = min(height, max(0.0, item["y"]))


def _measure(measurement: dict[str, Any], entities: list[dict[str, Any]]) -> float:
    selected = [item for item in entities if item["alive"] and (measurement["entity_type"] is None or item["type_id"] == measurement["entity_type"])]
    if measurement["op"] == "population_count":
        return float(len(selected))
    if measurement["op"] == "count_category":
        return float(sum(item["categorical"].get(measurement["state"]) == measurement["category"] for item in selected))
    values = [float(item["numeric"][measurement["state"]]) for item in selected]
    if not values:
        return 0.0
    if measurement["op"] == "sum_state":
        return float(sum(values))
    mean = sum(values) / len(values)
    if measurement["op"] == "variance_state":
        return float(sum((item - mean) ** 2 for item in values) / len(values))
    return float(mean)


def _points(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for item in entities:
        if not item["alive"]:
            continue
        category = next(iter(item["categorical"].values()), None)
        points.append({
            "entity_id": item["id"], "type_id": item["type_id"],
            "x": float(item["x"]), "y": float(item["y"]),
            "shape": item["appearance"]["shape"], "color": item["appearance"]["color"],
            "size": float(item["appearance"]["size"]), "category": category,
        })
    return points


__all__ = ["VisualEngineError", "execute_visual_spec"]
