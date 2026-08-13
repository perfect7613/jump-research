"""Small server-owned simulation templates selected by the frozen planner."""

from __future__ import annotations

from typing import Any


def compile_model_proposal(proposal: Any) -> dict[str, Any]:
    if not isinstance(proposal, dict) or set(proposal) != {"template_id", "hypothesis"}:
        raise ValueError("planner proposal must contain exactly template_id and hypothesis")
    template_id = proposal["template_id"]
    hypothesis = proposal["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 500:
        raise ValueError("planner hypothesis must be bounded nonempty text")
    compiler = {
        "monty_hall": _monty_hall,
        "queue_capacity": _queue_capacity,
        "traffic_capacity": _queue_capacity,
        "bernoulli_probability": _bernoulli_probability,
    }.get(template_id)
    if compiler is None:
        raise ValueError("unsupported experiment template")
    return compiler(hypothesis.strip(), template_id)


def _monty_hall(hypothesis: str, _template_id: str) -> dict[str, Any]:
    return {
        "plan": {
            "hypothesis": hypothesis,
            "variables": {
                "independent": [{"id": "strategy", "label": "Door strategy", "levels": ["stay", "switch"]}],
                "dependent": [{"id": "win_rate", "label": "Win rate", "unit": None}],
                "controlled": [{"id": "doors", "label": "Door count", "value": 3}],
            },
            "assumptions": ["The prize and first choice are uniform; the host knows the prize and opens a losing unchosen door."],
            "conditions": [
                {"id": "stay", "label": "Stay", "kind": "baseline", "assignments": {"strategy": "stay"}},
                {"id": "switch", "label": "Switch", "kind": "intervention", "assignments": {"strategy": "switch"}},
            ],
            "sampling_design": "paired_common_random_numbers",
            "prediction_before_run": {"required": True, "targets": [{"id": "switch_effect", "measurement_id": "win_rate", "baseline_condition_id": "stay", "intervention_condition_id": "switch"}]},
            "measurements": [{"id": "win_rate", "label": "Win rate", "unit": None, "aggregation": "mean", "display": "bar"}],
            "comparisons": [{"id": "switch_effect", "measurement_id": "win_rate", "baseline_condition_id": "stay", "intervention_condition_id": "switch", "statistic": "mean_difference", "pairing": "paired_by_repetition"}],
        },
        "source": '''import random

def simulate(plan):
    rows = []
    for condition in plan["conditions"]:
        for repetition in range(plan["sampling"]["repetitions"]):
            rng = random.Random(plan["sampling"]["seed"] + repetition)
            prize = rng.randrange(3)
            chosen = rng.randrange(3)
            stay_win = 1.0 if chosen == prize else 0.0
            win = stay_win if condition["assignments"]["strategy"] == "stay" else 1.0 - stay_win
            rows.append({"condition_id": condition["id"], "repetition": repetition, "pairing_key": "rep-" + str(repetition), "values": {"win_rate": win}})
    return {"measurements": rows}
''',
    }


def _queue_capacity(hypothesis: str, template_id: str) -> dict[str, Any]:
    label = "Traffic throughput" if template_id == "traffic_capacity" else "Queue service"
    return {
        "plan": {
            "hypothesis": hypothesis,
            "variables": {
                "independent": [{"id": "capacity", "label": f"{label} capacity", "levels": [4, 6]}],
                "dependent": [{"id": "average_queue", "label": "Average queue length", "unit": "agents"}],
                "controlled": [{"id": "arrival_probability", "label": "Arrival probability", "value": 0.65}, {"id": "steps", "label": "Steps", "value": 60}],
            },
            "assumptions": ["At each step, eight independent potential arrivals occur with probability 0.65."],
            "conditions": [
                {"id": "capacity_4", "label": "Capacity 4", "kind": "baseline", "assignments": {"capacity": 4}},
                {"id": "capacity_6", "label": "Capacity 6", "kind": "intervention", "assignments": {"capacity": 6}},
            ],
            "sampling_design": "paired_common_random_numbers",
            "prediction_before_run": {"required": True, "targets": [{"id": "capacity_effect", "measurement_id": "average_queue", "baseline_condition_id": "capacity_4", "intervention_condition_id": "capacity_6"}]},
            "measurements": [{"id": "average_queue", "label": "Average queue length", "unit": "agents", "aggregation": "mean", "display": "line"}],
            "comparisons": [{"id": "capacity_effect", "measurement_id": "average_queue", "baseline_condition_id": "capacity_4", "intervention_condition_id": "capacity_6", "statistic": "mean_difference", "pairing": "paired_by_repetition"}],
        },
        "source": '''import random

def simulate(plan):
    rows = []
    for condition in plan["conditions"]:
        capacity = condition["assignments"]["capacity"]
        for repetition in range(plan["sampling"]["repetitions"]):
            rng = random.Random(plan["sampling"]["seed"] + repetition)
            queued = 0
            total = 0
            for step in range(60):
                arrivals = 0
                for agent in range(8):
                    if rng.random() < 0.65:
                        arrivals += 1
                queued = max(0, queued + arrivals - capacity)
                total += queued
            rows.append({"condition_id": condition["id"], "repetition": repetition, "pairing_key": "rep-" + str(repetition), "values": {"average_queue": total / 60}})
    return {"measurements": rows}
''',
    }


def _bernoulli_probability(hypothesis: str, _template_id: str) -> dict[str, Any]:
    return {
        "plan": {
            "hypothesis": hypothesis,
            "variables": {
                "independent": [{"id": "probability", "label": "Success probability", "levels": [0.5, 0.7]}],
                "dependent": [{"id": "success_rate", "label": "Observed success rate", "unit": None}],
                "controlled": [{"id": "trials", "label": "Trials", "value": 100}],
            },
            "assumptions": ["Each draw is an independent Bernoulli trial."],
            "conditions": [
                {"id": "probability_05", "label": "Probability 0.5", "kind": "baseline", "assignments": {"probability": 0.5}},
                {"id": "probability_07", "label": "Probability 0.7", "kind": "intervention", "assignments": {"probability": 0.7}},
            ],
            "sampling_design": "paired_common_random_numbers",
            "prediction_before_run": {"required": True, "targets": [{"id": "probability_effect", "measurement_id": "success_rate", "baseline_condition_id": "probability_05", "intervention_condition_id": "probability_07"}]},
            "measurements": [{"id": "success_rate", "label": "Observed success rate", "unit": None, "aggregation": "mean", "display": "bar"}],
            "comparisons": [{"id": "probability_effect", "measurement_id": "success_rate", "baseline_condition_id": "probability_05", "intervention_condition_id": "probability_07", "statistic": "mean_difference", "pairing": "paired_by_repetition"}],
        },
        "source": '''import random

def simulate(plan):
    rows = []
    for condition in plan["conditions"]:
        probability = condition["assignments"]["probability"]
        for repetition in range(plan["sampling"]["repetitions"]):
            rng = random.Random(plan["sampling"]["seed"] + repetition)
            successes = 0
            for trial in range(100):
                if rng.random() < probability:
                    successes += 1
            rows.append({"condition_id": condition["id"], "repetition": repetition, "pairing_key": "rep-" + str(repetition), "values": {"success_rate": successes / 100}})
    return {"measurements": rows}
''',
    }


__all__ = ["compile_model_proposal"]
