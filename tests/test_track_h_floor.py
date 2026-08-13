from copy import deepcopy

import pytest

from jump_benchmark.baselines import CONDITIONS, build_request
from jump_benchmark.canonical import canonical_json
from jump_benchmark.scoring import score_answer
from jump_benchmark.simulator import DatasetSpec, EpisodeSpec, Law, generate_dataset, generate_episode


def test_six_object_generator_is_byte_deterministic_and_seed_disjoint():
    spec = DatasetSpec(seed=7613, split_counts={"train": 3, "test": 2})
    first, second = generate_dataset(spec), generate_dataset(spec)
    assert canonical_json(first) == canonical_json(second)
    assert all(episode["object_count"] == 6 for episode in first["episodes"])
    seeds = [episode["world_seed"] for episode in first["episodes"]]
    assert len(seeds) == len(set(seeds))


def test_exact_scorer_accepts_label_complement_but_not_free_text():
    target = generate_episode(
        EpisodeSpec(17, "test", Law("attract", "repel", 2), True)
    )["target"]
    answer = {**deepcopy(target), "confidence": 0.8}
    answer["partition"] = [1 - value for value in answer["partition"]]
    score = score_answer(answer, target, allowed_exponents=[1, 2])
    assert score["joint_theory_accuracy"] == 1
    assert score["force_nrmse_horizon_auc"] == pytest.approx(0)
    assert score_answer("looks attractive", target, allowed_exponents=[1, 2])["parse_success"] == 0


def test_all_four_baseline_interfaces_are_distinct_and_bounded():
    episode = generate_episode(EpisodeSpec(23, "test", Law("attract", "repel", 2), False))
    requests = {
        condition: build_request(episode, condition, exponents=[1, 2], lexical_token_budget=4096)
        for condition in CONDITIONS
    }
    assert tuple(requests) == ("A", "B", "C", "C-prime")
    assert len({request.sha256 for request in requests.values()}) == 4
    assert all(request.lexical_token_count <= request.lexical_token_budget for request in requests.values())
    assert "all 31 canonical partitions" in requests["C-prime"].prompt

