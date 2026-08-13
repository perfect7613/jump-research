"""Minimum delivery tests for the plain ExperimentSpec UI flow."""

from __future__ import annotations

import copy

import pytest

from jump_ui.flow import (
    ExperimentFlowError,
    LiveExperimentBackend,
    NonLiveContractFixtureBackend,
    plan_experiment,
    run_confirmed_experiment,
    verified_result,
)
from jump_ui.presentation import correctness_html, plan_html, result_sections


def planned(intent="Make matching objects repel instead of attract."):
    return plan_experiment(intent, session_id="test-session", seed=7, max_steps=4)


def test_plan_confirmation_and_scope_rejection():
    run = planned()
    assert run["status"] == "planned" and run["result"] is None
    html = plan_html(run)
    assert "Here’s the experiment I understood" in html
    assert all(label in html for label in ("World", "Observe", "Change", "Predict"))
    assert "Make matching objects" not in html  # raw intent is absent from the plan
    with pytest.raises(ExperimentFlowError, match="URLs|URI"):
        plan_experiment("Open https://example.com and run it", session_id="test-session")


def test_app_exposes_one_run_then_mandatory_confirmation():
    gr = pytest.importorskip("gradio")
    from jump_ui.app import QUESTION, create_app

    demo = create_app(backend=NonLiveContractFixtureBackend(), enable_queue=False)
    assert isinstance(demo, gr.Blocks)
    assert "jump-title" in demo._jump_css
    config = demo.get_config_file()
    labels = {component.get("props", {}).get("label") for component in config["components"]}
    assert QUESTION in labels
    values = {component.get("props", {}).get("value") for component in config["components"]}
    assert {"Run experiment", "Run this plan"} <= values
    dependencies = {dependency.get("api_name") for dependency in config["dependencies"]}
    assert {"parse_intent", "execute_plan"} <= dependencies


def test_live_failure_never_substitutes_a_fixture_or_recording():
    run = planned()
    with pytest.raises(ExperimentFlowError, match="No recorded result was substituted"):
        run_confirmed_experiment(
            run,
            backend=LiveExperimentBackend(token=""),
            intent="Make matching objects repel instead of attract.",
        )
    assert run["status"] == "planned" and run["result"] is None


def test_format_valid_and_exact_correct_are_separate_readouts():
    html = correctness_html(
        {
            "format_valid": True,
            "exact_correct": False,
            "partition_correct": True,
            "law_correct": False,
            "adequacy_correct": False,
            "force_score": None,
            "notes": "Well formed, not exactly correct.",
        }
    )
    assert "Answer format</span><strong class=\"good\">Yes" in html
    assert "Exact answer</span><strong class=\"bad\">No" in html


def test_same_z_and_learned_decoder_image_provenance_gate_presentation():
    completed = run_confirmed_experiment(planned(), backend=NonLiveContractFixtureBackend())
    checked = verified_result(completed)
    tensor = checked["evidence"]["tensor"]
    assert {
        tensor["world_latent_sha256"],
        tensor["encoder_output_sha256"],
        tensor["decoder_input_sha256"],
        tensor["injection_input_sha256"],
        checked["evidence"]["decoded_observation"]["world_latent_sha256"],
    } == {tensor["world_latent_sha256"]}
    cards = result_sections(completed, backend_label="NON-LIVE CONTRACT FIXTURE")
    assert "same learned world state" in cards[4]
    assert "NON-LIVE CONTRACT FIXTURE" in cards[0]

    tampered = copy.deepcopy(completed)
    tampered["result"]["decoded_image"]["data"] += "AAAA"
    with pytest.raises(ValueError, match="base64|bytes do not match"):
        verified_result(tampered)
