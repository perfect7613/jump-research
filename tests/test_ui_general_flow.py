from __future__ import annotations

from copy import deepcopy

import pytest

from jump_contracts.experiments import ExperimentContractError
from jump_ui.general_flow import GeneralUIError, confirm_fixture, execute_fixture, plan_rows, prepare_fixture, result_rows
from jump_ui.general_presentation import EXAMPLES, PLAN_LABELS, RESULT_LABELS


def test_general_fixture_requires_confirmation_and_uses_exact_contracts():
    planned = prepare_fixture(EXAMPLES[1], request_id="request-1", repetitions=3)
    assert planned["state"] == "awaiting_confirmation"
    assert planned["request"] == {
        "schema_version": "jump.experiment-question/v1",
        "request_id": "request-1",
        "session_id": "local-session",
        "intent": EXAMPLES[1],
        "seed": 7613,
        "repetitions": 3,
    }
    assert "source" not in planned["request"]
    assert tuple(plan_rows(planned["plan"])) == PLAN_LABELS
    with pytest.raises(GeneralUIError, match="confirm"):
        confirm_fixture(planned, confirmed=False)
    prepared = confirm_fixture(planned, confirmed=True)
    assert prepared.state == "prediction_ready"
    run = execute_fixture(prepared)
    assert run["status"] == "completed"
    assert tuple(result_rows(run, prepared.plan)) == RESULT_LABELS

    tampered = deepcopy(run)
    tampered["comparisons"][0]["estimate"] = 999
    with pytest.raises(ExperimentContractError):
        result_rows(tampered, prepared.plan)


@pytest.mark.parametrize("intent", (
    "Download traffic data from https://example.com", "Read ~/private.csv and simulate it",
    "```python\nimport os\n```", "Run a clinical trial on patients",
))
def test_general_ingress_rejects_urls_files_code_and_real_world_actions(intent):
    with pytest.raises(GeneralUIError):
        prepare_fixture(intent, request_id="request-1")


def test_general_app_has_plan_confirmation_and_particle_research_is_secondary():
    gr = pytest.importorskip("gradio")
    from jump_ui.general_app import create_general_app

    config = str(create_general_app().get_config_file())
    assert "GENERAL WORKBENCH · PLAN REVIEW REQUIRED" in config
    assert "Confirm plan and run simulation" in config
    assert "Test an idea. See what the simulation says." in config
    assert "Original research demo" in config


def test_bernoulli_fixture_passes_direct_plan_and_run_validation():
    planned = prepare_fixture(EXAMPLES[2], request_id="request-bernoulli", repetitions=4)
    prepared = confirm_fixture(planned, confirmed=True)
    sections = result_rows(execute_fixture(prepared), prepared.plan)
    assert sections["Was the prediction right?"].startswith("Yes")
    assert "0.2" in sections["Simulation"]
