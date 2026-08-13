from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from jump_contracts.experiments import (
    DEPENDENCY_LOCK_SHA256,
    EXPERIMENT_PLAN_SCHEMA_SHA256,
    EXPERIMENT_RUN_SCHEMA_SHA256,
    RESTRICTED_POLICY_SHA256,
    ExperimentContractError,
    build_experiment_run,
    canonical_json,
    comparison_records,
    validate_experiment_plan,
    validate_experiment_run,
)
from jump_contracts.evidence import seal_result_envelope
from jump_workbench.workflow import (
    ConfirmationRequired,
    FrozenModel,
    confirm_and_predict,
    finalize_run,
    prepare_plan,
)


SOURCE = """\
import random

def simulate(plan):
    rows = []
    for condition in plan["conditions"]:
        for repetition in range(plan["sampling"]["repetitions"]):
            rng = random.Random(plan["sampling"]["seed"] + repetition)
            value = rng.random() + condition["assignments"]["offset"]
            rows.append({"condition_id": condition["id"], "repetition": repetition,
                         "pairing_key": "rep-" + str(repetition),
                         "values": {"wait": value}})
    return {"measurements": rows}
"""


def _planner(_request):
    return {
        "source": SOURCE,
        "plan": {
            "hypothesis": "Adding one unit increases the measured value.",
            "variables": {
                "independent": [{"id": "offset", "label": "Offset", "levels": [0, 1]}],
                "dependent": [{"id": "wait", "label": "Measured value", "unit": None}],
                "controlled": [],
            },
            "assumptions": ["Random draws are independent between repetitions."],
            "conditions": [
                {"id": "base", "label": "No offset", "kind": "baseline", "assignments": {"offset": 0}},
                {"id": "add", "label": "Add one", "kind": "intervention", "assignments": {"offset": 1}},
            ],
            "sampling_design": "paired_common_random_numbers",
            "prediction_before_run": {
                "required": True,
                "targets": [{"id": "offset_effect", "measurement_id": "wait", "baseline_condition_id": "base", "intervention_condition_id": "add"}],
            },
            "measurements": [{"id": "wait", "label": "Measured value", "unit": None, "aggregation": "mean", "display": "bar"}],
            "comparisons": [{"id": "offset_effect", "measurement_id": "wait", "baseline_condition_id": "base", "intervention_condition_id": "add", "statistic": "mean_difference", "pairing": "paired_by_repetition"}],
        },
    }


def _model():
    return FrozenModel(model_id="google/gemma-frozen", revision="a" * 40)


def _planned():
    return prepare_plan(
        "In a toy random process, what happens if I add one to each draw?",
        session_id="session-1", request_id="request-1", seed=7613, repetitions=2,
        model=_model(), planner=_planner,
    )


def _prediction(_plan):
    return {
        "summary": "The intervention mean should be larger.",
        "claims": [{"target_id": "offset_effect", "expected_relation": "greater", "rationale": "Each paired draw receives one extra unit.", "expected_value": 1.0}],
    }


def _modal_result(plan):
    return {
        "measurements": [
            {"plan_sha256": plan["plan_sha256"], "condition_id": "base", "repetition": 0, "pairing_key": "rep-0", "values": {"wait": 0.2}},
            {"plan_sha256": plan["plan_sha256"], "condition_id": "add", "repetition": 0, "pairing_key": "rep-0", "values": {"wait": 1.2}},
            {"plan_sha256": plan["plan_sha256"], "condition_id": "base", "repetition": 1, "pairing_key": "rep-1", "values": {"wait": 0.7}},
            {"plan_sha256": plan["plan_sha256"], "condition_id": "add", "repetition": 1, "pairing_key": "rep-1", "values": {"wait": 1.7}},
        ],
        "stdout": "",
    }


def test_schema_hashes_and_server_owned_policy_are_frozen():
    assert EXPERIMENT_PLAN_SCHEMA_SHA256 == "4cadd0c9859add72c5c71c4cca0e71e3f0a7b5deb398948ae205a238035ffc08"
    assert EXPERIMENT_RUN_SCHEMA_SHA256 == "5e6be146224d6ae7661be9f911a0bce47df4bb5d145ee83967bc446c7ffc97ca"
    assert RESTRICTED_POLICY_SHA256 != hashlib.sha256(b"").hexdigest()
    assert DEPENDENCY_LOCK_SHA256 != hashlib.sha256(b"").hexdigest()
    plan = _planned()["plan"]
    tampered = deepcopy(plan)
    tampered["sandbox"]["policy_sha256"] = "f" * 64
    preimage = {key: value for key, value in tampered.items() if key not in {"plan_id", "plan_sha256"}}
    tampered["plan_sha256"] = hashlib.sha256(canonical_json(preimage)).hexdigest()
    tampered["plan_id"] = "plan-" + tampered["plan_sha256"][:24]
    with pytest.raises(ExperimentContractError, match="server-owned"):
        validate_experiment_plan(tampered)


def test_confirmation_and_prediction_precede_measurement_and_revision():
    planned = _planned()
    with pytest.raises(ConfirmationRequired):
        confirm_and_predict(planned, confirmed=False, model=_model(), predictor=_prediction)
    prepared = confirm_and_predict(
        planned, confirmed=True, model=_model(), predictor=_prediction,
        now=lambda: datetime(2026, 8, 13, 10, 0, tzinfo=timezone.utc),
    )
    reviewer_inputs = []
    run = finalize_run(
        prepared,
        _modal_result(prepared.plan),
        modal_call_id="fc-test",
        code_version="b" * 40,
        started_at=datetime(2026, 8, 13, 10, 1, tzinfo=timezone.utc),
        completed_at=datetime(2026, 8, 13, 10, 2, tzinfo=timezone.utc),
        model=_model(),
        reviewer=lambda value: reviewer_inputs.append(value) or {
            "disposition": "retain", "interpretation": "The measured paired difference matches the prediction.", "next_plan_sha256": None,
        },
    )
    assert run["status"] == "completed"
    assert run["comparisons"][0]["estimate"] == pytest.approx(1.0)
    assert reviewer_inputs[0]["measurements"] == run["measurements"]
    assert run["execution"]["prediction_recorded_at"] < run["execution"]["started_at"]


def test_run_requires_plan_evidence_and_complete_row_bound_comparisons():
    plan = _planned()["plan"]
    rows = _modal_result(plan)["measurements"]
    comparisons = comparison_records(plan, rows)
    prediction = _prediction(plan)
    execution = {
        "plan_sha256": plan["plan_sha256"], "prediction": prediction,
        "prediction_sha256": hashlib.sha256(canonical_json(prediction)).hexdigest(),
        "prediction_recorded_at": "2026-08-13T10:00:00Z", "started_at": "2026-08-13T10:01:00Z",
        "completed_at": "2026-08-13T10:02:00Z", "source_sha256": plan["sandbox"]["source"]["sha256"],
        "policy_sha256": plan["sandbox"]["policy_sha256"], "code_version": "b" * 40,
        "modal_call_id": "fc-test", "error": None,
    }
    revision = {"plan_sha256": plan["plan_sha256"], "disposition": "retain", "interpretation": "Retain.", "next_plan_sha256": None}
    evidence = {"plan_sha256": plan["plan_sha256"], "source_sha256": execution["source_sha256"],
                "policy_sha256": execution["policy_sha256"], "code_version": execution["code_version"],
                "modal_call_id": "fc-test", "run_result_sha256": "0" * 64,
                "artifact_inventory_sha256": "0" * 64, "sealed_payload_sha256": "0" * 64}
    run_result = {
        "schema_version": "jump.run-result/v1", "status": "completed",
        "metrics": [{"name": "offset_effect", "value": 1.0}], "artifacts": [],
        "provenance": {"manifest_sha256": plan["plan_sha256"], "run_id": "fc-test",
                       "code_version": "b" * 40, "source_sha256": execution["source_sha256"],
                       "policy_sha256": execution["policy_sha256"]},
        "plan_sha256": plan["plan_sha256"], "measurements": rows, "comparisons": comparisons,
    }
    sealed_payload = {"plan_sha256": plan["plan_sha256"], "prediction": prediction, "measurements": rows, "comparisons": comparisons, "revision": revision}
    sealed = seal_result_envelope(
        sealed_payload, source="live", manifest_sha256=plan["plan_sha256"], run_id="fc-test",
        code_version="b" * 40, checkpoint_id=plan["plan_id"],
    )
    run = build_experiment_run(
        plan, verified_run_result=run_result, artifact_bytes={}, sealed_result=sealed,
        plan_id=plan["plan_id"], plan_sha256=plan["plan_sha256"], status="completed", execution=execution,
        measurements=rows, comparisons=comparisons, revision=revision, evidence=evidence,
    )
    with pytest.raises(TypeError):
        validate_experiment_run(run)  # type: ignore[call-arg]
    tampered_comparisons = deepcopy(comparisons)
    tampered_comparisons[0]["estimate"] = 9.0
    tampered_result = {**run_result, "comparisons": tampered_comparisons}
    tampered_sealed = seal_result_envelope(
        {**sealed_payload, "comparisons": tampered_comparisons}, source="live",
        manifest_sha256=plan["plan_sha256"], run_id="fc-test", code_version="b" * 40,
        checkpoint_id=plan["plan_id"],
    )
    with pytest.raises(ExperimentContractError, match="recomputed"):
        build_experiment_run(
            plan, verified_run_result=tampered_result, artifact_bytes={}, sealed_result=tampered_sealed,
            plan_id=plan["plan_id"], plan_sha256=plan["plan_sha256"], status="completed",
            execution=execution, measurements=rows, comparisons=tampered_comparisons,
            revision=revision, evidence=evidence,
        )
    incomplete_rows = rows[:-1]
    incomplete_result = {**run_result, "measurements": incomplete_rows}
    incomplete_sealed = seal_result_envelope(
        {**sealed_payload, "measurements": incomplete_rows}, source="live",
        manifest_sha256=plan["plan_sha256"], run_id="fc-test", code_version="b" * 40,
        checkpoint_id=plan["plan_id"],
    )
    with pytest.raises(ExperimentContractError, match="every condition/repetition"):
        build_experiment_run(
            plan, verified_run_result=incomplete_result, artifact_bytes={}, sealed_result=incomplete_sealed,
            plan_id=plan["plan_id"], plan_sha256=plan["plan_sha256"], status="completed",
            execution=execution, measurements=incomplete_rows, comparisons=comparisons,
            revision=revision, evidence=evidence,
        )


def test_run_rejects_result_from_a_different_modal_call():
    planned = _planned()
    prepared = confirm_and_predict(
        planned, confirmed=True, model=_model(), predictor=_prediction,
        now=lambda: datetime(2026, 8, 13, 10, 0, tzinfo=timezone.utc),
    )
    run = finalize_run(
        prepared, _modal_result(prepared.plan), modal_call_id="fc-call-a", code_version="b" * 40,
        started_at=datetime(2026, 8, 13, 10, 1, tzinfo=timezone.utc),
        completed_at=datetime(2026, 8, 13, 10, 2, tzinfo=timezone.utc), model=_model(),
        reviewer=lambda _value: {"disposition": "retain", "interpretation": "Retain.", "next_plan_sha256": None},
    )
    rows = run["measurements"]
    comparisons = run["comparisons"]
    result_from_b = {
        "schema_version": "jump.run-result/v1", "status": "completed",
        "metrics": [{"name": "offset_effect", "value": 1.0}], "artifacts": [],
        "provenance": {"manifest_sha256": prepared.plan["plan_sha256"], "run_id": "fc-call-b",
                       "code_version": "b" * 40, "source_sha256": run["execution"]["source_sha256"],
                       "policy_sha256": run["execution"]["policy_sha256"]},
        "plan_sha256": prepared.plan["plan_sha256"], "measurements": rows, "comparisons": comparisons,
    }
    sealed_a = seal_result_envelope(
        {"plan_sha256": prepared.plan["plan_sha256"], "prediction": run["execution"]["prediction"],
         "measurements": rows, "comparisons": comparisons, "revision": run["revision"]},
        source="live", manifest_sha256=prepared.plan["plan_sha256"], run_id="fc-call-a",
        code_version="b" * 40, checkpoint_id=prepared.plan["plan_id"],
    )
    with pytest.raises(ExperimentContractError, match="run identity"):
        validate_experiment_run(
            run, prepared.plan, verified_run_result=result_from_b, artifact_bytes={}, sealed_result=sealed_a,
        )
