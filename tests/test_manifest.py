from __future__ import annotations

import pytest

from jump_runner.errors import BudgetError, ManifestError
from jump_runner.manifest import authorize_launch, make_plan, validate_manifest


def test_valid_manifest_and_retry_aware_plan(manifest):
    manifest["defaults"]["retry"]["max_attempts"] = 2
    validate_manifest(manifest)
    plan = make_plan(manifest)
    assert plan["phases"][0]["gpu_seconds"] == 10
    assert plan["total_forecast_cost_usd"] == pytest.approx(10 / 3600 * 0.5, abs=1e-6)


def test_rejects_non_preregistered_parameter_layer(manifest):
    manifest["phases"][0]["runs"][0]["task"]["parameters"]["layers"] = [5]
    with pytest.raises(ManifestError, match="non-preregistered layers"):
        validate_manifest(manifest)


def test_rejects_non_preregistered_nested_timepoint(manifest):
    manifest["phases"][0]["runs"][0]["task"]["parameters"]["intervention"] = {"timepoint": "T3"}
    with pytest.raises(ManifestError, match="non-preregistered timepoints"):
        validate_manifest(manifest)


def test_rejects_worst_case_retry_budget(manifest):
    manifest["defaults"]["retry"]["max_attempts"] = 3
    with pytest.raises(BudgetError, match="plans 15s"):
        validate_manifest(manifest)


def test_rejects_literal_secret_keys(manifest):
    manifest["phases"][0]["runs"][0]["task"]["parameters"]["api_key"] = "do-not-store-this"
    with pytest.raises(ManifestError, match="secret-like"):
        validate_manifest(manifest)


def test_dependencies_must_be_earlier(manifest):
    manifest["phases"][0]["depends_on"] = ["future"]
    with pytest.raises(ManifestError, match="earlier phases"):
        validate_manifest(manifest)


def test_cpu_resource_has_zero_gpu_seconds(manifest):
    phase = manifest["phases"][0]
    phase["budget"].update(
        {
            "allowed_gpu_types": ["cpu"],
            "max_concurrent_gpus": 0,
            "gpu_hourly_cost_usd": {"cpu": 0},
            "max_cost_usd": 0,
        }
    )
    manifest["defaults"]["resources"]["gpu"] = "cpu"
    validate_manifest(manifest)
    plan = make_plan(manifest)
    assert plan["phases"][0]["compute_seconds"] == 5
    assert plan["phases"][0]["gpu_seconds"] == 0
    assert plan["total_forecast_cost_usd"] == 0


def test_h100_requires_profile_gate_measurements_and_budget(manifest):
    profile = manifest["phases"][0]
    profile["runs"][0]["resources"] = {"gpu": "L40S", "timeout_seconds": 5}
    profile["budget"].update(
        {"allowed_gpu_types": ["L40S"], "gpu_hourly_cost_usd": {"L40S": 1.0}}
    )
    manifest["phases"].append(
        {
            "id": "h100-escalation",
            "depends_on": ["pilot"],
            "budget": {
                "allowed_gpu_types": ["H100"],
                "max_concurrent_gpus": 1,
                "max_runtime_seconds": 60,
                "max_cost_usd": 0.1,
                "gpu_hourly_cost_usd": {"H100": 3.0},
            },
            "h100_justification": {
                "opt_in": True,
                "profile_phase_id": "pilot",
                "measured_peak_memory_gb": 47.5,
                "measured_runtime_seconds": 120,
                "why_lower_gpu_insufficient": "profile exceeded L40S memory",
                "forecast_cost_usd": 0.05,
                "remaining_budget_usd": 0.1,
            },
            "runs": [
                {
                    "id": "h100-run",
                    "task": {
                        "module": "jump_runner.mock_task",
                        "parameters": {"layers": [4], "timepoints": ["T4"]},
                    },
                    "resources": {"gpu": "H100", "timeout_seconds": 60},
                }
            ],
            "gates": [
                {"id": "h100-pass", "metric": "accuracy", "operator": ">=", "threshold": 0.5}
            ],
        }
    )
    validate_manifest(manifest)


def test_h100_cannot_be_default(manifest):
    manifest["defaults"]["resources"]["gpu"] = "H100"
    with pytest.raises(ManifestError, match="cannot be the manifest default"):
        validate_manifest(manifest)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_nonfinite_and_boolean_budgets_are_rejected(manifest, value):
    manifest["phases"][0]["budget"]["max_cost_usd"] = value
    with pytest.raises(ManifestError):
        validate_manifest(manifest)


def test_invalid_gate_aggregation_rejected_before_execution(manifest):
    manifest["phases"][0]["gates"][0]["aggregation"] = "median"
    with pytest.raises(ManifestError, match="JSON Schema violation"):
        validate_manifest(manifest)


def test_remote_launch_policy_rejects_direct_and_smoke_bypasses(manifest):
    with pytest.raises(ManifestError, match="full matrix submission is locked"):
        authorize_launch(manifest, smoke=False, confirm_paid=False, confirm_h100=False)
    run = manifest["phases"][0]["runs"][0]
    run["resources"] = {"gpu": "L40S", "timeout_seconds": 5}
    manifest["phases"][0]["budget"].update(
        {"allowed_gpu_types": ["L40S"], "gpu_hourly_cost_usd": {"L40S": 1.0}}
    )
    with pytest.raises(ManifestError, match="smoke resources are limited"):
        authorize_launch(manifest, smoke=True, confirm_paid=False, confirm_h100=False)


def test_normative_schema_and_runtime_accept_and_reject_same_manifests(manifest):
    import json
    from importlib.resources import files

    import jsonschema

    schema = json.loads(files("jump_runner").joinpath("schemas/experiment-manifest-v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)
    validate_manifest(manifest)
    invalid = dict(manifest)
    invalid["phases"] = [dict(manifest["phases"][0])]
    invalid["phases"][0]["gates"] = [dict(manifest["phases"][0]["gates"][0], aggregation="median")]
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(invalid))
    with pytest.raises(ManifestError):
        validate_manifest(invalid)


def test_packaged_and_repository_schemas_are_identical():
    from importlib.resources import files
    from pathlib import Path

    for name in ("experiment-manifest-v1.schema.json", "run-result-v1.schema.json"):
        assert Path("schemas", name).read_bytes() == files("jump_runner").joinpath("schemas", name).read_bytes()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nested_task_parameter_nonfinite_is_rejected(manifest, value):
    manifest["phases"][0]["runs"][0]["task"]["parameters"]["nested"] = {
        "values": [0, {"bad": value}]
    }
    with pytest.raises(ManifestError, match=r"nested.values\[1\].bad contains a non-finite"):
        validate_manifest(manifest)


def test_nested_result_nonfinite_and_canonical_json_fail_closed():
    from jump_runner.io import canonical_json
    from jump_runner.manifest import validate_json_schema

    result = {
        "schema_version": "jump.run-result/v1",
        "status": "completed",
        "metrics": [],
        "artifacts": [],
        "provenance": {
            "manifest_sha256": "a" * 64,
            "run_id": "nested-result",
            "code_version": "test",
        },
        "metadata": {"nested": [float("nan")]},
    }
    with pytest.raises(ManifestError, match="non-finite"):
        validate_json_schema(result, "run-result-v1.schema.json")
    with pytest.raises(ManifestError, match="strict RFC JSON"):
        canonical_json(result)
