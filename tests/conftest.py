from __future__ import annotations

import copy

import pytest


@pytest.fixture
def manifest() -> dict:
    return {
        "schema_version": "jump.experiments/v1",
        "experiment_id": "test-experiment",
        "launch_policy": {"allow_full_matrix": False},
        "preregistration": {"layer_allowlist": [4, 12], "timepoint_allowlist": ["T1", "T4"]},
        "defaults": {
            "resources": {"gpu": "T4", "timeout_seconds": 5},
            "retry": {"max_attempts": 1, "retry_on_exit_codes": [7]},
        },
        "phases": [
            {
                "id": "pilot",
                "budget": {
                    "allowed_gpu_types": ["T4"],
                    "max_concurrent_gpus": 1,
                    "max_runtime_seconds": 10,
                    "max_cost_usd": 1,
                    "gpu_hourly_cost_usd": {"T4": 0.5},
                },
                "runs": [
                    {
                        "id": "pilot-1",
                        "smoke_test": True,
                        "task": {
                            "module": "jump_runner.mock_task",
                            "parameters": {
                                "checkpoint_id": "mock",
                                "layers": [4],
                                "timepoints": ["T4"],
                                "metric_name": "accuracy",
                                "metric_value": 0.75,
                            },
                        },
                    }
                ],
                "gates": [{"id": "pilot-pass", "metric": "accuracy", "operator": ">=", "threshold": 0.5}],
            }
        ],
    }


@pytest.fixture
def cloned_manifest(manifest):
    return copy.deepcopy(manifest)
