"""Restricted CPU-only Modal boundary for general toy simulations."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import modal

from jump_contracts.experiments import canonical_json, validate_experiment_plan

from .runtime import _execute_validated_source
from .workflow import PreparedExecution

app = modal.App("jump-general-experiment-workbench")
image = modal.Image.debian_slim(python_version="3.11").add_local_dir("src", remote_path="/opt/jump/src")


@app.function(
    image=image,
    cpu=1.0,
    memory=512,
    timeout=30,
    max_containers=1,
    restrict_modal_access=True,
    single_use_containers=True,
    block_network=True,
)
def execute_restricted_simulation(
    plan_value: dict[str, Any],
    source: str,
    confirmation: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate and execute one confirmed, predicted experiment in isolation."""
    plan = validate_experiment_plan(plan_value)
    prediction_sha = hashlib.sha256(canonical_json(prediction)).hexdigest()
    if confirmation != {
        "confirmed": True,
        "plan_sha256": plan["plan_sha256"],
        "prediction_sha256": prediction_sha,
    }:
        raise ValueError("execution requires confirmation bound to this plan and prediction")
    return _execute_validated_source(source, plan)


def execute_prepared_on_modal(prepared: PreparedExecution) -> dict[str, Any]:
    """Spawn exactly one restricted CPU container and return its call identity."""
    if prepared.state != "prediction_ready":
        raise ValueError("prepared execution must be prediction_ready")
    started_at = datetime.now(timezone.utc)
    call = execute_restricted_simulation.spawn(
        prepared.plan,
        prepared.source,
        prepared.confirmation,
        prepared.prediction,
    )
    result = call.get(timeout=35)
    completed_at = datetime.now(timezone.utc)
    return {
        "modal_call_id": call.object_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "result": result,
    }


__all__ = ["app", "execute_restricted_simulation", "execute_prepared_on_modal"]
