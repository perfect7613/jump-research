"""Restricted CPU-only Modal boundary for general toy simulations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal

from .runtime import _execute_validated_source, _validate_remote_plan

if TYPE_CHECKING:
    from .workflow import PreparedExecution

app = modal.App("jump-general-experiment-workbench")


def _code_version() -> str:
    configured = os.environ.get("JUMP_CODE_VERSION")
    if configured:
        return configured
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


CODE_VERSION = _code_version()
simulator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
gateway_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.116.1", "jsonschema==4.26.0")
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
visual_simulator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jsonschema==4.26.0")
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate==1.10.1",
        "huggingface_hub==1.5.0",
        "torch==2.11.0",
        "git+https://github.com/huggingface/transformers.git@918dbf131d0df5b46e3f6e1d96174d62aa4d16d6",
    )
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
model_cache = modal.Volume.from_name("jump-general-gemma-cache-v1", create_if_missing=True)
coordinator_state = modal.Dict.from_name("jump-general-coordinator-state-v1", create_if_missing=True)
visual_coordinator_state = modal.Dict.from_name("jump-visual-coordinator-state-v2", create_if_missing=True)


def _open_visual_result(compressed: bytes) -> dict[str, Any]:
    """Open one bounded zlib stream without allocating beyond the JSON cap."""
    if not isinstance(compressed, bytes) or len(compressed) > 200_000:
        raise ValueError("visual engine returned an invalid compressed result")
    import zlib

    opener = zlib.decompressobj()
    opened = opener.decompress(compressed, 1_000_001)
    if (
        len(opened) > 1_000_000
        or opener.unconsumed_tail
        or not opener.eof
        or opener.unused_data
    ):
        raise ValueError("opened visual result exceeds the canonical JSON cap")
    result = json.loads(opened)
    if not isinstance(result, dict):
        raise ValueError("opened visual result must be a JSON object")
    return result


@app.function(
    image=simulator_image,
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
    plan = _validate_remote_plan(plan_value)
    prediction_sha = hashlib.sha256(
        json.dumps(prediction, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    if confirmation != {
        "confirmed": True,
        "plan_sha256": plan["plan_sha256"],
        "prediction_sha256": prediction_sha,
    }:
        raise ValueError("execution requires confirmation bound to this plan and prediction")
    return _execute_validated_source(source, plan)


@app.function(
    image=visual_simulator_image,
    cpu=1.0,
    memory=512,
    timeout=60,
    max_containers=1,
    restrict_modal_access=True,
    single_use_containers=True,
    block_network=True,
    name="execute_visual_spec_v2",
)
def execute_visual_spec_v2(
    spec_value: dict[str, Any],
    confirmation: dict[str, Any],
    prediction: dict[str, Any],
) -> bytes:
    """Interpret a sealed declarative spec; no generated program is accepted."""
    from jump_contracts.thought_experiments import canonical_json, validate_experiment_spec
    from .visual_engine import execute_visual_spec

    spec = validate_experiment_spec(spec_value)
    prediction_sha = hashlib.sha256(canonical_json(prediction)).hexdigest()
    if confirmation != {
        "confirmed": True,
        "spec_sha256": spec["spec_sha256"],
        "prediction_sha256": prediction_sha,
    }:
        raise ValueError("visual execution requires confirmation bound to the spec and prediction")
    # Modal's restricted-access container cannot upload oversized function
    # outputs to blob storage. Compress the bounded JSON result in-container;
    # the trusted coordinator opens and validates it before constructing a run.
    import zlib

    result = canonical_json(execute_visual_spec(spec))
    if len(result) > 1_000_000:
        raise ValueError("visual result exceeds the one-megabyte canonical JSON cap")
    compressed = zlib.compress(result, level=9)
    if len(compressed) > 200_000:
        raise ValueError("compressed visual result exceeds the transport cap")
    return compressed


def execute_prepared_on_modal(prepared: "PreparedExecution") -> dict[str, Any]:
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


@app.function(
    image=model_image,
    gpu="H100",
    timeout=900,
    max_containers=1,
    volumes={"/hf-cache": model_cache},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="general_frozen_gemma",
)
@modal.concurrent(max_inputs=1)
def general_frozen_gemma(request: dict[str, Any]) -> dict[str, Any]:
    """Private frozen-base model function; no adapter is loaded."""
    from .gemma_planner import generate_with_frozen_gemma

    return generate_with_frozen_gemma(
        request,
        cache_root=Path("/hf-cache"),
        commit_cache=model_cache.commit,
    )


@app.function(
    image=gateway_image,
    cpu=1.0,
    memory=1024,
    timeout=1200,
    max_containers=1,
    name="general_coordinator_compute",
)
@modal.concurrent(max_inputs=1)
def general_coordinator_compute(action: str, body: dict[str, Any]) -> dict[str, Any]:
    """Private CPU coordinator; generated source never crosses the HTTP boundary."""
    from .coordinator import GeneralCoordinator
    from .gemma_planner import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION
    from .workflow import FrozenModel

    model = FrozenModel(model_id=BASE_REPO_ID, revision=BASE_REVISION)

    def generate(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return general_frozen_gemma.remote({"action": kind, "payload": payload})

    def simulate(
        plan: dict[str, Any],
        source: str,
        confirmation: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc)
        call = execute_restricted_simulation.spawn(plan, source, confirmation, prediction)
        result = call.get(timeout=90)
        return {
            "modal_call_id": call.object_id,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc),
            "result": result,
        }

    coordinator = GeneralCoordinator(
        state=coordinator_state,
        model=model,
        transformers_revision=TRANSFORMERS_REVISION,
        model_generate=generate,
        simulate=simulate,
        code_version=CODE_VERSION,
    )
    if action == "plan":
        return coordinator.plan(body)
    if action == "confirm":
        return coordinator.confirm(body)
    raise ValueError("unsupported coordinator action")


@app.function(
    image=gateway_image,
    cpu=1.0,
    memory=1024,
    timeout=1200,
    max_containers=1,
    name="visual_coordinator_compute_v2",
)
@modal.concurrent(max_inputs=1)
def visual_coordinator_compute_v2(action: str, body: dict[str, Any]) -> dict[str, Any]:
    """Compile and execute one confirmed visual ExperimentSpec v2."""
    from .gemma_planner import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION
    from .visual_coordinator import VisualCoordinator
    from .workflow import FrozenModel
    from jump_contracts.thought_experiments import canonical_json

    model = FrozenModel(model_id=BASE_REPO_ID, revision=BASE_REVISION)

    def generate(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return general_frozen_gemma.remote({"action": kind, "payload": payload})

    def simulate(spec: dict[str, Any], prediction: dict[str, Any], prediction_recorded_at: str) -> dict[str, Any]:
        confirmation = {
            "confirmed": True,
            "spec_sha256": spec["spec_sha256"],
            "prediction_sha256": hashlib.sha256(canonical_json(prediction)).hexdigest(),
        }
        started_at = datetime.now(timezone.utc)
        call = execute_visual_spec_v2.spawn(spec, confirmation, prediction)
        compressed = call.get(timeout=70)
        result = _open_visual_result(compressed)
        return {
            "modal_call_id": call.object_id,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc),
            "result": result,
        }

    coordinator = VisualCoordinator(
        state=visual_coordinator_state,
        model=model,
        transformers_revision=TRANSFORMERS_REVISION,
        model_generate=generate,
        simulate=simulate,
        code_version=CODE_VERSION,
    )
    if action == "visual_spec":
        return coordinator.compile(body)
    if action == "visual_confirm":
        return coordinator.confirm(body)
    raise ValueError("unsupported visual coordinator action")


@app.function(
    image=gateway_image,
    cpu=0.5,
    memory=512,
    timeout=1200,
    max_containers=1,
    secrets=[modal.Secret.from_name("jump-authentic-live-auth", required_keys=["JUMP_MODAL_TOKEN"])],
    name="general_experiment_gateway",
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def general_experiment_gateway():
    """Authenticated CPU gateway; unauthenticated requests cannot allocate H100."""
    from .gateway import build_general_gateway
    from .gemma_planner import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION

    async def run_action(action: str, body: dict[str, Any]) -> dict[str, Any]:
        if action in {"visual_spec", "visual_confirm"}:
            return await visual_coordinator_compute_v2.remote.aio(action, body)
        return await general_coordinator_compute.remote.aio(action, body)

    from jump_contracts.thought_experiments import (
        EXPERIMENT_SPEC_SCHEMA_SHA256,
        THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256,
    )

    return build_general_gateway(
        run_action,
        health={
            "status": "available",
            "schema_version": "jump.experiment-question/v1",
            "model": {
                "repo_id": BASE_REPO_ID,
                "revision": BASE_REVISION,
                "transformers_revision": TRANSFORMERS_REVISION,
                "frozen": True,
                "adapter_id": None,
            },
            "code_version": CODE_VERSION,
            "thought_experiments_v2": {
                "question_schema_version": "jump.thought-experiment-question/v2",
                "spec_schema_sha256": EXPERIMENT_SPEC_SCHEMA_SHA256,
                "run_schema_sha256": THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256,
                "engine_id": "jump.declarative-visual-engine/v2",
                "generated_code": False,
                "learned_decoder": False,
            },
        },
    )


__all__ = [
    "app", "execute_restricted_simulation", "execute_prepared_on_modal",
    "general_frozen_gemma", "general_coordinator_compute", "general_experiment_gateway",
    "execute_visual_spec_v2", "visual_coordinator_compute_v2",
]
