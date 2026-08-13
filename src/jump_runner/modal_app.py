"""Modal trust boundary for strictly sequential JUMP experiments."""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import modal

from .errors import RunnerError
from .executor import execute_local_run, read_status, run_manifest, run_root
from .manifest import authorize_launch, manifest_hash, resolve_run, validate_json_schema

APP_NAME = os.environ.get("JUMP_MODAL_APP_NAME", "jump-sequential-experiments")
VOLUME_NAME = os.environ.get("JUMP_MODAL_VOLUME_NAME", "jump-experiment-runs-v1")
VOLUME_PATH = Path("/jump-runs")
CONTROLLER_MAX_CONTAINERS = 1
STAGE_C_REQUIRED_SUBPROCESS_ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "HOME",
    "JUMP_CODE_VERSION",
)


def _code_version() -> str:
    configured = os.environ.get("JUMP_CODE_VERSION")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("JUMP_CODE_VERSION is required when git revision is unavailable")


CODE_VERSION = _code_version()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jsonschema>=4.23,<5", "PyYAML>=6.0,<7")
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
track_h_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jsonschema==4.26.0",
        "numpy==2.3.2",
        "Pillow==11.3.0",
        "safetensors==0.8.0",
        "torch==2.11.0",
    )
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
stage_d_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate==1.10.1",
        "fastapi[standard]==0.116.1",
        "huggingface_hub==1.5.0",
        "jsonschema==4.26.0",
        "numpy==2.3.2",
        "safetensors==0.8.0",
        "torch==2.11.0",
        "git+https://github.com/huggingface/transformers.git@918dbf131d0df5b46e3f6e1d96174d62aa4d16d6",
    )
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
live_gateway_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.116.1")
    .env({"PYTHONPATH": "/opt/jump/src", "JUMP_CODE_VERSION": CODE_VERSION})
    .add_local_dir("src", remote_path="/opt/jump/src")
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
stage_d_live_cache = modal.Volume.from_name("jump-stage-d-live-cache-v1", create_if_missing=True)
dispatch_leases = modal.Dict.from_name("jump-experiment-dispatch-lease-v1", create_if_missing=True)
app = modal.App(APP_NAME)

# JSON mapping of Modal Secret object name -> exhaustive environment key list.
# This contains names only, never secret values. Empty by default.
SECRET_SPECS: dict[str, list[str]] = json.loads(os.environ.get("JUMP_MODAL_SECRET_SPECS", "{}"))
if not isinstance(SECRET_SPECS, dict) or any(
    not isinstance(name, str)
    or not isinstance(keys, list)
    or not keys
    or not all(isinstance(key, str) and key for key in keys)
    for name, keys in SECRET_SPECS.items()
):
    raise RuntimeError("JUMP_MODAL_SECRET_SPECS must map names to nonempty environment-key lists")


@contextmanager
def _dispatch_lease(store: Any):
    """Atomic cross-function lease using Modal Dict.put(skip_if_exists=True).

    A worker crash leaves the lease in place and fails future dispatch closed;
    an operator must inspect and clear that named Dict before resuming.
    """
    token = uuid.uuid4().hex
    if not store.put("global", token, skip_if_exists=True):
        raise RunnerError("another experiment worker holds the global dispatch lease")
    try:
        yield
    finally:
        if store.get("global") == token:
            store.pop("global")


def _execute_authorized_worker(
    resource: str,
    mounted_secret: str | None,
    manifest: dict[str, Any],
    phase_id: str,
    run_id: str,
    smoke: bool,
    confirm_paid: bool,
    confirm_h100: bool,
    volume_root: Path,
    lease_store: Any,
    commit: Any = None,
) -> dict[str, Any]:
    # Workers are independently callable Modal entrypoints, so authorization and
    # canonical run selection are repeated here rather than trusted from a caller.
    authorize_launch(
        manifest, smoke=smoke, confirm_paid=confirm_paid, confirm_h100=confirm_h100
    )
    phase = next((item for item in manifest["phases"] if item["id"] == phase_id), None)
    if phase is None:
        raise ValueError(f"unknown phase id {phase_id!r}")
    original = next((item for item in phase["runs"] if item["id"] == run_id), None)
    if original is None or (smoke and original.get("smoke_test") is not True):
        raise ValueError(f"run {run_id!r} is not selected by the authorized manifest")
    run = resolve_run(manifest.get("defaults", {}), original)
    if run["resources"]["gpu"] != resource or run.get("secret") != mounted_secret:
        raise ValueError("worker resource/secret does not match the authorized manifest run")
    phase = dict(phase)
    phase["_preregistration"] = manifest["preregistration"]
    if mounted_secret:
        specs = {reference["name"]: reference["required_keys"] for reference in manifest.get("secrets", [])}
        if specs.get(mounted_secret) != SECRET_SPECS.get(mounted_secret):
            raise ValueError("mounted secret keys do not match the authorized manifest")
        phase["_secret_keys"] = specs[mounted_secret]
    canonical_path = (
        run_root(volume_root, manifest, "smoke" if smoke else "full")
        / "phases"
        / phase_id
        / "runs"
        / run_id
    )
    with _dispatch_lease(lease_store):
        experiment_root = run_root(volume_root, manifest, "smoke" if smoke else "full")
        for dependency in phase.get("depends_on", []):
            dependency_result = experiment_root / "phases" / dependency / "result.json"
            if not dependency_result.exists() or json.loads(dependency_result.read_text()).get("status") != "passed":
                raise RunnerError(f"dependency phase {dependency!r} has not passed")
        selected = [item for item in phase["runs"] if not smoke or item.get("smoke_test") is True]
        current_index = next(index for index, item in enumerate(selected) if item["id"] == run_id)
        for prior in selected[:current_index]:
            prior_result = experiment_root / "phases" / phase_id / "runs" / prior["id"] / "result.json"
            if not prior_result.exists() or json.loads(prior_result.read_text()).get("status") != "completed":
                raise RunnerError(f"prior run {prior['id']!r} has not completed")
        result = execute_local_run(phase, run, canonical_path, manifest_hash(manifest))
        if commit is not None:
            commit()
        return result


def _worker(
    resource: str,
    mounted_secret: str | None,
    manifest: dict[str, Any],
    phase_id: str,
    run_id: str,
    smoke: bool,
    confirm_paid: bool,
    confirm_h100: bool,
) -> dict[str, Any]:
    result = _execute_authorized_worker(
        resource,
        mounted_secret,
        manifest,
        phase_id,
        run_id,
        smoke,
        confirm_paid,
        confirm_h100,
        VOLUME_PATH,
        dispatch_leases,
        volume.commit,
    )
    return result


def _register_worker(resource: str, secret_name: str | None = None) -> modal.Function:
    gpu = None if resource == "cpu" else resource
    suffix = re.sub(r"[^a-zA-Z0-9_]", "_", secret_name or "none")
    kwargs: dict[str, Any] = {
        "image": image,
        "timeout": 3600,
        "serialized": True,
        "max_containers": 1,
        "volumes": {str(VOLUME_PATH): volume},
        "name": f"execute_{resource.lower().replace('-', '_')}_{suffix}",
    }
    if gpu is not None:
        kwargs["gpu"] = gpu
    if secret_name is not None:
        kwargs["secrets"] = [
            modal.Secret.from_name(secret_name, required_keys=SECRET_SPECS[secret_name])
        ]

    def execute(
        manifest: dict[str, Any],
        phase_id: str,
        run_id: str,
        smoke: bool = False,
        confirm_paid: bool = False,
        confirm_h100: bool = False,
    ) -> dict[str, Any]:
        return _worker(
            resource,
            secret_name,
            manifest,
            phase_id,
            run_id,
            smoke,
            confirm_paid,
            confirm_h100,
        )

    execute = modal.concurrent(max_inputs=1)(execute)
    return app.function(**kwargs)(execute)


RESOURCES = ("cpu", "T4", "L4", "A10", "L40S", "A100-80GB", "H100")
RESOURCE_FUNCTIONS = {(resource, None): _register_worker(resource) for resource in RESOURCES}
for _secret_name in SECRET_SPECS:
    for _resource in RESOURCES:
        RESOURCE_FUNCTIONS[(_resource, _secret_name)] = _register_worker(_resource, _secret_name)


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    max_containers=CONTROLLER_MAX_CONTAINERS,
    volumes={str(VOLUME_PATH): volume},
)
@modal.concurrent(max_inputs=1)
def orchestrate(
    manifest: dict[str, Any],
    smoke: bool = False,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> dict[str, Any]:
    """Authorize remotely and globally serialize controllers before dispatch."""
    authorize_launch(
        manifest, smoke=smoke, confirm_paid=confirm_paid, confirm_h100=confirm_h100
    )
    requested = {reference["name"]: reference["required_keys"] for reference in manifest.get("secrets", [])}
    for name, required_keys in requested.items():
        if SECRET_SPECS.get(name) != required_keys:
            raise ValueError(f"secret {name!r} is not deployed with the exact declared required_keys")

    def remote_executor(
        phase: dict[str, Any], run: dict[str, Any], run_path: Path, manifest_sha256: str
    ) -> dict[str, Any]:
        volume.commit()
        secret_name = run.get("secret")
        worker = RESOURCE_FUNCTIONS[(run["resources"]["gpu"], secret_name)]
        result = worker.remote(
            manifest,
            phase["id"],
            run["id"],
            smoke,
            confirm_paid,
            confirm_h100,
        )
        volume.reload()
        return result

    result = run_manifest(manifest, VOLUME_PATH, smoke=smoke, executor=remote_executor)
    volume.commit()
    return result


@app.function(image=image, timeout=60, volumes={str(VOLUME_PATH): volume})
def get_status(manifest: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    volume.reload()
    return read_status(manifest, VOLUME_PATH, smoke=smoke)


def _validate_track_h_runtime() -> dict[str, Any]:
    """Validate the exact Stage C image dependency before any evidence root write."""
    from importlib.metadata import version

    jsonschema_version = version("jsonschema")
    if jsonschema_version != "4.26.0":
        raise RuntimeError("Stage C image requires jsonschema==4.26.0")
    missing = [key for key in STAGE_C_REQUIRED_SUBPROCESS_ENV_KEYS if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Stage C subprocess environment is missing {missing}")
    if os.environ["JUMP_CODE_VERSION"] != CODE_VERSION:
        raise RuntimeError("Stage C subprocess code identity does not match deployed code")
    validate_json_schema(
        {
            "schema_version": "jump.run-result/v1",
            "status": "completed",
            "attempt": 1,
            "metrics": [],
            "artifacts": [],
            "provenance": {
                "manifest_sha256": "0" * 64,
                "run_id": "stage-c-image-preflight",
                "code_version": CODE_VERSION,
            },
        },
        "run-result-v1.schema.json",
    )
    return {
        "status": "passed",
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "jsonschema_version": jsonschema_version,
        "code_sha": CODE_VERSION,
        "gpu_allocated": False,
        "evidence_root_created": False,
    }


@app.function(
    image=track_h_image,
    timeout=60,
    max_containers=1,
    name="authentic_world_stage_c_preflight",
)
@modal.concurrent(max_inputs=1)
def authentic_world_stage_c_preflight(
    expected_manifest_sha256: str,
    expected_code_sha: str,
) -> dict[str, Any]:
    """CPU-only dependency/schema preflight; this function cannot allocate a GPU."""
    from jump_benchmark.authentic_stage_c import STAGE_C_MANIFEST_SHA256

    if expected_manifest_sha256 != STAGE_C_MANIFEST_SHA256:
        raise RunnerError("Stage C preflight manifest hash mismatch")
    if expected_code_sha != CODE_VERSION:
        raise RunnerError("Stage C preflight code revision mismatch")
    return {**_validate_track_h_runtime(), "manifest_sha256": STAGE_C_MANIFEST_SHA256}


@app.function(
    image=track_h_image,
    timeout=300,
    max_containers=1,
    name="authentic_world_stage_c_task_preflight",
)
@modal.concurrent(max_inputs=1)
def authentic_world_stage_c_task_preflight(
    expected_manifest_sha256: str,
    expected_code_sha: str,
) -> dict[str, Any]:
    """Run the actual allowlisted task through canonical promotion on CPU only."""
    from jump_benchmark.authentic_stage_c import (
        STAGE_C_MANIFEST_SHA256,
        stage_c_run_contract,
    )

    _validate_track_h_runtime()
    if expected_manifest_sha256 != STAGE_C_MANIFEST_SHA256 or expected_code_sha != CODE_VERSION:
        raise RunnerError("Stage C task preflight identity mismatch")
    phase, run = stage_c_run_contract(
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
        dry_run=True,
    )
    root = (
        Path("/tmp")
        / "jump-stage-c-task-preflight"
        / expected_manifest_sha256
        / uuid.uuid4().hex
    )
    result = execute_local_run(phase, run, root, expected_manifest_sha256)
    if result.get("status") != "completed" or result["provenance"]["code_version"] != CODE_VERSION:
        raise RunnerError("Stage C task preflight did not reach canonical completion")
    return {
        "status": "passed",
        "schema_version": result["schema_version"],
        "code_sha": result["provenance"]["code_version"],
        "manifest_sha256": result["provenance"]["manifest_sha256"],
        "attempt": result["attempt"],
        "artifacts_promoted": len(result["artifacts"]),
        "gpu_allocated": False,
        "persistent_root_created": False,
        "git_required": False,
    }


@app.function(
    image=track_h_image,
    gpu="H100",
    timeout=10_800,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
    name="authentic_world_stage_c",
)
@modal.concurrent(max_inputs=1)
def authentic_world_stage_c(
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> dict[str, Any]:
    """Run the frozen three-seed predictive-world pilot in one serial call."""
    from jump_benchmark.authentic_stage_c import (
        STAGE_C_MANIFEST_SHA256,
        authorize_stage_c_launch,
        stage_c_run_contract,
    )
    _validate_track_h_runtime()
    authorize_stage_c_launch(
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
        actual_code_sha=CODE_VERSION,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    run_root_path = VOLUME_PATH / "authentic-world-stage-c" / STAGE_C_MANIFEST_SHA256 / "run"
    phase, run = stage_c_run_contract(
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
    )
    with _dispatch_lease(dispatch_leases):
        result = execute_local_run(
            phase,
            run,
            run_root_path,
            expected_manifest_sha256,
        )
        volume.commit()
        if result.get("status") != "completed":
            raise RunnerError(f"Stage C task failed: {result.get('error')}")
        return result


def _authorize_long_horizon_launch(
    *,
    mode: str,
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool,
    confirm_h100: bool,
) -> dict[str, Any]:
    from jump_benchmark.long_horizon import long_horizon_manifest, manifest_sha256

    if confirm_paid is not True or confirm_h100 is not True:
        raise RunnerError("long-horizon Phase A requires literal paid and H100 confirmations")
    if expected_manifest_sha256 != manifest_sha256(mode) or expected_code_sha != CODE_VERSION:
        raise RunnerError("long-horizon immutable identity mismatch")
    execution = long_horizon_manifest(mode)["execution"]
    forecast = execution["timeout_seconds"] / 3600 * execution["h100_rate_usd_per_hour"]
    if (
        execution["resource"] != "H100"
        or execution["gpu_count"] != 1
        or execution["max_containers"] != 1
        or execution["max_inputs"] != 1
        or execution["max_attempts"] != 1
        or execution["serial"] is not True
        or abs(forecast - execution["forecast_usd"]) > 1e-9
        or forecast > execution["aggregate_authority_ceiling_usd"]
    ):
        raise RunnerError("long-horizon resource or cost contract mismatch")
    return execution


@app.function(
    image=track_h_image,
    timeout=300,
    max_containers=1,
    name="authentic_world_long_horizon_preflight",
)
@modal.concurrent(max_inputs=1)
def authentic_world_long_horizon_preflight(
    mode: str,
    expected_manifest_sha256: str,
    expected_code_sha: str,
) -> dict[str, Any]:
    """Exercise the actual task/promotion seam on a tiny CPU workload."""
    from jump_benchmark.long_horizon import manifest_sha256, run_contract

    _validate_track_h_runtime()
    if expected_manifest_sha256 != manifest_sha256(mode) or expected_code_sha != CODE_VERSION:
        raise RunnerError("long-horizon preflight identity mismatch")
    phase, run = run_contract(
        mode=mode,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
        dry_run=True,
    )
    root = Path("/tmp") / "jump-long-horizon-preflight" / uuid.uuid4().hex
    result = execute_local_run(phase, run, root, expected_manifest_sha256)
    if result.get("status") != "completed" or result["provenance"]["code_version"] != CODE_VERSION:
        raise RunnerError("long-horizon canonical CPU preflight failed")
    return {
        "status": "passed",
        "mode": mode,
        "manifest_sha256": expected_manifest_sha256,
        "code_sha": expected_code_sha,
        "schema_version": result["schema_version"],
        "artifacts_promoted": len(result["artifacts"]),
        "gpu_allocated": False,
        "persistent_root_created": False,
    }


@app.function(
    image=track_h_image,
    gpu="H100",
    timeout=7200,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
    name="authentic_world_long_horizon",
)
@modal.concurrent(max_inputs=1)
def authentic_world_long_horizon(
    mode: str,
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> dict[str, Any]:
    """Run one frozen Phase-A mode; pilot and replication have distinct roots."""
    from jump_benchmark.long_horizon import run_contract

    _validate_track_h_runtime()
    _authorize_long_horizon_launch(
        mode=mode,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    root = VOLUME_PATH / "authentic-world-long-horizon" / expected_manifest_sha256 / "run"
    phase, run = run_contract(
        mode=mode,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
    )
    with _dispatch_lease(dispatch_leases):
        result = execute_local_run(phase, run, root, expected_manifest_sha256)
        volume.commit()
        if result.get("status") != "completed":
            raise RunnerError(f"long-horizon task failed: {result.get('error')}")
        return result


@app.function(
    image=stage_d_image,
    timeout=300,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_world_long_horizon_stage_b_preflight",
)
@modal.concurrent(max_inputs=1)
def authentic_world_long_horizon_stage_b_preflight(
    expected_manifest_sha256: str,
    expected_code_sha: str,
) -> dict[str, Any]:
    """CPU/config-only Phase-B preflight; never loads base weights or allocates GPU."""
    import hashlib
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer
    from jump_benchmark.authentic import build_gated_residual_projector
    from jump_benchmark.authentic_stage_d import BASE_REPO_ID, BASE_REVISION, assert_prompt_identity
    from jump_benchmark.long_horizon import LATENT_DIM, build_long_horizon_modules
    from jump_benchmark.long_horizon_stage_b import (
        SOURCE_DECODER_SHA256,
        SOURCE_ENCODER_SHA256,
        SOURCE_RELATIVE_ROOT,
        STAGE_B_MANIFEST_SHA256,
        _encode,
        dynamic_32d_control_preflight,
        matched_pair_cpu_generation_preflight,
    )
    from safetensors.torch import load_file

    if expected_manifest_sha256 != STAGE_B_MANIFEST_SHA256 or expected_code_sha != CODE_VERSION:
        raise RunnerError("Phase B preflight identity mismatch")
    source = VOLUME_PATH / SOURCE_RELATIVE_ROOT
    encoder_path, decoder_path = source / "encoder.safetensors", source / "decoder.safetensors"
    if hashlib.sha256(encoder_path.read_bytes()).hexdigest() != SOURCE_ENCODER_SHA256 or hashlib.sha256(decoder_path.read_bytes()).hexdigest() != SOURCE_DECODER_SHA256:
        raise RunnerError("Phase B preflight source checksum mismatch")
    encoder, decoder = build_long_horizon_modules()
    encoder.load_state_dict(load_file(encoder_path), strict=True)
    decoder.load_state_dict(load_file(decoder_path), strict=True)
    from jump_benchmark.authentic import matched_world_pair
    from jump_benchmark.simulator import SimulatorConfig
    pair = matched_world_pair(pair_seed=33173, config=SimulatorConfig(steps=12))
    paired_z, paired_observation = _encode(encoder, pair["a"], "cpu")
    if tuple(paired_z.shape) != (1, LATENT_DIM) or paired_observation.sha256() != pair["a"]["encoder_input_sha256"]:
        raise RunnerError("Phase B matched-pair observation-only encoder path mismatch")
    config = AutoConfig.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION, trust_remote_code=False)
    model_class = AutoModelForMultimodalLM._model_mapping[type(config)]
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION, trust_remote_code=False)
    matched_pair_seam = matched_pair_cpu_generation_preflight(encoder, decoder, tokenizer)
    prompt = assert_prompt_identity(tokenizer)
    hidden_size = int(config.text_config.hidden_size)
    projector = build_gated_residual_projector(hidden_size, latent_dim=LATENT_DIM)
    if projector.projector.weight.shape != (hidden_size, LATENT_DIM):
        raise RunnerError("Phase B projector shape mismatch")
    control_seam = dynamic_32d_control_preflight()
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        phase = {
            "id": "long-horizon-stage-b",
            "_preregistration": {"layer_allowlist": [0], "timepoint_allowlist": ["answer"]},
        }
        run = {
            "id": "long-horizon-stage-b",
            "task": {"module": "jump_benchmark.long_horizon_stage_b_task", "parameters": {
                "expected_manifest_sha256": expected_manifest_sha256,
                "expected_code_sha": expected_code_sha,
                "dry_run": True,
            }},
            "resources": {"gpu": "cpu", "timeout_seconds": 60},
            "selection": {"layers": [], "timepoints": []},
            "retry": {"max_attempts": 1},
        }
        dry_result = execute_local_run(
            phase, run, Path(temporary) / "canonical-preflight-run", expected_manifest_sha256
        )
        if dry_result.get("status") != "completed" or len(dry_result.get("artifacts", [])) != 1:
            raise RunnerError("Phase B canonical executor/artifact promotion preflight failed")
    return {
        "status": "passed",
        "manifest_sha256": STAGE_B_MANIFEST_SHA256,
        "code_sha": CODE_VERSION,
        "model_type": config.model_type,
        "model_class": model_class.__name__,
        "hidden_size": hidden_size,
        "latent_dim": LATENT_DIM,
        "prompt_binding": prompt,
        "dynamic_32d_six_arm_seam": control_seam,
        "executor_precreated_empty_work_root": True,
        "canonical_artifact_promotion": True,
        "source_encoder_sha256": SOURCE_ENCODER_SHA256,
        "source_decoder_sha256": SOURCE_DECODER_SHA256,
        "matched_pair_observation_only_encode": True,
        "matched_pair_six_arm_generation_and_scoring": matched_pair_seam,
        "base_weights_loaded": False,
        "gpu_allocated": False,
        "persistent_root_created": False,
    }


def _authorize_long_horizon_stage_b(
    *, expected_manifest_sha256: str, expected_code_sha: str,
    confirm_paid: bool, confirm_h100: bool,
) -> dict[str, Any]:
    from jump_benchmark.long_horizon_stage_b import STAGE_B_MANIFEST_SHA256, stage_b_manifest
    if confirm_paid is not True or confirm_h100 is not True:
        raise RunnerError("Phase B requires literal paid and H100 confirmations")
    if expected_manifest_sha256 != STAGE_B_MANIFEST_SHA256 or expected_code_sha != CODE_VERSION:
        raise RunnerError("Phase B immutable identity mismatch")
    execution = stage_b_manifest()["execution"]
    forecast = execution["timeout_seconds"] / 3600 * execution["h100_rate_usd_per_hour"]
    if forecast != execution["forecast_usd"] or forecast > execution["aggregate_authority_ceiling_usd"]:
        raise RunnerError("Phase B spend contract mismatch")
    return execution


@app.function(
    image=stage_d_image,
    gpu="H100",
    timeout=7200,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_world_long_horizon_stage_b",
)
@modal.concurrent(max_inputs=1)
def authentic_world_long_horizon_stage_b(
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> dict[str, Any]:
    from jump_benchmark.long_horizon_stage_b import run_contract
    _authorize_long_horizon_stage_b(
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    root = VOLUME_PATH / "authentic-world-long-horizon-stage-b" / expected_manifest_sha256 / "run"
    phase, run = run_contract(expected_manifest_sha256, expected_code_sha)
    with _dispatch_lease(dispatch_leases):
        result = execute_local_run(phase, run, root, expected_manifest_sha256)
        volume.commit()
        if result.get("status") != "completed":
            raise RunnerError(f"Phase B failed: {result.get('error')}")
        return result



def _authorize_behavioral_distillation(*, expected_manifest_sha256: str, expected_code_sha: str, confirm_paid: bool, confirm_h100: bool) -> dict[str, Any]:
    from jump_benchmark.behavioral_distillation import MANIFEST_SHA256, behavioral_distillation_manifest
    if confirm_paid is not True or confirm_h100 is not True:
        raise RunnerError("behavioral distillation requires literal paid and H100 confirmations")
    if expected_manifest_sha256 != MANIFEST_SHA256 or expected_code_sha != CODE_VERSION:
        raise RunnerError("behavioral-distillation immutable identity mismatch")
    execution = behavioral_distillation_manifest()["execution"]
    forecast = execution["timeout_seconds"] / 3600 * execution["h100_rate_usd_per_hour"]
    if (
        execution["resource"] != "H100" or execution["gpu_count"] != 1
        or execution["max_containers"] != 1 or execution["max_inputs"] != 1
        or execution["max_attempts"] != 1 or abs(forecast - execution["forecast_usd"]) > 1e-9
        or forecast > execution["aggregate_authority_ceiling_usd"]
    ):
        raise RunnerError("behavioral-distillation resource or cost contract mismatch")
    return execution


@app.function(
    image=stage_d_image,
    timeout=300,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_world_behavioral_distillation_preflight",
)
@modal.concurrent(max_inputs=1)
def authentic_world_behavioral_distillation_preflight(expected_manifest_sha256: str, expected_code_sha: str) -> dict[str, Any]:
    """Same-image CPU/config/task-promotion preflight; never loads Gemma weights."""
    import hashlib
    import tempfile
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer
    from safetensors.torch import load_file
    from jump_benchmark.behavioral_distillation import MANIFEST_SHA256, cpu_preflight, run_contract
    from jump_benchmark.authentic_stage_d import BASE_REPO_ID, BASE_REVISION
    from jump_benchmark.long_horizon import build_long_horizon_modules
    from jump_benchmark.long_horizon_stage_b import SOURCE_DECODER_SHA256, SOURCE_ENCODER_SHA256, SOURCE_RELATIVE_ROOT
    if expected_manifest_sha256 != MANIFEST_SHA256 or expected_code_sha != CODE_VERSION:
        raise RunnerError("behavioral-distillation preflight identity mismatch")
    source=VOLUME_PATH/SOURCE_RELATIVE_ROOT; ep=source/"encoder.safetensors"; dp=source/"decoder.safetensors"
    if hashlib.sha256(ep.read_bytes()).hexdigest()!=SOURCE_ENCODER_SHA256 or hashlib.sha256(dp.read_bytes()).hexdigest()!=SOURCE_DECODER_SHA256:
        raise RunnerError("behavioral-distillation source checksum mismatch")
    encoder,decoder=build_long_horizon_modules();encoder.load_state_dict(load_file(ep),strict=True);decoder.load_state_dict(load_file(dp),strict=True);encoder.eval();decoder.eval()
    tokenizer=AutoTokenizer.from_pretrained(BASE_REPO_ID,revision=BASE_REVISION,trust_remote_code=False)
    config=AutoConfig.from_pretrained(BASE_REPO_ID,revision=BASE_REVISION,trust_remote_code=False)
    model_class=AutoModelForMultimodalLM._model_mapping[type(config)]
    seam=cpu_preflight(tokenizer,encoder,decoder)
    phase,run=run_contract(expected_manifest_sha256,expected_code_sha,dry_run=True)
    with tempfile.TemporaryDirectory() as temporary:
        result=execute_local_run(phase,run,Path(temporary)/"behavioral-distillation-preflight",expected_manifest_sha256)
    if result.get("status")!="completed" or result["provenance"]["code_version"]!=CODE_VERSION:
        raise RunnerError("behavioral-distillation canonical preflight failed")
    return {"status":"passed","manifest_sha256":MANIFEST_SHA256,"code_sha":CODE_VERSION,"model_type":config.model_type,"model_class":model_class.__name__,"seam":seam,"base_weights_loaded":False,"gpu_allocated":False,"persistent_root_created":False}


@app.function(
    image=stage_d_image,
    gpu="H100",
    timeout=3600,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_world_behavioral_distillation",
)
@modal.concurrent(max_inputs=1)
def authentic_world_behavioral_distillation(expected_manifest_sha256: str, expected_code_sha: str, confirm_paid: bool=False, confirm_h100: bool=False) -> dict[str, Any]:
    from jump_benchmark.behavioral_distillation import run_contract
    _authorize_behavioral_distillation(expected_manifest_sha256=expected_manifest_sha256,expected_code_sha=expected_code_sha,confirm_paid=confirm_paid,confirm_h100=confirm_h100)
    root=VOLUME_PATH/"authentic-world-behavioral-distillation"/expected_manifest_sha256/"run"
    phase,run=run_contract(expected_manifest_sha256,expected_code_sha)
    with _dispatch_lease(dispatch_leases):
        result=execute_local_run(phase,run,root,expected_manifest_sha256);volume.commit()
        if result.get("status")!="completed": raise RunnerError(f"behavioral distillation failed: {result.get('error')}")
        return result


@app.local_entrypoint(name="submit-behavioral-distillation")
def submit_behavioral_distillation(expected_manifest_sha256: str, expected_code_sha: str, confirm_paid: bool=False, confirm_h100: bool=False) -> None:
    execution=_authorize_behavioral_distillation(expected_manifest_sha256=expected_manifest_sha256,expected_code_sha=expected_code_sha,confirm_paid=confirm_paid,confirm_h100=confirm_h100)
    call=authentic_world_behavioral_distillation.spawn(expected_manifest_sha256,expected_code_sha,confirm_paid=True,confirm_h100=True)
    record={"app_name":APP_NAME,"function":"authentic_world_behavioral_distillation","call_id":call.object_id,"manifest_sha256":expected_manifest_sha256,"code_sha":expected_code_sha,"forecast_usd":execution["forecast_usd"],"hard_ceiling_usd":execution["aggregate_authority_ceiling_usd"]}
    registry=Path(".jump/submissions");registry.mkdir(parents=True,exist_ok=True);path=registry/f"{call.object_id}.json"
    with path.open("x") as handle:
        handle.write(json.dumps(record,indent=2,sort_keys=True)+"\n");handle.flush();os.fsync(handle.fileno())
    fd=os.open(registry,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    print(json.dumps({**record,"record_path":str(path)},sort_keys=True))


def _authorize_object_jepa(*, expected_manifest_sha256: str, expected_code_sha: str, confirm_paid: bool, confirm_h100: bool) -> dict[str, Any]:
    from jump_benchmark.object_jepa import MANIFEST_SHA256, object_jepa_manifest
    if confirm_paid is not True or confirm_h100 is not True: raise RunnerError("object JEPA requires literal paid and H100 confirmations")
    if expected_manifest_sha256!=MANIFEST_SHA256 or expected_code_sha!=CODE_VERSION: raise RunnerError("object JEPA immutable identity mismatch")
    execution=object_jepa_manifest()["execution"];forecast=execution["timeout_seconds"]/3600*execution["h100_rate_usd_per_hour"]
    if execution["resource"]!="H100" or execution["gpu_count"]!=1 or execution["max_containers"]!=1 or execution["max_inputs"]!=1 or execution["max_attempts"]!=1 or abs(forecast-execution["forecast_usd"])>1e-9 or forecast>execution["aggregate_authority_ceiling_usd"]: raise RunnerError("object JEPA resource/cost contract mismatch")
    return execution

@app.function(image=track_h_image,timeout=300,max_containers=1,name="authentic_world_object_jepa_preflight")
@modal.concurrent(max_inputs=1)
def authentic_world_object_jepa_preflight(expected_manifest_sha256: str,expected_code_sha: str)->dict[str,Any]:
    from jump_benchmark.object_jepa import MANIFEST_SHA256,cpu_preflight
    if expected_manifest_sha256!=MANIFEST_SHA256 or expected_code_sha!=CODE_VERSION:raise RunnerError("object JEPA preflight identity mismatch")
    return {"status":"passed","manifest_sha256":MANIFEST_SHA256,"code_sha":CODE_VERSION,"seam":cpu_preflight(),"gpu_allocated":False,"persistent_root_created":False}

@app.function(image=track_h_image,gpu="H100",timeout=3600,max_containers=1,volumes={str(VOLUME_PATH):volume},name="authentic_world_object_jepa_pilot")
@modal.concurrent(max_inputs=1)
def authentic_world_object_jepa_pilot(expected_manifest_sha256: str,expected_code_sha: str,confirm_paid: bool=False,confirm_h100: bool=False)->dict[str,Any]:
    from jump_benchmark.object_jepa import run_contract
    _authorize_object_jepa(expected_manifest_sha256=expected_manifest_sha256,expected_code_sha=expected_code_sha,confirm_paid=confirm_paid,confirm_h100=confirm_h100)
    root=VOLUME_PATH/"authentic-world-object-jepa"/expected_manifest_sha256/"run";phase,run=run_contract(expected_manifest_sha256,expected_code_sha)
    with _dispatch_lease(dispatch_leases):
        result=execute_local_run(phase,run,root,expected_manifest_sha256);volume.commit()
        if result.get("status")!="completed":raise RunnerError(f"object JEPA failed: {result.get('error')}")
        return result

@app.local_entrypoint(name="submit-long-horizon")
def submit_long_horizon(
    mode: str,
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> None:
    execution = _authorize_long_horizon_launch(
        mode=mode,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    call = authentic_world_long_horizon.spawn(
        mode,
        expected_manifest_sha256,
        expected_code_sha,
        confirm_paid=True,
        confirm_h100=True,
    )
    record = {
        "app_name": APP_NAME,
        "function": "authentic_world_long_horizon",
        "call_id": call.object_id,
        "mode": mode,
        "manifest_sha256": expected_manifest_sha256,
        "code_sha": expected_code_sha,
        "forecast_usd": execution["forecast_usd"],
        "aggregate_authority_ceiling_usd": execution["aggregate_authority_ceiling_usd"],
    }
    registry = Path(".jump/submissions")
    registry.mkdir(parents=True, exist_ok=True)
    record_path = registry / f"{call.object_id}.json"
    with record_path.open("x") as handle:
        handle.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(registry, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({**record, "record_path": str(record_path)}, sort_keys=True))


def _authorize_stage_d_launch(
    expected_manifest_sha256: str,
    expected_code_sha: str,
    *,
    confirm_paid: bool,
    confirm_h100: bool,
) -> dict[str, Any]:
    from jump_benchmark.authentic_stage_d import STAGE_D_MANIFEST_SHA256, stage_d_manifest
    from jump_mechanistic import STAGE_D_EXECUTION_CONTRACT_SHA256

    if confirm_paid is not True or confirm_h100 is not True:
        raise RunnerError("Stage D requires literal paid and H100 confirmations")
    if expected_manifest_sha256 != STAGE_D_MANIFEST_SHA256 or expected_code_sha != CODE_VERSION:
        raise RunnerError("Stage D immutable identity mismatch")
    execution = stage_d_manifest()["execution"]
    forecast = execution["timeout_seconds"] / 3600 * execution["h100_rate_usd_per_hour"]
    if (
        execution["resource"] != "H100"
        or execution["gpu_count"] != 1
        or execution["max_containers"] != 1
        or execution["max_inputs"] != 1
        or execution["max_attempts"] != 1
        or stage_d_manifest()["evaluation"]["execution_contract_sha256"]
        != STAGE_D_EXECUTION_CONTRACT_SHA256
        or abs(forecast - execution["retry_aware_forecast_usd"]) > 1e-9
        or forecast > execution["hard_ceiling_usd"]
    ):
        raise RunnerError("Stage D resource/cost contract mismatch")
    return execution


@app.function(
    image=stage_d_image,
    timeout=300,
    max_containers=1,
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_world_stage_d_preflight",
)
@modal.concurrent(max_inputs=1)
def authentic_world_stage_d_preflight(expected_manifest_sha256: str, expected_code_sha: str) -> dict[str, Any]:
    """CPU/config-only Gemma and projector preflight; never loads base weights."""
    from importlib.metadata import version
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer
    from jump_benchmark.authentic_stage_d import (
        BASE_REPO_ID,
        BASE_REVISION,
        STAGE_D_MANIFEST_SHA256,
        TRANSFORMERS_REVISION,
        assert_prompt_identity,
        stage_d_cpu_preflight,
    )

    if expected_manifest_sha256 != STAGE_D_MANIFEST_SHA256 or expected_code_sha != CODE_VERSION:
        raise RunnerError("Stage D preflight identity mismatch")
    config = AutoConfig.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION, trust_remote_code=False)
    model_class = AutoModelForMultimodalLM._model_mapping[type(config)]
    if (
        config.model_type != "gemma4_unified"
        or model_class.__name__ != "Gemma4UnifiedForConditionalGeneration"
    ):
        raise RunnerError("pinned Gemma config does not resolve through AutoModelForMultimodalLM")
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION)
    result = stage_d_cpu_preflight(tokenizer, hidden_size=int(config.text_config.hidden_size))
    assert_prompt_identity(tokenizer)
    return {
        **result,
        "code_sha": CODE_VERSION,
        "base_revision": BASE_REVISION,
        "transformers_source_revision": TRANSFORMERS_REVISION,
        "transformers_version": version("transformers"),
        "model_type": config.model_type,
        "model_class": model_class.__name__,
        "weights_loaded": False,
    }


@app.function(
    image=stage_d_image,
    gpu="H100",
    timeout=7200,
    max_containers=1,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_world_stage_d",
)
@modal.concurrent(max_inputs=1)
def authentic_world_stage_d(
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> dict[str, Any]:
    """Execute the one frozen Stage D projector/control pilot canonically."""
    from jump_benchmark.authentic_stage_d import STAGE_D_MANIFEST_SHA256

    _authorize_stage_d_launch(
        expected_manifest_sha256,
        expected_code_sha,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    root = VOLUME_PATH / "authentic-world-stage-d" / STAGE_D_MANIFEST_SHA256 / "run"
    phase = {
        "id": "stage-d", "_secret_keys": ["HF_TOKEN"],
        "_preregistration": {"layer_allowlist": [0], "timepoint_allowlist": ["answer"]},
    }
    run = {
        "id": "authentic-world-stage-d",
        "task": {"module": "jump_benchmark.stage_d_task", "parameters": {"expected_manifest_sha256": expected_manifest_sha256, "expected_code_sha": expected_code_sha}},
        "resources": {"gpu": "H100", "timeout_seconds": 7200},
        "selection": {"layers": [], "timepoints": []}, "retry": {"max_attempts": 1},
    }
    with _dispatch_lease(dispatch_leases):
        result = execute_local_run(phase, run, root, expected_manifest_sha256)
        volume.commit()
        if result.get("status") != "completed":
            raise RunnerError(f"Stage D task failed: {result.get('error')}")
        return result


@app.local_entrypoint(name="submit-stage-d")
def submit_stage_d(
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> None:
    """Authorize once, spawn once, and fsync the call identity before exit."""
    execution = _authorize_stage_d_launch(
        expected_manifest_sha256,
        expected_code_sha,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    call = authentic_world_stage_d.spawn(
        expected_manifest_sha256,
        expected_code_sha,
        confirm_paid=True,
        confirm_h100=True,
    )
    record = {
        "app_name": APP_NAME,
        "function": "authentic_world_stage_d",
        "call_id": call.object_id,
        "manifest_sha256": expected_manifest_sha256,
        "code_sha": expected_code_sha,
        "retry_aware_forecast_usd": execution["retry_aware_forecast_usd"],
        "hard_ceiling_usd": execution["hard_ceiling_usd"],
    }
    registry = Path(".jump/submissions")
    registry.mkdir(parents=True, exist_ok=True)
    record_path = registry / f"{call.object_id}.json"
    with record_path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(registry, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({**record, "record_path": str(record_path)}, sort_keys=True))


@app.function(
    image=live_gateway_image,
    timeout=900,
    max_containers=1,
    secrets=[
        modal.Secret.from_name("jump-authentic-live-auth", required_keys=["JUMP_MODAL_TOKEN"]),
    ],
    name="authentic_stage_d_live",
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def authentic_stage_d_live():
    """CPU-only authenticated gateway; unauthenticated traffic cannot allocate GPU."""
    from jump_runner.stage_d_live import build_live_gateway

    async def run_compute(body: dict[str, Any]) -> dict[str, Any]:
        return await authentic_stage_d_live_compute.remote.aio(body)

    return build_live_gateway(run_compute)


@app.function(
    image=stage_d_image,
    gpu="H100",
    timeout=900,
    max_containers=1,
    volumes={"/hf-cache": stage_d_live_cache},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_stage_d_live_compute",
)
@modal.concurrent(max_inputs=1)
def authentic_stage_d_live_compute(body: dict[str, Any]) -> dict[str, Any]:
    from jump_runner.stage_d_live import live_compute

    return live_compute(body, cache_root=Path("/hf-cache"), commit_cache=stage_d_live_cache.commit)


@app.function(
    image=stage_d_image,
    timeout=300,
    max_containers=1,
    volumes={"/hf-cache": stage_d_live_cache},
    name="authentic_stage_d_live_http_preflight",
)
@modal.concurrent(max_inputs=1)
def authentic_stage_d_live_http_preflight() -> dict[str, Any]:
    from jump_runner.stage_d_live import http_boundary_preflight

    return http_boundary_preflight(Path("/hf-cache"), stage_d_live_cache.commit)


@app.function(
    image=stage_d_image,
    gpu="H100",
    timeout=900,
    max_containers=1,
    volumes={"/hf-cache": stage_d_live_cache},
    secrets=[modal.Secret.from_name("jump-hf-read", required_keys=["HF_TOKEN"])],
    name="authentic_stage_d_injection_diagnostic",
)
@modal.concurrent(max_inputs=1)
def authentic_stage_d_injection_diagnostic() -> dict[str, Any]:
    from jump_runner.stage_d_live import injection_sensitivity_diagnostic

    return injection_sensitivity_diagnostic(Path("/hf-cache"), stage_d_live_cache.commit)


@app.local_entrypoint(name="submit-stage-c")
def submit_stage_c(
    expected_manifest_sha256: str,
    expected_code_sha: str,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> None:
    """Validate locally, spawn exactly once, and persist the call ID before exit."""
    from jump_benchmark.authentic_stage_c import authorize_stage_c_launch

    plan = authorize_stage_c_launch(
        expected_manifest_sha256=expected_manifest_sha256,
        expected_code_sha=expected_code_sha,
        actual_code_sha=CODE_VERSION,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    call = authentic_world_stage_c.spawn(
        expected_manifest_sha256,
        expected_code_sha,
        confirm_paid=True,
        confirm_h100=True,
    )
    record = {
        "app_name": APP_NAME,
        "function": "authentic_world_stage_c",
        "call_id": call.object_id,
        "manifest_sha256": expected_manifest_sha256,
        "code_sha": expected_code_sha,
        "experiment_id": plan["experiment_id"],
    }
    registry = Path(".jump/submissions")
    registry.mkdir(parents=True, exist_ok=True)
    record_path = registry / f"{call.object_id}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**record, "record_path": str(record_path)}, sort_keys=True))


@app.local_entrypoint(name="submit")
def modal_submit(
    manifest_path: str,
    smoke: bool = False,
    confirm_paid: bool = False,
    confirm_h100: bool = False,
) -> None:
    from .manifest import load_manifest

    manifest = load_manifest(manifest_path)
    authorize_launch(
        manifest, smoke=smoke, confirm_paid=confirm_paid, confirm_h100=confirm_h100
    )
    call = orchestrate.spawn(
        manifest,
        smoke=smoke,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    print(json.dumps({"call_id": call.object_id, "experiment_id": manifest["experiment_id"]}))


@app.local_entrypoint(name="status")
def modal_status(manifest_path: str, smoke: bool = False) -> None:
    from .manifest import load_manifest

    print(json.dumps(get_status.remote(load_manifest(manifest_path), smoke=smoke), indent=2, sort_keys=True))
