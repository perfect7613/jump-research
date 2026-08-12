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
from .manifest import authorize_launch, manifest_hash, resolve_run

APP_NAME = os.environ.get("JUMP_MODAL_APP_NAME", "jump-sequential-experiments")
VOLUME_NAME = os.environ.get("JUMP_MODAL_VOLUME_NAME", "jump-experiment-runs-v1")
VOLUME_PATH = Path("/jump-runs")
CONTROLLER_MAX_CONTAINERS = 1


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
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
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
