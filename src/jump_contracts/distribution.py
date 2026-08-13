"""Manifest-only contract for the canonical JUMP world-model distribution.

The Hugging Face repository contains weights and configuration, never a second
copy of model implementation code.  Loaders in application packages validate
this manifest and its files before instantiating their existing model classes.
Repository revisions are supplied by the loader and recorded after resolution;
they are deliberately not embedded in the same commit they would identify.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .evidence import (
    EvidenceError,
    canonical_json,
    is_sha256,
    normalize_json_object,
    sha256_file,
)


CANONICAL_WORLD_MODEL_REPO_ID = "Perfect7613/jump-world-model"
WORLD_MODEL_COMPONENT_MANIFEST_VERSION = "jump.world-model-components/v1"
WORLD_MODEL_LOAD_CONTRACT_VERSION = "jump.world-model-load/v1"
WORLD_MODEL_LOAD_RECORD_VERSION = "jump.world-model-load-record/v1"
SUPPORTED_TRANSFORMERS_REVISION = "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"
WORLD_MODEL_COMPONENT_ROLES = (
    "encoder",
    "decoder",
    "future_projector",
    "gemma_adapter",
)
WORLD_MODEL_LOAD_MODES = ("artifact_only", "gated_gemma")

_COMPONENT_FIELDS = {
    "architecture",
    "directory",
    "weights",
    "config",
    "identity_sha256",
}
_FILE_FIELDS = {"path", "sha256", "format"}
_LATENT_DTYPES = frozenset({"float16-le", "bfloat16-le", "float32-le", "float64-le"})
_CLAIM_GUARD_FIELDS = {
    "engineering_only",
    "behavioral_claim_allowed",
    "mechanistic_claim_allowed",
    "causal_claim_allowed",
    "benchmark_law_accuracy_claim_allowed",
    "track_r_claim_allowed",
}


def component_identity(
    *,
    directory: str,
    weights_path: str,
    weights_sha256: str,
    config_path: str,
    config_sha256: str,
    architecture: str,
) -> dict[str, Any]:
    """Build one code-free component identity from colocated weights/config."""
    core = {
        "architecture": architecture,
        "directory": directory,
        "weights": {
            "path": weights_path,
            "sha256": weights_sha256,
            "format": "safetensors",
        },
        "config": {
            "path": config_path,
            "sha256": config_sha256,
            "format": "json",
        },
    }
    value = {**core, "identity_sha256": _sha256_json(core)}
    _validate_component(value, role=None)
    return value


def build_world_model_component_manifest(
    *,
    base_model_repo_id: str,
    base_model_revision: str,
    transformers_revision: str,
    latent_dtype: str,
    latent_shape: Sequence[int],
    latent_order: str,
    projector_input_dim: int,
    projector_output_dim: int,
    projector_gate: str,
    injection_layer: int,
    injection_site: str,
    encoder: Mapping[str, Any],
    decoder: Mapping[str, Any],
    future_projector: Mapping[str, Any],
    gemma_adapter: Mapping[str, Any],
    artifact_only_ready: bool,
    end_to_end_injection: bool,
    live_ready: bool,
    claim_guards: Mapping[str, bool],
    claim_label: str | None = None,
) -> dict[str, Any]:
    """Build and self-hash the one canonical four-component manifest.

    All four real component identities are mandatory.  A pre-Stage-D producer
    must fail rather than inventing placeholder projector or adapter files.
    Readiness remains explicit even when the component bytes exist.
    """
    components = {
        "encoder": dict(encoder),
        "decoder": dict(decoder),
        "future_projector": dict(future_projector),
        "gemma_adapter": dict(gemma_adapter),
    }
    body = {
        "schema_version": WORLD_MODEL_COMPONENT_MANIFEST_VERSION,
        "repository": {"repo_id": CANONICAL_WORLD_MODEL_REPO_ID},
        "base_model": {
            "repo_id": base_model_repo_id,
            "revision": base_model_revision,
        },
        "components": components,
        "compatibility": {
            "transformers_revision": transformers_revision,
            "latent": {
                "dtype": latent_dtype,
                "shape": list(latent_shape) if isinstance(latent_shape, Sequence) else latent_shape,
                "order": latent_order,
            },
            "projector": {
                "input_dim": projector_input_dim,
                "output_dim": projector_output_dim,
                "gate": projector_gate,
                "injection_layer": injection_layer,
                "injection_site": injection_site,
            },
            "component_identity_sha256": {
                role: components[role].get("identity_sha256")
                for role in WORLD_MODEL_COMPONENT_ROLES
            },
        },
        "load_contract": _load_contract(),
        "status": {
            "artifact_only_ready": artifact_only_ready,
            "end_to_end_injection": end_to_end_injection,
            "live_ready": live_ready,
        },
        "claims": {
            "claim_guards": dict(claim_guards),
            **({"claim_label": claim_label} if claim_label is not None else {}),
        },
    }
    normalized = normalize_json_object(body, "world model component manifest")
    manifest = {**normalized, "manifest_sha256": _sha256_json(normalized)}
    return validate_world_model_component_manifest(manifest)


def validate_world_model_component_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the component/config/checksum and honest-readiness contract."""
    manifest = normalize_json_object(value, "world model component manifest")
    required = {
        "schema_version",
        "repository",
        "base_model",
        "components",
        "compatibility",
        "load_contract",
        "status",
        "claims",
        "manifest_sha256",
    }
    if set(manifest) != required:
        raise EvidenceError(f"world model manifest must contain exactly {sorted(required)}")
    if manifest["schema_version"] != WORLD_MODEL_COMPONENT_MANIFEST_VERSION:
        raise EvidenceError(
            f"schema_version must be {WORLD_MODEL_COMPONENT_MANIFEST_VERSION}"
        )
    repository = manifest["repository"]
    if repository != {"repo_id": CANONICAL_WORLD_MODEL_REPO_ID}:
        raise EvidenceError(
            f"world model repository must be {CANONICAL_WORLD_MODEL_REPO_ID}"
        )
    _validate_base_model(manifest["base_model"])

    components = manifest["components"]
    if not isinstance(components, dict) or set(components) != set(
        WORLD_MODEL_COMPONENT_ROLES
    ):
        raise EvidenceError(
            "world model components must contain exactly encoder, decoder, "
            "future_projector, and gemma_adapter"
        )
    directories: set[str] = set()
    paths: set[str] = set()
    for role in WORLD_MODEL_COMPONENT_ROLES:
        _validate_component(components[role], role=role)
        directory = components[role]["directory"]
        if directory in directories:
            raise EvidenceError("world model component directories must be unique")
        directories.add(directory)
        for kind in ("weights", "config"):
            path = components[role][kind]["path"]
            if path in paths:
                raise EvidenceError("world model component paths must be unique")
            paths.add(path)

    _validate_compatibility(manifest["compatibility"], components)
    if manifest["load_contract"] != _load_contract():
        raise EvidenceError("world model load_contract does not match the supported contract")
    _validate_status(manifest["status"])
    _validate_claims(manifest["claims"])
    if not is_sha256(manifest["manifest_sha256"]):
        raise EvidenceError("world model manifest_sha256 must be a SHA-256")
    unsigned = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if _sha256_json(unsigned) != manifest["manifest_sha256"]:
        raise EvidenceError("world model manifest_sha256 does not match the manifest")
    return manifest


def verify_world_model_component_files(
    value: Mapping[str, Any],
    root: str | Path,
    *,
    roles: Sequence[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Verify selected component directories and return their safe local paths."""
    manifest = validate_world_model_component_manifest(value)
    selected = list(WORLD_MODEL_COMPONENT_ROLES if roles is None else roles)
    if not selected or len(selected) != len(set(selected)):
        raise EvidenceError("component verification roles must be nonempty and unique")
    if any(role not in WORLD_MODEL_COMPONENT_ROLES for role in selected):
        raise EvidenceError("component verification requested an unsupported role")
    root_path = Path(root).resolve()
    verified: dict[str, dict[str, Path]] = {}
    for role in selected:
        component = manifest["components"][role]
        directory = _safe_local_path(root_path, component["directory"])
        if not directory.is_dir() or directory.is_symlink():
            raise EvidenceError(f"world model component directory is missing or unsafe: {role}")
        expected = {
            component["weights"]["path"],
            component["config"]["path"],
        }
        actual: set[str] = set()
        for candidate in sorted(directory.rglob("*")):
            relative = candidate.relative_to(root_path).as_posix()
            if candidate.is_symlink():
                raise EvidenceError(f"world model component file is a symlink: {relative}")
            if candidate.is_file():
                actual.add(relative)
        if actual != expected:
            raise EvidenceError(
                f"world model component directory coverage mismatch for {role}"
            )
        role_paths: dict[str, Path] = {}
        for kind in ("weights", "config"):
            record = component[kind]
            path = _safe_local_path(root_path, record["path"])
            if not path.is_file() or path.is_symlink():
                raise EvidenceError(f"world model {role} {kind} is missing or unsafe")
            if sha256_file(path) != record["sha256"]:
                raise EvidenceError(f"world model {role} {kind} checksum mismatch")
            if kind == "config":
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise EvidenceError(f"world model {role} config is not valid JSON") from exc
                if not isinstance(parsed, dict):
                    raise EvidenceError(f"world model {role} config must be a JSON object")
                normalize_json_object(parsed, f"world model {role} config")
                if "auto_map" in parsed or parsed.get("trust_remote_code") not in (None, False):
                    raise EvidenceError(
                        f"world model {role} config may not request remote code execution"
                    )
            role_paths[kind] = path
        verified[role] = role_paths
    return verified


def build_world_model_load_record(
    value: Mapping[str, Any],
    root: str | Path,
    *,
    expected_repository_revision: str,
    resolved_repository_revision: str,
    mode: str,
) -> dict[str, Any]:
    """Verify one load mode and record its externally resolved HF revision."""
    manifest = validate_world_model_component_manifest(value)
    if not _is_git_revision(expected_repository_revision):
        raise EvidenceError("expected repository revision must be a 40-character git SHA")
    if not _is_git_revision(resolved_repository_revision):
        raise EvidenceError("resolved repository revision must be a 40-character git SHA")
    if expected_repository_revision != resolved_repository_revision:
        raise EvidenceError("resolved repository revision does not match the expected revision")
    if mode not in WORLD_MODEL_LOAD_MODES:
        raise EvidenceError(f"world model load mode must be one of {WORLD_MODEL_LOAD_MODES}")
    status = manifest["status"]
    if mode == "artifact_only" and not status["artifact_only_ready"]:
        raise EvidenceError("artifact-only world model load is not ready")
    if mode == "gated_gemma" and not status["end_to_end_injection"]:
        raise EvidenceError("gated Gemma load requires verified end-to-end injection")
    mode_contract = manifest["load_contract"]["modes"][mode]
    roles = mode_contract["components"]
    verify_world_model_component_files(manifest, root, roles=roles)
    body = {
        "schema_version": WORLD_MODEL_LOAD_RECORD_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "repository": {
            "repo_id": CANONICAL_WORLD_MODEL_REPO_ID,
            "expected_revision": expected_repository_revision,
            "resolved_revision": resolved_repository_revision,
        },
        "mode": mode,
        "verified_component_identity_sha256": {
            role: manifest["components"][role]["identity_sha256"] for role in roles
        },
        "base_model": manifest["base_model"] if mode == "gated_gemma" else None,
    }
    record = {**body, "load_record_sha256": _sha256_json(body)}
    return validate_world_model_load_record(
        record,
        manifest,
        expected_repository_revision=expected_repository_revision,
        expected_mode=mode,
    )


def validate_world_model_load_record(
    value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    *,
    expected_repository_revision: str | None = None,
    expected_mode: str | None = None,
) -> dict[str, Any]:
    """Validate a stored load record against its component manifest."""
    manifest = validate_world_model_component_manifest(manifest_value)
    record = normalize_json_object(value, "world model load record")
    fields = {
        "schema_version",
        "manifest_sha256",
        "repository",
        "mode",
        "verified_component_identity_sha256",
        "base_model",
        "load_record_sha256",
    }
    if set(record) != fields:
        raise EvidenceError(f"world model load record must contain exactly {sorted(fields)}")
    if record["schema_version"] != WORLD_MODEL_LOAD_RECORD_VERSION:
        raise EvidenceError(f"schema_version must be {WORLD_MODEL_LOAD_RECORD_VERSION}")
    if record["manifest_sha256"] != manifest["manifest_sha256"]:
        raise EvidenceError("world model load record does not match its manifest")
    repository = record["repository"]
    repository_fields = {"repo_id", "expected_revision", "resolved_revision"}
    if not isinstance(repository, dict) or set(repository) != repository_fields:
        raise EvidenceError(
            f"load record repository must contain exactly {sorted(repository_fields)}"
        )
    if repository["repo_id"] != CANONICAL_WORLD_MODEL_REPO_ID:
        raise EvidenceError("load record repository does not match the canonical repository")
    for key in ("expected_revision", "resolved_revision"):
        if not _is_git_revision(repository[key]):
            raise EvidenceError(f"load record {key} must be a 40-character git SHA")
    if repository["expected_revision"] != repository["resolved_revision"]:
        raise EvidenceError("load record resolved revision does not match expected revision")
    if (
        expected_repository_revision is not None
        and repository["expected_revision"] != expected_repository_revision
    ):
        raise EvidenceError("load record does not match the caller's expected revision")
    mode = record["mode"]
    if mode not in WORLD_MODEL_LOAD_MODES:
        raise EvidenceError(f"world model load mode must be one of {WORLD_MODEL_LOAD_MODES}")
    if expected_mode is not None and mode != expected_mode:
        raise EvidenceError("load record does not match the expected mode")
    status = manifest["status"]
    if mode == "artifact_only" and not status["artifact_only_ready"]:
        raise EvidenceError("artifact-only world model load is not ready")
    if mode == "gated_gemma" and not status["end_to_end_injection"]:
        raise EvidenceError("gated Gemma load requires verified end-to-end injection")
    roles = manifest["load_contract"]["modes"][mode]["components"]
    expected_hashes = {
        role: manifest["components"][role]["identity_sha256"] for role in roles
    }
    if record["verified_component_identity_sha256"] != expected_hashes:
        raise EvidenceError("load record verified component identities do not match")
    expected_base = manifest["base_model"] if mode == "gated_gemma" else None
    if record["base_model"] != expected_base:
        raise EvidenceError("load record base model does not match its load mode")
    if not is_sha256(record["load_record_sha256"]):
        raise EvidenceError("load_record_sha256 must be a SHA-256")
    unsigned = {key: item for key, item in record.items() if key != "load_record_sha256"}
    if _sha256_json(unsigned) != record["load_record_sha256"]:
        raise EvidenceError("load_record_sha256 does not match the load record")
    return record


def _validate_component(value: Any, *, role: str | None) -> None:
    if not isinstance(value, dict) or set(value) != _COMPONENT_FIELDS:
        raise EvidenceError(
            f"world model component must contain exactly {sorted(_COMPONENT_FIELDS)}"
        )
    if not isinstance(value["architecture"], str) or not value["architecture"]:
        raise EvidenceError("world model component architecture must be nonempty")
    directory = _relative_path(value["directory"], "component directory")
    for kind, expected_format in (("weights", "safetensors"), ("config", "json")):
        record = value[kind]
        if not isinstance(record, dict) or set(record) != _FILE_FIELDS:
            raise EvidenceError(f"world model component {kind} identity is invalid")
        path = _relative_path(record["path"], f"component {kind} path")
        if str(PurePosixPath(path).parent) != directory:
            raise EvidenceError("component weights and config must be in their exact directory")
        if not is_sha256(record["sha256"]):
            raise EvidenceError(f"world model component {kind} sha256 must be a SHA-256")
        if record["format"] != expected_format:
            raise EvidenceError(
                f"world model component {kind} format must be {expected_format}"
            )
    if role == "gemma_adapter":
        if PurePosixPath(value["weights"]["path"]).name != "adapter_model.safetensors":
            raise EvidenceError("Gemma adapter weights must be adapter_model.safetensors")
        if PurePosixPath(value["config"]["path"]).name != "adapter_config.json":
            raise EvidenceError("Gemma adapter config must be adapter_config.json")
    core = {key: item for key, item in value.items() if key != "identity_sha256"}
    if not is_sha256(value["identity_sha256"]) or _sha256_json(core) != value[
        "identity_sha256"
    ]:
        raise EvidenceError("world model component identity_sha256 does not match")


def _validate_base_model(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"repo_id", "revision"}:
        raise EvidenceError("base_model must contain exactly repo_id and revision")
    if not isinstance(value["repo_id"], str) or not value["repo_id"]:
        raise EvidenceError("base model repo_id must be nonempty")
    if not _is_git_revision(value["revision"]):
        raise EvidenceError("base model revision must be a 40-character git SHA")


def _validate_compatibility(value: Any, components: Mapping[str, Any]) -> None:
    fields = {
        "transformers_revision",
        "latent",
        "projector",
        "component_identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"world model compatibility must contain exactly {sorted(fields)}")
    if value["transformers_revision"] != SUPPORTED_TRANSFORMERS_REVISION:
        raise EvidenceError(
            f"Transformers revision must be {SUPPORTED_TRANSFORMERS_REVISION}"
        )
    latent = value["latent"]
    if not isinstance(latent, dict) or set(latent) != {"dtype", "shape", "order"}:
        raise EvidenceError("latent compatibility requires dtype, shape, and order")
    if latent["dtype"] not in _LATENT_DTYPES:
        raise EvidenceError(f"latent dtype must be one of {sorted(_LATENT_DTYPES)}")
    if (
        not isinstance(latent["shape"], list)
        or not latent["shape"]
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in latent["shape"]
        )
    ):
        raise EvidenceError("latent shape must contain positive integers")
    if latent["order"] not in {"C", "F"}:
        raise EvidenceError("latent order must be C or F")
    projector = value["projector"]
    projector_fields = {
        "input_dim",
        "output_dim",
        "gate",
        "injection_layer",
        "injection_site",
    }
    if not isinstance(projector, dict) or set(projector) != projector_fields:
        raise EvidenceError(
            f"projector compatibility must contain exactly {sorted(projector_fields)}"
        )
    for key in ("input_dim", "output_dim"):
        if (
            isinstance(projector[key], bool)
            or not isinstance(projector[key], int)
            or projector[key] <= 0
        ):
            raise EvidenceError(f"projector {key} must be a positive integer")
    if projector["input_dim"] != latent["shape"][-1]:
        raise EvidenceError("projector input_dim must equal the final latent dimension")
    if (
        isinstance(projector["injection_layer"], bool)
        or not isinstance(projector["injection_layer"], int)
        or projector["injection_layer"] < 0
    ):
        raise EvidenceError("projector injection_layer must be a nonnegative integer")
    for key in ("gate", "injection_site"):
        if not isinstance(projector[key], str) or not projector[key]:
            raise EvidenceError(f"projector {key} must be nonempty")
    hashes = value["component_identity_sha256"]
    expected = {
        role: components[role]["identity_sha256"] for role in WORLD_MODEL_COMPONENT_ROLES
    }
    if hashes != expected:
        raise EvidenceError("compatibility component identity hashes do not match components")


def _validate_status(value: Any) -> None:
    fields = {"artifact_only_ready", "end_to_end_injection", "live_ready"}
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"world model status must contain exactly {sorted(fields)}")
    for key in ("artifact_only_ready", "end_to_end_injection", "live_ready"):
        if not isinstance(value[key], bool):
            raise EvidenceError(f"world model status {key} must be boolean")
    if value["end_to_end_injection"] and not value["artifact_only_ready"]:
        raise EvidenceError("end-to-end readiness requires artifact-only readiness")
    if value["live_ready"] and not value["end_to_end_injection"]:
        raise EvidenceError("live readiness requires verified end-to-end injection")


def _validate_claims(value: Any) -> None:
    if not isinstance(value, dict) or set(value) not in (
        {"claim_guards"},
        {"claim_guards", "claim_label"},
    ):
        raise EvidenceError("world model claims require claim_guards and optional claim_label")
    guards = value["claim_guards"]
    if not isinstance(guards, dict) or set(guards) != _CLAIM_GUARD_FIELDS:
        raise EvidenceError(f"claim_guards must contain exactly {sorted(_CLAIM_GUARD_FIELDS)}")
    if any(not isinstance(guards[key], bool) for key in _CLAIM_GUARD_FIELDS):
        raise EvidenceError("claim_guards values must be boolean")
    allowed = [key for key in _CLAIM_GUARD_FIELDS if key.endswith("_claim_allowed")]
    if guards["engineering_only"] and any(guards[key] for key in allowed):
        raise EvidenceError("engineering_only requires every scientific claim guard to be false")
    if "claim_label" in value and (
        not isinstance(value["claim_label"], str) or not value["claim_label"].strip()
    ):
        raise EvidenceError("claim_label must be a nonempty string when provided")


def _load_contract() -> dict[str, Any]:
    roles = list(WORLD_MODEL_COMPONENT_ROLES)
    return {
        "schema_version": WORLD_MODEL_LOAD_CONTRACT_VERSION,
        "component_order": roles,
        "verify_sha256_before_load": True,
        "reject_extra_component_files": True,
        "allow_remote_code": False,
        "strict_state_dict": True,
        "modes": {
            "artifact_only": {
                "components": ["encoder", "decoder"],
                "requires_authentication": False,
                "requires_base_model": False,
            },
            "gated_gemma": {
                "components": roles,
                "requires_authentication": True,
                "requires_base_model": True,
            },
        },
    }


def _relative_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceError(f"{where} must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise EvidenceError(f"{where} must be a canonical POSIX relative path")
    return value


def _safe_local_path(root: Path, relative: str) -> Path:
    relative = _relative_path(relative, "component path")
    target = root / relative
    try:
        target.resolve().relative_to(root)
    except ValueError as exc:
        raise EvidenceError("world model component path escapes its root") from exc
    cursor = target
    while cursor != root and cursor != cursor.parent:
        if cursor.is_symlink():
            raise EvidenceError(f"world model component path is a symlink: {relative}")
        cursor = cursor.parent
    return target


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _is_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
