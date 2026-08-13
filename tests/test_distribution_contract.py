from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from jump_contracts import (
    CANONICAL_WORLD_MODEL_REPO_ID,
    SUPPORTED_TRANSFORMERS_REVISION,
    EvidenceError,
    build_world_model_component_manifest,
    build_world_model_load_record,
    component_identity,
    validate_world_model_component_manifest,
    validate_world_model_load_record,
    verify_world_model_component_files,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _component(root: Path, role: str) -> dict:
    directory = root / "components" / role
    directory.mkdir(parents=True)
    adapter = role == "gemma_adapter"
    weights_name = "adapter_model.safetensors" if adapter else "model.safetensors"
    config_name = "adapter_config.json" if adapter else "config.json"
    weights = f"{role}-weights".encode()
    config = json.dumps({"architecture": role}, sort_keys=True).encode()
    (directory / weights_name).write_bytes(weights)
    (directory / config_name).write_bytes(config)
    relative = directory.relative_to(root).as_posix()
    return component_identity(
        directory=relative,
        weights_path=f"{relative}/{weights_name}",
        weights_sha256=_sha(weights),
        config_path=f"{relative}/{config_name}",
        config_sha256=_sha(config),
        architecture=f"jump-{role}-v1",
    )


def _claim_guards(**changes: bool) -> dict[str, bool]:
    guards = {
        "engineering_only": True,
        "behavioral_claim_allowed": False,
        "mechanistic_claim_allowed": False,
        "causal_claim_allowed": False,
        "benchmark_law_accuracy_claim_allowed": False,
        "track_r_claim_allowed": False,
    }
    guards.update(changes)
    return guards


def _manifest(
    root: Path,
    *,
    artifact_only_ready: bool = True,
    end_to_end_injection: bool = False,
    live_ready: bool = False,
    claim_guards: dict[str, bool] | None = None,
) -> dict:
    components = {role: _component(root, role) for role in (
        "encoder",
        "decoder",
        "future_projector",
        "gemma_adapter",
    )}
    return build_world_model_component_manifest(
        base_model_repo_id="google/gemma-test",
        base_model_revision="b" * 40,
        transformers_revision=SUPPORTED_TRANSFORMERS_REVISION,
        latent_dtype="float32-le",
        latent_shape=[1, 4],
        latent_order="C",
        projector_input_dim=4,
        projector_output_dim=256,
        projector_gate="learned_scalar_sigmoid",
        injection_layer=12,
        injection_site="residual_stream_post_attention",
        encoder=components["encoder"],
        decoder=components["decoder"],
        future_projector=components["future_projector"],
        gemma_adapter=components["gemma_adapter"],
        artifact_only_ready=artifact_only_ready,
        end_to_end_injection=end_to_end_injection,
        live_ready=live_ready,
        claim_guards=claim_guards or _claim_guards(),
        claim_label="engineering artifact; no scientific claim",
    )


def test_manifest_binds_one_repo_components_configs_compatibility_and_load_modes(tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest["repository"] == {"repo_id": CANONICAL_WORLD_MODEL_REPO_ID}
    assert "revision" not in manifest["repository"]
    assert set(manifest["components"]) == {
        "encoder",
        "decoder",
        "future_projector",
        "gemma_adapter",
    }
    assert manifest["compatibility"]["transformers_revision"] == (
        "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6"
    )
    assert manifest["compatibility"]["latent"] == {
        "dtype": "float32-le",
        "shape": [1, 4],
        "order": "C",
    }
    assert manifest["load_contract"]["modes"]["artifact_only"] == {
        "components": ["encoder", "decoder"],
        "requires_authentication": False,
        "requires_base_model": False,
    }
    assert manifest["load_contract"]["modes"]["gated_gemma"][
        "requires_authentication"
    ] is True
    assert validate_world_model_component_manifest(manifest) == manifest
    assert set(verify_world_model_component_files(manifest, tmp_path)) == set(
        manifest["components"]
    )


def test_artifact_only_load_records_external_revision_without_loading_gemma(tmp_path):
    manifest = _manifest(tmp_path)
    revision = "c" * 40

    record = build_world_model_load_record(
        manifest,
        tmp_path,
        expected_repository_revision=revision,
        resolved_repository_revision=revision,
        mode="artifact_only",
    )

    assert record["repository"] == {
        "repo_id": CANONICAL_WORLD_MODEL_REPO_ID,
        "expected_revision": revision,
        "resolved_revision": revision,
    }
    assert set(record["verified_component_identity_sha256"]) == {"encoder", "decoder"}
    assert record["base_model"] is None
    assert validate_world_model_load_record(
        record,
        manifest,
        expected_repository_revision=revision,
        expected_mode="artifact_only",
    ) == record

    drifted = copy.deepcopy(record)
    drifted["verified_component_identity_sha256"]["encoder"] = "0" * 64
    with pytest.raises(EvidenceError, match="verified component identities"):
        validate_world_model_load_record(drifted, manifest)

    with pytest.raises(EvidenceError, match="does not match the expected revision"):
        build_world_model_load_record(
            manifest,
            tmp_path,
            expected_repository_revision=revision,
            resolved_repository_revision="d" * 40,
            mode="artifact_only",
        )


def test_gated_mode_requires_e2e_readiness_but_claims_remain_orthogonal(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(EvidenceError, match="verified end-to-end injection"):
        build_world_model_load_record(
            manifest,
            tmp_path,
            expected_repository_revision="c" * 40,
            resolved_repository_revision="c" * 40,
            mode="gated_gemma",
        )

    ready_root = tmp_path / "ready"
    ready = _manifest(
        ready_root,
        end_to_end_injection=True,
        live_ready=True,
        claim_guards=_claim_guards(),
    )
    assert ready["status"]["live_ready"] is True
    assert all(
        value is False
        for key, value in ready["claims"]["claim_guards"].items()
        if key.endswith("_claim_allowed")
    )
    record = build_world_model_load_record(
        ready,
        ready_root,
        expected_repository_revision="e" * 40,
        resolved_repository_revision="e" * 40,
        mode="gated_gemma",
    )
    assert set(record["verified_component_identity_sha256"]) == set(ready["components"])
    assert record["base_model"] == ready["base_model"]


def test_manifest_rejects_overclaims_wrong_runtime_and_placeholder_component(tmp_path):
    with pytest.raises(EvidenceError, match="engineering_only"):
        _manifest(tmp_path / "claims", claim_guards=_claim_guards(behavioral_claim_allowed=True))

    manifest = _manifest(tmp_path / "runtime")
    tampered = copy.deepcopy(manifest)
    tampered["compatibility"]["transformers_revision"] = "f" * 40
    with pytest.raises(EvidenceError, match="Transformers revision"):
        validate_world_model_component_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    del tampered["components"]["future_projector"]
    with pytest.raises(EvidenceError, match="components must contain exactly"):
        validate_world_model_component_manifest(tampered)


def test_component_verification_rejects_checksum_drift_and_extra_directory_files(tmp_path):
    manifest = _manifest(tmp_path)
    encoder_weights = tmp_path / manifest["components"]["encoder"]["weights"]["path"]
    encoder_weights.write_bytes(b"changed")
    with pytest.raises(EvidenceError, match="encoder weights checksum mismatch"):
        verify_world_model_component_files(manifest, tmp_path, roles=["encoder"])

    encoder_weights.write_bytes(b"encoder-weights")
    encoder_dir = tmp_path / manifest["components"]["encoder"]["directory"]
    (encoder_dir / "undeclared.bin").write_bytes(b"extra")
    with pytest.raises(EvidenceError, match="directory coverage mismatch"):
        verify_world_model_component_files(manifest, tmp_path, roles=["encoder"])


def test_component_config_cannot_request_arbitrary_remote_code(tmp_path):
    manifest = _manifest(tmp_path)
    config_record = manifest["components"]["encoder"]["config"]
    config_path = tmp_path / config_record["path"]
    unsafe = json.dumps({"auto_map": {"AutoModel": "model.Custom"}}).encode()
    config_path.write_bytes(unsafe)
    config_record["sha256"] = _sha(unsafe)
    component = manifest["components"]["encoder"]
    core = {key: value for key, value in component.items() if key != "identity_sha256"}
    component["identity_sha256"] = _sha(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    )
    manifest["compatibility"]["component_identity_sha256"]["encoder"] = component[
        "identity_sha256"
    ]
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = _sha(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    )

    with pytest.raises(EvidenceError, match="remote code execution"):
        verify_world_model_component_files(manifest, tmp_path, roles=["encoder"])


def test_gemma_adapter_requires_peft_filenames_and_colocated_config(tmp_path):
    manifest = _manifest(tmp_path)
    bad = copy.deepcopy(manifest)
    adapter = bad["components"]["gemma_adapter"]
    adapter["weights"]["path"] = f"{adapter['directory']}/adapter.safetensors"
    with pytest.raises(EvidenceError, match="adapter_model.safetensors"):
        validate_world_model_component_manifest(bad)

    with pytest.raises(EvidenceError, match="exact directory"):
        component_identity(
            directory="components/encoder",
            weights_path="components/encoder/model.safetensors",
            weights_sha256="a" * 64,
            config_path="configs/encoder.json",
            config_sha256="b" * 64,
            architecture="encoder-v1",
        )
