from __future__ import annotations

import json
from pathlib import Path

import pytest

import jump_contracts.evidence as evidence_module

from jump_contracts.evidence import (
    EvidenceError,
    artifact_declaration,
    load_cached_result_envelope,
    load_task_evidence,
    load_verified_run_evidence,
    open_result_envelope,
    promote_task_artifacts,
    read_verified_artifact,
    seal_result_envelope,
    write_task_evidence,
)


def _metric() -> dict[str, object]:
    return {"name": "behavior.joint_theory_accuracy", "value": 1.0, "split": "pilot"}


def test_versioned_task_evidence_requires_exact_artifact_coverage(tmp_path):
    output = tmp_path / "work"
    output.mkdir()
    declared = output / "answer.json"
    declared.write_text('{"answer":"hidden type"}\n')
    (output / "unlabelled-latent.bin").write_bytes(b"latent")

    with pytest.raises(EvidenceError, match="undeclared artifact files: unlabelled-latent.bin"):
        write_task_evidence(
            output,
            metrics=[_metric()],
            artifacts=[
                artifact_declaration(
                    declared,
                    output,
                    name="answer",
                    media_type="application/json",
                    role="answer",
                )
            ],
        )

    assert not (output / "result.json").exists()


def test_task_evidence_rechecks_hash_and_preregistered_dimensions(tmp_path):
    output = tmp_path / "work"
    output.mkdir()
    artifact = output / "latent.bin"
    artifact.write_bytes(b"sealed-latent")
    result = write_task_evidence(
        output,
        metrics=[{"name": "capture.count", "value": 1, "layer": 8, "timepoint": "T4"}],
        artifacts=[artifact_declaration(artifact, output, role="latent")],
    )
    assert result["schema_version"] == "jump.task-evidence/v1"
    assert load_task_evidence(
        output / "result.json", allowed_layers=[8], allowed_timepoints=["T4"]
    ) == result

    artifact.write_bytes(b"changed")
    with pytest.raises(EvidenceError, match="artifact hash mismatch"):
        load_task_evidence(
            output / "result.json", allowed_layers=[8], allowed_timepoints=["T4"]
        )

    artifact.write_bytes(b"sealed-latent")
    with pytest.raises(EvidenceError, match="non-preregistered layer"):
        load_task_evidence(
            output / "result.json", allowed_layers=[9], allowed_timepoints=["T4"]
        )


@pytest.mark.parametrize("field", [{"path": Path("not-json")}, {"value": float("nan")}])
def test_task_evidence_wraps_noncanonical_domain_fields(tmp_path, field):
    output = tmp_path / "work"
    output.mkdir()
    with pytest.raises(EvidenceError, match="task evidence must be finite canonical JSON"):
        write_task_evidence(output, metrics=[_metric()], domain=field)


def test_promotion_rejects_file_that_appears_after_coverage_validation(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    artifact = work / "answer.json"
    artifact.write_text("{}")
    task = write_task_evidence(
        work,
        metrics=[_metric()],
        artifacts=[artifact_declaration(artifact, work)],
    )
    real_artifact_files = evidence_module._artifact_files
    calls = 0

    def changing_artifact_files(root):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "appeared-late.bin").write_bytes(b"late")
        return real_artifact_files(root)

    monkeypatch.setattr(evidence_module, "_artifact_files", changing_artifact_files)
    target = tmp_path / "promoted"
    with pytest.raises(EvidenceError, match="undeclared artifact appeared during promotion"):
        promote_task_artifacts(work, target, "attempts/0001/artifacts", task)
    assert not target.exists()


def test_promotion_preserves_metadata_and_verified_reader_detects_tampering(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    cache = work / "cache" / "flagship.json"
    cache.parent.mkdir()
    cache.write_text('{"provenance_label":"cached"}\n')
    task = write_task_evidence(
        work,
        metrics=[_metric()],
        artifacts=[
            artifact_declaration(
                cache,
                work,
                name="flagship-conversation",
                media_type="application/json",
                role="cache-entry",
            )
        ],
        condition_id="C-prime",
    )

    run_root = tmp_path / "run"
    records = promote_task_artifacts(
        work,
        run_root / "attempts/0001/artifacts",
        "attempts/0001/artifacts",
        task,
    )
    assert records[0]["role"] == "cache-entry"
    assert records[0]["media_type"] == "application/json"
    run_result = {
        "schema_version": "jump.run-result/v1",
        "status": "completed",
        "metrics": task["metrics"],
        "artifacts": records,
        "provenance": {
            "manifest_sha256": "a" * 64,
            "run_id": "baseline-c-prime",
            "code_version": "test-revision",
        },
    }
    result_path = run_root / "result.json"
    result_path.write_text(json.dumps(run_result))

    assert load_verified_run_evidence(
        result_path, expected_manifest_sha256="a" * 64
    ) == run_result
    record, content = read_verified_artifact(
        result_path,
        "flagship-conversation",
        expected_manifest_sha256="a" * 64,
    )
    assert record["role"] == "cache-entry"
    assert content == b'{"provenance_label":"cached"}\n'
    envelope = load_cached_result_envelope(
        result_path,
        "flagship-conversation",
        checkpoint_id="gemma-pinned-revision",
        expected_manifest_sha256="a" * 64,
    )
    assert open_result_envelope(
        envelope,
        expected_source="cached",
        expected_manifest_sha256="a" * 64,
        expected_checkpoint_id="gemma-pinned-revision",
    ) == {"provenance_label": "cached"}
    promoted = run_root / records[0]["path"]
    promoted.write_text("tampered")
    with pytest.raises(EvidenceError, match="artifact hash mismatch"):
        load_verified_run_evidence(result_path)


def test_legacy_metrics_only_task_output_remains_supported(tmp_path):
    work = tmp_path / "legacy-work"
    work.mkdir()
    (work / "legacy.txt").write_text("legacy artifact\n")
    result_path = work / "result.json"
    result_path.write_text(json.dumps({"metrics": [_metric()]}))

    legacy = load_task_evidence(result_path)
    records = promote_task_artifacts(
        work, tmp_path / "promoted", "attempts/0001/artifacts", legacy
    )
    assert records == [
        {
            "name": "legacy.txt",
            "path": "attempts/0001/artifacts/legacy.txt",
            "sha256": records[0]["sha256"],
            "media_type": "application/octet-stream",
        }
    ]


@pytest.mark.parametrize("unsafe", ["../answer.json", "/tmp/answer.json", "cache\\answer.json"])
def test_artifact_paths_are_canonical_relative_paths(tmp_path, unsafe):
    work = tmp_path / "work"
    work.mkdir()
    (work / "answer.json").write_text("{}")
    result_path = work / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "jump.task-evidence/v1",
                "metrics": [_metric()],
                "artifacts": [
                    {
                        "name": "answer",
                        "path": unsafe,
                        "sha256": "0" * 64,
                        "media_type": "application/json",
                    }
                ],
            }
        )
    )
    with pytest.raises(EvidenceError, match="artifact path"):
        load_task_evidence(result_path)


@pytest.mark.parametrize("source", ["cached", "live"])
def test_cached_and_live_results_share_one_sealed_envelope(source):
    payload = {
        "schema_version": "jump.ui-cache/v1",
        "answer": {"partition": [0, 0, 0, 1, 1, 1]},
        "latent_sha256": "b" * 64,
    }
    envelope = seal_result_envelope(
        payload,
        source=source,
        manifest_sha256="a" * 64,
        run_id=f"{source}-request-001",
        code_version="release-sha",
        checkpoint_id="gemma-pinned-revision",
    )

    assert envelope["schema_version"] == "jump.sealed-result/v1"
    assert open_result_envelope(
        envelope,
        expected_source=source,
        expected_manifest_sha256="a" * 64,
        expected_checkpoint_id="gemma-pinned-revision",
    ) == payload


def test_sealed_result_envelope_rejects_payload_or_provenance_drift():
    envelope = seal_result_envelope(
        {"answer": "exact structured result"},
        source="live",
        manifest_sha256="a" * 64,
        run_id="live-001",
        code_version="release-sha",
        checkpoint_id="checkpoint-revision",
    )
    envelope["payload"]["answer"] = "tampered"
    with pytest.raises(EvidenceError, match="payload hash mismatch"):
        open_result_envelope(envelope)

    envelope = seal_result_envelope(
        {"answer": "exact structured result"},
        source="live",
        manifest_sha256="a" * 64,
        run_id="live-001",
        code_version="release-sha",
        checkpoint_id="checkpoint-revision",
    )
    with pytest.raises(EvidenceError, match="expected manifest"):
        open_result_envelope(envelope, expected_manifest_sha256="c" * 64)
    with pytest.raises(EvidenceError, match="expected source"):
        open_result_envelope(envelope, expected_source="cached")


def test_sealed_result_envelope_rejects_nonfinite_or_unidentified_live_results():
    with pytest.raises(EvidenceError, match="finite canonical JSON"):
        seal_result_envelope(
            {"confidence": float("nan")},
            source="live",
            manifest_sha256="a" * 64,
            run_id="live-001",
            code_version="release-sha",
            checkpoint_id="checkpoint-revision",
        )
    with pytest.raises(EvidenceError, match="checkpoint_id must be a nonempty string"):
        seal_result_envelope(
            {"answer": "result"},
            source="live",
            manifest_sha256="a" * 64,
            run_id="live-001",
            code_version="release-sha",
            checkpoint_id="",
        )
