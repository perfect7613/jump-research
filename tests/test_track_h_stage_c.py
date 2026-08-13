import json

import pytest

from jump_contracts import EvidenceError, load_verified_run_evidence
from jump_benchmark.authentic import HOLDOUT_LAW_FAMILY, law_family
from jump_benchmark.authentic_stage_c import (
    STAGE_C_MANIFEST_SHA256,
    STAGE_C_LAUNCH_SPEC_SHA256,
    _load_safe_checkpoint,
    run_stage_c,
    stage_c_dataset,
    stage_c_manifest,
    stage_c_launch_spec,
    stage_c_run_contract,
)
from jump_benchmark.experiment_spec import compile_experiment_intent
from jump_runner.executor import execute_local_run


def _plan():
    return stage_c_launch_spec()


def test_stage_c_manifest_probe_and_persistence_contract_is_frozen():
    manifest = stage_c_manifest()
    assert STAGE_C_MANIFEST_SHA256 == "a90890d8ff4f154a6e1bdaca4af4dda00371754d2e5cc9ad203fadad2a24c84a"
    assert [item["seed_id"] for item in manifest["initialization"]["seeds"]] == [
        "seed-99173", "seed-99174", "seed-99175"
    ]
    assert manifest["dataset"]["counts"] == {
        "train": 1024,
        "id_validation": 256,
        "id_test": 256,
        "heldout_law_ood": 256,
    }
    probes = manifest["posthoc_probes"]
    assert probes["encoder_frozen_before_fit"] is True
    assert (probes["outer_folds"], probes["inner_folds"], probes["fold_group"]) == (3, 3, "world_seed")
    assert probes["clustered_ci"] == {"unit": "world_seed", "bootstrap_replicates": 10000, "level": 0.95, "seed": 44123}
    assert probes["relation_metrics"] == ["roc_auc", "balanced_accuracy", "roc_auc_clustered_ci95"]
    assert probes["relation_pairs_cofolded_and_clustered_by_world"] is True
    assert probes["id_and_ood_hypotheses_never_mixed"] is True
    assert manifest["decoder_evaluation"]["paired_improvement_min"] == 0.20
    assert manifest["partition_only_swap"]["paired_inference"]["bootstrap_replicates"] == 10000
    assert all(manifest["persistence_and_probe_gates"].values())
    assert manifest["decision"]["mechanistic_evidence"] is False
    assert manifest["launch_spec"]["sha256"] == STAGE_C_LAUNCH_SPEC_SHA256


def test_stage_c_small_dataset_is_seed_disjoint_with_pure_family_ood():
    data = stage_c_dataset(
        root_seed=99173,
        counts={"train": 8, "id_validation": 4, "id_test": 4, "heldout_law_ood": 4},
    )
    seeds = [row["world_seed"] for row in data["records"]]
    assert len(seeds) == len(set(seeds))
    for row in data["records"]:
        heldout = law_family(row["sealed_target"]["replacement_law"]) == HOLDOUT_LAW_FAMILY
        assert heldout is (row["split"] == "heldout_law_ood")


def test_stage_c_cpu_dry_run_verifies_artifacts_without_scientific_decision(tmp_path):
    import os

    os.environ["JUMP_CODE_VERSION"] = "0" * 40
    arbitrary = compile_experiment_intent(
        {
            "schema_version": "jump.experiment-intent/v1",
            "intent": "Discover the hidden interaction law from observed motion.",
            "session_id": "wrong-stage-c-plan",
            "seed": 99173,
            "max_steps": 4,
        }
    )
    with pytest.raises(ValueError, match="exact frozen canonical"):
        run_stage_c(
            output_root=tmp_path / "rejected",
            checkpoint_root=tmp_path / "rejected-checkpoints",
            expected_manifest_sha256=STAGE_C_MANIFEST_SHA256,
            expected_code_sha="0" * 40,
            experiment_spec=arbitrary,
            device="cpu",
            dry_run=True,
        )
    assert not (tmp_path / "rejected").exists()
    phase, run = stage_c_run_contract(
        expected_manifest_sha256=STAGE_C_MANIFEST_SHA256,
        expected_code_sha="0" * 40,
        dry_run=True,
    )
    run_root = tmp_path / "canonical-run"
    result = execute_local_run(phase, run, run_root, STAGE_C_MANIFEST_SHA256)
    assert result["schema_version"] == "jump.run-result/v1"
    assert result["status"] == "completed"
    assert result["track_h"]["dry_run"] is True
    first_result_bytes = (run_root / "result.json").read_bytes()
    repeated = execute_local_run(phase, run, run_root, STAGE_C_MANIFEST_SHA256)
    assert repeated == result
    assert (run_root / "result.json").read_bytes() == first_result_bytes
    assert [path.name for path in (run_root / "attempts").iterdir()] == ["0001"]
    loaded = load_verified_run_evidence(
        run_root / "result.json",
        artifact_root=run_root,
        expected_manifest_sha256=STAGE_C_MANIFEST_SHA256,
    )
    roles = {artifact["role"] for artifact in loaded["artifacts"]}
    assert {"world-encoder", "learned-decoder", "world-latent", "terminal-result"} <= roles
    artifacts = {artifact["name"]: run_root / artifact["path"] for artifact in loaded["artifacts"]}
    aggregate = json.loads(artifacts["aggregate-terminal.json"].read_text())
    assert aggregate["decision"] == "dry_run_not_evaluated"
    assert aggregate["seed_ids"] == ["seed-99173", "seed-99174", "seed-99175"]
    import torch

    for seed in aggregate["seed_ids"]:
        checkpoint = torch.load(
            run_root / "attempts" / "0001" / "checkpoint" / seed / "latest.pt",
            weights_only=True,
            map_location="cpu",
        )
        terminal = json.loads(artifacts[f"{seed}/seed-terminal.json"].read_text())
        probe_freeze = json.loads(artifacts[f"{seed}/probe-freeze.json"].read_text())
        final_ood = json.loads(artifacts[f"{seed}/final-ood-probes.json"].read_text())
        assert checkpoint["code_sha"] == terminal["code_sha"] == "0" * 40
        assert terminal["evaluation_protocol"].startswith("phase_1")
        assert terminal["metrics"]["id_test"]["nrmse_normalizer"] == terminal["metrics"]["heldout_law_ood"]["nrmse_normalizer"]
        assert probe_freeze["bh_fdr_scope"] == "id_validation_only"
        assert probe_freeze["relation_pairs_cofolded_and_clustered"] is True
        assert final_ood["phase"].startswith("phase_2")
        assert final_ood["probe_freeze_sha256"] == probe_freeze["probe_freeze_sha256"]
        assert final_ood["bh_fdr_scope"] == "final_ood_only_never_mixed_with_id"
        relation = final_ood["evaluations"]["relations"]
        assert {"roc_auc", "balanced_accuracy", "roc_auc_clustered_ci95"} <= set(relation)
    tampered = artifacts["seed-99173/encoder.safetensors"]
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    with pytest.raises(EvidenceError, match="hash"):
        load_verified_run_evidence(run_root / "result.json", artifact_root=run_root)


def _write_marker(path):
    path.write_text("executed")


def test_safe_checkpoint_loader_rejects_pickle_globals_before_execution(tmp_path):
    class Hostile:
        def __reduce__(self):
            return (_write_marker, (tmp_path / "marker",))

    import torch

    checkpoint = tmp_path / "hostile.pt"
    torch.save({"payload": Hostile()}, checkpoint)
    with pytest.raises(Exception):
        _load_safe_checkpoint(checkpoint, device="cpu")
    assert not (tmp_path / "marker").exists()
