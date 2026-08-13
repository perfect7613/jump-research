import json

import pytest

from jump_contracts import load_task_evidence
from jump_benchmark.authentic import HOLDOUT_LAW_FAMILY, law_family
from jump_benchmark.authentic_stage_c import (
    STAGE_C_MANIFEST_SHA256,
    _load_safe_checkpoint,
    run_stage_c,
    stage_c_dataset,
    stage_c_manifest,
)
from jump_benchmark.experiment_spec import compile_experiment_intent


def _plan():
    return compile_experiment_intent(
        {
            "schema_version": "jump.experiment-intent/v1",
            "intent": "Discover the hidden interaction law from observed motion.",
            "session_id": "stage-c-test",
            "seed": 99173,
            "max_steps": 4,
        }
    )


def test_stage_c_manifest_probe_and_persistence_contract_is_frozen():
    manifest = stage_c_manifest()
    assert STAGE_C_MANIFEST_SHA256 == "146e40f0115960d85587a21eefaf6111cefe648cef95960d36b4f82affc6af42"
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
    result = run_stage_c(
        output_root=tmp_path / "output",
        checkpoint_root=tmp_path / "checkpoints",
        expected_manifest_sha256=STAGE_C_MANIFEST_SHA256,
        expected_code_sha="0" * 40,
        experiment_spec=_plan(),
        device="cpu",
        dry_run=True,
    )
    assert result["status"] == "completed"
    assert result["decision"] == "dry_run_not_evaluated"
    assert result["artifact_verification"] == "passed"
    assert result["mechanistic_evidence"] is False
    assert result["seed_ids"] == ["seed-99173", "seed-99174", "seed-99175"]
    assert result["task_evidence"]["schema_version"] == "jump.task-evidence/v1"
    loaded = load_task_evidence(tmp_path / "output" / "result.json")
    assert loaded["experiment_spec_sha256"] == result["experiment_spec_sha256"]
    roles = {artifact["role"] for artifact in loaded["artifacts"]}
    assert {"world-encoder", "learned-decoder", "world-latent", "terminal-result"} <= roles
    import torch

    for seed in result["seed_ids"]:
        checkpoint = torch.load(
            tmp_path / "checkpoints" / seed / "latest.pt", weights_only=True, map_location="cpu"
        )
        terminal = json.loads((tmp_path / "output" / seed / "seed-terminal.json").read_text())
        probe_freeze = json.loads((tmp_path / "output" / seed / "probe-freeze.json").read_text())
        final_ood = json.loads((tmp_path / "output" / seed / "final-ood-probes.json").read_text())
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
