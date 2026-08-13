from jump_benchmark.authentic import HOLDOUT_LAW_FAMILY, law_family
from jump_benchmark.authentic_stage_c import (
    STAGE_C_MANIFEST_SHA256,
    run_stage_c,
    stage_c_dataset,
    stage_c_manifest,
)


def test_stage_c_manifest_probe_and_persistence_contract_is_frozen():
    manifest = stage_c_manifest()
    assert STAGE_C_MANIFEST_SHA256 == "4a7c5ed6f53ce7fbc69b8ae4158d86c1eee00eb30a9885d08974a4e244d99f94"
    assert manifest["dataset"]["counts"] == {
        "train": 1024,
        "id_validation": 256,
        "id_test": 256,
        "heldout_law_ood": 256,
    }
    probes = manifest["posthoc_probes"]
    assert probes["encoder_frozen_before_fit"] is True
    assert (probes["outer_folds"], probes["inner_folds"], probes["fold_group"]) == (3, 3, "world_seed")
    assert probes["clustered_ci"] == {"unit": "world_seed", "bootstrap_replicates": 1000, "level": 0.95, "seed": 44123}
    assert probes["multiplicity"].startswith("Benjamini-Hochberg")
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
        device="cpu",
        dry_run=True,
    )
    assert result["status"] == "completed"
    assert result["decision"] == "dry_run_not_evaluated"
    assert result["artifact_verification"] == "passed"
    assert result["mechanistic_evidence"] is False
