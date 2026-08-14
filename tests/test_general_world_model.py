from jump_benchmark.general_world_model import cpu_preflight, dataset, procedural_spec
from jump_contracts.thought_experiments import validate_experiment_spec


def test_general_world_model_v2_schema_split_and_z_only_preflight():
    for variant in ("force_attract", "diffusion", "contagion", "predator_dense", "traffic", "queue"):
        validate_experiment_spec(procedural_spec(12345, variant))
    train = dataset("train", 4)
    ood = dataset("family_ood", 4)
    assert set(train["split_manifest"]["family_ids"]).isdisjoint(ood["split_manifest"]["family_ids"])
    assert set(train["split_manifest"]["world_sha256"]).isdisjoint(ood["split_manifest"]["world_sha256"])
    seam = cpu_preflight()
    assert seam["tiny_overfit_improved"] is True
    assert seam["latent_shape"] == [9, 16]
    assert seam["z_only_decoders"] is True
    assert seam["same_z_raster_reproducible"] is True
