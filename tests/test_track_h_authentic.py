import copy
import json

import pytest

torch = pytest.importorskip("torch")

from jump_benchmark.authentic import (
    FORBIDDEN_ENCODER_FIELDS,
    HOLDOUT_LAW_FAMILY,
    LAW_FAMILIES,
    ObservationArtifact,
    assert_injection_prompt_token_identity,
    authentic_dataset,
    bind_latent_uses,
    build_gated_residual_projector,
    build_world_modules,
    component_stream_seeds,
    dataset_tensors,
    independent_law,
    law_family,
    literal_donor_swap,
    matched_world_pair,
    predictive_objective,
    serialize_latent_tensor,
)


def test_observation_schema_rejects_every_direct_lookup_channel():
    data = authentic_dataset(root_seed=7613, split_counts={"train": 8, "validation": 2, "test": 2})
    row = data["records"][0]
    serialized = json.dumps(row["encoder_input"])
    assert all(name not in serialized for name in ('"partition"', '"law"', '"target"', '"forces"', '"seed"', '"episode_id"'))
    for forbidden in FORBIDDEN_ENCODER_FIELDS | {"index", "hidden_types", "future_target_forces"}:
        leaked = copy.deepcopy(row["encoder_input"])
        leaked[forbidden] = "lookup-channel"
        with pytest.raises(ValueError):
            ObservationArtifact.from_payload(leaked)


def test_generator_domains_holdout_and_partition_only_pairs_are_structural():
    data = authentic_dataset(root_seed=44, split_counts={"train": 8, "validation": 2, "test": 2})
    assert all(len(set(component_stream_seeds(seed).values())) == 7 for seed in range(16))
    assert {law_family(independent_law(seed)) for seed in range(512)} == set(LAW_FAMILIES)
    assert all((law_family(row["sealed_target"]["replacement_law"]) == HOLDOUT_LAW_FAMILY) is (row["split"] == "test") for row in data["records"])
    pair = matched_world_pair(pair_seed=909)
    assert pair["a"]["encoder_input"]["values"][0] == pair["b"]["encoder_input"]["values"][0]
    assert pair["a"]["sealed_target"]["replacement_law"] == pair["b"]["sealed_target"]["replacement_law"]
    assert pair["a"]["decoder_target"] != pair["b"]["decoder_target"]


def test_predictive_encoder_loss_and_gradients_ignore_diagnostic_labels():
    data = authentic_dataset(root_seed=55, split_counts={"train": 8, "validation": 2, "test": 2})
    _, observations, future, relations, laws = dataset_tensors(data, "train")
    encoder, decoder = build_world_modules()
    loss_a, _, z_a = predictive_objective(encoder, decoder, observations, future)
    gradients_a = torch.autograd.grad(loss_a, tuple(encoder.parameters()))
    relations = 1 - relations
    laws = (laws + 3) % len(LAW_FAMILIES)
    encoder.zero_grad(); decoder.zero_grad()
    loss_b, _, z_b = predictive_objective(encoder, decoder, observations, future)
    gradients_b = torch.autograd.grad(loss_b, tuple(encoder.parameters()))
    assert torch.equal(loss_a, loss_b) and z_a.data == z_b.data and z_a.sha256 == z_b.sha256
    assert all(torch.equal(left, right) for left, right in zip(gradients_a, gradients_b))


def test_nontextual_injection_and_literal_donor_bytes_are_fail_closed():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [1] + list(text.encode()) + ([2] if add_special_tokens else [])}

    assert assert_injection_prompt_token_identity(Tokenizer())["token_ids_equal"]
    injector = build_gated_residual_projector(hidden_dim=12)
    hidden, z = torch.randn(2, 5, 12), torch.randn(2, 16)
    assert torch.equal(injector(hidden, z), hidden)
    bound = bind_latent_uses(z[0], z[0].clone(), z[0].clone())
    hash_fields = (
        "world_latent_sha256", "encoder_output_sha256", "decoder_input_sha256",
        "injection_input_sha256", "answer_manifest_world_latent_sha256", "cache_world_latent_sha256",
    )
    assert len({bound[field] for field in hash_fields}) == 1
    donor = serialize_latent_tensor(torch.arange(16, dtype=torch.float32))
    swap = literal_donor_swap("recipient", "donor", donor)
    assert swap["injected_world_latent_sha256"] == donor.sha256
    assert swap["reencoded"] is False and swap["normalization"] == "none"
