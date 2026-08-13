from jump_benchmark.authentic import FORBIDDEN_ENCODER_FIELDS, assert_observation_only
from jump_benchmark.long_horizon import FUTURE_HORIZON, build_long_horizon_modules, long_horizon_dataset


def test_long_horizon_is_observation_only_and_has_eight_step_shape():
    dataset = long_horizon_dataset(
        120731,
        {"train": 3, "id_validation": 2, "id_test": 2, "heldout_law_ood": 2},
        ("train", "id_validation", "id_test", "heldout_law_ood"),
    )
    row = dataset["records"][0]
    assert_observation_only(row["encoder_input"])
    assert not (set(row["encoder_input"]) & FORBIDDEN_ENCODER_FIELDS)
    assert len(row["future_positions"]) == FUTURE_HORIZON
    encoder, decoder = build_long_horizon_modules()
    import torch

    observation = torch.tensor([row["encoder_input"]["values"]], dtype=torch.float32)
    assert decoder(encoder(observation)).shape == (1, FUTURE_HORIZON * 6 * 2)
