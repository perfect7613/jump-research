import torch

from jump_benchmark.authentic import LATENT_DIM, build_gated_residual_projector, matched_world_pair, serialize_latent_tensor
from jump_benchmark.authentic_stage_d import (
    CONDITIONS,
    STAGE_D_MANIFEST_SHA256,
    assert_prompt_identity,
    inject_exact_z,
    persistent_z_injection,
    stage_d_cpu_preflight,
    stage_d_manifest,
    swap_lineage,
    verify_projector_compatibility,
)


class Tokenizer:
    def __call__(self, text, **kwargs):
        ids = [int(value) + 3 for value in text.encode("utf-8")]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return {"input_ids": ids}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(512, 8)
    def get_input_embeddings(self):
        return self.embedding


def test_stage_d_minimal_injection_and_swap_contract():
    tokenizer = Tokenizer()
    assert assert_prompt_identity(tokenizer)["equal"] is True
    assert CONDITIONS == ("own_z", "no_z", "scrambled_z", "wrong_world_z", "swap_a_to_b", "swap_b_to_a")
    projector = build_gated_residual_projector(8)
    compatibility = verify_projector_compatibility(projector, hidden_size=8)
    assert compatibility["input_dim"] == LATENT_DIM
    model = Model()
    prompt = tokenizer("x", return_tensors="pt")
    z = torch.arange(LATENT_DIM, dtype=torch.float32).unsqueeze(0)
    baseline = model.get_input_embeddings()(prompt["input_ids"])
    injected, binding = inject_exact_z(model, projector, prompt["input_ids"], prompt["attention_mask"], z, enabled=True)
    assert torch.equal(injected, baseline)  # exact zero-gate identity
    assert binding["world_latent_sha256"] == serialize_latent_tensor(z).sha256
    with persistent_z_injection(model, projector, z, enabled=True) as persistent:
        model.get_input_embeddings()(prompt["input_ids"])
        model.get_input_embeddings()(prompt["input_ids"][:, :1])
    assert persistent["forward_calls"]["count"] == 2
    pair = matched_world_pair(pair_seed=77231)
    wrong = torch.flip(z, dims=[1])
    swaps = swap_lineage(pair, z[0], wrong[0])
    assert swaps[0]["injected_world_latent_sha256"] == serialize_latent_tensor(z[0]).sha256
    assert swaps[1]["injected_world_latent_sha256"] == serialize_latent_tensor(wrong[0]).sha256
    preflight = stage_d_cpu_preflight(tokenizer, hidden_size=8)
    assert preflight["status"] == "passed" and preflight["gpu_allocated"] is False
    assert stage_d_manifest()["claims"]["mechanistic"] is False
    assert STAGE_D_MANIFEST_SHA256 == "c174f602e2a8df264839e0548224d4251f766400fe03faadfc1fcf296d03e182"
