"""Authentic Track H observation-only world latent and same-z decoder.

This module intentionally does not import or reuse the legacy supplied-partition
Gemma prompt path. Encoder inputs contain visible positions and velocities only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jump_contracts import (
    build_learned_latent_evidence,
    learned_decoder_identity,
    seal_learned_latent_result,
    tensor_bytes_sha256,
)
from .canonical import sha256_json
from .render import HEIGHT, WIDTH
from .simulator import EpisodeSpec, Law, SimulatorConfig, derive_seed, generate_episode

AUTHENTIC_SCHEMA_VERSION = "jump.track-h-authentic-worlds/v1"
OBSERVATION_SCHEMA_VERSION = "jump.track-h-observation-tensor/v1"
LATENT_SCHEMA_VERSION = "jump.track-h-world-latent/v1"
LATENT_DIM = 16
EVIDENCE_FRAMES = 4
LAW_FAMILIES = tuple(
    (same, different, exponent)
    for same in ("attract", "repel")
    for different in ("attract", "repel")
    for exponent in (1, 2)
)
HOLDOUT_LAW_FAMILY = ("repel", "repel", 2)
FORBIDDEN_ENCODER_FIELDS = frozenset(
    {"partition", "law", "replacement_law", "target", "forces", "world_seed", "seed", "episode_id", "adequacy", "prior_law"}
)
COMPONENT_DOMAINS = (
    "same-sign", "different-sign", "exponent", "partition", "appearance", "initial-state", "record-order"
)


def _choice(seed: int, domain: str, values: tuple[Any, ...]) -> Any:
    return random.Random(derive_seed(seed, "authentic:" + domain)).choice(values)


def independent_law(seed: int) -> Law:
    """Draw law components from distinct deterministic random streams."""
    return Law(
        _choice(seed, "same-sign", ("attract", "repel")),
        _choice(seed, "different-sign", ("attract", "repel")),
        _choice(seed, "exponent", (1, 2)),
    )


def component_stream_seeds(seed: int) -> dict[str, int]:
    """Expose structural independence evidence without exposing it to the encoder."""
    values = {
        domain: derive_seed(seed, domain if domain in {"appearance", "initial-state"} else "authentic:" + domain)
        for domain in COMPONENT_DOMAINS
    }
    if len(set(values.values())) != len(values):
        raise RuntimeError("independent generator stream collision")
    return values


def independent_partition(seed: int, domain: str = "partition") -> tuple[int, ...]:
    mask = random.Random(derive_seed(seed, "authentic:" + domain)).randint(1, 31)
    return (0,) + tuple((mask >> index) & 1 for index in range(5))


def law_family(law: Law | dict[str, Any]) -> tuple[str, str, int]:
    if isinstance(law, Law):
        return law.same, law.different, law.exponent
    return law["same"], law["different"], int(law["exponent"])


def serialize_visible_observations(episode: dict[str, Any], frames: int = EVIDENCE_FRAMES) -> dict[str, Any]:
    """Return the complete and only allowed encoder input payload."""
    timeline = [episode["initial_state"], *episode["observations"]]
    if frames < 2 or frames >= len(timeline):
        raise ValueError("evidence frames must leave at least one future decoder target")
    values = []
    for frame in timeline[:frames]:
        values.append(
            [
                [float(position[0]), float(position[1]), float(velocity[0]), float(velocity[1])]
                for position, velocity in zip(frame["positions"], frame["velocities"])
            ]
        )
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "dtype": "float32",
        "shape": [frames, 6, 4],
        "features": ["position_x", "position_y", "velocity_x", "velocity_y"],
        "values": values,
    }


def assert_observation_only(payload: dict[str, Any]) -> None:
    expected = {"schema_version", "dtype", "shape", "features", "values"}
    if set(payload) != expected or payload.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("encoder input fields are not the frozen observation-only contract")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    lowered = serialized.lower()
    found = sorted(field for field in FORBIDDEN_ENCODER_FIELDS if f'"{field}"' in lowered)
    if found:
        raise ValueError(f"encoder input contains forbidden direct-lookup fields: {found}")
    if payload.get("dtype") != "float32" or payload.get("shape") != [EVIDENCE_FRAMES, 6, 4]:
        raise ValueError("encoder input tensor dtype/shape mismatch")
    if payload.get("features") != ["position_x", "position_y", "velocity_x", "velocity_y"]:
        raise ValueError("encoder features differ from visible kinematics")


def _pair_relations(partition: list[int]) -> list[float]:
    return [float(partition[left] == partition[right]) for left in range(6) for right in range(left + 1, 6)]


def _record(episode: dict[str, Any], split: str) -> dict[str, Any]:
    encoder_input = serialize_visible_observations(episode)
    artifact = ObservationArtifact.from_payload(encoder_input)
    target_frame = episode["observations"][EVIDENCE_FRAMES - 1]
    law = episode["target"]["replacement_law"]
    return {
        "split": split,
        "world_seed": episode["world_seed"],
        "episode_id": episode["episode_id"],
        "encoder_input": artifact.payload(),
        "encoder_input_sha256": artifact.sha256(),
        "observation_binding": {"source_world_id": episode["episode_id"], "source_observation_sha256": artifact.sha256()},
        "decoder_target": {"next_positions": target_frame["positions"]},
        "diagnostic_target": {
            "pair_relations": _pair_relations(episode["target"]["partition"]),
            "law_family_index": LAW_FAMILIES.index(law_family(law)),
        },
        "sealed_target": {
            "partition": episode["target"]["partition"],
            "replacement_law": law,
        },
    }


def authentic_dataset(
    *, root_seed: int, split_counts: dict[str, int], config: SimulatorConfig = SimulatorConfig(steps=6)
) -> dict[str, Any]:
    """Generate seed-disjoint data with a fully held-out law family in test."""
    if set(split_counts) != {"train", "validation", "test"} or any(value <= 0 for value in split_counts.values()):
        raise ValueError("split_counts must contain positive train/validation/test counts")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for split in ("train", "validation", "test"):
        candidate = 0
        while sum(row["split"] == split for row in records) < split_counts[split]:
            seed = derive_seed(root_seed, f"authentic:{split}:{candidate}")
            candidate += 1
            if seed in seen:
                raise RuntimeError("world seed collision")
            law = independent_law(seed)
            is_holdout = law_family(law) == HOLDOUT_LAW_FAMILY
            if (split == "test") != is_holdout:
                continue
            seen.add(seed)
            partition = independent_partition(seed)
            episode = generate_episode(EpisodeSpec(seed, split, law, True, config, partition))
            record = _record(episode, split)
            record["generator_provenance"] = {
                "component_stream_seeds": component_stream_seeds(seed),
                "same_sign": law.same, "different_sign": law.different,
                "exponent": law.exponent, "partition": list(partition),
                "nuisance_appearance_stream_seed": derive_seed(seed, "appearance"),
                "nuisance_initial_state_stream_seed": derive_seed(seed, "initial-state"),
                "ordering_index_serialized_to_encoder": False,
            }
            records.append(record)
    # Input serialization has no row index. An independent ordering stream also
    # prevents accepted-candidate order from becoming an external label cue.
    for split in ("train", "validation", "test"):
        indices = [index for index, row in enumerate(records) if row["split"] == split]
        shuffled = [records[index] for index in indices]
        random.Random(derive_seed(root_seed, f"authentic:record-order:{split}")).shuffle(shuffled)
        for index, row in zip(indices, shuffled):
            records[index] = row
    return {
        "schema_version": AUTHENTIC_SCHEMA_VERSION,
        "root_seed": root_seed,
        "split_counts": {key: split_counts[key] for key in ("train", "validation", "test")},
        "split_policy": "world-seed-disjoint; held-out law family appears only in test",
        "heldout_law_family": {"same": HOLDOUT_LAW_FAMILY[0], "different": HOLDOUT_LAW_FAMILY[1], "exponent": HOLDOUT_LAW_FAMILY[2]},
        "law_component_streams": ["same-sign", "different-sign", "exponent"],
        "appearance_stream": "legacy appearance stream independent of authentic partition/law streams",
        "record_order_policy": "independent per-split shuffle; no index is serialized to encoder",
        "component_domain_separation": list(COMPONENT_DOMAINS),
        "records": records,
    }


def matched_world_pair(*, pair_seed: int, config: SimulatorConfig = SimulatorConfig(steps=6)) -> dict[str, Any]:
    """Build a literal A/B pair with identical visible t0 and different futures."""
    partition_a = independent_partition(pair_seed, "pair-a-partition")
    partition_b = independent_partition(pair_seed, "pair-b-partition")
    if partition_b == partition_a:
        partition_b = (0,) + tuple(1 - value for value in partition_a[1:])
    # Canonical swap family fixes the candidate/correct law and changes only
    # hidden partition, which changes the later consequence.
    pair_law_seed = derive_seed(pair_seed, "authentic:pair-law")
    same = _choice(pair_law_seed, "same-sign", ("attract", "repel"))
    # Canonical partition-swap pairs require a relation-dependent law so a
    # changed partition cannot yield a coincident consequence.
    law_a = Law(same, "repel" if same == "attract" else "attract", _choice(pair_law_seed, "exponent", (1, 2)))
    law_b = law_a
    a = generate_episode(EpisodeSpec(pair_seed, "swap-a", law_a, True, config, partition_a))
    b = generate_episode(EpisodeSpec(pair_seed, "swap-b", law_b, True, config, partition_b))
    if a["initial_state"]["positions"] != b["initial_state"]["positions"] or a["initial_state"]["velocities"] != b["initial_state"]["velocities"]:
        raise RuntimeError("matched pair visible prefix differs")
    if a["observations"][-1]["positions"] == b["observations"][-1]["positions"]:
        raise RuntimeError("matched pair has coincident consequence")
    return {
        "schema_version": "jump.track-h-matched-world-pair/v1",
        "pair_id": hashlib.sha256(f"authentic-pair:{pair_seed}".encode()).hexdigest()[:20],
        "visible_prefix_frames": 1,
        "candidate_laws": [list(item) for item in LAW_FAMILIES],
        "a": _record(a, "swap"),
        "b": _record(b, "swap"),
    }


@dataclass(frozen=True)
class SerializedLatent:
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    sha256: str


@dataclass(frozen=True)
class ObservationArtifact:
    """The sole structural input accepted by the world encoder."""
    dtype: str
    shape: tuple[int, ...]
    features: tuple[str, ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ObservationArtifact":
        assert_observation_only(payload)
        return cls(
            dtype=payload["dtype"], shape=tuple(payload["shape"]),
            features=tuple(payload["features"]),
            values=tuple(tuple(tuple(float(item) for item in obj) for obj in frame) for frame in payload["values"]),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION, "dtype": self.dtype,
            "shape": list(self.shape), "features": list(self.features),
            "values": [[list(obj) for obj in frame] for frame in self.values],
        }

    def bytes(self) -> bytes:
        import numpy as np
        return np.asarray(self.values, dtype="<f4", order="C").tobytes(order="C")

    def descriptor(self) -> dict[str, Any]:
        return {
            "artifact_name": "encoder-observation.f32le.bin", "media_type": "application/octet-stream",
            "dtype": "float32-le", "shape": list(self.shape), "order": "C",
            "byte_length": len(self.bytes()), "raw_bytes_sha256": self.sha256(),
            "features": list(self.features),
        }

    @classmethod
    def from_bytes(
        cls, raw: bytes, *, dtype: str, shape: list[int], order: str, features: list[str]
    ) -> "ObservationArtifact":
        import numpy as np
        if dtype != "float32-le" or order != "C" or shape != [EVIDENCE_FRAMES, 6, 4]:
            raise ValueError("observation tensor descriptor mismatch")
        if features != ["position_x", "position_y", "velocity_x", "velocity_y"]:
            raise ValueError("observation feature order mismatch")
        if len(raw) != math.prod(shape) * 4:
            raise ValueError("observation byte length does not match dtype/shape")
        values = np.frombuffer(raw, dtype="<f4").copy().reshape(shape, order="C")
        return cls(dtype="float32", shape=tuple(shape), features=tuple(features), values=tuple(tuple(tuple(float(item) for item in obj) for obj in frame) for frame in values))

    def sha256(self) -> str:
        return hashlib.sha256(self.bytes()).hexdigest()


def serialize_latent_tensor(tensor: Any) -> SerializedLatent:
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    shape = tuple(value.shape)
    if not shape or shape[-1] != LATENT_DIM:
        raise ValueError(f"world latent final dimension must be {LATENT_DIM}")
    raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
    return SerializedLatent(
        "float32-le", shape, raw,
        tensor_bytes_sha256(raw, dtype="float32-le", shape=shape, order="C"),
    )


def deserialize_latent_tensor(latent: SerializedLatent, *, device: str = "cpu") -> Any:
    import numpy as np
    import torch

    if tensor_bytes_sha256(latent.data, dtype=latent.dtype, shape=latent.shape, order="C") != latent.sha256:
        raise ValueError("latent bytes/header/hash mismatch")
    array = np.frombuffer(latent.data, dtype="<f4").copy().reshape(latent.shape)
    return torch.from_numpy(array).to(device)


def exact_bytes_straight_through(tensor: Any) -> tuple[Any, SerializedLatent]:
    """Use byte-roundtripped float32 values while retaining encoder gradients."""
    latent = serialize_latent_tensor(tensor)
    roundtrip = deserialize_latent_tensor(latent, device=str(tensor.device))
    return tensor + (roundtrip - tensor).detach(), latent


def bind_latent_uses(encoder_output: Any, decoder_input: Any, injection_input: Any) -> dict[str, Any]:
    """Independently materialize and bind each consumer's exact raw z bytes."""
    encoder = serialize_latent_tensor(encoder_output)
    decoder = serialize_latent_tensor(decoder_input)
    injection = serialize_latent_tensor(injection_input)
    if not (encoder.data == decoder.data == injection.data):
        raise RuntimeError("encoder/decoder/injection raw z bytes differ")
    if not (encoder.sha256 == decoder.sha256 == injection.sha256):
        raise RuntimeError("encoder/decoder/injection tensor hashes differ")
    value = {
        "world_latent_sha256": encoder.sha256,
        "encoder_output_sha256": encoder.sha256,
        "decoder_input_sha256": decoder.sha256,
        "injection_input_sha256": injection.sha256,
        "answer_manifest_world_latent_sha256": injection.sha256,
        "cache_world_latent_sha256": injection.sha256,
        "raw_bytes_sha256": hashlib.sha256(encoder.data).hexdigest(),
        "world_latent_id": "z-" + encoder.sha256[:20],
        "latent_dtype": encoder.dtype,
        "latent_shape": list(encoder.shape),
        "latent_order": "C",
        "latent_byte_length": len(encoder.data),
    }
    return value


def module_content_sha256(module: Any) -> str:
    """Hash names, dtype/shape headers, and raw contiguous parameter bytes."""
    digest = hashlib.sha256()
    for name, parameter in sorted(module.state_dict().items()):
        value = parameter.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(value.numpy().dtype), "shape": list(value.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        digest.update(header + b"\0" + value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _git_revision() -> str:
    configured = __import__("os").environ.get("JUMP_CODE_VERSION")
    if configured:
        return configured
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


INJECTION_PROMPT = (
    "A learned observation-only world state is available through the model adapter. "
    "Infer the six-object hidden partition and force law. Return canonical JSON only."
)


def matched_injection_prompt() -> str:
    """E and no-z use exactly this prompt; z never enters text."""
    return INJECTION_PROMPT


def assert_injection_prompt_token_identity(tokenizer: Any) -> dict[str, Any]:
    e_ids = tokenizer(matched_injection_prompt(), add_special_tokens=True)["input_ids"]
    no_z_ids = tokenizer(matched_injection_prompt(), add_special_tokens=True)["input_ids"]
    if e_ids != no_z_ids:
        raise RuntimeError("E and no-z prompt tokens differ")
    return {"token_ids_equal": True, "token_count": len(e_ids), "z_serialized_in_prompt": False}


def build_gated_residual_projector(hidden_dim: int):
    import torch

    class GatedResidualProjector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = torch.nn.Linear(LATENT_DIM, hidden_dim, bias=False)
            self.gate = torch.nn.Parameter(torch.zeros(()))

        def forward(self, hidden_states, exact_z):
            if exact_z.shape[-1] != LATENT_DIM:
                raise ValueError("injection z dimension mismatch")
            delta = self.projector(exact_z).unsqueeze(1) * torch.tanh(self.gate)
            mask = torch.zeros_like(hidden_states)
            mask[:, :1, :] = delta
            return hidden_states + mask

    return GatedResidualProjector()


def literal_donor_swap(recipient_world_id: str, donor_world_id: str, donor_latent: SerializedLatent) -> dict[str, Any]:
    if recipient_world_id == donor_world_id:
        raise ValueError("donor and recipient worlds must differ")
    # No transformation, re-encoding, or recipient-specific normalization.
    injection = SerializedLatent(donor_latent.dtype, donor_latent.shape, donor_latent.data, donor_latent.sha256)
    if injection.data != donor_latent.data or injection.sha256 != donor_latent.sha256:
        raise RuntimeError("donor latent bytes changed")
    return {
        "recipient_world_id": recipient_world_id,
        "donor_world_id": donor_world_id,
        "donor_latent_id": "z-" + donor_latent.sha256[:20],
        "injected_world_latent_sha256": injection.sha256,
        "normalization": "none",
        "reencoded": False,
    }


def build_world_modules(input_dim: int = EVIDENCE_FRAMES * 6 * 4, latent_dim: int = LATENT_DIM):
    import torch

    class ObservationEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(torch.nn.Linear(input_dim, 64), torch.nn.GELU(), torch.nn.Linear(64, latent_dim))

        def forward(self, value):
            return self.network(value.reshape(value.shape[0], -1))

    class SameZDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Sequential(torch.nn.Linear(latent_dim, 64), torch.nn.GELU())
            self.position_head = torch.nn.Linear(64, 12)

        def forward(self, z):
            hidden = self.trunk(z)
            return self.position_head(hidden).reshape(-1, 6, 2)

    return ObservationEncoder(), SameZDecoder()


def dataset_tensors(dataset: dict[str, Any], split: str, device: str = "cpu"):
    import torch

    rows = [row for row in dataset["records"] if row["split"] == split]
    if not rows:
        raise ValueError(f"no rows for split {split}")
    artifacts = [ObservationArtifact.from_payload(row["encoder_input"]) for row in rows]
    verified = [ObservationArtifact.from_bytes(item.bytes(), dtype="float32-le", shape=list(item.shape), order="C", features=list(item.features)) for item in artifacts]
    inputs = torch.tensor([item.values for item in verified], dtype=torch.float32, device=device)
    positions = torch.tensor([row["decoder_target"]["next_positions"] for row in rows], dtype=torch.float32, device=device)
    relations = torch.tensor([row["diagnostic_target"]["pair_relations"] for row in rows], dtype=torch.float32, device=device)
    laws = torch.tensor([row["diagnostic_target"]["law_family_index"] for row in rows], dtype=torch.long, device=device)
    return rows, inputs, positions, relations, laws


def predictive_objective(encoder: Any, decoder: Any, observations: Any, future_positions: Any):
    """Canonical objective has no label arguments and cannot consume them."""
    import torch
    z = encoder(observations)
    exact_z, latent = exact_bytes_straight_through(z)
    predicted = decoder(exact_z)
    position_loss = torch.nn.functional.mse_loss(predicted, future_positions)
    regularization = 1e-4 * torch.mean(exact_z**2)
    return position_loss + regularization, position_loss, latent


def train_world_modules(dataset: dict[str, Any], *, steps: int, learning_rate: float, device: str = "cpu") -> tuple[Any, Any, dict[str, float]]:
    import torch

    torch.manual_seed(7613)
    encoder, decoder = build_world_modules()
    encoder.to(device); decoder.to(device)
    _, inputs, positions, relations, laws = dataset_tensors(dataset, "train", device)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

    def forward_loss():
        return predictive_objective(encoder, decoder, inputs, positions)

    encoder.train(); decoder.train()
    initial, _, _ = forward_loss()
    initial_value = float(initial.detach().cpu())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = forward_loss()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite world-module loss")
        loss.backward(); optimizer.step()
    final, position_loss, latent = forward_loss()
    return encoder, decoder, {
        "initial_total_loss": initial_value,
        "final_total_loss": float(final.detach().cpu()),
        "final_position_mse": float(position_loss.detach().cpu()),
    }


def render_predicted_state_svg(predicted_positions: list[list[float]], world_latent_sha256: str, *, bounds: float = 3.0) -> str:
    """Render decoder output only; no episode or ground-truth fields are accepted."""
    if len(predicted_positions) != 6 or any(len(item) != 2 or any(not math.isfinite(float(v)) for v in item) for item in predicted_positions):
        raise ValueError("predicted positions must be six finite 2-D points")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="512" height="512" fill="#eff6ff"/>',
        '<text x="24" y="32" font-family="monospace" font-size="15" font-weight="bold" fill="#1e3a8a">PREDICTED FROM LEARNED z</text>',
        f'<text x="24" y="52" font-family="monospace" font-size="10" fill="#475569">z sha256 {world_latent_sha256[:16]}…</text>',
    ]
    for index, (px, py) in enumerate(predicted_positions):
        x = (float(px) + bounds) / (2 * bounds) * (WIDTH - 64) + 32
        y = HEIGHT - ((float(py) + bounds) / (2 * bounds) * (HEIGHT - 80) + 32)
        elements.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="17" fill="#2563eb" stroke="#172554" stroke-width="3"/>')
        elements.append(f'<text x="{x:.3f}" y="{y+4:.3f}" text-anchor="middle" font-family="monospace" font-size="11" fill="white">o{index}</text>')
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def stage_a_smoke(output_dir: Path, *, experiment_spec: dict[str, Any]) -> dict[str, Any]:
    from .experiment_spec import validate_experiment_spec
    from .task_adapter import write_track_h_task_evidence

    plan = validate_experiment_spec(experiment_spec)
    experiment_spec_sha256 = sha256_json(plan)
    dataset = authentic_dataset(root_seed=88173, split_counts={"train": 8, "validation": 2, "test": 2})
    encoder, decoder, metrics = train_world_modules(dataset, steps=400, learning_rate=3e-3)
    rows, inputs, _, _, _ = dataset_tensors(dataset, "train")
    import torch
    with torch.no_grad():
        z = encoder(inputs[:1]).squeeze(0); latent = serialize_latent_tensor(z); exact_z = deserialize_latent_tensor(latent).unsqueeze(0)
        predicted = decoder(exact_z)
        injection_z = deserialize_latent_tensor(latent)
        injector = build_gated_residual_projector(hidden_dim=12)
        hidden = torch.zeros(1, 3, 12)
        if not torch.equal(injector(hidden, injection_z.unsqueeze(0)), hidden):
            raise RuntimeError("zero-gate injection is not identity")
    if serialize_latent_tensor(exact_z.squeeze(0)).sha256 != latent.sha256:
        raise RuntimeError("decoder did not consume byte-identical z")
    if metrics["final_total_loss"] >= metrics["initial_total_loss"] * 0.2:
        raise RuntimeError("tiny overfit gate did not reduce loss by at least 80%")
    output_dir.mkdir(parents=True, exist_ok=False)
    model_config = {
        "encoder": "MLP 96->64->16 GELU; observation tensors only",
        "decoder": "same-z predictive MLP 16->64->12 future-position output",
        "latent_dim": LATENT_DIM,
        "injection": "trainable 16->Gemma-hidden projector plus scalar tanh gate at first token residual; z absent from text",
    }
    producer_code_version = _git_revision()
    training_manifest = {
        "schema_version": "jump.track-h-authentic-stage-a-manifest/v1",
        "producer_code_version": producer_code_version,
        "dataset_spec": {"root_seed": 88173, "split_counts": {"train": 8, "validation": 2, "test": 2}},
        "steps": 400, "learning_rate": 3e-3, "loss_weights": {"future_positions": 1.0, "bottleneck_l2": 1e-4},
        "label_supervision": "none; pair relations and law family are excluded from optimization",
        "posthoc_probe_protocol": "encoder frozen; train-world grouped nested CV; ID validation and held-out-family test reported separately",
        "model_config": model_config, "mechanistic_evidence": False,
    }
    manifest_sha = sha256_json(training_manifest)
    encoder_state_sha, decoder_state_sha = module_content_sha256(encoder), module_content_sha256(decoder)
    from safetensors.torch import save_file
    save_file({name: value.detach().cpu() for name, value in encoder.state_dict().items()}, output_dir / "encoder.safetensors")
    save_file({name: value.detach().cpu() for name, value in decoder.state_dict().items()}, output_dir / "decoder.safetensors")
    (output_dir / "training-manifest.json").write_text(json.dumps(training_manifest, sort_keys=True, separators=(",", ":")) + "\n")
    decoder_artifact_sha = hashlib.sha256((output_dir / "decoder.safetensors").read_bytes()).hexdigest()
    encoder_artifact_sha = hashlib.sha256((output_dir / "encoder.safetensors").read_bytes()).hexdigest()
    svg = render_predicted_state_svg(predicted[0].tolist(), latent.sha256)
    svg_bytes = svg.encode()
    (output_dir / "predicted-from-z.svg").write_bytes(svg_bytes)
    observation_artifact = ObservationArtifact.from_payload(rows[0]["encoder_input"])
    observation_bytes = observation_artifact.bytes()
    (output_dir / "encoder-observation.f32le.bin").write_bytes(observation_bytes)
    (output_dir / "encoder-observation-metadata.json").write_text(json.dumps(observation_artifact.descriptor(), sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "world-latent.f32le.bin").write_bytes(latent.data)
    encoder_capture = serialize_latent_tensor(z)
    decoder_capture = serialize_latent_tensor(exact_z.squeeze(0))
    injection_capture = serialize_latent_tensor(injection_z)
    observation_binding = bind_source_observation(
        source_record=rows[0], observation_bytes=observation_bytes,
        source_world_id=rows[0]["episode_id"],
    )
    observation_binding = {**observation_binding, **observation_artifact.descriptor()}
    encoder_identity = {
        "artifact_name": "encoder.safetensors", "artifact_sha256": encoder_artifact_sha,
        "training_manifest_sha256": manifest_sha,
        "architecture_config_sha256": sha256_json(model_config),
        "architecture": model_config["encoder"],
    }
    answer = {
        "predicted_next_positions": predicted[0].tolist(),
        "producer_bindings": {
            "encoder_identity": encoder_identity,
            "source_observation": observation_binding,
            "architecture_manifest_sha256": AUTHENTIC_ARCHITECTURE_MANIFEST_SHA256,
            "experiment_id": plan["experiment_id"],
            "experiment_spec_sha256": experiment_spec_sha256,
        },
    }
    evidence = build_learned_latent_evidence(
        encoder_output=encoder_capture.data,
        decoder_input=decoder_capture.data,
        injection_input=injection_capture.data,
        encoder_observation=observation_bytes,
        encoder_observation_artifact_name="encoder-observation.f32le.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le", shape=[LATENT_DIM], order="C",
        recipient_world_id=rows[0]["episode_id"], world_pair_id="stage-a-singleton",
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder.safetensors", artifact_sha256=decoder_artifact_sha,
            training_manifest_sha256=manifest_sha, code_version=producer_code_version,
            architecture="same-z-16d-to-six-object-next-position-v1",
        ),
        decoded_image=svg_bytes, decoded_image_media_type="image/svg+xml",
        answer=answer, tensor_artifact_name="world-latent.f32le.bin",
    )
    sealed = seal_learned_latent_result(
        evidence, source="cached", manifest_sha256=manifest_sha,
        run_id="stage-a-local-88173", code_version=producer_code_version, checkpoint_id=decoder_artifact_sha,
    )
    (output_dir / "learned-latent-evidence.json").write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "sealed-result.json").write_text(json.dumps(sealed, sort_keys=True, separators=(",", ":")) + "\n")
    provenance = bind_latent_uses(z, exact_z.squeeze(0), injection_z)
    result = {
        "schema_version": "jump.track-h-authentic-stage-a-result/v1",
        "status": "completed",
        "dataset_sha256": sha256_json(dataset),
        "encoder_input_schema": OBSERVATION_SCHEMA_VERSION,
        "forbidden_encoder_fields": sorted(FORBIDDEN_ENCODER_FIELDS),
        "direct_field_lookup_leakage": False,
        **provenance,
        "hash_equality": True,
        "metrics": metrics,
        "encoder_weights_sha256": encoder_artifact_sha,
        "decoder_weights_sha256": decoder_artifact_sha,
        "encoder_state_sha256": encoder_state_sha,
        "decoder_state_sha256": decoder_state_sha,
        "model_config_sha256": sha256_json(model_config),
        "model_config": model_config,
        "training_manifest_sha256": manifest_sha,
        "producer_code_version": producer_code_version,
        "shared_contracts": ["jump.learned-latent-evidence/v1", "jump.sealed-result/v1", "jump.tensor-preimage/v1"],
        "learned_latent_evidence_sha256": hashlib.sha256((output_dir / "learned-latent-evidence.json").read_bytes()).hexdigest(),
        "weights_frozen_before_prediction": True,
        "example_episode_id": rows[0]["episode_id"],
        "image_source": "learned decoder predicted positions followed by deterministic SVG renderer",
        "mechanistic_evidence": False,
    }
    evidence_result = write_track_h_task_evidence(
        output_dir,
        metrics=[
            {"name": "initial_total_loss", "value": metrics["initial_total_loss"]},
            {"name": "final_total_loss", "value": metrics["final_total_loss"]},
            {"name": "final_position_mse", "value": metrics["final_position_mse"]},
        ],
        terminal=result,
        experiment_spec=plan,
        track_h={"stage": "A", "mechanistic_evidence": False},
    )
    return {**result, "experiment_id": plan["experiment_id"], "experiment_spec_sha256": experiment_spec_sha256, "task_evidence": evidence_result}


def _binary_auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUC needs both relation classes")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def evaluate_world_modules(encoder: Any, decoder: Any, dataset: dict[str, Any], split: str, device: str) -> dict[str, Any]:
    import torch

    rows, inputs, positions, relations, laws = dataset_tensors(dataset, split, device)
    predicted_positions = []
    latent_records = []
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        for index, row in enumerate(rows):
            encoder_z = encoder(inputs[index:index + 1]).squeeze(0)
            encoder_capture = serialize_latent_tensor(encoder_z)
            decoder_tensor = deserialize_latent_tensor(encoder_capture, device=device)
            decoder_capture = serialize_latent_tensor(decoder_tensor)
            if encoder_capture.data != decoder_capture.data:
                raise RuntimeError("decoder site did not independently materialize identical z bytes")
            predicted = decoder(decoder_tensor.unsqueeze(0))
            predicted_positions.append(predicted.squeeze(0))
            injection_tensor = deserialize_latent_tensor(encoder_capture, device=device)
            latent_records.append({
                "world_id": row["episode_id"],
                **bind_latent_uses(encoder_z, decoder_tensor, injection_tensor),
                "z_norm": float(torch.linalg.vector_norm(decoder_tensor).cpu()),
            })
    predicted_positions_t = torch.stack(predicted_positions)
    rmse = torch.sqrt(torch.mean((predicted_positions_t - positions) ** 2))
    target_rms = torch.sqrt(torch.mean(positions**2)).clamp(min=1e-8)
    return {
        "examples": len(rows),
        "future_position_nrmse": float((rmse / target_rms).cpu()),
        "diagnostic_probe_status": "not_fit_in_world_training; post-hoc frozen-encoder protocol only",
        "latent_norm_mean": sum(item["z_norm"] for item in latent_records) / len(latent_records),
        "latent_norm_min": min(item["z_norm"] for item in latent_records),
        "latent_norm_max": max(item["z_norm"] for item in latent_records),
        "latent_records": latent_records,
        "first_prediction": predicted_positions_t[0].cpu().tolist(),
    }


def run_world_stage(
    *,
    stage: str,
    output_root: Path,
    checkpoint_root: Path,
    experiment_spec: dict[str, Any],
    device: str = "cuda",
) -> dict[str, Any]:
    """Train/evaluate the Stage B predictive pilot under its frozen contract.

    Stage C has a separate, stricter implementation and manifest in
    :mod:`jump_benchmark.authentic_stage_c`; accepting it here would create two
    conflicting Stage C contracts.
    """
    import platform
    import time
    import torch
    from safetensors.torch import save_file
    from .experiment_spec import validate_experiment_spec
    from .task_adapter import write_track_h_task_evidence

    plan = validate_experiment_spec(experiment_spec)
    experiment_spec_sha256 = sha256_json(plan)

    settings = {
        "B": {"counts": {"train": 128, "validation": 32, "test": 32}, "steps": 200, "learning_rate": 1e-3},
    }
    if stage not in settings:
        raise ValueError("run_world_stage is frozen to Stage B; use authentic_stage_c.run_stage_c for Stage C")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote world stages")
    spec = settings[stage]
    torch.manual_seed(88173)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(88173); torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    dataset = authentic_dataset(root_seed=88173, split_counts=spec["counts"])
    dataset_sha = sha256_json(dataset)
    encoder, decoder = build_world_modules(); encoder.to(device); decoder.to(device)
    _, inputs, positions, relations, laws = dataset_tensors(dataset, "train", device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=spec["learning_rate"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); first_loss = None
    for step in range(1, spec["steps"] + 1):
        encoder.train(); decoder.train(); optimizer.zero_grad(set_to_none=True)
        loss, position_loss, _ = predictive_objective(encoder, decoder, inputs, positions)
        if not torch.isfinite(loss): raise RuntimeError("non-finite authentic world loss")
        if first_loss is None: first_loss = float(loss.detach().cpu())
        loss.backward(); optimizer.step()
        if step % 50 == 0 or step == spec["steps"]:
            torch.save({"stage": stage, "step": step, "dataset_sha256": dataset_sha, "encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_root / "latest.pt")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    encoder_hash, decoder_hash = module_content_sha256(encoder), module_content_sha256(decoder)
    for parameter in list(encoder.parameters()) + list(decoder.parameters()): parameter.requires_grad_(False)
    validation = evaluate_world_modules(encoder, decoder, dataset, "validation", device)
    test = evaluate_world_modules(encoder, decoder, dataset, "test", device)
    if (encoder_hash, decoder_hash) != (module_content_sha256(encoder), module_content_sha256(decoder)):
        raise RuntimeError("world weights changed after freeze/evaluation")
    output_root.mkdir(parents=True, exist_ok=False)
    save_file({name: value.detach().cpu() for name, value in encoder.state_dict().items()}, output_root / "encoder.safetensors")
    save_file({name: value.detach().cpu() for name, value in decoder.state_dict().items()}, output_root / "decoder.safetensors")
    first_prediction = test.pop("first_prediction")
    validation.pop("first_prediction")
    config = {
        "stage": stage, "latent_dim": LATENT_DIM, "evidence_shape": [EVIDENCE_FRAMES, 6, 4],
        "encoder": "96->64->16 GELU MLP", "decoder": "16->64 GELU -> 12 future positions",
        "loss_weights": {"future_position_mse": 1.0, "bottleneck_l2": 1e-4},
        "label_supervision": "none; pair/law labels are excluded from encoder/decoder optimization",
        "posthoc_probe_protocol": "freeze encoder; train probes on train-z only; grouped nested CV by world seed; evaluate ID validation separately from held-out-law-family test",
        "training": spec,
    }
    encoder_artifact_sha = hashlib.sha256((output_root / "encoder.safetensors").read_bytes()).hexdigest()
    decoder_artifact_sha = hashlib.sha256((output_root / "decoder.safetensors").read_bytes()).hexdigest()
    test_rows, test_inputs, _, _, _ = dataset_tensors(dataset, "test", device)
    with torch.no_grad():
        encoder_z = encoder(test_inputs[:1]).squeeze(0)
        encoder_capture = serialize_latent_tensor(encoder_z)
        decoder_tensor = deserialize_latent_tensor(encoder_capture, device=device)
        decoder_capture = serialize_latent_tensor(decoder_tensor)
        injection_tensor = deserialize_latent_tensor(encoder_capture, device=device)
        injection_capture = serialize_latent_tensor(injection_tensor)
    observation = ObservationArtifact.from_payload(test_rows[0]["encoder_input"])
    observation_bytes = observation.bytes()
    bind_source_observation(source_record=test_rows[0], observation_bytes=observation_bytes, source_world_id=test_rows[0]["episode_id"])
    svg = render_predicted_state_svg(first_prediction, encoder_capture.sha256)
    svg_bytes = svg.encode()
    (output_root / "predicted-from-z.svg").write_bytes(svg_bytes)
    (output_root / "encoder-observation.f32le.bin").write_bytes(observation_bytes)
    (output_root / "encoder-observation-metadata.json").write_text(json.dumps(observation.descriptor(), sort_keys=True, separators=(",", ":")) + "\n")
    (output_root / "world-latent.f32le.bin").write_bytes(encoder_capture.data)
    answer = {
        "predicted_next_positions": first_prediction,
        "producer_bindings": {
            "encoder_identity": {
                "artifact_name": "encoder.safetensors", "artifact_sha256": encoder_artifact_sha,
                "training_manifest_sha256": AUTHENTIC_ARCHITECTURE_MANIFEST_SHA256,
                "architecture_config_sha256": sha256_json(config), "architecture": config["encoder"],
            },
            "source_observation": {
                **test_rows[0]["observation_binding"], **observation.descriptor(),
            },
            "architecture_manifest_sha256": AUTHENTIC_ARCHITECTURE_MANIFEST_SHA256,
            "experiment_id": plan["experiment_id"],
            "experiment_spec_sha256": experiment_spec_sha256,
        },
    }
    evidence = build_learned_latent_evidence(
        encoder_output=encoder_capture.data, decoder_input=decoder_capture.data,
        injection_input=injection_capture.data, encoder_observation=observation_bytes,
        encoder_observation_artifact_name="encoder-observation.f32le.bin", encoder_observation_media_type="application/octet-stream",
        dtype="float32-le", shape=[LATENT_DIM], order="C", tensor_artifact_name="world-latent.f32le.bin",
        recipient_world_id=test_rows[0]["episode_id"], world_pair_id=f"stage-{stage.lower()}-heldout-singleton",
        learned_decoder=learned_decoder_identity(
            artifact_name="decoder.safetensors", artifact_sha256=decoder_artifact_sha,
            training_manifest_sha256=AUTHENTIC_ARCHITECTURE_MANIFEST_SHA256,
            code_version=_git_revision(), architecture="same-z-16d-to-six-object-next-position-v1",
        ),
        decoded_image=svg_bytes, decoded_image_media_type="image/svg+xml", answer=answer,
    )
    sealed = seal_learned_latent_result(
        evidence, source="cached", manifest_sha256=AUTHENTIC_ARCHITECTURE_MANIFEST_SHA256,
        run_id=f"authentic-stage-{stage.lower()}-88173", code_version=_git_revision(), checkpoint_id=decoder_artifact_sha,
    )
    (output_root / "learned-latent-evidence.json").write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
    (output_root / "sealed-result.json").write_text(json.dumps(sealed, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "schema_version": "jump.track-h-authentic-world-result/v1",
        "stage": stage, "status": "completed", "dataset_sha256": dataset_sha,
        "generator_schema": AUTHENTIC_SCHEMA_VERSION, "heldout_law_family": list(HOLDOUT_LAW_FAMILY),
        "initial_train_loss": first_loss, "final_train_loss": float(loss.detach().cpu()),
        "validation": validation, "heldout_test": test,
        "encoder_weights_sha256": encoder_artifact_sha, "decoder_weights_sha256": decoder_artifact_sha,
        "encoder_state_sha256": encoder_hash, "decoder_state_sha256": decoder_hash,
        "model_config_sha256": sha256_json(config), "model_config": config,
        "weights_frozen_before_evaluation": True, "z_dimension": LATENT_DIM,
        "training_seconds": training_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "image_source": "learned_decoder_prediction", "mechanistic_evidence": False,
        "architecture_manifest_sha256": AUTHENTIC_ARCHITECTURE_MANIFEST_SHA256,
        "learned_latent_evidence_sha256": hashlib.sha256((output_root / "learned-latent-evidence.json").read_bytes()).hexdigest(),
    }
    (output_root / "world-config.json").write_text(json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n")
    evidence_result = write_track_h_task_evidence(
        output_root,
        metrics=[
            {"name": "initial_train_loss", "value": first_loss},
            {"name": "final_train_loss", "value": float(loss.detach().cpu())},
            {"name": "validation_future_position_nrmse", "value": validation["future_position_nrmse"]},
            {"name": "heldout_future_position_nrmse", "value": test["future_position_nrmse"]},
        ],
        terminal=result,
        experiment_spec=plan,
        track_h={"stage": stage, "mechanistic_evidence": False},
    )
    return {**result, "experiment_id": plan["experiment_id"], "experiment_spec_sha256": experiment_spec_sha256, "task_evidence": evidence_result, "artifact_root": str(output_root)}


def bind_source_observation(*, source_record: dict[str, Any], observation_bytes: bytes, source_world_id: str) -> dict[str, str]:
    binding = source_record["observation_binding"]
    observed_sha = hashlib.sha256(observation_bytes).hexdigest()
    if source_world_id != binding["source_world_id"] or observed_sha != binding["source_observation_sha256"]:
        raise ValueError("observation artifact is not bound to the latent source world")
    return {"source_world_id": source_world_id, "source_observation_sha256": observed_sha}


def authentic_architecture_manifest() -> dict[str, Any]:
    return {
        "schema_version": "jump.track-h-authentic-architecture/v1",
        "experiment_id": "track-h-authentic-learned-latent-88173",
        "claim_label": "Track H learned-latent engineering and swap demonstration; non-confirmatory, not causal or mechanistic evidence",
        "shared_contracts": ["jump.learned-latent-evidence/v1", "jump.sealed-result/v1", "jump.tensor-preimage/v1"],
        "generator": {
            "schema": AUTHENTIC_SCHEMA_VERSION, "root_seed": 88173,
            "independent_components": ["same_sign", "different_sign", "exponent", "hidden_partition", "appearance", "initial_state", "record_order"],
            "component_domain_separation": list(COMPONENT_DOMAINS),
            "splits": "world-seed-disjoint; (repel,repel,2) held out from all train/validation fitting",
            "ordering": "independent shuffled order; no index/identifier in encoder input",
        },
        "encoder": {
            "input_schema": OBSERVATION_SCHEMA_VERSION, "allowed_features": ["position_x", "position_y", "velocity_x", "velocity_y"],
            "evidence_shape": [EVIDENCE_FRAMES, 6, 4], "architecture": "MLP 96->64->16 GELU", "latent_dim": LATENT_DIM,
            "forbidden": sorted(FORBIDDEN_ENCODER_FIELDS),
        },
        "world_training": {
            "objective": "predict next visible positions from observation-only frames plus 1e-4 z L2 bottleneck regularization",
            "label_supervision": "none; partition, pair relations, laws, adequacy, forces, and target-derived fields cannot enter objective",
            "posthoc_probe_protocol": "after encoder freeze, train probes on train z using grouped nested CV by world seed; report ID validation separately from pure held-out-law-family test",
        },
        "decoder": {"architecture": "same-z MLP 16->64->12", "output": "six predicted future 2-D positions", "renderer": "deterministic SVG of decoder prediction, visibly labeled PREDICTED FROM LEARNED z"},
        "tensor_contract": {
            "api": "jump_contracts.tensor_bytes_sha256", "preimage_schema_version": "jump.tensor-preimage/v1",
            "dtype": "float32-le", "shape": [16], "order": "C", "byte_length": 64,
            "known_vector": {"raw": "bytes(range(64))", "sha256": "72a507373e1d8b984f28cd6c2258a5a09ef5fb24d3b39122c065cf046db49d36"},
            "consumer_policy": "independently materialize raw bytes at encoder output, decoder input, and injection input; require exact byte/hash equality",
        },
        "gemma_injection": {
            "base": "google/gemma-4-12B-it", "base_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "base_frozen": True,
            "site": "first non-special question-token residual immediately after input embeddings", "projector": "Linear(16,hidden_size,bias=False)", "gate": "scalar tanh initialized exactly zero",
            "prompt_policy": "E/no-z token IDs and counts identical; z and functions of z absent from text", "encoder_decoder_frozen": True,
            "conditions": ["own-z", "donor-z", "scrambled-z", "wrong-world-z", "random-matched-norm", "no-z-same-parameter", "C-prime-simulator-as-text"],
        },
        "swap": {
            "canonical_pair": "same initial visible prefix, nuisance, prompt tokens, candidate/correct law; different hidden partition and noncoincident future consequence",
            "directions": ["A_to_B", "B_to_A"], "donor_policy": "inject donor raw z bytes without re-encoding or recipient normalization",
        },
        "stages": {
            "A": {"resource": "cpu", "train": 8, "validation": 2, "test": 2, "steps": 400},
            "B": {"resource": "H100", "train": 128, "validation": 32, "heldout_family_test": 32, "steps": 200, "timeout_seconds": 1800, "max_attempts": 1, "forecast_usd": 1.9746, "hard_ceiling_usd": 10.0},
            "C": {"manifest_schema": "jump.track-h-authentic-stage-c-manifest/v1", "source": "jump_benchmark.authentic_stage_c.stage_c_manifest"},
            "D": {"resource": "H100", "gpu_count": 1, "max_containers": 1, "max_inputs": 1, "timeout_seconds": 3600, "max_attempts": 1, "forecast_usd": 3.9492, "hard_ceiling_usd": 20.0},
        },
        "gates": {
            "B": "finite loss, loss reduction, observation leakage false, shared tensor/evidence validation, artifact hashes verified",
            "C": "held-out future-position NRMSE and frozen-encoder post-hoc relation AUC reported; no G2 claim unless full three-seed threshold passes",
            "D": "two literal donor directions and all controls execute with exact scoring/provenance; no causal claim",
        },
        "mechanistic_evidence": False,
    }


AUTHENTIC_ARCHITECTURE_MANIFEST_SHA256 = sha256_json(authentic_architecture_manifest())
