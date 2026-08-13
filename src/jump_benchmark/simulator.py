"""Deterministic six-object 2D hidden-type simulator and data generator.

The simulator is intentionally CPU-only and dependency-free. Hidden types,
appearance, initial state, and law selection use domain-separated random
streams, so nuisance appearance cannot become a type-label encoding by
construction. All serialized floats are rounded at the artifact boundary.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

OBJECT_COUNT = 6
SCHEMA_VERSION = "jump.track-h-dataset/v1"
SIGNS = ("attract", "repel")
SHAPES = ("circle", "square", "triangle", "diamond", "hexagon", "star")
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def derive_seed(seed: int, domain: str) -> int:
    payload = f"jump-track-h-v1:{seed}:{domain}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rounded(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded


def _vector(values: Iterable[float]) -> list[float]:
    return [_rounded(value) for value in values]


@dataclass(frozen=True)
class Law:
    same: str
    different: str
    exponent: int

    def __post_init__(self) -> None:
        if self.same not in SIGNS or self.different not in SIGNS:
            raise ValueError("law signs must be attract or repel")
        if isinstance(self.exponent, bool) or not isinstance(self.exponent, int) or self.exponent <= 0:
            raise ValueError("law exponent must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {"same": self.same, "different": self.different, "exponent": self.exponent}


@dataclass(frozen=True)
class SimulatorConfig:
    dt: float = 0.04
    steps: int = 6
    coefficient: float = 0.18
    softening: float = 0.2
    bounds: float = 3.0

    def __post_init__(self) -> None:
        numeric = (self.dt, self.coefficient, self.softening, self.bounds)
        if any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError("simulator numeric parameters must be finite and positive")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 2 or self.steps > 100:
            raise ValueError("steps must be an integer in [2, 100]")


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    split: str
    law: Law
    adequate_prior: bool
    config: SimulatorConfig = SimulatorConfig()
    partition: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("episode seed must be an integer")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("episode split must be nonempty")
        if not isinstance(self.adequate_prior, bool):
            raise ValueError("adequate_prior must be Boolean")


@dataclass(frozen=True)
class DatasetSpec:
    seed: int
    split_counts: dict[str, int]
    exponents: tuple[int, ...] = (1, 2)
    config: SimulatorConfig = SimulatorConfig()

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("dataset seed must be an integer")
        if not self.split_counts or any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for name, count in self.split_counts.items()
        ):
            raise ValueError("split_counts must map split names to positive integers")
        if not self.exponents or len(set(self.exponents)) != len(self.exponents):
            raise ValueError("exponents must be nonempty and unique")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.exponents):
            raise ValueError("exponents must contain positive integers")


def canonical_partition(values: Iterable[int]) -> tuple[int, ...]:
    partition = tuple(values)
    if len(partition) != OBJECT_COUNT or any(isinstance(value, bool) or value not in (0, 1) for value in partition):
        raise ValueError("partition must contain exactly six binary integers")
    if len(set(partition)) != 2:
        raise ValueError("partition must be non-trivial")
    if partition[0] == 1:
        partition = tuple(1 - value for value in partition)
    return partition


def _partition(seed: int) -> tuple[int, ...]:
    # Object zero anchors the label-swap canonicalization. Masks 1..31 enumerate
    # the 31 non-trivial six-object partitions exactly once.
    mask = random.Random(derive_seed(seed, "hidden-types")).randint(1, 31)
    return (0,) + tuple((mask >> index) & 1 for index in range(5))


def _appearance(seed: int) -> list[dict[str, str]]:
    rng = random.Random(derive_seed(seed, "appearance"))
    shapes, colors = list(SHAPES), list(COLORS)
    rng.shuffle(shapes)
    rng.shuffle(colors)
    return [
        {"object_id": f"o{index}", "shape": shapes[index], "color": colors[index]}
        for index in range(OBJECT_COUNT)
    ]


def _initial_state(seed: int, bounds: float) -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(derive_seed(seed, "initial-state"))
    positions: list[list[float]] = []
    while len(positions) < OBJECT_COUNT:
        candidate = [rng.uniform(-0.72 * bounds, 0.72 * bounds), rng.uniform(-0.72 * bounds, 0.72 * bounds)]
        if all(math.dist(candidate, prior) >= 0.55 for prior in positions):
            positions.append(candidate)
    velocities = [[rng.uniform(-0.09, 0.09), rng.uniform(-0.09, 0.09)] for _ in range(OBJECT_COUNT)]
    return positions, velocities


def pairwise_forces(
    positions: list[list[float]], partition: tuple[int, ...], law: Law, *, coefficient: float, softening: float
) -> list[list[float]]:
    if len(positions) != OBJECT_COUNT or any(len(position) != 2 for position in positions):
        raise ValueError("positions must be six 2D vectors")
    partition = canonical_partition(partition)
    forces = [[0.0, 0.0] for _ in positions]
    for left in range(OBJECT_COUNT):
        for right in range(left + 1, OBJECT_COUNT):
            dx = positions[right][0] - positions[left][0]
            dy = positions[right][1] - positions[left][1]
            distance = math.sqrt(dx * dx + dy * dy + softening * softening)
            relation = "same" if partition[left] == partition[right] else "different"
            sign_name = law.same if relation == "same" else law.different
            sign = 1.0 if sign_name == "attract" else -1.0
            magnitude = coefficient / (distance ** law.exponent)
            fx, fy = sign * magnitude * dx / distance, sign * magnitude * dy / distance
            forces[left][0] += fx
            forces[left][1] += fy
            forces[right][0] -= fx
            forces[right][1] -= fy
    return forces


def _trajectory(
    positions: list[list[float]], velocities: list[list[float]], partition: tuple[int, ...], law: Law, config: SimulatorConfig
) -> list[dict[str, Any]]:
    positions = [list(value) for value in positions]
    velocities = [list(value) for value in velocities]
    frames: list[dict[str, Any]] = []
    for step in range(config.steps):
        forces = pairwise_forces(
            positions, partition, law, coefficient=config.coefficient, softening=config.softening
        )
        frames.append(
            {
                "step": step,
                "positions": [_vector(value) for value in positions],
                "velocities": [_vector(value) for value in velocities],
                "forces": [_vector(value) for value in forces],
            }
        )
        if step == config.steps - 1:
            break
        for index in range(OBJECT_COUNT):
            velocities[index][0] += config.dt * forces[index][0]
            velocities[index][1] += config.dt * forces[index][1]
            positions[index][0] += config.dt * velocities[index][0]
            positions[index][1] += config.dt * velocities[index][1]
            for axis in (0, 1):
                if positions[index][axis] < -config.bounds or positions[index][axis] > config.bounds:
                    positions[index][axis] = max(-config.bounds, min(config.bounds, positions[index][axis]))
                    velocities[index][axis] *= -1.0
    return frames


def _wrong_prior(law: Law) -> Law:
    return Law("repel" if law.same == "attract" else "attract", law.different, law.exponent)


def generate_episode(spec: EpisodeSpec) -> dict[str, Any]:
    partition = canonical_partition(spec.partition) if spec.partition is not None else _partition(spec.seed)
    appearance = _appearance(spec.seed)
    positions, velocities = _initial_state(spec.seed, spec.config.bounds)
    trajectory = _trajectory(positions, velocities, partition, spec.law, spec.config)
    prior_law = spec.law if spec.adequate_prior else _wrong_prior(spec.law)
    episode_id = hashlib.sha256(f"episode:{spec.split}:{spec.seed}".encode()).hexdigest()[:20]
    force_prediction = {
        str(frame["step"]): frame["forces"]
        for frame in trajectory[1:]
    }
    return {
        "schema_version": "jump.track-h-episode/v1",
        "episode_id": episode_id,
        "world_seed": spec.seed,
        "split": spec.split,
        "object_count": OBJECT_COUNT,
        "object_ids": [f"o{index}" for index in range(OBJECT_COUNT)],
        "appearance": appearance,
        "initial_state": trajectory[0],
        "observations": trajectory[1:],
        "prior_law": prior_law.as_dict(),
        "simulator": {
            "dt": spec.config.dt,
            "steps": spec.config.steps,
            "coefficient": spec.config.coefficient,
            "softening": spec.config.softening,
            "bounds": spec.config.bounds,
        },
        "target": {
            "partition": list(partition),
            "replacement_law": spec.law.as_dict(),
            "adequacy": spec.adequate_prior,
            "force_prediction": force_prediction,
        },
        "provenance": {
            "generator": SCHEMA_VERSION,
            "root_seed": spec.seed,
            "stream_seeds": {
                name: derive_seed(spec.seed, name)
                for name in ("hidden-types", "appearance", "initial-state")
            },
        },
    }


def _law_for(seed: int, exponents: tuple[int, ...]) -> Law:
    rng = random.Random(derive_seed(seed, "law"))
    same = rng.choice(SIGNS)
    # Generated hidden-structure episodes always have a real relation-dependent
    # sign flip. The scorer still accepts the full four-category law DSL fixed
    # by the PRD, and no-hidden-structure controls can be added explicitly later.
    different = "repel" if same == "attract" else "attract"
    return Law(same, different, rng.choice(exponents))


def generate_dataset(spec: DatasetSpec) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for split in sorted(spec.split_counts):
        for index in range(spec.split_counts[split]):
            world_seed = derive_seed(spec.seed, f"split:{split}:episode:{index}")
            if world_seed in seen_seeds:  # cryptographically implausible, but fail closed
                raise RuntimeError("world seed collision")
            seen_seeds.add(world_seed)
            episodes.append(
                generate_episode(
                    EpisodeSpec(
                        seed=world_seed,
                        split=split,
                        law=_law_for(world_seed, spec.exponents),
                        adequate_prior=index % 2 == 0,
                        config=spec.config,
                    )
                )
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_config": {
            "seed": spec.seed,
            "split_counts": {key: spec.split_counts[key] for key in sorted(spec.split_counts)},
            "exponents": list(spec.exponents),
            "simulator": {
                "dt": spec.config.dt,
                "steps": spec.config.steps,
                "coefficient": spec.config.coefficient,
                "softening": spec.config.softening,
                "bounds": spec.config.bounds,
            },
        },
        "episodes": episodes,
    }
