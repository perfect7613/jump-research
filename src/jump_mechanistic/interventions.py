"""Matched latent swaps, targeted interventions, and preregistered controls."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable

from .vectors import Vector, add, dot, norm, scale, sub, unit


@dataclass(frozen=True)
class MatchedWorldPair:
    pair_id: str
    world_a_id: str
    world_b_id: str
    latent_a: Vector
    latent_b: Vector
    nuisance_a: dict[str, object]
    nuisance_b: dict[str, object]
    target_a: object
    target_b: object

    def validate(self, *, tolerance: float = 0.0) -> None:
        if self.world_a_id == self.world_b_id:
            raise ValueError("a matched pair needs distinct world ids")
        if self.target_a == self.target_b:
            raise ValueError("World A/B must differ in the target latent property")
        if set(self.nuisance_a) != set(self.nuisance_b):
            raise ValueError("World A/B nuisance keys differ")
        for key in self.nuisance_a:
            left, right = self.nuisance_a[key], self.nuisance_b[key]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if abs(float(left) - float(right)) > tolerance:
                    raise ValueError(f"unmatched nuisance field: {key}")
            elif left != right:
                raise ValueError(f"unmatched nuisance field: {key}")
        if len(self.latent_a) != len(self.latent_b):
            raise ValueError("paired latent dimensions differ")


def evaluate_latent_swap(
    pair: MatchedWorldPair,
    readout: Callable[[str, Vector], float],
) -> dict[str, float | str]:
    """Compare aligned and crossed latents using the same observation/world input."""
    pair.validate()
    aa = float(readout(pair.world_a_id, pair.latent_a))
    ab = float(readout(pair.world_a_id, pair.latent_b))
    bb = float(readout(pair.world_b_id, pair.latent_b))
    ba = float(readout(pair.world_b_id, pair.latent_a))
    if isinstance(pair.target_a, (int, float)) and isinstance(pair.target_b, (int, float)):
        direction = 1.0 if float(pair.target_b) > float(pair.target_a) else -1.0
        directional_effect = ((ab - aa) * direction + (ba - bb) * -direction) / 2.0
    else:
        raise ValueError("directional swap scoring requires numeric World A/B targets")
    return {
        "pair_id": pair.pair_id,
        "aligned_mean": (aa + bb) / 2.0,
        "swapped_mean": (ab + ba) / 2.0,
        "swap_effect": directional_effect,
        "a_to_b_effect": ab - aa,
        "b_to_a_effect": ba - bb,
    }


def ablate(vector: Vector, direction: Vector) -> Vector:
    axis = unit(direction)
    return sub(vector, scale(axis, dot(vector, axis)))


def inject(vector: Vector, direction: Vector, *, magnitude: float) -> Vector:
    return add(vector, scale(unit(direction), magnitude))


def matched_norm_control(direction: Vector, *, seed: int, label: str = "matched_norm") -> Vector:
    """A deterministic random direction with exactly the target direction's norm."""
    target_norm = norm(direction)
    if target_norm <= 1e-12:
        raise ValueError("target direction must have nonzero norm")
    rng = random.Random(_stable_seed(seed, label))
    candidate = [rng.gauss(0.0, 1.0) for _ in direction]
    return scale(unit(candidate), target_norm)


def orthogonal_control(direction: Vector, *, seed: int) -> Vector:
    target_norm = norm(direction)
    if len(direction) < 2 or target_norm <= 1e-12:
        raise ValueError("orthogonal controls require a nonzero direction of dimension >= 2")
    candidate = matched_norm_control(direction, seed=seed, label="orthogonal")
    axis = unit(direction)
    orthogonal = sub(candidate, scale(axis, dot(candidate, axis)))
    if norm(orthogonal) <= 1e-10:  # deterministic basis fallback
        basis = [0.0] * len(direction)
        basis[min(range(len(direction)), key=lambda i: abs(axis[i]))] = 1.0
        orthogonal = sub(basis, scale(axis, dot(basis, axis)))
    return scale(unit(orthogonal), target_norm)


def generic_error_control(error_activations: list[Vector], clean_activations: list[Vector]) -> Vector:
    """Mean generic-error contrast, matched later at injection magnitude."""
    if len(error_activations) != len(clean_activations) or not error_activations:
        raise ValueError("generic-error contrasts require matched nonempty rows")
    width = len(error_activations[0])
    if any(len(row) != width for row in error_activations + clean_activations):
        raise ValueError("activation dimensions differ")
    return [
        sum(error[j] - clean[j] for error, clean in zip(error_activations, clean_activations))
        / len(error_activations)
        for j in range(width)
    ]


def build_control_directions(
    target: Vector,
    *,
    seed: int,
    generic_error: Vector,
) -> dict[str, Vector]:
    target_norm = norm(target)
    if norm(generic_error) <= 1e-12:
        raise ValueError("generic error direction must have nonzero norm")
    return {
        "target": target,
        "matched_norm": matched_norm_control(target, seed=seed),
        "orthogonal": orthogonal_control(target, seed=seed),
        "generic_error": scale(unit(generic_error), target_norm),
    }


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
