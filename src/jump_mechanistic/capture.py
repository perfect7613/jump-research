"""Allowlist-enforced activation capture at preregistered JUMP timepoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .vectors import Vector, as_vector


class Timepoint(str, Enum):
    T0 = "T0"  # before observation/prediction
    T1 = "T1"  # after expected-law prediction
    T2 = "T2"  # immediately after mismatch evidence
    T3 = "T3"  # hypothesis promotion / revision
    T4 = "T4"  # answer verbalization


@dataclass(frozen=True)
class CapturePolicy:
    layers: frozenset[str]
    timepoints: frozenset[Timepoint]

    @classmethod
    def from_allowlists(cls, layers: list[str], timepoints: list[str]) -> "CapturePolicy":
        if not layers:
            raise ValueError("layer_allowlist cannot be empty")
        if len(layers) != len(set(layers)):
            raise ValueError("layer_allowlist contains duplicates")
        try:
            points = frozenset(Timepoint(point) for point in timepoints)
        except ValueError as exc:
            raise ValueError("timepoints must be among T0, T1, T2, T3, T4") from exc
        if not points:
            raise ValueError("timepoint_allowlist cannot be empty")
        return cls(frozenset(layers), points)

    def require(self, layer: str, timepoint: str | Timepoint) -> Timepoint:
        try:
            point = Timepoint(timepoint)
        except ValueError as exc:
            raise ValueError(f"unknown timepoint: {timepoint}") from exc
        if layer not in self.layers:
            raise PermissionError(f"layer is not preregistered: {layer}")
        if point not in self.timepoints:
            raise PermissionError(f"timepoint is not preregistered: {point.value}")
        return point


@dataclass(frozen=True)
class ActivationRecord:
    episode_id: str
    checkpoint_id: str
    layer: str
    timepoint: Timepoint
    values: Vector
    labels: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "checkpoint_id": self.checkpoint_id,
            "layer": self.layer,
            "timepoint": self.timepoint.value,
            "values": self.values,
            "labels": self.labels,
        }


class ActivationCapture:
    """Explicit capture sink suitable for model hooks or pre-extracted activations.

    A caller must label each hook invocation with a semantic T0--T4 point. No
    unregistered layer or timepoint can be silently recorded.
    """

    def __init__(self, policy: CapturePolicy) -> None:
        self.policy = policy
        self._records: list[ActivationRecord] = []
        self._keys: set[tuple[str, str, str, Timepoint]] = set()

    @property
    def records(self) -> tuple[ActivationRecord, ...]:
        return tuple(self._records)

    def capture(
        self,
        *,
        episode_id: str,
        checkpoint_id: str,
        layer: str,
        timepoint: str | Timepoint,
        activation: object,
        labels: dict[str, Any] | None = None,
    ) -> ActivationRecord:
        point = self.policy.require(layer, timepoint)
        key = (episode_id, checkpoint_id, layer, point)
        if key in self._keys:
            raise ValueError(f"duplicate capture: {key}")
        record = ActivationRecord(
            episode_id=episode_id,
            checkpoint_id=checkpoint_id,
            layer=layer,
            timepoint=point,
            values=as_vector(activation),
            labels=dict(labels or {}),
        )
        self._records.append(record)
        self._keys.add(key)
        return record

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        return target

    def make_hook(
        self,
        *,
        episode_id: str,
        checkpoint_id: str,
        layer: str,
        timepoint: str | Timepoint,
        labels: dict[str, Any] | None = None,
        selector: Callable[[object], object] | None = None,
    ) -> Callable[[object, object, object], None]:
        """Return a PyTorch-style forward hook without importing a framework.

        ``selector`` is explicit so production code can preregister whether it
        stores a last-token state, pooled state, or another fixed slice.
        """
        self.policy.require(layer, timepoint)  # fail before hook registration

        def hook(_module: object, _inputs: object, output: object) -> None:
            activation = selector(output) if selector else output
            self.capture(
                episode_id=episode_id,
                checkpoint_id=checkpoint_id,
                layer=layer,
                timepoint=timepoint,
                activation=activation,
                labels=labels,
            )

        return hook
