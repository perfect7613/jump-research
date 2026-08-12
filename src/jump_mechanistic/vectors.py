"""Tiny deterministic vector utilities; production callers may pass tensor-like values."""

from __future__ import annotations

import math
from collections.abc import Iterable


Vector = list[float]


def as_vector(value: object) -> Vector:
    """Detach/copy a tensor-like value into a flat CPU list without importing torch."""
    current = value
    for method in ("detach", "float", "cpu"):
        fn = getattr(current, method, None)
        if callable(fn):
            current = fn()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        current = tolist()

    out: Vector = []

    def visit(item: object) -> None:
        if isinstance(item, (str, bytes)):
            raise TypeError("activations must be numeric")
        if isinstance(item, Iterable):
            for child in item:
                visit(child)
        else:
            number = float(item)  # type: ignore[arg-type]
            if not math.isfinite(number):
                raise ValueError("activations must be finite")
            out.append(number)

    visit(current)
    if not out:
        raise ValueError("activation vector cannot be empty")
    return out


def dot(a: Vector, b: Vector) -> float:
    _same_size(a, b)
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def add(a: Vector, b: Vector) -> Vector:
    _same_size(a, b)
    return [x + y for x, y in zip(a, b)]


def sub(a: Vector, b: Vector) -> Vector:
    _same_size(a, b)
    return [x - y for x, y in zip(a, b)]


def scale(a: Vector, factor: float) -> Vector:
    return [factor * x for x in a]


def unit(a: Vector) -> Vector:
    length = norm(a)
    if length <= 1e-12:
        raise ValueError("direction must have nonzero norm")
    return scale(a, 1.0 / length)


def mean(rows: list[Vector]) -> Vector:
    if not rows:
        raise ValueError("cannot average empty vectors")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("vectors must share a nonzero width")
    return [sum(row[j] for row in rows) / len(rows) for j in range(width)]


def _same_size(a: Vector, b: Vector) -> None:
    if len(a) != len(b):
        raise ValueError(f"vector dimensions differ: {len(a)} != {len(b)}")
