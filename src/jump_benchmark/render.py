"""Deterministic SVG renderer for observation frames (hidden types stay hidden)."""

from __future__ import annotations

import html
import math
from typing import Any

WIDTH = 512
HEIGHT = 512


def _point(position: list[float], bounds: float) -> tuple[float, float]:
    x = (position[0] + bounds) / (2 * bounds) * (WIDTH - 64) + 32
    y = HEIGHT - ((position[1] + bounds) / (2 * bounds) * (HEIGHT - 64) + 32)
    return x, y


def _shape(name: str, x: float, y: float, color: str) -> str:
    common = f'fill="{html.escape(color)}" stroke="#111827" stroke-width="3"'
    if name == "circle":
        return f'<circle cx="{x:.3f}" cy="{y:.3f}" r="18" {common}/>'
    if name == "square":
        return f'<rect x="{x-17:.3f}" y="{y-17:.3f}" width="34" height="34" rx="3" {common}/>'
    if name == "triangle":
        return f'<polygon points="{x:.3f},{y-20:.3f} {x-19:.3f},{y+17:.3f} {x+19:.3f},{y+17:.3f}" {common}/>'
    if name == "diamond":
        return f'<polygon points="{x:.3f},{y-21:.3f} {x-21:.3f},{y:.3f} {x:.3f},{y+21:.3f} {x+21:.3f},{y:.3f}" {common}/>'
    if name == "hexagon":
        points = " ".join(
            f"{x+20*math.cos(k*math.pi/3):.3f},{y+20*math.sin(k*math.pi/3):.3f}"
            for k in range(6)
        )
        return f'<polygon points="{points}" {common}/>'
    # A visually distinct, deterministic star; shape names are generator-owned.
    points = []
    for index in range(10):
        radius = 21 if index % 2 == 0 else 9
        angle = -math.pi / 2 + index * math.pi / 5
        points.append(f"{x+radius*math.cos(angle):.3f},{y+radius*math.sin(angle):.3f}")
    return f'<polygon points="{" ".join(points)}" {common}/>'


def render_svg(episode: dict[str, Any], *, observation_index: int = -1) -> str:
    observations = episode["observations"]
    frame = observations[observation_index]
    bounds = float(episode["simulator"]["bounds"])
    appearance = {item["object_id"]: item for item in episode["appearance"]}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="512" height="512" fill="#f8fafc"/>',
        '<rect x="24" y="24" width="464" height="464" rx="12" fill="none" stroke="#94a3b8" stroke-width="2"/>',
        f'<text x="32" y="48" font-family="monospace" font-size="14" fill="#334155">step {frame["step"]}</text>',
    ]
    for index, object_id in enumerate(episode["object_ids"]):
        item = appearance[object_id]
        x, y = _point(frame["positions"][index], bounds)
        elements.append(_shape(item["shape"], x, y, item["color"]))
        elements.append(
            f'<text x="{x:.3f}" y="{y+4:.3f}" text-anchor="middle" font-family="monospace" font-size="11" fill="#ffffff">{html.escape(object_id)}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"
