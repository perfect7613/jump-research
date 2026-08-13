"""Plain presentation of validated visual thought experiments."""

from __future__ import annotations

from html import escape
from typing import Any, Mapping


EXAMPLES = (
    "What if twelve particles start with x and y velocity 0, then attraction becomes repulsion halfway through?",
    "How does an infection spread on a ring network if transmission is stopped halfway through?",
    "What happens to predator and prey populations after available food drops?",
)


def spec_html(spec: Mapping[str, Any]) -> str:
    baseline = next(c for c in spec["conditions"] if c["kind"] == "baseline")
    counter = next(c for c in spec["conditions"] if c["kind"] == "counterfactual")
    rules = ", ".join(rule["op"].replace("_", " ") for rule in spec["dynamics"]["rules"])
    return (
        '<section class="visual-summary"><p class="eyebrow">Review the thought experiment</p>'
        f'<h2>{escape(spec["question"])}</h2>'
        f'<p><strong>Hypothesis:</strong> {escape(spec["hypothesis"])}</p>'
        f'<p><strong>Compare:</strong> {escape(baseline["label"])} versus {escape(counter["label"])}</p>'
        f'<p><strong>Rules:</strong> {escape(rules)}</p>'
        f'<p><strong>Length:</strong> {spec["schedule"]["duration_steps"]} steps; '
        f'{spec["schedule"]["repetitions"]} repetition(s).</p>'
        '<p class="limitation">This is a bounded deterministic simulator. It does not establish a real-world causal effect.</p></section>'
    )


def result_html(spec: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    prediction = run["execution"]["prediction"]
    comparisons = "; ".join(
        f"{item['measurement_id']}: {item['baseline_final']:.3g} → {item['counterfactual_final']:.3g} "
        f"(change {item['difference']:+.3g})" for item in run["comparisons"]
    )
    return (
        '<section class="visual-summary completed-summary"><p class="eyebrow">Result</p>'
        f'<h2>{escape(spec["question"])}</h2>'
        f'<p><strong>Prediction:</strong> {escape(prediction["summary"])}</p>'
        f'<p><strong>Measured result:</strong> {escape(comparisons)}</p>'
        f'<p><strong>Interpretation ({escape(run["revision"]["disposition"])}):</strong> '
        f'{escape(run["revision"]["interpretation"])}</p>'
        '<p class="limitation">The frames below are deterministic simulation states, not a learned-latent reconstruction.</p></section>'
    )


def visual_html(spec: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    labels = {item["id"]: item["label"] for item in spec["conditions"]}
    panels = "".join(_condition_svg(item, labels[item["condition_id"]], spec["world"]["bounds"]) for item in run["conditions"][:2])
    return '<section class="simulation-frames"><p class="eyebrow">Deterministic simulation frames</p><div class="visual-pair">' + panels + '</div></section>'


def chart_html(spec: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    wanted = set(spec["visualization"]["chart_measurement_ids"])
    series = []
    for condition in run["conditions"][:2]:
        for item in condition["series"]:
            if item["measurement_id"] in wanted:
                series.append((condition["condition_id"], item["measurement_id"], item["values"]))
    if not series:
        return ""
    all_values = [point["value"] for _, _, values in series for point in values]
    low, high = min(all_values), max(all_values)
    span = high - low or 1.0
    colors = ("#244fb3", "#b53224")
    paths = []
    legend = []
    for index, (condition, measurement, values) in enumerate(series[:4]):
        maximum_step = max(point["step"] for point in values) or 1
        points = " ".join(
            f"{30 + 440 * point['step'] / maximum_step:.1f},{170 - 135 * (point['value'] - low) / span:.1f}"
            for point in values
        )
        color = colors[index % 2]
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        legend.append(f'<span style="color:{color}">{escape(condition)} · {escape(measurement)}</span>')
    return (
        '<section class="comparison-chart"><p class="eyebrow">Measured comparison over time</p>'
        '<svg viewBox="0 0 500 190" role="img" aria-label="Measurement comparison chart">'
        '<line x1="30" y1="170" x2="470" y2="170" stroke="#777"/>' + "".join(paths) + '</svg>'
        '<div class="chart-legend">' + "".join(legend) + '</div></section>'
    )


def _condition_svg(condition: Mapping[str, Any], label: str, bounds: Mapping[str, Any]) -> str:
    frames = condition["frames"]
    points_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for frame in frames:
        for point in frame["points"]:
            points_by_id.setdefault(point["entity_id"], []).append(point)
    duration = max(3, min(12, len(frames) * 0.4))
    shapes = []
    for track in points_by_id.values():
        first = track[0]
        coords = ";".join(
            f"{20 + 360 * point['x'] / bounds['width']:.2f} {20 + 220 * point['y'] / bounds['height']:.2f}"
            for point in track
        )
        shape = _shape(first)
        shapes.append(
            f'<g>{shape}<animateTransform attributeName="transform" type="translate" values="{coords}" '
            f'dur="{duration}s" repeatCount="indefinite" calcMode="linear"/></g>'
        )
    return (
        f'<figure><figcaption>{escape(label)}</figcaption><svg viewBox="0 0 400 260" role="img" '
        f'aria-label="{escape(label)} deterministic simulation"><rect width="400" height="260" fill="#fffdf8" stroke="#c9c1b1"/>'
        + "".join(shapes) + '</svg></figure>'
    )


def _shape(point: Mapping[str, Any]) -> str:
    size, color = point["size"], point["color"]
    if point["shape"] == "square":
        return f'<rect x="{-size}" y="{-size}" width="{size*2}" height="{size*2}" fill="{color}"/>'
    if point["shape"] == "triangle":
        return f'<path d="M 0 {-size} L {size} {size} L {-size} {size} Z" fill="{color}"/>'
    return f'<circle r="{size}" fill="{color}"/>'
