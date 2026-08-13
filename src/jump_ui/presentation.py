"""Plain-language, testable HTML fragments for the JUMP Space."""

from __future__ import annotations

import base64
from html import escape
import json
from typing import Any

from .flow import verified_result


def plan_html(planned_run: dict[str, Any]) -> str:
    plan = planned_run["plan"]
    intervention = plan["intervention"]
    change = intervention["kind"].replace("_", " ")
    return (
        '<section class="sheet plan-sheet">'
        '<p class="eyebrow">Here’s the experiment I understood</p>'
        f'<h2>{escape(plan["summary"])}</h2>'
        '<dl class="plan-grid">'
        f'<div><dt>World</dt><dd>{plan["object_count"]} moving objects</dd></div>'
        f'<div><dt>Observe</dt><dd>{plan["observed_steps"]} motion steps</dd></div>'
        f'<div><dt>Change</dt><dd>{escape(change)}</dd></div>'
        f'<div><dt>Predict</dt><dd>{plan["prediction_horizon"]} step ahead</dd></div>'
        '</dl><p class="scope-note">This plan stays inside the bounded six-object simulator. '
        'Your original wording is not sent to the model.</p></section>'
    )


def correctness_html(correctness: dict[str, Any]) -> str:
    def verdict(value: bool) -> tuple[str, str]:
        return ("Yes", "good") if value else ("No", "bad")

    format_text, format_tone = verdict(correctness["format_valid"])
    exact_text, exact_tone = verdict(correctness["exact_correct"])
    return (
        '<div class="score-pair">'
        f'<div><span>Answer format</span><strong class="{format_tone}">{format_text}</strong></div>'
        f'<div><span>Exact answer</span><strong class="{exact_tone}">{exact_text}</strong></div>'
        '</div>'
        '<dl class="subscores">'
        f'<div><dt>Grouping</dt><dd>{"correct" if correctness["partition_correct"] else "incorrect"}</dd></div>'
        f'<div><dt>Force rule</dt><dd>{"correct" if correctness["law_correct"] else "incorrect"}</dd></div>'
        f'<div><dt>Old-rule decision</dt><dd>{"correct" if correctness["adequacy_correct"] else "incorrect"}</dd></div>'
        f'<div><dt>Force score</dt><dd>{escape(str(correctness["force_score"] if correctness["force_score"] is not None else "not scored"))}</dd></div>'
        f'</dl><p class="quiet">{escape(correctness["notes"])}</p>'
    )


def result_sections(run: dict[str, Any], *, backend_label: str) -> tuple[str, str, str, str, str]:
    checked = verified_result(run)
    presentation = checked["presentation"]
    image = base64.b64encode(checked["image_bytes"]).decode("ascii")
    world = (
        '<section class="result-card"><p class="step-number">01</p><h2>World built</h2>'
        f'<p>{escape(presentation["world_built"])}</p>'
        f'<img src="data:image/svg+xml;base64,{image}" alt="Prediction produced by the learned-z decoder"/>'
        f'<p class="fixture-label">{escape(backend_label)}</p></section>'
    )
    prediction = (
        '<section class="result-card"><p class="step-number">02</p><h2>Model prediction</h2>'
        f'<pre>{escape(json.dumps(presentation["model_prediction"], indent=2, sort_keys=True))}</pre></section>'
    )
    changed = (
        '<section class="result-card"><p class="step-number">03</p><h2>What changed</h2>'
        f'<p>{escape(presentation["what_changed"])}</p></section>'
    )
    correct = (
        '<section class="result-card"><p class="step-number">04</p><h2>Was it correct?</h2>'
        + correctness_html(presentation["correctness"])
        + '</section>'
    )
    evidence = (
        '<section class="result-card evidence-card"><p class="step-number">05</p><h2>Evidence</h2>'
        '<p>The same learned world state was used for the model input and the picture. '
        'The sealed answer and image bytes passed their checks.</p>'
        '<p class="quiet">This verifies engineering provenance only. Stage D null: '
        'own-z equaled no-z and the donor shift was zero.</p></section>'
    )
    return world, prediction, changed, correct, evidence
