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
    changes = {
        "none": "Keep the setup as shown",
        "change_force_law": "Change how the dots push or pull",
        "swap_hidden_types": "Swap the hidden groups",
        "swap_learned_latent": "Swap what the AI learned between scenes",
    }
    summaries = {
        "future-prediction": "Predict where the dots move next.",
        "hidden-law-discovery": "Find the hidden grouping and push-or-pull rule.",
        "falsified-prior": "Check whether the old movement rule is wrong.",
        "world-swap": "Compare two scenes after swapping their hidden setup.",
    }
    return (
        '<section class="sheet plan-sheet">'
        '<p class="eyebrow">Here’s the experiment I understood</p>'
        f'<h2>{escape(summaries[plan["template_id"]])}</h2>'
        '<dl class="plan-grid">'
        f'<div><dt>Dots</dt><dd>{plan["object_count"]} moving dots</dd></div>'
        f'<div><dt>Watch</dt><dd>{plan["observed_steps"]} moments of movement</dd></div>'
        f'<div><dt>Change</dt><dd>{escape(changes[intervention["kind"]])}</dd></div>'
        f'<div><dt>Guess</dt><dd>Where they move next</dd></div>'
        '</dl><p class="scope-note">Your request becomes this safe, fixed plan before anything runs.</p></section>'
    )


def correctness_html(correctness: dict[str, Any]) -> str:
    def verdict(value: bool) -> tuple[str, str]:
        return ("Yes", "good") if value else ("No", "bad")

    format_text, format_tone = verdict(correctness["format_valid"])
    exact_text, exact_tone = verdict(correctness["exact_correct"])
    if correctness["format_valid"] and not correctness["exact_correct"]:
        summary = (
            '<p class="score-summary">The answer was formatted correctly, but the model '
            'got the hidden rule wrong.</p>'
        )
    elif correctness["exact_correct"]:
        summary = '<p class="score-summary">The model got the hidden rule right.</p>'
    else:
        summary = '<p class="score-summary">The answer could not be checked because its format was wrong.</p>'
    return (
        summary
        +
        '<div class="score-pair">'
        f'<div><span>Could we read the answer?</span><strong class="{format_tone}">{format_text}</strong></div>'
        f'<div><span>Did it find the hidden rule?</span><strong class="{exact_tone}">{exact_text}</strong></div>'
        '</div>'
        '<dl class="subscores">'
        f'<div><dt>Hidden groups</dt><dd>{"right" if correctness["partition_correct"] else "wrong"}</dd></div>'
        f'<div><dt>Push-or-pull rule</dt><dd>{"right" if correctness["law_correct"] else "wrong"}</dd></div>'
        f'<div><dt>Did the old rule fail?</dt><dd>{"right" if correctness["adequacy_correct"] else "wrong"}</dd></div>'
        '</dl>'
    )


def result_sections(run: dict[str, Any], *, backend_label: str) -> tuple[str, str, str, str, str]:
    checked = verified_result(run)
    presentation = checked["presentation"]
    image = base64.b64encode(checked["image_bytes"]).decode("ascii")
    world = (
        '<section class="result-card"><p class="step-number">01</p><h2>World built</h2>'
        '<p>Six dots were placed in motion for the confirmed experiment.</p>'
        f'<img src="data:image/svg+xml;base64,{image}" alt="The AI prediction for where the dots move next"/></section>'
    )
    prediction = (
        '<section class="result-card"><p class="step-number">02</p><h2>Model prediction</h2>'
        '<p>The AI returned its guess for the hidden groups and movement rule.</p></section>'
    )
    changed = (
        '<section class="result-card"><p class="step-number">03</p><h2>What changed</h2>'
        '<p>The confirmed change was applied before the AI made its prediction.</p></section>'
    )
    correct = (
        '<section class="result-card"><p class="step-number">04</p><h2>Was it correct?</h2>'
        + correctness_html(presentation["correctness"])
        + '</section>'
    )
    evidence = (
        '<section class="result-card evidence-card"><p class="step-number">05</p><h2>Evidence</h2>'
        '<p>The prediction and picture came from the same saved experiment state, and '
        'their checks passed.</p></section>'
    )
    return world, prediction, changed, correct, evidence
