"""Contract-independent copy and layout for the general JUMP product shell.

No request, plan, or result fields are interpreted here. The canonical shared
ExperimentPlan/ExperimentRun validators own those mappings once available.
"""

from __future__ import annotations

from html import escape
from typing import Mapping

HEADLINE = "Test an idea. See what the simulation says."
SUBCOPY = (
    "Describe a question in plain English. JUMP turns it into a simple experiment, "
    "predicts the outcome, runs the simulation, and compares the two."
)
QUESTION = "What do you want to test?"
EXAMPLES = (
    "Does changing traffic-light timing reduce a traffic jam?",
    "How often does switching doors win in the Monty Hall problem?",
    "What happens to an epidemic if contact rates fall by half?",
)
SCOPE = (
    "Computational simulations only · no code, links, files, or real-world actions · "
    "600 characters max"
)
PLAN_LABELS = ("Question", "Hypothesis", "Change", "Measure", "Assumptions", "Repetitions")
RESULT_LABELS = (
    "Prediction",
    "Simulation",
    "Was the prediction right?",
    "What the model changed its mind about",
    "Evidence",
)


def hero_html() -> str:
    return (
        '<header class="jump-shell">'
        '<p class="jump-kicker">JUMP · computational thought experiments</p>'
        f'<h1 class="jump-title">{escape(HEADLINE)}</h1>'
        f'<p class="jump-deck">{escape(SUBCOPY)}</p>'
        '<p class="scope-note">JUMP supports bounded simulation questions. It does not run '
        'user code, open links or files, or take actions in the real world.</p>'
        '</header>'
    )


def particle_research_card() -> str:
    return (
        '<aside class="research-card">'
        '<p class="eyebrow">Original research demo</p>'
        '<h2>Can an AI revise a hidden physics rule?</h2>'
        '<p>Explore the six-particle benchmark that motivated JUMP: hidden groups, '
        'attraction and repulsion, and a model asked to replace a failed rule.</p>'
        '<a href="https://github.com/perfect7613/jump-research" target="_blank" '
        'rel="noopener noreferrer">Read the research →</a>'
        '</aside>'
    )


def plan_shell(rows: Mapping[str, str]) -> str:
    """Render only already-validated, presentation-safe row strings."""
    if tuple(rows) != PLAN_LABELS or any(not isinstance(value, str) or not value for value in rows.values()):
        raise ValueError("general plan presentation requires the six exact nonempty rows")
    content = "".join(
        f'<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>'
        for label, value in rows.items()
    )
    return (
        '<section class="sheet plan-sheet">'
        '<p class="eyebrow">Here’s the experiment I understood</p>'
        f'<dl class="general-plan-grid">{content}</dl>'
        '<p class="scope-note">Review this plan before the simulation runs.</p>'
        '</section>'
    )


def result_shell(sections: Mapping[str, str]) -> tuple[str, ...]:
    """Render only already-validated result summaries in the fixed product order."""
    if tuple(sections) != RESULT_LABELS or any(
        not isinstance(value, str) or not value for value in sections.values()
    ):
        raise ValueError("general result presentation requires the five exact nonempty sections")
    return tuple(
        '<section class="result-card">'
        f'<p class="step-number">{index:02}</p><h2>{escape(label)}</h2>'
        f'<p>{escape(value)}</p></section>'
        for index, (label, value) in enumerate(sections.items(), start=1)
    )
