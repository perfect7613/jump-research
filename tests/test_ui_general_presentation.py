"""Copy/layout checks that do not anticipate the shared general contract."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from jump_ui.general_presentation import (
    EXAMPLES,
    HEADLINE,
    PLAN_LABELS,
    QUESTION,
    RESULT_LABELS,
    SCOPE,
    SUBCOPY,
    hero_html,
    particle_research_card,
    plan_shell,
    result_shell,
)


def test_general_product_copy_is_exact_and_bounded():
    assert HEADLINE == "Test an idea. See what the simulation says."
    assert SUBCOPY.startswith("Describe a question in plain English.")
    assert QUESTION == "What do you want to test?"
    assert EXAMPLES == (
        "Does changing traffic-light timing reduce a traffic jam?",
        "How often does switching doors win in the Monty Hall problem?",
        "What happens to an epidemic if contact rates fall by half?",
    )
    assert "no code, links, files, or real-world actions" in SCOPE
    assert "particle" not in hero_html().lower()
    assert "six-particle benchmark" in particle_research_card()


def test_general_plan_and_result_order_is_fixed():
    rows = OrderedDict((label, f"Example {label.lower()}") for label in PLAN_LABELS)
    html = plan_shell(rows)
    assert [html.index(label) for label in PLAN_LABELS] == sorted(html.index(label) for label in PLAN_LABELS)
    sections = OrderedDict((label, f"Example {label.lower()}") for label in RESULT_LABELS)
    cards = result_shell(sections)
    assert len(cards) == 5
    assert all(label in card for label, card in zip(RESULT_LABELS, cards))


def test_unvalidated_or_incomplete_rows_fail_closed():
    with pytest.raises(ValueError):
        plan_shell({"Question": "Only one row"})
    with pytest.raises(ValueError):
        result_shell({"Prediction": "Only one section"})
