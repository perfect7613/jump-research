"""Gradio assembly for the plain-language JUMP experiment flow."""

from __future__ import annotations

import os
from typing import Any
import uuid

from .flow import (
    ExperimentFlowError,
    LiveExperimentBackend,
    NonLiveContractFixtureBackend,
    plan_experiment,
    run_confirmed_experiment,
    technical_details,
)
from .presentation import plan_html, result_sections

QUESTION = "What do you want to test in this six-object world?"
EXAMPLES = (
    "Make matching objects repel instead of attract.",
    "Swap what World A learned into World B.",
    "Test whether the old force rule still fits.",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap');
:root, .gradio-container {
  --paper: #f4f0e7; --sheet: #fffdf7; --ink: #181711; --blue: #214a9b;
  --red: #c8442f; --line: #d6d0c3; --green: #25704a;
}
body, .gradio-container { background: var(--paper) !important; color: var(--ink) !important; }
.gradio-container { max-width: 980px !important; font-family: 'Newsreader', Georgia, serif !important; }
.jump-shell { padding: 56px 8px 24px; }
.jump-kicker, .eyebrow, .step-number, .fixture-label, .scope-note,
.plan-grid dt, .subscores dt, .tech-copy { font-family: 'DM Mono', monospace; }
.jump-kicker { color: var(--blue); font-size: 12px; letter-spacing: .13em; text-transform: uppercase; }
.jump-title { font-size: clamp(48px, 8vw, 82px); line-height: .92; letter-spacing: -.045em; margin: 18px 0; max-width: 820px; }
.jump-deck { font-size: 21px; line-height: 1.45; max-width: 680px; color: #4c493f; }
.fixture-banner { margin-top: 28px; padding: 10px 14px; border: 1px solid var(--red); color: var(--red); font: 500 11px 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
.jump-input textarea { font-family: 'Newsreader', Georgia, serif !important; font-size: 20px !important; line-height: 1.45 !important; background: var(--sheet) !important; border: 1px solid var(--ink) !important; }
.example-label { font: 11px 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .1em; color: #6d685c; margin: 10px 0 2px; }
.example-chip button { border: 1px solid var(--line) !important; background: transparent !important; color: var(--ink) !important; border-radius: 999px !important; font: 12px 'DM Mono', monospace !important; transition: border-color 150ms ease, transform 150ms cubic-bezier(.23,1,.32,1) !important; }
@media (hover:hover) and (pointer:fine) { .example-chip button:hover { border-color: var(--blue) !important; } }
.example-chip button:active, .run-button button:active { transform: scale(.97); }
.run-button button { min-height: 52px; background: var(--blue) !important; color: white !important; border: 0 !important; border-radius: 2px !important; font: 500 14px 'DM Mono', monospace !important; transition: transform 150ms cubic-bezier(.23,1,.32,1) !important; }
.sheet, .result-card { background: var(--sheet); border-top: 3px solid var(--ink); padding: 30px 32px; margin-top: 34px; }
.eyebrow, .step-number { font-size: 11px; color: var(--blue); letter-spacing: .1em; text-transform: uppercase; }
.sheet h2, .result-card h2 { font-size: 32px; margin: 10px 0 20px; }
.plan-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); }
.plan-grid div { background: var(--sheet); padding: 18px; }
.plan-grid dt, .subscores dt { font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: #6d685c; }
.plan-grid dd, .subscores dd { margin: 8px 0 0; font-size: 17px; }
.scope-note { font-size: 11px; line-height: 1.6; color: #6d685c; margin-top: 20px; }
.result-card { position: relative; }
.result-card img { width: 100%; display: block; border: 1px solid var(--line); margin-top: 22px; }
.result-card pre { white-space: pre-wrap; background: var(--paper); border: 1px solid var(--line); padding: 18px; font: 12px/1.6 'DM Mono', monospace; }
.fixture-label { color: var(--red); font-size: 10px; letter-spacing: .08em; }
.score-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.score-pair div { border: 1px solid var(--line); padding: 18px; display: flex; justify-content: space-between; align-items: baseline; }
.score-pair span { font-size: 18px; }.score-pair strong { font: 500 15px 'DM Mono', monospace; }
.good { color: var(--green); }.bad { color: var(--red); }
.subscores { margin-top: 18px; }.subscores div { display: flex; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 10px 0; }
.quiet { color: #6d685c; font-size: 15px; }.evidence-card { border-color: var(--blue); }
.progress-copy { font: 12px 'DM Mono', monospace; color: var(--blue); }
.error-copy { border-left: 3px solid var(--red); padding: 14px 18px; background: #fff5f1; }
.tech-copy { font-size: 11px; color: #5d594f; }
@media (max-width: 720px) { .jump-shell { padding-top: 28px; } .plan-grid { grid-template-columns: 1fr 1fr; } .score-pair { grid-template-columns: 1fr; } .sheet, .result-card { padding: 24px 20px; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition-duration: 0.01ms !important; } }
"""


def _friendly_error(message: str) -> str:
    lowered = message.lower()
    if any(word in lowered for word in ("url", "uri", "path", "code", "shell", "control", "ambiguous")):
        return (
            "That request doesn’t fit this six-object world. Try changing motion, "
            "hidden groups, attraction or repulsion, force strength, or a World A/World B swap."
        )
    return message


def _backend_from_environment():
    if os.environ.get("JUMP_UI_NONLIVE_FIXTURE") == "1":
        return NonLiveContractFixtureBackend()
    return LiveExperimentBackend()


def create_app(*, backend=None, enable_queue: bool = True):
    import gradio as gr

    backend = backend or _backend_from_environment()

    def parse_intent(intent: str):
        session_id = "space-" + uuid.uuid4().hex[:20]
        try:
            planned = plan_experiment(intent, session_id=session_id)
        except ExperimentFlowError as exc:
            return (
                None,
                gr.update(value=f'<div class="error-copy">{_friendly_error(str(exc))}</div>', visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        return (
            {"intent": intent.strip(), "planned_run": planned},
            gr.update(value=plan_html(planned), visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def execute_plan(confirmation, progress=gr.Progress()):
        if not isinstance(confirmation, dict):
            return (
                gr.update(value='<div class="error-copy">Describe an experiment first.</div>', visible=True),
                gr.update(visible=False),
                *[gr.update(visible=False) for _ in range(5)],
                gr.update(value="", visible=False),
            )
        stages = (
            "Building the world",
            "Running the prediction",
            "Checking what changed",
            "Scoring the answer",
            "Verifying the evidence",
        )
        for index, stage in enumerate(stages):
            progress(index / len(stages), desc=stage)
        try:
            completed = run_confirmed_experiment(
                confirmation["planned_run"],
                backend=backend,
                intent=confirmation["intent"],
            )
            cards = result_sections(completed, backend_label=backend.label)
            details = technical_details(completed, backend_label=backend.label)
        except ExperimentFlowError as exc:
            return (
                gr.update(value=f'<div class="error-copy">We couldn’t run this experiment. {_friendly_error(str(exc))}</div>', visible=True),
                gr.update(visible=False),
                *[gr.update(visible=False) for _ in range(5)],
                gr.update(value="", visible=False),
            )
        progress(1, desc="Evidence verified")
        return (
            gr.update(value='<p class="progress-copy">Experiment complete.</p>', visible=True),
            gr.update(visible=True),
            *[gr.update(value=card, visible=True) for card in cards],
            gr.update(value=details, visible=True),
        )

    with gr.Blocks(title="JUMP — test an idea in a tiny physics world") as demo:
        planned_state = gr.State(None)

        gr.HTML(
            '<header class="jump-shell"><p class="jump-kicker">JUMP · six-object physics</p>'
            '<h1 class="jump-title">Test an idea in a tiny physics world.</h1>'
            '<p class="jump-deck">Describe a bounded experiment with six moving objects. '
            'JUMP turns it into a safe plan, runs it, and shows what happened.</p>'
            '<p class="scope-note">Engineering demo only. The current Stage D result was null: '
            'the learned world state did not improve or shift the frozen model’s answer. '
            'This is not evidence of an informative representation or mechanism.</p>'
            + (
                f'<p class="fixture-banner">{backend.label} · for local contract and visual QA only</p>'
                if not backend.is_live
                else f'<p class="fixture-banner">{backend.label}</p>'
            )
            + '</header>'
        )
        intent = gr.Textbox(
            label=QUESTION,
            placeholder="For example: What changes if objects that used to attract each other begin to repel?",
            info="Six objects only · no code, links, files, or real-world tasks · 600 characters max",
            lines=4,
            max_lines=7,
            elem_classes=["jump-input"],
        )
        gr.HTML('<p class="example-label">Try an example</p>')
        with gr.Row():
            chips = [gr.Button(text, elem_classes=["example-chip"]) for text in EXAMPLES]
        run_button = gr.Button("Run experiment", elem_classes=["run-button"])

        plan_view = gr.HTML(visible=False)
        with gr.Row(visible=False) as confirmation:
            confirm_button = gr.Button("Run this plan", elem_classes=["run-button"])
            revise_button = gr.Button("Change my request")
        status = gr.HTML(visible=False)

        with gr.Group(visible=False) as results:
            result_cards = [gr.HTML(visible=False) for _ in range(5)]
            with gr.Accordion("Technical details", open=False):
                gr.Markdown(
                    "Hashes and immutable component identities are shown here for verification. "
                    "They do not establish answer correctness or a causal mechanism.",
                    elem_classes=["tech-copy"],
                )
                details = gr.Code(language="json", visible=False)

        for chip, example in zip(chips, EXAMPLES):
            chip.click(lambda value=example: value, outputs=intent, queue=False)

        run_button.click(
            parse_intent,
            inputs=intent,
            outputs=[planned_state, plan_view, confirmation, results],
            queue=False,
            trigger_mode="once",
        )
        intent.submit(
            parse_intent,
            inputs=intent,
            outputs=[planned_state, plan_view, confirmation, results],
            queue=False,
            trigger_mode="once",
        )
        confirm_button.click(
            execute_plan,
            inputs=planned_state,
            outputs=[status, results, *result_cards, details],
            concurrency_limit=1,
            trigger_mode="once",
        )
        revise_button.click(
            lambda: (None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
            outputs=[planned_state, plan_view, confirmation, results],
            queue=False,
        )

    demo._jump_css = CSS
    if enable_queue:
        demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    demo = create_app()
    demo.launch(css=demo._jump_css, footer_links=[])


if __name__ == "__main__":
    main()
