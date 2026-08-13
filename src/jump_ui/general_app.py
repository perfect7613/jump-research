"""Local-only UI for the future general JUMP simulation workbench."""

from __future__ import annotations

import json
import uuid

from .app import CSS
from .general_flow import FIXTURE_LABEL, GeneralUIError, confirm_fixture, execute_fixture, plan_rows, prepare_fixture, result_rows
from .general_presentation import EXAMPLES, QUESTION, SCOPE, hero_html, particle_research_card, plan_shell, result_shell

GENERAL_CSS = CSS + """
.research-card { background: var(--sheet); border: 1px solid var(--line); padding: 22px 24px; margin: 34px 0 10px; }
.research-card h2 { color: var(--ink) !important; font-size: 24px; margin: 8px 0; }
.research-card p { color: var(--muted) !important; }
.research-card a { color: var(--blue) !important; font-family: 'DM Mono', monospace; font-size: 12px; }
.general-plan-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); border: 1px solid var(--line); }
.general-plan-grid div { background: var(--sheet); padding: 18px; }
.general-plan-grid dt { color: var(--muted) !important; font: 10px 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .08em; }
.general-plan-grid dd { color: var(--ink) !important; margin: 8px 0 0; font-size: 17px; }
@media (max-width: 720px) { .general-plan-grid { grid-template-columns: 1fr; } }
"""


def create_general_app():
    import gradio as gr

    def make_plan(intent: str):
        try:
            planned = prepare_fixture(intent, request_id="req-" + uuid.uuid4().hex[:20])
        except GeneralUIError as exc:
            return None, gr.update(value=f'<div class="error-copy">{exc}</div>', visible=True), gr.update(visible=False)
        return planned, gr.update(value=plan_shell(plan_rows(planned["plan"])), visible=True), gr.update(visible=True)

    def confirm_and_run(planned):
        if not isinstance(planned, dict):
            return gr.update(value='<div class="error-copy">Build and confirm a plan first.</div>', visible=True), *[gr.update(visible=False) for _ in range(5)], gr.update(value="", visible=False)
        try:
            prepared = confirm_fixture(planned, confirmed=True)
            run = execute_fixture(prepared)
            cards = result_shell(result_rows(run, prepared.plan))
            details = json.dumps({
                "mode": FIXTURE_LABEL, "plan_id": run["plan_id"], "run_id": run["run_id"],
                "plan_sha256": run["plan_sha256"], "run_sha256": run["run_sha256"],
                "policy_sha256": run["execution"]["policy_sha256"],
            }, indent=2, sort_keys=True)
        except GeneralUIError as exc:
            return gr.update(value=f'<div class="error-copy">{exc}</div>', visible=True), *[gr.update(visible=False) for _ in range(5)], gr.update(value="", visible=False)
        return gr.update(value='<p class="progress-copy">Non-live fixture complete.</p>', visible=True), *[gr.update(value=card, visible=True) for card in cards], gr.update(value=details, visible=True)

    with gr.Blocks(title="JUMP — test an idea with a simulation") as demo:
        state = gr.State(None)
        gr.HTML(hero_html())
        gr.HTML(f'<p class="fixture-banner">{FIXTURE_LABEL} · local QA only</p>')
        question = gr.Textbox(label=QUESTION, placeholder=EXAMPLES[0], info=SCOPE, lines=4, max_lines=7, elem_classes=["jump-input"])
        gr.HTML('<p class="example-label">Try an example</p>')
        with gr.Row():
            chips = [gr.Button(value, elem_classes=["example-chip"]) for value in EXAMPLES]
        plan_button = gr.Button("Build experiment plan", elem_classes=["run-button"])
        plan_view = gr.HTML(visible=False)
        confirm = gr.Button("Confirm prediction and run fixture", visible=False, elem_classes=["run-button"])
        status = gr.HTML(visible=False)
        cards = [gr.HTML(visible=False) for _ in range(5)]
        with gr.Accordion("Technical details", open=False):
            details = gr.Code(language="json", visible=False)
        gr.HTML(particle_research_card())
        for chip, example in zip(chips, EXAMPLES):
            chip.click(lambda value=example: value, outputs=question, queue=False)
        plan_button.click(make_plan, inputs=question, outputs=[state, plan_view, confirm], queue=False)
        confirm.click(confirm_and_run, inputs=state, outputs=[status, *cards, details], concurrency_limit=1)
    demo._jump_css = GENERAL_CSS
    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    demo = create_general_app()
    demo.launch(css=demo._jump_css, footer_links=[])


if __name__ == "__main__":
    main()
