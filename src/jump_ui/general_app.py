"""Local-only UI for the future general JUMP simulation workbench."""

from __future__ import annotations

from html import escape
import json
import uuid

from jump_workbench.workflow import QUESTION_VERSION, validate_user_intent

from .app import CSS
from .general_client import GeneralCoordinatorClient, GeneralCoordinatorError
from .general_flow import plan_rows, result_rows
from .general_presentation import (
    EXAMPLES, QUESTION, SCOPE, completed_summary, hero_html, no_learned_world_image,
    particle_research_card, plan_shell, plan_summary, ready_summary, result_shell,
)

GENERAL_CSS = CSS + """
.research-card { background: var(--sheet); border: 1px solid var(--line); padding: 22px 24px; margin: 34px 0 10px; }
.research-card h2 { color: var(--ink) !important; font-size: 24px; margin: 8px 0; }
.research-card p { color: var(--muted) !important; }
.research-card a { color: var(--blue) !important; font-family: 'DM Mono', monospace; font-size: 12px; }
.general-plan-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); border: 1px solid var(--line); }
.general-plan-grid div { background: var(--sheet); padding: 18px; }
.general-plan-grid dt { color: var(--muted) !important; font: 10px 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .08em; }
.general-plan-grid dd { color: var(--ink) !important; margin: 8px 0 0; font-size: 17px; }
.result-summary { background: var(--sheet); border: 2px solid var(--blue); padding: 22px 24px; margin: 18px 0; }
.result-summary h2 { color: var(--ink) !important; font-size: 25px; margin: 8px 0 12px; }
.result-summary p, .image-status p { color: var(--ink) !important; }
.image-status { background: var(--sheet); border: 1px solid var(--line); padding: 18px 22px; margin: 12px 0 22px; }
@media (max-width: 720px) { .general-plan-grid { grid-template-columns: 1fr; } }
"""


def create_general_app():
    import gradio as gr

    def make_plan(intent: str):
        yield (
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(
                value='<p class="progress-copy">Building the experiment plan… This usually takes about a minute.</p>',
                visible=True,
            ),
            gr.update(value="Building experiment plan…", interactive=False),
            gr.update(value=ready_summary(), visible=True),
        )
        try:
            normalized = validate_user_intent(intent)
            if len(normalized) > 600:
                raise GeneralCoordinatorError("The question must contain 600 characters or fewer.")
            planned = GeneralCoordinatorClient.from_environment().plan({
                "schema_version": QUESTION_VERSION,
                "request_id": "req-" + uuid.uuid4().hex[:20],
                "session_id": "ui-" + uuid.uuid4().hex[:20],
                "intent": normalized,
                "seed": 7613,
                "repetitions": 8,
            })
        except (GeneralCoordinatorError, ValueError) as exc:
            yield (
                None,
                gr.update(value=f'<div class="error-copy">{escape(str(exc))}</div>', visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="Build experiment plan", interactive=True),
                gr.update(
                    value=(
                        '<section class="result-summary" aria-live="polite">'
                        f'<p class="error-copy">{escape(str(exc))}</p></section>'
                    ),
                    visible=True,
                ),
            )
            return
        rows = plan_rows(planned["plan"])
        yield (
            planned,
            gr.update(value=plan_shell(rows), visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="Build experiment plan", interactive=True),
            gr.update(value=plan_summary(rows), visible=True),
        )

    def confirm_and_run(planned):
        if not isinstance(planned, dict):
            return gr.update(value='<div class="error-copy">Build and confirm a plan first.</div>', visible=True), gr.update(visible=False), *[gr.update(visible=False) for _ in range(5)], gr.update(value="", visible=False)
        try:
            completed = GeneralCoordinatorClient.from_environment().confirm(planned)
            run, plan = completed["run"], completed["plan"]
            sections = result_rows(run, plan)
            cards = result_shell(sections)
            details = json.dumps({
                "model": completed["model"], "plan_id": run["plan_id"], "run_id": run["run_id"],
                "plan_sha256": run["plan_sha256"], "run_sha256": run["run_sha256"],
                "policy_sha256": run["execution"]["policy_sha256"],
            }, indent=2, sort_keys=True)
        except (GeneralCoordinatorError, ValueError) as exc:
            return gr.update(value=f'<div class="error-copy">{escape(str(exc))}</div>', visible=True), gr.update(value=no_learned_world_image(), visible=True), *[gr.update(visible=False) for _ in range(5)], gr.update(value="", visible=False)
        return gr.update(value=completed_summary(sections), visible=True), gr.update(value=no_learned_world_image(), visible=True), *[gr.update(value=card, visible=True) for card in cards], gr.update(value=details, visible=True)

    with gr.Blocks(title="JUMP — test an idea with a simulation") as demo:
        state = gr.State(None)
        gr.HTML(hero_html())
        gr.HTML('<p class="fixture-banner">GENERAL WORKBENCH · PLAN REVIEW REQUIRED</p>')
        question = gr.Textbox(label=QUESTION, placeholder=EXAMPLES[0], info=SCOPE, lines=4, max_lines=7, elem_classes=["jump-input"])
        gr.HTML('<p class="example-label">Try an example</p>')
        with gr.Row():
            chips = [gr.Button(value, elem_classes=["example-chip"]) for value in EXAMPLES]
        plan_button = gr.Button("Build experiment plan", elem_classes=["run-button"])
        planning_status = gr.HTML(visible=False)
        status = gr.HTML(value=ready_summary(), visible=True, autoscroll=True)
        plan_view = gr.HTML(visible=False)
        confirm = gr.Button("Confirm plan and run simulation", visible=False, elem_classes=["run-button"])
        image_status = gr.HTML(visible=False)
        cards = [gr.HTML(visible=False) for _ in range(5)]
        with gr.Accordion("Technical details", open=False):
            details = gr.Code(language="json", visible=False)
        gr.HTML(particle_research_card())
        for chip, example in zip(chips, EXAMPLES):
            chip.click(lambda value=example: value, outputs=question, queue=False)
        plan_button.click(
            make_plan,
            inputs=question,
            outputs=[state, plan_view, confirm, planning_status, plan_button, status],
            concurrency_limit=1,
        )
        confirm.click(
            confirm_and_run,
            inputs=state,
            outputs=[status, image_status, *cards, details],
            concurrency_limit=1,
            scroll_to_output=True,
        )
    demo._jump_css = GENERAL_CSS
    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    demo = create_general_app()
    demo.launch(css=demo._jump_css, footer_links=[])


if __name__ == "__main__":
    main()
