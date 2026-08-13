"""Primary UI for declarative visual thought experiments, with explicit v1 fallback."""

from __future__ import annotations

from html import escape
import json
import uuid

from jump_contracts.thought_experiments import QUESTION_VERSION
from jump_workbench.workflow import validate_user_intent

from .app import CSS
from .general_client import GeneralCoordinatorClient, GeneralCoordinatorError
from .general_flow import plan_rows, result_rows
from .general_presentation import completed_summary, plan_shell, plan_summary
from .visual_client import VisualClientError, VisualCoordinatorClient
from .visual_presentation import EXAMPLES, chart_html, result_html, spec_html, visual_html

VISUAL_CSS = CSS + """
.visual-summary, .simulation-frames, .comparison-chart { background:var(--sheet); border:1px solid var(--line); padding:22px 24px; margin:18px 0; }
.visual-summary { border:2px solid var(--blue); }
.visual-summary h2 { color:var(--ink)!important; font-size:27px; margin:8px 0 14px; }
.visual-summary p, .simulation-frames p, .comparison-chart p { color:var(--ink)!important; }
.limitation { color:var(--muted)!important; font:11px/1.6 'DM Mono',monospace; }
.visual-pair { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.visual-pair figure { margin:0; }.visual-pair figcaption { font:12px 'DM Mono',monospace; margin-bottom:8px; }
.visual-pair svg, .comparison-chart svg { width:100%; display:block; }
.chart-legend { display:flex; flex-wrap:wrap; gap:14px; font:11px 'DM Mono',monospace; }
.service-status { color:var(--muted); font:11px 'DM Mono',monospace; }
@media(max-width:720px){.visual-pair{grid-template-columns:1fr}}
"""

PRIMARY = "Visual thought experiment"
FALLBACK = "Earlier numeric experiment"


def create_visual_app():
    import gradio as gr

    def health():
        try:
            VisualCoordinatorClient.from_environment().health()
            return '<p class="service-status">Visual simulator ready · deterministic engine · no generated code</p>'
        except VisualClientError as exc:
            return f'<p class="error-copy">Visual simulator unavailable: {escape(str(exc))}</p>'

    def build(intent: str, mode: str):
        yield None, gr.update(visible=False), gr.update(visible=False), gr.update(
            value='<p class="progress-copy">Building a bounded experiment… This can take about a minute.</p>', visible=True
        ), gr.update(value="Building experiment…", interactive=False)
        try:
            text = validate_user_intent(intent)
            request_id = "req-" + uuid.uuid4().hex[:20]
            session_id = "ui-" + uuid.uuid4().hex[:20]
            if mode == FALLBACK:
                planned = GeneralCoordinatorClient.from_environment().plan({
                    "schema_version": "jump.experiment-question/v1", "request_id": request_id,
                    "session_id": session_id, "intent": text, "seed": 7613, "repetitions": 8,
                })
                rows = plan_rows(planned["plan"])
                state = {"kind": "v1", "payload": planned}
                view = plan_shell(rows) + plan_summary(rows)
            else:
                planned = VisualCoordinatorClient.from_environment().spec({
                    "schema_version": QUESTION_VERSION, "request_id": request_id,
                    "session_id": session_id, "intent": text, "seed": 7613, "repetitions": 2,
                })
                state = {"kind": "v2", "payload": planned}
                view = spec_html(planned["spec"])
        except (VisualClientError, GeneralCoordinatorError, ValueError) as exc:
            yield None, gr.update(value=f'<section class="visual-summary"><p class="error-copy">{escape(str(exc))}</p><p>Edit the question and try again. No cached experiment was substituted.</p></section>', visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value="Build experiment", interactive=True)
            return
        yield state, gr.update(value=view, visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(value="Build experiment", interactive=True)

    def confirm(state):
        yield gr.update(value='<section class="visual-summary"><p class="eyebrow">Prediction and simulation</p><h2>Recording a prediction, then running both conditions…</h2></section>', visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False)
        if not isinstance(state, dict) or state.get("kind") not in {"v1", "v2"}:
            yield gr.update(value='<p class="error-copy">Build and review an experiment first.</p>', visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False)
            return
        try:
            if state["kind"] == "v1":
                completed = GeneralCoordinatorClient.from_environment().confirm(state["payload"])
                sections = result_rows(completed["run"], completed["plan"])
                summary = completed_summary(sections) + '<p class="limitation">Earlier numeric fallback: no visual simulation frames are produced.</p>'
                visual, chart = "", ""
                identity = {"plan_id": completed["run"]["plan_id"], "run_id": completed["run"]["run_id"], "mode": "v1 numeric fallback"}
            else:
                completed = VisualCoordinatorClient.from_environment().confirm(state["payload"])
                spec, run = completed["spec"], completed["run"]
                summary = result_html(spec, run)
                visual, chart = visual_html(spec, run), chart_html(spec, run)
                identity = {
                    "spec_id": spec["spec_id"], "run_id": run["run_id"],
                    "engine_id": run["execution"]["engine_id"],
                    "modal_call_id": run["execution"]["modal_call_id"],
                    "prediction_recorded_at": run["execution"]["prediction_recorded_at"],
                    "started_at": run["execution"]["started_at"],
                }
        except (VisualClientError, GeneralCoordinatorError, ValueError) as exc:
            yield gr.update(value=f'<section class="visual-summary"><p class="error-copy">{escape(str(exc))}</p><p>Edit the question and build a new spec. No cached result was substituted.</p></section>', visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False)
            return
        yield gr.update(value=summary, visible=True), gr.update(value=visual, visible=bool(visual)), gr.update(value=chart, visible=bool(chart)), gr.update(value=json.dumps(identity, indent=2, sort_keys=True), visible=True)

    with gr.Blocks(title="JUMP — visual thought experiments") as demo:
        state = gr.State(None)
        gr.HTML('<header class="jump-shell"><p class="jump-kicker">Bounded visual thought experiments</p><h1 class="jump-title">What changes when one rule changes?</h1><p class="jump-deck">Describe a small world and one counterfactual. Review the plan, record a prediction, then compare deterministic simulation frames.</p></header>')
        service = gr.HTML('<p class="service-status">Checking visual simulator…</p>')
        question = gr.Textbox(label="Question", placeholder=EXAMPLES[0], info="Supported bounded simulations only · no code, URLs, files, or real-world actions", lines=4, elem_classes=["jump-input"])
        gr.HTML('<p class="example-label">Try a concrete question</p>')
        with gr.Row():
            chips = [gr.Button(example, elem_classes=["example-chip"]) for example in EXAMPLES]
        mode = gr.Radio([PRIMARY, FALLBACK], value=PRIMARY, label="Experiment type", info="Use the earlier numeric route only when you explicitly need the v1 fallback.")
        build_button = gr.Button("Build experiment", elem_classes=["run-button"])
        progress = gr.HTML(visible=False)
        plan = gr.HTML(visible=False, autoscroll=True)
        confirm_button = gr.Button("Confirm and run both conditions", visible=False, elem_classes=["run-button"])
        result = gr.HTML(visible=False, autoscroll=True)
        frames = gr.HTML(visible=False)
        chart = gr.HTML(visible=False)
        with gr.Accordion("Technical details", open=False):
            details = gr.Code(language="json", visible=False)
        for chip, example in zip(chips, EXAMPLES):
            chip.click(lambda value=example: value, outputs=question, queue=False)
        build_button.click(build, [question, mode], [state, plan, confirm_button, progress, build_button], concurrency_limit=1)
        confirm_button.click(confirm, state, [result, frames, chart, details], concurrency_limit=1, scroll_to_output=True)
        demo.load(health, outputs=service, queue=False, api_name="visual_health")
    demo._jump_css = VISUAL_CSS
    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> None:
    demo = create_visual_app()
    demo.launch(css=demo._jump_css, footer_links=[])


if __name__ == "__main__":
    main()
