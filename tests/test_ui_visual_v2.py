from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from jump_contracts.thought_experiments import QUESTION_VERSION, validate_experiment_spec, validate_thought_experiment_run
from jump_ui.visual_client import CODE_VERSION, EXPECTED_MODEL, VisualCoordinatorClient, VisualClientError
from jump_ui.visual_presentation import chart_html, result_html, spec_html, visual_html
from jump_workbench.gemma_planner import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION
from jump_workbench.visual_coordinator import VisualCoordinator
from jump_workbench.visual_engine import execute_visual_spec
from jump_workbench.workflow import FrozenModel


def fields():
    return {
        "question": "What changes when attraction becomes repulsion halfway through?",
        "hypothesis": "Repulsion increases the particles' separation.",
        "world": {
            "bounds": {"width": 100.0, "height": 100.0, "boundary": "reflect"},
            "entities": [{
                "id": "particle", "label": "Particles", "count": 12,
                "appearance": {"shape": "circle", "color": "#244fb3", "size": 2.0},
                "initial_state": {"numeric": {"vx": 0.0, "vy": 0.0}, "categorical": {"kind": "matter"}},
                "initial_layout": {"kind": "ring", "center": [50.0, 50.0], "spread": 15.0},
            }],
            "graph": {"kind": "none", "edge_probability": 0.0, "directed": False},
        },
        "dynamics": {"rules": [
            {"id": "force", "op": "pairwise_force_2d", "target_type": "particle", "parameters": {"strength": 8.0, "exponent": 2.0, "softening": 2.0}},
            {"id": "motion", "op": "move_2d", "target_type": "particle", "parameters": {"damping": 0.98, "max_speed": 5.0}},
        ]},
        "conditions": [
            {"id": "baseline", "label": "Attraction", "kind": "baseline", "interventions": []},
            {"id": "counterfactual", "label": "Repulsion after step 5", "kind": "counterfactual", "interventions": [{"time": 5, "operation": "scale_rule_parameter", "target": "force", "field": "strength", "value": -1.0}]},
        ],
        "schedule": {"duration_steps": 20, "dt": 0.2},
        "measurements": [{"id": "mean_speed", "label": "Mean x velocity", "op": "mean_state", "entity_type": "particle", "state": "vx", "category": None}],
        "visualization": {"kind": "animated_2d", "frame_stride": 1, "max_frames": 21, "chart_measurement_ids": ["mean_speed"]},
    }


def responses():
    state = {}

    def model(action, _payload):
        if action == "visual_spec":
            return fields()
        if action == "visual_predict":
            return {"summary": "Repulsion should change mean velocity.", "expected_direction": "change", "measurement_id": "mean_speed"}
        return {"disposition": "retain", "interpretation": "The deterministic comparison changed as predicted."}

    def simulate(spec, _prediction, recorded):
        start = datetime.fromisoformat(recorded.replace("Z", "+00:00")) + timedelta(milliseconds=1)
        return {"modal_call_id": "fc-ui-visual", "started_at": start, "completed_at": start + timedelta(seconds=1), "result": execute_visual_spec(spec)}

    coordinator = VisualCoordinator(
        state=state,
        model=FrozenModel(model_id=BASE_REPO_ID, revision=BASE_REVISION),
        transformers_revision=TRANSFORMERS_REVISION,
        model_generate=model,
        simulate=simulate,
        code_version=CODE_VERSION,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    request = {"schema_version": QUESTION_VERSION, "request_id": "req-v2", "session_id": "session-v2", "intent": "Reverse attraction after step five.", "seed": 7613, "repetitions": 2}
    planned = coordinator.compile(request)
    completed = coordinator.confirm({**planned["confirmation"], "confirmed": True})
    return request, planned, completed


class Response:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def read(self): return json.dumps(self.value).encode()


def test_v2_client_uses_exact_routes_confirmation_and_direct_validators(monkeypatch):
    question, planned_response, run_response = responses()
    replies = iter((planned_response, run_response))
    requests = []

    def urlopen(request, *, timeout):
        requests.append(request)
        return Response(next(replies))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = VisualCoordinatorClient("https://visual.invalid", "server-secret")
    planned = client.spec(question)
    completed = client.confirm(planned)
    assert validate_experiment_spec(completed["spec"]) == completed["spec"]
    assert validate_thought_experiment_run(completed["run"], completed["spec"]) == completed["run"]
    assert [request.full_url for request in requests] == [
        "https://visual.invalid/v2/thought-experiments/spec",
        "https://visual.invalid/v2/thought-experiments/confirm",
    ]
    confirmation = json.loads(requests[1].data)
    assert set(confirmation) == {"schema_version", "request_id", "session_id", "spec_id", "spec_sha256", "confirmation_token", "confirmed"}
    assert confirmation == {**planned_response["confirmation"], "confirmed": True}
    assert requests[0].get_header("Authorization") == "Bearer server-secret"


def test_old_confirmation_and_request_extras_fail_before_transport(monkeypatch):
    question, planned, _ = responses()
    client = VisualCoordinatorClient("https://visual.invalid", "secret")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: pytest.fail("transport called"))
    with pytest.raises(VisualClientError, match="fields"):
        client.spec({**question, "source": "not accepted"})
    old = dict(planned)
    old["confirmation"] = {"schema_version": "jump.thought-experiment-confirmation/v2", "spec_id": planned["spec"]["spec_id"], "spec_sha256": planned["spec"]["spec_sha256"], "confirmed": False}
    with pytest.raises(VisualClientError, match="confirmation fields"):
        client.confirm(old)


def test_visual_presentation_labels_deterministic_frames_and_has_no_learned_alias():
    _, _, completed = responses()
    spec, run = completed["spec"], completed["run"]
    html = spec_html(spec) + result_html(spec, run) + visual_html(spec, run) + chart_html(spec, run)
    assert "Hypothesis" in html and "Prediction" in html and "Measured result" in html
    assert "Deterministic simulation frames" in html
    assert "not a learned-latent reconstruction" in html
    assert "animateTransform" in html
    assert "Measurement comparison" in html or "Measured comparison" in html


def test_visual_app_is_primary_and_v1_is_explicit_fallback():
    pytest.importorskip("gradio")
    from jump_ui.visual_app import create_visual_app
    config = create_visual_app().get_config_file()
    text = str(config)
    assert "Visual thought experiment" in text
    assert "Earlier numeric experiment" in text
    assert "What if twelve particles start with x and y velocity 0" in text
    build = next(item for item in config["dependencies"] if item["api_name"] == "build")
    confirm = next(item for item in config["dependencies"] if item["api_name"] == "confirm")
    assert build["queue"] is True and build["types"]["generator"] is True
    assert confirm["queue"] is True and confirm["types"]["generator"] is True
    assert confirm["scroll_to_output"] is True


def test_expected_model_is_frozen_base_gemma_without_adapter():
    assert EXPECTED_MODEL["frozen"] is True
    assert EXPECTED_MODEL["adapter_id"] is None
