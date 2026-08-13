from __future__ import annotations

import json

import pytest

from jump_ui.general_client import (
    GeneralCoordinatorClient,
    GeneralCoordinatorError,
    QUESTION_VERSION,
)
from jump_ui.general_flow import confirm_fixture, execute_fixture, prepare_fixture
from jump_ui.general_presentation import EXAMPLES


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def responses_for(intent: str):
    prepared = prepare_fixture(intent, request_id="req-test", repetitions=3)
    run = execute_fixture(confirm_fixture(prepared, confirmed=True))
    model = {
        "repo_id": "google/gemma-3-4b-it",
        "revision": "a" * 40,
        "transformers_revision": "b" * 40,
        "frozen": True,
        "adapter_id": None,
    }
    plan = {
        "schema_version": "jump.experiment-plan-response/v1",
        "status": "awaiting_confirmation",
        "request_id": "req-test",
        "session_id": "session-test",
        "plan": prepared["plan"],
        "model": model,
        "confirmation": {
            "schema_version": "jump.experiment-confirmation/v1",
            "plan_id": prepared["plan"]["plan_id"],
            "plan_sha256": prepared["plan"]["plan_sha256"],
            "confirmed": False,
        },
    }
    completed = {
        "schema_version": "jump.experiment-run-response/v1",
        "status": "completed",
        "request_id": "req-test",
        "session_id": "session-test",
        "plan": prepared["plan"],
        "run": run,
        "model": model,
    }
    return plan, completed


def question(intent: str) -> dict:
    return {
        "schema_version": QUESTION_VERSION,
        "request_id": "req-test",
        "session_id": "session-test",
        "intent": intent,
        "seed": 7613,
        "repetitions": 3,
    }


@pytest.mark.parametrize("intent", EXAMPLES[1:])
def test_client_sends_only_exact_question_then_exact_confirmation(monkeypatch, intent):
    plan_response, run_response = responses_for(intent)
    responses = iter((plan_response, run_response))
    requests = []

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        return Response(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    client = GeneralCoordinatorClient("https://coordinator.invalid/", "server-secret")
    planned = client.plan(question(intent))
    completed = client.confirm(planned)

    assert completed["status"] == "completed"
    assert [request.full_url for request, _ in requests] == [
        "https://coordinator.invalid/v1/experiments/plan",
        "https://coordinator.invalid/v1/experiments/confirm",
    ]
    plan_body = json.loads(requests[0][0].data)
    confirm_body = json.loads(requests[1][0].data)
    assert plan_body == question(intent)
    assert "source" not in plan_body and "code" not in plan_body and "url" not in plan_body
    assert confirm_body == {**plan_response["confirmation"], "confirmed": True}
    assert requests[0][0].get_header("Authorization") == "Bearer server-secret"


def test_client_rejects_question_extras_before_transport(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("transport called"))
    request = {**question(EXAMPLES[1]), "source": "print('unsafe')"}
    with pytest.raises(GeneralCoordinatorError, match="fields"):
        GeneralCoordinatorClient("https://coordinator.invalid", "secret").plan(request)


def test_client_has_no_unconfigured_fallback(monkeypatch):
    monkeypatch.delenv("JUMP_GENERAL_COORDINATOR_URL", raising=False)
    monkeypatch.delenv("JUMP_MODAL_TOKEN", raising=False)
    with pytest.raises(GeneralCoordinatorError, match="No result was substituted"):
        GeneralCoordinatorClient.from_environment()
