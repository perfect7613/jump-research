from __future__ import annotations

import json
from io import BytesIO
import urllib.error

import pytest

from jump_ui.general_client import (
    EXPECTED_COORDINATOR_CODE_VERSION,
    EXPECTED_MODEL,
    EXPECTED_PLAN_SCHEMA_SHA256,
    EXPECTED_RUN_SCHEMA_SHA256,
    GENERAL_COORDINATOR_URL,
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
    model = dict(EXPECTED_MODEL)
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


@pytest.mark.parametrize("intent", EXAMPLES[:2])
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


@pytest.mark.parametrize("field", ("source", "code", "url"))
def test_client_rejects_response_aliases_and_unsafe_extras(monkeypatch, field):
    plan_response, _ = responses_for(EXAMPLES[1])
    plan_response[field] = "not accepted"
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response(plan_response))
    with pytest.raises(GeneralCoordinatorError, match="unexpected fields"):
        GeneralCoordinatorClient("https://coordinator.invalid", "secret").plan(question(EXAMPLES[1]))


def test_client_has_no_unconfigured_fallback(monkeypatch):
    monkeypatch.delenv("JUMP_MODAL_TOKEN", raising=False)
    with pytest.raises(GeneralCoordinatorError, match="No result was substituted"):
        GeneralCoordinatorClient.from_environment()


def test_unsupported_coordinator_detail_is_shown_without_fallback(monkeypatch):
    def reject(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://coordinator.invalid/v1/experiments/plan",
            400,
            "Bad Request",
            {},
            BytesIO(json.dumps({"detail": "unsupported experiment: no supported template"}).encode()),
        )

    monkeypatch.setattr("urllib.request.urlopen", reject)
    with pytest.raises(GeneralCoordinatorError, match="unsupported experiment: no supported template"):
        GeneralCoordinatorClient("https://coordinator.invalid", "secret").plan(question(EXAMPLES[0]))


def test_environment_client_uses_only_reviewed_endpoint_and_server_token(monkeypatch):
    monkeypatch.setenv("JUMP_MODAL_TOKEN", "server-token")
    variables = {
        "JUMP_GENERAL_ENDPOINT": GENERAL_COORDINATOR_URL,
        "JUMP_GENERAL_CODE_VERSION": EXPECTED_COORDINATOR_CODE_VERSION,
        "JUMP_GENERAL_MODEL_REPO_ID": EXPECTED_MODEL["repo_id"],
        "JUMP_GENERAL_MODEL_REVISION": EXPECTED_MODEL["revision"],
        "JUMP_GENERAL_TRANSFORMERS_REVISION": EXPECTED_MODEL["transformers_revision"],
        "JUMP_GENERAL_PLAN_SCHEMA_SHA256": EXPECTED_PLAN_SCHEMA_SHA256,
        "JUMP_GENERAL_RUN_SCHEMA_SHA256": EXPECTED_RUN_SCHEMA_SHA256,
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    client = GeneralCoordinatorClient.from_environment()
    assert client.base_url == GENERAL_COORDINATOR_URL
    assert client.token == "server-token"
    assert client.expected_code_version == EXPECTED_COORDINATOR_CODE_VERSION


def test_environment_client_rejects_unreviewed_deployment_pin(monkeypatch):
    monkeypatch.setenv("JUMP_MODAL_TOKEN", "server-token")
    monkeypatch.setenv("JUMP_GENERAL_ENDPOINT", "https://unreviewed.invalid")
    with pytest.raises(GeneralCoordinatorError, match="JUMP_GENERAL_ENDPOINT"):
        GeneralCoordinatorClient.from_environment()
