"""Strict client for the declarative visual thought-experiment API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping
import urllib.error
import urllib.request

from jump_contracts.thought_experiments import (
    CONFIRMATION_VERSION,
    EXPERIMENT_SPEC_SCHEMA_SHA256,
    QUESTION_VERSION,
    RUN_RESPONSE_VERSION,
    SPEC_RESPONSE_VERSION,
    THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256,
    validate_experiment_spec,
    validate_thought_experiment_run,
)

ENDPOINT = "https://ameymuke252003--jump-general-experiment-workbench-genera-d81606.modal.run"
CODE_VERSION = "46176d8d69fc8b81e1cf0d1390871772ad270f8c"
SPEC_SCHEMA_SHA256 = "fa7674dc3c5f759dc74ff723cef7a194edc4186069496e631e65b4d0ebd84ab5"
RUN_SCHEMA_SHA256 = "55d1fd3fdef215abfb1a148080cc01aea3fff118ba1e779e02e6841f43941166"
ENGINE_ID = "jump.declarative-visual-engine/v2"
EXPECTED_MODEL = {
    "repo_id": "google/gemma-4-12B-it",
    "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "transformers_revision": "918dbf131d0df5b46e3f6e1d96174d62aa4d16d6",
    "frozen": True,
    "adapter_id": None,
}


class VisualClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisualCoordinatorClient:
    endpoint: str
    token: str
    timeout_seconds: float = 300.0

    @classmethod
    def from_environment(cls) -> "VisualCoordinatorClient":
        token = os.environ.get("JUMP_MODAL_TOKEN", "").strip()
        expected = {
            "JUMP_GENERAL_ENDPOINT": ENDPOINT,
            "JUMP_VISUAL_CODE_VERSION": CODE_VERSION,
            "JUMP_VISUAL_SPEC_SCHEMA_SHA256": SPEC_SCHEMA_SHA256,
            "JUMP_VISUAL_RUN_SCHEMA_SHA256": RUN_SCHEMA_SHA256,
        }
        mismatched = [key for key, value in expected.items() if os.environ.get(key, "").strip() != value]
        if not token or mismatched:
            suffix = f": {', '.join(mismatched)}" if mismatched else ""
            raise VisualClientError("The visual experiment service is not configured" + suffix)
        return cls(os.environ["JUMP_GENERAL_ENDPOINT"].strip(), token)

    def health(self) -> dict[str, Any]:
        payload = self._request("/health", None)
        visual = payload.get("thought_experiments_v2") if isinstance(payload, dict) else None
        expected = {
            "question_schema_version": QUESTION_VERSION,
            "spec_schema_sha256": SPEC_SCHEMA_SHA256,
            "run_schema_sha256": RUN_SCHEMA_SHA256,
            "engine_id": ENGINE_ID,
            "generated_code": False,
            "learned_decoder": False,
        }
        if payload.get("status") != "available" or payload.get("code_version") != CODE_VERSION or visual != expected:
            raise VisualClientError("The visual experiment health pins do not match the reviewed deployment")
        return payload

    def spec(self, request: Mapping[str, Any]) -> dict[str, Any]:
        fields = {"schema_version", "request_id", "session_id", "intent", "seed", "repetitions"}
        if set(request) != fields or request.get("schema_version") != QUESTION_VERSION:
            raise VisualClientError("The visual question fields do not match the v2 API")
        payload = self._request("/v2/thought-experiments/spec", dict(request))
        if set(payload) != {"schema_version", "status", "request_id", "session_id", "spec", "model", "confirmation"}:
            raise VisualClientError("The visual spec response contains unexpected fields")
        if payload["schema_version"] != SPEC_RESPONSE_VERSION or payload["status"] != "awaiting_confirmation":
            raise VisualClientError("The visual spec response state is invalid")
        if (payload["request_id"], payload["session_id"]) != (request["request_id"], request["session_id"]):
            raise VisualClientError("The visual spec response does not bind the request")
        payload["spec"] = _spec(payload["spec"])
        _model(payload["model"])
        _confirmation(payload["confirmation"], payload, confirmed=False)
        return payload

    def confirm(self, planned: Mapping[str, Any]) -> dict[str, Any]:
        if planned.get("schema_version") != SPEC_RESPONSE_VERSION or planned.get("status") != "awaiting_confirmation":
            raise VisualClientError("Only a reviewed visual spec can be confirmed")
        spec = _spec(planned.get("spec", {}))
        _model(planned.get("model"))
        confirmation = dict(planned.get("confirmation", {}))
        _confirmation(confirmation, planned, confirmed=False)
        confirmation["confirmed"] = True
        payload = self._request("/v2/thought-experiments/confirm", confirmation)
        if set(payload) != {"schema_version", "status", "request_id", "session_id", "spec", "run", "model"}:
            raise VisualClientError("The visual run response contains unexpected fields")
        if payload["schema_version"] != RUN_RESPONSE_VERSION or payload["status"] != "completed":
            raise VisualClientError("The visual run response state is invalid")
        returned_spec = _spec(payload["spec"])
        if returned_spec != spec or payload["model"] != planned["model"]:
            raise VisualClientError("The visual run changed the confirmed spec or model")
        _model(payload["model"])
        try:
            payload["run"] = validate_thought_experiment_run(payload["run"], returned_spec)
        except (TypeError, ValueError) as exc:
            raise VisualClientError(f"The visual run failed direct validation: {exc}") from exc
        if payload["run"]["execution"]["code_version"] != CODE_VERSION:
            raise VisualClientError("The visual run code pin does not match")
        return payload

    def _request(self, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint.rstrip("/") + path,
            data=None if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode(),
            headers={"Authorization": "Bearer " + self.token, "Accept": "application/json", "Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode()).get("detail", "request rejected")
            except Exception:
                detail = "request rejected"
            raise VisualClientError(f"Visual experiment request failed (HTTP {exc.code}): {detail}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise VisualClientError(f"Visual experiment request failed closed ({type(exc).__name__})") from exc
        if not isinstance(value, dict):
            raise VisualClientError("Visual experiment response must be an object")
        return value


def _spec(value: Any) -> dict[str, Any]:
    try:
        return validate_experiment_spec(value)
    except (TypeError, ValueError) as exc:
        raise VisualClientError(f"The visual spec failed direct validation: {exc}") from exc


def _model(value: Any) -> None:
    if value != EXPECTED_MODEL:
        raise VisualClientError("The visual compiler model identity does not match the reviewed pins")


def _confirmation(value: Any, payload: Mapping[str, Any], *, confirmed: bool) -> None:
    spec = payload["spec"]
    fields = {"schema_version", "request_id", "session_id", "spec_id", "spec_sha256", "confirmation_token", "confirmed"}
    if not isinstance(value, dict) or set(value) != fields:
        raise VisualClientError("The v2 confirmation fields are invalid")
    expected = {
        "schema_version": CONFIRMATION_VERSION,
        "request_id": payload["request_id"],
        "session_id": payload["session_id"],
        "spec_id": spec["spec_id"],
        "spec_sha256": spec["spec_sha256"],
        "confirmation_token": value["confirmation_token"],
        "confirmed": confirmed,
    }
    if value != expected or not isinstance(value["confirmation_token"], str) or len(value["confirmation_token"]) != 64:
        raise VisualClientError("The v2 confirmation does not bind the exact spec and session")


assert EXPERIMENT_SPEC_SCHEMA_SHA256 == SPEC_SCHEMA_SHA256
assert THOUGHT_EXPERIMENT_RUN_SCHEMA_SHA256 == RUN_SCHEMA_SHA256
