"""Server-side plan/confirm/run flow for the JUMP Space.

The planner consumes the canonical ExperimentSpec API directly. User text is
inert input to that planner and never becomes a model or system prompt. The
default backend has no fallback: until an authentic Stage D endpoint is wired,
confirmation ends in a clear unavailable result.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
import struct
from typing import Any, Protocol
import urllib.error
import urllib.request
import uuid

from jump_benchmark.canonical import sha256_json
from jump_benchmark.experiment_spec import (
    EXPERIMENT_SPEC_CONTRACT_SHA256,
    INTENT_SCHEMA_VERSION,
    build_planned_run,
    compile_experiment_intent,
    validate_experiment_run,
    validate_experiment_spec,
)
from jump_contracts import (
    build_learned_latent_evidence,
    learned_decoder_identity,
    open_result_envelope,
    seal_learned_latent_result,
    validate_learned_latent_evidence,
    verify_decoded_image_bytes,
    verify_latent_tensor_bytes,
)

EXPECTED_EXPERIMENT_SPEC_CONTRACT_SHA256 = (
    "6c35ea47f3c3ad614cfb053c37b1670764a9584bbe7e2706b793f5d8f5635ad6"
)
STAGE_D_ENDPOINT = (
    "https://ameymuke252003--jump-sequential-experiments-authentic-st-f0acae.modal.run/v1/experiment"
)
STAGE_D_CODE_VERSION = "9af95576423c11eeb48a628c6ddd1e1d81da5d84"
STAGE_D_MANIFEST_SHA256 = "09b6a0378af6073fd858ca2f1e86537771a96525d83e6f10e2f4813b63086406"
STAGE_D_CHECKPOINT_ID = "stage-d-13c3d963b9ec7171f5d138a9e737b4b6294d542d0887dfbf9a52c2efba422071"
STAGE_D_COMPONENT_MANIFEST_SHA256 = "04f4b4ea6c7f4e6d517cd5a27925ed948bcde1d5d744f84edcf1ee8cdbe890bb"
STAGE_D_REPOSITORY_REVISION = "d197b3825a37e95dfa7d50144fab3c18b6a7fd39"


class ExperimentFlowError(RuntimeError):
    """A bounded experiment cannot safely advance to the next UI state."""


class ExperimentBackend(Protocol):
    label: str
    is_live: bool

    def execute(
        self, planned_run: dict[str, Any], *, intent: str | None = None
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LiveExperimentBackend:
    """Authenticated client for the pinned authentic Stage D engineering endpoint."""

    endpoint: str = STAGE_D_ENDPOINT
    token: str | None = None
    timeout_seconds: float = 180.0
    label: str = "LIVE ENGINEERING RUN · STAGE D NULL"
    is_live: bool = True

    def execute(
        self, planned_run: dict[str, Any], *, intent: str | None = None
    ) -> dict[str, Any]:
        planned = validate_experiment_run(planned_run)
        token = self.token if self.token is not None else os.environ.get("JUMP_MODAL_TOKEN", "")
        if not token:
            raise ExperimentFlowError(
                "The authentic endpoint is not enabled. No recorded result was substituted."
            )
        if not isinstance(intent, str):
            raise ExperimentFlowError("The confirmed intent is required for the authentic endpoint")
        request = {
            "schema_version": INTENT_SCHEMA_VERSION,
            "intent": intent,
            "session_id": "ui-" + planned["request_id"][:20],
            "seed": planned["plan"]["seed"],
            "max_steps": planned["plan"]["observed_steps"],
        }
        body = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ExperimentFlowError(
                f"The authentic endpoint returned HTTP {exc.code}. No result was substituted."
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExperimentFlowError(
                f"The authentic endpoint failed closed ({type(exc).__name__}). No result was substituted."
            ) from exc
        try:
            completed = validate_experiment_run(payload)
        except ValueError as exc:
            raise ExperimentFlowError(f"The authentic endpoint returned invalid evidence: {exc}") from exc
        _verify_live_pins(completed)
        return completed


@dataclass(frozen=True)
class NonLiveContractFixtureBackend:
    """Deterministic visual-QA fixture; never deployed as the live backend."""

    label: str = "NON-LIVE CONTRACT FIXTURE"
    is_live: bool = False

    def execute(
        self, planned_run: dict[str, Any], *, intent: str | None = None
    ) -> dict[str, Any]:
        planned = validate_experiment_run(planned_run)
        if planned["status"] != "planned":
            raise ExperimentFlowError("Only a confirmed planned run can execute")
        return _fixture_completed_run(planned)


def plan_experiment(
    intent: str,
    *,
    session_id: str,
    seed: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Compile free text to the exact canonical plan and stop for confirmation."""
    if EXPERIMENT_SPEC_CONTRACT_SHA256 != EXPECTED_EXPERIMENT_SPEC_CONTRACT_SHA256:
        raise ExperimentFlowError("ExperimentSpec contract identity changed")
    try:
        plan = compile_experiment_intent(
            {
                "schema_version": INTENT_SCHEMA_VERSION,
                "intent": intent,
                "session_id": session_id,
                "seed": seed,
                "max_steps": max_steps,
            }
        )
        return build_planned_run(request_id="req-" + uuid.uuid4().hex[:20], plan=plan)
    except ValueError as exc:
        raise ExperimentFlowError(str(exc)) from exc


def run_confirmed_experiment(
    planned_run: dict[str, Any], *, backend: ExperimentBackend, intent: str | None = None
) -> dict[str, Any]:
    """Execute exactly the plan the user confirmed; never recompile or substitute."""
    try:
        planned = validate_experiment_run(planned_run)
    except ValueError as exc:
        raise ExperimentFlowError(f"The confirmed plan failed validation: {exc}") from exc
    if planned["status"] != "planned" or planned["result"] is not None:
        raise ExperimentFlowError("The confirmed plan is not in the planned state")
    completed = backend.execute(planned, intent=intent)
    try:
        validated = validate_experiment_run(completed)
    except ValueError as exc:
        raise ExperimentFlowError(f"The experiment result failed validation: {exc}") from exc
    if validated["plan"] != planned["plan"]:
        raise ExperimentFlowError("The live endpoint executed a different plan than the confirmed plan")
    return validated


def _verify_live_pins(run: dict[str, Any]) -> None:
    result = run["result"]
    envelope = result["sealed_result"]
    provenance = envelope["provenance"]
    if provenance != {
        "manifest_sha256": STAGE_D_MANIFEST_SHA256,
        "run_id": provenance.get("run_id"),
        "code_version": STAGE_D_CODE_VERSION,
        "checkpoint_id": STAGE_D_CHECKPOINT_ID,
    }:
        raise ExperimentFlowError("The authentic endpoint provenance does not match the pinned Stage D run")
    evidence = validate_learned_latent_evidence(
        open_result_envelope(
            envelope,
            expected_source="live",
            expected_manifest_sha256=STAGE_D_MANIFEST_SHA256,
            expected_checkpoint_id=STAGE_D_CHECKPOINT_ID,
        )
    )
    bindings = evidence["answer"].get("producer_bindings")
    if not isinstance(bindings, dict):
        raise ExperimentFlowError("The authentic endpoint omitted producer bindings")
    expected = {
        "canonical_repo_revision": STAGE_D_REPOSITORY_REVISION,
        "component_manifest_sha256": STAGE_D_COMPONENT_MANIFEST_SHA256,
        "engineering_only": True,
        "live_ready": False,
    }
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ExperimentFlowError("The authentic endpoint component identities or readiness flags changed")
    claim = str(bindings.get("claim_label", "")).lower()
    if "stage d null" not in claim or "no informative-z" not in claim:
        raise ExperimentFlowError("The authentic endpoint claim label softened the Stage D null result")


def verified_result(run: dict[str, Any]) -> dict[str, Any]:
    """Open a validated result and verify image bytes and same-z evidence again."""
    completed = validate_experiment_run(run)
    if completed["status"] != "completed":
        raise ExperimentFlowError("The experiment has no completed result")
    result = completed["result"]
    evidence = validate_learned_latent_evidence(
        open_result_envelope(result["sealed_result"], expected_source="live")
    )
    image = result["decoded_image"]
    try:
        image_bytes = base64.b64decode(image["data"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ExperimentFlowError("The learned-decoder image transport is invalid") from exc
    verify_decoded_image_bytes(evidence, image_bytes)
    tensor = evidence["tensor"]
    if len(
        {
            tensor["world_latent_sha256"],
            tensor["encoder_output_sha256"],
            tensor["decoder_input_sha256"],
            tensor["injection_input_sha256"],
            evidence["answer_binding"]["world_latent_sha256"],
            evidence["answer_binding"]["injection_input_sha256"],
            evidence["decoded_observation"]["world_latent_sha256"],
        }
    ) != 1:
        raise ExperimentFlowError("The result does not bind one learned z to every consumer")
    return {
        "run": completed,
        "evidence": evidence,
        "image_bytes": image_bytes,
        "presentation": result["presentation"],
    }


def _fixture_completed_run(planned: dict[str, Any]) -> dict[str, Any]:
    plan = validate_experiment_spec(planned["plan"])
    raw_z = struct.pack("<4f", 0.25, -1.5, 2.0, 0.0)
    image_bytes = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 360">'
        b'<rect width="720" height="360" fill="#f4efe4"/>'
        b'<path d="M90 245 C190 100 335 300 475 125 S650 105 675 190" '
        b'fill="none" stroke="#3154a5" stroke-width="5" stroke-dasharray="10 10"/>'
        b'<g fill="#e65332" stroke="#171610" stroke-width="4">'
        b'<circle cx="105" cy="230" r="17"/><circle cx="215" cy="155" r="17"/>'
        b'<circle cx="325" cy="235" r="17"/><circle cx="440" cy="150" r="17"/>'
        b'<circle cx="555" cy="115" r="17"/><circle cx="650" cy="190" r="17"/>'
        b'</g><text x="28" y="42" font-family="monospace" font-size="18" fill="#171610">'
        b'NON-LIVE CONTRACT FIXTURE \xe2\x80\x94 predicted from fixture z</text></svg>'
    )
    answer = {
        "schema_version": "jump.ui-fixture-answer/v1",
        "partition": [0, 0, 1, 1, 0, 1],
        "replacement_law": {"same": "attract", "different": "repel", "exponent": 2},
        "producer_bindings": {
            "experiment_id": plan["experiment_id"],
            "experiment_spec_sha256": sha256_json(plan),
        },
    }
    evidence = build_learned_latent_evidence(
        encoder_output=raw_z,
        decoder_input=raw_z,
        injection_input=raw_z,
        encoder_observation=b"non-live observation-only contract fixture",
        encoder_observation_artifact_name="fixture-observation.bin",
        encoder_observation_media_type="application/octet-stream",
        dtype="float32-le",
        shape=[4],
        order="C",
        tensor_artifact_name="fixture-world-latent.f32le.bin",
        recipient_world_id="fixture-world-a",
        world_pair_id="fixture-pair",
        learned_decoder=learned_decoder_identity(
            artifact_name="fixture-decoder.safetensors",
            artifact_sha256="d" * 64,
            training_manifest_sha256="e" * 64,
            code_version="fixture-only",
            architecture="fixture-latent-to-observation",
        ),
        decoded_image=image_bytes,
        decoded_image_media_type="image/svg+xml",
        answer=answer,
    )
    # Exercise the byte verifier before this fixture reaches the view.
    verify_latent_tensor_bytes(evidence, raw_z)
    verify_decoded_image_bytes(evidence, image_bytes)
    sealed = seal_learned_latent_result(
        evidence,
        source="live",
        manifest_sha256="a" * 64,
        run_id="fixture-" + planned["request_id"],
        code_version="fixture-only",
        checkpoint_id="fixture-only",
    )
    correctness = {
        "format_valid": True,
        "exact_correct": False,
        "partition_correct": True,
        "law_correct": False,
        "adequacy_correct": False,
        "force_score": None,
        "notes": "Fixture output is well formed, but intentionally not an exact answer.",
    }
    result = {
        "experiment_id": plan["experiment_id"],
        "experiment_spec_sha256": sha256_json(plan),
        "sealed_result": sealed,
        "decoded_image": {
            "artifact_name": "predicted-from-z.svg",
            "media_type": "image/svg+xml",
            "encoding": "base64",
            "data": base64.b64encode(image_bytes).decode("ascii"),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        },
        "presentation": {
            "world_built": "A deterministic six-object fixture world was built from the confirmed plan.",
            "model_prediction": answer,
            "what_changed": "The fixture changed the interaction rule selected by the parsed plan.",
            "correctness": correctness,
        },
    }
    return {
        "schema_version": "jump.experiment-run/v1",
        "status": "completed",
        "live": True,
        "request_id": planned["request_id"],
        "plan": plan,
        "result": result,
        "error": None,
    }


def technical_details(run: dict[str, Any], *, backend_label: str) -> str:
    checked = verified_result(run)
    evidence = checked["evidence"]
    tensor = evidence["tensor"]
    decoder = evidence["learned_decoder"]
    bindings = evidence["answer"].get("producer_bindings", {})
    envelope = run["result"]["sealed_result"]
    return json.dumps(
        {
            "mode": backend_label,
            "experiment_spec_contract_sha256": EXPERIMENT_SPEC_CONTRACT_SHA256,
            "experiment_id": run["plan"]["experiment_id"],
            "run_id": envelope["provenance"]["run_id"],
            "checkpoint_id": envelope["provenance"]["checkpoint_id"],
            "canonical_repo_revision": bindings.get("canonical_repo_revision"),
            "component_manifest_sha256": bindings.get("component_manifest_sha256"),
            "engineering_only": bindings.get("engineering_only"),
            "live_ready": bindings.get("live_ready"),
            "world_latent_sha256": tensor["world_latent_sha256"],
            "encoder_output_sha256": tensor["encoder_output_sha256"],
            "decoder_input_sha256": tensor["decoder_input_sha256"],
            "injection_input_sha256": tensor["injection_input_sha256"],
            "learned_decoder_artifact_sha256": decoder["artifact_sha256"],
            "learned_decoder_training_manifest_sha256": decoder["training_manifest_sha256"],
            "decoded_image_sha256": evidence["decoded_observation"]["image_sha256"],
        },
        indent=2,
        sort_keys=True,
    )
