"""Pluggable A/B/C/C-prime behavioral baseline request interfaces."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .canonical import sha256_json
from .simulator import Law, pairwise_forces

CONDITIONS = ("A", "B", "C", "C-prime")
BASELINE_SCHEMA_VERSION = "jump.track-h-baselines/v1"
_ADAPTER_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_BACKEND_ADAPTERS: dict[str, Callable[[dict[str, Any]], "BaselineBackend"]] = {}

ANSWER_INSTRUCTION = """Return JSON only with exactly these fields:
partition: six binary integers (both labels allowed);
replacement_law: {same: attract|repel, different: attract|repel, exponent: integer};
adequacy: boolean for whether the stated prior law fits the observations;
force_prediction: an object keyed by the requested integer horizons, each with six [fx,fy] vectors;
confidence: number in [0,1]."""


@dataclass(frozen=True)
class BaselineRequest:
    condition: str
    episode_id: str
    prompt: str
    media: tuple[str, ...]
    target_horizons: tuple[int, ...]
    lexical_token_count: int
    lexical_token_budget: int

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unsupported baseline condition {self.condition!r}")
        if self.lexical_token_count > self.lexical_token_budget:
            raise ValueError("prompt exceeds its lexical token budget")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "jump.track-h-baseline-request/v1",
            "condition": self.condition,
            "episode_id": self.episode_id,
            "prompt": self.prompt,
            "media": list(self.media),
            "target_horizons": list(self.target_horizons),
            "lexical_token_count": self.lexical_token_count,
            "lexical_token_budget": self.lexical_token_budget,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.as_dict())


class BaselineBackend(Protocol):
    @property
    def identity(self) -> dict[str, str]: ...

    def generate(self, request: BaselineRequest) -> Any: ...


class ContractStubBackend:
    """Offline plumbing backend. Its malformed response is never model evidence."""

    identity = {
        "backend_id": "contract-stub",
        "backend_kind": "pipeline_contract_only",
        "model_id": "none",
        "model_revision": "not-applicable",
        "tokenizer_revision": "not-applicable",
        "license": "not-applicable",
    }

    def generate(self, request: BaselineRequest) -> dict[str, str]:
        return {"status": "not_executed", "request_sha256": request.sha256}


class ReplayBackend:
    """Replay externally captured model responses bound to exact request hashes."""

    def __init__(self, path: str | Path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != "jump.track-h-replay/v1":
            raise ValueError("replay file has an unsupported schema_version")
        identity = payload.get("backend")
        required = {
            "backend_id", "backend_kind", "model_id", "model_revision", "tokenizer_revision", "license"
        }
        if not isinstance(identity, dict) or set(identity) != required:
            raise ValueError(f"replay backend identity must contain exactly {sorted(required)}")
        if identity["backend_kind"] != "model_replay":
            raise ValueError("replay backend_kind must be model_replay")
        for field in required:
            if not isinstance(identity[field], str) or not identity[field].strip():
                raise ValueError(f"replay backend {field} must be nonempty")
        if identity["model_revision"].lower() in {"main", "master", "latest", "head"}:
            raise ValueError("replay model_revision must be immutable")
        if identity["tokenizer_revision"].lower() in {"main", "master", "latest", "head"}:
            raise ValueError("replay tokenizer_revision must be immutable")
        self._identity = identity
        responses = payload.get("responses")
        if not isinstance(responses, list):
            raise ValueError("replay responses must be an array")
        self._responses: dict[tuple[str, str], dict[str, Any]] = {}
        for item in responses:
            if not isinstance(item, dict) or set(item) != {
                "condition", "episode_id", "request_sha256", "response"
            }:
                raise ValueError("each replay response has invalid fields")
            key = item["condition"], item["episode_id"]
            if key in self._responses:
                raise ValueError(f"duplicate replay response {key}")
            self._responses[key] = item

    @property
    def identity(self) -> dict[str, str]:
        return dict(self._identity)

    def generate(self, request: BaselineRequest) -> Any:
        key = request.condition, request.episode_id
        if key not in self._responses:
            raise ValueError(f"replay is missing response {key}")
        item = self._responses[key]
        if item["request_sha256"] != request.sha256:
            raise ValueError(f"replay request hash mismatch for {key}")
        return item["response"]


def register_backend_adapter(
    adapter_id: str, factory: Callable[[dict[str, Any]], BaselineBackend]
) -> None:
    """Register one server-owned model adapter; request data cannot import code."""
    if not isinstance(adapter_id, str) or _ADAPTER_ID.fullmatch(adapter_id) is None:
        raise ValueError("adapter_id must be a lowercase server identifier")
    if not callable(factory):
        raise ValueError("backend adapter factory must be callable")
    if adapter_id in _BACKEND_ADAPTERS:
        raise ValueError(f"backend adapter already registered: {adapter_id}")
    _BACKEND_ADAPTERS[adapter_id] = factory


def _validate_live_backend(backend: Any) -> BaselineBackend:
    identity = getattr(backend, "identity", None)
    if not isinstance(identity, dict) or identity.get("backend_kind") != "live_model":
        raise ValueError("registered backend must expose a live_model identity")
    for field in ("backend_id", "model_id", "model_revision", "tokenizer_revision", "license"):
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"registered backend identity is missing {field}")
    if identity["model_revision"].lower() in {"main", "master", "latest", "head"}:
        raise ValueError("registered backend model_revision must be immutable")
    if identity["tokenizer_revision"].lower() in {"main", "master", "latest", "head"}:
        raise ValueError("registered backend tokenizer_revision must be immutable")
    if not callable(getattr(backend, "generate", None)):
        raise ValueError("registered backend must define generate(request)")
    return backend


def load_backend(spec: dict[str, Any]) -> BaselineBackend:
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise ValueError("backend spec requires a type")
    backend_type = spec["type"]
    if backend_type == "contract_stub":
        if set(spec) != {"type"}:
            raise ValueError("contract_stub backend accepts no additional fields")
        return ContractStubBackend()
    if backend_type == "replay":
        if set(spec) != {"type", "path"}:
            raise ValueError("replay backend requires exactly type and path")
        return ReplayBackend(spec["path"])
    if backend_type == "registered":
        required = {"type", "adapter_id", "config"}
        if set(spec) != required or not isinstance(spec["config"], dict):
            raise ValueError(f"registered backend requires exactly {sorted(required)}")
        adapter_id = spec["adapter_id"]
        if not isinstance(adapter_id, str) or adapter_id not in _BACKEND_ADAPTERS:
            raise ValueError("registered backend adapter_id is not server-allowlisted")
        return _validate_live_backend(_BACKEND_ADAPTERS[adapter_id](dict(spec["config"])))
    raise ValueError(f"unsupported backend type {backend_type!r}")


def _state_text(episode: dict[str, Any]) -> str:
    lines = []
    for frame in [episode["initial_state"], *episode["observations"]]:
        states = []
        for index, object_id in enumerate(episode["object_ids"]):
            position, velocity = frame["positions"][index], frame["velocities"][index]
            states.append(
                f"{object_id}:p=({position[0]:.6f},{position[1]:.6f}),v=({velocity[0]:.6f},{velocity[1]:.6f})"
            )
        lines.append(f"step={frame['step']} " + "; ".join(states))
    return "\n".join(lines)


def _counterfactual_diagnostics(episode: dict[str, Any], exponents: list[int]) -> str:
    config = episode["simulator"]
    lines = []
    observed = [episode["initial_state"], *episode["observations"]]
    for same in ("attract", "repel"):
        for different in ("attract", "repel"):
            for exponent in exponents:
                law = Law(same, different, exponent)
                candidates: list[tuple[float, tuple[int, ...]]] = []
                for mask in range(1, 32):
                    partition = (0,) + tuple((mask >> index) & 1 for index in range(5))
                    squared_errors: list[float] = []
                    for left, right in zip(observed, observed[1:]):
                        forces = pairwise_forces(
                            left["positions"],
                            partition,
                            law,
                            coefficient=float(config["coefficient"]),
                            softening=float(config["softening"]),
                        )
                        for index in range(6):
                            predicted_v = [
                                left["velocities"][index][axis]
                                + float(config["dt"]) * forces[index][axis]
                                for axis in (0, 1)
                            ]
                            squared_errors.extend(
                                (predicted_v[axis] - right["velocities"][index][axis]) ** 2
                                for axis in (0, 1)
                            )
                    candidates.append((math.sqrt(sum(squared_errors) / len(squared_errors)), partition))
                rmse, best_partition = min(candidates)
                lines.append(
                    f"candidate same={same} different={different} exponent={exponent} "
                    f"best_partition={''.join(map(str, best_partition))} velocity_rmse={rmse:.9f}"
                )
    return "\n".join(lines)


def _cap_words(text: str, budget: int) -> tuple[str, int]:
    words = text.split()
    if len(words) > budget:
        words = words[:budget]
    return " ".join(words), len(words)


def build_request(
    episode: dict[str, Any],
    condition: str,
    *,
    exponents: list[int],
    lexical_token_budget: int,
) -> BaselineRequest:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported baseline condition {condition!r}")
    if isinstance(lexical_token_budget, bool) or not isinstance(lexical_token_budget, int) or lexical_token_budget < 128:
        raise ValueError("lexical_token_budget must be an integer >= 128")
    intro = (
        "Analyze the six labeled objects. The stated prior law is "
        f"same={episode['prior_law']['same']}, different={episode['prior_law']['different']}, "
        f"exponent={episode['prior_law']['exponent']}. "
        f"Requested force horizons: {','.join(sorted(episode['target']['force_prediction'], key=int))}."
    )
    media: tuple[str, ...] = ()
    if condition in {"A", "B"}:
        media = tuple(
            f"renders/{episode['episode_id']}/step-{frame['step']:03d}.svg"
            for frame in episode["observations"]
        )
        evidence = "Use only the attached rendered observation frames."
        if condition == "B":
            evidence += (
                " Ontology hint: objects may have two hidden binary types; pair-force sign may depend on whether types match."
            )
    else:
        evidence = "Observed coordinate/state trace:\n" + _state_text(episode)
        if condition == "C-prime":
            evidence += (
                "\nSimulator-as-tool counterfactual results "
                "(all 31 canonical partitions were searched; these are ordinary text tool results):\n"
                + _counterfactual_diagnostics(episode, exponents)
            )
    prompt, count = _cap_words(f"{intro}\n{evidence}\n{ANSWER_INSTRUCTION}", lexical_token_budget)
    return BaselineRequest(
        condition=condition,
        episode_id=episode["episode_id"],
        prompt=prompt,
        media=media,
        target_horizons=tuple(sorted(map(int, episode["target"]["force_prediction"]))),
        lexical_token_count=count,
        lexical_token_budget=lexical_token_budget,
    )
