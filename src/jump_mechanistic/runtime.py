"""Config-driven Gemma-style residual capture and activation patching.

The implementation is framework-light but follows PyTorch's forward-hook
contract. A live Transformers/PEFT model is resolved through ``named_modules``;
no concrete Gemma hook path is guessed in code.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from jump_contracts import (
    EvidenceError,
    validate_world_model_component_manifest,
    validate_world_model_load_record,
)

from .boundaries import ResolvedPrompt, require_matched_prompt_lengths
from .capture import Timepoint

PRIMARY_CONFIRMATORY_LAYER_INDICES = (7, 15, 23, 31, 39, 47)
PRIMARY_CONFIRMATORY_MODULE_NAMES = tuple(
    f"model.layers.{index}" for index in PRIMARY_CONFIRMATORY_LAYER_INDICES
)


class PatchMode(str, Enum):
    DENOISING = "denoising"  # corrupt prompt, clean activation restored
    NOISING = "noising"  # clean prompt, corrupt activation inserted


@dataclass(frozen=True)
class HookSite:
    site_id: str
    module_name: str
    module_type: str
    timepoint: Timepoint
    expected_rank: int
    expected_hidden_size: int
    output_selector: str = "tensor"

    def __post_init__(self) -> None:
        if self.output_selector not in {"tensor", "first"}:
            raise ValueError("output_selector must be tensor or first")
        if any(not isinstance(getattr(self, key), str) or not getattr(self, key) for key in ("site_id", "module_name", "module_type")):
            raise ValueError("hook site identifiers must be nonempty strings")
        if self.expected_rank != 3 or not isinstance(self.expected_hidden_size, int) or self.expected_hidden_size <= 0:
            raise ValueError("residual hook sites require rank 3 and a positive hidden size")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HookSite":
        required = {
            "site_id", "module_name", "module_type", "timepoint",
            "expected_rank", "expected_hidden_size", "output_selector",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"hook site must contain exactly {sorted(required)}")
        if value["output_selector"] not in {"tensor", "first"}:
            raise ValueError("output_selector must be tensor or first")
        site = cls(
            site_id=value["site_id"],
            module_name=value["module_name"],
            module_type=value["module_type"],
            timepoint=Timepoint(value["timepoint"]),
            expected_rank=value["expected_rank"],
            expected_hidden_size=value["expected_hidden_size"],
            output_selector=value["output_selector"],
        )
        if any(not isinstance(getattr(site, key), str) or not getattr(site, key) for key in ("site_id", "module_name", "module_type")):
            raise ValueError("hook site identifiers must be nonempty strings")
        if site.expected_rank != 3 or not isinstance(site.expected_hidden_size, int) or site.expected_hidden_size <= 0:
            raise ValueError("residual hook sites require rank 3 and a positive hidden size")
        return site


@dataclass(frozen=True)
class HookManifest:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    peft_base_model: bool
    study_scope: str
    sites: tuple[HookSite, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, field), str) or not getattr(self, field) for field in ("model_id", "model_revision", "tokenizer_revision")):
            raise ValueError("model/tokenizer identities must be immutable nonempty strings")
        if not isinstance(self.peft_base_model, bool):
            raise ValueError("peft_base_model must be Boolean")
        if self.study_scope not in {"synthetic", "confirmatory_primary", "confirmatory_secondary"}:
            raise ValueError("study_scope must be synthetic, confirmatory_primary, or confirmatory_secondary")
        if not isinstance(self.sites, tuple) or any(not isinstance(site, HookSite) for site in self.sites):
            raise ValueError("sites must be a tuple of HookSite records")
        keys = [(site.site_id, site.timepoint) for site in self.sites]
        if not self.sites or len(keys) != len(set(keys)):
            raise ValueError("hook manifest must have nonempty unique site/timepoint nodes")
        if self.study_scope == "confirmatory_primary":
            expected = {(module, point) for module in PRIMARY_CONFIRMATORY_MODULE_NAMES for point in Timepoint}
            actual = {(site.module_name, site.timepoint) for site in self.sites}
            if actual != expected or len(self.sites) != len(expected):
                raise ValueError(
                    "primary confirmatory hook map must contain exactly layers "
                    f"{PRIMARY_CONFIRMATORY_LAYER_INDICES} at T0--T4"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HookManifest":
        required = {"model_id", "model_revision", "tokenizer_revision", "peft_base_model", "study_scope", "sites"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"hook manifest must contain exactly {sorted(required)}")
        if not isinstance(value["sites"], list) or not value["sites"]:
            raise ValueError("hook manifest sites must be a nonempty array")
        manifest = cls(
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            tokenizer_revision=value["tokenizer_revision"],
            peft_base_model=value["peft_base_model"],
            study_scope=value["study_scope"],
            sites=tuple(HookSite.from_dict(site) for site in value["sites"]),
        )
        if any(not isinstance(getattr(manifest, field), str) or not getattr(manifest, field) for field in ("model_id", "model_revision", "tokenizer_revision")):
            raise ValueError("model/tokenizer identities must be immutable nonempty strings")
        if not isinstance(manifest.peft_base_model, bool):
            raise ValueError("peft_base_model must be Boolean")
        if manifest.study_scope not in {"synthetic", "confirmatory_primary", "confirmatory_secondary"}:
            raise ValueError("study_scope must be synthetic, confirmatory_primary, or confirmatory_secondary")
        keys = [(site.site_id, site.timepoint) for site in manifest.sites]
        if len(keys) != len(set(keys)):
            raise ValueError("hook manifest contains duplicate site/timepoint nodes")
        if manifest.study_scope == "confirmatory_primary":
            expected = {(module, point) for module in PRIMARY_CONFIRMATORY_MODULE_NAMES for point in Timepoint}
            actual = {(site.module_name, site.timepoint) for site in manifest.sites}
            if actual != expected or len(manifest.sites) != len(expected):
                raise ValueError(
                    "primary confirmatory hook map must contain exactly layers "
                    f"{PRIMARY_CONFIRMATORY_LAYER_INDICES} at T0--T4"
                )
        return manifest

    def select(self, site_id: str, timepoint: Timepoint | str) -> HookSite:
        point = Timepoint(timepoint)
        matches = [site for site in self.sites if site.site_id == site_id and site.timepoint == point]
        if len(matches) != 1:
            raise PermissionError(f"manifest must select exactly one fixed node: {site_id}/{point.value}")
        return matches[0]


@dataclass(frozen=True)
class WorldModelRuntimeBinding:
    """Validated identity of the exact shared world-model distribution loaded."""

    component_manifest_sha256: str
    load_record_sha256: str
    repository_revision: str
    base_model_repo_id: str
    base_model_revision: str
    gemma_adapter_identity_sha256: str

    @classmethod
    def from_contracts(
        cls,
        component_manifest: Mapping[str, Any],
        load_record: Mapping[str, Any],
    ) -> "WorldModelRuntimeBinding":
        manifest = validate_world_model_component_manifest(component_manifest)
        record = validate_world_model_load_record(
            load_record,
            manifest,
            expected_mode="gated_gemma",
        )
        if manifest["status"]["live_ready"] is not True:
            raise EvidenceError(
                "confirmatory mechanistic runtime requires a live-ready world model manifest"
            )
        return cls(
            component_manifest_sha256=manifest["manifest_sha256"],
            load_record_sha256=record["load_record_sha256"],
            repository_revision=record["repository"]["resolved_revision"],
            base_model_repo_id=manifest["base_model"]["repo_id"],
            base_model_revision=manifest["base_model"]["revision"],
            gemma_adapter_identity_sha256=manifest["components"]["gemma_adapter"][
                "identity_sha256"
            ],
        )


@dataclass(frozen=True)
class ActivationProvenance:
    schema_version: str
    episode_id: str
    run_kind: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    prompt_sha256: str
    hook_site_id: str
    hook_name: str
    hook_module_type: str
    hook_output_shape: tuple[int, ...]
    captured_shape: tuple[int, ...]
    token_position: int
    timepoint: Timepoint
    dtype: str
    device: str
    world_model_manifest_sha256: str | None
    world_model_load_record_sha256: str | None
    world_model_repository_revision: str | None
    gemma_adapter_identity_sha256: str | None
    activation_sha256: str
    content_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "run_kind": self.run_kind,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "prompt_sha256": self.prompt_sha256,
            "hook_site_id": self.hook_site_id,
            "hook_name": self.hook_name,
            "hook_module_type": self.hook_module_type,
            "hook_output_shape": list(self.hook_output_shape),
            "captured_shape": list(self.captured_shape),
            "token_position": self.token_position,
            "timepoint": self.timepoint.value,
            "dtype": self.dtype,
            "device": self.device,
            "world_model_manifest_sha256": self.world_model_manifest_sha256,
            "world_model_load_record_sha256": self.world_model_load_record_sha256,
            "world_model_repository_revision": self.world_model_repository_revision,
            "gemma_adapter_identity_sha256": self.gemma_adapter_identity_sha256,
            "activation_sha256": self.activation_sha256,
        }

    def verify(self) -> None:
        if _sha256(self.unsigned_dict()) != self.content_sha256:
            raise ValueError("immutable activation provenance hash mismatch")


@dataclass(frozen=True)
class ActivationSnapshot:
    values: tuple[Any, ...]
    provenance: ActivationProvenance

    def verify(self) -> None:
        self.provenance.verify()
        if _sha256(self.values) != self.provenance.activation_sha256:
            raise ValueError("activation values changed after capture")

    def to_dict(self) -> dict[str, Any]:
        self.verify()
        return {"values": _lists(self.values), "provenance": {**self.provenance.unsigned_dict(), "content_sha256": self.provenance.content_sha256}}


@dataclass(frozen=True)
class PatchRun:
    mode: PatchMode
    scale: float
    source: ActivationSnapshot
    source_prompt_sha256: str
    target_prompt_sha256: str
    site_id: str
    timepoint: Timepoint
    hook_name: str
    hook_shape: tuple[int, ...]
    dtype: str
    device: str
    output_sha256: str
    content_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "jump.activation-patch/v1",
            "mode": self.mode.value,
            "scale": self.scale,
            "source_activation_sha256": self.source.provenance.activation_sha256,
            "source_provenance_sha256": self.source.provenance.content_sha256,
            "source_prompt_sha256": self.source_prompt_sha256,
            "target_prompt_sha256": self.target_prompt_sha256,
            "site_id": self.site_id,
            "timepoint": self.timepoint.value,
            "hook_name": self.hook_name,
            "hook_shape": list(self.hook_shape),
            "dtype": self.dtype,
            "device": self.device,
            "output_sha256": self.output_sha256,
        }

    def verify(self) -> None:
        self.source.verify()
        if _sha256(self.unsigned_dict()) != self.content_sha256:
            raise ValueError("immutable patch provenance hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        self.verify()
        return {**self.unsigned_dict(), "content_sha256": self.content_sha256}


class MechanisticRuntime:
    def __init__(
        self,
        model: Any,
        manifest: HookManifest,
        *,
        world_model_manifest: Mapping[str, Any] | None = None,
        world_model_load_record: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.manifest = manifest
        if (world_model_manifest is None) != (world_model_load_record is None):
            raise EvidenceError(
                "world model component manifest and load record must be supplied together"
            )
        self.world_model_binding = (
            WorldModelRuntimeBinding.from_contracts(
                world_model_manifest,
                world_model_load_record,
            )
            if world_model_manifest is not None and world_model_load_record is not None
            else None
        )
        if manifest.study_scope != "synthetic":
            if self.world_model_binding is None:
                raise EvidenceError(
                    "confirmatory mechanistic runtime requires validated shared world model contracts"
                )
            if (
                manifest.model_id != self.world_model_binding.base_model_repo_id
                or manifest.model_revision != self.world_model_binding.base_model_revision
            ):
                raise EvidenceError(
                    "hook manifest base model identity does not match the shared world model contract"
                )
        if getattr(model, "training", False):
            raise RuntimeError("mechanistic runtime requires model.eval() before hook registration")
        self.hook_root = _base_model(model) if manifest.peft_base_model else model
        self._modules = dict(self.hook_root.named_modules())
        for site in manifest.sites:
            module = self._modules.get(site.module_name)
            if module is None:
                raise RuntimeError(f"hook map drift: missing module {site.module_name!r}")
            actual = type(module).__name__
            if actual != site.module_type:
                raise RuntimeError(
                    f"hook map drift: {site.module_name!r} expected {site.module_type}, got {actual}"
                )

    def capture(
        self,
        *,
        inputs: Mapping[str, Any],
        resolved_prompt: ResolvedPrompt,
        episode_id: str,
        site_id: str,
        timepoint: Timepoint | str,
        run_kind: str,
    ) -> tuple[Any, ActivationSnapshot]:
        resolved_prompt.verify()
        _validate_model_inputs(inputs, resolved_prompt)
        if resolved_prompt.tokenizer_revision != self.manifest.tokenizer_revision:
            raise ValueError("resolved prompt tokenizer revision does not match hook manifest")
        site = self.manifest.select(site_id, timepoint)
        position = resolved_prompt.positions[site.timepoint]
        captured: list[tuple[tuple[Any, ...], tuple[int, ...], str, str]] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = _select_output(output, site.output_selector)
            shape, dtype, device = _validate_tensor(tensor, site, len(resolved_prompt.input_ids))
            activation = tensor[:, position, :]
            values = _tuples(_tolist(_detach_clone(activation)))
            captured.append((values, shape, dtype, device))

        output = self._run_with_hook(site, hook, inputs)
        if len(captured) != 1:
            raise RuntimeError(
                f"hook {site.module_name!r} fired {len(captured)} times; this runtime supports "
                "exactly one teacher-forced full-sequence forward, not autoregressive generation. "
                "S3 generation requires a separately validated persistent-hook implementation"
            )
        values, output_shape, dtype, device = captured[0]
        activation_shape = _shape_from_values(values)
        binding = self.world_model_binding
        unsigned = {
            "schema_version": "jump.activation-provenance/v2",
            "episode_id": episode_id,
            "run_kind": run_kind,
            "model_id": self.manifest.model_id,
            "model_revision": self.manifest.model_revision,
            "tokenizer_revision": self.manifest.tokenizer_revision,
            "prompt_sha256": resolved_prompt.content_sha256,
            "hook_site_id": site.site_id,
            "hook_name": site.module_name,
            "hook_module_type": site.module_type,
            "hook_output_shape": list(output_shape),
            "captured_shape": list(activation_shape),
            "token_position": position,
            "timepoint": site.timepoint.value,
            "dtype": dtype,
            "device": device,
            "world_model_manifest_sha256": (
                binding.component_manifest_sha256 if binding else None
            ),
            "world_model_load_record_sha256": binding.load_record_sha256 if binding else None,
            "world_model_repository_revision": binding.repository_revision if binding else None,
            "gemma_adapter_identity_sha256": (
                binding.gemma_adapter_identity_sha256 if binding else None
            ),
            "activation_sha256": _sha256(values),
        }
        provenance = ActivationProvenance(
            schema_version=unsigned["schema_version"], episode_id=episode_id, run_kind=run_kind,
            model_id=self.manifest.model_id, model_revision=self.manifest.model_revision,
            tokenizer_revision=self.manifest.tokenizer_revision,
            prompt_sha256=resolved_prompt.content_sha256, hook_site_id=site.site_id,
            hook_name=site.module_name, hook_module_type=site.module_type,
            hook_output_shape=output_shape, captured_shape=activation_shape,
            token_position=position, timepoint=site.timepoint, dtype=dtype, device=device,
            world_model_manifest_sha256=unsigned["world_model_manifest_sha256"],
            world_model_load_record_sha256=unsigned["world_model_load_record_sha256"],
            world_model_repository_revision=unsigned["world_model_repository_revision"],
            gemma_adapter_identity_sha256=unsigned["gemma_adapter_identity_sha256"],
            activation_sha256=unsigned["activation_sha256"], content_sha256=_sha256(unsigned),
        )
        snapshot = ActivationSnapshot(values, provenance)
        snapshot.verify()
        return output, snapshot

    def patch(
        self,
        *,
        inputs: Mapping[str, Any],
        target_prompt: ResolvedPrompt,
        source_prompt: ResolvedPrompt,
        source: ActivationSnapshot,
        site_id: str,
        timepoint: Timepoint | str,
        mode: PatchMode | str,
        scale: float = 1.0,
    ) -> tuple[Any, PatchRun]:
        patch_mode = PatchMode(mode)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not float("-inf") < float(scale) < float("inf"):
            raise ValueError("patch scale must be finite")
        source.verify()
        require_matched_prompt_lengths(source_prompt, target_prompt)
        _validate_model_inputs(inputs, target_prompt)
        if source_prompt.tokenizer_revision != self.manifest.tokenizer_revision or target_prompt.tokenizer_revision != self.manifest.tokenizer_revision:
            raise ValueError("patch prompt tokenizer revision does not match hook manifest")
        if source.provenance.prompt_sha256 != source_prompt.content_sha256:
            raise ValueError("source activation provenance does not match source prompt")
        if (
            source.provenance.model_id != self.manifest.model_id
            or source.provenance.model_revision != self.manifest.model_revision
            or source.provenance.tokenizer_revision != self.manifest.tokenizer_revision
        ):
            raise ValueError("source activation model/tokenizer provenance does not match runtime")
        site = self.manifest.select(site_id, timepoint)
        if (source.provenance.hook_site_id, source.provenance.timepoint) != (site.site_id, site.timepoint):
            raise ValueError("source activation node does not match fixed manifest node")
        if patch_mode is PatchMode.DENOISING and source.provenance.run_kind != "clean":
            raise ValueError("denoising requires a clean activation source")
        if patch_mode is PatchMode.NOISING and source.provenance.run_kind != "corrupt":
            raise ValueError("noising requires a corrupt activation source")
        position = target_prompt.positions[site.timepoint]
        observed: list[tuple[tuple[int, ...], str, str]] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = _select_output(output, site.output_selector)
            shape, dtype, device = _validate_tensor(tensor, site, len(target_prompt.input_ids))
            if source.provenance.hook_output_shape != shape or source.provenance.captured_shape != (shape[0], shape[-1]):
                raise RuntimeError("patch source shape drifted from target activation")
            if (dtype, device) != (source.provenance.dtype, source.provenance.device):
                raise RuntimeError("patch source dtype/device drifted from target activation")
            replacement = _new_tensor(tensor, source.values)
            patched = _detach_clone(tensor)
            current = patched[:, position, :]
            patched[:, position, :] = current + (replacement - current) * float(scale)
            observed.append((shape, dtype, device))
            return _replace_output(output, patched, site.output_selector)

        output = self._run_with_hook(site, hook, inputs)
        if len(observed) != 1:
            raise RuntimeError(
                f"patch hook {site.module_name!r} fired {len(observed)} times; this runtime supports "
                "exactly one teacher-forced full-sequence forward, not autoregressive generation. "
                "S3 generation requires a separately validated persistent-hook implementation"
            )
        shape, dtype, device = observed[0]
        output_sha256 = _sha256(_hashable_output(output))
        unsigned = {
            "schema_version": "jump.activation-patch/v1",
            "mode": patch_mode.value, "scale": float(scale),
            "source_activation_sha256": source.provenance.activation_sha256,
            "source_provenance_sha256": source.provenance.content_sha256,
            "source_prompt_sha256": source_prompt.content_sha256,
            "target_prompt_sha256": target_prompt.content_sha256,
            "site_id": site.site_id, "timepoint": site.timepoint.value,
            "hook_name": site.module_name, "hook_shape": list(shape),
            "dtype": dtype, "device": device, "output_sha256": output_sha256,
        }
        patch_run = PatchRun(
            mode=patch_mode, scale=float(scale), source=source,
            source_prompt_sha256=source_prompt.content_sha256,
            target_prompt_sha256=target_prompt.content_sha256,
            site_id=site.site_id, timepoint=site.timepoint, hook_name=site.module_name,
            hook_shape=shape, dtype=dtype, device=device, output_sha256=output_sha256,
            content_sha256=_sha256(unsigned),
        )
        patch_run.verify()
        return output, patch_run

    def _run_with_hook(self, site: HookSite, hook: Callable[..., Any], inputs: Mapping[str, Any]) -> Any:
        handle = self._modules[site.module_name].register_forward_hook(hook)
        try:
            with _inference_context():
                return self.model(**dict(inputs))
        finally:
            handle.remove()


def normalized_logit_patch_score(
    *, clean_logit_difference: float, corrupt_logit_difference: float, patched_logit_difference: float
) -> dict[str, float]:
    """Return the Track R S3 metric with its clean/corrupt floor and ceiling."""
    values = (clean_logit_difference, corrupt_logit_difference, patched_logit_difference)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not float("-inf") < float(value) < float("inf") for value in values):
        raise ValueError("logit differences must be finite numeric values")
    denominator = float(clean_logit_difference) - float(corrupt_logit_difference)
    if abs(denominator) <= 1e-12:
        raise ValueError("clean/corrupt logit-difference floor and ceiling are indistinguishable")
    return {
        "normalized_patch_score": (float(patched_logit_difference) - float(corrupt_logit_difference)) / denominator,
        "clean_logit_difference": float(clean_logit_difference),
        "corrupt_logit_difference": float(corrupt_logit_difference),
        "patched_logit_difference": float(patched_logit_difference),
    }


def _base_model(model: Any) -> Any:
    getter = getattr(model, "get_base_model", None)
    if not callable(getter):
        raise RuntimeError("hook manifest requires a PEFT base model but get_base_model() is unavailable")
    base = getter()
    if base is model:
        raise RuntimeError("PEFT get_base_model() did not unwrap the model")
    return base


def _validate_model_inputs(inputs: Mapping[str, Any], prompt: ResolvedPrompt) -> None:
    if set(("input_ids", "attention_mask")) - set(inputs):
        raise ValueError("model inputs must contain input_ids and attention_mask")
    input_ids = _input_sequence(inputs["input_ids"], "input_ids")
    attention_mask = _input_sequence(inputs["attention_mask"], "attention_mask")
    if input_ids != prompt.input_ids or attention_mask != prompt.attention_mask:
        raise ValueError("model inputs do not match immutable resolved prompt IDs/mask")


def _input_sequence(value: Any, name: str) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"model {name} must be one batched integer sequence")
    return tuple(value)


def _select_output(output: Any, selector: str) -> Any:
    if selector == "tensor":
        return output
    if not isinstance(output, (tuple, list)) or not output:
        raise RuntimeError("hook output drift: expected a nonempty tuple/list")
    return output[0]


def _replace_output(output: Any, tensor: Any, selector: str) -> Any:
    if selector == "tensor":
        return tensor
    values = list(output)
    values[0] = tensor
    return tuple(values) if isinstance(output, tuple) else values


def _validate_tensor(tensor: Any, site: HookSite, prompt_length: int) -> tuple[tuple[int, ...], str, str]:
    shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
    if len(shape) != site.expected_rank or shape[-1] != site.expected_hidden_size:
        raise RuntimeError(
            f"hook shape drift at {site.module_name}: expected rank {site.expected_rank}/hidden "
            f"{site.expected_hidden_size}, got {shape}"
        )
    if shape[-2] != prompt_length:
        raise RuntimeError(f"hook sequence length {shape[-2]} does not match resolved prompt {prompt_length}")
    if shape[0] != 1:
        raise RuntimeError("activation capture/patch requires exactly one episode per forward pass")
    dtype, device = str(getattr(tensor, "dtype", "unknown")), str(getattr(tensor, "device", "unknown"))
    if dtype == "unknown" or device == "unknown":
        raise RuntimeError("hook tensor must expose dtype and device")
    return shape, dtype, device


def _detach_clone(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if not hasattr(value, "clone"):
        raise RuntimeError("hook tensor must support clone()")
    return value.clone()


def _new_tensor(reference: Any, values: tuple[Any, ...]) -> Any:
    creator = getattr(reference, "new_tensor", None)
    if not callable(creator):
        raise RuntimeError("hook tensor must support new_tensor() for dtype/device-safe patching")
    return creator(_lists(values))


def _tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise RuntimeError("activation/output must support tolist() for immutable hashing")


def _hashable_output(output: Any) -> Any:
    """Select a stable numeric output from Tensor or Transformers model output."""
    logits = getattr(output, "logits", None)
    if logits is not None:
        return _tuples(_tolist(logits))
    if hasattr(output, "tolist"):
        return _tuples(_tolist(output))
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return _hashable_output(item)
            except RuntimeError:
                continue
    raise RuntimeError("model output must expose numeric logits/tensor content for immutable hashing")


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("activation values must be finite numeric JSON")
    number = float(value)
    if not float("-inf") < number < float("inf"):
        raise ValueError("activation values must be finite")
    return number


def _lists(value: Any) -> Any:
    return [_lists(item) for item in value] if isinstance(value, tuple) else value


def _shape_from_values(value: Any) -> tuple[int, ...]:
    shape = []
    current = value
    while isinstance(current, tuple):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def _inference_context() -> Any:
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.inference_mode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(_lists(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
