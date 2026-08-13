import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jump_contracts import (
    SUPPORTED_TRANSFORMERS_REVISION,
    EvidenceError,
    build_world_model_component_manifest,
    build_world_model_load_record,
    component_identity,
)

from jump_mechanistic.boundaries import BoundaryManifest, require_matched_prompt_lengths, resolve_boundaries
from jump_mechanistic.runtime import (
    HookManifest,
    MechanisticRuntime,
    WorldModelRuntimeBinding,
    normalized_logit_patch_score,
)


SENTINELS = {f"T{i}": {"text": f"<{i}>", "token_ids": [10 + i]} for i in range(5)}
FOLLOWERS = {
    "T3": {
        "kind": "adequacy_boolean",
        "options": [
            {"text": "true", "token_ids": [56, 54, 57, 51]},
            {"text": "false", "token_ids": [52, 57, 58, 55, 51]},
        ],
    },
    "T4": {
        "kind": "replacement_law_tuple",
        "options": [{"text": "(", "token_ids": [50]}],
    },
}


class FakeTokenizer:
    def __init__(self, *, missing=None, retokenize=None):
        self.missing = missing
        self.retokenize = retokenize
        self.table = {f"<{i}>": 10 + i for i in range(5)}

    def __call__(self, text, **kwargs):
        if not kwargs.get("add_special_tokens", True):
            if text == self.retokenize:
                ids = [99]
            elif text in self.table:
                ids = [self.table[text]]
            else:
                ids = [50 + ord(char) % 10 for char in text]
            return {"input_ids": ids}
        ids = [1]
        cursor = 0
        while cursor < len(text):
            found = next((s for s in self.table if text.startswith(s, cursor)), None)
            if found:
                if found != self.missing:
                    ids.append(self.table[found])
                cursor += len(found)
            else:
                ids.append(50 + ord(text[cursor]) % 10)
                cursor += 1
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def boundary_manifest():
    return BoundaryManifest.from_dict(
        {
            "tokenizer_id": "fake", "tokenizer_revision": "tok-rev",
            "sentinels": SENTINELS, "immediate_followers": FOLLOWERS,
        }
    )


def prompt(prefix=""):
    return prefix + "<0>a<1>b<2>c<3>true<4>(law)"


class BoundaryTests(unittest.TestCase):
    def test_resolves_all_boundaries_and_hash_is_immutable(self):
        resolved = resolve_boundaries(FakeTokenizer(), prompt(), boundary_manifest())
        self.assertEqual(list(resolved.positions.values()), sorted(resolved.positions.values()))
        self.assertEqual(resolved.tokenizer_revision, "tok-rev")
        self.assertEqual(len(resolved.input_ids), len(resolved.attention_mask))
        resolved.verify()
        altered = copy.copy(resolved)
        object.__setattr__(altered, "prompt", "tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            altered.verify()

    def test_missing_duplicate_and_retokenized_sentinels_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "T2.*exactly once"):
            resolve_boundaries(FakeTokenizer(), prompt().replace("<2>", ""), boundary_manifest())
        with self.assertRaisesRegex(ValueError, "T2.*exactly once"):
            resolve_boundaries(FakeTokenizer(), prompt() + "<2>", boundary_manifest())
        with self.assertRaisesRegex(ValueError, "T2 token IDs drifted"):
            resolve_boundaries(FakeTokenizer(retokenize="<2>"), prompt(), boundary_manifest())
        with self.assertRaisesRegex(ValueError, "full tokenization"):
            resolve_boundaries(FakeTokenizer(missing="<2>"), prompt(), boundary_manifest())

    def test_matched_prompts_require_length_and_boundary_alignment(self):
        clean = resolve_boundaries(FakeTokenizer(), prompt("x"), boundary_manifest())
        corrupt = resolve_boundaries(FakeTokenizer(), prompt("y"), boundary_manifest())
        self.assertEqual(require_matched_prompt_lengths(clean, corrupt), len(clean.input_ids))
        shifted = resolve_boundaries(FakeTokenizer(), prompt("yy"), boundary_manifest())
        with self.assertRaisesRegex(ValueError, "matched tokenized lengths"):
            require_matched_prompt_lengths(clean, shifted)

    def test_t3_t4_must_immediately_precede_boolean_and_law_tuple(self):
        resolved = resolve_boundaries(FakeTokenizer(), prompt(), boundary_manifest())
        boundaries = {item.timepoint.value: item for item in resolved.boundaries}
        self.assertEqual(boundaries["T3"].immediate_follower_kind, "adequacy_boolean")
        self.assertEqual(boundaries["T3"].immediate_follower_text, "true")
        self.assertEqual(boundaries["T4"].immediate_follower_kind, "replacement_law_tuple")
        with self.assertRaisesRegex(ValueError, "T3.*immediately before adequacy_boolean"):
            resolve_boundaries(FakeTokenizer(), prompt().replace("<3>true", "<3> true"), boundary_manifest())
        with self.assertRaisesRegex(ValueError, "T4.*immediately before replacement_law_tuple"):
            resolve_boundaries(FakeTokenizer(), prompt().replace("<4>(", "<4> ("), boundary_manifest())
        with self.assertRaisesRegex(ValueError, "immediate follower T3 token IDs drifted"):
            resolve_boundaries(FakeTokenizer(retokenize="true"), prompt(), boundary_manifest())
        invalid = {
            "tokenizer_id": "fake", "tokenizer_revision": "tok-rev", "sentinels": SENTINELS,
            "immediate_followers": {"T3": FOLLOWERS["T3"]},
        }
        with self.assertRaisesRegex(ValueError, "exactly T3 and T4"):
            BoundaryManifest.from_dict(invalid)


class FakeTensor:
    dtype = "float32"
    device = "cpu"

    def __init__(self, values):
        self.values = copy.deepcopy(values)

    @property
    def shape(self):
        def dims(value):
            return () if not isinstance(value, list) else (len(value),) + dims(value[0])
        return dims(self.values)

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.values)

    def tolist(self):
        return copy.deepcopy(self.values)

    def new_tensor(self, values):
        return FakeTensor(values)

    def __getitem__(self, key):
        batch, position, hidden = key
        if isinstance(position, int):
            rows = [row[position] for row in self.values]
        else:
            rows = [row[position] for row in self.values]
        if hidden != slice(None):
            raise AssertionError
        return FakeTensor(rows)

    def __setitem__(self, key, value):
        _batch, position, _hidden = key
        for index, row in enumerate(self.values):
            row[position] = copy.deepcopy(value.values[index])

    def _binary(self, other, op):
        def apply(a, b):
            if isinstance(a, list):
                return [apply(x, y) for x, y in zip(a, b)] if isinstance(b, list) else [apply(x, b) for x in a]
            return op(a, b)
        rhs = other.values if isinstance(other, FakeTensor) else other
        return FakeTensor(apply(self.values, rhs))

    def __add__(self, other): return self._binary(other, lambda a, b: a + b)
    def __sub__(self, other): return self._binary(other, lambda a, b: a - b)
    def __mul__(self, other): return self._binary(other, lambda a, b: a * b)


class FakeHandle:
    def __init__(self, module, hook): self.module, self.hook = module, hook
    def remove(self): self.module.hooks.remove(self.hook)


class FakeDecoderLayer:
    def __init__(self): self.hooks = []
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self, hook)
    def __call__(self, tensor):
        output = tensor
        for hook in list(self.hooks):
            replacement = hook(self, (tensor,), output)
            if replacement is not None: output = replacement
        return output


class FakeModel:
    def __init__(self, *, width=2, tuple_output=False):
        self.layer = FakeDecoderLayer()
        self.width = width
        self.tuple_output = tuple_output
    def named_modules(self): return [("", self), ("model.layers.0", self.layer)]
    def __call__(self, input_ids, **_kwargs):
        prefix = input_ids[0][1]
        values = [[[float(token), float(token + prefix)][: self.width] for token in input_ids[0]]]
        output = FakeTensor(values)
        if self.tuple_output:
            hooked = (output, "cache")
            for hook in list(self.layer.hooks):
                replacement = hook(self.layer, (output,), hooked)
                if replacement is not None: hooked = replacement
            return hooked
        return self.layer(output)


class FakePeftModel:
    def __init__(self): self.base = FakeModel()
    def get_base_model(self): return self.base
    def __call__(self, **inputs): return self.base(**inputs)


class DoubleFireModel(FakeModel):
    def __call__(self, **inputs):
        first = super().__call__(**inputs)
        return self.layer(first)


def hook_manifest(
    module_type="FakeDecoderLayer",
    width=2,
    selector="tensor",
    peft=False,
    study_scope="synthetic",
    model_id="fake-gemma",
    model_revision="model-rev",
):
    return HookManifest.from_dict(
        {
            "model_id": model_id, "model_revision": model_revision,
            "tokenizer_revision": "tok-rev", "peft_base_model": peft,
            "study_scope": study_scope,
            "sites": [{"site_id": "L0.resid", "module_name": "model.layers.0",
                       "module_type": module_type, "timepoint": "T2", "expected_rank": 3,
                       "expected_hidden_size": width, "output_selector": selector}],
        }
    )


def world_model_contract(root, *, live_ready=True):
    components = {}
    for role in ("encoder", "decoder", "future_projector", "gemma_adapter"):
        directory = root / "components" / role
        directory.mkdir(parents=True)
        weights_name = "adapter_model.safetensors" if role == "gemma_adapter" else "model.safetensors"
        config_name = "adapter_config.json" if role == "gemma_adapter" else "config.json"
        weights = f"{role}-weights".encode()
        config = json.dumps({"architecture": role}, sort_keys=True).encode()
        (directory / weights_name).write_bytes(weights)
        (directory / config_name).write_bytes(config)
        relative = directory.relative_to(root).as_posix()
        components[role] = component_identity(
            directory=relative,
            weights_path=f"{relative}/{weights_name}",
            weights_sha256=hashlib.sha256(weights).hexdigest(),
            config_path=f"{relative}/{config_name}",
            config_sha256=hashlib.sha256(config).hexdigest(),
            architecture=f"fake-{role}",
        )
    manifest = build_world_model_component_manifest(
        base_model_repo_id="google/gemma-test",
        base_model_revision="b" * 40,
        transformers_revision=SUPPORTED_TRANSFORMERS_REVISION,
        latent_dtype="float32-le",
        latent_shape=[1, 2],
        latent_order="C",
        projector_input_dim=2,
        projector_output_dim=2,
        projector_gate="learned_scalar_sigmoid",
        injection_layer=0,
        injection_site="residual_stream_post_attention",
        encoder=components["encoder"],
        decoder=components["decoder"],
        future_projector=components["future_projector"],
        gemma_adapter=components["gemma_adapter"],
        artifact_only_ready=True,
        end_to_end_injection=True,
        live_ready=live_ready,
        claim_guards={
            "engineering_only": True,
            "behavioral_claim_allowed": False,
            "mechanistic_claim_allowed": False,
            "causal_claim_allowed": False,
            "benchmark_law_accuracy_claim_allowed": False,
            "track_r_claim_allowed": False,
        },
    )
    revision = "c" * 40
    load_record = build_world_model_load_record(
        manifest,
        root,
        expected_repository_revision=revision,
        resolved_repository_revision=revision,
        mode="gated_gemma",
    )
    return manifest, load_record


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.clean = resolve_boundaries(FakeTokenizer(), prompt("x"), boundary_manifest())
        self.corrupt = resolve_boundaries(FakeTokenizer(), prompt("y"), boundary_manifest())
        self.clean_inputs = {"input_ids": [list(self.clean.input_ids)],
                             "attention_mask": [list(self.clean.attention_mask)]}
        self.corrupt_inputs = {"input_ids": [list(self.corrupt.input_ids)],
                               "attention_mask": [list(self.corrupt.attention_mask)]}

    def test_capture_denoise_noise_zero_scale_and_immutable_provenance(self):
        runtime = MechanisticRuntime(FakeModel(), hook_manifest())
        clean_out, clean = runtime.capture(inputs=self.clean_inputs, resolved_prompt=self.clean,
                                           episode_id="e", site_id="L0.resid", timepoint="T2", run_kind="clean")
        corrupt_out, corrupt = runtime.capture(inputs=self.corrupt_inputs, resolved_prompt=self.corrupt,
                                               episode_id="e", site_id="L0.resid", timepoint="T2", run_kind="corrupt")
        clean.verify(); corrupt.verify()
        zero, run = runtime.patch(inputs=self.corrupt_inputs, target_prompt=self.corrupt,
                                  source_prompt=self.clean, source=clean, site_id="L0.resid",
                                  timepoint="T2", mode="denoising", scale=0.0)
        self.assertEqual(zero.tolist(), corrupt_out.tolist())
        run.verify()
        tampered_run = copy.copy(run)
        object.__setattr__(tampered_run, "scale", 1.0)
        with self.assertRaisesRegex(ValueError, "patch provenance hash mismatch"):
            tampered_run.verify()
        restored, _ = runtime.patch(inputs=self.corrupt_inputs, target_prompt=self.corrupt,
                                    source_prompt=self.clean, source=clean, site_id="L0.resid",
                                    timepoint="T2", mode="denoising")
        pos = self.corrupt.positions[next(p for p in self.corrupt.positions if p.value == "T2")]
        self.assertEqual(restored.tolist()[0][pos], clean_out.tolist()[0][pos])
        noised, _ = runtime.patch(inputs=self.clean_inputs, target_prompt=self.clean,
                                  source_prompt=self.corrupt, source=corrupt, site_id="L0.resid",
                                  timepoint="T2", mode="noising")
        self.assertEqual(noised.tolist()[0][pos], corrupt_out.tolist()[0][pos])
        self.assertEqual(run.dtype, "float32")
        tampered = copy.copy(clean.provenance)
        object.__setattr__(tampered, "hook_name", "drift")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            tampered.verify()

    def test_hook_name_type_and_shape_drift_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "missing module"):
            bad = hook_manifest()
            object.__setattr__(bad.sites[0], "module_name", "model.layers.9")
            MechanisticRuntime(FakeModel(), bad)
        with self.assertRaisesRegex(RuntimeError, "expected Wrong"):
            MechanisticRuntime(FakeModel(), hook_manifest(module_type="Wrong"))
        runtime = MechanisticRuntime(FakeModel(), hook_manifest(width=3))
        with self.assertRaisesRegex(RuntimeError, "hook shape drift"):
            runtime.capture(inputs=self.clean_inputs, resolved_prompt=self.clean, episode_id="e",
                            site_id="L0.resid", timepoint="T2", run_kind="clean")
        primary = {
            "model_id": "gemma", "model_revision": "rev", "tokenizer_revision": "tok",
            "peft_base_model": False, "study_scope": "confirmatory_primary",
            "sites": [{"site_id": "wrong", "module_name": "model.layers.8",
                       "module_type": "FakeDecoderLayer", "timepoint": "T2",
                       "expected_rank": 3, "expected_hidden_size": 2,
                       "output_selector": "tensor"}],
        }
        with self.assertRaisesRegex(ValueError, "exactly layers"):
            HookManifest.from_dict(primary)

    def test_peft_base_model_is_explicitly_unwrapped_and_inputs_are_bound(self):
        runtime = MechanisticRuntime(FakePeftModel(), hook_manifest(peft=True))
        _output, snapshot = runtime.capture(
            inputs=self.clean_inputs, resolved_prompt=self.clean, episode_id="e",
            site_id="L0.resid", timepoint="T2", run_kind="clean",
        )
        snapshot.verify()
        corrupt_inputs = copy.deepcopy(self.clean_inputs)
        corrupt_inputs["input_ids"][0][0] = 999
        with self.assertRaisesRegex(ValueError, "do not match"):
            runtime.capture(
                inputs=corrupt_inputs, resolved_prompt=self.clean, episode_id="e2",
                site_id="L0.resid", timepoint="T2", run_kind="clean",
            )

    def test_confirmatory_runtime_is_disabled_until_canonical_s1_loader(self):
        confirmatory = hook_manifest(
            peft=True,
            study_scope="confirmatory_secondary",
            model_id="google/gemma-test",
            model_revision="b" * 40,
        )
        with self.assertRaisesRegex(EvidenceError, "disabled until the canonical"):
            MechanisticRuntime(FakePeftModel(), confirmatory)

        with tempfile.TemporaryDirectory() as directory:
            component_manifest, load_record = world_model_contract(Path(directory))
            binding = WorldModelRuntimeBinding.from_contracts(
                component_manifest,
                load_record,
                verified_loader_attestation_sha256="d" * 64,
            )
            self.assertEqual(
                binding.component_manifest_sha256,
                component_manifest["manifest_sha256"],
            )
            self.assertEqual(
                binding.load_record_sha256,
                load_record["load_record_sha256"],
            )
            self.assertEqual(
                binding.gemma_adapter_identity_sha256,
                component_manifest["components"]["gemma_adapter"]["identity_sha256"],
            )
            self.assertFalse(
                component_manifest["load_contract"]["allow_remote_code"]
            )
            with self.assertRaisesRegex(EvidenceError, "disabled until the canonical"):
                MechanisticRuntime(
                    FakePeftModel(),
                    confirmatory,
                    world_model_manifest=component_manifest,
                    world_model_load_record=load_record,
                    verified_loader_attestation_sha256="d" * 64,
                )

            tampered = copy.deepcopy(load_record)
            tampered["repository"]["resolved_revision"] = "d" * 40
            with self.assertRaises(EvidenceError):
                WorldModelRuntimeBinding.from_contracts(
                    component_manifest,
                    tampered,
                    verified_loader_attestation_sha256="d" * 64,
                )

    def test_confirmatory_runtime_rejects_non_live_component_contract(self):
        confirmatory = hook_manifest(
            peft=True,
            study_scope="confirmatory_secondary",
            model_id="google/gemma-test",
            model_revision="b" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            component_manifest, load_record = world_model_contract(
                Path(directory), live_ready=False
            )
            with self.assertRaisesRegex(EvidenceError, "live-ready"):
                WorldModelRuntimeBinding.from_contracts(
                    component_manifest,
                    load_record,
                    verified_loader_attestation_sha256="d" * 64,
                )

    def test_patch_rejects_cross_adapter_or_loader_attestation_mix_and_match(self):
        with tempfile.TemporaryDirectory() as directory:
            component_manifest, load_record = world_model_contract(Path(directory))
            synthetic_bound = hook_manifest(
                peft=True,
                study_scope="synthetic",
            )
            source_runtime = MechanisticRuntime(
                FakePeftModel(),
                synthetic_bound,
                world_model_manifest=component_manifest,
                world_model_load_record=load_record,
                verified_loader_attestation_sha256="d" * 64,
            )
            _output, source = source_runtime.capture(
                inputs=self.clean_inputs,
                resolved_prompt=self.clean,
                episode_id="cross-loader",
                site_id="L0.resid",
                timepoint="T2",
                run_kind="clean",
            )
            target_runtime = MechanisticRuntime(
                FakePeftModel(),
                synthetic_bound,
                world_model_manifest=component_manifest,
                world_model_load_record=load_record,
                verified_loader_attestation_sha256="e" * 64,
            )
            with self.assertRaisesRegex(ValueError, "distribution/load/adapter identity"):
                target_runtime.patch(
                    inputs=self.corrupt_inputs,
                    target_prompt=self.corrupt,
                    source_prompt=self.clean,
                    source=source,
                    site_id="L0.resid",
                    timepoint="T2",
                    mode="denoising",
                )

            same_runtime_output, patch = source_runtime.patch(
                inputs=self.corrupt_inputs,
                target_prompt=self.corrupt,
                source_prompt=self.clean,
                source=source,
                site_id="L0.resid",
                timepoint="T2",
                mode="denoising",
            )
            self.assertIsNotNone(same_runtime_output)
            self.assertEqual(
                patch.target_gemma_adapter_identity_sha256,
                component_manifest["components"]["gemma_adapter"]["identity_sha256"],
            )
            self.assertEqual(patch.target_verified_loader_attestation_sha256, "d" * 64)
            patch.verify()

    def test_multi_forward_generation_is_explicitly_unsupported(self):
        runtime = MechanisticRuntime(DoubleFireModel(), hook_manifest())
        with self.assertRaisesRegex(
            RuntimeError, "teacher-forced full-sequence forward, not autoregressive generation"
        ):
            runtime.capture(
                inputs=self.clean_inputs, resolved_prompt=self.clean, episode_id="e",
                site_id="L0.resid", timepoint="T2", run_kind="clean",
            )

    def test_normalized_logit_patch_score_records_floor_and_ceiling(self):
        result = normalized_logit_patch_score(
            clean_logit_difference=4.0, corrupt_logit_difference=-2.0, patched_logit_difference=1.0
        )
        self.assertEqual(result["normalized_patch_score"], 0.5)
        self.assertEqual(result["clean_logit_difference"], 4.0)
        with self.assertRaisesRegex(ValueError, "indistinguishable"):
            normalized_logit_patch_score(
                clean_logit_difference=1.0, corrupt_logit_difference=1.0, patched_logit_difference=1.0
            )


if __name__ == "__main__":
    unittest.main()
