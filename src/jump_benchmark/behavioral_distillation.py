"""Label-free behavioral distillation into the frozen same-z32 Gemma adapter."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from jump_contracts import artifact_declaration, canonical_json, write_task_evidence

from .authentic import build_gated_residual_projector, matched_injection_prompt, matched_world_pair
from .authentic_stage_d import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION, freeze_base, persistent_z_injection
from .canonical import sha256_json
from .long_horizon import FUTURE_HORIZON, LATENT_DIM, REPLICATION_MANIFEST_SHA256
from .long_horizon_stage_b import (
    RATE_USD_PER_HOUR,
    SOURCE_DECODER_SHA256,
    SOURCE_ENCODER_SHA256,
    SOURCE_RELATIVE_ROOT,
    _encode,
    _episode,
    _load_world,
    _paired_ci,
)
from .simulator import SimulatorConfig, derive_seed


SCHEMA_VERSION = "jump.track-h-behavioral-distillation/v1"
TRAIN_SEED = 150731
HELDOUT_SEED = 66173
QUERY_SPECS = (
    *(dict(id=f"partition_{i}", choices=("0", "1")) for i in range(1, 6)),
    dict(id="law_same", choices=(" attract", " repel")),
    dict(id="law_different", choices=(" attract", " repel")),
    dict(id="law_exponent", choices=("1", "2")),
    dict(id="adequacy", choices=("true", "false")),
)


def behavioral_distillation_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "track-h-same-z32-behavioral-distillation-pilot-v1",
        "phase": "B-distillation-pilot",
        "claim_label": "label-free rollout-logit distillation engineering pilot; no behavioral, causal, or mechanistic claim",
        "world_model": {
            "manifest_sha256": REPLICATION_MANIFEST_SHA256,
            "encoder_sha256": SOURCE_ENCODER_SHA256,
            "decoder_sha256": SOURCE_DECODER_SHA256,
            "latent_shape": [LATENT_DIM],
            "frozen": True,
            "decoder_input": "same serialized z only",
        },
        "base_model": {
            "repo_id": BASE_REPO_ID,
            "revision": BASE_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "frozen": True,
        },
        "teacher": {
            "frozen": True,
            "input": "server-owned canonical decoder-predicted eight-step positions and motions only",
            "forbidden": ["partition", "relation", "law", "adequacy", "answer", "target", "target_prefix"],
            "target_token_vocabulary": [item["id"] for item in QUERY_SPECS],
            "teacher_data": "train worlds only",
        },
        "student": {
            "prompt": "identical server-owned normal prompt for own/no/scrambled/wrong controls",
            "injection": "nontextual 32D z residual at input embedding on every forward",
            "trainable": ["Linear(32,3840,bias=False)", "scalar tanh gate"],
            "objective": "KL(T=2) over each frozen binary structured-answer vocabulary plus 0.25 teacher/student logit-margin MSE",
            "steps": 96,
            "batch_size": 2,
            "train_worlds": 192,
            "learning_rate": 0.0003,
            "seed": TRAIN_SEED,
        },
        "evaluation": {
            "heldout_pairs": 24,
            "pair_seed_root": HELDOUT_SEED,
            "controls": ["no_z", "scrambled_z", "wrong_world_z"],
            "paired_world_bootstraps": 10000,
            "primary": "own-z target-answer logit margin minus each control",
            "pass": {"mean_exclusive": 0.0, "paired_ci_lower_exclusive": 0.0, "required_all_controls": True},
            "secondary": ["joint structured-answer exact accuracy", "parse rate"],
            "no_heldout_tuning": True,
        },
        "execution": {
            "modal_function": "authentic_world_behavioral_distillation",
            "resource": "H100", "gpu_count": 1, "max_containers": 1, "max_inputs": 1,
            "max_attempts": 1, "timeout_seconds": 3600,
            "h100_rate_usd_per_hour": RATE_USD_PER_HOUR,
            "forecast_usd": RATE_USD_PER_HOUR,
            "aggregate_authority_ceiling_usd": 100.0,
        },
        "claims": {"informative_z": False, "behavioral": False, "causal": False, "mechanistic": False},
    }


MANIFEST_SHA256 = sha256_json(behavioral_distillation_manifest())


def _query_prompt(query_id: str) -> str:
    descriptions = {
        **{f"partition_{i}": f"Return the hidden group bit for object {i}." for i in range(1, 6)},
        "law_same": "Return the force sign for same-group objects.",
        "law_different": "Return the force sign for different-group objects.",
        "law_exponent": "Return the force exponent.",
        "adequacy": "Return whether the proposed theory is adequate.",
    }
    return matched_injection_prompt() + " " + descriptions[query_id] + " Answer:"


def canonical_predictive_rollout(prediction: Any) -> str:
    """Serialize predictions only; field names and validation fail closed on labels."""
    value = prediction.detach().to(device="cpu", dtype=__import__("torch").float32)
    if tuple(value.shape) != (FUTURE_HORIZON, 6, 2) or not bool(__import__("torch").isfinite(value).all()):
        raise ValueError("predictive rollout shape/finite check failed")
    positions = value.tolist()
    motions = [
        [[round(positions[t][o][d] - positions[t - 1][o][d], 6) for d in range(2)] for o in range(6)]
        for t in range(1, FUTURE_HORIZON)
    ]
    payload = {
        "schema_version": "jump.predictive-rollout-observation/v1",
        "positions": [[[round(v, 6) for v in obj] for obj in frame] for frame in positions],
        "motions": motions,
    }
    forbidden = {"partition", "relation", "law", "adequacy", "answer", "target", "seed", "episode_id"}
    if forbidden & set(payload):
        raise RuntimeError("teacher rollout leaked a forbidden field")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _choice_token_ids(tokenizer: Any, prompt: str, choices: tuple[str, str]) -> list[int]:
    base = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    result = []
    for choice in choices:
        full = tokenizer(prompt + choice, add_special_tokens=True)["input_ids"]
        if full[: len(base)] != base or len(full) != len(base) + 1:
            raise RuntimeError(f"choice is not one frozen next token: {choice!r}")
        result.append(int(full[-1]))
    if len(set(result)) != 2:
        raise RuntimeError("query choice token IDs collide")
    return result


def _target_choice(target: dict[str, Any], query_id: str) -> int:
    if query_id.startswith("partition_"):
        return int(target["partition"][int(query_id.rsplit("_", 1)[1])])
    if query_id == "law_same":
        return 0 if target["replacement_law"]["same"] == "attract" else 1
    if query_id == "law_different":
        return 0 if target["replacement_law"]["different"] == "attract" else 1
    if query_id == "law_exponent":
        return int(target["replacement_law"]["exponent"]) - 1
    if query_id == "adequacy":
        return 0 if target["adequacy"] is True else 1
    raise ValueError(query_id)


def _student_logits(model: Any, tokenizer: Any, projector: Any, z: Any, prompt: str, enabled: bool):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"].to(z.device)
    mask = encoded["attention_mask"].to(z.device)
    with persistent_z_injection(model, projector, z, enabled=enabled):
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :]


def _teacher_logits(model: Any, tokenizer: Any, rollout: str, prompt: str, device: str):
    # The rollout is produced from decoder(z), never from simulator targets.
    teacher_prompt = "Predicted rollout observation:" + rollout + "\n" + prompt
    encoded = tokenizer(teacher_prompt, return_tensors="pt", add_special_tokens=True)
    with __import__("torch").no_grad():
        return model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            use_cache=False,
        ).logits[:, -1, :]


def _structured_prediction(margins: dict[str, float]) -> dict[str, Any]:
    return {
        "partition": [0, *[int(margins[f"partition_{i}"] < 0) for i in range(1, 6)]],
        "replacement_law": {
            "same": "attract" if margins["law_same"] >= 0 else "repel",
            "different": "attract" if margins["law_different"] >= 0 else "repel",
            "exponent": 1 if margins["law_exponent"] >= 0 else 2,
        },
        "adequate": margins["adequacy"] >= 0,
    }


def _prepare_root(output_root: Path, expected_manifest_sha256: str, expected_code_sha: str) -> None:
    if expected_manifest_sha256 != MANIFEST_SHA256 or os.environ.get("JUMP_CODE_VERSION") != expected_code_sha:
        raise ValueError("behavioral-distillation immutable identity mismatch")
    if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
        raise FileExistsError("behavioral-distillation requires an empty canonical runner workdir")
    attempt = output_root.parent
    config = json.loads((attempt.parent.parent / "config.json").read_text())
    parameters = json.loads((attempt / "task-parameters.json").read_text())
    if (
        os.environ.get("JUMP_RUN_ID") != "behavioral-distillation-pilot"
        or os.environ.get("JUMP_PHASE_ID") != "behavioral-distillation-pilot"
        or config.get("manifest_sha256") != expected_manifest_sha256
        or config.get("code_version") != expected_code_sha
        or parameters.get("expected_manifest_sha256") != expected_manifest_sha256
        or parameters.get("expected_code_sha") != expected_code_sha
    ):
        raise RuntimeError("behavioral-distillation canonical runner binding mismatch")


def cpu_preflight(tokenizer: Any, encoder: Any, decoder: Any) -> dict[str, Any]:
    import torch

    episode = _episode(derive_seed(TRAIN_SEED, "distill-preflight"), "distill-preflight")
    z, _ = _encode(encoder, episode, "cpu")
    with torch.no_grad():
        prediction = decoder(z).reshape(FUTURE_HORIZON, 6, 2)
    rollout = canonical_predictive_rollout(prediction)
    token_ids = {}
    for spec in QUERY_SPECS:
        prompt = _query_prompt(spec["id"])
        token_ids[spec["id"]] = _choice_token_ids(tokenizer, prompt, spec["choices"])
    return {
        "latent_shape": list(z.shape), "rollout_sha256": hashlib.sha256(rollout.encode()).hexdigest(),
        "query_token_ids": token_ids, "teacher_forbidden_fields_present": False,
        "target_values_inspected": False, "base_weights_loaded": False,
    }


def train_and_evaluate(source_root: Path, output_root: Path, expected_manifest_sha256: str, expected_code_sha: str, device: str = "cuda") -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    _prepare_root(output_root, expected_manifest_sha256, expected_code_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("behavioral distillation requires CUDA")
    manifest = behavioral_distillation_manifest(); cfg = manifest["student"]
    torch.manual_seed(TRAIN_SEED); torch.cuda.manual_seed_all(TRAIN_SEED); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(BASE_REPO_ID, revision=BASE_REVISION, torch_dtype=torch.bfloat16, trust_remote_code=False).to(device)
    base_parameters = freeze_base(model); model.eval(); model.config.use_cache = False; model.gradient_checkpointing_enable()
    encoder, decoder = _load_world(source_root, device)
    hidden_size = int(model.config.text_config.hidden_size)
    projector = build_gated_residual_projector(hidden_size, latent_dim=LATENT_DIM).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=cfg["learning_rate"])
    token_map = {spec["id"]: _choice_token_ids(tokenizer, _query_prompt(spec["id"]), spec["choices"]) for spec in QUERY_SPECS}
    train_seeds = [derive_seed(TRAIN_SEED, f"distill-train:{i}") for i in range(cfg["train_worlds"])]
    train_seed_digest = sha256_json(train_seeds)
    losses = []
    projector.train()
    temperature = 2.0
    for step in range(cfg["steps"]):
        optimizer.zero_grad(set_to_none=True); step_loss = 0.0
        spec = QUERY_SPECS[step % len(QUERY_SPECS)]
        prompt = _query_prompt(spec["id"]); choice_ids = torch.tensor(token_map[spec["id"]], device=device)
        for offset in range(cfg["batch_size"]):
            seed = train_seeds[(step * cfg["batch_size"] + offset) % len(train_seeds)]
            episode = _episode(seed, "distill-train")
            z, _ = _encode(encoder, episode, device)
            with torch.no_grad(): prediction = decoder(z).reshape(FUTURE_HORIZON, 6, 2)
            teacher = _teacher_logits(model, tokenizer, canonical_predictive_rollout(prediction), prompt, device)[:, choice_ids].float()
            student = _student_logits(model, tokenizer, projector, z, prompt, True)[:, choice_ids].float()
            kl = F.kl_div(F.log_softmax(student / temperature, dim=-1), F.softmax(teacher / temperature, dim=-1), reduction="batchmean") * temperature**2
            margin = F.mse_loss(student[:, 0] - student[:, 1], teacher[:, 0] - teacher[:, 1])
            step_loss = step_loss + kl + 0.25 * margin
        step_loss = step_loss / cfg["batch_size"]
        if not torch.isfinite(step_loss): raise RuntimeError("non-finite behavioral-distillation loss")
        step_loss.backward(); optimizer.step(); losses.append(float(step_loss.detach().cpu()))
    if any(p.requires_grad for p in model.parameters()) or any(p.requires_grad for p in encoder.parameters()) or any(p.requires_grad for p in decoder.parameters()):
        raise RuntimeError("a frozen component became trainable")
    projector.eval()
    evaluation = []
    controls = ("own_z", "no_z", "scrambled_z", "wrong_world_z")
    for index in range(manifest["evaluation"]["heldout_pairs"]):
        pair_seed = derive_seed(HELDOUT_SEED, f"distill-heldout-pair:{index}")
        pair = matched_world_pair(pair_seed=pair_seed, config=SimulatorConfig(steps=12))
        wrong = _episode(derive_seed(HELDOUT_SEED, f"distill-heldout-wrong:{index}"), "distill-heldout-wrong")
        own_z, _ = _encode(encoder, pair["a"], device); wrong_z, _ = _encode(encoder, wrong, device)
        permutation = list(range(LATENT_DIM)); random.Random(derive_seed(HELDOUT_SEED, f"distill-scramble:{index}")).shuffle(permutation)
        scrambled_z = own_z[..., permutation]
        arm_z = {"own_z": own_z, "no_z": own_z, "scrambled_z": scrambled_z, "wrong_world_z": wrong_z}
        target = pair["a"]["scoring_target"]
        arm_margins = {arm: {} for arm in controls}
        arm_choice_deltas = {arm: {} for arm in controls}
        with torch.no_grad():
            for spec in QUERY_SPECS:
                prompt = _query_prompt(spec["id"]); ids = torch.tensor(token_map[spec["id"]], device=device)
                correct = _target_choice(target, spec["id"]); alternative = 1 - correct
                for arm in controls:
                    logits = _student_logits(model, tokenizer, projector, arm_z[arm], prompt, arm != "no_z")[:, ids].float()
                    arm_margins[arm][spec["id"]] = float((logits[0, correct] - logits[0, alternative]).cpu())
                    arm_choice_deltas[arm][spec["id"]] = float((logits[0, 0] - logits[0, 1]).cpu())
        predictions = {arm: _structured_prediction(values) for arm, values in arm_choice_deltas.items()}
        evaluation.append({
            "pair_index": index, "pair_id": pair["pair_id"],
            "heldout_world_seed_sha256": hashlib.sha256(str(pair_seed).encode()).hexdigest(),
            "latent_sha256": hashlib.sha256(own_z.detach().float().cpu().numpy().astype("<f4").tobytes()).hexdigest(),
            "margins": {arm: sum(values.values()) / len(values) for arm, values in arm_margins.items()},
            "joint_exact": {arm: float(predictions[arm] == {"partition": target["partition"], "replacement_law": target["replacement_law"], "adequate": target["adequacy"]}) for arm in controls},
            "parse": {arm: 1.0 for arm in controls},
        })
    comparisons = {}
    for offset, control in enumerate(controls[1:]):
        values = [row["margins"]["own_z"] - row["margins"][control] for row in evaluation]
        ci = _paired_ci(values, 77117 + offset)
        comparisons[control] = {"mean": sum(values)/len(values), "paired_world_cluster_ci95": ci, "passed": sum(values)/len(values) > 0 and ci[0] > 0}
    gate_pass = all(v["passed"] for v in comparisons.values())
    for role in ("future_projector", "gemma_adapter"): (output_root / role).mkdir()
    save_file({"projector.weight": projector.projector.weight.detach().cpu()}, output_root / "future_projector/model.safetensors")
    save_file({"gate": projector.gate.detach().cpu()}, output_root / "gemma_adapter/adapter_model.safetensors")
    (output_root / "future_projector/config.json").write_bytes(canonical_json({"input_dim": LATENT_DIM, "output_dim": hidden_size, "objective": manifest["student"]["objective"]}))
    (output_root / "gemma_adapter/adapter_config.json").write_bytes(canonical_json({"gate": "scalar_tanh", "injection_site": "input_embedding_every_forward"}))
    (output_root / "manifest.json").write_bytes(canonical_json(manifest))
    (output_root / "evaluation.json").write_bytes(canonical_json(evaluation))
    runtime = time.monotonic() - started
    terminal = {
        "status": "completed", "decision": "pass" if gate_pass else "pivot", "phase_c_allowed": gate_pass,
        "manifest_sha256": MANIFEST_SHA256, "code_sha": expected_code_sha,
        "base_revision": BASE_REVISION, "transformers_revision": TRANSFORMERS_REVISION,
        "source_encoder_sha256": SOURCE_ENCODER_SHA256, "source_decoder_sha256": SOURCE_DECODER_SHA256,
        "train_world_seed_set_sha256": train_seed_digest, "heldout_pair_seed_root": HELDOUT_SEED,
        "base_parameters": base_parameters, "trainable_parameters": sum(p.numel() for p in projector.parameters()),
        "initial_loss": losses[0], "final_loss": losses[-1], "comparisons": comparisons,
        "secondary": {arm: {"joint_exact": sum(row["joint_exact"][arm] for row in evaluation)/len(evaluation), "parse": 1.0} for arm in controls},
        "runtime_seconds": runtime, "estimated_cost_usd": runtime/3600*RATE_USD_PER_HOUR,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "claims": {"informative_z": gate_pass, "behavioral": gate_pass, "causal": False, "mechanistic": False},
        "claim_label": manifest["claim_label"],
    }
    (output_root / "terminal.json").write_bytes(canonical_json(terminal))
    artifacts = [artifact_declaration(path, output_root, role="behavioral-distillation-evidence") for path in sorted(output_root.rglob("*")) if path.is_file()]
    task = write_task_evidence(output_root, metrics=[{"name":"initial_loss","value":losses[0]},{"name":"final_loss","value":losses[-1]}, *[{"name":"own_margin_minus_control","condition":name,"value":row["mean"]} for name,row in comparisons.items()]], artifacts=artifacts, track_h={"phase":"B-distillation-pilot","decision":terminal["decision"],"claims":terminal["claims"]})
    return {**terminal, "task_evidence": task}


def run_contract(expected_manifest_sha256: str, expected_code_sha: str, dry_run: bool = False):
    phase = {"id":"behavioral-distillation-pilot", "_secret_keys":["HF_TOKEN"], "_preregistration":{"layer_allowlist":[0],"timepoint_allowlist":["answer"]}}
    run = {"id":"behavioral-distillation-pilot", "task":{"module":"jump_benchmark.behavioral_distillation_task","parameters":{"expected_manifest_sha256":expected_manifest_sha256,"expected_code_sha":expected_code_sha, **({"dry_run":True} if dry_run else {})}}, "resources":{"gpu":"cpu" if dry_run else "H100","timeout_seconds":60 if dry_run else 3600}, "selection":{"layers":[],"timepoints":[]}, "retry":{"max_attempts":1}}
    return phase, run
