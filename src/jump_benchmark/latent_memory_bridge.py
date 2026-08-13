"""Fresh token-conditioned latent-memory bridge pilot for frozen Gemma."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jump_contracts import artifact_declaration, canonical_json, tensor_bytes_sha256, write_task_evidence

from .authentic import build_gated_residual_projector, independent_law, independent_partition, matched_injection_prompt
from .authentic_stage_d import BASE_REPO_ID, BASE_REVISION, TRANSFORMERS_REVISION, freeze_base
from .behavioral_distillation import QUERY_SPECS, _choice_token_ids, _paired_ci, _query_prompt, _structured_prediction, _target_choice
from .canonical import sha256_json
from .object_jepa import HORIZONS, LATENT_DIM, _exact_z, _png, _tensorize, _world, build_modules
from .object_jepa_residual import MANIFEST_SHA256 as WORLD_MANIFEST_SHA256, build_predictor
from .simulator import derive_seed


SCHEMA_VERSION = "jump.track-h-latent-memory-bridge/v1"
SOURCE_ROOT = f"authentic-world-object-jepa-residual/{WORLD_MANIFEST_SHA256}/run/attempts/0001/artifacts"
SOURCE_HASHES = {
    "encoder/model.safetensors": "d963a530ebecbbbff4821f9111624397df946c6d98113c572c1962ab52e1ffcf",
    "latent_predictor/model.safetensors": "a1f7344df438faeca5d0d3236cbf82d699a5804d3de43c5b8fcc6fc5c16a58d8",
    "rollout_decoder/model.safetensors": "4b111bde84430d4770757c8e9d5600d5d83c91777f462f40469d4ac8a421dba3",
    "pixel_decoder/model.safetensors": "818ed9ab110a0e831de47f9ab09d696313c9fdfcb58aa37e04ccc97022b568db",
}
LAYERS = (7, 23, 39)
RANK = 16
MEMORY_TOKENS = 7
TRAIN_SEED = 270814
HELDOUT_SEED = 280814
RATE_USD_PER_HOUR = 3.9492


def manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "track-h-token-conditioned-latent-memory-bridge-pilot-v1",
        "hypothesis": "multiple object-specific latent memories injected at selected layers improve frozen Gemma use relative to one global additive vector",
        "claim_label": "fresh latent-memory bridge engineering pilot; no causal or mechanistic claim",
        "world_model": {
            "source_manifest_sha256": WORLD_MANIFEST_SHA256,
            "frozen": True,
            "latent_dtype": "float32-le", "latent_shape": [LATENT_DIM], "latent_order": "C",
            "same_z_consumers": ["predictor_output", "bridge_input", "pixel_decoder", "cache", "evidence"],
        },
        "base_model": {"repo_id": BASE_REPO_ID, "revision": BASE_REVISION, "transformers_revision": TRANSFORMERS_REVISION, "frozen": True},
        "bridge": {
            "memory_tokens": MEMORY_TOKENS,
            "layout": "six object tokens from xy2+object_block3 plus one global token",
            "rank": RANK,
            "layers": list(LAYERS),
            "operation": "token-conditioned low-rank cross-attention residual at every forward",
            "trainable": ["object/global memory projections", "per-layer Q/K/V/out projections", "per-layer scalar sigmoid gates"],
            "prompt_tokens_identical": True, "z_text_serialization": False,
        },
        "teacher": {
            "input": "teacher-only canonical text of the frozen world model's own predicted positions at horizons 1/2/4/8",
            "forbidden": ["ground_truth_partition", "ground_truth_relation", "ground_truth_law", "adequacy", "answer", "target_answer_prefix", "future_context"],
            "objective": "binary structured-vocabulary KL(T=2) plus 0.25 teacher/student logit-margin MSE",
        },
        "data": {
            "train_worlds": 144, "steps": 72, "batch_size": 1,
            "train_seed_root": TRAIN_SEED, "fresh_heldout_worlds": 24, "heldout_seed_root": HELDOUT_SEED,
            "world_seed_disjoint": True, "prior_heldout_sets_reused": False, "heldout_tuning": False,
        },
        "evaluation": {
            "controls": ["no_z", "scrambled_z", "wrong_world_z"], "paired_bootstraps": 10000,
            "primary": {
                "own_minus_each_control_exact_answer": "paired CI lower > 0",
                "own_minus_each_control_target_logit_margin": "paired CI lower > 0",
                "parse_delta_abs_max": 0.02,
                "required_all": True,
            },
            "diagnostics": ["layer_gate_values", "bridge_norm", "synthetic_z_target_logit_delta"],
        },
        "execution": {"modal_function":"authentic_world_latent_memory_bridge","resource":"H100","gpu_count":1,"max_containers":1,"max_inputs":1,"max_attempts":1,"timeout_seconds":3600,"h100_rate_usd_per_hour":RATE_USD_PER_HOUR,"forecast_usd":RATE_USD_PER_HOUR,"aggregate_authority_ceiling_usd":100.0},
        "claims": {"informative_z":False,"behavioral":False,"causal":False,"mechanistic":False},
    }


MANIFEST_SHA256 = sha256_json(manifest())


def build_bridge(hidden_size: int):
    import torch

    class LatentMemoryBridge(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.object_memory = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.GELU(), torch.nn.Linear(32, RANK))
            self.global_memory = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.GELU(), torch.nn.Linear(32, RANK))
            self.q = torch.nn.ModuleDict({str(layer): torch.nn.Linear(hidden_size, RANK, bias=False) for layer in LAYERS})
            self.k = torch.nn.ModuleDict({str(layer): torch.nn.Linear(RANK, RANK, bias=False) for layer in LAYERS})
            self.v = torch.nn.ModuleDict({str(layer): torch.nn.Linear(RANK, RANK, bias=False) for layer in LAYERS})
            self.out = torch.nn.ModuleDict({str(layer): torch.nn.Linear(RANK, hidden_size, bias=False) for layer in LAYERS})
            self.gates = torch.nn.ParameterDict({str(layer): torch.nn.Parameter(torch.tensor(-2.0)) for layer in LAYERS})

        def memory(self, z):
            objects = torch.cat([z[:, :12].reshape(-1, 6, 2), z[:, 12:30].reshape(-1, 6, 3)], dim=-1)
            return torch.cat([self.object_memory(objects), self.global_memory(z[:, 30:32])[:, None]], dim=1)

        def inject(self, layer: int, hidden, z):
            memory = self.memory(z).to(dtype=hidden.dtype)
            q = self.q[str(layer)](hidden)
            k = self.k[str(layer)](memory)
            v = self.v[str(layer)](memory)
            attention = torch.softmax(torch.einsum("btr,bmr->btm", q, k) / (RANK**0.5), dim=-1)
            residual = self.out[str(layer)](torch.einsum("btm,bmr->btr", attention, v))
            return hidden + torch.sigmoid(self.gates[str(layer)]) * residual

    return LatentMemoryBridge()


@contextmanager
def latent_memory_injection(model: Any, bridge: Any, z: Any, *, enabled: bool):
    import torch
    exact, raw, digest = _exact_z(z)
    binding = {"world_latent_sha256": digest, "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(), "dtype":"float32-le", "shape":list(z.shape), "order":"C", "enabled":enabled, "forward_calls":{str(layer):0 for layer in LAYERS}}
    if not enabled:
        yield binding
        return
    modules = dict(model.named_modules())
    handles = []
    for layer in LAYERS:
        name = f"model.language_model.layers.{layer}"
        if name not in modules or type(modules[name]).__name__ != "Gemma4UnifiedTextDecoderLayer":
            raise RuntimeError(f"frozen bridge layer identity mismatch: {name}")
        def hook(_module, args, kwargs, layer=layer):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None: raise RuntimeError("bridge hook received no hidden states")
            local_z = exact.expand(hidden.shape[0], -1).to(device=hidden.device)
            changed = bridge.inject(layer, hidden, local_z)
            binding["forward_calls"][str(layer)] += 1
            return ((changed, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": changed})
        handles.append(modules[name].register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield binding
    finally:
        for handle in handles: handle.remove()


def _load_world(source: Path, device: str):
    from safetensors.torch import load_file
    for relative, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest() != expected: raise RuntimeError(f"source checksum mismatch: {relative}")
    encoder, _, rollout, pixel = build_modules(); predictor = build_predictor()
    encoder.load_state_dict(load_file(source/"encoder/model.safetensors"), strict=True)
    predictor.load_state_dict(load_file(source/"latent_predictor/model.safetensors"), strict=True)
    rollout.load_state_dict(load_file(source/"rollout_decoder/model.safetensors"), strict=True)
    pixel.load_state_dict(load_file(source/"pixel_decoder/model.safetensors"), strict=True)
    for module in (encoder,predictor,rollout,pixel):
        module.to(device).eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    return encoder,predictor,rollout,pixel


def _world_latents(record: dict[str,Any], modules: tuple[Any,...], device: str):
    import torch
    encoder,predictor,rollout,pixel=modules
    data=_tensorize([record],device)
    with torch.no_grad():
        z0=encoder(data["observed"],data["mask"]); z0,_,_=_exact_z(z0)
        predicted=[];positions=[]
        for index in range(4):
            zh=predictor(z0,data["action"],torch.tensor([index],device=device));zh,_,_=_exact_z(zh);predicted.append(zh);mean,_=rollout(zh);positions.append(mean[:,index].reshape(1,6,2))
        z=predicted[-1];z,raw,digest=_exact_z(z);raster=pixel(z)[:,4]
    rollout_text=json.dumps({"schema_version":"jump.teacher-predicted-rollout/v1","horizons":list(HORIZONS),"positions":[[[round(float(v),6) for v in obj] for obj in frame[0].cpu().tolist()] for frame in positions]},sort_keys=True,separators=(",",":"),allow_nan=False)
    return z,raw,digest,rollout_text,raster


def _student_logits(model,tokenizer,bridge,z,prompt,enabled):
    encoded=tokenizer(prompt,return_tensors="pt",add_special_tokens=True);ids=encoded["input_ids"].to(z.device);mask=encoded["attention_mask"].to(z.device)
    with latent_memory_injection(model,bridge,z,enabled=enabled): return model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,-1,:]


def _teacher_logits(model,tokenizer,rollout,prompt,device):
    import torch
    encoded=tokenizer("Predicted world-model rollout:"+rollout+"\n"+prompt,return_tensors="pt",add_special_tokens=True)
    with torch.no_grad(): return model(input_ids=encoded["input_ids"].to(device),attention_mask=encoded["attention_mask"].to(device),use_cache=False).logits[:,-1,:]


def _target(seed:int):
    law=independent_law(seed)
    return {"partition":list(independent_partition(seed)),"replacement_law":law.as_dict(),"adequacy":True}


def cpu_preflight(hidden_size:int=64):
    import torch
    bridge=build_bridge(hidden_size);z=torch.randn(2,LATENT_DIM);memory=bridge.memory(z);hidden=torch.randn(2,5,hidden_size);changed=bridge.inject(LAYERS[0],hidden,z)
    return {"latent_shape":list(z.shape),"memory_shape":list(memory.shape),"output_shape":list(changed.shape),"trainable_parameters":sum(p.numel() for p in bridge.parameters()),"prompt_tokens_unchanged_by_api":True,"z_text_serialization":False}


def train_and_evaluate(source_root:Path,output_root:Path,expected_manifest_sha256:str,expected_code_sha:str,device:str="cuda"):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM,AutoTokenizer
    if expected_manifest_sha256!=MANIFEST_SHA256 or os.environ.get("JUMP_CODE_VERSION")!=expected_code_sha:raise ValueError("latent-memory bridge identity mismatch")
    if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):raise FileExistsError("latent-memory bridge requires empty canonical workdir")
    if not torch.cuda.is_available():raise RuntimeError("latent-memory bridge requires CUDA")
    torch.manual_seed(TRAIN_SEED);torch.cuda.manual_seed_all(TRAIN_SEED);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(BASE_REPO_ID,revision=BASE_REVISION,trust_remote_code=False)
    model=AutoModelForMultimodalLM.from_pretrained(BASE_REPO_ID,revision=BASE_REVISION,torch_dtype=torch.bfloat16,trust_remote_code=False).to(device);base_parameters=freeze_base(model);model.eval();model.config.use_cache=False;model.gradient_checkpointing_enable()
    world=_load_world(source_root,device);hidden=int(model.config.text_config.hidden_size);bridge=build_bridge(hidden).to(device=device,dtype=torch.bfloat16);optimizer=torch.optim.AdamW(bridge.parameters(),lr=3e-4)
    token_map={spec["id"]:_choice_token_ids(tokenizer,_query_prompt(spec["id"]),spec["choices"]) for spec in QUERY_SPECS}
    prompt_hashes={spec["id"]:hashlib.sha256(canonical_json(tokenizer(_query_prompt(spec["id"]),add_special_tokens=True)["input_ids"])).hexdigest() for spec in QUERY_SPECS}
    train_seeds=[derive_seed(TRAIN_SEED,f"latent-memory-train:{i}") for i in range(manifest()["data"]["train_worlds"])]
    losses=[];bridge.train();temperature=2.0
    for step in range(manifest()["data"]["steps"]):
        seed=train_seeds[step%len(train_seeds)];record=_world(seed,"train");z,_,_,rollout_text,_=_world_latents(record,world,device);spec=QUERY_SPECS[step%len(QUERY_SPECS)];prompt=_query_prompt(spec["id"]);choice=torch.tensor(token_map[spec["id"]],device=device)
        teacher=_teacher_logits(model,tokenizer,rollout_text,prompt,device)[:,choice].float();student=_student_logits(model,tokenizer,bridge,z,prompt,True)[:,choice].float();kl=F.kl_div(F.log_softmax(student/temperature,-1),F.softmax(teacher/temperature,-1),reduction="batchmean")*temperature**2;margin=F.mse_loss(student[:,0]-student[:,1],teacher[:,0]-teacher[:,1]);loss=kl+.25*margin
        if not torch.isfinite(loss):raise RuntimeError("non-finite bridge loss")
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
    if any(p.requires_grad for p in model.parameters()) or any(p.requires_grad for m in world for p in m.parameters()):raise RuntimeError("frozen component became trainable")
    bridge.eval();heldout=[derive_seed(HELDOUT_SEED,f"latent-memory-heldout:{i}") for i in range(manifest()["data"]["fresh_heldout_worlds"])];records=[];controls=("own_z","no_z","scrambled_z","wrong_world_z")
    for index,seed in enumerate(heldout):
        record=_world(seed,"id");wrong_seed=derive_seed(HELDOUT_SEED,f"latent-memory-wrong:{index}");wrong_record=_world(wrong_seed,"id");z,raw,digest,_,raster=_world_latents(record,world,device);wrong_z,_,_,_,_=_world_latents(wrong_record,world,device);permutation=list(range(LATENT_DIM));random.Random(derive_seed(HELDOUT_SEED,f"latent-memory-scramble:{index}")).shuffle(permutation);scrambled=z[...,permutation];arms={"own_z":z,"no_z":z,"scrambled_z":scrambled,"wrong_world_z":wrong_z};target=_target(seed);margins={arm:{} for arm in controls};deltas={arm:{} for arm in controls}
        with torch.no_grad():
            for spec in QUERY_SPECS:
                prompt=_query_prompt(spec["id"]);choice=torch.tensor(token_map[spec["id"]],device=device);correct=_target_choice(target,spec["id"]);alt=1-correct
                for arm in controls:
                    logits=_student_logits(model,tokenizer,bridge,arms[arm],prompt,arm!="no_z")[:,choice].float();margins[arm][spec["id"]]=float((logits[0,correct]-logits[0,alt]).cpu());deltas[arm][spec["id"]]=float((logits[0,0]-logits[0,1]).cpu())
        predictions={arm:_structured_prediction(value) for arm,value in deltas.items()};truth=target
        png=_png(raster[0].cpu());records.append({"index":index,"world_seed_sha256":hashlib.sha256(str(seed).encode()).hexdigest(),"world_latent_sha256":digest,"predictor_output_sha256":digest,"bridge_input_sha256":digest,"pixel_decoder_input_sha256":digest,"cache_world_latent_sha256":digest,"raw_bytes_sha256":hashlib.sha256(raw).hexdigest(),"decoded_png_sha256":hashlib.sha256(png).hexdigest(),"mean_target_logit_margin":{arm:sum(v.values())/len(v) for arm,v in margins.items()},"exact_answer":{arm:float(predictions[arm]==truth) for arm in controls},"parse":{arm:1.0 for arm in controls}})
        if index==0:(output_root/"sample-z.f32le.bin").write_bytes(raw);(output_root/"sample-decoded.png").write_bytes(png)
    comparisons={}
    for offset,control in enumerate(controls[1:]):
        margin=[r["mean_target_logit_margin"]["own_z"]-r["mean_target_logit_margin"][control] for r in records];exact=[r["exact_answer"]["own_z"]-r["exact_answer"][control] for r in records];mci=_paired_ci(margin,31001+offset);eci=_paired_ci(exact,32001+offset);comparisons[control]={"margin_mean":sum(margin)/len(margin),"margin_ci95":mci,"exact_mean":sum(exact)/len(exact),"exact_ci95":eci,"parse_delta":0.0,"passed":mci[0]>0 and eci[0]>0}
    passed=all(v["passed"] for v in comparisons.values());synthetic=torch.ones_like(z);spec=QUERY_SPECS[0];prompt=_query_prompt(spec["id"]);choice=torch.tensor(token_map[spec["id"]],device=device)
    with torch.no_grad():synthetic_delta=float((_student_logits(model,tokenizer,bridge,synthetic,prompt,True)[:,choice]-_student_logits(model,tokenizer,bridge,torch.zeros_like(synthetic),prompt,True)[:,choice]).abs().max().cpu())
    (output_root/"bridge").mkdir();save_file({k:v.detach().cpu() for k,v in bridge.state_dict().items()},output_root/"bridge/model.safetensors");(output_root/"bridge/config.json").write_bytes(canonical_json({"layers":list(LAYERS),"rank":RANK,"memory_tokens":MEMORY_TOKENS,"latent_dim":LATENT_DIM}));(output_root/"manifest.json").write_bytes(canonical_json(manifest()));(output_root/"evaluation.json").write_bytes(canonical_json(records));duration=time.monotonic()-started;terminal={"status":"completed","decision":"pass" if passed else "pivot","causal_runner_allowed":passed,"manifest_sha256":MANIFEST_SHA256,"code_sha":expected_code_sha,"base_revision":BASE_REVISION,"transformers_revision":TRANSFORMERS_REVISION,"source_world_manifest_sha256":WORLD_MANIFEST_SHA256,"train_seed_set_sha256":sha256_json(train_seeds),"heldout_seed_set_sha256":sha256_json(heldout),"base_parameters":base_parameters,"trainable_parameters":sum(p.numel() for p in bridge.parameters()),"initial_loss":losses[0],"final_loss":losses[-1],"comparisons":comparisons,"gate_values":{k:float(torch.sigmoid(v).detach().cpu()) for k,v in bridge.gates.items()},"bridge_parameter_norm":float(sum(p.detach().float().pow(2).sum() for p in bridge.parameters()).sqrt().cpu()),"synthetic_z_max_target_logit_delta":synthetic_delta,"prompt_token_hashes":prompt_hashes,"runtime_seconds":duration,"estimated_cost_usd":duration/3600*RATE_USD_PER_HOUR,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated()),"claims":{"informative_z":passed,"behavioral":passed,"causal":False,"mechanistic":False},"claim_label":manifest()["claim_label"]};(output_root/"terminal.json").write_bytes(canonical_json(terminal));artifacts=[artifact_declaration(p,output_root,role="latent-memory-bridge-evidence") for p in sorted(output_root.rglob("*")) if p.is_file()];evidence=write_task_evidence(output_root,metrics=[{"name":"initial_loss","value":losses[0]},{"name":"final_loss","value":losses[-1]},*[{"name":"own_margin_minus_control","condition":k,"value":v["margin_mean"]} for k,v in comparisons.items()]],artifacts=artifacts,track_h={"phase":"latent-memory-bridge-pilot","decision":terminal["decision"],"claims":terminal["claims"]});return {**terminal,"task_evidence":evidence}


def run_contract(expected_manifest_sha256,expected_code_sha):return ({"id":"latent-memory-bridge-pilot","_secret_keys":["HF_TOKEN"],"_preregistration":{"layer_allowlist":list(LAYERS),"timepoint_allowlist":["all_tokens"]}},{"id":"latent-memory-bridge-pilot","task":{"module":"jump_benchmark.latent_memory_bridge_task","parameters":{"expected_manifest_sha256":expected_manifest_sha256,"expected_code_sha":expected_code_sha}},"resources":{"gpu":"H100","timeout_seconds":3600},"selection":{"layers":list(LAYERS),"timepoints":["all_tokens"]},"retry":{"max_attempts":1}})
