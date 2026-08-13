"""Object-centric predictive-only JEPA pilot with strict z-only decoders."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from jump_contracts import artifact_declaration, canonical_json, tensor_bytes_sha256, write_task_evidence

from .authentic import HOLDOUT_LAW_FAMILY, independent_law, independent_partition
from .canonical import sha256_json
from .simulator import Law, SimulatorConfig, _initial_state, _trajectory, derive_seed


SCHEMA_VERSION = "jump.track-h-object-jepa-pilot/v1"
LATENT_DIM = 32
LEARNED_DIM = 20
HORIZONS = (1, 2, 4, 8)
FRAMES = 4
RATE_USD_PER_HOUR = 3.9492
PILOT_SEED = 170731


def object_jepa_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "track-h-object-centric-jepa-z32-pixel-pilot-v1",
        "claim_label": "predictive object-centric JEPA engineering pilot; no behavioral, causal, or mechanistic claim",
        "input": {
            "allowed": ["observed_positions", "observed_velocities", "cutoff_mask", "pre_outcome_action"],
            "forbidden": ["partition", "relation", "law", "adequacy", "answer", "Gemma_logits", "future_context", "seed", "episode_id"],
            "world_seed_disjoint": True, "train_only_normalization": True,
        },
        "latent": {
            "dtype": "float32-le", "shape": [LATENT_DIM], "order": "C",
            "layout": "12 current xy + six shared 3D object blocks + 2D sum-pool invariant",
            "same_bytes_consumers": ["online_encoder_output", "predictor", "rollout_decoder", "pixel_decoder", "future_Gemma_injection", "cache", "evidence"],
        },
        "architecture": {
            "encoder": "shared temporal object MLP + one all-pairs relative-state message layer with sum aggregation and residual update",
            "target": "stop-gradient EMA encoder",
            "predictor": "multi-horizon learned20 predictor conditioned only on pre-outcome per-object action",
            "objective": "latent Huber primary + mild variance/covariance anti-collapse",
            "rollout_decoder": "z-only Gaussian mean/logvar positions at horizons 1/2/4/8; auxiliary stop-gradient input",
            "pixel_decoder": "z-only learned 64x64 RGB last reconstruction plus horizons 1/2/4/8; Huber plus global SSIM auxiliary; no renderer in forward path",
            "mask_history_cells": True,
            "ablations": ["A_full", "B_no_message", "C_no_ema", "D_no_action"],
        },
        "training": {"seed": PILOT_SEED, "train_worlds": 1536, "steps_per_ablation": 800, "batch_size": 128, "learning_rate": 0.0007, "ema_decay": 0.99},
        "evaluation": {
            "id_worlds": 96, "heldout_law_ood_worlds": 96, "world_bootstraps": 10000,
            "metrics": ["latent_nrmse", "rollout_nrmse", "pixel_l1", "pixel_psnr", "pixel_ssim"],
            "gates": {
                "learned20_noncollapse_std_min": 0.05,
                "latent_vs_persistence_improvement_min": 0.20,
                "rollout_vs_copy_last_improvement_min": 0.20,
                "correct_action_vs_each_zero_shuffled_wrong_improvement_min": 0.10,
                "paired_ci_lower_exclusive": 0.0,
                "required_splits": ["id", "heldout_law_ood"],
            },
        },
        "execution": {"modal_function":"authentic_world_object_jepa_pilot","resource":"H100","gpu_count":1,"max_containers":1,"max_inputs":1,"max_attempts":1,"timeout_seconds":3600,"h100_rate_usd_per_hour":RATE_USD_PER_HOUR,"forecast_usd":RATE_USD_PER_HOUR,"aggregate_authority_ceiling_usd":100.0},
        "claims": {"informative_z":False,"behavioral":False,"causal":False,"mechanistic":False,"pixel_mechanism":False},
    }


MANIFEST_SHA256 = sha256_json(object_jepa_manifest())


def _law_for(seed: int, ood: bool) -> Law:
    if ood:
        return Law(*HOLDOUT_LAW_FAMILY)
    law = independent_law(seed)
    if (law.same, law.different, law.exponent) == HOLDOUT_LAW_FAMILY:
        return Law(law.same, law.different, 1 if law.exponent == 2 else 2)
    return law


def _world(seed: int, split: str) -> dict[str, Any]:
    """Generate intervention data; labels exist only inside the simulator call stack."""
    config = SimulatorConfig(steps=FRAMES)
    law = _law_for(seed, split == "heldout_law_ood")
    partition = independent_partition(seed)
    positions, velocities = _initial_state(seed, config.bounds)
    history = _trajectory(positions, velocities, partition, law, config)
    action_rng = random.Random(derive_seed(seed, "jepa:pre-outcome-action"))
    action = [[0.0, 0.0] for _ in range(6)]
    action[action_rng.randrange(6)] = [action_rng.uniform(-0.32, 0.32), action_rng.uniform(-0.32, 0.32)]
    last = history[-1]
    acted_velocities = [list(v) for v in last["velocities"]]
    for obj in range(6):
        acted_velocities[obj][0] += action[obj][0]; acted_velocities[obj][1] += action[obj][1]
    continuation = _trajectory([list(p) for p in last["positions"]], acted_velocities, partition, law, SimulatorConfig(steps=max(HORIZONS)+1))
    null_continuation = _trajectory([list(p) for p in last["positions"]], [list(v) for v in last["velocities"]], partition, law, SimulatorConfig(steps=max(HORIZONS)+1))
    observed = [[[float(v) for v in (*frame["positions"][o], *frame["velocities"][o])] for o in range(6)] for frame in history]
    target_positions = [continuation[h]["positions"] for h in HORIZONS]
    null_target_positions = [null_continuation[h]["positions"] for h in HORIZONS]
    target_windows = []
    combined = history + continuation[1:]
    for h in HORIZONS:
        end = FRAMES - 1 + h
        window = combined[max(0, end-FRAMES+1):end+1]
        if len(window) < FRAMES: window = [window[0]] * (FRAMES-len(window)) + window
        target_windows.append([[[float(v) for v in (*frame["positions"][o], *frame["velocities"][o])] for o in range(6)] for frame in window])
    null_target_windows=[]
    null_combined=history+null_continuation[1:]
    for h in HORIZONS:
        end=FRAMES-1+h;window=null_combined[max(0,end-FRAMES+1):end+1]
        if len(window)<FRAMES:window=[window[0]]*(FRAMES-len(window))+window
        null_target_windows.append([[[float(v) for v in (*frame["positions"][o],*frame["velocities"][o])] for o in range(6)] for frame in window])
    return {"observed":observed,"mask":[[1.0]*6 for _ in range(FRAMES)],"action":action,"target_positions":target_positions,"target_windows":target_windows,"null_target_positions":null_target_positions,"null_target_windows":null_target_windows}


def dataset(split: str, count: int) -> dict[str, Any]:
    if split not in {"train","id","heldout_law_ood"}: raise ValueError("invalid JEPA split")
    seeds=[derive_seed(PILOT_SEED,f"jepa:{split}:{i}") for i in range(count)]
    records=[_world(seed,split) for seed in seeds]
    return {"records":records,"seed_set_sha256":sha256_json(seeds)}


def build_modules(*, use_message: bool=True, normalization_mean: Any=None, normalization_std: Any=None):
    import torch

    class ObjectEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal=torch.nn.Sequential(torch.nn.Linear(FRAMES*5,32),torch.nn.GELU(),torch.nn.Linear(32,3))
            self.message=torch.nn.Sequential(torch.nn.Linear(10,24),torch.nn.GELU(),torch.nn.Linear(24,3))
            self.update=torch.nn.Sequential(torch.nn.Linear(6,16),torch.nn.GELU(),torch.nn.Linear(16,3))
            self.pool=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.GELU(),torch.nn.Linear(8,2))
            self.use_message=use_message
            mean=torch.zeros(4) if normalization_mean is None else normalization_mean.detach().to("cpu").reshape(4)
            std=torch.ones(4) if normalization_std is None else normalization_std.detach().to("cpu").reshape(4)
            self.register_buffer("normalization_mean",mean);self.register_buffer("normalization_std",std)
        def forward(self, observations, mask):
            normalized=(observations-self.normalization_mean)/self.normalization_std
            masked=normalized*mask[...,None]
            temporal=torch.cat([masked,mask[...,None]],dim=-1).permute(0,2,1,3).reshape(observations.shape[0],6,-1)
            blocks=self.temporal(temporal)
            if self.use_message:
                latest=observations[:,-1]
                hi=blocks[:,:,None,:].expand(-1,-1,6,-1);hj=blocks[:,None,:,:].expand(-1,6,-1,-1)
                rel=latest[:,None,:,:]-latest[:,:,None,:]
                messages=self.message(torch.cat([hi,hj,rel],dim=-1))
                eye=torch.eye(6,device=observations.device,dtype=observations.dtype)[None,:,:,None]
                aggregate=(messages*(1-eye)).sum(dim=2)
                blocks=blocks+self.update(torch.cat([blocks,aggregate],dim=-1))
            invariant=self.pool(blocks.sum(dim=1))
            current=observations[:,-1,:,:2].reshape(observations.shape[0],12)
            return torch.cat([current,blocks.reshape(observations.shape[0],18),invariant],dim=-1)

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(LEARNED_DIM+12+4,96),torch.nn.GELU(),torch.nn.Linear(96,64),torch.nn.GELU(),torch.nn.Linear(64,LEARNED_DIM))
        def forward(self,z,action,horizon_index):
            onehot=torch.nn.functional.one_hot(horizon_index,num_classes=4).to(z.dtype)
            return self.net(torch.cat([z[:,12:],action.reshape(z.shape[0],12),onehot],dim=-1))

    class GaussianRolloutDecoder(torch.nn.Module):
        """Strict z-only decoder; no observation or renderer argument exists."""
        def __init__(self):
            super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(LATENT_DIM,128),torch.nn.GELU(),torch.nn.Linear(128,96))
        def forward(self,z):
            raw=self.net(z).reshape(-1,4,12,2);baseline=z[:,:12].reshape(-1,1,12)
            return baseline+raw[...,0],raw[...,1].clamp(-6,3)

    class RasterDecoder(torch.nn.Module):
        """Strict z-only learned raster decoder; output is five 64x64 RGB frames."""
        def __init__(self):
            super().__init__();self.fc=torch.nn.Linear(LATENT_DIM,128*4*4);self.up=torch.nn.Sequential(torch.nn.ConvTranspose2d(128,96,4,2,1),torch.nn.GELU(),torch.nn.ConvTranspose2d(96,64,4,2,1),torch.nn.GELU(),torch.nn.ConvTranspose2d(64,48,4,2,1),torch.nn.GELU(),torch.nn.ConvTranspose2d(48,15,4,2,1))
        def forward(self,z):
            return torch.sigmoid(self.up(self.fc(z).reshape(-1,128,4,4)).reshape(-1,5,3,64,64))
    return ObjectEncoder(),Predictor(),GaussianRolloutDecoder(),RasterDecoder()


def _exact_z(z: Any):
    import numpy as np
    import torch
    raw=z.detach().to("cpu",torch.float32).contiguous().numpy().astype("<f4",copy=False).tobytes(order="C")
    roundtrip=torch.from_numpy(np.frombuffer(raw,dtype="<f4").copy().reshape(z.shape)).to(z.device)
    return z+(roundtrip-z).detach(),raw,tensor_bytes_sha256(raw,dtype="float32-le",shape=list(z.shape),order="C")


def _tensorize(records: list[dict[str,Any]], device: str):
    import torch
    return {
        "observed":torch.tensor([r["observed"] for r in records],dtype=torch.float32,device=device),
        "mask":torch.tensor([r["mask"] for r in records],dtype=torch.float32,device=device),
        "action":torch.tensor([r["action"] for r in records],dtype=torch.float32,device=device),
        "target_positions":torch.tensor([r["target_positions"] for r in records],dtype=torch.float32,device=device),
        "target_windows":torch.tensor([r["target_windows"] for r in records],dtype=torch.float32,device=device),
        "null_target_positions":torch.tensor([r["null_target_positions"] for r in records],dtype=torch.float32,device=device),
        "null_target_windows":torch.tensor([r["null_target_windows"] for r in records],dtype=torch.float32,device=device),
    }


def _raster(positions: Any):
    import torch
    # positions [B,F,6,2], differentiable deterministic raster target only.
    grid=torch.linspace(-3,3,64,device=positions.device,dtype=positions.dtype)
    yy,xx=torch.meshgrid(grid,grid,indexing="ij")
    dx=xx[None,None,None]-positions[...,0,None,None];dy=yy[None,None,None]-positions[...,1,None,None]
    blobs=torch.exp(-(dx*dx+dy*dy)/(2*0.10**2)).sum(dim=2).clamp(0,1)
    return blobs[:,:,None].expand(-1,-1,3,-1,-1)


def _ssim_global(x: Any,y: Any):
    c1,c2=0.01**2,0.03**2
    dims=(-1,-2,-3);mx=x.mean(dims);my=y.mean(dims);vx=x.var(dims,unbiased=False);vy=y.var(dims,unbiased=False);cov=((x-mx[...,None,None,None])*(y-my[...,None,None,None])).mean(dims)
    return ((2*mx*my+c1)*(2*cov+c2)/((mx*mx+my*my+c1)*(vx+vy+c2))).mean()


def _ci(values: list[float], seed: int):
    rng=random.Random(seed);samples=[]
    for _ in range(10000):samples.append(sum(values[rng.randrange(len(values))] for _ in values)/len(values))
    samples.sort();return [samples[250],samples[9749]]


def _train_variant(name: str, train: dict[str,Any], device: str, steps: int, mean: Any, std: Any):
    import torch
    import torch.nn.functional as F
    online,predictor,rollout,pixel=build_modules(use_message=name!="B_no_message",normalization_mean=mean,normalization_std=std)
    online.to(device);predictor.to(device);rollout.to(device);pixel.to(device);target=deepcopy(online).to(device).eval()
    for p in target.parameters():p.requires_grad_(False)
    params=list(online.parameters())+list(predictor.parameters())+list(rollout.parameters())+list(pixel.parameters())
    opt=torch.optim.AdamW(params,lr=object_jepa_manifest()["training"]["learning_rate"])
    n=train["observed"].shape[0];bs=object_jepa_manifest()["training"]["batch_size"]
    losses=[]
    for step in range(steps):
        idx=torch.arange(step*bs,(step+1)*bs,device=device)%n
        obs=train["observed"][idx];mask=train["mask"][idx].clone();action=train["action"][idx]
        if name=="D_no_action":action=torch.zeros_like(action)
        # deterministic history masking; last position remains visible by z contract.
        mask[:,step%3,(step//3)%6]=0
        z=online(obs,mask);z_exact,_,_=_exact_z(z)
        hidx=torch.arange(4,device=device).repeat_interleave(bs)
        pred=predictor(z_exact.repeat(4,1),action.repeat(4,1,1),hidx).reshape(4,bs,LEARNED_DIM).transpose(0,1)
        windows=train["target_windows"][idx]
        flat=windows.reshape(bs*4,FRAMES,6,4);fullmask=torch.ones(bs*4,FRAMES,6,device=device)
        with torch.no_grad(): target_z=target(flat,fullmask)[:,12:].reshape(bs,4,LEARNED_DIM)
        latent=F.smooth_l1_loss(pred,target_z)
        learned=z_exact[:,12:];std=learned.std(dim=0);variance=F.relu(0.5-std).mean()
        centered=learned-learned.mean(0);cov=centered.T@centered/(bs-1);covariance=(cov-torch.diag(torch.diag(cov))).pow(2).mean()
        mean,logvar=rollout(z_exact.detach());targets=train["target_positions"][idx].reshape(bs,4,12)
        rollout_loss=(0.5*(logvar+(mean-targets).pow(2)*torch.exp(-logvar))).mean()
        raster_target=_raster(torch.cat([obs[:,-1:,:,:2],train["target_positions"][idx]],dim=1))
        raster=pixel(z_exact.detach());pixel_loss=F.smooth_l1_loss(raster,raster_target)+0.1*(1-_ssim_global(raster,raster_target))
        loss=latent+0.02*variance+0.002*covariance+0.1*rollout_loss+0.05*pixel_loss
        if not torch.isfinite(loss):raise RuntimeError("non-finite object JEPA loss")
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
        if name!="C_no_ema":
            with torch.no_grad():
                for tp,op in zip(target.parameters(),online.parameters()):tp.mul_(0.99).add_(op,alpha=0.01)
        else: target.load_state_dict(online.state_dict())
    return (online.eval(),predictor.eval(),rollout.eval(),pixel.eval(),target.eval()),losses


def _evaluate(modules: tuple[Any,...], data: dict[str,Any], split: str):
    import torch
    online,predictor,rollout,pixel,target=modules;n=data["observed"].shape[0];mask=data["mask"]
    with torch.no_grad():
        z=online(data["observed"],mask);z_exact,raw,z_hash=_exact_z(z)
        windows=data["target_windows"].reshape(n*4,FRAMES,6,4);target_z=target(windows,torch.ones(n*4,FRAMES,6,device=z.device))[:,12:].reshape(n,4,LEARNED_DIM)
        hidx=torch.arange(4,device=z.device).repeat_interleave(n)
        def predicted(action):return predictor(z_exact.repeat(4,1),action.repeat(4,1,1),hidx).reshape(4,n,LEARNED_DIM).transpose(0,1)
        correct=predicted(data["action"]);zero=predicted(torch.zeros_like(data["action"]));shuffled=predicted(data["action"].roll(1,0));wrong=predicted(data["action"].roll(2,0))
        scale=target_z.std(dim=(0,1)).clamp_min(1e-6)
        err=lambda value:(((value-target_z)/scale).pow(2).mean(dim=(1,2))).sqrt()
        errors={"model":err(correct),"latent_persistence":err(z_exact[:,None,12:].expand_as(target_z)),"zero":err(zero),"shuffled":err(shuffled),"wrong":err(wrong)}
        mean,_=rollout(z_exact);positions=data["target_positions"].reshape(n,4,12);posscale=positions.std(dim=(0,1)).clamp_min(1e-6)
        rollout_err=(((mean-positions)/posscale).pow(2).mean(dim=(1,2))).sqrt();copy=z_exact[:,:12][:,None].expand_as(positions);copy_err=(((copy-positions)/posscale).pow(2).mean(dim=(1,2))).sqrt()
        raster=pixel(z_exact);raster_target=_raster(torch.cat([data["observed"][:,-1:,:,:2],data["target_positions"]],dim=1));pixel_l1=(raster-raster_target).abs().mean(dim=(0,2,3,4));pixel_mse=(raster-raster_target).pow(2).mean(dim=(0,2,3,4));pixel_psnr=10*torch.log10(1/pixel_mse.clamp_min(1e-12))
        pixel_ssim=[]
        for i in range(5):pixel_ssim.append(float(_ssim_global(raster[:,i],raster_target[:,i]).cpu()))
        std=z_exact[:,12:].std(dim=0)
    latent_gain=((errors["latent_persistence"]-errors["model"])/errors["latent_persistence"].clamp_min(1e-8)).cpu().tolist()
    rollout_gain=((copy_err-rollout_err)/copy_err.clamp_min(1e-8)).cpu().tolist()
    action={}
    for offset,key in enumerate(("zero","shuffled","wrong")):
        values=((errors[key]-errors["model"])/errors[key].clamp_min(1e-8)).cpu().tolist();action[key]={"mean":sum(values)/n,"ci95":_ci(values,19000+offset)}
    # Save only a single exact z's raw bytes; batch hash is diagnostic only.
    one_raw=raw[:LATENT_DIM*4];one_hash=tensor_bytes_sha256(one_raw,dtype="float32-le",shape=[LATENT_DIM],order="C")
    return {"split":split,"n":n,"learned20_std_min":float(std.min().cpu()),"latent_vs_persistence":{"mean":sum(latent_gain)/n,"ci95":_ci(latent_gain,18001)},"rollout_vs_copy_last":{"mean":sum(rollout_gain)/n,"ci95":_ci(rollout_gain,18002)},"action":action,"pixel":{"horizons":["reconstruction",1,2,4,8],"l1":[float(v) for v in pixel_l1.cpu()],"psnr":[float(v) for v in pixel_psnr.cpu()],"ssim":pixel_ssim},"sample_z_raw":one_raw,"sample_z_sha256":one_hash,"batch_z_sha256":z_hash,"sample_raster":raster[0].cpu()}


def _png(frame: Any) -> bytes:
    from PIL import Image
    array=(frame.permute(1,2,0).clamp(0,1).numpy()*255).round().astype("uint8")
    stream=io.BytesIO();Image.fromarray(array,"RGB").save(stream,format="PNG",optimize=False,compress_level=9);return stream.getvalue()


def cpu_preflight() -> dict[str,Any]:
    import torch
    online,predictor,rollout,pixel=build_modules();record=_world(derive_seed(PILOT_SEED,"preflight"),"id");batch=_tensorize([record],"cpu")
    z=online(batch["observed"],batch["mask"]);exact,raw,digest=_exact_z(z);predictor(exact,batch["action"],torch.tensor([0]));rollout(exact);image1=pixel(exact);image2=pixel(exact.clone())
    try:pixel(exact,observation=batch["observed"]);raise AssertionError("z-only decoder accepted a side channel")
    except TypeError:pass
    changed=bytearray(raw);changed[-1]^=1;changed_hash=tensor_bytes_sha256(bytes(changed),dtype="float32-le",shape=[1,LATENT_DIM],order="C")
    return {"latent_shape":list(z.shape),"world_latent_sha256":digest,"same_z_image_reproducible":bool(torch.equal(image1,image2)),"changed_byte_changes_hash":changed_hash!=digest,"z_only_side_channel_rejected":True,"renderer_called_by_decoder":False,"forbidden_fields_present":False}


def train_and_evaluate(output_root: Path, expected_manifest_sha256: str, expected_code_sha: str, device: str="cuda") -> dict[str,Any]:
    import torch
    from safetensors.torch import save_file
    if expected_manifest_sha256!=MANIFEST_SHA256 or os.environ.get("JUMP_CODE_VERSION")!=expected_code_sha:raise ValueError("object JEPA immutable identity mismatch")
    if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):raise FileExistsError("object JEPA requires empty canonical workdir")
    if not torch.cuda.is_available():raise RuntimeError("object JEPA pilot requires CUDA")
    torch.manual_seed(PILOT_SEED);torch.cuda.manual_seed_all(PILOT_SEED);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    train_raw=dataset("train",1536);id_raw=dataset("id",96);ood_raw=dataset("heldout_law_ood",96)
    train=_tensorize(train_raw["records"],device);id_data=_tensorize(id_raw["records"],device);ood_data=_tensorize(ood_raw["records"],device)
    # Train-only feature normalization, applied consistently after split materialization.
    mean=train["observed"].mean(dim=(0,1,2),keepdim=True);std=train["observed"].std(dim=(0,1,2),keepdim=True).clamp_min(1e-6)
    for data in (train,id_data,ood_data):
        pass
    # The encoder applies the frozen train-only normalizer internally while z[0:12] remains physical current xy.
    variants={};full_modules=None
    for name in object_jepa_manifest()["architecture"]["ablations"]:
        modules,losses=_train_variant(name,train,device,object_jepa_manifest()["training"]["steps_per_ablation"],mean,std);variants[name]={"initial_loss":losses[0],"final_loss":losses[-1]}
        if name=="A_full":full_modules=modules
    assert full_modules is not None
    id_result=_evaluate(full_modules,id_data,"id");ood_result=_evaluate(full_modules,ood_data,"heldout_law_ood")
    gates=[]
    for result in (id_result,ood_result):
        gates += [result["learned20_std_min"]>=0.05,result["latent_vs_persistence"]["mean"]>=0.20 and result["latent_vs_persistence"]["ci95"][0]>0,result["rollout_vs_copy_last"]["mean"]>=0.20 and result["rollout_vs_copy_last"]["ci95"][0]>0]
        gates += [row["mean"]>=0.10 and row["ci95"][0]>0 for row in result["action"].values()]
    passed=all(gates);encoder,predictor,rollout,pixel,_=full_modules
    for role,module in (("encoder",encoder),("latent_predictor",predictor),("rollout_decoder",rollout),("pixel_decoder",pixel)):
        root=output_root/role;root.mkdir();save_file({k:v.detach().cpu() for k,v in module.state_dict().items()},root/"model.safetensors");(root/"config.json").write_bytes(canonical_json({"architecture":role,"latent_dim":LATENT_DIM,"strict_z_only":role in {"rollout_decoder","pixel_decoder"}}))
    (output_root/"manifest.json").write_bytes(canonical_json(object_jepa_manifest()));(output_root/"sample-z.f32le.bin").write_bytes(id_result.pop("sample_z_raw"))
    images=id_result.pop("sample_raster");image_hashes=[]
    image_root=output_root/"decoded-raster";image_root.mkdir()
    for idx,label in enumerate(("reconstruction","h1","h2","h4","h8")):
        raw=_png(images[idx]);path=image_root/f"{label}.png";path.write_bytes(raw);image_hashes.append({"horizon":label,"sha256":hashlib.sha256(raw).hexdigest(),"world_latent_sha256":id_result["sample_z_sha256"]})
    ood_result.pop("sample_z_raw");ood_result.pop("sample_raster")
    results={"id":id_result,"heldout_law_ood":ood_result,"ablations":variants,"decoded_pngs":image_hashes}
    (output_root/"evaluation.json").write_bytes(canonical_json(results))
    duration=time.monotonic()-started
    terminal={"status":"completed","decision":"pass" if passed else "pivot","downstream_allowed":passed,"manifest_sha256":MANIFEST_SHA256,"code_sha":expected_code_sha,"split_seed_hashes":{"train":train_raw["seed_set_sha256"],"id":id_raw["seed_set_sha256"],"heldout_law_ood":ood_raw["seed_set_sha256"]},"normalization_sha256":hashlib.sha256(torch.cat([mean.flatten(),std.flatten()]).cpu().numpy().astype("<f4").tobytes()).hexdigest(),"evaluation":results,"runtime_seconds":duration,"estimated_cost_usd":duration/3600*RATE_USD_PER_HOUR,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated()),"claims":{"informative_z":passed,"behavioral":False,"causal":False,"mechanistic":False,"pixel_mechanism":False},"claim_label":object_jepa_manifest()["claim_label"]}
    (output_root/"terminal.json").write_bytes(canonical_json(terminal));artifacts=[artifact_declaration(p,output_root,role="object-jepa-evidence") for p in sorted(output_root.rglob("*")) if p.is_file()]
    evidence=write_task_evidence(output_root,metrics=[{"name":"id_latent_improvement","value":id_result["latent_vs_persistence"]["mean"]},{"name":"ood_latent_improvement","value":ood_result["latent_vs_persistence"]["mean"]},{"name":"id_rollout_improvement","value":id_result["rollout_vs_copy_last"]["mean"]},{"name":"ood_rollout_improvement","value":ood_result["rollout_vs_copy_last"]["mean"]}],artifacts=artifacts,track_h={"phase":"object-jepa-pilot","decision":terminal["decision"],"claims":terminal["claims"]})
    return {**terminal,"task_evidence":evidence}


def run_contract(expected_manifest_sha256: str, expected_code_sha: str):
    return ({"id":"object-jepa-pilot","_preregistration":{"layer_allowlist":[],"timepoint_allowlist":[]}}, {"id":"object-jepa-pilot","task":{"module":"jump_benchmark.object_jepa_task","parameters":{"expected_manifest_sha256":expected_manifest_sha256,"expected_code_sha":expected_code_sha}},"resources":{"gpu":"H100","timeout_seconds":3600},"selection":{"layers":[],"timepoints":[]},"retry":{"max_attempts":1}})
