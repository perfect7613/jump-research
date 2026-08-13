"""Action-gated residual-transition successor to the frozen object-JEPA pilot."""
from __future__ import annotations
import hashlib,json,os,time
from copy import deepcopy
from pathlib import Path
from typing import Any
from jump_contracts import artifact_declaration,canonical_json,tensor_bytes_sha256,write_task_evidence
from .canonical import sha256_json
from .object_jepa import (
    FRAMES,HORIZONS,LATENT_DIM,LEARNED_DIM,PILOT_SEED,RATE_USD_PER_HOUR,
    _ci,_exact_z,_png,_raster,_ssim_global,_tensorize,build_modules,dataset,
)

SCHEMA_VERSION="jump.track-h-object-jepa-residual-pilot/v1"

def manifest()->dict[str,Any]:
    return {
        "schema_version":SCHEMA_VERSION,"experiment_id":"track-h-object-jepa-action-residual-pilot-v1",
        "claim_label":"predictive action-gated residual JEPA engineering pilot; no behavioral, causal, or mechanistic claim",
        "frozen_from_predecessor":{"manifest_sha256":"eb3fa356c6483e0a4dc835066b302786b3f77cabd57e5eb645ca9a67bd54c31e","encoder":"shared temporal object encoder and one all-pairs sum-message layer","latent":{"dtype":"float32-le","shape":[32],"order":"C","layout":"12 xy + six shared 3D blocks + 2D invariant pool"},"ema_target":True,"splits_and_normalization":True,"gaussian_rollout_decoder":True,"pixel_decoder":True},
        "only_architecture_change":"predictor is action-FiLM all-pairs object-message residual transition z_h=z0+Delta_h(z0,a)",
        "training":{"seed":PILOT_SEED,"train_worlds":1536,"steps":1200,"batch_size":128,"learning_rate":0.0007,"ema_decay":0.99,"balanced_arms":["intervention","matched_null"],"primary":"EMA latent-difference Huber","rollout_auxiliary":"Gaussian NLL with predicted z_h detached","pixel_auxiliary":"Huber+global SSIM with predicted z_h detached"},
        "leakage":{"allowed":["observed_trajectories","cutoff_masks","pre_outcome_action"],"forbidden":["law","partition","relation","answer","adequacy","Gemma_logits","future_context"],"labels_used":False},
        "evaluation":{"id_worlds":96,"heldout_law_ood_worlds":96,"bootstraps":10000,"gates":{"learned20_noncollapse_std_min":0.05,"latent_vs_persistence_improvement_min":0.20,"rollout_vs_copy_last_improvement_min":0.20,"correct_action_vs_each_zero_shuffled_wrong_improvement_min":0.10,"paired_ci_lower_exclusive":0.0,"required_all_splits":True}},
        "execution":{"modal_function":"authentic_world_object_jepa_residual_pilot","resource":"H100","gpu_count":1,"max_containers":1,"max_inputs":1,"max_attempts":1,"timeout_seconds":3600,"h100_rate_usd_per_hour":RATE_USD_PER_HOUR,"forecast_usd":RATE_USD_PER_HOUR,"aggregate_authority_ceiling_usd":100.0},
        "claims":{"informative_z":False,"behavioral":False,"causal":False,"mechanistic":False,"pixel_mechanism":False},
    }

MANIFEST_SHA256=sha256_json(manifest())

def build_predictor():
    import torch
    class ResidualTransition(torch.nn.Module):
        def __init__(self):
            super().__init__();self.film=torch.nn.Linear(2,6);self.message=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.GELU(),torch.nn.Linear(32,3));self.update=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.GELU(),torch.nn.Linear(32,3));self.xy=torch.nn.Sequential(torch.nn.Linear(5,24),torch.nn.GELU(),torch.nn.Linear(24,2));self.pool=torch.nn.Sequential(torch.nn.Linear(20,32),torch.nn.GELU(),torch.nn.Linear(32,2));self.horizon=torch.nn.Embedding(4,3)
        def forward(self,z,action,horizon_index):
            blocks=z[:,12:30].reshape(-1,6,3);film=self.film(action).reshape(-1,6,2,3);conditioned=blocks*(1+torch.tanh(film[:,:,0]))+film[:,:,1]+self.horizon(horizon_index)[:,None]
            left=conditioned[:,:,None,:].expand(-1,-1,6,-1);right=conditioned[:,None,:,:].expand(-1,6,-1,-1);relative_action=action[:,None,:,:]-action[:,:,None,:]
            messages=self.message(torch.cat([left,right,relative_action],dim=-1));eye=torch.eye(6,device=z.device,dtype=z.dtype)[None,:,:,None];aggregate=(messages*(1-eye)).sum(2)
            block_delta=self.update(torch.cat([conditioned,aggregate,action],dim=-1));xy_delta=self.xy(torch.cat([conditioned,action],dim=-1)).reshape(-1,12);pool_delta=self.pool(torch.cat([block_delta.reshape(-1,18),block_delta.sum(1)[:,:2]],dim=-1))
            return z+torch.cat([xy_delta,block_delta.reshape(-1,18),pool_delta],dim=-1)
    return ResidualTransition()

def _train(train:dict[str,Any],device:str,mean:Any,std:Any):
    import torch
    import torch.nn.functional as F
    online,_,rollout,pixel=build_modules(normalization_mean=mean,normalization_std=std);predictor=build_predictor();online.to(device);predictor.to(device);rollout.to(device);pixel.to(device);target=deepcopy(online).to(device).eval()
    for p in target.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW([*online.parameters(),*predictor.parameters(),*rollout.parameters(),*pixel.parameters()],lr=manifest()["training"]["learning_rate"]);n=train["observed"].shape[0];bs=manifest()["training"]["batch_size"];losses=[]
    for step in range(manifest()["training"]["steps"]):
        idx=torch.arange(step*bs,(step+1)*bs,device=device)%n;obs=train["observed"][idx];mask=train["mask"][idx].clone();mask[:,step%3,(step//3)%6]=0;z=online(obs,mask);z0,_,_=_exact_z(z)
        action=torch.cat([train["action"][idx],torch.zeros_like(train["action"][idx])]);base=torch.cat([z0,z0]);windows=torch.cat([train["target_windows"][idx],train["null_target_windows"][idx]]);positions=torch.cat([train["target_positions"][idx],train["null_target_positions"][idx]])
        hidx=torch.arange(4,device=device).repeat_interleave(2*bs);pred=predictor(base.repeat(4,1),action.repeat(4,1,1),hidx).reshape(4,2*bs,LATENT_DIM).transpose(0,1)
        flat=windows.reshape(2*bs*4,FRAMES,6,4)
        with torch.no_grad():target_z=target(flat,torch.ones(2*bs*4,FRAMES,6,device=device)).reshape(2*bs,4,LATENT_DIM)
        latent=F.smooth_l1_loss(pred-base[:,None],target_z-base[:,None]);learned=z0[:,12:];variance=F.relu(0.5-learned.std(0)).mean();centered=learned-learned.mean(0);cov=centered.T@centered/(bs-1);covguard=(cov-torch.diag(torch.diag(cov))).pow(2).mean()
        decoded=[];pixels=[]
        for h in range(4):
            zh,_,_=_exact_z(pred[:,h].detach());means,_=rollout(zh);decoded.append(means[:,h]);pixels.append(pixel(zh)[:,h+1])
        decoded=torch.stack(decoded,1);pixel_pred=torch.stack(pixels,1);position_flat=positions.reshape(2*bs,4,12);rollout_loss=F.smooth_l1_loss(decoded,position_flat);pixel_target=_raster(positions);pixel_loss=F.smooth_l1_loss(pixel_pred,pixel_target)+0.1*(1-_ssim_global(pixel_pred,pixel_target))
        loss=latent+0.02*variance+0.002*covguard+0.1*rollout_loss+0.05*pixel_loss
        if not torch.isfinite(loss):raise RuntimeError("non-finite residual JEPA loss")
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            for tp,op in zip(target.parameters(),online.parameters()):tp.mul_(0.99).add_(op,alpha=0.01)
    return (online.eval(),predictor.eval(),rollout.eval(),pixel.eval(),target.eval()),losses

def _evaluate(modules,data,split):
    import torch
    online,predictor,rollout,pixel,target=modules;n=data["observed"].shape[0]
    with torch.no_grad():
        z=online(data["observed"],data["mask"]);z0,raw,z0hash=_exact_z(z);windows=data["target_windows"].reshape(n*4,FRAMES,6,4);target_z=target(windows,torch.ones(n*4,FRAMES,6,device=z.device)).reshape(n,4,LATENT_DIM);hidx=torch.arange(4,device=z.device).repeat_interleave(n)
        def pred(action):return predictor(z0.repeat(4,1),action.repeat(4,1,1),hidx).reshape(4,n,LATENT_DIM).transpose(0,1)
        variants={"model":pred(data["action"]),"zero":pred(torch.zeros_like(data["action"])),"shuffled":pred(data["action"].roll(1,0)),"wrong":pred(data["action"].roll(2,0))};scale=target_z.std((0,1)).clamp_min(1e-6);err=lambda v:(((v-target_z)/scale).pow(2).mean((1,2))).sqrt();errors={k:err(v) for k,v in variants.items()};persistence=err(z0[:,None].expand_as(target_z))
        correct=variants["model"];decoded=[];pixel_frames=[];zh_hashes=[]
        for h in range(4):
            zh,zhraw,zhhash=_exact_z(correct[:,h]);zh_hashes.append(zhhash);means,_=rollout(zh);decoded.append(means[:,h]);pixel_frames.append(pixel(zh)[:,h+1])
        decoded=torch.stack(decoded,1);pixel_frames=torch.stack(pixel_frames,1);positions=data["target_positions"].reshape(n,4,12);posscale=positions.std((0,1)).clamp_min(1e-6);rollout_err=(((decoded-positions)/posscale).pow(2).mean((1,2))).sqrt();copy=(((z0[:,:12,None] if False else z0[:,:12][:,None].expand_as(positions))-positions)/posscale).pow(2).mean((1,2)).sqrt();pixel_target=_raster(data["target_positions"]);l1=(pixel_frames-pixel_target).abs().mean((0,2,3,4));mse=(pixel_frames-pixel_target).pow(2).mean((0,2,3,4));psnr=10*torch.log10(1/mse.clamp_min(1e-12));ssim=[float(_ssim_global(pixel_frames[:,i],pixel_target[:,i]).cpu()) for i in range(4)];std=z0[:,12:].std(0)
    latent_values=((persistence-errors["model"])/persistence.clamp_min(1e-8)).cpu().tolist();rollout_values=((copy-rollout_err)/copy.clamp_min(1e-8)).cpu().tolist();actions={}
    for offset,key in enumerate(("zero","shuffled","wrong")):
        vals=((errors[key]-errors["model"])/errors[key].clamp_min(1e-8)).cpu().tolist();actions[key]={"mean":sum(vals)/n,"ci95":_ci(vals,22000+offset)}
    one_raw=raw[:LATENT_DIM*4];return {"split":split,"n":n,"learned20_std_min":float(std.min().cpu()),"latent_vs_persistence":{"mean":sum(latent_values)/n,"ci95":_ci(latent_values,21001)},"rollout_vs_copy_last":{"mean":sum(rollout_values)/n,"ci95":_ci(rollout_values,21002)},"action":actions,"pixel":{"horizons":[1,2,4,8],"l1":[float(x) for x in l1.cpu()],"psnr":[float(x) for x in psnr.cpu()],"ssim":ssim},"z0_raw":one_raw,"z0_sha256":tensor_bytes_sha256(one_raw,dtype="float32-le",shape=[LATENT_DIM],order="C"),"predicted_z_h_batch_sha256":zh_hashes,"sample_pixels":pixel_frames[0].cpu()}

def train_and_evaluate(output_root:Path,expected_manifest_sha256:str,expected_code_sha:str,device:str="cuda"):
    import torch
    from safetensors.torch import save_file
    if expected_manifest_sha256!=MANIFEST_SHA256 or os.environ.get("JUMP_CODE_VERSION")!=expected_code_sha:raise ValueError("residual JEPA identity mismatch")
    if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):raise FileExistsError("residual JEPA requires empty canonical workdir")
    if not torch.cuda.is_available():raise RuntimeError("residual JEPA requires CUDA")
    torch.manual_seed(PILOT_SEED);torch.cuda.manual_seed_all(PILOT_SEED);torch.cuda.reset_peak_memory_stats();started=time.monotonic();tr=dataset("train",1536);idr=dataset("id",96);oodr=dataset("heldout_law_ood",96);train=_tensorize(tr["records"],device);idd=_tensorize(idr["records"],device);oodd=_tensorize(oodr["records"],device);mean=train["observed"].mean((0,1,2),keepdim=True);std=train["observed"].std((0,1,2),keepdim=True).clamp_min(1e-6);modules,losses=_train(train,device,mean,std);ide=_evaluate(modules,idd,"id");oode=_evaluate(modules,oodd,"heldout_law_ood")
    gates=[]
    for r in (ide,oode):gates += [r["learned20_std_min"]>=.05,r["latent_vs_persistence"]["mean"]>=.2 and r["latent_vs_persistence"]["ci95"][0]>0,r["rollout_vs_copy_last"]["mean"]>=.2 and r["rollout_vs_copy_last"]["ci95"][0]>0,*[v["mean"]>=.1 and v["ci95"][0]>0 for v in r["action"].values()]]
    passed=all(gates);online,predictor,rollout,pixel,_=modules
    for role,module in (("encoder",online),("latent_predictor",predictor),("rollout_decoder",rollout),("pixel_decoder",pixel)):
        root=output_root/role;root.mkdir();save_file({k:v.detach().cpu() for k,v in module.state_dict().items()},root/"model.safetensors");(root/"config.json").write_bytes(canonical_json({"architecture":role,"latent_dim":LATENT_DIM,"strict_z_only":role in {"rollout_decoder","pixel_decoder"}}))
    (output_root/"manifest.json").write_bytes(canonical_json(manifest()));(output_root/"sample-z0.f32le.bin").write_bytes(ide.pop("z0_raw"));pixels=ide.pop("sample_pixels");images=[];root=output_root/"decoded-raster";root.mkdir()
    for i,h in enumerate(HORIZONS):raw=_png(pixels[i]);(root/f"h{h}.png").write_bytes(raw);images.append({"horizon":h,"png_sha256":hashlib.sha256(raw).hexdigest(),"predicted_z_h_batch_sha256":ide["predicted_z_h_batch_sha256"][i]})
    oode.pop("z0_raw");oode.pop("sample_pixels");evaluation={"id":ide,"heldout_law_ood":oode,"decoded_pngs":images};(output_root/"evaluation.json").write_bytes(canonical_json(evaluation));duration=time.monotonic()-started;terminal={"status":"completed","decision":"pass" if passed else "pivot","downstream_allowed":passed,"manifest_sha256":MANIFEST_SHA256,"code_sha":expected_code_sha,"initial_loss":losses[0],"final_loss":losses[-1],"evaluation":evaluation,"split_seed_hashes":{"train":tr["seed_set_sha256"],"id":idr["seed_set_sha256"],"heldout_law_ood":oodr["seed_set_sha256"]},"normalization_sha256":hashlib.sha256(torch.cat([mean.flatten(),std.flatten()]).cpu().numpy().astype("<f4").tobytes()).hexdigest(),"runtime_seconds":duration,"estimated_cost_usd":duration/3600*RATE_USD_PER_HOUR,"peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated()),"claims":manifest()["claims"],"claim_label":manifest()["claim_label"]};(output_root/"terminal.json").write_bytes(canonical_json(terminal));artifacts=[artifact_declaration(p,output_root,role="object-jepa-residual-evidence") for p in sorted(output_root.rglob("*")) if p.is_file()];evidence=write_task_evidence(output_root,metrics=[{"name":"id_latent_improvement","value":ide["latent_vs_persistence"]["mean"]},{"name":"ood_latent_improvement","value":oode["latent_vs_persistence"]["mean"]},{"name":"id_rollout_improvement","value":ide["rollout_vs_copy_last"]["mean"]},{"name":"ood_rollout_improvement","value":oode["rollout_vs_copy_last"]["mean"]}],artifacts=artifacts,track_h={"phase":"object-jepa-residual-pilot","decision":terminal["decision"],"claims":terminal["claims"]});return {**terminal,"task_evidence":evidence}

def run_contract(expected_manifest_sha256,expected_code_sha):return ({"id":"object-jepa-residual-pilot","_preregistration":{"layer_allowlist":[],"timepoint_allowlist":[]}},{"id":"object-jepa-residual-pilot","task":{"module":"jump_benchmark.object_jepa_residual_task","parameters":{"expected_manifest_sha256":expected_manifest_sha256,"expected_code_sha":expected_code_sha}},"resources":{"gpu":"H100","timeout_seconds":3600},"selection":{"layers":[],"timepoints":[]},"retry":{"max_attempts":1}})
