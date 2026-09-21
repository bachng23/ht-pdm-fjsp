"""Evaluate a warm-started, frozen-base residual production-context policy."""

from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.entity_conditioned_experiment import _paired_summary
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.residual_context_policy import ResidualContextMaskablePolicy
from ht_pdm_fjsp.rl_experiment import ExperimentSettings, evaluate_ppo, resolve_device, train_model
from ht_pdm_fjsp.shared_policy_experiment import summarize_architectures

CONDITIONS=("shared_scorer_entropy","production_context_entropy","residual_context_entropy")

def defaults(profile:str)->dict[str,Any]:
    if profile=="smoke": return dict(total_timesteps=256,n_envs=1,n_steps=64,batch_size=32,n_epochs=2,train_seeds=(10000,11000),validation_seeds=tuple(range(46000,46005)))
    if profile=="full": return dict(total_timesteps=500000,n_envs=4,n_steps=1024,batch_size=256,n_epochs=10,train_seeds=(10000,11000,12000,13000,14000),validation_seeds=tuple(range(46000,46200)))
    raise ValueError(profile)

def _manifest(path:Path,name:str)->dict[str,Any]:
    p=path/name
    if not p.is_file(): raise FileNotFoundError(p)
    data=json.loads(p.read_text())
    if data.get("status")!="COMPLETED" or data.get("future_test_panel_opened"): raise ValueError(f"Invalid source run: {path}")
    return data

def _git()->str|None:
    r=subprocess.run(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None

def _tag(rows, evaluated, condition, seed):
    for row in evaluated: row.update(policy=condition,condition=condition,train_seed=seed)
    rows.extend(evaluated)

def run(args:argparse.Namespace)->Path:
    shared=Path(args.shared_source_run).resolve(); production=Path(args.production_source_run).resolve(); out=Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    sm=_manifest(shared,"architecture_manifest.json"); pm=_manifest(production,"production_context_manifest.json")
    if Path(pm["shared_source_run"]).name!=shared.name: raise ValueError("Source provenance mismatch.")
    d=defaults(args.profile)
    train=tuple(parse_seeds(args.train_seeds) if args.train_seeds else d["train_seeds"])
    available=set(map(int,sm["train_seeds"]))&set(map(int,pm["train_seeds"]))
    if not train or not set(train)<=available: raise ValueError("Missing training seed.")
    val=tuple(parse_seeds(args.validation_seeds) if args.validation_seeds else d["validation_seeds"])
    forbidden=set(range(40000,46000))|set(range(50000,50100))|set(map(int,sm["validation_seeds"]))|set(map(int,pm["validation_seeds"]))
    if not val or set(val)&forbidden: raise ValueError("Validation panel overlaps existing data.")
    config_path=Path(args.config).resolve(); raw=config_path.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    if pm.get("config_sha256")!=sha: raise ValueError("Config mismatch.")
    config=BenchmarkConfig.from_json(config_path); (out/"benchmark_config.json").write_bytes(raw)
    device=resolve_device(args.device)
    common=dict(profile=args.profile,total_timesteps=args.total_timesteps or d["total_timesteps"],n_envs=args.n_envs or d["n_envs"],n_steps=args.n_steps or d["n_steps"],batch_size=args.batch_size or d["batch_size"],n_epochs=args.n_epochs or d["n_epochs"],learning_rate=args.learning_rate,gamma=args.gamma,validation_seeds=val,test_seeds=(),device=device,include_slow_baselines=False)
    if common["n_steps"]*common["n_envs"]%common["batch_size"]: raise ValueError("Invalid PPO batch geometry.")
    manifest=dict(status="RUNNING",git_commit=_git(),config_sha256=sha,shared_source_run=str(shared),production_source_run=str(production),conditions=CONDITIONS,train_seeds=train,validation_seeds=val,reserved_future_test_seeds=list(range(50000,50100)),future_test_panel_opened=False,common_settings=common,entropy_coefficient=args.entropy_coefficient,max_residual_logit=args.max_residual_logit,freeze_base_actor=True,completed_training_cells=[],requested_device=args.device,resolved_device=device,runtime={k:version(k) for k in ("numpy","stable-baselines3","sb3-contrib","torch")})
    mp=out/"residual_context_manifest.json"; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    rows=[]
    sources=(("shared_scorer_entropy",shared,{}),("production_context_entropy",production,{"include_production_context":True}))
    bar=tqdm([(c,p,k,s) for c,p,k in sources for s in train],desc="Source validation",unit="model",disable=args.no_progress)
    for condition,path,kwargs,seed in bar:
        model=MaskablePPO.load(path/condition/f"train_seed_{seed}"/"maskable_ppo.zip",device=device)
        _tag(rows,evaluate_ppo(model,config,val,split="residual_validation",show_progress=False,env_kwargs=kwargs),condition,seed); del model
        _write_csv(rows,out/"residual_context_episodes.partial.csv")
    bar.close(); times={}
    bar=tqdm(train,desc="Residual-context PPO training",unit="model",disable=args.no_progress)
    for seed in bar:
        source=MaskablePPO.load(shared/"shared_scorer_entropy"/f"train_seed_{seed}"/"maskable_ppo.zip",device=device)
        def initialize(model, source=source): model.policy.load_shared_base(source.policy)
        model,elapsed=train_model(config,ExperimentSettings(train_seed=seed,**common),out/"residual_context_entropy"/f"train_seed_{seed}",show_progress=not args.no_progress,policy=ResidualContextMaskablePolicy,policy_kwargs=dict(context_hidden_dim=args.context_hidden_dim,max_residual_logit=args.max_residual_logit,freeze_base_actor=True),ent_coef=args.entropy_coefficient,env_kwargs={"include_production_context":True},model_initializer=initialize)
        del source; times[str(seed)]=elapsed
        _tag(rows,evaluate_ppo(model,config,val,split="residual_validation",show_progress=False,env_kwargs={"include_production_context":True}),"residual_context_entropy",seed); del model
        _write_csv(rows,out/"residual_context_episodes.partial.csv"); manifest["completed_training_cells"].append(seed); manifest["training_seconds"]=times; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    bar.close()
    expected=3*len(train)*len(val); unique={(r["condition"],r["train_seed"],r["seed"]) for r in rows}
    if len(rows)!=expected or len(unique)!=expected: raise ValueError("Incomplete panel.")
    summary=summarize_architectures(rows,conditions=CONDITIONS,baseline_condition="shared_scorer_entropy")
    summary["primary_endpoint"]="mean validation objective by training seed: residual minus shared entropy"
    summary["negative_control_comparison"]=_paired_summary(rows,"residual_context_entropy","production_context_entropy")
    _write_csv(rows,out/"residual_context_episodes.csv"); (out/"residual_context_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    manifest.update(status="COMPLETED",episode_count=len(rows),training_seconds=times,output_files=sorted(p.name for p in out.iterdir())); mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(f"Completed. Artifacts: {out}"); return out

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-source-run",required=True); p.add_argument("--production-source-run",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--config",default="configs/minimal_benchmark.json"); p.add_argument("--profile",choices=("smoke","full"),default="smoke"); p.add_argument("--train-seeds"); p.add_argument("--validation-seeds"); p.add_argument("--total-timesteps",type=int); p.add_argument("--n-envs",type=int); p.add_argument("--n-steps",type=int); p.add_argument("--batch-size",type=int); p.add_argument("--n-epochs",type=int); p.add_argument("--learning-rate",type=float,default=3e-4); p.add_argument("--gamma",type=float,default=.99); p.add_argument("--entropy-coefficient",type=float,default=.01); p.add_argument("--context-hidden-dim",type=int,default=64); p.add_argument("--max-residual-logit",type=float,default=1.0); p.add_argument("--device",default="auto"); p.add_argument("--no-progress",action="store_true"); p.add_argument("--overwrite",action="store_true"); return p
def main(): run(parser().parse_args())
if __name__=="__main__": main()
