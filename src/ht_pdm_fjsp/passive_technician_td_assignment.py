"""Online Double-Q TD with a consistent train-only assignment auxiliary loss."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,statistics
from dataclasses import asdict
from datetime import UTC,datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from ht_pdm_fjsp import passive_technician_oracle_representation as base
from ht_pdm_fjsp import passive_technician_role_context as role
from ht_pdm_fjsp import passive_technician_policy_supervision as sup
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig,PassiveTechnicianEnv

PROTOCOL='ra_qmix_td_assignment_v1'
ARMS=('monotonic_td','monotonic_td_assignment','monotonic_policy_control')
DATASET_SEEDS=(98100,98101,98102)
REWARD_SCALE=20.
FEATURE_CONTRACT=dict(sup.FEATURE_CONTRACT,teacher='consistent TRAIN-state constraints only',
    split_interpretation='held out from supervision; online RL may visit and learn TD rewards there',reward_scale=REWARD_SCALE)

def settings(profile):
    cfg=base.settings(profile)
    cfg.update(train_seeds=list(range(98000,98010)) if profile=='full' else [98400],
        evaluation_seeds=list(range(98200,98250)) if profile=='full' else [98410,98411,98412],
        env_steps=60000 if profile=='full' else 120,learning_starts=512 if profile=='full' else 16,
        train_frequency=4,batch_size=128 if profile=='full' else 32,replay_capacity=50000 if profile=='full' else 120,
        target_interval=500 if profile=='full' else 10,log_interval=100 if profile=='full' else 5)
    cfg['updates']=(cfg['env_steps']//cfg['train_frequency']-(cfg['learning_starts']-1)//cfg['train_frequency'])
    return cfg

def seed_audit(cfg):
    registry=json.loads((Path(__file__).resolve().parents[2]/'configs/ra_qmix_td_assignment_seed_registry.json').read_text())
    panels=[set(cfg['train_seeds']),set(cfg['evaluation_seeds']),set(DATASET_SEEDS),set(range(201,301))]
    if any(a&b for i,a in enumerate(panels) for b in panels[i+1:]):raise ValueError('seed panels overlap')
    if set().union(*panels[:3])&set(registry['declared_seed_values']):raise ValueError('prior seed overlap')
    return dict(scope=registry['scope'],prior_manifest_count=registry['manifest_count'],prior_panels_disjoint=True,
        panels_disjoint=True,sealed_panel_closed=True,environment_population_previously_inspected=True)

def model_for(arm,config,cfg,seed):
    if arm not in ARMS:raise ValueError(arm)
    return role.model_for('monotonic_role_context',config,cfg,seed)

def train_teacher(oracle,data,solver_seed):
    keys=[k for k in oracle.keys if data['split'][k]=='train']
    train=SimpleNamespace(keys=keys,nodes={k:oracle.nodes[k] for k in keys},config=oracle.config,
        key_to_id=oracle.key_to_id)
    witness=role.policy_feasibility(train,'monotonic_role_context',solver_seed)
    labels=sup.teacher_labels(train,witness)
    starts=[data['slices'][k].start for k in keys]
    pool=dict(x=data['x'][starts],masks=data['masks'][starts],labels=torch.tensor([labels[k] for k in keys]),
        ids=[oracle.key_to_id[k] for k in keys])
    return witness,labels,pool

class Replay:
    """Bounded float32 tensor ring; contains real transitions, no oracle labels."""
    def __init__(self,capacity):self.capacity=capacity;self.storage={};self.position=0;self.size=0
    def __len__(self):return self.size
    def add(self,x,masks,actions,reward,nx,nm,done):
        row=dict(x=x.detach(),masks=torch.tensor(masks,dtype=torch.bool),actions=torch.tensor(actions),
            rewards=torch.tensor(reward/REWARD_SCALE,dtype=torch.float32),nx=nx.detach(),
            nm=torch.tensor(nm,dtype=torch.bool),done=torch.tensor(done,dtype=torch.bool))
        if not self.storage:
            self.storage={k:torch.empty((self.capacity,*v.shape),dtype=v.dtype) for k,v in row.items()}
        for k,v in row.items():self.storage[k][self.position]=v
        self.position=(self.position+1)%self.capacity;self.size=min(self.size+1,self.capacity)
    def sample(self,size,rng):
        indices=torch.from_numpy(rng.integers(0,self.size,size=size))
        return {k:v[indices] for k,v in self.storage.items()}

def td_targets(online,target,batch):
    with torch.no_grad():
        action=online.agent_q(batch['nx']).masked_fill(~batch['nm'],-torch.inf).argmax(-1)
        value=target.total_q(batch['nx'],action)
        return batch['rewards']+base.GAMMA*(~batch['done']).float()*value

def assignment_loss(model,pool,ids):
    masks,labels=pool['masks'][ids],pool['labels'][ids]
    if not masks.gather(-1,labels.unsqueeze(-1)).all():raise RuntimeError('masked teacher label')
    logits=model.agent_q(pool['x'][ids]).masked_fill(~masks,-torch.inf)
    return F.cross_entropy(logits.flatten(0,1),labels.flatten())

def epsilon(step,cfg):return max(.05,1.-.95*step/(.8*cfg['env_steps']))

def fit(model,oracle,cfg,seed,arm,cell,pool,output,progress,training_episodes):
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    target=copy.deepcopy(model).eval()
    for p in target.parameters():p.requires_grad_(False)
    teacher_rng=torch.Generator().manual_seed(seed+23)
    replay_rng=np.random.default_rng(seed+17);explore=np.random.default_rng(seed+19)
    replay=Replay(cfg['replay_capacity'])
    env=PassiveTechnicianEnv(oracle.config,seed=seed+13)
    control=arm=='monotonic_policy_control'
    mixer_initial={k:v.clone() for k,v in model.mixer.state_dict().items()}
    obs=env.reset() if not control else None
    visited=set();updates=0;target_syncs=0;steps=0;digest=hashlib.sha256()
    episode=0; flushed=len(training_episodes)
    total=cfg['updates'] if control else cfg['env_steps']
    model.train()
    for tick in tqdm(range(1,total+1),desc=f'train {cell}/{arm}/{seed}',unit='update' if control else 'step',leave=False):
        if not control:
            sid=oracle.key_to_id[(env.time,base.canonical(env.time,env.state))];visited.add(sid)
            x=base._obs_tensor(obs,oracle.config);mask=env.action_masks()
            with torch.no_grad():greedy=model.agent_q(x[None]).masked_fill(~torch.tensor(mask)[None],-torch.inf).argmax(-1)[0].tolist()
            # Draw exploration decisions/choices every step so RNG call counts match.
            random_flags=explore.random(2);uniform=explore.random(2)
            actions=tuple(int(np.flatnonzero(mask[i])[min(int(uniform[i]*sum(mask[i])),sum(mask[i])-1)])
                if random_flags[i]<epsilon(tick,cfg) else greedy[i] for i in range(2))
            downtime=sum(env.state.failed)*oracle.config.downtime_cost
            next_obs,reward,done,info=env.step(actions)
            expected_cost=downtime+info['jobs']*oracle.config.maintenance_cost+info['waiting']*oracle.config.queue_waiting_cost+info['failures']*oracle.config.failure_cost
            if abs(-reward-expected_cost)>1e-9:raise RuntimeError('training reward reconciliation failed')
            if info['invalid_requests']:raise RuntimeError('invalid training action')
            replay.add(x,mask,actions,reward,base._obs_tensor(next_obs,oracle.config),env.action_masks(),done)
            steps=tick;obs=next_obs
            if done:
                episode+=1
                training_episodes.append(dict(cell=cell,algorithm=arm,train_seed=seed,episode=episode,
                    env_steps=tick,**env.metrics))
                obs=env.reset()
            if tick<cfg['learning_starts'] or tick%cfg['train_frequency']:continue
        ids=torch.randint(len(pool['ids']),(cfg['batch_size'],),generator=teacher_rng)
        digest.update(ids.numpy().tobytes())
        zero=pool['x'].new_zeros(())
        td=zero
        if not control:
            batch=replay.sample(cfg['batch_size'],replay_rng)
            td=(model.total_q(batch['x'],batch['actions'])-td_targets(model,target,batch)).square().mean()
        ce=assignment_loss(model,pool,ids) if arm!='monotonic_td' else zero
        loss=td+ce
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step();updates+=1
        if not control and updates%cfg['target_interval']==0:
            target.load_state_dict(model.state_dict());target_syncs+=1
        if updates%cfg['log_interval']==0 or updates==cfg['updates']:
            progress.append(dict(cell=cell,algorithm=arm,train_seed=seed,update=updates,env_steps=steps,
                td_loss=float(td.detach()),assignment_ce=float(ce.detach()),loss=float(loss.detach()),gradient_norm=float(norm),
                epsilon=0. if control else epsilon(tick,cfg),replay_size=len(replay),target_syncs=target_syncs))
            base._write_csv(output/'training_progress.csv',progress)
            if len(training_episodes)>flushed:
                path=output/'training_episodes.csv'
                with path.open('a') as f:
                    writer=csv.DictWriter(f,fieldnames=list(training_episodes[flushed]))
                    if f.tell()==0:writer.writeheader()
                    writer.writerows(training_episodes[flushed:])
                training_episodes.clear();flushed=0
    if updates!=cfg['updates']:raise RuntimeError('optimizer budget mismatch')
    if control and not all(torch.equal(v,model.mixer.state_dict()[k]) for k,v in mixer_initial.items()):raise RuntimeError('control mixer changed')
    return model.eval(),dict(optimizer_updates=updates,env_steps=steps,target_syncs=target_syncs,
        teacher_batch_sha256=digest.hexdigest(),supervision_heldout_rows=0,visited_state_ids=sorted(visited),
        visited_teacher_heldout_states=sum(s not in pool['ids'] for s in visited),training_episodes=episode)

def load_checkpoint(path):
    p=torch.load(path,weights_only=True)
    if p['protocol_version']!=PROTOCOL or p['feature_contract']!=FEATURE_CONTRACT or p['reward_scale']!=REWARD_SCALE:raise ValueError('checkpoint contract mismatch')
    m=model_for(p['algorithm'],PassiveConfig(**p['config']),p['settings'],p['train_seed']);m.load_state_dict(p['state_dict']);return m.eval()

def evaluate(model,arm,oracle,data,labels,cell,seed):
    indices,rows,aggregate=base.evaluate_tables(model,arm,oracle,data,cell,seed)
    for r in rows:
        key=oracle.keys[r['state_id']];node=oracle.nodes[key];a=node['actions'][indices[key]]
        r.update(selected_actions=json.dumps(a),oracle_optimal=r['regret']<=1e-8,
            teacher_actions=json.dumps(labels[key]) if key in labels else None,
            teacher_agreement=a==labels[key] if key in labels else None,
            original_inputs_identical=node['observations'][0]==node['observations'][1])
    for split in ['train','heldout','all']:
        group=rows if split=='all' else [r for r in rows if r['split']==split]
        aggregate[split+'_regret']=statistics.fmean(r['regret'] for r in group)
        aggregate[split+'_oracle_optimal']=statistics.fmean(r['oracle_optimal'] for r in group)
    aggregate['train_teacher_agreement']=statistics.fmean(r['teacher_agreement'] for r in rows if r['split']=='train')
    return indices,rows,aggregate

def summarize(rows,cfg,full):
    idx={(r['cell'],r['algorithm'],r['train_seed']):r for r in rows};paired=[];controls={}
    for seed in cfg['train_seeds']:
        costs={a:statistics.fmean(idx[c,a,seed]['undiscounted_policy_cost'] for c in role.DIAGNOSTIC_CELLS) for a in ARMS}
        paired.append(dict(train_seed=seed,td_cost=costs['monotonic_td'],td_assignment_cost=costs['monotonic_td_assignment'],
            cost_delta=costs['monotonic_td_assignment']-costs['monotonic_td']))
    mean=statistics.fmean(r['cost_delta'] for r in paired);baseline=statistics.fmean(r['td_cost'] for r in paired)
    nominal=lambda a:statistics.fmean(idx['nominal',a,s]['undiscounted_policy_cost'] for s in cfg['train_seeds'])
    rel=-mean/baseline if baseline>0 else None;ratio=nominal('monotonic_td_assignment')/nominal('monotonic_td')-1
    wins=sum(r['cost_delta']<0 for r in paired)
    radius=2.262157163*statistics.stdev(r['cost_delta'] for r in paired)/math.sqrt(10) if full else None
    primary=dict(mean_cost_delta=mean,baseline_cost=baseline,relative_reduction=rel,improving_seeds=wins,
        ci95=[mean-radius,mean+radius] if full else None,nominal_relative_increase=ratio,
        passed=(mean<=-2 and rel is not None and rel>=.10 and wins>=8 and ratio<=.10) if full else None)
    for cell in ('nominal',*role.DIAGNOSTIC_CELLS):
        rr=[idx[cell,'monotonic_policy_control',s] for s in cfg['train_seeds']]
        regret=statistics.fmean(r['train_regret'] for r in rr);success=sum(r['train_oracle_optimal']>=.99 for r in rr)
        controls[cell]=dict(mean_train_regret=regret,seeds_at_99_percent_optimal=success,passed=regret<=.10 and success>=8 if full else None)
    control=all(v['passed'] for v in controls.values()) if full else None
    return paired,dict(primary=primary,positive_control=dict(cells=controls,passed=control),scientific_gate_applicable=full,
        empirical_attribution_supported=bool(primary['passed'] and control) if full else None,
        interpretation=FEATURE_CONTRACT['split_interpretation'])

def run(args):
    if args.device!='cpu':raise ValueError('CPU-only protocol')
    torch.set_num_threads(1);cfg=settings(args.profile);audit=seed_audit(cfg)
    output=Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    revision,dirty=base._git_state();expected=3*len(ARMS)*len(cfg['train_seeds'])
    manifest=dict(protocol_version=PROTOCOL,status='RUNNING',profile=args.profile,device='cpu',
        started_at=datetime.now(UTC).isoformat(),git_revision=revision,git_dirty=dirty,runtime=base._runtime_metadata(),
        train_seeds=cfg['train_seeds'],evaluation_seeds=cfg['evaluation_seeds'],dataset_seeds=DATASET_SEEDS,
        sealed_test_evaluated=False,sealed_test_seeds=list(range(201,301)),expected_models=expected,
        expected_optimizer_updates=expected*cfg['updates'],expected_training_env_steps=expected//3*2*cfg['env_steps'],
        expected_evaluation_rows=expected*len(cfg['evaluation_seeds']))
    from importlib.metadata import version
    manifest['runtime']['packages']['ortools']=version('ortools')
    write_json,write_csv=base._write_json,base._write_csv
    write_json(output/'manifest.json',manifest);write_json(output/'seed_audit.json',audit)
    write_json(output/'feature_contract.json',FEATURE_CONTRACT)
    write_json(output/'resolved_config.json',dict(settings=cfg,reward_scale=REWARD_SCALE,assignment_weight=1.,
        gamma=base.GAMMA,learning_rate=3e-4,cf_weight=0.,teacher_uses_heldout_constraints=False,
        auxiliary_sampling='independent uniform training states; not replay-gated',
        cells={c:asdict(v) for c,v in base.cells(args.profile).items()}))
    progress,training_episodes,episodes,metrics,state_rows,counts,coverage=[],[],[],[],[],[],[]
    actual=0
    try:
        for (cell,config),dataset_seed in zip(base.cells(args.profile).items(),DATASET_SEEDS):
            oracle=base.ExactOracle(config);data=base.dataset(oracle,dataset_seed)
            if oracle.bellman_residual()>1e-9:raise RuntimeError('Bellman residual')
            data.update(mean=0.,std=REWARD_SCALE,y=data['raw_y']/REWARD_SCALE)
            cell_dir=output/cell;cell_dir.mkdir()
            witness,labels,pool=train_teacher(oracle,data,dataset_seed)
            write_json(cell_dir/'policy_feasibility.json',witness)
            teacher_rows=[dict(state_id=oracle.key_to_id[k],actions=list(a)) for k,a in labels.items()]
            write_json(cell_dir/'train_teacher_actions.json',teacher_rows)
            teacher_hash=hashlib.sha256((cell_dir/'train_teacher_actions.json').read_bytes()).hexdigest()
            states,table=[],[]
            for key in oracle.keys:
                node,sid=oracle.nodes[key],oracle.key_to_id[key]
                states.append(dict(state_id=sid,time=key[0],observations=node['observations'],state=asdict(key[1]),split=data['split'][key]))
                for a,q95,q1 in zip(node['actions'],node['q95'],node['q1']):
                    table.append(dict(state_id=sid,actions=json.dumps(a),q95=q95,q1=q1))
            write_json(cell_dir/'oracle_states.json',states);write_csv(cell_dir/'oracle_q.csv',table)
            split_hash=hashlib.sha256(json.dumps([(s['state_id'],s['split']) for s in states]).encode()).hexdigest()
            for seed in tqdm(cfg['train_seeds'],desc=f'TD assignment {cell}',unit='seed'):
                initial=model_for(ARMS[0],config,cfg,seed).state_dict()
                for arm in ARMS:
                    model=model_for(arm,config,cfg,seed)
                    if not all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items()):raise RuntimeError('initialization differs')
                    counts.append(dict(cell=cell,algorithm=arm,train_seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                        mixer_trained=arm!='monotonic_policy_control'))
                    write_csv(output/'parameter_counts.csv',counts)
                    model,record=fit(model,oracle,cfg,seed,arm,cell,pool,output,progress,training_episodes)
                    record.update(cell=cell,algorithm=arm,train_seed=seed);coverage.append(record)
                    model_dir=cell_dir/arm/f'train_seed_{seed}';model_dir.mkdir(parents=True)
                    write_json(model_dir/'training_coverage.json',record)
                    write_csv(output/'teacher_batches.csv',[{k:v for k,v in r.items() if k!='visited_state_ids'} for r in coverage])
                    checkpoint=model_dir/'model.pt'
                    torch.save(dict(protocol_version=PROTOCOL,state_dict=model.state_dict(),algorithm=arm,config=asdict(config),
                        settings=cfg,train_seed=seed,reward_scale=REWARD_SCALE,feature_contract=FEATURE_CONTRACT,
                        teacher_sha256=teacher_hash,split_sha256=split_hash,updates=cfg['updates']),checkpoint)
                    restored=load_checkpoint(checkpoint)
                    if not torch.equal(base.predictions(model,data),base.predictions(restored,data)):raise RuntimeError('checkpoint Q mismatch')
                    with torch.no_grad():
                        if not torch.equal(model.agent_q(data['x']),restored.agent_q(data['x'])):raise RuntimeError('checkpoint agent mismatch')
                    indices,individual,aggregate=evaluate(restored,arm,oracle,data,labels,cell,seed)
                    metrics.append(aggregate);state_rows.extend(individual)
                    write_csv(output/'model_metrics.csv',metrics);write_csv(output/'state_metrics.csv',state_rows)
                    for evaluation in tqdm(cfg['evaluation_seeds'],desc=f'evaluate {cell}/{arm}/{seed}',unit='episode',leave=False):
                        episodes.append(dict(cell=cell,algorithm=arm,train_seed=seed,**base.rollout(oracle,indices,evaluation)))
                    write_csv(output/'episodes.partial.csv',episodes);actual+=1
        paired,summary=summarize(metrics,cfg,args.profile=='full')
        hashes={}
        for r in coverage:hashes.setdefault((r['cell'],r['train_seed']),set()).add(r['teacher_batch_sha256'])
        audits=dict(seed_audit=audit,complete_models=actual==expected,complete_evaluations=len(episodes)==manifest['expected_evaluation_rows'],
            exact_optimizer_updates=sum(r['optimizer_updates'] for r in coverage)==manifest['expected_optimizer_updates'],
            exact_environment_steps=sum(r['env_steps'] for r in coverage)==manifest['expected_training_env_steps'],
            per_model_budget=all(r['optimizer_updates']==cfg['updates'] and r['env_steps']==(0 if r['algorithm']=='monotonic_policy_control' else cfg['env_steps']) for r in coverage),
            target_sync_cadence=all(r['target_syncs']==(0 if r['algorithm']=='monotonic_policy_control' else cfg['updates']//cfg['target_interval']) for r in coverage),
            unique_episode_keys=len({(r['cell'],r['algorithm'],r['train_seed'],r['eval_seed']) for r in episodes})==len(episodes),
            zero_invalid_requests=all(r['invalid_requests']==0 for r in episodes),training_feasibility_and_reward_reconciliation=True,
            exact_cost_reconciliation=all(abs(r['cost_reconciliation_error'])<1e-9 for r in episodes),
            finite_training_logs=all(math.isfinite(r[k]) for r in progress for k in ['td_loss','assignment_ce','loss','gradient_norm']),
            matched_teacher_batches=len(hashes)==3*len(cfg['train_seeds']) and all(len(v)==1 for v in hashes.values()),
            no_heldout_supervision=all(r['supervision_heldout_rows']==0 for r in coverage) and all(r['teacher_actions'] is None for r in state_rows if r['split']=='heldout'),
            checkpoints_reload_identical=True,identical_initialization=True,policy_control_mixer_unchanged=True,
            verified_train_only_teacher=True,oracle_bellman_and_observation_audits=True,sealed_panel_closed=True)
        if not all(v for v in audits.values() if isinstance(v,bool)):raise RuntimeError(f'engineering audit failed: {audits}')
        summary['audits']=audits;write_csv(output/'paired_seed_metrics.csv',paired)
        write_csv(output/'episodes.csv',episodes);write_csv(output/'coordination.csv',episodes);write_json(output/'summary.json',summary)
        manifest.update(status='COMPLETED',completed_at=datetime.now(UTC).isoformat(),actual_models=actual,
            actual_evaluation_rows=len(episodes),actual_optimizer_updates=sum(r['optimizer_updates'] for r in coverage),
            actual_training_env_steps=sum(r['env_steps'] for r in coverage),actual_training_episodes=sum(r['training_episodes'] for r in coverage),audits=audits)
        print(json.dumps(summary,indent=2))
    except BaseException as error:
        manifest.update(status='FAILED',error=f'{type(error).__name__}: {error}',actual_models=actual,actual_evaluation_rows=len(episodes));raise
    finally:
        manifest['outputs']=sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
        write_json(output/'manifest.json',manifest)
    return output

def build_parser():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',required=True,choices=('smoke','full'))
    parser.add_argument('--device',default='cpu',choices=('cpu',));parser.add_argument('--output-dir',required=True);return parser

def main():run(build_parser().parse_args())
if __name__=='__main__':main()
