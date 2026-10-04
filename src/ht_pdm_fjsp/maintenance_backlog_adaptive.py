"""Fresh-panel backlog-adaptive threshold diagnostic and selected-case interventions."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, platform, random, statistics, time, traceback
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm
from . import maintenance_repair_impact as repair
prod=repair.prod
base=repair.base
COMPONENTS=repair.COMPONENTS
CELLS=repair.CELLS
PROTOCOL='maintenance_backlog_adaptive_v1'
POLICIES=('H_health_risk_dispatch','A_backlog_health','S_capacity_reservation')
scenario=repair.scenario
transition=repair.transition
effective_config=repair.effective_config


def panels(profile):
    if profile=='smoke':
        return dict(dev_configs=[910100],configs=[910200],dev_episodes=[911100,911101],episodes=[911200,911201],machines=4,lengths=[3,6],ratios=[1,2,4],case_episodes=[901200])
    if profile!='full':raise ValueError(profile)
    return dict(dev_configs=list(range(920100,920106)),configs=list(range(920200,920224)),dev_episodes=list(range(921100,921106)),episodes=list(range(921200,921220)),machines=6,lengths=[16,64],ratios=[1,2,4],case_episodes=list(range(901200,901220)))


def grid(policy):
    if policy==POLICIES[0]:return [dict(offset=o) for o in (0,1,2,4)]
    if policy==POLICIES[1]:return [dict(offset=o,gain=g) for o,g in itertools.product((0,1,2,4),(0,1))]
    if policy==POLICIES[2]:return [dict(lead=l) for l in (0,2,4,6)]
    raise ValueError(policy)


def action(production,state,work,policy,offset=1,lead=2,gain=0):
    if policy in (POLICIES[0],POLICIES[2]):
        return repair.action(production,state,work,policy,offset,lead)
    assert policy==POLICIES[1] and offset in (0,1,2,4) and gain in (0,1)
    if gain==0:return repair.action(production,state,work,POLICIES[0],offset,lead)
    cfg=effective_config(production,state)
    masks=repair.options(cfg,state,work)
    eligible=[m for m,opt in enumerate(masks) if len(opt)>1 and (state.failed[m] or work[m]>repair.due(cfg,state,m))]
    eligible.sort(key=lambda m:(not state.failed[m],repair.due(cfg,state,m),m))
    backlog=[state.remaining[t]+sum(cfg.service[m][t] for m in q) for t,q in enumerate(state.queues)]
    a=[0]*len(work)
    for m in eligible:
        deadline=repair.due(cfg,state,m)
        t=min(range(len(backlog)),key=lambda t:(cfg.service[m][t]+backlog[t]+repair.repair_risk(cfg,state,m,max(0,backlog[t]-deadline),production.ratio),backlog[t],cfg.service[m][t],t))
        threshold_offset=min(cfg.failure_age-1,offset+math.ceil(gain*backlog[t]))
        if state.failed[m] or state.ages[m]>=cfg.failure_age-threshold_offset:
            a[m]=t+1;backlog[t]+=cfg.service[m][t]
    assert all(x in opt for x,opt in zip(a,masks))
    return tuple(a)


def intervene(production,state,work,a,machine,worker):
    """One new request only; no cancellations of committed FIFO/service."""
    cfg=effective_config(production,state)
    if worker<0 or worker>=len(state.remaining) or machine<0 or machine>=len(work):raise ValueError('bad intervention identity')
    if state.remaining[worker] or state.queues[worker]:return a,False,'worker already occupied or committed'
    masks=repair.options(cfg,state,work)
    if worker+1 not in masks[machine]:return a,False,'target unavailable or completed'
    changed=tuple(worker+1 if m==machine else 0 if x==worker+1 else x for m,x in enumerate(a))
    assert all(x in opt for x,opt in zip(changed,masks))
    return changed,True,'replace new requests only'

def episode(production,policy,seed,offset=1,lead=2,gain=0,trace=False,intervention=None):
    cfg=production.maintenance;state=base.initial(cfg);required=tuple(map(sum,production.jobs));work=required
    ideal=max(required);late_boundary=ideal//2
    totals=dict.fromkeys(('failures','waiting','planned_steps','failed_steps','jobs','productive_steps','active_service_steps','active_waiting','busy_requests','invalid_requests','late_waiting','late_productive_steps','late_active_service_steps','late_failed_steps','cm_jobs','pm_jobs','cm_service_steps','pm_service_steps'),0.)
    totals.update({'cost_'+k:0. for k in COMPONENTS})
    progress=[0]*len(work);job_index=[0]*len(work);within=[0]*len(work);elapsed=0.;ts=[];completions=[]
    for step in range(cfg.horizon):
        started=time.perf_counter();a=action(production,state,work,policy,offset,lead,gain)
        applied=False;reason="not intervention tick"
        if intervention is not None and step==intervention[0]:
            a,applied,reason=intervene(production,state,work,a,*intervention[1:])
        elapsed+=time.perf_counter()-started
        ns,left,produced,info=transition(production,state,work,a,tuple(u<cfg.probability for u in base.uniforms(seed,step,len(work))))
        for m in produced:
            progress[m]+=1;within[m]+=1
            if within[m]==production.jobs[m][job_index[m]]:
                completions.append(dict(machine=m,job=job_index[m],processing=within[m],finished=step+1));job_index[m]+=1;within[m]=0
        for k,v in info.items():totals[k]+=v
        if step>=late_boundary:
            for src,dst in [('active_waiting','late_waiting'),('productive_steps','late_productive_steps'),('active_service_steps','late_active_service_steps'),('failed_steps','late_failed_steps')]:totals[dst]+=info[src]
        if trace:ts.append(dict(step=step,intervention_applied=applied,intervention_reason=reason,events=json.dumps(tuple(u<cfg.probability for u in base.uniforms(seed,step,len(work)))),state=json.dumps(asdict(state)),work=json.dumps(work),action=json.dumps(a),next_state=json.dumps(asdict(ns)),next_work=json.dumps(left),**info))
        state,work=ns,left
        if not any(work):break
    else:raise RuntimeError(f'Episode safety cap hit: {seed}/{policy}, remaining={work}')
    makespan=step+1
    assert tuple(progress)==required and tuple(job_index)==tuple(map(len,production.jobs))
    assert totals['productive_steps']==sum(required) and makespan>=ideal and max(c['finished'] for c in completions)==makespan
    final_cfg=effective_config(production,state)
    outstanding=sum(state.remaining)+sum(final_cfg.service[m][t] for t,q in enumerate(state.queues) for m in q)
    result=dict(**totals,makespan=makespan,ideal_makespan=ideal,normalized_makespan=makespan/ideal,completed_jobs=len(completions),required_work=sum(required),outstanding_service_ticks=outstanding,late_boundary=late_boundary,mean_job_completion=statistics.mean(c['finished'] for c in completions),cost=sum(totals['cost_'+k] for k in COMPONENTS),decision_seconds=elapsed,feasibility_audit='PASS',work_balance_audit='PASS')
    return result,ts,completions if trace else []

def bootstrap(values):
    if not values:raise ValueError('empty bootstrap')
    rng=random.Random(922100);n=len(values)
    draws=sorted(sum(rng.choices(values,k=n))/n for _ in range(10000))
    return dict(mean=statistics.mean(values),ci95=[draws[250],draws[9750]],n_instances=n)


def quantile95(values):
    return sorted(values)[math.ceil(.95*len(values))-1]


def worst20(values):
    n=math.ceil(.2*len(values));return statistics.mean(sorted(values)[-n:])


def select(dev,ratio,policy):
    candidates=grid(policy)
    return min(candidates,key=lambda params:(statistics.mean(r['normalized_makespan'] for r in dev if r['ratio']==ratio and r['policy']==policy and r['parameter']==json.dumps(params,sort_keys=True)),tuple(params.values())))


def summarize(rows,profile,lengths):
    groups={}
    for r in rows:
        key=tuple(r[f] for f in ('instance_seed','pressure','skill','length','ratio','policy'))
        groups.setdefault(key,[]).append(r)
    seeds=sorted({r['instance_seed'] for r in rows});cells={}
    metrics=('makespan','normalized_makespan','cost','failures','waiting','failed_steps','planned_steps','cm_jobs','pm_jobs','cm_service_steps','pm_service_steps','mean_job_completion','decision_seconds')
    for (p,sk),l,ratio in itertools.product(CELLS,lengths,(1,2,4)):
        def vals(policy,f='makespan',stat=statistics.mean):
            return [stat([r[f] for r in groups[i,p,sk,l,ratio,policy]]) for i in seeds]
        h,a,s=[vals(pol) for pol in POLICIES]
        at,st=vals(POLICIES[1],stat=quantile95),vals(POLICIES[2],stat=quantile95)
        cells[f'{p}_{sk}_jobs{l}_ratio{ratio}']=dict(metrics={pol:{f:statistics.mean(vals(pol,f)) for f in metrics} for pol in POLICIES},tail={pol:dict(p95=statistics.mean(vals(pol,stat=quantile95)),worst20_mean=statistics.mean(vals(pol,stat=worst20))) for pol in POLICIES},H_minus_A=bootstrap([x-y for x,y in zip(h,a)]),H_minus_A_relative=bootstrap([(x-y)/x for x,y in zip(h,a)]),A_vs_S_relative=bootstrap([(x-y)/y for x,y in zip(a,s)]),A_vs_S_p95_relative=bootstrap([(x-y)/y for x,y in zip(at,st)]))
    primary=cells[f'high_heterogeneous_jobs{lengths[-1]}_ratio2']['A_vs_S_relative']
    tail=cells[f'high_heterogeneous_jobs{lengths[-1]}_ratio4']['A_vs_S_p95_relative']
    gates=dict(mean_adequacy=profile=='full' and primary['ci95'][1]<=.02,tail_guard=profile=='full' and tail['ci95'][1]<=.05)
    gates['overall_adequacy']=all(gates.values())
    return dict(cells=cells,primary=primary,tail_guard=tail,gates=gates,inference_unit='instance aggregates conditional on fixed common shock panel',scientific_status='ENGINEERING_ONLY' if profile=='smoke' else 'FRESH_PANEL_DIAGNOSTIC')


def validate(rows,panel,development=False):
    keys=('instance_seed','pressure','skill','length','ratio','episode_seed','policy')+ (('parameter',) if development else ())
    expected=set()
    for i,(p,sk),l,r,e,pol in itertools.product(panel['dev_configs' if development else 'configs'],CELLS,panel['lengths'],panel['ratios'],panel['dev_episodes' if development else 'episodes'],POLICIES):
        identity=(i,p,sk,l,r,e,pol)
        if development:
            expected.update(identity+(json.dumps(params,sort_keys=True),) for params in grid(pol))
        else:expected.add(identity)
    assert len(rows)==len(expected) and {tuple(r[k] for k in keys) for r in rows}==expected
    for r in rows:
        assert r['feasibility_audit']==r['work_balance_audit']=='PASS' and r['invalid_requests']==0
        assert r['productive_steps']==r['required_work'] and r['completed_jobs']==panel['machines']*r['length']
        assert r['makespan']>=r['ideal_makespan'] and r['cm_jobs']+r['pm_jobs']==r['jobs']
        assert r['cm_service_steps']+r['pm_service_steps']==r['planned_steps']
        assert math.isclose(r['cost'],sum(r['cost_'+k] for k in COMPONENTS),abs_tol=1e-8)
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))


def counterfactual(out,seeds):
    cfg=scenario(900215,'high','heterogeneous',64,4,6)
    variants=[('S_baseline',None)]+[(f'S_force_m{m}_w0_t{t}',(t,m,0)) for t,m in itertools.product((8,9,10),(4,5))]
    rows=[];diagnostics=[];jobs=[]
    for seed in tqdm(seeds,desc='opened selected-case counterfactual',unit='shock'):
        for name,intervention in variants:
            r,ts,cs=episode(cfg,POLICIES[2],seed,lead=6,trace=True,intervention=intervention)
            identity=dict(instance_seed=900215,episode_seed=seed,variant=name,exploratory=True)
            rows.append(dict(**identity,applied=any(t['intervention_applied'] for t in ts),**r))
            diagnostics.extend(dict(**identity,**t) for t in ts);jobs.extend(dict(**identity,**c) for c in cs)
        baseline=[t for t in diagnostics if t['episode_seed']==seed and t['variant']=='S_baseline']
        for name,intervention in variants[1:]:
            trace=[t for t in diagnostics if t['episode_seed']==seed and t['variant']==name]
            for a,b in zip(baseline[:intervention[0]],trace[:intervention[0]]):
                assert all(a[k]==b[k] for k in ('state','work','action','next_state','next_work','events'))
        prod.append_csv(out/'counterfactual_decisions.csv',diagnostics);prod.append_csv(out/'counterfactual_job_completions.csv',jobs);diagnostics=[];jobs=[]
    base.write_csv(out/'counterfactual_episodes.csv',rows)
    return dict(scope='hindsight intervention on one opened instance; no generalization CI',variants={name:dict(n=len(seeds),applied=sum(r['applied'] for r in rows if r['variant']==name),mean_makespan=statistics.mean(r['makespan'] for r in rows if r['variant']==name),mean_failures=statistics.mean(r['failures'] for r in rows if r['variant']==name)) for name,_ in variants},episodes=len(rows))


def run(args):
    out=Path(args.output_dir)
    if not out.name.startswith('maintenance_backlog_'):raise ValueError('Use maintenance_backlog_* timestamp directory')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty; no overwrite/resume')
    panel=panels(args.profile)
    sources={'source':Path(__file__),'repair':Path(repair.__file__),'production':Path(prod.__file__),'coupling':Path(base.__file__),'protocol':Path('docs/maintenance_backlog_adaptive_v1_plan.md')}
    manifest=dict(protocol=PROTOCOL,profile=args.profile,status='RUNNING',source_commit=base.git('rev-parse','HEAD'),source_dirty=bool(base.git('status','--porcelain')),hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()},panel=panel,full_panel_opened=False,python=platform.python_version(),platform=platform.platform(),device=args.device,training='not applicable',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    base.write_json(out/'manifest.json',manifest)
    try:
        if args.profile=='full' and manifest['source_dirty']:raise RuntimeError('Full requires clean Git source')
        base.write_json(out/'benchmark_config.json',dict(protocol=PROTOCOL,model=repair.PROTOCOL,policies=POLICIES,assumptions='hypothetical ratios; uncalibrated fixed routes'))
        dev=[];settings={}
        for ratio in tqdm(panel['ratios'],desc='develop ratio families',unit='ratio'):
            settings[str(ratio)]={}
            for pol in POLICIES:
                tasks=list(itertools.product(grid(pol),panel['dev_configs'],CELLS,panel['lengths']))
                for params,i,(p,sk),l in tqdm(tasks,desc=f'dev {pol}/R{ratio}',unit='cell',leave=False):
                    cfg=scenario(i,p,sk,l,ratio,panel['machines'])
                    for e in panel['dev_episodes']:
                        result,_,_=episode(cfg,pol,e,**params)
                        dev.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,episode_seed=e,policy=pol,parameter=json.dumps(params,sort_keys=True),**result))
                settings[str(ratio)][pol]=select(dev,ratio,pol)
        validate(dev,panel,True);base.write_csv(out/'development.csv',dev)
        checkpoint=dict(protocol=PROTOCOL,settings=settings);base.write_json(out/'policy.json',checkpoint)
        loaded=json.loads((out/'policy.json').read_text());assert loaded==checkpoint
        if args.profile=='full':
            manifest['full_panel_opened']=True; base.write_json(out/'manifest.json',manifest)
        configs=[dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=r,config=asdict(scenario(i,p,sk,l,r,panel['machines']))) for i,(p,sk),l,r in itertools.product(panel['configs'],CELLS,panel['lengths'],panel['ratios'])]
        base.write_json(out/'resolved_config.json',configs);rows=[];coords=[];nt=0;nj=0
        for rec in tqdm(configs,desc='paired fresh evaluation cells',unit='cell'):
            i,p,sk,l,ratio=(rec[f] for f in ('instance_seed','pressure','skill','length','ratio'));cfg=scenario(i,p,sk,l,ratio,panel['machines']);traces=[];completions=[]
            for e in tqdm(panel['episodes'],desc=f'{i}/{p}/{sk}/jobs{l}/R{ratio}',unit='shock',leave=False):
                for pol in POLICIES:
                    result,ts,cs=episode(cfg,pol,e,**loaded['settings'][str(ratio)][pol],trace=e==panel['episodes'][0])
                    identity=dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,episode_seed=e,policy=pol)
                    rows.append({**identity,**result});traces.extend({**identity,**t} for t in ts);completions.extend({**identity,**c} for c in cs)
            coords.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,feasibility_audit='PASS',work_balance_audit='PASS',invalid_requests=0))
            base.write_csv(out/'episodes.partial.csv',rows);prod.append_csv(out/'decisions.csv',traces);prod.append_csv(out/'job_completions.csv',completions);nt+=len(traces);nj+=len(completions)
        validate(rows,panel)
        base.write_csv(out/'episodes.csv',rows);base.write_csv(out/'coordination.csv',coords)
        cf=counterfactual(out,panel['case_episodes'])
        summary=summarize(rows,args.profile,panel['lengths']);summary['counterfactual']=cf
        summary['audits']=dict(all_passed=True,episodes=len(rows),development=len(dev),configs=len(configs),decisions=nt,job_completions=nj,full_panel_unopened=args.profile=='smoke')
        base.write_json(out/'summary.json',summary)
        manifest.update(status='COMPLETED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),outputs=sorted(f.name for f in out.iterdir()));base.write_json(out/'manifest.json',manifest)
        print(json.dumps(dict(status='COMPLETED',episodes=len(rows),development=len(dev),settings=settings,gates=summary['gates'],counterfactual=cf)))
    except BaseException:
        manifest.update(status='FAILED',error=traceback.format_exc());base.write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','full'),required=True);p.add_argument('--device',choices=('cpu',),default='cpu');p.add_argument('--output-dir',required=True);run(p.parse_args())

if __name__=='__main__':main()
