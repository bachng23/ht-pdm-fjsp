"""Forward EDF packing followed by whole-chain delay; no learning or rollouts."""
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
PROTOCOL='maintenance_deadline_packing_v1'
POLICIES=('H_health_risk_dispatch','S_capacity_reservation','D_deadline_packing')
scenario=repair.scenario
transition=repair.transition
effective_config=repair.effective_config


def panels(profile):
    if profile not in ('smoke','full'):raise ValueError(profile)
    smoke=profile=='smoke'
    regressions=[dict(instance_seed=i,episodes=list(range(first,first+(1 if smoke else 20)))) for i,first in [(900215,901200),(920216,921200),(920211,921200)]]
    if smoke:
        return dict(dev_configs=[930100],configs=[930200],dev_episodes=[931100,931101],episodes=[931200,931201],machines=4,lengths=[3,6],ratios=[1,2,4],regressions=regressions)
    return dict(dev_configs=list(range(940100,940106)),configs=list(range(940200,940224)),dev_episodes=list(range(941100,941106)),episodes=list(range(941200,941220)),machines=6,lengths=[16,64],ratios=[1,2,4],regressions=regressions)


def grid(policy):
    if policy==POLICIES[0]:return [dict(offset=o) for o in (0,1,2,4)]
    if policy in POLICIES[1:]:return [dict(lead=l) for l in (0,2,4,6)]
    raise ValueError(policy)


def plan(production,state,work,lead=2):
    """One hypothetical maintenance round, with immutable projected backlog."""
    assert lead in (0,2,4,6)
    cfg=effective_config(production,state)
    masks=repair.options(cfg,state,work)
    eligible=[m for m,opt in enumerate(masks) if len(opt)>1 and (state.failed[m] or work[m]>repair.due(cfg,state,m))]
    eligible.sort(key=lambda m:(not state.failed[m],repair.due(cfg,state,m),m))
    committed=[state.remaining[t]+sum(cfg.service[m][t] for m in q) for t,q in enumerate(state.queues)]
    available=committed[:];slots=[]
    for m in eligible:
        deadline=repair.due(cfg,state,m);release=max(0,deadline-lead)
        choices=[]
        for t,b in enumerate(available):
            start=max(b,release);duration=cfg.service[m][t]
            choices.append((max(0,start-deadline),start+duration,start,duration,t))
        late,_,start,duration,t=min(choices)
        slots.append(dict(machine=m,worker=t,start=start,duration=duration,deadline=deadline,release=release,forward_start=start,late=late))
        available[t]=start+duration
    for t in range(len(available)):
        chain=[s for s in slots if s['worker']==t]
        shift=max(0,min((s['deadline']-s['start'] for s in chain),default=0))
        previous=committed[t]
        for s in chain:
            s['start']+=shift;s['shift']=shift;s['late']=max(0,s['start']-s['deadline'])
            assert s['start']>=previous and s['start']>=s['release']
            previous=s['start']+s['duration']
    assert len({s['machine'] for s in slots})==len(slots)
    return slots


def action(production,state,work,policy,offset=1,lead=2):
    if policy in POLICIES[:2]:return repair.action(production,state,work,policy,offset,lead)
    assert policy==POLICIES[2]
    cfg=effective_config(production,state);masks=repair.options(cfg,state,work);a=[0]*len(work)
    for slot in plan(production,state,work,lead):
        t=slot['worker'];m=slot['machine']
        if slot['start']==0:
            assert state.remaining[t]==0 and not state.queues[t]
            assert t+1 not in a
            a[m]=t+1
    assert all(x in opt for x,opt in zip(a,masks))
    return tuple(a)

def episode(production,policy,seed,offset=1,lead=2,trace=False):
    cfg=production.maintenance;state=base.initial(cfg);required=tuple(map(sum,production.jobs));work=required
    ideal=max(required);late_boundary=ideal//2
    totals=dict.fromkeys(('failures','waiting','planned_steps','failed_steps','jobs','productive_steps','active_service_steps','active_waiting','busy_requests','invalid_requests','late_waiting','late_productive_steps','late_active_service_steps','late_failed_steps','cm_jobs','pm_jobs','cm_service_steps','pm_service_steps'),0.)
    totals.update({'cost_'+k:0. for k in COMPONENTS})
    progress=[0]*len(work);job_index=[0]*len(work);within=[0]*len(work);elapsed=0.;ts=[];completions=[]
    for step in range(cfg.horizon):
        started=time.perf_counter();a=action(production,state,work,policy,offset,lead);elapsed+=time.perf_counter()-started
        ns,left,produced,info=transition(production,state,work,a,tuple(u<cfg.probability for u in base.uniforms(seed,step,len(work))))
        for m in produced:
            progress[m]+=1;within[m]+=1
            if within[m]==production.jobs[m][job_index[m]]:
                completions.append(dict(machine=m,job=job_index[m],processing=within[m],finished=step+1));job_index[m]+=1;within[m]=0
        for k,v in info.items():totals[k]+=v
        if step>=late_boundary:
            for src,dst in [('active_waiting','late_waiting'),('productive_steps','late_productive_steps'),('active_service_steps','late_active_service_steps'),('failed_steps','late_failed_steps')]:totals[dst]+=info[src]
        if trace:ts.append(dict(step=step,deadline_plan=json.dumps(plan(production,state,work,lead)) if policy==POLICIES[2] else '[]',events=json.dumps(tuple(u<cfg.probability for u in base.uniforms(seed,step,len(work)))),state=json.dumps(asdict(state)),work=json.dumps(work),action=json.dumps(a),next_state=json.dumps(asdict(ns)),next_work=json.dumps(left),**info))
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
    rng=random.Random(942100);n=len(values)
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
        key=tuple(r[f] for f in ('instance_seed','pressure','skill','length','ratio','policy'));groups.setdefault(key,[]).append(r)
    seeds=sorted({r['instance_seed'] for r in rows});cells={}
    metrics=('makespan','normalized_makespan','cost','failures','waiting','failed_steps','planned_steps','cm_jobs','pm_jobs','cm_service_steps','pm_service_steps','mean_job_completion','decision_seconds')
    for (p,sk),l,ratio in itertools.product(CELLS,lengths,(1,2,4)):
        def vals(policy,f='makespan',stat=statistics.mean):
            return [stat([r[f] for r in groups[i,p,sk,l,ratio,policy]]) for i in seeds]
        h,s,d=[vals(pol) for pol in POLICIES];ht,st,dt=[vals(pol,stat=quantile95) for pol in POLICIES]
        cells[f'{p}_{sk}_jobs{l}_ratio{ratio}']=dict(metrics={pol:{f:statistics.mean(vals(pol,f)) for f in metrics} for pol in POLICIES},tail={pol:dict(p95=statistics.mean(vals(pol,stat=quantile95)),worst20_mean=statistics.mean(vals(pol,stat=worst20))) for pol in POLICIES},D_vs_S_relative=bootstrap([(a-b)/b for a,b in zip(d,s)]),D_vs_H_relative=bootstrap([(a-b)/b for a,b in zip(d,h)]),H_recovery_D=bootstrap([(a-b)/a for a,b in zip(h,d)]),D_minus_S_absolute=bootstrap([a-b for a,b in zip(d,s)]),D_vs_S_p95_relative=bootstrap([(a-b)/b for a,b in zip(dt,st)]),D_vs_H_p95_relative=bootstrap([(a-b)/b for a,b in zip(dt,ht)]))
    primary=cells[f'high_heterogeneous_jobs{lengths[-1]}_ratio2']['D_vs_S_relative']
    hetero=cells[f'high_heterogeneous_jobs{lengths[-1]}_ratio4']['D_vs_S_p95_relative']
    uniform=cells[f'high_uniform_jobs{lengths[-1]}_ratio4']['D_vs_H_p95_relative']
    gates=dict(mean_adequacy=profile=='full' and primary['ci95'][1]<=.02,heterogeneous_tail_guard=profile=='full' and hetero['ci95'][1]<=.05,uniform_tail_guard=profile=='full' and uniform['ci95'][1]<=.05)
    gates['overall_adequacy']=all(gates.values())
    return dict(cells=cells,primary=primary,heterogeneous_tail_guard=hetero,uniform_tail_guard=uniform,gates=gates,inference_unit='instance aggregates conditional on fixed common shock panel',scientific_status='ENGINEERING_ONLY' if profile=='smoke' else 'FRESH_PANEL_DIAGNOSTIC')


def regressions(out,panel,settings):
    rows=[];nt=nj=0
    for record in tqdm(panel['regressions'],desc='opened-case regressions',unit='instance'):
        i=record['instance_seed'];cfg=scenario(i,'high','heterogeneous',64,4,6);traces=[];jobs=[]
        for e in tqdm(record['episodes'],desc=f'opened case{i}',unit='shock',leave=False):
            for pol in POLICIES:
                params=dict(offset=4) if pol==POLICIES[0] else dict(lead=6) if pol==POLICIES[1] else settings['4'][pol]
                r,ts,cs=episode(cfg,pol,e,trace=e==record['episodes'][0],**params)
                identity=dict(instance_seed=i,episode_seed=e,policy=pol,exploratory=True)
                rows.append(dict(**identity,**r));traces.extend(dict(**identity,**t) for t in ts);jobs.extend(dict(**identity,**c) for c in cs)
        prod.append_csv(out/'regression_decisions.csv',traces);prod.append_csv(out/'regression_job_completions.csv',jobs);nt+=len(traces);nj+=len(jobs)
    expected={(rec['instance_seed'],e,pol) for rec in panel['regressions'] for e,pol in itertools.product(rec['episodes'],POLICIES)}
    assert len(rows)==len(expected) and {(r['instance_seed'],r['episode_seed'],r['policy']) for r in rows}==expected
    for r in rows:
        assert r['completed_jobs']==384 and r['productive_steps']==r['required_work'] and r['invalid_requests']==0
        assert r['pm_service_steps']+r['cm_service_steps']==r['planned_steps'] and r['pm_jobs']+r['cm_jobs']==r['jobs']
        assert math.isclose(r['cost'],sum(r['cost_'+k] for k in COMPONENTS),abs_tol=1e-8)
    base.write_csv(out/'regression_episodes.csv',rows)
    groups={}
    for r in rows:groups.setdefault((r['instance_seed'],r['policy']),[]).append(r)
    return dict(scope='opened selected cases, exploratory only; frozen H/S and new-dev-selected D',episodes=len(rows),decisions=nt,job_completions=nj,cases={str(i):{pol:dict(n=len(rs),makespan=statistics.mean(r['makespan'] for r in rs),failures=statistics.mean(r['failures'] for r in rs),max_makespan=max(r['makespan'] for r in rs)) for (j,pol),rs in groups.items() if j==i} for i in sorted({r['instance_seed'] for r in rows})})

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

def run(args):
    out=Path(args.output_dir)
    if not out.name.startswith('maintenance_deadline_'):raise ValueError('Use maintenance_deadline_* timestamp directory')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty; no overwrite/resume')
    panel=panels(args.profile)
    sources={'source':Path(__file__),'repair':Path(repair.__file__),'production':Path(prod.__file__),'coupling':Path(base.__file__),'protocol':Path('docs/maintenance_deadline_packing_v1_plan.md')}
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
        reg=regressions(out,panel,loaded['settings'])
        summary=summarize(rows,args.profile,panel['lengths']);summary['regressions']=reg
        summary['audits']=dict(all_passed=True,episodes=len(rows),development=len(dev),configs=len(configs),decisions=nt,job_completions=nj,full_panel_unopened=args.profile=='smoke')
        base.write_json(out/'summary.json',summary)
        manifest.update(status='COMPLETED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),outputs=sorted(f.name for f in out.iterdir()));base.write_json(out/'manifest.json',manifest)
        print(json.dumps(dict(status='COMPLETED',episodes=len(rows),development=len(dev),settings=settings,gates=summary['gates'],regressions=reg)))
    except BaseException:
        manifest.update(status='FAILED',error=traceback.format_exc());base.write_json(out/'manifest.json',manifest);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','full'),required=True);p.add_argument('--device',choices=('cpu',),default='cpu');p.add_argument('--output-dir',required=True);run(p.parse_args())

if __name__=='__main__':main()
