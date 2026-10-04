"""Fixed-route production impact of scarce heterogeneous maintenance capacity."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm import tqdm
from . import maintenance_coupling as base

PROTOCOL = 'maintenance_production_impact_v1'
POLICIES = ('I_individual_fast', 'H_health_risk_dispatch', 'S_capacity_reservation')
CELLS = base.CELLS
COMPONENTS = base.COMPONENTS

@dataclass(frozen=True)
class ProductionConfig:
    maintenance: base.Config
    jobs: tuple[tuple[int, ...], ...]

def panels(profile):
    if profile == 'smoke':
        return dict(dev_configs=[870100],configs=[870200],dev_episodes=[871100,871101],episodes=[871200,871201],machines=4,lengths=[3,6])
    return dict(dev_configs=list(range(880100,880106)),configs=list(range(880200,880224)),dev_episodes=list(range(881100,881106)),episodes=list(range(881200,881220)),machines=6,lengths=[16,64])

def scenario(seed, pressure, skill, length, machines=6):
    assert pressure in ('low','high') and skill in ('uniform','heterogeneous')
    rng=random.Random(seed)
    age_limit=rng.choice((12,16,20)); probability=rng.choice((.25,.4,.6))
    preferences=[int(rng.random()>=.75) for _ in range(machines)]
    ages=tuple(rng.randrange(age_limit//2+1) for _ in range(machines))
    streams=tuple(tuple(rng.randint(2,6) for _ in range(64))[:length] for _ in range(machines))
    k=2 if pressure=='high' else machines
    assert k%2==0
    service=tuple(tuple(4 if skill=='uniform' else 2 if t%2==pref else 6 for t in range(k)) for pref in preferences)
    cap=(1+6+age_limit)*max(map(sum,streams))+100
    cfg=base.Config(service,ages,age_limit,probability,cap,request_order=tuple(range(machines)))
    return ProductionConfig(cfg,streams)

def options(cfg,state,work):
    occupied=set(state.assigned)|{m for q in state.queues for m in q}
    return tuple((0,) if m in occupied or work[m]==0 else tuple(range(len(state.remaining)+1)) for m in range(len(work)))

def due(cfg,state,m):
    return 0 if state.failed[m] else max(0,cfg.failure_age-state.ages[m]-1)

def risk(cfg,state,m,late):
    return late if state.failed[m] else (1-(1-cfg.probability)**late)*sum(cfg.service[m])/len(cfg.service[m])

def overlapping(start,duration,intervals):
    return any(start<end and start+duration>begin for begin,end in intervals)

def earliest(lower,duration,intervals):
    start=lower
    for begin,end in sorted(intervals):
        if start<end and start+duration>begin: start=end
    return start

def action(cfg,state,work,policy,offset=1,lead=2):
    assert policy in POLICIES and offset in (0,1,2,4) and lead in (0,2,4,6)
    eligible=[m for m,opt in enumerate(options(cfg,state,work)) if len(opt)>1 and (state.failed[m] or work[m]>due(cfg,state,m))]
    eligible.sort(key=lambda m:(not state.failed[m],due(cfg,state,m),m))
    a=[0]*len(work)
    backlog=[state.remaining[t]+sum(cfg.service[m][t] for m in q) for t,q in enumerate(state.queues)]
    calendars=[[(0,x)] if x else [] for x in backlog]
    for m in eligible:
        deadline=due(cfg,state,m)
        if policy!=POLICIES[2]:
            if not state.failed[m] and state.ages[m]<cfg.failure_age-offset: continue
            if policy==POLICIES[0]:
                t=min(range(len(backlog)),key=lambda t:(cfg.service[m][t],t))
            else:
                t=min(range(len(backlog)),key=lambda t:(cfg.service[m][t]+backlog[t]+risk(cfg,state,m,max(0,backlog[t]-deadline)),backlog[t],cfg.service[m][t],t))
            a[m]=t+1;backlog[t]+=cfg.service[m][t]
        else:
            lower=max(0,deadline-lead); choices=[]
            for t,intervals in enumerate(calendars):
                d=cfg.service[m][t]
                start=next((x for x in range(deadline,lower-1,-1) if not overlapping(x,d,intervals)),None)
                if start is None: start=earliest(lower,d,intervals)
                late=max(0,start-deadline); early=max(0,deadline-start)
                score=d+early/cfg.failure_age*sum(cfg.service[m])/len(cfg.service[m])+late+risk(cfg,state,m,late)
                choices.append((score,-start,t,start,d))
            _,_,t,start,d=min(choices)
            assert not overlapping(start,d,calendars[t]);calendars[t].append((start,start+d))
            if start==0:a[m]=t+1
    result=tuple(a)
    assert all(x in opt for x,opt in zip(result,options(cfg,state,work)))
    return result

def transition(cfg,state,work,a,events):
    """Production resumes after service, completed streams stop aging/hazard."""
    assert all(x in opt for x,opt in zip(a,options(cfg,state,work)))
    assert len(a)==len(events)==len(work)
    rem=list(state.remaining); assigned=list(state.assigned); queues=[list(q) for q in state.queues]
    ages=list(state.ages);failed=list(state.failed);left=list(work)
    counts=dict.fromkeys(('failures','waiting','planned_steps','failed_steps','jobs','productive_steps','active_service_steps','busy_requests','invalid_requests'),0)
    costs=dict.fromkeys(COMPONENTS,0.)
    # Same public urgency admission order as assignment planning; existing FIFO
    # queues are immutable. Physical machine ID breaks urgency ties.
    request_order=sorted((m for m in cfg.request_order if a[m]),key=lambda m:(not state.failed[m],due(cfg,state,m),m))
    for m in request_order:
        if a[m]:
            t=a[m]-1;counts['busy_requests']+=int(rem[t]>0);queues[t].append(m)
    for t,q in enumerate(queues):
        if rem[t]==0 and q:
            m=q.pop(0); assigned[t]=m;rem[t]=cfg.service[m][t]
            counts['jobs']+=1;costs['maintenance']+=cfg.maintenance_cost
    servicing={m for m in assigned if m>=0}
    counts['waiting']=sum(map(len,queues));costs['queue']=counts['waiting']*cfg.waiting_cost
    counts['active_waiting']=sum(left[m]>0 for q in queues for m in q)
    produced=[]
    for m in range(len(work)):
        if m in servicing:
            counts['planned_steps']+=1;counts['active_service_steps']+=int(left[m]>0);costs['planned_down']+=cfg.planned_cost
        elif left[m]==0: continue
        elif failed[m]: counts['failed_steps']+=1;costs['failed_down']+=cfg.failed_cost
        else:
            left[m]-=1;produced.append(m);counts['productive_steps']+=1;ages[m]=min(cfg.failure_age,ages[m]+1)
            if left[m]>0 and ages[m]>=cfg.failure_age and events[m]:
                failed[m]=True;counts['failures']+=1;costs['failure']+=cfg.failure_cost
    for t,m in enumerate(assigned):
        if m>=0:
            rem[t]-=1
            if rem[t]==0:ages[m]=0;failed[m]=False;assigned[t]=-1
    ns=base.State(tuple(ages),tuple(failed),tuple(rem),tuple(assigned),tuple(tuple(q) for q in queues))
    base.audit(cfg,ns);assert all(0<=l<=w for l,w in zip(left,work))
    return ns,tuple(left),produced,{**counts,**{'cost_'+k:v for k,v in costs.items()}}

def episode(production,policy,seed,offset=1,lead=2,trace=False):
    cfg=production.maintenance;state=base.initial(cfg);required=tuple(map(sum,production.jobs));work=required
    ideal=max(required);late_boundary=ideal//2
    totals=dict.fromkeys(('failures','waiting','planned_steps','failed_steps','jobs','productive_steps','active_service_steps','active_waiting','busy_requests','invalid_requests','late_waiting','late_productive_steps','late_active_service_steps','late_failed_steps'),0.)
    totals.update({'cost_'+k:0. for k in COMPONENTS})
    progress=[0]*len(work);job_index=[0]*len(work);within=[0]*len(work);elapsed=0.;ts=[];completions=[]
    for step in range(cfg.horizon):
        started=time.perf_counter();a=action(cfg,state,work,policy,offset,lead);elapsed+=time.perf_counter()-started
        ns,left,produced,info=transition(cfg,state,work,a,tuple(u<cfg.probability for u in base.uniforms(seed,step,len(work))))
        for m in produced:
            progress[m]+=1;within[m]+=1
            if within[m]==production.jobs[m][job_index[m]]:
                completions.append(dict(machine=m,job=job_index[m],processing=within[m],finished=step+1));job_index[m]+=1;within[m]=0
        for k,v in info.items():totals[k]+=v
        if step>=late_boundary:
            for src,dst in [('active_waiting','late_waiting'),('productive_steps','late_productive_steps'),('active_service_steps','late_active_service_steps'),('failed_steps','late_failed_steps')]:totals[dst]+=info[src]
        if trace:ts.append(dict(step=step,state=json.dumps(asdict(state)),work=json.dumps(work),action=json.dumps(a),next_state=json.dumps(asdict(ns)),next_work=json.dumps(left),**info))
        state,work=ns,left
        if not any(work):break
    else:raise RuntimeError(f'Episode safety cap hit: {seed}/{policy}, remaining={work}')
    makespan=step+1
    assert tuple(progress)==required and tuple(job_index)==tuple(map(len,production.jobs))
    assert totals['productive_steps']==sum(required) and makespan>=ideal and max(c['finished'] for c in completions)==makespan
    outstanding=sum(state.remaining)+sum(cfg.service[m][t] for t,q in enumerate(state.queues) for m in q)
    result=dict(**totals,makespan=makespan,ideal_makespan=ideal,normalized_makespan=makespan/ideal,completed_jobs=len(completions),required_work=sum(required),outstanding_service_ticks=outstanding,late_boundary=late_boundary,mean_job_completion=statistics.mean(c['finished'] for c in completions),cost=sum(totals['cost_'+k] for k in COMPONENTS),decision_seconds=elapsed,feasibility_audit='PASS',work_balance_audit='PASS')
    return result,ts,completions if trace else []

def bootstrap(values):
    rng=random.Random(882100); n=len(values)
    draws=sorted(sum(rng.choices(values,k=n))/n for _ in range(10000))
    return dict(mean=statistics.mean(values),ci95=[draws[250],draws[9750]],n_instances=n)

def summarize(rows,profile,lengths):
    groups={}
    for r in rows:
        key=(r['instance_seed'],r['pressure'],r['skill'],r['length'],r['policy']);groups.setdefault(key,[]).append(r)
    metric=('makespan','normalized_makespan','failures','waiting','late_waiting','planned_steps','failed_steps','active_service_steps','productive_steps','late_productive_steps','late_active_service_steps','late_failed_steps','cost','jobs','mean_job_completion','outstanding_service_ticks','decision_seconds')
    means={k:{f:statistics.mean(r[f] for r in rs) for f in metric} for k,rs in groups.items()}
    seeds=sorted({r['instance_seed'] for r in rows});cells={}
    def vals(p,sk,l,pol,f='makespan'):return [means[i,p,sk,l,pol][f] for i in seeds]
    for p,sk,l in itertools.product(('low','high'),('uniform','heterogeneous'),lengths):
        i,h,s=(vals(p,sk,l,pol) for pol in POLICIES)
        cells[f'{p}_{sk}_jobs{l}']=dict(metrics={pol:{f:statistics.mean(vals(p,sk,l,pol,f)) for f in metric} for pol in POLICIES},H_minus_S=bootstrap([x-y for x,y in zip(h,s)]),H_minus_S_relative=bootstrap([(x-y)/x for x,y in zip(h,s)]),I_minus_H=bootstrap([x-y for x,y in zip(i,h)]),H_minus_S_late_service=bootstrap([x-y for x,y in zip(vals(p,sk,l,POLICIES[1],'late_active_service_steps'),vals(p,sk,l,POLICIES[2],'late_active_service_steps'))]))
    long=lengths[-1];short=lengths[0];primary=cells[f'high_heterogeneous_jobs{long}'];shortcell=cells[f'high_heterogeneous_jobs{short}']
    hhu,hhe,lhu,lhe=[vals(p,sk,long,POLICIES[1],'normalized_makespan') for p,sk in [('high','uniform'),('high','heterogeneous'),('low','uniform'),('low','heterogeneous')]]
    interaction=bootstrap([(a-b)-(c-d) for a,b,c,d in zip(hhe,hhu,lhe,lhu)])
    capacity={sk:bootstrap([a-b for a,b in zip(vals('high',sk,long,POLICIES[1],'normalized_makespan'),vals('low',sk,long,POLICIES[1],'normalized_makespan'))]) for sk in ('uniform','heterogeneous')}
    gates=dict(recovery=False,resource_specific_burden=False,persistence=False,problem_support=False)
    if profile=='full':
        gates['recovery']=primary['H_minus_S_relative']['mean']>=.05 and primary['H_minus_S']['ci95'][0]>0
        gates['resource_specific_burden']=interaction['ci95'][0]>0
        gates['persistence']=primary['H_minus_S_relative']['mean']>=.5*max(shortcell['H_minus_S_relative']['mean'],0) and primary['metrics'][POLICIES[1]]['late_waiting']>0
        gates['problem_support']=all(gates[k] for k in ('recovery','resource_specific_burden','persistence'))
    return dict(cells=cells,interaction=interaction,capacity_burden=capacity,gates=gates,inference_unit='instance means conditional on fixed common shock panel',scientific_status='ENGINEERING_ONLY' if profile=='smoke' else 'CONFIRMATORY_DIAGNOSTIC')

def validate_rows(rows,panel):
    keyfields=('instance_seed','pressure','skill','length','episode_seed','policy')
    actual={tuple(r[k] for k in keyfields) for r in rows}
    expected={(i,p,sk,l,e,pol) for i,(p,sk),l,e,pol in itertools.product(panel['configs'],CELLS,panel['lengths'],panel['episodes'],POLICIES)}
    assert len(rows)==len(actual)==len(expected) and actual==expected
    for r in rows:
        assert r['feasibility_audit']==r['work_balance_audit']=='PASS' and r['invalid_requests']==0
        assert r['productive_steps']==r['required_work'] and r['completed_jobs']==panel['machines']*r['length']
        assert r['makespan']>=r['ideal_makespan'] and math.isclose(r['cost'],sum(r['cost_'+k] for k in COMPONENTS),abs_tol=1e-8)
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))

def validate_development(rows,panel):
    fields=('policy','parameter','instance_seed','pressure','skill','length','episode_seed')
    expected=set()
    for pol,grid in [(POLICIES[1],(0,1,2,4)),(POLICIES[2],(0,2,4,6))]:
        expected.update((pol,v,i,p,sk,l,e) for v,i,(p,sk),l,e in itertools.product(grid,panel['dev_configs'],CELLS,panel['lengths'],panel['dev_episodes']))
    assert len(rows)==len(expected) and {tuple(r[k] for k in fields) for r in rows}==expected
    assert all(r['productive_steps']==r['required_work'] and r['makespan']>=r['ideal_makespan'] and r['invalid_requests']==0 for r in rows)

def append_csv(path, rows):
    if not rows:return
    exists=path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists:writer.writeheader()
        writer.writerows(rows)

def run(args):
    out=Path(args.output_dir)
    if not out.name.startswith('maintenance_production_'):raise ValueError('Use a maintenance_production_* timestamped output directory')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty; no overwrite/resume')
    panel=panels(args.profile)
    manifest=dict(protocol=PROTOCOL,profile=args.profile,status='RUNNING',source_commit=base.git('rev-parse','HEAD'),source_dirty=bool(base.git('status','--porcelain')),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dependency_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(Path('docs/maintenance_production_impact_v1_plan.md').read_bytes()).hexdigest(),panel=panel,full_panel_opened=args.profile=='full',python=platform.python_version(),platform=platform.platform(),device=args.device,training='not applicable',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    base.write_json(out/'manifest.json',manifest)
    try:
        if args.profile=='full' and manifest['source_dirty']:raise RuntimeError('Full run requires clean source')
        assert not set(panel['configs'])&set(panel['dev_configs']) and not set(panel['episodes'])&set(panel['dev_episodes'])
        base.write_json(out/'benchmark_config.json',dict(protocol=PROTOCOL,model='fixed_route_preempt_resume_production_v1',policies=POLICIES,primary_cell=f'high_heterogeneous_jobs{panel["lengths"][-1]}'))
        dev=[];settings={}
        for pol,grid,param in [(POLICIES[1],(0,1,2,4),'offset'),(POLICIES[2],(0,2,4,6),'lead')]:
            tasks=list(itertools.product(grid,panel['dev_configs'],CELLS,panel['lengths']))
            for v,i,(p,sk),l in tqdm(tasks,desc=f'development {pol}',unit='cell'):
                cfg=scenario(i,p,sk,l,panel['machines'])
                for e in panel['dev_episodes']:
                    result,_,_=episode(cfg,pol,e,**{param:v})
                    dev.append(dict(parameter=v,instance_seed=i,pressure=p,skill=sk,length=l,episode_seed=e,policy=pol,**result))
            settings[param]=min(grid,key=lambda v:(statistics.mean(r['normalized_makespan'] for r in dev if r['policy']==pol and r['parameter']==v),v))
        validate_development(dev,panel)
        base.write_csv(out/'development.csv',dev);base.write_json(out/'policy.json',dict(protocol=PROTOCOL,**settings))
        loaded=json.loads((out/'policy.json').read_text());assert loaded==dict(protocol=PROTOCOL,**settings)
        settings={k:loaded[k] for k in ('offset','lead')}
        configs=[dict(instance_seed=i,pressure=p,skill=sk,length=l,config=asdict(scenario(i,p,sk,l,panel['machines']))) for i,(p,sk),l in itertools.product(panel['configs'],CELLS,panel['lengths'])]
        base.write_json(out/'resolved_config.json',configs)
        rows=[];coords=[];trace_count=0;completion_count=0
        for record in tqdm(configs,desc='paired production capacity/skill/workload cells',unit='cell'):
            i,p,sk,l=(record[k] for k in ('instance_seed','pressure','skill','length'));cfg=scenario(i,p,sk,l,panel['machines'])
            traces=[];completions=[]
            for e in tqdm(panel['episodes'],desc=f'{i}/{p}/{sk}/jobs{l}',unit='shock',leave=False):
                for pol in POLICIES:
                    result,ts,cs=episode(cfg,pol,e,**settings,trace=e==panel['episodes'][0])
                    identity=dict(instance_seed=i,pressure=p,skill=sk,length=l,episode_seed=e,policy=pol)
                    rows.append({**identity,**result});traces.extend({**identity,**t} for t in ts);completions.extend({**identity,**c} for c in cs)
            coords.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,feasibility_audit='PASS',work_balance_audit='PASS',invalid_requests=0))
            base.write_csv(out/'episodes.partial.csv',rows);append_csv(out/'decisions.csv',traces);append_csv(out/'job_completions.csv',completions)
            trace_count+=len(traces);completion_count+=len(completions)
        validate_rows(rows,panel)
        for i in panel['configs']:
            assert len({tuple(statistics.mean(row) for row in scenario(i,p,sk,panel['lengths'][0],panel['machines']).maintenance.service) for p,sk in CELLS})==1
        base.write_csv(out/'episodes.csv',rows);base.write_csv(out/'coordination.csv',coords)
        summary=summarize(rows,args.profile,panel['lengths']);summary['audits']=dict(all_passed=True,expected_rows=len(configs)*len(panel['episodes'])*3,actual_rows=len(rows),development_rows=len(dev),decision_rows=trace_count,completion_rows=completion_count,matched_service_means=True,full_panel_unopened=args.profile=='smoke')
        base.write_json(out/'summary.json',summary)
        manifest.update(status='COMPLETED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),outputs=sorted(f.name for f in out.iterdir()));base.write_json(out/'manifest.json',manifest)
        print(json.dumps(dict(status='COMPLETED',episodes=len(rows),settings=settings,gates=summary['gates'],scientific_status=summary['scientific_status'])))
    except BaseException:
        manifest.update(status='FAILED',error=traceback.format_exc());base.write_json(out/'manifest.json',manifest);raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','full'),required=True);parser.add_argument('--device',choices=('cpu',),default='cpu');parser.add_argument('--output-dir',required=True)
    run(parser.parse_args())

if __name__=='__main__':main()
