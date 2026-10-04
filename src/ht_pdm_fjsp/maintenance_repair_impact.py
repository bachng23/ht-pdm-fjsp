"""Corrective/preventive duration sensitivity with an adaptive health comparator."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, platform, random, statistics, time, traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tqdm import tqdm
from . import maintenance_production_impact as prod
base=prod.base
PROTOCOL='maintenance_repair_impact_v1'
POLICIES=('F_frozen_health','H_health_risk_dispatch','S_capacity_reservation')
CELLS=prod.CELLS
COMPONENTS=prod.COMPONENTS
options=prod.options
due=prod.due
overlapping=prod.overlapping
earliest=prod.earliest
@dataclass(frozen=True)
class RepairProduction:
    maintenance: base.Config
    jobs: tuple[tuple[int,...],...]
    ratio: int
    def __post_init__(self):
        assert self.ratio in (1,2,4)

def panels(profile):
    if profile=='smoke':return dict(dev_configs=[890100],configs=[890200],dev_episodes=[891100,891101],episodes=[891200,891201],machines=4,lengths=[3,6],ratios=[1,2,4])
    return dict(dev_configs=list(range(900100,900106)),configs=list(range(900200,900224)),dev_episodes=list(range(901100,901106)),episodes=list(range(901200,901220)),machines=6,lengths=[16,64],ratios=[1,2,4])

def scenario(seed,pressure,skill,length,ratio,machines=6):
    p=prod.scenario(seed,pressure,skill,length,machines)
    cap=(1+6*ratio+p.maintenance.failure_age)*max(map(sum,p.jobs))+100
    return RepairProduction(replace(p.maintenance,horizon=cap),p.jobs,ratio)

def effective_config(production,state):
    cfg=production.maintenance
    service=tuple(tuple(d*(production.ratio if state.failed[m] else 1) for d in row) for m,row in enumerate(cfg.service))
    return replace(cfg,service=service)

def repair_risk(cfg,state,m,late,ratio):
    return late if state.failed[m] else (1-(1-cfg.probability)**late)*sum(cfg.service[m])/len(cfg.service[m])*ratio

def action(production,state,work,policy,offset=1,lead=2):
    cfg=effective_config(production,state)
    if policy==POLICIES[0]:offset=0
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
            t=min(range(len(backlog)),key=lambda t:(cfg.service[m][t]+backlog[t]+repair_risk(cfg,state,m,max(0,backlog[t]-deadline),production.ratio),backlog[t],cfg.service[m][t],t))
            a[m]=t+1;backlog[t]+=cfg.service[m][t]
        else:
            lower=max(0,deadline-lead); choices=[]
            for t,intervals in enumerate(calendars):
                d=cfg.service[m][t]
                start=next((x for x in range(deadline,lower-1,-1) if not overlapping(x,d,intervals)),None)
                if start is None: start=earliest(lower,d,intervals)
                late=max(0,start-deadline); early=max(0,deadline-start)
                score=d+early/cfg.failure_age*sum(cfg.service[m])/len(cfg.service[m])+late+repair_risk(cfg,state,m,late,production.ratio)
                choices.append((score,-start,t,start,d))
            _,_,t,start,d=min(choices)
            assert not overlapping(start,d,calendars[t]);calendars[t].append((start,start+d))
            if start==0:a[m]=t+1
    result=tuple(a)
    assert all(x in opt for x,opt in zip(result,options(cfg,state,work)))
    return result

def transition(production,state,work,a,events):
    cfg=effective_config(production,state)
    # Identify starts and service type before completion resets failed status.
    queues=[list(q) for q in state.queues]
    order=sorted((m for m in cfg.request_order if a[m]),key=lambda m:(not state.failed[m],due(cfg,state,m),m))
    for m in order:queues[a[m]-1].append(m)
    starts=[q[0] for t,q in enumerate(queues) if state.remaining[t]==0 and q]
    servicing={m for m in state.assigned if m>=0}|set(starts)
    ns,left,produced,info=prod.transition(cfg,state,work,a,events)
    cm=sum(state.failed[m] for m in servicing);cm_jobs=sum(state.failed[m] for m in starts)
    info.update(cm_jobs=cm_jobs,pm_jobs=info['jobs']-cm_jobs,cm_service_steps=cm,pm_service_steps=len(servicing)-cm)
    assert info['cm_service_steps']+info['pm_service_steps']==info['planned_steps']
    return ns,left,produced,info

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
        if trace:ts.append(dict(step=step,state=json.dumps(asdict(state)),work=json.dumps(work),action=json.dumps(a),next_state=json.dumps(asdict(ns)),next_work=json.dumps(left),**info))
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
    rng=random.Random(902100); n=len(values)
    draws=sorted(sum(rng.choices(values,k=n))/n for _ in range(10000))
    return dict(mean=statistics.mean(values),ci95=[draws[250],draws[9750]],n_instances=n)

def summarize_ratio(rows,profile,lengths):
    groups={}
    for r in rows:
        key=(r['instance_seed'],r['pressure'],r['skill'],r['length'],r['policy']);groups.setdefault(key,[]).append(r)
    metric=('makespan','normalized_makespan','failures','waiting','late_waiting','planned_steps','failed_steps','active_service_steps','productive_steps','late_productive_steps','late_active_service_steps','late_failed_steps','cost','jobs','cm_jobs','pm_jobs','cm_service_steps','pm_service_steps','mean_job_completion','outstanding_service_ticks','decision_seconds')
    means={k:{f:statistics.mean(r[f] for r in rs) for f in metric} for k,rs in groups.items()}
    seeds=sorted({r['instance_seed'] for r in rows});cells={}
    def vals(p,sk,l,pol,f='makespan'):return [means[i,p,sk,l,pol][f] for i in seeds]
    for p,sk,l in itertools.product(('low','high'),('uniform','heterogeneous'),lengths):
        i,h,s=(vals(p,sk,l,pol) for pol in POLICIES)
        cells[f'{p}_{sk}_jobs{l}']=dict(metrics={pol:{f:statistics.mean(vals(p,sk,l,pol,f)) for f in metric} for pol in POLICIES},H_minus_S=bootstrap([x-y for x,y in zip(h,s)]),H_minus_S_relative=bootstrap([(x-y)/x for x,y in zip(h,s)]),F_minus_H=bootstrap([x-y for x,y in zip(i,h)]),H_minus_S_late_service=bootstrap([x-y for x,y in zip(vals(p,sk,l,POLICIES[1],'late_active_service_steps'),vals(p,sk,l,POLICIES[2],'late_active_service_steps'))]))
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

def summarize(rows,profile,lengths):
    summaries={}
    for ratio in (1,2,4):
        summaries[str(ratio)]=summarize_ratio([r for r in rows if r['ratio']==ratio],'smoke',lengths)
        for k in ('gates','scientific_status'):summaries[str(ratio)].pop(k)
    means={}
    for r in rows:
        if r['pressure']=='high' and r['skill']=='heterogeneous' and r['length']==lengths[-1] and r['policy']==POLICIES[0]:
            means.setdefault((r['instance_seed'],r['ratio']),[]).append(r['makespan'])
    seeds=sorted({i for i,ratio in means})
    f1=[statistics.mean(means[i,1]) for i in seeds];f2=[statistics.mean(means[i,2]) for i in seeds]
    mechanism=dict(absolute=bootstrap([b-a for a,b in zip(f1,f2)]),relative=bootstrap([(b-a)/a for a,b in zip(f1,f2)]))
    primary=summaries['2']['cells'][f'high_heterogeneous_jobs{lengths[-1]}']
    groups={}
    for r in rows:
        if r['ratio']==2 and r['pressure']=='high' and r['skill']=='heterogeneous' and r['length']==lengths[-1]:groups.setdefault((r['instance_seed'],r['policy']),[]).append(r['makespan'])
    h=[statistics.mean(groups[i,POLICIES[1]]) for i in seeds];s=[statistics.mean(groups[i,POLICIES[2]]) for i in seeds]
    adequate=bootstrap([(a-b)/b for a,b in zip(h,s)])
    gates=dict(physical_mechanism=False,coordination_useful=False,threshold_adequacy=False)
    if profile=='full':
        gates['physical_mechanism']=mechanism['relative']['mean']>=.05 and mechanism['absolute']['ci95'][0]>0
        gates['coordination_useful']=primary['H_minus_S_relative']['mean']>=.05 and primary['H_minus_S']['ci95'][0]>0
        gates['threshold_adequacy']=adequate['ci95'][1]<=.02
    return dict(ratios=summaries,physical_mechanism=mechanism,H_vs_S_noninferiority=adequate,gates=gates,inference_unit='24 instance means conditional on fixed common shock panel' if profile=='full' else 'smoke engineering only',scientific_status='CONFIRMATORY_DIAGNOSTIC' if profile=='full' else 'ENGINEERING_ONLY')

def validate(rows,panel,development=False):
    fs=('instance_seed','pressure','skill','length','ratio','episode_seed','policy')
    if development:
        fs=fs+('parameter',)
        expected={(i,p,sk,l,r,e,pol,v) for pol,grid in [(POLICIES[1],(0,1,2,4)),(POLICIES[2],(0,2,4,6))] for i,(p,sk),l,r,e,v in itertools.product(panel['dev_configs'],CELLS,panel['lengths'],panel['ratios'],panel['dev_episodes'],grid)}
    else:expected={(i,p,sk,l,r,e,pol) for i,(p,sk),l,r,e,pol in itertools.product(panel['configs'],CELLS,panel['lengths'],panel['ratios'],panel['episodes'],POLICIES)}
    assert len(rows)==len(expected) and {tuple(r[f] for f in fs) for r in rows}==expected
    for r in rows:
        assert r['feasibility_audit']==r['work_balance_audit']=='PASS' and r['invalid_requests']==0
        assert r['productive_steps']==r['required_work'] and r['completed_jobs']==panel['machines']*r['length'] and r['makespan']>=r['ideal_makespan']
        assert r['pm_jobs']+r['cm_jobs']==r['jobs'] and r['pm_service_steps']+r['cm_service_steps']==r['planned_steps']
        assert math.isclose(r['cost'],sum(r['cost_'+k] for k in COMPONENTS),abs_tol=1e-8)
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))

def run(args):
    out=Path(args.output_dir)
    if not out.name.startswith('maintenance_repair_'):raise ValueError('Use a maintenance_repair_* timestamp directory')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty; no overwrite/resume')
    panel=panels(args.profile)
    hashes={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in [('source_sha256',Path(__file__)),('production_dependency_sha256',Path(prod.__file__)),('coupling_dependency_sha256',Path(base.__file__)),('protocol_sha256',Path('docs/maintenance_repair_impact_v1_plan.md'))]}
    manifest=dict(protocol=PROTOCOL,profile=args.profile,status='RUNNING',source_commit=base.git('rev-parse','HEAD'),source_dirty=bool(base.git('status','--porcelain')),**hashes,panel=panel,full_panel_opened=args.profile=='full',python=platform.python_version(),platform=platform.platform(),device=args.device,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training='not applicable')
    base.write_json(out/'manifest.json',manifest)
    try:
        if args.profile=='full' and manifest['source_dirty']:raise RuntimeError('Full requires clean Git source')
        assert not set(panel['configs'])&set(panel['dev_configs']) and not set(panel['episodes'])&set(panel['dev_episodes'])
        base.write_json(out/'benchmark_config.json',dict(protocol=PROTOCOL,model='fixed_route_cm_duration_sensitivity_v1',ratios=panel['ratios'],assumptions='hypothetical duration ratios; no real plant calibration',policies=POLICIES))
        dev=[];settings={}
        for ratio in tqdm(panel['ratios'],desc='develop CM ratio families',unit='ratio'):
            for pol,grid,param in [(POLICIES[1],(0,1,2,4),'offset'),(POLICIES[2],(0,2,4,6),'lead')]:
                tasks=list(itertools.product(grid,panel['dev_configs'],CELLS,panel['lengths']))
                for v,i,(p,sk),l in tqdm(tasks,desc=f'{pol}/ratio{ratio}',unit='cell',leave=False):
                    cfg=scenario(i,p,sk,l,ratio,panel['machines'])
                    for e in panel['dev_episodes']:
                        result,_,_=episode(cfg,pol,e,**{param:v})
                        dev.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,episode_seed=e,policy=pol,parameter=v,**result))
                settings.setdefault(str(ratio),{})[param]=min(grid,key=lambda v:(statistics.mean(r['normalized_makespan'] for r in dev if r['ratio']==ratio and r['policy']==pol and r['parameter']==v),v))
        validate(dev,panel,True);base.write_csv(out/'development.csv',dev)
        checkpoint=dict(protocol=PROTOCOL,settings=settings,frozen_offset=0);base.write_json(out/'policy.json',checkpoint)
        loaded=json.loads((out/'policy.json').read_text());assert loaded==checkpoint
        configs=[dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=r,config=asdict(scenario(i,p,sk,l,r,panel['machines']))) for i,(p,sk),l,r in itertools.product(panel['configs'],CELLS,panel['lengths'],panel['ratios'])]
        base.write_json(out/'resolved_config.json',configs);rows=[];coords=[];nt=0;nj=0
        for rec in tqdm(configs,desc='paired capacity/skill/workload/CM ratio cells',unit='cell'):
            i,p,sk,l,ratio=(rec[f] for f in ('instance_seed','pressure','skill','length','ratio'));cfg=scenario(i,p,sk,l,ratio,panel['machines']);traces=[];completions=[]
            for e in tqdm(panel['episodes'],desc=f'{i}/{p}/{sk}/jobs{l}/CM{ratio}',unit='shock',leave=False):
                for pol in POLICIES:
                    result,ts,cs=episode(cfg,pol,e,**loaded['settings'][str(ratio)],trace=e==panel['episodes'][0])
                    identity=dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,episode_seed=e,policy=pol)
                    rows.append({**identity,**result});traces.extend({**identity,**t} for t in ts);completions.extend({**identity,**c} for c in cs)
            coords.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,feasibility_audit='PASS',work_balance_audit='PASS',invalid_requests=0))
            base.write_csv(out/'episodes.partial.csv',rows);prod.append_csv(out/'decisions.csv',traces);prod.append_csv(out/'job_completions.csv',completions);nt+=len(traces);nj+=len(completions)
        validate(rows,panel)
        for i,l,r in itertools.product(panel['configs'],panel['lengths'],panel['ratios']):
            assert len({tuple(statistics.mean(row) for row in scenario(i,p,sk,l,r,panel['machines']).maintenance.service) for p,sk in CELLS})==1
        base.write_csv(out/'episodes.csv',rows);base.write_csv(out/'coordination.csv',coords)
        summary=summarize(rows,args.profile,panel['lengths']);summary['audits']=dict(all_passed=True,episodes=len(rows),development=len(dev),configs=len(configs),decisions=nt,job_completions=nj,full_panel_unopened=args.profile=='smoke')
        base.write_json(out/'summary.json',summary)
        manifest.update(status='COMPLETED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),outputs=sorted(f.name for f in out.iterdir()));base.write_json(out/'manifest.json',manifest)
        print(json.dumps(dict(status='COMPLETED',episodes=len(rows),settings=settings,gates=summary['gates'])))
    except BaseException:
        manifest.update(status='FAILED',error=traceback.format_exc());base.write_json(out/'manifest.json',manifest);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','full'),required=True);p.add_argument('--device',choices=('cpu',),default='cpu');p.add_argument('--output-dir',required=True);run(p.parse_args())

if __name__=='__main__':main()
