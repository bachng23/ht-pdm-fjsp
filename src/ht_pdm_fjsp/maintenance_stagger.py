"""Simple maintenance phase staggering and planned-downtime sensitivity."""
from __future__ import annotations
import argparse
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
from dataclasses import asdict, replace
from pathlib import Path
from tqdm import tqdm
from . import maintenance_coupling as base

PROTOCOL = 'maintenance_stagger_v1'
POLICIES = (base.POLICIES[1], 'E_backward_reservation', base.POLICIES[2])


def overlaps(start, duration, intervals):
    return any(start < end and start + duration > begin for begin, end in intervals)


def earliest_slot(lower, duration, intervals):
    start = lower
    for begin, end in sorted(intervals):
        if start < end and start + duration > begin:
            start = end
    return start


def reserve_action(cfg, state, lead, left, return_plan=False):
    assert lead >= 0 and left > 0
    calendars = []
    for t, queue in enumerate(state.queues):
        busy = state.remaining[t] + sum(cfg.service[m][t] for m in queue)
        calendars.append([(0, busy)] if busy else [])
    rank = {m:i for i,m in enumerate(cfg.request_order)}
    due = {m: 0 if state.failed[m] else max(0,cfg.failure_age-state.ages[m]-1)
           for m,opt in enumerate(base.options(cfg,state)) if len(opt)>1}
    order = sorted((m for m in due if state.failed[m] or due[m]<left),
                   key=lambda m:(not state.failed[m],due[m],rank[m]))
    action = [0]*len(state.ages); plan = []
    for m in order:
        candidates = []
        lower = max(0,due[m]-lead)
        for t, intervals in enumerate(calendars):
            duration = cfg.service[m][t]
            start = next((s for s in range(due[m],lower-1,-1) if not overlaps(s,duration,intervals)),None)
            if start is None: start = earliest_slot(lower,duration,intervals)
            late, early = max(0,start-due[m]),max(0,due[m]-start)
            risk = late*cfg.failed_cost if state.failed[m] else (1-(1-cfg.probability)**late)*(cfg.failure_cost+cfg.failed_cost)
            score = duration*cfg.planned_cost + cfg.maintenance_cost*early/cfg.failure_age + cfg.waiting_cost*late + risk
            candidates.append((score,-start,t,start,duration))
        score,_,t,start,duration = min(candidates)
        assert not overlaps(start,duration,calendars[t])
        calendars[t].append((start,start+duration))
        plan.append({'machine':m,'technician':t,'start':start,'duration':duration,'due':due[m],'score':score})
        if start == 0: action[m]=t+1
    assert all(a in opt for a,opt in zip(action,base.options(cfg,state)))
    return (tuple(action),plan) if return_plan else tuple(action)


class Solver(base.Solver):
    def __init__(self,cfg,offset=1,lead=2,depth=3,tail=2):
        super().__init__(cfg,offset,depth,tail)
        self.lead=lead
    def action(self,state,left,policy):
        if policy==POLICIES[1]: return reserve_action(self.cfg,state,self.lead,left)
        assert policy in POLICIES
        return super().action(state,left,policy)


def panels(profile):
    if profile=='smoke':
        return dict(dev_configs=[850100],dev_episodes=[851100,851101],configs=[850200],episodes=[851200,851201],prices=[1,8],horizons=[5],depth=2,tail=1)
    return dict(dev_configs=list(range(860100,860104)),dev_episodes=list(range(861100,861110)),configs=list(range(860200,860216)),episodes=list(range(861200,861230)),prices=[1,4,8],horizons=[12,24],depth=3,tail=2)


def cell_config(seed,pressure,skill,price,horizon):
    return replace(base.scenario(seed,pressure,skill,horizon),planned_cost=float(price))


def endpoint(values):
    if not values: return dict(mean=None,ci95=None,n_instances=0)
    if len(values)<2: return dict(mean=statistics.mean(values),ci95=None,n_instances=len(values))
    rng=random.Random(862100)
    draws=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000))
    return dict(mean=statistics.mean(values),ci95=[draws[250],draws[9750]],n_instances=len(values))


def summarize(rows,profile):
    grouped={}
    for r in rows:
        key=(r['instance_seed'],r['pressure'],r['skill'],r['planned_price'],r['horizon'],r['policy'])
        grouped.setdefault(key,[]).append(r)
    mean={k:{field:statistics.mean(r[field] for r in rs) for field in ('cost','operating_proxy','stage_cost','terminal_cost','cost_per_step')} for k,rs in grouped.items()}
    cells={}
    combos=sorted({(r['pressure'],r['skill'],r['planned_price'],r['horizon']) for r in rows})
    for p,skill,price,h in combos:
        seeds=sorted({r['instance_seed'] for r in rows if (r['pressure'],r['skill'],r['planned_price'],r['horizon'])==(p,skill,price,h)})
        def vals(pol,field='cost'): return [mean[i,p,skill,price,h,pol][field] for i in seeds]
        b,e,c=map(vals,POLICIES)
        relative=[(x-y)/y for x,y in zip(e,c) if y>0]
        cell={'pressure':p,'skill':skill,'planned_price':price,'horizon':h,'cost_means':{pol:statistics.mean(vals(pol)) for pol in POLICIES},
              'cost_per_step':{pol:statistics.mean(vals(pol,'cost_per_step')) for pol in POLICIES},
              'operating_proxy':{pol:statistics.mean(vals(pol,'operating_proxy')) for pol in POLICIES},
              'E_minus_C_relative':endpoint(relative) if relative else None,'C_positive_instances':len(relative),'B_positive_instances':sum(x>0 for x in b),
              'B_minus_E':endpoint([x-y for x,y in zip(b,e)]),'B_minus_C':endpoint([x-y for x,y in zip(b,c)]),
              'B_minus_E_relative':endpoint([(x-y)/x for x,y in zip(b,e) if x>0]),
              'E_minus_B_operating':endpoint([x-y for x,y in zip(vals(POLICIES[1],'operating_proxy'),vals(POLICIES[0],'operating_proxy'))]),
              'C_minus_B_operating':endpoint([x-y for x,y in zip(vals(POLICIES[2],'operating_proxy'),vals(POLICIES[0],'operating_proxy'))])}
        cells[f'{p}_{skill}_price{price}_h{h}']=cell
    primary=cells.get('high_heterogeneous_price1_h24')
    gates=dict(comparator_valid=False,E_cost_noninferior=False,E_useful=False,operating_guard=False,joint_operational=False)
    if profile=='full':
        assert primary is not None
        gates['comparator_valid']=primary['B_minus_C']['ci95'][0]>0
        gates['E_cost_noninferior']=primary['C_positive_instances']==16 and primary['E_minus_C_relative']['ci95'][1]<=.05
        gates['E_useful']=primary['B_positive_instances']==16 and primary['B_minus_E']['ci95'][0]>0 and primary['B_minus_E_relative']['mean']>=.05
        gates['operating_guard']=primary['E_minus_B_operating']['ci95'][0]>=-.02
        gates['joint_operational']=all(gates[k] for k in ('comparator_valid','E_cost_noninferior','E_useful','operating_guard'))
    return dict(cells=cells,gates=gates,scientific_status='ENGINEERING_ONLY' if profile=='smoke' else 'CONFIRMATORY_DIAGNOSTIC',inference_unit='independent instance means, conditional on fixed shared shock panel')


def measure(solver,policy,seed,trace=False):
    result,decisions=base.episode(solver,policy,seed,trace)
    h,n=solver.cfg.horizon,len(solver.cfg.service)
    result['operating_proxy']=1-(result['planned_steps']+result['failed_steps'])/(h*n)
    result['cost_per_step']=result['cost']/h
    assert 0<=result['operating_proxy']<=1
    return result,decisions


def run(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Output directory must be empty')
    panel=panels(args.profile)
    source=Path(__file__).read_bytes()
    manifest=dict(protocol=PROTOCOL,profile=args.profile,status='RUNNING',source_commit=base.git('rev-parse','HEAD'),source_dirty=bool(base.git('status','--porcelain')),
                  source_sha256=hashlib.sha256(source).hexdigest(),dependency_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(Path('docs/maintenance_stagger_v1_plan.md').read_bytes()).hexdigest(),panel=panel,full_panel_opened=args.profile=='full',
                  python=platform.python_version(),platform=platform.platform(),device=args.device,training='not applicable',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    base.write_json(out/'manifest.json',manifest)
    try:
        if args.profile=='full' and manifest['source_dirty']: raise RuntimeError('Full run requires clean Git source')
        assert not set(panel['configs'])&set(panel['dev_configs']) and not set(panel['episodes'])&set(panel['dev_episodes'])
        base.write_json(out/'benchmark_config.json',dict(protocol=PROTOCOL,model='unchanged completion_recovery_fifo_v1',policies=POLICIES))
        realized=[dict(instance_seed=seed,pressure=p,skill=sk,planned_price=price,horizon=h,config=asdict(cell_config(seed,p,sk,price,h))) for seed,(p,sk),price,h in itertools.product(panel['configs'],base.CELLS,panel['prices'],panel['horizons'])]
        base.write_json(out/'resolved_config.json',realized)
        dev=[]; settings={}
        for price in tqdm(panel['prices'],desc='development price families',unit='price'):
            for kind,grid in ((POLICIES[0],range(3)),(POLICIES[1],range(5))):
                tasks=list(itertools.product(grid,panel['dev_configs'],base.CELLS,panel['horizons']))
                for parameter,seed,(p,sk),h in tqdm(tasks,desc=f'tune {kind}/price{price}',unit='cell',leave=False):
                    solver=Solver(cell_config(seed,p,sk,price,h),offset=parameter if kind==POLICIES[0] else 1,lead=parameter if kind==POLICIES[1] else 2)
                    for e in panel['dev_episodes']:
                        result,_=measure(solver,kind,e)
                        dev.append(dict(policy=kind,parameter=parameter,instance_seed=seed,pressure=p,skill=sk,planned_price=price,horizon=h,episode_seed=e,**result))
                chosen=min(grid,key=lambda v:(statistics.mean(r['cost_per_step'] for r in dev if r['planned_price']==price and r['policy']==kind and r['parameter']==v),v))
                settings.setdefault(str(price),{})['offset' if kind==POLICIES[0] else 'lead']=chosen
        base.write_csv(out/'development.csv',dev)
        checkpoint=dict(protocol=PROTOCOL,settings=settings,depth=panel['depth'],tail=panel['tail'])
        base.write_json(out/'policy.json',checkpoint); loaded=json.loads((out/'policy.json').read_text());assert loaded==checkpoint
        rows=[];traces=[];coords=[]
        for record in tqdm(realized,desc='evaluate paired capacity/skill/price/horizon cells',unit='cell'):
            seed,p,sk,price,h=(record[k] for k in ('instance_seed','pressure','skill','planned_price','horizon'))
            cfg=cell_config(seed,p,sk,price,h)
            setting=loaded['settings'][str(price)]
            solver=Solver(cfg,**setting,depth=loaded['depth'],tail=loaded['tail'])
            for e in tqdm(panel['episodes'],desc=f'{seed}/{p}/{sk}/price{price}/h{h}',unit='seed',leave=False):
                for pol in POLICIES:
                    result,ts=measure(solver,pol,e,e==panel['episodes'][0])
                    identity=dict(instance_seed=seed,pressure=p,skill=sk,planned_price=price,horizon=h,episode_seed=e,policy=pol)
                    rows.append({**identity,**result});traces.extend({**identity,**t} for t in ts)
            coords.append(dict(instance_seed=seed,pressure=p,skill=sk,planned_price=price,horizon=h,invalid_requests=0,reservation_audit='PASS',state_audit='PASS'))
            base.write_csv(out/'episodes.partial.csv',rows);base.write_csv(out/'decisions.csv',traces)
            del solver;gc.collect()
        expected=len(realized)*len(panel['episodes'])*3
        keys={tuple(r[k] for k in ('instance_seed','pressure','skill','planned_price','horizon','episode_seed','policy')) for r in rows}
        assert len(keys)==len(rows)==expected
        assert all(r['invalid_requests']==0 and math.isfinite(r['cost']) and math.isclose(r['cost'],r['terminal_cost']+sum(r['cost_'+k] for k in base.COMPONENTS),abs_tol=1e-8) for r in rows)
        for seed in panel['configs']:
            assert len({tuple(statistics.mean(row) for row in cell_config(seed,p,sk,panel['prices'][0],panel['horizons'][0]).service) for p,sk in base.CELLS})==1
        base.write_csv(out/'episodes.csv',rows);base.write_csv(out/'coordination.csv',coords)
        summary=summarize(rows,args.profile)
        summary['audits']=dict(expected_rows=expected,actual_rows=len(rows),unique_rows=True,cost_reconciles=True,matched_service_means=True,sealed_panel_closed=args.profile=='smoke',all_passed=True)
        base.write_json(out/'summary.json',summary)
        manifest.update(status='COMPLETED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),outputs=sorted(f.name for f in out.iterdir()))
        base.write_json(out/'manifest.json',manifest)
        print(json.dumps(dict(status='COMPLETED',episodes=len(rows),settings=settings,gates=summary['gates'],scientific_status=summary['scientific_status'])))
    except BaseException:
        manifest.update(status='FAILED',error=traceback.format_exc());base.write_json(out/'manifest.json',manifest);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','full'),required=True)
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    parser.add_argument('--output-dir',required=True)
    run(parser.parse_args())


if __name__=='__main__': main()
