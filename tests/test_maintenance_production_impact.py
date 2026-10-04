import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import pytest
from ht_pdm_fjsp import maintenance_production_impact as m

def config(jobs=((2,4),),age=0,limit=100,p=1,service=2):
    n=len(jobs)
    cfg=m.base.Config(((service,service),)*n,(age,)*n,limit,p,1000,request_order=tuple(range(n)))
    return m.ProductionConfig(cfg,jobs)

def test_direct_production_completion_clock():
    r,t,c=m.episode(config(),m.POLICIES[1],871100,trace=True)
    assert r['makespan']==6 and r['productive_steps']==6 and r['completed_jobs']==2
    assert [(x['job'],x['finished']) for x in c]==[(0,2),(1,6)]
    assert len(t)==6 and r['cost']==0

def test_failure_repair_resumes_unfinished_job():
    r,t,c=m.episode(config(jobs=((4,),),limit=2),m.POLICIES[1],871100,offset=0,trace=True)
    assert r['makespan']==6 and r['failures']==1 and r['productive_steps']==4
    assert r['planned_steps']==2 and c[0]['finished']==6
    assert json.loads(t[1]['next_work'])==[2] and json.loads(t[3]['next_work'])==[2]

def test_final_productive_tick_has_no_future_failure():
    r,_,_=m.episode(config(jobs=((1,),),age=1,limit=2),m.POLICIES[1],871100,offset=0)
    assert r['makespan']==1 and r['failures']==0

def test_finished_machine_neither_ages_nor_fails():
    cfg=config(jobs=((1,),(1,)),age=1,limit=2).maintenance
    ns,w,prod,info=m.transition(cfg,m.base.initial(cfg),(0,1),(0,0),(True,True))
    assert ns.ages[0]==1 and not any(ns.failed) and prod==[1] and w==(0,0)
    assert m.options(cfg,ns,w)==((0,),(0,))

def test_healthy_queued_machine_produces_and_can_fail():
    cfg=config(jobs=((5,),(5,)),age=1,limit=2).maintenance
    state=m.base.State((1,1),(False,False),(2,0),(0,-1),((1,),()))
    ns,w,prod,info=m.transition(cfg,state,(5,5),(0,0),(True,True))
    assert w==(5,4) and prod==[1] and ns.failed==(False,True)
    assert ns.queues==((1,),()) and ns.remaining==(1,0)
    assert info['waiting']==1 and info['planned_steps']==1

def test_failed_waiting_machine_does_not_produce():
    cfg=config(jobs=((5,),(5,)),age=1,limit=2).maintenance
    state=m.base.State((1,2),(False,True),(2,0),(0,-1),((1,),()))
    _,w,prod,info=m.transition(cfg,state,(5,5),(0,0),(False,False))
    assert w==(5,5) and not prod and info['failed_steps']==1

def test_duplicate_request_rejected():
    cfg=config(jobs=((5,),(5,))).maintenance
    state=m.base.State((0,0),(False,False),(2,0),(0,-1),((),()))
    with pytest.raises(AssertionError):m.transition(cfg,state,(5,5),(1,0),(False,False))

def test_same_urgency_admission_order_for_simultaneous_requests():
    cfg=config(jobs=((5,),(5,)),age=1,limit=3).maintenance
    state=m.base.State((1,3),(False,True),(0,0),(-1,-1),((),()))
    ns,w,prod,_=m.transition(cfg,state,(5,5),(1,1),(False,False))
    assert ns.assigned[0]==1 and ns.queues[0]==(0,)
    assert prod==[0] and w==(4,5)

def test_summary_uses_instance_means_not_episode_n():
    rows=[]
    for i in (870100,870200):
        for p,sk in m.CELLS:
            for l in (3,6):
                for pol in m.POLICIES:
                    r,_,_=m.episode(m.scenario(i,p,sk,l,4),pol,871100)
                    for e in (871100,871101):rows.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,episode_seed=e,policy=pol,**r))
    s=m.summarize(rows,'smoke',[3,6])
    assert s['interaction']['n_instances']==2
    assert all(c['H_minus_S_relative']['n_instances']==2 for c in s['cells'].values())

def test_risk_dispatch_resolves_old_completion_tie():
    cfg=m.base.Config(((2,4),(2,4)),(2,2),3,.4,100)
    state=m.base.initial(cfg)
    assert m.action(cfg,state,(20,20),m.POLICIES[0])==(1,1)
    assert m.action(cfg,state,(20,20),m.POLICIES[1])==(1,2)

def test_calendar_does_not_double_book_now():
    cfg=m.base.Config(((2,4),(2,4)),(2,2),3,.4,100)
    a=m.action(cfg,m.base.initial(cfg),(20,20),m.POLICIES[2],lead=0)
    assert a==(1,2)

def test_all_policies_skip_unnecessary_end_maintenance():
    cfg=config(jobs=((2,),),age=9,limit=12).maintenance
    for pol in m.POLICIES:
        assert m.action(cfg,m.base.initial(cfg),(2,),pol,offset=4,lead=6)==(0,)

def test_failed_machine_not_skipped_by_remaining_work_guard():
    cfg=config(jobs=((1,),),age=9,limit=12).maintenance
    state=m.base.State((9,),(True,),(0,0),(-1,-1),((),()))
    for pol in m.POLICIES:assert any(m.action(cfg,state,(1,),pol))

def test_calendar_respects_busy_worker_and_queue():
    cfg=config(jobs=((20,),(20,),(20,)),age=11,limit=12).maintenance
    state=m.base.State((11,11,11),(False,False,False),(2,0),(0,-1),((1,),()))
    assert m.action(cfg,state,(20,20,20),m.POLICIES[2],lead=0)==(0,0,2)

def test_cell_controls_and_prefix_streams():
    cells=[m.scenario(870100,p,sk,16) for p,sk in m.CELLS]
    assert len({tuple(sum(row)/len(row) for row in c.maintenance.service) for c in cells})==1
    assert len({(c.jobs,c.maintenance.initial_ages,c.maintenance.failure_age,c.maintenance.probability) for c in cells})==1
    long=m.scenario(870100,'high','heterogeneous',64)
    assert all(short==full[:16] for short,full in zip(cells[0].jobs,long.jobs))

def test_trace_conservation_and_productive_exclusion():
    cfg=m.scenario(870200,'high','heterogeneous',16,4)
    for pol in m.POLICIES:
        r,ts,cs=m.episode(cfg,pol,871200,trace=True)
        assert r['makespan']==max(c['finished'] for c in cs)
        for t in ts:
            st=json.loads(t['state']);work=json.loads(t['work']);nw=json.loads(t['next_work']);a=json.loads(t['action'])
            servicing={x for x in st['assigned'] if x>=0}
            qs=[list(q) for q in st['queues']]
            for machine,x in enumerate(a):
                if x:qs[x-1].append(machine)
            for worker,q in enumerate(qs):
                if st['remaining'][worker]==0 and q:servicing.add(q[0])
            for machine,(w,left) in enumerate(zip(work,nw)):
                if machine in servicing or st['failed'][machine]:assert w==left
                else:assert w-left==int(w>0)
        assert r['late_waiting']>=0

def test_long_workload_has_contention_beyond_initial_phase():
    c=m.scenario(870100,'high','uniform',64,4)
    r,_,_=m.episode(c,m.POLICIES[0],871100)
    assert r['late_waiting']>0 and r['makespan']>r['ideal_makespan']

def test_policy_checkpoint_action_roundtrip():
    settings={'offset':2,'lead':4};loaded=json.loads(json.dumps(settings))
    c=m.scenario(870200,'high','heterogeneous',3,4);state=m.base.initial(c.maintenance);work=tuple(map(sum,c.jobs))
    for pol in m.POLICIES:assert m.action(c.maintenance,state,work,pol,**settings)==m.action(c.maintenance,state,work,pol,**loaded)

def test_smoke_artifact_contract_and_no_overwrite(tmp_path):
    out=tmp_path/'maintenance_production_smoke_20261004T000000Z'
    args=argparse.Namespace(profile='smoke',device='cpu',output_dir=str(out))
    m.run(args)
    manifest=json.loads((out/'manifest.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert manifest['status']=='COMPLETED' and not manifest['full_panel_opened']
    assert summary['audits']['actual_rows']==48 and summary['audits']['development_rows']==128
    assert summary['scientific_status']=='ENGINEERING_ONLY' and not any(summary['gates'].values())
    assert (out/'episodes.csv').read_bytes()==(out/'episodes.partial.csv').read_bytes()
    completions=list(csv.DictReader((out/'job_completions.csv').open()))
    decisions=list(csv.DictReader((out/'decisions.csv').open()))
    assert len(completions)==4*(3+6)*4*3 and len(decisions)==summary['audits']['decision_rows']
    before=(out/'manifest.json').read_bytes()
    with pytest.raises(ValueError):m.run(args)
    assert (out/'manifest.json').read_bytes()==before

def test_no_full_generation_on_smoke(monkeypatch,tmp_path):
    old=m.scenario;used=[]
    def guarded(seed,*args,**kwargs):
        assert seed in (870100,870200);used.append(seed);return old(seed,*args,**kwargs)
    monkeypatch.setattr(m,'scenario',guarded)
    m.run(argparse.Namespace(profile='smoke',device='cpu',output_dir=str(tmp_path/'maintenance_production_guard_smoke')))
    assert set(used)=={870100,870200}
