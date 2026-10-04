import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import pytest
from ht_pdm_fjsp import maintenance_repair_impact as m

def small(ratio=1,jobs=((4,),),age=0,limit=2):
    n=len(jobs);cfg=m.base.Config(((2,2),)*n,(age,)*n,limit,1,1000,request_order=tuple(range(n)))
    return m.RepairProduction(cfg,jobs,ratio)

@pytest.mark.parametrize('ratio,expected',[(1,6),(2,8),(4,12)])
def test_corrective_duration_changes_production_finish(ratio,expected):
    r,_,_=m.episode(small(ratio),m.POLICIES[0],891100)
    assert r['makespan']==expected and r['failures']==1 and r['cm_jobs']==1
    assert r['cm_service_steps']==2*ratio and r['productive_steps']==4 and r['pm_jobs']==0

def test_preventive_duration_not_scaled():
    c=small(4,limit=3)
    ns,w,produced,info=m.transition(c,m.base.initial(c.maintenance),(4,),(1,),(True,))
    assert ns.remaining[0]==1 and not produced and info['pm_jobs']==1 and info['cm_jobs']==0

def test_failure_while_queued_reclassifies_at_dispatch():
    c=small(4,jobs=((20,),(20,)),age=1)
    state=m.base.State((1,1),(False,False),(1,0),(0,-1),((1,),()))
    ns,w,_,_=m.transition(c,state,(20,20),(0,0),(True,True))
    assert ns.failed[1] and ns.remaining==(0,0)
    final,w2,produced,info=m.transition(c,ns,w,(0,0),(True,True))
    assert final.assigned[0]==1 and final.remaining[0]==7 and info['cm_jobs']==1
    assert w2[1]==w[1]

def test_ongoing_repair_remaining_not_rescaled():
    c=small(4);state=m.base.State((2,),(True,),(2,0),(0,-1),((),()))
    ns,w,prod,info=m.transition(c,state,(4,),(0,),(True,))
    assert ns.remaining[0]==1 and not prod and w==(4,) and info['cm_jobs']==0

def test_ratio_one_replays_old_production_module():
    for p,sk in m.CELLS:
        c=m.scenario(890100,p,sk,16,1,4)
        old=m.prod.ProductionConfig(c.maintenance,c.jobs)
        for pol,original in [(m.POLICIES[0],m.prod.POLICIES[1]),(m.POLICIES[1],m.prod.POLICIES[1]),(m.POLICIES[2],m.prod.POLICIES[2])]:
            offset=0 if pol==m.POLICIES[0] else 2
            r,t,j=m.episode(c,pol,891100,offset=offset,lead=4,trace=True)
            rr,tt,jj=m.prod.episode(old,original,891100,offset=offset,lead=4,trace=True)
            assert all(r[f]==v for f,v in rr.items() if f!='decision_seconds') and j==jj
            for a,b in zip(t,tt):assert all(a[f]==v for f,v in b.items())
            assert len(t)==len(tt)

def test_fixed_threshold_ignores_tuned_offset():
    c=m.scenario(890100,'high','heterogeneous',6,2,4);st=m.base.initial(c.maintenance);work=tuple(map(sum,c.jobs))
    assert m.action(c,st,work,m.POLICIES[0],offset=0)==m.action(c,st,work,m.POLICIES[0],offset=4)

def test_failure_guard_and_completed_streams():
    c=small(4,jobs=((1,),),age=1)
    r,_,_=m.episode(c,m.POLICIES[0],891100)
    assert r['makespan']==1 and r['failures']==0 and r['jobs']==0

def test_ratio_pair_only_changes_physics_and_cap():
    cs=[m.scenario(890100,'high','heterogeneous',16,r,4) for r in (1,2,4)]
    assert len({(c.jobs,c.maintenance.service,c.maintenance.initial_ages,c.maintenance.failure_age,c.maintenance.probability) for c in cs})==1
    assert cs[0].maintenance.horizon<cs[1].maintenance.horizon<cs[2].maintenance.horizon

def test_current_failed_durations_and_mean_controls():
    c=m.scenario(890100,'high','heterogeneous',6,4,4);state=m.base.initial(c.maintenance)
    state=replace(state,failed=(True,False,False,False))
    cfg=m.effective_config(c,state)
    assert cfg.service[0]==tuple(4*d for d in c.maintenance.service[0])
    assert cfg.service[1:]==c.maintenance.service[1:]

def test_invalid_ratio_and_occupied_request_rejected():
    with pytest.raises(AssertionError):small(3)
    c=small(2);state=m.base.State((2,),(True,),(2,0),(0,-1),((),()))
    with pytest.raises(AssertionError):m.transition(c,state,(4,),(1,),(False,))

def test_seed_split_is_sealed_and_nonoverlapping():
    p=m.panels('full');sm=m.panels('smoke')
    assert not set(p['configs'])&set(p['dev_configs']) and not set(p['episodes'])&set(p['dev_episodes'])
    assert not (set(sm['configs'])|set(sm['dev_configs']))&(set(p['configs'])|set(p['dev_configs']))

def test_checkpoint_action_roundtrip():
    c=m.scenario(890100,'high','heterogeneous',6,2,4);st=m.base.initial(c.maintenance);work=tuple(map(sum,c.jobs))
    settings={'offset':2,'lead':4};loaded=json.loads(json.dumps(settings))
    for pol in m.POLICIES:assert m.action(c,st,work,pol,**settings)==m.action(c,st,work,pol,**loaded)

@pytest.mark.parametrize('health_time,useful,adequate',[(108,True,False),(101,False,True)])
def test_gates_separate_physics_coordination_and_threshold(health_time,useful,adequate):
    template,_,_=m.episode(small(),m.POLICIES[0],891100)
    rows=[]
    for i,(p,sk),l,ratio,pol,e in __import__('itertools').product((890100,890200),m.CELLS,(3,6),(1,2,4),m.POLICIES,(891100,891101)):
        duration=100 if ratio==1 else 120 if pol==m.POLICIES[0] else health_time if pol==m.POLICIES[1] else 100
        r={**template,'makespan':duration,'normalized_makespan':duration/template['ideal_makespan']}
        rows.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,policy=pol,episode_seed=e,**r))
    summary=m.summarize(rows,'full',[3,6])
    assert summary['gates']==dict(physical_mechanism=True,coordination_useful=useful,threshold_adequacy=adequate)
    assert summary['physical_mechanism']['relative']['n_instances']==2
    assert summary['H_vs_S_noninferiority']['n_instances']==2

def test_safety_cap_raises_instead_of_accepting_partial_production():
    c=small(4);c=replace(c,maintenance=replace(c.maintenance,horizon=1))
    with pytest.raises(RuntimeError,match='safety cap'):m.episode(c,m.POLICIES[0],891100)

def test_full_smoke_contract_and_sealed_generation_guard(tmp_path,monkeypatch):
    original=m.scenario;used=[]
    def guarded(seed,*args,**kwargs):
        assert seed in (890100,890200);used.append(seed);return original(seed,*args,**kwargs)
    monkeypatch.setattr(m,'scenario',guarded)
    out=tmp_path/'maintenance_repair_smoke_20261005T000000Z';args=argparse.Namespace(profile='smoke',device='cpu',output_dir=str(out));m.run(args)
    mf=json.loads((out/'manifest.json').read_text());s=json.loads((out/'summary.json').read_text())
    assert mf['status']=='COMPLETED' and not mf['full_panel_opened'] and set(used)=={890100,890200}
    assert s['audits']['episodes']==144 and s['audits']['development']==384 and s['audits']['configs']==24
    assert s['audits']['job_completions']==1296 and s['scientific_status']=='ENGINEERING_ONLY' and not any(s['gates'].values())
    assert (out/'episodes.csv').read_bytes()==(out/'episodes.partial.csv').read_bytes()
    with (out/'job_completions.csv').open() as f:assert len(list(csv.DictReader(f)))==1296
    before=(out/'manifest.json').read_bytes()
    with pytest.raises(ValueError):m.run(args)
    assert (out/'manifest.json').read_bytes()==before
