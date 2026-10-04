import argparse, json, math, csv
from dataclasses import replace
import pytest
from ht_pdm_fjsp import maintenance_backlog_adaptive as m


def config():
    c=m.base.Config(((2,6),(2,6)),(8,3),12,.6,1000,request_order=(0,1))
    return m.repair.RepairProduction(c,((30,),(30,)),2)


def without_time(r):return {k:v for k,v in r.items() if k!='decision_seconds'}


@pytest.mark.parametrize('ratio',[1,2,4])
def test_zero_gain_exact_old_health_and_reservation_episodes(ratio):
    cfg=m.scenario(910100,'high','heterogeneous',6,ratio,4)
    for pol,params in [(m.POLICIES[0],dict(offset=2)),(m.POLICIES[1],dict(offset=2,gain=0)),(m.POLICIES[2],dict(lead=6))]:
        expected=pol if pol!=m.POLICIES[1] else m.POLICIES[0]
        r,ts,cs=m.episode(cfg,pol,911100,trace=True,**params)
        rr,tts,ccs=m.repair.episode(cfg,expected,911100,trace=True,**{k:v for k,v in params.items() if k!='gain'})
        assert without_time(r)==without_time(rr) and cs==ccs and len(ts)==len(tts)
        assert all(all(a[k]==v for k,v in b.items()) for a,b in zip(ts,tts))


def test_backlog_raises_threshold_with_same_risk_assignment():
    cfg=config();state=m.base.State((8,3),(False,False),(2,0),(1,-1),((),()))
    assert m.action(cfg,state,(30,30),m.POLICIES[0],offset=2)==(0,0)
    assert m.action(cfg,state,(30,30),m.POLICIES[1],offset=2,gain=1)==(1,0)
    empty=m.base.initial(cfg.maintenance)
    assert m.action(cfg,empty,(30,30),m.POLICIES[1],offset=2,gain=1)==(0,0)
    assert m.action(cfg,state,(1,30),m.POLICIES[1],offset=2,gain=1)==(0,0)


def test_committed_queue_and_busy_targets_are_preserved():
    cfg=config();state=m.base.State((12,3),(True,False),(2,0),(0,-1),((1,),()))
    assert m.action(cfg,state,(30,30),m.POLICIES[1],offset=2,gain=1)==(0,0)
    a=(0,0)
    assert m.intervene(cfg,state,(30,30),a,1,0)[0:2]==(a,False)
    assert m.intervene(cfg,state,(30,30),a,0,1)[0:2]==(a,False)


def test_intervention_replaces_only_new_requests():
    cfg=config();state=m.base.initial(cfg.maintenance)
    a,applied,_=m.intervene(cfg,state,(30,30),(1,0),1,0)
    assert a==(0,1) and applied
    assert m.intervene(cfg,state,(30,0),(1,0),1,0)[0:2]==((1,0),False)
    with pytest.raises(ValueError):m.intervene(cfg,state,(30,30),(0,0),1,2)


def test_single_tick_intervention_same_prefix_and_events():
    cfg=m.scenario(900215,'high','heterogeneous',64,4,6)
    a,ta,ja=m.episode(cfg,m.POLICIES[2],901200,lead=6,trace=True)
    b,tb,jb=m.episode(cfg,m.POLICIES[2],901200,lead=6,trace=True,intervention=(9,5,0))
    assert a['makespan']==657
    assert ta[:9]==tb[:9] and sum(t['intervention_applied'] for t in tb)==1
    assert all(x['events']==y['events'] for x,y in zip(ta,tb))
    assert all(t['intervention_reason']=='not intervention tick' for t in tb if t['step']!=9)
    assert b['productive_steps']==b['required_work'] and b['completed_jobs']==384


def test_p95_worst20_and_instance_inference():
    assert m.quantile95(list(range(1,21)))==19 and m.worst20(list(range(1,21)))==18.5
    assert m.quantile95([1,2])==2
    assert m.bootstrap([.01,.01])==dict(mean=.01,ci95=[.01,.01],n_instances=2)
    with pytest.raises(ValueError):m.bootstrap([])


def test_seed_panels_fresh_and_splits_disjoint():
    full=m.panels('full');smoke=m.panels('smoke');old=m.repair.panels('full')
    for field in ('configs','episodes'):
        dev='dev_'+field
        assert not set(full[field])&set(full[dev])
        assert not (set(full[field])|set(full[dev]))&(set(smoke[field])|set(smoke[dev])|set(old[field])|set(old[dev]))
    assert len(full['case_episodes'])==20 and len(m.grid(m.POLICIES[1]))==8


def test_checkpoint_selection_ties():
    rows=[dict(ratio=2,policy=m.POLICIES[1],parameter=json.dumps(p,sort_keys=True),normalized_makespan=1) for p in m.grid(m.POLICIES[1])]
    assert m.select(rows,2,m.POLICIES[1])==dict(offset=0,gain=0)


def test_full_dirty_fails_without_opening_scenarios(tmp_path,monkeypatch):
    monkeypatch.setattr(m.base,'git',lambda *args:'dirty' if args[0]=='status' else 'commit')
    monkeypatch.setattr(m,'scenario',lambda *a,**k:pytest.fail('full scenario opened'))
    out=tmp_path/'maintenance_backlog_full_test'
    with pytest.raises(RuntimeError):m.run(argparse.Namespace(profile='full',output_dir=str(out),device='cpu'))
    failed=json.loads((out/'manifest.json').read_text())
    assert failed['status']=='FAILED' and not failed['full_panel_opened']


def test_small_smoke_artifacts_sealed_guard_and_no_overwrite(tmp_path,monkeypatch):
    original=m.scenario
    def sealed(seed,*args,**kwargs):
        assert seed in (910100,910200,900215)
        return original(seed,*args,**kwargs)
    monkeypatch.setattr(m,'scenario',sealed)
    out=tmp_path/'maintenance_backlog_smoke_test'
    args=argparse.Namespace(profile='smoke',output_dir=str(out),device='cpu')
    m.run(args)
    manifest=json.loads((out/'manifest.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert manifest['status']=='COMPLETED' and not manifest['full_panel_opened']
    assert summary['audits']['development']==768 and summary['audits']['episodes']==144 and summary['audits']['configs']==24
    assert summary['counterfactual']['episodes']==7 and not any(summary['gates'].values())
    assert (out/'episodes.csv').read_bytes()==(out/'episodes.partial.csv').read_bytes()
    for f in ('decisions.csv','job_completions.csv','counterfactual_decisions.csv','counterfactual_job_completions.csv','coordination.csv','development.csv'):
        with (out/f).open() as stream:assert len(list(csv.DictReader(stream)))>0
    with pytest.raises(ValueError):m.run(args)


def test_safety_cap_marks_failure_without_truncation():
    cfg=config();cfg=replace(cfg,maintenance=replace(cfg.maintenance,horizon=1))
    with pytest.raises(RuntimeError):m.episode(cfg,m.POLICIES[1],911100,gain=1)

@pytest.mark.parametrize('tail,passes',[(104,True),(106,False)])
def test_mean_and_tail_gates_are_separate_instance_aggregates(tail,passes):
    template,_,_=m.episode(config(),m.POLICIES[0],911100)
    rows=[]
    for i,(p,sk),l,ratio,pol,e in __import__('itertools').product((910100,910200),m.CELLS,(3,6),(1,2,4),m.POLICIES,(911100,911101)):
        value=100 if pol==m.POLICIES[2] else (101 if ratio==2 else tail) if pol==m.POLICIES[1] else 102
        rows.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio,policy=pol,episode_seed=e,**{**template,'makespan':value}))
    summary=m.summarize(rows,'full',[3,6])
    assert summary['gates']==dict(mean_adequacy=True,tail_guard=passes,overall_adequacy=passes)
    assert summary['primary']['n_instances']==2 and math.isclose(summary['primary']['mean'],.01)
    assert not any(m.summarize(rows,'smoke',[3,6])['gates'].values())
