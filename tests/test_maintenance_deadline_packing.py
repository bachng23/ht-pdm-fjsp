import argparse,csv,json,itertools,math
from dataclasses import replace
import pytest
from ht_pdm_fjsp import maintenance_deadline_packing as m


def small(service=((2,6),),ages=(6,),limit=12,ratio=4):
    c=m.base.Config(service,ages,limit,.6,2000,request_order=tuple(range(len(service))))
    return m.repair.RepairProduction(c,((30,),)*len(service),ratio)


def metrics(r):return {k:v for k,v in r.items() if k!='decision_seconds'}


@pytest.mark.parametrize('ratio',[1,2,4])
def test_old_H_S_exact_episode_trace_and_completions(ratio):
    cfg=m.scenario(930100,'high','heterogeneous',6,ratio,4)
    for pol,params in [(m.POLICIES[0],dict(offset=2)),(m.POLICIES[1],dict(lead=6))]:
        r,ts,cs=m.episode(cfg,pol,931100,trace=True,**params)
        rr,tt,cc=m.repair.episode(cfg,pol,931100,trace=True,**params)
        assert metrics(r)==metrics(rr) and cs==cc and len(ts)==len(tt)
        assert all(all(a[k]==v for k,v in b.items()) for a,b in zip(ts,tt))


def test_forward_packing_avoids_latest_slot_collision_with_exact_tiny_oracle():
    # Worker1 busy four ticks; safe starts are <=1 and <=2 on fast worker0.
    cfg=small(service=((2,6),)*3,ages=(10,9,4))
    state=m.base.State((10,9,4),(False,)*3,(0,4),(-1,2),((),()))
    slots=m.plan(cfg,state,(30,30,30),lead=6)
    assert [(s['machine'],s['worker'],s['start']) for s in slots]==[(0,0,0),(1,0,2)]
    assert m.action(cfg,state,(30,30,30),m.POLICIES[2],lead=6)==(1,0,0)
    assert m.action(cfg,state,(30,30,30),m.POLICIES[1],lead=6)==(0,0,0)
    exact=[(x,y) for x,y in itertools.product(range(2),range(3)) if x+2<=y]
    assert exact==[(0,2)] and 1+2>2


def test_whole_chain_shift_preserves_slack_and_avoids_premature_service():
    cfg=small(ages=(6,));state=m.base.initial(cfg.maintenance)
    slots=m.plan(cfg,state,(30,),lead=6)
    assert slots[0]['forward_start']==0 and slots[0]['shift']==5 and slots[0]['start']==5
    assert m.action(cfg,state,(30,),m.POLICIES[2],lead=6)==(0,)
    assert m.plan(cfg,state,(5,),lead=6)==[]


def test_specialist_assignment_and_release_windows():
    cfg=small(service=((6,2),(2,6)),ages=(10,10))
    state=m.base.initial(cfg.maintenance);slots=m.plan(cfg,state,(30,30),lead=0)
    assert [(s['machine'],s['worker'],s['start']) for s in slots]==[(0,1,1),(1,0,1)]
    assert all(s['start']>=s['release'] and s['duration']==2 for s in slots)


def test_failed_CM_duration_and_no_new_queues():
    cfg=small(ages=(11,));state=m.base.State((12,),(True,),(0,0),(-1,-1),((),()))
    assert m.plan(cfg,state,(30,),lead=6)[0]['duration']==8
    assert m.action(cfg,state,(30,),m.POLICIES[2],lead=6)==(1,)
    ns,w,produced,info=m.transition(cfg,state,(30,),(1,),(True,))
    assert ns.remaining[0]==7 and info['cm_jobs']==1 and not any(ns.queues)


def test_committed_service_FIFO_and_occupied_masks_preserved():
    cfg=small(service=((2,6),)*3,ages=(10,9,8))
    state=m.base.State((10,9,8),(False,)*3,(1,0),(0,-1),((1,),()))
    slots=m.plan(cfg,state,(30,30,30),lead=6)
    assert [s['machine'] for s in slots]==[2]
    assert all(s['start']>=3 for s in slots if s['worker']==0)
    assert m.action(cfg,state,(30,30,0),m.POLICIES[2],lead=6)==(0,0,0)
    a=m.action(cfg,state,(30,30,30),m.POLICIES[2],lead=6)
    assert a[0]==a[1]==0


def test_planner_cannot_access_hazards_and_is_deterministic(monkeypatch):
    cfg=small();state=m.base.initial(cfg.maintenance)
    monkeypatch.setattr(m.base,'uniforms',lambda *args:pytest.fail('planner read shocks'))
    assert m.plan(cfg,state,(30,),6)==m.plan(cfg,state,(30,),6)


def test_fresh_split_and_equal_tuning_grids():
    p=m.panels('full');s=m.panels('smoke')
    from ht_pdm_fjsp import maintenance_backlog_adaptive as prev
    old=prev.panels('full')
    for k in ('configs','episodes'):
        assert not set(p[k])&set(p['dev_'+k])
        assert not set(p[k]+p['dev_'+k])&set(s[k]+s['dev_'+k]+old[k]+old['dev_'+k])
    assert all(len(m.grid(pol))==4 for pol in m.POLICIES)


def test_three_independent_gates_and_instance_inference():
    template,_,_=m.episode(small(),m.POLICIES[0],931100)
    rows=[]
    for i,(p,sk),l,r,pol,e in itertools.product((930100,930200),m.CELLS,(3,6),(1,2,4),m.POLICIES,(931100,931101)):
        value=101 if pol==m.POLICIES[2] else 100
        rows.append(dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=r,policy=pol,episode_seed=e,**{**template,'makespan':value}))
    ss=m.summarize(rows,'full',[3,6]);assert all(ss['gates'].values()) and ss['primary']['n_instances']==2
    for r in rows:
        if r['ratio']==4 and r['pressure']=='high' and r['skill']=='uniform' and r['policy']==m.POLICIES[2]:r['makespan']=106
    ss=m.summarize(rows,'full',[3,6])
    assert ss['gates']==dict(mean_adequacy=True,heterogeneous_tail_guard=True,uniform_tail_guard=False,overall_adequacy=False)
    assert not any(m.summarize(rows,'smoke',[3,6])['gates'].values())


def test_checkpoint_tie_and_roundtrip():
    pol=m.POLICIES[2];rows=[dict(ratio=2,policy=pol,parameter=json.dumps(params,sort_keys=True),normalized_makespan=1) for params in m.grid(pol)]
    assert m.select(rows,2,pol)==dict(lead=0)
    cfg=small();state=m.base.initial(cfg.maintenance)
    params=json.loads(json.dumps(dict(lead=6)))
    assert m.action(cfg,state,(30,),pol,**params)==m.action(cfg,state,(30,),pol,lead=6)


def test_dirty_full_stops_before_any_full_scenario(tmp_path,monkeypatch):
    monkeypatch.setattr(m.base,'git',lambda *args:'dirty' if args[0]=='status' else 'commit')
    monkeypatch.setattr(m,'scenario',lambda *a,**kw:pytest.fail('opened full scenario'))
    out=tmp_path/'maintenance_deadline_full_test'
    with pytest.raises(RuntimeError):m.run(argparse.Namespace(profile='full',output_dir=str(out),device='cpu'))
    status=json.loads((out/'manifest.json').read_text());assert status['status']=='FAILED' and not status['full_panel_opened']


def test_complete_smoke_artifacts_sealed_guard_and_plan_audits(tmp_path,monkeypatch):
    original=m.scenario
    def guard(seed,*args,**kwargs):
        assert seed in (930100,930200,900215,920216,920211)
        return original(seed,*args,**kwargs)
    monkeypatch.setattr(m,'scenario',guard)
    out=tmp_path/'maintenance_deadline_smoke_test';args=argparse.Namespace(profile='smoke',output_dir=str(out),device='cpu')
    m.run(args);status=json.loads((out/'manifest.json').read_text());s=json.loads((out/'summary.json').read_text())
    assert status['status']=='COMPLETED' and not status['full_panel_opened']
    assert s['audits']['development']==576 and s['audits']['episodes']==144 and s['audits']['configs']==24
    assert s['regressions']['episodes']==9 and s['regressions']['job_completions']==3456
    assert not any(s['gates'].values())
    assert (out/'episodes.csv').read_bytes()==(out/'episodes.partial.csv').read_bytes()
    for file in ('decisions.csv','regression_decisions.csv'):
        with (out/file).open() as f:
            for row in csv.DictReader(f):
                slots=json.loads(row['deadline_plan']);state=json.loads(row['state']);action=json.loads(row['action'])
                if row['policy']==m.POLICIES[2]:
                    emitted=[slot for slot in slots if slot['start']==0]
                    assert sum(x>0 for x in action)==len(emitted)
                    for slot in emitted:assert state['remaining'][slot['worker']]==0 and not state['queues'][slot['worker']]
                    assert float(row['waiting'])==0
    with pytest.raises(ValueError):m.run(args)


def test_cap_failure_not_censored():
    cfg=small();cfg=replace(cfg,maintenance=replace(cfg.maintenance,horizon=1))
    with pytest.raises(RuntimeError):m.episode(cfg,m.POLICIES[2],931100,lead=6)
