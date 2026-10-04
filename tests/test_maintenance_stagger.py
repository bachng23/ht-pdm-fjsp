import argparse
import hashlib
import itertools
import json
from dataclasses import replace

import pytest
from ht_pdm_fjsp import maintenance_coupling as base
from ht_pdm_fjsp import maintenance_stagger as mod


def test_backward_reservation_staggers_shared_specialist_before_trigger():
    cfg=base.Config(((2,4),(2,4)),(1,1),5,.4,12)
    state=base.initial(cfg)
    assert mod.reserve_action(cfg,state,2,12)==(0,0)
    state,_,_=base.transition(cfg,state,(0,0),(False,False))
    action,plan=mod.reserve_action(cfg,state,2,11,True)
    assert action==(0,1)  # age 2, health trigger would wait until age 4
    assert [(r['start'],r['technician']) for r in plan]==[(2,0),(0,0)]
    assert base.Solver(cfg,1).baseline(state,base.POLICIES[1])==(0,0)


def test_lead_zero_cannot_start_healthy_maintenance_early():
    cfg=base.Config(((2,4),(2,4)),(2,2),5,.4,12)
    action,plan=mod.reserve_action(cfg,base.initial(cfg),0,12,True)
    assert action==(0,0)
    assert all(r['start']>=r['due'] for r in plan)


def test_calendar_accounts_for_busy_work_and_immutable_fifo_queue():
    cfg=base.Config(((2,4),(2,4)),(0,4),5,.4,12)
    state=base.State((0,4),(False,False),(2,0),(0,-1),((),()))
    action,plan=mod.reserve_action(cfg,state,2,12,True)
    assert action[0]==0 and plan[0]['machine']==1
    assert plan[0]['start']==0 and plan[0]['technician']==1
    queued=replace(state,assigned=(-1,-1),remaining=(0,0),queues=((0,),()))
    assert mod.reserve_action(cfg,queued,2,12)==action
    assert queued.queues==((0,),())


def test_failed_machine_prioritized_and_late_risk_uses_prices():
    cfg=base.Config(((1,3),(1,3)),(0,0),3,.6,12)
    state=replace(base.initial(cfg),failed=(True,False))
    a,plan=mod.reserve_action(cfg,state,2,12,True)
    assert plan[0]['machine']==0 and a[0]>0
    # Changing planned price changes score; no artificial future shock access.
    _,other=mod.reserve_action(replace(cfg,planned_cost=8),state,2,12,True)
    assert other[0]['score']>plan[0]['score']


def test_safe_deadline_beyond_horizon_does_not_trigger_useless_work():
    cfg=base.Config(((2,4),(4,2)),(0,0),5,.4,2)
    assert mod.reserve_action(cfg,base.initial(cfg),4,2)==(0,0)


def test_interval_search_respects_noncontiguous_reservations():
    intervals=[(0,2),(4,6)]
    assert mod.earliest_slot(1,2,intervals)==2
    assert mod.earliest_slot(1,3,intervals)==6
    assert not mod.overlaps(2,2,intervals)
    assert mod.overlaps(2,3,intervals)


def test_policy_feasibility_multiple_configurations_without_overlap():
    for seed in range(5):
        cfg=mod.cell_config(seed,'high','heterogeneous',4,12)
        for lead in range(5):
            state=base.initial(cfg)
            for step in range(12):
                a,plan=mod.reserve_action(cfg,state,lead,12-step,True)
                assert all(x in o for x,o in zip(a,base.options(cfg,state)))
                for x,y in itertools.combinations(plan,2):
                    if x['technician']==y['technician']:
                        assert not mod.overlaps(x['start'],x['duration'],[(y['start'],y['start']+y['duration'])])
                state,_,_=base.transition(cfg,state,a,tuple(u<cfg.probability for u in base.uniforms(10,step,2)))


def test_policy_json_round_trip_and_operating_proxy(tmp_path):
    cfg=mod.cell_config(850100,'high','heterogeneous',8,5)
    settings=dict(offset=1,lead=2,depth=2,tail=1)
    p=tmp_path/'policy.json';p.write_text(json.dumps(settings))
    a,b=mod.Solver(cfg,**settings),mod.Solver(cfg,**json.loads(p.read_text()))
    assert a.action(base.initial(cfg),5,mod.POLICIES[1])==b.action(base.initial(cfg),5,mod.POLICIES[1])
    result,_=mod.measure(a,mod.POLICIES[1],851200)
    assert result['operating_proxy']==1-(result['planned_steps']+result['failed_steps'])/10
    assert result['cost_per_step']==result['cost']/5


def test_panels_fresh_disjoint_and_simulator_unchanged():
    f,s=mod.panels('full'),mod.panels('smoke')
    for key in ('dev_configs','dev_episodes','configs','episodes'):
        assert not set(f[key])&set(s[key])
        assert not set(f[key])&set(base.panels('full')[key])
    assert not set(f['configs'])&set(f['dev_configs'])
    assert hashlib.sha256(__import__('pathlib').Path(base.__file__).read_bytes()).hexdigest()=='4dcbf3acefd4ae4258c591da1f5e0ed11ae74ffbd0de059ee557b4ccff63bc52'


def test_price_capacity_horizon_cells_only_change_declared_factors():
    configs=[mod.cell_config(850100,p,s,price,h) for (p,s),price,h in itertools.product(base.CELLS,(1,4,8),(12,24))]
    assert len({(c.initial_ages,c.failure_age,c.probability,c.failure_cost) for c in configs})==1
    assert len({tuple(sum(row)/len(row) for row in c.service) for c in configs})==1


def test_primary_noninferiority_and_operating_guard_use_instance_units():
    rows=[]
    for seed in range(16):
        for ep in range(4):
            for pol,cost,operating in zip(mod.POLICIES,(20.,16.,16.),(.6,.59,.58)):
                rows.append(dict(instance_seed=seed,episode_seed=ep,pressure='high',skill='heterogeneous',planned_price=1,horizon=24,policy=pol,cost=cost,stage_cost=cost,terminal_cost=0.,cost_per_step=cost/24,operating_proxy=operating))
    summary=mod.summarize(rows,'full')
    assert summary['gates']['joint_operational']
    assert summary['cells']['high_heterogeneous_price1_h24']['E_minus_C_relative']['n_instances']==16
    for r in rows:
        if r['policy']==mod.POLICIES[1]:r['operating_proxy']=.5
    assert not mod.summarize(rows,'full')['gates']['joint_operational']


def test_full_shock_schedule_is_unchanged():
    assert base.uniforms(10,2,2)==base.uniforms(10,2,2)
    assert base.uniforms(10,2,2)!=base.uniforms(10,3,2)


def test_smoke_artifacts_complete_and_overwrite_rejected(tmp_path):
    args=argparse.Namespace(profile='smoke',device='cpu',output_dir=str(tmp_path/'run'))
    mod.run(args)
    path=tmp_path/'run';manifest=json.loads((path/'manifest.json').read_text());summary=json.loads((path/'summary.json').read_text())
    assert manifest['status']=='COMPLETED' and not manifest['full_panel_opened']
    assert summary['audits']['actual_rows']==48 and summary['audits']['all_passed']
    assert not summary['gates']['joint_operational']
    assert (path/'episodes.csv').read_bytes()==(path/'episodes.partial.csv').read_bytes()
    with pytest.raises(ValueError,match='empty'):mod.run(args)


def test_zero_cost_cell_keeps_undefined_relative_endpoint_explicit():
    rows=[dict(instance_seed=1,episode_seed=1,pressure='high',skill='heterogeneous',planned_price=1,horizon=5,policy=pol,cost=0.,stage_cost=0.,terminal_cost=0.,cost_per_step=0.,operating_proxy=1.) for pol in mod.POLICIES]
    cell=mod.summarize(rows,'smoke')['cells']['high_heterogeneous_price1_h5']
    assert cell['E_minus_C_relative'] is None
    assert cell['B_minus_E_relative']['mean'] is None and cell['B_positive_instances']==0
