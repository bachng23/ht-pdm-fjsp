import argparse
import itertools
import json
import math
from dataclasses import replace

import pytest

from ht_pdm_fjsp.maintenance_coupling import (
    CELLS, COMPONENTS, POLICIES, Config, Solver, State, audit, closeout,
    initial, options, panels, run, scenario, summarize, transition, uniforms,
)


def test_service_is_down_and_recovers_only_at_completion():
    cfg = Config(((2,),), (2,), 3, 1., 3, request_order=(0,))
    s, cost, metrics = transition(cfg, initial(cfg), (1,), (True,))
    assert s.ages == (2,) and s.remaining == (1,) and not s.failed[0]
    assert metrics['planned_steps'] == 1 and metrics['failures'] == 0
    assert cost == cfg.maintenance_cost + cfg.planned_cost
    s, cost, metrics = transition(cfg, s, (0,), (True,))
    assert s.ages == (0,) and s.remaining == (0,) and s.assigned == (-1,)
    assert metrics['planned_steps'] == 1 and metrics['failures'] == 0
    s, _, _ = transition(cfg, s, (0,), (True,))
    assert s.ages == (1,) and not s.failed[0]


def test_busy_request_is_valid_queue_ages_fails_and_fifo_persists():
    cfg = Config(((3,), (2,)), (0, 2), 3, 1.)
    s, _, info = transition(cfg, initial(cfg), (1, 1), (True, True))
    assert s.queues == ((1,),) and s.failed == (False, True)
    assert info['waiting'] == 1 and info['planned_steps'] == 1 and info['failures'] == 1
    assert options(cfg, s) == ((0,), (0,))
    s, _, info = transition(cfg, s, (0, 0), (False, False))
    assert info['failed_steps'] == 1 and s.failed[1]
    s, _, _ = transition(cfg, s, (0, 0), (False, False))
    assert s.assigned == (-1,) and s.queues == ((1,),)
    s, _, info = transition(cfg, s, (0, 0), (True, True))
    assert s.assigned == (1,) and s.failed[1] and info['planned_steps'] == 1
    s, _, _ = transition(cfg, s, (0, 0), (True, True))
    assert not s.failed[1] and s.ages[1] == 0


def test_failed_service_does_not_double_charge_failed_downtime():
    cfg = Config(((2,),), (0,), request_order=(0,))
    s = replace(initial(cfg), failed=(True,))
    ns, _, info = transition(cfg, s, (1,), (True,))
    assert ns.failed[0] and info['cost_failed_down'] == 0
    assert info['cost_planned_down'] == cfg.planned_cost


def test_b_assigns_around_busy_fast_worker_without_changing_trigger():
    cfg = Config(((1, 3), (1, 3)), (2, 2), 3)
    s = State((2, 2), (False, False), (4, 0), (0, -1), ((), ()))
    solver = Solver(cfg, offset=1)
    assert solver.baseline(s, POLICIES[0]) == (0, 1)
    assert solver.baseline(s, POLICIES[1]) == (0, 2)
    for policy in POLICIES[:2]:
        assert tuple(m for m,a in enumerate(solver.baseline(s, policy)) if a) == solver.triggered(s)


def independent_tree(cfg, s, left):
    """No solver branches/memoized recurrence: enumerate full Bernoulli action tree."""
    if left == 0: return closeout(cfg, s)
    values = []
    for action in itertools.product(*options(cfg, s)):
        value = 0.
        for shocks in itertools.product((False, True), repeat=len(s.ages)):
            prob = math.prod(cfg.probability if f else 1-cfg.probability for f in shocks)
            if prob:
                ns, cost, _ = transition(cfg, s, action, shocks)
                value += prob * (cost + independent_tree(cfg, ns, left-1))
        values.append(value)
    return min(values)


@pytest.mark.parametrize('probability', [0., .37, 1.])
def test_oracle_matches_independent_unmemoized_action_tree(probability):
    cfg = Config(((1,), (2,)), (1, 1), 2, probability, 3)
    solver = Solver(cfg, depth=2, tail=1)
    optimal = solver.oracle_value(initial(cfg), 3)
    assert optimal == pytest.approx(independent_tree(cfg, initial(cfg), 3), abs=1e-9)
    for policy in POLICIES:
        assert solver.exact_policy_cost(policy, 3) >= optimal - 1e-9


def test_hand_calculated_oracle_and_irrelevant_service_shocks():
    cfg = Config(((2,),), (2,), 3, 1., 2, request_order=(0,))
    solver = Solver(cfg)
    assert solver.oracle_value(initial(cfg), 2) == 4.
    assert list(solver.branches(initial(cfg), (1,)))[0][0] == 1.


def test_physical_machine_permutation_preserves_fifo_oracle_value():
    cfg = Config(((1,3), (3,1)), (1, 0), 2, .4, 3)
    permuted = replace(cfg, service=cfg.service[::-1], initial_ages=cfg.initial_ages[::-1], request_order=(1,0))
    a = Solver(cfg).oracle_value(initial(cfg), 3)
    b = Solver(permuted).oracle_value(initial(permuted), 3)
    assert a == pytest.approx(b)


def test_cells_match_duration_means_and_only_capacity_or_skills_change():
    for seed in range(10):
        cfgs = [scenario(seed, p, skill, 5) for p,skill in CELLS]
        assert len({tuple(sum(r)/len(r) for r in cfg.service) for cfg in cfgs}) == 1
        assert len({(cfg.initial_ages, cfg.failure_age, cfg.probability, cfg.failure_cost) for cfg in cfgs}) == 1
        assert [len(c.service[0]) for c in cfgs] == [4,4,2,2]


def test_seed_panels_and_shock_schedule_are_policy_independent():
    sm, full = panels('smoke'), panels('full')
    for key in ('configs','episodes','dev_configs','dev_episodes'):
        assert not set(sm[key]) & set(full[key])
    assert not set(full['configs']) & set(full['dev_configs'])
    assert not set(full['episodes']) & set(full['dev_episodes'])
    assert uniforms(10,2,2) == uniforms(10,2,2)
    assert uniforms(10,2,2) != uniforms(10,3,2)


def test_cache_budget_fails_instead_of_silent_approximation():
    cfg = Config(((2,),), (1,), 2, .5, 3, request_order=(0,))
    with pytest.raises(RuntimeError, match='cache budget'):
        Solver(cfg, max_states=1).oracle_value(initial(cfg),3)


def test_policy_json_roundtrip_and_cost_components(tmp_path):
    cfg = scenario(770200, 'high', 'heterogeneous', 5)
    settings = {'offset':1,'depth':2,'tail':1}
    path = tmp_path / 'policy.json'; path.write_text(json.dumps(settings))
    a, b = Solver(cfg, **settings), Solver(cfg, **json.loads(path.read_text()))
    assert a.choose(initial(cfg),5) == b.choose(initial(cfg),5)
    _, cost, info = transition(cfg, initial(cfg), (0,0), (True,True))
    assert cost == sum(info['cost_'+key] for key in COMPONENTS)


def test_runner_completes_artifacts_and_refuses_overwrite(tmp_path):
    out = tmp_path / 'run'
    args = argparse.Namespace(profile='smoke', device='cpu', output_dir=str(out))
    run(args)
    manifest = json.loads((out/'manifest.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert manifest['status'] == 'COMPLETED' and not manifest['full_panel_opened']
    assert summary['audits']['actual_rows'] == 48 and summary['audits']['all_passed']
    assert summary['scientific_status'] == 'ENGINEERING_ONLY'
    assert summary['primary_gate'] is False
    for file in ('policy.json','episodes.csv','decisions.csv','coordination.csv','exact_oracle.csv'):
        assert (out/file).stat().st_size > 0
    with pytest.raises(ValueError, match='empty'):
        run(args)


def test_fixed_timing_control_has_same_trigger_but_cost_aware_planning():
    cfg = Config(((1,3), (3,1)), (1,1), 3, .4, 5)
    solver = Solver(cfg, 1, 2, 1)
    s = initial(cfg)
    assert solver.choose_fixed(s, 5) == (0,0)
    s = replace(s, ages=(2,2))
    assert all(a > 0 for a in solver.choose_fixed(s, 5))
    fixed = solver.fixed_value(s, 2, 1)
    unrestricted = solver.value(s, 2, 1)
    assert unrestricted <= fixed + 1e-9


def test_inference_uses_instances_not_episode_rows():
    rows = []
    for seed in (1,2,3):
        for pressure,skill in CELLS:
            for ep in range(20):
                for policy,cost in zip(POLICIES,(12.,10.,8.,9.)):
                    rows.append(dict(instance_seed=seed, pressure=pressure, skill=skill, episode_seed=ep, policy=policy, cost=cost))
    result = summarize(rows,'full')
    assert result['cells']['high_heterogeneous']['B_minus_C']['n_instances'] == 3
    assert result['cells']['high_heterogeneous']['B_minus_C']['mean'] == 2.
    assert result['cells']['high_heterogeneous']['D_minus_C']['mean'] == 1.
    assert result['primary_gate'] is True
    assert result['stronger_mechanism_gate'] is False  # no interaction
