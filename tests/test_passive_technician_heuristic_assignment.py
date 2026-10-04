import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import passive_technician_heuristic_assignment as exp
from ht_pdm_fjsp.passive_technician_marl import PassiveTechnicianEnv, PassiveState


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_feature_order_and_padding(n):
    local = torch.arange(n*15, dtype=torch.float32).reshape(1, n, 15)
    augmented = exp.conditioned_inputs(local)
    assert augmented.shape == (1, n, 80)
    for i in range(n):
        assert torch.equal(augmented[0, i, :15], local[0, i])
        assert augmented[0, i, 15:20].tolist() == [float(i == j) for j in range(5)]
        others = [j for j in range(n) if i != j]
        assert torch.equal(augmented[0, i, 20:20+15*(n-1)].reshape(n-1, 15), local[0, others])
        assert (augmented[0, i, 20+15*(n-1):] == 0).all()


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_observable_teacher_determinism_feasibility_and_matching(n):
    config = exp.cells('full', n)['medium']
    env = PassiveTechnicianEnv(config, seed=42)
    for _ in range(config.horizon):
        observations, masks = env.observations(), env.action_masks()
        selected = exp.heuristic_action(observations, masks, config)
        assert selected == exp.heuristic_action(observations, masks, config)
        assert all(masks[i][a] for i, a in enumerate(selected))
        requests = [a for a in selected if a]
        assert len(requests) == len(set(requests))
        env.step(selected)
    assert env.metrics['invalid_requests'] == 0


def test_teacher_breaks_symmetry_and_defers_healthy_machine():
    config = exp.cells('full', 3)['medium']
    env = PassiveTechnicianEnv(config)
    assert exp.heuristic_action(env.observations(), env.action_masks(), config) == (0, 0, 0)
    env.state = replace(env.state, ages=(3,)*3)
    action = exp.heuristic_action(env.observations(), env.action_masks(), config)
    assert sorted(action) == [0, 1, 2]
    assert exp.expected_proxy_cost(0, False, 0, config) == 0


def test_teacher_reconstructs_busy_and_queue_work_from_observations():
    config = exp.cells('full', 3)['medium']
    env = PassiveTechnicianEnv(config)
    env.time = 4
    env.state = PassiveState((0, 3, 3), (False, False, False), (6, 4), (0, -1), ((1,), ()))
    action = exp.heuristic_action(env.observations(), env.action_masks(), config)
    assert action[:2] == (0, 0)
    assert action[2] == 2
    # Hide the alternative technician: teacher defers instead of joining a
    # queue whose visible workload exceeds its three-step proxy.
    masks = list(env.action_masks())
    masks[2] = (True, True, False)
    assert exp.heuristic_action(env.observations(), masks, config)[2] == 0


def test_teacher_never_calls_simulator_or_oracle(monkeypatch):
    config = exp.cells('full', 5)['high']
    env = PassiveTechnicianEnv(config)
    env.state = replace(env.state, ages=(4,)*5)
    obs, masks = env.observations(), env.action_masks()
    def forbidden(*args, **kwargs):
        raise AssertionError('privileged access')
    monkeypatch.setattr(PassiveTechnicianEnv, 'transition', forbidden)
    monkeypatch.setattr(exp.base, 'ExactOracle', forbidden)
    assert len(exp.heuristic_action(obs, masks, config)) == 5


def test_failure_grid_is_policy_independent_and_cost_reconciles():
    config = exp.cells('full', 4)['high']
    assert exp.failure_grid(config, 99200) == exp.failure_grid(config, 99200)
    assert exp.failure_grid(config, 99200) != exp.failure_grid(config, 99201)
    result = exp.rollout(config, 99200)
    assert result['invalid_requests'] == 0
    assert result['cost_reconciliation_error'] == 0
    assert result['objective'] == sum(result[k] for k in ('maintenance_cost', 'failure_cost', 'downtime_cost', 'queue_waiting_cost'))


def test_config_split_and_locked_budget():
    cfg = exp.settings('full')
    assert cfg['updates'] == 14873
    assert exp.seed_audit(cfg)['panels_disjoint']
    for n in range(2, 6):
        matrices = [c.service_time for c in exp.cells('full', n).values()]
        assert len(set(matrices)) == 6
        assert len(exp.arms_for(n)) == (3 if n == 2 else 2)


def test_td_baseline_has_no_teacher_labels_or_oracle_access(tmp_path, monkeypatch):
    configs = exp.cells('smoke', 3)
    cfg = exp.settings('smoke')
    cfg.update(env_steps=16, updates=1)
    def forbidden(*args, **kwargs):
        raise AssertionError('baseline teacher access')
    monkeypatch.setattr(exp, 'heuristic_action', forbidden)
    monkeypatch.setattr(exp.base, 'ExactOracle', forbidden)
    _, coverage = exp.fit(exp.model_for(configs['train_a'], cfg, 99400), configs, cfg,
                          99400, exp.ARMS[0], tmp_path, [])
    assert coverage['teacher_labels_seen'] == 0
    assert coverage['optimizer_updates'] == 1


def test_replay_labels_are_paired_with_actual_sampled_transitions():
    replay = exp.Replay(2)
    for i in range(3):
        x = torch.full((3, 15), float(i))
        replay.add_labeled(x, ((True,)*3,)*3, (0,)*3, -i, x, ((True,)*3,)*3, False, (i,)*3)
    batch = replay.sample(100, np.random.default_rng(4))
    assert torch.equal(batch['labels'][:, 0].float(), batch['x'][:, 0, 0])
    assert len(replay) == 2


def test_checkpoint_contract_and_original_monotonic_mixer(tmp_path):
    cfg = exp.settings('smoke')
    config = exp.cells('smoke', 5)['train_a']
    model = exp.model_for(config, cfg, 99400)
    exp.monotonic_audit(model, config)
    path = tmp_path/'model.pt'
    from dataclasses import asdict
    torch.save(dict(protocol_version=exp.PROTOCOL, feature_contract=exp.FEATURE_CONTRACT,
                    config=asdict(config), settings=cfg, train_seed=99400,
                    state_dict=model.state_dict()), path)
    restored = exp.load_checkpoint(path)
    x = torch.randn(2, 5, 15)
    assert torch.equal(model.agent_q(x), restored.agent_q(x))
    payload = torch.load(path, weights_only=True)
    payload['protocol_version'] = 'old'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        exp.load_checkpoint(path)


def test_nonempty_output_never_overwritten(tmp_path):
    (tmp_path/'keep').write_text('untouched')
    with pytest.raises(FileExistsError):
        exp.run(SimpleNamespace(profile='smoke', device='cpu', output_dir=str(tmp_path)))
    assert (tmp_path/'keep').read_text() == 'untouched'


def test_scientific_gate_uses_training_seeds_not_episode_count():
    cfg = exp.settings('full')
    episodes = []
    for n in range(2, 6):
        for arm in exp.arms_for(n):
            for s in cfg['train_seeds']:
                for cell in ('nominal', 'medium', 'high'):
                    for ev in cfg['evaluation_seeds']:
                        cost = 100. if arm == exp.ARMS[0] else 90.
                        episodes.append(dict(machines=n, algorithm=arm, train_seed=s, cell=cell,
                                             eval_seed=ev, objective=cost))
    pairs, summary = exp.summarize(episodes, cfg, True)
    assert len(pairs) == 10
    assert summary['primary']['passed']
    assert summary['primary']['relative_reduction'] == .1
    # A 4% effect remains FAIL even with zero-width CI and 10 wins.
    for r in episodes:
        if r['algorithm'] == exp.ARMS[1]:
            r['objective'] = 96.
    assert not exp.summarize(episodes, cfg, True)[1]['primary']['passed']


def test_source_removed_during_training_uses_frozen_provenance(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    source = tmp_path/'transient_runner.py'
    original = Path(exp.__file__).read_bytes()
    source.write_bytes(original)
    expected_hash = hashlib.sha256(original).hexdigest()
    seed_record = exp.seed_audit(exp.settings('smoke'))
    # Simulate a checkout change after preflight, without touching repository code.
    monkeypatch.setattr(exp, '__file__', str(source))
    monkeypatch.setattr(exp, 'seed_audit', lambda cfg: seed_record)
    original_settings = exp.settings
    def tiny_settings(profile):
        cfg = original_settings(profile)
        cfg.update(env_steps=16, updates=1)
        return cfg
    monkeypatch.setattr(exp, 'settings', tiny_settings)
    original_fit = exp.fit
    def delete_then_fit(*args, **kwargs):
        source.unlink(missing_ok=True)
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(exp, 'fit', delete_then_fit)
    output = tmp_path/'run'
    exp.run(SimpleNamespace(profile='smoke', device='cpu', output_dir=str(output)))
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED'
    assert manifest['source_sha256'] == expected_hash
    assert (output/'source_snapshot.py').read_bytes() == original
    assert not source.exists()
    for path in output.rglob('model.pt'):
        assert torch.load(path, weights_only=True)['teacher_definition_sha256'] == expected_hash
    teacher = json.loads((output/'teacher_audit.json').read_text())
    assert teacher['heuristic_definition_sha256'] == expected_hash
    assert 'source_snapshot.py' in manifest['outputs']
