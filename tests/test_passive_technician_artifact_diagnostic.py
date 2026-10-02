import json
from pathlib import Path

import pytest
import torch

from ht_pdm_fjsp.passive_technician_artifact_diagnostic import (
    METRICS, STEPS, decision_trace, log_summary, run, source_inputs, summarize,
)
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig
from ht_pdm_fjsp.passive_technician_value_decomposition import PassiveValueDecomposition


def test_paired_seed_averaging_and_boundary_delta():
    rows = []
    for regime in ('nominal_100', 'curriculum_replay'):
        for seed in (1, 2):
            for step in STEPS:
                for evaluation in (3, 4):
                    delta = 0 if step == STEPS[0] else seed
                    value = evaluation + (delta if regime == 'curriculum_replay' else 0)
                    rows.append(dict(training_regime=regime, train_seed=seed, scenario='in_distribution',
                        budget=step, eval_seed=evaluation, **{m: value for m in METRICS}))
    seeds, pairs, aggregates = summarize(rows)
    assert len(seeds) == 12
    row = next(r for r in aggregates if r['metric'] == 'objective' and r['budget'] == STEPS[-1])
    assert row['training_seed_count'] == 2
    assert row['delta_mean'] == 1.5
    assert row['delta_from_boundary_mean'] == 1.5
    assert all(r['delta'] == 0 for r in pairs if r['budget'] == STEPS[0])


def test_trace_null_margin_when_only_defer_is_valid():
    cfg = PassiveConfig(horizon=6)
    model = PassiveValueDecomposition(cfg, 'tqmix')
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.agent.defer.bias.fill_(-1)
    rows = decision_trace(cfg, model, 0, 1, 'nominal_100', 'in_distribution', STEPS[0])
    assert len(rows) == 18
    restricted = [r for r in rows if r['feasible_actions'] == 1]
    assert restricted
    assert all(r['action'] == 0 and r['q_margin'] is None for r in restricted)
    assert all(r['q_margin'] >= 0 for r in rows if r['q_margin'] is not None)


def test_nonfinite_q_is_rejected():
    cfg = PassiveConfig()
    model = PassiveValueDecomposition(cfg, 'tqmix')
    with torch.no_grad():
        model.agent.defer.bias.fill_(float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        decision_trace(cfg, model, 0, 1, 'nominal_100', 'in_distribution', STEPS[0])


def test_log_summary_retains_missing_loss_and_gradient(tmp_path):
    (tmp_path / 'training_progress.csv').write_text(
        'training_regime,train_seed,training_phase,environment_steps,update_count,epsilon,td_loss,raw_cf_loss,weighted_cf_loss,total_loss\n'
        'nominal_100,1,nominal_training,12,0,1,,,,\n'
        'nominal_100,1,nominal_training,24,1,0.9,2,4,0.2,2.2\n')
    result = log_summary(tmp_path, [1])[0]
    assert result['td_loss_episode_mean'] == 2
    assert result['weighted_cf_loss_episode_mean'] == 0.2
    assert result['historical_gradient_norm'] is None
    assert result['episode_count'] == 2


def test_source_wrong_protocol_rejected(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'COMPLETED'}))
    with pytest.raises(ValueError, match='Source identity'):
        source_inputs(tmp_path)


def test_output_overwrite_and_source_descendant_rejected(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    with pytest.raises(ValueError, match='outside source'):
        run(source, source / 'new', 'smoke')
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        run(source, output, 'smoke')


@pytest.fixture
def source_panel(tmp_path):
    import csv
    from dataclasses import asdict
    from ht_pdm_fjsp.passive_technician_artifact_diagnostic import SOURCE_PROTOCOL, SOURCE_SHA, REGIMES
    from ht_pdm_fjsp.passive_technician_leave_one_out import environment_cells
    configs = {name: asdict(config) for name, config in environment_cells().items()}
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(status='COMPLETED', protocol_version=SOURCE_PROTOCOL,
        git_revision=SOURCE_SHA, train_seeds=list(range(84600, 84610)), evaluation_seeds=list(range(84700, 84750)), value_settings={})))
    (tmp_path / 'benchmark_config.json').write_text(json.dumps({'environment_cells': configs}))
    (tmp_path / 'resolved_config.json').write_text(json.dumps(dict(step_budgets=STEPS, algorithm='tqmix', settings={},
        regime_specs={'curriculum_replay': {'phase_two_nominal_replay_share': 0.875}})))
    with (tmp_path / 'episodes.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['training_regime', 'train_seed', 'scenario', 'budget', 'eval_seed'])
        for regime in REGIMES:
            for seed in range(84600, 84610):
                for scenario in configs:
                    for step in STEPS:
                        for evaluation in range(84700, 84750):
                            writer.writerow([regime, seed, scenario, step, evaluation])
                for step in STEPS:
                    path = tmp_path / regime / f'train_seed_{seed}' / 'checkpoints' / f'step_{step}' / 'model.pt'
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b'preflight fixture; never loaded')
    for name in ('training_progress.csv', 'adaptation_diagnostics.csv', 'training_schedule.csv'):
        (tmp_path / name).write_text('fixture\n')
    return tmp_path


def test_preflight_requires_all_checkpoints_even_for_smoke(source_panel):
    _, _, index, files = source_inputs(source_panel)
    assert len(index) == 12000
    assert len(files) == 67
    files[-1].unlink()
    with pytest.raises(ValueError, match='Missing source'):
        source_inputs(source_panel)


def test_preflight_rejects_duplicate_episode(source_panel):
    path = source_panel / 'episodes.csv'
    with path.open('a') as handle:
        handle.write('nominal_100,84600,in_distribution,240192,84700\n')
    with pytest.raises(ValueError, match='coverage'):
        source_inputs(source_panel)


def test_preflight_rejects_changed_environment(source_panel):
    path = source_panel / 'benchmark_config.json'
    payload = json.loads(path.read_text())
    payload['environment_cells']['in_distribution']['horizon'] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='configurations'):
        source_inputs(source_panel)
