import argparse
import json

import pytest
import torch

from ht_pdm_fjsp.passive_technician_cf_adaptation_factorial import (
    ALGORITHMS, FULL_TRAIN_SEEDS, FULL_EVALUATION_SEEDS,
    interaction_results, prior_seed_registry, profile_settings, run,
)
from ht_pdm_fjsp.passive_technician_leave_one_out import environment_cells
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition, ValueTrainSettings, record_loss_gradients,
    train_value_decomposition_step_checkpoints,
)


def test_panels_and_four_arm_budget_are_locked():
    registry, used = prior_seed_registry()
    assert len(registry['manifests_scanned']) == 163
    assert not (set(FULL_TRAIN_SEEDS) | set(FULL_EVALUATION_SEEDS)) & used
    full = profile_settings('full')
    assert full[2] == (240192, 360288, 480384)
    assert len(full[0]) == 10 and len(full[1]) == 50
    assert full[-1] == 112


def test_interaction_is_within_seed_and_smoke_has_no_gate():
    arms = {}
    for algorithm in ALGORITHMS:
        rows = []
        for seed in range(10):
            for regime in ('nominal_100', 'curriculum_replay'):
                baseline = 10 if algorithm == 'tqmix' else 9
                gap = 3 if algorithm == 'tqmix' else 1
                rows.append(dict(training_regime=regime, train_seed=seed,
                    scenario='in_distribution', step_budget=480384,
                    objective_mean=baseline + (gap if regime == 'curriculum_replay' else 0)))
        arms[algorithm] = rows
    rows, decision = interaction_results(arms, True)
    assert len(rows) == 10
    assert decision['interaction_mean'] == -2
    assert decision['improving_seed_count'] == 10
    assert decision['primary_hypothesis_passed'] is True
    assert decision['no_cf_curriculum_nominal_delta_mean'] == -3
    _, smoke = interaction_results(arms, False)
    assert smoke['primary_hypothesis_passed'] is None


def test_full_and_no_cf_have_identical_initial_capacity_and_q():
    config = environment_cells()['in_distribution']
    models = []
    for algorithm in ALGORITHMS:
        torch.manual_seed(85000)
        models.append(PassiveValueDecomposition(config, algorithm))
    assert models[0].parameter_counts() == models[1].parameter_counts()
    assert all(torch.equal(p, models[1].state_dict()[k]) for k, p in models[0].state_dict().items())
    assert models[0].use_counterfactual and not models[1].use_counterfactual


def test_gradient_observer_records_active_cf_without_mutation():
    config = environment_cells()['in_distribution']
    torch.manual_seed(1)
    model = PassiveValueDecomposition(config, 'tqmix', 8, 8)
    local = torch.rand(4, config.machines, config.observation_dim)
    actions = torch.zeros(4, config.machines, dtype=torch.long)
    masks = torch.ones(4, config.machines, config.technicians + 1, dtype=torch.bool)
    td = (model.total_q(local, actions) - 1).square().mean()
    cf = .05 * model.local_edge_consistency(local, actions, masks)
    rng = torch.random.get_rng_state().clone()
    rows = []
    record_loss_gradients(model, td, cf, rows, 24, 1)
    assert rows[0]['td_gradient_norm'] > 0
    assert rows[0]['weighted_cf_gradient_norm'] > 0
    assert -1.00001 <= rows[0]['gradient_cosine'] <= 1.00001
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(rng, torch.random.get_rng_state())
    (td + cf).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_no_cf_gradient_is_zero_and_cosine_null():
    config = environment_cells()['in_distribution']
    model = PassiveValueDecomposition(config, 'tqmix_no_cf', 8, 8)
    local = torch.rand(2, config.machines, config.observation_dim)
    actions = torch.zeros(2, config.machines, dtype=torch.long)
    td = model.total_q(local, actions).square().mean()
    rows = []
    record_loss_gradients(model, td, torch.tensor(0.), rows, 24, 1)
    assert rows[0]['weighted_cf_gradient_norm'] == 0
    assert rows[0]['gradient_cosine'] is None


def test_sampled_gradient_observer_leaves_training_trajectory_identical(tmp_path):
    cells = environment_cells()
    config = cells['in_distribution']
    settings = ValueTrainSettings(episodes=4, hidden_dim=8, mixer_hidden_dim=8,
        batch_size=4, replay_capacity=64, learning_starts=4, train_frequency=2,
        target_update_interval=24)
    states = []
    records = []
    for diagnostic in (False, True):
        checkpoints, _ = train_value_decomposition_step_checkpoints(config, 'tqmix', 77,
            (24, 48), tmp_path / str(diagnostic), settings, torch.device('cpu'),
            cells, ['in_distribution'] * 4, gradient_diagnostic_interval=12 if diagnostic else 0,
            gradient_records=records if diagnostic else None)
        states.append(PassiveValueDecomposition.load(checkpoints[48], config, torch.device('cpu')).state_dict())
    assert records
    assert all(torch.equal(v, states[1][k]) for k, v in states[0].items())


def test_output_directory_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run(argparse.Namespace(profile='smoke', output_dir=tmp_path, device='cpu'))


def test_smoke_four_arms_complete_and_cross_arm_audits_pass(tmp_path):
    output = tmp_path / 'factorial'
    run(argparse.Namespace(profile='smoke', output_dir=output, device='cpu'))
    manifest = json.loads((output / 'manifest.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    assert manifest['status'] == 'COMPLETED'
    assert manifest['expected_trajectories'] == 4
    assert manifest['expected_checkpoints'] == 12
    assert manifest['expected_evaluation_rows'] == 144
    assert summary['audits']['trace_row_count'] == 7128
    assert all(v for v in summary['audits'].values() if isinstance(v, bool))
    assert summary['promotion_passed'] is None
    assert all(g['gate_applicable'] is False for g in summary['algorithm_gates'].values())
    no_cf = (output / 'tqmix_no_cf' / 'training_progress.csv').read_text()
    assert 'cumulative_optimizer_updates' in no_cf


def test_good_interaction_cannot_promote_a_worse_absolute_policy():
    from ht_pdm_fjsp.passive_technician_cf_adaptation_factorial import practical_promotion
    primary = dict(gate_applicable=True, primary_hypothesis_passed=True,
                   no_cf_curriculum_nominal_delta_mean=2)
    assert practical_promotion(primary, {'confirmation_passed': True}, -0.1) is False
    primary['no_cf_curriculum_nominal_delta_mean'] = -1
    assert practical_promotion(primary, {'confirmation_passed': True}, 0.1) is False
    assert practical_promotion(primary, {'confirmation_passed': False}, -0.1) is False
    assert practical_promotion(primary, {'confirmation_passed': True}, -0.1) is True
