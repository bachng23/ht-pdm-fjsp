"""Fresh-seed, parameter-identical CF-by-adaptation factorial experiment."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import replace, asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_artifact_diagnostic import decision_trace
from ht_pdm_fjsp.passive_technician_curriculum_confirmation import (
    REGIMES, REGIME_SPECS, profile_settings as base_profile_settings,
    run as run_confirmation, _paired_t_interval,
)
from ht_pdm_fjsp.passive_technician_leave_one_out import (
    _git_state, _runtime_metadata, _write_json, _write_csv, environment_cells,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import PassiveValueDecomposition

PROTOCOL_VERSION = 'ra_qmix_cf_adaptation_factorial_v1'
ALGORITHMS = ('tqmix', 'tqmix_no_cf')
FULL_TRAIN_SEEDS = tuple(range(85000, 85010))
FULL_EVALUATION_SEEDS = tuple(range(85100, 85150))
SMOKE_TRAIN_SEEDS = (85400,)
SMOKE_EVALUATION_SEEDS = (85410, 85411, 85412)


def profile_settings(profile):
    base = base_profile_settings(profile)
    seeds, evaluations = ((FULL_TRAIN_SEEDS, FULL_EVALUATION_SEEDS) if profile == 'full'
                         else (SMOKE_TRAIN_SEEDS, SMOKE_EVALUATION_SEEDS))
    return (seeds, evaluations, *base[2:])


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def interaction_results(arm_summaries, gate_applicable):
    """Compare adaptation effects within seeds, then between architecture arms."""
    lookup = {}
    for algorithm, rows in arm_summaries.items():
        for row in rows:
            lookup[algorithm, row['training_regime'], int(row['train_seed']),
                   row['scenario'], int(row['step_budget'])] = float(row['objective_mean'])
    seeds = sorted({k[2] for k in lookup})
    budget = max(k[4] for k in lookup)
    rows = []
    for seed in seeds:
        costs = {(a, r): lookup[a, r, seed, 'in_distribution', budget]
                 for a in ALGORITHMS for r in REGIMES}
        gaps = {a: costs[a, 'curriculum_replay'] - costs[a, 'nominal_100'] for a in ALGORITHMS}
        ratios = {a: costs[a, 'curriculum_replay'] / costs[a, 'nominal_100'] - 1 for a in ALGORITHMS}
        rows.append(dict(train_seed=seed, step_budget=budget,
            full_nominal=costs['tqmix', 'nominal_100'], full_curriculum=costs['tqmix', 'curriculum_replay'],
            no_cf_nominal=costs['tqmix_no_cf', 'nominal_100'], no_cf_curriculum=costs['tqmix_no_cf', 'curriculum_replay'],
            full_nominal_gap=gaps['tqmix'], no_cf_nominal_gap=gaps['tqmix_no_cf'],
            absolute_interaction=gaps['tqmix_no_cf'] - gaps['tqmix'],
            relative_interaction=ratios['tqmix_no_cf'] - ratios['tqmix'],
            no_cf_curriculum_minus_full=costs['tqmix_no_cf', 'curriculum_replay'] - costs['tqmix', 'curriculum_replay']))
    values = [r['absolute_interaction'] for r in rows]
    ci = _paired_t_interval(values)
    mean = statistics.fmean(values)
    wins = sum(v < 0 for v in values)
    return rows, dict(primary_budget=budget, interaction_mean=mean,
        interaction_sd=statistics.stdev(values) if len(values) > 1 else None,
        interaction_ci95=list(ci), improving_seed_count=wins,
        primary_hypothesis_passed=(mean < 0 and wins >= 8) if gate_applicable else None,
        no_cf_curriculum_nominal_delta_mean=statistics.fmean(r['no_cf_curriculum_minus_full'] for r in rows),
        relative_interaction_mean=statistics.fmean(r['relative_interaction'] for r in rows),
        gate_applicable=gate_applicable)


def practical_promotion(primary, no_cf_gate, stress_delta):
    if not primary['gate_applicable']:
        return None
    return bool(primary['primary_hypothesis_passed'] and no_cf_gate['confirmation_passed']
                and primary['no_cf_curriculum_nominal_delta_mean'] <= 0 and stress_delta <= 0)


def prior_seed_registry():
    path = Path(__file__).resolve().parents[2] / 'configs' / 'ra_qmix_cf_seed_registry.json'
    registry = json.loads(path.read_text())
    panels = [set(FULL_TRAIN_SEEDS), set(FULL_EVALUATION_SEEDS),
              set(SMOKE_TRAIN_SEEDS), set(SMOKE_EVALUATION_SEEDS), set(range(201, 301))]
    if any(a & b for i, a in enumerate(panels) for b in panels[i + 1:]):
        raise ValueError('Factorial panels overlap')
    previous = frozenset(registry['declared_seed_values'])
    if set().union(*panels[:4]) & previous:
        raise ValueError('Fresh factorial panels overlap frozen prior registry')
    return registry, previous


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Choose a new timestamped artifact directory')
    registry, previous = prior_seed_registry()
    profile = profile_settings(args.profile)
    seeds, evaluations, budgets, boundary, settings, nominal_batch = profile
    output.mkdir(parents=True)
    revision, dirty = _git_state()
    manifest = dict(protocol_version=PROTOCOL_VERSION, status='RUNNING', profile=args.profile,
        git_revision=revision, git_dirty=dirty, runtime=_runtime_metadata(), device='cpu',
        algorithms=ALGORITHMS, regimes=REGIMES, regime_specs=REGIME_SPECS,
        train_seeds=seeds, evaluation_seeds=evaluations, step_budgets=budgets,
        sealed_test_seeds=list(range(201, 301)), sealed_test_evaluated=False,
        primary_endpoint='paired absolute nominal adaptation interaction: no-CF minus Full at final steps',
        stopping_rule='fixed steps; abort errors/nonfinite/audit failure; no automatic retry',
        gradient_sample_step_interval=1000 if args.profile == 'full' else 24,
        expected_trajectories=4 * len(seeds), expected_checkpoints=12 * len(seeds),
        expected_evaluation_rows=48 * len(seeds) * len(evaluations),
        started_at=datetime.now(UTC).isoformat())
    _write_json(output / 'manifest.json', manifest)
    _write_json(output / 'seed_audit.json', registry)
    _write_json(output / 'resolved_config.json', dict(algorithms=ALGORITHMS, regimes=REGIME_SPECS,
        base_settings=asdict(settings), no_cf_lambda=0.0, phase_boundary_steps=boundary, nominal_batch_size=nominal_batch,
        train_seeds=seeds, evaluation_seeds=evaluations, step_budgets=budgets,
        environments={name: asdict(config) for name, config in environment_cells().items()}))
    try:
        torch.set_num_threads(1)
        cells = environment_cells()
        initial_identical = True
        for seed in seeds:
            models = []
            for algorithm in ALGORITHMS:
                torch.manual_seed(seed)
                models.append(PassiveValueDecomposition(cells['in_distribution'], algorithm,
                    settings.hidden_dim, settings.mixer_hidden_dim))
            initial_identical &= all(torch.equal(v, models[1].state_dict()[k]) for k, v in models[0].state_dict().items())
        if not initial_identical:
            raise RuntimeError('Full and no-CF initial parameters differ')
        arm_summaries, arm_decisions, arm_audits = {}, {}, {}
        for algorithm in ALGORITHMS:
            arm_settings = replace(settings, lambda_cf=0.05 if algorithm == 'tqmix' else 0.0)
            arm_profile = (seeds, evaluations, budgets, boundary, arm_settings, nominal_batch)
            run_confirmation(argparse.Namespace(profile=args.profile, device='cpu', output_dir=output / algorithm),
                algorithm=algorithm, profile_override=arm_profile,
                protocol_version=PROTOCOL_VERSION + '/' + algorithm,
                prior_used_seeds=previous, gradient_diagnostic_interval=manifest['gradient_sample_step_interval'])
            arm = json.loads((output / algorithm / 'summary.json').read_text())
            arm_summaries[algorithm] = read_csv(output / algorithm / 'budget_summary.csv')
            arm_decisions[algorithm] = arm['confirmation_result']
            arm_audits[algorithm] = arm['audits']
        interactions, primary = interaction_results(arm_summaries, args.profile == 'full')
        _write_csv(output / 'primary_interactions.csv', interactions)
        _write_json(output / 'algorithm_gates.json', arm_decisions)
        stress_curriculum = {}
        for algorithm in ALGORITHMS:
            rows = read_csv(output / algorithm / 'regret_scores.csv')
            stress_curriculum[algorithm] = statistics.fmean(float(r['stress_max_relative_regret']) for r in rows
                if r['row_type'] == 'training_seed' and r['training_regime'] == 'curriculum_replay'
                and int(r['step_budget']) == max(budgets))
        stress_delta = stress_curriculum['tqmix_no_cf'] - stress_curriculum['tqmix']
        promotion = practical_promotion(primary, arm_decisions['tqmix_no_cf'], stress_delta)
        # Record deterministic checkpoint decisions; reconcile requests with evaluation CSV.
        expected_requests = {}
        for algorithm in ALGORITHMS:
            for row in read_csv(output / algorithm / 'episodes.csv'):
                expected_requests[algorithm, row['training_regime'], int(row['train_seed']), row['scenario'],
                                  int(row['budget']), int(row['eval_seed'])] = row
        decision_count = 0
        with (output / 'decisions.csv').open('w', newline='') as handle:
            writer = None
            for algorithm in ALGORITHMS:
                tasks = [(regime, seed, step) for regime in REGIMES for seed in seeds for step in budgets]
                for regime, seed, step in tqdm(tasks, desc=f'decision replay/{algorithm}', unit='checkpoint'):
                    path = output / algorithm / regime / f'train_seed_{seed}' / 'checkpoints' / f'step_{step}' / 'model.pt'
                    model = PassiveValueDecomposition.load(path, cells['in_distribution'], torch.device('cpu'))
                    for scenario, config in cells.items():
                        for evaluation in evaluations:
                            trace, metrics = decision_trace(config, model, evaluation, seed, regime, scenario, step, return_metrics=True)
                            expected = expected_requests[algorithm, regime, seed, scenario, step, evaluation]
                            mapping = {'failures': 'failure_events', 'jobs': 'service_starts', 'waiting': 'queue_waiting_steps'}
                            if any(abs(float(value) - float(expected[mapping.get(name, name)])) > 1e-6 for name, value in metrics.items()):
                                raise RuntimeError('Trace episode metrics disagree with primary evaluation')
                            if sum(int(r['action'] > 0) for r in trace) != int(expected['request_count']):
                                raise RuntimeError('Trace requests disagree with primary evaluation')
                            records = [dict(algorithm=algorithm, **r) for r in trace]
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                                writer.writeheader()
                            writer.writerows(records)
                            decision_count += len(records)
        totals = []
        gradient_rows = []
        for algorithm in ALGORITHMS:
            cumulative = {}
            with (output / algorithm / 'training_progress.csv').open(newline='') as handle:
                for r in csv.DictReader(handle):
                    cumulative[r['training_regime'], int(r['train_seed'])] = int(r['cumulative_optimizer_updates'])
            totals.extend(dict(algorithm=algorithm, regime=k[0], seed=k[1], optimizer_updates=v) for k, v in cumulative.items())
            gradient_rows.extend({**r, 'algorithm': algorithm} for r in read_csv(output / algorithm / 'gradient_diagnostics.csv'))
        _write_csv(output / 'optimizer_update_totals.csv', totals)
        parameter_rows = {a: read_csv(output / a / 'parameter_counts.csv') for a in ALGORITHMS}
        schedules = {a: read_csv(output / a / 'training_schedule.csv') for a in ALGORITHMS}
        audits = dict(initial_parameter_tensors_identical=initial_identical,
            complete_factorial_evaluations=len(expected_requests) == manifest['expected_evaluation_rows'],
            complete_factorial_checkpoints=sum(a['actual_checkpoints'] for a in arm_audits.values()) == manifest['expected_checkpoints'],
            equal_parameter_counts=len({int(r['total_parameters']) for rs in parameter_rows.values() for r in rs}) == 1,
            matched_updates_across_four_arms=len({r['optimizer_updates'] for r in totals}) == 1,
            matched_schedules_across_algorithms=schedules['tqmix'] == schedules['tqmix_no_cf'],
            full_gradients_sampled=any(float(r['weighted_cf_gradient_norm']) > 0 for r in gradient_rows if r['algorithm'] == 'tqmix'),
            no_cf_gradients_zero=all(float(r['weighted_cf_gradient_norm']) == 0 and not r['gradient_cosine']
                                    for r in gradient_rows if r['algorithm'] == 'tqmix_no_cf'),
            trace_row_count=decision_count,
            expected_trace_rows=2 * 2 * len(seeds) * len(budgets) * len(evaluations) * sum(c.horizon * c.machines for c in cells.values()),
            prior_seed_panels_disjoint=True, sealed_panel_closed=True)
        audits['trace_coverage_complete'] = audits['trace_row_count'] == audits['expected_trace_rows']
        if not all(v for v in audits.values() if isinstance(v, bool)):
            raise RuntimeError(f'Factorial audit failed: {audits}')
        summary = dict(protocol_version=PROTOCOL_VERSION, primary=primary,
            no_cf_curriculum_stress_regret_minus_full=stress_delta,
            stress_adaptation_interaction=(arm_decisions['tqmix_no_cf']['criteria']['stress_delta_mean']
                - arm_decisions['tqmix']['criteria']['stress_delta_mean']), promotion_passed=promotion,
            algorithm_gates=arm_decisions, arm_audits=arm_audits, audits=audits,
            interpretation='paired training-seed factorial contrast; gradient diagnostics do not establish mediation')
        _write_json(output / 'summary.json', summary)
        manifest.update(status='COMPLETED', completed_at=datetime.now(UTC).isoformat(),
            summary=summary, outputs=sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()))
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        manifest.update(status='FAILED', error=str(error))
        raise
    finally:
        _write_json(output / 'manifest.json', manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('smoke', 'full'), required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
