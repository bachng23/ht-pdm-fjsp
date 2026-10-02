"""Read-only checkpoint and log diagnosis of the locked curriculum confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import _obs_tensor
from ht_pdm_fjsp.passive_technician_leave_one_out import evaluate_diagnostic_episode, environment_cells
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveTechnicianEnv
from ht_pdm_fjsp.passive_technician_value_decomposition import PassiveValueDecomposition

PROTOCOL = 'ra_qmix_artifact_diagnostic_v1'
SOURCE_PROTOCOL = 'ra_qmix_curriculum_confirmation_v1'
SOURCE_SHA = '33ab6196f0dd4d4e12038b74b65fd3c5782096f7'
REGIMES = ('nominal_100', 'curriculum_replay')
STEPS = (240192, 360288, 480384)
METRICS = ('objective', 'failure_cost', 'downtime_cost', 'maintenance_cost',
           'queue_waiting_cost', 'failure_events', 'preventive_starts',
           'corrective_starts', 'queue_waiting_steps', 'request_count')
TOLERANCE = 1e-6


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path.name}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return (row['training_regime'], int(row['train_seed']), row['scenario'],
            int(row['budget']), int(row['eval_seed']))


def source_inputs(source):
    manifest = read_json(source / 'manifest.json')
    if (manifest.get('status') != 'COMPLETED' or
        manifest.get('protocol_version') != SOURCE_PROTOCOL or
        manifest.get('git_revision') != SOURCE_SHA or
        manifest.get('train_seeds') != list(range(84600, 84610)) or
        manifest.get('evaluation_seeds') != list(range(84700, 84750))):
        raise ValueError('Source identity, completion or seed panel differs from locked protocol')
    configs = read_json(source / 'benchmark_config.json')['environment_cells']
    expected_configs = json.loads(json.dumps({name: vars(config) for name, config in environment_cells().items()}))
    resolved = read_json(source / 'resolved_config.json')
    if (configs != expected_configs or resolved.get('step_budgets') != list(STEPS)
        or resolved.get('algorithm') != 'tqmix' or resolved.get('settings') != manifest['value_settings']
        or resolved.get('regime_specs', {}).get('curriculum_replay', {}).get('phase_two_nominal_replay_share') != 0.875):
        raise ValueError('Source configurations differ from locked confirmation')
    rows = read_csv(source / 'episodes.csv')
    index = {key(row): row for row in rows}
    expected = {(regime, seed, scenario, step, evaluation)
                for regime in REGIMES for seed in range(84600, 84610)
                for scenario in configs for step in STEPS for evaluation in range(84700, 84750)}
    if len(configs) != 4 or len(index) != len(rows) or set(index) != expected:
        raise ValueError('Source evaluation coverage is incomplete or duplicated')
    files = [source / name for name in ('manifest.json', 'benchmark_config.json',
             'resolved_config.json', 'episodes.csv', 'training_progress.csv',
             'adaptation_diagnostics.csv', 'training_schedule.csv')]
    files += [source / regime / f'train_seed_{seed}' / 'checkpoints' / f'step_{step}' / 'model.pt'
              for regime in REGIMES for seed in range(84600, 84610) for step in STEPS]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise ValueError(f'Missing source files: {missing[:3]}')
    return manifest, configs, index, files


def decision_trace(config, model, eval_seed, train_seed, regime, scenario, step, *, return_metrics=False):
    env = PassiveTechnicianEnv(config, seed=eval_seed)
    observations = env.reset()
    rows = []
    done = False
    while not done:
        local = _obs_tensor(observations, config).unsqueeze(0)
        masks = torch.tensor(env.action_masks(), dtype=torch.bool)
        with torch.no_grad():
            q = model.agent_q(local)[0]
        if not bool(torch.isfinite(q).all()):
            raise ValueError('Nonfinite checkpoint Q values')
        actions = q.masked_fill(~masks, -torch.inf).argmax(-1).tolist()
        for machine, action in enumerate(actions):
            valid = q[machine][masks[machine]].sort(descending=True).values
            services = [config.service_time[machine][k] for k in range(config.technicians)
                        if masks[machine, k + 1]]
            rows.append(dict(training_regime=regime, train_seed=train_seed, scenario=scenario,
                budget=step, eval_seed=eval_seed, time=env.time, machine=machine,
                age=env.state.ages[machine], failed=int(env.state.failed[machine]),
                action=action, feasible_actions=len(valid),
                q_margin=float(valid[0] - valid[1]) if len(valid) > 1 else None,
                chosen_service_time=config.service_time[machine][action - 1] if action else None,
                min_feasible_service_time=min(services) if services else None,
                selected_queue_length=len(env.state.queues[action - 1]) if action else None,
                selected_busy_remaining=max(0, env.state.busy_until[action - 1] - env.time) if action else None))
        observations, _, done, _ = env.step(actions)
    return (rows, env.metrics) if return_metrics else rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['training_regime'], row['train_seed'], row['scenario'], row['budget'])].append(row)
    seeds = []
    for (regime, seed, scenario, step), group in sorted(groups.items()):
        seeds.append(dict(training_regime=regime, train_seed=seed, scenario=scenario,
                          budget=step, evaluation_count=len(group),
                          **{metric: statistics.fmean(float(r[metric]) for r in group) for metric in METRICS}))
    lookup = {(r['training_regime'], r['train_seed'], r['scenario'], r['budget']): r for r in seeds}
    paired = []
    for row in seeds:
        if row['training_regime'] != 'curriculum_replay':
            continue
        seed, scenario, step = row['train_seed'], row['scenario'], row['budget']
        control = lookup[('nominal_100', seed, scenario, step)]
        boundary_left = lookup[('curriculum_replay', seed, scenario, STEPS[0])]
        boundary_right = lookup[('nominal_100', seed, scenario, STEPS[0])]
        for metric in METRICS:
            delta = row[metric] - control[metric]
            paired.append(dict(train_seed=seed, scenario=scenario, budget=step, metric=metric,
                               delta=delta, delta_from_boundary=delta - (boundary_left[metric] - boundary_right[metric])))
    aggregates = []
    for scenario in sorted({r['scenario'] for r in paired}):
        for step in STEPS:
            for metric in METRICS:
                selected = [r for r in paired if (r['scenario'], r['budget'], r['metric']) == (scenario, step, metric)]
                values = [r['delta'] for r in selected]
                aggregates.append(dict(scenario=scenario, budget=step, metric=metric,
                    training_seed_count=len(values), delta_mean=statistics.fmean(values),
                    delta_sd=statistics.stdev(values) if len(values) > 1 else None,
                    delta_from_boundary_mean=statistics.fmean(r['delta_from_boundary'] for r in selected)))
    return seeds, paired, aggregates


def log_summary(source, selected_seeds):
    groups = {}
    columns = ('td_loss', 'raw_cf_loss', 'weighted_cf_loss', 'total_loss')
    with (source / 'training_progress.csv').open(newline='') as handle:
        for row in tqdm(csv.DictReader(handle), desc='Read training logs', unit='episode'):
            if int(row['train_seed']) not in selected_seeds:
                continue
            step = int(row['environment_steps'])
            interval = next((s for s in STEPS if step <= s), None)
            if interval is None:
                raise ValueError('Training log exceeds locked budget')
            group_key = (row['training_regime'], int(row['train_seed']), row['training_phase'], interval)
            group = groups.setdefault(group_key, dict(count=0, sums={c: 0.0 for c in columns},
                counts={c: 0 for c in columns}, first_epsilon=float(row['epsilon']),
                last_epsilon=0.0, first_updates=int(row['update_count']), last_updates=0))
            group['count'] += 1
            group['last_epsilon'] = float(row['epsilon'])
            group['last_updates'] = int(row['update_count'])
            for column in columns:
                if row.get(column):
                    value = float(row[column])
                    if not math.isfinite(value):
                        raise ValueError(f'Nonfinite logged {column}')
                    group['sums'][column] += value
                    group['counts'][column] += 1
    return [dict(training_regime=k[0], train_seed=k[1], phase=k[2], interval_end=k[3],
        episode_count=g['count'], first_epsilon=g['first_epsilon'], last_epsilon=g['last_epsilon'],
        first_updates=g['first_updates'], last_updates=g['last_updates'],
        **{c + '_episode_mean': g['sums'][c] / g['counts'][c] if g['counts'][c] else None for c in columns},
        historical_gradient_norm=None) for k, g in sorted(groups.items())]


def run(source, output, profile):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be outside source artifacts')
    if output.exists():
        raise FileExistsError('Output directory already exists; choose a new timestamp')
    source_manifest, configs, index, files = source_inputs(source)
    hashes = {str(p.relative_to(source)): digest(p) for p in files}
    output.mkdir(parents=True)
    train_seeds = list(range(84600, 84610)) if profile == 'full' else [84600]
    eval_seeds = list(range(84700, 84750)) if profile == 'full' else [84700, 84701, 84702]
    manifest = dict(protocol_version=PROTOCOL, source_protocol=SOURCE_PROTOCOL,
        source_commit=SOURCE_SHA, source_run=str(source), status='RUNNING', profile=profile,
        train_seeds=train_seeds, evaluation_seeds=eval_seeds, steps=STEPS,
        started_at=datetime.now(UTC).isoformat(), device='cpu', training_performed=False,
        sealed_test_evaluated=False, source_settings=source_manifest['value_settings'],
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        torch_version=torch.__version__, python_version=sys.version,
        expected_replay_episodes=len(REGIMES) * len(train_seeds) * len(STEPS) * len(configs) * len(eval_seeds),
        source_checkpoint_count=len(REGIMES) * len(train_seeds) * len(STEPS),
        git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()))
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'source_hashes.json', hashes)
    try:
        torch.set_num_threads(1)
        for name in ('benchmark_config.json', 'resolved_config.json', 'adaptation_diagnostics.csv', 'training_schedule.csv'):
            shutil.copyfile(source / name, output / name)
        episodes, traces = [], []
        boundary = {}
        tasks = [(regime, seed, step) for regime in REGIMES for seed in train_seeds for step in STEPS]
        for regime, seed, step in tqdm(tasks, desc='Replay checkpoints', unit='checkpoint'):
            path = source / regime / f'train_seed_{seed}' / 'checkpoints' / f'step_{step}' / 'model.pt'
            for scenario, payload in configs.items():
                config = PassiveConfig(**{**payload, 'service_time': tuple(tuple(r) for r in payload['service_time'])})
                model = PassiveValueDecomposition.load(path, config, torch.device('cpu'))
                model.eval()
                if model.algorithm != 'tqmix':
                    raise ValueError('Unexpected checkpoint algorithm')
                if step == STEPS[0] and scenario == 'in_distribution':
                    if regime == REGIMES[0]:
                        boundary[seed] = {k: v.clone() for k, v in model.state_dict().items()}
                    elif not all(torch.equal(v, boundary[seed][k]) for k, v in model.state_dict().items()):
                        raise ValueError('Shared boundary checkpoint mismatch')
                for evaluation in tqdm(eval_seeds, desc=f'{regime}/{seed}/{step}/{scenario}', leave=False, unit='episode'):
                    row = evaluate_diagnostic_episode(config, model, evaluation, seed, scenario, step, torch.device('cpu'))
                    row['training_regime'] = regime
                    original = index[key(row)]
                    for metric, value in row.items():
                        if metric in original and isinstance(value, (int, float)):
                            if not math.isfinite(float(value)) or abs(float(value) - float(original[metric])) > TOLERANCE:
                                raise ValueError(f'Replay mismatch {key(row)}: {metric}')
                    if row['invalid_requests'] or abs(row['cost_reconciliation_error']) > TOLERANCE:
                        raise ValueError('Feasibility/cost audit failed')
                    episodes.append(row)
                    traces.extend(decision_trace(config, model, evaluation, seed, regime, scenario, step))
            write_csv(output / 'episodes.csv', episodes)
        seeds, pairs, aggregates = summarize(episodes)
        write_csv(output / 'decisions.csv', traces)
        write_csv(output / 'seed_summary.csv', seeds)
        write_csv(output / 'paired_deltas.csv', pairs)
        write_csv(output / 'training_log_summary.csv', log_summary(source, train_seeds))
        if any(digest(source / p) != h for p, h in hashes.items()):
            raise ValueError('Source artifacts changed during diagnostic')
        expected = len(tasks) * len(configs) * len(eval_seeds)
        if len(episodes) != expected or len({key(r) for r in episodes}) != expected:
            raise ValueError('Replay coverage audit failed')
        summary = dict(protocol_version=PROTOCOL, audits=dict(expected_rows=expected,
            actual_rows=len(episodes), replay_matches_source=True, source_unchanged=True,
            shared_boundary_equal=True, sealed_panel_closed=True), paired_aggregates=aggregates,
            missing_metrics=['historical gradient norms', 'historical replay sample states'],
            interpretation='Exploratory paired training-seed summaries; no causal or independent episode inference')
        write_json(output / 'summary.json', summary)
        lines = ['# Artifact diagnostic', '', f'Profile: {profile}; {len(train_seeds)} training seeds; {len(episodes)} replay episodes.',
            '', 'Delta = curriculum − nominal control. Positive cost delta is degradation.', '',
            '| Scenario | Steps | Cost delta | Failure | Downtime | Maintenance | Queue |', '|---|---:|---:|---:|---:|---:|---:|']
        lookup = {(r['scenario'], r['budget'], r['metric']): r['delta_mean'] for r in aggregates}
        for scenario in configs:
            for step in STEPS:
                values = [lookup[(scenario, step, m)] for m in METRICS[:5]]
                lines.append(f'| {scenario} | {step} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
        lines += ['', 'Missing historical gradient norms and replay states remain unavailable. Q margins are checkpoint-time diagnostics, not training gradients.',
                  'Source recovery-at-start semantics retained. Associations do not establish causality.',
                  'Smoke results are engineering evidence only; scientific interpretation requires the full panel.']
        (output / 'diagnostic_report.md').write_text('\n'.join(lines) + '\n')
        manifest.update(status='COMPLETED', completed_at=datetime.now(UTC).isoformat(),
                        artifact_files=sorted(p.name for p in output.iterdir()))
    except Exception as error:
        manifest.update(status='FAILED', error=str(error))
        raise
    finally:
        write_json(output / 'manifest.json', manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--profile', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    args = parser.parse_args()
    if args.preflight:
        source_inputs(args.source_run.resolve())
        print('Source preflight passed: locked CSV panel and all 60 checkpoints present')
    elif args.output_dir is None:
        parser.error('--output-dir is required unless --preflight is used')
    else:
        print(run(args.source_run, args.output_dir, args.profile))


if __name__ == '__main__':
    main()
