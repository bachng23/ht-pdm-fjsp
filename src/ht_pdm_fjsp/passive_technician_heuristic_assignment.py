"""Train-only observable assignment supervision on held-out service matrices."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from ht_pdm_fjsp import passive_technician_oracle_representation as base
from ht_pdm_fjsp import passive_technician_td_assignment as td
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveTechnicianEnv
from ht_pdm_fjsp.passive_technician_role_context import ConditionedEdgeQ
from ht_pdm_fjsp.passive_technician_value_decomposition import PassiveValueDecomposition

PROTOCOL = 'ra_qmix_heuristic_assignment_v1'
ARMS = ('monotonic_td', 'monotonic_td_heuristic', 'monotonic_td_oracle')
MAX_MACHINES = 5
FEATURE_CONTRACT = dict(
    local_features=15, identity_slots=5, ordered_peer_slots=4, trunk_features=80,
    padding='zeros; stable ascending other-machine index; ID one-hot padded to five',
    execution='shared masked local argmax; original monotonic queue mixer used only in training',
    telemetry_required=True, cf_weight=0., teacher_training_only=True,
    heuristic_inputs='joint current observations, masks, known fixed configuration only',
    oracle_scope='N=2 training matrices only; no evaluation labels',
    reward_scale=20., auxiliary='weight=1 masked CE on real TD replay batch',
)


def settings(profile):
    full = profile == 'full'
    if profile not in ('smoke', 'full'):
        raise ValueError(profile)
    cfg = dict(train_seeds=list(range(99000, 99010)) if full else [99400],
               evaluation_seeds=list(range(99200, 99250)) if full else [99410, 99411, 99412],
               development_seeds=list(range(99100, 99120)) if full else [99420],
               env_steps=60000 if full else 120, learning_starts=512 if full else 16,
               train_frequency=4, batch_size=128 if full else 32,
               replay_capacity=50000 if full else 120, target_interval=500 if full else 10,
               log_interval=100 if full else 5, hidden=64 if full else 16,
               mixer_hidden=64 if full else 16)
    cfg['updates'] = cfg['env_steps']//4 - (cfg['learning_starts']-1)//4
    return cfg


def cells(profile, machines):
    config = PassiveConfig(machines=machines, technicians=2,
                           horizon=(6 if machines == 2 else 12) if profile == 'full' else 3,
                           failure_age=(3 if machines == 2 else 4) if profile == 'full' else 2, failure_probability=.55,
                           max_age=8, service_time=tuple((2, 3) for _ in range(machines)))
    matrices = {
        'train_a': tuple((2+i%2, 3+(i+1)%2) for i in range(machines)),
        'train_b': tuple((3+(i+1)%2, 2+i%2) for i in range(machines)),
        'development': tuple((2+i%2, 2+(i+1)%2) for i in range(machines)),
        'nominal': tuple((1+i%2, 2-i%2) for i in range(machines)),
        'medium': tuple((2, 3) for _ in range(machines)),
        'high': tuple((3, 4) for _ in range(machines)),
    }
    return {name: replace(config, service_time=matrix) for name, matrix in matrices.items()}


def arms_for(machines):
    return ARMS if machines == 2 else ARMS[:2]


def seed_audit(cfg):
    path = Path(__file__).resolve().parents[2]/'configs/ra_qmix_heuristic_assignment_seed_registry.json'
    registry = json.loads(path.read_text())
    panels = [set(cfg[k]) for k in ('train_seeds', 'development_seeds', 'evaluation_seeds')]
    panels.extend({seed+offset for seed in cfg['train_seeds']} for offset in (1000, 2000, 3000, 4000))
    used = set().union(*panels)
    panels.append(set(range(201, 301)))
    if any(a & b for i, a in enumerate(panels) for b in panels[i+1:]):
        raise ValueError('seed panel overlap')
    if used & set(registry['declared_seed_values']):
        raise ValueError('prior declared seed overlap')
    return dict(panels_disjoint=True, prior_panels_disjoint=True, sealed_panel_closed=True,
                registry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                training_rng_offsets=dict(env_a=1000, env_b=2000, exploration=3000, replay=4000),
                historical_scope=registry['scope'], manifest_count=registry['manifest_count'])


def conditioned_inputs(local):
    if local.ndim != 3 or not 2 <= local.shape[1] <= MAX_MACHINES:
        raise ValueError('two to five machines required')
    batch, machines, features = local.shape
    if features != 15:
        raise ValueError('locked two-technician observation')
    identity = torch.eye(MAX_MACHINES, device=local.device, dtype=local.dtype)[:machines]
    peer = local.new_zeros((batch, machines, MAX_MACHINES-1, features))
    for i in range(machines):
        peer[:, i, :machines-1] = local[:, [j for j in range(machines) if j != i]]
    return torch.cat((local, identity.expand(batch, -1, -1), peer.flatten(2)), dim=-1)


class AssignmentModel(PassiveValueDecomposition):
    def __init__(self, config, cfg):
        super().__init__(config, 'tqmix_no_cf', cfg['hidden'], cfg['mixer_hidden'])
        self.agent = ConditionedEdgeQ(config, cfg['hidden'])
        self.agent.trunk[0] = nn.Linear(FEATURE_CONTRACT['trunk_features'], cfg['hidden'])

    def agent_q(self, local):
        inputs = conditioned_inputs(local)
        b, n, d = inputs.shape
        return self.agent(inputs.reshape(b*n, d)).view(b, n, self.action_dim)


def model_for(config, cfg, seed):
    torch.manual_seed(seed)
    return AssignmentModel(config, cfg)


def expected_proxy_cost(age, failed, remaining, config, start=None):
    """One maintenance start, expected hazard cost; no simulator/future sampling."""
    probability_failed = float(failed)
    cost = 0.
    for tick in range(remaining):
        cost += base.GAMMA**tick * probability_failed * config.downtime_cost
        if start == tick:
            age = 0
            probability_failed = 0.
            cost += base.GAMMA**tick * config.maintenance_cost
        else:
            age = min(config.max_age, age+1)
            hazard = config.failure_probability if age >= config.failure_age else 0.
            new_failure = (1-probability_failed)*hazard
            cost += base.GAMMA**tick * new_failure * config.failure_cost
            probability_failed += new_failure
    return cost


def heuristic_action(observations, masks, config):
    """Deterministic collision-free matching from visible current telemetry."""
    obs = np.asarray(observations)
    if obs.shape != (config.machines, config.observation_dim):
        raise ValueError('observation shape')
    time = int(obs[0, 0])
    window = min(3, config.horizon-time)
    scores = np.full((config.machines, config.technicians+1), -np.inf)
    scores[:, 0] = 0.
    for m in range(config.machines):
        defer = expected_proxy_cost(int(obs[m, 1]), bool(obs[m, 2]), window, config)
        for technician in range(config.technicians):
            if not masks[m][technician+1]:
                continue
            slot = 3+6*technician
            # All queued identities/positions/service times are visible in peers.
            delay = int(obs[m, slot+1]) + sum(int(row[slot+5]) for row in obs if row[slot+4] > 0)
            if delay >= window:
                continue
            action_cost = expected_proxy_cost(int(obs[m, 1]), bool(obs[m, 2]), window, config, delay)
            action_cost += config.queue_waiting_cost*sum(base.GAMMA**t for t in range(delay))
            # Busy service after reset can fail again; reflect duration beyond the
            # three-step proxy through a small fixed throughput charge.
            action_cost += .1*max(0, int(obs[m, slot+5])-1)
            scores[m, technician+1] = defer-action_cost
    best, selected = -math.inf, None
    valid = [[a for a, permitted in enumerate(mask) if permitted] for mask in masks]
    for candidate in itertools.product(*valid):
        nonzero = [a for a in candidate if a]
        if len(nonzero) != len(set(nonzero)):
            continue
        score = sum(scores[m, a] for m, a in enumerate(candidate))
        if score > best + 1e-12:
            best, selected = score, candidate
    if selected is None:
        raise RuntimeError('no feasible matching')
    return selected


def make_oracle_tables(configs, output):
    tables = {}
    audits = []
    for name in ('train_a', 'train_b'):
        oracle = base.ExactOracle(configs[name])
        if oracle.bellman_residual() > 1e-9:
            raise RuntimeError('oracle Bellman residual')
        table = {key: node['actions'][int(np.argmax(node['q95']))] for key, node in oracle.nodes.items()}
        rows = [dict(time=key[0], state=asdict(key[1]), actions=action) for key, action in table.items()]
        path = output/f'oracle_{name}_teacher.json'
        base._write_json(path, rows)
        audits.append(dict(config=name, states=len(table), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           bellman_residual=oracle.bellman_residual(), optimal_labels_verified=True))
        tables[name] = table
    return tables, audits


class Replay(td.Replay):
    def add_labeled(self, x, masks, actions, reward, nx, nm, done, labels):
        index = self.position
        super().add(x, masks, actions, reward, nx, nm, done)
        if labels is not None:
            if 'labels' not in self.storage:
                self.storage['labels'] = torch.empty((self.capacity, len(labels)), dtype=torch.long)
            self.storage['labels'][index] = torch.tensor(labels)


def greedy(model, observations, masks, config):
    with torch.no_grad():
        logits = model.agent_q(base._obs_tensor(observations, config)[None])[0]
        return tuple(logits.masked_fill(~torch.tensor(masks), -torch.inf).argmax(-1).tolist())


def reconcile(config, failed_before, reward, info):
    cost = (sum(failed_before)*config.downtime_cost + info['jobs']*config.maintenance_cost
            + info['waiting']*config.queue_waiting_cost + info['failures']*config.failure_cost)
    if abs(-reward-cost) > 1e-8 or info['invalid_requests']:
        raise RuntimeError('cost/feasibility reconciliation failed')
    return cost


def append_csv(path, rows):
    if not rows:
        return
    with path.open('a') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def fit(model, configs, cfg, seed, arm, output, progress, oracle_tables=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    target = copy.deepcopy(model).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    replay = Replay(cfg['replay_capacity'])
    replay_rng = np.random.default_rng(seed+4000)
    exploration = np.random.default_rng(seed+3000)
    envs = {name: PassiveTechnicianEnv(configs[name], seed=seed+1000*(i+1))
            for i, name in enumerate(('train_a', 'train_b'))}
    names = list(envs)
    episode = updates = syncs = labels_seen = 0
    env = envs[names[0]]
    observations = env.reset()
    pending = []
    label_digest = hashlib.sha256()
    disagreement_count = evaluated_teacher = 0
    for step in tqdm(range(1, cfg['env_steps']+1), desc=f'train N={model.machines}/{arm}/{seed}', unit='step', leave=False):
        config, masks = env.config, env.action_masks()
        x = base._obs_tensor(observations, config)
        proposed = greedy(model, observations, masks, config)
        random_flags, uniforms = exploration.random(config.machines), exploration.random(config.machines)
        actions = tuple(int(np.flatnonzero(masks[i])[min(int(uniforms[i]*sum(masks[i])), sum(masks[i])-1)])
                        if random_flags[i] < td.epsilon(step, cfg) else proposed[i] for i in range(config.machines))
        labels = None
        if arm == 'monotonic_td_heuristic':
            labels = heuristic_action(observations, masks, config)
        elif arm == 'monotonic_td_oracle':
            if config.machines != 2 or oracle_tables is None:
                raise RuntimeError('oracle teacher restricted to N=2')
            key = (env.time, base.canonical(env.time, env.state))
            labels = oracle_tables[names[episode%2]][key]
        if labels is not None:
            if not all(masks[i][a] for i, a in enumerate(labels)):
                raise RuntimeError('invalid teacher action')
            if arm.endswith('heuristic') and len([a for a in labels if a]) != len(set(a for a in labels if a)):
                raise RuntimeError('teacher collision')
            evaluated_teacher += 1
            disagreement_count += proposed != labels
            label_digest.update(np.asarray(labels, dtype=np.int64).tobytes())
            labels_seen += 1
        failed_before = env.state.failed
        following, reward, done, info = env.step(actions)
        reconcile(config, failed_before, reward, info)
        replay.add_labeled(x, masks, actions, reward, base._obs_tensor(following, config), env.action_masks(), done, labels)
        observations = following
        if done:
            episode += 1
            pending.append(dict(machines=config.machines, algorithm=arm, train_seed=seed,
                                episode=episode, config=names[(episode-1)%2], env_steps=step, **env.metrics))
            env = envs[names[episode%2]]
            observations = env.reset()
        if step < cfg['learning_starts'] or step%cfg['train_frequency']:
            continue
        batch = replay.sample(cfg['batch_size'], replay_rng)
        td_loss = (model.total_q(batch['x'], batch['actions']) - td.td_targets(model, target, batch)).square().mean()
        ce = td_loss.new_zeros(())
        if arm != 'monotonic_td':
            logits = model.agent_q(batch['x']).masked_fill(~batch['masks'], -torch.inf)
            labels = batch['labels']
            if not batch['masks'].gather(-1, labels.unsqueeze(-1)).all():
                raise RuntimeError('masked replay label')
            ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss = td_loss+ce
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        if updates%cfg['target_interval'] == 0:
            target.load_state_dict(model.state_dict())
            syncs += 1
        if updates%cfg['log_interval'] == 0 or updates == cfg['updates']:
            progress.append(dict(machines=config.machines, algorithm=arm, train_seed=seed, update=updates,
                                 env_steps=step, td_loss=float(td_loss.detach()), assignment_ce=float(ce.detach()),
                                 loss=float(loss.detach()), gradient_norm=float(norm), target_syncs=syncs,
                                 epsilon=td.epsilon(step, cfg), replay_size=len(replay)))
            base._write_csv(output/'training_progress.csv', progress)
            append_csv(output/'training_episodes.csv', pending)
            pending.clear()
    append_csv(output/'training_episodes.csv', pending)
    if updates != cfg['updates']:
        raise RuntimeError('optimizer budget mismatch')
    return model.eval(), dict(env_steps=cfg['env_steps'], optimizer_updates=updates, target_syncs=syncs,
                              training_episodes=episode, teacher_labels_seen=labels_seen,
                              rng_seeds=dict(initialization=seed, env_a=seed+1000, env_b=seed+2000, exploration=seed+3000, replay=seed+4000),
                              teacher_label_sha256=label_digest.hexdigest(),
                              teacher_greedy_disagreement=disagreement_count/max(1, evaluated_teacher),
                              evaluation_config_training_transitions=0)


def failure_grid(config, seed):
    rng = random.Random(seed)
    return tuple(tuple(rng.random() < config.failure_probability for _ in range(config.machines))
                 for _ in range(config.horizon))


def rollout(config, seed, model=None):
    """Policy-independent failure uniforms give paired exogenous noise."""
    env = PassiveTechnicianEnv(config)
    env.reset()
    components = dict(maintenance_cost=0., queue_waiting_cost=0., failure_cost=0., downtime_cost=0.)
    for events in failure_grid(config, seed):
        obs, masks = env.observations(), env.action_masks()
        actions = heuristic_action(obs, masks, config) if model is None else greedy(model, obs, masks, config)
        if not all(masks[i][a] for i, a in enumerate(actions)):
            raise RuntimeError('invalid deployed action')
        before = env.state.failed
        state, cost, info = env.transition(env.state, actions, env.time, events)
        reconcile(config, before, -cost, info)
        components['maintenance_cost'] += info['jobs']*config.maintenance_cost
        components['queue_waiting_cost'] += info['waiting']*config.queue_waiting_cost
        components['failure_cost'] += info['failures']*config.failure_cost
        components['downtime_cost'] += sum(before)*config.downtime_cost
        env.state, env.time = state, env.time+1
        for key, value in info.items():
            env.metrics[key] += value
    error = env.metrics['objective']-sum(components.values())
    if abs(error) > 1e-8:
        raise RuntimeError('episode reconciliation failed')
    return dict(eval_seed=seed, **env.metrics, **components, cost_reconciliation_error=error)


def load_checkpoint(path):
    payload = torch.load(path, weights_only=True)
    if payload['protocol_version'] != PROTOCOL or payload['feature_contract'] != FEATURE_CONTRACT:
        raise ValueError('checkpoint contract mismatch')
    config = PassiveConfig(**payload['config'])
    model = model_for(config, payload['settings'], payload['train_seed'])
    model.load_state_dict(payload['state_dict'])
    return model.eval()


def small_exact_metrics(model, config, cell, seed, arm, oracle=None):
    if config.machines != 2:
        raise ValueError('exact evaluation restricted to N=2')
    oracle = base.ExactOracle(config) if oracle is None else oracle
    if oracle.bellman_residual() > 1e-9:
        raise RuntimeError('evaluation oracle residual')
    indices = {}
    for key, node in oracle.nodes.items():
        action = greedy(model, node['observations'], node['masks'], config)
        indices[key] = node['actions'].index(action)
    return dict(machines=2, cell=cell, train_seed=seed, algorithm=arm,
                policy_cost=oracle.policy_cost(indices, 1.), oracle_cost=-max(oracle.nodes[oracle.initial]['q1']),
                discounted_policy_cost=oracle.policy_cost(indices, base.GAMMA),
                discounted_oracle_cost=-max(oracle.nodes[oracle.initial]['q95']))


def monotonic_audit(model, config):
    observations = PassiveTechnicianEnv(config).observations()
    local = base._obs_tensor(observations, config)
    actions = torch.tensor(list(itertools.product(range(3), repeat=config.machines)))
    x = local[None].expand(len(actions), -1, -1)
    with torch.no_grad():
        values = model.total_q(x, actions)
        selected = torch.tensor(greedy(model, observations, tuple((True,)*3 for _ in range(config.machines)), config))[None]
        deployed = model.total_q(local[None], selected)[0]
    if float(values.max()-deployed) > 1e-4:
        raise RuntimeError('monotonic greedy audit failed')


def summarize(episodes, cfg, full):
    means = {}
    for n in range(2, 6):
        for arm in arms_for(n):
            for seed in cfg['train_seeds']:
                for cell in ('nominal', 'medium', 'high'):
                    values = [r['objective'] for r in episodes if
                              (r['machines'], r['algorithm'], r['train_seed'], r['cell']) == (n, arm, seed, cell)]
                    if len(values) != len(cfg['evaluation_seeds']):
                        raise RuntimeError('evaluation panel incomplete')
                    means[n, arm, seed, cell] = statistics.fmean(values)
    paired = []
    for seed in cfg['train_seeds']:
        baseline = statistics.fmean(means[n, ARMS[0], seed, c] for n in (3, 4, 5) for c in ('medium', 'high'))
        candidate = statistics.fmean(means[n, ARMS[1], seed, c] for n in (3, 4, 5) for c in ('medium', 'high'))
        paired.append(dict(train_seed=seed, baseline_cost=baseline, heuristic_cost=candidate, cost_delta=candidate-baseline))
    delta = statistics.fmean(r['cost_delta'] for r in paired)
    baseline = statistics.fmean(r['baseline_cost'] for r in paired)
    nominal = {a: statistics.fmean(means[n, a, s, 'nominal'] for n in (3, 4, 5) for s in cfg['train_seeds']) for a in ARMS[:2]}
    guard = nominal[ARMS[1]]/nominal[ARMS[0]]-1 if nominal[ARMS[0]] else (0. if nominal[ARMS[1]] == 0 else math.inf)
    relative = -delta/baseline if baseline else 0.
    radius = 2.262157163*statistics.stdev(r['cost_delta'] for r in paired)/math.sqrt(10) if full else None
    wins = sum(r['cost_delta'] < -1e-8 for r in paired)
    return paired, dict(primary=dict(mean_cost_delta=delta, baseline_cost=baseline, relative_reduction=relative,
                                    ci95=[delta-radius, delta+radius] if full else None, improving_seeds=wins,
                                    nominal_relative_increase=guard,
                                    passed=(relative >= .05 and wins >= 8 and delta+radius < 0 and guard <= .10) if full else None),
                        scientific_gate_applicable=full, inference_unit='paired training seed, N=3..5 diagnostic mean',
                        small_oracle_control='secondary privileged comparator; no positive-control pass gate')


def run(args):
    if args.device != 'cpu':
        raise ValueError('CPU-only protocol')
    torch.set_num_threads(1)
    # Freeze provenance before spending training budget. The process keeps its
    # imported code even if another session later changes the checkout.
    source_bytes = Path(__file__).read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    cfg = settings(args.profile)
    seed_record = seed_audit(cfg)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'source_snapshot.py').write_bytes(source_bytes)
    revision, dirty = base._git_state()
    expected = 9*len(cfg['train_seeds'])
    manifest = dict(protocol_version=PROTOCOL, status='RUNNING', profile=args.profile, device='cpu',
                    git_revision=revision, git_dirty=dirty, source_sha256=source_sha256, source_snapshot='source_snapshot.py', started_at=datetime.now(UTC).isoformat(),
                    runtime=base._runtime_metadata(), train_seeds=cfg['train_seeds'],
                    evaluation_seeds=cfg['evaluation_seeds'], development_seeds=cfg['development_seeds'],
                    sealed_test_evaluated=False, expected_models=expected,
                    expected_training_env_steps=expected*cfg['env_steps'], expected_optimizer_updates=expected*cfg['updates'],
                    expected_test_episodes=expected*3*len(cfg['evaluation_seeds']),
                    expected_development_episodes=expected*len(cfg['development_seeds']))
    all_configs = {n: cells(args.profile, n) for n in range(2, 6)}
    base._write_json(output/'manifest.json', manifest)
    base._write_json(output/'seed_audit.json', seed_record)
    base._write_json(output/'feature_contract.json', FEATURE_CONTRACT)
    base._write_json(output/'resolved_config.json', dict(settings=cfg, gamma=.95, reward_scale=20., assignment_weight=1.,
                       learning_rate=3e-4, heuristic_window=3, throughput_charge=.1,
                       cells={str(n): {c: asdict(v) for c, v in configs.items()} for n, configs in all_configs.items()}))
    episodes, development, references, progress, counts, coverage, exact, teacher_audits = [], [], [], [], [], [], [], []
    actual = 0
    try:
        for n, configs in all_configs.items():
            directory = output/f'machines_{n}'
            directory.mkdir()
            tables, oracle_audit = make_oracle_tables(configs, directory) if n == 2 else (None, [])
            teacher_audits.extend(oracle_audit)
            evaluation_oracles = {c: base.ExactOracle(configs[c]) for c in ('development', 'nominal', 'medium', 'high')} if n == 2 else {}
            # Reference costs cannot tune this locked heuristic/protocol.
            for cell in ('development', 'nominal', 'medium', 'high'):
                panel = cfg['development_seeds'] if cell == 'development' else cfg['evaluation_seeds']
                for ev in tqdm(panel, desc=f'heuristic reference N={n}/{cell}', unit='episode', leave=False):
                    references.append(dict(machines=n, cell=cell, **rollout(configs[cell], ev)))
            base._write_csv(output/'reference_episodes.csv', references)
            for seed in tqdm(cfg['train_seeds'], desc=f'assignment N={n}', unit='seed'):
                initial = model_for(configs['train_a'], cfg, seed).state_dict()
                for arm in arms_for(n):
                    model = model_for(configs['train_a'], cfg, seed)
                    if not all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items()):
                        raise RuntimeError('initialization mismatch')
                    counts.append(dict(machines=n, train_seed=seed, algorithm=arm, parameters=sum(p.numel() for p in model.parameters())))
                    base._write_csv(output/'parameter_counts.csv', counts)
                    model, record = fit(model, configs, cfg, seed, arm, output, progress,
                                        tables if arm == ARMS[2] else None)
                    record.update(machines=n, algorithm=arm, train_seed=seed)
                    coverage.append(record)
                    path = directory/arm/f'train_seed_{seed}'
                    path.mkdir(parents=True)
                    base._write_json(path/'training_coverage.json', record)
                    checkpoint = path/'model.pt'
                    torch.save(dict(protocol_version=PROTOCOL, state_dict=model.state_dict(), config=asdict(configs['train_a']),
                                    settings=cfg, algorithm=arm, train_seed=seed, feature_contract=FEATURE_CONTRACT,
                                    teacher_definition_sha256=source_sha256), checkpoint)
                    restored = load_checkpoint(checkpoint)
                    for cell in ('train_a', 'train_b', 'development', 'nominal', 'medium', 'high'):
                        env = PassiveTechnicianEnv(configs[cell])
                        # Exercise reload and greedy monotonicity on visited states.
                        for _ in range(env.config.horizon):
                            x = base._obs_tensor(env.observations(), env.config)[None]
                            with torch.no_grad():
                                if not torch.equal(model.agent_q(x), restored.agent_q(x)):
                                    raise RuntimeError('checkpoint logits mismatch')
                                if not torch.equal(model.total_q(x, torch.zeros((1, n), dtype=torch.long)),
                                                   restored.total_q(x, torch.zeros((1, n), dtype=torch.long))):
                                    raise RuntimeError('checkpoint mixer mismatch')
                            env.step(greedy(restored, env.observations(), env.action_masks(), env.config))
                        monotonic_audit(restored, configs[cell])
                    for cell in ('development', 'nominal', 'medium', 'high'):
                        panel = cfg['development_seeds'] if cell == 'development' else cfg['evaluation_seeds']
                        target_rows = development if cell == 'development' else episodes
                        for ev in tqdm(panel, desc=f'evaluate N={n}/{arm}/{seed}/{cell}', unit='episode', leave=False):
                            target_rows.append(dict(machines=n, cell=cell, algorithm=arm, train_seed=seed,
                                                    **rollout(configs[cell], ev, restored)))
                        if n == 2:
                            exact.append(small_exact_metrics(restored, configs[cell], cell, seed, arm, evaluation_oracles[cell]))
                    base._write_csv(output/'episodes.partial.csv', episodes)
                    base._write_csv(output/'development_episodes.csv', development)
                    base._write_csv(output/'small_exact_metrics.csv', exact)
                    actual += 1
        paired, summary = summarize(episodes, cfg, args.profile == 'full')
        audits = dict(complete_models=actual == expected,
                      complete_test_episodes=len(episodes) == manifest['expected_test_episodes'],
                      complete_development_episodes=len(development) == manifest['expected_development_episodes'],
                      complete_reference_episodes=len(references) == 4*(3*len(cfg['evaluation_seeds'])+len(cfg['development_seeds'])),
                      per_model_budget=all(r['env_steps'] == cfg['env_steps'] and r['optimizer_updates'] == cfg['updates'] for r in coverage),
                      target_cadence=all(r['target_syncs'] == cfg['updates']//cfg['target_interval'] for r in coverage),
                      zero_invalid_requests=all(r['invalid_requests'] == 0 for r in episodes+development+references),
                      finite_logs=all(math.isfinite(r[k]) for r in progress for k in ('loss', 'td_loss', 'assignment_ce', 'gradient_norm')),
                      cost_reconciled=all(abs(r['cost_reconciliation_error']) < 1e-8 for r in episodes+development+references),
                      unique_evaluation_keys=len({(r['machines'], r['algorithm'], r['train_seed'], r['cell'], r['eval_seed']) for r in episodes+development}) == len(episodes+development),
                      labels_scope=all(r['teacher_labels_seen'] == (0 if r['algorithm'] == ARMS[0] else cfg['env_steps']) for r in coverage),
                      no_evaluation_config_training=all(r['evaluation_config_training_transitions'] == 0 for r in coverage),
                      checkpoint_reload_identical=True, initialization_matched=True, monotonic_greedy_verified=True,
                      heuristic_mask_and_matching_verified=True, training_feasibility_and_cost_verified=True,
                      seed_panels_disjoint=True, sealed_panel_closed=True)
        if not all(audits.values()):
            raise RuntimeError(f'engineering audits failed: {audits}')
        summary['audits'] = audits
        summary['development_headroom'] = [dict(machines=n, algorithm=a,
                mean_cost=statistics.fmean(r['objective'] for r in development if r['machines'] == n and r['algorithm'] == a),
                heuristic_cost=statistics.fmean(r['objective'] for r in references if r['machines'] == n and r['cell'] == 'development'),
                interpretation='teacher is achievable comparator, not certified floor; gates unchanged')
                for n in range(2, 6) for a in arms_for(n)]
        base._write_json(output/'teacher_audit.json', dict(oracle=teacher_audits, heuristic_definition_sha256=source_sha256,
                                                          oracle_access_in_heuristic=False, heuristic_future_events_access=False))
        base._write_csv(output/'paired_seed_metrics.csv', paired)
        base._write_csv(output/'episodes.csv', episodes)
        base._write_csv(output/'coordination.csv', episodes+development)
        base._write_json(output/'summary.json', summary)
        manifest.update(status='COMPLETED', completed_at=datetime.now(UTC).isoformat(), actual_models=actual,
                        actual_test_episodes=len(episodes), actual_development_episodes=len(development),
                        actual_reference_episodes=len(references), actual_training_env_steps=sum(r['env_steps'] for r in coverage),
                        actual_optimizer_updates=sum(r['optimizer_updates'] for r in coverage),
                        actual_training_episodes=sum(r['training_episodes'] for r in coverage), audits=audits)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        manifest.update(status='FAILED', error=f'{type(error).__name__}: {error}', actual_models=actual)
        raise
    finally:
        manifest['outputs'] = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
        base._write_json(output/'manifest.json', manifest)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=('smoke', 'full'))
    parser.add_argument('--device', default='cpu', choices=('cpu',))
    parser.add_argument('--output-dir', required=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
