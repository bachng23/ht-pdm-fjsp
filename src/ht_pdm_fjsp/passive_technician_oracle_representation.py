"""Exact-oracle supervised diagnostic of technician-aware Q factorization."""
from __future__ import annotations

import argparse
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
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import _obs_tensor
from ht_pdm_fjsp.passive_technician_leave_one_out import (
    _git_state, _runtime_metadata, _write_csv, _write_json,
)
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveState, PassiveTechnicianEnv
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition, TechnicianAwareMixer,
)

PROTOCOL = "ra_qmix_oracle_representation_v1"
ARMS = ("monotonic_no_cf", "monotonic_cf", "signed_no_cf", "joint_q")
GAMMA = 0.95
DATASET_SEEDS = (95100, 95101, 95102)
Key = tuple[int, PassiveState]


def cells(profile: str) -> dict[str, PassiveConfig]:
    base = PassiveConfig(machines=2, technicians=2, horizon=6, max_age=5,
                         failure_age=3, service_time=((2, 3), (3, 2)))
    result = {"nominal": base,
              "shared_preference": replace(base, service_time=((2, 3), (2, 3))),
              "pressure": replace(base, failure_age=2, failure_probability=.65,
                                  service_time=((3, 4), (3, 4)))}
    if profile == "smoke":
        result = {k: replace(v, horizon=3, max_age=4) for k, v in result.items()}
    return result


def settings(profile: str) -> dict:
    if profile == "smoke":
        return dict(train_seeds=[95400], evaluation_seeds=[95410, 95411, 95412],
                    updates=160, batch_size=64, hidden=16, mixer_hidden=16, log_interval=20)
    if profile == "full":
        return dict(train_seeds=list(range(95000, 95010)),
                    evaluation_seeds=list(range(95200, 95250)), updates=10000,
                    batch_size=256, hidden=64, mixer_hidden=64, log_interval=100)
    raise ValueError(profile)


def canonical(time: int, state: PassiveState) -> PassiveState:
    """Discard elapsed service timestamps without changing observations/future law."""
    return replace(state, busy_until=tuple(max(time, b) for b in state.busy_until),
                   assigned_machine=tuple(m if b > time else -1
                                          for b, m in zip(state.busy_until, state.assigned_machine)))


def failure_outcomes(config: PassiveConfig, state: PassiveState):
    eligible = [not f and a + 1 >= config.failure_age for a, f in zip(state.ages, state.failed)]
    for events in itertools.product((False, True), repeat=config.machines):
        probability = math.prod((config.failure_probability if e else 1-config.failure_probability)
                                if can_fail else float(not e)
                                for can_fail, e in zip(eligible, events))
        if probability:
            yield events, probability


class ExactOracle:
    """Reachable-state DAG, all feasible actions, exact stochastic Bellman labels."""
    def __init__(self, config: PassiveConfig, *, progress: bool = True, max_states: int = 100000):
        config.validate()
        self.config = config
        self.env = PassiveTechnicianEnv(config)
        self.nodes: dict[Key, dict] = {}
        self.max_states = max_states
        self.initial = (0, canonical(0, self.env.initial_state()))
        with tqdm(desc="exact oracle states", unit="state", disable=not progress) as bar:
            self._solve(self.initial, bar)
        self.keys = sorted(self.nodes, key=lambda k: (k[0], repr(k[1])))
        self.key_to_id = {k: i for i, k in enumerate(self.keys)}
        self.audit_observations()

    def _solve(self, key: Key, bar) -> tuple[float, float]:
        time, state = key
        if time == self.config.horizon:
            return 0., 0.
        if key in self.nodes:
            n = self.nodes[key]
            return max(n["q95"]), max(n["q1"])
        if len(self.nodes) >= self.max_states:
            raise RuntimeError("oracle state cap exceeded; preserve incomplete run")
        # Recursion always advances time, so no in-progress node can be revisited.
        self.env.time, self.env.state = time, state
        obs, masks = self.env.observations(), self.env.action_masks()
        actions = list(itertools.product(*[[a for a, valid in enumerate(mask) if valid] for mask in masks]))
        node = dict(observations=obs, masks=masks, actions=actions, edges=[], q95=[], q1=[])
        for action in actions:
            edges, q95, q1 = [], 0., 0.
            mass = 0.
            for events, probability in failure_outcomes(self.config, state):
                nxt, cost, info = self.env.transition(state, action, time, events)
                expected_cost = (sum(state.failed)*self.config.downtime_cost
                                 + info["waiting"]*self.config.queue_waiting_cost
                                 + info["jobs"]*self.config.maintenance_cost
                                 + info["failures"]*self.config.failure_cost)
                if info["invalid_requests"] or abs(cost-expected_cost) > 1e-9:
                    raise RuntimeError("oracle feasibility/cost reconciliation failed")
                next_key = (time+1, canonical(time+1, nxt))
                future95, future1 = self._solve(next_key, bar)
                q95 += probability*(-cost + GAMMA*future95)
                q1 += probability*(-cost + future1)
                edges.append((probability, next_key, cost))
                mass += probability
            if abs(mass-1) > 1e-12 or not math.isfinite(q95+q1):
                raise RuntimeError("invalid oracle probability/value")
            node["edges"].append(edges)
            node["q95"].append(q95)
            node["q1"].append(q1)
        self.nodes[key] = node
        bar.update()
        return max(node["q95"]), max(node["q1"])

    def audit_observations(self):
        seen = {}
        for key, n in self.nodes.items():
            signature = n["observations"]
            if signature in seen:
                old = seen[signature]
                if old["actions"] != n["actions"] or not np.allclose(old["q95"], n["q95"], atol=1e-9, rtol=0):
                    raise RuntimeError("joint observation aliases different oracle Q targets")
                # Split by observation group to prevent cross-split aliases.
            seen[signature] = n

    def bellman_residual(self) -> float:
        residual = 0.
        for n in self.nodes.values():
            for index, edges in enumerate(n["edges"]):
                for gamma, name in ((GAMMA, "q95"), (1., "q1")):
                    value = sum(p*(-cost + gamma*(max(self.nodes[nxt][name]) if nxt in self.nodes else 0.))
                                for p, nxt, cost in edges)
                    residual = max(residual, abs(value-n[name][index]))
        return residual

    def policy_cost(self, indices: dict[Key, int], gamma: float) -> float:
        values = {}
        for key in reversed(self.keys):
            edges = self.nodes[key]["edges"][indices[key]]
            values[key] = sum(p*(cost + gamma*values.get(nxt, 0.)) for p, nxt, cost in edges)
        return values[self.initial]


def rank_reversal(actions, values, tolerance=1e-8):
    """Return a four-entry witness for a forbidden monotonic ranking reversal."""
    table = dict(zip(map(tuple, actions), values))
    for machine in range(len(actions[0])):
        other = 1-machine  # locked two-machine diagnostic
        for a, b in itertools.combinations(sorted({x[machine] for x in actions}), 2):
            deltas = []
            for context in sorted({x[other] for x in actions}):
                left, right = [0, 0], [0, 0]
                left[machine], right[machine] = a, b
                left[other] = right[other] = context
                left, right = tuple(left), tuple(right)
                if left in table and right in table:
                    deltas.append((table[left]-table[right], left, right))
            positive = [x for x in deltas if x[0] > tolerance]
            negative = [x for x in deltas if x[0] < -tolerance]
            if positive and negative:
                pos, neg = max(positive), min(negative)
                return dict(machine=machine, action_pair=[a, b],
                            positive_context=dict(actions=[pos[1], pos[2]],
                                                  q=[table[pos[1]], table[pos[2]]], delta=pos[0]),
                            negative_context=dict(actions=[neg[1], neg[2]],
                                                  q=[table[neg[1]], table[neg[2]]], delta=neg[0]),
                            minimum_reversal_margin=min(pos[0], -neg[0]))
    return None


class SignedMixer(TechnicianAwareMixer):
    """Same tensors and state features; weights retain signs."""
    def forward(self, agent_q, state):
        state = self._state(state)
        batch = agent_q.shape[0]
        w1 = self.hyper_w1(state).view(batch, self.agents, self.hidden_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.hidden_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_q.unsqueeze(1), w1)+b1)
        w2 = self.hyper_w2(state).view(batch, self.hidden_dim, 1)
        return torch.bmm(hidden, w2).squeeze((1, 2)) + self.value(state).squeeze(-1)


class JointQ(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.action_dim = config.technicians+1
        self.network = nn.Sequential(nn.Linear(config.machines*config.observation_dim, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU(),
                                     nn.Linear(width, self.action_dim**config.machines))

    def total_q(self, local, actions):
        index = actions[:, 0]*self.action_dim + actions[:, 1]
        return self.network(local.flatten(start_dim=1)).gather(1, index[:, None]).squeeze(1)


def model_for(arm, config, cfg, seed):
    torch.manual_seed(seed)
    model = PassiveValueDecomposition(config, "tqmix" if arm == "monotonic_cf" else "tqmix_no_cf",
                                     cfg["hidden"], cfg["mixer_hidden"])
    if arm == "signed_no_cf":
        signed = SignedMixer(config.machines, config.observation_dim, config.technicians, cfg["mixer_hidden"])
        signed.load_state_dict(model.mixer.state_dict())
        model.mixer = signed
    if arm == "joint_q":
        target = sum(p.numel() for p in model.parameters())
        d, a = config.machines*config.observation_dim, (config.technicians+1)**config.machines
        width = min(range(1, 513), key=lambda w: abs(w*w+(d+a+2)*w+a-target))
        torch.manual_seed(seed)
        model = JointQ(config, width)
    return model


def dataset(oracle: ExactOracle, split_seed: int):
    rng = random.Random(split_seed)
    split = {}
    by_time = {}
    # Observation-equivalent states are an indivisible split unit.
    for key in oracle.keys:
        by_time.setdefault(key[0], {}).setdefault(oracle.nodes[key]["observations"], []).append(key)
    for groups in by_time.values():
        groups = list(groups.values())
        rng.shuffle(groups)
        count = max(1, int(.8*len(groups)))
        for i, keys in enumerate(groups):
            for key in keys:
                split[key] = "train" if i < count else "heldout"
    xs, acts, ys, masks, ids, train = [], [], [], [], [], []
    slices = {}
    for key in oracle.keys:
        n = oracle.nodes[key]
        start = len(ys)
        obs = _obs_tensor(n["observations"], oracle.config)
        for action, target in zip(n["actions"], n["q95"]):
            xs.append(obs); acts.append(action); ys.append(target); masks.append(n["masks"])
            ids.append(oracle.key_to_id[key]); train.append(split[key] == "train")
        slices[key] = slice(start, len(ys))
    y = torch.tensor(ys, dtype=torch.float32)
    train_mask = torch.tensor(train, dtype=torch.bool)
    mean, std = y[train_mask].mean().item(), max(y[train_mask].std(unbiased=False).item(), 1e-6)
    if not (~train_mask).any():
        raise RuntimeError("no held-out states")
    return dict(x=torch.stack(xs), actions=torch.tensor(acts), y=(y-mean)/std, raw_y=y,
                masks=torch.tensor(masks, dtype=torch.bool), train_indices=torch.where(train_mask)[0],
                ids=ids, slices=slices, split=split, mean=mean, std=std)


def fit(model, data, cfg, seed, arm, cell, output, progress_rows):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed+17)
    indices = data["train_indices"]
    model.train()
    for update in tqdm(range(1, cfg["updates"]+1), desc=f"fit {cell}/{arm}/{seed}", unit="update", leave=False):
        selected = indices[torch.randint(len(indices), (cfg["batch_size"],), generator=generator)]
        x, actions, target = data["x"][selected], data["actions"][selected], data["y"][selected]
        prediction = model.total_q(x, actions)
        mse = (prediction-target).square().mean()
        cf = model.local_edge_consistency(x, actions, data["masks"][selected]) if arm == "monotonic_cf" else mse.new_zeros(())
        loss = mse + .05*cf
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if update % cfg["log_interval"] == 0 or update == cfg["updates"]:
            progress_rows.append(dict(cell=cell, algorithm=arm, train_seed=seed, update=update,
                                      normalized_mse=mse.item(), normalized_cf=cf.item(),
                                      raw_mse=mse.item()*data["std"]**2, raw_cf=cf.item()*data["std"]**2,
                                      gradient_norm=float(norm)))
            _write_csv(output / "training_progress.csv", progress_rows)
    return model.eval()


def predictions(model, data):
    with torch.no_grad():
        return torch.cat([model.total_q(data["x"][i:i+1024], data["actions"][i:i+1024])
                          for i in range(0, len(data["y"]), 1024)])


def evaluate_tables(model, arm, oracle, data, cell, seed):
    pred = predictions(model, data)
    indices, exhaustive, rows = {}, {}, []
    for key in oracle.keys:
        n, block = oracle.nodes[key], data["slices"][key]
        joint_index = int(pred[block].argmax())
        local_index = joint_index
        if arm != "joint_q":
            with torch.no_grad():
                values = model.agent_q(data["x"][block.start:block.start+1])[0]
                masks = data["masks"][block.start]
                action = tuple(values.masked_fill(~masks, -torch.inf).argmax(dim=-1).tolist())
                local_index = n["actions"].index(action)
        deployed = local_index if arm.startswith("monotonic") else joint_index
        indices[key], exhaustive[key] = deployed, joint_index
        if arm.startswith("monotonic") and float(pred[block][joint_index]-pred[block][local_index]) > 1e-4:
            raise RuntimeError("monotonic greedy vs exhaustive maximum audit failed")
        rows.append(dict(cell=cell, algorithm=arm, train_seed=seed, state_id=oracle.key_to_id[key],
                         time=key[0], split=data["split"][key],
                         normalized_mse=float((pred[block]-data["y"][block]).square().mean()),
                         regret=max(n["q95"])-n["q95"][deployed],
                         exhaustive_regret=max(n["q95"])-n["q95"][joint_index],
                         local_greedy_regret=None if arm == "joint_q" else max(n["q95"])-n["q95"][local_index],
                         greedy_joint_agreement=None if arm == "joint_q" else local_index == joint_index))
    metrics = {}
    for split in ("train", "heldout"):
        group = [r for r in rows if r["split"] == split]
        metrics[split+"_regret"] = statistics.fmean(r["regret"] for r in group)
        metrics[split+"_exhaustive_regret"] = statistics.fmean(r["exhaustive_regret"] for r in group)
        metrics[split+"_normalized_rmse"] = math.sqrt(statistics.fmean(r["normalized_mse"] for r in group))
    oracle95 = -max(oracle.nodes[oracle.initial]["q95"])
    oracle1 = -max(oracle.nodes[oracle.initial]["q1"])
    metrics.update(discounted_policy_cost=oracle.policy_cost(indices, GAMMA),
                   undiscounted_policy_cost=oracle.policy_cost(indices, 1.),
                   discounted_oracle_cost=oracle95, undiscounted_oracle_cost=oracle1)
    metrics["discounted_policy_gap"] = metrics["discounted_policy_cost"]-oracle95
    metrics["undiscounted_policy_gap"] = metrics["undiscounted_policy_cost"]-oracle1
    if min(metrics["discounted_policy_gap"], metrics["undiscounted_policy_gap"]) < -1e-8:
        raise RuntimeError("policy cost beats exact oracle")
    return indices, rows, dict(cell=cell, algorithm=arm, train_seed=seed, **metrics)


def rollout(oracle, indices, seed):
    env = PassiveTechnicianEnv(oracle.config, seed=seed)
    env.reset()
    components = dict(maintenance_cost=0., queue_waiting_cost=0., failure_cost=0., downtime_cost=0.)
    while env.time < env.config.horizon:
        key = (env.time, canonical(env.time, env.state))
        action = oracle.nodes[key]["actions"][indices[key]]
        for m, a in enumerate(action):
            if not env.action_mask(m)[a]:
                raise RuntimeError("policy selected invalid action")
        components["downtime_cost"] += sum(env.state.failed)*env.config.downtime_cost
        _, _, _, info = env.step(action)
        components["maintenance_cost"] += info["jobs"]*env.config.maintenance_cost
        components["queue_waiting_cost"] += info["waiting"]*env.config.queue_waiting_cost
        components["failure_cost"] += info["failures"]*env.config.failure_cost
    error = env.metrics["objective"]-sum(components.values())
    if abs(error) > 1e-9 or env.metrics["invalid_requests"]:
        raise RuntimeError("rollout reconciliation/feasibility failed")
    return dict(eval_seed=seed, **env.metrics, **components, cost_reconciliation_error=error)


def seed_audit(cfg):
    source = Path(__file__).resolve().parents[2] / "configs/ra_qmix_oracle_seed_registry.json"
    registry = json.loads(source.read_text())
    panels = [set(cfg["train_seeds"]), set(cfg["evaluation_seeds"]), set(DATASET_SEEDS), set(range(201, 301))]
    if any(a & b for i, a in enumerate(panels) for b in panels[i+1:]):
        raise ValueError("seed panels overlap")
    fresh = set().union(*panels[:3])
    if fresh & set(registry["declared_seed_values"]):
        raise ValueError("seed panel overlaps prior local manifests")
    return dict(scope=registry["scope"], prior_manifest_count=registry["manifest_count"],
                prior_panels_disjoint=True, panels_disjoint=True, sealed_panel_closed=True)


def summarize(metric_rows, cfg, structural):
    paired = []
    for seed in cfg["train_seeds"]:
        values = {arm: statistics.fmean(r["heldout_regret"] for r in metric_rows
                                       if r["train_seed"] == seed and r["algorithm"] == arm) for arm in ARMS}
        paired.append(dict(train_seed=seed, **values,
                           signed_minus_monotonic=values["signed_no_cf"]-values["monotonic_no_cf"]))
    diffs = [r["signed_minus_monotonic"] for r in paired]
    delta = statistics.fmean(diffs)
    baseline = statistics.fmean(r["monotonic_no_cf"] for r in paired)
    full = len(diffs) == 10
    half = 2.262157163*statistics.stdev(diffs)/math.sqrt(10) if full else None
    joint_rmse = statistics.fmean(r["heldout_normalized_rmse"] for r in metric_rows if r["algorithm"] == "joint_q")
    joint_regret = statistics.fmean(r["heldout_regret"] for r in metric_rows if r["algorithm"] == "joint_q")
    control = joint_rmse <= .10 and joint_regret <= .10
    support = delta <= -.05 and -delta >= .2*baseline and sum(d < 0 for d in diffs) >= 8
    return paired, dict(primary=dict(signed_minus_monotonic_regret=delta, baseline_regret=baseline,
                                    ci95=None if half is None else [delta-half, delta+half],
                                    improving_seed_count=sum(d < 0 for d in diffs),
                                    hypothesis_passed=support if full else None),
                         positive_control=dict(heldout_normalized_rmse=joint_rmse, heldout_regret=joint_regret,
                                               passed=control if full else None),
                         empirical_attribution_supported=(support and control) if full else None,
                         structural=structural,
                         interpretation="Exact Q-table restriction certificates do not establish optimal-policy impossibility or cause of adaptation failure.")


def run(args):
    if args.device != "cpu":
        raise ValueError("this protocol is CPU-only")
    torch.set_num_threads(1)
    cfg = settings(args.profile)
    audit = seed_audit(cfg)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    git_revision, git_dirty = _git_state()
    manifest = dict(protocol_version=PROTOCOL, status="RUNNING", profile=args.profile, device="cpu",
                    started_at=datetime.now(UTC).isoformat(), git_revision=git_revision, git_dirty=git_dirty,
                    runtime=_runtime_metadata(), train_seeds=cfg["train_seeds"],
                    evaluation_seeds=cfg["evaluation_seeds"], dataset_seeds=DATASET_SEEDS,
                    sealed_test_seeds=list(range(201, 301)), sealed_test_evaluated=False,
                    expected_models=3*4*len(cfg["train_seeds"]),
                    expected_evaluation_rows=3*4*len(cfg["train_seeds"])*len(cfg["evaluation_seeds"]))
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "seed_audit.json", audit)
    _write_json(output / "resolved_config.json", dict(settings=cfg, cells={k: asdict(v) for k, v in cells(args.profile).items()},
                gamma=GAMMA, learning_rate=3e-4, cf_weight=.05, max_oracle_states=100000,
                target_normalization="train mean/std; raw TD and CF losses divided by same variance"))
    training_rows, episode_rows, state_rows, metric_rows, counts, structural = [], [], [], [], [], {}
    models = 0
    try:
        for (cell, config), split_seed in zip(cells(args.profile).items(), DATASET_SEEDS):
            oracle = ExactOracle(config)
            data = dataset(oracle, split_seed)
            cell_dir = output / cell
            cell_dir.mkdir()
            oracle_rows, states, witnesses = [], [], []
            for key in oracle.keys:
                n = oracle.nodes[key]
                state_id = oracle.key_to_id[key]
                states.append(dict(state_id=state_id, time=key[0], state=asdict(key[1]),
                                   observations=n["observations"], split=data["split"][key]))
                for action, q95, q1 in zip(n["actions"], n["q95"], n["q1"]):
                    oracle_rows.append(dict(state_id=state_id, actions=json.dumps(action), q95=q95, q1=q1))
                witness = rank_reversal(n["actions"], n["q95"])
                if witness:
                    witnesses.append(dict(state_id=state_id, time=key[0], **witness))
            residual = oracle.bellman_residual()
            if residual > 1e-9:
                raise RuntimeError("oracle Bellman audit failed")
            structural[cell] = dict(state_count=len(oracle.keys), action_rows=len(oracle_rows),
                                    reversal_state_count=len(witnesses), structure_hypothesis_passed=bool(witnesses),
                                    bellman_residual=residual)
            _write_json(cell_dir / "oracle_states.json", states)
            _write_json(cell_dir / "structural_witnesses.json", witnesses)
            _write_csv(cell_dir / "oracle_q.csv", oracle_rows)
            for seed in tqdm(cfg["train_seeds"], desc=f"models {cell}", unit="seed"):
                reference = model_for("monotonic_no_cf", config, cfg, seed).state_dict()
                reference_count = sum(v.numel() for v in reference.values())
                for arm in ARMS:
                    model = model_for(arm, config, cfg, seed)
                    count = sum(p.numel() for p in model.parameters())
                    if arm != "joint_q" and not all(torch.equal(v, reference[k]) for k, v in model.state_dict().items()):
                        raise RuntimeError("factorized initialization differs")
                    if abs(count/reference_count-1) > (.10 if args.profile == "smoke" else .05):
                        raise RuntimeError("parameter matching failed")
                    counts.append(dict(cell=cell, algorithm=arm, train_seed=seed, parameters=count,
                                       reference_parameters=reference_count, relative_deviation=count/reference_count-1))
                    _write_csv(output / "parameter_counts.csv", counts)
                    fit(model, data, cfg, seed, arm, cell, output, training_rows)
                    checkpoint = cell_dir / arm / f"train_seed_{seed}" / "model.pt"
                    checkpoint.parent.mkdir(parents=True)
                    torch.save(dict(state_dict=model.state_dict(), algorithm=arm, config=asdict(config), settings=cfg,
                                    target_mean=data["mean"], target_std=data["std"], updates=cfg["updates"],
                                    protocol_version=PROTOCOL), checkpoint)
                    restored = model_for(arm, config, cfg, seed)
                    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
                    restored.eval()
                    if not torch.equal(predictions(model, data), predictions(restored, data)):
                        raise RuntimeError("checkpoint round-trip differs")
                    indices, individual, aggregate = evaluate_tables(restored, arm, oracle, data, cell, seed)
                    state_rows.extend(individual); metric_rows.append(aggregate)
                    _write_csv(output / "state_metrics.csv", state_rows)
                    _write_csv(output / "model_metrics.csv", metric_rows)
                    for eval_seed in tqdm(cfg["evaluation_seeds"], desc=f"evaluate {cell}/{arm}/{seed}", unit="episode", leave=False):
                        episode_rows.append(dict(cell=cell, algorithm=arm, train_seed=seed,
                                                 **rollout(oracle, indices, eval_seed)))
                    _write_csv(output / "episodes.partial.csv", episode_rows)
                    models += 1
        paired, summary = summarize(metric_rows, cfg, structural)
        audits = dict(seed_audit=audit, expected_models_complete=models == manifest["expected_models"],
                      expected_evaluation_rows_complete=len(episode_rows) == manifest["expected_evaluation_rows"],
                      unique_episode_keys=len({(r["cell"],r["algorithm"],r["train_seed"],r["eval_seed"]) for r in episode_rows}) == len(episode_rows),
                      zero_invalid_requests=all(r["invalid_requests"] == 0 for r in episode_rows),
                      cost_reconciliation=all(abs(r["cost_reconciliation_error"]) < 1e-9 for r in episode_rows),
                      observation_alias_audit_passed=True, bellman_audits_passed=True,
                      checkpoints_reload_identical=True, matched_updates=True, matched_minibatch_streams=True,
                      factorized_parameters_and_initialization_matched=True, joint_q_capacity_matched=True,
                      sealed_panel_closed=True)
        if not all(v for v in audits.values() if isinstance(v, bool)):
            raise RuntimeError(f"engineering audit failed: {audits}")
        summary["audits"] = audits
        summary["scientific_gate_applicable"] = args.profile == "full"
        _write_csv(output / "paired_seed_metrics.csv", paired)
        _write_csv(output / "episodes.csv", episode_rows)
        _write_csv(output / "coordination.csv", episode_rows)
        _write_json(output / "summary.json", summary)
        manifest.update(status="COMPLETED", completed_at=datetime.now(UTC).isoformat(),
                        actual_models=models, actual_evaluation_rows=len(episode_rows), audits=audits)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                        actual_models=models, actual_evaluation_rows=len(episode_rows))
        raise
    finally:
        manifest["outputs"] = sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
        _write_json(output / "manifest.json", manifest)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("smoke", "full"))
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
