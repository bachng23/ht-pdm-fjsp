"""PPO study of learned maintenance priority and passive technician allocation."""

from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_dispatch as envmod
from ht_pdm_fjsp import maintenance_dispatch_policy as policymod
from ht_pdm_fjsp import passive_technician_oracle_representation as helpers
from ht_pdm_fjsp.maintenance_dispatch import DispatchEnv, DispatchState
from ht_pdm_fjsp.maintenance_dispatch_policy import (
    DispatchActorCritic,
    MODES,
    RULE_MODE,
    DEPLOYMENT_MODES,
    batch_observations,
    batch_sequences,
    rule_action,
)

PROTOCOL = "maintenance_priority_allocation_v2"
PRIMARY = ("n3_nominal", "n3_pressure", "n4_nominal", "n4_pressure")
NOMINAL = ("n3_nominal", "n4_nominal")
REWARD_SCALE = 20.0
REPOSITORY = Path(__file__).resolve().parents[2]


def settings(profile):
    if profile not in ("full", "smoke"):
        raise ValueError(profile)
    full = profile == "full"
    return dict(
        train_seeds=list(range(116000, 116010)) if full else [119000],
        development_seeds=list(range(117000, 117020)) if full else [119020],
        evaluation_seeds=list(range(118000, 118050))
        if full
        else [119010, 119011, 119012],
        env_steps=120000 if full else 48,
        rollout=480 if full else 24,
        epochs=4 if full else 2,
        minibatch=120 if full else 12,
        hidden=64 if full else 16,
        learning_rate=3e-4,
        clip=0.2,
        entropy_weight=0.01,
        value_weight=0.5,
        gradient_clip=0.5,
        gamma=1.0,
        gae_lambda=1.0,
    )


def seed_audit(cfg):
    path = REPOSITORY / "configs/maintenance_priority_allocation_v2_seed_registry.json"
    registry = json.loads(path.read_text())
    panels = [
        set(cfg[k]) for k in ("train_seeds", "development_seeds", "evaluation_seeds")
    ] + [set(range(201, 301)), set(range(601, 701))]
    if any(a & b for i, a in enumerate(panels) for b in panels[i + 1 :]):
        raise ValueError("seed overlap")
    if set().union(*panels[:3]) & set(registry["declared_seed_values"]):
        raise ValueError("prior seed overlap")
    return dict(
        panels_disjoint=True,
        prior_declared_disjoint=True,
        sealed_panels_closed=True,
        registry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        historical_manifest_count=registry["manifest_count"],
        scope=registry["scope"],
        derived_streams="SHA256 seed+role; config/environment/actions/minibatches separated",
    )


def append_csv(path, rows):
    if not rows:
        return
    with path.open("a") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def returns_for_episodes(rewards, dones):
    if not dones[-1]:
        raise RuntimeError("rollout must end on complete episode")
    result = [0.0] * len(rewards)
    value = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if dones[i]:
            value = 0.0
        value = rewards[i] + value
        result[i] = value
    return torch.tensor(result, dtype=torch.float32)


def model_for(cfg, seed):
    torch.manual_seed(seed)
    return DispatchActorCritic(cfg["hidden"])


def fit(
    model,
    mode,
    cfg,
    seed,
    profile,
    output,
    logs,
    *,
    environment_factory=DispatchEnv,
    training_configuration=envmod.training_config,
    algorithm=None,
):
    algorithm = algorithm or mode
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    actions = torch.Generator().manual_seed(envmod.role_seed(seed, "actions"))
    minibatches = torch.Generator().manual_seed(envmod.role_seed(seed, "minibatches"))
    steps = episodes = optimizer_steps = 0
    updates = cfg["env_steps"] // cfg["rollout"]
    if cfg["env_steps"] % cfg["rollout"] or cfg["rollout"] % cfg["minibatch"]:
        raise ValueError("exact rollout/minibatch budget")
    for update in tqdm(
        range(1, updates + 1),
        desc=f"train {algorithm}/{seed}",
        unit="rollout",
        leave=False,
    ):
        observations, sequences, old_probs, old_values, rewards, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        records = []
        sample_pairs = 0
        while len(rewards) < cfg["rollout"]:
            config_seed = envmod.role_seed(seed, f"config:{episodes}")
            environment_seed = envmod.role_seed(seed, f"environment:{episodes}")
            config = training_configuration(config_seed, profile)
            env = environment_factory(config)
            obs = env.reset(environment_seed)
            if (cfg["rollout"] - len(rewards)) < config.horizon:
                raise RuntimeError("incomplete episode rollout")
            start = len(rewards)
            for _ in range(config.horizon):
                pairs, tokens, logprob, value = model.act(obs, mode, generator=actions)
                observations.append(obs)
                sequences.append(tokens)
                old_probs.append(logprob)
                old_values.append(value)
                sample_pairs += len(pairs)
                obs, reward, done, info = env.step(pairs)
                rewards.append(reward / REWARD_SCALE)
                dones.append(done)
                steps += 1
            episodes += 1
            if (
                abs(sum(rewards[start:]) + env.metrics["objective"] / REWARD_SCALE)
                > 1e-8
            ):
                raise RuntimeError("episode return reconciliation")
            records.append(
                dict(
                    algorithm=algorithm,
                    train_seed=seed,
                    episode=episodes,
                    physical_steps=steps,
                    config_seed=config_seed,
                    environment_seed=environment_seed,
                    machines=config.machines,
                    technicians=config.technicians,
                    config_sha256=hashlib.sha256(
                        json.dumps(asdict(config), sort_keys=True).encode()
                    ).hexdigest(),
                    physical_config_sha256=hashlib.sha256(
                        json.dumps(
                            {
                                k: v
                                for k, v in asdict(config).items()
                                if k not in ("waiting_price", "waiting_limit")
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                    **env.metrics,
                )
            )
        batch = batch_observations(observations)
        tokens = batch_sequences(sequences)
        old_probs = torch.tensor(old_probs)
        old_values = torch.tensor(old_values)
        targets = returns_for_episodes(rewards, dones)
        episode_start = 0
        for i, done in enumerate(dones):
            if done:
                if (
                    abs(
                        float(targets[episode_start])
                        - sum(rewards[episode_start : i + 1])
                    )
                    > 1e-5
                ):
                    raise RuntimeError("Monte Carlo return reconciliation")
                episode_start = i + 1
        advantage = targets - old_values
        advantage = (advantage - advantage.mean()) / advantage.std(
            unbiased=False
        ).clamp_min(1e-8)
        model.train()
        with torch.no_grad():
            _, checked_probs, _, checked_values = model.decode(batch, mode, tokens)
        if not torch.allclose(
            checked_probs, old_probs, atol=5e-5, rtol=1e-5
        ) or not torch.allclose(checked_values, old_values, atol=1e-5, rtol=1e-5):
            raise RuntimeError("sampled vs reevaluated sequence/value mismatch")
        statistics_rows = []
        for epoch in range(cfg["epochs"]):
            order = torch.randperm(len(rewards), generator=minibatches)
            for begin in range(0, len(order), cfg["minibatch"]):
                indices = order[begin : begin + cfg["minibatch"]]
                mini = {k: v[indices] for k, v in batch.items()}
                _, probs, entropy, values = model.decode(mini, mode, tokens[indices])
                ratio = (probs - old_probs[indices]).exp()
                unclipped = ratio * advantage[indices]
                clipped = (
                    ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * advantage[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (values - targets[indices]).square().mean()
                loss = (
                    policy_loss
                    + cfg["value_weight"] * value_loss
                    - cfg["entropy_weight"] * entropy.mean()
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite PPO loss")
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer_steps += 1
                statistics_rows.append(
                    dict(
                        policy_loss=float(policy_loss.detach()),
                        value_loss=float(value_loss.detach()),
                        entropy=float(entropy.mean().detach()),
                        loss=float(loss.detach()),
                        gradient_norm=float(norm),
                        clipped_fraction=float(
                            (torch.abs(ratio.detach() - 1) > cfg["clip"]).float().mean()
                        ),
                    )
                )
        model.eval()
        logs.append(
            dict(
                algorithm=algorithm,
                train_seed=seed,
                rollout_update=update,
                physical_steps=steps,
                episodes=episodes,
                optimizer_steps=optimizer_steps,
                mean_episode_cost=statistics.fmean(r["objective"] for r in records),
                mean_matching_size=sample_pairs / len(rewards),
                **{
                    k: statistics.fmean(r[k] for r in statistics_rows)
                    for k in statistics_rows[0]
                },
            )
        )
        append_csv(output / "training_episodes.csv", records)
        helpers._write_csv(output / "training_progress.csv", logs)
    expected = updates * cfg["epochs"] * cfg["rollout"] // cfg["minibatch"]
    if optimizer_steps != expected or steps != cfg["env_steps"]:
        raise RuntimeError("training budget mismatch")
    return model.eval(), dict(
        env_steps=steps,
        episodes=episodes,
        rollout_updates=updates,
        optimizer_steps=optimizer_steps,
        evaluation_config_training_transitions=0,
        teacher_label_rows=0,
        sampled_sequence_probability_audit=True,
        complete_episode_return_audit=True,
    )


def load_checkpoint(path):
    payload = torch.load(path, weights_only=True)
    if (
        payload["protocol"] != PROTOCOL
        or payload["environment_version"] != envmod.ENV_VERSION
        or payload["observation_contract"] != envmod.OBSERVATION_CONTRACT
    ):
        raise ValueError("checkpoint contract mismatch")
    if payload["algorithm"] not in MODES:
        raise ValueError("checkpoint algorithm")
    model = model_for(payload["settings"], payload["train_seed"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def rollout(
    config, seed, mode, model=None, trace=False, *, environment_factory=DispatchEnv
):
    env = environment_factory(config)
    obs = env.reset(seed)
    decisions = []
    n = config.machines
    jobs = [0] * n
    waits = [0] * n
    unavailable = [0] * n
    max_wait = [0] * n
    actionable = [0] * n
    defer = [0] * n
    early_actionable = [0] * n
    early_defer = [0] * n
    edge_starts = [[0] * config.technicians for _ in range(n)]
    busy_ticks = [0] * config.technicians
    max_duration = max(max(row) for row in config.service_time)
    for tick in range(config.horizon):
        before = env.state
        mask = envmod.eligible_edges(config, before)
        if model is None:
            pairs = rule_action(obs)
            tokens = [*pairs, (-1, -1)]
        else:
            pairs, tokens, _, _ = model.act(obs, mode, deterministic=True)
        selected = {m for m, j in pairs}
        occupied = {m for m in before.assigned if m >= 0} | selected
        selected_technicians = {j for m, j in pairs}
        for m, j in pairs:
            edge_starts[m][j] += 1
        for j in range(config.technicians):
            busy_ticks[j] += int(before.remaining[j] > 0 or j in selected_technicians)
        for m in range(n):
            jobs[m] += int(m in selected)
            unavailable[m] += int(before.failed[m] or m in occupied)
            waits[m] += int(before.failed[m] and m not in occupied)
            max_wait[m] = max(max_wait[m], before.pending_wait[m])
            if before.failed[m] and any(mask[m]):
                actionable[m] += 1
                defer[m] += int(m not in selected)
                if config.horizon - tick > max_duration:
                    early_actionable[m] += 1
                    early_defer[m] += int(m not in selected)
        obs, reward, done, info = env.step(pairs)
        if trace:
            decisions.append(
                dict(
                    time=tick,
                    ages=json.dumps(before.ages),
                    failed=json.dumps(before.failed),
                    pending_wait=json.dumps(before.pending_wait),
                    remaining=json.dumps(before.remaining),
                    assigned=json.dumps(before.assigned),
                    sequence=json.dumps(tokens),
                    pairs=json.dumps(pairs),
                    **info,
                )
            )
    result = dict(
        eval_seed=seed,
        **env.metrics,
        failed_pending_final=sum(
            f and m not in env.state.assigned for m, f in enumerate(env.state.failed)
        ),
        unfinished_services=sum(r > 0 for r in env.state.remaining),
        actionable_failed_decisions=sum(actionable),
        failed_deferred_decisions=sum(defer),
        early_actionable_failed_decisions=sum(early_actionable),
        early_failed_deferred_decisions=sum(early_defer),
        technician_busy_ticks=json.dumps(busy_ticks),
        cost_reconciliation_error=env.metrics["objective"]
        - env.metrics.get("waiting_cost", 0.0)
        - sum(
            env.metrics[k]
            for k in ("maintenance_cost", "unavailability_cost", "failure_cost")
        ),
    )
    if (
        sum(jobs) != result["jobs"]
        or sum(waits) != result["failed_waiting_ticks"]
        or sum(unavailable) != result["unavailable_ticks"]
    ):
        raise RuntimeError("per-machine reconciliation")
    machine_rows = [
        dict(
            machine=m,
            service_starts=jobs[m],
            failed_waiting_ticks=waits[m],
            unavailable_ticks=unavailable[m],
            max_pending_wait=max(max_wait[m], env.state.pending_wait[m]),
            pending_wait_final=env.state.pending_wait[m],
            failed_at_terminal=env.state.failed[m],
            actionable_failed_decisions=actionable[m],
            failed_deferred_decisions=defer[m],
            early_actionable_failed_decisions=early_actionable[m],
            early_failed_deferred_decisions=early_defer[m],
            technician_service_starts=json.dumps(edge_starts[m]),
        )
        for m in range(n)
    ]
    return result, decisions, machine_rows


class SmallOracle:
    """Finite exact matching DP, evaluation-only; no teacher labels in PPO."""

    def __init__(self, config, max_states=25000):
        if config.machines != 2 or config.technicians != 2:
            raise ValueError("oracle restricted to small N2/K2")
        self.config = config
        self.nodes = {}
        self.max_states = max_states
        self.initials = [
            DispatchState(ages, (False,) * 2, (0,) * 2, (0,) * 2, (-1,) * 2)
            for ages in itertools.product(range(config.failure_age), repeat=2)
        ]
        with tqdm(desc="small exact matching states", unit="state") as bar:
            for state in self.initials:
                self.solve(0, state, bar)
        self.optimum = statistics.fmean(
            min(self.nodes[0, s]["q"]) for s in self.initials
        )

    def solve(self, time, state, bar):
        if time == self.config.horizon:
            return 0.0
        if (time, state) in self.nodes:
            return min(self.nodes[time, state]["q"])
        if len(self.nodes) >= self.max_states:
            raise RuntimeError("small oracle state cap")
        node = dict(actions=envmod.matching_actions(self.config, state), edges=[], q=[])
        self.nodes[time, state] = node
        bar.update(1)
        for action in node["actions"]:
            servicing = {m for m in state.assigned if m >= 0} | {m for m, j in action}
            eligible = [
                m
                for m in range(2)
                if m not in servicing
                and not state.failed[m]
                and state.ages[m] + 1 >= self.config.failure_age
            ]
            edges = []
            q = 0.0
            for flags in itertools.product((False, True), repeat=len(eligible)):
                events = [False] * 2
                probability = 1.0
                for m, event in zip(eligible, flags):
                    events[m] = event
                    probability *= (
                        self.config.failure_probability
                        if event
                        else 1 - self.config.failure_probability
                    )
                if not probability:
                    continue
                following, cost, info = envmod.transition(
                    self.config, state, action, events
                )
                edges.append((probability, following, cost))
                q += probability * (cost + self.solve(time + 1, following, bar))
            node["edges"].append(edges)
            node["q"].append(q)
        return min(node["q"])

    def evaluate(self, mode, model=None):
        selected = {}
        for (time, state), node in self.nodes.items():
            obs = envmod.observation(self.config, state, time)
            action = (
                rule_action(obs)
                if model is None
                else model.act(obs, mode, deterministic=True)[0]
            )
            action = tuple(sorted(action))
            selected[time, state] = node["actions"].index(action)
        values = {}
        for time, state in sorted(self.nodes, key=lambda k: k[0], reverse=True):
            edges = self.nodes[time, state]["edges"][selected[time, state]]
            values[time, state] = sum(
                p
                * (
                    cost
                    + (
                        values[time + 1, following]
                        if time + 1 < self.config.horizon
                        else 0.0
                    )
                )
                for p, following, cost in edges
            )
        result = statistics.fmean(values[0, s] for s in self.initials)
        if result < self.optimum - 1e-8:
            raise RuntimeError("policy beats exact floor")
        # On-policy occupancy weights give an additive regret decomposition:
        # first fix the selected machine set, then optimize its technician matching.
        occupancy = {(0, s): 1 / len(self.initials) for s in self.initials}
        priority_regret = allocation_regret = 0.0
        for key in sorted(self.nodes, key=lambda k: k[0]):
            weight = occupancy.get(key, 0.0)
            if not weight:
                continue
            time, state = key
            node = self.nodes[key]
            index = selected[key]
            machines = {m for m, j in node["actions"][index]}
            best_for_set = min(
                q
                for action, q in zip(node["actions"], node["q"])
                if {m for m, j in action} == machines
            )
            priority_regret += weight * (best_for_set - min(node["q"]))
            allocation_regret += weight * (node["q"][index] - best_for_set)
            if time + 1 < self.config.horizon:
                for probability, following, cost in node["edges"][index]:
                    next_key = (time + 1, following)
                    occupancy[next_key] = (
                        occupancy.get(next_key, 0.0) + weight * probability
                    )
        error = priority_regret + allocation_regret - (result - self.optimum)
        if abs(error) > 1e-8:
            raise RuntimeError("exact regret decomposition")
        return dict(
            exact_policy_cost=result,
            exact_optimum=self.optimum,
            optimality_gap=result - self.optimum,
            exact_selection_regret=priority_regret,
            exact_allocation_regret=allocation_regret,
            regret_decomposition_error=error,
        )

    def save(self, output):
        residual = 0.0
        records = []
        for (time, state), node in self.nodes.items():
            for action, edges, q in zip(node["actions"], node["edges"], node["q"]):
                expected = sum(
                    p
                    * (
                        cost
                        + (
                            min(self.nodes[time + 1, following]["q"])
                            if time + 1 < self.config.horizon
                            else 0.0
                        )
                    )
                    for p, following, cost in edges
                )
                residual = max(residual, abs(expected - q))
                records.append(
                    dict(
                        time=time,
                        state=json.dumps(asdict(state)),
                        pairs=json.dumps(action),
                        cost_to_go=q,
                    )
                )
        if residual > 1e-9:
            raise RuntimeError("oracle Bellman residual")
        helpers._write_csv(output / "small_oracle_actions.csv", records)
        helpers._write_json(
            output / "small_oracle_audit.json",
            dict(
                states=len(self.nodes),
                initials=len(self.initials),
                optimum=self.optimum,
                bellman_residual=residual,
                max_states=self.max_states,
            ),
        )


def summarize(episodes, cfg, full, references):
    if full and len(cfg["train_seeds"]) != 10:
        raise RuntimeError("confirmatory protocol requires ten training seeds")
    expected_episode_keys = {
        (arm, seed, family, ev)
        for arm in MODES
        for seed in cfg["train_seeds"]
        for family in envmod.FAMILIES
        for ev in cfg["evaluation_seeds"]
    }
    episode_keys = {
        (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"]) for r in episodes
    }
    if episode_keys != expected_episode_keys or len(episodes) != len(
        expected_episode_keys
    ):
        raise RuntimeError("incomplete or duplicate evaluation panel")
    # The rule is evaluated once on the SAME held-out episode panel. Reuse its
    # panel mean against each training seed without fabricating rule replicates.
    reference_keys = {(r["family"], r["eval_seed"]) for r in references}
    expected_keys = {
        (family, seed)
        for family in (*envmod.FAMILIES, "development")
        for seed in (
            cfg["development_seeds"]
            if family == "development"
            else cfg["evaluation_seeds"]
        )
    }
    if (
        reference_keys != expected_keys
        or len(references) != len(expected_keys)
        or any(r["algorithm"] != RULE_MODE for r in references)
    ):
        raise RuntimeError("incomplete or duplicate reference evaluation panel")
    idx = {}
    for arm in MODES:
        for seed in cfg["train_seeds"]:
            for family in envmod.FAMILIES:
                rr = [
                    r["objective"]
                    for r in episodes
                    if (r["algorithm"], r["train_seed"], r["family"])
                    == (arm, seed, family)
                ]
                if len(rr) != len(cfg["evaluation_seeds"]):
                    raise RuntimeError("incomplete evaluation panel")
                idx[arm, seed, family] = statistics.fmean(rr)
    for family in envmod.FAMILIES:
        costs = [r["objective"] for r in references if r["family"] == family]
        for seed in cfg["train_seeds"]:
            idx[RULE_MODE, seed, family] = statistics.fmean(costs)
    pairs = []
    contrasts = {}

    def mean(arm, seed):
        return statistics.fmean(idx[arm, seed, f] for f in PRIMARY)

    def nominal(arm):
        return statistics.fmean(
            idx[arm, s, f] for s in cfg["train_seeds"] for f in NOMINAL
        )

    for control in MODES[1:]:
        group = []
        for seed in cfg["train_seeds"]:
            a, b = mean(control, seed), mean(MODES[0], seed)
            row = dict(
                control=control,
                train_seed=seed,
                control_cost=a,
                learned_cost=b,
                cost_delta=b - a,
            )
            group.append(row)
            pairs.append(row)
        delta = statistics.fmean(r["cost_delta"] for r in group)
        baseline = statistics.fmean(r["control_cost"] for r in group)
        nc, nb = nominal(control), nominal(MODES[0])
        wins = sum(r["cost_delta"] < -1e-8 for r in group)
        radius = (
            2.685010847
            * statistics.stdev(r["cost_delta"] for r in group)
            / math.sqrt(10)
            if full
            else None
        )
        contrasts[control] = dict(
            control_cost=baseline,
            learned_cost=baseline + delta,
            mean_delta=delta,
            relative_reduction=-delta / baseline if baseline else None,
            improving_seeds=wins,
            ci97_5_bonferroni=[delta - radius, delta + radius] if full else None,
            nominal_control=nc,
            nominal_learned=nb,
            nominal_relative_increase=nb / nc - 1 if nc else None,
            passed=(
                baseline > 0
                and delta <= -0.05 * baseline
                and wins >= 8
                and delta + radius < 0
                and nb <= 1.1 * nc
            )
            if full
            else None,
        )
    reference_contrasts = {}
    for arm in MODES:
        deltas = [
            mean(arm, seed) - mean(RULE_MODE, seed) for seed in cfg["train_seeds"]
        ]
        rule_cost = mean(RULE_MODE, cfg["train_seeds"][0])
        delta = statistics.fmean(deltas)
        radius = (
            2.262157163 * statistics.stdev(deltas) / math.sqrt(10) if full else None
        )
        reference_contrasts[arm] = dict(
            rule_cost=rule_cost,
            learned_cost=rule_cost + delta,
            mean_delta=delta,
            relative_reduction=-delta / rule_cost if rule_cost else None,
            improving_training_seeds=sum(d < -1e-8 for d in deltas),
            ci95_descriptive=[delta - radius, delta + radius] if full else None,
            nominal_rule=nominal(RULE_MODE),
            nominal_learned=nominal(arm),
            nominal_relative_increase=nominal(arm) / nominal(RULE_MODE) - 1
            if nominal(RULE_MODE)
            else None,
            confirmatory=False,
            inference_unit="training seed conditional on fixed evaluation panel; rule evaluated once",
        )
    primary_passed = all(r["passed"] for r in contrasts.values()) if full else None
    rule_comparison = reference_contrasts[MODES[0]]
    return pairs, dict(
        contrasts=contrasts,
        reference_contrasts=reference_contrasts,
        primary_passed=primary_passed,
        learning_benefit_with_rule_descriptive=(
            primary_passed
            and rule_comparison["mean_delta"] < 0
            and rule_comparison["nominal_learned"] <= rule_comparison["nominal_rule"]
        )
        if full
        else None,
        scientific_gate_applicable=full,
        inference_unit="paired training seed; mean N3/4 nominal/pressure",
        secondary_families=["n5_pressure", "n5_k3_sparse", "small"],
        family_mean_costs={
            arm: {
                f: statistics.fmean(idx[arm, s, f] for s in cfg["train_seeds"])
                for f in envmod.FAMILIES
            }
            for arm in DEPLOYMENT_MODES
        },
    )


def exact_headroom(exact, *, require_complete=True):
    """Exact-cell ceiling only; never adjust the confirmatory practical gate."""
    rows = []
    if (
        not exact
        or max(r["exact_optimum"] for r in exact)
        - min(r["exact_optimum"] for r in exact)
        > 1e-8
    ):
        raise RuntimeError("inconsistent exact optimum")
    for record in exact:
        cost = record["exact_policy_cost"]
        optimum = record["exact_optimum"]
        gap = record["optimality_gap"]
        if abs(cost - optimum - gap) > 1e-8 or gap < -1e-8:
            raise RuntimeError("exact headroom cost reconciliation")
        if (
            abs(
                record["exact_selection_regret"]
                + record["exact_allocation_regret"]
                - gap
            )
            > 1e-8
        ):
            raise RuntimeError("exact headroom regret reconciliation")
        rows.append(
            dict(
                **record,
                maximum_relative_reduction=max(0.0, gap) / cost if cost > 0 else None,
                five_percent_possible_unrestricted=gap + 1e-8 >= 0.05 * cost
                if cost > 0
                else False,
            )
        )
    rule = [r for r in rows if r["algorithm"] == RULE_MODE]
    if len(rule) != 1:
        raise RuntimeError("exact rule headroom missing or duplicate")
    indices = {(r["algorithm"], r["train_seed"]): r for r in rows}
    if len(indices) != len(rows):
        raise RuntimeError("duplicate exact policy")
    contrasts = []
    for record in rows:
        if record["algorithm"] != MODES[0]:
            continue
        seed = record["train_seed"]
        for control in (*MODES[1:], RULE_MODE):
            baseline = rule[0] if control == RULE_MODE else indices.get((control, seed))
            if baseline is None:
                if require_complete:
                    raise RuntimeError("incomplete exact controls")
                continue
            cost = baseline["exact_policy_cost"]
            delta = record["exact_policy_cost"] - cost
            contrasts.append(
                dict(
                    control=control,
                    train_seed=seed,
                    learned_cost=record["exact_policy_cost"],
                    control_cost=cost,
                    exact_optimum=baseline["exact_optimum"],
                    cost_delta=delta,
                    relative_reduction=-delta / cost if cost > 0 else None,
                    control_maximum_relative_reduction=baseline[
                        "maximum_relative_reduction"
                    ],
                    five_percent_possible_unrestricted=baseline[
                        "five_percent_possible_unrestricted"
                    ],
                )
            )
    summary = dict(
        scope="N2/K2 exact cell only, unrestricted optimal continuation; not N3/N4 headroom or restricted architecture feasibility",
        rule_cost=rule[0]["exact_policy_cost"],
        exact_optimum=rule[0]["exact_optimum"],
        rule_maximum_relative_reduction=rule[0]["maximum_relative_reduction"],
        rule_five_percent_possible_unrestricted=rule[0][
            "five_percent_possible_unrestricted"
        ],
        policies_with_less_than_five_percent_headroom=[
            dict(algorithm=r["algorithm"], train_seed=r["train_seed"])
            for r in rows
            if not r["five_percent_possible_unrestricted"]
        ],
        confirmatory_threshold_unchanged=True,
        used_for_training_or_selection=False,
    )
    return rows, contrasts, summary


def run(args):
    if args.device != "cpu":
        raise ValueError("CPU-only protocol")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    torch.set_num_threads(1)
    cfg = settings(args.profile)
    revision, dirty = helpers._git_state()
    if args.profile == "full" and (revision is None or dirty is not False):
        raise RuntimeError("full experiment requires a clean Git checkout")
    audit = seed_audit(cfg)
    # Freeze all new source modules before training; never reread source during checkpoints.
    sources = {
        Path(module.__file__).name: Path(module.__file__).read_bytes()
        for module in (envmod, policymod)
    }
    sources[Path(__file__).name] = Path(__file__).read_bytes()
    for path in (
        REPOSITORY / "docs/maintenance_priority_allocation_v2_plan.md",
        REPOSITORY / "configs/maintenance_priority_allocation_v2_seed_registry.json",
    ):
        sources[path.name] = path.read_bytes()
    source_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in sources.items()
    }
    bundle_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir()
    for name, data in sources.items():
        (output / "source_snapshot" / name).write_bytes(data)
    models = len(MODES) * len(cfg["train_seeds"])
    per_updates = cfg["env_steps"] // cfg["rollout"]
    per_optimizer = per_updates * cfg["epochs"] * cfg["rollout"] // cfg["minibatch"]
    manifest = dict(
        protocol=PROTOCOL,
        status="RUNNING",
        profile=args.profile,
        device="cpu",
        git_revision=revision,
        git_dirty=dirty,
        started_at=datetime.now(UTC).isoformat(),
        runtime=helpers._runtime_metadata(),
        settings=cfg,
        deployment_modes=DEPLOYMENT_MODES,
        rule_trained=False,
        fixed_priority_contract="rule order of all feasible candidates; learned serve/STOP and allocation",
        source_hashes=source_hashes,
        source_bundle_sha256=bundle_hash,
        sealed_test_evaluated=False,
        expected_models=models,
        expected_training_steps=models * cfg["env_steps"],
        expected_optimizer_steps=models * per_optimizer,
        expected_test_episodes=models
        * len(envmod.FAMILIES)
        * len(cfg["evaluation_seeds"]),
        expected_development_episodes=models * len(cfg["development_seeds"]),
    )
    write_json, write_csv = helpers._write_json, helpers._write_csv
    write_json(output / "manifest.json", manifest)
    write_json(output / "seed_audit.json", audit)
    write_json(output / "observation_contract.json", envmod.OBSERVATION_CONTRACT)
    write_json(
        output / "benchmark_config.json",
        dict(
            environment_version=envmod.ENV_VERSION,
            completion_restoration=True,
            fifo=False,
            physical_capacity="partial matching of free machines and idle compatible technicians",
            start_costs=dict(preventive=1.0, corrective=2.0),
            unavailability_rate=6.0,
            failure_cost=15.0,
            no_duplicate_wait_penalty=True,
        ),
    )
    write_json(
        output / "resolved_config.json",
        dict(
            settings=cfg,
            modes=MODES,
            deployment_modes=DEPLOYMENT_MODES,
            training_population=dict(
                machines=[2, 3, 4],
                technicians=[1, 2],
                failure_age=[3, 4, 5],
                failure_probability=[0.25, 0.45, 0.60],
                durations=[1, 2, 3, 4],
                restoration=[0.5, 0.75, 1.0],
                initial="uniform healthy operating ages",
            ),
            families=envmod.FAMILIES,
            primary=PRIMARY,
            nominal=NOMINAL,
        ),
    )
    evaluation_configs = {
        family: {
            str(seed): asdict(envmod.family_config(family, seed, args.profile))
            for seed in cfg["evaluation_seeds"]
        }
        for family in envmod.FAMILIES
    }
    evaluation_configs["development"] = {
        str(seed): asdict(envmod.family_config("development", seed, args.profile))
        for seed in cfg["development_seeds"]
    }
    write_json(output / "evaluation_configs.json", evaluation_configs)
    episodes, development, references, exact, logs, counts, coverage = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    actual = 0
    try:
        small = SmallOracle(envmod.family_config("small", 0, args.profile))
        small.save(output)
        exact.append(
            dict(
                algorithm="risk_skill_rule",
                train_seed=-1,
                **small.evaluate("risk_skill_rule"),
            )
        )
        # Persist the rule's exact ceiling before any model is trained. This
        # is a diagnostic report, never a budget or threshold adaptation.
        headroom, exact_contrasts, headroom_summary = exact_headroom(exact)
        write_csv(output / "exact_headroom.csv", headroom)
        write_json(output / "exact_headroom_summary.json", headroom_summary)
        for family in (*envmod.FAMILIES, "development"):
            seeds = (
                cfg["development_seeds"]
                if family == "development"
                else cfg["evaluation_seeds"]
            )
            for seed in tqdm(seeds, desc=f"rule {family}", unit="episode", leave=False):
                config = envmod.family_config(family, seed, args.profile)
                row, traces, machines = rollout(
                    config, seed, RULE_MODE, trace=family != "development"
                )
                key = dict(family=family, algorithm=RULE_MODE, train_seed=-1)
                if family != "development":
                    append_csv(
                        output / "reference_decisions.csv",
                        [dict(**key, eval_seed=seed, **r) for r in traces],
                    )
                    append_csv(
                        output / "reference_machine_metrics.csv",
                        [dict(**key, eval_seed=seed, **r) for r in machines],
                    )
                references.append(
                    dict(
                        family=family, algorithm="risk_skill_rule", train_seed=-1, **row
                    )
                )
        write_csv(output / "reference_episodes.csv", references)
        for seed in tqdm(
            cfg["train_seeds"], desc="maintenance priority/allocation", unit="seed"
        ):
            initial = model_for(cfg, seed).state_dict()
            for mode in MODES:
                model = model_for(cfg, seed)
                if not all(
                    torch.equal(v, model.state_dict()[k]) for k, v in initial.items()
                ):
                    raise RuntimeError("initialization mismatch")
                counts.append(
                    dict(
                        algorithm=mode,
                        train_seed=seed,
                        parameters=sum(p.numel() for p in model.parameters()),
                    )
                )
                write_csv(output / "parameter_counts.csv", counts)
                model, record = fit(model, mode, cfg, seed, args.profile, output, logs)
                record.update(algorithm=mode, train_seed=seed)
                coverage.append(record)
                directory = output / mode / f"train_seed_{seed}"
                directory.mkdir(parents=True)
                write_json(directory / "training_coverage.json", record)
                checkpoint = directory / "model.pt"
                torch.save(
                    dict(
                        protocol=PROTOCOL,
                        environment_version=envmod.ENV_VERSION,
                        observation_contract=envmod.OBSERVATION_CONTRACT,
                        algorithm=mode,
                        train_seed=seed,
                        settings=cfg,
                        state_dict=model.state_dict(),
                        source_bundle_sha256=bundle_hash,
                    ),
                    checkpoint,
                )
                restored, payload = load_checkpoint(checkpoint)
                for family in envmod.FAMILIES:
                    env = DispatchEnv(
                        envmod.family_config(
                            family, cfg["evaluation_seeds"][0], args.profile
                        )
                    )
                    obs = env.reset(cfg["evaluation_seeds"][0])
                    for _ in range(env.config.horizon):
                        batch = batch_observations([obs])
                        with torch.no_grad():
                            a = model.decode(batch, mode, deterministic=True)
                            b = restored.decode(batch, mode, deterministic=True)
                        if not all(torch.equal(x, y) for x, y in zip(a, b)):
                            raise RuntimeError("checkpoint policy/value mismatch")
                        tokens = a[0][0].tolist()
                        pairs = tuple((m, j) for m, j in tokens if m >= 0)
                        obs, _, _, _ = env.step(pairs)
                for family in (*envmod.FAMILIES, "development"):
                    panel = (
                        cfg["development_seeds"]
                        if family == "development"
                        else cfg["evaluation_seeds"]
                    )
                    trace_rows, machine_rows = [], []
                    for ev in tqdm(
                        panel,
                        desc=f"evaluate {mode}/{seed}/{family}",
                        unit="episode",
                        leave=False,
                    ):
                        config = envmod.family_config(family, ev, args.profile)
                        row, traces, machines = rollout(
                            config, ev, mode, restored, trace=family != "development"
                        )
                        key = dict(family=family, algorithm=mode, train_seed=seed)
                        (development if family == "development" else episodes).append(
                            dict(**key, **row)
                        )
                        trace_rows.extend(
                            dict(**key, eval_seed=ev, **r) for r in traces
                        )
                        machine_rows.extend(
                            dict(**key, eval_seed=ev, **r) for r in machines
                        )
                    append_csv(output / "decisions.csv", trace_rows)
                    append_csv(output / "machine_metrics.csv", machine_rows)
                exact.append(
                    dict(
                        algorithm=mode,
                        train_seed=seed,
                        **small.evaluate(mode, restored),
                    )
                )
                write_csv(output / "episodes.partial.csv", episodes)
                write_csv(output / "development_episodes.csv", development)
                write_csv(output / "exact_small_metrics.csv", exact)
                headroom, exact_contrasts, headroom_summary = exact_headroom(
                    exact, require_complete=False
                )
                write_csv(output / "exact_headroom.csv", headroom)
                write_csv(output / "exact_headroom_contrasts.csv", exact_contrasts)
                write_json(output / "exact_headroom_summary.json", headroom_summary)
                actual += 1
        headroom, exact_contrasts, headroom_summary = exact_headroom(exact)
        pairs, summary = summarize(episodes, cfg, args.profile == "full", references)
        summary["exact_headroom"] = headroom_summary
        audits = dict(
            complete_models=actual == models,
            complete_test_episodes=len(episodes) == manifest["expected_test_episodes"],
            complete_development_episodes=len(development)
            == manifest["expected_development_episodes"],
            complete_reference_episodes=len(references)
            == len(envmod.FAMILIES) * len(cfg["evaluation_seeds"])
            + len(cfg["development_seeds"]),
            complete_exact_rows=len(exact) == models + 1,
            complete_exact_headroom_rows=len(headroom) == models + 1,
            complete_exact_headroom_contrasts=len(exact_contrasts)
            == 3 * len(cfg["train_seeds"]),
            four_deployment_baselines=len(summary["family_mean_costs"]) == 4,
            unique_reference_keys=len(
                {(r["family"], r["eval_seed"]) for r in references}
            )
            == len(references),
            matched_capacity=len({r["parameters"] for r in counts}) == 1,
            complete_training_budget=sum(r["env_steps"] for r in coverage)
            == manifest["expected_training_steps"],
            complete_optimizer_budget=sum(r["optimizer_steps"] for r in coverage)
            == manifest["expected_optimizer_steps"],
            per_model_budget=all(
                r["env_steps"] == cfg["env_steps"]
                and r["optimizer_steps"] == per_optimizer
                and r["rollout_updates"] == per_updates
                for r in coverage
            ),
            unique_evaluation_keys=len(
                {
                    (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"])
                    for r in episodes + development
                }
            )
            == len(episodes + development),
            zero_invalid_assignments=all(
                r["invalid_assignments"] == 0
                for r in episodes + development + references
            ),
            cost_reconciled=all(
                abs(r["cost_reconciliation_error"]) < 1e-8
                for r in episodes + development + references
            ),
            finite_training_logs=all(
                math.isfinite(r[k])
                for r in logs
                for k in (
                    "policy_loss",
                    "value_loss",
                    "loss",
                    "entropy",
                    "gradient_norm",
                )
            ),
            no_teacher_labels=all(r["teacher_label_rows"] == 0 for r in coverage),
            no_test_config_training=all(
                r["evaluation_config_training_transitions"] == 0 for r in coverage
            ),
            sampled_sequence_probabilities_verified=True,
            checkpoint_reload_identical=True,
            initialization_matched=True,
            per_machine_metrics_reconciled=True,
            training_rewards_and_MC_returns_reconciled=True,
            exact_oracle_evaluation_only=True,
            sealed_panels_closed=True,
        )
        if not all(audits.values()):
            raise RuntimeError(f"engineering audit failure {audits}")
        summary["audits"] = audits
        write_csv(output / "paired_seed_metrics.csv", pairs)
        write_csv(
            output / "reference_contrasts.csv",
            [
                dict(algorithm=arm, **record)
                for arm, record in summary["reference_contrasts"].items()
            ],
        )
        write_csv(output / "episodes.csv", episodes)
        write_csv(output / "coordination.csv", episodes + development)
        write_json(output / "summary.json", summary)
        manifest.update(
            status="COMPLETED",
            completed_at=datetime.now(UTC).isoformat(),
            actual_models=actual,
            actual_test_episodes=len(episodes),
            actual_development_episodes=len(development),
            actual_reference_episodes=len(references),
            actual_training_steps=sum(r["env_steps"] for r in coverage),
            actual_optimizer_steps=sum(r["optimizer_steps"] for r in coverage),
            actual_training_episodes=sum(r["episodes"] for r in coverage),
            audits=audits,
        )
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        manifest.update(
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
            actual_models=actual,
        )
        raise
    finally:
        manifest["outputs"] = sorted(
            str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
        )
        write_json(output / "manifest.json", manifest)
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
