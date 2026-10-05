"""Per-agent PPO ratios and complete-episode shared team returns."""

import hashlib
import json
import statistics
from dataclasses import asdict

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_dispatch as physics
from ht_pdm_fjsp import maintenance_dispatch_policy as bp
from ht_pdm_fjsp import maintenance_priority_allocation as core
from ht_pdm_fjsp import maintenance_waiting as envmod
from ht_pdm_fjsp import passive_technician_oracle_representation as helpers
from ht_pdm_fjsp.maintenance_coordination_policy import resolve


def row_weights(batch):
    mask = batch["machine_exists"].float()
    return mask / mask.sum(-1, keepdim=True).clamp_min(1)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum()


def proposal_batch(observations, proposals, probs, values):
    batch = bp.batch_observations(observations)
    b, n = batch["machine_exists"].shape
    k = batch["technicians"].shape[1]
    tokens = torch.full((b, n), k, dtype=torch.int64)
    old_probs, old_values = torch.zeros(b, n), torch.zeros(b, n)
    for i, (obs, p, lp, v) in enumerate(
        zip(observations, proposals, probs, values, strict=True)
    ):
        actual_k = len(obs["technicians"])
        tokens[i, : len(p)] = torch.tensor([k if j == actual_k else j for j in p])
        old_probs[i, : len(p)], old_values[i, : len(p)] = lp, v
    return batch, tokens, old_probs, old_values


def fit(model, cfg, seed, profile, output, logs):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    actions = torch.Generator().manual_seed(physics.role_seed(seed, "actions"))
    minibatches = torch.Generator().manual_seed(physics.role_seed(seed, "minibatches"))
    steps = episodes = optimizer_steps = 0
    if cfg["env_steps"] % cfg["rollout"] or cfg["rollout"] % cfg["minibatch"]:
        raise ValueError("exact rollout/minibatch budget")
    updates = cfg["env_steps"] // cfg["rollout"]
    for update in tqdm(
        range(1, updates + 1),
        desc=f"train {model.arm}/{seed}",
        unit="rollout",
        leave=False,
    ):
        observations, proposals, old_probs, old_values, rewards, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        records = []
        while len(rewards) < cfg["rollout"]:
            cs = physics.role_seed(seed, f"config:{episodes}")
            es = physics.role_seed(seed, f"environment:{episodes}")
            config = envmod.training_config(cs, profile, envmod.WAIT_PRICE)
            env = envmod.WaitingEnv(config)
            obs = env.reset(es)
            if cfg["rollout"] - len(rewards) < config.horizon:
                raise RuntimeError("incomplete episode rollout")
            start = len(rewards)
            for _ in range(config.horizon):
                p, lp, v = model.propose(obs, generator=actions)
                pairs, _ = resolve(obs, p)
                observations.append(obs)
                proposals.append(p)
                old_probs.append(lp)
                old_values.append(v)
                obs, reward, done, _ = env.step(pairs)
                rewards.append(reward / core.REWARD_SCALE)
                dones.append(done)
                steps += 1
            episodes += 1
            if (
                abs(sum(rewards[start:]) + env.metrics["objective"] / core.REWARD_SCALE)
                > 1e-8
            ):
                raise RuntimeError("episode return reconciliation")
            physical = {
                k: v
                for k, v in asdict(config).items()
                if k not in ("waiting_limit", "waiting_price")
            }
            records.append(
                dict(
                    algorithm=model.arm,
                    train_seed=seed,
                    episode=episodes,
                    physical_steps=steps,
                    config_seed=cs,
                    environment_seed=es,
                    machines=config.machines,
                    technicians=config.technicians,
                    config_sha256=hashlib.sha256(
                        json.dumps(asdict(config), sort_keys=True).encode()
                    ).hexdigest(),
                    physical_config_sha256=hashlib.sha256(
                        json.dumps(physical, sort_keys=True).encode()
                    ).hexdigest(),
                    **env.metrics,
                )
            )
        batch, tokens, old_probs, old_values = proposal_batch(
            observations, proposals, old_probs, old_values
        )
        targets = core.returns_for_episodes(rewards, dones)
        start = 0
        for i, done in enumerate(dones):
            if done:
                if abs(float(targets[start]) - sum(rewards[start : i + 1])) > 1e-5:
                    raise RuntimeError("Monte Carlo return reconciliation")
                start = i + 1
        weights = row_weights(batch)
        advantage = targets[:, None] - old_values
        mean = weighted_mean(advantage, weights)
        variance = weighted_mean((advantage - mean).square(), weights)
        advantage = (advantage - mean) / variance.sqrt().clamp_min(1e-8)
        with torch.no_grad():
            checked_probs, _, checked_values = model.evaluate(batch, tokens)
        mask = batch["machine_exists"]
        if not torch.allclose(
            checked_probs[mask], old_probs[mask], atol=5e-5, rtol=1e-5
        ) or not torch.allclose(
            checked_values[mask], old_values[mask], atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError(
                "sampled vs reevaluated per-agent probability/value mismatch"
            )
        stats = []
        model.train()
        for _ in range(cfg["epochs"]):
            order = torch.randperm(len(rewards), generator=minibatches)
            for begin in range(0, len(order), cfg["minibatch"]):
                idx = order[begin : begin + cfg["minibatch"]]
                mini = {k: v[idx] for k, v in batch.items()}
                probs, entropy, values = model.evaluate(mini, tokens[idx])
                w = weights[idx]
                ratio = (probs - old_probs[idx]).exp()
                surrogate = torch.minimum(
                    ratio * advantage[idx],
                    ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * advantage[idx],
                )
                policy_loss = -weighted_mean(surrogate, w)
                value_loss = weighted_mean((values - targets[idx, None]).square(), w)
                ent = weighted_mean(entropy, w)
                loss = (
                    policy_loss
                    + cfg["value_weight"] * value_loss
                    - cfg["entropy_weight"] * ent
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
                stats.append(
                    dict(
                        policy_loss=float(policy_loss.detach()),
                        value_loss=float(value_loss.detach()),
                        entropy=float(ent.detach()),
                        loss=float(loss.detach()),
                        gradient_norm=float(norm),
                        clipped_fraction=float(
                            weighted_mean(
                                (torch.abs(ratio.detach() - 1) > cfg["clip"]).float(), w
                            )
                        ),
                    )
                )
        model.eval()
        logs.append(
            dict(
                algorithm=model.arm,
                train_seed=seed,
                rollout_update=update,
                physical_steps=steps,
                episodes=episodes,
                optimizer_steps=optimizer_steps,
                mean_episode_cost=statistics.fmean(r["objective"] for r in records),
                **{k: statistics.fmean(r[k] for r in stats) for k in stats[0]},
            )
        )
        core.append_csv(output / "training_episodes.csv", records)
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
        sampled_agent_probability_audit=True,
        complete_episode_return_audit=True,
    )
