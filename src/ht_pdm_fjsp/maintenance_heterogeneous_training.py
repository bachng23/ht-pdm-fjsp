"""Complete-episode vector-inference PPO with exact physical/optimizer budgets."""

import hashlib
import json
import math
from dataclasses import asdict

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_heterogeneous_model import cohort_config
from ht_pdm_fjsp.maintenance_heterogeneous_policy import feature_batch
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

REWARD_SCALE = 20.0


def episode_spec(seed, index, horizon, forbidden):
    cs = role_seed(seed, f"heterogeneous_learning_configuration:{index}")
    es = role_seed(seed, f"heterogeneous_learning_environment:{index}")
    if cs in forbidden or es in forbidden or cs == es:
        raise ValueError("training seed overlap with reserved/historical panel")
    capacity = "nominal" if index % 5 == 0 else "specialized"
    return cohort_config(cs, capacity, horizon), capacity, cs, es


def monte_carlo(rewards):
    """Rewards are [complete episode time, parallel episode]; gamma=1."""
    if rewards.ndim != 2 or not torch.isfinite(rewards).all():
        raise ValueError("complete-episode reward matrix")
    return rewards.flip(0).cumsum(0).flip(0)


def fit(model, cfg, seed, forbidden, journals, on_checkpoint):
    horizon = cfg["horizons"][0]
    b = cfg["vector_envs"]
    if (
        cfg["env_steps"] % cfg["rollout"]
        or cfg["rollout"] % cfg["minibatch"]
        or cfg["rollout"] % (b * horizon)
    ):
        raise ValueError("exact vector rollout budget")
    rng_seeds = [
        role_seed(seed, f"heterogeneous_learning_{role}")
        for role in ("actions", "minibatches")
    ]
    if len(set(rng_seeds)) != 2 or any(x in forbidden for x in rng_seeds):
        raise ValueError("training seed overlap with reserved/historical panel")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    actions = torch.Generator().manual_seed(
        role_seed(seed, "heterogeneous_learning_actions")
    )
    minibatches = torch.Generator().manual_seed(
        role_seed(seed, "heterogeneous_learning_minibatches")
    )
    episodes = steps = optimizer_steps = 0
    mix = {"nominal": 0, "specialized": 0}
    updates = cfg["env_steps"] // cfg["rollout"]
    initial_parameters = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for update in tqdm(
        range(1, updates + 1),
        desc=f"{model.algorithm} seed {seed}",
        unit="rollout",
        leave=False,
    ):
        feature_rows = []
        action_rows = []
        prob_rows = []
        value_rows = []
        target_rows = []
        episode_records = []
        model.eval()
        for _ in range(cfg["rollout"] // (b * horizon)):
            specs = [
                episode_spec(seed, episodes + i + 1, horizon, forbidden)
                for i in range(b)
            ]
            configs = [s[0] for s in specs]
            envs = [WaitingEnv(c) for c in configs]
            for env, spec in zip(envs, specs, strict=True):
                env.reset(spec[3])
            group_rewards = []
            for t in range(horizon):
                batch = feature_batch(
                    configs, [env.state for env in envs], [horizon - t] * b
                )
                action, logprob, value = model.sample(batch, generator=actions)
                selected = [
                    model.decode(c, env.state, horizon - t, i)[0]
                    for c, env, i in zip(configs, envs, action, strict=True)
                ]
                rewards = []
                for env, choice in zip(envs, selected, strict=True):
                    _, reward, done, _ = env.step(choice)
                    if done != (t == horizon - 1):
                        raise RuntimeError("episode boundary mismatch")
                    rewards.append(reward / REWARD_SCALE)
                feature_rows.append(batch)
                action_rows.append(action)
                prob_rows.append(logprob)
                value_rows.append(value)
                group_rewards.append(rewards)
                steps += b
            rewards = torch.tensor(group_rewards, dtype=torch.float32)
            targets = monte_carlo(rewards)
            target_rows.append(targets.flatten())
            for i, (env, spec) in enumerate(zip(envs, specs, strict=True)):
                episodes += 1
                c, capacity, cs, es = spec
                mix[capacity] += 1
                if (
                    abs(
                        float(rewards[:, i].sum())
                        + env.metrics["objective"] / REWARD_SCALE
                    )
                    > 5e-5
                    or abs(float(targets[0, i]) - float(rewards[:, i].sum())) > 5e-5
                ):
                    raise RuntimeError("complete-episode reward/return reconciliation")
                record = dict(
                    algorithm=model.algorithm,
                    train_seed=seed,
                    episode=episodes,
                    rollout_update=update,
                    config_seed=cs,
                    environment_seed=es,
                    condition=capacity,
                    config_sha256=hashlib.sha256(
                        json.dumps(asdict(c), sort_keys=True).encode()
                    ).hexdigest(),
                    physical_steps=steps,
                    **env.metrics,
                )
                journals["training_episodes"].add(record)
                episode_records.append(record)
        batch = {k: torch.cat([r[k] for r in feature_rows]) for k in feature_rows[0]}
        action = torch.cat(action_rows)
        old_prob = torch.cat(prob_rows)
        old_value = torch.cat(value_rows)
        target = torch.cat(target_rows)
        if len(target) != cfg["rollout"]:
            raise RuntimeError("rollout population mismatch")
        advantage = target - old_value
        advantage = (advantage - advantage.mean()) / advantage.std(
            unbiased=False
        ).clamp_min(1e-8)
        with torch.no_grad():
            checked, _, values = model.evaluate(batch, action)
        if not torch.allclose(
            checked, old_prob, atol=5e-5, rtol=1e-5
        ) or not torch.allclose(values, old_value, atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                "sampled/reevaluated masked probabilities or values mismatch"
            )
        model.train()
        stats = []
        for _ in range(cfg["epochs"]):
            order = torch.randperm(cfg["rollout"], generator=minibatches)
            for begin in range(0, cfg["rollout"], cfg["minibatch"]):
                idx = order[begin : begin + cfg["minibatch"]]
                mini = {k: v[idx] for k, v in batch.items()}
                prob, entropy, value = model.evaluate(mini, action[idx])
                ratio = (prob - old_prob[idx]).exp()
                surrogate = torch.minimum(
                    ratio * advantage[idx],
                    ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * advantage[idx],
                )
                policy_loss = -surrogate.mean()
                value_loss = (value - target[idx]).square().mean()
                ent = entropy.mean()
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
                        loss=float(loss.detach()),
                        policy_loss=float(policy_loss.detach()),
                        value_loss=float(value_loss.detach()),
                        entropy=float(ent.detach()),
                        gradient_norm=float(norm),
                        clipped_fraction=float(
                            (abs(ratio.detach() - 1) > cfg["clip"]).float().mean()
                        ),
                    )
                )
        metrics = {k: sum(row[k] for row in stats) / len(stats) for k in stats[0]}
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError("nonfinite training statistics")
        journals["training_progress"].add(
            dict(
                algorithm=model.algorithm,
                train_seed=seed,
                rollout_update=update,
                physical_steps=steps,
                episodes=episodes,
                optimizer_steps=optimizer_steps,
                mean_episode_cost=sum(r["objective"] for r in episode_records)
                / len(episode_records),
                **metrics,
            )
        )
        if update in cfg["checkpoint_updates"]:
            on_checkpoint(model.eval(), update, steps)
    expected = updates * cfg["epochs"] * cfg["rollout"] // cfg["minibatch"]
    changed = any(
        not torch.equal(initial_parameters[k], v) for k, v in model.state_dict().items()
    )
    if steps != cfg["env_steps"] or optimizer_steps != expected or not changed:
        raise RuntimeError("training budget/weight-update audit")
    return dict(
        train_seed=seed,
        env_steps=steps,
        episodes=episodes,
        optimizer_steps=optimizer_steps,
        mixture_counts=mix,
        parameters_changed=changed,
        sampled_probability_audit=True,
        complete_episode_returns=True,
        reserved_seed_training_transitions=0,
        teacher_label_rows=0,
    )
