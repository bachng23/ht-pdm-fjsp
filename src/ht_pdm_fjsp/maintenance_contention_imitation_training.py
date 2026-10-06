"""Offline planner supervision; actor inputs never contain teacher scores."""

from dataclasses import asdict
import hashlib
import json
import random
import time

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_contention_model import (
    actions,
    cohort_config,
    compact,
    physical_action,
    plan,
    subsets,
)
from ht_pdm_fjsp.maintenance_contention_learning_policy import feature_batch
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

TIE = 1e-6


def checked_seed(seed, role, forbidden):
    value = role_seed(seed, role)
    if value in forbidden:
        raise ValueError("teacher seed overlaps reserved/historical panel")
    return value


def accepted_targets(candidates, means, n):
    target = torch.zeros(len(subsets(n)), dtype=torch.bool)
    minimum = min(means)
    for choice, score in zip(candidates, means, strict=True):
        if score <= minimum + TIE:
            target[subsets(n).index(tuple(choice))] = True
    if not target.any():
        raise ValueError("empty accepted teacher set")
    return target


def set_loss(model, features, targets):
    distribution, _ = model.distribution(features)
    if targets.shape != features["mask"].shape or not targets.any(-1).all():
        raise ValueError("teacher target dimensions/empty set")
    if (targets & ~features["mask"]).any():
        raise ValueError("infeasible teacher target")
    return -torch.logsumexp(
        distribution.logits.masked_fill(~targets, -torch.inf), -1
    ).mean()


def collect(cfg, seed, forbidden, selected_refs, journals):
    """Current-state planner labels with independent actual/forecast shock streams."""
    begin = time.perf_counter()
    data = {}
    coverage = dict(
        train_seed=seed,
        trajectories=0,
        physical_steps=0,
        planner_calls=0,
        noise_probes=0,
        train_rows=0,
        validation_rows=0,
        reserved_seed_transitions=0,
        future_event_access=False,
    )
    for split, count in [
        ("train", cfg["teacher_episodes"]),
        ("validation", cfg["teacher_validation_episodes"]),
    ]:
        features, targets, scores, metadata = [], [], [], []
        for index in tqdm(
            range(1, count + 1), desc=f"teacher {split}/{seed}", leave=False
        ):
            capacity = "low" if index % 5 == 0 else "middle"
            cs = checked_seed(
                seed, f"imitation_{split}_configuration:{index}", forbidden
            )
            es = checked_seed(seed, f"imitation_{split}_environment:{index}", forbidden)
            bs = checked_seed(seed, f"imitation_{split}_behavior:{index}", forbidden)
            if len({cs, es, bs}) != 3:
                raise ValueError("teacher role seed collision")
            c = cohort_config(cs, capacity, cfg["horizons"][0])
            env = WaitingEnv(c)
            env.reset(es)
            rng = random.Random(bs)
            for t in range(c.horizon):
                state = compact(env.state)
                h = c.horizon - t
                ps = checked_seed(
                    seed, f"imitation_{split}_planner:{index}:{t}", forbidden
                )
                chosen, diag = plan(
                    c, state, h, selected_refs[capacity], ps, cfg["scenarios"]
                )
                candidates = actions(c, state)
                if chosen not in candidates or tuple(diag["candidates"]) != candidates:
                    raise RuntimeError("teacher feasibility/candidate contract")
                target = accepted_targets(candidates, diag["means"], c.machines)
                batch = feature_batch([c], [state], [h])
                score = torch.full((len(subsets(c.machines)),), torch.inf)
                for choice, cost in zip(candidates, diag["means"], strict=True):
                    score[subsets(c.machines).index(choice)] = cost
                executed = (
                    candidates[rng.randrange(len(candidates))]
                    if rng.random() < 0.2
                    else chosen
                )
                _, reward, done, info = env.step(physical_action(env.state, executed))
                if (
                    done != (t == c.horizon - 1)
                    or abs(-reward - info["objective"]) > 1e-7
                ):
                    raise RuntimeError("teacher physical reward reconciliation")
                record = dict(
                    train_seed=seed,
                    split=split,
                    episode=index,
                    time=t,
                    capacity=capacity,
                    config_seed=cs,
                    environment_seed=es,
                    behavior_seed=bs,
                    planner_seed=ps,
                    remaining_horizon=h,
                    config=asdict(c),
                    state=state,
                    continuation=selected_refs[capacity],
                )
                journals["teacher_rows"].add(
                    dict(
                        **{
                            k: v
                            for k, v in record.items()
                            if k not in ("config", "state")
                        },
                        config=json.dumps(asdict(c), sort_keys=True),
                        config_sha256=hashlib.sha256(
                            json.dumps(asdict(c), sort_keys=True).encode()
                        ).hexdigest(),
                        state=json.dumps(state),
                        teacher_diagnostics=json.dumps(diag),
                        accepted_indices=json.dumps(
                            torch.nonzero(target).flatten().tolist()
                        ),
                        executed_subset=json.dumps(executed),
                        following=json.dumps(compact(env.state)),
                        physical_metrics=json.dumps(info),
                    )
                )
                features.append(batch)
                targets.append(target)
                scores.append(score)
                metadata.append(record)
                coverage["physical_steps"] += 1
                coverage["planner_calls"] += 1
                coverage[f"{split}_rows"] += 1
                if split == "validation" and len(metadata) <= cfg["noise_probes"]:
                    ns = checked_seed(seed, f"imitation_noise:{index}:{t}", forbidden)
                    probe, nd = plan(
                        c, state, h, selected_refs[capacity], ns, cfg["noise_scenarios"]
                    )
                    old_index = nd["candidates"].index(chosen)
                    journals["teacher_noise"].add(
                        dict(
                            train_seed=seed,
                            validation_row=len(metadata) - 1,
                            configuration_seed=cs,
                            environment_seed=es,
                            time=t,
                            original_planner_seed=ps,
                            probe_planner_seed=ns,
                            original_scenarios=cfg["scenarios"],
                            probe_scenarios=cfg["noise_scenarios"],
                            original_subset=json.dumps(chosen),
                            probe_subset=json.dumps(probe),
                            probe_regret_of_original=nd["means"][old_index]
                            - min(nd["means"]),
                            changed=int(probe != chosen),
                            probe_diagnostics=json.dumps(nd),
                        )
                    )
                    coverage["noise_probes"] += 1
            coverage["trajectories"] += 1
        data[split] = dict(
            features={k: torch.cat([f[k] for f in features]) for k in features[0]},
            targets=torch.stack(targets),
            scores=torch.stack(scores),
            metadata=metadata,
        )
    coverage["seconds"] = time.perf_counter() - begin
    return data, coverage


@torch.no_grad()
def diagnostics(model, dataset, batch_size=192):
    model.eval()
    population = len(dataset["targets"])
    losses = []
    agrees = []
    regrets = []
    choice_rows = choice_agreements = 0
    choice_regrets = []
    for start in range(0, population, batch_size):
        stop = min(start + batch_size, population)
        features = {k: v[start:stop] for k, v in dataset["features"].items()}
        targets = dataset["targets"][start:stop]
        prediction, _, _ = model.sample(features, deterministic=True)
        losses.append(float(set_loss(model, features, targets)) * (stop - start))
        agrees.append(targets.gather(1, prediction[:, None]).float().sum().item())
        opportunities = features["mask"].sum(-1) > 1
        choice_rows += int(opportunities.sum())
        choice_agreements += int(
            targets.gather(1, prediction[:, None]).squeeze(1)[opportunities].sum()
        )
        root = dataset["scores"][start:stop]
        regret = root.gather(1, prediction[:, None]).squeeze(1) - root.min(-1).values
        choice_regrets.extend(regret[opportunities].tolist())
        regrets.extend(
            (
                root.gather(1, prediction[:, None]).squeeze(1) - root.min(-1).values
            ).tolist()
        )
    return dict(
        rows=population,
        accepted_set_loss=sum(losses) / population,
        teacher_agreement=sum(agrees) / population,
        mean_teacher_score_regret=sum(regrets) / population,
        max_teacher_score_regret=max(regrets),
        choice_opportunity_rows=choice_rows,
        choice_teacher_agreement=choice_agreements / choice_rows
        if choice_rows
        else None,
        choice_mean_teacher_score_regret=sum(choice_regrets) / choice_rows
        if choice_rows
        else None,
    )


def fit_imitation(model, cfg, seed, data, journals, on_checkpoint):
    train = data["train"]
    population = len(train["targets"])
    if (
        population != cfg["teacher_episodes"] * cfg["horizons"][0]
        or population % cfg["bc_minibatch"]
    ):
        raise ValueError("exact imitation minibatch population")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    rng = torch.Generator().manual_seed(role_seed(seed, "imitation_minibatches"))
    steps = 0

    def record(epoch):
        for split in ("train", "validation"):
            journals["bc_diagnostics"].add(
                dict(
                    train_seed=seed,
                    epoch=epoch,
                    split=split,
                    **diagnostics(model, data[split], cfg["bc_minibatch"]),
                )
            )

    record(0)
    for epoch in tqdm(
        range(1, cfg["bc_epochs"] + 1), desc=f"BC seed{seed}", leave=False
    ):
        model.train()
        order = torch.randperm(population, generator=rng)
        losses = []
        norms = []
        for start in range(0, population, cfg["bc_minibatch"]):
            idx = order[start : start + cfg["bc_minibatch"]]
            features = {k: v[idx] for k, v in train["features"].items()}
            loss = set_loss(model, features, train["targets"][idx])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite BC loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        journals["bc_progress"].add(
            dict(
                train_seed=seed,
                epoch=epoch,
                optimizer_steps=steps,
                training_rows=population,
                loss=sum(losses) / len(losses),
                gradient_norm=sum(norms) / len(norms),
            )
        )
        if epoch in cfg["bc_checkpoints"]:
            record(epoch)
            on_checkpoint(model.eval(), epoch, steps)
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    critic_unchanged = all(
        torch.equal(before[k], v)
        for k, v in model.state_dict().items()
        if k.startswith("critic.")
    )
    if (
        not changed
        or not critic_unchanged
        or steps != cfg["bc_epochs"] * population // cfg["bc_minibatch"]
    ):
        raise RuntimeError("imitation update/critic/budget audit")
    return dict(
        train_seed=seed,
        env_steps=0,
        episodes=0,
        optimizer_steps=steps,
        parameters_changed=changed,
        critic_head_unchanged=critic_unchanged,
        validation_training_rows=0,
        teacher_label_rows=population,
        reserved_seed_training_transitions=0,
    )
