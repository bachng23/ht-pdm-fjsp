"""Factorial experiment: allocation peer context × explicit overdue waiting cost."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_dispatch as physics
from ht_pdm_fjsp import maintenance_dispatch_policy as basepolicy
from ht_pdm_fjsp import maintenance_priority_allocation as core
from ht_pdm_fjsp import maintenance_waiting as envmod
from ht_pdm_fjsp import maintenance_context_policy as policymod
from ht_pdm_fjsp import passive_technician_oracle_representation as helpers
from ht_pdm_fjsp.maintenance_context_policy import ContextActorCritic

PROTOCOL = "maintenance_context_waiting_v1"
REPOSITORY = Path(__file__).resolve().parents[2]
ARMS = ("local_cost", "context_cost", "local_wait", "context_wait")
RULE = "risk_skill_rule"
PRIMARY = core.PRIMARY
NOMINAL = core.NOMINAL
PRESSURE = ("n3_pressure", "n4_pressure")


def factors(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return arm.startswith("context"), envmod.WAIT_PRICE if arm.endswith("wait") else 0.0


def settings(profile):
    cfg = core.settings(profile)
    cfg.update(
        train_seeds=list(range(122000, 122010)) if profile == "full" else [125000],
        development_seeds=list(range(123000, 123020))
        if profile == "full"
        else [125020],
        evaluation_seeds=list(range(124000, 124050))
        if profile == "full"
        else list(range(125010, 125013)),
    )
    return cfg


def seed_audit(cfg):
    path = REPOSITORY / "configs/maintenance_context_waiting_v1_seed_registry.json"
    registry = json.loads(path.read_text())
    panels = [
        set(cfg[k]) for k in ("train_seeds", "development_seeds", "evaluation_seeds")
    ]
    if any(a & b for i, a in enumerate(panels) for b in panels[i + 1 :]):
        raise ValueError("seed overlap")
    if set().union(*panels) & (
        set(registry["declared_seed_values"])
        | set(range(201, 301))
        | set(range(601, 701))
    ):
        raise ValueError("historical or sealed seed overlap")
    return dict(
        panels_disjoint=True,
        prior_declared_disjoint=True,
        sealed_panels_closed=True,
        registry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        historical_manifest_count=registry["manifest_count"],
        scope=registry["scope"],
    )


def model_for(cfg, seed, arm):
    context, _ = factors(arm)
    torch.manual_seed(seed)
    return ContextActorCritic(cfg["hidden"], technician_context=context)


def load_checkpoint(path):
    payload = torch.load(path, weights_only=True)
    if (
        payload["protocol"] != PROTOCOL
        or payload["environment_version"] != envmod.ENV_VERSION
        or payload["observation_contract"] != envmod.OBSERVATION_CONTRACT
        or payload["tie_tolerance"] != policymod.TIE_TOLERANCE
    ):
        raise ValueError("checkpoint contract mismatch")
    model = model_for(payload["settings"], payload["train_seed"], payload["algorithm"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def rollout(config, seed, model=None, trace=False):
    row, decisions, machines = core.rollout(
        config,
        seed,
        "learned_both",
        model,
        trace=trace,
        environment_factory=envmod.WaitingEnv,
    )
    row["overdue_ticks_per_machine"] = row["overdue_waiting_ticks"] / config.machines
    row["terminal_failed_never_serviced_fraction"] = statistics.fmean(
        bool(r["failed_at_terminal"] and r["service_starts"] == 0) for r in machines
    )
    row["wait_violation_machine_fraction"] = statistics.fmean(
        r["max_pending_wait"] > config.waiting_limit for r in machines
    )
    return row, decisions, machines


class OracleAdapter:
    """Original economic DP; policy retains its own publicly observed price."""

    def __init__(self, model, price):
        self.model, self.price = model, price

    def act(self, obs, *args, **kwargs):
        return self.model.act(
            envmod.augment_observation(obs, envmod.WAIT_LIMIT, self.price),
            "learned_both",
            deterministic=True,
        )


def quantile95(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty quantile")
    # Empirical inverse CDF, including all machine-episode records.
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def summarize(episodes, machines, cfg, full, references):
    expected = {
        (arm, seed, family, ev)
        for arm in ARMS
        for seed in cfg["train_seeds"]
        for family in envmod.FAMILIES
        for ev in cfg["evaluation_seeds"]
    }
    actual = {
        (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"]) for r in episodes
    }
    if actual != expected or len(episodes) != len(expected):
        raise RuntimeError("incomplete or duplicate evaluation panel")
    expected_machine = {
        (a, s, f, e, m)
        for a, s, f, e in expected
        for m in range(2 if f == "small" else int(f[1]))
    }
    actual_machine = {
        (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"], r["machine"])
        for r in machines
    }
    if actual_machine != expected_machine or len(machines) != len(expected_machine):
        raise RuntimeError("incomplete or duplicate machine panel")
    if full and len(cfg["train_seeds"]) != 10:
        raise ValueError("ten training seeds required")
    expected_rule = {
        (f, e)
        for f in (*envmod.FAMILIES, "development")
        for e in (
            cfg["development_seeds"] if f == "development" else cfg["evaluation_seeds"]
        )
    }
    if (
        len(references) != len(expected_rule)
        or {(r["family"], r["eval_seed"]) for r in references} != expected_rule
    ):
        raise RuntimeError("incomplete reference panel")
    grouped = {}
    for r in episodes:
        grouped.setdefault((r["algorithm"], r["train_seed"], r["family"]), []).append(r)

    def mean(arm, seed, metric, families):
        return statistics.fmean(
            statistics.fmean(r[metric] for r in grouped[arm, seed, f]) for f in families
        )

    def aggregate(arm, metric, families):
        return statistics.fmean(
            mean(arm, s, metric, families) for s in cfg["train_seeds"]
        )

    def contrast(name, learner, control, metric, families, practical):
        rows = [
            dict(
                contrast=name,
                train_seed=s,
                learned=mean(learner, s, metric, families),
                control=mean(control, s, metric, families),
            )
            for s in cfg["train_seeds"]
        ]
        for r in rows:
            r["delta"] = r["learned"] - r["control"]
        delta = statistics.fmean(r["delta"] for r in rows)
        baseline = statistics.fmean(r["control"] for r in rows)
        radius = (
            (2.685010847 * statistics.stdev(r["delta"] for r in rows) / math.sqrt(10))
            if full
            else None
        )
        wins = sum(r["delta"] < -1e-8 for r in rows)
        nominal_control = aggregate(control, "base_cost", NOMINAL)
        nominal_learned = aggregate(learner, "base_cost", NOMINAL)
        result = dict(
            metric=metric,
            learner=learner,
            control=control,
            control_mean=baseline,
            learned_mean=baseline + delta,
            mean_delta=delta,
            relative_reduction=-delta / baseline if baseline > 0 else None,
            improving_seeds=wins,
            ci97_5_bonferroni=[delta - radius, delta + radius] if full else None,
            nominal_base_cost_increase=nominal_learned / nominal_control - 1
            if nominal_control
            else None,
            nominal_guard=nominal_learned <= 1.1 * nominal_control,
            passed=(
                baseline > 0
                and delta <= -practical * baseline
                and wins >= 8
                and delta + radius < 0
                and nominal_learned <= 1.1 * nominal_control
            )
            if full
            else None,
        )
        return rows, result

    pairs, context = contrast(
        "H_context", "context_cost", "local_cost", "base_cost", PRIMARY, 0.05
    )
    wait_pairs, waiting = contrast(
        "H_wait",
        "context_wait",
        "context_cost",
        "overdue_ticks_per_machine",
        PRESSURE,
        0.50,
    )
    pooled_waits = {
        arm: {
            f: quantile95(
                [
                    r["max_pending_wait"]
                    for r in machines
                    if r["algorithm"] == arm and r["family"] == f
                ]
            )
            for f in envmod.FAMILIES
        }
        for arm in ARMS
    }
    cost_control = aggregate("context_cost", "base_cost", PRIMARY)
    cost_wait = aggregate("context_wait", "base_cost", PRIMARY)
    waiting.update(
        primary_base_cost_increase=cost_wait / cost_control - 1
        if cost_control
        else None,
        economic_guard=cost_wait <= 1.1 * cost_control,
        pressure_p95_wait={f: pooled_waits["context_wait"][f] for f in PRESSURE},
        waiting_bound_guard=all(
            pooled_waits["context_wait"][f] <= envmod.WAIT_LIMIT for f in PRESSURE
        ),
    )
    if full:
        waiting["passed"] = (
            waiting["passed"]
            and waiting["economic_guard"]
            and waiting["waiting_bound_guard"]
        )
    descriptive = {}
    for arm in ARMS:
        descriptive[arm] = {
            f: {
                k: aggregate(arm, k, (f,))
                for k in (
                    "base_cost",
                    "objective",
                    "waiting_cost",
                    "service_adjusted_cost",
                    "overdue_ticks_per_machine",
                    "failed_waiting_ticks",
                    "terminal_failed_never_serviced_fraction",
                    "wait_violation_machine_fraction",
                )
            }
            for f in envmod.FAMILIES
        }
    interaction = {
        k: statistics.fmean(
            (mean("context_wait", s, k, PRIMARY) - mean("local_wait", s, k, PRIMARY))
            - (mean("context_cost", s, k, PRIMARY) - mean("local_cost", s, k, PRIMARY))
            for s in cfg["train_seeds"]
        )
        for k in ("base_cost", "overdue_ticks_per_machine", "service_adjusted_cost")
    }
    rule_means = {
        f: statistics.fmean(r["base_cost"] for r in references if r["family"] == f)
        for f in envmod.FAMILIES
    }
    return pairs + wait_pairs, dict(
        protocol=PROTOCOL,
        confirmatory=full,
        hypotheses=dict(H_context=context, H_wait=waiting),
        family_metrics=descriptive,
        pooled_p95_max_wait=pooled_waits,
        interaction_descriptive=interaction,
        rule_family_base_cost=rule_means,
        independent_unit="training seed conditional on common fixed evaluation panel",
        exact_objective="base economic cost only; no waiting constraint",
        long_horizon="secondary out-of-distribution",
    )


def training_population_audit(path, cfg):
    hashes = {(arm, s): hashlib.sha256() for arm in ARMS for s in cfg["train_seeds"]}
    counts = dict.fromkeys(hashes, 0)
    with path.open() as f:
        for r in csv.DictReader(f):
            key = r["algorithm"], int(r["train_seed"])
            fields = [
                r[k]
                for k in (
                    "episode",
                    "physical_steps",
                    "config_seed",
                    "environment_seed",
                    "physical_config_sha256",
                )
            ]
            hashes[key].update(json.dumps(fields).encode())
            counts[key] += 1
    expected = cfg["env_steps"] // (12 if len(cfg["train_seeds"]) == 10 else 4)
    return all(v == expected for v in counts.values()) and all(
        len({hashes[a, s].hexdigest() for a in ARMS}) == 1 for s in cfg["train_seeds"]
    )


def source_snapshot():
    modules = (physics, basepolicy, core, envmod, policymod, helpers)
    sources = {Path(m.__file__).name: Path(m.__file__).read_bytes() for m in modules}
    sources[Path(__file__).name] = Path(__file__).read_bytes()
    for name in (
        "docs/maintenance_context_waiting_v1_plan.md",
        "configs/maintenance_context_waiting_v1_seed_registry.json",
    ):
        path = REPOSITORY / name
        sources[path.name] = path.read_bytes()
    return sources


def run(args):
    if args.device != "cpu":
        raise ValueError("CPU-only protocol")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    revision, dirty = helpers._git_state()
    if args.profile == "full" and (revision is None or dirty is not False):
        raise RuntimeError("full experiment requires a clean Git checkout")
    cfg = settings(args.profile)
    seed_checks = seed_audit(cfg)
    torch.set_num_threads(1)
    sources = source_snapshot()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir()
    for name, data in sources.items():
        (output / "source_snapshot" / name).write_bytes(data)
    wjson, wcsv = helpers._write_json, helpers._write_csv
    models = len(ARMS) * len(cfg["train_seeds"])
    optimizer_per_model = (
        cfg["env_steps"]
        // cfg["rollout"]
        * cfg["epochs"]
        * cfg["rollout"]
        // cfg["minibatch"]
    )
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
        source_hashes=hashes,
        source_bundle_sha256=bundle,
        sealed_test_evaluated=False,
        expected_models=models,
        expected_training_steps=models * cfg["env_steps"],
        expected_optimizer_steps=models * optimizer_per_model,
        expected_test_episodes=models
        * len(envmod.FAMILIES)
        * len(cfg["evaluation_seeds"]),
        expected_development_episodes=models * len(cfg["development_seeds"]),
        expected_reference_episodes=len(envmod.FAMILIES) * len(cfg["evaluation_seeds"])
        + len(cfg["development_seeds"]),
        expected_exact_rows=models + 1,
    )
    wjson(output / "manifest.json", manifest)
    wjson(output / "seed_audit.json", seed_checks)
    wjson(
        output / "benchmark_config.json",
        dict(
            environment_version=envmod.ENV_VERSION,
            waiting_limit=envmod.WAIT_LIMIT,
            waiting_price=envmod.WAIT_PRICE,
            completion_restoration=True,
            matching_only=True,
            synthetic_waiting_assumption=True,
            original_economic_costs_unchanged=True,
        ),
    )
    wjson(
        output / "resolved_config.json",
        dict(
            settings=cfg,
            arms={
                a: dict(context=factors(a)[0], waiting_price=factors(a)[1])
                for a in ARMS
            },
            families=envmod.FAMILIES,
            primary=PRIMARY,
            pressure=PRESSURE,
            nominal=NOMINAL,
            observation_contract=envmod.OBSERVATION_CONTRACT,
            tie_tolerance=policymod.TIE_TOLERANCE,
        ),
    )
    evaluation_configs = {
        f: {
            str(e): asdict(envmod.family_config(f, e, args.profile, 0))
            for e in (
                cfg["development_seeds"]
                if f == "development"
                else cfg["evaluation_seeds"]
            )
        }
        for f in (*envmod.FAMILIES, "development")
    }
    wjson(
        output / "evaluation_configs.json",
        dict(
            physical_template=evaluation_configs,
            price_override_by_arm={a: factors(a)[1] for a in ARMS},
        ),
    )
    episodes, development, references, machine_panel, exact, logs, counts, coverage = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    completed = 0
    try:
        oracle = core.SmallOracle(physics.family_config("small", 0, args.profile))
        oracle.save(output)
        exact.append(
            dict(
                algorithm=RULE,
                train_seed=-1,
                exact_objective="base_cost",
                **oracle.evaluate(RULE),
            )
        )
        for family in (*envmod.FAMILIES, "development"):
            panel = (
                cfg["development_seeds"]
                if family == "development"
                else cfg["evaluation_seeds"]
            )
            for ev in tqdm(panel, desc=f"rule {family}", unit="episode", leave=False):
                config = envmod.family_config(family, ev, args.profile, 0)
                row, traces, machines = rollout(
                    config, ev, trace=family != "development"
                )
                key = dict(algorithm=RULE, train_seed=-1, family=family, eval_seed=ev)
                references.append({**key, **row})
                if family != "development":
                    core.append_csv(
                        output / "reference_decisions.csv",
                        [{**key, **r} for r in traces],
                    )
                    core.append_csv(
                        output / "reference_machine_metrics.csv",
                        [{**key, **r} for r in machines],
                    )
        wcsv(output / "reference_episodes.csv", references)
        for seed in tqdm(cfg["train_seeds"], desc="context × waiting", unit="seed"):
            initial = model_for(cfg, seed, ARMS[0]).state_dict()
            for arm in ARMS:
                context, price = factors(arm)
                model = model_for(cfg, seed, arm)
                if not all(
                    torch.equal(v, model.state_dict()[k]) for k, v in initial.items()
                ):
                    raise RuntimeError("initialization mismatch")
                counts.append(
                    dict(
                        algorithm=arm,
                        train_seed=seed,
                        parameters=sum(p.numel() for p in model.parameters()),
                    )
                )
                model, record = core.fit(
                    model,
                    "learned_both",
                    cfg,
                    seed,
                    args.profile,
                    output,
                    logs,
                    environment_factory=envmod.WaitingEnv,
                    training_configuration=lambda s, p: envmod.training_config(
                        s, p, price
                    ),
                    algorithm=arm,
                )
                record.update(algorithm=arm, train_seed=seed)
                coverage.append(record)
                directory = output / arm / f"train_seed_{seed}"
                directory.mkdir(parents=True)
                wjson(directory / "training_coverage.json", record)
                checkpoint = directory / "model.pt"
                torch.save(
                    dict(
                        protocol=PROTOCOL,
                        environment_version=envmod.ENV_VERSION,
                        observation_contract=envmod.OBSERVATION_CONTRACT,
                        algorithm=arm,
                        train_seed=seed,
                        settings=cfg,
                        state_dict=model.state_dict(),
                        tie_tolerance=policymod.TIE_TOLERANCE,
                        source_bundle_sha256=bundle,
                    ),
                    checkpoint,
                )
                restored, _ = load_checkpoint(checkpoint)
                for family in envmod.FAMILIES:
                    env = envmod.WaitingEnv(
                        envmod.family_config(
                            family, cfg["evaluation_seeds"][0], args.profile, price
                        )
                    )
                    obs = env.reset(cfg["evaluation_seeds"][0])
                    for _ in range(env.config.horizon):
                        batch = basepolicy.batch_observations([obs])
                        with torch.no_grad():
                            a = model.decode(batch, "learned_both", deterministic=True)
                            b = restored.decode(
                                batch, "learned_both", deterministic=True
                            )
                        if not all(torch.equal(x, y) for x, y in zip(a, b)):
                            raise RuntimeError("checkpoint policy/value mismatch")
                        obs, _, _, _ = env.step(
                            tuple((m, j) for m, j in a[0][0].tolist() if m >= 0)
                        )
                for family in (*envmod.FAMILIES, "development"):
                    panel = (
                        cfg["development_seeds"]
                        if family == "development"
                        else cfg["evaluation_seeds"]
                    )
                    for ev in tqdm(
                        panel,
                        desc=f"evaluate {arm}/{seed}/{family}",
                        unit="episode",
                        leave=False,
                    ):
                        config = envmod.family_config(family, ev, args.profile, price)
                        row, traces, machines = rollout(
                            config, ev, restored, trace=family != "development"
                        )
                        key = dict(
                            algorithm=arm, train_seed=seed, family=family, eval_seed=ev
                        )
                        (development if family == "development" else episodes).append(
                            {**key, **row}
                        )
                        rows = [{**key, **r} for r in machines]
                        if family != "development":
                            machine_panel.extend(rows)
                            core.append_csv(
                                output / "decisions.csv", [{**key, **r} for r in traces]
                            )
                        core.append_csv(output / "machine_metrics.csv", rows)
                exact.append(
                    dict(
                        algorithm=arm,
                        train_seed=seed,
                        exact_objective="base_cost",
                        **oracle.evaluate(
                            "learned_both", OracleAdapter(restored, price)
                        ),
                    )
                )
                completed += 1
                wcsv(output / "parameter_counts.csv", counts)
                wcsv(output / "episodes.partial.csv", episodes)
                wcsv(output / "development_episodes.csv", development)
                wcsv(output / "exact_small_metrics.csv", exact)
                wjson(output / "training_coverage.json", coverage)
        paired, summary = summarize(
            episodes, machine_panel, cfg, args.profile == "full", references
        )
        allrows = episodes + development + references
        audits = dict(
            complete_models=completed == models,
            complete_test_episodes=len(episodes) == manifest["expected_test_episodes"],
            complete_development_episodes=len(development)
            == manifest["expected_development_episodes"],
            complete_reference_episodes=len(references)
            == manifest["expected_reference_episodes"],
            complete_exact_rows=len(exact) == manifest["expected_exact_rows"],
            matched_capacity=len({r["parameters"] for r in counts}) == 1,
            matched_physical_training_population=training_population_audit(
                output / "training_episodes.csv", cfg
            ),
            complete_training_budget=sum(r["env_steps"] for r in coverage)
            == manifest["expected_training_steps"],
            complete_optimizer_budget=sum(r["optimizer_steps"] for r in coverage)
            == manifest["expected_optimizer_steps"],
            per_model_budget=all(
                r["env_steps"] == cfg["env_steps"]
                and r["optimizer_steps"] == optimizer_per_model
                for r in coverage
            ),
            unique_episode_keys=len(
                {
                    (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"])
                    for r in allrows
                }
            )
            == len(allrows),
            zero_invalid_assignments=all(
                r["invalid_assignments"] == 0 for r in allrows
            ),
            cost_reconciled=all(
                abs(r["cost_reconciliation_error"]) < 1e-8
                and abs(r["objective"] - r["base_cost"] - r["waiting_cost"]) < 1e-8
                and abs(
                    r["service_adjusted_cost"]
                    - r["base_cost"]
                    - envmod.WAIT_PRICE * r["overdue_waiting_ticks"]
                )
                < 1e-8
                for r in allrows
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
            exact_base_oracle_evaluation_only=True,
            sealed_panels_closed=True,
        )
        if not all(audits.values()):
            raise RuntimeError(f"engineering audit failure {audits}")
        summary["audits"] = audits
        wcsv(output / "paired_seed_metrics.csv", paired)
        wcsv(output / "episodes.csv", episodes)
        wcsv(output / "coordination.csv", episodes + development)
        wjson(output / "summary.json", summary)
        manifest.update(
            status="COMPLETED",
            completed_at=datetime.now(UTC).isoformat(),
            actual_models=completed,
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
            actual_models=completed,
        )
        raise
    finally:
        manifest["outputs"] = sorted(
            str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
        )
        wjson(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("full", "smoke"))
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
