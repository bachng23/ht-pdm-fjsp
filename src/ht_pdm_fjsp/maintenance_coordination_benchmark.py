"""Locked learned scheduling, critic and communication benchmark (CPU only)."""

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
from ht_pdm_fjsp import maintenance_dispatch_policy as bp
from ht_pdm_fjsp import maintenance_priority_allocation as core
from ht_pdm_fjsp import maintenance_waiting as envmod
from ht_pdm_fjsp import maintenance_context_policy as context
from ht_pdm_fjsp import maintenance_coordination_policy as agents
from ht_pdm_fjsp import maintenance_coordination_references as refs
from ht_pdm_fjsp import maintenance_coordination_training as training
from ht_pdm_fjsp import passive_technician_oracle_representation as helpers

PROTOCOL = "maintenance_coordination_benchmark_v1"
REPOSITORY = Path(__file__).resolve().parents[2]
ARMS = ("central_ar", *agents.PROPOSAL_ARMS)
PRIMARY, NOMINAL = core.PRIMARY, core.NOMINAL
PRESSURE = ("n3_pressure", "n4_pressure")
CI_CRITICAL = 3.110934823179475  # Two-sided 98.75% Student-t, df9, four hypotheses.
DIAGNOSTICS = (
    "proposal_count",
    "contested_technicians",
    "rejected_proposals",
    "avoidable_unassigned_requests",
    "message_scalars",
)


def settings(profile):
    cfg = core.settings(profile)
    cfg.update(
        train_seeds=list(range(128000, 128010)) if profile == "full" else [131000],
        development_seeds=list(range(129000, 129020))
        if profile == "full"
        else [131020],
        evaluation_seeds=list(range(130000, 130050))
        if profile == "full"
        else [131010, 131011, 131012],
    )
    return cfg


def seed_audit(cfg):
    path = REPOSITORY / f"configs/{PROTOCOL}_seed_registry.json"
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
        scope=registry["scope"],
    )


def model_for(cfg, seed, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(seed)
    return (
        context.ContextActorCritic(cfg["hidden"], technician_context=True)
        if arm == "central_ar"
        else agents.ProposalActorCritic(cfg["hidden"], arm)
    )


def load_checkpoint(path):
    payload = torch.load(path, weights_only=True)
    if (
        payload["protocol"] != PROTOCOL
        or payload["environment_version"] != envmod.ENV_VERSION
        or payload["observation_contract"] != envmod.OBSERVATION_CONTRACT
        or payload["tie_tolerance"] != context.TIE_TOLERANCE
    ):
        raise ValueError("checkpoint contract mismatch")
    model = model_for(payload["settings"], payload["train_seed"], payload["algorithm"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


class Recorder:
    def __init__(self, controller):
        self.controller, self.records = controller, []

    def act(self, *args, **kwargs):
        result = self.controller.act(*args, **kwargs)
        diagnostics = getattr(self.controller, "last_diagnostics", {})
        self.records.append(
            {
                "proposals": diagnostics.get("proposals", "[]"),
                **{k: diagnostics.get(k, 0) for k in DIAGNOSTICS},
            }
        )
        return result


def rollout(config, seed, controller):
    recorder = Recorder(controller)
    row, traces, machines = core.rollout(
        config,
        seed,
        "learned_both",
        recorder,
        trace=True,
        environment_factory=envmod.WaitingEnv,
    )
    if len(recorder.records) != len(traces):
        raise RuntimeError("decision diagnostic count mismatch")
    for trace, diagnostic in zip(traces, recorder.records, strict=True):
        trace.update(diagnostic)
    for k in DIAGNOSTICS:
        row[k] = sum(r[k] for r in recorder.records)
    row["overdue_ticks_per_machine"] = row["overdue_waiting_ticks"] / config.machines
    row["wait_violation_machine_fraction"] = statistics.fmean(
        r["max_pending_wait"] > config.waiting_limit for r in machines
    )
    row["terminal_failed_never_serviced_fraction"] = statistics.fmean(
        bool(r["failed_at_terminal"] and r["service_starts"] == 0) for r in machines
    )
    return row, traces, machines


class OracleAdapter:
    def __init__(self, controller):
        self.controller = controller

    def act(self, obs, *args, **kwargs):
        return self.controller.act(
            envmod.augment_observation(obs, envmod.WAIT_LIMIT, envmod.WAIT_PRICE),
            "learned_both",
            deterministic=True,
        )


def quantile95(values):
    values = sorted(values)
    if not values:
        raise ValueError("empty quantile")
    return values[math.ceil(0.95 * len(values)) - 1]


def choose_reference(development, machines):
    rows = []
    for name in refs.REFERENCES:
        panel = [
            r
            for r in development
            if r["algorithm"] == name and r["family"] == "development"
        ]
        waits = [
            r["max_pending_wait"]
            for r in machines
            if r["algorithm"] == name and r["family"] == "development"
        ]
        rows.append(
            dict(
                algorithm=name,
                base_cost=statistics.fmean(r["base_cost"] for r in panel),
                wait_p95=quantile95(waits),
                eligible=quantile95(waits) <= envmod.WAIT_LIMIT,
            )
        )
    eligible = [r for r in rows if r["eligible"]]
    chosen = min(eligible or rows, key=lambda r: (r["base_cost"], r["algorithm"]))[
        "algorithm"
    ]
    return dict(
        selected=chosen,
        feasible_reference_exists=bool(eligible),
        selection_panel="development only",
        candidates=rows,
        selection_frozen_before_reference_test=True,
    )


def validate_panel(rows, algorithms, seeds, families, eval_seeds):
    expected = {
        (a, s, f, e)
        for a in algorithms
        for s in seeds
        for f in families
        for e in eval_seeds
    }
    actual = {
        (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"]) for r in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("incomplete or duplicate episode panel")


def hypothesis(
    differences, reductions, learner_sla, base_guard, nominal_guard, full, eligible=True
):
    if full and len(differences) != 10:
        raise RuntimeError("ten paired training seeds required")
    average = statistics.fmean(differences)
    half = CI_CRITICAL * statistics.stdev(differences) / math.sqrt(10) if full else None
    gates = dict(
        practical_reduction=reductions >= 0.05,
        seed_wins=sum(d < 0 for d in differences) >= 8,
        uncertainty=full and average + half < 0,
        pressure_wait_p95=learner_sla,
        base_cost_guard=base_guard,
        nominal_guard=nominal_guard,
        reference_eligible=eligible,
    )
    return dict(
        status=("PASS" if all(gates.values()) else "FAIL")
        if full
        else "ENGINEERING_ONLY",
        mean_paired_difference=average,
        ci_9875=[average - half, average + half] if full else None,
        relative_reduction=reductions,
        seed_wins=sum(d < 0 for d in differences),
        gates=gates,
    )


def summarize(episodes, machines, reference_rows, selection, cfg, full):
    validate_panel(
        episodes, ARMS, cfg["train_seeds"], envmod.FAMILIES, cfg["evaluation_seeds"]
    )
    if full and len(cfg["train_seeds"]) != 10:
        raise RuntimeError("ten training seeds required")
    data = episodes + [r for r in reference_rows if r["family"] != "development"]

    def mean(arm, seed, metric, families):
        actual_seed = -1 if arm in refs.REFERENCES else seed
        return statistics.fmean(
            statistics.fmean(
                r[metric]
                for r in data
                if r["algorithm"] == arm
                and r["train_seed"] == actual_seed
                and r["family"] == f
            )
            for f in families
        )

    def pressure_sla(arm):
        return all(
            quantile95(
                [
                    r["max_pending_wait"]
                    for r in machines
                    if r["algorithm"] == arm
                    and (seed is None or r["train_seed"] == seed)
                    and r["family"] == f
                ]
            )
            <= envmod.WAIT_LIMIT
            for seed in [*cfg["train_seeds"], None]
            for f in PRESSURE
        )

    comparisons = (
        ("H_learning", "central_ar", selection["selected"], "base_cost"),
        ("H_multiagent", "mappo_message", selection["selected"], "base_cost"),
        ("H_critic", "mappo_local", "ippo_local", "service_adjusted_cost"),
        ("H_message", "mappo_message", "mappo_local", "service_adjusted_cost"),
    )
    paired, results = [], {}
    for name, learner, control, metric in comparisons:
        rows = [
            dict(
                contrast=name,
                train_seed=s,
                learner=mean(learner, s, metric, PRIMARY),
                control=mean(control, s, metric, PRIMARY),
            )
            for s in cfg["train_seeds"]
        ]
        for r in rows:
            r["difference"] = r["learner"] - r["control"]
        reduction = 1 - statistics.fmean(r["learner"] for r in rows) / statistics.fmean(
            r["control"] for r in rows
        )
        base_guard = statistics.fmean(
            mean(learner, s, "base_cost", PRIMARY) for s in cfg["train_seeds"]
        ) <= 1.1 * statistics.fmean(
            mean(control, s, "base_cost", PRIMARY) for s in cfg["train_seeds"]
        )
        nominal_guard = statistics.fmean(
            mean(learner, s, "base_cost", NOMINAL) for s in cfg["train_seeds"]
        ) <= 1.1 * statistics.fmean(
            mean(control, s, "base_cost", NOMINAL) for s in cfg["train_seeds"]
        )
        results[name] = dict(
            learner=learner,
            control=control,
            metric=metric,
            **hypothesis(
                [r["difference"] for r in rows],
                reduction,
                pressure_sla(learner),
                base_guard,
                nominal_guard,
                full,
                selection["feasible_reference_exists"]
                if control in refs.REFERENCES
                else True,
            ),
        )
        paired.extend(rows)
    family_metrics = []
    for a in (*ARMS, *refs.REFERENCES):
        for f in envmod.FAMILIES:
            panel = [r for r in data if r["algorithm"] == a and r["family"] == f]
            machine_panel = [
                r for r in machines if r["algorithm"] == a and r["family"] == f
            ]
            numeric = [
                k
                for k, v in panel[0].items()
                if isinstance(v, (int, float)) and k not in ("eval_seed", "train_seed")
            ]
            family_metrics.append(
                dict(
                    algorithm=a,
                    family=f,
                    **{k: statistics.fmean(r[k] for r in panel) for k in numeric},
                    wait_p95=quantile95([r["max_pending_wait"] for r in machine_panel]),
                    max_wait=max(r["max_pending_wait"] for r in machine_panel),
                )
            )
    return paired, dict(
        protocol=PROTOCOL,
        hypotheses=results,
        family_metrics=family_metrics,
        reference_selection=selection,
        information_regime="user-approved synthetic sensitivity",
        marl_necessity_claim=False,
        scope="Conditional evidence only; requires actual deployment constraints before claiming necessity",
        uncertainty_unit="training seed; shared finite evaluation panel, no population-of-environments guarantee",
    )


def population_audit(path, cfg, profile):
    hashes = {(a, s): hashlib.sha256() for a in ARMS for s in cfg["train_seeds"]}
    counts = dict.fromkeys(hashes, 0)
    with path.open() as f:
        for r in csv.DictReader(f):
            key = r["algorithm"], int(r["train_seed"])
            hashes[key].update(
                json.dumps(
                    [
                        r[k]
                        for k in (
                            "episode",
                            "physical_steps",
                            "config_seed",
                            "environment_seed",
                            "physical_config_sha256",
                        )
                    ]
                ).encode()
            )
            counts[key] += 1
    return all(
        c == cfg["env_steps"] // (12 if profile == "full" else 4)
        for c in counts.values()
    ) and all(
        len({hashes[a, s].hexdigest() for a in ARMS}) == 1 for s in cfg["train_seeds"]
    )


def source_snapshot():
    modules = (physics, bp, core, envmod, context, agents, refs, training, helpers)
    sources = {Path(m.__file__).name: Path(m.__file__).read_bytes() for m in modules}
    sources[Path(__file__).name] = Path(__file__).read_bytes()
    for name in (f"docs/{PROTOCOL}_plan.md", f"configs/{PROTOCOL}_seed_registry.json"):
        p = REPOSITORY / name
        sources[p.name] = p.read_bytes()
    return sources


def run(args):
    if args.device != "cpu":
        raise ValueError("CPU-only protocol")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    revision, dirty = helpers._git_state()
    if args.profile == "full" and (revision is None or dirty is not False):
        raise RuntimeError("full experiment requires clean Git checkout")
    cfg = settings(args.profile)
    seed_checks = seed_audit(cfg)
    torch.set_num_threads(1)
    sources = source_snapshot()
    hashes = {n: hashlib.sha256(d).hexdigest() for n, d in sources.items()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir()
    for n, d in sources.items():
        (output / "source_snapshot" / n).write_bytes(d)
    wjson, wcsv = helpers._write_json, helpers._write_csv
    models = len(ARMS) * len(cfg["train_seeds"])
    per_optimizer = (
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
        source_bundle_sha256=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        sealed_test_evaluated=False,
        expected_models=models,
        expected_training_steps=models * cfg["env_steps"],
        expected_optimizer_steps=models * per_optimizer,
        expected_test_episodes=models
        * len(envmod.FAMILIES)
        * len(cfg["evaluation_seeds"]),
        expected_development_episodes=models * len(cfg["development_seeds"]),
        expected_reference_episodes=len(refs.REFERENCES)
        * (
            len(envmod.FAMILIES) * len(cfg["evaluation_seeds"])
            + len(cfg["development_seeds"])
        ),
        expected_message_off_episodes=len(cfg["train_seeds"])
        * len(envmod.FAMILIES)
        * len(cfg["evaluation_seeds"]),
        expected_exact_rows=models + len(refs.REFERENCES),
    )
    wjson(output / "manifest.json", manifest)
    wjson(output / "seed_audit.json", seed_checks)
    wjson(
        output / "benchmark_config.json",
        dict(
            environment_version=envmod.ENV_VERSION,
            waiting_limit=envmod.WAIT_LIMIT,
            waiting_price=envmod.WAIT_PRICE,
            synthetic_local_information=True,
            execution_arbiter="fixed reported urgency, ID ties, no retry",
            message_width=4,
        ),
    )
    wjson(
        output / "resolved_config.json",
        dict(
            settings=cfg,
            arms=ARMS,
            references=refs.REFERENCES,
            primary=PRIMARY,
            families=envmod.FAMILIES,
            observation_contract=envmod.OBSERVATION_CONTRACT,
            tie_tolerance=context.TIE_TOLERANCE,
            ci_critical=CI_CRITICAL,
        ),
    )
    configurations = {
        f: {
            str(e): asdict(envmod.family_config(f, e, args.profile, envmod.WAIT_PRICE))
            for e in (
                cfg["development_seeds"]
                if f == "development"
                else cfg["evaluation_seeds"]
            )
        }
        for f in (*envmod.FAMILIES, "development")
    }
    wjson(output / "evaluation_configs.json", configurations)
    (
        episodes,
        development,
        reference_rows,
        machines,
        message_off,
        exact,
        logs,
        coverage,
        counts,
    ) = [], [], [], [], [], [], [], [], []
    completed = 0

    def evaluate_controller(controller, arm, seed, families, target):
        for family in families:
            panel = (
                cfg["development_seeds"]
                if family == "development"
                else cfg["evaluation_seeds"]
            )
            for ev in tqdm(
                panel, desc=f"eval {arm}/{seed}/{family}", unit="episode", leave=False
            ):
                config = envmod.WaitingConfig(**configurations[family][str(ev)])
                row, traces, ms = rollout(config, ev, controller)
                key = dict(algorithm=arm, train_seed=seed, family=family, eval_seed=ev)
                target.append({**key, **row})
                machine_rows = [{**key, **r} for r in ms]
                machines.extend(machine_rows)
                core.append_csv(output / "machine_metrics.csv", machine_rows)
                if family != "development":
                    core.append_csv(
                        output / "decisions.csv", [{**key, **r} for r in traces]
                    )

    try:
        oracle = core.SmallOracle(physics.family_config("small", 0, args.profile))
        oracle.save(output)
        for name in refs.REFERENCES:
            evaluate_controller(
                refs.MatchingReference(name), name, -1, ("development",), reference_rows
            )
        selection = choose_reference(reference_rows, machines)
        wjson(output / "reference_selection.json", selection)
        for name in refs.REFERENCES:
            controller = refs.MatchingReference(name)
            evaluate_controller(controller, name, -1, envmod.FAMILIES, reference_rows)
            exact.append(
                dict(
                    algorithm=name,
                    train_seed=-1,
                    exact_objective="base_cost",
                    **oracle.evaluate("learned_both", OracleAdapter(controller)),
                )
            )
            wcsv(output / "reference_episodes.csv", reference_rows)
        for seed in tqdm(
            cfg["train_seeds"], desc="coordination benchmark", unit="seed"
        ):
            initial = model_for(cfg, seed, "ippo_local").state_dict()
            for arm in ARMS:
                model = model_for(cfg, seed, arm)
                if arm != "central_ar" and not all(
                    torch.equal(v, model.state_dict()[k]) for k, v in initial.items()
                ):
                    raise RuntimeError("proposal initialization mismatch")
                counts.append(
                    dict(
                        algorithm=arm,
                        train_seed=seed,
                        parameters=sum(p.numel() for p in model.parameters()),
                    )
                )
                if arm == "central_ar":
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
                            s, p, envmod.WAIT_PRICE
                        ),
                        algorithm=arm,
                    )
                else:
                    model, record = training.fit(
                        model, cfg, seed, args.profile, output, logs
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
                        tie_tolerance=context.TIE_TOLERANCE,
                        algorithm=arm,
                        train_seed=seed,
                        settings=cfg,
                        state_dict=model.state_dict(),
                    ),
                    checkpoint,
                )
                restored, _ = load_checkpoint(checkpoint)
                if not all(
                    torch.equal(v, restored.state_dict()[k])
                    for k, v in model.state_dict().items()
                ):
                    raise RuntimeError("checkpoint reload mismatch")
                probe = envmod.WaitingEnv(
                    envmod.training_config(
                        physics.role_seed(seed, "reload"),
                        args.profile,
                        envmod.WAIT_PRICE,
                    )
                ).reset(seed)
                if (
                    model.act(probe, "learned_both", deterministic=True)[:2]
                    != restored.act(probe, "learned_both", deterministic=True)[:2]
                ):
                    raise RuntimeError("checkpoint action mismatch")
                evaluate_controller(restored, arm, seed, ("development",), development)
                evaluate_controller(restored, arm, seed, envmod.FAMILIES, episodes)
                exact.append(
                    dict(
                        algorithm=arm,
                        train_seed=seed,
                        exact_objective="base_cost",
                        **oracle.evaluate("learned_both", OracleAdapter(restored)),
                    )
                )
                if arm == "mappo_message":
                    restored.message_enabled = False
                    evaluate_controller(
                        restored,
                        "mappo_message_off",
                        seed,
                        envmod.FAMILIES,
                        message_off,
                    )
                completed += 1
                wcsv(output / "parameter_counts.csv", counts)
                wcsv(output / "episodes.partial.csv", episodes)
                wcsv(output / "development_episodes.csv", development)
                if message_off:
                    wcsv(output / "message_off_episodes.csv", message_off)
                wcsv(output / "exact_small_metrics.csv", exact)
                wjson(output / "training_coverage.json", coverage)
        validate_panel(
            development,
            ARMS,
            cfg["train_seeds"],
            ("development",),
            cfg["development_seeds"],
        )
        validate_panel(
            message_off,
            ("mappo_message_off",),
            cfg["train_seeds"],
            envmod.FAMILIES,
            cfg["evaluation_seeds"],
        )
        validate_panel(
            [r for r in reference_rows if r["family"] != "development"],
            refs.REFERENCES,
            (-1,),
            envmod.FAMILIES,
            cfg["evaluation_seeds"],
        )
        validate_panel(
            [r for r in reference_rows if r["family"] == "development"],
            refs.REFERENCES,
            (-1,),
            ("development",),
            cfg["development_seeds"],
        )
        paired, summary = summarize(
            episodes, machines, reference_rows, selection, cfg, args.profile == "full"
        )
        allrows = episodes + development + reference_rows + message_off
        expected_machine_keys = {
            (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"], m)
            for r in allrows
            for m in range(
                3
                if r["family"] == "development"
                else 2
                if r["family"] == "small"
                else int(r["family"][1])
            )
        }
        actual_machine_keys = {
            (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"], r["machine"])
            for r in machines
        }
        audits = dict(
            complete_models=completed == models,
            complete_test_episodes=len(episodes) == manifest["expected_test_episodes"],
            complete_development_episodes=len(development)
            == manifest["expected_development_episodes"],
            complete_reference_episodes=len(reference_rows)
            == manifest["expected_reference_episodes"],
            complete_message_off_episodes=len(message_off)
            == manifest["expected_message_off_episodes"],
            complete_exact_rows=len(exact) == manifest["expected_exact_rows"],
            matched_proposal_capacity=len(
                {r["parameters"] for r in counts if r["algorithm"] != "central_ar"}
            )
            == 1,
            matched_physical_training_population=population_audit(
                output / "training_episodes.csv", cfg, args.profile
            ),
            complete_training_budget=sum(r["env_steps"] for r in coverage)
            == manifest["expected_training_steps"],
            complete_optimizer_budget=sum(r["optimizer_steps"] for r in coverage)
            == manifest["expected_optimizer_steps"],
            per_model_budget=all(
                r["env_steps"] == cfg["env_steps"]
                and r["optimizer_steps"] == per_optimizer
                for r in coverage
            ),
            unique_episode_keys=len(
                {
                    (r["algorithm"], r["train_seed"], r["family"], r["eval_seed"])
                    for r in allrows
                }
            )
            == len(allrows),
            complete_machine_panel=actual_machine_keys == expected_machine_keys
            and len(machines) == len(expected_machine_keys),
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
            checkpoint_reload_identical=True,
            proposal_initialization_matched=True,
            sealed_panels_closed=True,
        )
        if not all(audits.values()):
            raise RuntimeError(f"engineering audit failure {audits}")
        summary["audits"] = audits
        wcsv(output / "paired_seed_metrics.csv", paired)
        wcsv(output / "episodes.csv", episodes)
        wcsv(output / "coordination.csv", allrows)
        wjson(output / "summary.json", summary)
        manifest.update(
            status="COMPLETED",
            completed_at=datetime.now(UTC).isoformat(),
            actual_models=completed,
            actual_test_episodes=len(episodes),
            actual_development_episodes=len(development),
            actual_reference_episodes=len(reference_rows),
            actual_message_off_episodes=len(message_off),
            actual_training_steps=sum(r["env_steps"] for r in coverage),
            actual_optimizer_steps=sum(r["optimizer_steps"] for r in coverage),
            actual_training_episodes=sum(r["episodes"] for r in coverage),
            audits=audits,
        )
        print(
            json.dumps(
                dict(
                    status=manifest["status"],
                    hypotheses=summary["hypotheses"],
                    audits=audits,
                ),
                indent=2,
            )
        )
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
