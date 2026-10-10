"""Fixed-budget six-cell training coverage intervention; greedy execution only."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import hashlib
from pathlib import Path
import json
import platform
import shutil
import subprocess
import time
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_solver_comparison as base
from ht_pdm_fjsp.maintenance_contention_headroom import (
    Journal,
    write_json,
    write_csv,
    file_hash,
)
from ht_pdm_fjsp.maintenance_solver_model import (
    CONDITIONS,
    REFERENCES,
    cohort_config,
    reference,
    plan,
    profile_audit,
)
from ht_pdm_fjsp.maintenance_solver_policy import CENTRAL, LOCAL, MARL, FEATURE_CONTRACT
from ht_pdm_fjsp.maintenance_solver_training import fit
from ht_pdm_fjsp.maintenance_solver_diagnostics import interval, WIRE_CONTRACT
from ht_pdm_fjsp.maintenance_dispatch import validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, ENV_VERSION
from ht_pdm_fjsp.maintenance_heterogeneous_model import isolated, gain

VERSION = "maintenance_training_coverage_v1"
ROOT = base.ROOT
ARMS = (CENTRAL, LOCAL, MARL)
SCHEDULES = {
    "legacy": ["specialized"] * 4 + ["nominal"],
    "covered": ["specialized", "skill_mask", "specialized", "skill_mask", "nominal"],
}
METRICS = (
    "objective",
    "base_cost",
    "maintenance_cost",
    "failure_cost",
    "unavailability_cost",
    "waiting_cost",
    "failures",
    "unavailable_ticks",
    "waiting_violation",
    "terminal_pending",
    "corrective_requests",
    "served_requests",
    "censored_requests",
    "request_overdue_observed",
    "max_wait",
    "p95_wait",
    "inference_seconds",
    "communication_bytes",
    "communication_rounds",
    "served_waiting_cost",
    "censored_waiting_cost",
)


def settings(profile):
    cfg = base.settings(profile)
    full = profile == "full"
    cfg.update(
        train_seeds=list(range(180000, 180010)) if full else [189000],
        development_seeds=list(range(184000, 184010)) if full else [189020],
        test_seeds=list(range(185000, 185020)) if full else [189040, 189041],
        arms=list(ARMS),
        controls=["selected_rule", "joint_rollout"],
        training_condition_mixtures={
            k: {c: n / len(v) for c, n in Counter(v).items()}
            for k, v in SCHEDULES.items()
        },
        selection="final_checkpoint_only",
        decoder="greedy",
        primary_confidence=0.9875,
        source_protocol=base.VERSION,
        protocol=VERSION,
    )
    cfg.pop("training_condition_mixture")
    return cfg


def counts(cfg):
    n = len(SCHEDULES) * len(ARMS) * len(cfg["train_seeds"])
    dev = (
        len(cfg["development_cohorts"])
        * len(cfg["development_seeds"])
        * len(CONDITIONS)
    )
    panel = len(cfg["test_cohorts"]) * len(cfg["test_seeds"]) * len(CONDITIONS)
    ep = (n + len(cfg["controls"])) * panel * len(cfg["horizons"])
    return dict(
        models=n,
        training_steps=n * cfg["env_steps"],
        training_episodes=n * cfg["env_steps"] // cfg["horizons"][0],
        optimizer_steps=n
        * (cfg["env_steps"] // cfg["rollout"])
        * cfg["epochs"]
        * cfg["rollout"]
        // cfg["minibatch"],
        checkpoint_files=n * (len(cfg["checkpoint_updates"]) + 2),
        development_episodes=(len(REFERENCES) + n) * dev,
        test_episodes=ep,
        test_decision_intervals=(n + len(cfg["controls"]))
        * panel
        * sum(cfg["horizons"]),
        test_machine_rows=ep * 4,
        latency_rows=(n + len(cfg["controls"]))
        * len(CONDITIONS)
        * cfg["horizons"][0]
        * 3,
    )


def seed_audit(cfg):
    path = ROOT / f"configs/{VERSION}_seed_registry.json"
    registry = json.loads(path.read_text())
    parent = ROOT / "configs" / registry["historical_registry"]
    forbidden = set(json.loads(parent.read_text())["declared_seed_values"])
    for lo, hi in registry["inherited_panels"]:
        forbidden.update(range(lo, hi))
    fresh = [set(range(lo, hi)) for lo, hi in registry["new_panels"]]
    if any(a & b for a, b in itertools.combinations(fresh, 2)) or any(
        a & forbidden for a in fresh
    ):
        raise ValueError("new training/development/test seed overlap")
    frozen = settings(cfg["profile"])
    for key in ("train_seeds", "development_seeds", "test_seeds"):
        if not cfg[key] or not set(cfg[key]) <= set(frozen[key]):
            raise ValueError("seed outside registered panels")
    forbidden.update(set().union(*fresh))
    # Familiar profile labels are reserved as RNG values, while intentionally reused as labels.
    forbidden.update(cfg["development_cohorts"] + cfg["test_cohorts"])
    return dict(
        status="PASS",
        registry_sha256=file_hash(path),
        historical_registry_sha256=file_hash(parent),
        fresh_panels_disjoint=True,
        familiar_profile_labels_reused=True,
        full_shock_panel_opened=False,
    ), forbidden


def save_checkpoint(path, model, cfg, seed, regime, update, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            protocol=VERSION,
            environment_version=ENV_VERSION,
            feature_contract=FEATURE_CONTRACT,
            algorithm=model.algorithm,
            train_seed=seed,
            training_regime=regime,
            rollout_update=update,
            physical_steps=steps,
            settings=cfg,
            state_dict=model.state_dict(),
        ),
        path,
    )
    restored, payload = load_checkpoint(path)
    if any(
        not torch.equal(v, restored.state_dict()[k])
        for k, v in model.state_dict().items()
    ):
        raise RuntimeError("checkpoint reload tensor mismatch")
    c = cohort_config(
        cfg["development_cohorts"][0],
        "skill_mask",
        cfg["horizons"][0],
        profile=cfg["profile"],
    )
    env = WaitingEnv(c)
    env.reset(cfg["development_seeds"][0])
    for t in range(c.horizon):
        a = model.select(c, env.state, c.horizon - t)
        if a != restored.select(c, env.state, c.horizon - t):
            raise RuntimeError("checkpoint action/probability/value mismatch")
        env.step(a[0])
    return payload


def load_checkpoint(path):
    p = torch.load(path, map_location="cpu", weights_only=True)
    if (
        p["protocol"] != VERSION
        or p["environment_version"] != ENV_VERSION
        or p["feature_contract"] != FEATURE_CONTRACT
        or p["algorithm"] not in ARMS
        or p["training_regime"] not in SCHEDULES
        or p["settings"]["training_condition_schedule"]
        != SCHEDULES[p["training_regime"]]
    ):
        raise ValueError("coverage checkpoint contract mismatch")
    m = base.model_for(p["settings"], p["train_seed"], p["algorithm"])
    m.load_state_dict(p["state_dict"])
    return m.eval(), p


class CountedJournal(Journal):
    def __init__(self, path):
        super().__init__(path)
        self.count = 0

    def add(self, row):
        super().add(row)
        self.count += 1


class Tagged:
    def __init__(self, target, **tags):
        self.target, self.tags = target, tags

    def add(self, row):
        self.target.add({**row, **self.tags})


class Ticks:
    def __init__(self, target=None):
        self.target, self.count = target, 0

    def add(self, row):
        self.count += 1
        if self.target is not None:
            self.target.add(row)


class RequestJournal:
    def __init__(self, journal):
        self.journal = journal
        self.groups = defaultdict(Counter)
        self.episode = Counter()
        self.count = 0

    def add(self, row):
        overdue = max(row["wait"] - 4, 0)
        cost = 12.0 * overdue
        r = {
            **row,
            "overdue_ticks": overdue,
            "observed_waiting_cost": cost,
            "served_waiting_cost": cost if row["served"] else 0.0,
            "censored_waiting_cost": cost if row["censored"] else 0.0,
        }
        if (
            bool(row["served"]) == bool(row["censored"])
            or row["wait"] != row["end"] - row["arrival"]
        ):
            raise RuntimeError("request censoring/wait mismatch")
        key = (
            row["training_regime"],
            row["controller"],
            row["condition"],
            row["horizon"],
        )
        values = dict(
            requests=1,
            served=int(row["served"]),
            censored=int(row["censored"]),
            overdue_ticks=overdue,
            served_waiting_cost=r["served_waiting_cost"],
            censored_waiting_cost=r["censored_waiting_cost"],
        )
        self.groups[key].update(values)
        self.episode.update(values)
        self.count += 1
        self.journal.add(r)


def evaluate(
    cfg,
    cs,
    cond,
    h,
    shock,
    name,
    regime,
    train_seed,
    selected_refs,
    model=None,
    journals=None,
):
    actual = selected_refs[cond] if name == "selected_rule" else name
    tagged = (
        {k: Tagged(v, controller=name) for k, v in journals.items()}
        if journals
        else None
    )
    if journals:
        journals["requests"].episode.clear()
    row = base.episode(
        cohort_config(cs, cond, h, profile=cfg["profile"]),
        shock,
        actual,
        selected_refs[cond],
        cfg["scenarios"],
        dict(
            training_regime=regime,
            train_seed=train_seed,
            checkpoint_step=cfg["env_steps"] if model else -1,
            cohort=cs,
            condition=cond,
            horizon=h,
            **(
                {"training_horizon": cfg["training_horizons"].get(regime, -1)}
                if "training_horizons" in cfg
                else {}
            ),
        ),
        model,
        tagged,
    )
    row["controller"] = name
    if journals:
        q = journals["requests"].episode
        if (
            q["overdue_ticks"] != row["overdue_waiting_ticks"]
            or q["requests"] != row["corrective_requests"]
            or q["censored"] != row["terminal_pending"]
            or q["served_waiting_cost"] + q["censored_waiting_cost"]
            != row["waiting_cost"]
        ):
            raise RuntimeError("request/episode waiting decomposition mismatch")
        row.update(
            served_waiting_cost=q["served_waiting_cost"],
            censored_waiting_cost=q["censored_waiting_cost"],
        )
    return row


def summarize(rows, cfg):
    expected = {
        (regime, arm, seed, cs, cond, h, shock)
        for regime, arm, seeds in [
            (r, a, cfg["train_seeds"]) for r, a in itertools.product(SCHEDULES, ARMS)
        ]
        + [("reference", a, [-1]) for a in cfg["controls"]]
        for seed, cs, cond, h, shock in itertools.product(
            seeds, cfg["test_cohorts"], CONDITIONS, cfg["horizons"], cfg["test_seeds"]
        )
    }
    keys = [
        (
            r["training_regime"],
            r["controller"],
            r["train_seed"],
            r["cohort"],
            r["condition"],
            r["horizon"],
            r["seed"],
        )
        for r in rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("incomplete/duplicated evaluation grid")
    grouped = defaultdict(list)
    for r in rows:
        grouped[
            (
                r["training_regime"],
                r["controller"],
                r["condition"],
                r["horizon"],
                r["train_seed"],
            )
        ].append(r)
    means = {
        k: {m: float(np.mean([r[m] for r in p])) for m in METRICS}
        for k, p in grouped.items()
    }

    def avg(reg, arm, cond, h, metric="objective"):
        seeds = [-1] if reg == "reference" else cfg["train_seeds"]
        return float(np.mean([means[(reg, arm, cond, h, s)][metric] for s in seeds]))

    paired, contrasts = [], {}
    for cond, h in itertools.product(CONDITIONS, cfg["horizons"]):
        for arm, label in (
            [(a, "coverage") for a in ARMS]
            + [(a, "gap_closure") for a in (LOCAL, MARL)]
            + [(a, "residual_gap") for a in (LOCAL, MARL)]
        ):
            ds = []
            for s in cfg["train_seeds"]:
                covered = means[("covered", arm, cond, h, s)]["objective"]
                legacy = means[("legacy", arm, cond, h, s)]["objective"]
                cc = means[("covered", CENTRAL, cond, h, s)]["objective"]
                lc = means[("legacy", CENTRAL, cond, h, s)]["objective"]
                d = (
                    covered - legacy
                    if label == "coverage"
                    else (covered - cc) - (legacy - lc)
                    if label == "gap_closure"
                    else covered - cc
                )
                ds.append(d)
                paired.append(
                    dict(
                        contrast=f"{label}:{arm}:{cond}_H{h}",
                        train_seed=s,
                        difference=d,
                        covered_cost=covered,
                        legacy_cost=legacy,
                        central_covered_cost=cc,
                        central_legacy_cost=lc,
                    )
                )
            primary = (
                cond == "skill_mask"
                and h == cfg["horizons"][-1]
                and arm in (LOCAL, MARL)
                and label in ("coverage", "gap_closure")
            )
            confidence = cfg["primary_confidence"] if primary else 0.95
            ci = interval(ds, confidence) if cfg["profile"] == "full" else None
            old, new = avg("legacy", arm, cond, h), avg("covered", arm, cond, h)
            old_gap = old - avg("legacy", CENTRAL, cond, h)
            new_gap = new - avg("covered", CENTRAL, cond, h)
            reduction = 1 - new / old
            closure = (old_gap - new_gap) / old_gap if old_gap > 0 else None
            gates = dict(
                practical_effect=reduction >= 0.02
                if label == "coverage"
                else closure is not None and closure >= 0.25,
                wins_80pct=sum(d < -1e-9 for d in ds) >= 0.8 * len(ds),
                ci_upper_negative=ci is not None and ci[1] < 0,
                waiting_guard=avg("covered", arm, cond, h, "waiting_violation")
                <= avg("legacy", arm, cond, h, "waiting_violation") + 0.02,
                pending_guard=avg("covered", arm, cond, h, "terminal_pending")
                <= avg("legacy", arm, cond, h, "terminal_pending") + 0.05,
                specialized_preserved=avg(
                    "covered", arm, "specialized", cfg["horizons"][0]
                )
                <= 1.05 * avg("legacy", arm, "specialized", cfg["horizons"][0]),
                nominal_preserved=avg("covered", arm, "nominal", cfg["horizons"][0])
                <= 1.05 * avg("legacy", arm, "nominal", cfg["horizons"][0]),
            )
            if label == "residual_gap":
                gates = {}
            status = "ENGINEERING_ONLY" if cfg["profile"] == "smoke" else "EXPLORATORY"
            if cfg["profile"] == "full" and primary:
                status = (
                    "NOT_APPLICABLE"
                    if label == "gap_closure" and old_gap <= 0
                    else "PASS"
                    if all(gates.values())
                    else "FAIL"
                )
            contrasts[f"{label}:{arm}:{cond}_H{h}"] = dict(
                primary=primary,
                status=status,
                mean_difference=float(np.mean(ds)),
                ci=ci,
                confidence=confidence,
                unit="train_seed",
                n=len(ds),
                wins=sum(d < -1e-9 for d in ds),
                legacy_cost=old,
                covered_cost=new,
                priced_reduction=reduction,
                legacy_gap_to_central=old_gap,
                covered_gap_to_central=new_gap,
                gap_fraction_closed=closure,
                gates=gates,
                uncertainty_scope="training-replica variation conditional on familiar profile/fresh shock panel",
            )
    aggregates = {}
    for reg, arm, cond, h in {(k[0], k[1], k[2], k[3]) for k in means}:
        aggregates[f"{reg}:{arm}:{cond}_H{h}"] = {
            m: avg(reg, arm, cond, h, m) for m in METRICS
        }
    return paired, dict(contrasts=contrasts, aggregate=aggregates)


def benchmark_latency(
    output, cfg, selected_refs, selections, *, checkpoint_loader=None
):
    checkpoint_loader = checkpoint_loader or load_checkpoint
    states = []
    h = cfg["horizons"][0]
    shock = cfg["development_seeds"][0]
    for cond, h in itertools.product(
        CONDITIONS, cfg.get("latency_horizons", cfg["horizons"][:1])
    ):
        c = cohort_config(
            cfg["development_cohorts"][0], cond, h, profile=cfg["profile"]
        )
        env = WaitingEnv(c)
        env.reset(shock)
        for t in range(h):
            states.append((cond, c, env.state, h - t))
            env.step(reference(c, env.state, h - t, selected_refs[cond]))
    write_json(
        output / "latency_states.json",
        [
            dict(condition=cond, config=asdict(c), state=asdict(s), remaining=h)
            for cond, c, s, h in states
        ],
    )
    jobs = [
        dict(algorithm=a, training_regime="reference", train_seed=-1, checkpoint=None)
        for a in cfg["controls"]
    ] + list(selections.values())
    rows = []
    summaries = []
    for job in tqdm(jobs, desc="common-state latency", unit="controller"):
        model = (
            checkpoint_loader(output / job["checkpoint"])[0]
            if job["checkpoint"]
            else None
        )
        isolated.cache_clear()
        gain.cache_clear()
        measured = []
        for repeat in range(4):
            for i, (cond, c, s, h) in enumerate(states):
                began = time.perf_counter_ns()
                name = job["algorithm"]
                if model:
                    pairs, _ = model.select(c, s, h)
                elif name == "joint_rollout":
                    pairs, _ = plan(
                        c,
                        s,
                        h,
                        selected_refs[cond],
                        base.role_seed(shock, f"coverage_latency:{i}"),
                        cfg["scenarios"],
                    )
                else:
                    pairs = reference(c, s, h, selected_refs[cond])
                seconds = (time.perf_counter_ns() - began) / 1e9
                validate_matching(c, s, pairs)
                if repeat:
                    measured.append(seconds)
                    rows.append(
                        dict(
                            controller=name,
                            training_regime=job["training_regime"],
                            train_seed=job["train_seed"],
                            repeat=repeat,
                            state_index=i,
                            condition=cond,
                            remaining=h,
                            seconds=seconds,
                            action=json.dumps(pairs),
                            **(
                                {
                                    "training_horizon": cfg["training_horizons"].get(
                                        job["training_regime"], -1
                                    ),
                                    "evaluation_horizon": c.horizon,
                                }
                                if "training_horizons" in cfg
                                else {}
                            ),
                        )
                    )
        summaries.append(
            dict(
                controller=job["algorithm"],
                training_regime=job["training_regime"],
                train_seed=job["train_seed"],
                count=len(measured),
                mean_seconds=float(np.mean(measured)),
                p50_seconds=float(np.percentile(measured, 50)),
                p95_seconds=float(np.percentile(measured, 95)),
            )
        )
    write_csv(output / "latency.csv", rows)
    write_json(
        output / "latency_summary.json",
        dict(
            controllers=summaries,
            protocol="same development states; clear caches/controller; one warm then three measured passes; CPU1thread",
            caveat="Shared frontend included, central critic diagnostic included; logical communication is not measured network latency.",
        ),
    )
    isolated.cache_clear()
    gain.cache_clear()
    return len(rows)


def cell_settings(cfg, regime):
    return {
        **cfg,
        "training_regime": regime,
        "training_condition_schedule": SCHEDULES[regime],
    }


def audit_sequence(sequences, record, cfg):
    seed = record["train_seed"]
    digest = record["training_seed_sequence_sha256"]
    if seed in sequences and sequences[seed] != digest:
        raise RuntimeError("unmatched training configuration/shock seed sequence")
    sequences[seed] = digest


def run(output, profile, *, protocol=None):
    """Execute an explicit protocol module; defaults preserve coverage v1."""
    engine = protocol or sys.modules[__name__]
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new/empty; no resume")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    dirty = git("status", "--porcelain")
    if profile == "full" and (platform.system() != "Linux" or dirty):
        raise ValueError("full requires human-run clean committed Linux lab checkout")
    cfg = engine.settings(profile)
    torch.set_num_threads(1)
    audits, forbidden = engine.seed_audit(cfg)
    expected = engine.counts(cfg)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        experiment=engine.VERSION,
        status="RUNNING",
        commit=git("rev-parse", "HEAD"),
        dirty=dirty,
        started=datetime.now(timezone.utc).isoformat(),
        runtime=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            torch=str(torch.__version__),
            numpy=np.__version__,
        ),
        config=cfg,
        expected_counts=expected,
    )
    write_json(output / "manifest.json", manifest)
    journals = {}
    try:
        snap = output / "source_snapshot"
        snap.mkdir()
        sources = [
            *Path(__file__).parent.glob("maintenance_solver*.py"),
            *Path(__file__).parent.glob("maintenance_heterogeneous*.py"),
            *Path(__file__).parent.glob("maintenance_contention*.py"),
            Path(__file__),
            Path(engine.__file__),
            ROOT / "src/ht_pdm_fjsp/maintenance_waiting.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch_policy.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_coordination_references.py",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / f"docs/{engine.VERSION}_plan.md",
            ROOT / f"configs/{engine.VERSION}_seed_registry.json",
            ROOT / f"configs/{base.VERSION}_seed_registry.json",
        ]
        sources.extend(
            ROOT / "configs" / name for name in cfg.get("extra_snapshot_registries", [])
        )
        for p in dict.fromkeys(sources):
            shutil.copy2(p, snap / p.name)
        write_json(output / "benchmark_config.json", cfg)
        write_json(
            output / "resolved_config.json",
            {**cfg, "device": "cpu", "feature_contract": FEATURE_CONTRACT},
        )
        write_json(output / "seed_audit.json", audits)
        pa = profile_audit(profile)
        pa.update(familiar_test_profiles=True, fresh_shock_panel_opened=False)
        write_json(output / "profile_audit.json", pa)
        write_json(output / "communication_contract.json", WIRE_CONTRACT)
        write_json(
            output / "evaluation_configs.json",
            {
                f"{phase}:{cs}:{cond}:H{h}": asdict(
                    cohort_config(cs, cond, h, profile=profile)
                )
                for phase, cohorts in [
                    ("development", cfg["development_cohorts"]),
                    ("test", cfg["test_cohorts"]),
                ]
                for cs, cond, h in itertools.product(
                    cohorts,
                    CONDITIONS,
                    cfg["horizons"] if phase == "test" else cfg["horizons"][:1],
                )
            },
        )
        for key, file in [
            ("training_episodes", "training_episodes.csv"),
            ("training_progress", "training_progress.csv"),
            ("development", "development_episodes.csv"),
            ("episodes", "episodes.partial.csv"),
            ("machines", "machine_metrics.csv"),
            ("requests", "request_metrics.csv"),
        ]:
            journals[key] = CountedJournal(output / file)
        if profile == "smoke":
            journals["decision_file"] = Journal(output / "decisions.csv")
        ticks = Ticks(journals.get("decision_file"))
        requests = RequestJournal(journals["requests"])
        eval_journals = dict(
            decisions=ticks, machines=journals["machines"], requests=requests
        )
        dev = []
        jobs = list(
            itertools.product(
                cfg["development_cohorts"],
                CONDITIONS,
                cfg["development_seeds"],
                REFERENCES,
            )
        )
        for cs, cond, shock, name in tqdm(jobs, desc="development rules"):
            row = base.episode(
                cohort_config(cs, cond, cfg["horizons"][0], profile=profile),
                shock,
                name,
                None,
                cfg["scenarios"],
                dict(
                    training_regime="reference",
                    train_seed=-1,
                    checkpoint_step=-1,
                    cohort=cs,
                    condition=cond,
                    horizon=cfg["horizons"][0],
                    **({"training_horizon": -1} if "training_horizons" in cfg else {}),
                ),
            )
            dev.append(row)
            journals["development"].add(row)
        selected_refs = {
            cond: min(
                REFERENCES,
                key=lambda a: (
                    base.mean(
                        [
                            r
                            for r in dev
                            if r["condition"] == cond and r["controller"] == a
                        ],
                        "objective",
                    ),
                    a,
                ),
            )
            for cond in CONDITIONS
        }
        write_json(
            output / "reference_selection.json",
            dict(
                selected=selected_refs, frozen_before_test=True, development_only=True
            ),
        )
        initial_hashes = {}
        sequence_hashes = {}
        coverage = []
        selections = {}
        development_count = len(dev)
        for regime, arm, seed in tqdm(
            list(itertools.product(engine.SCHEDULES, ARMS, cfg["train_seeds"])),
            desc=cfg.get("replica_progress_label", "coverage learning replicas"),
            unit="model",
        ):
            cell = engine.cell_settings(cfg, regime)
            model = base.model_for(cell, seed, arm)
            digest = hashlib.sha256(
                b"".join(
                    v.detach().numpy().tobytes() for v in model.state_dict().values()
                )
            ).hexdigest()
            if seed in initial_hashes and initial_hashes[seed] != digest:
                raise RuntimeError("unmatched initialization")
            initial_hashes[seed] = digest
            directory = output / regime / arm / f"train_seed_{seed}"

            def checkpoint(m, update, steps):
                engine.save_checkpoint(
                    directory / "checkpoints" / f"step_{steps:08d}.pt",
                    m,
                    cell,
                    seed,
                    regime,
                    update,
                    steps,
                )

            checkpoint(model, 0, 0)
            training_journals = {
                k: Tagged(
                    journals[k],
                    training_regime=regime,
                    **(
                        {"training_horizon": cell["training_horizon"]}
                        if "training_horizon" in cell
                        else {}
                    ),
                )
                for k in ("training_episodes", "training_progress")
            }
            record = fit(model, cell, seed, forbidden, training_journals, checkpoint)
            engine.audit_sequence(
                sequence_hashes, {**record, "training_regime": regime}, cfg
            )
            n = record["episodes"]
            expected_mix = Counter(
                cell["training_condition_schedule"][
                    (i - 1) % len(cell["training_condition_schedule"])
                ]
                for i in range(1, n + 1)
            )
            if record["mixture_counts"] != dict(expected_mix):
                raise RuntimeError("training coverage mixture mismatch")
            coverage.append(
                {
                    **record,
                    "training_regime": regime,
                    "algorithm": arm,
                    "initial_parameters_sha256": digest,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "actor_information": model.actor_information,
                    "critic_type": model.critic_type,
                }
            )
            write_json(output / "training_coverage.json", coverage)
            shutil.copy2(
                directory / "checkpoints" / f"step_{cfg['env_steps']:08d}.pt",
                directory / "model.pt",
            )
            restored, payload = engine.load_checkpoint(directory / "model.pt")
            if payload["physical_steps"] != cfg["env_steps"]:
                raise RuntimeError("nonfinal checkpoint")
            selections[f"{regime}:{arm}:{seed}"] = dict(
                algorithm=arm,
                training_regime=regime,
                train_seed=seed,
                selected_steps=cfg["env_steps"],
                checkpoint=str((directory / "model.pt").relative_to(output)),
                checkpoint_sha256=file_hash(directory / "model.pt"),
                selection="FINAL_ONLY",
                **(
                    {"training_horizon": cell["training_horizon"]}
                    if "training_horizon" in cell
                    else {}
                ),
            )
            for cs, cond, shock in tqdm(
                list(
                    itertools.product(
                        cfg["development_cohorts"], CONDITIONS, cfg["development_seeds"]
                    )
                ),
                desc=f"final dev {regime}/{arm}/{seed}",
                leave=False,
            ):
                row = evaluate(
                    cfg,
                    cs,
                    cond,
                    cfg["horizons"][0],
                    shock,
                    arm,
                    regime,
                    seed,
                    selected_refs,
                    restored,
                )
                journals["development"].add(row)
                development_count += 1
        write_json(output / "checkpoint_selection.json", selections)
        if len({r["parameter_count"] for r in coverage}) != 1:
            raise RuntimeError("unmatched parameter counts")
        latency_count = benchmark_latency(
            output,
            cfg,
            selected_refs,
            selections,
            checkpoint_loader=engine.load_checkpoint,
        )
        # This is the only point opening the new sealed full shock trajectories.
        pa.update(
            full_test_rollouts_opened=profile == "full",
            fresh_shock_panel_opened=profile == "full",
        )
        write_json(output / "profile_audit.json", pa)
        audits["full_shock_panel_opened"] = profile == "full"
        write_json(output / "seed_audit.json", audits)
        rows = []
        jobs = [
            dict(
                training_regime="reference", algorithm=a, train_seed=-1, checkpoint=None
            )
            for a in cfg["controls"]
        ] + list(selections.values())
        panel = list(
            itertools.product(
                cfg["test_cohorts"], CONDITIONS, cfg["horizons"], cfg["test_seeds"]
            )
        )
        for job in tqdm(
            jobs,
            desc=cfg.get("test_progress_label", "test coverage controllers"),
            unit="controller",
        ):
            model = (
                engine.load_checkpoint(output / job["checkpoint"])[0]
                if job["checkpoint"]
                else None
            )
            for cs, cond, h, shock in tqdm(
                panel,
                desc=f"test {job['training_regime']}/{job['algorithm']}/{job['train_seed']}",
                leave=False,
            ):
                row = evaluate(
                    cfg,
                    cs,
                    cond,
                    h,
                    shock,
                    job["algorithm"],
                    job["training_regime"],
                    job["train_seed"],
                    selected_refs,
                    model,
                    eval_journals,
                )
                rows.append(row)
                journals["episodes"].add(row)
        journals["episodes"].close()
        shutil.copy2(output / "episodes.partial.csv", output / "episodes.csv")
        paired, summary = engine.summarize(rows, cfg)
        write_csv(output / "paired_seed_metrics.csv", paired)
        summary["audits"] = dict(
            complete_grid=True,
            feasible_actions=True,
            request_cost_reconciliation=True,
            matched_initial_parameters=True,
            training_sequence_pairing=cfg.get(
                "training_sequence_pairing", "identical_full_sequence"
            ),
            **(
                {
                    "matched_training_seed_prefix": True,
                    "matched_training_sequences_within_regime": True,
                }
                if "training_horizons" in cfg
                else {"matched_training_seed_sequences": True}
            ),
            matched_parameter_counts=True,
            exact_training_budget=True,
            checkpoint_reload=True,
            decoder="greedy",
            checkpoint_selection="final_only",
            full_tick_traces_written=profile == "smoke",
        )
        write_json(output / "summary.json", summary)
        write_json(
            output / "request_wait_decomposition.json",
            {
                f"{r}:{a}:{c}_H{h}": dict(values)
                for (r, a, c, h), values in requests.groups.items()
            },
        )
        actual = dict(
            models=len(coverage),
            training_steps=sum(r["env_steps"] for r in coverage),
            training_episodes=journals["training_episodes"].count,
            optimizer_steps=sum(r["optimizer_steps"] for r in coverage),
            checkpoint_files=len(list(output.rglob("*.pt"))),
            development_episodes=journals["development"].count,
            test_episodes=journals["episodes"].count,
            test_decision_intervals=ticks.count,
            test_machine_rows=journals["machines"].count,
            latency_rows=latency_count,
        )
        if actual != expected:
            raise RuntimeError(f"budget/population mismatch {actual} != {expected}")
        pa.update(
            full_test_rollouts_opened=profile == "full",
            fresh_shock_panel_opened=profile == "full",
        )
        write_json(output / "profile_audit.json", pa)
        manifest.update(
            status="COMPLETED", actual_counts=actual, request_rows=requests.count
        )
    except BaseException as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = {
            str(p.relative_to(output)): dict(
                bytes=p.stat().st_size, sha256=file_hash(p)
            )
            for p in output.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    print(
        json.dumps(dict(output=str(output), status=manifest["status"], counts=actual))
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.output_dir
        or ROOT
        / "artifacts"
        / f"{VERSION}_{args.profile}_cpu_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    run(output.resolve(), args.profile)


if __name__ == "__main__":
    main()
