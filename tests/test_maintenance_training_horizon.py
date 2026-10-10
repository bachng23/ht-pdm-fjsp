import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_training_horizon as r
from ht_pdm_fjsp.maintenance_solver_training import episode_spec, fit, monte_carlo
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_solver_model import profile_partitions, profile_id
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


def test_budget_equal_ticks_updates_and_explicit_unequal_episodes():
    cfg = r.settings("full")
    assert r.counts(cfg) == dict(
        models=60,
        training_steps=29491200,
        training_episodes=1843200,
        optimizer_steps=614400,
        checkpoint_files=360,
        development_episodes=40960,
        test_episodes=317440,
        test_decision_intervals=5713920,
        test_machine_rows=1269760,
        latency_rows=26784,
    )
    assert r.seed_audit(cfg)[0]["status"] == "PASS"
    for regime, h in [("h12", 12), ("h24", 24)]:
        c = r.cell_settings(cfg, regime)
        assert c["training_horizon"] == h and c["env_steps"] == 491520
        assert c["paired_training_episodes"] == 20480
        assert c["rollout"] % (c["vector_envs"] * h) == 0
        assert c["env_steps"] // h in (40960, 20480)
    cfg["test_seeds"] = [185000]
    with pytest.raises(ValueError, match="registered"):
        r.seed_audit(cfg)


def test_training_config_and_shock_prefixes_match_excluding_horizon():
    cfg = r.settings("full")
    forbidden = r.seed_audit(cfg)[1]
    for i in range(1, 11):
        short = episode_spec(190000, i, 12, forbidden, "full", r.SCHEDULES["h12"])
        long = episode_spec(190000, i, 24, forbidden, "full", r.SCHEDULES["h24"])
        assert short[1:] == long[1:]
        assert replace(short[0], horizon=24) == long[0]
        e1 = WaitingEnv(short[0])
        e2 = WaitingEnv(long[0])
        e1.reset(short[3])
        e2.reset(long[3])
        assert np.array_equal(e1.events, e2.events[:12])
        # Training configurations belong to original TRAIN profile pool only.
        assert (
            profile_id(short[2], split="train", profile="full")
            in profile_partitions()["train"]
        )


def test_prefix_audit_checks_both_pairing_levels():
    cfg = r.settings("smoke")
    seen = {}

    def record(reg, full, prefix):
        return dict(
            train_seed=199000,
            training_regime=reg,
            episodes=256 // cfg["training_horizons"][reg],
            shared_prefix_episodes=32,
            training_seed_sequence_sha256=full,
            training_shared_prefix_sha256=prefix,
        )

    r.audit_sequence(seen, record("h12", "long", "shared"), cfg)
    r.audit_sequence(seen, record("h24", "short", "shared"), cfg)
    assert len(seen) == 3
    with pytest.raises(RuntimeError, match="prefix/sequence"):
        r.audit_sequence(seen, record("h24", "changed", "shared"), cfg)
    with pytest.raises(RuntimeError, match="prefix/sequence"):
        r.audit_sequence(seen, record("h12", "long", "changed"), cfg)


class MemoryJournal:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)


def test_actual12and24_fit_complete_episode_returns_same_budget():
    torch.set_num_threads(1)
    result = []
    for reg, h in [("h12", 12), ("h24", 24)]:
        cfg = r.cell_settings(r.settings("full"), reg)
        cfg.update(
            env_steps=1536,
            checkpoint_updates=[1],
            epochs=1,
            hidden=16,
            paired_training_episodes=64,
        )
        m = r.shared.base.model_for(cfg, 190000, r.CENTRAL)
        journals = {
            k: MemoryJournal() for k in ("training_episodes", "training_progress")
        }
        checkpoints = []
        record = fit(
            m, cfg, 190000, set(), journals, lambda _, u, s: checkpoints.append((u, s))
        )
        assert (
            record["env_steps"] == 1536
            and record["optimizer_steps"] == 8
            and record["episodes"] == 1536 // h
        )
        assert record["training_horizon"] == h and checkpoints == [(1, 1536)]
        assert len(journals["training_episodes"].rows) == 1536 // h
        assert (
            record["complete_episode_returns"] and record["sampled_probability_audit"]
        )
        assert all(
            v["unavailable_ticks"] <= 4 * h for v in journals["training_episodes"].rows
        )
        result.append(record)
    assert (
        result[0]["training_shared_prefix_sha256"]
        == result[1]["training_shared_prefix_sha256"]
    )
    assert (
        result[0]["training_seed_sequence_sha256"]
        != result[1]["training_seed_sequence_sha256"]
    )
    rewards = torch.arange(24 * 2, dtype=torch.float32).reshape(24, 2)
    assert torch.equal(monte_carlo(rewards)[0], rewards.sum(0))
    assert torch.equal(monte_carlo(rewards)[-1], rewards[-1])


def synthetic():
    cfg = r.settings("full")
    cfg["test_cohorts"] = cfg["test_cohorts"][:1]
    cfg["test_seeds"] = cfg["test_seeds"][:1]
    costs = {
        "h12": {r.CENTRAL: 100, r.LOCAL: 140, r.MARL: 150},
        "h24": {r.CENTRAL: 90, r.LOCAL: 110, r.MARL: 115},
    }
    jobs = [
        (reg, a, cfg["train_seeds"])
        for reg, a in itertools.product(r.SCHEDULES, r.ARMS)
    ] + [("reference", a, [-1]) for a in cfg["controls"]]
    rows = []
    for reg, a, seeds in jobs:
        for seed, cs, c, h, shock in itertools.product(
            seeds,
            cfg["test_cohorts"],
            cfg["conditions"],
            cfg["horizons"],
            cfg["test_seeds"],
        ):
            row = {m: 0.0 for m in r.shared.METRICS}
            row.update(
                training_regime=reg,
                training_horizon=cfg["training_horizons"].get(reg, -1),
                controller=a,
                train_seed=seed,
                cohort=cs,
                condition=c,
                horizon=h,
                seed=shock,
                objective=costs[reg][a] if reg != "reference" else 100.0,
            )
            rows.append(row)
    return cfg, rows


def test_four_primary_estimands_and_output_identity():
    cfg, rows = synthetic()
    paired, summary = r.summarize(rows, cfg)
    primary = [v for v in summary["contrasts"].values() if v["primary"]]
    assert len(primary) == 4 and all(v["status"] == "PASS" for v in primary)
    assert all(
        v["confidence"] == 0.9875 and v["n"] == 10 and v["unit"] == "train_seed"
        for v in primary
    )
    v = summary["contrasts"][f"gap_closure:{r.LOCAL}:skill_mask_H24"]
    assert v["mean_difference"] == -20 and v["gap_fraction_closed"] == 0.5
    assert (
        v["short_training_gap_to_central"] == 40
        and v["long_training_gap_to_central"] == 20
    )
    assert all("legacy:" not in k and "covered:" not in k for k in summary["aggregate"])
    assert all("coverage:" not in x["contrast"] for x in paired)
    for bad in [rows[:-1], rows + rows[:1]]:
        with pytest.raises(RuntimeError, match="grid"):
            r.summarize(bad, cfg)
    wrong = [dict(v) for v in rows]
    wrong[0]["training_horizon"] = 24
    with pytest.raises(RuntimeError, match="identity"):
        r.summarize(wrong, cfg)


@pytest.mark.parametrize(
    "metric,value,gate",
    [
        ("objective", 200.0, "homogeneous_cost_preserved"),
        ("waiting_violation", 0.03, "homogeneous_waiting_guard"),
        ("terminal_pending", 0.06, "homogeneous_pending_guard"),
    ],
)
def test_homogeneous_guards_can_block_mask_success(metric, value, gate):
    cfg, rows = synthetic()
    for row in rows:
        if (
            row["training_regime"] == "h24"
            and row["controller"] == r.LOCAL
            and row["condition"] == "homogeneous"
            and row["horizon"] == 24
        ):
            row[metric] = value
    summary = r.summarize(rows, cfg)[1]
    for label in ("horizon", "gap_closure"):
        v = summary["contrasts"][f"{label}:{r.LOCAL}:skill_mask_H24"]
        assert (
            v["gates"]["practical_effect"]
            and not v["gates"][gate]
            and v["status"] == "FAIL"
        )


def test_equal_central_gain_does_not_close_gap():
    cfg, rows = synthetic()
    for row in rows:
        if row["training_regime"] == "h24":
            row["objective"] = {r.CENTRAL: 90, r.LOCAL: 130, r.MARL: 140}[
                row["controller"]
            ]
    v = r.summarize(rows, cfg)[1]["contrasts"][f"gap_closure:{r.LOCAL}:skill_mask_H24"]
    assert v["mean_difference"] == 0 and v["status"] == "FAIL"


def test_failed_manifest_preserves_horizon_identity(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected horizon training failure")

    monkeypatch.setattr(r.shared, "fit", fail)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        r.run(out, "smoke")
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "FAILED" and m["experiment"] == r.VERSION
    assert "injected" in m["error"] and (out / "episodes.partial.csv").exists()


def test_mac_full_forbidden(tmp_path, monkeypatch):
    monkeypatch.setattr(r.shared.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="human-run"):
        r.run(tmp_path / "full", "full")
    assert not (tmp_path / "full").exists()


def test_integrated_smoke_and_replay(tmp_path):
    out = tmp_path / "smoke"
    r.run(out, "smoke")
    cfg = r.settings("smoke")
    m = json.loads((out / "manifest.json").read_text())
    assert (
        m["status"] == "COMPLETED"
        and m["actual_counts"] == r.counts(cfg)
        and m["experiment"] == r.VERSION
    )
    assert not json.loads((out / "seed_audit.json").read_text())[
        "full_shock_panel_opened"
    ]
    assert not json.loads((out / "profile_audit.json").read_text())[
        "full_test_rollouts_opened"
    ]
    for f, meta in m["artifacts"].items():
        assert hashlib.sha256((out / f).read_bytes()).hexdigest() == meta["sha256"]
    coverage = json.loads((out / "training_coverage.json").read_text())
    assert len({v["initial_parameters_sha256"] for v in coverage}) == 1
    assert len({v["training_shared_prefix_sha256"] for v in coverage}) == 1
    for reg, h, n in [("h12", 4, 64), ("h24", 8, 32)]:
        entries = [v for v in coverage if v["training_regime"] == reg]
        assert (
            len(entries) == 3
            and len({v["training_seed_sequence_sha256"] for v in entries}) == 1
        )
        assert all(
            v["training_horizon"] == h
            and v["episodes"] == n
            and v["env_steps"] == 256
            and v["optimizer_steps"] == 16
            for v in entries
        )
    choices = json.loads((out / "checkpoint_selection.json").read_text())
    for v in choices.values():
        model, p = r.load_checkpoint(out / v["checkpoint"])
        assert (
            p["training_horizon"]
            == v["training_horizon"]
            == cfg["training_horizons"][v["training_regime"]]
        )
        assert p["physical_steps"] == 256 and v["selection"] == "FINAL_ONLY"
        with pytest.raises(ValueError, match="coverage checkpoint"):
            r.shared.load_checkpoint(out / v["checkpoint"])
    payload = torch.load(
        out / next(iter(choices.values()))["checkpoint"], weights_only=True
    )
    payload["training_horizon"] = 8
    torch.save(payload, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="contract"):
        r.load_checkpoint(tmp_path / "wrong.pt")
    for row in csv.DictReader((out / "decisions.csv").open()):
        assert int(row["training_horizon"]) == cfg["training_horizons"].get(
            row["training_regime"], -1
        )
        c = r.cohort_config(
            int(row["cohort"]), row["condition"], int(row["horizon"]), profile="smoke"
        )
        env = WaitingEnv(c)
        env.reset(int(row["seed"]))
        state = DispatchState(
            **{k: tuple(v) for k, v in json.loads(row["state"]).items()}
        )
        action = tuple(tuple(v) for v in json.loads(row["action"]))
        following, cost, info = transition(
            c, state, action, env.events[int(row["time"])]
        )
        assert json.loads(json.dumps(asdict(following))) == json.loads(row["following"])
        assert info == json.loads(row["physical_metrics"])
    for filename in [
        "episodes.csv",
        "machine_metrics.csv",
        "request_metrics.csv",
        "training_episodes.csv",
        "training_progress.csv",
        "development_episodes.csv",
        "latency.csv",
    ]:
        assert "training_horizon" in next(csv.reader((out / filename).open()))
    assert all(
        v["status"] == "ENGINEERING_ONLY"
        for v in json.loads((out / "summary.json").read_text())["contrasts"].values()
    )
    with pytest.raises(ValueError, match="new/empty"):
        r.run(out, "smoke")


def test_nonpositive_baseline_gap_not_applicable():
    cfg, rows = synthetic()
    for row in rows:
        if row["training_regime"] == "h12" and row["controller"] == r.LOCAL:
            row["objective"] = 95
    v = r.summarize(rows, cfg)[1]["contrasts"][f"gap_closure:{r.LOCAL}:skill_mask_H24"]
    assert v["status"] == "NOT_APPLICABLE" and v["gap_fraction_closed"] is None


@pytest.mark.parametrize(
    "condition,gate",
    [("nominal", "nominal_preserved"), ("specialized", "specialized_preserved")],
)
def test_short_horizon_preservation_guard(condition, gate):
    cfg, rows = synthetic()
    for row in rows:
        if (
            row["training_regime"] == "h24"
            and row["controller"] == r.LOCAL
            and row["condition"] == condition
            and row["horizon"] == 12
        ):
            row["objective"] = 200
    v = r.summarize(rows, cfg)[1]["contrasts"][f"horizon:{r.LOCAL}:skill_mask_H24"]
    assert v["status"] == "FAIL" and not v["gates"][gate]
