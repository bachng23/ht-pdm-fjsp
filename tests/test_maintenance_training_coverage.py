import csv
from dataclasses import asdict
import hashlib
import itertools
import json
from collections import Counter

import pytest

from ht_pdm_fjsp import maintenance_training_coverage as r
from ht_pdm_fjsp.maintenance_solver_training import episode_spec
from ht_pdm_fjsp.maintenance_solver_model import profile_partitions, profile_id
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


def test_training_schedule_is_the_only_episode_intervention():
    for i in range(1, 31):
        old = episode_spec(189000, i, 4, set(), "smoke")
        legacy = episode_spec(189000, i, 4, set(), "smoke", r.SCHEDULES["legacy"])
        covered = episode_spec(189000, i, 4, set(), "smoke", r.SCHEDULES["covered"])
        assert old == legacy
        assert old[2:] == covered[2:]
        assert legacy[1] == ("nominal" if i % 5 == 0 else "specialized")
        if i % 5 in (0, 1, 3):
            assert legacy == covered
        else:
            assert covered[1] == "skill_mask" and legacy[1] == "specialized"
    for schedule in ([], ["unknown"]):
        with pytest.raises(ValueError, match="schedule"):
            episode_spec(189000, 1, 4, set(), "smoke", schedule)
    x = episode_spec(189000, 1, 4, set(), "smoke")
    with pytest.raises(ValueError, match="overlap"):
        episode_spec(189000, 1, 4, {x[2]}, "smoke")


def test_locked_counts_seed_panels_and_no_full_smoke_profiles():
    cfg = r.settings("full")
    assert r.counts(cfg) == dict(
        models=60,
        training_steps=29491200,
        training_episodes=2457600,
        optimizer_steps=614400,
        checkpoint_files=360,
        development_episodes=40960,
        test_episodes=317440,
        test_decision_intervals=5713920,
        test_machine_rows=1269760,
        latency_rows=8928,
    )
    assert r.seed_audit(cfg)[0]["status"] == "PASS"
    smoke = r.settings("smoke")
    assert r.seed_audit(smoke)[0]["status"] == "PASS"
    assert r.counts(smoke)["test_episodes"] == 128
    for cs in smoke["development_cohorts"] + smoke["test_cohorts"]:
        assert profile_id(cs, profile="smoke") in profile_partitions()["train"]
    cfg["test_seeds"] = [171000]
    with pytest.raises(ValueError, match="registered"):
        r.seed_audit(cfg)


def synthetic():
    cfg = r.settings("full")
    cfg["test_cohorts"] = cfg["test_cohorts"][:1]
    cfg["test_seeds"] = cfg["test_seeds"][:1]
    costs = {
        "legacy": {r.CENTRAL: 100, r.LOCAL: 140, r.MARL: 150},
        "covered": {r.CENTRAL: 90, r.LOCAL: 110, r.MARL: 115},
    }
    rows = []
    jobs = [
        (reg, a, cfg["train_seeds"])
        for reg, a in itertools.product(r.SCHEDULES, r.ARMS)
    ] + [("reference", a, [-1]) for a in cfg["controls"]]
    for reg, a, seeds in jobs:
        for s, cs, c, h, shock in itertools.product(
            seeds, cfg["test_cohorts"], r.CONDITIONS, cfg["horizons"], cfg["test_seeds"]
        ):
            cost = costs[reg][a] if reg != "reference" else 100.0
            row = {m: 0.0 for m in r.METRICS}
            row.update(
                training_regime=reg,
                controller=a,
                train_seed=s,
                cohort=cs,
                condition=c,
                horizon=h,
                seed=shock,
                objective=cost,
                base_cost=cost,
            )
            rows.append(row)
    return cfg, rows


def test_difference_in_differences_four_primary_and_grid_rejection():
    cfg, rows = synthetic()
    paired, summary = r.summarize(rows, cfg)
    primary = [v for v in summary["contrasts"].values() if v["primary"]]
    assert len(primary) == 4 and all(v["status"] == "PASS" for v in primary)
    assert all(
        v["confidence"] == 0.9875 and v["unit"] == "train_seed" and v["n"] == 10
        for v in primary
    )
    d = summary["contrasts"][f"gap_closure:{r.LOCAL}:skill_mask_H24"]
    assert d["mean_difference"] == -20 and d["gap_fraction_closed"] == 0.5
    assert d["legacy_gap_to_central"] == 40 and d["covered_gap_to_central"] == 20
    for bad in (rows[:-1], rows + rows[:1]):
        with pytest.raises(RuntimeError, match="grid"):
            r.summarize(bad, cfg)


def test_equal_central_and_cooperative_gains_do_not_close_gap():
    cfg, rows = synthetic()
    for row in rows:
        if row["training_regime"] == "covered":
            row["objective"] = {r.CENTRAL: 90, r.LOCAL: 130, r.MARL: 140}[
                row["controller"]
            ]
    summary = r.summarize(rows, cfg)[1]
    for a in (r.LOCAL, r.MARL):
        d = summary["contrasts"][f"gap_closure:{a}:skill_mask_H24"]
        assert (
            d["mean_difference"] == 0
            and d["gap_fraction_closed"] == 0
            and d["status"] == "FAIL"
        )


def test_nonpositive_old_gap_is_not_applicable():
    cfg, rows = synthetic()
    for row in rows:
        if row["training_regime"] == "legacy" and row["controller"] == r.LOCAL:
            row["objective"] = 95
    d = r.summarize(rows, cfg)[1]["contrasts"][f"gap_closure:{r.LOCAL}:skill_mask_H24"]
    assert d["status"] == "NOT_APPLICABLE" and d["gap_fraction_closed"] is None


def test_request_decomposition_catches_survivorship(tmp_path):
    j = r.Journal(tmp_path / "requests.csv")
    q = r.RequestJournal(j)
    tags = dict(
        training_regime="covered",
        controller=r.LOCAL,
        condition="skill_mask",
        horizon=24,
    )
    q.add(dict(**tags, wait=3, arrival=0, end=3, served=True, censored=False))
    q.add(dict(**tags, wait=20, arrival=4, end=24, served=False, censored=True))
    assert (
        q.episode["served_waiting_cost"] == 0
        and q.episode["censored_waiting_cost"] == 192
    )
    assert q.episode["overdue_ticks"] == 16
    with pytest.raises(RuntimeError, match="censoring"):
        q.add(dict(**tags, wait=3, arrival=0, end=3, served=True, censored=True))
    j.close()


def test_failed_run_records_manifest_and_partial_files(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected coverage training failure")

    monkeypatch.setattr(r, "fit", fail)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        r.run(out, "smoke")
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "FAILED" and "injected" in m["error"]
    assert (out / "episodes.partial.csv").exists()


def test_integrated_smoke_final_only_and_replay(tmp_path):
    out = tmp_path / "smoke"
    r.run(out, "smoke")
    cfg = r.settings("smoke")
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "COMPLETED" and m["actual_counts"] == r.counts(cfg)
    assert not json.loads((out / "seed_audit.json").read_text())[
        "full_shock_panel_opened"
    ]
    for file, meta in m["artifacts"].items():
        assert hashlib.sha256((out / file).read_bytes()).hexdigest() == meta["sha256"]
    coverage = json.loads((out / "training_coverage.json").read_text())
    assert len({v["initial_parameters_sha256"] for v in coverage}) == 1
    assert len({v["training_seed_sequence_sha256"] for v in coverage}) == 1
    selections = json.loads((out / "checkpoint_selection.json").read_text())
    assert len(selections) == 6 and all(
        v["selection"] == "FINAL_ONLY" and v["selected_steps"] == 256
        for v in selections.values()
    )
    for v in selections.values():
        model, p = r.load_checkpoint(out / v["checkpoint"])
        assert p["training_regime"] == v["training_regime"]
        assert (
            p["settings"]["training_condition_schedule"]
            == r.SCHEDULES[v["training_regime"]]
        )
    training = list(csv.DictReader((out / "training_episodes.csv").open()))
    for a in r.ARMS:
        old = [
            v
            for v in training
            if v["training_regime"] == "legacy" and v["algorithm"] == a
        ]
        new = [
            v
            for v in training
            if v["training_regime"] == "covered" and v["algorithm"] == a
        ]
        assert len(old) == len(new) == 64
        assert [(v["config_seed"], v["environment_seed"]) for v in old] == [
            (v["config_seed"], v["environment_seed"]) for v in new
        ]
        assert Counter(v["condition"] for v in new) == Counter(
            skill_mask=26, specialized=26, nominal=12
        )
    for row in csv.DictReader((out / "decisions.csv").open()):
        c = r.cohort_config(
            int(row["cohort"]), row["condition"], int(row["horizon"]), profile="smoke"
        )
        state = DispatchState(
            **{k: tuple(v) for k, v in json.loads(row["state"]).items()}
        )
        env = WaitingEnv(c)
        env.reset(int(row["seed"]))
        action = tuple(tuple(v) for v in json.loads(row["action"]))
        following, cost, info = transition(
            c, state, action, env.events[int(row["time"])]
        )
        assert json.loads(json.dumps(asdict(following))) == json.loads(row["following"])
        assert info == json.loads(row["physical_metrics"])
    summary = json.loads((out / "summary.json").read_text())
    assert all(v["status"] == "ENGINEERING_ONLY" for v in summary["contrasts"].values())
    assert (
        len(list(csv.DictReader((out / "machine_metrics.csv").open())))
        == r.counts(cfg)["test_machine_rows"]
    )
    with pytest.raises(ValueError, match="new/empty"):
        r.run(out, "smoke")


def test_preservation_guards_prevent_mask_only_win():
    cfg, rows = synthetic()
    for row in rows:
        if (
            row["training_regime"] == "covered"
            and row["controller"] == r.LOCAL
            and row["condition"] == "nominal"
        ):
            row["objective"] = 200
    d = r.summarize(rows, cfg)[1]["contrasts"][f"coverage:{r.LOCAL}:skill_mask_H24"]
    assert d["gates"]["practical_effect"] and not d["gates"]["nominal_preserved"]
    assert d["status"] == "FAIL"


def test_full_cannot_open_lab_panel_on_mac(tmp_path, monkeypatch):
    monkeypatch.setattr(r.platform, "system", lambda: "Darwin")
    out = tmp_path / "forbidden_full"
    with pytest.raises(ValueError, match="human-run"):
        r.run(out, "full")
    assert not out.exists()
