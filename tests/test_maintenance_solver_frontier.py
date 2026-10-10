import csv
from dataclasses import asdict
import itertools
import json

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_solver_frontier as r
from ht_pdm_fjsp import maintenance_frontier_planner as planner
from ht_pdm_fjsp import maintenance_frontier_statistics as stats
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION, WaitingEnv, transition
from ht_pdm_fjsp.maintenance_solver_policy import FEATURE_CONTRACT, CENTRAL
from ht_pdm_fjsp.maintenance_solver_model import (
    cohort_config,
    reference,
    profile_id,
    profile_partitions,
)


def test_frozen_counts_and_registered_seeds():
    cfg = r.settings("full")
    assert r.counts(cfg) == dict(
        new_training_steps=0,
        frozen_models=10,
        development_episodes=20480,
        test_episodes=71680,
        test_decision_intervals=1290240,
        test_machine_rows=286720,
        common_states=2304,
        warm_latency_rows=516096,
        cold_latency_rows=112,
    )
    assert cfg["cost_margin"] == 0.02 and cfg["planner_budgets"] == [8, 32, 128]
    assert r.seed_audit(cfg)["status"] == "PASS"
    cfg["test_seeds"] = [185000]
    with pytest.raises(ValueError, match="unregistered"):
        r.seed_audit(cfg)
    for cs in [165010, 165020]:
        assert profile_id(cs, profile="smoke") in profile_partitions()["train"]


def test_nested_forecasts_at_every_time():
    c = cohort_config(165020, "skill_mask", 6, profile="smoke")
    large = planner.forecasts(c, 6, 1024, 128, 128)
    for budget in [8, 32]:
        assert np.array_equal(
            planner.forecasts(c, 6, 1024, budget, 128), large[:, :budget]
        )
    with pytest.raises(ValueError, match="budget"):
        planner.forecasts(c, 6, 1024, 1, 128)


@pytest.mark.parametrize("condition", list(r.CONDITIONS))
def test_vectorized_planner_matches_scalar_physics_and_scores(condition):
    c = cohort_config(165020, condition, 6, profile="smoke")
    state = DispatchState(
        ages=(3, 4, 5, 1),
        failed=(True, False, True, False),
        pending_wait=(6, 0, 4, 0),
        assigned=(-1,) * c.technicians,
        remaining=(0,) * c.technicians,
    )
    events = planner.forecasts(c, 6, 9024, 4, 8)
    for continuation in r.REFERENCES:
        candidates, scores = planner.score_rollouts(c, state, 6, continuation, events)
        expected = []
        for root in candidates:
            vals = []
            for s in range(events.shape[1]):
                current, total = state, 0
                for t in range(6):
                    action = (
                        root if t == 0 else reference(c, current, 6 - t, continuation)
                    )
                    current, cost, _ = transition(c, current, action, events[t, s])
                    total += cost
                vals.append(total)
            expected.append(vals)
        assert np.array_equal(scores, expected)
        chosen, diag = planner.plan(c, state, 6, continuation, 9024, 4, 8)
        assert chosen == candidates[int(np.asarray(expected).mean(1).argmin())]
        assert diag["scenarios"] == 4


def test_bootstrap_pairs_shocks_profiles_and_handles_replica_variation():
    right = np.array([[1, 100], [30, 3000]], float)
    left = np.stack([right * 2, right * 2, right * 2])
    assert np.allclose(stats.bootstrap_ratio(left, right, 1000, 52), 2)
    varying = np.stack([right, right * 2, right * 3])
    one = stats.bootstrap_ratio(varying, right, 1000, 52)
    assert np.array_equal(one, stats.bootstrap_ratio(varying, right, 1000, 52))
    assert one.min() < 2 < one.max()
    with pytest.raises(ValueError, match="dimensions"):
        stats.bootstrap_ratio(left, np.zeros_like(right), 100, 52)


@pytest.mark.parametrize(
    "upper,closed,common,wait,pending,status",
    [
        (1.02, 0.4, 0.4, 0.02, 0.05, "LOWER_COMPUTE_TRADEOFF_SCREEN"),
        (1.021, 0.4, 0.4, 0, 0, "INCONCLUSIVE_OR_NO_SCREENED_ADVANTAGE"),
        (1.01, 0.6, 0.4, 0, 0, "INCONCLUSIVE_OR_NO_SCREENED_ADVANTAGE"),
        (1.01, 0.4, 0.6, 0, 0, "INCONCLUSIVE_OR_NO_SCREENED_ADVANTAGE"),
        (1.01, 0.4, 0.4, 0.021, 0, "INCONCLUSIVE_OR_NO_SCREENED_ADVANTAGE"),
        (1.01, 0.4, 0.4, 0, 0.051, "INCONCLUSIVE_OR_NO_SCREENED_ADVANTAGE"),
        (0.98, 2, 2, 0, 0, "QUALITY_ADVANTAGE_SCREEN"),
    ],
)
def test_joint_directional_screens(upper, closed, common, wait, pending, status):
    got = stats.classify(1.0, upper, closed, common, wait, pending, r.settings("full"))
    assert got["status"] == status


def synthetic():
    cfg = r.settings("full")
    cfg.update(
        test_cohorts=[163000, 163001], test_seeds=[205000, 205001], bootstrap_draws=1000
    )
    rows = []
    common = {}
    for a, seeds in [(CENTRAL, cfg["model_seeds"])] + [
        (a, [-1]) for a in cfg["controls"]
    ]:
        for s, p, c, h, q in itertools.product(
            seeds,
            cfg["test_cohorts"],
            cfg["conditions"],
            cfg["horizons"],
            cfg["test_seeds"],
        ):
            row = {m: 0 for m in stats.METRICS}
            row.update(
                controller=a,
                train_seed=s,
                cohort=p,
                condition=c,
                horizon=h,
                seed=q,
                objective=100 if a == CENTRAL else 101,
                inference_seconds=h * (0.1 if a == CENTRAL else 0.4),
            )
            rows.append(row)
            common[(a, c, h)] = {"mean_seconds": 0.1 if a == CENTRAL else 0.4}
    return cfg, rows, common


def test_all_cells_budgets_retained_and_grid_rejection():
    cfg, rows, common = synthetic()
    paired, summary = stats.summarize(rows, cfg, common)
    assert len(summary["comparisons"]) == 32 and len(paired) == 320
    assert all(
        v["multiplicity_family"] == 32
        and v["nominal_confidence"] == 0.9984375
        and v["status"] == "LOWER_COMPUTE_TRADEOFF_SCREEN"
        for v in summary["comparisons"].values()
    )
    assert len(summary["aggregates"]) == 40
    for bad in [rows[:-1], rows + rows[:1]]:
        with pytest.raises(ValueError, match="grid"):
            stats.summarize(bad, cfg, common)


@pytest.fixture
def source_fixture(tmp_path):
    # Synthetic final-checkpoint schema for engineering tests, NOT trained evidence.
    source = tmp_path / "synthetic_source"
    source.mkdir()
    cfg = r.coverage.cell_settings(r.coverage.settings("full"), "covered")
    m = r.base.model_for(cfg, 180000, CENTRAL)
    rel = f"covered/{CENTRAL}/train_seed_180000/model.pt"
    p = source / rel
    p.parent.mkdir(parents=True)
    torch.save(
        dict(
            protocol=r.coverage.VERSION,
            environment_version=ENV_VERSION,
            feature_contract=FEATURE_CONTRACT,
            algorithm=CENTRAL,
            train_seed=180000,
            training_regime="covered",
            rollout_update=320,
            physical_steps=491520,
            settings=cfg,
            state_dict=m.state_dict(),
        ),
        p,
    )
    selection = {
        f"covered:{CENTRAL}:180000": dict(
            checkpoint=rel,
            selection="FINAL_ONLY",
            selected_steps=491520,
            checkpoint_sha256=r.file_hash(p),
        )
    }
    r.write_json(source / "checkpoint_selection.json", selection)
    r.write_json(
        source / "training_coverage.json",
        [
            dict(
                training_regime="covered",
                algorithm=CENTRAL,
                train_seed=180000,
                env_steps=491520,
                training_wall_seconds=0,
            )
        ],
    )
    manifest = dict(
        status="COMPLETED",
        experiment=r.coverage.VERSION,
        commit=r.SOURCE_COMMIT,
        dirty="",
        config=r.coverage.settings("full"),
        actual_counts=r.coverage.counts(r.coverage.settings("full")),
        runtime={"test_fixture": True},
    )
    manifest["artifacts"] = {
        str(p.relative_to(source)): dict(bytes=p.stat().st_size, sha256=r.file_hash(p))
        for p in source.rglob("*")
        if p.is_file()
    }
    r.write_json(source / "manifest.json", manifest)
    return source


def test_preflight_hash_corruption_and_nonfinal_selection(source_fixture):
    cfg = r.settings("smoke")
    jobs, info = r.preflight(source_fixture, cfg)
    assert len(jobs) == 1 and info["relevant_artifact_hashes_verified"]
    path = source_fixture / jobs[0]["checkpoint"]
    with path.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash/size"):
        r.preflight(source_fixture, cfg)


def test_nonfinal_source_rejected_after_valid_hash(source_fixture):
    path = source_fixture / "checkpoint_selection.json"
    selection = json.loads(path.read_text())
    next(iter(selection.values()))["selection"] = "DEVELOPMENT_BEST"
    r.write_json(path, selection)
    m = json.loads((source_fixture / "manifest.json").read_text())
    m["artifacts"][path.name] = dict(
        bytes=path.stat().st_size, sha256=r.file_hash(path)
    )
    r.write_json(source_fixture / "manifest.json", m)
    with pytest.raises(ValueError, match="final checkpoint"):
        r.preflight(source_fixture, r.settings("smoke"))


def test_mac_full_rejected_before_source_or_test_open(tmp_path, monkeypatch):
    monkeypatch.setattr(r.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="human-run"):
        r.run(tmp_path / "full", tmp_path / "missing", "full")
    assert not (tmp_path / "full").exists()


def test_failed_run_keeps_manifest_and_partial_files(
    source_fixture, tmp_path, monkeypatch
):
    def fail(*args):
        raise RuntimeError("injected development failure")

    monkeypatch.setattr(r, "select_development", fail)
    out = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="injected"):
        r.run(out, source_fixture, "smoke")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "FAILED" and (out / "episodes.partial.csv").exists()
    assert "injected" in manifest["error"]


def test_integrated_smoke_counts_sealing_hashes_and_physical_replay(
    source_fixture, tmp_path
):
    out = tmp_path / "smoke"
    r.run(out, source_fixture, "smoke")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and manifest["actual_counts"] == r.counts(
        r.settings("smoke")
    )
    assert manifest["actual_counts"]["new_training_steps"] == 0
    assert not json.loads((out / "seed_audit.json").read_text())[
        "fresh_shock_panel_opened"
    ]
    assert not json.loads((out / "profile_audit.json").read_text())[
        "full_test_rollouts_opened"
    ]
    for file, meta in manifest["artifacts"].items():
        assert (out / file).stat().st_size == meta["bytes"] and r.file_hash(
            out / file
        ) == meta["sha256"]
    for row in csv.DictReader((out / "decisions.csv").open()):
        c = cohort_config(
            int(row["cohort"]), row["condition"], int(row["horizon"]), profile="smoke"
        )
        env = WaitingEnv(c)
        env.reset(int(row["seed"]))
        state = DispatchState(
            **{k: tuple(v) for k, v in json.loads(row["state"]).items()}
        )
        action = tuple(tuple(v) for v in json.loads(row["action"]))
        following, _, info = transition(c, state, action, env.events[int(row["time"])])
        assert json.loads(json.dumps(asdict(following))) == json.loads(row["following"])
        assert info == json.loads(row["physical_metrics"])
    summary = json.loads((out / "summary.json").read_text())
    assert all(
        v["status"] == "ENGINEERING_ONLY" for v in summary["comparisons"].values()
    )
    for file in ["quality_compute.png", "quality_compute.pdf", "resource_usage.csv"]:
        assert (out / file).stat().st_size > 0
    with pytest.raises(ValueError, match="new/empty"):
        r.run(out, source_fixture, "smoke")
