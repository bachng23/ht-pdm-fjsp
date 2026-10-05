import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from ht_pdm_fjsp import maintenance_priority_allocation as experiment
from ht_pdm_fjsp.maintenance_dispatch import DispatchConfig, family_config
from ht_pdm_fjsp.maintenance_dispatch_policy import MODES


def test_monte_carlo_returns_stop_at_episode_boundaries():
    result = experiment.returns_for_episodes(
        [-1.0, -2.0, -3.0, -4.0], [False, True, False, True]
    )
    assert result.tolist() == [-3.0, -2.0, -7.0, -4.0]
    with pytest.raises(RuntimeError, match="complete episode"):
        experiment.returns_for_episodes([-1.0], [False])


def test_locked_full_budget_and_seed_panels():
    cfg = experiment.settings("full")
    assert cfg["env_steps"] == 120000 and len(cfg["train_seeds"]) == 10
    assert (
        cfg["env_steps"]
        // cfg["rollout"]
        * cfg["epochs"]
        * cfg["rollout"]
        // cfg["minibatch"]
        == 4000
    )
    audit = experiment.seed_audit(cfg)
    assert (
        audit["panels_disjoint"]
        and audit["prior_declared_disjoint"]
        and audit["sealed_panels_closed"]
    )


def test_exact_oracle_has_independently_known_boundary_cost():
    # p=1, threshold1: from initial age0, no-service fails for cost15;
    # preventive service duration1 costs1+6 and finishes at the boundary.
    cfg = DispatchConfig(2, 2, 1, 1, 1.0, ((1, 1), (1, 1)), ((1.0, 1.0), (1.0, 1.0)))
    oracle = experiment.SmallOracle(cfg)
    assert oracle.optimum == 14.0
    assert oracle.evaluate("risk_skill_rule")["exact_policy_cost"] == 14.0


def test_full_exact_small_graph_fits_cap_and_bellman_audit(tmp_path):
    oracle = experiment.SmallOracle(family_config("small", 0, "full"))
    oracle.save(tmp_path)
    audit = json.loads((tmp_path / "small_oracle_audit.json").read_text())
    assert audit["initials"] == 9 and audit["states"] < audit["max_states"]
    assert audit["bellman_residual"] < 1e-9
    result = oracle.evaluate("risk_skill_rule")
    assert result["optimality_gap"] >= 0
    assert (
        result["exact_selection_regret"] >= 0 and result["exact_allocation_regret"] >= 0
    )
    assert result["exact_selection_regret"] + result[
        "exact_allocation_regret"
    ] == pytest.approx(result["optimality_gap"])


def test_exact_stop_policy_gap_is_selection_regret():
    class StopPolicy:
        def act(self, *args, **kwargs):
            return (), [(-1, -1)], 0.0, 0.0

    cfg = DispatchConfig(2, 2, 1, 1, 1.0, ((1, 1), (1, 1)), ((1.0, 1.0), (1.0, 1.0)))
    result = experiment.SmallOracle(cfg).evaluate("learned_both", StopPolicy())
    assert (
        result["exact_policy_cost"] == 30.0 and result["exact_selection_regret"] == 16.0
    )
    assert result["exact_allocation_regret"] == 0.0


def test_exact_matching_regret_detects_allocation_with_same_machine_set():
    class FastPolicy:
        def act(self, obs, *args, **kwargs):
            pairs = tuple(
                (m, m)
                for m in range(2)
                if obs["technicians"][m, 0] == 1 and obs["machines"][m, 3] == 0
            )
            return pairs, [*pairs, (-1, -1)], 0.0, 0.0

    # Over two always-hazardous intervals, both selected machines stay down
    # either way. Two one-tick services incur two PM starts per machine;
    # one two-tick service incurs one. Allocation alone explains the cost gap.
    cfg = DispatchConfig(2, 2, 2, 1, 1.0, ((1, 2), (2, 1)), ((1.0, 1.0), (1.0, 1.0)))
    result = experiment.SmallOracle(cfg).evaluate("learned_both", FastPolicy())
    assert result["exact_optimum"] == 26.0 and result["exact_policy_cost"] == 28.0
    assert (
        result["exact_selection_regret"] == 0.0
        and result["exact_allocation_regret"] == 2.0
    )


def synthetic_episodes(
    cfg, learned=90.0, priority=100.0, allocation=100.0, nominal=None
):
    result = []
    for arm, cost in zip(MODES, (learned, priority, allocation)):
        for seed in cfg["train_seeds"]:
            for family in experiment.envmod.FAMILIES:
                c = (
                    nominal
                    if nominal is not None
                    and arm == MODES[0]
                    and family in experiment.NOMINAL
                    else cost
                )
                result.extend(
                    dict(
                        algorithm=arm,
                        train_seed=seed,
                        family=family,
                        eval_seed=e,
                        objective=c,
                    )
                    for e in cfg["evaluation_seeds"]
                )
    return result


def test_confirmatory_contrasts_require_both_controls_and_nominal_guard():
    cfg = experiment.settings("full")
    _, summary = experiment.summarize(synthetic_episodes(cfg), cfg, True)
    assert summary["primary_passed"]
    assert summary["contrasts"]["fixed_priority"]["ci97_5_bonferroni"] == [-10.0, -10.0]
    _, summary = experiment.summarize(
        synthetic_episodes(cfg, allocation=85.0), cfg, True
    )
    assert not summary["primary_passed"]
    # Strong pressure gains cannot excuse nominal regression.
    _, summary = experiment.summarize(
        synthetic_episodes(cfg, learned=40.0, nominal=120.0), cfg, True
    )
    assert not summary["primary_passed"]
    _, summary = experiment.summarize(
        synthetic_episodes(experiment.settings("smoke")),
        experiment.settings("smoke"),
        False,
    )
    assert summary["primary_passed"] is None
    with pytest.raises(RuntimeError, match="incomplete"):
        experiment.summarize(synthetic_episodes(cfg)[:-1], cfg, True)


def test_complete_smoke_freezes_source_once_and_preserves_artifacts(
    tmp_path, monkeypatch
):
    # Reproduce the previous experiment's mid-run source-disappearance failure:
    # startup reads copies, then they vanish during training. Checkpoints still
    # refer to the frozen bundle, so a branch switch cannot break later models.
    cfg = experiment.settings("smoke")
    original_audit = experiment.seed_audit(cfg)
    captured_paths = []
    for module in (experiment.envmod, experiment.policymod, experiment):
        path = tmp_path / Path(module.__file__).name
        path.write_bytes(Path(module.__file__).read_bytes())
        captured_paths.append(path)
        monkeypatch.setattr(module, "__file__", str(path))
    monkeypatch.setattr(experiment, "seed_audit", lambda settings: original_audit)
    original_fit = experiment.fit

    def disappearing_source(*args, **kwargs):
        for path in captured_paths:
            path.unlink(missing_ok=True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(experiment, "fit", disappearing_source)
    output = tmp_path / "smoke"
    args = argparse.Namespace(profile="smoke", device="cpu", output_dir=str(output))
    assert experiment.run(args) == output
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["actual_models"] == 3 and manifest["actual_training_steps"] == 144
    assert (
        manifest["actual_optimizer_steps"] == 24
        and manifest["actual_training_episodes"] == 36
    )
    assert (
        manifest["actual_test_episodes"] == 63
        and manifest["actual_development_episodes"] == 3
    )
    assert manifest["actual_reference_episodes"] == 22 and all(
        manifest["audits"].values()
    )
    with (output / "decisions.csv").open() as f:
        assert sum(1 for _ in csv.DictReader(f)) == 243
    with (output / "machine_metrics.csv").open() as f:
        assert sum(1 for _ in csv.DictReader(f)) == 243
    for name, expected in manifest["source_hashes"].items():
        assert (
            hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest()
            == expected
        )
    for checkpoint in output.glob("*/train_seed_*/model.pt"):
        model, payload = experiment.load_checkpoint(checkpoint)
        assert payload["source_bundle_sha256"] == manifest["source_bundle_sha256"]
        assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(FileExistsError):
        experiment.run(args)


def test_failure_status_and_partial_outputs_are_preserved(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("intentional fit failure")

    monkeypatch.setattr(experiment, "fit", failure)
    args = argparse.Namespace(
        profile="smoke", device="cpu", output_dir=str(tmp_path / "failed")
    )
    with pytest.raises(RuntimeError, match="intentional"):
        experiment.run(args)
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "FAILED" and "intentional" in manifest["error"]
    assert (tmp_path / "failed/reference_episodes.csv").is_file()
    with pytest.raises(FileExistsError):
        experiment.run(args)
