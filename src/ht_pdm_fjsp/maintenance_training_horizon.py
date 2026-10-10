"""Training horizon intervention at equal physical/PPO budget; covered mixture frozen."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import torch

from ht_pdm_fjsp import maintenance_training_coverage as shared
from ht_pdm_fjsp.maintenance_contention_headroom import file_hash
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, ENV_VERSION
from ht_pdm_fjsp.maintenance_solver_policy import (
    FEATURE_CONTRACT,
    CENTRAL as CENTRAL,
    LOCAL as LOCAL,
    MARL as MARL,
)
from ht_pdm_fjsp.maintenance_solver_model import cohort_config

VERSION = "maintenance_training_horizon_v1"
ROOT = shared.ROOT
ARMS = shared.ARMS
SCHEDULES = {reg: list(shared.SCHEDULES["covered"]) for reg in ("h12", "h24")}


def settings(profile):
    cfg = shared.settings(profile)
    full = profile == "full"
    cfg.update(
        protocol=VERSION,
        source_protocol=shared.VERSION,
        replica_progress_label="horizon learning replicas",
        test_progress_label="test horizon controllers",
        train_seeds=list(range(190000, 190010)) if full else [199000],
        development_seeds=list(range(194000, 194010)) if full else [199020],
        test_seeds=list(range(195000, 195020)) if full else [199040, 199041],
        horizons=[12, 24] if full else [4, 8],
        training_horizons={"h12": 12, "h24": 24} if full else {"h12": 4, "h24": 8},
        latency_horizons=[12, 24] if full else [4, 8],
        training_condition_mixtures={
            reg: {c: n / 5 for c, n in Counter(s).items()}
            for reg, s in SCHEDULES.items()
        },
        training_sequence_pairing="identical_shared_episode_seed_prefix; identical_full_sequence_within_regime",
        extra_snapshot_registries=[f"{shared.VERSION}_seed_registry.json"],
    )
    return cfg


def cell_settings(cfg, regime):
    return {
        **cfg,
        "training_regime": regime,
        "training_horizon": cfg["training_horizons"][regime],
        "paired_training_episodes": cfg["env_steps"]
        // max(cfg["training_horizons"].values()),
        "training_condition_schedule": SCHEDULES[regime],
    }


def counts(cfg):
    result = shared.counts(cfg)
    per_regime = len(ARMS) * len(cfg["train_seeds"])
    result["training_episodes"] = per_regime * sum(
        cfg["env_steps"] // h for h in cfg["training_horizons"].values()
    )
    result["latency_rows"] = (
        (result["models"] + len(cfg["controls"]))
        * len(cfg["conditions"])
        * sum(cfg["latency_horizons"])
        * 3
    )
    return result


def seed_audit(cfg):
    path = ROOT / f"configs/{VERSION}_seed_registry.json"
    registry = json.loads(path.read_text())
    parent = ROOT / "configs" / registry["historical_registry"]
    old = set(json.loads(parent.read_text())["declared_seed_values"])
    for lo, hi in registry["inherited_panels"]:
        old.update(range(lo, hi))
    fresh = [set(range(lo, hi)) for lo, hi in registry["new_panels"]]
    if any(a & b for a, b in itertools.combinations(fresh, 2)) or any(
        a & old for a in fresh
    ):
        raise ValueError("new horizon seed overlap")
    frozen = settings(cfg["profile"])
    for key in ("train_seeds", "development_seeds", "test_seeds"):
        if not cfg[key] or not set(cfg[key]) <= set(frozen[key]):
            raise ValueError("seed outside registered horizon panels")
    forbidden = (
        old
        | set().union(*fresh)
        | set(cfg["development_cohorts"] + cfg["test_cohorts"])
    )
    return dict(
        status="PASS",
        registry_sha256=file_hash(path),
        historical_registry_sha256=file_hash(parent),
        fresh_panels_disjoint=True,
        familiar_profile_labels_reused=True,
        full_shock_panel_opened=False,
    ), forbidden


def audit_sequence(sequences, record, cfg):
    seed, reg = record["train_seed"], record["training_regime"]
    n = cfg["env_steps"] // max(cfg["training_horizons"].values())
    if (
        record["shared_prefix_episodes"] != n
        or record["episodes"] != cfg["env_steps"] // cfg["training_horizons"][reg]
    ):
        raise RuntimeError("horizon episode/prefix budget mismatch")
    for key, digest in [
        ((reg, seed), record["training_seed_sequence_sha256"]),
        (("shared_prefix", seed), record["training_shared_prefix_sha256"]),
    ]:
        if key in sequences and sequences[key] != digest:
            raise RuntimeError("unmatched horizon training seed prefix/sequence")
        sequences[key] = digest


def load_checkpoint(path):
    p = torch.load(path, map_location="cpu", weights_only=True)
    if (
        p["protocol"] != VERSION
        or p["environment_version"] != ENV_VERSION
        or p["feature_contract"] != FEATURE_CONTRACT
        or p["algorithm"] not in ARMS
        or p["training_regime"] not in SCHEDULES
        or p["settings"]["protocol"] != VERSION
        or p["settings"]["training_condition_schedule"]
        != SCHEDULES[p["training_regime"]]
        or p["training_horizon"]
        != p["settings"]["training_horizons"][p["training_regime"]]
        or p["training_horizon"] != p["settings"]["training_horizon"]
    ):
        raise ValueError("horizon checkpoint contract mismatch")
    model = shared.base.model_for(p["settings"], p["train_seed"], p["algorithm"])
    model.load_state_dict(p["state_dict"])
    return model.eval(), p


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
            training_horizon=cfg["training_horizon"],
            rollout_update=update,
            physical_steps=steps,
            settings=cfg,
            state_dict=model.state_dict(),
        ),
        path,
    )
    restored, p = load_checkpoint(path)
    if any(
        not torch.equal(v, restored.state_dict()[k])
        for k, v in model.state_dict().items()
    ):
        raise RuntimeError("checkpoint reload tensor mismatch")
    for h in cfg["horizons"]:
        c = cohort_config(
            cfg["development_cohorts"][0], "skill_mask", h, profile=cfg["profile"]
        )
        env = WaitingEnv(c)
        env.reset(cfg["development_seeds"][0])
        for t in range(h):
            chosen = model.select(c, env.state, h - t)
            if chosen != restored.select(c, env.state, h - t):
                raise RuntimeError("checkpoint action/probability/value mismatch")
            env.step(chosen[0])
    return p


def summarize(rows, cfg):
    """Reuse independently checked paired estimator, explicitly relabel regimes.

    No global mutation. Only columns used by the estimator are copied; snapshots
    and raw output preserve true h12/h24 identities and actual training horizons.
    """
    names = {"h12": "legacy", "h24": "covered", "reference": "reference"}
    columns = (
        "controller",
        "train_seed",
        "cohort",
        "condition",
        "horizon",
        "seed",
        *shared.METRICS,
    )
    for row in rows:
        if row["training_regime"] not in names:
            raise RuntimeError("unknown training horizon regime")
        if row.get("training_horizon") != cfg["training_horizons"].get(
            row["training_regime"], -1
        ):
            raise RuntimeError("episode training horizon identity mismatch")
    translated = [
        {**{k: r[k] for k in columns}, "training_regime": names[r["training_regime"]]}
        for r in rows
    ]
    paired, result = shared.summarize(translated, cfg)
    result["aggregate"] = {
        k.replace("legacy:", "h12:", 1).replace("covered:", "h24:", 1): v
        for k, v in result["aggregate"].items()
    }

    def avg(reg, a, c, h, metric):
        return result["aggregate"][f"{reg}:{a}:{c}_H{h}"][metric]

    rename = {
        "legacy_cost": "short_training_cost",
        "covered_cost": "long_training_cost",
        "legacy_gap_to_central": "short_training_gap_to_central",
        "covered_gap_to_central": "long_training_gap_to_central",
        "central_legacy_cost": "central_short_training_cost",
        "central_covered_cost": "central_long_training_cost",
    }
    contrasts = {}
    for key, v in result["contrasts"].items():
        label, a, _ = key.split(":")
        if v["gates"]:
            h = cfg["horizons"][-1]
            v["gates"].update(
                homogeneous_cost_preserved=avg("h24", a, "homogeneous", h, "objective")
                <= 1.05 * avg("h12", a, "homogeneous", h, "objective"),
                homogeneous_waiting_guard=avg(
                    "h24", a, "homogeneous", h, "waiting_violation"
                )
                <= avg("h12", a, "homogeneous", h, "waiting_violation") + 0.02,
                homogeneous_pending_guard=avg(
                    "h24", a, "homogeneous", h, "terminal_pending"
                )
                <= avg("h12", a, "homogeneous", h, "terminal_pending") + 0.05,
            )
        if (
            v["primary"]
            and cfg["profile"] == "full"
            and v["status"] != "NOT_APPLICABLE"
        ):
            v["status"] = "PASS" if all(v["gates"].values()) else "FAIL"
        newkey = key.replace("coverage:", "horizon:", 1)
        contrasts[newkey] = {rename.get(k, k): x for k, x in v.items()}
    result["contrasts"] = contrasts
    result["interpretation"] = (
        "H24 versus H12 training, equal physical/optimizer budgets, fixed covered mixture; includes reset/return/state-distribution changes. Homogeneous guards do not certify absolute SLA."
    )
    paired = [
        {
            rename.get(k, k): (
                v.replace("coverage:", "horizon:", 1) if k == "contrast" else v
            )
            for k, v in r.items()
        }
        for r in paired
    ]
    return paired, result


def run(output, profile):
    shared.run(output, profile, protocol=sys.modules[__name__])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    p.add_argument("--device", choices=("cpu",), default="cpu")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    output = (
        args.output_dir
        or ROOT
        / "artifacts"
        / f"{VERSION}_{args.profile}_cpu_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    run(output.resolve(), args.profile)


if __name__ == "__main__":
    main()
