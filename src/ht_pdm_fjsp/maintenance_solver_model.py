"""Frozen profile splits; physics and controls inherited unchanged from v1."""

from functools import lru_cache
import itertools
import random

from ht_pdm_fjsp.maintenance_heterogeneous_model import (
    CONDITIONS, REFERENCES, feasible, plan, reference, contention,
)
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig

PROFILE_SPLIT_SEED = 166000
PROFILES = tuple(itertools.product(itertools.product((2, 3), repeat=4), range(2), range(4)))


@lru_cache(maxsize=1)
def profile_partitions():
    ids = list(range(len(PROFILES)))
    random.Random(PROFILE_SPLIT_SEED).shuffle(ids)
    return {"train": tuple(ids[:80]), "development": tuple(ids[80:96]), "test": tuple(ids[96:])}


def profile_id(seed, split=None, profile="full"):
    if profile not in ("full", "smoke"):
        raise ValueError("unknown execution profile")
    if profile == "smoke":
        if split is None:
            split = {165020: "development", 165010: "test"}.get(seed, "train")
        # Engineering-only physical profiles all belong to the FULL TRAIN pool.
        return profile_partitions()["train"][("train", "development", "test").index(split)]
    if split is None:
        split = "development" if 161000 <= seed < 161016 else "test" if 163000 <= seed < 163032 else "train"
    pool = profile_partitions()[split]
    if split == "development":
        if not 161000 <= seed < 161016:
            raise ValueError("development profile seed outside frozen panel")
        return pool[seed - 161000]
    if split == "test":
        if not 163000 <= seed < 163032:
            raise ValueError("test profile seed outside frozen panel")
        return pool[seed - 163000]
    return pool[random.Random(seed).randrange(len(pool))]


def cohort_config(seed, condition, horizon, split=None, profile="full"):
    base, common, exception = PROFILES[profile_id(seed, split, profile)]
    regime, k = CONDITIONS[condition]
    favorite = [1 - common if m == exception else common for m in range(4)]
    excluded = [m for m in range(4) if m != exception][:2]
    duration, effect = [], []
    for m in range(4):
        if regime == "homogeneous":
            duration.append((base[m],) * k)
            effect.append((0.75,) * k)
        else:
            ds = [base[m] - 1 if j == favorite[m] else base[m] + 1 for j in range(k)]
            if regime == "skill_mask" and m in excluded:
                ds[1 - common] = 0
            duration.append(tuple(ds))
            effect.append(tuple(1.0 if j == favorite[m] else 0.5 for j in range(k)))
    c = WaitingConfig(4, k, horizon, 3, 0.6, tuple(duration), tuple(effect), waiting_price=12.0)
    c.validate()
    return c


def profile_audit(profile):
    pools = profile_partitions()
    if set().union(*map(set, pools.values())) != set(range(128)) or any(
        set(a) & set(b) for a, b in itertools.combinations(pools.values(), 2)
    ):
        raise RuntimeError("profile partitions overlap or incomplete")
    return dict(
        status="PASS", permutation_seed=PROFILE_SPLIT_SEED,
        labeled_specialized_profile_splits=pools,
        full_test_rollouts_opened=False,
        smoke_profile_ids=list(pools["train"][:3]) if profile == "smoke" else None,
        interpretation="Labeled specialized combinations disjoint. Homogeneous projections and permutation-equivalent physical profiles can overlap; no unseen equivalence-class claim.",
    )
