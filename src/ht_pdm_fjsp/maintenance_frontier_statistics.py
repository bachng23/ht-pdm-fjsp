"""Paired crossed bootstrap; all predeclared cells and planner budgets reported."""

import itertools
import numpy as np

from ht_pdm_fjsp.maintenance_solver_policy import CENTRAL

METRICS = (
    "objective",
    "base_cost",
    "waiting_cost",
    "waiting_violation",
    "terminal_pending",
    "overdue_waiting_ticks",
    "corrective_requests",
    "served_requests",
    "censored_requests",
    "served_waiting_cost",
    "censored_waiting_cost",
    "max_wait",
    "p95_wait",
    "failures",
    "unavailable_ticks",
    "inference_seconds",
)


def validate_grid(rows, cfg):
    expected = {
        (a, s, p, c, h, q)
        for a, seeds in [(CENTRAL, cfg["model_seeds"])]
        + [(a, [-1]) for a in cfg["controls"]]
        for s, p, c, h, q in itertools.product(
            seeds,
            cfg["test_cohorts"],
            cfg["conditions"],
            cfg["horizons"],
            cfg["test_seeds"],
        )
    }
    keys = [
        (
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
        raise ValueError("incomplete/duplicate evaluation grid")
    for r in rows:
        if not all(np.isfinite(float(r[m])) for m in METRICS) or r["objective"] <= 0:
            raise ValueError("nonfinite/nonpositive evaluation metrics")


def bootstrap_ratio(left, right, draws, seed):
    """left: training replica x profile x shock; right: profile x shock.

    Resample labeled profiles, shocks and training replicas independently, pairing
    environment weights across solvers. These are empirical percentile intervals,
    not certified finite-sample familywise coverage or new-workload inference.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 3 or right.shape != left.shape[1:] or np.any(right <= 0):
        raise ValueError("paired bootstrap dimensions/positive denominator")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite bootstrap values")
    rng = np.random.default_rng(seed)
    nr, npf, nq = left.shape
    result = []
    for start in range(0, draws, 256):
        b = min(256, draws - start)
        wr = rng.multinomial(nr, np.full(nr, 1 / nr), size=b) / nr
        wp = rng.multinomial(npf, np.full(npf, 1 / npf), size=b) / npf
        wq = rng.multinomial(nq, np.full(nq, 1 / nq), size=b) / nq
        profiles = np.einsum("br,rpq->bpq", wr, left, optimize=True)
        lm = np.einsum("bpq,bp,bq->b", profiles, wp, wq, optimize=True)
        rm = np.einsum("pq,bp,bq->b", right, wp, wq, optimize=True)
        result.append(lm / rm)
    return np.concatenate(result)


def classify(
    cost_ratio,
    upper,
    latency_ratio,
    common_latency_ratio,
    waiting_delta,
    pending_delta,
    cfg,
):
    guards = (
        waiting_delta <= cfg["waiting_margin"]
        and pending_delta <= cfg["pending_margin"]
    )
    ni = upper is not None and upper <= 1 + cfg["cost_margin"]
    speed = max(latency_ratio, common_latency_ratio) <= cfg["latency_ratio_threshold"]
    quality = upper is not None and upper <= 1 - cfg["cost_margin"]
    if cfg["profile"] == "smoke":
        status = "ENGINEERING_ONLY"
    elif guards and quality:
        status = "QUALITY_ADVANTAGE_SCREEN"
    elif guards and ni and speed:
        status = "LOWER_COMPUTE_TRADEOFF_SCREEN"
    else:
        status = "INCONCLUSIVE_OR_NO_SCREENED_ADVANTAGE"
    return dict(
        status=status,
        cost_noninferiority_screen=ni,
        quality_advantage_screen=quality,
        lower_compute_screen=speed,
        waiting_guard=waiting_delta <= cfg["waiting_margin"],
        pending_guard=pending_delta <= cfg["pending_margin"],
        cost_ratio=cost_ratio,
    )


def summarize(rows, cfg, common):
    validate_grid(rows, cfg)
    lookup = {
        (
            r["controller"],
            r["train_seed"],
            r["cohort"],
            r["condition"],
            r["horizon"],
            r["seed"],
        ): r
        for r in rows
    }
    aggregates, comparisons, paired = {}, {}, []
    cells = list(itertools.product(cfg["conditions"], cfg["horizons"]))
    family = len(cells) * len(cfg["controls"])
    confidence = 1 - 0.05 / family
    for cell, (cond, h) in enumerate(cells):

        def panel(a, s, metric):
            return np.array(
                [
                    [lookup[(a, s, p, cond, h, q)][metric] for q in cfg["test_seeds"]]
                    for p in cfg["test_cohorts"]
                ]
            )

        for a in [*cfg["controls"], CENTRAL]:
            seeds = cfg["model_seeds"] if a == CENTRAL else [-1]
            agg = {m: float(np.mean([panel(a, s, m) for s in seeds])) for m in METRICS}
            agg["mean_decision_seconds"] = agg["inference_seconds"] / h
            agg["common_state_mean_seconds"] = common[(a, cond, h)]["mean_seconds"]
            agg["cost_by_train_seed"] = {
                str(s): float(panel(a, s, "objective").mean()) for s in seeds
            }
            aggregates[f"{a}:{cond}_H{h}"] = agg
        central = aggregates[f"{CENTRAL}:{cond}_H{h}"]
        left = np.array([panel(CENTRAL, s, "objective") for s in cfg["model_seeds"]])
        for a in cfg["controls"]:
            other = aggregates[f"{a}:{cond}_H{h}"]
            right = panel(a, -1, "objective")
            ratio = central["objective"] / other["objective"]
            ci = None
            if cfg["profile"] == "full":
                # Reuse the exact same resampling stream across controls in each cell.
                bs = bootstrap_ratio(
                    left, right, cfg["bootstrap_draws"], cfg["bootstrap_seed"] + cell
                )
                ci = np.quantile(
                    bs, [(1 - confidence) / 2, (1 + confidence) / 2]
                ).tolist()
            latency = central["mean_decision_seconds"] / other["mean_decision_seconds"]
            shared = (
                central["common_state_mean_seconds"]
                / other["common_state_mean_seconds"]
            )
            wd = central["waiting_violation"] - other["waiting_violation"]
            pd = central["terminal_pending"] - other["terminal_pending"]
            key = f"{CENTRAL}_vs_{a}:{cond}_H{h}"
            comparisons[key] = dict(
                **classify(ratio, ci[1] if ci else None, latency, shared, wd, pd, cfg),
                cost_ratio_ci=ci,
                nominal_confidence=confidence,
                multiplicity_family=family,
                cost_margin=cfg["cost_margin"],
                closed_loop_latency_ratio=latency,
                common_state_latency_ratio=shared,
                waiting_violation_difference=wd,
                pending_difference=pd,
                sensitivity=[
                    dict(
                        cost_margin=m,
                        point_cost_acceptable=ratio <= 1 + m,
                        ci_cost_acceptable=ci is not None and ci[1] <= 1 + m,
                    )
                    for m in [0.0, 0.01, 0.02, 0.05]
                ],
            )
            for s, cost in central["cost_by_train_seed"].items():
                paired.append(
                    dict(
                        contrast=key,
                        train_seed=int(s),
                        central_cost=cost,
                        reference_cost=other["objective"],
                        difference=cost - other["objective"],
                    )
                )
    return paired, dict(
        aggregates=aggregates,
        comparisons=comparisons,
        inference="crossed empirical percentile bootstrap, Bonferroni nominal cost intervals; approximate with finite draws",
        scope="familiar labeled profiles and new shock draws; not workload or scale generalization",
        joint_screen_caveat="Latency and waiting/pending guards are descriptive; screens are not joint confirmatory tests or SLA certification.",
        replica_caveat="RL family mean describes ten trained policies, not an ensemble or a selected deployable checkpoint.",
    )
