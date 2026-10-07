"""Paired, preregistered contrasts; independent unit is never the episode row."""

import itertools
import numpy as np

from ht_pdm_fjsp.maintenance_solver_model import CONDITIONS
from ht_pdm_fjsp.maintenance_solver_policy import ARMS, CENTRAL, FIXED, MARL, LOCAL, INDEPENDENT
from ht_pdm_fjsp.maintenance_solver_diagnostics import interval


def mean(rows, metric):
    if not rows:
        raise ValueError("empty panel")
    return float(np.mean([r[metric] for r in rows]))


def summarize(rows, selected_refs, cfg):
    expected = {
        (name, seed, cs, cond, h, shock)
        for name, seeds in [(a, cfg["train_seeds"]) for a in ARMS] + [(a, [-1]) for a in cfg["controls"]]
        for seed, cs, cond, h, shock in itertools.product(seeds, cfg["test_cohorts"], CONDITIONS, cfg["horizons"], cfg["test_seeds"])
    }
    keys = [(r["controller"], r["train_seed"], r["cohort"], r["condition"], r["horizon"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("incomplete/duplicated evaluation grid")
    groups = {}
    for r in rows:
        groups.setdefault((r["controller"], r["condition"], r["horizon"]), []).append(r)
    paired, cohorts, contrasts = [], [], {}
    H = cfg["horizons"][0]
    for cond, h in itertools.product(CONDITIONS, cfg["horizons"]):
        ref = selected_refs[cond]
        comparisons = [
            ("coordination", "joint_rollout", "adaptive_isolated_matching", .03, .02),
            ("learning", FIXED, ref, .05, .03),
            ("cooperative", LOCAL, INDEPENDENT, .02, None),
            ("decomposition", MARL, CENTRAL, .02, None),
            ("assignment", CENTRAL, FIXED, None, None),
            ("information", MARL, LOCAL, None, None),
            ("central_vs_planner", CENTRAL, "joint_rollout", None, None),
            ("fixed_vs_planner", FIXED, "joint_rollout", None, None),
            *(("economic", a, ref, None, None) for a in ARMS if a != FIXED),
        ]
        for label, left, right, threshold, base_threshold in comparisons:
            lp, rp = groups[(left, cond, h)], groups[(right, cond, h)]
            unit = "cohort" if label == "coordination" else "train_seed"
            ids = cfg["test_cohorts"] if unit == "cohort" else cfg["train_seeds"]
            deltas = []
            key = f"{label}:{left}_vs_{right}:{cond}_H{h}"
            for ident in ids:
                l = [r for r in lp if r[unit] == ident]
                r = [r for r in rp if r[unit] == ident] if right in ARMS or unit == "cohort" else rp
                lc, rc = mean(l, "objective"), mean(r, "objective")
                d = lc - rc
                deltas.append(d)
                paired.append(dict(contrast=key, unit=unit, replicate=ident, left_cost=lc, right_cost=rc, difference=d))
            for cs in cfg["test_cohorts"]:
                l = [r for r in lp if r["cohort"] == cs]
                r = [r for r in rp if r["cohort"] == cs]
                cohorts.append(dict(contrast=key, cohort=cs, left_cost=mean(l, "objective"), right_cost=mean(r, "objective"), difference=mean(l, "objective")-mean(r, "objective")))
            confirmatory = threshold is not None and cond == "specialized" and h == H
            confidence = .9875 if confirmatory else .95
            ci = interval(deltas, confidence)
            lc, rc = mean(lp, "objective"), mean(rp, "objective")
            reduction = 1 - lc / rc
            base = 1 - mean(lp, "base_cost") / mean(rp, "base_cost")
            nominal = mean(groups[(left, "nominal", H)], "base_cost") / mean(groups[(selected_refs["nominal"], "nominal", H)], "base_cost")
            violation = mean(lp, "waiting_violation")
            reference_wait = mean(groups[(ref, cond, h)], "waiting_violation")
            wins = sum(x < -1e-9 for x in deltas)
            gates = dict(
                priced_reduction=reduction >= threshold if threshold is not None else None,
                wins_at_least_80pct=wins >= .8 * len(ids),
                ci_upper_below_zero=ci is not None and ci[1] < 0,
                nominal_base_ratio_110pct=nominal <= 1.1,
                waiting_violation_guard=violation <= reference_wait + .02,
            )
            if base_threshold is not None:
                gates["base_reduction"] = base >= base_threshold
            status = "ENGINEERING_ONLY" if cfg["profile"] == "smoke" else ("PASS" if all(gates.values()) else "FAIL") if confirmatory else "EXPLORATORY"
            contrasts[key] = dict(
                left=left, right=right, condition=cond, horizon=h,
                left_cost=lc, right_cost=rc, paired_difference=float(np.mean(deltas)),
                priced_reduction=reduction, base_reduction=base,
                unit=unit, replicates=len(ids), wins=wins, confidence_level=confidence, ci=ci,
                uncertainty_scope="profile variation conditional on shared shock panel" if unit == "cohort" else "training-replica variation conditional on fixed labeled-profile/shock panel",
                confirmatory=confirmatory, nominal_base_ratio=nominal,
                left_waiting_violation=violation, reference_waiting_violation=reference_wait,
                gates=gates, status=status,
            )
    # Both planner root sets may use different labels but interchangeable workers
    # must give the same physical costs with identical forecasts and continuation.
    for cond in ("homogeneous", "nominal"):
        for h in cfg["horizons"]:
            a = {(r["cohort"], r["seed"]):r["objective"] for r in groups[("joint_rollout",cond,h)]}
            b = {(r["cohort"], r["seed"]):r["objective"] for r in groups[("fixed_allocation_rollout",cond,h)]}
            if any(abs(v-b[k]) > 1e-9 for k,v in a.items()):
                raise RuntimeError("homogeneous planner allocation invariance")
    return paired, cohorts, contrasts


def mechanisms(rows, selected_refs, cfg):
    metrics = (
        "objective", "base_cost", "maintenance_cost", "failure_cost", "unavailability_cost",
        "waiting_cost", "failures", "waiting_violation", "max_wait", "p95_wait",
        "terminal_pending", "utilization", "mean_preferred_worker_excess", "inference_seconds",
        "decision_latency_p50", "decision_latency_p95", "communication_bytes", "communication_rounds",
        "corrective_requests", "served_requests", "censored_requests", "request_overdue_observed",
    )
    groups = {}
    for r in rows:
        groups.setdefault((r["condition"],r["horizon"],r["controller"]),[]).append(r)
    return dict(
        aggregate={f"{c}_H{h}:{a}":{k:mean(p,k) for k in metrics} for (c,h,a),p in groups.items()},
        interpretation="Online latency percentiles are per episode then averaged, mixed-cache. See common-state warm-cache latency CSV for pooled per-decision quantiles. Censored waits are lower bounds, not completed waits. Rollout is achievable cost, never an optimum.",
    )
