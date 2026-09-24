# Parallel-Machine Maintenance Calibration Diagnostic Plan

## Scope

- Branch: `codex/parallel-maintenance-calibration-diagnostic`
- Type: simulator calibration and mechanism diagnostic, not an RL algorithm
  comparison.
- Parent screen: `parallel_maintenance_baseline_scan_full_cpu_20260924T044417Z`.

The parent screen found that QMIX was stable but improved the best heuristic by
only 1.41%, almost never selected preventive maintenance, and encountered fewer
than one technician-conflict step per episode. This diagnostic asks whether a
nearby, predeclared simulator regime makes preventive maintenance valuable and
technician contention observable before any new MARL method is developed.

## Hypotheses

- **H1 (PM value):** at least one factorial cell lets coordinated preventive
  FIBT reduce paired mean objective by at least 5% relative to reactive-only
  FIBT.
- **H2 (failure mechanism):** in every promoted cell, preventive FIBT reduces
  mean failures by at least 20% relative to reactive-only FIBT.
- **H3 (coordination pressure):** in a promoted cell, independent preventive
  proposals produce technician conflicts in 2%--20% of decision steps, while
  coordinated preventive FIBT executes with zero conflicts and zero resource
  violations.

If no cell passes all three gates, the next step is model redesign rather than
constraint-aware QMIX. Passing identifies a candidate development configuration
only; it does not establish algorithm superiority.

## Locked factorial

Protocol amendment before the full run: the first engineering smoke showed that
tripling absence produced only about 0.2% conflict steps. A local preflight found
that a lower, still predeclared PM trigger creates natural simultaneous demand
without imposing implausible absence. The full grid therefore replaces the
absence factor with PM risk threshold. No full-development or sealed-test result
was inspected before this amendment.

The full diagnostic crosses:

- failure penalty: `{18, 36, 54}`;
- PM duration factor relative to CM: `{0.35, 0.50, 0.65}`;
- PM one-step failure-risk threshold: `{0.005, 0.020}`.

Technician absence remains at the base values (`T1=0.04`, `T2=0.05`). All other
Weibull, skill, duration-variance, learning, success, restoration, reward,
resolver, and horizon settings remain fixed. The grid has 18 cells.

The final smoke profile uses only the current cell `(18, 0.65, 0.020)` and one
stronger PM cell `(36, 0.35, 0.005)` as an engineering check.

## Fixed policies

1. `reactive_fibt`: repairs failed machines only and assigns the fastest
   available technician.
2. `preventive_fibt`: adds PM when one-step Weibull failure probability reaches
   the fixed threshold `0.02`.
3. `preventive_balanced`: uses the same maintenance trigger but prioritizes the
   least-used available technician.
4. `independent_preventive`: every eligible machine independently requests its
   fastest currently available technician; the environment resolver handles
   collisions.

The comparison between reactive and preventive FIBT isolates maintenance timing
while retaining the same priority and technician rule. Common keyed shocks and
paired evaluation seeds are used within every cell.

## Metrics

Primary endpoint:

- paired objective delta, `preventive_fibt - reactive_fibt`, by evaluation seed.

Secondary and mechanism endpoints:

- production, downtime, failures, PM, CM, waiting, rework, and early-PM cost;
- proposal conflicts and conflict-step incidence;
- eligible maintenance-demand steps, excess-demand steps, mean excess demand,
  and same-best-technician collision opportunity;
- technician utilization and workload imbalance;
- reward/objective identity and all assignment audits.

## Seeds and stopping rule

- Smoke evaluation seeds: `63490:63493` (three seeds).
- Full development seeds: `63400:63450` (50 seeds).
- Sealed confirmation seeds remain `63200:63300` and must not be evaluated.
- Every policy runs one complete 168-step episode for every selected cell and
  seed. There is no adaptive stopping, result-dependent grid expansion, or
  test-set evaluation.

## Promotion rule

A cell is promoted only if all conditions hold:

1. paired mean objective improvement from preventive FIBT is at least 5%;
2. mean failure reduction is at least 20%;
3. independent conflict-step incidence is between 2% and 20%;
4. coordinated preventive FIBT has zero conflicts;
5. all feasibility, reward, completion, and sealed-panel audits pass.

If multiple cells qualify, choose the cell with the smallest normalized change
from the current configuration `(18, 0.65, 0.020)`, then the larger objective
improvement, then lexicographic cell id. This selection is a development
calibration decision, not a test result.

## Artifact schema

Each invocation refuses a non-empty output directory and writes:

```text
artifacts/<timestamped-run-id>/
├── calibration_manifest.json
├── resolved_base_config.json
├── cells.json
├── episodes.partial.csv
├── episodes.csv
├── decisions.csv
└── summary.json
```

Training is not part of this diagnostic. Multi-cell and multi-seed evaluation
must display `tqdm` progress. The manifest records the commit, exact cell grid,
seeds, versions, status, output inventory, and audit gate.

## Interpretation boundary

The factorial changes costs, PM service time, and availability simultaneously
across predeclared cells. It identifies useful calibration regions; it does not
estimate a real factory's parameters or prove that conflict causes a policy's
objective. Any promoted cell must undergo a separate RL baseline replication
before an algorithm-specific improvement is justified.
