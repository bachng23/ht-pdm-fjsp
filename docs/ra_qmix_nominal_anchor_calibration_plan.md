# Experiment 4 — RA-QMIX nominal-anchor mixture calibration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-09-29
- Verification Status: IMPLEMENTED — macOS CPU smoke verified
- Version Label: `ra_qmix_nominal_anchor_calibration_v1`

## 1. Research question

Equal-episode four-scenario training improved every stress scenario on all ten
training seeds but degraded in-distribution performance by 33.5%. Because the
three stress episodes have horizon 18 and the nominal episode has horizon 12,
that schedule allocated only 18.2% of replay transitions to the nominal cell.

This development experiment asks whether increasing the nominal transition
share can retain the stress gains while satisfying a locked nominal-performance
constraint. It calibrates a mixture; it is not a confirmation experiment and
does not open the sealed panel.

The falsifiable hypothesis is that at least one nominal-anchored mixture meets
both the nominal constraint and the directional stress-improvement rule versus
nominal-only training at equal environment-step compute.

## 2. Locked intervention

All arms use the unchanged Full RA-QMIX (`tqmix`) architecture, TD loss,
counterfactual-consistency loss with `lambda_cf=0.05`, optimizer, replay,
exploration schedule, network size, paired training seeds, and total environment
steps. Only the training-scenario schedule changes.

| Regime | Nominal transition share | Transition share per stress cell | Exact randomized block |
|---|---:|---:|---|
| `nominal_100` | 100% | 0% | nominal only |
| `nominal_87_5` | 87.5% | 4.17% | 63 nominal + 2 of each stress cell |
| `nominal_75` | 75% | 8.33% | 27 nominal + 2 of each stress cell |
| `nominal_50` | 50% | 16.67% | 9 nominal + 2 of each stress cell |

Every complete block is shuffled with a dedicated schedule RNG derived from the
training seed. It does not consume the action-exploration or replay-sampling RNG.
The failed 18.2%-nominal configuration is not repeated.

## 3. Compute, seeds, and stopping

Step checkpoints are 240,192, 360,288, and 480,384 environment steps. They are
all divisible by 864, the largest mixture block, and therefore end on complete
episodes and blocks in every arm. The primary checkpoint is 480,384 steps.

| Item | Full CPU calibration | macOS smoke |
|---|---|---|
| Training seeds | 84000–84004 | 84200 |
| Evaluation seeds | 84100–84149 | 84210–84212 |
| Regimes | 4 | 4 |
| Scenarios | 4 | 4 |
| Checkpoints | 3 | 3 |
| Training episodes | 742,260 | 801 |
| Evaluation episodes | 12,000 | 144 |
| Sealed seeds | 201–300, closed | closed |

A pre-implementation audit of pulled manifests and repository protocols found no
prior use of these panels. They are disjoint from all prior RA-QMIX training,
development, holdout and smoke panels.

Training stops at the fixed final step budget. It never stops for performance.
A run stops only for an execution error, non-finite loss, missing checkpoint or
audit failure. A rerun uses a new timestamped directory.

## 4. Metrics and calibration rule

Scenario references remain frozen at the development-sweep values:

| Scenario | Objective reference |
|---|---:|
| `in_distribution` | 7.600 |
| `early_failure` | 102.496 |
| `slow_service` | 71.568 |
| `combined_pressure` | 150.250 |

For each training seed, the stress score is maximum relative regret over only
`early_failure`, `slow_service`, and `combined_pressure`. Each candidate is
paired with `nominal_100` on the same training and evaluation seeds.

A candidate is feasible only if all four conditions hold at the primary
checkpoint:

1. mean paired nominal relative degradation is at most 5%;
2. nominal degradation is at most 10% on at least four of five seeds;
3. mean paired stress-score delta is below zero;
4. the candidate stress score is lower on at least four of five seeds.

Among feasible candidates, select lexicographically by lower mean stress score,
lower mean nominal degradation, then higher nominal transition share. If none is
feasible, no candidate is selected and the calibration fails. This descriptive
development rule is not a p-value.

Secondary endpoints are overall four-scenario max and mean regret, checkpoint
trajectory, per-scenario objective and cost per timestep, cost decomposition,
failure/downtime counts, queue metrics, action diversity, and utilization.
Mechanism endpoints are diagnostic associations rather than causal mediation.

A selected candidate licenses a separately preregistered confirmation against
`nominal_100` using ten completely new training seeds. It does not license use
of seeds 201–300. If no candidate is feasible, the next intervention must be a
curriculum, stratified replay, or explicitly constrained loss rather than a
confirmation of this sweep.

## 5. Artifact and audit contract

```text
manifest.json
benchmark_config.json
resolved_config.json
mixture_registry.csv
parameter_counts.csv
training_schedule.csv
training_progress.csv
training_episodes.csv
evaluation_progress.csv
episodes.partial.csv
episodes.csv
coordination.csv
budget_summary.csv
regret_scores.csv
primary_results.csv
selection_result.json
summary.json
<regime>/train_seed_<seed>/checkpoints/step_<budget>/model.pt
```

The runner rejects a non-empty output directory, displays `tqdm` for training
and evaluation, streams progress artifacts, and writes a `FAILED` manifest on
post-creation errors. Required audits cover exact counts, unique evaluation
keys, exact step boundaries, complete checkpoints, positive updates, finite
losses, exact transition shares, identical architecture/loss, paired seeds,
cost reconciliation, resource semantics, progress completeness, seed
disjointness, and the closed sealed panel.

Smoke passes with 801 training episodes, 144 evaluation rows, twelve
checkpoints, four updated trajectories, exact schedule shares, checkpoint
save/load, visible progress and every required audit true. Smoke performance is
engineering evidence only.

The full calibration is CPU-first on the Ubuntu lab. Codex implements, tests and
smokes locally, commits and pushes verified code, then provides one compound lab
command and the canonical resumable `rsync` command. Codex does not SSH to the
lab or launch the full run.

## 6. Local verification record

On 2026-09-29, the CLI smoke completed on macOS CPU at
`/private/tmp/ra_qmix_nominal_anchor_smoke.pYP3Xv/run_20260929T131603Z`.
It produced 801 training episodes, 144 evaluation rows, twelve reloadable
checkpoints, four updated training trajectories, and a `COMPLETED` manifest.
Every required audit passed. The complete repository suite passed with 270
tests; the three emitted warnings were pre-existing dependency warnings.

These smoke observations validate execution and artifact integrity only. The
calibration gate is deliberately disabled under the one-seed smoke profile, so
they are not scientific evidence for any mixture.
