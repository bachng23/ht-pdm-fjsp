# Experiment 5 — RA-QMIX nominal-policy anchor preservation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-09-30
- Verification Status: IMPLEMENTED — macOS CPU smoke verified
- Version Label: `ra_qmix_anchor_preservation_v1`

## 1. Research question and prior evidence

The nominal-anchor mixture calibration found a stable robustness–nominal
trade-off. At 480,384 environment steps, the 87.5%-nominal mixture reduced
paired stress-max regret on four of five training seeds but degraded nominal
performance by 22.1% on average and met the per-seed nominal constraint on only
one of five seeds. Stronger stress mixtures improved stress more while damaging
nominal performance further. No locked checkpoint met both sides of the gate.

This development experiment asks whether nominal pretraining followed by
stratified stress adaptation can retain robustness while an explicit behavior
anchor prevents the nominal policy from drifting.

The primary falsifiable hypothesis is that the curriculum plus behavior-KL
anchor meets all four locked feasibility conditions versus nominal-only
training at equal total environment-step compute. A curriculum without the
anchor is an ablation, and fixed 87.5%-nominal training is an active comparator.
The sealed panel remains closed.

## 2. Locked interventions

Every arm uses Full RA-QMIX (`tqmix`), identical architecture, optimizer,
replay capacity, exploration schedule, TD loss, counterfactual-consistency loss
with `lambda_cf=0.05`, paired seeds, and 480,384 total environment steps.

| Regime | Steps 0–240,192 | Steps 240,192–480,384 | Phase-2 replay | Behavior anchor |
|---|---|---|---|---|
| `nominal_100` | nominal only | nominal only | pooled | none |
| `fixed_87_5` | 87.5% nominal mixture | same mixture | pooled | none |
| `curriculum_replay` | nominal only | 87.5% nominal mixture | exactly 112 nominal + 16 stress transitions per 128-sample batch after stress replay becomes available | none |
| `curriculum_kl_anchor` | nominal only | same as curriculum replay | same stratification | KL to frozen phase-boundary nominal action distribution |

The 87.5% schedule uses a randomized 864-step block containing 63 nominal
episodes and two episodes from each of the three stress cells. Curriculum arms
therefore receive 93.75% nominal transitions over the complete run, but phase-2
replay is kept at 87.5% nominal rather than being dominated by the nominal
pretraining history. Replay retains the common total capacity of 50,000: the
phase-boundary buffer keeps the latest 43,750 nominal transitions and assigns
6,250 slots to stress transitions. A dedicated schedule RNG does not consume
the action or replay RNG.

At the phase boundary, `curriculum_kl_anchor` freezes a copy of the nominal
network. On nominal replay states, the auxiliary loss is

`10 * KL(softmax(Q_teacher) || softmax(Q_online))`

over valid local actions, with temperature 1.0. The coefficient 10 is locked
before this run; it makes a dimensionless behavior loss non-negligible relative
to the median phase-2 TD loss observed in the completed 87.5% calibration arm.
It is not tuned on this experiment. The teacher, anchor coefficient, replay
ratio, and phase boundary never change in response to performance.

## 3. Compute, seeds, and stopping

The checkpoints are 240,192, 360,288, and 480,384 steps. The first is the exact
curriculum phase boundary and the last is primary.

| Item | Full CPU development run | macOS CPU smoke |
|---|---:|---:|
| Training seeds | 84300–84304 | 84500 |
| Evaluation seeds | 84400–84449 | 84510–84512 |
| Regimes | 4 | 4 |
| Scenarios | 4 | 4 |
| Checkpoints | 3 | 3 |
| Step checkpoints | 240,192 / 360,288 / 480,384 | 1,728 / 2,592 / 3,456 |
| Training episodes | 783,960 | 1,128 |
| Evaluation episodes | 12,000 | 144 |
| Sealed seeds | 201–300, closed | closed |

The new panels are disjoint from all earlier RA-QMIX protocols. Training uses a
fixed step budget and never stops for performance. It stops only for an error,
non-finite loss, missing checkpoint, or audit violation. Every rerun uses a new
UTC-timestamped artifact directory.

## 4. Metrics and decision rule

The frozen scenario references remain 7.600 (nominal), 102.496 (early failure),
71.568 (slow service), and 150.250 (combined pressure). For each training seed,
stress score is maximum relative regret over the three stress scenarios.

At 480,384 steps, each curriculum candidate is feasible only if all conditions
hold against paired `nominal_100` results:

1. mean nominal relative degradation is at most 5%;
2. nominal degradation is at most 10% on at least four of five seeds;
3. mean paired stress-score delta is below zero;
4. stress score is lower on at least four of five seeds.

The primary hypothesis passes only if `curriculum_kl_anchor` is feasible. The
unanchored curriculum uses the same gate as an ablation. If at least one
curriculum is feasible, the development selection minimizes mean stress score,
then nominal degradation, then intervention complexity. `fixed_87_5` is not a
selection candidate because it already failed the previous calibration; it is
included to replicate that failure on paired fresh seeds.

Secondary endpoints include the paired anchor-minus-unanchored nominal and
stress effects, checkpoint trajectories, per-scenario objective, cost
components, failures, downtime, queues, action diversity, and utilization.
Anchor loss and phase-2 nominal/stress replay counts are engineering diagnostics.
Mechanism associations are not causal mediation evidence.

A feasible curriculum licenses a separately preregistered ten-seed confirmation
with new seeds. Failure does not license the sealed panel; the next intervention
would test observable risk context or a hard nominal constraint rather than
another fixed mixture.

## 5. Artifact and audit contract

```text
manifest.json
benchmark_config.json
resolved_config.json
regime_registry.csv
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

The runner rejects a non-empty output directory and displays `tqdm` during
training and multi-seed evaluation. Required audits cover exact counts and step
boundaries, schedule shares, checkpoint completeness, positive and matched
optimizer-update counts, finite TD/counterfactual/anchor losses, identical
architecture and base loss, identical curriculum states at the phase boundary,
locked stratified replay after stress data becomes available, paired seeds,
cost reconciliation, resource semantics, seed disjointness, and the unopened
sealed panel.

Smoke is an engineering gate only. It must produce 1,128 training episodes, 144
evaluation rows, twelve reloadable checkpoints, four updated trajectories, and
all required audits true. Codex runs tests and this smoke locally, commits and
pushes only verified code, then provides the exact Ubuntu CPU command and the
canonical resumable `rsync` command without accessing the lab.

## 6. Local verification record

On 2026-09-30, the final CLI smoke completed on macOS CPU at
`/private/tmp/ra_qmix_anchor_preservation_smoke.BE3pJY/run_20260930T074808Z`.
It produced the locked 1,128 training episodes, 144 evaluation rows, twelve
reloadable checkpoints, and a `COMPLETED` manifest. Every required audit passed,
including identical curriculum phase-boundary states and schedules, exact
transition shares, active KL only in the anchored arm, and exact stratified
phase-two replay. The full repository suite passed with 277 tests and one
pre-existing dependency warning.

Smoke performance is not scientific evidence. Its one-seed gate is disabled;
only the full profile may evaluate the locked decision rule.
