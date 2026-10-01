# Experiment 6 — RA-QMIX curriculum-replay confirmation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-10-01
- Verification Status: IMPLEMENTED — macOS CPU smoke verified
- Version Label: `ra_qmix_curriculum_confirmation_v1`

## 1. Research question and status of prior evidence

The completed anchor-preservation development experiment selected
`curriculum_replay`. On five paired training seeds it reduced mean stress-max
relative regret from 0.665 to 0.058, improved stress on five of five seeds,
kept mean nominal degradation at -1.66%, and met the per-seed nominal constraint
on four of five seeds. The KL-anchored arm was also feasible but had a worse
mean stress score, so it is not carried forward.

This experiment asks whether the already selected curriculum-replay recipe
repeats its joint nominal-preservation and stress-robustness result on ten fresh
paired training seeds. It is a confirmation, not another hyperparameter search.

The falsifiable primary hypothesis is that the locked candidate passes all four
confirmation conditions against a paired nominal-only control at equal total
environment-step compute. No intervention coefficient, schedule, checkpoint,
metric, seed, or gate may be changed after viewing the confirmation results.

## 2. Locked arms and training recipe

Both arms use Full RA-QMIX (`tqmix`) with the same architecture, optimizer,
50,000-transition replay capacity, exploration schedule, TD loss,
counterfactual-consistency loss (`lambda_cf=0.05`), paired training seeds, and
480,384 environment steps.

| Regime | Steps 0–240,192 | Steps 240,192–480,384 | Phase-2 replay |
|---|---|---|---|
| `nominal_100` | nominal only | nominal only | pooled nominal replay |
| `curriculum_replay` | nominal only | locked 87.5%-nominal mixture | exactly 112 nominal + 16 stress transitions per 128-sample batch after stress replay becomes available |

The candidate is byte-for-byte the selected development recipe: the phase-2
schedule uses randomized 864-step blocks with 63 nominal episodes and two
episodes from each stress cell. Across the complete run, 93.75% of candidate
training transitions are nominal. At the phase boundary, replay retains the
latest 43,750 nominal transitions and allocates 6,250 slots to stress. There is
no KL or other behavior anchor. Schedule randomization uses the same dedicated
RNG construction as the development experiment.

## 3. Compute, seed panels, and stopping

| Item | Full CPU confirmation | macOS CPU smoke |
|---|---:|---:|
| Training seeds | 84600–84609 | 84800 |
| Evaluation seeds | 84700–84749 | 84810–84812 |
| Regimes | 2 | 2 |
| Scenarios | 4 | 4 |
| Checkpoints | 240,192 / 360,288 / 480,384 | 1,728 / 2,592 / 3,456 |
| Training episodes | 792,300 | 570 |
| Evaluation episodes | 12,000 | 72 |
| Saved checkpoints | 60 | 6 |
| Sealed evaluation seeds | 201–300, closed | closed |

The confirmation panels are disjoint from all earlier RA-QMIX development and
smoke panels. Training uses a fixed environment-step budget and never stops for
performance. It stops only for an error, non-finite loss, missing checkpoint,
or audit violation. Every invocation must use a new UTC-timestamped artifact
directory; the runner rejects a non-empty output directory.

## 4. Primary endpoint and locked decision rule

The frozen reference objectives remain 7.600 (nominal), 102.496 (early
failure), 71.568 (slow service), and 150.250 (combined pressure). For each
training seed, stress score is the maximum relative regret across the three
stress scenarios. Candidate-control differences are paired by training seed and
use the same 50 evaluation seeds in every scenario.

At 480,384 steps, confirmation passes only if all four conditions hold:

1. mean nominal relative degradation is at most 5%;
2. nominal relative degradation is at most 10% on at least eight of ten seeds;
3. mean paired stress-score delta is below zero;
4. stress score is lower on at least eight of ten seeds.

The 80% per-seed thresholds preserve the development gate rather than
strengthening it after observing the five-seed result. Paired Student t 95%
intervals across the ten training seeds are reported as supportive uncertainty
descriptors and are not additional pass/fail conditions. The primary decision
does not depend on a null-hypothesis p-value.

Secondary endpoints are the two intermediate checkpoints, per-scenario
objective and cost components, failures, downtime, queue behavior, action
diversity, utilization, and the supportive paired intervals. They are
descriptive and cannot overturn the primary gate.

A pass licenses a separately preregistered evaluation on the sealed seed panel;
it does not open that panel automatically. A failure keeps the sealed panel
closed and does not authorize tuning on these confirmation seeds.

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
adaptation_diagnostics.csv
episodes.partial.csv
episodes.csv
coordination.csv
budget_summary.csv
regret_scores.csv
primary_results.csv
confirmation_result.json
summary.json
<regime>/train_seed_<seed>/checkpoints/step_<budget>/model.pt
```

Required audits cover exact counts and step boundaries, checkpoint reloads,
matched positive optimizer-update counts, finite losses, identical architecture
and base loss, exact equality of control and candidate model states at the
phase boundary, the locked candidate schedule and transition shares, exact
112/16 stratified replay, zero anchor loss, paired seeds, cost reconciliation,
resource semantics, disjoint seed panels, and the unopened sealed panel.

Smoke is an engineering gate only. It must produce 570 training episodes, 72
evaluation rows, six reloadable checkpoints, two updated trajectories, a
`COMPLETED` manifest, and all required audits true. Its one-seed primary gate is
disabled and its performance values are not scientific evidence.

## 6. Workflow handoff

Codex runs unit/integration tests and the timestamped macOS CPU smoke, commits
and pushes only after every engineering gate passes, then supplies one compound
Ubuntu CPU command and the canonical resumable `rsync` command. Codex does not
SSH to the lab, start the full run, or transfer the full artifacts.

## 7. Local verification record

On 2026-10-01, the final CLI smoke completed on macOS CPU at
`/private/tmp/ra_qmix_curriculum_confirmation_smoke.VaAFJ5/run_20261001T081621Z`.
It produced the locked 570 training episodes, 72 evaluation rows, six reloadable
checkpoints, two updated trajectories, and a `COMPLETED` manifest. Every
required audit passed, including exact phase-boundary model equality, the
locked phase schedule and nominal transition share, exact stratified replay,
zero anchor loss, matched optimizer updates, seed isolation, and the unopened
sealed panel. The full repository suite passed with 283 tests and three
pre-existing dependency/deprecation warnings.

Smoke performance is not confirmation evidence. Its one-seed decision gate is
disabled; only the full profile can test the locked primary hypothesis.
