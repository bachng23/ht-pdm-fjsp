# Experiment 3 — RA-QMIX training-distribution alignment screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-09-28
- Verification Status: IMPLEMENTED / LOCAL CPU SMOKE VERIFIED — full screen pending
- Version Label: `ra_qmix_domain_randomization_v1`

## 1. Research question

The robust-checkpoint holdout found that Full RA-QMIX at 40k nominal-training
episodes reduced mean per-training-seed worst-scenario regret relative to the
50k endpoint, but uncertainty across five training seeds remained large. The
current learner trains only in the in-distribution environment even though its
robustness target spans four environments.

This screen asks whether exposing the unchanged Full RA-QMIX architecture and
loss to an equal mixture of the four evaluation environments improves robust
generalization relative to nominal-only training at equal environment-step
compute.

The experiment isolates training distribution. It does not tune the
counterfactual-loss weight, architecture, optimizer, reward, evaluation metric,
or checkpoint using the new evaluation panel.

## 2. Conditions and intervention

Both arms use algorithm ID `tqmix`, the existing edge utilities,
queue-conditioned mixer, counterfactual-consistency loss with `lambda_cf=0.05`,
optimizer settings, replay settings, exploration schedule, network size and
paired training seeds.

| Training regime | Episode environment |
|---|---|
| `nominal_only` | Always `in_distribution` |
| `uniform_four_scenario` | Exactly balanced randomized blocks containing one episode from each of `in_distribution`, `early_failure`, `slow_service`, and `combined_pressure` |

Scenario-block permutations use a dedicated RNG derived from the training seed;
they do not consume the action-exploration/replay RNG stream. Every four-episode
block is exactly balanced.

## 3. Compute budget and checkpoints

The nominal environment has horizon 12 while the other three cells have horizon
18. Equal episode counts would therefore give the treatment arm more
transitions and optimizer opportunities. Budgets are locked in environment
steps instead:

- diagnostic checkpoint 1: 240,240 steps;
- diagnostic checkpoint 2: 360,360 steps;
- primary checkpoint: 480,480 steps.

All budgets are divisible by 12 and by the 66 steps in one complete four-cell
training block. Checkpoints therefore occur on episode and balanced-block
boundaries in both arms. The primary budget corresponds to 40,040 nominal
episodes and differs by only 0.1% from the earlier 40k-episode checkpoint.

Training stops at 480,480 environment steps. It does not stop early for
performance. A run stops only for an execution error, non-finite loss, missing
checkpoint, or audit violation.

## 4. Seeds and scale

| Item | Full CPU screen | macOS smoke |
|---|---|---|
| Training seeds | 83800–83809 | 83710 |
| Evaluation seeds | 83900–83999 | 83720–83722 |
| Training regimes | 2 | 2 |
| Scenarios | 4 | 4 |
| Step checkpoints | 3 | 3 |
| Evaluation episodes | 24,000 | 72 |
| Sealed seeds | 201–300, closed | closed |

A manifest and protocol audit performed before implementation found no prior use
of the full or smoke panels. Training, evaluation, previous development/holdout,
smoke, and sealed panels are disjoint.

## 5. Frozen normalization and endpoints

Scenario references are frozen from the completed development checkpoint sweep
and are never recomputed from this experiment:

| Scenario | Reference objective |
|---|---:|
| `in_distribution` | 7.600 |
| `early_failure` | 102.496 |
| `slow_service` | 71.568 |
| `combined_pressure` | 150.250 |

For regime `r`, training seed `i`, and scenario `s`, define relative regret at
the primary checkpoint as:

`regret(r,i,s) = mean_objective(r,i,s) / frozen_reference(s) - 1`.

The primary score is the maximum regret over the four scenarios. The paired
primary estimand is:

`delta_R(i) = max_regret(uniform_four_scenario,i) - max_regret(nominal_only,i)`.

The directional hypothesis predicts a negative mean delta. The locked screening
gate requires all of:

1. mean primary delta below zero;
2. treatment better in at least 7 of 10 paired training seeds;
3. aggregate in-distribution objective degradation at the primary checkpoint no
   greater than 5% relative to nominal-only training.

This is a prespecified screening rule, not a p-value. Passing it licenses a new
10-seed confirmation protocol; it does not open the sealed panel or establish a
final claim.

Secondary endpoints are the checkpoint trajectory, mean regret, each scenario's
objective and cost per timestep, failure/downtime/maintenance/queue cost,
failure events, downtime steps, queue metrics, action diversity and technician
utilization. Mechanism endpoints are diagnostic associations, not causal
mediation evidence. Checkpoint comparisons and individual scenario contrasts
are secondary and require multiplicity-aware interpretation.

## 6. Artifact and audit contract

Each run uses a new timestamped directory and contains:

```text
manifest.json
benchmark_config.json
resolved_config.json
parameter_counts.csv
training_schedule.csv
training_progress.csv
training_episodes.csv
evaluation_progress.csv
episodes.partial.csv
episodes.csv
coordination.csv
budget_summary.csv
robust_scores.csv
primary_results.csv
summary.json
<training-regime>/train_seed_<seed>/checkpoints/step_<budget>/model.pt
```

The runner must reject a non-empty output directory; display `tqdm` progress for
training seeds, training episodes and evaluation episodes; stream training and
evaluation progress; and record a `FAILED` manifest on post-creation errors.

Required audits are exact evaluation counts and unique keys, exact step budgets,
three checkpoints per trajectory, positive optimizer updates, finite losses,
exact four-scenario balance, identical architecture/loss settings, cost
reconciliation, resource semantics, disjoint panels and a closed sealed panel.

Smoke passes with 72 evaluation rows, six checkpoints, two updated trajectories,
exact step/checkpoint boundaries, balanced treatment exposure, successful
checkpoint save/load, visible progress and every required audit true. Smoke
performance is engineering evidence only.

The full experiment is CPU-first on the Ubuntu lab. Codex implements, tests and
smokes locally, commits and pushes verified code, and then provides one compound
lab command plus the canonical resumable `rsync` command. Codex does not SSH to
the lab or launch the full run.

## 7. Local verification record

Verified on 2026-09-28 before the full evaluation panel was opened:

- repository suite: 265 tests passed;
- CLI CPU smoke: 57/57 training episodes, 72/72 evaluation rows, six/six
  checkpoints and two/two updated training trajectories;
- both regimes stopped at exactly 396 environment steps and the treatment saw
  six episodes from each of the four scenarios;
- checkpoint save/load, finite losses, unique keys, cost reconciliation,
  resource semantics, progress completeness, seed disjointness and the closed
  sealed panel all passed;
- training and multi-seed evaluation progress bars were visible;
- local smoke artifact:
  `/private/tmp/ra_qmix_domain_randomization_smoke.ip3JSL/run`.

The smoke primary comparison used one seed and a deliberately tiny budget. It
is not evidence for or against the screening hypothesis.
