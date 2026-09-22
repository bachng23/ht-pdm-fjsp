# MARL combined-policy confirmatory replication plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED
- Version Label: marl_combined_replication_v1

## Research question and locked hypotheses

The factorial follow-up found that independent actors with broadcast context had
the lowest validation objective, but the predeclared promotion gate failed
because mean failures increased by 0.014 per episode. With only five training
seeds, the 95% confidence interval for that failure delta was too wide to
distinguish a material regression from training-seed noise.

This fresh replication tests two locked claims:

1. **Objective superiority**: independent actors with broadcast context have a
   lower mean objective than independent actors without broadcast context.
2. **Failure non-inferiority**: the broadcast treatment increases failures by
   no more than 0.05 per episode. The margin means at most one additional
   failure per 20 episodes and is about 5.5% of the incumbent's observed
   0.906 failures per episode.

The treatment factor is broadcast context. Both conditions use independent
machine actors, the existing global critic, identical PPO settings, and equal
interaction budgets. Conclusions remain bounded to the current synthetic
benchmark instance.

## Metrics, seeds, and stopping rule

- **Primary endpoint**: paired per-training-seed validation objective delta,
  combined minus independent.
- **Safety endpoint**: paired per-training-seed failure delta, combined minus
  independent.
- **Secondary endpoints**: makespan, tardiness, preventive and corrective
  maintenance, total cost, conflict counts, and rejection rate.
- **Full training seeds**: 15000, 16000, ..., 24000; ten new independent
  training replications per condition.
- **Full validation seeds**: 55000--55199, paired across every trained model.
- **Smoke-only seeds**: training seeds 25000 and 26000; validation seeds
  55900--55904. Smoke results are not scientific evidence.
- **Reserved final test**: 50000--50099 remains unopened.
- **Training budget**: exactly 500,000 joint environment interactions per
  condition and training seed.
- **Stopping rule**: fixed budget only. No early stopping, adaptive retry,
  validation checkpoint selection, or test evaluation.
- **Replication unit**: independent training seed. Environment seeds are paired
  repeated measurements and are not treated as independent model replications.

## Locked confirmation gate

The combined policy is eligible for a separately authorized held-out test only
when all checks pass on the ten fresh training seeds:

1. the one-sided 95% upper confidence bound for objective delta is below zero;
2. at least 7/10 objective deltas are nonpositive;
3. the one-sided 95% upper confidence bound for failure delta is at most +0.05;
4. both coordination audits pass with zero invalid and duplicate executions;
5. exactly ten independent training seeds completed.

The confidence intervals use the Student-t distribution over training-seed
means. This is an engineering confirmation gate, not permission to open the
reserved final panel automatically.

## Artifact schema and integrity gates

Every invocation writes to a new timestamped directory containing:

- `marl_replication_manifest.json`
- `benchmark_config.json`
- `marl_replication_episodes.csv` and `.partial.csv`
- `marl_replication_coordination.csv` and `.partial.csv`
- `marl_replication_summary.json`
- for each condition/seed: `policy.pt`, recovery checkpoints,
  `training_progress.csv`, and `training_episodes.csv`
- the foreground terminal log produced by `tee`

Completion requires 2 conditions x 10 training seeds x 200 validation seeds =
4,000 unique full-profile rows, exact reward/objective identity, complete
500,000-step cells, passing coordination audits, and an unopened final panel.
Training and multi-seed evaluation expose `tqdm` progress.

## Local smoke command

```bash
RUN_ID="marl_combined_replication_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
uv run ht-pdm-fjsp-marl-replication \
  --profile smoke \
  --device cpu \
  --prior-factorial-run lab_results/marl_factorial_20260922T012036Z \
  --output-dir "artifacts/$RUN_ID"
```

The final handoff supplies one foreground Ubuntu command to fetch the committed
branch, verify the prior artifact, run the full profile with `tqdm`, and tee a
timestamped log. A separate Mac command retrieves the complete artifact tree
from `bachng@100.111.83.52` using `rsync -avhP --partial`.
