# PPO J1-O1 route-guard counterfactual plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-20
- Verification Status: UNVERIFIED
- Version Label: ppo_route_guard_v1

## Experiment overview

- **Objective**: test whether the weak PPO mode is caused by routing J1-O1 to
  M2 instead of waiting for its M1 route.
- **Hypothesis**: the route guard materially lowers failures and objective for
  training seeds 11000 and 13000 while leaving the other policies unchanged.
- **Type**: paired counterfactual simulation; no training and no model selection.

## Locked design

- Conditions: original deterministic PPO and the same PPO with one action-mask
  intervention.
- Intervention: suppress `J1-O1 -> M2` whenever it is feasible, forcing the
  policy to wait when M1 is busy. The environment's native validity mask and
  transition model remain unchanged.
- Models: all five final PPO training replicates.
- Counterfactual diagnostic seeds: 40200–40399, identical across both conditions.
- Reserved future test seeds: 50000–50099 remain unopened.
- Primary endpoint: guard-minus-original objective, averaged within each
  training seed before any across-seed summary.
- Secondary endpoints: failures, cost, makespan, tardiness, maintenance,
  intervention count and paired episode win rate.
- Stopping rule: exactly 200 paired episodes per model and condition, without
  retry, tuning, early stopping or checkpoint selection.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| `route_guard_manifest.json` | JSON | status `COMPLETED`; future test unopened |
| `route_guard_episodes.csv` | CSV | 5 × 2 × 200 complete episodes |
| `route_guard_decisions.csv` | CSV | every original and counterfactual decision |
| `route_guard_summary.json` | JSON | paired effects at training-seed level |

The hard route guard is a diagnostic intervention, not a proposed final policy.
If it confirms the failure mode, the subsequent algorithm should learn
structured operation-machine scores rather than hard-code this route.

## Ubuntu lab execution

The failure-sensitive commands run in a subshell so an error cannot close the
interactive terminal tab:

```bash
cd "$HOME/ht-pdm-fjsp"
RUN_ID="ppo_route_guard_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p artifacts/lab_logs
LOG="artifacts/lab_logs/$RUN_ID.log"

(
  set -euo pipefail
  git fetch origin
  git switch codex/ppo-route-guard 2>/dev/null || \
    git switch --track origin/codex/ppo-route-guard
  git pull --ff-only
  uv sync --frozen --extra dev

  test -f artifacts/ppo_replicates_20260920T110405Z/run_manifest.json
  test -f artifacts/ppo_replicates_20260920T110405Z/aggregate_summary.json

  uv run ht-pdm-fjsp-ppo-route-guard \
    --source-run artifacts/ppo_replicates_20260920T110405Z \
    --device cpu \
    --output-dir "artifacts/$RUN_ID"
) 2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
echo "EXIT_STATUS=$STATUS"
echo "LOG=$LOG"
```

Pull all results back to the Mac:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

## Interpretation boundary

The intervention was motivated by the previous diagnostic panel, so it is
tested on a fresh diagnostic panel. It still concerns one synthetic instance
and cannot establish cross-instance generalization.
