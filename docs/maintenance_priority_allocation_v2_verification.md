## Material Passport

- Type: implementation and local macOS engineering verification.
- Protocol: `maintenance_priority_allocation_v2`; parent implementation `31b2b9e`.
- Status: VERIFIED engineering smoke; full scientific result unavailable.
- Branch: `codex/maintenance-priority-allocation-v2`.
- Locked amendment: [v2 plan](maintenance_priority_allocation_v2_plan.md).
- Workflow: [canonical experiment workflow](experiment_workflow.md).

## Changes and interpretation

Four deployment baselines are reported explicitly: learned_both, fixed_priority, fixed_allocation and risk_skill_rule. The rule already existed in v1 as a reference; v2 adds its family cost table, descriptive comparisons for every learned arm, physical traces and exact-headroom reporting. Only three arms train. Original two confirmatory comparisons and Bonferroni correction remain unchanged; the rule comparison is descriptive conditional on the common fixed evaluation panel.

fixed_priority now ranks all feasible machines, including nonurgent healthy machines. Only the top-ranked candidate and STOP remain available in the actor distribution. Both serve/STOP and technician allocation are learned and included in PPO sequence probability/entropy; rule identity/order is verified during replay. Fixed_allocation still learns machine/STOP selection with deterministic technician allocation. Parameter counts and budgets remain matched, but action-space cardinalities and available entropy differ by the intended ablation.

Exact N2 reports cost, optimum, selection/allocation regret, maximum possible relative reduction against each deployed policy, and whether an unrestricted optimum could achieve 5%. It also reports learned-both-vs-controls/rule exact differences. This diagnostic does not adjust thresholds or training and cannot establish headroom on larger cells. New v2 checkpoint version rejects v1 checkpoints; source, plan and registry are frozen at startup. Full runs reject dirty Git before generating evaluation configurations or creating outputs.

## Tests

Complete repository suite: **414 passed**, one existing SB3 rich/tqdm warning, 175.98 seconds. After formatting and removal of an unused test variable, the **44 directly relevant tests passed**, 4.73 seconds. Ruff 0.14.0 lint and format checks passed on all four edited Python files; git diff check passed.

Tests cover original environment boundaries plus learned fixed-priority STOP probability/entropy and nonzero policy gradients, selection of nonurgent candidates, rejection of wrong rule order, sampled versus replayed sequence probabilities, reference completeness/duplicate keys, conditional reference uncertainty, paired-seed CI arithmetic, zero-headroom known-optimum cases, full dirty-checkout refusal, v1 checkpoint refusal, save/load, artifact overwrite refusal, preserved failed runs and frozen source surviving source disappearance.

## Actual CLI smoke

Run: `maintenance_priority_allocation_v2_smoke_cpu_20261005T041112Z`.
Local artifacts: `/Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_priority_allocation_v2_smoke_cpu_20261005T041112Z`; adjacent `.log` preserves tqdm progress.
Audit script and machine-readable audit: `/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_priority_allocation_v2_preparation_20261005T041112Z`.

| Engineering artifact | Count |
|---|---:|
| Saved/reloaded models | 3 |
| Physical training steps / optimizer steps | 144 / 24 |
| Training episodes / rollout logs | 36 / 6 |
| Learned test / development episodes | 63 / 3 |
| Rule test / development episodes | 21 / 1 |
| Exact policy/headroom rows / exact comparisons | 4 / 3 |
| Learned decision traces / rule decision traces | 243 / 81 |
| Learned machine rows including development / rule test machine rows | 243 / 78 |
| Engineering gates passed | 26 / 26 |

Post-smoke audit replayed all **84 test episodes / 324 physical steps**, verified every recorded state, action and cost component, compared the deployed policy with every logged matching, independently checked pre-transition occupancy/start costs, family means, exact-gap/headroom arithmetic, all frozen source hashes and all expected CSV counts. Maximum numerical discrepancy: **0**. This is shared-simulator trace reproduction with independent arithmetic checks, not a second independent simulator.

All three greedy learned smoke policies chose STOP in the logged test trajectories; there were zero fixed-priority dispatches in those traces. Forced deterministic fixtures and sampled-sequence tests exercise nonurgent dispatch and rule ordering separately. Smoke contains only 48 training transitions per model; no learning-quality claim is made and no settings were changed in response to these outcomes. Scientific gate/CI fields are null. Sealed panels stayed closed.

Frozen source/plan/registry bundle SHA256: `68667c8171baaf76a23aafd188394619e54fa063d11b3f2b6bf38dc397651bcb`. The smoke manifest reports the clean parent commit plus dirty status because verification precedes the required commit; the bundle identifies tested code.

## Full handoff scope

CPU, 10 training seeds, 50 common held-out episode seeds per family and 20 development seeds. Thirty trained models, 3.6m physical training transitions, 300000 training episodes, 7500 rollout logs, 120000 optimizer steps, 10500 learned test episodes, 600 learned development episodes, 370 rule episodes and 31 exact/headroom rows. Fresh seed panels are documented in the v2 plan; registry covers 166 readable local manifests plus inherited history and reserved v1 panels, excluding unseen external runs.

Use a fresh detached lab worktree pinned to the pushed commit, with artifacts in the canonical `$HOME/ht-pdm-fjsp/artifacts/<timestamp>`. No branch switch or shared virtual environment with concurrent experiments. Keep the terminal open; no automatic resume/retry. User launches the full run and transfers artifacts using the two handoff commands. The experiment tests coordinated centralized PPO priority/allocation on the redesigned environment, not MARL necessity or the source of old QMIX failures.

