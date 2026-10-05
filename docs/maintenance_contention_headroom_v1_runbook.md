# Contention/headroom v1: execution and interpretation

The frozen protocol is `maintenance_contention_headroom_v1_plan.md`. Run:

```bash
uv run --frozen --no-sync python -m ht_pdm_fjsp.maintenance_contention_headroom \
  --profile smoke --device cpu
```

For the lab, use the final handoff command pinned to the verified commit in a
fresh detached worktree and a new UTC output directory. The full profile needs
clean committed source. Do not reuse an output directory or resume: interruption
leaves partial journals and a FAILED manifest; start a fresh run instead.
The log is a sibling of the output directory so it does not defeat the empty
output guard. CPU is the supported execution device. There is no training.

Full workload: 2,400 development reference episodes, then 15,000 paired test
episodes, then 30 capped exact jobs. Each exact job attempts all 27 initial age
vectors. The full test set contains every controller, capacity, cohort, seed,
and both horizons; none is dropped based on performance. A reference is selected
per capacity on development priced cost before test or exact policy evaluation.
The continuation reference is the same at H12 and H24.

`controller_checkpoints/controllers.json` stores executable controller settings
and seed roles, replacing neural weights. `evaluation_configs.json` contains
all physical configurations. Each decision records the pre/post state, physical
action, costs, subset count, and planner candidates/scores/uncertainty. These
support action and scalar-physics replay. Planner noise is independent of actual
future hazards, shared across candidate root actions, and held fixed across
paired capacity/method/horizon comparisons. Planner MC standard errors describe
forecast noise; the confirmatory CI uses paired cohort-level observed costs.

Metrics use the simulator's interval timing. Utilization counts occupied workers
after new services start and before completions. Urgent excess counts currently
unserviced failed or imminent-risk machines beyond idle capacity. Choice
opportunity means more than one feasible subset (including STOP). Wait p95 is
the pooled machine-interval pending-wait state after that interval's waiting
increment; it is not the p95 over completed service jobs. Maximum wait and
violations include unfinished demand. Terminal pending excludes ongoing service;
terminal failed also includes failed machines still in corrective service.

`manifest.status=COMPLETED` means all prescribed jobs were attempted and all
episode panels completed. Check `exact_coverage` and `exact_capped_jobs` as well:
CAPPED jobs retain partial graphs and do not have a known optimum. Exact rows
contain the priced cost, independent relaxation, joint optimum, unavoidable
scarcity gap, and avoidable policy gap. Priority regret is split by episode half;
allocation regret is zero because worker profiles are identical. The graph's
Bellman residual is recomputed from its stored child values and transition law.

Only H12 planner-versus-selected-reference contrasts are confirmatory in full.
The three primary contrasts require >=5% reduction, >=8/10 cohort wins, paired
99% CI wholly below zero and <=10% base-cost degradation. H24 and exact panels
are diagnostic. Smoke statuses remain ENGINEERING_ONLY regardless of costs.
Even a positive planner gain does not establish RL learnability or MARL necessity.
A small N3 exact gap cannot establish an N4/H12 or H24 ceiling.

## Local engineering validation

- CLI smoke on macOS:
  `maintenance_contention_headroom_v1_smoke_cpu_20261005T183139Z`.
  Completed 12 development episodes, 90 test episodes, 3 exact jobs, no caps.
- Tests independently compare compact and vector transitions to WaitingEnv,
  compare existing reference actions, check worker-permutation symmetry and
  paired streams, replay planner forecasts, and verify exact values against
  scalar brute force. They also verify relaxation and regret decomposition,
  explicit cap behavior, FAILED artifacts, seed separation and overwrite refusal.
- Artifact replay covers all 450 smoke decision intervals: controller actions,
  scenario scores, scalar physical transitions, costs and file hashes agree.
- Additional engineering probe uses only smoke cohort137000: H24/32-scenario
  planner and N3/H6/K2 exact search complete. The latter visits34,739 states;
  recomputed Bellman residual1.42e-14. This probe is not a scientific test-panel
  result and is not used to choose the controller or alter the frozen budgets.
