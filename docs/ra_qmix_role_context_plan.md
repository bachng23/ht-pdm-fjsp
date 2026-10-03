# Shared-agent role × peer-information diagnostic — locked v1, 2026-10-03

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: targeted experiment design/run.
- Protocol: `ra_qmix_role_context_v1`; locked before new smoke/full outcomes.
- Base: `a541b2d`. Prior oracle results motivate this follow-up; prior gates remain FAIL.

## Questions, interventions and hypotheses
Can explicit identity break the shared-policy symmetry restriction, and does observing the peer improve coordination beyond identity alone, without replacing the monotonic mixer?

2×2 input ablation, four shared edge-agent + original technician-aware monotonic mixer arms:
- `monotonic_base`: local observation, zero identity block, zero peer block.
- `monotonic_role`: local observation + two-dimensional one-hot machine identity, zero peer block.
- `monotonic_context`: local observation + peer observation, zero identity block.
- `monotonic_role_context`: local observation + identity + peer observation.

Identity is the stable machine index in this finite benchmark, not a learned or transferable role. Peer context is the other machine's existing 15-feature observation, in self-relative order; there is no own-observation duplication, future state, oracle label, hidden simulator field or other machine's chosen action in model input. It requires an explicit telemetry/broadcast contract at execution. Raw environment/mixer observations and masks are unchanged. Each agent makes its own masked greedy decision; no central joint selection for factorial arms, resolver or post-processing. Joint-Q is a separate diagnostic control with central selection, not a decentralized baseline.

All four arms have identical tensors and parameter counts; input dimension is always 15+2+15=32, dropped blocks are zero-filled. Identical padded-input initialization controls architecture/counts; inactive input weights are dormant in ablated arms, so equal nominal counts do not imply equal usable information/function classes. Mixer architecture, state inputs and parameter initialization remain identical across factorial arms. CF=0 for all arms to isolate information interventions; this run does not establish behavior with CF or during online RL adaptation.

### Locked primary family (practical gates, not p-value significance claims)
At final 10000 updates, held-out oracle one-step regret, state means within cell, then equal weighting across shared-preference and pressure cells; training seed is independent inference unit.

H-role: role-only minus base mean delta ≤−0.10 cost units, ≥20% mean relative reduction, improvement in ≥8/10 paired seeds.

H-peer: role+peer minus role-only mean delta ≤−0.05 cost units, ≥15% mean relative reduction, improvement in ≥8/10 paired seeds.

Each contrast additionally requires non-increasing exact undiscounted deployed-policy cost averaged equally over the two diagnostic cells, and nominal mean deployed cost no more than 10% above its own contrast control. Report supportive paired Student-t CI95% on ten seed deltas; it is not an extra gate. No per-cell or CF superiority claims from multiple unadjusted tests. Report factorial interaction and peer-only effects as secondary.

Joint-Q decision control: **in each of all three cells**, mean held-out regret ≤0.10 and mean exact undiscounted policy gap ≤2.0 cost units. Normalized RMSE is descriptive here: the question is policy decision quality, distinct from prior complete-Q fit calibration. Do not relabel the old positive-control failure as a pass. Empirical attribution for either primary contrast requires its practical gate and this new control gate.

### Structural information-feasibility diagnostic
For all reachable states, group identical augmented local inputs; assert a group never has conflicting masks. One categorical action variable per input group defines an arbitrary shared deterministic memoryless lookup policy. Every state constrains its two group variables to the exact gamma=.95 oracle-optimal joint-action set (tie tolerance 1e-8). Solve the finite constraint system using CP-SAT, one worker, deterministic cell seed, ≤30 seconds/arm/cell. Duplicate-input columns must choose the same action; a state whose optimal tuples all violate that is a direct infeasibility certificate. Otherwise record solver SAT/INFEASIBLE/UNKNOWN; verify a SAT assignment against every state, never treat UNKNOWN as infeasible. MODEL_INVALID aborts. This diagnostic tests optimal decision feasibility, not full Q-table representation or whether the neural net can learn a feasible mapping. It is stronger than showing a conflict at a reachable state only, but all-states infeasibility alone still does not prove initial-state policy impossibility. No new lower-bound claim without a separate proof.

## Cells and data
Reuse the previous three finite cells intentionally: two machines/two technicians, horizon6/max_age5, FIFO, recovery-at-start, unchanged costs and failure law. Nominal age3/p=.45/service((2,3),(3,2)); shared-preference age3/p=.45/service((2,3),(2,3)); pressure age2/p=.65/service((3,4),(3,4)). Smoke horizon3/max_age4. These cells/oracle population have already been inspected: this is a targeted mechanism diagnostic, not a fresh-environment generalization claim.

Regenerate exact reachable-state oracle using the tested implementation from a541b2d: q95 labels for fitting; q1 values for undiscounted gaps; all joint observations, feasible masks and Bellman audits. Split observation-equivalent states as indivisible groups, 80/20 stratified by time, fresh fixed dataset seeds 96100/96101/96102. Normalize labels using training-only mean/std. Same label tables/splits across arms; whole action tables stay within each split. Uniform train state-action sampling, held-out regret averages states uniformly, not on-policy occupancy.

Full training seeds 96000–96009, rollout evaluation seeds 96200–96249. Smoke seed96400/evaluation96410–96412. Audit disjointness/freshness against frozen registry from 168 local manifests; sealed 201–300 stays closed. Dataset seeds select a new split of the same known population, not new independent environments.

## Models, budget and stopping
Shared edge agent hidden64; original queue mixer hidden64; all inputs padded to32. Joint-Q MLP sees only the concatenated existing observations, hidden width chosen by nearest parameter count before outcomes (within5%, smoke10%). Shared arms start from identical tensors; each cell/seed uses the same dedicated minibatch RNG stream across all five arms.

Supervised oracle normalized MSE, Adam lr3e-4, batch256, clipping10; 10000 updates/model. No exploration, TD bootstrap, target network, CF, curriculum or new simulator semantics. Smoke hidden16/mixer16, batch64, 160 updates/model. Log every100 updates (smoke20); save final checkpoint only. Fixed budget; abort exceptions/nonfinite/engineering audit failure; retain FAILED/partial files, no automatic retry, resume or overwrite. Full scope 3×5×10=150 fits, 1.5 million optimizer updates, 7500 rollout episodes. tqdm for oracle, fitting and multi-seed evaluation.

## Secondary/mechanism metrics and artifact contract
Per-cell training/held-out normalized RMSE and regret; exact expected discounted and undiscounted policy cost/gaps; rollouts and cost decomposition/failures/waiting; local-greedy versus exhaustive agreement (must match maxima for monotonic arms); same-input/same-action behavior; regret on states where identical original observations require unequal oracle actions; feasible-set solver status and verified witness policies. Store selected joint actions for each state to inspect assignment, rather than infer decisions from mean cost.

New timestamped directory only; runner rejects nonempty outputs. Parent manifest/config/seed audit/feature contract/summary; per-cell oracle state/Q tables and `policy_feasibility.json`; parameter counts; `training_progress.csv`, `state_metrics.csv`, `model_metrics.csv`, `paired_seed_metrics.csv`, `episodes.partial.csv`, `episodes.csv`, `coordination.csv`; 150 final checkpoint files full. Manifest records Git SHA/dirty/runtime/panels/expected and actual counts/audits/output paths, remains FAILED after an exception. Checkpoints include information flags, identity/peer ordering, normalization, config/settings and weights. Empty critical subsets use null metrics, not zero.

Smoke pass: all15 models/45 episodes complete, finite logs, checkpoint reload bitwise-identical predictions, zero invalid requests and cost errors, aligned counts/initialization/mixer/parameter capacities, verified feasible CSP assignments, no hidden input leakage, closed sealed panel, progress visible. Scientific thresholds are disabled in smoke; no tuning based on smoke performance. Tests cover input toggles/indexing, role-free symmetry, ability to break it with identity, unchanged monotonic mixer/state, exact feasibility/SAT/UNSAT/UNKNOWN handling, primary/control gates, save/load and failure/overwrite contracts.

## Scope and handoff
This tests fixed-shape identity and peer telemetry in oracle supervised fitting, not permutation-equivariant scale transfer or online adaptation. Fixed identity can exploit FIFO machine-index priority; keep that limitation explicit. Improvement does not prove a new architecture is necessary or that adaptation nominal regression is solved. Unknown feasibility outcomes or poor control performance mean inconclusive attribution. Preserve all existing source/results.

Workflow: separate codex branch → tests + real Mac CPU smoke → commit/push → exact user-run Ubuntu CPU compound command → user-run rsync -avhP --partial. Assistant does not SSH, launch full or transfer results. Both command handoffs are mandatory after successful push.
