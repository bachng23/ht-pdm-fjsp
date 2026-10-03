# RA-QMIX oracle representation diagnostic — locked v1, 2026-10-03

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent; mode: experiment design.
- Protocol: `ra_qmix_oracle_representation_v1`; locked before smoke outcomes.
- Source base: CF adaptation commit `9abc1f6`. This is a supervised diagnostic, not a new RL leaderboard.

## Question and hypotheses
Can the existing monotonic technician-aware factorization represent action interactions needed by this simulator, and does a more flexible mixer reduce its policy regret on identical oracle labels?

H-structure: a reachable fixed state has a strict reversal of a machine's two-action oracle Q ranking as the other machine's action changes. Such a witness certifies that no monotonic mixer of scalar local utilities can exactly represent the complete Q table at that state, regardless of capacity or optimizer. It does NOT establish inability to represent the optimal joint action (IGM) or explain the previous 3-machine adaptation result.

H-fit, primary empirical contrast: signed mixer minus monotonic-no-CF held-out oracle one-step decision regret, averaged equally across three cells and paired by initialization/training seed, at the fixed final update. Support requires mean reduction ≥0.05 cost units, reduction ≥20% of monotonic mean regret, and improvement in ≥8/10 seeds. CI95% across ten training seeds is supportive, not an additional gate. A positive result still includes changes to centralized action selection; report all-action selection and local-greedy selection for every factorized model to expose this difference.

Joint-Q is an approximate positive control, not an oracle and not an execution-information-matched MARL baseline. Control succeeds if its mean held-out normalized RMSE ≤0.10 and decision regret ≤0.10 cost units. If it fails, empirical fit attribution is inconclusive. Structural certificates remain independently interpretable. No superiority claims from secondary comparisons or checkpoint selection.

## Cells and exact oracle
All cells: 2 machines, 2 technicians, horizon 6, max_age 5, FIFO, recovery-at-service-start, unchanged failure RNG law and costs from PassiveConfig. Nominal: failure_age 3, p=.45, service ((2,3),(3,2)); shared preference: same age/p, service ((2,3),(2,3)); pressure: failure_age 2, p=.65, service ((3,4),(3,4)). Smoke uses horizon 3/max_age 4, same other settings.

Enumerate every reachable state and every masked valid joint action from the initial state. Enumerate independent Bernoulli failure events with exact probabilities; DP Q labels use gamma=.95, matching previous learners, and reward = negative cost. Also compute gamma=1 oracle values for undiscounted cost comparisons. Canonicalize expired busy_until to current time, retaining every future-relevant component. Assert probability mass=1, zero invalid requests, finite values, Bellman residual and cost reconciliation. Assert identical joint observations imply identical feasible actions/Q tables; abort on observation aliasing rather than compare a privileged-state oracle unfairly.

State population is uniform over reachable decision states, not on-policy occupancy. Split states (all their joint actions together) 80% training/20% held-out with fixed dataset seeds 95100/95101/95102, stratified by time; ensure at least one training state each time. Normalize targets by training-only mean and standard deviation. Independent local observations are the existing normalized observations. Central models see only concatenation of the same local observations; no extra hidden simulator fields. Record raw complete tables, state split and structural witnesses.

## Arms and budgets
1. `monotonic_no_cf`: existing `tqmix_no_cf` agent + queue mixer; deployed local masked argmax.
2. `monotonic_cf`: existing `tqmix`, lambda_cf=.05; secondary regularizer control.
3. `signed_no_cf`: identical local agent, hypernetworks/parameter count and initialization; remove absolute-value constraints on mixer weights. Central masked exhaustive joint selection; report local greedy separately.
4. `joint_q`: MLP on concatenated existing observations, one output per joint action; choose two-layer hidden width closest to factorized total parameter count without looking at outcomes. Require capacity deviation ≤5% full, ≤10% smoke.

Full hidden64/mixer64, 10000 Adam updates/model, lr3e-4, batch256, gradient clipping10. Smoke hidden16/mixer16, 160 updates/model, batch64. Shared seeded minibatch index stream across arms within a cell/seed; factorized arms share initial parameter tensors. Uniform sampling of training state-action pairs; no online exploration, replay, TD bootstrap or target network. Supervised fitting minimizes normalized MSE on oracle labels; CF secondary adds consistency loss with lambda=.05; both raw TD MSE and raw consistency loss are divided by the same training-target variance for optimization, retaining their relative weight. Log raw and normalized losses. Keep fixed budgets, log every 100 updates (smoke20), save only final checkpoint; abort errors/nonfinite/audit fail, no automatic retry/resume.

Train seeds 95000–95009; rollout evaluation 95200–95249. Smoke train 95400, evaluation 95410–95412. Dataset seeds also fixed for smoke (same numbers, different profile graph; smoke is engineering evidence only). Freshness checked against frozen registry of 167 local manifests; abort on overlaps. Sealed 201–300 remains closed. Models are fit separately per cell: this design tests representation and fitting, not nominal/stress multi-task adaptation.

## Metrics and artifacts
Primary regret at state s: max_a Q*(s,a) − Q*(s,argmax model Q), averaging held-out states equally. For monotonic arms primary execution uses local greedy; additionally report exhaustive selection regret and agreement. Report in-training-state metrics alongside held-out to distinguish interpolation/generalization problems. Secondary: normalized RMSE, rank reversals/count/margins, exact expected deployed-policy discounted/undiscounted costs from the initial state, each oracle gap, 50 paired seeded simulator rollouts per model/cell, failures/maintenance/waiting/downtime decomposition and invalid requests. Report uncertainty across training seeds, not episodes or states treated as independent training replications.

New timestamped output directory only; reject nonempty output. `manifest.json` records protocol/git/runtime/status/panels/counts/audits, `resolved_config.json`, `seed_audit.json`, per-cell `oracle_states.json`, `oracle_q.csv`, `structural_witnesses.json`, `parameter_counts.csv`, `training_progress.csv`, `state_metrics.csv`, `paired_seed_metrics.csv`, `episodes.csv`, `episodes.partial.csv`, `coordination.csv`, `summary.json`, per-model final checkpoints. Manifest remains FAILED with partial files on exceptions. tqdm for oracle enumeration, training and multi-seed evaluation; console tee log beside output.

Smoke pass = complete expected model/evaluation counts, finite losses/metrics, all engineering audits, correct checkpoint reload, mask safety and closed sealed panel. Scientific thresholds never gate smoke. Tests validate DP against brute-force tiny oracle, stochastic event probabilities/terminal costs, observation canonicalization, strict rank-reversal witness, monotonic/signed mixing properties, checkpoint round trip and runner failure/overwrite behavior.

## Workflow and limits
Isolated branch, tests and actual Mac CPU smoke → commit/push → user runs full in Ubuntu terminal → user rsync -avhP --partial. Assistant does not SSH, run full or transfer artifacts. No changes to the main simulator or existing experiments; preserve original checkout edits. This diagnostic cannot prove the cause of adaptation failures, architecture superiority at 3-machine scale, or necessity of switching to central execution. If labels fit but regret persists, inspect execution/local-information constraints; if train fit succeeds but held-out fails, inspect generalization before blaming monotonic representation.
