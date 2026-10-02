# RA-QMIX CF × adaptation factorial — locked v1, 2026-10-02

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent; mode: plan.
- Version: ra_qmix_cf_adaptation_factorial_v1; status: locked before fresh outcomes.
- Motivation: artifact diagnostic commit 81840a9 attributed final nominal cost gap to queue (+0.90) and maintenance (+0.20); no final failure gap. This does not establish CF causality.

## Hypotheses/design
2 × 2: Full (`tqmix`, lambda_cf=0.05) versus no-CF (`tqmix_no_cf`, lambda_cf=0), crossed with nominal_100 versus the locked curriculum_replay. Edge/mixer architecture, parameter count, masks, FIFO/recovery-at-start semantics, observations, gamma=0.95, Adam, schedules, batch/replay capacities and epsilon remain identical. No KL/graph/capacity-matched QMIX in this targeted experiment; it cannot establish capacity-independent architecture superiority.

Primary endpoint: at 480384 steps, per-seed interaction in nominal cost: [(curriculum − nominal)_noCF − (curriculum − nominal)_Full]. H-CF: interaction mean < 0, directional improvement in >=8/10 paired training seeds. Negative indicates less adaptation penalty; report both within-arm gaps and all four absolute costs. A negative interaction alone may reflect a worse no-CF control and is not a successful policy. Practical promotion additionally requires no-CF curriculum nominal cost no higher than Full curriculum, stress-max regret no higher than Full curriculum on average, and the existing within-noCF preservation gate: mean paired nominal relative degradation <=5%, >=8/10 seeds within 10%, stress regret delta mean <0 and improvement in >=8/10. Full's gate reported separately. No change to thresholds after outcomes. No best checkpoint selection.

Secondary: relative nominal interaction, stress-max regret interaction, per-scenario absolute cost, failure/downtime/maintenance/queue components, preventive/corrective starts, request ages, busy requests, service-time choices and Q margins. Mean within eval panel first; training seed is independent unit. Paired Student-t intervals on 10 seed deltas are supportive, not another gate; no p-values/multiple-testing claims. Stress regret uses frozen scenario references, not oracle. Intermediate checkpoints exploratory; no pooling raw costs across horizons.

## Panels, budget, stopping
Full train 85000–85009; evaluation 85100–85149; smoke train 85400, eval 85410–85412. Sealed 201–300 stays closed. Declared prior panels audited from 163 local manifests and frozen in configs/ra_qmix_cf_seed_registry.json. Train episodes use seed*100000 + episode-1; streams can diverge once policies diverge, so paired seeds are not an identical-noise-trajectory guarantee. Seed audit establishes local declared-panel freshness, not unseen external runs.

Full steps 240192/360288/480384; boundary 240192, phase-two batch 112 nominal/16 stress, capacities 43750/6250, epsilon based on steps. Same deterministic scenario schedule across architecture arms. 40 trajectories, 19215360 environment steps, 1584600 training episodes, 120 checkpoints, 24000 evaluation episodes. Smoke 1728/2592/3456 steps; boundary 1728; hidden16/mixer8, batch8 (7/1), replay1024; four trajectories, 1140 training episodes, 12 checkpoints, 144 evaluations. Fixed budgets; abort error/nonfinite/audit failure, preserve FAILED output; no automatic retry, resume or overwrite.

## Mechanism instrumentation
Sample pre-clip TD-gradient norm, weighted-CF gradient norm and cosine on the same training minibatch every 1000 environment steps (smoke24), conditional on an optimizer update. No-CF gradient norm=0, cosine=null. Observer uses autograd.grad, no optimizer steps/RNG draws; test observer leaves training output unchanged. Scalar loss and gradients have different meanings. No intervention based on diagnostics during training. Record per-episode and cumulative optimizer counts distinctly. Replay all fixed checkpoints on existing eval seeds to record decisions and check exact match with primary episode metrics; no extra training/test panels.

## Engineering gates/artifacts
All expected checkpoints/rows/episodes present, unique keys, loss/gradient/Q values finite, zero invalid requests, exact cost sum, same schedules/updates across four arms, shared initialization parameter tensors Full/noCF identical, same boundary state within each architecture's two regimes (NOT across architectures), equal parameter counts, no-CF weighted loss and gradient zero, CF active and sampled. Same-seed training trajectories may differ after CF updates by design.

New timestamped directory: parent manifest/config/seed audit/summary, primary_interactions.csv, algorithm_gates.json, decision traces and derived paired seed tables; each algorithm directory carries existing confirmation artifact schema with checkpoints, training schedules/logs, gradient_diagnostics.csv, evaluations/cost components and audits. Partial evaluation files updated after each block. tqdm for training, multi-seed evaluation and checkpoint replay. No-CF raw CF loss null, not zero. Parent FAILED on partial arm failure; completed arm artifacts retained. Store Git SHA/dirty/runtime/resolved settings. Decision metrics derive only from observable rollout context; longer chosen service does not prove bad assignment.

Workflow: isolated codex/ra-qmix-cf-adaptation-factorial, tests and real Mac CPU smoke, then commit/push; exact Ubuntu CPU compound command and rsync -avhP --partial handed to user. No SSH/full launch/transfer by assistant.
