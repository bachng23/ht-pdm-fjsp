# TD + train-only assignment diagnostic — locked v1, 2026-10-04

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent; user-authorized implementation and Mac smoke.
- Protocol: ra_qmix_td_assignment_v1; locked before outcomes; base8d15799.

## Question, arms, information
Does a consistent TRAIN-only oracle assignment auxiliary loss improve online TD learning without replacing the shared agent or monotonic mixer?
Three arms: monotonic_td (Double-Q TD only), monotonic_td_assignment (same TD+1.0×masked local CE), monotonic_policy_control (CE only; no environmental training, mixer unchanged). All use fixed ID+peer, same ConditionedEdgeQ/original technician-aware monotonic mixer, CF0, hidden64/mixer64, Adam3e-4, clip10, gamma.95. Fixed reward divisor20 in both TD arms; positive scaling preserves discounted reward ordering, gives explicit utility units and avoids fitting raw large costs against unit-weight CE. No fitted normalization, oracle Q targets, bootstrap targets from oracle, warmstart, curriculum, signed mixer or central execution.

Same three known2machine/2technician cells/horizon6/max_age5 as previous diagnostics. Original FIFO/costs/failure laws unchanged; smoke horizon3/max_age4/hidden16. No fresh-environment or scale/adaptation claim.

Create80/20 state split, observation-equivalence groups indivisible, stratified by time. Solve SAT constraints ONLY on training states, one worker, cell dataset seed,30s timeout. Training Q-optimal-action sets come from exact finite oracle: computing a training state's optimal action still uses future model dynamics, including continuations through other states. The teacher excludes held-out state constraints/labels, not future dynamics. Abort UNKNOWN/UNSAT/invalid and verify each training assignment. Hash/store train-state membership and teacher map. Held-out assignments are never constructed. Changing held-out action-values after split cannot change teacher solver inputs/output. Full oracle tables are for labels on TRAIN states and post-training diagnostic evaluation only.

CE uses an independent uniform TRAIN-state minibatch each gradient update. TD-only draws/hash-records the same CE indices but never reads labels or computes CE. TD+CE and CE-only control receive the same teacher minibatch index sequence, seed+23; no held-out CE inputs/labels. Auxiliary batches are not limited to on-policy visited states: this intentionally tests offline supervised anchors plus online RL. Both RL arms can visit teacher-held-out states and learn TD from their real rewards; hence 'heldout' means held out from teacher supervision, NOT unvisited or unseen to RL.

## Budgets and locked hypotheses
Full:60000environment steps/RL model; horizon6→10000episodes/model. Replay50000, float32 observations, learning_starts512, frequency4, one update per trigger, batch128, target sync every500optimizer updates. Epsilon1→.05 over first80%steps, then .05. Reset environment with initial RNG seed+13; exploration RNG seed+19, replay sampling seed+17, teacher seed+23. Same initialization/cadence/RNG initializations across arms; policies cause different experiences, so do NOT claim identical replay contents/trajectories. TD targets use online masked next argmax and detached target-model utility, done zeroes bootstrap. CE control gets exactly same14873optimizer updates, no training envsteps. Final checkpoint only. Smoke120steps, learning_starts16, frequency4,targetinterval10,batch32→27updates/model,40episodes/RLmodel.

Full3cells×3arms×10seeds=90models;60RLmodels×60000=3.6million training interactions,600000training episodes;90×14873=1338570optimizer updates;4500evaluation episodes. Smoke9models,720trainingsteps,240trainingepisodes,243updates,27evaluation episodes. Fixed stop; failures preserve partial/FAILED outputs, no retry/resume/overwrite.

Primary practical gate: paired TD+CE−TD EXACT expected undiscounted initial-state policy cost, equal mean weighting shared_preference/pressure per training seed. Need absolute reduction>=2cost units, relative reduction>=10%, improvement>=8/10seeds; nominal mean cost <=1.10×TD baseline. Supportive Student-t pairedCI95%,df9,t2.262157163; not a p-value threshold.

CE-only control gate: EACH cell mean TRAIN-state regret<=.10 and >=8/10seeds have>=99%training oracle-optimal decisions. Control fit, not held-out transfer/cost, gates empirical attribution of the primary contrast. Poor teacher-held-out performance is reported separately and does not silently alter control gate. Primary FAIL/poor control do not identify a unique TD/architecture cause.

Secondary: exact discounted costs, all/train/teacher-heldout regret/optimal fractions, critical symmetric-state assignments, rollout costs/collisions/failures/waiting, replay reward/coverage, fraction supervision-heldout visited in RL, TD/CE/gradient traces and nominal seed tails. CE-control versusRL cost descriptive. No best-seed/heldout tuning or oracle labels into TD targets.

## Seeds, artifacts and verification
Training98000–98009; evaluation98200–98249; dataset/solver98100/98101/98102. Smoke98400/eval98410–98412. Freeze prior local registry before outcomes including policy-supervision run; sealed201–300 closed. Fresh panels within local declared history, same known environment population.

New UTC timestamp directory only. Root manifest/resolved_config/seed_audit/feature_contract/summary; parameter_counts,training_progress,training_episodes,teacher_batches,state_metrics,model_metrics,paired_seed_metrics,episodes.partial/episodes/coordination CSVs. Per-cell oracle_states/oracle_q, training-only policy_feasibility.json, train_teacher_actions.json; per-cell/arm/train_seed_N/model.pt and training_coverage.json. Checkpoints store protocol/objective/config/settings/reward_scale/teacherSHA/splitSHA/featurecontract/state_dict. No held-out teacher labels in checkpoints or teacher artifacts. Training and multi-seed evaluation tqdm.

Tests: train-only teacher invariance to heldoutQ/constraint changes, consistent labels/unknown handling, masking and exact terminal/nonterminal Double-Q targets, no target gradient, CE restrictedtrain, same initial tensors, RL cadence/epsilon/replay/rng streams, CE-only mixer unchanged, primary guard/control, checkpointmetadata/reload, failure/no-overwrite. Real Mac smoke engineering gate: declared counts/updates/envsteps, finite logs, bitwise Q/agent reload, teacherbatch index hashes matching, verifiedtrainteacher/membership, invalid/cost errors0, targetupdates and heldout-supervision count0, outputs/progress/sealedpanel. Scientific gates disabled in smoke.

Scope: oracle training supervision is privileged, not a deployable/scalable teacher claim. No teacher at execution. Independent held-out teacher construction fixes the previous transductive constraint leakage, but RL experience can include those states; separate these information contracts. Supervised control's mixer/value calibration has no meaning. User authorization overrides ARS executor's generic no-script-generation guidance.

Canonical workflow: tests+Macsmoke→verifiedcommit/push→exactuser-run CPUlabcompound command→user-run rsync-avhP--partial. Assistant never SSH/full-launches/transfers. Preserve root checkout changes and all prior raw artifacts.
