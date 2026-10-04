# TD + assignment Mac smoke — 2026-10-04

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent, user-authorized code implementation/run.
- Verification Status: EXECUTED Mac smoke; full experiment pending user lab execution.
- Protocol: ra_qmix_td_assignment_v1; locked before outcomes.
- Base:8d15799, experiment edits uncommitted when smoke ran as required by workflow.
- Smoke artifact: `artifacts/ra_qmix_td_assignment_smoke_20261004T053249Z` in `/private/tmp/marl-oracle-representation-20261003/`.
- Completed: 2026-10-04T05:32:52.137166+00:00, CPU/macOS arm64/Python3.12.13, torch2.14.0,OR-Tools9.15.6755.

## Engineering results
Full repository suite: **351 passed**, one existing SB3 tqdm rich experimental warning,170.98s.
9/9models/checkpoints,720/720training environment steps,240training episodes,243/243optimizer updates,27/27evaluation episodes,348state-model metrics; all45manifest outputs exist. Training/evaluation tqdm confirmed in console log. Every engineering audit passed: final budgets, target sync cadence, same initial model tensors, teacher-batch hashes matched, train-only optimal consistent teacher, no heldout teacher labels, bitwise checkpoint Q/agent reload, CE-only mixer unchanged, finite TD/CE/gradient logs, invalid requests0, real training rewards reconciled step-by-step, evaluation costs reconciled, sealed201–300closed.

Scientific primary/control gates disabled (null) in smoke. No threshold,weight,rewardscale,budget or teacher rule tuned from smoke performance. Tests verify heldout constraint/value changes cannot affect train-only teacher, terminal/nonterminal masked detached Double-Q targets, float32 ring replay/nooraclefields, TD-only never computes CE, loss masks/gradient paths, budgets/epsilon/shared initialization, primarycost/control/nominal guards, checkpointcontract,failure/noretry/nooverwrite. Full-config teacher test covers375/381/584reachable populations with train-only constraints and no full training.

## Full scope and limitations
90models,60RLmodels×60000steps=3.6million interactions,600000training episodes,1338570optimizer updates,4500evaluation episodes,40200state-model rows. Artifacts roughly100–200MB, dominated by streaming trainingepisodesCSV and90checkpoints. CPU may require several hours. Float32 tensor replay ring and chunked append of training episodes keep memory bounded; no retention of all600000episode rows in RAM.

Rewards divided by fixed20 for both TD arms, no data-derived normalization. Assignment CE weight1, independent uniform training-state anchor batches; supervised control receives14873updates and no training env interactions. Common RNG initializations do not mean same replaycontents/trajectories. Heldout states are closed to teacher/CE labels but may be visited/TD-trained online. Train oracle optimal actions use known future dynamics; still privileged supervision, not a scalable teacher implementation.

Fresh panels98000–98009/98200–98249/98100–98102; local registry170prior manifests. Same known benchmark population; sealed201–300closed. No SSH, full launch, or rsync by assistant. Branch/commit and both required user-run commands supplied after verified push.
