# Mac verification — CF × adaptation factorial, 2026-10-02

- Branch: codex/ra-qmix-cf-adaptation-factorial; base: 81840a91bf8bee6ccd03c6cb0a434412c35133fc.
- Protocol: docs/ra_qmix_cf_adaptation_factorial_plan.md.
- Locked dependencies: uv run --frozen, Python 3.12/macOS ARM64, torch 2.14.0, CPU with one torch thread.
- Targeted tests: 24 passed, including original confirmation and artifact diagnostic compatibility.
- Repository suite: 301 passed in 172.46 seconds; three dependency warnings (boolean inversion deprecation and tqdm rich experimental warning), no failures.
- Final factorial tests: 9 passed in 26.14 seconds after adding the planned secondary stress interaction to summary.
- CLI smoke: artifacts/ra_qmix_cf_adaptation_smoke_20261002T021140Z, adjacent console log retained. Four trajectories, 1140 training episodes, 12 checkpoints, 144 evaluation episodes and 7128 decision records.
- Each trajectory has 1725 optimizer updates; schedules and counts match across algorithms/regimes. Full/no-CF initial parameter tensors and parameter counts identical. Within each algorithm, control/curriculum boundary checkpoint equal.
- 288 gradient samples per algorithm (two regimes combined); Full has active CF gradients, no-CF has zero weighted CF loss/gradient and null cosine. No-CF raw CF loss missing/null by design. Observer-invariance test confirms identical final parameter tensors with gradient recording enabled/disabled.
- All feasibility, cost reconciliation, coverage, trace-to-evaluation reconciliation, fresh-seed and sealed-panel audits passed. No-CF is not promoted based on smoke; gates are null for this profile.
- Source seed registry covers 163 prior local manifests, 6377 declared seed values; new full/smoke panels have no overlap. Prior declared-panel audit does not assert knowledge of untransferred external runs.
- No full run, SSH or rsync performed. The user's main checkout and unrelated uncommitted changes remain untouched.
- Full scope: 40 trajectories, 19,215,360 environment steps, 1,584,600 training episodes, 120 checkpoints, 24,000 evaluation episodes, 1,188,000 decision rows. Decision traces stream to disk. Logs dominate artifact size; plan roughly 1–2 GB for the run rather than assuming smoke size scales directly.
