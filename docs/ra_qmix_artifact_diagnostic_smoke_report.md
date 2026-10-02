# Mac smoke verification — 2026-10-02

- Branch: codex/ra-qmix-artifact-diagnostic; base/source commit: 33ab6196f0dd4d4e12038b74b65fd3c5782096f7.
- Locked protocol: docs/ra_qmix_artifact_diagnostic_plan.md.
- Environment: macOS ARM64, Python 3.12, torch 2.14.0; isolated worktree uv sync --frozen --extra dev passed.
- Repository suite: 289 passed, one existing tqdm rich experimental warning, 143.63 seconds. Nine final diagnostic tests passed separately, including three preflight tests added after suite collection.
- Installed CLI smoke: uv run --frozen ht-pdm-fjsp-passive-artifact-diagnostic, CPU, source confirmation artifacts, train seed 84600, eval seeds 84700–84702.
- Output: artifacts/ra_qmix_artifact_diagnostic_smoke_20261002T015057Z and adjacent console log. Previous smoke directories preserved.
- 72/72 episode rows, 3564 decision rows, six checkpoints loaded, all replayed numeric metrics match source within 1e-6, shared phase-boundary network states equal, zero invalid requests/cost reconciliation errors, source SHA256 checks unchanged, sealed panel closed.
- Source training log streamed: 792300 records; only selected training seed summarized. Historical gradients and replay-state features explicitly unavailable.
- No training, source edits, full lab run, SSH or rsync performed. Smoke is engineering evidence, not a mechanism conclusion.
- Full scope: 12000 replay episodes, 60 unique source checkpoints, 594000 decision rows, plus paired seed summaries and source-log aggregates. Expected artifact size roughly tens of MB, dominated by decision traces; no model copies.
