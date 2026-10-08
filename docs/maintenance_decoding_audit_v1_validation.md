# Engineering validation — decoding audit v1

Date: 2026-10-08. Original source commit: 4b1cffb0cc1e2413e27f0e2c3a7730f8d3e3d7a8.
New branch: codex/maintenance-decoding-audit-v1.

- `uv sync --frozen --extra dev`: checked 33 packages, unchanged lockfile.
- Full regression suite: 585 passed in 194.29 seconds; one inherited tqdm/rich warning.
- Focused decoder/runner tests after auxiliary summaries and final input guard:
  12 passed in 5.75 seconds.
- df9 two-sided 97.5% critical value independently checked using 40-digit
  regularized incomplete-beta inversion: 2.6850108468164553.
- Contrived policy test: greedy probability .30, joint-MAP probability .396;
  independent teacher-forced likelihood tests cover both cooperative actors,
  rotating order, occupied technicians, skill masks and K4.
- Legacy greedy decoder unchanged; physical episode totals match the original
  runner for the same state/config/shock/checkpoint.
- Tampered checkpoints rejected before evaluation; FAILED manifest and partial
  output preserved for injected crashes; original inputs hash-identical after smoke.

Real full-checkpoint Mac smoke:
`artifacts/maintenance_decoding_audit_v1_smoke_cpu_20261008T134750Z`.
COMPLETED: 4 checkpoints, zero training/optimizer steps, 128 episodes,
640 physical decisions, 32 decoder audits, 384 common-state latency rows.
Invalid assignments: zero. Maximum probability-mass error 9.574768256026545e-7;
maximum MAP numerical selection regret 4.76837158203125e-7.
49 artifacts / 1014878 bytes, all output hashes verified. 34 source artifacts
verified (27 snapshots + 3 selection/config metadata + 4 checkpoints).
Fresh full shock panel remained sealed; engineering profiles belong to the full
training pool. Smoke shows compatibility, not scientific improvement.

Full run is handed off to the human lab terminal. Expected 40 checkpoints,
317440 episodes, 5713920 decisions, 960 decoder audits, 8928 latency rows.
No full lab execution, SSH or rsync performed by assistant.
