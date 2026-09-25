# Passive technician duplicate-request fix

## Hypothesis

The learned baseline failure is partly caused by an invalid action interface:
a machine can request a technician while it is already queued or being served.
Rejecting those duplicate requests in the simulator and masking them during
Q-learning/PPO should eliminate invalid requests and reduce avoidable queue
waiting without removing legitimate FIFO requests to a busy technician.

## Locked change and metrics

The transition rejects a nonzero request when the machine is already queued or
assigned to a technician. The local action mask leaves defer valid and masks
those duplicate requests. Requests to busy technicians remain valid and are
queued, preserving the FIFO stress semantics. Training and deterministic
evaluation use the same masks.

Primary smoke checks are zero `invalid_requests` and successful checkpoint
save/load. Secondary diagnostics are objective, waiting, collisions, and
`busy_requests`. No training-seed or evaluation-seed inference is claimed from
the smoke run.

## Verification protocol

The smoke profile uses training seed 11, evaluation seeds 101--103, and the
existing budgets 8, 16, and 32. It must produce the existing timestamped
artifact schema, complete all expected rows, and keep the sealed panel closed.
The fixed stopping rule is the existing budget checkpoint; no full run is
started until the smoke and unit tests pass.

## Interpretation

This change corrects environment/action-interface semantics and is not an
algorithm comparison. A later full run must use a new artifact directory and
compare the corrected baseline against the prior run only as a protocol-change
diagnostic, not as a paired scientific result.
