# Corrected passive-technician baseline diagnostic

## Research question and hypotheses

After duplicate-request semantics and action masking are corrected, do the
learned baselines stop producing invalid requests and stop collapsing to a
single fixed joint action? The primary diagnostic hypothesis is that masked
training yields zero duplicate requests. A secondary hypothesis is that
Centralized PPO has greater joint-action diversity and lower queue waiting
than the pre-fix behavior.

## Locked protocol

The environment is the three-machine/two-technician FIFO stress cell. The
compared policies are `random_feasible`, `skill_aware_fifo`, masked
Independent Q, masked Independent PPO, and masked Centralized PPO. Training
seeds are 11, 12, and 13; evaluation seeds are 101--200; budgets are 5,000,
10,000, and 20,000 episodes. Every checkpoint is evaluated on the same common
development panel. No sealed test seeds are opened and no adaptive checkpoint
selection is used.

The primary diagnostic metric is `invalid_requests`, which must be zero after
the semantics fix. Secondary mechanism metrics are `busy_requests`, objective,
failures, jobs, queue waiting, collisions, the number of unique joint actions
per episode, defer fraction, and request count. The stopping rule is the
locked training budget.

## Smoke and full-run gates

The local smoke run uses seed 11, evaluation seeds 101--103, and budgets 8,
16, and 32. It must pass the unit suite, complete all 33 expected rows, save
and reload all nine checkpoints, record the diagnostic columns, and leave the
sealed panel closed. A full run is valid only when its timestamped manifest is
`COMPLETED`, all expected rows and 27 checkpoints are present, and every
learned-policy row has `invalid_requests=0`.

## Interpretation limits

This is a corrected-baseline diagnostic, not a final algorithm comparison.
The prior unmasked run is not a paired scientific control because the action
interface changed. If invalid requests disappear but joint-action diversity
remains low and objective cost remains high, the next experiment should target
credit assignment or joint coordination rather than more training episodes.
