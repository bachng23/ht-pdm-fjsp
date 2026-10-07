"""Per-demand censoring, explicit logical wire model and conditional inference."""

import math
import numpy as np


class Requests:
    """Failures arrive at boundaries; service-start, not completion, ends waiting."""

    def __init__(self, state):
        self.pending = {}
        self.rows = []
        self.sequence = [0] * len(state.ages)
        for m, failed in enumerate(state.failed):
            if failed and m not in state.assigned:
                self._arrive(m, 0)

    def _arrive(self, machine, arrival):
        if machine in self.pending:
            raise RuntimeError("duplicate outstanding corrective demand")
        self.sequence[machine] += 1
        self.pending[machine] = dict(machine=machine, request=self.sequence[machine], arrival=arrival)

    def step(self, time, before, pairs, after):
        for m, _ in pairs:
            if before.failed[m]:
                if m not in self.pending:
                    raise RuntimeError("corrective service without outstanding request")
                row = self.pending.pop(m)
                self.rows.append({**row, "end": time, "wait": time - row["arrival"], "served": True, "censored": False})
        servicing = {m for m in before.assigned if m >= 0} | {m for m, _ in pairs}
        for m, failed in enumerate(after.failed):
            if failed and not before.failed[m] and m not in servicing:
                self._arrive(m, time + 1)

    def finish(self, horizon):
        for row in self.pending.values():
            self.rows.append({**row, "end": horizon, "wait": horizon - row["arrival"], "served": False, "censored": True})
        self.pending.clear()
        return self.rows


WIRE_CONTRACT = dict(
    version="logical_wire_v1", numeric_bytes=4,
    endpoint="logical point-to-point payload bytes; excludes packet headers, retransmission and transport latency",
    static="once per episode: 9 global scalars + N*K*(compatibility,duration,restoration); destinations below",
    machine_telemetry="4 scalars/machine: age,failed,wait,service remaining",
    worker_telemetry="3 scalars/worker: free,remaining,owner; 2 scalars for local actor broadcast (free,remaining)",
    dispatch="2 int32 addresses per physical start sent separately to machine and technician",
    reservation="4 serial turns even forced WAIT; K int32 availability flags down + chosen worker int32 up",
    centralized="machine+worker telemetry to controller; dispatch back to both endpoints",
    local="worker status to resource coordinator and broadcast availability/remaining to N agents; own state remains local",
    full_cooperative="central telemetry plus full dynamic telemetry broadcast to all N agents and full static provisioning per agent",
    rounds="logical sequential phases, not measured network synchronization or decentralized runtime",
)


def communication(c, name, pairs, first=False):
    n, k = c.machines, c.technicians
    serial = name in ("independent_local_ppo", "cooperative_local_ppo", "cooperative_full_ppo")
    local = name in ("independent_local_ppo", "cooperative_local_ppo")
    telemetry = (3 * k + 2 * k * n) * 4 if local else (4 * n + 3 * k) * 4
    broadcast = 0 if local or not serial else n * (4 * n + 3 * k) * 4
    reservation = n * (k + 1) * 4 if serial else 0
    dispatch = len(pairs) * 2 * 4 * 2
    if local:
        static = n * (9 + 3 * k) * 4
    else:
        static = (9 + 3 * n * k) * 4 * (n + 1 if serial else 1)
    static = static if first else 0
    return dict(
        telemetry_bytes=telemetry, broadcast_bytes=broadcast,
        reservation_bytes=reservation, dispatch_bytes=dispatch,
        static_provisioning_bytes=static,
        communication_bytes=telemetry + broadcast + reservation + dispatch + static,
        communication_rounds=(2 + n + int(not local)) if serial else 2,
    )


def interval(values, confidence):
    """Frozen df9/df31 t quantiles; no unpinned statistical dependency.

    Quantiles verified by high-precision inversion of regularized incomplete beta.
    Only the two preregistered sample sizes support scientific intervals.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return None
    if not np.isfinite(x).all():
        raise ValueError("nonfinite paired differences")
    critical = {
        (10, 0.95): 2.2621571627982055,
        (10, 0.9875): 3.1109348231794737,
        (32, 0.95): 2.0395134463964085,
        (32, 0.9875): 2.6519126890308002,
    }[(len(x), confidence)]
    half = critical * float(x.std(ddof=1)) / math.sqrt(len(x))
    return [float(x.mean()) - half, float(x.mean()) + half]
