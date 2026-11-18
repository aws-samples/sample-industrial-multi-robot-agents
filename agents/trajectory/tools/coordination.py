"""
Coordinated multi-arm execution — every arm follows its OWN path, and the arms
are kept apart by ORDER, not by detours.

The stepwise flow couples arms geometrically: a conflicting pair is resolved by a
detour, a park at HOME, an interlock pose retracted toward HOME, or by running the
arms one after the other. All of those move the arms away from the part and most
of them throw parallel time away. But an inter-arm conflict is almost always a
question of WHEN, not WHERE: both arms want the same space at the same moment.

So here each arm's legs are planned on their own — tight to the part, shaped by the
approach / leave strategy the planner LLM chose — and the arms are coordinated in
time (path–velocity decomposition):

  * Each arm's program is a chain of OCCUPANCY ITEMS: a move (the leg's swept
    volume) and a stand (the pose it holds after the leg, until its next move).
  * Two items of different arms CONFLICT if any configuration of one comes within
    the cell's warning band of any configuration of the other — a time-agnostic,
    swept-volume test using the same spheres, radius and band as
    ``inter_robot_clearance`` and the runtime monitor.
  * A discrete-event simulation starts every move as early as its conflicts allow:
    first come, first served, an arm that must wait holds WHERE IT STANDS (at its
    last weld, i.e. near the part). Waits are therefore short and local.
  * Every conflicting pair is turned into a GATE on the arm that goes second: it
    waits until the other arm has COMPLETED a specific operation. Gates are events,
    not clock times, so the order — and with it the separation — holds whatever
    the real execution speed. The gates come from one simulated run, so they are
    consistent and cannot deadlock.
  * If the simulation itself deadlocks — an arm stands where another must pass,
    and cannot move on until that other arm has passed — no ordering resolves it.
    The report names the arm and the operation after which it needs a SAFE SPOT: a
    pose near the part, clear of the other arm's path. Choosing that spot is the
    planner LLM's job; it plans it as a ``move`` in the arm's sequence and
    coordinates again.

The scheduler core (:func:`schedule`) is pure Python with the conflict test
injected, so it is testable without Pinocchio.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Optional

# The planning mode of the run in progress, bound by the planner (agent.py) so a
# tool can adapt without the LLM passing it: in coordinated mode the MRTA
# pipeline's step composition is never used (only allocation + per-arm order),
# so the pipeline skips that expensive search.
_MODE: ContextVar[Optional[str]] = ContextVar("planning_mode", default=None)


@contextmanager
def bind_planning_mode(mode: Optional[str]):
    token = _MODE.set(mode)
    try:
        yield mode
    finally:
        _MODE.reset(token)


def current_planning_mode() -> Optional[str]:
    return _MODE.get()

HOME = "__home__"

#: Seconds the executor spends on an operation beyond its trajectory: the hold
#: (hold_duration_s, per op) plus its completion slack (ExecConfig
#: completion_margin_s). Used only to ESTIMATE the schedule — safety comes from
#: the gates, which do not depend on it.
DEFAULT_SETTLE_S = 1.0


@dataclass
class Item:
    arm: str
    index: int            # position in the arm's item chain
    kind: str             # "move" | "stand"
    op_id: str            # the operation this move performs / this stand follows
    duration: float = 0.0


@dataclass
class ArmPlan:
    arm: str
    ops: list                          # operation ids, in execution order
    durations: dict                    # op_id -> seconds (trajectory + hold + settle)
    gates: dict = field(default_factory=dict)   # op_id -> {other_arm: op_id to await}
    starts: dict = field(default_factory=dict)  # op_id -> simulated start time
    waits: dict = field(default_factory=dict)   # op_id -> seconds waited before it


def _items(arm: str, ops: list, durations: dict) -> list:
    """HOME stand, then move/stand per operation."""
    chain = [Item(arm, 0, "stand", HOME)]
    for op in ops:
        chain.append(Item(arm, len(chain), "move", op, float(durations.get(op, 0.0))))
        chain.append(Item(arm, len(chain), "stand", op))
    return chain


def schedule(programs: dict, conflict: Callable[[Item, Item], bool]) -> dict:
    """Coordinate arms that each follow their own operation sequence.

    Args:
        programs: ``{arm: (ops, durations)}`` — ops in order, durations per op.
        conflict: ``conflict(item_a, item_b)`` for items of DIFFERENT arms.

    Returns ``{"arms": {arm: ArmPlan}, "makespan_s", "parallel_s",
    "deadlock": None | {...}}``. On deadlock the plans cover the operations
    scheduled so far, and ``deadlock`` names the blocked arm, the operation it
    is waiting to start, and the arm/item in its way.
    """
    arms = sorted(programs)
    chains = {a: _items(a, list(programs[a][0]), programs[a][1]) for a in arms}
    plans = {a: ArmPlan(a, list(programs[a][0]), dict(programs[a][1])) for a in arms}
    pos = {a: 0 for a in arms}              # index of the arm's CURRENT item
    busy_until = {a: 0.0 for a in arms}     # end time of its current move
    done: dict = {a: [] for a in arms}      # items of this arm already FINISHED
    t = 0.0
    motion = []                             # (start, end) of every move, for stats

    def active(b):
        """Items of ``b`` that are, or will be, occupied while it is on its
        current item: a move is followed by its stand."""
        cur = chains[b][pos[b]]
        if cur.kind == "move":
            # The stand it is leaving lasts until this move completes (the move
            # starts from that pose), and the stand after it follows.
            return [chains[b][pos[b] - 1], cur, chains[b][pos[b] + 1]]
        return [cur]

    def finished(b):
        """Items arm ``b`` is fully done with: a move once it completed, a stand
        once the move AFTER it completed (the arm has left the pose)."""
        return [it for it in chains[b]
                if (it.kind == "move" and it.index < pos[b])
                or (it.kind == "stand" and it.index + 1 < pos[b])]

    def finished_event(b, item):
        """The operation whose COMPLETION ends ``item`` of arm ``b``: a move ends
        when that op completes; a stand ends when the NEXT op completes (the arm
        has moved off the pose)."""
        if item.kind == "move":
            return item.op_id
        nxt = item.index + 1
        return chains[b][nxt].op_id if nxt < len(chains[b]) else None

    while True:
        # Moves that finish now: the arm is on its stand.
        for a in arms:
            cur = chains[a][pos[a]]
            if cur.kind == "move" and busy_until[a] <= t + 1e-9:
                done[a].append(cur)
                pos[a] += 1
        progressed = False
        waiting = []
        order = sorted(arms, key=lambda a: (-(len(chains[a]) - pos[a]), a))
        for a in order:
            cur = chains[a][pos[a]]
            if cur.kind != "stand" or pos[a] + 1 >= len(chains[a]):
                continue
            move, stand = chains[a][pos[a] + 1], chains[a][pos[a] + 2]
            blocked_by = None
            for b in arms:
                if b == a:
                    continue
                for other in active(b):
                    if conflict(move, other) or conflict(stand, other):
                        blocked_by = (b, other)
                        break
                if blocked_by:
                    break
            if blocked_by:
                waiting.append((a, move, blocked_by))
                continue
            # Start it. Gate on every FINISHED item of another arm that conflicts
            # with this move or the stand after it — the order the simulation
            # relied on; the gate makes the execution keep it at any speed.
            gate = {}
            for b in arms:
                if b == a:
                    continue
                last = None
                for other in finished(b):
                    if conflict(move, other) or conflict(stand, other):
                        ev = finished_event(b, other)
                        if ev is not None and (last is None or
                                               plans[b].ops.index(ev) > plans[b].ops.index(last)):
                            last = ev
                if last is not None:
                    gate[b] = last
            p = plans[a]
            p.starts[move.op_id] = t
            if gate:
                p.gates[move.op_id] = gate
            prev_end = motion_end_of(a, done, chains, p)
            p.waits[move.op_id] = max(0.0, t - prev_end)
            pos[a] += 1
            busy_until[a] = t + move.duration
            motion.append((a, t, t + move.duration))
            progressed = True
        if progressed:
            continue
        running = [busy_until[a] for a in arms
                   if chains[a][pos[a]].kind == "move" and busy_until[a] > t + 1e-9]
        if running:
            t = min(running)
            continue
        if waiting:
            a, move, (b, other) = waiting[0]
            blocker_stand = other.kind == "stand"
            return _result(plans, motion, deadlock={
                "arm": a, "waiting_to_start": move.op_id,
                "after_op": chains[a][move.index - 1].op_id,
                "blocked_by_arm": b, "blocked_by": f"{other.kind} {other.op_id}",
                # The arm IN THE WAY needs somewhere else to stand: a safe spot
                # near the part, clear of the waiting arm's path.
                "needs_safe_spot": ({"arm": b, "after_op": other.op_id}
                                    if blocker_stand else {"arm": a,
                                                           "after_op": chains[a][move.index - 1].op_id}),
            })
        return _result(plans, motion, deadlock=None)


def motion_end_of(a, done, chains, plan) -> float:
    """When arm ``a`` finished its last move (0 at HOME)."""
    moves = [i for i in done[a] if i.kind == "move"]
    if not moves:
        return 0.0
    last = moves[-1]
    return plan.starts.get(last.op_id, 0.0) + last.duration


def _result(plans, motion, deadlock):
    makespan = max((e for _, _, e in motion), default=0.0)
    # Seconds in which two or more arms are moving at once.
    events = sorted([(s, 1) for _, s, _ in motion] + [(e, -1) for _, _, e in motion])
    par, depth, last = 0.0, 0, 0.0
    for tt, d in events:
        if depth >= 2:
            par += tt - last
        depth += d
        last = tt
    return {"arms": plans, "makespan_s": round(makespan, 2),
            "parallel_s": round(par, 2), "deadlock": deadlock}


# ─── the geometric conflict model ─────────────────────────────────────────────

def geometric_conflict(cell_id: str, trajectories: dict, home: dict,
                       *, max_samples: int = 24) -> Callable[[Item, Item], bool]:
    """``conflict(item_a, item_b)`` from the arms' real kinematics.

    ``trajectories``: ``{arm: {op_id: [{"positions", "time_s"}, ...]}}`` — the
    planned legs. ``home``: ``{arm: joints}``. A move's occupancy is the sphere
    model sampled along its leg; a stand's is the sphere model at the leg's end.
    Two items conflict when their closest spheres come within the cell's warning
    band (clearance = centre distance − 2 × sphere radius, exactly as
    ``inter_robot_clearance`` computes it).
    """
    import numpy as np
    from agents.trajectory.tools.robot_kinematics import (
        _config_spheres, _load_collision_config, _load_pinocchio_model,
        _load_robot_config,
    )
    from agents.trajectory.tools.time_parameterization import _sample_at_time

    band = float(_load_collision_config(cell_id)["warning_distance"])
    radius = 0.12
    geo = {}
    for arm in trajectories:
        model, _ = _load_pinocchio_model(cell_id, arm)
        cfg = _load_robot_config(cell_id, arm)
        geo[arm] = (model, model.createData(), np.array(cfg["base_position"]),
                    float(cfg.get("base_orientation_yaw", 0.0)), cfg.get("tool_collision"))

    cloud: dict = {}

    def spheres(arm, q):
        model, data, base, yaw, tool = geo[arm]
        return np.array(_config_spheres(model, data, np.asarray(q, float), base, yaw, tool))

    def occupancy(item: Item):
        key = (item.arm, item.kind, item.op_id)
        if key in cloud:
            return cloud[key]
        if item.op_id == HOME:
            pts = spheres(item.arm, home[item.arm])
        else:
            traj = trajectories[item.arm][item.op_id]
            if item.kind == "stand":
                pts = spheres(item.arm, traj[-1]["positions"])
            else:
                t_end = float(traj[-1].get("time_s", 0.0))
                n = max(2, min(max_samples, int(t_end * 10) + 2))
                pts = np.vstack([spheres(item.arm, _sample_at_time(traj, t_end * k / (n - 1)))
                                 for k in range(n)])
        cloud[key] = pts
        return pts

    memo: dict = {}
    thr = band + 2 * radius
    boxes: dict = {}

    def box(item: Item, pts):
        key = (item.arm, item.kind, item.op_id)
        if key not in boxes:
            boxes[key] = (pts.min(0), pts.max(0))
        return boxes[key]

    def conflict(a: Item, b: Item) -> bool:
        key = ((a.arm, a.kind, a.op_id), (b.arm, b.kind, b.op_id))
        if key not in memo:
            pa, pb = occupancy(a), occupancy(b)
            memo[key] = memo[(key[1], key[0])] = _clouds_within(
                pa, pb, thr, box(a, pa), box(b, pb))
        return memo[key]

    return conflict


def _clouds_within(pa, pb, thr: float, box_a=None, box_b=None) -> bool:
    """``min |a - b| <= thr`` over two point clouds — the coordinator's conflict
    test (``d.min() - 2r <= band`` with ``thr = band + 2r``), without the full
    distance matrix it used to build for every pair (13,245 pairs, 11.5 s of an
    11.8 s coordination on welding_cell_2_small).

    1. The clouds' bounding boxes: the gap between the boxes is a lower bound on
       every point-to-point distance, so boxes further apart than ``thr`` cannot
       conflict — most item pairs, which belong to arms far apart.
    2. Otherwise squared distances against ``thr**2`` (no square roots), in
       chunks that stop at the first pair within reach — compiled
       (``fastgeom``) when numba is available.

    The same answer as the full matrix (tests/trajectory/test_vectorised_collision.py)."""
    import numpy as np
    pa = np.asarray(pa, float).reshape(-1, 3)
    pb = np.asarray(pb, float).reshape(-1, 3)
    if not len(pa) or not len(pb):
        return False
    lo_a, hi_a = box_a if box_a is not None else (pa.min(0), pa.max(0))
    lo_b, hi_b = box_b if box_b is not None else (pb.min(0), pb.max(0))
    gap = np.maximum(0.0, np.maximum(lo_b - hi_a, lo_a - hi_b))
    if float(gap @ gap) > thr * thr:
        return False
    from agents.trajectory.tools import fastgeom
    if fastgeom.AVAILABLE:
        return fastgeom.moving_first_hit(pa, pb, np.full(len(pb), np.nextafter(thr, np.inf))) \
            is not None
    t2 = thr * thr
    for i in range(0, len(pa), 256):
        diff = pa[i:i + 256, None, :] - pb[None, :, :]
        if (np.einsum("ijk,ijk->ij", diff, diff) <= t2).any():
            return True
    return False
