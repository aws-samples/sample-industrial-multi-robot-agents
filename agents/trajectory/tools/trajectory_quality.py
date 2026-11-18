"""The single owner of "is this calculated trajectory any good?".

A plan that executes is not the same as a plan worth executing. Every motion in
this repo is produced by a sampling-based planner and then shortcut, and the
failure mode of that pipeline is silent: the path is valid, collision-free, and
four times longer than it needs to be. Nothing in the tree says so. The plan is
accepted, saved, replayed, and the cell simply runs slow forever.

Measured on ``welding_cell_2_small``'s ``bt_1788022470`` plan, which passed every gate:

* ``home_robot2`` travels **25.87 rad** of joint space to cover a **6.06 rad**
  move — a 4.3x detour, 362 waypoints, 17.98 s. Its peer ``home_robot1`` covers a
  *longer* move (7.36 rad) in 7.67 rad and 5.25 s: a 1.04x ratio, essentially
  optimal. Same arm model, same home pose, same planner. One number separates
  them, and it was not being computed.
* The tree spends **59.5 %** of its makespan with only one arm moving, and
  robot1 idles **50.5 %** of the program.

Both are efficiency defects a human found by hand, once, after the fact. This
module computes them for every plan, every time, and writes the answer beside the
plan (see :func:`write_quality_log`).

WHICH KPIs, AND WHY THOSE
=========================
The metrics here were chosen because each one *diagnoses a specific defect this
pipeline actually produces*, not because it is conventional to report it. A
metric nobody can act on is noise in a log file.

**Path efficiency — the primary group.** ``detour_ratio`` is the travelled
joint-space path length over the straight-line joint-space distance between the
same endpoints. It is the standard post-shortcut quality measure for a
sampling-based planner, and it is the one that catches the defect above. Ideal is
1.0; a shortcut RRT path should land inside ~1.15. ``waypoints_per_rad`` catches
the same defect from the other side: a bloated path carries waypoints in
proportion to its wandering (60/rad on the bad leg, 14/rad on the good one).

**Schedule efficiency.** ``parallel_efficiency`` and ``serial_fraction`` are the
multi-arm answer: a per-leg-perfect plan that runs the arms one at a time is not
efficient. These are computed from a real timeline (see :func:`schedule_metrics`)
that honours ``Sequence`` / ``Parallel`` / ``Delay``, because the number that
matters is when each arm is *actually* moving. Note this is deliberately NOT the
``estimated_makespan_s`` in ``run_logger.summarize_behaviour_tree``, which is the
longest single operation (7.10 s on the plan above) and not a makespan at all.

**Time efficiency.** ``speed_utilisation`` against the arm's declared
``joint_velocity_limits`` separates "the path is long" from "the path is slow" —
two different repairs. ``interior_dwell_s`` is time the arm spends stationary in
the *middle* of a leg, which is pure loss; it exists because concatenating
per-operation pieces leaves a stalled seam at each join.

**Smoothness.** ``dimensionless_jerk`` is scale-invariant, so legs of different
length and duration are comparable. Absolute jerk thresholds are not portable
between arms, so this module never invents one: smoothness is flagged by
*relative outlier* only — a leg far above the median of its peers in the same
plan. That is what makes the finding actionable ("this leg, unlike its siblings")
instead of arguable ("this number exceeds a constant someone picked").

**Continuity.** ``max_seam_gap_rad`` is the joint distance between where one leg
ends and where the same arm's next leg begins. Non-zero means a teleport the
executor has to absorb.

SAFETY IS REPORTED, NEVER SCORED
================================
Clearances are recorded in the ``safety`` block and are deliberately excluded
from every score and from :data:`WEIGHTS`. A composite score is a thing people
optimise, and a score that rewards a shorter path for passing closer to a fixture
turns a safety margin into a tunable. Margins are set by measurement elsewhere;
this module only ever reports what a plan achieved.

BACKEND-NEUTRAL ON PURPOSE
==========================
Everything here is numpy-only and reads plain dicts. Cartesian metrics need
forward kinematics, which would drag in a kinematic model, so FK arrives as an
optional ``fk`` callback — the same idiom (and for the same reason) as
``task_allocation._regional_allocate``'s ``can_service``. Without it the module
still answers every joint-space question, which is where the defects have been.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from agents.trajectory.tools.btcpp import (
    CONTROL_NODES,
    DECORATOR_NODES,
    children as bt_children,
    node_kind,
    port as bt_port,
    trajectory_of,
)
# "Did this leg wander?" already has an owner, and it is not this module. The
# OMPL backend picks candidates with these, so a quality report that measured
# wandering its own way would grade the planner against a different ruler than the
# planner used. They are backend-neutral (tools/path_metrics.py), so the report
# scores any backend's plan with the same ruler and needs no backend importable.
from agents.trajectory.tools.path_metrics import (
    _path_directness as path_directness,
    _path_joint_length as path_joint_length,
    _trajectory_excess_travel as trajectory_excess_travel,
)

# ── thresholds ───────────────────────────────────────────────────────────────
#
# Every constant here is a REPORTING threshold: it decides what gets called out,
# never what a planner is allowed to do. None of them is a safety limit.

#: There is deliberately NO detour-RATIO threshold here. The ratio is reported —
#: it is the number a reader recognises — but nothing is graded on it, so no
#: constant exists for a future change to grade on by accident.
#:
#: Excess travel in RADIANS is what a wandering finding is graded on, because a
#: ratio inflates on short moves and would rank legs wrongly — the argument is
#: ``ompl_planner._path_excess_travel``'s, and this module defers to it rather
#: than arguing it again. Its recorded calibration on ``bt_1788022470``: legs
#: nobody objected to spend 0.05-1.20 rad beyond the straight line, W14 spends
#: 2.35 rad, and the leg an operator DID complain about spends 10.87 rad. So the
#: acceptable band ends just above the accepted legs, and the bad band sits well
#: below the complaint, with W14 landing between them — which is where the
#: existing owner says it belongs.
EXCESS_TRAVEL_GOOD_RAD = 1.5
EXCESS_TRAVEL_BAD_RAD = 4.0

#: Below this share of the declared joint velocity limit, a leg is slow for no
#: stated reason — worth naming, since it is a different repair from a long path.
SPEED_UTILISATION_LOW = 0.30

#: Interior stillness above this share of a leg's duration is called out. Seams
#: from concatenating per-operation pieces are the usual source.
DWELL_FRACTION_HIGH = 0.10

#: A leg is a smoothness outlier when its dimensionless jerk exceeds this
#: multiple of the median across the plan's other legs. Relative, because an
#: absolute jerk threshold does not transfer between arms or payloads.
JERK_OUTLIER_FACTOR = 3.0

#: Schedule bands. parallel_efficiency == 1.0 means every arm moves for the whole
#: makespan; 0.5 on a two-arm cell means one arm might as well not be there.
PARALLEL_GOOD = 0.85
PARALLEL_ACCEPTABLE = 0.70

#: Per-joint seam tolerance. Same value as ``behaviour_tree._CHAIN_TOL_RAD``;
#: kept here as the reporting threshold so this module has no import cycle back
#: into the assembler.
SEAM_TOL_RAD = 1e-3

#: A joint is "moving" above this speed. Below it, finite-difference noise.
STILL_EPS_RAD = 1e-6

#: Composite weights. Derived from the measured cost distribution on
#: ``welding_cell_2_small``: the path defect cost ~13 s and the schedule defect
#: ~11.5 s of a 58 s makespan, so those two dominate and are weighted together;
#: dwell and smoothness were second-order there. Tunable in ONE place, on
#: purpose — a weight is an opinion and opinions belong in a named constant.
WEIGHTS = {
    "path": 0.40,
    "schedule": 0.35,
    "smoothness": 0.15,
    "dwell": 0.10,
}

VERDICT_GOOD = "good"
VERDICT_ACCEPTABLE = "acceptable"
VERDICT_POOR = "poor"


# ── small helpers ────────────────────────────────────────────────────────────

def _positions(waypoint) -> Optional[list]:
    if not isinstance(waypoint, dict):
        return None
    pos = waypoint.get("positions")
    if not isinstance(pos, (list, tuple)) or not pos:
        return None
    try:
        return [float(v) for v in pos]
    except (TypeError, ValueError):
        return None


def _clean(waypoints) -> list[tuple[list, float]]:
    """``[(positions, time_s)]`` for the waypoints that are actually usable.

    Malformed entries are dropped rather than raising: a quality report on a
    partly-broken trajectory is more useful than no report, and the drop shows up
    as ``waypoints_dropped`` in the leg's metrics.
    """
    out: list[tuple[list, float]] = []
    width = None
    for wp in waypoints or []:
        pos = _positions(wp)
        if pos is None:
            continue
        if width is None:
            width = len(pos)
        elif len(pos) != width:
            continue
        try:
            t = float(wp.get("time_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        out.append((pos, t))
    return out


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _l1(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _band(value: float, good: float, acceptable: float,
          higher_is_better: bool) -> str:
    if higher_is_better:
        if value >= good:
            return VERDICT_GOOD
        return VERDICT_ACCEPTABLE if value >= acceptable else VERDICT_POOR
    if value <= good:
        return VERDICT_GOOD
    return VERDICT_ACCEPTABLE if value <= acceptable else VERDICT_POOR


# ── per-leg metrics ──────────────────────────────────────────────────────────

def leg_metrics(waypoints: list, *,
                joint_velocity_limits: Optional[Sequence[float]] = None,
                joint_acceleration_limits: Optional[Sequence[float]] = None,
                hold_duration_s: float = 0.0,
                fk: Optional[Callable[[Sequence[float]], Sequence[float]]] = None,
                ) -> dict:
    """Quality of ONE arm's motion: the numbers that describe a single leg.

    ``waypoints`` is the ``[{"positions": [...], "time_s": t}, ...]`` list the
    planner produces. ``joint_velocity_limits`` /
    ``joint_acceleration_limits`` come from the cell's ``robots.yaml``; without
    them the utilisation fields are None rather than guessed against a made-up
    limit. ``hold_duration_s`` is the deliberate dwell AT the weld, carried as a
    port rather than baked into the waypoints, so it is added to the leg's
    duration but never counted as interior dwell.

    ``fk(positions) -> (x, y, z)`` is optional. Given it, the leg also reports
    Cartesian path length and its own detour ratio in task space, which
    distinguishes "the joints wander" from "the tool wanders".
    """
    pts = _clean(waypoints)
    raw_n = len(waypoints or [])
    if len(pts) < 2:
        return {
            "waypoints": raw_n,
            "waypoints_dropped": raw_n - len(pts),
            "usable": False,
            "note": "fewer than two usable waypoints — nothing to measure",
        }

    q_first, t_first = pts[0]
    q_last, t_last = pts[-1]
    n_joints = len(q_first)
    duration = max(0.0, t_last - t_first) + max(0.0, float(hold_duration_s or 0.0))

    # ── path length, travelled vs needed ────────────────────────────────────
    # Measured by the owners in ompl_planner, so this report and the planner's own
    # candidate selection agree by construction.
    configs = [q for q, _ in pts]
    travel_l2 = path_joint_length(configs)
    excess = trajectory_excess_travel([{"positions": q} for q in configs])
    travel_l1 = 0.0
    per_joint_travel = [0.0] * n_joints
    for (qa, _), (qb, _) in zip(pts, pts[1:]):
        travel_l1 += _l1(qa, qb)
        for j in range(n_joints):
            per_joint_travel[j] += abs(qb[j] - qa[j])

    direct_l2 = _l2(q_first, q_last)
    direct_l1 = _l1(q_first, q_last)
    # A leg that returns to where it started has no meaningful detour ratio (the
    # denominator is zero and the path is not therefore infinitely bad), so it is
    # reported as None instead of a number that would dominate every average. The
    # excess in radians stays valid for such a leg, which is a second reason
    # findings are graded on it.
    detour = path_directness(configs) if direct_l2 > 1e-6 else None

    # ── derivatives ─────────────────────────────────────────────────────────
    vel: list[list[float]] = []
    dts: list[float] = []
    for (qa, ta), (qb, tb) in zip(pts, pts[1:]):
        dt = tb - ta
        if dt <= 0:
            # Non-monotonic or duplicated timestamps: no velocity is defined over
            # a zero interval. Counted, so a caller can see the trajectory is
            # malformed rather than wonder why the peaks look low.
            continue
        dts.append(dt)
        vel.append([(qb[j] - qa[j]) / dt for j in range(n_joints)])

    bad_dt = (len(pts) - 1) - len(dts)

    acc: list[list[float]] = []
    for (va, dta), (vb, dtb) in zip(zip(vel, dts), zip(vel[1:], dts[1:])):
        span = 0.5 * (dta + dtb)
        if span <= 0:
            continue
        acc.append([(vb[j] - va[j]) / span for j in range(n_joints)])

    jerk: list[list[float]] = []
    for ja, jb in zip(acc, acc[1:]):
        # Reuse the same nominal spacing; the dimensionless form below divides it
        # out again, so a uniform estimate is enough and avoids compounding the
        # noise of a second variable-width difference.
        span = _median(dts) or 1.0
        jerk.append([(jb[j] - ja[j]) / span for j in range(n_joints)])

    peak_vel = [max((abs(v[j]) for v in vel), default=0.0) for j in range(n_joints)]
    peak_acc = [max((abs(a[j]) for a in acc), default=0.0) for j in range(n_joints)]

    def _utilisation(peaks, limits):
        if not limits or len(limits) < n_joints:
            return None, None
        ratios = []
        for j in range(n_joints):
            try:
                lim = abs(float(limits[j]))
            except (TypeError, ValueError):
                continue
            if lim > 1e-9:
                ratios.append(peaks[j] / lim)
        if not ratios:
            return None, None
        return max(ratios), sum(ratios) / len(ratios)

    vel_peak_util, vel_mean_util = _utilisation(peak_vel, joint_velocity_limits)
    acc_peak_util, acc_mean_util = _utilisation(peak_acc, joint_acceleration_limits)

    # ── dimensionless jerk ──────────────────────────────────────────────────
    # sqrt( T^5 / L^2 * integral ||jerk||^2 dt ), the scale-invariant form: a leg
    # twice as long and twice as slow scores the same, which is what makes legs
    # comparable to each other at all.
    jerk_sq_integral = 0.0
    span = _median(dts) or 0.0
    for jv in jerk:
        jerk_sq_integral += sum(v * v for v in jv) * span
    if duration > 1e-9 and travel_l2 > 1e-9 and jerk_sq_integral > 0:
        dimensionless_jerk = math.sqrt(
            jerk_sq_integral * (duration ** 5) / (travel_l2 ** 2))
    else:
        dimensionless_jerk = 0.0

    # ── stillness ───────────────────────────────────────────────────────────
    # Leading and trailing stillness is a start/stop ramp or the weld hold; only
    # the stalls BETWEEN motions are unambiguous waste, so they are separated.
    still_flags = [max(abs(v[j]) for j in range(n_joints)) < STILL_EPS_RAD
                   for v in vel]
    total_dwell = sum(dt for dt, still in zip(dts, still_flags) if still)
    lead = 0
    while lead < len(still_flags) and still_flags[lead]:
        lead += 1
    trail = len(still_flags)
    while trail > lead and still_flags[trail - 1]:
        trail -= 1
    interior_dwell = sum(dt for dt, still in zip(dts[lead:trail],
                                                 still_flags[lead:trail]) if still)
    dwell_segments = 0
    prev = False
    for still in still_flags[lead:trail]:
        if still and not prev:
            dwell_segments += 1
        prev = still

    # ── reversals ───────────────────────────────────────────────────────────
    reversals = 0
    for j in range(n_joints):
        sign = 0
        for v in vel:
            s = 0 if abs(v[j]) < STILL_EPS_RAD else (1 if v[j] > 0 else -1)
            if s and sign and s != sign:
                reversals += 1
            if s:
                sign = s

    out = {
        "usable": True,
        "waypoints": raw_n,
        "waypoints_dropped": raw_n - len(pts),
        "non_monotonic_steps": bad_dt,
        "duration_s": round(duration, 4),
        "hold_duration_s": round(float(hold_duration_s or 0.0), 4),

        "joint_travel_l2_rad": round(travel_l2, 4),
        "joint_travel_l1_rad": round(travel_l1, 4),
        "direct_distance_l2_rad": round(direct_l2, 4),
        "direct_distance_l1_rad": round(direct_l1, 4),
        "detour_ratio": None if detour is None else round(detour, 4),
        "excess_travel_rad": round(excess, 4),
        # The verdict follows the radians, not the ratio: see EXCESS_TRAVEL_*.
        "detour_verdict": _band(excess, EXCESS_TRAVEL_GOOD_RAD,
                                EXCESS_TRAVEL_BAD_RAD, higher_is_better=False),
        "waypoints_per_rad": (round(len(pts) / travel_l2, 2)
                              if travel_l2 > 1e-6 else None),
        "per_joint_travel_rad": [round(v, 4) for v in per_joint_travel],

        "peak_velocity_rad_s": [round(v, 4) for v in peak_vel],
        "peak_acceleration_rad_s2": [round(v, 4) for v in peak_acc],
        "speed_utilisation_peak": (None if vel_peak_util is None
                                   else round(vel_peak_util, 4)),
        "speed_utilisation_mean": (None if vel_mean_util is None
                                   else round(vel_mean_util, 4)),
        "accel_utilisation_peak": (None if acc_peak_util is None
                                   else round(acc_peak_util, 4)),
        "accel_utilisation_mean": (None if acc_mean_util is None
                                   else round(acc_mean_util, 4)),

        "dimensionless_jerk": round(dimensionless_jerk, 4),
        "direction_reversals": reversals,

        "dwell_total_s": round(total_dwell, 4),
        "interior_dwell_s": round(interior_dwell, 4),
        "interior_dwell_segments": dwell_segments,
        "interior_dwell_fraction": (round(interior_dwell / duration, 4)
                                    if duration > 1e-9 else 0.0),
    }

    if fk is not None:
        try:
            tcp = [tuple(float(v) for v in fk(q)[:3]) for q, _ in pts]
        except Exception:  # noqa: BLE001 - FK is a caller-supplied callback
            tcp = []
        if len(tcp) >= 2:
            cart = sum(_l2(a, b) for a, b in zip(tcp, tcp[1:]))
            straight = _l2(tcp[0], tcp[-1])
            out["cartesian_path_m"] = round(cart, 4)
            out["cartesian_direct_m"] = round(straight, 4)
            out["cartesian_detour_ratio"] = (round(cart / straight, 4)
                                             if straight > 1e-6 else None)
            out["mean_tcp_speed_m_s"] = (round(cart / duration, 4)
                                         if duration > 1e-9 else None)
    return out


# ── schedule metrics ─────────────────────────────────────────────────────────

def _leaf_duration(node: dict, blackboard: Optional[dict]) -> float:
    traj = trajectory_of(node, blackboard)
    pts = _clean(traj)
    base = (pts[-1][1] - pts[0][1]) if len(pts) >= 2 else 0.0
    try:
        hold = float(bt_port(node, "hold_duration_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        hold = 0.0
    return max(0.0, base) + max(0.0, hold)


def _timeline(node: dict, start: float, blackboard: Optional[dict],
              out: list) -> float:
    """Lay ``node`` out on an absolute timeline; return the time it finishes.

    Appends ``(robot_ns, t_start, t_end, node)`` for every motion leaf, which is
    what makes concurrency measurable rather than inferred from step counts. An
    unrecognised control node is treated as a ``Sequence`` — the pessimistic
    reading, so an unknown node can never make a plan look faster than it is.
    """
    kind = node_kind(node)
    kids = bt_children(node)

    if kind == "Parallel":
        end = start
        for child in kids:
            end = max(end, _timeline(child, start, blackboard, out))
        return end

    if kind in DECORATOR_NODES:
        offset = 0.0
        if kind == "Delay":
            try:
                offset = float(bt_port(node, "delay_msec", 0) or 0) / 1000.0
            except (TypeError, ValueError):
                offset = 0.0
        cursor = start + offset
        for child in kids:
            cursor = _timeline(child, cursor, blackboard, out)
        return cursor

    if kids or kind in CONTROL_NODES:
        cursor = start
        for child in kids:
            cursor = _timeline(child, cursor, blackboard, out)
        return cursor

    # A motion leaf. WaitForRobot / CheckSafety have no duration of their own and
    # fall out here with 0.0, which is correct: their cost is the gate they
    # impose on a sibling, already expressed by the tree's shape.
    dur = _leaf_duration(node, blackboard)
    if dur > 0.0:
        out.append((bt_port(node, "robot_ns"), start, start + dur, node))
    return start + dur


def _concurrency(intervals: list) -> tuple[float, float]:
    """``(time_with_2+_arms_moving, time_with_1_arm_moving)`` by event sweep.

    Counting DISTINCT arms, not intervals: two legs of the same arm overlapping is
    a scheduling bug, not parallelism, and must not read as concurrency.
    """
    events: list[tuple[float, int, object]] = []
    for arm, t0, t1, _node in intervals:
        if t1 > t0:
            events.append((t0, 1, arm))
            events.append((t1, -1, arm))
    if not events:
        return 0.0, 0.0
    events.sort(key=lambda e: e[0])
    active: dict = {}
    multi = single = 0.0
    prev = events[0][0]
    i = 0
    while i < len(events):
        now = events[i][0]
        n_arms = len([a for a, c in active.items() if c > 0])
        span = now - prev
        if span > 0:
            if n_arms >= 2:
                multi += span
            elif n_arms == 1:
                single += span
        while i < len(events) and events[i][0] == now:
            _t, delta, arm = events[i]
            active[arm] = active.get(arm, 0) + delta
            i += 1
        prev = now
    return multi, single


def schedule_metrics(tree: dict) -> dict:
    """How well the plan uses the arms it has — the multi-arm efficiency answer.

    Built from an absolute timeline, so ``Parallel`` steps overlap and ``Sequence``
    steps do not, and "one arm moving" is measured in seconds rather than counted
    in steps. A step count says 6 of 9 steps are paired; the timeline says 59.5 %
    of the clock has one arm idle. Only the second is a cycle-time statement.
    """
    root = tree.get("root") if isinstance(tree, dict) else None
    blackboard = tree.get("blackboard") if isinstance(tree, dict) else None
    if not isinstance(root, dict):
        return {"available": False, "note": "tree has no root node"}

    intervals: list = []
    makespan = _timeline(root, 0.0, blackboard, intervals)
    if not intervals:
        return {"available": False, "note": "no motion leaves with a duration"}

    busy: dict[str, float] = {}
    ops: dict[str, int] = {}
    for arm, t0, t1, _node in intervals:
        key = str(arm)
        busy[key] = busy.get(key, 0.0) + (t1 - t0)
        ops[key] = ops.get(key, 0) + 1

    n_arms = len(busy)
    multi_s, single_s = _concurrency(intervals)
    total_busy = sum(busy.values())

    parallel_efficiency = (total_busy / (n_arms * makespan)
                           if n_arms and makespan > 1e-9 else 0.0)
    # Only meaningful with more than one arm: on a single-arm cell "serial" is
    # not a defect, it is the cell, so it is reported as None rather than 1.0.
    serial_fraction = ((1.0 - multi_s / makespan)
                       if n_arms > 1 and makespan > 1e-9 else None)
    load_balance = (min(busy.values()) / max(busy.values())
                    if n_arms > 1 and max(busy.values()) > 1e-9 else None)

    # Operations that had no partner running alongside them: the concrete,
    # nameable form of "the plan degenerated to a single arm".
    unpaired = []
    for arm, t0, t1, node in intervals:
        overlapped = any(other_arm != arm and o0 < t1 and o1 > t0
                         for other_arm, o0, o1, _o in intervals)
        if not overlapped:
            unpaired.append({
                "robot_ns": arm,
                "operation_id": bt_port(node, "operation_id"),
                "node": node.get("name") or node_kind(node),
                "duration_s": round(t1 - t0, 4),
                "start_s": round(t0, 4),
            })

    return {
        "available": True,
        "makespan_s": round(makespan, 4),
        "arms": sorted(busy),
        "motion_legs": len(intervals),
        "operations_per_arm": ops,
        "busy_s": {a: round(v, 4) for a, v in busy.items()},
        "idle_s": {a: round(makespan - v, 4) for a, v in busy.items()},
        "utilisation": {a: round(v / makespan, 4) for a, v in busy.items()}
                       if makespan > 1e-9 else {},
        "concurrent_s": round(multi_s, 4),
        "single_arm_s": round(single_s, 4),
        "parallel_efficiency": round(parallel_efficiency, 4),
        "parallel_verdict": _band(parallel_efficiency, PARALLEL_GOOD,
                                  PARALLEL_ACCEPTABLE, higher_is_better=True),
        "serial_fraction": None if serial_fraction is None else round(serial_fraction, 4),
        "load_balance": None if load_balance is None else round(load_balance, 4),
        "unpaired_legs": unpaired,
        "unpaired_time_s": round(sum(u["duration_s"] for u in unpaired), 4),
    }


# ── continuity ───────────────────────────────────────────────────────────────

def continuity_metrics(tree: dict) -> dict:
    """Seam gaps between consecutive legs of the SAME arm, in execution order.

    A gap means the arm is asked to start a leg somewhere other than where the
    previous leg left it. The executor absorbs it as an unplanned jump, so it is
    both a motion nobody validated and time nobody costed.
    """
    root = tree.get("root") if isinstance(tree, dict) else None
    blackboard = tree.get("blackboard") if isinstance(tree, dict) else None
    if not isinstance(root, dict):
        return {"available": False}

    intervals: list = []
    _timeline(root, 0.0, blackboard, intervals)
    by_arm: dict[str, list] = {}
    for arm, t0, _t1, node in sorted(intervals, key=lambda i: i[1]):
        by_arm.setdefault(str(arm), []).append(node)

    breaks: list[dict] = []
    max_gap = 0.0
    for arm, nodes in by_arm.items():
        prev_end = None
        prev_id = None
        for node in nodes:
            pts = _clean(trajectory_of(node, blackboard))
            if not pts:
                continue
            if prev_end is not None:
                gap = max(abs(a - b) for a, b in zip(prev_end, pts[0][0])) \
                    if len(prev_end) == len(pts[0][0]) else None
                if gap is not None:
                    max_gap = max(max_gap, gap)
                    if gap > SEAM_TOL_RAD:
                        breaks.append({
                            "robot_ns": arm,
                            "from_operation": prev_id,
                            "to_operation": bt_port(node, "operation_id"),
                            "gap_rad": round(gap, 6),
                        })
            prev_end = pts[-1][0]
            prev_id = bt_port(node, "operation_id")

    return {
        "available": True,
        "max_seam_gap_rad": round(max_gap, 6),
        "seam_tolerance_rad": SEAM_TOL_RAD,
        "breaks": breaks,
        "continuous": not breaks,
    }


# ── the whole-plan report ────────────────────────────────────────────────────

def _robot_limits(robots: Optional[dict]) -> dict:
    """``{robot_ns: (velocity_limits, acceleration_limits)}`` from robots.yaml.

    Accepts either the namespaced shape (``robots: {arms: {robot1: {...}}}``) or a
    flat ``{robot1: {...}}``, because callers hold it at both depths and a second
    unwrapping rule at each call site is how the two drift apart.
    """
    out: dict[str, tuple] = {}
    if not isinstance(robots, dict):
        return out
    node = robots.get("robots", robots)
    stack = [node]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        for key, val in cur.items():
            if not isinstance(val, dict):
                continue
            if "joint_velocity_limits" in val or "base_position" in val:
                out[str(key)] = (val.get("joint_velocity_limits"),
                                 val.get("joint_acceleration_limits"))
            else:
                stack.append(val)
    return out


def score_plan(plan: dict, robots: Optional[dict] = None,
               fk: Optional[Callable] = None) -> dict:
    """The full quality report for one calculated plan.

    ``plan`` is either the stored plan envelope (``{"tree_id", "behaviour_tree",
    ...}``) or a bare behaviour tree — both are handed around by different
    callers, and refusing one of them just moves an unwrapping branch into every
    caller. ``robots`` is the cell's ``robots.yaml`` (any nesting); without it the
    velocity/acceleration utilisation fields are None rather than invented.

    Returns a dict safe to ``json.dump`` — every value is a plain scalar, list or
    dict, so the report and the log file are the same object.
    """
    tree = plan.get("behaviour_tree") if isinstance(plan, dict) else None
    if not isinstance(tree, dict):
        tree = plan if isinstance(plan, dict) else {}
    blackboard = tree.get("blackboard") or {}
    limits = _robot_limits(robots)

    intervals: list = []
    _timeline(tree.get("root") or {}, 0.0, blackboard, intervals)

    legs: list[dict] = []
    for arm, t0, _t1, node in sorted(intervals, key=lambda i: i[1]):
        vlim, alim = limits.get(str(arm), (None, None))
        try:
            hold = float(bt_port(node, "hold_duration_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            hold = 0.0
        m = leg_metrics(trajectory_of(node, blackboard),
                        joint_velocity_limits=vlim,
                        joint_acceleration_limits=alim,
                        hold_duration_s=hold, fk=fk)
        m["robot_ns"] = arm
        m["operation_id"] = bt_port(node, "operation_id")
        m["node"] = node.get("name") or node_kind(node)
        m["node_kind"] = node_kind(node)
        m["start_s"] = round(t0, 4)
        legs.append(m)

    schedule = schedule_metrics(tree)
    continuity = continuity_metrics(tree)
    usable = [l for l in legs if l.get("usable")]

    detours = [l["detour_ratio"] for l in usable if l.get("detour_ratio")]
    jerks = [l["dimensionless_jerk"] for l in usable if l.get("dimensionless_jerk")]
    dwell_total = sum(l.get("interior_dwell_s", 0.0) for l in usable)
    makespan = schedule.get("makespan_s") or 0.0

    # ── scores. Each is 0..1, higher is better, and each is an average of things
    # already reported above — no score is a new measurement, only a rollup, so a
    # reader can always see which leg moved it.
    # Aggregate, not a mean of per-leg ratios: total straight-line distance over
    # total distance travelled. That weights every leg by how much motion it
    # actually contributes, so a short 3x transit cannot outweigh the long legs
    # that set the cycle time — the same reason findings are graded on radians.
    total_travel = sum(l.get("joint_travel_l2_rad", 0.0) for l in usable)
    total_direct = sum(l.get("direct_distance_l2_rad", 0.0) for l in usable)
    path_score = min(1.0, total_direct / total_travel) if total_travel > 1e-9 else 1.0
    schedule_score = float(schedule.get("parallel_efficiency") or 0.0) \
        if schedule.get("available") else 1.0
    # Smoothness is graded RELATIVELY (see module docstring): the median leg is
    # the reference, so this rewards consistency and punishes the outlier rather
    # than asserting an absolute jerk budget.
    med_jerk = _median(jerks)
    if med_jerk > 1e-9:
        ratios = [min(1.0, med_jerk / j) for j in jerks if j > 1e-9]
        smoothness_score = sum(ratios) / len(ratios) if ratios else 1.0
    else:
        smoothness_score = 1.0
    dwell_score = max(0.0, 1.0 - (dwell_total / makespan)) if makespan > 1e-9 else 1.0

    components = {
        "path": round(path_score, 4),
        "schedule": round(schedule_score, 4),
        "smoothness": round(smoothness_score, 4),
        "dwell": round(dwell_score, 4),
    }
    score = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)

    findings = _findings(legs, schedule, continuity, jerks)
    if legs and not usable:
        # Every per-leg score above defaults to 1.0 when there is nothing to
        # measure, so a tree with no waypoints at all scored "acceptable"
        # (bt_1790951179: 0.85, 19 of 19 legs empty). No motion measured is not
        # good motion: the plan scores 0 and says why, first.
        score = 0.0
        findings.insert(0, {
            "severity": "high", "kind": "no_measurable_motion",
            "value": len(legs),
            "detail": (f"none of the {len(legs)} motion legs carries a usable "
                       f"trajectory — the tree cannot move an arm; re-plan it"),
        })
    return {
        "tree_id": (plan.get("tree_id") if isinstance(plan, dict) else None)
                   or tree.get("tree_id"),
        "cell_id": (plan.get("cell_id") if isinstance(plan, dict) else None)
                   or tree.get("cell_id"),
        "efficiency_score": round(score, 4),
        "verdict": (VERDICT_GOOD if score >= 0.85
                    else VERDICT_ACCEPTABLE if score >= 0.70 else VERDICT_POOR),
        "score_components": components,
        "score_weights": dict(WEIGHTS),
        "summary": {
            "makespan_s": makespan,
            "motion_legs": len(legs),
            "unusable_legs": len(legs) - len(usable),
            "joint_travel_rad": round(total_travel, 4),
            "direct_distance_rad": round(total_direct, 4),
            # The headline path number. Everything above the direct distance is
            # motion the cell paid for and did not need.
            "total_excess_travel_rad": round(total_travel - total_direct, 4),
            "worst_excess_travel_rad": max(
                (l.get("excess_travel_rad") or 0.0 for l in usable), default=None),
            "worst_detour_ratio": max(detours) if detours else None,
            "mean_detour_ratio": (round(sum(detours) / len(detours), 4)
                                  if detours else None),
            "total_interior_dwell_s": round(dwell_total, 4),
            "parallel_efficiency": schedule.get("parallel_efficiency"),
            "serial_fraction": schedule.get("serial_fraction"),
            "load_balance": schedule.get("load_balance"),
        },
        "schedule": schedule,
        "continuity": continuity,
        "legs": legs,
        # Report-only, and deliberately outside every score: see the module
        # docstring. Clearances are set by measurement, not by optimising a KPI.
        "safety": {
            "note": ("clearances are recorded, never scored — a score that "
                     "rewarded a tighter pass would turn a safety margin into a "
                     "tunable"),
            "achieved_clearance_m": None,
        },
        "findings": findings,
    }


def _findings(legs: list, schedule: dict, continuity: dict,
              jerks: list) -> list[dict]:
    """The actionable list: what is wrong, where, and by how much.

    Ordered worst-first. A report nobody reads is a report with a hundred numbers
    and no ranking, so the numbers above exist to produce THIS.
    """
    out: list[dict] = []

    for leg in legs:
        if not leg.get("usable"):
            out.append({
                "severity": "high", "kind": "unusable_leg",
                "robot_ns": leg.get("robot_ns"), "node": leg.get("node"),
                "detail": leg.get("note", "leg could not be measured"),
            })
            continue
        # Graded on excess RADIANS, not the ratio — the ratio is reported in the
        # detail because it is what a reader recognises, but ranking by it puts a
        # short 3.4x transit above a home return that wanders 10 rad.
        ex = leg.get("excess_travel_rad") or 0.0
        d = leg.get("detour_ratio")
        ratio_txt = f" ({d:.2f}x)" if d else ""
        if ex > EXCESS_TRAVEL_BAD_RAD:
            out.append({
                "severity": "high", "kind": "path_detour",
                "robot_ns": leg.get("robot_ns"), "node": leg.get("node"),
                "metric": "excess_travel_rad", "value": ex,
                "threshold": EXCESS_TRAVEL_BAD_RAD,
                "detail": (f"travels {leg['joint_travel_l2_rad']} rad to cover "
                           f"{leg['direct_distance_l2_rad']} rad{ratio_txt} over "
                           f"{leg['waypoints']} waypoints — {ex:.2f} rad of it is "
                           f"beyond the straight line, so shortcutting did not "
                           f"converge on this leg"),
            })
        elif ex > EXCESS_TRAVEL_GOOD_RAD:
            out.append({
                "severity": "medium", "kind": "path_detour",
                "robot_ns": leg.get("robot_ns"), "node": leg.get("node"),
                "metric": "excess_travel_rad", "value": ex,
                "threshold": EXCESS_TRAVEL_GOOD_RAD,
                "detail": (f"{ex:.2f} rad beyond the straight line{ratio_txt}"),
            })
        df = leg.get("interior_dwell_fraction") or 0.0
        if df > DWELL_FRACTION_HIGH:
            out.append({
                "severity": "medium", "kind": "interior_dwell",
                "robot_ns": leg.get("robot_ns"), "node": leg.get("node"),
                "metric": "interior_dwell_fraction", "value": df,
                "threshold": DWELL_FRACTION_HIGH,
                "detail": (f"stationary {leg['interior_dwell_s']} s mid-leg across "
                           f"{leg['interior_dwell_segments']} stall(s)"),
            })

    # ── speed: one finding for the plan, not one per leg ────────────────────
    # Measured on bt_1788022470, EVERY leg peaks at 14-21 % of the declared joint
    # velocity limits. A signal that fires on all seventeen legs is not seventeen
    # defects — it is one property of how the plan was time-parameterised, and
    # printing it seventeen times buries the two findings that name a single leg.
    speeds = [(l, l.get("speed_utilisation_peak")) for l in legs
              if l.get("speed_utilisation_peak") is not None]
    slow = [(l, su) for l, su in speeds if su < SPEED_UTILISATION_LOW]
    if speeds and len(slow) >= max(2, len(speeds) // 2):
        med_su = _median([su for _l, su in speeds])
        out.append({
            "severity": "medium", "kind": "slow_plan",
            "metric": "speed_utilisation_peak", "value": round(med_su, 4),
            "threshold": SPEED_UTILISATION_LOW,
            "detail": (f"{len(slow)} of {len(speeds)} legs peak below "
                       f"{SPEED_UTILISATION_LOW:.0%} of the declared joint "
                       f"velocity limits (median {med_su:.0%}) — the whole plan is "
                       f"time-scaled well under the arms' rated speed, which is a "
                       f"different repair from a long path"),
        })
    else:
        for leg, su in slow:
            out.append({
                "severity": "low", "kind": "slow_leg",
                "robot_ns": leg.get("robot_ns"), "node": leg.get("node"),
                "metric": "speed_utilisation_peak", "value": su,
                "threshold": SPEED_UTILISATION_LOW,
                "detail": (f"peaks at {su:.0%} of the declared joint velocity "
                           f"limit while its peers do not — slow, not long"),
            })

    med = _median(jerks)
    if med > 1e-9:
        for leg in legs:
            j = leg.get("dimensionless_jerk") or 0.0
            if j > JERK_OUTLIER_FACTOR * med:
                out.append({
                    "severity": "medium", "kind": "smoothness_outlier",
                    "robot_ns": leg.get("robot_ns"), "node": leg.get("node"),
                    "metric": "dimensionless_jerk", "value": j,
                    "threshold": round(JERK_OUTLIER_FACTOR * med, 4),
                    "detail": (f"{j / med:.1f}x the median leg in this plan — "
                               f"graded against its peers, not an absolute budget"),
                })

    if schedule.get("available"):
        pe = schedule.get("parallel_efficiency") or 0.0
        if pe < PARALLEL_ACCEPTABLE:
            worst = min(schedule.get("utilisation", {}).items(),
                        key=lambda kv: kv[1], default=(None, None))
            out.append({
                "severity": "high" if pe < 0.5 else "medium",
                "kind": "low_parallelism",
                "metric": "parallel_efficiency", "value": pe,
                "threshold": PARALLEL_ACCEPTABLE,
                "detail": (f"{schedule.get('single_arm_s')} s of the "
                           f"{schedule.get('makespan_s')} s makespan runs one arm; "
                           f"least-used arm {worst[0]} at "
                           f"{(worst[1] or 0):.0%} utilisation"),
            })
        if schedule.get("unpaired_legs"):
            out.append({
                "severity": "medium", "kind": "unpaired_operations",
                "value": len(schedule["unpaired_legs"]),
                "detail": (f"{len(schedule['unpaired_legs'])} leg(s) totalling "
                           f"{schedule.get('unpaired_time_s')} s run with no other "
                           f"arm moving: "
                           + ", ".join(f"{u['node']}" for u in
                                       schedule["unpaired_legs"][:6])),
            })

    if continuity.get("available") and continuity.get("breaks"):
        out.append({
            "severity": "high", "kind": "seam_discontinuity",
            "metric": "max_seam_gap_rad",
            "value": continuity.get("max_seam_gap_rad"),
            "threshold": SEAM_TOL_RAD,
            "detail": (f"{len(continuity['breaks'])} seam(s) where an arm starts a "
                       f"leg away from where the previous one ended"),
        })

    rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda f: (rank.get(f.get("severity"), 3),
                            -float(f.get("value") or 0)))
    return out


# ── the log file ─────────────────────────────────────────────────────────────

def write_quality_log(cell_id: str, plan: dict,
                      cell_base: Optional[str] = None,
                      robots: Optional[dict] = None,
                      fk: Optional[Callable] = None):
    """Score ``plan`` and write the report to the cell's quality log.

    The log is named for the tree it describes — ``logs/<tree_id>.json``, the same
    basename the plan and the recording already use — so a reader who has a
    ``tree_id`` can find all three without a lookup table. The path comes from
    ``cell_layout.quality_log_file``, which owns it.

    Written atomically through :mod:`utils.cell_store`, so in the planner's
    container the report lands via the orchestrator rather than on a shared mount.
    Returns the ``Path`` the report occupies in the owning cell tree, or None on
    any failure — like plan persistence, a quality report must never be the reason
    a plan is lost.
    """
    import json

    from utils import cell_layout, cell_store

    try:
        tree_id = ((plan.get("tree_id") if isinstance(plan, dict) else None)
                   or (plan.get("behaviour_tree") or {}).get("tree_id"))
        if not tree_id:
            return None
        if robots is None:
            robots = _load_robots(cell_id, cell_base)
        report = score_plan(plan, robots=robots, fk=fk)
        report["cell_id"] = report.get("cell_id") or cell_id
        path = cell_layout.quality_log_file(cell_id, tree_id, cell_base)
        cell_store.write_text(cell_id, cell_store.KIND_LOGS, path.name,
                              json.dumps(report, indent=2),
                              cell_base=cell_base)
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"[trajectory_quality] WARNING: could not write quality log: {exc}",
              flush=True)
        return None


def report_for(cell_id: str, tree_id: Optional[str] = None,
               cell_base: Optional[str] = None) -> dict:
    """The report for one STORED plan, and where the numbers came from.

    The read side of :func:`write_quality_log`, and the whole of what the MCP tool
    and the dashboard's Performance card serve. Here rather than in
    ``mcp_server.py`` because it is the policy — which plan, stored report or
    fresh score, and what a missing one means — and that file cannot be imported
    without the LLM stack, so the policy would otherwise be untestable.

    ``tree_id`` omitted or ``"latest"`` resolves to the newest SUCCESSFUL plan. A
    failed run is never scored: its legs are a fragment, and scoring a fragment
    against a whole program flatters a run for stopping early.

    Returns ``{"success": True, "tree_id", "source": "stored"|"computed", ...}``
    with the report merged in, or ``{"success": False, "error"}`` when there is no
    such plan. ``source`` is ``"stored"`` for the report ``save_plan`` already
    wrote, ``"computed"`` for a plan older than these metrics, scored now.

    Lets :class:`utils.cell_store.CellStoreUnavailable` propagate. "The owner is
    unreachable" is not "this plan has no findings", and the caller has to be able
    to tell those apart.
    """
    from agents.trajectory.tools import plan_store
    from utils import cell_store

    wanted = tree_id if tree_id and tree_id != "latest" else None
    if wanted is None:
        usable = [p for p in plan_store.list_plans(cell_id)
                  if p.get("status") != plan_store.STATUS_FAILED]
        if not usable:
            # list_plans swallows every failure and answers [], so "this cell has
            # never been planned" and "the owner is unreachable" reach here as the
            # same empty list. Re-ask in a way that raises, before calling a cell
            # unplanned on the strength of a failed round trip.
            cell_store.list_entries(cell_id, cell_store.KIND_PLANS, suffix=".json")
            return {"success": False, "cell_id": cell_id,
                    "error": f"no calculated plans stored for '{cell_id}' yet"}
        wanted = usable[0].get("tree_id")

    # The stored report first: save_plan already paid for it, and reading one
    # artefact beats fetching a whole tree with its waypoints.
    report = cell_store.read_json(cell_id, cell_store.KIND_LOGS, f"{wanted}.json",
                                  cell_base=cell_base)
    source = "stored"
    if not report or "efficiency_score" not in report:
        plan = plan_store.get_plan(cell_id, wanted)
        if not plan:
            return {"success": False, "cell_id": cell_id, "tree_id": wanted,
                    "error": (f"no plan '{wanted}' to score — a FAILED run is not "
                              "scored, and 'latest' never resolves to one")}
        report = score_plan(plan, robots=_load_robots(cell_id, cell_base))
        source = "computed"
    return {"success": True, "cell_id": cell_id, "tree_id": wanted,
            "source": source, **report}


def _load_robots(cell_id: str, cell_base: Optional[str] = None) -> Optional[dict]:
    """The cell's ``robots.yaml``, or None when it cannot be read.

    Only the joint limits are wanted, and a missing file must degrade to "no
    utilisation figures" rather than failing the whole report — the path metrics
    are the ones that have caught real defects and they need no limits at all.
    """
    try:
        from utils import cell_layout, cell_store
        return cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG,
                                    cell_layout.robots_yaml(cell_id).name,
                                    cell_base=cell_base) or None
    except Exception:  # noqa: BLE001
        return None
