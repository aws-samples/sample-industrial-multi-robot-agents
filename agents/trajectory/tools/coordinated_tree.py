"""
The coordinated planning mode's tools: schedule the arms' own legs in time, and
assemble the tree that runs them — each arm its own sequence, all at once,
ordered at conflicts by completion gates. See ``coordination.py`` for the model.

The planner LLM owns the GEOMETRY — the strategy and waypoints for approaching
and leaving each weld, and a safe spot for an arm that is in another's way. These
tools own the TIMING, deterministically.
"""
from __future__ import annotations

import time as _time
from typing import Optional

from langchain_core.tools import tool

from agents.trajectory.tools.coordination import DEFAULT_SETTLE_S, geometric_conflict, schedule
from agents.trajectory.tools.trajectory_cache import get_cached_trajectory, new_tree_id, weld_pose_record


#: Longest stretch of a transit that is coordinated as ONE block. Coordinating a
#: whole leg at once means it waits if ANY part of it conflicts; cut into
#: segments, an arm runs the part of its path that is clear and waits at an
#: intermediate point — near the part — only where the conflict actually is.
MAX_SEGMENT_S = 1.0
SEP = "@"


def _segments(traj: list, max_s: float = MAX_SEGMENT_S) -> list:
    """Cut a leg at its transit→approach seam and every ``max_s`` of transit.
    Adjacent segments share their boundary waypoint, so they chain exactly."""
    if len(traj) < 3:
        return [traj]
    cuts = [0]
    t_last = float(traj[0].get("time_s", 0.0))
    for i in range(1, len(traj) - 1):
        seam = traj[i].get("phase") != traj[i - 1].get("phase")
        if seam or float(traj[i].get("time_s", 0.0)) - t_last >= max_s:
            cuts.append(i)
            t_last = float(traj[i].get("time_s", 0.0))
    cuts.append(len(traj) - 1)
    segs = [traj[a:b + 1] for a, b in zip(cuts, cuts[1:]) if b > a]
    return segs or [traj]


def _retime(seg: list) -> list:
    t0 = float(seg[0].get("time_s", 0.0))
    return [{**p, "time_s": round(float(p.get("time_s", 0.0)) - t0, 4)} for p in seg]


def _op_of(seg_id: str) -> str:
    return seg_id.split(SEP, 1)[0]


def _load(cell_id: str, sequences: dict, hold_duration_s: Optional[dict]):
    """Cached legs + durations per arm, or the list of what is missing/broken."""
    from agents.trajectory.tools.behaviour_tree import _chain_breaks

    holds = hold_duration_s or {}
    trajs: dict = {}
    programs: dict = {}
    missing, breaks = [], []
    seg_ids: dict = {}
    for ns, ops in (sequences or {}).items():
        ops = [str(o) for o in (ops or [])]
        trajs[ns] = {}
        durs = {}
        for op in ops:
            tr = get_cached_trajectory(cell_id, ns, op)
            if not tr:
                missing.append({"robot_ns": ns, "operation_id": op})
                continue
            segs = _segments(tr)
            for k, seg in enumerate(segs):
                sid = f"{op}{SEP}{k}"
                seg_ids.setdefault(ns, []).append(sid)
                trajs[ns][sid] = _retime(seg)
                dur = float(seg[-1].get("time_s", 0.0)) - float(seg[0].get("time_s", 0.0))
                if k == len(segs) - 1:
                    dur += float(holds.get(op, 0.5)) + DEFAULT_SETTLE_S
                durs[sid] = dur
        programs[ns] = (seg_ids.get(ns, []), durs)
        if ops and not missing:
            _, _, brk = _chain_breaks(cell_id, ns, ops)
            breaks.extend(brk)
    return trajs, programs, missing, breaks, {ns: [str(o) for o in (ops or [])]
                                              for ns, ops in (sequences or {}).items()}


#: Suffix of the return-home leg the coordinator adds for a finished arm.
EARLY_HOME = "_HOME"


def _plan_early_home(cell_id: str, arm: str, last_op: str) -> Optional[list]:
    """A planned return to HOME from where ``arm`` finished — the same planner
    (on-axis lift-off, then a collision-free path) as the end-of-program home."""
    from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose
    from agents.trajectory.tools.planner_backend import plan_home_trajectory
    last = get_cached_trajectory(cell_id, arm, last_op)
    if not last:
        return None
    try:
        return plan_home_trajectory(cell_id, arm, list(last[-1]["positions"]),
                                    list(home_pose(cell_id, arm, HOME_JOINTS_FALLBACK)))
    except Exception:  # noqa: BLE001
        return None


def _coordinate(cell_id, sequences, hold_duration_s):
    """Coordinate; an arm that has FINISHED its work and stands in another's way
    is sent home early (planned, then coordinated like any leg) instead of
    staying there as a permanent obstacle. Only an arm that still has work needs
    a safe spot from the planner."""
    from agents.trajectory.tools.trajectory_cache import cache_trajectory
    seqs = {ns: [str(o) for o in (ops or [])] for ns, ops in (sequences or {}).items()}
    homed: list = []
    res = report = None
    for _ in range(len(seqs) + 1):
        res, report = _coordinate_once(cell_id, seqs, hold_duration_s)
        dl = (report or {}).get("deadlock")
        if not dl:
            break
        arm = dl["needs_safe_spot"]["arm"]
        after = dl["needs_safe_spot"]["after_op"]
        finished = seqs.get(arm) and seqs[arm][-1] == after
        if not finished or arm in homed:
            break
        leg = _plan_early_home(cell_id, arm, after)
        if not leg:
            report["error"] += (f" ({arm} has finished its work but no return home "
                                f"could be planned from {after}.)")
            break
        op = f"{arm}{EARLY_HOME}"
        cache_trajectory(cell_id, arm, op, leg)
        seqs[arm].append(op)
        homed.append(arm)
    if report is not None:
        report["early_home"] = homed
        report["sequences"] = seqs
    return res, report


def _coordinate_once(cell_id, sequences, hold_duration_s):
    from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose

    trajs, programs, missing, breaks, ops_by_arm = _load(cell_id, sequences, hold_duration_s)
    if missing:
        return None, {"success": False, "missing": missing,
                      "error": "run run_trajectory_planner for these operations first"}
    if breaks:
        return None, {"success": False, "breaks": breaks,
                      "error": ("an operation does not start where the arm's previous one "
                                "ended — re-plan it chained from that end, in order")}
    home = {ns: list(home_pose(cell_id, ns, HOME_JOINTS_FALLBACK)) for ns in programs}
    t0 = _time.monotonic()
    conflict = geometric_conflict(cell_id, trajs, home)
    res = schedule(programs, conflict)
    def per_op(d):
        out: dict = {}
        for sid, v in d.items():
            out.setdefault(_op_of(sid), []).append(v)
        return out

    report = {
        "success": res["deadlock"] is None,
        "makespan_estimate_s": res["makespan_s"],
        "parallel_s": res["parallel_s"],
        "coordination_time_s": round(_time.monotonic() - t0, 2),
        "arms": {ns: {"operations": ops_by_arm.get(ns, []),
                      "waits_s": {o: round(sum(w), 2) for o, w in per_op(p.waits).items()
                                  if sum(w) > 0.05},
                      "gates": {o: {b: _op_of(e) for g in gs for b, e in g.items()}
                                for o, gs in per_op(p.gates).items()}}
                 for ns, p in res["arms"].items()},
    }
    if res["deadlock"]:
        dl = dict(res["deadlock"])
        for k in ("waiting_to_start", "after_op"):
            dl[k] = _op_of(str(dl[k]))
        dl["blocked_by"] = " ".join(_op_of(x) for x in str(dl["blocked_by"]).split(" "))
        dl["needs_safe_spot"] = {"arm": dl["needs_safe_spot"]["arm"],
                                 "after_op": _op_of(dl["needs_safe_spot"]["after_op"])}
        spot = dl["needs_safe_spot"]
        report["deadlock"] = dl
        report["error"] = (
            f"{dl['arm']} cannot start {dl['waiting_to_start']}: {dl['blocked_by_arm']} "
            f"({dl['blocked_by']}) is in its way and cannot move on first. Give "
            f"{spot['arm']} a SAFE SPOT after {spot['after_op']}: a pose close to the part, "
            f"on its own side, clear of {dl['arm']}'s path — plan it as a `move` "
            f"(operation_type 'move', approach false) chained after {spot['after_op']}, "
            f"insert it into {spot['arm']}'s sequence, and coordinate again.")
    return res, report


@tool
def coordinate_program(cell_id: str, sequences: dict,
                       hold_duration_s: Optional[dict] = None) -> dict:
    """
    Coordinate the arms IN TIME — each follows its own planned legs, all at once.

    Call this after every operation of every arm has been planned with
    `run_trajectory_planner` (each arm in its own calls, chained in order).

    Args:
        cell_id:          Cell identifier
        sequences:        {robot_ns: [operation_id, ...]} — each arm's operations,
                          in execution order (welds, and any safe-spot moves).
        hold_duration_s:  optional {operation_id: seconds} dwell at each weld
                          (default 0.5).

    Returns {"success", "makespan_estimate_s", "parallel_s", "arms": {robot_ns:
    {"operations", "waits_s", "gates"}}} — where an arm waits it holds WHERE IT
    STANDS (near the part), released when the other arm COMPLETES the operation
    named in `gates`. On failure "deadlock" says which arm is in the way and after
    which operation it needs a SAFE SPOT; plan that `move`, add it to the sequence
    and call this again. Use `makespan_estimate_s` to compare alternatives
    (a different approach/leave strategy or safe spot) and keep the fastest.
    """
    try:
        _, report = _coordinate(cell_id, sequences, hold_duration_s)
        return report
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"coordination failed: {exc}"}


@tool
def assemble_coordinated_tree(cell_id: str, sequences: dict,
                              hold_duration_s: Optional[dict] = None,
                              operation_types: Optional[dict] = None,
                              home_at_end: bool = True) -> dict:
    """
    Assemble the Behaviour Tree for a coordinated program (the default planning
    mode): one Sequence per arm, all arms in a Parallel, and a WaitForRobot gate
    wherever an arm must let another arm COMPLETE an operation first. The gates
    are events, not clock times, so the separation holds at any real speed.
    Ends with the coupled, gated return HOME (as `assemble_behaviour_tree`).

    Same arguments as `coordinate_program`, plus `operation_types`
    ({operation_id: "weld"|"move"|...}, default "weld"). Refuses a program that
    deadlocks — resolve it with a safe spot first.
    """
    return assemble_coordinated(cell_id, sequences, hold_duration_s=hold_duration_s,
                                operation_types=operation_types, home_at_end=home_at_end)


def assemble_coordinated(cell_id: str, sequences: dict,
                         hold_duration_s: Optional[dict] = None,
                         operation_types: Optional[dict] = None,
                         home_at_end: bool = True,
                         tail_steps: Optional[list] = None) -> dict:
    """:func:`assemble_coordinated_tree`, plus ``tail_steps``: steps (the
    assembler's plan format) that run AFTER every arm has finished its sequence —
    the stepwise program's own returns home, when the barrier-free pass
    (``behaviour_tree._barrier_free``) coordinates the work before them. Each is
    built and gated as the stepwise assembler gates a step, against where the
    arms finished; their chaining and legs are checked like the rest."""
    from agents.trajectory.tools.behaviour_tree import (
        _approach_refusal, _home_step, _leg_audit, build_behaviour_tree_node, to_btcpp,
        validate_behaviour_tree,
    )
    from agents.trajectory.tools.trajectory_cache import get_cached_trajectory
    from agents.trajectory.tools.leg_verdict import refusal as leg_refusal
    try:
        res, report = _coordinate(cell_id, sequences, hold_duration_s)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"coordination failed: {exc}"}
    if res is None or not report.get("success"):
        return report

    holds = hold_duration_s or {}
    kinds = operation_types or {}
    build = getattr(build_behaviour_tree_node, "func", build_behaviour_tree_node)
    trajs = _load(cell_id, report.get("sequences") or sequences, hold_duration_s)[0]

    # Group each arm's segments into executor nodes: a new node where a gate is
    # needed, and at the end of every operation (its hold belongs there).
    groups: dict = {}
    node_of: dict = {}
    for ns, plan in res["arms"].items():
        gl, cur = [], []
        segs = plan.ops
        for i, sid in enumerate(segs):
            if plan.gates.get(sid) and cur:
                gl.append(cur)
                cur = []
            cur.append(sid)
            last_of_op = i + 1 == len(segs) or _op_of(segs[i + 1]) != _op_of(sid)
            if last_of_op:
                gl.append(cur)
                cur = []
        if cur:
            gl.append(cur)
        groups[ns] = gl
        for g in gl:
            op = _op_of(g[0])
            whole = len([x for x in segs if _op_of(x) == op]) == len(g)
            nid = op if whole else f"{op}{SEP}{g[0].split(SEP)[1]}-{g[-1].split(SEP)[1]}"
            for sid in g:
                node_of[(ns, sid)] = nid

    arm_seqs = []
    per_robot_ops = {}
    leg_hits: list = []
    for ns, plan in sorted(res["arms"].items()):
        nodes = []
        for g in groups[ns]:
            op = _op_of(g[0])
            nid = node_of[(ns, g[0])]
            deps = {}
            for sid in g:
                for b, ev in (plan.gates.get(sid) or {}).items():
                    deps[(b, node_of[(b, ev)])] = True
            if deps:
                nodes.append({"id": f"gate_{ns}_{nid}", "type": "WaitForRobot",
                              "robot_ns": ns,
                              "depends_on_ops": [{"robot": b, "operation_id": o}
                                                 for b, o in sorted(deps)]})
            traj, offset = [], 0.0
            for sid in g:
                seg = trajs[ns][sid]
                pts = seg if not traj else seg[1:]
                traj.extend({**p, "time_s": round(float(p["time_s"]) + offset, 4)} for p in pts)
                offset = traj[-1]["time_s"]
            ends_op = g[-1] == [x for x in plan.ops if _op_of(x) == op][-1]
            # Every leg the tree will execute gets the platform's leg check — the
            # same final audit as assemble_behaviour_tree (umarell issues 13, 14) —
            # static obstacles included: the coordinator's conflict model is
            # arm-against-arm only, so no other gate of this tree sees them.
            why = leg_refusal(cell_id, ns, traj, static=True) if traj else None
            if why is not None:
                leg_hits.append({"operation_id": nid, **why})
            nodes.append(build(
                node_type="ExecuteTrajectory", robot_ns=ns, operation_id=nid,
                operation_type=("move" if op.endswith(EARLY_HOME) else kinds.get(op, "weld")),
                trajectory=traj,
                hold_duration_s=float(holds.get(op, 0.5)) if ends_op else 0.0,
                velocity_scale=1.0))
        per_robot_ops[ns] = list(dict.fromkeys(_op_of(x) for x in plan.ops))
        # Each weld's working stroke against its specification, on the whole leg.
        for op in per_robot_ops[ns]:
            why = _approach_refusal(cell_id, ns, op, get_cached_trajectory(cell_id, ns, op))
            if why is not None:
                leg_hits.append({"operation_id": op, **why})
        arm_seqs.append(build(node_type="Sequence", children=nodes))
    children = [build(node_type="Parallel", children=arm_seqs)]
    tail_s = None
    if tail_steps:
        from agents.trajectory.tools.behaviour_tree import _arm_home, _chain_breaks, _gated_steps
        at_pose = {}
        for ns in res["arms"]:
            last = get_cached_trajectory(cell_id, ns, per_robot_ops[ns][-1]) \
                if per_robot_ops.get(ns) else None
            q = (last[-1].get("positions") if last else None) or _arm_home(cell_id, ns)
            if q:
                at_pose[ns] = list(q)
        tail_nodes, tail_reports, tail_s = _gated_steps(cell_id, tail_steps, at_pose)
        if tail_nodes is None:
            return {**report, "success": False, "tail_steps": tail_reports,
                    "error": "the returns home after the work could not be gated"}
        tail_ops: dict = {}
        for st in tail_steps:
            for r in st.get("robots") or []:
                if r.get("robot_ns") and r.get("operation_id"):
                    tail_ops.setdefault(r["robot_ns"], []).append(str(r["operation_id"]))
        breaks = []
        for ns, ops in tail_ops.items():
            breaks += _chain_breaks(cell_id, ns, per_robot_ops.get(ns, []) + ops)[2]
            for op in ops:
                tr = get_cached_trajectory(cell_id, ns, op)
                why = leg_refusal(cell_id, ns, tr, static=True) if tr else None
                if why is not None:
                    leg_hits.append({"operation_id": op, **why})
            per_robot_ops.setdefault(ns, []).extend(ops)
        if breaks:
            return {**report, "success": False, "breaks": breaks,
                    "error": "a return home does not start where its arm finished"}
        children.extend(tail_nodes)
    home_report = None
    # Arms the coordinator already sent home need no end-of-program return.
    to_home = sorted(ns for ns in per_robot_ops if ns not in report.get("early_home", []))
    if home_at_end and to_home:
        home_node, home_report = _home_step(cell_id, to_home,
                                            {ns: per_robot_ops[ns] for ns in to_home})
        if home_node is None:
            return {**report, "success": False, "home_step": home_report,
                    "error": ("the end-of-program return to HOME cannot be executed "
                              "safely from where the arms finish — give the arm in the "
                              "way a safe spot as its last operation and assemble again")}
        children.append(home_node)
        leg_hits += _leg_audit(cell_id, [], home_node, {ns: per_robot_ops[ns] for ns in to_home})
    if leg_hits:
        return {**report, "success": False, "leg_refusals": leg_hits,
                **({"home_step": home_report} if home_report is not None else {}),
                "error": ("these legs cannot be executed as planned: they leave their "
                          "arm's joint limits, or drive the arm into its OWN collision "
                          "hulls along the executed path. Re-plan each listed operation "
                          "(another arm, a different approach orientation, or a via pose)")}
    root = build(node_type="Sequence", children=children)
    tree = to_btcpp({
        "tree_id": new_tree_id(cell_id),
        "cell_id": cell_id,
        "root": root,
        "metadata": {"mode": "coordinated",
                     "operations": sum(len(o) for o in per_robot_ops.values()),
                     "makespan_estimate_s": report["makespan_estimate_s"],
                     "parallel_s": report["parallel_s"],
                     # See assemble_behaviour_tree: the home reset's lift-off.
                     "weld_poses": weld_pose_record(cell_id, list(per_robot_ops))},
    })
    validation = validate_behaviour_tree.func(tree)
    out = {**report, "success": bool(validation.get("valid", False)),
           "behaviour_tree": tree, "validation": validation}
    if home_report is not None:
        out["home_step"] = home_report
    if tail_s is not None:
        out["tail_s"] = tail_s
    return out
