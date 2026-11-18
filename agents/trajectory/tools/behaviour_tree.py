"""
Behaviour Tree tools — assemble a BehaviorTree.CPP v4 tree from planned trajectories.

Called by the trajectory agent's LLM after planning each step. The tree is
consumed by the cell agent to coordinate robot execution.

The emitted format is **BehaviorTree.CPP v4** (https://www.behaviortree.dev):

  Sequence           steps run in order                (control)
  Parallel           arms move simultaneously          (control, success_count=-1)
  Delay              stagger one arm's start           (decorator, delay_msec)
  ExecuteTrajectory  run one arm's planned motion      (Action)
  MoveHome           return one arm to its home config (Action)
  WaitForRobot       block until another arm clears    (Action)
  CheckSafety        gate on an external condition     (Condition)

``assemble_behaviour_tree`` returns that tree together with its XML
serialisation, so the same plan can be executed by the Python cell agent, opened
in Groot2, or loaded by a C++ ``BehaviorTreeFactory``.

The node builders below construct an intermediate shape and
``agents.trajectory.tools.btcpp.to_btcpp`` converts once at the end; joint arrays
are moved onto the tree's blackboard and referenced from ports as ``{key}``, the
idiomatic BT.CPP split that keeps the XML readable.
"""
from __future__ import annotations

from contextvars import ContextVar

import json
import uuid
from typing import Optional

from langchain_core.tools import tool

from agents.home_pose import HOME_JOINTS_FALLBACK
from agents.trajectory.tools.btcpp import (
    LEGACY_TO_BTCPP,
    children as bt_children,
    node_kind,
    port as bt_port,
    to_btcpp,
    to_xml,
)


def _node_id() -> str:
    return str(uuid.uuid4())[:8]


@tool
def build_behaviour_tree_node(
    node_type: str,
    robot_ns: Optional[str] = None,
    trajectory: Optional[list[dict]] = None,
    operation_id: Optional[str] = None,
    operation_type: Optional[str] = None,
    children: Optional[list[dict]] = None,
    depends_on_robots: Optional[list[str]] = None,
    depends_on_positions: Optional[list[dict]] = None,
    has_interlock: bool = False,
    interlock_target: Optional[dict] = None,
    start_delay_s: float = 0.0,
    hold_duration_s: float = 0.5,
    velocity_scale: float = 1.0,
    condition: Optional[str] = None,
) -> dict:
    """
    Build a single BehaviorTree.CPP v4 node.

    Node types and required ports:
      Parallel:          children=[node, ...]
      Sequence:          children=[node, ...]
      ExecuteTrajectory: robot_ns, trajectory, operation_id, operation_type
      WaitForRobot:      robot_ns, depends_on_robots, depends_on_positions
      CheckSafety:       condition
      MoveHome:          robot_ns

    Args:
        node_type:            Sequence | Parallel | ExecuteTrajectory |
                              WaitForRobot | CheckSafety | MoveHome. The legacy
                              names (SEQUENCE / PARALLEL / EXECUTE /
                              INTERLOCK_WAIT / SAFETY_GATE / HOME) are accepted
                              and normalised, so an older caller keeps working.
        robot_ns:             Robot namespace (ExecuteTrajectory / WaitForRobot /
                              MoveHome)
        trajectory:           [{"positions": [j1..j6], "time_s": float}]
        operation_id:         Operation point ID from operations.yaml (e.g. "W001")
        operation_type:       "weld" | "rivet" | "glue" | "laser_cut" | "mark" | "move"
        children:             Child nodes (Parallel / Sequence)
        depends_on_robots:    Robot namespaces this node waits for
        depends_on_positions: [{"robot": ns, "position": [x,y,z], "tolerance_m": 0.05}]
        has_interlock:        True if this node has a pre-movement interlock
        interlock_target:     {"ee_position": [...], "joint_positions": [...]}
        start_delay_s:        Seconds to wait before starting this motion. The
                              assembler turns a non-zero value into a wrapping
                              ``Delay`` decorator (``delay_msec``), which is how
                              BT.CPP expresses a timed gate.
        hold_duration_s:      Seconds to hold at target before completing
        velocity_scale:       Speed scaling (0.1 – 1.0)
        condition:            Safety gate condition string

    Returns:
        A BT node dict ready to be embedded in a tree. ``assemble_behaviour_tree``
        (or ``to_btcpp``) canonicalises it — moving bulky values to the tree's
        blackboard — when the whole tree is finished.
    """
    # Accept either vocabulary: a legacy name from an older caller or prompt is
    # mapped once here so everything downstream sees one spelling.
    node_type = LEGACY_TO_BTCPP.get(str(node_type), str(node_type))

    node: dict = {
        "id":        _node_id(),
        "type":      node_type,
    }

    if node_type == "ExecuteTrajectory":
        node.update({
            "robot_ns":             robot_ns,
            "operation_id":         operation_id,
            "operation_type":       operation_type,
            "precomputed_trajectory": trajectory or [],
            "has_interlock":        has_interlock,
            "interlock_target":     interlock_target,
            "depends_on_robots":    depends_on_robots or [],
            "depends_on_positions": depends_on_positions or [],
            "start_delay_s":        start_delay_s,
            "hold_duration_s":      hold_duration_s,
            "velocity_scale":       velocity_scale,
        })

    elif node_type in ("Parallel", "Sequence"):
        node["children"] = children or []

    elif node_type == "WaitForRobot":
        node.update({
            "robot_ns":             robot_ns,
            "depends_on_robots":    depends_on_robots or [],
            "depends_on_positions": depends_on_positions or [],
        })

    elif node_type == "CheckSafety":
        node["condition"] = condition or "estop_clear"

    elif node_type == "MoveHome":
        node["robot_ns"] = robot_ns

    return node


@tool
def emit_parallel_node(
    robot1_ns: str,
    trajectory1: list[dict],
    operation_id1: str,
    operation_type1: str,
    robot2_ns: str,
    trajectory2: list[dict],
    operation_id2: str,
    operation_type2: str,
    start_delay_r2_s: float = 0.0,
    has_interlock_r1: bool = False,
    interlock_target_r1: Optional[dict] = None,
    has_interlock_r2: bool = False,
    interlock_target_r2: Optional[dict] = None,
    hold_duration_s: float = 0.5,
    velocity_scale: float = 1.0,
) -> dict:
    """
    Convenience: emit a Parallel node with two ExecuteTrajectory children.

    Shortcut for the common case of two robots working simultaneously.
    Use build_behaviour_tree_node for more complex structures.

    Args:
        robot1_ns:            First robot namespace
        trajectory1:          First robot trajectory
        operation_id1:        First operation ID
        operation_type1:      First operation type
        robot2_ns:            Second robot namespace
        trajectory2:          Second robot trajectory
        operation_id2:        Second operation ID
        operation_type2:      Second operation type
        start_delay_r2_s:     Stagger delay for robot2 start (0 = simultaneous)
        has_interlock_r1/r2:  Whether each robot has an interlock waypoint
        interlock_target_r1/r2: Interlock target for each robot
        hold_duration_s:      Hold time at target
        velocity_scale:       Speed scaling

    Returns:
        A Parallel node with two ExecuteTrajectory children.
    """
    child1 = build_behaviour_tree_node.invoke({
        "node_type":       "ExecuteTrajectory",
        "robot_ns":        robot1_ns,
        "trajectory":      trajectory1,
        "operation_id":    operation_id1,
        "operation_type":  operation_type1,
        "has_interlock":   has_interlock_r1,
        "interlock_target": interlock_target_r1,
        "hold_duration_s": hold_duration_s,
        "velocity_scale":  velocity_scale,
    })
    child2 = build_behaviour_tree_node.invoke({
        "node_type":       "ExecuteTrajectory",
        "robot_ns":        robot2_ns,
        "trajectory":      trajectory2,
        "operation_id":    operation_id2,
        "operation_type":  operation_type2,
        "has_interlock":   has_interlock_r2,
        "interlock_target": interlock_target_r2,
        "start_delay_s":   start_delay_r2_s,
        "hold_duration_s": hold_duration_s,
        "velocity_scale":  velocity_scale,
    })
    return build_behaviour_tree_node.invoke({
        "node_type": "Parallel",
        "children":  [child1, child2],
    })


#: Node kinds this repo's executor implements. Anything else in a tree is an
#: error here rather than an "unregistered node" surprise at execution time.
_KNOWN_KINDS = frozenset({
    "Sequence", "SequenceWithMemory", "Parallel", "Delay",
    "ExecuteTrajectory", "MoveHome", "WaitForRobot", "CheckSafety",
})


@tool
def validate_behaviour_tree(tree: dict) -> dict:
    """
    Validate a BehaviorTree.CPP tree for structural correctness.

    Checks:
      - Every node is one the executor implements (Sequence, Parallel, Delay,
        ExecuteTrajectory, MoveHome, WaitForRobot, CheckSafety)
      - Every ExecuteTrajectory has robot_ns, a non-empty trajectory, operation_id
      - Every control node has at least one child; a Delay has exactly one
      - No duplicate operation IDs
      - Trajectory timestamps are monotonically increasing

    Accepts the legacy dialect too (EXECUTE/PARALLEL/…), reporting findings under
    the BT.CPP names, so a recorded tree can still be checked.

    Args:
        tree: BT dict with keys tree_id, root, metadata, and optionally blackboard

    Returns:
        {"valid": bool, "errors": [...], "warnings": [...],
         "execute_count": int, "operation_ids": [...]}
    """
    from agents.trajectory.tools.btcpp import resolve as bt_resolve

    errors:   list[str] = []
    warnings: list[str] = []
    seen_ops: set[str]  = set()
    execute_count = 0
    blackboard = tree.get("blackboard") or {}

    def walk(node: dict, depth: int = 0) -> None:
        nonlocal execute_count

        if not isinstance(node, dict):
            errors.append(f"Non-dict node at depth {depth}")
            return

        kind = node_kind(node)
        if kind not in _KNOWN_KINDS:
            errors.append(f"Unregistered node '{kind}' (id={node.get('id')})")

        if kind == "ExecuteTrajectory":
            execute_count += 1
            if not bt_port(node, "robot_ns"):
                errors.append(f"ExecuteTrajectory {node.get('id')} missing robot_ns")
            traj = bt_resolve(bt_port(node, "trajectory"), blackboard)
            if not traj:
                # An unresolvable {key} is the same defect as an empty array: the
                # executor would have nothing to dispatch.
                errors.append(f"ExecuteTrajectory {node.get('id')} has empty trajectory")
                traj = []
            if not bt_port(node, "operation_id"):
                warnings.append(f"ExecuteTrajectory {node.get('id')} missing operation_id")
            else:
                oid = bt_port(node, "operation_id")
                if oid in seen_ops:
                    warnings.append(f"Duplicate operation_id '{oid}'")
                seen_ops.add(oid)
            # Check monotonic timestamps
            if isinstance(traj, list) and len(traj) > 1:
                ts = [pt["time_s"] for pt in traj]
                for i in range(1, len(ts)):
                    if ts[i] <= ts[i-1]:
                        errors.append(f"ExecuteTrajectory {node.get('id')}: "
                                      f"timestamps not monotonic at idx {i}")
                        break

        if kind in ("Sequence", "SequenceWithMemory", "Parallel", "Delay"):
            kids = bt_children(node)
            if not kids:
                warnings.append(f"{kind} {node.get('id')} has no children")
            if kind == "Delay" and len(kids) > 1:
                errors.append(f"Delay {node.get('id')} has {len(kids)} children — "
                              f"a decorator takes exactly one")
            for child in kids:
                walk(child, depth + 1)

    root = tree.get("root")
    if root is None:
        errors.append("Tree has no 'root' key")
    else:
        walk(root)

    return {
        "valid":          len(errors) == 0,
        "errors":         errors,
        "warnings":       warnings,
        "execute_count":  execute_count,
        "operation_ids":  sorted(seen_ops),
    }


# ─── Weld-to-weld chaining check ─────────────────────────────────────────────
# Consecutive operations for the same arm must move CONTINUOUSLY: every piece's
# first joint config must equal the previous piece's last joint config, or the
# executor teleports between welds. run_trajectory_planner already chains each op
# from the arm's previous goal (trajectory_cache), so this holds automatically
# when ops are planned in execution order. These helpers only CHECK the cached
# trajectories — they never edit or recompute motion.
_CHAIN_TOL_RAD = 1e-3  # per-joint seam tolerance (radians)


def _seam_gap(prev_traj: list[dict], next_traj: list[dict]) -> Optional[float]:
    """Max per-joint |Δ| (rad) between prev_traj's last config and next_traj's
    first config. None if either piece is empty or malformed."""
    if not prev_traj or not next_traj:
        return None
    end = prev_traj[-1].get("positions")
    start = next_traj[0].get("positions")
    if not end or not start or len(end) != len(start):
        return None
    return max(abs(float(a) - float(b)) for a, b in zip(end, start))


_HOME_JOINTS = list(HOME_JOINTS_FALLBACK)


def _last_op_end_joints(cell_id: str, robot_ns: str,
                        operation_ids: list[str]) -> Optional[list]:
    """Where the arm actually IS: the end of its last planned operation.

    One owner, because the return home is planned FROM here — by the per-arm node
    below and by the coordinated set in :func:`_home_step`. Two readers of "current
    pose" that disagreed would plan a return from a pose the arm is not in.
    """
    from agents.trajectory.tools.trajectory_cache import get_cached_trajectory
    for op_id in reversed(operation_ids or []):
        t = get_cached_trajectory(cell_id, robot_ns, op_id)
        if t:
            q = list(t[-1].get("positions") or [])
            if q:
                return q
    return None


def _planned_home_node(cell_id: str, robot_ns: str, operation_ids: list[str],
                       trajectory: Optional[list] = None) -> dict:
    """Build a HOME node carrying a PLANNED, collision-free return-to-home path.

    A bare HOME node makes the executor dispatch a single HOME_JOINTS waypoint —
    the arm interpolates straight there through whatever is in the way (fixtures,
    the workpiece, the other arm). This plans the return like any other motion:

      1. lift off the last weld on-axis, at that weld's own orientation, so the
         tool leaves the seam the way it entered (same rule as weld-to-weld), then
      2. OMPL-plan joint-space from there to the HOME configuration.

    ``trajectory`` is this arm's leg out of the COORDINATED plan (:func:`_home_step`
    → ``plan_home_trajectories``), which is how the return is normally planned:
    every arm against the swept volume of the arms already planned. Only when that
    produced nothing for this arm does it fall back to planning this one arm alone,
    against static obstacles only — which is exactly the leg that has to be gated
    afterwards, because it knows nothing about the other arms.

    The resulting trajectory is attached as ``precomputed_trajectory``, which
    ``_exec_home`` already prefers over its blind fallback. Planning is
    best-effort: if OMPL/Pinocchio is unavailable or no path is found we return
    the plain node and the executor falls back to its slow single-waypoint home,
    so a plan is never lost to a failed home leg — it just degrades.
    """
    # Built directly rather than through build_behaviour_tree_node, which
    # normalises node_type for the LLM-facing surface. This is an INTERMEDIATE
    # node: to_btcpp maps HOME -> MoveHome and lifts the trajectory onto the
    # blackboard when the tree is finalised, so the assembled tree is the same
    # either way and the home-planning helper keeps one stable shape.
    node = {"id": _node_id(), "type": "HOME", "robot_ns": robot_ns}

    if trajectory:
        node["precomputed_trajectory"] = trajectory
        return node

    start_q = _last_op_end_joints(cell_id, robot_ns, operation_ids)
    if not start_q:
        return node

    try:
        # Through the backend facade: the selected backend's home planner, or the
        # default one — never a reach into a particular backend's module.
        from agents.trajectory.tools.planner_backend import (
            home_config as _home_config, plan_home_trajectory,
        )
        # The cell's own declared home_pose_deg when it has one — planning to a
        # different pose than the cell parks at would leave the arm somewhere the
        # next program does not expect.
        home_q = _home_config(cell_id, robot_ns, _HOME_JOINTS)
        traj = plan_home_trajectory(cell_id, robot_ns, start_q, home_q)
    except Exception:
        traj = None
    if traj:
        node["precomputed_trajectory"] = traj
    return node


def _parked(q: list) -> list[dict]:
    """A standing arm as a one-point trajectory: it holds this pose for all time.

    ``_sample_at_time`` returns the first/last point outside a trajectory's own
    span, so a single point is how "not moving, but still occupying space" is
    expressed to the pair gate. An arm that has finished is not an absent arm.
    """
    return [{"positions": list(q), "time_s": 0.0}]


def _gate_result(res: dict) -> str:
    """What one ``validate_trajectory_set`` answer means to the gate:

    ``clear`` / ``clash`` (arm against arm) — the question the gate asks;
    ``leg_fault`` — a leg breaches a static obstacle or its joint box ON ITS OWN:
    no order or delay fixes that, so it is never answered by serialising;
    ``failed`` — the check ran and broke: it has not said yes (refuse);
    ``unavailable`` — no geometry stack / model here: no opinion (degrade).
    """
    if res.get("error"):
        return "failed" if res.get("error_kind") == "check_failed" else "unavailable"
    if res.get("leg_check_failures"):
        return "failed"
    if res.get("static_collisions") or res.get("joint_limit_violations"):
        return "leg_fault"
    return "clear" if res.get("collision_free") else "clash"


def _leg_faults(res: dict) -> list:
    """The per-leg faults of a set answer, one per arm, each with its reason code."""
    out = []
    for v in res.get("joint_limit_violations") or []:
        out.append({"robot_ns": v.get("robot_ns"), "reason_code": "joint_limit_violation",
                    "joint_limit_violation": v})
    for c in res.get("static_collisions") or []:
        out.append({"robot_ns": c.get("robot_ns"), "reason_code": "trajectory_in_collision",
                    "collision": c})
    for f in res.get("leg_check_failures") or []:
        out.append({"robot_ns": f.get("robot_ns"), "reason_code": "leg_check_error",
                    "error": f.get("error")})
    return out


def _refuse_gate(report: dict, res: dict, kind: str, arms) -> dict:
    """Close the gate on a leg fault or a broken check (see :func:`_gate_result`)."""
    if kind == "leg_fault" or res.get("leg_check_failures"):
        faults = _leg_faults(res)
        report.update(mode=None, gated=True, leg_faults=faults,
                      blocked_arms=sorted({f["robot_ns"] for f in faults if f["robot_ns"]}
                                          or set(arms)))
    else:
        report.update(mode=None, gated=False, reason_code="inter_arm_check_error",
                      blocked_arms=sorted(arms),
                      note=f"the inter-arm check broke: {res.get('error')}")
    return report


def _gate_multi_arm_step(cell_id: str, legs: dict, *,
                         allow_parallel: bool = True,
                         standing: "Optional[dict]" = None,
                         delays: "Optional[dict]" = None,
                         bystanders: "Optional[dict]" = None) -> dict:
    """THE decision "may these arms move at once?" — for EVERY multi-arm step.

    One owner, one ladder, applied to weld steps and to the return home alike:

      1. **together** — validate the whole set as planned. Clear ⇒ PARALLEL.
      2. **serialise** — not clear ⇒ separate the arms in TIME. Greedily take an arm
         whose leg is clear of the others WHERE THEY STAND (an arm that has already
         moved stands at its leg's END, the rest at their leg's START). Costs cycle
         time, always available at plan time, and leaves every leg exactly as
         planned. For two arms this tries both orders; for more it finds an order if
         a nested one exists.
      3. **refuse** — no order is clear ⇒ the step cannot be executed, and a tree
         containing it is not a plan.

    THIS FUNCTION EXISTS BECAUSE THE ANSWER HAD TWO OWNERS. The planner prompt
    *told* the LLM that inter-robot clearance is enforced on parallel steps; the
    assembler actually enforced it on the home step only. Weld steps were wrapped in
    a ``Parallel`` node exactly as handed over, so the enforcement the prompt
    promised did not exist. Measured on ``welding_cell_2_small``, step 3 of a shipped
    plan — ``Parallel(robot1 W02, robot2 W15)`` — closed to **0.1153 m** against the
    cell's 0.2 m warning band and e-stopped the replay; the pair had never been
    checked by anything.

    IT HAS TO LIVE HERE, where the trajectories are. That pair is clear at BOTH
    endpoints — goal poses 0.9212 m apart (the clearest of the plan's six pairs),
    start poses 0.6908 m — and breaches only in transit. No check on positions, weld
    order, or scheduling metadata can see that; only the swept volumes can. Which is
    also why the scheduler upstream does not try: see
    ``task_allocation._least_movement_schedule`` for the -0.607 correlation that
    made weld-point separation an anti-objective.

    ``allow_parallel=False`` skips step 1 for a step the caller already intends to
    run sequentially — an arm moving while another merely STANDS is still an
    unchecked pair, so a declared ``SEQUENTIAL`` gets the order verified rather than
    trusted.

    A ``standing`` report from the caller is carried through untouched (the home step
    computes one before its legs exist).

    Returns a report. ``mode`` is the verdict, and it is the only thing the caller
    needs to act on:
      ``"PARALLEL"``    — run the set together (``gated`` says whether geometry
                          actually confirmed it, or the gate could not run).
      ``"SEQUENTIAL"``  — run one at a time, in ``order``; arms absent from
                          ``order`` were not gated and keep the declared order.
      ``None``          — REFUSE. ``blocked_arms`` and ``serial_attempts`` say why;
                          ``leg_faults`` when a leg breaches on its own (a static
                          obstacle, its joint box — re-plan it, no order helps);
                          ``reason_code: inter_arm_check_error`` when the check
                          broke (fail-closed: a broken check has not said yes).
    """
    from agents.trajectory.tools.planning_tools import validate_trajectory_set
    _validate = getattr(validate_trajectory_set, "func", validate_trajectory_set)

    report: dict = {"arms": sorted(legs)}
    if standing is not None:
        report["standing"] = standing

    # Arms NOT in this step still stand in the cell — where their last leg left
    # them. A step is checked against them too: an arm with no work in a step is
    # parked, not absent. (welding_cell_2_small: robot1's single-arm W04 ran past
    # robot2 parked at W10_VIA, 0.1465 m against a 0.2 m band, and e-stopped —
    # the step had been waved through as "single arm — no pair to check".)
    parked_others = {ns: _parked(q) for ns, q in (bystanders or {}).items()
                     if ns not in legs and q}
    if parked_others:
        report["bystanders"] = sorted(parked_others)

    if len(legs) < 2 and not parked_others:
        report.update(mode="PARALLEL", gated=False,
                      note="fewer than two planned legs — no pair to check")
        return report
    if len(legs) < 2:
        res = _validate(cell_id=cell_id, trajectories={**legs, **parked_others},
                        suggest_delay=False)
        report["parallel"] = res
        kind = _gate_result(res)
        if kind == "unavailable":
            report.update(mode="PARALLEL", gated=False)
        elif kind in ("failed", "leg_fault"):
            _refuse_gate(report, res, kind, legs)
        elif kind == "clear":
            report.update(mode="PARALLEL", gated=True)
        else:
            report.update(mode=None, gated=True, blocked_arms=sorted(legs),
                          blocked_by_parked=sorted(parked_others))
        return report

    if allow_parallel:
        # On the timeline the step EXECUTES: each arm's start_delay_s applied.
        # Checking a staggered step as if both arms started together would pass
        # (or refuse) a step that is not the one the robots run.
        from agents.trajectory.tools.planning_tools import apply_start_delays
        active = {ns: d for ns, d in (delays or {}).items() if d}
        together = _validate(cell_id=cell_id,
                             trajectories={**apply_start_delays(legs, active),
                                           **parked_others},
                             suggest_delay=False)
        if active:
            report["start_delays"] = active
        report["parallel"] = together
        kind = _gate_result(together)
        if kind == "clear":
            report.update(mode="PARALLEL", gated=True)
            return report
        if kind == "unavailable":
            # The gate itself could not run (no Pinocchio, no model). It cannot then
            # be the thing that refuses a tree: the legs were planned by the same
            # geometry stack, so this is only reachable in a stubbed environment.
            # Degrade visibly instead of pretending the step was checked.
            report.update(mode="PARALLEL", gated=False)
            return report
        if kind in ("failed", "leg_fault"):
            # A leg that breaches on its own is not a pairing problem (serialising
            # would only re-find it, then call it "no safe order"); a check that
            # broke has not said yes.
            return _refuse_gate(report, together, kind, legs)

    parked = {ns: list(legs[ns][0].get("positions") or []) for ns in legs}
    order: list[str] = []
    attempts: list[dict] = []
    remaining = sorted(legs)
    while remaining:
        for ns in remaining:
            others = {**{o: _parked(parked[o]) for o in legs if o != ns},
                      **parked_others}
            res = _validate(cell_id=cell_id,
                            trajectories={ns: legs[ns], **others},
                            suggest_delay=False)
            kind = _gate_result(res)
            if kind == "unavailable":
                report.update(mode="SEQUENTIAL", gated=False, order=order,
                              serial_attempts=attempts,
                              note=f"gate could not run: {res.get('error')}")
                return report
            if kind in ("failed", "leg_fault"):
                report["serial_attempts"] = attempts
                return _refuse_gate(report, res, kind, [ns])
            attempts.append({"robot_ns": ns, "after": list(order),
                             "min_clearance_m": res.get("min_clearance_m"),
                             "collision_free": bool(res.get("collision_free"))})
            if res.get("collision_free"):
                order.append(ns)
                parked[ns] = list(legs[ns][-1].get("positions") or [])
                remaining.remove(ns)
                break
        else:
            report.update(mode=None, gated=True, order=order,
                          serial_attempts=attempts, blocked_arms=sorted(remaining))
            return report

    report.update(mode="SEQUENTIAL", gated=True, order=order,
                  serial_attempts=attempts)
    return report


def _order_step_nodes(nodes: list[dict], report: dict) -> dict:
    """Turn one step's action nodes into the node the gate's verdict calls for.

    ``PARALLEL`` keeps the caller's order under a ``Parallel``; ``SEQUENTIAL``
    re-orders to the gate's ``order`` under a ``Sequence``. Arms the gate did not
    place (it degraded, or they had no leg to check) keep the declared order behind
    the ones it did — dropping them would drop their work.

    A single node needs no wrapper: a ``Parallel`` of one is the same motion with an
    extra tick.
    """
    if report.get("mode") == "SEQUENTIAL":
        order = report.get("order") or []
        nodes = ([n for ns in order for n in nodes if n.get("robot_ns") == ns]
                 + [n for n in nodes if n.get("robot_ns") not in order])
    if len(nodes) == 1:
        return nodes[0]
    return build_behaviour_tree_node.func(
        node_type="Sequence" if report.get("mode") == "SEQUENTIAL" else "Parallel",
        children=nodes)


def _home_step(cell_id: str, robots: list[str],
                per_robot_ops: dict) -> "tuple[Optional[dict], dict]":
    """:func:`_home_step_planned`, reused when nothing it depends on has changed.

    The return home depends only on WHERE each arm stands (the end of its last
    planned leg), the cell's config and the planning backend — not on the rest of
    the program. Planning it is the most expensive single call of an assembly
    (539 s of bt_1793277501's), and an assembly is repeated whenever the agent
    fixes a refused step, so an accepted return is memoised (plan_memo) on exactly
    those inputs. A refused return is never reused: it is planned again, since a
    sampling planner may find one next time. The memo lives for ONE planning run
    (plan_memo.run_scope): the return also reads the trajectory cache, which only
    the run keeps consistent. A hit leaves the trajectory cache as the planning did
    (each homed arm no longer stands on a seam)."""
    from agents.trajectory.tools import plan_memo
    starts = {ns: _last_op_end_joints(cell_id, ns, per_robot_ops.get(ns) or [])
              for ns in robots}
    try:
        from agents.trajectory.tools.planner_backend import resolve_planning_backend
        backend = resolve_planning_backend()
    except Exception:  # noqa: BLE001
        backend = None
    key = plan_memo.make_key(
        cell_id, "home", sorted(robots), backend,
        {ns: ([round(float(x), 6) for x in q] if q else None) for ns, q in starts.items()},
        # The seam each arm lifts off from is its LAST operation's weld pose.
        {ns: list(per_robot_ops.get(ns) or [])[-1:] for ns in robots})
    hit = plan_memo.get_run("home_step", key)
    if hit is not None:
        node, report = hit
        try:
            from agents.trajectory.tools.trajectory_cache import clear_last_pose
            for ns, q in starts.items():
                if q:
                    clear_last_pose(cell_id, ns)
        except Exception:  # noqa: BLE001
            pass
        report["memoised"] = True
        return node, report
    node, report = _home_step_planned(cell_id, robots, per_robot_ops)
    if node is not None:
        plan_memo.put_run("home_step", key, (node, report))
    return node, report


def _home_step_planned(cell_id: str, robots: list[str],
                       per_robot_ops: dict) -> "tuple[Optional[dict], dict]":
    """The end-of-program return to HOME — planned COUPLED and GATED like any step.

    Every other PARALLEL step in the tree was validated by the caller with
    ``validate_trajectory_set`` before the assembler ever saw it. This one is
    different: the ASSEMBLER makes it, so nothing upstream had ever seen these
    trajectories — it was the one motion in the tree that no inter-robot check
    covered. Measured on ``welding_cell_2_small``: two home legs, each individually
    valid because each was planned against static obstacles ONLY, passing 0.089 m
    apart against the cell's 0.2 m warning band. The replay e-stopped 18 nodes in,
    on the last motion of the program.

    Going home is an ordinary planned motion and gets the ordinary treatment:

      1. the set is planned COUPLED (``plan_home_trajectories``) — each arm
         against the swept volume of the arms already planned, and each leaving
         its seam on-axis first, the same coupling a PARALLEL weld step gets;
      2. the result goes through the SAME gate the LLM applies to every other
         parallel step;
      3. if simultaneous homing still enters the warning band, the arms are homed
         one after another — a Sequence, always available at plan time, costing
         only cycle time — in an order where each mover clears the others where
         they stand;
      4. if no order is clear either, the step is REFUSED, and with it the tree.
         A plan whose last motion e-stops the cell is not a plan.

    Case 0, checked before any of that: are the arms clear of each other WHERE
    THEY STAND? Measured on ``welding_cell_2_small``, arms finishing on W01 and W04 are
    0.0367 m apart while both are stationary — already inside the 0.2 m warning
    band with nothing moving. No home plan repairs that: the monitor measures the
    PAIR, not the mover, so it e-stops whichever arm goes first, and homing them
    one at a time cannot separate poses they are already in. It is a different
    fault with a different repair — the PROGRAM put them there, so its last welds
    are what have to be re-staged — so it is named rather than reported as "no
    safe order", and it is checked first because it also saves planning a return
    that cannot be executed.

    Steps 2-4 are not this function's own: they are ``_gate_multi_arm_step``, the
    same gate every multi-arm weld step now goes through. What is specific here is
    step 1 (the coupled plan) and case 0 below.

    Returns ``(node, report)``; ``node`` is None only for case 0 and case 4.
    """
    from agents.trajectory.tools.planning_tools import validate_trajectory_set
    _validate = getattr(validate_trajectory_set, "func", validate_trajectory_set)

    starts = {}
    for ns in robots:
        q = _last_op_end_joints(cell_id, ns, per_robot_ops.get(ns) or [])
        if q:
            starts[ns] = q

    standing = None
    if len(starts) > 1:
        standing = _validate(
            cell_id=cell_id,
            trajectories={ns: _parked(q) for ns, q in starts.items()},
            suggest_delay=False)
        # An ``error`` here means the gate could not run at all (no geometry
        # stack); that must not masquerade as a breach.
        if not (standing.get("collision_free") or standing.get("error")):
            return None, {
                "arms": list(robots), "coupled_arms": [], "blind_arms": [],
                "standing": standing, "mode": None, "gated": True,
                "blocked_arms": sorted(starts),
                "note": ("the arms are inside the cell's warning band where they "
                         "STAND, before any home motion exists"),
            }

    coupled: dict = {}
    if len(starts) > 1:
        try:
            from agents.trajectory.tools.planner_backend import plan_home_trajectories
            coupled = plan_home_trajectories(cell_id, starts) or {}
        except Exception:
            coupled = {}

    nodes = [_planned_home_node(cell_id, ns, per_robot_ops.get(ns) or [],
                                trajectory=coupled.get(ns))
             for ns in robots]
    legs = {n["robot_ns"]: n["precomputed_trajectory"]
            for n in nodes if n.get("precomputed_trajectory")}

    # An arm whose planned return was REFUSED — no on-axis way out of its seam,
    # or a leg that failed a check — has no planned leg, and the blind home that
    # would replace it is that same unchecked motion (dragging the tool out of the
    # seam, or crossing the obstacle the planned leg hit). Refuse the step (and the
    # tree) with the reason, so the caller repositions that arm first.
    from agents.trajectory.tools.planner_backend import home_refusal
    refused = {n["robot_ns"]: home_refusal(cell_id, n["robot_ns"], starts.get(n["robot_ns"]))
               for n in nodes
               if not n.get("precomputed_trajectory") and n["robot_ns"] in starts}
    refused = {ns: why for ns, why in refused.items() if why}
    if refused:
        liftoff = {ns: w for ns, w in refused.items()
                   if w.get("reason_code") == "home_liftoff_blocked"}
        return None, {
            "arms": list(robots), "coupled_arms": sorted(coupled), "blind_arms": [],
            "standing": standing, "mode": None, "gated": True,
            "blocked_arms": sorted(refused), "home_refused": refused,
            "liftoff_blocked": liftoff,
            "note": ("no safe planned return for " + ", ".join(sorted(refused)) + " ("
                     + "; ".join(f"{ns}: {w.get('reason_code')}" for ns, w in
                                 sorted(refused.items()))
                     + "), and the blind home that would replace it is the same "
                       "unchecked motion"),
        }

    report = _gate_multi_arm_step(cell_id, legs, standing=standing)
    report["arms"] = list(robots)
    report["coupled_arms"] = sorted(coupled)
    # An arm with no planned leg homes on the executor's blind straight line.
    # Nothing can gate that motion, so it is named rather than counted as checked.
    report["blind_arms"] = [n["robot_ns"] for n in nodes
                            if not n.get("precomputed_trajectory")]
    if len(legs) < 2:
        report["note"] = "fewer than two planned home legs — no pair to check"

    if report["mode"] is None:
        return None, report
    if report["mode"] == "SEQUENTIAL" and len(legs) > 1:
        staggered = _staggered_home(cell_id, nodes, legs, report)
        if staggered is not None:
            return staggered
    return _order_step_nodes(nodes, report), report


def _staggered_home(cell_id: str, nodes: list, legs: dict, report: dict):
    """A serialised return home, started STAGGERED instead when that is faster.

    One arm after the other costs the sum of the legs; the smallest start delay
    that clears them (``find_start_delays``, robust to ±its tolerance) usually
    costs a fraction of one. Used only when it is shorter and the gate, run again
    on the delayed timeline, says PARALLEL. Returns ``(node, report)`` or None."""
    try:
        from agents.trajectory.tools.planning_tools import find_start_delays
        fix = find_start_delays(cell_id, legs)
    except Exception:  # noqa: BLE001 — no stagger: the serial order stands
        return None
    if not fix:
        return None
    delays = {ns: float(d) for ns, d in (fix.get("start_delays") or {}).items() if d}
    dur = {ns: float(l[-1]["time_s"]) for ns, l in legs.items()}
    serial_s = sum(dur.values())
    stag_s = max(dur[ns] + delays.get(ns, 0.0) for ns in dur)
    if stag_s >= serial_s - 0.25:
        return None
    check = _gate_multi_arm_step(cell_id, legs, delays=delays)
    if check.get("mode") != "PARALLEL" or not check.get("gated"):
        return None
    wrapped = []
    for n in nodes:
        d = delays.get(n.get("robot_ns"), 0.0)
        wrapped.append({"type": "Delay", "ports": {"delay_msec": int(round(d * 1000))},
                        "children": [n]} if d > 0 else n)
    out = {**report, "mode": "PARALLEL", "gated": True, "start_delays": delays,
           "staggered": {"serial_s": round(serial_s, 2), "staggered_s": round(stag_s, 2),
                         "saved_s": round(serial_s - stag_s, 2),
                         "min_clearance_m": (check.get("parallel") or {}).get("min_clearance_m")},
           "serial_order": report.get("order"), "parallel": check.get("parallel")}
    out.pop("order", None)
    return build_behaviour_tree_node.func(node_type="Parallel", children=wrapped), out


def _arm_home(cell_id: str, robot_ns: str) -> Optional[list]:
    """Where the arm stands when the program starts (``agents.home_pose``), or None
    when that cannot be answered — then the first-piece check of
    :func:`_chain_breaks` is skipped rather than measured against a guess."""
    from agents.home_pose import program_start_pose
    return program_start_pose(cell_id, robot_ns)


def _chain_breaks(cell_id: str, robot_ns: str, operation_ids: list[str],
                  tol: float = _CHAIN_TOL_RAD) -> tuple[int, float, list[dict]]:
    """Check one arm's ordered operation_ids for weld-to-weld continuity.

    Returns (pieces_found, max_gap_rad, breaks). A break is either a missing
    cached trajectory, a seam where the piece does not start where the previous
    one ended, or the arm's FIRST piece not starting at the arm's HOME (where the
    pre-run home reset leaves it, and where every backend chains a program's
    first leg from). Does NOT modify anything.
    """
    from agents.trajectory.tools.trajectory_cache import get_cached_trajectory
    breaks: list[dict] = []
    max_gap = 0.0
    pieces = 0
    prev_id: Optional[str] = None
    prev_t: Optional[list] = None
    # The first piece must start at HOME, or the executor jumps from HOME to its
    # first waypoint. That happens when an operation is re-planned from the end
    # of its OWN earlier plan, which the seam check between consecutive pieces
    # cannot see for the first operation (umarell issue 16).
    home = _arm_home(cell_id, robot_ns)
    first = True
    for op_id in operation_ids:
        t = get_cached_trajectory(cell_id, robot_ns, op_id)
        if t and first and home:
            gap = _seam_gap([{"positions": home}], t)
            if gap is not None:
                max_gap = max(max_gap, gap)
                if gap > tol:
                    breaks.append({
                        "robot_ns": robot_ns,
                        "from_operation_id": "HOME",
                        "to_operation_id": op_id,
                        "gap_rad": round(gap, 5),
                        "end_joints": [round(float(x), 6) for x in home],
                        "start_joints": [round(float(x), 6) for x in t[0].get("positions", [])],
                        "reason": "start_gap",
                        "fix": (f"'{op_id}' is {robot_ns}'s first operation but does not "
                                f"start at HOME (end_joints): re-plan it with "
                                f"current_joints set to HOME"),
                    })
        first = False          # the FIRST operation, whether or not it is cached
        if not t:
            breaks.append({
                "robot_ns": robot_ns,
                "operation_id": op_id,
                "reason": "missing_trajectory",
                "fix": ("no cached trajectory — run run_trajectory_planner for this "
                        "operation_id before chaining/assembling"),
            })
            prev_id, prev_t = op_id, None
            continue
        pieces += 1
        if prev_t:
            gap = _seam_gap(prev_t, t)
            if gap is not None:
                max_gap = max(max_gap, gap)
                if gap > tol:
                    breaks.append({
                        "robot_ns": robot_ns,
                        "from_operation_id": prev_id,
                        "to_operation_id": op_id,
                        "gap_rad": round(gap, 5),
                        "end_joints": [round(float(x), 6) for x in prev_t[-1].get("positions", [])],
                        "start_joints": [round(float(x), 6) for x in t[0].get("positions", [])],
                        "reason": "chain_gap",
                        "fix": (f"re-plan '{op_id}' with current_joints set to the end "
                                f"of '{prev_id}' (end_joints), then this arm's later ops "
                                f"in execution order — all in ONE run_trajectory_planner "
                                f"call with steps=[...]; the later ones chain "
                                f"automatically (omit their current_joints)"),
                    })
        prev_id, prev_t = op_id, t
    return pieces, round(max_gap, 5), breaks


@tool
def chain_trajectories(cell_id: str, robots: list[dict]) -> dict:
    """Verify weld-to-weld CHAINING for each robot and report any broken seam.

    Call this AFTER every operation is planned and BEFORE assembling, to confirm
    each arm moves CONTINUOUSLY — every operation's trajectory must START exactly
    where that arm's PREVIOUS operation ended, or the executor teleports/jumps
    between welds.

    run_trajectory_planner already chains each op from the arm's previous goal, so
    a break almost always means the ops were planned OUT OF ORDER, two ops for the
    same arm went in one call, or an op got re-planned from HOME. This tool only
    CHECKS the cached trajectories (matched by operation_id) — it never edits or
    recomputes them. It does the chaining bookkeeping FOR you and reports exactly
    which piece to re-plan.

    Args:
        cell_id: the cell.
        robots:  per robot, its operation_ids IN EXECUTION ORDER, e.g.
                 [{"robot_ns": "robot1", "operation_ids": ["R1_W1", "R1_W2", "R1_W3"]},
                  {"robot_ns": "robot2", "operation_ids": ["R2_W7", "R2_W8"]}]

    Returns:
        {"success": true,  "chained": true,
         "robots": {ns: {"pieces": N, "max_gap_rad": g}}}
        or, on a discontinuity:
        {"success": false, "chained": false, "robots": {...}, "breaks": [ ... ],
         "advice": "re-plan each to_operation_id with current_joints = the
                    end_joints of its from_operation_id, then chain again"}
    """
    entries = robots or []
    if not entries:
        return {"success": False,
                "error": "robots is empty — pass per-robot ordered operation_ids"}
    summary: dict[str, dict] = {}
    all_breaks: list[dict] = []
    for e in entries:
        ns = e.get("robot_ns")
        ops = e.get("operation_ids") or []
        pieces, max_gap, breaks = _chain_breaks(cell_id, ns, ops)
        summary[ns] = {"pieces": pieces, "max_gap_rad": max_gap}
        all_breaks.extend(breaks)
    if all_breaks:
        return {
            "success": False,
            "chained": False,
            "robots": summary,
            "breaks": all_breaks,
            "advice": ("each listed operation does not start where the previous one "
                       "ended. Re-plan each `to_operation_id` with `current_joints` set "
                       "to the `end_joints` of its `from_operation_id`, followed by that "
                       "arm's later operations in execution order — one "
                       "run_trajectory_planner call with `steps=[...]` covers them all, "
                       "the later ones chaining automatically — then call "
                       "chain_trajectories again. A `missing_trajectory` means you must "
                       "run_trajectory_planner for that operation_id first."),
        }
    return {"success": True, "chained": True, "robots": summary}


@tool
def assemble_behaviour_tree(cell_id: str, plan: dict) -> dict:
    """
    Assemble the COMPLETE Behaviour Tree in one shot from a lightweight plan of
    DECISIONS (you have already made). This is the standard way to build the tree
    — do NOT hand-assemble nodes one by one.

    The full joint trajectories are pulled from the server-side cache populated by
    run_trajectory_planner (matched by operation_id), so you do NOT pass any
    trajectory arrays here — only the step plan.

    plan schema:
    {
      "steps": [
        { "mode": "PARALLEL" | "SEQUENTIAL",
          "robots": [
            { "robot_ns": "robot1",
              "operation_id": "R1_W1",          # must match what you passed to
                                                # run_trajectory_planner
              "operation_type": "weld",
              "start_delay_s": 0.0,             # optional
              "has_interlock": false,           # optional
              "interlock_target": {...},        # optional
              "hold_duration_s": 0.5,           # optional
              "velocity_scale": 1.0,            # optional
              "depends_on_robots": [...],       # optional (interlock gating)
              "depends_on_positions": [...]     # optional
            }
          ]
        }
      ],
      "home_at_end": true                       # optional (default true)
    }

    Each step becomes a `Parallel` (arms simultaneously) or `Sequence` (one after
    another) node; steps run in order under a top-level `Sequence`. A
    `start_delay_s` becomes a `Delay` decorator wrapping that arm's action. The
    tree is validated before returning.

    Your `mode` on a multi-arm step is a REQUEST, not the outcome. Every step with
    two or more arms is checked here as SWEPT VOLUMES against the cell's warning
    distance, and the check decides: clear together ⇒ `Parallel`; not clear ⇒
    `Sequence` in an order where each mover clears the others where they stand
    (reported in `serialised_steps`, and it costs cycle time); no safe order ⇒
    assembly FAILS with `blocked_steps`. A declared `SEQUENTIAL` still gets its order
    verified, because an arm moving past another that merely STANDS is a real pair.
    `step_gates` reports the verdict and measured clearance for every gated step.

    Do not try to pre-empt this by pairing operations that are far apart: two arms
    facing each other are FURTHEST apart at the poses where each is most extended
    toward the other. On `welding_cell_2_small` weld-point separation correlates with true
    arm clearance at -0.607, and the breaching pair is clear at both its start
    (0.6908 m) and goal (0.9212 m) poses while closing to 0.1153 m in transit. If a
    step is refused, pair those operations with a DIFFERENT arm's operation, or give
    each its own step.

    The final `MoveHome` step (unless `home_at_end: false`) is planned HERE, and it
    is planned like any other motion: every arm lifts off its last seam on-axis,
    the arms are planned against each other's swept volume, and the resulting set
    goes through the same inter-robot gate you apply to a parallel weld step. If
    homing together would enter the cell's warning band the arms are homed one
    after another instead; if no order is safe, assembly FAILS with `home_step`
    explaining which pair and how close — recover by staging the arms apart before
    they home, or set `home_at_end: false` and plan the returns as ordinary
    operations. `home_step.blind_arms` lists any arm whose return could not be
    planned at all: it homes on the executor's blind straight line, which nothing
    checks.

    One failure there is NOT about the return at all: `home_step.standing` reports
    whether the arms are clear of each other in the poses their LAST operations
    leave them in. If they are already inside the warning band standing still, no
    home plan and no homing order can help — fix the program instead (re-order or
    re-allocate those last operations, or add a reposition as a final operation on
    one arm) and the return plans itself.

    Returns:
        {"success": bool,
         "behaviour_tree": {tree_id, cell_id, BTCPP_format: "4",
                            main_tree_to_execute: "MainTree", root, blackboard,
                            xml, metadata},
         "validation": {...},
         "step_gates": [{step, requested_mode, mode, gated, arms, ...}],
         "serialised_steps": [n, ...],   # steps the gate turned into a Sequence
         "home_step": {mode, gated, arms, coupled_arms, blind_arms, ...}}
        or {"success": false, "error": str, "missing": [...]}  # if cache miss
        or {"success": false, "error": str, "repeated_arms": [...]}  # arm twice in a step
        or {"success": false, "error": str, "blocked_steps": [...]}  # unsafe pairing
        or {"success": false, "error": str, "home_step": {...}}  # unsafe homing

    ``root`` is BehaviorTree.CPP JSON (nodes carry ``type`` = registration ID and
    ``ports``); ``xml`` is the same tree as ``<root BTCPP_format="4">`` for Groot2
    or a C++ factory; ``blackboard`` holds the joint arrays the ports reference.
    """
    timer = _AssemblyTimer()
    token = _ASSEMBLY_TIMER.set(timer)
    try:
        out = _assemble_behaviour_tree(cell_id, plan)
    finally:
        _ASSEMBLY_TIMER.reset(token)
    report = timer.report()
    if isinstance(out, dict):
        out["timings_s"] = report
    print(f"[assemble] {cell_id}: {report['total']:.1f} s — "
          + ", ".join(f"{k} {v:.1f}" for k, v in report.items()
                      if k not in ("total", "slowest_step_gates") and v >= 0.05)
          + (f"; slowest step gates {report['slowest_step_gates']}"
             if report.get("slowest_step_gates") else ""), flush=True)
    return out


class _AssemblyTimer:
    """Wall time per assembly phase — where an assembly's minutes go
    (``timings_s`` on every result, and one ``[assemble]`` log line)."""

    def __init__(self):
        import time as _t
        self._now = _t.monotonic
        self._t0 = self._last = self._now()
        self.phases: dict = {}
        self.step_gates: list = []

    def mark(self, phase: str) -> None:
        now = self._now()
        self.phases[phase] = self.phases.get(phase, 0.0) + (now - self._last)
        self._last = now

    def step_gate(self, step, seconds: float) -> None:
        self.step_gates.append((float(seconds), step))

    def report(self) -> dict:
        self.mark("other")
        out = {k: round(v, 2) for k, v in self.phases.items()}
        out["total"] = round(self._now() - self._t0, 2)
        if self.step_gates:
            top = sorted(self.step_gates, reverse=True)[:3]
            out["slowest_step_gates"] = {f"step {st}": round(sec, 2) for sec, st in top}
        return out


_ASSEMBLY_TIMER: "ContextVar[Optional[_AssemblyTimer]]" = ContextVar("assembly_timer",
                                                                     default=None)


def _mark(phase: str) -> None:
    t = _ASSEMBLY_TIMER.get()
    if t is not None:
        t.mark(phase)


def _assemble_behaviour_tree(cell_id: str, plan: dict) -> dict:
    """The body of :func:`assemble_behaviour_tree` (timed by it)."""
    import time as _time
    from agents.trajectory.tools.trajectory_cache import (
        get_cached_trajectory, new_tree_id, weld_pose_record)

    steps = plan.get("steps") or []
    if not steps:
        return {"success": False, "error": "plan.steps is empty"}

    missing: list[dict] = []
    step_specs: list[dict] = []
    op_count = 0

    # REFUSE a plan that reuses an operation_id. The trajectory cache is keyed on
    # (cell_id, robot_ns, operation_id), so a reused id means the second plan
    # silently OVERWROTE the first and both nodes would execute the same motion —
    # one operation would never be performed. Undetectable downstream, so it is
    # fatal here rather than a warning.
    dup_ids: list[dict] = []
    seen_pairs: set = set()
    for si, step in enumerate(steps):
        for r in (step.get("robots") or []):
            pair = (r.get("robot_ns"), r.get("operation_id"))
            if pair[1] is None:
                continue
            if pair in seen_pairs:
                dup_ids.append({"step": si + 1, "robot_ns": pair[0], "operation_id": pair[1]})
            seen_pairs.add(pair)
    if dup_ids:
        return {
            "success": False,
            "error": ("duplicate operation_id in the plan — the trajectory cache is keyed "
                      "on (cell_id, robot_ns, operation_id), so a reused id means one "
                      "operation's trajectory overwrote another's and that operation would "
                      "never be performed. Give every operation a unique id and re-plan the "
                      "ones that collided."),
            "duplicates": dup_ids,
        }

    # REFUSE a step that lists the same arm twice. One arm cannot execute two
    # motions at once, so in a PARALLEL step it is a contradiction; and the pair gate
    # below reasons about one leg per arm, so in a SEQUENTIAL step it would silently
    # gate only one of the two. Consecutive steps express the same intent without the
    # ambiguity, so this is a plan-shape error, not a judgement call.
    repeat_arms: list[dict] = []
    for si, step in enumerate(steps):
        seen_ns: set = set()
        for r in (step.get("robots") or []):
            ns = r.get("robot_ns")
            if ns is None:
                continue
            if ns in seen_ns:
                repeat_arms.append({"step": si + 1, "robot_ns": ns,
                                    "operation_id": r.get("operation_id")})
            seen_ns.add(ns)
    if repeat_arms:
        return {
            "success": False,
            "error": ("a step lists the same robot_ns more than once — one arm cannot "
                      "run two motions simultaneously, and the inter-robot gate checks "
                      "one leg per arm per step, so the extra one would go unchecked. "
                      "Split those operations into consecutive steps (one op per arm "
                      "per step); the arm still performs them back to back."),
            "repeated_arms": repeat_arms,
        }

    for si, step in enumerate(steps):
        mode   = str(step.get("mode", "PARALLEL")).upper()
        robots = step.get("robots") or []
        exec_nodes: list[dict] = []
        legs: dict = {}
        for r in robots:
            ns    = r.get("robot_ns")
            op_id = r.get("operation_id")
            traj  = get_cached_trajectory(cell_id, ns, op_id)
            if not traj:
                missing.append({"step": si + 1, "robot_ns": ns, "operation_id": op_id})
                continue
            legs[ns] = traj
            exec_nodes.append(build_behaviour_tree_node.func(
                node_type="ExecuteTrajectory",
                robot_ns=ns,
                trajectory=traj,
                operation_id=op_id,
                operation_type=r.get("operation_type", "weld"),
                has_interlock=bool(r.get("has_interlock", False)),
                interlock_target=r.get("interlock_target"),
                depends_on_robots=r.get("depends_on_robots"),
                depends_on_positions=r.get("depends_on_positions"),
                start_delay_s=float(r.get("start_delay_s", 0.0)),
                hold_duration_s=float(r.get("hold_duration_s", 0.5)),
                # MUST stay 1.0. Speed is already baked into each waypoint's
                # time_s by the planner, and it is baked in PER PHASE (transit
                # fast, terminal approach slow). The executor and driver divide
                # time_s by this value, so anything other than 1.0 would re-scale
                # both phases uniformly and destroy the two-phase speed split.
                # To change a speed, re-plan with velocity_scale /
                # approach_velocity_scale on the motion request.
                velocity_scale=1.0,
            ))
            op_count += 1

        if not exec_nodes:
            continue
        # Hold the step, do not wrap it yet: whether it becomes a Parallel or a
        # Sequence is the gate's answer, and the gate is geometry — too expensive to
        # run on a plan the cheap checks below are about to reject.
        step_specs.append({"index": si + 1, "mode": mode,
                           "nodes": exec_nodes, "legs": legs,
                           "ops": {r.get("robot_ns"): r.get("operation_id") for r in robots},
                           "delays": {r.get("robot_ns"): float(r.get("start_delay_s", 0.0) or 0.0)
                                      for r in robots if r.get("robot_ns") in legs}})

    _mark("load_steps")
    if missing:
        return {
            "success": False,
            "error": ("missing cached trajectories — run run_trajectory_planner with a "
                      "matching operation_id for each of these before assembling"),
            "missing": missing,
        }
    # GUARANTEE weld-to-weld chaining: per robot (in execution order across the
    # steps), every operation must start where its previous operation ended. The
    # planner chains from the arm's previous goal, so this holds when ops were
    # planned in order — but we REFUSE to build a tree with a gap rather than let
    # the executor teleport. We only CHECK the cached trajectories; we never
    # rewrite motion. A break is reported back so the LLM re-plans that op.
    per_robot_ops: dict[str, list[str]] = {}
    for step in steps:
        for r in (step.get("robots") or []):
            rns, rop = r.get("robot_ns"), r.get("operation_id")
            if rns and rop:
                per_robot_ops.setdefault(rns, []).append(rop)
    chain_breaks: list[dict] = []
    for rns, rops in per_robot_ops.items():
        _, _, brks = _chain_breaks(cell_id, rns, rops)
        chain_breaks.extend(brks)
    _mark("chain_check")
    if chain_breaks:
        return {
            "success": False,
            "error": ("weld-to-weld chain broken — an operation does not start where "
                      "the arm's previous operation ended (the executor would teleport). "
                      "Re-plan each `to_operation_id` with current_joints = the "
                      "`end_joints` of its `from_operation_id`, in execution order, then "
                      "assemble again."),
            "breaks": chain_breaks,
        }
    if not step_specs:
        return {"success": False, "error": "no executable steps could be assembled"}

    # The per-leg checks first (joint limits, the arm against itself along the
    # path): cheaper than the step gates below, and a leg that fails them cannot
    # run in any order (umarell issues 13, 14). The home legs are checked once the
    # home step exists.
    leg_hits = _leg_audit(cell_id, step_specs, None, per_robot_ops)
    _mark("leg_audit")
    if leg_hits:
        return _leg_refusal_result(leg_hits)

    # INTER-ROBOT GATE on every multi-arm step — the same one the return home gets.
    # Until this existed the assembler wrapped whatever it was handed in a Parallel
    # and only the home step was checked, so a pair that breached in transit reached
    # the executor unexamined (welding_cell_2_small step 3: 0.1153 m against a 0.2 m
    # warning band, e-stopped the replay). A step the gate can serialise is
    # serialised; one it cannot is refused, and ALL of them are reported at once so
    # the caller re-plans in a single pass instead of discovering them one replay at
    # a time.
    # Back-to-back moves of ONE arm (no other arm working in between) run as one
    # leg: each separate node costs a hold and a dispatch round, and a retreat the
    # agent made in three pieces is still one motion. The merged leg is the same
    # path on one clock, and it is gated below like any step.
    step_specs, merged_moves = _merge_moves(step_specs)

    step_nodes: list[dict] = []
    step_reports: list[dict] = []
    # Where each arm stands between steps: the end of its latest leg — and HOME for
    # an arm that has not moved yet. It is in the cell there, and an arm working
    # past it must clear it (a first step with one arm used to go unchecked).
    at_pose: dict = {}
    try:
        from agents.trajectory.tools.planning_tools import _cell_robots
        cell_arms = list(_cell_robots(cell_id))
    except Exception:  # noqa: BLE001 — no robots.yaml here: the plan's arms
        cell_arms = []
    for ns in dict.fromkeys([*per_robot_ops, *cell_arms]):
        home = _arm_home(cell_id, ns)
        if home:
            at_pose[ns] = list(home)
    for spec in step_specs:
        bystanders = {ns: q for ns, q in at_pose.items() if ns not in spec["legs"]}
        if len(spec["legs"]) > 1 or bystanders:
            import time as _time
            _g0 = _time.monotonic()
            report = _gate_multi_arm_step(
                cell_id, spec["legs"],
                allow_parallel=not spec["mode"].startswith("SEQ"),
                delays=spec["delays"], bystanders=bystanders)
            _tm = _ASSEMBLY_TIMER.get()
            if _tm is not None:
                _tm.step_gate(spec["index"], _time.monotonic() - _g0)
            report["step"] = spec["index"]
            report["requested_mode"] = spec["mode"]
            step_reports.append(report)
        else:
            report = {"step": spec["index"], "requested_mode": spec["mode"],
                      "arms": sorted(spec["legs"]), "mode": spec["mode"],
                      "gated": False, "note": "single arm — no pair to check"}
        if report.get("mode") is not None:
            step_nodes.append(_order_step_nodes(spec["nodes"], report))
        for ns, leg in spec["legs"].items():
            if leg:
                at_pose[ns] = list(leg[-1].get("positions") or [])

    _mark("step_gates")
    blocked = [r for r in step_reports if r.get("mode") is None]
    faulty = [r for r in blocked if r.get("leg_faults") or r.get("reason_code")]
    if faulty:
        # Not a pairing problem: a leg breaches on its own, or the check broke.
        return {
            "success": False,
            "error": ("these steps cannot be gated: "
                      + "; ".join(
                          f"step {r.get('step')}: "
                          + (", ".join(f"{f['robot_ns']} {f['reason_code']}"
                                       for f in r.get("leg_faults") or [])
                             or r.get("note") or "check failed")
                          for r in faulty)
                      + ". A leg that breaches a static obstacle or its joint box ON "
                        "ITS OWN is fixed by re-planning that leg (no order, delay or "
                        "pairing helps); a check that broke: retry once, then report it."),
            "blocked_steps": blocked,
            "leg_refusals": [{"step": r.get("step"), **f}
                             for r in faulty for f in r.get("leg_faults") or []],
            "step_gates": step_reports,
        }
    if blocked:
        return {
            "success": False,
            "error": ("these steps put two arms inside the cell's warning band and no "
                      "order runs them one at a time clear of each other either — the "
                      "runtime monitor would E-STOP. The arms are checked as SWEPT "
                      "VOLUMES, so a pair can be clear at both its start and its goal "
                      "poses and still breach in transit; re-pairing on positions will "
                      "not fix it. Recover by pairing each listed operation with a "
                      "DIFFERENT arm's operation (call find_interlock_set for the pair "
                      "to see where they conflict), or by giving each its own step so "
                      "the arms move one at a time — but first try a start delay "
                      "(validate_trajectory_set returns delay_fix) or a different weld "
                      "pairing, which keep the step parallel. If even one-at-a-time fails, the "
                      "IDLE arm is parked in the way at its last weld: plan a 'move' "
                      "operation that retreats it clear (back toward its own base, "
                      "raised) and put it in its own step before the other arm works. "
                      "Do NOT take an arm's reachable welds away to avoid this."),
            "blocked_steps": blocked,
            "step_gates": step_reports,
        }

    # Optional HOME step at the end (all robots return home). Planned coupled and
    # inter-robot gated in _home_step — it is the one step the assembler itself
    # creates, so it is the one step the caller's own validation never saw.
    home_report: Optional[dict] = None
    home_node: Optional[dict] = None
    if plan.get("home_at_end", True):
        robots_all = sorted({
            r.get("robot_ns")
            for s in steps for r in (s.get("robots") or [])
            if r.get("robot_ns")
        })
        if robots_all:
            _mark("before_home")
            home_node, home_report = _home_step(cell_id, robots_all, per_robot_ops)
            _mark("home_step")
            if home_node is None:
                # Two different faults, two different repairs. Saying "no safe
                # order" when the arms are ALREADY too close sends the caller
                # looking for an ordering that cannot exist.
                standing = (home_report or {}).get("standing") or {}
                if (home_report or {}).get("home_refused"):
                    why = (
                        f"{home_report['note']}. Recover by ending that arm's sequence "
                        "with a 'move' operation to a clear pose (it starts with the "
                        "on-axis lift-off), or by re-ordering so its last weld is one "
                        "it can leave"
                    )
                elif not (standing.get("collision_free", True) or standing.get("error")):
                    why = (
                        "the arms END this program inside the cell's warning band "
                        f"where they STAND ({standing.get('min_clearance_m')} m apart, "
                        f"warning {standing.get('warning_distance_m')} m) — before any "
                        "home motion exists. Nothing a return can do fixes that: the "
                        "monitor measures the pair, not the mover, so it E-STOPS "
                        "whichever arm moves first, and homing them one at a time "
                        "cannot separate poses they are already in. Recover by moving "
                        "the arms' LAST operations apart (re-order or re-allocate them, "
                        "or add a reposition from find_interlock_set as a final "
                        "operation on one arm)"
                    )
                else:
                    why = (
                        "the arms' home legs enter the cell's warning band together, "
                        "and no order homes them one at a time clear of each other "
                        "either — the runtime monitor would E-STOP on the last motion "
                        "of the program. Recover by staging the arms apart BEFORE they "
                        "home (find_interlock_set with these arms, then add that "
                        "reposition as an ordinary operation at the end of each arm's "
                        "sequence)"
                    )
                return {
                    "success": False,
                    "error": ("the end-of-program return to HOME cannot be executed "
                              f"safely: {why}, or set \"home_at_end\": false and plan "
                              "the returns as ordinary operations so they go through "
                              "the normal validation."),
                    "home_step": home_report,
                }
            step_nodes.append(home_node)

    # FINAL check of EVERY leg the tree will execute — weld legs, agent-requested
    # 'move' / park / stage legs, single-arm steps no inter-arm gate saw, the
    # planned home legs and the blind home's straight line: joint limits, and the
    # arm against ITSELF along the interpolated path (planner_backend._leg_refusal). Legs from another backend or a hand-made
    # cache entry are caught here, never trusted (umarell issues 13 and 14).
    _mark("before_final_audit")
    leg_hits = _leg_audit(cell_id, [], home_node, per_robot_ops)
    _mark("final_leg_audit")
    if leg_hits:
        return _leg_refusal_result(leg_hits, step_reports, home_report)

    root = build_behaviour_tree_node.func(node_type="Sequence", children=step_nodes)
    # Convert once, here: the builders above work in the internal shape, and
    # to_btcpp lifts the whole tree to BehaviorTree.CPP v4 — moving joint arrays
    # onto the blackboard and emitting the XML alongside.
    tree = to_btcpp({
        "tree_id":  new_tree_id(cell_id),
        "cell_id":  cell_id,
        "root":     root,
        "metadata": {"steps": len(steps), "operations": op_count,
                     # Every weld's pose travels with the tree: the cell's home
                     # reset (another process) needs them to lift off on-axis.
                     "weld_poses": weld_pose_record(cell_id, list(per_robot_ops))},
    })
    validation = validate_behaviour_tree.func(tree)
    _mark("build_and_validate_tree")
    out = {
        "success":        bool(validation.get("valid", False)),
        "behaviour_tree": tree,
        "validation":     validation,
    }
    if step_reports:
        # Say how each multi-arm step was gated — including any the gate serialised,
        # which changes the cycle time the plan was costed with.
        out["step_gates"] = step_reports
        serialised = [r["step"] for r in step_reports if r.get("mode") == "SEQUENTIAL"]
        if serialised:
            out["serialised_steps"] = serialised
    if home_report is not None:
        # Say how the return home was planned and gated — including any arm left
        # to the executor's blind home, which no gate can cover.
        out["home_step"] = home_report
    if merged_moves:
        out["merged_moves"] = merged_moves
    if out["success"]:
        # The cost of what was just accepted — in front of the agent every time.
        try:
            from agents.trajectory.tools.program_eval import program_cost
            out["program_cost"] = program_cost(cell_id, plan, step_reports,
                                               home_node, home_report)
        except Exception as exc:  # noqa: BLE001 — a report, never a reason to refuse
            out["program_cost"] = {"error": f"{type(exc).__name__}: {exc}"}
    _mark("program_cost")
    from agents.trajectory.tools.coordination import current_planning_mode
    classic = current_planning_mode() == "classic"
    serial = [] if classic else _serial_steps(steps)
    if serial:
        # Steps where an arm that still has welds ahead of it stands idle. Each is
        # parallel time lost; the planner must remove them (a different weld
        # point, or an interlock position) — serialising is not a resolution.
        out["serial_steps"] = serial
        out["serial_warning"] = (
            f"{len(serial)} step(s) leave an arm idle while it still has welds to do. "
            "Remove each by changing the weld point (give that arm a pending weld "
            "clear of the other, closest to its own base) or by an interlock position "
            "that minimises execution time, then re-assemble.")
    idle = [] if classic else _idle_arms(cell_id, steps)
    if idle:
        # Not a refusal (a cell may legitimately give one arm nothing), but never
        # silent: a program that parks an arm the whole time is a cycle-time and
        # allocation decision the operator must see.
        out["idle_arms"] = idle
        out["warning"] = (
            f"arm(s) {', '.join(idle)} do no work in this program. Unreachable welds "
            "do not justify that: if the arms block each other, retreat the idle arm "
            "with a planned 'move' operation between their steps instead of "
            "consolidating every weld onto one arm.")
    if out["success"] and current_planning_mode() == "stepwise":
        _mark("warnings")
        alt = _barrier_free(cell_id, plan, steps, out)
        _mark("barrier_free")
        if alt is not None:
            return alt
    return out


def _is_home_move(r: dict) -> bool:
    """A planned return home the agent made itself (``home_at_end: false``)."""
    return (str(r.get("operation_type") or "") == "move"
            and "HOME" in str(r.get("operation_id") or "").upper())


def _split_home_tail(steps: list) -> "tuple[list, list]":
    """``(work_steps, home_tail)``: the trailing steps in which every arm is going
    home. Those run after all the work, as steps — not inside the per-arm schedule,
    where arms already home are permanent obstacles to the ones still returning
    (bt_1793792763: "robot1 cannot start robot1_HOME2: robot4 is in its way")."""
    n = 0
    for st in reversed(steps):
        robots = [r for r in (st.get("robots") or []) if r.get("operation_id")]
        if robots and all(_is_home_move(r) for r in robots):
            n += 1
        else:
            break
    return steps[:len(steps) - n], steps[len(steps) - n:]


def _gated_steps(cell_id: str, steps: list, at_pose: dict):
    """Build ``steps`` exactly as the assembler builds a step — one
    ExecuteTrajectory per arm (its delay, hold), the inter-arm gate against the
    step's arms AND every other arm where it stands (``at_pose``), ordered or
    serialised by the gate's verdict.

    ``(nodes, reports, seconds)``, or ``(None, reports, None)`` when a leg is not
    cached or the gate refuses a step (``reports[-1]`` says which)."""
    from agents.trajectory.tools.trajectory_cache import get_cached_trajectory
    nodes, reports, total = [], [], 0.0
    at_pose = {ns: list(q) for ns, q in (at_pose or {}).items()}
    for si, st in enumerate(steps):
        mode = str(st.get("mode", "PARALLEL")).upper()
        legs, exec_nodes, delays = {}, [], {}
        for r in st.get("robots") or []:
            ns, op = r.get("robot_ns"), r.get("operation_id")
            traj = get_cached_trajectory(cell_id, ns, op)
            if not traj:
                reports.append({"step": si + 1, "mode": None,
                                "error": f"no cached trajectory for {ns} {op}"})
                return None, reports, None
            legs[ns] = traj
            delays[ns] = float(r.get("start_delay_s", 0.0) or 0.0)
            exec_nodes.append(build_behaviour_tree_node.func(
                node_type="ExecuteTrajectory", robot_ns=ns, trajectory=traj,
                operation_id=op, operation_type=r.get("operation_type", "move"),
                start_delay_s=delays[ns],
                hold_duration_s=float(r.get("hold_duration_s", 0.5)), velocity_scale=1.0))
        if not exec_nodes:
            continue
        bystanders = {ns: q for ns, q in at_pose.items() if ns not in legs}
        if len(legs) > 1 or bystanders:
            report = _gate_multi_arm_step(cell_id, legs,
                                          allow_parallel=not mode.startswith("SEQ"),
                                          delays=delays, bystanders=bystanders)
        else:
            report = {"mode": mode, "gated": False, "note": "single arm — no pair to check"}
        report["step"] = si + 1
        reports.append(report)
        if report.get("mode") is None:
            return None, reports, None
        nodes.append(_order_step_nodes(exec_nodes, report))
        durs = {ns: float(t[-1].get("time_s", 0.0)) + delays.get(ns, 0.0)
                for ns, t in legs.items()}
        total += (max(durs.values()) if report.get("mode") == "PARALLEL"
                  else sum(durs.values()))
        for ns, t in legs.items():
            at_pose[ns] = list(t[-1].get("positions") or [])
    return nodes, reports, round(total, 2)


#: Smallest estimated saving for which the barrier-free tree replaces the stepwise one.
BARRIER_FREE_MIN_GAIN_S = 1.0


def _barrier_free(cell_id: str, plan: dict, steps: list, stepwise: dict) -> Optional[dict]:
    """The same program without its step barriers, when that is faster and valid.

    A stepwise tree runs each step as a Parallel node: every arm waits at the end
    of the step for the SLOWEST arm, whether or not anything conflicts. On
    welding_cell_2_small (bt_1793277501) that cost the 104.4 s cycle 22.8 s of imbalance
    and left an arm idle with work to do in 4 of 11 steps.

    The coordinated scheduler (``coordinated_tree``) removes exactly that: each
    arm runs its own operations back to back, and waits only where its motion
    really conflicts with another arm's — at an event gate ("until robot2 has
    COMPLETED W18"), not a clock time, so the separation holds at any speed. It is
    given the SAME cached trajectories in the same per-arm order, so no motion
    changes, only when each starts. Its tree is accepted only when it assembles
    and validates through ``assemble_coordinated_tree`` — the coordinated mode's
    own conflict model, leg audit, approach check and gated return HOME (reused,
    not re-planned: plan_memo) — and is estimated at least
    :data:`BARRIER_FREE_MIN_GAIN_S` faster. Stepwise mode only (classic keeps its
    29-Aug behaviour; coordinated assembles this way already). Otherwise the stepwise tree stands,
    and the result says why. ``plan["barrier_free"] = false`` or
    ``STEPWISE_BARRIER_FREE=0`` keeps the stepwise tree.
    """
    import os
    if plan.get("barrier_free") is False or os.environ.get(
            "STEPWISE_BARRIER_FREE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    sequences: dict = {}
    holds: dict = {}
    kinds: dict = {}
    # The agent's own returns home (home_at_end: false) stay STEPS after the work.
    work_steps, home_tail = _split_home_tail(steps)
    for st in work_steps:
        for r in st.get("robots") or []:
            ns, op = r.get("robot_ns"), r.get("operation_id")
            if not ns or not op:
                continue
            sequences.setdefault(ns, []).append(str(op))
            if r.get("hold_duration_s") is not None:
                holds[str(op)] = float(r["hold_duration_s"])
            if r.get("operation_type"):
                kinds[str(op)] = str(r["operation_type"])
    if len(sequences) < 2:
        return None
    cost = stepwise.get("program_cost") or {}
    cycle = cost.get("cycle_time_s")
    home_s = float(cost.get("home_s") or 0.0)
    note: dict = {"used": False, "stepwise_cycle_time_s": cycle}
    try:
        from agents.trajectory.tools.coordinated_tree import assemble_coordinated
        alt = assemble_coordinated(cell_id, sequences, hold_duration_s=holds or None,
                                   operation_types=kinds or None,
                                   home_at_end=plan.get("home_at_end", True),
                                   tail_steps=home_tail or None)
    except Exception as exc:  # noqa: BLE001 — the stepwise tree stands
        note["reason"] = f"coordination failed: {type(exc).__name__}: {exc}"
        stepwise["barrier_free"] = note
        return None
    makespan = alt.get("makespan_estimate_s")
    if not alt.get("success") or not (alt.get("validation") or {}).get("valid"):
        note["reason"] = (alt.get("error") or "the barrier-free tree did not validate")
        if alt.get("deadlock"):
            note["deadlock"] = alt["deadlock"]
        stepwise["barrier_free"] = note
        return None
    tail_s = float(alt.get("tail_s") or 0.0)
    est = None if makespan is None else round(float(makespan) + tail_s + home_s, 2)
    if home_tail:
        note["home_tail_steps"] = len(home_tail)
    note["coordinated_cycle_time_s"] = est
    if cycle is None or est is None or est > float(cycle) - BARRIER_FREE_MIN_GAIN_S:
        note["reason"] = "not faster than the stepwise tree"
        stepwise["barrier_free"] = note
        return None
    saved = round(float(cycle) - est, 2)
    alt["barrier_free"] = {
        "used": True, "stepwise_cycle_time_s": cycle, "coordinated_cycle_time_s": est,
        "saved_s": saved, "stepwise_tree_id": (stepwise.get("behaviour_tree") or {}).get("tree_id"),
        "note": ("the program's steps were replaced by per-arm sequences: each arm "
                 "runs its operations back to back and waits only at a WaitForRobot "
                 "gate where its motion conflicts with another arm's. Same "
                 "trajectories, same order per arm; return this tree."),
    }
    alt["program_cost"] = {"cycle_time_s": est, "home_s": home_s, "mode": "coordinated",
                           "stepwise_cycle_time_s": cycle}
    alt["step_gates"] = stepwise.get("step_gates")
    return alt


def _merge_moves(step_specs: list) -> "tuple[list, list]":
    """Fold consecutive single-arm ``move`` steps of the same arm into one.

    Only plain moves (no interlock, no start delay, no dependency) of one arm with
    nothing else in their steps, each starting where the previous ended. Returns
    ``(specs, merges)`` — each merge ``{"robot_ns", "operation_ids", "saved_s"}``."""
    def plain(spec):
        if len(spec["nodes"]) != 1 or any(spec["delays"].values()):
            return None
        n = spec["nodes"][0]
        if n.get("operation_type") != "move" or n.get("has_interlock") \
                or n.get("depends_on_robots") or n.get("depends_on_positions"):
            return None
        return n

    out: list = []
    merges: list = []
    for spec in step_specs:
        prev = out[-1] if out else None
        a, b = (plain(prev) if prev else None), plain(spec)
        if a is not None and b is not None and a.get("robot_ns") == b.get("robot_ns"):
            ta, tb = a["precomputed_trajectory"], b["precomputed_trajectory"]
            gap = _seam_gap(ta, tb)
            if gap is not None and gap <= 1e-3:
                t0 = float(ta[-1]["time_s"])
                leg = list(ta) + [{**p, "time_s": round(t0 + float(p["time_s"]), 4)}
                                  for p in tb[1:]]
                ns = a["robot_ns"]
                ops = a.get("merged_operation_ids") or [a.get("operation_id")]
                ops = ops + [b.get("operation_id")]
                node = {**a, "precomputed_trajectory": leg,
                        "operation_id": "+".join(str(o) for o in ops),
                        "merged_operation_ids": ops,
                        "hold_duration_s": b.get("hold_duration_s", 0.5)}
                out[-1] = {**prev, "nodes": [node], "legs": {ns: leg},
                           "ops": {ns: node["operation_id"]}}
                if merges and merges[-1]["operation_ids"] == ops[:-1]:
                    merges[-1].update(operation_ids=ops)
                    merges[-1]["saved_s"] = round(merges[-1]["saved_s"]
                                                  + float(a.get("hold_duration_s", 0.5)), 2)
                else:
                    merges.append({"robot_ns": ns, "operation_ids": ops,
                                   "saved_s": round(float(a.get("hold_duration_s", 0.5)), 2)})
                continue
        out.append(spec)
    return out, merges


def _leg_refusal_result(leg_hits: list, step_reports: Optional[list] = None,
                        home_report: Optional[dict] = None) -> dict:
    return {
        "success": False,
        "error": ("these legs cannot be executed as planned: they leave their arm's "
                  "joint limits, or drive the arm into its OWN collision hulls "
                  "somewhere along the executed path. No ordering or delay fixes that: "
                  "re-plan each listed operation (another arm via reallocate_operation, "
                  "a different approach orientation, or a via pose); for a 'home' leg, "
                  "add a reposition as the arm's last operation or set \"home_at_end\": "
                  "false and plan the return as an operation."),
        "leg_refusals": leg_hits,
        **({"step_gates": step_reports} if step_reports else {}),
        **({"home_step": home_report} if home_report is not None else {}),
    }


def _approach_refusal(cell_id: str, robot_ns: str, operation_id, leg) -> Optional[dict]:
    """A weld leg whose working stroke does not match what its operation
    specified (``validate_approach``: the tool axis within the tool's tolerance,
    the point reached, a measurable on-axis entry) — the platform's check, so the
    agent need not run it per weld. None when it matches, when the operation has
    no constrained stroke, or when the check cannot run (geometry stack absent)."""
    if not leg or not operation_id:
        return None
    from agents.trajectory.tools.trajectory_cache import op_spec
    spec = op_spec(cell_id, robot_ns, operation_id)
    if not (spec and spec.get("approach") and spec.get("constrain_orientation")
            and spec.get("orientation") and spec.get("target")):
        return None
    try:
        from agents.trajectory.tools.pinocchio import validate_approach
    except Exception:  # noqa: BLE001 — no geometry stack here
        return None
    check = getattr(validate_approach, "func", validate_approach)
    v = check(cell_id=cell_id, robot_ns=robot_ns,
              expected_orientation=spec["orientation"], trajectory=leg,
              target_position=spec["target"], approach_vector=spec["approach_vector"])
    if v.get("matches") or str(v.get("reason", "")).startswith("validation error"):
        return None
    return {"robot_ns": robot_ns, "reason_code": "approach_mismatch",
            "approach_check": {k: v.get(k) for k in (
                "tool_axis_error_deg", "position_error_m",
                "approach_direction_error_deg", "entry_travel_m", "joint_limits_ok")},
            "error": f"{robot_ns}/{operation_id}: the working stroke does not match its "
                     f"specification: {v.get('reason')}"}


def _leg_audit(cell_id: str, step_specs: list, home_node: Optional[dict],
               per_robot_ops: dict) -> list:
    """Every leg of an assembled tree through the platform's leg check
    (``planner_backend._leg_refusal``). Returns the refusals, each naming its step,
    robot and operation."""
    from agents.trajectory.tools.planner_backend import _leg_refusal, home_config
    hits: list = []
    for spec in step_specs:
        for ns, leg in spec["legs"].items():
            op = (spec.get("ops") or {}).get(ns)
            why = _leg_refusal(cell_id, ns, leg) if leg else None
            if why is None:
                why = _approach_refusal(cell_id, ns, op, leg)
            if why is not None:
                hits.append({"step": spec["index"], "operation_id": op, **why})
    homes: list = []

    def walk(n):
        if not isinstance(n, dict):
            return
        if n.get("type") == "HOME":
            homes.append(n)
        for c in n.get("children") or []:
            walk(c)
    walk(home_node)
    for n in homes:
        ns = n.get("robot_ns")
        leg = n.get("precomputed_trajectory")
        if not leg:
            # Blind home: the executor interpolates straight to HOME from where
            # the arm stands. That line is a leg too.
            start = _last_op_end_joints(cell_id, ns, per_robot_ops.get(ns) or [])
            if not start:
                continue
            leg = [{"positions": list(start), "time_s": 0.0},
                   {"positions": list(home_config(cell_id, ns, _HOME_JOINTS)),
                    "time_s": 1.0}]
        why = _leg_refusal(cell_id, ns, leg)
        if why is not None:
            hits.append({"step": "home", "operation_id": "home",
                         "blind": not n.get("precomputed_trajectory"), **why})
    return hits


def _serial_steps(steps: list) -> list:
    """``[{"step", "working", "idle_with_work"}]`` for every step in which an arm
    that appears in a LATER step does nothing — i.e. work that could have been
    parallel with this step but was left for later."""
    arms_at = [[r.get("robot_ns") for r in (s.get("robots") or []) if r.get("robot_ns")]
               for s in steps]
    out = []
    for i, working in enumerate(arms_at):
        later = {ns for step in arms_at[i + 1:] for ns in step}
        idle = sorted(later - set(working))
        if idle:
            out.append({"step": i + 1, "working": sorted(set(working)),
                        "idle_with_work": idle})
    return out


def _idle_arms(cell_id: str, steps: list) -> list:
    """Arms the cell declares that no step of the plan uses."""
    try:
        from agents.trajectory.tools.planning_tools import _cell_robots
        declared = set(_cell_robots(cell_id))
    except Exception:  # noqa: BLE001 — a reporting aid, never a reason to fail
        return []
    used = {r.get("robot_ns") for s in steps for r in (s.get("robots") or [])}
    return sorted(declared - used)


@tool
def export_behaviour_tree_xml(tree: dict, include_model: bool = True) -> dict:
    """
    Serialise a Behaviour Tree as BehaviorTree.CPP v4 XML.

    ``assemble_behaviour_tree`` already returns the XML in the tree's ``xml``
    field, so you rarely need this. Use it to re-emit the XML after editing a
    tree, or to render a legacy-dialect tree (EXECUTE/PARALLEL/…) in the standard
    format — the conversion is applied automatically.

    Args:
        tree:          A tree dict (BT.CPP or legacy) with a ``root``.
        include_model: Emit the ``<TreeNodesModel>`` section describing the custom
                       nodes' ports. Groot2 needs it to draw the tree; a C++
                       factory ignores it. Default true.

    Returns:
        {"success": true, "xml": "<root BTCPP_format=\"4\">…",
         "blackboard": {...}, "main_tree_to_execute": "MainTree"}

    The ``blackboard`` must be loaded before ticking the tree: bulky ports
    (trajectories, interlock targets, dependency lists) are referenced from the
    XML as ``{key}`` rather than inlined, so the structure stays readable.
    """
    if not isinstance(tree, dict) or tree.get("root") is None:
        return {"success": False, "error": "tree must be a dict with a 'root' node"}
    try:
        converted = tree if tree.get("BTCPP_format") else to_btcpp(tree)
        return {
            "success":              True,
            "xml":                  to_xml(converted, include_model=include_model),
            "blackboard":           converted.get("blackboard") or {},
            "main_tree_to_execute": converted.get("main_tree_to_execute"),
        }
    except (TypeError, ValueError) as exc:
        return {"success": False, "error": str(exc)}
