"""Re-plan ONE operation of an existing BehaviourTree — not the whole program.

An operator looking at a finished plan often wants to change one weld: a longer
standoff, a different lean, a slower stroke. Re-running the planner for that
re-plans every weld, takes minutes of model turns, and can come back with a
different program. This module edits the tree instead, deterministically and
without the model:

  1. The tree's step plan is recovered from its structure (each top-level child
     is a step: ``Parallel`` → PARALLEL, ``Sequence`` → SEQUENTIAL, a ``Delay``
     wrapper → ``start_delay_s``; the final ``MoveHome`` step → ``home_at_end``),
     and every leg's trajectory is loaded into the planner's cache.
  2. The chosen operation is re-planned with the SAME motion request the run used
     (read from the run's plan trace; the task file is the fallback), with the
     operator's adjustments applied, chained from where that arm's previous leg
     ends.
  3. If the new leg ends somewhere else (a different IK at the weld), the arm's
     following legs are re-planned in turn until the chain closes — each with its
     own original request — so the arm never jumps between welds.
  4. The edited plan goes through ``assemble_behaviour_tree`` — the same inter-arm
     gate, chain check and planned return home every tree gets — and is saved as a
     NEW tree whose metadata names the tree it was derived from. The original is
     never modified.

The request names a ``backend`` and a ``mode``, like ``plan_trajectories``
(``agents/trajectory/plan_selection.py`` resolves them):

* the **backend** calculates the legs. With ``ompl-compound`` a PARALLEL step is
  re-planned as ONE compound query — every arm of the step, from where each arm's
  previous leg ends — so the step keeps its joint timing; the walk then goes on
  step by step, re-planning each step that holds an arm whose chain is broken.
* the **mode** decides the new tree's shape: ``classic`` / ``stepwise`` rebuild
  the steps with ``assemble_behaviour_tree``; ``coordinated`` builds one gated
  Sequence per arm with ``assemble_coordinated_tree`` — from a step-shaped tree
  or a coordinated one (its per-arm legs are recovered and its gates derived
  again).

Refused rather than guessed: an unknown operation, an operation that appears
twice, a tree whose legs are summaries, a coordinated tree asked to become a
step tree (it has no steps to recover), and step trees with interlock gates.
"""
from __future__ import annotations

import copy
import time
from typing import Optional

#: What an operator may change about the operation's motion request.
ADJUSTABLE = frozenset({
    "target", "approach_vector", "orientation", "constrain_orientation", "approach",
    "approach_offset_m", "retreat_offset_m", "approach_velocity_scale",
    "velocity_scale",
})
#: Per-joint seam tolerance — the assembler's chain check uses the same.
CHAIN_TOL_RAD = 1e-3
#: coordinated_tree's piece separator ("W05@0-1") and early-home suffix
#: ("robot2_HOME"). Mirrored, not imported: that module needs LangChain, and
#: reading a tree must not. tests/trajectory/test_replan_backend_and_mode.py
#: pins them equal.
COORD_SEP = "@"
COORD_EARLY_HOME = "_HOME"


class EditError(ValueError):
    """The edit cannot be made as asked."""


# ── reading the tree ─────────────────────────────────────────────────────────

def _leaves(node: dict, blackboard: dict, delay_s: float = 0.0) -> list[dict]:
    """``[{robot_ns, operation_id, operation_type, …, "_traj": [...]}]`` for every
    ExecuteTrajectory under ``node``, in tick order."""
    from agents.trajectory.tools import btcpp
    kind = node.get("type")
    if kind == "Delay":
        ms = (node.get("ports") or {}).get("delay_msec") or 0
        out: list = []
        for child in btcpp.children(node):
            out += _leaves(child, blackboard, delay_s + float(ms) / 1000.0)
        return out
    if kind in ("Sequence", "SequenceWithMemory", "Parallel"):
        out = []
        for child in btcpp.children(node):
            out += _leaves(child, blackboard, delay_s)
        return out
    if kind == "ExecuteTrajectory":
        ports = node.get("ports") or {}
        entry = {k: ports[k] for k in ("robot_ns", "operation_id", "operation_type",
                                        "hold_duration_s", "velocity_scale",
                                        "has_interlock") if k in ports}
        if delay_s:
            entry["start_delay_s"] = round(delay_s, 3)
        entry["_traj"] = btcpp.trajectory_of(node, blackboard)
        return [entry]
    if kind == "MoveHome":
        return []
    raise EditError(f"the tree contains a {kind!r} node; trees with interlock or "
                    "completion gates cannot be edited one operation at a time yet")


def read_tree(tree: dict) -> dict:
    """Recover ``{"steps", "home_at_end", "order"}`` from an assembled tree.

    ``order`` is each arm's legs in execution order: ``{robot: [(op_id, traj)]}``.
    """
    blackboard = tree.get("blackboard") or {}
    root = tree.get("root") or {}
    steps: list = []
    home = False
    order: dict = {}
    for child in (root.get("children") or []):
        kinds = _kinds(child)
        if "MoveHome" in kinds and "ExecuteTrajectory" not in kinds:
            home = True
            continue
        legs = _leaves(child, blackboard)
        if not legs:
            continue
        mode = "PARALLEL" if child.get("type") == "Parallel" else "SEQUENTIAL"
        for leg in legs:
            traj = leg.get("_traj")
            if not isinstance(traj, list) or len(traj) < 2 \
                    or not isinstance(traj[0], dict) or "positions" not in traj[0]:
                raise EditError(f"leg {leg.get('operation_id')} carries no waypoints "
                                "(a summarised plan) — it cannot be edited")
            order.setdefault(leg["robot_ns"], []).append((leg["operation_id"], traj))
        steps.append({"mode": mode,
                      "robots": [{k: v for k, v in leg.items() if k != "_traj"}
                                 for leg in legs]})
    if not steps:
        raise EditError("the tree has no trajectory steps")
    return {"steps": steps, "home_at_end": home, "order": order}


def read_coordinated_tree(tree: dict) -> dict:
    """Recover ``{"order", "holds", "types", "home_at_end"}`` from a coordinated tree.

    A coordinated tree is ``Sequence[Parallel[one Sequence per arm], home?]``.
    An arm's operation may be split into pieces around its completion gates
    (``W05@0-1``, ``W05@2-3``); the pieces are joined back into the one leg the
    coordinator cut them from. ``WaitForRobot`` gates are dropped — the
    coordinator derives them again — and so are early returns home
    (``robot2_HOME``), which it plans again when it needs them.
    """
    from agents.trajectory.tools import btcpp
    SEP, EARLY_HOME = COORD_SEP, COORD_EARLY_HOME
    blackboard = tree.get("blackboard") or {}
    children = (tree.get("root") or {}).get("children") or []
    if not children or children[0].get("type") != "Parallel":
        raise EditError("not a coordinated tree (no Parallel of per-arm sequences)")
    order, holds, types = {}, {}, {}
    for arm_seq in btcpp.children(children[0]):
        for node in btcpp.children(arm_seq):
            if node.get("type") in ("WaitForRobot", "CheckSafety"):
                continue
            if node.get("type") != "ExecuteTrajectory":
                raise EditError(f"unexpected {node.get('type')!r} in a coordinated arm")
            ports = node.get("ports") or {}
            ns, nid = ports.get("robot_ns"), str(ports.get("operation_id"))
            op = nid.split(SEP)[0]
            if op.endswith(EARLY_HOME):
                continue
            piece = btcpp.trajectory_of(node, blackboard)
            if not isinstance(piece, list) or len(piece) < 2 \
                    or "positions" not in (piece[0] or {}):
                raise EditError(f"leg {nid} carries no waypoints — it cannot be edited")
            legs = order.setdefault(ns, [])
            if legs and legs[-1][0] == op:
                prev = legs[-1][1]
                t0 = float(prev[-1]["time_s"]) - float(piece[0]["time_s"])
                legs[-1] = (op, prev + [{**p, "time_s": round(float(p["time_s"]) + t0, 4)}
                                        for p in piece[1:]])
            else:
                legs.append((op, list(piece)))
            # Only an operation's LAST piece carries its hold; pieces come in order.
            holds[op] = float(ports.get("hold_duration_s", 0.5) or 0.0)
            types[op] = ports.get("operation_type", "weld")
    if not order:
        raise EditError("the tree has no trajectory legs")
    return {"order": order, "holds": holds, "types": types,
            "home_at_end": len(children) > 1}


def _kinds(node: dict) -> set:
    out = {node.get("type")}
    for child in node.get("children") or []:
        out |= _kinds(child)
    return out


# ── the motion requests the run used ─────────────────────────────────────────

def requests_from_trace(trace: Optional[dict]) -> dict:
    """``{(robot, op_id): request}`` — the last request each leg was planned with
    successfully, with the call's ``velocity_scale`` folded in."""
    found: dict = {}
    for d in ((trace or {}).get("decisions") or []):
        if d.get("tool") != "run_trajectory_planner":
            continue
        args = d.get("args") or {}
        result = d.get("result") or {}
        failed = {f.get("operation_id") for f in (result.get("failures") or [])
                  if isinstance(f, dict)}
        if result.get("ok") is False and not failed:
            continue
        batches = [args.get("motion_requests") or []] + list(args.get("steps") or [])
        for reqs in batches:
            for r in reqs or []:
                if not isinstance(r, dict) or r.get("operation_id") in failed:
                    continue
                req = copy.deepcopy(r)
                if args.get("velocity_scale") is not None:
                    req.setdefault("velocity_scale", args["velocity_scale"])
                found[(req.get("robot_ns"), req.get("operation_id"))] = req
    return found


def request_from_task(op: dict, robot_ns: str) -> dict:
    """A weld request built from the task file — the fallback with no trace."""
    pos = op.get("position") or [0.0, 0.0, 0.0]
    ori = op.get("weld_orientation") or op.get("orientation")
    req = {"robot_ns": robot_ns, "operation_id": op.get("id"),
           "operation_type": op.get("type", "weld"),
           "target": {"x": pos[0], "y": pos[1], "z": pos[2]},
           "approach_vector": op.get("approach_vector") or [0.0, 0.0, -1.0],
           "approach": True}
    if ori:
        req.update(orientation=ori, constrain_orientation=True)
    return req


def _gap(a: list, b: list) -> float:
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def _end(traj: list) -> list:
    return list(traj[-1].get("positions") or [])


# ── the edit ─────────────────────────────────────────────────────────────────

#: Modes whose tree is a list of steps; "coordinated" is one gated Sequence per arm.
STEP_MODES = ("classic", "stepwise")


def _is_coordinated(tree: dict) -> bool:
    return ((tree.get("metadata") or {}).get("mode") == "coordinated"
            or "WaitForRobot" in _kinds(tree.get("root") or {}))


def planning_units(order: dict, steps: Optional[list], joint: bool) -> list[list]:
    """The legs in execution order, grouped into what is planned in ONE call.

    ``[[(robot, leg_index), ...], ...]``. A step-shaped tree yields its steps in
    order; with ``joint`` (the ompl-compound backend) a PARALLEL step's legs are
    one unit, so they are planned — and timed — together; otherwise every leg is
    its own unit. A coordinated tree (``steps`` None) has no steps: each arm's
    legs are units of one, arm after arm.
    """
    if steps is None:
        return [[(ns, k)] for ns, legs in order.items() for k in range(len(legs))]
    seen: dict = {}
    units: list = []
    for step in steps:
        legs = []
        for r in step.get("robots") or []:
            ns = r["robot_ns"]
            legs.append((ns, seen.get(ns, 0)))
            seen[ns] = seen.get(ns, 0) + 1
        if joint and step.get("mode") == "PARALLEL" and len(legs) > 1:
            units.append(legs)
        else:
            units.extend([leg] for leg in legs)
    return units


def _plans_steps_jointly(backend) -> bool:
    """The backend declares that a PARALLEL step is one motion (``BackendSpec.joint_step``).

    An unknown or unloadable backend is per-leg — the shape every backend can
    answer — and dispatch reports what is wrong with it.
    """
    try:
        from agents.trajectory.tools.planner_backend import backend_spec
        return bool(backend_spec(backend).joint_step)
    except Exception:  # noqa: BLE001
        return backend == "ompl-compound"


def replan_operation(cell_id: str, tree_id: str, operation_id: str,
                     adjustments: Optional[dict] = None,
                     operation: Optional[dict] = None,
                     save: bool = True,
                     backend: str = "ompl",
                     mode: str = "classic",
                     backend_options: Optional[dict] = None) -> dict:
    """:func:`_replan_operation` holding the cell (one planning run per cell — it
    resets the cell's caches, like a plan does). Busy: :class:`EditError`."""
    from agents.trajectory import cell_lock
    holder = f"replan_{tree_id}_{operation_id}"
    if not cell_lock.acquire(cell_id, holder):
        busy = cell_lock.holder_of(cell_id) or {}
        raise EditError(f"cell '{cell_id}' is being planned by {busy.get('holder')}; "
                        "edit it when that run has finished")
    try:
        return _replan_operation(cell_id, tree_id, operation_id, adjustments=adjustments,
                                 operation=operation, save=save, backend=backend,
                                 mode=mode, backend_options=backend_options)
    finally:
        cell_lock.release(cell_id, holder)


def _replan_operation(cell_id: str, tree_id: str, operation_id: str,
                      adjustments: Optional[dict] = None,
                      operation: Optional[dict] = None,
                      save: bool = True,
                      backend: str = "ompl",
                      mode: str = "classic",
                      backend_options: Optional[dict] = None) -> dict:
    """Re-plan ``operation_id`` of plan ``tree_id`` and build a new tree from it.

    ``adjustments`` overrides fields of the operation's motion request (see
    :data:`ADJUSTABLE`); ``operation`` supplies the request outright when the run
    left no trace and the task file does not know the operation. ``backend``
    calculates the legs (``ompl-compound``: a PARALLEL step's legs together);
    ``mode`` decides the new tree's shape (see the MCP tool).
    """
    from agents.trajectory.tools import plan_store, trajectory_cache as tc
    from agents.trajectory.tools.behaviour_tree import assemble_behaviour_tree
    from agents.trajectory.tools.coordinated_tree import assemble_coordinated_tree
    from agents.trajectory.tools.planner_backend import (
        reset_backend_options, reset_planning_backend, set_backend_options,
        set_planning_backend,
    )
    from agents.trajectory.tools.planning_tools import run_trajectory_planner

    adjustments = dict(adjustments or {})
    bad = set(adjustments) - ADJUSTABLE
    if bad:
        raise EditError(f"cannot adjust {sorted(bad)} — adjustable: {sorted(ADJUSTABLE)}")

    tree = plan_store.get_plan(cell_id, tree_id)   # the tree; failed plans excluded
    if not isinstance(tree, dict) or "root" not in tree:
        raise EditError(f"no usable plan {tree_id!r} for cell {cell_id!r}")
    # The plan is stored (and its run traced) under the id it was requested as,
    # which can differ from the id the assembler stamped inside the tree.
    source_id = tree_id
    if _is_coordinated(tree):
        if mode in STEP_MODES:
            raise EditError(f"{source_id} is a coordinated tree (per-arm sequences with "
                            f"completion gates): it has no steps to rebuild in mode "
                            f"{mode!r} — re-plan it with mode='coordinated'")
        src = read_coordinated_tree(tree)
        steps, order, home = None, src["order"], src["home_at_end"]
        holds, types = src["holds"], src["types"]
    else:
        plan = read_tree(tree)
        steps, order, home = plan["steps"], plan["order"], plan["home_at_end"]
        holds = {r["operation_id"]: float(r.get("hold_duration_s", 0.5))
                 for s in steps for r in s["robots"]}
        types = {r["operation_id"]: r.get("operation_type", "weld")
                 for s in steps for r in s["robots"]}

    owners = [(ns, i) for ns, legs in order.items()
              for i, (op, _t) in enumerate(legs) if op == operation_id]
    if not owners:
        known = sorted(op for legs in order.values() for op, _t in legs)
        raise EditError(f"{operation_id!r} is not in {source_id} (it has {known})")
    if len(owners) > 1:
        raise EditError(f"{operation_id!r} appears {len(owners)} times in {source_id}")
    robot, index = owners[0]

    requests = requests_from_trace(_read_trace(cell_id, source_id))
    task_ops = _task_operations(cell_id)

    def request_for(ns: str, op_id: str) -> dict:
        req = requests.get((ns, op_id))
        if req is None and op_id == operation_id and operation:
            req = {**operation, "robot_ns": ns, "operation_id": op_id}
        if req is None and op_id in task_ops:
            req = request_from_task(task_ops[op_id], ns)
        if req is None:
            raise EditError(f"no motion request is known for {ns}/{op_id}: the "
                            "run left no trace of it and no task file defines it — "
                            "pass `operation` with its request")
        return copy.deepcopy(req)

    t0 = time.monotonic()
    tc.clear_trajectory_cache(cell_id)
    tc.reset_robot_configs(cell_id)
    for ns, arm_legs in order.items():
        for op_id, traj in arm_legs:
            tc.cache_trajectory(cell_id, ns, op_id, traj)
            _register_weld_pose(cell_id, ns, op_id, request_for, traj)
    # The poses the run itself recorded, when the tree carries them: exact, and
    # present even for a leg whose request left no trace.
    tc.load_weld_pose_record(cell_id, (tree.get("metadata") or {}).get("weld_poses"))

    units = planning_units(order, steps, joint=_plans_steps_jointly(backend))
    token = set_planning_backend(backend)
    options_token = set_backend_options(backend_options)
    replanned: list = []
    dirty: set = set()            # arms whose chain is broken entering the next unit
    last_end: dict = {}           # where each arm stands, as the walk goes
    started = False
    try:
        for unit in units:
            members = {ns for ns, _k in unit}
            is_edit = (robot, index) in unit
            started = started or is_edit
            if started and (is_edit or members & dirty):
                # Arms not in this unit stand where their latest leg left them —
                # the bystanders the planner and its step gate must see.
                for ns, q in last_end.items():
                    if ns not in members:
                        tc.set_last_config(cell_id, ns, q)
                reqs = []
                for ns, k in unit:
                    op_id, _old = order[ns][k]
                    req = request_for(ns, op_id)
                    if (ns, k) == (robot, index):
                        req.update(copy.deepcopy(adjustments))
                    start = _end(order[ns][k - 1][1]) if k > 0 else None
                    if req.get("current_joints") and start:
                        # The run pinned this leg's start: pin it to where the arm now is.
                        req["current_joints"] = start
                    else:
                        req.pop("current_joints", None)
                        if start:
                            tc.set_last_config(cell_id, ns, start)
                        if k > 0:
                            _set_lift_off(cell_id, ns, request_for(ns, order[ns][k - 1][0]))
                        else:
                            tc.clear_last_pose(cell_id, ns)
                    reqs.append(req)
                # One call, one transit speed: a joint step is timed as one vector.
                speeds = [v for v in (r.pop("velocity_scale", None) for r in reqs)
                          if v is not None]
                res = run_trajectory_planner.func(
                    cell_id=cell_id, motion_requests=reqs,
                    velocity_scale=min(speeds) if speeds else None)
                trajs = res.get("trajectories") or {}
                if res.get("failures") or any(ns not in trajs for ns, _k in unit):
                    return {"success": False,
                            "operation_id": ", ".join(order[ns][k][0] for ns, k in unit),
                            "robot_ns": robot, "error": (
                                f"re-planning {[order[ns][k][0] for ns, k in unit]} "
                                f"with {backend} failed"),
                            "failures": res.get("failures"),
                            "replanned": replanned, "derived_from": source_id}
                for ns, k in unit:
                    op_id, old = order[ns][k]
                    new = trajs[ns]
                    order[ns][k] = (op_id, new)
                    shift = _gap(_end(new), _end(old))
                    replanned.append({"operation_id": op_id, "robot_ns": ns,
                                      "waypoints": len(new),
                                      "duration_s": round(float(new[-1].get("time_s") or 0), 3),
                                      "end_shift_rad": round(shift, 5)})
                    if shift <= CHAIN_TOL_RAD:
                        dirty.discard(ns)   # the arm ends where it did: the rest chains
                    else:
                        dirty.add(ns)
                if not dirty:
                    break
            for ns, k in unit:
                last_end[ns] = _end(order[ns][k][1])
    finally:
        reset_backend_options(options_token)
        reset_planning_backend(token)

    if mode in STEP_MODES:
        assembled = assemble_behaviour_tree.func(
            cell_id=cell_id, plan={"steps": steps, "home_at_end": home})
    else:
        assembled = assemble_coordinated_tree.func(
            cell_id=cell_id,
            sequences={ns: [op for op, _t in legs] for ns, legs in order.items()},
            hold_duration_s=holds, operation_types=types, home_at_end=home)
    if not assembled.get("success"):
        return {"success": False, "error": assembled.get("error"),
                "blocked_steps": assembled.get("blocked_steps"),
                "replanned": replanned, "derived_from": source_id}
    tree = assembled["behaviour_tree"]
    meta = tree.setdefault("metadata", {})
    meta["edit"] = {
        "derived_from": source_id, "operation_id": operation_id,
        "robot_ns": robot, "adjustments": adjustments,
        "replanned": [r["operation_id"] for r in replanned],
    }
    # How the edit was calculated and built — beside the edit record, which
    # names WHAT was edited.
    meta["edit_planning"] = {"backend": backend, "mode": mode}
    saved = plan_store.save_plan(cell_id, tree) if save else None
    return {"success": True, "tree_id": tree.get("tree_id"), "saved": bool(saved),
            "derived_from": source_id, "robot_ns": robot,
            "replanned": replanned, "planning_time_s": round(time.monotonic() - t0, 3),
            "step_gates": assembled.get("step_gates"),
            "serialised_steps": assembled.get("serialised_steps")}


def _set_lift_off(cell_id: str, robot: str, prev_req: dict) -> None:
    """Make the next leg leave the previous weld on-axis, as the run did."""
    from agents.trajectory.tools import trajectory_cache as tc
    t = prev_req.get("target") or {}
    if prev_req.get("constrain_orientation") and prev_req.get("orientation") and t:
        tc.set_last_pose(cell_id, robot, [t.get("x"), t.get("y"), t.get("z")],
                         prev_req.get("approach_vector") or [0, 0, -1],
                         prev_req["orientation"])
    else:
        tc.clear_last_pose(cell_id, robot)


def _register_weld_pose(cell_id: str, robot: str, op_id: str, request_for,
                        traj: Optional[list] = None) -> None:
    """Record the seam a kept leg ends on, as its planner did when the run made it.

    ``reset_robot_configs`` forgets every weld pose, and the kept legs are only
    re-cached, not re-planned — so without this a re-planned leg that starts on one
    of their seams (pinned, or chained) could not see it and would leave it with a
    joint-space transit instead of the on-axis lift-off (umarell issue 15).
    """
    from agents.trajectory.tools import trajectory_cache as tc
    from agents.trajectory.tools.planner_backend import apply_tool_spin
    try:
        req = apply_tool_spin(request_for(robot, op_id))
    except EditError:
        return
    t = req.get("target") or {}
    if req.get("approach") and req.get("constrain_orientation") and req.get("orientation") and t:
        # The orientation the leg ENDS in (its planner may have turned the tool,
        # weld_orientation.select), else the request's.
        ori = None
        if traj:
            from agents.trajectory.tools.weld_orientation import tcp_orientation
            ori = tcp_orientation(cell_id, robot, (traj[-1] or {}).get("positions") or [])
        tc.set_weld_pose(cell_id, robot, op_id, [t.get("x"), t.get("y"), t.get("z")],
                         req.get("approach_vector") or [0, 0, -1], ori or req["orientation"],
                         stroke_m=tc.stroke_of(req))


def _read_trace(cell_id: str, tree_id: str) -> Optional[dict]:
    from utils import cell_store
    try:
        return cell_store.read_json(cell_id, cell_store.KIND_LOGS,
                                    f"plantrace_{tree_id}.json")
    except Exception:  # noqa: BLE001 — no trace: fall back to the task file
        return None


def _task_operations(cell_id: str) -> dict:
    """Every operation the cell's task files define, by id."""
    from utils import cell_store
    ops: dict = {}
    try:
        entries = cell_store.list_entries(cell_id, cell_store.KIND_TASKS, suffix=".json")
    except Exception:  # noqa: BLE001
        return ops
    for e in entries:
        try:
            prog = cell_store.read_json(cell_id, cell_store.KIND_TASKS, e["name"]) or {}
        except Exception:  # noqa: BLE001
            continue
        for op in prog.get("operations") or []:
            if isinstance(op, dict) and op.get("id") is not None:
                ops.setdefault(str(op["id"]), op)
    return ops
