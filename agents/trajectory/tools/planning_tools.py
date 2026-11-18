"""
Planning tools exposed to the LLM — backend-neutral LangChain ``@tool`` surface.

Everything the trajectory agent can call: run the MRTA pipeline, plan a step,
validate a parallel step for inter-robot collision, and find interlock/staging
poses. The LLM supplies strategy; these execute it deterministically.

Motion planning itself is NOT implemented here — ``run_trajectory_planner``
dispatches through ``planner_backend.dispatch_plan``, which resolves the
task-bound backend (``ompl`` in production, ``rrt`` legacy). Geometry comes from
``robot_kinematics``; scheduling from ``task_allocation``.

This surface used to live in ``rrt_planner``, so selecting the OMPL backend
still imported the legacy planner module to reach the validators. Now the tool
surface, the shared geometry and each backend are separate: the ``rrt`` module
is reachable only when a task explicitly selects that backend.

Validation note: ``validate_trajectory_set`` enforces the SAME swept volume
(links + ``tool_collision``) and the SAME threshold (the cell's
``warning_distance``) as the runtime collision monitor, so a plan that passes
here is not E-STOPPED on the floor.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from langchain_core.tools import tool

from agents.trajectory.tools import plan_cancellation
from agents.trajectory.tools.robot_kinematics import (
    MONITOR_RATE_HZ,
    _check_robot_pair_collision,
    _config_spheres,
    _fk_ee,
    _load_collision_config,
    _load_pinocchio_model,
    _load_robot_config,
    _load_robots_config,
    inter_robot_clearance,
    leg_static_breach,
    resolve_velocity_scale,
    validation_sample_count,
)
from agents.trajectory.tools.time_parameterization import _sample_at_time, _shortcut_path
from agents.trajectory.tools.task_allocation import (
    _least_movement_schedule,
    _reallocation_candidates,
    _regional_allocate,
    _thermal_spread,
    _tsp_2opt,
)
from agents.trajectory.tools.trajectory_cache import cache_planner_result
from utils import cell_layout


def _cell_robots(cell_id: str) -> dict:
    """``{robot_id: config}`` for every arm in the cell.

    Flattens both the legacy ``{robot1: {robot1: {..}}}`` layout and the
    shared-namespace ``{arms: {robot1: {..}, robot2: {..}}}`` one; both yield
    ``{robot1: cfg, robot2: cfg}``. The allocator and the re-allocator read the
    robot set through here so they cannot disagree about which arms exist.
    """
    cfg = _load_robots_config(cell_id)
    robots: dict = {}
    for _ns, ns_block in (cfg.get("robots") or {}).items():
        if isinstance(ns_block, dict):
            for rid, rcfg in ns_block.items():
                if isinstance(rcfg, dict):
                    robots[rid] = rcfg
    return robots


def _home_tcp(cell_id: str, robot_ids: list) -> dict:
    """``{robot_id: [x, y, z]}`` — the tool position at each arm's HOME pose."""
    out: dict = {}
    try:
        from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose
        from agents.trajectory.tools.pinocchio import run_fk
        _fk = getattr(run_fk, "func", run_fk)
    except Exception:
        return out
    for rid in robot_ids:
        try:
            r = _fk(cell_id=cell_id, robot_ns=rid,
                    joint_positions=list(home_pose(cell_id, rid, HOME_JOINTS_FALLBACK)))
            pos = r.get("ee_position") or r.get("position")
            if pos:
                out[rid] = [round(float(v), 3) for v in pos[:3]]
        except Exception:
            continue
    return out


def _servicability_check(cell_id: str):
    """A ``can_service(robot_id, operation)`` predicate for ``_regional_allocate``.

    Answers the one thing the allocator's score cannot: can this arm actually STAND
    at this weld — IK solves, and no arm link is inside a fixture. The score sees only
    base positions and approach vectors, so it can resolve a tie toward an arm that
    physically cannot be there; without this, that surfaces as a planner
    ``goal_in_collision`` two stages later and has to be repaired by a handover.

    It is a filter on the IMPOSSIBLE, so a quiet one: on ``welding_cell_2_small`` it
    currently vetoes nothing, every arm having a collision-free configuration at
    every weld. Do not read that as "it never fires" — read it as the cell being
    reachable. It fires on a weld genuinely out of an arm's usable envelope.

    It asks ``run_ik`` and reads the answer — it does NOT run its own collision check.
    "Is there a usable configuration at this pose" is a branch-selection question, and
    :func:`~agents.trajectory.tools.pinocchio.run_ik` owns it: it ranks its seeds
    free-branch-first and reports ``goal_collision_free``. A second collision model
    here would be a second opinion about the same pose, and two models of "can the arm
    be here" is exactly how the planner's and the monitor's collision models drifted
    apart before. So the filter's whole job is to relay one field.

    Deliberately GOAL-POSE ONLY. Whether a PATH exists is the planner's question, and
    the swept-volume question between two arms is the behaviour tree's gate; this
    stage has no trajectories and does not pretend to. It is a filter on the
    impossible, not a promise about the possible.

    Results are memoised per (robot, position): the caller asks about the same tie
    from several groups, and each miss costs an IK solve. Any failure to set the
    check up at all (no Pinocchio) returns None, and the allocator then behaves
    exactly as it did before this existed.

    ``run_ik`` is unwrapped from its ``@tool`` decorator HERE, at setup, so a wiring
    failure returns None rather than a predicate that raises per-operation. A raising
    predicate would have to be caught, and the only safe reading of a caught
    exception is "yes, serviceable" — a filter that silently passes everything, which
    is worse than no filter because it looks like one. Setup failing says so by
    absence. For the same reason a ``run_ik`` that reports no ``goal_collision_free``
    (no obstacle model loaded) means "no opinion", not "no".
    """
    try:
        import pinocchio  # noqa: F401
        from agents.trajectory.tools.pinocchio import run_ik
        _ik = getattr(run_ik, "func", run_ik)
    except Exception:
        return None

    cache: dict = {}

    def can_service(robot_ns: str, op: dict) -> bool:
        pos = op.get("position") or [0.0, 0.0, 0.0]
        key = (robot_ns, tuple(round(float(c), 6) for c in pos))
        if key not in cache:
            ik = _ik(cell_id=cell_id, robot_ns=robot_ns,
                     x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            cache[key] = bool(ik.get("success")) and ik.get("goal_collision_free", True)
        return cache[key]

    return can_service


@tool
def run_mrta_pipeline(
    cell_id: str,
    operations: list[dict],
    operation_type_default: str = "weld",
) -> dict:
    """
    Run the full 6-stage MRTA allocation and sequence optimisation pipeline.

    Stages:
      1. Regional allocation (approach-vector dot-product; contested welds cut off
         as a contiguous group and checked against the fixture with IK)
      2. TSP 2-opt sequence per robot (open path — the arm does not return to its
         first weld)
      3. Thermal spread (same-face consecutive gap enforcement)
      4. Parallel steps: with the geometry stack, composed for ALL arms from
         their own IK poses, each arm's tour walked in the direction (computed,
         not fixed) that makes the program most parallel; weld pairs two arms
         can never work at once are reported in ``contention`` / ``advice``.
         Without it, least-movement interleaving.
      5. Cycle-time balance ratio

    Works for ANY number of robots in the cell (1, 2, 3, ...) — every stage runs
    over the full robot set, and a totality gate refuses the plan if scheduling
    ever loses an allocated operation.

    Robot allocation is decided here (Stage 1), NOT by the caller. The run's locks
    (the operator's assignments and the task file's own ``assigned_robot``) bind;
    an ``assigned_robot`` / ``robot_ns`` added in this call is IGNORED and listed
    in ``ignored_pins``. Each weld goes to an arm within reach on its approach side
    (no arm is assumed to reach everything); one no arm can take is returned in
    ``unreachable``, never forced; idle arms are given work they are a good second
    choice for.

    Args:
        cell_id:                Cell identifier
        operations:             List of operation dicts with id, type, position,
                                approach_vector, sequence_group, priority
                                (an assigned_robot here is ignored unless it is
                                one of the run's locks)
        operation_type_default: Default type if op["type"] is missing

    Returns:
        {"total_steps": int, "steps": [...], "cycle_time_balance": float,
         "thermal_violations_fixed": int, "unreachable": [],
         "robots": [robot_id, ...], "operation_counts": {robot_id: int},
         "<robot_id>_count": int, ...}

        Each step is ``{"step": n, <robot_id>: op, ..., "min_inter_robot_dist_m":
        float?}`` — one entry per robot working in that step.
        On a scheduler fault: ``{"error": str, "dropped_operation_ids": [...]}``.

        A step says the operations in it MAY run in parallel on cost grounds; it is
        not a safety claim, and ``min_inter_robot_dist_m`` is the distance between
        the weld POINTS, reported for telemetry only. Measured on
        ``welding_cell_2_small`` it correlates with the arms' true swept clearance at
        -0.607 — the wrong sign — so do not gate on it. Whether a step's arms may
        actually move at once is decided by ``behaviour_tree._gate_multi_arm_step``,
        which validates the swept volumes and reorders, serialises or refuses.
    """
    try:
        robots = _cell_robots(cell_id)
        nrobots = len(robots)

        if nrobots == 0:
            return {"error": f"No robots in {cell_layout.robots_yaml(cell_id)}"}

        # Robot assignment is the ALLOCATOR's decision, or the operator's (the run's
        # locks: the request's assignments and the task file's own assigned_robot,
        # bound at the start of the run). A pin the MODEL writes into this call is
        # neither, so it is dropped and reported: on welding_cell_2_small the model pinned all
        # 27 welds onto two of four arms from a misread clearance and the plan ran
        # on half the cell. The run's real locks are then put back, as before —
        # including one the model dropped.
        from agents.trajectory import operator_directives
        locks = operator_directives.current_locks()
        ignored_pins = {}
        stripped = []
        for op in operations or []:
            if isinstance(op, dict) and (op.get("assigned_robot") or op.get("robot_ns")):
                pin = op.get("assigned_robot") or op.get("robot_ns")
                if locks.get(str(op.get("id"))) != pin:
                    ignored_pins[str(op.get("id"))] = pin
                op = {k: v for k, v in op.items() if k not in ("assigned_robot", "robot_ns")}
            stripped.append(op)
        operations = stripped
        operations = operator_directives.apply_locks(operations)

        # The allocation is a pure function of the (locked) operations, the cell's
        # config and the planning mode, and composing it costs minutes (206 s on
        # welding_cell_2_small). The same task on the same cell reuses it; any config edit
        # misses (plan_memo fingerprints the config content).
        from agents.trajectory.tools import plan_memo
        from agents.trajectory.tools.coordination import current_planning_mode as _mode
        memo_key = plan_memo.make_key(cell_id, "mrta", operations, robots,
                                      operation_type_default, _mode())
        cached = plan_memo.get("mrta", memo_key, cell_id=cell_id)
        if cached is not None:
            cached["cached"] = True
            return _with_ignored_pins(cached, ignored_pins)

        # Apply default type
        ops = []
        for op in operations:
            o = dict(op)
            o.setdefault("type", operation_type_default)
            ops.append(o)

        # Stage 1: Regional allocation — every robot in the cell participates. The
        # servicability check only breaks TIES, where the score is noise and the
        # fixture is the real constraint; it costs one IK solve per contested
        # (arm, weld) and degrades to None when there is no geometry stack.
        unreachable: list = []
        allocation = _regional_allocate(ops, robots,
                                        can_service=_servicability_check(cell_id),
                                        unreachable=unreachable)
        robot_names = list(allocation.keys())

        # Stages 2+3: TSP 2-opt then thermal spread, per robot — ALL robots, not
        # just the first two. Previously everything past robot 2 was dropped here.
        # Each arm's route starts at its end of the part — the weld closest to its
        # base — so arms on opposite sides begin far apart. Thermal spreading is a
        # per-cell choice (robots.yaml planning.thermal_spread, default on): it
        # reorders welds by face, which also scrambles the near-first order.
        thermal_gap = 3
        try:
            thermal_on = bool(((_load_robots_config(cell_id) or {}).get("planning")
                               or {}).get("thermal_spread", True))
        except Exception:  # noqa: BLE001
            thermal_on = True
        bases = {rid: (robots.get(rid) or {}).get("base_position") for rid in robot_names}
        sequences: dict[str, list[dict]] = {}
        for rid in robot_names:
            seq = _tsp_2opt(allocation[rid], start_near=bases.get(rid))
            sequences[rid] = _thermal_spread(seq, thermal_gap) if thermal_on else seq
        violations_fixed = 0  # actual detection requires face tracking — approximate

        # Stage 4: compose PARALLEL steps for ALL arms at once. With the geometry
        # stack, each step is chosen so every arm that works in it is clear of every
        # other arm — moving or parked — using the arms' own IK poses and joint
        # sweeps (agents/trajectory/tools/parallel_schedule.py). Without it (host),
        # the least-movement interleave: every arm keeps its own tour and the
        # assembler's gate decides. Either way the gate has the final word.
        from agents.trajectory.tools.parallel_schedule import (
            contention_advice, geometric_pair_model, orient_for_parallelism,
            standing_conflicts, unposable_operations,
        )
        from agents.trajectory.tools.coordination import current_planning_mode
        # Coordinated mode uses only the allocation and each arm's order; the
        # composed steps would be thrown away, and composing them costs minutes.
        # Only the stepwise mode uses composed steps; coordinated uses the order
        # alone, and classic (31 Aug) never had the composer.
        skip_compose = current_planning_mode() in ("coordinated", "classic")
        pair_clear = None if skip_compose else geometric_pair_model(cell_id, robots, ops)
        orientation_info = conflicts = no_pose = None
        if pair_clear is not None and len(sequences) > 1:
            # Each arm's tour costs the same walked either way; the direction is
            # CHOSEN by composing the program with the arms' own geometry and
            # keeping the most parallel one — not fixed to "start near the base",
            # which makes arms that share a boundary converge on it together.
            sequences, scheduled, orientation_info = orient_for_parallelism(
                sequences, pair_clear)
            schedule_mode = "parallel_composed"
            # Welds two arms can never work at the same time, and welds an arm has
            # no collision-free pose at: told, not discovered step by step.
            conflicts = standing_conflicts(sequences, pair_clear)
            no_pose = unposable_operations(sequences, pair_clear)
        else:
            scheduled = _least_movement_schedule(sequences)
            schedule_mode = "least_movement"

        # TOTALITY GATE — fail closed. An operation that was allocated but never
        # scheduled would silently never be performed: no cache entry, so the BT
        # assembler cannot even detect it is missing. Refuse the plan instead.
        allocated_ids = [op.get("id") for seq in sequences.values() for op in seq]
        scheduled_ids = [
            op.get("id") for st in scheduled
            for rid, op in st.items() if rid in sequences and isinstance(op, dict)
        ]
        if len(scheduled_ids) != len(allocated_ids) or set(scheduled_ids) != set(allocated_ids):
            dropped = sorted(str(i) for i in (set(allocated_ids) - set(scheduled_ids)))
            return {
                "error": (
                    "scheduling lost operations — the allocator assigned work that never "
                    "reached a step, so those operations would never be performed. "
                    "This is a bug in the scheduler, not a bad input; refusing the plan."
                ),
                "dropped_operation_ids": dropped,
                "allocated_count": len(allocated_ids),
                "scheduled_count": len(scheduled_ids),
            }

        # Stage 5: Cycle-time balance across ALL robots (least-loaded / most-loaded).
        counts = {rid: len(seq) for rid, seq in sequences.items()}
        active = [c for c in counts.values() if c > 0]
        balance = (round(min(active) / max(active), 3)
                   if len(active) > 1 and max(active) > 0 else 1.0)

        # Build steps output
        steps = []
        for i, sched in enumerate(scheduled):
            step: dict = {"step": i + 1}
            step.update(sched)
            steps.append(step)

        result = {
            "total_steps":              len(steps),
            "steps":                    steps,
            "cycle_time_balance":       balance,
            "thermal_violations_fixed": violations_fixed,
            # Operations no arm can take (out of every arm's reach, or only
            # approachable from where no arm stands): not allocated, never forced.
            "unreachable":              unreachable,
            "robots":                   robot_names,
            "operation_counts":         counts,
            "schedule_mode":            schedule_mode,
            "thermal_spread":           thermal_on,
        }
        if schedule_mode == "parallel_composed":
            # Where each arm's tool is at HOME: the target of a `park` (a planned
            # `move` that clears an arm out of the others' way within a step).
            result["home_tcp"] = _home_tcp(cell_id, robot_names)
            # How parallel the program is by construction: steps where every arm
            # that still had work was working.
            # Steps with two or more arms actually working at once.
            result["parallel_steps"] = sum(1 for st in scheduled
                                           if st.get("parallel_arms", 0) > 1)
            result["conflict_steps"] = [i + 1 for i, st in enumerate(scheduled)
                                        if st.get("conflict")]
            if orientation_info is not None:
                result["tour_orientation"] = orientation_info
            advice = contention_advice(conflicts or [])
            if no_pose:
                by_arm: dict = {}
                for u in no_pose:
                    by_arm.setdefault(u["robot"], []).append(u["operation"])
                advice += [f"{r} has no collision-free pose at {', '.join(o)} at any "
                           f"orientation the tool allows: scheduled alone. Check the "
                           f"weld target / approach there, or reallocate it."
                           for r, o in sorted(by_arm.items())]
                result["no_pose"] = no_pose
            if conflicts:
                result["contention"] = conflicts[:50]
                result["contention_count"] = len(conflicts)
            if advice:
                result["advice"] = advice
        # Per-robot counts as "<robot_id>_count" — kept for existing consumers.
        for rid, c in counts.items():
            result[f"{rid}_count"] = c
        plan_memo.put("mrta", memo_key, result, cell_id=cell_id)
        return _with_ignored_pins(result, ignored_pins)

    except Exception as e:
        return {"error": str(e)}


def _with_ignored_pins(result: dict, ignored_pins: dict) -> dict:
    """Report the pins the model wrote into this call and that were dropped."""
    if ignored_pins:
        result["ignored_pins"] = ignored_pins
        result["note"] = ("assigned_robot / robot_ns in this call were IGNORED for "
                          f"{len(ignored_pins)} operation(s): robot assignment is the "
                          "allocator's (or the operator's) decision, not the model's. "
                          "If a parallel step conflicts, sequence or stagger it — do "
                          "not drop arms.")
    return result


@tool
def reallocate_operation(
    cell_id: str,
    operation: dict,
    tried_robots: list[str],
    robots: list[dict] | None = None,
    reason_code: str | None = None,
) -> dict:
    """
    Hand an operation its ASSIGNED robot cannot perform to the next-best arm.

    Call this when ``run_trajectory_planner`` has failed an operation on its
    current robot and the cheap recoveries for that ``reason_code`` (approach flag,
    standoff/entry change, via-point, retry) are exhausted. It changes the
    DISTRIBUTION — the allocation ``run_mrta_pipeline`` produced is a first guess
    made from geometry alone, before any IK or collision check ran, so a weld the
    first-choice arm cannot reach is a reason to REALLOCATE it, not to drop it. An
    operation is only unreachable once EVERY arm has actually failed it.

    Ranking comes from the same approach-vector/proximity policy that made the
    original allocation (``task_allocation._robot_scores``), so a handover is not a
    second, improvised allocation rule. Candidates already in ``tried_robots`` are
    skipped, and each remaining one is checked against its declared
    ``workspace_radius`` from ``robots.yaml`` — an arm that cannot physically reach
    the point is reported ``within_workspace: false`` so you do not spend a
    planning attempt proving it.

    An operation carrying ``assigned_robot`` is LOCKED (part handling, gluing — the
    task requires that specific arm) and is REFUSED here: reason_code
    ``robot_locked``. That is not a failure to work around; it means the only
    remaining answers are a different entry strategy or genuinely unreachable.

    The handover APPENDS the operation to the end of the receiving arm's sequence.
    That is deliberate: inserting it mid-sequence would change the start pose of
    every operation after it on that arm, invalidating trajectories you have
    already planned and validated. Appending costs some cycle time and can put two
    same-face welds next to each other (thermal spread is an optimisation, not a
    gate), and it buys a handover that re-plans exactly ONE operation.

    Args:
        cell_id:      Cell identifier.
        operation:    The operation dict that failed — needs ``id``, ``position``
                      and ``approach_vector`` (plus ``assigned_robot`` if locked).
        tried_robots: Every robot that has ALREADY failed this operation, so it is
                      not proposed again. Include the current owner.
        robots:       Optional current per-robot execution order, same shape as
                      ``chain_trajectories``:
                      ``[{"robot_ns": "robot1", "operation_ids": [...]}, ...]``.
                      Used to tell you which operation the handover chains from,
                      and to weigh each arm's LOAD: among equally good takers the
                      least loaded one wins, and the result says how the load ends.
        reason_code:  The failure that prompted this. ``blocked_by_parked_arm`` /
                      ``inter_arm_clearance`` are REFUSED (``reallocation_not_needed``):
                      the weld is reachable, the other arm is in the way — clear it
                      (``find_clear_pose``) instead of overloading another arm.

    Returns:
        ``{"success": true, "operation_id", "to_robot", "candidates": [...],
           "insert_after_operation_id": str|null, "replan_operation_ids": [id],
           "instruction": str}``
        or ``{"success": false, "reason_code": "robot_locked" |
           "no_candidate_robot" | "out_of_reach_everywhere", ...}``.
    """
    try:
        op_id = operation.get("id") or operation.get("operation_id")
        from agents.trajectory import operator_directives
        run_lock = operator_directives.locked_robot(op_id)
        locked = operation.get("assigned_robot") or (run_lock[1] if run_lock else None)
        if locked:
            return {
                "success": False,
                "reason_code": "robot_locked",
                "operation_id": op_id,
                "assigned_robot": locked,
                "error": (
                    f"operation '{op_id}' is LOCKED to {locked} by its assigned_robot "
                    "field, so it cannot be reallocated. Recover by changing the ENTRY "
                    "(approach_offset_m, approach from above, a via-point) on that arm, "
                    "or record it unreachable — do not reassign it."
                ),
            }

        if reason_code in ("blocked_by_parked_arm", "inter_arm_clearance"):
            return {
                "success": False, "reason_code": "reallocation_not_needed",
                "operation_id": op_id,
                "error": (
                    f"'{op_id}' failed because ANOTHER ARM is in the way "
                    f"({reason_code}), not because its arm cannot do it. Handing it "
                    "over moves the work, not the obstruction, and unbalances the "
                    "program. Keep it: call find_clear_pose for the blocking arm, "
                    "plan that one move, then plan this weld again."),
            }

        cell_robots = _cell_robots(cell_id)
        if not cell_robots:
            return {"success": False, "reason_code": "no_candidate_robot",
                    "error": f"No robots in {cell_layout.robots_yaml(cell_id)}"}

        candidates = _reallocation_candidates(operation, cell_robots, tried_robots or [])
        if not candidates:
            return {
                "success": False,
                "reason_code": "no_candidate_robot",
                "operation_id": op_id,
                "tried_robots": list(tried_robots or []),
                "robots_in_cell": sorted(cell_robots),
                "error": (
                    f"every arm in the cell has already failed operation '{op_id}' "
                    f"({', '.join(sorted(tried_robots or [])) or 'none recorded'}). "
                    "NOW it is genuinely unreachable: record it in your final summary "
                    "with the per-robot failure reasons and continue with the rest."
                ),
            }

        reachable = [c for c in candidates if c["within_workspace"]]
        if not reachable:
            return {
                "success": False,
                "reason_code": "out_of_reach_everywhere",
                "operation_id": op_id,
                "candidates": candidates,
                "error": (
                    f"operation '{op_id}' lies outside the workspace_radius of every "
                    "remaining arm, so no handover can succeed. Record it unreachable "
                    "and continue — do not spend planning attempts on it."
                ),
            }

        order = {e.get("robot_ns"): list(e.get("operation_ids") or [])
                 for e in (robots or []) if isinstance(e, dict)}
        load = {ns: len(order.get(ns) or []) for ns in cell_robots}
        # Among takers whose geometry score is about as good as the best (within
        # 15%), the least loaded one: a handover is also a scheduling decision, and
        # piling every handed-over weld onto one arm idles the others.
        best = abs(float(reachable[0].get("score") or 0.0)) or 1.0
        close = [c for c in reachable
                 if abs(float(c.get("score") or 0.0) - float(reachable[0].get("score") or 0.0))
                 <= 0.15 * best]
        to_robot = min(close, key=lambda c: (load.get(c["robot_ns"], 0), c["rank"]))["robot_ns"]
        after = {**load, to_robot: load.get(to_robot, 0) + 1}
        tail = order.get(to_robot) or []
        insert_after = tail[-1] if tail else None

        return {
            "success": True,
            "operation_id": op_id,
            "to_robot": to_robot,
            "from_robots": list(tried_robots or []),
            "candidates": candidates,
            "insert_after_operation_id": insert_after,
            **({"load_after": after} if order else {}),
            **({"load_warning": (
                f"{to_robot} would carry {after[to_robot]} operations against "
                f"{min(after.values())} on the least loaded arm: the cycle time grows "
                "with the busiest arm. Check evaluate_program before keeping it.")}
               if order and after[to_robot] - min(after.values()) > 2 else {}),
            # Appending only ever re-plans the handed-over operation itself; every
            # trajectory already planned on the receiving arm keeps its start pose.
            "replan_operation_ids": [op_id] if op_id else [],
            "instruction": (
                f"Re-plan '{op_id}' on {to_robot} with run_trajectory_planner, as the "
                f"LAST operation of {to_robot}"
                + (f" (chained after '{insert_after}' — omit current_joints and it "
                   f"chains automatically from that goal)" if insert_after
                   else " (its first operation, so it starts from HOME)")
                + f". Then add '{op_id}' to the END of {to_robot}'s operation_ids for "
                "chain_trajectories and put it in a SEQUENTIAL step of the assembly "
                f"plan under {to_robot}. Do not leave it under its old robot."
            ),
        }

    except Exception as e:
        return {"success": False, "reason_code": "tool_error", "error": str(e)}


@tool
def run_trajectory_planner(
    cell_id: str,
    motion_requests: Optional[list[dict]] = None,
    approach_offset_m: float = 0.1,
    velocity_scale: Optional[float] = None,
    max_rrt_iterations: int = 3000,
    steps: Optional[list[list[dict]]] = None,
) -> dict:
    """
    Plan collision-free joint-space trajectories for one or more robots in a step
    — or, with ``steps``, for SEVERAL consecutive steps in one call.

    BATCH (preferred — one call instead of one per step): pass ``steps`` as a list
    of steps in EXECUTION order, each step being the list of motion requests (one
    per arm) that move together. They are planned in that order, each arm chained
    from where its previous step left it. When a step fails or the step gate
    refuses it (``stopped_at``), only the later steps that move one of its arms are
    skipped (``skipped_steps``) — steps moving other arms are still planned and
    kept (``kept_steps_after_stop``). Fix the failed step and re-submit exactly the
    failed and skipped steps (``advice`` lists them). ``motion_requests`` is
    ignored then.

    Every planned step carries ``step_gate``: the inter-arm check the assembler
    will apply — the step's arms checked as swept volumes against each other AND
    against every arm parked where its last planned leg ended. ``verdict``
    PARALLEL = clear; SEQUENTIAL = only one-at-a-time (``order``) is clear;
    REFUSED = no order is clear — fix it NOW (re-pair, retreat the parked arm),
    not at assembly. When the verdict is PARALLEL you do not need a separate
    validate_trajectory_set call for that step.

    Trajectories are cached by ``operation_id``; the result returns each arm's
    trajectory SUMMARY (waypoints, duration, start/end joints), not the arrays.
    Refer to a planned trajectory by its operation_id (validate_trajectory_set
    accepts ``{robot_ns: operation_id}``).

    Each motion request:
      {"robot_ns": str, "target": {"x", "y", "z"},
       "approach_vector": [ax, ay, az],
       "operation_type": str,
       "operation_id": str,          # optional but recommended — used to cache
                                     # the trajectory for assemble_behaviour_tree
       "current_joints": [j1..j6],   # optional
       "approach": bool,             # optional — set true when this request's
                                     # descent along the approach vector is the
                                     # terminal working stroke: its final segment is
                                     # the straight on-axis descent at the lean and
                                     # the approach speed (same collision radius as
                                     # transit; no reduced approach radius).
                                     # Leave false/omit for pure transit moves.
       "approach_offset_m": float,    # optional PER-OPERATION standoff: where
                                     # normal transit hands over to the
                                     # orientation-enforced approach. Overrides
                                     # the call-level default for this robot only,
                                     # so two arms in one parallel step may use
                                     # different standoffs.
       "retreat_offset_m": float,     # optional: how far to LEAVE the arm's
                                     # previous weld (on-axis lift-off) before
                                     # this motion; defaults to the standoff.
       "tool_spin_deg": float,       # optional: turn the tool about its OWN axis
                                     # by this many degrees. The lean (the axis
                                     # validate_approach checks) is unchanged; use
                                     # it to turn a wrist out of a self-collision
                                     # or out of the other arm's way. Never edit
                                     # the orientation quaternion by hand for this.
       "approach_velocity_scale": float}  # optional PER-OPERATION approach speed.
                                     # The transit phase runs at velocity_scale;
                                     # the terminal stroke runs at this. Defaults
                                     # to the cell's approach_velocity_scale.

    The backend is bound deterministically by the submitted task (never the LLM)
    and dispatched through agents/trajectory/tools/planner_backend.py:
      - "ompl" (default): Pinocchio IK → OMPL RRTConnect/BIT* → PathSimplifier →
                          TOPP-RA. Two-phase clearance AND speed, orientation-
                          enforced Cartesian descent, inter-robot coupling.
      - "rrt"           : legacy hand-rolled Space-Time RRT. NON-CONFORMING —
                          no inter-robot coupling, no orientation constraint, no
                          two-phase split. Results are stamped accordingly.

    Args:
        cell_id:            Cell identifier
        motion_requests:    List of per-robot motion requests
        approach_offset_m:  Default standoff, when a request does not set its own
        velocity_scale:     Transit-phase speed scaling. Omit it to run at the
                            cell's ``planning.max_velocity_scaling``
                            (robots.yaml); a value above that ceiling is clamped.
        max_rrt_iterations: Max iterations per robot (legacy RRT backend only)
        steps:              Optional batch: ``[[request, ...], [request, ...], ...]``
                            — consecutive steps in execution order (see above).

    Returns:
        {"success": bool,
         "trajectories": {robot_ns: [{"positions", "time_s", "phase"}]},
         "planning_time_s": float, "failures": [...], "backend": str,
         "step_gate": {...}}
        — with ``steps``: {"success", "steps": [per-step result], "planned_steps",
          "stopped_at": index | None, "planning_time_s"}
    """
    if steps:
        # One call, several steps: each step is THIS function on its own requests,
        # so a batched step is planned, cached and gated exactly like a single one.
        return _plan_steps(cell_id, steps, approach_offset_m=approach_offset_m,
                           velocity_scale=velocity_scale,
                           max_rrt_iterations=max_rrt_iterations)
    motion_requests = list(motion_requests or [])
    # The operator's assignments are enforced here, where motion is made: a
    # locked operation requested on another arm is refused before any solve.
    from agents.trajectory import operator_directives, retry_budget
    refused = [v for v in (operator_directives.violation(r) for r in motion_requests) if v]
    # The run's retry cap: an arm that has already failed this operation (or this
    # helper waypoint, under any name) too often is not asked again.
    refused += [v for v in (retry_budget.refusal(r) for r in motion_requests
                            if not any(x["robot_ns"] == r.get("robot_ns")
                                       and x["operation_id"] == r.get("operation_id")
                                       for x in refused)) if v]
    if refused:
        from agents.trajectory.reason_codes import recovery as _recovery
        for v in refused:
            rec = _recovery(v.get("reason_code"))
            if rec and "recovery" not in v:
                v["recovery"] = rec
    if refused:
        refused_ids = {(v["robot_ns"], v["operation_id"]) for v in refused}
        motion_requests = [r for r in motion_requests
                           if (r.get("robot_ns"), r.get("operation_id")) not in refused_ids]
    # Cancellation checkpoint. Raises PlanCancelled (a BaseException, so neither
    # the except below nor LangGraph's ToolNode swallows it) when the operator has
    # stopped this run. Placed before the solve, so a Stop pressed during step N
    # costs at most the remainder of step N-1's dispatch.
    plan_cancellation.raise_if_cancelled("run_trajectory_planner")
    if refused and not motion_requests:
        return {"success": False, "trajectories": {}, "failures": refused,
                "planning_time_s": 0.0, "backend": "none"}
    try:
        # Dispatch is owned by the planner_backend facade: it resolves the
        # context-bound backend and calls it through the result contract. No
        # fallback — a failed plan is reported as-is so the LLM can re-strategise.
        from agents.trajectory.tools.planner_backend import dispatch_plan

        velocity_scale = resolve_velocity_scale(cell_id, velocity_scale)
        # What each arm stood on before this step, so a step its gate REFUSES can
        # be taken back: otherwise the next attempt at the step would chain from
        # the end of a leg that will never run.
        from agents.trajectory.tools import trajectory_cache as _tc
        before = [(r.get("robot_ns"), r.get("operation_id"),
                   _tc.chain_state(cell_id, r.get("robot_ns"), r.get("operation_id")),
                   _tc.get_cached_trajectory(cell_id, r.get("robot_ns"), r.get("operation_id")))
                  for r in motion_requests if isinstance(r, dict) and r.get("robot_ns")]
        result = dispatch_plan(
            cell_id,
            motion_requests,
            approach_offset_m=approach_offset_m,
            velocity_scale=velocity_scale,
            max_iterations=max_rrt_iterations,
        )
        # Cache trajectories by operation_id (planner-agnostic) so
        # assemble_behaviour_tree can pull them back without the LLM.
        cache_planner_result(cell_id, motion_requests, result)
        retry_budget.record(result.get("failures"))
        if refused:
            result["failures"] = list(result.get("failures") or []) + refused
        gate = _step_gate(cell_id, result.get("trajectories") or {})
        if gate is not None:
            result["step_gate"] = gate
            if gate.get("verdict") == "REFUSED":
                for ns, op, chain, prior in before:
                    if ns in (result.get("trajectories") or {}):
                        _tc.restore_chain_state(cell_id, ns, chain)
                        _tc.restore_cached_trajectory(cell_id, ns, op, prior)
                gate["taken_back"] = True
                gate["advice"] = (gate.get("advice") or "") + (
                    " The step's legs were NOT kept: each arm still stands where it "
                    "stood before this call.")
        return result

    except Exception as e:
        return {"success": False, "trajectories": {}, "failures": [{"error": str(e)}],
                "planning_time_s": 0.0, "backend": "unknown"}


def _plan_steps(cell_id: str, steps: list, **kw) -> dict:
    """Plan consecutive steps in order; after a step fails, skip only what depends on it.

    Step N+1 chains from where step N left each arm, so a step that moves an arm
    whose previous step failed (or was skipped) starts from the wrong pose and is
    wasted work: it is not planned (``not_planned``), and its arms are blocked for
    the rest of the batch too. A later step whose arms are all untouched by the
    failure DOES get planned — on welding_cell_2_small one failed step used to throw away
    the 23 operations after it. ``stopped_at`` is the first step that failed;
    ``skipped_steps`` are the ones to send again once it is fixed (and only those:
    the independent steps planned after it are kept, and re-sending them would
    re-plan them from the wrong pose). The assembler re-gates every step against
    the arms' final poses, so a step planned next to an arm whose earlier step is
    later fixed is checked again there.

    The top level reads like a single call's result, so the trace and the live
    feed attribute every operation correctly: ``success`` when anything was
    planned, and ``failures`` naming each operation that failed, was refused by
    its step gate, or was never reached.
    """
    t0 = time.monotonic()
    out: list = []
    failures: list = []
    stopped_at = None
    blocked: set = set()
    skipped: list = []
    planned = 0
    for i, reqs in enumerate(steps):
        reqs = list(reqs or [])
        arms = {r.get("robot_ns") for r in reqs if isinstance(r, dict) and r.get("robot_ns")}
        if arms & blocked:
            skipped.append(i)
            failures.extend(
                {"operation_id": r.get("operation_id"), "robot_ns": r.get("robot_ns"),
                 "reason_code": "not_planned",
                 "error": (f"not planned: the batch stopped at step {stopped_at} and "
                           f"this step moves {', '.join(sorted(arms & blocked))}, whose "
                           "earlier step did not plan")}
                for r in reqs if isinstance(r, dict))
            blocked |= arms
            continue
        res = run_trajectory_planner.func(cell_id=cell_id, motion_requests=reqs, **kw)
        res["step"] = i
        out.append(res)
        failures.extend(f for f in (res.get("failures") or []) if isinstance(f, dict))
        gate = res.get("step_gate") or {}
        if gate.get("verdict") == "REFUSED":
            failures.extend(
                {"operation_id": r.get("operation_id"), "robot_ns": r.get("robot_ns"),
                 "reason_code": "step_gate_refused", "error": gate.get("advice")}
                for r in reqs if isinstance(r, dict))
        if res.get("failures") or not res.get("success") or gate.get("verdict") == "REFUSED":
            if stopped_at is None:
                stopped_at = i
            blocked |= arms
        else:
            planned += 1
    from agents.trajectory.reason_codes import recovery
    for f in failures:
        if f.get("reason_code") and "recovery" not in f:
            rec = recovery(f["reason_code"])
            if rec:
                f["recovery"] = rec
    kept_after = [r["step"] for r in out
                  if stopped_at is not None and r["step"] > stopped_at
                  and not (r.get("failures") or not r.get("success")
                           or (r.get("step_gate") or {}).get("verdict") == "REFUSED")]
    result = {
        "success": any(r.get("success") for r in out),
        "complete": stopped_at is None,
        "steps": out,
        "planned_steps": planned,
        "stopped_at": stopped_at,
        "failures": failures,
        "planning_time_s": round(time.monotonic() - t0, 3),
    }
    if stopped_at is not None:
        result["skipped_steps"] = skipped
        result["kept_steps_after_stop"] = kept_after
        if not kept_after:
            result["advice"] = (
                f"step {stopped_at} needs attention (see its failures / step_gate); the "
                f"{len(steps) - stopped_at - 1} step(s) after it were NOT planned. Fix step "
                f"{stopped_at} and call again with the steps from {stopped_at} on.")
        else:
            failed = [r["step"] for r in out
                      if r["step"] >= stopped_at and r["step"] not in kept_after]
            result["advice"] = (
                f"step(s) {failed} need attention (see their failures / step_gate). "
                f"Step(s) {kept_after} move other arms and WERE planned — keep them, do "
                f"not send them again. Step(s) {skipped} were NOT planned because they "
                f"move an arm whose earlier step failed: fix {failed}, then call again "
                f"with exactly {sorted(failed + skipped)} in execution order.")
    return result


def _step_gate(cell_id: str, legs: dict) -> Optional[dict]:
    """The assembler's inter-arm gate, run on a step as soon as it is planned.

    The same function ``assemble_behaviour_tree`` applies (``_gate_multi_arm_step``),
    against the same bystanders: every arm not in this step, standing where its
    last planned leg ended. Run here, a clash is reported while the step is fresh
    — on bt_1790857060 the assembler found three of them only at the end, and
    each cost the agent a round of re-planning. ``PLAN_STEP_GATE=0`` turns it off.
    """
    import os
    if os.environ.get("PLAN_STEP_GATE", "1").strip().lower() in ("0", "false", "off"):
        return None
    legs = {ns: t for ns, t in (legs or {}).items() if t}
    if not legs:
        return None
    try:
        from agents.trajectory.tools.trajectory_cache import get_last_config
        from agents.trajectory.tools.behaviour_tree import _arm_home, _gate_multi_arm_step
        bystanders = {}
        for ns in _cell_robots(cell_id):
            if ns not in legs:
                # Where it last stopped — or HOME, where an arm that has not moved
                # yet stands (the assembler gates against the same poses).
                q = get_last_config(cell_id, ns) or _arm_home(cell_id, ns)
                if q:
                    bystanders[ns] = list(q)
        if len(legs) < 2 and not bystanders:
            return {"checked": False, "verdict": "PARALLEL",
                    "note": "single arm and no other arm in the cell"}
        rep = _gate_multi_arm_step(cell_id, legs, bystanders=bystanders)
    except Exception as e:  # noqa: BLE001 — the assembler gates again; say so
        return {"checked": False, "verdict": "UNCHECKED", "error": str(e),
                "advice": "the step gate could not run here; assemble_behaviour_tree "
                          "gates the step again and refuses what it cannot clear"}
    par = rep.get("parallel") or {}
    verdict = {"PARALLEL": "PARALLEL", "SEQUENTIAL": "SEQUENTIAL"}.get(rep.get("mode"),
                                                                       "REFUSED")
    gate = {"checked": bool(rep.get("gated")), "verdict": verdict,
            "arms": rep.get("arms"), "bystanders": rep.get("bystanders") or [],
            "min_clearance_m": par.get("min_clearance_m"),
            "warning_distance_m": par.get("warning_distance_m"),
            "worst_pair": par.get("worst_pair"),
            "collision_time_s": par.get("collision_time_s")}
    if verdict == "SEQUENTIAL":
        gate["order"] = rep.get("order")
        gate["advice"] = ("together these legs enter the warning band; one at a time "
                          f"in order {rep.get('order')} is clear. To keep the step "
                          "parallel, pair a different operation or stagger the start.")
    elif verdict == "REFUSED" and (rep.get("leg_faults") or rep.get("reason_code")):
        gate["blocked_arms"] = rep.get("blocked_arms")
        gate["leg_faults"] = rep.get("leg_faults") or []
        gate["advice"] = (
            "a leg check broke: retry once, then report it" if rep.get("reason_code")
            else "a leg breaches on its own (" + ", ".join(
                f"{f['robot_ns']}: {f['reason_code']}" for f in gate["leg_faults"])
            + "): re-plan that leg (a via point, another IK branch, another arm) — no "
              "order, delay or pairing fixes it")
    elif verdict == "REFUSED":
        gate["blocked_arms"] = rep.get("blocked_arms")
        gate["blocked_by_parked"] = rep.get("blocked_by_parked")
        gate["serial_attempts"] = rep.get("serial_attempts")
        gate["advice"] = ("assemble_behaviour_tree will REFUSE this step: no order runs "
                          "these arms clear of each other"
                          + (f" and of the parked {rep.get('bystanders')}"
                             if rep.get("bystanders") else "")
                          + ". Fix it now: pair a different operation, or first plan a "
                          "'move' that retreats the parked arm clear (toward its own "
                          "base, raised), then re-plan this step.")
    return gate


@tool
def run_path_shortcut(
    waypoints: list[list[float]],
    threshold: float = 0.06,
) -> dict:
    """
    Remove redundant RRT nodes from a joint-space path.

    Args:
        waypoints:  List of joint-space configs [[j1..j6], ...]
        threshold:  Max angular change (rad) that can be removed per shortcut

    Returns:
        {"waypoints": [[j1..j6], ...], "original_len": int, "reduced_len": int}
    """
    np_wps = [np.array(w) for w in waypoints]
    reduced = _shortcut_path(np_wps, threshold)
    return {
        "waypoints":    [w.tolist() for w in reduced],
        "original_len": len(waypoints),
        "reduced_len":  len(reduced),
    }


# ─── Start delays: separating arms in TIME inside one parallel step ───────────
# A PARALLEL step whose arms breach each other's swept volume is usually a
# TIMING conflict, not a geometric one: both arms want the same space, at the
# same moment. Starting one of them a little later keeps the step parallel —
# far cheaper than serialising it — and is what `start_delay_s` (a Delay
# decorator in the tree) executes. The search lives here, next to the check it
# answers to, so the delay offered and the delay verified are the same gate.

DELAY_STEP_S = 0.25        # search resolution
DELAY_ROBUSTNESS_S = 0.25  # a delay must also clear at ±this — the executor's
                           # start is not a perfect clock


def _resolve_trajectories(cell_id: str, trajectories: dict) -> dict:
    """``{robot_ns: operation_id | trajectory}`` → ``{robot_ns: trajectory}``.

    An operation id is looked up in the planner's cache, so the agent refers to a
    planned leg instead of sending its waypoint array back. Unknown ids raise —
    validating against nothing is not a pass.
    """
    from agents.trajectory.tools.trajectory_cache import get_cached_trajectory
    out: dict = {}
    for ns, value in (trajectories or {}).items():
        if isinstance(value, str):
            traj = get_cached_trajectory(cell_id, ns, value)
            if not traj:
                raise ValueError(f"no planned trajectory cached for {ns}/{value} — "
                                 "run_trajectory_planner it first")
            out[ns] = traj
        elif isinstance(value, dict) and value.get("operation_id"):
            out[ns] = _resolve_trajectories(cell_id, {ns: value["operation_id"]})[ns]
        else:
            out[ns] = value
    return out


def delayed(traj: list[dict], delay_s: float) -> list[dict]:
    """``traj`` started ``delay_s`` later. Sampling holds the first pose until
    then (see ``_sample_at_time``), so the arm waits exactly where it stands."""
    d = max(0.0, float(delay_s or 0.0))
    if d <= 0 or not traj:
        return traj
    return [{**p, "time_s": round(float(p.get("time_s", 0.0)) + d, 4)} for p in traj]


def apply_start_delays(legs: dict, delays: Optional[dict]) -> dict:
    """Every leg shifted by its arm's declared start delay."""
    delays = delays or {}
    return {ns: delayed(tr, delays.get(ns, 0.0)) for ns, tr in legs.items()}


def find_start_delays(cell_id: str, legs: dict, *,
                      base_delays: Optional[dict] = None,
                      step_s: float = DELAY_STEP_S,
                      robustness_s: float = DELAY_ROBUSTNESS_S,
                      warning_distance_m: Optional[float] = None) -> Optional[dict]:
    """The smallest start delays that make a set of legs clear of each other.

    Works for ANY number of arms: arms are staggered one at a time, each given
    the smallest delay that clears it against every arm already scheduled, and
    every leader order is tried (up to 3 arms; beyond that, the given order) —
    the order with the least total delay wins. A delay is only accepted if the
    set also clears at ``delay ± robustness_s``.

    Returns ``{"start_delays": {ns: s}, "total_delay_s", "min_clearance_m"}``,
    or ``None`` when no delay short of running the arms one after the other
    clears them — then the step needs a different pairing, not a delay.
    """
    from itertools import permutations

    names = [ns for ns, tr in legs.items() if tr]
    if len(names) < 2:
        return None
    if warning_distance_m is None:
        warning_distance_m = _load_collision_config(cell_id)["warning_distance"]
    base = {ns: float((base_delays or {}).get(ns, 0.0)) for ns in names}
    dur = {ns: float(legs[ns][-1].get("time_s", 0.0)) for ns in names}

    def clear(subset, dl):
        if len(subset) < 2:
            return True, float("inf")
        r = inter_robot_clearance(
            cell_id, {ns: delayed(legs[ns], dl[ns]) for ns in subset},
            warning_distance_m=warning_distance_m)
        return bool(r.get("collision_free")), float(r.get("min_clearance_m", 0.0))

    orders = list(permutations(names)) if len(names) <= 3 else [tuple(names)]
    best = None
    for order in orders:
        dl = dict(base)
        placed = [order[0]]
        ok = True
        for ns in order[1:]:
            # Past the moment every placed arm has finished, a later start is
            # just "run afterwards" — a pairing problem, not a timing one.
            limit = max(dl[o] + dur[o] for o in placed)
            d = dl[ns]
            found = None
            while d <= limit + 1e-9:
                trial = {**dl, ns: d}
                if all(clear(placed + [ns], {**trial, ns: max(0.0, d + e)})[0]
                       for e in (0.0, robustness_s, -robustness_s)):
                    found = d
                    break
                d = round(d + step_s, 4)
            if found is None:
                ok = False
                break
            dl[ns] = found
            placed.append(ns)
        if not ok:
            continue
        total = sum(dl[ns] - base[ns] for ns in names)
        if best is None or total < best[0]:
            best = (total, dl)
    if best is None:
        return None
    _, dl = best
    return {"start_delays": {ns: round(dl[ns], 3) for ns in names},
            "total_delay_s": round(best[0], 3),
            "min_clearance_m": round(clear(names, dl)[1], 4)}


@tool
def validate_trajectory_set(
    cell_id: str,
    trajectories: dict,
    collision_radius_m: float = 0.12,
    n_samples: int = 20,
    start_delays: Optional[dict] = None,
    suggest_delay: bool = True,
    check_static_obstacles: bool = True,
    check_joint_limits: bool = True,
) -> dict:
    """
    Validate ANY number of robot trajectories for inter-robot collision — and
    each one against the cell's static obstacles and its arm's joint limits.

    Checks every unordered robot pair at ``n_samples`` instants on a shared
    wall-clock timeline spanning the longest trajectory, so arms with different
    durations are compared at the same real time. Use this for a PARALLEL step
    with two OR MORE arms; ``validate_trajectory_pair`` is the two-robot special
    case and delegates here.

    Each arm's swept volume INCLUDES its configured ``tool_collision`` (torch,
    plate), and ``collision_free`` means clearance stays ABOVE the cell's
    ``warning_distance`` — the same volume and the same threshold the runtime
    collision monitor enforces. A plan that merely avoids touching still gets
    E-STOPPED on the floor, so it is reported as not collision-free here.

    Args:
        cell_id:             Cell identifier
        trajectories:        ``{robot_ns: operation_id}`` — the cached trajectory
                             run_trajectory_planner planned for that operation —
                             or ``{robot_ns: [{"positions": [j1..j6], "time_s": float}]}``.
                             Two or more robots.
        collision_radius_m:  Sphere radius per link (default 0.12m)
        n_samples:           MINIMUM number of time samples. The check is always
                             run at least as densely as the runtime monitor
                             samples the cell, so raising this refines the check
                             and lowering it cannot coarsen it. See
                             ``samples_checked`` in the result for what was used.
        start_delays:        ``{robot_ns: seconds}`` — the `start_delay_s` each arm
                             will get in the step. The check runs on that
                             staggered timeline, exactly as the step executes.

    ``check_static_obstacles`` (default on) also runs every trajectory through the
    planner's emitted-trajectory check (floor, fixtures, self-collision — the
    runtime monitor's rule): a breach sets ``collision_free`` False and lists
    ``static_collisions``. No start delay fixes one. If the check cannot run, the
    arm-vs-arm verdict stands and ``static_obstacles_checked`` is False.

    ``check_joint_limits`` (default on) checks every sample of every leg (and so,
    the box being convex, the straight motion between samples) against the arm's
    joint position limits (robots.yaml ∩ the arm's model). A leg past a limit
    cannot be executed: ``collision_free`` is False, ``joint_limits_ok`` False and
    the excursions are in ``joint_limit_violations``. No start delay fixes one.

    When the set collides, the result also carries ``delay_fix``: the smallest
    ``start_delays`` that clear it (``None`` if no delay can — then pair the arm
    with a different weld). Put those values as ``start_delay_s`` on the step's
    robots in ``assemble_behaviour_tree`` and the step stays PARALLEL.

    Returns:
        {"collision_free": bool, "min_clearance_m": float, "warning_distance_m": float,
         "robots": [...], "pairs_checked": int, "worst_pair": [nsA, nsB],
         "samples_checked": int, "monitor_rate_hz": float,
         "collision_time_s": float?, "colliding_pairs": [...]?, "suggestion": str?,
         "joint_limits_ok": bool?, "joint_limit_violations": [...]?}
    """
    try:
        trajectories = _resolve_trajectories(cell_id, trajectories)
        # ONE owner for "do these arms clear each other while moving?" — this tool
        # exposes it to the agent, and the OMPL planner consults the SAME function
        # when it decides whether a route is admissible. What used to live here was
        # a copy the planner did not share; see ``inter_robot_clearance``.
        timed = apply_start_delays(trajectories, start_delays)
        result = inter_robot_clearance(
            cell_id, timed,
            collision_radius_m=collision_radius_m,
            n_samples=n_samples,
            # The band is the cell's collision policy, resolved HERE so this tool
            # reports against the same threshold as the rest of this module (see
            # ``validate_trajectory_pair``'s clearance report) rather than against
            # a second read taken deeper in the stack.
            warning_distance_m=_load_collision_config(cell_id)["warning_distance"],
        )
        if start_delays:
            result["start_delays"] = dict(start_delays)
        if check_joint_limits:
            limits = _joint_limit_violations(cell_id, trajectories)
            result["joint_limits_ok"] = limits["checked"] and not limits["violations"]
            if limits.get("error"):
                result["joint_limit_check_error"] = limits["error"]
            if limits.get("failed"):
                # A check that ran and broke has not said yes (leg_verdict).
                result["leg_check_failures"] = limits["failed"]
                result["collision_free"] = False
                result["suggestion"] = ("a leg check broke instead of answering "
                                        f"({limits['failed'][0]['error']}): retry once, "
                                        "then report it")
                return result
            if limits["violations"]:
                from agents.trajectory.tools.joint_limits import describe
                result["joint_limit_violations"] = limits["violations"]
                result["collision_free"] = False
                worst = max(limits["violations"],
                            key=lambda h: h.get("excess_rad") or float("inf"))
                result["suggestion"] = describe(worst, worst["robot_ns"])
                return result
        static = _static_collisions(cell_id, trajectories) if check_static_obstacles else None
        if static is not None:
            result["static_obstacles_checked"] = static["checked"]
            if static.get("error"):
                result["static_check_error"] = static["error"]
            if static.get("failed"):
                result["leg_check_failures"] = static["failed"]
                result["collision_free"] = False
                result["suggestion"] = ("a leg check broke instead of answering "
                                        f"({static['failed'][0]['error']}): retry once, "
                                        "then report it")
                return result
            if static["collisions"]:
                result["static_collisions"] = static["collisions"]
                result["collision_free"] = False
                result["suggestion"] = (
                    "A trajectory breaches a static obstacle or the arm itself "
                    f"({static['collisions'][0]['robot_ns']} vs "
                    f"{static['collisions'][0]['object_id']} at "
                    f"{static['collisions'][0]['time_s']} s). No start delay fixes this: "
                    "re-plan that robot's leg (e.g. replan_operation).")
                return result
        from agents.trajectory.tools.coordination import current_planning_mode
        if (result.get("collision_free") is False and suggest_delay
                and current_planning_mode() != "classic"):
            fix = find_start_delays(cell_id, trajectories, base_delays=start_delays,
                                    warning_distance_m=result["warning_distance_m"])
            result["delay_fix"] = fix
            head = (f"Clearance reaches {result['min_clearance_m']} m, at or below the "
                    f"cell's warning_distance of {result['warning_distance_m']} m — the "
                    "runtime collision monitor would E-STOP this step. ")
            if fix:
                result["suggestion"] = head + (
                    f"Keep the step PARALLEL with start delays {fix['start_delays']} "
                    "(set them as start_delay_s on those robots in the step plan and "
                    "re-validate with start_delays).")
            else:
                result["suggestion"] = head + (
                    "No start delay clears it: pair the conflicting arm with a DIFFERENT "
                    "pending weld and re-plan the step with all arms in one call, or "
                    "run these arms one after another (a SEQUENCE step).")
        return result

    except Exception as e:
        # Two different answers, kept apart for the gates (_gate_multi_arm_step):
        # "cannot run here" (no geometry stack / no model for this cell) has no
        # opinion; "ran and broke" has not said yes.
        unavailable = isinstance(e, (ImportError, FileNotFoundError))
        return {"error": str(e),
                "error_kind": "unavailable" if unavailable else "check_failed"}


def _joint_limit_violations(cell_id: str, trajectories: dict) -> dict:
    """Every leg through ``joint_limits.leg_limit_violation`` (``leg_verdict``'s
    cache and policy). Never raises. A check that broke is in ``failed``."""
    out: dict = {"checked": True, "violations": [], "failed": []}
    try:
        from agents.trajectory.tools.joint_limits import leg_limit_violation
        from agents.trajectory.tools.leg_verdict import judged
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "violations": [], "failed": [],
                "error": f"{type(exc).__name__}: {exc}"}
    for ns, traj in (trajectories or {}).items():
        if not traj:
            continue
        status, v = judged("limits", cell_id, ns, traj, leg_limit_violation)
        if status != "hit":
            out["checked"] = False
            out["error"] = f"{ns}: {v}"
            if status == "error":
                out["failed"].append({"robot_ns": ns, "check": "joint limits", "error": v})
            continue
        if v is not None:
            out["violations"].append({"robot_ns": ns, **v})
    return out


def _static_collisions(cell_id: str, trajectories: dict) -> dict:
    """Each trajectory through ``leg_static_breach`` (``leg_verdict``'s cache and
    policy). Never raises. A check that broke is in ``failed``."""
    from agents.trajectory.tools.leg_verdict import judged
    out: dict = {"checked": True, "collisions": [], "failed": []}
    for ns, traj in (trajectories or {}).items():
        if not traj:
            continue
        status, hit = judged("static", cell_id, ns, traj, leg_static_breach)
        if status != "hit":
            out["checked"] = False
            out["error"] = f"{ns}: {hit}"
            if status == "error":
                out["failed"].append({"robot_ns": ns, "check": "static obstacles",
                                      "error": hit})
            continue
        if hit is not None:
            i = int(hit.get("sample_index", 0))
            out["collisions"].append({
                "robot_ns": ns, "type": hit.get("type"), "object_id": hit.get("object_id"),
                "link_index": hit.get("link_index"), "clearance_m": hit.get("clearance_m"),
                "sample_index": i, "phase": hit.get("phase"),
                "time_s": round(float(traj[min(i, len(traj) - 1)].get("time_s", 0.0)), 4),
            })
    return out


@tool
def validate_trajectory_pair(
    cell_id: str,
    robot1_ns: str,
    trajectory1: Optional[list[dict]] = None,
    robot2_ns: str = "",
    trajectory2: Optional[list[dict]] = None,
    collision_radius_m: float = 0.12,
    n_samples: int = 20,
    operation_id1: Optional[str] = None,
    operation_id2: Optional[str] = None,
) -> dict:
    """
    Validate two robot trajectories for inter-robot collision.

    Two-robot special case of ``validate_trajectory_set`` (which it delegates to).
    For a step with three or more arms call ``validate_trajectory_set`` directly —
    this signature cannot express it.

    Args:
        cell_id:             Cell identifier
        robot1_ns:           First robot namespace
        trajectory1:         [{"positions": [j1..j6], "time_s": float}]
        robot2_ns:           Second robot namespace
        trajectory2:         Same format as trajectory1
        operation_id1/2:     Instead of an array: the operation_id whose cached
                             trajectory to use (preferred — no arrays to pass).
        collision_radius_m:  Sphere radius per link (default 0.12m)
        n_samples:           MINIMUM number of time samples — see
                             ``validate_trajectory_set``; the check never runs
                             more coarsely than the runtime monitor.

    Returns:
        {"collision_free": bool, "min_clearance_m": float}
    """
    return validate_trajectory_set.func(
        cell_id=cell_id,
        trajectories={robot1_ns: trajectory1 or operation_id1,
                      robot2_ns: trajectory2 or operation_id2},
        collision_radius_m=collision_radius_m,
        n_samples=n_samples,
    )


@tool
def find_interlock_set(
    cell_id: str,
    targets: list[dict],
    operation_type: str = "weld",
) -> dict:
    """
    Find safe staging positions enabling parallel motion for ANY number of robots.

    The N-robot generalisation of ``find_interlocks``. Applies each strategy to
    every robot at once and accepts it only when EVERY unordered pair clears
    ``min_clear`` — a 3-way convergence is not resolved by checking two arms and
    hoping, which is why the two-robot signature could not express it.

    Strategies, in order: ``raise_arm``, ``rotate_outward`` (fanned out per robot
    around the cell centroid, so N arms separate instead of two swapping sides),
    ``partial_retract``. Falls back to ``sequential_fallback``.

    Args:
        cell_id:        Cell identifier
        targets:        ``[{"robot_ns": str, "x": float, "y": float, "z": float}, ...]``
                        — two or more robots and their final targets.
        operation_type: Affects the minimum clearance margin.

    Returns:
        {"needs_interlock": bool, "strategy": str, "robots": [...],
         "min_clearance_m": float?, "interlocks": {ns: {ee_position, joint_positions}}?,
         "execution_plan": [...]}
    """
    CLEARANCE = {"laser": 0.30, "laser_weld": 0.30, "laser_cut": 0.30}
    # A staging pose that sits inside the monitor's warning band is not "safe" —
    # never accept less than the band execution enforces.
    min_clear = max(CLEARANCE.get(operation_type, 0.24),
                    _load_collision_config(cell_id)["warning_distance"])

    names = [str(t.get("robot_ns")) for t in (targets or []) if t.get("robot_ns")]
    goals = {str(t["robot_ns"]): np.array(
        [float(t.get("x", 0.0)), float(t.get("y", 0.0)), float(t.get("z", 0.0))])
        for t in (targets or []) if t.get("robot_ns")}

    if len(names) < 2:
        return {"needs_interlock": False, "strategy": "single_robot", "robots": names}

    # If every pair of TARGETS is already far apart, no staging is needed.
    pair_dists = {(a, b): float(np.linalg.norm(goals[a] - goals[b]))
                  for i, a in enumerate(names) for b in names[i + 1:]}
    closest = min(pair_dists.values())
    if closest > 0.4:
        return {
            "needs_interlock": False,
            "strategy": "direct_parallel",
            "robots": names,
            "closest_target_pair_m": round(closest, 3),
        }

    try:
        from agents.trajectory.tools.pinocchio import run_ik as _ik

        from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose

        # Per robot: two arms in one cell may declare different home poses, and
        # partial_retract interpolates TOWARD home, so a shared literal retracts
        # one of them toward a pose it never parks at.
        HOME = {ns: np.array(home_pose(cell_id, ns, HOME_JOINTS_FALLBACK), float)
                for ns in names}
        models, bases, yaws, q_goal, tool_cfgs = {}, {}, {}, {}, {}
        for ns in names:
            cfg = _load_robot_config(cell_id, ns)
            models[ns], _ = _load_pinocchio_model(cell_id, ns)
            bases[ns] = np.array(cfg["base_position"])
            yaws[ns] = float(cfg.get("base_orientation_yaw", 0.0))
            tool_cfgs[ns] = cfg.get("tool_collision")
            g = goals[ns]
            ik = _ik.invoke({"cell_id": cell_id, "robot_ns": ns,
                             "x": float(g[0]), "y": float(g[1]), "z": float(g[2])})
            if not ik.get("success"):
                return {"needs_interlock": True, "strategy": "ik_failed",
                        "robots": names, "error": f"IK failed for {ns}"}
            q_goal[ns] = np.array(ik["joint_positions"])

        # Fan rotate_outward per robot: each arm turns AWAY from the centroid of the
        # other bases, so N arms diverge instead of two merely swapping sides.
        centroid = np.mean([bases[ns] for ns in names], axis=0)
        fan_sign = {}
        for ns in names:
            out = bases[ns] - centroid
            fan_sign[ns] = 1.0 if float(np.arctan2(out[1], out[0])) >= 0 else -1.0

        strategies = [
            ("raise_arm",
             lambda q, ns: q + np.array([0.0, -0.5, 0.4, 0.0, 0.0, 0.0])),
            ("rotate_outward",
             lambda q, ns: q + np.array([0.6 * fan_sign[ns], 0.0, 0.0, 0.0, 0.0, 0.0])),
            ("partial_retract",
             lambda q, ns: 0.7 * q + 0.3 * HOME[ns]),
        ]

        for strategy_name, fn in strategies:
            il_q = {ns: np.clip(fn(q_goal[ns], ns), -3.14, 3.14) for ns in names}
            spheres = {ns: _config_spheres(models[ns], models[ns].createData(),
                                           il_q[ns], bases[ns], yaws[ns], tool_cfgs[ns])
                       for ns in names}
            # EVERY pair must clear — not just one.
            worst = float("inf")
            ok = True
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    col, clearance = _check_robot_pair_collision(spheres[a], spheres[b], 0.12)
                    worst = min(worst, clearance)
                    if col or clearance < min_clear:
                        ok = False
            if not ok:
                continue

            il_ee = {ns: _fk_ee(models[ns], models[ns].createData(),
                                il_q[ns], bases[ns], yaws[ns]).tolist() for ns in names}
            return {
                "needs_interlock": True,
                "strategy":        strategy_name,
                "robots":          names,
                "min_clearance_m": round(worst, 4),
                "pairs_checked":   len(names) * (len(names) - 1) // 2,
                "interlocks": {ns: {"ee_position": il_ee[ns],
                                    "joint_positions": il_q[ns].tolist()} for ns in names},
                "execution_plan": [
                    {"step": 1, "action": "PARALLEL",
                     "motions": [{"robot": ns, "target": il_ee[ns]} for ns in names],
                     "depends_on": []},
                    {"step": 2, "action": "PARALLEL",
                     "motions": [{"robot": ns, "target": goals[ns].tolist()} for ns in names],
                     "depends_on": [{"robot": ns, "position": il_ee[ns], "tolerance_m": 0.05}
                                    for ns in names]},
                ],
            }

        # Fallback: run them one after another.
        return {
            "needs_interlock": True,
            "strategy":        "sequential_fallback",
            "robots":          names,
            "execution_plan": [
                {"step": i + 1, "action": "SEQUENTIAL", "robot": ns,
                 "target": goals[ns].tolist()}
                for i, ns in enumerate(names)
            ],
        }

    except Exception as e:
        return {"needs_interlock": True, "strategy": "error", "robots": names,
                "error": str(e)}


@tool
def find_interlocks(
    cell_id: str,
    robot1_ns: str,
    x1: float,
    y1: float,
    z1: float,
    robot2_ns: str,
    x2: float,
    y2: float,
    z2: float,
    operation_type: str = "weld",
) -> dict:
    """
    Find safe intermediate positions enabling parallel robot motion without collision.

    Two-robot special case of ``find_interlock_set`` (which it delegates to). For
    three or more converging arms call ``find_interlock_set`` directly — this
    signature cannot express it, and resolving two pairs of a 3-way convergence
    does not make the third pair safe.

    Generates interlock candidates per robot using these strategies (in order):
      raise_arm, rotate_outward, partial_retract, approach_from_above, home.

    Args:
        cell_id:         Cell identifier
        robot1_ns:       First robot namespace
        x1, y1, z1:      First robot's final target (world frame, metres)
        robot2_ns:       Second robot namespace
        x2, y2, z2:      Second robot's final target
        operation_type:  Affects minimum clearance margin

    Returns:
        {"needs_interlock": bool, "strategy": str, ...execution_plan}
    """
    out = find_interlock_set.func(
        cell_id=cell_id,
        targets=[
            {"robot_ns": robot1_ns, "x": x1, "y": y1, "z": z1},
            {"robot_ns": robot2_ns, "x": x2, "y": y2, "z": z2},
        ],
        operation_type=operation_type,
    )
    # Preserve this tool's original two-robot output keys for existing callers.
    if out.get("closest_target_pair_m") is not None:
        out["target_distance_m"] = out["closest_target_pair_m"]
    il = out.get("interlocks") or {}
    if robot1_ns in il:
        out["robot1_interlock"] = il[robot1_ns]
    if robot2_ns in il:
        out["robot2_interlock"] = il[robot2_ns]
    return out
