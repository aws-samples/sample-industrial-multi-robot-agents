"""What a program COSTS, and where to put an arm so it is out of the way.

The planning tools answer "is this valid?". These answer the question the agent
is there for once everything is valid — "is it fast?" — so it can compare
alternatives instead of stopping at the first plan that passes:

  * :func:`evaluate_program` — the cycle time of a step plan (the shape
    ``assemble_behaviour_tree`` takes), each arm's busy and idle time, the steps
    where an arm with work ahead stands idle, the tightest inter-arm clearances,
    and whether the arms can return HOME together from where they finish (the
    assembler serialises — or staggers — a return that cannot, and that costs
    seconds at the end of every cycle).
  * :func:`find_clear_pose` — the ONE pose, on the way from where an arm stands
    toward its own HOME, that clears another arm's upcoming legs with margin.
    ``blocked_by_parked_arm`` used to leave the agent guessing a retreat target,
    checking it, and retreating again — three moves where one does.

Deterministic. Nothing here plans a leg: trajectories come from the cache, the
inter-arm check is ``validate_trajectory_set`` (the gates' own), and the return
home is estimated from straight joint lines at the cell's home speed.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

#: A step clearing the warning band by less than this is reported as tight.
TIGHT_MARGIN_M = 0.03
DEFAULT_HOLD_S = 0.5


def _leg(cell_id: str, ns: str, op) -> Optional[list]:
    from agents.trajectory.tools.trajectory_cache import get_cached_trajectory
    return get_cached_trajectory(cell_id, ns, op) if op else None


def _dur(leg) -> float:
    return float(leg[-1].get("time_s", 0.0)) if leg else 0.0


def _home_leg(cell_id: str, ns: str, start: list) -> Optional[list]:
    """A straight joint line from ``start`` to the arm's HOME, timed at the cell's
    home speed — an estimate of the return the assembler will plan."""
    from agents.trajectory.tools.behaviour_tree import _arm_home
    home = _arm_home(cell_id, ns)
    if not home or not start:
        return None
    try:
        from agents.trajectory.tools.robot_kinematics import (
            _load_robot_config, resolve_home_velocity_scale,
        )
        lim = _load_robot_config(cell_id, ns).get("joint_velocity_limits") or []
        scale = resolve_home_velocity_scale(cell_id, None)
    except Exception:  # noqa: BLE001 — unknown limits: one rad/s
        lim, scale = [], 1.0
    t = 0.0
    for i, (a, b) in enumerate(zip(start, home)):
        v = (float(lim[i]) if i < len(lim) else 1.0) * max(scale, 0.05)
        t = max(t, abs(float(b) - float(a)) / max(v, 1e-3))
    t = max(t * 1.5, 1.0)            # accelerating from and to rest
    return [{"positions": list(map(float, start)), "time_s": 0.0},
            {"positions": list(map(float, home)), "time_s": round(t, 3)}]


def _validate(cell_id: str, legs: dict) -> dict:
    from agents.trajectory.tools.planning_tools import validate_trajectory_set
    fn = getattr(validate_trajectory_set, "func", validate_trajectory_set)
    return fn(cell_id=cell_id, trajectories=legs, suggest_delay=False,
              check_static_obstacles=False)


def _parked(q) -> list:
    return [{"positions": list(q), "time_s": 0.0}]


@tool
def evaluate_program(cell_id: str, plan: dict, check_clearance: bool = True,
                     check_home: bool = True) -> dict:
    """
    What a step plan COSTS: cycle time, per-arm load, idle time, tight steps and
    the return home. Call it on the plan you are about to assemble, and on any
    alternative you are weighing (a different pairing, a handover, a park move) —
    the faster valid program wins.

    Args:
        cell_id: Cell identifier.
        plan:    The plan exactly as ``assemble_behaviour_tree`` takes it:
                 ``{"steps": [{"mode": "PARALLEL"|"SEQUENTIAL", "robots": [
                 {"robot_ns", "operation_id", "hold_duration_s"?, "start_delay_s"?}]}],
                 "home_at_end": bool}``. Trajectories are read from the cache by
                 ``operation_id`` — plan them first.
        check_clearance: Measure each step against the other arms where they
                 stand (the gate's own check) and report the tight ones.
        check_home: Predict whether the arms can return HOME together from where
                 they finish.

    Returns:
        {"cycle_time_s", "arms": {ns: {"busy_s", "idle_s", "operations", "moves"}},
         "imbalance_s", "steps": [{"step", "mode", "duration_s", "arms"}],
         "idle_with_work": [...], "tight_steps": [...], "home": {...},
         "suggestions": [str]}
    """
    from agents.trajectory.tools.behaviour_tree import _arm_home, _serial_steps
    try:
        steps = (plan or {}).get("steps") or []
        arms: dict = {}
        out_steps: list = []
        missing: list = []
        at_pose: dict = {}
        tight: list = []
        warning = None
        cycle = 0.0
        for si, step in enumerate(steps, start=1):
            mode = str(step.get("mode") or "PARALLEL").upper()
            legs, spans = {}, []
            for r in step.get("robots") or []:
                ns, op = r.get("robot_ns"), r.get("operation_id")
                leg = _leg(cell_id, ns, op)
                if not leg:
                    missing.append({"step": si, "robot_ns": ns, "operation_id": op})
                    continue
                hold = float(r.get("hold_duration_s", DEFAULT_HOLD_S) or 0.0)
                delay = float(r.get("start_delay_s", 0.0) or 0.0)
                spans.append(delay + _dur(leg) + hold)
                legs[ns] = leg
                a = arms.setdefault(ns, {"busy_s": 0.0, "operations": 0, "moves": 0})
                a["busy_s"] += _dur(leg) + hold
                from agents.trajectory.tools.trajectory_cache import op_spec
                kind = str(r.get("operation_type")
                           or (op_spec(cell_id, ns, op) or {}).get("operation_type") or "weld")
                a["moves" if kind == "move" else "operations"] += 1
            d = (max(spans) if mode.startswith("PAR") else sum(spans)) if spans else 0.0
            cycle += d
            entry = {"step": si, "mode": mode, "duration_s": round(d, 2), "arms": sorted(legs)}
            if check_clearance and legs:
                for ns in list(at_pose) + list(arms):
                    if ns not in at_pose:
                        home = _arm_home(cell_id, ns)
                        if home:
                            at_pose[ns] = list(home)
                others = {ns: _parked(q) for ns, q in at_pose.items() if ns not in legs}
                if len(legs) + len(others) > 1:
                    res = _validate(cell_id, {**legs, **others})
                    if not res.get("error"):
                        warning = res.get("warning_distance_m", warning)
                        mc = res.get("min_clearance_m")
                        entry["min_clearance_m"] = mc
                        if mc is not None and warning is not None \
                                and mc < warning + TIGHT_MARGIN_M:
                            tight.append({"step": si, "min_clearance_m": mc,
                                          "margin_m": round(mc - warning, 4),
                                          "pair": res.get("worst_pair")})
            for ns, leg in legs.items():
                at_pose[ns] = list(leg[-1]["positions"])
            out_steps.append(entry)

        home = None
        if check_home and (plan or {}).get("home_at_end", True) and len(at_pose) > 1:
            home_legs = {ns: _home_leg(cell_id, ns, q) for ns, q in at_pose.items()}
            home_legs = {ns: l for ns, l in home_legs.items() if l}
            together = _validate(cell_id, home_legs) if len(home_legs) > 1 else {}
            serial_s = sum(_dur(l) for l in home_legs.values())
            par_s = max([_dur(l) for l in home_legs.values()] or [0.0])
            home = {"estimate": "straight joint lines at the cell's home speed",
                    "together": bool(together.get("collision_free")),
                    "min_clearance_m": together.get("min_clearance_m"),
                    "parallel_s": round(par_s, 2), "serial_s": round(serial_s, 2)}
            if not home["together"] and not together.get("error"):
                from agents.trajectory.tools.planning_tools import find_start_delays
                fix = find_start_delays(cell_id, home_legs)
                if fix:
                    stag = max(_dur(l) + fix["start_delays"].get(ns, 0.0)
                               for ns, l in home_legs.items())
                    home.update(staggered_s=round(stag, 2), start_delays=fix["start_delays"])
                home["extra_cost_s"] = round(min(home.get("staggered_s", serial_s),
                                                 serial_s) - par_s, 2)
                home["pair"] = together.get("worst_pair")
            cycle += home.get("staggered_s", serial_s) if not home["together"] else par_s

        for a in arms.values():
            a["busy_s"] = round(a["busy_s"], 2)
            a["idle_s"] = round(max(0.0, cycle - a["busy_s"]), 2)
        busy = [a["busy_s"] for a in arms.values()] or [0.0]
        idle_work = _serial_steps(steps)
        suggestions = []
        if len(busy) > 1 and max(busy) - min(busy) > 0.15 * max(cycle, 1e-9):
            heavy = max(arms, key=lambda ns: arms[ns]["busy_s"])
            suggestions.append(
                f"{heavy} is busy {max(busy) - min(busy):.1f} s longer than the least "
                "loaded arm: the cycle follows the busiest arm. Give a pending weld of "
                f"{heavy} to the other arm (reallocate_operation with its load), or "
                "pair the idle arm's remaining welds with it.")
        if idle_work:
            suggestions.append(
                f"{len(idle_work)} step(s) leave an arm with work ahead standing idle "
                "(idle_with_work): pair a pending weld of that arm into them.")
        if home and not home["together"]:
            suggestions.append(
                f"the arms cannot go home together from where they finish "
                f"(+{home.get('extra_cost_s', 0):.1f} s): end the arm in the way with a "
                "move clear of the other (find_clear_pose), or re-order so the last "
                "welds are apart.")
        if tight:
            suggestions.append(
                f"{len(tight)} step(s) clear the warning band by less than "
                f"{TIGHT_MARGIN_M} m (tight_steps): valid, but runtime tracking error "
                "eats that margin — prefer a pairing with more room when one exists.")
        out = {"success": not missing, "cycle_time_s": round(cycle, 2), "arms": arms,
               "imbalance_s": round(max(busy) - min(busy), 2),
               "steps": out_steps, "idle_with_work": idle_work, "tight_steps": tight,
               "home": home, "suggestions": suggestions}
        if missing:
            out["missing_trajectories"] = missing
            out["error"] = "plan these operations first (no cached trajectory)"
        return out
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def _stroke_of(cell_id: str, ns: str, op) -> Optional[list]:
    """A weld that could NOT be planned yet (its descent was blocked) has no cached
    leg; its working stroke — approach waypoint to weld, at the requested lean — is
    what the other arm must clear. Solved by IK from the recorded request."""
    from agents.trajectory.tools.trajectory_cache import get_last_config, op_spec
    spec = op_spec(cell_id, ns, op) or {}
    t = spec.get("target")
    if isinstance(t, dict):
        t = [t.get("x"), t.get("y"), t.get("z")]
    if not t or None in t:
        return None
    try:
        import numpy as np
        from agents.trajectory.tools.pinocchio import run_ik
        av = np.asarray(spec.get("approach_vector") or [0, 0, -1], float)
        av /= (np.linalg.norm(av) + 1e-9)
        off = float(spec.get("approach_offset_m") or 0.1)
        o = spec.get("orientation") or {}
        ik_ori = ({"qx": float(o.get("x", 0)), "qy": float(o.get("y", 0.707)),
                   "qz": float(o.get("z", 0)), "qw": float(o.get("w", 0.707)),
                   "constrain_orientation": True} if spec.get("constrain_orientation") and o
                  else {})
        seed = get_last_config(cell_id, ns)
        pts = []
        for back in (off, 0.0):
            xyz = [float(t[i]) - av[i] * back for i in range(3)]
            r = run_ik.invoke({"cell_id": cell_id, "robot_ns": ns, "x": xyz[0], "y": xyz[1],
                               "z": xyz[2], **({"warm_start_joints": seed} if seed else {}),
                               **ik_ori})
            if not r.get("success"):
                return None
            seed = r["joint_positions"]
            pts.append(list(r["joint_positions"]))
        return [{"positions": pts[0], "time_s": 0.0}, {"positions": pts[1], "time_s": 1.0}]
    except Exception:  # noqa: BLE001
        return None


@tool
def find_clear_pose(cell_id: str, robot_ns: str, clear_of: list,
                    margin_m: float = 0.05) -> dict:
    """
    ONE pose for ``robot_ns`` that is out of the way of other arms' upcoming legs —
    the answer to ``blocked_by_parked_arm`` (and to an arm idling where the other
    must work), so the arm moves clear once instead of retreating step by step.

    Searched on the way from where the arm stands toward its own HOME (back
    toward its base), nearest first: the first pose whose straight move from the
    current pose is free (joint limits, itself, static obstacles) and which clears
    every listed leg by ``margin_m`` above the cell's warning distance.

    Args:
        cell_id:  Cell identifier.
        robot_ns: The arm to move out of the way.
        clear_of: The legs it must not obstruct, by operation:
                  ``[{"robot_ns": "robot2", "operation_ids": ["W05", "W04"]}]``. A
                  weld that is not planned yet (its descent was blocked) counts as
                  its working stroke, from its recorded request.
        margin_m: Extra clearance above the warning distance (default 0.05 m).

    Returns:
        ``{"success": true, "joints": [...], "fraction_to_home", "min_clearance_m",
          "motion_request": {...}}`` — plan ``motion_request`` with
          ``run_trajectory_planner`` as this arm's next operation (a ``move``).
        ``{"success": false, "reason": ...}`` when not even HOME clears them.
    """
    from agents.trajectory.tools.behaviour_tree import _arm_home
    from agents.trajectory.tools.leg_verdict import refusal
    from agents.trajectory.tools.trajectory_cache import get_last_config
    try:
        here = get_last_config(cell_id, robot_ns) or _arm_home(cell_id, robot_ns)
        home = _arm_home(cell_id, robot_ns)
        if not here or not home:
            return {"success": False, "reason": f"no pose known for {robot_ns}"}
        legs = {}
        for c in clear_of or []:
            ns = c.get("robot_ns")
            for op in c.get("operation_ids") or []:
                leg = _leg(cell_id, ns, op) or _stroke_of(cell_id, ns, op)
                if leg:
                    legs.setdefault(ns, []).append((op, leg))
        if not legs:
            return {"success": False, "reason": "no cached leg to clear (plan them first)"}
        from agents.trajectory.tools.planning_tools import _load_collision_config
        warning = _load_collision_config(cell_id)["warning_distance"]
        need = warning + float(margin_m)
        tried = []
        for k in range(1, 11):
            f = k / 10.0
            q = [float(a) + f * (float(b) - float(a)) for a, b in zip(here, home)]
            move = [{"positions": list(map(float, here)), "time_s": 0.0},
                    {"positions": q, "time_s": 1.0}]
            if refusal(cell_id, robot_ns, move, static=True) is not None:
                tried.append({"fraction_to_home": f, "blocked": "the move itself"})
                continue
            worst = None
            for ns, items in legs.items():
                for op, leg in items:
                    res = _validate(cell_id, {ns: leg, robot_ns: _parked(q)})
                    mc = res.get("min_clearance_m")
                    if res.get("error") or mc is None:
                        worst = -1.0
                        break
                    worst = mc if worst is None else min(worst, mc)
            tried.append({"fraction_to_home": f, "min_clearance_m": worst})
            if worst is not None and worst >= need:
                from agents.trajectory.tools.robot_kinematics import (
                    _fk_ee, _load_pinocchio_model, _load_robot_config,
                )
                from agents.trajectory.tools.weld_orientation import tcp_orientation
                import numpy as np
                model, data = _load_pinocchio_model(cell_id, robot_ns)
                rcfg = _load_robot_config(cell_id, robot_ns)
                xyz = _fk_ee(model, data, np.asarray(q[: model.nq]),
                             np.asarray(rcfg["base_position"], float),
                             float(rcfg.get("base_orientation_yaw", 0.0)))
                ori = tcp_orientation(cell_id, robot_ns, q)
                req = {"robot_ns": robot_ns, "operation_type": "move",
                       "operation_id": f"{robot_ns.upper().replace('ROBOT', 'R')}_CLEAR",
                       "target": {"x": round(float(xyz[0]), 4), "y": round(float(xyz[1]), 4),
                                  "z": round(float(xyz[2]), 4)},
                       "approach": False}
                if ori:
                    req.update(orientation=ori, constrain_orientation=True)
                return {"success": True, "joints": [round(v, 5) for v in q],
                        "fraction_to_home": f, "min_clearance_m": round(worst, 4),
                        "required_m": round(need, 4), "motion_request": req,
                        "instruction": (
                            "Plan motion_request as ONE move of this arm (rename "
                            "operation_id if it is taken), in its own step before the "
                            "blocked weld — or in the same step as the other arm's "
                            "previous operation, if evaluate_program shows that is "
                            "faster.")}
        return {"success": False, "reason": (
            f"no pose between {robot_ns}'s current pose and its HOME clears those legs "
            f"by {need:.3f} m: home it first, or re-order the other arm's welds"),
            "tried": tried}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "reason": f"{type(exc).__name__}: {exc}"}


def _tree_duration(node: dict) -> float:
    """Wall-clock length of an internal-shape (pre-BT.CPP) node."""
    t = str(node.get("type") or "")
    kids = node.get("children") or []
    if t in ("EXECUTE", "ExecuteTrajectory", "HOME", "MoveHome"):
        leg = node.get("precomputed_trajectory") or []
        return (_dur(leg) if leg else 7.0) + float(node.get("hold_duration_s", 0.0) or 0.0)
    if t == "Delay":
        return (node.get("ports") or {}).get("delay_msec", 0) / 1000.0 + \
            sum(_tree_duration(k) for k in kids)
    if t in ("PARALLEL", "Parallel"):
        return max([_tree_duration(k) for k in kids] or [0.0])
    return sum(_tree_duration(k) for k in kids)


def program_cost(cell_id: str, plan: dict, step_reports: list,
                 home_node: Optional[dict], home_report: Optional[dict]) -> dict:
    """What an ASSEMBLED program costs — attached to every accepted tree, so the
    cycle time, the balance and the tight spots are in front of the agent without
    it having to ask (it often does not). Uses the gates' own clearances and the
    return home the assembler actually planned; plans nothing."""
    ev = getattr(evaluate_program, "func", evaluate_program)(
        cell_id, {**(plan or {}), "home_at_end": False},
        check_clearance=False, check_home=False)
    if not ev.get("success"):
        return {"error": ev.get("error")}
    tight = []
    for r in step_reports or []:
        mc = (r.get("parallel") or {}).get("min_clearance_m")
        wd = (r.get("parallel") or {}).get("warning_distance_m")
        if mc is not None and wd is not None and mc < wd + TIGHT_MARGIN_M:
            tight.append({"step": r.get("step"), "min_clearance_m": mc,
                          "margin_m": round(mc - wd, 4)})
    home_s = _tree_duration(home_node) if home_node else 0.0
    cycle = ev["cycle_time_s"] + home_s
    suggestions = [x for x in ev.get("suggestions") or [] if "tight" not in x]
    if tight:
        suggestions.append(
            f"{len(tight)} step(s) clear the warning band by less than {TIGHT_MARGIN_M} m: "
            "valid, but runtime tracking error eats that margin.")
    if home_report and home_report.get("mode") == "SEQUENTIAL":
        suggestions.append(
            "the return home runs one arm at a time: end the arm in the way with a move "
            "clear of the other (find_clear_pose) to bring them home together.")
    for a in ev["arms"].values():
        a["idle_s"] = round(max(0.0, cycle - a["busy_s"]), 2)
    return {"cycle_time_s": round(cycle, 2), "home_s": round(home_s, 2),
            "arms": ev["arms"], "imbalance_s": ev["imbalance_s"],
            "idle_with_work": ev["idle_with_work"], "tight_steps": tight,
            "suggestions": suggestions}
