"""Build the home-reset behaviour tree used before a replay.

A bare ``MoveHome`` node carries no path, so the executor falls back to a single
HOME waypoint and the arm interpolates straight there — through the fixtures, the
workpiece, or the other arm. Driving every arm home in Parallel that way is the
riskiest motion in a replay: it is the one nothing has checked, and it runs
BEFORE the carefully validated plan.

So the reset is planned like any other motion, and planned for all arms TOGETHER
so each is routed around the accumulated swept volume of the others.

And then it is GATED like any other multi-arm step. Planning the set is not the
same as checking it: ``plan_home_trajectories`` returns the best paths it can
find and gives an arm up when it cannot route one, and the give-up case is
precisely the dangerous one. Measured on ``welding_cell_2_small`` out of the pose a
prior e-stop leaves the cell in — robot1 at
``[-2.1956,-1.4599,1.3817,3.1797,-1.6873,-2.1900]``, robot2 at
``[1.1381,-1.0962,0.7677,-1.9782,-0.8277,1.6803]``, already only 0.1385 m apart
against a 0.2 m warning band — robot2 could not be routed at all, and robot1's
planned return ran in Parallel with robot2's BLIND straight line. That blind line
closes to **-0.1063 m**: the arms overlap. One arm at a time, robot2 first, never
drops below the 0.1385 m they already stand at and opens to 1.4601 m once robot2
is home. A safe order existed and nothing looked for it, so the monitor e-stopped
the reset before the validated plan ever started.

``_gate_multi_arm_step`` is the one owner of "may these arms move at once?" — the
same ladder the assembler applies to a weld step and to the planned return home.
A blind arm is handed to it as the straight line the executor will actually drive,
because that motion is predictable and therefore checkable; what it must never be
is invisible.

This lives apart from ``mcp_server`` deliberately. Tree building needs live robot
state and the planner; it does not need the LLM agent, and importing
``agents.cell.agent`` would drag LangGraph in behind it — enough to make this
logic untestable without the whole model stack.
"""
from __future__ import annotations

from agents.cell.tools.cell_state import get_all_robot_states

try:
    from agents.trajectory.tools.behaviour_tree import _gate_multi_arm_step
except Exception:  # pragma: no cover - trajectory stack absent
    # No gate available is an AVAILABILITY failure, and it is handled the way the
    # gate handles its own: it refuses nothing. Never a reason to break a replay.
    _gate_multi_arm_step = None


def _blind_home_leg(cell_id: str, robot_ns: str, start: list) -> "list | None":
    """The motion the executor performs for a ``MoveHome`` with no planned path.

    It dispatches ONE waypoint — the cell's HOME — so the arm interpolates
    straight there in joint space. Two waypoints express that exactly: the
    validator interpolates linearly between them, which is the same line. Both
    endpoints come from the executor's own sources (``home_pose`` and
    ``ExecConfig.home_return_time_s``) so this is a description of the fallback,
    never a second definition of it.

    Returns None when the arm's pose is unknown — a start we would have to guess
    makes the whole leg a guess, and a guess is not a measurement.
    """
    if not start:
        return None
    from agents.cell.bt_executor import HOME_JOINTS, ExecConfig
    from agents.home_pose import home_pose

    home = list(home_pose(cell_id, robot_ns, HOME_JOINTS))
    return [{"positions": list(start), "time_s": 0.0},
            {"positions": home,
             "time_s": float(ExecConfig.for_cell(cell_id).home_return_time_s)}]


def home_reset_tree(cell_id: str, robots: list) -> dict:
    """A home-reset tree whose returns are PLANNED, then GATED as a set.

    Each arm's current pose is read from live state and all arms are planned in
    one coordinated call (``plan_home_trajectories``). The resulting set — with a
    blind straight line standing in for any arm that could not be planned — goes
    through the same inter-robot gate a parallel weld step gets. Its verdict picks
    the node: clear ⇒ ``Parallel``; not clear ⇒ ``Sequence`` in an order where
    each mover clears the others where they stand; no safe order ⇒ the arms it
    blocked are left where they are and the operator is told.

    Best-effort by design, in the availability sense only: an arm that cannot be
    planned keeps the plain node and the executor's blind fallback, and a gate
    that cannot run (no Pinocchio, no model) refuses nothing. What is no longer
    best-effort is safety — a pair the gate MEASURES as unsafe is not driven in
    parallel, and an arm it cannot clear at all is not driven.
    """
    planned: dict = {}
    starts: dict = {}

    try:
        states = get_all_robot_states.invoke({"cell_id": cell_id}).get("robots", {})
        starts = {
            r: list(states[r]["joint_positions"])
            for r in robots
            if (states.get(r) or {}).get("joint_positions")
        }
        if starts:
            from agents.trajectory.tools.planner_backend import plan_home_trajectories
            planned = plan_home_trajectories(cell_id, starts) or {}
    except Exception as e:
        print(f"[replay] home-reset planning unavailable ({e}) — "
              f"falling back to blind HOME for all arms", flush=True)

    # An arm whose planned return was refused (no on-axis way out of its seam, or
    # a leg that failed a check) has no planned leg, and the blind line would be
    # that same unchecked motion: it is not moved. The record must be for the pose
    # the arm is in now (home_refusal compares the start).
    stuck: list = []
    try:
        from agents.trajectory.tools.planner_backend import home_refusal
        stuck = [r for r in robots
                 if r in starts and not planned.get(r) and home_refusal(cell_id, r, starts[r])]
    except Exception:  # noqa: BLE001 — no record is not a refusal
        stuck = []
    if stuck:
        print(f"[replay] home reset: {', '.join(stuck)} have no safe planned return "
              f"(lift-off blocked, or the planned leg failed a check) — NOT moving "
              f"them. Jog them clear, then re-run the reset.", flush=True)
        robots = [r for r in robots if r not in stuck]

    nodes: dict = {}
    for r in robots:
        node = {"type": "MoveHome", "ports": {"robot_ns": r}}
        if planned.get(r):
            node["ports"]["trajectory"] = planned[r]
        nodes[r] = node

    blind = [r for r in robots if not planned.get(r)]
    if blind:
        print(f"[replay] home reset: NO planned path for {', '.join(blind)} — "
              f"using the executor's blind single-waypoint home for these arms",
              flush=True)
    if planned:
        print(f"[replay] home reset: planned collision-free return for "
              f"{', '.join(sorted(planned))}", flush=True)

    # What each arm will ACTUALLY do — the planned path, or the line the executor
    # drives when there is none. An arm whose pose is unknown gets no leg: its
    # motion cannot be predicted, so it cannot be measured against anything.
    legs: dict = {}
    for r in robots:
        leg = planned.get(r) or _blind_home_leg(cell_id, r, starts.get(r))
        if leg:
            legs[r] = leg
    unmeasurable = [r for r in robots if r not in legs]
    if unmeasurable and len(robots) > 1:
        print(f"[replay] home reset: {', '.join(unmeasurable)} cannot be checked "
              f"(no live pose) — the arms are homed ONE AT A TIME rather than "
              f"alongside a motion nothing measured", flush=True)

    unhomed: list = list(stuck)
    mode, order = "PARALLEL", list(robots)
    if (len(legs) > 1 or (stuck and legs)) and _gate_multi_arm_step is not None:
        report = _gate_multi_arm_step(
            cell_id, legs,
            # An unmeasurable arm still moves, so the set is not verifiable as a
            # whole: separate the arms in TIME instead of trusting it.
            allow_parallel=not unmeasurable,
            # The arms left standing are still in the cell: the movers are gated
            # against where they stand.
            **({"bystanders": {r: starts[r] for r in stuck}} if stuck else {}))
        verdict = report.get("mode")
        gate_order = [r for r in (report.get("order") or []) if r in nodes]
        if verdict is None:
            # No order runs these arms clear of each other. Drive the ones the
            # gate did clear, leave the rest standing — the alternative is the
            # fail-open blind motion this gate exists to stop.
            blocked = report.get("blocked_arms") or [r for r in legs
                                                     if r not in gate_order]
            print(f"[replay] home reset: {', '.join(sorted(blocked))} cannot be "
                  f"homed clear of the other arms in ANY order — NOT moving "
                  f"them. The arms are inside the cell's warning band where they "
                  f"stand; jog one arm clear by hand, or re-run the reset once "
                  f"the cleared arms are home.", flush=True)
            mode, order = "SEQUENTIAL", gate_order
            # Every arm the gate did not place stays where it is and is reported
            # — the blocked ones, and any it never reached (a leg fault or a
            # broken check stops the gate before it orders the rest).
            unhomed += [r for r in [*blocked, *nodes] if r not in gate_order
                        and r not in unhomed]
        elif str(verdict).startswith("SEQ"):
            mode = "SEQUENTIAL"
            # Arms the gate did not rank keep their declared order behind it.
            order = gate_order + [r for r in robots if r not in gate_order]
        else:
            mode = "PARALLEL"
    elif unmeasurable and len(robots) > 1:
        mode = "SEQUENTIAL"

    children = [nodes[r] for r in order if r in nodes]
    root_type = "Sequence" if mode == "SEQUENTIAL" else "Parallel"
    root: dict = {"type": root_type, "children": children}
    if root_type == "Parallel":
        root["ports"] = {"success_count": -1, "failure_count": 1}
    else:
        root["ports"] = {}

    # ``unhomed``: the arms this reset leaves where they stand. A program must not
    # start while any is listed — its first leg starts at HOME, and the executor
    # would drive that arm there from wherever it stands, unchecked.
    return {"tree_id": f"home_reset_{cell_id}", "cell_id": cell_id, "root": root,
            "unhomed": sorted(unhomed)}
