"""What every backend does to a leg besides searching for it — one implementation.

A backend's job is the SEARCH: a collision-free joint path from where the arm
stands to the approach waypoint of its target. Everything around that search is
the same for every backend and lives here, so a backend does not import another
backend's private helpers (the compound and RRT backends used to reach into
``backends/ompl/planner.py`` for these):

  * the on-axis lift-off out of the seam the arm starts in
    (:func:`cartesian_retreat`) and the straight terminal descent into the weld
    (:func:`cartesian_descent`, :func:`descent_failure` for its refusal);
  * the tool-rise measure of a path and the cap on an emitted one
    (:func:`path_tool_rise`, :func:`trajectory_tool_rise`,
    :func:`emitted_rise_limit`; the policy is ``path_metrics._path_rise_policy``);
  * the arm's acceleration limits for timing (:func:`accel_limits`), its HOME
    (:func:`home_config`) and where a leg ends (:func:`trajectory_end`).

The platform then checks whatever a backend returns (``leg_verdict``, the gates);
see ``docs/creating-a-backend.md``. Moved verbatim from the OMPL backend, which
keeps its old private names as aliases.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

def path_tool_rise(waypoints, model, data, base, yaw) -> float:
    """How far above BOTH endpoints this path lifts the tool, in metres.

    The measured quantity behind every height complaint on this cell: "the return
    of one of the arms is going up too much", "the penultimate weld is going up too
    much", "the start of one of the robot arms is going up too much". Each was a
    transit that climbed over the cell and came back down, so the excursion above
    the endpoints — not absolute height, and not path length — is the defect.

    Measuring it relative to ``max(z_start, z_end)`` is deliberate: a leg that
    simply moves to a HIGHER weld has to gain height, and that is not an excursion.
    Such a leg scores ~0 and is accepted immediately, which is the point.

    Costs one FK per waypoint. A simplified RRTConnect path carries about three
    states, so this is ~3 FK calls per candidate against a multi-second solve —
    the reason the cheap proxy is no longer worth its inaccuracy.
    """
    from agents.trajectory.tools.robot_kinematics import _fk_ee

    if len(waypoints) < 2:
        return 0.0
    nq = model.nq
    z = [float(_fk_ee(model, data, np.asarray(q, float)[:nq], base, yaw)[2])
         for q in waypoints]
    return max(0.0, max(z) - max(z[0], z[-1]))

def emitted_rise_limit() -> float:
    """The tool rise at which an EMITTED trajectory is refused, in metres.

    A different number from :func:`_path_rise_policy`, and deliberately so.
    0.15 m is a *preference*: candidate selection keeps solving while it is
    exceeded, then ships the best it found. This is a *hard refusal*, and a hard
    refusal needs headroom that a preference does not — a false positive here does
    not cost a few extra solves, it fails the operation.

    0.35 m by default, placed in the empty half of the same measured gap: on
    ``welding_cell_2_small`` the tallest leg nobody objected to rose 0.131 m, and the
    shortest leg an operator did object to rose 0.585 m. Nothing was ever measured
    between those two, so 0.35 m refuses every complained-about leg with ~2.7x
    margin over every accepted one.

    ``OMPL_MAX_EMITTED_RISE_M=0`` disables the refusal, for a cell whose fixtures
    genuinely require arcing over a workpiece.
    """
    try:
        limit = float(os.environ.get("OMPL_MAX_EMITTED_RISE_M", "0.35") or 0.35)
    except ValueError:
        limit = 0.35
    return max(0.0, limit)

def trajectory_tool_rise(model, traj, base, yaw) -> float:
    """Tool rise of a TIMED trajectory — the array that lands on the blackboard.

    :func:`path_tool_rise` scores planner states during the search; this measures
    what actually ships, after time parameterisation has interpolated those states
    into the points the controller will stream. Same definition, later object: the
    resolution fix taught us to measure the artefact, not its ancestor.

    Returns 0.0 when FK is unusable — an unmeasurable rise must not be reported as
    an excessive one.
    """
    if not traj or len(traj) < 2:
        return 0.0
    try:
        return path_tool_rise(
            [np.asarray(p["positions"], float)[:model.nq] for p in traj],
            model, model.createData(), base, yaw,
        )
    except Exception:
        return 0.0

def cartesian_descent(
    model, ik_tool, cell_id: str, robot_ns: str,
    via_xyz, goal_xyz, ik_ori: dict,
    start_q, base, yaw,
    obstacle_spheres: list, static_obstacles: list,
    sphere_radius: float, tool_cfg, n_steps: int = 6,
    sphere_radii=None, extra_margin: float = 0.0,
    why: "Optional[dict]" = None,
) -> "Optional[list[np.ndarray]]":
    """Straight-line Cartesian descent from the approach waypoint INTO the point.

    A joint-space OMPL segment reaches the goal config but the end-effector
    arrives on whatever curved path the joints interpolate — so the tool enters
    the weld LATERALLY, which fails the on-axis `validate_approach` check and is
    physically wrong for a welding stroke. This instead interpolates the EE
    position in a straight line from ``via_xyz`` to ``goal_xyz`` (i.e. down the
    approach vector), solving full-pose IK at the required lean for each step and
    collision-checking it. Returns the joint waypoints (ending at the goal), or
    ``None`` if any step is unreachable or collides — in which case the on-axis
    descent is genuinely blocked and the caller reports goal_in_collision.

    ``why``, when given, is filled with what stopped it — ``{"cause": "ik" |
    "collision", "step", "collision": info}`` — so the caller can tell an arm that
    folds into ITSELF on the stroke from one that hits an obstacle.
    """
    from agents.trajectory.tools.robot_kinematics import (
        _config_collision_info, _config_is_free)
    col_data = model.createData()
    via = np.asarray(via_xyz, float)
    goal = np.asarray(goal_xyz, float)
    warm = list(np.asarray(start_q, float)[:model.nq])
    wps: list = []
    for i in range(1, n_steps + 1):
        a = i / n_steps
        p = via * (1.0 - a) + goal * a
        ik = ik_tool.func(
            cell_id, robot_ns, float(p[0]), float(p[1]), float(p[2]),
            warm_start_joints=warm, **ik_ori,
        )
        if not ik.get("success"):
            if why is not None:
                why.update(cause="ik", step=i, steps=n_steps)
            return None
        q = np.asarray(ik["joint_positions"][:model.nq], float)
        if not _config_is_free(model, col_data, q, base, yaw,
                               obstacle_spheres, static_obstacles,
                               sphere_radius=sphere_radius, tool_cfg=tool_cfg,
                               sphere_radii=sphere_radii, extra_margin=extra_margin):
            if why is not None:
                why.update(cause="collision", step=i, steps=n_steps,
                           collision=_config_collision_info(
                               model, col_data, q, base, yaw, obstacle_spheres,
                               static_obstacles, sphere_radius=sphere_radius,
                               sphere_radii=sphere_radii, extra_margin=extra_margin))
            return None
        wps.append(q)
        warm = ik["joint_positions"]
    return wps

def descent_failure(why: dict) -> dict:
    """The failure for a blocked straight descent, named by what blocked it."""
    info = why.get("collision") or {}
    at = f" at step {why.get('step')}/{why.get('steps')} of the stroke" if why.get("step") else ""
    if why.get("cause") == "collision" and info.get("type") == "self":
        return {"reason_code": "approach_self_collision", "collision": info,
                "reason": (f"the straight on-axis descent folds the arm into ITSELF{at}: "
                           f"{info.get('object_id')} (signed distance "
                           f"{info.get('clearance_m')} m). The weld pose may be clear but "
                           f"the stroke into it is not. The tool's spin and tilt were "
                           f"already searched (orientation_search) — a hand-set "
                           f"tool_spin_deg cannot do better against the arm itself: use a "
                           f"shorter approach_offset_m (see suggestion), or another arm.")}
    if why.get("cause") == "ik":
        return {"reason_code": "approach_unreachable",
                "reason": (f"no IK solution at the required lean{at} of the straight "
                           f"descent (the tool's spin and tilt were already searched): use "
                           f"a shorter approach_offset_m (see suggestion), or another arm.")}
    out = {"reason_code": "goal_in_collision",
           "reason": ("the straight on-axis descent to the weld is blocked "
                      f"or unreachable at the required lean{at}")}
    if info:
        out["collision"] = info
    return out

def cartesian_retreat(
    model, ik_tool, cell_id: str, robot_ns: str,
    seam_xyz, retreat_xyz, ik_ori: dict,
    start_q, base, yaw,
    obstacle_spheres: list, static_obstacles: list,
    sphere_radius: float, tool_cfg, n_steps: int = 6,
    sphere_radii=None, extra_margin: float = 0.0,
) -> "Optional[list[np.ndarray]]":
    """Straight-line Cartesian LIFT-OFF from a weld back out to its standoff.

    The mirror image of :func:`cartesian_descent`. A weld's trajectory ends ON
    the seam (that is where the dwell happens and what ``validate_approach``
    measures), so the arm is still in contact when the next operation begins. If
    the next motion starts with a joint-space transit, the tool is dragged
    SIDEWAYS out of the seam — physically wrong, and it shows up as the tool
    rotating/scraping across the workpiece.

    This retracts along the SAME approach vector the arm came in on, holding the
    weld's OWN orientation (``ik_ori``), so entry and exit are symmetric. It is
    prepended to the NEXT operation rather than appended to the weld, which keeps
    the weld trajectory ending exactly on the seam.

    Returns joint waypoints ending at the standoff, or ``None`` if the lift-off is
    blocked/unreachable — in which case the caller falls back to planning straight
    from the seam (degraded, but never a hard failure of the whole plan).
    """
    from agents.trajectory.tools.robot_kinematics import _config_is_free
    col_data = model.createData()
    seam = np.asarray(seam_xyz, float)
    out = np.asarray(retreat_xyz, float)
    warm = list(np.asarray(start_q, float)[:model.nq])
    wps: list = []
    for i in range(1, n_steps + 1):
        a = i / n_steps
        p = seam * (1.0 - a) + out * a
        ik = ik_tool.func(
            cell_id, robot_ns, float(p[0]), float(p[1]), float(p[2]),
            warm_start_joints=warm, **ik_ori,
        )
        if not ik.get("success"):
            return None
        q = np.asarray(ik["joint_positions"][:model.nq], float)
        if not _config_is_free(model, col_data, q, base, yaw,
                               obstacle_spheres, static_obstacles,
                               sphere_radius=sphere_radius, tool_cfg=tool_cfg,
                               sphere_radii=sphere_radii, extra_margin=extra_margin):
            return None
        wps.append(q)
        warm = ik["joint_positions"]
    return wps

def accel_limits(cell_id: str, rcfg: dict):
    """The robot's acceleration limits, per the cell (see
    ``robot_kinematics.joint_acceleration_limits``)."""
    from agents.trajectory.tools.robot_kinematics import joint_acceleration_limits
    return joint_acceleration_limits(cell_id, rcfg)

def home_config(cell_id: str, robot_ns: str, fallback: Sequence[float]) -> list:
    """The arm's HOME configuration, from ``robots.yaml`` when it declares one.

    A cell may park its arms somewhere other than the module default, and the
    default is what the executor's blind fallback would drive to. Reading the
    cell's own ``home_pose_deg`` keeps the PLANNED home and the cell's declared
    home the same pose.

    Thin wrapper over ``agents.home_pose.home_pose`` — kept because it is the
    name the planner and its tests already use, but no longer a second
    implementation of the lookup.
    """
    from agents.home_pose import home_pose

    return home_pose(cell_id, robot_ns, fallback)

def trajectory_end(traj: Sequence, fallback) -> list:
    """The joints a timed trajectory actually ends at — what the next leg must
    start from. ``fallback`` (the IK goal) only when the trajectory is empty."""
    if traj:
        q = traj[-1].get("positions") if isinstance(traj[-1], dict) else None
        if q:
            return [float(v) for v in q]
    return [float(v) for v in np.asarray(fallback, float).tolist()]


def approach_ik(ik, cell_id: str, robot_ns: str, xyz, ik_ori: dict, seeds) -> "Optional[list]":
    """The joint solution at the approach waypoint (``xyz``, the weld's own lean),
    trying each seed in turn — the arm's chained start first, then the WELD's
    solution: the waypoint is a short way up the weld's own axis, so the weld's
    configuration is the natural start for it. A single seed used to be all there
    was; when it failed, the leg lost its straight descent and drove into the weld
    in joint space, off its axis. None when no seed converges."""
    for seed in seeds:
        if seed is None:
            continue
        out = ik.invoke({"cell_id": cell_id, "robot_ns": robot_ns, "x": float(xyz[0]),
                         "y": float(xyz[1]), "z": float(xyz[2]),
                         "warm_start_joints": [float(v) for v in seed], **(ik_ori or {})})
        if out.get("success"):
            return list(out["joint_positions"])
    return None


def approach_waypoint_failure(robot_ns: str, offset_m: float) -> dict:
    """A constrained weld whose approach waypoint has no solution at the weld's lean.

    Without it there is no straight on-axis descent, and a joint-space entry into
    the weld fails the working-stroke check at assembly (``approach_mismatch``)
    anyway — so it is refused here, where a shorter stroke can still fix it."""
    shorter = round(max(0.02, 0.5 * float(offset_m)), 3)
    return {"robot_ns": robot_ns, "reason_code": "approach_unreachable",
            "suggested_approach_offset_m": shorter,
            "error": (f"no IK solution at the approach waypoint {offset_m:.3f} m up the "
                      f"weld's axis at the required lean (the arm's start and the "
                      f"weld's own solution were both tried): the straight on-axis "
                      f"descent cannot start there. Try approach_offset_m={shorter}, "
                      f"or another arm.")}


def blocking_arms(descent_ok, bystanders_by_arm: dict) -> list:
    """Which parked arms, each ON ITS OWN, block a stroke — ``descent_ok(spheres)``
    re-runs the stroke against those spheres only (no static obstacles). Empty
    when no single arm does (then it is the fixtures, or several together)."""
    return sorted(ns for ns, spheres in (bystanders_by_arm or {}).items()
                  if spheres and not descent_ok(spheres))


def blocked_by_parked_failure(robot_ns: str, arms: list, what: str) -> dict:
    """A stroke blocked by another arm standing where its last leg left it.

    Named, because the cure is to move THAT arm out first — not to give the weld to
    another arm (the weld is fine; the space above it is occupied)."""
    who = ", ".join(arms)
    return {"robot_ns": robot_ns, "reason_code": "blocked_by_parked_arm",
            "blocked_by": list(arms),
            "error": (f"the {what} is blocked by {who}, standing where its last leg "
                      f"left it. The weld is reachable: first plan a 'move' that takes "
                      f"{who} clear (toward its own base, raised), or move it in the "
                      f"same step, then plan this weld again.")}
