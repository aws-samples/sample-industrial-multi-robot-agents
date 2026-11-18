"""
LEGACY backend — hand-rolled single-tree Space-Time RRT.

NON-CONFORMING (see ``planner_backend.NON_CONFORMING_BACKENDS``) and FROZEN: it
does not couple robots to each other and drops the tool-orientation constraint.
It is kept only to the platform's minimum — the shared lift-off out of a seam,
per-request stroke and stroke speed, the emitted tool-rise cap and the emitted-
sample collision check — and gets no new features; the platform checks its legs
like any backend's (``leg_verdict``). Production planning uses
``ompl_planner``; this is kept for comparison and is reachable ONLY when a task
explicitly selects ``backend="rrt"``, which stamps the result
``non_conforming_backend: true``.

This module is now JUST that algorithm. The geometry it validates against
(``robot_kinematics``), the task scheduling (``task_allocation``) and the LLM
``@tool`` surface (``planning_tools``) were extracted, because sharing them from
here meant every consumer — the OMPL backend, the runtime collision monitor, the
MRTA scheduler — imported a legacy planner to reach code that was never
RRT-specific. Nothing outside this file should import from it.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from agents.trajectory.tools.robot_kinematics import (
    _config_is_free,
    _endpoint_collision_failure,
    _load_pinocchio_model,
    _load_robot_config,
    _target_obstacle_suggestion,
)
from agents.trajectory.tools.time_parameterization import _hermite_spline, _shortcut_path
from agents.trajectory.tools.trajectory_cache import get_last_config, set_last_config



def _timed_parts(parts: list) -> list:
    """``[(waypoints, at_rest, velocity_scale, phase), ...]`` timed with the
    Hermite scheme, one after the other on one clock, each sample tagged."""
    out: list = []
    t0 = 0.0
    for wps, at_rest, vs, phase in parts:
        if len(wps) < 2:
            continue
        seg = _hermite_spline(wps, dt=0.08, at_rest=at_rest)
        scale = vs if vs and vs > 0 else 1.0
        first = 1 if out else 0                 # the junction sample once
        for pt in seg[first:]:
            out.append({"positions": pt["positions"],
                        "time_s": round(t0 + pt["time_s"] / scale, 4), "phase": phase})
        t0 = out[-1]["time_s"]
    return out


# ─── Space-Time RRT ───────────────────────────────────────────────────────────

def _rrt_single(
    model, data, start_q: np.ndarray, goal_q: np.ndarray,
    jl: np.ndarray, ju: np.ndarray,
    obstacle_spheres: list[tuple[np.ndarray, float]],
    base: np.ndarray, yaw: float,
    max_iter: int = 3000, step_size: float = 0.15,
    static_obstacles: list[dict] | None = None,
    time_budget_s: float = 8.0,
    sphere_radius: float = 0.12,
    sphere_radii: "Optional[np.ndarray]" = None,
    extra_margin: float = 0.0,
) -> Optional[list[np.ndarray]]:
    """Single-robot RRT in joint space with sphere obstacle avoidance.

    Checks both moving robot-sphere obstacles and static cell obstacles
    loaded from obstacles.yaml. Aborts after ``time_budget_s`` so an
    unsolvable query fails fast instead of burning the whole iteration budget.

    Clearance vs a box = per-sphere radius + per-box margin + ``extra_margin``.
    ``sphere_radii`` carries the cell's per-body radii (arm vs tool) and
    ``sphere_radius`` is the fallback for spheres it does not cover; both come
    from the shared builder in ``robot_kinematics`` so this backend, the OMPL
    backend and the runtime monitor occupy one collision model. ``extra_margin``
    is the speed allowance — the only term that legitimately varies between a
    fast transit and a slow working stroke.
    """
    import time as _time
    import pinocchio as pin

    nq = model.nq
    tree = [start_q.copy()]
    parent = [-1]
    static_obstacles = static_obstacles or []
    _deadline = _time.monotonic() + time_budget_s

    # Reuse a single Data buffer for all collision checks — allocating a fresh
    # pin.createData() per sample was the dominant cost (called up to max_iter×).
    col_data = model.createData()

    def sample():
        if np.random.random() < 0.15:
            return goal_q.copy()
        return np.random.uniform(jl[:nq], ju[:nq])

    def is_free(q):
        # Delegates to the shared predicate so the RRT and OMPL backends apply
        # identical collision semantics (single source of truth).
        return _config_is_free(
            model, col_data, q, base, yaw,
            obstacle_spheres, static_obstacles, sphere_radius,
            sphere_radii=sphere_radii, extra_margin=extra_margin,
        )

    for _it in range(max_iter):
        # Wall-clock abort: fail fast on unsolvable queries (checked every 64 iters).
        if (_it & 63) == 0 and _time.monotonic() > _deadline:
            return None
        q_rand  = sample()
        # Vectorised nearest-neighbour over the current tree.
        tree_arr = np.asarray(tree)
        nearest = int(np.argmin(np.linalg.norm(tree_arr - q_rand, axis=1)))
        q_near  = tree[nearest]
        diff    = q_rand - q_near
        norm    = float(np.linalg.norm(diff))
        if norm < 1e-9:
            continue
        q_new   = q_near + min(step_size, norm) / norm * diff
        q_new   = np.clip(q_new, jl[:nq], ju[:nq])

        if not is_free(q_new):
            continue

        tree.append(q_new)
        parent.append(nearest)

        if float(np.linalg.norm(q_new - goal_q)) < step_size * 1.5:
            # Reconstruct path
            path, idx = [goal_q.copy()], len(tree) - 1
            while idx != -1:
                path.append(tree[idx].copy())
                idx = parent[idx]
            path.reverse()
            return path

    return None  # timeout



def plan_trajectories_rrt(
    cell_id: str,
    motion_requests: list[dict],
    approach_offset_m: float = 0.1,
    velocity_scale: Optional[float] = None,
    max_iterations: int = 3000,
    **_ignored,
) -> dict:
    """Legacy hand-rolled Space-Time RRT backend. See NON_CONFORMING_BACKENDS.

    Kept selectable for comparison against OMPL. It does NOT couple robots to
    each other, does NOT enforce the tool-orientation constraint, and does NOT
    apply the two-phase speed/clearance split — do not use it to gate a real
    cell. Dispatched only via ``planner_backend.dispatch_plan``.
    """
    max_rrt_iterations = max_iterations
    try:
        import pinocchio as pin

        t0 = time.monotonic()
        trajectories: dict[str, list[dict]] = {}
        failures: list[dict] = []

        # Static obstacles and the cell clearance policy, both read through the
        # shared owner so this backend cannot drift from OMPL or the runtime
        # monitor. Clearance vs a box = per-body radius (arm / tool) + per-box
        # margin + speed allowance.
        from agents.trajectory.tools.robot_kinematics import (
            _cell_sphere_radii, _load_clearance_policy, _speed_margin,
            load_static_obstacles, resolve_velocity_scale,
        )
        velocity_scale = resolve_velocity_scale(cell_id, velocity_scale)
        policy = _load_clearance_policy(cell_id)
        normal_radius = policy["normal"]
        static_obstacles: list[dict] = load_static_obstacles(cell_id)
        # The speed allowance for this whole call's transit motion.
        transit_margin = _speed_margin(cell_id, velocity_scale)

        # Pre-load models once per robot
        models: dict[str, tuple] = {}
        configs: dict[str, dict] = {}
        for req in motion_requests:
            ns = req["robot_ns"]
            if ns not in models:
                try:
                    models[ns]  = _load_pinocchio_model(cell_id, ns)
                    configs[ns] = _load_robot_config(cell_id, ns)
                except Exception as e:
                    failures.append({"robot_ns": ns, "error": str(e)})

        for req in motion_requests:
            ns = req["robot_ns"]
            if ns not in models:
                continue

            model, _data = models[ns]
            data   = model.createData()
            rcfg   = configs[ns]
            base   = np.array(rcfg["base_position"])
            yaw    = float(rcfg.get("base_orientation_yaw", 0.0))
            from agents.trajectory.tools.joint_limits import planning_box
            jl, ju = planning_box(rcfg, model)
            from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose
            HOME   = np.array(home_pose(cell_id, ns, HOME_JOINTS_FALLBACK), float)

            tgt = req["target"]
            x, y, z = float(tgt["x"]), float(tgt["y"]), float(tgt["z"])
            av  = np.array(req.get("approach_vector", [0, 0, -1]), dtype=float)
            av  /= (np.linalg.norm(av) + 1e-9)

            # Approach waypoint — at the request's own stroke, as in every backend.
            try:
                req_offset = float(req.get("approach_offset_m", approach_offset_m))
            except (TypeError, ValueError):
                req_offset = approach_offset_m
            approach_x = x - av[0] * req_offset
            approach_y = y - av[1] * req_offset
            approach_z = z - av[2] * req_offset

            # IK for approach + target
            from agents.trajectory.tools.pinocchio import run_ik as _ik
            # Chain from this robot's previous goal so consecutive welds move
            # weld-to-weld instead of returning HOME each time. Warm-start IK
            # from the same pose for smoother, nearby solutions.
            chained_start = (
                req.get("current_joints")
                or get_last_config(cell_id, ns)
                or HOME.tolist()
            )
            warm = chained_start

            ik_target = _ik.invoke({
                "cell_id": cell_id, "robot_ns": ns,
                "x": x, "y": y, "z": z,
                "warm_start_joints": warm,
            })
            if not ik_target.get("success"):
                failures.append({"robot_ns": ns, "reason_code": "ik_failed", "error": f"IK failed for target: {ik_target.get('reason')}"})
                continue

            ik_approach = _ik.invoke({
                "cell_id": cell_id, "robot_ns": ns,
                "x": approach_x, "y": approach_y, "z": approach_z,
                "warm_start_joints": warm,
            })

            goal_q  = np.array(ik_target["joint_positions"][:model.nq])
            start_q = np.array(chained_start[:model.nq])
            via_q   = np.array(ik_approach["joint_positions"][:model.nq]) if ik_approach.get("success") else None

            # Build obstacle list from already-planned robots (Space-Time coupling)
            obstacles: list[tuple[np.ndarray, float]] = []

            # Per-body radii for this arm (arm steel vs the welding torch),
            # identical to what the monitor will apply to the live joint state.
            sphere_radii = _cell_sphere_radii(
                cell_id, model, rcfg.get("tool_collision"))

            # Terminal working stroke? The LLM sets "approach": true on the
            # request whose descent along the approach vector is the actual weld
            # approach. That stroke is executed SLOWLY, so it earns a smaller
            # speed allowance — the geometry is the same arm at both ends.
            use_approach = bool(req.get("approach", False))
            # The request's own working-stroke speed when it sets one (as OMPL).
            try:
                approach_vs = float(req["approach_velocity_scale"]) \
                    if req.get("approach_velocity_scale") is not None \
                    else velocity_scale * policy["approach_velocity_fraction"]
            except (TypeError, ValueError):
                approach_vs = velocity_scale * policy["approach_velocity_fraction"]
            approach_vs = max(approach_vs, 0.01)
            approach_margin = _speed_margin(
                cell_id, approach_vs if use_approach else velocity_scale)

            # ── On-axis LIFT-OFF from the seam the arm starts on (shared rule) ──
            # Where its TCP is decides it (robot_kinematics.seam_at_start), and a
            # blocked lift-off refuses the leg rather than dragging the tool out
            # of the seam sideways — as in every other backend.
            lift: list = []
            from agents.trajectory.tools.robot_kinematics import (
                liftoff_blocked_failure, seam_at_start,
            )
            seam = seam_at_start(cell_id, ns, model, start_q, base, yaw,
                                 pinned=bool(req.get("current_joints")))
            try:
                leave = float(req.get("retreat_offset_m") or approach_offset_m)
            except (TypeError, ValueError):
                leave = approach_offset_m
            if seam and seam.get("orientation") and float(seam.get("rise_m") or 0.0) < leave - 1e-3:
                from agents.trajectory.tools.leg_builder import cartesian_retreat
                pav = np.asarray(seam["approach_vector"], float)
                pav /= (np.linalg.norm(pav) + 1e-9)
                ptgt = np.asarray(seam["target"], float)
                po = seam["orientation"]
                wp = cartesian_retreat(
                    model, _ik, cell_id, ns, np.asarray(seam.get("from_point") or ptgt, float),
                    ptgt - pav * leave,
                    {"qx": float(po.get("x", 0.0)), "qy": float(po.get("y", 0.707)),
                     "qz": float(po.get("z", 0.0)), "qw": float(po.get("w", 0.707)),
                     "constrain_orientation": True},
                    start_q, base, yaw, obstacles, static_obstacles, normal_radius,
                    rcfg.get("tool_collision"), sphere_radii=sphere_radii,
                    extra_margin=approach_margin)
                if wp:
                    lift = [start_q] + [np.asarray(w, float)[:model.nq] for w in wp]
                    start_q = lift[-1]
                else:
                    refusal = liftoff_blocked_failure(ns, seam)
                    if refusal is not None:
                        failures.append({**refusal, "operation_id": req.get("operation_id"),
                                         "error": f"RRT: {refusal['error']}"})
                        continue

            # Pre-diagnose the endpoints so a collision failure is ACTIONABLE for
            # the LLM (names the object + world collision point) instead of a bare
            # timeout. A goal/start already in collision has no valid path, so we
            # fail fast rather than burning the whole RRT budget.
            endpoint_fail = _endpoint_collision_failure(
                model, start_q, goal_q, jl, ju, base, yaw, obstacles, static_obstacles,
                goal_radius=normal_radius, start_radius=normal_radius,
                sphere_radii=sphere_radii,
                goal_margin=approach_margin, start_margin=transit_margin,
            )
            if endpoint_fail is not None:
                entry = {
                    "robot_ns": ns,
                    "error": f"RRT: {endpoint_fail['reason']}",
                    "reason_code": endpoint_fail["reason_code"],
                    "collision": endpoint_fail["collision"],
                }
                # If the GOAL failed, also diagnose the target POINT vs the boxes:
                # is the weld itself inside / too close to a fixture box? If so,
                # emit an actionable move-the-weld-out suggestion.
                if endpoint_fail["reason_code"] == "goal_in_collision":
                    diag = _target_obstacle_suggestion(
                        [x, y, z], av, static_obstacles, policy["tool"] + 0.05,
                    )
                    if diag is not None:
                        entry["target_diagnosis"] = diag
                        entry["suggestion"] = diag["suggestion"]
                        entry["error"] = (f"RRT: goal in collision with "
                                          f"'{diag.get('object_id')}' — {diag['suggestion']}")
                failures.append(entry)
                continue

            waypoints = [start_q]
            if via_q is not None:
                # Transit to the approach waypoint — always normal clearance.
                path_to_approach = _rrt_single(
                    model, data, start_q, via_q, jl, ju, obstacles, base, yaw, max_rrt_iterations,
                    static_obstacles=static_obstacles, sphere_radius=normal_radius,
                    sphere_radii=sphere_radii, extra_margin=transit_margin,
                )
                if path_to_approach:
                    waypoints.extend(path_to_approach[1:])

            # Final descent to the target — same geometry, smaller speed
            # allowance when this request is the terminal working stroke.
            path_to_target = _rrt_single(
                model, data, waypoints[-1], goal_q, jl, ju, obstacles, base, yaw, max_rrt_iterations,
                static_obstacles=static_obstacles, sphere_radius=normal_radius,
                sphere_radii=sphere_radii, extra_margin=approach_margin,
            )
            if path_to_target is None:
                failures.append({
                    "robot_ns": ns,
                    "error": (
                        "RRT: no collision-free path found to the target within the "
                        "iteration/time budget although the endpoints are valid; try an "
                        "intermediate waypoint or a larger budget."
                    ),
                    "reason_code": "no_path_found",
                })
                continue

            # Time the parts apart, each at its own speed and tagged with its phase,
            # so every sample is judged at the clearance its speed earns: the
            # lift-off (exact, never shortcut: it is the on-axis stroke), the
            # transit, and the terminal working stroke.
            transit_wps = _shortcut_path(waypoints)
            stroke_wps = [waypoints[-1]] + list(path_to_target[1:])
            if use_approach:
                parts = [(lift, True, velocity_scale, "transit"),
                         (transit_wps, False, velocity_scale, "transit"),
                         (_shortcut_path(stroke_wps), False, approach_vs, "approach")]
            else:
                parts = [(lift, True, velocity_scale, "transit"),
                         (_shortcut_path(transit_wps + stroke_wps[1:]), False,
                          velocity_scale, "transit")]
            traj = _timed_parts(parts)

            # Validate what is EMITTED: the shortcut drops nodes unchecked and the
            # spline leaves the polyline, so the samples (and, for the arm against
            # itself, the motion between them) get the monitor's rule, as the OMPL
            # backends do.
            from agents.trajectory.tools.robot_kinematics import _trajectory_collision_info
            breach = _trajectory_collision_info(
                model, model.createData(), traj, base, yaw, obstacles, static_obstacles,
                sphere_radius=normal_radius, sphere_radii=sphere_radii,
                transit_margin=transit_margin, approach_margin=approach_margin)
            if breach is not None:
                failures.append({
                    "robot_ns": ns, "reason_code": "trajectory_in_collision",
                    "collision": breach,
                    "error": (f"RRT: the emitted trajectory breaches '{breach.get('object_id')}' "
                              f"at sample {breach.get('sample_index')}"),
                })
                continue

            # The emitted tool-rise cap every backend applies (leg_builder).
            from agents.trajectory.tools.leg_builder import (
                emitted_rise_limit, trajectory_tool_rise,
            )
            rise_m, rise_limit = trajectory_tool_rise(model, traj, base, yaw), \
                emitted_rise_limit()
            if rise_limit > 0.0 and rise_m > rise_limit:
                failures.append({
                    "robot_ns": ns, "reason_code": "excessive_tool_rise",
                    "error": (f"RRT: the emitted trajectory lifts the tool {rise_m:.3f} m "
                              f"above both of its endpoints, over the {rise_limit:.2f} m "
                              "limit; add a via point that routes around the obstruction"),
                })
                continue

            trajectories[ns] = traj
            # Remember where this arm ended so the next weld starts from here.
            set_last_config(cell_id, ns, goal_q.tolist())

        # Caching is the facade's job (run_trajectory_planner → dispatch_plan),
        # so this backend only produces the contract-shaped result.
        return {
            "success":         len(trajectories) > 0,
            "trajectories":    trajectories,
            "planning_time_s": round(time.monotonic() - t0, 3),
            "failures":        failures,
            "backend":         "rrt",
        }
    except Exception as e:
        return {"success": False, "trajectories": {}, "failures": [{"error": str(e)}],
                "planning_time_s": 0.0, "backend": "rrt"}




# ─── Backwards-compatible re-exports ─────────────────────────────────────────
# These names moved to backend-neutral modules (``task_allocation`` /
# ``time_parameterization``). They are re-exported here ONLY so existing callers
# and tests that import them from this module keep working. New code must import
# from the owning module — nothing below is RRT-specific.
from agents.trajectory.tools.task_allocation import (  # noqa: E402,F401
    _least_movement_schedule,
    _max_distance_schedule,
    _regional_allocate,
    _thermal_spread,
    _tsp_2opt,
)
from agents.trajectory.tools.time_parameterization import _sample_at_time  # noqa: E402,F401
