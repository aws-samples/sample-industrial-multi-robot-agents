"""
Pinocchio tools — deterministic FK/IK math bound to the trajectory agent.

These are LangChain @tool functions. The trajectory agent's LLM calls them
when a skill's instructions say to compute FK, IK, or collision checks.

All math is deterministic. The LLM provides strategy; these tools do the math.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from langchain_core.tools import tool

from utils import arm_model, cell_layout, cell_store


def _load_robot_config(cell_id: str, robot_ns: str) -> dict:
    """Load robot config from the cell's ``robots.yaml``, asked of its owner."""
    robots_yaml = cell_layout.robots_yaml(cell_id)
    cfg = cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG,
                               robots_yaml.name)
    if cfg is None:
        raise FileNotFoundError(
            f"no robots.yaml for cell '{cell_id}' ({cell_store.describe()})")
    # Match by robot identity (inner key). Works for both the legacy layout
    # {robot1: {robot1: {..}}} and the shared-namespace layout
    # {arms: {robot1: {..}, robot2: {..}}}.
    for _ns, ns_block in cfg.get("robots", {}).items():
        if isinstance(ns_block, dict):
            for rid, rcfg in ns_block.items():
                if rid == robot_ns and isinstance(rcfg, dict):
                    return rcfg
    raise KeyError(f"Robot '{robot_ns}' not found in {robots_yaml}")


def _load_obstacles(cell_id: str) -> list[dict]:
    """Load static obstacles from the cell's calculated ``obstacles.yaml``.

    Delegated to :func:`robot_kinematics.load_static_obstacles`, the one owner of
    this read: an empty list must mean "the cell declares no fixtures" and never
    "I could not find out". Each obstacle dict has: id, type, and type-specific
    geometry fields.
    """
    from agents.trajectory.tools.robot_kinematics import load_static_obstacles
    return load_static_obstacles(cell_id)


def _load_clearance_radii(cell_id: str) -> dict:
    """The cell's clearance policy — see ``robot_kinematics._load_clearance_policy``,
    which owns it.

    A thin delegation on purpose. This module used to parse obstacles.yaml itself,
    which made three independent readings of one safety policy (here, the OMPL
    backend, and the RRT backend) and let them drift. The radii now have a single
    owner; this keeps the name the LLM tool surface already imports.
    """
    from agents.trajectory.tools.robot_kinematics import _load_clearance_policy
    return _load_clearance_policy(cell_id)


def _load_pinocchio_model(cell_id: str, robot_ns: str):
    """The arm's model, with the cell's TCP frame attached.

    One line, and deliberately so: which file describes the arm (referenced in
    ``cell.yaml`` or baked under ``urdf/``) and whether the tool frame has to be
    added at load time are :mod:`utils.arm_model`'s decisions, not this module's.
    """
    model, data, _ee_id = arm_model.load(cell_id, robot_ns)
    return model, data


def _nearest_branch(joints, ref, jl, ju) -> list:
    """Re-express ``joints`` in the ±2π branch closest to ``ref``, within limits.

    A revolute wrist joint at -3.0 rad and one at +3.28 rad are the SAME tool pose,
    but interpolating between them spins the tool most of a turn. IK returns
    whichever branch its solver happened to land in, so consecutive welds can be
    handed neighbouring poses that differ by nearly a full revolution — the visible
    "strange rotation of the tool". This picks, per joint, the branch nearest to
    where the arm already is, provided it stays inside the joint limits.
    """
    if ref is None:
        return list(joints)
    out = []
    two_pi = 2.0 * np.pi
    for i, val in enumerate(joints):
        v = float(val)
        if i >= len(ref):
            out.append(v)
            continue
        r = float(ref[i])
        best = v
        for k in (-1, 1):
            cand = v + k * two_pi
            if jl[i] - 1e-9 <= cand <= ju[i] + 1e-9 and abs(cand - r) < abs(best - r):
                best = cand
        out.append(best)
    return out


def polish_6d(pin, model, data, ee_id, oMdes, q, jl, ju, tol=1e-6, max_iter=60):
    """A few full Gauss-Newton steps from a converged iterate to the exact pose.

    Returns the polished configuration (never worse than ``q``) and leaves the
    frame placements of ``data`` at it. Clamped to the joint box like the solve.
    """
    damp = 1e-6
    q = np.array(q, float)
    n6 = min(6, model.nq)

    def err_at(qq):
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        iMd = data.oMf[ee_id].actInv(oMdes)
        return iMd, pin.log(iMd).vector

    iMd, e = err_at(q)
    n = float(np.linalg.norm(e))
    for _ in range(max_iter):
        if n < tol:
            break
        J = -pin.Jlog6(iMd.inverse()) @ pin.computeFrameJacobian(model, data, q, ee_id, pin.LOCAL)
        dq = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
        qn = pin.integrate(model, q, dq)
        qn[:n6] = np.clip(qn[:n6], jl[:n6], ju[:n6])
        iMn, en = err_at(qn)
        nn = float(np.linalg.norm(en))
        if not nn < n:
            break
        q, iMd, e, n = qn, iMn, en, nn
    err_at(q)
    return q


def _joint_travel(joints, ref) -> float:
    """Largest per-joint move from ``ref`` to ``joints`` (radians); 0.0 if no ref."""
    if ref is None:
        return 0.0
    n = min(len(joints), len(ref))
    if n == 0:
        return 0.0
    return max(abs(float(joints[i]) - float(ref[i])) for i in range(n))


@tool
def run_ik(
    cell_id: str,
    robot_ns: str,
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.707,
    qz: float = 0.0,
    qw: float = 0.707,
    warm_start_joints: Optional[list[float]] = None,
    constrain_orientation: bool = False,
) -> dict:
    """
    Solve inverse kinematics for a robot using Pinocchio damped least-squares.

    Tries 4 initial seeds (plus warm_start_joints if provided).
    Validates all solutions against the cell's floor (its declared planes, else
    z = 0), arm and tool, with robot_kinematics.floor_clear.
    Returns the best floor-safe solution, preferring one that is also clear of the
    cell's fixtures.

    Branch selection is this function's job and nobody else's. Several seeds reach
    the same weld to sub-millimetre accuracy with different elbow/wrist branches,
    so "which of them do you hand back" is a real decision — and ranking them by
    joint travel alone, against a floor check only, lets a branch that is inside a
    fixture win over an equally accurate free one. Measured on ``welding_cell_2_small``:
    for W08/robot1 the default seeds return a configuration with arm link 10 buried
    0.1700 m inside ``FIXTURE_A_v180``, while the HOME-seeded branch for the
    SAME pose is free; downstream, the planner reported that weld unreachable
    (``goal_in_collision``) for an arm that can plainly stand there.

    Obstacle freedom is therefore a RANKING key here, ahead of travel — not a
    filter. When every branch collides the best one is still returned, with
    ``goal_collision_free: False``, because a caller telling "out of reach" apart
    from "reachable but obstructed" needs the configuration in order to explain the
    obstruction. Callers that need a usable goal must read that field; ``success``
    has always meant "IK converged", and still does.

    Args:
        cell_id:            Cell identifier
        robot_ns:           Robot namespace (e.g. "robot1")
        x, y, z:            Target EE position in world frame (metres)
        qx, qy, qz, qw:     Target EE orientation quaternion (default: tool down)
        warm_start_joints:  Optional initial joint config from knowledge base

    Returns:
        {"success": true, "joint_positions": [j1..j6], "position_error_m": float,
         "goal_collision_free": bool}   # field absent when no obstacle model loaded
        or {"success": false, "reason": str, "distance_from_base": float}
    """
    try:
        import pinocchio as pin

        rcfg      = _load_robot_config(cell_id, robot_ns)
        model, data = _load_pinocchio_model(cell_id, robot_ns)

        base = np.array(rcfg["base_position"])
        yaw  = float(rcfg.get("base_orientation_yaw", 0.0))
        from agents.trajectory.tools.joint_limits import planning_box
        jl, ju = planning_box(rcfg, model)

        # Transform target from world to robot local frame
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        rel = np.array([x, y, z]) - base
        local_target = np.array([
            rel[0]*cy - rel[1]*sy,
            rel[0]*sy + rel[1]*cy,
            rel[2],
        ])

        # Find welding_tcp / tool0 frame
        ee_id = arm_model.ee_frame_id(model, arm_model.tcp_spec(cell_id, robot_ns))

        # Seeds: home config + rotated variants + warm start. The home seed is
        # the cell's declared one — seeding from a pose the arm never parks at
        # biases every solution away from where it actually starts.
        from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose
        HOME  = np.array(home_pose(cell_id, robot_ns, HOME_JOINTS_FALLBACK), float)
        seeds = [
            HOME.copy(),
            np.array([0.0, -1.0, 1.5, 0.0, -0.5, 0.0]),
            np.array([0.5, -1.0, 1.5, 0.0, -0.5, 0.0]),
            np.array([-0.5, -1.0, 1.5, 0.0, -0.5, 0.0]),
        ]
        if warm_start_joints:
            seeds.insert(0, np.array(warm_start_joints[:model.nq]))

        # Reference pose for smoothness: where the arm already is. Every candidate
        # solution is scored on how far the joints (especially the wrist) must
        # travel from here, so IK cannot hand back a kinematically distant elbow/
        # wrist branch that reaches the same point by spinning the tool.
        ref = list(warm_start_joints[:model.nq]) if warm_start_joints else None

        # Obstacle awareness for branch ranking. It reuses ``_config_is_free`` — the
        # same validity predicate the RRT samples with and the same radii — rather
        # than a second collision model, so a branch this function calls free is a
        # branch the planner will also accept. Setting it up is best-effort: without
        # an obstacle model every candidate scores 0 and the ranking degenerates to
        # exactly the floor-and-travel behaviour that came before.
        _penalty_available = False
        try:
            from agents.trajectory.tools.robot_kinematics import (
                _cell_sphere_radii, _config_is_free, _load_clearance_policy,
            )
            _obstacles = _load_obstacles(cell_id)
            _radius    = float(_load_clearance_policy(cell_id)["normal"])
            _radii     = _cell_sphere_radii(cell_id, model, rcfg.get("tool_collision"))
            _col_data  = model.createData()
            _penalty_available = True
        except Exception:
            pass

        def _on_floor_ok(joints) -> bool:
            """The cell's floor (its declared planes, else z = 0), arm and tool —
            robot_kinematics.floor_clear, the rule every other check applies."""
            try:
                from agents.trajectory.tools.robot_kinematics import (
                    _load_clearance_policy as _lcp, _cell_sphere_radii as _csr, floor_clear)
                qv = np.zeros(model.nq)
                qv[:min(len(joints), model.nq)] = np.asarray(joints, float)[:model.nq]
                obs = _obstacles if _penalty_available else _load_obstacles(cell_id)
                return floor_clear(model, model.createData(), qv, base, yaw, obs,
                                   sphere_radius=float(_lcp(cell_id)["normal"]),
                                   sphere_radii=_csr(cell_id, model, rcfg.get("tool_collision")),
                                   tool_cfg=rcfg.get("tool_collision"))
            except Exception:  # noqa: BLE001 — fail closed: an unchecked branch is not "clear"
                return False

        def _branch_penalty(joints) -> int:
            """0 when this branch is clear of the fixtures, 1 when it is not."""
            if not _penalty_available:
                return 0
            try:
                qv = np.zeros(model.nq)
                qv[:min(len(joints), model.nq)] = np.asarray(joints, float)[:model.nq]
                return 0 if _config_is_free(
                    model, _col_data, qv, base, yaw, [], _obstacles,
                    sphere_radius=_radius, sphere_radii=_radii,
                ) else 1
            except Exception:
                return 0

        best_joints, best_err = None, float("inf")
        best_key = (2, float("inf"))
        eps, max_iter, dt, damp = 1e-3, 150, 0.1, 1e-6

        # ── Position-only IK (robust; always run first) ──────────────────────
        for seed in seeds:
            q = np.zeros(model.nq)
            q[:min(len(seed), model.nq)] = seed[:model.nq]

            for _ in range(max_iter):
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                p    = data.oMf[ee_id].translation
                err  = local_target - p
                norm = float(np.linalg.norm(err))

                if norm < eps:
                    joints = np.clip(q[:6], jl[:6], ju[:6])
                    joints = _nearest_branch(joints, ref, jl, ju)
                    # Floor safety: the cell's floor, arm and tool (_on_floor_ok).
                    # Every solution here is already sub-millimetre accurate
                    # (norm < eps), so they are equally good positionally — rank
                    # them by joint travel instead of by a meaningless error
                    # difference, and only fall back to error when there is no
                    # reference pose to measure travel against.
                    travel = _joint_travel(joints, ref)
                    # Free branches beat colliding ones outright; within a class the
                    # old ordering decides — travel when there is a pose to measure
                    # travel from, positional error otherwise.
                    key = (_branch_penalty(joints), travel if ref else norm)
                    if key < best_key and _on_floor_ok(joints):
                        best_key    = key
                        best_err    = norm
                        best_joints = list(joints)
                    break

                J3 = pin.computeFrameJacobian(
                    model, data, q, ee_id, pin.LOCAL_WORLD_ALIGNED
                )[:3, :]
                JJt = J3 @ J3.T + damp * np.eye(3)
                dq  = J3.T @ np.linalg.solve(JJt, err)
                q   = pin.integrate(model, q, dt * dq)
                # Keep the iterate inside the joint box (projected Newton).
                # ``pin.integrate`` is deliberately unbounded, so without this the
                # solve converges to a pose the arm cannot reach and the clip at
                # acceptance silently substitutes a DIFFERENT configuration —
                # see the comment on the same line in the 6D branch below.
                q[:6] = np.clip(q[:6], jl[:6], ju[:6])

        # ── Orientation refinement (6D), seeded from the position solution ────
        # Solve for tool orientation (perpendicular / pointing down the approach
        # vector) so the weld is approached top-down. Adding orientation from a
        # cold seed often fails to converge, so we seed from the position-only
        # solution above: the wrist then only has to rotate to align the tool,
        # which converges reliably. Falls back to the position-only solution if
        # no floor-safe oriented pose is found — a weld is never failed for
        # orientation alone.
        if constrain_orientation and best_joints is not None:
            q_des = pin.Quaternion(np.array([qx, qy, qz, qw], dtype=float))
            q_des.normalize()
            cyw, syw = np.cos(-yaw), np.sin(-yaw)
            Rz = np.array([[cyw, -syw, 0.0], [syw, cyw, 0.0], [0.0, 0.0, 1.0]])
            oMdes = pin.SE3(Rz @ q_des.toRotationMatrix(), local_target)
            ori_seeds = [np.array(best_joints)] + seeds
            ori_eps, ori_max_iter, ori_damp = 1.5e-2, 400, 1e-4
            ori_best, ori_best_err = None, float("inf")
            ori_best_key = (2, float("inf"))
            for seed in ori_seeds:
                q = np.zeros(model.nq)
                q[:min(len(seed), model.nq)] = seed[:model.nq]
                for _ in range(ori_max_iter):
                    pin.forwardKinematics(model, data, q)
                    pin.updateFramePlacements(model, data)
                    iMd  = data.oMf[ee_id].actInv(oMdes)
                    err6 = pin.log(iMd).vector
                    if float(np.linalg.norm(err6)) < ori_eps:
                        # Converged at the acceptance tolerance (up to ~0.9 deg /
                        # 15 mm): polish to the exact pose. A tilted weld
                        # orientation (weld_orientation.select) has only a
                        # fraction of a degree left to the axis gate.
                        q    = polish_6d(pin, model, data, ee_id, oMdes, q, jl, ju)
                        iMd  = data.oMf[ee_id].actInv(oMdes)
                        err6 = pin.log(iMd).vector
                        joints  = np.clip(q[:6], jl[:6], ju[:6])
                        # Same wrist-branch normalisation as the position solve:
                        # the 6D solve is the one that sets tool orientation, so
                        # this is where a full-turn wrist branch would otherwise
                        # be baked into the weld pose.
                        joints  = _nearest_branch(joints, ref, jl, ju)
                        pos_err = float(np.linalg.norm(err6[:3]))
                        key     = (_branch_penalty(joints), pos_err)
                        if key < ori_best_key and _on_floor_ok(joints):
                            ori_best_key = key
                            ori_best_err = pos_err
                            ori_best     = list(joints)
                        break
                    J6   = pin.computeFrameJacobian(model, data, q, ee_id, pin.LOCAL)
                    Jlog = pin.Jlog6(iMd.inverse())
                    J    = -Jlog @ J6
                    dq   = -J.T @ np.linalg.solve(J @ J.T + ori_damp * np.eye(6), err6)
                    q    = pin.integrate(model, q, dt * dq)
                    # Keep the iterate inside the joint box (projected Newton).
                    #
                    # pinocchio is not wrong here and neither is the input: the
                    # limits are in the model (model.lowerPositionLimit, parsed
                    # from the URDF and identical to robots.yaml) and the solver
                    # never consulted them. It solved an UNCONSTRAINED problem,
                    # converged to a pose outside the box, and the ``np.clip`` at
                    # acceptance then edited the answer — on welding_cell_2_small/W14
                    # it pinned FOUR joints to their limits and moved the tool
                    # 2.83 m, while ``pos_err`` still reported the 0.0066 m of the
                    # pose that was discarded. Clamping each step makes the
                    # converged solution feasible BY CONSTRUCTION, so the error we
                    # report is the error of the configuration we return. An
                    # unreachable pose now simply fails to converge, which is the
                    # honest answer; the caller already falls back to the
                    # position-only solution.
                    q[:6] = np.clip(q[:6], jl[:6], ju[:6])
                if ori_best is not None and ori_best_key[0] == 0:
                    # The position-seeded solve is the best candidate — but only if
                    # it is out of the fixtures. If it is not, keep trying seeds:
                    # stopping here is what pinned the oriented pose to a colliding
                    # branch. With no obstacle model the penalty is always 0 and
                    # this stops on the first solution, as it always did.
                    break
            if ori_best is not None:
                out = {
                    "success": True, "joint_positions": ori_best,
                    "position_error_m": round(ori_best_err, 5),
                    "orientation_constrained": True,
                }
                if _penalty_available:
                    out["goal_collision_free"] = ori_best_key[0] == 0
                return out
            # else: fall through and return the position-only solution.

        if best_joints is not None:
            out = {"success": True, "joint_positions": best_joints,
                   "position_error_m": round(best_err, 5)}
            if _penalty_available:
                out["goal_collision_free"] = best_key[0] == 0
            return out

        dist = float(np.linalg.norm(np.array([x, y, z]) - base))
        return {
            "success": False,
            "reason": "IK did not converge to a floor-safe solution",
            "distance_from_base": round(dist, 3),
        }

    except Exception as e:
        return {"success": False, "reason": str(e)}


def _compute_link_spheres_local(model, data, base, yaw) -> list:
    """Return world-frame sphere positions for all frames."""
    import pinocchio as pin
    cy, sy = np.cos(yaw), np.sin(yaw)
    positions = []
    for i in range(len(model.frames)):
        p  = data.oMf[i].translation.copy()
        wx = base[0] + p[0]*cy - p[1]*sy
        wy = base[1] + p[0]*sy + p[1]*cy
        wz = base[2] + p[2]
        positions.append(np.array([wx, wy, wz]))
    return positions


@tool
def run_fk(
    cell_id: str,
    robot_ns: str,
    joint_positions: list[float],
) -> dict:
    """
    Compute end-effector position via Pinocchio forward kinematics.

    Args:
        cell_id:          Cell identifier
        robot_ns:         Robot namespace
        joint_positions:  6 joint angles in radians

    Returns:
        {"ee_position": [x, y, z], "ee_quaternion": [qx, qy, qz, qw]}
    """
    try:
        import pinocchio as pin

        rcfg        = _load_robot_config(cell_id, robot_ns)
        model, data = _load_pinocchio_model(cell_id, robot_ns)

        base = np.array(rcfg["base_position"])
        yaw  = float(rcfg.get("base_orientation_yaw", 0.0))
        cy, sy = np.cos(yaw), np.sin(yaw)

        q = np.zeros(model.nq)
        q[:min(6, model.nq)] = joint_positions[:model.nq]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        ee_id = arm_model.ee_frame_id(model, arm_model.tcp_spec(cell_id, robot_ns))

        p  = data.oMf[ee_id].translation
        wx = base[0] + p[0]*cy - p[1]*sy
        wy = base[1] + p[0]*sy + p[1]*cy
        wz = base[2] + p[2]

        return {"ee_position": [round(float(wx), 4), round(float(wy), 4), round(float(wz), 4)]}
    except Exception as e:
        return {"error": str(e)}


def _quat_to_R(ori: dict) -> "np.ndarray":
    """3x3 rotation matrix from a quaternion dict {x,y,z,w} (normalised)."""
    x = float(ori.get("x", 0.0)); y = float(ori.get("y", 0.0))
    z = float(ori.get("z", 0.0)); w = float(ori.get("w", 1.0))
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _tool_axis(R: "np.ndarray") -> "np.ndarray":
    """Tool approach axis in world frame (project convention: -R[:,0])."""
    a = -R[:, 0]
    return a / (np.linalg.norm(a) + 1e-9)


def _ee_world_pos(model, data, base, cy, sy, ee_id) -> "np.ndarray":
    p = data.oMf[ee_id].translation
    return np.array([base[0] + p[0] * cy - p[1] * sy,
                     base[1] + p[0] * sy + p[1] * cy,
                     base[2] + p[2]])


@tool
def validate_approach(
    cell_id: str,
    robot_ns: str,
    expected_orientation: dict,
    trajectory: Optional[list[dict]] = None,
    joint_positions: Optional[list[float]] = None,
    operation_id: Optional[str] = None,
    target_position: Optional[list[float]] = None,
    approach_vector: Optional[list[float]] = None,
    orientation_tolerance_deg: Optional[float] = None,
    position_tolerance_m: float = 0.01,
    approach_direction_tolerance_deg: float = 15.0,
    approach_span_m: float = 0.04,
    min_entry_travel_m: Optional[float] = None,
) -> dict:
    """
    Validate the TERMINAL WORKING APPROACH of a planned weld — i.e. only the last
    part of the motion, where the tool moves in to do the work at the point.

    This tool judges NOTHING about HOW the arm got near the point (top approach,
    corridor, direct — that is YOUR free choice, guided by `approach-strategy`).
    It checks only that the WORKING STROKE is correct:
      1. the FINAL tool pose matches the specified `weld_orientation` (the lean),
      2. (optional) the FINAL position reaches the target, and
      3. (optional) over the last `approach_span_m` of travel the tool moves IN
         along the specified `approach_vector` (a straight, on-axis entry — not
         a sideways scrape).

    Deterministic FK + comparison; it decides nothing. YOU decide what to do with
    the result: if `matches` is false, re-plan ONLY the terminal segment
    (standoff → point) with the orientation constrained and the entry along the
    approach vector (see the plan-trajectories skill). Never accept/assemble a
    weld whose working approach does not match its specification.

    Provide `trajectory` (preferred — needed for the direction check), or just
    `joint_positions` (final config; pose/position checks only), or `operation_id`
    to have the trajectory looked up from the planner's cache. Calling with none
    of the three cannot validate anything — supply one rather than retrying.

    Args:
        cell_id, robot_ns:       cell + robot
        expected_orientation:    specified tool pose quaternion {x,y,z,w}
                                 (the operation's weld_orientation)
        trajectory:              list of {"positions":[...], "time_s":...}
        joint_positions:         6 final joint angles (rad) if no trajectory
        operation_id:            operation whose cached trajectory to validate,
                                 used only when `trajectory` is omitted
        target_position:         optional [x,y,z] to also check final position
        approach_vector:         optional [ax,ay,az] the terminal entry must follow
                                 (world frame; e.g. the on-axis weld entry)
        *_tolerance_*:           thresholds (axis: the tool's robots.yaml
                                 ``tool_orientation.axis_tolerance_deg``, 5deg
                                 by default; pos 10mm, dir 15deg)
        approach_span_m:         how much terminal travel to measure direction over
        min_entry_travel_m:      when `approach_vector` and `trajectory` are given,
                                 the least terminal TCP travel the entry check
                                 needs. Default: `approach_span_m`, or the leg's
                                 own working stroke (its `phase == "approach"`
                                 samples) when that is shorter, never less than
                                 `position_tolerance_m`. A leg that moves the TCP
                                 less — e.g. one that starts standing on the weld —
                                 has no measurable entry and does NOT match.

    Returns:
        {"matches": bool, "tool_axis_error_deg", "orientation_error_deg",
         "position_error_m"|null, "approach_direction_error_deg"|null,
         "entry_travel_m"|null, "actual_tool_axis", "expected_tool_axis", "reason"}
    """
    try:
        import pinocchio as pin

        # Fall back to the planner's own cached trajectory when the caller names
        # the operation but does not re-send the waypoints. Without this, a call
        # that omits `trajectory` is a wasted round trip: the direction check is
        # skipped and the caller must call again with the array inlined.
        if trajectory is None and operation_id:
            from agents.trajectory.tools.trajectory_cache import get_cached_trajectory
            trajectory = get_cached_trajectory(cell_id, robot_ns, operation_id)

        q_final = joint_positions
        if q_final is None and trajectory:
            last = trajectory[-1]
            q_final = last.get("positions") if isinstance(last, dict) else last
        if not q_final:
            return {
                "matches": False,
                "reason": (
                    "nothing to validate: pass `trajectory` (preferred — enables the "
                    "entry-direction check), or `joint_positions`, or an `operation_id` "
                    "whose trajectory the planner has already cached. Do NOT retry with "
                    "the same arguments; they carry no pose to check."
                ),
            }

        rcfg        = _load_robot_config(cell_id, robot_ns)
        model, data = _load_pinocchio_model(cell_id, robot_ns)
        if orientation_tolerance_deg is None:
            # The tool's own axis tolerance — the same number the weld pose
            # selection tilts within, so a pose it picks is a pose this accepts.
            from agents.trajectory.tools.weld_orientation import policy
            orientation_tolerance_deg = float(policy(rcfg)["axis_tolerance_deg"])
        base = np.array(rcfg["base_position"])
        yaw  = float(rcfg.get("base_orientation_yaw", 0.0))
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

        def _fk(qj):
            q = np.zeros(model.nq)
            q[:min(6, model.nq)] = np.array(qj[:model.nq])
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            return q

        ee_id = arm_model.ee_frame_id(model, arm_model.tcp_spec(cell_id, robot_ns))

        _fk(q_final)
        R_world = Rz @ np.array(data.oMf[ee_id].rotation)
        p_final = _ee_world_pos(model, data, base, cy, sy, ee_id)

        R_exp = _quat_to_R(expected_orientation)
        actual_axis   = _tool_axis(R_world)
        expected_axis = _tool_axis(R_exp)
        axis_err_deg = float(np.degrees(np.arccos(
            float(np.clip(np.dot(actual_axis, expected_axis), -1.0, 1.0)))))

        Rrel = R_world.T @ R_exp
        ori_err_deg = float(np.degrees(np.arccos(
            float(np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)))))

        pos_err = None
        if target_position is not None:
            pos_err = float(np.linalg.norm(p_final - np.array(target_position[:3])))

        # Terminal entry direction: walk back from the last waypoint until the EE
        # has travelled ~approach_span_m, and compare that motion to approach_vector.
        dir_err_deg = None
        entry_travel = 0.0
        # An entry to judge is required whenever one was specified: without a
        # trajectory there is no entry, and that is not a pass (umarell issue 07).
        entry_required = approach_vector is not None
        stroke = None
        if approach_vector is not None and trajectory and len(trajectory) >= 2:
            av = np.array(approach_vector[:3], dtype=float)
            av /= (np.linalg.norm(av) + 1e-9)
            p_end = p_final
            p_start = None
            for k in range(len(trajectory) - 2, -1, -1):
                qk = trajectory[k].get("positions") if isinstance(trajectory[k], dict) else trajectory[k]
                _fk(qk)
                pk = _ee_world_pos(model, data, base, cy, sy, ee_id)
                if np.linalg.norm(p_end - pk) >= approach_span_m:
                    p_start = pk
                    break
                p_start = pk
            if p_start is not None:
                move = p_end - p_start
                nm = np.linalg.norm(move)
                entry_travel = float(nm)
                if nm > 1e-6:
                    dir_err_deg = float(np.degrees(np.arccos(
                        float(np.clip(np.dot(move / nm, av), -1.0, 1.0)))))

            # The leg's own working stroke, when its samples say which they are: a
            # cell may plan a stroke shorter than approach_span_m, and that is a
            # real entry, not a missing one.
            first = next((k for k, w in enumerate(trajectory)
                          if isinstance(w, dict) and w.get("phase") == "approach"), None)
            if first is not None:
                _fk(trajectory[first]["positions"])
                stroke = float(np.linalg.norm(
                    p_final - _ee_world_pos(model, data, base, cy, sy, ee_id)))

        ori_ok = axis_err_deg <= orientation_tolerance_deg
        pos_ok = (pos_err is None) or (pos_err <= position_tolerance_m)
        dir_ok = (dir_err_deg is None) or (dir_err_deg <= approach_direction_tolerance_deg)
        # No measurable entry is not a pass: a stationary or too-short leg would
        # otherwise satisfy the entry gate vacuously (direction None).
        if min_entry_travel_m is not None:
            need = float(min_entry_travel_m)
        else:
            need = approach_span_m if stroke is None else min(approach_span_m, stroke)
            need = max(need, position_tolerance_m)
        travel_ok = (not entry_required) or (
            dir_err_deg is not None and entry_travel >= need - 1e-6)
        # Joint limits on everything this call judges: the whole trajectory when
        # given — a leg that swings a wrist past its limit on the way in is not a
        # valid approach however well it ends — else the final pose.
        from agents.trajectory.tools.joint_limits import (
            describe as _lim_describe, joint_names, position_limits,
            trajectory_limit_violation,
        )
        _lo, _hi = position_limits(rcfg, model)
        limit_v = trajectory_limit_violation(
            trajectory if trajectory else [list(q_final)], _lo, _hi,
            names=joint_names(rcfg, model))
        limits_ok = limit_v is None
        matches = bool(ori_ok and pos_ok and dir_ok and travel_ok and limits_ok)

        reasons = []
        if not ori_ok:
            reasons.append(f"tool lean off by {axis_err_deg:.1f}deg (> {orientation_tolerance_deg})")
        if pos_err is not None and not pos_ok:
            reasons.append(f"position off by {pos_err*1000:.0f}mm (> {position_tolerance_m*1000:.0f}mm)")
        if dir_err_deg is not None and not dir_ok:
            reasons.append(f"entry direction off by {dir_err_deg:.1f}deg (> {approach_direction_tolerance_deg})")
        if not travel_ok and not trajectory:
            reasons.append(
                "entry not checkable: an approach_vector was given but no trajectory "
                "(pass `trajectory`, or an operation_id whose trajectory is cached)")
        elif not travel_ok:
            reasons.append(
                f"no measurable entry: the TCP travels {entry_travel*1000:.1f}mm over the "
                f"end of the leg (< {need*1000:.0f}mm), so the entry direction cannot be "
                f"checked; plan a real terminal stroke along approach_vector")
        if not limits_ok:
            reasons.append(_lim_describe(limit_v, robot_ns))
        reason = "terminal working approach matches the specification" if matches else "; ".join(reasons)

        return {
            "matches": matches,
            "tool_axis_error_deg": round(axis_err_deg, 2),
            "orientation_error_deg": round(ori_err_deg, 2),
            "position_error_m": None if pos_err is None else round(pos_err, 4),
            "approach_direction_error_deg": None if dir_err_deg is None else round(dir_err_deg, 2),
            "entry_travel_m": round(entry_travel, 4) if entry_required else None,
            "joint_limits_ok": limits_ok,
            **({"joint_limit_violation": limit_v} if limit_v is not None else {}),
            "actual_tool_axis": [round(float(v), 4) for v in actual_axis],
            "expected_tool_axis": [round(float(v), 4) for v in expected_axis],
            "reason": reason,
        }
    except Exception as e:
        return {"matches": False, "reason": f"validation error: {e}"}


@tool
def check_self_collision(
    cell_id: str,
    robot_ns: str,
    joint_positions: list[float],
    approach: bool = False,
) -> dict:
    """
    Check a joint configuration for self-collision, floor and obstacle collision.

    Sphere radii come from the cell clearance policy in obstacles.yaml (NOT
    hardcoded), PER BODY: ``robot_link_radius`` for the arm structure and
    ``tool_link_radius`` for the end-effector frames. The obstacle clearance of a
    sphere is ``its radius + collision_margin``, e.g. 0.12 + 0.05 = 0.17 m for a
    casting and 0.05 + 0.05 = 0.10 m for the torch.

    ``approach`` is accepted but INERT, and kept only so existing callers and the
    tool schema do not break. It used to select a reduced radius for the whole arm
    on the terminal stroke — which thinned the castings too, and which the runtime
    monitor never honoured, so a configuration could pass here and e-stop on the
    floor. Phase now affects the SPEED allowance (see
    ``robot_kinematics._speed_margin``), not the geometry.

    Checks:
    - Self-collision: the arm's collision hulls against each other
      (``self_collision`` in the result says whether that check ran)
    - Floor and static obstacles: the cell's planes (else z = 0), boxes and
      spheres, against the arm and the tool's static spheres
    - Static obstacles from obstacles.yaml (``tool_contact`` boxes — and
      ``is_workpiece`` ones, read as such — are exempt for the tool only)
    - Joint limits: all joints within limits from robots.yaml

    Args:
        cell_id:          Cell identifier
        robot_ns:         Robot namespace
        joint_positions:  6 joint angles in radians
        approach:         Accepted and ignored (see above)

    Returns:
        {"collision_free": bool, "violations": [...]}
    """
    try:
        import pinocchio as pin

        rcfg        = _load_robot_config(cell_id, robot_ns)
        model, data = _load_pinocchio_model(cell_id, robot_ns)
        from agents.trajectory.tools.joint_limits import planning_box
        jl, ju = planning_box(rcfg, model)
        base = np.array(rcfg["base_position"])
        yaw  = float(rcfg.get("base_orientation_yaw", 0.0))

        joints = np.array(joint_positions[:6])
        violations = []

        # Joint limits
        for i, (j, lo, hi) in enumerate(zip(joints, jl[:6], ju[:6])):
            if j < lo or j > hi:
                violations.append(f"joint_{i+1}={j:.3f} outside [{lo:.3f},{hi:.3f}]")

        # FK + floor check
        q = np.zeros(model.nq)
        q[:min(6, model.nq)] = joints
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # The cell's static bodies — the floor (its declared planes, else z = 0),
        # boxes and spheres — against the arm AND the tool's static spheres, each
        # at its own radius: the same check every planner and the monitor apply
        # (robot_kinematics._static_first_breach). Every body that is hit is
        # reported, not only the first.
        from agents.trajectory.tools.robot_kinematics import (
            _cell_sphere_radii, _compile_static, _sphere_positions, _static_bodies,
            _static_first_breach,
        )
        obstacles = _load_obstacles(cell_id)
        sphere_radius = _load_clearance_radii(cell_id)["normal"]
        radii = _cell_sphere_radii(cell_id, model, rcfg.get("tool_collision"))
        P, R, on_floor, label = _static_bodies(
            model, data, base, yaw, _sphere_positions(model, data, base, yaw),
            sphere_radius, radii, rcfg.get("tool_collision"))
        comp = _compile_static(obstacles)
        for kind in ("pl", "box", "sph"):
            ids = comp[f"{kind}_id"]
            for m in range(len(ids)):
                one = {**comp,
                       "pl_id": comp["pl_id"][m:m + 1] if kind == "pl" else [],
                       "pl_n": comp["pl_n"][m:m + 1] if kind == "pl" else comp["pl_n"][:0],
                       "pl_d": comp["pl_d"][m:m + 1] if kind == "pl" else comp["pl_d"][:0],
                       "pl_m": comp["pl_m"][m:m + 1] if kind == "pl" else comp["pl_m"][:0],
                       "box_k": comp["box_k"][m:m + 1] if kind == "box" else [],
                       "box_id": comp["box_id"][m:m + 1] if kind == "box" else [],
                       "box_c": comp["box_c"][m:m + 1], "box_h": comp["box_h"][m:m + 1],
                       "box_m": comp["box_m"][m:m + 1],
                       "box_tool_ok": comp["box_tool_ok"][m:m + 1],
                       "sph_k": comp["sph_k"][m:m + 1] if kind == "sph" else [],
                       "sph_id": comp["sph_id"][m:m + 1] if kind == "sph" else [],
                       "sph_c": comp["sph_c"][m:m + 1], "sph_r": comp["sph_r"][m:m + 1],
                       "sph_m": comp["sph_m"][m:m + 1]}
                hit = _static_first_breach(P, R, on_floor if kind == "pl" else
                                           np.zeros(len(P), bool), label, one, 0.0)
                if hit is None:
                    continue
                who = (f"link_{hit['link_index']}" if hit.get("link_index") is not None
                       else f"tool sphere {hit.get('tool_sphere')}")
                if hit["type"] in ("floor", "plane"):
                    violations.append(f"{who} below {hit['object_id']}: "
                                      f"clearance={hit['clearance_m']:.3f}m")
                elif hit["type"] == "box":
                    violations.append(f"{who} penetrates obstacle '{hit['object_id']}': "
                                      f"clearance={hit['clearance_m']:.3f}m")
                else:
                    violations.append(f"{who} penetrates sphere obstacle "
                                      f"'{hit['object_id']}': clearance={hit['clearance_m']:.3f}m")

        # The arm against itself — its links and the tool on the tip link, as
        # convex hulls (tools/self_collision.py). Reported as checked or not:
        # "no violation" must not read as "self-collision free" when nothing ran.
        from agents.trajectory.tools.self_collision import checker
        sc = checker(cell_id, robot_ns)
        if sc.available:
            hit = sc.violation(q)
            if hit is not None:
                violations.append(f"self-collision: {hit['object_id']} "
                                  f"(signed distance {hit['clearance_m']:.3f}m)")

        if violations:
            shown = "; ".join(violations[:4])
            extra = f" (+{len(violations) - 4} more)" if len(violations) > 4 else ""
            print(f"[collision] {robot_ns}: {shown}{extra}", flush=True)

        return {
            "collision_free": len(violations) == 0,
            "violations":     violations,
            "self_collision": sc.status(),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_link_spheres(
    cell_id: str,
    robot_ns: str,
    joint_positions: list[float],
) -> dict:
    """
    Get world-frame sphere positions for all robot links (for collision checking).

    Args:
        cell_id:          Cell identifier
        robot_ns:         Robot namespace
        joint_positions:  6 joint angles in radians

    Returns:
        {"spheres": [[x, y, z], ...], "radius_m": 0.12}
    """
    try:
        import pinocchio as pin

        rcfg        = _load_robot_config(cell_id, robot_ns)
        model, data = _load_pinocchio_model(cell_id, robot_ns)
        base = np.array(rcfg["base_position"])
        yaw  = float(rcfg.get("base_orientation_yaw", 0.0))

        q = np.zeros(model.nq)
        q[:min(6, model.nq)] = joint_positions[:model.nq]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        spheres = _compute_link_spheres_local(model, data, base, yaw)
        return {
            "spheres":   [[round(float(v), 4) for v in s] for s in spheres],
            "radius_m":  0.12,
        }
    except Exception as e:
        return {"error": str(e)}
