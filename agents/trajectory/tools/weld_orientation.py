"""The weld pose's free degrees of freedom: spin about the tool axis, small tilt.

A weld specifies a tool AXIS (``validate_approach`` gates the axis within the
tool's ``axis_tolerance_deg``), but ``weld_orientation`` is a full quaternion, so
the IK used to solve it with the spin about the electrode FIXED. Where that one
spin folds the arm into itself, the weld had no taker (umarell issue 12: robot1's
W01 on welding_cell_2_small, the tool plate 45 mm into link_3).

:func:`select` searches the freedom the TOOL allows — declared per robot by the
converter in ``robots.yaml`` ``tool_orientation`` (``spin_free``, ``max_tilt_deg``,
``axis_tolerance_deg``) — for the first pose, nearest the specified one, whose
weld point AND standoff both have a collision-free oriented IK solution (fixtures,
floor and the arm itself, the same predicate the planners use). It runs before
dispatch (``planner_backend.dispatch_plan``), so every backend plans the chosen
pose and records it as the seam the next leg lifts off from.

The specified pose is kept whenever it works: the search only runs when it does
not. A spin the agent chose (``tool_spin_deg``) is never overridden, and a request
can opt out with ``free_spin: false``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

#: Spin candidates, nearest the specified pose first: every 15 degrees.
SPIN_STEP_DEG = 15.0
#: Tilt directions tried per tilt (about the tool axis).
TILT_DIRECTIONS = 8
#: An accepted tilted pose keeps this much of the axis gate in reserve (IK
#: residual, rounding).
AXIS_GATE_RESERVE_DEG = 0.25
#: Preferred hull clearance of the arm with itself along the weld stroke, and the
#: least a chosen pose may have (below it only as a fallback, never below 0+min).
SELF_MARGIN_M = 0.01
SELF_MIN_M = 0.002
#: Candidates of one tilt group verified with the planner's own IK.
MAX_VERIFY = 3
#: Straight IK steps of the on-axis entry checked (as the planner's descent).
DESCENT_STEPS = 6

#: What a robot whose robots.yaml has no ``tool_orientation`` block gets — the
#: converter's defaults (scripts/convert_isaac_scene.py).
DEFAULT_POLICY = {"spin_free": True, "max_tilt_deg": 4.5, "axis_tolerance_deg": 5.0}


def policy(rcfg: dict) -> dict:
    """The robot's ``tool_orientation`` block over :data:`DEFAULT_POLICY`."""
    out = dict(DEFAULT_POLICY)
    block = (rcfg or {}).get("tool_orientation") or {}
    if "spin_free" in block:
        out["spin_free"] = bool(block["spin_free"])
    for key in ("max_tilt_deg", "axis_tolerance_deg"):
        try:
            v = float(block[key])
        except (KeyError, TypeError, ValueError):
            continue
        if v >= 0.0:
            out[key] = v
    out["max_tilt_deg"] = min(out["max_tilt_deg"],
                              max(out["axis_tolerance_deg"] - AXIS_GATE_RESERVE_DEG, 0.0))
    return out


# ── quaternion helpers ({x, y, z, w} dicts, as the requests carry them) ──────

def _q(d: dict) -> np.ndarray:
    return np.array([float(d.get("w", 1.0)), float(d.get("x", 0.0)),
                     float(d.get("y", 0.0)), float(d.get("z", 0.0))])


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _axis_angle(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    h = math.radians(deg) / 2.0
    return np.array([math.cos(h), *(math.sin(h) * a)])


def _d(q: np.ndarray) -> dict:
    q = q / (np.linalg.norm(q) + 1e-12)
    return {"w": round(float(q[0]), 9), "x": round(float(q[1]), 9),
            "y": round(float(q[2]), 9), "z": round(float(q[3]), 9)}


def _R(d: dict) -> np.ndarray:
    w, x, y, z = _q(d) / (np.linalg.norm(_q(d)) + 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _spins(pol: dict) -> list:
    if not pol["spin_free"]:
        return [0.0]
    out, k = [0.0], 1
    while k * SPIN_STEP_DEG <= 180.0 + 1e-9:
        s = k * SPIN_STEP_DEG
        out += [s] if abs(s - 180.0) < 1e-9 else [s, -s]
        k += 1
    return out


def candidates(orientation: dict, pol: dict) -> list:
    """``[(spin_deg, tilt_deg, tilt_dir_deg, orientation)]``, nearest first.

    Every spin at the specified axis first, then, for each tilt (half the
    allowed tilt, then all of it) and tilt direction, every spin again. The tool
    axis is the TCP frame's local x (``planner_backend.apply_tool_spin``), so a
    spin is a local rotation about x and a tilt a local rotation about an axis in
    the y-z plane, at ``tilt_dir_deg`` about x.
    """
    base = _q(orientation)
    spins = _spins(pol)
    out = [(s, 0.0, 0.0, _d(_mul(base, _axis_angle([1, 0, 0], s)))) for s in spins[1:]]
    tmax = float(pol["max_tilt_deg"])
    if tmax > 0.0:
        for tilt in (tmax / 2.0, tmax):
            for k in range(TILT_DIRECTIONS):
                phi = 360.0 * k / TILT_DIRECTIONS
                axis = [0.0, math.cos(math.radians(phi)), math.sin(math.radians(phi))]
                tq = _mul(base, _axis_angle(axis, tilt))
                out += [(s, tilt, phi, _d(_mul(tq, _axis_angle([1, 0, 0], s))))
                        for s in spins]
    return out


def _transit_rise_preference() -> float:
    """The tool rise a transit is preferred to stay under (the planner's
    ``OMPL_PATH_MAX_RISE_M``, 0.15 m by default)."""
    from agents.trajectory.tools.path_metrics import _path_rise_policy
    return float(_path_rise_policy())


def _wanted(request: dict, pol: dict) -> bool:
    return bool(request.get("approach") and request.get("constrain_orientation")
                and isinstance(request.get("orientation"), dict)
                and "applied_tool_spin_deg" not in request
                and "tool_spin_deg" not in request       # the agent chose (0 included)
                and request.get("free_spin", True) is not False
                and (pol["spin_free"] or pol["max_tilt_deg"] > 0.0))


_MEMO: "dict" = {}
_MEMO_MAX = 512
_MEMO_LOCK = __import__("threading").Lock()


def select(cell_id: str, request: dict, start_q,
           trace: Optional[list] = None) -> Optional[dict]:
    """The pose :func:`search` chose, or None (keep the request as it is)."""
    return search(cell_id, request, start_q, trace)["chosen"]


def search(cell_id: str, request: dict, start_q,
           trace: Optional[list] = None) -> dict:
    """What the weld-pose search did for ``request``, memoised (same request, same
    start, same tool policy: same answer — it is asked again on every re-plan and
    on every stroke hint).

    ``{"status", "chosen", "candidates"}`` — ``status`` is ``not_applicable`` (no
    freedom allowed, the agent fixed the spin, not a constrained weld),
    ``specified_ok`` (the specified pose is usable as it is), ``chosen`` (a turned
    / tilted pose is planned instead, in ``chosen``) or ``none_usable`` (every
    candidate was tried and none is usable: re-sending the same request with
    another hand-picked spin cannot succeed)."""
    if trace is not None:
        return _search(cell_id, request, start_q, trace)
    try:
        from agents.trajectory.tools.robot_kinematics import _load_robot_config
        pol = policy(_load_robot_config(cell_id, request.get("robot_ns")))
    except Exception:  # noqa: BLE001 — no config: let _search report it
        return _search(cell_id, request, start_q, None)
    try:
        t = request.get("target") or {}
        o = request.get("orientation") or {}
        key = (str(cell_id), str(request.get("robot_ns")),
               tuple(round(float(t[k]), 6) for k in ("x", "y", "z")),
               tuple(round(float(o.get(k, 0.0)), 6) for k in ("x", "y", "z", "w")),
               tuple(round(float(v), 6) for v in request.get("approach_vector") or ()),
               round(float(request.get("approach_offset_m") or 0.1), 6),
               bool(request.get("approach")), bool(request.get("constrain_orientation")),
               "tool_spin_deg" in request or "applied_tool_spin_deg" in request,
               request.get("free_spin", True) is not False,
               None if start_q is None else tuple(round(float(v), 4) for v in start_q),
               tuple(sorted(pol.items())))
    except (KeyError, TypeError, ValueError):
        return _search(cell_id, request, start_q, None)
    with _MEMO_LOCK:
        if key in _MEMO:
            hit = _MEMO[key]
            return {**hit, "chosen": None if hit["chosen"] is None else dict(hit["chosen"])}
    out = _search(cell_id, request, start_q, None)
    with _MEMO_LOCK:
        _MEMO[key] = dict(out)
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.pop(next(iter(_MEMO)))
    return out


def _search(cell_id, request, start_q, trace) -> dict:
    status: dict = {"status": "not_applicable", "candidates": 0}
    chosen = _select(cell_id, request, start_q, trace, status)
    return {"status": status["status"], "candidates": status["candidates"],
            "chosen": chosen}


def clear_cache() -> None:
    with _MEMO_LOCK:
        _MEMO.clear()


def _select(cell_id: str, request: dict, start_q,
            trace: Optional[list] = None, status: Optional[dict] = None) -> Optional[dict]:
    """The weld pose to plan for ``request`` when its specified one is not usable,
    or None (keep the request as it is).

    A weld leg is USABLE when the weld pose, the standoff and the straight entry
    between them solve oriented, keep ``SELF_MARGIN_M`` of hull clearance from the
    arm itself (``SELF_MIN_M`` as a fallback) and are clear of the fixtures at the
    approach-phase margin — the planner's own predicate.

    1. The specified pose is kept whenever its leg is usable.
    2. Otherwise every candidate (:func:`candidates`) is SCREENED with a fast
       oriented solve polished to the exact pose (weld, standoff, mid-stroke),
       ranked by clearance class, then joint travel from ``start_q``, then how
       far it turns from the specified pose.
    3. The best few of each tilt group are VERIFIED with ``pinocchio.run_ik``
       called exactly as the planner calls it (same warm start), so the choice
       and the plan cannot disagree. The first verified with margin wins.

    Returns ``{"orientation", "spin_deg", "tilt_deg", "tilt_dir_deg",
    "self_clearance_m", "joint_positions"}``, or None (no freedom allowed, the
    agent fixed the spin, the specified pose is fine, or nothing verifies — the
    planner then reports the specified pose's failure, which names the cause).
    ``trace`` (a list) collects why each candidate was dropped, for diagnosis.
    """
    def note(*what):
        if trace is not None:
            trace.append(what)

    if status is None:
        status = {}

    import pinocchio as pin
    from agents.trajectory.tools import self_collision
    from agents.trajectory.tools.joint_limits import planning_box
    from agents.trajectory.tools.pinocchio import _joint_travel, polish_6d, run_ik
    from agents.trajectory.tools.robot_kinematics import (
        _cell_sphere_radii, _config_is_free, _load_clearance_policy,
        _load_pinocchio_model, _load_robot_config, _speed_margin,
        load_static_obstacles, resolve_velocity_scale,
    )
    from utils import arm_model

    ns = request.get("robot_ns")
    try:
        rcfg = _load_robot_config(cell_id, ns)
    except Exception:  # noqa: BLE001 — no config: the planner reports it
        return None
    pol = policy(rcfg)
    if not _wanted(request, pol):
        return None
    tgt = request.get("target") or {}
    try:
        p = np.array([float(tgt["x"]), float(tgt["y"]), float(tgt["z"])])
        av = np.asarray(request.get("approach_vector", [0, 0, -1]), float)
        av /= (np.linalg.norm(av) + 1e-9)
        offset = float(request.get("approach_offset_m") or 0.1)
    except (KeyError, TypeError, ValueError):
        return None
    via = p - av * offset
    nominal = request["orientation"]

    model, _ = _load_pinocchio_model(cell_id, ns)
    data = model.createData()
    jl, ju = planning_box(rcfg, model)
    ee = arm_model.ee_frame_id(model, arm_model.tcp_spec(cell_id, ns))
    base = np.asarray(rcfg["base_position"], float)
    yaw = float(rcfg.get("base_orientation_yaw", 0.0))
    c, s_ = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    gate = pol["axis_tolerance_deg"] - AXIS_GATE_RESERVE_DEG
    axis0 = _R(nominal)[:, 0]
    ref = None if start_q is None else [float(v) for v in start_q]
    sc = self_collision.checker(cell_id, ns)
    tool_cfg = rcfg.get("tool_collision")
    radii = _cell_sphere_radii(cell_id, model, tool_cfg)
    clear_pol = _load_clearance_policy(cell_id)
    margin = _speed_margin(cell_id, resolve_velocity_scale(cell_id, None)
                           * clear_pol["approach_velocity_fraction"])
    boxes = load_static_obstacles(cell_id)
    col = model.createData()

    def self_c(q) -> Optional[float]:
        if not sc.available:
            return None
        hit = sc.closest(q)
        return None if hit is None else float(hit["clearance_m"])

    def tier(clear: Optional[float]) -> int:
        if clear is None or clear >= SELF_MARGIN_M:
            return 0
        return 1 if clear >= SELF_MIN_M else 2

    def free(q) -> bool:
        qv = np.zeros(model.nq)
        qv[:min(len(q), model.nq)] = np.asarray(q, float)[:model.nq]
        return bool(_config_is_free(model, col, qv, base, yaw, [], boxes,
                                    sphere_radius=clear_pol["normal"], tool_cfg=tool_cfg,
                                    sphere_radii=radii, extra_margin=margin))

    def local(pt):
        rel = np.asarray(pt, float) - base
        return np.array([rel[0] * c - rel[1] * s_, rel[0] * s_ + rel[1] * c, rel[2]])

    def quick(R_w, pt, seeds):
        """A fast oriented solve of (R_w, pt), polished, within the axis gate."""
        oMdes = pin.SE3(Rz @ R_w, local(pt))
        for seed in seeds:
            if seed is None:
                continue
            q = np.zeros(model.nq)
            q[:min(len(seed), model.nq)] = np.asarray(seed, float)[:model.nq]
            for _ in range(100):                       # position first, as run_ik
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                ep = oMdes.translation - data.oMf[ee].translation
                if float(np.linalg.norm(ep)) < 1e-3:
                    break
                J3 = pin.computeFrameJacobian(model, data, q, ee,
                                              pin.LOCAL_WORLD_ALIGNED)[:3, :]
                q = pin.integrate(model, q, 0.5 * (
                    J3.T @ np.linalg.solve(J3 @ J3.T + 1e-6 * np.eye(3), ep)))
                q = np.clip(q, jl, ju)
            for _ in range(400):
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                iMd = data.oMf[ee].actInv(oMdes)
                e = pin.log(iMd).vector
                if float(np.linalg.norm(e)) < 1.5e-2:
                    break
                J = -pin.Jlog6(iMd.inverse()) @ pin.computeFrameJacobian(
                    model, data, q, ee, pin.LOCAL)
                q = pin.integrate(model, q, 0.1 * (
                    -J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)))
                q = np.clip(q, jl, ju)
            q = polish_6d(pin, model, data, ee, oMdes, q, jl, ju)
            e = pin.log(data.oMf[ee].actInv(oMdes)).vector
            if float(np.linalg.norm(e[:3])) < 1e-3 and float(np.linalg.norm(e[3:])) < 1e-3:
                ax = Rz.T @ np.asarray(data.oMf[ee].rotation)[:, 0]
                if math.degrees(math.acos(max(-1.0, min(1.0, float(ax @ axis0))))) <= gate:
                    return np.array(q)
        return None

    rise_pref = _transit_rise_preference()

    def line_rise(a, b) -> float:
        """How far above both ends the straight joint line a -> b lifts the TCP —
        what the planner's direct-transit candidate does, and what its rise limit
        refuses."""
        if a is None:
            return 0.0
        from agents.trajectory.tools.robot_kinematics import _fk_ee
        qa = np.asarray(a, float)[:model.nq]
        qb = np.asarray(b, float)[:model.nq]
        z = [float(_fk_ee(model, data, qa + t * (qb - qa), base, yaw)[2])
             for t in np.linspace(0.0, 1.0, 21)]
        return max(0.0, max(z) - max(z[0], z[-1]))

    def screen(R_w, seeds):
        g = quick(R_w, p, seeds)
        if g is None:
            return note("screen", "weld pose does not solve")
        v = quick(R_w, via, [ref, g])
        if v is None:
            return note("screen", "standoff does not solve")
        m = quick(R_w, 0.5 * (via + p), [v])
        if m is None:
            return note("screen", "mid-stroke does not solve")
        cs = [x for x in (self_c(q) for q in (g, v, m)) if x is not None]
        worst = min(cs) if cs else None
        return tier(worst), worst, _joint_travel(list(g), ref), g, line_rise(ref, v)

    ik = getattr(run_ik, "func", run_ik)

    def solve(pt, ori, warm):
        return ik(cell_id=cell_id, robot_ns=ns, x=float(pt[0]), y=float(pt[1]),
                  z=float(pt[2]), qx=ori["x"], qy=ori["y"], qz=ori["z"], qw=ori["w"],
                  warm_start_joints=warm, constrain_orientation=True)

    def axis_error(q) -> float:
        qv = np.zeros(model.nq)
        qv[:min(len(q), model.nq)] = np.asarray(q, float)[:model.nq]
        pin.forwardKinematics(model, data, qv)
        pin.updateFramePlacements(model, data)
        ax = Rz.T @ np.asarray(data.oMf[ee].rotation)[:, 0]
        return math.degrees(math.acos(max(-1.0, min(1.0, float(ax @ axis0)))))

    def verify(ori):
        worst = None

        def ok(r, at_weld):
            nonlocal worst
            if not (r.get("success") and r.get("orientation_constrained")):
                return note("verify", "run_ik: no oriented solution") or False
            if r.get("goal_collision_free") is False:
                return note("verify", "run_ik: best branch collides") or False
            if float(r.get("position_error_m") or 0.0) > 0.01:
                return note("verify", "run_ik: position error") or False
            # What the planner will ship must pass the axis gate it is judged by:
            # run_ik accepts a residual the tilt budget does not have room for.
            if axis_error(r["joint_positions"]) > gate:
                return note("verify", "run_ik: axis error over the gate") or False
            cl = self_c(r["joint_positions"])
            if cl is not None:
                if cl < SELF_MIN_M:
                    return note("verify", f"self clearance {cl:.4f}") or False
                worst = cl if worst is None else min(worst, cl)
            return free(r["joint_positions"]) or note("verify", "fixture") or False

        g = solve(p, ori, ref)
        if not ok(g, True):
            return None
        r = solve(via, ori, ref)
        if not ok(r, False):
            return None
        for i in range(1, DESCENT_STEPS + 1):
            a = i / DESCENT_STEPS
            r = solve(via * (1.0 - a) + p * a, ori, r["joint_positions"])
            if not ok(r, i == DESCENT_STEPS):
                return None
        return tier(worst), worst, g["joint_positions"]

    def result(ori, spin, tilt, phi, info):
        return {"orientation": ori, "spin_deg": round(spin, 2), "tilt_deg": round(tilt, 2),
                "tilt_dir_deg": round(phi, 2),
                "self_clearance_m": None if info[1] is None else round(info[1], 4),
                "joint_positions": [float(v) for v in info[2]]}

    # 1. The specified pose, when its leg is usable as it is.
    from agents.home_pose import home_pose
    home = list(home_pose(cell_id, ns))
    s0 = screen(_R(nominal), [ref, home])
    if s0 is not None and s0[0] == 0 and free(s0[3]):
        # Confirmed with the planner's own IK call (same warm start): the fast
        # screen can land on another arm branch than the planner will.
        v0 = verify(nominal)
        if v0 is not None and v0[0] == 0:
            status["status"] = "specified_ok"
            return None
    # 2./3. Every spin at the specified axis, then each tilt.
    seed_g = None if s0 is None else s0[3]
    cands = candidates(nominal, pol)
    fallback = None
    for group in sorted({t for _s, t, _p, _o in cands}):
        screened = []
        prev = seed_g
        for spin, tilt, phi, ori in (x for x in cands if x[1] == group):
            status["candidates"] = status.get("candidates", 0) + 1
            r = screen(_R(ori), [prev, seed_g, ref, home])
            note("cand", spin, tilt, phi, None if r is None else (r[0], r[1]))
            if r is None:
                continue
            prev = r[3]
            if r[0] <= 1:
                # A pose whose transit from the start would arc over the cell is
                # ranked after one that does not; then, with margin: least joint
                # travel; without: most clearance.
                screened.append((r[0], r[4] > rise_pref,
                                 r[2] if r[0] == 0 else -(r[1] or 0.0),
                                 abs(spin) + tilt, spin, tilt, phi, ori))
        screened.sort(key=lambda t: t[:4])
        for t, _arc, _k, _dev, spin, tilt, phi, ori in screened[:MAX_VERIFY]:
            if fallback is not None and t > 0:
                break
            info = verify(ori)
            if info is None:
                continue
            if info[0] == 0:
                status["status"] = "chosen"
                return result(ori, spin, tilt, phi, info)
            if fallback is None:
                fallback = (ori, spin, tilt, phi, info)
    if fallback is not None:
        status["status"] = "chosen"
        return result(*fallback)
    status["status"] = "none_usable"
    return None


def tcp_orientation(cell_id: str, robot_ns: str, q) -> Optional[dict]:
    """The TCP's world orientation ``{x, y, z, w}`` at joint config ``q`` — the
    pose a kept leg actually ends in, whatever spin it was planned with. None when
    the model cannot be loaded."""
    try:
        import pinocchio as pin
        from agents.trajectory.tools.robot_kinematics import (
            _load_pinocchio_model, _load_robot_config,
        )
        from utils import arm_model
        model, data = _load_pinocchio_model(cell_id, robot_ns)
        rcfg = _load_robot_config(cell_id, robot_ns)
        qv = np.zeros(model.nq)
        qa = np.asarray([float(v) for v in q], float)[: model.nq]
        qv[: len(qa)] = qa
        pin.forwardKinematics(model, data, qv)
        pin.updateFramePlacements(model, data)
        ee = arm_model.ee_frame_id(model, arm_model.tcp_spec(cell_id, robot_ns))
        yaw = float(rcfg.get("base_orientation_yaw", 0.0))
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(
            data.oMf[ee].rotation)
        quat = pin.Quaternion(R)
        return {"x": float(quat.x), "y": float(quat.y), "z": float(quat.z),
                "w": float(quat.w)}
    except Exception:  # noqa: BLE001
        return None
