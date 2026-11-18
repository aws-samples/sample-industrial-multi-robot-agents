"""
Robot kinematics and the collision model — backend-neutral, shared by everything.

This is the single source of truth for "where is the robot and what does it
occupy": URDF/Pinocchio loading, the sphere approximation of each arm (links
PLUS its configured ``tool_collision``), the collision predicate every planner
validates against, and the cell's ``collision:`` thresholds.

It deliberately holds NO planning algorithm. It previously lived inside
``rrt_planner``, which made the legacy Space-Time RRT backend the de-facto owner
of geometry used by the production OMPL backend, the runtime collision monitor
and the MRTA scheduler. Extracting it means:

  * the OMPL backend no longer imports from a legacy planner it does not use;
  * the runtime monitor and planning-time validation provably share one model —
    the class of bug where a plan validated "collision-free" then E-STOPPED on
    the floor because only one side counted the welding torch;
  * deleting or replacing the legacy backend touches no shared geometry.

Import-safe: numpy at module scope, Pinocchio lazily inside the functions that
need it, so this imports on a host with no ROS.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from utils import arm_model, cell_layout, cell_store


# The rate at which the runtime collision monitor samples a moving cell, in Hz.
# Public, and owned HERE rather than by the monitor, for the same reason the
# sphere model is: two copies of one number is how planning-time validation and
# runtime enforcement drift apart. The monitor ticks at this rate; planning-time
# validation must sample at least this densely or it checks a coarser trajectory
# than the one the monitor will actually watch, and passes steps the monitor
# then E-STOPs.
#
# Measured on ``welding_cell_2_small``: a 9.5 s two-arm home step reported minimum
# robot1<->robot2 clearance 0.2080 m at 20 samples and 0.2144 m at 40 (both
# "pass"), but 0.1848 m at 200 and 0.1847 m at 800 — and the monitor, at this
# rate (~238 samples over that leg), measured 0.1862 m and stopped the run.
# Coarse grids alias over clearance dips, and non-monotonically, so a "safe"
# reading at low density is not conservative — it is simply blind.
MONITOR_RATE_HZ = float(os.environ.get("COLLISION_MONITOR_RATE_HZ", "25.0"))


def validation_sample_count(horizon_s: float, floor: int = 2) -> int:
    """How many samples a trajectory of ``horizon_s`` needs to match the monitor.

    ``floor`` is a minimum, never a maximum: a caller may ask for a *finer*
    check than the monitor, never a coarser one.
    """
    rate = max(1.0, float(MONITOR_RATE_HZ))
    ticks = int(np.ceil(max(0.0, float(horizon_s)) * rate)) + 1
    return max(int(floor), 2, ticks)


def _load_robot_config(cell_id: str, robot_ns: str) -> dict:
    cfg = _load_robots_config(cell_id)
    # Match by robot identity (inner key) — supports legacy and shared-namespace layouts.
    for _ns, ns_block in cfg.get("robots", {}).items():
        if isinstance(ns_block, dict):
            for rid, rcfg in ns_block.items():
                if rid == robot_ns and isinstance(rcfg, dict):
                    return rcfg
    raise KeyError(f"Robot '{robot_ns}' not found in {cell_id}")


def _load_robots_config(cell_id: str) -> dict:
    """The cell's ``robots.yaml``, asked of whoever owns the cell.

    Not read off a mounted volume any more: the arm geometry a plan is validated
    against is cell information, and the planner is not its keeper. Absent or
    unreadable raises rather than returning ``{}`` — an empty robot config would
    put every arm at the origin with a 2.1 m default reach and validate happily.
    """
    cfg = cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG,
                               cell_layout.robots_yaml(cell_id).name)
    if cfg is None:
        raise FileNotFoundError(
            f"no robots.yaml for cell '{cell_id}' ({cell_store.describe()})")
    return cfg

def _load_pinocchio_model(cell_id: str, robot_ns: str):
    """The arm's model, via the one loader that also attaches the TCP frame.

    Delegated to :mod:`utils.arm_model`: a referenced cell's URDF ends at the
    flange, so the tool frame this module's FK targets exists only if someone adds
    it at load time. Globbing for the URDF here would skip that.
    """
    model, data, _ee_id = arm_model.load(cell_id, robot_ns)
    return model, data


#: A start counts as ON a recorded seam when its TCP is within this of that
#: weld's point — the weld position tolerance validate_approach gates on.
SEAM_TOL_M = 0.01


def seam_at_start(cell_id: str, robot_ns: str, model, start_q, base, yaw, *,
                  pinned: bool, max_rise_m: float = 0.25) -> Optional[dict]:
    """The weld an arm STARTING at ``start_q`` must lift off from, or None.

    The arm leaves a seam along the axis it entered on (``_cartesian_retreat``);
    which seam is decided by where its TCP IS, not by how the start was given.
    That used to be "the robot's most recent weld, unless ``current_joints`` was
    pinned" — so a chain repair, a coupled re-plan of a PARALLEL step or an edit
    of one weld, all of which pin the start, dragged the tool out of the seam
    sideways (umarell issue 05), and an out-of-order re-plan lifted off the wrong
    weld (issue 15).

    A start is on a weld's seam when its TCP lies on that weld's approach axis —
    within :data:`SEAM_TOL_M` of it laterally, between the weld point and the top
    of that weld's own stroke (its recorded ``stroke_m``, else ``max_rise_m``) up
    the axis (an arm stopped part-way up the stroke is still in the seam's
    channel). Every weld of the run is a candidate
    (``trajectory_cache.weld_poses`` plus the most recent seam); the nearest wins.
    The result carries ``from_point``, where the lift-off starts (the TCP's place
    on the axis), and ``rise_m``, how far up it already is.

    An UNPINNED start with no match keeps today's rule (the most recent seam):
    that start is, by construction, where the last leg ended.
    """
    from agents.trajectory.tools.trajectory_cache import get_last_pose, weld_poses
    last = get_last_pose(cell_id, robot_ns)
    candidates = list(weld_poses(cell_id, robot_ns))
    if last:
        candidates.append((None, last))
    candidates = [(op, p) for op, p in candidates
                  if p.get("orientation") and p.get("target") and p.get("approach_vector")]
    if candidates:
        try:
            q = np.asarray([float(v) for v in start_q][: model.nq], dtype=float)
            tcp = _fk_ee(model, model.createData(), q, np.asarray(base, float), float(yaw))
            best = None
            for op, pose in candidates:
                tgt = np.asarray(pose["target"], float)
                up = -np.asarray(pose["approach_vector"], float)
                up /= (np.linalg.norm(up) + 1e-9)
                rise = float((tcp - tgt) @ up)
                lateral = float(np.linalg.norm(tcp - tgt - rise * up))
                top = float(pose.get("stroke_m") or max_rise_m) + SEAM_TOL_M
                if lateral <= SEAM_TOL_M and -SEAM_TOL_M <= rise <= top:
                    score = lateral + abs(min(rise, 0.0)) + 1e-3 * max(rise, 0.0)
                    if best is None or score < best[0]:
                        best = (score, op, pose, max(rise, 0.0), tgt + max(rise, 0.0) * up)
            if best is not None:
                _s, op, pose, rise, from_point = best
                return {**pose, "operation_id": op, "rise_m": round(rise, 4),
                        "from_point": [float(v) for v in from_point]}
        except Exception:  # noqa: BLE001 — FK unavailable: fall back below
            pass
    return None if pinned else last


def liftoff_blocked_failure(robot_ns: str, seam: dict) -> Optional[dict]:
    """The failure for a leg whose on-axis lift-off from ``seam`` is blocked.

    Without the lift-off the leg would leave the seam with a joint-space transit
    and drag the tool sideways out of it (umarell issue 17), so the leg is refused,
    by every backend, with one reason code the agent's recovery knows. ``None``
    when ``OMPL_ALLOW_BLOCKED_LIFTOFF=1`` restores the old warn-and-drag.
    """
    if os.environ.get("OMPL_ALLOW_BLOCKED_LIFTOFF", "0") == "1":
        return None
    return {
        "robot_ns": robot_ns,
        "reason_code": "liftoff_blocked",
        "from_operation_id": seam.get("operation_id"),
        "from_target": list(seam.get("target") or []),
        "error": ("the on-axis lift-off from the weld this leg starts at is blocked or "
                  "unreachable (by the other arms' motion in this call, an obstacle, or "
                  "IK); without it the leg would drag the tool sideways out of the seam. "
                  "Plan this operation in its own call / step after the other arm has "
                  "moved, give the other arm a different operation in this step, or "
                  "park it first."),
    }


def _fk_ee(model, data, q: np.ndarray, base: np.ndarray, yaw: float) -> np.ndarray:
    """Return world-frame EE position for a config."""
    import pinocchio as pin
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p = data.oMf[arm_model.ee_frame_id(model)].translation
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        base[0] + p[0]*cy - p[1]*sy,
        base[1] + p[0]*sy + p[1]*cy,
        base[2] + p[2],
    ])


# Spacing of the link-fill spheres, in metres. Frame origins alone do NOT cover
# an arm: this cell's links measure up to 1.466 m (joint_a3), so two 0.12 m
# spheres at a link's endpoints leave ~1.2 m of steel invisible to every
# collision check. We therefore walk the kinematic chain and drop additional
# spheres ALONG each link. A step <= 2*sphere_radius makes consecutive spheres
# overlap, so the union approximates a capsule with no gaps.
# Override with LINK_SPHERE_STEP_M (larger = faster planning, coarser geometry).
_LINK_SPHERE_STEP_M = float(os.environ.get("LINK_SPHERE_STEP_M", "0.12"))


def _link_fill_positions(model, data, base, yaw,
                         step: float = _LINK_SPHERE_STEP_M) -> list[np.ndarray]:
    """World-frame sphere centres spaced along each link of the kinematic chain.

    Walks consecutive JOINT origins (``data.oMi``) — topologically correct for a
    serial arm, and the actual load-bearing steel — emitting interior points
    every ``step`` metres. Endpoints are excluded because ``_sphere_positions``
    already covers them, so this returns purely additional coverage.

    Assumes ``forwardKinematics`` has already run.
    """
    if step <= 0:
        return []
    from agents.trajectory.tools import fastgeom
    if fastgeom.AVAILABLE and model.njoints > 2:
        # Compiled, the same points in the same order; the body below is the reference.
        J = np.array([data.oMi[j].translation for j in range(model.njoints)], float)
        return fastgeom.link_fill(J, base, yaw, step)
    cy, sy = np.cos(yaw), np.sin(yaw)

    def _to_world(p) -> np.ndarray:
        return np.array([
            base[0] + p[0]*cy - p[1]*sy,
            base[1] + p[0]*sy + p[1]*cy,
            base[2] + p[2],
        ])

    out: list[np.ndarray] = []
    for j in range(1, model.njoints - 1):
        a = _to_world(data.oMi[j].translation)
        b = _to_world(data.oMi[j + 1].translation)
        seg = float(np.linalg.norm(b - a))
        n = int(seg / step)
        if n < 1:
            continue
        d = (b - a) / seg
        for k in range(1, n + 1):
            t = k * step
            if t >= seg:
                break
            out.append(a + d * t)
    return out


def _sphere_positions(model, data, base, yaw) -> list[np.ndarray]:
    """World-frame arm collision spheres: one per frame origin, PLUS link-fill
    spheres along the chain so the links themselves are covered.

    ORDERING CONTRACT — the first ``len(model.frames)`` entries are the frame
    origins, in frame-index order. Callers slice this list positionally to skip
    base frames (``spheres[2:]`` / ``spheres[3:]`` here and in pinocchio.py), so
    the fill spheres MUST stay appended at the end. Because they are appended,
    every one of those slices includes all fill spheres — which is the intent.
    """
    return _frames_world(data, len(model.frames), base, yaw) \
        + _link_fill_positions(model, data, base, yaw)


def _frames_world(data, n: int, base, yaw) -> list[np.ndarray]:
    """Every frame origin in world coordinates — :func:`_frame_world` for frames
    ``0..n-1``, as ONE vectorised transform (it ran ~570k times per assembly on
    welding_cell_2_small, one small array each)."""
    if n <= 0:
        return []
    T = np.array([data.oMf[i].translation for i in range(n)], float).reshape(n, 3)
    cy, sy = np.cos(yaw), np.sin(yaw)
    W = np.empty((n, 3))
    W[:, 0] = base[0] + T[:, 0] * cy - T[:, 1] * sy
    W[:, 1] = base[1] + T[:, 0] * sy + T[:, 1] * cy
    W[:, 2] = base[2] + T[:, 2]
    return list(W)


# ─── The arm's envelope, and how far it travels before it stops ─────────────
# The skeleton spheres above (one per frame + fill along the joint chain) are a
# cheap proxy: on the KR240 they cover 1-29 % of the castings' hull vertices, so a
# pose pair 300 mm interpenetrated passed the arm-vs-arm gate (umarell issue 04).
# The converter fits ``link_collision`` — spheres covering each link's own
# collision hull — and the EXACT checks (the inter-arm gate, the static checks,
# the monitor) use them on top of the skeleton. The planner's swept-volume tube
# and the compound search keep the skeleton: they are heuristics the gate
# adjudicates afterwards.
#
# A clearance is geometry PLUS the distance a body travels before it stands
# still: d_stop(v) = v * reaction_time_s + v^2 / (2 * brake_decel_m_s2), per
# sphere at its own speed (robots.yaml ``stopping``). Two arms each bring their
# own d_stop into the gap; a fixture brings none.
_ENVELOPE: dict = {}


def _robot_spec(model) -> dict:
    """``{"env": {...} | None, "stop": (t_r, a) | None, "tool": dict | None}`` for a model loaded by
    ``utils.arm_model.load`` (cached per model)."""
    hit = _ENVELOPE.get(id(model))
    if hit is not None and hit[0] is model:
        return hit[1]
    spec: dict = {"env": None, "stop": None, "tool": None}
    owner = arm_model.owner_of(model)
    if owner is not None:
        try:
            cfg = _load_robot_config(*owner) or {}
        except Exception:  # noqa: BLE001
            cfg = {}
        spec["tool"] = cfg.get("tool_collision")
        lc = cfg.get("link_collision") or {}
        fids, offs = [], []
        names = {f.name: i for i, f in enumerate(model.frames)}
        for name, centres in (lc.get("links") or {}).items():
            fid = names.get(name)
            if fid is None:
                continue
            for c in centres or []:
                fids.append(fid)
                offs.append([float(v) for v in c][:3])
        if fids:
            fids_a = np.asarray(fids, int)
            spec["env"] = {
                "fids": fids_a, "offs": np.asarray(offs, float),
                "fixed": np.asarray([arm_model.frame_parent_joint(model.frames[f]) == 0 for f in fids], bool),
                "unique": sorted(set(fids)),
                "radius": float(lc.get("radius_m") or 0.12),
            }
        st = cfg.get("stopping") or {}
        tr, a = st.get("reaction_time_s"), st.get("brake_decel_m_s2")
        if tr is not None and a is not None and float(a) > 0:
            spec["stop"] = (float(tr), float(a))
    _ENVELOPE[id(model)] = (model, spec)
    while len(_ENVELOPE) > 256:              # as many as utils.arm_model keeps owners
        _ENVELOPE.pop(next(iter(_ENVELOPE)))
    return spec


def _envelope_world(model, data, base, yaw):
    """World positions of the model's envelope spheres (FK already run) and which
    are fixed to the world: ``(P (K,3), fixed (K,))``, or ``(None, None)``."""
    env = _robot_spec(model)["env"]
    if env is None:
        return None, None
    P = np.empty_like(env["offs"])
    for fid in env["unique"]:
        sel = env["fids"] == fid
        M = data.oMf[fid]
        P[sel] = env["offs"][sel] @ np.asarray(M.rotation).T + np.asarray(M.translation)
    cy, sy = np.cos(yaw), np.sin(yaw)
    W = np.empty_like(P)
    W[:, 0] = base[0] + P[:, 0] * cy - P[:, 1] * sy
    W[:, 1] = base[1] + P[:, 0] * sy + P[:, 1] * cy
    W[:, 2] = base[2] + P[:, 2]
    return W, env["fixed"]


def stopping_distance(model, speeds) -> np.ndarray:
    """Per-sphere stopping distance at ``speeds`` (m/s); zeros when the robot
    declares no ``stopping`` data (speed then is NOT accounted for)."""
    v = np.asarray(speeds, float)
    stop = _robot_spec(model)["stop"]
    if stop is None:
        return np.zeros_like(v)
    tr, a = stop
    return v * tr + v * v / (2.0 * a)


def path_deviation_rad(rcfg: dict) -> Optional[float]:
    """How far (rad, per joint) a timed path may leave the polyline the planner
    validated — robots.yaml ``path_following.max_deviation_rad``, derived by the
    converter from the arm's geometry. ``None`` when undeclared: follow exactly."""
    v = ((rcfg or {}).get("path_following") or {}).get("max_deviation_rad")
    try:
        return float(v) if v is not None and float(v) > 0 else None
    except (TypeError, ValueError):
        return None


def accounts_for_speed(model) -> bool:
    return _robot_spec(model)["stop"] is not None


def sphere_speed_bound(model, data, base, yaw, P, vel_limits) -> np.ndarray:
    """Upper bound on each point's linear speed (FK already run): a point moves no
    faster than ``sum_j qdot_max_j * |p - o_j|`` over the joints driving it (all
    joints are counted — larger, never smaller). For when no measured speed is
    trustworthy: an unmeasured speed must never buy a smaller stopping distance."""
    P = np.asarray(P, float).reshape(-1, 3)
    lim = np.abs(np.asarray(vel_limits, float))
    n = min(len(lim), model.njoints - 1)
    if n == 0:
        return np.full(len(P), np.inf)
    cy, sy = np.cos(yaw), np.sin(yaw)
    O = np.asarray([np.asarray(data.oMi[j].translation, float) for j in range(1, n + 1)])
    W = np.column_stack([base[0] + O[:, 0] * cy - O[:, 1] * sy,
                         base[1] + O[:, 0] * sy + O[:, 1] * cy, base[2] + O[:, 2]])
    return np.linalg.norm(P[:, None, :] - W[None, :, :], axis=2) @ lim[:n]


def _gate_points(model, data, q, base, yaw, tool_cfg) -> np.ndarray:
    """Every sphere the arm-vs-arm gate checks: skeleton + tool + envelope."""
    import pinocchio as pin
    pin.forwardKinematics(model, data, np.asarray(q, float))
    pin.updateFramePlacements(model, data)
    pts = _sphere_positions(model, data, base, yaw) + _tool_spheres(model, data, base, yaw, tool_cfg)
    P = np.asarray(pts, float).reshape(-1, 3)
    E, _fixed = _envelope_world(model, data, base, yaw)
    return P if E is None else np.vstack([P, E])


def _tool_frame_start(model, tool_cfg: "Optional[dict]") -> int:
    """First frame index that is TOOL geometry rather than arm structure.

    The tool hangs off ``tool_collision.mount_link``, so every frame AFTER the
    deepest frame matching that substring belongs to the end-effector: for this
    cell's ``kr240r2900__sticktool``, ``link_6`` is frame 15 and frames 16
    (``welding_tcp_joint``) and 17 (``welding_tcp``) are the torch.

    The DEEPEST match is used because ``mount_link`` is a substring: a cell
    mounting on ``link_1`` would also match ``link_10``, and taking the first hit
    would classify the entire outer arm as tool.

    Returns ``len(model.frames)`` — i.e. "nothing here is tool" — when no tool is
    configured or the mount link matches no frame. That is the fail-closed answer:
    an unrecognised mount leaves every sphere at the (larger) arm radius rather
    than silently thinning the arm.
    """
    n = len(model.frames)
    token = (tool_cfg or {}).get("mount_link", "") if tool_cfg else ""
    if not token:
        return n
    idx = max((i for i, f in enumerate(model.frames) if token in f.name), default=-1)
    return n if idx < 0 else idx + 1


def _sphere_radii(
    model, arm_radius: float, tool_radius: float, tool_cfg: "Optional[dict]" = None,
) -> np.ndarray:
    """Per-sphere collision radius: ``arm_radius`` for the arm, ``tool_radius``
    for the end-effector frames.

    A collision radius is a property of the BODY, not of the motion phase. The
    heavy castings measure 0.32–0.45 m across; the welding stick's own URDF
    collision hull measures 0.084 m, i.e. r ≈ 0.042 m. Modelling both at one
    number is what forced the choice between a torch that cannot reach a seam
    (0.12 everywhere) and an arm that is invisible to the fixture (0.05
    everywhere, the retired ``approach`` relaxation).

    Length is ``len(model.frames)`` — the frame-origin block of the ORDERING
    CONTRACT in :func:`_sphere_positions`. It deliberately does NOT cover the
    link-fill spheres appended after that block: those interpolate JOINT origins,
    so they are always load-bearing arm steel, and callers treat any index past
    the end of this vector as arm. That also keeps the vector independent of the
    configuration, so it can be built once per robot and reused.
    """
    radii = np.full(len(model.frames), float(arm_radius), dtype=float)
    radii[_tool_frame_start(model, tool_cfg):] = float(tool_radius)
    return radii


def _cell_sphere_radii(cell_id: str, model, tool_cfg: "Optional[dict]" = None):
    """The radius vector this cell declares, or ``None`` when it declares none.

    ``None`` means "use the caller's uniform radius", which is exactly today's
    behaviour — so a cell that does not declare ``tool_link_radius`` is unaffected
    by this mechanism, and one that declares it equal to ``robot_link_radius``
    collapses to the same thing.

    This is the ONE builder both the planner and the runtime collision monitor
    call. They cannot be given different vectors for the same robot, which is the
    whole point: a configuration that is legal to the planner and illegal to the
    monitor is a plan that e-stops on the floor.
    """
    policy = _load_clearance_policy(cell_id)
    arm, tool = policy["normal"], policy["tool"]
    if abs(tool - arm) < 1e-12:
        return None
    # A model that cannot enumerate its frames cannot carry a per-frame vector.
    # ``None`` then means the caller's uniform radius, which is the ARM radius —
    # the LARGER of the two — so an un-enumerable model over-models the torch
    # rather than thinning the arm. Fail-closed, and it keeps this builder usable
    # by callers holding a reduced kinematic model.
    if not getattr(model, "frames", None):
        return None
    return _sphere_radii(model, arm, tool, tool_cfg)


def _frame_world(data, i: int, base, yaw) -> np.ndarray:
    """One frame origin in world coordinates (base offset + base yaw applied)."""
    p = data.oMf[i].translation
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        base[0] + p[0]*cy - p[1]*sy,
        base[1] + p[0]*sy + p[1]*cy,
        base[2] + p[2],
    ])


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cyw, syw = np.cos(y), np.sin(y)
    Rz = np.array([[cyw, -syw, 0.0], [syw, cyw, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


_TOOL_OFFSETS_CACHE: dict = {}
_TOOL_MOUNT_IDX: dict = {}


def _tool_offsets(tool_cfg: "Optional[dict]") -> list[np.ndarray]:
    """Collision-sphere centres for the end-effector tool, expressed in the
    mount-link frame, built ENTIRELY from ``tool_cfg`` (no hard-coded geometry).

    ``tool_cfg`` schema (from the robot's entry in robots.yaml, key
    ``tool_collision``)::

        mount_link: <substring matched against a frame name>
        axis: {direction: [x,y,z], start_m, end_m, step_m}   # torch/electrode
        plates:                                              # flat plate(s)
          - {center: [x,y,z], rpy: [r,p,y], size: [w,h], grid: [nu,nv]}
        spheres:                                             # explicit centres
          - [x, y, z]

    ``spheres`` is what scripts/convert_isaac_scene.py writes: centres fitted to
    the tool's own meshes (utils/tool_geometry.py). ``axis`` and ``plates`` are the
    older hand-authored forms; all three may be combined.

    Returns an empty list when ``tool_cfg`` is falsy, so a robot with no tool
    geometry configured adds nothing (the planner stays generic)."""
    if not tool_cfg:
        return []
    key = id(tool_cfg)
    cached = _TOOL_OFFSETS_CACHE.get(key)
    if cached is not None:
        return cached
    offs: list[np.ndarray] = []
    ax = tool_cfg.get("axis")
    if ax:
        d = np.array(ax.get("direction", [-1.0, 0.0, 0.0]), dtype=float)
        n = float(np.linalg.norm(d)) or 1.0
        d = d / n
        s = float(ax.get("start_m", 0.15))
        e = float(ax.get("end_m", 0.9))
        step = float(ax.get("step_m", 0.15))
        x = s
        while x <= e + 1e-9:
            offs.append(d * x)
            x += step
    for plate in tool_cfg.get("plates", []) or []:
        c = np.array(plate.get("center", [0.0, 0.0, 0.0]), dtype=float)
        rpy = plate.get("rpy", [0.0, 0.0, 0.0])
        Rp = _rpy_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        w, h = plate.get("size", [0.3, 0.45])
        gu, gv = plate.get("grid", [5, 5])
        for u in np.linspace(-float(w) / 2.0, float(w) / 2.0, int(gu)):
            for v in np.linspace(-float(h) / 2.0, float(h) / 2.0, int(gv)):
                offs.append(c + Rp @ np.array([u, v, 0.0]))
    for c in tool_cfg.get("spheres", []) or []:
        offs.append(np.array([float(v) for v in c][:3], dtype=float))
    _TOOL_OFFSETS_CACHE[key] = offs
    return offs


def _tool_spheres(model, data, base, yaw, tool_cfg: "Optional[dict]" = None) -> list[np.ndarray]:
    """World-frame tool sphere centres from ``tool_cfg`` (assumes FK already
    run). Empty when no tool is configured."""
    offs = _tool_offsets(tool_cfg)
    if not offs:
        return []
    token = tool_cfg.get("mount_link", "")
    mkey = (id(model), token)
    idx = _TOOL_MOUNT_IDX.get(mkey)
    if idx is None:
        idx = next((i for i, f in enumerate(model.frames) if token and token in f.name), -1)
        _TOOL_MOUNT_IDX[mkey] = idx
    if idx < 0:
        return []
    oMf = data.oMf[idx]
    R = oMf.rotation
    t = oMf.translation
    cy, sy = np.cos(yaw), np.sin(yaw)
    out = []
    for off in offs:
        p = R @ off + t
        out.append(np.array([base[0] + p[0]*cy - p[1]*sy,
                             base[1] + p[0]*sy + p[1]*cy,
                             base[2] + p[2]]))
    return out


def _all_spheres(model, data, base, yaw, tool_cfg: "Optional[dict]" = None) -> list[np.ndarray]:
    """Frame-origin spheres PLUS the configured tool spheres. Assumes FK +
    updateFramePlacements have run. Used by the inter-robot coordination so a
    configured tool/plate is never ignored."""
    return _sphere_positions(model, data, base, yaw) + _tool_spheres(model, data, base, yaw, tool_cfg)


def _config_spheres(model, data_r, q: np.ndarray, base, yaw, tool_cfg: "Optional[dict]" = None) -> list[np.ndarray]:
    import pinocchio as pin
    pin.forwardKinematics(model, data_r, q)
    pin.updateFramePlacements(model, data_r)
    return _all_spheres(model, data_r, base, yaw, tool_cfg)


def _read_obstacles_yaml(cell_id: str) -> Optional[dict]:
    """The cell's raw ``obstacles.yaml`` mapping, or ``None`` when the cell has none.

    One reader, because obstacles.yaml is read for two different things — the
    static obstacle list (:func:`load_static_obstacles`) and the clearance policy
    (:func:`_load_clearance_policy`) — and they used to be four copies of an
    ``os.path.exists`` + ``yaml.safe_load`` block scattered across the planner,
    each with its own idea of what a failure meant.

    Absent is ``None`` and is a real answer: a cell may genuinely declare no
    fixtures. Unreachable and unparseable both raise, because the alternative is a
    planner that routes an arm through a fixture it could not read about.
    """
    return cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG,
                                cell_layout.obstacles_yaml(cell_id).name)


def load_static_obstacles(cell_id: str) -> list:
    """The cell's declared static obstacles — fixtures, tables, walls.

    The one entry point for every planner component that needs to know what is in
    the way. It raises on failure and returns ``[]`` only when the cell really
    declares nothing: "no obstacles" is the single most dangerous wrong answer
    this repo can produce, and it used to be the default for any read error.
    """
    cfg = _read_obstacles_yaml(cell_id)
    if not cfg:
        return []
    obstacles = cfg.get("obstacles", [])
    return normalize_static_obstacles(obstacles) if isinstance(obstacles, list) else []


def normalize_static_obstacles(obstacles: list) -> list:
    """obstacles.yaml entries as the checks read them.

    ``is_workpiece: true`` in a cell's file means what the file says it means —
    "the tool tip may touch it (arm links are still checked)" — so it is read as
    ``tool_contact: true``: exempt for the TOOL's bodies only. Taken literally
    (:func:`_compile_static` skips an ``is_workpiece`` box for every body) it hid
    the box from the whole arm: on welding_cell_2_small, 210 boxes, a quarter of the car
    body, were invisible to every link.
    """
    out = []
    for o in obstacles or []:
        if isinstance(o, dict) and o.get("is_workpiece"):
            o = {k: v for k, v in o.items() if k != "is_workpiece"}
            o["tool_contact"] = True
        out.append(o)
    return out


def _load_collision_config(cell_id: str) -> dict:
    """The cell's ``collision:`` block — the SAME numbers the runtime monitor enforces.

    The monitor e-stops at ``clearance <= warning_distance``, so a plan validated
    only against penetration (clearance >= 0) can be geometrically valid and still
    be halted on the floor. Planning-time validation reads this so the gate it
    applies is the gate execution will apply.
    """
    try:
        coll = (_load_robots_config(cell_id) or {}).get("collision", {}) or {}
    except cell_store.CellStoreUnavailable:
        # "I could not ask" is not "the cell declares no collision policy":
        # falling back to the built-in gate here would validate a plan against
        # numbers the floor does not enforce.
        raise
    except Exception:  # noqa: BLE001
        coll = {}
    return {
        "warning_distance": float(coll.get("warning_distance", 0.20)),
        "min_clearance": float(coll.get("min_clearance", 0.05)),
    }


# Transit speed when a cell declares no ``planning.max_velocity_scaling`` — the
# value the planner has always used, so undeclared cells keep today's timing.
DEFAULT_TRANSIT_VELOCITY_SCALE = 0.5

# Fallbacks for the rest of the ``planning:`` speed keys, for a cell that does
# not declare them. They are today's values, so an undeclared cell moves exactly
# as before; a declared cell's robots.yaml is the source of truth.
PLANNING_SPEED_DEFAULTS: dict[str, float] = {
    "home_velocity_scale":   0.4,   # planned return to HOME + lift-off (gentle)
    "acceleration_factor":   2.5,   # accel = factor × vel when a robot declares
                                    # no joint_acceleration_limits
    "interlock_move_time_s": 2.5,   # unplanned single-point interlock move
    "home_return_time_s":    7.0,   # unplanned fallback return to HOME
}


# Planning settings a cell converted before they existed may not declare: absent
# (or invalid) reads as the default here, which is also what the converter writes
# (scripts/convert_isaac_scene.py, planning: block). Kept apart from
# PLANNING_SPEED_DEFAULTS, which every shipped cell is required to declare.
PLANNING_SETTING_DEFAULTS: dict[str, float] = {
    "home_liftoff_m":      0.10,  # on-axis lift-off out of the last seam on the way HOME
    "min_home_liftoff_m":  0.05,  # shortest lift-off still tried before the return is refused
    "completion_margin_s": 1.0,   # executor slack after every ExecuteTrajectory/MoveHome leaf
}
# Settings for which 0 is a legal value (no slack); the rest must be > 0.
_NON_NEGATIVE_SETTINGS = frozenset({"completion_margin_s"})


def planning_setting(cell_id: str, key: str) -> float:
    """One of :data:`PLANNING_SETTING_DEFAULTS` from the cell's ``planning:`` block.

    Undeclared or invalid reads as the default. A store outage is NOT read as
    "undeclared" (it raises), the same rule as :func:`_load_planning_config`.
    """
    import math
    default = PLANNING_SETTING_DEFAULTS[key]
    try:
        plan = (_load_robots_config(cell_id) or {}).get("planning", {}) or {}
    except cell_store.CellStoreUnavailable:
        raise
    except Exception:  # noqa: BLE001
        plan = {}
    if not isinstance(plan, dict):
        return default
    try:
        v = float(plan.get(key))
    except (TypeError, ValueError):
        return default
    ok = v >= 0.0 if key in _NON_NEGATIVE_SETTINGS else v > 0.0
    return v if ok and math.isfinite(v) else default


def _positive(value, fallback: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return v if v > 0 else fallback


def _load_planning_config(cell_id: str) -> dict:
    """The cell's ``planning:`` block from ``robots.yaml`` — every speed and
    acceleration number the planner and executor use.

    Keys::

      max_velocity_scaling   the cell's speed ceiling as a fraction of the joint
                             velocity limits. It is both the transit speed a plan
                             runs at when the caller does not ask for one, and the
                             cap on any speed a caller does ask for.
      default_velocity_scale what a caller who names no speed gets
      declared               whether the cell set a ceiling at all
      home_velocity_scale    speed of the planned return to HOME (capped too)
      acceleration_factor    accel limit = factor × velocity limit, for a robot
                             with no ``joint_acceleration_limits``
      interlock_move_time_s  duration of the single-point move to an interlock pose
      home_return_time_s     duration of the unplanned fallback return to HOME

    Undeclared: no cap (1.0), the historic 0.5 default, and
    :data:`PLANNING_SPEED_DEFAULTS` — i.e. today's behaviour.
    """
    try:
        plan = (_load_robots_config(cell_id) or {}).get("planning", {}) or {}
    except cell_store.CellStoreUnavailable:
        raise
    except Exception:  # noqa: BLE001
        plan = {}
    out = {k: _positive(plan.get(k), v) for k, v in PLANNING_SPEED_DEFAULTS.items()}
    raw = plan.get("max_velocity_scaling")
    try:
        cap = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        cap = None
    if cap is None or not cap > 0:
        out.update({"max_velocity_scaling": 1.0,
                    "default_velocity_scale": DEFAULT_TRANSIT_VELOCITY_SCALE,
                    "declared": False})
    else:
        cap = min(cap, 1.0)
        out.update({"max_velocity_scaling": cap, "default_velocity_scale": cap,
                    "declared": True})
    out["home_velocity_scale"] = min(out["home_velocity_scale"],
                                     out["max_velocity_scaling"])
    return out


def resolve_home_velocity_scale(cell_id: str, requested: Optional[float]) -> float:
    """Speed of a return to HOME: the request (capped), else the cell's
    ``planning.home_velocity_scale``."""
    cfg = _load_planning_config(cell_id)
    if requested is None:
        return cfg["home_velocity_scale"]
    return max(min(_positive(requested, cfg["home_velocity_scale"]),
                   cfg["max_velocity_scaling"]), 0.01)


def joint_acceleration_limits(cell_id: str, rcfg: dict) -> Optional[list]:
    """A robot's acceleration limits: declared, else ``acceleration_factor`` ×
    its velocity limits. ``None`` only when it declares neither."""
    declared = rcfg.get("joint_acceleration_limits")
    if declared is not None:
        return list(declared)
    vel = rcfg.get("joint_velocity_limits")
    if vel is None:
        return None
    factor = _load_planning_config(cell_id)["acceleration_factor"]
    return [float(v) * factor for v in vel]


def resolve_velocity_scale(cell_id: str, requested: Optional[float]) -> float:
    """The transit speed a plan actually runs at: the request, capped by the cell.

    ``None`` means "the cell's speed" (``planning.max_velocity_scaling``). A value
    above the ceiling is clamped to it, never honoured — the ceiling is the cell's
    declaration, not a suggestion.
    """
    cfg = _load_planning_config(cell_id)
    if requested is None:
        return cfg["default_velocity_scale"]
    try:
        vs = float(requested)
    except (TypeError, ValueError):
        return cfg["default_velocity_scale"]
    if not vs > 0:
        return cfg["default_velocity_scale"]
    return max(min(vs, cfg["max_velocity_scaling"]), 0.01)


def _load_clearance_policy(cell_id: str) -> dict:
    """The cell's clearance policy from ``obstacles.yaml`` — sphere radii + the
    speed allowance. Read by BOTH the planner and the runtime monitor.

    Keys::

      robot_link_radius          arm-structure sphere radius            (0.12)
      tool_link_radius           end-effector sphere radius             (defaults
                                 to robot_link_radius — i.e. no tool model)
      approach_link_radius       RETIRED phase relaxation, still parsed
      default_collision_margin   fallback per-obstacle margin           (0.05)
      speed_margin_full_speed_m  extra clearance a body must hold when moving at
                                 FULL commanded speed, tapering to 0 at rest
      approach_velocity_fraction how much slower the terminal working stroke runs
                                 than transit, as a fraction of the transit scale
                                 (0.4) — the input to the speed allowance above

    ``speed_margin_full_speed_m`` is the physical content of what used to be the
    two-radius ``approach`` switch: a fast body needs more room than a slow one
    because the controller cannot stop instantly (reaction latency × tip speed),
    which is why the terminal working stroke — deliberately slow — may legitimately
    come closer to the fixture than a transit move. Expressing it as a SPEED term
    rather than as a second radius means the geometry stays honest (a casting does
    not shrink because the arm is on its final approach) and both components can
    compute it from the same formula: see :func:`_speed_margin`.

    It defaults to 0.0 — the gate the runtime enforces today — because the value is
    a property of the controller's e-stop latency, which this repo does not yet
    declare. Set it per cell once that latency is measured.
    """
    normal, margin = 0.12, 0.05
    approach, tool, speed_margin, approach_vs_frac = 0.05, None, 0.0, 0.4
    # The TOOL's standoff from static bodies (floor, fixtures), when the cell
    # declares one; else each obstacle's own collision_margin, as for the arm.
    tool_margin = None
    cfg = _read_obstacles_yaml(cell_id)
    if cfg is not None:
        normal = float(cfg.get("robot_link_radius", normal))
        approach = float(cfg.get("approach_link_radius", approach))
        margin = float(cfg.get("default_collision_margin", margin))
        speed_margin = float(cfg.get("speed_margin_full_speed_m", speed_margin))
        approach_vs_frac = float(
            cfg.get("approach_velocity_fraction", approach_vs_frac))
        if cfg.get("tool_link_radius") is not None:
            tool = float(cfg["tool_link_radius"])
        if cfg.get("tool_collision_margin") is not None:
            tool_margin = float(cfg["tool_collision_margin"])
    return {
        "normal": normal,
        "approach": approach,
        # No declared tool geometry ⇒ the tool is modelled as arm, which is the
        # larger radius: undeclared cells keep exactly today's behaviour.
        "tool": normal if tool is None else tool,
        "default_margin": margin,
        "speed_margin_full_speed_m": max(0.0, speed_margin),
        "approach_velocity_fraction": approach_vs_frac,
        "tool_margin": tool_margin,
    }


def _speed_margin(cell_id: str, velocity_scale: float) -> float:
    """Extra clearance (m) required of a body moving at ``velocity_scale``.

    ``velocity_scale`` is the fraction of the robot's top speed: the planner knows
    it per phase (transit vs the slower approach stroke), and the monitor MEASURES
    it from consecutive published joint states. One formula, two inputs of the same
    kind, so "free" means the same thing on both sides — a plan cannot be validated
    against a laxer gate than the one the floor enforces.

    Linear in speed: the reaction distance is latency × speed, and latency is a
    constant of the controller. At rest the allowance is zero, so a stationary arm
    is judged on geometry alone.
    """
    return _speed_margin_from(
        _load_clearance_policy(cell_id)["speed_margin_full_speed_m"], velocity_scale)


def _speed_margin_from(full_speed_margin: float, velocity_scale: float) -> float:
    """The formula itself, with the cell's full-speed allowance already in hand.

    Split out for the runtime monitor, which needs this on every poll and must not
    re-read ``obstacles.yaml`` at 25 Hz to get it. Callers that can afford the read
    should use :func:`_speed_margin`; either way the arithmetic lives here once.
    """
    return float(full_speed_margin) * min(max(float(velocity_scale), 0.0), 1.0)


def gate_groups(model, n_points: int) -> list:
    """Index slices of :func:`_gate_points` that move together: the skeleton and
    tool block, then each envelope link (contiguous, in fit order)."""
    env = _robot_spec(model)["env"]
    if env is None or n_points < len(env["fids"]):
        return [slice(0, n_points)]
    f = env["fids"]
    start = n_points - len(f)
    out = [slice(0, start)] if start else []
    run = 0
    for k in range(1, len(f) + 1):
        if k == len(f) or f[k] != f[run]:
            out.append(slice(start + run, start + k))
            run = k
    return out


def _check_robot_pair_collision(
    s1: list[np.ndarray], s2: list[np.ndarray], radius: float = 0.12,
    stop1=None, stop2=None, groups1=None, groups2=None, radii1=None, radii2=None,
) -> tuple[bool, float]:
    """Returns (collision, min_clearance). ``stop1``/``stop2`` are per-sphere
    stopping distances subtracted from each pair's gap (speed-dependent clearance).

    ``groups1``/``groups2`` (:func:`gate_groups`) make it a branch and bound: each
    group's bounding sphere gives a lower bound on its pairs' gaps, and a group
    pair is only measured sphere by sphere while that bound is below the best so
    far — the same minimum, without 700 x 700 distances for arms metres apart."""
    A = np.asarray(s1, float).reshape(-1, 3)
    B = np.asarray(s2, float).reshape(-1, 3)
    if not len(A) or not len(B):
        return False, float("inf")
    # Per-sphere radii (``radii1``/``radii2``) fold into the per-sphere offsets:
    # a gap is centre distance - r_a - r_b - stop_a - stop_b either way.
    ra_ = np.full(len(A), float(radius)) if radii1 is None else np.asarray(radii1, float)
    rb_ = np.full(len(B), float(radius)) if radii2 is None else np.asarray(radii2, float)
    sa = (np.zeros(len(A)) if stop1 is None else np.asarray(stop1, float)) + ra_ - radius
    sb = (np.zeros(len(B)) if stop2 is None else np.asarray(stop2, float)) + rb_ - radius
    from agents.trajectory.tools import fastgeom
    if fastgeom.AVAILABLE:
        # Compiled, the same branch and bound; the NumPy body below is the reference.
        min_d = fastgeom.pair_min_gap(A, B, sa, sb, groups1, groups2, radius)
        return min_d < 0, min_d
    ga = groups1 or [slice(0, len(A))]
    gb = groups2 or [slice(0, len(B))]

    def bound(P, S, groups):
        c = np.asarray([P[g].mean(0) for g in groups])
        r = np.asarray([np.sqrt(((P[g] - c[k]) ** 2).sum(-1)).max() for k, g in enumerate(groups)])
        return c, r, np.asarray([S[g].max() for g in groups])

    ca, ra, ma = bound(A, sa, ga)
    cb, rb, mb = bound(B, sb, gb)
    lb = (np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
          - ra[:, None] - rb[None, :] - 2 * radius - ma[:, None] - mb[None, :])
    min_d = float("inf")
    for flat in np.argsort(lb, axis=None):
        u, v = divmod(int(flat), len(gb))
        if lb[u, v] >= min_d:
            break
        a, b = A[ga[u]], B[gb[v]]
        D = (np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)) - 2 * radius
             - sa[ga[u]][:, None] - sb[gb[v]][None, :])
        min_d = min(min_d, float(D.min()))
    return min_d < 0, min_d


# ─── THE inter-arm gate ───────────────────────────────────────────────────────
#
# "Do these arms clear each other while moving?" has exactly ONE answer, and this
# is it. ``validate_trajectory_set`` is a thin @tool wrapper over this function and
# the OMPL planner calls it directly, so a route the planner accepts and a route
# the gate accepts are the same set by construction.
#
# It did not use to be. The planner had its own copy of this decision: each arm was
# planned against the SWEPT TUBE of every arm already planned in the same call —
# every place that arm would ever be, with no reference to *when*. Measured on
# ``bt_1787955158`` (welding_cell_2_small), the two copies had drifted in BOTH
# directions:
#
#   * too strict — step 2 ``robot2``/``W07``: with the tube, the only routes the
#     planner could find rose 0.196 m / 2.287 m or failed outright, and it shipped
#     a 1.335 m arc. Without it, a route with 0.000 m rise exists and clears this
#     gate at 0.2488 m (band 0.2 m). Step 11's ``MoveHome`` was worse: the tube
#     admitted NO path at the full margin, and the ladder's fallback shipped a
#     0.585 m arc where a 0.000 m route clears at 0.5314 m.
#   * too lax — the same step-2 tube ACCEPTED a 0.196 m route that this gate then
#     measures at 0.1551 m, inside the band. Satisfying the tube never implied
#     satisfying the monitor.
#
# So the tube was neither necessary nor sufficient, which is the whole argument for
# there being one owner rather than two.

def inter_robot_clearance(
    cell_id: str,
    trajectories: dict,
    collision_radius_m: float = 0.12,
    n_samples: int = 20,
    warning_distance_m: "Optional[float]" = None,
) -> dict:
    """Clearance between every pair of arms in ``trajectories``, on a shared clock.

    ``trajectories`` is ``{robot_ns: [{"positions": [...], "time_s": float}, ...]}``.
    Arms are compared at the same wall-clock instant, because two arms can only
    collide if they occupy conflicting configurations *at the same time*; an arm
    that has finished holds its final configuration (it is parked there, and still
    occupies space).

    ``collision_free`` means clearance stays ABOVE the cell's ``warning_distance``,
    not merely that the arms never touch: the runtime monitor E-STOPs at the band,
    so a plan that only avoids contact still stops the cell.

    ``n_samples`` is a FLOOR. The grid is never coarser than the monitor's own
    sampling rate — see ``validation_sample_count``.

    ``warning_distance_m`` overrides the band this compares against; ``None`` reads
    the cell's own ``warning_distance``. Callers that have ALREADY resolved the
    cell's collision policy pass it in rather than making this reload it, so one
    verdict cannot be computed against two different bands within one call.

    Raises on unloadable models/configs; callers that need a verdict either way
    should catch. Fewer than two non-empty trajectories is not an error: there is
    nothing to check, so it returns ``collision_free`` with infinite clearance.
    """
    from agents.trajectory.tools.time_parameterization import _sample_at_time

    names = [ns for ns, tr in (trajectories or {}).items() if tr]
    if len(names) < 2:
        return {
            "collision_free": True,
            "min_clearance_m": float("inf"),
            "robots": names,
            "pairs_checked": 0,
            "note": "fewer than two non-empty trajectories — nothing to check",
        }

    models, datas, bases, yaws, tool_cfgs = {}, {}, {}, {}, {}
    for ns in names:
        models[ns], _ = _load_pinocchio_model(cell_id, ns)
        datas[ns] = models[ns].createData()
        cfg = _load_robot_config(cell_id, ns)
        bases[ns] = np.array(cfg["base_position"])
        yaws[ns] = float(cfg.get("base_orientation_yaw", 0.0))
        # The welding torch is part of the robot's swept volume; omitting it here
        # while the runtime monitor includes it is how a "collision-free" plan
        # e-stops on the floor.
        tool_cfgs[ns] = cfg.get("tool_collision")

    warning_distance = (
        float(warning_distance_m) if warning_distance_m is not None
        else _load_collision_config(cell_id)["warning_distance"])

    horizon = max(float(trajectories[ns][-1].get("time_s", 0.0)) for ns in names)
    n = validation_sample_count(horizon, floor=int(n_samples))
    grid = [horizon * i / (n - 1) for i in range(n)] if horizon > 0 else [0.0]

    min_clearance = float("inf")
    worst_pair: "Optional[list]" = None
    collision_t = None
    colliding: dict = {}

    # Every sphere the gate checks (skeleton + tool + the fitted envelope), at
    # every grid instant, then each sphere's speed along the grid — the speed it
    # will move at when that instant is executed.
    pts = {ns: np.stack([_gate_points(models[ns], datas[ns],
                                      _sample_at_time(trajectories[ns], t),
                                      bases[ns], yaws[ns], tool_cfgs[ns]) for t in grid])
           for ns in names}
    stops = {}
    for ns in names:
        if len(grid) > 1 and accounts_for_speed(models[ns]):
            vel = np.gradient(pts[ns], np.asarray(grid), axis=0)
            stops[ns] = stopping_distance(models[ns], np.linalg.norm(vel, axis=2))
        else:
            stops[ns] = None
    min_geometric = float("inf")
    groups = {ns: gate_groups(models[ns], pts[ns].shape[1]) for ns in names}
    # The envelope is only an envelope at the radius it was fitted at (the
    # converter's link_collision.radius_m): a caller's smaller collision_radius_m
    # must not thin it, or the gate passes poses whose hulls interpenetrate
    # (umarell issue 04). Its spheres — the LAST rows of the gate points — get
    # that floor; the skeleton and the tool keep the caller's radius.
    radii = {}
    for ns in names:
        env = _robot_spec(models[ns])["env"]
        r = np.full(pts[ns].shape[1], float(collision_radius_m))
        if env is not None:
            k = len(env["offs"])
            r[-k:] = max(float(collision_radius_m), float(env.get("radius") or 0.0))
        radii[ns] = r

    for k, t in enumerate(grid):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                _gcol, geometric = _check_robot_pair_collision(
                    pts[a][k], pts[b][k], collision_radius_m,
                    groups1=groups[a], groups2=groups[b],
                    radii1=radii[a], radii2=radii[b])
                min_geometric = min(min_geometric, geometric)
                if stops[a] is None and stops[b] is None:
                    col, clearance = _gcol, geometric
                else:
                    col, clearance = _check_robot_pair_collision(
                        pts[a][k], pts[b][k], collision_radius_m,
                        None if stops[a] is None else stops[a][k],
                        None if stops[b] is None else stops[b][k],
                        groups[a], groups[b], radii1=radii[a], radii2=radii[b])
                if clearance < min_clearance:
                    min_clearance, worst_pair = clearance, [a, b]
                # The monitor halts at the warning band, so flag anything that
                # enters it — not just actual penetration.
                if col or clearance <= warning_distance:
                    key = f"{a}|{b}"
                    if key not in colliding:
                        colliding[key] = {"robots": [a, b], "time_s": round(t, 4),
                                          "clearance_m": round(clearance, 4),
                                          "penetrating": bool(col)}
                    if collision_t is None:
                        collision_t = round(t, 4)

    result: dict = {
        "collision_free":     min_clearance > warning_distance,
        "min_clearance_m":    round(min_clearance, 4),
        "warning_distance_m": warning_distance,
        "robots":             names,
        "pairs_checked":      len(names) * (len(names) - 1) // 2,
        "worst_pair":         worst_pair,
        "samples_checked":    len(grid),
        "monitor_rate_hz":    MONITOR_RATE_HZ,
        # min_clearance_m is net of both arms' stopping distances when the robots
        # declare them; this is the bare geometric gap of the same spheres.
        "min_geometric_clearance_m": round(min_geometric, 4),
        "speed_accounted": {ns: stops[ns] is not None for ns in names},
    }
    if colliding:
        result["collision_time_s"] = collision_t
        result["colliding_pairs"] = list(colliding.values())
    return result


# ─── Shared collision predicate ───────────────────────────────────────────────

# ─── Static obstacles: one generic, vectorised check ──────────────────────────
# Every static body a cell declares in obstacles.yaml — boxes, spheres AND planes
# (the floor is a plane like any other) — is checked here against the arm's
# spheres and the tool's static spheres, each at its own radius plus the
# obstacle's collision_margin (+ the speed allowance). Used by the OMPL validity
# predicate, the collision diagnosis, the emitted-trajectory backstop, IK branch
# ranking and the runtime monitor, so they cannot disagree.
#
# Schema of a plane (obstacles.yaml): ``normal`` n and ``offset`` d; a point x is
# on the FREE side when n.x + d >= 0, and a sphere of radius r is clear when
# n.c + d >= r + collision_margin. A cell that declares no plane gets the physical
# floor: z = 0, no margin — the floor is never optional.
DEFAULT_FLOOR = {"id": "floor", "type": "plane", "normal": [0.0, 0.0, 1.0],
                 "offset": 0.0, "collision_margin": 0.0, "implicit": True}

_COMPILED_STATIC: dict = {}


def _compile_static(static_obstacles: list) -> dict:
    """obstacles.yaml entries as arrays, in declaration order (cached per list)."""
    key = id(static_obstacles)
    hit = _COMPILED_STATIC.get(key)
    if hit is not None and hit[0] is static_obstacles:
        return hit[1]
    boxes, spheres, planes = [], [], []
    for k, obs in enumerate(static_obstacles or []):
        if obs.get("is_workpiece", False):
            continue  # the tool may touch the workpiece
        otype = obs.get("type", "")
        m = float(obs.get("collision_margin", 0.05))
        oid = str(obs.get("id", "obstacle"))
        if otype == "box":
            boxes.append((k, oid, obs["center"], obs["half_extents"], m,
                          bool(obs.get("tool_contact", False))))
        elif otype == "sphere":
            spheres.append((k, oid, obs["center"], float(obs["radius"]), m))
        elif otype == "plane":
            n = np.asarray(obs.get("normal", [0.0, 0.0, 1.0]), dtype=float)
            nn = float(np.linalg.norm(n)) or 1.0
            planes.append((k, oid, (n / nn).tolist(), float(obs.get("offset", 0.0)) / nn, m))
    if not planes:
        f = DEFAULT_FLOOR
        planes.append((-1, f["id"], f["normal"], f["offset"], f["collision_margin"]))

    comp = {
        "box_k": [r[0] for r in boxes], "box_id": [r[1] for r in boxes],
        "box_c": np.asarray([r[2] for r in boxes], float).reshape(-1, 3),
        "box_h": np.asarray([r[3] for r in boxes], float).reshape(-1, 3),
        "box_m": np.asarray([r[4] for r in boxes], float),
        # ``tool_contact`` boxes: the surface the tool works on — the tool's bodies
        # may touch it, the arm's may not.
        "box_tool_ok": np.asarray([r[5] for r in boxes], bool),
        "sph_k": [r[0] for r in spheres], "sph_id": [r[1] for r in spheres],
        "sph_c": np.asarray([r[2] for r in spheres], float).reshape(-1, 3),
        "sph_r": np.asarray([r[3] for r in spheres], float),
        "sph_m": np.asarray([r[4] for r in spheres], float),
        "pl_id": [r[1] for r in planes],
        "pl_n": np.asarray([r[2] for r in planes], float).reshape(-1, 3),
        "pl_d": np.asarray([r[3] for r in planes], float),
        "pl_m": np.asarray([r[4] for r in planes], float),
    }
    _COMPILED_STATIC[key] = (static_obstacles, comp)
    if len(_COMPILED_STATIC) > 64:
        _COMPILED_STATIC.pop(next(iter(_COMPILED_STATIC)))
    return comp


class EnvelopeSphere(int):
    """Label of a link-envelope sphere in :func:`_static_bodies` (its index in the
    envelope) — neither an arm frame index nor a ``("tool", j)`` tool sphere."""


def _static_bodies(model, col_data, base, yaw, spheres, sphere_radius, sphere_radii,
                   tool_cfg):
    """The spheres checked against static obstacles: ``(P, R, on_floor, label)``.

    * arm spheres from index 2 (the rest of the base block was never checked
      against fixtures), at their per-body radius;
    * ``on_floor``: the planes apply to every sphere that MOVES — frames rigidly
      attached to the world (parentJoint 0: universe, root, base) are structure,
      whatever their index in this URDF;
    * the tool's ``static_spheres`` (fitted to the tool meshes at
      ``static_radius_m`` by the converter), placed on the mount frame.
    ``label[i]`` is the arm sphere index, or ``("tool", j)``.
    """
    n_r = 0 if sphere_radii is None else len(sphere_radii)
    nframes = len(model.frames) if getattr(model, "frames", None) is not None else 0
    tool_cfg = _tool_cfg_for(model, tool_cfg)
    # Frames past the tool's mount link are the tool (as for their radius); the
    # link-fill spheres past the frame block are always arm steel.
    tool_from = _tool_frame_start(model, tool_cfg) if nframes else nframes
    pts, rad, floor, label, own_margin, is_tool = [], [], [], [], [], []
    for idx in range(2, len(spheres)):
        pts.append(spheres[idx])
        rad.append(float(sphere_radii[idx]) if idx < n_r else float(sphere_radius))
        fixed = idx < nframes and arm_model.frame_parent_joint(model.frames[idx]) == 0
        floor.append(not fixed)
        label.append(idx)
        own_margin.append(np.nan)                 # the arm: each obstacle's margin
        is_tool.append(tool_from <= idx < nframes)
    # The arm's fitted envelope (link_collision): every MOVING sphere of it, at the
    # arm radius. Spheres on world-fixed links (the base) are structure: a fixture
    # next to the base would otherwise reject every configuration.
    E, fixed_e = _envelope_world(model, col_data, base, yaw)
    if E is not None:
        er = float(_robot_spec(model)["env"]["radius"])
        for j in np.nonzero(~fixed_e)[0]:
            pts.append(E[j])
            rad.append(er)
            floor.append(True)
            label.append(EnvelopeSphere(j))
            own_margin.append(np.nan)
            is_tool.append(False)
    static = (tool_cfg or {}).get("static_spheres") or []
    if static:
        tr = float(tool_cfg.get("static_radius_m") or
                   (sphere_radii[n_r - 1] if n_r else sphere_radius))
        # A view kept ON the config: _tool_offsets caches by id(), so a fresh dict
        # per call could pick up another dict's cached centres. The cell's tool
        # margin is resolved once, here, not on every check (it is a file read).
        view = tool_cfg.get("_static_view")
        if view is None:
            view = tool_cfg["_static_view"] = {
                "mount_link": tool_cfg.get("mount_link", ""), "spheres": static,
                "_margin": _tool_static_margin(model)}
        tm = view.get("_margin")
        for j, c in enumerate(_tool_spheres(model, col_data, base, yaw, view)):
            pts.append(c)
            rad.append(tr)
            floor.append(True)
            label.append(("tool", j))
            own_margin.append(np.nan if tm is None else tm)
            is_tool.append(True)
    return (np.asarray(pts, float).reshape(-1, 3), np.asarray(rad, float),
            np.asarray(floor, bool), _Labels(label, own_margin, is_tool))


class _Labels(list):
    """Sphere labels, carrying each sphere's own margin (NaN = the obstacle's) and
    whether the sphere is the TOOL's (default: the ``("tool", j)`` spheres)."""

    def __init__(self, labels, margins, tool=None):
        super().__init__(labels)
        self.margins = np.asarray(margins, float)
        self.tool = (np.asarray(tool, bool) if tool is not None else
                     np.asarray([isinstance(lab, tuple) for lab in labels], bool))


def _tool_static_margin(model) -> Optional[float]:
    """The cell's ``tool_collision_margin`` (obstacles.yaml), or None."""
    try:
        owner = arm_model.owner_of(model)
        return None if owner is None else _load_clearance_policy(owner[0]).get("tool_margin")
    except Exception:  # noqa: BLE001
        return None


def _tool_cfg_for(model, tool_cfg):
    """The caller's tool config, else the one its model's robot declares."""
    if tool_cfg is not None:
        return tool_cfg
    # Read once per model (with the envelope): a robots.yaml parse per validity
    # check is most of the check's time.
    return _robot_spec(model)["tool"]


_BOX_CHUNK = 32


def _first_box_hit(P, R, own, comp, extra_margin, _m, tool=None):
    """``(box m, sphere i, closest point, clearance)`` of the first breached box in
    declaration order (then sphere order), or None.

    Broad phase first: consecutive spheres are neighbours (one link's envelope is
    contiguous), so each chunk of them gets a bounding box grown by the largest
    clearance any of them needs, and only the obstacle boxes that overlap it are
    measured. Same answer as measuring every sphere against every box."""
    from agents.trajectory.tools import fastgeom
    if fastgeom.AVAILABLE:
        # Compiled, stops at the first hit; the NumPy body below is the reference.
        return fastgeom.box_first_hit(P, R, own, comp, extra_margin, tool)
    bc, bh = comp["box_c"], comp["box_h"]
    # Pairs a ``tool_contact`` box does not count for: a tool sphere against it.
    # Fail-closed: a flag vector of the wrong shape exempts nothing.
    tool_ok = comp.get("box_tool_ok")
    exempt = (tool is not None and tool_ok is not None and len(tool_ok) == len(bc)
              and len(tool) == len(P) and bool(np.any(tool)) and bool(np.any(tool_ok)))
    own_max = float(np.nanmax(own)) if np.any(~np.isnan(own)) else 0.0
    reach = max(float(comp["box_m"].max()), own_max) + float(extra_margin)
    found = []                                                # (m, i, closest, clearance)
    for a in range(0, len(P), _BOX_CHUNK):
        Pc = P[a:a + _BOX_CHUNK]
        grow = float(np.max(R[a:a + _BOX_CHUNK])) + reach
        lo, hi = Pc.min(0) - grow, Pc.max(0) + grow
        near = np.nonzero(np.all((bc + bh >= lo) & (bc - bh <= hi), axis=1))[0]
        if not len(near):
            continue
        diff = Pc[:, None, :] - bc[near][None, :, :]
        cl = np.clip(diff, -bh[near][None], bh[near][None])
        d = np.linalg.norm(diff - cl, axis=2)
        # Margins for THIS chunk and the NEAR boxes only (was the full spheres x
        # all-boxes matrix per chunk, then sliced): the same numbers, as _m gives.
        own_c = own[a:a + _BOX_CHUNK]
        need = (R[a:a + _BOX_CHUNK][:, None]
                + np.where(np.isnan(own_c)[:, None], comp["box_m"][near][None, :],
                           own_c[:, None])
                + extra_margin)
        hit = d < need
        if exempt:
            hit &= ~(tool[a:a + _BOX_CHUNK][:, None] & tool_ok[near][None, :])
        if hit.any():
            c = int(np.nonzero(hit.any(axis=0))[0][0])
            r = int(np.nonzero(hit[:, c])[0][0])
            found.append((int(near[c]), a + r, bc[near[c]] + cl[r, c], d[r, c] - need[r, c]))
    return min(found, key=lambda f: (f[0], f[1])) if found else None


def _static_first_breach(P, R, on_floor, label, comp, extra_margin: float,
                         extra_per_sphere=None) -> Optional[dict]:
    """The first breach, in the old order: planes first, then boxes and spheres in
    declaration order, then sphere order. ``None`` when every body is clear.

    ``extra_per_sphere`` (N,) adds each sphere's own allowance — its stopping
    distance at the speed it moves at — on top of ``extra_margin``."""
    if extra_per_sphere is not None and len(extra_per_sphere) == len(P):
        R = np.asarray(R, float) + np.asarray(extra_per_sphere, float)
    own = getattr(label, "margins", None)
    own = np.full(len(P), np.nan) if own is None or len(own) != len(P) else own

    def _m(obstacle_margins):
        """(N, K) margins: each sphere's own where it has one, else the obstacle's."""
        return np.where(np.isnan(own)[:, None], obstacle_margins[None, :], own[:, None])

    def _lp(i):
        lab = label[i]
        if isinstance(lab, EnvelopeSphere):
            return {"link_index": None, "envelope_sphere": int(lab)}
        if not isinstance(lab, tuple):
            return {"link_index": lab}
        return {"link_index": None, "tool_sphere": lab[1]}

    if len(P) and len(comp["pl_d"]) and on_floor.any():
        idx = np.nonzero(on_floor)[0]
        sd = P[idx] @ comp["pl_n"].T + comp["pl_d"][None, :]            # (n, planes)
        need = R[idx][:, None] + _m(comp["pl_m"])[idx] + extra_margin
        hit = sd < need
        if hit.any():
            j = int(np.nonzero(hit.any(axis=0))[0][0])
            i = int(idx[np.nonzero(hit[:, j])[0][0]])
            n, s_ = comp["pl_n"][j], float(sd[np.nonzero(idx == i)[0][0], j])
            oid = comp["pl_id"][j]
            return {"type": "floor" if oid.startswith("floor") else "plane",
                    "object_id": oid, **_lp(i),
                    "point": [round(float(v), 4) for v in P[i] - s_ * n],
                    "link_point": [round(float(v), 4) for v in P[i]],
                    "clearance_m": round(s_ - float(need[np.nonzero(idx == i)[0][0], j]), 4)}

    best = None                                     # (declaration k, sphere i, info)
    if len(P) and len(comp["box_k"]):
        tool = getattr(label, "tool", None)
        hit_mi = _first_box_hit(P, R, own, comp, extra_margin, _m,
                                None if tool is None or len(tool) != len(P) else tool)
        if hit_mi is not None:
            m, i, closest, clearance = hit_mi
            best = (comp["box_k"][m], i, {
                "type": "box", "object_id": comp["box_id"][m], **_lp(i),
                "point": [round(float(v), 4) for v in closest],
                "link_point": [round(float(v), 4) for v in P[i]],
                "clearance_m": round(float(clearance), 4)})
    if len(P) and len(comp["sph_k"]):
        d = np.linalg.norm(P[:, None, :] - comp["sph_c"][None, :, :], axis=2)
        need = R[:, None] + comp["sph_r"][None, :] + _m(comp["sph_m"]) + extra_margin
        hit = d < need
        if hit.any():
            m = int(np.nonzero(hit.any(axis=0))[0][0])
            if best is None or comp["sph_k"][m] < best[0]:
                i = int(np.nonzero(hit[:, m])[0][0])
                best = (comp["sph_k"][m], i, {
                    "type": "sphere", "object_id": comp["sph_id"][m], **_lp(i),
                    "point": [round(float(v), 4) for v in comp["sph_c"][m]],
                    "link_point": [round(float(v), 4) for v in P[i]],
                    "clearance_m": round(float(d[i, m] - need[i, m]), 4)})
    return None if best is None else best[2]


def floor_clear(model, col_data, q, base, yaw, static_obstacles: list,
                sphere_radius: float = 0.12, sphere_radii=None, tool_cfg=None,
                extra_margin: float = 0.0) -> bool:
    """Every moving sphere (arm and tool) clear of the cell's planes — the floor.

    The IK branch ranking's hard filter; the same rule ``_config_is_free`` applies.
    """
    import pinocchio as pin
    pin.forwardKinematics(model, col_data, np.asarray(q, float))
    pin.updateFramePlacements(model, col_data)
    spheres = _sphere_positions(model, col_data, base, yaw)
    P, R, on_floor, label = _static_bodies(model, col_data, base, yaw, spheres,
                                           sphere_radius, sphere_radii, tool_cfg)
    comp = _compile_static(static_obstacles)
    return _static_first_breach(P, R, on_floor, label,
                                {**comp, "box_k": [], "sph_k": []}, extra_margin) is None


# ── moving obstacles (other arms' spheres), vectorised ─────────────────────────
# The check used to be a Python double loop — every obstacle sphere against every
# arm sphere, one np.linalg.norm call each. Measured on welding_cell_2_small (4 arms, 150
# parked-arm spheres) it cost 8.3 ms per state, ~95 % of the compound search's
# validity check, while forward kinematics costs 0.4 µs. One distance matrix gives
# the same answer (tests/trajectory/test_vectorised_collision.py holds them
# equal). The obstacle LIST is fixed for a whole search, so its array is built
# once per list, not once per state.
_OBSTACLE_ARRAYS: list = []      # [(list object, len, O (m,3), R (m,))], newest last


def _obstacle_arrays(obstacle_spheres):
    for ref, n, O, R in reversed(_OBSTACLE_ARRAYS):
        if ref is obstacle_spheres and n == len(obstacle_spheres):
            return O, R
    O = np.asarray([np.asarray(p, float).reshape(3) for p, _ in obstacle_spheres], float)
    R = np.asarray([float(r) for _, r in obstacle_spheres], float)
    # The list itself is held, so its id cannot be reused by another list while
    # the entry lives; a mutated list changes length or is rebuilt by its owner.
    _OBSTACLE_ARRAYS.append((obstacle_spheres, len(obstacle_spheres), O, R))
    del _OBSTACLE_ARRAYS[:-8]
    return O, R


def _moving_hit(points, obstacle_spheres, sphere_radius: float):
    """The FIRST ``(obstacle index, point index, gap)`` with ``|p - o| < r_o +
    sphere_radius`` — in the old loop's order (obstacle-major) — or None."""
    if not len(points) or not len(obstacle_spheres):
        return None
    O, R = _obstacle_arrays(obstacle_spheres)
    P = np.asarray(points, float).reshape(-1, 3)
    from agents.trajectory.tools import fastgeom
    if fastgeom.AVAILABLE:
        # Compiled, stops at the first hit; the NumPy body below is the reference.
        return fastgeom.moving_first_hit(P, O, R + float(sphere_radius))
    # Squared distances against squared thresholds: every threshold is positive
    # (a radius plus sphere_radius), so d < t  <=>  d^2 < t^2 — no sqrt per pair.
    diff = O[:, None, :] - P[None, :, :]                       # (m, n, 3)
    d2 = np.einsum("mnk,mnk->mn", diff, diff)
    thr = R[:, None] + float(sphere_radius)
    hit = d2 < thr * thr
    if not hit.any():
        return None
    k, i = np.unravel_index(int(np.argmax(hit)), hit.shape)
    return int(k), int(i), float(np.sqrt(d2[k, i]) - thr[k, 0])


def _config_is_free(
    model, col_data, q: np.ndarray,
    base: np.ndarray, yaw: float,
    obstacle_spheres: list[tuple[np.ndarray, float]],
    static_obstacles: list[dict],
    sphere_radius: float = 0.12,
    tool_cfg: "Optional[dict]" = None,
    sphere_radii: "Optional[np.ndarray]" = None,
    extra_margin: float = 0.0,
) -> bool:
    """Sphere-model collision predicate shared by the RRT and OMPL backends.

    Returns ``True`` when configuration ``q`` is collision-free: above the floor,
    clear of the moving robot-sphere obstacles, and clear of the static
    box/sphere obstacles from ``obstacles.yaml`` (``is_workpiece`` boxes are
    exempt so the tool tip may reach the weld surface).

    This is the SINGLE source of truth for validity. ``_rrt_single`` calls it per
    sample, and OMPL's ``StateValidityChecker`` wraps it unchanged, so both
    backends accept/reject exactly the same configurations.

    ``col_data`` is a reusable ``model.createData()`` buffer supplied by the
    caller — allocating one per call is the dominant cost when this runs
    thousands of times during planning.

    ``sphere_radii`` (from :func:`_cell_sphere_radii`) gives the arm and the tool
    their own radii for the STATIC check; ``sphere_radius`` is the fallback for any
    sphere it does not cover and stays the only radius used against the other arm
    (see below). ``extra_margin`` is the speed allowance from
    :func:`_speed_margin`, added to the static clearance — the phase-dependent part
    of the gate, expressed where it belongs.
    """
    import pinocchio as pin

    pin.forwardKinematics(model, col_data, q)
    pin.updateFramePlacements(model, col_data)
    # Arm-link frame spheres are validated against the floor + static fixture.
    # The tool + plate spheres are checked ONLY against the other arm (moving
    # obstacles): a parametric tool model is unreliable against the close fixture
    # walls (false positives on tilted far-reach configs) but IS the right
    # envelope for arm-vs-arm clearance — and including them here makes the
    # minimum-retract search lift the idle arm far enough to clear the plate.
    spheres = _sphere_positions(model, col_data, base, yaw)
    tool = _tool_spheres(model, col_data, base, yaw, tool_cfg)
    # Per-sphere radii cover the frame-origin block only; every index past the end
    # is a link-fill sphere on the arm and takes the uniform arm radius.
    n_r = 0 if sphere_radii is None else len(sphere_radii)

    # Moving robot obstacles (other robots' current sphere positions) — checked
    # against arm links AND the tool/plate.
    #
    # Deliberately UNIFORM: a fixture's pose is exactly known, but two arms each
    # carry tracking error and reporting latency, so the pair envelope is not a
    # pure geometry question and must not shrink because one of the bodies is thin.
    # Robot-vs-robot separation is the one clearance this change does not touch.
    if obstacle_spheres and _moving_hit(spheres + tool, obstacle_spheres,
                                        sphere_radius) is not None:
        return False

    # Static obstacles from obstacles.yaml — boxes, spheres and planes (the floor),
    # against the arm and the tool's static spheres (_static_first_breach).
    P, R, on_floor, label = _static_bodies(model, col_data, base, yaw, spheres,
                                           sphere_radius, sphere_radii, tool_cfg)
    if _static_first_breach(P, R, on_floor, label, _compile_static(static_obstacles),
                            extra_margin) is not None:
        return False

    # The arm against ITSELF — last, because it is the costliest check and the
    # cheap ones above reject most states. See tools/self_collision.py.
    return _self_free(model, q)


def _self_free(model, q) -> bool:
    """No self-collision (or no self check for this model's robot)."""
    from agents.trajectory.tools.self_collision import checker_for_model
    c = checker_for_model(model)
    return True if c is None else c.is_free(q)


def _self_violation(model, q) -> Optional[dict]:
    from agents.trajectory.tools.self_collision import checker_for_model
    c = checker_for_model(model)
    return None if c is None else c.violation(q)


# ─── Collision diagnosis (one-shot, for actionable failures) ──────────────────

def _config_collision_info(
    model, col_data, q: np.ndarray,
    base: np.ndarray, yaw: float,
    obstacle_spheres: list,
    static_obstacles: list,
    sphere_radius: float = 0.12,
    sphere_radii: "Optional[np.ndarray]" = None,
    extra_margin: float = 0.0,
    sphere_extra=None,
) -> Optional[dict]:
    """Diagnose the FIRST collision of configuration ``q`` (or ``None`` if free).

    Companion to the lean boolean ``_config_is_free``. This variant is heavier
    (it builds a result dict), so it is meant to be called ONCE on a specific
    configuration (a plan's start/goal) to explain WHY it is invalid — NOT in the
    per-sample planning hot loop. It names the object involved and the world-frame
    collision point so the LLM can choose a recovery strategy:

        {"type": "floor"|"robot"|"box"|"sphere",
         "object_id": str,               # obstacle id from obstacles.yaml
         "link_index": int,              # index into the sphere list: a frame
                                        # index when < model.nframes, otherwise a
                                        # link-fill sphere (see _sphere_positions)
         "point": [x, y, z],             # world point ON THE OBJECT nearest the arm
         "link_point": [x, y, z],        # world position of the offending arm link
         "clearance_m": float}           # signed; negative = penetration depth

    ``sphere_radii`` / ``extra_margin`` mean exactly what they mean in
    :func:`_config_is_free` — the two functions must apply the same gate or a
    configuration would be rejected without an explanation, or explained without
    being rejected.
    """
    import pinocchio as pin

    pin.forwardKinematics(model, col_data, q)
    pin.updateFramePlacements(model, col_data)
    spheres = _sphere_positions(model, col_data, base, yaw)
    n_r = 0 if sphere_radii is None else len(sphere_radii)

    P, R, on_floor, label = _static_bodies(model, col_data, base, yaw, spheres,
                                           sphere_radius, sphere_radii, None)
    comp = _compile_static(static_obstacles)
    # The floor (and any declared plane) first, as before.
    hit = _static_first_breach(P, R, on_floor, label,
                               {**comp, "box_k": [], "sph_k": []}, extra_margin,
                               sphere_extra)
    if hit is not None:
        return hit

    # Moving robot obstacles (other robots' current sphere positions)
    hit = _moving_hit(spheres, obstacle_spheres, sphere_radius) if obstacle_spheres else None
    if hit is not None:
        k, idx, gap = hit
        obs_pos, s_pos = obstacle_spheres[k][0], spheres[idx]
        return {
            "type": "robot", "object_id": "other_robot", "link_index": idx,
            "point": [round(float(v), 4) for v in obs_pos],
            "link_point": [round(float(v), 4) for v in s_pos],
            "clearance_m": round(gap, 4),
        }

    # Static obstacles from obstacles.yaml (boxes, spheres), arm and tool.
    hit = _static_first_breach(P, R, np.zeros(len(P), bool), label, comp, extra_margin,
                               sphere_extra)
    if hit is not None:
        return hit
    return _self_violation(model, q)


def _static_points(model, col_data, q, base, yaw, sphere_radius, sphere_radii):
    """The spheres the static check uses at ``q`` (in its order), as an array."""
    import pinocchio as pin
    pin.forwardKinematics(model, col_data, np.asarray(q, float))
    pin.updateFramePlacements(model, col_data)
    P, _R, _f, _l = _static_bodies(model, col_data, base, yaw,
                                   _sphere_positions(model, col_data, base, yaw),
                                   sphere_radius, sphere_radii, None)
    return P


def _sample_stopping(model, col_data, samples, base, yaw, sphere_radius, sphere_radii):
    """Per sample, each static-check sphere's stopping distance at its speed."""
    if len(samples) < 2:
        return None
    P = np.stack([_static_points(model, col_data,
                                 np.asarray(pt["positions"], float)[: model.nq],
                                 base, yaw, sphere_radius, sphere_radii) for pt in samples])
    t = np.asarray([float(pt.get("time_s", k)) for k, pt in enumerate(samples)])
    if np.any(np.diff(t) <= 0):
        return None
    speeds = np.linalg.norm(np.gradient(P, t, axis=0), axis=2)
    return stopping_distance(model, speeds)


def _measured_scale(samples: list, limits, phase: str) -> float:
    """The fraction of top joint speed a trajectory reaches in one phase — what the
    runtime monitor measures from consecutive states. 1.0 (fail-closed) when the
    robot declares no velocity limits or the timestamps do not advance."""
    lim = np.asarray(limits or [], float)
    scale = 0.0
    for prev, pt in zip(samples, samples[1:]):
        if pt.get("phase", "transit") != phase:
            continue
        dt = float(pt.get("time_s", 0.0)) - float(prev.get("time_s", 0.0))
        dq = np.abs(np.asarray(pt["positions"], float) - np.asarray(prev["positions"], float))
        if dt <= 0 or lim.size < dq.size or not np.all(lim[: dq.size] > 0):
            return 1.0
        scale = max(scale, float(np.max(dq / (lim[: dq.size] * dt))))
    return min(scale, 1.0)


def leg_static_breach(cell_id: str, robot_ns: str, samples: list) -> Optional[dict]:
    """First sample of one robot's trajectory that breaches the cell's static
    obstacles (floor, fixtures, the arm itself), or ``None`` — the planner's
    emitted-trajectory backstop and the runtime monitor's rule, for a trajectory
    that reached a gate any other way (cached, edited, imported, re-chained)."""
    if not samples:
        return None
    model, _ = _load_pinocchio_model(cell_id, robot_ns)
    rcfg = _load_robot_config(cell_id, robot_ns)
    policy = _load_clearance_policy(cell_id)
    full = policy["speed_margin_full_speed_m"]
    lim = rcfg.get("joint_velocity_limits")
    return _trajectory_collision_info(
        model, model.createData(), samples, np.array(rcfg["base_position"]),
        float(rcfg.get("base_orientation_yaw", 0.0)), [], load_static_obstacles(cell_id),
        sphere_radius=policy["normal"],
        sphere_radii=_cell_sphere_radii(cell_id, model, rcfg.get("tool_collision")),
        transit_margin=_speed_margin_from(full, _measured_scale(samples, lim, "transit")),
        approach_margin=_speed_margin_from(full, _measured_scale(samples, lim, "approach")),
    )


def _trajectory_collision_info(
    model, col_data, samples: list,
    base: np.ndarray, yaw: float,
    obstacle_spheres: list,
    static_obstacles: list,
    sphere_radius: float = 0.12,
    sphere_radii: "Optional[np.ndarray]" = None,
    transit_margin: float = 0.0,
    approach_margin: float = 0.0,
) -> Optional[dict]:
    """First EMITTED sample that breaches the gate, or ``None`` if all are clear.

    A planner does not verify a path, it SAMPLES one: the knots it searched, plus
    the straight-line motion between consecutive knots at some finite resolution.
    Whatever happens between two checked samples is assumed, and on a long arm that
    assumption is generous — the welding torch sits 0.9 m past the flange, so a
    fifth of a radian of joint motion swings the tip a long way. A path can dip
    into an obstacle and back out entirely inside one unchecked interval.

    Measured on this cell: every knot of ``robot2``/``W13`` was clear and OMPL
    certified the segment, yet sampling the same path at 401 points per span shows
    it 6.5 mm inside fixture voxel FIXTURE_A_v143. It was emitted, executed,
    and caught by the runtime collision monitor as an e-stop five seconds into the
    replay — the worst place to find out.

    So this checks the artefact that is actually sent to the controller, with the
    same predicate the runtime monitor applies. It is a BACKSTOP, not a substitute
    for searching at an honest resolution: it catches what the search missed at the
    density the trajectory is emitted at, which is not the same as proving the
    continuous motion clear.

    Each sample is judged under its own phase's speed allowance (``"phase"`` is
    stamped by ``ompl_planner._parameterize_two_phase``), because the whole point of
    the phase split is that a slow working stroke reserves less stopping room than a
    fast transit — the geometry is identical at both ends.

    Returns ``_config_collision_info``'s dict with ``"sample_index"`` and
    ``"phase"`` added, so the caller can say WHERE along the trajectory it broke.
    """
    # Each sphere's stopping distance at the speed it moves at on this trajectory,
    # when the robot declares its stopping data (robots.yaml ``stopping``).
    extras = _sample_stopping(model, col_data, samples, base, yaw, sphere_radius,
                              sphere_radii) if accounts_for_speed(model) else None
    for i, pt in enumerate(samples):
        q = np.asarray(pt["positions"], dtype=float)[: model.nq]
        phase = pt.get("phase", "transit")
        info = _config_collision_info(
            model, col_data, q, base, yaw, obstacle_spheres, static_obstacles,
            sphere_radius=sphere_radius, sphere_radii=sphere_radii,
            extra_margin=approach_margin if phase == "approach" else transit_margin,
            sphere_extra=None if extras is None else extras[i],
        )
        if info is not None:
            info["sample_index"] = i
            info["phase"] = phase
            return info
    # The arm against itself BETWEEN the samples too: the executed motion is the
    # straight joint-space segment between them, and two clear samples can have a
    # self-contact in between (umarell issue 14).
    return _self_violation_along(model, samples)


def _self_violation_along(model, samples: list) -> Optional[dict]:
    from agents.trajectory.tools.self_collision import checker_for_model
    c = checker_for_model(model)
    return None if c is None else c.trajectory_violation(samples)


def _collision_failure(which: str, info: dict) -> dict:
    """Build a structured, LLM-actionable failure from a collision-info dict.

    ``which`` is "goal" or "start". Returns
    ``{"reason_code", "reason", "collision"}`` where ``reason`` is a human string
    that names the object + world collision point and suggests a recovery, and
    ``collision`` is the raw info dict for programmatic use by the agent.
    """
    if info.get("type") == "self":
        # The arm against ITSELF: no obstacle to route around, so the fixture
        # advice below would mislead (a cell may have no fixtures at all).
        hint = ("The arm folds into itself at this pose. Rotate the tool about its own "
                "axis (the spin about the electrode is free), choose a different "
                "approach, or reallocate the operation to another arm."
                if which == "goal" else
                "The arm is already in self-collision: retract it to HOME first.")
        return {"reason_code": f"{which}_self_collision",
                "reason": (f"{which} configuration is in self-collision: "
                           f"{info.get('object_id')} (signed distance "
                           f"{info.get('clearance_m')} m). {hint}"),
                "collision": info}
    obj = info.get("object_id", "?")
    otype = info.get("type", "obstacle")
    pt = info.get("point")
    clr = info.get("clearance_m")
    link = info.get("link_index")
    if which == "goal":
        hint = (
            "Apply an approach strategy: stage the arm above the object and descend "
            "vertically onto the target, insert an intermediate waypoint that routes "
            "around it, or reorient/retract the arm."
        )
    else:
        hint = "Retract the arm to a safe pose (e.g. HOME) before planning to this target."
    reason = (
        f"{which} configuration is in collision: arm link {link} vs {otype} "
        f"'{obj}' at world point {pt} (signed clearance {clr} m). {hint}"
    )
    return {"reason_code": f"{which}_in_collision", "reason": reason, "collision": info}


def _target_obstacle_suggestion(
    target, approach_vector, static_obstacles, clearance: float,
) -> "Optional[dict]":
    """Diagnose whether the WELD TARGET itself is inside / too close to a box, and
    suggest moving it out along the approach axis so the robot can reach it.

    A ``goal_in_collision`` may be caused by an arm LINK grazing the fixture, but
    frequently the real cause is that the target POINT sits inside (or within the
    required clearance of) a fixture box — the tool simply cannot get there along
    the specified approach. This turns that into an actionable output: it names
    the box, says whether the point is INSIDE or merely TOO CLOSE, and computes a
    moved-out target (``target - approach_vector * offset``, i.e. backed off along
    the entry axis) that clears the box by ``clearance``. Returns ``None`` when the
    target itself is clear (the collision is then purely an arm-link issue).
    """
    t = np.asarray(target, dtype=float)
    av = np.asarray(approach_vector, dtype=float)
    n = float(np.linalg.norm(av))
    u = (-av / n) if n > 1e-9 else np.array([0.0, 0.0, 1.0])  # move OUT along entry

    boxes = [o for o in static_obstacles
             if o.get("type") == "box"
             and not (o.get("is_workpiece") or o.get("tool_contact"))]

    def nearest_box(p):
        worst, wid = 1e9, None
        for o in boxes:
            c = np.asarray(o["center"], float); h = np.asarray(o["half_extents"], float)
            d = np.abs(p - c) - h
            dd = float(np.max(d)) if np.all(d < 0) else float(np.linalg.norm(np.maximum(d, 0.0)))
            if dd < worst:
                worst, wid = dd, o.get("id", "?")
        return worst, wid

    d0, box0 = nearest_box(t)
    if box0 is None or d0 >= clearance:
        return None  # target is fine on its own → collision is an arm-link issue

    # March the target OUT along the entry axis until it clears the box by `clearance`.
    off, new = 0.0, t.copy()
    for _ in range(80):  # up to 0.80 m
        off += 0.01
        cand = t + u * off
        dc, _wid = nearest_box(cand)
        if dc >= clearance:
            new = cand
            break
    else:
        new = t + u * off

    inside = d0 < 0
    new_xyz = [round(float(v), 4) for v in new]
    where = "INSIDE" if inside else f"within {round(d0, 3)} m of"
    return {
        "target_status":      "inside_box" if inside else "too_close",
        "object_id":          box0,
        "target_clearance_m": round(d0, 3),
        "required_clearance_m": round(clearance, 3),
        "suggested_offset_m": round(off, 3),
        "suggested_position": new_xyz,
        "suggestion": (
            f"Weld target {target} is {where} box '{box0}' (needs >= {clearance:.2f} m "
            f"of clearance to be reachable). Move the weld out ~{off:.2f} m along the "
            f"approach axis to {new_xyz} so the robot can reach it in the specified "
            f"approach, OR verify box '{box0}' geometry (possible false/oversized "
            f"obstacle)."
        ),
    }


def _endpoint_collision_failure(
    model, start_q, goal_q, jl, ju, base, yaw,
    obstacle_spheres, static_obstacles,
    goal_radius: float = 0.12, start_radius: float = 0.12,
    sphere_radii: "Optional[np.ndarray]" = None,
    goal_margin: float = 0.0, start_margin: float = 0.0,
) -> Optional[dict]:
    """Return a structured collision failure if the goal or start config is invalid.

    Checks the GOAL first (the usual culprit for a weld enclosed by the fixture),
    then the START. Shared by the RRT and OMPL backends so a collision failure
    carries the object involved + world collision point for the LLM. Returns
    ``None`` when both endpoints are collision-free (a genuine no-path case).

    ``goal_radius``/``start_radius`` are fallbacks for spheres ``sphere_radii``
    does not cover. What legitimately differs between the two endpoints is the
    SPEED allowance (``goal_margin`` / ``start_margin``): the goal of a working
    stroke is reached slowly, a transit pose is not. The geometry is the same arm
    at both ends.
    """
    nq = model.nq
    col_data = model.createData()
    start_c = np.clip(np.asarray(start_q, dtype=float)[:nq], jl[:nq], ju[:nq])
    goal_c = np.clip(np.asarray(goal_q, dtype=float)[:nq], jl[:nq], ju[:nq])
    ginfo = _config_collision_info(
        model, col_data, goal_c, base, yaw, obstacle_spheres, static_obstacles,
        sphere_radius=goal_radius, sphere_radii=sphere_radii,
        extra_margin=goal_margin,
    )
    if ginfo is not None:
        return _collision_failure("goal", ginfo)
    sinfo = _config_collision_info(
        model, col_data, start_c, base, yaw, obstacle_spheres, static_obstacles,
        sphere_radius=start_radius, sphere_radii=sphere_radii,
        extra_margin=start_margin,
    )
    if sinfo is not None:
        return _collision_failure("start", sinfo)
    return None


