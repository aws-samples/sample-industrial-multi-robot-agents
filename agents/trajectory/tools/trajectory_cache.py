"""
Planner-independent trajectory cache.

``run_trajectory_planner`` (whichever backend the facade dispatches to)
stores each computed joint trajectory here, keyed by
``(cell_id, robot_ns, operation_id)``. ``assemble_behaviour_tree`` then pulls the
full arrays back by ``operation_id`` so trajectories never have to round-trip
through the LLM context.

The cache is deliberately backend-agnostic: it knows nothing about how a
trajectory was produced, only how to store and retrieve one. Both planners feed
it through the single ``cache_planner_result`` choke point in
``run_trajectory_planner``.

Import-safe and thread-safe: standard library only (no ROS, no LangChain, no
numpy), so it can be imported and unit-tested on any host.
"""
from __future__ import annotations

import threading
from typing import Optional

# (cell_id, robot_ns, operation_id) -> trajectory (list[{"positions", "time_s"}])
_CACHE: dict[tuple[str, str, str], list[dict]] = {}
_LOCK = threading.Lock()

# (cell_id, robot_ns) -> the goal joint config of that robot's most recently
# planned operation. The planners chain from this so consecutive welds move
# directly weld-to-weld instead of returning to HOME between every operation.
# Reset at the start of each planning session (see reset_robot_configs).
_LAST_CONFIG: "dict[tuple[str, str], list[float]]" = {}

# (cell_id, robot_ns) -> the WORKING POSE that robot's most recent operation
# ended in: {"target": [x,y,z], "approach_vector": [ax,ay,az], "orientation": {..}}.
# A weld's trajectory ENDS on the seam (that is what validate_approach checks and
# where the dwell happens), so the arm is still in contact when the next motion
# starts. The next motion therefore has to LEAVE along the same axis it came in
# on — retracting to the standoff at the weld's own orientation before any
# transit. Without this the next transit drags the tool sideways out of the seam.
# Reset per session alongside _LAST_CONFIG.
_LAST_POSE: "dict[tuple[str, str], dict]" = {}
# (cell, robot, operation) -> the working pose that operation ends in (every weld,
# not only the latest) — see set_weld_pose.
_WELD_POSES: "dict[tuple[str, str, str], dict]" = {}
# (cell, robot) -> why that robot's last planned return to HOME was refused
# (no on-axis lift-off, or a leg that failed a check) — see set_home_refusal.
_HOME_REFUSALS: "dict[tuple[str, str], dict]" = {}
# (cell, robot, operation) -> what that operation's working stroke must be — the
# specified orientation, target and approach vector of the request — so the
# assembler checks every weld leg against it (validate_approach), whichever
# backend planned it. See set_op_spec.
_OP_SPECS: "dict[tuple[str, str, str], dict]" = {}


def _key(cell_id, robot_ns, operation_id) -> Optional[tuple[str, str, str]]:
    """Build a cache key, or ``None`` if any component is missing.

    A missing ``operation_id`` (or cell/robot) means the trajectory simply is
    not cacheable — callers treat that as a silent no-op rather than an error.
    """
    if not cell_id or not robot_ns or not operation_id:
        return None
    return (str(cell_id), str(robot_ns), str(operation_id))


def cache_trajectory(cell_id, robot_ns, operation_id, trajectory) -> bool:
    """Store one trajectory. Returns ``True`` if it was cached.

    No-op (returns ``False``) when any key component or the trajectory itself is
    missing, so a plan issued without an ``operation_id`` just isn't cached
    instead of raising.
    """
    key = _key(cell_id, robot_ns, operation_id)
    if key is None or not trajectory:
        return False
    with _LOCK:
        _CACHE[key] = list(trajectory)
    return True


def get_cached_trajectory(cell_id, robot_ns, operation_id) -> Optional[list[dict]]:
    """Return a copy of the cached trajectory, or ``None`` on a cache miss."""
    key = _key(cell_id, robot_ns, operation_id)
    if key is None:
        return None
    with _LOCK:
        traj = _CACHE.get(key)
    return list(traj) if traj is not None else None


def restore_cached_trajectory(cell_id, robot_ns, operation_id, trajectory) -> None:
    """Put back what :func:`get_cached_trajectory` returned before a leg was cached
    (``None``: there was nothing — the entry is removed)."""
    key = _key(cell_id, robot_ns, operation_id)
    if key is None:
        return
    with _LOCK:
        if trajectory:
            _CACHE[key] = list(trajectory)
        else:
            _CACHE.pop(key, None)


def cache_planner_result(cell_id, motion_requests, result) -> int:
    """Cache every trajectory in a planner result, planner-agnostically.

    Every backend returns the same contract shape
    (``{"trajectories": {robot_ns: [...]}, ...}``). This maps each robot's
    trajectory to the ``operation_id`` of its originating motion request and
    caches it. Trajectories whose request carried no ``operation_id`` are
    skipped. Returns the number of trajectories cached.
    """
    if not isinstance(result, dict):
        return 0
    trajectories = result.get("trajectories") or {}
    if not trajectories:
        return 0
    op_by_ns = {
        r.get("robot_ns"): r.get("operation_id")
        for r in (motion_requests or [])
        if isinstance(r, dict) and r.get("robot_ns")
    }
    cached = 0
    for robot_ns, traj in trajectories.items():
        if cache_trajectory(cell_id, robot_ns, op_by_ns.get(robot_ns), traj):
            cached += 1
    return cached


def stroke_of(request: dict) -> Optional[float]:
    """How far up its axis a leg's arm may stand and still be in the weld's
    channel: the longer of the request's approach and retreat strokes."""
    vals = []
    for k in ("approach_offset_m", "retreat_offset_m"):
        try:
            v = float(request.get(k))
        except (TypeError, ValueError):
            continue
        if v > 0.0:
            vals.append(v)
    return max(vals) if vals else None


def set_last_config(cell_id, robot_ns, joints) -> None:
    """Record a robot's most recently planned goal config (for weld-to-weld chaining)."""
    if not cell_id or not robot_ns or joints is None:
        return
    try:
        vals = [float(v) for v in joints]
    except (TypeError, ValueError):
        return
    with _LOCK:
        _LAST_CONFIG[(str(cell_id), str(robot_ns))] = vals


def get_last_config(cell_id, robot_ns) -> Optional[list[float]]:
    """Return a robot's last planned goal config, or ``None`` if none recorded yet."""
    if not cell_id or not robot_ns:
        return None
    with _LOCK:
        cfg = _LAST_CONFIG.get((str(cell_id), str(robot_ns)))
        return list(cfg) if cfg is not None else None


def set_last_pose(cell_id, robot_ns, target, approach_vector, orientation) -> None:
    """Record the working pose a robot's most recent operation ended in.

    Stored so the NEXT operation can retract back out along ``approach_vector``
    at ``orientation`` before transiting — the arm leaves the seam the same way
    it entered. A missing target/approach vector means there is nothing to
    retract along, so the entry is cleared rather than half-written.
    """
    if not cell_id or not robot_ns:
        return
    if target is None or approach_vector is None:
        clear_last_pose(cell_id, robot_ns)
        return
    try:
        tgt = [float(v) for v in target][:3]
        av = [float(v) for v in approach_vector][:3]
    except (TypeError, ValueError):
        return
    if len(tgt) < 3 or len(av) < 3:
        return
    with _LOCK:
        _LAST_POSE[(str(cell_id), str(robot_ns))] = {
            "target": tgt,
            "approach_vector": av,
            "orientation": dict(orientation) if orientation else None,
        }


def get_last_pose(cell_id, robot_ns) -> Optional[dict]:
    """Return the pose a robot's last operation ended in, or ``None``."""
    if not cell_id or not robot_ns:
        return None
    with _LOCK:
        pose = _LAST_POSE.get((str(cell_id), str(robot_ns)))
        if pose is None:
            return None
        out = dict(pose)
        out["target"] = list(pose["target"])
        out["approach_vector"] = list(pose["approach_vector"])
        if pose.get("orientation"):
            out["orientation"] = dict(pose["orientation"])
        return out


def set_weld_pose(cell_id, robot_ns, operation_id, target, approach_vector,
                  orientation, stroke_m: Optional[float] = None) -> None:
    """Record the working pose operation ``operation_id`` ends in, for good.

    :func:`set_last_pose` keeps only a robot's MOST RECENT seam, so a leg planned
    out of order — a chain repair that pins ``current_joints``, a re-planned
    PARALLEL step, an edit of one weld — could not tell which seam its arm is on
    and left it without the on-axis lift-off (umarell issue 05/15). Every weld's
    pose is kept here and looked up by where the arm's TCP actually is.

    ``stroke_m`` is how far up the axis that weld's stroke reaches (its approach /
    retreat offset): the channel ``seam_at_start`` matches a start in. ``None``
    leaves the matcher's default.
    """
    if not cell_id or not robot_ns or not operation_id or target is None \
            or approach_vector is None:
        return
    try:
        tgt = [float(v) for v in target][:3]
        av = [float(v) for v in approach_vector][:3]
    except (TypeError, ValueError):
        return
    if len(tgt) < 3 or len(av) < 3:
        return
    with _LOCK:
        _WELD_POSES[(str(cell_id), str(robot_ns), str(operation_id))] = {
            "target": tgt, "approach_vector": av,
            "orientation": dict(orientation) if orientation else None,
            "stroke_m": float(stroke_m) if stroke_m else None,
        }


def weld_poses(cell_id, robot_ns) -> list:
    """``[(operation_id, pose), ...]`` of every weld recorded for one robot."""
    with _LOCK:
        return [(op, {**pose, "target": list(pose["target"]),
                      "approach_vector": list(pose["approach_vector"])})
                for (c, r, op), pose in _WELD_POSES.items()
                if c == str(cell_id) and r == str(robot_ns)]


def set_home_refusal(cell_id, robot_ns, why: Optional[dict], start=None) -> None:
    """Record (``why``) or clear (``None``) that a robot's planned return to HOME
    from ``start`` was refused — no on-axis lift-off out of its seam
    (``home_liftoff_blocked``), or a planned leg that failed a check and was not
    shipped (``home_return_refused``).

    Either way no planned leg exists, and a leg the planner did not produce is the
    executor's blind straight line, which nothing checks and which drags the tool
    out of a seam just the same. The assembler and the home reset read this
    (``planner_backend.home_refusal``) to refuse that motion, whichever backend
    planned the return. The start is kept so a record never speaks for another pose.
    """
    if not cell_id or not robot_ns:
        return
    key = (str(cell_id), str(robot_ns))
    with _LOCK:
        if why:
            q = None
            if start is not None:
                try:
                    q = [float(v) for v in start]
                except (TypeError, ValueError):
                    q = None
            _HOME_REFUSALS[key] = {**dict(why), "start": q}
        else:
            _HOME_REFUSALS.pop(key, None)


def home_refusal(cell_id, robot_ns, start=None, tol: float = 1e-3) -> Optional[dict]:
    """Why the last planned return of ``robot_ns`` was refused, or None. With
    ``start``, only a record made for that same start (within ``tol`` rad per
    joint) answers."""
    with _LOCK:
        why = _HOME_REFUSALS.get((str(cell_id), str(robot_ns)))
        why = dict(why) if why else None
    if why is None or start is None:
        return why
    q = why.get("start")
    try:
        cur = [float(v) for v in start]
    except (TypeError, ValueError):
        return None
    if q is None or len(q) != len(cur) or max(abs(a - b) for a, b in zip(q, cur)) > tol:
        return None
    return why


def set_op_spec(cell_id, robot_ns, operation_id, request: dict) -> None:
    """Record what operation ``operation_id``'s working stroke must be, from the
    request as SPECIFIED (before any spin the platform chose — the spin is free,
    the axis is not)."""
    key = _key(cell_id, robot_ns, operation_id)
    if key is None or not isinstance(request, dict):
        return
    t = request.get("target") or {}
    if isinstance(t, dict):
        t = [t.get("x"), t.get("y"), t.get("z")]
    spec = {"approach": bool(request.get("approach")),
            "constrain_orientation": bool(request.get("constrain_orientation")),
            "orientation": dict(request["orientation"]) if request.get("orientation") else None,
            "target": [float(v) for v in t] if t and None not in t else None,
            "approach_vector": list(request.get("approach_vector") or [0.0, 0.0, -1.0]),
            "approach_offset_m": request.get("approach_offset_m"),
            "operation_type": request.get("operation_type")}
    with _LOCK:
        _OP_SPECS[key] = spec


def op_spec(cell_id, robot_ns, operation_id) -> Optional[dict]:
    key = _key(cell_id, robot_ns, operation_id)
    with _LOCK:
        hit = _OP_SPECS.get(key) if key else None
        return dict(hit) if hit else None


def chain_state(cell_id, robot_ns, operation_id=None) -> dict:
    """What a planner commits for one arm when it ships a leg — the chained
    config, the last working pose and that operation's weld pose — so a leg the
    platform refuses AFTER the backend committed it can be taken back
    (:func:`restore_chain_state`)."""
    c, r = str(cell_id), str(robot_ns)
    with _LOCK:
        weld = _WELD_POSES.get((c, r, str(operation_id))) if operation_id else None
        return {"last_config": list(_LAST_CONFIG[(c, r)]) if (c, r) in _LAST_CONFIG else None,
                "last_pose": dict(_LAST_POSE[(c, r)]) if (c, r) in _LAST_POSE else None,
                "operation_id": operation_id,
                "weld_pose": dict(weld) if weld else None}


def restore_chain_state(cell_id, robot_ns, state: dict) -> None:
    """Put back a :func:`chain_state` — as if the refused leg was never planned."""
    c, r = str(cell_id), str(robot_ns)
    with _LOCK:
        for store, key, val in ((_LAST_CONFIG, (c, r), state.get("last_config")),
                                (_LAST_POSE, (c, r), state.get("last_pose"))):
            if val is None:
                store.pop(key, None)
            else:
                store[key] = val
        op = state.get("operation_id")
        if op:
            if state.get("weld_pose"):
                _WELD_POSES[(c, r, str(op))] = state["weld_pose"]
            else:
                _WELD_POSES.pop((c, r, str(op)), None)


def cell_chain_snapshot(cell_id) -> dict:
    """Every arm of ``cell_id`` that has chaining state, with that state:
    ``{robot_ns: {"last_config", "last_pose"}}``. An arm absent from it stands at
    HOME. What the next planned step can depend on, as one value (the leg store
    keys on it)."""
    c = str(cell_id)
    with _LOCK:
        arms = {k[1] for k in _LAST_CONFIG if k[0] == c} | {k[1] for k in _LAST_POSE if k[0] == c}
        return {ns: {"last_config": list(_LAST_CONFIG[(c, ns)]) if (c, ns) in _LAST_CONFIG else None,
                     "last_pose": dict(_LAST_POSE[(c, ns)]) if (c, ns) in _LAST_POSE else None}
                for ns in sorted(arms)}


def clear_last_pose(cell_id, robot_ns) -> None:
    """Forget a robot's last working pose (e.g. after it has retracted/homed)."""
    if not cell_id or not robot_ns:
        return
    with _LOCK:
        _LAST_POSE.pop((str(cell_id), str(robot_ns)), None)


def reset_robot_configs(cell_id: Optional[str] = None) -> int:
    """Forget chained per-robot configs so the next session starts from HOME.

    With no argument clears all; with ``cell_id`` clears only that cell's robots.
    Returns the number of entries removed. Also clears the matching working-pose
    entries, so a fresh session cannot retract along a stale previous weld's axis.
    """
    with _LOCK:
        if cell_id is None:
            n = len(_LAST_CONFIG)
            _LAST_CONFIG.clear()
            _LAST_POSE.clear()
            _WELD_POSES.clear()
            _HOME_REFUSALS.clear()
            _OP_SPECS.clear()
            return n
        doomed = [k for k in _LAST_CONFIG if k[0] == str(cell_id)]
        for k in doomed:
            del _LAST_CONFIG[k]
        for k in [k for k in _LAST_POSE if k[0] == str(cell_id)]:
            del _LAST_POSE[k]
        for k in [k for k in _WELD_POSES if k[0] == str(cell_id)]:
            del _WELD_POSES[k]
        for k in [k for k in _HOME_REFUSALS if k[0] == str(cell_id)]:
            del _HOME_REFUSALS[k]
        for k in [k for k in _OP_SPECS if k[0] == str(cell_id)]:
            del _OP_SPECS[k]
        return len(doomed)


def clear_trajectory_cache(cell_id: Optional[str] = None) -> int:
    """Drop cached trajectories and return how many entries were removed.

    With no argument, clears everything. With ``cell_id``, clears only that
    cell's entries (useful to reset state between planning runs).
    """
    with _LOCK:
        if cell_id is None:
            n = len(_CACHE)
            _CACHE.clear()
            return n
        doomed = [k for k in _CACHE if k[0] == str(cell_id)]
        for k in doomed:
            del _CACHE[k]
        return len(doomed)


# ─── Snapshot / restore ───────────────────────────────────────────────────────
# A planning run's *computed work* lives entirely in this module: the joint
# trajectories, and the per-robot chaining state (last config, last working pose)
# that makes consecutive welds continuous. Everything else a run holds is either
# cheap to rebuild or is in the message history.
#
# So a run that stops halfway — a paused operator, an expired token — can be
# resumed without re-solving anything, provided this state is written down and put
# back. That is what these two functions are for; see
# ``agents.trajectory.plan_checkpoint``.
#
# JSON-shaped on purpose (tuple keys flattened into records): the checkpoint is a
# file on the shared volume, read by a process that may be a different container.

def snapshot(cell_id: str) -> dict:
    """Everything this cell's in-flight planning run has computed so far."""
    cid = str(cell_id)
    with _LOCK:
        return {
            "trajectories": [
                {"robot_ns": robot, "operation_id": op, "trajectory": traj}
                for (c, robot, op), traj in _CACHE.items() if c == cid
            ],
            "last_config": [
                {"robot_ns": robot, "joints": joints}
                for (c, robot), joints in _LAST_CONFIG.items() if c == cid
            ],
            "last_pose": [
                {"robot_ns": robot, "pose": pose}
                for (c, robot), pose in _LAST_POSE.items() if c == cid
            ],
            "weld_poses": [
                {"robot_ns": robot, "operation_id": op, "pose": pose}
                for (c, robot, op), pose in _WELD_POSES.items() if c == cid
            ],
            "op_specs": [
                {"robot_ns": robot, "operation_id": op, "spec": spec}
                for (c, robot, op), spec in _OP_SPECS.items() if c == cid
            ],
        }


def restore(cell_id: str, data: dict) -> dict:
    """Put a :func:`snapshot` back. Returns what was restored, by kind.

    Additive, and deliberately so: it is called on a cache this cell's fresh-run
    reset has just emptied, and restoring is the whole point of resuming. A
    malformed record is skipped rather than aborting the restore — a resume that
    recovers 13 of 14 trajectories still saves 13 solves, and the planner will
    simply re-plan the one it cannot find.
    """
    cid = str(cell_id)
    counts = {"trajectories": 0, "last_config": 0, "last_pose": 0}
    if not isinstance(data, dict):
        return counts
    with _LOCK:
        for record in data.get("trajectories") or []:
            key = _key(cid, (record or {}).get("robot_ns"),
                       (record or {}).get("operation_id"))
            traj = (record or {}).get("trajectory")
            if key and isinstance(traj, list) and traj:
                _CACHE[key] = traj
                counts["trajectories"] += 1
        for record in data.get("last_config") or []:
            robot = (record or {}).get("robot_ns")
            joints = (record or {}).get("joints")
            if robot and isinstance(joints, list) and joints:
                _LAST_CONFIG[(cid, str(robot))] = list(joints)
                counts["last_config"] += 1
        for record in data.get("last_pose") or []:
            robot = (record or {}).get("robot_ns")
            pose = (record or {}).get("pose")
            if robot and isinstance(pose, dict) and pose:
                _LAST_POSE[(cid, str(robot))] = dict(pose)
                counts["last_pose"] += 1
        # Every weld's pose (set_weld_pose); absent from checkpoints older than
        # it, which then resume with the most recent seam per robot only.
        for record in data.get("weld_poses") or []:
            robot = (record or {}).get("robot_ns")
            op = (record or {}).get("operation_id")
            pose = (record or {}).get("pose")
            if robot and op and isinstance(pose, dict) and pose.get("target"):
                _WELD_POSES[(cid, str(robot), str(op))] = dict(pose)
        for record in data.get("op_specs") or []:
            robot = (record or {}).get("robot_ns")
            op = (record or {}).get("operation_id")
            spec = (record or {}).get("spec")
            if robot and op and isinstance(spec, dict):
                _OP_SPECS[(cid, str(robot), str(op))] = dict(spec)
    return counts


def new_tree_id(cell_id: str) -> str:
    """A fresh BehaviourTree id. The random part keeps two plans of one cell made
    in the same second (a plan and a plan edit, two submissions) from saving over
    each other."""
    import time
    import uuid
    return f"bt_{int(time.time())}_{uuid.uuid4().hex[:6]}_{cell_id}"


def weld_pose_record(cell_id, robots=None) -> dict:
    """``{robot_ns: {operation_id: pose}}`` of the welds recorded for ``cell_id``
    (``robots`` limits it), JSON-ready — what an assembled tree carries
    (``metadata.weld_poses``) so a process that did not plan it can still lift an
    arm off its seam on-axis (the cell's e-stop home reset)."""
    keep = None if robots is None else {str(r) for r in robots}
    out: dict = {}
    with _LOCK:
        for (c, r, op), pose in _WELD_POSES.items():
            if c == str(cell_id) and (keep is None or r in keep):
                out.setdefault(r, {})[op] = {
                    **pose, "target": list(pose["target"]),
                    "approach_vector": list(pose["approach_vector"]),
                    "orientation": dict(pose["orientation"]) if pose.get("orientation")
                    else None}
    return out


def load_weld_pose_record(cell_id, record) -> int:
    """Register a :func:`weld_pose_record` (e.g. a tree's ``metadata.weld_poses``)
    in this process. Malformed entries are skipped. Returns how many were kept."""
    n = 0
    for robot, ops in (record or {}).items() if isinstance(record, dict) else ():
        for op, pose in (ops or {}).items() if isinstance(ops, dict) else ():
            if not isinstance(pose, dict):
                continue
            set_weld_pose(cell_id, robot, op, pose.get("target"),
                          pose.get("approach_vector"), pose.get("orientation"),
                          stroke_m=pose.get("stroke_m"))
            n += (str(cell_id), str(robot), str(op)) in _WELD_POSES
    return n
