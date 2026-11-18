"""
Time parameterization for planned joint-space paths.

Turns a geometric joint path (a list of waypoint configurations produced by the
OMPL/RRT planners) into a timed trajectory ``[{"positions", "time_s"}]``.

Primary method: **TOPP-RA** (Time-Optimal Path Parameterization by Reachability
Analysis) — computes the fastest timing along the fixed path subject to per-joint
velocity and acceleration limits, so the resulting motion is dynamically
feasible (no instantaneous jumps, bounded joint speeds/accels). This is the
substrate the space-time coordinator relies on for consistent, physically
meaningful timing.

Fallback: if the ``toppra`` package is unavailable (or a path is degenerate) the
module falls back to the existing cubic-Hermite spline timing, so planning never
fails just because TOPP-RA is missing.

Import-safe: ``toppra``/``numpy`` are imported lazily inside the functions.
"""
from __future__ import annotations

from typing import Optional, Sequence

_TOPPRA_OK: Optional[bool] = None


def _toppra_available() -> bool:
    global _TOPPRA_OK
    if _TOPPRA_OK is None:
        try:
            import toppra  # noqa: F401
            _TOPPRA_OK = True
        except Exception:
            _TOPPRA_OK = False
    return _TOPPRA_OK


def _hermite_spline(waypoints, dt: float = 0.1, at_rest: bool = False) -> list[dict]:
    """C² cubic Hermite spline through waypoints, zero velocity at endpoints.

    ``at_rest`` makes every waypoint a stop (zero tangents): each segment is then
    a smoothstep along its straight line, so the path IS the polyline and cannot
    leave a convex joint box its waypoints are in.

    Lives here rather than in rrt_planner because it is pure numpy shared by both
    backends: keeping it in this ROS/LangChain-free module means the timing math
    is importable (and unit-testable) on a bare host, and removes the circular
    import the previous placement required.

    NOTE: the "time" produced is joint-space Euclidean path length, not a
    dynamically-feasible schedule. It is a FALLBACK for when TOPP-RA is absent;
    prefer :func:`parameterize` with velocity limits.
    """
    import numpy as np

    n = len(waypoints)
    if n == 0:
        return []
    if n < 2:
        return [{"positions": np.asarray(waypoints[0]).tolist(), "time_s": 0.0}]

    wps = [np.asarray(w, dtype=float) for w in waypoints]

    # Compute velocities (zero at endpoints, centripetal elsewhere)
    vels = [np.zeros_like(wps[0])] * n
    for i in range(1, n - 1):
        if not at_rest:
            vels[i] = 0.5 * (wps[i + 1] - wps[i - 1])

    traj = []
    t_acc = 0.0
    for i in range(n - 1):
        p0, p1 = wps[i], wps[i + 1]
        v0, v1 = vels[i], vels[i + 1]
        seg_len = max(float(np.linalg.norm(p1 - p0)), 0.05)
        n_pts = max(2, int(seg_len / dt))
        for k in range(n_pts):
            s = k / n_pts
            h00 = 2*s**3 - 3*s**2 + 1
            h10 = s**3 - 2*s**2 + s
            h01 = -2*s**3 + 3*s**2
            h11 = s**3 - s**2
            q = h00*p0 + h10*v0 + h01*p1 + h11*v1
            traj.append({"positions": q.tolist(), "time_s": round(t_acc + s * seg_len, 4)})
        t_acc += seg_len
    traj.append({"positions": wps[-1].tolist(), "time_s": round(t_acc, 4)})
    return traj


def _hermite_fallback(waypoints, velocity_scale: float, at_rest: bool = False) -> list[dict]:
    """Fixed-dt cubic-Hermite timing (the original scheme)."""
    traj = _hermite_spline(waypoints, dt=0.08, at_rest=at_rest)
    if velocity_scale and velocity_scale != 1.0 and velocity_scale > 0:
        for pt in traj:
            pt["time_s"] = round(pt["time_s"] / velocity_scale, 4)
    return traj


def parameterize(
    waypoints: Sequence,
    vel_limits: Optional[Sequence[float]] = None,
    accel_limits: Optional[Sequence[float]] = None,
    velocity_scale: float = 1.0,
    dt: float = 0.05,
    max_deviation_rad: Optional[float] = None,
    position_limits: Optional[tuple] = None,
) -> list[dict]:
    """Time-parameterize a joint-space path.

    Args:
        waypoints:      Ordered joint configurations (list of length-``dof`` seqs).
        vel_limits:     Per-joint velocity limits (rad/s). Required for TOPP-RA;
                        if ``None`` the Hermite fallback is used.
        accel_limits:   Per-joint acceleration limits (rad/s^2). If ``None`` a
                        conservative multiple of the velocity limits is used.
        velocity_scale: Global speed scaling in (0, 1]; scales the velocity and
                        acceleration limits (slower, still dynamically valid).
        dt:             Output sampling period (s).
        max_deviation_rad: How far (per joint) the timed path may leave the
                        polyline through ``waypoints`` — the path the planner
                        validated. Corners are rounded within it; ``None`` (the
                        robot declares no ``path_following``) or 0 follows the
                        polyline exactly, stopping at every corner.
        position_limits: ``(lower, upper)`` joint box (``joint_limits.position_limits``).
                        A candidate whose EMITTED samples leave it is dropped; the
                        exact candidate (straight segments) never leaves a convex
                        box its waypoints are in. The Hermite fallback then stops
                        at every waypoint, for the same reason.

    The timed path follows the polyline. A cubic spline through a few knots at a
    uniform parameter does not: it overshoots between them (umarell issue 18,
    up to 0.7 rad on corner-heavy transits), so the arm executed a path nobody
    checked. Each corner is replaced by a circular arc within
    ``max_deviation_rad`` of it, the knots are refined until the timed path is
    MEASURED within that bound, and the joint limits are re-checked on the timed
    path itself (TOPP-RA enforces them only at its gridpoints).

    Returns:
        ``[{"positions": [...], "time_s": float}, ...]`` starting at t=0.
    """
    import numpy as np

    wp = [list(map(float, q)) for q in waypoints]
    if len(wp) == 0:
        return []
    if len(wp) == 1:
        return [{"positions": wp[0], "time_s": 0.0}]

    vscale = float(velocity_scale) if velocity_scale and velocity_scale > 0 else 1.0
    boxed = position_limits is not None

    def fallback():
        return _hermite_fallback([np.array(q) for q in wp], vscale, at_rest=boxed)

    if not (vel_limits is not None and _toppra_available()):
        return fallback()

    try:
        import toppra as ta
        import toppra.constraint as constraint
        import toppra.algorithm as algo

        arr = np.array(wp, dtype=float)          # (N, dof)
        n, dof = arr.shape
        vl = np.asarray(vel_limits, dtype=float)[:dof] * vscale
        if accel_limits is not None:
            al = np.asarray(accel_limits, dtype=float)[:dof] * (vscale ** 2)
        else:
            al = 2.5 * vl                         # conservative default

        # Drop consecutive duplicate waypoints (zero-length segments break the
        # spline / make the path parameter non-monotonic).
        keep = [0]
        for i in range(1, n):
            if float(np.linalg.norm(arr[i] - arr[keep[-1]])) > 1e-6:
                keep.append(i)
        arr = arr[keep]
        n = len(arr)
        if n < 2:
            return [{"positions": arr[0].tolist(), "time_s": 0.0}]

        # Every candidate is MEASURED against the polyline, and the fastest one
        # within the bound wins. Exact (each straight segment rest-to-rest, no
        # corner to cut) is always a candidate; with a declared bound, so are the
        # spline through the waypoints (dense, smooth paths: a descent's IK steps)
        # and the corner-blended path (sparse corners: a simplified OMPL path).
        delta = float(max_deviation_rad or 0.0)
        exact = [_timed_path(ta, constraint, algo, arr[i:i + 2], vl, al) for i in range(n - 1)]
        exact_ok = all(p is not None for p in exact)
        candidates = [exact] if exact_ok else []
        if delta > 0 and n > 2:
            direct = _timed_path(ta, constraint, algo, arr, vl, al)
            if direct is not None and _within(arr, direct, delta):
                candidates.append([direct])
            blended = _follow_blended(ta, constraint, algo, arr, vl, al, delta)
            if blended is not None:
                candidates.append(blended)
        if boxed and candidates:
            # A spline through in-box knots can overshoot between them (umarell
            # issue 13): keep only the candidates whose emitted samples stay in.
            # None inside: the exact one (straight segments, inside the box exactly
            # when the waypoints are), else the stop-at-every-waypoint fallback —
            # never a spline known to leave the box.
            inside = [c for c in candidates if _inside_box(c, dt, position_limits)]
            if inside:
                candidates = inside
            elif exact_ok:
                candidates = [exact]
            else:
                return fallback()
        if not candidates:
            return fallback()
        pieces = min(candidates, key=lambda c: sum(float(p.duration) for p in c))

        out: list = []
        t0 = 0.0
        for jnt_traj in pieces:
            duration = float(jnt_traj.duration)
            if not (duration > 0):
                return fallback()
            ts = np.arange(0.0, duration, dt)
            if len(ts) == 0 or ts[-1] < duration:
                ts = np.append(ts, duration)
            qs = jnt_traj(ts)                    # (len(ts), dof)
            first = 1 if out else 0              # the junction point once
            out += [{"positions": [round(float(v), 6) for v in qs[i]],
                     "time_s": round(t0 + float(ts[i]), 4)} for i in range(first, len(ts))]
            t0 += duration
        return out
    except Exception:
        # Any TOPP-RA hiccup → safe fallback.
        return fallback()


def _inside_box(pieces, dt: float, box) -> bool:
    """Whether every sample :func:`parameterize` would emit for ``pieces`` is in
    the joint box (``joint_limits`` tolerance)."""
    import numpy as np
    from agents.trajectory.tools.joint_limits import JOINT_LIMIT_TOL_RAD

    lo, hi = (np.asarray(b, float) for b in box)
    for jt in pieces:
        duration = float(jt.duration)
        ts = np.append(np.arange(0.0, duration, dt), duration)
        qs = np.round(np.asarray(jt(ts), float), 6)
        n = min(qs.shape[1], len(lo), len(hi))
        if (qs[:, :n] > hi[:n] + JOINT_LIMIT_TOL_RAD).any() \
                or (qs[:, :n] < lo[:n] - JOINT_LIMIT_TOL_RAD).any():
            return False
    return True


def _timed_path(ta, constraint, algo, knots, vl, al):
    """TOPP-RA through ``knots`` (chord-length parameter), with the limits checked
    on the TIMED path and tightened by the measured excess until it meets them.
    ``None`` when TOPP-RA finds no parameterization."""
    import numpy as np

    ss = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))])
    ss = ss / ss[-1]
    path = ta.SplineInterpolator(ss, knots)
    grid = np.union1d(ss, (ss[:-1] + ss[1:]) / 2) if len(ss) > 2 else None
    v_use, a_use = vl.copy(), al.copy()
    best = None
    for _ in range(len(vl) + 8):              # converges in a few; bounded loop
        kw = {"gridpoints": grid} if grid is not None else {}
        jt = algo.TOPPRA([constraint.JointVelocityConstraint(np.vstack([-v_use, v_use]).T),
                          constraint.JointAccelerationConstraint(np.vstack([-a_use, a_use]).T)],
                         path, parametrizer="ParametrizeConstAccel", **kw).compute_trajectory()
        if jt is None or not float(jt.duration) > 0:
            return best
        ts = np.linspace(0.0, float(jt.duration), max(1000, 40 * len(knots)))
        rv = float(np.max(np.abs(jt(ts, 1)) / vl))
        ra = float(np.max(np.abs(jt(ts, 2)) / al))
        best = jt
        if rv <= 1.0 + 1e-6 and ra <= 1.0 + 1e-6:
            return jt
        v_use, a_use = v_use / max(rv, 1.0), a_use / max(ra, 1.0)
    return None


def _off_polyline(P, Q):
    """Per-sample max-joint distance of ``Q`` from the polyline through ``P``."""
    import numpy as np

    best = np.full(len(Q), np.inf)
    for a, b in zip(P[:-1], P[1:]):
        d = b - a
        t = np.clip(((Q - a) @ d) / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.abs(Q - (a + t[:, None] * d)).max(axis=1))
    return best


def _within(P, jt, delta: float) -> bool:
    """Is the timed path ``jt`` within ``delta`` of the polyline through ``P``?"""
    import numpy as np

    ts = np.linspace(0.0, float(jt.duration), max(400, 8 * len(P)))
    return float(_off_polyline(P, jt(ts)).max()) <= delta


def _blended_knots(P, delta: float, step: float):
    """The polyline through ``P`` with each corner replaced by a circular arc whose
    distance from the corner is at most ``delta`` (and that uses at most half of
    either segment), sampled every ``step`` (max joint motion)."""
    import numpy as np

    def line(a, b):
        k = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / step)))
        return [a + (b - a) * (j / k) for j in range(1, k + 1)]

    pts, cur = [P[0]], P[0]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        u1 = (b - a) / np.linalg.norm(b - a)
        u2 = (c - b) / np.linalg.norm(c - b)
        turn = float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
        if turn < 1e-9 or np.pi - turn < 1e-9:
            continue                           # straight on, or a reversal: keep the corner
        half = turn / 2
        r = delta / (1.0 / np.cos(half) - 1.0)
        tangent = min(r * np.tan(half), np.linalg.norm(b - a) / 2, np.linalg.norm(c - b) / 2)
        r = tangent / np.tan(half)
        s0 = b - tangent * u1
        pts += line(cur, s0)
        w = (u2 - u1) / np.linalg.norm(u2 - u1)
        centre = b + w * (r / np.cos(half))
        e1 = (s0 - centre) / r
        m = max(2, int(np.ceil(r * turn / step)))
        pts += [centre + r * (np.cos(t) * e1 + np.sin(t) * u1)
                for t in np.linspace(0.0, turn, m + 1)[1:]]
        cur = b + tangent * u2
    pts += line(cur, P[-1])
    return np.asarray(pts, dtype=float)


def _follow_blended(ta, constraint, algo, P, vl, al, delta: float):
    """``[trajectory]`` through the corner-blended polyline, MEASURED within
    ``delta`` of it; ``None`` when no knot spacing gets there."""
    import numpy as np

    if len(P) < 3:
        return None                            # one segment: exact is free
    step = float(np.max(np.abs(np.diff(P, axis=0))))   # the longest segment, then halved
    while step >= delta:
        knots = _blended_knots(P, delta, step)
        jt = _timed_path(ta, constraint, algo, knots, vl, al)
        if jt is not None and _within(P, jt, delta):
            return [jt]
        step /= 2                              # too coarse to time or to follow: refine
    return None


# ─── Shared path utilities ────────────────────────────────────────────────────
# Pure numpy, planner-agnostic, and needed by both the planners and the
# validators. They live here for the same reason _hermite_spline does: this
# module imports without ROS or LangChain, so the geometry is testable on a
# bare host and no consumer has to import a planner to reach it.

def _shortcut_path(path: list, threshold: float = 0.06) -> list:
    """Remove path nodes whose removal changes total path length < threshold."""
    import numpy as np

    if len(path) <= 2:
        return path
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Try to skip ahead as far as possible
        j = len(path) - 1
        while j > i + 1:
            removed_delta = sum(
                float(np.linalg.norm(path[k+1] - path[k])) for k in range(i, j)
            ) - float(np.linalg.norm(path[j] - path[i]))
            if removed_delta < threshold:
                break
            j -= 1
        result.append(path[j])
        i = j
    return result


def _sample_at_time(traj: list[dict], t: float) -> np.ndarray:
    """Joint configuration of ``traj`` at wall-clock time ``t`` (linear interp).

    Sampling on real time — not on an index fraction — is what makes a set of
    trajectories with DIFFERENT durations comparable: two arms are only ever in
    collision if they occupy conflicting configurations at the same instant. A
    robot that has finished holds its final configuration (it is parked there,
    and still occupies space).
    """
    import numpy as np

    if not traj:
        return np.zeros(6)
    times = [float(p.get("time_s", 0.0)) for p in traj]
    if t <= times[0]:
        return np.asarray(traj[0]["positions"], dtype=float)
    if t >= times[-1]:
        return np.asarray(traj[-1]["positions"], dtype=float)
    hi = next(i for i, tt in enumerate(times) if tt >= t)
    lo = max(hi - 1, 0)
    span = times[hi] - times[lo]
    a = np.asarray(traj[lo]["positions"], dtype=float)
    b = np.asarray(traj[hi]["positions"], dtype=float)
    if span <= 0:
        return b
    f = (t - times[lo]) / span
    return a + f * (b - a)
