"""Per-point joint velocities for a FollowJointTrajectory goal. ROS-free.

The controller interpolates between the points it is sent with a cubic per
joint, from each point's position AND velocity. Every point used to go out with
``velocities = [0] * 6``: each segment then starts and ends at rest, so a
timed (TOPP-RA) leg was executed as a stop-and-go chain whose mid-segment speed
peaks at 1.5x the planned speed — past the very limits the timing was computed
for — and the arm shudders at every sample.

The velocity sent at an interior point is the monotone (Fritsch–Carlson)
tangent of the two segments it joins: their harmonic mean when the joint keeps
moving the same way, zero where it reverses or holds. A monotone tangent keeps
each joint inside the range of its segment, so the executed motion stays in the
joint box of the validated straight segment (TOPP-RA samples are dense: the
deviation from the segment is second order), and it never overshoots a hold.
Ends are at rest. A point that carries its own ``velocities`` keeps them.
"""
from __future__ import annotations


def point_velocities(trajectory: list, n_joints: int = 6) -> list:
    """``[[v_j, ...] per point]`` for ``trajectory`` (``positions`` + ``time_s``,
    already time-scaled)."""
    pts = list(trajectory or [])
    n = len(pts)
    out = [[0.0] * n_joints for _ in range(n)]
    if n < 3:
        return _own(pts, out, n_joints)

    def q(k):
        p = list(pts[k].get("positions") or [])[:n_joints]
        return [float(v) for v in p] + [0.0] * (n_joints - len(p))

    t = [float(p.get("time_s", 0.0)) for p in pts]
    Q = [q(k) for k in range(n)]
    for k in range(1, n - 1):
        h0, h1 = t[k] - t[k - 1], t[k + 1] - t[k]
        if h0 <= 0.0 or h1 <= 0.0:
            continue                      # a repeated time: hold there
        for j in range(n_joints):
            s0 = (Q[k][j] - Q[k - 1][j]) / h0
            s1 = (Q[k + 1][j] - Q[k][j]) / h1
            if s0 * s1 > 0.0:             # same direction on both sides
                out[k][j] = 2.0 * s0 * s1 / (s0 + s1)
    return _own(pts, out, n_joints)


def _own(pts, out, n_joints):
    for k, p in enumerate(pts):
        v = p.get("velocities") if isinstance(p, dict) else None
        if v:
            vv = [float(x) for x in list(v)[:n_joints]]
            out[k] = vv + [0.0] * (n_joints - len(vv))
    return out
