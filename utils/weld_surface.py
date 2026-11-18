"""A weld target against the part's own mesh: ON the surface, approached from free space.

A program's weld point and approach come from the product definition; nothing
checked them against the geometry the cell is built from. On welding_cell_2_small the 27
points sit 0.2-4.6 mm off the sheet, and for six of them the straight approach
runs into metal 15-45 mm in front of the weld. A point off the surface is
reached at the wrong place (or inside the part), and an approach through metal is
not an approach.

:func:`correct_target` answers, for one weld, from the mesh alone:

* **where the tool touches** — the first surface met coming in along the
  approach, within ``snap_window_m`` of the point (else the nearest surface within
  ``snap_max_m``); the point is moved onto it;
* **from where** — the approach is kept when it is within ``max_tilt_deg`` of the
  surface normal and its corridor (a cylinder of ``corridor_radius_m`` back along
  the approach for ``corridor_length_m``) is free of metal; otherwise the free
  direction nearest the normal is taken, on the side the tool came from, searched
  in cones up to ``max_search_deg``;
* **under another sheet** — when no direction is free, the point lies under a
  sheet the electrode meets first: it is moved out along the approach onto that
  outer surface (within ``max_stack_m``) when the corridor is free from there.

A weld still without a free approach is reported ``unresolved`` and left as it
was — never forced.

``approach`` is the direction the TCP travels INTO the weld (the task programs'
``approach_vector``); the surface normal is reported pointing OUT, toward the tool.

numpy only; geometry is world-space triangles ``(N, 3, 3)``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

SNAP_WINDOW_M = 0.010
SNAP_MAX_M = 0.020
MAX_TILT_DEG = 15.0
MAX_SEARCH_DEG = 45.0
#: The electrode's own radius (a 16 mm cap): what must pass, not the gun body.
CORRIDOR_RADIUS_M = 0.008
CORRIDOR_LENGTH_M = 0.20
#: How far out along the approach an outer sheet may lie over a weld point that
#: nothing reaches (the point moved onto it); beyond, the weld is unresolved.
MAX_STACK_M = 0.06
#: A weld closer than this to its surface is already ON it.
ON_SURFACE_M = 0.0005
#: Triangles this close to the contact point are the face it lies on.
FACE_TOL_M = 0.0002
#: Ignore metal this close in front of the contact: the sheet itself / its edge.
SHEET_TOL_M = 0.003


def as_triangles(triangles) -> np.ndarray:
    return np.asarray(list(triangles) if not isinstance(triangles, np.ndarray) else triangles,
                      float).reshape(-1, 3, 3)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def near(T: np.ndarray, p, radius: float) -> np.ndarray:
    """The triangles whose bounding box comes within ``radius`` of ``p``."""
    lo, hi = T.min(axis=1), T.max(axis=1)
    p = np.asarray(p, float)
    return T[np.all((lo - radius <= p) & (p <= hi + radius), axis=1)]


def closest_points(p, T: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Closest point on every triangle to ``p`` and its distance (Ericson 5.1.5)."""
    p = np.asarray(p, float)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp, cp = p - b, p - c
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v = np.where(denom != 0, vb / denom, 0.0)
        w = np.where(denom != 0, vc / denom, 0.0)
        q = a + ab * v[:, None] + ac * w[:, None]                       # interior
        m = (d1 <= 0) & (d2 <= 0); q[m] = a[m]
        m = (d3 >= 0) & (d4 <= d3); q[m] = b[m]
        m = (d6 >= 0) & (d5 <= d6); q[m] = c[m]
        t = np.clip(np.where(d1 - d3 != 0, d1 / (d1 - d3), 0), 0, 1)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); q[m] = (a + ab * t[:, None])[m]
        t = np.clip(np.where(d2 - d6 != 0, d2 / (d2 - d6), 0), 0, 1)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); q[m] = (a + ac * t[:, None])[m]
        t = np.clip(np.where((d4 - d3) + (d5 - d6) != 0,
                             (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0), 0, 1)
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        q[m] = (b + (c - b) * t[:, None])[m]
    return q, np.linalg.norm(q - p, axis=1)


def ray_hits(origin, direction, T: np.ndarray) -> np.ndarray:
    """Distance along the (unit) ray to every triangle it crosses, either face;
    NaN for a miss (Möller–Trumbore)."""
    o, d = np.asarray(origin, float), _unit(direction)
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - T[:, 0]
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = np.einsum("j,ij->i", d, qv) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t >= 0)
    return np.where(hit, t, np.nan)


def surface_normal(T: np.ndarray, q, toward, tol: float = FACE_TOL_M):
    """Normal of the FACE ``q`` lies on, oriented to ``toward``.

    The face itself, not an average over a neighbourhood: welds sit on flanges,
    next to bends, where faces within 10 mm of the point differ by up to 90 deg —
    an averaged normal there is 35-40 deg off every real face (measured on
    welding_cell_2_small, whose programmed approaches match their own face to 0.0 deg). On
    an edge shared by several faces, the one nearest ``toward``."""
    loc = near(T, q, tol)
    if not len(loc):
        return None
    _cp, d = closest_points(q, loc)
    loc = loc[d <= tol]
    if not len(loc):
        return None
    n = np.cross(loc[:, 1] - loc[:, 0], loc[:, 2] - loc[:, 0])
    keep = np.linalg.norm(n, axis=1) > 1e-15
    if not keep.any():
        return None
    n = n[keep] / np.linalg.norm(n[keep], axis=1)[:, None]
    toward = _unit(toward)
    n = np.where((n @ toward)[:, None] < 0, -n, n)
    return _unit(n[int(np.argmax(n @ toward))])


def corridor_blocked(T: np.ndarray, contact, out_dir, radius: float, length: float,
                     normal=None, step: float = 0.002) -> Optional[float]:
    """Distance in front of ``contact`` (along ``out_dir``, toward the tool) of the
    first metal closer than ``radius`` to the approach axis, or None when the
    corridor is free.

    Measured as the true distance from points along the axis to the mesh, so a
    wall running PARALLEL to the approach (a flange beside the weld) counts — a ray
    test along the approach never meets one. The axis starts where it has left the
    contact sheet: ``radius`` above its plane (``normal``, the face's), plus the
    sheet tolerance."""
    out_dir = _unit(out_dir)
    loc = near(T, contact + out_dir * (length / 2), length / 2 + radius)
    if not len(loc):
        return None
    rise = float(np.dot(out_dir, _unit(normal))) if normal is not None else 1.0
    start = (radius + SHEET_TOL_M) / max(rise, 0.2)
    for t in np.arange(start, length + 1e-9, step):
        q = contact + out_dir * t
        if not len(loc):
            break
        _cp, d = closest_points(q, loc)
        if float(d.min()) < radius:
            return float(t)
    return None


def _cone(axis, deg: float, n_az: int) -> list:
    axis = _unit(axis)
    if deg <= 0:
        return [axis]
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(axis, helper))
    w = np.cross(axis, u)
    s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
    return [_unit(c * axis + s * (math.cos(t) * u + math.sin(t) * w))
            for t in np.linspace(0, 2 * math.pi, n_az, endpoint=False)]


def _angle(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_unit(a), _unit(b)))))))


def correct_target(T: np.ndarray, position, approach, *,
                   snap_window_m: float = SNAP_WINDOW_M, snap_max_m: float = SNAP_MAX_M,
                   max_tilt_deg: float = MAX_TILT_DEG, max_search_deg: float = MAX_SEARCH_DEG,
                   corridor_radius_m: float = CORRIDOR_RADIUS_M,
                   corridor_length_m: float = CORRIDOR_LENGTH_M,
                   max_stack_m: float = MAX_STACK_M) -> dict:
    """One weld against the mesh. Returns::

        {"status": "ok" | "corrected" | "unresolved",
         "position": [3], "approach_vector": [3], "normal": [3] | None,
         "moved_mm": float, "approach_change_deg": float,
         "tilt_deg_before": float | None, "tilt_deg_after": float | None,
         "corridor_blocked_at_m": float | None,      # of the ORIGINAL approach
         "reasons": [str]}
    """
    p = np.asarray(position, float)
    a = _unit(approach)
    out = {"status": "ok", "position": [float(v) for v in p],
           "approach_vector": [float(v) for v in a], "normal": None,
           "moved_mm": 0.0, "approach_change_deg": 0.0, "tilt_deg_before": None,
           "tilt_deg_after": None, "corridor_blocked_at_m": None, "reasons": []}
    reach = corridor_length_m + snap_max_m + corridor_radius_m
    loc = near(T, p, reach)
    if not len(loc):
        out["status"] = "unresolved"
        out["reasons"].append(f"no part surface within {reach:.3f} m")
        return out

    # 1. Where the tool touches: coming in along the approach, the first surface
    #    within the snap window of the point; else the nearest surface.
    back = corridor_length_m
    t = ray_hits(p - a * back, a, loc)
    s = back - t[np.isfinite(t)]                 # > 0: in front of the point
    window = s[np.abs(s) <= snap_window_m]
    if len(window):
        contact = p - a * float(window.max())
    else:
        q, d = closest_points(p, loc)
        k = int(np.argmin(d))
        if d[k] > snap_max_m:
            out["status"] = "unresolved"
            out["reasons"].append(f"the point is {d[k] * 1000:.1f} mm from the nearest "
                                  f"surface (> {snap_max_m * 1000:.0f} mm): not snapped")
            return out
        contact = q[k]
    moved = float(np.linalg.norm(contact - p))

    n = surface_normal(T, contact, -a)
    if n is None:
        n = -a
    out["normal"] = [float(v) for v in n]
    tilt = _angle(-a, n)
    out["tilt_deg_before"] = round(tilt, 2)

    # 2. From where: keep the approach when it is near-normal and its corridor free.
    def corridor(c, u, nrm=None):
        return corridor_blocked(T, c, u, corridor_radius_m, corridor_length_m,
                                normal=n if nrm is None else nrm)

    def free_direction(c, nrm):
        """The free direction (toward the tool) nearest the normal, ties to the
        original approach; None when there is none within max_search_deg."""
        step, k = 5.0, 0
        while k * step <= max_search_deg + 1e-9:
            deg = k * step
            cands = _cone(nrm, deg, 1 if deg == 0 else max(8, int(round(deg / step)) * 8))
            free = [u for u in cands if corridor(c, u, nrm) is None]
            if free:
                return min(free, key=lambda u: _angle(-u, a))
            k += 1
        return None

    blocked = corridor(contact, -a)
    out["corridor_blocked_at_m"] = None if blocked is None else round(blocked, 4)
    new_a = a
    if blocked is not None or tilt > max_tilt_deg:
        if blocked is not None:
            out["reasons"].append(f"metal in the approach corridor {blocked * 1000:.0f} mm "
                                  f"in front of the weld")
        if tilt > max_tilt_deg:
            out["reasons"].append(f"approach {tilt:.1f} deg off the surface normal "
                                  f"(> {max_tilt_deg:.0f})")
        best = free_direction(contact, n)
        if best is None:
            # 3. The point is UNDER another sheet: nothing reaches it from the tool
            #    side. The electrode meets the outer sheet first — the weld is made
            #    there — so move the point out along the approach onto the first
            #    surface from which the corridor is free.
            t2 = ray_hits(contact, -a, loc)
            fronts = sorted(float(x) for x in t2[np.isfinite(t2)]
                            if SHEET_TOL_M < x <= max_stack_m)
            for sk in fronts:
                c2 = contact - a * sk
                n2 = surface_normal(T, c2, -a)
                n2 = -a if n2 is None else n2
                if corridor(c2, -a, n2) is None and _angle(-a, n2) <= max_tilt_deg:
                    cand = -a
                else:
                    cand = free_direction(c2, n2)
                if cand is not None:
                    out["reasons"].append(
                        f"the point is under another sheet: moved {sk * 1000:.0f} mm out "
                        f"along the approach onto the outer surface the electrode meets "
                        f"(review the joint in the product definition)")
                    contact, n, best = c2, n2, cand
                    moved = float(np.linalg.norm(contact - p))
                    out["normal"] = [float(v) for v in n]
                    break
        if best is None:
            out["status"] = "unresolved"
            out["reasons"].append(f"no free approach within {max_search_deg:.0f} deg of "
                                  f"the surface normal, nor from an outer sheet within "
                                  f"{max_stack_m * 1000:.0f} mm: left as it was")
            out["moved_mm"] = round(moved * 1000, 2)
            if moved > ON_SURFACE_M:
                out["position"] = [float(v) for v in contact]
            return out
        new_a = -best
    out["position"] = [float(v) for v in contact]
    out["approach_vector"] = [float(v) for v in _unit(new_a)]
    out["moved_mm"] = round(moved * 1000, 2)
    out["approach_change_deg"] = round(_angle(a, new_a), 2)
    out["tilt_deg_after"] = round(_angle(-new_a, n), 2)
    if moved > ON_SURFACE_M and not any("under another sheet" in r for r in out["reasons"]):
        out["reasons"].insert(0, f"point {moved * 1000:.1f} mm off the surface it is "
                                 f"welded on")
    if moved > ON_SURFACE_M or out["approach_change_deg"] > 0.05:
        out["status"] = "corrected"
    return out


# ── orientation follows the approach ─────────────────────────────────────────

def rotation_between(a, b) -> np.ndarray:
    """The smallest rotation taking direction ``a`` onto ``b`` (3x3)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = _unit(np.cross(a, helper))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rotate_quaternion(q: dict, R: np.ndarray) -> dict:
    """``{x, y, z, w}`` rotated by ``R`` in the world frame (R · R_q)."""
    w, x, y, z = (float(q.get(k, d)) for k, d in (("w", 1.0), ("x", 0.0), ("y", 0.0), ("z", 0.0)))
    nq = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / nq, x / nq, y / nq, z / nq
    Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = R @ Rq
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (M[2, 1] - M[1, 2]) / S, (M[0, 2] - M[2, 0]) / S, (M[1, 0] - M[0, 1]) / S
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        S = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w, x, y, z = (M[2, 1] - M[1, 2]) / S, 0.25 * S, (M[0, 1] + M[1, 0]) / S, (M[0, 2] + M[2, 0]) / S
    elif M[1, 1] > M[2, 2]:
        S = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w, x, y, z = (M[0, 2] - M[2, 0]) / S, (M[0, 1] + M[1, 0]) / S, 0.25 * S, (M[1, 2] + M[2, 1]) / S
    else:
        S = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w, x, y, z = (M[1, 0] - M[0, 1]) / S, (M[0, 2] + M[2, 0]) / S, (M[1, 2] + M[2, 1]) / S, 0.25 * S
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return {"x": round(x, 6), "y": round(y, 6), "z": round(z, 6), "w": round(w, 6)}
