"""Compiled (Numba) kernels for the collision checks' innermost loops — optional.

The NumPy versions in ``robot_kinematics`` are correct and stay the reference;
these are the same rules as plain loops compiled to machine code, which can stop
at the FIRST hit (NumPy always computes the whole matrix) and avoid the
per-call overhead of many small array operations. Measured on welding_cell_2_small the
assembler made ~518k small NumPy reductions per assembly.

* :func:`moving_first_hit` — ``robot_kinematics._moving_hit``: the first
  ``(obstacle k, point i)`` in obstacle-major order with ``|p - o|^2 < thr_k^2``.
* :func:`box_first_hit` — ``robot_kinematics._first_box_hit``: the first breached
  box in declaration order, then sphere order, with the same margins, the same
  tool-contact exemption and the same closest point and clearance.
* :func:`pair_min_gap` — ``robot_kinematics._check_robot_pair_collision``: the
  same branch and bound over gate groups, giving the same exact minimum gap.
* :func:`link_fill` — ``robot_kinematics._link_fill_positions``: the same points
  along the joint chain, in the same order.

Both are held equal to the NumPy versions and to brute force by
``tests/trajectory/test_vectorised_collision.py``.

Availability: used only when ``numba`` imports and ``NUMBA_GEOM`` is not ``0``.
Otherwise :data:`AVAILABLE` is False and the callers keep their NumPy path, so
removing numba (or setting ``NUMBA_GEOM=0``) is a complete rollback.
Compiled code is cached on disk (``cache=True``; the directory is
``NUMBA_CACHE_DIR`` when set — the stack points it at the planner's persistent
``/scratch`` volume), so a restart does not recompile; numba rebuilds the cache
itself when this file, numba or the CPU changes.
"""
from __future__ import annotations

import os

import numpy as np

AVAILABLE = False
_REASON = "not tried"

if os.environ.get("NUMBA_GEOM", "1").strip().lower() in ("0", "false", "no", "off"):
    _REASON = "disabled by NUMBA_GEOM=0"
else:
    try:
        import numba as _nb

        @_nb.njit(cache=True, nogil=True)
        def _moving_first_hit(P, O, thr2):
            for k in range(O.shape[0]):
                ox, oy, oz, t = O[k, 0], O[k, 1], O[k, 2], thr2[k]
                for i in range(P.shape[0]):
                    dx = P[i, 0] - ox
                    dy = P[i, 1] - oy
                    dz = P[i, 2] - oz
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < t:
                        return k, i, d2
            return -1, -1, 0.0

        @_nb.njit(cache=True, nogil=True)
        def _box_first_hit(P, R, own, bc, bh, bm, extra, tool, tool_ok, exempt):
            n = P.shape[0]
            if n == 0:
                return -1, -1, 0.0, 0.0, 0.0, 0.0
            # The spheres' bounding box, grown by the largest clearance any pair
            # can need: a box outside it cannot be breached (the NumPy broad phase,
            # done once instead of per chunk).
            lo0, lo1, lo2 = P[0, 0], P[0, 1], P[0, 2]
            hi0, hi1, hi2 = lo0, lo1, lo2
            rmax, omax = R[0], -1.0
            for i in range(n):
                lo0 = min(lo0, P[i, 0]); hi0 = max(hi0, P[i, 0])
                lo1 = min(lo1, P[i, 1]); hi1 = max(hi1, P[i, 1])
                lo2 = min(lo2, P[i, 2]); hi2 = max(hi2, P[i, 2])
                rmax = max(rmax, R[i])
                if not np.isnan(own[i]):
                    omax = max(omax, own[i])
            for m in range(bc.shape[0]):
                g = rmax + max(bm[m], omax) + extra
                if (bc[m, 0] + bh[m, 0] < lo0 - g or bc[m, 0] - bh[m, 0] > hi0 + g or
                        bc[m, 1] + bh[m, 1] < lo1 - g or bc[m, 1] - bh[m, 1] > hi1 + g or
                        bc[m, 2] + bh[m, 2] < lo2 - g or bc[m, 2] - bh[m, 2] > hi2 + g):
                    continue
                for i in range(n):
                    if exempt and tool[i] and tool_ok[m]:
                        continue
                    need = R[i] + (bm[m] if np.isnan(own[i]) else own[i]) + extra
                    c0 = min(max(P[i, 0] - bc[m, 0], -bh[m, 0]), bh[m, 0])
                    c1 = min(max(P[i, 1] - bc[m, 1], -bh[m, 1]), bh[m, 1])
                    c2 = min(max(P[i, 2] - bc[m, 2], -bh[m, 2]), bh[m, 2])
                    e0 = P[i, 0] - bc[m, 0] - c0
                    e1 = P[i, 1] - bc[m, 1] - c1
                    e2 = P[i, 2] - bc[m, 2] - c2
                    d = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
                    if d < need:
                        return m, i, d - need, bc[m, 0] + c0, bc[m, 1] + c1, bc[m, 2] + c2
            return -1, -1, 0.0, 0.0, 0.0, 0.0

        @_nb.njit(cache=True, nogil=True)
        def _group_bounds(P, S, gs, ge):
            k = gs.shape[0]
            c = np.zeros((k, 3))
            r = np.zeros(k)
            m = np.full(k, -np.inf)
            for g in range(k):
                n = ge[g] - gs[g]
                for i in range(gs[g], ge[g]):
                    c[g, 0] += P[i, 0]; c[g, 1] += P[i, 1]; c[g, 2] += P[i, 2]
                    m[g] = max(m[g], S[i])
                if n > 0:
                    c[g] /= n
                for i in range(gs[g], ge[g]):
                    dx = P[i, 0] - c[g, 0]; dy = P[i, 1] - c[g, 1]; dz = P[i, 2] - c[g, 2]
                    r[g] = max(r[g], np.sqrt(dx * dx + dy * dy + dz * dz))
            return c, r, m

        @_nb.njit(cache=True, nogil=True)
        def _pair_min_gap(A, B, sa, sb, ga_s, ga_e, gb_s, gb_e, radius):
            ca, ra, ma = _group_bounds(A, sa, ga_s, ga_e)
            cb, rb, mb = _group_bounds(B, sb, gb_s, gb_e)
            nu, nv = ca.shape[0], cb.shape[0]
            lb = np.empty(nu * nv)
            for u in range(nu):
                for v in range(nv):
                    dx = ca[u, 0] - cb[v, 0]; dy = ca[u, 1] - cb[v, 1]; dz = ca[u, 2] - cb[v, 2]
                    lb[u * nv + v] = (np.sqrt(dx * dx + dy * dy + dz * dz) - ra[u] - rb[v]
                                      - 2.0 * radius - ma[u] - mb[v])
            order = np.argsort(lb)
            best = np.inf
            for f in order:
                if lb[f] >= best:
                    break
                u, v = f // nv, f % nv
                for i in range(ga_s[u], ga_e[u]):
                    for j in range(gb_s[v], gb_e[v]):
                        dx = A[i, 0] - B[j, 0]; dy = A[i, 1] - B[j, 1]; dz = A[i, 2] - B[j, 2]
                        d = np.sqrt(dx * dx + dy * dy + dz * dz) - 2.0 * radius - sa[i] - sb[j]
                        if d < best:
                            best = d
            return best

        @_nb.njit(cache=True, nogil=True)
        def _link_fill(J, base, yaw, step):
            cy, sy = np.cos(yaw), np.sin(yaw)
            nj = J.shape[0]
            W = np.empty((nj, 3))
            for j in range(nj):
                W[j, 0] = base[0] + J[j, 0] * cy - J[j, 1] * sy
                W[j, 1] = base[1] + J[j, 0] * sy + J[j, 1] * cy
                W[j, 2] = base[2] + J[j, 2]
            cap = 0
            for j in range(1, nj - 1):
                dx = W[j + 1, 0] - W[j, 0]; dy = W[j + 1, 1] - W[j, 1]; dz = W[j + 1, 2] - W[j, 2]
                cap += int(np.sqrt(dx * dx + dy * dy + dz * dz) / step) + 1
            out = np.empty((cap, 3))
            n_out = 0
            for j in range(1, nj - 1):
                ax, ay, az = W[j, 0], W[j, 1], W[j, 2]
                ex, ey, ez = W[j + 1, 0] - ax, W[j + 1, 1] - ay, W[j + 1, 2] - az
                seg = np.sqrt(ex * ex + ey * ey + ez * ez)
                n = int(seg / step)
                if n < 1:
                    continue
                dx, dy, dz = ex / seg, ey / seg, ez / seg
                for k in range(1, n + 1):
                    t = k * step
                    if t >= seg:
                        break
                    out[n_out, 0] = ax + dx * t
                    out[n_out, 1] = ay + dy * t
                    out[n_out, 2] = az + dz * t
                    n_out += 1
            return out[:n_out]

        AVAILABLE = True
        _REASON = f"numba {_nb.__version__}"
    except Exception as exc:  # noqa: BLE001 — no numba: the NumPy path stands
        AVAILABLE = False
        _REASON = f"numba unavailable: {type(exc).__name__}: {exc}"


def status() -> str:
    return _REASON


def moving_first_hit(P: np.ndarray, O: np.ndarray, thr: np.ndarray):
    """``(k, i, gap)`` of the first point within its obstacle's threshold, or None."""
    k, i, d2 = _moving_first_hit(np.ascontiguousarray(P, dtype=np.float64),
                                 np.ascontiguousarray(O, dtype=np.float64),
                                 np.ascontiguousarray(thr * thr, dtype=np.float64))
    if k < 0:
        return None
    return int(k), int(i), float(np.sqrt(d2) - thr[k])


def box_first_hit(P, R, own, comp, extra_margin, tool=None):
    """``(box m, sphere i, closest point, clearance)`` — ``_first_box_hit``'s answer."""
    bc = np.ascontiguousarray(comp["box_c"], dtype=np.float64)
    if not len(bc) or not len(P):
        return None
    tool_ok = comp.get("box_tool_ok")
    n = len(P)
    exempt = (tool is not None and tool_ok is not None and len(tool_ok) == len(bc)
              and len(tool) == n and bool(np.any(tool)) and bool(np.any(tool_ok)))
    t = np.ascontiguousarray(tool if exempt else np.zeros(n, bool), dtype=np.bool_)
    tok = np.ascontiguousarray(tool_ok if exempt else np.zeros(len(bc), bool), dtype=np.bool_)
    m, i, gap, x, y, z = _box_first_hit(
        np.ascontiguousarray(P, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64),
        np.ascontiguousarray(own, dtype=np.float64), bc,
        np.ascontiguousarray(comp["box_h"], dtype=np.float64),
        np.ascontiguousarray(comp["box_m"], dtype=np.float64),
        float(extra_margin), t, tok, bool(exempt))
    if m < 0:
        return None
    return int(m), int(i), np.array([x, y, z]), float(gap)


def _starts_ends(groups, n):
    gs = groups or [slice(0, n)]
    return (np.array([g.start or 0 for g in gs], np.int64),
            np.array([n if g.stop is None else g.stop for g in gs], np.int64))


def pair_min_gap(A, B, sa, sb, groups1, groups2, radius: float) -> float:
    """``_check_robot_pair_collision``'s minimum gap (same branch and bound)."""
    ga_s, ga_e = _starts_ends(groups1, len(A))
    gb_s, gb_e = _starts_ends(groups2, len(B))
    return float(_pair_min_gap(np.ascontiguousarray(A, dtype=np.float64),
                               np.ascontiguousarray(B, dtype=np.float64),
                               np.ascontiguousarray(sa, dtype=np.float64),
                               np.ascontiguousarray(sb, dtype=np.float64),
                               ga_s, ga_e, gb_s, gb_e, float(radius)))


def link_fill(J, base, yaw, step: float) -> list:
    """``_link_fill_positions``' points from the joint origins ``J`` (njoints, 3)."""
    out = _link_fill(np.ascontiguousarray(J, dtype=np.float64),
                     np.ascontiguousarray(np.asarray(base, float)[:3], dtype=np.float64),
                     float(yaw), float(step))
    return list(out)
