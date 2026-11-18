"""Fit a robot tool's collision model from the tool's own meshes.

``scripts/convert_isaac_scene.py`` writes each robot's ``tool_collision`` block
(robots.yaml) and the cell's ``tool_link_radius`` (obstacles.yaml). They used to
be fixed placeholders, and the hand-measured replacements were fitted to a table
that dropped the URDF ``<origin rpy>`` (umarell upstream issue 02: 72.9 %
coverage, worst gap 0.295 m, 21 of 35 centres in empty space). This module
derives both from the geometry instead, so re-converting a cell reproduces them.

* :func:`fit_tool_spheres` — sphere centres, in the mount (tip-link) frame, such
  that every sampled point of the tool's surface lies within ``radius`` of a
  centre, every centre lies ON the metal, and the count is small (greedy set
  cover). ``radius`` is the radius the runtime gives these spheres: the cell's
  ``robot_link_radius`` (they are checked arm against arm).
* :func:`tool_link_radius` — the tool's radial extent around its axis near the
  TCP: what the TCP frame's sphere must cover.

Input is triangles already placed in the mount frame (the converter applies each
mesh's full ``<origin xyz rpy>`` and scale). Numpy only; deterministic.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

#: Surface samples per square metre of tool surface (plus every vertex).
SAMPLES_PER_M2 = 20000.0
MAX_SAMPLES = 20000
#: Candidate centres are the surface points thinned on this grid (m).
CANDIDATE_GRID_M = 0.02
#: The cover is fitted this much INSIDE the runtime radius, so the surface between
#: samples is covered too (the samples are ~1 cm apart at SAMPLES_PER_M2).
FIT_MARGIN_M = 0.01


def _as_array(triangles: Iterable[Sequence]) -> np.ndarray:
    tri = np.asarray([[list(v) for v in t] for t in triangles], dtype=float)
    if tri.size == 0:
        return np.zeros((0, 3, 3))
    return tri.reshape(-1, 3, 3)


def surface_points(triangles, *, per_m2: float = SAMPLES_PER_M2,
                   max_samples: int = MAX_SAMPLES, seed: int = 0) -> np.ndarray:
    """Every vertex plus area-weighted random samples of the surface."""
    tri = _as_array(triangles)
    if len(tri) == 0:
        return np.zeros((0, 3))
    verts = np.unique(np.round(tri.reshape(-1, 3), 6), axis=0)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(area.sum())
    n = int(min(max_samples, max(0.0, total * per_m2)))
    if n == 0 or total <= 0.0:
        return verts
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), size=n, p=area / total)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    t = tri[pick]
    samples = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return np.vstack([verts, samples])


def _thin(points: np.ndarray, grid: float) -> np.ndarray:
    """One representative point per occupied grid cell (the first, deterministic)."""
    if len(points) == 0:
        return points
    keys = np.floor(points / grid).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def fit_tool_spheres(triangles, radius: float, *, candidate_grid: float = CANDIDATE_GRID_M,
                     seed: int = 0) -> Optional[dict]:
    """Sphere centres covering the tool surface at ``radius``.

    The cover is built at ``radius - FIT_MARGIN_M`` and REPORTED at ``radius``.

    Returns ``{"centres": [[x,y,z], ...], "vertex_coverage", "sample_coverage",
    "worst_gap_m", "max_centre_to_metal_m", "points", "vertices"}``, or ``None``
    when there is no geometry. Coverage is the fraction of points within
    ``radius`` of a centre; by construction it is 1.0 for every point that took
    part in the fit.
    """
    tri = _as_array(triangles)
    if len(tri) == 0:
        return None
    verts = np.unique(np.round(tri.reshape(-1, 3), 6), axis=0)
    points = surface_points(tri, seed=seed)
    cands = _thin(points, candidate_grid)

    fit_r = max(radius - FIT_MARGIN_M, 0.5 * radius)
    r2 = fit_r * fit_r
    uncovered = np.ones(len(points), dtype=bool)
    chosen: list[int] = []
    # covers[i, j]: candidate i covers point j. Built in chunks to bound memory.
    covers = np.zeros((len(cands), len(points)), dtype=bool)
    for s in range(0, len(cands), 64):
        d2 = ((cands[s:s + 64, None, :] - points[None, :, :]) ** 2).sum(-1)
        covers[s:s + 64] = d2 <= r2
    while uncovered.any():
        gain = covers[:, uncovered].sum(axis=1)
        best = int(np.argmax(gain))
        if gain[best] == 0:          # cannot happen: every point is near a candidate
            break
        chosen.append(best)
        uncovered &= ~covers[best]
    centres = cands[chosen]

    def _nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.empty(len(a))
        for s in range(0, len(a), 2048):
            out[s:s + 2048] = np.sqrt(((a[s:s + 2048, None, :] - b[None, :, :]) ** 2)
                                      .sum(-1).min(axis=1))
        return out

    gap_pts = _nearest(points, centres)
    gap_verts = _nearest(verts, centres)
    # Nearest the mount first, so the YAML reads flange -> tip.
    centres = centres[np.argsort(np.linalg.norm(centres, axis=1), kind="stable")]
    return {
        "centres": [[round(float(c), 4) for c in p] for p in centres],
        "vertex_coverage": float((gap_verts <= radius + 1e-9).mean()),
        "sample_coverage": float((gap_pts <= radius + 1e-9).mean()),
        "worst_gap_m": float(gap_pts.max()),
        "max_centre_to_metal_m": float(_nearest(centres, points).max()),
        "points": int(len(points)),
        "vertices": int(len(verts)),
    }


def tool_link_radius(triangles, tcp_offset: Sequence[float], *,
                     zone_m: float = 0.05, step_m: float = 0.005) -> Optional[float]:
    """The tool's radial extent around the TCP axis within ``zone_m`` of the TCP.

    The axis runs from the mount origin to the TCP (``tcp_offset``). Rounded UP
    to ``step_m``. ``None`` when the TCP is at the mount origin (no axis) or no
    geometry lies in the zone.
    """
    t = np.asarray(tcp_offset, dtype=float)
    length = float(np.linalg.norm(t))
    pts = surface_points(triangles)
    if length < 1e-6 or len(pts) == 0:
        return None
    a = t / length
    s = pts @ a
    near = (s >= length - zone_m) & (s <= length + zone_m)
    if not near.any():
        return None
    radial = np.linalg.norm(pts[near] - np.outer(s[near], a), axis=1)
    return math.ceil(float(radial.max()) / step_m - 1e-9) * step_m
