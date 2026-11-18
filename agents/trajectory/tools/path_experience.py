"""Experience-based planning: retrieve a path solved before, repair it, check it.

The idea of OMPL's Lightning / Thunder experience planners, done where this
platform's searches run (``ompl_compound._solve_compound``) and with its own
checker: a robotic cell repeats the same motions — the same arms between the
same fixtures, task after task — so most "new" problems are close to one the
sampling search has already solved, and the search is the expensive part
(5 s x 4 attempts per compound step on welding_cell_2_small, and every failure pays all
of it).

* :func:`record` keeps each path the SEARCH found (not the trivial straight
  lines), per cell and per set of moving arms, in stacked joint space.
* :func:`candidates` returns, for a new start/goal of the same arms, the nearest
  stored paths (joint-space distance of the endpoints, per arm, at most
  ``PLAN_EXPERIENCE_MAX_RAD``) REPAIRED to the new endpoints: the new start, the
  stored interior waypoints (whole, and trimmed to the stretch nearest the new
  endpoints), the new goal.

A candidate is only a suggestion. The caller keeps it only if the CURRENT
problem's validity checker passes every segment of it — the same checker the
search's own solutions are held to, with this step's bystanders and obstacles —
and every emitted-leg check downstream still runs. Retrieval can only save a
search; it cannot admit a motion the search could not have returned.

numpy only; persistence through ``experience_store`` (``PLAN_EXPERIENCE=1``).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from agents.trajectory.tools import experience_store as es

VERSION = 1
#: Nearest stored paths tried per problem.
TOP_K = 3


def _max_rad() -> float:
    try:
        return max(0.0, float(os.environ.get("PLAN_EXPERIENCE_MAX_RAD", "1.5")))
    except ValueError:
        return 1.5


def _distance(n_arms: int, s, g, s_i, g_i) -> float:
    """Endpoint distance per arm (rad, L2), so the threshold does not scale with
    how many arms move."""
    return (float(np.linalg.norm(s - s_i)) + float(np.linalg.norm(g - g_i))) / max(1, n_arms)


def candidates(cell_id: Optional[str], arms: list, start, goal) -> list:
    """Repaired stored paths for this problem, nearest first (stacked vectors)."""
    if not cell_id or not es.enabled(es.PATH_EXPERIENCE):
        return []
    arms = [str(a) for a in arms]
    s, g = np.asarray(start, float).ravel(), np.asarray(goal, float).ravel()
    limit = _max_rad()
    scored = []
    for e in es.entries(cell_id, es.PATH_EXPERIENCE):
        if e.get("v") != VERSION or e.get("arms") != arms:
            continue
        try:
            path = np.asarray(e["path"], float)
            s_i, g_i = np.asarray(e["start"], float), np.asarray(e["goal"], float)
        except (KeyError, TypeError, ValueError):
            continue
        if path.ndim != 2 or len(path) < 2 or path.shape[1] != s.size or s_i.size != s.size:
            continue
        d = _distance(len(arms), s, g, s_i, g_i)
        if d <= limit:
            scored.append((d, path))
    scored.sort(key=lambda t: t[0])
    out: list = []
    for _, path in scored[:TOP_K]:
        interior = path[1:-1]
        out.append([s.tolist(), *interior.tolist(), g.tolist()])
        # Trimmed: from the stored waypoint nearest the new start to the one
        # nearest the new goal — a stored path that started further back is not
        # walked back to its old start first.
        i = int(np.argmin(np.linalg.norm(path - s, axis=1)))
        j = int(np.argmin(np.linalg.norm(path - g, axis=1)))
        if i < j and (i, j) != (0, len(path) - 1):
            out.append([s.tolist(), *path[i + 1:j].tolist(), g.tolist()])
    # Identical candidates (a stored path with no interior) are tried once.
    seen, uniq = set(), []
    for c in out:
        k = tuple(round(v, 6) for q in c for v in q)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def record(cell_id: Optional[str], arms: list, start, goal, path) -> bool:
    """Keep a path the search found for ``arms`` from ``start`` to ``goal``."""
    if not cell_id or not es.enabled(es.PATH_EXPERIENCE):
        return False
    try:
        p = np.asarray(path, float)
        s, g = np.asarray(start, float).ravel(), np.asarray(goal, float).ravel()
    except (TypeError, ValueError):
        return False
    if p.ndim != 2 or len(p) < 3 or p.shape[1] != s.size:
        return False                      # a straight line teaches nothing
    arms = [str(a) for a in arms]
    for e in es.entries(cell_id, es.PATH_EXPERIENCE):
        if (e.get("arms") == arms and _distance(len(arms), s, g, np.asarray(e.get("start")),
                                                np.asarray(e.get("goal"))) < 1e-3):
            return False                  # this problem is already in the library
    return es.append(cell_id, es.PATH_EXPERIENCE, {
        "v": VERSION, "arms": arms,
        "start": [round(float(v), 6) for v in s],
        "goal": [round(float(v), 6) for v in g],
        "path": [[round(float(v), 6) for v in q] for q in p]})
