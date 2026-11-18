"""Joint-space path metrics — backend-neutral.

How long a path is, how far out of its way it went. Used by the OMPL backend to
choose between path candidates, and by ``trajectory_quality`` to score a plan
whatever backend produced it — which is why they live here and not in a
backend's module. Numpy only.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def _path_joint_length(waypoints) -> float:
    """Joint-space arc length of a waypoint list, in radians."""
    return float(sum(
        np.linalg.norm(np.asarray(b, float) - np.asarray(a, float))
        for a, b in zip(waypoints, waypoints[1:])
    ))


def _path_directness(waypoints) -> float:
    """How much longer than a straight joint-space line this path is.

    ``||q_goal - q_start||`` is the unreachable lower bound on any path between
    the endpoints, so the ratio is >= 1.0, and 1.0 means "went straight there".
    Degenerate (zero-travel) endpoints score 1.0: a path that need not move
    anywhere is already as direct as it can be.

    This was the score the segment planner picked candidates by, and it is now
    the FALLBACK for when FK is unavailable (see :func:`_path_score_policy`). It
    is a proxy: on ``welding_cell_2_small`` it correlated with tool rise at +0.878 over
    ten solves of one weld transit, and cost no collision checks and no FK. The
    remaining 0.122 is not academic — plan ``bt_1787948812`` shipped a home return
    that rose 0.893 m (peak tool z 3.162 m) with a directness ratio comfortably
    under the 1.6 cutoff, because joint-space length says nothing about WHERE the
    length went. :func:`_path_tool_rise` measures the thing being complained
    about instead of predicting it.
    """
    if len(waypoints) < 2:
        return 1.0
    straight = float(np.linalg.norm(
        np.asarray(waypoints[-1], float) - np.asarray(waypoints[0], float)))
    if straight <= 1e-9:
        return 1.0
    return _path_joint_length(waypoints) / straight


def _path_excess_travel(waypoints) -> float:
    """How far out of its way this path went, in radians. THE owner of "wanders?".

    ``_path_directness`` answers the same question as a RATIO, and a ratio is the
    wrong instrument for a cutoff: it inflates on short moves. Measured on
    ``bt_1788022470``, the W14 transit scores x3.41 — worse-looking than several
    legs nobody objected to — only because its straight-line distance is 0.97 rad.
    The excess in radians is 2.35, which ranks it where it belongs.

    Calibration from that plan, so the separation is on record: legs nobody
    objected to spend 0.05-1.20 rad beyond the straight line (robot1's home 0.14,
    robot2's home in the PREVIOUS plan 1.20), W14 spends 2.35, and the leg an
    operator did complain about — robot2's home return — spends **10.87**.

    Lower is better, 0.0 is a straight line, and it never goes negative: the
    straight line is the unreachable lower bound on any path between the
    endpoints.
    """
    if len(waypoints) < 2:
        return 0.0
    straight = float(np.linalg.norm(
        np.asarray(waypoints[-1], float) - np.asarray(waypoints[0], float)))
    return max(0.0, _path_joint_length(waypoints) - straight)


def _trajectory_excess_travel(traj, nq: "Optional[int]" = None) -> float:
    """:func:`_path_excess_travel` of the TIMED array that ships.

    Same reason :func:`_trajectory_tool_rise` exists alongside
    :func:`_path_tool_rise`: the thing an operator watches is the emitted
    trajectory, not the planner states it was interpolated from. A leg assembled
    by concatenating separately-planned segments (``_concat_timed``) has no single
    set of planner states to score at all.
    """
    if not traj or len(traj) < 2:
        return 0.0
    q = [np.asarray(p["positions"], float) for p in traj]
    if nq is not None:
        q = [x[:nq] for x in q]
    return _path_excess_travel(q)


def _path_rise_policy() -> float:
    """The accepted tool rise for a transit, in metres.

    0.15 m by default, measured rather than chosen: on ``welding_cell_2_small`` the legs
    nobody complained about rose 0.007-0.131 m (the visible 0.100-0.120 m of that
    is the program's own ``approach_offset_m``, i.e. deliberate), while every leg
    an operator objected to rose 0.5-2.8 m. There is no data anywhere in that gap,
    so 0.15 m sits in empty space just above the legitimate approach lift.

    Floored at 0.0 — a negative rise is unreachable by construction and would burn
    every attempt for nothing. ``OMPL_PATH_MAX_RISE_M`` tunes it without a code
    change, for a cell whose fixtures genuinely require arcing over a workpiece.
    """
    try:
        rise = float(os.environ.get("OMPL_PATH_MAX_RISE_M", "0.15") or 0.15)
    except ValueError:
        rise = 0.15
    return max(0.0, rise)
