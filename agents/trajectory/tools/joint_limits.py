"""Joint position limits: one owner for "is this motion inside the arm's joint box?".

Every planner, validator and home/park path asks the same question, so it is
answered here, once (umarell issue 13):

* :func:`position_limits` — the arm's ``(lower, upper)`` position limits. The
  declared source is the cell's ``robots.yaml`` (``joint_limits_lower`` /
  ``joint_limits_upper``, which the converter writes from the URDF ``<limit>``),
  intersected with the loaded model's own limits when a model is given. The
  tighter bound wins, so a cell that narrows a joint is honoured and a yaml that
  omits or widens a joint cannot loosen the URDF. Sized by the model, so it works
  for any arm, whatever its joint count.
* :func:`trajectory_limit_violation` — the worst excursion of a trajectory
  outside the box, or ``None``.

Why checking the samples is enough for the motion BETWEEN them: the executor and
every validator interpolate linearly between consecutive samples, and the joint
box is convex, so a straight segment between two in-box samples cannot leave the
box. A segment can only leave it if one of its END samples is outside, and that
sample is reported. (A spline drawn THROUGH in-box knots is a different matter —
it can overshoot between them; ``time_parameterization.parameterize`` filters
those candidates with this same box.)

Pure numpy; the robot config is read lazily. No ROS, no LangChain.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

# Emitted positions are rounded to 1e-6 rad, and IK clips onto the limit itself,
# so a sample exactly at the limit may read up to ~5e-7 past it. 1e-5 rad
# (0.0006 deg) absorbs that and nothing a controller would notice.
JOINT_LIMIT_TOL_RAD = 1e-5


def _model_limits(model) -> "tuple[Optional[np.ndarray], Optional[np.ndarray]]":
    try:
        lo = np.asarray(model.lowerPositionLimit, float)
        hi = np.asarray(model.upperPositionLimit, float)
    except Exception:  # noqa: BLE001 — a stub model has none
        return None, None
    return lo, hi


def position_limits(rcfg: dict, model=None,
                    nq: Optional[int] = None) -> "tuple[np.ndarray, np.ndarray]":
    """``(lower, upper)`` position limits (rad) of the arm's first ``nq`` joints.

    ``nq`` defaults to the model's, else to the yaml's joint count. The yaml
    limits intersected with the model's. A joint whose model limit is missing or
    non-finite (continuous joint, or a model built without limits) keeps the yaml
    value; a joint the yaml does not list keeps the model value; with neither, the
    joint is unbounded.
    """
    yl = list(rcfg.get("joint_limits_lower") or [])
    yu = list(rcfg.get("joint_limits_upper") or [])
    if nq is None:
        # The model's joint count, else the yaml's; with neither, a 6-axis arm
        # (what every caller assumed before this module).
        nq = int(getattr(model, "nq", 0) or 0) or max(len(yl), len(yu)) or 6
    lo = np.full(nq, -np.inf)
    hi = np.full(nq, np.inf)
    for i, v in enumerate(yl[:nq]):
        lo[i] = float(v)
    for i, v in enumerate(yu[:nq]):
        hi[i] = float(v)
    ulo, uhi = _model_limits(model) if model is not None else (None, None)
    if ulo is not None and uhi is not None:
        for i in range(min(nq, len(ulo), len(uhi))):
            # Pinocchio stores "no limit" as +-max float; only finite, ordered
            # limits narrow the box.
            a, b = float(ulo[i]), float(uhi[i])
            if math.isfinite(a) and math.isfinite(b) and a < b and abs(a) < 1e6 and abs(b) < 1e6:
                lo[i] = max(lo[i], a)
                hi[i] = min(hi[i], b)
    return lo, hi


def planning_box(rcfg: dict, model=None, unbounded_rad: float = math.pi
                 ) -> "tuple[np.ndarray, np.ndarray]":
    """:func:`position_limits` with every unbounded side closed at
    ``±unbounded_rad`` — the finite bounds a search space and an IK clip need.
    (Replaces the old literal ``[-3.0]*6`` default: same order of magnitude, but
    the arm's own joint count and its declared limits wherever there are any.)"""
    lo, hi = position_limits(rcfg, model)
    return (np.where(np.isfinite(lo), lo, -unbounded_rad),
            np.where(np.isfinite(hi), hi, unbounded_rad))


def joint_names(rcfg: dict, model=None) -> Optional[list]:
    """The arm's joint names for reports: robots.yaml ``joint_names``, else the
    model's movable joints, else None (reported as ``a1..aN``)."""
    names = rcfg.get("joint_names")
    if names:
        return [str(n) for n in names]
    try:
        return [str(n) for n in list(model.names)[1:]]
    except Exception:  # noqa: BLE001
        return None


def _positions(traj) -> "tuple[np.ndarray, list]":
    rows, meta = [], []
    for pt in traj or []:
        if isinstance(pt, dict):
            rows.append([float(v) for v in (pt.get("positions") or [])])
            meta.append(pt)
        else:
            rows.append([float(v) for v in pt])
            meta.append(None)
    if not rows:
        return np.zeros((0, 0)), meta
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], float), meta


def trajectory_limit_violation(traj, lower: Sequence[float], upper: Sequence[float],
                               tol: float = JOINT_LIMIT_TOL_RAD,
                               names: Optional[Sequence[str]] = None) -> Optional[dict]:
    """Worst position-limit excursion of ``traj`` (dicts with ``positions`` or raw
    vectors), or ``None`` when every sample is inside ``[lower - tol, upper + tol]``.

    Covers the linear motion between samples too (convex box, see the module
    docstring). Returns ``{"sample_index", "joint_index", "joint", "value_rad",
    "lower_rad", "upper_rad", "excess_rad", "excess_deg", "samples_outside",
    "time_s"?, "phase"?}``.
    """
    q, meta = _positions(traj)
    if q.size == 0:
        return None
    n = min(q.shape[1], len(lower), len(upper))
    lo = np.asarray(lower, float)[:n]
    hi = np.asarray(upper, float)[:n]
    q = q[:, :n]
    over = np.where(~np.isfinite(q), np.inf, np.maximum(q - hi, lo - q))
    outside = over > tol
    if not outside.any():
        return None
    k, j = np.unravel_index(int(np.argmax(over)), over.shape)
    ex = float(over[k, j])
    out = {
        "sample_index": int(k),
        "joint_index": int(j),
        "joint": (names[j] if names and j < len(names) else f"a{j + 1}"),
        "value_rad": round(float(q[k, j]), 6),
        "lower_rad": round(float(lo[j]), 6),
        "upper_rad": round(float(hi[j]), 6),
        "excess_rad": round(ex, 6) if math.isfinite(ex) else None,
        "excess_deg": round(math.degrees(ex), 3) if math.isfinite(ex) else None,
        "samples_outside": int(outside.any(axis=1).sum()),
    }
    m = meta[k]
    if isinstance(m, dict):
        for key in ("time_s", "phase"):
            if key in m:
                out[key] = m.get(key)
    return out


def leg_limit_violation(cell_id: str, robot_ns: str, traj) -> Optional[dict]:
    """:func:`trajectory_limit_violation` of one arm's leg against its own
    :func:`position_limits` (the cell's robots.yaml ∩ the arm's model)."""
    if not traj:
        return None
    from agents.trajectory.tools.robot_kinematics import (
        _load_pinocchio_model, _load_robot_config,
    )
    rcfg = _load_robot_config(cell_id, robot_ns)
    model, _ = _load_pinocchio_model(cell_id, robot_ns)
    lo, hi = position_limits(rcfg, model)
    return trajectory_limit_violation(traj, lo, hi, names=joint_names(rcfg, model))


def describe(v: dict, robot_ns: str = "") -> str:
    """One-line, actionable description of a violation."""
    who = f"{robot_ns}'s " if robot_ns else ""
    side = "upper" if v.get("value_rad", 0.0) > v.get("upper_rad", 0.0) else "lower"
    return (f"{who}joint {v.get('joint')} reaches {v.get('value_rad')} rad at sample "
            f"{v.get('sample_index')}, {v.get('excess_deg')} deg past its {side} limit "
            f"[{v.get('lower_rad')}, {v.get('upper_rad')}] rad "
            f"({v.get('samples_outside')} sample(s) outside). The arm cannot execute "
            "this motion: re-plan the leg (another IK branch / an intermediate "
            "waypoint), do not delay or reorder it.")
