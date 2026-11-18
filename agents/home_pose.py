"""The one place that answers "where is HOME for this arm?".

Six modules used to answer it independently, each with its own literal copy of
``[0, -1.57, 2.2, 0, -0.8, 0]``, while ``welding_cell_2_small`` declares
``home_pose_deg: [0, -90, 90, 0, 0, 0]`` — a different pose, off by 36 deg at the
elbow and 46 deg at the wrist. That divergence is not cosmetic. The planner
plans a coordinated return to the DECLARED home; the executor's blind fallback
and the robot agents drove to the LITERAL one. So the pose an arm was validated
into was not the pose it was commanded to, and the two disagreed by a third of a
right angle on a joint that swings the whole forearm.

Hardcoding it also silently defeats the cell config: an integrator who changes
``home_pose_deg`` changes what the planner targets and nothing else, so the fix
appears to work while the fallback path keeps the old behaviour.

This module is the single source of truth. It reads the cell's own
``robots.yaml`` and falls back to the historical literal ONLY when a cell
declares no home of its own — so an undeclared HOME is still one pose across the
whole system, which is the property that was actually missing.

Deliberately dependency-light: yaml and math, no planner and no Pinocchio, so
the robot agents and the executor can import it without pulling the trajectory
stack into their containers.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

# Used ONLY when a cell declares no home pose. Kept at its historical value so
# that cells which never declared one do not silently start parking somewhere
# new — the point here is agreement, not a new pose.
HOME_JOINTS_FALLBACK: tuple = (0.0, -1.57, 2.2, 0.0, -0.8, 0.0)


def home_pose(cell_id: str, robot_ns: str,
              fallback: Optional[Sequence[float]] = None) -> list:
    """The arm's HOME configuration in RADIANS, from the cell that owns it.

    Accepts ``home_pose_deg`` (how the cells declare it, converted here) or
    ``home_pose`` / ``home_joints`` in radians. Any failure to read the config
    yields ``fallback`` rather than raising: HOME is needed most on the failure
    path — an e-stop, a plan that would not solve — and a resolver that throws
    there would take out the recovery it was asked to serve.
    """
    fb = [float(v) for v in (HOME_JOINTS_FALLBACK if fallback is None else fallback)]
    try:
        # Imported lazily and from the shared geometry module rather than
        # duplicated: the namespace-matching rules for robots.yaml belong in one
        # place, and a second copy here is exactly how the two would drift apart.
        from agents.trajectory.tools.robot_kinematics import _load_robot_config

        rcfg = _load_robot_config(cell_id, robot_ns)
    except Exception:
        return fb
    deg = rcfg.get("home_pose_deg")
    if deg:
        return [math.radians(float(v)) for v in deg]
    rad = rcfg.get("home_pose") or rcfg.get("home_joints")
    if rad:
        return [float(v) for v in rad]
    return fb


def program_start_pose(cell_id: str, robot_ns: str) -> Optional[list]:
    """Where an arm stands when a program STARTS: its HOME — the pre-run home reset
    drives it there, and every backend chains a program's first leg from it. The
    assembler's first-piece check (``behaviour_tree._chain_breaks``) measures
    against this. None when it cannot be answered."""
    # Read here rather than through home_pose, whose fallback answers for a cell
    # it could not read: a guess must not be measured against.
    try:
        from agents.trajectory.tools.robot_kinematics import _load_robot_config
        _load_robot_config(cell_id, robot_ns)
    except Exception:  # noqa: BLE001
        return None
    try:
        return [float(v) for v in home_pose(cell_id, robot_ns)]
    except Exception:  # noqa: BLE001
        return None
