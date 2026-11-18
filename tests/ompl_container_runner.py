#!/usr/bin/env python3
"""
In-container OMPL backend smoke runner.

This script runs INSIDE the project container, where Pinocchio and the ``ompl``
Python bindings are installed. It invokes the real ``plan_trajectories_ompl``
backend on a cell and prints a single JSON object to stdout for a host-side
client/test to parse.

The host never imports Pinocchio or OMPL — it only shells into the container
(``docker run``) and reads the ``OMPL_RESULT_JSON=`` line this script emits.
See ``tests/test_ompl_backend_container.py``.

Environment:
  CELL_ID          cell to plan for            (default: welding_cell_2_small)
  CELL_BASE_PATH   cells mount inside container (default: /cells)
  OMPL_PLANNER     OMPL planner name           (default: RRTConnect)

Output (stdout):
  A line ``OMPL_RESULT_JSON=<json>`` where <json> is:
    {"ompl_available": bool, "import_error": str|null, "result": {...}|null}
"""
from __future__ import annotations

import json
import os
import sys

# The repo is mounted at the WORKDIR; make sure it is importable.
sys.path.insert(0, os.getcwd())

SENTINEL = "OMPL_RESULT_JSON="


def _emit(payload: dict) -> None:
    print(SENTINEL + json.dumps(payload), flush=True)


def _enclosed_target(cell_id: str, robot_ns: str):
    """The centre of the highest fixture box within 0.8 of ``robot_ns``'s
    workspace radius — inside the fixture, near its top, where the arm reaches
    without folding into itself — or None (no fixtures)."""
    import numpy as np
    from agents.trajectory.tools.robot_kinematics import (
        _load_robot_config, load_static_obstacles,
    )
    rcfg = _load_robot_config(cell_id, robot_ns)
    base = np.asarray(rcfg["base_position"], float)
    reach = 0.8 * float(rcfg.get("workspace_radius") or 2.5)
    best = None
    for o in load_static_obstacles(cell_id):
        if o.get("type") != "box" or o.get("is_workpiece"):
            continue
        c = np.asarray(o["center"], float)
        d = float(np.linalg.norm(c - base))
        if d <= reach and (best is None or c[2] > best[2][2]):
            best = (d, o["id"], c)
    if best is None:
        return None
    _d, oid, c = best
    return {"object_id": oid, "target": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}}


def main() -> int:
    out: dict = {"ompl_available": False, "import_error": None, "result": None}

    try:
        from agents.trajectory.tools.ompl_planner import (
            plan_trajectories_ompl,
            ompl_available,
            ompl_import_error,
        )
    except Exception as exc:  # noqa: BLE001 - report import failure as data
        out["import_error"] = f"backend import failed: {exc!r}"
        _emit(out)
        return 1

    out["ompl_available"] = bool(ompl_available())
    out["import_error"] = ompl_import_error()

    cell_id = os.environ.get("CELL_ID", "welding_cell_2_small")
    planner = os.environ.get("OMPL_PLANNER", "RRTConnect")

    # The deterministic planning tool has TWO first-class behaviours in the
    # agentic flow (OMPL is the real tool):
    #
    #   "free" — a collision-free, reachable motion the tool can plan directly.
    #            Represents both a normal move and the kind of retract/staging
    #            "intermediate point" the LLM uses during collision recovery.
    #
    #   "weld" — a weld point at the CENTRE of one of the cell's own fixture boxes
    #            (the highest one within robot1's reach): any collision model must flag
    #            the goal, so the tool MUST fail with an ACTIONABLE reason. That
    #            failure is the trigger for the LLM to pick a strategy (retract,
    #            stage-above-and-descend, intermediate waypoint, reorient). Taken
    #            from the cell, not a coordinate, so it holds for any cell and any
    #            model fidelity; a cell without fixtures skips it.
    enclosed = _enclosed_target(cell_id, "robot1")
    out["enclosed_target"] = enclosed
    scenarios = {
        "free": [
            {"robot_ns": "robot1", "operation_id": "P1", "operation_type": "move",
             "target": {"x": -0.4, "y": -2.4, "z": 1.4}, "approach_vector": [0.0, 0.0, -1.0]},
            {"robot_ns": "robot2", "operation_id": "P2", "operation_type": "move",
             "target": {"x": -0.4, "y": 0.0, "z": 1.4}, "approach_vector": [0.0, 0.0, -1.0]},
        ],
    }
    if enclosed:
        scenarios["weld"] = [
            {"robot_ns": "robot1", "operation_id": "W01", "operation_type": "weld",
             "target": enclosed["target"], "approach_vector": [0.0, 0.0, -1.0]},
        ]

    out["scenarios"] = {}
    for name, requests in scenarios.items():
        try:
            result = plan_trajectories_ompl(
                cell_id, requests,
                approach_offset_m=0.05, velocity_scale=0.5, planner_name=planner,
            )
        except Exception as exc:  # noqa: BLE001 - surface runtime failures as data
            out["scenarios"][name] = {"success": False, "error": f"planner raised: {exc!r}"}
            continue

        trajectories = result.get("trajectories") or {}
        out["scenarios"][name] = {
            "success": result.get("success"),
            "backend": result.get("backend"),
            "planner": result.get("planner"),
            "planning_time_s": result.get("planning_time_s"),
            "failures": result.get("failures"),
            "trajectory_point_counts": {ns: len(tr) for ns, tr in trajectories.items()},
        }

    _emit(out)
    # Exit non-zero only if the "free" mechanism scenario failed — the "weld"
    # collision failure is the EXPECTED, correct behaviour.
    return 0 if out["scenarios"].get("free", {}).get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
