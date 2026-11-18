"""``roboballet`` — the backend entry point.

``plan`` loads the selected policy, describes the planning cell in the policy's
terms (chains + obstacle primitives, :mod:`.cell_scene`) and runs
``ompl-compound``'s step pipeline with the policy rollout as its transit search
(:mod:`.rollout`). The cell the plan is for need not be one the policy was
trained on: the graph is built from the cell's own robots.yaml, URDF and
obstacles, which is the generalisation RoboBallet is trained for.

Fail-closed: no policy, an unreadable policy or a policy for another graph
schema refuses every request of the step with ``planner_error`` and says why.
Nothing falls back to another planner — the operator chose this one.

Environment:
  ROBOBALLET_POLICY_DIR    where policies live (default ``/models``)
  ROBOBALLET_POLICY        which one (a sub-directory; default ``current``)
  ROBOBALLET_HORIZON_S     simulated seconds a rollout may take (30)
  ROBOBALLET_MAX_DOCK_RAD  largest joint offset closed by docking (0.5)
  ROBOBALLET_TIME_BUDGET_S wall-clock budget per step (60 s). Its own, not
                           ompl-compound's 5 s per arm: every rollout step is held
                           to the planner's full collision envelope, so a rollout
                           that is still arriving would otherwise be cut off long
                           before its simulated horizon.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

BACKEND_NAME = "roboballet"
DEFAULT_POLICY_DIR = "/models"
DEFAULT_POLICY = "current"
DEFAULT_TIME_BUDGET_S = 60.0

_lock = threading.Lock()
_policies: dict = {}


def policy_dir() -> Path:
    return Path(os.environ.get("ROBOBALLET_POLICY_DIR") or DEFAULT_POLICY_DIR)


def load_policy(name: Optional[str] = None):
    """The named policy (``ROBOBALLET_POLICY``, else ``current``), cached per file version."""
    from agents.trajectory.backends.roboballet.core.policy_io import CARD, WEIGHTS, Policy, PolicyError
    name = name or os.environ.get("ROBOBALLET_POLICY") or DEFAULT_POLICY
    if "/" in name or name.startswith("."):
        raise PolicyError(f"policy name {name!r} is not a plain directory name")
    d = policy_dir() / name
    try:
        stamp = ((d / CARD).stat().st_mtime_ns, (d / WEIGHTS).stat().st_mtime_ns)
    except OSError:
        raise PolicyError(f"no policy {name!r} in {policy_dir()} — train one with the "
                          "policy-trainer service and import it") from None
    with _lock:
        hit = _policies.get(d)
        if hit and hit[0] == stamp:
            return hit[1]
        pol = Policy.load(d, name)
        _policies[d] = (stamp, pol)
        return pol


def check_options(options: dict) -> None:
    """Refuse a request whose ``policy`` this planner cannot run — at acceptance,
    not minutes later in every step of the run (``BackendSpec.check_options``)."""
    from agents.trajectory.backends.roboballet.core.policy_io import PolicyError
    name = options.get("policy")
    if name is None:
        return
    if not isinstance(name, str) or not name:
        raise ValueError("roboballet option 'policy' must be a policy name")
    try:
        load_policy(name)
    except PolicyError as e:
        raise ValueError(str(e)) from None


def _cell_model(cell_id: str, robot_ids: list[str]):
    """Chains, acceleration limits and obstacle primitives of the planning cell."""
    from agents.trajectory.backends.roboballet import cell_scene
    from agents.trajectory.backends.roboballet.core.chain import Chain
    from agents.trajectory.backends.roboballet.core.geometry import BoxSet, boxes_from_obstacles
    from agents.trajectory.tools.robot_kinematics import load_static_obstacles

    robots = cell_scene.cell_robots(cell_id)
    chains, accel = {}, {}
    for rid in robot_ids:
        if rid not in robots:
            continue
        spec = cell_scene.robot_spec(cell_id, rid, robots[rid])
        chains[rid] = Chain(spec["chain"])
        accel[rid] = spec["acc_limits"]
    import numpy as np
    centre = (np.mean([c.base[:3, 3] for c in chains.values()], axis=0).tolist()
              if chains else None)
    boxes = BoxSet(boxes_from_obstacles(load_static_obstacles(cell_id), centre))
    return chains, accel, boxes


def _refuse_all(motion_requests: list, error: str, t0: float) -> dict:
    return {"success": False, "trajectories": {},
            "failures": [{"robot_ns": r.get("robot_ns"), "operation_id": r.get("operation_id"),
                          "reason_code": "planner_error", "error": f"{BACKEND_NAME}: {error}"}
                         for r in motion_requests],
            "planning_time_s": round(time.monotonic() - t0, 3), "backend": BACKEND_NAME}


def plan(cell_id: str, motion_requests: list[dict], *, approach_offset_m: float = 0.1,
         velocity_scale: Optional[float] = None, policy: Optional[str] = None,
         time_budget_s: Optional[float] = None, **kwargs) -> dict:
    """Plan one step's motion requests with the trained policy (contract: ``RESULT_KEYS``)."""
    t0 = time.monotonic()
    from agents.trajectory.backends.roboballet.core.policy_io import PolicyError
    try:
        pol = load_policy(policy)
    except PolicyError as e:
        return _refuse_all(motion_requests, str(e), t0)
    except Exception as e:  # noqa: BLE001 — a broken file refuses, it does not crash the run
        return _refuse_all(motion_requests, f"policy could not be loaded: {e}", t0)

    robot_ids = sorted({r.get("robot_ns") for r in motion_requests if r.get("robot_ns")})
    try:
        chains, accel, boxes = _cell_model(cell_id, robot_ids)
    except Exception as e:  # noqa: BLE001 — "could not describe the cell" is not "no obstacles"
        return _refuse_all(motion_requests, f"cannot describe cell {cell_id!r} to the policy: {e}", t0)

    from agents.trajectory.backends.ompl_compound.planner import plan_trajectories_ompl_compound
    from agents.trajectory.backends.roboballet.rollout import make_transit_solver

    info: dict = {}
    solver = make_transit_solver(pol, chains, boxes, accel_limits=accel, info=info)
    budget = time_budget_s
    if budget is None:
        try:
            budget = float(os.environ.get("ROBOBALLET_TIME_BUDGET_S") or DEFAULT_TIME_BUDGET_S)
        except ValueError:
            budget = DEFAULT_TIME_BUDGET_S
    kwargs.pop("planner_name", None)
    result = plan_trajectories_ompl_compound(
        cell_id, motion_requests, approach_offset_m=approach_offset_m,
        velocity_scale=velocity_scale, planner_name=f"policy:{pol.name}",
        time_budget_s=budget, transit_solver=solver, backend_name=BACKEND_NAME, **kwargs)
    training = pol.card.get("training") or {}
    result["policy"] = {"name": pol.name, "job": training.get("job"),
                        "trained_on": training.get("cells"), **info}
    return result
