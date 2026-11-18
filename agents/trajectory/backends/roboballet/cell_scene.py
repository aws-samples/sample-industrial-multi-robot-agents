"""A cell, in the existing cell format, as a RoboBallet **scene pack**.

The same function feeds both sides, so training and planning read one cell the
same way:

* the policy trainer exports the TRAINING cells' packs and ships them to
  SageMaker, where the procedural generator varies them (robot placement,
  obstacle sets, task sets);
* the backend builds the PLANNING cell's chains from it at plan time, so the
  graph features are computed from the same envelope, base pose and TCP the
  training cells were described with.

Nothing new is asked of a cell. Everything comes from what every backend reads:

==============================  ============================================
``robots.yaml`` (cell store)    base pose, joint/velocity/acceleration
                                limits, HOME, ``tip_link`` + ``tcp_offset``,
                                the fitted ``link_collision`` envelope
the arm's URDF (scene bundle)   the kinematic chain (``cell_layout.robot_urdf``)
``obstacles.yaml`` (cell store) boxes / spheres / planes → oriented boxes
``tasks/*.json`` (cell store)   weld points → task poses (training only)
==============================  ============================================

A scene pack is plain JSON; its ``schema_version`` is the graph schema's.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from agents.trajectory.backends.roboballet.core import graph as G
from agents.trajectory.backends.roboballet.core.chain import quat_matrix, parse_urdf
from agents.trajectory.backends.roboballet.core.geometry import boxes_from_obstacles

#: Envelope spheres kept per arm in a pack. The fitted envelope is hundreds of
#: spheres; the policy's features and the training simulator need its shape, not
#: its every vertex. The planner's own checker keeps the full envelope.
MAX_SPHERES_PER_ARM = 48

DEFAULT_ACCEL = 3.0


def _robots_yaml(cell_id: str) -> dict:
    from utils import cell_layout, cell_store
    cfg = cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG, cell_layout.robots_yaml(cell_id).name)
    if not cfg:
        raise FileNotFoundError(f"no robots.yaml for cell {cell_id!r}")
    return cfg


def cell_robots(cell_id: str, robots_cfg: Optional[dict] = None) -> dict[str, dict]:
    """``{robot_id: config}`` — both robots.yaml layouts (as ``planning_tools._cell_robots``)."""
    cfg = robots_cfg if robots_cfg is not None else _robots_yaml(cell_id)
    out: dict[str, dict] = {}
    for block in (cfg.get("robots") or {}).values():
        if isinstance(block, dict):
            out.update({rid: rc for rid, rc in block.items() if isinstance(rc, dict)})
    return out


def home_of(rcfg: dict) -> list[float]:
    """HOME in radians, by the rules of ``agents.home_pose.home_pose``."""
    if rcfg.get("home_pose_deg"):
        return [math.radians(float(v)) for v in rcfg["home_pose_deg"]]
    rad = rcfg.get("home_pose") or rcfg.get("home_joints")
    if rad:
        return [float(v) for v in rad]
    from agents.home_pose import HOME_JOINTS_FALLBACK
    return list(HOME_JOINTS_FALLBACK)


def _subsample(points: dict, budget: int) -> dict:
    """Keep about ``budget`` envelope spheres in total, spread over every link."""
    total = sum(len(p) for p in points.values())
    if total <= budget:
        return points
    out = {}
    for link, pts in points.items():
        n = max(1, round(len(pts) * budget / total))
        idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
        out[link] = [list(map(float, pts[i])) for i in sorted(set(idx.tolist()))]
    return out


def robot_spec(cell_id: str, robot_id: str, rcfg: dict, *,
               max_spheres: int = MAX_SPHERES_PER_ARM) -> dict:
    from utils import cell_layout
    urdf = cell_layout.robot_urdf(cell_id, robot_id)
    if urdf is None:
        raise FileNotFoundError(f"cell {cell_id!r} has no URDF for {robot_id!r}")
    joints = parse_urdf(urdf.read_text(), rcfg.get("tip_link") or "")
    lower, upper = rcfg.get("joint_limits_lower"), rcfg.get("joint_limits_upper")
    movable = [j for j in joints if j["type"] in ("revolute", "continuous", "prismatic")]
    if lower and upper and len(lower) == len(movable) == len(upper):
        # robots.yaml is the cell's word on the joint box (it may be tighter than the URDF).
        for j, lo, hi in zip(movable, lower, upper):
            j["lower"], j["upper"] = max(j["lower"], float(lo)), min(j["upper"], float(hi))
    lc = rcfg.get("link_collision") or {}
    spheres = {k: v for k, v in (lc.get("links") or {}).items() if v}
    dof = len(movable)
    if dof > G.MAX_DOF:
        raise ValueError(f"{robot_id} has {dof} joints; the policy graph supports {G.MAX_DOF}")
    vel = rcfg.get("joint_velocity_limits") or [1.0] * dof
    acc = rcfg.get("joint_acceleration_limits") or [DEFAULT_ACCEL] * dof
    return {
        "robot_id": robot_id,
        "chain": {"joints": joints, "base": [float(v) for v in rcfg.get("base_position", (0, 0, 0))],
                  "yaw": float(rcfg.get("base_orientation_yaw", 0.0)),
                  "tcp_offset": [float(v) for v in (rcfg.get("tcp_offset") or (0, 0, 0))][:3],
                  "link_spheres": _subsample(spheres, max_spheres),
                  "sphere_radius": float(lc.get("radius_m", 0.12))},
        "vel_limits": [float(v) for v in vel][:dof],
        "acc_limits": [float(v) for v in acc][:dof],
        "home": home_of(rcfg)[:dof],
    }


def _task_rotation(op: dict) -> list:
    ori = op.get("weld_orientation") or op.get("orientation")
    if isinstance(ori, dict):
        return quat_matrix(ori).tolist()
    return np.eye(3).tolist()


def operations(cell_id: str) -> list[dict]:
    """Every operation point of the cell's task programs, as task poses."""
    from utils import cell_store
    out, seen = [], set()
    try:
        entries = cell_store.list_entries(cell_id, cell_store.KIND_TASKS, suffix=".json")
    except Exception:  # noqa: BLE001 — a cell may have no task programs
        return []
    for e in entries:
        prog = cell_store.read_json(cell_id, cell_store.KIND_TASKS, e["name"]) or {}
        for op in prog.get("operations") or []:
            pos = op.get("position")
            if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
                continue
            key = (e["name"], op.get("id"))
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": f"{prog.get('program', e['name'])}:{op.get('id')}",
                        "position": [float(v) for v in pos],
                        "rotation": _task_rotation(op),
                        "approach_vector": [float(v) for v in op.get("approach_vector") or (0, 0, -1)]})
    return out


def scene_pack(cell_id: str, *, max_spheres: int = MAX_SPHERES_PER_ARM,
               with_operations: bool = True) -> dict:
    """The whole cell as one JSON document (see the module docstring)."""
    from utils import cell_layout, cell_store
    rcfg_all = _robots_yaml(cell_id)
    robots = [robot_spec(cell_id, rid, rc, max_spheres=max_spheres)
              for rid, rc in sorted(cell_robots(cell_id, rcfg_all).items())]
    if not robots:
        raise ValueError(f"cell {cell_id!r} declares no robots")
    centre = np.mean([r["chain"]["base"] for r in robots], axis=0).tolist()
    obs_cfg = cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG,
                                   cell_layout.obstacles_yaml(cell_id).name) or {}
    collision = (rcfg_all.get("collision") or {})
    return {
        "schema_version": G.SCHEMA_VERSION,
        "cell_id": cell_id,
        "robots": robots,
        "obstacles": boxes_from_obstacles(obs_cfg.get("obstacles") or [], centre),
        "operations": operations(cell_id) if with_operations else [],
        "warning_distance_m": float(collision.get("warning_distance", 0.20)),
    }
