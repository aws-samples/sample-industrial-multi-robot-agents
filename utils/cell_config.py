"""
Cell configuration loader — parse robots.yaml and operations.yaml.

Provides typed access to cell spec for agents and scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from utils import cell_layout


@dataclass
class RobotConfig:
    namespace:             str
    base_position:         list[float]
    base_orientation_yaw:  float
    workspace_radius:      float
    joint_limits_lower:    list[float]
    joint_limits_upper:    list[float]
    end_effector_link:     str = "tool0"
    robot_model:           str = ""


@dataclass
class OperationPoint:
    id:               str
    type:             str               # weld | rivet | glue | laser_cut | mark | move
    position:         list[float]       # [x, y, z]
    approach_vector:  list[float]       # [ax, ay, az]
    face:             str = ""
    sequence_group:   str = ""
    priority:         int = 1


@dataclass
class CellConfig:
    cell_id:    str
    robots:     dict[str, RobotConfig] = field(default_factory=dict)
    operations: list[OperationPoint]   = field(default_factory=list)


def load_cell_config(cell_id: str, cell_base: str | None = None) -> CellConfig:
    """
    Load cell configuration from the cell's ``config/`` (see cell_layout).

    Args:
        cell_id:   Cell identifier
        cell_base: Base path override (default: CELL_BASE_PATH env or /cells)

    Returns:
        CellConfig with parsed robots and operations.
    """
    cfg = CellConfig(cell_id=cell_id)

    # Robots
    robots_path = cell_layout.robots_yaml(cell_id, cell_base)
    with open(robots_path) as f:
        robots_raw = yaml.safe_load(f)

    for ns, ns_block in robots_raw.get("robots", {}).items():
        if not isinstance(ns_block, dict):
            continue
        for _, rcfg in ns_block.items():
            if not isinstance(rcfg, dict):
                continue
            cfg.robots[ns] = RobotConfig(
                namespace=ns,
                base_position=rcfg["base_position"],
                base_orientation_yaw=float(rcfg.get("base_orientation_yaw", 0.0)),
                workspace_radius=float(rcfg.get("workspace_radius", 2.1)),
                joint_limits_lower=rcfg.get("joint_limits_lower", [-3.14]*6),
                joint_limits_upper=rcfg.get("joint_limits_upper", [ 3.14]*6),
                end_effector_link=rcfg.get("end_effector_link", "tool0"),
                robot_model=rcfg.get("robot_model", ""),
            )
            break

    # Operations
    ops_path = cell_layout.operations_yaml(cell_id, cell_base)
    if Path(ops_path).exists():
        with open(ops_path) as f:
            ops_raw = yaml.safe_load(f) or {}
        for op in ops_raw.get("operations", []):
            cfg.operations.append(OperationPoint(
                id=op["id"],
                type=op.get("type", "weld"),
                position=op["position"],
                approach_vector=op.get("approach_vector", [0, 0, -1]),
                face=op.get("face", ""),
                sequence_group=op.get("sequence_group", ""),
                priority=int(op.get("priority", 1)),
            ))

    return cfg
