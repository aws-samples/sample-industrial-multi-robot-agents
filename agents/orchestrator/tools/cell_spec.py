"""
Cell spec tools — read cell configuration and publish to ROS 2 topics.

The orchestrator agent uses these to:
  1. Load cell definition (robots, operations, task programs)
  2. Publish robot_description to /cell_id/robot_description
  3. Publish scene URDF/SRDF to the parameter server
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.tools import tool

from utils import cell_layout


@tool
def read_cell_spec(cell_id: str) -> dict:
    """
    Read cell specification: robots, operation programs, workpiece bounds.

    Args:
        cell_id: Cell identifier (e.g. "welding_cell_2_small")

    Returns:
        {"cell_id": str, "robots": {...}, "programs": [...], "workpiece": {...}}
    """
    result: dict = {"cell_id": cell_id}

    # Robots
    with open(cell_layout.robots_yaml(cell_id)) as f:
        robots_cfg = yaml.safe_load(f)
    result["robots"] = robots_cfg.get("robots", {})

    # Programs (task JSON files)
    tasks_dir = cell_layout.tasks_dir(cell_id)
    programs = []
    if tasks_dir.exists():
        for p in sorted(tasks_dir.glob("*.json")):
            programs.append(p.stem)
    result["programs"] = programs

    # Operations summary — derived from task JSON files (not a static yaml)
    op_types: dict = {}
    total_ops = 0
    if tasks_dir.exists():
        for p in sorted(tasks_dir.glob("*.json")):
            try:
                with open(p) as f:
                    task_data = json.load(f)
                for op in task_data.get("operations", []):
                    t = op.get("type", "weld")
                    op_types[t] = op_types.get(t, 0) + 1
                    total_ops += 1
            except Exception:
                pass
    if op_types:
        result["operations"] = {"total": total_ops, "by_type": op_types}

    return result


@tool
def read_task_program(cell_id: str, program_name: str) -> dict:
    """
    Load a task program JSON from the cell's agent-owned ``tasks/`` directory.

    Task program format:
      {"program": str, "description": str, "operations": [...]}

    Args:
        cell_id:      Cell identifier
        program_name: Program name without .json extension

    Returns:
        The task program dict.
    """
    with open(cell_layout.task_file(cell_id, program_name)) as f:
        return json.load(f)


@tool
def publish_cell_description(cell_id: str) -> dict:
    """
    Report which robot descriptions (URDF) of the cell are available to consumers.

    Nothing is published on ROS 2 any more: the ``/<cell>/<ns>/robot_description``
    topic this used to write had no subscriber (the RViz viewer builds its own
    view), and a planning-only stack has no ROS graph at all. Consumers fetch a
    description from the trajectory agent's MCP (``get_robot_description``).

    Args:
        cell_id: Cell identifier

    Returns:
        {"available": [robot_id, ...], "missing": [robot_id, ...],
         "served_by": "trajectory-agent MCP get_robot_description"}
    """
    try:
        with open(cell_layout.robots_yaml(cell_id)) as f:
            robots_cfg = yaml.safe_load(f) or {}
        available, missing = [], []
        for ns, block in (robots_cfg.get("robots") or {}).items():
            ids = [rid for rid, c in block.items() if isinstance(c, dict)] \
                if isinstance(block, dict) else []
            for rid in ids or [ns]:
                (available if cell_layout.robot_urdf(cell_id, rid) else missing).append(rid)
        return {"available": available, "missing": missing,
                "served_by": "trajectory-agent MCP get_robot_description"}
    except Exception as e:
        return {"error": str(e)}
