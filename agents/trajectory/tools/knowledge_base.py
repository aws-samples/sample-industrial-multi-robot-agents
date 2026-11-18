"""
Trajectory Knowledge Base tools.

Stores successful trajectories keyed by (cell_id, operation_type, robot_ns,
rounded_target) and retrieves warm-start joint configs to seed Pinocchio IK.

Storage: in the cell's ``recordings`` kind, beside the .rrd files it indexes,
read and written through :mod:`utils.cell_store` — so the warm-start index the
planner learns from is kept by the agent that owns the cell's data, and a
replaced planner container inherits everything the previous one learned.

* ``trajectory_index.json`` — the index as a whole (``_save_index``);
* ``trajectory_index.jsonl`` — one line per stored trajectory, APPENDED.

``store_trajectory`` appends instead of reading, editing and rewriting the whole
index: two runs storing at once (two planners, or a plan and a plan edit) each
read the old index and the second write erased the first one's entries. An
append is one atomic call in both store modes. Readers fold the JSON and then
the lines, a later line winning for the same key.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from utils import cell_store

_INDEX_NAME = "trajectory_index.json"
_ENTRIES_NAME = "trajectory_index.jsonl"


def _load_index(cell_id: str) -> dict:
    import json
    index = cell_store.read_json(cell_id, cell_store.KIND_RECORDINGS, _INDEX_NAME) or {}
    text = cell_store.read_text(cell_id, cell_store.KIND_RECORDINGS, _ENTRIES_NAME) or ""
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue                          # a torn last line is skipped, not fatal
        if isinstance(entry, dict) and isinstance(entry.get("key"), str):
            index[entry["key"]] = entry.get("value")
    return index


def _append_entry(cell_id: str, key: str, value: dict) -> None:
    import json
    cell_store.append_text(cell_id, cell_store.KIND_RECORDINGS, _ENTRIES_NAME,
                           json.dumps({"key": key, "value": value}) + "\n")


def _save_index(cell_id: str, index: dict) -> None:
    cell_store.write_json(cell_id, cell_store.KIND_RECORDINGS, _INDEX_NAME,
                          index, indent=2)


def _make_key(operation_type: str, robot_ns: str, target: list[float]) -> str:
    tx, ty, tz = round(target[0], 2), round(target[1], 2), round(target[2], 2)
    return f"{operation_type}|{robot_ns}|{tx}|{ty}|{tz}"


@tool
def store_trajectory(
    cell_id: str,
    operation_type: str,
    robot_ns: str,
    target: list[float],
    final_joints: list[float],
    position_error_m: float,
    recording_file: Optional[str] = None,
) -> dict:
    """
    Store a successful trajectory in the knowledge base for future warm-starts.

    Called after each COMPLETED batch to build up the KB.

    Args:
        cell_id:          Cell identifier
        operation_type:   "weld" | "rivet" etc.
        robot_ns:         Robot namespace
        target:           [x, y, z] final target
        final_joints:     [j1..j6] joint config at target
        position_error_m: Achieved position accuracy
        recording_file:   Optional path to full recording JSON

    Returns:
        {"stored": true, "key": str}
    """
    key = _make_key(operation_type, robot_ns, target)
    _append_entry(cell_id, key, {
        "final_joints":     final_joints,
        "position_error_m": position_error_m,
        "recording_file":   recording_file,
    })
    return {"stored": True, "key": key}


@tool
def get_warm_start(
    cell_id: str,
    operation_type: str,
    robot_ns: str,
    target: list[float],
) -> dict:
    """
    Get warm-start joint configuration from the knowledge base.

    Looks for an exact match, then searches nearby targets within 0.05m.
    Returns the closest match's final_joints to seed Pinocchio IK.

    Args:
        cell_id:        Cell identifier
        operation_type: "weld" | "rivet" etc.
        robot_ns:       Robot namespace
        target:         [x, y, z] target position

    Returns:
        {"found": true, "joint_positions": [j1..j6], "position_error_m": float}
        or {"found": false}
    """
    import numpy as np

    index = _load_index(cell_id)
    key   = _make_key(operation_type, robot_ns, target)

    # Exact match
    if key in index:
        entry = index[key]
        return {
            "found":            True,
            "joint_positions":  entry["final_joints"],
            "position_error_m": entry.get("position_error_m", 0.0),
            "match_type":       "exact",
        }

    # Nearest match within 0.05m
    t = np.array(target[:3])
    best_key, best_dist = None, 0.05

    for k, v in index.items():
        parts = k.split("|")
        if len(parts) != 5:
            continue
        op, rns, tx, ty, tz = parts
        if op != operation_type or rns != robot_ns:
            continue
        dist = float(np.linalg.norm(t - np.array([float(tx), float(ty), float(tz)])))
        if dist < best_dist:
            best_dist, best_key = dist, k

    if best_key:
        entry = index[best_key]
        return {
            "found":            True,
            "joint_positions":  entry["final_joints"],
            "position_error_m": entry.get("position_error_m", 0.0),
            "match_type":       "nearby",
            "distance_m":       round(best_dist, 4),
        }

    return {"found": False}


@tool
def get_similar_trajectories(
    cell_id: str,
    operation_type: str,
    robot_ns: str,
    target: list[float],
    radius_m: float = 0.2,
    max_results: int = 5,
) -> dict:
    """
    Get all similar trajectories within a radius for LLM context.

    The trajectory agent's LLM can use these to understand the robot's
    typical configurations near this region of the workspace.

    Args:
        cell_id:        Cell identifier
        operation_type: "weld" | "rivet" etc.
        robot_ns:       Robot namespace
        target:         [x, y, z] centre of search
        radius_m:       Search radius (default 0.2m)
        max_results:    Maximum entries to return

    Returns:
        {"results": [{"target": [...], "joints": [...], "error_m": float}]}
    """
    import numpy as np

    index = _load_index(cell_id)
    t     = np.array(target[:3])
    results = []

    for k, v in index.items():
        parts = k.split("|")
        if len(parts) != 5:
            continue
        op, rns, tx, ty, tz = parts
        if op != operation_type or rns != robot_ns:
            continue
        dist = float(np.linalg.norm(t - np.array([float(tx), float(ty), float(tz)])))
        if dist <= radius_m:
            results.append({
                "target":    [float(tx), float(ty), float(tz)],
                "joints":    v["final_joints"],
                "error_m":   v.get("position_error_m", 0.0),
                "distance_m": round(dist, 4),
            })

    results.sort(key=lambda r: r["distance_m"])
    return {"results": results[:max_results]}
