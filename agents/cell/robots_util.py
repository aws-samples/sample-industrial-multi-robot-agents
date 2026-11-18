"""
Helpers for resolving robot addressing from a cell's robots.yaml.

robots.yaml layout:
    robots:
      <namespace>:
        <robot_id>:
          { config }

Topics for a robot live under  /{cell_id}/{namespace}/{robot_id}/...

These helpers flatten the nested structure and work for both the shared-namespace
layout ({arms: {robot1: {..}, robot2: {..}}}) and the legacy layout
({robot1: {robot1: {..}}}).
"""
from __future__ import annotations

from utils import cell_layout, cell_store


def _robots_path(cell_id: str) -> str:
    """Where robots.yaml lives in the tree of whoever owns the cell's data."""
    return str(cell_layout.robots_yaml(cell_id))


def iter_robots(cell_id: str):
    """Yield (namespace, robot_id, config) for every robot in the cell.

    Read through :mod:`utils.cell_store`, so a process that owns ``/cells`` reads
    the file and one that does not (the trajectory planner) asks the agent that
    does. Same answer either way — which is the point, because the robot identities
    this yields are what every topic name in the cell is built from.
    """
    cfg = cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG,
                               cell_layout.robots_yaml(cell_id).name) or {}
    for ns, ns_block in (cfg.get("robots") or {}).items():
        if isinstance(ns_block, dict):
            for robot_id, rcfg in ns_block.items():
                if isinstance(rcfg, dict):
                    yield ns, robot_id, rcfg


def robot_ids(cell_id: str) -> list[str]:
    """List of unique robot identities in the cell."""
    return [rid for _ns, rid, _cfg in iter_robots(cell_id)]


def robot_topic_prefix(cell_id: str, robot_id: str) -> str:
    """
    Topic prefix '{cell_id}/{namespace}/{robot_id}' for a robot identity.

    Falls back to '{cell_id}/{robot_id}/{robot_id}' if the robot is not found,
    matching the legacy single-robot-per-namespace convention.
    """
    for ns, rid, _cfg in iter_robots(cell_id):
        if rid == robot_id:
            return f"{cell_id}/{ns}/{rid}"
    return f"{cell_id}/{robot_id}/{robot_id}"
