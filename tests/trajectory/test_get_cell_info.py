"""Unit tests for get_cell_info's robot enumeration (agents/trajectory/mcp_server.py).

Regression guard: cells whose arms share one ROS namespace
({arms: {robot1: .., robot2: ..}}) must report EVERY arm, not just the first.
Enumeration is delegated to agents.cell.robots_util.iter_robots, which is what
these tests exercise through the same nesting shapes get_cell_info must handle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.cell.robots_util import iter_robots  # noqa: E402

SHARED_NS = """
robots:
  arms:
    robot1:
      base_position: [0.07, -3.32, 0]
      workspace_radius: 2.9
      end_effector_link: welding_tcp
    robot2:
      base_position: [-1.18, 1.04, 0]
      workspace_radius: 2.9
      end_effector_link: welding_tcp
"""

LEGACY_NS = """
robots:
  robot1:
    robot1:
      base_position: [0.0, 0.0, 0]
      end_effector_link: tool0
  robot2:
    robot2:
      base_position: [1.0, 0.0, 0]
      end_effector_link: tool0
"""


def _write_cell(tmp_path: Path, cell_id: str, robots_yaml: str) -> None:
    cfg_dir = tmp_path / cell_id / "config"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "robots.yaml").write_text(robots_yaml)


def _robots_info(cell_id: str) -> dict:
    """Mirror get_cell_info's robots_info construction."""
    return {
        rid: {
            "namespace": ns,
            "base_position": rcfg.get("base_position"),
            "workspace_radius": rcfg.get("workspace_radius", 2.1),
            "end_effector": rcfg.get("end_effector_link", "tool0"),
        }
        for ns, rid, rcfg in iter_robots(cell_id)
    }


def test_shared_namespace_reports_both_arms(tmp_path, monkeypatch):
    """Two arms under one namespace must both appear (the fixed bug)."""
    _write_cell(tmp_path, "welding_cell_2_small", SHARED_NS)
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))

    info = _robots_info("welding_cell_2_small")

    assert len(info) == 2, f"expected both arms, got {list(info)}"
    assert set(info) == {"robot1", "robot2"}
    # Keyed by robot identity, and each arm keeps its OWN base position.
    assert info["robot1"]["base_position"] == [0.07, -3.32, 0]
    assert info["robot2"]["base_position"] == [-1.18, 1.04, 0]
    # Namespace is preserved for topic addressing.
    assert info["robot1"]["namespace"] == "arms"
    assert info["robot2"]["namespace"] == "arms"


def test_legacy_one_robot_per_namespace_still_works(tmp_path, monkeypatch):
    """The legacy {robot1: {robot1: ..}} layout must keep working."""
    _write_cell(tmp_path, "legacy_cell", LEGACY_NS)
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))

    info = _robots_info("legacy_cell")

    assert set(info) == {"robot1", "robot2"}
    assert info["robot1"]["namespace"] == "robot1"


def test_defaults_applied_when_fields_absent(tmp_path, monkeypatch):
    """Missing workspace_radius / end_effector_link fall back to defaults."""
    _write_cell(tmp_path, "sparse_cell", """
robots:
  arms:
    robot1:
      base_position: [0, 0, 0]
""")
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))

    info = _robots_info("sparse_cell")

    assert info["robot1"]["workspace_radius"] == 2.1
    assert info["robot1"]["end_effector"] == "tool0"


def test_robot_keys_are_valid_reachability_arguments(tmp_path, monkeypatch):
    """get_cell_info's robot keys must be usable as check_reachability(robot_ns=..).

    check_reachability resolves robot_ns against the INNER key, so get_cell_info
    keying by namespace (the old behaviour) produced unusable identifiers for
    shared-namespace cells.
    """
    _write_cell(tmp_path, "welding_cell_2_small", SHARED_NS)
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))

    keys = set(_robots_info("welding_cell_2_small"))
    inner_keys = {rid for _ns, rid, _cfg in iter_robots("welding_cell_2_small")}

    assert keys == inner_keys
    assert "arms" not in keys
