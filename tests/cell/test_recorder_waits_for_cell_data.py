"""The cell controller's joint-state recorder waits for the cell-data owner.

With planning and execution started separately the execution side is often up
first. One failed read of robots.yaml used to end the recorder thread for good
("cell-data MCP http://orchestrator:8893/mcp is unreachable"), so it stayed dead
after the orchestrator came up. It now retries until the owner answers.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_the_recorder_retries_an_unreachable_owner(monkeypatch):
    # Loaded from its file, not imported: tests/cell/test_mcp_server.py leaves a
    # stub registered as agents.cell.server for the rest of the session.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_cell_server_under_test", ROOT / "agents/cell/server.py")
    server = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(server)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cell server not importable here: {exc}")
    from utils import cell_store

    # The thread imports ROS before it reads the robots; stub it so the test is
    # about the read, whichever ROS (real, stubbed, none) this host has.
    for name in ("rclpy", "rclpy.node", "rclpy.executors", "std_msgs", "std_msgs.msg"):
        mod = types.ModuleType(name)
        mod.Node = mod.SingleThreadedExecutor = mod.String = object
        monkeypatch.setitem(sys.modules, name, mod)

    calls = {"n": 0}

    def iter_robots(cell_id):
        calls["n"] += 1
        if calls["n"] < 3:
            raise cell_store.CellStoreUnavailable("cell-data MCP x is unreachable")
        return []                       # reachable, no robots: the thread returns

    # Every module the thread imports is pinned here, so whatever another test
    # left in sys.modules, this one is about the read and nothing else.
    fake_utils = types.ModuleType("agents.cell.robots_util")
    fake_utils.iter_robots = iter_robots
    fake_rec = types.ModuleType("agents.cell.rerun_recorder")
    fake_rec.get_recorder = lambda: None
    monkeypatch.setitem(sys.modules, "agents.cell.robots_util", fake_utils)
    monkeypatch.setitem(sys.modules, "agents.cell.rerun_recorder", fake_rec)
    monkeypatch.setitem(sys.modules, "utils.cell_store", cell_store)
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    server.ros2_joint_state_sub()
    assert calls["n"] == 3
