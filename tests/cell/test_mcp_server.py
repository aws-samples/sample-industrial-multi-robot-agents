"""Unit tests for agents/cell/mcp_server.py — MCP tool functions."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _stub_module(name: str, **attrs) -> MagicMock:
    """Return a MagicMock registered in sys.modules under *name*."""
    mod = MagicMock(spec=ModuleType(name))
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ── Stub every external dependency before importing mcp_server ───────────────

# rerun / ROS / messaging
for mod_name in (
    "rerun",
    "rclpy",
    "rclpy.node",
    "std_msgs",
    "std_msgs.msg",
):
    sys.modules.setdefault(mod_name, MagicMock())

# langchain / deepagents / mcp internals used by agent.py and tools
for mod_name in (
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.tools",
    "langchain_core.runnables",
    "langchain",
    "langchain.agents",
    "langchain.tools",
    "deepagents",
):
    sys.modules.setdefault(mod_name, MagicMock())

# Stub the cell agent module so we control _agent_execute_behaviour_tree
_fake_agent_mod = _stub_module("agents.cell.agent")
_fake_agent_mod.execute_behaviour_tree = MagicMock(return_value={"success": True})

# Stub tools used by mcp_server
_fake_cell_state_mod = _stub_module("agents.cell.tools.cell_state")
_fake_cell_state_mod.get_all_robot_states = MagicMock()
_fake_cell_state_mod.trigger_estop = MagicMock()

# Stub server helpers
_fake_server_mod = _stub_module("agents.cell.server")
_fake_server_mod.ros2_listener = MagicMock()
_fake_server_mod.http_server = MagicMock()
_fake_server_mod.ros2_joint_state_sub = MagicMock()

# ── Now import mcp_server (and the recorder it uses) ─────────────────────────
import agents.cell.mcp_server as _mcp_mod
from agents.cell.mcp_server import (
    execute_behaviour_tree,
    get_cell_status,
    trigger_estop,
    get_robot_states,
    list_recordings,
)


# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteBehaviourTree:

    def test_normal_mode_calls_agent(self):
        """execute_behaviour_tree(replay_mode=False) delegates to cell agent."""
        bt = {"tree_id": "bt_test_001", "root": {"type": "SEQUENCE", "children": []}}
        fake_result = {"success": True, "result": "done", "elapsed_s": 1.2}

        with patch.object(_mcp_mod, "_agent_execute_behaviour_tree", return_value=fake_result) as mock_agent, \
             patch.object(_mcp_mod, "get_recorder") as mock_get_recorder:
            mock_recorder = MagicMock()
            mock_get_recorder.return_value = mock_recorder

            result = execute_behaviour_tree(bt, replay_mode=False)

        mock_agent.assert_called_once_with(bt)
        mock_recorder.start_session.assert_called_once_with(tree_id="bt_test_001", cell_id=_mcp_mod._CELL_ID)
        mock_recorder.end_session.assert_called_once()
        assert result == fake_result

    def test_replay_mode_with_tree_id_calls_replay(self):
        """execute_behaviour_tree(replay_mode=True, tree_id=...) calls replay_execution."""
        bt = {"tree_id": "bt_test_001"}
        fake_result = {"success": True, "replayed_operations": 5, "elapsed_s": 0.8}

        with patch.object(_mcp_mod, "get_recorder") as mock_get_recorder:
            mock_recorder = MagicMock()
            mock_recorder.replay_execution.return_value = fake_result
            mock_get_recorder.return_value = mock_recorder

            result = execute_behaviour_tree(bt, replay_mode=True)

        mock_recorder.replay_execution.assert_called_once_with("bt_test_001", _mcp_mod._CELL_ID)
        assert result == fake_result

    def test_replay_mode_without_tree_id_returns_error(self):
        """execute_behaviour_tree(replay_mode=True, no tree_id) returns error dict."""
        bt = {"root": {"type": "SEQUENCE", "children": []}}  # no tree_id

        result = execute_behaviour_tree(bt, replay_mode=True)

        assert result["success"] is False
        assert "tree_id" in result["error"].lower()

    def test_normal_mode_end_session_called_on_exception(self):
        """end_session is called even when the agent raises an exception."""
        bt = {"tree_id": "bt_fail_001"}

        with patch.object(_mcp_mod, "_agent_execute_behaviour_tree", side_effect=RuntimeError("agent exploded")), \
             patch.object(_mcp_mod, "get_recorder") as mock_get_recorder:
            mock_recorder = MagicMock()
            mock_get_recorder.return_value = mock_recorder

            with pytest.raises(RuntimeError):
                execute_behaviour_tree(bt, replay_mode=False)

        mock_recorder.end_session.assert_called_once()


class TestToolDocstrings:

    def test_all_tools_have_docstrings(self):
        """All five MCP tools have non-empty docstrings for schema generation."""
        tools = [
            execute_behaviour_tree,
            get_cell_status,
            trigger_estop,
            get_robot_states,
            list_recordings,
        ]
        for tool_fn in tools:
            doc = tool_fn.__doc__ or ""
            assert len(doc.strip()) > 0, (
                f"Tool {tool_fn.__name__} has an empty docstring"
            )


class TestToolSignatures:

    def test_execute_behaviour_tree_has_replay_mode_param(self):
        """execute_behaviour_tree accepts replay_mode parameter."""
        import inspect
        sig = inspect.signature(execute_behaviour_tree)
        assert "replay_mode" in sig.parameters
        assert sig.parameters["replay_mode"].default is False

    def test_trigger_estop_has_required_params(self):
        """trigger_estop has cell_id and reason parameters."""
        import inspect
        sig = inspect.signature(trigger_estop)
        assert "cell_id" in sig.parameters
        assert "reason" in sig.parameters


class TestReplayPlanRefusesFailedRuns:
    """The plans directory holds failed runs too, so the executor has to check.

    A failed run is kept on purpose — one that solved thirteen welds before its
    token expired is worth reading and worth resuming — and it is kept in the same
    directory, in the same shape, as a good plan. ``replay_plan`` reads that
    directory straight off disk, so if it does not look at ``status`` then the
    partial tree the assembler REJECTED gets handed to the executor and real arms
    move through a plan nothing validated. Worse, ``bt = plan["behaviour_tree"] or
    plan`` means a record with no tree at all is passed on as though the failure
    report itself were a tree.

    A UI refusal is not a substitute: the tool is reachable by the orchestrator,
    by an agent, and by anything else holding the MCP endpoint.
    """

    @staticmethod
    def _plans(tmp_path, monkeypatch):
        monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
        d = tmp_path / "welding_cell_2_small" / "agentic_planner" / "plans"
        d.mkdir(parents=True)
        return d

    def _write(self, d, tree_id, status, tree=None, **extra):
        import json
        body = {"tree_id": tree_id, "cell_id": "welding_cell_2_small",
                "created_s": 1.0, "status": status, "behaviour_tree": tree}
        body.update(extra)
        (d / f"{tree_id}.json").write_text(json.dumps(body))

    def test_a_failed_run_named_explicitly_is_refused(self, tmp_path, monkeypatch):
        d = self._plans(tmp_path, monkeypatch)
        self._write(d, "bt_bad_welding_cell_2_small", "failed",
                    tree={"tree_id": "bt_bad_welding_cell_2_small",
                          "root": {"type": "Sequence", "children": []}},
                    error="ExpiredTokenException", error_kind="credentials",
                    num_planned=13, resumable=True)
        with patch.object(_mcp_mod, "_agent_execute_behaviour_tree") as executor, \
             patch.object(_mcp_mod, "get_recorder") as recorder:
            recorder.return_value = MagicMock()
            result = _mcp_mod.replay_plan("welding_cell_2_small",
                                          tree_id="bt_bad_welding_cell_2_small")
        assert result["success"] is False
        assert result["status"] == "failed"
        assert "not executable" in result["error"]
        assert "ExpiredTokenException" in result["error"]
        # The evidence an operator needs next: it can be continued, not run.
        assert result["resumable"] is True
        assert result["num_planned"] == 13
        # Nothing reached the executor, and no recording session was opened for a
        # run that never happened.
        executor.assert_not_called()
        recorder.return_value.start_session.assert_not_called()

    def test_the_latest_plan_never_resolves_to_a_failed_run(self, tmp_path,
                                                            monkeypatch):
        """The newest file is very often the one that just broke."""
        import os
        d = self._plans(tmp_path, monkeypatch)
        good = {"tree_id": "bt_good_welding_cell_2_small",
                "root": {"type": "Sequence", "children": []}}
        self._write(d, "bt_good_welding_cell_2_small", "ok", tree=good)
        self._write(d, "bt_bad_welding_cell_2_small", "failed", tree=None, error="boom")
        # Make the failure unambiguously the newest file.
        os.utime(d / "bt_good_welding_cell_2_small.json", (1000, 1000))
        os.utime(d / "bt_bad_welding_cell_2_small.json", (2000, 2000))
        with patch.object(_mcp_mod, "_agent_execute_behaviour_tree",
                          return_value={"success": True}) as executor, \
             patch.object(_mcp_mod, "get_recorder") as recorder:
            recorder.return_value = MagicMock()
            _mcp_mod.replay_plan("welding_cell_2_small", home_reset=False)
        assert executor.call_args[0][0]["tree_id"] == "bt_good_welding_cell_2_small"

    def test_a_run_that_never_assembled_a_tree_is_refused_not_executed(
            self, tmp_path, monkeypatch):
        """The `behaviour_tree or plan` fallback would pass the failure report on."""
        d = self._plans(tmp_path, monkeypatch)
        self._write(d, "bt_empty_welding_cell_2_small", "failed", tree=None,
                    error="ExpiredTokenException", resumable=False)
        with patch.object(_mcp_mod, "_agent_execute_behaviour_tree") as executor, \
             patch.object(_mcp_mod, "get_recorder") as recorder:
            recorder.return_value = MagicMock()
            result = _mcp_mod.replay_plan("welding_cell_2_small",
                                          tree_id="bt_empty_welding_cell_2_small")
        assert result["success"] is False
        executor.assert_not_called()

    def test_a_good_plan_still_replays(self, tmp_path, monkeypatch):
        """The guard must not cost the ordinary case."""
        d = self._plans(tmp_path, monkeypatch)
        self._write(d, "bt_good_welding_cell_2_small", "ok",
                    tree={"tree_id": "bt_good_welding_cell_2_small",
                          "root": {"type": "Sequence", "children": []}})
        with patch.object(_mcp_mod, "_agent_execute_behaviour_tree",
                          return_value={"success": True}) as executor, \
             patch.object(_mcp_mod, "get_recorder") as recorder:
            recorder.return_value = MagicMock()
            result = _mcp_mod.replay_plan("welding_cell_2_small",
                                          tree_id="bt_good_welding_cell_2_small",
                                          home_reset=False)
        assert result["success"] is True
        assert result["tree_id"] == "bt_good_welding_cell_2_small"
        executor.assert_called_once()

    def test_a_plan_saved_before_the_status_field_existed_still_replays(
            self, tmp_path, monkeypatch):
        """No status means it predates failed plans, when the store held only good ones."""
        import json
        d = self._plans(tmp_path, monkeypatch)
        (d / "bt_legacy_welding_cell_2_small.json").write_text(json.dumps({
            "tree_id": "bt_legacy_welding_cell_2_small", "cell_id": "welding_cell_2_small",
            "behaviour_tree": {"tree_id": "bt_legacy_welding_cell_2_small",
                               "root": {"type": "Sequence", "children": []}}}))
        with patch.object(_mcp_mod, "_agent_execute_behaviour_tree",
                          return_value={"success": True}) as executor, \
             patch.object(_mcp_mod, "get_recorder") as recorder:
            recorder.return_value = MagicMock()
            result = _mcp_mod.replay_plan("welding_cell_2_small",
                                          tree_id="bt_legacy_welding_cell_2_small",
                                          home_reset=False)
        assert result["success"] is True
        executor.assert_called_once()
