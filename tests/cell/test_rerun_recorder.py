"""Unit tests for RerunRecorder — session lifecycle, logging, and error handling."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Pre-register rerun mock so _RERUN_AVAILABLE = True in the module
rr_mock = MagicMock()
sys.modules["rerun"] = rr_mock

import importlib
import agents.cell.rerun_recorder as _mod
importlib.reload(_mod)
from agents.cell.rerun_recorder import RerunRecorder, NoOpRecorder, _sanitise_filename


# ─── Session lifecycle ────────────────────────────────────────────────────────

class TestStartSession:

    def test_calls_rr_init_with_correct_path(self, tmp_path):
        """start_session calls rr.init(safe_name, save_to=expected_path)."""
        os.environ["CELL_BASE_PATH"] = str(tmp_path)
        recorder = RerunRecorder()

        with patch.object(_mod, "_RERUN_AVAILABLE", True), \
             patch.object(_mod, "rr") as mock_rr:
            mock_rr.init = MagicMock()
            recorder.start_session(tree_id="bt_test_42", cell_id="welding_cell_2_small")

        expected = str(tmp_path / "welding_cell_2_small" / "recordings" / "bt_test_42.rrd")
        assert recorder._save_path == expected
        assert recorder._active is True

    def test_creates_recordings_directory(self, tmp_path):
        """start_session creates the recordings dir if it doesn't exist."""
        os.environ["CELL_BASE_PATH"] = str(tmp_path)
        recorder = RerunRecorder()
        recordings_dir = tmp_path / "welding_cell_2_small" / "recordings"
        assert not recordings_dir.exists()

        with patch.object(_mod, "_RERUN_AVAILABLE", True), \
             patch.object(_mod, "rr") as mock_rr:
            mock_rr.init = MagicMock()
            recorder.start_session(tree_id="session1", cell_id="welding_cell_2_small")

        assert recordings_dir.exists()

    def test_warns_on_existing_file(self, tmp_path, capsys):
        """start_session prints a warning when the .rrd file already exists."""
        os.environ["CELL_BASE_PATH"] = str(tmp_path)
        rrd_path = tmp_path / "welding_cell_2_small" / "recordings" / "bt_existing.rrd"
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        rrd_path.write_bytes(b"existing")

        recorder = RerunRecorder()
        with patch.object(_mod, "_RERUN_AVAILABLE", True), \
             patch.object(_mod, "rr") as mock_rr:
            mock_rr.init = MagicMock()
            recorder.start_session(tree_id="bt_existing", cell_id="welding_cell_2_small")

        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "overwriting" in captured.out.lower()

    def test_noop_when_rerun_unavailable(self):
        """start_session is a no-op when _RERUN_AVAILABLE is False."""
        recorder = RerunRecorder()
        with patch.object(_mod, "_RERUN_AVAILABLE", False):
            recorder.start_session(tree_id="bt_test", cell_id="test_cell")
        assert recorder._active is False


class TestEndSession:

    def test_calls_rr_disconnect(self, tmp_path):
        """end_session calls rr.disconnect() and resets state."""
        os.environ["CELL_BASE_PATH"] = str(tmp_path)
        recorder = RerunRecorder()
        recorder._active = True
        recorder._save_path = str(tmp_path / "test.rrd")

        with patch.object(_mod, "_RERUN_AVAILABLE", True), \
             patch.object(_mod, "rr") as mock_rr:
            mock_rr.disconnect = MagicMock()
            recorder.end_session()

        mock_rr.disconnect.assert_called_once()
        assert recorder._active is False
        assert recorder._save_path is None

    def test_noop_when_not_active(self):
        """end_session is a no-op when session is not active."""
        recorder = RerunRecorder()
        with patch.object(_mod, "_RERUN_AVAILABLE", True), \
             patch.object(_mod, "rr") as mock_rr:
            mock_rr.disconnect = MagicMock()
            recorder.end_session()  # not active
        mock_rr.disconnect.assert_not_called()


# ─── log_joint_state ─────────────────────────────────────────────────────────

class TestLogJointState:

    def test_uses_provided_timestamp(self):
        """log_joint_state uses the provided timestamp for rr.set_time_seconds."""
        recorder = RerunRecorder()
        recorder._active = True
        ts = 1718000000.5

        with patch.object(_mod, "_RERUN_AVAILABLE", True), \
             patch.object(_mod, "rr") as mock_rr:
            mock_rr.set_time_seconds = MagicMock()
            mock_rr.SeriesLine = MagicMock(return_value=MagicMock())
            mock_rr.Points3D = MagicMock(return_value=MagicMock())
            mock_rr.log = MagicMock()
            recorder.log_joint_state("robot1", [0.1]*6, [1.0, 0.0, 1.5], timestamp=ts)

        mock_rr.set_time_seconds.assert_called_with("execution", ts)

    def test_falls_back_to_time_when_timestamp_is_none(self):
        """log_joint_state falls back to time.time() when timestamp is None."""
        recorder = RerunRecorder()
        recorder._active = True

        with patch.object(_mod, "_RERUN_AVAILABLE", True), \
             patch.object(_mod, "rr") as mock_rr, \
             patch.object(_mod, "time") as mock_time:
            mock_time.time.return_value = 999.0
            mock_rr.set_time_seconds = MagicMock()
            mock_rr.SeriesLine = MagicMock(return_value=MagicMock())
            mock_rr.Points3D = MagicMock(return_value=MagicMock())
            mock_rr.log = MagicMock()
            recorder.log_joint_state("robot1", [0.1]*6, [1.0, 0.0, 1.5], timestamp=None)

        mock_rr.set_time_seconds.assert_called_with("execution", 999.0)


# ─── replay_execution error cases ────────────────────────────────────────────

class TestReplayExecution:

    def test_returns_not_found_when_rrd_missing(self, tmp_path):
        """replay_execution returns error dict when .rrd file does not exist."""
        os.environ["CELL_BASE_PATH"] = str(tmp_path)
        recorder = RerunRecorder()

        result = recorder.replay_execution("nonexistent_tree", "welding_cell_2_small")

        assert result["success"] is False
        assert result["error"] == "recording not found"
        assert result["tree_id"] == "nonexistent_tree"


# ─── NoOpRecorder ────────────────────────────────────────────────────────────

class TestNoOpRecorder:

    def test_all_methods_return_without_error(self):
        """NoOpRecorder methods return immediately without raising."""
        noop = NoOpRecorder()
        noop.start_session("test", "cell")
        noop.log_joint_state("robot1", [0.0]*6, [0.0, 0.0, 0.0])
        noop.log_dispatch_event("W001", "robot1", "weld", [], "dispatched")
        noop.end_session()
        result = noop.replay_execution("tree1", "cell1")
        assert result["success"] is False
        result2 = noop.list_recordings("cell1")
        assert "recordings" in result2
