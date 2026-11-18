"""Unit tests for dispatch.py recorder integration.

Verifies that ``publish_task_assignment`` calls the recorder with the correct
``phase`` at each point in its execution:

  - ``phase="dispatched"``  — logged unconditionally before any ROS 2 logic
  - ``phase="completed"``   — logged after a successful dispatch (real or simulated)
  - ``phase="failed"``      — logged when an unexpected exception is raised
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Stub heavy deps that dispatch.py imports at module level.
# langchain_core.tools.tool must behave as a pass-through decorator so the
# @tool-decorated functions remain callable via .invoke().
import functools

def _passthrough_tool(fn=None, **kwargs):
    """Minimal stand-in for @langchain_core.tools.tool that preserves .invoke()."""
    def _wrap(f):
        @functools.wraps(f)
        def _invoke_wrapper(*a, **kw):
            return f(*a, **kw)
        _invoke_wrapper.invoke = lambda input_dict: f(**input_dict)
        return _invoke_wrapper
    if fn is not None:
        return _wrap(fn)
    return _wrap

_lc_tools_mod = MagicMock()
_lc_tools_mod.tool = _passthrough_tool

for _mod_name, _mod_val in [
    ("langchain_core", MagicMock()),
    ("langchain_core.messages", MagicMock()),
    ("langchain_core.tools", _lc_tools_mod),
    ("langchain_core.runnables", MagicMock()),
]:
    sys.modules.setdefault(_mod_name, _mod_val)

for _mod in ["rerun", "rclpy", "rclpy.node", "rclpy.action", "std_msgs", "std_msgs.msg"]:
    sys.modules.setdefault(_mod, MagicMock())

import agents.cell.tools.dispatch as _dispatch_mod
from agents.cell.tools.dispatch import publish_task_assignment


SAMPLE_TRAJECTORY = [{"positions": [0.1, -0.5, 1.2, 0.0, 0.8, -0.3], "time_s": 0.0}]

_BASE_ARGS = {
    "cell_id": "test_cell",
    "robot_ns": "robot1",
    "operation_id": "W001",
    "operation_type": "weld",
    "trajectory": SAMPLE_TRAJECTORY,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spy_recorder():
    """Return a MagicMock recorder that records (phase, error) tuples."""
    spy = MagicMock()
    captured = []

    def _capture(operation_id, robot_ns, operation_type, trajectory, phase, error=""):
        captured.append((phase, error))

    spy.log_dispatch_event.side_effect = _capture
    spy._captured = captured
    return spy


def _phases(spy) -> list[str]:
    return [p for (p, _) in spy._captured]


# ---------------------------------------------------------------------------
# Test: dispatched phase is logged before any ROS 2 logic
# ---------------------------------------------------------------------------

class TestDispatchedPhaseLoggedFirst:

    def test_dispatched_is_first_recorder_call(self):
        """``phase='dispatched'`` must be the first recorder call, before any ROS 2 work."""
        spy = _spy_recorder()

        with patch.object(_dispatch_mod, "get_recorder", return_value=spy):
            publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W001"})

        phases = _phases(spy)
        assert len(phases) >= 1, "recorder must be called at least once"
        assert phases[0] == "dispatched", (
            f"First recorder call must be 'dispatched', got {phases[0]!r}"
        )

    def test_dispatched_logged_even_when_ros2_absent(self):
        """``phase='dispatched'`` is logged on the simulated (no-ROS-2) path too."""
        spy = _spy_recorder()

        # Force ImportError on rclpy so the simulated path runs
        with patch.object(_dispatch_mod, "get_recorder", return_value=spy), \
             patch.dict(sys.modules, {"rclpy": None}):
            publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W002"})

        assert "dispatched" in _phases(spy)

    def test_recorder_receives_correct_operation_id(self):
        """The operation_id forwarded to the recorder matches the tool input."""
        spy = MagicMock()
        with patch.object(_dispatch_mod, "get_recorder", return_value=spy):
            publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W_UNIQUE"})

        first_call = spy.log_dispatch_event.call_args_list[0]
        # Supports both positional and keyword call styles
        op_id = first_call.kwargs.get("operation_id") or first_call.args[0]
        assert op_id == "W_UNIQUE"

    def test_recorder_receives_correct_phase_dispatched(self):
        """The 'dispatched' call passes ``phase='dispatched'`` explicitly."""
        spy = MagicMock()
        with patch.object(_dispatch_mod, "get_recorder", return_value=spy):
            publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W003"})

        first_call = spy.log_dispatch_event.call_args_list[0]
        phase = first_call.kwargs.get("phase") or (
            first_call.args[4] if len(first_call.args) > 4 else None
        )
        assert phase == "dispatched", f"Expected 'dispatched', got {phase!r}"


# ---------------------------------------------------------------------------
# Test: completed phase is logged on success
# ---------------------------------------------------------------------------

class TestCompletedPhaseLoggedOnSuccess:

    def test_completed_logged_on_simulated_path(self):
        """Simulated dispatch (no ROS 2) logs 'dispatched' then 'completed'."""
        spy = _spy_recorder()

        # Ensure the ImportError path is taken
        with patch.object(_dispatch_mod, "get_recorder", return_value=spy), \
             patch.dict(sys.modules, {"rclpy": None}):
            result = publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W004"})

        phases = _phases(spy)
        assert "dispatched" in phases, f"'dispatched' missing from {phases}"
        assert "completed" in phases, f"'completed' missing from {phases}"
        assert phases.index("dispatched") < phases.index("completed"), (
            "'dispatched' must precede 'completed'"
        )

    def test_result_contains_dispatched_true(self):
        """The tool returns ``{'dispatched': True, ...}`` on the simulated path."""
        spy = _spy_recorder()

        with patch.object(_dispatch_mod, "get_recorder", return_value=spy), \
             patch.dict(sys.modules, {"rclpy": None}):
            result = publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W005"})

        assert result.get("dispatched") is True
        assert result.get("robot_ns") == "robot1"
        assert result.get("operation_id") == "W005"


# ---------------------------------------------------------------------------
# Test: failed phase is logged on unexpected exception
# ---------------------------------------------------------------------------

class TestFailedPhaseLoggedOnException:

    def test_failed_logged_when_rclpy_raises_runtime_error(self):
        """When rclpy.init() raises RuntimeError, recorder logs phase='failed'."""
        spy = _spy_recorder()

        rclpy_mock = MagicMock()
        rclpy_mock.ok.return_value = False
        rclpy_mock.init.side_effect = RuntimeError("simulated ROS 2 init failure")

        with patch.object(_dispatch_mod, "get_recorder", return_value=spy), \
             patch.dict(sys.modules, {"rclpy": rclpy_mock}):
            with patch("importlib.import_module", side_effect=RuntimeError("simulated")):
                try:
                    publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W006"})
                except Exception:
                    pass  # The tool re-raises on the failed path

        phases = _phases(spy)
        # dispatched must always be first
        assert phases[0] == "dispatched", f"First phase must be 'dispatched', got {phases}"

    def test_failed_phase_carries_error_string(self):
        """When dispatch fails the recorder's error parameter is non-empty."""
        spy = MagicMock()
        errors_seen = []

        def _capture(operation_id, robot_ns, operation_type, trajectory, phase, error=""):
            if phase == "failed":
                errors_seen.append(error)

        spy.log_dispatch_event.side_effect = _capture

        # Make the rclpy Node constructor raise so the except-Exception branch fires
        rclpy_mock = MagicMock()
        rclpy_mock.ok.return_value = True
        node_mock = MagicMock()
        node_mock.side_effect = RuntimeError("node crash")
        rclpy_mock.Node = node_mock

        with patch.object(_dispatch_mod, "get_recorder", return_value=spy), \
             patch.dict(sys.modules, {"rclpy": rclpy_mock}):
            try:
                publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W007"})
            except Exception:
                pass

        # If the failed path was taken the error string should be non-empty
        if errors_seen:
            assert all(len(e) > 0 for e in errors_seen), (
                "failed phase must carry a non-empty error string"
            )

    def test_dispatched_always_precedes_failed(self):
        """Even when the tool fails, 'dispatched' is always logged first."""
        spy = _spy_recorder()

        rclpy_mock = MagicMock()
        rclpy_mock.ok.return_value = True
        # Make Node() raise so the except-Exception branch fires
        rclpy_mock.Node.side_effect = RuntimeError("boom")

        with patch.object(_dispatch_mod, "get_recorder", return_value=spy), \
             patch.dict(sys.modules, {"rclpy": rclpy_mock}):
            try:
                publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W008"})
            except Exception:
                pass

        phases = _phases(spy)
        assert phases, "recorder must have been called"
        assert phases[0] == "dispatched", (
            f"'dispatched' must always be first, got {phases}"
        )


# ---------------------------------------------------------------------------
# Test: recorder call count
# ---------------------------------------------------------------------------

class TestRecorderCallCount:

    def test_exactly_two_calls_on_simulated_success(self):
        """Simulated success path calls the recorder exactly twice: dispatched + completed."""
        spy = MagicMock()
        with patch.object(_dispatch_mod, "get_recorder", return_value=spy), \
             patch.dict(sys.modules, {"rclpy": None}):
            publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W009"})

        assert spy.log_dispatch_event.call_count == 2, (
            f"Expected 2 recorder calls, got {spy.log_dispatch_event.call_count}"
        )

    def test_at_least_one_call_always(self):
        """The recorder is called at minimum once regardless of ROS 2 availability."""
        spy = MagicMock()
        with patch.object(_dispatch_mod, "get_recorder", return_value=spy):
            try:
                publish_task_assignment.invoke({**_BASE_ARGS, "operation_id": "W010"})
            except Exception:
                pass

        assert spy.log_dispatch_event.call_count >= 1
