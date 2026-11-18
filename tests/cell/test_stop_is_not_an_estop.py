"""Stop ends the activity; Emergency stop halts the arms. Two buttons, two paths.

The panel now has both, and the whole point of separating them is that Stop is
cheap and E-stop is not: Stop dispatches no further nodes and lets the in-flight
trajectory finish, so the cell is left in a known, resumable state. E-stop
cancels the live ``FollowJointTrajectory`` goal and latches the cell — arms halt
where they are, and something has to clear the latch afterwards.

The failure that motivates this file is a one-line drift in the other direction:
if ``abort()`` ever grew an ``estop()`` call, every Stop press would become a
latched emergency halt. Nothing in the return value would change, the run would
still report ``operator_abort``, and the operator would find the cell latched and
the arms stopped mid-weld with no idea why. So ``rt.estops == []`` is asserted
after every abort here — it is the invariant that keeps the two buttons different.

The executor is exercised for real (its own fakes, no ROS, no Pinocchio, as in
``tests/test_bt_executor.py``); the wiring that hands the panel a handle on the
live executor is read as source, because it is a process-global slot.
"""
from __future__ import annotations

import ast
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.cell.bt_executor import BehaviourTreeExecutor, ExecConfig  # noqa: E402

CELL_AGENT = PROJECT_ROOT / "agents" / "cell" / "agent.py"
CELL_MCP = PROJECT_ROOT / "agents" / "cell" / "mcp_server.py"


# ─── Fakes (same contract as tests/test_bt_executor.py) ───────────────────────

class FakeRuntime:
    def __init__(self):
        self._t0 = time.monotonic()
        self.dispatches: list[tuple[float, str, dict]] = []
        self.estops: list[str] = []
        self.cleared: list[str] = []

    def now(self) -> float:
        return time.monotonic()

    def robots(self):
        return [("arms", "robot1"), ("arms", "robot2")]

    def dispatch(self, robot_id: str, task: dict) -> None:
        self.dispatches.append((time.monotonic() - self._t0, robot_id, task))

    def latest_status(self, robot_id: str):
        return None

    def estop(self, reason: str) -> None:
        self.estops.append(reason)

    def clear_estop(self, reason: str = "cleared") -> None:
        self.cleared.append(reason)

    def shutdown(self) -> None:
        pass


class FakeMonitor:
    def __init__(self, detail=None):
        self.detail = detail

    def imminent(self):
        return self.detail


def _cfg(**over):
    base = dict(
        tick_hz=200.0,
        position_gate_tolerance_m=0.05,
        position_gate_timeout_s=0.15,
        schedule_deviation_rad=100.0,      # disabled; these tests are about abort
        schedule_deviation_grace_s=0.05,
        schedule_check_warmup_s=0.0,
        completion_margin_s=0.02,
        home_time_s=0.02,
    )
    base.update(over)
    return ExecConfig(**base)


def _traj(end=0.02):
    return [{"positions": [0.0] * 6, "time_s": 0.0},
            {"positions": [0.05] * 6, "time_s": end}]


def _execute_node(rid, op="op", end=0.02, **kw):
    node = {
        "id": f"n_{rid}_{op}", "type": "EXECUTE", "robot_ns": rid,
        "operation_id": op, "operation_type": "weld",
        "precomputed_trajectory": _traj(end),
        "has_interlock": False, "interlock_target": None,
        "depends_on_robots": [], "depends_on_positions": [],
        "start_delay_s": 0.0, "hold_duration_s": 0.0, "velocity_scale": 1.0,
    }
    node.update(kw)
    return node


def _run_in_thread(ex, tree):
    """Start a run and hand back a thread plus a place for its result."""
    out: dict = {}

    def go():
        out["result"] = ex.execute(tree)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t, out


def _await(predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ─── Stop, from outside the executing thread ─────────────────────────────────

class TestStopEndsTheRun:
    def test_a_running_tree_stops_and_the_rest_is_never_dispatched(self):
        """The behaviour the operator is buying: no further motion is commanded."""
        rt, ex = FakeRuntime(), None
        ex = BehaviourTreeExecutor(rt, FakeMonitor(), _cfg())
        root = {"id": "seq", "type": "SEQUENCE", "children": [
            _execute_node("robot1", "A", end=1.5),
            _execute_node("robot2", "B", end=1.5),
            _execute_node("robot1", "C", end=1.5),
        ]}
        t, out = _run_in_thread(ex, {"root": root, "metadata": {}})
        assert _await(lambda: rt.dispatches), "first node never dispatched"

        assert ex.abort("stopped by operator") is True
        t.join(timeout=5)
        assert not t.is_alive(), "the run ignored abort() and is still executing"

        result = out["result"]
        assert result["success"] is False
        assert result["failure"]["reason_code"] == "operator_abort"
        assert "operator" in result["failure"]["reason"]
        assert len(rt.dispatches) == 1, \
            f"nodes were dispatched after the stop: {[d[1:] for d in rt.dispatches]}"

    def test_stopping_does_not_estop_the_cell(self):
        """The line this whole file exists for. Stop must command nothing — no
        e-stop, no latch. If it halted the arms, the two buttons would be one."""
        rt = FakeRuntime()
        ex = BehaviourTreeExecutor(rt, FakeMonitor(), _cfg())
        t, out = _run_in_thread(
            ex, {"root": _execute_node("robot1", "A", end=1.5), "metadata": {}})
        assert _await(lambda: rt.dispatches)
        ex.abort()
        t.join(timeout=5)

        assert out["result"]["success"] is False
        assert rt.estops == [], \
            f"Stop published an e-stop ({rt.estops}) — it must only end the " \
            "activity; halting the arms is the Emergency stop button"
        assert rt.cleared == [], "Stop cleared an e-stop latch it never set"

    def test_stop_reports_whether_it_was_the_one_that_stopped_the_run(self):
        """Pressed twice — or pressed just after a collision already aborted —
        the second call must not claim it stopped anything, or the log would
        blame the operator for a fault the cell found itself."""
        ex = BehaviourTreeExecutor(FakeRuntime(), FakeMonitor(), _cfg())
        assert ex.abort("first") is True
        assert ex.abort("second") is False
        assert ex.aborted is True

    def test_a_real_fault_outranks_a_later_stop(self):
        """A collision, then the operator hits Stop. The report must still say
        collision: that is the reason the run is over and the thing to fix."""
        ex = BehaviourTreeExecutor(FakeRuntime(), FakeMonitor(), _cfg())
        ex._raise_abort({"node_type": "EXECUTE", "reason_code": "collision_imminent",
                         "reason": "robot1 ↔ robot2 clearance -0.05 m"})
        ex.abort("stopped by operator")
        assert ex._abort_failure["reason_code"] == "collision_imminent", \
            "the operator's stop overwrote the first failure"

    def test_a_stop_pressed_while_idle_does_not_poison_the_next_run(self):
        """Executors are reused. A latched abort flag would make the next Replay
        fail instantly with 'operator_abort' — a dead button with no explanation."""
        rt = FakeRuntime()
        ex = BehaviourTreeExecutor(rt, FakeMonitor(), _cfg())
        ex.abort("stray press")
        result = ex.execute({"root": _execute_node("robot1", "A"), "metadata": {}})
        assert result["success"] is True, \
            f"a stop pressed before the run leaked into it: {result.get('failure')}"
        assert rt.estops == []

    def test_stop_lands_during_a_start_delay_too(self):
        """Trees start with waits. If abort were only honoured between nodes, a
        Stop pressed during one would hang until the delay expired."""
        rt = FakeRuntime()
        ex = BehaviourTreeExecutor(rt, FakeMonitor(), _cfg())
        node = _execute_node("robot1", "A", start_delay_s=10.0)
        t, out = _run_in_thread(ex, {"root": node, "metadata": {}})
        time.sleep(0.05)
        ex.abort()
        t.join(timeout=5)
        assert not t.is_alive(), "abort was not noticed during the start delay"
        assert out["result"]["success"] is False
        assert rt.dispatches == [], "the node was dispatched after the stop"
        assert rt.estops == []


# ─── The handle the panel needs on the live run ───────────────────────────────

class TestTheLiveRunIsReachableFromOutside:
    """``abort()`` is useless without a reference to the executor that is
    currently running; the HTTP thread that serves Stop has none."""

    def test_the_executor_is_published_while_it_runs_and_withdrawn_after(self):
        source = CELL_AGENT.read_text()
        body = _function_source(source, "execute_behaviour_tree")
        assert "_ACTIVE_EXECUTOR = executor" in body, \
            "the live executor is no longer published; Stop has nothing to abort"
        assert body.index("finally:") < body.index("_ACTIVE_EXECUTOR = None"), \
            "the slot is not cleared in a finally — a crashed run would leave a " \
            "dead executor published and Stop would report aborting it"

    def test_stopping_nothing_says_so_instead_of_claiming_success(self):
        source = CELL_AGENT.read_text()
        body = _function_source(source, "abort_active_execution")
        assert '"aborted": False' in body, \
            "an idle cell would report a successful abort; the operator would " \
            "believe a run was stopped that was never running"

    def test_the_slot_is_guarded_by_a_lock(self):
        source = CELL_AGENT.read_text()
        assert "_ACTIVE_LOCK" in source
        body = _function_source(source, "abort_active_execution")
        assert "_ACTIVE_LOCK" in body, \
            "the abort reads the slot without the lock it is written under"


class TestTheTwoToolsSayWhichKindOfStopTheyAre:
    def test_abort_execution_never_reaches_for_the_estop(self):
        """Its docstring names the e-stop (it points the caller at the other
        tool); its code must never call one."""
        code = _function_code(CELL_MCP.read_text(), "abort_execution")
        for forbidden in ("trigger_estop", "estop_pub", "clear_estop"):
            assert forbidden not in code, \
                f"abort_execution touches '{forbidden}'; Stop must not halt arms"

    def test_abort_execution_warns_that_it_is_not_an_emergency_stop(self):
        """The tool is called by an LLM as well as by the panel. If the docstring
        does not draw the line, the model will pick this for 'stop the robots'."""
        body = _function_source(CELL_MCP.read_text(), "abort_execution")
        doc = ast.get_docstring(ast.parse(body).body[0]) or ""
        assert "NOT AN EMERGENCY STOP" in doc.upper(), \
            "nothing tells a caller that abort_execution leaves the arms moving"

    def test_homing_goes_through_the_gated_planner_not_a_blind_jog(self):
        """Reset moves every arm at once from wherever the stop left them. Jogging
        them home one at a time, unplanned, is how two arms meet in the middle."""
        body = _function_source(CELL_MCP.read_text(), "home_robots")
        assert "_home_reset_tree" in body, \
            "home_robots no longer uses the gated home-reset tree; check whether " \
            "it now commands unplanned motion"
        doc = ast.get_docstring(ast.parse(body).body[0]) or ""
        assert "MOVES THE ROBOTS" in doc.upper(), \
            "home_robots does not announce that it commands motion"
        assert "estop" in doc.lower(), \
            "home_robots clears a latched e-stop to move; that must be stated"


def _function_source(source: str, name: str) -> str:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    pytest.fail(f"{name} is gone; this test guards nothing")


def _function_code(source: str, name: str) -> str:
    """One ``def`` with its docstring dropped — for asking what it DOES, not what
    it says. These docstrings discuss the e-stop on purpose."""
    fn = ast.parse(_function_source(source, name)).body[0]
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    return "\n".join(ast.unparse(stmt) for stmt in body)
