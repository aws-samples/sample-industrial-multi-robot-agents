"""Host unit tests for the deterministic BehaviourTree executor.

These run on the host with NO ROS and NO Pinocchio: the executor talks only to a
duck-typed runtime + collision monitor, which we fake here. They verify the
runtime contract the cell depends on — ordering, gating, schedule-vs-time
deviation, and collision → abort + e-stop — independently of the robot fleet.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.cell.bt_executor import (  # noqa: E402
    BehaviourTreeExecutor, ExecConfig, interpolate_trajectory,
)


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeRuntime:
    def __init__(self):
        self._t0 = time.monotonic()
        self.dispatches: list[tuple[float, str, dict]] = []
        self.estops: list[str] = []
        self.statuses: dict[str, dict] = {}

    def now(self) -> float:
        return time.monotonic()

    def robots(self):
        return [("arms", "robot1"), ("arms", "robot2")]

    def dispatch(self, robot_id: str, task: dict) -> None:
        self.dispatches.append((time.monotonic() - self._t0, robot_id, task))

    def latest_status(self, robot_id: str):
        return self.statuses.get(robot_id)

    def estop(self, reason: str) -> None:
        self.estops.append(reason)

    def clear_estop(self, reason: str = "cleared") -> None:
        pass

    def shutdown(self) -> None:
        pass


class FakeMonitor:
    def __init__(self, detail=None):
        self.detail = detail

    def imminent(self):
        return self.detail


# ─── Config / builders ──────────────────────────────────────────────────────

def _cfg(**over):
    base = dict(
        tick_hz=200.0,
        position_gate_tolerance_m=0.05,
        position_gate_timeout_s=0.15,
        schedule_deviation_rad=100.0,      # effectively disabled unless overridden
        schedule_deviation_grace_s=0.05,
        schedule_check_warmup_s=0.0,       # tests drive steady-state; no dispatch
                                           # latency to warm up past
        completion_margin_s=0.02,
        home_time_s=0.02,
    )
    base.update(over)
    return ExecConfig(**base)


def _traj(end=0.02):
    return [
        {"positions": [0.0] * 6, "time_s": 0.0},
        {"positions": [0.05] * 6, "time_s": end},
    ]


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


def _run(root, runtime=None, monitor=None, cfg=None):
    runtime = runtime or FakeRuntime()
    ex = BehaviourTreeExecutor(runtime, monitor, cfg or _cfg())
    result = ex.execute({"root": root, "metadata": {}})
    return result, runtime


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_interpolate_trajectory_endpoints_and_midpoint():
    traj = [{"positions": [0.0, 0.0], "time_s": 0.0},
            {"positions": [2.0, 4.0], "time_s": 2.0}]
    assert interpolate_trajectory(traj, -1) == [0.0, 0.0]
    assert interpolate_trajectory(traj, 3) == [2.0, 4.0]
    assert interpolate_trajectory(traj, 1.0) == [1.0, 2.0]


def test_sequence_dispatches_in_order():
    root = {"id": "seq", "type": "SEQUENCE",
            "children": [_execute_node("robot1", "A"), _execute_node("robot2", "B")]}
    result, rt = _run(root)
    assert result["success"] is True, result
    assert result["executed_nodes"] == 2
    dispatched_ids = [rid for _t, rid, _task in rt.dispatches]
    assert dispatched_ids == ["robot1", "robot2"]
    assert rt.estops == []


def test_parallel_dispatches_both():
    root = {"id": "par", "type": "PARALLEL",
            "children": [_execute_node("robot1", "A"), _execute_node("robot2", "B")]}
    result, rt = _run(root)
    assert result["success"] is True, result
    assert {rid for _t, rid, _task in rt.dispatches} == {"robot1", "robot2"}


def test_start_delay_respected():
    root = _execute_node("robot1", "A", start_delay_s=0.1)
    result, rt = _run(root)
    assert result["success"] is True, result
    assert rt.dispatches, "expected a dispatch"
    first_dispatch_t = rt.dispatches[0][0]
    assert first_dispatch_t >= 0.1, f"dispatch happened before the start delay: {first_dispatch_t}"


def test_collision_aborts_and_estops():
    detail = {"kind": "robot_pair", "robot_a": "robot1", "robot_b": "robot2",
              "clearance_m": -0.05, "reason": "robot1 ↔ robot2 clearance -0.05 m"}
    root = _execute_node("robot1", "A", end=0.3)
    result, rt = _run(root, monitor=FakeMonitor(detail))
    assert result["success"] is False
    fail = result["failure"]
    assert fail["reason_code"] == "collision", fail
    assert fail["collision"]["robot_a"] == "robot1"
    assert rt.estops, "collision must trigger an e-stop"


def test_schedule_deviation_aborts_and_estops():
    rt = FakeRuntime()
    # Robot reports a joint config far from the planned (~0) trajectory.
    rt.statuses["robot1"] = {"joint_positions": [10.0] * 6, "ee_position": [0, 0, 0]}
    root = _execute_node("robot1", "A", end=0.3)
    # The start-pose check would refuse this leg before dispatch (the arm is not
    # at its start); off here, so the in-motion deviation check is what is tested.
    result, rt = _run(root, runtime=rt,
                      cfg=_cfg(schedule_deviation_rad=0.1, schedule_deviation_grace_s=0.02,
                               start_pose_tolerance_rad=None))
    assert result["success"] is False
    fail = result["failure"]
    assert fail["reason_code"] == "schedule_deviation", fail
    assert rt.estops, "schedule deviation must trigger an e-stop"


def test_on_schedule_execute_succeeds():
    rt = FakeRuntime()
    # Robot tracks the plan well (near 0); with a tight limit this must NOT trip.
    rt.statuses["robot1"] = {"joint_positions": [0.01] * 6, "ee_position": [0, 0, 0]}
    root = _execute_node("robot1", "A", end=0.1)
    result, rt = _run(root, runtime=rt,
                      cfg=_cfg(schedule_deviation_rad=0.2, schedule_deviation_grace_s=0.02))
    assert result["success"] is True, result
    assert rt.estops == []


def test_interlock_wait_position_gate_timeout():
    rt = FakeRuntime()
    rt.statuses["robot1"] = {"joint_positions": [0] * 6, "ee_position": [0.0, 0.0, 0.0]}
    root = {"id": "iw", "type": "INTERLOCK_WAIT", "robot_ns": "robot1",
            "depends_on_robots": [], "depends_on_positions": [
                {"robot": "robot1", "position": [1.0, 1.0, 1.0], "tolerance_m": 0.05}]}
    result, rt = _run(root, runtime=rt)
    assert result["success"] is False
    assert result["failure"]["reason_code"] == "position_gate_timeout", result


def test_interlock_wait_position_gate_pass():
    rt = FakeRuntime()
    rt.statuses["robot1"] = {"joint_positions": [0] * 6, "ee_position": [1.0, 1.0, 1.0]}
    root = {"id": "iw", "type": "INTERLOCK_WAIT", "robot_ns": "robot1",
            "depends_on_robots": [], "depends_on_positions": [
                {"robot": "robot1", "position": [1.0, 1.0, 1.0], "tolerance_m": 0.05}]}
    result, rt = _run(root, runtime=rt)
    assert result["success"] is True, result


def test_unknown_node_type_fails():
    result, _rt = _run({"id": "x", "type": "FROBNICATE"})
    assert result["success"] is False
    assert result["failure"]["reason_code"] == "unknown_node", result


def test_empty_trajectory_fails():
    node = _execute_node("robot1", "A")
    node["precomputed_trajectory"] = []
    result, _rt = _run(node)
    assert result["success"] is False
    assert result["failure"]["reason_code"] == "empty_trajectory", result
