"""The executor confirms motion by where the arm IS, and runs only while the
collision monitor is watching. Host-runnable (fakes from test_bt_executor)."""
from __future__ import annotations

import time

import pytest

from agents.cell.bt_executor import BehaviourTreeExecutor
from tests.test_bt_executor import FakeMonitor, FakeRuntime, _cfg, _execute_node


def _long(rid, op, end=0.05):
    """A leg that moves 1 rad: far more than the arrival tolerance."""
    node = _execute_node(rid, op, end=end)
    node["precomputed_trajectory"] = [{"positions": [0.0] * 6, "time_s": 0.0},
                                      {"positions": [1.0] * 6, "time_s": end}]
    return node


def _run(root, rt=None, mon=None, **cfg):
    rt = rt or FakeRuntime()
    ex = BehaviourTreeExecutor(rt, mon, _cfg(**cfg))
    return ex.execute({"root": root, "metadata": {}}), rt


class MovingRuntime(FakeRuntime):
    """Reports the arm on its last dispatched trajectory, ``lag_s`` behind."""

    def __init__(self, lag_s=0.0, stall_at=None):
        super().__init__()
        self.lag_s, self.stall_at = lag_s, stall_at
        self.start = {"robot1": [0.0] * 6}

    def latest_status(self, rid):
        mine = [(t, task) for t, r, task in self.dispatches if r == rid]
        if not mine:
            return {"joint_positions": self.start.get(rid, [0.0] * 6)}
        t0, task = mine[-1]
        from agents.cell.bt_executor import interpolate_trajectory
        t = (time.monotonic() - self._t0) - t0 - self.lag_s
        if self.stall_at is not None:
            t = min(t, self.stall_at)
        return {"joint_positions": interpolate_trajectory(task["trajectory"], max(t, 0.0))}


# ── completion is position-confirmed ─────────────────────────────────────────

def test_an_arm_that_arrives_completes():
    result, _ = _run(_long("robot1", "A"), MovingRuntime())
    assert result["success"] is True, result


def test_an_arm_that_lags_is_waited_for_not_declared_done():
    rt = MovingRuntime(lag_s=0.15)
    t0 = time.monotonic()
    result, _ = _run(_long("robot1", "A"), rt, arrival_timeout_s=1.0)
    assert result["success"] is True, result
    assert time.monotonic() - t0 >= 0.15


def test_an_arm_that_stalls_short_of_the_end_fails_the_leg():
    rt = MovingRuntime(stall_at=0.0)          # accepted the goal, never moved
    root = {"id": "s", "type": "SEQUENCE",
            "children": [_long("robot1", "A"), _long("robot1", "B")]}
    result, rt = _run(root, rt, arrival_timeout_s=0.1, start_pose_tolerance_rad=None)
    assert result["failure"]["reason_code"] == "arrival_timeout"
    assert [task["operation_id"] for _t, _r, task in rt.dispatches] == ["A"]


def test_without_joint_feedback_the_clock_still_decides():
    result, _ = _run(_execute_node("robot1", "A", end=0.02))      # no statuses at all
    assert result["success"] is True, result


# ── a leg starts where the arm stands ────────────────────────────────────────

def test_a_leg_from_elsewhere_is_not_sent():
    rt = FakeRuntime()
    rt.statuses["robot1"] = {"joint_positions": [1.0] * 6}
    result, rt = _run(_execute_node("robot1", "A"), rt, start_settle_s=0.05)
    assert result["failure"]["reason_code"] == "start_pose_mismatch"
    assert rt.dispatches == [] and rt.estops == []


def test_a_single_waypoint_move_is_not_start_checked():
    rt = FakeRuntime()
    rt.statuses["robot1"] = {"joint_positions": [0.05] * 6}
    node = _execute_node("robot1", "A")
    node["precomputed_trajectory"] = [{"positions": [0.05] * 6, "time_s": 0.02}]
    result, _ = _run(node, rt, start_settle_s=0.05)
    assert result["success"] is True, result


# ── the monitor must be watching ─────────────────────────────────────────────

class SickMonitor(FakeMonitor):
    def __init__(self, why=None, raises=False):
        super().__init__()
        self.why, self.raises = why, raises

    def imminent(self):
        if self.raises:
            raise RuntimeError("monitor crashed")
        return None

    def health(self):
        return self.why


def test_nothing_moves_while_the_monitor_is_not_watching():
    result, rt = _run(_execute_node("robot1", "A"), mon=SickMonitor("disabled (no kinematics)"))
    assert result["failure"]["reason_code"] == "monitor_fault"
    assert rt.dispatches == []


def test_a_monitor_that_raises_mid_run_stops_the_cell():
    mon = SickMonitor()
    root = _execute_node("robot1", "A", end=0.3)
    rt = FakeRuntime()
    ex = BehaviourTreeExecutor(rt, mon, _cfg())
    import threading
    out = {}
    th = threading.Thread(target=lambda: out.update(r=ex.execute({"root": root})))
    th.start()
    time.sleep(0.05)
    mon.raises = True
    th.join(timeout=3)
    assert out["r"]["failure"]["reason_code"] == "monitor_fault"
    assert rt.estops, "a monitor that broke mid-motion e-stops the cell"


def test_a_monitor_without_a_health_report_is_taken_at_its_word():
    result, _ = _run(_execute_node("robot1", "A"), mon=FakeMonitor())
    assert result["success"] is True, result


# ── the monitor reports its own health ──────────────────────────────────────

def test_the_monitor_says_why_it_is_not_watching(monkeypatch):
    cm = pytest.importorskip("agents.cell.collision_monitor")
    m = cm.CollisionMonitor("c", runtime=FakeRuntime())
    assert "disabled" in m.health()
    monkeypatch.setattr(m, "_required", False)
    assert m.health() is None
    m._enabled, m._required = True, True
    m._started_at, m._last_ok = time.monotonic(), time.monotonic()
    assert m.health() is None
    m._blind = ["robot2"]
    assert "robot2" in m.health()
    m._blind, m._last_ok = [], time.monotonic() - 10
    assert "old" in m.health()


def test_stale_feedback_is_not_where_the_arm_is():
    cm = pytest.importorskip("agents.cell.collision_monitor")
    assert cm._fresh({"_recv_t": time.time()}) is True
    assert cm._fresh({"_recv_t": time.time() - 30}) is False
    assert cm._fresh({}) is True


# ── the controller is sent the planned velocities ───────────────────────────

def test_interior_points_carry_the_profile_velocity():
    from agents.robot.fjt_profile import point_velocities
    traj = [{"positions": [0.0], "time_s": 0.0}, {"positions": [1.0], "time_s": 1.0},
            {"positions": [3.0], "time_s": 2.0}, {"positions": [3.0], "time_s": 3.0},
            {"positions": [2.0], "time_s": 4.0}]
    v = [p[0] for p in point_velocities(traj, 1)]
    assert v[0] == 0.0 and v[-1] == 0.0                # ends at rest
    assert v[1] == pytest.approx(2 * 1 * 2 / 3)         # harmonic mean of 1 and 2
    assert v[2] == 0.0                                  # into a hold
    assert v[3] == 0.0                                  # reversal


def test_a_point_with_its_own_velocities_keeps_them():
    from agents.robot.fjt_profile import point_velocities
    traj = [{"positions": [0.0], "time_s": 0.0},
            {"positions": [1.0], "time_s": 1.0, "velocities": [0.7]},
            {"positions": [2.0], "time_s": 2.0}]
    assert point_velocities(traj, 1)[1] == [0.7]
