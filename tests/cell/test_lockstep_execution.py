"""Lockstep execution of ompl-compound steps (opt-in, ``LOCKSTEP=1``).

A compound step's legs are collision-free only in step with each other. With
lockstep on, they start off one barrier and an arm that drifts off the shared
clock halts the whole cell; with it off, nothing changes. Host-runnable.
"""
from __future__ import annotations

import time

from agents.cell.bt_executor import (
    BehaviourTreeExecutor, lockstep_from_env, trajectory_lag_s,
)
from tests.test_bt_executor import FakeRuntime, _cfg


def leg(rid, op, group="g1", members=("robot1", "robot2"), end=0.3, steps=6):
    traj = [{"positions": [i / steps] * 6, "time_s": end * i / steps} for i in range(steps + 1)]
    if group:
        traj[0]["sync_group"] = group
        traj[0]["sync_members"] = list(members)
    return {"id": f"n_{rid}_{op}", "type": "ExecuteTrajectory", "robot_ns": rid,
            "operation_id": op, "operation_type": "weld", "precomputed_trajectory": traj,
            "hold_duration_s": 0.0, "velocity_scale": 1.0}


def parallel(*children):
    return {"id": "p", "type": "Parallel", "children": list(children)}


def run(root, runtime=None, **cfg):
    runtime = runtime or FakeRuntime()
    ex = BehaviourTreeExecutor(runtime, None, _cfg(**cfg))
    return ex.execute({"root": root, "metadata": {}}), runtime


class TestSwitch:
    def test_env_values(self):
        assert lockstep_from_env({"LOCKSTEP": "1"}) is True
        assert lockstep_from_env({"LOCKSTEP": "on"}) is True
        assert lockstep_from_env({"LOCKSTEP": "0"}) is False
        assert lockstep_from_env({}) is False


class TestBarrier:
    def test_off_a_staggered_partner_does_not_hold_the_other_back(self):
        root = parallel(leg("robot1", "a"),
                        {"id": "d", "type": "Delay", "delay_msec": 200,
                         "children": [leg("robot2", "b")]})
        result, rt = run(root, lockstep=False)
        assert result["success"]
        t = {rid: ts for ts, rid, _ in rt.dispatches}
        assert t["robot2"] - t["robot1"] > 0.15

    def test_on_the_legs_dispatch_together(self):
        root = parallel(leg("robot1", "a"),
                        {"id": "d", "type": "Delay", "delay_msec": 200,
                         "children": [leg("robot2", "b")]})
        result, rt = run(root, lockstep=True)
        assert result["success"], result
        t = {rid: ts for ts, rid, _ in rt.dispatches}
        assert abs(t["robot2"] - t["robot1"]) < 0.05

    def test_a_partner_that_never_comes_fails_without_dispatching(self):
        root = leg("robot1", "a")          # robot2's leg is not in the tree
        result, rt = run(root, lockstep=True, lockstep_start_timeout_s=0.2)
        assert not result["success"]
        assert result["failure"]["reason_code"] == "lockstep_partner_missing"
        assert rt.dispatches == []

    def test_legs_in_sequence_cannot_be_in_step(self):
        root = {"id": "s", "type": "Sequence",
                "children": [leg("robot1", "a"), leg("robot2", "b")]}
        result, rt = run(root, lockstep=True, lockstep_start_timeout_s=0.2)
        assert result["failure"]["reason_code"] == "lockstep_partner_missing"

    def test_different_velocity_scales_are_refused(self):
        slow = leg("robot2", "b")
        slow["velocity_scale"] = 0.5
        result, _ = run(parallel(leg("robot1", "a"), slow), lockstep=True,
                        lockstep_start_timeout_s=0.3)
        assert result["failure"]["reason_code"] in ("lockstep_velocity_mismatch",
                                                    "lockstep_partner_missing")

    def test_untagged_legs_are_unaffected_by_lockstep(self):
        root = parallel(leg("robot1", "a", group=None), leg("robot2", "b", group=None))
        result, _ = run(root, lockstep=True)
        assert result["success"]


class TestDrift:
    def test_lag_reads_off_the_trajectory(self):
        traj = [{"positions": [float(i)] * 6, "time_s": float(i)} for i in range(6)]
        assert trajectory_lag_s(traj, [2.0] * 6, 3.0) == 1.0     # one second behind
        assert trajectory_lag_s(traj, [3.0] * 6, 3.0) == 0.0
        assert trajectory_lag_s(traj, [4.0] * 6, 3.0) == -1.0    # ahead

    def test_a_hold_is_on_time(self):
        traj = [{"positions": [0.0] * 6, "time_s": t} for t in (0.0, 1.0, 2.0, 3.0)]
        assert trajectory_lag_s(traj, [0.0] * 6, 2.0) == 0.0

    def test_an_arm_stuck_at_its_start_halts_every_arm(self):
        rt = FakeRuntime()
        rt.statuses = {"robot1": {"joint_positions": [0.0] * 6},   # never moves
                       "robot2": {"joint_positions": [0.0] * 6}}
        root = parallel(leg("robot1", "a", end=2.0, steps=20),
                        leg("robot2", "b", end=2.0, steps=20))
        t0 = time.monotonic()
        result, rt = run(root, runtime=rt, lockstep=True, lockstep_warmup_s=0.0,
                         lockstep_tolerance_s=0.3, lockstep_grace_s=0.05)
        assert not result["success"]
        assert result["failure"]["reason_code"] == "lockstep_lost"
        assert rt.estops and "lockstep lost" in rt.estops[0]
        assert time.monotonic() - t0 < 2.0       # halted, did not run to the end

    def test_the_same_drift_without_lockstep_is_not_a_lockstep_halt(self):
        rt = FakeRuntime()
        rt.statuses = {"robot1": {"joint_positions": [0.0] * 6},
                       "robot2": {"joint_positions": [0.0] * 6}}
        root = parallel(leg("robot1", "a", end=0.5, steps=5),
                        leg("robot2", "b", end=0.5, steps=5))
        result, rt = run(root, runtime=rt, lockstep=False, arrival_timeout_s=0.1)
        # No lockstep halt and no e-stop. The arms never left their start, so the
        # legs did not COMPLETE either (position-confirmed completion): the run
        # stops on the arrival check, not on the clock.
        assert rt.estops == []
        assert result["failure"]["reason_code"] == "arrival_timeout"
