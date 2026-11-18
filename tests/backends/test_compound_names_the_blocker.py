"""A stroke blocked by a parked arm names that arm; an approach waypoint gets a
second IK seed (the weld's own solution) before the leg is refused; and the log
line says why a planner call failed. See leg_builder."""
from __future__ import annotations

import pytest

from agents.trajectory.tools import leg_builder as lb


class _IK:
    """Converges only from the seeds in ``good``."""

    def __init__(self, good):
        self.good, self.calls = [list(g) for g in good], []

    def invoke(self, args):
        self.calls.append(args["warm_start_joints"])
        ok = args["warm_start_joints"] in self.good
        return {"success": ok, "joint_positions": [9.0] * 6} if ok else {"success": False}


def test_the_weld_solution_is_the_second_seed():
    ik = _IK(good=[[0.5] * 6])
    q = lb.approach_ik(ik, "c", "robot1", (0, 0, 1), {"constrain_orientation": True},
                       [[0.0] * 6, [0.5] * 6])
    assert q == [9.0] * 6 and ik.calls == [[0.0] * 6, [0.5] * 6]


def test_no_seed_no_waypoint_and_a_shorter_stroke_is_offered():
    assert lb.approach_ik(_IK(good=[]), "c", "r", (0, 0, 1), {}, [[0.0] * 6, None]) is None
    f = lb.approach_waypoint_failure("robot1", 0.1)
    assert f["reason_code"] == "approach_unreachable"
    assert f["suggested_approach_offset_m"] == 0.05


def test_only_the_arm_that_blocks_on_its_own_is_named():
    arms = {"robot2": ["s2"], "robot3": ["s3"], "robot4": []}
    out = lb.blocking_arms(lambda sph: not (set(sph) & {"s2"}), arms)
    assert out == ["robot2"]
    f = lb.blocked_by_parked_failure("robot1", out, "straight descent into the weld")
    assert f["reason_code"] == "blocked_by_parked_arm" and f["blocked_by"] == ["robot2"]
    assert "robot2" in f["error"]


def test_the_recovery_for_a_parked_blocker_is_to_move_it_not_reallocate():
    from agents.trajectory.reason_codes import recovery
    rec = recovery("blocked_by_parked_arm")
    assert rec["escalate"] == "resequence" and "move" in rec["levers"][0]


def test_a_failed_planner_call_logs_its_reason_codes():
    import json
    agent = pytest.importorskip("agents.trajectory.agent")
    line = agent._summarize_tool_result("run_trajectory_planner", json.dumps({
        "success": False, "trajectories": {}, "planning_time_s": 1.0,
        "failures": [{"operation_id": "W05", "reason_code": "blocked_by_parked_arm",
                      "blocked_by": ["robot2"]}]}))
    assert "?" not in line and "W05:blocked_by_parked_arm<-robot2" in line
