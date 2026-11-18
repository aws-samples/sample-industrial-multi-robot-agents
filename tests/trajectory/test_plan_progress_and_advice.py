"""A run says how far it is; a plan says what to fix; an empty plan is not a plan.

Found on welding_cell_2_small's bt_1790951179: the stored tree carried
agent_view SUMMARIES of every leg instead of waypoints, scored 0.85
"acceptable", and reported success. These tests pin the four answers that came
out of that:

* ``plan_events.progress`` — a percentage of the program, from the event stream;
* ``RunRegistry.note_progress`` — live progress on a running run, never a
  lifecycle change;
* ``trajectory_quality.score_plan`` — no measurable motion scores 0, not 0.85;
* ``plan_advice.advise`` — ranked problems with evidence and a suggestion;
* ``agent._full_tree`` — the persisted tree is the assembler's real one
  (container only: the agent module needs langchain).

Host-runnable except the last class.
"""
from __future__ import annotations

import json

import pytest

from agents.orchestrator.run_registry import RUNNING, SUCCEEDED, RunRegistry
from agents.trajectory import plan_advice, plan_events
from agents.trajectory.tools import trajectory_quality as tq


def stream(*events) -> str:
    return "\n".join(json.dumps({"seq": i + 1, "at": f"2026-10-02T14:{i:02d}:00",
                                 **e}) for i, e in enumerate(events))


PROGRAM = ["W01", "W02", "W03", "W04"]


class TestProgress:
    def test_no_events_is_queued_at_zero(self):
        p = plan_events.progress("c", "t", PROGRAM, text="")
        assert p["phase"] == "queued" and p["percent"] == 0.0

    def test_percent_is_the_solved_share_of_the_planning_phase(self):
        text = stream(
            {"kind": "run_started", "operations": 4},
            {"kind": "operation_planned", "operation_id": "W01"},
            {"kind": "operation_failed", "operation_id": "W02"},
        )
        p = plan_events.progress("c", "t", PROGRAM, text=text)
        assert p["phase"] == "planning"
        assert p["operations_solved"] == 1 and p["operations_total"] == 4
        assert p["percent"] == pytest.approx(plan_events.PLANNING_SHARE / 4)
        assert p["failed_attempts"] == 1

    def test_a_renamed_retry_still_solves_its_operation(self):
        text = stream({"kind": "operation_validated", "operation_id": "W03_fix"},
                      {"kind": "operation_planned", "operation_id": "W04_final"})
        assert plan_events.progress("c", "t", PROGRAM, text=text)["operations_solved"] == 2

    def test_an_arm_prefixed_id_solves_its_operation(self):
        # Seen live on bt_1790964385: the planner named W13 "R2_W13".
        text = stream({"kind": "operation_planned", "operation_id": "R2_W02"})
        assert plan_events.progress("c", "t", PROGRAM, text=text)["operations_solved"] == 1

    def test_a_prefix_is_not_a_rename(self):
        # W1 must not be solved by W10_final.
        text = stream({"kind": "operation_planned", "operation_id": "W10_final"})
        assert plan_events.progress("c", "t", ["W1"], text=text)["operations_solved"] == 0

    def test_a_later_failure_does_not_take_progress_back(self):
        text = stream({"kind": "operation_planned", "operation_id": "W01"},
                      {"kind": "operation_failed", "operation_id": "W01"})
        assert plan_events.progress("c", "t", PROGRAM, text=text)["operations_solved"] == 1

    def test_assembling_and_done(self):
        assembling = stream({"kind": "operation_started", "operation_id": "W01",
                             "tool": "assemble_behaviour_tree"})
        assert plan_events.progress("c", "t", PROGRAM, text=assembling)["percent"] == 95.0
        done = stream({"kind": "finished", "success": True})
        p = plan_events.progress("c", "t", PROGRAM, text=done)
        assert p["phase"] == "done" and p["percent"] == 100.0

    def test_a_failed_finish_is_not_100(self):
        p = plan_events.progress("c", "t", PROGRAM,
                                 text=stream({"kind": "finished", "success": False}))
        assert p["phase"] == "failed" and p["percent"] < 100.0

    def test_without_the_program_the_total_comes_from_run_started(self):
        text = stream({"kind": "run_started", "operations": 2},
                      {"kind": "operation_planned", "operation_id": "X"})
        p = plan_events.progress("c", "t", None, text=text)
        assert p["approximate"] is True and p["operations_total"] == 2
        assert p["percent"] == pytest.approx(plan_events.PLANNING_SHARE / 2)


class TestRunProgress:
    def test_progress_is_attached_to_a_running_run(self):
        reg = RunRegistry()
        env = reg.start("c", "plan", program="p")
        reg.note_progress(env["run_id"], progress={"percent": 40.0}, tree_id="bt_1")
        got = reg.get(env["run_id"])
        assert got["progress"] == {"percent": 40.0} and got["tree_id"] == "bt_1"
        assert got["status"] == RUNNING and got["done"] is False

    def test_progress_cannot_touch_the_lifecycle(self):
        reg = RunRegistry()
        env = reg.start("c", "plan")
        reg.note_progress(env["run_id"], status="succeeded", done=True, result={})
        got = reg.get(env["run_id"])
        assert got["status"] == RUNNING and got["done"] is False and got["result"] is None

    def test_progress_after_the_end_is_ignored(self):
        reg = RunRegistry()
        env = reg.start("c", "plan")
        reg.finish(env["run_id"], {"success": True})
        assert reg.note_progress(env["run_id"], progress={"percent": 1}) is None
        assert reg.get(env["run_id"])["status"] == SUCCEEDED


def summarised_tree():
    """bt_1790951179's shape: every leg an agent_view summary."""
    leg = {"waypoints": 119, "duration_s": 5.8, "start_joints": [0] * 6,
           "end_joints": [1] * 6}
    node = lambda n, r, k: {"type": "ExecuteTrajectory", "name": n, "id": f"id_{n}",
                            "ports": {"robot_ns": r, "operation_id": n,
                                      "hold_duration_s": 0.5,
                                      "trajectory": "{%s}" % k}}
    return {"tree_id": "bt_x", "cell_id": "c",
            "root": {"type": "Parallel", "id": "root",
                     "children": [node("W01", "robot1", "a"),
                                  node("W13", "robot2", "b")]},
            "blackboard": {"a": dict(leg), "b": dict(leg)}}


class TestEmptyPlanScoresZero:
    def test_a_tree_with_no_waypoints_scores_zero_and_says_why(self):
        r = tq.score_plan(summarised_tree())
        assert r["efficiency_score"] == 0.0
        assert r["verdict"] == tq.VERDICT_POOR
        assert r["findings"][0]["kind"] == "no_measurable_motion"


class TestAdvice:
    REPORT = {
        "tree_id": "bt_1", "cell_id": "c", "verdict": "acceptable",
        "efficiency_score": 0.8, "summary": {"makespan_s": 45.0},
        "findings": [
            {"severity": "medium", "kind": "low_parallelism", "detail": "pe 0.55"},
            {"severity": "high", "kind": "path_detour", "robot_ns": "robot1",
             "node": "weld_W01", "detail": "3 rad excess"},
        ],
    }
    TRACE = {
        "run": {"elapsed_s": 1000.0, "tool_calls": 130, "step_budget": 160},
        "points": [
            {"operation_id": "W01", "robot": "robot1", "attempts": 7,
             "flags": {"in_tree": True},
             "failures": [{"error": "OMPL (RRTConnect): the emitted trajectory "
                                    "lifts the tool 0.372 m above both endpoints"}]},
            {"operation_id": "W02", "attempts": 1,
             "failures": [{"error": "not planned: the batch stopped at step 0"}]},
        ],
        "decisions": [
            {"tool": "run_trajectory_planner",
             "result": {"metrics": {"planning_time_s": 100.0}}},
            {"tool": "validate_approach", "result": {"metrics": {}}},
        ],
    }

    def test_problems_are_ranked_and_each_known_kind_has_a_suggestion(self):
        a = plan_advice.advise(self.REPORT, self.TRACE)
        assert a["success"] is True and a["commanded"] is False
        assert a["problems"][0]["severity"] == "high"
        kinds = {p["kind"] for p in a["problems"]}
        assert {"path_detour", "low_parallelism", "hot_spot",
                "planner_failure:tool_lift", "planner_failure:batch_stopped",
                "step_budget", "time_in_reasoning"} <= kinds
        for p in a["problems"]:
            assert p["suggestion"], p

    def test_time_breakdown_separates_the_motion_planner(self):
        t = plan_advice.advise(self.REPORT, self.TRACE)["time_breakdown"]
        assert t["elapsed_s"] == 1000.0 and t["motion_planner_s"] == 100.0
        assert t["other_s"] == 900.0
        assert t["calls_by_tool"]["run_trajectory_planner"] == 1

    def test_overlapping_planner_time_leaves_other_unknown(self):
        trace = dict(self.TRACE, run={"elapsed_s": 50.0})
        assert plan_advice.advise(self.REPORT, trace)["time_breakdown"]["other_s"] is None

    def test_cycle_time_target_is_checked_when_a_program_is_given(self):
        program = {"extensions": {"cycle_time": {"target_s": 20.0, "max_s": 30.0}}}
        a = plan_advice.advise(self.REPORT, None, program)
        ct = [p for p in a["problems"] if p["kind"] == "cycle_time"]
        assert ct and ct[0]["severity"] == "high"

    def test_the_summary_is_one_line(self):
        s = plan_advice.advise(self.REPORT, self.TRACE)["summary"]
        assert "bt_1" in s and "\n" not in s


class TestPersistedTreeIsTheRealOne:
    @pytest.fixture()
    def agent(self):
        pytest.importorskip("langchain_core")
        from agents.trajectory import agent as agent_mod
        agent_mod._FULL_TREES.clear()
        return agent_mod

    @staticmethod
    def full_tree():
        t = summarised_tree()
        wp = [{"positions": [0.0] * 6, "time_s": 0.0},
              {"positions": [1.0] * 6, "time_s": 1.0}]
        t["blackboard"] = {"a": list(wp), "b": list(wp)}
        return t

    def test_the_summary_the_model_saw_maps_back_to_the_full_tree(self, agent):
        full = self.full_tree()
        agent._remember_full_tree({"success": True, "behaviour_tree": full})
        seen = agent.agent_view({"success": True, "behaviour_tree": full})
        msg = type("M", (), {"name": "assemble_behaviour_tree",
                             "content": json.dumps(seen)})()
        assert agent._extract_behaviour_tree([msg]) == full

    def test_an_unknown_summarised_tree_is_refused(self, agent):
        seen = agent.agent_view({"success": True, "behaviour_tree": self.full_tree()})
        msg = type("M", (), {"name": "assemble_behaviour_tree",
                             "content": json.dumps(seen)})()
        assert agent._extract_behaviour_tree([msg]) is None
