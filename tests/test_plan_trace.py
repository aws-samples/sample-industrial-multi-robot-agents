"""The planning log must answer "what did the planner decide about W12?".

These tests pin the properties that make the Plan Log tab trustworthy rather
than merely populated:

  * a decision is paired with its OWN result, by id — pairing by arrival order
    silently mislabels the welds of a parallel step;
  * a weld nobody ever attempted still appears, because a silently skipped weld
    is the finding, and a view of only-what-was-attempted hides it;
  * a point's status is the LAST verdict, not the best one — "planned then
    rejected" is not "planned";
  * the run's closing reasoning survives even when it came with no tool call,
    which is exactly the message that used to be dropped;
  * a truncated run's unanswered call is labelled as unanswered, not as a tool
    that returned nothing;
  * waypoint arrays are summarised, so a 15-weld trace stays a file a browser
    can hold;
  * a trace id cannot escape the logs directory.

Host-only: no ROS, no LLM, no container.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.trajectory import plan_trace  # noqa: E402


# ── doubles ───────────────────────────────────────────────────────────────────

class AI:
    """An AIMessage as the trace reader sees it: text plus tool decisions."""

    type = "ai"

    def __init__(self, text="", tool_calls=None):
        self.content = text
        self.tool_calls = tool_calls or []


class Tool:
    """A ToolMessage: a JSON payload tagged with the call it answers."""

    type = "tool"

    def __init__(self, call_id, payload):
        self.tool_call_id = call_id
        self.content = payload if isinstance(payload, str) else json.dumps(payload)


def call(cid, name, args):
    return {"id": cid, "name": name, "args": args}


OPS = [
    {"id": "W01", "type": "spot_weld", "position": [1.0, 0.2, 0.9], "robot_ns": "robot1"},
    {"id": "W02", "type": "spot_weld", "position": [1.0, 0.4, 0.9], "robot_ns": "robot2"},
    {"id": "W03", "type": "spot_weld", "position": [1.0, 0.6, 0.9], "robot_ns": "robot2"},
]


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """A cell tree whose logs dir is disposable."""
    base = tmp_path / "cells"
    (base / "c1" / "agentic_planner" / "logs").mkdir(parents=True)
    monkeypatch.setenv("CELL_BASE_PATH", str(base))
    return base / "c1" / "agentic_planner" / "logs"


# ── pairing ───────────────────────────────────────────────────────────────────

def test_a_result_is_paired_with_its_own_call_not_the_next_one():
    """Two calls in one step, results returning reversed. Each keeps its own."""
    messages = [
        AI("Planning both arms at once.", [
            call("a", "run_trajectory_planner", {"motion_requests": [{"operation_id": "W01"}]}),
            call("b", "run_trajectory_planner", {"motion_requests": [{"operation_id": "W02"}]}),
        ]),
        Tool("b", {"success": False, "error": "W02 unreachable"}),
        Tool("a", {"success": True, "planning_time_s": 1.5}),
    ]
    decisions = plan_trace._decisions(messages)
    by_op = {d["operations"][0]: d for d in decisions}
    assert by_op["W01"]["result"]["ok"] is True
    assert by_op["W02"]["result"]["ok"] is False


def test_the_reasoning_before_a_call_is_attached_to_it():
    messages = [
        AI("W01 first because robot1 is closer.", [call("a", "run_ik", {"operation_id": "W01"})]),
        Tool("a", {"success": True}),
    ]
    assert "robot1 is closer" in plan_trace._decisions(messages)[0]["why"]


def test_reasoning_with_no_tool_call_is_kept_as_a_note():
    """The give-up message. Dropping it is what made runs look unexplained."""
    messages = [
        AI("Trying W01.", [call("a", "run_ik", {"operation_id": "W01"})]),
        Tool("a", {"success": False}),
        AI("I cannot reach W01, so I am not assembling a tree."),
    ]
    decisions = plan_trace._decisions(messages)
    notes = [d for d in decisions if d["kind"] == "note"]
    assert len(notes) == 1
    assert "not assembling a tree" in notes[0]["text"]


def test_an_unanswered_call_is_labelled_unanswered():
    """A truncated run: the call was emitted, no ToolMessage ever came back."""
    messages = [AI("Planning.", [call("a", "run_trajectory_planner",
                                      {"motion_requests": [{"operation_id": "W01"}]})])]
    record = plan_trace._decisions(messages)[0]
    assert record["result"]["unanswered"] is True
    assert record["result"]["ok"] is None


def test_step_numbers_count_tool_calls_and_ignore_notes():
    messages = [
        AI("a", [call("1", "t", {})]),
        Tool("1", {"success": True}),
        AI("thinking out loud"),
        AI("b", [call("2", "t", {})]),
        Tool("2", {"success": True}),
    ]
    steps = [d["step"] for d in plan_trace._decisions(messages) if d["kind"] == "decision"]
    assert steps == [1, 2]


# ── identity ──────────────────────────────────────────────────────────────────

def test_operation_ids_are_found_at_every_depth_each_tool_uses():
    """Top level, nested per-request, and as a list — one walk, all three."""
    assert plan_trace.operation_ids({"operation_id": "W01"}) == ["W01"]
    assert plan_trace.operation_ids(
        {"motion_requests": [{"operation_id": "W02"}, {"operation_id": "W03"}]}
    ) == ["W02", "W03"]
    assert plan_trace.operation_ids({"operation_ids": ["W04", "W05"]}) == ["W04", "W05"]


def test_robot_names_are_collected_from_args_and_results():
    assert plan_trace.robot_names({"robot_ns": "robot1"}) == ["robot1"]
    assert plan_trace.robot_names({"trajectories": {"robot2": [], "robot1": []}}) \
        == ["robot2", "robot1"]


# ── points ────────────────────────────────────────────────────────────────────

def test_every_task_point_appears_even_if_never_attempted():
    """W03 is untouched. It must still be listed, as pending."""
    messages = [
        AI("W01.", [call("a", "run_trajectory_planner",
                         {"motion_requests": [{"operation_id": "W01"}]})]),
        Tool("a", {"success": True}),
    ]
    points = plan_trace._index_points(OPS, plan_trace._decisions(messages))
    assert [p["operation_id"] for p in points] == ["W01", "W02", "W03"]
    assert {p["operation_id"]: p["status"] for p in points}["W03"] == "pending"


def test_status_is_the_last_verdict_not_the_best_one():
    """Planned, then rejected by the assembler. The point is failed."""
    messages = [
        AI("Plan W01.", [call("a", "run_trajectory_planner",
                              {"motion_requests": [{"operation_id": "W01"}]})]),
        Tool("a", {"success": True, "planning_time_s": 2.0}),
        AI("Assemble.", [call("b", "assemble_behaviour_tree", {"operation_ids": ["W01"]})]),
        Tool("b", {"success": False, "failures": [{"operation_id": "W01",
                                                   "error": "no trajectory cached"}]}),
    ]
    point = plan_trace._index_points(OPS, plan_trace._decisions(messages))[0]
    assert point["status"] == "failed"
    assert point["flags"].get("planned") is True   # the history is still there
    assert point["failures"]


def test_a_validated_then_untouched_point_stays_validated():
    messages = [
        AI("Check W01.", [call("a", "validate_approach", {"operation_id": "W01"})]),
        Tool("a", {"success": True, "matches": True, "tool_axis_error_deg": 0.6}),
    ]
    point = plan_trace._index_points(OPS, plan_trace._decisions(messages))[0]
    assert point["status"] == "validated"
    assert point["metrics"]["tool_axis_error_deg"] == 0.6


def test_an_operation_the_task_never_listed_is_surfaced_not_dropped():
    """The planner inventing an id is a finding, not noise to filter."""
    messages = [
        AI("Plan W99.", [call("a", "run_trajectory_planner",
                              {"motion_requests": [{"operation_id": "W99"}]})]),
        Tool("a", {"success": True}),
    ]
    points = plan_trace._index_points(OPS, plan_trace._decisions(messages))
    extra = [p for p in points if p["operation_id"] == "W99"]
    assert extra and extra[0]["flags"].get("unlisted") is True


def test_attempts_count_every_decision_that_touched_the_point():
    messages = [
        AI("", [call("a", "run_ik", {"operation_id": "W01"})]),
        Tool("a", {"success": True}),
        AI("", [call("b", "validate_approach", {"operation_id": "W01"})]),
        Tool("b", {"success": True, "matches": True}),
    ]
    point = plan_trace._index_points(OPS, plan_trace._decisions(messages))[0]
    assert point["attempts"] == 2
    assert point["decisions"] == [1, 2]


# ── bulk ──────────────────────────────────────────────────────────────────────

def test_waypoint_arrays_are_summarised_not_carried():
    traj = [{"positions": [0.0] * 6, "time_s": i * 0.1} for i in range(80)]
    out = plan_trace.compact({"trajectories": {"robot1": traj}})
    assert out["trajectories"]["robot1"] == {"waypoints": 80, "duration_s": 7.9}


def test_a_long_string_is_marked_where_it_was_cut():
    out = plan_trace.compact("x" * (plan_trace._MAX_TEXT_CHARS + 50))
    assert "more chars" in out


def test_a_planner_result_keeps_its_headline_numbers():
    metrics = plan_trace._result_metrics(
        {"success": True, "planning_time_s": 3.2, "min_clearance_m": 0.05,
         "irrelevant": "x" * 10, "trajectories": {"robot1": [1, 2, 3]}})
    assert metrics["planning_time_s"] == 3.2
    assert metrics["min_clearance_m"] == 0.05
    assert metrics["waypoints"] == {"robot1": 3}
    assert "irrelevant" not in metrics


# ── the whole trace ───────────────────────────────────────────────────────────

def test_build_reports_the_run_and_both_views():
    messages = [
        AI("Plan W01.", [call("a", "run_trajectory_planner",
                              {"motion_requests": [{"operation_id": "W01"}]})]),
        Tool("a", {"success": True}),
    ]
    trace = plan_trace.build(
        "c1", messages, OPS,
        {"success": False, "error_kind": "truncated", "error": "cut off",
         "note": "raise LLM_MAX_TOKENS", "planning_time_s": 12.5},
        trace_id="t1", backend="ompl", model="m", step_budget=60)
    run = trace["run"]
    assert trace["schema"] == plan_trace.SCHEMA
    assert run["trace_id"] == "t1" and run["cell_id"] == "c1"
    assert run["success"] is False and run["error_kind"] == "truncated"
    assert run["note"] == "raise LLM_MAX_TOKENS"
    assert run["tool_calls"] == 1 and run["step_budget"] == 60
    assert run["operations"] == 3
    assert run["points_by_status"] == {"planned": 1, "pending": 2}
    assert len(trace["points"]) == 3 and len(trace["decisions"]) == 1


def test_a_failed_run_is_recorded_too(logs):
    """The runs worth reading are the ones that failed."""
    saved = plan_trace.capture("c1", [], OPS, {"success": False, "error": "nope"},
                               trace_id="fail1")
    assert saved and Path(saved).is_file()
    assert plan_trace.read("c1", "fail1")["run"]["error"] == "nope"


def test_the_history_is_newest_first_and_metadata_only(logs):
    for tid in ("c1_20260101_000000", "c1_20260102_000000"):
        plan_trace.capture("c1", [], OPS, {"success": True}, trace_id=tid)
    history = plan_trace.listing("c1")
    assert [h["trace_id"] for h in history] == ["c1_20260102_000000", "c1_20260101_000000"]
    assert history[0]["points"] == 3 and "decisions" in history[0]
    assert "points_by_status" in history[0]


def test_listing_an_unplanned_cell_is_empty_not_an_error(logs):
    assert plan_trace.listing("no_such_cell") == []


def test_capture_never_raises_into_the_planner(monkeypatch):
    """A lost trace must never cost the plan."""
    monkeypatch.setattr(plan_trace.cell_layout, "logs_dir",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert plan_trace.capture("c1", [], OPS, {"success": True}, trace_id="x") is None


# ── path safety ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "", "..", "x\x00y", "a b"])
def test_a_trace_id_cannot_escape_the_logs_directory(bad):
    assert plan_trace.is_safe_id(bad) is False
    assert plan_trace.read("c1", bad) is None


def test_a_normal_trace_id_is_accepted():
    assert plan_trace.is_safe_id("welding_cell_2_small_20260908_171725") is True


def test_reading_a_missing_trace_is_none_not_an_exception(logs):
    assert plan_trace.read("c1", "never_written") is None
