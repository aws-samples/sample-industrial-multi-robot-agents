"""A planning run must survive losing its place.

Twice in one afternoon a 15-weld plan was thrown away whole — once because the
AWS token expired at step 18, once because a Bedrock socket timed out at 2025 s —
and each time the only recovery was to replan from W01. Nothing was wrong with the
welds. These tests pin the properties that make "continue where it stopped" real
rather than nominal:

  * a checkpoint is only ever taken at a boundary a provider will accept, so a
    resume cannot be rejected for a ``toolUse`` with no ``toolResult``;
  * the trajectories already solved come back with the conversation, because a
    resume that restored only the messages would re-solve everything while telling
    the model it had not — the one outcome worse than starting over;
  * a half-written checkpoint never exists;
  * a failed run's computed work is KEPT, listed, and readable — and is never
    handed to anything that would drive a robot;
  * the live feed's sequence continues across a resume, so a poller does not skip
    the resumed half;
  * "the system changed strategy" is evidence (the changed argument), not prose;
  * neither a checkpoint id nor a trace id can escape the logs directory.

Host-only: no ROS, no LLM, no container.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.trajectory import plan_checkpoint, plan_events        # noqa: E402
from agents.trajectory.tools import plan_store, trajectory_cache  # noqa: E402


OPS = [
    {"id": "W01", "type": "spot_weld", "position": [1.0, 0.2, 0.9]},
    {"id": "W02", "type": "spot_weld", "position": [1.0, 0.4, 0.9]},
]

TRAJ = [{"positions": [0.0] * 6, "time_s": 0.0},
        {"positions": [0.1] * 6, "time_s": 1.0}]


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """A disposable cell tree, with the caches emptied around each test.

    The trajectory cache is process-global (the planner's tools reach it without
    passing it around), so a test that leaves entries behind would hand the next
    one a resume that "worked" on someone else's data.
    """
    base = tmp_path / "cells"
    (base / "c1" / "agentic_planner" / "logs").mkdir(parents=True)
    (base / "c1" / "agentic_planner" / "plans").mkdir(parents=True)
    monkeypatch.setenv("CELL_BASE_PATH", str(base))
    trajectory_cache.clear_trajectory_cache("c1")
    trajectory_cache.reset_robot_configs("c1")
    plan_events.forget("t1")
    yield "c1"
    trajectory_cache.clear_trajectory_cache("c1")
    trajectory_cache.reset_robot_configs("c1")
    plan_events.forget("t1")


# ── doubles ───────────────────────────────────────────────────────────────────

class AI:
    """An AIMessage carrying tool decisions, as the checkpoint reader sees it."""

    type = "ai"

    def __init__(self, text="", tool_calls=None):
        self.content = text
        self.tool_calls = tool_calls or []


class Tool:
    """A ToolMessage: the answer to one call."""

    type = "tool"
    tool_calls = None

    def __init__(self, call_id, payload="{}"):
        self.tool_call_id = call_id
        self.content = payload


class Human:
    type = "human"
    tool_calls = None

    def __init__(self, text=""):
        self.content = text


def call(cid, name="run_trajectory_planner", args=None):
    return {"id": cid, "name": name, "args": args or {}}


# ── the resume boundary ───────────────────────────────────────────────────────

def test_a_clean_boundary_is_kept_whole():
    """Every call answered: nothing is dropped."""
    messages = [Human("plan"), AI("planning W01", [call("a")]), Tool("a")]
    prefix, dropped = plan_checkpoint.resumable_prefix(messages)
    assert dropped == 0
    assert len(prefix) == 3


def test_an_unanswered_step_is_trimmed_not_resumed_from():
    """A history ending in an unanswered tool_use is what the provider rejects.

    This is the shape a run dies in: the model asked for a solve and the socket
    dropped before the answer. Resuming from it would fail at the first request.
    """
    messages = [Human("plan"), AI("W01", [call("a")]), Tool("a"),
                AI("W02", [call("b")])]
    prefix, dropped = plan_checkpoint.resumable_prefix(messages)
    assert dropped == 1
    assert prefix[-1] is messages[2]
    assert plan_checkpoint._unanswered_tool_calls(prefix) == 0


def test_a_partially_answered_step_is_trimmed_entirely():
    """Two calls in one step, one answered. The whole step goes.

    Keeping the answered half would leave the other call unanswered, which is the
    same rejection with extra steps.
    """
    messages = [Human("plan"), AI("both arms", [call("a"), call("b")]), Tool("a")]
    prefix, dropped = plan_checkpoint.resumable_prefix(messages)
    assert prefix == [messages[0]]
    assert dropped == 2


def test_nothing_resumable_yields_an_empty_prefix():
    """A run that died inside its first step has no boundary to return to."""
    messages = [AI("W01", [call("a")])]
    prefix, dropped = plan_checkpoint.resumable_prefix(messages)
    assert prefix == []
    assert dropped == 1


def test_a_trimmed_run_costs_one_step_not_the_run():
    """Fourteen welds decided, the fifteenth in flight: fourteen are kept."""
    messages = [Human("plan")]
    for i in range(14):
        messages += [AI(f"W{i:02d}", [call(f"c{i}")]), Tool(f"c{i}")]
    messages.append(AI("W14", [call("c14")]))
    prefix, dropped = plan_checkpoint.resumable_prefix(messages)
    assert dropped == 1
    assert len(prefix) == 29


# ── save and restore ──────────────────────────────────────────────────────────

def test_a_checkpoint_restores_the_conversation_and_the_solved_work(cell):
    """The whole point: the messages AND the trajectories come back together."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    trajectory_cache.cache_trajectory(cell, "robot1", "W01", TRAJ)
    trajectory_cache.set_last_config(cell, "robot1", [0.1] * 6)
    messages = [HumanMessage(content="plan"),
                AIMessage(content="W01", tool_calls=[call("a")]),
                ToolMessage(content="{\"success\": true}", tool_call_id="a")]

    assert plan_checkpoint.save(cell, "t1", messages=messages, operations=OPS,
                                constraints={"max_parallel": 2}, step=7,
                                backend="ompl", model="opus-5", reason="token expired")

    # A fresh session, as a recreated container would be.
    trajectory_cache.clear_trajectory_cache(cell)
    trajectory_cache.reset_robot_configs(cell)
    assert trajectory_cache.get_cached_trajectory(cell, "robot1", "W01") is None

    resumed = plan_checkpoint.restore(cell, "t1")
    assert resumed is not None
    assert resumed["step"] == 7
    assert resumed["operations"] == OPS
    assert resumed["constraints"] == {"max_parallel": 2}
    assert len(resumed["messages"]) == 3
    assert resumed["cache_restored"]["trajectories"] == 1
    assert trajectory_cache.get_cached_trajectory(cell, "robot1", "W01") == TRAJ
    assert trajectory_cache.get_last_config(cell, "robot1") == [0.1] * 6


def test_the_checkpoint_stores_the_trimmed_prefix_not_the_raw_history(cell):
    """What the file says is exactly what a resume will do."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages = [HumanMessage(content="plan"),
                AIMessage(content="W01", tool_calls=[call("a")]),
                ToolMessage(content="{}", tool_call_id="a"),
                AIMessage(content="W02", tool_calls=[call("b")])]
    plan_checkpoint.save(cell, "t1", messages=messages, operations=OPS,
                         constraints=None, step=2)
    record = plan_checkpoint.load_checkpoint(cell, "t1")
    assert record["discarded_messages"] == 1
    assert len(record["messages"]) == 3


def test_a_run_with_no_completed_step_is_not_marked_resumable(cell):
    """Refusing beats a "resume" that silently replans from the first weld."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    plan_checkpoint.save(cell, "t1",
                         messages=[AIMessage(content="W01", tool_calls=[call("a")])],
                         operations=OPS, constraints=None, step=0)
    assert plan_checkpoint.describe(cell, "t1")["resumable"] is False
    assert plan_checkpoint.restore(cell, "t1") is None


def test_a_run_that_died_before_its_first_step_is_not_resumable(cell):
    """Observed live: the token expired before the model's first reply.

    That history is a valid conversation — just the opening prompt — so it passes
    the boundary check while preserving nothing. Continuing it and replanning from
    the first weld are the same run, and offering "resume" for a restart is the
    surprise this mechanism exists to remove. One answered tool call is the bar.
    """
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    plan_checkpoint.save(cell, "t1", messages=[HumanMessage(content="plan")],
                         operations=OPS, constraints=None, step=0,
                         reason="aborted: credentials")
    # The record is kept — it says what happened — but it is not offered as a resume.
    assert plan_checkpoint.load_checkpoint(cell, "t1") is not None
    assert plan_checkpoint.describe(cell, "t1")["resumable"] is False
    assert plan_checkpoint.restore(cell, "t1") is None
    assert plan_checkpoint.listing(cell)[0]["resumable"] is False


def test_a_stale_checkpoint_claiming_resumable_is_still_refused(cell):
    """Observed live: a file from an older build carried resumable: true.

    The flag is a cache of a property of the messages, so a checkpoint that
    outlived the code that wrote it can claim more than it holds. Recomputed at
    restore time, where the messages are decoded anyway.
    """
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    plan_checkpoint.save(cell, "t1", messages=[HumanMessage(content="plan")],
                         operations=OPS, constraints=None, step=0)
    path = plan_checkpoint.path_for(cell, "t1")
    record = json.loads(path.read_text())
    record["resumable"] = True          # what the older writer would have stored
    path.write_text(json.dumps(record))

    assert plan_checkpoint.describe(cell, "t1")["resumable"] is True   # the file's claim
    assert plan_checkpoint.restore(cell, "t1") is None                 # the truth


def test_the_checkpoint_is_written_atomically(cell):
    """No reader ever sees a half-written checkpoint."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    plan_checkpoint.save(cell, "t1", messages=[HumanMessage(content="plan")],
                         operations=OPS, constraints=None, step=1)
    logs = Path(os.environ["CELL_BASE_PATH"]) / "c1" / "agentic_planner" / "logs"
    assert (logs / "checkpoint_t1.json").is_file()
    assert not list(logs.glob("*.tmp"))


def test_a_finished_run_leaves_no_checkpoint(cell):
    """A resumable checkpoint must only exist while a run has somewhere to go."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    plan_checkpoint.save(cell, "t1", messages=[HumanMessage(content="plan")],
                         operations=OPS, constraints=None, step=1)
    assert plan_checkpoint.discard(cell, "t1") is True
    assert plan_checkpoint.describe(cell, "t1") is None
    assert plan_checkpoint.restore(cell, "t1") is None


def test_the_cell_holding_a_checkpoint_is_discoverable(cell):
    """A resume happens in a NEW process — the job registry that knew is gone."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    plan_checkpoint.save(cell, "t1", messages=[HumanMessage(content="plan")],
                         operations=OPS, constraints=None, step=1)
    assert plan_checkpoint.find_cell("t1") == cell
    assert plan_checkpoint.find_cell("nope") is None


def test_a_checkpoint_id_cannot_escape_the_logs_directory(cell):
    assert plan_checkpoint.is_safe_id("../../etc/passwd") is False
    assert plan_checkpoint.save(cell, "../evil", messages=[], operations=[],
                                constraints=None, step=0) is None
    assert plan_checkpoint.load_checkpoint(cell, "../evil") is None
    assert plan_checkpoint.discard(cell, "../evil") is False


def test_a_missing_checkpoint_is_an_answer_not_an_error(cell):
    assert plan_checkpoint.load_checkpoint(cell, "never-ran") is None
    assert plan_checkpoint.describe(cell, "never-ran") is None
    assert plan_checkpoint.listing(cell) == []


def test_saving_never_raises_into_the_planner(cell, monkeypatch):
    """A checkpoint that cannot be written is a resume that is not available."""
    monkeypatch.setattr(plan_checkpoint, "path_for",
                        lambda c, t: Path("/proc/nope/checkpoint.json"))
    assert plan_checkpoint.save(cell, "t1", messages=[], operations=OPS,
                                constraints=None, step=1) is None


def test_listing_reports_what_a_resume_would_recover(cell):
    """`trajectories` is the number an operator weighs the resume against."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage

    for op in ("W01", "W02", "W03"):
        trajectory_cache.cache_trajectory(cell, "robot1", op, TRAJ)
    plan_checkpoint.save(cell, "t1", messages=[HumanMessage(content="plan")],
                         operations=OPS, constraints=None, step=9,
                         reason="paused by operator")
    items = plan_checkpoint.listing(cell)
    assert len(items) == 1
    assert items[0]["trajectories"] == 3
    assert items[0]["step"] == 9
    assert items[0]["reason"] == "paused by operator"


# ── the pause request ─────────────────────────────────────────────────────────

def test_pause_is_requested_then_taken_once():
    plan_checkpoint.clear_pause("t1")
    assert plan_checkpoint.pause_requested("t1") is None
    plan_checkpoint.request_pause("t1", reason="token expiring")
    assert plan_checkpoint.pause_requested("t1") == "token expiring"
    # Checking must not consume it — the run checks at every state boundary.
    assert plan_checkpoint.pause_requested("t1") == "token expiring"
    assert plan_checkpoint.take_pause("t1") == "token expiring"
    # And a resumed run must not immediately pause again on a stale request.
    assert plan_checkpoint.pause_requested("t1") is None


def test_pausing_an_idle_stack_is_not_an_error():
    result = plan_checkpoint.request_pause(None, reason="operator")
    assert result["success"] is True
    assert isinstance(result["paused"], list)


# ── the live feed ─────────────────────────────────────────────────────────────

def test_events_are_appended_one_line_each_and_readable_at_once(cell):
    plan_events.run_started(cell, "t1", operations=2, backend="ompl",
                            model="opus-5", step_budget=100)
    plan_events.operation_started(cell, "t1", "W01", robot="robot1")
    feed = plan_events.read(cell, "t1")
    assert [e["kind"] for e in feed["events"]] == ["run_started", "operation_started"]
    assert feed["next_seq"] == 2
    assert feed["running"] is True
    raw = plan_events.path_for(cell, "t1").read_text().strip().split("\n")
    assert len(raw) == 2 and all(json.loads(line)["seq"] for line in raw)


def test_polling_with_since_returns_only_what_is_new(cell):
    plan_events.run_started(cell, "t1", operations=2, backend="ompl",
                            model="opus-5", step_budget=100)
    first = plan_events.read(cell, "t1")
    plan_events.operation_started(cell, "t1", "W02")
    second = plan_events.read(cell, "t1", since=first["next_seq"])
    assert [e["operation_id"] for e in second["events"]] == ["W02"]


def test_the_sequence_continues_across_a_resume(cell):
    """A restarted writer must not restart the seq.

    Numbering from 1 again would make a reader polling ``?since=3`` skip
    everything the resumed half emits — the feed would go silent exactly when the
    recovery it is there to show is happening.
    """
    plan_events.run_started(cell, "t1", operations=2, backend="ompl",
                            model="opus-5", step_budget=100)
    plan_events.operation_started(cell, "t1", "W01")
    plan_events.forget("t1")          # a new process, same stream
    plan_events.resumed(cell, "t1", step=1, discarded=1)
    seqs = [e["seq"] for e in plan_events.read(cell, "t1")["events"]]
    assert seqs == [1, 2, 3]


def test_a_terminal_event_means_the_run_is_not_running(cell):
    plan_events.run_started(cell, "t1", operations=1, backend="ompl",
                            model="opus-5", step_budget=100)
    assert plan_events.read(cell, "t1")["running"] is True
    plan_events.paused(cell, "t1", reason="operator", step=4)
    assert plan_events.read(cell, "t1")["running"] is False


def test_a_slow_reader_is_shown_the_present_not_a_stale_window(cell):
    for i in range(10):
        plan_events.operation_started(cell, "t1", f"W{i:02d}")
    feed = plan_events.read(cell, "t1", limit=3)
    assert feed["dropped"] == 7
    assert [e["operation_id"] for e in feed["events"]] == ["W07", "W08", "W09"]


def test_a_repeat_with_a_changed_argument_is_a_strategy_change(cell):
    """"The system changed strategy" carries its evidence.

    This is the W14 case from the live run: the same weld, re-attempted with the
    standoff shrunk. The reader is told WHAT changed, not merely that something did.
    """
    assert plan_events.note_attempt(cell, "t1", "W14",
                                    {"approach_offset_m": 0.10}) is None
    event = plan_events.note_attempt(cell, "t1", "W14",
                                     {"approach_offset_m": 0.07})
    assert event["kind"] == "strategy"
    assert event["changed"] == {"approach_offset_m": [0.10, 0.07]}
    assert "0.1 → 0.07" in event["text"]


def test_a_repeat_with_the_same_arguments_is_not_a_strategy_change(cell):
    plan_events.note_attempt(cell, "t1", "W14", {"approach_offset_m": 0.10})
    assert plan_events.note_attempt(cell, "t1", "W14",
                                    {"approach_offset_m": 0.10}) is None


def test_bookkeeping_that_changes_every_call_is_not_a_strategy_change(cell):
    """A diff over every field would call a new waypoint count a strategy."""
    plan_events.note_attempt(cell, "t1", "W14",
                             {"approach_offset_m": 0.1, "waypoints": 40})
    assert plan_events.note_attempt(cell, "t1", "W14",
                                    {"approach_offset_m": 0.1,
                                     "waypoints": 55}) is None


def test_planned_and_validated_are_different_events(cell):
    """A trajectory existing and the tool arriving at the right angle differ."""
    plan_events.operation_result(cell, "t1", "W01", ok=True,
                                 tool="run_trajectory_planner")
    plan_events.operation_result(cell, "t1", "W01", ok=True,
                                 tool="validate_approach")
    plan_events.operation_result(cell, "t1", "W02", ok=False,
                                 tool="validate_approach", error="too_close")
    kinds = [e["kind"] for e in plan_events.read(cell, "t1")["events"]]
    assert kinds == ["operation_planned", "operation_validated", "operation_failed"]


def test_the_newest_stream_is_the_default_to_watch(cell):
    """A dashboard not told which run to watch wants the current one."""
    plan_events.emit(cell, "older", "run_started")
    plan_events.emit(cell, "t1", "run_started")
    # Set the mtimes explicitly: written a millisecond apart, the two files tie at
    # the filesystem's granularity, and a test that depends on which side of a tick
    # it landed on proves nothing either way.
    os.utime(plan_events.path_for(cell, "older"), (1000, 1000))
    os.utime(plan_events.path_for(cell, "t1"), (2000, 2000))
    assert plan_events.latest_trace_id(cell) == "t1"
    assert plan_events.read(cell)["trace_id"] == "t1"
    plan_events.forget("older")


def test_reading_a_cell_that_never_planned_is_an_answer_not_an_error(cell):
    feed = plan_events.read(cell, "no-such-run")
    assert feed["events"] == []
    assert feed["running"] is False


def test_a_trace_id_cannot_escape_the_logs_directory(cell):
    assert plan_events.is_safe_id("../../etc/passwd") is False
    assert plan_events.emit(cell, "../evil", "run_started") is None
    assert plan_events.read(cell, "../evil")["events"] == []


def test_emitting_never_raises_into_the_planner(cell, monkeypatch):
    monkeypatch.setattr(plan_events, "path_for",
                        lambda c, t: Path("/proc/nope/events.jsonl"))
    assert plan_events.emit(cell, "t1", "run_started") is None


def test_a_long_event_field_stays_one_short_line(cell):
    plan_events.thinking(cell, "t1", "x" * 5000)
    text = plan_events.read(cell, "t1")["events"][0]["text"]
    assert len(text) <= 401 and text.endswith("…")


# ── failed plans ──────────────────────────────────────────────────────────────

def test_a_failed_run_keeps_the_trajectories_it_did_compute(cell):
    """Thirteen solves are hours of work. They are kept, and they are readable."""
    plan_store.save_failed_plan(
        cell, "bt_fail", error="Read timeout on endpoint URL",
        error_kind="network", note="infrastructure", operations=OPS,
        trajectories=[{"robot_ns": "robot1", "operation_id": "W01",
                       "trajectory": TRAJ}],
        resumable=True)
    record = plan_store.get_failed_plan(cell, "bt_fail")
    assert record["status"] == plan_store.STATUS_FAILED
    assert record["error_kind"] == "network"
    assert record["resumable"] is True
    assert record["num_planned"] == 1
    assert record["planned"][0] == {"robot_ns": "robot1", "operation_id": "W01",
                                    "waypoints": 2}


def test_a_failed_run_is_listed_beside_the_real_plans(cell):
    plan_store.save_plan(cell, {"tree_id": "bt_ok", "cell_id": cell, "root": {}})
    plan_store.save_failed_plan(cell, "bt_fail", error="boom",
                                error_kind="planning", operations=OPS)
    listed = {p["tree_id"]: p for p in plan_store.list_plans(cell)}
    assert listed["bt_ok"]["status"] == plan_store.STATUS_OK
    assert listed["bt_fail"]["status"] == plan_store.STATUS_FAILED
    assert listed["bt_fail"]["error"] == "boom"


def test_a_failed_plan_is_never_the_latest_plan(cell):
    """The fail-closed half of storing them at all.

    ``get_plan`` results are fed straight to ``execute_behaviour_tree``, so "the
    newest plan" resolving to a run that never assembled a valid one is how a
    partial plan reaches a robot.
    """
    plan_store.save_plan(cell, {"tree_id": "bt_ok", "cell_id": cell, "root": {}})
    plan_store.save_failed_plan(cell, "bt_fail", error="boom", operations=OPS,
                                behaviour_tree={"tree_id": "bt_fail", "root": {}})
    latest = plan_store.get_plan(cell)
    assert latest["tree_id"] == "bt_ok"


def test_a_failed_plan_is_refused_by_id_unless_asked_for_explicitly(cell):
    plan_store.save_failed_plan(cell, "bt_fail", error="boom", operations=OPS,
                                behaviour_tree={"tree_id": "bt_fail", "root": {}})
    assert plan_store.get_plan(cell, "bt_fail") is None
    assert plan_store.get_plan(cell, "bt_fail",
                               include_failed=True)["tree_id"] == "bt_fail"


def test_get_failed_plan_refuses_a_successful_one(cell):
    """The two readers do not overlap: each answers only about its own kind."""
    plan_store.save_plan(cell, {"tree_id": "bt_ok", "cell_id": cell, "root": {}})
    assert plan_store.get_failed_plan(cell, "bt_ok") is None


def test_a_plan_saved_before_the_status_field_existed_reads_as_ok(cell):
    """It was ok — the store only ever held assembled trees at that point."""
    path = Path(os.environ["CELL_BASE_PATH"]) / "c1" / "agentic_planner" / "plans"
    (path / "bt_legacy.json").write_text(json.dumps({
        "tree_id": "bt_legacy", "cell_id": cell, "created_s": 1.0,
        "behaviour_tree": {"tree_id": "bt_legacy", "root": {}}}))
    assert plan_store.list_plans(cell)[0]["status"] == plan_store.STATUS_OK
    assert plan_store.get_plan(cell, "bt_legacy")["tree_id"] == "bt_legacy"


def test_a_failed_plan_is_written_atomically(cell):
    plan_store.save_failed_plan(cell, "bt_fail", error="boom", operations=OPS)
    plans = Path(os.environ["CELL_BASE_PATH"]) / "c1" / "agentic_planner" / "plans"
    assert (plans / "bt_fail.json").is_file()
    assert not list(plans.glob("*.tmp"))


def test_saving_a_failed_plan_never_raises(cell):
    assert plan_store.save_failed_plan(cell, None, error="boom") is None


# ── the trajectory-cache snapshot ─────────────────────────────────────────────

def test_the_snapshot_round_trips_through_json(cell):
    """It travels in a JSON file, so its tuple keys have to be flattened."""
    trajectory_cache.cache_trajectory(cell, "robot1", "W01", TRAJ)
    trajectory_cache.set_last_config(cell, "robot2", [0.2] * 6)
    snapshot = json.loads(json.dumps(trajectory_cache.snapshot(cell)))
    trajectory_cache.clear_trajectory_cache(cell)
    trajectory_cache.reset_robot_configs(cell)
    counts = trajectory_cache.restore(cell, snapshot)
    assert counts["trajectories"] == 1
    assert counts["last_config"] == 1
    assert trajectory_cache.get_cached_trajectory(cell, "robot1", "W01") == TRAJ


def test_a_malformed_record_costs_one_trajectory_not_the_resume(cell):
    """Recovering 13 of 14 still saves 13 solves."""
    counts = trajectory_cache.restore(cell, {"trajectories": [
        {"robot_ns": "robot1", "operation_id": "W01", "trajectory": TRAJ},
        {"robot_ns": None, "operation_id": "W02", "trajectory": TRAJ},
        {"robot_ns": "robot1", "operation_id": "W03", "trajectory": []},
    ]})
    assert counts["trajectories"] == 1


def test_the_snapshot_is_scoped_to_its_own_cell(cell):
    """Restoring another cell's leftovers is how a stale trajectory gets chained."""
    trajectory_cache.cache_trajectory(cell, "robot1", "W01", TRAJ)
    trajectory_cache.cache_trajectory("other_cell", "robot1", "W99", TRAJ)
    ops = {r["operation_id"] for r in trajectory_cache.snapshot(cell)["trajectories"]}
    assert ops == {"W01"}
    trajectory_cache.clear_trajectory_cache("other_cell")


def test_restoring_junk_is_a_no_op(cell):
    assert trajectory_cache.restore(cell, None) == {
        "trajectories": 0, "last_config": 0, "last_pose": 0}
