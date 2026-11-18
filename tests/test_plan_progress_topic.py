"""The planner's progress goes out on DDS, and the dashboard listens for it there.

Before this, the only way to see inside a live planning run was to poll the
orchestrator, which read the planner's event file off a shared volume: three
processes and a filesystem in the path of a question about *now*. The planner is
already in the ROS graph, so it publishes each plan event on a topic of its own —
``/<cell_id>/plan_progress`` — and the dashboard subscribes.

What is pinned here:

* the topic name, composed once on each side and therefore able to drift;
* the payload's self-describing envelope (``schema``, ``cell_id``, ``trace_id``),
  because a subscriber has to know which run it is watching;
* that publishing is *best effort*: no ``rclpy``, no middleware, a dead publisher —
  the run continues and the durable event stream is unaffected. Losing the live
  view is an availability failure; losing the plan because nobody was watching
  would be a different kind of failure entirely;
* that the durable write still happens, so the record does not depend on DDS.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from agents.trajectory import plan_events, plan_progress


@pytest.fixture(autouse=True)
def _quiet_publisher(monkeypatch):
    """Every test starts with no publisher and no remembered failure."""
    monkeypatch.setattr(plan_progress, "_node", None)
    monkeypatch.setattr(plan_progress, "_publishers", {})
    monkeypatch.setattr(plan_progress, "_disabled_reason", None)
    yield


class _Recorder:
    """Stands in for a Publisher, remembering what it was handed."""

    def __init__(self):
        self.messages: list = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


@pytest.fixture
def wire(monkeypatch):
    """A publisher that records, without a middleware underneath it."""
    recorder = _Recorder()
    monkeypatch.setattr(plan_progress, "_ensure_publisher",
                        lambda cell_id: recorder)
    if "std_msgs.msg" not in sys.modules:
        module = types.ModuleType("std_msgs")
        module.__path__ = []
        msg = types.ModuleType("std_msgs.msg")

        class String:
            def __init__(self):
                self.data = ""

        msg.String = String
        sys.modules.setdefault("std_msgs", module)
        sys.modules["std_msgs.msg"] = msg
    return recorder


# ── the name ─────────────────────────────────────────────────────────────────

def test_the_topic_is_one_per_cell_and_never_a_shared_one():
    """Two cells planning at once must not appear as one run to a dashboard."""
    assert plan_progress.topic_for("welding_cell_1x") == \
        "/welding_cell_1x/plan_progress"
    assert plan_progress.topic_for("a") != plan_progress.topic_for("b")


def test_the_dashboard_composes_the_same_name_from_the_same_suffix():
    """The two sides build the name independently — the suffix is what keeps them
    from drifting, so it is checked against the dashboard's own source."""
    from pathlib import Path
    source = Path("dashboard/web_control_panel.py").read_text()
    assert f'cell_topic("{plan_progress.TOPIC_SUFFIX}")' in source


def test_the_history_depth_is_bounded():
    """A late subscriber gets the interesting part of the run, not a session held
    in the middleware's memory — the unbounded record is the JSONL artefact."""
    assert 0 < plan_progress.HISTORY_DEPTH <= 1000


# ── the payload ──────────────────────────────────────────────────────────────

def test_a_published_event_says_what_it_is_and_which_run_it_belongs_to(wire):
    assert plan_progress.publish("welding_cell_1x", "t1",
                                 {"seq": 4, "kind": "operation_planned"}) is True
    sent = wire.messages[0]
    assert sent["schema"] == plan_progress.SCHEMA
    assert sent["cell_id"] == "welding_cell_1x"
    assert sent["trace_id"] == "t1"
    assert sent["seq"] == 4 and sent["kind"] == "operation_planned"


def test_the_event_fields_survive_verbatim(wire):
    plan_progress.publish("c1", "t1", {"seq": 1, "kind": "weld_failed",
                                       "operation_id": "op_3", "error": "IK"})
    sent = wire.messages[0]
    assert sent["operation_id"] == "op_3" and sent["error"] == "IK"


def test_a_payload_that_json_cannot_render_still_goes_out(wire):
    """Events carry whatever the planner had at hand; a datetime must not silence
    the live view."""
    from datetime import datetime
    assert plan_progress.publish("c1", "t1",
                                 {"seq": 1, "at": datetime(2026, 1, 1)}) is True
    assert wire.messages[0]["at"].startswith("2026-01-01")


def test_the_payload_is_one_json_object_per_message(wire):
    """A subscriber parses each message on its own; batching would make a dropped
    message lose several events instead of one."""
    for seq in range(3):
        plan_progress.publish("c1", "t1", {"seq": seq, "kind": "k"})
    assert [m["seq"] for m in wire.messages] == [0, 1, 2]


# ── best effort, always ──────────────────────────────────────────────────────

def test_with_no_middleware_publishing_says_no_and_raises_nothing(monkeypatch):
    """A test, a host script, a container without rclpy."""
    monkeypatch.setattr(plan_progress, "_ensure_publisher", lambda cell_id: None)
    assert plan_progress.publish("c1", "t1", {"seq": 1}) is False


def test_a_publisher_that_throws_does_not_throw_into_the_planner(monkeypatch):
    class _Broken:
        def publish(self, message):
            raise RuntimeError("rmw write failed")

    monkeypatch.setattr(plan_progress, "_ensure_publisher", lambda c: _Broken())
    assert plan_progress.publish("c1", "t1", {"seq": 1}) is False


def test_the_reason_is_remembered_so_the_planner_log_stays_readable(monkeypatch):
    """A planner that logs the same missing-rclpy line once per weld has a log
    nobody can read."""
    calls = {"n": 0}

    def _import_fails(name, *args, **kwargs):
        if name == "rclpy":
            calls["n"] += 1
            raise ImportError("no rclpy here")
        return _real_import(name, *args, **kwargs)

    import builtins
    _real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _import_fails)
    for _ in range(5):
        plan_progress.publish("c1", "t1", {"seq": 1})
    assert calls["n"] == 1
    assert plan_progress.describe()["available"] is False
    assert "no rclpy here" in plan_progress.describe()["unavailable_because"]


def test_describe_names_the_topic_and_the_cells_it_is_publishing_for(wire,
                                                                    monkeypatch):
    monkeypatch.setattr(plan_progress, "_publishers", {"c2": wire, "c1": wire})
    described = plan_progress.describe()
    assert described["schema"] == plan_progress.SCHEMA
    assert described["topic_suffix"] == plan_progress.TOPIC_SUFFIX
    assert described["cells"] == ["c1", "c2"]
    assert described["available"] is True


# ── the planner's event vocabulary uses it ───────────────────────────────────

def test_every_plan_event_is_also_published(wire, tmp_path, monkeypatch):
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("CELL_DATA_MODE", raising=False)
    plan_events._SEQ.clear()
    plan_events.emit("c1", "t1", "run_started", backend="ompl")
    plan_events.emit("c1", "t1", "finished", success=True)
    assert [m["kind"] for m in wire.messages] == ["run_started", "finished"]
    assert [m["seq"] for m in wire.messages] == [1, 2]
    assert wire.messages[0]["backend"] == "ompl"


def test_the_durable_stream_is_written_even_though_dds_carried_it(wire, tmp_path,
                                                                 monkeypatch):
    """The topic is the *view*. The record is the artefact, and a subscriber that
    joined late reads the rest of the run from it."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("CELL_DATA_MODE", raising=False)
    plan_events._SEQ.clear()
    plan_events.emit("c1", "t1", "run_started")
    from utils import cell_store
    text = cell_store.read_text("c1", cell_store.KIND_LOGS, "planevents_t1.jsonl")
    assert json.loads(text.splitlines()[0])["kind"] == "run_started"


def test_a_dead_topic_does_not_stop_the_durable_write(tmp_path, monkeypatch):
    """The order in emit() is publish-then-store, and the two failure domains are
    separate on purpose."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("CELL_DATA_MODE", raising=False)
    monkeypatch.setattr(plan_progress, "publish",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    plan_events._SEQ.clear()
    assert plan_events.emit("c1", "t1", "run_started")["kind"] == "run_started"
    from utils import cell_store
    text = cell_store.read_text("c1", cell_store.KIND_LOGS, "planevents_t1.jsonl")
    assert text and "run_started" in text


def test_a_failed_durable_write_does_not_stop_the_publish(wire, monkeypatch):
    """And the other way round, which is the one an operator notices: the run is
    still visible while the store is having a bad day."""
    monkeypatch.setattr(plan_events, "_store_event",
                        lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    plan_events._SEQ.clear()
    plan_events.emit("c1", "t1", "run_started")
    assert [m["kind"] for m in wire.messages] == ["run_started"]
