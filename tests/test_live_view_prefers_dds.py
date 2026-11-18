"""The live planning view reads DDS, and falls back to the durable stream.

Following a run used to be: browser → dashboard → orchestrator HTTP → the
planner's event file on a shared volume. Three processes and a filesystem in the
path of a question about *now*, while the panel was already a ROS node subscribed
to this cell's other topics. So the planner publishes on
``/<cell>/plan_progress`` and the panel buffers what arrives.

The fallback is the interesting half. The DDS feed is bounded, is empty in a panel
built without ``rclpy``, and says nothing at all to a reader who opens the page
after the run — so ``/api/a2a/plan_events`` stays exactly as it was and the browser
must use it whenever the DDS feed is not serving. The failure this guards against
is specific: an empty answer from a panel that is not listening rendering as "no
run in progress", which is the same wrong picture the whole visibility work was
about.

Python half: the buffer's semantics (one run at a time, duplicate replay, bounded
with an honest count, staleness) and the route's contract. JS half: the browser
prefers DDS, falls back on anything else, and shows which one it used — run in the
same DOM harness as :mod:`tests.test_dashboard_script_loads`, so a stub artefact
cannot pass here and fail there.
"""
from __future__ import annotations

import importlib
import json
import re
import subprocess
import textwrap
import types

import pytest

from tests.test_dashboard_asks_mcp_not_the_volume import _install_stubs
from tests.test_dashboard_script_loads import NODE, _dom_harness, _render, _script


@pytest.fixture(scope="module")
def panel():
    _install_stubs()
    return importlib.import_module("dashboard.web_control_panel")


@pytest.fixture
def feed(panel):
    """An empty buffer, and the route wired to a request of our own."""
    panel._plan_progress.update({"trace_id": None, "events": [], "seen": set(),
                                 "dropped": 0, "last_s": 0.0, "finished": False})
    yield panel
    panel._plan_progress.update({"trace_id": None, "events": [], "seen": set(),
                                 "dropped": 0, "last_s": 0.0, "finished": False})


def _event(seq: int, kind: str = "operation_planned", trace: str = "t1", **extra):
    """One message as the planner publishes it, envelope included."""
    return {"schema": "plan_progress/1", "cell_id": "welding_cell_1x",
            "trace_id": trace, "seq": seq, "at": "2026-09-14T10:00:00",
            "kind": kind, **extra}


def _get(panel, monkeypatch, **args):
    """Call the route with query args, as Flask would."""
    monkeypatch.setattr(panel, "flask_request", types.SimpleNamespace(
        args=types.SimpleNamespace(get=lambda k, d=None: args.get(k, d))))
    return panel.plan_progress_feed()


# ── the buffer ───────────────────────────────────────────────────────────────

def test_an_event_off_the_topic_is_buffered_without_its_envelope(feed):
    feed._record_plan_progress(_event(1, text="Planning 3 operations"))
    row = feed._plan_progress["events"][0]
    assert row["seq"] == 1 and row["kind"] == "operation_planned"
    assert row["text"] == "Planning 3 operations"
    assert "schema" not in row and "cell_id" not in row, (
        "the envelope is addressing, not content — the renderer never shows it")


def test_a_message_that_is_not_an_event_is_ignored_rather_than_buffered(feed):
    for junk in ({}, {"seq": "one", "trace_id": "t1"}, {"seq": 1},
                 {"trace_id": "t1"}, {"seq": None, "trace_id": "t1"}):
        feed._record_plan_progress(junk)
    assert feed._plan_progress["events"] == []


def test_a_replayed_event_is_not_shown_twice(feed):
    """TRANSIENT_LOCAL replays the run's history to a late subscriber, and this
    panel may be that subscriber more than once."""
    for _ in range(3):
        feed._record_plan_progress(_event(1))
    assert [e["seq"] for e in feed._plan_progress["events"]] == [1]


def test_events_that_arrive_out_of_order_are_shown_in_order(feed):
    for seq in (3, 1, 2):
        feed._record_plan_progress(_event(seq))
    assert [e["seq"] for e in feed._plan_progress["events"]] == [1, 2, 3]


def test_a_new_run_replaces_the_one_on_screen(feed):
    """Two interleaved runs in one feed is worse than losing the older one,
    because a reader cannot tell which line belongs to which."""
    feed._record_plan_progress(_event(1, trace="t1"))
    feed._record_plan_progress(_event(1, trace="t2"))
    assert feed._plan_progress["trace_id"] == "t2"
    assert len(feed._plan_progress["events"]) == 1


def test_the_buffer_is_bounded_and_says_how_much_it_dropped(feed):
    """A fifteen-minute run outruns any buffer. Silently dropping the start of it
    would make the feed look like the run began in the middle."""
    for seq in range(1, feed.MAX_PLAN_PROGRESS_EVENTS + 51):
        feed._record_plan_progress(_event(seq))
    assert len(feed._plan_progress["events"]) == feed.MAX_PLAN_PROGRESS_EVENTS
    assert feed._plan_progress["dropped"] == 50
    assert feed._plan_progress["events"][-1]["seq"] == \
        feed.MAX_PLAN_PROGRESS_EVENTS + 50


def test_a_dropped_seq_can_be_buffered_again_after_a_reconnect(feed):
    """The seen-set is pruned with the buffer; otherwise a replay after an
    overflow would be silently discarded forever."""
    for seq in range(1, feed.MAX_PLAN_PROGRESS_EVENTS + 2):
        feed._record_plan_progress(_event(seq))
    assert 1 not in feed._plan_progress["seen"]


# ── answering one poll ───────────────────────────────────────────────────────

def test_a_poll_carries_only_what_is_new(feed):
    for seq in (1, 2, 3):
        feed._record_plan_progress(_event(seq))
    answer = feed._plan_progress_since(2, 200)
    assert [e["seq"] for e in answer["events"]] == [3]
    assert answer["next_seq"] == 3, "the next poll must not re-read the run"


def test_a_poll_from_the_start_carries_the_run_so_far(feed):
    for seq in (1, 2):
        feed._record_plan_progress(_event(seq))
    assert [e["seq"] for e in feed._plan_progress_since(0, 200)["events"]] == [1, 2]


def test_a_poll_says_which_topic_it_is_reading(feed):
    feed._record_plan_progress(_event(1))
    answer = feed._plan_progress_since(0, 200)
    assert answer["source"] == "dds"
    assert answer["topic"] == feed.cell_topic("/plan_progress")


def test_a_run_is_running_until_it_says_otherwise(feed):
    feed._record_plan_progress(_event(1, "run_started"))
    assert feed._plan_progress_since(0, 200)["running"] is True


@pytest.mark.parametrize("kind", ["finished", "aborted", "paused"])
def test_the_planners_own_terminal_event_ends_the_run(feed, kind):
    feed._record_plan_progress(_event(1, "run_started"))
    feed._record_plan_progress(_event(2, kind))
    assert feed._plan_progress_since(0, 200)["running"] is False


def test_a_planner_that_died_without_saying_so_does_not_blink_forever(feed):
    """The stale-indicator bug: a container killed mid-run never publishes
    ``aborted``, and PLANNING would stay lit for the rest of the session."""
    feed._record_plan_progress(_event(1, "run_started"))
    feed._plan_progress["last_s"] -= feed.PLAN_PROGRESS_IDLE_S + 1
    assert feed._plan_progress_since(0, 200)["running"] is False


def test_a_cell_with_no_run_at_all_is_not_running(feed):
    answer = feed._plan_progress_since(0, 200)
    assert answer["running"] is False and answer["trace_id"] is None


# ── the route ────────────────────────────────────────────────────────────────

def test_the_route_is_unavailable_when_nothing_has_been_published(feed,
                                                                 monkeypatch):
    """The distinction the fallback turns on: "I am not listening" must not read
    as "there is no run"."""
    monkeypatch.setattr(feed, "ros_node", object())
    body, status = _get(feed, monkeypatch, since="0")
    assert status == 200
    assert body["available"] is False
    assert body["events"] == []


def test_the_route_is_available_once_a_run_is_on_the_feed(feed, monkeypatch):
    monkeypatch.setattr(feed, "ros_node", object())
    feed._record_plan_progress(_event(1, "run_started"))
    body, _ = _get(feed, monkeypatch, since="0")
    assert body["available"] is True
    assert body["trace_id"] == "t1"
    assert [e["seq"] for e in body["events"]] == [1]


def test_a_panel_with_no_ros_node_says_so_instead_of_answering_empty(feed,
                                                                    monkeypatch):
    monkeypatch.setattr(feed, "ros_node", None)
    body, status = _get(feed, monkeypatch, since="0")
    assert status == 200
    assert body["available"] is False and body["error"]


def test_a_nonsense_since_is_refused_and_still_not_available(feed, monkeypatch):
    monkeypatch.setattr(feed, "ros_node", object())
    body, status = _get(feed, monkeypatch, since="soon")
    assert status == 400
    assert body["success"] is False and body["available"] is False


def test_a_negative_since_is_read_as_the_beginning(feed, monkeypatch):
    monkeypatch.setattr(feed, "ros_node", object())
    feed._record_plan_progress(_event(1, "run_started"))
    body, _ = _get(feed, monkeypatch, since="-5")
    assert [e["seq"] for e in body["events"]] == [1]


def test_the_answer_has_the_same_shape_the_durable_feed_has(feed, monkeypatch):
    """One renderer serves both sources, so the two answers must be one shape."""
    monkeypatch.setattr(feed, "ros_node", object())
    feed._record_plan_progress(_event(1, "run_started"))
    body, _ = _get(feed, monkeypatch, since="0")
    assert {"success", "events", "next_seq", "trace_id", "running"} <= set(body)


# ── the subscription ─────────────────────────────────────────────────────────

def test_the_panel_subscribes_to_the_planners_topic_with_replay(panel):
    """TRANSIENT_LOCAL, or opening the page mid-run shows a blank feed until the
    next event — which on a long weld is a minute away."""
    from pathlib import Path
    text = Path("dashboard/web_control_panel.py").read_text()
    # Anchored on the create_subscription call and not on the topic name alone:
    # the route composes the same name to report it, and that match comes first.
    start = text.index('String, cell_topic("/plan_progress")')
    block = text[start:start + 400]
    assert "TRANSIENT_LOCAL" in block
    assert "MAX_PLAN_PROGRESS_EVENTS" in block


def test_the_subscription_is_the_message_handler_and_nothing_else(panel):
    """The callback carries no policy of its own — otherwise what the tests below
    exercise is not what a running executor calls. (``rclpy`` is a MagicMock here,
    so the node class itself cannot be instantiated or introspected.)"""
    from pathlib import Path
    text = Path("dashboard/web_control_panel.py").read_text()
    body = text[text.index("def _plan_progress_cb(self, msg):"):][:600]
    body = body[:body.index("\n    def ", 10)]
    assert "handle_plan_progress_message(msg.data" in body
    assert "json.loads" not in body, "the decoding belongs to the handler"


def test_a_malformed_message_does_not_take_the_executor_down(panel, feed):
    """One bad publisher would otherwise cost every other subscription in the
    panel — robot state, collision alerts, the lot."""
    for data in ("not json", "[]", "null", json.dumps({"seq": "x"}), b"\xff",
                 json.dumps({"seq": 1})):
        assert panel.handle_plan_progress_message(data) is not None
    assert panel.handle_plan_progress_message(json.dumps(_event(1))) is True
    assert [e["seq"] for e in feed._plan_progress["events"]] == [1]


def test_a_buffer_that_throws_is_reported_and_not_raised(panel, feed, monkeypatch):
    warned: list = []
    monkeypatch.setattr(panel, "_record_plan_progress",
                        lambda event: (_ for _ in ()).throw(RuntimeError("full")))
    assert panel.handle_plan_progress_message(json.dumps(_event(1)),
                                              warn=warned.append) is False
    assert warned and "full" in warned[0]


def test_an_event_marks_the_planner_as_alive_on_the_agents_tab(panel, feed,
                                                              monkeypatch):
    touched: list = []
    monkeypatch.setattr(panel, "_touch", lambda agent: touched.append(agent))
    panel.handle_plan_progress_message(json.dumps(_event(1)))
    assert touched == ["collaborative-planner"]


def test_the_topic_is_declared_on_the_planners_row_in_the_diagram(panel):
    planner = next(a for a in panel.AGENT_TOPOLOGY
                   if a["id"] == "collaborative-planner")
    assert "/plan_progress" in planner["topics_out"]


# ── the browser ──────────────────────────────────────────────────────────────

DDS_ANSWER = {"success": True, "source": "dds", "available": True,
              "cell_id": "some_cell", "topic": "/some_cell/plan_progress",
              "trace_id": "t_dds", "next_seq": 2, "dropped": 0, "running": True,
              "events": [{"seq": 1, "at": "2026-09-14T10:00:00",
                          "kind": "run_started", "text": "from dds"},
                         {"seq": 2, "at": "2026-09-14T10:00:01",
                          "kind": "operation_planned", "text": "weld 1"}]}
HTTP_ANSWER = {"success": True, "trace_id": "t_http", "next_seq": 1,
               "running": True,
               "events": [{"seq": 1, "at": "2026-09-14T10:00:00",
                           "kind": "run_started", "text": "from the file"}]}


# The shared harness's elements are a subset of a real one: ``textContent`` is a
# plain property and there is no ``childNodes``. ``liveAppend`` reads
# ``feed.childNodes.length`` — always present on a real Element — so without this
# the read throws inside the poll's promise chain, is swallowed by its ``catch``,
# and the DDS answer silently becomes an HTTP fallback: a stub artefact that would
# have read as a production defect. Backing it with the harness's own ``children``
# is the smallest completion that keeps the rendered rows readable.
FEED_READER = textwrap.dedent("""
    const _feed = document.getElementById('live-feed');
    Object.defineProperty(_feed, 'childNodes', { get() { return this.children; } });
    Object.defineProperty(_feed, 'firstChild',
                          { get() { return this.children[0] || null; } });
    function feedText() {
      return (_feed.children || [])
        .map(row => (row.children || []).map(s => s.textContent).join(' '))
        .join('\\n');
    }
""")


def _probe(tmp_path, probe):
    html = _render(["robot1", "robot2"])
    js = tmp_path / "panel.js"
    js.write_text(_dom_harness(html) + _script(html) + FEED_READER + probe)
    return subprocess.run([NODE, str(js)], capture_output=True, text=True,
                          timeout=60)


def _out(stdout: str) -> dict:
    line = re.search(r"^OUT=(\{.*\})$", stdout, re.M)
    assert line, f"the probe printed no OUT= line:\n{stdout}"
    return json.loads(line.group(1))


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheBrowserPrefersDds:

    def test_an_available_dds_feed_is_what_gets_rendered(self, tmp_path):
        probe = textwrap.dedent(f"""
            ROUTES["/api/plan_progress"] = {json.dumps(DDS_ANSWER)};
            ROUTES["/api/a2a/plan_events"] = {json.dumps(HTTP_ANSWER)};
            livePoll();
            setImmediate(() => setImmediate(() => {{
              console.log("OUT=" + JSON.stringify({{
                feed: feedText(),
                run: $('live-run').textContent,
                status: $('live-status').textContent}}));
            }}));
        """)
        out = _out(_probe(tmp_path, probe).stdout)
        assert "from dds" in out["feed"]
        assert "from the file" not in out["feed"], "the durable stream won a race"
        assert out["run"] == "t_dds"
        assert "DDS" in out["status"], "the operator cannot tell which source this is"

    def test_an_unavailable_dds_feed_falls_back_to_the_durable_stream(self,
                                                                     tmp_path):
        probe = textwrap.dedent(f"""
            ROUTES["/api/plan_progress"] = {json.dumps(
                {"success": True, "available": False, "events": [],
                 "trace_id": None, "next_seq": 0, "running": False})};
            ROUTES["/api/a2a/plan_events"] = {json.dumps(HTTP_ANSWER)};
            livePoll();
            setImmediate(() => setImmediate(() => {{
              console.log("OUT=" + JSON.stringify({{
                feed: feedText(),
                run: $('live-run').textContent}}));
            }}));
        """)
        out = _out(_probe(tmp_path, probe).stdout)
        assert "from the file" in out["feed"]
        assert out["run"] == "t_http"

    def test_a_panel_whose_dds_route_answers_nothing_still_shows_the_run(self,
                                                                        tmp_path):
        """An unknown route answers ``{}`` in this harness — the same shape a
        dashboard built before this route would give. It must not read as "no run
        in progress"; that is the whole reason ``available`` has to be positive."""
        probe = textwrap.dedent(f"""
            ROUTES["/api/a2a/plan_events"] = {json.dumps(HTTP_ANSWER)};
            livePoll();
            setImmediate(() => setImmediate(() => {{
              console.log("OUT=" + JSON.stringify({{
                feed: feedText(),
                run: $('live-run').textContent}}));
            }}));
        """)
        out = _out(_probe(tmp_path, probe).stdout)
        assert "from the file" in out["feed"]
        assert out["run"] == "t_http"

    def test_the_dds_feed_reports_what_it_had_to_drop(self, tmp_path):
        answer = dict(DDS_ANSWER, dropped=37)
        probe = textwrap.dedent(f"""
            ROUTES["/api/plan_progress"] = {json.dumps(answer)};
            livePoll();
            setImmediate(() => setImmediate(() => {{
              console.log("OUT=" + JSON.stringify(
                {{status: $('live-status').textContent}}));
            }}));
        """)
        assert "37 older events" in _out(_probe(tmp_path, probe).stdout)["status"]

    def test_a_live_run_lights_the_panels_planning_indicator(self, tmp_path):
        """The DDS feed is the only source that sees a run this panel did not
        start, so it is what the run indicator has to follow."""
        probe = textwrap.dedent(f"""
            ROUTES["/api/plan_progress"] = {json.dumps(DDS_ANSWER)};
            livePoll();
            setImmediate(() => setImmediate(() => {{
              console.log("OUT=" + JSON.stringify({{
                planning: _planRunning,
                bar: $('run-blink') ? $('run-blink').textContent : ''}}));
            }}));
        """)
        out = _out(_probe(tmp_path, probe).stdout)
        assert out["planning"] is True

    def test_polling_asks_only_for_what_it_has_not_seen(self, tmp_path):
        """A fifteen-minute run polled once a second must not be re-sent every
        time — the whole feed is built around ?since=."""
        probe = textwrap.dedent(f"""
            ROUTES["/api/plan_progress"] = {json.dumps(DDS_ANSWER)};
            const urls = [];
            const realFetch = globalThis.fetch;
            globalThis.fetch = (u, o) => {{ urls.push(String(u)); return realFetch(u, o); }};
            livePoll();
            setImmediate(() => setImmediate(() => {{
              livePoll();
              setImmediate(() => setImmediate(() => {{
                console.log("OUT=" + JSON.stringify({{urls: urls}}));
              }}));
            }}));
        """)
        urls = _out(_probe(tmp_path, probe).stdout)["urls"]
        assert urls[0] == "/api/plan_progress?since=0"
        assert urls[-1] == "/api/plan_progress?since=2", urls
