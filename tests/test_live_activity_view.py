"""The Agents tab's live activity view, and the pause/resume surface behind it.

Same two halves as the Plan Log tab's tests, for the same reason — they fail
separately.

**The endpoints** (orchestrator, in-process): a poll that returns only what is
new, a checkpoint list, and pause/resume. The two properties that matter here are
that an incremental poll is genuinely incremental (a fifteen-minute run must not be
re-sent every second) and that resuming a run with nothing to resume from is
REFUSED rather than quietly restarted — a "resume" that replans from the first weld
is the exact surprise this whole mechanism exists to remove.

**The pane** (source text): the view sits inside the Agents tab, under the agent
schema, where the request asked for it; every id the JS looks up is served; the
feed's data goes in via ``textContent``; and polling stops when the tab is left,
because a feed that keeps polling from a hidden pane is just load.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PANEL = (ROOT / "dashboard" / "web_control_panel.py").read_text()
SPEC = (ROOT / "docs" / "api" / "openapi.yaml").read_text()

try:
    import flask  # noqa: F401

    from agents.orchestrator import server as srv

    CELL = srv.CELL_ID
except Exception:  # pragma: no cover — depends on the host's packages
    srv = None
    CELL = "welding_cell_2_small"

from agents.trajectory import plan_checkpoint, plan_events  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    if srv is None:
        pytest.skip("flask is not installed on this host")
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    (tmp_path / CELL / "agentic_planner" / "logs").mkdir(parents=True)
    plan_events.forget("t1")
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as client:
        yield client
    plan_events.forget("t1")


# ─── the live feed endpoint ───────────────────────────────────────────────────

def test_the_feed_is_empty_and_successful_before_any_run(client) -> None:
    """A cell that has never planned is a normal state, not a server fault."""
    body = client.get("/a2a/plan_events").get_json()
    assert body["success"] is True
    assert body["events"] == []
    assert body["trace_id"] is None
    assert body["running"] is False


def test_the_feed_serves_a_run_in_order(client) -> None:
    plan_events.run_started(CELL, "t1", operations=2, backend="ompl",
                            model="opus-5", step_budget=100)
    plan_events.operation_started(CELL, "t1", "W01", robot="robot1")
    body = client.get("/a2a/plan_events").get_json()
    assert body["trace_id"] == "t1"
    assert [e["kind"] for e in body["events"]] == ["run_started", "operation_started"]
    assert body["running"] is True


def test_polling_with_since_returns_only_what_is_new(client) -> None:
    """The property that makes a one-second refresh affordable."""
    plan_events.run_started(CELL, "t1", operations=1, backend="ompl",
                            model="opus-5", step_budget=100)
    first = client.get("/a2a/plan_events").get_json()
    plan_events.operation_started(CELL, "t1", "W02")
    second = client.get(f"/a2a/plan_events?since={first['next_seq']}").get_json()
    assert [e["operation_id"] for e in second["events"]] == ["W02"]
    # And a poll with nothing new says so without moving the cursor.
    third = client.get(f"/a2a/plan_events?since={second['next_seq']}").get_json()
    assert third["events"] == []
    assert third["next_seq"] == second["next_seq"]


def test_a_strategy_change_reaches_the_feed_with_its_evidence(client) -> None:
    """"Changed strategy" is the diff, not an adjective."""
    plan_events.note_attempt(CELL, "t1", "W14", {"approach_offset_m": 0.10})
    plan_events.note_attempt(CELL, "t1", "W14", {"approach_offset_m": 0.07})
    events = client.get("/a2a/plan_events").get_json()["events"]
    strategy = [e for e in events if e["kind"] == "strategy"]
    assert strategy and strategy[0]["changed"] == {"approach_offset_m": [0.10, 0.07]}


def test_the_feed_rejects_a_non_integer_cursor(client) -> None:
    assert client.get("/a2a/plan_events?since=soon").status_code == 400


def test_the_feed_refuses_a_traversing_trace_id(client) -> None:
    resp = client.get("/a2a/plan_events?trace_id=../../etc/passwd")
    assert resp.status_code == 400
    assert "plain identifier" in resp.get_json()["error"]


# ─── checkpoints ─────────────────────────────────────────────────────────────

def test_checkpoints_are_empty_before_any_run(client) -> None:
    body = client.get("/a2a/checkpoints").get_json()
    assert body["success"] is True
    assert body["checkpoints"] == []
    assert body["resumable"] == 0


def test_a_checkpoint_is_listed_with_what_a_resume_would_recover(client) -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from agents.trajectory.tools import trajectory_cache
    trajectory_cache.cache_trajectory(CELL, "robot1", "W01",
                                      [{"positions": [0.0] * 6, "time_s": 0.0}])
    try:
        # One answered tool call, because that is the threshold for resumable: a
        # history of just the opening prompt would preserve nothing.
        conversation = [
            HumanMessage(content="plan"),
            AIMessage(content="", tool_calls=[
                {"id": "c1", "name": "plan_weld", "args": {"operation_id": "W01"}}]),
            ToolMessage(content="ok", tool_call_id="c1"),
        ]
        plan_checkpoint.save(CELL, "t1", messages=conversation,
                             operations=[{"id": "W01"}], constraints=None, step=5,
                             reason="token expiring")
        body = client.get("/a2a/checkpoints").get_json()
        assert body["resumable"] == 1
        assert body["checkpoints"][0]["tree_id"] == "t1"
        assert body["checkpoints"][0]["trajectories"] == 1
        assert body["checkpoints"][0]["step"] == 5
    finally:
        trajectory_cache.clear_trajectory_cache(CELL)


def test_the_checkpoint_limit_must_be_an_integer(client) -> None:
    assert client.get("/a2a/checkpoints?limit=lots").status_code == 400


# ─── pause and resume ────────────────────────────────────────────────────────

def test_resume_without_a_tree_id_is_refused_with_a_pointer(client) -> None:
    resp = client.post("/a2a/plan/resume", json={})
    assert resp.status_code == 400
    assert "/a2a/checkpoints" in resp.get_json()["error"]


def test_resuming_a_run_with_no_checkpoint_is_refused_not_restarted(client,
                                                                   monkeypatch) -> None:
    """The whole point of the refusal: 'resume' must never mean 'replan from W01'."""
    monkeypatch.setattr(srv, "_mcp_call",
                        lambda url, name, args, timeout=600.0: {
                            "accepted": False,
                            "error": "no checkpoint found for bt_nope"})
    resp = client.post("/a2a/plan/resume", json={"tree_id": "bt_nope"})
    assert resp.status_code == 409
    assert resp.get_json()["success"] is False


def test_a_resume_is_accepted_and_reports_what_it_recovered(client, monkeypatch) -> None:
    seen: dict = {}

    def fake(url, name, args, timeout=600.0):
        seen.update({"name": name, "args": args})
        return {"accepted": True, "tree_id": args["tree_id"], "status": "planning",
                "resumed_from_step": 18, "trajectories_restored": 13}

    monkeypatch.setattr(srv, "_mcp_call", fake)
    body = client.post("/a2a/plan/resume", json={"tree_id": "bt_1"}).get_json()
    assert seen["name"] == "resume_planning"
    assert seen["args"]["tree_id"] == "bt_1"
    assert body["resumed_from_step"] == 18
    assert body["trajectories_restored"] == 13


def test_pause_reaches_the_planner_and_is_not_an_estop(client, monkeypatch) -> None:
    """Pause is planning-only. It must not touch the cell at all."""
    calls: list = []

    def fake(url, name, args, timeout=600.0):
        calls.append((name, args))
        return {"success": True, "paused": ["bt_1"]}

    monkeypatch.setattr(srv, "_mcp_call", fake)
    body = client.post("/a2a/plan/pause", json={"tree_id": "bt_1"}).get_json()
    assert [name for name, _ in calls] == ["pause_planning"]
    assert body["paused"] == ["bt_1"]
    assert "trigger_estop" not in [name for name, _ in calls]


def test_pausing_an_idle_stack_is_not_an_error(client, monkeypatch) -> None:
    monkeypatch.setattr(srv, "_mcp_call",
                        lambda url, name, args, timeout=600.0: {
                            "success": True, "paused": [],
                            "note": "no planning run is active"})
    resp = client.post("/a2a/plan/pause", json={})
    assert resp.status_code == 200
    assert resp.get_json()["paused"] == []


# ─── the pane ────────────────────────────────────────────────────────────────

def test_the_live_view_sits_inside_the_agents_tab_under_the_schema() -> None:
    """Where the request put it: in the Agents tab, below the agent schema."""
    pane = PANEL[PANEL.index('id="pane-agents"'):]
    pane = pane[:pane.index('id="tip"')]
    assert 'id="agents-wrap"' in pane
    assert 'id="live-wrap"' in pane
    assert pane.index('id="agents-wrap"') < pane.index('id="live-wrap"')


@pytest.mark.parametrize("element_id", [
    "live-wrap", "live-run", "live-follow", "live-checkpoints", "live-status",
    "live-feed",
])
def test_every_id_the_view_uses_is_served(element_id) -> None:
    assert f'id="{element_id}"' in PANEL


def test_the_view_polls_incrementally() -> None:
    """?since= is not optional: without it the feed re-reads the run every tick."""
    assert "/api/a2a/plan_events?since='+_liveSeq" in PANEL


def test_polling_stops_when_the_tab_is_left() -> None:
    """A hidden pane that keeps polling is load with nobody reading it."""
    assert "function liveStop" in PANEL
    switch = PANEL[PANEL.index("function switchTab"):]
    switch = switch[:switch.index("\n}")]
    assert "liveStart()" in switch and "liveStop()" in switch


def test_the_view_offers_pause_and_resume() -> None:
    for fn in ("function livePause", "function liveResume",
               "function liveLoadCheckpoints", "function liveToggleFollow"):
        assert fn in PANEL, fn
    assert "/api/a2a/plan/pause" in PANEL
    assert "/api/a2a/plan/resume" in PANEL


def test_every_event_kind_the_planner_emits_has_a_colour() -> None:
    """A kind with no colour renders as anonymous grey in the feed."""
    colours = re.search(r"const LIVE_KIND=\{(.+?)\};", PANEL, re.S)
    assert colours
    source = (ROOT / "agents" / "trajectory" / "plan_events.py").read_text()
    emitted = set(re.findall(r'emit\(cell_id, trace_id, "([a-z_]+)"', source))
    emitted |= set(re.findall(r'kind, word = "([a-z_]+)"', source))
    missing = [kind for kind in sorted(emitted) if kind not in colours.group(1)]
    assert missing == [], missing


def test_the_feed_puts_event_text_in_via_textcontent() -> None:
    """Event text is model output and tool payloads. Never innerHTML."""
    offenders = [line.strip() for line in PANEL.splitlines()
                 if "innerHTML" in line and re.search(r"\blive[A-Z]|_live[A-Z]", line)]
    assert offenders == [], offenders


def test_the_dashboard_proxies_exist_for_every_call_the_view_makes() -> None:
    for route in ("/api/a2a/plan_events", "/api/a2a/checkpoints",
                  "/api/a2a/plan/pause", "/api/a2a/plan/resume"):
        assert route in PANEL, route
    assert '@app.route("/api/a2a/plan_events")' in PANEL
    assert '@app.route("/api/a2a/checkpoints")' in PANEL
    assert '@app.route("/api/a2a/plan/pause", methods=["POST"])' in PANEL
    assert '@app.route("/api/a2a/plan/resume", methods=["POST"])' in PANEL


def test_every_new_route_is_documented_on_both_servers() -> None:
    for path in ("/a2a/plan_events:", "/a2a/checkpoints:",
                 "/a2a/plan/pause:", "/a2a/plan/resume:",
                 "/api/a2a/plan_events:", "/api/a2a/checkpoints:",
                 "/api/a2a/plan/pause:", "/api/a2a/plan/resume:"):
        assert f"\n  {path}" in SPEC, path


# ─── the planner's own wiring ────────────────────────────────────────────────

def test_the_planner_checkpoints_before_reporting_an_infrastructure_abort() -> None:
    """The order matters: the reply must be able to say 'resume it'.

    An expired token says nothing about the welds — the run only lost its place. If
    the failure were reported before the place was written, the reply could not
    honestly offer a resume.
    """
    source = (ROOT / "agents" / "trajectory" / "agent.py").read_text()
    # Anchored on the planning loop, so this reads the run's own abort handler and
    # not some tool's.
    handler = source[source.index("for state in agent.stream("):]
    handler = handler[handler.index("except Exception as exc:"):]
    handler = handler[:handler.index("    finally:")]
    assert handler.index('_checkpoint(f"aborted: {kind}")') < handler.index("return _finish({")
    assert '"resume_with"' in handler


def test_a_successful_run_discards_its_checkpoint() -> None:
    """A resumable checkpoint must only exist while a run has somewhere to go."""
    source = (ROOT / "agents" / "trajectory" / "agent.py").read_text()
    assert "plan_checkpoint.discard(cell_id, tree_id)" in source


def test_a_pause_is_only_taken_at_a_state_boundary() -> None:
    """Mid-step is not a conversation any provider will continue."""
    source = (ROOT / "agents" / "trajectory" / "agent.py").read_text()
    loop = source[source.index("for state in agent.stream("):]
    loop = loop[:loop.index("    except Exception as exc:")]
    assert "plan_checkpoint.pause_requested(tree_id)" in loop
    assert "break" in loop


def test_a_pause_is_not_recorded_as_a_failed_plan() -> None:
    """A coffee break must not leave a wreck in the plan list."""
    source = (ROOT / "agents" / "trajectory" / "agent.py").read_text()
    assert 'result.get("error_kind") != "paused"' in source


def test_the_agent_exposes_pause_resume_and_the_feed_as_tools() -> None:
    source = (ROOT / "agents" / "trajectory" / "mcp_server.py").read_text()
    for tool in ("def pause_planning", "def resume_planning",
                 "def list_checkpoints", "def get_plan_events"):
        assert tool in source, tool


def test_a_paused_job_is_not_reported_as_a_failure() -> None:
    """"failed" would not suggest the one correct next action, which is resume."""
    source = (ROOT / "agents" / "trajectory" / "mcp_server.py").read_text()
    job = source[source.index("def _run_plan_job"):source.index("# ─── Transport")]
    assert 'status="paused"' in job
    assert 'error_kind") == "paused"' in job
