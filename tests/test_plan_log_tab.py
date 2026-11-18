"""The Plan Log tab: its HTTP contract, and the rules its markup must keep.

Two halves, tested separately because they fail separately.

**The endpoints** (orchestrator, booted in-process): a history list, one trace by
id, ``latest`` as a shorthand, and a 404 — not a 500 and not an empty success —
when nothing has been recorded. Traces exist only for runs from the moment the
recorder shipped, so "no traces yet" is the normal first state of this tab and
must read as such.

**The pane** (source text): the tab is registered, every id the JS looks up is
served, and — the one that matters for safety — no plan render function pushes
trace data through ``innerHTML``. A trace is model output and tool payloads,
the least trustworthy data on the page; the panel already had an unescaped
``innerHTML`` log path, and this pane must not add a second one.

tests/test_dashboard_script_loads.py separately proves the script still boots.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PANEL = (ROOT / "dashboard" / "web_control_panel.py").read_text()

# The pane half is source text and must run on any host. Only the endpoint half
# needs the server, so flask is skipped per-fixture rather than per-module — a
# module-level importorskip silently took the markup and safety checks with it.
try:
    import flask  # noqa: F401

    from agents.orchestrator import server as srv

    CELL = srv.CELL_ID
except Exception:  # pragma: no cover — depends on the host's packages
    srv = None
    CELL = "welding_cell_2_small"

from agents.trajectory import plan_trace  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    if srv is None:
        pytest.skip("flask is not installed on this host")
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    (tmp_path / CELL / "agentic_planner" / "logs").mkdir(parents=True)
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as client:
        yield client


def _record(trace_id, **run):
    """Write one trace the endpoints can serve."""
    return plan_trace.capture(
        CELL, [], [{"id": "W01", "type": "spot_weld"}],
        {"success": run.pop("success", True), **run}, trace_id=trace_id)


# ─── the history ─────────────────────────────────────────────────────────────

def test_history_is_empty_and_successful_before_any_run(client) -> None:
    body = client.get("/a2a/plan_traces").get_json()
    assert body["success"] is True
    assert body["traces"] == []


def test_history_lists_recorded_runs_newest_first(client) -> None:
    _record(f"{CELL}_20260101_000000")
    _record(f"{CELL}_20260202_000000", success=False, error_kind="truncated")
    traces = client.get("/a2a/plan_traces").get_json()["traces"]
    assert [t["trace_id"] for t in traces] == [
        f"{CELL}_20260202_000000", f"{CELL}_20260101_000000"]
    assert traces[0]["error_kind"] == "truncated"


def test_history_limit_must_be_an_integer(client) -> None:
    assert client.get("/a2a/plan_traces?limit=lots").status_code == 400


def test_history_honours_the_limit(client) -> None:
    for i in range(3):
        _record(f"{CELL}_2026010{i}_000000")
    assert len(client.get("/a2a/plan_traces?limit=2").get_json()["traces"]) == 2


# ─── one trace ───────────────────────────────────────────────────────────────

def test_a_trace_is_served_with_both_views(client) -> None:
    _record("t1")
    body = client.get("/a2a/plan_trace/t1").get_json()
    assert body["success"] is True
    assert body["schema"] == plan_trace.SCHEMA
    assert body["run"]["trace_id"] == "t1"
    assert isinstance(body["points"], list) and isinstance(body["decisions"], list)


def test_latest_resolves_to_the_newest_trace(client) -> None:
    _record(f"{CELL}_20260101_000000")
    _record(f"{CELL}_20260303_000000")
    body = client.get("/a2a/plan_trace/latest").get_json()
    assert body["run"]["trace_id"] == f"{CELL}_20260303_000000"


def test_latest_is_404_with_a_readable_reason_when_nothing_is_recorded(client) -> None:
    """The tab's normal first state. It must not read as a server fault."""
    resp = client.get("/a2a/plan_trace/latest")
    assert resp.status_code == 404
    assert "no planning traces" in resp.get_json()["error"]


def test_a_missing_trace_is_404(client) -> None:
    assert client.get("/a2a/plan_trace/never_written").status_code == 404


def test_a_traversing_trace_id_is_refused(client) -> None:
    """Whatever the router leaves in the path must not reach the filesystem."""
    for bad in ("..", "%2e%2e%2fetc%2fpasswd", "a%00b"):
        resp = client.get(f"/a2a/plan_trace/{bad}")
        assert resp.status_code in (400, 404), bad
        if resp.status_code == 400:
            assert "plain identifier" in resp.get_json()["error"]


# ─── the pane ────────────────────────────────────────────────────────────────

def test_the_tab_is_registered_in_the_tab_machinery() -> None:
    tabs = re.search(r"const TABS=\[([^\]]+)\]", PANEL)
    assert tabs and "'plan'" in tabs.group(1)
    assert 'id="tab-plan"' in PANEL and 'id="pane-plan"' in PANEL
    assert "planLoadHistory()" in PANEL


@pytest.mark.parametrize("element_id", [
    "plan-run-select", "plan-summary", "plan-points", "plan-point-count",
    "plan-steps", "plan-step-count", "plan-detail",
])
def test_every_id_the_pane_uses_is_served(element_id) -> None:
    assert f'id="{element_id}"' in PANEL


def test_navigation_covers_points_decisions_and_stepping() -> None:
    """The three ways the request asked to navigate: per point, per decision, history."""
    for fn in ("function planSelectPoint", "function planSelectDecision",
               "function planStep", "function planLoadHistory", "function planLoad"):
        assert fn in PANEL, fn


def test_the_pane_reaches_the_orchestrator_through_the_dashboard_proxy() -> None:
    assert "/api/a2a/plan_traces" in PANEL
    assert "/api/a2a/plan_trace/" in PANEL
    assert '@app.route("/api/a2a/plan_traces")' in PANEL
    assert '@app.route("/api/a2a/plan_trace/<trace_id>")' in PANEL


def test_no_plan_render_function_injects_trace_data_into_innerhtml() -> None:
    """Trace text is model output. It goes in via textContent, or not at all."""
    offenders = [
        line.strip() for line in PANEL.splitlines()
        if "innerHTML" in line and re.search(r"\bplan[A-Z_]|_plan(Trace|Point|Seq)\b", line)
    ]
    assert offenders == [], offenders


def test_the_status_colours_cover_every_status_the_recorder_can_emit() -> None:
    """A status with no colour renders grey and silently reads as 'pending'."""
    colours = re.search(r"const PLAN_STATUS=\{([^}]+)\}", PANEL)
    assert colours
    for status in ("pending", "planned", "validated", "in_tree", "failed"):
        assert status in colours.group(1), status


def test_both_dashboard_routes_are_documented() -> None:
    spec = (ROOT / "docs" / "api" / "openapi.yaml").read_text()
    assert "/api/a2a/plan_traces:" in spec
    assert "/api/a2a/plan_trace/{trace_id}:" in spec
    assert "/a2a/plan_traces:" in spec
    assert "/a2a/plan_trace/{trace_id}:" in spec


# ─── the planner records on every exit path ──────────────────────────────────

def test_every_planner_exit_path_records_a_trace() -> None:
    """The runs worth reading are the failures, so no exit may skip _finish.

    Scanned from the ``_finish`` definition on. Everything above it is pre-flight
    argument validation that returns before the agent is built — there is no run
    to trace yet, and ``_finish`` is not even in scope. Everything below it is an
    outcome of an actual planning run, and every one of those must be recorded:
    the abort, the assembler rejection, the truncation, the step-budget
    exhaustion and the generic no-tree case, not just the success.
    """
    source = (ROOT / "agents" / "trajectory" / "agent.py").read_text()
    body = source[source.index("def _finish(result: dict) -> dict:"):]
    bare = [line.strip() for line in body.splitlines()
            if re.match(r"return\s*\{", line.strip())]
    assert bare == [], bare
    assert body.count("return _finish({") >= 6
