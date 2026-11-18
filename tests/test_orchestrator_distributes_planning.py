"""The orchestrator's A2A side of planning distribution.

* POST /a2a/planners/register stores a planner; GET /a2a/planners lists them
  with ``free``; a registration without an http(s) MCP URL is refused (400).
* A resume goes to the planner that OWNS the run (it holds the checkpoint); a
  run no planner knows is refused (409).
* Stop (/a2a/abort) cancels only THIS cell's planning, on whichever planner
  runs it — another planner busy with another cell is left alone.

Skipped on a host without flask (it runs in the tools container).
"""
from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")

from agents.orchestrator import planner_registry, run_registry, server as srv  # noqa: E402

CELL = srv.CELL_ID


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(run_registry, "_REGISTRY", run_registry.RunRegistry())
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    planner_registry.reset()
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as c:
        yield c
    planner_registry.reset()


def _reg(client, pid, busy=False, active=None):
    return client.post("/a2a/planners/register", json={
        "planner_id": pid, "mcp_url": f"http://{pid}:8890/mcp", "busy": busy,
        "active": active})


def test_planners_register_and_are_listed(client):
    assert _reg(client, "trajectory-agent").status_code == 200
    assert _reg(client, "trajectory-agent-2", busy=True).status_code == 200
    body = client.get("/a2a/planners").get_json()
    assert [p["planner_id"] for p in body["planners"]] == ["trajectory-agent",
                                                          "trajectory-agent-2"]
    assert body["free"] == 1
    bad = client.post("/a2a/planners/register", json={"planner_id": "x", "mcp_url": "x"})
    assert bad.status_code == 400


def test_a_resume_goes_to_the_planner_that_owns_the_run(client, monkeypatch):
    _reg(client, "trajectory-agent")
    _reg(client, "trajectory-agent-2")
    calls = []

    def fake(url, tool, args, timeout=600.0):
        calls.append((url, tool))
        if tool == "get_plan_status":
            return {"status": "interrupted"} if "agent-2" in url else {"status": "unknown"}
        if tool == "resume_planning":
            return {"accepted": True, "tree_id": args["tree_id"]}
        return {}
    monkeypatch.setattr(srv, "_mcp_call", fake)
    ok = client.post("/a2a/plan/resume", json={"cell_id": CELL, "tree_id": "bt_7"})
    assert ok.status_code == 200
    assert ("http://trajectory-agent-2:8890/mcp", "resume_planning") in calls
    assert ("http://trajectory-agent:8890/mcp", "resume_planning") not in calls
    monkeypatch.setattr(srv, "_mcp_call", lambda *a, **k: {"status": "unknown"})
    none = client.post("/a2a/plan/resume", json={"cell_id": CELL, "tree_id": "bt_nobody"})
    assert none.status_code == 409


def test_stop_cancels_only_this_cells_planning(client, monkeypatch):
    _reg(client, "trajectory-agent")
    _reg(client, "trajectory-agent-2")
    cancelled = []

    def fake(url, tool, args, timeout=600.0):
        if tool == "planner_status":
            cell = CELL if "agent-2" in url else "another_cell"
            return {"busy": True, "active": {"tree_id": f"bt_{cell}", "cell_id": cell}}
        if tool == "cancel_planning":
            cancelled.append((url, args.get("tree_id")))
            return {"success": True, "cancelled": [args.get("tree_id")]}
        return {}
    monkeypatch.setattr(srv, "_mcp_call", fake)
    body = client.post("/a2a/abort", json={"cell_id": CELL}).get_json()
    assert cancelled == [("http://trajectory-agent-2:8890/mcp", f"bt_{CELL}")]
    assert body["planning"]["cancelled"] == [f"bt_{CELL}"]
