"""Several planners, one orchestrator: registration, discovery, distribution.

* A planner registers its MCP URL and heartbeats whether it is busy; one that
  stops beating is no longer given work. With no registration at all, the
  configured TRAJECTORY_AGENT_HOST is the (static) planner, as before.
* A new run goes to a FREE planner — asked, not assumed from a heartbeat — least
  recently used first; a planner that turns out busy (``planner_busy``) is
  skipped for the next; with none free the answer is ``no_free_planner``.
* The planner that accepted a run owns it; after an orchestrator restart the
  owner is found by asking the planners.
* The planner side posts its registration to /a2a/planners/register (a real
  HTTP round trip here).

Host-runnable: the orchestrator's MCP client is a fake.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agents.orchestrator import planner_registry as reg


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    reg.reset()
    monkeypatch.setenv("TRAJECTORY_AGENT_HOST", "trajectory-agent")
    monkeypatch.delenv("TRAJECTORY_PLANNERS", raising=False)
    yield
    reg.reset()


class Fleet:
    """Fake planners: busy flags and what each accepts."""

    def __init__(self, busy: dict):
        self.busy, self.calls, self.accept_busy = dict(busy), [], set()

    def __call__(self, url, tool, args, timeout):
        pid = url.split("//")[1].split(":")[0]
        self.calls.append((pid, tool))
        if tool == "planner_status":
            return {"planner_id": pid, "busy": self.busy.get(pid, False)}
        if tool == "plan_trajectories":
            if self.busy.get(pid) or pid in self.accept_busy:
                return {"accepted": False, "error_kind": "planner_busy"}
            self.busy[pid] = True
            return {"accepted": True, "tree_id": f"bt_{pid}"}
        if tool == "get_plan_status":
            return {"status": "interrupted"} if args["tree_id"] == f"bt_{pid}" else \
                {"status": "unknown"}
        return {}


def _register(pid, busy=False, hb=10.0):
    return reg.register({"planner_id": pid, "mcp_url": f"http://{pid}:8890/mcp",
                         "busy": busy, "heartbeat_s": hb})


def test_with_no_registration_the_configured_agent_is_the_planner():
    (p,) = reg.planners()
    assert p["mcp_url"] == "http://trajectory-agent:8890/mcp" and p["static"]
    assert reg.any_url() == "http://trajectory-agent:8890/mcp"


def test_registered_planners_replace_the_static_one_and_stale_ones_get_no_work(monkeypatch):
    _register("trajectory-agent")
    _register("trajectory-agent-2", hb=0.01)
    time.sleep(0.05)                                  # planner 2 missed its beats
    live = {p["planner_id"]: p["live"] for p in reg.planners()}
    assert live == {"trajectory-agent": True, "trajectory-agent-2": False}
    fleet = Fleet({})
    assert reg.pick_free(fleet)["planner_id"] == "trajectory-agent"
    with pytest.raises(ValueError):
        reg.register({"planner_id": "x", "mcp_url": "file:///etc"})


def test_only_allowed_urls_may_register_as_planners(monkeypatch):
    # A registered planner receives planning requests (cell data): not any URL.
    with pytest.raises(ValueError, match="not an allowed planner URL"):
        reg.register({"planner_id": "evil", "mcp_url": "http://attacker.example:80/mcp"})
    monkeypatch.setenv("PLANNER_ALLOWED_URLS", r"^http://planner-host-[0-9]+:8890/mcp$")
    assert reg.register({"planner_id": "far", "mcp_url": "http://planner-host-1:8890/mcp"})


def test_a_run_goes_to_a_free_planner_and_the_next_to_another():
    for pid in ("trajectory-agent", "trajectory-agent-2", "trajectory-agent-3"):
        _register(pid)
    fleet = Fleet({"trajectory-agent": True})        # its heartbeat is stale: it IS busy
    first = reg.submit(fleet, "plan_trajectories", {})
    second = reg.submit(fleet, "plan_trajectories", {})
    assert {first["planner_id"], second["planner_id"]} == {"trajectory-agent-2",
                                                          "trajectory-agent-3"}
    none_left = reg.submit(fleet, "plan_trajectories", {})
    assert none_left["accepted"] is False and none_left["error_kind"] == "no_free_planner"
    assert len(none_left["planners"]) == 3
    assert reg.owner_of(first["tree_id"])["planner_id"] == first["planner_id"]


def test_a_planner_claimed_between_the_check_and_the_call_is_skipped():
    _register("trajectory-agent")
    _register("trajectory-agent-2")
    fleet = Fleet({})
    fleet.accept_busy.add("trajectory-agent")        # free when asked, busy when submitted
    out = reg.submit(fleet, "plan_trajectories", {})
    assert out["accepted"] and out["planner_id"] == "trajectory-agent-2"


def test_the_owner_of_a_run_is_found_after_an_orchestrator_restart():
    _register("trajectory-agent")
    _register("trajectory-agent-2")
    fleet = Fleet({})
    assert reg.owner_of("bt_trajectory-agent-2", fleet)["planner_id"] == "trajectory-agent-2"
    assert reg.owner_of("bt_nobody", fleet) is None


def test_a_heartbeat_reports_the_active_run_and_who_owns_it():
    reg.register({"planner_id": "trajectory-agent-2", "busy": True,
                  "mcp_url": "http://trajectory-agent-2:8890/mcp",
                  "active": {"tree_id": "bt_77", "cell_id": "c"}})
    assert reg.owner_of("bt_77")["planner_id"] == "trajectory-agent-2"


def test_a_planner_registers_over_http(monkeypatch):
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"success": true}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        from agents.trajectory import planner_registration as pr
        monkeypatch.setenv("ORCHESTRATOR_URL", f"http://127.0.0.1:{srv.server_port}")
        monkeypatch.setenv("PLANNER_ADVERTISE_URL", "http://trajectory-agent-2:8890/mcp")
        out = pr.beat(lambda: {"planner_id": "trajectory-agent-2", "busy": False,
                               "active": None, "backends": ["ompl"]})
        assert out == {"success": True}
        path, body = got[0]
        assert path == "/a2a/planners/register"
        assert body["planner_id"] == "trajectory-agent-2" and body["busy"] is False
        assert body["mcp_url"] == "http://trajectory-agent-2:8890/mcp"
        reg.register(body)                             # what the route does with it
        (p,) = reg.planners()
        assert p["planner_id"] == "trajectory-agent-2" and p["live"] and p["busy"] is False
    finally:
        srv.shutdown()


def test_registration_is_skipped_when_not_configured(monkeypatch):
    from agents.trajectory import planner_registration as pr
    monkeypatch.delenv("ORCHESTRATOR_URL", raising=False)
    assert "skipped" in pr.beat(lambda: {})
