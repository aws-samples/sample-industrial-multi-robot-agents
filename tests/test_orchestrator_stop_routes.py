"""HTTP contract of the three routes behind the new buttons.

``POST /a2a/plan`` (Planning), ``POST /a2a/abort`` (Stop) and ``POST /a2a/reset``
(Reset) are what the panel — and any operator with curl — actually presses. What
matters here is not that they return 200: it is *which MCP tools each one is
allowed to reach*, because that is the difference between a plan and a motion,
and between a software stop and a latched halt.

So every test below stubs ``_mcp_call`` with a recorder and asserts on the calls
that were made. A route that quietly grew an ``execute_behaviour_tree`` or a
``trigger_estop`` would keep its status code and its envelope, and the button
would mean something new.

The 409/idle cases are here for the same reason: Stop pressed on an idle cell must
succeed (an operator hitting a dead button presses harder, or reaches for the
physical e-stop), and Reset pressed *during* a run must not abort that run and
then refuse to home — which is exactly what it did before it learned to wait.

Skipped on any host that does not ship flask.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from agents.orchestrator import run_registry, server as srv

CELL = srv.CELL_ID

# Tool names that command a robot. None of these may be reached by a plan or a
# software stop; only Reset's homing is allowed to move anything.
MOTION_TOOLS = ("execute_behaviour_tree", "replay_recording", "move_robot",
                "dispatch_trajectory", "trigger_estop")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(run_registry, "_REGISTRY", run_registry.RunRegistry())
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    (tmp_path / CELL / "tasks").mkdir(parents=True)
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as c:
        yield c


@pytest.fixture
def calls(monkeypatch):
    """Record every MCP call, and answer plausibly for the tools these routes use."""
    recorded: list[tuple[str, str, dict]] = []

    def fake(url, name, arguments, timeout=None):
        recorded.append((url, name, dict(arguments or {})))
        if name == "plan_trajectories":
            return {"accepted": True, "tree_id": "bt_test_1"}
        if name == "get_plan_status":
            return {"status": "done", "tree_id": "bt_test_1",
                    "num_operations": 3, "model": "test"}
        if name == "cancel_planning":
            return {"success": True, "cancelled": [], "still_planning": []}
        if name == "abort_execution":
            return {"success": True, "aborted": False,
                    "reason": "nothing is executing"}
        if name == "home_robots":
            return {"success": True, "robots": ["robot1", "robot2"],
                    "cleared_estop": True}
        return {"success": True}

    monkeypatch.setattr(srv, "_mcp_call", fake)
    return recorded


def _program(tmp_path, name="welds", operations=(("op1", "weld"),)):
    ops = [{"id": oid, "type": otype, "target": {"position": [0.5, 0.0, 0.9]}}
           for oid, otype in operations]
    path = tmp_path / CELL / "tasks" / f"{name}.json"
    path.write_text(json.dumps({"program": name, "operations": ops}))
    return name


def _names(recorded) -> list[str]:
    return [n for _url, n, _a in recorded]


# ─── /a2a/plan — plans the named program, commands nothing ───────────────────

class TestPlan:
    def test_400_without_a_program(self, client):
        """No default. Which welds get planned is the operator's choice."""
        resp = client.post("/a2a/plan", json={})
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False
        assert "program" in body["error"].lower()
        assert "/a2a/tasks" in body.get("hint", ""), \
            "the refusal does not say where the list of programs is"

    def test_400_for_a_program_name_that_is_a_path(self, client, calls):
        """``program`` becomes a filename. A traversal must be refused by name,
        before anything opens it."""
        resp = client.post("/a2a/plan", json={"program": "../../etc/passwd"})
        assert resp.status_code == 400
        assert "invalid program name" in resp.get_json()["error"]
        assert calls == [], "a rejected program name still reached the agents"

    def test_404_for_a_program_that_does_not_exist(self, client, calls):
        resp = client.post("/a2a/plan", json={"program": "no_such_program"})
        assert resp.status_code == 404
        assert calls == [], "a missing program still started a planning job"

    def test_400_for_a_program_with_no_operations(self, client, tmp_path, calls):
        """Planning nothing for minutes and reporting success is worse than a 400."""
        (tmp_path / CELL / "tasks" / "empty.json").write_text(
            json.dumps({"program": "empty", "operations": []}))
        resp = client.post("/a2a/plan", json={"program": "empty"})
        assert resp.status_code == 400
        assert "no operations" in resp.get_json()["error"]
        assert calls == []

    def test_a_finished_plan_reports_its_tree_and_that_nothing_moved(
            self, client, tmp_path, calls):
        name = _program(tmp_path)
        resp = client.post("/a2a/plan", json={"program": name})
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["success"] is True
        assert body["tree_id"] == "bt_test_1"
        assert body["commanded"] is False, \
            "a plan-only run does not state that nothing was commanded"
        assert "replay" in body["next"].lower(), \
            "the reply does not tell the operator how to execute the plan"
        assert body["program"] == name

    def test_planning_reaches_the_planner_and_no_robot(self, client, tmp_path, calls):
        name = _program(tmp_path)
        client.post("/a2a/plan", json={"program": name})
        assert "plan_trajectories" in _names(calls)
        for tool in MOTION_TOOLS:
            assert tool not in _names(calls), \
                f"Planning called '{tool}' — the button that plans must not move arms"

    def test_the_operations_planned_are_the_ones_in_the_chosen_program(
            self, client, tmp_path, calls):
        """The failure this catches is planning the right count of the wrong welds."""
        name = _program(tmp_path, "doors", operations=(("d1", "weld"), ("d2", "weld")))
        _program(tmp_path, "other", operations=(("x1", "weld"),))
        client.post("/a2a/plan", json={"program": name})
        args = next(a for _u, n, a in calls if n == "plan_trajectories")
        assert [op["id"] for op in args["operations"]] == ["d1", "d2"]

    def test_async_returns_202_and_a_run_id_to_poll(self, client, tmp_path, calls):
        name = _program(tmp_path)
        resp = client.post("/a2a/plan", json={"program": name, "async": True})
        assert resp.status_code == 202
        env = resp.get_json()
        assert env["done"] is False
        assert env["run_id"].startswith(CELL + "_")

    def test_a_second_planning_press_is_refused_not_queued(self, client, tmp_path):
        """Two agents planning one cell at once is the state the run registry
        exists to prevent."""
        name = _program(tmp_path)
        started = run_registry.start(CELL, "task", program="already-running")
        resp = client.post("/a2a/plan", json={"program": name})
        assert resp.status_code == 409
        assert resp.get_json()["active_run"]["run_id"] == started["run_id"]

    def test_a_refused_plan_request_is_reported_as_a_failure(
            self, client, tmp_path, monkeypatch):
        """The trajectory agent says no (already planning, bad backend). That must
        not surface as a successful plan with no tree."""
        def refuse(url, name, arguments, timeout=None):
            if name == "plan_trajectories":
                return {"accepted": False, "error": "backend 'nope' is unknown"}
            return {"success": True}
        monkeypatch.setattr(srv, "_mcp_call", refuse)
        prog = _program(tmp_path)
        resp = client.post("/a2a/plan", json={"program": prog, "backend": "nope"})
        body = resp.get_json()
        assert body["success"] is False
        assert body["commanded"] is False
        assert "nope" in body["error"]

    def test_a_cancelled_plan_is_not_reported_as_success(
            self, client, tmp_path, monkeypatch):
        """Stop was pressed. The run ended, but there is no plan — and the reply
        must not read like there is one."""
        def cancelled(url, name, arguments, timeout=None):
            if name == "plan_trajectories":
                return {"accepted": True, "tree_id": "bt_test_1"}
            if name == "get_plan_status":
                return {"status": "cancelled", "error": "cancelled by operator"}
            return {"success": True}
        monkeypatch.setattr(srv, "_mcp_call", cancelled)
        prog = _program(tmp_path)
        resp = client.post("/a2a/plan", json={"program": prog})
        body = resp.get_json()
        assert body["success"] is False
        assert body["status"] == "cancelled"

    def test_a_chosen_backend_is_passed_as_a_constraint(self, client, tmp_path, calls):
        prog = _program(tmp_path)
        client.post("/a2a/plan", json={"program": prog, "backend": "ompl"})
        args = next(a for _u, n, a in calls if n == "plan_trajectories")
        assert args["constraints"]["trajectory_backend"] == "ompl"


# ─── /a2a/abort — Stop: software only ────────────────────────────────────────

class TestAbort:
    def test_it_stops_both_halves_of_the_system(self, client, calls):
        """Planning lives in one process and the executing tree in another; a Stop
        that only reached one of them would leave the other running."""
        resp = client.post("/a2a/abort", json={})
        assert resp.status_code == 200
        assert "cancel_planning" in _names(calls)
        assert "abort_execution" in _names(calls)

    def test_it_never_trips_the_estop(self, client, calls):
        resp = client.post("/a2a/abort", json={})
        for tool in MOTION_TOOLS:
            assert tool not in _names(calls), \
                f"Stop called '{tool}'; it is a software stop, not a halt"
        assert resp.get_json()["estop"] is False

    def test_it_says_that_an_arm_in_motion_keeps_moving(self, client, calls):
        """The operator's mental model is the safety property here. If they read
        this as 'the cell is stopped', they will walk up to it."""
        note = client.post("/a2a/abort", json={}).get_json()["note"]
        assert "finishes" in note.lower() or "already-dispatched" in note.lower(), \
            f"the Stop reply does not warn that in-flight motion continues: {note}"
        assert "/a2a/stop" in note, \
            "the Stop reply does not point at the endpoint that does halt the arms"

    def test_stopping_an_idle_cell_succeeds(self, client, calls):
        """Nothing running is not an error. A 4xx here reads as a broken button."""
        resp = client.post("/a2a/abort", json={})
        body = resp.get_json()
        assert resp.status_code == 200
        assert body["success"] is True
        assert body["execution"]["aborted"] is False
        assert body["active_run"] is None

    def test_pressing_it_twice_is_the_same_as_pressing_it_once(self, client, calls):
        first = client.post("/a2a/abort", json={}).get_json()
        second = client.post("/a2a/abort", json={}).get_json()
        assert first["success"] is second["success"] is True

    def test_it_does_not_take_a_run_slot(self, client, calls):
        """Stop must work *while* the cell is busy — that is when it is pressed."""
        run_registry.start(CELL, "task", program="running")
        resp = client.post("/a2a/abort", json={})
        assert resp.status_code == 200, \
            "Stop was refused because the cell was busy; it is only ever pressed " \
            "on a busy cell"
        assert resp.get_json()["active_run"]["program"] == "running"

    def test_the_reason_reaches_both_agents(self, client, calls):
        client.post("/a2a/abort", json={"reason": "operator stop from dashboard"})
        for _u, name, args in calls:
            if name in ("cancel_planning", "abort_execution"):
                assert args["reason"] == "operator stop from dashboard"


# ─── /a2a/reset — Stop, then home. This one moves. ───────────────────────────

class TestReset:
    def test_it_stops_first_and_then_homes(self, client, calls):
        resp = client.post("/a2a/reset", json={})
        assert resp.status_code == 200, resp.get_json()
        names = _names(calls)
        assert names.index("cancel_planning") < names.index("home_robots"), \
            "Reset homes the arms before stopping the activity — two things would " \
            "be driving the cell at once"
        assert names.index("abort_execution") < names.index("home_robots")

    def test_it_reports_that_it_cleared_the_estop_latch(self, client, calls):
        """It has to clear the latch to move at all. An operator who does not know
        that will treat a reset cell as a stopped one."""
        body = client.post("/a2a/reset", json={}).get_json()
        assert body["cleared_estop"] is True
        note = body["note"].lower().replace("-", "")
        assert "estop" in note, \
            f"the reset reply does not mention the latch it cleared: {body['note']}"
        assert body["success"] is True

    def test_it_waits_for_the_run_it_aborted_instead_of_answering_409(
            self, client, calls, monkeypatch):
        """The bug this encodes: Reset aborts the run, the run's thread takes a
        moment to unwind, and the registry still holds the cell — so Reset used to
        stop everything and then refuse to home, arms left mid-cell."""
        monkeypatch.setattr(srv, "_RESET_FREE_TIMEOUT_S", 5.0)
        envelope = run_registry.start(CELL, "task", program="running")

        def unwind():
            time.sleep(0.4)
            run_registry.finish(envelope["run_id"], {"success": False})

        threading.Thread(target=unwind, daemon=True).start()
        resp = client.post("/a2a/reset", json={})
        assert resp.status_code == 200, \
            f"Reset refused to home after aborting the run: {resp.get_json()}"
        assert "home_robots" in _names(calls)
        assert resp.get_json()["waited_for_run_s"] > 0

    def test_a_run_that_will_not_end_gets_a_409_and_the_stop_still_happened(
            self, client, calls, monkeypatch):
        """Declaring the cell idle to force the homing through would drive arms
        while something else is still commanding them. 409 is the honest answer —
        but the abort must not be silently skipped along with it."""
        monkeypatch.setattr(srv, "_RESET_FREE_TIMEOUT_S", 0.3)
        run_registry.start(CELL, "task", program="wedged")
        resp = client.post("/a2a/reset", json={})
        assert resp.status_code == 409
        assert "abort_execution" in _names(calls), \
            "Reset gave up without stopping anything; the operator pressed a " \
            "button that did nothing at all"
        assert "home_robots" not in _names(calls), \
            "Reset homed the arms while another run still held the cell"

    def test_async_reset_returns_202(self, client, calls):
        resp = client.post("/a2a/reset", json={"async": True})
        assert resp.status_code == 202
        assert resp.get_json()["run_id"].startswith(CELL + "_")

    def test_homing_failure_is_reported_as_failure(self, client, monkeypatch):
        """An arm that could not be cleared is left where it stands. Reporting
        success would tell the operator the cell is parked when it is not."""
        def stuck(url, name, arguments, timeout=None):
            if name == "home_robots":
                return {"success": False,
                        "error": "robot2 could not be homed clear of robot1"}
            return {"success": True, "cancelled": [], "aborted": False}
        monkeypatch.setattr(srv, "_mcp_call", stuck)
        body = client.post("/a2a/reset", json={}).get_json()
        assert body["success"] is False
        assert "robot2" in json.dumps(body["home"])
