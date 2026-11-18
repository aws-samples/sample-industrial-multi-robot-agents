"""End-to-end HTTP shape of the orchestrator A2A server.

The point of this file is to lock the *contract* — status codes, envelope
keys, the 409-busy shape — so a future refactor cannot quietly change what a
poller sees. It boots the Flask app in-process and hits it with the test
client; the MCP calls to the cell controller and trajectory agent are stubbed,
because "does the server refuse a second run on a busy cell" is a property of
the server, not of the network stack it dispatches to.

Skipped on any host that does not ship flask.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from agents.orchestrator import run_registry, server as srv


CELL = srv.CELL_ID


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Reset the registry between tests: it is a process-wide singleton and a
    # leftover RUNNING run from a previous test would refuse the next one on
    # the same cell for the wrong reason.
    monkeypatch.setattr(run_registry, "_REGISTRY", run_registry.RunRegistry())
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    (tmp_path / CELL / "tasks").mkdir(parents=True)
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as client:
        yield client


# ─── /health ───────────────────────────────────────────────────────────────

def test_health_reports_the_cell(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "cell_id": CELL}


# ─── /a2a/run ──────────────────────────────────────────────────────────────

def test_run_400_without_program(client) -> None:
    resp = client.post("/a2a/run", json={})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "program" in body["error"].lower()


def test_run_async_returns_202_with_envelope(client, monkeypatch) -> None:
    # Stub run_task so it looks like a fast success — this test is about the
    # HTTP layer, not the agent loop.
    def fake_run(cell_id, program, model=None, run_id=None):
        return {"success": True, "run_id": run_id, "result": "done",
                "elapsed_s": 0.01, "model": "test"}
    monkeypatch.setattr(srv, "run_task", fake_run)
    resp = client.post("/a2a/run",
                       json={"cell_id": CELL, "program": "p", "async": True})
    assert resp.status_code == 202
    env = resp.get_json()
    assert env["status"] == "running"
    assert env["done"] is False
    assert env["run_id"].startswith(CELL + "_")


def test_run_second_call_while_first_is_running_returns_409(client, monkeypatch) -> None:
    """The concurrency guard is *safety*, not availability — two schedulers
    driving one cell each would enforce their own gates unaware of the
    other's motion."""
    started = run_registry.start(CELL, "task", program="already-running")
    resp = client.post("/a2a/run", json={"cell_id": CELL, "program": "p"})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["success"] is False
    assert body["active_run"]["run_id"] == started["run_id"]


def test_run_unknown_model_returns_400_with_hint(client, monkeypatch) -> None:
    def fake_run(cell_id, program, model=None, run_id=None):
        raise ValueError(f"unknown model {model!r}")
    monkeypatch.setattr(srv, "run_task", fake_run)
    resp = client.post("/a2a/run",
                       json={"cell_id": CELL, "program": "p", "model": "no-such"})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "/a2a/models" in (body.get("hint") or "")


# ─── /a2a/task/submit ──────────────────────────────────────────────────────

def test_submit_400_without_program(client) -> None:
    resp = client.post("/a2a/task/submit", json={"operations": [{"op": 1}]})
    assert resp.status_code == 400


def test_submit_400_without_operations(client) -> None:
    resp = client.post("/a2a/task/submit",
                       json={"program": "p", "operations": []})
    assert resp.status_code == 400


def test_submit_rejects_bad_program_names(client) -> None:
    for bad in ("../etc/passwd", "a b", "p/x", ".hidden", ""):
        resp = client.post("/a2a/task/submit",
                           json={"program": bad, "operations": [{"op": 1}]})
        assert resp.status_code == 400, f"expected 400 for {bad!r}"


def test_submit_409_on_existing_program_unless_overwrite(client, monkeypatch, tmp_path) -> None:
    program_path = tmp_path / CELL / "tasks" / "existing.json"
    program_path.write_text('{"program": "existing"}')

    resp = client.post("/a2a/task/submit",
                       json={"program": "existing", "operations": [{"op": 1}]})
    assert resp.status_code == 409

    # An `overwrite: true` submit should proceed past the 409 gate.
    def fake_run(cell_id, program, model=None, run_id=None):
        return {"success": True, "run_id": run_id, "elapsed_s": 0.01}
    monkeypatch.setattr(srv, "run_task", fake_run)
    resp = client.post("/a2a/task/submit",
                       json={"program": "existing", "operations": [{"op": 1}],
                             "overwrite": True, "async": True})
    assert resp.status_code == 202


def test_submit_persists_the_program_file(client, monkeypatch, tmp_path) -> None:
    def fake_run(cell_id, program, model=None, run_id=None):
        return {"success": True, "run_id": run_id, "elapsed_s": 0.01}
    monkeypatch.setattr(srv, "run_task", fake_run)
    resp = client.post("/a2a/task/submit",
                       json={"program": "new_prog", "operations": [{"step": 1}],
                             "async": True, "model": "opus-5"})
    assert resp.status_code == 202
    saved = tmp_path / CELL / "tasks" / "new_prog.json"
    assert saved.is_file(), "the program file was not written before dispatch"
    persisted = json.loads(saved.read_text())
    # Control fields are stripped; program/cell are forced.
    assert persisted["program"] == "new_prog"
    assert persisted["cell_id"] == CELL
    assert "model" not in persisted
    assert "async" not in persisted


# ─── /a2a/move ─────────────────────────────────────────────────────────────

def test_move_400_without_robot_ns(client) -> None:
    resp = client.post("/a2a/move", json={"x": 0, "y": 0, "z": 1})
    assert resp.status_code == 400


def test_move_400_without_a_target(client) -> None:
    resp = client.post("/a2a/move", json={"robot_ns": "robot1"})
    assert resp.status_code == 400


def test_move_400_with_both_targets(client) -> None:
    resp = client.post("/a2a/move", json={
        "robot_ns": "robot1", "x": 0, "y": 0, "z": 1,
        "joint_positions": [0, 0, 0, 0, 0, 0]})
    assert resp.status_code == 400
    assert "both" in resp.get_json()["error"] or "two targets" in resp.get_json()["error"]


def test_move_400_on_nonpositive_duration(client) -> None:
    resp = client.post("/a2a/move", json={
        "robot_ns": "robot1", "joint_positions": [0]*6, "duration_s": 0})
    assert resp.status_code == 400


# ─── /a2a/result and /a2a/runs ─────────────────────────────────────────────

def test_result_404_for_unknown_run(client) -> None:
    resp = client.get("/a2a/result/nope_20260904_120000")
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["success"] is False
    assert body["done"] is False


def test_result_reads_live_registry(client) -> None:
    env = run_registry.start(CELL, "task", program="p")
    resp = client.get(f"/a2a/result/{env['run_id']}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["run_id"] == env["run_id"]
    assert body["status"] == "running"


def test_runs_lists_live_registry_entries(client) -> None:
    env = run_registry.start(CELL, "task", program="p")
    run_registry.finish(env["run_id"], {"success": True})
    resp = client.get("/a2a/runs")
    assert resp.status_code == 200
    body = resp.get_json()
    ids = [r["run_id"] for r in body["runs"]]
    assert env["run_id"] in ids
    assert body["success"] is True


def test_runs_400_on_bad_limit(client) -> None:
    resp = client.get("/a2a/runs?limit=nope")
    assert resp.status_code == 400
