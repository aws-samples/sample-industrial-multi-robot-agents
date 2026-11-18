"""POST /a2a/task/save: the task file lands on disk and nothing is dispatched.

The endpoint the panel's task wizard finishes on. It shares every line of
validation with ``/a2a/task/submit`` (same helper, ``_persist_submitted_program``)
and differs in exactly one way: no run is started. That difference is the whole
point, so it is what is asserted — no ``run_id``, no ``run_task`` call, no busy
cell afterwards, and a save that works *while* the cell is running.

Lives apart from ``tests/test_task_wizard.py`` because this half needs flask and
that half needs node; a module-level ``importorskip`` in the shared file would
skip the JS tests on every host without flask.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "dashboard" / "web_control_panel.py"

flask = pytest.importorskip("flask")

from agents.orchestrator import run_registry, server as srv  # noqa: E402

CELL = srv.CELL_ID


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(run_registry, "_REGISTRY", run_registry.RunRegistry())
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    (tmp_path / CELL / "tasks").mkdir(parents=True)
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as client:
        yield client


class TestTaskSaveWritesTheFileAndStartsNoRun:

    def test_it_writes_the_program_where_the_planner_looks_for_it(self, client, tmp_path):
        resp = client.post("/a2a/task/save", json={
            "program": "wizard_prog", "description": "authored",
            "operations": [{"id": "W01", "type": "weld", "position": [0, 0, 1],
                            "approach_vector": [0, 0, -1]}]})
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["success"] is True
        assert body["commanded"] is False and body["ran"] is False
        assert body["num_operations"] == 1
        saved = tmp_path / CELL / "tasks" / "wizard_prog.json"
        assert saved.is_file(), "the task file was not written"
        assert body["saved"] == str(saved)
        persisted = json.loads(saved.read_text())
        assert persisted["program"] == "wizard_prog"
        assert persisted["cell_id"] == CELL, \
            "the file must name the cell it runs on"
        assert persisted["operations"][0]["id"] == "W01"

    def test_it_registers_no_run_at_all(self, client, monkeypatch):
        """The whole point: a save is not a dispatch. If it registered a run, the
        cell would read as busy and the next real run would be refused."""
        called = []
        monkeypatch.setattr(srv, "run_task",
                            lambda *a, **k: called.append(a) or {"success": True})
        resp = client.post("/a2a/task/save", json={
            "program": "p", "operations": [{"id": "W01", "type": "weld"}]})
        assert resp.status_code == 200
        assert not called, "saving a task program ran it"
        assert "run_id" not in resp.get_json(), \
            "a save must not report a run id — there is no run"
        assert run_registry.active(CELL) is None, \
            "saving a task program left the cell marked busy"

    def test_it_saves_while_the_cell_is_busy(self, client):
        """Authoring the next program is safe during a run — the operator should
        not have to stop the cell to write a file."""
        run_registry.start(CELL, "task", program="already-running")
        resp = client.post("/a2a/task/save", json={
            "program": "next_prog", "operations": [{"id": "W01", "type": "weld"}]})
        assert resp.status_code == 200, (
            f"a save was refused because the cell was busy: {resp.get_json()}")

    def test_it_applies_the_same_validation_as_submit(self, client):
        for bad, why in ((({"operations": [{"id": "W01"}]}), "no program name"),
                         ({"program": "p", "operations": []}, "no operations"),
                         ({"program": "../etc/passwd",
                           "operations": [{"id": "W01"}]}, "a path, not a name"),
                         ({"program": "a b", "operations": [{"id": "W01"}]},
                          "a space in the file name")):
            resp = client.post("/a2a/task/save", json=bad)
            assert resp.status_code == 400, f"accepted {why}: {resp.get_json()}"

    def test_it_refuses_to_replace_an_existing_program_silently(self, client, tmp_path):
        existing = tmp_path / CELL / "tasks" / "existing.json"
        existing.write_text('{"program": "existing", "operations": []}')
        resp = client.post("/a2a/task/save", json={
            "program": "existing", "operations": [{"id": "W01", "type": "weld"}]})
        assert resp.status_code == 409, resp.get_json()
        assert "overwrite" in json.dumps(resp.get_json()).lower()
        # Untouched: a hand-authored program is the only copy of itself.
        assert json.loads(existing.read_text())["operations"] == []

        resp = client.post("/a2a/task/save", json={
            "program": "existing", "overwrite": True,
            "operations": [{"id": "W01", "type": "weld"}]})
        assert resp.status_code == 200, resp.get_json()
        assert len(json.loads(existing.read_text())["operations"]) == 1

    def test_control_fields_never_reach_the_file(self, client, tmp_path):
        client.post("/a2a/task/save", json={
            "program": "p", "overwrite": True, "model": "opus-5", "async": True,
            "operations": [{"id": "W01", "type": "weld"}]})
        persisted = json.loads((tmp_path / CELL / "tasks" / "p.json").read_text())
        for control in ("overwrite", "model", "async", "run_id"):
            assert control not in persisted, \
                f"{control!r} is a server control field, not part of the program"


def test_the_dashboard_proxies_the_save_endpoint() -> None:
    """The wizard posts to the panel, which relays to the orchestrator. A missing
    proxy is a 404 the operator meets only at the last step of the wizard."""
    source = PANEL.read_text()
    assert '@app.route("/api/a2a/task/save", methods=["POST"])' in source, \
        "the dashboard does not proxy /a2a/task/save"
    proxy = source[source.index('@app.route("/api/a2a/task/save"'):]
    proxy = proxy[:proxy.index("@app.route", 20)]
    assert '_a2a_post("/a2a/task/save"' in proxy, \
        "the proxy relays to the wrong upstream path"
    assert "async" not in proxy, \
        "a save has no run to dispatch, so it must not ask for async"
