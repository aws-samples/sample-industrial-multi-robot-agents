"""A planner container runs ONE planning run at a time.

* A new run (plan_trajectories), a resume (resume_planning) and a re-plan
  (replan_operation) each claim the planner atomically; while one holds it the
  others are refused with ``planner_busy`` and who holds it — even when twenty
  submissions arrive at once, exactly one wins.
* ``planner_status`` and ``/health`` say whether the planner is free.
* The running job can be stopped by id: ``cancel_planning(tree_id)`` (discards)
  or ``pause_planning(tree_id)`` (checkpoints) — and then the planner is free. A
  run accepted but not yet started is cancellable too: its worker plans nothing.

Host-runnable: the LLM agent and the geometry stack are stubbed out; what is
under test is the job bookkeeping of agents/trajectory/mcp_server.py.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from unittest.mock import MagicMock

import pytest

STUBS = ("agents.trajectory.agent", "agents.trajectory.tools.pinocchio",
         "agents.trajectory.tools.planning_tools", "agents.trajectory.tools.knowledge_base")


@pytest.fixture
def ms(monkeypatch, tmp_path):
    for name in STUBS:
        if name not in sys.modules or isinstance(sys.modules[name], MagicMock):
            monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.delitem(sys.modules, "agents.trajectory.mcp_server", raising=False)
    monkeypatch.setenv("PLANNER_SCRATCH_PATH", str(tmp_path / "scratch"))
    monkeypatch.setenv("PLANNER_ID", "planner-test")
    import agents.trajectory.mcp_server as server
    monkeypatch.setattr(server, "_PLANNER_ID", "planner-test")
    server._PLAN_JOBS.clear()
    yield server
    server._PLAN_JOBS.clear()
    sys.modules.pop("agents.trajectory.mcp_server", None)


def test_the_planner_is_claimed_once(ms):
    assert ms._reserve("bt_1_c", cell_id="c") is None
    holder = ms._reserve("bt_2_d", cell_id="d")
    assert holder["tree_id"] == "bt_1_c"                     # another CELL is refused too
    ms._set_job("bt_1_c", status="done")
    assert ms._reserve("bt_2_d", cell_id="d") is None


def test_twenty_simultaneous_submissions_one_wins(ms):
    wins, barrier = [], threading.Barrier(20)

    def submit(i):
        barrier.wait()
        if ms._reserve(f"bt_{i}_c", cell_id="c") is None:
            wins.append(i)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(wins) == 1


def _plan(ms, cell="cellA"):
    return asyncio.run(ms.plan_trajectories(cell, [{"id": "W1", "position": [0, 0, 1]}]))


def test_a_second_plan_is_refused_while_one_runs(ms, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(ms, "_run_plan_job", lambda *a, **k: release.wait(5))
    monkeypatch.setattr(ms, "_planning_model", lambda m=None: "test-model")

    async def both():
        first = await ms.plan_trajectories("cellA", [{"id": "W1", "position": [0, 0, 1]}])
        second = await ms.plan_trajectories("cellB", [{"id": "W2", "position": [0, 0, 1]}])
        status = ms.planner_status()
        release.set()
        return first, second, status

    first, second, status = asyncio.run(both())
    assert first["accepted"] is True
    assert second["accepted"] is False and second["error_kind"] == "planner_busy"
    assert second["busy_with"]["tree_id"] == first["tree_id"]
    assert status["busy"] is True and status["active"]["tree_id"] == first["tree_id"]
    assert status["planner_id"] == "planner-test"


def test_resume_and_replan_are_refused_while_a_run_holds_the_planner(ms, monkeypatch):
    from agents.trajectory import plan_checkpoint
    monkeypatch.setattr(ms, "_planning_model", lambda m=None: "test-model")
    monkeypatch.setattr(plan_checkpoint, "describe", lambda c, t: {"resumable": True, "step": 3})
    assert ms._reserve("bt_live_cellA", cell_id="cellA") is None
    resumed = asyncio.run(ms.resume_planning("bt_old_cellB", cell_id="cellB"))
    assert resumed["accepted"] is False and resumed["error_kind"] == "planner_busy"
    edited = asyncio.run(ms.replan_operation("cellB", "bt_old_cellB", "W1"))
    assert edited["success"] is False and edited["error_kind"] == "planner_busy"


def test_a_running_job_can_be_cancelled_or_paused_by_id_and_frees_the_planner(ms):
    assert ms._reserve("bt_9_cellA", cell_id="cellA") is None
    out = ms.cancel_planning("bt_9_cellA", reason="test")
    assert "bt_9_cellA" in str(out)
    assert ms._PLAN_JOBS["bt_9_cellA"]["status"] == "cancelling"
    ms._run_plan_job("cellA", [], None, "bt_9_cellA")         # its worker starts...
    assert ms._PLAN_JOBS["bt_9_cellA"]["status"] == "cancelled"   # ...and plans nothing
    assert ms.planner_status()["busy"] is False
    assert ms._reserve("bt_10_cellA", cell_id="cellA") is None
    paused = ms.pause_planning("bt_10_cellA", reason="test")
    assert "bt_10_cellA" in str(paused)
