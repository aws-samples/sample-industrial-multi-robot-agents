"""One planning run per cell: the lock, and every entry point that takes it."""
from __future__ import annotations

import threading

import pytest

from agents.trajectory import cell_lock


@pytest.fixture(autouse=True)
def _clean():
    cell_lock._HOLDERS.clear()
    yield
    cell_lock._HOLDERS.clear()


def test_a_second_holder_is_refused_and_the_first_keeps_the_cell():
    assert cell_lock.acquire("c1", "run_a")
    assert not cell_lock.acquire("c1", "run_b")
    assert cell_lock.holder_of("c1")["holder"] == "run_a"


def test_only_the_holder_releases():
    cell_lock.acquire("c1", "run_a")
    cell_lock.release("c1", "run_b")
    assert cell_lock.holder_of("c1")["holder"] == "run_a"
    cell_lock.release("c1", "run_a")
    assert cell_lock.holder_of("c1") is None


def test_cells_are_independent():
    assert cell_lock.acquire("c1", "run_a")
    assert cell_lock.acquire("c2", "run_b")


def test_held_raises_when_busy_and_releases_on_error():
    with pytest.raises(RuntimeError):
        with cell_lock.held("c1", "run_a"):
            with pytest.raises(cell_lock.CellBusy):
                with cell_lock.held("c1", "run_b"):
                    pass
            raise RuntimeError("boom")
    assert cell_lock.holder_of("c1") is None


def test_acquire_is_atomic_under_contention():
    wins = []
    barrier = threading.Barrier(16)

    def race(i):
        barrier.wait()
        if cell_lock.acquire("c1", f"run_{i}"):
            wins.append(i)

    threads = [threading.Thread(target=race, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1


def test_plan_refuses_a_busy_cell_without_planning(monkeypatch):
    from agents.trajectory import agent
    called = []
    monkeypatch.setattr(agent, "_plan", lambda *a, **k: called.append(1) or {"success": True})
    cell_lock.acquire("c1", "someone_else")
    out = agent.plan("c1", [], tree_id="bt_x")
    assert out["success"] is False and out["error_kind"] == "cell_busy"
    assert "someone_else" in out["error"]
    assert not called


def test_plan_releases_the_cell_when_the_run_raises(monkeypatch):
    from agents.trajectory import agent

    def boom(*a, **k):
        raise RuntimeError("planner crashed")

    monkeypatch.setattr(agent, "_plan", boom)
    with pytest.raises(RuntimeError):
        agent.plan("c1", [], tree_id="bt_x")
    assert cell_lock.holder_of("c1") is None


def test_plan_holds_the_cell_for_the_run(monkeypatch):
    from agents.trajectory import agent
    seen = {}
    monkeypatch.setattr(agent, "_plan",
                        lambda *a, **k: seen.update(h=cell_lock.holder_of("c1")) or {})
    agent.plan("c1", [], tree_id="bt_x")
    assert seen["h"]["holder"] == "bt_x"
    assert cell_lock.holder_of("c1") is None


def test_replan_operation_is_refused_while_the_cell_is_planned():
    from agents.trajectory.tools import plan_edit
    cell_lock.acquire("c1", "bt_running")
    with pytest.raises(plan_edit.EditError, match="bt_running"):
        plan_edit.replan_operation("c1", "bt_old", "op1")
    assert cell_lock.holder_of("c1")["holder"] == "bt_running"


def test_generated_tree_ids_are_unique_within_one_second():
    from agents.trajectory.tools.trajectory_cache import new_tree_id
    ids = {new_tree_id("c1") for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("bt_") and i.endswith("_c1") for i in ids)
