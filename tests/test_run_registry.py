"""What ``agents.orchestrator.run_registry`` promises the rest of the system.

The registry is the single owner of "what is running on this cell, and how did
it go?" — so every property this test file exercises is one another part of the
codebase (the A2A server, the concurrency guard, ``/a2a/result``) reads back
without asking twice. A regression in any of them is a regression in a promise,
not an implementation detail.

Host-runnable (pure Python — no flask, no ROS).
"""
from __future__ import annotations

import threading
import time

import pytest

from agents.orchestrator.run_registry import (
    CellBusy, FAILED, RUNNING, RunRegistry, SUCCEEDED,
)


CELL = "welding_cell_2_small"


def test_start_returns_envelope_running_and_bound_to_cell() -> None:
    reg = RunRegistry()
    env = reg.start(CELL, "task", program="p")
    assert env["cell_id"] == CELL
    assert env["kind"] == "task"
    assert env["program"] == "p"
    assert env["status"] == RUNNING
    assert env["done"] is False
    assert env["run_id"].startswith(CELL + "_")


def test_start_refuses_second_run_on_same_cell() -> None:
    """The concurrency guard is a *safety* answer, not availability. Two
    schedulers driving one cell each would enforce their own gates and neither
    would see the other's motion, so a busy cell is refused."""
    reg = RunRegistry()
    first = reg.start(CELL, "task")
    with pytest.raises(CellBusy) as exc:
        reg.start(CELL, "move")
    assert exc.value.cell_id == CELL
    assert exc.value.active["run_id"] == first["run_id"]


def test_start_allows_a_new_run_after_the_previous_finished() -> None:
    reg = RunRegistry()
    first = reg.start(CELL, "task")
    reg.finish(first["run_id"], {"success": True})
    second = reg.start(CELL, "move")  # must not raise
    assert second["run_id"] != first["run_id"]


def test_finish_marks_success_only_if_result_explicitly_says_so() -> None:
    """The tools this wraps report failure in-band as ``{"success": False}``
    with HTTP 200 — treating "it returned" as "it worked" would mark every
    failed execution succeeded. A result with no ``success`` key is treated
    as a failure for the same reason."""
    reg = RunRegistry()

    ok = reg.start(CELL + "_a", "task")
    reg.finish(ok["run_id"], {"success": True, "result": "done"})
    assert reg.get(ok["run_id"])["status"] == SUCCEEDED

    reg2 = RunRegistry()
    bad = reg2.start(CELL + "_b", "task")
    reg2.finish(bad["run_id"], {"success": False, "error": "collision"})
    assert reg2.get(bad["run_id"])["status"] == FAILED
    assert reg2.get(bad["run_id"])["error"] == "collision"

    reg3 = RunRegistry()
    weird = reg3.start(CELL + "_c", "task")
    reg3.finish(weird["run_id"], {"something": "else"})   # no success key
    assert reg3.get(weird["run_id"])["status"] == FAILED


def test_finish_prefers_the_runs_own_elapsed_measurement() -> None:
    reg = RunRegistry()
    env = reg.start(CELL, "task")
    reg.finish(env["run_id"], {"success": True, "elapsed_s": 12.5})
    assert reg.get(env["run_id"])["elapsed_s"] == 12.5


def test_active_names_the_holder_for_the_operator() -> None:
    reg = RunRegistry()
    assert reg.active(CELL) is None
    env = reg.start(CELL, "replay", program="bt_1")
    active = reg.active(CELL)
    assert active is not None
    assert active["run_id"] == env["run_id"]
    reg.finish(env["run_id"], {"success": True})
    assert reg.active(CELL) is None


def test_evict_never_drops_a_running_run() -> None:
    """A running run is the poller's only handle on it AND the concurrency
    guard's entry. Dropping it would lose both — an availability failure
    caused by bookkeeping, exactly what the guard exists to avoid being."""
    reg = RunRegistry(max_remembered=3)

    # Fill the ring with FINISHED runs from cells that are now free.
    for i in range(3):
        e = reg.start(f"c{i}", "task")
        reg.finish(e["run_id"], {"success": True})

    # Now a live run on a fresh cell, and more finished ones behind it.
    live = reg.start("live_cell", "task")
    for i in range(3, 8):
        e = reg.start(f"c{i}", "task")
        reg.finish(e["run_id"], {"success": True})

    assert reg.get(live["run_id"]) is not None, "running run was evicted"
    assert reg.get(live["run_id"])["status"] == RUNNING


def test_recent_returns_newest_first_and_filters_by_cell() -> None:
    reg = RunRegistry()
    a = reg.start("cell_a", "task"); reg.finish(a["run_id"], {"success": True})
    b = reg.start("cell_b", "task"); reg.finish(b["run_id"], {"success": True})
    c = reg.start("cell_a", "task"); reg.finish(c["run_id"], {"success": True})

    only_a = [r["run_id"] for r in reg.recent("cell_a")]
    assert only_a == [c["run_id"], a["run_id"]]

    everything = [r["run_id"] for r in reg.recent()]
    assert everything == [c["run_id"], b["run_id"], a["run_id"]]


def test_concurrent_starts_on_one_cell_lose_all_but_one() -> None:
    """Threads race the guard; only one may win. Any other outcome is a race
    condition in the guard itself."""
    reg = RunRegistry()
    starts = 32
    winners: list[str] = []
    losers = 0
    lock = threading.Lock()
    barrier = threading.Barrier(starts)

    def _try() -> None:
        nonlocal losers
        barrier.wait()
        try:
            env = reg.start(CELL, "task")
        except CellBusy:
            with lock:
                losers += 1
        else:
            with lock:
                winners.append(env["run_id"])

    threads = [threading.Thread(target=_try) for _ in range(starts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1
    assert losers == starts - 1


def test_widen_run_id_collision_never_loses_the_first_run() -> None:
    """Two runs minted in the same second collide on id. The registry MUST
    widen the second rather than overwrite the first — a lost id is a lost
    log link and a lost concurrency entry."""
    reg = RunRegistry()
    fixed_ts = "welding_cell_2_small_20260904_120000"

    first = reg.start(CELL, "task", run_id=fixed_ts)
    reg.finish(first["run_id"], {"success": True})

    # Second start with the same suggested id.
    second = reg.start(CELL, "task", run_id=fixed_ts)
    assert second["run_id"] != first["run_id"]
    assert reg.get(first["run_id"]) is not None
    assert reg.get(second["run_id"]) is not None


def test_run_in_background_records_a_raised_exception_as_failed() -> None:
    """A worker that raises must land in FAILED, not RUNNING — or the cell
    stays held forever."""
    from agents.orchestrator.run_registry import run_in_background
    reg = RunRegistry()
    env = reg.start(CELL, "task")

    def _boom() -> dict:
        raise RuntimeError("nope")

    thread = run_in_background(env, _boom, reg=reg)
    thread.join(timeout=5)
    assert not thread.is_alive()

    final = reg.get(env["run_id"])
    assert final["status"] == FAILED
    assert final["done"] is True
    assert "nope" in final["error"]


def test_unknown_kind_is_a_programmer_error_not_a_mystery_string() -> None:
    reg = RunRegistry()
    with pytest.raises(ValueError):
        reg.start(CELL, "not-a-kind")
