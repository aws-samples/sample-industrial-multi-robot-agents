"""Host unit tests for N-robot arity across the planning stack.

The cell shipped with two arms, and several stages silently assumed exactly two.
The worst case was MRTA scheduling: Stage 1 allocated work to every robot in
robots.yaml, but Stage 4 collapsed the allocation to the first two robots, so a
third arm's operations were dropped with no error and no cache entry — invisible
to the behaviour-tree assembler, which can only detect operations it was told
about. These tests pin the arity contracts so that cannot regress.

Host-only: they exercise the pure scheduling/geometry math with no ROS, no
Pinocchio and no OMPL.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# rrt_planner imports langchain_core for its @tool decorators; the scheduling and
# geometry helpers under test are pure numpy. Stub the decorator so the module
# imports on a bare host without pulling in the LLM stack.
if "langchain_core.tools" not in sys.modules:
    _lc = types.ModuleType("langchain_core")
    _lct = types.ModuleType("langchain_core.tools")
    _lct.tool = lambda f: f
    sys.modules.setdefault("langchain_core", _lc)
    sys.modules["langchain_core.tools"] = _lct

import numpy as np  # noqa: E402

from agents.trajectory.tools.rrt_planner import (  # noqa: E402
    _regional_allocate,
    _max_distance_schedule,
    _sample_at_time,
)


def _robots(n: int) -> dict:
    """n robots on a ring around the workpiece, so allocation spreads work."""
    out = {}
    for i in range(n):
        a = 2.0 * np.pi * i / max(n, 1)
        out[f"robot{i + 1}"] = {"base_position": [2.0 * np.cos(a), 2.0 * np.sin(a), 0.0]}
    return out


def _ops(n: int) -> list[dict]:
    return [
        {"id": f"W{i}", "position": [0.3 * i, (-1) ** i * 0.5, 1.0],
         "approach_vector": [0, 0, -1]}
        for i in range(n)
    ]


def _ops_around(n: int, nrobots: int) -> list[dict]:
    """Operations spread around the ring so the allocator spreads them too.

    Collinear points all score toward whichever base is nearest the line, which
    makes "did robot3 get work" depend on the heuristic rather than on arity.
    """
    out = []
    for i in range(n):
        a = 2.0 * np.pi * (i % max(nrobots, 1)) / max(nrobots, 1)
        out.append({
            "id": f"W{i}",
            "position": [1.1 * np.cos(a) + 0.05 * i, 1.1 * np.sin(a), 1.0],
            "approach_vector": [0, 0, -1],
        })
    return out


# ─── The regression: scheduling must never lose an operation ───────────────────

@pytest.mark.parametrize("nrobots", [1, 2, 3, 4, 6])
def test_scheduling_loses_no_operations_at_any_robot_count(nrobots):
    """THE regression. Every allocated operation must reach exactly one step.

    Before the N-ary scheduler, robots 3..N were dropped entirely: their welds
    were allocated and then discarded, so those welds would never be performed.
    """
    ops = _ops(11)
    alloc = _regional_allocate(ops, _robots(nrobots))
    sched = _max_distance_schedule(alloc)

    placed = [op["id"] for st in sched for rid, op in st.items() if rid in alloc]
    assert sorted(placed) == sorted(o["id"] for o in ops), (
        f"{nrobots} robots: expected all {len(ops)} ops, got {len(placed)}"
    )
    assert len(placed) == len(set(placed)), "an operation was scheduled twice"


def test_a_third_robots_work_is_not_discarded():
    """Explicit three-robot case: robot3 must appear in the schedule."""
    robots = _robots(3)
    ops = _ops_around(9, 3)
    alloc = _regional_allocate(ops, robots)
    # Force a non-trivial load on robot3 so the assertion is meaningful.
    assert alloc["robot3"], "fixture precondition: robot3 should get work"

    sched = _max_distance_schedule(alloc)
    scheduled_robots = {rid for st in sched for rid in st if rid in alloc}
    assert "robot3" in scheduled_robots
    for op in alloc["robot3"]:
        assert any(st.get("robot3", {}).get("id") == op["id"] for st in sched), (
            f"robot3 operation {op['id']} never reached a step"
        )


def test_step_count_is_the_longest_single_robot_sequence():
    """No robot may be truncated because another has less to do."""
    alloc = _regional_allocate(_ops_around(10, 3), _robots(3))
    sched = _max_distance_schedule(alloc)
    assert len(sched) == max(len(v) for v in alloc.values())


def test_a_step_uses_each_robot_at_most_once():
    """A robot cannot perform two operations simultaneously."""
    alloc = _regional_allocate(_ops_around(12, 4), _robots(4))
    for st in _max_distance_schedule(alloc):
        rids = [rid for rid in st if rid in alloc]
        assert len(rids) == len(set(rids))


def test_single_robot_cell_serialises_every_operation():
    alloc = _regional_allocate(_ops(5), _robots(1))
    sched = _max_distance_schedule(alloc)
    assert len(sched) == 5
    assert all(len([r for r in st if r in alloc]) == 1 for st in sched)


def test_empty_allocation_yields_no_steps():
    assert _max_distance_schedule({"robot1": [], "robot2": []}) == []


def test_schedule_reports_min_inter_robot_distance_when_arms_share_a_step():
    """The parallel-safety hint must survive at N > 2."""
    alloc = _regional_allocate(_ops_around(9, 3), _robots(3))
    sched = _max_distance_schedule(alloc)
    multi = [st for st in sched if len([r for r in st if r in alloc]) > 1]
    assert multi, "fixture precondition: some step should hold 2+ arms"
    assert all("min_inter_robot_dist_m" in st for st in multi)


# ─── Allocation participates all robots ───────────────────────────────────────

@pytest.mark.parametrize("nrobots", [2, 3, 5])
def test_allocation_considers_every_robot_in_the_cell(nrobots):
    alloc = _regional_allocate(_ops(15), _robots(nrobots))
    assert set(alloc) == set(_robots(nrobots))


def test_pinned_operations_are_honoured_beyond_the_second_robot():
    """assigned_robot must work for robot3+, not just the first two arms."""
    ops = _ops(6)
    ops[0]["assigned_robot"] = "robot3"
    ops[1]["assigned_robot"] = "robot3"
    alloc = _regional_allocate(ops, _robots(3))
    ids = [o["id"] for o in alloc["robot3"]]
    assert "W0" in ids and "W1" in ids


# ─── Shared-timeline sampling (what makes N-way validation correct) ────────────

def test_sample_at_time_interpolates_between_waypoints():
    traj = [{"positions": [0.0, 0, 0, 0, 0, 0], "time_s": 0.0},
            {"positions": [1.0, 0, 0, 0, 0, 0], "time_s": 1.0}]
    assert _sample_at_time(traj, 0.5)[0] == pytest.approx(0.5)


def test_sample_at_time_holds_the_final_pose_after_the_end():
    """A finished robot is parked in its final pose and still occupies space.

    If it snapped back to the start instead, a set-validation on the shared
    timeline would compare against a configuration the arm is not in.
    """
    traj = [{"positions": [0.0, 0, 0, 0, 0, 0], "time_s": 0.0},
            {"positions": [1.0, 0, 0, 0, 0, 0], "time_s": 1.0}]
    assert _sample_at_time(traj, 5.0)[0] == pytest.approx(1.0)


def test_sample_at_time_clamps_before_the_start():
    traj = [{"positions": [0.25, 0, 0, 0, 0, 0], "time_s": 2.0}]
    assert _sample_at_time(traj, 0.0)[0] == pytest.approx(0.25)


def test_sample_at_time_handles_an_empty_trajectory():
    assert _sample_at_time([], 1.0).shape[0] == 6


def test_differing_durations_are_compared_at_the_same_real_time():
    """Two arms of different duration must both be sampled on wall-clock time.

    Index-fraction sampling (the previous scheme) would pair the 50%-index of a
    2 s path with the 50%-index of a 10 s path — 1 s against 5 s.
    """
    fast = [{"positions": [float(i), 0, 0, 0, 0, 0], "time_s": i * 1.0} for i in range(3)]
    slow = [{"positions": [float(i), 0, 0, 0, 0, 0], "time_s": i * 5.0} for i in range(3)]
    t = 5.0
    assert _sample_at_time(fast, t)[0] == pytest.approx(2.0)   # finished, parked
    assert _sample_at_time(slow, t)[0] == pytest.approx(1.0)   # one third in
