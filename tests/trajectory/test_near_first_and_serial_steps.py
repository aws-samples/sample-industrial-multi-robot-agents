"""Near-first routes, and every idle-with-work step reported.

* ``_tsp_2opt(start_near=base)`` walks the same optimal open path from the end
  nearest the arm, so arms on opposite sides of a part start far apart.
* ``_serial_steps`` names each step in which an arm that still has welds does
  nothing — the parallel time the planner must win back.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools.task_allocation import _tsp_2opt  # noqa: E402


def _row(ys):
    return [{"id": f"P{i}", "position": [0.0, y, 1.0]} for i, y in enumerate(ys)]


def _ys(seq):
    return [p["position"][1] for p in seq]


def test_the_route_starts_at_the_weld_closest_to_the_arm():
    pts = _row([-1.8, -1.5, -1.2, -0.9, -0.6])
    near_low = _tsp_2opt(pts, start_near=[0.0, -3.3, 0.0])
    near_high = _tsp_2opt(pts, start_near=[0.0, 1.0, 0.0])
    assert _ys(near_low)[0] == -1.8 and _ys(near_high)[0] == -0.6
    assert sorted(_ys(near_low)) == sorted(_ys(near_high))   # same welds
    assert _ys(near_low) == _ys(near_high)[::-1]             # same path, reversed


def test_without_a_base_the_route_is_unchanged():
    pts = _row([-1.8, -1.5, -1.2])
    assert _tsp_2opt(pts) == _tsp_2opt(list(pts))


sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weld_step_gate import bt  # noqa: E402,F401 (LangChain-stub fixture)


def _step(*pairs):
    return {"mode": "PARALLEL",
            "robots": [{"robot_ns": ns, "operation_id": op} for ns, op in pairs]}


def test_a_serial_program_reports_every_idle_step(bt):
    steps = [_step(("robot1", "A1")), _step(("robot1", "A2")),
             _step(("robot2", "B1")), _step(("robot2", "B2"))]
    assert [s["step"] for s in bt._serial_steps(steps)] == [1, 2]
    assert bt._serial_steps(steps)[0]["idle_with_work"] == ["robot2"]


def test_a_parallel_program_reports_none(bt):
    steps = [_step(("robot1", "A1"), ("robot2", "B1")),
             _step(("robot1", "A2"), ("robot2", "B2")),
             _step(("robot1", "A3"))]
    assert bt._serial_steps(steps) == []
