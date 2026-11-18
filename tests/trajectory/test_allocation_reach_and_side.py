"""Who welds what: the arm behind the approach, within reach, and no arm idle.

Measured on welding_cell_2_small (four arms, two per side of a car body, side-on welds):
the allocation score had its approach term inverted, so it ranked the arm on the
FAR side first — 24 of 27 welds went to the wrong arm, 15 of them beyond that
arm's reach — and nothing checked reach before allocating. The run then
collapsed onto two arms when the model pinned the welds itself.

* A side-on weld goes to the arm standing BEHIND its approach; a top-down weld
  still goes to the nearest arm (where the old sign was invisible).
* No arm is assumed to reach every point: an arm beyond its workspace_radius
  (horizontal, the reach is centred on the shoulder) or on the far side is not a
  candidate; a weld no arm can take is reported ``unreachable``, not forced.
* An idle arm that could work is given work it is a good second choice for.
* run_mrta_pipeline ignores robot pins the model writes into its call; the run's
  own locks (operator / task file) still bind.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if "langchain_core.tools" not in sys.modules:
    _lc = types.ModuleType("langchain_core")
    _lct = types.ModuleType("langchain_core.tools")
    _lct.tool = lambda f: f
    sys.modules.setdefault("langchain_core", _lc)
    sys.modules["langchain_core.tools"] = _lct

from agents.trajectory.tools.task_allocation import (  # noqa: E402
    _alignment, _regional_allocate, _robot_scores, _robot_bases,
)

# welding_cell_2_small's layout: two arms per side of a body along y, facing it.
BODY_ARMS = {
    "robot1": {"base_position": [-3.0, -1.5, 0.0], "workspace_radius": 2.9},
    "robot2": {"base_position": [3.0, -1.5, 0.0], "workspace_radius": 2.9},
    "robot3": {"base_position": [3.0, 1.5, 0.0], "workspace_radius": 2.9},
    "robot4": {"base_position": [-3.0, 1.5, 0.0], "workspace_radius": 2.9},
}


def _side_welds():
    """Left welds approach +x (from the -x arms), right welds -x; along the body."""
    out = []
    for i, y in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0, 1.4]):
        for z in (0.7, 1.4):
            out.append({"id": f"L{i}{int(z*10)}", "position": [-0.85, y, z],
                        "approach_vector": [1.0, 0.0, -0.1]})
            out.append({"id": f"R{i}{int(z*10)}", "position": [0.85, y, z],
                        "approach_vector": [-1.0, 0.0, -0.1]})
    return out


def test_a_side_on_weld_goes_to_the_arm_behind_its_approach():
    weld = {"id": "W01", "position": [-0.74, 0.5, 1.3], "approach_vector": [0.97, 0.22, -0.11]}
    ranked = _robot_scores(weld, _robot_bases(BODY_ARMS))
    assert ranked[0][0] == "robot4"                     # left, nearest — not robot3 across
    assert _alignment(weld, np.array(BODY_ARMS["robot3"]["base_position"])) < -0.9


def test_a_top_down_weld_still_goes_to_the_nearest_arm():
    weld = {"id": "T", "position": [-2.0, -1.0, 1.0], "approach_vector": [0, 0, -1]}
    assert _robot_scores(weld, _robot_bases(BODY_ARMS))[0][0] == "robot1"


def test_every_weld_goes_to_a_feasible_arm_and_every_arm_works():
    welds = _side_welds()
    unreachable: list = []
    alloc = _regional_allocate(welds, BODY_ARMS, unreachable=unreachable)
    assert not unreachable
    for rid, seq in alloc.items():
        base = np.array(BODY_ARMS[rid]["base_position"])
        for op in seq:
            assert _alignment(op, base) > 0, (op["id"], rid)          # behind the approach
            assert np.linalg.norm((np.array(op["position"]) - base)[:2]) <= 2.9
    loads = sorted(len(seq) for seq in alloc.values())
    assert loads[0] >= 4, alloc                                       # no arm idle


def test_a_roof_rail_high_above_the_base_is_still_in_reach():
    """3.0 m from the base on the floor, ~2.7 m from the shoulder: reachable."""
    rail = {"id": "rail", "position": [-0.71, 0.2, 1.44], "approach_vector": [0.95, 0, -0.31]}
    unreachable: list = []
    alloc = _regional_allocate([rail], BODY_ARMS, unreachable=unreachable)
    assert not unreachable and alloc["robot4"]


def test_a_weld_no_arm_can_take_is_reported_not_forced():
    far = {"id": "FAR", "position": [0.0, 9.0, 1.0], "approach_vector": [0, 0, -1]}
    behind = {"id": "WRONG", "position": [-0.85, 0.0, 1.0], "approach_vector": [-1.0, 0, 0]}
    unreachable: list = []
    alloc = _regional_allocate([far, behind], {"robot1": BODY_ARMS["robot1"],
                                               "robot4": BODY_ARMS["robot4"]},
                               unreachable=unreachable)
    assert {u["operation_id"] for u in unreachable} == {"FAR", "WRONG"}
    assert not any(alloc.values())
    # Without the report list (an older caller), nothing is silently lost.
    legacy = _regional_allocate([far], BODY_ARMS)
    assert sum(len(s) for s in legacy.values()) == 1


def test_an_idle_arm_that_could_work_is_given_work():
    """Ten welds that all score best for robot1, but robot4 is a close second."""
    welds = [{"id": f"W{i}", "position": [-0.9, -0.3 + 0.06 * i, 1.0],
              "approach_vector": [1, 0, 0]} for i in range(10)]
    arms = {"robot1": BODY_ARMS["robot1"], "robot4": BODY_ARMS["robot4"]}
    alloc = _regional_allocate(welds, arms)
    assert len(alloc["robot4"]) >= 3 and len(alloc["robot1"]) >= 3, \
        {r: len(s) for r, s in alloc.items()}


def test_the_pipeline_ignores_pins_the_model_invents(monkeypatch):
    monkeypatch.setenv("CELL_BASE_PATH", str(PROJECT_ROOT / "cells"))
    from agents.trajectory import operator_directives
    from agents.trajectory.tools import planning_tools as pt
    fn = getattr(pt.run_mrta_pipeline, "func", pt.run_mrta_pipeline)
    ops = [{"id": f"W{i:02d}", "type": "weld", "position": [-1.5 + 0.3 * (i % 3), -2.0 + 0.4 * (i // 3), 1.0],
            "approach_vector": [0, 0, -1], "assigned_robot": "robot1"} for i in range(9)]
    with operator_directives.scope():
        operator_directives.set_locks({"W00": "robot1"})        # the run's ONE real lock
        out = fn("welding_cell_2_small", ops)
    if "error" in out:
        pytest.skip(f"cell not usable on this host: {out['error']}")
    assert set(out["ignored_pins"]) == {f"W{i:02d}" for i in range(1, 9)}
    owner = {op["id"]: rid for st in out["steps"] for rid, op in st.items()
             if isinstance(op, dict) and "id" in op}
    assert owner["W00"] == "robot1"                               # the lock still binds
    assert len({owner[k] for k in owner}) == 2                    # both arms work again


# ── a distribution the MISSION defines is used, not recalculated ──────────────

def _pipeline_with_mission_locks(monkeypatch, ops):
    """run_mrta_pipeline as a planning run calls it: the mission's assigned_robot
    bound as the run's locks first (agents/trajectory/agent.py, all_locks)."""
    monkeypatch.setenv("CELL_BASE_PATH", str(PROJECT_ROOT / "cells"))
    from agents.trajectory import operator_directives
    from agents.trajectory.tools import planning_tools as pt
    fn = getattr(pt.run_mrta_pipeline, "func", pt.run_mrta_pipeline)
    with operator_directives.scope():
        operator_directives.set_locks(operator_directives.all_locks(ops))
        # the model passes the operations WITHOUT the pins: they come back anyway
        out = fn("welding_cell_2_small", [{k: v for k, v in o.items() if k != "assigned_robot"}
                                     for o in ops])
    if "error" in out:
        pytest.skip(f"cell not usable on this host: {out['error']}")
    return {op["id"]: rid for st in out["steps"] for rid, op in st.items()
            if isinstance(op, dict) and "id" in op}, out


def _grid(n=9):
    return [{"id": f"W{i:02d}", "type": "weld",
             "position": [-1.5 + 0.3 * (i % 3), -2.0 + 0.4 * (i // 3), 1.0],
             "approach_vector": [0, 0, -1]} for i in range(n)]


def test_a_distribution_the_mission_defines_is_used_as_is(monkeypatch):
    ops = _grid()
    for i, op in enumerate(ops):                 # deliberately NOT the geometric choice
        op["assigned_robot"] = "robot2" if i < 7 else "robot1"
    owner, out = _pipeline_with_mission_locks(monkeypatch, ops)
    assert owner == {op["id"]: op["assigned_robot"] for op in ops}
    assert "ignored_pins" not in out             # mission pins are not "invented"


def test_only_the_undefined_part_of_a_mission_is_calculated(monkeypatch):
    ops = _grid()
    ops[0]["assigned_robot"] = "robot2"          # defined by the mission
    owner, _out = _pipeline_with_mission_locks(monkeypatch, ops)
    assert owner["W00"] == "robot2"
    assert len(owner) == 9 and len(set(owner.values())) == 2   # the rest calculated
