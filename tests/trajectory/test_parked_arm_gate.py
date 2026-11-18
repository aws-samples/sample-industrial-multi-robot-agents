"""An arm with no work in a step is PARKED, not absent — every step is gated
against it.

welding_cell_2_small, plan bt_1790749984: robot1's single-arm step W04 ran while
robot2 stood at the end of W10_VIA. The assembler waved the step through as
"single arm — no pair to check"; the planning validator puts that motion at
0.1465 m from the parked arm against a 0.2 m band, and the runtime monitor
e-stopped the replay at 0.1678 m. Same band on both sides — the gap was that
the step was never checked.

Reuses the stubs of ``test_weld_step_gate.py``: arms are "moving" when their
trajectory has more than one sample and "parked" when it has one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weld_step_gate import CELL, _leg, bt, gate  # noqa: E402,F401 (fixtures)


def _plan(steps):
    return {"steps": [{"mode": "PARALLEL",
                       "robots": [{"robot_ns": ns, "operation_id": op,
                                   "operation_type": "weld"} for ns, op in st]}
                      for st in steps], "home_at_end": False}


def _cache(ns, op, k):
    from agents.trajectory.tools.trajectory_cache import cache_trajectory
    cache_trajectory(CELL, ns, op, _leg(ns, k))


def test_a_single_arm_step_is_checked_against_the_parked_arm(bt, gate):
    calls = gate(lambda moving, parked: True)
    _cache("robot1", "A1", 1); _cache("robot2", "B1", 1); _cache("robot1", "A2", 2)
    out = bt.assemble_behaviour_tree.func(
        CELL, _plan([[("robot1", "A1"), ("robot2", "B1")], [("robot1", "A2")]]))
    assert out["success"] is True, out.get("error")
    last = [c for c in calls if c["moving"] == ["robot1"]]
    assert last and last[-1]["parked"] == ["robot2"], calls
    gated = [g for g in out["step_gates"] if g["step"] == 2]
    assert gated and gated[0]["gated"] is True and gated[0]["bystanders"] == ["robot2"]


def test_a_single_arm_step_that_hits_the_parked_arm_refuses_the_tree(bt, gate):
    gate(lambda moving, parked: not (moving == ["robot1"] and "robot2" in parked))
    _cache("robot1", "A1", 1); _cache("robot2", "B1", 1); _cache("robot1", "A2", 2)
    # the first step is two moving arms: clear; the second is robot1 alone past
    # robot2 parked — the rule says that breaches
    out = bt.assemble_behaviour_tree.func(
        CELL, _plan([[("robot1", "A1"), ("robot2", "B1")], [("robot1", "A2")]]))
    assert out["success"] is False
    blocked = [b for b in out["blocked_steps"] if b["step"] == 2]
    assert blocked and blocked[0]["blocked_by_parked"] == ["robot2"]
