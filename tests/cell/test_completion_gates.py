"""WaitForRobot ``depends_on_ops``: an arm waits for another arm to COMPLETE an
operation — the gate the coordinated planner orders conflicting arms with.

Each arm runs its own Sequence inside a Parallel. The gate must hold the waiting
arm until the named operation has completed, whatever the relative speeds, and
release immediately if it already has.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from test_bt_executor import FakeRuntime, _cfg, _execute_node  # noqa: E402

from agents.cell.bt_executor import BehaviourTreeExecutor  # noqa: E402


def _gate(rid, deps):
    return {"id": f"g_{rid}", "type": "WaitForRobot", "robot_ns": rid,
            "depends_on_ops": [{"robot": r, "operation_id": o} for r, o in deps]}


def _seq(*children):
    return {"id": "s", "type": "Sequence", "children": list(children)}


def _run(root, **cfg):
    rt = FakeRuntime()
    ex = BehaviourTreeExecutor(rt, None, _cfg(**cfg))
    return ex.execute({"root": root, "metadata": {}}), rt


def _first_dispatch(rt, op):
    return next(t for t, _, task in rt.dispatches if task.get("operation_id") == op)


def test_the_gated_arm_starts_only_after_the_other_completes():
    # robot1's A1 takes ~0.3 s; robot2's B1 is gated on it.
    root = {"id": "p", "type": "Parallel", "children": [
        _seq(_execute_node("robot1", "A1", end=0.3)),
        _seq(_gate("robot2", [("robot1", "A1")]), _execute_node("robot2", "B1")),
    ]}
    result, rt = _run(root)
    assert result.get("success") is True, result
    assert _first_dispatch(rt, "B1") >= _first_dispatch(rt, "A1") + 0.3


def test_an_already_completed_dependency_does_not_wait():
    root = _seq(_execute_node("robot1", "A1"),
                _gate("robot2", [("robot1", "A1")]),
                _execute_node("robot2", "B1"))
    result, rt = _run(root)
    assert result.get("success") is True, result
    assert _first_dispatch(rt, "B1") - _first_dispatch(rt, "A1") < 0.3


def test_a_dependency_that_never_completes_times_out_visibly():
    root = _seq(_gate("robot2", [("robot1", "NEVER")]), _execute_node("robot2", "B1"))
    result, rt = _run(root, op_gate_timeout_s=0.2)
    assert result.get("success") is False
    assert "NEVER" in str(result)
    assert not any(task.get("operation_id") == "B1" for _, _, task in rt.dispatches)
