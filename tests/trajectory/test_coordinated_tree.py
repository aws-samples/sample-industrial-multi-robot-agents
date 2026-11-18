"""The coordinated tree: one Sequence per arm, all arms in a Parallel, and a
WaitForRobot completion gate wherever the coordinator ordered two arms.

Geometry is stubbed (the conflict model is injected); the LangChain surface is
the stub from ``test_weld_step_gate.py``.
"""
from __future__ import annotations

import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weld_step_gate import CELL, _leg, bt  # noqa: E402,F401 (fixtures)


@pytest.fixture(autouse=True)
def _synthetic_legs_have_no_obstacles(bt, monkeypatch):
    """The coordinated assembler now checks every leg against the cell's static
    obstacles too. The legs here are SYNTHETIC (``[0.1] * 6`` ...) in a real cell
    and what is under test is the gate logic, so the obstacles are taken out —
    after ``bt`` re-imported the tools, so the patch lands on the live module.
    The static check itself is covered by ``test_leg_verdict.py``."""
    import agents.trajectory.tools.robot_kinematics as rk
    monkeypatch.setattr(rk, "leg_static_breach", lambda *a, **k: None)


def _cache_program(n=2):
    from agents.trajectory.tools.trajectory_cache import cache_trajectory
    seqs = {}
    for ns in ("robot1", "robot2"):
        seqs[ns] = []
        for k in range(1, n + 1):
            op = f"{ns}_W{k}"
            cache_trajectory(CELL, ns, op, _leg(ns, k))
            seqs[ns].append(op)
    return seqs


def _stub_conflicts(monkeypatch, pairs):
    import agents.trajectory.tools.coordinated_tree as ct
    bad = {frozenset(p) for p in pairs}

    def model(cell_id, trajectories, home, **kw):
        def conflict(x, y):
            # items are SEGMENTS (``op@k``); a clash declared on an op holds for
            # every segment of it
            ox, oy = x.op_id.split("@")[0], y.op_id.split("@")[0]
            return frozenset(((x.arm, x.kind, ox), (y.arm, y.kind, oy))) in bad
        return conflict
    monkeypatch.setattr(ct, "geometric_conflict", model)
    return ct


def _walk(node, out):
    out.append(node)
    for c in node.get("children") or []:
        _walk(c, out)
    return out


def test_arms_run_as_parallel_sequences_with_a_gate_where_they_conflict(bt, monkeypatch):
    seqs = _cache_program()
    ct = _stub_conflicts(monkeypatch, [(("robot1", "move", "robot1_W2"),
                                        ("robot2", "move", "robot2_W2"))])
    out = ct.assemble_coordinated_tree.func(CELL, seqs, home_at_end=False)
    assert out["success"] is True, out.get("error")
    root = out["behaviour_tree"]["root"]
    par = root["children"][0]
    assert par["type"] == "Parallel" and len(par["children"]) == 2
    assert all(c["type"] == "Sequence" for c in par["children"])
    gates = [n for n in _walk(root, []) if n.get("type") == "WaitForRobot"]
    assert gates, "the conflicting W2 legs must be ordered"
    bb = out["behaviour_tree"]["blackboard"]
    for g in gates:
        deps = g["ports"]["depends_on_ops"]
        deps = bb[deps.strip("{}")] if isinstance(deps, str) else deps
        waiter = g["ports"]["robot_ns"]
        # each gate waits on the OTHER arm's conflicting W2 (a segment of it)
        assert all(d["robot"] != waiter and d["operation_id"].startswith(f"{d['robot']}_W2")
                   for d in deps), deps
    assert out["parallel_s"] > 0


def test_a_deadlocked_program_is_refused_with_the_safe_spot_to_add(bt, monkeypatch):
    seqs = _cache_program()
    # robot1 finishes at W2 and stays there; robot2's W2 must pass through it.
    ct = _stub_conflicts(monkeypatch, [(("robot1", "stand", "robot1_W2"),
                                        ("robot2", "move", "robot2_W2"))])
    out = ct.assemble_coordinated_tree.func(CELL, seqs, home_at_end=False)
    assert out["success"] is False
    assert "behaviour_tree" not in out
    assert out["deadlock"]["needs_safe_spot"] == {"arm": "robot1", "after_op": "robot1_W2"}
    assert "SAFE SPOT" in out["error"]


def test_a_missing_trajectory_is_reported_before_coordinating(bt, monkeypatch):
    ct = _stub_conflicts(monkeypatch, [])
    out = ct.coordinate_program.func(CELL, {"robot1": ["nope"]})
    assert out["success"] is False and out["missing"] == [{"robot_ns": "robot1",
                                                           "operation_id": "nope"}]


def test_a_finished_arm_in_the_way_is_sent_home_not_asked_for_a_safe_spot(bt, monkeypatch):
    from agents.trajectory.tools.trajectory_cache import cache_trajectory
    for ns, n in (("robot1", 2), ("robot2", 1)):
        for k in range(1, n + 1):
            cache_trajectory(CELL, ns, f"{ns}_W{k}", _leg(ns, k))
    seqs = {"robot1": ["robot1_W1", "robot1_W2"], "robot2": ["robot2_W1"]}
    # robot2 finishes at W1 and would stand there for good; robot1's W2 passes it.
    ct = _stub_conflicts(monkeypatch, [(("robot2", "stand", "robot2_W1"),
                                        ("robot1", "move", "robot1_W2"))])
    end = _leg("robot2", 1)[-1]["positions"]
    monkeypatch.setattr(ct, "_plan_early_home", lambda cell, arm, last: [
        {"positions": list(end), "time_s": 0.0},
        {"positions": [0.0] * 6, "time_s": 1.0}])
    out = ct.assemble_coordinated_tree.func(CELL, seqs, home_at_end=False)
    assert out["success"] is True, out.get("error")
    assert out["early_home"] == ["robot2"]
    assert out["sequences"]["robot2"] == ["robot2_W1", "robot2_HOME"]
    # robot1's W2 waits for robot2 to have LEFT (its home leg completed)
    assert out["arms"]["robot1"]["gates"].get("robot1_W2") == {"robot2": "robot2_HOME"}
    kinds = [n for n in _walk(out["behaviour_tree"]["root"], [])
             if (n.get("ports") or {}).get("operation_id") == "robot2_HOME"]
    assert kinds and kinds[0]["ports"]["operation_type"] == "move"


def test_an_arm_that_still_has_work_gets_a_safe_spot_request(bt, monkeypatch):
    seqs = _cache_program()
    ct = _stub_conflicts(monkeypatch, [(("robot1", "stand", "robot1_W1"),
                                        ("robot2", "move", "robot2_W2")),
                                       (("robot1", "move", "robot1_W2"),
                                        ("robot2", "stand", "robot2_W1"))])
    called = []
    monkeypatch.setattr(ct, "_plan_early_home", lambda *a: called.append(a))
    out = ct.coordinate_program.func(CELL, seqs)
    assert out["success"] is False and "SAFE SPOT" in out["error"]
    assert called == [], "an arm with work left is never sent home early"
