"""The tools that let the agent optimise, not only validate: evaluate_program,
find_clear_pose, load-aware reallocation — and the assembler's own savings
(merged back-to-back moves, a staggered return home)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weld_step_gate import CELL, bt, gate  # noqa: E402,F401 (fixtures)


def _leg(a, b, t):
    return [{"positions": [a] * 6, "time_s": 0.0}, {"positions": [b] * 6, "time_s": t}]


def _mods():
    import importlib
    return (importlib.import_module("agents.trajectory.tools.program_eval"),
            importlib.import_module("agents.trajectory.tools.trajectory_cache"))


@pytest.fixture
def cache(bt):
    pe, tc = _mods()
    tc.clear_trajectory_cache(CELL)
    tc.reset_robot_configs(CELL)
    yield pe, tc
    tc.clear_trajectory_cache(CELL)
    tc.reset_robot_configs(CELL)


def _plan(*steps, home=False):
    return {"steps": [{"mode": m, "robots": [{"robot_ns": ns, "operation_id": op,
                                              "hold_duration_s": 0.5} for ns, op in rs]}
                      for m, rs in steps], "home_at_end": home}


# ── evaluate_program ─────────────────────────────────────────────────────────

def test_cycle_time_is_the_slowest_arm_of_a_parallel_step_and_the_sum_of_a_serial_one(cache):
    pe, tc = cache
    tc.cache_trajectory(CELL, "robot1", "A1", _leg(0.0, 0.1, 2.0))
    tc.cache_trajectory(CELL, "robot2", "B1", _leg(0.0, -0.1, 3.0))
    tc.cache_trajectory(CELL, "robot1", "A2", _leg(0.1, 0.2, 1.0))
    tc.cache_trajectory(CELL, "robot2", "B2", _leg(-0.1, -0.2, 1.0))
    ev = getattr(pe.evaluate_program, "func", pe.evaluate_program)
    out = ev(CELL, _plan(("PARALLEL", [("robot1", "A1"), ("robot2", "B1")]),
                         ("SEQUENTIAL", [("robot1", "A2"), ("robot2", "B2")])),
             check_clearance=False, check_home=False)
    assert out["success"] is True
    assert out["cycle_time_s"] == pytest.approx(3.5 + 3.0)
    assert out["arms"]["robot1"]["busy_s"] == pytest.approx(4.0)
    assert out["arms"]["robot2"]["busy_s"] == pytest.approx(5.0)


def test_an_unbalanced_program_says_which_arm_carries_it(cache):
    pe, tc = cache
    tc.cache_trajectory(CELL, "robot1", "A1", _leg(0.0, 0.1, 1.0))
    for k in range(3):
        tc.cache_trajectory(CELL, "robot2", f"B{k}", _leg(-0.1 * k, -0.1 * (k + 1), 3.0))
    ev = getattr(pe.evaluate_program, "func", pe.evaluate_program)
    out = ev(CELL, _plan(("PARALLEL", [("robot1", "A1"), ("robot2", "B0")]),
                         ("PARALLEL", [("robot2", "B1")]), ("PARALLEL", [("robot2", "B2")])),
             check_clearance=False, check_home=False)
    assert out["imbalance_s"] == pytest.approx(9.0)
    assert any("robot2 is busy" in s for s in out["suggestions"])


def test_an_unplanned_operation_is_named(cache):
    pe, _tc = cache
    ev = getattr(pe.evaluate_program, "func", pe.evaluate_program)
    out = ev(CELL, _plan(("PARALLEL", [("robot1", "NOPE")])), check_clearance=False,
             check_home=False)
    assert out["success"] is False and out["missing_trajectories"][0]["operation_id"] == "NOPE"


def test_a_return_home_that_cannot_run_together_is_costed(cache, monkeypatch):
    pe, tc = cache
    tc.cache_trajectory(CELL, "robot1", "A1", _leg(0.0, 0.1, 1.0))
    tc.cache_trajectory(CELL, "robot2", "B1", _leg(0.0, -0.1, 1.0))
    monkeypatch.setattr(pe, "_validate", lambda c, legs: {
        "collision_free": False, "min_clearance_m": 0.1, "warning_distance_m": 0.2,
        "worst_pair": ["robot1", "robot2"]})
    import agents.trajectory.tools.behaviour_tree as btm
    monkeypatch.setattr(btm, "_arm_home", lambda c, ns: [0.0] * 6)
    import agents.trajectory.tools.planning_tools as pt
    monkeypatch.setattr(pt, "find_start_delays", lambda c, legs: None)
    ev = getattr(pe.evaluate_program, "func", pe.evaluate_program)
    out = ev(CELL, _plan(("PARALLEL", [("robot1", "A1"), ("robot2", "B1")]), home=True),
             check_clearance=False)
    assert out["home"]["together"] is False and out["home"]["extra_cost_s"] > 0
    assert any("cannot go home together" in s for s in out["suggestions"])


# ── find_clear_pose ──────────────────────────────────────────────────────────

def test_the_nearest_pose_toward_home_that_clears_is_chosen(cache, monkeypatch):
    pe, tc = cache
    tc.set_last_config(CELL, "robot1", [1.0] * 6)
    tc.cache_trajectory(CELL, "robot2", "B5", _leg(0.0, -0.1, 1.0))
    import agents.trajectory.tools.behaviour_tree as btm
    monkeypatch.setattr(btm, "_arm_home", lambda c, ns: [0.0] * 6)
    import agents.trajectory.tools.leg_verdict as lv
    monkeypatch.setattr(lv, "refusal", lambda *a, **k: None)
    import agents.trajectory.tools.planning_tools as pt
    monkeypatch.setattr(pt, "_load_collision_config", lambda c: {"warning_distance": 0.2})

    def validate(c, legs):           # clearance grows as robot1 moves toward HOME
        q = legs["robot1"][0]["positions"][0]
        return {"min_clearance_m": 0.6 - 0.5 * q}
    monkeypatch.setattr(pe, "_validate", validate)
    fcp = getattr(pe.find_clear_pose, "func", pe.find_clear_pose)
    out = fcp(CELL, "robot1", [{"robot_ns": "robot2", "operation_ids": ["B5"]}], margin_m=0.05)
    if not out.get("success") and "pinocchio" in str(out.get("reason", "")).lower():
        pytest.skip("no FK here")
    # needs 0.25 m: 0.6 - 0.5 q >= 0.25  ->  q <= 0.7  ->  first fraction 0.3
    assert out["fraction_to_home"] == pytest.approx(0.3), out
    assert out["motion_request"]["operation_type"] == "move"


# ── reallocation ─────────────────────────────────────────────────────────────

def test_a_weld_blocked_by_the_other_arm_is_not_handed_over(bt):
    import agents.trajectory.tools.planning_tools as pt
    fn = getattr(pt.reallocate_operation, "func", pt.reallocate_operation)
    out = fn(CELL, {"id": "W05", "position": [0, 0, 1]}, ["robot1"],
             reason_code="blocked_by_parked_arm")
    assert out["success"] is False and out["reason_code"] == "reallocation_not_needed"


def test_among_equal_takers_the_least_loaded_wins(bt, monkeypatch):
    import agents.trajectory.tools.planning_tools as pt
    monkeypatch.setattr(pt, "_cell_robots", lambda c: {"r1": {}, "r2": {}, "r3": {}})
    monkeypatch.setattr(pt, "_reallocation_candidates", lambda op, robots, tried: [
        {"robot_ns": "r2", "rank": 1, "score": 1.00, "within_workspace": True},
        {"robot_ns": "r3", "rank": 2, "score": 0.95, "within_workspace": True}])
    fn = getattr(pt.reallocate_operation, "func", pt.reallocate_operation)
    out = fn(CELL, {"id": "W9", "position": [0, 0, 1]}, ["r1"], robots=[
        {"robot_ns": "r1", "operation_ids": ["a"]},
        {"robot_ns": "r2", "operation_ids": ["b", "c", "d", "e"]},
        {"robot_ns": "r3", "operation_ids": ["f"]}])
    assert out["to_robot"] == "r3" and out["load_after"]["r3"] == 2


# ── the assembler's own savings ──────────────────────────────────────────────

def _spec(i, ns, op, leg, kind="move", hold=0.5):
    node = {"type": "EXECUTE", "robot_ns": ns, "operation_id": op, "operation_type": kind,
            "precomputed_trajectory": leg, "hold_duration_s": hold}
    return {"index": i, "mode": "PARALLEL", "nodes": [node], "legs": {ns: leg},
            "ops": {ns: op}, "delays": {ns: 0.0}}


def test_back_to_back_moves_of_one_arm_become_one_leg(bt):
    specs = [_spec(1, "robot1", "R1_RETREAT1", _leg(0.0, 0.1, 1.0)),
             _spec(2, "robot1", "R1_RETREAT2", _leg(0.1, 0.2, 0.5)),
             _spec(3, "robot1", "R1_RETREAT3", _leg(0.2, 0.3, 0.5)),
             _spec(4, "robot2", "W07", _leg(0.0, -0.1, 1.0), kind="weld")]
    out, merges = bt._merge_moves(specs)
    assert len(out) == 2
    leg = out[0]["legs"]["robot1"]
    assert leg[-1]["time_s"] == pytest.approx(2.0) and leg[-1]["positions"][0] == 0.3
    assert merges == [{"robot_ns": "robot1",
                       "operation_ids": ["R1_RETREAT1", "R1_RETREAT2", "R1_RETREAT3"],
                       "saved_s": 1.0}]


def test_welds_and_other_arms_are_never_merged(bt):
    specs = [_spec(1, "robot1", "M1", _leg(0.0, 0.1, 1.0)),
             _spec(2, "robot2", "M2", _leg(0.0, -0.1, 1.0)),
             _spec(3, "robot2", "W1", _leg(-0.1, -0.2, 1.0), kind="weld"),
             _spec(4, "robot2", "M3", _leg(0.5, 0.6, 1.0))]     # does not chain
    out, merges = bt._merge_moves(specs)
    assert len(out) == 4 and merges == []


def test_a_serial_return_home_is_staggered_when_that_is_faster(bt, monkeypatch):
    import agents.trajectory.tools.planning_tools as pt
    legs = {"robot1": _leg(0.3, 0.0, 5.0), "robot2": _leg(-0.3, 0.0, 6.0)}
    monkeypatch.setattr(pt, "find_start_delays", lambda c, l: {
        "start_delays": {"robot1": 1.5, "robot2": 0.0}})
    monkeypatch.setattr(bt, "_gate_multi_arm_step", lambda c, l, **k: {
        "mode": "PARALLEL", "gated": True, "parallel": {"min_clearance_m": 0.31}})
    nodes = [{"type": "HOME", "robot_ns": ns, "precomputed_trajectory": l}
             for ns, l in legs.items()]
    node, rep = bt._staggered_home(CELL, nodes, legs, {"mode": "SEQUENTIAL",
                                                       "order": ["robot2", "robot1"]})
    assert rep["mode"] == "PARALLEL" and rep["staggered"]["staggered_s"] == pytest.approx(6.5)
    assert rep["staggered"]["saved_s"] == pytest.approx(4.5)
    kinds = [c["type"] for c in node["children"]]
    assert "Delay" in kinds


def test_a_stagger_the_gate_rejects_keeps_the_serial_order(bt, monkeypatch):
    import agents.trajectory.tools.planning_tools as pt
    legs = {"robot1": _leg(0.3, 0.0, 5.0), "robot2": _leg(-0.3, 0.0, 6.0)}
    monkeypatch.setattr(pt, "find_start_delays", lambda c, l: {
        "start_delays": {"robot1": 1.5}})
    monkeypatch.setattr(bt, "_gate_multi_arm_step", lambda c, l, **k: {"mode": "SEQUENTIAL"})
    assert bt._staggered_home(CELL, [], legs, {"mode": "SEQUENTIAL"}) is None


def test_every_accepted_tree_carries_its_cost(bt, gate, monkeypatch):
    import importlib
    import agents.home_pose
    # synthetic legs, not anchored at HOME (see conftest._synthetic_legs)
    monkeypatch.setattr(agents.home_pose, "program_start_pose", lambda c, ns: None)
    gate(lambda moving, parked: True)
    tc = importlib.import_module("agents.trajectory.tools.trajectory_cache")
    tc.cache_trajectory(CELL, "robot1", "A1", _leg(0.1, 0.2, 2.0))
    tc.cache_trajectory(CELL, "robot2", "B1", _leg(-0.1, -0.2, 3.0))
    out = bt.assemble_behaviour_tree.func(CELL, _plan(
        ("PARALLEL", [("robot1", "A1"), ("robot2", "B1")])))
    assert out["success"] is True, out.get("error")
    cost = out["program_cost"]
    assert cost["cycle_time_s"] == pytest.approx(3.5)
    assert set(cost["arms"]) == {"robot1", "robot2"}
