"""One leg verdict for every gate, one fail-closed policy, and a gate that tells a
leg's own fault from an arm clash."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weld_step_gate import CELL, _leg, bt, gate  # noqa: E402,F401 (fixtures)

LEG = [{"positions": [0.0] * 6, "time_s": 0.0}, {"positions": [0.1] * 6, "time_s": 1.0}]


def _lv():
    import importlib
    return importlib.import_module("agents.trajectory.tools.leg_verdict")


@pytest.fixture
def lv(monkeypatch):
    mod = _lv()
    mod.clear_cache()
    yield mod
    mod.clear_cache()


# ── the policy ───────────────────────────────────────────────────────────────

def test_a_check_that_broke_refuses_the_leg(lv, monkeypatch):
    jl = pytest.importorskip("agents.trajectory.tools.joint_limits")

    def boom(*a):
        raise RuntimeError("bad config")
    monkeypatch.setattr(jl, "leg_limit_violation", boom)
    why = lv.refusal("c", "robot1", LEG)
    assert why["reason_code"] == "leg_check_error" and "bad config" in why["error"]


def test_a_check_that_cannot_run_refuses_nothing_and_says_so(lv, monkeypatch):
    jl = pytest.importorskip("agents.trajectory.tools.joint_limits")

    def absent(*a):
        raise FileNotFoundError("no URDF")
    monkeypatch.setattr(jl, "leg_limit_violation", absent)
    assert lv.limits("c", "robot1", LEG)[0] == "unavailable"
    assert any(u["check"] == "limits" for u in lv.unchecked("c", "robot1", LEG))


def test_a_verdict_is_computed_once_per_leg(lv, monkeypatch):
    jl = pytest.importorskip("agents.trajectory.tools.joint_limits")
    calls = []
    monkeypatch.setattr(jl, "leg_limit_violation", lambda *a: calls.append(1))
    for _ in range(3):
        lv.limits("c", "robot1", LEG)
    assert len(calls) == 1
    lv.limits("c", "robot1", LEG[:1])                  # another leg: judged anew
    assert len(calls) == 2


def test_a_replaced_check_is_not_answered_from_the_old_ones_cache(lv, monkeypatch):
    jl = pytest.importorskip("agents.trajectory.tools.joint_limits")
    monkeypatch.setattr(jl, "leg_limit_violation", lambda *a: None)
    assert lv.limits("c", "robot1", LEG) == ("hit", None)
    monkeypatch.setattr(jl, "leg_limit_violation", lambda *a: {"joint": "a1"})
    assert lv.limits("c", "robot1", LEG) == ("hit", {"joint": "a1"})


def test_an_edited_cell_file_is_judged_afresh(lv, monkeypatch, tmp_path):
    from utils import cell_layout
    f = tmp_path / "robots.yaml"
    f.write_text("a: 1\n")
    monkeypatch.setattr(cell_layout, "robots_yaml", lambda c, b=None: f)
    first = lv.cell_stamp("c")
    import os
    os.utime(f, ns=(1, 1))
    assert lv.cell_stamp("c") != first


def test_a_remote_cell_is_never_cached(lv, monkeypatch):
    from utils import cell_store
    monkeypatch.setattr(cell_store, "is_remote", lambda: True)
    assert lv.cell_stamp("c") is None
    n = []
    lv._cached("x", "c", "r", LEG, lambda: n.append(1))
    lv._cached("x", "c", "r", LEG, lambda: n.append(1))
    assert len(n) == 2


# ── the gate ─────────────────────────────────────────────────────────────────

def _stub_set(monkeypatch, answer):
    import agents.trajectory.tools.planning_tools as pt
    monkeypatch.setattr(pt, "validate_trajectory_set", lambda **kw: answer(kw))


def test_a_leg_that_breaches_on_its_own_is_not_called_a_pairing_problem(bt, monkeypatch):
    calls = []

    def answer(kw):
        calls.append(sorted(kw["trajectories"]))
        return {"collision_free": False, "static_collisions": [
            {"robot_ns": "robot1", "object_id": "fixture_v1", "clearance_m": -0.02}]}
    _stub_set(monkeypatch, answer)
    rep = bt._gate_multi_arm_step(CELL, {"robot1": _leg("robot1", 1),
                                         "robot2": _leg("robot2", 1)})
    assert rep["mode"] is None and rep["blocked_arms"] == ["robot1"]
    assert rep["leg_faults"][0]["reason_code"] == "trajectory_in_collision"
    assert len(calls) == 1, "no serialisation is tried for a leg's own fault"


def test_a_broken_inter_arm_check_refuses(bt, monkeypatch):
    _stub_set(monkeypatch, lambda kw: {"error": "boom", "error_kind": "check_failed"})
    rep = bt._gate_multi_arm_step(CELL, {"robot1": _leg("robot1", 1),
                                         "robot2": _leg("robot2", 1)})
    assert rep["mode"] is None and rep["reason_code"] == "inter_arm_check_error"


def test_an_unavailable_check_still_degrades_visibly(bt, monkeypatch):
    _stub_set(monkeypatch, lambda kw: {"error": "no pinocchio", "error_kind": "unavailable"})
    rep = bt._gate_multi_arm_step(CELL, {"robot1": _leg("robot1", 1),
                                         "robot2": _leg("robot2", 1)})
    assert rep["mode"] == "PARALLEL" and rep["gated"] is False


def test_the_set_check_reports_a_broken_static_check_as_not_clear(monkeypatch):
    import agents.trajectory.tools.planning_tools as pt
    monkeypatch.setattr(pt, "inter_robot_clearance", lambda *a, **k: {
        "collision_free": True, "min_clearance_m": 1.0, "warning_distance_m": 0.2})
    monkeypatch.setattr(pt, "_load_collision_config", lambda c: {"warning_distance": 0.2})
    monkeypatch.setattr(pt, "_joint_limit_violations",
                        lambda c, t: {"checked": True, "violations": [], "failed": []})
    _lv().clear_cache()

    def boom(c, ns, t):
        raise RuntimeError("corrupt mesh")
    monkeypatch.setattr(pt, "leg_static_breach", boom)
    validate = getattr(pt.validate_trajectory_set, "func", pt.validate_trajectory_set)
    r = validate("c", {"robot1": LEG})
    assert r["collision_free"] is False and r["leg_check_failures"][0]["robot_ns"] == "robot1"


# ── program order: an arm that has not moved stands at HOME ─────────────────

def test_a_first_step_with_one_arm_is_gated_against_the_other_at_home(bt, gate,
                                                                       monkeypatch):
    import agents.home_pose
    homes = {"robot1": [0.0] * 6, "robot2": [0.3] * 6}
    monkeypatch.setattr(agents.home_pose, "program_start_pose",
                        lambda cell_id, ns: homes.get(ns))
    seen = []

    def rule(moving, parked):
        seen.append(parked)
        return True
    calls = gate(rule)
    from agents.trajectory.tools.trajectory_cache import cache_trajectory
    cache_trajectory(CELL, "robot1", "A1", LEG)               # starts at its HOME
    out = bt.assemble_behaviour_tree.func(CELL, {"steps": [{"mode": "PARALLEL", "robots": [
        {"robot_ns": "robot1", "operation_id": "A1", "operation_type": "weld"}]}],
        "home_at_end": False})
    assert out["success"] is True, out.get("error")
    assert calls and calls[0]["parked"] == ["robot2"], calls
    assert seen[0]["robot2"] == homes["robot2"]


# ── a refused step is taken back ─────────────────────────────────────────────

def test_a_step_its_gate_refuses_leaves_the_arms_where_they_stood(monkeypatch):
    import agents.trajectory.tools.planning_tools as pt
    from agents.trajectory.tools import planner_backend as pb, trajectory_cache as tc
    tc.reset_robot_configs("c")
    tc.clear_trajectory_cache("c")
    tc.set_last_config("c", "robot1", [0.5] * 6)

    def fake_dispatch(cell, reqs, **k):
        tc.set_last_config(cell, "robot1", [0.9] * 6)        # the backend commits
        return {"success": True, "trajectories": {"robot1": LEG}, "failures": [],
                "planning_time_s": 0.0}
    monkeypatch.setattr(pb, "dispatch_plan", fake_dispatch)
    monkeypatch.setattr(pt, "_step_gate", lambda c, legs: {"verdict": "REFUSED",
                                                          "advice": "no order"})
    monkeypatch.setattr(pt, "resolve_velocity_scale", lambda c, v: v)
    run = getattr(pt.run_trajectory_planner, "func", pt.run_trajectory_planner)
    out = run(cell_id="c", motion_requests=[{"robot_ns": "robot1", "operation_id": "W1",
                                            "target": {"x": 0, "y": 0, "z": 1}}])
    assert out["step_gate"]["taken_back"] is True
    assert tc.get_last_config("c", "robot1") == [0.5] * 6
    assert tc.get_cached_trajectory("c", "robot1", "W1") is None
