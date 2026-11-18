"""The umarell upstream fixes, ported generically (umarell-upstream-fixes, set v3).

Each fix lives where the design puts it — the converter for anything a cell
declares, a backend-neutral tool for anything every backend shares, the facade /
the assembler for anything the platform checks after a backend — so these tests
pin the behaviour there rather than a copy of the upstream patch:

  05/17  a blocked on-axis lift-off refuses the leg, in ``ompl`` and (the whole
         step) in ``ompl-compound``                      robot_kinematics
  07     a leg with no measurable entry does not pass ``validate_approach``
  09     the executor's per-leaf margin comes from the cell
  11     a frame's parent joint under Pinocchio 2.x and >= 3   utils.arm_model
  12     the weld pose's spin is searched when the specified one is not free
  13     joint limits: one owner, checked for every backend      joint_limits
  14     self-collision BETWEEN samples                          self_collision
  15     per-weld lift-off zone; weld poses survive a plan edit
  16     a re-planned last operation starts where it did; the first leg at HOME
  19     a blocked lift-off on the way home refuses, never drags
  08     programs that name another cell or carry placeholders are flagged

Geometry tests run on ``welding_cell_2_small`` and skip without the geometry stack.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

np = pytest.importorskip("numpy")

CELL = "welding_cell_2_small"


def _mod(name: str):
    return importlib.import_module(f"agents.trajectory.tools.{name}")


@pytest.fixture
def cells(monkeypatch):
    monkeypatch.setenv("CELL_BASE_PATH", str(REPO / "cells"))
    yield


@pytest.fixture
def geometry(cells):
    pytest.importorskip("pinocchio")
    rk = _mod("robot_kinematics")
    try:
        model, _ = rk._load_pinocchio_model(CELL, "robot1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{CELL} model unavailable: {exc}")
    return model


@pytest.fixture
def clean_cache():
    tc = _mod("trajectory_cache")
    tc.clear_trajectory_cache() if hasattr(tc, "clear_trajectory_cache") else None
    tc.reset_robot_configs()
    yield tc
    tc.reset_robot_configs()


# ── 13: joint limits ────────────────────────────────────────────────────────

class _LimModel:
    nq = 3
    lowerPositionLimit = np.array([-1.0, -2.0, -1.7976931348623157e308])
    upperPositionLimit = np.array([1.0, 2.0, 1.7976931348623157e308])


def test_the_tighter_of_yaml_and_model_wins_and_unbounded_stays_unbounded():
    jl = _mod("joint_limits")
    lo, hi = jl.position_limits({"joint_limits_lower": [-2.0, -1.5],
                                 "joint_limits_upper": [0.5, 3.0]}, _LimModel())
    assert lo.tolist()[:2] == [-1.0, -1.5] and hi.tolist()[:2] == [0.5, 2.0]
    assert math.isinf(lo[2]) and math.isinf(hi[2])
    plo, phi = jl.planning_box({}, _LimModel())
    assert plo[2] == -math.pi and phi[2] == math.pi


def test_the_box_is_sized_by_the_arm_not_by_six():
    jl = _mod("joint_limits")
    lo, _hi = jl.position_limits({"joint_limits_lower": [-1] * 7,
                                  "joint_limits_upper": [1] * 7})
    assert len(lo) == 7


def test_a_violation_names_the_worst_sample_and_joint():
    jl = _mod("joint_limits")
    traj = [{"positions": [0.0, 0.0], "time_s": 0.0},
            {"positions": [0.0, 1.3], "time_s": 1.0, "phase": "transit"},
            {"positions": [0.0, 1.0 + 1e-6], "time_s": 2.0}]
    v = jl.trajectory_limit_violation(traj, [-1.0, -1.0], [1.0, 1.0], names=["a", "b"])
    assert v["sample_index"] == 1 and v["joint"] == "b" and v["samples_outside"] == 1
    assert v["excess_rad"] == pytest.approx(0.3)
    assert jl.trajectory_limit_violation(traj[:1], [-1, -1], [1, 1]) is None


def test_the_facade_drops_an_out_of_limit_leg_and_takes_its_chaining_back(monkeypatch,
                                                                          clean_cache):
    pb, tc, jl = _mod("planner_backend"), clean_cache, _mod("joint_limits")
    tc.set_last_config("c", "robot1", [0.0] * 6)
    before = {"robot1": tc.chain_state("c", "robot1", "W1"),
              "robot2": tc.chain_state("c", "robot2", "W2")}
    tc.set_last_config("c", "robot1", [9.0] * 6)          # what the backend committed
    tc.set_last_config("c", "robot2", [1.0] * 6)
    bad = {"joint": "a2", "sample_index": 1, "value_rad": 9.0, "lower_rad": -1.0,
           "upper_rad": 1.0, "excess_rad": 8.0, "excess_deg": 458.0, "samples_outside": 1}
    monkeypatch.setattr(jl, "leg_limit_violation",
                        lambda cell, ns, traj: bad if ns == "robot1" else None)
    result = {"success": True, "failures": [],
              "trajectories": {"robot1": [{"positions": [9.0] * 6, "time_s": 0.0}],
                               "robot2": [{"positions": [1.0] * 6, "time_s": 0.0}]}}
    reqs = [{"robot_ns": "robot1", "operation_id": "W1"},
            {"robot_ns": "robot2", "operation_id": "W2"}]
    pb._limit_refusals("c", result, reqs, before)
    assert sorted(result["trajectories"]) == ["robot2"]
    assert result["failures"][0]["reason_code"] == "joint_limit_violation"
    assert result["failures"][0]["operation_id"] == "W1"
    assert tc.get_last_config("c", "robot1") == [0.0] * 6      # taken back
    assert tc.get_last_config("c", "robot2") == [1.0] * 6      # untouched


def test_a_compound_step_is_refused_together(monkeypatch, clean_cache):
    pb, jl = _mod("planner_backend"), _mod("joint_limits")
    bad = {"joint": "a1", "sample_index": 0, "value_rad": 5.0, "lower_rad": -1.0,
           "upper_rad": 1.0, "excess_rad": 4.0, "excess_deg": 229.0, "samples_outside": 1}
    monkeypatch.setattr(jl, "leg_limit_violation",
                        lambda cell, ns, traj: bad if ns == "robot1" else None)
    tag = {"sync_group": "cmp_1", "sync_members": ["robot1", "robot2"]}
    result = {"success": True, "failures": [], "trajectories": {
        "robot1": [{"positions": [5.0] * 6, "time_s": 0.0, **tag}],
        "robot2": [{"positions": [0.0] * 6, "time_s": 0.0, **tag}]}}
    pb._limit_refusals("c", result, [{"robot_ns": "robot1"}, {"robot_ns": "robot2"}], {})
    assert result["trajectories"] == {} and result["success"] is False
    assert {f["robot_ns"] for f in result["failures"]} == {"robot1", "robot2"}


def test_validate_trajectory_set_refuses_a_leg_past_its_limits(cells):
    pytest.importorskip("pinocchio")
    pt = _mod("planning_tools")
    home = _mod("planner_backend").home_config(CELL, "robot1", None)
    past = list(home)
    past[1] = 1.5                                   # A2's upper limit is 0.61 rad
    legs = {"robot1": [{"positions": home, "time_s": 0.0},
                       {"positions": past, "time_s": 1.0}]}
    out = pt.validate_trajectory_set.func(cell_id=CELL, trajectories=legs,
                                          check_static_obstacles=False)
    assert out["joint_limits_ok"] is False and out["collision_free"] is False
    assert out["joint_limit_violations"][0]["joint_index"] == 1


def test_the_timing_never_leaves_the_joint_box(geometry):
    pytest.importorskip("toppra")
    tp = _mod("time_parameterization")
    lo, hi = np.array([-1.0] * 6), np.array([1.0] * 6)
    # A corner-heavy path whose knots sit ON the limit: the spline candidates
    # overshoot past it, the exact one cannot.
    wps = [[0.0] * 6, [1.0, 0.9, 0, 0, 0, 0], [0.0, 1.0, 0, 0, 0, 0],
           [1.0, 1.0, 0.5, 0, 0, 0], [0.2, 0.3, 1.0, 0, 0, 0]]
    traj = tp.parameterize(wps, vel_limits=[2.0] * 6, accel_limits=[4.0] * 6,
                           max_deviation_rad=0.3, position_limits=(lo, hi))
    v = _mod("joint_limits").trajectory_limit_violation(traj, lo, hi)
    assert v is None, v


# ── 14: self-collision between the samples ──────────────────────────────────

def test_a_self_contact_between_two_clear_samples_is_found(geometry):
    sc = _mod("self_collision")
    c = sc.checker(CELL, "robot1")
    if not c.available:
        pytest.skip(c.reason)
    jl = _mod("joint_limits")
    rcfg = _mod("robot_kinematics")._load_robot_config(CELL, "robot1")
    lo, hi = jl.planning_box(rcfg, geometry)
    rng = np.random.default_rng(7)
    leg = None
    for _ in range(4000):
        q_bad = rng.uniform(lo, hi)
        if c.is_free(q_bad):
            continue
        d = rng.normal(size=len(q_bad))
        d /= np.linalg.norm(d)
        for step in (0.2, 0.35, 0.5):
            qa = np.clip(q_bad - step * d, lo, hi)
            qb = np.clip(q_bad + step * d, lo, hi)
            if c.is_free(qa) and c.is_free(qb):
                leg = [{"positions": qa.tolist(), "time_s": 0.0},
                       {"positions": qb.tolist(), "time_s": 1.0}]
                break
        if leg:
            break
    if leg is None:
        pytest.skip("no clear-clear-colliding segment found for this arm")
    assert c.trajectory_violation(leg, dense=False) is None
    hit = c.trajectory_violation(leg)
    assert hit is not None and 0.0 < hit["segment_fraction"] < 1.0
    why = _mod("planner_backend")._leg_refusal(CELL, "robot1", leg)
    assert why and why["reason_code"] == "trajectory_self_collision"


def test_lever_arms_bound_every_joint(geometry):
    c = _mod("self_collision").checker(CELL, "robot1")
    if not c.available:
        pytest.skip(c.reason)
    levers = c.lever_arms()
    assert levers is not None and len(levers) == geometry.nq and (levers > 0).all()


# ── 05/17: a blocked lift-off refuses ───────────────────────────────────────

def test_a_blocked_liftoff_is_a_refusal_unless_explicitly_allowed(monkeypatch):
    rk = _mod("robot_kinematics")
    seam = {"operation_id": "W01", "target": [1.0, 0.0, 1.0]}
    why = rk.liftoff_blocked_failure("robot1", seam)
    assert why["reason_code"] == "liftoff_blocked" and why["from_operation_id"] == "W01"
    monkeypatch.setenv("OMPL_ALLOW_BLOCKED_LIFTOFF", "1")
    assert rk.liftoff_blocked_failure("robot1", seam) is None


def test_both_ompl_backends_use_the_shared_refusal():
    for path in ("agents/trajectory/backends/ompl/planner.py",
                 "agents/trajectory/backends/ompl_compound/planner.py"):
        src = (REPO / path).read_text()
        assert "liftoff_blocked_failure(" in src, path
        assert "liftoff_warnings" in src, path


@pytest.mark.slow
def test_a_compound_step_with_a_blocked_liftoff_ships_no_arm(geometry, monkeypatch,
                                                            clean_cache):
    pytest.importorskip("ompl")
    cmp = importlib.import_module("agents.trajectory.backends.ompl_compound.planner")
    op = importlib.import_module("agents.trajectory.backends.ompl.planner")
    rk = _mod("robot_kinematics")
    seam = {"operation_id": "W00", "target": [0.0, 0.0, 0.0], "approach_vector": [0, 0, -1],
            "orientation": {"x": 0, "y": 0.707, "z": 0, "w": 0.707}, "rise_m": 0.0}
    monkeypatch.setattr(rk, "seam_at_start",
                        lambda cell, ns, *a, **k: seam if ns == "robot1" else None)
    monkeypatch.setattr(op, "_cartesian_retreat", lambda *a, **k: None)
    # The compound backend takes the lift-off from the shared leg_builder.
    monkeypatch.setattr(_mod("leg_builder"), "cartesian_retreat", lambda *a, **k: None)
    task = REPO / "cells" / CELL / "agentic_planner" / "tasks" / "car_door_spot_welding.json"
    import json
    ops = json.loads(task.read_text())["operations"]
    reqs = []
    for ns, o in (("robot1", ops[0]), ("robot2", ops[-1])):
        x, y, z = o["position"]
        reqs.append({"robot_ns": ns, "operation_id": o["id"],
                     "target": {"x": x, "y": y, "z": z}, "approach": True,
                     "approach_vector": o.get("approach_vector", [0, 0, -1])})
    res = cmp.plan_trajectories_ompl_compound(CELL, reqs) \
        if hasattr(cmp, "plan_trajectories_ompl_compound") else cmp.plan(CELL, reqs)
    assert res["trajectories"] == {}
    by_arm = {f["robot_ns"]: f for f in res["failures"]}
    assert set(by_arm) == {"robot1", "robot2"}
    assert by_arm["robot1"]["reason_code"] == "liftoff_blocked"
    # the healthy partner is told why, without being blamed
    assert by_arm["robot2"]["reason_code"] == "step_refused"
    assert by_arm["robot2"]["caused_by"] == {"robot_ns": "robot1",
                                             "reason_code": "liftoff_blocked"}


# ── 15: per-weld lift-off zone, any backend records its welds ───────────────

def test_any_backends_welds_are_recorded_at_the_pose_they_end_in(clean_cache, monkeypatch):
    """The facade records the weld of a backend that does not, from the request
    that was PLANNED and the orientation the leg really ends in; a pose the
    backend recorded in this call wins."""
    pb, tc = _mod("planner_backend"), clean_cache
    wo = _mod("weld_orientation")
    ended = {"x": 0.2, "y": 0.68, "z": 0.0, "w": 0.7}
    monkeypatch.setattr(wo, "tcp_orientation", lambda *a: dict(ended))
    ori = {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}
    req = {"robot_ns": "robot1", "operation_id": "W1", "approach": True,
           "constrain_orientation": True, "orientation": ori,
           "target": {"x": 1.0, "y": 0.0, "z": 1.0}, "approach_vector": [0, 0, -1],
           "retreat_offset_m": 0.14}
    res = {"trajectories": {"robot1": [{"positions": [0.0] * 6, "time_s": 0.0}]}}
    before = {"robot1": tc.chain_state("c", "robot1", "W1")}
    pb._record_weld_poses("c", [req], res, before, default_offset=0.06)
    [(op, pose)] = tc.weld_poses("c", "robot1")
    assert op == "W1" and pose["orientation"] == ended
    assert pose["stroke_m"] == pytest.approx(0.14)
    # the backend recorded its own pose in this call: left alone
    spun = {"x": 0.1, "y": 0.7, "z": 0.0, "w": 0.7}
    tc.reset_robot_configs("c")
    before = {"robot1": tc.chain_state("c", "robot1", "W1")}
    tc.set_weld_pose("c", "robot1", "W1", [1, 0, 1], [0, 0, -1], spun, stroke_m=0.1)
    pb._record_weld_poses("c", [req], res, before, default_offset=0.06)
    assert tc.weld_poses("c", "robot1")[0][1]["orientation"] == spun


def test_the_call_default_stroke_is_the_one_judged():
    pb = _mod("planner_backend")
    assert pb._with_stroke({"robot_ns": "r"}, 0.05)["approach_offset_m"] == 0.05
    assert pb._with_stroke({"approach_offset_m": 0.12}, 0.05)["approach_offset_m"] == 0.12


# ── 16: a re-planned last operation starts where it did ─────────────────────

def test_a_replan_of_the_last_operation_is_pinned_to_its_old_start(clean_cache):
    pb, tc = _mod("planner_backend"), clean_cache
    start, end = [0.0] * 6, [0.5] * 6
    tc.cache_trajectory("c", "robot1", "W1", [{"positions": start, "time_s": 0.0},
                                              {"positions": end, "time_s": 1.0}])
    tc.set_last_config("c", "robot1", [0.5 + 0.01] * 6)        # where W1 ended
    out = pb._replan_start("c", {"robot_ns": "robot1", "operation_id": "W1"})
    assert out["current_joints"] == start
    other = pb._replan_start("c", {"robot_ns": "robot1", "operation_id": "W2"})
    assert "current_joints" not in other
    pinned = {"robot_ns": "robot1", "operation_id": "W1", "current_joints": [1.0] * 6}
    assert pb._replan_start("c", pinned) is pinned


def test_a_first_leg_that_does_not_start_at_home_is_a_chain_break(cells, clean_cache):
    bt, tc = _mod("behaviour_tree"), clean_cache
    home = _mod("planner_backend").home_config(CELL, "robot1", None)
    away = [v + 0.4 for v in home]
    tc.cache_trajectory(CELL, "robot1", "W1", [{"positions": away, "time_s": 0.0},
                                               {"positions": home, "time_s": 1.0}])
    _n, _gap, breaks = bt._chain_breaks(CELL, "robot1", ["W1"])
    assert [b["reason"] for b in breaks] == ["start_gap"]
    tc.cache_trajectory(CELL, "robot1", "W1", [{"positions": home, "time_s": 0.0},
                                               {"positions": away, "time_s": 1.0}])
    assert bt._chain_breaks(CELL, "robot1", ["W1"])[2] == []


# ── 19: the return home refuses a blocked lift-off ──────────────────────────

def test_a_refused_return_is_recorded_for_its_start_only(clean_cache):
    pb, tc = _mod("planner_backend"), clean_cache
    why = {"robot_ns": "robot2", "reason_code": "home_liftoff_blocked"}
    tc.set_home_refusal("c", "robot2", why, start=[0.1] * 6)
    assert pb.home_refusal("c", "robot2", [0.1] * 6)["reason_code"] == "home_liftoff_blocked"
    assert pb.home_refusal("c", "robot2", [0.3] * 6) is None        # another pose
    assert pb.home_refusal("c", "robot1", [0.1] * 6) is None
    tc.reset_robot_configs("c")
    assert pb.home_refusal("c", "robot2", [0.1] * 6) is None


def test_a_planned_return_the_platform_refuses_is_recorded_not_blind(clean_cache, monkeypatch):
    """H1: dropping a refused leg must not hand the arm to the blind home."""
    pb, tc = _mod("planner_backend"), clean_cache
    leg = [{"positions": [0.1] * 6, "time_s": 0.0}, {"positions": [0.2] * 6, "time_s": 1.0}]
    monkeypatch.setattr(pb, "_home_provider", lambda attr: (
        (lambda cell, starts, **k: {ns: list(leg) for ns in starts}) if attr == "home"
        else (lambda cell, ns, s, h, **k: list(leg))))
    monkeypatch.setattr(pb, "_leg_refusal", lambda cell, ns, l: {
        "robot_ns": ns, "reason_code": "trajectory_self_collision", "error": "folds"})
    assert pb.plan_home_trajectories("c", {"robot1": [0.1] * 6}) == {}
    assert pb.home_refusal("c", "robot1", [0.1] * 6)["reason_code"] == "home_return_refused"
    assert pb.plan_home_trajectory("c", "robot2", [0.1] * 6, [0.0] * 6) is None
    assert pb.home_refusal("c", "robot2", [0.1] * 6)["check"] == "trajectory_self_collision"


def test_the_home_step_refuses_an_arm_stuck_on_its_seam(cells, monkeypatch, clean_cache):
    bt, tc, pb = _mod("behaviour_tree"), clean_cache, _mod("planner_backend")
    home = pb.home_config(CELL, "robot1", None)
    for ns in ("robot1", "robot2"):
        tc.cache_trajectory(CELL, ns, f"{ns}_W", [{"positions": home, "time_s": 0.0},
                                                  {"positions": home, "time_s": 1.0}])
    monkeypatch.setattr(pb, "plan_home_trajectories", lambda cell, starts, **k: {})
    monkeypatch.setattr(bt, "_planned_home_node",
                        lambda cell, ns, ops, trajectory=None: {"id": ns, "type": "HOME",
                                                                "robot_ns": ns})
    pt = _mod("planning_tools")
    monkeypatch.setattr(pt.validate_trajectory_set, "func",
                        lambda **k: {"collision_free": True}, raising=False)
    tc.set_home_refusal(CELL, "robot2", {"robot_ns": "robot2",
                                         "reason_code": "home_return_refused"}, start=home)
    node, report = bt._home_step(CELL, ["robot1", "robot2"],
                                 {"robot1": ["robot1_W"], "robot2": ["robot2_W"]})
    assert node is None and sorted(report["home_refused"]) == ["robot2"]


# ── 07: no measurable entry is not a pass ───────────────────────────────────

def test_a_leg_standing_on_the_weld_does_not_match(geometry):
    pin = _mod("pinocchio")
    wo = _mod("weld_orientation")
    q = _mod("planner_backend").home_config(CELL, "robot1", None)
    ori = wo.tcp_orientation(CELL, "robot1", q)
    still = [{"positions": q, "time_s": 0.0}, {"positions": q, "time_s": 1.0}]
    v = pin.validate_approach.func(cell_id=CELL, robot_ns="robot1",
                                   expected_orientation=ori, trajectory=still,
                                   approach_vector=[0, 0, -1])
    assert v["matches"] is False and v["entry_travel_m"] == 0.0
    assert "no measurable entry" in v["reason"]
    # The pose itself is fine: without an entry to judge, it matches.
    pose = pin.validate_approach.func(cell_id=CELL, robot_ns="robot1",
                                      expected_orientation=ori, joint_positions=q)
    assert pose["matches"] is True and pose["joint_limits_ok"] is True


# ── 12: the spin about the tool axis is free ────────────────────────────────

def _axis(q: dict) -> np.ndarray:
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])


def test_a_spin_keeps_the_tool_axis_and_a_tilt_stays_inside_the_gate():
    wo = _mod("weld_orientation")
    ori = {"x": 0.0, "y": 0.9238795, "z": 0.0, "w": -0.3826834}
    pol = wo.policy({"tool_orientation": {"spin_free": True, "max_tilt_deg": 4.5,
                                          "axis_tolerance_deg": 5.0}})
    cands = wo.candidates(ori, pol)
    a0 = _axis(ori)
    spins = [c for c in cands if c[1] == 0.0]
    assert len(spins) == 23 and abs(spins[0][0]) == 15.0
    for _s, _t, _p, o in spins:
        assert np.allclose(_axis(o), a0, atol=1e-6)
    for _s, tilt, _p, o in (c for c in cands if c[1] > 0.0):
        ang = math.degrees(math.acos(min(1.0, float(_axis(o) @ a0))))
        assert ang == pytest.approx(tilt, abs=0.01) and ang < 5.0


def test_the_tool_decides_whether_the_spin_is_free():
    wo = _mod("weld_orientation")
    assert wo.policy({})["spin_free"] is True                     # converter default
    assert wo.policy({})["max_tilt_deg"] == 4.5
    fixed = wo.policy({"tool_orientation": {"spin_free": False, "max_tilt_deg": 0}})
    assert wo.candidates({"x": 0, "y": 0, "z": 0, "w": 1}, fixed) == []
    # a tilt can never reach the gate it must pass
    assert wo.policy({"tool_orientation": {"max_tilt_deg": 9,
                                           "axis_tolerance_deg": 5}})["max_tilt_deg"] < 5
    req = {"approach": True, "constrain_orientation": True,
           "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}
    assert wo._wanted(req, wo.policy({}))
    assert not wo._wanted({**req, "applied_tool_spin_deg": 30.0}, wo.policy({}))
    assert not wo._wanted({**req, "free_spin": False}, wo.policy({}))


# ── 09 + the planning settings: the cell declares them, the code defaults ───

def test_planning_settings_default_to_what_the_converter_writes(monkeypatch):
    rk = _mod("robot_kinematics")
    monkeypatch.setattr(rk, "_load_robots_config", lambda c: {"planning": {
        "completion_margin_s": 0.0, "home_liftoff_m": -1, "min_home_liftoff_m": "x"}})
    assert rk.planning_setting("c", "completion_margin_s") == 0.0     # 0 is legal
    assert rk.planning_setting("c", "home_liftoff_m") == 0.1
    assert rk.planning_setting("c", "min_home_liftoff_m") == 0.05
    monkeypatch.setattr(rk, "_load_robots_config", lambda c: {})
    assert rk.planning_setting("c", "completion_margin_s") == 1.0


def test_the_executor_takes_its_margin_from_the_cell(monkeypatch):
    rk = _mod("robot_kinematics")
    monkeypatch.setattr(rk, "_load_robots_config",
                        lambda c: {"planning": {"completion_margin_s": 0.25}})
    bx = importlib.import_module("agents.cell.bt_executor")
    assert bx.ExecConfig.for_cell("c").completion_margin_s == 0.25


# ── 11: Pinocchio 2.x frames ────────────────────────────────────────────────

def test_a_frames_parent_joint_under_any_pinocchio():
    from utils import arm_model
    assert arm_model.frame_parent_joint(type("F3", (), {"parentJoint": 4})()) == 4
    assert arm_model.frame_parent_joint(type("F2", (), {"parent": 3})()) == 3


# ── 08: programs that cannot be trusted as-is are named ─────────────────────

def test_a_program_from_another_cell_or_with_placeholders_is_flagged():
    tp = importlib.import_module("agents.trajectory.task_programs")
    task = {"cell_id": "another_cell", "operations": [
        {"id": "W01"}, {"id": "W10", "placeholder": True}]}
    warnings = tp.program_warnings(CELL, task)
    assert any("another_cell" in w for w in warnings)
    assert any("W10" in w for w in warnings)
    assert tp.program_warnings(CELL, {"cell_id": CELL, "operations": [{"id": "W1"}]}) == []


# ── the converter writes every new setting, generically ─────────────────────

def _converter():
    spec = importlib.util.spec_from_file_location(
        "convert_isaac_scene", REPO / "scripts" / "convert_isaac_scene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_converter_writes_the_new_settings_for_any_arm():
    yaml = pytest.importorskip("yaml")
    conv = _converter()
    robot = {"namespace": "arms", "robot_id": "robot1", "model_name": "m",
             "robot_model": "m", "isaac_prim": "R1", "base_position": (0.0, 0.0, 0.0),
             "yaw": 0.0, "pose_source": "test", "tip_link": "link_7",
             "lowers": [-1.0] * 7, "uppers": [1.0] * 7, "vels": [2.0] * 7,
             "joint_names": [f"j{i}" for i in range(1, 8)],
             "guessed_limits": ["j7"], "guessed_velocity": []}
    defaults = {"gripper": "g", "workspace_radius": 2.0, "planning_group": "p",
                "end_effector_link": "tcp", "tcp_offset": (0.0, 0.0, 0.1),
                "accel": [3.0, 3.0, 3.0, 6.0, 6.0, 8.0], "rrt_j2": [0, 1],
                "rrt_j3": [0, 1], "mount_link": "link_7", "tool_axis_dir": [-1.0, 0.0, 0.0],
                "tool_axis_start": 0.15, "tool_axis_end": 0.9, "tool_axis_step": 0.15,
                "home_liftoff_m": 0.08, "min_home_liftoff_m": 0.04,
                "completion_margin_s": 0.5,
                "tool_orientation": {"spin_free": False, "max_tilt_deg": 2.0,
                                     "axis_tolerance_deg": 4.0}}
    text = conv.render_robots_yaml("c", "scene", [robot], defaults)
    cfg = yaml.safe_load(text)
    r = cfg["robots"]["arms"]["robot1"]
    assert r["joint_names"][-1] == "j7"
    assert len(r["joint_acceleration_limits"]) == 7
    assert r["tool_orientation"] == {"spin_free": False, "max_tilt_deg": 2.0,
                                     "axis_tolerance_deg": 4.0}
    assert "REVIEW: no URDF <limit> for j7" in text
    p = cfg["planning"]
    assert (p["home_liftoff_m"], p["min_home_liftoff_m"], p["completion_margin_s"]) \
        == (0.08, 0.04, 0.5)


def test_a_placeholder_box_is_machine_readable():
    yaml = pytest.importorskip("yaml")
    conv = _converter()
    text = conv.render_obstacles_yaml("c", [{"id": "FX", "kind": "placeholder",
                                             "box": {"c": (0, 0, 1), "h": (1, 1, 1)}}],
                                      "voxel", 0.05)
    boxes = [o for o in yaml.safe_load(text)["obstacles"] if o["id"] == "FX"]
    assert boxes and boxes[0]["placeholder"] is True
    assert "THICKEST" not in text


def test_a_cell_converted_before_the_new_keys_reads_the_converter_defaults(cells, monkeypatch):
    """A robots.yaml without the new keys (a cell converted before them): the
    runtime reads exactly what the converter would have written."""
    pytest.importorskip("yaml")
    rk, wo = _mod("robot_kinematics"), _mod("weld_orientation")
    real = rk._load_robots_config(CELL)
    stripped = {**real, "planning": {k: v for k, v in (real.get("planning") or {}).items()
                                     if k not in rk.PLANNING_SETTING_DEFAULTS}}
    monkeypatch.setattr(rk, "_load_robots_config", lambda c: stripped)
    defaults = _converter().build_parser().parse_args(
        ["--scene-yaml", "x", "--out-dir", "y"])
    assert rk.planning_setting(CELL, "home_liftoff_m") == defaults.home_liftoff
    assert rk.planning_setting(CELL, "min_home_liftoff_m") == defaults.min_home_liftoff
    assert rk.planning_setting(CELL, "completion_margin_s") == defaults.completion_margin
    assert wo.policy({}) == {"spin_free": defaults.tool_spin == "free",
                             "max_tilt_deg": defaults.tool_max_tilt,
                             "axis_tolerance_deg": defaults.tool_axis_tolerance}
    assert _converter().TILT_RESERVE_DEG == wo.AXIS_GATE_RESERVE_DEG


def test_the_shared_player_finds_any_cells_plans():
    sys.path.insert(0, str(REPO / "scripts"))
    player = importlib.import_module("trajectory_player")
    cell_dir = player.resolve_cell_dir(str(REPO / "cells" / CELL))
    assert player.plans_dir(cell_dir).endswith(os.path.join("agentic_planner", "plans"))
    homes = player.apply_cell_timing(cell_dir)
    if homes:
        assert all(len(v) == 6 for v in homes.values())


def test_a_hand_moved_point_is_a_placeholder_by_rule_in_any_cell():
    tp = importlib.import_module("agents.trajectory.task_programs")
    moved = {"id": "W10", "original_position": [0, 0, 0], "position_note": "moved"}
    assert tp.is_placeholder(moved) and not tp.is_placeholder({"id": "W01"})
    assert not tp.is_placeholder({**moved, "placeholder": False})     # explicit wins
    fixed, changes = tp.normalized_program(
        CELL, {"cell_id": "other", "operations": [{"id": "W01"}, moved]})
    assert fixed["cell_id"] == CELL and fixed["operations"][1]["placeholder"] is True
    assert "placeholder" not in fixed["operations"][0] and len(changes) == 2
    assert tp.normalized_program(CELL, fixed)[1] == []                   # idempotent


# ── the emitted return-home leg is validated like every other leg ───────────

@pytest.fixture
def home_planner(cells, monkeypatch):
    op = pytest.importorskip("agents.trajectory.tools.ompl_planner")
    monkeypatch.setattr(op, "_import_ompl", lambda: object())
    monkeypatch.setattr(op, "_home_liftoff", lambda *a, **k: [])
    solves: list = []

    def transit(cell_id, ns, start_q, goal_q, **kw):
        solves.append(ns)
        return [{"positions": [float(v) for v in start_q], "time_s": 0.0},
                {"positions": [float(v) for v in goal_q], "time_s": 2.0}]

    monkeypatch.setattr(op, "plan_joint_path", transit)
    monkeypatch.setattr(op, "_plan_staged_home", lambda *a, **k: (None, None))
    return op, solves


def test_a_breaching_return_is_re_solved_not_shipped(home_planner, monkeypatch):
    op, solves = home_planner
    verdicts = iter([{"object_id": "box", "sample_index": 1, "clearance_m": -0.001}])
    monkeypatch.setattr(op, "_home_leg_breach", lambda *a: next(verdicts, None))
    out = op.plan_home_trajectories(CELL, {"robot1": [0.0] * 6})
    assert "robot1" in out and len(solves) >= 2


def test_a_return_that_always_breaches_is_not_planned(home_planner, monkeypatch):
    op, solves = home_planner
    monkeypatch.setattr(op, "_home_leg_breach",
                        lambda *a: {"object_id": "box", "sample_index": 1,
                                    "clearance_m": -0.001})
    monkeypatch.setenv("OMPL_HOME_EMIT_RETRIES", "2")
    out = op.plan_home_trajectories(CELL, {"robot1": [0.0] * 6})
    assert "robot1" not in out and len(solves) == 1 + 2


# ── 12, end to end: the welds that had no taker now plan ─────────────────────

def _weld_request(opid: str, offset: float = 0.1) -> dict:
    import json
    task = REPO / "cells" / CELL / "agentic_planner" / "tasks" / "car_door_spot_welding.json"
    o = {op["id"]: op for op in json.loads(task.read_text())["operations"]}[opid]
    x, y, z = o["position"]
    return {"robot_ns": "robot1", "operation_id": opid, "operation_type": "weld",
            "target": {"x": x, "y": y, "z": z}, "approach_vector": o["approach_vector"],
            "orientation": o["weld_orientation"], "constrain_orientation": True,
            "approach": True, "approach_offset_m": offset}


@pytest.mark.slow
def test_w01_is_planned_with_a_free_spin_where_the_exact_pose_folds_the_arm(
        geometry, clean_cache, monkeypatch):
    pytest.importorskip("ompl")
    monkeypatch.setenv("TRAJECTORY_BACKEND", "ompl")
    pb, tc = _mod("planner_backend"), clean_cache
    w02 = _weld_request("W02")
    first = pb.dispatch_plan(CELL, [w02])
    tc.cache_planner_result(CELL, [w02], first)
    assert "robot1" in first["trajectories"], first.get("failures")
    res = pb.dispatch_plan(CELL, [_weld_request("W01")])
    assert "robot1" in res["trajectories"], res.get("failures")
    chosen = res["weld_orientations"]["W01"]
    assert chosen["spin_deg"] != 0.0 and chosen["tilt_deg"] <= 4.5
    # still the weld it was asked for: the axis is inside the gate
    v = _mod("pinocchio").validate_approach.func(
        cell_id=CELL, robot_ns="robot1", expected_orientation=_weld_request("W01")["orientation"],
        trajectory=res["trajectories"]["robot1"], approach_vector=[0, 0, -1],
        target_position=[_weld_request("W01")["target"][k] for k in "xyz"])
    assert v["matches"] is True, v["reason"]


@pytest.mark.slow
def test_a_weld_that_needs_a_shorter_stroke_is_told_which(geometry, clean_cache, monkeypatch):
    pytest.importorskip("ompl")
    monkeypatch.setenv("TRAJECTORY_BACKEND", "ompl")
    pb, tc = _mod("planner_backend"), clean_cache
    w06 = _weld_request("W06")
    tc.cache_planner_result(CELL, [w06], pb.dispatch_plan(CELL, [w06]))
    res = pb.dispatch_plan(CELL, [_weld_request("W07", 0.1)])
    if "robot1" in res["trajectories"]:
        pytest.skip("W07 planned at the full stroke on this geometry")
    hint = [f for f in res["failures"] if f.get("suggested_approach_offset_m")]
    assert hint, res["failures"]
    again = pb.dispatch_plan(CELL, [_weld_request("W07", hint[0]["suggested_approach_offset_m"])])
    assert "robot1" in again["trajectories"], again.get("failures")


def test_the_task_audit_edits_only_the_changed_fields():
    sys.path.insert(0, str(REPO / "scripts"))
    audit = importlib.import_module("audit_task_programs")
    tp = importlib.import_module("agents.trajectory.task_programs")
    import json
    text = ('{\n  "cell_id": "other",\n  "operations": [\n'
            '    {"id": "W01", "position": [0, 0, 1]},\n'
            '    {"id": "W10", "position": [0, 0, 1], "position_note": "moved"}\n  ]\n}\n')
    task = json.loads(text)
    fixed, _changes = tp.normalized_program(CELL, task)
    out = audit.minimal_rewrite(text, task, fixed)
    assert json.loads(out) == fixed
    changed = [(a, b) for a, b in zip(text.splitlines(), out.splitlines()) if a != b]
    assert len(changed) == 2 and len(out.splitlines()) == len(text.splitlines())


# ── review fixes (the port's own defects) ───────────────────────────────────

def test_a_home_reset_that_leaves_an_arm_standing_says_so(clean_cache, monkeypatch):
    """H2: the replay must be able to tell the reset did not bring every arm home."""
    hr = pytest.importorskip("agents.cell.home_reset")
    pb, tc = _mod("planner_backend"), clean_cache
    starts = {"robot1": [0.1] * 6, "robot2": [0.2] * 6}
    import types
    monkeypatch.setattr(hr, "get_all_robot_states", types.SimpleNamespace(
        invoke=lambda a: {"robots": {ns: {"joint_positions": q} for ns, q in starts.items()}}))
    monkeypatch.setattr(pb, "plan_home_trajectories", lambda cell, s, **k: {
        "robot1": [{"positions": starts["robot1"], "time_s": 0.0},
                   {"positions": [0.0] * 6, "time_s": 1.0}]})
    monkeypatch.setattr(hr, "_gate_multi_arm_step",
                        lambda *a, **k: {"mode": "PARALLEL", "order": ["robot1"]})
    tc.set_home_refusal("c", "robot2", {"reason_code": "home_liftoff_blocked"},
                        start=starts["robot2"])
    tree = hr.home_reset_tree("c", ["robot1", "robot2"])
    assert tree["unhomed"] == ["robot2"]
    kids = [n["ports"]["robot_ns"] for n in tree["root"]["children"]]
    assert kids == ["robot1"], "the stuck arm is not driven"


def test_the_arm_model_is_loaded_once_and_stays_owned(geometry):
    from utils import arm_model
    m1, d1, _ = arm_model.load(CELL, "robot1")
    m2, d2, _ = arm_model.load(CELL, "robot1")
    assert m1 is m2 and d1 is not d2
    for _ in range(300):                    # the old registry evicted after 256
        arm_model.load(CELL, "robot1")
    assert arm_model.owner_of(m1) == (CELL, "robot1")


def test_validate_approach_gates_at_the_tools_own_tolerance(geometry, monkeypatch):
    pin_mod, wo = _mod("pinocchio"), _mod("weld_orientation")
    rk = _mod("robot_kinematics")
    q = _mod("planner_backend").home_config(CELL, "robot1", None)
    ori = wo.tcp_orientation(CELL, "robot1", q)
    # the expected pose 3 deg off the actual axis
    tilted = wo._d(wo._mul(wo._q(ori), wo._axis_angle([0, 1, 0], 3.0)))
    real = rk._load_robot_config
    for tol, want in ((5.0, True), (2.0, False)):
        monkeypatch.setattr(pin_mod, "_load_robot_config", lambda c, n, _t=tol: {
            **real(c, n), "tool_orientation": {"axis_tolerance_deg": _t}})
        v = pin_mod.validate_approach.func(cell_id=CELL, robot_ns="robot1",
                                           expected_orientation=tilted, joint_positions=q)
        assert v["matches"] is want, (tol, v)


def test_an_entry_with_nothing_to_judge_does_not_match(geometry):
    pin_mod, wo = _mod("pinocchio"), _mod("weld_orientation")
    q = _mod("planner_backend").home_config(CELL, "robot1", None)
    v = pin_mod.validate_approach.func(cell_id=CELL, robot_ns="robot1",
                                       expected_orientation=wo.tcp_orientation(CELL, "robot1", q),
                                       joint_positions=q, approach_vector=[0, 0, -1])
    assert v["matches"] is False and "not checkable" in v["reason"]


def test_planning_settings_refuse_infinity_and_a_malformed_block(monkeypatch):
    rk = _mod("robot_kinematics")
    monkeypatch.setattr(rk, "_load_robots_config",
                        lambda c: {"planning": {"home_liftoff_m": float("inf")}})
    assert rk.planning_setting("c", "home_liftoff_m") == 0.1
    monkeypatch.setattr(rk, "_load_robots_config", lambda c: {"planning": [1, 2]})
    assert rk.planning_setting("c", "completion_margin_s") == 1.0


def test_a_bad_dense_scan_step_is_the_default_never_zero(monkeypatch):
    sc = _mod("self_collision")
    for bad in ("0", "-1", "nan", "inf", "abc"):
        monkeypatch.setenv("SELF_COLLISION_STEP_M", bad)
        assert sc._step() == sc.SELF_COLLISION_STEP_M, bad
    monkeypatch.setenv("SELF_COLLISION_STEP_M", "0.002")
    assert sc._step() == 0.002


def test_an_explicit_zero_spin_is_never_overridden():
    wo = _mod("weld_orientation")
    req = {"approach": True, "constrain_orientation": True,
           "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}, "tool_spin_deg": 0}
    assert not wo._wanted(req, wo.policy({}))


def test_no_start_pose_is_guessed_for_an_unreadable_cell():
    from agents.home_pose import program_start_pose
    assert program_start_pose("no_such_cell_xyz", "robot1") is None


def test_a_missing_first_operation_does_not_blame_the_second(cells, clean_cache):
    bt, tc = _mod("behaviour_tree"), clean_cache
    home = _mod("planner_backend").home_config(CELL, "robot1", None)
    away = [v + 0.4 for v in home]
    tc.cache_trajectory(CELL, "robot1", "W2", [{"positions": away, "time_s": 0.0},
                                               {"positions": home, "time_s": 1.0}])
    _n, _g, breaks = bt._chain_breaks(CELL, "robot1", ["W1", "W2"])
    assert [b["reason"] for b in breaks] == ["missing_trajectory"]


def test_every_backend_lifts_off_and_compound_refuses_whole_steps():
    rrt = (REPO / "agents/trajectory/backends/rrt/planner.py").read_text()
    assert "seam_at_start(" in rrt and "liftoff_blocked_failure(" in rrt
    cmp = (REPO / "agents/trajectory/backends/ompl_compound/planner.py").read_text()
    assert "prep_failed" in cmp and "refused as a whole" in cmp


def test_the_player_serves_local_pages_only():
    sys.path.insert(0, str(REPO / "scripts"))
    player = importlib.import_module("trajectory_player")
    ok = player.origin_allowed
    assert ok(None) and ok("http://localhost:3000") and ok("http://127.0.0.1:9902")
    assert ok("vscode-webview://abc")
    assert not ok("https://evil.example") and not ok("null")


def test_the_task_audit_is_a_gate(tmp_path):
    import json
    import subprocess
    tasks = tmp_path / CELL / "agentic_planner" / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "p.json").write_text(json.dumps({"cell_id": "other", "operations": [
        {"id": "W1", "position_note": "moved", "position": [0, 0, 1]}]}, indent=2))
    run = lambda *a: subprocess.run([sys.executable, str(REPO / "scripts/audit_task_programs.py"),
                                     "--cell", CELL, "--cell-base", str(tmp_path), *a],
                                    capture_output=True, text=True)
    assert run().returncode == 1                  # pending changes
    assert run("--write").returncode == 0
    assert run().returncode == 0                  # clean now
    (tasks / "bad.json").write_text("{")
    assert run().returncode == 1                  # unreadable is reported


def test_the_converter_takes_a_per_arm_tool_policy():
    yaml = pytest.importorskip("yaml")
    conv = _converter()
    robot = {"namespace": "arms", "robot_id": "robot2", "model_name": "m",
             "robot_model": "m", "isaac_prim": "R2", "base_position": (0.0, 0.0, 0.0),
             "yaw": 0.0, "pose_source": "t", "tip_link": "link_6",
             "lowers": [-1.0] * 6, "uppers": [1.0] * 6, "vels": [2.0] * 6,
             "joint_names": ["on", "null", "j3", "j4", "j5", "j6"]}
    defaults = {"gripper": "g", "workspace_radius": 2.0, "planning_group": "p",
                "end_effector_link": "tcp", "tcp_offset": (0, 0, 0.1), "accel": [3.0] * 6,
                "rrt_j2": [0, 1], "rrt_j3": [0, 1], "mount_link": "link_6",
                "tool_axis_dir": [-1.0, 0.0, 0.0], "tool_axis_start": 0.15,
                "tool_axis_end": 0.9, "tool_axis_step": 0.15,
                "tool_orientation": {"spin_free": True, "max_tilt_deg": 4.5,
                                     "axis_tolerance_deg": 5.0},
                "tool_orientation_by_robot": {"robot2": {"spin_free": False,
                                                         "max_tilt_deg": 0.0}}}
    r = yaml.safe_load(conv.render_robots_yaml("c", "s", [robot], defaults))["robots"]["arms"]["robot2"]
    assert r["tool_orientation"]["spin_free"] is False and r["tool_orientation"]["max_tilt_deg"] == 0
    assert r["joint_names"][:2] == ["on", "null"], "names stay strings"


def test_the_converter_refuses_a_tilt_the_runtime_would_clamp():
    import subprocess
    out = subprocess.run([sys.executable, str(REPO / "scripts/convert_isaac_scene.py"),
                          "--scene-yaml", "nope.yaml", "--out-dir", "/tmp/x",
                          "--tool-max-tilt", "4.9"], capture_output=True, text=True)
    assert out.returncode != 0 and "tool-max-tilt" in (out.stdout + out.stderr)
