"""Where welding_cell_2_small's 35-minute plan went, and what now takes it back.

Measured on bt_1793277501 (27 welds, 4 arms, ompl-compound, 2,091 s):

* the final assembly took 539 s — the coupled return HOME, planned one arm after
  another through a ladder of margins;
* ~750 s of solver time went to failures, most of them ``no_path_found`` at the
  default standoff that planned at the next try with ``approach_offset_m: 0.30``;
* the opening batch stopped at its first step and threw away 23 operations;
* one via point (``W18_VIA``) was tried six times and never planned;
* the allocation took 206 s, for the same task as the runs before;
* every step waited for its slowest arm: 4 of 11 steps left an arm idle with work.

Pinned here, one block per fix:

1. HOME: the arms' direct returns are planned at once and kept ONLY when the
   gate runs them together; anything else falls back to the coupled planner.
   An accepted return is reused within the run.
2. The recovery skill has the model DECIDE on the approach distance (from the
   obstacle geometry) before reaching for a via point; the code prescribes nothing.
3. A batch skips only the steps that depend on a failed one.
4. ``ompl-compound`` runs its independent solves at once (opt-in).
5. The allocation is reused for an unchanged task and cell config (opt-in).
6. The retry cap is enforced by the planner, per arm and operation family.
7. A stepwise program is offered without its step barriers when that is faster.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")


def _leg(*qs, dt=1.0):
    return [{"positions": list(q), "time_s": round(i * dt, 4)} for i, q in enumerate(qs)]


def _shape(traj):
    return [list(p["positions"]) for p in (traj or [])]


# ── 1. the return HOME ───────────────────────────────────────────────────────

CELL = "welding_cell_2_small"
HOME = [0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0]
SEAM = {"robot1": [-2.372, -0.923, 0.918, 0.611, 2.12, 0.953],
        "robot2": [1.029, -1.39, 1.355, 0.717, 1.981, -1.94]}


class _FakeModel:
    nq = 6

    def createData(self):
        return None


@pytest.fixture
def op():
    return pytest.importorskip("agents.trajectory.tools.ompl_planner")


@pytest.fixture
def home_rig(op, monkeypatch):
    # The gate is patched where the planner imports it from.
    pytest.importorskip("agents.trajectory.tools.behaviour_tree")
    rig = {"gate_mode": "PARALLEL", "gate_calls": [], "asked": []}

    def fake_plan(cell_id, ns, start, goal, moving_obstacle_spheres=None, **kw):
        tube = bool(moving_obstacle_spheres)
        rig["asked"].append({"robot_ns": ns, "with_tube": tube})
        return _leg(list(start), list(goal)) if not tube else _leg(list(start), [9.0] * 6,
                                                                   list(goal))

    def fake_gate(cell_id, legs, **kw):
        rig["gate_calls"].append(sorted(legs))
        mode = rig["gate_mode"]
        return {"mode": mode, "order": sorted(legs) if mode else [],
                "parallel": {"min_clearance_m": 0.4, "warning_distance_m": 0.2}}

    monkeypatch.setattr(op, "plan_joint_path", fake_plan)
    monkeypatch.setattr("agents.trajectory.tools.behaviour_tree._gate_multi_arm_step",
                        fake_gate, raising=False)
    monkeypatch.setattr(op, "_plan_staged_home", lambda *a, **k: (None, None))
    monkeypatch.setattr(op, "_home_liftoff", lambda *a, **k: [])
    monkeypatch.setattr(op, "_import_ompl", lambda: object())
    monkeypatch.setattr(op, "_home_config", lambda *a, **k: list(HOME))
    monkeypatch.setattr(op, "_trajectory_tool_rise", lambda model, traj, base, yaw: 0.0)
    monkeypatch.setattr(op, "_home_leg_breach", lambda *a, **k: None)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_collision_config",
        lambda _cell: {"warning_distance": 0.2}, raising=False)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_pinocchio_model",
        lambda *a, **k: (_FakeModel(), None), raising=False)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_robot_config",
        lambda *a, **k: {"base_position": [0.0, 0.0, 0.0], "base_orientation_yaw": 0.0},
        raising=False)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._config_spheres",
        lambda *a, **k: [np.array([0.5, 0.5, 1.0])], raising=False)
    monkeypatch.setenv("OMPL_PARALLEL_CANDIDATES", "0")
    return rig


def test_the_fast_path_is_off_unless_the_stack_turns_it_on(op, home_rig, monkeypatch):
    monkeypatch.delenv("HOME_PARALLEL_FASTPATH", raising=False)
    op.plan_home_trajectories(CELL, dict(SEAM))
    assert any(a["with_tube"] for a in home_rig["asked"]), (
        "with the flag unset the arms must be planned coupled, as before")


def test_direct_returns_the_gate_runs_together_are_kept(op, home_rig, monkeypatch):
    monkeypatch.setenv("HOME_PARALLEL_FASTPATH", "1")
    out = op.plan_home_trajectories(CELL, dict(SEAM))
    assert set(out) == {"robot1", "robot2"}
    assert not any(a["with_tube"] for a in home_rig["asked"]), \
        "an accepted fast path must not also run the coupled ladder"
    assert len(home_rig["asked"]) == 2, "one direct solve per arm"
    assert home_rig["gate_calls"][0] == ["robot1", "robot2"], \
        "the WHOLE set goes through the gate, not arm by arm"
    for ns, q in SEAM.items():
        assert _shape(out[ns])[0] == q and _shape(out[ns])[-1] == HOME


@pytest.mark.parametrize("verdict", ["SEQUENTIAL", None])
def test_anything_short_of_parallel_falls_back_to_the_coupled_planner(
        op, home_rig, monkeypatch, verdict):
    """A SEQUENTIAL verdict would cost cycle time, None is a refusal: neither ships
    from the fast path."""
    monkeypatch.setenv("HOME_PARALLEL_FASTPATH", "1")
    home_rig["gate_mode"] = verdict
    op.plan_home_trajectories(CELL, dict(SEAM))
    assert any(a["with_tube"] for a in home_rig["asked"]), \
        "the coupled planner must take over when the gate does not run them together"


def test_a_gate_that_raises_is_not_a_yes(op, home_rig, monkeypatch):
    monkeypatch.setenv("HOME_PARALLEL_FASTPATH", "1")

    def boom(*a, **k):
        raise RuntimeError("no geometry")

    monkeypatch.setattr("agents.trajectory.tools.behaviour_tree._gate_multi_arm_step",
                        boom, raising=False)
    op.plan_home_trajectories(CELL, dict(SEAM))
    assert any(a["with_tube"] for a in home_rig["asked"])


def test_a_tall_or_breaching_direct_return_falls_back(op, home_rig, monkeypatch):
    monkeypatch.setenv("HOME_PARALLEL_FASTPATH", "1")
    monkeypatch.setattr(op, "_emitted_rise_limit", lambda: 0.3)
    monkeypatch.setattr(op, "_trajectory_tool_rise", lambda m, traj, b, y: 0.9)
    assert op._parallel_direct_home(CELL, ["robot1", "robot2"], dict(SEAM), None, 0.4) is None
    monkeypatch.setattr(op, "_trajectory_tool_rise", lambda m, traj, b, y: 0.0)
    monkeypatch.setattr(op, "_home_leg_breach", lambda *a, **k: {"object_id": "fx"})
    assert op._parallel_direct_home(CELL, ["robot1", "robot2"], dict(SEAM), None, 0.4) is None


def test_an_arm_with_no_way_out_of_its_seam_falls_back(op, home_rig, monkeypatch):
    monkeypatch.setenv("HOME_PARALLEL_FASTPATH", "1")
    monkeypatch.setattr(op, "_home_liftoff", lambda *a, **k: None)
    assert op._parallel_direct_home(CELL, ["robot1", "robot2"], dict(SEAM), None, 0.4) is None


def test_fork_map_returns_results_in_order_or_none_when_disabled(op, monkeypatch):
    monkeypatch.setenv("OMPL_PARALLEL_CANDIDATES", "0")
    assert op._fork_map([lambda: 1, lambda: 2]) is None
    monkeypatch.setenv("OMPL_PARALLEL_CANDIDATES", "1")
    if not op._parallel_enabled():
        pytest.skip("no fork start method here")
    assert op._fork_map([lambda: 1, lambda: [2, 3], lambda: 1 / 0]) == [1, [2, 3], None]


def test_an_accepted_home_step_is_reused_within_a_run(monkeypatch):
    bt = pytest.importorskip("agents.trajectory.tools.behaviour_tree")
    from agents.trajectory.tools import plan_memo
    calls = []

    def planned(cell_id, robots, per_robot_ops):
        calls.append(sorted(robots))
        return {"type": "Parallel", "children": []}, {"mode": "PARALLEL"}

    monkeypatch.setattr(bt, "_home_step_planned", planned)
    monkeypatch.setattr(bt, "_last_op_end_joints", lambda c, ns, ops: [0.1] * 6)
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: "fp")
    ops = {"robot1": ["W1"], "robot2": ["W2"]}
    with plan_memo.run_scope():
        n1, r1 = bt._home_step("c", ["robot1", "robot2"], ops)
        n2, r2 = bt._home_step("c", ["robot1", "robot2"], ops)
    assert calls == [["robot1", "robot2"]], "the second assembly must not re-plan HOME"
    assert r2.get("memoised") is True and n2 == n1
    bt._home_step("c", ["robot1", "robot2"], ops)          # outside a run: planned
    assert len(calls) == 2


def test_a_refused_home_step_is_planned_again(monkeypatch):
    bt = pytest.importorskip("agents.trajectory.tools.behaviour_tree")
    from agents.trajectory.tools import plan_memo
    calls = []
    monkeypatch.setattr(bt, "_home_step_planned",
                        lambda c, r, o: calls.append(1) or (None, {"mode": None}))
    monkeypatch.setattr(bt, "_last_op_end_joints", lambda c, ns, ops: [0.1] * 6)
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: "fp")
    with plan_memo.run_scope():
        bt._home_step("c", ["robot1", "robot2"], {})
        bt._home_step("c", ["robot1", "robot2"], {})
    assert len(calls) == 2


# ── 2. the skill: a longer approach first ────────────────────────────────────

def test_the_skill_asks_the_model_to_decide_the_approach_before_a_via_point():
    """The decision is the model's, made from the skill — not a rule in code."""
    text = (PROJECT_ROOT / "skills/trajectory/recover-from-planner-failure/SKILL.md").read_text()
    assert "Before any via point, decide on the APPROACH DISTANCE" in text
    sec = text.split("### `no_path_found`", 1)[1].split("###", 1)[0]
    assert sec.index("approach distance") < sec.index("VIA-POINT"), \
        "the approach decision must come before the via point in the ladder"
    assert "When NOT to lengthen it" in text, "the skill must say when it does not apply"
    plan = (PROJECT_ROOT / "skills/trajectory/plan-trajectories/SKILL.md").read_text()
    assert "decide on the approach distance before a via point" in plan


def test_the_code_does_not_prescribe_the_approach_change():
    """The recovery table the planner attaches to failures stays neutral: the
    approach-distance decision is taken by the model through the skill."""
    from agents.trajectory.reason_codes import recovery
    for code in ("no_path_found", "excessive_tool_rise"):
        assert not any("approach_offset_m" in lv for lv in recovery(code)["levers"]), code


# ── 3. a batch skips only what depends on the failure ────────────────────────

@pytest.fixture
def pt():
    return pytest.importorskip("agents.trajectory.tools.planning_tools")


def _fake_planner(pt, monkeypatch, fail_ops=()):
    from agents.trajectory.tools import planner_backend
    calls = []

    def dispatch(cell_id, reqs, **kw):
        calls.append([r["operation_id"] for r in reqs])
        trajs = {r["robot_ns"]: _leg([0.0] * 6, [0.1] * 6)
                 for r in reqs if r["operation_id"] not in fail_ops}
        fails = [{"operation_id": r["operation_id"], "robot_ns": r["robot_ns"],
                  "reason_code": "no_path_found"} for r in reqs
                 if r["operation_id"] in fail_ops]
        return {"success": bool(trajs), "trajectories": trajs, "failures": fails,
                "planning_time_s": 0.1, "backend": "ompl"}

    monkeypatch.setattr(planner_backend, "dispatch_plan", dispatch)
    monkeypatch.setattr(pt, "resolve_velocity_scale", lambda c, v: 0.5)
    monkeypatch.setattr(pt, "cache_planner_result", lambda *a: 0)
    monkeypatch.setattr(pt, "_step_gate", lambda c, legs: {"verdict": "PARALLEL"})
    return calls


def _req(ns, op):
    return {"robot_ns": ns, "operation_id": op, "target": {}}


def test_a_step_on_other_arms_is_still_planned_after_a_failure(pt, monkeypatch):
    calls = _fake_planner(pt, monkeypatch, fail_ops=("W1",))
    res = pt.run_trajectory_planner.func(cell_id="c", steps=[
        [_req("robot1", "W1"), _req("robot2", "W2")],     # fails (robot1)
        [_req("robot3", "W3"), _req("robot4", "W4")],     # independent: planned
        [_req("robot1", "W5")],                            # robot1 blocked: skipped
        [_req("robot3", "W6"), _req("robot2", "W7")],     # robot2 blocked: skipped
        [_req("robot4", "W8")],                            # independent: planned
    ])
    assert calls == [["W1", "W2"], ["W3", "W4"], ["W8"]]
    assert res["stopped_at"] == 0 and res["complete"] is False
    assert res["skipped_steps"] == [2, 3] and res["kept_steps_after_stop"] == [1, 4]
    assert res["planned_steps"] == 2
    by_op = {f["operation_id"]: f["reason_code"] for f in res["failures"]}
    assert by_op == {"W1": "no_path_found", "W5": "not_planned",
                     "W6": "not_planned", "W7": "not_planned"}
    assert "[0, 2, 3]" in res["advice"] and "do not send them again" in res["advice"]


# ── 4. ompl-compound runs its solves at once (opt-in) ────────────────────────

def test_compound_solves_use_the_shared_parallel_batch():
    src = (PROJECT_ROOT / "agents/trajectory/backends/ompl_compound/planner.py").read_text()
    assert "_parallel_solves(_one_solve, n_attempts)" in src
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
    assert 'OMPL_PARALLEL_CANDIDATES: "${OMPL_PARALLEL_CANDIDATES:-1}"' in compose


# ── 5. the allocation is reused for an unchanged task ────────────────────────

def test_the_process_memo_is_opt_in_and_keyed_on_config(monkeypatch):
    from agents.trajectory.tools import plan_memo
    plan_memo.clear()
    fp = {"v": "A"}
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: fp["v"])
    monkeypatch.delenv("PLAN_MEMO_PROCESS", raising=False)
    k = plan_memo.make_key("c", "mrta", [1])
    plan_memo.put("mrta", k, {"x": 1})
    assert plan_memo.get("mrta", k) is None, "off unless the stack turns it on"
    monkeypatch.setenv("PLAN_MEMO_PROCESS", "1")
    plan_memo.put("mrta", k, {"x": 1})
    got = plan_memo.get("mrta", k)
    assert got == {"x": 1}
    got["x"] = 2
    assert plan_memo.get("mrta", k) == {"x": 1}, "a caller must not mutate the memo"
    fp["v"] = "B"
    assert plan_memo.get("mrta", plan_memo.make_key("c", "mrta", [1])) is None, \
        "a config edit must miss"
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: None)
    assert plan_memo.make_key("c", "mrta", [1]) is None, "unknown config never hits"
    plan_memo.clear()


def test_the_allocation_is_computed_once_for_the_same_task(pt, monkeypatch, tmp_path):
    from agents.trajectory.tools import plan_memo
    from agents.trajectory.tools import experience_store as es
    plan_memo.clear()
    es.clear()
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))    # the memo persists: not in ./cells
    monkeypatch.setenv("CELL_DATA_MODE", "local")
    monkeypatch.setenv("PLAN_MEMO_PROCESS", "1")
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: "fp")
    robots = {"robot1": {"base_position": [0, 1, 0]}, "robot2": {"base_position": [0, -1, 0]}}
    monkeypatch.setattr(pt, "_cell_robots", lambda c: robots)
    n = {"alloc": 0}
    real = pt._regional_allocate

    def counting(*a, **k):
        n["alloc"] += 1
        return real(*a, **k)

    monkeypatch.setattr(pt, "_regional_allocate", counting)
    monkeypatch.setattr(pt, "_servicability_check", lambda c: None)
    ops = [{"id": "W1", "position": [0.0, 0.8, 1.0], "approach_vector": [0, -1, 0]},
           {"id": "W2", "position": [0.0, -0.8, 1.0], "approach_vector": [0, 1, 0]}]
    a = pt.run_mrta_pipeline.func(cell_id="c", operations=ops)
    b = pt.run_mrta_pipeline.func(cell_id="c", operations=ops)
    assert "error" not in a, a
    assert n["alloc"] == 1 and b.get("cached") is True
    assert b["steps"] == a["steps"]
    pinned = [dict(ops[0], assigned_robot="robot2"), ops[1]]
    c = pt.run_mrta_pipeline.func(cell_id="c", operations=pinned)
    assert c.get("ignored_pins") == {"W1": "robot2"}, \
        "a model's pin is reported on a cache hit too"
    plan_memo.clear()


# ── 6. the retry cap ─────────────────────────────────────────────────────────

def test_helper_waypoints_share_their_weld_family():
    from agents.trajectory import retry_budget as rb
    assert rb.family("W18_VIA") == rb.family("W18_VIA2") == rb.family("W18_VIA_3")
    assert rb.family("W18_R_B_pillar") == "W18_R_B_pillar"
    assert rb.family("R1_CLEAR") == rb.family("R1_CLEAR_2")


def test_the_cap_refuses_after_three_failures_on_one_arm(monkeypatch):
    from agents.trajectory import retry_budget as rb
    monkeypatch.delenv("PLAN_RETRY_CAP", raising=False)
    req = {"robot_ns": "robot2", "operation_id": "W18_VIA4"}
    assert rb.refusal(req) is None, "outside a run nothing is refused"
    with rb.scope():
        for op in ("W18_VIA", "W18_VIA2", "W18_VIA3"):
            assert rb.refusal({"robot_ns": "robot2", "operation_id": op}) is None
            rb.record([{"robot_ns": "robot2", "operation_id": op,
                        "reason_code": "excessive_tool_rise"}])
        hit = rb.refusal(req)
        assert hit["reason_code"] == "retry_cap_reached" and hit["attempts"] == 3
        assert rb.refusal({"robot_ns": "robot4", "operation_id": "W18_VIA"}) is None, \
            "another arm has its own budget"
        rb.record([{"robot_ns": "robot1", "operation_id": "W1", "reason_code": c}
                   for c in ("step_refused", "not_planned", "blocked_by_parked_arm")] * 3)
        assert rb.refusal({"robot_ns": "robot1", "operation_id": "W1"}) is None, \
            "failures another arm caused do not count"


def test_the_planner_refuses_a_capped_request_without_solving(pt, monkeypatch):
    from agents.trajectory import retry_budget as rb
    calls = _fake_planner(pt, monkeypatch, fail_ops=("W18_VIA",))
    with rb.scope():
        for _ in range(3):
            pt.run_trajectory_planner.func(cell_id="c",
                                           motion_requests=[_req("robot2", "W18_VIA")])
        res = pt.run_trajectory_planner.func(cell_id="c",
                                             motion_requests=[_req("robot2", "W18_VIA")])
    assert len(calls) == 3, "the fourth attempt must not reach the solver"
    f = res["failures"][0]
    assert f["reason_code"] == "retry_cap_reached"
    assert f["recovery"]["escalate"] == "reallocate"


# ── 7. no waiting for the slowest arm ────────────────────────────────────────

@pytest.fixture
def bt():
    return pytest.importorskip("agents.trajectory.tools.behaviour_tree")


_PLAN = {"steps": [
    {"mode": "PARALLEL", "robots": [
        {"robot_ns": "robot1", "operation_id": "W1", "operation_type": "weld"},
        {"robot_ns": "robot2", "operation_id": "W2", "operation_type": "weld",
         "hold_duration_s": 0.8}]},
    {"mode": "PARALLEL", "robots": [
        {"robot_ns": "robot1", "operation_id": "W3", "operation_type": "weld"}]}]}
_STEPWISE = {"success": True, "behaviour_tree": {"tree_id": "bt_step"},
             "program_cost": {"cycle_time_s": 40.0, "home_s": 5.0}}


def _coord(bt, monkeypatch, result):
    from agents.trajectory.tools import coordinated_tree as ct
    seen = {}

    def fake(cell_id, sequences, hold_duration_s=None, operation_types=None,
             home_at_end=True, tail_steps=None):
        seen.update(sequences=sequences, holds=hold_duration_s, kinds=operation_types)
        return dict(result)

    monkeypatch.setattr(ct, "assemble_coordinated", fake)
    return seen


def test_a_faster_barrier_free_tree_replaces_the_stepwise_one(bt, monkeypatch):
    seen = _coord(bt, monkeypatch, {"success": True, "validation": {"valid": True},
                                    "behaviour_tree": {"tree_id": "bt_coord"},
                                    "makespan_estimate_s": 28.0})
    out = bt._barrier_free("c", _PLAN, _PLAN["steps"], dict(_STEPWISE))
    assert seen["sequences"] == {"robot1": ["W1", "W3"], "robot2": ["W2"]}, \
        "same operations, same order per arm"
    assert seen["holds"] == {"W2": 0.8} and seen["kinds"]["W1"] == "weld"
    assert out["behaviour_tree"]["tree_id"] == "bt_coord"
    assert out["barrier_free"]["used"] is True and out["barrier_free"]["saved_s"] == 7.0
    assert out["program_cost"]["cycle_time_s"] == 33.0


@pytest.mark.parametrize("result,why", [
    ({"success": True, "validation": {"valid": True}, "makespan_estimate_s": 34.5},
     "not faster"),
    ({"success": False, "error": "deadlock", "deadlock": {"arm": "robot1"}}, "deadlock"),
    ({"success": True, "validation": {"valid": False}, "makespan_estimate_s": 10.0},
     "did not validate"),
])
def test_otherwise_the_stepwise_tree_stands_and_says_why(bt, monkeypatch, result, why):
    _coord(bt, monkeypatch, result)
    stepwise = dict(_STEPWISE)
    assert bt._barrier_free("c", _PLAN, _PLAN["steps"], stepwise) is None
    assert stepwise["barrier_free"]["used"] is False
    assert why in stepwise["barrier_free"]["reason"]


def test_the_barrier_free_pass_can_be_turned_off(bt, monkeypatch):
    seen = _coord(bt, monkeypatch, {"success": True})
    assert bt._barrier_free("c", {**_PLAN, "barrier_free": False}, _PLAN["steps"],
                            dict(_STEPWISE)) is None
    monkeypatch.setenv("STEPWISE_BARRIER_FREE", "0")
    assert bt._barrier_free("c", _PLAN, _PLAN["steps"], dict(_STEPWISE)) is None
    assert not seen, "nothing coordinated when it is off"


def test_a_single_arm_program_has_no_barriers_to_remove(bt, monkeypatch):
    seen = _coord(bt, monkeypatch, {"success": True})
    plan = {"steps": [{"mode": "SEQUENTIAL", "robots": [
        {"robot_ns": "robot1", "operation_id": "W1"}]}]}
    assert bt._barrier_free("c", plan, plan["steps"], dict(_STEPWISE)) is None
    assert not seen
