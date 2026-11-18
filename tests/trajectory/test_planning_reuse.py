"""What the planner already solved is kept across runs — and never trusted.

Two stores (``tools/experience_store.py``), both off unless the stack enables them:

* the LEG STORE: a whole planned step, reused when the cell config, the backend,
  the requests and where every arm stands are all unchanged — after the
  platform's leg check has passed every stored leg again;
* PATH EXPERIENCE: paths the compound search found, reconnected to a new start
  and goal of the same arms and offered to the search as candidates its own
  checker must pass (the retrieve-and-repair idea of OMPL Lightning/Thunder).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")

CELL = "reuse_cell"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway local cell tree with both stores on."""
    (tmp_path / CELL / "agentic_planner" / "config").mkdir(parents=True)
    (tmp_path / CELL / "agentic_planner" / "config" / "robots.yaml").write_text("robots: {}\n")
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("CELL_DATA_MODE", "local")
    monkeypatch.setenv("PLAN_LEG_STORE", "1")
    monkeypatch.setenv("PLAN_EXPERIENCE", "1")
    from agents.trajectory.tools import experience_store as es
    from agents.trajectory.tools import trajectory_cache as tc
    es.clear()
    tc.reset_robot_configs(CELL)
    tc.clear_trajectory_cache(CELL)
    yield es
    es.clear()
    tc.reset_robot_configs(CELL)
    tc.clear_trajectory_cache(CELL)


def _leg(*qs):
    return [{"positions": list(q), "time_s": float(i)} for i, q in enumerate(qs)]


# ── the store itself ─────────────────────────────────────────────────────────

def test_both_stores_are_off_unless_enabled(monkeypatch):
    from agents.trajectory.tools import experience_store as es
    monkeypatch.delenv("PLAN_LEG_STORE", raising=False)
    monkeypatch.delenv("PLAN_EXPERIENCE", raising=False)
    es.clear()
    assert es.entries(CELL, es.LEG_STORE) == []
    assert es.append(CELL, es.PATH_EXPERIENCE, {"x": 1}) is False


def test_entries_persist_across_a_process_restart(store):
    es = store
    assert es.append(CELL, es.LEG_STORE, {"key": "a", "n": 1})
    assert es.append(CELL, es.LEG_STORE, {"key": "b", "n": 2})
    es.clear()                                    # what a restart forgets
    assert [e["key"] for e in es.entries(CELL, es.LEG_STORE)] == ["a", "b"]


def test_a_torn_line_is_skipped(store, tmp_path):
    es = store
    es.append(CELL, es.LEG_STORE, {"key": "a"})
    from utils import cell_store
    cell_store.append_text(CELL, cell_store.KIND_RECORDINGS, es.LEG_STORE, '{"key": "b"')
    es.clear()
    assert [e["key"] for e in es.entries(CELL, es.LEG_STORE)] == ["a"]


def test_the_store_is_compacted_when_it_outgrows_its_cap(store, monkeypatch):
    es = store
    monkeypatch.setattr(es, "MAX_ENTRIES", 4)
    for i in range(7):
        es.append(CELL, es.LEG_STORE, {"key": str(i)})
    es.clear()
    keys = [e["key"] for e in es.entries(CELL, es.LEG_STORE)]
    assert keys[-1] == "6" and len(keys) <= 6


# ── the leg store ────────────────────────────────────────────────────────────

REQ = [{"robot_ns": "robot1", "operation_id": "W1", "target": {"x": 1, "y": 0, "z": 1}}]


def _key(**over):
    from agents.trajectory.tools import leg_reuse
    args = dict(cell_id=CELL, backend="ompl", motion_requests=REQ, approach_offset_m=0.1,
                velocity_scale=None, options={}, backend_kwargs={})
    args.update(over)
    return leg_reuse.step_key(**args)


def test_the_key_moves_with_everything_the_step_depends_on(store, monkeypatch):
    from agents.trajectory.tools import plan_memo
    from agents.trajectory.tools import trajectory_cache as tc
    base = _key()
    assert base is not None and base == _key()
    assert _key(backend="ompl-compound") != base
    assert _key(motion_requests=[{**REQ[0], "approach_offset_m": 0.3}]) != base
    assert _key(approach_offset_m=0.2) != base
    tc.set_last_config(CELL, "robot2", [0.1] * 6)        # a BYSTANDER moved
    assert _key() != base
    tc.reset_robot_configs(CELL)
    monkeypatch.setenv("OMPL_COMPOUND_ATTEMPTS", "2")
    assert _key() != base
    monkeypatch.delenv("OMPL_COMPOUND_ATTEMPTS")
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: "edited")
    assert _key() != base, "a config edit must miss"
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: None)
    assert _key() is None, "an unknown config is never reused"


@pytest.fixture
def dispatch(store, monkeypatch):
    """dispatch_plan with a counting fake backend and the geometry-bound
    post-checks stood in for."""
    from agents.trajectory.tools import planner_backend as pb
    from agents.trajectory.tools import trajectory_cache as tc
    calls = []

    def backend(cell_id, reqs, **kw):
        calls.append([r["operation_id"] for r in reqs])
        trajs = {}
        for r in reqs:
            start = r.get("current_joints") or tc.get_last_config(cell_id, r["robot_ns"]) \
                or [0.0] * 6
            end = [v + 0.5 for v in start]
            trajs[r["robot_ns"]] = _leg(start, end)
            tc.set_last_config(cell_id, r["robot_ns"], end)        # the commit
        return {"success": True, "trajectories": trajs, "failures": [],
                "planning_time_s": 12.5}

    monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: backend)
    monkeypatch.setattr(pb, "resolve_planning_backend", lambda *a: "ompl")
    monkeypatch.setattr(pb, "_self_collision_refusals", lambda c, r: [])
    monkeypatch.setattr(pb, "_auto_weld_orientation", lambda c, r, a: (r, None, None))
    for name in ("_contract_refusals", "_limit_refusals", "_record_weld_poses",
                 "_stroke_hints"):
        monkeypatch.setattr(pb, name, lambda *a, **k: None)
    verdicts = {"refuse": False}
    monkeypatch.setattr("agents.trajectory.tools.leg_verdict.refusal",
                        lambda c, ns, leg, static=False: (
                            {"reason_code": "trajectory_in_collision"}
                            if verdicts["refuse"] else None))
    return pb, calls, verdicts


def _fresh_run():
    """What a new planning run starts from: no chaining state, nothing cached."""
    from agents.trajectory.tools import trajectory_cache as tc
    tc.reset_robot_configs(CELL)
    tc.clear_trajectory_cache(CELL)


def test_the_same_step_in_a_later_run_is_reused_not_solved(dispatch):
    pb, calls, _ = dispatch
    from agents.trajectory.tools import experience_store as es
    from agents.trajectory.tools import trajectory_cache as tc
    first = pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    end = tc.get_last_config(CELL, "robot1")
    _fresh_run()
    es.clear()                                       # and a process restart
    second = pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    assert calls == [["W1"]], "the second run must not reach the backend"
    assert second["trajectories"] == first["trajectories"]
    assert second["reused_step"]["planning_time_s_saved"] == 12.5
    assert second["planning_time_s"] == 0.0
    assert tc.get_last_config(CELL, "robot1") == end, \
        "a reused step commits what the backend committed, so the next leg chains"


def test_a_step_from_a_different_pose_is_solved(dispatch):
    pb, calls, _ = dispatch
    from agents.trajectory.tools import trajectory_cache as tc
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    _fresh_run()
    tc.set_last_config(CELL, "robot1", [0.3] * 6)
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    assert len(calls) == 2


def test_a_stored_step_that_fails_its_recheck_is_forgotten_and_solved(dispatch):
    pb, calls, verdicts = dispatch
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    _fresh_run()
    verdicts["refuse"] = True
    out = pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    assert len(calls) == 2 and "reused_step" not in out
    verdicts["refuse"] = False
    _fresh_run()
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    assert len(calls) == 2, "the fresh plan was stored in its place and is reused"


def test_a_step_that_failed_is_not_stored(dispatch, monkeypatch):
    pb, calls, _ = dispatch

    def failing(cell_id, reqs, **kw):
        calls.append([r["operation_id"] for r in reqs])
        return {"success": False, "trajectories": {}, "planning_time_s": 1.0,
                "failures": [{"robot_ns": "robot1", "reason_code": "no_path_found"}]}

    monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: failing)
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    _fresh_run()
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    assert len(calls) == 2


def test_with_the_store_off_every_step_is_solved(dispatch, monkeypatch):
    pb, calls, _ = dispatch
    monkeypatch.setenv("PLAN_LEG_STORE", "0")
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    _fresh_run()
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    assert len(calls) == 2


# ── path experience ──────────────────────────────────────────────────────────

ARMS = ["robot1", "robot2"]
S = [0.0] * 12
G = [1.0] * 12
DETOUR = [S, [0.5] * 6 + [0.2] * 6, [0.8] * 12, G]


def test_a_found_path_is_offered_for_a_nearby_problem_repaired_to_it(store):
    from agents.trajectory.tools import path_experience as px
    assert px.record(CELL, ARMS, S, G, DETOUR)
    s2, g2 = [0.05] * 12, [0.95] * 12
    cands = px.candidates(CELL, ARMS, s2, g2)
    assert cands, "a nearby problem must get the stored path as a candidate"
    assert cands[0][0] == s2 and cands[0][-1] == g2, "repaired to the NEW endpoints"
    assert cands[0][1:-1] == DETOUR[1:-1], "the stored interior is kept"


def test_experience_does_not_cross_arm_sets_or_far_problems(store, monkeypatch):
    from agents.trajectory.tools import path_experience as px
    px.record(CELL, ARMS, S, G, DETOUR)
    assert px.candidates(CELL, ["robot1", "robot3"], S, G) == []
    assert px.candidates(CELL, ARMS, [3.0] * 12, G) == []
    monkeypatch.setenv("PLAN_EXPERIENCE_MAX_RAD", "0")
    assert px.candidates(CELL, ARMS, [0.05] * 12, G) == []


def test_straight_lines_and_repeats_are_not_recorded(store):
    from agents.trajectory.tools import path_experience as px
    from agents.trajectory.tools import experience_store as es
    assert px.record(CELL, ARMS, S, G, [S, G]) is False
    assert px.record(CELL, ARMS, S, G, DETOUR) is True
    assert px.record(CELL, ARMS, S, G, DETOUR) is False
    assert len(es.entries(CELL, es.PATH_EXPERIENCE)) == 1


def test_experience_survives_a_restart(store):
    from agents.trajectory.tools import path_experience as px
    from agents.trajectory.tools import experience_store as es
    px.record(CELL, ARMS, S, G, DETOUR)
    es.clear()
    assert px.candidates(CELL, ARMS, S, G)


def test_the_compound_search_consults_experience_before_searching():
    """Wiring, pinned at source level (the solver itself needs OMPL): experience
    candidates are checked with ``_clear`` and only returned when they meet the
    rise target; search results are recorded."""
    src = (PROJECT_ROOT / "agents/trajectory/backends/ompl_compound/planner.py").read_text()
    body = src.split("def _solve_compound(", 1)[1].split("\ndef ", 1)[0]
    i_exp = body.index("path_experience.candidates(")
    assert "if _clear(p)]" in body[i_exp:i_exp + 200], "every candidate must pass _clear"
    assert body.index("score(best) <= target") < body.index("_one_solve()"), \
        "a good stored path must be able to skip the search"
    assert "path_experience.record(" in body
    assert "cell_id=cell_id)" in src


# ── the allocation survives a restart ────────────────────────────────────────

def test_the_allocation_memo_is_kept_on_disk(store, monkeypatch):
    from agents.trajectory.tools import plan_memo
    from agents.trajectory.tools import experience_store as es
    monkeypatch.setenv("PLAN_MEMO_PROCESS", "1")
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: "fp")
    plan_memo.clear()
    k = plan_memo.make_key(CELL, "mrta", [1])
    plan_memo.put("mrta", k, {"steps": [1, 2]}, cell_id=CELL)
    plan_memo.clear()                 # a restarted planner: memory is empty ...
    es.clear()
    assert plan_memo.get("mrta", k, cell_id=CELL) == {"steps": [1, 2]}, "... the disk is not"
    assert plan_memo.get("mrta", k) == {"steps": [1, 2]}, "and the hit warmed memory"
    plan_memo.clear()
    monkeypatch.setattr(plan_memo, "config_fingerprint", lambda c: "edited")
    assert plan_memo.get("mrta", plan_memo.make_key(CELL, "mrta", [1]), cell_id=CELL) is None
    plan_memo.clear()


def test_only_the_allocation_namespace_is_persisted(store, monkeypatch):
    from agents.trajectory.tools import plan_memo
    from agents.trajectory.tools import experience_store as es
    monkeypatch.setenv("PLAN_MEMO_PROCESS", "1")
    plan_memo.put("other", "k", {"x": 1}, cell_id=CELL)
    assert es.entries(CELL, es.MRTA_MEMO) == []
    plan_memo.clear()


# ── the agent's own returns home stay steps after the coordinated work ───────

def test_trailing_home_moves_are_split_off():
    bt = pytest.importorskip("agents.trajectory.tools.behaviour_tree")
    steps = [
        {"robots": [{"robot_ns": "robot1", "operation_id": "W1", "operation_type": "weld"}]},
        {"robots": [{"robot_ns": "robot1", "operation_id": "robot1_HOME", "operation_type": "move"},
                    {"robot_ns": "robot2", "operation_id": "robot2_HOME2", "operation_type": "move"}]},
        {"robots": [{"robot_ns": "robot4", "operation_id": "robot4_HOME3", "operation_type": "move"}]},
    ]
    work, tail = bt._split_home_tail(steps)
    assert work == steps[:1] and tail == steps[1:]
    # a home move in the MIDDLE of the program is ordinary work
    work, tail = bt._split_home_tail(steps[1:2] + steps[:1])
    assert tail == []


def test_the_barrier_free_pass_hands_the_home_tail_over_and_costs_it(monkeypatch):
    bt = pytest.importorskip("agents.trajectory.tools.behaviour_tree")
    from agents.trajectory.tools import coordinated_tree as ct
    seen = {}

    def fake(cell_id, sequences, hold_duration_s=None, operation_types=None,
             home_at_end=True, tail_steps=None):
        seen.update(sequences=sequences, tail=tail_steps, home=home_at_end)
        return {"success": True, "validation": {"valid": True}, "makespan_estimate_s": 60.0,
                "tail_s": 9.0, "behaviour_tree": {"tree_id": "bt_c"}}

    monkeypatch.setattr(ct, "assemble_coordinated", fake)
    plan = {"home_at_end": False, "steps": [
        {"robots": [{"robot_ns": "robot1", "operation_id": "W1"},
                    {"robot_ns": "robot2", "operation_id": "W2"}]},
        {"robots": [{"robot_ns": "robot1", "operation_id": "robot1_HOME", "operation_type": "move"},
                    {"robot_ns": "robot2", "operation_id": "robot2_HOME", "operation_type": "move"}]}]}
    stepwise = {"success": True, "behaviour_tree": {"tree_id": "bt_s"},
                "program_cost": {"cycle_time_s": 93.1, "home_s": 0.0}}
    out = bt._barrier_free("c", plan, plan["steps"], stepwise)
    assert seen["sequences"] == {"robot1": ["W1"], "robot2": ["W2"]}, \
        "the returns home are not coordinated with the work"
    assert seen["tail"] == plan["steps"][1:] and seen["home"] is False
    assert out["program_cost"]["cycle_time_s"] == 69.0, "the tail's time is counted"


def test_the_tail_is_gated_against_where_the_arms_finished(monkeypatch):
    bt = pytest.importorskip("agents.trajectory.tools.behaviour_tree")
    from agents.trajectory.tools import trajectory_cache as tc
    tc.clear_trajectory_cache("cT")
    tc.cache_trajectory("cT", "robot1", "robot1_HOME", _leg([1.0] * 6, [0.0] * 6))
    tc.cache_trajectory("cT", "robot2", "robot2_HOME", _leg([2.0] * 6, [0.0] * 6, [0.0] * 6))
    seen = {}

    def gate(cell_id, legs, allow_parallel=True, delays=None, bystanders=None, **k):
        seen.update(legs=sorted(legs), bystanders=bystanders)
        return {"mode": seen.get("verdict", "PARALLEL"), "order": sorted(legs)}

    monkeypatch.setattr(bt, "_gate_multi_arm_step", gate)
    steps = [{"robots": [{"robot_ns": "robot1", "operation_id": "robot1_HOME", "operation_type": "move"},
                         {"robot_ns": "robot2", "operation_id": "robot2_HOME", "operation_type": "move"}]}]
    nodes, reports, secs = bt._gated_steps("cT", steps, {"robot1": [1.0] * 6, "robot2": [2.0] * 6,
                                                          "robot3": [3.0] * 6})
    assert nodes and secs == 2.0, "parallel: the longer leg"
    assert seen["bystanders"] == {"robot3": [3.0] * 6}, "every other arm is a bystander"
    seen["verdict"] = None
    nodes, reports, secs = bt._gated_steps("cT", steps, {})
    assert nodes is None and reports[-1]["mode"] is None, "a refused step refuses the tail"
    tc.clear_trajectory_cache("cT")


# ── where an assembly's time goes ────────────────────────────────────────────

def test_every_assembly_reports_its_phase_timings(monkeypatch):
    bt = pytest.importorskip("agents.trajectory.tools.behaviour_tree")
    out = bt.assemble_behaviour_tree.func(cell_id="cX", plan={"steps": []})
    assert "timings_s" in out and "total" in out["timings_s"]
    t = bt._AssemblyTimer()
    t.mark("a"); t.step_gate(3, 1.5); t.step_gate(1, 0.2)
    rep = t.report()
    assert set(rep) >= {"a", "total", "slowest_step_gates"}
    assert list(rep["slowest_step_gates"]) == ["step 3", "step 1"]


def test_a_different_backend_implementation_never_reuses_a_step(dispatch, monkeypatch):
    """The key carries the code that would plan: another backend function (or a
    code change) is a miss, so a stored plan cannot stand in for it."""
    pb, calls, _ = dispatch
    pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    _fresh_run()
    other_calls = []

    def other_backend(cell_id, reqs, **kw):
        other_calls.append(1)
        return {"success": False, "trajectories": {}, "planning_time_s": 0.1,
                "failures": [{"robot_ns": "robot1", "reason_code": "no_path_found"}]}

    monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: other_backend)
    out = pb.dispatch_plan(CELL, [dict(r) for r in REQ])
    assert other_calls == [1] and "reused_step" not in out


def test_the_code_fingerprint_names_the_entry_point():
    from agents.trajectory.tools import leg_reuse

    def a(): pass

    def b(): pass

    assert leg_reuse.code_fingerprint(a) != leg_reuse.code_fingerprint(b)
    assert leg_reuse.code_fingerprint(a) == leg_reuse.code_fingerprint(a)
