"""A plan_trajectories request can direct the run — and the directives are
enforced, not merely suggested.

* ``assignments`` ({op: robot}) are hard locks: written into the operations, put
  back by the allocator if the model drops them, refused by the motion planner
  on any other arm (``assignment_violation``) and by the re-allocator.
* ``skill_priorities`` re-order / switch off skills for that run only, on top of
  the cell's skills.yaml, which is never written.
* ``instructions`` go in front of the model as a binding section.
* Anything unknown (operation, robot, skill, field) refuses the request.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.trajectory import operator_directives as od  # noqa: E402
from utils import skill_registry as sr  # noqa: E402

OPS = [{"id": "W01", "type": "weld", "position": [0, 0, 1]},
       {"id": "W05", "type": "weld", "position": [1, 0, 1]},
       {"id": "W10", "type": "weld", "position": [2, 0, 1], "assigned_robot": "robot1"}]
ROBOTS = ["robot1", "robot2"]


# ── validation ───────────────────────────────────────────────────────────────

def test_good_directives_are_normalised():
    d = od.normalise({"assignments": {"W05": "robot2"},
                      "skill_priorities": {"find-interlocks": 90,
                                           "optimize-sequence": {"enabled": False}},
                      "instructions": "  keep robot1 on the front row \n"},
                     OPS, ROBOTS, "trajectory-classic")
    assert d == {"assignments": {"W05": "robot2"},
                 "skill_priorities": {"find-interlocks": {"priority": 90},
                                      "optimize-sequence": {"enabled": False}},
                 "instructions": "keep robot1 on the front row"}


@pytest.mark.parametrize("directives, match", [
    ({"assignments": {"W99": "robot1"}}, "unknown operation 'W99'"),
    ({"assignments": {"W05": "robot9"}}, "unknown robot 'robot9'"),
    ({"skill_priorities": {"no-such-skill": 50}}, "no-such-skill"),
    ({"skill_priorities": {"find-interlocks": 150}}, "0-100"),
    ({"skill_priorities": {"find-interlocks": {"weight": 3}}}, "unknown field"),
    ({"assignment": {"W05": "robot2"}}, "unknown directive"),
    ({"instructions": 42}, "instructions must be text"),
])
def test_anything_unknown_refuses(directives, match):
    with pytest.raises(od.DirectiveError, match=match):
        od.normalise(directives, OPS, ROBOTS, "trajectory-classic")


def test_a_required_skill_cannot_be_switched_off_for_a_run():
    with pytest.raises(od.DirectiveError, match="required"):
        od.normalise({"skill_priorities": {"validate-collision": {"enabled": False}}},
                     OPS, ROBOTS, "trajectory")


# ── assignments are hard locks ───────────────────────────────────────────────

def test_assignments_become_locks_alongside_the_task_files_own():
    ops = od.merge_assignments(OPS, {"assignments": {"W05": "robot2"}})
    assert ops[1]["assigned_robot"] == "robot2" and "assigned_robot" not in OPS[1]
    assert od.all_locks(ops) == {"W05": "robot2", "W10": "robot1"}


def test_a_lock_covers_the_legs_derived_from_its_operation():
    with od.scope():
        od.set_locks({"W01": "robot1"})
        assert od.locked_robot("W01") == ("W01", "robot1")
        assert od.locked_robot("R2_W01") == ("W01", "robot1")
        assert od.locked_robot("W01_VIA") == ("W01", "robot1")
        assert od.locked_robot("W010") is None
    assert od.current_locks() == {}, "a run's locks must not outlive it"


def test_the_allocator_gets_the_locks_back_if_the_model_drops_them():
    with od.scope():
        od.set_locks({"W05": "robot2"})
        ops = od.apply_locks([{"id": "W05"}, {"id": "W01"}])
    assert ops == [{"id": "W05", "assigned_robot": "robot2"}, {"id": "W01"}]
    src = (ROOT / "agents/trajectory/tools/planning_tools.py").read_text()
    assert "operations = operator_directives.apply_locks(operations)" in src


@pytest.fixture
def pt():
    return pytest.importorskip("agents.trajectory.tools.planning_tools")


def test_the_planner_refuses_a_locked_operation_on_another_arm(pt, monkeypatch):
    from agents.trajectory.tools import planner_backend
    seen = []

    def dispatch(cell_id, reqs, **kw):
        seen.append([(r["robot_ns"], r["operation_id"]) for r in reqs])
        return {"success": True, "failures": [], "planning_time_s": 0.1,
                "trajectories": {r["robot_ns"]: [{"positions": [0] * 6, "time_s": 0},
                                                 {"positions": [0] * 6, "time_s": 1}]
                                 for r in reqs}}

    monkeypatch.setattr(planner_backend, "dispatch_plan", dispatch)
    monkeypatch.setattr(pt, "resolve_velocity_scale", lambda c, v: 0.5)
    monkeypatch.setattr(pt, "cache_planner_result", lambda *a: 0)
    monkeypatch.setattr(pt, "_step_gate", lambda c, legs: None)
    with od.scope():
        od.set_locks({"W05": "robot2"})
        res = pt.run_trajectory_planner.func(cell_id="c", motion_requests=[
            {"robot_ns": "robot1", "operation_id": "W05", "target": {}},
            {"robot_ns": "robot2", "operation_id": "W01", "target": {}}])
        assert seen == [[("robot2", "W01")]], "the locked op must never be solved"
        (f,) = res["failures"]
        assert f["reason_code"] == "assignment_violation" and "robot2" in f["error"]

        only_bad = pt.run_trajectory_planner.func(cell_id="c", motion_requests=[
            {"robot_ns": "robot1", "operation_id": "R1_W05", "target": {}}])
        assert only_bad["success"] is False and len(seen) == 1
        ok = pt.run_trajectory_planner.func(cell_id="c", motion_requests=[
            {"robot_ns": "robot2", "operation_id": "W05", "target": {}}])
        assert ok["success"] and not ok["failures"]


def test_a_locked_operation_cannot_be_reallocated(pt):
    with od.scope():
        od.set_locks({"W05": "robot2"})
        res = pt.reallocate_operation.func(cell_id="c", operation={"id": "W05"},
                                           tried_robots=["robot2"])
    assert res["success"] is False and res["reason_code"] == "robot_locked"
    assert res["assigned_robot"] == "robot2"


# ── per-run skill priorities ─────────────────────────────────────────────────

def test_run_priorities_reorder_and_switch_off_without_writing(monkeypatch):
    monkeypatch.setattr(sr, "_write_overrides",
                        lambda *a, **k: pytest.fail("a run must not write skills.yaml"))
    base = {s["name"]: s for s in sr.list_skills(None, "trajectory-classic")}
    run = sr.normalise_run_overrides("trajectory-classic",
                                     {"find-interlocks": 99,
                                      "optimize-sequence": {"enabled": False}})
    with sr.bind_run_overrides(run):
        skills = sr.list_skills(None, "trajectory-classic", include_disabled=False)
        assert skills[0]["name"] == "find-interlocks" and skills[0]["priority"] == 99
        assert skills[0]["run_override"] is True
        assert "optimize-sequence" not in {s["name"] for s in skills}
        assert sr.get_skill("trajectory-classic", "optimize-sequence")["enabled"] is False
        assert "[priority 99] find-interlocks" in sr.catalog_text("trajectory-classic")
    after = {s["name"]: s for s in sr.list_skills(None, "trajectory-classic")}
    assert after["find-interlocks"]["priority"] == base["find-interlocks"]["priority"]


# ── what the model is told ───────────────────────────────────────────────────

def test_the_model_is_given_the_directives_as_binding():
    text = od.prompt_section(
        {"skill_priorities": {"find-interlocks": {"priority": 90},
                              "optimize-sequence": {"enabled": False}},
         "instructions": "keep robot1 on the front row"},
        {"W05": "robot2"})
    assert text.startswith("OPERATOR DIRECTIVES FOR THIS RUN — binding")
    assert "W05 -> robot2" in text and "ENFORCED" in text
    assert "find-interlocks: priority 90" in text and "optimize-sequence: DISABLED" in text
    assert "keep robot1 on the front row" in text and "say which and why" in text
    assert od.prompt_section({}, {}) == ""


def test_the_run_applies_the_directives_before_building_the_agent():
    src = (ROOT / "agents/trajectory/agent.py").read_text()
    body = src[src.index("def _plan("):]
    assert body.index("operator_directives.set_locks(locks)") \
        < body.index("agent = create_trajectory_agent(")
    assert body.index("skill_registry.set_run_overrides(") \
        < body.index("agent = create_trajectory_agent(")
    assert "operator_directives.prompt_section(directives, locks)" in body
    assert "skill_registry.bind_run_overrides(None)" in src
    assert "operator_directives.scope()" in src


# ── the MCP entry point ──────────────────────────────────────────────────────

@pytest.fixture
def mcp_mod(monkeypatch):
    try:
        import agents.trajectory.mcp_server as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"trajectory MCP not importable here: {exc}")
    from agents.trajectory.tools import planning_tools
    monkeypatch.setattr(planning_tools, "_cell_robots",
                        lambda c: {"robot1": {}, "robot2": {}})
    return m


@pytest.mark.parametrize("kwargs", [
    {"assignments": {"W99": "robot1"}},
    {"assignments": {"W05": "robot7"}},
    {"skill_priorities": {"nope": 10}},
])
def test_the_mcp_call_refuses_bad_directives(mcp_mod, kwargs):
    fn = getattr(mcp_mod.plan_trajectories, "fn", mcp_mod.plan_trajectories)
    res = asyncio.run(fn(cell_id="c", operations=OPS, **kwargs))
    assert res["accepted"] is False and "directives refused" in res["error"]
