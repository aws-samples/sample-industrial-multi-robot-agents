"""A plan request can bring skills of its own — for that run only.

To try a different strategy without shipping a SKILL.md, a request
(plan_trajectories / POST /a2a/plan) may carry ``run_skills``: temporary
skills listed in that run's catalog (marked temporary), loadable with
load_skill, and written nowhere. With ``skill_priorities`` (re-order, switch
off) and ``instructions`` (binding prose) they are how an operator adjusts one
plan to look for a better optimisation.

* A temporary skill cannot reuse an existing skill's name (no shadowing
  validate-collision), cannot be required, and is size-limited.
* Outside its run it does not exist; ``skill_priorities`` may name it.
* ``skill_template`` returns a complete example SKILL.md that validates, the
  fields and limits, and example requests.
* The orchestrator's /a2a/plan passes the directives through; /a2a/skills/
  template and /a2a/skills/validate proxy the planner's tools.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.trajectory import operator_directives as od  # noqa: E402
from utils import skill_registry as sr  # noqa: E402

OPS = [{"id": "W01", "type": "weld", "position": [0, 0, 1]}]
SKILL = {"name": "prefer-short-transits",
         "description": "Use when choosing a via point: prefer the shorter transit.",
         "content": "## Steps\n1. Compare the candidates' transit length.\n",
         "priority": 85}


def test_the_template_is_a_skill_that_validates():
    t = sr.skill_template("trajectory")
    assert t["example_skill_md"].startswith("---\nname: ")
    [skill] = sr.normalise_run_skills("trajectory", [{"skill_md": t["example_skill_md"]}])
    assert skill["name"] == "stagger-shared-boundary" and skill["priority"] == 70
    assert "validate-collision" in t["existing_skills"]
    assert {"temporary_skill", "disable_and_reprioritise", "adjust_the_prompt"} \
        <= set(t["example_requests"])
    with pytest.raises(sr.SkillError):
        sr.skill_template("nobody")


@pytest.mark.parametrize("bad, match", [
    ({**SKILL, "name": "validate-collision"}, "already exists"),
    ({**SKILL, "name": "Bad Name"}, "must match"),
    ({**SKILL, "description": ""}, "description is required"),
    ({**SKILL, "content": "  "}, "no instructions"),
    ({**SKILL, "content": "x" * (sr.RUN_SKILL_CONTENT_MAX + 1)}, "over"),
    ({**SKILL, "priority": 101}, "0-100"),
    ({**SKILL, "extra": 1}, "unknown field"),
    ({"skill_md": "no frontmatter here"}, "frontmatter"),
    ({"skill_md": "---\nname: x-required\ndescription: d\nmetadata:\n  required: true\n"
                  "---\nbody\n"}, "cannot be required"),
])
def test_a_temporary_skill_the_registry_would_not_load_is_refused(bad, match):
    with pytest.raises(sr.SkillError, match=match):
        sr.normalise_run_skills("trajectory", [bad])


def test_too_many_or_duplicated_are_refused():
    many = [{**SKILL, "name": f"s-{i}"} for i in range(sr.RUN_SKILLS_MAX + 1)]
    with pytest.raises(sr.SkillError, match="at most"):
        sr.normalise_run_skills("trajectory", many)
    with pytest.raises(sr.SkillError, match="twice"):
        sr.normalise_run_skills("trajectory", [SKILL, SKILL])


def test_it_is_in_the_runs_catalog_and_loadable_and_nowhere_else(monkeypatch):
    monkeypatch.setattr(sr, "_write_overrides",
                        lambda *a, **k: pytest.fail("a run must not write skills.yaml"))
    skills = sr.normalise_run_skills("trajectory", [SKILL])
    with sr.bind_run_skills({"trajectory": skills}):
        listed = {s["name"]: s for s in sr.list_skills(None, "trajectory",
                                                        include_disabled=False)}
        mine = listed["prefer-short-transits"]
        assert mine["temporary"] and mine["priority"] == 85 and "_content" not in mine
        assert not listed["validate-collision"]["temporary"]
        assert "[priority 85, temporary" in sr.catalog_text("trajectory")
        got = sr.get_skill("trajectory", "prefer-short-transits")
        assert got["content"].startswith("## Steps") and got["path"] is None
        with pytest.raises(sr.SkillError, match="temporary"):
            sr.set_skill_priority("c", "trajectory", "prefer-short-transits", priority=1)
        # The classic skill set does not see it.
        assert "prefer-short-transits" not in {
            s["name"] for s in sr.list_skills(None, "trajectory-classic")}
    assert "prefer-short-transits" not in {s["name"] for s in sr.list_skills(None, "trajectory")}


def test_directives_carry_temporary_skills_and_priorities_may_name_them():
    out = od.normalise({"run_skills": [SKILL],
                        "skill_priorities": {"prefer-short-transits": {"enabled": False},
                                             "optimize-cycle-time": 90}},
                       OPS, ["robot1"], "trajectory")
    assert out["run_skills"][0]["name"] == "prefer-short-transits"
    assert out["skill_priorities"]["prefer-short-transits"] == {"enabled": False}
    text = od.prompt_section(out, {})
    assert "Temporary skills the operator added" in text
    assert "prefer-short-transits (priority 85)" in text
    assert "do not relax any collision check" in text
    with pytest.raises(od.DirectiveError, match="already exists"):
        od.normalise({"run_skills": [{**SKILL, "name": "validate-collision"}]},
                     OPS, ["robot1"], "trajectory")
    with pytest.raises(od.DirectiveError, match="required skill"):
        od.normalise({"skill_priorities": {"validate-collision": {"enabled": False}}},
                     OPS, ["robot1"], "trajectory")


def test_the_run_binds_its_temporary_skills_before_building_the_agent():
    src = (ROOT / "agents/trajectory/agent.py").read_text()
    body = src[src.index("def _plan("):]
    assert body.index("skill_registry.set_run_skills(") \
        < body.index("agent = create_trajectory_agent(")
    assert "skill_registry.bind_run_skills(None)" in src


# ── the A2A face ─────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    try:
        from agents.orchestrator import planner_registry, run_registry, server as srv
    except ImportError as exc:
        pytest.skip(f"orchestrator not importable here: {exc}")
    monkeypatch.setattr(run_registry, "_REGISTRY", run_registry.RunRegistry())
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    srv.app.config["TESTING"] = True
    calls = []

    def fake(url, tool, args, timeout=600.0):
        calls.append((tool, args))
        if tool == "skill_template":
            return {"success": True, **sr.skill_template(args["agent"])}
        if tool == "validate_run_skills":
            try:
                return {"success": True, "valid": True,
                        "skills": sr.normalise_run_skills(args["agent"], args["run_skills"])}
            except sr.SkillError as exc:
                return {"success": False, "valid": False, "error": str(exc)}
        return {}
    monkeypatch.setattr(srv, "_mcp_call", fake)
    with srv.app.test_client() as c:
        yield c, calls, srv, planner_registry


def test_a2a_returns_the_template_and_validates(client):
    c, calls, _srv, _pr = client
    t = c.get("/a2a/skills/template").get_json()
    assert t["success"] and t["example_skill_md"].startswith("---")
    ok = c.post("/a2a/skills/validate", json={"run_skills": [SKILL]})
    assert ok.status_code == 200 and ok.get_json()["valid"]
    bad = c.post("/a2a/skills/validate", json={"run_skills": [{**SKILL, "name": "X"}]})
    assert bad.status_code == 400 and not bad.get_json()["valid"]


def test_a2a_plan_passes_the_directives_to_the_planner(client, monkeypatch):
    c, _calls, srv, planner_registry = client
    sent = {}

    def submit(call, tool, args, timeout):
        sent.update(args)
        return {"accepted": False, "error": "stop here"}
    monkeypatch.setattr(planner_registry, "submit", submit)
    class _Program:
        @staticmethod
        def invoke(args):
            return {"operations": OPS}
    monkeypatch.setattr(srv, "read_task_program", _Program)
    body = {"program": "p", "run_skills": [SKILL], "instructions": "minimise cycle time",
            "skill_priorities": {"optimize-cycle-time": 90},
            "assignments": {"W01": "robot1"}}
    c.post("/a2a/plan", json=body)
    for key in ("run_skills", "instructions", "skill_priorities", "assignments"):
        assert sent.get(key) == body[key], key
