"""Planning spends its time in the model's turns, so the turns are made fewer and
smaller.

Measured on ``bt_1790857060`` (15 welds, 1,287 s): 1,124 s waiting on the model
over 43 turns, 108 s in tools, 45 s of it OMPL. Two rework loops made most of the
turns — a chain re-plan and three assembler refusals — and every turn re-sent
every waypoint array returned so far.

Pinned here:

* **Chaining is exact.** The planner chains the next leg from the last EMITTED
  sample, not from the IK goal the Cartesian descent only approximates (that
  0.01–0.03 rad gap made chain_trajectories refuse every seam).
* **The assembler's gate runs as each step is planned** (``step_gate``), against
  the arms parked where they last stopped, so a refusal is found while the step
  is fresh.
* **Several steps per call** (``steps=``), fail-fast, with per-operation outcomes.
* **The model sees summaries, not waypoint arrays**, and refers to a leg by its
  operation_id.
* **Sonnet plans by default**; a request's ``model`` still wins; Bedrock prompt
  caching is on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ── the planning model ───────────────────────────────────────────────────────

@pytest.fixture
def lm(monkeypatch):
    mod = pytest.importorskip("agents.llm_model")
    for var in ("LLM_MODEL", "LLM_PROVIDER", "PLANNING_LLM_MODEL", "BEDROCK_MODEL_ID",
                "BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION", "LLM_PROMPT_CACHE"):
        monkeypatch.delenv(var, raising=False)
    return mod


def test_planning_defaults_to_sonnet(lm):
    assert lm.DEFAULT_PLANNING_MODEL_ALIAS == "sonnet-5"
    assert lm.planning_model() == "bedrock:eu.anthropic.claude-sonnet-5"


def test_the_stack_default_does_not_move_planning(lm, monkeypatch):
    """The compose file always sets LLM_MODEL; honouring it would make the
    planning default unreachable."""
    monkeypatch.setenv("LLM_MODEL", "opus-5")
    assert lm.planning_model() == "bedrock:eu.anthropic.claude-sonnet-5"
    assert lm.resolve_model() == "bedrock:eu.anthropic.claude-opus-5"


def test_a_request_model_wins(lm, monkeypatch):
    monkeypatch.setenv("PLANNING_LLM_MODEL", "haiku")
    assert lm.planning_model("opus-5") == "bedrock:eu.anthropic.claude-opus-5"


def test_the_planning_setting_beats_the_default(lm, monkeypatch):
    monkeypatch.setenv("PLANNING_LLM_MODEL", "opus-5")
    assert lm.planning_model() == "bedrock:eu.anthropic.claude-opus-5"


def test_an_empty_planning_setting_means_the_default(lm, monkeypatch):
    """The compose file interpolates ``${PLANNING_LLM_MODEL:-}`` — empty when unset."""
    monkeypatch.setenv("PLANNING_LLM_MODEL", "")
    assert lm.planning_model() == "bedrock:eu.anthropic.claude-sonnet-5"


def test_a_bound_model_wins_over_the_planning_setting(lm, monkeypatch):
    monkeypatch.setenv("PLANNING_LLM_MODEL", "haiku")
    with lm.bind_model("opus-5"):
        assert lm.planning_model() == "bedrock:eu.anthropic.claude-opus-5"


def test_a_resolved_planning_model_binds_to_itself(lm):
    """plan() binds planning_model(model); the binding must round-trip."""
    resolved = lm.planning_model()
    with lm.bind_model(resolved):
        assert lm.resolve_model() == resolved


def test_another_provider_keeps_its_own_default(lm, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert lm.planning_model() == "openai:gpt-4o"


def test_prompt_cache_is_on_by_default(lm, monkeypatch):
    assert lm.prompt_cache_enabled() is True
    monkeypatch.setenv("LLM_PROMPT_CACHE", "0")
    assert lm.prompt_cache_enabled() is False


def test_bedrock_tools_are_bound_with_a_cache_point(lm):
    pytest.importorskip("langchain_aws")
    from langchain_aws import ChatBedrockConverse
    from langchain_core.tools import tool

    @tool
    def ping(x: int) -> int:
        """Echo x."""
        return x

    llm = ChatBedrockConverse(model_id="eu.anthropic.claude-sonnet-5",
                              region_name="eu-west-1")
    bound = lm.bind_tools_for_agent(llm, [ping])
    assert bound.kwargs.get("cache_control") == {"ttl": "5m"}
    assert [t["function"]["name"] for t in bound.kwargs["tools"]] == ["ping"]


def test_discovery_reports_the_planning_model(lm):
    d = lm.describe_models()
    assert d["planning_default"] == "sonnet-5"
    assert d["planning_effective"] == lm.planning_model()


def test_the_compose_files_carry_the_planning_setting():
    import yaml
    env = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())[
        "services"]["trajectory-agent"]["environment"]
    assert env["PLANNING_LLM_MODEL"] == "${PLANNING_LLM_MODEL:-}"
    gen = (PROJECT_ROOT / "scripts/generate_compose.py").read_text()
    assert '"PLANNING_LLM_MODEL": "${PLANNING_LLM_MODEL:-}"' in gen
    mk = (PROJECT_ROOT / "Makefile").read_text()
    assert "export PLANNING_LLM_MODEL" in mk


# ── what the model sees ──────────────────────────────────────────────────────

@pytest.fixture
def agent_mod():
    try:
        import agents.trajectory.agent as A
    except Exception as exc:  # noqa: BLE001 — no LangChain on this host
        pytest.skip(f"agent not importable here: {exc}")
    return A


def _leg(n=5, start=0.0):
    return [{"positions": [start + i * 0.1] * 6, "time_s": i * 0.2, "phase": "transit"}
            for i in range(n)]


def test_waypoint_arrays_reach_the_model_as_summaries(agent_mod):
    result = {"success": True, "trajectories": {"robot1": _leg(5)},
              "steps": [{"trajectories": {"robot2": _leg(4, 1.0)}}],
              "failures": [{"operation_id": "W1"}], "target": [1.0, 2.0, 3.0]}
    view = agent_mod.agent_view(result)
    r1 = view["trajectories"]["robot1"]
    assert r1 == {"waypoints": 5, "duration_s": 0.8,
                  "start_joints": [0.0] * 6, "end_joints": [0.4] * 6}
    assert view["steps"][0]["trajectories"]["robot2"]["waypoints"] == 4
    # Everything that is not a waypoint array is untouched.
    assert view["failures"] == [{"operation_id": "W1"}]
    assert view["target"] == [1.0, 2.0, 3.0]


def test_every_tool_the_model_gets_is_the_summarising_one(agent_mod):
    wrapped = agent_mod._agent_facing(agent_mod.run_trajectory_planner)
    assert wrapped.name == "run_trajectory_planner"
    assert wrapped.args_schema is agent_mod.run_trajectory_planner.args_schema
    assert wrapped.description == agent_mod.run_trajectory_planner.description


def test_every_mode_is_told_to_batch_and_refer_by_id(agent_mod):
    for mode in agent_mod.PLANNING_MODES:
        p = agent_mod.system_prompt_for(None, mode)
        assert "steps=[[" in p and "step_gate" in p
        assert "{robot_ns: operation_id}" in p


def test_the_trace_counts_waypoints_from_a_summary():
    from agents.trajectory import plan_trace
    m = plan_trace._result_metrics(
        {"trajectories": {"robot1": {"waypoints": 71, "duration_s": 2.0}}})
    assert m["waypoints"] == {"robot1": 71}


# ── chaining ─────────────────────────────────────────────────────────────────

def test_the_next_leg_chains_from_the_emitted_end():
    ompl = pytest.importorskip("agents.trajectory.tools.ompl_planner")
    traj = _leg(3)
    assert ompl._trajectory_end(traj, [9.0] * 6) == [0.2] * 6
    assert ompl._trajectory_end([], [9.0] * 6) == [9.0] * 6
    src = (PROJECT_ROOT / "agents/trajectory/backends/ompl/planner.py").read_text()
    assert "set_last_config(cell_id, ns, _trajectory_end(traj, goal_q))" in src
    assert "set_last_config(cell_id, ns, goal_q.tolist())" not in src


# ── referring to a leg by its operation_id ───────────────────────────────────

@pytest.fixture
def pt():
    return pytest.importorskip("agents.trajectory.tools.planning_tools")


def test_validators_resolve_operation_ids_from_the_cache(pt, monkeypatch):
    from agents.trajectory.tools import trajectory_cache as tc
    tc.clear_trajectory_cache("cX")
    tc.cache_trajectory("cX", "robot1", "W1", _leg(4))
    resolved = pt._resolve_trajectories("cX", {"robot1": "W1", "robot2": _leg(3)})
    assert len(resolved["robot1"]) == 4 and len(resolved["robot2"]) == 3
    assert pt._resolve_trajectories("cX", {"robot1": {"operation_id": "W1"}})["robot1"]
    with pytest.raises(ValueError, match="robot1/W9"):
        pt._resolve_trajectories("cX", {"robot1": "W9"})
    # Through the tool: an unknown id is an error, never a pass.
    res = pt.validate_trajectory_set.func(cell_id="cX",
                                          trajectories={"robot1": "W9", "robot2": "W1"})
    assert "error" in res and res.get("collision_free") is not True
    tc.clear_trajectory_cache("cX")


# ── the step gate, as each step is planned ───────────────────────────────────

def test_the_step_gate_checks_against_the_parked_arms(pt, monkeypatch):
    from agents.trajectory.tools import behaviour_tree as bt
    from agents.trajectory.tools import trajectory_cache as tc
    seen = {}

    def gate(cell_id, legs, **kw):
        seen.update(legs=sorted(legs), bystanders=kw.get("bystanders"))
        return {"arms": sorted(legs), "mode": None, "gated": True,
                "bystanders": sorted(kw.get("bystanders") or {}),
                "blocked_arms": sorted(legs), "parallel": {"min_clearance_m": 0.07}}

    monkeypatch.setattr(pt, "_cell_robots", lambda c: {"robot1": {}, "robot2": {}})
    monkeypatch.setattr(tc, "get_last_config",
                        lambda c, ns: [0.5] * 6 if ns == "robot2" else None)
    monkeypatch.setattr(bt, "_gate_multi_arm_step", gate)
    g = pt._step_gate("c", {"robot1": _leg(3)})
    assert seen == {"legs": ["robot1"], "bystanders": {"robot2": [0.5] * 6}}
    assert g["verdict"] == "REFUSED" and g["bystanders"] == ["robot2"]
    assert "REFUSE" in g["advice"] and g["min_clearance_m"] == 0.07


def test_a_lone_arm_with_nobody_parked_is_not_gated(pt, monkeypatch):
    from agents.trajectory.tools import trajectory_cache as tc
    monkeypatch.setattr(pt, "_cell_robots", lambda c: {"robot1": {}, "robot2": {}})
    monkeypatch.setattr(tc, "get_last_config", lambda c, ns: None)
    g = pt._step_gate("c", {"robot1": _leg(3)})
    assert g["checked"] is False and g["verdict"] == "PARALLEL"


def test_the_step_gate_can_be_turned_off(pt, monkeypatch):
    monkeypatch.setenv("PLAN_STEP_GATE", "0")
    assert pt._step_gate("c", {"robot1": _leg(3), "robot2": _leg(3)}) is None


# ── several steps in one call ────────────────────────────────────────────────

def _fake_planner(pt, monkeypatch, fail_ops=(), refuse_step=None):
    from agents.trajectory.tools import planner_backend
    calls = []

    def dispatch(cell_id, reqs, **kw):
        calls.append([r["operation_id"] for r in reqs])
        trajs = {r["robot_ns"]: _leg(3) for r in reqs if r["operation_id"] not in fail_ops}
        fails = [{"operation_id": r["operation_id"], "robot_ns": r["robot_ns"],
                  "reason_code": "no_path"} for r in reqs if r["operation_id"] in fail_ops]
        return {"success": bool(trajs), "trajectories": trajs, "failures": fails,
                "planning_time_s": 0.1, "backend": "ompl"}

    def gate(cell_id, legs):
        step = len(calls) - 1
        return {"verdict": "REFUSED" if step == refuse_step else "PARALLEL",
                "advice": "refused"}

    monkeypatch.setattr(planner_backend, "dispatch_plan", dispatch)
    monkeypatch.setattr(pt, "resolve_velocity_scale", lambda c, v: 0.5)
    monkeypatch.setattr(pt, "cache_planner_result", lambda *a: 0)
    monkeypatch.setattr(pt, "_step_gate", gate)
    return calls


def _step(*ops):
    return [{"robot_ns": f"robot{i + 1}", "operation_id": op, "target": {}}
            for i, op in enumerate(ops)]


def test_a_batch_plans_every_step_in_order(pt, monkeypatch):
    calls = _fake_planner(pt, monkeypatch)
    res = pt.run_trajectory_planner.func(
        cell_id="c", steps=[_step("W1", "W2"), _step("W3", "W4"), _step("W5")])
    assert calls == [["W1", "W2"], ["W3", "W4"], ["W5"]]
    assert res["success"] and res["complete"] and res["stopped_at"] is None
    assert res["planned_steps"] == 3 and res["failures"] == []
    assert all(s["step_gate"]["verdict"] == "PARALLEL" for s in res["steps"])


def test_a_batch_stops_at_the_first_failed_step(pt, monkeypatch):
    calls = _fake_planner(pt, monkeypatch, fail_ops=("W3",))
    res = pt.run_trajectory_planner.func(
        cell_id="c", steps=[_step("W1", "W2"), _step("W3", "W4"), _step("W5", "W6")])
    assert calls == [["W1", "W2"], ["W3", "W4"]]
    assert res["stopped_at"] == 1 and res["planned_steps"] == 1
    assert res["success"] is True and res["complete"] is False
    by_op = {f["operation_id"]: f["reason_code"] for f in res["failures"]}
    assert by_op == {"W3": "no_path", "W5": "not_planned", "W6": "not_planned"}
    assert "steps from 1" in res["advice"]


def test_a_batch_stops_at_a_refused_step(pt, monkeypatch):
    calls = _fake_planner(pt, monkeypatch, refuse_step=0)
    res = pt.run_trajectory_planner.func(cell_id="c",
                                         steps=[_step("W1", "W2"), _step("W3")])
    assert calls == [["W1", "W2"]] and res["stopped_at"] == 0
    by_op = {f["operation_id"]: f["reason_code"] for f in res["failures"]}
    assert by_op == {"W1": "step_gate_refused", "W2": "step_gate_refused",
                     "W3": "not_planned"}


def test_a_single_step_call_is_unchanged(pt, monkeypatch):
    calls = _fake_planner(pt, monkeypatch)
    res = pt.run_trajectory_planner.func(cell_id="c", motion_requests=_step("W1"))
    assert calls == [["W1"]] and "steps" not in res
    assert res["step_gate"]["verdict"] == "PARALLEL"
