"""Re-plan one operation of an existing tree and build a new tree from it.

* The step plan is recovered from the tree's structure (Parallel / Sequence /
  Delay / the final MoveHome step) and every leg's trajectory is reused.
* Only the chosen operation is re-planned — with the run's own motion request
  plus the operator's adjustments — and the arm's following legs only while the
  chain is broken.
* The result goes through assemble_behaviour_tree and is saved as a NEW tree that
  names its source; the source is never touched.
* Anything that cannot be recovered faithfully is refused.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pe = pytest.importorskip("agents.trajectory.tools.plan_edit")


def _leg(start, end, n=3):
    return [{"positions": [start + (end - start) * i / (n - 1)] * 6, "time_s": float(i)}
            for i in range(n)]


def _exec(ns, op, kind="weld"):
    return {"type": "ExecuteTrajectory", "name": f"{kind}_{ns}_{op}",
            "ports": {"robot_ns": ns, "operation_id": op, "operation_type": kind,
                      "trajectory": "{traj_%s_%s}" % (ns, op), "hold_duration_s": 0.5,
                      "velocity_scale": 1.0, "has_interlock": False}}


def _tree():
    bb = {"traj_robot1_A": _leg(0.0, 1.0), "traj_robot2_B": _leg(0.0, 2.0),
          "traj_robot1_C": _leg(1.0, 3.0), "traj_robot2_D": _leg(2.0, 4.0),
          "traj_robot1_E": _leg(3.0, 5.0)}
    root = {"type": "Sequence", "children": [
        {"type": "Sequence", "children": [_exec("robot1", "A"), _exec("robot2", "B")]},
        {"type": "Parallel", "children": [
            {"type": "Delay", "ports": {"delay_msec": 500},
             "children": [_exec("robot1", "C")]},
            _exec("robot2", "D")]},
        _exec("robot1", "E"),
        {"type": "Parallel", "children": [
            {"type": "MoveHome", "ports": {"robot_ns": "robot1"}},
            {"type": "MoveHome", "ports": {"robot_ns": "robot2"}}]},
    ]}
    return {"tree_id": "bt_inner", "root": root, "blackboard": bb, "metadata": {}}


# ── reading a tree ───────────────────────────────────────────────────────────

def test_the_step_plan_is_recovered_from_the_tree():
    plan = pe.read_tree(_tree())
    assert plan["home_at_end"] is True
    assert [s["mode"] for s in plan["steps"]] == ["SEQUENTIAL", "PARALLEL", "SEQUENTIAL"]
    c = plan["steps"][1]["robots"][0]
    assert (c["robot_ns"], c["operation_id"], c["start_delay_s"]) == ("robot1", "C", 0.5)
    assert c["operation_type"] == "weld" and c["hold_duration_s"] == 0.5
    assert [op for op, _t in plan["order"]["robot1"]] == ["A", "C", "E"]
    assert [op for op, _t in plan["order"]["robot2"]] == ["B", "D"]
    assert all("_traj" not in r for s in plan["steps"] for r in s["robots"])


def test_a_summarised_tree_is_refused():
    t = _tree()
    t["blackboard"]["traj_robot1_C"] = {"waypoints": 3, "duration_s": 2.0}
    with pytest.raises(pe.EditError, match="no waypoints"):
        pe.read_tree(t)


def test_a_tree_with_completion_gates_is_refused():
    t = _tree()
    t["root"]["children"].insert(0, {"type": "WaitForRobot",
                                     "ports": {"robot_ns": "robot1"}})
    with pytest.raises(pe.EditError, match="WaitForRobot"):
        pe.read_tree(t)


# ── the run's own requests ───────────────────────────────────────────────────

def test_the_last_successful_request_of_each_leg_is_reused():
    trace = {"decisions": [
        {"tool": "run_trajectory_planner", "result": {"ok": True},
         "args": {"velocity_scale": 0.3, "motion_requests": [
             {"robot_ns": "robot1", "operation_id": "C", "approach_offset_m": 0.1}]}},
        {"tool": "run_trajectory_planner",
         "result": {"ok": True, "failures": [{"operation_id": "C"}]},
         "args": {"motion_requests": [
             {"robot_ns": "robot1", "operation_id": "C", "approach_offset_m": 0.9}]}},
        {"tool": "run_trajectory_planner", "result": {"ok": True},
         "args": {"steps": [[{"robot_ns": "robot2", "operation_id": "D"}]]}},
        {"tool": "validate_approach", "args": {"operation_id": "C"}},
    ]}
    req = pe.requests_from_trace(trace)
    assert req[("robot1", "C")]["approach_offset_m"] == 0.1      # the failed one is skipped
    assert req[("robot1", "C")]["velocity_scale"] == 0.3
    assert ("robot2", "D") in req                                 # batched steps too


def test_the_task_file_is_the_fallback():
    req = pe.request_from_task({"id": "W1", "position": [1, 2, 3],
                                "approach_vector": [0, 0, -1],
                                "weld_orientation": {"x": 0, "y": 1, "z": 0, "w": 0}},
                               "robot2")
    assert req["target"] == {"x": 1, "y": 2, "z": 3} and req["robot_ns"] == "robot2"
    assert req["constrain_orientation"] is True and req["approach"] is True


# ── the edit ─────────────────────────────────────────────────────────────────

@pytest.fixture
def harness(monkeypatch):
    # The planner tools need LangChain, which the `make test` runner image lacks.
    planning_tools = pytest.importorskip("agents.trajectory.tools.planning_tools")
    from agents.trajectory.tools import plan_store, behaviour_tree
    calls = {"plans": [], "assembled": None, "saved": None}
    ends = {}

    def plan(cell_id, motion_requests, velocity_scale=None, **kw):
        (req,) = motion_requests
        calls["plans"].append(dict(req, _vs=velocity_scale))
        end = ends.get(req["operation_id"], 9.0)
        return {"success": True, "failures": [],
                "trajectories": {req["robot_ns"]: _leg(0.0, end)}}

    def assemble(cell_id, plan):
        calls["assembled"] = plan
        return {"success": True, "behaviour_tree": {"tree_id": "bt_new", "root": {},
                                                    "metadata": {}},
                "step_gates": [], "serialised_steps": []}

    monkeypatch.setattr(plan_store, "get_plan", lambda c, t: _tree())
    monkeypatch.setattr(plan_store, "save_plan",
                        lambda c, tree: calls.__setitem__("saved", tree) or tree["tree_id"])
    monkeypatch.setattr(planning_tools, "run_trajectory_planner",
                        types.SimpleNamespace(func=plan))
    monkeypatch.setattr(behaviour_tree, "assemble_behaviour_tree",
                        types.SimpleNamespace(func=assemble))
    monkeypatch.setattr(pe, "_read_trace", lambda c, t: {"decisions": [
        {"tool": "run_trajectory_planner", "result": {"ok": True},
         "args": {"velocity_scale": 0.3, "motion_requests": [
             {"robot_ns": "robot1", "operation_id": op, "approach_offset_m": 0.1}
             for op in ("A", "C", "E")]}}]})
    monkeypatch.setattr(pe, "_task_operations", lambda c: {})
    return calls, ends


def test_only_the_operation_is_replanned_when_its_end_is_unchanged(harness):
    calls, ends = harness
    ends["C"] = 3.0                                    # same end as before
    res = pe.replan_operation("c", "bt_src", "C", {"approach_offset_m": 0.2})
    assert res["success"] and res["tree_id"] == "bt_new" and res["derived_from"] == "bt_src"
    assert [p["operation_id"] for p in calls["plans"]] == ["C"]
    assert calls["plans"][0]["approach_offset_m"] == 0.2       # the adjustment
    assert calls["plans"][0]["_vs"] == 0.3                     # the run's speed
    assert [s["mode"] for s in calls["assembled"]["steps"]] == \
        ["SEQUENTIAL", "PARALLEL", "SEQUENTIAL"]
    assert calls["assembled"]["home_at_end"] is True
    edit = calls["saved"]["metadata"]["edit"]
    assert edit == {"derived_from": "bt_src", "operation_id": "C", "robot_ns": "robot1",
                    "adjustments": {"approach_offset_m": 0.2}, "replanned": ["C"]}


def test_the_following_legs_are_replanned_until_the_chain_closes(harness):
    calls, ends = harness
    ends.update(A=1.7, C=3.0)                          # A moves; C ends as before
    res = pe.replan_operation("c", "bt_src", "A", {"approach_offset_m": 0.2})
    assert [p["operation_id"] for p in calls["plans"]] == ["A", "C"]
    assert "approach_offset_m" in calls["plans"][1] and calls["plans"][1]["approach_offset_m"] == 0.1, \
        "only the chosen operation gets the adjustment"
    assert [r["operation_id"] for r in res["replanned"]] == ["A", "C"]


def test_a_failed_replan_saves_nothing(harness, monkeypatch):
    calls, _ends = harness
    from agents.trajectory.tools import planning_tools
    monkeypatch.setattr(planning_tools, "run_trajectory_planner", types.SimpleNamespace(
        func=lambda **kw: {"success": False, "trajectories": {},
                           "failures": [{"reason_code": "no_path"}]}))
    res = pe.replan_operation("c", "bt_src", "C")
    assert res["success"] is False and res["failures"][0]["reason_code"] == "no_path"
    assert calls["saved"] is None and calls["assembled"] is None


@pytest.mark.parametrize("op, adjustments, match", [
    ("Z", None, "not in bt_src"),
    ("C", {"robot_ns": "robot2"}, "cannot adjust"),
])
def test_bad_edits_are_refused(harness, op, adjustments, match):
    with pytest.raises(pe.EditError, match=match):
        pe.replan_operation("c", "bt_src", op, adjustments)


def test_the_mcp_tool_exists_and_waits_for_running_jobs():
    src = (ROOT / "agents/trajectory/mcp_server.py").read_text()
    body = src[src.index("async def replan_operation("):]
    assert body.index('job.get("status") == "planning"') \
        < body.index("plan_edit.replan_operation")
