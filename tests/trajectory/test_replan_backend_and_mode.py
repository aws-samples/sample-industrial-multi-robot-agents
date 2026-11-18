"""Re-planning one operation names its backend and its mode.

* backend ``ompl-compound``: the operation's PARALLEL step is re-planned in ONE
  call with every arm of that step, timed as one; ``ompl``: the leg alone.
* mode: ``classic`` / ``stepwise`` rebuild the steps; ``coordinated`` builds the
  per-arm gated tree — from a step tree or a coordinated one.

The unit grouping is host-runnable; the edits need the planner tools
(LangChain), i.e. the container.
"""
from __future__ import annotations

import types

import pytest

pe = pytest.importorskip("agents.trajectory.tools.plan_edit")
from tests.trajectory.test_plan_edit import _exec, _leg, _tree  # noqa: E402


class TestUnits:
    ORDER = {"robot1": [("A", 0), ("C", 0), ("E", 0)], "robot2": [("B", 0), ("D", 0)]}
    STEPS = [
        {"mode": "SEQUENTIAL", "robots": [{"robot_ns": "robot1"}, {"robot_ns": "robot2"}]},
        {"mode": "PARALLEL", "robots": [{"robot_ns": "robot1"}, {"robot_ns": "robot2"}]},
        {"mode": "SEQUENTIAL", "robots": [{"robot_ns": "robot1"}]},
    ]

    def test_joint_groups_a_parallel_step(self):
        units = pe.planning_units(self.ORDER, self.STEPS, joint=True)
        assert units == [[("robot1", 0)], [("robot2", 0)],
                         [("robot1", 1), ("robot2", 1)], [("robot1", 2)]]

    def test_otherwise_every_leg_is_its_own_unit(self):
        units = pe.planning_units(self.ORDER, self.STEPS, joint=False)
        assert all(len(u) == 1 for u in units) and len(units) == 5

    def test_a_coordinated_tree_has_no_steps(self):
        units = pe.planning_units(self.ORDER, None, joint=True)
        assert units == [[("robot1", 0)], [("robot1", 1)], [("robot1", 2)],
                         [("robot2", 0)], [("robot2", 1)]]


def _coordinated_tree():
    """robot1: A, then C split around a gate; robot2: B, D; and a home step."""
    bb = {"traj_robot1_A": _leg(0.0, 1.0), "traj_robot1_C@0-0": _leg(1.0, 2.0),
          "traj_robot1_C@1-1": _leg(2.0, 3.0), "traj_robot2_B": _leg(0.0, 2.0),
          "traj_robot2_D": _leg(2.0, 4.0)}
    c0, c1 = _exec("robot1", "C@0-0"), _exec("robot1", "C@1-1")
    c0["ports"]["hold_duration_s"] = 0.0
    arm1 = {"type": "Sequence", "children": [
        _exec("robot1", "A"), c0,
        {"type": "WaitForRobot", "ports": {"robot_ns": "robot1"}}, c1]}
    arm2 = {"type": "Sequence", "children": [_exec("robot2", "B"), _exec("robot2", "D")]}
    root = {"type": "Sequence", "children": [
        {"type": "Parallel", "children": [arm1, arm2]},
        {"type": "MoveHome", "ports": {"robot_ns": "robot1"}}]}
    return {"tree_id": "bt_coord", "root": root, "blackboard": bb,
            "metadata": {"mode": "coordinated"}}


class TestReadCoordinated:
    def test_the_mirrored_constants_match_the_assembler(self):
        ct = pytest.importorskip("agents.trajectory.tools.coordinated_tree")
        assert (pe.COORD_SEP, pe.COORD_EARLY_HOME) == (ct.SEP, ct.EARLY_HOME)

    def test_pieces_are_joined_and_gates_dropped(self):
        src = pe.read_coordinated_tree(_coordinated_tree())
        assert [op for op, _t in src["order"]["robot1"]] == ["A", "C"]
        c = dict(src["order"]["robot1"])["C"]
        assert [p["positions"][0] for p in c] == [1.0, 1.5, 2.0, 2.5, 3.0]
        times = [p["time_s"] for p in c]
        assert times == sorted(times) and len(set(times)) == len(times)
        assert src["holds"]["C"] == 0.5 and src["home_at_end"] is True


@pytest.fixture
def harness(monkeypatch):
    planning_tools = pytest.importorskip("agents.trajectory.tools.planning_tools")
    from agents.trajectory.tools import (
        behaviour_tree, coordinated_tree, plan_store, planner_backend,
    )
    calls = {"plans": [], "steps": None, "coordinated": None, "saved": None,
             "backend": [], "tree": _tree()}

    def plan(cell_id, motion_requests, velocity_scale=None, **kw):
        calls["plans"].append(([r["operation_id"] for r in motion_requests], velocity_scale))
        calls["backend"].append(planner_backend.current_planning_backend())
        old = {"C": 3.0, "D": 4.0, "A": 1.0, "B": 2.0, "E": 5.0}
        return {"success": True, "failures": [],
                "trajectories": {r["robot_ns"]: _leg(0.0, old[r["operation_id"]])
                                 for r in motion_requests}}

    def assemble(cell_id, plan):
        calls["steps"] = plan
        return {"success": True, "behaviour_tree": {"tree_id": "bt_new", "root": {},
                                                    "metadata": {}}}

    def assemble_coord(cell_id, sequences, hold_duration_s=None,
                       operation_types=None, home_at_end=True):
        calls["coordinated"] = dict(sequences=sequences, holds=hold_duration_s,
                                    types=operation_types, home=home_at_end)
        return {"success": True, "behaviour_tree": {"tree_id": "bt_coord_new",
                                                    "root": {}, "metadata": {}}}

    monkeypatch.setattr(plan_store, "get_plan", lambda c, t: calls["tree"])
    monkeypatch.setattr(plan_store, "save_plan",
                        lambda c, tree: calls.__setitem__("saved", tree) or tree["tree_id"])
    monkeypatch.setattr(planning_tools, "run_trajectory_planner",
                        types.SimpleNamespace(func=plan))
    monkeypatch.setattr(behaviour_tree, "assemble_behaviour_tree",
                        types.SimpleNamespace(func=assemble))
    monkeypatch.setattr(coordinated_tree, "assemble_coordinated_tree",
                        types.SimpleNamespace(func=assemble_coord))
    monkeypatch.setattr(pe, "_read_trace", lambda c, t: {"decisions": [
        {"tool": "run_trajectory_planner", "result": {"ok": True},
         "args": {"motion_requests": [
             {"robot_ns": ns, "operation_id": op, "velocity_scale": vs}
             for ns, op, vs in (("robot1", "A", 0.5), ("robot1", "C", 0.6),
                                ("robot1", "E", 0.5), ("robot2", "B", 0.5),
                                ("robot2", "D", 0.4))]}}]})
    monkeypatch.setattr(pe, "_task_operations", lambda c: {})
    return calls


class TestBackend:
    def test_compound_replans_the_whole_parallel_step_in_one_call(self, harness):
        res = pe.replan_operation("c", "bt_src", "C", {"approach_offset_m": 0.2},
                                  backend="ompl-compound", mode="classic")
        assert res["success"]
        assert harness["plans"] == [(["C", "D"], 0.4)]       # joint, one speed (the slower)
        assert harness["backend"] == ["ompl-compound"]
        assert {r["robot_ns"] for r in res["replanned"]} == {"robot1", "robot2"}

    def test_ompl_replans_the_leg_alone(self, harness):
        pe.replan_operation("c", "bt_src", "C", backend="ompl", mode="classic")
        assert harness["plans"] == [(["C"], 0.6)]
        assert harness["backend"] == ["ompl"]

    def test_a_sequential_step_stays_single_even_with_compound(self, harness):
        pe.replan_operation("c", "bt_src", "A", backend="ompl-compound", mode="classic")
        assert harness["plans"] == [(["A"], 0.5)]

    def test_the_choice_is_recorded_on_the_new_tree(self, harness):
        pe.replan_operation("c", "bt_src", "C", backend="ompl-compound", mode="stepwise")
        assert harness["saved"]["metadata"]["edit_planning"] == {
            "backend": "ompl-compound", "mode": "stepwise"}


class TestMode:
    def test_classic_rebuilds_the_steps(self, harness):
        pe.replan_operation("c", "bt_src", "C", mode="classic")
        assert harness["steps"] is not None and harness["coordinated"] is None

    def test_coordinated_builds_per_arm_sequences_from_a_step_tree(self, harness):
        res = pe.replan_operation("c", "bt_src", "C", mode="coordinated")
        assert res["tree_id"] == "bt_coord_new" and harness["steps"] is None
        co = harness["coordinated"]
        assert co["sequences"] == {"robot1": ["A", "C", "E"], "robot2": ["B", "D"]}
        assert co["holds"]["C"] == 0.5 and co["types"]["D"] == "weld" and co["home"] is True

    def test_a_coordinated_tree_can_be_edited_coordinated(self, harness):
        harness["tree"] = _coordinated_tree()
        res = pe.replan_operation("c", "bt_coord", "C", mode="coordinated",
                                  backend="ompl-compound")
        assert res["success"]
        assert harness["plans"] == [(["C"], 0.6)]          # no steps: one arm
        assert harness["coordinated"]["sequences"] == {"robot1": ["A", "C"],
                                                       "robot2": ["B", "D"]}

    def test_a_coordinated_tree_cannot_become_a_step_tree(self, harness):
        harness["tree"] = _coordinated_tree()
        with pytest.raises(pe.EditError, match="mode='coordinated'"):
            pe.replan_operation("c", "bt_coord", "C", mode="classic")


def test_the_mcp_tools_take_backend_and_mode():
    from pathlib import Path
    src = (Path(pe.__file__).parents[1] / "mcp_server.py").read_text()
    for tool in ("async def plan_trajectories(", "async def replan_operation("):
        sig = src[src.index(tool):src.index(") -> dict:", src.index(tool))]
        assert "backend: Optional[str] = None" in sig and "mode: Optional[str] = None" in sig
    assert src.count("plan_selection.select(") >= 2
