"""The home reset is real motion and must be planned like any other.

``replay_plan(home_reset=True)`` drove every arm home with a bare ``MoveHome``
node. That node carries no path, so the executor falls back to dispatching a
single HOME waypoint and the arm interpolates straight there — through the
fixtures, the workpiece, or the other arm. Done for all arms in Parallel, it was
the least-checked motion in a replay, and it ran BEFORE the carefully validated
plan.

Two things are pinned here:

1. Each ``MoveHome`` in the reset tree carries a planned trajectory.
2. The arms are planned TOGETHER, so each is routed around the others' swept
   volume — two individually-valid straight lines can still intersect.

Degradation stays allowed: an arm that cannot be planned keeps the plain node and
the blind fallback, which is no worse than the previous behaviour. What must not
happen is silence — the caller is told which arms are unplanned.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"
ROBOTS = ["robot1", "robot2"]


def _traj(n: int, start: float = 0.0) -> list:
    """A trivially valid n-point trajectory."""
    return [{"positions": [start + i * 0.01] * 6, "time_s": i * 0.1} for i in range(n)]


@pytest.fixture
def home_reset():
    """The tree builder.

    Imported from ``agents.cell.home_reset``, NOT ``mcp_server``: the latter pulls
    in the LLM agent (and LangGraph behind it), which is both slow and enough to
    disturb the module stubs other cell tests install.
    """
    return pytest.importorskip(
        "agents.cell.home_reset", reason="needs the cell runtime")


@pytest.fixture
def fake_planning(home_reset, monkeypatch):
    """Drive home_reset_tree with stubbed state + planner; capture what it asked for."""
    import agents.trajectory.tools.ompl_planner as ompl

    calls: dict = {}

    def install(states: dict, planned, raises: bool = False):
        # A StructuredTool is a frozen pydantic model, so replace the whole tool
        # rather than patching .invoke on it.
        monkeypatch.setattr(
            home_reset, "get_all_robot_states",
            SimpleNamespace(invoke=lambda _args: {"robots": states}))

        def fake_plan(cell_id, starts, home_by_robot=None, velocity_scale=0.4):
            calls["cell_id"] = cell_id
            calls["starts"] = starts
            if raises:
                raise RuntimeError("OMPL unavailable")
            return planned(starts) if callable(planned) else planned

        monkeypatch.setattr(ompl, "plan_home_trajectories", fake_plan)
        # The legs and states are synthetic joint values (inside the floor of the
        # real cell): these tests are about the reset tree, so the step gate's
        # static check reports the legs clear.
        import importlib
        pt = importlib.import_module("agents.trajectory.tools.planning_tools")
        monkeypatch.setattr(pt, "_static_collisions",
                            lambda cell_id, trajectories: {"checked": True, "collisions": []})
        return calls

    return install


def _nodes(tree: dict) -> dict:
    return {c["ports"]["robot_ns"]: c for c in tree["root"]["children"]}


# ── the reset carries planned paths ──────────────────────────────────────────

def test_every_arm_gets_its_planned_trajectory(home_reset, fake_planning):
    """The whole point: no arm goes home on an unchecked straight line."""
    fake_planning(
        states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
        planned={"robot1": _traj(20), "robot2": _traj(25)},
    )

    nodes = _nodes(home_reset.home_reset_tree(CELL, ROBOTS))

    assert len(nodes["robot1"]["ports"]["trajectory"]) == 20
    assert len(nodes["robot2"]["ports"]["trajectory"]) == 25


def test_the_planner_is_given_each_arms_actual_pose(home_reset, fake_planning):
    """Planning from a guessed start would produce a path the arm cannot follow."""
    calls = fake_planning(
        states={"robot1": {"joint_positions": [0.3] * 6},
                "robot2": {"joint_positions": [-0.7] * 6}},
        planned={},
    )

    home_reset.home_reset_tree(CELL, ROBOTS)

    assert calls["starts"] == {"robot1": [0.3] * 6, "robot2": [-0.7] * 6}
    assert calls["cell_id"] == CELL


def test_all_arms_are_planned_in_one_coordinated_call(home_reset, fake_planning):
    """One call with every arm is what enables inter-robot coupling.

    Planning each arm in a separate call cannot route it around the others.
    """
    calls = fake_planning(
        states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
        planned={},
    )

    home_reset.home_reset_tree(CELL, ROBOTS)

    assert sorted(calls["starts"]) == ROBOTS, (
        "every arm must be in a single plan_home_trajectories call, or the "
        "returns are not coupled to each other"
    )


def test_the_tree_still_drives_the_arms_in_parallel(home_reset, fake_planning):
    """Coupling is what makes Parallel safe — it must not become a Sequence."""
    fake_planning(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
                  planned={r: _traj(10) for r in ROBOTS})

    tree = home_reset.home_reset_tree(CELL, ROBOTS)

    assert tree["root"]["type"] == "Parallel"
    assert len(tree["root"]["children"]) == len(ROBOTS)
    assert all(c["type"] == "MoveHome" for c in tree["root"]["children"])


# ── degradation is allowed, but never silent ──────────────────────────────────

def test_an_unplannable_arm_keeps_the_blind_fallback(home_reset, fake_planning):
    """Never lose a replay to a failed home leg — just degrade that one arm."""
    fake_planning(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
                  planned={"robot1": _traj(12)})

    nodes = _nodes(home_reset.home_reset_tree(CELL, ROBOTS))

    assert nodes["robot1"]["ports"]["trajectory"]
    assert "trajectory" not in nodes["robot2"]["ports"], (
        "an unplanned arm must carry NO trajectory port, so the executor uses "
        "its documented fallback rather than an empty path"
    )


def test_unplanned_arms_are_reported(home_reset, fake_planning, capsys):
    """An operator must be able to tell a planned reset from a blind one."""
    fake_planning(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
                  planned={"robot1": _traj(12)})

    home_reset.home_reset_tree(CELL, ROBOTS)

    out = capsys.readouterr().out
    assert "robot2" in out and "blind" in out.lower()


def test_a_planner_failure_degrades_the_whole_reset_without_raising(
        home_reset, fake_planning, capsys):
    """No OMPL / no Pinocchio must not break replay entirely."""
    fake_planning(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
                  planned=None, raises=True)

    tree = home_reset.home_reset_tree(CELL, ROBOTS)
    nodes = _nodes(tree)

    assert len(nodes) == len(ROBOTS)
    assert all("trajectory" not in n["ports"] for n in nodes.values())
    assert "fall" in capsys.readouterr().out.lower()


def test_missing_live_state_does_not_invent_a_start_pose(home_reset, fake_planning):
    """Planning an arm whose pose is unknown would be planning from a guess."""
    calls = fake_planning(
        states={"robot1": {"joint_positions": [0.2] * 6}, "robot2": {}},
        planned=lambda starts: {ns: _traj(8) for ns in starts},
    )

    nodes = _nodes(home_reset.home_reset_tree(CELL, ROBOTS))

    assert list(calls["starts"]) == ["robot1"]
    assert nodes["robot1"]["ports"].get("trajectory")
    assert "trajectory" not in nodes["robot2"]["ports"]


# ── HOME is the cell's declared pose ─────────────────────────────────────────

def test_home_comes_from_the_cells_own_config():
    """welding_cell_2_small declares home_pose_deg [0,-90,90,0,0,0]; planning must use it.

    The module default is a DIFFERENT pose ([0,-1.57,2.2,0,-0.8,0]), so planning
    to the default would park the arm somewhere the cell does not consider home.
    """
    import math

    _home_config = pytest.importorskip(
        "agents.trajectory.tools.ompl_planner",
        reason="needs the trajectory runtime")._home_config

    q = _home_config(CELL, "robot1", (0.0, -1.57, 2.2, 0.0, -0.8, 0.0))

    assert q == pytest.approx([0.0, math.radians(-90), math.radians(90), 0.0, 0.0, 0.0])


def test_an_undeclared_home_falls_back_to_the_executors_pose():
    """When a cell declares nothing, planned and blind home must agree."""
    _home_config = pytest.importorskip(
        "agents.trajectory.tools.ompl_planner",
        reason="needs the trajectory runtime")._home_config
    from agents.cell.bt_executor import HOME_JOINTS

    q = _home_config("no_such_cell", "robot1", HOME_JOINTS)

    assert q == pytest.approx(list(HOME_JOINTS))
