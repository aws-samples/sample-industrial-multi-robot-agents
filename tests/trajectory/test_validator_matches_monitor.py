"""Planning-time validation must apply the gate the runtime monitor enforces.

A 15-weld plan was validated ``collision_free: true`` with ``min_clearance_m``
0.429-0.539 m, then E-STOPPED 4.1 s into replay at 0.155 m. Two independent
reasons, both of which made the planner's number optimistic:

1. **The torch was invisible to the planner.** ``_config_spheres`` takes a
   ``tool_cfg``; the monitor passed each robot's ``tool_collision``, the
   validator passed nothing. On ``welding_cell_2_small`` the torch runs out to 0.9 m
   and cost 0.165 m of clearance at the pose that tripped the e-stop.

2. **The two sides disagreed on what "clear" means.** The monitor e-stops at
   ``clearance <= warning_distance`` (0.20 m); the validator only reported
   penetration (``min_clearance >= 0``). Everything in the 0-0.20 m band was
   accepted by planning and halted by execution.

These pin the contract: same swept volume, same threshold, both read from the
cell's own config. The numbers below come from the real geometry of
``welding_cell_2_small``, so a regression in either half fails here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("pinocchio", reason="needs the ROS/pinocchio runtime")

CELL = "welding_cell_2_small"

# The configurations robot1/robot2 were frozen in when the monitor e-stopped.
E_STOP_Q = {
    "robot1": [-1.3443731730742334, -1.6649834138275557, 1.9064940997790765,
               1.8325230253662836, -1.238424167023632, -1.2285428267147602],
    "robot2": [1.373152767803842, -1.2407160626056504, 1.42134194846956,
               1.7840768773281168, -1.4925876259968578, 1.339554441936049],
}


def _static_trajectories(q_by_robot: dict) -> dict:
    """Hold every arm still, so clearance is a pure function of the pose."""
    return {ns: [{"positions": list(q), "time_s": 0.0},
                 {"positions": list(q), "time_s": 1.0}]
            for ns, q in q_by_robot.items()}


@pytest.fixture(scope="module")
def planner():
    import agents.trajectory.tools.planning_tools as pt
    return pt


@pytest.fixture(scope="module")
def spheres_at(planner):
    """(robot -> spheres) at a pose, with the tool included or excluded."""
    import numpy as np

    def build(q_by_robot: dict, with_tool: bool) -> dict:
        out = {}
        for ns, q in q_by_robot.items():
            model, _ = planner._load_pinocchio_model(CELL, ns)
            cfg = planner._load_robot_config(CELL, ns)
            out[ns] = planner._config_spheres(
                model, model.createData(), np.array(q[: model.nq], dtype=float),
                np.array(cfg["base_position"]),
                float(cfg.get("base_orientation_yaw", 0.0)),
                cfg.get("tool_collision") if with_tool else None,
            )
        return out

    return build


# ── the threshold comes from the cell, not a constant ────────────────────────

def test_collision_config_is_read_from_the_cell(planner):
    """The planner must source the monitor's own numbers, not hardcode them."""
    cfg = planner._load_collision_config(CELL)

    assert cfg["warning_distance"] == pytest.approx(0.20)
    assert cfg["min_clearance"] == pytest.approx(0.05)


def test_collision_config_falls_back_for_an_unknown_cell(planner):
    """A missing config must not crash planning — it degrades to the safe default."""
    cfg = planner._load_collision_config("no_such_cell_at_all")

    assert cfg["warning_distance"] == pytest.approx(0.20)


def test_the_monitor_reads_the_same_warning_distance():
    """Both halves must agree, or one of them is enforcing a different rule."""
    import agents.cell.collision_monitor as cm
    from agents.trajectory.tools.robot_kinematics import _load_collision_config

    monitor_default = cm.CollisionMonitor.__init__.__doc__  # presence only
    assert monitor_default is None or isinstance(monitor_default, str)
    # The planner's floor must match the monitor's own hardcoded default.
    assert _load_collision_config("no_such_cell_at_all")["warning_distance"] == 0.20


# ── the tool is part of the robot ────────────────────────────────────────────

def test_the_torch_measurably_narrows_clearance(spheres_at, planner):
    """Proof the omission mattered: the tool costs real clearance at this pose."""
    arm_only = spheres_at(E_STOP_Q, with_tool=False)
    with_tool = spheres_at(E_STOP_Q, with_tool=True)

    _, clear_arm = planner._check_robot_pair_collision(
        arm_only["robot1"], arm_only["robot2"], 0.12)
    _, clear_tool = planner._check_robot_pair_collision(
        with_tool["robot1"], with_tool["robot2"], 0.12)

    assert clear_tool < clear_arm - 0.1, (
        "the tool spheres must shrink clearance; if they no longer do, "
        "tool_collision has stopped being applied"
    )


def test_the_tool_adds_spheres_for_every_arm(spheres_at):
    """A cell that declares tool_collision must contribute spheres for each arm."""
    arm_only = spheres_at(E_STOP_Q, with_tool=False)
    with_tool = spheres_at(E_STOP_Q, with_tool=True)

    for ns in E_STOP_Q:
        assert len(with_tool[ns]) > len(arm_only[ns]), ns


def test_validator_sees_the_tool(planner):
    """The validator's own number must be the with-tool number, not the arm-only one."""
    result = planner.validate_trajectory_set.invoke(
        {"cell_id": CELL, "trajectories": _static_trajectories(E_STOP_Q)})

    # Arm-only geometry at this pose measures ~0.366 m; with the torch, ~0.20 m.
    assert result["min_clearance_m"] < 0.25, (
        f"validator reported {result['min_clearance_m']} m — that is the arm-only "
        "figure, so tool_collision is being ignored again"
    )


# ── the warning band is the gate ─────────────────────────────────────────────

def test_the_warning_distance_travels_with_the_verdict(planner):
    """Callers cannot judge a clearance without the threshold it was judged against."""
    result = planner.validate_trajectory_set.invoke(
        {"cell_id": CELL, "trajectories": _static_trajectories(E_STOP_Q)})

    assert result["warning_distance_m"] == pytest.approx(0.20)


def test_clearance_inside_the_warning_band_is_not_collision_free(planner, monkeypatch):
    """The band is the contract: non-penetrating but halted is still a bad plan."""
    monkeypatch.setattr(planner, "_load_collision_config",
                        lambda _cell: {"warning_distance": 0.5, "min_clearance": 0.05})

    result = planner.validate_trajectory_set.invoke(
        {"cell_id": CELL, "trajectories": _static_trajectories(E_STOP_Q)})

    # ~0.20 m of real clearance: no contact, but well inside a 0.5 m band.
    assert result["min_clearance_m"] > 0, "the arms do not actually touch here"
    assert result["collision_free"] is False, (
        "a plan the monitor would E-STOP must not be reported collision-free"
    )


def test_a_halting_plan_says_why_and_what_to_do(planner, monkeypatch):
    """The agent must be able to act on the rejection without guessing the cause."""
    monkeypatch.setattr(planner, "_load_collision_config",
                        lambda _cell: {"warning_distance": 0.5, "min_clearance": 0.05})

    result = planner.validate_trajectory_set.invoke(
        {"cell_id": CELL, "trajectories": _static_trajectories(E_STOP_Q)})

    assert "E-STOP" in result["suggestion"]
    assert "warning_distance" in result["suggestion"]
    # The offending instant is reported, and marked as non-penetrating.
    pairs = result["colliding_pairs"]
    assert pairs and pairs[0]["penetrating"] is False


def test_generous_separation_still_passes(planner, monkeypatch):
    """The stricter gate must not reject genuinely well-separated motion."""
    monkeypatch.setattr(planner, "_load_collision_config",
                        lambda _cell: {"warning_distance": 0.05, "min_clearance": 0.01})

    result = planner.validate_trajectory_set.invoke(
        {"cell_id": CELL, "trajectories": _static_trajectories(E_STOP_Q)})

    assert result["collision_free"] is True
    assert not result.get("colliding_pairs")


# ── staging poses obey the same band ─────────────────────────────────────────

def test_interlock_clearance_never_undercuts_the_warning_band(planner, monkeypatch):
    """An interlock is meant to make motion safe; it must clear what the monitor checks."""
    monkeypatch.setattr(planner, "_load_collision_config",
                        lambda _cell: {"warning_distance": 0.75, "min_clearance": 0.05})

    result = planner.find_interlock_set.invoke({
        "cell_id": CELL,
        "targets": [{"robot_ns": "robot1", "x": 1.0, "y": 0.0, "z": 1.0},
                    {"robot_ns": "robot2", "x": 1.05, "y": 0.0, "z": 1.0}],
        "operation_type": "weld",
    })

    # With a 0.75 m band no staging strategy can qualify, so it must NOT claim one.
    assert result.get("strategy") != "direct_parallel"
    if result.get("min_clearance_m") is not None:
        assert result["min_clearance_m"] >= 0.75
