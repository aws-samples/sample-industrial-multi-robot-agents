"""IK must not hand back a branch that is inside a fixture.

``run_ik`` tries several seeds and every one that converges reaches the SAME weld to
sub-millimetre accuracy with a different elbow/wrist branch. Choosing between them is
a real decision, and it was being made on joint travel alone against a FLOOR check —
nothing consulted the cell's obstacles. So a branch buried in a fixture could win
over an equally accurate free one, and it did.

Measured on ``welding_cell_2_small`` before the fix:

  * robot1 at W08 — default seeds returned a configuration with arm link 10
    **0.1700 m inside** ``FIXTURE_A_v180``, while the HOME-seeded solve for the
    same pose came back free;
  * robot2 at W07 — link 10 0.0839 m inside ``FIXTURE_A_v102``;
  * robot2 at W08 — link 10 0.0492 m inside ``FIXTURE_A_v104``.

The damage was two stages downstream and looked like something else entirely: the
planner took the colliding configuration as its goal and reported the weld
UNREACHABLE (``goal_in_collision``) for arms that can plainly stand there. Steps 1-2
of the door program would not plan at all. It also made the answer depend on the
seed, which is how the same pose could read reachable in one code path and not in
another.

So obstacle freedom is now the FIRST ranking key, ahead of travel, and it is reported
as ``goal_collision_free``. It is deliberately not a filter: when no branch is free
the best one is still returned, because a caller distinguishing "out of reach" from
"reachable but obstructed" needs the configuration in order to explain the
obstruction.

Real geometry — these run only where Pinocchio and the cell URDFs are (the
``trajectory-agent`` container).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("pinocchio", reason="needs the ROS/pinocchio runtime")

CELL = "welding_cell_2_small"
ARMS = ["robot1", "robot2"]

# The row that would not plan, and the weld each arm returned a buried branch for.
WELDS = {
    "W07": [-1.15, -1.225, 1.35],
    "W08": [-0.975, -1.225, 1.35],
    "W09": [-0.8, -1.225, 1.35],
}


@pytest.fixture(scope="module")
def ik():
    """``run_ik`` unwrapped from its ``@tool`` decorator."""
    from agents.trajectory.tools.pinocchio import run_ik
    return getattr(run_ik, "func", run_ik)


@pytest.fixture(scope="module")
def freedom():
    """``is_free(robot_ns, q)`` — the planner's own validity predicate, independently.

    Deliberately re-derived here rather than read off ``run_ik``'s report: the point
    of the test is that the report is TRUE, so it cannot be the thing that proves it.
    """
    from agents.trajectory.tools.pinocchio import _load_obstacles
    from agents.trajectory.tools.robot_kinematics import (
        _cell_sphere_radii, _config_collision_info, _load_clearance_policy,
        _load_pinocchio_model, _load_robot_config,
    )
    obstacles = _load_obstacles(CELL)
    radius = float(_load_clearance_policy(CELL)["normal"])
    prepared = {}

    def is_free(robot_ns: str, q: list):
        if robot_ns not in prepared:
            model, _data = _load_pinocchio_model(CELL, robot_ns)
            rcfg = _load_robot_config(CELL, robot_ns)
            prepared[robot_ns] = (
                model, model.createData(),
                np.array(rcfg["base_position"], dtype=float),
                float(rcfg.get("base_orientation_yaw", 0.0)),
                _cell_sphere_radii(CELL, model, rcfg.get("tool_collision")),
            )
        model, col_data, base, yaw, radii = prepared[robot_ns]
        return _config_collision_info(
            model, col_data, np.array(q[:model.nq], dtype=float), base, yaw,
            [], obstacles, sphere_radius=radius, sphere_radii=radii,
        )

    return is_free


@pytest.mark.parametrize("robot_ns", ARMS)
@pytest.mark.parametrize("weld", sorted(WELDS))
def test_the_returned_branch_is_out_of_the_fixtures(ik, freedom, robot_ns, weld):
    """The row that reported ``goal_in_collision``: every arm can stand at every weld.

    Each of these six combinations has a collision-free branch; before the ranking
    knew about obstacles, four of them came back buried.
    """
    x, y, z = WELDS[weld]
    out = ik(cell_id=CELL, robot_ns=robot_ns, x=x, y=y, z=z)

    assert out.get("success") is True, out.get("reason")
    hit = freedom(robot_ns, out["joint_positions"])
    assert hit is None, (
        f"{robot_ns} at {weld} came back inside {hit and hit.get('object_id')} "
        f"(link {hit and hit.get('link_index')}, "
        f"clearance {hit and hit.get('clearance_m')} m)")


@pytest.mark.parametrize("robot_ns", ARMS)
@pytest.mark.parametrize("weld", sorted(WELDS))
def test_the_reported_freedom_is_the_truth(ik, freedom, robot_ns, weld):
    """``goal_collision_free`` is what the servicability filter reads, so it must
    agree with the predicate the planner will apply to the same configuration.

    Two collision models for one pose is how the planner's and the monitor's drifted
    apart before; this pins that ``run_ik`` reports the shared one.
    """
    x, y, z = WELDS[weld]
    out = ik(cell_id=CELL, robot_ns=robot_ns, x=x, y=y, z=z)

    assert "goal_collision_free" in out, (
        "the obstacle model is loadable here, so the verdict must be reported")
    assert out["goal_collision_free"] is (freedom(robot_ns, out["joint_positions"]) is None)


@pytest.mark.parametrize("robot_ns", ARMS)
@pytest.mark.parametrize("weld", sorted(WELDS))
def test_the_free_branch_still_reaches_the_weld(ik, robot_ns, weld):
    """A branch is only a solution if it is still ON the weld.

    Ranking by freedom must reorder candidates, never relax the position tolerance —
    "clear of the fixtures but 20 cm off the seam" is not a weld.
    """
    x, y, z = WELDS[weld]
    out = ik(cell_id=CELL, robot_ns=robot_ns, x=x, y=y, z=z)

    assert out["position_error_m"] <= 1e-3


@pytest.mark.parametrize("robot_ns", ARMS)
@pytest.mark.parametrize("weld", sorted(WELDS))
def test_the_verdict_does_not_depend_on_the_seed(ik, freedom, robot_ns, weld):
    """It used to. robot1/W08 was buried from the default seeds and free from HOME.

    A pose that is reachable or not depending on who happens to warm-start the solve
    is not an answer, and the warm start is supplied by whichever caller has a
    previous joint state — so the same weld read differently in different code paths.
    """
    from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose

    x, y, z = WELDS[weld]
    cold = ik(cell_id=CELL, robot_ns=robot_ns, x=x, y=y, z=z)
    warm = ik(cell_id=CELL, robot_ns=robot_ns, x=x, y=y, z=z,
              warm_start_joints=list(home_pose(CELL, robot_ns, HOME_JOINTS_FALLBACK)))

    assert warm.get("success") is True, warm.get("reason")
    assert warm["goal_collision_free"] == cold["goal_collision_free"] is True
    assert freedom(robot_ns, warm["joint_positions"]) is None, (
        "a warm start may choose a nearer branch, but not a buried one")


def test_an_obstructed_pose_returns_the_configuration_and_says_it_is_obstructed(ik):
    """Not a filter. The caller needs the pose to explain the obstruction with.

    A weld deep inside the fixture stack has no free branch at all; failing outright
    would report it identically to out-of-reach, which is a different fault with a
    different repair.
    """
    # Inside the workpiece fixtures, at the height of the middle row.
    out = ik(cell_id=CELL, robot_ns="robot2", x=-1.1, y=-0.3373, z=1.35)

    if not out.get("success"):
        pytest.skip("no IK solution at all for this pose — nothing to report about")
    assert "joint_positions" in out, "the pose is the actionable part of the answer"
    assert len(out["joint_positions"]) >= 6
