"""Approach/retreat symmetry, wrist-branch selection, and planned return-to-HOME.

Covers three motion-quality rules that had no guard:

  1. A weld's trajectory ends ON the seam, so the NEXT motion must first retract
     along the SAME approach vector at the SAME orientation. The working pose is
     handed forward through the trajectory cache (``set_last_pose``/``get_last_pose``).
  2. IK must return the wrist branch NEAREST the arm's current pose. Two configs a
     full turn apart are the same tool pose but interpolating between them spins
     the tool — the "strange rotation" symptom.
  3. The return to HOME must be planned, not a blind single waypoint: the HOME node
     carries a ``precomputed_trajectory`` that ``_exec_home`` prefers.

Import-safe: exercises the pure helpers (no ROS, no OMPL, no Pinocchio needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools.pinocchio import _joint_travel, _nearest_branch  # noqa: E402
from agents.trajectory.tools.trajectory_cache import (  # noqa: E402
    clear_last_pose,
    get_last_pose,
    reset_robot_configs,
    set_last_config,
    set_last_pose,
)

TWO_PI = 2.0 * np.pi
ORI = {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_robot_configs()
    yield
    reset_robot_configs()


# ── 1. working-pose hand-off (approach/retreat symmetry) ──────────────────────

def test_working_pose_round_trips():
    """The next operation can read back the exact axis and orientation to leave on."""
    set_last_pose("cellA", "robot1", [1.0, 2.0, 3.0], [0.0, 0.0, -1.0], ORI)

    pose = get_last_pose("cellA", "robot1")

    assert pose["target"] == [1.0, 2.0, 3.0]
    assert pose["approach_vector"] == [0.0, 0.0, -1.0]
    assert pose["orientation"] == ORI


def test_retreat_point_is_the_mirror_of_the_approach_point():
    """Retreat standoff == approach standoff: entry and exit are the same stroke.

    The planner descends from ``target - av*offset`` into ``target`` and retreats
    from ``target`` back to ``target - av*offset``. Same two points, reversed —
    that is what makes approach and leave symmetric.
    """
    target = np.array([1.0, 2.0, 3.0])
    av = np.array([0.0, 0.0, -1.0])
    offset = 0.05

    approach_from = target - av * offset
    set_last_pose("cellA", "robot1", target.tolist(), av.tolist(), ORI)

    pose = get_last_pose("cellA", "robot1")
    retreat_av = np.asarray(pose["approach_vector"], float)
    retreat_av /= np.linalg.norm(retreat_av)
    retreat_to = np.asarray(pose["target"], float) - retreat_av * offset

    assert np.allclose(retreat_to, approach_from)
    # Above the seam for a top-down (-Z) weld — the tool lifts off, never digs in.
    assert retreat_to[2] > target[2]


def test_pose_is_isolated_per_robot():
    """robot2 must not retract along robot1's axis."""
    set_last_pose("cellA", "robot1", [1, 0, 0], [0, 0, -1], ORI)
    set_last_pose("cellA", "robot2", [0, 1, 0], [0, -1, 0], ORI)

    assert get_last_pose("cellA", "robot1")["approach_vector"] == [0.0, 0.0, -1.0]
    assert get_last_pose("cellA", "robot2")["approach_vector"] == [0.0, -1.0, 0.0]


def test_unconstrained_move_clears_the_pose():
    """A plain move has no seam to leave, so no retreat may be inherited from it."""
    set_last_pose("cellA", "robot1", [1, 0, 0], [0, 0, -1], ORI)

    clear_last_pose("cellA", "robot1")

    assert get_last_pose("cellA", "robot1") is None


def test_missing_axis_does_not_half_write_a_pose():
    """No approach vector => nothing to retract along; the entry is cleared."""
    set_last_pose("cellA", "robot1", [1, 0, 0], [0, 0, -1], ORI)

    set_last_pose("cellA", "robot1", [1, 0, 0], None, ORI)

    assert get_last_pose("cellA", "robot1") is None


def test_session_reset_forgets_stale_working_poses():
    """A fresh planning session must not retract along the previous run's weld."""
    set_last_config("cellA", "robot1", [0.0] * 6)
    set_last_pose("cellA", "robot1", [1, 0, 0], [0, 0, -1], ORI)
    set_last_pose("cellB", "robot1", [2, 0, 0], [0, 0, -1], ORI)

    reset_robot_configs("cellA")

    assert get_last_pose("cellA", "robot1") is None
    assert get_last_pose("cellB", "robot1") is not None, "other cells untouched"


# ── 2. wrist branch selection (strange tool rotation) ─────────────────────────

JL = np.array([-6.3] * 6)
JU = np.array([6.3] * 6)


def test_full_turn_branch_is_pulled_back_to_the_reference():
    """A solution a full turn away from the current pose is re-expressed nearby."""
    ref = [0.0, -1.0, 1.5, 0.0, -0.5, 0.1]
    far = [0.0, -1.0, 1.5, 0.0, -0.5, 0.1 + TWO_PI]   # identical tool pose

    fixed = _nearest_branch(far, ref, JL, JU)

    assert fixed[5] == pytest.approx(0.1)
    assert _joint_travel(fixed, ref) < 1e-9


def test_negative_branch_is_also_corrected():
    ref = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    far = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0 - TWO_PI]

    fixed = _nearest_branch(far, ref, JL, JU)

    assert fixed[5] == pytest.approx(3.0)


def test_joint_limits_are_never_violated_to_shorten_travel():
    """Smoothness must not be bought with an out-of-limit joint value."""
    jl = np.array([-3.2] * 6)
    ju = np.array([3.2] * 6)
    ref = [0.0] * 6
    # +2.0 is in limits; its nearer-to-ref branch (2.0 - 2π = -4.28) is NOT.
    joints = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]

    fixed = _nearest_branch(joints, ref, jl, ju)

    assert fixed[5] == pytest.approx(2.0)
    assert all(jl[i] <= fixed[i] <= ju[i] for i in range(6))


def test_already_nearest_branch_is_left_alone():
    ref = [0.1, -1.0, 1.5, 0.0, -0.5, 0.2]
    joints = [0.15, -1.02, 1.48, 0.01, -0.49, 0.22]

    assert _nearest_branch(joints, ref, JL, JU) == pytest.approx(joints)


def test_no_reference_pose_is_a_no_op():
    """Cold start (no warm start) must not invent a branch shift."""
    joints = [0.0, -1.0, 1.5, 0.0, -0.5, 5.0]

    assert _nearest_branch(joints, None, JL, JU) == joints
    assert _joint_travel(joints, None) == 0.0


def test_travel_is_the_worst_joint_not_the_sum():
    """A single spinning wrist must dominate the score, not be averaged away."""
    ref = [0.0] * 6
    joints = [0.05, 0.05, 0.05, 0.05, 0.05, 3.0]

    assert _joint_travel(joints, ref) == pytest.approx(3.0)


# ── 3. planned return-to-HOME ────────────────────────────────────────────────

def _behaviour_tree_module():
    """Import behaviour_tree against the REAL langchain tool decorator.

    Other test modules install a stub ``langchain_core.tools`` into
    ``sys.modules`` that lacks the ``.func`` attribute LangChain's ``@tool``
    provides. Whichever module imports first wins for the whole session, so this
    drops any stub and re-imports so these tests exercise the real decorator
    regardless of test order.
    """
    import importlib

    stubbed = [
        n for n in list(sys.modules)
        if n.startswith("langchain") and getattr(sys.modules[n], "__file__", None) is None
    ]
    if stubbed:
        for n in stubbed:
            del sys.modules[n]
        for n in [n for n in list(sys.modules) if n.startswith("agents.trajectory.tools")]:
            del sys.modules[n]
    return importlib.import_module("agents.trajectory.tools.behaviour_tree")


def test_home_node_carries_a_planned_trajectory(monkeypatch):
    """With a planner available, the HOME node must ship a precomputed path."""
    bt = _behaviour_tree_module()
    from agents.trajectory.tools.trajectory_cache import cache_trajectory

    cache_trajectory("cellA", "robot1", "W15",
                     [{"positions": [0.1] * 6, "time_s": 0.0},
                      {"positions": [0.2] * 6, "time_s": 1.0}])

    planned = [{"positions": [0.2] * 6, "time_s": 0.0},
               {"positions": bt._HOME_JOINTS, "time_s": 4.0}]
    seen = {}

    def fake_plan(cell_id, robot_ns, start_q, home_q, *a, **kw):
        seen.update(start_q=list(start_q), home_q=list(home_q))
        return planned

    import agents.trajectory.backends.ompl.planner as op
    monkeypatch.setattr(op, "plan_home_trajectory", fake_plan, raising=False)

    node = bt._planned_home_node("cellA", "robot1", ["W15"])

    assert node["type"] == "HOME"
    assert node["precomputed_trajectory"] == planned
    # Planned FROM where the arm actually is (end of its last weld) TO home.
    assert seen["start_q"] == [0.2] * 6
    assert seen["home_q"] == bt._HOME_JOINTS


def test_home_node_degrades_when_planning_is_unavailable(monkeypatch):
    """A failed home plan must not lose the tree — the executor falls back."""
    bt = _behaviour_tree_module()
    from agents.trajectory.tools.trajectory_cache import cache_trajectory

    cache_trajectory("cellA", "robot1", "W15", [{"positions": [0.2] * 6, "time_s": 1.0}])

    import agents.trajectory.backends.ompl.planner as op
    monkeypatch.setattr(op, "plan_home_trajectory",
                        lambda *a, **kw: None, raising=False)

    node = bt._planned_home_node("cellA", "robot1", ["W15"])

    assert node["type"] == "HOME"
    assert "precomputed_trajectory" not in node


def test_home_node_without_any_cached_operation_stays_plain():
    """Nothing to plan from (no ops executed) => plain node, no crash."""
    bt = _behaviour_tree_module()

    node = bt._planned_home_node("cellA", "robot1", [])

    assert node["type"] == "HOME"
    assert "precomputed_trajectory" not in node


def test_home_joints_match_the_executor():
    """The planned goal must be the pose the executor/controllers call HOME."""
    bt = _behaviour_tree_module()

    exec_home = [0.0, -1.57, 2.2, 0.0, -0.8, 0.0]
    assert bt._HOME_JOINTS == exec_home
