"""A leg that starts on a weld lifts off on-axis — pinned ``current_joints`` or not.

The planner used to skip the on-axis lift-off whenever the request pinned
``current_joints`` ("an intentional re-plan"), and otherwise lifted off the
robot's MOST RECENT weld only. But a chain repair, a coupled re-plan of a
PARALLEL step and an edit of one weld all pin the start at a weld's end, so those
legs dragged the tool out of the seam sideways (umarell issue 05: 34 mm below the
weld plane), and an out-of-order re-plan lifted off the wrong weld (issue 15).

``robot_kinematics.seam_at_start`` decides by where the TCP IS: every weld of the
run is recorded (``trajectory_cache.set_weld_pose``) and a start on a weld's
approach axis — at the seam or part-way up the stroke — lifts off along it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

np = pytest.importorskip("numpy")


class _Live:
    """The modules as imported NOW — other tests reload them, and a reference
    taken at collection time would then point at a copy nobody reads."""

    def __getattr__(self, name):
        import importlib
        mod = {"rk": "agents.trajectory.tools.robot_kinematics",
               "tc": "agents.trajectory.tools.trajectory_cache"}[name]
        return importlib.import_module(mod)


_live = _Live()


class _Proxy:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        return getattr(getattr(_live, self._name), attr)


rk, tc = _Proxy("rk"), _Proxy("tc")

CELL, NS = "cell_liftoff_test", "robot1"
ORI = {"x": 0.0, "y": 0.707, "z": 0.0, "w": 0.707}
DOWN = [0.0, 0.0, -1.0]


class _Model:
    nq = 6

    def createData(self):
        return None


@pytest.fixture
def tcp(monkeypatch):
    """The start TCP is whatever the test sets — FK is not the subject here."""
    where = {"p": np.zeros(3)}
    monkeypatch.setattr(_live.rk, "_fk_ee", lambda m, d, q, b, y: np.asarray(where["p"], float))
    tc.reset_robot_configs(CELL)
    yield where
    tc.reset_robot_configs(CELL)


def _weld(op, xyz):
    tc.set_weld_pose(CELL, NS, op, xyz, DOWN, ORI)


def _seam(pinned=True):
    return rk.seam_at_start(CELL, NS, _Model(), [0] * 6, [0, 0, 0], 0.0, pinned=pinned)


def test_a_pinned_start_on_a_weld_lifts_off_that_weld(tcp):
    _weld("W01", [1.0, 0.0, 1.0])
    _weld("W02", [1.5, 0.0, 1.0])
    tc.set_last_pose(CELL, NS, [1.5, 0.0, 1.0], DOWN, ORI)       # most recent: W02
    tcp["p"] = [1.0, 0.003, 1.0]                                  # but the arm is at W01
    s = _seam(pinned=True)
    assert s["operation_id"] == "W01" and s["target"] == [1.0, 0.0, 1.0]
    assert s["rise_m"] == 0.0


def test_a_start_part_way_up_the_stroke_lifts_off_from_where_it_is(tcp):
    _weld("W01", [1.0, 0.0, 1.0])
    tcp["p"] = [1.0, 0.0, 1.04]                                   # 40 mm up the axis
    s = _seam(pinned=True)
    assert s["operation_id"] == "W01" and s["rise_m"] == pytest.approx(0.04)
    assert s["from_point"] == pytest.approx([1.0, 0.0, 1.04])


def test_a_pinned_start_off_every_seam_does_not_lift_off(tcp):
    _weld("W01", [1.0, 0.0, 1.0])
    tc.set_last_pose(CELL, NS, [1.0, 0.0, 1.0], DOWN, ORI)
    for p in ([1.05, 0.0, 1.0],          # 5 cm beside the seam
              [1.0, 0.0, 0.95],          # below the weld point
              [1.0, 0.0, 1.5]):          # far above the stroke
        tcp["p"] = p
        assert _seam(pinned=True) is None, p


def test_a_chained_start_keeps_the_most_recent_seam(tcp):
    """Unpinned, the start is where the last leg ended: today's rule stands."""
    tc.set_last_pose(CELL, NS, [1.0, 0.0, 1.0], DOWN, ORI)
    tcp["p"] = [3.0, 3.0, 3.0]
    assert _seam(pinned=False)["target"] == [1.0, 0.0, 1.0]


def test_a_plain_move_leaves_nothing_to_lift_off(tcp):
    tc.set_weld_pose(CELL, NS, "PARK", [1.0, 0.0, 1.0], DOWN, None)   # no orientation
    tcp["p"] = [1.0, 0.0, 1.0]
    assert _seam(pinned=True) is None


def test_weld_poses_survive_a_checkpoint_and_a_reset_clears_them(tcp):
    _weld("W01", [1.0, 0.0, 1.0])
    snap = tc.snapshot(CELL)
    assert snap["weld_poses"][0]["operation_id"] == "W01"
    tc.reset_robot_configs(CELL)
    assert tc.weld_poses(CELL, NS) == []
    tc.restore(CELL, snap)
    assert [op for op, _ in tc.weld_poses(CELL, NS)] == ["W01"]


def test_every_lift_off_site_uses_it():
    ompl = (REPO / "agents/trajectory/backends/ompl/planner.py").read_text()
    compound = (REPO / "agents/trajectory/backends/ompl_compound/planner.py").read_text()
    assert 'None if req.get("current_joints") else get_last_pose' not in ompl + compound
    assert ompl.count("seam_at_start(") >= 2              # transit legs + return home
    assert "seam_at_start(" in compound
    assert ompl.count("set_weld_pose(") == 1 and compound.count("set_weld_pose(") == 1
    assert '"liftoff_warnings"' in ompl
