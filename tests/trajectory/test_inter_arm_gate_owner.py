"""The inter-arm gate has ONE owner, and the planner consults it.

Two arms clearing each other while moving is a single decision. It used to have two
implementations: ``validate_trajectory_set`` (time-aware, sampled on a shared
wall clock) and, inside the OMPL planner, the swept TUBE — each arm planned against
every place the arms before it would ever be, with no reference to *when*.

Measured on ``bt_1787955158`` / ``welding_cell_2_small``, the two had drifted in both
directions at once:

  * too strict — step 2 ``robot2``/``W07`` shipped a 1.335 m arc because the tube
    admitted nothing flatter; the direct route rises 0.000 m and clears the gate at
    0.2488 m against a 0.2 m band.
  * too lax — the same tube ACCEPTED a route the gate measures at 0.1551 m, inside
    the band, i.e. an E-STOP.

These tests pin the consolidation: one function owns the verdict, the tool is a
wrapper over it, and the planner's relaxation is admitted only by that verdict.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.trajectory.tools import ompl_planner as op
from agents.trajectory.tools import robot_kinematics as rk


# ── the owner exists and the tool delegates to it ────────────────────────────

def test_the_gate_is_a_plain_function_not_only_a_tool():
    """The planner cannot depend on a langchain @tool, so the owner is a function."""
    assert callable(rk.inter_robot_clearance)


def test_validate_trajectory_set_delegates_to_the_owner(monkeypatch):
    """The tool must not carry a second copy of the verdict."""
    pytest.importorskip("langchain_core")
    from agents.trajectory.tools import planning_tools as pt

    seen = {}

    def fake(cell_id, trajectories, collision_radius_m=0.12, n_samples=20,
             warning_distance_m=None):
        seen["args"] = (cell_id, sorted(trajectories), collision_radius_m, n_samples)
        seen["band"] = warning_distance_m
        return {"collision_free": True, "min_clearance_m": 0.9, "robots": ["a", "b"]}

    monkeypatch.setattr(pt, "inter_robot_clearance", fake)
    monkeypatch.setattr(pt, "_load_collision_config",
                        lambda _cell: {"warning_distance": 0.31})
    got = pt.validate_trajectory_set.func(
        cell_id="c", trajectories={"a": [{"positions": [0], "time_s": 0.0}],
                                   "b": [{"positions": [0], "time_s": 0.0}]},
        n_samples=33,
    )
    assert seen["args"] == ("c", ["a", "b"], 0.12, 33)
    assert got["min_clearance_m"] == 0.9
    # The band travels WITH the call: the tool resolves the cell's collision policy
    # and hands it to the owner, so overriding the policy at this layer changes the
    # verdict instead of being silently reloaded and discarded deeper down.
    assert seen["band"] == pytest.approx(0.31)


def test_the_tool_adds_the_suggestion_the_owner_does_not(monkeypatch):
    """Advice is presentation; the verdict is the owner's. Only the wrapper phrases it."""
    pytest.importorskip("langchain_core")
    from agents.trajectory.tools import planning_tools as pt

    monkeypatch.setattr(pt, "_load_collision_config",
                        lambda _cell: {"warning_distance": 0.2})
    monkeypatch.setattr(pt, "inter_robot_clearance", lambda *a, **k: {
        "collision_free": False, "min_clearance_m": 0.1551,
        "warning_distance_m": 0.2, "colliding_pairs": [{"robots": ["a", "b"]}],
    })
    got = pt.validate_trajectory_set.func(cell_id="c", trajectories={})
    assert "0.1551" in got["suggestion"] and "0.2" in got["suggestion"]
    assert "SEQUENCE" in got["suggestion"]


def test_fewer_than_two_arms_is_not_an_error():
    r = rk.inter_robot_clearance("any-cell", {"solo": [{"positions": [0.0]}]})
    assert r["collision_free"] is True
    assert r["min_clearance_m"] == float("inf")
    assert r["pairs_checked"] == 0


def test_the_gate_samples_at_the_monitors_resolution(monkeypatch):
    """The floor established by the resolution fix still applies through the owner."""
    monkeypatch.setattr(rk, "MONITOR_RATE_HZ", 25.0)
    assert rk.validation_sample_count(9.5, floor=20) == 239


# ── the emitted-rise limit is a separate, more generous number ────────────────

def test_the_emitted_limit_sits_above_the_selection_preference():
    """A hard refusal needs headroom a preference does not."""
    assert op._emitted_rise_limit() > op._path_rise_policy()


def test_the_emitted_limit_lands_in_the_measured_gap():
    """Tallest accepted leg 0.131 m; shortest complained-about leg 0.585 m."""
    limit = op._emitted_rise_limit()
    assert 0.131 < limit < 0.585


def test_the_emitted_limit_is_env_tunable(monkeypatch):
    monkeypatch.setenv("OMPL_MAX_EMITTED_RISE_M", "1.25")
    assert op._emitted_rise_limit() == pytest.approx(1.25)


def test_zero_disables_the_refusal(monkeypatch):
    """A cell that genuinely arcs over a fixture must be able to opt out."""
    monkeypatch.setenv("OMPL_MAX_EMITTED_RISE_M", "0")
    assert op._emitted_rise_limit() == 0.0


def test_an_unparseable_limit_falls_back_to_the_measured_default(monkeypatch):
    monkeypatch.setenv("OMPL_MAX_EMITTED_RISE_M", "waist-high")
    assert op._emitted_rise_limit() == pytest.approx(0.35)


def test_a_negative_limit_is_floored_not_honoured(monkeypatch):
    """A negative limit would refuse every leg ever planned."""
    monkeypatch.setenv("OMPL_MAX_EMITTED_RISE_M", "-2.0")
    assert op._emitted_rise_limit() == 0.0


# ── rise is measured on the EMITTED array, not on planner states ──────────────

class _FakeModel:
    """z of the tool == base z + q[0], so rise is readable straight off the joints."""
    nq = 2

    def createData(self):
        return None


#: Patched by dotted path, not through the ``rk`` alias above. ``_path_tool_rise``
#: imports ``_fk_ee`` from ``sys.modules`` at CALL time, and other tests in this
#: suite install stub modules under that name — so a patch applied to the module
#: object this file imported can land on a different object than the one the
#: planner reaches, and the fake FK silently never runs. When that happens every
#: rise measures 0.0, which is what these tests assert is WRONG, so the failure is
#: only visible in a full-suite run. Resolving the target by name each time makes
#: the patch follow whichever module is installed.
_FK_TARGET = "agents.trajectory.tools.robot_kinematics._fk_ee"


@pytest.fixture
def fk(monkeypatch):
    monkeypatch.setattr(
        _FK_TARGET,
        lambda model, data, q, base, yaw: np.array([0.0, 0.0, base[2] + q[0]]),
        raising=False)


BASE = np.array([0.0, 0.0, 0.0])


def test_emitted_rise_is_the_excursion_above_both_endpoints(fk):
    traj = [{"positions": [1.0, 0.0]}, {"positions": [1.9, 0.0]},
            {"positions": [1.2, 0.0]}]
    assert op._trajectory_tool_rise(_FakeModel(), traj, BASE, 0.0) == pytest.approx(0.7)


def test_a_climb_to_a_higher_endpoint_is_not_a_rise(fk):
    """Welds at different heights are legitimate; only excursions count."""
    traj = [{"positions": [1.0, 0.0]}, {"positions": [1.5, 0.0]},
            {"positions": [2.0, 0.0]}]
    assert op._trajectory_tool_rise(_FakeModel(), traj, BASE, 0.0) == pytest.approx(0.0)


def test_extra_joints_beyond_nq_are_ignored(fk):
    """A trajectory point may carry more numbers than the model has joints."""
    traj = [{"positions": [1.0, 0.0, 7.0, 7.0]}, {"positions": [1.6, 0.0, 7.0, 7.0]},
            {"positions": [1.0, 0.0, 7.0, 7.0]}]
    assert op._trajectory_tool_rise(_FakeModel(), traj, BASE, 0.0) == pytest.approx(0.6)


def test_a_degenerate_trajectory_has_no_rise(fk):
    assert op._trajectory_tool_rise(_FakeModel(), [], BASE, 0.0) == 0.0
    assert op._trajectory_tool_rise(
        _FakeModel(), [{"positions": [1.0, 0.0]}], BASE, 0.0) == 0.0


def test_unmeasurable_rise_reports_zero_not_excessive(monkeypatch):
    """FK that cannot run must not be reported as a tall leg — that would refuse
    every operation on a cell without Pinocchio."""
    def boom(*a, **k):
        raise RuntimeError("no FK here")

    monkeypatch.setattr(_FK_TARGET, boom, raising=False)
    traj = [{"positions": [1.0, 0.0]}, {"positions": [9.0, 0.0]},
            {"positions": [1.0, 0.0]}]
    assert op._trajectory_tool_rise(_FakeModel(), traj, BASE, 0.0) == 0.0


def test_emitted_rise_sees_what_planner_state_scoring_cannot(fk):
    """The two measurements share a definition and differ in WHICH object they read.

    Candidate scoring reads the planner's states; this reads the timed array those
    states were interpolated into. Where interpolation introduces a hump, only the
    emitted measurement reports it — which is the whole reason for a second check.
    """
    states = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    emitted = [{"positions": [1.0, 0.0]}, {"positions": [2.4, 0.0]},
               {"positions": [1.0, 0.0]}]
    model = _FakeModel()
    assert op._path_tool_rise(states, model, None, BASE, 0.0) == pytest.approx(0.0)
    assert op._trajectory_tool_rise(model, emitted, BASE, 0.0) == pytest.approx(1.4)
