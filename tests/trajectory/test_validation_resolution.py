"""The gate may not check a coarser trajectory than the monitor will enforce.

Geometry and thresholds were already shared — ``robot_kinematics`` owns the sphere
model, and ``validate_trajectory_set`` reads the cell's ``warning_distance``. What
was NOT shared was RESOLUTION. The gate sampled a step at ``n_samples=20`` while
the runtime monitor watched it at 25 Hz, which for a 9.5 s two-arm leg is ~238
samples. Twenty points cannot see a dip between them.

Measured on ``welding_cell_2_small``, plan ``bt_1787948775``, the home step — one
trajectory pair, one clearance, five answers:

    n=20   ->  0.2080 m   PASS
    n=40   ->  0.2144 m   PASS      (higher than n=20: aliasing is not monotone,
    n=200  ->  0.1848 m   BREACH     so a coarse reading is not a conservative
    n=800  ->  0.1847 m   BREACH     one — it is simply blind)
    monitor -> 0.1862 m   E-STOP after 17 nodes / 60.97 s

Had the gate seen 0.1847 it would have serialised the step and the run would have
finished; the cost of being right is cycle time. So ``n_samples`` becomes a FLOOR:
a caller may refine the check, never coarsen it below the monitor's rate.

Nothing here loosens anything — the 0.2 m warning band and the sphere radii are
untouched. This pins the direction of the inequality, which is the whole fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools.robot_kinematics import (  # noqa: E402
    MONITOR_RATE_HZ,
    validation_sample_count,
)


# ── the shared number has one owner ──────────────────────────────────────────

def test_the_monitor_rate_is_defined_once_and_the_monitor_defers_to_it():
    """Two copies of this number is precisely how the two sides drifted before."""
    monitor = pytest.importorskip(
        "agents.cell.collision_monitor", reason="needs the cell stack")

    import inspect
    sig = inspect.signature(monitor.CollisionMonitor.__init__)
    assert sig.parameters["rate_hz"].default is None, (
        "a literal default here is a second definition of the monitor rate; "
        "it must fall through to robot_kinematics.MONITOR_RATE_HZ")

    inst = monitor.CollisionMonitor.__new__(monitor.CollisionMonitor)
    monitor.CollisionMonitor.__init__(inst, "welding_cell_2_small", runtime=None)
    assert inst._rate_hz == pytest.approx(MONITOR_RATE_HZ)


def test_a_faster_monitor_is_allowed_a_slower_one_is_still_honoured():
    """The constant is a default, not a clamp — an explicit rate still applies."""
    monitor = pytest.importorskip(
        "agents.cell.collision_monitor", reason="needs the cell stack")
    inst = monitor.CollisionMonitor.__new__(monitor.CollisionMonitor)
    monitor.CollisionMonitor.__init__(inst, "welding_cell_2_small", None, rate_hz=100.0)
    assert inst._rate_hz == pytest.approx(100.0)


# ── the sample count ─────────────────────────────────────────────────────────

def test_a_leg_is_sampled_at_least_as_finely_as_the_monitor_ticks():
    """9.5 s at 25 Hz is 238 ticks, so 239 grid points; 20 was the bug."""
    n = validation_sample_count(9.5, floor=20)
    assert n == 239
    # the property that matters: no gap in the grid is longer than a monitor tick
    assert 9.5 / (n - 1) <= 1.0 / MONITOR_RATE_HZ


def test_the_floor_never_becomes_a_ceiling():
    """An LLM asking for 40 must not thereby request a coarser check than 239."""
    assert validation_sample_count(9.5, floor=40) == 239
    assert validation_sample_count(9.5, floor=1000) == 1000, (
        "a caller wanting a FINER check than the monitor must get it")


def test_a_short_leg_still_gets_the_requested_floor():
    """0.4 s is 11 ticks — the caller's 20 is the binding constraint there."""
    assert validation_sample_count(0.4, floor=20) == 20
    assert validation_sample_count(0.4, floor=2) == 11


def test_a_degenerate_horizon_never_yields_fewer_than_two_samples():
    """Availability: a zero-length or nonsense horizon must not crash the gate."""
    for horizon in (0.0, -1.0):
        for floor in (0, 1, -5):
            assert validation_sample_count(horizon, floor=floor) == 2


def test_the_count_grows_with_duration():
    """A longer motion spends longer under the monitor, so it needs more checks."""
    counts = [validation_sample_count(d, floor=2) for d in (1.0, 5.0, 9.5, 20.0)]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)


# ── the gate itself ──────────────────────────────────────────────────────────

def test_the_gate_grid_is_derived_from_the_monitor_not_from_the_argument():
    """The end-to-end contract: a coarse n_samples must not buy a coarse check.

    Geometry needs Pinocchio, so this drives the real tool on the real cell and
    asserts on the reported sample count rather than re-deriving clearance.
    """
    pytest.importorskip("pinocchio", reason="needs the geometry stack")
    planning_tools = pytest.importorskip(
        "agents.trajectory.tools.planning_tools", reason="needs the trajectory stack")

    cell = "welding_cell_2_small"
    # A 10 s two-arm hold. Identical, static, far-apart configs: this test is
    # about the GRID, so the poses are deliberately uninteresting.
    def hold(q, duration=10.0, points=3):
        return [{"positions": list(q), "time_s": duration * i / (points - 1)}
                for i in range(points)]

    zero = [0.0] * 6
    out = planning_tools.validate_trajectory_set.func(
        cell_id=cell,
        trajectories={"robot1": hold(zero), "robot2": hold(zero)},
        n_samples=20,
    )
    if "error" in out:
        pytest.skip(f"cell {cell} unavailable here: {out['error']}")

    assert out["monitor_rate_hz"] == pytest.approx(MONITOR_RATE_HZ)
    assert out["samples_checked"] == validation_sample_count(10.0, floor=20)
    assert out["samples_checked"] > 20, (
        "20 samples over 10 s is 2 Hz against a 25 Hz monitor — the gate would "
        "be blind to 92% of the ticks that can e-stop the cell")


def test_asking_for_a_finer_grid_than_the_monitor_is_still_honoured_end_to_end():
    pytest.importorskip("pinocchio", reason="needs the geometry stack")
    planning_tools = pytest.importorskip(
        "agents.trajectory.tools.planning_tools", reason="needs the trajectory stack")

    traj = [{"positions": [0.0] * 6, "time_s": 0.0},
            {"positions": [0.0] * 6, "time_s": 2.0}]
    out = planning_tools.validate_trajectory_set.func(
        cell_id="welding_cell_2_small",
        trajectories={"robot1": traj, "robot2": traj},
        n_samples=300,
    )
    if "error" in out:
        pytest.skip(f"cell unavailable here: {out['error']}")
    assert out["samples_checked"] == 300


def test_one_trajectory_is_not_a_pair_and_short_circuits():
    """Unchanged behaviour, pinned because the grid now runs before this point."""
    planning_tools = pytest.importorskip(
        "agents.trajectory.tools.planning_tools", reason="needs the trajectory stack")
    out = planning_tools.validate_trajectory_set.func(
        cell_id="welding_cell_2_small",
        trajectories={"robot1": [{"positions": [0.0] * 6, "time_s": 0.0}]},
    )
    assert out["collision_free"] is True
    assert out["pairs_checked"] == 0
