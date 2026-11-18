"""Host unit tests for the two-phase (transit / approach) trajectory contract.

The design splits every working motion into a fast transit phase at normal
clearance and a slow terminal stroke at reduced clearance. Clearance was already
two-phase; these tests pin the SPEED half of that split, plus the continuity
property the behaviour-tree gap validator depends on.

Host-only: TOPP-RA is optional, so the Hermite fallback is exercised when it is
absent. Either path must satisfy the same contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.backends.ompl.planner import _parameterize_two_phase  # noqa: E402

VEL = [1.83, 1.75, 1.87, 3.05, 3.05, 4.01]
ACC = [3.0, 3.0, 3.0, 6.0, 6.0, 8.0]


def _path(n=5):
    """A straight joint-space path in joint 0 only, so speed is easy to measure."""
    return [[i * 0.2, 0, 0, 0, 0, 0] for i in range(n)]


def _mean_speed(seg):
    if len(seg) < 2:
        return 0.0
    vals = []
    for a, b in zip(seg, seg[1:]):
        dt = float(b["time_s"]) - float(a["time_s"])
        if dt > 0:
            vals.append(abs(b["positions"][0] - a["positions"][0]) / dt)
    return sum(vals) / len(vals) if vals else 0.0


# ─── Phase tagging ────────────────────────────────────────────────────────────

def test_every_waypoint_is_tagged_with_its_phase():
    traj = _parameterize_two_phase(_path(), seam_index=2, vel_limits=VEL, accel_limits=ACC)
    assert traj
    assert all(pt["phase"] in ("transit", "approach") for pt in traj)


def test_both_phases_are_present_when_a_seam_exists():
    traj = _parameterize_two_phase(_path(), seam_index=2, vel_limits=VEL, accel_limits=ACC)
    phases = [pt["phase"] for pt in traj]
    assert "transit" in phases and "approach" in phases
    # Transit must precede approach, and the phase must switch exactly once.
    assert phases == sorted(phases, key=lambda p: 0 if p == "transit" else 1)
    assert phases.count("transit") >= 1 and phases.count("approach") >= 1


def test_no_seam_means_a_single_transit_phase():
    """A pure transit move (no terminal stroke) is one phase at transit speed."""
    traj = _parameterize_two_phase(_path(), seam_index=0, vel_limits=VEL, accel_limits=ACC)
    assert {pt["phase"] for pt in traj} == {"transit"}


def test_seam_at_the_final_waypoint_is_single_phase():
    path = _path()
    traj = _parameterize_two_phase(path, seam_index=len(path) - 1,
                                   vel_limits=VEL, accel_limits=ACC)
    assert {pt["phase"] for pt in traj} == {"transit"}


# ─── The actual two-phase SPEED guarantee ─────────────────────────────────────

def test_approach_phase_runs_slower_than_transit():
    """The heart of the design: the terminal stroke is slower than transit."""
    traj = _parameterize_two_phase(
        _path(), seam_index=2, vel_limits=VEL, accel_limits=ACC,
        transit_velocity_scale=0.5, approach_velocity_scale=0.1,
    )
    transit = [p for p in traj if p["phase"] == "transit"]
    approach = [p for p in traj if p["phase"] == "approach"]
    # Include the seam sample in the approach segment: speed is a property of the
    # interval, and the first approach interval starts at the seam.
    approach_with_seam = transit[-1:] + approach
    assert _mean_speed(approach_with_seam) < _mean_speed(transit)


def test_a_slower_approach_scale_yields_a_slower_approach():
    """Monotonicity: halving the approach scale must not speed the stroke up."""
    fast = _parameterize_two_phase(
        _path(), seam_index=2, vel_limits=VEL, accel_limits=ACC,
        transit_velocity_scale=0.5, approach_velocity_scale=0.4)
    slow = _parameterize_two_phase(
        _path(), seam_index=2, vel_limits=VEL, accel_limits=ACC,
        transit_velocity_scale=0.5, approach_velocity_scale=0.05)

    def approach_duration(tr):
        ap = [p for p in tr if p["phase"] == "approach"]
        tp = [p for p in tr if p["phase"] == "transit"]
        return float(ap[-1]["time_s"]) - float(tp[-1]["time_s"])

    assert approach_duration(slow) > approach_duration(fast)


# ─── Continuity — what the BT gap validator relies on ─────────────────────────

def test_timestamps_are_strictly_monotonic_across_the_seam():
    """A duplicated or non-increasing timestamp would stall the executor."""
    traj = _parameterize_two_phase(
        _path(), seam_index=2, vel_limits=VEL, accel_limits=ACC,
        transit_velocity_scale=0.5, approach_velocity_scale=0.1)
    times = [float(p["time_s"]) for p in traj]
    assert times[0] == 0.0
    assert all(b > a for a, b in zip(times, times[1:])), times
    assert len(times) == len(set(times))


def test_seam_has_no_position_discontinuity():
    """The phases must join without a jump — the executor would otherwise teleport."""
    traj = _parameterize_two_phase(
        _path(), seam_index=2, vel_limits=VEL, accel_limits=ACC,
        transit_velocity_scale=0.5, approach_velocity_scale=0.1)
    phases = [p["phase"] for p in traj]
    i = phases.index("approach")
    before, after = traj[i - 1]["positions"], traj[i]["positions"]
    step = max(abs(a - b) for a, b in zip(before, after))
    # One sampling step apart, not a teleport. Bounded well below a joint's range.
    assert step < 0.2, f"seam jump {step} rad too large"


def test_path_endpoints_are_preserved():
    path = _path()
    traj = _parameterize_two_phase(path, seam_index=2, vel_limits=VEL, accel_limits=ACC)
    assert traj[0]["positions"][0] == pytest.approx(path[0][0], abs=1e-6)
    assert traj[-1]["positions"][0] == pytest.approx(path[-1][0], abs=1e-3)


# ─── Degenerate inputs ────────────────────────────────────────────────────────

def test_empty_path_returns_empty():
    assert _parameterize_two_phase([], seam_index=1, vel_limits=VEL) == []


def test_single_waypoint_is_a_single_tagged_sample():
    traj = _parameterize_two_phase([[0, 0, 0, 0, 0, 0]], seam_index=1, vel_limits=VEL)
    assert len(traj) == 1
    assert traj[0]["phase"] == "transit"
    assert traj[0]["time_s"] == 0.0


def test_works_without_velocity_limits_via_hermite_fallback():
    """No limits → Hermite fallback, but the phase contract still holds."""
    traj = _parameterize_two_phase(_path(), seam_index=2, vel_limits=None,
                                   transit_velocity_scale=0.5,
                                   approach_velocity_scale=0.1)
    assert traj
    assert {p["phase"] for p in traj} == {"transit", "approach"}
    times = [float(p["time_s"]) for p in traj]
    assert all(b > a for a, b in zip(times, times[1:]))
