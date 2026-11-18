"""The monitor has to MEASURE the speed the planner is simply told.

The clearance gate is ``per-body radius + per-box margin + speed allowance``, and
the allowance is linear in the fraction of full joint speed the arm is moving at.
The planner receives that fraction as ``velocity_scale``. The monitor gets no such
thing: ``robot_status`` publishes joint POSITIONS and a timestamp at ~10 Hz and no
velocities. So it finite-differences consecutive published samples and normalises
each joint by its URDF velocity limit, producing the same quantity — otherwise the
two halves are back to enforcing different gates, which is the bug this whole
change exists to remove.

Every path where that measurement is not trustworthy must fail CLOSED, i.e. report
FULL speed and demand the widest clearance. Guessing "stopped" would license a
tighter gate than the arm has earned; guessing "flat out" only ever asks for more
room. The cases: no velocity limits declared, no previous sample yet, a
non-advancing timestamp (the monitor polls faster than the publisher), and a gap
too wide for a finite difference to mean anything.

Pure unit tests over a stub runtime — no ROS, no Pinocchio, no cell mount.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.cell.collision_monitor import CollisionMonitor  # noqa: E402

RID = "robot1"
# joint_velocity_limits for kr240r2900, from robots.yaml (URDF <limit velocity>).
LIMITS = [2.722714, 2.722714, 2.722714, 5.759586, 5.759586, 10.73377]


@pytest.fixture
def monitor():
    """A monitor with the velocity limits loaded and nothing else."""
    m = CollisionMonitor("welding_cell_2_small", runtime=object())
    m._vel_limits[RID] = np.array(LIMITS, dtype=float)
    m._speed_scale[RID] = 1.0
    return m


def _q(*vals) -> np.ndarray:
    return np.array(vals, dtype=float)


ZERO = _q(0, 0, 0, 0, 0, 0)


def test_the_first_sample_assumes_full_speed():
    """One position tells you nothing about speed — reserve the most room."""
    m = CollisionMonitor("welding_cell_2_small", runtime=object())
    m._vel_limits[RID] = np.array(LIMITS, dtype=float)

    assert m._speed_scale_for(RID, ZERO, {"timestamp": 100.0}) == 1.0


def test_a_stationary_arm_measures_as_stopped(monitor):
    """Two samples, no movement: the allowance may legitimately fall to zero."""
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})

    assert monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.1}) == 0.0


def test_a_joint_at_its_limit_measures_as_full_speed(monitor):
    """Moving joint 0 through exactly its limit in 0.1 s is, by definition, 1.0."""
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    q = _q(LIMITS[0] * 0.1, 0, 0, 0, 0, 0)

    assert monitor._speed_scale_for(RID, q, {"timestamp": 100.1}) == pytest.approx(1.0)


def test_half_limit_speed_measures_as_a_half(monitor):
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    q = _q(LIMITS[0] * 0.05, 0, 0, 0, 0, 0)

    assert monitor._speed_scale_for(RID, q, {"timestamp": 100.1}) == pytest.approx(0.5)


def test_the_fastest_joint_sets_the_scale(monitor):
    """Each joint is normalised by ITS OWN limit, and the worst one governs.

    The wrist (limit 10.7 rad/s) moving at 0.2 of its limit is slower in the sense
    that matters than the base (2.72) moving at 0.6 of its. Normalising before the
    max is what makes those comparable.
    """
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    q = _q(LIMITS[0] * 0.06, 0, 0, 0, 0, LIMITS[5] * 0.02)

    assert monitor._speed_scale_for(RID, q, {"timestamp": 100.1}) == pytest.approx(0.6)


def test_an_impossibly_fast_reading_is_clamped_not_amplified(monitor):
    """A teleporting joint state must not scale the margin without bound."""
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    q = _q(50.0, 0, 0, 0, 0, 0)

    assert monitor._speed_scale_for(RID, q, {"timestamp": 100.1}) == 1.0


# ── fail-closed paths ─────────────────────────────────────────────────────────

def test_a_repolled_sample_holds_the_last_estimate(monitor):
    """The monitor runs at 25 Hz over a 10 Hz publisher, so this is the common case.

    A non-advancing timestamp is not evidence the arm stopped — it is the absence of
    new evidence. Differencing it against itself would read 0.0 and quietly relax
    the gate on most polls.
    """
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    q = _q(LIMITS[0] * 0.1, 0, 0, 0, 0, 0)
    monitor._speed_scale_for(RID, q, {"timestamp": 100.1})

    assert monitor._speed_scale_for(RID, q, {"timestamp": 100.1}) == pytest.approx(1.0)


def test_a_repolled_sample_does_not_destroy_the_reference(monitor):
    """The older sample must survive a re-poll, or the next fresh one has nothing
    to difference against and the estimate stalls forever."""
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})    # same sample again
    q = _q(LIMITS[0] * 0.05, 0, 0, 0, 0, 0)

    assert monitor._speed_scale_for(RID, q, {"timestamp": 100.1}) == pytest.approx(0.5)


def test_a_stale_pair_is_not_trusted(monitor):
    """Across a wide gap the arm could have accelerated and stopped again, so the
    average says nothing about its speed NOW."""
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})

    assert monitor._speed_scale_for(RID, ZERO, {"timestamp": 105.0}) == 1.0


def test_a_stale_pair_still_reseeds_the_reference(monitor):
    """Discarding the estimate must not also discard the chance to recover: the
    stale sample becomes the new reference so the next poll can measure."""
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 105.0})    # discarded
    q = _q(LIMITS[0] * 0.05, 0, 0, 0, 0, 0)

    assert monitor._speed_scale_for(RID, q, {"timestamp": 105.1}) == pytest.approx(0.5)


def test_a_backwards_timestamp_is_not_trusted(monitor):
    """Out-of-order status must not produce a negative speed."""
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})

    assert monitor._speed_scale_for(RID, _q(1, 0, 0, 0, 0, 0),
                                    {"timestamp": 99.0}) == 1.0


def test_a_missing_timestamp_holds_the_last_estimate(monitor):
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.1})    # measures 0.0

    assert monitor._speed_scale_for(RID, ZERO, {}) == 0.0


def test_undeclared_velocity_limits_assume_full_speed(monitor):
    """Without limits there is no scale to normalise against — reserve the most."""
    monitor._vel_limits[RID] = np.array([], dtype=float)
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})

    assert monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.1}) == 1.0


def test_a_zero_velocity_limit_assumes_full_speed(monitor):
    """A zero limit would divide by zero; treat the config as unusable."""
    monitor._vel_limits[RID] = np.array([0.0] + LIMITS[1:], dtype=float)
    monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.0})

    assert monitor._speed_scale_for(RID, ZERO, {"timestamp": 100.1}) == 1.0


def test_an_unknown_robot_assumes_full_speed(monitor):
    """A robot the monitor never loaded limits for gets the conservative answer."""
    assert monitor._speed_scale_for("robot_99", ZERO, {"timestamp": 100.0}) == 1.0
