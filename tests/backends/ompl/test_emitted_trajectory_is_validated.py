"""A planner does not verify a path, it SAMPLES one — so check what it emits.

OMPL certifies a segment by checking its knots plus the straight-line motion
between them at ``validity_resolution``. Whatever happens between two checked
samples is assumed. And that resolution is a FRACTION of the state space's maximum
extent, not radians: this arm's joint space spans roughly 20 rad, so the old default
of 0.01 left about 0.2 rad — 11 degrees — between checks. The welding torch sits
0.9 m past the flange, so it sweeps a long way in 11 degrees, and a path can dip
into a fixture voxel and back out again entirely inside one unchecked interval.

It did. Planning ``robot2``/``W13`` from the pose ``W12`` left the arm in: every knot
OMPL certified was clear, and re-sampling that same certified path at 401 points per
span shows it 6.5 mm inside fixture voxel ``FIXTURE_A_v143``. Nothing re-checked
it, so it was emitted, executed, and caught by the runtime collision monitor as an
e-stop five seconds into the replay — the worst place to find out.

(The first diagnosis was that the time-parameterising spline bowed off the polyline
between knots. It does not, materially: adding knots at the polyline midpoints over
six rounds — 3 to 65 knots — moved the worst clearance from -0.00620 to -0.00620 m.
The path itself was never clear. Hence the fix below is resolution, not smoothing.)

Two things close it, and these tests pin both:

  * ``_trajectory_collision_info`` walks the EMITTED samples with the same predicate
    the runtime monitor applies, each under its own phase's speed allowance, and a
    breach FAILS the request. An availability failure the agent can act on beats a
    trajectory that stops the cell mid-motion.
  * the search resolution itself is finer, so paths this close are rejected during
    the search and a clear one is found instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.backends.ompl.planner import _search_resolution  # noqa: E402


# ── the search resolution ─────────────────────────────────────────────────────

def test_the_default_resolution_is_fine_enough_for_the_measured_case():
    """0.005 is a measurement, not a preference.

    It is the coarsest resolution at which W13's known-bad path is rejected by the
    search and a clear one found instead; 0.01 cannot see the 6.5 mm breach at all.
    Cost across four consecutive welds, 0.01 → 0.005 → 0.002 → 0.001: 60 s → 115 s
    → 253 s → 668 s. Loosening this without re-measuring re-opens the e-stop.
    """
    os.environ.pop("OMPL_VALIDITY_RESOLUTION", None)

    _, resolution = _search_resolution()

    assert resolution <= 0.005, (
        "0.01 leaves ~0.2 rad between motion checks on this arm, which hides a "
        "6.5 mm penetration by the torch tip"
    )


def test_the_resolution_is_overridable_per_run():
    """A cell that needs more care must not require a code change to get it."""
    os.environ["OMPL_VALIDITY_RESOLUTION"] = "0.002"
    try:
        assert _search_resolution()[1] == pytest.approx(0.002)
    finally:
        os.environ.pop("OMPL_VALIDITY_RESOLUTION", None)


def test_the_step_size_and_its_resolution_come_from_one_place():
    """Three call sites read this pair; they must not be able to disagree.

    The step size and the resolution that validates motions of that size are one
    decision. Duplicating the read is how two of the three sites would drift — the
    same N-copies-of-one-decision bug that let the planner and the monitor apply
    different collision models in the first place.
    """
    source = (PROJECT_ROOT / "agents" / "trajectory" / "backends" / "ompl" / "planner.py").read_text()

    assert source.count('os.environ.get("OMPL_VALIDITY_RESOLUTION"') == 1
    assert source.count('os.environ.get("OMPL_RANGE"') == 1
    assert source.count("_search_resolution()") >= 4, "one definition, three callers"


# ── the emitted-sample check ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cell():
    """robot2 of the real cell, or skip where the geometry is not mounted."""
    pytest.importorskip("pinocchio", reason="needs the ROS/pinocchio runtime")
    from agents.trajectory.tools.robot_kinematics import (
        _cell_sphere_radii, _load_clearance_policy, _load_pinocchio_model,
        _load_robot_config,
    )
    from agents.trajectory.tools.pinocchio import _load_obstacles
    cid = "welding_cell_2_small"
    try:
        model, _ = _load_pinocchio_model(cid, "robot2")
        rcfg = _load_robot_config(cid, "robot2")
        obstacles = _load_obstacles(cid)
    except Exception as exc:  # noqa: BLE001 — cell mount absent off-container
        pytest.skip(f"cell geometry unavailable: {exc}")
    if not obstacles:
        pytest.skip("no obstacles.yaml for this cell")
    return {
        "model": model, "data": model.createData(), "obstacles": obstacles,
        "base": np.array(rcfg["base_position"], dtype=float),
        "yaw": float(rcfg.get("base_orientation_yaw", 0.0)),
        "radii": _cell_sphere_radii(cid, model, rcfg.get("tool_collision")),
        "policy": _load_clearance_policy(cid),
    }


def _walk(cell, samples, transit_margin=0.0, approach_margin=0.0):
    from agents.trajectory.tools.robot_kinematics import _trajectory_collision_info
    return _trajectory_collision_info(
        cell["model"], cell["data"], samples, cell["base"], cell["yaw"],
        [], cell["obstacles"],
        sphere_radius=cell["policy"]["normal"], sphere_radii=cell["radii"],
        transit_margin=transit_margin, approach_margin=approach_margin,
    )


# Samples 8, 9 and 10 of the spline through W13's certified knots, read off the
# emitted trajectory while planning W13 from the pose W12 left robot2 in. Each is a
# configuration no validity checker ever saw; measured signed clearances against
# fixture voxel FIXTURE_A_v143 are -0.0001, -0.0053 and -0.0062 m, all on
# sphere 16 — the welding torch.
W13_SPLINE_BREACH = [
    [1.263574, -1.834288, 2.435279, -0.597554, -0.585179, 0.006544],
    [1.267063, -1.850225, 2.440420, -0.603452, -0.591870, 0.006584],
    [1.270962, -1.868038, 2.446166, -0.610045, -0.599350, 0.006629],
]
BLOCKING_VOXEL = "FIXTURE_A_v143"

# A sample from the SAME trajectory that is clear, with its headroom measured by
# bisection: it absorbs 0.0956 m of extra margin before the gate closes on it.
W13_CLEAR = [1.325371, -2.116578, 2.526338, -0.702029, -0.703706, 0.007252]
W13_CLEAR_HEADROOM_M = 0.0956


def test_a_breaching_sample_is_found_and_located(cell):
    """The check must name WHICH sample broke, or the failure is not actionable."""
    samples = [{"positions": q, "phase": "transit"} for q in W13_SPLINE_BREACH]

    info = _walk(cell, samples)

    assert info is not None, "these configurations penetrate a fixture voxel"
    assert info["sample_index"] == 0, "the FIRST breach is reported"
    assert info["phase"] == "transit"
    assert info["object_id"] == BLOCKING_VOXEL
    assert info["clearance_m"] < 0


def test_the_breach_is_the_torch_not_an_arm_link(cell):
    """Guard the diagnosis: the spline bows the TOOL into the fixture.

    If an arm link were the offender the remedy would be different — the arm is
    modelled at 0.12 m and moves far less per radian out here than the torch tip
    0.9 m past the flange, which is precisely why the tip is what a bowing spline
    swings into something.
    """
    from agents.trajectory.tools.robot_kinematics import _tool_frame_start
    from agents.trajectory.tools.robot_kinematics import _load_robot_config
    tool_cfg = _load_robot_config("welding_cell_2_small", "robot2").get("tool_collision")
    samples = [{"positions": q, "phase": "transit"} for q in W13_SPLINE_BREACH]

    info = _walk(cell, samples)

    assert info["link_index"] >= _tool_frame_start(cell["model"], tool_cfg)


def test_a_clear_trajectory_passes(cell):
    """No false positives: a sample from the same trajectory that is clear passes."""
    assert _walk(cell, [{"positions": W13_CLEAR, "phase": "transit"}]) is None


def test_each_sample_is_judged_under_its_own_phases_allowance(cell):
    """The phase tag is load-bearing, not decoration.

    A working stroke creeping onto the seam reserves less stopping room than a
    fast transit, so the same configuration can be legal in one phase and not the
    other. If the walk ignored ``phase`` it would apply one envelope to both and
    reintroduce exactly the planner/monitor disagreement this all exists to remove.
    """
    big = W13_CLEAR_HEADROOM_M + 0.02      # more room than this pose has
    small = W13_CLEAR_HEADROOM_M - 0.02    # less

    assert _walk(cell, [{"positions": W13_CLEAR, "phase": "approach"}],
                 transit_margin=big, approach_margin=small) is None, \
        "an approach sample must not be charged the transit allowance"
    assert _walk(cell, [{"positions": W13_CLEAR, "phase": "transit"}],
                 transit_margin=big, approach_margin=small) is not None, \
        "a transit sample must be charged the transit allowance"


def test_an_untagged_sample_is_charged_the_transit_allowance(cell):
    """Fail closed: an unlabelled sample gets the WIDER envelope, not the narrower.

    Transit is the fast phase, so its allowance is the larger of the two. Defaulting
    the other way would let an unlabelled sample buy the slow stroke's tighter gate
    without having slowed down.
    """
    assert _walk(cell, [{"positions": W13_CLEAR}],
                 transit_margin=W13_CLEAR_HEADROOM_M + 0.02,
                 approach_margin=0.0) is not None
