"""The collision radius is a property of the GEOMETRY; the phase changes the SPEED.

A 15-weld plan validated clean and then e-stopped 5 s into every replay, on
``robot1 link 16 penetrates box 'FIXTURE_A_v121'`` at -0.0007 to -0.0082 m.
Frame 16 is ``welding_tcp_joint`` — the welding torch tip, not an arm link. The
true distance from that sphere centre to the fixture voxel is 0.1618 m; the URDF's
own tool hulls put the torch's radial half-thickness at <= 0.047 m over the 0.60 m
nearest the tip. It was being modelled at 0.12 m.

Two components disagreed because each applied ONE uniform radius to every sphere:

  * the planner (``ompl_planner``/``rrt_planner``) validated transit segments at
    ``robot_link_radius`` (0.12) and the terminal descent at
    ``approach_link_radius`` (0.05) — a phase-dependent relaxation applied to the
    WHOLE arm, which also under-modelled the heavy castings on that stroke;
  * the collision monitor validated live joint states at a fixed 0.12, always.

So a weld pose was legal to the planner (0.05 uniform: +0.0505 m) and illegal to
the monitor (0.12 uniform: -0.0195 m).

The split that resolves it, and that these tests pin:

  * GEOMETRY does not depend on phase. Each body gets its own radius — arm 0.12,
    torch 0.05 — from one shared builder both components call. Measured across all
    15 planned welds the tool frame is the ONLY sphere that ever decides a terminal
    pose, so this reproduces the exemption's effect where it was needed and
    withdraws it where it was unearned.
  * SPEED does depend on phase, and that is the real reason a slow approach may
    come closer than a fast transit: stopping distance is latency x tip speed. It
    lives in ``_speed_margin`` — one formula, fed by the scale the planner is TOLD
    and the scale the monitor MEASURES from published joint states.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools.robot_kinematics import (  # noqa: E402
    _sphere_radii,
    _speed_margin_from,
    _tool_frame_start,
)

CELL = "welding_cell_2_small"

# The frame layout of kr240r2900__sticktool, from the real model: 18 frames, the
# tool mount is frame 15, and everything past it is torch geometry.
FRAME_NAMES = [
    "universe", "World", "root_joint", "like_kr240r2900_base_link",
    "joint_a1", "like_kr240r2900_link_1", "joint_a2", "like_kr240r2900_link_2",
    "joint_a3", "like_kr240r2900_link_3", "joint_a4", "like_kr240r2900_link_4",
    "joint_a5", "like_kr240r2900_link_5", "joint_a6", "like_kr240r2900_link_6",
    "welding_tcp_joint", "welding_tcp",
]
MOUNT_IDX = 15
ARM_R, TOOL_R = 0.12, 0.05
TOOL_CFG = {"mount_link": "link_6",
            "axis": {"direction": [-1, 0, 0], "start_m": 0.15,
                     "end_m": 0.9, "step_m": 0.15}}


def _model(names=FRAME_NAMES):
    """A stand-in exposing only what the radius helpers read: frame names."""
    return SimpleNamespace(frames=[SimpleNamespace(name=n) for n in names])


# ── the tool/arm split ────────────────────────────────────────────────────────

def test_the_mount_link_divides_arm_from_tool():
    """Frames past the mount link are torch; the mount link itself is arm."""
    assert _tool_frame_start(_model(), TOOL_CFG) == MOUNT_IDX + 1


def test_the_deepest_matching_frame_wins():
    """``mount_link`` is a substring, and several frames can contain it.

    A cell naming its mount ``link_1`` would match ``link_1`` AND ``link_10`` — the
    tool hangs off the DEEPEST match, so a shallow one must not capture the whole
    outer arm and thin it to the torch's radius.
    """
    names = ["universe", "arm_link_1", "arm_link_10", "tcp"]
    assert _tool_frame_start(_model(names), {"mount_link": "link_1"}) == 3


def test_a_robot_with_no_tool_has_no_tool_frames():
    """No ``tool_collision`` means nothing on the arm is torch geometry."""
    m = _model()
    assert _tool_frame_start(m, None) == len(FRAME_NAMES)
    assert _tool_frame_start(m, {}) == len(FRAME_NAMES)


def test_an_unmatched_mount_link_claims_nothing():
    """A typo'd mount must not silently make the whole arm thin — fail closed."""
    assert _tool_frame_start(_model(), {"mount_link": "no_such_link"}) == len(FRAME_NAMES)


# ── the radius vector ─────────────────────────────────────────────────────────

def test_only_the_torch_frames_get_the_tool_radius():
    r = _sphere_radii(_model(), ARM_R, TOOL_R, TOOL_CFG)

    assert list(r[MOUNT_IDX + 1:]) == [TOOL_R, TOOL_R], (
        "welding_tcp_joint and welding_tcp are the torch — they carry its radius")
    assert set(r[:MOUNT_IDX + 1]) == {ARM_R}, "every arm frame keeps the arm radius"


def test_the_vector_covers_exactly_the_frame_block():
    """The ORDERING CONTRACT this rests on.

    ``_sphere_positions`` emits one sphere per frame in frame-index order and then
    APPENDS the link-fill spheres. So a vector as long as the frame list indexes
    correctly, and every index past its end is a fill sphere — which
    ``_link_fill_positions`` takes from JOINT origins, i.e. always load-bearing arm
    steel, never tool. That is why the consumers fall back to the arm radius past
    the end, and why this vector needs no configuration-dependent length.
    """
    r = _sphere_radii(_model(), ARM_R, TOOL_R, TOOL_CFG)

    assert r.shape == (len(FRAME_NAMES),)


def test_the_arm_is_never_relaxed_to_the_tool_radius():
    """The regression guard: the retired exemption made the WHOLE arm thin.

    If a phase-dependent uniform radius is ever reintroduced, the heavy castings go
    back to being modelled at the torch's radius on the terminal stroke.
    """
    r = _sphere_radii(_model(), ARM_R, TOOL_R, TOOL_CFG)

    assert np.all(r[:MOUNT_IDX + 1] == ARM_R)
    assert np.count_nonzero(r == TOOL_R) == 2, "exactly the two torch frames"


def test_no_tool_config_reproduces_todays_uniform_model_exactly():
    """A cell that declares no torch geometry must behave byte-identically."""
    assert np.all(_sphere_radii(_model(), ARM_R, TOOL_R, None) == ARM_R)


def test_equal_radii_collapse_to_the_scalar_model():
    """With tool radius == arm radius the vector is the old uniform model."""
    assert np.all(_sphere_radii(_model(), ARM_R, ARM_R, TOOL_CFG) == ARM_R)


# ── the speed allowance ───────────────────────────────────────────────────────

def test_a_stationary_body_is_judged_on_geometry_alone():
    """At rest there is no stopping distance to reserve."""
    assert _speed_margin_from(0.08, 0.0) == 0.0


def test_the_allowance_is_linear_in_speed():
    """Reaction distance = latency x speed, and latency is a controller constant.

    This is what carries the phase dependence the two-radius switch used to carry:
    the terminal stroke runs at ``approach_velocity_fraction`` of transit speed, so
    it reserves proportionally less room — without pretending the arm got thinner.
    """
    assert _speed_margin_from(0.10, 1.0) == pytest.approx(0.10)
    assert _speed_margin_from(0.10, 0.4) == pytest.approx(0.04)


def test_a_speed_scale_outside_the_unit_range_is_clamped():
    """A bad measurement must not invent negative clearance or unbounded margin.

    The monitor DERIVES its scale from a finite difference, so a bad timestamp or a
    teleporting joint state can push it out of range.
    """
    assert _speed_margin_from(0.10, -3.0) == 0.0
    assert _speed_margin_from(0.10, 7.5) == pytest.approx(0.10)


def test_the_default_policy_preserves_todays_gate():
    """``speed_margin_full_speed_m`` defaults to 0.0 on purpose.

    The value is a property of the cell's controller e-stop latency, which this repo
    does not declare. Shipping a made-up number would be inventing a safety margin;
    shipping 0.0 ships the MECHANISM at exactly today's behaviour and leaves the
    number to measurement.
    """
    assert _speed_margin_from(0.0, 1.0) == 0.0


# ── the real geometry that blocked every replay ───────────────────────────────

# The tests above are pure and run anywhere. The ones below need the real cell
# geometry, so they skip per-fixture rather than at module scope — a module-level
# importorskip would take the pure tests down with them on a host with no ROS.

# robot1's live joint state while parked on the W03 seam — the configuration the
# monitor e-stopped on, read from get_robot_states during a failed replay.
PARKED_Q1 = [-1.722719, -1.666336, 1.869197, 2.341833, -1.330543, -1.54295]
BLOCKING_VOXEL = "FIXTURE_A_v121"


@pytest.fixture(scope="module")
def robot1():
    """Model + placement for robot1 of the real cell, or skip if unavailable."""
    from agents.trajectory.tools.robot_kinematics import (
        _load_pinocchio_model, _load_robot_config,
    )
    pytest.importorskip("pinocchio", reason="needs the ROS/pinocchio runtime")
    try:
        model, data = _load_pinocchio_model(CELL, "robot1")
        rcfg = _load_robot_config(CELL, "robot1")
    except Exception as exc:  # noqa: BLE001 — cell mount absent off-container
        pytest.skip(f"cell geometry unavailable: {exc}")
    return {
        "model": model, "data": data,
        "base": np.array(rcfg.get("base_position", [0, 0, 0]), dtype=float),
        "yaw": float(rcfg.get("base_orientation_yaw", 0.0)),
        "tool_cfg": rcfg.get("tool_collision"),
    }


@pytest.fixture(scope="module")
def obstacles():
    from agents.trajectory.tools.pinocchio import _load_obstacles
    obs = _load_obstacles(CELL)
    if not obs:
        pytest.skip("no obstacles.yaml for this cell")
    return obs


def _info(robot1, obstacles, radii, extra_margin=0.0):
    from agents.trajectory.tools.robot_kinematics import _config_collision_info
    return _config_collision_info(
        robot1["model"], robot1["data"], np.array(PARKED_Q1, dtype=float),
        robot1["base"], robot1["yaw"], [], obstacles,
        sphere_radius=ARM_R, sphere_radii=radii, extra_margin=extra_margin,
    )


def _radii(robot1, arm=ARM_R, tool=TOOL_R):
    return _sphere_radii(robot1["model"], arm, tool, robot1["tool_cfg"])


def test_the_uniform_model_reports_the_penetration_that_blocked_replay(robot1, obstacles):
    """Baseline: today's monitor flags the torch tip against the fixture voxel."""
    info = _info(robot1, obstacles, None)

    assert info is not None, "the uniform 0.12 model must still flag this pose"
    assert info["object_id"] == BLOCKING_VOXEL
    assert info["clearance_m"] < 0, "reported as penetration"


def test_the_torchs_own_radius_clears_the_same_pose(robot1, obstacles):
    """The fix: at the torch's real radius the parked pose is legal.

    Measured true distance is 0.1618 m to the voxel surface. With the tool radius
    (0.05) plus the box margin (0.05) that is +0.0618 m of clearance — the arm can
    legally transit out, which is what a plain 0.12 forbade.
    """
    assert _info(robot1, obstacles, _radii(robot1)) is None


def test_the_offending_sphere_really_is_the_torch(robot1, obstacles):
    """Guard the diagnosis: if the offender were an arm link, thinning the torch
    would be treating the wrong body. Frame 16 must be tool geometry."""
    start = _tool_frame_start(robot1["model"], robot1["tool_cfg"])
    info = _info(robot1, obstacles, None)

    assert info["link_index"] >= start, (
        f"link_index {info['link_index']} is an ARM frame (tool starts at {start})"
    )


def test_thinning_the_torch_does_not_thin_the_arm(robot1, obstacles):
    """The arm keeps its radius, so an arm-link breach is still caught.

    Same pose, same call, but with the arm radius inflated: the check must fail
    again on an ARM sphere. Proves the tool exemption did not become a blanket one.

    0.55 is measured, not chosen: at this pose the nearest ARM sphere (frame 10)
    sits 0.5407 m from voxel FIXTURE_A_v180, so with that box's 0.05 m margin
    the arm first breaches at r ≈ 0.4907 — the pose clears at 0.45 and fails at
    0.50. The torch is what stands close here; the castings do not.
    """
    info = _info(robot1, obstacles, _radii(robot1, arm=0.55))

    assert info is not None, "a fat arm must still collide"
    assert info["link_index"] < _tool_frame_start(robot1["model"], robot1["tool_cfg"])


def test_the_speed_allowance_tightens_the_same_gate(robot1, obstacles):
    """The allowance is a real term, not decoration: enough of it re-blocks the pose.

    This is what makes the phase dependence honest. The pose clears with +0.0618 m
    of margin, so an arm moving fast enough to need 0.08 m of stopping room is NOT
    cleared to be there — while the same arm creeping in on a working stroke is.
    """
    radii = _radii(robot1)

    assert _info(robot1, obstacles, radii, extra_margin=0.02) is None
    assert _info(robot1, obstacles, radii, extra_margin=0.08) is not None


def test_the_cell_declares_a_torch_radius_smaller_than_its_arm(robot1):
    """welding_cell_2_small must actually opt in, or the fix is inert there."""
    from agents.trajectory.tools.robot_kinematics import _load_clearance_policy
    pol = _load_clearance_policy(CELL)

    assert pol["tool"] < pol["normal"], (
        f"tool radius {pol['tool']} must be below the arm radius {pol['normal']}")
    assert pol["tool"] >= 0.047, (
        "the URDF tool hulls measure 0.047 m radially over the 0.60 m nearest the "
        "TCP; a smaller modelled radius would claim the torch is thinner than its "
        "own geometry")


def test_the_planner_and_the_monitor_build_one_vector(robot1):
    """The invariant the whole change exists to establish.

    Not that two functions agree — that there is only ONE. Both the planners and
    the runtime monitor reach the radii through ``_cell_sphere_radii``, keyed on the
    cell's own config, so a configuration cannot be free to one and a collision to
    the other. A second builder is how this bug happened.
    """
    from agents.trajectory.tools.robot_kinematics import _cell_sphere_radii
    pol_radii = _cell_sphere_radii(CELL, robot1["model"], robot1["tool_cfg"])

    assert pol_radii is not None, "the cell declares a tool radius, so it must build one"
    assert np.array_equal(pol_radii, _radii(robot1))
