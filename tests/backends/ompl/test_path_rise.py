"""Rank candidates by the height an operator can SEE, not by a proxy for it.

Directness — joint-space path length over ``||q_goal - q_start||`` — predicted tool
rise at +0.878 on this cell, and that was enough to fix the three legs that were
complained about. It was not enough to keep them fixed. Plan ``bt_1787948812``
shipped a robot2 home return that rose **0.893 m** (peak tool z 3.162 m) with a
directness ratio comfortably under the 1.6 cutoff: joint-space length says how far
the arm travelled, never where it went, and a compact rotation of the lower joints
lifts the tool a long way for very little arc length.

So the score becomes the measured quantity. ``_path_tool_rise`` runs FK on each
waypoint and reports the excursion above BOTH endpoints — about three FK calls per
candidate, against a solve that costs 6-10 s.

Measured above the endpoints, not absolute: a leg travelling to a HIGHER weld must
gain height, and that is not an excursion. The 0.15 m cutoff comes from the gap in
the data — legs nobody objected to rose 0.007-0.131 m (of which the visible
0.100-0.120 m is the program's own ``approach_offset_m``), legs operators objected
to rose 0.5-2.8 m, and nothing was measured in between.

What must NOT regress: a segment that used to plan must still plan. Selection is a
quality mechanism and may never turn an available path into a refusal — including
when FK itself is unavailable or throws.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

ompl_planner = pytest.importorskip(
    "agents.trajectory.backends.ompl.planner", reason="needs the trajectory stack")

_path_tool_rise = ompl_planner._path_tool_rise
_path_directness = ompl_planner._path_directness
_path_rise_policy = ompl_planner._path_rise_policy
_path_score_policy = ompl_planner._path_score_policy
_pick_path_candidate = ompl_planner._pick_path_candidate


# ── a fake arm, so the metric is testable without Pinocchio ──────────────────
#
# The contract _path_tool_rise depends on is exactly `_fk_ee(model, data, q, base,
# yaw) -> xyz`. This stands in for it with a one-joint prismatic "arm" whose tool z
# is base[2] + q[0], which makes every expected rise arithmetic rather than a
# fixture read.

class _FakeModel:
    def __init__(self, nq: int = 1):
        self.nq = nq


def _fake_fk(monkeypatch, fn):
    """Point robot_kinematics._fk_ee (imported inside the metric) at `fn`."""
    rk = pytest.importorskip("agents.trajectory.tools.robot_kinematics")
    monkeypatch.setattr(rk, "_fk_ee", fn)


def _z_is_first_joint(model, data, q, base, yaw):
    return np.array([0.0, 0.0, float(base[2]) + float(q[0])])


BASE = np.zeros(3)


def _profile(*zs):
    """A path whose tool z follows `zs`, under _z_is_first_joint."""
    return [np.array([float(z)]) for z in zs]


# ── the metric ───────────────────────────────────────────────────────────────

def test_a_level_transit_has_no_rise(monkeypatch):
    _fake_fk(monkeypatch, _z_is_first_joint)
    assert _path_tool_rise(_profile(1.35, 1.35, 1.35),
                           _FakeModel(), None, BASE, 0.0) == pytest.approx(0.0)


def test_an_arc_over_the_cell_is_measured_at_its_peak(monkeypatch):
    """The 0.893 m hump the directness proxy let through."""
    _fake_fk(monkeypatch, _z_is_first_joint)
    rise = _path_tool_rise(_profile(2.269, 3.162, 2.269),
                           _FakeModel(), None, BASE, 0.0)
    assert rise == pytest.approx(0.893, abs=1e-6)


def test_climbing_to_a_higher_goal_is_not_an_excursion(monkeypatch):
    """Measured against BOTH endpoints, or every ascent looks like a defect."""
    _fake_fk(monkeypatch, _z_is_first_joint)
    model = _FakeModel()
    # a monotone climb from 1.35 to 1.65 — the W15 weld is genuinely higher
    assert _path_tool_rise(_profile(1.35, 1.5, 1.65), model, None, BASE,
                           0.0) == pytest.approx(0.0)
    # ...and the same climb with a 0.4 m hump on the way IS an excursion
    assert _path_tool_rise(_profile(1.35, 2.05, 1.65), model, None, BASE,
                           0.0) == pytest.approx(0.4)


def test_a_descent_below_the_endpoints_is_not_counted_as_rise(monkeypatch):
    """This metric owns height, not every kind of detour — and it never goes < 0."""
    _fake_fk(monkeypatch, _z_is_first_joint)
    assert _path_tool_rise(_profile(1.35, 0.4, 1.35),
                           _FakeModel(), None, BASE, 0.0) == 0.0


def test_a_path_that_need_not_move_has_no_rise(monkeypatch):
    _fake_fk(monkeypatch, _z_is_first_joint)
    model = _FakeModel()
    assert _path_tool_rise(_profile(1.2, 1.2), model, None, BASE, 0.0) == 0.0
    assert _path_tool_rise(_profile(1.2), model, None, BASE, 0.0) == 0.0


def test_the_base_height_offsets_the_tool_but_not_the_excursion(monkeypatch):
    """Rise is relative, so bolting the arm on a plinth must not change it."""
    _fake_fk(monkeypatch, _z_is_first_joint)
    model = _FakeModel()
    flat, high = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.75])
    assert (_path_tool_rise(_profile(1.0, 1.9, 1.0), model, None, flat, 0.0)
            == _path_tool_rise(_profile(1.0, 1.9, 1.0), model, None, high, 0.0)
            == pytest.approx(0.9))


def test_extra_joint_values_beyond_nq_are_ignored(monkeypatch):
    """Blackboard trajectories carry padded position vectors; FK takes model.nq."""
    seen = {}

    def fk(model, data, q, base, yaw):
        seen["nq"] = len(q)
        return np.array([0.0, 0.0, float(q[0])])

    _fake_fk(monkeypatch, fk)
    padded = [np.array([1.0, 9.0, 9.0]), np.array([1.0, 9.0, 9.0])]
    _path_tool_rise(padded, _FakeModel(nq=1), None, BASE, 0.0)
    assert seen["nq"] == 1


# ── the metric beats the proxy on the case that shipped ──────────────────────

def test_rise_rejects_the_arc_that_directness_accepted(monkeypatch):
    """The regression this replaces, in one test.

    Two candidates between the SAME endpoints, both at tool height 1.0, one radian
    apart in joint space. Only joint 0 carries height here, so:

      * ``arc``  swings joint 0 up 0.9 rad and back — 0.900 m of rise for only
        2.06 rad of travel, so directness scores it a mild 2.06;
      * ``flat`` wanders 1.5 rad along joint 1, which changes no height at all —
        0.000 m of rise for 4.00 rad of travel, so directness scores it 4.00.

    Directness therefore ranks the arc BETTER, and that ordering is precisely how
    ``bt_1787948812`` shipped a 0.893 m hump under a 1.6 cutoff. Rise inverts it.
    """
    _fake_fk(monkeypatch, _z_is_first_joint)
    model = _FakeModel(nq=2)

    arc = [np.array([1.0, 0.0]), np.array([1.9, 0.5]), np.array([1.0, 1.0])]
    flat = [np.array([1.0, 0.0]), np.array([1.0, -1.5]), np.array([1.0, 1.0])]

    # the proxy prefers the arc...
    assert _path_directness(arc) == pytest.approx(2.059, abs=0.001)
    assert _path_directness(flat) == pytest.approx(4.0, abs=0.001)
    assert _path_directness(arc) < 1.6 * 1.3 < _path_directness(flat)
    # ...and it is wrong: all of the arc's height, none of the wander's.
    assert _path_tool_rise(arc, model, None, BASE, 0.0) == pytest.approx(0.9)
    assert _path_tool_rise(flat, model, None, BASE, 0.0) == pytest.approx(0.0)

    score, target = _path_score_policy(model, None, BASE, 0.0)
    calls = {"n": 0}

    def solve():
        calls["n"] += 1
        return (arc, None) if calls["n"] == 1 else (flat, None)

    picked, failure = _pick_path_candidate(solve, attempts=4, directness_target=target,
                                           score=score)
    assert failure is None
    assert picked is flat, "the arc must not ship when a flat draw is available"
    assert calls["n"] == 2, "it must keep solving while only arcs have turned up"


def test_a_leg_within_the_approach_lift_ships_on_the_first_solve(monkeypatch):
    """0.120 m is the program's own approach_offset_m; paying for a re-solve to
    shave a deliberate lift would make every good segment slower for nothing."""
    _fake_fk(monkeypatch, _z_is_first_joint)
    model = _FakeModel()
    good = _profile(1.35, 1.47, 1.35)   # rise 0.120
    score, target = _path_score_policy(model, None, BASE, 0.0)

    calls = {"n": 0}

    def solve():
        calls["n"] += 1
        return good, None

    picked, _ = _pick_path_candidate(solve, attempts=4, directness_target=target,
                                     score=score)
    assert picked is good
    assert calls["n"] == 1


# ── the policy ───────────────────────────────────────────────────────────────

def test_the_default_cutoff_sits_in_the_gap_between_good_and_bad_legs():
    monkey = pytest.MonkeyPatch()
    try:
        monkey.delenv("OMPL_PATH_MAX_RISE_M", raising=False)
        rise = _path_rise_policy()
    finally:
        monkey.undo()
    assert rise == pytest.approx(0.15)
    # above every leg nobody objected to (max measured 0.131) and far below the
    # cheapest one they did (0.5)
    assert 0.131 < rise < 0.5


def test_the_cutoff_is_tunable_without_a_code_change(monkeypatch):
    monkeypatch.setenv("OMPL_PATH_MAX_RISE_M", "0.4")
    assert _path_rise_policy() == pytest.approx(0.4)


def test_an_unreachable_cutoff_is_floored(monkeypatch):
    """Rise cannot be negative, so a negative target would burn every solve."""
    monkeypatch.setenv("OMPL_PATH_MAX_RISE_M", "-1")
    assert _path_rise_policy() == 0.0


def test_unparseable_tuning_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("OMPL_PATH_MAX_RISE_M", "low-ish")
    assert _path_rise_policy() == pytest.approx(0.15)


def test_the_selected_score_and_its_cutoff_are_in_the_same_unit(monkeypatch):
    """Metres against a ratio, or vice versa, silently disables the stop rule."""
    _fake_fk(monkeypatch, _z_is_first_joint)
    score, target = _path_score_policy(_FakeModel(), None, BASE, 0.0)
    assert target == pytest.approx(_path_rise_policy())
    assert score(_profile(1.0, 1.9, 1.0)) == pytest.approx(0.9), "metres"


# ── availability ─────────────────────────────────────────────────────────────

def test_when_fk_is_unusable_the_proxy_takes_over_with_its_own_cutoff(monkeypatch):
    """No Pinocchio, no tool frame — plan anyway, ranked by the cheap proxy."""
    def boom(*a, **k):
        raise RuntimeError("no pinocchio here")

    _fake_fk(monkeypatch, boom)
    monkeypatch.delenv("OMPL_PATH_DIRECTNESS", raising=False)
    score, target = _path_score_policy(_FakeModel(), None, BASE, 0.0)

    assert score is _path_directness
    assert target == pytest.approx(1.6), "the directness cutoff, not 0.15 m"


def test_a_score_that_throws_mid_batch_still_ships_a_valid_path():
    """Selection is quality. It may never turn an available path into a refusal."""
    path = [np.array([0.0]), np.array([1.0])]

    def bad_score(_waypoints):
        raise ValueError("frame index out of range")

    picked, failure = _pick_path_candidate(lambda: (path, None), attempts=4,
                                           directness_target=0.15, score=bad_score)
    assert picked is path
    assert failure is None


def test_selection_by_rise_never_turns_a_failure_into_a_lost_path(monkeypatch):
    _fake_fk(monkeypatch, _z_is_first_joint)
    score, target = _path_score_policy(_FakeModel(), None, BASE, 0.0)
    flat = _profile(1.0, 1.0)
    results = [(None, {"reason_code": "no_path_found"}), (flat, None)]

    def solve():
        return results.pop(0) if results else (flat, None)

    picked, failure = _pick_path_candidate(solve, attempts=3,
                                           directness_target=target, score=score)
    assert picked is flat and failure is None


def test_when_every_attempt_fails_the_reason_still_reaches_the_agent(monkeypatch):
    _fake_fk(monkeypatch, _z_is_first_joint)
    score, target = _path_score_policy(_FakeModel(), None, BASE, 0.0)
    bad = {"reason_code": "goal_in_collision", "reason": "the fixture is in the way"}
    results = [(None, {"reason_code": "no_path_found"}), (None, bad)]

    picked, failure = _pick_path_candidate(lambda: results.pop(0), attempts=2,
                                           directness_target=target, score=score)
    assert picked is None
    assert failure == bad


def test_the_default_score_is_still_the_proxy():
    """`score=None` must keep every existing caller and test on directness."""
    arc = [np.array([0.0, 0.0]), np.array([0.5, 2.0]), np.array([1.0, 0.0])]
    picked, _ = _pick_path_candidate(lambda: (arc, None), attempts=1,
                                     directness_target=1.6)
    assert picked is arc


# ── the real cell ────────────────────────────────────────────────────────────

def test_the_real_arm_reports_a_plausible_rise_for_a_known_pose():
    """Guards the FK plumbing (nq slice, base/yaw, frame pick) against the model
    the fake above cannot exercise."""
    pytest.importorskip("pinocchio", reason="needs the geometry stack")
    rk = pytest.importorskip("agents.trajectory.tools.robot_kinematics")
    try:
        model, data = rk._load_pinocchio_model("welding_cell_2_small", "robot1")
        cfg = rk._load_robot_config("welding_cell_2_small", "robot1")
    except Exception as e:                      # pragma: no cover - env dependent
        pytest.skip(f"cell welding_cell_2_small unavailable here: {e}")

    base = np.asarray(cfg["base_position"], float)
    yaw = float(cfg.get("base_orientation_yaw", 0.0))
    q0 = np.zeros(model.nq)
    # joint 2 folded up and back down: an out-and-back excursion by construction
    up = q0.copy()
    up[1] = -1.0

    score, target = _path_score_policy(model, model.createData(), base, yaw)
    assert target == pytest.approx(_path_rise_policy()), "FK works, so use metres"

    flat = score([q0, q0, q0])
    arc = score([q0, up, q0])
    assert flat == pytest.approx(0.0)
    assert arc > 0.0, "folding joint 2 up and back must register as an excursion"
    assert arc < 5.0, "a KR240 cannot lift its tool 5 m — the FK wiring is wrong"
