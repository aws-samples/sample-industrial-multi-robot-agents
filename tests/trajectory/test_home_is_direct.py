"""The return home must not wander, and the reasons it used to.

The complaint: "one of the robot return home not directly but is making many
useless micro movement before arriving to the home position". Measured on the two
shipped plans, the SAME leg (robot2, seam -> HOME on ``welding_cell_2_small``):

    plan              travel        detour  reversals  points  duration  rise
    bt_1787955158     4.254 rad     x1.39      4         94     4.62 s   0.585 m
    bt_1788022470    13.929 rad     x4.56     36        362    17.98 s   0.249 m

So it was a REGRESSION, introduced by making tool rise the only thing a route was
judged by. Two owners were involved and each is pinned here:

1. ``plan_home_trajectories`` asked ``inter_robot_clearance`` — "are these legs
   clear if they run AT THE SAME TIME?" — and treated a no as a reason to distort
   the path. But concurrency is ``_gate_multi_arm_step``'s decision, and both
   consumers of the home plan put the emitted legs through it and SERIALISE when
   parallel is unsafe. The flat direct legs measure -0.2133 m in parallel and
   0.7389 m / 1.2699 m serialised robot2 -> robot1 — which is the order the
   shipped tree ran them in anyway.

2. The staging rung compared rise and nothing else, so it bought 0.70 m of rise
   for 9.7 rad of extra travel and logged the trade as a win.

These tests are geometry-free on purpose: they drive the rungs with stubbed
planners so the decisions are pinned deterministically. ``ompl.util.RNG`` is
absent from these bindings, so a real solve cannot be seeded and a test that
asserted on real path shape would be flaky. Real-geometry coverage of the same
leg belongs in the ``slow`` suite.
"""
import numpy as np
import pytest

from agents.trajectory.tools import ompl_planner as op


# ── The measurement itself ────────────────────────────────────────────────────

class TestExcessTravel:
    """``_path_excess_travel`` — the single owner of "did this leg wander?"."""

    def test_a_straight_line_wanders_by_nothing(self):
        path = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]
        assert op._path_excess_travel(path) == pytest.approx(0.0, abs=1e-9)

    def test_a_detour_costs_the_distance_it_added(self):
        # Out one radian and back: the straight line is zero, so ALL of the
        # travel is excess.
        path = [[0.0], [1.0], [0.0]]
        assert op._path_excess_travel(path) == pytest.approx(2.0)

    def test_it_never_goes_negative(self):
        # The straight line is the unreachable lower bound; floating point must
        # not be able to push a two-point path below it.
        path = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0000001]]
        assert op._path_excess_travel(path) >= 0.0

    def test_a_degenerate_path_wanders_by_nothing(self):
        assert op._path_excess_travel([]) == 0.0
        assert op._path_excess_travel([[0.0, 0.0]]) == 0.0

    def test_radians_rank_short_moves_the_way_a_ratio_does_not(self):
        """Why this exists alongside ``_path_directness``.

        W14 scores x3.41 — worse-looking than legs nobody objected to — only
        because its straight-line distance is 0.97 rad. In radians it is 2.35,
        against 10.87 for the leg that was actually reported.
        """
        # A 3x detour on a tiny move, and a 1.5x detour on a huge one.
        small = [[0.0], [0.1], [0.0], [0.1]]
        big = [[0.0], [10.0], [5.0]]
        assert op._path_directness(small) > op._path_directness(big)
        assert op._path_excess_travel(small) < op._path_excess_travel(big)


class TestEmittedExcessTravel:
    """``_trajectory_excess_travel`` measures the array that SHIPS."""

    def test_it_measures_the_timed_array(self):
        traj = [{"positions": [0.0], "time_s": 0.0},
                {"positions": [1.0], "time_s": 1.0},
                {"positions": [0.0], "time_s": 2.0}]
        assert op._trajectory_excess_travel(traj) == pytest.approx(2.0)

    def test_extra_joints_beyond_nq_are_ignored(self):
        # A cell whose trajectory carries more columns than the model has joints
        # must not have the surplus counted as travel.
        traj = [{"positions": [0.0, 9.0], "time_s": 0.0},
                {"positions": [1.0, -9.0], "time_s": 1.0}]
        assert op._trajectory_excess_travel(traj, nq=1) == pytest.approx(0.0)

    def test_a_leg_too_short_to_have_a_shape_wanders_by_nothing(self):
        assert op._trajectory_excess_travel(None) == 0.0
        assert op._trajectory_excess_travel(
            [{"positions": [0.0], "time_s": 0.0}]) == 0.0

    def test_it_can_score_a_concatenation_planner_states_cannot(self):
        """The reason it exists rather than reusing the path-state version.

        A staged home is ``_concat_timed`` of separately-planned segments. There
        is no single set of planner states for it — only the emitted array.
        """
        legs = [[{"positions": [0.0], "time_s": 0.0},
                 {"positions": [2.0], "time_s": 1.0}],
                [{"positions": [2.0], "time_s": 0.0},
                 {"positions": [1.0], "time_s": 1.0}]]
        joined = op._concat_timed(legs)
        # Travel 3.0 over a straight-line 1.0 => 2.0 of excess.
        assert op._trajectory_excess_travel(joined) == pytest.approx(2.0)


class TestStagingTravelPolicy:
    """The allowance is measured, tunable, and cannot forbid staging outright."""

    def test_the_default_is_the_measured_cost_of_a_good_staged_route(self):
        assert op._staging_extra_travel_limit() == pytest.approx(1.2)

    def test_it_is_tunable_without_a_code_change(self, monkeypatch):
        monkeypatch.setenv("OMPL_STAGING_EXTRA_TRAVEL_RAD", "4.0")
        assert op._staging_extra_travel_limit() == pytest.approx(4.0)

    def test_a_negative_allowance_is_floored_not_honoured(self, monkeypatch):
        # A negative allowance would forbid staging even in the case it exists
        # for — an arm with no direct path at any margin.
        monkeypatch.setenv("OMPL_STAGING_EXTRA_TRAVEL_RAD", "-1")
        assert op._staging_extra_travel_limit() == 0.0

    def test_a_junk_value_falls_back_rather_than_raising(self, monkeypatch):
        monkeypatch.setenv("OMPL_STAGING_EXTRA_TRAVEL_RAD", "soon")
        assert op._staging_extra_travel_limit() == pytest.approx(1.2)


# ── The rungs ────────────────────────────────────────────────────────────────

CELL = "welding_cell_2_small"
HOME = [0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0]
SEAM = {"robot1": [-2.372, -0.923, 0.918, 0.611, 2.12, 0.953],
        "robot2": [1.029, -1.39, 1.355, 0.717, 1.981, -1.94]}


def _leg(*qs, dt=1.0):
    return [{"positions": list(q), "time_s": round(i * dt, 4)}
            for i, q in enumerate(qs)]


def _shape(traj):
    return [list(p["positions"]) for p in (traj or [])]


def _rise_of(tall, tall_rise=0.95, flat_rise=0.0):
    """A rise stub keyed on CONTENT, not identity.

    ``_with_head`` returns a fresh list, so the trajectory a rung measures is
    never the object the test handed the planner. An identity check silently
    reports every leg as flat, which makes the rungs unreachable and the tests
    pass for the wrong reason.
    """
    marker = _shape(tall)

    def _rise(model, traj, base, yaw):
        return tall_rise if _shape(traj) == marker else flat_rise

    return _rise


@pytest.fixture
def home_rig(monkeypatch):
    """Drive ``plan_home_trajectories``' rungs with stubs, no geometry.

    Returns a dict the test fills in: which route each planner call yields, what
    the gate says, and what staging produces. Records what was asked.
    """
    rig = {
        "tube_route": None,      # what the swept-volume ladder finds
        "direct_route": None,    # what the obstacle-free solve finds
        "staged": (None, None),
        "gate_mode": "SEQUENTIAL",
        "gate_calls": [],
        "asked": [],
    }

    def fake_plan(cell_id, ns, start, goal, moving_obstacle_spheres=None,
                  velocity_scale=0.4, **kw):
        tube = bool(moving_obstacle_spheres)
        rig["asked"].append({"robot_ns": ns, "with_tube": tube})
        return rig["tube_route"] if tube else rig["direct_route"]

    def fake_gate(cell_id, legs, **kw):
        rig["gate_calls"].append(sorted(legs))
        mode = rig["gate_mode"]
        return {"mode": mode, "order": sorted(legs) if mode else [],
                "parallel": {"min_clearance_m": -0.2133,
                             "warning_distance_m": 0.2}}

    monkeypatch.setattr(op, "plan_joint_path", fake_plan)
    monkeypatch.setattr("agents.trajectory.tools.behaviour_tree."
                        "_gate_multi_arm_step", fake_gate, raising=False)
    monkeypatch.setattr(op, "_plan_staged_home",
                        lambda *a, **k: rig["staged"])
    # No retract: the head is a separate concern with its own tests.
    monkeypatch.setattr(op, "_home_liftoff", lambda *a, **k: [])
    monkeypatch.setattr(op, "_import_ompl", lambda: object())
    monkeypatch.setattr(op, "_home_config", lambda *a, **k: list(HOME))
    monkeypatch.setattr(op, "_trajectory_tool_rise",
                        lambda model, traj, base, yaw: 0.0)

    return rig


class _FakeModel:
    """Just enough model for the parts of the home planner under test.

    ``createData`` matters: the swept-volume accumulation is wrapped in a bare
    ``except``, so a model missing it leaves ``planned_swept`` empty and every
    rung that fires only for an arm with others to avoid becomes unreachable —
    silently.
    """
    nq = 6

    def createData(self):
        return None


@pytest.fixture
def home_geometry(monkeypatch, home_rig):
    """The geometry reads ``plan_home_trajectories`` makes before any rung.

    ``_config_spheres`` must return something: an arm whose swept volume is empty
    never becomes an obstacle, ``planned_swept`` stays empty, and the rung under
    test — which only fires for an arm that has others to avoid — is unreachable.
    """
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_collision_config",
        lambda _cell: {"warning_distance": 0.2}, raising=False)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_pinocchio_model",
        lambda *a, **k: (_FakeModel(), None), raising=False)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_robot_config",
        lambda *a, **k: {"base_position": [0.0, 0.0, 0.0],
                         "base_orientation_yaw": 0.0}, raising=False)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._config_spheres",
        lambda *a, **k: [np.array([0.5, 0.5, 1.0])], raising=False)
    return home_rig


class TestConcurrencyHasOneOwner:
    """The rung must ask ``_gate_multi_arm_step``, not a parallel-only proxy."""

    def test_a_serial_only_direct_route_is_accepted(self, home_geometry,
                                                    monkeypatch):
        """The regression, in one assertion.

        The flat direct legs are NOT clear in parallel (-0.2133 m) but ARE clear
        serialised. The old rung asked ``inter_robot_clearance``, got a no, and
        sent the arm through a x4.56 staging detour. The assembler then serialised
        the step anyway — so the refusal protected against an arrangement nothing
        was going to run.
        """
        rig = home_geometry
        tube = _leg([1.0] * 6, [2.0] * 6, HOME)
        direct = _leg([1.0] * 6, HOME)
        rig["tube_route"], rig["direct_route"] = tube, direct
        rig["gate_mode"] = "SEQUENTIAL"       # parallel refused, serial clear
        monkeypatch.setattr(op, "_trajectory_tool_rise", _rise_of(tube))

        out = op.plan_home_trajectories(CELL, dict(SEAM))

        assert set(out) == {"robot1", "robot2"}
        # robot2 is planned second, so it is the one with a tube to relax.
        assert _shape(out["robot2"]) == _shape(direct), (
            "a route the gate clears SERIALISED was refused as though the arms "
            "had to move at once")
        assert rig["gate_calls"], "the concurrency owner was never asked"

    def test_a_route_no_order_clears_is_refused(self, home_geometry,
                                                monkeypatch):
        """``mode=None`` is a refusal. Fail-closed is not relaxed by this fix."""
        rig = home_geometry
        tube = _leg([1.0] * 6, [2.0] * 6, HOME)
        direct = _leg([1.0] * 6, HOME)
        rig["tube_route"], rig["direct_route"] = tube, direct
        rig["gate_mode"] = None               # no safe order exists
        rig["staged"] = (None, None)
        monkeypatch.setattr(op, "_trajectory_tool_rise", _rise_of(tube))

        out = op.plan_home_trajectories(CELL, dict(SEAM))

        assert _shape(out["robot2"]) == _shape(tube), (
            "a direct route the gate refused outright was shipped anyway")

    def test_a_gate_that_raises_is_a_refusal(self, home_geometry, monkeypatch):
        """A gate that cannot run has not said yes."""
        rig = home_geometry
        tube = _leg([1.0] * 6, [2.0] * 6, HOME)
        direct = _leg([1.0] * 6, HOME)
        rig["tube_route"], rig["direct_route"] = tube, direct
        monkeypatch.setattr(op, "_trajectory_tool_rise", _rise_of(tube))

        def boom(*a, **k):
            raise RuntimeError("no pinocchio")

        monkeypatch.setattr("agents.trajectory.tools.behaviour_tree."
                            "_gate_multi_arm_step", boom, raising=False)

        out = op.plan_home_trajectories(CELL, dict(SEAM))
        assert _shape(out["robot2"]) == _shape(tube)


class TestStagingMustNotBuyHeightWithTravel:
    """The rung that shipped the reported leg."""

    def test_a_flatter_but_wandering_staged_route_is_refused(
            self, home_geometry, monkeypatch):
        """Exactly the trade that produced the complaint.

        The staged route is flatter (0.25 m against 0.95 m) but wanders far
        beyond the allowance. Rise alone said "take it"; travel says otherwise,
        and the arcing route is the lesser of two visible defects.
        """
        rig = home_geometry
        tube = _leg([0.0] * 6, [0.4] * 6, HOME)          # modest travel
        wanderer = _leg([0.0] * 6, [9.0] * 6, [0.2] * 6, HOME)  # far and back
        # robot1 is planned first, so it has no tube to avoid and the
        # obstacle-free solve is the route it gets.
        rig["tube_route"] = rig["direct_route"] = tube
        rig["staged"] = (wanderer, {"waypoints": [[1.0] * 6]})
        monkeypatch.setattr(op, "_trajectory_tool_rise",
                            _rise_of(tube, flat_rise=0.25))

        out = op.plan_home_trajectories(CELL, dict(SEAM))

        assert _shape(out["robot1"]) == _shape(tube), (
            "staging bought 0.70 m of rise with a wandering path and nothing "
            "measured the travel it spent")

    def test_a_flatter_and_compact_staged_route_is_still_taken(
            self, home_geometry, monkeypatch):
        """The guard must not disable staging — it exists for a reason."""
        rig = home_geometry
        tube = _leg([0.0] * 6, [0.4] * 6, HOME)
        staged = _leg([0.0] * 6, [0.1] * 6, HOME)
        rig["tube_route"] = rig["direct_route"] = tube
        rig["staged"] = (staged, {"waypoints": [[0.1] * 6]})
        monkeypatch.setattr(op, "_trajectory_tool_rise",
                            _rise_of(tube, flat_rise=0.25))

        out = op.plan_home_trajectories(CELL, dict(SEAM))
        assert _shape(out["robot1"]) == _shape(staged)

    def test_staging_is_still_the_fallback_when_no_route_exists(
            self, home_geometry, monkeypatch):
        """A wandering staged route beats the executor's BLIND straight line.

        The travel guard applies only where a route already exists to compare
        against. With none, staging is the last checked option and its shape is
        not a reason to fall through to the one motion nothing has verified.
        """
        rig = home_geometry
        wanderer = _leg([0.0] * 6, [9.0] * 6, [0.2] * 6, HOME)
        rig["tube_route"], rig["direct_route"] = None, None
        rig["staged"] = (wanderer, {"waypoints": [[1.0] * 6]})

        out = op.plan_home_trajectories(CELL, dict(SEAM))
        assert _shape(out["robot1"]) == _shape(wanderer)
