"""The quality report has to say the same thing a human said by hand.

The module exists because ``bt_1788022470_welding_cell_2_small`` passed every gate and
was still slow, and finding out why took a person an afternoon of scripts. These
tests pin the conclusions that afternoon reached, so the report cannot quietly
stop reaching them:

    makespan               58.24 s   (structural, honouring Sequence/Parallel)
    parallel efficiency    0.679
    robot1 idle            29.43 s   (50.5 % of the program)
    home_robot2 excess     10.87 rad (the leg an operator complained about)
    weld_robot2_W14 excess  2.35 rad (the leg that only LOOKS bad as a ratio)

Most tests here are geometry-free — hand-built waypoint lists whose right answer
can be computed on paper — because a metric that needs a real solve to test is a
metric nobody can reason about. The two that read the shipped plan are marked and
skip when it is absent, so the suite still runs on a checkout without cells.
"""
import json
import math

import pytest

from agents.trajectory.tools import trajectory_quality as tq
from utils import cell_layout


PLAN_CELL = "welding_cell_2_small"
PLAN_TREE = "bt_1788022470_welding_cell_2_small"


def wp(positions, t, phase="transit"):
    return {"positions": list(positions), "time_s": float(t), "phase": phase}


def straight_leg(n=11, span=1.0, duration=1.0):
    """A single-joint move from 0 to ``span``, evenly in space and time."""
    return [wp([span * i / (n - 1), 0.0], duration * i / (n - 1))
            for i in range(n)]


# ── per-leg path metrics ──────────────────────────────────────────────────────

class TestDetourIsMeasuredAgainstTheStraightLine:
    """The metric that found the real defect, and the only one it could have."""

    def test_a_straight_move_has_no_detour_and_no_excess(self):
        m = tq.leg_metrics(straight_leg())
        assert m["detour_ratio"] == pytest.approx(1.0)
        assert m["excess_travel_rad"] == pytest.approx(0.0, abs=1e-9)
        assert m["detour_verdict"] == tq.VERDICT_GOOD

    def test_a_detour_costs_exactly_what_it_wanders(self):
        # Out to 1.0, back to 0.0, on to 1.0: travels 3.0 to cover 1.0.
        legs = [wp([0.0, 0.0], 0.0), wp([1.0, 0.0], 1.0),
                wp([0.0, 0.0], 2.0), wp([1.0, 0.0], 3.0)]
        m = tq.leg_metrics(legs)
        assert m["joint_travel_l2_rad"] == pytest.approx(3.0)
        assert m["direct_distance_l2_rad"] == pytest.approx(1.0)
        assert m["detour_ratio"] == pytest.approx(3.0)
        assert m["excess_travel_rad"] == pytest.approx(2.0)

    def test_a_leg_that_returns_home_has_no_ratio_but_still_has_excess(self):
        # Zero direct distance: the ratio is undefined, and reporting a number
        # would let one leg dominate every average. The radians stay valid.
        legs = [wp([0.0, 0.0], 0.0), wp([1.0, 0.0], 1.0), wp([0.0, 0.0], 2.0)]
        m = tq.leg_metrics(legs)
        assert m["detour_ratio"] is None
        assert m["excess_travel_rad"] == pytest.approx(2.0)

    def test_the_verdict_follows_radians_not_the_ratio(self):
        """A short 3x transit must not outrank a long wandering leg.

        This is the whole argument in ``ompl_planner._path_excess_travel``, and it
        is the reason a ratio is not the gate. Pinned because scoring by ratio is
        the obvious-looking thing a future change would reach for.
        """
        # 3x ratio, but only 0.2 rad of wander.
        short = [wp([0.0, 0.0], 0.0), wp([0.2, 0.0], 1.0),
                 wp([0.0, 0.0], 2.0), wp([0.1, 0.0], 3.0)]
        # 1.5x ratio, but 5 rad of wander.
        long = [wp([0.0, 0.0], 0.0), wp([12.5, 0.0], 1.0), wp([10.0, 0.0], 2.0)]
        m_short, m_long = tq.leg_metrics(short), tq.leg_metrics(long)
        assert m_short["detour_ratio"] > m_long["detour_ratio"]
        assert m_short["excess_travel_rad"] < m_long["excess_travel_rad"]
        assert m_short["detour_verdict"] == tq.VERDICT_GOOD
        assert m_long["detour_verdict"] == tq.VERDICT_POOR

    def test_waypoint_density_reports_points_per_radian(self):
        m = tq.leg_metrics(straight_leg(n=11, span=1.0))
        assert m["waypoints_per_rad"] == pytest.approx(11.0, rel=1e-3)


class TestLimitUtilisation:
    """Slow and long are different defects, so they get different numbers."""

    def test_utilisation_is_the_peak_against_the_declared_limit(self):
        # 1 rad in 1 s => 1.0 rad/s peak; a 2.0 rad/s limit is half used.
        m = tq.leg_metrics(straight_leg(n=2, span=1.0, duration=1.0),
                           joint_velocity_limits=[2.0, 2.0])
        assert m["speed_utilisation_peak"] == pytest.approx(0.5)

    def test_no_limits_means_no_answer_rather_than_a_guessed_one(self):
        m = tq.leg_metrics(straight_leg())
        assert m["speed_utilisation_peak"] is None
        assert m["accel_utilisation_peak"] is None

    def test_a_zero_limit_is_ignored_not_divided_by(self):
        m = tq.leg_metrics(straight_leg(n=2, span=1.0, duration=1.0),
                           joint_velocity_limits=[0.0, 2.0])
        assert m["speed_utilisation_peak"] is not None
        assert math.isfinite(m["speed_utilisation_peak"])


class TestDwell:
    """Stillness in the MIDDLE of a leg is the loss; at the ends it is a ramp."""

    def test_a_stall_between_two_motions_is_interior_dwell(self):
        legs = [wp([0.0, 0.0], 0.0), wp([1.0, 0.0], 1.0),
                wp([1.0, 0.0], 3.0),           # stalled 2 s
                wp([2.0, 0.0], 4.0)]
        m = tq.leg_metrics(legs)
        assert m["interior_dwell_s"] == pytest.approx(2.0)
        assert m["interior_dwell_segments"] == 1

    def test_stillness_at_the_ends_is_not_counted_as_interior(self):
        legs = [wp([0.0, 0.0], 0.0), wp([0.0, 0.0], 1.0),   # start ramp
                wp([1.0, 0.0], 2.0),
                wp([1.0, 0.0], 5.0)]                        # settle / weld hold
        m = tq.leg_metrics(legs)
        assert m["interior_dwell_s"] == pytest.approx(0.0)
        assert m["dwell_total_s"] == pytest.approx(4.0)

    def test_the_weld_hold_lengthens_the_leg_without_being_dwell(self):
        m = tq.leg_metrics(straight_leg(duration=2.0), hold_duration_s=0.5)
        assert m["duration_s"] == pytest.approx(2.5)
        assert m["interior_dwell_s"] == pytest.approx(0.0)


class TestMalformedInput:
    """A partly-broken trajectory gets a report, not an exception."""

    def test_too_few_waypoints_is_reported_as_unusable(self):
        m = tq.leg_metrics([wp([0.0], 0.0)])
        assert m["usable"] is False
        assert "note" in m

    def test_ragged_and_junk_waypoints_are_dropped_and_counted(self):
        legs = [wp([0.0, 0.0], 0.0), {"positions": [0.5], "time_s": 0.5},
                {"time_s": 0.7}, wp([1.0, 0.0], 1.0)]
        m = tq.leg_metrics(legs)
        assert m["usable"] is True
        assert m["waypoints_dropped"] == 2
        assert m["detour_ratio"] == pytest.approx(1.0)

    def test_a_non_advancing_timestamp_is_counted_not_divided_by(self):
        legs = [wp([0.0, 0.0], 0.0), wp([1.0, 0.0], 0.0), wp([2.0, 0.0], 1.0)]
        m = tq.leg_metrics(legs)
        assert m["non_monotonic_steps"] == 1
        assert math.isfinite(m["dimensionless_jerk"])


class TestSmoothnessIsScaleInvariant:
    def test_the_same_shape_slower_and_longer_scores_the_same(self):
        """Dimensionless jerk is the point: legs of different size compare.

        An absolute jerk number would rank a big fast leg worse than a small slow
        one of identical shape, and the report would flag arms rather than paths.
        """
        base = [wp([0.0, 0.0], 0.0), wp([0.1, 0.0], 1.0), wp([0.0, 0.0], 2.0),
                wp([0.3, 0.0], 3.0)]
        scaled = [wp([p["positions"][0] * 4, 0.0], p["time_s"] * 3)
                  for p in base]
        a = tq.leg_metrics(base)["dimensionless_jerk"]
        b = tq.leg_metrics(scaled)["dimensionless_jerk"]
        assert a > 0
        assert b == pytest.approx(a, rel=1e-6)


class TestCartesianMetricsAreOptional:
    def test_without_fk_no_cartesian_fields_are_invented(self):
        assert "cartesian_path_m" not in tq.leg_metrics(straight_leg())

    def test_with_fk_the_tool_path_is_measured_too(self):
        m = tq.leg_metrics(straight_leg(n=3, span=1.0),
                           fk=lambda q: (q[0], 0.0, 0.0))
        assert m["cartesian_path_m"] == pytest.approx(1.0)
        assert m["cartesian_detour_ratio"] == pytest.approx(1.0)

    def test_a_failing_fk_callback_does_not_fail_the_report(self):
        def boom(_q):
            raise RuntimeError("no kinematic model here")
        m = tq.leg_metrics(straight_leg(), fk=boom)
        assert m["usable"] is True
        assert "cartesian_path_m" not in m


# ── the schedule ─────────────────────────────────────────────────────────────

def exec_node(name, robot_ns, key, op_id=None):
    return {"type": "ExecuteTrajectory", "name": name,
            "ports": {"robot_ns": robot_ns, "operation_id": op_id or name,
                      "trajectory": "{%s}" % key}}


def tree(root, blackboard, tree_id="t1", cell_id="c1"):
    return {"tree_id": tree_id, "cell_id": cell_id, "root": root,
            "blackboard": blackboard}


class TestScheduleHonoursTreeStructure:
    """The makespan is structural. A max over single operations is not one."""

    def test_parallel_children_overlap(self):
        bb = {"a": straight_leg(duration=4.0), "b": straight_leg(duration=3.0)}
        t = tree({"type": "Parallel",
                  "children": [exec_node("A", "robot1", "a"),
                               exec_node("B", "robot2", "b")]}, bb)
        m = tq.schedule_metrics(t)
        assert m["makespan_s"] == pytest.approx(4.0)
        assert m["concurrent_s"] == pytest.approx(3.0)
        assert m["single_arm_s"] == pytest.approx(1.0)

    def test_sequence_children_do_not(self):
        bb = {"a": straight_leg(duration=4.0), "b": straight_leg(duration=3.0)}
        t = tree({"type": "Sequence",
                  "children": [exec_node("A", "robot1", "a"),
                               exec_node("B", "robot2", "b")]}, bb)
        m = tq.schedule_metrics(t)
        assert m["makespan_s"] == pytest.approx(7.0)
        assert m["concurrent_s"] == pytest.approx(0.0)
        assert m["serial_fraction"] == pytest.approx(1.0)

    def test_parallel_efficiency_is_busy_over_arms_times_makespan(self):
        bb = {"a": straight_leg(duration=4.0), "b": straight_leg(duration=2.0)}
        t = tree({"type": "Parallel",
                  "children": [exec_node("A", "robot1", "a"),
                               exec_node("B", "robot2", "b")]}, bb)
        m = tq.schedule_metrics(t)
        assert m["parallel_efficiency"] == pytest.approx(6.0 / (2 * 4.0))
        assert m["load_balance"] == pytest.approx(0.5)
        assert m["idle_s"]["robot2"] == pytest.approx(2.0)

    def test_a_delay_pushes_its_child_out(self):
        bb = {"a": straight_leg(duration=1.0)}
        t = tree({"type": "Delay", "ports": {"delay_msec": 500},
                  "children": [exec_node("A", "robot1", "a")]}, bb)
        assert tq.schedule_metrics(t)["makespan_s"] == pytest.approx(1.5)

    def test_an_unknown_control_node_is_read_as_a_sequence(self):
        """Pessimistic on purpose: an unknown node must never shorten a makespan.

        Reading it as Parallel would make a plan with a node this module has not
        heard of look faster than it is — an efficiency report that flatters the
        plan is worse than no report.
        """
        bb = {"a": straight_leg(duration=2.0), "b": straight_leg(duration=2.0)}
        t = tree({"type": "SomeNodeInventedLater",
                  "children": [exec_node("A", "robot1", "a"),
                               exec_node("B", "robot2", "b")]}, bb)
        assert tq.schedule_metrics(t)["makespan_s"] == pytest.approx(4.0)

    def test_two_legs_of_one_arm_overlapping_is_not_parallelism(self):
        """Concurrency counts distinct ARMS. One arm twice over is a bug."""
        bb = {"a": straight_leg(duration=3.0), "b": straight_leg(duration=3.0)}
        t = tree({"type": "Parallel",
                  "children": [exec_node("A", "robot1", "a"),
                               exec_node("B", "robot1", "b")]}, bb)
        m = tq.schedule_metrics(t)
        assert m["concurrent_s"] == pytest.approx(0.0)
        assert m["single_arm_s"] == pytest.approx(3.0)

    def test_gate_nodes_have_no_duration_of_their_own(self):
        bb = {"a": straight_leg(duration=2.0)}
        t = tree({"type": "Sequence", "children": [
            {"type": "CheckSafety", "ports": {"robot_ns": "robot1"}},
            {"type": "WaitForRobot", "ports": {"interlock_target": "robot2"}},
            exec_node("A", "robot1", "a")]}, bb)
        m = tq.schedule_metrics(t)
        assert m["makespan_s"] == pytest.approx(2.0)
        assert m["motion_legs"] == 1

    def test_unpaired_legs_are_named(self):
        bb = {"a": straight_leg(duration=2.0), "b": straight_leg(duration=2.0),
              "c": straight_leg(duration=2.0)}
        t = tree({"type": "Sequence", "children": [
            {"type": "Parallel", "children": [exec_node("A", "robot1", "a"),
                                              exec_node("B", "robot2", "b")]},
            exec_node("C", "robot2", "c", op_id="W99")]}, bb)
        m = tq.schedule_metrics(t)
        assert [u["operation_id"] for u in m["unpaired_legs"]] == ["W99"]
        assert m["unpaired_time_s"] == pytest.approx(2.0)

    def test_a_tree_with_no_motion_says_so_instead_of_dividing_by_zero(self):
        assert tq.schedule_metrics({"root": {"type": "Sequence"}})["available"] is False
        assert tq.schedule_metrics({})["available"] is False


class TestContinuity:
    def test_a_matching_seam_is_continuous(self):
        bb = {"a": [wp([0.0, 0.0], 0.0), wp([1.0, 0.0], 1.0)],
              "b": [wp([1.0, 0.0], 0.0), wp([2.0, 0.0], 1.0)]}
        t = tree({"type": "Sequence", "children": [exec_node("A", "r1", "a"),
                                                   exec_node("B", "r1", "b")]}, bb)
        c = tq.continuity_metrics(t)
        assert c["continuous"] is True
        assert c["max_seam_gap_rad"] == pytest.approx(0.0)

    def test_a_jump_between_legs_of_one_arm_is_a_break(self):
        bb = {"a": [wp([0.0, 0.0], 0.0), wp([1.0, 0.0], 1.0)],
              "b": [wp([1.5, 0.0], 0.0), wp([2.0, 0.0], 1.0)]}
        t = tree({"type": "Sequence", "children": [exec_node("A", "r1", "a"),
                                                   exec_node("B", "r1", "b")]}, bb)
        c = tq.continuity_metrics(t)
        assert c["continuous"] is False
        assert c["max_seam_gap_rad"] == pytest.approx(0.5)
        assert c["breaks"][0]["to_operation"] == "B"

    def test_different_arms_do_not_form_a_seam(self):
        bb = {"a": [wp([0.0, 0.0], 0.0), wp([1.0, 0.0], 1.0)],
              "b": [wp([9.0, 0.0], 0.0), wp([9.5, 0.0], 1.0)]}
        t = tree({"type": "Sequence", "children": [exec_node("A", "r1", "a"),
                                                   exec_node("B", "r2", "b")]}, bb)
        assert tq.continuity_metrics(t)["continuous"] is True


# ── the composite score ──────────────────────────────────────────────────────

class TestScorePlan:
    def test_a_perfect_two_arm_plan_scores_one(self):
        bb = {"a": straight_leg(duration=3.0), "b": straight_leg(duration=3.0)}
        t = tree({"type": "Parallel", "children": [exec_node("A", "r1", "a"),
                                                   exec_node("B", "r2", "b")]}, bb)
        r = tq.score_plan(t)
        assert r["efficiency_score"] == pytest.approx(1.0)
        assert r["verdict"] == tq.VERDICT_GOOD
        assert r["findings"] == []

    def test_the_weights_sum_to_one_so_the_score_is_a_fraction(self):
        assert sum(tq.WEIGHTS.values()) == pytest.approx(1.0)

    def test_every_component_is_reported_beside_the_score(self):
        bb = {"a": straight_leg(duration=3.0)}
        r = tq.score_plan(tree(exec_node("A", "r1", "a"), bb))
        assert set(r["score_components"]) == set(tq.WEIGHTS)
        assert r["score_weights"] == tq.WEIGHTS

    def test_safety_is_reported_and_never_scored(self):
        """A score is a thing people optimise. Margins must not be in one.

        If a clearance ever earns points, a shorter path that passes closer to a
        fixture starts winning — which is exactly the trade nobody is allowed to
        make automatically.
        """
        assert "safety" not in tq.WEIGHTS
        bb = {"a": straight_leg(duration=1.0)}
        r = tq.score_plan(tree(exec_node("A", "r1", "a"), bb))
        assert "safety" in r
        assert "safety" not in r["score_components"]

    def test_the_path_score_is_aggregate_not_a_mean_of_ratios(self):
        """One short wandering leg must not sink a plan of good long ones."""
        bb = {"long": straight_leg(n=3, span=10.0, duration=5.0),
              "short": [wp([0.0, 0.0], 0.0), wp([0.1, 0.0], 0.5),
                        wp([0.0, 0.0], 1.0), wp([0.05, 0.0], 1.5)]}
        t = tree({"type": "Parallel", "children": [exec_node("L", "r1", "long"),
                                                    exec_node("S", "r2", "short")]},
                 bb)
        r = tq.score_plan(t)
        assert r["score_components"]["path"] > 0.9

    def test_a_plan_envelope_and_a_bare_tree_score_the_same(self):
        bb = {"a": straight_leg(duration=2.0)}
        t = tree(exec_node("A", "r1", "a"), bb)
        env = {"tree_id": "t1", "cell_id": "c1", "status": "ok",
               "behaviour_tree": t}
        assert (tq.score_plan(env)["efficiency_score"]
                == tq.score_plan(t)["efficiency_score"])

    def test_robot_limits_are_found_at_either_nesting_depth(self):
        flat = {"robot1": {"joint_velocity_limits": [1.0], }}
        nested = {"robots": {"arms": {"robot1": {"joint_velocity_limits": [1.0]}}}}
        assert "robot1" in tq._robot_limits(flat)
        assert "robot1" in tq._robot_limits(nested)

    def test_the_report_is_json_serialisable(self):
        bb = {"a": straight_leg(duration=2.0)}
        r = tq.score_plan(tree(exec_node("A", "r1", "a"), bb))
        json.dumps(r)   # must not raise


class TestFindingsAreRanked:
    def test_a_wandering_leg_outranks_a_slow_one(self):
        bb = {"a": [wp([0.0, 0.0], 0.0), wp([9.0, 0.0], 2.0),
                    wp([1.0, 0.0], 4.0)],
              "b": straight_leg(duration=5.0)}
        t = tree({"type": "Parallel", "children": [exec_node("A", "r1", "a"),
                                                    exec_node("B", "r2", "b")]}, bb)
        r = tq.score_plan(t)
        kinds = [f["kind"] for f in r["findings"]]
        assert kinds[0] == "path_detour"
        assert r["findings"][0]["severity"] == "high"

    def test_low_parallelism_names_the_idle_arm(self):
        bb = {"a": straight_leg(duration=8.0), "b": straight_leg(duration=1.0)}
        t = tree({"type": "Sequence", "children": [exec_node("A", "r1", "a"),
                                                    exec_node("B", "r2", "b")]}, bb)
        f = [x for x in tq.score_plan(t)["findings"]
             if x["kind"] == "low_parallelism"]
        assert f and "r2" in f[0]["detail"]

    def test_a_signal_on_most_legs_becomes_one_plan_finding(self):
        """Seventeen copies of one fact buries the findings that name a leg."""
        bb = {f"k{i}": straight_leg(n=2, span=0.1, duration=1.0)
              for i in range(4)}
        t = tree({"type": "Parallel",
                  "children": [exec_node(f"N{i}", f"r{i}", f"k{i}")
                               for i in range(4)]}, bb)
        robots = {f"r{i}": {"joint_velocity_limits": [10.0, 10.0]}
                  for i in range(4)}
        kinds = [f["kind"] for f in tq.score_plan(t, robots=robots)["findings"]]
        assert kinds.count("slow_plan") == 1
        assert "slow_leg" not in kinds


# ── against the plan that started this ───────────────────────────────────────

@pytest.fixture
def shipped_plan():
    path = cell_layout.plan_file(PLAN_CELL, PLAN_TREE)
    if not path.is_file():
        pytest.skip(f"{path} not present in this checkout")
    return json.load(open(path))


class TestTheShippedPlanReproducesTheHandMeasurement:
    """The numbers a person derived by hand, now derived by the module.

    If these move, either the plan file changed or a metric's meaning did, and
    both are things a reader of an efficiency report needs to know about.
    """

    def test_the_makespan_is_structural(self, shipped_plan):
        m = tq.schedule_metrics(shipped_plan["behaviour_tree"])
        assert m["makespan_s"] == pytest.approx(58.24, abs=0.05)

    def test_the_parallel_efficiency_and_idle_time_match(self, shipped_plan):
        m = tq.schedule_metrics(shipped_plan["behaviour_tree"])
        assert m["parallel_efficiency"] == pytest.approx(0.679, abs=0.005)
        assert m["idle_s"]["robot1"] == pytest.approx(29.43, abs=0.05)
        assert m["utilisation"]["robot1"] == pytest.approx(0.495, abs=0.005)

    def test_the_home_leg_an_operator_complained_about_is_the_top_finding(
            self, shipped_plan):
        r = tq.score_plan(shipped_plan)
        top = r["findings"][0]
        assert top["kind"] == "path_detour"
        assert top["node"] == "home_robot2"
        # 10.87 rad is the figure ompl_planner's calibration records for this leg.
        assert top["value"] == pytest.approx(10.87, abs=0.05)

    def test_the_short_transit_ranks_below_it_despite_a_worse_ratio(
            self, shipped_plan):
        r = tq.score_plan(shipped_plan)
        legs = {l["node"]: l for l in r["legs"]}
        w14, home = legs["weld_robot2_W14"], legs["home_robot2"]
        assert w14["excess_travel_rad"] == pytest.approx(2.35, abs=0.05)
        assert w14["detour_ratio"] < home["detour_ratio"]
        assert w14["excess_travel_rad"] < home["excess_travel_rad"]

    def test_the_single_arm_tail_is_reported_as_unpaired(self, shipped_plan):
        m = tq.schedule_metrics(shipped_plan["behaviour_tree"])
        unpaired = {u["operation_id"] for u in m["unpaired_legs"]}
        assert {"W13", "W14", "W15"} <= unpaired

    def test_the_plan_is_continuous(self, shipped_plan):
        c = tq.continuity_metrics(shipped_plan["behaviour_tree"])
        assert c["continuous"] is True


# ── the log file ─────────────────────────────────────────────────────────────

class TestQualityLogPath:
    def test_it_shares_the_basename_with_the_plan_and_the_recording(self):
        cell, tree_id = "c", "bt_42_c"
        assert (cell_layout.quality_log_file(cell, tree_id).stem
                == cell_layout.plan_file(cell, tree_id).stem
                == cell_layout.recording_file(cell, tree_id).stem)
        assert cell_layout.quality_log_file(cell, tree_id).suffix == ".json"

    def test_it_lives_in_logs_and_cannot_collide_with_a_run_log(self):
        q = cell_layout.quality_log_file("c", "bt_42_c")
        assert q.parent == cell_layout.logs_dir("c")
        assert q != cell_layout.run_log_file("c", "bt_42_c")


class TestWriteQualityLog:
    def test_it_writes_the_report_under_the_tree_id(self, tmp_path):
        bb = {"a": straight_leg(duration=2.0)}
        plan = {"tree_id": "bt_1_c", "cell_id": "c", "status": "ok",
                "behaviour_tree": tree(exec_node("A", "r1", "a"), bb,
                                       tree_id="bt_1_c")}
        out = tq.write_quality_log("c", plan, cell_base=str(tmp_path))
        assert out is not None and out.is_file()
        assert out.name == "bt_1_c.json"
        report = json.load(open(out))
        assert report["tree_id"] == "bt_1_c"
        assert report["cell_id"] == "c"
        assert "efficiency_score" in report and "findings" in report

    def test_a_plan_without_a_tree_id_writes_nothing(self, tmp_path):
        assert tq.write_quality_log("c", {"behaviour_tree": {}},
                                    cell_base=str(tmp_path)) is None

    def test_a_broken_plan_never_raises(self, tmp_path):
        """Quality measurement must not be able to lose a computed plan."""
        plan = {"tree_id": "bt_2_c", "behaviour_tree": {"root": "not a dict"}}
        assert tq.write_quality_log("c", plan, cell_base=str(tmp_path)) is not None

    def test_saving_a_plan_writes_its_quality_log(self, tmp_path, monkeypatch):
        from agents.trajectory.tools import plan_store
        monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
        bb = {"a": straight_leg(duration=2.0)}
        bt = tree(exec_node("A", "r1", "a"), bb, tree_id="bt_3_c", cell_id="c")
        assert plan_store.save_plan("c", bt) == "bt_3_c"
        assert cell_layout.quality_log_file("c", "bt_3_c").is_file()

    def test_a_failing_quality_log_does_not_fail_the_save(self, tmp_path,
                                                          monkeypatch):
        from agents.trajectory.tools import plan_store
        monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))

        def boom(*_a, **_k):
            raise RuntimeError("measurement exploded")
        monkeypatch.setattr(tq, "write_quality_log", boom)
        bb = {"a": straight_leg(duration=2.0)}
        bt = tree(exec_node("A", "r1", "a"), bb, tree_id="bt_4_c", cell_id="c")
        assert plan_store.save_plan("c", bt) == "bt_4_c"
        assert cell_layout.plan_file("c", "bt_4_c").is_file()
