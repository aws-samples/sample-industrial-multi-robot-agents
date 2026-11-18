"""Returning HOME must clear the same gate a weld step clears.

``plan_home_trajectory`` plans ONE arm against static obstacles only. The home
reset drives every arm home at once, so two arms can each hold an individually
valid path that sweeps through the other's — and the runtime monitor e-stops on
the result. ``plan_home_trajectories`` plans them together, feeding each planned
arm's swept volume forward as an obstacle, so the set is safe at any relative
timing.

These tests measure that on the REAL cell geometry with the REAL collision model
(link spheres plus the welding torch), against the cell's own
``warning_distance`` — the threshold the monitor actually e-stops at, not merely
non-penetration.

They need Pinocchio and OMPL, so they skip on a bare host and run in the
container. That is deliberate: the property being pinned is geometric, and a
mocked geometry would pin nothing.

Real planning is slow (seconds per arm, and a margin that admits no path burns
the whole OMPL time budget before relaxing), so every scenario is planned ONCE in
a module-scoped fixture and the assertions read from the cached result. Marked
``slow`` — deselect with ``-m 'not slow'``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"
ROBOTS = ["robot1", "robot2"]

pytestmark = pytest.mark.slow

pytest.importorskip("pinocchio", reason="needs the ROS/Pinocchio runtime")
pytest.importorskip("numpy")

# Ordinary working poses, arms reaching inboard toward the shared workpiece.
WORKING_Q = {
    "robot1": [0.6, -1.2, 1.4, 0.0, -0.6, 0.0],
    "robot2": [-0.6, -1.2, 1.4, 0.0, -0.6, 0.0],
}

# The pose the cell was left in by the recorded collision e-stop — the arms are
# 0.201 m apart, i.e. already at the warning gate. Homing FROM here is the real
# worst case: it is exactly the state an operator hits "reset" in.
E_STOP_Q = {
    "robot1": [-1.3443731730742334, -1.6649834138275557, 1.9064940997790765,
               1.8325230253662836, -1.238424167023632, -1.2285428267147602],
    "robot2": [1.373152767803842, -1.2407160626056504, 1.42134194846956,
               1.7840768773281168, -1.4925876259968578, 1.339554441936049],
}


@pytest.fixture(scope="module")
def ompl_available():
    from agents.trajectory.backends.ompl.planner import _import_ompl
    if _import_ompl() is None:
        pytest.skip("OMPL not installed")


@pytest.fixture(scope="module")
def warning_distance():
    from agents.trajectory.tools.robot_kinematics import _load_collision_config
    return _load_collision_config(CELL)["warning_distance"]


@pytest.fixture(scope="module")
def homed_from_working(ompl_available):
    """The coordinated reset from ordinary working poses — planned once."""
    from agents.trajectory.backends.ompl.planner import plan_home_trajectories
    return plan_home_trajectories(CELL, WORKING_Q)


@pytest.fixture(scope="module")
def homed_from_estop(ompl_available):
    """The coordinated reset out of the recorded e-stop pose — planned once."""
    from agents.trajectory.backends.ompl.planner import plan_home_trajectories
    return plan_home_trajectories(CELL, E_STOP_Q)


@pytest.fixture(scope="module")
def homed_uncoupled_from_estop(ompl_available):
    """Each arm planned ALONE out of the e-stop pose — the coupling baseline."""
    from agents.trajectory.backends.ompl.planner import (
        HOME_JOINTS_DEFAULT, _home_config, plan_joint_path,
    )
    out = {}
    for ns, q in E_STOP_Q.items():
        t = plan_joint_path(CELL, ns, q,
                            _home_config(CELL, ns, HOME_JOINTS_DEFAULT),
                            velocity_scale=0.4)
        if t:
            out[ns] = t
    return out


_GEO_CACHE: dict = {}


def _geometry(robot_ns: str):
    """Model, data, base pose and tool config for one arm — the monitor's view.

    Cached: URDF parsing dominates the cost of measuring clearance.
    """
    import numpy as np

    from agents.trajectory.tools.robot_kinematics import (
        _load_pinocchio_model, _load_robot_config,
    )
    if robot_ns not in _GEO_CACHE:
        model, _ = _load_pinocchio_model(CELL, robot_ns)
        rcfg = _load_robot_config(CELL, robot_ns)
        _GEO_CACHE[robot_ns] = {
            "model": model,
            "data": model.createData(),
            "base": np.array(rcfg["base_position"]),
            "yaw": float(rcfg.get("base_orientation_yaw", 0.0)),
            # The torch counts. Omitting it here is exactly the discrepancy that
            # let a "collision-free" plan get e-stopped on the floor.
            "tool": rcfg.get("tool_collision"),
        }
    return _GEO_CACHE[robot_ns]


def _min_clearance_simultaneous(trajs: dict, samples: int = 60) -> float:
    """Worst clearance between two arms playing their returns at the same time.

    Sampled on a shared wall clock, so it measures the arms as the monitor sees
    them: two motions running concurrently, not two paths in isolation.
    """
    import numpy as np

    from agents.trajectory.tools.robot_kinematics import (
        _check_robot_pair_collision, _config_spheres,
    )
    from agents.trajectory.tools.time_parameterization import _sample_at_time

    geo = {ns: _geometry(ns) for ns in trajs}
    horizon = max(float(t[-1]["time_s"]) for t in trajs.values())
    worst = float("inf")

    for k in range(samples + 1):
        t = horizon * k / samples
        spheres = {}
        for ns, traj in trajs.items():
            q = np.array(_sample_at_time(traj, t)[:geo[ns]["model"].nq])
            g = geo[ns]
            spheres[ns] = _config_spheres(
                g["model"], g["data"], q, g["base"], g["yaw"], g["tool"])
        names = sorted(spheres)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                _, clear = _check_robot_pair_collision(
                    spheres[names[i]], spheres[names[j]])
                worst = min(worst, clear)
    return worst


# ── the coordinated return is genuinely collision-free ───────────────────────

def _clearance_at(q_by_robot: dict) -> float:
    """Clearance with the arms standing still at the given configurations."""
    import numpy as np

    from agents.trajectory.tools.robot_kinematics import (
        _check_robot_pair_collision, _config_spheres,
    )
    geo = {ns: _geometry(ns) for ns in q_by_robot}
    sp = {}
    for ns, q in q_by_robot.items():
        g = geo[ns]
        sp[ns] = _config_spheres(
            g["model"], g["data"], np.array(q[:g["model"].nq]),
            g["base"], g["yaw"], g["tool"])
    names = sorted(sp)
    return _check_robot_pair_collision(sp[names[0]], sp[names[1]])[1]


def test_a_well_separated_home_reset_clears_the_monitors_gate(
        homed_from_working, warning_distance):
    """From ordinary working poses, both arms go home with margin to spare."""
    assert sorted(homed_from_working) == ROBOTS, (
        f"both arms must be planned, got {sorted(homed_from_working)}")

    worst = _min_clearance_simultaneous(homed_from_working)
    assert worst > warning_distance, (
        f"simultaneous home returns close to {worst:.3f} m, at or inside the "
        f"cell's warning_distance of {warning_distance} m — the runtime monitor "
        f"would e-stop this reset")


def test_homing_out_of_an_estop_pose_never_closes_the_gap(homed_from_estop):
    """The reset must not make a bad situation worse.

    Started already at the warning gate, no return can OPEN the gap beyond where
    it began — the minimum over the motion is bounded by the standing clearance.
    So the property that matters is that homing does not close it further: the
    arms separate from here, they do not converge first.
    """
    standing = _clearance_at(E_STOP_Q)

    assert sorted(homed_from_estop) == ROBOTS, (
        f"both arms must still be planned out of an e-stop pose, got "
        f"{sorted(homed_from_estop)} — an unplanned arm falls back to the BLIND "
        f"home, which is the more dangerous motion")

    # The set runs the way the step gate decides (every consumer puts it through
    # _gate_multi_arm_step): together when that is clear, else one arm after the
    # other in a proven order — and the planner admits a route on that verdict.
    # Only a set that runs TOGETHER is judged by its simultaneous clearance.
    from agents.trajectory.tools.behaviour_tree import _gate_multi_arm_step
    report = _gate_multi_arm_step(CELL, homed_from_estop)
    assert report.get("mode") is not None, "no order homes the arms clear of each other"
    if report["mode"] != "PARALLEL":
        pytest.skip("this run ships the returns one arm after the other: no "
                    "simultaneous motion to measure (the gate above is the check)")
    worst = _min_clearance_simultaneous(homed_from_estop)
    assert worst >= standing - 0.01, (
        f"the coordinated return dips to {worst:.3f} m from a standing "
        f"{standing:.3f} m — the arms converge before separating, so the reset "
        f"is more dangerous than standing still")


def test_coupling_is_what_keeps_the_estop_reset_safe(homed_from_estop,
                                                     homed_uncoupled_from_estop):
    """Quantifies the coupling: uncoupled returns from here converge sharply.

    Planned individually, each arm holds a valid path that sweeps through where
    the other goes. This is the failure the coupling exists to prevent, measured
    rather than asserted.
    """
    if sorted(homed_uncoupled_from_estop) != ROBOTS:
        pytest.skip("uncoupled baseline unavailable")
    if sorted(homed_from_estop) != ROBOTS:
        pytest.fail("coordinated planning failed where uncoupled succeeded")

    from agents.trajectory.tools.behaviour_tree import _gate_multi_arm_step
    if _gate_multi_arm_step(CELL, homed_from_estop).get("mode") != "PARALLEL":
        # Shipped to run one arm after the other (the planner's gate-verified
        # direct route): there is no simultaneous motion for coupling to protect.
        pytest.skip("this run ships the returns serialised: coupling is not "
                    "what keeps it safe")
    worst_uncoupled = _min_clearance_simultaneous(homed_uncoupled_from_estop)
    worst_coupled = _min_clearance_simultaneous(homed_from_estop)

    assert worst_coupled > worst_uncoupled, (
        f"coupling bought nothing: coupled {worst_coupled:.3f} m vs uncoupled "
        f"{worst_uncoupled:.3f} m. Either the swept volume is no longer fed "
        f"forward, or it is sized too small to matter.")


def test_each_return_actually_reaches_the_cells_home_pose(homed_from_working):
    """A path that stops short leaves the next program's assumptions broken."""
    import numpy as np

    from agents.trajectory.backends.ompl.planner import (
        HOME_JOINTS_DEFAULT, _home_config,
    )

    assert homed_from_working, "nothing was planned"
    for ns, traj in homed_from_working.items():
        home = np.asarray(_home_config(CELL, ns, HOME_JOINTS_DEFAULT), float)
        end = np.asarray(traj[-1]["positions"], float)
        n = min(len(home), len(end))
        assert np.allclose(end[:n], home[:n], atol=1e-3), (
            f"{ns} ends at {end[:n]}, not the cell's home {home[:n]}")


def test_the_returns_start_where_the_arms_actually_are(homed_from_working):
    """Dispatching a path that starts elsewhere is a jump, not a move."""
    import numpy as np

    assert homed_from_working, "nothing was planned"
    for ns, traj in homed_from_working.items():
        q0 = np.asarray(traj[0]["positions"], float)
        s = np.asarray(WORKING_Q[ns], float)
        n = min(len(q0), len(s))
        assert np.allclose(q0[:n], s[:n], atol=1e-3)


def test_the_returns_are_time_parameterised(homed_from_working):
    """The executor waits on time_s, so timing must be monotonic and real."""
    assert homed_from_working, "nothing was planned"
    for ns, traj in homed_from_working.items():
        times = [float(p["time_s"]) for p in traj]
        assert len(traj) >= 2, f"{ns} return is not a path"
        assert times[0] == pytest.approx(0.0), f"{ns} does not start at t=0"
        assert all(b > a for a, b in zip(times, times[1:])), \
            f"{ns} timing is not strictly increasing"
        assert times[-1] > 0.5, f"{ns} return duration {times[-1]}s is implausibly fast"


# ── coupling is what buys the safety ─────────────────────────────────────────
#
# These pin the wiring rather than the geometry, so they stub the per-arm planner
# rather than paying for real OMPL runs. A stub still exercises the real swept
# volume accumulation: the spheres are derived from the returned waypoints.

def _stub_path(goal, start=None):
    """A minimal two-point path the swept-volume accumulator can consume."""
    pts = [{"positions": list(start), "time_s": 0.0}] if start is not None else []
    return pts + [{"positions": list(goal), "time_s": 2.0}]


def test_arms_are_planned_against_each_others_swept_volume(ompl_available,
                                                           monkeypatch):
    """Coupling must be observable, not incidental to these particular poses.

    Whichever arm is planned second must receive the first arm's swept spheres.
    Pinned directly, because a coupling regression would otherwise only show up
    as an occasional e-stop on the floor.
    """
    import agents.trajectory.backends.ompl.planner as ompl

    seen: list = []

    def spy(cell_id, robot_ns, start, goal, **kw):
        seen.append((robot_ns, len(kw.get("moving_obstacle_spheres") or [])))
        return _stub_path(goal, start)

    monkeypatch.setattr(ompl, "plan_joint_path", spy)
    ompl.plan_home_trajectories(
        CELL, {ns: [0.5, -1.2, 1.5, 0.0, -0.6, 0.0] for ns in ROBOTS})

    assert len(seen) == 2, f"expected one planning call per arm, got {seen}"
    assert seen[0][1] == 0, "the first arm plans in an otherwise empty cell"
    assert seen[1][1] > 0, (
        f"the second arm ({seen[1][0]}) was planned against 0 moving obstacles — "
        "it cannot know where the first arm swept, so the two returns are "
        "uncoupled")


def test_the_arm_with_the_longest_travel_is_planned_first(ompl_available,
                                                          monkeypatch):
    """It needs the emptiest cell; shorter movers can route around it."""
    import agents.trajectory.backends.ompl.planner as ompl

    order: list = []
    monkeypatch.setattr(ompl, "plan_joint_path",
                        lambda cell_id, robot_ns, *a, **kw: order.append(robot_ns) or [])

    # robot2 is left far from home; robot1 starts nearly there.
    ompl.plan_home_trajectories(CELL, {
        "robot1": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
        "robot2": [1.2, -0.6, 2.4, 0.5, -1.2, 0.8],
    })

    assert order[0] == "robot2", f"planning order {order} ignores travel distance"


def test_an_unplannable_arm_does_not_block_the_others(ompl_available,
                                                      monkeypatch):
    """One impossible return must not cost every other arm its planned path."""
    import agents.trajectory.backends.ompl.planner as ompl

    monkeypatch.setattr(
        ompl, "plan_joint_path",
        lambda cell_id, robot_ns, start, goal, **kw: (
            [] if robot_ns == "robot1" else _stub_path(goal, start)))

    trajs = ompl.plan_home_trajectories(
        CELL, {ns: [0.4, -1.3, 1.6, 0.0, -0.5, 0.0] for ns in ROBOTS})

    assert "robot1" not in trajs, "an unplanned arm must be absent, not empty"
    assert trajs.get("robot2"), "robot2 lost its planned path to robot1's failure"


def test_the_margin_is_sized_from_the_monitors_own_gate(ompl_available,
                                                        monkeypatch):
    """The widest margin tried must be derived from ``warning_distance``.

    The validity checker rejects at ``d < SWEPT_R + 0.12`` while the monitor
    measures ``d - 0.24``, so a swept radius that ignores warning_distance
    produces paths the monitor halts. The relationship is
    ``SWEPT_R = warning_distance + 0.12``.
    """
    import agents.trajectory.backends.ompl.planner as ompl
    import agents.trajectory.tools.robot_kinematics as rk

    radii: list = []
    monkeypatch.setattr(rk, "_load_collision_config",
                        lambda cid: {"warning_distance": 0.30,
                                     "min_clearance": 0.05})

    def spy(cell_id, robot_ns, start, goal, **kw):
        radii.extend(r for _, r in (kw.get("moving_obstacle_spheres") or []))
        return _stub_path(goal, start)

    monkeypatch.setattr(ompl, "plan_joint_path", spy)
    ompl.plan_home_trajectories(CELL, {ns: [0.4, -1.3, 1.6, 0.0, -0.5, 0.0]
                                       for ns in ROBOTS})

    assert radii, "the second arm saw no swept obstacles at all"
    assert max(radii) == pytest.approx(0.42), (
        f"widest swept radius {max(radii)} is not warning_distance (0.30) + "
        f"sphere radius (0.12) — the planning margin has drifted from the gate "
        f"the monitor enforces")


def test_an_impossible_margin_relaxes_rather_than_giving_up(ompl_available,
                                                           monkeypatch):
    """A margin no path can hold must not cost the arm its planned path.

    Fail-closed on SAFETY, not on planning: refusing to plan sends the arm home
    on the executor's blind straight line, which is strictly more dangerous than
    a planned path at a smaller margin. So the margin shrinks and the arm still
    gets a real, checked trajectory.
    """
    import agents.trajectory.backends.ompl.planner as ompl
    import agents.trajectory.tools.robot_kinematics as rk

    # A margin far beyond the cell's geometry: nothing can satisfy the widest.
    monkeypatch.setattr(rk, "_load_collision_config",
                        lambda cid: {"warning_distance": 0.20,
                                     "min_clearance": 0.05})

    attempts: list = []

    def only_small_margins_work(cell_id, robot_ns, start, goal, **kw):
        rs = [r for _, r in (kw.get("moving_obstacle_spheres") or [])]
        attempts.append((robot_ns, max(rs) if rs else 0.0))
        if rs and max(rs) > 0.20:
            return None      # stand in for "no path at this margin"
        return _stub_path(goal, start)

    monkeypatch.setattr(ompl, "plan_joint_path", only_small_margins_work)
    trajs = ompl.plan_home_trajectories(CELL, E_STOP_Q)

    second = [a for a in attempts if a[1] > 0.0]
    assert len(second) > 1, (
        f"the second arm gave up after one margin: {attempts} — it must retry "
        f"at a smaller margin before falling back to a blind home")
    assert sorted(trajs) == ROBOTS, (
        f"only {sorted(trajs)} planned; relaxation must recover the rest")


def test_no_ompl_yields_no_paths_rather_than_unchecked_ones(monkeypatch):
    """Fail closed: without a planner, say so — never emit an unplanned path."""
    import agents.trajectory.backends.ompl.planner as ompl

    monkeypatch.setattr(ompl, "_import_ompl", lambda: None)

    assert ompl.plan_home_trajectories(
        CELL, {ns: [0.1] * 6 for ns in ROBOTS}) == {}


# ── HOME is the cell's declared pose, everywhere ─────────────────────────────

def test_planned_home_matches_the_blind_fallbacks_intent():
    """The cell declares home_pose_deg; planning must target THAT pose.

    Four modules hardcode [0,-1.57,2.2,0,-0.8,0] while welding_cell_2_small declares
    [0,-90,90,0,0,0]. Planning to the hardcoded default would park each arm
    somewhere the cell does not call home, so the declaration wins.
    """
    import math

    from agents.trajectory.backends.ompl.planner import (
        HOME_JOINTS_DEFAULT, _home_config,
    )

    for ns in ROBOTS:
        q = _home_config(CELL, ns, HOME_JOINTS_DEFAULT)
        assert q == pytest.approx(
            [0.0, math.radians(-90), math.radians(90), 0.0, 0.0, 0.0]), ns
        assert q != pytest.approx(list(HOME_JOINTS_DEFAULT)), (
            "the test would pass vacuously if the declared pose equalled the "
            "hardcoded default")
