"""The timed path follows the path the planner validated (umarell issue 18).

TOPP-RA was fed a cubic spline through the waypoints at a UNIFORM parameter, so
between unevenly spaced knots it overshot: up to 0.7 rad off the polyline on
corner-heavy transits. The arm executed a path nobody checked.

* With no bound declared, every straight segment is timed rest-to-rest: exact.
* With robots.yaml ``path_following.max_deviation_rad`` (the converter derives it
  from a Cartesian tolerance and the arm's lever arms), the spline through the
  waypoints and a corner-blended path are candidates too. Each is MEASURED against
  the polyline, and the fastest within the bound ships.
* The joint limits are checked on the timed path itself, not only at TOPP-RA's
  gridpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

np = pytest.importorskip("numpy")


def _tp():
    import importlib
    return importlib.import_module("agents.trajectory.tools.time_parameterization")


# HOME, two close pairs of knots (unequal spacing, as PathSimplifier leaves them),
# a weld pose: the shape that overshot by 0.17 rad.
SPARSE = np.array([[0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0],
                   [0.6, -1.4, 1.5, 0.4, 0.3, 0.2], [0.65, -1.38, 1.48, 0.45, 0.32, 0.25],
                   [1.5, -1.2, 1.3, 1.6, -0.6, 1.2], [1.55, -1.21, 1.35, 1.7, -0.65, 1.25],
                   [1.98, -1.46, 1.68, 2.43, -1.69, 2.02]])
VEL = np.array([1.6, 1.3, 1.5, 2.3, 2.3, 3.4])
ACC = 2.5 * VEL
SCALE = 0.35

_t = np.linspace(0, 1, 7)
DESCENT = np.array([[0.5 + 0.05 * t, -1.0 + 0.04 * t + 0.004 * t * t, 1.2 - 0.06 * t,
                     0.1, 0.8 + 0.03 * t * t, 0.2] for t in _t])


@pytest.fixture
def toppra():
    return pytest.importorskip("toppra")


def _q(out):
    return np.array([p["positions"] for p in out], float)


def _limit_ratios(out, vel, acc):
    t = np.array([p["time_s"] for p in out], float)
    q = _q(out)
    keep = np.concatenate([[True], np.diff(t) > 1e-4])
    t, q = t[keep], q[keep]
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    a = np.diff(v, axis=0) / ((t[2:] - t[:-2]) / 2)[:, None]
    return float(np.max(np.abs(v) / vel)), float(np.max(np.abs(a[:-1]) / acc))


def test_without_a_bound_the_path_is_followed_exactly(toppra):
    tp = _tp()
    out = tp.parameterize(SPARSE, VEL, ACC, SCALE)
    assert float(tp._off_polyline(SPARSE, _q(out)).max()) < 1e-6
    assert _q(out)[0] == pytest.approx(SPARSE[0]) and _q(out)[-1] == pytest.approx(SPARSE[-1])
    rv, ra = _limit_ratios(out, VEL * SCALE, ACC * SCALE ** 2)
    assert rv <= 1.02 and ra <= 1.05


def test_with_a_bound_the_path_stays_within_it_and_within_the_limits(toppra):
    tp = _tp()
    delta = 0.0016
    out = tp.parameterize(SPARSE, VEL, ACC, SCALE, max_deviation_rad=delta)
    assert float(tp._off_polyline(SPARSE, _q(out)).max()) <= delta
    rv, ra = _limit_ratios(out, VEL * SCALE, ACC * SCALE ** 2)
    assert rv <= 1.02 and ra <= 1.05


def test_the_fastest_verified_candidate_ships(toppra):
    tp = _tp()
    exact = tp.parameterize(SPARSE, VEL, ACC, SCALE)
    bounded = tp.parameterize(SPARSE, VEL, ACC, SCALE, max_deviation_rad=0.0016)
    assert bounded[-1]["time_s"] <= exact[-1]["time_s"] + 1e-6


def test_a_descent_does_not_stop_at_every_ik_step_when_a_bound_is_declared(toppra):
    tp = _tp()
    vel, acc = [2.7, 2.7, 2.7, 5.7, 5.7, 10.7], [3, 3, 3, 6, 6, 8]
    exact = tp.parameterize(DESCENT, vel, acc, 0.2)
    smooth = tp.parameterize(DESCENT, vel, acc, 0.2, max_deviation_rad=0.0016)
    assert smooth[-1]["time_s"] < 0.6 * exact[-1]["time_s"]
    assert float(tp._off_polyline(DESCENT, _q(smooth)).max()) <= 0.0016


def test_the_timed_path_meets_the_limits_between_gridpoints(toppra):
    import toppra.algorithm as algo
    import toppra.constraint as constraint
    tp = _tp()
    vl, al = VEL * SCALE, ACC * SCALE ** 2
    knots = tp._blended_knots(SPARSE, 0.0016, 0.02)
    jt = tp._timed_path(toppra, constraint, algo, knots, vl, al)
    ts = np.linspace(0, jt.duration, 4000)
    assert np.max(np.abs(jt(ts, 1)) / vl) <= 1 + 1e-3        # numerical, not 8-23 % over
    assert np.max(np.abs(jt(ts, 2)) / al) <= 1 + 1e-3


def test_a_corner_blend_stays_within_the_bound_of_the_corner():
    tp = _tp()
    P = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    knots = tp._blended_knots(P, 0.01, 0.01)
    assert float(tp._off_polyline(P, knots).max()) <= 0.01 + 1e-9
    assert np.allclose(knots[0], P[0]) and np.allclose(knots[-1], P[-1])
    assert np.min(np.linalg.norm(knots - P[1], axis=1)) > 0.0       # the corner is cut


# ── the bound comes from the converter ───────────────────────────────────────

def test_the_bound_is_read_from_robots_yaml():
    import importlib
    rk = importlib.import_module("agents.trajectory.tools.robot_kinematics")
    assert rk.path_deviation_rad({"path_following": {"max_deviation_rad": 0.0015}}) == 0.0015
    for cfg in ({}, {"path_following": {}}, {"path_following": {"max_deviation_rad": 0}},
                {"path_following": {"max_deviation_rad": "x"}}, None):
        assert rk.path_deviation_rad(cfg) is None


def _box_obj(path: Path, lo, hi) -> None:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)]
    path.write_text("".join(f"v {a} {b} {c}\n" for a, b, c in v)
                    + "".join("f " + " ".join(map(str, q)) + "\n" for q in f))


def _arm(tmp_path: Path) -> Path:
    """Joint a1 at the base, a 1 m link, joint a2, a 0.5 m link (the tip)."""
    _box_obj(tmp_path / "l1.obj", (0.0, -0.1, -0.1), (1.0, 0.1, 0.1))
    _box_obj(tmp_path / "l2.obj", (0.0, -0.05, -0.05), (0.5, 0.05, 0.05))
    col = lambda m: f'<collision><geometry><mesh filename="{m}"/></geometry></collision>'  # noqa: E731
    u = tmp_path / "arm.urdf"
    u.write_text(f"""<robot name="arm">
  <link name="base"/>
  <link name="l1">{col("l1.obj")}</link>
  <link name="l2">{col("l2.obj")}</link>
  <joint name="a1" type="revolute"><parent link="base"/><child link="l1"/>
    <limit lower="-1" upper="1" velocity="1"/></joint>
  <joint name="a2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0"/><limit lower="-1" upper="1" velocity="1"/></joint>
</robot>""")
    return u


def test_the_converter_bounds_every_point_by_its_lever_arms(tmp_path):
    import convert_isaac_scene as cv
    lev = cv.path_lever_arms(_arm(tmp_path), "l2", (0.8, 0.0, 0.0))
    # a2: the TCP 0.8 m out on l2. a1: 1 m to a2, then 0.8 m.
    corner_l1 = float(np.linalg.norm([1.0, 0.1, 0.1]))
    assert lev["levers"]["l1->l2"] == pytest.approx(0.8)
    assert lev["levers"]["base->l1"] == pytest.approx(max(corner_l1, 1.8))
    assert lev["lever_sum_m"] == pytest.approx(0.8 + 1.8)


def test_without_measurable_geometry_no_bound_is_derived(tmp_path):
    import convert_isaac_scene as cv
    u = tmp_path / "bare.urdf"
    u.write_text('<robot name="b"><link name="base"/><link name="l1"/>'
                 '<joint name="a1" type="revolute"><parent link="base"/><child link="l1"/>'
                 '<limit lower="-1" upper="1" velocity="1"/></joint></robot>')
    assert cv.path_lever_arms(u, "l1", (0.0, 0.0, 0.0)) is None


def test_robots_yaml_carries_the_derived_bound(tmp_path):
    import yaml
    import convert_isaac_scene as cv
    entry = {"namespace": "arms", "robot_id": "robot1", "model_name": "m", "robot_model": "m",
             "isaac_prim": "r", "base_position": (0, 0, 0), "yaw": 0.0, "pose_source": "s",
             "lowers": [-1], "uppers": [1], "vels": [1], "tip_link": "l2", "tool_fit": None,
             "path_levers": cv.path_lever_arms(_arm(tmp_path), "l2", (0.8, 0.0, 0.0))}
    defaults = {"gripper": "g", "workspace_radius": 2.9, "planning_group": "p",
                "end_effector_link": "tcp", "tcp_offset": (0.8, 0, 0), "accel": [1],
                "rrt_j2": [0, 1], "rrt_j3": [0, 1], "mount_link": "l2",
                "tool_axis_dir": [-1, 0, 0], "tool_axis_start": 0.15, "tool_axis_end": 0.9,
                "tool_axis_step": 0.15, "thermal_spread": False, "path_tolerance_m": 0.02}
    pf = yaml.safe_load(cv.render_robots_yaml("c", "s", [entry], defaults))[
        "robots"]["arms"]["robot1"]["path_following"]
    assert pf["tolerance_m"] == 0.02 and pf["lever_sum_m"] == pytest.approx(2.6)
    assert pf["max_deviation_rad"] == pytest.approx(0.02 / 2.6, rel=1e-4)
    no_tol = yaml.safe_load(cv.render_robots_yaml("c", "s", [entry], {
        **defaults, "path_tolerance_m": 0.0}))["robots"]["arms"]["robot1"]
    assert "path_following" not in no_tol


def test_the_reference_cell_declares_its_bound():
    import yaml
    cfg = yaml.safe_load((REPO / "cells/welding_cell_2_small/agentic_planner/config/"
                          "robots.yaml").read_text())
    arms = [r for ns in cfg["robots"].values() for r in ns.values() if isinstance(r, dict)]
    assert arms and all(0 < r["path_following"]["max_deviation_rad"] < 0.01 for r in arms)


def test_every_planner_timing_call_passes_the_bound():
    for f in ("agents/trajectory/backends/ompl/planner.py",
              "agents/trajectory/backends/ompl_compound/planner.py"):
        src = (REPO / f).read_text()
        calls = src.count("parameterize(") - src.count("def _parameterize_two_phase(")
        passed = src.count("max_deviation_rad=") + src.count("**lim")
        assert calls and passed >= calls, f
