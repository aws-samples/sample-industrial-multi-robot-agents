"""The arm is an envelope of its links, and a moving sphere needs room to stop.

The arm-vs-arm gate used one sphere per frame plus fill spheres on the skeleton:
1-29 % of the KR240's hull vertices, so it passed two arms whose castings
interpenetrated by 300 mm (umarell issue 04). And a sphere moving at v needs
``v * reaction_time + v^2 / (2 * decel)`` more than a parked one, which a fixed
radius cannot express.

* The converter fits each link's ``<collision>`` hull (``link_collision``) and
  writes the robot's ``stopping`` data (``null`` + REVIEW until someone gives it).
* The gate, the timed static backstop and the runtime monitor check skeleton,
  tool AND envelope, each sphere widened by its own stopping distance.
* ``validate_trajectory_set`` also checks every leg against the static obstacles
  (umarell issue 06: it was blind to the fixture).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

np = pytest.importorskip("numpy")


def _rk():
    import importlib
    return importlib.import_module("agents.trajectory.tools.robot_kinematics")


# ── the converter fits and writes it ─────────────────────────────────────────

def _box_obj(path: Path, lo, hi) -> None:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)]
    path.write_text("".join(f"v {a} {b} {c}\n" for a, b, c in v)
                    + "".join("f " + " ".join(map(str, q)) + "\n" for q in f))


def _arm(tmp_path: Path) -> Path:
    """A 1.0 m upper arm on link 'upper', and a flange (the tool's link)."""
    _box_obj(tmp_path / "upper.obj", (0.0, -0.1, -0.1), (1.0, 0.1, 0.1))
    _box_obj(tmp_path / "flange.obj", (0.0, -0.05, -0.05), (0.05, 0.05, 0.05))
    col = lambda m: f'<collision><geometry><mesh filename="{m}"/></geometry></collision>'  # noqa: E731
    u = tmp_path / "arm.urdf"
    u.write_text(f"""<robot name="arm">
  <link name="base"/>
  <link name="upper">{col("upper.obj")}</link>
  <link name="flange">{col("flange.obj")}</link>
  <joint name="a1" type="revolute"><parent link="base"/><child link="upper"/>
    <limit lower="-1" upper="1" velocity="1"/></joint>
  <joint name="a2" type="revolute"><parent link="upper"/><child link="flange"/>
    <origin xyz="1 0 0"/><limit lower="-1" upper="1" velocity="1"/></joint>
</robot>""")
    return u


def test_every_link_hull_is_covered_and_the_tool_link_is_left_to_the_tool(tmp_path):
    import convert_isaac_scene as cv
    env = cv.fit_link_envelope(_arm(tmp_path), "flange", 0.12)
    assert set(env["links"]) == {"upper"}
    assert env["vertex_coverage"] == 1.0 and env["worst_gap_m"] <= 0.12
    c = np.asarray(env["links"]["upper"])
    corners = np.array([[x, y, z] for x in (0, 1) for y in (-.1, .1) for z in (-.1, .1)])
    assert np.min(np.linalg.norm(corners[:, None] - c[None], axis=2), axis=1).max() <= 0.12


def test_a_urdf_without_collision_meshes_has_no_envelope(tmp_path):
    import convert_isaac_scene as cv
    u = tmp_path / "bare.urdf"
    u.write_text('<robot name="b"><link name="base"/><link name="flange"/></robot>')
    assert cv.fit_link_envelope(u, "flange", 0.12) is None


def _entry(env):
    return {"namespace": "arms", "robot_id": "robot1", "model_name": "m", "robot_model": "m",
            "isaac_prim": "r", "base_position": (0, 0, 0), "yaw": 0.0, "pose_source": "s",
            "lowers": [-1], "uppers": [1], "vels": [1], "tip_link": "flange",
            "tool_fit": None, "envelope": env}


_DEFAULTS = {"gripper": "g", "workspace_radius": 2.9, "planning_group": "p",
             "end_effector_link": "tcp", "tcp_offset": (-0.9, 0, 0), "accel": [1],
             "rrt_j2": [0, 1], "rrt_j3": [0, 1], "mount_link": "link_6",
             "tool_axis_dir": [-1, 0, 0], "tool_axis_start": 0.15, "tool_axis_end": 0.9,
             "tool_axis_step": 0.15, "thermal_spread": False}


def test_robots_yaml_carries_the_envelope_and_unset_stopping_data(tmp_path):
    import yaml
    import convert_isaac_scene as cv
    env = cv.fit_link_envelope(_arm(tmp_path), "flange", 0.12)
    text = cv.render_robots_yaml("c", "s", [_entry(env)], _DEFAULTS)
    r = yaml.safe_load(text)["robots"]["arms"]["robot1"]
    assert r["link_collision"]["radius_m"] == 0.12
    assert np.allclose(r["link_collision"]["links"]["upper"], env["links"]["upper"], atol=1e-4)
    assert r["stopping"] == {"reaction_time_s": None, "brake_decel_m_s2": None}
    assert "REVIEW" in text


def test_the_stopping_data_comes_from_the_command_line(tmp_path):
    import yaml
    import convert_isaac_scene as cv
    text = cv.render_robots_yaml("c", "s", [_entry(None)], {
        **_DEFAULTS, "stopping": {"reaction_time_s": 0.1, "brake_decel_m_s2": 4.0}})
    r = yaml.safe_load(text)["robots"]["arms"]["robot1"]
    assert r["stopping"] == {"reaction_time_s": 0.1, "brake_decel_m_s2": 4.0}
    assert "link_collision" not in r


# ── stopping distance ────────────────────────────────────────────────────────

class _M:
    njoints = 3


@pytest.fixture
def spec(monkeypatch):
    rk = _rk()
    s = {"env": None, "stop": (0.1, 4.0), "tool": None}
    monkeypatch.setattr(rk, "_robot_spec", lambda model: s)
    return s


def test_the_stopping_distance_is_reaction_plus_braking(spec):
    rk = _rk()
    d = rk.stopping_distance(_M(), [0.0, 1.0, 2.0])
    assert d == pytest.approx([0.0, 0.1 + 1 / 8, 0.2 + 4 / 8])
    assert rk.accounts_for_speed(_M())


def test_without_stopping_data_speed_is_not_accounted_for(spec):
    rk = _rk()
    spec["stop"] = None
    assert not rk.accounts_for_speed(_M())
    assert rk.stopping_distance(_M(), [3.0]) == pytest.approx([0.0])


def test_each_spheres_stop_narrows_the_pair_clearance():
    rk = _rk()
    A, B = np.array([[0.0, 0, 0], [5, 5, 5]]), np.array([[1.0, 0, 0]])
    assert rk._check_robot_pair_collision(A, B, 0.12)[1] == pytest.approx(0.76)
    col, d = rk._check_robot_pair_collision(A, B, 0.12, [0.5, 0.0], [0.3])
    assert d == pytest.approx(-0.04) and col


def test_the_speed_bound_is_never_below_the_real_speed():
    """|v| <= sum_j qdot_max_j * |p - o_j| for any point driven by revolute joints."""
    rk = _rk()
    o = [np.zeros(3), np.array([1.0, 0, 0])]
    data = SimpleNamespace(oMi=[None] + [SimpleNamespace(translation=t) for t in o])
    p = np.array([[2.0, 0, 0]])
    # Both joints about z at their limits: v = 1*2 + 2*1 = 4 m/s at most.
    assert rk.sphere_speed_bound(_M(), data, np.zeros(3), 0.0, p, [1.0, 2.0]) == \
        pytest.approx([4.0])
    assert np.isinf(rk.sphere_speed_bound(_M(), data, np.zeros(3), 0.0, p, []))[0]


# ── the gate is exact and fast ───────────────────────────────────────────────

def test_the_grouped_pair_check_finds_the_same_minimum():
    rk = _rk()
    rng = np.random.default_rng(7)
    for _ in range(40):
        A = rng.uniform(-1, 1, (60, 3))
        B = rng.uniform(-1, 1, (50, 3)) + rng.uniform(-3, 3, 3)
        sa, sb = rng.uniform(0, 0.3, 60), rng.uniform(0, 0.3, 50)
        ga = [slice(0, 10), slice(10, 35), slice(35, 60)]
        gb = [slice(0, 25), slice(25, 50)]
        full = rk._check_robot_pair_collision(A, B, 0.12, sa, sb)
        grouped = rk._check_robot_pair_collision(A, B, 0.12, sa, sb, ga, gb)
        assert grouped[0] == full[0] and grouped[1] == pytest.approx(full[1])


def test_envelope_groups_follow_the_links(monkeypatch):
    rk = _rk()
    monkeypatch.setattr(rk, "_robot_spec", lambda m: {
        "env": {"fids": np.array([4, 4, 7, 7, 7, 9])}, "stop": None, "tool": None})
    assert rk.gate_groups(_M(), 10) == [slice(0, 4), slice(4, 6), slice(6, 9), slice(9, 10)]
    monkeypatch.setattr(rk, "_robot_spec", lambda m: {"env": None, "stop": None, "tool": None})
    assert rk.gate_groups(_M(), 10) == [slice(0, 10)]


def test_the_box_broad_phase_finds_the_same_first_breach():
    rk = _rk()
    rng = np.random.default_rng(3)
    for trial in range(30):
        obs = [{"id": f"b{k}", "type": "box", "center": rng.uniform(-2, 2, 3).tolist(),
                "half_extents": rng.uniform(0.02, 0.2, 3).tolist(),
                "collision_margin": float(rng.uniform(0, 0.05))} for k in range(80)]
        comp = rk._compile_static(obs)
        P = rng.uniform(-2, 2, (100, 3))
        R = rng.uniform(0.03, 0.12, 100)
        own = np.where(rng.uniform(size=100) < 0.2, 0.02, np.nan)
        _m = lambda om: np.where(np.isnan(own)[:, None], om[None, :], own[:, None])  # noqa: E731
        diff = P[:, None] - comp["box_c"][None]
        d = np.linalg.norm(diff - np.clip(diff, -comp["box_h"][None], comp["box_h"][None]), axis=2)
        hit = d < R[:, None] + _m(comp["box_m"]) + 0.01
        got = rk._first_box_hit(P, R, own, comp, 0.01, _m)
        if not hit.any():
            assert got is None
            continue
        m = int(np.nonzero(hit.any(axis=0))[0][0])
        assert got[:2] == (m, int(np.nonzero(hit[:, m])[0][0])), trial


def test_an_envelope_sphere_is_labelled_as_neither_arm_frame_nor_tool():
    rk = _rk()
    box = rk._compile_static([{"id": "b", "type": "box", "center": [0, 0, 1],
                               "half_extents": [.1, .1, .1], "collision_margin": 0.0}])
    lab = rk._Labels([rk.EnvelopeSphere(5)], [np.nan])
    hit = rk._static_first_breach(np.array([[0, 0, 1.0]]), np.array([0.12]),
                                  np.zeros(1, bool), lab, box, 0.0)
    assert hit["link_index"] is None and hit["envelope_sphere"] == 5
    assert not isinstance(rk.EnvelopeSphere(5), tuple)


def test_a_sphere_stop_reaches_the_static_check():
    rk = _rk()
    box = rk._compile_static([{"id": "b", "type": "box", "center": [0, 0, 1],
                               "half_extents": [.1, .1, .1], "collision_margin": 0.0}])
    P, R = np.array([[0, 0, 1.3]]), np.array([0.12])
    args = (P, R, np.zeros(1, bool), [2], box, 0.0)
    assert rk._static_first_breach(*args) is None                    # 0.2 - 0.12 clear
    assert rk._static_first_breach(*args, np.array([0.1]))["clearance_m"] == pytest.approx(-0.02)


# ── the per-step gate sees the fixture (issue 06) ────────────────────────────

@pytest.fixture
def gate(monkeypatch):
    pytest.importorskip("langchain_core")
    import importlib
    pt = importlib.import_module("agents.trajectory.tools.planning_tools")
    monkeypatch.setattr(pt, "_resolve_trajectories", lambda c, t: t)
    monkeypatch.setattr(pt, "_load_collision_config", lambda c: {"warning_distance": 0.2})
    monkeypatch.setattr(pt, "inter_robot_clearance", lambda c, t, **kw: {
        "collision_free": True, "min_clearance_m": 1.0, "warning_distance_m": 0.2})
    return pt, getattr(pt.validate_trajectory_set, "func", pt.validate_trajectory_set)


_LEGS = {"robot1": [{"positions": [0] * 6, "time_s": 0.0}, {"positions": [0] * 6, "time_s": 1.0}],
         "robot2": [{"positions": [0] * 6, "time_s": 0.0}]}


def test_a_leg_in_a_fixture_is_not_collision_free(gate, monkeypatch):
    pt, validate = gate
    monkeypatch.setattr(pt, "leg_static_breach", lambda c, ns, t: None if ns == "robot2" else {
        "type": "box", "object_id": "fixture_v1", "link_index": 4, "clearance_m": -0.02,
        "sample_index": 1, "phase": "transit"})
    r = validate("c", _LEGS)
    assert r["collision_free"] is False and r["static_obstacles_checked"] is True
    assert r["static_collisions"] == [{
        "robot_ns": "robot1", "type": "box", "object_id": "fixture_v1", "link_index": 4,
        "clearance_m": -0.02, "sample_index": 1, "phase": "transit", "time_s": 1.0}]
    assert "No start delay" in r["suggestion"] and "delay_fix" not in r


def test_clear_legs_keep_the_arm_verdict(gate, monkeypatch):
    pt, validate = gate
    monkeypatch.setattr(pt, "leg_static_breach", lambda c, ns, t: None)
    r = validate("c", _LEGS)
    assert r["collision_free"] is True and "static_collisions" not in r


def test_a_static_check_that_cannot_run_says_so(gate, monkeypatch):
    pt, validate = gate

    def boom(c, ns, t):
        raise FileNotFoundError("no URDF")
    monkeypatch.setattr(pt, "leg_static_breach", boom)
    r = validate("c", _LEGS)
    assert r["static_obstacles_checked"] is False and "no URDF" in r["static_check_error"]
    assert r["collision_free"] is True


def test_the_static_check_can_be_turned_off(gate, monkeypatch):
    pt, validate = gate
    monkeypatch.setattr(pt, "leg_static_breach", lambda *a: pytest.fail("called"))
    assert "static_obstacles_checked" not in validate("c", _LEGS, check_static_obstacles=False)


def test_the_measured_scale_is_the_fastest_joint_against_its_limit():
    rk = _rk()
    s = [{"positions": [0, 0], "time_s": 0.0}, {"positions": [0.5, 0.1], "time_s": 1.0},
         {"positions": [0.5, 0.1], "time_s": 2.0, "phase": "approach"}]
    assert rk._measured_scale(s, [1.0, 1.0], "transit") == pytest.approx(0.5)
    assert rk._measured_scale(s, [1.0, 1.0], "approach") == pytest.approx(0.0)
    assert rk._measured_scale(s, [], "transit") == 1.0                # fail-closed


# ── the runtime monitor measures each sphere's speed ─────────────────────────

def test_the_monitor_uses_the_bound_until_it_can_measure(spec, monkeypatch):
    from agents.cell.collision_monitor import CollisionMonitor
    rk = _rk()
    monkeypatch.setattr(rk, "sphere_speed_bound", lambda *a: np.array([3.0]))
    mon = CollisionMonitor("c", runtime=SimpleNamespace())
    mon._models["r"] = (_M(), None)
    mon._bases["r"], mon._yaws["r"] = np.zeros(3), 0.0
    P0 = np.array([[0.0, 0, 0]])
    first = mon._sphere_stops("r", "gate", P0, {"timestamp": 10.0})
    assert first == pytest.approx(rk.stopping_distance(_M(), [3.0]))     # unmeasured: bound
    moved = mon._sphere_stops("r", "gate", P0 + [0.1, 0, 0], {"timestamp": 10.1})
    assert moved == pytest.approx(rk.stopping_distance(_M(), [1.0]))     # 0.1 m in 0.1 s
    again = mon._sphere_stops("r", "gate", P0 + [0.1, 0, 0], {"timestamp": 10.1})
    assert again == pytest.approx(moved)                                 # same sample polled
    stale = mon._sphere_stops("r", "gate", P0, {"timestamp": 20.0})
    assert stale == pytest.approx(first)                                 # too old: bound
    spec["stop"] = None
    assert mon._sphere_stops("r", "gate", P0, {"timestamp": 21.0}) is None


# ── the real cell ────────────────────────────────────────────────────────────

@pytest.fixture
def cell2s(monkeypatch):
    pytest.importorskip("pinocchio")
    cells = REPO / "cells"
    urdf = cells / "welding_cell_2_small/robots/kr240r2900__sticktool/kr240r2900__sticktool.urdf"
    if not urdf.is_file():
        pytest.skip("the cell's robot assets are not in this checkout")
    monkeypatch.setenv("CELL_BASE_PATH", str(cells))
    rk = _rk()
    m, _ = rk._load_pinocchio_model("welding_cell_2_small", "robot1")
    return rk, m, rk._load_robot_config("welding_cell_2_small", "robot1")


def test_the_converted_cell_declares_its_envelope(cell2s):
    rk, m, rc = cell2s
    env = rk._robot_spec(m)["env"]
    assert env is not None and len(env["fids"]) == sum(
        len(v) for v in rc["link_collision"]["links"].values())
    assert env["radius"] == pytest.approx(0.12)
    assert set(rc["stopping"]) == {"reaction_time_s", "brake_decel_m_s2"}


def test_the_gate_carries_skeleton_tool_and_envelope(cell2s):
    import pinocchio as pin
    rk, m, rc = cell2s
    from agents.home_pose import home_pose
    q = np.asarray(home_pose("welding_cell_2_small", "robot1"), float)
    base, yaw = np.array(rc["base_position"]), float(rc.get("base_orientation_yaw", 0))
    G = rk._gate_points(m, m.createData(), q, base, yaw, rc.get("tool_collision"))
    d = m.createData()
    skel = rk._config_spheres(m, d, q, base, yaw, rc.get("tool_collision"))
    assert len(G) == len(skel) + len(rk._robot_spec(m)["env"]["fids"])
    assert np.allclose(G[: len(skel)], np.asarray(skel))
    groups = rk.gate_groups(m, len(G))
    assert groups[0] == slice(0, len(skel)) and groups[-1].stop == len(G)
    pin.forwardKinematics(m, d, q)
    pin.updateFramePlacements(m, d)
    # The bound is an upper bound on the measured motion of a small step.
    dq = np.full(m.nq, 1e-3)
    G2 = rk._gate_points(m, m.createData(), q + dq, base, yaw, rc.get("tool_collision"))
    v = np.linalg.norm(G2 - G, axis=1) / 1e-3                          # at |qdot| = 1 rad/s
    pin.forwardKinematics(m, d, q)
    bound = rk.sphere_speed_bound(m, d, base, yaw, G, np.ones(m.nq))
    assert np.all(v <= bound * 1.01 + 1e-6)
