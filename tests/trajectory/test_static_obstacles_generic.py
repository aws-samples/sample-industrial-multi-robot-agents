"""Static obstacles are checked generically — floor included, tool included.

The floor was declared as a plane in obstacles.yaml and read by nothing: five
places hard-coded "sphere CENTRE z >= -0.05" with no radius, and the tool's
collision spheres were exempt from the floor and the fixtures (umarell issue 03).

* Every ``plane`` entry is an obstacle: a sphere is clear when
  ``n.c + offset >= radius + collision_margin``. A cell with no plane gets the
  physical floor (z = 0). Frames rigidly fixed to the world are not checked
  against planes, by structure (parentJoint 0), not by index.
* The tool's ``static_spheres`` (fitted by the converter at the tool's own
  thickness) are checked against floor, boxes and spheres.
* One vectorised check serves the validity predicate, the diagnosis, the IK
  branch filter and ``check_self_collision`` — and for the arm it returns what the
  old loops returned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

np = pytest.importorskip("numpy")


def _rk():
    import importlib
    return importlib.import_module("agents.trajectory.tools.robot_kinematics")


def _breach(P, R, obstacles, floor_mask=None, extra=0.0):
    rk = _rk()
    P = np.asarray(P, float).reshape(-1, 3)
    R = np.asarray(R, float)
    mask = np.ones(len(P), bool) if floor_mask is None else np.asarray(floor_mask, bool)
    return rk._static_first_breach(P, R, mask, list(range(len(P))),
                                   rk._compile_static(obstacles), extra)


# ── planes ───────────────────────────────────────────────────────────────────

def test_a_declared_plane_is_an_obstacle_with_radius_and_margin():
    floor = [{"id": "floor", "type": "plane", "normal": [0, 0, 1], "offset": 0.0,
              "collision_margin": 0.05}]
    assert _breach([[0, 0, 0.18]], [0.12], floor) is None          # 0.18 >= 0.12 + 0.05
    hit = _breach([[0, 0, 0.16]], [0.12], floor)
    assert hit["type"] == "floor" and hit["clearance_m"] == pytest.approx(-0.01)
    assert hit["point"] == [0.0, 0.0, 0.0]


def test_the_plane_sign_follows_the_schema():
    """n.x + d >= 0 is free: offset -0.05 puts the plane at z = +0.05."""
    raised = [{"id": "floor", "type": "plane", "normal": [0, 0, 2], "offset": -0.1,
               "collision_margin": 0.0}]                            # normalised: z >= 0.05
    assert _breach([[0, 0, 0.06]], [0.0], raised) is None
    assert _breach([[0, 0, 0.04]], [0.0], raised) is not None


def test_a_cell_without_a_plane_still_has_the_floor():
    hit = _breach([[0, 0, 0.05]], [0.12], [])
    assert hit["type"] == "floor" and hit["object_id"] == "floor"
    assert _breach([[0, 0, 0.13]], [0.12], []) is None


def test_bodies_excluded_from_the_floor_are_not_tested_against_it():
    assert _breach([[0, 0, -1.0]], [0.12], [], floor_mask=[False]) is None


# ── boxes / spheres keep their order ─────────────────────────────────────────

def test_the_first_breach_is_in_declaration_order_then_sphere_order():
    obs = [{"id": "far", "type": "box", "center": [5, 0, 1], "half_extents": [.1, .1, .1]},
           {"id": "b1", "type": "box", "center": [0, 0, 1], "half_extents": [.1, .1, .1]},
           {"id": "s1", "type": "sphere", "center": [0, 0, 1], "radius": 0.1},
           {"id": "floor", "type": "plane", "normal": [0, 0, 1], "offset": 0.0,
            "collision_margin": 0.0}]
    hit = _breach([[3, 3, 3], [0, 0, 1.25], [0, 0, 1.0]], [0.1, 0.1, 0.1], obs)
    assert hit["object_id"] == "b1" and hit["link_index"] == 1
    assert _breach([[0, 0, 1.0]], [0.1], [{**obs[1], "is_workpiece": True}, obs[3]]) is None


# ── the real cells ───────────────────────────────────────────────────────────

@pytest.fixture
def cell2t(monkeypatch):
    pytest.importorskip("pinocchio")
    urdf = REPO / "cells/welding_cell_2_small/robots/kr240r2900__sticktool/kr240r2900__sticktool.urdf"
    if not urdf.is_file():
        pytest.skip("the cell's robot assets are not in this checkout")
    monkeypatch.setenv("CELL_BASE_PATH", str(REPO / "cells"))
    rk = _rk()
    m, _ = rk._load_pinocchio_model("welding_cell_2_small", "robot1")
    rc = rk._load_robot_config("welding_cell_2_small", "robot1")
    return rk, m, rc


def _old_arm_breach(rk, model, data, q, base, yaw, obstacles, r, radii):
    """The pre-change loops, arm spheres only (boxes and spheres)."""
    import pinocchio as pin
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    spheres = rk._sphere_positions(model, data, base, yaw)
    n_r = len(radii) if radii is not None else 0
    for obs in obstacles:
        if obs.get("is_workpiece"):
            continue
        m = float(obs.get("collision_margin", 0.05))
        if obs.get("type") == "box":
            c, h = np.array(obs["center"]), np.array(obs["half_extents"])
            for idx in range(2, len(spheres)):
                cl = (radii[idx] if idx < n_r else r) + m
                if float(np.linalg.norm(spheres[idx] - (c + np.clip(spheres[idx] - c, -h, h)))) < cl:
                    return obs["id"], idx
    return None


def test_for_the_arm_the_vectorised_check_returns_what_the_loops_returned(cell2t,
                                                                         monkeypatch):
    rk, m, rc = cell2t
    # The reference loops know the skeleton spheres only; the fitted envelope
    # (link_collision) is compared by its own tests.
    monkeypatch.setattr(rk, "_envelope_world", lambda *a, **k: (None, None))
    obs = [o for o in rk.load_static_obstacles("welding_cell_2_small") if o.get("type") == "box"]
    pol = rk._load_clearance_policy("welding_cell_2_small")
    radii = rk._cell_sphere_radii("welding_cell_2_small", m, rc.get("tool_collision"))
    base, yaw = np.array(rc["base_position"]), float(rc.get("base_orientation_yaw", 0))
    rng = np.random.default_rng(3)
    no_tool = {"mount_link": rc["tool_collision"]["mount_link"]}
    seen_hit = seen_free = 0
    for q in rng.uniform(m.lowerPositionLimit, m.upperPositionLimit, (60, m.nq)):
        old = _old_arm_breach(rk, m, m.createData(), q, base, yaw, obs, pol["normal"], radii)
        d = m.createData()
        import pinocchio as pin
        pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
        P, R, _f, label = rk._static_bodies(m, d, base, yaw, rk._sphere_positions(m, d, base, yaw),
                                            pol["normal"], radii, no_tool)
        new = rk._static_first_breach(P, R, np.zeros(len(P), bool), label,
                                      rk._compile_static(obs), 0.0)
        assert (old is None) == (new is None)
        if old:
            assert (new["object_id"], new["link_index"]) == old
            seen_hit += 1
        else:
            seen_free += 1
    assert seen_hit and seen_free


def test_the_tool_is_checked_against_the_fixtures_and_the_floor(cell2t):
    rk, m, rc = cell2t
    tc = rc["tool_collision"]
    assert tc.get("static_spheres") and tc.get("static_radius_m") == pytest.approx(0.05)
    from agents.home_pose import home_pose
    import pinocchio as pin
    q = np.asarray(home_pose("welding_cell_2_small", "robot1"), float)
    d = m.createData(); pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    base, yaw = np.array(rc["base_position"]), float(rc.get("base_orientation_yaw", 0))
    P, R, floor, label = rk._static_bodies(m, d, base, yaw, rk._sphere_positions(m, d, base, yaw),
                                           0.12, None, tc)
    tool = [i for i, lab in enumerate(label) if isinstance(lab, tuple)]
    assert len(tool) == len(tc["static_spheres"]) and all(floor[i] for i in tool)
    probe = P[tool[len(tool) // 2]]
    hit = _breach([probe], [R[tool[0]]], [{"id": "probe", "type": "box", "center": probe.tolist(),
                                           "half_extents": [0.01] * 3, "collision_margin": 0.0}])
    assert hit is not None


def test_no_floor_literal_is_left():
    for f in ("agents/trajectory/tools/robot_kinematics.py", "agents/trajectory/tools/pinocchio.py"):
        src = (REPO / f).read_text()
        assert "< -0.05" not in src and ">= -0.05" not in src, f


# ── the converter writes it ──────────────────────────────────────────────────

def test_the_converter_writes_an_unambiguous_floor():
    import yaml
    import convert_isaac_scene as cv
    floor = [o for o in yaml.safe_load(cv.render_obstacles_yaml("c", [], "voxel", 0.05))[
        "obstacles"] if o["id"] == "floor"][0]
    assert floor == {"id": "floor", "type": "plane", "description": "Cell floor",
                     "normal": [0.0, 0.0, 1.0], "offset": 0.0, "collision_margin": 0.05}


# ── the tool keeps its own margin ────────────────────────────────────────────

def test_a_sphere_with_its_own_margin_uses_it_against_every_static_body():
    rk = _rk()
    box = [{"id": "b", "type": "box", "center": [0, 0, 1], "half_extents": [.1, .1, .1],
            "collision_margin": 0.05}]
    P = np.asarray([[0, 0, 1.17], [0, 0, 1.17]], float)        # 0.07 above the box top
    R = np.asarray([0.03, 0.03])
    arm_and_tool = rk._Labels([2, ("tool", 0)], [np.nan, 0.02])
    hit = rk._static_first_breach(P, R, np.zeros(2, bool), arm_and_tool,
                                  rk._compile_static(box), 0.0)
    # The arm sphere needs 0.03 + 0.05 = 0.08 (breach); the tool sphere 0.03 + 0.02.
    assert hit["link_index"] == 2
    tool_only = rk._Labels([("tool", 0)], [0.02])
    assert rk._static_first_breach(P[1:], R[1:], np.zeros(1, bool), tool_only,
                                   rk._compile_static(box), 0.0) is None


def test_the_converter_declares_the_tool_margin():
    import yaml
    import convert_isaac_scene as cv
    cfg = yaml.safe_load(cv.render_obstacles_yaml("c", [], "voxel", 0.05))
    assert cfg["tool_collision_margin"] == cv.DEFAULT_TOOL_COLLISION_MARGIN == 0.02


def test_the_policy_reads_the_tool_margin(monkeypatch):
    rk = _rk()
    monkeypatch.setattr(rk, "_read_obstacles_yaml", lambda c: {"tool_collision_margin": 0.03})
    assert rk._load_clearance_policy("c")["tool_margin"] == 0.03
    monkeypatch.setattr(rk, "_read_obstacles_yaml", lambda c: {})
    assert rk._load_clearance_policy("c")["tool_margin"] is None
