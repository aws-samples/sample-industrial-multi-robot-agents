"""The converter DERIVES each tool's collision model from the tool's meshes.

``tool_collision`` (robots.yaml) and ``tool_link_radius`` (obstacles.yaml) were a
fixed placeholder, then hand-measured values fitted to a table that dropped the
URDF ``<origin rpy>`` (umarell upstream issue 02: 72.9 % coverage, worst gap
0.295 m, 21 of 35 centres in empty space). A hand edit of a generated file is
lost on the next convert, so the fix lives in the converter:

* ``tool_triangles`` places every mesh on the tip link (and below it, through
  fixed joints) with its FULL ``<origin xyz rpy>``, in the tip-link frame;
* ``utils.tool_geometry.fit_tool_spheres`` covers that surface at the arm radius
  with centres ON the metal;
* the runtime reads the fitted ``spheres:`` list.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

np = pytest.importorskip("numpy")
import convert_isaac_scene as cv  # noqa: E402
from utils import tool_geometry as tg  # noqa: E402


def _box_obj(path: Path, lo, hi) -> None:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)]
    path.write_text("".join(f"v {a} {b} {c}\n" for a, b, c in v)
                    + "".join("f " + " ".join(map(str, q)) + "\n" for q in f))


def _urdf(tmp_path: Path, collisions: str, extra: str = "") -> Path:
    u = tmp_path / "arm.urdf"
    u.write_text(f"""<robot name="arm">
  <link name="base"/>
  <link name="flange">{collisions}</link>
  <joint name="a6" type="revolute"><parent link="base"/><child link="flange"/>
    <limit lower="-1" upper="1" velocity="1"/></joint>
  {extra}
</robot>""")
    return u


def _col(mesh: str, xyz="0 0 0", rpy="0 0 0") -> str:
    return (f'<collision><origin xyz="{xyz}" rpy="{rpy}"/>'
            f'<geometry><mesh filename="{mesh}"/></geometry></collision>')


# ── placement ────────────────────────────────────────────────────────────────

def test_the_tool_is_placed_with_its_full_origin_rotation(tmp_path):
    """A 0.6 m bar along +y, rotated -90 deg about z by its origin, lies along +x."""
    _box_obj(tmp_path / "bar.obj", (-0.02, 0.0, -0.02), (0.02, 0.6, 0.02))
    u = _urdf(tmp_path, _col("bar.obj", xyz="0.1 0 0", rpy=f"0 0 {-math.pi / 2}"))
    pts = np.array([v for t in cv.tool_triangles(u, "flange") for v in t])
    assert pts[:, 0].min() == pytest.approx(0.1, abs=1e-6)
    assert pts[:, 0].max() == pytest.approx(0.7, abs=1e-6)
    assert abs(pts[:, 1]).max() <= 0.02 + 1e-6, "dropping the rpy would leave it along y"


def test_links_below_the_tip_through_fixed_joints_belong_to_the_tool(tmp_path):
    _box_obj(tmp_path / "b.obj", (0, 0, 0), (0.1, 0.1, 0.1))
    extra = ('<link name="gun">' + _col("b.obj") + '</link>'
             '<joint name="mount" type="fixed"><parent link="flange"/><child link="gun"/>'
             '<origin xyz="0 0 0.5"/></joint>')
    u = _urdf(tmp_path, "", extra)
    pts = np.array([v for t in cv.tool_triangles(u, "flange") for v in t])
    assert pts[:, 2].min() == pytest.approx(0.5) and pts[:, 2].max() == pytest.approx(0.6)


def test_no_tool_meshes_means_no_fit(tmp_path):
    u = _urdf(tmp_path, "")
    assert cv.tool_triangles(u, "flange") == []
    assert cv.fit_tool_model(u, "flange", (-0.9, 0, 0), 0.12) is None


# ── the fit ──────────────────────────────────────────────────────────────────

def _tool(tmp_path):
    """A 0.6 m bar on the -x axis to a TCP at x=-0.9, and a 0.5 x 0.4 plate."""
    _box_obj(tmp_path / "bar.obj", (-0.9, -0.03, -0.03), (-0.3, 0.03, 0.03))
    _box_obj(tmp_path / "plate.obj", (-0.32, -0.2, -0.25), (-0.28, 0.2, 0.25))
    return _urdf(tmp_path, _col("bar.obj") + _col("plate.obj"))


def test_the_fit_covers_the_whole_tool_from_the_metal(tmp_path):
    fit = cv.fit_tool_model(_tool(tmp_path), "flange", (-0.9, 0, 0), 0.12)
    assert fit["vertex_coverage"] == 1.0 and fit["sample_coverage"] == 1.0
    assert fit["worst_gap_m"] <= 0.12 - tg.FIT_MARGIN_M + 1e-9
    assert fit["max_centre_to_metal_m"] < 0.02, "a centre in empty space"
    c = np.array(fit["centres"])
    assert np.all(np.diff(np.linalg.norm(c, axis=1)) >= -1e-9), "flange -> tip order"
    assert 4 <= len(c) <= 30          # a greedy cover, not hundreds of spheres


def test_the_fit_is_deterministic(tmp_path):
    u = _tool(tmp_path)
    assert cv.fit_tool_model(u, "flange", (-0.9, 0, 0), 0.12)["centres"] == \
        cv.fit_tool_model(u, "flange", (-0.9, 0, 0), 0.12)["centres"]


def test_the_tcp_radius_is_the_tools_width_near_the_tcp(tmp_path):
    fit = cv.fit_tool_model(_tool(tmp_path), "flange", (-0.9, 0, 0), 0.12)
    # bar half-width 0.03 -> corner 0.0424 m from the axis -> rounded up to 0.045
    assert fit["tool_link_radius"] == pytest.approx(0.045)
    assert tg.tool_link_radius(cv.tool_triangles(_tool(tmp_path), "flange"), (0, 0, 0)) is None


# ── what the converter writes ────────────────────────────────────────────────

def _entry(fit, tip="flange"):
    return {"namespace": "arms", "robot_id": "robot1", "model_name": "m", "robot_model": "m",
            "isaac_prim": "r", "base_position": (0, 0, 0), "yaw": 0.0, "pose_source": "s",
            "lowers": [-1], "uppers": [1], "vels": [1], "tip_link": tip, "tool_fit": fit}


_DEFAULTS = {"gripper": "g", "workspace_radius": 2.9, "planning_group": "p",
             "end_effector_link": "tcp", "tcp_offset": (-0.9, 0, 0), "accel": [1],
             "rrt_j2": [0, 1], "rrt_j3": [0, 1], "mount_link": "link_6",
             "tool_axis_dir": [-1, 0, 0], "tool_axis_start": 0.15, "tool_axis_end": 0.9,
             "tool_axis_step": 0.15, "thermal_spread": False}


def test_robots_yaml_carries_the_fitted_spheres(tmp_path):
    import yaml
    fit = cv.fit_tool_model(_tool(tmp_path), "flange", (-0.9, 0, 0), 0.12)
    text = cv.render_robots_yaml("c", "s", [_entry(fit)], _DEFAULTS)
    cfg = yaml.safe_load(text)
    tc = cfg["robots"]["arms"]["robot1"]["tool_collision"]
    assert tc["mount_link"] == "flange" and "axis" not in tc and "plates" not in tc
    assert tc["spheres"] == fit["centres"]
    assert "DERIVED from the tool's own meshes" in text
    assert cfg["planning"]["thermal_spread"] is False


def test_without_tool_meshes_the_placeholder_is_written_and_flagged():
    import yaml
    text = cv.render_robots_yaml("c", "s", [_entry(None)], _DEFAULTS)
    tc = yaml.safe_load(text)["robots"]["arms"]["robot1"]["tool_collision"]
    assert tc["axis"]["start_m"] == 0.15 and "spheres" not in tc
    assert "REVIEW: tool collision geometry is a generic placeholder" in text


def test_obstacles_yaml_carries_the_derived_tool_radius():
    import yaml
    derived = yaml.safe_load(cv.render_obstacles_yaml(
        "c", [], "voxel", 0.05, {"value": 0.065, "models": "m"}))
    assert derived["tool_link_radius"] == 0.065
    assert derived["robot_link_radius"] == cv.ROBOT_LINK_RADIUS
    default = yaml.safe_load(cv.render_obstacles_yaml("c", [], "voxel", 0.05))
    assert default["tool_link_radius"] == cv.DEFAULT_TOOL_LINK_RADIUS


# ── the runtime reads it ─────────────────────────────────────────────────────

def test_the_runtime_builds_the_fitted_spheres():
    rk = pytest.importorskip("agents.trajectory.tools.robot_kinematics")
    cfg = {"mount_link": "flange", "spheres": [[-0.3, 0, 0], [-0.9, 0.01, 0]]}
    offs = rk._tool_offsets(cfg)
    assert [list(o) for o in offs] == [[-0.3, 0, 0], [-0.9, 0.01, 0]]


# ── the real tool of welding_cell_2_small ────────────────────────────────────

def test_the_sticktool_is_covered_from_its_own_hulls():
    urdf = REPO / "cells/welding_cell_2_small/robots/kr240r2900__sticktool/kr240r2900__sticktool.urdf"
    if not urdf.is_file():
        pytest.skip("the cell's robot assets are not in this checkout")
    fit = cv.fit_tool_model(urdf, "like_kr240r2900_link_6", (-0.9092, 0.0005, -0.0056), 0.12)
    assert fit["vertex_coverage"] == 1.0 and fit["sample_coverage"] == 1.0
    assert fit["max_centre_to_metal_m"] < 0.02 and len(fit["centres"]) < 35
