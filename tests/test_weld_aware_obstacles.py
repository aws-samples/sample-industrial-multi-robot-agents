"""Obstacles built around the welds, and welds checked against the part.

Measured on welding_cell_2_small (a car body, 27 spot welds, a 0.05 m voxel shell):

* the sampler lost 148 surface cells (1.1%) — long thin triangles; the voxel test
  is now exact (separating axes), so no cell the surface passes through is lost;
* every weld point sat inside an inflated box, so the welds were reachable only
  because 210 merged boxes (up to 2.05 m long, a quarter of the body) were flagged
  ``is_workpiece`` — and that flag hid them from the WHOLE arm. Around each weld
  the shell is now rebuilt finely and only the cells the tool touches on its
  approach stroke are ``tool_contact`` (the tool may touch them; the arm may not);
* three welds sat under another sheet (the approach axis met metal 18-51 mm in
  front) and eight approaches grazed metal beside the electrode: the converter
  moves such a point onto the outer surface and turns such an approach off the
  metal, writing the correction to the program and to its product definition.

Host-runnable: the converter is pure Python; the weld check is numpy.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import convert_isaac_scene as cv  # noqa: E402
from agents.trajectory import task_programs  # noqa: E402
from utils import weld_surface as ws  # noqa: E402


def _plate(x0, x1, y0, y1, z):
    """A horizontal rectangle at height z, two triangles."""
    return [((x0, y0, z), (x1, y0, z), (x1, y1, z)), ((x0, y0, z), (x1, y1, z), (x0, y1, z))]


def _wall(x, y0, y1, z0, z1):
    """A vertical rectangle in the plane x = const."""
    return [((x, y0, z0), (x, y1, z0), (x, y1, z1)), ((x, y0, z0), (x, y1, z1), (x, y0, z1))]


# ── the voxel test is exact ───────────────────────────────────────────────────

def _old_sampler(v0, v1, v2, voxel):
    """The half-voxel barycentric sampler the exact test replaced (256 cap)."""
    out = set()
    e1 = [v1[k] - v0[k] for k in range(3)]
    e2 = [v2[k] - v0[k] for k in range(3)]
    n = min(int(max(math.dist(v0, v1), math.dist(v0, v2), math.dist(v1, v2)) / (voxel / 2)) + 1, 256)
    for i in range(n + 1):
        for j in range(n - i + 1):
            p = [v0[k] + i / n * e1[k] + j / n * e2[k] for k in range(3)]
            out.add(tuple(math.floor(c / voxel) for c in p))
    return out


def test_a_long_thin_triangle_keeps_every_cell_it_crosses():
    sliver = ((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 0.004, 0.06))
    cells: set = set()
    cv._rasterize_triangle(*sliver, 0.05, cells)
    lost = cells - _old_sampler(*sliver, 0.05)
    assert lost, "the sampler this replaces must miss cells here"
    for c in lost:                    # each one really is crossed by the triangle
        centre = tuple((c[k] + 0.5) * 0.05 for k in range(3))
        assert cv._tri_box_overlap(centre, 0.025 + 1e-9, *sliver)


def test_a_face_on_a_cell_boundary_belongs_to_the_cell_above_only():
    cells: set = set()
    for tri in _plate(0.0, 0.1, 0.0, 0.1, 0.5):
        cv._rasterize_triangle(*tri, 0.1, cells)
    assert {c[2] for c in cells} == {5}


# ── weld points and approaches against the mesh ───────────────────────────────

def _tris(*parts):
    return ws.as_triangles([t for p in parts for t in p])


def test_a_weld_on_its_surface_with_a_free_approach_is_left_alone():
    T = _tris(_plate(-1, 1, -1, 1, 1.0))
    r = ws.correct_target(T, [0, 0, 1.0], [0, 0, -1])
    assert r["status"] == "ok" and r["moved_mm"] == 0 and r["approach_change_deg"] == 0


def test_a_weld_under_another_sheet_is_moved_onto_the_outer_one():
    # Inner sheet at z = 1.0 (the point), outer sheet 20 mm above it, approach down.
    T = _tris(_plate(-1, 1, -1, 1, 1.0), _plate(-1, 1, -1, 1, 1.02))
    r = ws.correct_target(T, [0, 0, 1.0], [0, 0, -1])
    assert r["status"] == "corrected"
    assert r["position"][2] == pytest.approx(1.02) and r["moved_mm"] == pytest.approx(20, abs=0.1)
    assert any("under another sheet" in why for why in r["reasons"])


def test_an_approach_grazing_metal_is_turned_off_it():
    # A wall 5 mm beside the weld, standing 50 mm up in front of it.
    T = _tris(_plate(-1, 1, -1, 1, 1.0), _wall(0.005, -1, 1, 1.01, 1.05))
    r = ws.correct_target(T, [0, 0, 1.0], [0, 0, -1])
    assert r["status"] == "corrected" and r["corridor_blocked_at_m"] is not None
    assert 0 < r["approach_change_deg"] <= 45
    assert r["approach_vector"][0] > 0          # leaning away from the wall (at +x)
    assert ws.corridor_blocked(T, np.array(r["position"]), -np.array(r["approach_vector"]),
                               ws.CORRIDOR_RADIUS_M, ws.CORRIDOR_LENGTH_M) is None


def test_the_normal_is_the_face_the_weld_is_on_not_an_average_over_a_bend():
    # A flange: the weld face is horizontal, a wall rises 4 mm away. Averaged over
    # 10 mm the "normal" would lean toward the wall; the face's own does not.
    T = _tris(_plate(-1, 0.004, -1, 1, 1.0), _wall(0.004, -1, 1, 1.0, 1.1))
    n = ws.surface_normal(T, np.array([0.0, 0.0, 1.0]), np.array([0, 0, 1.0]))
    assert n == pytest.approx([0, 0, 1], abs=1e-9)


def test_no_free_approach_is_reported_not_forced():
    # A closed box around the weld: nothing reaches it.
    T = _tris(_plate(-1, 1, -1, 1, 1.0), _plate(-0.3, 0.3, -0.3, 0.3, 1.1),
              _wall(0.2, -1, 1, 1.0, 1.2), _wall(-0.2, -1, 1, 1.0, 1.2),
              [((-1, 0.2, 1.0), (1, 0.2, 1.0), (1, 0.2, 1.2)), ((-1, 0.2, 1.0), (1, 0.2, 1.2), (-1, 0.2, 1.2))],
              [((-1, -0.2, 1.0), (1, -0.2, 1.0), (1, -0.2, 1.2)), ((-1, -0.2, 1.0), (1, -0.2, 1.2), (-1, -0.2, 1.2))])
    r = ws.correct_target(T, [0, 0, 1.0], [0, 0, -1], max_stack_m=0.05)
    assert r["status"] == "unresolved"
    assert r["approach_vector"] == pytest.approx([0, 0, -1])


def test_the_orientation_follows_a_turned_approach():
    q = {"x": 0.0, "y": -0.7071068, "z": 0.0, "w": 0.7071068}  # tool -x points down
    R = ws.rotation_between([0, 0, -1], [math.sin(0.2), 0, -math.cos(0.2)])
    q2 = ws.rotate_quaternion(q, R)
    w, x, y, z = q2["w"], q2["x"], q2["y"], q2["z"]
    minus_x = -np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
    assert minus_x == pytest.approx([math.sin(0.2), 0, -math.cos(0.2)], abs=1e-5)


# ── the shell around a weld ───────────────────────────────────────────────────

def _weld(p, out=(0, 0, 1), reach=0.10, clear=True):
    return {"p": p, "out": out, "reach": reach, "corridor": 0.008, "stroke": 0.10,
            "clear": clear}


def test_only_the_stroke_neighbourhood_is_the_tools_and_it_is_fine():
    tris = _plate(-1, 1, -1, 1, 1.0)
    vox = cv.weld_aware_voxels(tris, 0.05, 0.025, [_weld((0.0, 0.0, 1.0))], 0.12)
    assert vox["contact"] and vox["fine"] and vox["coarse"]
    for cell in vox["contact"]:
        c = [(cell[k] + 0.5) * 0.025 for k in range(3)]
        assert math.dist(c, (0, 0, 1.0)) <= 0.10 + 0.025 * math.sqrt(3) / 2 + 1e-9
    boxes = cv.weld_aware_boxes(vox, 0.05, 0.025)
    # A contact box never reaches past the weld's neighbourhood.
    for b in boxes:
        if b.get("tool_contact"):
            assert all(abs(b["c"][k]) + b["h"][k] <= 0.15 for k in range(2))
    # Far from the weld the shell is the coarse one, as before.
    assert any(not b.get("fine") and abs(b["c"][0]) > 0.5 for b in boxes)


def test_metal_in_front_of_a_blocked_weld_stays_an_obstacle_for_the_tool():
    tris = _plate(-1, 1, -1, 1, 1.0) + _plate(-0.05, 0.05, -0.05, 0.05, 1.03)
    blocked = cv.weld_aware_voxels(tris, 0.05, 0.025, [_weld((0.0, 0.0, 1.0), clear=False)], 0.12)
    # Cells in front of the surface ON the electrode's axis: obstacles for the tool.
    on_axis = [c for c in blocked["fine"] | blocked["contact"]
               if (c[2] + 0.5) * 0.025 > 1.02
               and math.hypot((c[0] + 0.5) * 0.025, (c[1] + 0.5) * 0.025)
               < 0.008 + 0.025 * math.sqrt(3) / 2]
    assert on_axis and not any(c in blocked["contact"] for c in on_axis)
    clear = cv.weld_aware_voxels(tris, 0.05, 0.025, [_weld((0.0, 0.0, 1.0), clear=True)], 0.12)
    assert all(c in clear["contact"] for c in on_axis)


# ── the planner reads tool_contact as the tool's only ─────────────────────────

def _rk():
    import importlib
    return importlib.import_module("agents.trajectory.tools.robot_kinematics")


def test_a_tool_contact_box_is_the_tools_to_touch_not_the_arms():
    rk = _rk()
    box = [{"id": "sheet", "type": "box", "center": [0, 0, 1], "half_extents": [.1, .1, .01],
            "collision_margin": 0.05, "tool_contact": True}]
    P = np.asarray([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
    R = np.asarray([0.05, 0.05, 0.12])
    labels = rk._Labels([17, ("tool", 0), 9], [np.nan, 0.02, np.nan],
                        tool=[True, True, False])         # tool frame, tool sphere, arm
    hit = rk._static_first_breach(P, R, np.zeros(3, bool), labels, rk._compile_static(box), 0.0)
    assert hit is not None and hit["link_index"] == 9      # the arm is still checked
    tool_only = rk._Labels([17, ("tool", 0)], [np.nan, 0.02], tool=[True, True])
    assert rk._static_first_breach(P[:2], R[:2], np.zeros(2, bool), tool_only,
                                   rk._compile_static(box), 0.0) is None


def test_is_workpiece_in_a_cell_file_is_read_as_tool_only():
    rk = _rk()
    out = rk.normalize_static_obstacles([{"id": "w", "type": "box", "is_workpiece": True},
                                         {"id": "b", "type": "box"}])
    assert out[0]["tool_contact"] is True and "is_workpiece" not in out[0]
    assert "tool_contact" not in out[1]


# ── the converter, end to end ─────────────────────────────────────────────────

def _cell(tmp_path):
    """A cell with a sheet fixture (an inner sheet with an outer one over part of
    it), an existing obstacles.yaml with a custom floor, a hand-added box and a
    tuned policy, a task program and its product definition."""
    cell = tmp_path / "c"
    (cell / "fixtures").mkdir(parents=True)
    tri = _plate(-0.5, 0.5, -0.5, 0.5, 1.0) + _plate(0.2, 0.4, -0.1, 0.1, 1.02)
    obj = []
    for t in tri:
        for v in t:
            obj.append("v %f %f %f" % v)
    obj += [f"f {3 * i + 1} {3 * i + 2} {3 * i + 3}" for i in range(len(tri))]
    (cell / "fixtures" / "sheet.obj").write_text("\n".join(obj) + "\n")
    (cell / "fixtures" / "sheet.urdf").write_text(
        '<robot name="sheet"><link name="l"><visual><geometry>'
        '<mesh filename="sheet.obj"/></geometry></visual></link></robot>')
    (cell / "scene.yaml").write_text(yaml.safe_dump({
        "name": "c", "robots": [],
        "fixtures": [{"id": "sheet", "urdf": "fixtures/sheet.urdf",
                      "position": {"x": 0, "y": 0, "z": 0}}]}))
    ap = cell / "agentic_planner"
    (ap / "config").mkdir(parents=True)
    (ap / "tasks").mkdir()
    (ap / "config" / "obstacles.yaml").write_text(
        "# old header\nobstacles:\n"
        "  - id: floor\n    type: plane\n    normal: [0.0, 0.0, 1.0]\n    offset: -0.05\n"
        "    collision_margin: 0.05\n"
        "  - id: sheet_v0\n    type: box\n    center: [0, 0, 1]\n    half_extents: [0.5, 0.5, 0.05]\n"
        "    collision_margin: 0.05\n    is_workpiece: true\n"
        "  - id: clamp\n    type: box\n    center: [2, 0, 1]\n    half_extents: [0.1, 0.1, 0.1]\n"
        "    collision_margin: 0.05\n"
        "\ndefault_collision_margin: 0.05\nrobot_link_radius: 0.12\ntool_link_radius: 0.05\n"
        "custom_key: 7\n")
    q = {"x": 0.0, "y": -0.7071068, "z": 0.0, "w": 0.7071068}
    ops = [{"id": "OK1", "type": "weld", "position": [-0.2, 0.0, 1.0], "approach_vector": [0, 0, -1],
            "orientation": q, "weld_orientation": q},
           {"id": "UNDER", "type": "weld", "position": [0.3, 0.0, 1.0], "approach_vector": [0, 0, -1],
            "orientation": q, "weld_orientation": q}]
    (ap / "tasks" / "prog.json").write_text(json.dumps(
        {"program": "prog", "source": {"product": "part"}, "operations": ops}))
    (cell / "products" / "part").mkdir(parents=True)
    (cell / "products" / "part" / "product.json").write_text(json.dumps(
        {"name": "part", "tasks": [{"operations": [
            {"id": "OK1", "target": {"position": [-0.2, 0, 1.0], "normal": [0, 0, 1]}},
            {"id": "UNDER", "target": {"position": [0.3, 0, 1.0], "normal": [0, 0, 1]}}]}]}))
    return cell


def _run(cell, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/convert_isaac_scene.py"),
                        "--scene-yaml", str(cell / "scene.yaml"), "--out-dir", str(cell),
                        "--only-obstacles", *extra], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_only_obstacles_rebuilds_the_fixture_and_keeps_everything_else(tmp_path):
    cell = _cell(tmp_path)
    robots_before = sorted(p.name for p in (cell / "agentic_planner").rglob("*"))
    _run(cell)
    text = (cell / "agentic_planner" / "config" / "obstacles.yaml").read_text()
    cfg = yaml.safe_load(text)
    ids = [o["id"] for o in cfg["obstacles"]]
    floor = [o for o in cfg["obstacles"] if o["id"] == "floor"][0]
    assert floor["offset"] == -0.05                         # the floor as declared
    assert "clamp" in ids                                   # a hand-added box is kept
    assert not any(o.get("is_workpiece") for o in cfg["obstacles"])
    assert any(o.get("tool_contact") for o in cfg["obstacles"])
    assert cfg["custom_key"] == 7 and cfg["tool_link_radius"] == 0.05   # policy kept
    assert not (cell / "agentic_planner" / "cell.yaml").exists()        # nothing else written
    assert not (cell / "agentic_planner" / "config" / "robots.yaml").exists()
    assert list((cell / "agentic_planner" / "config").glob(".obstacles.yaml.bak-*"))
    assert robots_before                                    # (sanity)


def test_a_weld_under_another_sheet_is_corrected_in_the_program_and_the_product(tmp_path):
    cell = _cell(tmp_path)
    _run(cell)
    prog = json.loads((cell / "agentic_planner" / "tasks" / "prog.json").read_text())
    ops = {o["id"]: o for o in prog["operations"]}
    assert "weld_correction" not in ops["OK1"] and ops["OK1"]["position"] == [-0.2, 0.0, 1.0]
    under = ops["UNDER"]
    assert under["position"][2] == pytest.approx(1.02)
    rec = under["weld_correction"]
    assert rec["status"] == "corrected" and rec["from_position"] == [0.3, 0.0, 1.0]
    # A corrected weld is measured geometry, not a placeholder stand-in.
    assert not task_programs.is_placeholder(under)
    prod = json.loads((cell / "products" / "part" / "product.json").read_text())
    tgt = {o["id"]: o["target"] for o in prod["tasks"][0]["operations"]}
    assert tgt["UNDER"]["position"][2] == pytest.approx(1.02)
    assert tgt["UNDER"]["normal"] == pytest.approx([0, 0, 1])
    assert list((cell / "agentic_planner" / "tasks").glob(".prog.json.bak-*"))
    # Re-running finds it on its surface now: nothing more to correct.
    _run(cell)
    again = json.loads((cell / "agentic_planner" / "tasks" / "prog.json").read_text())
    assert {o["id"]: o["position"] for o in again["operations"]} == \
        {o["id"]: o["position"] for o in prog["operations"]}


def test_report_mode_changes_no_program(tmp_path):
    cell = _cell(tmp_path)
    before = (cell / "agentic_planner" / "tasks" / "prog.json").read_text()
    r = _run(cell, "--weld-correction", "report")
    assert "UNDER" in r.stdout + r.stderr
    assert (cell / "agentic_planner" / "tasks" / "prog.json").read_text() == before
