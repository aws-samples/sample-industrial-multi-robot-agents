#!/usr/bin/env python3
"""
convert_isaac_scene.py — Turn a scene export into a planning-ready cell.

Given a scene manifest (``scene.yaml``, optionally alongside the ``scene.usda``
it was exported with), this script writes every configuration file the planning
agents need — EXCEPT the task file (``tasks/*.json``), which describes the work
to perform and is authored separately.

It COPIES NOTHING. A scene bundle already contains the robots' dressed
URDF/SRDF, their meshes, the fixture geometry and the floor plan, and
``scene.yaml`` already says where each of those is. Copying them into the cell
would produce a second answer to "what shape is this arm" — the failure this
converter used to have. Instead it writes ``agentic_planner/cell.yaml``, the
manifest that *references* each asset where it already sits, and every consumer
resolves through :func:`utils.cell_layout.robot_urdf`.

What it writes under cells/<cell-id>/
─────────────────────────────────────
A cell has two layers, and this script writes into exactly one of them. The ROOT
is the scene bundle as its author shipped it and **nothing is written there** —
not one file. Everything this script produces is derived from the scene, so all of
it goes under ``agentic_planner/``, the planning agents' workspace: ``config/``
and the manifest included. ``utils.cell_layout`` owns that split; ask it for a
path rather than composing one here.

  agentic_planner/cell.yaml
                         the cell manifest. For each arm, the cell-local id
                         (``robot1``) ↔ the scene's own id
                         (``kr240r2900__sticktool``), plus the ``urdf:`` and
                         ``srdf:`` that describe it. Without the ``urdf:`` a
                         reader falls back to guessing a path; without a sibling
                         SRDF a collision checker retains every self-collision
                         pair and reports links that touch by design.
                         Every path in it is relative to the CELL directory, not
                         to the manifest's own — so ``robots/x/x.urdf`` still
                         names the bundle's file from in here.
  agentic_planner/config/robots.yaml
                         two-level robots -> <namespace> -> <robotN> layout.
                         base_position + base_orientation_yaw come from the
                         scene; joint limits from the arm's URDF; ``tip_link`` +
                         ``tcp_offset`` declare the tool frame (see below). The
                         remaining fields are defaults marked REVIEW.
  agentic_planner/config/cyclonedds.xml
                         the standard per-cell DDS config (auto network iface).
                         Named to the containers by ``cell_layout.config_uri``,
                         never by a hand-written path.
  agentic_planner/config/obstacles.yaml
                         the cell floor plane + fixture collision geometry,
                         plus the clearance policy. By default fixtures are
                         voxelised into an occupancy shell (--obstacle-mode
                         voxel) so open cavities stay clear; --obstacle-mode
                         aabb emits one conservative solid box per fixture.
  agentic_planner/{tasks,plans,logs,recordings}/
                         created empty, so a freshly converted cell is writable
                         by the agents without anyone mkdir-ing by hand.

Re-converting overwrites in place. Every file above is DERIVED from the scene, so
a second run simply recomputes it. Nothing is archived and nothing is renamed
aside — but note that hand-tuned REVIEW fields in ``robots.yaml`` and hand-refined
boxes in ``obstacles.yaml`` are overwritten with them.

The tool centre point
─────────────────────
The planner targets the electrode tip, not the flange, and a vendor dressed URDF
ends at ``link_6``. Because this converter no longer writes a modified copy of
that URDF, the tool frame is DECLARED rather than baked:

    end_effector_link: welding_tcp    # the frame to target
    tip_link:          <last movable joint's child link, from the URDF>
    tcp_offset:        [x, y, z]      # --tcp-offset, in the tip-link frame

``utils.arm_model`` attaches that frame when it loads the model, so every
consumer targets the same point. Pass --tcp-offset or every weld lands at the
flange.

Where geometry comes from
─────────────────────────
  Robot base positions come from ``scene.yaml``. Base yaw comes from the USD
  quaternion (``xformOp:orient``) when --scene is given — it is authoritative —
  and otherwise from ``scene.yaml``'s ``orientation.a``, which is the KUKA A
  angle about +Z **in degrees**. A non-zero B or C is warned about and dropped:
  this converter models a yaw-only base pose.

  Fixture collision geometry is voxelised from the fixture's own meshes, reached
  through the ``urdf:`` that ``scene.yaml`` names — full-resolution CAD, with
  each ``<visual>``/``<collision>`` entry's own ``<origin>`` and mesh ``scale``
  applied, then placed at the fixture's world pose. ``<collision>`` geometry is
  preferred when present. If that URDF does not resolve, the mesh data embedded
  in ``scene.usda`` is used instead; if there is neither, a placeholder box is
  emitted and marked REVIEW.

  The floor plan is a 2D CAD overlay for the viewer, not collision geometry: it
  is referenced, never voxelised. Cell walls are not obstacles here.

Usage
─────
  # A self-contained bundle: scene.yaml sits in the cell beside its assets.
  python3 scripts/convert_isaac_scene.py \
      --scene-yaml cells/my_cell/scene.yaml \
      --out-dir    cells/my_cell \
      --tcp-offset "-0.9092 0.0005 -0.0056"

  # An Isaac export handed over on its own: scene.yaml's asset paths are written
  # relative to the authoring workspace ("../urdf-dressed-robots/..."), so give
  # the root they were walking up to. Assets are RESOLVED there, not copied.
  python3 scripts/convert_isaac_scene.py \
      --scene       path/to/scene.usda \
      --scene-yaml  path/to/scene.yaml \
      --isaacsim-root-cell-path path/to/isaac_workspace \
      --out-dir     cells/my_new_cell

  # cell-id defaults to the --out-dir basename; override only if the deployed
  # directory name will differ:  --cell-id my_new_cell

This script has no third-party dependencies beyond PyYAML (already required by
the repo). It does not need the USD Python bindings (pxr), numpy, or trimesh.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

import yaml

# The cell layout — which directories a cell has and which of them belong to the
# agent workspace — is owned by utils.cell_layout. The converter CREATES that
# layout, so it must ask rather than compose its own idea of it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import cell_layout  # noqa: E402


MOVABLE_JOINT_TYPES = {"revolute", "prismatic", "continuous"}

# The arm's sphere radius (obstacles.yaml robot_link_radius). Also the radius the
# runtime gives the tool_collision spheres, so the tool model is fitted to it.
ROBOT_LINK_RADIUS = 0.12
# tool_link_radius when the tool cannot be measured (no meshes, no tcp_offset).
DEFAULT_TOOL_LINK_RADIUS = 0.05
# The tool's standoff from static bodies (obstacles.yaml tool_collision_margin).
DEFAULT_TOOL_COLLISION_MARGIN = 0.02

# The share of the axis gate the runtime keeps for the IK residual when it tilts
# a weld pose (agents/trajectory/tools/weld_orientation.AXIS_GATE_RESERVE_DEG).
TILT_RESERVE_DEG = 0.25


# ─── small helpers ────────────────────────────────────────────────────────────

def _warn(msg: str) -> None:
    print(f"[convert_isaac_scene] WARNING: {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[convert_isaac_scene] {msg}")


def _fatal(msg: str) -> "None":
    print(f"[convert_isaac_scene] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _fmt(v: float) -> str:
    """Format a float compactly (drops trailing zeros, keeps precision)."""
    return f"{v:.7g}"


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Yaw (rotation about +Z) from a quaternion (w, x, y, z)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _is_identity_quat(w: float, x: float, y: float, z: float, tol: float = 1e-6) -> bool:
    return abs(w - 1.0) < tol and abs(x) < tol and abs(y) < tol and abs(z) < tol


def _quat_rotate(q: tuple, v: tuple) -> tuple:
    """
    Rotate vector v by unit quaternion q = (w, x, y, z).
    v' = v + 2w(u x v) + 2(u x (u x v)),  u = (x, y, z).
    """
    w, ux, uy, uz = q
    vx, vy, vz = v
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)
    rx = vx + w * tx + (uy * tz - uz * ty)
    ry = vy + w * ty + (uz * tx - ux * tz)
    rz = vz + w * tz + (ux * ty - uy * tx)
    return (rx, ry, rz)


_IDENTITY_MAT4 = "((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))"


def _has_nonidentity_mat4(text: str) -> bool:
    """True if any `matrix4d xformOp:transform` in `text` is not the identity."""
    for m in re.finditer(r"matrix4d\s+xformOp:transform\s*=\s*(\(.*?\)\s*\))", text):
        compact = re.sub(r"\s+", "", m.group(1))
        # Normalise floats like 1.0 -> 1 for the comparison.
        compact = re.sub(r"(\d)\.0(?=[,)])", r"\1", compact)
        if compact != _IDENTITY_MAT4:
            return True
    return False


# ─── 3D transform helpers (stdlib only) ───────────────────────────────────────
#
# A 3x3 rotation is a tuple of three row tuples; a pose is (R, t). Kept in plain
# Python because this script must run on a bare host — the containers have numpy,
# the machine converting a scene may not.

_IDENTITY_R = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> tuple:
    """URDF ``rpy`` -> rotation matrix. Fixed-axis convention: R = Rz·Ry·Rx."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp,     cp * sr,                cp * cr),
    )


def _yaw_matrix(yaw: float) -> tuple:
    c, s = math.cos(yaw), math.sin(yaw)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _mat_mul(a: tuple, b: tuple) -> tuple:
    return tuple(
        tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
              for j in range(3))
        for i in range(3)
    )


def _mat_apply(r: tuple, v: tuple) -> tuple:
    return (r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2])


def _compose(pose_a: tuple, pose_b: tuple) -> tuple:
    """``pose_a ∘ pose_b`` — apply b first, then a."""
    ra, ta = pose_a
    rb, tb = pose_b
    rot = _mat_mul(ra, rb)
    off = _mat_apply(ra, tb)
    return rot, (ta[0] + off[0], ta[1] + off[1], ta[2] + off[2])


# ─── USD (text) parsing ───────────────────────────────────────────────────────
#
# We parse the .usda as plain text. This avoids a hard dependency on the USD
# Python bindings (pxr), which are not installed in this environment. The two
# things we extract are (1) each robot Xform's translate + orient and (2) the
# world-space geometry of each fixture Xform — used only as the FALLBACK for a
# fixture whose URDF does not resolve.

_FLOAT_TUPLE_RE = r"\(\s*([^)]*?)\s*\)"


def _match_block(text: str, open_index: int) -> str:
    """
    Given an index at/just-before a '{', return the text inside the matching
    braces (brace-balanced). Used to isolate one prim's subtree.
    """
    start = text.find("{", open_index)
    if start == -1:
        return ""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return text[start + 1:]  # unbalanced; return the rest


def _find_xform_block(text: str, prim_name: str) -> Optional[str]:
    """Return the brace-enclosed body of `def Xform "prim_name"`, or None."""
    m = re.search(r'def\s+Xform\s+"' + re.escape(prim_name) + r'"', text)
    if not m:
        return None
    return _match_block(text, m.end())


def _first_tuple(text: str, key: str) -> Optional[tuple]:
    """Return the first `key = (a, b, c, ...)` tuple of floats, or None."""
    m = re.search(re.escape(key) + r"\s*=\s*" + _FLOAT_TUPLE_RE, text)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",") if p.strip() != ""]
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def parse_robot_pose_from_usd(usd_text: str, robot_id: str) -> Optional[dict]:
    """
    Extract {'translate': (x,y,z), 'yaw': float, 'orient': (w,x,y,z)} for a
    robot Xform. Reads the prim-level translate/orient (the first ones in the
    block, before any child prim).
    """
    body = _find_xform_block(usd_text, robot_id)
    if body is None:
        return None
    # The prim-level ops appear before the first child `def`.
    head = body.split("\n            def ", 1)[0]
    head = head.split("\n        def ", 1)[0]
    translate = _first_tuple(head, "xformOp:translate") or (0.0, 0.0, 0.0)
    orient = _first_tuple(head, "xformOp:orient") or (1.0, 0.0, 0.0, 0.0)
    if len(orient) != 4:
        orient = (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = orient
    return {
        "translate": (translate[0], translate[1], translate[2] if len(translate) > 2 else 0.0),
        "orient": (w, x, y, z),
        "yaw": quat_to_yaw(w, x, y, z),
    }


def fixture_aabb_from_usd(usd_text: str, fixture_id: str) -> Optional[dict]:
    """
    Compute the world-space axis-aligned bounding box of a fixture Xform from
    the mesh point data embedded in the USD.

    World point ~= fixture_translate + mesh_translate + local_point, assuming
    identity rotations (checked; a non-identity rotation is warned about and
    approximated as axis-aligned).

    Returns {'min': (x,y,z), 'max': (x,y,z), 'rotated': bool} or None if the
    fixture has no mesh points.
    """
    body = _find_xform_block(usd_text, fixture_id)
    if body is None:
        return None

    head = body.split("\n            def ", 1)[0].split("\n        def ", 1)[0]
    ftrans = _first_tuple(head, "xformOp:translate") or (0.0, 0.0, 0.0)
    forient = _first_tuple(head, "xformOp:orient") or (1.0, 0.0, 0.0, 0.0)
    if len(forient) != 4:
        forient = (1.0, 0.0, 0.0, 0.0)
    ft = (ftrans[0], ftrans[1], ftrans[2] if len(ftrans) > 2 else 0.0)

    # We compose fixture(orient,translate) o mesh(orient,translate) o point,
    # assuming the intermediate rigid-body Xform (world_link) is identity —
    # which is the standard Isaac fixture export. Flag it if that is not true.
    approx = _has_nonidentity_mat4(body)

    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    found = False

    # Iterate mesh sub-blocks so each points array is paired with its own
    # mesh-level transform.
    for mm in re.finditer(r'def\s+Mesh\s+"[^"]*"', body):
        mesh_body = _match_block(body, mm.end())
        pm = re.search(r"point3f\[\]\s+points\s*=\s*\[([^\]]*)\]", mesh_body)
        if not pm:
            continue
        mtrans = _first_tuple(mesh_body, "xformOp:translate") or (0.0, 0.0, 0.0)
        morient = _first_tuple(mesh_body, "xformOp:orient") or (1.0, 0.0, 0.0, 0.0)
        if len(morient) != 4:
            morient = (1.0, 0.0, 0.0, 0.0)
        mt = (mtrans[0], mtrans[1], mtrans[2] if len(mtrans) > 2 else 0.0)

        for pt in re.finditer(_FLOAT_TUPLE_RE, pm.group(1)):
            comps = [c.strip() for c in pt.group(1).split(",") if c.strip() != ""]
            if len(comps) < 3:
                continue
            try:
                p = (float(comps[0]), float(comps[1]), float(comps[2]))
            except ValueError:
                continue
            # p_fixture = t_m + R_m * p_local
            rm = _quat_rotate(morient, p)
            pf = (mt[0] + rm[0], mt[1] + rm[1], mt[2] + rm[2])
            # p_world = t_f + R_f * p_fixture
            rf = _quat_rotate(forient, pf)
            wx, wy, wz = ft[0] + rf[0], ft[1] + rf[1], ft[2] + rf[2]
            if wx < mins[0]: mins[0] = wx
            if wy < mins[1]: mins[1] = wy
            if wz < mins[2]: mins[2] = wz
            if wx > maxs[0]: maxs[0] = wx
            if wy > maxs[1]: maxs[1] = wy
            if wz > maxs[2]: maxs[2] = wz
            found = True

    if not found:
        return None
    return {"min": tuple(mins), "max": tuple(maxs), "approx": approx}


def _parse_int_array(text: str, key: str) -> Optional[list]:
    """Parse `key = [i0, i1, ...]` into a list of ints, or None."""
    m = re.search(re.escape(key) + r"\s*=\s*\[([^\]]*)\]", text)
    if not m:
        return None
    out = []
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            try:
                out.append(int(float(tok)))
            except ValueError:
                pass
    return out


# ─── voxel occupancy ──────────────────────────────────────────────────────────
#
# Geometry-source-agnostic: everything below consumes world-space triangles, so
# the same pipeline serves USD point arrays and OBJ meshes reached through a
# fixture URDF.

# A voxel is the half-open cell [i*v, (i+1)*v): a surface lying exactly on a cell
# boundary belongs to the cell above it, never to both and never to neither. The
# exact test below enforces that by shifting each cell down by this much.
_VOXEL_EPS = 1e-9


def _tri_box_overlap(c: tuple, h: float, v0: tuple, v1: tuple, v2: tuple) -> bool:
    """Whether a triangle intersects the cube centred at ``c`` with half-size ``h``
    (separating-axis test, Akenine-Möller): the 3 box axes, the triangle normal and
    the 9 edge × axis cross products. Touching counts as overlapping."""
    a = (v0[0] - c[0], v0[1] - c[1], v0[2] - c[2])
    b = (v1[0] - c[0], v1[1] - c[1], v1[2] - c[2])
    d = (v2[0] - c[0], v2[1] - c[1], v2[2] - c[2])
    for k in range(3):
        lo = min(a[k], b[k], d[k])
        hi = max(a[k], b[k], d[k])
        if lo > h or hi < -h:
            return False
    e0 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e1 = (d[0] - b[0], d[1] - b[1], d[2] - b[2])
    e2 = (a[0] - d[0], a[1] - d[1], a[2] - d[2])
    for e in (e0, e1, e2):
        # axis = unit_k × e, for k = x, y, z
        for ax in ((0.0, -e[2], e[1]), (e[2], 0.0, -e[0]), (-e[1], e[0], 0.0)):
            pa = a[0] * ax[0] + a[1] * ax[1] + a[2] * ax[2]
            pb = b[0] * ax[0] + b[1] * ax[1] + b[2] * ax[2]
            pd = d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2]
            r = h * (abs(ax[0]) + abs(ax[1]) + abs(ax[2]))
            if min(pa, pb, pd) > r or max(pa, pb, pd) < -r:
                return False
    n = (e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2],
         e0[0] * e1[1] - e0[1] * e1[0])
    dist = n[0] * a[0] + n[1] * a[1] + n[2] * a[2]
    r = h * (abs(n[0]) + abs(n[1]) + abs(n[2]))
    return abs(dist) <= r


def _rasterize_triangle(v0: tuple, v1: tuple, v2: tuple, voxel: float,
                        occupied: set, keep=None) -> None:
    """
    Mark every voxel the triangle surface passes through — EXACTLY: each cell of
    the triangle's bounding range is tested with :func:`_tri_box_overlap`. The
    sampler this replaces walked the triangle at half-voxel spacing, capped at 256
    samples per edge, and lost long thin triangles: on welding_cell_2_small 148 surface
    cells (1.1%) had no box, 11 of them within 0.15 m of a weld. Grid origin is the
    world origin, so cell key = floor(coord / voxel). ``keep(i, j, k)``, when
    given, limits the marking to the cells it accepts.
    """
    h = voxel / 2.0
    lo = [math.floor((min(v0[k], v1[k], v2[k]) + _VOXEL_EPS) / voxel) for k in range(3)]
    hi = [math.floor((max(v0[k], v1[k], v2[k]) + _VOXEL_EPS) / voxel) for k in range(3)]
    for i in range(lo[0], hi[0] + 1):
        cx = (i + 0.5) * voxel - _VOXEL_EPS
        for j in range(lo[1], hi[1] + 1):
            cy = (j + 0.5) * voxel - _VOXEL_EPS
            for k in range(lo[2], hi[2] + 1):
                if (i, j, k) in occupied:
                    continue
                if keep is not None and not keep(i, j, k):
                    continue
                if _tri_box_overlap((cx, cy, (k + 0.5) * voxel - _VOXEL_EPS), h, v0, v1, v2):
                    occupied.add((i, j, k))


def voxelise_triangles(triangles: Iterator[tuple], voxel: float) -> Optional[dict]:
    """Rasterise world-space triangles into an occupancy set + its bounds.

    Returns the same shape :func:`fixture_voxels_from_usd` does, so the writer
    does not care which geometry source produced it.
    """
    occupied: set = set()
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for (a, b, c) in triangles:
        for v in (a, b, c):
            if v[0] < mins[0]: mins[0] = v[0]
            if v[1] < mins[1]: mins[1] = v[1]
            if v[2] < mins[2]: mins[2] = v[2]
            if v[0] > maxs[0]: maxs[0] = v[0]
            if v[1] > maxs[1]: maxs[1] = v[1]
            if v[2] > maxs[2]: maxs[2] = v[2]
        _rasterize_triangle(a, b, c, voxel, occupied)
    if not occupied:
        return None
    return {"voxels": occupied, "voxel": voxel, "approx": False,
            "min": tuple(mins), "max": tuple(maxs)}


def aabb_of_triangles(triangles: Iterator[tuple]) -> Optional[dict]:
    """World-space axis-aligned bounds of a triangle stream, or None if empty."""
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    found = False
    for tri in triangles:
        for v in tri:
            found = True
            if v[0] < mins[0]: mins[0] = v[0]
            if v[1] < mins[1]: mins[1] = v[1]
            if v[2] < mins[2]: mins[2] = v[2]
            if v[0] > maxs[0]: maxs[0] = v[0]
            if v[1] > maxs[1]: maxs[1] = v[1]
            if v[2] > maxs[2]: maxs[2] = v[2]
    if not found:
        return None
    return {"min": tuple(mins), "max": tuple(maxs), "approx": False}


def fixture_voxels_from_usd(usd_text: str, fixture_id: str,
                            voxel: float) -> Optional[dict]:
    """
    Build a surface-occupancy voxel set for a fixture from its USD mesh
    geometry (points + faceVertexIndices/faceVertexCounts), transformed to
    world. Only voxels the fixture surface passes through are marked, so open
    cavities and openings stay empty — a robot can still reach into them.

    Returns {'voxels': set[(i,j,k)], 'voxel': float, 'approx': bool,
             'min': (x,y,z), 'max': (x,y,z)} or None if no geometry was found.
    """
    body = _find_xform_block(usd_text, fixture_id)
    if body is None:
        return None

    head = body.split("\n            def ", 1)[0].split("\n        def ", 1)[0]
    ftrans = _first_tuple(head, "xformOp:translate") or (0.0, 0.0, 0.0)
    forient = _first_tuple(head, "xformOp:orient") or (1.0, 0.0, 0.0, 0.0)
    if len(forient) != 4:
        forient = (1.0, 0.0, 0.0, 0.0)
    ft = (ftrans[0], ftrans[1], ftrans[2] if len(ftrans) > 2 else 0.0)
    approx = _has_nonidentity_mat4(body)

    occupied: set = set()
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]

    # Prefer the fixture's dedicated collision meshes when present (they are the
    # geometry meant for collision, and using them alone avoids voxelising the
    # visually-detailed duplicate). Fall back to all meshes otherwise.
    mesh_iter = list(re.finditer(r'def\s+Mesh\s+"([^"]*)"', body))
    prefer_collision = any(m.group(1).lower().startswith("collision") for m in mesh_iter)

    for mm in mesh_iter:
        if prefer_collision and not mm.group(1).lower().startswith("collision"):
            continue
        mesh_body = _match_block(body, mm.end())
        pm = re.search(r"point3f\[\]\s+points\s*=\s*\[([^\]]*)\]", mesh_body)
        if not pm:
            continue
        mtrans = _first_tuple(mesh_body, "xformOp:translate") or (0.0, 0.0, 0.0)
        morient = _first_tuple(mesh_body, "xformOp:orient") or (1.0, 0.0, 0.0, 0.0)
        if len(morient) != 4:
            morient = (1.0, 0.0, 0.0, 0.0)
        mt = (mtrans[0], mtrans[1], mtrans[2] if len(mtrans) > 2 else 0.0)

        # Transform this mesh's vertices to world once.
        verts = []
        for pt in re.finditer(_FLOAT_TUPLE_RE, pm.group(1)):
            comps = [c.strip() for c in pt.group(1).split(",") if c.strip() != ""]
            if len(comps) < 3:
                continue
            try:
                p = (float(comps[0]), float(comps[1]), float(comps[2]))
            except ValueError:
                continue
            rm = _quat_rotate(morient, p)
            pf = (mt[0] + rm[0], mt[1] + rm[1], mt[2] + rm[2])
            rf = _quat_rotate(forient, pf)
            w = (ft[0] + rf[0], ft[1] + rf[1], ft[2] + rf[2])
            verts.append(w)
            if w[0] < mins[0]: mins[0] = w[0]
            if w[1] < mins[1]: mins[1] = w[1]
            if w[2] < mins[2]: mins[2] = w[2]
            if w[0] > maxs[0]: maxs[0] = w[0]
            if w[1] > maxs[1]: maxs[1] = w[1]
            if w[2] > maxs[2]: maxs[2] = w[2]

        if not verts:
            continue

        fvi = _parse_int_array(mesh_body, "faceVertexIndices")
        fvc = _parse_int_array(mesh_body, "faceVertexCounts")
        if fvi and fvc:
            idx = 0
            nverts = len(verts)
            for c in fvc:
                if c < 3 or idx + c > len(fvi):
                    idx += c
                    continue
                face = fvi[idx:idx + c]
                idx += c
                # Fan-triangulate the polygon.
                for k in range(1, c - 1):
                    a, b, d = face[0], face[k], face[k + 1]
                    if a >= nverts or b >= nverts or d >= nverts:
                        continue
                    _rasterize_triangle(verts[a], verts[b], verts[d], voxel, occupied)
        else:
            # No connectivity: fall back to marking vertices (dense meshes only).
            for w in verts:
                occupied.add((math.floor(w[0] / voxel),
                              math.floor(w[1] / voxel),
                              math.floor(w[2] / voxel)))

    if not occupied:
        return None
    return {"voxels": occupied, "voxel": voxel, "approx": approx,
            "min": tuple(mins), "max": tuple(maxs)}


def _merge_voxels(voxels: set) -> list:
    """
    Greedily cover an occupancy set of (i,j,k) cells with as few axis-aligned
    boxes as possible (classic 'greedy meshing': grow along i, then j, then k).
    Returns a list of (i0, j0, k0, i1, j1, k1) inclusive index ranges.
    """
    remaining = set(voxels)
    boxes = []
    while remaining:
        i0, j0, k0 = min(remaining)
        i1 = i0
        while (i1 + 1, j0, k0) in remaining:
            i1 += 1
        j1 = j0
        while all((i, j1 + 1, k0) in remaining for i in range(i0, i1 + 1)):
            j1 += 1
        k1 = k0
        while all((i, j, k1 + 1) in remaining
                  for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)):
            k1 += 1
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    remaining.discard((i, j, k))
        boxes.append((i0, j0, k0, i1, j1, k1))
    return boxes


def _voxel_boxes_world(merged: list, voxel: float) -> list:
    """Convert merged voxel index-ranges into world-space {'c','h'} boxes."""
    out = []
    for (i0, j0, k0, i1, j1, k1) in merged:
        x0, y0, z0 = i0 * voxel, j0 * voxel, k0 * voxel
        x1, y1, z1 = (i1 + 1) * voxel, (j1 + 1) * voxel, (k1 + 1) * voxel
        out.append({
            "c": ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0),
            "h": (max((x1 - x0) / 2.0, 1e-4),
                  max((y1 - y0) / 2.0, 1e-4),
                  max((z1 - z0) / 2.0, 1e-4)),
        })
    return out


# ─── weld-aware voxels ─────────────────────────────────────────────────────────
#
# A 0.05 m shell blurs a sheet by up to a voxel, and its 0.05 m margin by as much
# again: on welding_cell_2_small every weld point sat inside an inflated box and 19 of 27
# standoffs too, so the welds were reachable only by flagging whole merged boxes
# (up to 2.05 m long) as workpiece. Near each weld the shell is therefore rebuilt
# from the mesh at ``fine`` voxels, and only the cells the TOOL legitimately
# touches there are ``tool_contact``: those within ``reach`` (the tool's sphere
# radius + its margin) of the approach STROKE — the segment from the weld out to
# its standoff, which the TCP sphere sweeps. The electrode's own path was checked
# against the mesh itself (utils/weld_surface: corridor free of metal); the sphere
# is a much fatter stand-in for it. Where that corridor is NOT free (an unresolved
# weld), the cells in front of the surface around the axis stay obstacles for the
# tool too. Every arm link is checked against every box, contact or not.
#
# Each class is merged on its own, so a contact box never extends past its weld.

def weld_aware_voxels(triangles: list, coarse: float, fine: float, welds: list,
                      refine_radius: float) -> dict:
    """``{"coarse": set, "fine": set, "contact": set, ...}`` voxel index sets.

    ``welds``: ``[{"p": (x,y,z), "out": unit vector toward the tool, "reach": m,
    "corridor": m, "stroke": m, "clear": bool}]`` — ``stroke`` the standoff the
    TCP comes in from, ``clear`` whether the electrode's corridor is free of metal.
    ``fine`` must divide ``coarse`` (cells nest)."""
    ratio = coarse / fine
    if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
        raise ValueError(f"weld voxel {fine} must divide the voxel size {coarse}")
    ratio = int(round(ratio))
    # The contact cells must all be fine cells: never refine less than the reach.
    refine_radius = max([refine_radius] + [w["reach"] + fine for w in welds])
    # The stroke's far end is covered by the coarse cells it reaches (labelled below).
    shell = voxelise_triangles(iter(triangles), coarse)
    coarse_set = set(shell["voxels"]) if shell else set()
    half_diag = coarse * math.sqrt(3) / 2
    pts = [w["p"] for w in welds]

    def near_weld(c, r):
        return any((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2 <= r * r
                   for p in pts)

    refined = {cell for cell in coarse_set
               if near_weld(tuple((cell[k] + 0.5) * coarse for k in range(3)),
                            refine_radius + half_diag)}
    fine_set: set = set()
    if refined:
        keep = (lambda i, j, k: (i // ratio, j // ratio, k // ratio) in refined)
        r2 = refine_radius + 2 * half_diag
        for (a, b, c) in triangles:
            lo = [min(a[k], b[k], c[k]) for k in range(3)]
            hi = [max(a[k], b[k], c[k]) for k in range(3)]
            if not any(all(lo[k] - r2 <= p[k] <= hi[k] + r2 for k in range(3)) for p in pts):
                continue
            _rasterize_triangle(a, b, c, fine, fine_set, keep)
    def touched(c: tuple, slack: float) -> bool:
        """Whether a cell centred at ``c`` is the tool's to touch at some weld."""
        for w in welds:
            p, u = w["p"], w["out"]
            d = (c[0] - p[0], c[1] - p[1], c[2] - p[2])
            front = d[0] * u[0] + d[1] * u[1] + d[2] * u[2]
            along = min(max(front, 0.0), w.get("stroke", 0.0))
            q = (d[0] - along * u[0], d[1] - along * u[1], d[2] - along * u[2])
            if math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2) > w["reach"] + slack:
                continue
            if not w.get("clear", True):
                dist2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
                lateral = math.sqrt(max(dist2 - front * front, 0.0))
                if front > fine and lateral < w["corridor"] + slack:
                    continue                   # metal in the electrode's way
            return True
        return False

    fh = fine * math.sqrt(3) / 2
    contact = {cell for cell in fine_set
               if touched(tuple((cell[k] + 0.5) * fine for k in range(3)), fh)}
    coarse_kept = coarse_set - refined
    coarse_contact = {cell for cell in coarse_kept
                      if touched(tuple((cell[k] + 0.5) * coarse for k in range(3)), 0.0)}
    return {"coarse": coarse_kept - coarse_contact, "fine": fine_set - contact,
            "contact": contact, "coarse_contact": coarse_contact, "refined_cells": len(refined),
            "min": shell["min"] if shell else None, "max": shell["max"] if shell else None}


def weld_aware_boxes(vox: dict, coarse: float, fine: float) -> list:
    """World boxes for :func:`weld_aware_voxels`, each class merged on its own."""
    out = _voxel_boxes_world(_merge_voxels(vox["coarse"]), coarse)
    out += [{**b, "fine": True} for b in _voxel_boxes_world(_merge_voxels(vox["fine"]), fine)]
    out += [{**b, "fine": True, "tool_contact": True}
            for b in _voxel_boxes_world(_merge_voxels(vox["contact"]), fine)]
    out += [{**b, "tool_contact": True}
            for b in _voxel_boxes_world(_merge_voxels(vox.get("coarse_contact") or set()), coarse)]
    return out


# ─── weld targets ──────────────────────────────────────────────────────────────
#
# The task programs' weld points, checked against the fixture's own mesh
# (utils/weld_surface.py): moved onto the surface the electrode meets, the
# approach turned off metal in its corridor. ``--weld-correction apply`` writes
# the corrections back to the task file AND to the product definition it was
# derived from (so a re-derivation keeps them), each with a backup; every changed
# operation carries a ``weld_correction`` record of what moved and why.

def _backup(path: Path) -> Path:
    """A timestamped copy beside ``path`` (hidden), before it is overwritten."""
    import shutil
    import time
    dst = path.with_name(f".{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, dst)
    return dst


class WeldTargets:
    def __init__(self, paths: list, cell_dir: Path, args) -> None:
        self.cell_dir = Path(cell_dir)
        self.mode = getattr(args, "weld_correction", "apply")
        self.opts = {"corridor_radius_m": args.weld_corridor_radius,
                     "max_tilt_deg": args.max_weld_tilt,
                     "max_stack_m": args.weld_max_stack}
        self.files: list = []                       # [(path, data, changed)]
        self.ops: list = []                         # [(file index, op)]
        self.results: dict = {}                     # (file index, op id) -> result
        self.reach = 0.10
        self.stroke = float(getattr(args, "weld_standoff", 0.10))
        for path in paths:
            try:
                data = json.loads(Path(path).read_text())
            except (OSError, ValueError) as exc:
                _warn(f"--weld-targets {path}: unreadable ({exc}); skipped")
                continue
            fi = len(self.files)
            self.files.append([Path(path), data, False])
            for op in data.get("operations") or []:
                if (isinstance(op, dict) and op.get("position") and op.get("approach_vector")
                        and str(op.get("type", "weld")) == "weld"):
                    self.ops.append((fi, op))
        _info(f"weld targets: {len(self.ops)} weld(s) from {len(self.files)} program(s)")

    def set_reach(self, policy: dict) -> None:
        """The tool's static reach at the weld: its sphere radius + the margin it
        keeps (obstacles.yaml tool_link_radius + default_collision_margin)."""
        r = float(policy.get("tool_link_radius") or DEFAULT_TOOL_LINK_RADIUS)
        m = float(policy.get("default_collision_margin") or 0.05)
        self.reach = r + m

    def correct(self, fid: str, triangles: list) -> list:
        """Check (and in ``apply`` mode correct) every weld on this fixture; return
        the welds as :func:`weld_aware_voxels` takes them."""
        import numpy as np
        from utils import weld_surface as ws
        T = np.asarray(triangles, float).reshape(-1, 3, 3)
        if not len(T):
            return []
        lo, hi = T.min(axis=(0, 1)) - 0.05, T.max(axis=(0, 1)) + 0.05
        welds = []
        for fi, op in self.ops:
            p = np.asarray(op["position"], float)
            if not (np.all(p >= lo) and np.all(p <= hi)):
                continue
            r = ws.correct_target(T, p, op["approach_vector"], **self.opts)
            self.results[(fi, str(op.get("id")))] = (fid, r)
            if self.mode == "apply" and r["status"] == "corrected":
                self._apply(fi, op, r)
            pos = np.asarray(op["position"], float)
            a = ws._unit(op["approach_vector"])
            tilt = r["tilt_deg_after"] if r["tilt_deg_after"] is not None else (
                r["tilt_deg_before"] or 0.0)
            welds.append({"p": tuple(float(v) for v in pos),
                          "out": tuple(float(v) for v in -a),
                          "reach": self.reach / max(math.cos(math.radians(tilt)), 0.5),
                          "corridor": self.opts["corridor_radius_m"],
                          "stroke": self.stroke,
                          # the approach the program will use has a free corridor
                          "clear": r["status"] == "ok" or (
                              r["status"] == "corrected" and self.mode == "apply")})
        return welds

    def _apply(self, fi: int, op: dict, r: dict) -> None:
        import numpy as np
        from utils import weld_surface as ws
        a_old, a_new = op["approach_vector"], r["approach_vector"]
        R = ws.rotation_between(a_old, a_new)
        record = {"status": "corrected", "by": "convert_isaac_scene --weld-targets",
                  "from_position": [round(float(v), 6) for v in op["position"]],
                  "from_approach_vector": [round(float(v), 6) for v in a_old],
                  "moved_mm": r["moved_mm"], "approach_change_deg": r["approach_change_deg"],
                  "reasons": r["reasons"]}
        op["position"] = [round(float(v), 6) for v in r["position"]]
        op["approach_vector"] = [round(float(v), 6) for v in a_new]
        for key in ("orientation", "weld_orientation"):
            if isinstance(op.get(key), dict) and not np.allclose(R, np.eye(3)):
                op[key] = ws.rotate_quaternion(op[key], R)
        op["weld_correction"] = record
        self.files[fi][2] = True

    def finish(self) -> None:
        """Report every weld that is not ``ok``; in ``apply`` mode write the changed
        programs and their product definitions (backed up first)."""
        bad = [(k, v) for k, v in self.results.items() if v[1]["status"] != "ok"]
        for (fi, oid), (fid, r) in sorted(bad, key=lambda kv: kv[0][1]):
            verb = ("corrected" if r["status"] == "corrected" and self.mode == "apply"
                    else "UNRESOLVED" if r["status"] == "unresolved" else "would correct")
            _warn(f"weld {oid} ({self.files[fi][0].name}): {verb} — moved "
                  f"{r['moved_mm']:.1f} mm, approach turned {r['approach_change_deg']:.1f} deg: "
                  + "; ".join(r["reasons"]))
        _info(f"weld targets: {len(self.results) - len(bad)} on the surface with a free "
              f"approach, {sum(1 for _k, v in bad if v[1]['status'] == 'corrected')} "
              f"{'corrected' if self.mode == 'apply' else 'to correct'}, "
              f"{sum(1 for _k, v in bad if v[1]['status'] == 'unresolved')} unresolved")
        if self.mode != "apply":
            return
        for fi, (path, data, changed) in enumerate(self.files):
            if not changed:
                continue
            _backup(path)
            path.write_text(json.dumps(data, indent=2) + "\n")
            _info(f"wrote {path} (corrected welds; backup beside it)")
            self._update_product(data)

    def _update_product(self, program: dict) -> None:
        name = ((program.get("source") or {}).get("product"))
        if not name:
            return
        cands = [p for p in self.cell_dir.rglob("product.json")
                 if p.parent.name == name] if self.cell_dir.is_dir() else []
        if not cands:
            _warn(f"product '{name}' not found under {self.cell_dir}: its weld targets "
                  f"were NOT corrected — a re-derivation of the program would undo this")
            return
        path = cands[0]
        prod = json.loads(path.read_text())
        fixed = {str(o["id"]): o for o in program.get("operations") or []
                 if isinstance(o, dict) and (o.get("weld_correction") or {}).get("status") == "corrected"}
        n = 0
        for task in prod.get("tasks") or []:
            for op in task.get("operations") or []:
                src = fixed.get(str(op.get("id")))
                tgt = op.get("target") if isinstance(op, dict) else None
                if src is None or not isinstance(tgt, dict):
                    continue
                tgt["position"] = list(src["position"])
                if "normal" in tgt:
                    tgt["normal"] = [round(-float(v), 6) for v in src["approach_vector"]]
                n += 1
        if n:
            _backup(path)
            path.write_text(json.dumps(prod, indent=2) + "\n")
            _info(f"wrote {path} ({n} weld target(s) corrected; backup beside it)")


def splice_obstacles(existing: str, fresh: str, fixtures: list) -> str:
    """``existing`` obstacles.yaml with its fixture geometry replaced by ``fresh``'s.

    Kept from ``existing``: every obstacle that is not a rebuilt fixture's (the
    floor plane as declared, hand-added boxes and spheres) and everything after the
    obstacle list — the clearance policy, as tuned. Taken from ``fresh``: the
    header and the fixture blocks."""
    fids = {fx["id"] for fx in fixtures}
    pat = re.compile(r"^(?P<fid>.+)_v\d+$")

    def rebuilt(o: dict) -> bool:
        oid = str(o.get("id", ""))
        m = pat.match(oid)
        return oid in fids or (m is not None and m.group("fid") in fids)

    old = (yaml.safe_load(existing) or {}).get("obstacles") or []
    kept = [o for o in old if isinstance(o, dict) and not rebuilt(o)]
    lines = existing.split("\n")
    start = next(i for i, l in enumerate(lines) if l.rstrip() == "obstacles:")
    tail_at = next((i for i in range(start + 1, len(lines))
                    if lines[i] and not lines[i][0].isspace()), len(lines))
    tail = "\n".join(lines[tail_at:])
    head = fresh[:fresh.index("obstacles:\n")]
    first = fixtures[0]["id"] if fixtures else None
    body = ""
    if first is not None:
        k = fresh.find(f"\n  # {first}:")
        e = fresh.find("\n# Clearance policy")
        body = fresh[k:e if e >= 0 else len(fresh)].rstrip("\n") if k >= 0 else ""
    out = [head + "obstacles:"]
    for o in kept:
        dumped = yaml.safe_dump([o], sort_keys=False, default_flow_style=None).rstrip("\n")
        out.append("")
        out.extend("  " + l for l in dumped.split("\n"))
    return "\n".join(out) + ("\n" + body if body else "") + "\n\n" + tail.lstrip("\n")


# ─── OBJ meshes reached through a fixture URDF ────────────────────────────────
#
# This is where fixture geometry actually comes from: scene.yaml names the
# fixture's URDF, the URDF names one OBJ per <visual>/<collision> entry with its
# own <origin> and mesh scale, and the fixture as a whole sits at a world pose.
# A stdlib OBJ reader keeps that dependency-free — trimesh is not installed and
# is not in docker/requirements.txt.

def read_obj_triangles(path: Path) -> Iterator[tuple]:
    """Yield ``(v0, v1, v2)`` triangles from a Wavefront OBJ, in mesh-local units.

    Only ``v`` and ``f`` are read; normals, texture coords, materials and groups
    are irrelevant to occupancy. Faces may be polygons (fan-triangulated) and
    may index vertices negatively (OBJ's "counting back from here").
    """
    verts: list = []
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        _warn(f"cannot read mesh {path}: {exc}")
        return
    with fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        pass
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    spec = tok.split("/", 1)[0]
                    try:
                        i = int(spec)
                    except ValueError:
                        continue
                    # OBJ is 1-based; a negative index counts back from the end.
                    i = (len(verts) + i) if i < 0 else (i - 1)
                    if 0 <= i < len(verts):
                        idx.append(i)
                for k in range(1, len(idx) - 1):
                    yield (verts[idx[0]], verts[idx[k]], verts[idx[k + 1]])


def _parse_origin(el: Optional[ET.Element]) -> tuple:
    """A URDF ``<origin xyz rpy>`` as a ``(R, t)`` pose. Missing == identity."""
    if el is None:
        return _IDENTITY_R, (0.0, 0.0, 0.0)
    xyz = [float(v) for v in (el.get("xyz") or "0 0 0").replace(",", " ").split()]
    rpy = [float(v) for v in (el.get("rpy") or "0 0 0").replace(",", " ").split()]
    while len(xyz) < 3:
        xyz.append(0.0)
    while len(rpy) < 3:
        rpy.append(0.0)
    return _rpy_matrix(rpy[0], rpy[1], rpy[2]), (xyz[0], xyz[1], xyz[2])


def _link_poses(root: ET.Element) -> dict:
    """Each link's pose in the URDF's root frame, composed along fixed joints.

    A fixture is a rigid assembly, so its links are wired with fixed joints and
    this walk is exact. A movable joint is composed at its zero position and
    warned about — the alternative is silently dropping the geometry below it.
    """
    children: dict = {}
    for j in root.findall("joint"):
        parent_el, child_el = j.find("parent"), j.find("child")
        if parent_el is None or child_el is None:
            continue
        if j.get("type") in MOVABLE_JOINT_TYPES:
            _warn(f"fixture joint '{j.get('name')}' is {j.get('type')}; its "
                  f"geometry is placed at the zero position.")
        children.setdefault(parent_el.get("link"), []).append(
            (child_el.get("link"), _parse_origin(j.find("origin"))))

    names = [l.get("name") for l in root.findall("link") if l.get("name")]
    child_names = {c for kids in children.values() for c, _ in kids}
    roots = [n for n in names if n not in child_names] or names[:1]

    poses = {}
    stack = [(n, (_IDENTITY_R, (0.0, 0.0, 0.0))) for n in roots]
    while stack:
        name, pose = stack.pop()
        if name in poses:          # a cycle, or a link with two parents
            continue
        poses[name] = pose
        for child, joint_pose in children.get(name, []):
            stack.append((child, _compose(pose, joint_pose)))
    # Links reachable from no root at all still get placed at the origin rather
    # than dropped: geometry that is there is better than geometry that vanished.
    for n in names:
        poses.setdefault(n, (_IDENTITY_R, (0.0, 0.0, 0.0)))
    return poses


def urdf_geometry_triangles(urdf_path: Path, world_pose: tuple,
                            report_every: int = 200) -> Iterator[tuple]:
    """World-space triangles for every mesh a URDF references.

    ``<collision>`` entries are used when the URDF has any (that is the geometry
    authored for collision); otherwise ``<visual>``. Each entry contributes its
    own ``<origin>`` and mesh ``scale``, composed with its link's pose and the
    fixture's world pose — a fixture whose 894 mesh entries share one link still
    has 894 distinct offsets, and ignoring them collapses the whole assembly onto
    itself.
    """
    try:
        root = ET.fromstring(urdf_path.read_text())
    except (OSError, ET.ParseError) as exc:
        _warn(f"cannot parse {urdf_path}: {exc}")
        return

    poses = _link_poses(root)
    links = list(root.findall("link"))
    tag = "collision" if any(l.find("collision") is not None for l in links) else "visual"

    entries = []      # (mesh_path, (R, t), scale)
    missing = 0
    for link in links:
        link_pose = poses.get(link.get("name"), (_IDENTITY_R, (0.0, 0.0, 0.0)))
        for el in link.findall(tag):
            mesh = el.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            rel = mesh.get("filename").split("://", 1)[-1]
            mesh_path = Path(rel)
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / rel
            if not mesh_path.is_file():
                missing += 1
                continue
            scale = [float(v) for v in (mesh.get("scale") or "1 1 1").replace(",", " ").split()]
            while len(scale) < 3:
                scale.append(scale[-1] if scale else 1.0)
            entries.append((mesh_path,
                            _compose(world_pose, _compose(link_pose, _parse_origin(el.find("origin")))),
                            tuple(scale[:3])))

    if missing:
        _warn(f"{urdf_path.name}: {missing} referenced mesh file(s) are missing; "
              f"their geometry is absent from the obstacle model.")
    if not entries:
        return

    _info(f"  {urdf_path.name}: {len(entries)} <{tag}> mesh entr(ies) to read")
    for n, (mesh_path, (rot, off), scale) in enumerate(entries, start=1):
        if report_every and n % report_every == 0:
            _info(f"    …{n}/{len(entries)} meshes")
        sx, sy, sz = scale
        for tri in read_obj_triangles(mesh_path):
            out = []
            for v in tri:
                s = (v[0] * sx, v[1] * sy, v[2] * sz)
                r = _mat_apply(rot, s)
                out.append((r[0] + off[0], r[1] + off[1], r[2] + off[2]))
            yield (out[0], out[1], out[2])


def tool_triangles(urdf_path: Path, tip_link: str) -> list:
    """The tool's triangles in the TIP-LINK frame: every mesh attached to
    ``tip_link`` or to a link below it through fixed joints.

    ``<collision>`` meshes when the subtree has any, else ``<visual>``; each with
    its FULL ``<origin xyz rpy>`` and ``scale`` (umarell issue 02 was a tool model
    fitted to a table that dropped the rpy). Returns ``[]`` when nothing is there.
    """
    try:
        root = ET.fromstring(urdf_path.read_text())
    except (OSError, ET.ParseError) as exc:
        _warn(f"cannot parse {urdf_path}: {exc}")
        return []
    links = {l.get("name"): l for l in root.findall("link")}
    if tip_link not in links:
        return []
    poses = {tip_link: (_IDENTITY_R, (0.0, 0.0, 0.0))}
    stack = [tip_link]
    while stack:                                 # fixed-joint subtree below the tip
        parent = stack.pop()
        for j in root.findall("joint"):
            pe, ce = j.find("parent"), j.find("child")
            if pe is None or ce is None or pe.get("link") != parent:
                continue
            if j.get("type") in MOVABLE_JOINT_TYPES:
                continue                         # not rigidly part of the tool
            child = ce.get("link")
            if child in links and child not in poses:
                poses[child] = _compose(poses[parent], _parse_origin(j.find("origin")))
                stack.append(child)
    tag = ("collision" if any(links[n].find("collision") is not None for n in poses)
           else "visual")
    out: list = []
    for name, link_pose in poses.items():
        for el in links[name].findall(tag):
            mesh = el.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            rel = mesh.get("filename").split("://", 1)[-1]
            mesh_path = Path(rel)
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / rel
            if not mesh_path.is_file():
                _warn(f"{urdf_path.name}: tool mesh {rel} is missing")
                continue
            scale = [float(v) for v in (mesh.get("scale") or "1 1 1").replace(",", " ").split()]
            while len(scale) < 3:
                scale.append(scale[-1] if scale else 1.0)
            rot, off = _compose(link_pose, _parse_origin(el.find("origin")))
            for tri in read_obj_triangles(mesh_path):
                placed = []
                for v in tri:
                    r = _mat_apply(rot, (v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]))
                    placed.append((r[0] + off[0], r[1] + off[1], r[2] + off[2]))
                out.append(tuple(placed))
    return out


def link_triangles(urdf_path: Path, link: str) -> list:
    """Triangles of the ``<collision>`` meshes ON one link (not its subtree), in
    that link's frame, each with its full ``<origin xyz rpy>`` and scale."""
    try:
        root = ET.fromstring(urdf_path.read_text())
    except (OSError, ET.ParseError) as exc:
        _warn(f"cannot parse {urdf_path}: {exc}")
        return []
    el_link = next((l for l in root.findall("link") if l.get("name") == link), None)
    if el_link is None:
        return []
    out: list = []
    for el in el_link.findall("collision"):
        mesh = el.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            continue
        rel = mesh.get("filename").split("://", 1)[-1]
        mesh_path = Path(rel) if Path(rel).is_absolute() else urdf_path.parent / rel
        if not mesh_path.is_file():
            _warn(f"{urdf_path.name}: collision mesh {rel} of {link} is missing")
            continue
        scale = [float(v) for v in (mesh.get("scale") or "1 1 1").replace(",", " ").split()]
        while len(scale) < 3:
            scale.append(scale[-1] if scale else 1.0)
        rot, off = _parse_origin(el.find("origin"))
        for tri in read_obj_triangles(mesh_path):
            placed = []
            for v in tri:
                r = _mat_apply(rot, (v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]))
                placed.append((r[0] + off[0], r[1] + off[1], r[2] + off[2]))
            out.append(tuple(placed))
    return out


def fit_link_envelope(urdf_path: Path, tip_link: str, radius: float) -> Optional[dict]:
    """A sphere cover of EVERY link's collision hull (the tip link excepted — its
    flange and tool are the tool model), in each link's own frame.

    The arm used to be one sphere per frame plus fill spheres on the skeleton,
    covering 1-29 % of the KR240's hull vertices (umarell issue 04): the arm-vs-arm
    gate passed pose pairs whose castings interpenetrated by 300 mm. A cover of
    the hulls themselves, at the radius the runtime gives arm spheres, is an
    envelope by construction. ``None`` when the URDF has no collision meshes.
    """
    try:
        from utils import tool_geometry
    except ImportError as exc:
        _warn(f"cannot fit the link envelope without numpy ({exc})")
        return None
    try:
        root = ET.fromstring(urdf_path.read_text())
    except (OSError, ET.ParseError):
        return None
    links: dict = {}
    stats = {"vertex_coverage": 1.0, "worst_gap_m": 0.0, "spheres": 0}
    for l in root.findall("link"):
        name = l.get("name")
        if not name or name == tip_link or l.find("collision") is None:
            continue
        tris = link_triangles(urdf_path, name)
        if not tris:
            continue
        fit = tool_geometry.fit_tool_spheres(tris, radius)
        if fit is None:
            continue
        links[name] = fit["centres"]
        stats["vertex_coverage"] = min(stats["vertex_coverage"], fit["vertex_coverage"])
        stats["worst_gap_m"] = max(stats["worst_gap_m"], fit["worst_gap_m"])
        stats["spheres"] += len(fit["centres"])
    if not links:
        return None
    return {"links": links, "radius": radius, **stats}


def path_lever_arms(urdf_path: Path, tip_link: str, tcp_offset: tuple) -> Optional[dict]:
    """Per movable joint, the farthest any point of the arm (and tool, and TCP)
    beyond it can be from its axis — in every configuration.

    A point of link k sits at most ``sum of the joint offsets from joint j to k``
    plus ``k's own extent`` from joint j (triangle inequality along the chain), so
    a joint motion of d rad moves no point more than ``d * sum_j lever_j``. That is
    what turns a Cartesian path tolerance into the joint-space deviation the
    time-parameterisation may round a corner by. ``None`` without a chain.
    """
    try:
        root = ET.fromstring(urdf_path.read_text())
    except (OSError, ET.ParseError):
        return None
    joints = {}
    for j in root.findall("joint"):
        pe, ce = j.find("parent"), j.find("child")
        if pe is None or ce is None:
            continue
        _r, t = _parse_origin(j.find("origin"))
        joints[ce.get("link")] = (pe.get("link"), j.get("type"), math.dist(t, (0.0, 0.0, 0.0)))
    children: dict = {}
    for child, (parent, _type, _off) in joints.items():
        children.setdefault(parent, []).append(child)

    def extent(link: str) -> float:
        pts = [v for tri in link_triangles(urdf_path, link) for v in tri]
        if link == tip_link:
            pts += [v for tri in tool_triangles(urdf_path, tip_link) for v in tri]
            pts.append(tuple(tcp_offset))
        return max((math.dist(v, (0.0, 0.0, 0.0)) for v in pts), default=0.0)

    def reach(link: str) -> float:
        """Farthest point of ``link`` and everything below it, from its frame."""
        best = extent(link)
        for c in children.get(link, []):
            best = max(best, joints[c][2] + reach(c))
        return best

    levers = {joints[c][0] + "->" + c: reach(c) for c, (_p, typ, _o) in joints.items()
              if typ in MOVABLE_JOINT_TYPES}
    total = sum(levers.values())
    if not levers or total <= 0:
        return None                # no measurable geometry: nothing to bound by
    return {"levers": levers, "lever_sum_m": total}


def fit_tool_model(urdf_path: Path, tip_link: str, tcp_offset: tuple,
                   radius: float) -> Optional[dict]:
    """``tool_collision`` spheres and the TCP radius, derived from the tool meshes.

    ``None`` when the URDF carries no tool geometry below ``tip_link`` — the
    caller then writes the generic placeholder and says so.
    """
    try:
        from utils import tool_geometry          # numpy; the rest of the converter is not
    except ImportError as exc:
        _warn(f"cannot fit the tool model without numpy ({exc}); writing the placeholder")
        return None
    tris = tool_triangles(urdf_path, tip_link)
    if not tris:
        return None
    fit = tool_geometry.fit_tool_spheres(tris, radius)
    if fit is None:
        return None
    fit["tool_link_radius"] = tool_geometry.tool_link_radius(tris, tcp_offset)
    fit["triangles"] = len(tris)
    fit["radius"] = radius
    # The tool against the STATIC world (floor, fixtures) is checked at the tool's
    # own thickness, not at the pair radius: a cover at 0.12 m would claim up to
    # ~10 cm of metal that is not there next to a fixture. So a second, finer
    # cover is fitted at tool_link_radius.
    static_r = fit["tool_link_radius"] or DEFAULT_TOOL_LINK_RADIUS
    fine = tool_geometry.fit_tool_spheres(tris, static_r)
    if fine is not None:
        fit["static"] = {**fine, "radius": static_r}
    return fit


def self_collision_pairs(urdf_path: Path, srdf_path: Optional[Path]) -> dict:
    """Link pairs the self-collision check must NOT test: they touch by design.

    Every parent/child pair of the URDF joint tree (adjacent links share a joint
    and their hulls overlap there), plus every ``<disable_collisions>`` the SRDF
    declares. ``{"pairs": [[a, b], ...], "sources": [...]}``, sorted.
    """
    pairs: set = set()
    sources = []
    try:
        root = ET.fromstring(urdf_path.read_text())
        for j in root.findall("joint"):
            pe, ce = j.find("parent"), j.find("child")
            if pe is not None and ce is not None and pe.get("link") and ce.get("link"):
                pairs.add(tuple(sorted((pe.get("link"), ce.get("link")))))
        sources.append("adjacent links of the URDF joint tree")
    except (OSError, ET.ParseError) as exc:
        _warn(f"cannot read {urdf_path} for self-collision pairs: {exc}")
    if srdf_path is not None:
        try:
            sroot = ET.fromstring(srdf_path.read_text())
            n = 0
            for d in sroot.findall("disable_collisions"):
                a, b = d.get("link1"), d.get("link2")
                if a and b:
                    pairs.add(tuple(sorted((a, b))))
                    n += 1
            sources.append(f"{n} pair(s) disabled in {srdf_path.name}")
        except (OSError, ET.ParseError) as exc:
            _warn(f"cannot read {srdf_path} for self-collision pairs: {exc}")
    return {"pairs": sorted([list(p) for p in pairs]), "sources": sources}


# ─── robot model URDF parsing ─────────────────────────────────────────────────

def parse_urdf_model(urdf_text: str) -> dict:
    """
    Extract from the model URDF:
      robot_name       the <robot name="..."> value
      movable          ordered list of movable joints
                       [{name, lower, upper, velocity}, ...] (document order)
      tip_link         child link of the last movable joint (TCP parent)
    """
    root = ET.fromstring(urdf_text)
    robot_name = root.get("name") or "robot"

    movable = []
    tip_link = None
    for j in root.findall("joint"):
        if j.get("type") not in MOVABLE_JOINT_TYPES:
            continue
        lim = j.find("limit")
        lower = upper = vel = None
        if lim is not None:
            lo, up, ve = lim.get("lower"), lim.get("upper"), lim.get("velocity")
            lower = float(lo) if lo is not None else None
            upper = float(up) if up is not None else None
            vel = float(ve) if ve is not None else None
        child_el = j.find("child")
        child_link = child_el.get("link") if child_el is not None else None
        movable.append({
            "name": j.get("name"),
            "lower": lower,
            "upper": upper,
            "velocity": vel,
            "child": child_link,
        })
        if child_link:
            tip_link = child_link

    return {"robot_name": robot_name, "movable": movable, "tip_link": tip_link}


# ─── scene.yaml handling ──────────────────────────────────────────────────────

def load_scene_yaml(path: Path, need_robots: bool = True) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if need_robots and ("robots" not in data or not data["robots"]):
        _fatal(f"scene.yaml has no robots: {path}")
    return data


def robot_id_for_index(idx: int) -> str:
    """The cell-local robot id for the idx-th robot in scene.yaml (1-based).

    Both ``robots.yaml`` and ``cell.yaml`` name the same arm, so the id
    is derived once here rather than spelled out in each writer.
    """
    return f"robot{idx}"


def scene_yaw(entry: dict, what: str) -> float:
    """Yaw in radians from a scene.yaml ``orientation`` block.

    The manifest format is KUKA A/B/C in **degrees**, with A the rotation about
    +Z — see ``cells/cellsim-assets/fitted-robots/hall_a-cell.yaml``, where the
    values are 90, 164.4, 254.4. A non-zero B or C tilts the base out of plane,
    which neither ``robots.yaml`` (one yaw) nor the collision model represents,
    so it is dropped loudly rather than silently mixed into the yaw.
    """
    ori = entry.get("orientation") or {}

    def _f(key: str) -> float:
        try:
            return float(ori.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    b, c = _f("b"), _f("c")
    if abs(b) > 1e-9 or abs(c) > 1e-9:
        _warn(f"{what}: orientation b={_fmt(b)} c={_fmt(c)} (deg) are ignored — "
              f"only yaw (a) is modelled.")
    return math.radians(_f("a"))


def match_robot_model(robot_id: str, models: dict) -> Optional[str]:
    """
    Pick the robot-model key whose name is the longest prefix of robot_id.
    If only one model is available, always use it.
    """
    if len(models) == 1:
        return next(iter(models))
    best = None
    best_len = -1
    for key in models:
        if robot_id == key or robot_id.startswith(key):
            if len(key) > best_len:
                best, best_len = key, len(key)
    return best


# ─── scene asset resolution ───────────────────────────────────────────────────
#
# scene.yaml names its assets by path, not by value: `dressed_urdf`, each
# fixture's `urdf`, and `floor_plan.geometry_file` are written relative to the
# scene folder as it sits INSIDE the Isaac authoring workspace, so they read
# `../urdf-dressed-robots/...`. Once the export is copied out on its own those
# paths dangle. --isaacsim-root-cell-path supplies the workspace root the ".."
# was walking up to; `extra_roots` covers a bundle that was REARRANGED on the way
# out (the 2t cell keeps the same assets under robots/ and fixtures/), and the
# basename search is the last resort that makes such a bundle convert at all.

def _strip_updirs(rel: Path) -> Path:
    """Drop leading ".."/"." components from a relative path.

    The ".." in ``../urdf-dressed-robots/x.urdf`` is what walks from the scene
    folder up to the workspace root. Given that root directly, the same asset is
    at the remainder of the path.
    """
    parts = [p for p in rel.parts if p not in ("..", ".")]
    return Path(*parts) if parts else rel


def resolve_scene_asset(rel: str, scene_dir: Path,
                        isaac_root: Optional[Path],
                        extra_roots: tuple = (),
                        basename_search: tuple = ()) -> tuple:
    """Locate an asset scene.yaml refers to. Returns ``(path_or_None, tried)``.

    ``tried`` carries every candidate so a caller that fails can name them all —
    a bare "not found" against a path the user never typed is unactionable.

    The scene folder is tried FIRST: when the export still sits in its Isaac
    workspace the path as written is already correct, so the root option can
    never shadow an asset that is genuinely where scene.yaml says it is. Only
    after every composed path misses is ``basename_search`` walked for a file of
    the same name — that answer is a guess, so the caller is told out loud.
    """
    relp = Path(rel)
    cands = [relp] if relp.is_absolute() else [scene_dir / relp]
    if not relp.is_absolute():
        stripped = _strip_updirs(relp)
        for root in ([isaac_root] if isaac_root is not None else []) + list(extra_roots):
            if root is None:
                continue
            # The path as written, then with the leading ".." peeled off — the
            # root may be either the scene's parent or the workspace top.
            cands.append(root / relp)
            if stripped != relp:
                cands.append(root / stripped)

    tried = []
    for c in cands:
        try:
            r = c.resolve()
        except OSError:      # pragma: no cover - unresolvable mount
            r = c
        if str(r) not in tried:
            tried.append(str(r))
        if r.exists():
            return r, tried

    name = relp.name
    for root in basename_search:
        if root is None or not Path(root).is_dir():
            continue
        hits = sorted(Path(root).rglob(name))
        if hits:
            _warn(f"'{rel}' does not resolve as written; matched by FILENAME to "
                  f"{hits[0]}"
                  + (f" ({len(hits)} candidates, first taken)" if len(hits) > 1 else "")
                  + ". Check it is the asset you meant.")
            return hits[0].resolve(), tried
    return None, tried


def _cell_relative(path: Path, cell_dir: Path, what: str) -> str:
    """How ``cell.yaml`` should name ``path``: cell-relative, else absolute.

    A manifest path is resolved against the cell directory (see
    ``cell_layout._manifest_asset``), and the same manifest is read on the host
    and inside a container at ``/cells/<cell>``. So a path INSIDE the cell is
    recorded relative — it then means the same file in both — while anything
    outside can only be named absolutely, and that cell is not self-contained.
    """
    try:
        rel = path.resolve().relative_to(cell_dir.resolve())
    except (ValueError, OSError):
        _warn(f"{what} lives outside the cell ({path}); cell.yaml records it as "
              f"an absolute path, so this cell is not portable to a container "
              f"unless that path is mounted there too.")
        return str(path.resolve())
    return rel.as_posix()


def find_srdf_beside(urdf_path: Path) -> Optional[Path]:
    """The SRDF that goes with a dressed URDF, if the bundle shipped one.

    Same stem first, then any sibling ``.srdf``. It matters: a collision checker
    with no SRDF retains every self-collision pair, so adjacent links that touch
    by design report as colliding at every waypoint.
    """
    same = urdf_path.with_suffix(".srdf")
    if same.is_file():
        return same
    others = sorted(urdf_path.parent.glob("*.srdf"))
    return others[0] if others else None


# ─── output writers ───────────────────────────────────────────────────────────

CYCLONEDDS_XML = """\
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true"/>
      </Interfaces>
    </General>
    <Discovery>
      <!-- The whole cell shares one ROS_DOMAIN on one host: the agents,
           per-robot controllers, and the RViz web viewer (which alone starts
           ~8 nodes: a robot_state_publisher + ghost per robot, static TF, a
           joint relay and rviz2). CycloneDDS's default cap is ~10 participants
           per domain/host, so a multi-robot cell + RViz overruns it and nodes
           die with "failed to find a free participant index" (empty RViz).
           Lift the cap well above any realistic node count. -->
      <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
"""


def _list(vals) -> str:
    return "[" + ", ".join(_fmt(v) for v in vals) + "]"


def render_cell_yaml(cell_id: str, scene_name: str, scene_rel: Optional[str],
                     robots: list) -> str:
    """The cell manifest: which files describe each arm.

    ``robots: [{id, scene_id, …}]`` is the bridge between the two names the same
    arm has — the cell-local id the planner and the ROS graph use (``robot1``)
    and the id the scene export gave it (``kr240r2900__sticktool``). Readers
    outside this repo depend on that mapping and on ``- id:`` being the first key
    of each entry (the viewer's cell.yaml scan is a line-based stdlib parser), so
    this is written as text rather than dumped from a dict.

    ``scene_rel`` and every ``urdf``/``srdf`` are relative to the **cell
    directory**, which is not the directory this file is written into any more. The
    header says so explicitly: a reader who assumed manifest-relative would look
    for ``agentic_planner/robots/...`` and find nothing.
    """
    L = []
    L.append(f"# Cell manifest for '{cell_id}' — {scene_name}")
    L.append("# Generated by scripts/convert_isaac_scene.py. Overwritten on re-convert.")
    L.append("#")
    L.append("# This cell REFERENCES its assets where the scene bundle already put")
    L.append("# them; nothing was copied in. Paths below are relative to the CELL")
    L.append("# DIRECTORY — not to this file, which sits one level down in")
    L.append(f"# {cell_layout.AGENT_ROOT_NAME}/ — so they mean the same thing on the host")
    L.append("# and at /cells/<cell> in a container.")
    L.append("#")
    L.append("#   id        the cell-local robot id — ROS namespaces, robots.yaml,")
    L.append("#             telemetry and task programs all use this.")
    L.append("#   scene_id  what scene.yaml calls the same arm.")
    L.append("#   urdf      the dressed URDF. Without it a reader falls back to")
    L.append("#             guessing <env>/model/<model>.urdf: the trajectory provider")
    L.append("#             degrades to heuristic timing and the collision checker")
    L.append("#             loads no geometry at all.")
    L.append("#   srdf      the sibling SRDF. Without it every self-collision pair is")
    L.append("#             retained, so links that touch by design report as")
    L.append("#             colliding at every waypoint.")
    L.append(f"name: {scene_name}")
    L.append(f"cell_id: {cell_id}")
    if scene_rel:
        L.append(f"scene: {scene_rel}")
    L.append("robots:")
    for r in robots:
        L.append(f"  - id: {r['robot_id']}")
        L.append(f"    scene_id: {r['isaac_prim']}")
        L.append(f"    namespace: {r['namespace']}")
        L.append(f"    model: {r['robot_model']}")
        if r.get("urdf_rel"):
            L.append(f"    urdf: {r['urdf_rel']}")
        if r.get("srdf_rel"):
            L.append(f"    srdf: {r['srdf_rel']}")
        else:
            L.append("    # srdf: NONE FOUND beside the URDF — self-collision checking")
            L.append("    #       will retain every pair. Add one to silence the noise.")
    L.append("")
    return "\n".join(L)


def render_robots_yaml(
    cell_id: str,
    scene_name: str,
    robots: list,
    defaults: dict,
) -> str:
    """
    robots: ordered list of dicts:
      {namespace, robot_id, model_name, robot_model, base_position(x,y,z),
       yaw, lowers[], uppers[], vels[], home_pose_deg(optional),
       isaac_prim, pose_source, tip_link}
    """
    L = []
    L.append(f"# {scene_name} — cell '{cell_id}'")
    L.append("# Generated by scripts/convert_isaac_scene.py from a scene export.")
    L.append("#")
    L.append("# DERIVED automatically (safe to trust):")
    L.append("#   base_position          — from scene.yaml")
    L.append("#   base_orientation_yaw   — from the USD orient quaternion, or")
    L.append("#                            scene.yaml orientation.a (deg) without it")
    L.append("#   joint_limits_*         — from the arm's dressed URDF <limit>")
    L.append("#   joint_velocity_limits  — from the arm's URDF <limit velocity>")
    L.append("#   tip_link               — child link of the URDF's last movable joint")
    L.append("#")
    L.append("# The tool frame is DECLARED, not baked into a URDF: this cell references")
    L.append("# its dressed URDF where the bundle put it, so utils.arm_model attaches")
    L.append("# end_effector_link to tip_link at tcp_offset when it loads the model.")
    L.append("# Every consumer (IK, FK, the controller, collision) therefore targets")
    L.append("# the same point — get tcp_offset wrong and every weld is off by it.")
    L.append("#")
    L.append("# DEFAULTS below are marked 'REVIEW' — they are not derivable from the")
    L.append("# scene and should be checked against your robot/tooling.")
    L.append("#")
    L.append("# Structure: robots -> <namespace> -> <robot_id> -> { config }")
    L.append("robots:")

    # Group robots by namespace, preserving order.
    ns_order = []
    by_ns = {}
    for r in robots:
        if r["namespace"] not in by_ns:
            by_ns[r["namespace"]] = []
            ns_order.append(r["namespace"])
        by_ns[r["namespace"]].append(r)

    for ns in ns_order:
        L.append(f"  {ns}:                     # ROS namespace (ROBOT_NAMESPACE)")
        for r in by_ns[ns]:
            L.append(f"    {r['robot_id']}:")
            L.append(f"      name: {r['model_name']}")
            L.append(f"      robot_model: {r['robot_model']}         # ROBOT_MODEL env var")
            L.append(f"      # scene id: {r['isaac_prim']}")
            L.append(f"      base_position: {_list(r['base_position'])}   # world frame (m), from scene")
            L.append(f"      base_orientation_yaw: {_fmt(r['yaw'])}       # rad, from {r['pose_source']}")
            if r.get("home_pose_deg"):
                L.append(f"      home_pose_deg: {_list(r['home_pose_deg'])}   # from scene.yaml")
            L.append(f"      gripper: {defaults['gripper']}                 # REVIEW")
            L.append(f"      workspace_radius: {_fmt(defaults['workspace_radius'])}          # REVIEW (robot reach, m)")
            L.append(f"      planning_group: {defaults['planning_group']}")
            L.append(f"      end_effector_link: {defaults['end_effector_link']}     # tool frame added at load time")
            L.append(f"      tip_link: {r['tip_link']}   # its parent, from the URDF")
            L.append(f"      tcp_offset: {_list(defaults['tcp_offset'])}   # m, in the tip-link frame")
            if r.get("joint_names"):
                # Quoted: a URDF joint called `on`, `null` or `1` must stay a name.
                L.append("      joint_names: [" + ", ".join(json.dumps(str(n)) for n in r["joint_names"])
                         + "]   # movable joints, URDF order")
            guessed = r.get("guessed_limits") or []
            L.append(f"      joint_limits_lower: {_list(r['lowers'])}"
                     + (f"   # REVIEW: no URDF <limit> for {', '.join(guessed)} (±pi assumed)"
                        if guessed else ""))
            L.append(f"      joint_limits_upper: {_list(r['uppers'])}")
            guessed_v = r.get("guessed_velocity") or []
            L.append(f"      joint_velocity_limits:     {_list(r['vels'])}"
                     + (f"   # REVIEW: no URDF velocity for {', '.join(guessed_v)} (1.0 assumed)"
                        if guessed_v else ""))
            accel = _sized(defaults["accel"], len(r["vels"]))
            L.append(f"      joint_acceleration_limits: {_list(accel)}   # REVIEW (rad/s^2)")
            L.append(f"      rrt_j2_range: {_list(defaults['rrt_j2'])}   # REVIEW (safe sampling)")
            L.append(f"      rrt_j3_range: {_list(defaults['rrt_j3'])}   # REVIEW (safe sampling)")
            env = r.get("envelope")
            if env:
                L.append("      # The arm's ENVELOPE (agents/trajectory/tools/robot_kinematics.py):")
                L.append("      # spheres fitted to each link's <collision> hull, in the link's frame,")
                L.append(f"      # at robot_link_radius ({_fmt(env['radius'])} m): {env['spheres']} spheres, "
                         f"{100 * env['vertex_coverage']:.1f} % of the hull")
                L.append(f"      # vertices covered (worst {env['worst_gap_m']:.3f} m). Checked against the other")
                L.append("      # arm and against static bodies, on top of the skeleton spheres.")
                L.append("      link_collision:")
                L.append(f"        radius_m: {_fmt(env['radius'])}")
                L.append("        links:")
                for name, centres in env["links"].items():
                    L.append(f"          {name}:")
                    for c in centres:
                        L.append(f"            - {_list(c)}")
            pf = r.get("path_levers")
            tol = defaults.get("path_tolerance_m")
            if pf and tol:
                L.append("      # How far a TIMED path may leave the polyline the planner validated")
                L.append("      # (agents/trajectory/tools/time_parameterization.py rounds corners")
                L.append("      # within it). A joint deviation of d rad moves no point of the arm or")
                L.append("      # tool more than d * lever_sum_m (lever arms from this URDF), so")
                L.append("      # max_deviation_rad = tolerance_m / lever_sum_m. Absent = exact,")
                L.append("      # stopping at every corner.")
                L.append("      path_following:")
                L.append(f"        tolerance_m: {_fmt(tol)}")
                L.append(f"        lever_sum_m: {pf['lever_sum_m']:.4f}")
                L.append(f"        max_deviation_rad: {tol / pf['lever_sum_m']:.6f}")
            stop = defaults.get("stopping") or {}
            L.append("      # How far this arm travels before it stands still once a stop is")
            L.append("      # commanded: d = v * reaction_time_s + v^2 / (2 * brake_decel_m_s2), for")
            L.append("      # each sphere at its own speed v. Added to every clearance the planner")
            L.append("      # and the monitor check. REVIEW: take them from the manufacturer's")
            L.append("      # stopping-distance data; empty = speed is NOT accounted for.")
            L.append("      stopping:")
            for key in ("reaction_time_s", "brake_decel_m_s2"):
                val = stop.get(key)
                L.append(f"        {key}: {'null' if val is None else _fmt(val)}"
                         + ("            # REVIEW" if val is None else ""))
            to = {**(defaults.get("tool_orientation") or {}),
                  **((defaults.get("tool_orientation_by_robot") or {}).get(r["robot_id"]) or {})}
            L.append("      # The weld pose's free degrees of freedom (agents/trajectory/tools/")
            L.append("      # weld_orientation.py): a weld specifies the tool AXIS; with spin_free the")
            L.append("      # planner may turn the tool about it (a stick electrode, a torch) to keep")
            L.append("      # the arm out of itself, and with max_tilt_deg > 0 lean it that far off")
            L.append("      # the specified axis (inside axis_tolerance_deg, what validate_approach")
            L.append("      # gates). REVIEW: spin_free false for a tool whose roll matters (a gripper,")
            L.append("      # a gun with a side arm).")
            L.append("      tool_orientation:")
            L.append(f"        spin_free: {'true' if to.get('spin_free', True) else 'false'}   # REVIEW")
            L.append(f"        max_tilt_deg: {_fmt(to.get('max_tilt_deg', 4.5))}   # REVIEW")
            L.append(f"        axis_tolerance_deg: {_fmt(to.get('axis_tolerance_deg', 5.0))}")
            sp = r.get("self_pairs")
            if sp is not None:
                L.append("      # The arm against ITSELF (agents/trajectory/tools/self_collision.py):")
                L.append("      # its <collision> hulls, tool included, pairwise. DERIVED: the pairs")
                L.append("      # below touch by design and are not tested — " + "; ".join(sp["sources"]) + ".")
                L.append("      # required: planning refuses this robot when the check cannot run")
                L.append("      # (no meshes); false plans without it. margin_m: clearance to keep.")
                L.append("      self_collision:")
                L.append(f"        required: {'true' if defaults.get('self_collision_required', True) else 'false'}")
                L.append(f"        margin_m: {_fmt(defaults.get('self_collision_margin', 0.0))}")
                L.append("        disabled_pairs:")
                for a, b in sp["pairs"]:
                    L.append(f"          - [{a}, {b}]")
            fit = r.get("tool_fit")
            if fit:
                L.append("      # DERIVED from the tool's own meshes (the <collision> geometry on")
                L.append("      # tip_link and below, each placed with its full <origin xyz rpy>),")
                L.append("      # in the tip-link frame. Spheres are checked against the other arm at")
                L.append(f"      # robot_link_radius ({_fmt(fit['radius'])} m): {len(fit['centres'])} centres, all on")
                L.append(f"      # the metal (<= {fit['max_centre_to_metal_m']:.3f} m), covering "
                         f"{100 * fit['vertex_coverage']:.1f} % of {fit['vertices']} vertices")
                L.append(f"      # and {100 * fit['sample_coverage']:.1f} % of {fit['points']} surface points "
                         f"(worst {fit['worst_gap_m']:.3f} m from a centre).")
                L.append("      # Re-generated on every convert: change the meshes, not these numbers.")
                L.append("      tool_collision:")
                L.append(f"        mount_link: {r['tip_link']}")
                L.append("        spheres:")
                for c in fit["centres"]:
                    L.append(f"          - {_list(c)}")
                st = fit.get("static")
                if st:
                    L.append(f"        # Against the floor and the fixtures: a finer cover at the tool's")
                    L.append(f"        # own thickness ({_fmt(st['radius'])} m), {len(st['centres'])} centres on the metal,")
                    L.append(f"        # {100 * st['sample_coverage']:.1f} % of the surface within it "
                             f"(worst {st['worst_gap_m']:.3f} m).")
                    L.append(f"        static_radius_m: {_fmt(st['radius'])}")
                    L.append("        static_spheres:")
                    for c in st["centres"]:
                        L.append(f"          - {_list(c)}")
            else:
                L.append("      # REVIEW: tool collision geometry is a generic placeholder — the URDF")
                L.append("      # has no meshes on tip_link to derive it from.")
                L.append("      tool_collision:")
                L.append(f"        mount_link: {defaults['mount_link']}          # substring of the flange link name")
                L.append("        axis:")
                L.append(f"          direction: {_list(defaults['tool_axis_dir'])}")
                L.append(f"          start_m: {_fmt(defaults['tool_axis_start'])}")
                L.append(f"          end_m: {_fmt(defaults['tool_axis_end'])}")
                L.append(f"          step_m: {_fmt(defaults['tool_axis_step'])}")

    L.append("")
    L.append("collision:")
    L.append("  min_clearance: 0.05")
    L.append("  warning_distance: 0.20")
    L.append("  check_frequency: 100.0")
    L.append("")
    L.append("planning:")
    L.append("  default_planner: RRTConnect")
    L.append("  planning_time: 5.0")
    L.append("  # Speed & acceleration — the ONE place these numbers live. Speeds are fractions")
    L.append("  # of each robot's joint_velocity_limits; acceleration scales by the square.")
    L.append("  max_velocity_scaling: 0.7    # transit speed AND the ceiling on any speed asked for")
    L.append("  home_velocity_scale: 0.4     # planned return to HOME + lift-off out of the last seam")
    L.append("  acceleration_factor: 2.5     # accel = factor x vel, for a robot with no joint_acceleration_limits")
    L.append("  interlock_move_time_s: 2.5   # unplanned single-point move to an interlock pose")
    L.append("  home_return_time_s: 7.0      # unplanned fallback return to HOME")
    L.append(f"  home_liftoff_m: {_fmt(defaults.get('home_liftoff_m', 0.1))}"
             "         # on-axis lift-off out of the last seam on the way HOME")
    L.append(f"  min_home_liftoff_m: {_fmt(defaults.get('min_home_liftoff_m', 0.05))}"
             "     # shortest lift-off tried before the planned return is refused")
    L.append(f"  completion_margin_s: {_fmt(defaults.get('completion_margin_s', 1.0))}"
             "    # executor slack after every ExecuteTrajectory/MoveHome leaf (0 = none)")
    L.append(f"  thermal_spread: {'true' if defaults.get('thermal_spread', True) else 'false'}"
             "        # spread consecutive welds apart (--thermal-spread)")
    L.append("")
    L.append("welding:")
    L.append("  type: spot_weld")
    L.append("  weld_time: 0.3")
    L.append("")
    return "\n".join(L)


def _sized(values: list, n: int) -> list:
    """``values`` for an arm of ``n`` joints: cut, or padded with its last value."""
    vals = list(values)[:n]
    return vals + [vals[-1] if vals else 0.0] * (n - len(vals))


def _emit_box(L: list, box_id: str, description: str, c: tuple, h: tuple,
              placeholder: bool = False, tool_contact: bool = False) -> None:
    L.append(f"  - id: {box_id}")
    L.append("    type: box")
    L.append(f'    description: "{description}"')
    L.append(f"    center: [{_fmt(c[0])}, {_fmt(c[1])}, {_fmt(c[2])}]")
    L.append(f"    half_extents: [{_fmt(h[0])}, {_fmt(h[1])}, {_fmt(h[2])}]")
    L.append("    collision_margin: 0.05")
    if placeholder:
        # Machine-readable, not only a comment: a box that is not the fixture's
        # real geometry must be findable by a tool, not just by a reader.
        L.append("    placeholder: true")
    if tool_contact:
        # The surface a weld is made on, next to the weld: the tool's bodies may
        # touch it, the arm's may not (robot_kinematics._compile_static).
        L.append("    tool_contact: true")


def render_obstacles_yaml(cell_id: str, fixtures: list, obstacle_mode: str,
                          voxel_size: float,
                          tool_radius: Optional[dict] = None) -> str:
    """
    fixtures: ordered list of dicts, one of:
      {id, kind: 'voxel',       boxes: [{c,h}, ...], voxel, approx, nvox, source}
      {id, kind: 'aabb',        box: {c,h}, approx, source}
      {id, kind: 'placeholder', box: {c,h}}
    """
    L = []
    L.append(f"# Static collision obstacles for cell '{cell_id}'")
    L.append("# Generated by scripts/convert_isaac_scene.py. Overwritten on re-convert.")
    L.append("#")
    if obstacle_mode == "voxel":
        L.append(f"# Fixtures are modelled as a voxel-occupancy shell (voxel = {_fmt(voxel_size)} m)")
        L.append("# computed from the fixture's own meshes: only the space the surface")
        L.append("# occupies becomes boxes, so open cavities and openings stay clear and a")
        L.append("# robot can still reach into them. Boxes were merged greedily to keep")
        L.append("# the count low.")
    else:
        L.append("# Each fixture is ONE axis-aligned bounding box (a conservative solid block).")
    L.append("#")
    L.append("# The floor plan is NOT here: it is a 2D CAD overlay for the viewer, not")
    L.append("# collision geometry. Cell walls are not modelled as obstacles.")
    L.append("#")
    L.append("# tool_contact: true on the boxes of the surface a weld is made on, next to")
    L.append("# the weld: the TOOL may touch them, every arm link is still checked.")
    L.append("# Written by --weld-targets from the task programs' weld points. (A legacy")
    L.append("# is_workpiece: true is read the same way: tool only.)")
    L.append("#")
    L.append("# Schema:")
    L.append("#   box    — center [x,y,z] + half_extents [hx,hy,hz]")
    L.append("#   plane  — normal [nx,ny,nz] + offset d   (n.x + d >= 0 is free; a sphere")
    L.append("#            of radius r needs n.c + d >= r + collision_margin)")
    L.append("#   sphere — center [x,y,z] + radius")
    L.append("obstacles:")
    L.append("")
    L.append("  # Cell floor: the plane z = 0. A sphere is clear when n.c + offset >=")
    L.append("  # its radius + collision_margin — every moving arm and tool sphere keeps")
    L.append("  # collision_margin of air under it. Planes are obstacles like any other.")
    L.append("  - id: floor")
    L.append("    type: plane")
    L.append('    description: "Cell floor"')
    L.append("    normal: [0.0, 0.0, 1.0]")
    L.append("    offset: 0.0")
    L.append("    collision_margin: 0.05")

    for fx in fixtures:
        fid = fx["id"]
        kind = fx["kind"]
        src = fx.get("source", "scene")
        L.append("")

        if kind == "placeholder":
            b = fx["box"]
            L.append(f"  # {fid}: PLACEHOLDER box (no mesh geometry could be read).")
            L.append("  #   REVIEW: set the real size/position from your fixture geometry.")
            _emit_box(L, fid, f"{fid} — placeholder bounding box (REVIEW)", b["c"], b["h"],
                      placeholder=True)

        elif kind == "aabb":
            note = " (intermediate transform ignored; approximate)" if fx.get("approx") else ""
            b = fx["box"]
            L.append(f"  # {fid}: auto AABB from {src} mesh geometry{note}.")
            _emit_box(L, fid, f"{fid} — auto bounding box from scene (REVIEW: solid block)",
                      b["c"], b["h"])

        else:  # voxel
            note = " (intermediate transform ignored; approximate)" if fx.get("approx") else ""
            boxes = fx["boxes"]
            L.append(f"  # {fid}: voxel-occupancy shell — {len(boxes)} box(es) from "
                     f"{fx['nvox']} cells at {_fmt(fx['voxel'])} m, "
                     f"source: {src}{note}.")
            if fx.get("weld_note"):
                for line in fx["weld_note"]:
                    L.append(f"  #   {line}")
            width = len(str(len(boxes) - 1))
            for n, b in enumerate(boxes):
                what = ("weld contact surface" if b.get("tool_contact") else
                        "near a weld" if b.get("fine") else "voxel shell")
                _emit_box(L, f"{fid}_v{n:0{width}d}",
                          f"{fid} {what} {n}", b["c"], b["h"],
                          tool_contact=bool(b.get("tool_contact")))

    L.append("")
    L.append("# Clearance policy — how collision distance is computed")
    L.append("#")
    L.append("# A robot link is modelled as ONE sphere on its frame (the skeleton) plus the")
    L.append("# link_collision envelope robots.yaml fits to the link's <collision> hull; an")
    L.append("# obstacle box keeps its real size (it is NOT inflated). A link collides when:")
    L.append("#   distance(link_sphere_center, box_surface) < link_radius + collision_margin")
    L.append("#")
    L.append("# The two terms mean different things — they are NOT two safety buffers:")
    L.append("#   link_radius       the radius of each sphere. The skeleton sphere is a PROXY")
    L.append("#                     for the link, not an envelope of it (a long link reaches")
    L.append("#                     well past one sphere); the envelope is link_collision in")
    L.append("#                     robots.yaml, fitted to the hulls at this radius. A cell")
    L.append("#                     without collision meshes has the skeleton only (REVIEW).")
    L.append("#   collision_margin  the ONLY safety standoff: extra distance kept beyond")
    L.append("#                     physical contact (absorbs IK / tracking / spline error).")
    L.append("#                     Per-obstacle; default_collision_margin is the fallback.")
    L.append("#")
    L.append("# A radius is a property of the BODY, so there is one per body — not one per")
    L.append("# motion phase:")
    tr = (tool_radius or {}).get("value")
    tr_value = tr if tr is not None else DEFAULT_TOOL_LINK_RADIUS
    L.append(f"#   robot_link_radius  arm structure     ({_fmt(ROBOT_LINK_RADIUS)} + 0.05 m standoff)")
    L.append(f"#   tool_link_radius   the tool          ({_fmt(tr_value)} + 0.05 m standoff)")
    L.append("#")
    L.append("# tool_link_radius applies to the frames past the tool mount (the TCP frame")
    L.append("# utils.arm_model attaches at the cell's tcp_offset).")
    if tr is not None:
        L.append("# DERIVED from the tool meshes: the tool's largest distance from its axis")
        L.append("# (mount origin -> TCP) within 0.05 m of the TCP, rounded up to 5 mm,")
        L.append(f"# over {tool_radius.get('models')} — re-generated on every convert.")
    else:
        L.append("# REVIEW: the default — no tool geometry or no tcp_offset to derive it from.")
        L.append("# Modelling a thin torch at robot_link_radius claims ~3x its real thickness")
        L.append("# and can leave most welds without a collision-free goal pose.")
    L.append("#")
    L.append("# approach_link_radius is RETIRED (it relaxed the whole arm on approach=true")
    L.append("# while the runtime monitor did not) — still parsed, read by nothing.")
    L.append("# Both the planner and the monitor compute the same gate from these keys.")
    L.append("default_collision_margin: 0.05")
    L.append(f"robot_link_radius: {_fmt(ROBOT_LINK_RADIUS)}")
    L.append(f"tool_link_radius: {_fmt(tr_value)}")
    L.append("# The TOOL's own standoff from static bodies (floor, fixtures) — the arm keeps")
    L.append("# each obstacle's collision_margin. The tool is positioned to the weld by the")
    L.append("# TCP, so it carries less of the arm's tracking error; checked on the tool's")
    L.append("# static_spheres (robots.yaml), fitted to its meshes.")
    L.append(f"tool_collision_margin: {_fmt(DEFAULT_TOOL_COLLISION_MARGIN)}")
    L.append("approach_link_radius: 0.05   # RETIRED — parsed, read by nothing")
    L.append("")
    L.append("# Extra clearance a body must hold when moving at FULL commanded joint speed,")
    L.append("# tapering linearly to 0 at rest (reaction distance = e-stop latency x tip")
    L.append("# speed). 0.0 = no speed allowance. It is NOT a measured value: set it once the")
    L.append("# cell's controller e-stop latency is measured (latency_s x max tip speed).")
    L.append("speed_margin_full_speed_m: 0.0")
    L.append("")
    L.append("# How much slower the terminal working stroke runs than transit, as a fraction of")
    L.append("# velocity_scale. Overridable per operation via a motion request's")
    L.append('# "approach_velocity_scale".')
    L.append("approach_velocity_fraction: 0.4")
    L.append("")
    return "\n".join(L)


# ─── main conversion ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """The converter's command line (its defaults are what every converted cell
    declares, and the runtime falls back to the same values)."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scene-yaml", required=True,
                    help="Path to the scene manifest (scene.yaml). The only "
                         "required input: it names the robots, their poses and "
                         "the assets that describe them.")
    ap.add_argument("--scene", default=None,
                    help="Path to scene.usda, when the export has one. Used for "
                         "two things only: the authoritative base-yaw quaternion, "
                         "and fixture geometry as a fallback when a fixture's "
                         "urdf does not resolve. Optional — a bundle with no USD "
                         "converts from scene.yaml alone.")
    ap.add_argument("--isaacsim-root-cell-path", default=None, metavar="DIR",
                    help="Root of the Isaac authoring workspace that scene.yaml's "
                         "relative asset paths (dressed_urdf, each fixture's urdf, "
                         "floor_plan.geometry_file) were written against. Assets "
                         "are RESOLVED there and referenced in place — nothing is "
                         "copied into the cell.")
    ap.add_argument("--robot-model", default=None, metavar="DIR",
                    help="A directory holding the arms' model assets, searched as "
                         "an extra root when scene.yaml's dressed_urdf does not "
                         "resolve. Like --isaacsim-root-cell-path this only helps "
                         "FIND the URDF; the cell references it where it is.")
    ap.add_argument("--out-dir", required=True, metavar="DIR",
                    help="Target cell directory, e.g. cells/my_new_cell. Every "
                         "file the converter writes is derived from the scene, so "
                         "an existing cell is overwritten in place.")
    ap.add_argument("--cell-id", default=None,
                    help="Runtime cell id recorded in cell.yaml and robots.yaml. "
                         "Defaults to the basename of --out-dir. Override only if "
                         "the cell will be deployed under a different directory "
                         "name.")
    ap.add_argument("--namespace", default="arms",
                    help="Shared ROS namespace for all robots (default: arms)")
    ap.add_argument("--namespace-per-robot", action="store_true",
                    help="Give each robot its own namespace (== its robot_id)")
    ap.add_argument("--tcp-offset", default="0 0 0",
                    help='Tool tip offset in the flange (tip-link) frame, "x y z" '
                         '(m). Default "0 0 0" (tip at the flange) — set this to '
                         "your electrode/torch tip or every weld is targeted at "
                         "the flange.")
    ap.add_argument("--ee-link", default="welding_tcp",
                    help="Name of the tool-centre-point frame to target "
                         "(default: welding_tcp)")
    ap.add_argument("--workspace-radius", type=float, default=2.9,
                    help="Robot reach in metres for robots.yaml (default: 2.9)")
    ap.add_argument("--obstacle-mode", choices=["voxel", "aabb"], default="voxel",
                    help="How to model each fixture as collision geometry: "
                         "'voxel' (default) = occupancy-shell boxes that preserve "
                         "cavities/openings so robots can reach in; 'aabb' = one "
                         "conservative solid bounding box per fixture.")
    ap.add_argument("--voxel-size", type=float, default=0.05, metavar="M",
                    help="Voxel edge length in metres for --obstacle-mode voxel "
                         "(default: 0.05). Smaller = finer geometry but more boxes; "
                         "0.10 over-approximates a surface by up to a full voxel and "
                         "blocked reachable welds on welding_cell_2_small.")
    ap.add_argument("--only-obstacles", action="store_true",
                    help="Write ONLY agentic_planner/config/obstacles.yaml: rebuild the "
                         "fixtures' collision geometry into the existing file, keeping its "
                         "other obstacles and its clearance policy. cell.yaml, robots.yaml "
                         "and cyclonedds.xml are not touched.")
    ap.add_argument("--weld-targets", action="append", default=[], metavar="TASK.json",
                    help="A task program whose weld points shape the obstacles: each weld "
                         "is checked against the fixture mesh (on the surface, approach "
                         "corridor free of metal) and the shell is rebuilt at "
                         "--weld-voxel-size around it, the surface it is made on marked "
                         "tool_contact. Repeatable. With --only-obstacles the default is "
                         "every program in the cell's tasks/.")
    ap.add_argument("--weld-correction", choices=["apply", "report", "off"], default="apply",
                    help="What to do with a weld that is not on the surface or whose "
                         "approach runs into metal: 'apply' (default) corrects it in the "
                         "task program and its product definition (backups kept), "
                         "'report' only warns, 'off' ignores the welds altogether.")
    ap.add_argument("--weld-voxel-size", type=float, default=0.025, metavar="M",
                    help="Voxel edge near the welds (default 0.025; must divide --voxel-size). "
                         "0.01 is 4x the boxes: on welding_cell_2_small ~9900 against ~2600, and the "
                         "static check 3x slower")
    ap.add_argument("--weld-refine-radius", type=float, default=0.12, metavar="M",
                    help="How far around each weld the shell is rebuilt finely (default 0.12; "
                         "never less than the tool's reach at the weld)")
    ap.add_argument("--weld-standoff", type=float, default=0.10, metavar="M",
                    help="The approach stroke: how far out along the approach the TCP "
                         "starts its straight entry (default 0.10, the planner's "
                         "approach_offset_m)")
    ap.add_argument("--weld-corridor-radius", type=float, default=0.008, metavar="M",
                    help="Radius of the electrode's approach corridor that must be free of "
                         "metal (default 0.008, a 16 mm cap)")
    ap.add_argument("--max-weld-tilt", type=float, default=15.0, metavar="DEG",
                    help="Approach angle off the surface normal accepted as is (default 15)")
    ap.add_argument("--weld-max-stack", type=float, default=0.06, metavar="M",
                    help="How far out along the approach an outer sheet may cover a weld "
                         "point nothing reaches; the point is moved onto it (default 0.06)")
    ap.add_argument("--stop-reaction-time", type=float, default=None, metavar="S",
                    help="robots.yaml stopping.reaction_time_s: seconds from a stop "
                         "command to the brakes acting (monitor + controller). With "
                         "--stop-decel it makes every clearance speed-dependent.")
    ap.add_argument("--stop-decel", type=float, default=None, metavar="M_S2",
                    help="robots.yaml stopping.brake_decel_m_s2: braking deceleration "
                         "of a point on the arm, from the manufacturer's data.")
    ap.add_argument("--path-tolerance", type=float, default=DEFAULT_TOOL_COLLISION_MARGIN,
                    metavar="M",
                    help="robots.yaml path_following.tolerance_m: how far (m) a timed path "
                         "may leave the path the planner validated, at any point of the arm "
                         "or tool (default: the tool collision margin, "
                         f"{DEFAULT_TOOL_COLLISION_MARGIN} m). 0 = follow exactly.")
    ap.add_argument("--self-collision-margin", type=float, default=0.0, metavar="M",
                    help="Clearance the arm must keep from itself, between its "
                         "collision hulls (default: 0.0 — the hulls already envelope "
                         "the links).")
    ap.add_argument("--self-collision-optional", action="store_true",
                    help="Write self_collision.required: false — plan even when the "
                         "check cannot run (no meshes). Default: required.")
    ap.add_argument("--thermal-spread", choices=["on", "off"], default="on",
                    help="robots.yaml planning.thermal_spread: spread consecutive "
                         "welds of an arm apart to limit heat build-up (default: on). "
                         "'off' sequences each arm nearest-first.")
    ap.add_argument("--home-liftoff", type=float, default=0.1, metavar="M",
                    help="robots.yaml planning.home_liftoff_m: the on-axis lift-off out of "
                         "an arm's last seam before its return HOME (default: 0.1)")
    ap.add_argument("--min-home-liftoff", type=float, default=0.05, metavar="M",
                    help="robots.yaml planning.min_home_liftoff_m: the shortest lift-off "
                         "still tried before the planned return is refused rather than "
                         "dragged out of the seam (default: 0.05)")
    ap.add_argument("--completion-margin", type=float, default=1.0, metavar="S",
                    help="robots.yaml planning.completion_margin_s: executor slack after "
                         "every ExecuteTrajectory/MoveHome leaf (default: 1.0; 0 = none)")
    ap.add_argument("--tool-spin", choices=["free", "fixed"], default="free",
                    help="robots.yaml tool_orientation.spin_free: may the planner turn the "
                         "tool about its own axis to keep the arm out of itself (default: "
                         "free — a stick electrode / torch; 'fixed' for a tool whose roll "
                         "matters)")
    ap.add_argument("--tool-max-tilt", type=float, default=4.5, metavar="DEG",
                    help="robots.yaml tool_orientation.max_tilt_deg: how far the planner "
                         "may lean the tool off the weld's specified axis (default: 4.5, "
                         "inside the default 5 deg axis tolerance; 0 = never; must stay "
                         "under --tool-axis-tolerance)")
    ap.add_argument("--robot-tool-spin", action="append", default=[], metavar="ROBOT=free|fixed",
                    help="Per-arm override of --tool-spin, e.g. robot2=fixed for a gripper "
                         "arm next to a torch arm (repeatable)")
    ap.add_argument("--robot-tool-max-tilt", action="append", default=[], metavar="ROBOT=DEG",
                    help="Per-arm override of --tool-max-tilt (repeatable)")
    ap.add_argument("--tool-axis-tolerance", type=float, default=5.0, metavar="DEG",
                    help="robots.yaml tool_orientation.axis_tolerance_deg: the tool-axis "
                         "error a weld accepts, what validate_approach gates (default: 5)")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    for flag, val in (("--home-liftoff", args.home_liftoff),
                      ("--min-home-liftoff", args.min_home_liftoff),
                      ("--completion-margin", args.completion_margin),
                      ("--tool-max-tilt", args.tool_max_tilt),
                      ("--tool-axis-tolerance", args.tool_axis_tolerance)):
        if not math.isfinite(val):
            _fatal(f"{flag} must be a finite number, got {val}")
    for flag, val in (("--home-liftoff", args.home_liftoff),
                      ("--min-home-liftoff", args.min_home_liftoff),
                      ("--tool-axis-tolerance", args.tool_axis_tolerance)):
        if not val > 0:
            _fatal(f"{flag} must be > 0, got {val}")
    if args.completion_margin < 0:
        _fatal(f"--completion-margin must be >= 0, got {args.completion_margin}")
    if args.min_home_liftoff > args.home_liftoff:
        _warn(f"--min-home-liftoff {args.min_home_liftoff} exceeds --home-liftoff "
              f"{args.home_liftoff}; using {args.home_liftoff}")

    def _tilt_ok(tilt: float, what: str) -> None:
        # The runtime keeps TILT_RESERVE_DEG of the axis gate for the IK residual
        # (agents/trajectory/tools/weld_orientation.py); a tilt past that would be
        # silently clamped there, so it is refused here.
        if not (math.isfinite(tilt) and 0.0 <= tilt <= args.tool_axis_tolerance - TILT_RESERVE_DEG):
            _fatal(f"{what} ({tilt}) must be >= 0 and at most --tool-axis-tolerance "
                   f"({args.tool_axis_tolerance}) - {TILT_RESERVE_DEG}")
    _tilt_ok(args.tool_max_tilt, "--tool-max-tilt")
    tool_by_robot: dict = {}
    for item in args.robot_tool_spin:
        rid, _, mode = item.partition("=")
        if mode not in ("free", "fixed") or not rid:
            _fatal(f"--robot-tool-spin expects ROBOT=free|fixed, got {item!r}")
        tool_by_robot.setdefault(rid, {})["spin_free"] = mode == "free"
    for item in args.robot_tool_max_tilt:
        rid, _, val = item.partition("=")
        try:
            tilt = float(val)
        except ValueError:
            _fatal(f"--robot-tool-max-tilt expects ROBOT=DEG, got {item!r}")
        _tilt_ok(tilt, f"--robot-tool-max-tilt {rid}")
        tool_by_robot.setdefault(rid, {})["max_tilt_deg"] = tilt

    scene_yaml_path = Path(args.scene_yaml)
    if not scene_yaml_path.exists():
        _fatal(f"scene.yaml not found: {scene_yaml_path}")

    scene_path = Path(args.scene) if args.scene else None
    if scene_path is not None and not scene_path.exists():
        _fatal(f"scene.usda not found: {scene_path}")

    try:
        tcp_parts = [float(v) for v in args.tcp_offset.replace(",", " ").split()]
    except ValueError:
        _fatal(f"--tcp-offset must be three numbers, got: {args.tcp_offset!r}")
    if len(tcp_parts) != 3:
        _fatal(f"--tcp-offset must be three numbers, got: {args.tcp_offset!r}")
    tcp_offset = tuple(tcp_parts)
    if tcp_offset == (0.0, 0.0, 0.0) and not args.only_obstacles:
        _warn("--tcp-offset is 0 0 0: the tool centre point defaults to the flange. "
              "Weld targeting will be at the flange, not the electrode tip. Pass "
              "--tcp-offset \"x y z\" for accurate targeting.")

    # ── Load inputs ─────────────────────────────────────────────────────────
    scene = load_scene_yaml(scene_yaml_path, need_robots=not args.only_obstacles)
    usd_text = scene_path.read_text() if scene_path is not None else ""
    if not usd_text:
        _info("no --scene given: converting from scene.yaml alone (base yaw from "
              "orientation.a, fixture geometry from each fixture's urdf).")

    scene_dir = scene_yaml_path.resolve().parent
    isaac_root = None
    if args.isaacsim_root_cell_path:
        isaac_root = Path(args.isaacsim_root_cell_path).expanduser()
        if not isaac_root.is_dir():
            _fatal(f"--isaacsim-root-cell-path is not a directory: {isaac_root}")
        isaac_root = isaac_root.resolve()
        _info(f"Isaac workspace root: {isaac_root}")

    # ── Target layout ───────────────────────────────────────────────────────
    # The destination is an explicit directory the caller chooses. cell_id
    # defaults to that directory's name.
    cell_dir = Path(args.out_dir)
    cell_id = args.cell_id or cell_dir.name
    cell_base = cell_dir.parent
    scene_name = scene.get("name") or cell_id

    # A bundle handed over as a cell keeps its assets under the cell directory
    # itself (robots/, fixtures/, floorplans/) rather than in the Isaac tree
    # scene.yaml's paths were written against, so the cell is a search root too —
    # and the last-resort filename match is scoped to it.
    model_root = None
    if args.robot_model:
        model_root = Path(args.robot_model).expanduser()
        if not model_root.is_dir():
            _fatal(f"--robot-model is not a directory: {model_root}")
        model_root = model_root.resolve()
        _info(f"extra asset root: {model_root}")

    extra_roots = tuple(r for r in (
        cell_dir.resolve() if cell_dir.exists() else None, model_root) if r)
    search_roots = tuple(r for r in (
        cell_dir if cell_dir.exists() else None, model_root) if r)

    def _resolve(rel: str) -> tuple:
        return resolve_scene_asset(rel, scene_dir, isaac_root,
                                   extra_roots=extra_roots,
                                   basename_search=search_roots)

    unresolved = []

    def _fixtures(weld_ctx) -> tuple:
        """Every fixture's collision geometry: ``(fixtures_out, total_boxes)``."""
        fixtures_out = []
        total_boxes = 0
        for fx in scene.get("fixtures", []) or []:
            fid = fx.get("id")
            if not fid:
                continue
            pos = fx.get("position", {}) or {}
            fpos = (float(pos.get("x", 0.0)), float(pos.get("y", 0.0)),
                    float(pos.get("z", 0.0)))
            fyaw = scene_yaw(fx, f"fixture '{fid}'")
            world_pose = (_yaw_matrix(fyaw), fpos)

            # Geometry source, in order of fidelity: the fixture's own URDF (full-res
            # CAD, one <origin> per mesh entry), then the mesh data embedded in the
            # USD, then nothing.
            fixture_urdf = None
            rel = fx.get("urdf")
            if rel:
                found, tried = _resolve(rel)
                if found is None:
                    unresolved.append((f"fixture '{fid}' urdf", rel, tried))
                else:
                    fixture_urdf = found

            def _placeholder():
                _warn(f"fixture '{fid}': no mesh geometry could be read — emitting a "
                      f"placeholder box (REVIEW).")
                return {"id": fid, "kind": "placeholder",
                        "box": {"c": (fpos[0], fpos[1], fpos[2] + 0.5),
                                "h": (0.5, 0.5, 0.5)}}

            if args.obstacle_mode == "voxel":
                vox = None
                source = None
                if fixture_urdf is not None:
                    _info(f"fixture '{fid}': voxelising {fixture_urdf} at "
                          f"{_fmt(args.voxel_size)} m (full-resolution CAD — this "
                          f"takes a while)")
                    source = f"{fixture_urdf.name} meshes"
                    if weld_ctx is not None and weld_ctx.ops:
                        tris = list(urdf_geometry_triangles(fixture_urdf, world_pose))
                        welds = weld_ctx.correct(fid, tris)
                        if welds:
                            wv = weld_aware_voxels(tris, args.voxel_size, args.weld_voxel_size,
                                                   welds, args.weld_refine_radius)
                            boxes = weld_aware_boxes(wv, args.voxel_size, args.weld_voxel_size)
                            nvox = (len(wv["coarse"]) + len(wv["fine"]) + len(wv["contact"])
                                + len(wv["coarse_contact"]))
                            n_contact = sum(1 for b in boxes if b.get("tool_contact"))
                            total_boxes += len(boxes)
                            fixtures_out.append({
                                "id": fid, "kind": "voxel", "boxes": boxes,
                                "voxel": args.voxel_size, "approx": False, "nvox": nvox,
                                "source": source, "weld_note": [
                                    f"rebuilt at {_fmt(args.weld_voxel_size)} m within "
                                    f"{_fmt(args.weld_refine_radius)} m of {len(welds)} weld(s) "
                                    f"({len(wv['fine']) + len(wv['contact'])} fine cells);",
                                    f"{n_contact} box(es) are the weld contact surface "
                                    f"(tool_contact: the tool may touch them, the arm may not)."]})
                            _info(f"fixture '{fid}': {len(boxes)} box(es) — "
                                  f"{len(wv['coarse'])} cells at {_fmt(args.voxel_size)} m, "
                                  f"{len(wv['fine'])} + {len(wv['contact'])} (contact) at "
                                  f"{_fmt(args.weld_voxel_size)} m around {len(welds)} weld(s)")
                            continue
                        vox = voxelise_triangles(iter(tris), args.voxel_size)
                    else:
                        vox = voxelise_triangles(
                            urdf_geometry_triangles(fixture_urdf, world_pose),
                            args.voxel_size)
                if vox is None and usd_text:
                    vox = fixture_voxels_from_usd(usd_text, fid, args.voxel_size)
                    source = "scene.usda mesh points"
                if vox is None:
                    fixtures_out.append(_placeholder())
                    total_boxes += 1
                    continue
                merged = _merge_voxels(vox["voxels"])
                boxes = _voxel_boxes_world(merged, args.voxel_size)
                total_boxes += len(boxes)
                fixtures_out.append({"id": fid, "kind": "voxel", "boxes": boxes,
                                     "voxel": args.voxel_size, "approx": vox["approx"],
                                     "nvox": len(vox["voxels"]), "source": source})
                _info(f"fixture '{fid}': {len(boxes)} voxel box(es) from "
                      f"{len(vox['voxels'])} cells (voxel={_fmt(args.voxel_size)} m, "
                      f"source: {source})")
            else:  # aabb
                aabb = None
                source = None
                if fixture_urdf is not None:
                    aabb = aabb_of_triangles(
                        urdf_geometry_triangles(fixture_urdf, world_pose))
                    source = f"{fixture_urdf.name} meshes"
                if aabb is None and usd_text:
                    aabb = fixture_aabb_from_usd(usd_text, fid)
                    source = "scene.usda mesh points"
                if aabb is None:
                    fixtures_out.append(_placeholder())
                    total_boxes += 1
                    continue
                mn, mx = aabb["min"], aabb["max"]
                box = {
                    "c": ((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0),
                    "h": (max((mx[0] - mn[0]) / 2.0, 1e-4),
                          max((mx[1] - mn[1]) / 2.0, 1e-4),
                          max((mx[2] - mn[2]) / 2.0, 1e-4)),
                }
                total_boxes += 1
                fixtures_out.append({"id": fid, "kind": "aabb", "box": box,
                                     "approx": aabb.get("approx", False),
                                     "source": source})
                _info(f"fixture '{fid}': AABB from {source}")
        return fixtures_out, total_boxes

    # ── Weld targets: the programs' weld points against the fixture mesh ─────
    weld_ctx = None
    weld_paths = [Path(p) for p in args.weld_targets]
    if not weld_paths and args.only_obstacles:
        tdir = cell_layout.tasks_dir(cell_id, cell_base)
        weld_paths = sorted(Path(tdir).glob("*.json")) if Path(tdir).is_dir() else []
    if weld_paths and args.obstacle_mode == "voxel" and args.weld_correction != "off":
        weld_ctx = WeldTargets(weld_paths, cell_dir, args)

    if args.only_obstacles:
        # Only obstacles.yaml: the cell's manifest and robots.yaml — and their
        # hand-tuned REVIEW fields — are left exactly as they are.
        obstacles_path = cell_layout.obstacles_yaml(cell_id, cell_base)
        existing = obstacles_path.read_text() if obstacles_path.is_file() else ""
        policy = (yaml.safe_load(existing) or {}) if existing else {}
        if weld_ctx is not None:
            weld_ctx.set_reach(policy)
        fixtures_out, total_boxes = _fixtures(weld_ctx)
        if weld_ctx is not None:
            weld_ctx.finish()
        fresh = render_obstacles_yaml(cell_id, fixtures_out, args.obstacle_mode,
                                      args.voxel_size)
        obstacles_path.parent.mkdir(parents=True, exist_ok=True)
        if existing:
            _backup(obstacles_path)
        obstacles_path.write_text(splice_obstacles(existing, fresh, fixtures_out)
                                  if existing else fresh)
        _info(f"wrote {obstacles_path}  ({len(fixtures_out)} fixture(s), {total_boxes} "
              f"box(es); every other entry and the clearance policy kept)")
        for what, rel, tried in unresolved:
            _warn(f"{what} does not resolve: {rel}\n    tried: " + "\n           ".join(tried))
        return

    # ── Resolve each arm's dressed URDF (referenced, never copied) ───────────
    models = {}  # model_key -> {urdf_path, srdf_path, name, parsed}

    for sr in scene["robots"]:
        rel = sr.get("dressed_urdf")
        rid_scene = sr.get("robot_id", "?")
        if not rel:
            _warn(f"{rid_scene}: scene.yaml declares no dressed_urdf.")
            continue
        found, tried = _resolve(rel)
        if found is None:
            unresolved.append((f"robot '{rid_scene}' dressed_urdf", rel, tried))
            continue
        model_key = found.stem
        if model_key in models:
            continue          # two arms may share one dressed model
        try:
            urdf_text = found.read_text()
        except OSError as exc:
            _fatal(f"cannot read {found}: {exc}")
        parsed = parse_urdf_model(urdf_text)
        if not parsed["movable"]:
            _fatal(f"robot model has no movable joints: {found}")
        srdf = find_srdf_beside(found)
        models[model_key] = {
            "urdf_path": found,
            "srdf_path": srdf,
            "name": found.stem,
            "parsed": parsed,
        }
        _info(f"robot model '{model_key}': {found}")
        _info(f"    {len(parsed['movable'])} movable joints, "
              f"tip_link={parsed['tip_link']}, "
              f"srdf={srdf.name if srdf else 'NONE'}")
        if srdf is None:
            _warn(f"no .srdf beside {found.name}: a collision checker will retain "
                  f"every self-collision pair, so links that touch by design "
                  f"report as colliding.")

    if not models:
        _fatal(
            "no robot model resolved from scene.yaml.\n"
            "  Each robot's dressed_urdf is written relative to the scene folder "
            "inside the\n"
            "  authoring workspace; pass --isaacsim-root-cell-path DIR so those "
            "paths resolve,\n"
            "  or place the assets under the cell directory.")

    # ── Create the layout ───────────────────────────────────────────────────
    # The agent workspace: created up front, whole, so a freshly converted cell is
    # immediately writable by the agents. Creating it here is also what puts this
    # cell on the new layout, which is why every derived path below — config/
    # included — resolves into it (see utils.cell_layout.is_migrated).
    #
    # NOTHING is written at the cell root — not even cell.yaml. The root is the
    # scene bundle as its author shipped it; every file this script generates is
    # derived from the scene, so all of it goes under agentic_planner/.
    for d in cell_layout.create_agent_layout(cell_id, cell_base):
        _info(f"agent workspace: {d}")

    robots_yaml_path = cell_layout.agent_config_file(
        cell_id, "robots.yaml", cell_base)

    # A cell converted before config/ and cell.yaml moved into the workspace still
    # has a root copy. Both would then be on disk and only the agent one is read —
    # dead config that reads as live. Say so; deleting someone else's file is not
    # this script's call.
    stale_candidates = [cell_layout.root_config_dir(cell_id, cell_base) / name
                        for name in ("robots.yaml", "cyclonedds.xml")]
    stale_candidates.append(
        cell_layout.cell_dir(cell_id, cell_base) / cell_layout.CELL_MANIFEST)
    for stale in stale_candidates:
        if stale.is_file():
            _warn(f"{stale} predates the move into "
                  f"{cell_layout.AGENT_ROOT_NAME}/. The copy written below is the "
                  f"one that will be read — delete the old one so the tree cannot "
                  f"be misread.")

    # ── Build robot entries ─────────────────────────────────────────────────
    robot_entries = []

    for idx, sr in enumerate(scene["robots"], start=1):
        robot_id_scene = sr.get("robot_id", f"robot_{idx}")
        model_key = match_robot_model(robot_id_scene, models)
        if model_key is None:
            _fatal(f"no resolved robot model matches scene robot "
                   f"'{robot_id_scene}'. Available: {list(models)}")
        model = models[model_key]
        parsed = model["parsed"]

        # Namespace / robot-id assignment.
        rid = robot_id_for_index(idx)
        ns = rid if args.namespace_per_robot else args.namespace

        # Pose: position from scene.yaml, yaw from the USD when we have it.
        pos = sr.get("position", {}) or {}
        base_pos = (
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)),
        )
        usd_pose = parse_robot_pose_from_usd(usd_text, robot_id_scene) if usd_text else None
        if usd_pose is not None:
            yaw = usd_pose["yaw"]
            pose_source = "USD xformOp:orient"
            # Cross-check position against the USD translate.
            ut = usd_pose["translate"]
            if any(abs(a - b) > 1e-3 for a, b in zip(base_pos, ut)):
                _warn(f"{robot_id_scene}: scene.yaml position {base_pos} differs "
                      f"from USD translate {ut}; using scene.yaml position.")
        else:
            yaw = scene_yaw(sr, robot_id_scene)
            pose_source = "scene.yaml orientation.a (deg)"
            if usd_text:
                _warn(f"{robot_id_scene}: not found in the USD; yaw taken from "
                      f"scene.yaml orientation.a.")

        # Movable joint limits (in order).
        lowers, uppers, vels = [], [], []
        guessed_limits, guessed_velocity = [], []
        for j in parsed["movable"]:
            lowers.append(j["lower"] if j["lower"] is not None else -math.pi)
            uppers.append(j["upper"] if j["upper"] is not None else math.pi)
            vels.append(j["velocity"] if j["velocity"] is not None else 1.0)
            if j["lower"] is None or j["upper"] is None:
                guessed_limits.append(j["name"])
            if j["velocity"] is None:
                guessed_velocity.append(j["name"])
        if guessed_limits or guessed_velocity:
            _warn(f"{robot_id_scene}: no URDF limit for "
                  f"{', '.join(guessed_limits + guessed_velocity)} — assumed (REVIEW)")

        # robot_model: portion before "__", else the whole model key.
        robot_model = model_key.split("__", 1)[0] if "__" in model_key else model_key

        # The tool model, derived from the arm's own meshes (once per model).
        if "tool_fit" not in model:
            model["tool_fit"] = (fit_tool_model(model["urdf_path"], parsed["tip_link"],
                                                tcp_offset, ROBOT_LINK_RADIUS)
                                 if parsed["tip_link"] else None)
            tf = model["tool_fit"]
            if tf:
                _info(f"tool model '{model_key}': {len(tf['centres'])} spheres from "
                      f"{tf['triangles']} triangles, {100 * tf['sample_coverage']:.1f} % "
                      f"of the surface within {_fmt(ROBOT_LINK_RADIUS)} m"
                      + (f", TCP radius {_fmt(tf['tool_link_radius'])} m"
                         if tf.get("tool_link_radius") is not None else ""))
            else:
                _warn(f"tool model '{model_key}': no meshes on {parsed['tip_link']} — "
                      f"writing the generic tool_collision placeholder (REVIEW).")

        if "envelope" not in model:
            model["envelope"] = (fit_link_envelope(model["urdf_path"], parsed["tip_link"],
                                                   ROBOT_LINK_RADIUS)
                                 if parsed["tip_link"] else None)
            env = model["envelope"]
            if env:
                _info(f"link envelope '{model_key}': {env['spheres']} spheres over "
                      f"{len(env['links'])} links, {100 * env['vertex_coverage']:.1f} % of "
                      f"the hull vertices within {_fmt(ROBOT_LINK_RADIUS)} m")
            else:
                _warn(f"link envelope '{model_key}': no collision meshes — the arm is "
                      f"modelled on its skeleton only (REVIEW)")
        if "self_pairs" not in model:
            model["self_pairs"] = self_collision_pairs(model["urdf_path"], model["srdf_path"])
        if "path_levers" not in model:
            model["path_levers"] = (path_lever_arms(model["urdf_path"], parsed["tip_link"],
                                                    tcp_offset) if parsed["tip_link"] else None)
            if model["path_levers"] and args.path_tolerance > 0:
                _info(f"path following '{model_key}': lever sum "
                      f"{model['path_levers']['lever_sum_m']:.3f} m -> corners rounded within "
                      f"{args.path_tolerance / model['path_levers']['lever_sum_m']:.5f} rad")
            else:
                _warn(f"path following '{model_key}': no tolerance or no measurable arm "
                      f"geometry — timed paths follow the polyline exactly, stopping at "
                      f"every corner (REVIEW)")

        robot_entries.append({
            "path_levers": model["path_levers"],
            "envelope": model["envelope"],
            "self_pairs": model["self_pairs"],
            "tool_fit": model["tool_fit"],
            "namespace": ns,
            "robot_id": rid,
            "isaac_prim": robot_id_scene,
            "model_name": model["name"],
            "robot_model": robot_model,
            "base_position": base_pos,
            "yaw": yaw,
            "pose_source": pose_source,
            "lowers": lowers,
            "uppers": uppers,
            "vels": vels,
            "joint_names": [j["name"] for j in parsed["movable"]],
            "guessed_limits": guessed_limits,
            "guessed_velocity": guessed_velocity,
            "home_pose_deg": sr.get("home_pose_deg"),
            "tip_link": parsed["tip_link"] or "",
            "urdf_rel": _cell_relative(model["urdf_path"], cell_dir,
                                       f"{rid} dressed URDF"),
            "srdf_rel": (_cell_relative(model["srdf_path"], cell_dir,
                                        f"{rid} SRDF")
                         if model["srdf_path"] else None),
        })
        _info(f"{rid}: references {robot_entries[-1]['urdf_rel']}  "
              f"(scene_id={robot_id_scene}, base={_list(base_pos)}, "
              f"yaw={_fmt(yaw)})")

    # ── cell.yaml — the manifest that replaces copying ──────────────────────
    scene_rel = None
    try:
        scene_rel = scene_yaml_path.resolve().relative_to(cell_dir.resolve()).as_posix()
    except (ValueError, OSError):
        _warn(f"scene.yaml is outside the cell ({scene_yaml_path}); cell.yaml "
              f"records no 'scene:' path. A viewer that places robot bases from "
              f"the scene will need it passed explicitly.")
    # The manifest is derived from the scene, so it is written into the workspace —
    # agent_cell_yaml, not cell_yaml, so an existing root copy cannot steer the
    # write back out to the root. The paths INSIDE it stay relative to the cell
    # directory (see _cell_relative above and cell_layout._manifest_asset), which is
    # why moving the file does not change what any entry resolves to.
    cell_yaml_path = cell_layout.agent_cell_yaml(cell_id, cell_base)
    cell_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    cell_yaml_path.write_text(
        render_cell_yaml(cell_id, scene_name, scene_rel, robot_entries))
    _info(f"wrote {cell_yaml_path}")

    # ── robots.yaml ─────────────────────────────────────────────────────────
    defaults = {
        "gripper": "welding_gun",
        "workspace_radius": args.workspace_radius,
        "planning_group": "kuka_arm",
        "end_effector_link": args.ee_link,
        "tcp_offset": tcp_offset,
        "accel": [3.0, 3.0, 3.0, 6.0, 6.0, 8.0],
        "rrt_j2": [-1.5, 0.0],
        "rrt_j3": [0.3, 2.5],
        "mount_link": "link_6",
        "tool_axis_dir": [-1.0, 0.0, 0.0],
        "tool_axis_start": 0.15,
        "tool_axis_end": 0.90,
        "tool_axis_step": 0.15,
        "thermal_spread": args.thermal_spread == "on",
        "self_collision_margin": args.self_collision_margin,
        "path_tolerance_m": args.path_tolerance,
        "stopping": {"reaction_time_s": args.stop_reaction_time,
                     "brake_decel_m_s2": args.stop_decel},
        "self_collision_required": not args.self_collision_optional,
        "home_liftoff_m": args.home_liftoff,
        "min_home_liftoff_m": min(args.min_home_liftoff, args.home_liftoff),
        "completion_margin_s": args.completion_margin,
        "tool_orientation": {"spin_free": args.tool_spin == "free",
                             "max_tilt_deg": args.tool_max_tilt,
                             "axis_tolerance_deg": args.tool_axis_tolerance},
        "tool_orientation_by_robot": tool_by_robot,
    }
    robots_yaml_path.write_text(
        render_robots_yaml(cell_id, scene_name, robot_entries, defaults)
    )
    _info(f"wrote {robots_yaml_path}")

    # ── obstacles.yaml (floor + fixture collision geometry) ─────────────────
    if weld_ctx is not None:
        fitted = [md["tool_fit"]["tool_link_radius"] for md in models.values()
                  if md.get("tool_fit") and md["tool_fit"].get("tool_link_radius") is not None]
        weld_ctx.set_reach({"tool_link_radius": max(fitted) if fitted else DEFAULT_TOOL_LINK_RADIUS,
                            "default_collision_margin": 0.05})
    fixtures_out, total_boxes = _fixtures(weld_ctx)
    if weld_ctx is not None:
        weld_ctx.finish()

    # Obstacles are DERIVED — computed by this run and re-tuned by whoever plans
    # against the cell — so they belong to the agent workspace, not to the cell's
    # shared inputs.
    obstacles_path = cell_layout.obstacles_yaml(cell_id, cell_base)
    # One tool_link_radius per cell: the largest any of its tools needs.
    measured = {m: md["tool_fit"]["tool_link_radius"] for m, md in models.items()
                if md.get("tool_fit") and md["tool_fit"].get("tool_link_radius") is not None}
    tool_radius = ({"value": max(measured.values()), "models": ", ".join(sorted(measured))}
                   if measured else None)
    obstacles_path.write_text(
        render_obstacles_yaml(cell_id, fixtures_out, args.obstacle_mode, args.voxel_size,
                              tool_radius)
    )
    _info(f"wrote {obstacles_path}  ({len(fixtures_out)} fixture(s), "
          f"{total_boxes} box(es) + floor, mode={args.obstacle_mode})")

    # ── cyclonedds.xml ──────────────────────────────────────────────────────
    # Generated, so it lives in the workspace with the rest. The CycloneDDS C
    # runtime reads it via CYCLONEDDS_URI; the compose generator computes that URI
    # from the same probe (cell_layout.config_uri), so the env var follows the file.
    dds_path = cell_layout.agent_config_file(cell_id, "cyclonedds.xml", cell_base)
    dds_path.write_text(CYCLONEDDS_XML)
    _info(f"wrote {dds_path}")

    # ── Unresolved assets ───────────────────────────────────────────────────
    for what, rel, tried in unresolved:
        _warn(f"{what} does not resolve: {rel}\n    tried: "
              + "\n           ".join(tried)
              + ("\n    Pass --isaacsim-root-cell-path <workspace root> so the "
                 "relative paths resolve." if isaac_root is None else ""))

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    _info(f"Done. Cell '{cell_id}' written at {cell_dir}")
    _info(f"  robots:    {len(robot_entries)}  (namespace(s): "
          f"{sorted({r['namespace'] for r in robot_entries})})")
    _info(f"  fixtures:  {len(fixtures_out)} fixture(s), {total_boxes} "
          f"obstacle box(es) + floor")
    _info(f"  assets:    referenced in place, nothing copied"
          + (f"  ({len(unresolved)} unresolved)" if unresolved else ""))
    _info("  NOT generated: tasks/*.json (author the task separately).")
    print()
    _info("Next steps:")
    _info(f"  1. Review REVIEW-marked fields in {robots_yaml_path}")
    _info(f"     (tool_collision, workspace_radius, gripper)"
          + (" and tcp_offset — it is 0 0 0, so the tool frame is the flange."
             if tcp_offset == (0.0, 0.0, 0.0) else "."))
    _info(f"  2. Refine fixture obstacle boxes in {obstacles_path} if robots must "
          f"reach into/onto them.")
    _info(f"  3. Add a task file under {cell_layout.tasks_dir(cell_id, cell_base)}/.")
    _info(f"  4. Launch:  make sim SIM={cell_id}")


if __name__ == "__main__":
    main()
