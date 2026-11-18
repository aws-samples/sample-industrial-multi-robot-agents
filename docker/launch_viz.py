#!/usr/bin/env python3
"""
RViz visualization launcher for a multi-robot cell.

Runs inside the rviz-web container. For every robot defined in the cell's
robots.yaml it:

  1. Loads the robot URDF, rewrites relative mesh paths to absolute file://
     URIs and prefixes every link/joint name with "<ns>_" so multiple robots
     share the global /tf tree without frame collisions.
  2. Starts one robot_state_publisher per robot (namespace /viz/<ns>) which
     publishes /viz/<ns>/robot_description (transient-local, latched) and the
     link transforms on /tf.
  3. Publishes a static transform  world -> <ns>_World  at the robot's
     base_position (+ yaw) so each robot sits at its real cell location.
  4. Relays the robot's real joint_states topic into /viz/<ns>/joint_states,
     remapping BY POSITION onto the URDF joints (the runtime joint names do
     not match the URDF names). If no joint_states have arrived yet it
     publishes a HOME pose so the robot is always visible.
  5. Generates an RViz config (fixed frame = world, Grid, TF and one
     RobotModel per robot) and launches rviz2.

Env:
  CELL_ID         cell directory under CELL_BASE_PATH  (default welding_cell_2_small)
  CELL_BASE_PATH  mount point of the cells tree        (default /cells)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from utils import cell_layout

# Repeated literal, not an import: this runs in the RViz image, which has no
# agents package. agents/cell_defaults.py owns the value; tests/test_default_cell.py
# keeps this copy in step with it.
CELL_ID   = os.environ.get("CELL_ID", "welding_cell_2_small")
CELL_BASE = os.environ.get("CELL_BASE_PATH", "/cells").rstrip("/")
CELL_DIR  = Path(f"{CELL_BASE}/{CELL_ID}")

# Idle pose shown until real joint_states arrive. Read per robot from the cell's
# own robots.yaml (see load_robot_spec) so the viewer opens on the pose the cell
# declares instead of drawing the arms somewhere they never park. This is the
# last resort for a cell that declares nothing.
HOME_JOINTS = [0.0, -1.57, 2.2, 0.0, -0.8, 0.0]

_ACTUATED = {"revolute", "continuous", "prismatic"}

# ─── Step-through preview ("ghost" arm) ─────────────────────────────────────────
# RViz can only ever show what the controllers actually did: the arms follow
# /joint_states, and nothing in this project publishes a planned path (there is no
# DisplayTrajectory anywhere). So stepping through one node of a plan to see what
# it contributes had no way to be shown WITHOUT commanding the cell to move.
#
# The ghost is that way. Every robot gets a second robot_state_publisher over the
# same URDF under a distinct name prefix, so its links and TF frames cannot collide
# with the real arm's, and a second RobotModel display draws it translucent. It is
# driven from JSON on /<cell>/node_preview and NEVER from the controllers — a
# preview moves this copy and only this copy. The real arm does not move, no
# trajectory is dispatched, and no cell state changes.
GHOST_SUFFIX      = "_ghost"
GHOST_ALPHA       = 0.4
PREVIEW_TOPIC     = f"/{CELL_ID}/node_preview"
PREVIEW_RATE_HZ   = 20.0
# While no preview is running the ghost is parked ON the real arm rather than left
# where the last preview ended: a stale ghost frozen mid-air reads as a robot that
# is somewhere it is not. Parked, it is the real silhouette, slightly tinted.
PREVIEW_IDLE_MIRRORS_REAL = True


def ghost_id(robot_id: str) -> str:
    return f"{robot_id}{GHOST_SUFFIX}"


def log(msg: str) -> None:
    print(f"[viz] {msg}", flush=True)


def interpolate_joints(waypoints: list, t: float) -> list:
    """Joint positions at time ``t`` along ``waypoints`` (linear, clamped).

    A local copy of what ``agents/cell/bt_executor.py`` does for its monitor loop.
    Deliberately not imported: this file runs in the RViz image, which mounts
    ``utils`` only and has no ``agents`` package (see the CELL_ID note above).
    """
    if not waypoints:
        return []
    if t <= float(waypoints[0].get("time_s") or 0.0):
        return list(waypoints[0].get("positions") or [])
    for prev, nxt in zip(waypoints, waypoints[1:]):
        t0 = float(prev.get("time_s") or 0.0)
        t1 = float(nxt.get("time_s") or 0.0)
        if t <= t1:
            span = t1 - t0
            frac = 0.0 if span <= 0 else (t - t0) / span
            p0 = list(prev.get("positions") or [])
            p1 = list(nxt.get("positions") or [])
            return [a + (b - a) * frac for a, b in zip(p0, p1)]
    return list(waypoints[-1].get("positions") or [])


def _declared_home(rcfg: dict) -> list:
    """The robot's declared home in radians, or [] if the cell declares none.

    Read straight from the robots.yaml block rather than importing the shared
    resolver: the viewer runs in its own image with neither the agents package
    nor numpy, and drawing an idle pose is not worth a dependency on either.
    """
    deg = rcfg.get("home_pose_deg")
    if deg:
        return [math.radians(float(v)) for v in deg]
    rad = rcfg.get("home_pose") or rcfg.get("home_joints")
    return [float(v) for v in rad] if rad else []


# ─── Obstacle / fixture visualization ───────────────────────────────────────────

def load_obstacles() -> list[dict]:
    """Load static obstacle geometry from the cell's calculated obstacles.yaml.

    This is the same file the planner + collision monitor use. Here we read it
    only to DRAW the fixture (walls, clamp rail) and the workpiece object in the
    centre so RViz shows the scene the robots are actually avoiding. Returns an
    empty list if the file is missing.
    """
    obstacles_yaml = cell_layout.obstacles_yaml(CELL_ID, CELL_BASE)
    if not obstacles_yaml.exists():
        log(f"no obstacles.yaml at {obstacles_yaml}; no obstacle markers")
        return []
    try:
        with open(obstacles_yaml) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: could not parse {obstacles_yaml}: {e}")
        return []
    obs = cfg.get("obstacles", []) or []
    log(f"loaded {len(obs)} obstacle(s) for visualization")
    return obs


def _obstacle_colors(obs: dict):
    """(fill, wire) RGBA for an obstacle, colour-coded by role."""
    oid = str(obs.get("id", "")).lower()
    is_workpiece = bool(obs.get("is_workpiece")) or "panel" in oid or "surface" in oid or "workpiece" in oid
    if is_workpiece:
        # Orange — the workpiece/object held in the middle of the cell.
        return (0.95, 0.55, 0.10, 0.55), (0.80, 0.35, 0.05, 1.0)
    if any(k in oid for k in ("fixture", "wall", "rail", "floor", "support", "body", "frame")):
        # Steel-blue — the fixture structure.
        return (0.55, 0.65, 0.75, 0.30), (0.30, 0.40, 0.55, 1.0)
    return (0.70, 0.70, 0.70, 0.30), (0.50, 0.50, 0.50, 1.0)


def build_obstacle_markers(obstacles: list[dict]):
    """Build a MarkerArray (CUBE + wireframe) for box/sphere obstacles.

    Boxes (type: box, center + half_extents) → semi-transparent CUBE + LINE_LIST
    wireframe. Spheres → SPHERE. Planes (the floor) are skipped — the RViz Grid
    already shows the floor. All geometry is in the world frame.
    """
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    from geometry_msgs.msg import Point

    markers = MarkerArray()
    mid = 0
    for obs in obstacles:
        otype = str(obs.get("type", "box")).lower()
        oid = str(obs.get("id", f"obstacle_{mid}"))
        nsp = oid.replace(" ", "_")[:24]
        fill, wire = _obstacle_colors(obs)

        if otype == "box":
            cx, cy, cz = (float(v) for v in obs.get("center", [0, 0, 0]))
            hx, hy, hz = (float(v) for v in obs.get("half_extents", [0.1, 0.1, 0.1]))

            box = Marker()
            box.header.frame_id = "world"
            box.ns = f"{nsp}_box"; box.id = mid; mid += 1
            box.type = Marker.CUBE; box.action = Marker.ADD
            box.pose.position.x = cx; box.pose.position.y = cy; box.pose.position.z = cz
            box.pose.orientation.w = 1.0
            box.scale.x = 2.0 * hx; box.scale.y = 2.0 * hy; box.scale.z = 2.0 * hz
            box.color = ColorRGBA(r=fill[0], g=fill[1], b=fill[2], a=fill[3])
            markers.markers.append(box)

            corners = [
                (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
                (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
                (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
                (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
            ]
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
            w = Marker()
            w.header.frame_id = "world"
            w.ns = f"{nsp}_wire"; w.id = mid; mid += 1
            w.type = Marker.LINE_LIST; w.action = Marker.ADD
            w.pose.orientation.w = 1.0; w.scale.x = 0.01
            w.color = ColorRGBA(r=wire[0], g=wire[1], b=wire[2], a=wire[3])
            for a, b in edges:
                w.points.append(Point(x=float(corners[a][0]), y=float(corners[a][1]), z=float(corners[a][2])))
                w.points.append(Point(x=float(corners[b][0]), y=float(corners[b][1]), z=float(corners[b][2])))
            markers.markers.append(w)

        elif otype == "sphere":
            cx, cy, cz = (float(v) for v in obs.get("center", [0, 0, 0]))
            r = float(obs.get("radius", 0.1))
            s = Marker()
            s.header.frame_id = "world"
            s.ns = f"{nsp}_sphere"; s.id = mid; mid += 1
            s.type = Marker.SPHERE; s.action = Marker.ADD
            s.pose.position.x = cx; s.pose.position.y = cy; s.pose.position.z = cz
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 2.0 * r
            s.color = ColorRGBA(r=fill[0], g=fill[1], b=fill[2], a=fill[3])
            markers.markers.append(s)
        # planes (floor) intentionally skipped — the Grid represents the floor.

    return markers


# ─── Weld-point visualization ────────────────────────────────────────────────────

def load_weld_points() -> list[dict]:
    """Load weld/operation targets from the cell's agent-owned tasks/*.json.

    Reads the task program and returns every operation that carries a 3D
    ``position`` (weld spots, via points). Used only to DRAW the targets in RViz
    so the operator can see where the robots will weld. Honours a TASK_FILE env
    override; otherwise picks the first task file that contains operations.
    Returns [] if no task file is present.
    """
    tasks_dir = cell_layout.tasks_dir(CELL_ID, CELL_BASE)
    if not tasks_dir.is_dir():
        log(f"no tasks dir at {tasks_dir}; no weld-point markers")
        return []

    override = os.environ.get("TASK_FILE")
    if override:
        p = Path(override)
        task_files = [p if p.is_absolute() else tasks_dir / override]
    else:
        task_files = sorted(tasks_dir.glob("*.json"))

    for tf in task_files:
        if not tf.exists():
            continue
        try:
            with open(tf) as f:
                prog = json.load(f)
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: could not parse task {tf}: {e}")
            continue
        points: list[dict] = []
        for op in prog.get("operations", []) or []:
            pos = op.get("position")
            if not (isinstance(pos, (list, tuple)) and len(pos) >= 3):
                continue
            points.append({
                "id":       str(op.get("id", f"op{len(points) + 1}")),
                "type":     str(op.get("type", "weld")),
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "approach": op.get("approach_vector") or [0.0, 0.0, -1.0],
                # The real tool lean at the weld lives in weld_orientation; the
                # approach arrow is drawn along it so RViz shows the 45-degree
                # lean rather than the top-down approach_vector.
                "weld_orientation": op.get("weld_orientation"),
                "group":    str(op.get("sequence_group", "")),
                "robot":    op.get("assigned_robot"),
                # Optional display label from the task file; none = no label.
                "label":    _label(op.get("label")),
            })
        if points:
            log(f"loaded {len(points)} weld point(s) from {tf.name} for visualization")
            return points

    log("no weld points found in any task file")
    return []


def _weld_group_color(group: str):
    """RGBA for a weld spot, colour-coded by sequence group."""
    g = group.lower()
    if g == "structural":
        return (0.95, 0.15, 0.10, 1.0)   # red   — structural welds
    if g == "secondary":
        return (1.0, 0.75, 0.0, 1.0)     # gold  — secondary welds
    return (0.10, 0.85, 0.95, 1.0)       # cyan  — unclassified


def _weld_state_color(group: str, state: str):
    """RGBA for a weld spot given its live execution state.

    pending → sequence-group colour, active (being welded now) → bright yellow,
    done → green.
    """
    if state == "done":
        return (0.10, 0.85, 0.20, 1.0)   # green  — completed
    if state == "active":
        return (1.0, 0.95, 0.10, 1.0)    # yellow — welding right now
    return _weld_group_color(group)      # pending


def _tool_axis_from_quat(q: dict):
    """Tool approach axis (world frame) for an orientation quaternion, using the
    project convention axis = -R[:,0]. Returns a unit (x, y, z) tuple pointing
    from the tool toward the weld (downward/leaned)."""
    x = float(q.get("x", 0.0)); y = float(q.get("y", 0.0))
    z = float(q.get("z", 0.0)); w = float(q.get("w", 1.0))
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    ax = -(1.0 - 2.0 * (y * y + z * z))
    ay = -(2.0 * (x * y + z * w))
    az = -(2.0 * (x * z - y * w))
    m = (ax * ax + ay * ay + az * az) ** 0.5 or 1.0
    return (ax / m, ay / m, az / m)


def build_weld_markers(points: list[dict], states: "dict[str, str] | None" = None):
    """Build a MarkerArray for weld targets: SPHERE spot + approach ARROW + TEXT id.

    Colour reflects live progress when ``states`` (weld_id → pending|active|done)
    is given: pending = sequence-group colour (structural=red, secondary=gold),
    active = bright yellow (enlarged), done = green. The arrow points along the
    approach vector into the spot. All geometry is in the world frame.
    """
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    from geometry_msgs.msg import Point

    states = states or {}
    markers = MarkerArray()
    mid = 0
    for p in points:
        x, y, z = p["position"]
        wid = p["id"]
        nsp = wid.replace(" ", "_")[:24]
        state = states.get(wid, "pending")
        r, g, b, a = _weld_state_color(p["group"], state)
        # Enlarge the spot while it is actively being welded so it stands out.
        spot_d = 0.06 if state == "active" else 0.04

        # Weld spot — sphere at the target.
        s = Marker()
        s.header.frame_id = "world"
        s.ns = f"{nsp}_spot"; s.id = mid; mid += 1
        s.type = Marker.SPHERE; s.action = Marker.ADD
        s.pose.position.x = x; s.pose.position.y = y; s.pose.position.z = z
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = spot_d
        s.color = ColorRGBA(r=r, g=g, b=b, a=a)
        markers.markers.append(s)

        # Approach arrow — points along the tool axis into the spot. Prefer the
        # weld's leaned tool orientation (weld_orientation) so the arrow matches
        # the 45-degree lean actually executed; fall back to approach_vector.
        wo = p.get("weld_orientation")
        if wo:
            ax, ay, az = _tool_axis_from_quat(wo)
        else:
            ax, ay, az = (list(p["approach"]) + [0.0, 0.0, -1.0])[:3]
        length = 0.18
        tail = Point(x=x - ax * length, y=y - ay * length, z=z - az * length)
        head = Point(x=x, y=y, z=z)
        arr = Marker()
        arr.header.frame_id = "world"
        arr.ns = f"{nsp}_approach"; arr.id = mid; mid += 1
        arr.type = Marker.ARROW; arr.action = Marker.ADD
        arr.pose.orientation.w = 1.0
        arr.points = [tail, head]
        arr.scale.x = 0.008   # shaft diameter
        arr.scale.y = 0.025   # head diameter
        arr.scale.z = 0.04    # head length
        arr.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
        markers.markers.append(arr)

        # The weld's configured label — camera-facing text just above the spot,
        # white so it reads on any state colour. No label configured, no text.
        label = _label(p.get("label"))
        if not label:
            continue
        t = Marker()
        t.header.frame_id = "world"
        t.ns = f"{nsp}_label"; t.id = mid; mid += 1
        t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
        t.pose.position.x = x; t.pose.position.y = y; t.pose.position.z = z + WELD_LABEL_LIFT_M
        t.pose.orientation.w = 1.0
        t.scale.z = WELD_LABEL_HEIGHT_M        # text height; x/y are unused
        t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        t.text = label
        markers.markers.append(t)

    return markers


# Label sizes, in metres. A weld label sits just above its spot; a robot label
# sits low beside the robot's base on the CELL side (toward the centroid of the
# bases), in front of the robot where the default camera sees it — outward put it
# behind the base or out of view — next to a coloured disc marking the base.
WELD_LABEL_HEIGHT_M = 0.08
WELD_LABEL_LIFT_M = 0.07
ROBOT_LABEL_HEIGHT_M = 0.30
ROBOT_LABEL_OFFSET_M = 1.0
ROBOT_LABEL_LIFT_M = 0.30
ROBOT_DISC_DIAMETER_M = 1.5
_ROBOT_COLORS = [(0.20, 0.60, 1.00), (1.00, 0.55, 0.10), (0.40, 0.85, 0.40),
                 (0.85, 0.40, 0.85)]


def _label(value) -> "str | None":
    """A configured display label, or None when absent or blank."""
    text = str(value).strip() if value is not None else ""
    return text or None


def build_robot_label_markers(robots: list[dict]):
    """A label and a coloured floor disc at the base of each LABELLED robot.

    The text is the robot's ``label`` from robots.yaml; a robot without one gets
    neither label nor disc. It is placed beside
    the base, offset toward the centroid of all bases (the cell side, in front of
    the robot). With a single robot there is no centroid to face, and the label
    goes on +x.
    """
    import math
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA

    markers = MarkerArray()
    if not robots:
        return markers
    bases = [list(r.get("base") or [0.0, 0.0, 0.0]) + [0.0] * 3 for r in robots]
    cx = sum(b[0] for b in bases) / len(bases)
    cy = sum(b[1] for b in bases) / len(bases)
    mid = 0
    for i, (r, base) in enumerate(zip(robots, bases)):
        label = _label(r.get("label"))
        if not label:
            continue
        bx, by, bz = float(base[0]), float(base[1]), float(base[2])
        dx, dy = cx - bx, cy - by
        n = math.hypot(dx, dy)
        ux, uy = (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)
        cr, cg, cb = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
        nsp = str(r.get("robot_id", f"robot{i + 1}"))

        disc = Marker()
        disc.header.frame_id = "world"
        disc.ns = f"{nsp}_base"; disc.id = mid; mid += 1
        disc.type = Marker.CYLINDER; disc.action = Marker.ADD
        disc.pose.position.x = bx; disc.pose.position.y = by; disc.pose.position.z = bz + 0.005
        disc.pose.orientation.w = 1.0
        disc.scale.x = disc.scale.y = ROBOT_DISC_DIAMETER_M
        disc.scale.z = 0.01
        disc.color = ColorRGBA(r=cr, g=cg, b=cb, a=0.6)
        markers.markers.append(disc)

        t = Marker()
        t.header.frame_id = "world"
        t.ns = f"{nsp}_label"; t.id = mid; mid += 1
        t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
        t.pose.position.x = bx + ux * ROBOT_LABEL_OFFSET_M
        t.pose.position.y = by + uy * ROBOT_LABEL_OFFSET_M
        t.pose.position.z = bz + ROBOT_LABEL_LIFT_M
        t.pose.orientation.w = 1.0
        t.scale.z = ROBOT_LABEL_HEIGHT_M
        t.color = ColorRGBA(r=cr, g=cg, b=cb, a=1.0)
        t.text = label
        markers.markers.append(t)
    return markers


# ─── URDF handling ────────────────────────────────────────────────────────────

def load_robot_spec() -> list[dict]:
    """
    Parse robots.yaml → list of {ns, robot_id, name, base, yaw, urdf_path}.

    Iterates every (namespace, robot_id) pair, so a single shared namespace
    (e.g. 'arms') containing multiple robots produces one entry per robot.
    Frames/topics are keyed by robot_id so they stay unique even when the
    namespace is shared.
    """
    robots_yaml = cell_layout.robots_yaml(CELL_ID, CELL_BASE)
    with open(robots_yaml) as f:
        cfg = yaml.safe_load(f)

    robots: list[dict] = []
    for ns, ns_block in (cfg.get("robots") or {}).items():
        if not isinstance(ns_block, dict):
            continue
        for robot_id, rcfg in ns_block.items():
            if not isinstance(rcfg, dict):
                continue
            name = rcfg.get("name")
            # The cell's single answer to "what shape is this arm" — the same
            # document the planners load, so RViz cannot show a robot the planner
            # is not planning for. cell_layout picks it: referenced in cell.yaml
            # where the source bundle put it, or the converter's baked urdf/<rid>/.
            urdf = cell_layout.robot_urdf(CELL_ID, robot_id, CELL_BASE)
            if urdf is None:
                log(f"WARNING: no URDF for robot '{robot_id}' in cell {CELL_ID}; skipping")
                continue
            # Where a RELATIVE "meshes/..." ref resolves. A referenced URDF keeps
            # its meshes beside itself; a baked one carries absolute URIs and never
            # reaches the rewrite below. Preferring the URDF's own directory when it
            # actually holds the meshes covers both without a flag saying which.
            mesh_base = cell_layout.urdf_dir(CELL_ID, cell_base=CELL_BASE)
            if (urdf.parent / "meshes").is_dir():
                mesh_base = urdf.parent
            robots.append({
                "ns":        ns,
                "robot_id":  robot_id,
                "name":      name,
                "urdf_xml":  _robot_description(robot_id),
                "model_dir": urdf.parent,
                "mesh_base": mesh_base,
                "urdf_path": urdf,
                "base":      rcfg.get("base_position", [0.0, 0.0, 0.0]),
                "yaw":       float(rcfg.get("base_orientation_yaw", 0.0)),
                "home":      _declared_home(rcfg),
                # Optional display label from robots.yaml; none = no label.
                "label":     _label(rcfg.get("label")),
            })
    return robots


def _robot_description(robot_id: str) -> "str | None":
    """The arm's URDF from the trajectory agent's MCP (``get_robot_description``)
    — the document the planner plans with — not from a DDS topic. The meshes it
    references still resolve on this container's read-only mount. Falls back to the
    mounted file, loudly: the viewer is read-only, and a blank view helps nobody.
    Retries a refused connection for a while: the viewer can start before the agent."""
    import time
    import urllib.error
    import urllib.request
    host = os.environ.get("TRAJECTORY_AGENT_HOST", "trajectory-agent")
    port = os.environ.get("TRAJECTORY_AGENT_PORT", "8890")
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "get_robot_description",
                                  "arguments": {"cell_id": CELL_ID, "robot_id": robot_id}}}).encode()
    deadline = time.monotonic() + float(os.environ.get("ROBOT_DESCRIPTION_WAIT_S", "60"))
    while True:
        try:
            req = urllib.request.Request(f"http://{host}:{port}/mcp", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                reply = json.loads(resp.read().decode() or "{}")
            r = json.loads(((reply.get("result") or {}).get("content") or [{}])[0].get("text") or "{}")
            if r.get("found") and r.get("urdf"):
                return r["urdf"]
            log(f"WARNING: no robot description for '{robot_id}' from the planner "
                f"({r.get('error') or 'not found'}); using the mounted URDF")
            return None
        except (urllib.error.URLError, OSError) as exc:
            if time.monotonic() > deadline:
                log(f"WARNING: robot description for '{robot_id}' not available over MCP "
                    f"({exc}); using the mounted URDF")
                return None
            time.sleep(2.0)
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: robot description for '{robot_id}' unreadable over MCP ({exc}); "
                f"using the mounted URDF")
            return None


def build_prefixed_urdf(robot: dict) -> tuple[str, list[str]]:
    """
    Return (urdf_xml_string, ordered_actuated_joint_names) for a robot, with:
      - every link/joint name prefixed with "<ns>_"
      - mesh filenames rewritten to absolute file:// URIs
    """
    robot_id = robot["robot_id"]
    mesh_base = robot["mesh_base"].resolve()
    tree = (ET.ElementTree(ET.fromstring(robot["urdf_xml"])) if robot.get("urdf_xml")
            else ET.parse(robot["urdf_path"]))
    root = tree.getroot()

    def pref(v: str) -> str:
        return f"{robot_id}_{v}"

    for link in root.findall("link"):
        link.set("name", pref(link.get("name")))

    joint_names: list[str] = []
    for joint in root.findall("joint"):
        joint.set("name", pref(joint.get("name")))
        for tag in ("parent", "child"):
            el = joint.find(tag)
            if el is not None and el.get("link"):
                el.set("link", pref(el.get("link")))
        if joint.get("type") in _ACTUATED:
            joint_names.append(joint.get("name"))

    # Rewrite RELATIVE mesh paths: "./meshes/..." or "meshes/..." -> an absolute
    # file:// URI under the cell's urdf/. A filename that already carries a scheme
    # or is already absolute is left exactly as written — the converter bakes
    # absolute in-container URIs, and prefixing those would produce
    # "file://<dir>/file:///cells/..." and silently lose every mesh.
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        if "://" in fn or fn.startswith("/"):
            continue
        rel = fn[2:] if fn.startswith("./") else fn
        if rel.startswith("meshes/"):
            mesh.set("filename", f"file://{mesh_base}/{rel}")
        elif "/meshes/" in rel:
            # Some other relative shape ("../meshes/x", "sub/meshes/x"). Resolve it
            # the way URDF says to — against the directory holding the URDF.
            here = Path(robot["urdf_path"]).resolve().parent
            mesh.set("filename", f"file://{(here / rel).resolve()}")

    xml_str = ET.tostring(root, encoding="unicode")
    return xml_str, joint_names


# ─── Process launchers ─────────────────────────────────────────────────────────

def start_robot_state_publisher(robot_id: str, urdf_xml: str) -> subprocess.Popen:
    params = {"/**": {"ros__parameters": {
        "robot_description": urdf_xml,
        "publish_frequency": 30.0,
    }}}
    params_file = f"/tmp/viz_{robot_id}_rsp.yaml"
    with open(params_file, "w") as f:
        yaml.safe_dump(params, f)
    cmd = [
        "ros2", "run", "robot_state_publisher", "robot_state_publisher",
        "--ros-args",
        "-r", f"__ns:=/viz/{robot_id}",
        "--params-file", params_file,
    ]
    log(f"robot_state_publisher for '{robot_id}' (ns=/viz/{robot_id})")
    return subprocess.Popen(cmd)


def start_static_tf(robot_id: str, base: list[float], yaw: float) -> subprocess.Popen:
    x, y, z = (list(base) + [0.0, 0.0, 0.0])[:3]
    cmd = [
        "ros2", "run", "tf2_ros", "static_transform_publisher",
        "--x", str(x), "--y", str(y), "--z", str(z),
        "--yaw", str(yaw), "--pitch", "0", "--roll", "0",
        "--frame-id", "world", "--child-frame-id", f"{robot_id}_World",
    ]
    log(f"static tf world -> {robot_id}_World at [{x}, {y}, {z}] yaw={yaw}")
    return subprocess.Popen(cmd)


# ─── RViz config generation ─────────────────────────────────────────────────────

def write_rviz_config(robots: list[dict], path: str) -> None:
    displays = [
        {"Class": "rviz_default_plugins/Grid", "Name": "Grid",
         "Enabled": True, "Value": True, "Plane Cell Count": 20,
         "Reference Frame": "world"},
        {"Class": "rviz_default_plugins/TF", "Name": "TF", "Enabled": True,
         "Value": True, "Show Names": False, "Marker Scale": 1.0},
    ]
    for r in robots:
        displays.append({
            "Class": "rviz_default_plugins/RobotModel",
            "Name": f"RobotModel {r['robot_id']}",
            "Enabled": True,
            "Visual Enabled": True,
            "Collision Enabled": False,
            "Description Topic": {
                "Value": f"/viz/{r['robot_id']}/robot_description",
                "Depth": 5,
                "Durability Policy": "Transient Local",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
            },
        })
        # The step-through ghost (see GHOST_SUFFIX). Translucent so the real arm
        # stays readable underneath it, and listed after the solid model so it
        # blends over it rather than the other way round.
        displays.append({
            "Class": "rviz_default_plugins/RobotModel",
            "Name": f"Preview ghost {r['robot_id']}",
            "Enabled": True,
            "Visual Enabled": True,
            "Collision Enabled": False,
            "Alpha": GHOST_ALPHA,
            "Description Topic": {
                "Value": f"/viz/{ghost_id(r['robot_id'])}/robot_description",
                "Depth": 5,
                "Durability Policy": "Transient Local",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
            },
        })

    # Fixture + workpiece obstacle markers (the box/support in the middle).
    displays.append({
        "Class": "rviz_default_plugins/MarkerArray",
        "Name": "Fixture / Obstacles",
        "Enabled": True,
        "Namespaces": {},
        "Topic": {
            "Value": "/viz/obstacles",
            "Depth": 5,
            "Durability Policy": "Transient Local",
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
        },
    })

    # Weld targets (spot spheres + approach arrows + id labels).
    displays.append({
        "Class": "rviz_default_plugins/MarkerArray",
        "Name": "Weld Points",
        "Enabled": True,
        "Namespaces": {},
        "Topic": {
            "Value": "/viz/weld_points",
            "Depth": 5,
            "Durability Policy": "Transient Local",
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
        },
    })

    # Robot names at their bases.
    displays.append({
        "Class": "rviz_default_plugins/MarkerArray",
        "Name": "Robot Labels",
        "Enabled": True,
        "Namespaces": {},
        "Topic": {
            "Value": "/viz/robot_labels",
            "Depth": 5,
            "Durability Policy": "Transient Local",
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
        },
    })

    cfg = {
        "Panels": [{"Class": "rviz_common/Displays", "Name": "Displays"}],
        "Visualization Manager": {
            "Class": "",
            "Displays": displays,
            "Global Options": {
                "Fixed Frame": "world",
                "Background Color": "48; 48; 48",
                "Frame Rate": 30,
            },
            "Tools": [
                {"Class": "rviz_default_plugins/MoveCamera"},
                {"Class": "rviz_default_plugins/Select"},
                {"Class": "rviz_default_plugins/FocusCamera"},
            ],
            "Views": {
                "Current": {
                    "Class": "rviz_default_plugins/Orbit",
                    "Name": "Current View",
                    "Distance": 9.0,
                    "Pitch": 0.6,
                    "Yaw": 0.8,
                    "Focal Point": {"X": -0.5, "Y": -1.1, "Z": 0.6},
                },
            },
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    log(f"wrote RViz config -> {path}")


# ─── joint_states relay node ─────────────────────────────────────────────────────

def run_relay_and_rviz(robots: list[dict], rviz_cfg: str) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray

    rclpy.init()
    node = Node("viz_joint_relay")

    # ── Fixture / workpiece obstacle markers ────────────────────────────────
    # Transient-local + reliable so a late-joining RViz still receives them; we
    # also republish on a slow timer as a keepalive.
    _obstacle_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    obstacle_markers = build_obstacle_markers(load_obstacles())
    obstacle_pub = node.create_publisher(MarkerArray, "/viz/obstacles", _obstacle_qos)

    def _publish_obstacles():
        stamp = node.get_clock().now().to_msg()
        for m in obstacle_markers.markers:
            m.header.stamp = stamp
        obstacle_pub.publish(obstacle_markers)

    if obstacle_markers.markers:
        _publish_obstacles()
        node.create_timer(2.0, _publish_obstacles)
        log(f"publishing {len(obstacle_markers.markers)} obstacle marker(s) on /viz/obstacles")

    # ── Robot name labels at the bases ──────────────────────────────────────
    robot_label_markers = build_robot_label_markers(robots)
    robot_label_pub = node.create_publisher(MarkerArray, "/viz/robot_labels", _obstacle_qos)

    def _publish_robot_labels():
        stamp = node.get_clock().now().to_msg()
        for m in robot_label_markers.markers:
            m.header.stamp = stamp
        robot_label_pub.publish(robot_label_markers)

    if robot_label_markers.markers:
        _publish_robot_labels()
        node.create_timer(2.0, _publish_robot_labels)
        log(f"publishing labels for {len(robot_label_markers.markers) // 2} robot(s) "
            f"on /viz/robot_labels")

    # ── Weld-point markers with live progress colouring ─────────────────────
    weld_points = load_weld_points()
    weld_ids = [p["id"] for p in weld_points]
    weld_id_set = set(weld_ids)
    weld_state = {wid: "pending" for wid in weld_ids}   # pending|active|done
    last_active: dict[str, str] = {}                    # robot_id -> active weld id
    weld_pub = node.create_publisher(MarkerArray, "/viz/weld_points", _obstacle_qos)

    def _publish_welds():
        markers = build_weld_markers(weld_points, weld_state)
        stamp = node.get_clock().now().to_msg()
        for m in markers.markers:
            m.header.stamp = stamp
        weld_pub.publish(markers)

    if weld_points:
        _publish_welds()
        node.create_timer(2.0, _publish_welds)
        log(f"publishing {len(weld_points)} weld point(s) on /viz/weld_points")

        # Track weld progress from each robot's status stream so the markers
        # recolour live: the operation a robot is currently executing turns
        # yellow; once it moves on (or goes idle/home) that weld turns green.
        def _make_status_cb(robot_id: str):
            def _cb(msg):
                try:
                    data = json.loads(msg.data)
                except Exception:
                    return
                task = data.get("current_task")
                active_weld = task if task in weld_id_set else None
                prev = last_active.get(robot_id)
                changed = False
                if active_weld and active_weld != prev:
                    if prev and weld_state.get(prev) != "done":
                        weld_state[prev] = "done"; changed = True
                    if weld_state.get(active_weld) != "active":
                        weld_state[active_weld] = "active"; changed = True
                    last_active[robot_id] = active_weld
                elif active_weld is None and prev is not None:
                    if weld_state.get(prev) != "done":
                        weld_state[prev] = "done"; changed = True
                    last_active[robot_id] = None
                if changed:
                    _publish_welds()
            return _cb

        for r in robots:
            status_topic = f"/{CELL_ID}/{r['ns']}/{r['robot_id']}/robot_status"
            node.create_subscription(String, status_topic, _make_status_cb(r["robot_id"]), 10)
            log(f"tracking weld progress from {status_topic}")

    # Ghost plumbing, filled in per robot below: the publisher that drives each
    # ghost, its URDF joint names (prefixed differently from the real arm's), and
    # the live real pose the ghost parks on when nothing is being previewed.
    ghost_pubs: dict = {}
    ghost_joints: dict = {}
    real_state: dict = {}

    for r in robots:
        ns = r["ns"]
        robot_id = r["robot_id"]
        joints = r["joints"]
        n = len(joints)
        if r.get("ghost_joints"):
            ghost_joints[robot_id] = r["ghost_joints"]
            ghost_pubs[robot_id] = node.create_publisher(
                JointState, f"/viz/{ghost_id(robot_id)}/joint_states",
                QoSProfile(depth=10))
        idle = r.get("home") or HOME_JOINTS
        state = {"pos": list(idle[:n]) + [0.0] * max(0, n - len(idle))}

        pub = node.create_publisher(JointState, f"/viz/{robot_id}/joint_states", QoSProfile(depth=10))
        src_topic = f"/{CELL_ID}/{ns}/{robot_id}/joint_states"

        def make_cb(st):
            def _cb(msg):
                if msg.position:
                    st["pos"] = list(msg.position)
            return _cb

        node.create_subscription(JointState, src_topic, make_cb(state), 10)

        def make_timer(js_pub, jnames, st):
            def _tick():
                m = JointState()
                m.header.stamp = node.get_clock().now().to_msg()
                m.name = jnames
                pos = st["pos"]
                # map by position; pad/trim to the URDF joint count
                m.position = [float(pos[i]) if i < len(pos) else 0.0 for i in range(len(jnames))]
                js_pub.publish(m)
            return _tick

        node.create_timer(0.05, make_timer(pub, joints, state))  # 20 Hz
        log(f"relay {src_topic} -> /viz/{robot_id}/joint_states ({n} joints, HOME fallback)")

        real_state[robot_id] = state

    import time

    # ── Step-through preview: drive the ghost arms, never the real ones ──────
    # One entry per robot while a preview is running:
    #   {waypoints, duration_s, t0, speed, loop, label}
    # Absent ⇒ that ghost mirrors its real arm (see PREVIEW_IDLE_MIRRORS_REAL).
    preview: dict[str, dict] = {}

    def _preview_cb(msg) -> None:
        """Handle one /<cell>/node_preview message.

        Contract (std_msgs/String holding JSON), all optional but ``robot_ns``:
          {"robot_ns": "robot1", "waypoints": [{"positions": [...], "time_s": 0.0}],
           "speed": 1.0, "loop": true, "label": "move_robot1_VIA_W06 · approach"}
        ``{"clear": true}`` with no ``robot_ns`` parks every ghost.
        """
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            log(f"preview: ignoring unparseable message ({exc})")
            return
        rid = data.get("robot_ns") or data.get("robot_id")
        waypoints = data.get("waypoints") or []
        if data.get("clear") or not waypoints:
            if rid:
                preview.pop(rid, None)
                log(f"preview: cleared for '{rid}'")
            else:
                preview.clear()
                log("preview: cleared for every robot")
            return
        if rid not in ghost_pubs:
            log(f"preview: no ghost for robot '{rid}' — known: {sorted(ghost_pubs)}")
            return
        speed = float(data.get("speed") or 1.0)
        preview[rid] = {
            "waypoints": waypoints,
            "duration_s": float(waypoints[-1].get("time_s") or 0.0),
            "t0": time.time(),
            # A zero or negative speed would freeze or run the motion backwards
            # forever; neither is a preview anyone asked for.
            "speed": speed if speed > 0 else 1.0,
            "loop": bool(data.get("loop", True)),
            "label": str(data.get("label") or ""),
        }
        log(f"preview: '{rid}' {len(waypoints)} waypoint(s), "
            f"{preview[rid]['duration_s']:.2f}s at {preview[rid]['speed']}x "
            f"{'looping' if preview[rid]['loop'] else 'once'} — {preview[rid]['label']}")

    node.create_subscription(String, PREVIEW_TOPIC, _preview_cb, 10)
    log(f"step-through previews on {PREVIEW_TOPIC} -> ghost arms "
        f"({', '.join(ghost_pubs) or 'none'})")

    def make_ghost_timer(robot_id: str):
        pub_g = ghost_pubs[robot_id]
        jnames = ghost_joints[robot_id]
        st = real_state.get(robot_id, {"pos": []})

        def _tick():
            active = preview.get(robot_id)
            if active:
                elapsed = (time.time() - active["t0"]) * active["speed"]
                if elapsed >= active["duration_s"]:
                    if active["loop"]:
                        active["t0"] = time.time()
                        elapsed = 0.0
                    else:
                        elapsed = active["duration_s"]
                pos = interpolate_joints(active["waypoints"], elapsed)
            elif PREVIEW_IDLE_MIRRORS_REAL:
                pos = st["pos"]
            else:
                return
            m = JointState()
            m.header.stamp = node.get_clock().now().to_msg()
            m.name = jnames
            m.position = [float(pos[i]) if i < len(pos) else 0.0
                          for i in range(len(jnames))]
            pub_g.publish(m)
        return _tick

    for robot_id in ghost_pubs:
        node.create_timer(1.0 / PREVIEW_RATE_HZ, make_ghost_timer(robot_id))

    def _spawn_rviz() -> subprocess.Popen:
        return subprocess.Popen(["rviz2", "-d", rviz_cfg])

    rviz_proc = _spawn_rviz()
    spawn_t = time.time()
    backoff = 2.0
    log("rviz2 launched")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if rviz_proc.poll() is not None:
                # rviz2 exited — usually because the X server went away (e.g. a
                # container restart). Keep the relay + marker publishers alive
                # and relaunch rviz2 with a capped backoff so a transient
                # display loss self-heals. If rviz survived a while, reset the
                # backoff so the next crash recovers quickly.
                if time.time() - spawn_t > 30.0:
                    backoff = 2.0
                log(f"rviz2 exited (code {rviz_proc.returncode}); relaunching in {backoff:.0f}s")
                wait_until = time.time() + backoff
                while rclpy.ok() and time.time() < wait_until:
                    rclpy.spin_once(node, timeout_sec=0.2)  # keep publishers alive
                rviz_proc = _spawn_rviz()
                spawn_t = time.time()
                backoff = min(backoff * 1.5, 15.0)
    except KeyboardInterrupt:
        pass
    finally:
        if rviz_proc is not None and rviz_proc.poll() is None:
            rviz_proc.terminate()
        node.destroy_node()
        rclpy.shutdown()


# ─── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    log(f"cell='{CELL_ID}' base='{CELL_BASE}'")
    robots = load_robot_spec()
    if not robots:
        log("ERROR: no robots found; launching bare RViz")
        subprocess.Popen(["rviz2"])
        return 0

    for r in robots:
        urdf_xml, joints = build_prefixed_urdf(r)
        r["joints"] = joints
        start_robot_state_publisher(r["robot_id"], urdf_xml)
        start_static_tf(r["robot_id"], r["base"], r["yaw"])
        log(f"robot '{r['robot_id']}' (ns={r['ns']}) model='{r['name']}' joints={joints}")

        # Second copy of the same arm for step-through previews. The name prefix
        # comes from robot_id, so re-prefixing a renamed copy is all it takes to
        # get non-colliding link and TF frame names — and the same static base
        # transform puts it exactly where the real arm stands.
        ghost = dict(r, robot_id=ghost_id(r["robot_id"]))
        ghost_urdf, ghost_joint_names = build_prefixed_urdf(ghost)
        r["ghost_joints"] = ghost_joint_names
        start_robot_state_publisher(ghost["robot_id"], ghost_urdf)
        start_static_tf(ghost["robot_id"], r["base"], r["yaw"])

    rviz_cfg = "/tmp/viz.rviz"
    write_rviz_config(robots, rviz_cfg)
    run_relay_and_rviz(robots, rviz_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
