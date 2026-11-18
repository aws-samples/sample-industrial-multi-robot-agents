#!/usr/bin/env python3
"""
Web Control Panel for Multi-Arm Robot System

Provides a browser-based GUI at http://localhost:8888 with:
- Architecture diagram of deployed agents (live blink on DDS message)
- Start/Stop controls, plus Replay on the tree it executes (BTree tab)
- Per-weld planning log: allocated or not, the failures, and the reasoning
- Live planning log with enriched action annotations
- Collision alerts log
"""

import os
import sys
import json
import glob
import re
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

from flask import Flask, render_template_string, jsonify, request as flask_request

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace/multi_robots_agents")
sys.path.insert(0, "/agents")
sys.path.insert(0, "/utils")

# Cell identity + topic prefix. multi_robots_agents topics are cell-scoped as
# /{cell}/{namespace}/{robot_id}/... and cell-wide as /{cell}/<topic>.
from agents.cell_defaults import default_cell_id
from utils.llm_text import extract_text as _extract_text

from utils import cell_layout

CELL_ID = default_cell_id().strip("/")


def cell_topic(t: str) -> str:
    return f"/{CELL_ID}{t}"


# ─── Ask the owning agent, not the volume ─────────────────────────────────────
#
# This panel shares the /cells mount with the agents, and it used to answer
# several questions by reading that mount itself: robots.yaml for the arm list,
# tasks/*.json for the chat context, recordings/*.rrd and plans/*.json for the
# recording list. Every one of those files belongs to an agent that already
# publishes it as an MCP tool, and a second independent reader is exactly how the
# two drifted apart — this file globbed `plan_*.json` for recordings long after
# the recorder had moved to `<tree_id>.rrd`, so the panel reported "no
# recordings" with two of them on disk, and it summarised task programs with a
# schema (`tasks[].parameters.points`) that no writer in the repo has produced
# for as long as the format has been `operations[]`.
#
# So the reads go through MCP, over the same plain JSON-RPC `/mcp` endpoint the
# orchestrator uses (see utils/mcp_http.py). urllib, because this process has no
# httpx; short timeouts, because these calls sit inside a page load.
MCP_TIMEOUT_S = 6.0


def _traj_mcp() -> str:
    """The trajectory agent's endpoint — owns plans, traces, task programs."""
    host = os.environ.get("TRAJECTORY_AGENT_HOST", "trajectory-agent")
    port = os.environ.get("TRAJECTORY_AGENT_PORT", "8890")
    return f"http://{host}:{port}/mcp"


def _cell_mcp() -> str:
    """The cell controller's endpoint — owns execution and its recordings."""
    host = os.environ.get("CELL_CONTROLLER_HOST", "cell-controller")
    port = os.environ.get("CELL_MCP_PORT", "8892")
    return f"http://{host}:{port}/mcp"


def _mcp_call(url: str, tool: str, arguments: dict,
              timeout: float = MCP_TIMEOUT_S) -> dict:
    """Call one MCP tool and return its parsed result.

    Never raises. An agent that is down, slow or answering nonsense has to
    degrade one panel of this page, not blank it — so every failure comes back as
    ``{"success": False, "error": ...}`` and the caller decides what to show.
    ``_mcp_failed`` is the matching "did this work" test.
    """
    import urllib.request as _ur
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": tool, "arguments": arguments}}
    try:
        req = _ur.Request(url, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"},
                          method="POST")
        with _ur.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except Exception as exc:      # noqa: BLE001 — HTTPError, timeout, bad JSON
        return {"success": False, "error": f"{tool}: MCP call failed: {exc}"}
    if body.get("error"):
        return {"success": False,
                "error": f"{tool}: {(body['error'] or {}).get('message', 'MCP error')}"}
    content = (body.get("result") or {}).get("content") or []
    if not content:
        return {"success": False, "error": f"{tool}: empty MCP response"}
    try:
        return json.loads(content[0].get("text") or "{}")
    except Exception:             # noqa: BLE001 — a tool that returned prose
        return {"success": False, "error": f"{tool}: MCP result was not JSON"}


def _mcp_failed(result: dict) -> str:
    """The reason an MCP result is unusable, or "" when it carries data.

    Not every tool answers with a ``success`` flag — ``get_cell_info`` just
    returns the cell — so a MISSING flag is not a failure. An explicitly false
    one, or an ``error``, is.
    """
    if not isinstance(result, dict):
        return "MCP result was not an object"
    if result.get("success") is False or result.get("error"):
        return str(result.get("error") or "refused")
    return ""


# How long the arm list is reused before asking again. It is asked on every
# /api/topology poll and on every page render, and it changes only when the cell
# is reconfigured — which is a container restart, not a page refresh.
_ROBOTS_TTL_S = 30.0
_robots_cache: dict = {"pairs": [], "at": 0.0}


def _robots_from_volume() -> list:
    """[(namespace, robot_id)] read straight from robots.yaml (two-level layout).

    The fallback for ``_discover_robots``, and deliberately still here: see the
    note there for why the panel must not depend on another container to know
    which arms exist.
    """
    import yaml
    path = cell_layout.robots_yaml(CELL_ID)
    pairs = []
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        for ns, block in (cfg.get("robots") or {}).items():
            if isinstance(block, dict):
                for rid, rcfg in block.items():
                    if isinstance(rcfg, dict):
                        pairs.append((ns, rid))
    except Exception:
        pass
    return pairs


def _discover_robots() -> list:
    """Return [(namespace, robot_id)] for this cell's arms.

    multi_robots_agents uses a shared namespace ('arms') with unique robot ids
    (robot1, robot2), so topics are /{cell}/arms/robot1/... — NOT /{cell}/ns/ns.

    Asked of the trajectory agent, which owns robots.yaml, and cached for
    _ROBOTS_TTL_S. It falls back to reading that file directly when the agent
    cannot be reached, because this list decides which ROS topics the panel
    subscribes to: a panel that started while the trajectory agent was still
    booting and got an empty answer would sit subscribed to no arm at all, with
    nothing on screen to say why. The fallback reads the same file the agent
    reads, so the two can only disagree about freshness, never about content.
    """
    now = time.time()
    if _robots_cache["pairs"] and now - _robots_cache["at"] < _ROBOTS_TTL_S:
        return _robots_cache["pairs"]

    pairs = []
    info = _mcp_call(_traj_mcp(), "get_cell_info", {"cell_id": CELL_ID})
    if not _mcp_failed(info):
        for rid, rcfg in (info.get("robots") or {}).items():
            if isinstance(rcfg, dict) and rcfg.get("namespace"):
                pairs.append((str(rcfg["namespace"]), str(rid)))
    if not pairs:
        pairs = _robots_from_volume()
    if pairs:
        _robots_cache.update(pairs=pairs, at=now)
    return pairs or [("arms", "robot1"), ("arms", "robot2")]


_base_pose_cache: dict = {"poses": {}, "at": 0.0}


def _robot_base_pose(robot_id: str) -> dict:
    """``{"base_position": [x, y, z], "base_orientation_yaw": rad}`` for one arm.

    Asked of the trajectory agent (``get_cell_info``) rather than read out of
    robots.yaml, so the world placement the panel draws with is the same one the
    planner planned with. Cached for _ROBOTS_TTL_S because forward kinematics
    runs on every joint_states message — at 10 Hz per arm an uncached MCP call
    per sample would spend the whole budget asking a question whose answer only
    changes on a container restart.

    An unknown arm, or an unreachable agent, yields the identity pose: FK then
    reports the position in the arm's OWN base frame. That is wrong-but-bounded
    (an offset), and it is what the previous code did when robots.yaml had no
    entry for the namespace.
    """
    now = time.time()
    if now - _base_pose_cache["at"] >= _ROBOTS_TTL_S:
        # The attempt is timestamped BEFORE the call and whatever its outcome, so
        # a trajectory agent that is down costs one blocked call per TTL and not
        # one per joint_states sample — the negative case is the hot one here.
        _base_pose_cache["at"] = now
        info = _mcp_call(_traj_mcp(), "get_cell_info", {"cell_id": CELL_ID})
        if not _mcp_failed(info):
            poses = {}
            for rid, rcfg in (info.get("robots") or {}).items():
                if isinstance(rcfg, dict):
                    poses[str(rid)] = {
                        "base_position": list(rcfg.get("base_position") or [0, 0, 0]),
                        "base_orientation_yaw": float(
                            rcfg.get("base_orientation_yaw", 0.0)),
                    }
            if poses:
                _base_pose_cache["poses"] = poses
    return _base_pose_cache["poses"].get(
        robot_id, {"base_position": [0.0, 0.0, 0.0], "base_orientation_yaw": 0.0})


# _default_program() lived here: "the first .json in the tasks directory", used by
# the old Run/Agentic button. It is gone with that button on purpose — the
# Planning button sends the program the operator SELECTED, and a fallback default
# is how minutes of planning would silently aim at the wrong welds.


app = Flask(__name__)

MAX_LOG_ENTRIES = 200

# Host port the rviz-web container publishes noVNC on. The simulation view in the
# left column is an iframe onto THAT container, not onto this panel, so the port
# has to be the published one — scripts/generate_compose.py passes it in from the
# single place the mapping is declared (NOVNC_HOST_PORT), and the default only
# matters when the panel is run by hand.
NOVNC_PORT = os.environ.get("NOVNC_HOST_PORT", "6080")

# What POST /api/command accepts. Anything else is a 400, not an ack — see the
# end of send_command().
KNOWN_COMMANDS = ("planning", "stop", "reset", "estop")

# ─── Shared state ─────────────────────────────────────────────────────────────

state = {
    "status": "idle",
    "planning_log": [],
    "collision_alerts": [],
    "recordings": [],
    "current_plan_info": {},
    "robot_positions": {},
    "agent_activity": {},
    "robot_states": {},
    "velocity_scale": 2.0,   # default 2× speed
    "loop_active": False,
}

# ─── Live planning progress, straight off DDS ─────────────────────────────────
#
# The planner publishes every plan event on ``/<cell>/plan_progress`` (see
# ``agents/trajectory/plan_progress.py``). Following a run used to mean polling
# this panel, which polled the orchestrator, which read the planner's event file
# off a shared volume — three processes and a filesystem in the path of a question
# about *now*, and the panel is already a ROS node subscribed to this cell's other
# topics. So the live view listens instead.
#
# The JSONL artefact the orchestrator keeps is still the record: DDS history is
# bounded and answers "what is happening", a file is unbounded and answers "what
# happened". ``/api/a2a/plan_events`` therefore stays exactly as it was, and the
# browser falls back to it whenever this feed says it is not available — a panel
# built without rclpy, or a middleware that is down, must still show the run.
MAX_PLAN_PROGRESS_EVENTS = 400

# How long a silent run stays "planning". A long weld can take a minute of
# solver time with nothing to say, so this is generous; the planner's own
# terminal event (finished/aborted/paused) is what normally ends the run, and this
# only catches the planner that died without saying so.
PLAN_PROGRESS_IDLE_S = 180.0

_plan_progress_lock = threading.Lock()
_plan_progress: dict = {
    "trace_id": None,   # the run currently on this feed
    "events": [],       # bounded, ascending by seq
    "seen": set(),      # seqs already buffered — TRANSIENT_LOCAL redelivers
    "dropped": 0,       # events aged out of the buffer, so the panel can say so
    "last_s": 0.0,      # when the last event arrived
    "finished": False,  # the run announced its own end
}

# The event kinds that end a run. Anything else means it is still going.
_PLAN_PROGRESS_TERMINAL = ("finished", "aborted", "paused")


def _record_plan_progress(event: dict) -> None:
    """Buffer one plan event from DDS. Never raises into the ROS executor.

    A new ``trace_id`` clears the buffer: two interleaved runs in one feed is
    worse than losing the older one, because the reader cannot tell them apart.
    Duplicate seqs are dropped — ``TRANSIENT_LOCAL`` replays the run's history to
    a late subscriber, and this panel may be that subscriber twice.
    """
    seq = event.get("seq")
    trace_id = event.get("trace_id")
    if not isinstance(seq, int) or not trace_id:
        return
    row = {k: v for k, v in event.items() if k not in ("schema", "cell_id")}
    with _plan_progress_lock:
        if trace_id != _plan_progress["trace_id"]:
            _plan_progress.update({"trace_id": trace_id, "events": [],
                                   "seen": set(), "dropped": 0,
                                   "finished": False})
        if seq in _plan_progress["seen"]:
            return
        _plan_progress["seen"].add(seq)
        events = _plan_progress["events"]
        events.append(row)
        events.sort(key=lambda e: e.get("seq") or 0)
        overflow = len(events) - MAX_PLAN_PROGRESS_EVENTS
        if overflow > 0:
            for gone in events[:overflow]:
                _plan_progress["seen"].discard(gone.get("seq"))
            del events[:overflow]
            _plan_progress["dropped"] += overflow
        _plan_progress["last_s"] = time.time()
        if event.get("kind") in _PLAN_PROGRESS_TERMINAL:
            _plan_progress["finished"] = True


def handle_plan_progress_message(data, warn=None) -> bool:
    """Decode one message off the plan-progress topic and buffer it.

    Module level rather than only the subscription's body, because this is the
    policy and not the plumbing: the planner counts as alive the moment anything
    arrives, anything that is not an event object is ignored, and no publisher —
    however broken — reaches the ROS executor through here. A node cannot be
    built without a middleware, so the policy would otherwise be untestable.

    Returns whether the message was buffered. Never raises.
    """
    _touch("collaborative-planner")
    try:
        event = json.loads(data)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(event, dict):
        return False
    try:
        _record_plan_progress(event)
        return True
    except Exception as exc:  # noqa: BLE001
        if warn is not None:
            warn(f"plan_progress not buffered: {exc}")
        return False


def _plan_progress_since(since: int, limit: int) -> dict:
    """The feed's answer to one poll: what happened after ``since``.

    ``running`` is the planner's own verdict where it gave one and a staleness
    check where it did not — a planner whose container was killed mid-run never
    gets to publish ``aborted``, and a panel that blinks PLANNING for the rest of
    the session over it is the stale-indicator bug this feed is meant to avoid.
    """
    with _plan_progress_lock:
        events = [e for e in _plan_progress["events"]
                  if (e.get("seq") or 0) > since][:max(1, limit)]
        highest = max([e.get("seq") or 0 for e in _plan_progress["events"]],
                      default=since)
        trace_id = _plan_progress["trace_id"]
        finished = _plan_progress["finished"]
        last_s = _plan_progress["last_s"]
        dropped = _plan_progress["dropped"]
    stale = bool(last_s) and (time.time() - last_s) > PLAN_PROGRESS_IDLE_S
    return {"success": True, "source": "dds", "cell_id": CELL_ID,
            "topic": cell_topic("/plan_progress"),
            "trace_id": trace_id, "events": events,
            "next_seq": max(highest, since), "dropped": dropped,
            "running": bool(trace_id) and not finished and not stale}

# ─── Agent topology (static) ──────────────────────────────────────────────────
# Each entry describes one deployed agent/container visible in the diagram.
# "layer" groups them visually; "topics" maps incoming topic suffixes to
# a human-readable action label shown in the tooltip.

AGENT_TOPOLOGY = [
    {
        "id": "rai-orchestrator",
        "label": "RAI Orchestrator",
        "icon": "🤖",
        "layer": "orchestration",
        "description": "LLM-driven task allocation. Translates natural language goals into robot subtasks.",
        "topics_out": {
            "/plan_motion/subtask":    "Dispatch subtask to robots",
        },
    },
    {
        "id": "collaborative-planner",
        "label": "Collaborative Planner",
        "icon": "📐",
        "layer": "planning",
        "description": "Pinocchio IK + Space-Time RRT. Computes collision-free multi-robot trajectories.",
        "topics_in":  {
            "/collaborative_plan_request": "Receive motion request",
            "/collision_alerts":           "React to collision alert",
        },
        "topics_out": {
            "/plan_progress":              "Publish live planning progress",
        },
    },
    {
        "id": "collision-monitor",
        "label": "Collision Monitor",
        "icon": "🛡",
        "layer": "safety",
        "description": "Pinocchio pairwise distance checks at 250Hz. Emits alerts before contact occurs.",
        "topics_in":  {},
        "topics_out": {
            "/collision_alerts":      "Publish predictive collision alert",
            "/collision_alerts_json": "Publish JSON alert for web panel",
        },
    },
    {
        "id": "cell-controller",
        "label": "Cell Controller",
        "icon": "🏭",
        "layer": "cell",
        "description": "Executes behaviour trees, supervises inter-arm clearance (collision monitor), records runs.",
        "topics_in":  {
            "/weld_completed":   "Record weld completion",
            "/collision_alerts": "Relay collision to web panel",
        },
        "topics_out": {
            "/weld_points_viz":              "Publish weld point markers",
            "/structure_viz":               "Publish workpiece geometry",
            "/collision_alerts_json":       "Relay alert as JSON",
        },
    },
    # NOTE: the per-robot "robot-agent-<rid>" execution nodes are NOT listed here.
    # They are generated from robots.yaml by _robot_agent_nodes() and spliced in by
    # agent_topology(), so a cell with 1, 2, 3 or more arms renders all of them.
    {
        "id": "rviz-web",
        "label": "RViz · sim-controller",
        "icon": "👁",
        "layer": "visualization",
        "description": "RViz2 visualization + sim_controller per robot (TF pipeline + FollowJointTrajectory).",
        "topics_in":  {
            "/viz/{ns}/robot_description":  "URDF fetched over MCP (get_robot_description) → robot_state_publisher → /tf",
            "/{ns}/{rid}/joint_commands":     "Execute trajectory (Pinocchio interpolation)",
        },
        "topics_out": {
            "/{ns}/{rid}/joint_states": "Publish Pinocchio joint state at 250Hz",
            "/tf_static":               "Publish fixed transforms (world→base_link)",
            "/tf":                      "Publish revolute transforms via RSP",
        },
    },
    {
        "id": "web-panel",
        "label": "Web Control Panel",
        "icon": "🖥",
        "layer": "visualization",
        "description": "This panel. Subscribes to all cell topics for monitoring.",
        "topics_in":  {
            "/collision_alerts_json":  "Show collision alerts",
        },
        "topics_out": {
        },
    },
]


def _robot_agent_nodes() -> list:
    """One execution node per robot in robots.yaml — any robot count.

    Previously robot1/robot2 were hardcoded here, so a 3-arm cell silently showed
    only two arms. Derived from _discover_robots() so the panel matches the cell.
    """
    return [
        {
            "id": f"robot-agent-{rid}",
            "label": f"Robot Agent · {rid}",
            "icon": "🦾",
            "layer": "execution",
            "description": "Executes trajectories via FollowJointTrajectory.",
            "topics_in": {
                "/{ns}/{rid}/dispatch_subtask":  "Receive subtask from dispatch coordinator",
                "/{ns}/{rid}/status_request":    "Respond to status poll",
                "/collision_alerts":             "Emergency stop on collision",
            },
            "topics_out": {
                "/{ns}/{rid}/robot_status":
                    "Publish execution status (RECEIVED→EXECUTING→COMPLETED)",
            },
        }
        for _ns, rid in _discover_robots()
    ]


def agent_topology() -> list:
    """AGENT_TOPOLOGY with the per-robot execution nodes spliced in.

    Robot nodes are inserted before the visualization layer so the rendered order
    stays orchestration → planning → execution → visualization.
    """
    nodes = list(AGENT_TOPOLOGY)
    idx = next((i for i, n in enumerate(nodes) if n.get("layer") == "visualization"),
               len(nodes))
    return nodes[:idx] + _robot_agent_nodes() + nodes[idx:]


def robot_graph_js() -> str:
    """JS literals for the topology graph: per-robot node positions + edges.

    Arms are spread evenly across the execution row, so N arms lay out without
    overlapping instead of assuming the two fixed x-positions 300 and 700.
    """
    import json as _json
    rids = [rid for _ns, rid in _discover_robots()]
    n = max(len(rids), 1)
    positions, edges = {}, []
    for i, rid in enumerate(rids):
        node = f"robot-agent-{rid}"
        # Spread across x∈[200,800] — centred when n == 1.
        cx = 500 if n == 1 else int(200 + 600 * i / (n - 1))
        positions[node] = {"cx": cx, "cy": 495}
        edges += [
            ["rai-orchestrator", node, "Dispatch task"],
            ["collaborative-planner", node, "Trajectory"],
            ["collision-monitor", node, "Safety alert"],
            ["cell-controller", node, "URDF"],
            [node, "rviz-web", "Joint states"],
            [node, "rai-orchestrator", "Status"],
        ]
    return (f"const ROBOT_NODE_POS = {_json.dumps(positions)};\n"
            f"const ROBOT_EDGES = {_json.dumps(edges)};\n"
            f"const ROBOT_IDS = {_json.dumps(rids)};\n")

# ─── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Arm Control Panel — 3-col</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;overflow:hidden;height:100vh}
.shell{display:grid;grid-template-rows:48px 1fr;height:100vh}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:0 20px;display:flex;align-items:center;gap:14px}
.header h1{font-size:15px;color:#58a6ff;font-weight:700}
.cell-badge{background:#21262d;border:1px solid #30363d;border-radius:12px;padding:2px 10px;font-size:11px;color:#8b949e}
.sdot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}
.dot-idle{background:#6e7681}.dot-running{background:#3fb950;box-shadow:0 0 8px #3fb95099}.dot-error{background:#f85149}
/* The states the panel's own status field actually holds. "planning" used to
   arrive here as dot-planning, which no rule matched — so the one state that
   lasts minutes and needs saying was drawn in the same grey as idle. */
.dot-planning{background:#d29922;box-shadow:0 0 8px #d29922aa}
.dot-homing{background:#58a6ff;box-shadow:0 0 8px #58a6ffaa}
.dot-estopped{background:#f85149;box-shadow:0 0 8px #f85149aa}
/* One animation for everything that has to be noticed from across the cell.
   Opacity only: no layout, so a blinking element never moves its neighbours. */
@keyframes blink-fade{0%,100%{opacity:1}50%{opacity:.18}}
.blink{animation:blink-fade 1.1s ease-in-out infinite}
.busy-msg{font-size:11px;font-weight:700;letter-spacing:.4px}
/* 3-column resizable layout */
.main{display:grid;grid-template-columns:var(--col1,35%) 4px var(--col2,35%) 4px 1fr;overflow:hidden;height:100%}
.divider{background:#21262d;cursor:col-resize;transition:background 0.2s;user-select:none}
.divider:hover{background:#58a6ff}
.vdivider{background:#21262d;cursor:row-resize;height:6px;transition:background 0.2s;user-select:none}
.vdivider:hover{background:#58a6ff}
/* left column — live simulation (noVNC → RViz) */
#canvas-wrap{position:relative;overflow:hidden;background:#0d1117;height:100%;
  display:grid;grid-template-rows:auto 1fr}
#vnc-bar{display:flex;align-items:center;gap:8px;padding:6px 10px;background:#161b22;
  border-bottom:1px solid #30363d;font-size:10px;color:#8b949e;
  text-transform:uppercase;letter-spacing:1px}
#vnc-bar a,#vnc-bar button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  border-radius:5px;padding:2px 8px;font-size:10px;cursor:pointer;text-decoration:none;
  text-transform:none;letter-spacing:0}
#vnc-frame{width:100%;height:100%;border:0;background:#000;display:block}
#vnc-down{display:none;padding:20px;font-size:11px;color:#8b949e;line-height:1.7}
/* agent diagram — moved into its own tab, the left column shows the sim.
   A fixed height, not 100%: inside a tab pane that grid-sizes to its content,
   a percentage height resolves against the content and collapses to nothing. */
#agents-wrap{position:relative;width:100%;height:460px}
#arch-svg{width:100%;height:100%}
.layer-title{font-size:10px;fill:#6e7681;font-family:-apple-system,sans-serif;text-transform:uppercase;letter-spacing:1px}
.actor-ring{fill:#161b22;stroke:#30363d;stroke-width:2;cursor:pointer}
.actor-icon{font-size:18px;dominant-baseline:central;text-anchor:middle;pointer-events:none}
.actor-label{font-size:10px;fill:#c9d1d9;text-anchor:middle;font-family:-apple-system,sans-serif;font-weight:600;pointer-events:none}
.flow-arrow{stroke:#30363d;stroke-width:1.5;fill:none;marker-end:url(#arr)}
.flow-label{font-size:9px;fill:#484f58;font-family:-apple-system,sans-serif;text-anchor:middle}
@keyframes ring-pulse{
  0%{stroke:#f0883e;filter:drop-shadow(0 0 12px #f0883e)}
  40%{stroke:#f0883e;filter:drop-shadow(0 0 8px #f0883e88)}
  100%{stroke:#30363d;filter:none}
}
#tip{position:fixed;background:#1c2128;border:1px solid #30363d;border-radius:8px;
  padding:10px 13px;font-size:11px;line-height:1.7;max-width:260px;pointer-events:none;
  display:none;z-index:400;color:#c9d1d9;box-shadow:0 8px 24px #0008}
#tip h4{color:#58a6ff;margin-bottom:5px;font-size:12px}
#tip .tin{color:#3fb950;font-weight:600;font-size:10px;margin-bottom:2px}
#tip .tout{color:#58a6ff;font-weight:600;font-size:10px;margin:5px 0 2px}
#tip .trow{display:flex;gap:5px;margin-bottom:2px;font-size:10.5px}
#tip .tarrow{flex-shrink:0;color:#6e7681;width:10px}
/* center column */
.center-panel{display:grid;grid-template-rows:auto 6px 1fr 6px auto;overflow:hidden;border-left:1px solid #30363d}
.controls{padding:12px;border-bottom:1px solid #30363d}
.ctrl-title{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.btn-row{display:flex;gap:6px;flex-wrap:wrap}
.btn{padding:7px 13px;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;transition:opacity .15s,transform .1s}
.btn:hover{opacity:.85;transform:translateY(-1px)}.btn:active{transform:translateY(0)}
.btn-ag{background:#1a7f37;color:#fff}.btn-run{background:#1f6feb;color:#fff}
/* Planning, in flight. Colour AND a changed label AND a blink: the same green
   button before and after the click is why a run that takes minutes got started
   twice, and colour on its own is not a difference every operator sees. */
.btn-ag.btn-busy{background:#9e6a03;color:#fff}
.btn-stop{background:#6e4a00;color:#fff}.btn-reset{background:#9a6700;color:#fff}
.btn-st{background:#21262d;color:#c9d1d9;border:1px solid #30363d}
/* The emergency stop must not look like its neighbours: it is the only button
   that halts an arm, and it latches. Red is reserved for it alone — Stop is an
   amber "end the activity", which is what it actually does. */
.btn-estop{flex:1;background:#b62324;color:#fff;font-size:13px;letter-spacing:1px;
  border:2px solid #f85149;text-transform:uppercase}
.btn-estop:hover{background:#da3633;opacity:1}
.task-sel{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;
  padding:6px 8px;font-size:11.5px;max-width:200px}
.log-wrap{overflow-y:auto;padding:12px;height:100%}
.log-title{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
#log-container{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:10.5px}
.le{padding:3px 0;border-bottom:1px solid #0d1117;line-height:1.5}
.le-time{color:#6e7681;margin-right:4px;font-size:9px}
.le-actor{background:#21262d;border-radius:3px;padding:0 4px;font-size:9px;margin-right:4px;color:#8b949e}
.le-info .le-msg{color:#c9d1d9}.le-plan .le-msg{color:#58a6ff}
.le-warn .le-msg{color:#e3b341}.le-error .le-msg,.le-collision .le-msg{color:#f85149;font-weight:700}
@keyframes le-flash{0%{border-left:3px solid #f0883e;background:#f0883e11}100%{border-left:3px solid transparent;background:transparent}}
.le-new{animation:le-flash 2s ease-out forwards}
.le-dds{border-left:3px solid #3fb95066}
.dds-badge{display:inline-block;background:#1a3a2a;border-radius:3px;padding:0 4px;font-size:9px;color:#3fb950;margin-right:4px;font-weight:700}
.cmd-feed{padding:10px 12px;overflow-y:auto;height:100%}
.cmd-title{font-size:10px;color:#e3b341;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.cmd-entry{font-size:11px;padding:3px 0;border-bottom:1px solid #161b22;line-height:1.5}
.cmd-time{color:#6e7681;font-size:9px;margin-right:4px}
.cmd-badge{display:inline-block;border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700;margin-right:5px}
.cb-plan{background:#1f3a5f;color:#58a6ff}.cb-move{background:#1a3a2a;color:#3fb950}
.cb-safe{background:#3a1a1a;color:#f85149}.cb-cell{background:#2a2a1a;color:#e3b341}
/* behaviour-tree view */
#bt-canvas{overflow:auto;background:#0d1117;border:1px solid #30363d;border-radius:6px}
.bt-box{cursor:pointer}
.bt-box rect{stroke-width:1.5}
.bt-box:hover rect{filter:brightness(1.35)}
.bt-type{font-size:10px;font-weight:700;text-anchor:middle;pointer-events:none}
.bt-name{font-size:9px;fill:#8b949e;text-anchor:middle;pointer-events:none}
.bt-edge{stroke:#30363d;stroke-width:1.5;fill:none}
/* Sibling-order arrow: drawn between children a control node runs IN SEQUENCE. */
.bt-order{stroke:#8b949e;stroke-width:1.2;fill:none;opacity:.85}
.bt-slot{font-size:9px;font-weight:700;text-anchor:middle;pointer-events:none}
.bt-sel rect{stroke:#f0883e !important;stroke-width:2.5}
.bt-kv{display:grid;grid-template-columns:auto 1fr;gap:2px 8px;font-size:10px;
  font-family:ui-monospace,monospace}
.bt-kv b{color:#8b949e;font-weight:600}
</style>
</head>
<body data-cell="{{ cell_id }}">
<div class="shell">
  <div class="header">
    <h1>&#x1F916; Multi-Arm Welding Cell</h1>
    <span class="cell-badge" id="cell-badge">&#x2026;</span>
    <!-- #status-msg is the blinking line: empty when the cell is idle, so the bar
         is quiet unless something is actually happening. renderRunState() owns
         all three of these. -->
    <span style="margin-left:auto;display:flex;align-items:center;gap:8px">
      <span class="busy-msg" id="status-msg"></span>
      <span class="sdot dot-idle" id="status-dot"></span>
      <span id="status-text" style="font-size:12px;font-weight:600">IDLE</span>
    </span>
  </div>
  <!-- 3-column resizable: left=live sim (noVNC/RViz) | center=controls+log+events | right=tabs -->
  <div class="main" id="main-grid">
    <!-- LEFT: the simulation itself — RViz inside the rviz-web container, over
         noVNC. Served by that container (websockify on :80, published as
         {{ novnc_port }}), NOT by this panel, so the src is built client-side
         from the hostname the operator actually reached the dashboard on:
         hardcoding "localhost" breaks the moment the panel is opened remotely. -->
    <div id="canvas-wrap">
      <div id="vnc-bar">
        <span>&#x1F5A5; Live Simulation &middot; RViz</span>
        <span id="vnc-state" style="text-transform:none;letter-spacing:0;color:#6e7681">connecting&#x2026;</span>
        <span style="margin-left:auto;display:flex;gap:6px">
          <button onclick="reloadVnc()" title="Reconnect the viewer">&#x21BB; Reconnect</button>
          <a id="vnc-link" target="_blank" rel="noopener" title="Open the viewer full-screen">&#x2197; Full</a>
        </span>
      </div>
      <iframe id="vnc-frame" title="RViz simulation view (noVNC)" allowfullscreen></iframe>
      <div id="vnc-down"></div>
    </div>
    <!-- DIVIDER 1 -->
    <div class="divider" id="div1"></div>
    <!-- CENTER: controls + log + events -->
    <div class="center-panel">
      <div class="controls">
        <div class="ctrl-title">Controls</div>
        <!-- Planning takes the task to plan from the selector next to it: this
             cell has several programs, and "plan the first .json in the
             directory" was a hidden choice the operator could not see or
             change. The list comes from GET /api/a2a/tasks (the orchestrator's
             own answer), so it cannot disagree with what is runnable. -->
        <div class="btn-row">
          <button id="btn-planning" class="btn btn-ag" onclick="cmd('planning')"
                  title="Calculate trajectories for the selected task. Plans only — nothing moves.">
            &#x1F9E9; Planning</button>
          <select id="plan-task" class="task-sel"
                  title="Which task program to plan">
            <option value="">loading tasks…</option>
          </select>
          <button class="btn btn-stop" onclick="cmd('stop')"
                  title="Stop what the system is doing (planning or replay). Does NOT halt the arms — a trajectory already sent finishes.">
            &#x23F8; Stop</button>
          <button class="btn btn-reset" onclick="cmd('reset')"
                  title="Stop, then send the arms back to HOME. This MOVES the robots and clears a latched e-stop.">
            &#x1F3E0; Reset</button>
        </div>
        <!-- Emergency stop sits on its own row, full width and unmistakable. It
             is the ONLY button here that halts an arm where it stands: it
             publishes the latched cell e-stop straight from this panel's ROS
             node, so it still works when the orchestrator is the wedged thing.
             Stop and Reset above are software stops of the ACTIVITY. -->
        <div class="btn-row" style="margin-top:6px">
          <button class="btn btn-estop" onclick="cmd('estop')"
                  title="HALT THE ROBOTS NOW. Cancels the motion in progress and latches; Reset (or a new run) clears it.">
            &#x1F6D1; EMERGENCY STOP</button>
        </div>
        <!-- Every command posts to /api/command and used to say nothing back, so
             a rejected one looked exactly like a working one. -->
        <div id="cmd-status" style="font-size:10px;color:#6e7681;min-height:14px;margin-top:7px"></div>
        <!-- Speed and loop: POST /api/speed and /api/loop, both of which this
             panel's JS has always driven — the controls themselves had gone
             missing from the markup, which made updateUI() throw on every poll
             (see the $() note there) and took the log panel down with it. -->
        <div style="display:flex;align-items:center;gap:8px;margin-top:10px">
          <span style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px">Speed</span>
          <input id="speed-slider" type="range" min="0.1" max="5" step="0.1" value="2"
                 oninput="updateSpeed(this.value)" style="flex:1;accent-color:#1f6feb">
          <span id="speed-val" style="font-size:11px;font-weight:700;color:#58a6ff;min-width:34px;text-align:right">2.0&#xD7;</span>
          <button id="speed-toggle" class="btn btn-st" style="padding:4px 9px;font-size:11px"
                  onclick="toggleSpeed()">&#x23EA; Slow</button>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
          <span style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px">Loop replay</span>
          <input id="loop-interval" type="number" value="10" min="1" step="1" title="Seconds between replays"
                 style="width:58px;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:3px 5px;color:#c9d1d9;font-size:11px;outline:none">
          <span style="font-size:10px;color:#6e7681">s</span>
          <button id="loop-btn" class="btn btn-st" style="margin-left:auto;padding:4px 11px;font-size:11px"
                  onclick="toggleLoop()">&#x25B6; Start</button>
        </div>
      </div>
      <div class="vdivider" data-which="top"></div>
      <div class="log-wrap" id="log-wrap">
        <div class="log-title">System Log</div>
        <div id="log-container"></div>
      </div>
      <div class="vdivider" data-which="bottom"></div>
      <div class="cmd-feed">
        <div class="cmd-title">&#x26A1; Live Commands &amp; Events</div>
        <div id="cmd-log"></div>
      </div>
    </div>
    <!-- DIVIDER 2 -->
    <div class="divider" id="div2"></div>
    <!-- RIGHT: tabbed panel (Chat / Operator / Recordings) -->
    <div id="chat-panel" style="display:grid;grid-template-rows:auto auto 1fr auto;overflow:hidden;border-left:1px solid #30363d">
      <!-- Tab bar -->
      <div style="display:flex;border-bottom:1px solid #30363d;background:#161b22">
        <button id="tab-chat" onclick="switchTab('chat')"
          style="flex:1;padding:9px;border:none;background:#1f6feb22;color:#58a6ff;font-size:11px;font-weight:700;cursor:pointer;border-bottom:2px solid #1f6feb">
          &#x1F4AC; Assistant
        </button>
        <button id="tab-op" onclick="switchTab('op')"
          style="flex:1;padding:9px;border:none;background:transparent;color:#8b949e;font-size:11px;font-weight:600;cursor:pointer;border-bottom:2px solid transparent">
          &#x1F527; Operator
        </button>
        <button id="tab-tree" onclick="switchTab('tree')"
          style="flex:1;padding:9px;border:none;background:transparent;color:#8b949e;font-size:11px;font-weight:600;cursor:pointer;border-bottom:2px solid transparent">
          &#x1F333; BTree
        </button>
        <button id="tab-plan" onclick="switchTab('plan')"
          style="flex:1;padding:9px;border:none;background:transparent;color:#8b949e;font-size:11px;font-weight:600;cursor:pointer;border-bottom:2px solid transparent">
          &#x1F4DD; Log
        </button>
        <!-- The Agents tab carries the overall status too: the run it reports on
             takes minutes, and a reader sitting on another tab could not tell a
             live run from an idle cell without switching to look. The dot draws
             its own colour rather than inheriting the button's, which switchTab
             sets inline and would win. -->
        <button id="tab-agents" onclick="switchTab('agents')"
          style="flex:1;padding:9px;border:none;background:transparent;color:#8b949e;font-size:11px;font-weight:600;cursor:pointer;border-bottom:2px solid transparent">
          <span class="sdot dot-idle" id="tab-agents-dot"></span>&#x1F578; Agents
        </button>
      </div>
      <!-- Chat tab contents (replaces old header) -->
      <div id="pane-chat" style="display:contents">
      <div style="padding:10px 12px;border-bottom:1px solid #30363d;font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;display:none">
        &#x1F4AC; Cell Assistant
      </div>
      <div id="chat-messages" style="overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;height:100%"></div>
      <div style="padding:8px 10px;border-top:1px solid #30363d">
        <div id="chat-chips" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px"></div>
        <div style="display:flex;gap:6px">
          <input id="chat-input" type="text" placeholder="Ask about status, trajectory, weld points..."
                 style="flex:1;background:#161b22;border:1px solid #30363d;border-radius:6px;padding:7px 10px;color:#c9d1d9;font-size:12px;outline:none"
                 onkeydown="if(event.key==='Enter')sendChat()">
          <button onclick="sendChat()" style="background:#1f6feb;color:#fff;border:none;border-radius:6px;padding:7px 14px;font-size:12px;font-weight:600;cursor:pointer">Send</button>
        </div>
      </div>
      </div><!-- /pane-chat -->

      <!-- OPERATOR TAB -->
      <div id="pane-op" style="display:none;overflow-y:auto;padding:12px;flex-direction:column;gap:12px">
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px">
          <div style="font-size:11px;font-weight:700;color:#e3b341;margin-bottom:10px">&#x1F916; Move Robot to Position</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
            <div><label style="font-size:10px;color:#6e7681">Robot</label>
              <select id="op-robot" style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:12px;outline:none;margin-top:3px">
                {{ robot_options|safe }}
              </select></div>
            <div><label style="font-size:10px;color:#6e7681">Duration (s)</label>
              <input id="op-duration" type="number" value="3.0" min="0.1" step="0.1"
                style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:12px;outline:none;margin-top:3px"></div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px">
            <div><label style="font-size:10px;color:#6e7681">X (m)</label>
              <input id="op-x" type="number" value="-0.982" step="0.01" style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:12px;outline:none;margin-top:3px"></div>
            <div><label style="font-size:10px;color:#6e7681">Y (m)</label>
              <input id="op-y" type="number" value="-1.239" step="0.01" style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:12px;outline:none;margin-top:3px"></div>
            <div><label style="font-size:10px;color:#6e7681">Z (m)</label>
              <input id="op-z" type="number" value="1.1" step="0.01" style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:12px;outline:none;margin-top:3px"></div>
          </div>
          <div style="display:flex;gap:8px;margin-bottom:6px">
            <button onclick="opMove()" style="flex:1;background:#1a7f37;color:#fff;border:none;border-radius:6px;padding:8px;font-size:12px;font-weight:700;cursor:pointer">&#x25B6; Send Move</button>
            <button onclick="opMoveHome()" style="flex:1;background:#9a6700;color:#fff;border:none;border-radius:6px;padding:8px;font-size:12px;font-weight:700;cursor:pointer">&#x23F9; Home</button>
          </div>
          <div id="op-move-status" style="font-size:10px;color:#8b949e;min-height:16px"></div>
        </div>
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px">
          <div style="font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:8px">&#x1F4CB; Submit Task JSON</div>
          <textarea id="op-task-json" rows="7" placeholder='{"program":"my_task","operations":[...]}'
            style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:8px;color:#c9d1d9;font-size:10.5px;font-family:monospace;outline:none;resize:vertical"></textarea>
          <div style="display:flex;gap:8px;margin-top:8px">
            <button onclick="opLoadTemplate()" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:7px 12px;font-size:11px;cursor:pointer">&#x1F4C2; Template</button>
            <button onclick="opTaskWizard()" title="Answer a few questions; the wizard writes the task file. It runs nothing." style="background:#21262d;color:#d2a8ff;border:1px solid #6e40c9;border-radius:6px;padding:7px 12px;font-size:11px;cursor:pointer">&#x1F9ED; Wizard</button>
            <button onclick="opSubmitTask()" style="flex:1;background:#1f6feb;color:#fff;border:none;border-radius:6px;padding:8px;font-size:12px;font-weight:700;cursor:pointer">&#x1F680; Submit &amp; Run</button>
          </div>
          <div id="op-task-status" style="font-size:10px;color:#8b949e;margin-top:6px;min-height:16px"></div>
        </div>
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
            <div style="font-size:11px;font-weight:700;color:#a371f7">&#x1F9E0; LLM Reasoning</div>
            <button onclick="opLoadReasoning()" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:3px 9px;font-size:10px;cursor:pointer">&#x21BB; Refresh</button>
          </div>
          <div id="op-reasoning" style="font-family:monospace;font-size:9.5px;color:#8b949e;max-height:160px;overflow-y:auto;background:#0d1117;border-radius:5px;padding:6px;white-space:pre-wrap">Click Refresh to load...</div>
        </div>
      </div>

      <!-- BEHAVIOUR TREE TAB — the tree, and the button that executes it.
           Replay lives HERE and not in a recordings browser: /a2a/replay takes a
           tree_id and runs a *calculated plan*, so the set of replayable things is
           the set of plans — which is what this selector lists. The old Rec tab
           listed recordings instead, i.e. only the plans that had already run at
           least once, and so quietly hid every plan calculated but never executed.
           Shape + ports only: see GET /api/plans/<tree_id>. -->
      <div id="pane-tree" style="display:none;overflow:auto;padding:12px;flex-direction:column;gap:8px">
        <div style="display:flex;gap:8px;align-items:center">
          <span style="font-size:11px;font-weight:700;color:#3fb950">Behaviour Tree</span>
          <button onclick="loadPlanList()" style="margin-left:auto;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:3px 9px;font-size:10px;cursor:pointer">&#x21BB; Refresh</button>
        </div>
        <select id="bt-select" onchange="loadTree(this.value)"
          style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:11px;outline:none">
          <option value="">Loading&#x2026;</option>
        </select>
        <!-- REPLAY the tree shown above. This MOVES THE ARMS, which is why it says
             so and why it stays on this tab: the reader is looking at the tree it is
             about to execute, and being thrown to another pane on press is how an
             operator loses track of what they just started. -->
        <div style="display:flex;gap:8px;align-items:center">
          <select id="bt-speed" style="background:#161b22;border:1px solid #30363d;border-radius:5px;padding:4px 8px;color:#c9d1d9;font-size:11px;outline:none" title="Cell velocity scale applied before the replay starts">
            <option value="0.25">0.25x</option><option value="0.5">0.5x</option>
            <option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option>
          </select>
          <button id="bt-replay" onclick="btReplay()" style="flex:1;background:#9a6700;color:#fff;border:none;border-radius:6px;padding:7px;font-size:12px;font-weight:700;cursor:pointer" title="Execute this tree on the real cell — the arms will move">&#x25B6; Replay this tree</button>
        </div>
        <div id="bt-replay-status" style="font-size:10px;color:#6e7681;min-height:14px"></div>
        <div id="bt-meta" style="font-size:10px;color:#6e7681;min-height:14px"></div>
        <!-- PERFORMANCE of the selected plan — the PUBLISHED METRIC, and nothing
             else. Above the drawing on purpose: the tree says what the plan does,
             this says whether it is worth executing, and a valid collision-free
             plan that wanders four times further than it needs to looks perfectly
             good in the drawing. That defect is why these numbers exist.
             One number, its verdict, and how many findings sit behind it. The
             thirty-odd numbers that justify the score open in a window of their
             own (#bt-kpi-modal) on click: this pane's job is the tree, and a
             report inline pushed the drawing off the screen on every load — a
             reader looking for the tree had to scroll past a page of metrics they
             had not asked for.
             All of it is the planner's own report, read over MCP (GET
             /api/plans/<tree_id>/quality → get_plan_quality); the browser
             computes nothing. -->
        <div id="bt-kpi" style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px;display:flex;flex-direction:column;gap:8px">
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:11px;font-weight:700;color:#3fb950">&#x1F4C8; Performance</span>
            <span id="bt-kpi-source" style="font-size:9px;color:#6e7681"></span>
            <button id="bt-kpi-refresh" onclick="kpiRefresh()" style="margin-left:auto;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:3px 9px;font-size:10px;cursor:pointer">&#x21BB; Refresh</button>
          </div>
          <div id="bt-kpi-status" style="font-size:10px;color:#6e7681;min-height:14px">Select a plan to see its KPIs.</div>
          <!-- Clickable, and it says so: a number that hides a report behind it
               with no affordance is a report nobody opens. -->
          <div id="bt-kpi-score" onclick="kpiOpen()" style="display:none;align-items:center;gap:10px;cursor:pointer" title="Open the full performance report"></div>
          <button id="bt-kpi-open" onclick="kpiOpen()" style="display:none;width:100%;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:5px;font-size:10px;cursor:pointer;text-align:left"></button>
        </div>
        <div id="bt-canvas"><svg id="bt-svg" xmlns="http://www.w3.org/2000/svg"></svg></div>
        <!-- The notation, because "1 → 2 → 3" and "∥" are the difference between
             welding in turn and welding at the same time. -->
        <div style="font-size:9px;color:#6e7681;display:flex;gap:12px;flex-wrap:wrap">
          <span><b style="color:#c9d1d9">1 &#x2192; 2</b> runs in order (Sequence / Fallback)</span>
          <span><b style="color:#c297ff">&#x2225;</b> runs at the same time (Parallel — dashed edges)</span>
        </div>
        <div id="bt-detail" style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:9px;font-size:10px;color:#6e7681">
          Click a node to see its ports.
        </div>
        <!-- Step-through: what is INSIDE the selected node. The tree above ends at
             a leaf; the motion a leaf runs is split into its phases here, and each
             phase into its waypoints. Every play button previews on the ghost arm
             in RViz and commands nothing (POST /api/preview). -->
        <div id="bt-steps" style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:9px;font-size:10px;color:#6e7681"></div>
        <div id="bt-preview-status" style="font-size:10px;color:#6e7681;min-height:14px"></div>
      </div>

      <!-- PLANNING LOG TAB — one planning run's decisions, navigable.
           Two views of one history: the POINTS column is the task's welds, each
           with the verdict it ended on; the DECISIONS column is the tool calls in
           order, each with the reasoning that preceded it. Clicking a point
           narrows the decisions to that weld, which is the question people
           actually ask ("what did it decide about W12, and in what order?").
           Served by GET /api/a2a/plan_trace/<trace_id>; nothing is recomputed
           here, the planner recorded it as it ran. -->
      <div id="pane-plan" style="display:none;overflow:auto;padding:12px;flex-direction:column;gap:8px">
        <div style="display:flex;gap:8px;align-items:center">
          <span style="font-size:11px;font-weight:700;color:#a371f7">&#x1F4DD; Log</span>
          <button onclick="planLoadHistory()" style="margin-left:auto;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:3px 9px;font-size:10px;cursor:pointer">&#x21BB; Refresh</button>
        </div>
        <select id="plan-run-select" onchange="planLoad(this.value)"
          style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:11px;outline:none">
          <option value="">Loading&#x2026;</option>
        </select>
        <div id="plan-summary" style="font-size:10px;color:#6e7681;min-height:14px;line-height:1.5"></div>
        <div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:8px;align-items:start">
          <div style="min-width:0">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">
              <span style="font-size:10px;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:.5px">Points</span>
              <span id="plan-point-count" style="font-size:9px;color:#6e7681"></span>
              <button onclick="planSelectPoint(null)" style="margin-left:auto;background:#0d1117;color:#8b949e;border:1px solid #30363d;border-radius:4px;padding:2px 6px;font-size:9px;cursor:pointer">All</button>
            </div>
            <div id="plan-points" style="display:flex;flex-direction:column;gap:4px;max-height:300px;overflow-y:auto"></div>
          </div>
          <div style="min-width:0">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">
              <span style="font-size:10px;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:.5px">Decisions</span>
              <span id="plan-step-count" style="font-size:9px;color:#6e7681"></span>
              <button onclick="planStep(-1)" style="margin-left:auto;background:#0d1117;color:#8b949e;border:1px solid #30363d;border-radius:4px;padding:2px 7px;font-size:9px;cursor:pointer">&#x25C0;</button>
              <button onclick="planStep(1)" style="background:#0d1117;color:#8b949e;border:1px solid #30363d;border-radius:4px;padding:2px 7px;font-size:9px;cursor:pointer">&#x25B6;</button>
            </div>
            <div id="plan-steps" style="display:flex;flex-direction:column;gap:4px;max-height:300px;overflow-y:auto"></div>
          </div>
        </div>
        <!-- ONE WELD'S STORY. Selecting a point above fills this in: whether it was
             allocated to a robot and made it into the tree, every failed attempt in
             order with the error that ended it, and the reasoning the model gave for
             the attempt that finally worked. That last part is the reason to keep a
             log at all — "W12 succeeded" is not useful, "W12 succeeded once the
             standoff came down to 0.07 m because the fixture clamp was in the
             approach cone" is. -->
        <div id="plan-point-detail" style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:9px;font-size:10px;color:#6e7681"></div>
        <div id="plan-detail" style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:9px;font-size:10px;color:#6e7681">
          Pick a run above, then a decision, to see its reasoning, arguments and result.
        </div>
      </div>

      <!-- AGENT TOPOLOGY TAB — the diagram that used to occupy the left column,
           which now shows the simulation itself. -->
      <div id="pane-agents" style="display:none;overflow:auto;flex-direction:column">
        <!-- The overall status in words, above the schema: the diagram says who
             the agents are, this says what the cell as a whole is doing. Written
             by renderRunState() from the same state as the header bar, so the two
             can never disagree. -->
        <div id="agents-overall" class="busy-msg"
             style="padding:8px 12px;border-bottom:1px solid #30363d;color:#6e7681">
          Overall status: IDLE
        </div>
        <div id="agents-wrap">
          <svg id="arch-svg" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="#484f58"/>
              </marker>
            </defs>
            <g id="layers-g"></g><g id="arrows-g"></g><g id="actors-g"></g>
          </svg>
        </div>

        <!-- LIVE PLANNING ACTIVITY — under the schema, because it says what the
             agents in the diagram above are doing right now: this weld started,
             that one validated, the planner shrank a standoff and retried. Read
             off the planner's own DDS topic via GET /api/plan_progress, falling
             back to the orchestrator's durable stream at GET /api/a2a/plan_events;
             both are polled with ?since=, so following a fifteen-minute run never
             means re-reading it.

             Pause is here and not next to Stop on purpose: Stop discards the run,
             Pause keeps every solved trajectory and the model's conversation so
             Resume continues at the next unplanned weld. That is the answer to an
             expiring token — the failure that used to cost a whole run. -->
        <div id="live-wrap" style="padding:10px 12px 14px;display:flex;flex-direction:column;gap:7px;border-top:1px solid #30363d">
          <div style="display:flex;gap:7px;align-items:center;flex-wrap:wrap">
            <span style="font-size:11px;font-weight:700;color:#3fb950">&#x25CF; Live Planning Activity</span>
            <span id="live-run" style="font-size:9px;color:#6e7681"></span>
            <button id="live-follow" onclick="liveToggleFollow()" style="margin-left:auto;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:3px 9px;font-size:10px;cursor:pointer">&#x23F8; Follow: on</button>
            <button onclick="livePause()" style="background:#21262d;color:#d29922;border:1px solid #4d3800;border-radius:5px;padding:3px 9px;font-size:10px;cursor:pointer" title="Stop at the next step and keep everything computed so far">&#x23F8; Pause planning</button>
            <button onclick="liveResume()" style="background:#21262d;color:#3fb950;border:1px solid #124d24;border-radius:5px;padding:3px 9px;font-size:10px;cursor:pointer" title="Continue the checkpointed run instead of replanning from the first weld">&#x25B6; Resume</button>
          </div>
          <select id="live-checkpoints" style="width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:5px;color:#c9d1d9;font-size:10px;outline:none">
            <option value="">Resumable runs&#x2026;</option>
          </select>
          <div id="live-status" style="font-size:10px;color:#6e7681;min-height:14px"></div>
          <div id="live-feed" style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;font-size:10px;color:#6e7681;max-height:300px;overflow-y:auto;display:flex;flex-direction:column;gap:3px">
            Waiting for a planning run&#x2026;
          </div>
        </div>
      </div>

      <div id="tip"></div>
    </div><!-- /chat-panel -->
  </div>
</div>

<!-- THE FULL PERFORMANCE REPORT, in a window of its own.
     Opened from the Performance card's score. Outside every pane on purpose: it is
     position:fixed and a fixed child of a resizable, scrolling pane is clipped by
     it, which is how a report becomes a sliver of a report. It holds the whole of
     what the planner returned — the four weighted components, the numbers behind
     them, every finding rather than the first few, the worst legs, and the safety
     line — because the point of moving it out of the pane was to stop having to
     choose what to leave out.
     Clearances live at the BOTTOM, under their own heading and outside the score
     block: reported, never scored. A score is a thing people optimise, and a
     clearance rendered beside one reads as a KPI to push. -->
<div id="bt-kpi-modal" onclick="kpiBackdrop(event)" style="display:none;position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(1,4,9,.74);z-index:60;align-items:center;justify-content:center;padding:24px">
  <div id="bt-kpi-sheet" style="background:#0d1117;border:1px solid #30363d;border-radius:10px;box-shadow:0 18px 48px rgba(1,4,9,.85);width:760px;max-width:100%;max-height:86vh;display:flex;flex-direction:column;overflow:hidden">
    <div style="display:flex;align-items:center;gap:10px;padding:11px 14px;border-bottom:1px solid #30363d">
      <span style="font-size:12px;font-weight:700;color:#3fb950">&#x1F4C8; Plan performance</span>
      <span id="bt-kpi-modal-tree" style="font-size:9.5px;color:#6e7681;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
      <button onclick="kpiClose()" style="margin-left:auto;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:3px 10px;font-size:10px;cursor:pointer" title="Esc also closes">&#x2715; Close</button>
    </div>
    <div style="padding:13px 14px 16px;overflow-y:auto;display:flex;flex-direction:column;gap:12px">
      <!-- Same order as the report: verdict, then the components it is made of,
           then the numbers behind them, then what to do about it. -->
      <div id="bt-kpi-modal-score" style="display:none;align-items:center;gap:10px"></div>
      <div id="bt-kpi-bars" style="display:none;flex-direction:column;gap:3px"></div>
      <div id="bt-kpi-grid" style="display:none;grid-template-columns:1fr 1fr;gap:6px 14px"></div>
      <div id="bt-kpi-findings" style="display:none;flex-direction:column;gap:4px"></div>
      <div id="bt-kpi-legs" style="display:none;flex-direction:column;gap:3px"></div>
      <div id="bt-kpi-safety" style="display:none;flex-direction:column;gap:3px;border-top:1px solid #21262d;padding-top:9px"></div>
    </div>
  </div>
</div>
<script>
let TOPOLOGY=[], _activeTab='chat';
/* The planning log currently on screen. _planTrace is the whole run as the
   planner recorded it; _planPoint narrows the decision list to one weld (null =
   all of them); _planSeq is the decision being detailed. Kept as module state so
   the two columns and the detail pane stay in step without re-fetching. */
let _planTrace=null, _planPoint=null, _planSeq=null;
/* Every element lookup goes through this. The panel had accumulated
   getElementById calls for controls that no longer existed in the markup
   (#speed-slider among them), and one of those sat in updateUI() BEFORE the log
   render — so every 1.5 s poll threw a TypeError on null and the log panel
   stayed empty while /api/state was answering perfectly.
   tests/test_dashboard_script_loads.py now fails on a reference to an id the
   page does not serve, so this cannot silently come back. */
const $=id=>document.getElementById(id);
const TABS=['chat','op','tree','plan','agents'];
// Server-rendered (data-cell="{{ cell_id }}"), never a literal: the hardcoded
// fallback that used to sit here named a cell that does not exist on disk.
const CELL_ID=document.body.dataset.cell;
$("cell-badge").textContent=CELL_ID;
function switchTab(name){
  _activeTab=name;
  TABS.forEach(t=>{
    const btn=$('tab-'+t),pane=$('pane-'+t);
    if(!btn||!pane)return;
    if(t===name){btn.style.background='#1f6feb22';btn.style.color='#58a6ff';btn.style.borderBottom='2px solid #1f6feb';pane.style.display=t==='chat'?'contents':'flex';pane.style.flexDirection='column';}
    else{btn.style.background='transparent';btn.style.color='#8b949e';btn.style.borderBottom='2px solid transparent';pane.style.display='none';}
  });
  if(name==='tree')loadPlanList();
  // Only on first open: re-fetching on every tab switch would throw away the
  // point/decision the reader had selected, mid-investigation.
  if(name==='plan'&&!_planTrace)planLoadHistory();
  // The diagram scales to the SVG's measured box, which is 0×0 while the pane
  // is display:none — so it has to be (re)drawn once the pane is visible.
  if(name==='agents'){buildDiagram();liveStart();}else liveStop();
}

/* ── LIVE PLANNING ACTIVITY ──────────────────────────────────────────────────
   The feed under the agent schema: what the planner is doing while it does it.
   Same textContent-only rule as the planning log — every string here is model
   output or a tool payload.

   Polling is incremental (?since=<next_seq>) and only runs while the Agents tab
   is open. A 15-weld run emits a few hundred events; re-fetching all of them
   every second to show the last one is the difference between a live feed and a
   busy loop. */
const LIVE_KIND={run_started:'#58a6ff',resumed:'#58a6ff',think:'#8b949e',
  operation_started:'#d29922',operation_planned:'#58a6ff',
  operation_validated:'#3fb950',operation_failed:'#f85149',strategy:'#a371f7',
  tool_started:'#6e7681',tool_failed:'#f85149',paused:'#d29922',
  checkpoint:'#6e7681',aborted:'#f85149',finished:'#3fb950'};
let _liveTrace=null,_liveSeq=0,_liveTimer=null,_liveFollow=true,_liveRunning=false;
/* Which source answered last, for the status line — not a latch: the feed is
   re-chosen every poll, so a middleware that comes back is used again. */
let _liveSource='dds';
const LIVE_POLL_MS=1200;
function liveStart(){
  if(_liveTimer)return;
  livePoll();liveLoadCheckpoints();
  _liveTimer=setInterval(livePoll,LIVE_POLL_MS);
}
/* Stopping the poll also drops what it told the status indicators: nothing is
   watching the run any more, so /api/state — which keeps being polled — becomes
   the only honest source again. Leaving the flag set would blink "PLANNING" in
   the header for the rest of the session. */
function liveStop(){
  if(_liveTimer){clearInterval(_liveTimer);_liveTimer=null;}
  if(_planRunning){_planRunning=false;renderRunState();}
}
function liveToggleFollow(){
  _liveFollow=!_liveFollow;
  const b=$('live-follow');
  if(b)b.textContent=(_liveFollow?'⏸':'▶')+' Follow: '+(_liveFollow?'on':'off');
}
/* Two sources, one renderer.

   DDS first: the planner publishes every event on /<cell>/plan_progress and this
   panel's ROS node buffers them, so a poll is a local read instead of a hop
   through the orchestrator and a shared volume. The orchestrator's durable JSONL
   stream is the fallback, and it is not a lesser one — it is the only source that
   answers a reader who arrives after the run, or a panel with no middleware.

   Only a positive `available` lets the DDS feed serve the view. An empty or
   unrecognised answer means "this panel is not listening", which must never be
   rendered as "no run in progress" — so it falls through to HTTP in the same
   tick. */
function livePoll(){
  fetch('/api/plan_progress?since='+_liveSeq)
    .then(r=>r.json()).then(d=>{
      if(d&&d.available===true){_liveSource='dds';liveConsume(d);return;}
      _liveSource='http';livePollHttp();
    }).catch(()=>{_liveSource='http';livePollHttp();});
}
function livePollHttp(){
  fetch('/api/a2a/plan_events?since='+_liveSeq+(_liveTrace?'&trace_id='+encodeURIComponent(_liveTrace):''))
    .then(r=>r.json()).then(d=>liveConsume(d)).catch(()=>{});
}
function liveConsume(d){
      if(!d||d.error){$('live-status').textContent=d&&d.error?String(d.error):'';return;}
      /* A different run than the one on screen means a new plan was submitted:
         clear, so the feed is never two runs interleaved. */
      if(d.trace_id&&d.trace_id!==_liveTrace){
        _liveTrace=d.trace_id;_liveSeq=0;$('live-feed').textContent='';
        $('live-run').textContent=d.trace_id;
        liveAppend(d.events||[]);
      }else{
        liveAppend(d.events||[]);
      }
      if(typeof d.next_seq==='number')_liveSeq=d.next_seq;
      _liveRunning=!!d.running;
      /* The planner's own verdict on whether a run is live, which is the only
         one that sees a run this panel did not start. */
      _planRunning=_liveRunning;renderRunState();
      $('live-status').textContent=_liveTrace
        ?(_liveRunning?'planning… '+_liveSeq+' events':'not running — '+_liveSeq+' events')
          +(d.dropped?'  ('+d.dropped+' older events not shown)':'')
          +(_liveSource==='dds'?'  · live (DDS)':'')
        :'No planning run has emitted events for this cell yet.';
}
function liveAppend(events){
  const feed=$('live-feed');if(!feed||!events.length)return;
  if(feed.childNodes.length===1&&!feed.firstChild.style)feed.textContent='';
  events.forEach(e=>{
    const row=planEl('div',null,'display:flex;gap:6px;align-items:baseline');
    row.appendChild(planEl('span',(e.at||'').slice(11),'color:#484f58;flex:0 0 auto'));
    row.appendChild(planEl('span',e.kind||'?','color:'+(LIVE_KIND[e.kind]||'#8b949e')+';flex:0 0 auto;font-weight:700'));
    row.appendChild(planEl('span',e.text||'','color:#c9d1d9;min-width:0;word-break:break-word'));
    feed.appendChild(row);
  });
  if(_liveFollow)feed.scrollTop=feed.scrollHeight;
}
function liveLoadCheckpoints(){
  const sel=$('live-checkpoints');if(!sel)return;
  fetch('/api/a2a/checkpoints').then(r=>r.json()).then(d=>{
    const items=(d&&d.checkpoints)||[];
    sel.textContent='';
    if(!items.length){sel.appendChild(planEl('option','No resumable runs'));return;}
    sel.appendChild(planEl('option','Resumable runs…'));
    items.forEach(c=>{
      /* Only offer what can actually be continued. An entry that cannot is not a
         choice, and listing it as one invites a "resume" that would replan. */
      if(!c.resumable)return;
      const o=planEl('option',(c.tree_id||'?')+'  — step '+(c.step||0)
        +', '+(c.trajectories||0)+' trajectories  ('+(c.reason||'')+')');
      o.value=c.tree_id||'';
      sel.appendChild(o);
    });
  }).catch(()=>{});
}
function livePause(){
  $('live-status').textContent='Pausing at the next step…';
  fetch('/api/a2a/plan/pause',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({reason:'paused from the dashboard'})})
    .then(r=>r.json()).then(d=>{
      const paused=(d&&d.paused)||[];
      $('live-status').textContent=paused.length
        ?'Pause requested for '+paused.join(', ')+' — it stops at its next step and keeps everything.'
        :((d&&d.note)||'No planning run is active.');
      liveLoadCheckpoints();
    }).catch(e=>{$('live-status').textContent='Pause failed: '+e;});
}
function liveResume(){
  const sel=$('live-checkpoints');
  const tid=(sel&&sel.value)||_liveTrace;
  if(!tid){$('live-status').textContent='Pick a resumable run first.';return;}
  $('live-status').textContent='Resuming '+tid+'…';
  fetch('/api/a2a/plan/resume',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({tree_id:tid})})
    .then(r=>r.json()).then(d=>{
      if(!d||d.error||d.accepted===false){
        $('live-status').textContent='Cannot resume: '+((d&&(d.error||d.note))||'unknown');
        return;
      }
      $('live-status').textContent='Resumed '+tid+' from step '+(d.resumed_from_step||0)
        +' with '+(d.trajectories_restored||0)+' trajectories restored.';
      _liveTrace=null;_liveSeq=0;livePoll();
    }).catch(e=>{$('live-status').textContent='Resume failed: '+e;});
}

/* ── PLANNING LOG ────────────────────────────────────────────────────────────
   Renders the structured trace the planner wrote as it ran (plan_trace/1). All
   text goes in via textContent, never innerHTML: the strings here are model
   output and tool payloads, i.e. the least trustworthy data on the page, and a
   weld id containing a '<' must not be able to rewrite the panel. */
const PLAN_STATUS={pending:'#6e7681',planned:'#58a6ff',validated:'#3fb950',in_tree:'#a371f7',failed:'#f85149'};
const planColor=s=>PLAN_STATUS[s]||'#6e7681';
function planEl(tag,text,css){const e=document.createElement(tag);if(text!==undefined&&text!==null)e.textContent=String(text);if(css)e.style.cssText=css;return e;}
function planLoadHistory(){
  const sel=$('plan-run-select');if(!sel)return;
  sel.textContent='';sel.appendChild(planEl('option','Loading…'));
  fetch('/api/a2a/plan_traces?limit=50').then(r=>r.json()).then(d=>{
    sel.textContent='';
    const runs=(d&&d.traces)||[];
    if(!runs.length){
      sel.appendChild(planEl('option','No planning traces recorded yet'));
      $('plan-summary').textContent='Traces are written by each planning run — submit a plan and this fills in.';
      $('plan-points').textContent='';$('plan-steps').textContent='';
      $('plan-point-count').textContent='';$('plan-step-count').textContent='';
      $('plan-point-detail').textContent='';
      return;
    }
    runs.forEach(r=>{
      const o=planEl('option',(r.success?'✓ ':'✗ ')+(r.started_at||r.trace_id)
        +'  ·  '+(r.tool_calls||0)+' decisions'
        +(r.error_kind?'  ·  '+r.error_kind:''));
      o.value=r.trace_id||'';sel.appendChild(o);
    });
    planLoad(runs[0].trace_id||'');
  }).catch(e=>{sel.textContent='';sel.appendChild(planEl('option','Failed: '+e));});
}
function planLoad(traceId){
  if(!traceId)return;
  _planPoint=null;_planSeq=null;
  $('plan-summary').textContent='Loading '+traceId+'…';
  fetch('/api/a2a/plan_trace/'+encodeURIComponent(traceId)).then(r=>r.json()).then(d=>{
    if(!d||d.success===false){$('plan-summary').textContent='Error: '+((d&&d.error)||'unknown');return;}
    _planTrace=d;
    const sel=$('plan-run-select');if(sel&&sel.value!==traceId)sel.value=traceId;
    planRenderSummary();planRenderPoints();planRenderSteps();
    planRenderPointDetail();planRenderDetail();
  }).catch(e=>{$('plan-summary').textContent='Failed: '+e;});
}
function planRenderSummary(){
  const box=$('plan-summary');box.textContent='';
  const run=(_planTrace&&_planTrace.run)||{};
  const head=planEl('div',null,'display:flex;gap:6px;flex-wrap:wrap;align-items:center');
  head.appendChild(planEl('span',run.success?'✓ tree assembled':'✗ no tree',
    'font-weight:700;color:'+(run.success?'#3fb950':'#f85149')));
  const bits=[run.trace_id,run.model,run.backend,
    (run.tool_calls||0)+'/'+(run.step_budget||'?')+' steps',
    run.elapsed_s!=null?Number(run.elapsed_s).toFixed(1)+'s':null];
  bits.filter(Boolean).forEach(b=>head.appendChild(planEl('span',b,'color:#8b949e')));
  box.appendChild(head);
  const counts=run.points_by_status||{};
  const row=planEl('div',null,'display:flex;gap:8px;flex-wrap:wrap;margin-top:2px');
  Object.keys(counts).forEach(k=>row.appendChild(planEl('span',k+': '+counts[k],'color:'+planColor(k))));
  box.appendChild(row);
  // The error and the note are the whole point of reading a failed run: the note
  // is what distinguishes "a weld is unreachable" from "the reply was truncated".
  if(run.error)box.appendChild(planEl('div',run.error,'color:#f85149;margin-top:3px'));
  if(run.note)box.appendChild(planEl('div',run.note,'color:#e3b341;margin-top:2px'));
}
function planRenderPoints(){
  const box=$('plan-points');box.textContent='';
  const points=(_planTrace&&_planTrace.points)||[];
  $('plan-point-count').textContent=points.length+(_planPoint?'  ·  filtered: '+_planPoint:'');
  points.forEach(p=>{
    const on=_planPoint===p.operation_id,col=planColor(p.status);
    const row=planEl('div',null,'border:1px solid '+(on?col:'#30363d')+';border-left:3px solid '+col
      +';border-radius:5px;padding:5px 7px;cursor:pointer;background:'+(on?'#1f6feb22':'#0d1117'));
    row.onclick=()=>planSelectPoint(p.operation_id);
    const top=planEl('div',null,'display:flex;gap:6px;align-items:baseline');
    // The allocation mark comes FIRST, because it is the question this list is
    // scanned for: which welds a robot will actually perform. The status word
    // stays beside it — "failed" and "planned but not in the tree" are both
    // "not allocated", and which one it is decides what to do next.
    const alloc=PLAN_ALLOC[p.status]||PLAN_ALLOC.pending;
    top.appendChild(planEl('span',alloc[0].split(' ')[0],'font-size:11px;color:'+alloc[1]));
    top.appendChild(planEl('span',p.operation_id,'font-weight:700;font-size:11px;color:#c9d1d9'));
    top.appendChild(planEl('span',p.status,'font-size:9px;color:'+col));
    const fails=(p.failures||[]).length;
    if(fails)top.appendChild(planEl('span',fails+' failure'+(fails>1?'s':''),
      'font-size:9px;color:#f85149'));
    top.appendChild(planEl('span',(p.attempts||0)+' ×','font-size:9px;color:#6e7681;margin-left:auto'));
    row.appendChild(top);
    const meta=[p.type,p.robot||'no robot',Array.isArray(p.position)?'['+p.position.map(v=>Number(v).toFixed(2)).join(', ')+']':null]
      .filter(Boolean).join('  ·  ');
    if(meta)row.appendChild(planEl('div',meta,'font-size:9px;color:#6e7681'));
    if(p.flags&&p.flags.unlisted)
      row.appendChild(planEl('div','not in the task — the planner named this id itself','font-size:9px;color:#e3b341'));
    box.appendChild(row);
  });
}
function planVisibleSteps(){
  const all=(_planTrace&&_planTrace.decisions)||[];
  if(!_planPoint)return all;
  return all.filter(d=>(d.operations||[]).indexOf(_planPoint)>=0);
}
function planRenderSteps(){
  const box=$('plan-steps');box.textContent='';
  const steps=planVisibleSteps(),total=((_planTrace&&_planTrace.decisions)||[]).length;
  $('plan-step-count').textContent=_planPoint?steps.length+' of '+total:String(total);
  if(!steps.length){box.appendChild(planEl('div','No decision touched this point.','color:#e3b341'));return;}
  steps.forEach(d=>{
    const on=_planSeq===d.seq;
    const res=d.result||{};
    const col=d.kind==='note'?'#8b949e':(res.unanswered?'#e3b341':res.ok===false?'#f85149':res.ok?'#3fb950':'#58a6ff');
    const row=planEl('div',null,'border:1px solid '+(on?col:'#30363d')+';border-left:3px solid '+col
      +';border-radius:5px;padding:5px 7px;cursor:pointer;background:'+(on?'#1f6feb22':'#0d1117'));
    row.onclick=()=>planSelectDecision(d.seq);
    const top=planEl('div',null,'display:flex;gap:6px;align-items:baseline');
    top.appendChild(planEl('span',d.kind==='note'?'—':'#'+d.step,'font-size:9px;color:#6e7681'));
    top.appendChild(planEl('span',d.kind==='note'?'reasoning':(d.tool||'?'),'font-weight:700;font-size:10px;color:#c9d1d9'));
    if(res.unanswered)top.appendChild(planEl('span','no result','font-size:9px;color:#e3b341;margin-left:auto'));
    row.appendChild(top);
    const line=d.kind==='note'?(d.text||''):((d.operations||[]).join(', ')||(d.why||''));
    if(line)row.appendChild(planEl('div',line.length>110?line.slice(0,110)+'…':line,'font-size:9px;color:#6e7681'));
    box.appendChild(row);
  });
}
function planSelectPoint(opId){
  _planPoint=(_planPoint===opId)?null:opId;
  // Jumping to a point should land on its first decision, not leave the detail
  // pane showing a step the filtered list no longer contains.
  const steps=planVisibleSteps();
  _planSeq=steps.length?steps[0].seq:null;
  planRenderPoints();planRenderSteps();planRenderPointDetail();planRenderDetail();
}
/* ── ONE WELD'S STORY ────────────────────────────────────────────────────────
   Was it allocated, and if it took several tries, what changed between them.

   "Allocated" is a stricter question than "planned": a trajectory can exist for a
   weld that never reached the tree, and a view that reports the first as the second
   is how a plan ships with a weld missing. It is answered from the point's flags —
   in_tree means a robot will actually execute it — never from the last thing that
   happened to look successful.

   The failures are kept in full and in order, because the interesting runs are the
   ones that failed twice and then worked: the error text of the failed attempts is
   what the successful attempt was a response to. */
const PLAN_ALLOC={
  in_tree:  ['✓ allocated','#3fb950','a robot executes this weld in the tree'],
  validated:['⚠ validated, not in the tree','#e3b341','the motion checks out but no tree node runs it'],
  planned:  ['⚠ planned, not in the tree','#e3b341','a trajectory exists but no tree node runs it'],
  failed:   ['✗ not allocated','#f85149','the last attempt failed'],
  pending:  ['— never attempted','#6e7681','no decision touched this weld'],
};
function planRenderPointDetail(){
  const box=$('plan-point-detail');if(!box)return;
  box.textContent='';
  const points=(_planTrace&&_planTrace.points)||[];
  const p=points.find(x=>x.operation_id===_planPoint);
  if(!p){box.appendChild(planEl('div',
    'Pick a weld on the left to see whether it was allocated, what failed, and the reasoning that got it working.',
    'color:#6e7681'));return;}

  const alloc=PLAN_ALLOC[p.status]||PLAN_ALLOC.pending;
  const head=planEl('div',null,'display:flex;gap:8px;align-items:baseline;flex-wrap:wrap');
  head.appendChild(planEl('span',p.operation_id,'font-size:12px;font-weight:700;color:#c9d1d9'));
  head.appendChild(planEl('span',alloc[0],'font-size:10px;font-weight:700;color:'+alloc[1]));
  head.appendChild(planEl('span',alloc[2],'font-size:9px;color:#6e7681'));
  box.appendChild(head);

  const facts=planEl('div',null,'margin-top:4px;display:flex;flex-direction:column;gap:2px');
  planKV(facts,'robot',p.robot||'not assigned',p.robot?'#c9d1d9':'#e3b341');
  planKV(facts,'attempts',String(p.attempts||0));
  if(p.type)planKV(facts,'type',p.type);
  if(Array.isArray(p.position))
    planKV(facts,'position','['+p.position.map(v=>Number(v).toFixed(3)).join(', ')+']');
  // Which verdicts this weld ever reached — the history the single status hides.
  const reached=Object.keys(p.flags||{}).filter(k=>k!=='last'&&k!=='unlisted');
  if(reached.length)planKV(facts,'reached',reached.join(' → '));
  box.appendChild(facts);

  // Every attempt, in order, with what it changed and how it ended.
  const all=(_planTrace&&_planTrace.decisions)||[];
  const mine=all.filter(d=>(p.decisions||[]).indexOf(d.seq)>=0);
  const fails=mine.filter(d=>{const r=d.result||{};
    return r.ok===false||((r.failures||[]).some(f=>f&&f.operation_id===p.operation_id));});
  const wins=mine.filter(d=>(d.result||{}).ok===true);

  box.appendChild(planEl('div',
    'Attempt history — '+fails.length+' failed, '+wins.length+' succeeded',
    'font-size:9px;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-top:8px'));
  if(!mine.length)box.appendChild(planEl('div','No decision named this weld.','color:#e3b341'));
  mine.forEach(d=>{
    const r=d.result||{};
    const bad=r.ok===false||(r.failures||[]).some(f=>f&&f.operation_id===p.operation_id);
    const col=r.unanswered?'#e3b341':bad?'#f85149':r.ok?'#3fb950':'#58a6ff';
    const row=planEl('div',null,'border-left:3px solid '+col+';padding:3px 0 3px 6px;margin-top:3px;cursor:pointer');
    row.onclick=()=>planSelectDecision(d.seq);
    const top=planEl('div',null,'display:flex;gap:6px;align-items:baseline;flex-wrap:wrap');
    top.appendChild(planEl('span','#'+d.step,'font-size:9px;color:#6e7681'));
    top.appendChild(planEl('span',d.tool||'?','font-size:10px;font-weight:700;color:#c9d1d9'));
    top.appendChild(planEl('span',r.unanswered?'no result':bad?'FAILED':r.ok?'ok':'returned',
      'font-size:9px;font-weight:700;color:'+col));
    row.appendChild(top);
    // The error is the point of a failed attempt, so it is shown here rather than
    // one click away in the decision pane.
    const errs=(r.failures||[]).filter(f=>f&&(f.operation_id===p.operation_id||!f.operation_id))
      .map(f=>f.error||f.reason||f.message).filter(Boolean);
    if(bad&&!errs.length&&r.payload&&r.payload.error)errs.push(r.payload.error);
    errs.slice(0,3).forEach(e=>row.appendChild(planEl('div',e,'font-size:9px;color:#f85149')));
    box.appendChild(row);
  });

  // The reasoning that got there: the model's own words before the attempt that
  // worked. If none did, the last failure's reasoning is what a human has to act on.
  const decisive=wins.length?wins[wins.length-1]:(fails.length?fails[fails.length-1]:null);
  if(decisive&&decisive.why)
    planBlock(box,wins.length?'Reasoning that led to success':'Reasoning on the last (failed) attempt',
      decisive.why,wins.length?'#3fb950':'#f85149');
  else if(decisive)
    box.appendChild(planEl('div','The model recorded no reasoning text for the deciding attempt.',
      'color:#6e7681;margin-top:7px'));
}
function planSelectDecision(seq){_planSeq=seq;planRenderSteps();planRenderDetail();}
function planStep(delta){
  const steps=planVisibleSteps();if(!steps.length)return;
  let i=steps.findIndex(d=>d.seq===_planSeq);
  if(i<0)i=delta>0?-1:0;
  const next=Math.min(steps.length-1,Math.max(0,i+delta));
  _planSeq=steps[next].seq;planRenderSteps();planRenderDetail();
}
function planKV(parent,label,value,color){
  const row=planEl('div',null,'display:flex;gap:6px');
  row.appendChild(planEl('span',label,'color:#6e7681;min-width:96px'));
  row.appendChild(planEl('span',value,'color:'+(color||'#c9d1d9')+';word-break:break-word'));
  parent.appendChild(row);
}
function planBlock(parent,title,value,color){
  parent.appendChild(planEl('div',title,'font-size:9px;font-weight:700;color:'+(color||'#8b949e')
    +';text-transform:uppercase;letter-spacing:.5px;margin-top:7px'));
  parent.appendChild(planEl('pre',typeof value==='string'?value:JSON.stringify(value,null,2),
    'margin:2px 0 0;padding:6px;background:#0d1117;border:1px solid #30363d;border-radius:4px;'
    +'font-size:9px;color:#c9d1d9;white-space:pre-wrap;word-break:break-word;max-height:180px;overflow:auto'));
}
function planRenderDetail(){
  const box=$('plan-detail');box.textContent='';
  const all=(_planTrace&&_planTrace.decisions)||[];
  const d=all.find(x=>x.seq===_planSeq);
  if(!d){box.appendChild(planEl('div','Pick a decision to see its reasoning, arguments and result.','color:#6e7681'));return;}
  if(d.kind==='note'){planBlock(box,'Reasoning (no tool call)',d.text||'');return;}
  const head=planEl('div',null,'display:flex;gap:8px;align-items:baseline;flex-wrap:wrap');
  head.appendChild(planEl('span','Step '+d.step+' · '+(d.tool||'?'),'font-size:11px;font-weight:700;color:#58a6ff'));
  const res=d.result||{};
  head.appendChild(planEl('span',res.unanswered?'never answered':res.ok===false?'failed':res.ok?'ok':'returned',
    'font-size:9px;color:'+(res.unanswered?'#e3b341':res.ok===false?'#f85149':res.ok?'#3fb950':'#8b949e')));
  box.appendChild(head);
  const facts=planEl('div',null,'margin-top:5px;display:flex;flex-direction:column;gap:2px');
  if((d.operations||[]).length)planKV(facts,'points',d.operations.join(', '));
  if((d.robots||[]).length)planKV(facts,'robots',d.robots.join(', '));
  const m=res.metrics||{};
  Object.keys(m).forEach(k=>planKV(facts,k,typeof m[k]==='object'?JSON.stringify(m[k]):String(m[k])));
  box.appendChild(facts);
  if(d.why)planBlock(box,'Why (the model’s reasoning)',d.why);
  planBlock(box,'Arguments',d.args===undefined?{}:d.args);
  if(res.failures)planBlock(box,'Failures',res.failures,'#f85149');
  if(res.unanswered)box.appendChild(planEl('div',
    'This call has no result: the run ended before the tool replied (cancelled, or the reply was truncated).',
    'color:#e3b341;margin-top:7px'));
  else planBlock(box,'Result',res.payload===undefined?null:res.payload);
}
function opMove(){
  const robot=document.getElementById('op-robot').value,x=parseFloat(document.getElementById('op-x').value),y=parseFloat(document.getElementById('op-y').value),z=parseFloat(document.getElementById('op-z').value),duration=parseFloat(document.getElementById('op-duration').value),st=document.getElementById('op-move-status');
  st.textContent='Sending...';st.style.color='#e3b341';
  fetch('/api/a2a/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({robot_ns:robot,x,y,z,duration_s:duration})}).then(r=>r.json()).then(d=>{
    if(d.error){st.textContent='Error: '+d.error;st.style.color='#f85149';}
    else{st.textContent='\u2713 Dispatched run_id='+d.run_id;st.style.color='#3fb950';if(d.run_id)pollRunId(d.run_id,'op-move-status');}
  }).catch(e=>{st.textContent='Failed: '+e;st.style.color='#f85149';});
}
function opMoveHome(){fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'reset'})});}
// Template uses the repo's real task schema, the one the planner reads:
// `program` + `operations[]`, each operation {id, type, position,
// approach_vector, ...} per skills/orchestrator/run-welding-program/references/
// task-json-format.md. Serving anything else from the button that says
// "Template" hands the operator a program that cannot be planned — an earlier
// version emitted `step_number`/`operation_type`/`parameters.points`, which no
// tool in the repo reads.
function opLoadTemplate(){document.getElementById('op-task-json').value=JSON.stringify({"program":"custom_task","cell_id":CELL_ID,"description":"authored in the operator panel","operations":[{"id":"W01","type":"weld","position":[-1.082,-1.489,1.60],"approach_vector":[0.0,0.0,-1.0],"sequence_group":"structural","priority":1},{"id":"W02","type":"weld","position":[-0.982,-1.239,1.10],"approach_vector":[0.0,0.0,-1.0],"sequence_group":"structural","priority":2}]},null,2);document.getElementById('op-task-status').textContent='Template loaded. Fill it in, or use \u{1F9ED} Wizard to be asked for the values.';}
// The wizard is a SEPARATE page (GET /wizard) in its own window, so the
// operator can keep watching the cell while authoring. It writes the task file
// through POST /api/a2a/task/save and commands nothing; when it is done it
// calls opWizardSaved() back here through window.opener.
function opTaskWizard(){
  const st=document.getElementById('op-task-status');
  const w=window.open('/wizard','task_wizard','width=820,height=900,resizable=yes,scrollbars=yes');
  if(!w){st.textContent='The wizard window was blocked — allow pop-ups for this page, or open /wizard directly.';st.style.color='#f85149';return;}
  st.textContent='Task wizard opened in a new window.';st.style.color='#8b949e';
  try{w.focus();}catch(e){}
}
// Called BY the wizard window once the file is on disk. Fills the textarea with
// what was written so Submit & Run is one click away, and refreshes the Planning
// tab's program list so the new program is selectable there.
function opWizardSaved(program,path,jsonText){
  const st=document.getElementById('op-task-status');
  if(jsonText){const ta=document.getElementById('op-task-json');if(ta)ta.value=jsonText;}
  if(st){st.textContent='✓ Wizard wrote "'+(program||'')+'" → '+(path||'')+' (nothing has run yet)';st.style.color='#3fb950';}
  try{loadPlanTasks();}catch(e){}
}
function opSubmitTask(){
  let d;try{d=JSON.parse(document.getElementById('op-task-json').value);}catch(e){const s=document.getElementById('op-task-status');s.textContent='JSON error: '+e.message;s.style.color='#f85149';return;}
  const st=document.getElementById('op-task-status');st.textContent='Submitting...';st.style.color='#e3b341';
  fetch('/api/a2a/task/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(r=>r.json()).then(r=>{
    if(r.error){st.textContent='Error: '+r.error;st.style.color='#f85149';}
    else{st.textContent='\u2713 Saved "'+(r.saved||r.program||'')+'" run_id='+r.run_id;st.style.color='#3fb950';if(r.run_id)pollRunId(r.run_id,'op-task-status');}
  });
}
function opLoadReasoning(){
  document.getElementById('op-reasoning').textContent='Loading...';
  // Contract from GET /a2a/reasoning: {entries:[{seq,role,msg,...}], status, ...}.
  // "seq" is an integer, not a unix time \u2014 the old renderer was formatting it as
  // a date and printing "01:00:00" for every row.
  fetch('/api/a2a/reasoning').then(r=>r.json()).then(d=>{
    const t=(d.entries||[]).slice(-30);
    const el=document.getElementById('op-reasoning');
    if(!t.length){el.textContent=(d.note||d.error||'No reasoning yet.');return;}
    el.textContent=t.map(e=>'['+String(e.seq).padStart(3,'0')+' '+e.role+'] '+(e.msg||'')).join('\n');
    el.scrollTop=99999;
  });
}
/* ── Behaviour-tree view ──────────────────────────────────────────────────
   Draws the BehaviorTree.CPP v4 tree the planner produced: the control nodes
   that decide the order, and the leaves that actually move an arm. The server
   sends the shape and the ports only — the waypoint arrays stay behind
   /api/plans/<tree_id>'s stripped blackboard, so a 200-weld plan is no more
   expensive to draw than a two-leaf one. `trajectory_sizes` carries the sample
   count per referenced key, which is what a leaf's badge shows. */
// GAPX is 26, not the old 14: the sibling-order arrows are drawn in that gap, and
// an arrowhead needs somewhere to land.
const BT_W=170,BT_H=46,BT_GAPX=26,BT_GAPY=40,BT_PAD=16;
const BT_STYLE={
  control:{fill:"#132c4d",stroke:"#58a6ff",text:"#79c0ff"},
  parallel:{fill:"#2a1f47",stroke:"#a371f7",text:"#c297ff"},
  fallback:{fill:"#43290f",stroke:"#e3b341",text:"#e3b341"},
  action:  {fill:"#10301f",stroke:"#3fb950",text:"#56d364"},
  other:   {fill:"#1c2128",stroke:"#30363d",text:"#c9d1d9"},
};
function btKind(t){
  t=String(t||"");
  if(/^Parallel/.test(t))return "parallel";
  if(/Fallback|Selector/.test(t))return "fallback";
  if(/^(Sequence|ReactiveSequence|SequenceWithMemory|SubTree|Root|Repeat|RetryUntilSuccessful|Inverter|ForceSuccess|ForceFailure|Timeout|Delay)/.test(t))return "control";
  return "action";
}
/* Does this node run its children IN ORDER, or all at once?
   The distinction is the plan's whole meaning — "weld, then move, then weld" and
   "weld while moving" are different cell behaviours and different cycle times —
   and the drawing used to lose it: every parent laid its children out side by side,
   so a Sequence looked exactly like a Parallel. A reader could not tell what the
   planner had decided. Ordered children now carry their execution index and an
   arrow to the next one; concurrent children carry a ∥ and no arrows.

   Fallback is ordered too: it tries child 1, and only on failure moves to child 2.
   Numbering it is right — the numbers are attempt order rather than a schedule. */
function btOrdered(t){
  t=String(t||"");
  if(/^Parallel/.test(t))return false;
  return /^(Sequence|ReactiveSequence|SequenceWithMemory|Fallback|ReactiveFallback|Selector|Root|SubTree)/.test(t);
}
let _btNodes={}, _btWant=null, _btSizes={};
/* tree_id -> its row from /api/plans, and whether the tree on screen may be run.
   The plans directory holds FAILED runs as well as good ones (deliberately — a run
   that solved thirteen welds is worth reading and resuming), so "is this one
   executable" is a real question the tab has to be able to answer. */
let _btStatus={}, _btExecutable=true;
function btReplaySay(txt,col){const el=$('bt-replay-status');
  if(el){el.textContent=txt||'';el.style.color=col||'#6e7681';}}
/* Tidy top-down layout: leaves take a slot each, a parent centres over its
   children. Cursor-based, so the width comes out of the tree instead of being
   guessed — a 3-arm plan just makes the canvas wider and scrolls. */
function btLayout(node,depth,x0,out){
  const kids=(node.children||[]);
  let cx, x1;
  if(!kids.length){x1=x0+BT_W;cx=x0+BT_W/2;}
  else{
    let cur=x0;const centres=[];
    for(const k of kids){const laid=btLayout(k,depth+1,cur,out);centres.push(laid.cx);cur=laid.x1+BT_GAPX;}
    x1=Math.max(cur-BT_GAPX,x0+BT_W);
    cx=(centres[0]+centres[centres.length-1])/2;
  }
  const rec={node,depth,cx,x1,y:BT_PAD+depth*(BT_H+BT_GAPY)};
  out.push(rec);
  return rec;
}
function renderTree(bt,sizes){
  const svg=$("bt-svg"),detail=$("bt-detail");if(!svg)return;
  svg.innerHTML="";_btNodes={};_btSizes=sizes||{};
  const root=bt&&bt.root;
  if(!root){svg.setAttribute("width","10");svg.setAttribute("height","10");return;}
  const laid=[];btLayout(root,0,BT_PAD,laid);
  const W=Math.max(...laid.map(r=>r.x1))+BT_PAD;
  const H=Math.max(...laid.map(r=>r.y))+BT_H+BT_PAD;
  svg.setAttribute("width",W);svg.setAttribute("height",H);
  svg.setAttribute("viewBox","0 0 "+W+" "+H);
  const NS="http://www.w3.org/2000/svg";
  const el=(tag,attrs)=>{const e=document.createElementNS(NS,tag);
    for(const k in attrs)e.setAttribute(k,attrs[k]);return e;};
  // An arrowhead for the sibling-order arrows. Defined per render because
  // svg.innerHTML="" above wipes the previous defs with everything else.
  const defs=el("defs",{});
  const mk=el("marker",{id:"bt-arr",viewBox:"0 0 10 10",refX:"9",refY:"5",
    markerWidth:"5",markerHeight:"5",orient:"auto-start-reverse"});
  mk.appendChild(el("path",{d:"M0,0 L10,5 L0,10 z",fill:"#8b949e"}));
  defs.appendChild(mk);svg.appendChild(defs);
  // How each node is reached: which slot it occupies among its siblings, and
  // whether the parent runs those siblings in order. Collected here so a box can
  // draw its own index without walking back up the tree.
  const slot=new Map();
  laid.forEach(r=>{
    const kids=r.node.children||[];
    if(kids.length<2)return;                 // one child has no order to show
    const ordered=btOrdered(r.node.type);
    kids.forEach((child,i)=>slot.set(child,{i:i+1,n:kids.length,ordered:ordered}));
  });
  // Edges first so the boxes sit on top of them. A parallel parent's edges are
  // dashed: the branch is entered at the same time as its siblings, not after them.
  laid.forEach(r=>{
    const kids=r.node.children||[];
    const ordered=btOrdered(r.node.type);
    kids.forEach(child=>{
      const c=laid.find(x=>x.node===child);if(!c)return;
      const y1=r.y+BT_H,y2=c.y,my=(y1+y2)/2;
      const path=el("path",{class:"bt-edge",
        d:"M"+r.cx+","+y1+" C"+r.cx+","+my+" "+c.cx+","+my+" "+c.cx+","+y2});
      if(!ordered)path.setAttribute("stroke-dasharray","4 3");
      svg.appendChild(path);
    });
    // …and, when the parent is ordered, an arrow from each child to the next one.
    // This is the fix for "the blocks under a Sequence all look parallel": the
    // execution order is now drawn instead of merely implied by left-to-right
    // position, which a Parallel's children have too.
    if(!ordered||kids.length<2)return;
    for(let i=0;i<kids.length-1;i++){
      const a=laid.find(x=>x.node===kids[i]),b=laid.find(x=>x.node===kids[i+1]);
      if(!a||!b)continue;
      const y=a.y+BT_H/2;
      // A <line>, not a <path>: it is a straight segment, and keeping it a
      // different element from the parent→child <path> edges means the two kinds of
      // link stay countable apart — which is how the drawing is tested.
      svg.appendChild(el("line",{class:"bt-order",
        x1:a.cx+BT_W/2+2,y1:y,x2:b.cx-BT_W/2-6,y2:y,
        "marker-end":"url(#bt-arr)"}));
    }
  });
  laid.forEach(r=>{
    const n=r.node,st=BT_STYLE[btKind(n.type)],x=r.cx-BT_W/2;
    const nid=n.id||("n"+r.depth+"_"+Math.round(r.cx));
    _btNodes[nid]=n;
    const g=el("g",{class:"bt-box",id:"bt-"+nid});
    g.appendChild(el("rect",{x:x,y:r.y,width:BT_W,height:BT_H,rx:7,
      fill:st.fill,stroke:st.stroke}));
    const t1=el("text",{x:r.cx,y:r.y+18,class:"bt-type",fill:st.text});
    t1.textContent=n.type||"?";g.appendChild(t1);
    const traj=(n.ports||{}).trajectory||"";
    const key=/^\{(.+)\}$/.exec(String(traj));
    const pts=key&&_btSizes[key[1]]!==undefined?_btSizes[key[1]]+" pts":"";
    const sub=[n.name||"",pts].filter(Boolean).join("  ·  ");
    const t2=el("text",{x:r.cx,y:r.y+34,class:"bt-name"});
    t2.textContent=sub.length>30?sub.slice(0,29)+"…":sub;g.appendChild(t2);
    // Its place among its siblings: "2/5" when the parent runs them in order,
    // "∥" when they all start together. Without this the position of a box on the
    // canvas is the only hint at order, and position lies for a Parallel.
    const s=slot.get(n);
    if(s){
      const badge=el("circle",{cx:x+11,cy:r.y+11,r:8.5,
        fill:s.ordered?"#0d1117":"#2a1f47",stroke:s.ordered?"#8b949e":"#a371f7"});
      g.appendChild(badge);
      const bt=el("text",{x:x+11,y:r.y+14.5,class:"bt-slot",
        fill:s.ordered?"#c9d1d9":"#c297ff"});
      bt.textContent=s.ordered?String(s.i):"∥";
      g.appendChild(bt);
      const title=el("title",{});
      title.textContent=s.ordered
        ? "step "+s.i+" of "+s.n+" — runs after the sibling to its left"
        : "runs at the same time as its "+(s.n-1)+" sibling(s)";
      g.appendChild(title);
    }
    g.addEventListener("click",()=>btSelect(nid));
    svg.appendChild(g);
  });
  if(detail)detail.innerHTML='<span style="color:#6e7681">Click a node to see its ports.</span>';
}
function btSelect(nid){
  const n=_btNodes[nid],detail=$("bt-detail");if(!n||!detail)return;
  document.querySelectorAll("#bt-svg .bt-box").forEach(g=>g.classList.remove("bt-sel"));
  const g=$("bt-"+nid);if(g)g.classList.add("bt-sel");
  const rows=Object.entries(n.ports||{}).map(([k,v])=>{
    const key=/^\{(.+)\}$/.exec(String(v));
    const extra=key&&_btSizes[key[1]]!==undefined
      ? " <span style=\"color:#6e7681\">(" + _btSizes[key[1]] + " waypoints, kept server-side)</span>" : "";
    return "<b>"+esc(k)+"</b><span>"+esc(String(v))+extra+"</span>";
  }).join("");
  detail.innerHTML="<div style='font-size:11px;font-weight:700;color:#c9d1d9;margin-bottom:6px'>"
    +esc(n.type||"?")+(n.name?" <span style='color:#8b949e;font-weight:400'>"+esc(n.name)+"</span>":"")+"</div>"
    +(rows?"<div class='bt-kv'>"+rows+"</div>"
          :"<span style='color:#6e7681'>No ports — a control node. Its children run "
           +(btOrdered(n.type)?"ONE AFTER ANOTHER, left to right (numbered in the tree)."
                              :"ALL AT THE SAME TIME (marked ∥ in the tree).")+"</span>");
  // …and one level further down: the motion inside this node.
  loadNodeDetail(nid);
}
/* Node names and port values come off disk. They are rendered as text, never as
   markup: a plan file is not a trusted template. */
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function loadPlanList(){
  const sel=$('bt-select');if(!sel)return;
  fetch('/api/plans').then(r=>r.json()).then(d=>{
    const plans=d.plans||[];
    if(!plans.length){
      sel.innerHTML='<option value="">'+esc(d.error||'No calculated plans yet')+'</option>';
      const meta=$('bt-meta');
      if(meta)meta.textContent='Run "make plan" or the Agentic button to calculate one.';
      renderTree(null,{});return;
    }
    const want=_btWant||sel.value||plans[0].tree_id;
    // A FAILED run is listed on purpose — a token that expired at weld 13 says
    // nothing about welds 1-12, and that run is worth reading and worth resuming.
    // So it is labelled, not hidden: the selector has to say which rows Replay
    // will refuse, or the refusal arrives as a surprise after the click.
    _btStatus={};
    plans.forEach(p=>{_btStatus[p.tree_id]=p;});
    sel.innerHTML=plans.map(p=>'<option value="'+esc(p.tree_id)+'">'
      +(p.status==='failed'?'⚠ FAILED  ':'')+esc(p.tree_id)
      +'  ('+(p.status==='failed'
                ? (p.num_planned||0)+' solved, not executable'
                : (p.num_operations||0)+' ops')+')</option>').join('');
    const have=plans.some(p=>p.tree_id===want);
    sel.value=have?want:plans[0].tree_id;
    // A recording can outlive its plan (and home_reset_* never had one), so say
    // so instead of quietly drawing a different tree than the one asked for.
    const missed=(_btWant&&!have)?_btWant:null;
    _btWant=null;
    loadTree(sel.value, missed?'no stored plan for "'+missed+'", showing '+sel.value:'');
  }).catch(e=>{sel.innerHTML='<option value="">list failed: '+esc(e)+'</option>';});
}
function loadTree(tree_id,note){
  const meta=$('bt-meta');
  // Drop the previous plan's numbers NOW, not when the new ones arrive: for the
  // length of the fetch the card would otherwise publish one plan's score under
  // another plan's name, and an open report window would outlive its plan
  // entirely if the new one turns out not to be scorable. kpiSay closes it.
  kpiSay(tree_id?'Loading '+tree_id+'…':'Select a plan to see its KPIs.');
  if(!tree_id){if(meta)meta.textContent='';return;}
  if(meta){meta.textContent='Loading '+tree_id+'…';meta.style.color='#6e7681';}
  fetch('/api/plans/'+encodeURIComponent(tree_id)).then(r=>r.json()).then(d=>{
    if(!d.success){if(meta){meta.textContent='Could not load '+tree_id+': '+(d.error||'?');meta.style.color='#f85149';}renderTree(null,{});return;}
    // A failed run reads, it just does not run. Its evidence goes where the tree
    // summary goes, because that is where the operator is already looking, and
    // Replay is disabled rather than left to fail at the executor.
    _btExecutable=(d.executable!==false);
    const bt=d.behaviour_tree||{};
    // Which tree the per-node fetches address. Set from the RESPONSE, not the
    // argument, so 'latest' resolves to a real id before anything asks for a node.
    _btTree=d.tree_id||tree_id;_btDetail={};_btOpenPhase=null;
    const steps=$('bt-steps');if(steps)steps.innerHTML=
      '<span style="color:#6e7681">Click a node in the tree to step through its motion.</span>';
    const wp=Object.values(d.trajectory_sizes||{}).reduce((a,b)=>a+b,0);
    if(d.status==='failed'){
      if(meta){meta.style.color='#f85149';
        meta.textContent='⚠ FAILED run — not executable. '+(d.error||'no tree assembled')
          +' · '+(d.num_operations||0)+' weld(s) solved before it stopped'
          +(d.resumable?' · resumable: use Resume in the Agents tab'
                       :' · nothing usable was captured, so it cannot be resumed');}
      // Whatever partial tree the assembler produced is still worth drawing; a run
      // that died before assembling one has nothing to draw, and says so.
      renderTree(bt.root?bt:null,d.trajectory_sizes||{});
      btReplaySay(bt.root?'This is a FAILED run — replay is disabled.'
                         :'This run stopped before it built a tree.','#f85149');
      // A fragment is not scored: the numbers compare a whole program against
      // itself, and a partial run would post a flattering score for stopping early.
      kpiSay('A FAILED run is not scored — its legs are a fragment, not a program.');
      return;
    }
    if(meta){meta.style.color=note?'#e3b341':'#6e7681';
      meta.textContent=(note?note+' · ':'')
        +d.num_operations+' executable operation(s) · '
        +Object.keys(d.trajectory_sizes||{}).length+' trajectories · '+wp+' waypoints · BTCPP '
        +(bt.BTCPP_format||'4');}
    btReplaySay('');
    renderTree(bt,d.trajectory_sizes||{});
    // The score follows the tree it belongs to: _btTree, not the argument, so
    // 'latest' has already resolved to a real id.
    loadPlanQuality(_btTree);
  }).catch(e=>{if(meta){meta.textContent='Load failed: '+e;meta.style.color='#f85149';}});
}
/* ── Performance of the selected plan ────────────────────────────────────────
   The KPIs come from the planner over MCP (GET /api/plans/<id>/quality →
   get_plan_quality → tools/trajectory_quality.py). Nothing is computed here: a
   second implementation of "how good is this plan" in the browser is a second
   answer to the question, and the two would drift the first time a weight moved.

   Two surfaces, one fetch. The CARD publishes the metric: the weighted score, its
   verdict, and how many findings are behind it. The WINDOW (#bt-kpi-modal) holds
   the report itself — the four components, the numbers behind them, every finding
   worst first, the worst legs, and the safety line. Both are filled the moment the
   answer arrives, so opening the window is instant and asks the planner nothing a
   second time; only visibility waits for the click. */
const KPI_VERDICT_COLOUR={good:'#3fb950',acceptable:'#e3b341',poor:'#f85149'};
const KPI_SEVERITY_COLOUR={high:'#f85149',medium:'#e3b341',low:'#8b949e'};
/* A missing metric prints as "—", never as 0: the report says None when it could
   not measure something (no robots.yaml, a single-arm cell has no load balance),
   and a zero there reads as a measured perfect score. */
function qnum(v,dp,suffix){
  if(v===null||v===undefined||v==='')return '—';
  const n=Number(v);if(!isFinite(n))return '—';
  return n.toFixed(dp===undefined?2:dp)+(suffix||'');
}
function qpct(v){return (v===null||v===undefined)?'—':Math.round(Number(v)*100)+'%';}
/* Everything the two surfaces show, or null when there is nothing measured. Held
   so the window can open without re-asking, and so kpiOpen can refuse rather than
   present an empty report as a report. */
let _kpiReport=null;
const KPI_BLOCKS=['bt-kpi-score','bt-kpi-open','bt-kpi-modal-score','bt-kpi-bars',
                  'bt-kpi-grid','bt-kpi-findings','bt-kpi-legs','bt-kpi-safety'];
function kpiSay(txt,col){
  const st=$('bt-kpi-status');
  // Empty them, do not merely hide them: hidden-but-filled markup is one
  // stray style change away from publishing the previous plan's numbers again.
  KPI_BLOCKS.forEach(id=>{const el=$(id);if(el){el.style.display='none';el.innerHTML='';}});
  // Close it too: a report left open over a plan that is no longer measured is
  // the same wrong picture the status line exists to prevent.
  _kpiReport=null;kpiClose();
  const src=$('bt-kpi-source');if(src)src.textContent='';
  if(st){st.style.display='block';st.textContent=txt;st.style.color=col||'#6e7681';}
}
/* ── the dedicated window ─────────────────────────────────────────────────── */
function kpiOpen(){
  // Nothing measured means nothing to open. The card's status line already says
  // why, and an empty window over it would hide the reason.
  if(!_kpiReport)return;
  const m=$('bt-kpi-modal');if(m)m.style.display='flex';
}
function kpiClose(){const m=$('bt-kpi-modal');if(m)m.style.display='none';}
/* Only the backdrop closes, never the sheet: a click on the report itself — to
   select a leg name, to read a finding — must not dismiss what is being read. */
function kpiBackdrop(ev){if(ev&&ev.target&&ev.target.id==='bt-kpi-modal')kpiClose();}
/* Named rather than inline so it is reachable from a test and from any future
   second dialog: Escape closes the topmost thing, and that rule has to live in
   one place. */
function kpiKey(ev){if(ev&&(ev.key==='Escape'||ev.key==='Esc'))kpiClose();}
/* The button re-asks for the tree currently shown. Takes the id from the loaded
   tree rather than the selector, so it cannot ask about a plan the pane is not
   displaying (the selector moves before loadTree resolves 'latest'). */
function kpiRefresh(){loadPlanQuality(_btTree);}
function loadPlanQuality(tree_id){
  if(!tree_id){kpiSay('Select a plan to see its KPIs.');return;}
  kpiSay('Measuring '+tree_id+'…');
  fetch('/api/plans/'+encodeURIComponent(tree_id)+'/quality')
    .then(r=>r.json())
    .then(d=>{
      // "Not measured, because X" and "measured badly" are different answers, and
      // the panel must never render the first as the second.
      if(!d||d.success===false){kpiSay(esc((d&&d.error)||'not measured'),'#e3b341');return;}
      renderPlanQuality(d);
    })
    .catch(e=>kpiSay('KPIs unavailable: '+esc(String(e)),'#f85149'));
}
function renderPlanQuality(d){
  const st=$('bt-kpi-status');if(st)st.style.display='none';
  _kpiReport=d;
  const s=d.summary||{}, sched=d.schedule||{}, comp=d.score_components||{},
        weights=d.score_weights||{}, legs=(d.legs||[]).slice();
  const src=$('bt-kpi-source');
  if(src)src.textContent=(d.source==='computed'
      ? 'scored on demand — this plan predates the metrics'
      : 'as recorded when the plan was saved');
  const verdict=String(d.verdict||'').toLowerCase();
  const col=KPI_VERDICT_COLOUR[verdict]||'#8b949e';
  // The published metric. The same block on both surfaces: the window has to
  // repeat the number it is explaining, or the reader who opened it loses it.
  const headline='<span style="font-size:26px;font-weight:800;color:'+col+'">'
      +qnum(d.efficiency_score,2)+'</span>'
      +'<span style="font-size:10px;font-weight:700;color:'+col
      +';border:1px solid '+col+';border-radius:10px;padding:1px 8px;text-transform:uppercase;letter-spacing:.5px">'
      +esc(verdict||'unscored')+'</span>'
      +'<span style="font-size:9.5px;color:#6e7681;margin-left:auto;text-align:right">'
      +qnum(s.makespan_s,1,' s')+' makespan · '+(s.motion_legs||0)+' legs</span>';
  ['bt-kpi-score','bt-kpi-modal-score'].forEach(id=>{
    const el=$(id);if(!el)return;
    el.style.display='flex';el.innerHTML=headline;
  });
  const tree=$('bt-kpi-modal-tree');
  if(tree)tree.textContent=String(d.tree_id||'')+(d.cell_id?' · '+d.cell_id:'');
  // What the click is FOR, said in the card: the count is the reason to open the
  // window, and "Details" alone gives a reader no idea whether it is worth it.
  const nf=(d.findings||[]).length;
  const open=$('bt-kpi-open');
  if(open){
    open.style.display='block';
    open.innerHTML='<span style="color:'+(nf?'#e3b341':'#3fb950')+'">'
      +(nf?nf+' finding'+(nf===1?'':'s'):'no findings')+'</span>'
      +'<span style="color:#6e7681"> · components, per-leg numbers and clearance</span>'
      +'<span style="color:#8b949e;float:right">Full report &#x2192;</span>';
  }
  // One bar per weighted component, so a poor score says WHICH of the four it is.
  const bars=$('bt-kpi-bars');
  if(bars){
    bars.style.display='flex';
    bars.innerHTML=['path','schedule','smoothness','dwell'].map(k=>{
      const v=comp[k], pc=(v===null||v===undefined)?0:Math.max(0,Math.min(1,Number(v)))*100;
      const c=pc>=85?'#3fb950':pc>=70?'#e3b341':'#f85149';
      return '<div style="display:flex;align-items:center;gap:6px">'
        +'<span style="font-size:9px;color:#8b949e;width:74px">'+esc(k)
        +'<span style="color:#484f58"> &#xD7;'+qnum(weights[k],2)+'</span></span>'
        +'<span style="flex:1;height:5px;background:#0d1117;border-radius:3px;overflow:hidden">'
        +'<span style="display:block;height:5px;width:'+pc.toFixed(0)+'%;background:'+c+'"></span></span>'
        +'<span style="font-size:9px;color:#c9d1d9;width:30px;text-align:right">'
        +qnum(v,2)+'</span></div>';
    }).join('');
  }
  // The evidence. Excess travel is the headline: it is motion the cell paid for
  // and did not need, and it is what the defect these metrics exist for looks like.
  const util=sched.utilisation||{};
  const rows=[
    ['excess travel', qnum(s.total_excess_travel_rad,2,' rad'),
     'travelled '+qnum(s.joint_travel_rad,1)+' rad to cover '+qnum(s.direct_distance_rad,1)+' rad'],
    ['worst detour', qnum(s.worst_detour_ratio,2,'×'),
     'mean '+qnum(s.mean_detour_ratio,2)+'× · ideal 1.00, shortcut RRT ≤ ~1.15'],
    ['parallel eff.', qpct(s.parallel_efficiency),
     'serial '+qpct(s.serial_fraction)+' of the makespan'],
    ['load balance', qpct(s.load_balance),
     'busiest vs. idlest arm'],
    ['interior dwell', qnum(s.total_interior_dwell_s,2,' s'),
     'time stopped mid-leg — pure loss'],
    ['arm use', Object.keys(util).map(a=>qpct(util[a])).join(' / ')||'—',
     Object.keys(util).join(' / ')||'no timeline'],
  ];
  const grid=$('bt-kpi-grid');
  if(grid){
    grid.style.display='grid';
    grid.innerHTML=rows.map(r=>
      '<div title="'+esc(r[2])+'"><div style="font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px">'
      +esc(r[0])+'</div><div style="font-size:12px;font-weight:700;color:#c9d1d9">'
      +esc(r[1])+'</div></div>').join('');
  }
  // Findings, worst first and ALL of them — the window scrolls, and the reason it
  // is a window is that nothing has to be left out. Truncating the tail here would
  // hide exactly the low-severity findings a reader opens the report to hunt for.
  const fnd=(d.findings||[]);
  const fbox=$('bt-kpi-findings');
  if(fbox){
    fbox.style.display='flex';
    if(!fnd.length){
      fbox.innerHTML='<div style="font-size:10px;color:#3fb950">No findings — every leg is within its thresholds.</div>';
    }else{
      fbox.innerHTML=fnd.map(f=>{
        const c=KPI_SEVERITY_COLOUR[String(f.severity||'').toLowerCase()]||'#8b949e';
        const where=f.node||f.robot_ns||'plan';
        return '<div style="display:flex;gap:6px;align-items:flex-start;font-size:9.5px;line-height:1.45">'
          +'<span style="color:'+c+';font-weight:700;min-width:52px">'+esc(String(f.severity||'?'))+'</span>'
          +'<span style="color:#c9d1d9;min-width:0"><b>'+esc(String(f.kind||''))+'</b> '
          +'<span style="color:#8b949e">'+esc(String(where))+'</span> — '+esc(String(f.detail||''))
          +'</span></div>';
      }).join('');
    }
  }
  // The worst legs by excess radians, not by ratio: ranking on the ratio puts a
  // short 3.4x transit above a home return that wanders 10 rad.
  const worst=legs.filter(l=>l&&l.usable)
    .sort((a,b)=>(b.excess_travel_rad||0)-(a.excess_travel_rad||0)).slice(0,10);
  const lbox=$('bt-kpi-legs');
  if(lbox){
    if(!worst.length){lbox.style.display='none';}
    else{
      lbox.style.display='flex';
      lbox.innerHTML='<div style="font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px">worst legs — detour, excess, waypoints, duration</div>'
        +worst.map(l=>
          '<div style="display:flex;gap:6px;font-size:9.5px;color:#8b949e">'
          +'<span style="color:#c9d1d9;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1">'
          +esc(String(l.node||l.robot_ns||'leg'))+'</span>'
          +'<span>'+qnum(l.detour_ratio,2,'×')+'</span>'
          +'<span>+'+qnum(l.excess_travel_rad,2)+' rad</span>'
          +'<span>'+(l.waypoints||0)+' wp</span>'
          +'<span>'+qnum(l.duration_s,1,' s')+'</span></div>').join('');
    }
  }
  /* Clearance, last and separate. Reported because an operator asking "is this
     plan good" is also asking "how close did it come", and stated as excluded
     because a number printed beside a score gets optimised like one. */
  const safe=d.safety||{};
  const sbox=$('bt-kpi-safety');
  if(sbox){
    sbox.style.display='flex';
    sbox.innerHTML='<div style="font-size:9px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px">clearance — reported, never scored</div>'
      +'<div style="font-size:11px;font-weight:700;color:#c9d1d9">'
      +qnum(safe.achieved_clearance_m,3,' m')+' achieved</div>'
      +'<div style="font-size:9px;color:#6e7681;line-height:1.5">'
      +esc(String(safe.note||'no clearance recorded for this plan'))
      +' — it is deliberately outside every score above: a score is a thing people'
      +' optimise, and a scored safety margin is a tunable one.</div>';
  }
}
/* ── Step-through: descend INTO a node ───────────────────────────────────────
   The tree drawing stops at a leaf, and a leaf is opaque: "ExecuteTrajectory
   move_robot1_VIA_W06" does not say what the arm does. One level further down is
   the motion itself — its phase runs (the planner labels each waypoint "transit"
   or "approach", the only record of which part is the fast cross-cell move and
   which the slow tool-oriented run-in) and then the waypoints.

   Waypoints are fetched PER NODE, never for the whole plan: one node is a couple
   of hundred samples whatever the size of the plan, so this costs the same on a
   200-weld tree as on a two-leaf one. */
let _btTree=null, _btDetail={}, _btOpenPhase=null;
function btStepsEl(){return $('bt-steps');}
function previewSay(txt,col){const el=$('bt-preview-status');if(el){el.textContent=txt;el.style.color=col||'#6e7681';}}
function loadNodeDetail(nid){
  const box=btStepsEl();if(!box||!_btTree)return;
  if(_btDetail[nid]){renderNodeSteps(nid);return;}
  box.innerHTML='<span style="color:#6e7681">Loading node&#x2026;</span>';
  fetch('/api/plans/'+encodeURIComponent(_btTree)+'/nodes/'+encodeURIComponent(nid))
    .then(r=>r.json()).then(d=>{
      if(!d.success){box.innerHTML='<span style="color:#f85149">'+esc(d.error||'node not found')+'</span>';return;}
      _btDetail[nid]=d;_btOpenPhase=null;renderNodeSteps(nid);
    }).catch(e=>{box.innerHTML='<span style="color:#f85149">'+esc('load failed: '+e)+'</span>';});
}
function renderNodeSteps(nid){
  const box=btStepsEl(),d=_btDetail[nid];if(!box||!d)return;
  const segs=d.phase_segments||[];
  const btn=(label,call,title)=>'<button title="'+esc(title||'')+'" onclick="'+call
    +'" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;'
    +'padding:1px 6px;font-size:10px;cursor:pointer;margin-right:4px">'+label+'</button>';
  let h='<div style="display:flex;align-items:center;gap:6px;margin-bottom:7px">'
    +'<b style="color:#c9d1d9">'+esc(d.type||'?')+'</b>'
    +'<span style="color:#8b949e">'+esc(d.name||nid)+'</span>';
  if((d.waypoints||[]).length){
    h+='<span style="margin-left:auto">'
      +btn('&#x25B6; Preview node',"btPreview('"+nid+"',{})","Animate the whole motion on the ghost arm — the cell does not move")
      +btn('&#x23F9;','btPreviewClear()','Park the ghost back on the real arm')+'</span>';
  }
  h+='</div>';
  if(!segs.length){
    // A control node runs its children; the tree above already shows them.
    h+='<span style="color:#6e7681">'+(d.children&&d.children.length
      ? 'Control node — runs '+d.children.length+' child node(s); pick one in the tree above.'
      : 'No trajectory on this node, so there is nothing to step through.')+'</span>';
    box.innerHTML=h;return;
  }
  h+='<div style="color:#6e7681;margin-bottom:5px">'+segs.length+' phase segment(s) &#xB7; '
    +(d.waypoints||[]).length+' waypoint(s) &#xB7; '+(d.duration_s||0).toFixed(2)+'s</div>';
  const PH={transit:'#58a6ff',approach:'#e3b341',retreat:'#a371f7'};
  segs.forEach((s,i)=>{
    const col=PH[s.phase]||'#8b949e';
    h+='<div style="border-top:1px solid #21262d;padding:4px 0">'
      +'<div style="display:flex;align-items:center;gap:6px">'
      +'<span onclick="btTogglePhase(\''+nid+'\','+i+')" style="cursor:pointer;color:#6e7681;width:10px">'
      +(_btOpenPhase===i?'&#x25BE;':'&#x25B8;')+'</span>'
      +'<b style="color:'+col+'">'+esc(s.phase)+'</b>'
      +'<span style="color:#6e7681">'+s.points+' pts &#xB7; '
      +s.start_s.toFixed(2)+' &#x2192; '+s.end_s.toFixed(2)+'s</span>'
      +'<span style="margin-left:auto">'
      +btn('&#x25B6;',"btPreview('"+nid+"',{phase_index:"+i+"})",'Preview just this phase')
      +'</span></div>';
    if(_btOpenPhase===i){
      const wps=(d.waypoints||[]).slice(s.start_index,s.end_index+1);
      h+='<div style="max-height:150px;overflow:auto;margin:4px 0 0 16px;font-family:ui-monospace,monospace">';
      wps.forEach((w,j)=>{
        const idx=s.start_index+j;
        const joints=(w.positions||[]).map(v=>Number(v).toFixed(3)).join(', ');
        h+='<div style="display:flex;gap:6px;padding:1px 0">'
          +btn('&#x25B8;',"btPreview('"+nid+"',{waypoint_index:"+idx+",loop:false})",
               'Hold the ghost at this exact pose')
          +'<span style="color:#6e7681;min-width:74px">#'+idx+'  t='+Number(w.time_s||0).toFixed(2)+'s</span>'
          +'<span style="color:#8b949e">['+esc(joints)+']</span></div>';
      });
      h+='</div>';
    }
    h+='</div>';
  });
  box.innerHTML=h;
}
function btTogglePhase(nid,i){_btOpenPhase=(_btOpenPhase===i?null:i);renderNodeSteps(nid);}
/* Previews drive a translucent GHOST copy of the arm in RViz. Nothing is
   dispatched to a robot agent — see POST /api/preview and launch_viz's
   GHOST_SUFFIX — so this is safe to press at any time, including mid-run. */
function btPreview(nid,opts){
  if(!_btTree){previewSay('no tree selected','#f85149');return;}
  const body=Object.assign({tree_id:_btTree,node_id:nid},opts||{});
  previewSay('sending preview…','#e3b341');
  fetch('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(r=>r.json().then(d=>({ok:r.ok,d:d}))).then(x=>{
      if(!x.ok||!x.d.success){previewSay('✗ '+(x.d.error||'preview failed'),'#f85149');return;}
      previewSay('▶ ghost: '+x.d.label+' — '+x.d.waypoints+' pt, '
        +x.d.duration_s+'s on '+x.d.robot_ns+' (cell not moved)','#3fb950');
    }).catch(e=>previewSay('✗ preview failed: '+(e.message||e),'#f85149'));
}
function btPreviewClear(){
  fetch('/api/preview/clear',{method:'POST'}).then(r=>r.json())
    .then(()=>previewSay('ghost parked on the real arm','#6e7681'))
    .catch(e=>previewSay('✗ clear failed: '+(e.message||e),'#f85149'));
}
/* Replay the tree that is on screen — the button the removed Rec tab used to own.
   Every CALCULATED plan is replayable (/a2a/replay takes a tree_id), so the
   selector above is the right list to drive it from; the recordings browser listed
   only the plans that had already executed at least once, and so hid the rest.

   Status goes to #bt-replay-status and the tab does NOT change. The old version
   ended with switchTab('chat'), so pressing Replay threw the operator out of the
   pane they were reading and the tree they had just launched vanished from under
   them. The run is still narrated in the log panel by the poller; the operator is
   simply not dragged away from it. */
// Speed is the CELL's velocity scale, set separately via POST /api/speed. The
// orchestrator's /a2a/replay does not take a speed \u2014 passing one there would be
// a second, weaker knob on the same setting.
function btReplay(){
  const st=$('bt-replay-status'),sel=$('bt-select');
  const tree=(sel&&sel.value)||_btTree;
  const speed=parseFloat(($('bt-speed')||{}).value||'1');
  const say=(t,c)=>{if(st){st.textContent=t;st.style.color=c||'#6e7681';}};
  if(!tree){say('No plan selected \u2014 calculate one first.','#e3b341');return;}
  // Refused before it is sent, because a FAILED run's partial tree is an artefact
  // the assembler rejected and handing it to the executor would move real arms
  // through a plan nothing validated. The executor refuses it too (replay_plan);
  // this is the polite half, so the operator gets the reason and not an error.
  const known=_btStatus[tree];
  if(_btExecutable===false||(known&&known.status==='failed')){
    say('\u2716 '+tree+' is a FAILED run \u2014 not executable. '
        +((known&&known.resumable)?'Resume it in the Agents tab, or plan again.'
                                  :'Plan again; there is nothing usable to resume.'),
        '#f85149');
    return;
  }
  say('Replaying '+tree+' at '+speed+'x\u2026','#e3b341');
  fetch('/api/speed',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scale:speed})}).then(()=>
    fetch('/api/a2a/replay',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({tree_id:tree})}).then(r=>r.json())
  ).then(d=>{
    if(!d.run_id){say('Replay error: '+(d.error||'?'),'#f85149');return;}
    say('\u25B6 Replaying '+tree+' at '+speed+'x (run_id='+d.run_id
        +') \u2014 the arms are moving','#3fb950');
    pollRunId(d.run_id,'bt-replay-status');
  }).catch(e=>say('Replay failed: '+e,'#f85149'));
}
// Envelope contract from GET /a2a/result: {done, status:'succeeded'|'failed',
// result, error, ...}. The old check `d.result==='ok'` compared a dict to a
// string and was always false, so every completed run rendered as "Done: [object..".
function pollRunId(run_id,elId,max=60){let n=0;const iv=setInterval(()=>{fetch('/api/a2a/result/'+run_id).then(r=>r.json()).then(d=>{n++;const el=document.getElementById(elId);if(d.done){clearInterval(iv);if(el){const ok=d.status==='succeeded';const detail=d.error||(d.result&&d.result.result)||d.status||'';el.textContent=ok?'\u2713 Completed':'Done: '+String(detail).slice(0,80);el.style.color=ok?'#3fb950':'#e3b341';}}else if(n>=max){clearInterval(iv);if(el)el.textContent='\u23F1 Still running';}});},2000);}
{{ robot_graph_js|safe }}
const LAYOUT={
  "rai-orchestrator":      {cx:500,cy:70},
  "collaborative-planner": {cx:270,cy:215},
  "collision-monitor":     {cx:730,cy:215},
  "cell-controller":       {cx:500,cy:355},
  "rviz-web":              {cx:500,cy:635},
};
/* Per-robot execution nodes come from robots.yaml (any robot count) — injected
   as ROBOT_NODE_POS / ROBOT_EDGES / ROBOT_IDS by the server.
   Merged into LAYOUT, the table buildDiagram() actually reads. This said
   NODE_POS, a name nothing declares: a top-level ReferenceError, so the whole
   <script> aborted here and NOTHING below it ever ran — no diagram, no log
   polling, no chat. tests/test_dashboard_script_loads.py runs the served page. */
Object.assign(LAYOUT, (typeof ROBOT_NODE_POS !== "undefined") ? ROBOT_NODE_POS : {});
const ARROWS=[
  ["rai-orchestrator","collaborative-planner","Plan motion"],
  ["cell-controller","rviz-web","URDF + markers"],
].concat((typeof ROBOT_EDGES !== "undefined") ? ROBOT_EDGES : []);
/* reduced actor radius for narrower left column */
const R=40;
const CMD_RULES=[
  // ── Trajectory planning ────────────────────────────────────────────────────
  [/CALL:.*plan_parallel_motion/,  "cb-plan","Trajectory planning: computing collision-free paths for both robots in parallel"],
  [/plan_parallel_motion.*Dispat/,"cb-plan","Trajectory computed — dispatching to robot controllers"],
  [/CALL:.*optimize_weld_sequence/,"cb-plan","Weld sequencing: optimising order for least arm movement"],
  [/CALL:.*check_thermal_spread/,  "cb-plan","Thermal validation: checking heat spread between consecutive welds"],
  [/CALL:.*balance_cycle_time/,    "cb-plan","Load balancing: distributing weld points to equalise cycle time"],
  [/CALL:.*plan_motion_skill/,     "cb-plan","Single-robot trajectory: IK + RRT path planning"],
  [/Solving IK|IK.*solved|Pinocchio.*IK|_compute_ik/i,"cb-plan","Inverse kinematics: computing joint angles for target pose"],
  [/Space.Time RRT|RRT.*expand|RRT.*node/i,"cb-plan","Space-Time RRT: exploring collision-free paths in joint-time space"],
  [/conflicts_resolved|ConflictRes/i,"cb-plan","Conflict resolution: adjusting timing to prevent robot collisions"],
  [/Dispatched.*batch.*robot1/,    "cb-move","Sending trajectory to robot1 controller"],
  [/Dispatched.*batch.*robot2/,    "cb-move","Sending trajectory to robot2 controller"],
  [/Dispatched.*robot1/,           "cb-move","robot1 ← trajectory dispatched, moving to weld point"],
  [/Dispatched.*robot2/,           "cb-move","robot2 ← trajectory dispatched, moving to weld point"],
  // ── Execution states ───────────────────────────────────────────────────────
  [/\bCOMPLETED\b/,                "cb-move","Execution complete — robot reached weld target ✓"],
  [/\bEXECUTING\b/,                "cb-move","Execution active — robot in motion along trajectory"],
  [/\bRECEIVED\b/,                 "cb-move","Controller acknowledged task — validating trajectory"],
  [/\bPLANNING\b/,                 "cb-plan","On-demand planning: computing trajectory via Pinocchio/OMPL"],
  [/FAILED_PLANNING/,              "cb-safe","Planning failed — no IK solution found for target pose"],
  [/ABORTED_COLLISION/,            "cb-safe","Emergency stop — proximity violation detected"],
  [/COLLISION.*ALERT/i,            "cb-safe","Safety alert — robots approaching minimum clearance threshold"],
  // ── Cell infrastructure ────────────────────────────────────────────────────
  [/CALL:.*wait_seconds/,          "cb-move","Welding dwell: holding at weld point for burn-in time"],
  [/CALL:.*home_robots/,           "cb-move","Homing: all robots returning to safe home configuration"],
  [/robot_description|URDF/,       "cb-cell","Cell controller: broadcasting robot URDF to all containers"],
  [/Pinocchio.*loaded|model.*loaded/i,"cb-plan","Robot model loaded: Pinocchio kinematic model ready"],
  [/WELDING.*COMPLET/i,            "cb-move","Cycle complete — all weld points executed ✓"],
  [/Agentic plan recorded/,        "cb-plan","Plan saved to recordings — available for replay"],
];
function enrichCmd(msg){for(const [re,b,t] of CMD_RULES)if(re.test(msg))return{badge:b,text:t};return null;}
function actorOf(msg){
  const map=[
    [/orchestrator|agentic|bedrock|llm|LLM/i,"rai-orchestrator"],
    [/collaborative.planner|RRT|pinocchio.*ik|IK.*pino|conflicts_resolved/i,"collaborative-planner"],
    [/collision.*monitor|COLLISION.*ALERT/i,"collision-monitor"],
    [/cell.state|cell.controller|robot_description|URDF|CellState/i,"cell-controller"],
    [/rviz|sim.controller/i,"rviz-web"],
  ];
  /* Per-robot rules built from robots.yaml, so a 3rd/4th arm routes too. Checked
     BEFORE the generic rules: a line naming a specific robot belongs to it. */
  const rids=(typeof ROBOT_IDS!=="undefined")?ROBOT_IDS:[];
  for(const rid of rids){
    const esc=rid.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");
    const re=new RegExp(esc+".*RECEIV|"+esc+".*EXEC|"+esc+".*COMPLET|Dispatched.*"+esc+"|\\["+esc+"\\]","i");
    if(re.test(msg))return "robot-agent-"+rid;
  }
  for(const [re,id] of map)if(re.test(msg))return id;
  return null;
}
function buildDiagram(){
  if(!TOPOLOGY.length)return;
  const svg=document.getElementById("arch-svg");
  const W=svg.clientWidth||600,H=svg.clientHeight||680;
  const sx=W/1000,sy=H/720;
  const lg=document.getElementById("layers-g");lg.innerHTML="";
  const bands=[
    {y:0,  h:145,label:"Orchestration"},
    {y:145,h:155,label:"Planning  ·  Safety"},
    {y:300,h:155,label:"Cell Control"},
    {y:455,h:160,label:"Execution"},
    {y:615,h:105,label:"Visualization"},
  ];
  bands.forEach((b,i)=>{
    const rc=document.createElementNS("http://www.w3.org/2000/svg","rect");
    rc.setAttribute("x","0");rc.setAttribute("y",b.y*sy);
    rc.setAttribute("width",W);rc.setAttribute("height",b.h*sy);
    rc.setAttribute("fill",i%2?"#0f1117":"#0d1117");
    rc.setAttribute("stroke","#1e2330");rc.setAttribute("stroke-width","1");
    lg.appendChild(rc);
    const tx=document.createElementNS("http://www.w3.org/2000/svg","text");
    tx.setAttribute("x","8");tx.setAttribute("y",(b.y*sy+13));
    tx.setAttribute("class","layer-title");tx.textContent=b.label;lg.appendChild(tx);
  });
  const ag=document.getElementById("arrows-g");ag.innerHTML="";
  ARROWS.forEach(([from,to,label])=>{
    const lf=LAYOUT[from],lt=LAYOUT[to];if(!lf||!lt)return;
    const x1=lf.cx*sx,y1=lf.cy*sy,x2=lt.cx*sx,y2=lt.cy*sy;
    const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy);
    const ux=dx/len,uy=dy/len,pad=R*Math.min(sx,sy)+5;
    const ax1=x1+ux*pad,ay1=y1+uy*pad,ax2=x2-ux*(pad+8),ay2=y2-uy*(pad+8);
    const mx=(ax1+ax2)/2+uy*20,my=(ay1+ay2)/2-ux*20;
    const p=document.createElementNS("http://www.w3.org/2000/svg","path");
    p.setAttribute("id","arrow-"+from+"-"+to);
    p.setAttribute("d","M"+ax1+","+ay1+" Q"+mx+","+my+" "+ax2+","+ay2);
    p.setAttribute("class","flow-arrow");ag.appendChild(p);
    const t=document.createElementNS("http://www.w3.org/2000/svg","text");
    t.setAttribute("x",(ax1+ax2)/2+uy*28);t.setAttribute("y",(ay1+ay2)/2-ux*28);
    t.setAttribute("class","flow-label");t.textContent=label;ag.appendChild(t);
  });
  const acg=document.getElementById("actors-g");acg.innerHTML="";
  TOPOLOGY.forEach(agent=>{
    const pos=LAYOUT[agent.id];if(!pos)return;
    const cx=pos.cx*sx,cy=pos.cy*sy,r=R*Math.min(sx,sy);
    const g=document.createElementNS("http://www.w3.org/2000/svg","g");
    g.setAttribute("id","actor-"+agent.id);
    const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
    c.setAttribute("cx",cx);c.setAttribute("cy",cy);c.setAttribute("r",r);
    c.setAttribute("class","actor-ring");g.appendChild(c);
    const ic=document.createElementNS("http://www.w3.org/2000/svg","text");
    ic.setAttribute("x",cx);ic.setAttribute("y",cy-3);
    ic.setAttribute("class","actor-icon");ic.textContent=agent.icon;g.appendChild(ic);
    const lb=document.createElementNS("http://www.w3.org/2000/svg","text");
    lb.setAttribute("x",cx);lb.setAttribute("y",cy+r+14);
    lb.setAttribute("class","actor-label");
    lb.textContent=agent.label.replace(" · "," ");g.appendChild(lb);
    g.addEventListener("mouseenter",ev=>showTip(agent,ev));
    g.addEventListener("mouseleave",()=>{document.getElementById("tip").style.display="none";});
    acg.appendChild(g);
  });
}
function showTip(agent,ev){
  const tip=document.getElementById("tip");
  const inE=Object.entries(agent.topics_in||{}),outE=Object.entries(agent.topics_out||{});
  let h="<h4>"+agent.icon+" "+agent.label+"</h4>";
  h+="<div style='color:#6e7681;margin-bottom:7px;font-size:10px'>"+agent.description+"</div>";
  if(inE.length){h+="<div class='tin'>&#x2190; Receives</div>";inE.forEach(([,a])=>h+="<div class='trow'><span class='tarrow'>&#x2190;</span><span>"+a+"</span></div>");}
  if(outE.length){h+="<div class='tout'>&#x2192; Sends</div>";outE.forEach(([,a])=>h+="<div class='trow'><span class='tarrow'>&#x2192;</span><span>"+a+"</span></div>");}
  tip.innerHTML=h;tip.style.display="block";
  tip.style.left=Math.min(ev.clientX+14,window.innerWidth-280)+"px";
  tip.style.top=Math.min(ev.clientY+14,window.innerHeight-220)+"px";
}
function blink(id){
  const g=document.getElementById("actor-"+id);if(!g)return;
  const ring=g.querySelector(".actor-ring");if(!ring)return;
  ring.style.animation="none";void ring.offsetWidth;
  ring.style.animation="ring-pulse 4s ease-out forwards";
  ARROWS.filter(a=>a[0]===id||a[1]===id).forEach(([f,t])=>{
    const p=document.getElementById("arrow-"+f+"-"+t);
    if(p){p.setAttribute("stroke","#f0883e88");setTimeout(()=>p.setAttribute("stroke","#30363d"),1800);}
  });
}
/* Fire-and-forget is what made "the buttons do nothing" impossible to tell from
   "the buttons worked": no .then, no .catch, so a 500 or an unreachable
   orchestrator looked exactly like success. Every press now reports. */
function cmd(c){
  const st=$("cmd-status");
  const say=(txt,col)=>{if(st){st.textContent=txt;st.style.color=col;}};
  const body={command:c};
  if(c==="planning"){
    /* The selected task, sent explicitly. Refusing here rather than falling back
       to "the first program" keeps the plan the operator asked for the only plan
       that can happen. */
    const sel=$("plan-task");
    const program=sel?sel.value:"";
    if(!program)return say("✗ pick a task to plan first","#f85149");
    body.program=program;
  }
  /* Reset drives the arms home AND clears a latched e-stop to do it, so it is
     confirmed. Nothing else here asks — Stop only ends an activity, and an
     emergency stop must never be one click slower than the emergency. */
  if(c==="reset"&&!confirm("Reset: stop the current activity and drive the arms "
      +"HOME.\n\nThis MOVES the robots and CLEARS a latched e-stop. Continue?"))
    return say("reset cancelled","#6e7681");
  say("… "+c,"#e3b341");
  fetch("/api/command",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)})
    .then(r=>r.text().then(t=>{
      if(!r.ok)throw new Error("HTTP "+r.status+" — "+t.replace(/<[^>]*>/g," ").trim().slice(0,160));
      try{return JSON.parse(t);}catch(_){throw new Error("non-JSON reply: "+t.slice(0,160));}
    }))
    .then(d=>{
      if(d.error)return say("✗ "+c+": "+String(d.error).slice(0,160),"#f85149");
      /* Show it at once rather than on the next 1.5 s poll: the gap between the
         click and the button changing colour is exactly when it gets clicked
         again. This is not a guess — the run was accepted. */
      if(c==="planning"&&d.dispatched){_runStatus="planning";renderRunState();}
      say("✓ "+c+(d.dispatched?" dispatched"+(d.program?" ("+d.program+")":"")+" — follow it in the log"
                                    :" acknowledged"),"#3fb950");
    })
    .catch(e=>say("✗ "+c+" failed: "+(e.message||e),"#f85149"));
}
/* The Planning selector's options. Fetched once at load from the orchestrator's
   own task list, so a program that is not runnable cannot be offered. An empty
   or unreachable list leaves the button unusable rather than guessing a task —
   planning the wrong program is minutes of work aimed at the wrong welds. */
function loadPlanTasks(){
  const sel=$("plan-task");
  if(!sel)return;
  fetch("/api/a2a/tasks").then(r=>r.json()).then(d=>{
    const progs=(d&&d.programs)||[];
    sel.innerHTML="";
    if(!progs.length){
      sel.innerHTML='<option value="">no task programs found</option>';
      return;
    }
    progs.forEach(p=>{
      const name=(typeof p==="string")?p:(p.program||p.name||"");
      if(!name)return;
      const o=document.createElement("option");
      o.value=name;
      o.textContent=name+((typeof p==="object"&&p.num_operations)
        ?" ("+p.num_operations+" ops)":"");
      sel.appendChild(o);
    });
  }).catch(e=>{sel.innerHTML='<option value="">tasks unavailable</option>';});
}
/* ── Live simulation view (noVNC → the rviz-web container) ───────────────── */
const NOVNC_PORT="{{ novnc_port }}";
function vncUrl(){
  return location.protocol+"//"+location.hostname+":"+NOVNC_PORT
    +"/vnc.html?autoconnect=1&reconnect=1&resize=scale&show_dot=1";
}
function reloadVnc(){
  const f=$("vnc-frame"),st=$("vnc-state"),down=$("vnc-down");
  if(!f)return;
  if(st){st.textContent="connecting…";st.style.color="#6e7681";}
  if(down)down.style.display="none";
  f.style.display="block";
  f.src=vncUrl();
  const link=$("vnc-link");if(link)link.href=vncUrl();
  /* An iframe gives no readable error for a refused connection (cross-origin),
     so the viewer is probed separately and the reason is stated in-panel rather
     than leaving a black rectangle to be interpreted. */
  fetch(location.protocol+"//"+location.hostname+":"+NOVNC_PORT+"/vnc.html",{mode:"no-cors"})
    .then(()=>{if(st){st.textContent="connected";st.style.color="#3fb950";}})
    .catch(()=>{
      if(st){st.textContent="unreachable";st.style.color="#f85149";}
      if(down){
        down.style.display="block";
        down.innerHTML="The RViz viewer is not answering on port "+NOVNC_PORT+"."
          +"<br>It runs in the <b>rviz-web</b> container, which is behind the "
          +"<code>rviz</code> compose profile:<br>"
          +"<code style='color:#58a6ff'>docker compose --profile rviz up -d rviz-web</code>";
        f.style.display="none";
      }
    });
}

// ── Speed control ─────────────────────────────────────────────────────────────
let _loopActive = false;

function updateSpeed(val) {
  val = parseFloat(val);
  const lbl=$("speed-val"), sl=$("speed-slider"), btn=$("speed-toggle");
  if(lbl) lbl.textContent = val.toFixed(1) + "\xD7";
  if(sl) sl.value = val;
  if(btn) btn.textContent = val > 1 ? "\u23EA Slow" : "\u23E9 Fast";
  fetch("/api/speed",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({scale:val})});
}

function toggleSpeed() {
  const sl=$("speed-slider"); if(!sl) return;
  updateSpeed(parseFloat(sl.value) > 1.0 ? 0.5 : 2.0);
}

// ── Loop replay ───────────────────────────────────────────────────────────────
function setLoopBtn() {
  const btn=$("loop-btn"); if(!btn) return;
  btn.textContent = _loopActive ? "\u23F9 Stop" : "\u25B6 Start";
  btn.style.background = _loopActive ? "#b62324" : "#21262d";
  btn.style.color = _loopActive ? "#fff" : "#c9d1d9";
}
function toggleLoop() {
  _loopActive = !_loopActive;
  const iv=$("loop-interval");
  const interval = iv ? parseFloat(iv.value) : 10;
  setLoopBtn();
  fetch("/api/loop",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({enabled:_loopActive, interval_s:interval})})
    .then(r => r.json().then(d => ({ok: r.ok && d.ok !== false, d})))
    .then(({ok, d}) => {
      if (!ok) { _loopActive = false; setLoopBtn();
                 addChatMsg("assistant", "\u26A0 " + (d.error || "loop replay refused")); return; }
      addChatMsg("assistant", _loopActive
        ? "\u25B6 Loop replay started (every " + interval + "s)"
        : "\u23F9 Loop replay stopped");
    })
    .catch(e => addChatMsg("assistant", "\u26A0 loop request failed: " + e));
}
/* ── OVERALL RUN STATE ───────────────────────────────────────────────────────
   One renderer for every place that reports it: the header dot, the blinking
   message next to it, the Planning button, the Agents tab's dot and the overall
   line inside that pane. A planning run takes MINUTES and moves nothing, and the
   panel used to look identical whether one was in flight or not — /api/state
   answered "planning", updateUI wrote it into a dot-planning class no rule
   matched, and the button that started the run stayed exactly as green as before
   the click. So it got pressed again.

   Two inputs, one indicator. _runStatus is this panel's own state machine (GET
   /api/state), _planRunning is the planner's own answer from the event feed (GET
   /api/a2a/plan_events → running), which is the only one of the two that sees a
   run started from the chat, from an MCP client or from `make task`. The feed
   polls only while the Agents tab is open, so it may ADD planning to an idle
   status and never take it away — an indicator that goes quiet because nobody is
   looking at the right tab would be worse than none. */
let _runStatus="idle",_planRunning=false;
const RUN_MSG={
  planning:"⚙ PLANNING IN PROGRESS — trajectories only, nothing moves",
  homing:"⚙ RETURNING HOME — the arms are moving",
  estopped:"⛔ E-STOP LATCHED — Reset to clear it"};
const RUN_TINT={planning:"#d29922",homing:"#58a6ff",estopped:"#f85149"};
function effectiveStatus(){
  return (_planRunning&&(!_runStatus||_runStatus==="idle"))?"planning":(_runStatus||"idle");
}
function renderRunState(){
  const st=effectiveStatus(),msg=RUN_MSG[st]||"",tint=RUN_TINT[st]||"#6e7681";
  /* Every lookup guarded: this runs on the /api/state poll, and one null here
     used to take the whole handler — log panel included — down with it. */
  const dot=$("status-dot");
  if(dot)dot.className="sdot dot-"+st+(msg?" blink":"");
  const txt=$("status-text");
  if(txt)txt.textContent=st.toUpperCase();
  const bar=$("status-msg");
  if(bar){bar.textContent=msg;bar.className="busy-msg"+(msg?" blink":"");
          bar.style.color=tint;}
  const btn=$("btn-planning");
  if(btn){
    const running=st==="planning";
    btn.className="btn btn-ag"+(running?" btn-busy blink":"");
    btn.textContent=running?"\u{1F9E9} Planning…":"\u{1F9E9} Planning";
    btn.title=running
      ?"A planning run is in flight — follow it in the Agents tab. Pressing "
       +"this again starts a SECOND run."
      :"Calculate trajectories for the selected task. Plans only — nothing moves.";
  }
  const tdot=$("tab-agents-dot");
  if(tdot)tdot.className="sdot dot-"+st+(msg?" blink":"");
  const note=$("agents-overall");
  if(note){note.textContent="Overall status: "+st.toUpperCase()+(msg?"  ·  "+msg:"");
           note.className="busy-msg"+(msg?" blink":"");
           note.style.color=tint;}
}
let lastCount=0;const cmdFeed=[];
function updateUI(){
  fetch("/api/state").then(r=>r.json()).then(data=>{
    _runStatus=String(data.status||"idle");
    renderRunState();
    // Sync speed slider with server state. Guarded, and everything that can
    // render is below: this was two unguarded lookups of controls the markup no
    // longer had, so the TypeError landed here and the log panel \u2014 rendered
    // further down the same handler \u2014 never got written.
    if (data.velocity_scale !== undefined) {
      const s=data.velocity_scale, sl=$("speed-slider"), lbl=$("speed-val");
      if (sl) sl.value = s;
      if (lbl) lbl.textContent = s.toFixed(1) + "\xD7";
    }
    // Sync loop button with server state
    if (data.loop_active !== undefined && data.loop_active !== _loopActive) {
      _loopActive = data.loop_active;
      setLoopBtn();
    }
    const now=Date.now()/1000;
    Object.entries(data.agent_activity||{}).forEach(([id,ts])=>{if(now-ts<30)blink(id);});
    const log=data.planning_log||[];
    if(log.length!==lastCount){
      log.slice(lastCount).forEach(e=>{
        const actor=actorOf(e.msg);if(actor)blink(actor);
        const rich=enrichCmd(e.msg);
        if(rich)cmdFeed.unshift({time:e.time,badge:rich.badge,text:rich.text});
      });
      lastCount=log.length;
      const logEl=document.getElementById("log-container");
      const newStart=lastCount-log.slice(-80).length;
      logEl.innerHTML=log.slice(-80).map((e,idx)=>{
        const actor=actorOf(e.msg);
        const aTag=actor?"<span class='le-actor'>"+actor.replace("robot-agent-","")+"</span>":"";
        const ddsBadge=actor?"<span class='dds-badge'>DDS</span>":"";
        const clean=e.msg.replace(/\[interactive_orchestrator\]|\[collaborative_planner\]/g,"")
          .replace(/\[LLM\]\s*/g,"").substring(0,110);
        const isNew=(log.length-log.slice(-80).length+idx)>=lastCount-1;
        const cls="le le-"+e.level+(actor?" le-dds":"")+(isNew?" le-new":"");
        return "<div class='"+cls+"'><span class='le-time'>"+e.time+"</span>"+ddsBadge+aTag+"<span class='le-msg'>"+clean+"</span></div>";
      }).join("");
      document.getElementById("log-wrap").scrollTop=9999;
      document.getElementById("cmd-log").innerHTML=cmdFeed.slice(0,12).map(c=>
        "<div class='cmd-entry'><span class='cmd-time'>"+c.time+"</span><span class='cmd-badge "+c.badge+"'>●</span><span>"+c.text+"</span></div>"
      ).join("");
    }
    (data.collision_alerts||[]).forEach(a=>{
      if(!cmdFeed.find(e=>e.text===a))
        cmdFeed.unshift({time:new Date().toTimeString().slice(0,8),badge:"cb-safe",text:a});
    });
  });
}
/* ── Resizable columns (horizontal) + rows (vertical) ───────────────── */
function initResizable(){
  // Horizontal column dividers
  document.querySelectorAll('.divider').forEach((div,idx)=>{
    let dragging=false,startX=0;
    div.addEventListener('mousedown',e=>{dragging=true;startX=e.clientX;e.preventDefault();document.body.style.cursor='col-resize';});
    document.addEventListener('mousemove',e=>{
      if(!dragging)return;
      const main=document.querySelector('.main');
      const cols=getComputedStyle(main).gridTemplateColumns.split(' ');
      const delta=e.clientX-startX;startX=e.clientX;
      if(idx===0){
        const c1=Math.max(180,parseInt(cols[0])+delta);
        main.style.gridTemplateColumns=c1+'px 4px '+cols[2]+' 4px 1fr';
      }else{
        const c2=Math.max(180,parseInt(cols[2])+delta);
        main.style.gridTemplateColumns=cols[0]+' 4px '+c2+'px 4px 1fr';
      }
      buildDiagram();
    });
    document.addEventListener('mouseup',()=>{dragging=false;document.body.style.cursor='';});
  });
  // Vertical row dividers inside center-panel and chat-panel
  document.querySelectorAll('.vdivider').forEach(vdiv=>{
    let dragging=false,startY=0;
    vdiv.addEventListener('mousedown',e=>{dragging=true;startY=e.clientY;e.preventDefault();document.body.style.cursor='row-resize';});
    document.addEventListener('mousemove',e=>{
      if(!dragging)return;
      const panel=vdiv.parentElement;
      const rows=getComputedStyle(panel).gridTemplateRows.split(' ');
      const delta=e.clientY-startY;startY=e.clientY;
      const which=vdiv.dataset.which;
      if(which==='top'){
        const r0=Math.max(60,parseInt(rows[0])+delta);
        panel.style.gridTemplateRows=r0+'px 6px '+rows[2]+' 6px '+rows[4];
      }else{
        const r2=Math.max(60,parseInt(rows[2])+delta);
        panel.style.gridTemplateRows=rows[0]+' 6px '+r2+'px 6px 1fr';
      }
    });
    document.addEventListener('mouseup',()=>{dragging=false;document.body.style.cursor='';});
  });
}
/* ── Chat panel ──────────────────────────────────────────────────────── */
const CHIPS=["Cell status","Robot positions","Last trajectory","Describe task","List recordings","Run agentic","Stop"];
function initChat(){
  const chips=document.getElementById("chat-chips");
  CHIPS.forEach(c=>{
    const btn=document.createElement("button");
    btn.textContent=c;
    btn.style.cssText="background:#21262d;border:1px solid #30363d;border-radius:4px;padding:3px 8px;font-size:10px;color:#c9d1d9;cursor:pointer";
    btn.onclick=()=>{document.getElementById("chat-input").value=c;sendChat();};
    chips.appendChild(btn);
  });
  addChatMsg("assistant","Hello! I'm the cell assistant. Ask me about status, trajectories, or the welding program.");
}
/* Two generations of recordings browser lived here and are both gone: first a
   loadRecordings()/replaySelected() pair that filled a #rec-select the tabbed
   layout had already removed (so both threw on null), then the Rec tab's
   loadRecTab/selectRec. Replay is now btReplay() in the BTree tab, driven by the
   list of CALCULATED plans rather than the list of past executions. */
function addChatMsg(role,text){
  const div=document.createElement("div");
  div.style.cssText=role==="user"
    ?"align-self:flex-end;background:#1f6feb22;border:1px solid #1f6feb44;border-radius:8px;padding:7px 10px;font-size:11px;max-width:85%;white-space:pre-wrap"
    :"align-self:flex-start;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:7px 10px;font-size:11px;max-width:90%;white-space:pre-wrap;line-height:1.5";
  div.innerHTML=text.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>");
  document.getElementById("chat-messages").appendChild(div);
  document.getElementById("chat-messages").scrollTop=99999;
}
function sendChat(){
  const input=document.getElementById("chat-input");
  const msg=input.value.trim();if(!msg)return;
  addChatMsg("user",msg);input.value="";
  const thinking=document.createElement("div");
  thinking.style.cssText="align-self:flex-start;color:#6e7681;font-size:11px;padding:4px";
  thinking.textContent="\u2026";document.getElementById("chat-messages").appendChild(thinking);
  fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg})})
    /* Read as text first: a Flask traceback is an HTML page, and r.json() on it
       reports 'Unexpected token <' instead of the status that would explain it. */
    .then(r=>r.text().then(t=>{
      if(!r.ok)throw new Error("HTTP "+r.status+" — "+t.replace(/<[^>]*>/g," ").trim().slice(0,200));
      try{return JSON.parse(t);}catch(_){throw new Error("Non-JSON reply: "+t.slice(0,200));}
    })).then(d=>{
      thinking.remove();
      addChatMsg("assistant",d.reply||"No response.");
    }).catch(e=>{thinking.remove();addChatMsg("assistant","Error: "+(e.message||e));});
}
/* ── Boot ────────────────────────────────────────────────────────────── */
fetch("/api/topology").then(r=>r.json()).then(data=>{
  TOPOLOGY=data;buildDiagram();
  window.addEventListener("resize",buildDiagram);
}).catch(e=>{
  const w=$("agents-wrap");
  if(w)w.innerHTML="<div style='color:#f85149;padding:20px'>Topology failed: "+e+"</div>";
});
initResizable();
initChat();
reloadVnc();
/* Escape closes the performance window. Registered on the document, not the
   sheet: a dialog you can only dismiss with the mouse is one an operator with a
   hand on the keyboard leaves open over the tree. */
document.addEventListener('keydown',kpiKey);
loadPlanList();
loadPlanTasks();
setInterval(updateUI,1500);updateUI();
</script>
</body>
</html>
"""



# ─── The task wizard: its own page, in its own window ─────────────────────────
#
# Served by GET /wizard and opened by the Operator tab's "Wizard" button. It is a
# separate document on purpose: the operator keeps the live cell view on screen
# while authoring, and a mistake in the wizard's script cannot blank the panel
# (the panel is one inline <script> — see tests/test_dashboard_script_loads.py).
#
# It ends by WRITING THE TASK FILE through POST /api/a2a/task/save. That endpoint
# runs nothing: authoring a program and moving arms are separate decisions, and
# the operator gets the file on disk without a run being started behind their
# back. Planning/execution stay explicit clicks in the panel afterwards.
WIZARD_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Task Wizard &mdash; {{ cell_id }}</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px}
 .wrap{max-width:760px;margin:0 auto;padding:18px}
 h1{font-size:16px;margin:0 0 2px}
 .sub{color:#8b949e;font-size:11px;margin-bottom:14px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:12px}
 .q{font-size:13px;font-weight:700;color:#58a6ff;margin-bottom:4px}
 .hint{color:#8b949e;font-size:11px;margin-bottom:8px;line-height:1.45}
 label{display:block;font-size:11px;color:#8b949e;margin:8px 0 3px}
 input,select,textarea{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:7px;color:#c9d1d9;font-size:12px;outline:none}
 input:focus,select:focus,textarea:focus{border-color:#58a6ff}
 textarea{font-family:monospace;font-size:11px;resize:vertical}
 .row{display:flex;gap:8px}.row>div{flex:1}
 button{border-radius:6px;padding:8px 14px;font-size:12px;font-weight:700;cursor:pointer;border:1px solid #30363d;background:#21262d;color:#c9d1d9}
 button.primary{background:#1f6feb;color:#fff;border:none}
 button.go{background:#1a7f37;color:#fff;border:none}
 button.ghost{background:transparent;color:#8b949e}
 .nav{display:flex;gap:8px;align-items:center}
 .nav .spacer{flex:1}
 .steps{display:flex;gap:6px;margin-bottom:12px;font-size:10px;color:#6e7681}
 .steps span{padding:3px 8px;border:1px solid #30363d;border-radius:99px}
 .steps span.on{color:#fff;background:#1f6feb;border-color:#1f6feb}
 table{width:100%;border-collapse:collapse;font-size:11px}
 th{text-align:left;color:#6e7681;font-weight:600;padding:4px 6px;border-bottom:1px solid #30363d}
 td{padding:4px 6px;border-bottom:1px solid #21262d;font-family:monospace}
 .none{color:#6e7681;font-size:11px;padding:8px 0}
</style></head><body>
<div class="wrap">
  <h1>&#x1F9ED; Task Wizard</h1>
  <div class="sub">Cell <b>{{ cell_id }}</b> &middot; answer the questions, and the wizard writes
    <code>cells/{{ cell_id }}/tasks/&lt;program&gt;.json</code>. Nothing is planned and no robot moves.</div>
  <div class="steps">
    <span id="wz-s1" class="on">1 &middot; Program</span><span id="wz-s2">2 &middot; Planner</span>
    <span id="wz-s3">3 &middot; Points</span><span id="wz-s4">4 &middot; Review</span><span id="wz-s5">5 &middot; Written</span>
  </div>

  <div class="card" id="wz-p1">
    <div class="q">What is this program called?</div>
    <div class="hint">It becomes the file name under <code>tasks/</code> and the name you pick in the
      Planning tab. Letters, digits, <code>.</code>, <code>_</code> and <code>-</code> only.</div>
    <label for="wz-program">Program name</label>
    <input id="wz-program" placeholder="door_frame_welding" autocomplete="off">
    <label for="wz-desc">Description (optional) &mdash; what this program is for</label>
    <input id="wz-desc" placeholder="15 structural welds on the inner door frame" autocomplete="off">
  </div>

  <div class="card" id="wz-p2" style="display:none">
    <div class="q">Which motion planner should calculate the trajectories?</div>
    <div class="hint">Leave it on the cell default unless you have a reason. There is no fallback
      between planners &mdash; the one bound here is the one that runs.</div>
    <label for="wz-backend">Trajectory backend</label>
    <select id="wz-backend">
      <option value="">Use the cell default (recommended)</option>
      <option value="ompl">ompl</option>
      <option value="rrt">rrt</option>
      <option value="ompl-compound">ompl-compound</option>
    </select>
  </div>

  <div class="card" id="wz-p3" style="display:none">
    <div class="q">Where are the operation points?</div>
    <div class="hint">One point per answer &mdash; fill it in and press <b>Add this point</b>, as many
      times as you need. Positions are metres in the cell/world frame. Which robot performs a point is
      decided at planning time; pin one only if only that robot can do it.</div>
    <div class="row">
      <div><label for="wz-op-id">Point id</label><input id="wz-op-id" placeholder="W01" autocomplete="off"></div>
      <div><label for="wz-op-type">Operation</label>
        <select id="wz-op-type">
          <option value="weld">weld</option><option value="rivet">rivet</option>
          <option value="glue">glue</option><option value="laser_cut">laser_cut</option>
          <option value="mark">mark</option><option value="move">move</option>
        </select></div>
    </div>
    <label>Position [x, y, z] in metres</label>
    <div class="row">
      <div><input id="wz-x" placeholder="x e.g. -1.082" autocomplete="off"></div>
      <div><input id="wz-y" placeholder="y e.g. -1.489" autocomplete="off"></div>
      <div><input id="wz-z" placeholder="z e.g. 1.60" autocomplete="off"></div>
    </div>
    <label for="wz-approach">How does the tool come in to the point?</label>
    <select id="wz-approach" onchange="wzApproachChanged()">
      <option value="0,0,-1">From above, straight down (0, 0, -1)</option>
      <option value="0,0,1">From below, straight up (0, 0, 1)</option>
      <option value="0,1,0">Along +Y (0, 1, 0)</option>
      <option value="0,-1,0">Along -Y (0, -1, 0)</option>
      <option value="1,0,0">Along +X (1, 0, 0)</option>
      <option value="-1,0,0">Along -X (-1, 0, 0)</option>
      <option value="custom">Custom direction&hellip;</option>
    </select>
    <div id="wz-custom-av" style="display:none">
      <label>Approach vector [ax, ay, az]</label>
      <div class="row">
        <div><input id="wz-ax" value="0" autocomplete="off"></div>
        <div><input id="wz-ay" value="0" autocomplete="off"></div>
        <div><input id="wz-az" value="-1" autocomplete="off"></div>
      </div>
    </div>
    <div class="row">
      <div><label for="wz-group">Sequence group (ordering, not robot choice)</label>
        <input id="wz-group" value="structural" autocomplete="off"></div>
      <div><label for="wz-prio">Priority within the group</label>
        <input id="wz-prio" value="1" autocomplete="off"></div>
    </div>
    <label for="wz-robot">Lock this point to one robot?</label>
    <select id="wz-robot">
      <option value="">No &mdash; let the planner allocate it (recommended)</option>
      {{ robot_options|safe }}
    </select>
    <div style="margin-top:12px"><button class="go" onclick="wzAddPoint()">&#x2795; Add this point</button></div>
    <div style="margin-top:14px">
      <div class="q" style="font-size:11px">Points added: <span id="wz-count">0</span></div>
      <div id="wz-points"><div class="none">No points yet &mdash; add at least one.</div></div>
    </div>
  </div>

  <div class="card" id="wz-p4" style="display:none">
    <div class="q">This is the task file. Write it?</div>
    <div class="hint">Edit it here if you want; what is shown is exactly what gets written. Writing
      the file does not plan it and does not move anything.</div>
    <textarea id="wz-json" rows="16"></textarea>
    <label style="display:flex;gap:7px;align-items:center;margin-top:10px;color:#e3b341">
      <input type="checkbox" id="wz-overwrite" style="width:auto"> Overwrite an existing program of this name
    </label>
  </div>

  <div class="card" id="wz-p5" style="display:none">
    <div class="q">Written.</div>
    <div id="wz-result" class="hint"></div>
  </div>

  <div class="nav">
    <button id="wz-back" class="ghost" onclick="wzBack()" style="display:none">&larr; Back</button>
    <div class="spacer"></div>
    <div id="wz-status" style="font-size:11px;color:#8b949e;margin-right:8px"></div>
    <button id="wz-next" class="primary" onclick="wzNext()">Next &rarr;</button>
    <button id="wz-write" class="go" onclick="wzWrite()" style="display:none">&#x1F4BE; Write task file</button>
    <button id="wz-close" onclick="window.close()" style="display:none">Close</button>
  </div>
</div>
<script>
const CELL_ID="{{ cell_id }}";
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const NAME_RE=/^[A-Za-z0-9._-]+$/;   // same rule the orchestrator enforces
const OPS=[];                        // the operations as they will be written
let STEP=1;
const LAST=5;

function say(txt,col){const el=$('wz-status');if(el){el.textContent=txt||'';el.style.color=col||'#8b949e';}}

function wzShow(n){
  STEP=n;
  for(let i=1;i<=LAST;i++){
    const pane=$('wz-p'+i);if(pane)pane.style.display=(i===n?'block':'none');
    const chip=$('wz-s'+i);if(chip)chip.className=(i===n?'on':'');
  }
  $('wz-back').style.display=(n>1&&n<LAST)?'inline-block':'none';
  $('wz-next').style.display=(n<4)?'inline-block':'none';
  $('wz-write').style.display=(n===4)?'inline-block':'none';
  $('wz-close').style.display=(n===LAST)?'inline-block':'none';
  if(n===4)$('wz-json').value=JSON.stringify(wzBuild(),null,2);
  say('');
}

function wzApproachChanged(){
  $('wz-custom-av').style.display=($('wz-approach').value==='custom')?'block':'none';
}

/* The approach direction as [ax,ay,az], or null if the custom boxes are not numbers. */
function wzApproach(){
  const v=$('wz-approach').value;
  if(v!=='custom')return v.split(',').map(Number);
  const a=['wz-ax','wz-ay','wz-az'].map(id=>parseFloat($(id).value));
  return a.some(n=>!isFinite(n))?null:a;
}

/* Suggest the next id in the series the operator has been using: W01 -> W02. */
function wzSuggestId(){
  const last=OPS.length?OPS[OPS.length-1].id:'';
  const m=/^([A-Za-z_-]*)(\\d+)$/.exec(last);
  if(!m){$('wz-op-id').value=OPS.length?'':'W01';return;}
  const n=String(parseInt(m[2],10)+1).padStart(m[2].length,'0');
  $('wz-op-id').value=m[1]+n;
}

function wzAddPoint(){
  const id=($('wz-op-id').value||'').trim();
  if(!id){say('Give the point an id (e.g. W01).','#f85149');return;}
  if(OPS.some(o=>o.id===id)){say('There is already a point called "'+id+'".','#f85149');return;}
  const pos=['wz-x','wz-y','wz-z'].map(k=>parseFloat($(k).value));
  if(pos.some(n=>!isFinite(n))){say('x, y and z must all be numbers (metres).','#f85149');return;}
  const av=wzApproach();
  if(!av){say('The custom approach vector must be three numbers.','#f85149');return;}
  const op={id:id,type:$('wz-op-type').value,position:pos,approach_vector:av};
  const grp=($('wz-group').value||'').trim();if(grp)op.sequence_group=grp;
  const prio=parseInt($('wz-prio').value,10);if(isFinite(prio))op.priority=prio;
  const rob=$('wz-robot').value;if(rob)op.assigned_robot=rob;
  OPS.push(op);
  wzRenderPoints();
  ['wz-x','wz-y','wz-z'].forEach(k=>{$(k).value='';});
  wzSuggestId();
  say('Added '+id+'. '+OPS.length+' point(s) so far.','#3fb950');
}

function wzDelPoint(i){OPS.splice(i,1);wzRenderPoints();say('Point removed.','#8b949e');}

function wzRenderPoints(){
  $('wz-count').textContent=OPS.length;
  const box=$('wz-points');
  if(!OPS.length){box.innerHTML="<div class='none'>No points yet &mdash; add at least one.</div>";return;}
  box.innerHTML="<table><tr><th>id</th><th>type</th><th>position</th><th>approach</th><th>group</th><th>robot</th><th></th></tr>"
    +OPS.map((o,i)=>"<tr><td>"+esc(o.id)+"</td><td>"+esc(o.type)+"</td><td>"
      +o.position.map(n=>n.toFixed(3)).join(', ')+"</td><td>"+o.approach_vector.join(', ')+"</td><td>"
      +esc(o.sequence_group||'-')+"</td><td>"+esc(o.assigned_robot||'planner')+"</td>"
      +"<td><button class='ghost' style='padding:1px 6px;font-size:10px' onclick='wzDelPoint("+i+")'>remove</button></td></tr>").join('')
    +"</table>";
}

/* The task program, in the schema the planner reads (task-json-format.md). */
function wzBuild(){
  const prog={program:($('wz-program').value||'').trim(),cell_id:CELL_ID};
  const desc=($('wz-desc').value||'').trim();if(desc)prog.description=desc;
  const be=$('wz-backend').value;if(be)prog.trajectory_backend=be;
  prog.operations=OPS;
  return prog;
}

function wzNext(){
  if(STEP===1){
    const name=($('wz-program').value||'').trim();
    if(!name){say('The program needs a name.','#f85149');return;}
    if(!NAME_RE.test(name)){say('Letters, digits, ".", "_" and "-" only \\u2014 it becomes a file name.','#f85149');return;}
    wzShow(2);return;
  }
  if(STEP===2){wzShow(3);wzSuggestId();return;}
  if(STEP===3){
    if(!OPS.length){say('Add at least one point \\u2014 a program with no operations cannot run.','#f85149');return;}
    wzShow(4);return;
  }
}
function wzBack(){if(STEP>1)wzShow(STEP-1);}

function wzWrite(){
  let body;
  try{body=JSON.parse($('wz-json').value);}
  catch(e){say('That is not valid JSON: '+e.message,'#f85149');return;}
  if($('wz-overwrite').checked)body.overwrite=true;
  say('Writing\\u2026','#e3b341');
  fetch('/api/a2a/task/save',{method:'POST',headers:{'Content-Type':'application/json'},
                              body:JSON.stringify(body)}).then(r=>r.json()).then(r=>{
    if(!r||r.error||r.success===false){
      const msg=(r&&(r.error||r.hint))||'the server did not say why';
      say('\\u2716 Not written: '+msg,'#f85149');
      if(String(msg).indexOf('already exists')>=0)
        say('\\u2716 '+msg+' Tick "Overwrite" above to replace it.','#f85149');
      return;
    }
    const text=JSON.stringify(body,null,2);
    $('wz-result').innerHTML="Wrote <code>"+esc(r.saved||'')+"</code> \\u2014 "
      +esc(String(r.num_operations||0))+" operation(s). Nothing has been planned and no robot has moved."
      +"<br>Plan it from the <b>Planning</b> tab, or press Submit &amp; Run in the Operator tab.";
    wzShow(5);
    say('\\u2713 Saved.','#3fb950');
    // Hand it back to the panel that opened us: fills its textarea and refreshes
    // the program list. Guarded \\u2014 /wizard opened directly has no opener.
    try{if(window.opener&&window.opener.opWizardSaved)
          window.opener.opWizardSaved(r.program||body.program,r.saved||'',text);}catch(e){}
  }).catch(e=>say('\\u2716 Not written: '+e,'#f85149'));
}

wzRenderPoints();   // the count reflects OPS from the first paint, not the markup
wzShow(1);
</script></body></html>
"""


# ─── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        cell_id=CELL_ID,
        novnc_port=NOVNC_PORT,
        robot_graph_js=robot_graph_js(),
        robot_options="".join(
            f'<option value="{rid}">{rid}</option>' for _ns, rid in _discover_robots()
        ),
    )


@app.route("/wizard")
def task_wizard():
    """The task wizard page, opened in its own window by the Operator tab.

    Same robot list as the panel, so a point can be pinned to a robot that
    actually exists in this cell.
    """
    return render_template_string(
        WIZARD_TEMPLATE,
        cell_id=CELL_ID,
        robot_options="".join(
            f'<option value="{rid}">Yes &mdash; always {rid}</option>'
            for _ns, rid in _discover_robots()
        ),
    )


@app.route("/health")
def health():
    """Quick health check — returns 200 if Flask is running."""
    import rclpy as _r
    ros_ok = ros_node is not None
    return jsonify({
        "ok":      True,
        "cell_id": CELL_ID,
        "ros":     ros_ok,
        "cell_base": os.environ.get("CELL_BASE_PATH", "/cells"),
    }), 200


@app.route("/api/topology")
def get_topology():
    # agent_topology() splices in one node per robot from robots.yaml.
    return jsonify(agent_topology())


@app.route("/api/state")
def get_state():
    # `recordings` stays in the response (it is part of this route's shape) and
    # stays EMPTY here. It used to be filled by globbing three directories for
    # `plan_*.json` on every 1.5 s poll — a filename the recorder stopped writing
    # when recordings became rerun logs keyed by tree_id, so the list was always
    # empty anyway, and nothing on the page reads it: the recordings browser it
    # once fed is gone, and Replay works off the plan list in the BTree tab.
    # The live list is GET /api/recordings, which asks the cell controller.
    # Include sanitised robot_states (strip large arrays if needed)
    state_out = dict(state)
    state_out["robot_states"] = {
        ns: {
            "joints":      rs.get("joints", []),
            "velocities":  rs.get("velocities", []),
            "ee_position": rs.get("ee_position"),
            "age_s":       round(time.time() - rs.get("stamp", 0), 1),
        }
        for ns, rs in state.get("robot_states", {}).items()
    }
    return jsonify(state_out)


@app.route("/api/command", methods=["POST"])
def send_command():
    data = flask_request.json or {}
    cmd = data.get("command", "")

    # PLANNING → orchestrator POST /a2a/plan {cell_id, program}. Plan only: the
    # finished tree is persisted and NOTHING is dispatched to a robot. Running it
    # is a separate, deliberate act (the Replay tab).
    #
    # Planning takes minutes, so it is DISPATCHED (`"async": true`) and polled:
    # the orchestrator answers with a run id immediately and the outcome is read
    # from /a2a/result. Holding an HTTP request open for the whole run is what
    # used to make this path report a timeout for every run that actually started.
    if cmd == "planning":
        program = (data.get("program") or "").strip()
        if not program:
            # No silent fallback to "the first program on disk": which welds get
            # planned is the operator's choice, and a default would hide it.
            return jsonify({"ok": False, "command": cmd,
                            "error": "no task selected — pick a task program to "
                                     "plan"}), 400

        def _run():
            _log(f"A2A PLAN → '{program}' (planning only, no motion; may take "
                 f"minutes)…", "plan", "rai-orchestrator")
            state["status"] = "planning"
            res, _st = _a2a_post("/a2a/plan", {"cell_id": CELL_ID,
                                              "program": program, "async": True})
            run_id = res.get("run_id")
            if not run_id:
                _log(f"A2A run not dispatched: {str(res.get('error'))[:160]}",
                     "error", "rai-orchestrator")
                state["status"] = "idle"
                return
            _log(f"A2A run dispatched — run_id={run_id}", "info", "rai-orchestrator")
            deadline = time.time() + RUN_POLL_TIMEOUT_S
            while time.time() < deadline:
                time.sleep(RUN_POLL_INTERVAL_S)
                envelope, poll_status = _a2a_get(f"/a2a/result/{run_id}")
                if poll_status == 404:
                    # The id stopped resolving: the orchestrator restarted
                    # mid-run and its registry is in-memory. Say so rather than
                    # polling a dead id until the deadline.
                    _log(f"A2A run {run_id} is no longer known to the "
                         f"orchestrator (did it restart?)", "warn", "rai-orchestrator")
                    break
                if envelope.get("done"):
                    result = envelope.get("result")
                    result = result if isinstance(result, dict) else {}
                    ok = (envelope.get("status") == "succeeded"
                          and result.get("success"))
                    detail = envelope.get("error") or result.get("error") or ""
                    tree = result.get("tree_id")
                    if ok and tree:
                        # The tree_id is the whole point of a plan-only run: it is
                        # what the Replay tab needs to execute it.
                        _log(f"PLAN READY — tree_id={tree} "
                             f"({result.get('num_operations', '?')} operations). "
                             f"Nothing has moved; replay it to execute.",
                             "plan", "rai-orchestrator")
                    else:
                        _log(f"A2A plan {result.get('status') or envelope.get('status')}"
                             f": {str(detail)[:200]}", "warn", "rai-orchestrator")
                    break
            else:
                # Stop watching, but do NOT claim the run ended: it may still be
                # executing, and the log saying "finished" would be a lie the
                # operator would act on.
                _log(f"stopped watching run {run_id} after "
                     f"{RUN_POLL_TIMEOUT_S / 60:.0f} min — it may still be running; "
                     f"check /api/a2a/result/{run_id}", "warn", "rai-orchestrator")
            state["status"] = "idle"

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True, "command": cmd, "program": program, "dispatched": True})

    if cmd == "estop":
        # EMERGENCY STOP. Published straight to ROS rather than through the
        # orchestrator's POST /a2a/stop, which exists and does the same thing via
        # the cell controller. This button takes the path with the fewest hops on
        # purpose: routing it through HTTP would make it fail whenever the
        # orchestrator is the thing that is wedged. A stop that depends on a
        # healthy agent is not a stop.
        #
        # It is the ONLY button here that halts an arm: the robot agents cancel
        # the FollowJointTrajectory goal in flight and LATCH, rejecting every
        # dispatch until the latch is cleared (Reset clears it, and so does the
        # start of a new execution).
        if ros_node:
            ros_node.estop_pub.publish(String(data=json.dumps(
                {"reason": "EMERGENCY STOP from dashboard",
                 "timestamp": time.time()})))
            _log("EMERGENCY STOP → /{}/estop — arms halting, cell LATCHED"
                 .format(CELL_ID), "error", "web-panel")
            state["status"] = "estopped"
            return jsonify({"ok": True, "command": cmd, "latched": True})
        # No ROS node means nothing was published — a hollow "ok" here would tell
        # the operator the cell is stopping when nothing heard the stop.
        _log("EMERGENCY STOP could not be published — this panel has no ROS "
             "node. Use the physical e-stop.", "error", "web-panel")
        return jsonify({"ok": False, "command": cmd,
                        "error": "no ROS node in this panel — nothing was "
                                 "published; use the physical e-stop"}), 503

    # STOP / RESET both stop the system's ACTIVITY through the orchestrator, which
    # is the only process that can reach both halves of it (the planning job in
    # the trajectory agent and the executing tree in the cell controller). Neither
    # halts an arm — that is the emergency stop above.
    if cmd in ("stop", "reset"):
        is_reset = cmd == "reset"
        payload = {"cell_id": CELL_ID,
                   "reason": f"operator {cmd} from dashboard"}
        if is_reset:
            payload["async"] = True
        res, st = _a2a_post("/a2a/reset" if is_reset else "/a2a/abort", payload)
        if st >= 400 or res.get("error"):
            _log(f"{cmd.upper()} failed: {str(res.get('error'))[:200]}",
                 "error", "rai-orchestrator")
            return jsonify({"ok": False, "command": cmd,
                            "error": res.get("error", f"HTTP {st}")}), 502

        if is_reset:
            # Loud, because it moves the arms and it clears a latched e-stop to do
            # it — an operator reading the log later must see both facts.
            _log("RESET → activity stopped; arms returning HOME (this CLEARS a "
                 "latched e-stop). The arms are moving.", "warn",
                 "rai-orchestrator")
            state["status"] = "homing"
            return jsonify({"ok": True, "command": cmd, "dispatched": True,
                            "run_id": res.get("run_id")})

        planning = (res.get("planning") or {}).get("cancelled") or []
        aborted = bool((res.get("execution") or {}).get("aborted"))
        what = ", ".join(
            ([f"planning ({len(planning)} job(s))"] if planning else [])
            + (["the executing plan"] if aborted else []))
        _log(f"STOP → {what or 'nothing was running'}. "
             f"The arms are NOT halted — a trajectory already sent finishes; "
             f"use EMERGENCY STOP to halt them.", "warn", "rai-orchestrator")
        state["status"] = "idle"
        return jsonify({"ok": True, "command": cmd,
                        "stopped_planning": planning, "stopped_execution": aborted,
                        "estop": False})

    if cmd == "status":
        # The Status BUTTON is gone from the controls (the panel polls /api/state
        # every 1.5 s, so it told the operator nothing they were not already
        # seeing). The command itself is kept as a round-trip probe for API
        # clients — curl, CI — and because
        # tests/test_command_buttons_are_accepted.py pins it as acked. It is
        # deliberately NOT in KNOWN_COMMANDS: nothing in the UI sends it.
        return jsonify({"ok": True, "command": cmd,
                        "note": "no-op probe; the panel polls /api/state"})

    # Anything else is rejected rather than acked. An unknown command used to fall
    # through to `{"ok": true}`, so a button whose name had drifted reported
    # success and did nothing — the same "the commands seem not to work" symptom
    # the front end was just fixed for, one layer down.
    return jsonify({"ok": False, "command": cmd,
                    "error": f"unknown command '{str(cmd)[:40]}' — expected one of "
                             f"{', '.join(KNOWN_COMMANDS)}"}), 400


@app.route("/api/replay", methods=["POST"])
def do_replay():
    """Trigger a replay of a specific recording via the orchestrator."""
    # This route published "replay <speed> <file>" on /{cell}/orchestrator/command,
    # a topic NOTHING subscribes to: it answered ok and nothing ran. Replay is an
    # orchestrator action (POST /api/a2a/replay with a tree_id); say so instead.
    return jsonify({"ok": False,
                    "error": ("replay by file/speed went out on /orchestrator/command, which "
                              "nothing reads — use the BTree tab (POST /api/a2a/replay with "
                              "a tree_id)")}), 410


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Natural language chat backed by the same LLM as the orchestrator.

    System context injected automatically:
      - Cell status (idle/running)
      - Live joint angles for each robot (from joint_states subscriptions)
      - End-effector Cartesian position computed via Pinocchio FK
      - Last planning summary
      - Recent log entries
      - Active task description

    Actions the LLM can trigger by including JSON action blocks in its response:
      {"action":"run"}  {"action":"stop"}  {"action":"reset"}
      {"action":"move","robot":"robot1","x":0.3,"y":0,"z":0.8}
      {"action":"create_weld","x":0.1,"y":0,"z":0.9}
    """
    import re as _re

    request_data = flask_request.json
    user_msg = (request_data.get("message") or "").strip()
    # Per-message model override, same vocabulary as the A2A/MCP 'model'
    # argument: an alias ("opus-5"), a full model id, or "provider:model".
    chat_model = (request_data.get("model") or "").strip() or None
    if not user_msg:
        return jsonify({"reply": "Please enter a message."})

    # ── Build system context ──────────────────────────────────────
    context_lines = [
        f"Cell: {CELL_ID}  Status: {state.get('status','idle').upper()}",
    ]

    # Live robot state from joint_states subscriptions
    robot_states = state.get("robot_states", {})
    if robot_states:
        context_lines.append("\nRobot states:")
        for ns, rs in robot_states.items():
            joints = rs.get("joints", [])
            joints_str = ", ".join(f"j{i+1}={v:.3f}" for i, v in enumerate(joints))
            ee = rs.get("ee_position")
            ee_str = f"EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})" if ee else "EE=unknown"
            context_lines.append(f"  {ns}: joints=[{joints_str}]  {ee_str}")
    else:
        context_lines.append("\nRobot states: not yet received (subscribing to joint_states topics)")

    # Planning summary
    plan_summary = state.get("current_plan_info", {}).get("summary", "")
    if plan_summary:
        context_lines.append(f"\nLast plan:\n{plan_summary}")

    # Recent events (last 8 lines)
    recent = state.get("planning_log", [])[-8:]
    if recent:
        context_lines.append("\nRecent events:")
        for e in recent:
            context_lines.append(f"  [{e['time']}] {e['msg'][:80]}")

    # What programs this cell has, asked of the trajectory agent that owns the
    # tasks/ directory. This block used to open `tasks/welding_task.json` — a
    # program name no cell in this repo defines — and count
    # `tasks[].parameters.points` where `operation_type == "WELDING"`, a schema
    # nothing has written since the format became `operations[]` of `type: weld`.
    # Both failures were silent: the missing file went into an except that logged
    # at DEBUG, so the model answered questions about the cell's programs having
    # been told nothing about them at all.
    catalogue = _mcp_call(_traj_mcp(), "list_task_programs", {"cell_id": CELL_ID})
    problem = _mcp_failed(catalogue)
    if problem:
        context_lines.append(f"\nTask programs: unavailable ({problem})")
    elif catalogue.get("programs"):
        ops = catalogue.get("operations") or {}
        by_type = ", ".join(f"{n} {t}"
                            for t, n in (ops.get("by_type") or {}).items())
        context_lines.append(
            f"\nTask programs ({catalogue.get('count', 0)}): "
            + ", ".join(str(p) for p in catalogue["programs"])
            + f"\n  {ops.get('total', 0)} operation(s) in total"
            + (f" — {by_type}" if by_type else "")
        )
    else:
        context_lines.append("\nTask programs: none defined for this cell yet")
    recs = _list_recordings()
    if recs:
        context_lines.append(f"\nAvailable recordings ({len(recs)}):")
        for r in recs[:5]:
            ops = r.get("operations")
            context_lines.append(
                f"  • {r['tree_id']}  "
                f"{f'{ops} operations  ' if ops else ''}{r.get('timestamp','')}"
            )

    system_context = "\n".join(context_lines)

    system_prompt = f"""You are the assistant for a multi-robot welding cell control panel.
You have real-time access to the cell state and can answer questions and trigger actions.

LIVE CELL STATE:
{system_context}

CAPABILITIES:
- Answer questions about robot positions, joint angles, cell status, trajectories, tasks
- Explain what is happening in plain language an operator can understand
- Trigger actions by including a JSON block at the END of your response (do not explain the JSON):
  To run agentic:         {{"action":"run"}}
  To stop:                {{"action":"stop"}}
  To reset:               {{"action":"reset"}}
  To replay last plan:    {{"action":"replay"}}
  To replay at 0.5x:      {{"action":"replay","speed":0.5}}
  To replay specific file:{{"action":"replay","filename":"plan_20240101_120000.json","speed":1.0}}
  To list recordings:     {{"action":"list_recordings"}}
  To move a robot:        {{"action":"move","robot":"robot1","x":0.3,"y":0.0,"z":0.8}}
  To add weld point:      {{"action":"create_weld","x":0.1,"y":0.0,"z":0.9}}

STYLE: Be concise, plain language, no jargon. If you trigger an action, briefly confirm it.
Joint angles are in radians. Positions are in metres, world frame, origin at cell centre.
When asked to replay, use the most recent recording unless the user specifies a filename."""

    # ── Call LLM with MCP skill tools ─────────────────────────────
    # Tries to bind MCP tools so the LLM can call planning/analysis
    # skills directly. Falls back to a plain LLM call if MCP is unavailable.
    try:
        llm_provider  = os.environ.get("LLM_PROVIDER", "bedrock").lower()
        mcp_base      = os.environ.get("MCP_BASE_URL",
                                        f"http://localhost:{os.environ.get('MCP_PORT','8765')}")
        a2a_base      = os.environ.get("A2A_BASE_URL",
                                        f"http://rai-orchestrator:{os.environ.get('A2A_PORT','8889')}")

        if llm_provider == "ollama":
            # Kept local: this branch needs langchain_ollama's base_url, which the
            # shared builder does not carry.
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=os.environ.get("OLLAMA_MODEL", "llama3"),
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
                temperature=0.3,
            )
        else:
            # Built by agents.llm_model so the panel talks to the same model as
            # the agents it is inspecting. It used to pin a hardcoded Sonnet 4.5
            # in eu-central-1 and report the agents' answers as if they came from
            # the configured model.
            #
            # No temperature/max_tokens: Claude 5 rejects temperature outright
            # (HTTP 400), and a max_tokens cap now bounds thinking + text
            # together, so a tight cap truncates the reply.
            from agents.llm_model import build_llm
            llm = build_llm(chat_model)

        # Try to load MCP tools from the skills server (SSE on port MCP_PORT)
        mcp_tools = []
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            import asyncio as _asyncio
            mcp_client = MultiServerMCPClient({
                "welding-skills": {
                    "url":       f"{mcp_base}/sse",
                    "transport": "sse",
                }
            })
            # get_tools() is a coroutine — run it synchronously
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    # Inside an existing event loop (e.g. Flask with async support)
                    import concurrent.futures as _cf
                    with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                        mcp_tools = _pool.submit(
                            _asyncio.run, mcp_client.get_tools()
                        ).result(timeout=5.0)
                else:
                    mcp_tools = loop.run_until_complete(mcp_client.get_tools())
            except RuntimeError:
                mcp_tools = _asyncio.run(mcp_client.get_tools())
        except Exception as _mcp_err:
            # MCP server not yet running or package not available — degrade gracefully
            import logging as _log
            _log.getLogger("web_control_panel").debug(
                f"[Chat] MCP tools not loaded: {_mcp_err}"
            )

        if mcp_tools:
            # Tool-aware ReAct loop — LLM can call skills directly
            from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
            import asyncio as _asyncio
            llm_with_tools = llm.bind_tools(mcp_tools)
            tool_map = {t.name: t for t in mcp_tools}
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_msg),
            ]
            reply_text = ""

            def _invoke_tool(fn, args):
                """Invoke an MCP tool — handles both sync and async tools."""
                import inspect
                if inspect.iscoroutinefunction(fn.func if hasattr(fn, 'func') else fn):
                    try:
                        loop = _asyncio.get_event_loop()
                        if loop.is_running():
                            import concurrent.futures as _cf
                            with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                                return _pool.submit(_asyncio.run, fn.ainvoke(args)).result(timeout=10.0)
                        else:
                            return loop.run_until_complete(fn.ainvoke(args))
                    except RuntimeError:
                        return _asyncio.run(fn.ainvoke(args))
                else:
                    try:
                        return fn.invoke(args)
                    except NotImplementedError:
                        # StructuredTool from MCP — use ainvoke via asyncio
                        try:
                            import concurrent.futures as _cf
                            with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                                return _pool.submit(_asyncio.run, fn.ainvoke(args)).result(timeout=10.0)
                        except Exception as _e:
                            return json.dumps({"error": f"Tool invocation failed: {_e}"})

            for _ in range(8):  # max 8 tool calls per chat turn
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                if not getattr(response, "tool_calls", None):
                    reply_text = _extract_text(response.content)
                    break
                for tc in response.tool_calls:
                    fn = tool_map.get(tc["name"])
                    if fn:
                        result = _invoke_tool(fn, tc["args"])
                    else:
                        result = json.dumps({"error": f"Unknown tool: {tc['name']}"})
                    messages.append(ToolMessage(content=str(result),
                                                tool_call_id=tc["id"]))
            if not reply_text:
                reply_text = _extract_text(response.content)
        else:
            # Plain LLM call — no tools
            from langchain_core.messages import HumanMessage, SystemMessage
            response   = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_msg),
            ])
            reply_text = _extract_text(response.content)

    except Exception as e:
        import traceback as _tb
        import logging as _logging
        _logging.getLogger("web_control_panel").error(
            f"[Chat] LLM call failed: {e}\n{_tb.format_exc()}")
        reply_text = _chat_fallback(user_msg)
        if any(k in str(e).lower() for k in ("credentials", "token", "expired", "invalid")):
            reply_text = f"⚠ LLM credentials expired. Using local mode.\n\n{reply_text}"
        elif str(e):
            reply_text = f"⚠ LLM error: {str(e)[:120]}\n\n{reply_text}"

    # ── Execute any action embedded in the response ───────────────
    action_match = _re.search(r'\{[^{}]*"action"[^{}]*\}', reply_text)
    action_result = ""
    if action_match:
        try:
            act = json.loads(action_match.group(0))
            action_result = _execute_chat_action(act)
            # Remove JSON block from displayed reply
            reply_text = reply_text[:action_match.start()].rstrip()
        except Exception as _act_err:
            import logging as _logging
            _logging.getLogger("web_control_panel").warning(
                f"[WebPanel] Chat action execution failed: {_act_err}"
            )

    if action_result:
        reply_text = reply_text + "\n\n" + action_result

    return jsonify({"reply": reply_text})


def _chat_fallback(msg: str) -> str:
    """Deterministic fallback when LLM is unavailable."""
    import re as _re
    msg_l = msg.lower()

    if _re.search(r"status|what.*happen|how.*going", msg_l):
        s = state.get("status", "idle")
        last = state["planning_log"][-1]["msg"][:80] if state.get("planning_log") else "No activity"
        return f"Cell is **{s.upper()}**.\nLast event: {last}"

    robot_states = state.get("robot_states", {})
    m = _re.search(r"robot\s*(\d+)", msg_l)
    if m and robot_states:
        ns = f"robot{m.group(1)}"
        rs = robot_states.get(ns, {})
        joints = rs.get("joints", [])
        ee = rs.get("ee_position")
        if joints:
            jstr = ", ".join(f"j{i+1}={v:.3f} rad" for i, v in enumerate(joints))
            eestr = f"\nEE position: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m" if ee else ""
            return f"**{ns}** joint angles:\n{jstr}{eestr}"
        return f"No joint state data received yet for {ns}."

    if _re.search(r"trajectory|path|plan", msg_l):
        return state.get("current_plan_info", {}).get("summary", "No plan computed yet.")

    if _re.search(r"replay|recording|playback|list.*record", msg_l):
        recs = _list_recordings()
        if not recs:
            return "No recordings found. Run an agentic cycle first to generate one."
        lines = [f"**{len(recs)} recording(s)** available. Use the Replay panel below or say 'replay last'."]
        for r in recs[:5]:
            ops = r.get("operations")
            lines.append(f"  • {r['tree_id']}"
                         + (f"  ({ops} operations)" if ops else ""))
        lines.append("\nCommands: **replay last** · **replay slow** · **replay 0.5**")
        return "\n".join(lines)

    return ("LLM unavailable. Available commands:\n"
            "**status** · **robot1 position** · **last trajectory** · "
            "**list recordings** · **replay last** · **run** · **stop** · **reset**")


def _list_recordings() -> list:
    """Metadata for the cell's execution recordings, newest first.

    A recording is one ``<tree_id>.rrd``, written by the cell controller when it
    executes a behaviour tree — so the cell controller is asked for the list
    (MCP ``list_recordings``), and the trajectory agent for the operation count
    of the matching plan (``list_calculated_plans``), joined on tree_id.

    A recording with no plan beside it keeps ``operations: None`` and is still
    listed: it can outlive its plan, or be a bare ``home_reset_*``, and hiding it
    would hide something replayable.

    This used to glob the volume here — first ``plan_*.json``, a name the recorder
    stopped writing when recordings became rerun logs keyed by tree_id, and then
    ``*.rrd`` across the cell dir plus the legacy ``/recordings`` and
    ``/sim_recordings`` flat mounts. The legacy mounts are no longer consulted,
    because the agent that writes recordings does not consult them either: one
    owner, one answer, no disagreement to debug.
    """
    cell_id = default_cell_id()
    listing = _mcp_call(_cell_mcp(), "list_recordings", {"cell_id": cell_id})
    if _mcp_failed(listing):
        return []

    plans = _mcp_call(_traj_mcp(), "list_calculated_plans", {"cell_id": cell_id})
    ops_by_tree = {}
    if not _mcp_failed(plans):
        ops_by_tree = {p.get("tree_id"): p.get("num_operations")
                       for p in (plans.get("plans") or []) if isinstance(p, dict)}

    recs = []
    for r in (listing.get("recordings") or [])[:20]:
        if not isinstance(r, dict):
            continue
        tree_id = str(r.get("tree_id") or "")
        path = r.get("path") or ""
        recs.append({
            "name":       os.path.basename(path) or f"{tree_id}.rrd",
            "tree_id":    tree_id,
            "path":       path,
            "size_bytes": r.get("size_bytes"),
            "timestamp":  r.get("timestamp", ""),
            "operations": ops_by_tree.get(tree_id),
        })
    return recs


@app.route("/api/recordings")
def get_recordings():
    """Return list of available recordings for the chat panel."""
    return jsonify(_list_recordings())


@app.route("/api/speed", methods=["POST"])
def set_speed():
    """Record the panel's replay speed setting (0.1 – 5.0) — display state only."""
    data  = flask_request.json or {}
    scale = float(data.get("scale", 1.0))
    scale = max(0.1, min(5.0, scale))
    # Display state only. It used to be published on /cell_config/velocity_scale,
    # which nothing subscribes to; motion speed is set in the cell's robots.yaml.
    state["velocity_scale"] = scale
    return jsonify({"ok": True, "scale": scale})


_loop_thread: "Optional[threading.Thread]" = None
_loop_active  = threading.Event()


@app.route("/api/loop", methods=["POST"])
def set_loop():
    """Toggle trajectory replay loop."""
    global _loop_thread
    data    = flask_request.json or {}
    enabled = bool(data.get("enabled", False))
    interval = float(data.get("interval_s", 30.0))

    if enabled:
        # The loop published "replay 1.0" on /orchestrator/command every interval —
        # a topic nothing reads, so it looped without ever replaying anything.
        return jsonify({"ok": False, "loop": False,
                        "error": ("loop replay went out on /orchestrator/command, which "
                                  "nothing reads — replay a plan from the BTree tab")}), 410
    else:
        _loop_active.clear()
        state["loop_active"] = False
        _log("Loop replay DISABLED", "info", "rai-orchestrator")
        return jsonify({"ok": True, "loop": False})


# Everything below used to sit HERE, after set_loop's return, with no `def` line
# of its own: an older duplicate of do_replay's body followed by this function's
# body. So `chat()`'s call to _execute_chat_action raised NameError on every
# action the LLM emitted — run, stop, reset, replay, move, create_weld — and the
# except around it logged "Chat action execution failed" at DEBUG and answered as
# if only the confirmation text had been lost. The duplicate replay body is gone
# (do_replay owns that route) and the signature is back.
def _execute_chat_action(act: dict) -> str:
    """Execute an action dict extracted from the LLM response.

    Returns the operator-facing confirmation to append to the reply, or "" when
    the action is not one this panel performs.
    """
    import re as _re
    action = act.get("action", "")

    if action in ("run", "agentic"):
        # Prefer A2A HTTP endpoint (non-blocking, returns run_id) over DDS topic
        a2a_base = os.environ.get("A2A_BASE_URL",
                                   f"http://rai-orchestrator:{os.environ.get('A2A_PORT','8889')}")
        try:
            import urllib.request as _ur
            req  = _ur.Request(f"{a2a_base}/a2a/run",
                               data=json.dumps({"task": "run"}).encode(),
                               headers={"Content-Type": "application/json"},
                               method="POST")
            with _ur.urlopen(req, timeout=3) as resp:
                result = json.loads(resp.read())
            _log(f"Chat → A2A run ({result.get('run_id','?')})", "info", "rai-orchestrator")
            return f"✓ Started agentic welding cycle (run_id={result.get('run_id','?')})."
        except Exception as _a2a_err:
            # The old DDS fallback published on a topic nothing reads — no cycle
            # started. Report the failure instead of claiming success.
            _log(f"A2A run failed: {_a2a_err}", "warn", "rai-orchestrator")
            return f"✗ Could not start the welding cycle: orchestrator unreachable ({_a2a_err})."

    if action == "stop":
        a2a_base = os.environ.get("A2A_BASE_URL",
                                   f"http://rai-orchestrator:{os.environ.get('A2A_PORT','8889')}")
        try:
            import urllib.request as _ur
            req = _ur.Request(f"{a2a_base}/a2a/stop",
                              data=b"{}",
                              headers={"Content-Type": "application/json"},
                              method="POST")
            _ur.urlopen(req, timeout=2)
        except Exception as _stop_err:
            _log(f"A2A stop failed: {_stop_err}", "warn", "rai-orchestrator")
            return (f"✗ Stop did not reach the orchestrator ({_stop_err}). Use the "
                    "E-STOP button to halt the arms.")
        return "✓ Stop command sent."

    if action == "reset":
        # Was a publish on /{cell}/command, which NOTHING subscribes to — so the
        # chat promised "robots returning to home" and no arm moved. It goes
        # through the same endpoint the Reset button uses, and reports what that
        # endpoint actually answered.
        res, st = _a2a_post("/a2a/reset", {"cell_id": CELL_ID, "async": True,
                                           "reason": "operator reset from chat"})
        if st >= 400 or res.get("error"):
            _log(f"Chat reset failed: {str(res.get('error'))[:160]}",
                 "error", "rai-orchestrator")
            return f"✗ Reset failed: {str(res.get('error'))[:160]}"
        _log("Chat → RESET: activity stopped; arms returning HOME (this CLEARS a "
             "latched e-stop).", "warn", "rai-orchestrator")
        return ("✓ Reset dispatched — activity stopped and the arms are returning "
                "to HOME. This cleared any latched e-stop; watch the log.")

    if action == "replay":
        # Was a publish on /orchestrator/command, which nothing reads: it promised
        # a replay and nothing ran. Replay is started from the BTree tab.
        return ("✗ Chat replay is not wired to execution. Open the BTree tab, pick the "
                "plan and press Replay (POST /api/a2a/replay).")

    if action == "list_recordings":
        recs = _list_recordings()
        if not recs:
            return f"No recordings found for cell '{CELL_ID}'."
        lines = [f"**{len(recs)} recording(s):**"]
        for r in recs[:8]:
            lines.append(f"  • {r['tree_id']}  ({r.get('timestamp','')}, "
                         f"{(r.get('size_bytes') or 0) / 1024:.1f} KB)")
        return "\n".join(lines)

    if action == "move":
        ns = act.get("robot", "robot1")
        x, y, z = float(act.get("x", 0)), float(act.get("y", 0)), float(act.get("z", 0.8))
        if ros_node:
            try:
                from agent_interfaces.msg import TaskAssignment
                from geometry_msgs.msg import Pose, Point, Quaternion
                from trajectory_msgs.msg import JointTrajectory as _JT
                t = TaskAssignment()
                t.task_id = f"chat_{int(time.time())}"
                t.batch_id = t.task_id
                t.robot_namespace = ns
                t.operation_type = "move"
                t.end_effector_link = "welding_tcp"
                t.planning_group = "kuka_arm"
                t.trajectory_planner = "pinocchio"
                t.max_duration = 15.0
                t.velocity_scaling = 0.5
                t.goal_tolerance = 0.05
                t.priority = 1.0
                t.target_pose = Pose(
                    position=Point(x=x, y=y, z=z),
                    orientation=Quaternion(x=0.0, y=0.707, z=0.0, w=0.707),
                )
                t.precomputed_trajectory = _JT()
                pub = ros_node.create_publisher(
                    TaskAssignment, f"/{CELL_ID}/{ns}/{ns}/dispatch_subtask", 10
                )
                pub.publish(t)
                _log(f"Chat → move {ns} to ({x:.2f},{y:.2f},{z:.2f})", "info", f"robot-agent-{ns}")
                return f"✓ Move dispatched to {ns}: ({x:.2f}, {y:.2f}, {z:.2f}) m"
            except Exception as e:
                return f"Move failed: {e}"
        return "ROS not ready."

    if action == "create_weld":
        # Appends one weld operation to a task program, through the trajectory
        # agent that owns the tasks/ directory (get_task_program →
        # save_task_program). It used to open cells/<cell>/tasks/welding_task.json
        # here and append to `tasks[].parameters.points` — a program name no cell
        # in this repo defines, in a schema no reader has accepted since the
        # format became `operations[]` — and report success either way, because
        # the failure landed in a bare except. So: name the program, write the
        # schema the planner reads, and say which file changed.
        x, y, z = float(act.get("x", 0)), float(act.get("y", 0)), float(act.get("z", 0))
        program = (act.get("program") or "").strip()
        if not program:
            catalogue = _mcp_call(_traj_mcp(), "list_task_programs",
                                  {"cell_id": CELL_ID})
            if _mcp_failed(catalogue):
                return f"Could not list task programs: {_mcp_failed(catalogue)}"
            names = [str(p) for p in (catalogue.get("programs") or [])]
            if len(names) == 1:
                program = names[0]
            elif not names:
                return (f"Cell '{CELL_ID}' has no task program yet — author one "
                        f"with the Wizard in the Operator tab first.")
            else:
                # Guessing which program to modify is how a weld lands in the
                # wrong one, and nothing about the result would look wrong.
                return ("Which program should the weld go in? "
                        + ", ".join(names))

        current = _mcp_call(_traj_mcp(), "get_task_program",
                            {"cell_id": CELL_ID, "program": program})
        if _mcp_failed(current):
            return f"Could not read program '{program}': {_mcp_failed(current)}"

        task_data = current.get("task") or {}
        operations = list(task_data.get("operations") or [])
        point_id = f"CHAT_{int(time.time())}"
        operations.append({
            "id": point_id, "type": "weld",
            "position": [x, y, z],
            # Straight down is the only default that is a decision rather than a
            # guess: it is the approach the cell's own programs use.
            "approach_vector": [0.0, 0.0, -1.0],
            "sequence_group": "chat",
        })
        saved = _mcp_call(_traj_mcp(), "save_task_program",
                          {"cell_id": CELL_ID, "program": program,
                           "task": {**task_data, "operations": operations},
                           "overwrite": True})
        if _mcp_failed(saved):
            return f"Could not add weld point: {_mcp_failed(saved)}"
        return (f"✓ Weld {point_id} added at ({x}, {y}, {z}) m to '{program}' "
                f"({saved.get('num_operations')} operations now). "
                f"Nothing has been planned or moved — plan it when ready.")

    return ""


# ─── New operator API routes ──────────────────────────────────────────────────
# These routes expose explicit REST endpoints for all operator actions.
# The web UI calls them directly (no LLM inference required).

A2A_BASE = os.environ.get(
    "A2A_BASE_URL",
    f"http://{os.environ.get('ORCHESTRATOR_HOST', 'orchestrator')}:"
    f"{os.environ.get('ORCHESTRATOR_PORT', '8889')}",
)


#
# These are PROXIES: the dashboard forwards to the orchestrator and relays the
# answer. So they relay the STATUS too. They used to fold every upstream failure
# into `{"error": ...}` returned with HTTP 200, which left no way for any client —
# the panel's own JavaScript included — to tell "the orchestrator refused this"
# from "here is your result". A proxy that reports 200 for a 404 is a proxy that
# lies about the thing it exists to forward.
#
# The timeout is short on purpose: every call below either returns immediately or
# was dispatched with `"async": true` and is polled through /api/a2a/result. A
# request that legitimately takes minutes must not be held open here.
A2A_TIMEOUT_S = 10

# How the `/api/command` watcher thread follows an async run to completion.
# The interval is a compromise: a run takes tens of seconds to minutes, so
# polling faster only adds requests, and polling slower makes the operator log
# lag behind what the robots are doing.
RUN_POLL_INTERVAL_S = 2.0

# When to STOP WATCHING — not a run timeout. Nothing here can cancel a run (only
# POST /api/a2a/stop can), so the watcher gives up on reporting rather than
# claiming an outcome it does not have. Generous, because a long welding program
# legitimately runs for a long time.
RUN_POLL_TIMEOUT_S = 1800.0


def _a2a_post(path: str, payload: dict) -> "tuple[dict, int]":
    """POST to the orchestrator. Returns (parsed JSON, HTTP status to relay)."""
    import urllib.request as _ur
    import urllib.error  as _ue
    try:
        req = _ur.Request(
            f"{A2A_BASE}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=A2A_TIMEOUT_S) as resp:
            return json.loads(resp.read()), resp.status
    except _ue.HTTPError as e:
        return _upstream_error(e), e.code
    except Exception as e:
        # The orchestrator could not be reached at all: this dashboard is a
        # gateway that has no upstream, which is what 502 means.
        return {"error": f"orchestrator unreachable: {e}"}, 502


def _a2a_get(path: str) -> "tuple[dict, int]":
    """GET from the orchestrator. Returns (parsed JSON, HTTP status to relay)."""
    import urllib.request as _ur
    import urllib.error  as _ue
    try:
        with _ur.urlopen(f"{A2A_BASE}{path}", timeout=A2A_TIMEOUT_S) as resp:
            return json.loads(resp.read()), resp.status
    except _ue.HTTPError as e:
        return _upstream_error(e), e.code
    except Exception as e:
        return {"error": f"orchestrator unreachable: {e}"}, 502


def _upstream_error(exc) -> dict:
    """Turn an upstream HTTP error into the body to relay.

    The orchestrator answers errors as JSON, so pass its own message through when
    it is parseable — the caller wants "cell is busy running run X", not the
    status line. Fall back to the raw text when it is not JSON (a proxy or a
    traceback page), truncated because that path has no length contract.
    """
    try:
        raw = exc.read().decode()
    except Exception:  # noqa: BLE001
        return {"error": f"HTTP {exc.code} from orchestrator"}
    try:
        body = json.loads(raw)
    except ValueError:
        return {"error": f"HTTP {exc.code}: {raw[:200]}"}
    if isinstance(body, dict):
        body.setdefault("error", f"HTTP {exc.code} from orchestrator")
        return body
    return {"error": f"HTTP {exc.code}: {raw[:200]}"}


@app.route("/api/a2a/run", methods=["POST"])
def a2a_run():
    """
    Plan and execute a task program via A2A. Dispatches; does not wait.

    Body: {"program": str?, "cell_id": str?, "model": str?}
      program defaults to the cell's first task program if omitted upstream.

    Returns the run envelope — poll /api/a2a/result/<run_id>.
    """
    data = flask_request.json or {}
    body = {"cell_id": data.get("cell_id", CELL_ID), "async": True}
    for key in ("program", "model"):
        if data.get(key):
            body[key] = data[key]
    result, status = _a2a_post("/a2a/run", body)
    _log(f"A2A run → run_id={result.get('run_id','?')}", "info", "rai-orchestrator")
    return jsonify(result), status


@app.route("/api/a2a/stop", methods=["POST"])
def a2a_stop():
    """Stop the cell via A2A — trips the cell-wide e-stop the robot agents honour."""
    data = flask_request.json or {}
    result, status = _a2a_post("/a2a/stop", {
        "cell_id": data.get("cell_id", CELL_ID),
        "reason": data.get("reason") or "operator stop from dashboard",
    })
    _log("A2A stop → e-stop tripped", "warn", "rai-orchestrator")
    return jsonify(result), status


@app.route("/api/a2a/status")
def a2a_status():
    """Live per-robot state from the cell, via the orchestrator."""
    result, status = _a2a_get(f"/a2a/status?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/a2a/runs")
def a2a_runs():
    """Runs this orchestrator knows about for the cell, newest first."""
    result, status = _a2a_get(f"/a2a/runs?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/a2a/result/<run_id>")
def a2a_result(run_id):
    """Poll a run by id: {run_id, status, done, result, error}."""
    result, status = _a2a_get(f"/a2a/result/{run_id}?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/a2a/reasoning")
def a2a_reasoning():
    """The LLM reasoning + action trace of the newest run (or ?run_id=)."""
    run_id = flask_request.args.get("run_id", "")
    query = f"?cell_id={CELL_ID}" + (f"&run_id={run_id}" if run_id else "")
    result, status = _a2a_get(f"/a2a/reasoning{query}")
    return jsonify(result), status


@app.route("/api/a2a/plan_traces")
def a2a_plan_traces():
    """History: the metadata of every recorded planning run, newest first.

    Distinct from /api/a2a/reasoning, which is the orchestrator's own trace of a
    run and exists only after the run finishes. This is the trajectory planner's
    per-decision trace, written from inside the ReAct loop.
    """
    limit = flask_request.args.get("limit", "50")
    result, status = _a2a_get(f"/a2a/plan_traces?cell_id={CELL_ID}&limit={limit}")
    return jsonify(result), status


@app.route("/api/a2a/plan_trace/<trace_id>")
def a2a_plan_trace(trace_id):
    """One planning run's decisions, indexed per weld point. ``latest`` allowed."""
    result, status = _a2a_get(f"/a2a/plan_trace/{trace_id}?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/a2a/plan_events")
def a2a_plan_events():
    """The live feed of a planning run, for the Agents tab's activity view.

    Polled with ?since=<next_seq>, so each call carries only what has happened
    since the last one — the panel refreshes about once a second and a fifteen
    minute run must not be re-sent every time.
    """
    since = flask_request.args.get("since", "0")
    limit = flask_request.args.get("limit", "200")
    trace = flask_request.args.get("trace_id", "")
    query = f"?cell_id={CELL_ID}&since={since}&limit={limit}"
    if trace:
        query += f"&trace_id={trace}"
    result, status = _a2a_get(f"/a2a/plan_events{query}")
    return jsonify(result), status


@app.route("/api/plan_progress")
def plan_progress_feed():
    """The live planning feed as it arrives on DDS, polled with ?since=<next_seq>.

    Same response shape as ``/api/a2a/plan_events`` so one renderer serves both,
    plus ``available``: false means this panel has no DDS feed (no ROS node, or
    nothing published yet on this cell's topic) and the browser must fall back to
    the orchestrator's durable stream. Only a positive ``available`` lets this
    route serve the view — an empty answer from a panel that is not listening must
    never read as "no run in progress".
    """
    if ros_node is None:
        return jsonify({"success": True, "source": "dds", "available": False,
                        "cell_id": CELL_ID, "events": [], "trace_id": None,
                        "next_seq": 0, "running": False,
                        "error": "this panel has no ROS node"}), 200
    try:
        since = int(flask_request.args.get("since", "0"))
        limit = int(flask_request.args.get("limit", "200"))
    except ValueError:
        return jsonify({"success": False, "available": False,
                        "error": "since and limit must be integers"}), 400
    answer = _plan_progress_since(max(0, since), limit)
    answer["available"] = answer["trace_id"] is not None
    return jsonify(answer), 200


@app.route("/api/a2a/checkpoints")
def a2a_checkpoints():
    """Planning runs that can be resumed instead of restarted."""
    limit = flask_request.args.get("limit", "50")
    result, status = _a2a_get(f"/a2a/checkpoints?cell_id={CELL_ID}&limit={limit}")
    return jsonify(result), status


@app.route("/api/a2a/plan/pause", methods=["POST"])
def a2a_plan_pause():
    """Pause planning, keeping its work. Not Stop — see /api/a2a/stop for that."""
    body = flask_request.get_json(silent=True) or {}
    payload = {"cell_id": CELL_ID,
               "reason": body.get("reason") or "paused from the dashboard"}
    if body.get("tree_id"):
        payload["tree_id"] = body["tree_id"]
    result, status = _a2a_post("/a2a/plan/pause", payload)
    return jsonify(result), status


@app.route("/api/a2a/plan/resume", methods=["POST"])
def a2a_plan_resume():
    """Continue a paused or interrupted planning run from its checkpoint."""
    body = flask_request.get_json(silent=True) or {}
    tree_id = (body.get("tree_id") or "").strip()
    if not tree_id:
        return jsonify({"error": "tree_id is required"}), 400
    payload = {"cell_id": CELL_ID, "tree_id": tree_id}
    if body.get("model"):
        payload["model"] = body["model"]
    result, status = _a2a_post("/a2a/plan/resume", payload)
    return jsonify(result), status


@app.route("/api/a2a/move", methods=["POST"])
def a2a_move():
    """
    Jog a single robot via the orchestrator. Dispatches; does not wait.

    Body:
      {
        "robot_ns":   "robot1",   required
        "x","y","z":  floats,     required (world frame, metres)
        "duration_s": 3.0         optional — how long the move takes
      }

    This is an unplanned move: IK to the point, then a straight joint
    interpolation, with no collision checking (`collision_checked: false` comes
    back in the run result). Use the task pipeline for anything that must be
    collision-free.
    """
    data = flask_request.json or {}
    robot = data.get("robot_ns") or data.get("robot")
    if not robot:
        return jsonify({"error": "robot_ns is required"}), 400
    missing = [k for k in ("x", "y", "z") if k not in data]
    if missing:
        return jsonify({"error": f"missing coordinate(s): {', '.join(missing)}"}), 400
    try:
        target = {k: float(data[k]) for k in ("x", "y", "z")}
        duration_s = float(data.get("duration_s", 3.0))
    except (TypeError, ValueError):
        return jsonify({"error": "x, y, z and duration_s must be numbers"}), 400

    result, status = _a2a_post("/a2a/move", {
        "cell_id": CELL_ID, "robot_ns": robot, **target,
        "duration_s": duration_s, "async": True,
    })
    _log(f"Jog {robot} → ({target['x']:.2f},{target['y']:.2f},{target['z']:.2f})",
         "info", f"robot-agent-{robot}")
    return jsonify(result), status


@app.route("/api/a2a/task/save", methods=["POST"])
def a2a_task_save():
    """
    SAVE a task program to the cell's tasks dir and run NOTHING.

    Body: a task program (`program` + non-empty `operations`), plus optional
    `overwrite`. This is what the task wizard (GET /wizard) calls when the
    operator presses "Write task file" — the file lands on disk, no run is
    registered, and no robot is commanded. Planning it is a separate click.
    """
    data = flask_request.json or {}
    if not (data.get("program") or "").strip():
        return jsonify({"error": "Missing 'program' field (the program name)"}), 400
    if not isinstance(data.get("operations"), list) or not data["operations"]:
        return jsonify({"error": "'operations' must be a non-empty array"}), 400
    body = {**data, "cell_id": data.get("cell_id", CELL_ID)}
    result, status = _a2a_post("/a2a/task/save", body)
    if status < 400:
        _log(f"Task file written: {data['program']} (not run)", "info", "web-panel")
    return jsonify(result), status


@app.route("/api/a2a/task/submit", methods=["POST"])
def a2a_task_submit():
    """
    Submit a task program and run it. Dispatches; does not wait.

    The body is a task program per
    `skills/orchestrator/run-welding-program/references/task-json-format.md`:
    `program` plus a non-empty `operations` array. The orchestrator saves it to
    cells/<cell>/tasks/<program>.json and runs it; an existing program of that
    name is refused (409) unless the body sets `"overwrite": true`.
    """
    data = flask_request.json or {}
    if not (data.get("program") or "").strip():
        return jsonify({"error": "Missing 'program' field (the program name)"}), 400
    if not isinstance(data.get("operations"), list) or not data["operations"]:
        return jsonify({"error": "'operations' must be a non-empty array"}), 400
    body = {**data, "cell_id": data.get("cell_id", CELL_ID), "async": True}
    result, status = _a2a_post("/a2a/task/submit", body)
    _log(f"Task submitted: {data['program']}", "info", "rai-orchestrator")
    return jsonify(result), status


@app.route("/api/a2a/replay", methods=["POST"])
def a2a_replay():
    """
    Replay a calculated plan via the orchestrator. Dispatches; does not wait.

    Body: {"tree_id": "bt_...", "home_reset": true}
      tree_id is optional — omit to replay the most recent plan.

    Playback speed is NOT set here: the cell's velocity scale is its own setting,
    owned by POST /api/speed. Passing a speed to replay would be a second, weaker
    control over the same number.
    """
    data = flask_request.json or {}
    body = {"cell_id": CELL_ID, "async": True,
            "home_reset": bool(data.get("home_reset", True))}
    if data.get("tree_id"):
        body["tree_id"] = data["tree_id"]
    result, status = _a2a_post("/a2a/replay", body)
    _log(f"Replay {body.get('tree_id', 'latest')}", "info", "rai-orchestrator")
    return jsonify(result), status


@app.route("/api/a2a/recordings")
def a2a_recordings():
    """List the cell's execution recordings, via the orchestrator."""
    result, status = _a2a_get(f"/a2a/recordings?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/a2a/recordings/<tree_id>")
def a2a_recording_detail(tree_id):
    """Metadata for one recording, by tree_id."""
    result, status = _a2a_get(f"/a2a/recordings/{tree_id}?cell_id={CELL_ID}")
    return jsonify(result), status


def _valid_id(value: str) -> bool:
    """Is this safe to put in a path that ends at a filesystem lookup?

    Plan ids reach ``plan_store.get_plan``, which builds ``<dir>/<tree_id>.json``,
    two hops away — so they are validated at this edge rather than trusted for
    being "internal": one id shape, no separators, nothing that climbs out of the
    plans directory or names another route.
    """
    return bool(value) and bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value)) \
        and not value.startswith(".")


@app.route("/api/plans")
def api_plans():
    """The behaviour trees calculated for this cell (newest first).

    What the Tree tab's selector is filled from. Same store the Recordings tab
    replays out of: a recording IS the execution of one of these plans, keyed by
    the same tree_id.
    """
    result, status = _a2a_get(f"/a2a/plans?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/plans/<tree_id>")
def api_plan(tree_id):
    """One behaviour tree, shape and ports only — for the graphical tree view.

    The orchestrator strips the waypoint blackboard (see its /a2a/plans/<id>):
    the browser draws the tree, it never needs the joint samples.
    """
    # The id reaches a filesystem lookup two hops away, so it is validated here
    # rather than trusted for being "internal": one tree_id shape, no separators,
    # nothing that could climb out of the plans directory or name another route.
    if not _valid_id(tree_id):
        return jsonify({"success": False, "error": f"invalid tree_id '{tree_id[:40]}'"}), 400
    result, status = _a2a_get(f"/a2a/plans/{tree_id}?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/plans/<tree_id>/nodes/<node_id>")
def api_plan_node(tree_id, node_id):
    """One node of one tree, WITH its waypoints — what descending into a leaf needs.

    A leaf's ports say which robot and which operation; they say nothing about the
    motion. This is the only endpoint that ships joint samples to the browser, and
    it ships exactly one node's worth (a couple of hundred, whatever the plan's
    size), which is why the descent is per-node instead of the whole blackboard.
    """
    if not _valid_id(tree_id):
        return jsonify({"success": False, "error": f"invalid tree_id '{tree_id[:40]}'"}), 400
    if not _valid_id(node_id):
        return jsonify({"success": False, "error": f"invalid node_id '{node_id[:40]}'"}), 400
    result, status = _a2a_get(
        f"/a2a/plans/{tree_id}/nodes/{node_id}?cell_id={CELL_ID}")
    return jsonify(result), status


@app.route("/api/plans/<tree_id>/quality")
def api_plan_quality(tree_id):
    """The trajectory-quality KPIs for one plan — score, findings, per-leg numbers.

    Asked of the planner over MCP (``get_plan_quality``), like every other
    question this panel asks about a cell: the metric definitions belong to
    ``tools/trajectory_quality.py`` and the report belongs to the agent that owns
    the cell's logs. This panel renders it and computes nothing.

    A failure comes back as ``success: false`` with the reason, which the KPI card
    shows in place of the numbers — an unreachable planner must read as "not
    measured", never as a plan that scores zero.
    """
    if not _valid_id(tree_id):
        return jsonify({"success": False,
                        "error": f"invalid tree_id '{tree_id[:40]}'"}), 400
    # 200 with the refusal in the body, like the other MCP-backed reads here: the
    # card has to distinguish "not measured, because X" from "measured badly", and
    # a status code cannot carry the reason to the place that renders it.
    return jsonify(_mcp_call(_traj_mcp(), "get_plan_quality",
                             {"cell_id": CELL_ID, "tree_id": tree_id})), 200


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """Show one node's motion on the GHOST arm in RViz. Commands nothing.

    Step-through debugging without moving the cell. The waypoints are published as
    JSON on ``/<cell>/node_preview``, which only the viewer listens to: it drives a
    translucent second copy of the arm (see ``docker/launch_viz.py`` → GHOST_SUFFIX).
    Nothing is dispatched to a robot agent, no FollowJointTrajectory goal is sent,
    and the cell's state is untouched — so this is safe to press mid-plan, and it
    can NOT be used to drive the cell.

    Body: ``{tree_id, node_id, phase_index?, speed?, loop?}``. ``phase_index``
    selects one phase segment of the node (see the orchestrator's
    ``_phase_segments``) instead of the whole motion.
    """
    data = flask_request.json or {}
    tree_id = str(data.get("tree_id") or "latest")
    node_id = str(data.get("node_id") or "")
    if not _valid_id(tree_id) or not _valid_id(node_id):
        return jsonify({"success": False,
                        "error": "tree_id and node_id must both be given and valid"}), 400
    if ros_node is None:
        return jsonify({"success": False,
                        "error": "no ROS node in this panel — previews need the "
                                 "viewer's ghost arm, which is driven over ROS"}), 503

    node, status = _a2a_get(f"/a2a/plans/{tree_id}/nodes/{node_id}?cell_id={CELL_ID}")
    if status != 200 or not node.get("success"):
        return jsonify({"success": False, "tree_id": tree_id, "node_id": node_id,
                        "error": node.get("error", f"could not load node ({status})")}), status

    waypoints = node.get("waypoints") or []
    label = f"{node.get('type')} {node.get('name') or node_id}"

    # One phase of the motion rather than all of it: the approach run-in is the
    # part worth watching on its own, and it is a handful of the samples.
    phase_index = data.get("phase_index")
    if phase_index is not None:
        segments = node.get("phase_segments") or []
        try:
            seg = segments[int(phase_index)]
        except (ValueError, TypeError, IndexError):
            return jsonify({"success": False, "error":
                            f"phase_index {phase_index} is not one of this node's "
                            f"{len(segments)} phase segment(s)"}), 400
        waypoints = waypoints[seg["start_index"]:seg["end_index"] + 1]
        label += f" · {seg['phase']}"

    # A single sample: the ghost holds that exact pose so it can be inspected
    # against the fixture. Stepping one waypoint at a time is the finest grain
    # this offers, and it is still not a command.
    wp_index = data.get("waypoint_index")
    if wp_index is not None:
        try:
            waypoints = [waypoints[int(wp_index)]]
        except (ValueError, TypeError, IndexError):
            return jsonify({"success": False, "error":
                            f"waypoint_index {wp_index} is out of range for "
                            f"{len(waypoints)} waypoint(s)"}), 400
        label += f" · waypoint {wp_index}"

    if not waypoints:
        return jsonify({"success": False, "tree_id": tree_id, "node_id": node_id,
                        "error": f"'{label}' carries no trajectory to preview "
                                 f"(a control node runs its children, not a motion)"}), 400

    # Re-base to zero so a mid-trajectory phase starts immediately instead of the
    # ghost sitting still for the seconds that precede it in the full motion.
    t0 = float(waypoints[0].get("time_s") or 0.0)
    waypoints = [dict(w, time_s=float(w.get("time_s") or 0.0) - t0) for w in waypoints]

    robot_ns = (node.get("ports") or {}).get("robot_ns")
    payload = {"robot_ns": robot_ns, "waypoints": waypoints, "label": label,
               "speed": float(data.get("speed") or 1.0),
               "loop": bool(data.get("loop", True))}
    ros_node.preview_pub.publish(String(data=json.dumps(payload)))
    _log(f"Preview (ghost only, cell not moved): {label} — "
         f"{len(waypoints)} waypoint(s)", "info", "web-panel")
    return jsonify({"success": True, "tree_id": node.get("tree_id"),
                    "node_id": node_id, "robot_ns": robot_ns, "label": label,
                    "waypoints": len(waypoints),
                    "duration_s": round(waypoints[-1].get("time_s") or 0.0, 3),
                    "commanded": False})


@app.route("/api/preview/clear", methods=["POST"])
def api_preview_clear():
    """Park every ghost back on its real arm."""
    if ros_node is None:
        return jsonify({"success": False, "error": "no ROS node in this panel"}), 503
    ros_node.preview_pub.publish(String(data=json.dumps({"clear": True})))
    return jsonify({"success": True, "cleared": True})


@app.route("/api/a2a/tasks")
def a2a_tasks():
    """The task programs defined for the cell."""
    result, status = _a2a_get(f"/a2a/tasks?cell_id={CELL_ID}")
    return jsonify(result), status


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "info", agent_id: str = None):
    state["planning_log"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "msg": msg,
    })
    if agent_id:
        state["agent_activity"][agent_id] = time.time()
    if len(state["planning_log"]) > MAX_LOG_ENTRIES:
        state["planning_log"] = state["planning_log"][-MAX_LOG_ENTRIES:]


def _touch(agent_id: str):
    """Mark an agent as active (triggers blink on next poll)."""
    state["agent_activity"][agent_id] = time.time()


# ─── ROS 2 node ────────────────────────────────────────────────────────────────

ros_node = None


class ControlPanelROS(Node):
    def __init__(self):
        super().__init__("web_control_panel")

        # The e-stop publisher is live: the robot agents honour /{cell}/estop.
        self.estop_pub = self.create_publisher(String, cell_topic("/estop"), 1)

        # Step-through previews. Read ONLY by the viewer, which draws them on a
        # translucent ghost copy of the arm — no robot agent subscribes to this
        # topic, so a message here cannot move the cell. That is the whole point:
        # the operator can watch what a node does without executing it.
        self.preview_pub = self.create_publisher(
            String, cell_topic("/node_preview"), 10
        )

        self.create_subscription(
            String, cell_topic("/collision_alerts_json"), self._collision_json_cb, 10
        )
        # The planner's live progress. TRANSIENT_LOCAL and a deep history to match
        # the publisher, so a panel opened halfway through a fifteen-minute run
        # renders the run rather than waiting for its next event.
        self.create_subscription(
            String, cell_topic("/plan_progress"), self._plan_progress_cb,
            QoSProfile(depth=MAX_PLAN_PROGRESS_EVENTS,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        # Per-robot status — multi_robots_agents publishes std_msgs/String JSON
        # (NOT agent_interfaces/RobotStatus) at 10 Hz on
        #   /{cell}/{namespace}/{robot_id}/robot_status
        # carrying {joint_positions, ee_position, current_task, status, ...}.
        self._robots = _discover_robots()
        self._last_task: dict = {}
        for ns, rid in self._robots:
            self.create_subscription(
                String, f"/{CELL_ID}/{ns}/{rid}/robot_status",
                lambda msg, r=rid: self._robot_status_cb(msg, r),
                10,
            )

        # Cell-wide collision alerts (JSON) from the cell collision monitor.
        self.create_subscription(
            String, cell_topic("/collision_alerts_json"), self._collision_json_cb, 10
        )

        try:
            from rcl_interfaces.msg import Log
            self.create_subscription(Log, "/rosout", self._rosout_cb, 50)
        except Exception:
            pass

        self.get_logger().info(
            f"Web Control Panel ROS node ready — {len(self._robots)} robot(s), cell={CELL_ID}"
        )

    def _plan_progress_cb(self, msg):
        """One plan event off DDS. Swallows everything: a malformed message must
        not take the executor down and with it every other subscription."""
        handle_plan_progress_message(msg.data,
                                     warn=lambda text: self.get_logger().warn(text))

    def _collision_json_cb(self, msg):
        _touch("collision-monitor")
        try:
            data = json.loads(msg.data)
            alert = (
                f"⚠ COLLISION: {data.get('robot_a','?')} ↔ {data.get('robot_b','?')} "
                f"dist={float(data.get('distance', 0)):.3f}m — "
                f"{data.get('action','STOPPED')}"
            )
            state["collision_alerts"].append(alert)
            state["collision_alerts"] = state["collision_alerts"][-20:]
            _log(alert, "collision", "collision-monitor")
        except Exception as _coll_err:
            import logging as _logging
            _logging.getLogger("web_control_panel").debug(
                f"[WebPanel] Collision alert parse error: {_coll_err}"
            )

    def _robot_status_cb(self, msg, rid: str):
        """Parse the std_msgs/String JSON robot_status published by robot_agent."""
        agent_id = f"robot-agent-{rid}"
        _touch(agent_id)
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        task = data.get("current_task")
        state["robot_states"][rid] = {
            "joints":      data.get("joint_positions", []) or [],
            "velocities":  [],
            "ee_position": data.get("ee_position"),
            "stamp":       time.time(),
        }
        if task:
            state["status"] = "running"
        # Log only on task transitions to avoid 10 Hz spam.
        if self._last_task.get(rid) != task:
            self._last_task[rid] = task
            _log(f"[{rid}] {'busy — ' + str(task) if task else 'idle'}", "info", agent_id)

    def _rosout_cb(self, msg):
        relevant = ["interactive_orchestrator", "collaborative_planner",
                    "robot_agent", "pinocchio_robot_controller", "cell_state",
                    "dispatch_coordinator"]
        if not any(r in msg.name for r in relevant):
            return
        level_map = {10: "info", 20: "info", 30: "warn", 40: "error", 50: "error"}
        level = level_map.get(msg.level, "info")
        rids = [rid for _ns, rid in _discover_robots()]
        first_robot = f"robot-agent-{rids[0]}" if rids else None
        agent = next(
            (a for r, a in [
                ("orchestrator",            "rai-orchestrator"),
                ("planner",                 "collaborative-planner"),
                ("robot_agent",             first_robot),
                ("pinocchio_robot_control", first_robot),
                ("cell_state",              "cell-controller"),
                ("dispatch_coordinator",    "cell-controller"),
            ] if r in msg.name), None
        )
        # Robot-agent nodes share a logger name, so attribute by the robot id named
        # in the message. Works for any arm count, not just a robot2 special case.
        if agent and agent == first_robot:
            for rid in rids:
                if rid in msg.msg:
                    agent = f"robot-agent-{rid}"
                    break
        _log(f"[{msg.name}] {msg.msg}", level, agent)

    def _joint_states_cb(self, msg, ns: str):
        """
        Receive joint_states for a robot and compute EE position via Pinocchio FK.

        Stores in state["robot_states"][ns]:
          joints:      list of 6 joint angles (radians)
          velocities:  list of 6 joint velocities (rad/s)
          ee_position: [x, y, z] end-effector in world frame (metres)
          stamp:       wall-clock time of last update
        """
        joints = list(msg.position[:6]) if len(msg.position) >= 6 else list(msg.position)
        vels   = list(msg.velocity[:6])  if len(msg.velocity)  >= 6 else list(msg.velocity)

        ee_pos = self._compute_fk(ns, joints)

        state["robot_states"][ns] = {
            "joints":      joints,
            "velocities":  vels,
            "ee_position": ee_pos,
            "stamp":       time.time(),
        }
        _touch(f"robot-agent-{ns}")

    def _compute_fk(self, ns: str, joints: list):
        """
        Compute end-effector position using Pinocchio FK.
        Returns [x, y, z] in world frame, or None if Pinocchio unavailable.

        The model and its tool frame come from :mod:`utils.arm_model`, which is
        the one place that knows where an arm's URDF lives (referenced from
        cell.yaml or generated under ``urdf/``) and how the TCP is offset from
        the flange. The base pose is applied here because the URDF is the bare
        arm — and it comes from ``get_cell_info`` on the trajectory agent, not
        from opening ``config/robots.yaml`` here, so the panel and the planner
        cannot disagree about where the arm stands.

        The URDF/Pinocchio load stays local: it is a model the panel needs in
        process to evaluate FK per sample, not a fact to fetch.
        """
        try:
            import numpy as np
            import pinocchio as pin

            from utils import arm_model

            # Load and cache model per namespace
            if ns not in self._pin_models:
                try:
                    model, data, ee_id = arm_model.load(CELL_ID, ns)
                    self._pin_models[ns] = (model, data, ee_id)
                    self.get_logger().info(
                        f"[WebPanel] Pinocchio model loaded for {ns} (nq={model.nq})"
                    )
                except Exception as e:
                    self._pin_models[ns] = None
                    self.get_logger().warn(f"[WebPanel] Pinocchio load failed for {ns}: {e}")

            cached = self._pin_models.get(ns)
            if cached is None:
                return None

            model, data, ee_id = cached
            # Looked up per call, not captured with the model: the pose is cached
            # for _ROBOTS_TTL_S inside the helper, so this is a dict lookup, and
            # an arm whose pose was unavailable at model-load time starts being
            # placed correctly as soon as the agent answers — rather than staying
            # at the origin for the life of the process.
            rcfg = _robot_base_pose(ns)
            nq = model.nq
            q  = np.zeros(nq)
            for i, v in enumerate(joints[:min(len(joints), nq)]):
                q[i] = v

            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            p   = data.oMf[ee_id].translation.copy()
            base = np.array(rcfg.get("base_position", [0, 0, 0]))
            yaw  = float(rcfg.get("base_orientation_yaw", 0.0))
            cy, sy = np.cos(yaw), np.sin(yaw)
            wx = base[0] + p[0] * cy - p[1] * sy
            wy = base[1] + p[0] * sy + p[1] * cy
            wz = base[2] + p[2]
            return [round(float(wx), 4), round(float(wy), 4), round(float(wz), 4)]

        except Exception:
            return None


# ─── Entry point ───────────────────────────────────────────────────────────────

def ros_spin_thread():
    global ros_node
    rclpy.init()
    ros_node = ControlPanelROS()
    rclpy.spin(ros_node)


def main():
    t = threading.Thread(target=ros_spin_thread, daemon=True)
    t.start()
    time.sleep(1.0)

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print("=" * 50)
    print(f"  Web Control Panel  [{CELL_ID}]")
    print("  Open: http://localhost:8888")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8888, debug=False)


if __name__ == "__main__":
    main()
