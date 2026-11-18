"""
Cell Controller MCP Server

Exposes the cell agent and its tools as MCP tools over HTTP/SSE so that any
MCP-compatible client can call them:
  - Dashboard chat  (via langchain-mcp-adapters)
  - IDE agents      (via the IDE's MCP client config)
  - Orchestrator    (HTTP A2A call)
  - Claude Desktop  (stdio or SSE)

Transport: SSE (default, port CELL_MCP_PORT=8892)
         or stdio (MCP_TRANSPORT=stdio, for IDE use)

Each tool is a clean Python function with typed inputs and structured JSON
output. The MCP SDK generates tool schemas automatically from type hints and
docstrings.

Background threads:
  - http_server()          — Flask :8891 health endpoint (existing)
  - ros2_joint_state_sub() — subscribes /{cell_id}/{ns}/robot_status → recorder

Usage:
    # Container (SSE)
    python3 /agents/cell/mcp_server.py

    # IDE / offline (stdio)
    MCP_TRANSPORT=stdio python3 /agents/cell/mcp_server.py
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ─── Cell agent and tools ─────────────────────────────────────────────────────
from agents.cell_defaults import default_cell_id
from agents.llm_model import (
    MODEL_ALIASES as _MODEL_ALIASES,
    describe_models as _describe_models,
    normalize_model as _normalize_model,
)
from agents.cell.agent import execute_behaviour_tree as _agent_execute_behaviour_tree
# The abort is reached through the module, not bound at import: it reads the
# process-wide slot that holds the LIVE executor, so resolving it at call time
# keeps this pointed at whatever agent module is actually running the tree.
from agents.cell import agent as _cell_agent
from agents.cell.home_reset import home_reset_tree as _home_reset_tree
from agents.trajectory.tools.btcpp import children as bt_children, port as bt_port
from agents.cell.tools.cell_state import get_all_robot_states, trigger_estop as _trigger_estop
from agents.cell.server import http_server, ros2_joint_state_sub

# ─── Recorder ────────────────────────────────────────────────────────────────
from agents.cell.rerun_recorder import (
    RerunRecorder,
    NoOpRecorder,
    get_recorder,
    set_recorder,
)

from utils import cell_layout, cell_store

# ─── Environment config ───────────────────────────────────────────────────────
_CELL_ID    = default_cell_id()
_PORT       = int(os.environ.get("CELL_MCP_PORT", "8892"))
_TRANSPORT  = os.environ.get("MCP_TRANSPORT", "sse").lower()
_HTTP_PORT  = int(os.environ.get("CELL_HTTP_PORT", "8891"))
_BASE_PATH  = os.environ.get("CELL_BASE_PATH", "/cells")

# ─── MCP server ───────────────────────────────────────────────────────────────
mcp = FastMCP(
    "cell-controller",
    instructions=(
        "Cell controller agent for multi-robot industrial welding cells.\n"
        "\n"
        "MAIN TOOL: execute_behaviour_tree — submit a BehaviourTree JSON to run\n"
        "all welding/riveting operations in the correct sequence. Set\n"
        "replay_mode=True with a tree_id to replay a previously recorded session.\n"
        "\n"
        "REPLAY: replay_plan — re-run a previously CALCULATED plan by tree_id (or\n"
        "the latest) with an optional HOME reset first, no re-planning.\n"
        "\n"
        "MANUAL: move_robot — jog a single robot to a target joint configuration.\n"
        "\n"
        "STATE TOOLS: get_cell_status, get_robot_states — query live robot state.\n"
        "\n"
        "MANUAL: home_robots — send the arms back to HOME (planned + gated).\n"
        "\n"
        "STOPPING, two different things:\n"
        "  abort_execution — end the RUN. Commands nothing; a trajectory already\n"
        "                    accepted by a controller finishes. Not an e-stop.\n"
        "  trigger_estop   — HALT THE ARMS now: cancels the active goal and\n"
        "                    latches, so nothing dispatches until it is cleared.\n"
        "\n"
        "RECORDINGS: list_recordings — list all saved .rrd recording files.\n"
        "\n"
        "MODEL SELECTION: execute_behaviour_tree and replay_plan accept `model`\n"
        "(an alias such as 'opus-5', a full model id, or 'provider:model'); it\n"
        "reaches ONLY the LLM recovery layer that diagnoses an execution failure,\n"
        "never the motion itself. list_models shows the options and the default.\n"
    ),
)


# ─── MCP tools ────────────────────────────────────────────────────────────────

def _model_kwargs(model: Optional[str]) -> dict:
    """``{"model": ...}`` when a model was asked for, otherwise nothing.

    An absent override must leave the call it forwards to byte-identical, so the
    default path stays exactly what it was before per-operation selection existed.
    """
    return {"model": model} if model else {}


def _model_error(model: Optional[str]) -> Optional[dict]:
    """Reject an unusable `model` before anything moves.

    Returns an error payload to hand straight back to the caller, or None when
    the value is acceptable. Validating up front matters more here than in the
    planner: the alternative is discovering a typo'd model name only after the
    arms have executed and a recovery reasoner is needed.
    """
    try:
        _normalize_model(model)
    except ValueError as exc:
        return {"success": False, "error": str(exc),
                "hint": f"valid aliases: {', '.join(sorted(_MODEL_ALIASES))}"}
    return None


@mcp.tool()
def execute_behaviour_tree(
    behaviour_tree: dict,
    replay_mode: bool = False,
    model: Optional[str] = None,
) -> dict:
    """
    Execute or replay a BehaviourTree JSON on the cell.

    When replay_mode=False (default): runs the DETERMINISTIC executor — it walks
    the tree, dispatches precomputed trajectories to robot agents, enforces
    interlock/position/timing gates, and runs the collision monitor (e-stop on an
    imminent hit). No planner or LLM drives execution. If execution fails, the
    LLM recovery layer diagnoses the structured failure and recommends a next
    action. Recording is started before execution and closed when it finishes.

    When replay_mode=True: re-dispatches previously recorded trajectories from
    the matching .rrd file without invoking the trajectory planner. Requires
    behaviour_tree["tree_id"] to locate the recording.

    Args:
        behaviour_tree: BehaviourTree JSON from the trajectory agent.
                        Must contain "tree_id" when replay_mode=True.
        replay_mode:    False = execute; True = replay from recording.
        model:          LLM for the RECOVERY reasoner on this call only — an
                        alias ("opus-5"), a full model id, or "provider:model".
                        Omit for the startup default (see list_models). It has
                        no effect on the motion: the trajectories were planned
                        and validated before this call, and no model may
                        influence what the arms are commanded to do.

    Returns:
        success: {"success": True, "executed_nodes": int, "elapsed_s": float}
        failure: {"success": False, "elapsed_s": float,
                  "failure": {"reason_code", "reason", "collision"?, "robot_ns"?, ...},
                  "recovery": {"recommended_action", "rationale",
                               "propagate_to_orchestrator", ...}}
    """
    bad = _model_error(model)
    if bad:
        return bad

    recorder = get_recorder()

    if replay_mode:
        tree_id = behaviour_tree.get("tree_id")
        if not tree_id:
            return {"success": False, "error": "tree_id required for replay mode"}
        return recorder.replay_execution(tree_id, _CELL_ID)
    else:
        tree_id = behaviour_tree.get("tree_id") or behaviour_tree.get("program_name", "unknown")
        recorder.start_session(tree_id=tree_id, cell_id=_CELL_ID)
        try:
            result = _agent_execute_behaviour_tree(behaviour_tree, **_model_kwargs(model))
        finally:
            recorder.end_session()
        return result


def _remote_plan(cell_id: str, tree_id: Optional[str], _failed_record) -> Optional[dict]:
    """A plan record read through the cell-data MCP (``CELL_DATA_MODE=orchestrator``):
    the named one, or the most recent one that is not a failed run."""
    if tree_id:
        name = tree_id if str(tree_id).endswith(".json") else f"{tree_id}.json"
        return cell_store.read_json(cell_id, cell_store.KIND_PLANS, name)
    entries = cell_store.list_entries(cell_id, cell_store.KIND_PLANS, prefix="bt_",
                                      suffix=f"_{cell_id}.json")
    for e in sorted(entries, key=lambda e: float(e.get("mtime_s") or 0), reverse=True):
        record = cell_store.read_json(cell_id, cell_store.KIND_PLANS, e["name"])
        if record and not _failed_record(record):
            return record
    return None


@mcp.tool()
def replay_plan(cell_id: str, tree_id: Optional[str] = None,
                home_reset: bool = True, model: Optional[str] = None) -> dict:
    """
    Replay a previously CALCULATED plan on the cell WITHOUT re-planning.

    Loads the saved BehaviourTree (with its precomputed trajectories) from
    the cell's agent-owned plans/<tree_id>.json and runs it on the deterministic
    executor. If tree_id is omitted, the most recently calculated plan is used.
    When home_reset is True (default) all arms are first driven to HOME so the
    plan starts from the pose it was planned against. List available tree_ids
    with the trajectory agent's list_calculated_plans tool.

    Args:
        cell_id:    Cell identifier, e.g. "welding_cell_2_small".
        tree_id:    Plan id (e.g. "bt_1784722248_welding_cell_2_small"), or None = latest.
        home_reset: Drive all arms to HOME before replaying (default True).
        model:      LLM for the RECOVERY reasoner on this call only — an alias
                    ("opus-5"), a full model id, or "provider:model". Replay
                    itself is deterministic and consults no model; this is only
                    consulted if the replay fails. Omit for the default.

    Returns:
        {"success": bool, "executed_nodes": int, "elapsed_s": float, "tree_id": str}
        or {"success": False, "error": str}
    """
    import glob
    import json as _json
    bad = _model_error(model)
    if bad:
        return bad
    def _failed(candidate: str) -> Optional[dict]:
        """The failed-run record at this path, or None. Never raises."""
        try:
            record = _json.load(open(candidate))
        except Exception:  # noqa: BLE001 — an unreadable file is not a failed run
            return None
        return record if str(record.get("status") or "") == "failed" else None

    if cell_store.is_remote():
        # The cell controller runs on another machine than the planner: plans are
        # on the ORCHESTRATOR's disk and are read through its cell-data MCP — the
        # same "latest non-failed plan" rule, on the owner's listing.
        plan = _remote_plan(cell_id, tree_id, _failed_record=lambda r: str(
            (r or {}).get("status") or "") == "failed")
        if plan is None:
            return {"success": False,
                    "error": f"plan not found for tree_id={tree_id!r} (via cell-data MCP)"}
        path = f"{plan.get('tree_id') or tree_id or 'plan'}.json"
    else:
        pdir = str(cell_layout.plans_dir(cell_id))
        if tree_id:
            fn = tree_id if str(tree_id).endswith(".json") else f"{tree_id}.json"
            path = f"{pdir}/{fn}"
        else:
            files = sorted(glob.glob(f"{pdir}/bt_*_{cell_id}.json"), key=os.path.getmtime)
            # "The latest plan" must never resolve to a run that failed. The plans
            # directory holds both kinds by design — a run that solved thirteen welds
            # before its token expired is kept, because it is worth reading and worth
            # resuming — and the newest file is quite often the one that just broke.
            files = [f for f in files if not _failed(f)]
            path = files[-1] if files else None
        if not path or not os.path.exists(path):
            return {"success": False, "error": f"plan not found for tree_id={tree_id!r} in {pdir}"}
        plan = _json.load(open(path))
    # Refused HERE, at the executor, not merely in whatever UI offered the button.
    # A failed record's tree is a partial artefact the assembler rejected, and the
    # fallback below (`or plan`) would otherwise hand the raw failure record to the
    # executor as if it were a tree. Read it with the trajectory agent's
    # get_calculated_plan(include_failed=True); continue it with resume_planning.
    if str(plan.get("status") or "") == "failed":
        return {"success": False, "status": "failed", "tree_id": plan.get("tree_id"),
                "error": (f"plan {plan.get('tree_id')!r} is a FAILED run and is not "
                          "executable: " + str(plan.get("error") or "no tree")),
                "error_kind": plan.get("error_kind"),
                "num_planned": plan.get("num_planned"),
                "resumable": bool(plan.get("resumable")),
                "note": ("Its computed trajectories are kept. Continue the run with "
                         "resume_planning(tree_id), or plan again.")}
    bt = plan.get("behaviour_tree") or plan
    tid = bt.get("tree_id", os.path.basename(path).replace(".json", ""))
    recorder = get_recorder()

    robots: set = set()
    def _robots_in(node):
        # robot_ns is a BT.CPP port; bt_port also finds it on a legacy node, so a
        # plan saved before the format change still yields its robots.
        if isinstance(node, dict):
            rid = bt_port(node, "robot_ns")
            if rid:
                robots.add(rid)
            for c in bt_children(node):
                _robots_in(c)
    _robots_in(bt.get("root", bt))

    if home_reset and robots:
        # The arms may stand in one of THIS plan's seams (the run it is a replay
        # of stopped there): the reset needs its weld poses to lift off on-axis.
        _cell_agent._remember_weld_poses(cell_id, bt)
        home = _home_reset_tree(cell_id, sorted(robots))
        recorder.start_session(tree_id=home["tree_id"], cell_id=cell_id)
        try:
            home_result = _agent_execute_behaviour_tree(home, **_model_kwargs(model))
        finally:
            recorder.end_session()
        # The plan's first legs start at HOME. An arm the reset could not bring
        # there (no safe planned return, or no clear order) — or a reset that did
        # not complete — would be driven to HOME by the first leg, unchecked, from
        # wherever it stands: the replay does not start.
        unhomed = list(home.get("unhomed") or [])
        home_failed = isinstance(home_result, dict) and home_result.get("success") is False
        if unhomed or home_failed:
            return {"success": False, "tree_id": tid, "executed_nodes": 0,
                    "unhomed": unhomed,
                    "error": ("replay not started: the home reset left "
                              + (", ".join(unhomed) if unhomed else "the cell")
                              + " away from HOME"
                              + ("" if not home_failed else
                                 f" (reset failed: {home_result.get('error')})")
                              + ". Bring the arm(s) home (jog clear of the seam / the "
                              "other arm), then replay again.")}

    recorder.start_session(tree_id=tid, cell_id=cell_id)
    try:
        result = _agent_execute_behaviour_tree(bt, **_model_kwargs(model))
    finally:
        recorder.end_session()
    if isinstance(result, dict):
        result.setdefault("tree_id", tid)
    return result


@mcp.tool()
def list_models() -> dict:
    """
    The models this agent can reason with, and the one it uses by default.

    Returns the provider, the startup default, the model currently in effect,
    and the accepted aliases — the valid values for the `model` argument of
    execute_behaviour_tree and replay_plan. A model only ever reaches the
    failure-recovery reasoner; execution is deterministic.

    Returns:
        {"provider": str, "default": str, "effective": str,
         "region": str, "aliases": {alias: model_id}, "note": str}
    """
    return _describe_models()


@mcp.tool()
def move_robot(cell_id: str, robot_ns: str, joint_positions: list[float],
               duration_s: float = 3.0) -> dict:
    """
    Manually move a SINGLE robot to a target joint configuration (jog).

    Reads the robot's current joints, then dispatches a timed two-point
    trajectory (current -> target) and waits for completion. Call
    get_robot_states first to read current joints and pick a safe target. This
    does NOT plan around obstacles, so use modest, verified moves.

    Args:
        cell_id:         Cell identifier.
        robot_ns:        Robot to move, e.g. "robot1".
        joint_positions: 6 target joint angles (radians).
        duration_s:      Time to reach the target (default 3 s; larger = slower).

    Returns:
        {"success": bool, "executed_nodes": int, "elapsed_s": float}
    """
    if not joint_positions or len(joint_positions) < 6:
        return {"success": False, "error": "joint_positions must have at least 6 values"}
    target = [float(v) for v in joint_positions[:6]]
    # Prefer starting the trajectory at the live pose so the executor's
    # schedule-adherence check does not read the jog as a large deviation.
    traj = [{"positions": target, "time_s": float(duration_s)}]
    try:
        states = get_all_robot_states.invoke({"cell_id": cell_id}).get("robots", {})
        cur = (states.get(robot_ns) or {}).get("joint_positions")
        if cur and len(cur) >= 6:
            traj = [{"positions": [float(v) for v in cur[:6]], "time_s": 0.0},
                    {"positions": target, "time_s": float(duration_s)}]
    except Exception:
        pass
    tree = {"tree_id": f"manual_move_{robot_ns}", "cell_id": cell_id,
            "root": {"type": "ExecuteTrajectory",
                     "ports": {"robot_ns": robot_ns, "operation_id": "manual_move",
                               "operation_type": "move", "trajectory": traj}}}
    recorder = get_recorder()
    recorder.start_session(tree_id=tree["tree_id"], cell_id=cell_id)
    try:
        return _agent_execute_behaviour_tree(tree)
    finally:
        recorder.end_session()


@mcp.tool()
def abort_execution(cell_id: str, reason: str = "stopped by operator") -> dict:
    """
    Stop the behaviour tree this controller is executing. Commands NO robot.

    The "stop the activity" half of the operator's Stop button, on the execution
    side (planning is stopped through the trajectory agent's cancel_planning).
    No further node is dispatched and the run ends with
    ``reason_code: "operator_abort"``.

    IT IS NOT AN EMERGENCY STOP, and the difference is physical: a trajectory a
    controller has already accepted RUNS TO ITS END, so an arm mid-weld finishes
    that motion (seconds) and then stops. Use trigger_estop to halt the arms
    where they are — that cancels the active FollowJointTrajectory goal and
    LATCHES, so nothing dispatches again until it is cleared.

    Args:
        cell_id: Cell identifier (recorded; this controller serves one cell).
        reason:  Recorded on the run's failure and shown to the operator.

    Returns:
        {"success": true, "aborted": bool, "cell_id": str, "reason": str}
        ``aborted`` is false when nothing was executing — Stop on an idle cell is
        not an error.
    """
    result = _cell_agent.abort_active_execution(reason)
    return {"success": True, "cell_id": cell_id, **result,
            "note": ("no behaviour tree is executing" if not result.get("aborted")
                     else "no further nodes will be dispatched; an already "
                          "accepted trajectory finishes (this is not an e-stop)")}


@mcp.tool()
def home_robots(cell_id: str, robots: Optional[list[str]] = None) -> dict:
    """
    Send the cell's arms back to their HOME pose. THIS MOVES THE ROBOTS.

    The reposition half of the operator's Reset. It reuses the same gated
    home-reset the replay path runs, so the return is PLANNED (all arms in one
    coordinated call) and then measured: arms clear of each other go home in
    Parallel, arms that are not are homed one at a time in an order where each
    mover clears the others, and an arm that cannot be cleared in ANY order is
    LEFT WHERE IT STANDS and reported. It is never a blind straight line to HOME
    for every arm at once.

    It also CLEARS A LATCHED E-STOP, necessarily: execution starts by clearing
    the latch, or the robot agents would reject every dispatch and this call
    would report a hollow success while nothing moved. So do not call it to
    recover from an e-stop until the reason for that e-stop is gone.

    Args:
        cell_id: Cell identifier.
        robots:  Arms to home. Omit for every arm declared in the cell's
                 robots.yaml (the usual case).

    Returns:
        {"success": bool, "executed_nodes": int, "elapsed_s": float,
         "robots": [...], "cleared_estop": true}
    """
    targets = list(robots or [])
    if not targets:
        try:
            from agents.cell.robots_util import robot_ids
            targets = robot_ids(cell_id)
        except Exception as exc:
            return {"success": False,
                    "error": f"cannot read the cell's robots: {exc}"}
    if not targets:
        return {"success": False, "error": f"no robots declared for '{cell_id}'"}

    tree = _home_reset_tree(cell_id, sorted(targets))
    recorder = get_recorder()
    recorder.start_session(tree_id=tree["tree_id"], cell_id=cell_id)
    try:
        result = _agent_execute_behaviour_tree(tree)
    finally:
        recorder.end_session()
    if isinstance(result, dict):
        result["robots"] = sorted(targets)
        result["cleared_estop"] = True
        # The arms the reset deliberately left where they stand: not home, so
        # the reset is not a success for them, whatever the tree reported.
        unhomed = list(tree.get("unhomed") or [])
        if unhomed:
            result["unhomed"] = unhomed
            result["success"] = False
            result.setdefault("error", f"{', '.join(unhomed)} left away from HOME "
                                       "(no safe planned return); jog clear and reset again")
    return result


@mcp.tool()
def get_cell_status(cell_id: str) -> dict:
    """
    Get current state of all robots in the cell.

    Reads latest joint positions, end-effector positions, and status strings
    from the robot_status ROS 2 topics. Returns promptly (2-second timeout
    per robot) even when ROS 2 is unavailable.

    Args:
        cell_id: Cell identifier, e.g. "welding_cell_2_small"

    Returns:
        {"robots": {ns: {"joint_positions": [...], "ee_position": [...], "status": str}}}
    """
    return get_all_robot_states.invoke({"cell_id": cell_id})


@mcp.tool()
def trigger_estop(cell_id: str, reason: str) -> dict:
    """
    Trigger emergency stop for all robots in the cell.

    Publishes an ESTOP command to /{cell_id}/estop with the given reason.
    Use immediately on any safety violation or unexpected collision detection.

    Args:
        cell_id: Cell identifier
        reason:  Human-readable reason for the emergency stop

    Returns:
        {"triggered": bool, "cell_id": str, "reason": str}
    """
    return _trigger_estop.invoke({"cell_id": cell_id, "reason": reason})


@mcp.tool()
def get_robot_states(cell_id: str) -> dict:
    """
    Get per-robot joint and end-effector state for all robots in the cell.

    Equivalent to get_cell_status — both return the same robot state snapshot.
    Use this tool when the caller specifically needs kinematic state rather than
    a general health check.

    Args:
        cell_id: Cell identifier, e.g. "welding_cell_2_small"

    Returns:
        {"robots": {ns: {"joint_positions": [...], "ee_position": [...], "status": str}}}
    """
    return get_all_robot_states.invoke({"cell_id": cell_id})


@mcp.tool()
def list_recordings(cell_id: str) -> dict:
    """
    List all .rrd recording files stored for a cell.

    Each recording corresponds to one execute_behaviour_tree session and can be
    replayed by passing its tree_id to execute_behaviour_tree with
    replay_mode=True.

    Args:
        cell_id: Cell identifier

    Returns:
        {"recordings": [{"tree_id": str, "path": str, "size_bytes": int}]}
        When recording is disabled:
        {"recordings": [], "warning": "recording disabled (rerun-sdk not installed)"}
    """
    return get_recorder().list_recordings(cell_id)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Initialise recorder ────────────────────────────────────────────────
    try:
        recorder = RerunRecorder()
    except ImportError:
        recorder = NoOpRecorder()
        print("[CellMCP] rerun-sdk not installed — recording disabled (NoOpRecorder active)", flush=True)

    set_recorder(recorder)

    # ── 2. Start background daemon threads ───────────────────────────────────
    threading.Thread(target=http_server,           daemon=True, name="http-health").start()
    threading.Thread(target=ros2_joint_state_sub,  daemon=True, name="joint-state-sub").start()

    # ── 3. Startup banner ────────────────────────────────────────────────────
    if _TRANSPORT == "stdio":
        print("[CellMCP] Starting stdio transport (offline/IDE mode)", flush=True)
    else:
        print(
            f"[CellMCP] Starting SSE on 0.0.0.0:{_PORT} "
            f"(cell={_CELL_ID}, http={_HTTP_PORT}, base={_BASE_PATH})",
            flush=True,
        )

    # ── 4. Run MCP server ────────────────────────────────────────────────────
    try:
        if _TRANSPORT == "stdio":
            mcp.run(transport="stdio")
        else:
            import uvicorn
            from utils.mcp_http import add_jsonrpc_mcp_route
            app = mcp.sse_app()
            # Plain JSON-RPC /mcp endpoint for HTTP callers (orchestrator, Makefile).
            add_jsonrpc_mcp_route(app, mcp)
            uvicorn.run(app, host="0.0.0.0", port=_PORT, log_level="warning")
    except OSError as exc:
        print(f"[CellMCP] ERROR: port {_PORT} already in use — exiting", flush=True)
        sys.exit(1)
