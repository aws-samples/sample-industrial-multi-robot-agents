"""
Run logger for the orchestrator.

Captures, for each A2A task run, a timestamped record of:
  1. The LLM reasoning + actions (the ReAct message trace: assistant reasoning,
     tool calls with arguments, and tool results)
  2. The trajectory planning result (success / planning time / error)
  3. A summary of the assembled Behaviour Tree (node counts, per-robot
     operations, waypoints, interlocks, estimated makespan)

Two files are written per run into RUN_LOG_DIR (default: the cell's agent
workspace — ``utils.cell_layout.logs_dir``):

  run_<cell>_<YYYYMMDD_HHMMSS>.json   full structured log (machine-readable)
  run_<cell>_<YYYYMMDD_HHMMSS>.md     human-readable summary

Logging never raises into the caller: any failure is caught and reported so a
logging problem can't break task execution.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.llm_model import current_provider, effective_model_id
from utils import cell_layout
from agents.trajectory.tools.btcpp import (
    children as bt_children,
    node_kind,
    port as bt_port,
    resolve as bt_resolve,
)

# Tool result content longer than this is truncated in the reasoning trace
# (the Behaviour Tree is captured separately in structural form).
_MAX_TOOL_CONTENT_CHARS = 2000


# ─── Path resolution ──────────────────────────────────────────────────────────

def _log_dir(cell_id: str) -> Path:
    """Resolve the directory for run logs.

    Uses RUN_LOG_DIR if set, otherwise the cell's agent-owned logs directory —
    ``cell_layout`` decides whether that is under ``agentic_planner/`` or flat.
    """
    explicit = os.environ.get("RUN_LOG_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return cell_layout.logs_dir(cell_id)


# ─── Run identity ─────────────────────────────────────────────────────────────
#
# The run id has ONE owner: this module. Two callers need it now — the A2A server
# mints one BEFORE dispatching an asynchronous run (so it has something to hand
# back for ``GET /a2a/result/<run_id>``), and ``save_run_log`` names the log files
# after it. A second implementation of the format would break the single property
# every reader depends on: that the id alone locates its own log on disk.

_RUN_ID_TIME_FMT = "%Y%m%d_%H%M%S"


def new_run_id(cell_id: str, when: Optional[datetime] = None) -> str:
    """Mint a run id of the form ``<cell_id>_<YYYYMMDD>_<HHMMSS>``."""
    return f"{cell_id}_{(when or datetime.now()).strftime(_RUN_ID_TIME_FMT)}"


def cell_id_of_run(run_id: str) -> str:
    """Recover the cell id a run id was minted for.

    The id is ``{cell_id}_{YYYYMMDD}_{HHMMSS}`` and cell ids themselves contain
    underscores, so the split is from the RIGHT: exactly two trailing fields are
    the timestamp. This is what lets ``GET /a2a/result/<run_id>`` find the log
    without the caller having to say which cell the run belonged to.
    """
    parts = run_id.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else run_id


def run_log_path(run_id: str, cell_id: Optional[str] = None) -> Path:
    """Where :func:`save_run_log` wrote (or will write) this run's JSON log."""
    return _log_dir(cell_id or cell_id_of_run(run_id)) / f"run_{run_id}.json"


def read_run_log(run_id: str, cell_id: Optional[str] = None) -> Optional[dict]:
    """The persisted log for a run, or None if it is absent or unreadable.

    Unreadable is folded into absent on purpose: a truncated log (a run killed
    mid-write) proves nothing about the run, and the caller's not-found branch is
    the honest answer.
    """
    path = run_log_path(run_id, cell_id)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def list_run_logs(cell_id: str, limit: int = 50) -> list[dict]:
    """Run metadata for a cell, newest first — the ``run`` block of each log.

    Only the metadata block is returned: a full log carries the whole reasoning
    trace and the stripped tree, which is megabytes per run and not what a
    listing is for.
    """
    out: list[dict] = []
    directory = _log_dir(cell_id)
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("run_*.json"), reverse=True)[:limit]:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        meta = record.get("run")
        if isinstance(meta, dict):
            out.append({**meta, "log_json": str(path)})
    return out


# ─── Message-trace extraction ─────────────────────────────────────────────────

def _msg_role(message: Any) -> str:
    return getattr(message, "type", message.__class__.__name__)


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def serialize_trace(messages: list) -> list[dict]:
    """Turn a LangGraph/LangChain message list into a JSON-friendly trace.

    Each entry captures the role, any reasoning text, tool calls (name + args),
    and tool results. Long tool results are truncated for readability.
    """
    trace: list[dict] = []
    for message in messages:
        role = _msg_role(message)
        entry: dict = {"role": role}

        content = getattr(message, "content", "")
        if content:
            text = _stringify(content)
            if role == "tool" and len(text) > _MAX_TOOL_CONTENT_CHARS:
                entry["content"] = text[:_MAX_TOOL_CONTENT_CHARS] + \
                    f"... [truncated {len(text) - _MAX_TOOL_CONTENT_CHARS} chars]"
                entry["content_truncated"] = True
            else:
                entry["content"] = text

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in tool_calls
            ]

        name = getattr(message, "name", None)
        if name and role == "tool":
            entry["tool_name"] = name

        trace.append(entry)
    return trace


def extract_trajectory_result(messages: list) -> Optional[dict]:
    """Find the most recent `call_trajectory_agent` tool result in the trace."""
    for message in reversed(messages):
        if _msg_role(message) == "tool" and getattr(message, "name", "") == "call_trajectory_agent":
            try:
                return json.loads(_stringify(getattr(message, "content", "{}")))
            except (json.JSONDecodeError, ValueError):
                return {"raw": _stringify(getattr(message, "content", ""))}
    return None


# ─── Behaviour Tree summarisation ─────────────────────────────────────────────

def _summarise_waypoints(traj: list) -> dict:
    duration = traj[-1].get("time_s", 0.0) if traj else 0.0
    return {"waypoints": len(traj), "duration_s": round(duration, 3)}


def _strip_trajectories(node: Any) -> Any:
    """Deep-copy a BT node, replacing bulky waypoint arrays with counts."""
    if not isinstance(node, dict):
        return node
    out = copy.deepcopy(node)
    for holder in (out, out.get("ports")):
        if not isinstance(holder, dict):
            continue
        for key in ("precomputed_trajectory", "trajectory"):
            traj = holder.get(key)
            if isinstance(traj, list):
                holder[key] = _summarise_waypoints(traj)
    if isinstance(out.get("children"), list):
        out["children"] = [_strip_trajectories(c) for c in out["children"]]
    return out


def strip_tree(tree: Any) -> Any:
    """Strip a whole tree: node trajectories AND the blackboard they point at.

    In a BehaviorTree.CPP tree the waypoint arrays live on the blackboard, not on
    the nodes, so stripping only the nodes would leave the bulk in the log.
    """
    if not isinstance(tree, dict):
        return tree
    out = dict(tree)
    if isinstance(tree.get("root"), dict):
        out["root"] = _strip_trajectories(tree["root"])
    bb = tree.get("blackboard")
    if isinstance(bb, dict):
        out["blackboard"] = {
            k: (_summarise_waypoints(v)
                if isinstance(v, list) and v and isinstance(v[0], dict)
                and "positions" in v[0] else v)
            for k, v in bb.items()
        }
    out.pop("xml", None)          # the XML repeats the structure verbatim
    return out


def summarize_behaviour_tree(bt: Optional[dict]) -> dict:
    """Produce a compact summary of a Behaviour Tree.

    Returns node-type counts, per-robot operation counts, operation IDs,
    total waypoints, interlock count, and an estimated makespan (the longest
    single-operation duration).
    """
    if not isinstance(bt, dict):
        return {"available": False}

    node_counts: dict[str, int] = {}
    executions: list[dict] = []
    total_waypoints = 0
    interlock_count = 0
    max_duration = 0.0
    blackboard = bt.get("blackboard") or {}

    def walk(node: Any) -> None:
        nonlocal total_waypoints, interlock_count, max_duration
        if not isinstance(node, dict):
            return
        ntype = node_kind(node) or "?"
        node_counts[ntype] = node_counts.get(ntype, 0) + 1

        if ntype == "ExecuteTrajectory":
            traj = bt_resolve(bt_port(node, "trajectory"), blackboard)
            # A stripped tree replaced the array with {"waypoints": n, ...}, so
            # accept either shape rather than reporting 0 for a logged tree.
            if isinstance(traj, dict):
                n_wp = int(traj.get("waypoints", 0) or 0)
                duration = float(traj.get("duration_s", 0.0) or 0.0)
            else:
                traj = traj if isinstance(traj, list) else []
                n_wp = len(traj)
                duration = traj[-1].get("time_s", 0.0) if traj else 0.0
            max_duration = max(max_duration, duration)
            total_waypoints += n_wp
            if bt_port(node, "has_interlock"):
                interlock_count += 1
            executions.append({
                "operation_id":   bt_port(node, "operation_id"),
                "operation_type": bt_port(node, "operation_type"),
                "robot_ns":       bt_port(node, "robot_ns"),
                "waypoints":      n_wp,
                "duration_s":     round(duration, 3),
                "has_interlock":  bool(bt_port(node, "has_interlock")),
            })

        for child in bt_children(node):
            walk(child)

    root = bt.get("root")
    if root is not None:
        walk(root)

    per_robot: dict[str, int] = {}
    for ex in executions:
        per_robot[ex["robot_ns"]] = per_robot.get(ex["robot_ns"], 0) + 1

    return {
        "available":             True,
        "tree_id":               bt.get("tree_id"),
        "node_counts":           node_counts,
        "execute_count":         len(executions),
        "operations_per_robot":  per_robot,
        "operation_ids":         [e["operation_id"] for e in executions],
        "total_waypoints":       total_waypoints,
        "interlock_count":       interlock_count,
        "estimated_makespan_s":  round(max_duration, 3),
        "executions":            executions,
    }


# ─── Markdown rendering ───────────────────────────────────────────────────────

def _render_markdown(record: dict) -> str:
    meta = record["run"]
    traj = record.get("trajectory_result") or {}
    bt   = record.get("behaviour_tree_summary") or {}
    lines: list[str] = []

    lines.append(f"# Task run — {meta['program']} @ {meta['cell_id']}")
    lines.append("")
    lines.append(f"- **Run ID:** {meta['run_id']}")
    lines.append(f"- **Timestamp:** {meta['timestamp']}")
    lines.append(f"- **Status:** {'SUCCESS' if meta['success'] else 'FAILED'}")
    lines.append(f"- **Elapsed:** {meta['elapsed_s']} s")
    lines.append(f"- **LLM:** {meta['llm_provider']} ({meta['llm_model']})")
    if meta.get("error"):
        lines.append(f"- **Error:** {meta['error']}")
    lines.append("")

    # Trajectory planning
    lines.append("## Trajectory planning")
    lines.append("")
    if traj:
        lines.append(f"- **Success:** {traj.get('success')}")
        if "planning_time_s" in traj:
            lines.append(f"- **Planning time:** {traj.get('planning_time_s')} s")
        if traj.get("error"):
            lines.append(f"- **Error:** {traj.get('error')}")
    else:
        lines.append("_No trajectory planning result was recorded._")
    lines.append("")

    # Behaviour tree summary
    lines.append("## Behaviour tree summary")
    lines.append("")
    if bt.get("available"):
        lines.append(f"- **Tree ID:** {bt.get('tree_id')}")
        lines.append(f"- **Execute nodes:** {bt.get('execute_count')}")
        lines.append(f"- **Node counts:** {bt.get('node_counts')}")
        lines.append(f"- **Operations per robot:** {bt.get('operations_per_robot')}")
        lines.append(f"- **Total waypoints:** {bt.get('total_waypoints')}")
        lines.append(f"- **Interlocks:** {bt.get('interlock_count')}")
        lines.append(f"- **Estimated makespan:** {bt.get('estimated_makespan_s')} s")
        execs = bt.get("executions") or []
        if execs:
            lines.append("")
            lines.append("| Operation | Type | Robot | Waypoints | Duration (s) | Interlock |")
            lines.append("|---|---|---|---|---|---|")
            for e in execs:
                lines.append(
                    f"| {e['operation_id']} | {e['operation_type']} | {e['robot_ns']} "
                    f"| {e['waypoints']} | {e['duration_s']} | {'yes' if e['has_interlock'] else 'no'} |"
                )
    else:
        lines.append("_No behaviour tree was produced._")
    lines.append("")

    # Behaviour tree structure (the actual BehaviorTree.CPP tree — Sequence /
    # Parallel / Delay / ExecuteTrajectory / WaitForRobot / MoveHome — so the
    # coordination reads at a glance).
    def _render_bt(node, depth=0, blackboard=None):
        out: list[str] = []
        if not isinstance(node, dict):
            return out
        ind = "  " * depth
        t = node_kind(node)
        if t == "ExecuteTrajectory":
            tr = bt_resolve(bt_port(node, "trajectory"), blackboard)
            if isinstance(tr, dict):
                wp, dur = tr.get("waypoints", "?"), tr.get("duration_s", "?")
            elif isinstance(tr, list) and tr:
                wp, dur = len(tr), round(tr[-1].get("time_s", 0), 2)
            else:
                wp, dur = "?", "?"
            il = " +interlock" if bt_port(node, "has_interlock") else ""
            out.append(f"{ind}ExecuteTrajectory  {bt_port(node, 'robot_ns')}  "
                       f"{bt_port(node, 'operation_id')} "
                       f"({bt_port(node, 'operation_type')})  [{wp} wp, {dur}s]{il}")
        elif t == "WaitForRobot":
            deps = (bt_resolve(bt_port(node, "depends_on_joints"), blackboard)
                    or bt_resolve(bt_port(node, "depends_on_positions"), blackboard) or [])
            who = ", ".join(str(d.get("robot", "?")) for d in deps)
            out.append(f"{ind}WaitForRobot  {bt_port(node, 'robot_ns')}  "
                       f"until [{who}] retracts/clears")
        elif t == "MoveHome":
            out.append(f"{ind}MoveHome  {bt_port(node, 'robot_ns')}")
        elif t == "CheckSafety":
            out.append(f"{ind}CheckSafety ({bt_port(node, 'condition')})")
        else:  # Sequence / Parallel / Delay / root
            if t:
                label = t
                if t == "Delay":
                    label = f"Delay ({bt_port(node, 'delay_msec', 0)} ms)"
                out.append(f"{ind}{label}")
            for c in bt_children(node):
                out.extend(_render_bt(c, depth + (1 if t else 0), blackboard))
        return out

    struct = record.get("behaviour_tree_structure")
    lines.append("## Behaviour tree structure")
    lines.append("")
    if isinstance(struct, dict):
        lines.append("```")
        lines.extend(_render_bt(struct.get("root", struct),
                                blackboard=struct.get("blackboard") or {}))
        lines.append("```")
    else:
        lines.append("_Tree structure not captured._")
    lines.append("")

    # LLM reasoning & actions
    lines.append("## LLM reasoning & actions")
    lines.append("")
    step = 0
    for entry in record.get("reasoning_trace", []):
        role = entry.get("role")
        if role == "human":
            lines.append(f"**Request:** {entry.get('content', '')}")
            lines.append("")
        elif role == "ai":
            step += 1
            if entry.get("content"):
                lines.append(f"**Step {step} — reasoning:** {entry['content']}")
            for tc in entry.get("tool_calls", []):
                lines.append(f"    - action: `{tc['name']}` args={json.dumps(tc['args'], default=str)}")
            lines.append("")
        elif role == "tool":
            content = entry.get("content", "")
            lines.append(f"    - result of `{entry.get('tool_name', '?')}`: {content}")
            lines.append("")
    lines.append("")
    return "\n".join(lines)


# ─── Public entry point ───────────────────────────────────────────────────────

def save_run_log(
    cell_id: str,
    program_name: str,
    messages: list,
    elapsed_s: float,
    success: bool,
    error: str = "",
    run_id: Optional[str] = None,
) -> dict:
    """Write timestamped JSON + Markdown logs for a task run.

    ``run_id`` is the id the run was dispatched under. Pass it whenever the caller
    handed one out before the run started (the A2A server does, so an async caller
    can poll ``/a2a/result/<run_id>``); a log named after a freshly minted id would
    be unreachable by the id the caller holds. Omit it and one is minted here.

    Returns {"json": path, "markdown": path} on success, or
    {"error": str} if logging failed (never raises).
    """
    try:
        now = datetime.now()
        run_id = run_id or new_run_id(cell_id, now)

        trace = serialize_trace(messages)
        traj_result = extract_trajectory_result(messages)
        # The tree is now referenced (not inlined) to keep it out of the LLM;
        # resolve the ref from the in-process plan store for logging. Fall back
        # to an inlined tree for backward compatibility.
        bt = (traj_result or {}).get("behaviour_tree")
        if bt is None and traj_result is not None:
            ref = traj_result.get("behaviour_tree_ref")
            if ref:
                try:
                    from agents.orchestrator.tools.plan_store import get_plan
                    bt = get_plan(ref)
                except Exception:
                    bt = None
        bt_summary = summarize_behaviour_tree(bt)

        # Trajectory result metadata without the bulky BT payload.
        traj_meta = None
        if traj_result is not None:
            traj_meta = {k: v for k, v in traj_result.items() if k != "behaviour_tree"}

        record = {
            "run": {
                "run_id":       run_id,
                "cell_id":      cell_id,
                "program":      program_name,
                "timestamp":    now.isoformat(timespec="seconds"),
                "success":      success,
                "elapsed_s":    elapsed_s,
                "error":        error,
                # The model ACTUALLY used, including a per-request override — the
                # environment variable alone would log the container default and
                # misattribute every run that asked for something else.
                "llm_provider": current_provider(),
                "llm_model":    effective_model_id(),
            },
            "trajectory_result":       traj_meta,
            "behaviour_tree_summary":  bt_summary,
            "behaviour_tree_structure": strip_tree(bt) if bt else None,
            "reasoning_trace":         trace,
        }

        out_dir = _log_dir(cell_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"run_{run_id}.json"
        md_path   = out_dir / f"run_{run_id}.md"

        json_path.write_text(json.dumps(record, indent=2, default=str))
        md_path.write_text(_render_markdown(record))

        print(f"[RunLogger] Saved run log: {json_path}", flush=True)
        return {"json": str(json_path), "markdown": str(md_path)}

    except Exception as exc:  # never let logging break the task
        print(f"[RunLogger] WARNING: failed to write run log: {exc}", flush=True)
        return {"error": str(exc)}
