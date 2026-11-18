"""A planning run's decisions, structured per weld point, persisted for reading.

The planning ReAct loop already printed everything a reader could want — the
reasoning, each tool call with its arguments, each result — to the trajectory
agent's stdout. That is the wrong medium for the question people actually ask,
which is never "what happened at 15:34:06" but:

    "W12 came out wrong. What did the planner decide about W12, in what order,
     and what did it see when it decided it?"

Answering that from a console needs `docker logs` plus grep plus the knowledge
that a weld's identity lives inside a nested ``motion_requests`` array. And the
logs are gone the moment the container is recreated, which happened three times
in one afternoon of debugging.

So the trace is built as a structure instead:

  * **decisions** — one record per tool call, in order, carrying the reasoning
    text that preceded it (the *why*), the arguments (the *what*), and the result
    (the *outcome*). This is the run's history, navigable step by step.
  * **points** — one record per operation in the task, each listing the decisions
    that touched it and rolling up what became of it. This is the same history
    indexed by weld, which is how a reader looks for it.

Every point in the task appears, including the ones nothing was ever decided
about: a weld the planner silently never attempted is a finding, and a view that
only lists what was attempted hides it.

Stored as ``plantrace_<trace_id>.json`` in the cell's ``logs`` kind, through
:mod:`utils.cell_store` — so the trace the trajectory agent produces is written by
the orchestrator, the agent that owns the cell's data. That is what lets the
dashboard serve a trace the planner wrote, and what lets it survive the planner's
container without the planner keeping a copy of the cell's history of its own.

Bulk is stripped, not truncated blindly: a planner result carries thousands of
waypoints, so waypoint arrays become ``{"waypoints": n, "duration_s": t}``. The
numbers a reader needs are kept; the megabytes are not.

Never raises into the planner. A trace that fails to write is a lost trace, and
losing the plan over it would be the worse outcome.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional

from utils import cell_layout, cell_store
from utils.llm_text import extract_text

SCHEMA = "plan_trace/1"

_FILE_PREFIX = "plantrace_"

# Bounds. Generous enough to keep a whole decision readable, small enough that a
# 15-weld run stays a file a browser can hold.
_MAX_TEXT_CHARS = 4000
_MAX_LIST_ITEMS = 40
_MAX_DEPTH = 8

# A waypoint array is recognised by its elements, not by its key: the planner
# calls it "trajectories", the cache "precomputed_trajectory", the tree ports
# something else again.
_WAYPOINT_KEYS = ("positions", "time_s")

_ID_KEYS = ("operation_id", "operation_ids", "operations")
# ``trajectories`` is in here because a planner *result* names its robots only as
# the keys of that map — there is no ``robot_ns`` field anywhere in the reply. Omit
# it and every successful planning decision renders with no robot attributed.
_ROBOT_KEYS = ("robot_ns", "robot", "robots", "trajectories")


# ─── Shrinking payloads ───────────────────────────────────────────────────────

def _is_waypoints(value: Any) -> bool:
    return (
        isinstance(value, list) and len(value) > 2
        and isinstance(value[0], dict)
        and any(k in value[0] for k in _WAYPOINT_KEYS)
    )


def _waypoint_summary(traj: list) -> dict:
    last = traj[-1] if isinstance(traj[-1], dict) else {}
    return {"waypoints": len(traj),
            "duration_s": round(float(last.get("time_s") or 0.0), 3)}


def compact(value: Any, depth: int = 0) -> Any:
    """A JSON-safe, bounded copy of ``value`` with waypoint arrays summarised."""
    if depth > _MAX_DEPTH:
        return "…"
    if _is_waypoints(value):
        return _waypoint_summary(value)
    if isinstance(value, dict):
        return {str(k): compact(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [compact(v, depth + 1) for v in list(value)[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"…+{len(value) - _MAX_LIST_ITEMS} more")
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > _MAX_TEXT_CHARS:
            return value[:_MAX_TEXT_CHARS] + f"… [{len(value) - _MAX_TEXT_CHARS} more chars]"
        return value
    return str(value)


# ─── Reading identity out of a payload ────────────────────────────────────────

def _collect(obj: Any, keys: tuple[str, ...], out: list[str], depth: int = 0) -> None:
    """Depth-first collect of string values under any of ``keys``, in order."""
    if depth > _MAX_DEPTH:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                if isinstance(value, str):
                    if value not in out:
                        out.append(value)
                elif isinstance(value, dict):
                    for name in value:
                        if isinstance(name, str) and name not in out:
                            out.append(name)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str) and item not in out:
                            out.append(item)
                        elif isinstance(item, dict):
                            _collect(item, keys, out, depth + 1)
            else:
                _collect(value, keys, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect(item, keys, out, depth + 1)


def operation_ids(payload: Any) -> list[str]:
    """Every operation id mentioned anywhere in a tool's args or result.

    The id is what ties a decision to a weld, and it hides at different depths
    per tool: ``operation_id`` at the top for ``validate_approach``, one per
    element of ``motion_requests`` for the planner, a list for the assembler.
    One walk covers all three, so a new tool is indexed without being taught.
    """
    found: list[str] = []
    _collect(payload, _ID_KEYS, found)
    return found


def robot_names(payload: Any) -> list[str]:
    found: list[str] = []
    _collect(payload, _ROBOT_KEYS, found)
    return found


# ─── Summarising a result ─────────────────────────────────────────────────────

_HEADLINE_KEYS = (
    "planning_time_s", "position_error_m", "orientation_error_deg",
    "tool_axis_error_deg", "approach_direction_error_deg", "matches",
    "collision_free", "goal_collision_free", "min_clearance_m",
    "needs_interlock", "strategy", "reachable", "total_steps", "tree_id",
)


def _result_metrics(payload: Any) -> dict:
    """The handful of numbers a reader scans a result for."""
    if not isinstance(payload, dict):
        return {}
    metrics = {k: compact(payload[k]) for k in _HEADLINE_KEYS if k in payload}
    traj = payload.get("trajectories")
    if isinstance(traj, dict):
        metrics["waypoints"] = {
            # A list is a raw leg; a dict is the agent-facing summary of one
            # (agent.agent_view), which already carries the count.
            robot: (len(points) if isinstance(points, list)
                    else points["waypoints"] if isinstance(points, dict)
                    and isinstance(points.get("waypoints"), int)
                    else compact(points))
            for robot, points in traj.items()
        }
    return metrics


def _result_ok(payload: Any) -> Optional[bool]:
    if isinstance(payload, dict):
        for key in ("success", "matches", "valid"):
            if isinstance(payload.get(key), bool):
                return payload[key]
    return None


def _parse_result(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


# ─── Building the trace ───────────────────────────────────────────────────────

def _decisions(messages: list) -> list[dict]:
    """One record per tool call: the reasoning behind it, its args, its result."""
    # Results are matched to their call by id, not by position: LangGraph emits a
    # parallel step's calls together and their results in whatever order the tools
    # finished, so pairing by arrival order silently mislabels concurrent welds.
    results: dict[str, Any] = {}
    for msg in messages:
        call_id = getattr(msg, "tool_call_id", None)
        if call_id:
            results[call_id] = msg

    decisions: list[dict] = []
    pending_why = ""
    step = 0

    for msg in messages:
        role = getattr(msg, "type", "") or msg.__class__.__name__.lower()
        if "ai" not in role:
            continue

        text = extract_text(getattr(msg, "content", "")).strip()
        calls = getattr(msg, "tool_calls", None) or []

        if text and not calls:
            # Reasoning with no call attached: either the run's closing statement
            # or a pause between steps. Kept as its own record so a final "I give
            # up because…" is never lost, which is exactly what used to happen.
            decisions.append({
                "seq": len(decisions) + 1, "kind": "note", "text": compact(text),
            })
            pending_why = ""
            continue

        if text:
            pending_why = text

        for call in calls:
            step += 1
            args = call.get("args") or {}
            record: dict = {
                "seq":        len(decisions) + 1,
                "kind":       "decision",
                "step":       step,
                "tool":       call.get("name") or "?",
                "why":        compact(pending_why),
                "args":       compact(args),
                "operations": operation_ids(args),
                "robots":     robot_names(args),
            }
            pending_why = ""

            reply = results.get(call.get("id"))
            if reply is not None:
                payload = _parse_result(
                    extract_text(getattr(reply, "content", "")).strip())
                record["result"] = {
                    "ok":      _result_ok(payload),
                    "metrics": _result_metrics(payload),
                    "payload": compact(payload),
                }
                failures = payload.get("failures") if isinstance(payload, dict) else None
                if isinstance(failures, list) and failures:
                    record["result"]["failures"] = compact(failures)
                for op in operation_ids(payload):
                    if op not in record["operations"]:
                        record["operations"].append(op)
                for robot in robot_names(payload):
                    if robot not in record["robots"]:
                        record["robots"].append(robot)
            else:
                # A call with no result is the signature of a run that was
                # cancelled or truncated mid-step. Say so rather than implying it
                # returned nothing.
                record["result"] = {"ok": None, "metrics": {},
                                    "payload": None, "unanswered": True}
            decisions.append(record)

    return decisions


def _point_status(flags: dict) -> str:
    """The verdict to show for a point.

    Deliberately the LAST verdict recorded, not the best one: a weld that planned
    and was then rejected is not "planned", and a view that reports the high-water
    mark of a point's history is the kind of optimism that hides a regression. The
    individual flags stay alongside it, so the whole history is still legible.
    """
    return flags.get("last") or "pending"


def _index_points(operations: list[dict], decisions: list[dict]) -> list[dict]:
    """Every task operation, with the decisions that touched it rolled up."""
    points: dict[str, dict] = {}
    order: list[str] = []

    for op in operations or []:
        op_id = str(op.get("id") or op.get("operation_id") or f"op{len(order) + 1}")
        position = op.get("position")
        if position is None and all(k in op for k in "xyz"):
            position = [op.get("x"), op.get("y"), op.get("z")]
        points[op_id] = {
            "operation_id": op_id,
            "type":         op.get("type"),
            "position":     compact(position),
            "robot":        op.get("robot_ns") or op.get("robot"),
            "decisions":    [],
            "attempts":     0,
            "flags":        {},
            "metrics":      {},
            "failures":     [],
        }
        order.append(op_id)

    def _ensure(op_id: str) -> dict:
        if op_id not in points:
            # A decision naming an operation the task never listed. Surfaced, not
            # dropped: it means the planner invented an id, and that is worth
            # seeing rather than silently filtering out.
            points[op_id] = {"operation_id": op_id, "type": None, "position": None,
                             "robot": None, "decisions": [], "attempts": 0,
                             "flags": {"unlisted": True}, "metrics": {},
                             "failures": []}
            order.append(op_id)
        return points[op_id]

    for record in decisions:
        if record.get("kind") != "decision":
            continue
        tool = record.get("tool") or ""
        result = record.get("result") or {}
        ok = result.get("ok")
        failed_ids = {
            f.get("operation_id") for f in (result.get("failures") or [])
            if isinstance(f, dict)
        }

        for op_id in record.get("operations") or []:
            point = _ensure(op_id)
            point["decisions"].append(record["seq"])
            point["attempts"] += 1
            if record.get("robots") and not point.get("robot"):
                point["robot"] = record["robots"][0]

            verdict = None
            if op_id in failed_ids or ok is False:
                verdict = "failed"
                point["failures"].extend(
                    compact(f) for f in (result.get("failures") or [])
                    if isinstance(f, dict) and f.get("operation_id") in (op_id, None)
                )
            elif "validate" in tool and ok:
                verdict = "validated"
            elif "assemble" in tool and ok:
                verdict = "in_tree"
            elif ok:
                verdict = "planned"

            if verdict:
                point["flags"][verdict] = True
                point["flags"]["last"] = verdict
            point["metrics"].update(result.get("metrics") or {})

    out = []
    for op_id in order:
        point = points[op_id]
        point["status"] = _point_status(point["flags"])
        out.append(point)
    return out


def build(
    cell_id: str,
    messages: list,
    operations: list[dict],
    result: dict,
    *,
    trace_id: Optional[str] = None,
    tree_id: Optional[str] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    started_at: Optional[datetime] = None,
    step_budget: Optional[int] = None,
) -> dict:
    """Assemble the structured trace of one planning run."""
    decisions = _decisions(messages)
    points = _index_points(operations, decisions)
    finished = datetime.now()
    counted = sum(1 for d in decisions if d.get("kind") == "decision")

    by_status: dict[str, int] = {}
    for point in points:
        by_status[point["status"]] = by_status.get(point["status"], 0) + 1

    return {
        "schema": SCHEMA,
        "run": {
            "trace_id":    trace_id or f"{cell_id}_{finished.strftime('%Y%m%d_%H%M%S')}",
            "cell_id":     cell_id,
            "tree_id":     tree_id or (result or {}).get("tree_id"),
            "backend":     backend,
            "model":       model or (result or {}).get("model"),
            "started_at":  (started_at or finished).isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "elapsed_s":   (result or {}).get("planning_time_s"),
            "success":     bool((result or {}).get("success")),
            "error":       (result or {}).get("error"),
            "error_kind":  (result or {}).get("error_kind"),
            "note":        (result or {}).get("note"),
            "tool_calls":  counted,
            "step_budget": step_budget,
            "operations":  len(points),
            "points_by_status": by_status,
        },
        "points":    points,
        "decisions": decisions,
    }


# ─── Persistence ──────────────────────────────────────────────────────────────

def trace_name(trace_id: str) -> str:
    """The artefact name of one trace — the only naming this module does now."""
    return f"{_FILE_PREFIX}{trace_id}.json"


def path_for(cell_id: str, trace_id: str) -> "Any":
    """Where the trace sits in the tree of whoever owns the cell's data.

    A LOCAL answer: true in the orchestrator (which holds the volume) and in a
    test, and in the planner it names the file on the owner's disk rather than on
    its own. Kept because callers report *where the trace is* to a human, and a
    bare artefact name is not an answer to that.
    """
    return cell_layout.logs_dir(cell_id) / trace_name(trace_id)


def save(cell_id: str, trace: dict) -> Optional[str]:
    """Write the trace; return where it landed, or None if it could not be.

    The write goes through :mod:`utils.cell_store` — in the planner that means the
    orchestrator performs it — while the returned string is the path in the owning
    cell tree, which is what a reader is told and what the dashboard links to.
    """
    try:
        trace_id = (trace.get("run") or {}).get("trace_id") or "unknown"
        name = trace_name(trace_id)
        where = str(path_for(cell_id, trace_id))
        cell_store.write_text(cell_id, cell_store.KIND_LOGS, name,
                              json.dumps(trace, indent=2, default=str))
        return where
    except Exception as exc:  # a lost trace must never cost the plan
        print(f"[plan] trace not saved ({exc})", flush=True)
        return None


def capture(
    cell_id: str,
    messages: list,
    operations: list[dict],
    result: dict,
    **kwargs,
) -> Optional[str]:
    """Build and save in one call — the planner's single entry point."""
    try:
        return save(cell_id, build(cell_id, messages, operations, result, **kwargs))
    except Exception as exc:
        print(f"[plan] trace not built ({exc})", flush=True)
        return None


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def is_safe_id(trace_id: str) -> bool:
    """Reject anything that could escape the logs directory."""
    return bool(trace_id) and bool(_SAFE_ID.match(trace_id)) and ".." not in trace_id


def read(cell_id: str, trace_id: str) -> Optional[dict]:
    """One saved trace, or None if it is absent, unsafe or unreadable."""
    if not is_safe_id(trace_id):
        return None
    try:
        return cell_store.read_json(cell_id, cell_store.KIND_LOGS,
                                    trace_name(trace_id))
    except Exception:  # noqa: BLE001 - unreadable and unreachable both read as absent
        return None


def listing(cell_id: str, limit: int = 50) -> list[dict]:
    """The ``run`` block of each saved trace, newest first — the history list.

    Metadata only. A full trace is the whole decision history and is not what a
    picker needs to render its rows.
    """
    out: list[dict] = []
    try:
        rows = cell_store.summarize(cell_id, cell_store.KIND_LOGS, ["run"],
                                    count_fields=["points", "decisions"],
                                    prefix=_FILE_PREFIX, suffix=".json",
                                    limit=max(1, limit))
    except Exception as exc:  # noqa: BLE001
        print(f"[plan] traces not listed ({exc})", flush=True)
        return out
    for row in rows:
        meta = row.get("run")
        if isinstance(meta, dict):
            # Only the counts a picker renders travel; the decision history stays
            # where it is until someone opens the trace.
            out.append({**meta,
                        "points": row.get("points_count") or 0,
                        "decisions": row.get("decisions_count") or 0})
    return out
