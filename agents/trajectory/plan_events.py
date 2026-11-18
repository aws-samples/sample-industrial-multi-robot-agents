"""What the planner is doing, right now, as a stream a reader can follow.

:mod:`plan_trace` answers "what happened in that run" after the fact. This module
answers the other question — "what is it doing *now*" — while a run that takes
minutes is still going: this weld started, that one is planned, the model just
changed its approach strategy, the run paused because the token expired.

Two channels, one vocabulary
----------------------------
The planner runs in the trajectory-agent container and the dashboard talks to the
orchestrator, so an event has to cross a process boundary twice. Each event goes
out two ways, and the split is deliberate:

  * **DDS, for watching.** Every event is published as ``std_msgs/String`` JSON on
    ``/<cell_id>/plan_progress`` (see :mod:`agents.trajectory.plan_progress`). The
    dashboard already owns a ROS node and already subscribes to this cell's other
    topics, so following a live run costs it no polling and no hop through anyone
    else's HTTP. This is the visualisation path.
  * **An append-only JSONL artefact, for keeping.** The same event is appended to
    ``planevents_<trace_id>.jsonl`` in the cell's ``logs`` kind through
    :mod:`utils.cell_store` — which means the ORCHESTRATOR writes it, since it is
    the agent that owns the cell's data. One line per event, so a reader that has
    seen ``seq`` can ask for what came after it without re-reading the run. This is
    the durable path: it survives the container, which is the whole point when the
    thing being debugged is a container that died, and it is what answers a reader
    who arrives after the run (DDS history is bounded, a file is not).

A DDS subscriber that missed the first half of a run and a reader with no ROS at
all therefore get the same answer from the same events, in the same shape.

What counts as an event
-----------------------
Only facts the planner actually knows, never a guess at intent:

  * ``run_started`` / ``finished`` / ``aborted`` — the run's own boundaries,
  * ``operation_started`` / ``operation_planned`` / ``operation_failed`` /
    ``operation_validated`` — a weld's progress, keyed by ``operation_id``,
  * ``strategy`` — a *derived* fact, but a hard one: the planner re-attempted an
    operation with different arguments. The changed arguments are the evidence and
    they are carried in the event, so "changed strategy" is never editorial —
    ``approach_offset_m 0.1 → 0.07`` is what happened,
  * ``think`` — the reasoning text between steps, so the feed reads as a narrative
    and not as a list of function names,
  * ``paused`` / ``resumed`` / ``checkpoint`` — see :mod:`plan_checkpoint`.

Never raises into the planner. A dropped event is a gap in a log; an exception
here would cost the plan.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.trajectory import plan_progress
from utils import cell_layout, cell_store

SCHEMA = "plan_events/1"

_FILE_PREFIX = "planevents_"
_SUFFIX = ".jsonl"

# A single event must stay one short line: this is a live feed, not the trace.
_MAX_TEXT = 400
_MAX_ARGS = 12

_LOCK = threading.Lock()
# trace_id -> next seq. Process-local; the file is the durable record, and a
# restarted writer continues from the file's own length (see _next_seq).
_SEQ: dict[str, int] = {}
# (trace_id, operation_id) -> the args of the last attempt, for change detection.
_LAST_ATTEMPT: dict[tuple, dict] = {}

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def is_safe_id(trace_id: str) -> bool:
    """Reject anything that could escape the logs directory."""
    return bool(trace_id) and bool(_SAFE_ID.match(trace_id)) and ".." not in trace_id


def event_name(trace_id: str) -> str:
    """The artefact name of one run's stream — how every reader asks for it."""
    return f"{_FILE_PREFIX}{trace_id}{_SUFFIX}"


def path_for(cell_id: str, trace_id: str) -> Path:
    """Where the stream sits on the disk of whoever owns the cell's data.

    LOCAL answer only — meaningful in the orchestrator (which holds the volume) and
    in a test, and meaningless in the planner, which reaches the same artefact by
    name through :mod:`utils.cell_store`. Kept because the local owner does have a
    path and code that inspects the tree should not have to rebuild the name.
    """
    return cell_layout.logs_dir(cell_id) / event_name(trace_id)


def _store_event(cell_id: str, trace_id: str, line: str) -> None:
    """Append one line to the run's durable stream, on whichever side owns it.

    Remote: one ``cell_append`` to the orchestrator. Local: this process IS the
    owner, so it writes through :func:`path_for` — the module's own declaration of
    where its stream lives, and the seam the orchestrator's routes and the tests
    both hold. Same ``flush``+``fsync`` as the store's own appender, because the
    reader is a different process polling a run that is still going.
    """
    if cell_store.is_remote():
        cell_store.append_text(cell_id, cell_store.KIND_LOGS,
                               event_name(trace_id), line)
        return
    path = path_for(cell_id, trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _short(value: Any) -> Any:
    """A one-line-safe rendering of an event field."""
    if isinstance(value, str):
        collapsed = " ".join(value.split())
        return collapsed if len(collapsed) <= _MAX_TEXT else collapsed[:_MAX_TEXT] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _short(v) for k, v in list(value.items())[:_MAX_ARGS]}
    if isinstance(value, (list, tuple)):
        return [_short(v) for v in list(value)[:_MAX_ARGS]]
    return _short(str(value))


def _next_seq(cell_id: str, trace_id: str) -> int:
    """The next sequence number, continued from the file if this writer is new.

    A resumed run appends to the stream its earlier attempt started, so the seq
    has to pick up where the file left off — restarting at 1 would make a reader
    that polls ``?since=`` silently skip everything the resumed half emits.
    """
    if trace_id in _SEQ:
        _SEQ[trace_id] += 1
        return _SEQ[trace_id]
    existing = 0
    try:
        text = cell_store.read_text(cell_id, cell_store.KIND_LOGS,
                                    event_name(trace_id))
        existing = sum(1 for line in (text or "").splitlines() if line.strip())
    except Exception:  # noqa: BLE001 - a stream we cannot read starts a new count
        existing = 0
    _SEQ[trace_id] = existing + 1
    return _SEQ[trace_id]


def emit(cell_id: str, trace_id: Optional[str], kind: str, **fields) -> Optional[dict]:
    """Append one event. Returns it, or None if it could not be written."""
    if not trace_id or not is_safe_id(trace_id):
        return None
    try:
        with _LOCK:
            event = {
                "seq":  _next_seq(cell_id, trace_id),
                "at":   datetime.now().isoformat(timespec="seconds"),
                "kind": kind,
                **{k: _short(v) for k, v in fields.items() if v is not None},
            }
            # Published BEFORE it is stored, and in its own failure domain in
            # BOTH directions: the live view must not go dark because the durable
            # write is having a bad day, and the durable record — which is what
            # anyone reads after the run — must not be lost because the middleware
            # is. plan_progress.publish already swallows its own failures; the
            # guard is here so that stays true of this call site no matter what it
            # grows into. Inside the lock, so subscribers see the same order the
            # seq numbers claim.
            try:
                plan_progress.publish(cell_id, trace_id, event)
            except Exception as exc:  # noqa: BLE001
                print(f"[plan] event not published ({exc})", flush=True)
            _store_event(cell_id, trace_id,
                         json.dumps(event, default=str) + "\n")
        return event
    except Exception as exc:  # a dropped event must never cost the plan
        print(f"[plan] event not written ({exc})", flush=True)
        return None


# ─── the planner's vocabulary ────────────────────────────────────────────────

def run_started(cell_id, trace_id, *, operations, backend, model, step_budget,
                resumed_from_step=None) -> None:
    emit(cell_id, trace_id, "run_started", operations=operations, backend=backend,
         model=model, step_budget=step_budget, resumed_from_step=resumed_from_step,
         text=(f"Planning {operations} operations with {backend}"
               if not resumed_from_step
               else f"Resuming after step {resumed_from_step} — "
                    f"{operations} operations, {backend}"))


def thinking(cell_id, trace_id, text) -> None:
    emit(cell_id, trace_id, "think", text=text)


def operation_started(cell_id, trace_id, operation_id, *, robot=None,
                      tool=None, args=None) -> None:
    emit(cell_id, trace_id, "operation_started", operation_id=operation_id,
         robot=robot, tool=tool, args=args,
         text=f"{operation_id} started"
              + (f" on {robot}" if robot else "")
              + (f" ({tool})" if tool else ""))


def operation_result(cell_id, trace_id, operation_id, *, ok, tool=None,
                     robot=None, metrics=None, error=None) -> None:
    """A weld's outcome from one decision. ``validate*`` tools report separately.

    The kind distinguishes *planned* from *validated* because they answer
    different questions — a trajectory exists, versus the tool actually arrives
    on the seam at the right angle — and a feed that merges them lets a run look
    finished when only half the check ran.
    """
    if ok is False:
        kind, word = "operation_failed", "FAILED"
    elif "validate" in (tool or ""):
        kind, word = "operation_validated", "validated"
    else:
        kind, word = "operation_planned", "planned"
    emit(cell_id, trace_id, kind, operation_id=operation_id, tool=tool,
         robot=robot, metrics=metrics, error=error,
         text=f"{operation_id} {word}" + (f": {error}" if error else ""))


def _changed(before: dict, after: dict) -> dict:
    """The arguments that differ, as ``{key: [before, after]}``."""
    keys = set(before) | set(after)
    return {k: [before.get(k), after.get(k)]
            for k in sorted(keys)
            if before.get(k) != after.get(k)}


# Arguments whose change IS the strategy, as opposed to bookkeeping that changes
# on every call. Listed explicitly: a diff over every field would call a new
# waypoint count a strategy change.
_STRATEGY_ARGS = (
    "approach_offset_m", "approach_velocity_scale", "approach_clearance_m",
    "retreat_offset_m", "retreat_velocity_scale", "retreat_clearance_m",
    "constrain_orientation", "orientation", "approach_vector", "robot_ns",
    "planner", "planning_time_limit_s", "clearance_m", "standoff_m",
    "velocity_scale", "operation_type",
)


def note_attempt(cell_id, trace_id, operation_id, args: dict,
                 *, tool=None, robot=None) -> Optional[dict]:
    """Record an attempt and, if it re-tries a weld differently, say so.

    This is the "system changed strategy" signal, and it is derived from evidence
    rather than from the model's prose: the same operation, attempted again, with
    different planning arguments. The diff travels with the event, so the reader
    sees *what* changed — a standoff shrunk from 0.10 m to 0.07 m — instead of
    being told that something did.
    """
    key = (trace_id, operation_id)
    interesting = {k: args.get(k) for k in _STRATEGY_ARGS if k in (args or {})}
    previous = _LAST_ATTEMPT.get(key)
    _LAST_ATTEMPT[key] = interesting
    if previous is None:
        return None
    delta = _changed(previous, interesting)
    if not delta:
        return None
    summary = ", ".join(f"{k} {v[0]} → {v[1]}" for k, v in delta.items())
    return emit(cell_id, trace_id, "strategy", operation_id=operation_id,
                tool=tool, robot=robot, changed=delta,
                text=f"strategy change on {operation_id}: {summary}")


def paused(cell_id, trace_id, *, reason, step, checkpoint=None) -> None:
    emit(cell_id, trace_id, "paused", reason=reason, step=step,
         checkpoint=checkpoint,
         text=f"PAUSED after step {step} ({reason}) — resumable")


def resumed(cell_id, trace_id, *, step, discarded=0) -> None:
    emit(cell_id, trace_id, "resumed", step=step, discarded=discarded,
         text=f"resumed from step {step}"
              + (f", {discarded} incomplete step(s) discarded" if discarded else ""))


def checkpointed(cell_id, trace_id, *, step, path=None) -> None:
    emit(cell_id, trace_id, "checkpoint", step=step, path=path,
         text=f"checkpoint saved at step {step}")


def aborted(cell_id, trace_id, *, kind, error, resumable=False) -> None:
    emit(cell_id, trace_id, "aborted", error_kind=kind, error=error,
         resumable=resumable,
         text=f"ABORTED ({kind}): {error}"
              + ("  — resume to continue where it stopped" if resumable else ""))


def finished(cell_id, trace_id, *, success, tool_calls, elapsed_s=None,
             error=None, tree_id=None) -> None:
    emit(cell_id, trace_id, "finished", success=success, tool_calls=tool_calls,
         elapsed_s=elapsed_s, error=error, tree_id=tree_id,
         text=("tree assembled" if success else f"no tree: {error or 'unknown'}"))


def forget(trace_id: str) -> None:
    """Drop this run's writer state. Called when a run ends."""
    with _LOCK:
        _SEQ.pop(trace_id, None)
        for key in [k for k in _LAST_ATTEMPT if k[0] == trace_id]:
            _LAST_ATTEMPT.pop(key, None)


# ─── reading ─────────────────────────────────────────────────────────────────

def latest_trace_id(cell_id: str) -> Optional[str]:
    """The most recently written event stream for the cell."""
    try:
        streams = cell_store.list_entries(cell_id, cell_store.KIND_LOGS,
                                          prefix=_FILE_PREFIX, suffix=_SUFFIX)
    except Exception:  # noqa: BLE001
        return None
    if not streams:
        return None
    # Name breaks an mtime tie, because two runs started inside the filesystem's
    # timestamp granularity would otherwise resolve in listing order — i.e. at
    # random. A trace id carries its own timestamp, so the name is the right
    # tie-break.
    streams.sort(key=lambda e: (e.get("mtime_s") or 0, e.get("name") or ""),
                 reverse=True)
    return streams[0]["name"][len(_FILE_PREFIX):-len(_SUFFIX)]


def read(cell_id: str, trace_id: Optional[str] = None, *, since: int = 0,
         limit: int = 200) -> dict:
    """Events after ``since``, oldest first — the live feed's poll.

    ``trace_id=None`` means the newest stream, which is what a dashboard that has
    not been told which run to watch actually wants.
    """
    wanted = trace_id or latest_trace_id(cell_id)
    if not wanted or not is_safe_id(wanted):
        return {"trace_id": None, "events": [], "next_seq": since, "running": False}

    try:
        text = cell_store.read_text(cell_id, cell_store.KIND_LOGS,
                                    event_name(wanted))
    except Exception:  # noqa: BLE001
        text = None
    if text is None:
        return {"trace_id": wanted, "events": [], "next_seq": since, "running": False}

    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("seq", 0) > since:
            events.append(event)

    # Truncation keeps the NEWEST events, because a live feed that falls behind
    # should show the present, not replay a stale window forever.
    dropped = max(0, len(events) - limit)
    if dropped:
        events = events[-limit:]

    terminal = {"finished", "aborted", "paused"}
    running = not any(e.get("kind") in terminal
                      for e in _tail_kinds(cell_id, wanted, text=text))
    return {
        "trace_id": wanted,
        "events":   events,
        "next_seq": events[-1]["seq"] if events else since,
        "dropped":  dropped,
        "running":  running,
    }


#: Share of the bar the per-operation planning phase owns; assembly and storing
#: the tree are the rest. Planning is where the minutes go.
PLANNING_SHARE = 90.0
_ASSEMBLY_TOOLS = ("assemble_behaviour_tree", "assemble_coordinated_tree")
_SOLVED_KINDS = ("operation_planned", "operation_validated")


def _covers(event_op: str, program_op: str) -> bool:
    # The planner renames as it goes — W03 comes back as W03_fix, W10_final,
    # W01_via, or with the arm in front as R2_W13 — and every one of those still
    # solves W03/W10/W01/W13. Matched on whole "_" tokens, so W1 is not W10_final.
    return event_op == program_op or program_op in event_op.split("_")


def progress(cell_id: str, trace_id: str,
             operation_ids: Optional[list] = None,
             text: Optional[str] = None) -> dict:
    """How far a run is, as a percentage of its program, read off its events.

    ``percent`` is ``PLANNING_SHARE`` x (program operations with a solved
    trajectory / program operations) while planning, 95 once the tree is being
    assembled and 100 when the run finished with a tree. It only moves forward:
    an operation counts as solved from its first planned/validated event, and a
    later re-plan of it does not take it back.

    ``operation_ids`` is the program's own list. Without it the total comes from
    ``run_started`` and solved ids are counted as they appear, which over-counts
    via points — ``approximate`` says so.

    An estimate of WORK, not of time: an LLM-driven run spends very different
    minutes on different welds, so no ETA is derived from it.
    """
    if text is None:
        try:
            text = cell_store.read_text(cell_id, cell_store.KIND_LOGS,
                                        event_name(trace_id))
        except Exception:  # noqa: BLE001
            text = None
    events: list[dict] = []
    for line in (text or "").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue

    ids = [str(o) for o in (operation_ids or []) if o]
    total = len(ids)
    for e in events:
        if e.get("kind") == "run_started" and not total:
            total = int(e.get("operations") or 0)

    solved_ids = {str(e.get("operation_id")) for e in events
                  if e.get("kind") in _SOLVED_KINDS and e.get("operation_id")}
    if ids:
        solved = sum(1 for p in ids if any(_covers(s, p) for s in solved_ids))
    else:
        solved = min(len(solved_ids), total)

    kinds = [e.get("kind") for e in events]
    last = events[-1] if events else {}
    finished = next((e for e in reversed(events) if e.get("kind") == "finished"), None)
    if not events:
        phase = "queued"
    elif finished is not None:
        phase = "done" if finished.get("success") else "failed"
    elif last.get("kind") == "aborted":
        phase = "aborted"
    elif last.get("kind") == "paused":
        phase = "paused"
    elif any(e.get("tool") in _ASSEMBLY_TOOLS for e in events):
        phase = "assembling"
    else:
        phase = "planning"

    percent = PLANNING_SHARE * solved / total if total else 0.0
    if phase == "assembling":
        percent = max(percent, 95.0)
    elif phase == "done":
        percent = 100.0

    return {
        "trace_id": trace_id,
        "phase": phase,
        "percent": round(percent, 1),
        "operations_solved": solved,
        "operations_total": total,
        "attempts": kinds.count("operation_started"),
        "failed_attempts": kinds.count("operation_failed"),
        "last_event": last.get("text"),
        "last_event_at": last.get("at"),
        "approximate": not ids,
    }


def _tail_kinds(cell_id: str, trace_id: str, window: int = 3,
                text: Optional[str] = None) -> list[dict]:
    """The last few events, for deciding whether the run is still going.

    ``text`` lets a caller that already holds the stream pass it in — reading it
    twice per poll was free on a local file and is a second round trip when the
    stream lives with its owner.
    """
    if text is None:
        try:
            text = cell_store.read_text(cell_id, cell_store.KIND_LOGS,
                                        event_name(trace_id))
        except Exception:  # noqa: BLE001
            return []
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()][-window:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
