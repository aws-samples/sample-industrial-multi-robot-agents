"""Stop a planning run where it is, and pick it up there later.

A 15-weld plan is fifteen minutes of LLM reasoning wrapped around a hundred OMPL
solves. Twice in one afternoon that work was thrown away whole — once because the
AWS token expired at step 18, once because a Bedrock socket timed out at 2025 s —
and each time the only recovery was to start again from W01. Nothing had gone
wrong with the welds; the run simply lost its place.

A run's place is two things, and only two:

  * **the conversation** — the message history, which is what the model needs to
    know what it already decided, and
  * **the computed work** — the cached joint trajectories and the per-robot
    chaining state (:func:`agents.trajectory.tools.trajectory_cache.snapshot`),
    which is what makes resuming cheap instead of merely possible.

Both are written to one JSON file under the planner's own scratch root
(:func:`utils.cell_layout.scratch_root`), so the resume survives the *process* that
was planning — which is exactly the case that matters when the failure was "the
token expired, recreate the container".

This is the one thing the planner still keeps on its own disk, and it is here on
purpose. A checkpoint is an UNFINISHED run: a conversation nobody else can
continue and trajectories nobody else may execute, because they have not been
assembled into a validated plan yet. Everything a cell keeps — the plan, its
trace, its events, its obstacle model — goes to the agent that owns the cell's
data (:mod:`utils.cell_store`). Unfinished work stays with the worker; finished
work belongs to the cell. With ``PLANNER_SCRATCH_PATH`` unset the root is the
ordinary cells root, so a test or the orchestrator reads exactly what it always
did.

Where a checkpoint is safe to take
----------------------------------
At a **state boundary** of the ReAct loop — after a step's tool results have come
back — never mid-step. That is a boundary the conversation is valid at: a history
ending in an ``AIMessage`` whose ``tool_calls`` were never answered is not a
resumable conversation, it is one Bedrock rejects outright (a ``toolUse`` with no
``toolResult``). :func:`resumable_prefix` enforces that by trimming any trailing
unanswered step, so the worst a resume costs is repeating the one step that was
in flight.

Pausing is therefore cooperative and *quiet*: no exception is thrown through the
planning tools. The run notices the request at its next state boundary and stops
having just written a checkpoint. Cancellation is the opposite of this — see
:mod:`agents.trajectory.tools.plan_cancellation`, which throws, discards, and
leaves nothing to resume. Pause keeps everything; Stop keeps nothing. They are
different buttons because they answer different questions.

Never raises into the planner. A checkpoint that cannot be written is a resume
that will not be available, which is strictly better than a run that dies trying
to save one.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils import cell_layout

SCHEMA = "plan_checkpoint/1"

_FILE_PREFIX = "checkpoint_"

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def is_safe_id(tree_id: str) -> bool:
    return bool(tree_id) and bool(_SAFE_ID.match(tree_id)) and ".." not in tree_id


def path_for(cell_id: str, tree_id: str) -> Path:
    """This run's checkpoint file, under the planner's scratch root."""
    return _logs_dir(cell_id) / f"{_FILE_PREFIX}{tree_id}.json"


def _logs_dir(cell_id: str) -> Path:
    """``<scratch>/<cell>/…/logs`` — the scratch root's ordinary cell layout."""
    return cell_layout.logs_dir(cell_id, cell_base=str(cell_layout.scratch_root()))


# ─── the pause request ───────────────────────────────────────────────────────
# Mirrors plan_cancellation's shape on purpose: an operator-facing verb, a
# process-wide registry, and a check the run itself performs. The difference is
# what happens on notice — pause returns a result, cancel raises.

_LOCK = threading.Lock()
# tree_id -> reason. Presence means "pause has been asked for and not yet taken".
_PAUSE_REQUESTS: dict[str, str] = {}


def request_pause(tree_id: Optional[str], reason: str = "paused by operator") -> dict:
    """Ask a run to stop at its next state boundary. Returns immediately.

    ``tree_id=None`` pauses every in-flight run, which is what an operator asking
    for "pause" without naming one means.
    """
    with _LOCK:
        if tree_id:
            _PAUSE_REQUESTS[str(tree_id)] = reason
            return {"success": True, "paused": [str(tree_id)], "reason": reason}
        from agents.trajectory.tools import plan_cancellation
        targets = [str(t) for t in plan_cancellation.active()]
        for target in targets:
            _PAUSE_REQUESTS[target] = reason
        return {"success": True, "paused": targets, "reason": reason,
                "note": ("no run named — pause requested for every active run"
                         if targets else "no planning run is active")}


def pause_requested(tree_id: Optional[str]) -> Optional[str]:
    """The reason a pause was asked for, or None. Does not clear the request."""
    if not tree_id:
        return None
    with _LOCK:
        return _PAUSE_REQUESTS.get(str(tree_id))


def take_pause(tree_id: Optional[str]) -> Optional[str]:
    """Consume the pause request, so a resumed run does not pause again at once."""
    if not tree_id:
        return None
    with _LOCK:
        return _PAUSE_REQUESTS.pop(str(tree_id), None)


def clear_pause(tree_id: Optional[str]) -> None:
    take_pause(tree_id)


# ─── message serialization ───────────────────────────────────────────────────

def encode_messages(messages: list) -> list:
    """LangChain messages as JSON, keeping tool calls and reasoning blocks intact.

    ``dumpd`` is used rather than a hand-rolled encoder because the parts that
    matter for a resume are the awkward ones: a Bedrock reply carries signed
    ``reasoning_content`` blocks, and the provider rejects a continued
    conversation whose earlier blocks were reshaped.
    """
    from langchain_core.load import dumpd
    return [dumpd(message) for message in messages]


def decode_messages(records: list) -> list:
    """Rebuild messages from :func:`encode_messages`.

    Restricted to chat-message classes. The file lives on a shared volume, and a
    deserializer that will instantiate whatever a file names is a code-execution
    primitive — a checkpoint must only ever be able to become messages.
    """
    from langchain_core.load import load
    out: list = []
    for record in records or []:
        try:
            out.append(load(record, allowed_objects="messages"))
        except TypeError:
            # Older langchain_core has no allowed_objects. Still restricted, just
            # by the library's own valid-namespace list.
            out.append(load(record))
        except Exception as exc:  # one bad message must not void the checkpoint
            print(f"[plan] checkpoint message skipped ({exc})", flush=True)
    return out


def _unanswered_tool_calls(messages: list) -> int:
    """How many trailing tool calls have no result. 0 means a clean boundary."""
    answered = {getattr(m, "tool_call_id", None) for m in messages
                if getattr(m, "tool_call_id", None)}
    missing = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("id") not in answered:
                missing += 1
    return missing


def resumable_prefix(messages: list) -> tuple[list, int]:
    """The longest prefix a provider will accept, and how many messages it dropped.

    A conversation may only be continued from a point where every tool call has
    its result. Anything after the last such point was the step that was in
    flight when the run stopped, and it has to be replayed rather than resumed
    from — one step's cost, not a run's.
    """
    for end in range(len(messages), 0, -1):
        prefix = messages[:end]
        last = prefix[-1]
        # Ending on an AIMessage that asked for tools is only valid if they were
        # all answered — which, at the end of a prefix, they cannot have been.
        if getattr(last, "tool_calls", None):
            continue
        if _unanswered_tool_calls(prefix) == 0:
            return prefix, len(messages) - end
    return [], len(messages)


def _has_completed_step(messages: list) -> bool:
    """Whether the conversation contains work a resume would actually recover.

    A prefix of just the opening prompt is a valid conversation and a *pointless*
    resume: it preserves nothing, so continuing it and replanning from the first
    weld are the same run. Calling that "resumable" is how an operator ends up
    pressing Resume and quietly getting a restart — the surprise this whole
    mechanism exists to remove. One answered tool call is the threshold.
    """
    return any(getattr(m, "tool_call_id", None) for m in messages)


# ─── saving and loading ──────────────────────────────────────────────────────

def save(cell_id: str, tree_id: Optional[str], *, messages: list,
         operations: list, constraints: Optional[dict], step: int,
         backend: Optional[str] = None, model: Optional[str] = None,
         step_budget: Optional[int] = None, reason: str = "",
         resumable: bool = True) -> Optional[str]:
    """Write the run's place. Returns the path, or None if it could not be saved."""
    if not tree_id or not is_safe_id(str(tree_id)):
        return None
    try:
        from agents.trajectory.tools import trajectory_cache

        prefix, discarded = resumable_prefix(messages)
        payload = {
            "schema":      SCHEMA,
            "tree_id":     str(tree_id),
            "cell_id":     cell_id,
            "saved_at":    datetime.now().isoformat(timespec="seconds"),
            "reason":      reason,
            "resumable":   bool(resumable and prefix
                                and _has_completed_step(prefix)),
            "step":        step,
            "step_budget": step_budget,
            "backend":     backend,
            "model":       model,
            "operations":  operations,
            "constraints": constraints,
            # The in-flight step is dropped here, not at resume time, so what the
            # file says is exactly what a resume will do.
            "discarded_messages": discarded,
            "messages":    encode_messages(prefix),
            "cache":       trajectory_cache.snapshot(cell_id),
            # The assembler's full trees, by the ids the model saw them under: the
            # conversation holds only summaries (agent_view), so a run resumed after
            # it had assembled could otherwise not store the tree it accepted.
            "full_trees":  _full_trees(),
        }
        path = path_for(cell_id, str(tree_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        from utils.cell_store import temp_path
        tmp = temp_path(path)                   # this writer's own, never shared
        try:
            with open(tmp, "w") as handle:
                json.dump(payload, handle, default=str)
            tmp.replace(path)   # atomic: a half-written checkpoint is worse than none
        finally:
            tmp.unlink(missing_ok=True)
        return str(path)
    except Exception as exc:
        print(f"[plan] checkpoint not saved ({exc})", flush=True)
        return None


def _full_trees() -> list:
    """The agent's remembered full trees (``agent._FULL_TREES``), JSON-ready."""
    import sys
    agent = sys.modules.get("agents.trajectory.agent")
    return list(getattr(agent, "_FULL_TREES", {}).values()) if agent else []


def _restore_full_trees(trees: list) -> int:
    """Put saved full trees back where the agent looks them up. Returns how many."""
    try:
        from agents.trajectory import agent
    except Exception:  # noqa: BLE001 — no agent here: nothing to restore into
        return 0
    n = 0
    for tree in trees:
        if isinstance(tree, dict) and "root" in tree:
            agent._remember_full_tree({"behaviour_tree": tree})
            n += 1
    return n


def load_checkpoint(cell_id: str, tree_id: str) -> Optional[dict]:
    """The raw checkpoint record, or None."""
    if not is_safe_id(tree_id):
        return None
    try:
        data = json.loads(path_for(cell_id, tree_id).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_cell(tree_id: str) -> Optional[str]:
    """Which cell holds this run's checkpoint, found by looking.

    A resume usually happens in a *new* process — the container was recreated with
    a fresh token, which is the failure being recovered from — so the in-memory job
    registry that knew the cell_id is gone. The file on the shared volume is not,
    so the cell is discoverable rather than something the caller must remember.
    """
    if not is_safe_id(tree_id):
        return None
    try:
        root = cell_layout.scratch_root()
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir():
                continue
            if path_for(candidate.name, tree_id).is_file():
                return candidate.name
    except OSError:
        return None
    return None


def restore(cell_id: str, tree_id: str) -> Optional[dict]:
    """Load a checkpoint AND put its computed work back into the cache.

    Returns ``{messages, operations, constraints, step, cache_restored, ...}``, or
    None if there is nothing resumable. The cache restore happens here, next to
    the message decode, because a resume that recovered the conversation but not
    the trajectories would re-solve everything while telling the model it had
    already been solved — the one failure mode worse than starting over.
    """
    record = load_checkpoint(cell_id, tree_id)
    if not record or not record.get("resumable"):
        return None
    messages = decode_messages(record.get("messages") or [])
    if not messages or not _has_completed_step(messages):
        # The stored flag is a *cache* of a property of the messages, and a file
        # written by an older build can claim resumable for a history that
        # preserves nothing. Recomputed here, where the messages are already
        # decoded, so a stale checkpoint cannot resume into a silent restart.
        return None
    from agents.trajectory.tools import trajectory_cache
    restored = trajectory_cache.restore(cell_id, record.get("cache") or {})
    restored["full_trees"] = _restore_full_trees(record.get("full_trees") or [])
    return {
        "tree_id":        record.get("tree_id"),
        "messages":       messages,
        "operations":     record.get("operations") or [],
        "constraints":    record.get("constraints"),
        "step":           record.get("step") or 0,
        "step_budget":    record.get("step_budget"),
        "backend":        record.get("backend"),
        "model":          record.get("model"),
        "reason":         record.get("reason"),
        "saved_at":       record.get("saved_at"),
        "discarded":      record.get("discarded_messages") or 0,
        "cache_restored": restored,
    }


def discard(cell_id: str, tree_id: str) -> bool:
    """Delete a checkpoint. Called once its run has finished for good."""
    if not is_safe_id(tree_id):
        return False
    try:
        path_for(cell_id, tree_id).unlink()
        return True
    except OSError:
        return False


def describe(cell_id: str, tree_id: str) -> Optional[dict]:
    """A checkpoint's metadata, without decoding the conversation."""
    record = load_checkpoint(cell_id, tree_id)
    if not record:
        return None
    return {
        "tree_id":    record.get("tree_id"),
        "cell_id":    record.get("cell_id"),
        "saved_at":   record.get("saved_at"),
        "reason":     record.get("reason"),
        "resumable":  bool(record.get("resumable")),
        "step":       record.get("step"),
        "messages":   len(record.get("messages") or []),
        "discarded":  record.get("discarded_messages") or 0,
        "operations": len(record.get("operations") or []),
        "trajectories": len(((record.get("cache") or {}).get("trajectories")) or []),
        "backend":    record.get("backend"),
        "model":      record.get("model"),
    }


def listing(cell_id: str, limit: int = 50) -> list[dict]:
    """Every saved checkpoint for the cell, newest first."""
    directory = _logs_dir(cell_id)
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob(f"{_FILE_PREFIX}*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict] = []
    for path in paths[:max(1, limit)]:
        meta = describe(cell_id, path.stem[len(_FILE_PREFIX):])
        if meta:
            out.append(meta)
    return out
