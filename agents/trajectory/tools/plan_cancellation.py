"""Cooperative cancellation for an in-flight planning run.

A planning run is minutes of LLM reasoning wrapped around seconds-to-minutes of
OMPL solving, and it is launched with ``asyncio.to_thread`` — a worker thread
CPython cannot kill from outside. So the operator's Stop button cannot be a
kill: the run has to notice. The contract here is deliberately small:

  * ``begin(tree_id)`` — the run announces itself when it starts,
  * ``request(tree_id)`` — Stop marks it cancelled and returns immediately,
  * ``raise_if_cancelled()`` — the run's own checkpoints unwind it,
  * ``end(tree_id)`` — the run deregisters, cancelled or not.

Where the checkpoints go decides how long Stop takes to bite. They sit at the
two natural boundaries: the entry to ``run_trajectory_planner`` (one LLM step)
and the top of the OMPL per-motion-request loop, so the worst case is one
motion request's solve, not one whole step, and never the whole run.

WHY ``PlanCancelled`` IS A ``BaseException``
--------------------------------------------
This is load-bearing, not style. LangGraph's ``ToolNode`` catches ``Exception``
from a tool and hands the text back to the model as a tool result. A cancel
raised as an ``Exception`` would therefore be *narrated to the LLM*, which would
read "cancelled", reason about it, and keep planning — a Stop button that makes
the agent think instead of stopping. ``BaseException`` is not caught there, nor
by the ``except Exception`` guards inside the planning tools, so it unwinds
through the agent loop to ``_run_plan_job``.

Cancellation is a *safety-side* failure mode: it never leaves a half-written
plan behind, because the plan is only stored on a successful return.
"""
from __future__ import annotations

import threading

__all__ = ["PlanCancelled", "begin", "end", "request", "active",
           "is_cancelled", "raise_if_cancelled"]


class PlanCancelled(BaseException):
    """Raised inside a planning run that an operator has stopped.

    Derives from ``BaseException`` on purpose — see this module's docstring.
    """

    def __init__(self, tree_id: str, reason: str = "cancelled by operator"):
        self.tree_id = tree_id
        self.reason = reason
        super().__init__(f"planning run {tree_id} cancelled: {reason}")


_LOCK = threading.Lock()
# tree_id -> reason if cancelled, else None. Presence means "running".
_RUNS: dict[str, str | None] = {}
# The run bound to the calling thread, so a checkpoint deep inside the planner
# needs no plumbing to know which run it belongs to. Planning runs one job per
# worker thread; threading.local is exactly that scope.
_CURRENT = threading.local()


def begin(tree_id: str) -> None:
    """Register ``tree_id`` as running on this thread and clear any stale flag."""
    with _LOCK:
        _RUNS[tree_id] = None
    _CURRENT.tree_id = tree_id


def end(tree_id: str) -> None:
    """Deregister ``tree_id``. Safe to call twice, and on the cancel path."""
    with _LOCK:
        _RUNS.pop(tree_id, None)
    if getattr(_CURRENT, "tree_id", None) == tree_id:
        _CURRENT.tree_id = None


def active() -> list[str]:
    """The tree_ids currently registered as planning."""
    with _LOCK:
        return sorted(_RUNS)


def request(tree_id: str | None = None,
            reason: str = "cancelled by operator") -> list[str]:
    """Ask a run (or every run) to stop; return the tree_ids marked.

    Returns ``[]`` when there is nothing planning — Stop is idempotent and must
    not error just because the operator pressed it twice.
    """
    with _LOCK:
        targets = [tree_id] if tree_id is not None else list(_RUNS)
        marked = []
        for tid in targets:
            if tid in _RUNS:
                _RUNS[tid] = reason
                marked.append(tid)
    return marked


def is_cancelled(tree_id: str | None = None) -> str | None:
    """The cancel reason for ``tree_id`` (default: this thread's run), or None."""
    tid = tree_id if tree_id is not None else getattr(_CURRENT, "tree_id", None)
    if tid is None:
        return None
    with _LOCK:
        return _RUNS.get(tid)


def raise_if_cancelled(where: str = "") -> None:
    """Checkpoint: unwind the run if it has been cancelled, else do nothing.

    ``where`` names the checkpoint in the log line the job records, so an
    operator can see how far the plan got before it stopped.
    """
    tid = getattr(_CURRENT, "tree_id", None)
    if tid is None:
        return
    with _LOCK:
        reason = _RUNS.get(tid)
    if reason:
        raise PlanCancelled(tid, f"{reason}{f' (at {where})' if where else ''}")
