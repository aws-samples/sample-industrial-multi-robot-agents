"""The one place that answers "what is running on this cell, and how did it go?".

Every A2A call that makes a robot move is a **run**: it is dispatched, it takes
seconds to minutes, and somebody polls it. Three separate needs want a single
answer about runs, and each would otherwise grow its own bookkeeping:

  * ``GET /a2a/result/<run_id>`` — is it done, and what happened?
  * ``GET /a2a/runs``           — what has this orchestrator run?
  * the concurrency guard       — a second motion command while one is already
    executing would drive one cell from two schedulers at once, each unaware of
    the other's gates. That is a *safety* failure, not an availability one, so a
    busy cell is refused rather than interleaved (see ``CLAUDE.md`` → fail-closed,
    and the orchestrator's own "one program at a time per cell" principle).

The registry is deliberately **in-process and bounded**: it is a live view, not an
archive. The archive is the run log on disk
(:mod:`agents.orchestrator.run_logger`), which is why ``GET /a2a/result`` falls
back to it — a run started before a restart is still answerable, just without the
live ``running`` state.

Run ids are not minted here. :mod:`agents.orchestrator.run_logger` owns that
format, because the id has to locate the run's own log on disk; a second
implementation of it is how the two would drift apart.
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable, Optional

from agents.orchestrator.run_logger import new_run_id

# How many finished runs to remember. A live view needs enough history for a
# dashboard to scroll, not the whole life of the process — the on-disk logs are
# the archive. Running runs are never evicted, whatever this number is.
MAX_REMEMBERED = 200

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

# What a run was asked to do. Kept as an explicit set so an unknown kind is a
# programming error caught at dispatch rather than a mystery string in the API.
# "plan" computes and commands nothing; it still takes the slot, because two
# agents planning one cell at once is the state this registry exists to prevent.
# "reset" does move — it stops the activity and drives the arms home.
KINDS = ("task", "replay", "move", "plan", "reset")


class CellBusy(RuntimeError):
    """A motion command arrived while the cell already had one running.

    Carries the run that holds the cell so the caller can report *which* run is
    in the way — "busy" without naming the holder gives an operator nothing to
    act on.
    """

    def __init__(self, cell_id: str, active: dict):
        self.cell_id = cell_id
        self.active = active
        super().__init__(
            f"cell '{cell_id}' is already running {active.get('kind')} "
            f"run {active.get('run_id')} (started {active.get('started_at')})"
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class RunRegistry:
    """Registry of runs for this process. Thread-safe; Flask serves concurrently.

    Instantiate directly in tests. Production shares the module-level singleton
    through the module functions below, so "what is running" has one answer per
    process rather than one per importer.
    """

    def __init__(self, max_remembered: int = MAX_REMEMBERED):
        self._lock = threading.RLock()
        self._runs: dict[str, dict] = {}          # run_id -> envelope
        self._order: list[str] = []               # insertion order, oldest first
        self._max = max_remembered

    # ── dispatch ─────────────────────────────────────────────────────────────

    def start(self, cell_id: str, kind: str, program: Optional[str] = None,
              run_id: Optional[str] = None) -> dict:
        """Claim the cell for a new run and return its envelope.

        Raises :class:`CellBusy` if another run already holds this cell, and
        :class:`ValueError` for an unknown ``kind``.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown run kind {kind!r} — expected one of {KINDS}")
        with self._lock:
            active = self.active(cell_id)
            if active is not None:
                raise CellBusy(cell_id, active)
            rid = run_id or new_run_id(cell_id)
            # A second run inside the same second would otherwise reuse the id and
            # overwrite the first one's entry (and its log). Widen rather than
            # collide: the id must stay unique to stay poll-able.
            if rid in self._runs:
                rid = f"{rid}_{len(self._order)}"
            envelope = {
                "run_id":      rid,
                "cell_id":     cell_id,
                "kind":        kind,
                "program":     program,
                "status":      RUNNING,
                "done":        False,
                "started_at":  _now(),
                "finished_at": None,
                "elapsed_s":   None,
                "result":      None,
                "error":       None,
            }
            self._runs[rid] = envelope
            self._order.append(rid)
            self._evict()
            return dict(envelope)

    def finish(self, run_id: str, result: dict) -> Optional[dict]:
        """Record a completed run. Success is read from the result, not assumed.

        The tools this wraps all report failure in-band as ``{"success": False,
        ...}`` with HTTP 200, so treating "it returned" as "it worked" would mark
        every failed execution as succeeded. A result with no ``success`` key at
        all is treated as a failure for the same reason: an unrecognised shape is
        not evidence that the robot did what was asked.
        """
        ok = bool(isinstance(result, dict) and result.get("success") is True)
        error = None
        if not ok and isinstance(result, dict):
            error = result.get("error") or result.get("failure") or None
        return self._close(run_id, SUCCEEDED if ok else FAILED, result, error)

    def fail(self, run_id: str, error: str) -> Optional[dict]:
        """Record a run that raised instead of returning a result."""
        return self._close(run_id, FAILED, None, str(error))

    #: Fields a running run may report about itself. The lifecycle fields
    #: (status, done, result, ...) are owned by start/finish/fail and are not
    #: writable here, so a progress report can never close or reopen a run.
    PROGRESS_FIELDS = ("progress", "tree_id")

    def note_progress(self, run_id: str, **fields) -> Optional[dict]:
        """Attach live progress to a RUNNING run; ignored once it has ended."""
        with self._lock:
            envelope = self._runs.get(run_id)
            if envelope is None or envelope["status"] != RUNNING:
                return None
            for key, value in fields.items():
                if key in self.PROGRESS_FIELDS:
                    envelope[key] = value
            return dict(envelope)

    def _close(self, run_id: str, status: str, result, error) -> Optional[dict]:
        with self._lock:
            envelope = self._runs.get(run_id)
            if envelope is None:
                return None
            envelope["status"] = status
            envelope["done"] = True
            envelope["finished_at"] = _now()
            envelope["result"] = result
            envelope["error"] = error
            started = envelope.get("started_at")
            if isinstance(result, dict) and result.get("elapsed_s") is not None:
                # Prefer the elapsed time the run itself measured — it excludes
                # the dispatch and bookkeeping this module adds.
                envelope["elapsed_s"] = result["elapsed_s"]
            elif started:
                envelope["elapsed_s"] = round(
                    (datetime.fromisoformat(envelope["finished_at"])
                     - datetime.fromisoformat(started)).total_seconds(), 2)
            self._evict()
            return dict(envelope)

    # ── reads ────────────────────────────────────────────────────────────────

    def get(self, run_id: str) -> Optional[dict]:
        with self._lock:
            envelope = self._runs.get(run_id)
            return dict(envelope) if envelope else None

    def active(self, cell_id: str) -> Optional[dict]:
        """The run currently holding ``cell_id``, if any."""
        with self._lock:
            for rid in reversed(self._order):
                envelope = self._runs.get(rid)
                if (envelope and envelope["cell_id"] == cell_id
                        and envelope["status"] == RUNNING):
                    return dict(envelope)
            return None

    def recent(self, cell_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Runs newest first, optionally for one cell."""
        with self._lock:
            out = []
            for rid in reversed(self._order):
                envelope = self._runs.get(rid)
                if envelope is None:
                    continue
                if cell_id and envelope["cell_id"] != cell_id:
                    continue
                out.append(dict(envelope))
                if len(out) >= limit:
                    break
            return out

    # ── housekeeping ─────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Drop the oldest FINISHED runs past the cap.

        A running run is never evicted: forgetting it would both lose the poller's
        only handle on it and silently release the cell's concurrency guard while
        the robots are still moving.
        """
        with self._lock:
            surplus = len(self._order) - self._max
            if surplus <= 0:
                return
            keep: list[str] = []
            for rid in self._order:
                envelope = self._runs.get(rid)
                if surplus > 0 and envelope is not None and envelope["done"]:
                    del self._runs[rid]
                    surplus -= 1
                    continue
                keep.append(rid)
            self._order = keep


# ─── Process-wide singleton ───────────────────────────────────────────────────

_REGISTRY = RunRegistry()


def registry() -> RunRegistry:
    """The registry this process shares."""
    return _REGISTRY


def start(cell_id: str, kind: str, program: Optional[str] = None,
          run_id: Optional[str] = None) -> dict:
    return _REGISTRY.start(cell_id, kind, program=program, run_id=run_id)


def finish(run_id: str, result: dict) -> Optional[dict]:
    return _REGISTRY.finish(run_id, result)


def fail(run_id: str, error: str) -> Optional[dict]:
    return _REGISTRY.fail(run_id, error)


def note_progress(run_id: str, **fields) -> Optional[dict]:
    return _REGISTRY.note_progress(run_id, **fields)


def get(run_id: str) -> Optional[dict]:
    return _REGISTRY.get(run_id)


def active(cell_id: str) -> Optional[dict]:
    return _REGISTRY.active(cell_id)


def recent(cell_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    return _REGISTRY.recent(cell_id, limit)


def run_in_background(envelope: dict, work: Callable[[], dict],
                      reg: Optional[RunRegistry] = None) -> threading.Thread:
    """Run ``work`` on a daemon thread, closing the run whatever happens.

    The ``finally``-free shape is deliberate: an exception must be recorded as a
    FAILED run, not merely logged. A run left ``running`` forever would hold the
    cell's concurrency guard and block every later command — an availability
    failure caused by bookkeeping, which is exactly what the guard exists to
    avoid being.
    """
    reg = reg or _REGISTRY
    run_id = envelope["run_id"]

    def _body() -> None:
        try:
            reg.finish(run_id, work())
        except Exception as exc:                       # noqa: BLE001
            reg.fail(run_id, str(exc))

    thread = threading.Thread(target=_body, daemon=True,
                              name=f"a2a-run-{run_id}")
    thread.start()
    return thread
