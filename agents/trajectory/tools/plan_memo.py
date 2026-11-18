"""Reuse of expensive, deterministic planning results — one small owner.

Two results are recomputed far more often than their inputs change:

* the MRTA allocation (``run_mrta_pipeline``): measured on welding_cell_2_small
  (bt_1793277501) at 206 s, for the same task on the same cell as the runs
  before it;
* the coordinated return HOME (``behaviour_tree._home_step``): the same arms
  standing at the same joints get the same planned return, yet every assembly
  re-plans it — 539 s of the same run's final assembly.

A memo entry is keyed on everything the result depends on, and on the cell's
configuration CONTENT (:func:`config_fingerprint`): edit robots.yaml or
obstacles.yaml and every entry for that cell misses. When the fingerprint cannot
be computed the memo is bypassed — "unknown" is never treated as "unchanged".

Values are deep-copied in and out, so a caller that mutates a result cannot
change what the next caller gets. ``PLAN_MEMO=0``
turns it off. Two lifetimes:

* process-wide (:func:`get` / :func:`put`, bounded LRU) — for results that are a
  pure function of their key (the allocation), reused across runs; the
  allocation is ALSO kept on disk in the cell's recordings (``mrta_memo.jsonl``,
  via ``experience_store``), so a restarted planner still has it. Opt-in with
  ``PLAN_MEMO_PROCESS=1`` (the stack's compose file sets it): a test that swaps
  a geometry function between two calls must not be handed the first answer;
* one planning run (:func:`run_scope`, :func:`get_run` / :func:`put_run`) — for
  results that also depend on state outside the key (the planned return home
  reads the trajectory cache and the planner backend's own state). Outside a run
  scope nothing is memoised.

Standard library only (plus ``utils.cell_store`` for the fingerprint).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

#: Config artefacts a planning result depends on. A missing file hashes as absent.
FINGERPRINT_FILES = ("robots.yaml", "obstacles.yaml", "operations.yaml")

MAX_ENTRIES = 64

_lock = threading.Lock()
_store: "OrderedDict[tuple, Any]" = OrderedDict()


def enabled() -> bool:
    return os.environ.get("PLAN_MEMO", "1").strip().lower() not in ("0", "false", "no", "off")


def process_enabled() -> bool:
    return enabled() and os.environ.get("PLAN_MEMO_PROCESS", "0").strip().lower() in (
        "1", "true", "yes", "on")


def config_fingerprint(cell_id: str) -> Optional[str]:
    """A hash of the cell's planning config content, or None when it cannot be read."""
    try:
        from utils import cell_store
        h = hashlib.sha256(str(cell_id).encode())
        for name in FINGERPRINT_FILES:
            text = cell_store.read_text(cell_id, cell_store.KIND_CONFIG, name)
            h.update(f"\0{name}\0".encode())
            h.update(b"<absent>" if text is None else text.encode())
        return h.hexdigest()
    except Exception:  # noqa: BLE001 — unknown config: never reuse a result
        return None


def make_key(cell_id: str, *parts: Any) -> Optional[str]:
    """The memo key for ``parts`` on ``cell_id``, or None (bypass) when the memo is
    off or the cell's config cannot be fingerprinted."""
    if not enabled():
        return None
    fp = config_fingerprint(cell_id)
    if fp is None:
        return None
    try:
        blob = json.dumps([cell_id, fp, list(parts)], sort_keys=True, default=_jsonable)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(blob.encode()).hexdigest()


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except Exception:  # noqa: BLE001
        pass
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


#: Namespaces whose entries are also kept ON DISK, in the cell's recordings, so
#: they survive a restart of the planner (experience_store.MRTA_MEMO).
PERSISTED = {"mrta"}


def _disk_get(cell_id: Optional[str], namespace: str, key: str):
    if not cell_id or namespace not in PERSISTED:
        return None
    from agents.trajectory.tools import experience_store as es
    for e in reversed(es.entries(cell_id, es.MRTA_MEMO)):
        if e.get("ns") == namespace and e.get("key") == key:
            return e.get("value")
    return None


def get(namespace: str, key: Optional[str], cell_id: Optional[str] = None):
    """A deep copy of the stored value, or None on a miss (or a None key). With
    ``cell_id``, a persisted namespace also looks on disk (and warms memory)."""
    if key is None or not process_enabled():
        return None
    with _lock:
        k = (namespace, key)
        if k in _store:
            _store.move_to_end(k)
            return copy.deepcopy(_store[k])
    try:
        hit = _disk_get(cell_id, namespace, key)
    except Exception:  # noqa: BLE001 — an unreadable store is a miss
        hit = None
    if hit is None:
        return None
    _remember(namespace, key, hit)
    return copy.deepcopy(hit)


def _remember(namespace: str, key: str, value: Any) -> None:
    with _lock:
        _store[(namespace, key)] = copy.deepcopy(value)
        _store.move_to_end((namespace, key))
        while len(_store) > MAX_ENTRIES:
            _store.popitem(last=False)


def put(namespace: str, key: Optional[str], value: Any,
        cell_id: Optional[str] = None) -> None:
    """Store ``value``; with ``cell_id``, a persisted namespace is written to disk
    too (the key already carries the cell's config fingerprint)."""
    if key is None or not process_enabled():
        return
    _remember(namespace, key, value)
    if cell_id and namespace in PERSISTED:
        try:
            from agents.trajectory.tools import experience_store as es
            es.append(cell_id, es.MRTA_MEMO, {"ns": namespace, "key": key,
                                              "value": json.loads(json.dumps(
                                                  value, default=_jsonable))})
        except Exception:  # noqa: BLE001 — not persisted: costs reuse, not safety
            pass


def clear(namespace: Optional[str] = None) -> None:
    with _lock:
        for k in [k for k in _store if namespace is None or k[0] == namespace]:
            del _store[k]


# ── one planning run ─────────────────────────────────────────────────────────
_RUN: ContextVar[Optional[dict]] = ContextVar("plan_memo_run", default=None)


@contextmanager
def run_scope():
    """A fresh run-scoped memo for one planning run."""
    token = _RUN.set({})
    try:
        yield
    finally:
        _RUN.reset(token)


def get_run(namespace: str, key: Optional[str]):
    store = _RUN.get()
    if store is None or key is None or (namespace, key) not in store:
        return None
    return copy.deepcopy(store[(namespace, key)])


def put_run(namespace: str, key: Optional[str], value: Any) -> None:
    store = _RUN.get()
    if store is None or key is None:
        return
    store[(namespace, key)] = copy.deepcopy(value)
