"""What the planner has already solved, kept across runs — two stores, one owner.

* **Leg store** (``leg_store.jsonl``): a whole planned STEP — every arm's
  trajectory plus what the backend committed for each arm — keyed on everything
  the plan depends on (``planner_backend.dispatch_plan`` builds the key). The same
  step asked again on an unchanged cell is answered from here instead of being
  solved again. Re-planning welding_cell_2_small's task re-solves 57 planner calls.
* **Path experience** (``path_experience.jsonl``): joint-space paths the sampling
  search found (``ompl_compound._solve_compound``), retrieved by similarity for a
  NEW problem of the same arms and offered to the search as candidates — the
  retrieve-and-repair idea of OMPL's Lightning/Thunder planners: a stored path is
  reconnected to the new start and goal and kept only if the current validity
  checker passes every segment of it.

Neither store is trusted. A stored step is re-checked by the platform's leg
checks before it is used, and a stored path is only a candidate the current
problem's checker must pass. A store is advice to skip work, never a reason to
ship a motion nothing has checked.

Persistence goes through ``utils.cell_store`` (KIND_RECORDINGS — the trajectory
knowledge base's kind), as append-only JSON lines, loaded once per process per
cell and compacted when they outgrow :data:`MAX_ENTRIES`. A torn or foreign line
is skipped. Both stores are OFF unless enabled: ``PLAN_LEG_STORE=1``,
``PLAN_EXPERIENCE=1`` (the stack's compose file sets them).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional

LEG_STORE = "leg_store.jsonl"
PATH_EXPERIENCE = "path_experience.jsonl"
#: The allocation memo's on-disk copy (tools/plan_memo.py), so a restart keeps it.
MRTA_MEMO = "mrta_memo.jsonl"

#: Entries kept per store per cell; older ones are compacted away.
MAX_ENTRIES = 2000

_FLAGS = {LEG_STORE: "PLAN_LEG_STORE", PATH_EXPERIENCE: "PLAN_EXPERIENCE",
          MRTA_MEMO: "PLAN_MEMO_PROCESS"}

_lock = threading.Lock()
_loaded: dict = {}          # (cell_id, name) -> list[dict] (oldest first)


def enabled(name: str) -> bool:
    flag = _FLAGS.get(name)
    return bool(flag) and os.environ.get(flag, "0").strip().lower() in (
        "1", "true", "yes", "on")


def _read(cell_id: str, name: str) -> list:
    from utils import cell_store
    text = cell_store.read_text(cell_id, cell_store.KIND_RECORDINGS, name) or ""
    out = []
    for line in text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def entries(cell_id: str, name: str) -> list:
    """Every stored entry for ``cell_id`` (oldest first), or [] when disabled or
    unreadable. The list is shared: do not mutate it."""
    if not enabled(name) or not cell_id:
        return []
    key = (str(cell_id), name)
    with _lock:
        if key in _loaded:
            return _loaded[key]
    try:
        got = _read(cell_id, name)
    except Exception:  # noqa: BLE001 — an unreadable store is an empty one
        got = []
    with _lock:
        _loaded.setdefault(key, got[-MAX_ENTRIES:])
        return _loaded[key]


def append(cell_id: str, name: str, entry: dict) -> bool:
    """Add ``entry`` (in memory and on disk). False when disabled or not written."""
    if not enabled(name) or not cell_id or not isinstance(entry, dict):
        return False
    try:
        line = json.dumps(entry, separators=(",", ":"))
    except (TypeError, ValueError):
        return False
    lst = entries(cell_id, name)
    from utils import cell_store
    with _lock:
        lst.append(entry)
        compact = len(lst) > MAX_ENTRIES * 1.5
        if compact:
            del lst[:len(lst) - MAX_ENTRIES]
            snapshot = list(lst)
    try:
        if compact:
            cell_store.write_text(cell_id, cell_store.KIND_RECORDINGS, name,
                                  "".join(json.dumps(e, separators=(",", ":")) + "\n"
                                          for e in snapshot))
        else:
            cell_store.append_text(cell_id, cell_store.KIND_RECORDINGS, name, line + "\n")
        return True
    except Exception:  # noqa: BLE001 — a store that cannot write costs reuse, not safety
        return False


def forget(cell_id: str, name: str, predicate) -> int:
    """Drop in-memory entries matching ``predicate`` (e.g. one that failed its
    re-check); they are left out of the file at the next compaction."""
    lst = entries(cell_id, name)
    with _lock:
        keep = [e for e in lst if not predicate(e)]
        n = len(lst) - len(keep)
        lst[:] = keep
    return n


def clear(cell_id: Optional[str] = None) -> None:
    """Forget what was loaded (tests; a cell reset). The files are untouched."""
    with _lock:
        for k in [k for k in _loaded if cell_id is None or k[0] == str(cell_id)]:
            del _loaded[k]
