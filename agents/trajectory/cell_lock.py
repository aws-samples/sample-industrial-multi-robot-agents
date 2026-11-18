"""One planning run per cell at a time.

Every planning cache (``trajectory_cache``: trajectories, chained configs, weld
poses, home refusals, operation specs) is keyed by CELL, and a fresh run begins by
clearing that cell's entries. Two runs on one cell at once — two submissions, a
plan and a plan edit, a resume during a plan — therefore wipe and overwrite each
other mid-flight. So whatever plans on a cell holds that cell's lock for its whole
run, taken atomically; a second caller is refused with the holder's name rather
than queued (a queued run would start from whatever the first one left).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Optional

_LOCK = threading.Lock()
_HOLDERS: dict = {}          # cell_id -> {"holder", "since_s"}


class CellBusy(RuntimeError):
    """The cell is already being planned by ``holder``."""

    def __init__(self, cell_id: str, holder: dict):
        self.cell_id, self.holder = cell_id, dict(holder)
        super().__init__(f"cell '{cell_id}' is already being planned by "
                         f"{holder.get('holder')} (since {holder.get('since_s')})")


def acquire(cell_id: str, holder: str) -> bool:
    """Take the cell for ``holder``; False when another holder has it."""
    with _LOCK:
        if str(cell_id) in _HOLDERS:
            return False
        _HOLDERS[str(cell_id)] = {"holder": str(holder), "since_s": round(time.time(), 1)}
        return True


def release(cell_id: str, holder: str) -> None:
    with _LOCK:
        if (_HOLDERS.get(str(cell_id)) or {}).get("holder") == str(holder):
            del _HOLDERS[str(cell_id)]


def holder_of(cell_id: str) -> Optional[dict]:
    with _LOCK:
        hit = _HOLDERS.get(str(cell_id))
        return dict(hit) if hit else None


@contextmanager
def held(cell_id: str, holder: str):
    """``with held(cell, who):`` — the cell for the block, or :class:`CellBusy`."""
    if not acquire(cell_id, holder):
        raise CellBusy(cell_id, holder_of(cell_id) or {})
    try:
        yield
    finally:
        release(cell_id, holder)
