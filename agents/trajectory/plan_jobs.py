"""A planning run's STATUS on the planner's own disk — one file per run.

``get_plan_status`` used to answer from memory only, so a planner that restarted
(an expired token, a recreated container) answered ``unknown`` for every run it
had been planning — even a run with a checkpoint that ``resume_planning`` could
continue. Each run's status now also lives in its own file, named after the run,
beside its checkpoint under the planner's scratch root::

    <scratch>/<cell>/agentic_planner/logs/planjob_<tree_id>.json

Runs are independent files, so any number of cells plan in parallel without
sharing one, and one run is found, reported and resumed by its ``tree_id``.

A file that still says ``planning`` (or ``pausing`` / ``cancelling``) when no live
job in this process has that id was left by a process that died mid-run: it is
reported as ``interrupted``, with ``resumable`` and ``resume_with`` when its
checkpoint is there.

Unfinished-work bookkeeping, like the checkpoint itself: it stays on the
planner's scratch, not in the cell's data. Never raises into the planner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agents.trajectory import plan_checkpoint
from utils import cell_layout

_FILE_PREFIX = "planjob_"
LIVE = ("planning", "pausing", "cancelling")


def path_for(cell_id: str, tree_id: str) -> Path:
    return plan_checkpoint._logs_dir(cell_id) / f"{_FILE_PREFIX}{tree_id}.json"


def save(tree_id: str, record: dict) -> Optional[str]:
    """Write one run's status atomically (best effort: ``None`` when it could not)."""
    cell_id = record.get("cell_id")
    if not cell_id or not plan_checkpoint.is_safe_id(str(tree_id)):
        return None
    try:
        from utils.cell_store import temp_path
        path = path_for(cell_id, str(tree_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path(path)
        try:
            tmp.write_text(json.dumps({"tree_id": tree_id, **record}, default=str))
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001 — a status file never breaks a run
        print(f"[plan] status of {tree_id} not saved ({exc})", flush=True)
        return None


def load(tree_id: str, cell_id: Optional[str] = None) -> Optional[dict]:
    """The saved status of one run, found by its id (the cell is looked up)."""
    if not plan_checkpoint.is_safe_id(str(tree_id)):
        return None
    cells = [cell_id] if cell_id else _cells()
    for cell in cells:
        path = path_for(cell, str(tree_id))
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return None


def _cells() -> list[str]:
    try:
        return sorted(p.name for p in cell_layout.scratch_root().iterdir() if p.is_dir())
    except OSError:
        return []


def as_seen_after_restart(record: dict) -> dict:
    """A saved status, as it stands once the process that wrote it is gone."""
    out = dict(record)
    if out.get("status") in LIVE:
        tree_id, cell_id = out.get("tree_id"), out.get("cell_id")
        saved = (plan_checkpoint.describe(cell_id, tree_id) or {}) if cell_id else {}
        resumable = bool(saved.get("resumable"))
        out.update(status="interrupted", success=False, resumable=resumable,
                   resume_with=tree_id if resumable else None,
                   checkpoint_step=saved.get("step"),
                   note=("The planner stopped while this run was in flight (restart or "
                         "crash). " + (f"resume_planning('{tree_id}') continues it from "
                                       f"step {saved.get('step')}." if resumable else
                                       "It left no checkpoint; plan it again.")))
    return out


def listing(cell_id: str, limit: int = 50) -> list[dict]:
    """Every saved run status of a cell, newest first."""
    directory = plan_checkpoint._logs_dir(cell_id)
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob(f"{_FILE_PREFIX}*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in paths[:max(1, limit)]:
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out
