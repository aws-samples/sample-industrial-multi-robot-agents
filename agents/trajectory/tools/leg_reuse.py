"""Reuse a planned STEP across runs (the leg store, ``experience_store``).

``dispatch_plan`` asks :func:`lookup` before solving a step and :func:`remember`
after a step planned cleanly. A step is reused only when EVERYTHING it depends on
is identical:

* the cell's config content (``plan_memo.config_fingerprint``) and the backend,
  its options and the planner's environment knobs (``OMPL_*``, ``ROBOBALLET_*``);
* the motion requests as dispatched (after the re-plan start pin and the tool
  spin), the call's default standoff and transit speed;
* where EVERY arm of the cell stands and what seam it stands on
  (``trajectory_cache.cell_chain_snapshot``) — the moving arms chain from it, the
  others are the step's bystanders;
* the planner CODE (:func:`code_fingerprint`): every module under
  ``agents/trajectory`` and the backend entry point that would run — a code
  change never gets yesterday's plan back;
* :data:`VERSION`, bumped when planner output changes meaning.

A hit is not trusted: :func:`verified` puts every stored leg through the
platform's leg check with the cell's static obstacles (``leg_verdict.refusal``,
``static=True``), and ``dispatch_plan`` then runs its usual contract and joint
limit refusals and ``run_trajectory_planner`` its step gate, exactly as for a
fresh plan. A leg that fails is forgotten and the step is solved again.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Optional

from agents.trajectory.tools import experience_store as es

VERSION = 1

_TREE_FP: list = []          # the planner tree's source hash, once per process


def code_fingerprint(entry=None) -> str:
    """A hash of the planner's source (``agents/trajectory/**.py``, once per
    process) and of the backend entry point that would plan (its file, its first
    line, its name — so two different backends never share a key)."""
    from pathlib import Path
    if not _TREE_FP:
        root = Path(__file__).resolve().parents[1]           # agents/trajectory
        h = hashlib.sha256()
        for p in sorted(root.rglob("*.py")):
            try:
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
            except OSError:
                h.update(b"<unreadable>")
        _TREE_FP.append(h.hexdigest())
    fn = getattr(entry, "func", entry)
    code = getattr(fn, "__code__", None)
    ident = ("none" if entry is None else
             f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}:"
             f"{getattr(code, 'co_filename', '?')}:{getattr(code, 'co_firstlineno', '?')}")
    return hashlib.sha256((_TREE_FP[0] + "|" + ident).encode()).hexdigest()


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except Exception:  # noqa: BLE001
        pass
    return repr(o)


def step_key(cell_id: str, backend: str, motion_requests: list, approach_offset_m,
             velocity_scale, options: Optional[dict], backend_kwargs: Optional[dict],
             code: Optional[str] = None) -> Optional[str]:
    """The key of this step, or None (do not reuse) when the store is off or the
    cell's config cannot be fingerprinted."""
    if not es.enabled(es.LEG_STORE):
        return None
    from agents.trajectory.tools import plan_memo
    from agents.trajectory.tools.trajectory_cache import cell_chain_snapshot
    fp = plan_memo.config_fingerprint(cell_id)
    if fp is None:
        return None
    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith(("OMPL_", "ROBOBALLET_"))}
    payload = [VERSION, code or code_fingerprint(), str(cell_id), fp, backend,
               motion_requests, approach_offset_m,
               velocity_scale, options or {}, backend_kwargs or {},
               cell_chain_snapshot(cell_id), env]
    try:
        blob = json.dumps(payload, sort_keys=True, default=_jsonable)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(blob.encode()).hexdigest()


def lookup(cell_id: str, key: Optional[str]) -> Optional[dict]:
    """The newest stored entry for ``key`` (a deep copy), or None."""
    if key is None:
        return None
    for e in reversed(es.entries(cell_id, es.LEG_STORE)):
        if e.get("key") == key and e.get("version") == VERSION:
            return copy.deepcopy(e)
    return None


def verified(cell_id: str, entry: dict) -> bool:
    """Every stored leg passes the platform's leg check, static obstacles included."""
    from agents.trajectory.tools.leg_verdict import refusal
    trajs = ((entry.get("result") or {}).get("trajectories")) or {}
    if not trajs:
        return False
    try:
        return all(refusal(cell_id, ns, leg, static=True) is None
                   for ns, leg in trajs.items())
    except Exception:  # noqa: BLE001 — a check that broke has not said yes
        return False


def forget(cell_id: str, key: str) -> None:
    es.forget(cell_id, es.LEG_STORE, lambda e: e.get("key") == key)


def remember(cell_id: str, key: Optional[str], motion_requests: list, result: dict,
             after: dict) -> bool:
    """Store a step that planned CLEANLY: every request has a trajectory and
    nothing failed. ``after`` is each moving arm's ``chain_state`` as the backend
    committed it."""
    if key is None or not isinstance(result, dict):
        return False
    trajs = result.get("trajectories") or {}
    if (not result.get("success") or result.get("failures")
            or any(r.get("robot_ns") not in trajs for r in motion_requests)):
        return False
    keep = {k: v for k, v in result.items() if k not in ("planning_time_s",)}
    entry = {"version": VERSION, "key": key, "requests": motion_requests,
             "result": keep, "after": after,
             "planning_time_s": result.get("planning_time_s")}
    try:
        entry = json.loads(json.dumps(entry, default=_jsonable))
    except (TypeError, ValueError):
        return False
    return es.append(cell_id, es.LEG_STORE, entry)
