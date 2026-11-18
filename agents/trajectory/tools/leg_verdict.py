"""One verdict per leg: may this arm execute this trajectory on its own?

A leg is judged by the same checks wherever it is judged — the dispatch facade
after a backend returned it, the step gate after a step was planned, the
assembler before a tree ships, the coordinated assembler, the home reset:

  * ``limits``  — every sample inside the arm's joint box (``joint_limits``);
  * ``self``    — the arm against its OWN hulls along the executed path, densely
                  (``self_collision``);
  * ``static``  — the cell's static obstacles at the runtime margins
                  (``robot_kinematics.leg_static_breach``).

The inter-arm question ("may these arms move at once?") is not here: it needs
every arm's leg, and ``behaviour_tree._gate_multi_arm_step`` owns it.

One policy for a check that does not answer, so no caller invents its own:

  * UNAVAILABLE (no geometry stack, no model for this cell) — the check has no
    opinion and refuses nothing; :func:`unchecked` says which checks were skipped.
    This is reachable only where the legs were not planned by the same stack.
  * ERRORED (the check ran and raised) — refused, ``leg_check_error``. A check
    that broke has not said yes.

Verdicts are cached per (cell, robot, leg content, cell files' mtimes), so the
four gates a leg passes through cost one evaluation; a re-converted cell (new
``robots.yaml`` / ``obstacles.yaml`` / URDF) is judged afresh.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Optional

_LOCK = threading.Lock()
_VERDICTS: "OrderedDict" = OrderedDict()
_MAX = 4096
# What a check that cannot run raises: an availability failure, not a verdict
# (no geometry stack, no file for this cell, no config entry for this robot —
# dispatch refuses an unknown robot loudly on its own).
_UNAVAILABLE = (ImportError, FileNotFoundError, KeyError)


def cell_stamp(cell_id: str, robot_ns: Optional[str] = None) -> Optional[tuple]:
    """The mtimes of the files a leg is judged against (``None`` for one that is
    absent) — part of every cache key that depends on the cell. ``None`` when the
    cell's files are not local (``CELL_DATA_MODE=orchestrator``): an edit there
    cannot be seen, so nothing keyed on the cell is cached."""
    paths = []
    try:
        from utils import cell_layout, cell_store
        if cell_store.is_remote():
            return None        # the files are on another machine: nothing to stamp
        paths = [cell_layout.robots_yaml(cell_id), cell_layout.obstacles_yaml(cell_id)]
        if robot_ns:
            paths.append(cell_layout.robot_urdf(cell_id, robot_ns))
    except Exception:  # noqa: BLE001 — no layout: an empty stamp
        return ()
    out = []
    for p in paths:
        try:
            out.append(p.stat().st_mtime_ns if p is not None else None)
        except OSError:
            out.append(None)
    return tuple(out)


def _digest(leg) -> str:
    h = hashlib.sha1()
    for pt in leg or []:
        q = pt.get("positions") if isinstance(pt, dict) else pt
        t = pt.get("time_s") if isinstance(pt, dict) else None
        ph = pt.get("phase") if isinstance(pt, dict) else None
        # The values as given: a leg is judged on its content, and a malformed one
        # is the check's to refuse, not the cache's to crash on.
        h.update(repr((list(q or []), t, ph)).encode())
    return h.hexdigest()


def _cached(kind: str, cell_id: str, robot_ns: str, leg, compute, judge=None):
    # ``judge`` is the check doing the work: a replaced one (a re-built checker, a
    # test double) is a different judge and gets its own verdicts.
    stamp = cell_stamp(cell_id, robot_ns)
    if stamp is None:
        return compute()
    key = (kind, id(judge), str(cell_id), str(robot_ns), _digest(leg), stamp)
    with _LOCK:
        hit = _VERDICTS.get(key)
        # The entry holds the judge itself: an id is only an address, and a freed
        # judge's address can be handed to a new one (a test double replaced by
        # another), which must not inherit the old one's verdicts.
        if hit is not None and hit[0] is judge:
            _VERDICTS.move_to_end(key)
            return hit[1]
    value = compute()
    with _LOCK:
        _VERDICTS[key] = (judge, value)
        while len(_VERDICTS) > _MAX:
            _VERDICTS.popitem(last=False)
    return value


def _run(check, *args):
    """``("hit", value)`` / ``("unavailable", reason)`` / ``("error", reason)``."""
    try:
        return "hit", check(*args)
    except _UNAVAILABLE as exc:
        return "unavailable", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 — a check that broke has not said yes
        return "error", f"{type(exc).__name__}: {exc}"


def judged(kind: str, cell_id: str, robot_ns: str, leg, check) -> tuple:
    """``check(cell_id, robot_ns, leg)`` under this module's policy and cache:
    ``("hit", result)`` / ``("unavailable", reason)`` / ``("error", reason)``."""
    return _cached(kind, cell_id, robot_ns, leg,
                   lambda: _run(check, cell_id, robot_ns, leg), check)


def limits(cell_id: str, robot_ns: str, leg) -> tuple:
    """Joint box: ``(status, violation | reason)``."""
    try:
        from agents.trajectory.tools.joint_limits import leg_limit_violation
    except _UNAVAILABLE as exc:
        return "unavailable", f"{type(exc).__name__}: {exc}"
    return judged("limits", cell_id, robot_ns, leg, leg_limit_violation)


def self_hit(cell_id: str, robot_ns: str, leg) -> tuple:
    """The arm against itself: ``(status, hit | reason)``."""
    try:
        from agents.trajectory.tools import self_collision
        c = self_collision.checker(cell_id, robot_ns)
    except _UNAVAILABLE as exc:
        return "unavailable", f"{type(exc).__name__}: {exc}"
    if not c.available:
        return "unavailable", c.reason
    # The checker caches its own verdicts per leg; it is the judge here.
    return _run(c.trajectory_violation, leg)


def static_hit(cell_id: str, robot_ns: str, leg) -> tuple:
    """The cell's static obstacles: ``(status, hit | reason)``."""
    try:
        from agents.trajectory.tools import robot_kinematics as rk
    except _UNAVAILABLE as exc:
        return "unavailable", f"{type(exc).__name__}: {exc}"
    return judged("static", cell_id, robot_ns, leg, rk.leg_static_breach)


def _error(robot_ns: str, check: str, why: str) -> dict:
    return {"robot_ns": robot_ns, "reason_code": "leg_check_error", "check": check,
            "error": f"{robot_ns}'s leg could not be checked ({check}): {why}"}


def refusal(cell_id: str, robot_ns: str, leg, *, static: bool = False) -> Optional[dict]:
    """Why this arm may not execute ``leg``, or None. ``static`` adds the cell's
    obstacles (the gates; a backend's own emitted check covers them at plan time)."""
    if not leg:
        return None
    status, v = limits(cell_id, robot_ns, leg)
    if status == "error":
        return _error(robot_ns, "joint limits", v)
    if status == "hit" and v is not None:
        from agents.trajectory.tools.joint_limits import describe
        return {"robot_ns": robot_ns, "reason_code": "joint_limit_violation",
                "joint_limit_violation": v, "error": describe(v, robot_ns)}
    status, hit = self_hit(cell_id, robot_ns, leg)
    if status == "error":
        return {"robot_ns": robot_ns, "reason_code": "self_collision_check_error",
                "error": (f"{robot_ns}'s leg could not be checked against the arm "
                          f"itself ({hit})")}
    if status == "hit" and hit is not None:
        return {"robot_ns": robot_ns, "reason_code": "trajectory_self_collision",
                "collision": hit,
                "error": (f"{robot_ns}'s leg drives the arm into itself ({hit['object_id']}, "
                          f"{hit['clearance_m']} m) between samples {hit['sample_index']} "
                          f"and {hit['sample_index'] + 1}")}
    if static:
        status, hit = static_hit(cell_id, robot_ns, leg)
        if status == "error":
            return _error(robot_ns, "static obstacles", hit)
        if status == "hit" and hit is not None:
            return {"robot_ns": robot_ns, "reason_code": "trajectory_in_collision",
                    "collision": hit,
                    "error": (f"{robot_ns}'s leg breaches {hit.get('object_id')} "
                              f"({hit.get('clearance_m')} m) at sample "
                              f"{hit.get('sample_index')}")}
    return None


def unchecked(cell_id: str, robot_ns: str, leg, *, static: bool = False) -> list:
    """The checks that could not run on ``leg`` (each ``{"check", "reason"}``) —
    what a caller reports instead of calling the leg checked."""
    out = []
    for name, fn in (("limits", limits), ("self", self_hit)) + (
            (("static", static_hit),) if static else ()):
        status, why = fn(cell_id, robot_ns, leg)
        if status == "unavailable":
            out.append({"check": name, "reason": why})
    return out


def clear_cache() -> None:
    with _LOCK:
        _VERDICTS.clear()
