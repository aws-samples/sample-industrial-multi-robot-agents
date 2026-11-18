"""The planners this orchestrator distributes planning runs to.

A stack may run several planner containers (``make sim PLANNERS=3``). Each one
runs ONE planning run at a time and registers itself here — its MCP URL, and
with every heartbeat whether it is busy (``agents/trajectory/planner_registration``).
The orchestrator then:

* **submits** a new run to a FREE planner (:func:`pick_free`): the registry's
  view first, confirmed by asking the planner itself (``planner_status``), because
  a heartbeat can be seconds old; the planner refuses atomically if it is busy
  anyway, and the next free one is tried;
* **routes** everything about an existing run — status, pause, resume, cancel —
  to the planner that OWNS it (:func:`owner_of`): a run's checkpoint lives on
  that planner's private scratch, so only it can resume the run. The owner is
  remembered at submission and, after an orchestrator restart, found by asking
  the planners (each knows its runs by id, ``plan_jobs``);
* **reads** (plans, skills, reachability) from any live planner.

A planner that misses ``STALE_AFTER_BEATS`` heartbeats is not given new work.
``TRAJECTORY_AGENT_HOST`` / ``TRAJECTORY_AGENT_PORT`` remain a static planner, so
a single-planner stack that never registers works exactly as before, and
``TRAJECTORY_PLANNERS`` (comma-separated MCP URLs) lists static planners.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Optional

STALE_AFTER_BEATS = 3

#: Which MCP URLs may register as planners. The registration route has no
#: authentication of its own (like every endpoint of this stack, it is published
#: on this machine only), and a registered planner receives planning requests —
#: cell data — so by default only the stack's own planner containers qualify.
#: A split deployment with planners elsewhere widens it deliberately
#: (PLANNER_ALLOWED_URLS, a regular expression).
DEFAULT_ALLOWED_URLS = r"^http://trajectory-agent(-[0-9]+)?:8890/mcp$"
DEFAULT_HEARTBEAT_S = 10.0

_lock = threading.Lock()
_planners: dict[str, dict] = {}       # planner_id -> record
_owners: dict[str, str] = {}          # tree_id -> planner_id
_last_assigned: dict[str, float] = {} # planner_id -> when it last got a run

#: ``call(url, tool, args, timeout) -> dict`` — the orchestrator's MCP client.
Caller = Callable[[str, str, dict, float], dict]


def _static() -> dict[str, dict]:
    out = {}
    host = os.environ.get("TRAJECTORY_AGENT_HOST", "trajectory-agent")
    port = os.environ.get("TRAJECTORY_AGENT_PORT", "8890")
    urls = [f"http://{host}:{port}/mcp"] + [
        u.strip() for u in (os.environ.get("TRAJECTORY_PLANNERS") or "").split(",") if u.strip()]
    for url in urls:
        out.setdefault(url, {"planner_id": url, "mcp_url": url, "static": True,
                             "busy": None, "last_seen": None})
    return out


def register(record: dict) -> dict:
    """A planner's registration or heartbeat. Returns what was stored."""
    pid, url = str(record.get("planner_id") or ""), str(record.get("mcp_url") or "")
    if not pid or not url.startswith(("http://", "https://")):
        raise ValueError("a planner registers with a planner_id and an http(s) mcp_url")
    allowed = os.environ.get("PLANNER_ALLOWED_URLS") or DEFAULT_ALLOWED_URLS
    if not re.fullmatch(allowed, url):
        raise ValueError(f"mcp_url {url!r} is not an allowed planner URL "
                         "(PLANNER_ALLOWED_URLS)")
    now = time.time()
    with _lock:
        prev = _planners.get(pid, {})
        rec = {"planner_id": pid, "mcp_url": url, "static": False,
               "busy": bool(record.get("busy")), "active": record.get("active"),
               "backends": record.get("backends") or [],
               "heartbeat_s": float(record.get("heartbeat_s") or DEFAULT_HEARTBEAT_S),
               "registered": prev.get("registered", now), "last_seen": now}
        _planners[pid] = rec
        active = rec.get("active") or {}
        if active.get("tree_id"):
            _owners[active["tree_id"]] = pid
        return dict(rec)


def _live(rec: dict, now: float) -> bool:
    if rec.get("static"):
        return True
    return now - (rec.get("last_seen") or 0) <= STALE_AFTER_BEATS * rec.get("heartbeat_s", 10)


def planners() -> list[dict]:
    """Every known planner, registered ones first, with ``live`` / ``busy``."""
    now = time.time()
    with _lock:
        regs = [dict(r, live=_live(r, now)) for r in _planners.values()]
        urls = {r["mcp_url"] for r in regs}
    statics = [dict(r, live=True) for u, r in _static().items() if u not in urls]
    # A registered planner supersedes the static default for the same container.
    return sorted(regs, key=lambda r: r["planner_id"]) + ([] if regs else statics)


def url_of(planner_id: str) -> Optional[str]:
    for p in planners():
        if p["planner_id"] == planner_id:
            return p["mcp_url"]
    return None


def any_url() -> str:
    """A planner for READS: a live free one if there is one, else any live one."""
    live = [p for p in planners() if p["live"]]
    free = [p for p in live if not p.get("busy")]
    pick = (free or live or [{"mcp_url": next(iter(_static()))}])[0]
    return pick["mcp_url"]


def pick_free(call: Caller, exclude: tuple = ()) -> Optional[dict]:
    """A planner that is free NOW (asked, not assumed), least recently used first."""
    cands = [p for p in planners() if p["live"] and p["planner_id"] not in exclude
             and p.get("busy") is not True]
    cands.sort(key=lambda p: _last_assigned.get(p["planner_id"], 0.0))
    for p in cands:
        status = call(p["mcp_url"], "planner_status", {}, 5.0)
        if isinstance(status, dict) and status.get("busy") is False:
            return {**p, "status": status}
        if isinstance(status, dict) and status.get("busy") is True and not p.get("static"):
            with _lock:
                if p["planner_id"] in _planners:
                    _planners[p["planner_id"]]["busy"] = True
    return None


def submit(call: Caller, tool: str, args: dict, timeout: float = 120.0) -> dict:
    """Run ``tool`` (plan_trajectories) on a free planner; remember who owns the run.

    Tries each free planner once: one that turns out busy (``planner_busy``, it was
    claimed between the check and the call) is skipped for the next. With none
    free the reply says so, with the planners and what each is doing.
    """
    tried: list = []
    while True:
        p = pick_free(call, exclude=tuple(tried))
        if p is None:
            return {"accepted": False, "error_kind": "no_free_planner",
                    "error": ("every planner is busy or unreachable — one planning run "
                              "per planner; wait, stop a run, or start more planners "
                              "(make sim PLANNERS=N)"),
                    "planners": [{k: q.get(k) for k in ("planner_id", "live", "busy", "active")}
                                 for q in planners()]}
        tried.append(p["planner_id"])
        reply = call(p["mcp_url"], tool, args, timeout)
        if isinstance(reply, dict) and reply.get("error_kind") == "planner_busy":
            continue
        if isinstance(reply, dict) and reply.get("accepted") and reply.get("tree_id"):
            with _lock:
                _owners[reply["tree_id"]] = p["planner_id"]
                _last_assigned[p["planner_id"]] = time.time()
        return {**(reply if isinstance(reply, dict) else {"result": reply}),
                "planner_id": p["planner_id"]}


def owner_of(tree_id: str, call: Optional[Caller] = None) -> Optional[dict]:
    """The planner that owns ``tree_id`` — remembered, else found by asking each."""
    with _lock:
        pid = _owners.get(tree_id)
    if pid:
        for p in planners():
            if p["planner_id"] == pid:
                return p
    if call is None:
        return None
    for p in planners():
        if not p["live"]:
            continue
        status = call(p["mcp_url"], "get_plan_status", {"tree_id": tree_id}, 10.0)
        if isinstance(status, dict) and status.get("status") not in (None, "unknown") \
                and not status.get("error_kind"):
            with _lock:
                _owners[tree_id] = p["planner_id"]
            return p
    return None


def remember(tree_id: str, planner_id: str) -> None:
    with _lock:
        _owners[tree_id] = planner_id


def reset() -> None:
    """Forget everything (tests)."""
    with _lock:
        _planners.clear()
        _owners.clear()
        _last_assigned.clear()
