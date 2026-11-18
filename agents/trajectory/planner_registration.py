"""A planner announces itself to the orchestrator, and keeps announcing.

Several planner containers can serve one stack; the orchestrator distributes
planning runs to whichever is free (``agents/orchestrator/planner_registry.py``).
For that it has to know they exist, so each planner registers its MCP endpoint
at start-up and then heartbeats every ``PLANNER_HEARTBEAT_S`` seconds with
whether it is busy:

    POST {ORCHESTRATOR_URL}/a2a/planners/register
    {"planner_id", "mcp_url", "busy", "active": {tree_id, cell_id, status} | null,
     "backends": [...], "heartbeat_s"}

The heartbeat is also how a planner disappears: one that stops sending is
dropped from distribution after a few missed beats. Registration is best
effort — an orchestrator that is not up yet is retried at the next beat, and
nothing here can stop the planner from serving.

Configuration (all set by the compose generator):
  PLANNER_ID            this planner's name (default: the hostname)
  PLANNER_ADVERTISE_URL the MCP URL the orchestrator should call,
                        e.g. http://trajectory-agent-2:8890/mcp
  ORCHESTRATOR_URL      http://orchestrator:8889 (empty: no registration)
  PLANNER_HEARTBEAT_S   seconds between beats (10)
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Callable, Optional

_thread: Optional[threading.Thread] = None
_stop = threading.Event()
last_result: dict = {}


def config() -> dict:
    return {"orchestrator_url": (os.environ.get("ORCHESTRATOR_URL") or "").rstrip("/"),
            "mcp_url": os.environ.get("PLANNER_ADVERTISE_URL") or "",
            "heartbeat_s": float(os.environ.get("PLANNER_HEARTBEAT_S") or 10.0)}


def beat(status: Callable[[], dict], timeout: float = 5.0) -> dict:
    """Send one registration/heartbeat; the reply, or ``{"error": ...}``."""
    cfg = config()
    if not cfg["orchestrator_url"] or not cfg["mcp_url"]:
        return {"skipped": "ORCHESTRATOR_URL / PLANNER_ADVERTISE_URL not set"}
    body = json.dumps({**status(), "mcp_url": cfg["mcp_url"],
                       "heartbeat_s": cfg["heartbeat_s"]}).encode()
    req = urllib.request.Request(f"{cfg['orchestrator_url']}/a2a/planners/register",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"error": str(exc)}


def start(status: Callable[[], dict]) -> None:
    """Register now and heartbeat in the background (idempotent)."""
    global _thread
    if _thread and _thread.is_alive():
        return

    def loop():
        while True:
            last_result.clear()
            last_result.update(beat(status))
            if _stop.wait(config()["heartbeat_s"]):
                return

    _thread = threading.Thread(target=loop, name="planner-registration", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
