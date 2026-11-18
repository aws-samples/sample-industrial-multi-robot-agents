"""The planner's progress, on a DDS topic of its own.

A planning run takes minutes and the operator watching it is looking at the
dashboard. Until now the only way for the dashboard to see inside a live run was
to poll the orchestrator, which read the planner's event file off a shared volume
— three processes and a filesystem in the path of a question about *now*.

The planner already lives in a ROS graph. So every plan event is also published
here, on one dedicated topic per cell:

    /<cell_id>/plan_progress        std_msgs/String, one JSON event per message

The cell name is the prefix, and it is what keeps two cells in the same plant from
reading as one run. It is used verbatim: the dashboard subscribes to
``/<CELL_ID>/plan_progress`` composed from its own ``CELL_ID``, so a normalised
prefix would be a live view that is silently always empty. A cell id that cannot be
a ROS name is therefore refused rather than rewritten, and a missing one is refused
rather than published on the shared ``/plan_progress`` — see :func:`cell_of`. The
fallback for a caller that does not carry the cell is ``CELL_ID``, the identity the
container was started with.

The transport itself is also configuration this process is *given*, never one it
decides: ``ROS_DOMAIN_ID``, ``RMW_IMPLEMENTATION`` and the cell's
``cyclonedds.xml`` (or the Zenoh broker override) are written into the container's
environment by ``scripts/generate_compose.py`` at start-up, and ``rclpy.init`` and
the RMW read them from there. :func:`transport` reports them for ``/health``,
because a domain or RMW mismatch is otherwise invisible — both sides are healthy
and simply never hear each other.

Shape of the payload — the plan event exactly as :mod:`plan_events` records it,
plus the two fields a subscriber needs to know what it is looking at:

    {"schema": "plan_progress/1", "cell_id": …, "trace_id": …,
     "seq": 12, "at": "…", "kind": "operation_planned", …}

Why ``std_msgs/String`` carrying JSON and not a typed message: that is this repo's
convention for agent-level state (per-robot status is published the same way), and
``ros2_packages/agent_interfaces`` is not a built package — it ships ``msg/`` with
no ``CMakeLists.txt``, so a typed message would mean a new build step in every
container that wants to read this. The event vocabulary also changes with the
planner; a string keeps the schema where the schema is decided.

QoS is ``KEEP_LAST(200)`` + ``TRANSIENT_LOCAL``: a dashboard that opens mid-run
gets the recent history of the run in progress rather than a blank feed, and one
that opens after it ends still sees how it finished. Bounded on purpose — the
unbounded record of a run is the JSONL artefact the orchestrator keeps.

Never raises into the planner, and never blocks it: a cell with no ROS (a test, a
host script, ``rclpy`` absent) publishes nothing and says so once. Losing the live
view is an availability failure; losing the plan over it would not be.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Optional

SCHEMA = "plan_progress/1"

# The topic suffix. The dashboard composes the same name through its own
# ``cell_topic()`` helper (``/<CELL_ID>`` + suffix), so this constant and that
# helper are the only two places the name exists.
TOPIC_SUFFIX = "/plan_progress"

# Deep enough to carry the interesting part of a run to a late subscriber, small
# enough that the middleware is not asked to keep a planning session in RAM.
HISTORY_DEPTH = 200

# A legal ROS 2 name token: alphabetic or underscore first, then alphanumerics and
# underscores. The cell id becomes the topic's first token, and it is used
# VERBATIM — see cell_of() on why it is validated rather than normalised.
_CELL_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The transport the container was started with. Read, never defaulted: a default
# written here would be a second configuration competing with the one compose
# passes in, and the symptom of the wrong one winning — nobody hears the planner —
# surfaces a long way from the cause.
_TRANSPORT_VARS = ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI",
                   "ZENOH_CONFIG_OVERRIDE", "ZENOH_SESSION_CONFIG_URI")

_lock = threading.Lock()
_node = None            # rclpy Node, created on first publish
_publishers: dict = {}  # cell_id -> Publisher
_disabled_reason: Optional[str] = None
# cell_id -> why this ONE cell cannot be published for. Separate from
# _disabled_reason, which is about the middleware: a cell whose id cannot be a ROS
# name must not take the live view down for every other cell in the process.
_refused: dict = {}


def transport() -> dict:
    """The DDS/Zenoh configuration this process was started with, as passed in.

    The generator writes it into every container's environment (see
    ``scripts/generate_compose.py``), so this is a report and not a decision.
    Absent variables are absent from the answer.
    """
    return {var: os.environ[var] for var in _TRANSPORT_VARS
            if os.environ.get(var)}


def cell_of(cell_id: Optional[str]) -> str:
    """The cell this event belongs to, or ``ValueError``.

    Falls back to ``CELL_ID`` — the identity compose hands the container — so a
    call site that does not carry the cell around is still scoped to the right
    one. Validated and never normalised: the dashboard subscribes to
    ``/<CELL_ID>/plan_progress`` built from the cell name verbatim, so a rewritten
    prefix would be a live view that is silently always empty. And no cell at all
    is refused rather than published on ``/plan_progress``, which is the one topic
    every cell in a plant would collide on.
    """
    resolved = (cell_id or os.environ.get("CELL_ID") or "").strip()
    if not resolved:
        raise ValueError("no cell id: a plan-progress topic is always cell-scoped")
    if not _CELL_TOKEN.match(resolved):
        raise ValueError(f"cell id {resolved!r} cannot be a ROS name — the "
                         f"dashboard subscribes to the cell name verbatim")
    return resolved


def topic_for(cell_id: Optional[str]) -> str:
    """``/<cell_id>/plan_progress`` — one topic per cell, never a shared one."""
    return f"/{cell_of(cell_id)}{TOPIC_SUFFIX}"


def _ensure_publisher(cell_id: str):
    """The publisher for this cell, creating the node on first use.

    Lazy because importing this module must not start a ROS node: it is imported
    by the event vocabulary, which is imported by tests and by host scripts that
    have no middleware. ``None`` means "not available", with the reason printed
    once — a planner that logs the same missing-rclpy line per weld is a planner
    whose real log is unreadable.
    """
    global _node, _disabled_reason

    if _disabled_reason is not None:
        return None
    if os.environ.get("DDS_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        # A planning-only stack (make sim ONLY=trajectory) runs without the ROS
        # graph: no rclpy is imported and no node is started. The durable event
        # stream is unaffected — progress is read over MCP (get_plan_events).
        _disabled_reason = "DDS disabled (DDS_DISABLED=1): progress is on MCP only"
        print(f"[plan_progress] {_disabled_reason}", flush=True)
        return None
    publisher = _publishers.get(cell_id)
    if publisher is not None:
        return publisher
    try:
        topic = topic_for(cell_id)
    except ValueError:
        # A cell the topic cannot carry. Refused here as well as in publish(),
        # and WITHOUT latching _disabled_reason: this is one cell's problem, not
        # the middleware's.
        return None

    try:
        import rclpy
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from std_msgs.msg import String

        if not rclpy.ok():
            # The planner process may have no node of its own (it is an MCP
            # server, not a ROS component), so init here — with args=None so it
            # picks up the same ROS_DOMAIN_ID / RMW settings as every other
            # service in the compose file.
            rclpy.init(args=None)
        if _node is None:
            _node = rclpy.create_node("plan_progress_publisher")
        qos = QoSProfile(
            depth=HISTORY_DEPTH,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            # A dashboard that opens mid-run must see the run, not wait for the
            # next event — which on a long weld can be a minute away.
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        publisher = _node.create_publisher(String, topic, qos)
        _publishers[cell_id] = publisher
        # Says the transport too, because a domain or RMW mismatch is otherwise
        # invisible: both sides are healthy and simply never hear each other.
        print(f"[plan_progress] publishing on {topic} "
              f"({transport() or 'transport unconfigured — image defaults'})",
              flush=True)
        return publisher
    except Exception as exc:  # noqa: BLE001
        _disabled_reason = str(exc)
        print(f"[plan_progress] no live progress topic ({exc}) — the durable "
              f"event stream is unaffected", flush=True)
        return None


def publish(cell_id: Optional[str], trace_id: str, event: dict) -> bool:
    """Publish one plan event. ``True`` if it went out on the wire.

    Swallows everything: this is the *view* of a planning run, and a planning run
    must not fail because nobody was watching.
    """
    try:
        with _lock:
            try:
                cell = cell_of(cell_id)
            except ValueError as exc:
                # Per cell, and said once: a planner that logs this per weld has a
                # log nobody can read. The durable event stream is unaffected —
                # it is not scoped by a ROS name.
                key = str(cell_id)
                if key not in _refused:
                    _refused[key] = str(exc)
                    print(f"[plan_progress] not published for {key!r}: {exc}",
                          flush=True)
                return False
            publisher = _ensure_publisher(cell)
            if publisher is None:
                return False
            from std_msgs.msg import String

            message = String()
            message.data = json.dumps(
                {"schema": SCHEMA, "cell_id": cell, "trace_id": trace_id,
                 **event}, default=str)
            publisher.publish(message)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[plan_progress] event not published ({exc})", flush=True)
        return False


def describe() -> dict:
    """Whether the live topic is up — for /health and for a puzzled operator."""
    return {"schema": SCHEMA, "topic_suffix": TOPIC_SUFFIX,
            "cells": sorted(_publishers), "available": _disabled_reason is None,
            "unavailable_because": _disabled_reason,
            # What compose handed this container, so an operator can compare it
            # with the dashboard's without shelling into either.
            "transport": transport(),
            "refused_cells": dict(_refused)}
