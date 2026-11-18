"""
Trajectory client — calls the trajectory agent via MCP over HTTP/SSE.

The orchestrator agent uses this tool to:
  1. Send planning requests to the trajectory-agent container
  2. Receive the BehaviourTree JSON for delivery to the cell agent
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

import httpx
from langchain_core.tools import tool

from agents.llm_model import current_llm_model
from agents.orchestrator.tools.plan_store import put_plan
from agents.trajectory.tools.btcpp import (
    children as bt_children,
    node_kind,
    port as bt_port,
)


# The backend / mode THIS request asked for (POST /a2a/run), bound for the run
# like the model: the orchestrator LLM writes the planning call, so a choice the
# caller made must not depend on the LLM copying it into constraints.
_SELECTION: ContextVar[dict] = ContextVar("plan_selection", default={})


@contextmanager
def bind_plan_selection(backend: Optional[str] = None, mode: Optional[str] = None,
                        backend_options: Optional[dict] = None):
    """Backend, mode and backend options are forwarded as they are: the planner
    validates them (the orchestrator knows no backend's options)."""
    token = _SELECTION.set({k: v for k, v in (("backend", backend), ("mode", mode),
                                              ("backend_options", backend_options)) if v})
    try:
        yield
    finally:
        _SELECTION.reset(token)


def _traj_base_url(free: bool = False) -> Optional[str]:
    """A planner's base URL from the registry (agents/orchestrator/planner_registry):
    with ``free`` one that can take a planning run now (``None`` if none can),
    else any live one, for reads."""
    from agents.orchestrator import planner_registry
    if free:
        p = planner_registry.pick_free(_registry_call)
        url = p["mcp_url"] if p else None
    else:
        url = planner_registry.any_url()
    return url[: -len("/mcp")] if url and url.endswith("/mcp") else url


def _registry_call(url: str, tool: str, args: dict, timeout: float) -> dict:
    """The registry's MCP probe (planner_status) over the plain /mcp route."""
    try:
        r = httpx.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": tool, "arguments": args}},
                       timeout=timeout, headers={"Content-Type": "application/json"})
        r.raise_for_status()
        content = (r.json().get("result") or {}).get("content") or []
        return json.loads(content[0]["text"]) if content else {}
    except Exception as exc:  # noqa: BLE001 — an unreachable planner is "not free"
        return {"error": str(exc)}


def _summarize_tree(bt: dict) -> dict:
    """Lightweight, LLM-safe summary of a BehaviourTree (no trajectory arrays)."""
    from collections import Counter

    counts: "Counter[str]" = Counter()
    op_ids: list[str] = []

    def walk(n):
        if isinstance(n, dict):
            t = node_kind(n)
            if t:
                counts[t] += 1
            if t == "ExecuteTrajectory":
                oid = bt_port(n, "operation_id")
                if oid:
                    op_ids.append(oid)
            for ch in bt_children(n):
                walk(ch)
            if isinstance(n.get("root"), dict):
                walk(n["root"])

    walk(bt.get("root") or bt)
    return {
        "node_counts": dict(counts),
        "execute_count": counts.get("ExecuteTrajectory", 0),
        "operation_ids": op_ids,
    }


@tool
def call_trajectory_agent(
    cell_id: str,
    operations: list[dict],
    constraints: Optional[dict] = None,
    timeout_s: float = 1800.0,
) -> dict:
    """
    Request trajectory planning from the trajectory agent via MCP.

    Calls the `plan_trajectories` MCP tool on the trajectory-agent service
    and returns the BehaviourTree JSON.

    Args:
        cell_id:     Cell identifier (e.g. "welding_cell_2_small")
        operations:  Operation points list from the task program
        constraints: Optional constraints e.g. {"trajectory_backend": "ompl"}
        timeout_s:   Request timeout in seconds

    Returns:
        On success: {"success": True, "behaviour_tree_ref": str,
                     "behaviour_tree_summary": {node_counts, execute_count,
                     operation_ids}, "planning_time_s": float}. Pass
                     ``behaviour_tree_ref`` straight to
                     ``execute_behaviour_tree_on_cell``.
        On failure: {"success": False, "error": str}

    IMPORTANT: the full planned trajectories are held server-side and referenced
    by ``behaviour_tree_ref`` — you do NOT receive or need to pass the raw
    trajectory arrays. Never try to hand-build or echo the tree yourself.
    """
    # A planner that is FREE: one planning run per planner container.
    base_url = _traj_base_url(free=True)
    if base_url is None:
        return {"success": False, "error_kind": "no_free_planner",
                "error": "every planner is busy or unreachable — wait for a run to "
                         "finish, stop one, or start more planners (make sim PLANNERS=N)"}

    # The model comes from the REQUEST's binding, never from an argument the LLM
    # supplies — same discipline as the planner backend (planner_backend.py): a
    # deterministic choice made by the caller must not be re-decidable by the
    # model executing the task. Omitted when unbound, so the trajectory agent
    # applies its own startup default.
    arguments = {
        "cell_id":    cell_id,
        "operations": operations,
        "constraints": constraints or {},
    }
    bound_model = current_llm_model()
    if bound_model:
        arguments["model"] = bound_model
    # Same for backend / mode: the request's binding wins over whatever the LLM
    # put in constraints; unbound, the planner applies the stack default.
    bound = _SELECTION.get()
    if bound:
        c = dict(arguments["constraints"])
        if bound.get("backend"):
            c["trajectory_backend"] = bound["backend"]
        if bound.get("mode"):
            c["planning_mode"] = bound["mode"]
        if bound.get("backend_options"):
            c["backend_options"] = dict(bound["backend_options"])
        arguments["constraints"] = c

    # MCP tool call via HTTP (streamable HTTP transport)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "plan_trajectories", "arguments": arguments},
    }

    try:
        response = httpx.post(
            f"{base_url}/mcp",
            json=payload,
            timeout=timeout_s,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        rpc_result = response.json()

        if "error" in rpc_result:
            return {"success": False, "error": rpc_result["error"]["message"]}

        content = rpc_result.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text", "{}")
            result = json.loads(text)

            # Stash the full tree server-side and return only a reference + a
            # lightweight summary, so the huge trajectory arrays never pass
            # through the orchestrator LLM (which would corrupt them).
            bt = result.get("behaviour_tree")
            if isinstance(bt, dict) and bt.get("root"):
                tree_id = bt.get("tree_id") or f"bt_{cell_id}_{id(bt)}"
                bt.setdefault("tree_id", tree_id)
                put_plan(tree_id, bt)
                result.pop("behaviour_tree", None)
                result["behaviour_tree_ref"] = tree_id
                result["behaviour_tree_summary"] = _summarize_tree(bt)
            return result

        return {"success": False, "error": "Empty response from trajectory agent"}

    except httpx.ConnectError:
        return {
            "success": False,
            "error": f"Cannot connect to trajectory agent at {base_url}. "
                     "Is the trajectory-agent container running?",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def check_robot_reachability(
    cell_id: str,
    robot_ns: str,
    x: float,
    y: float,
    z: float,
) -> dict:
    """
    Check if a position is reachable by a robot (via trajectory agent MCP).

    Args:
        cell_id:  Cell identifier
        robot_ns: Robot namespace
        x, y, z:  Target position in world frame (metres)

    Returns:
        {"reachable": bool, "ik_solution": [...]} or {"reachable": false, ...}
    """
    base_url = _traj_base_url()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "check_reachability",
            "arguments": {
                "cell_id": cell_id, "robot_ns": robot_ns,
                "x": x, "y": y, "z": z,
            },
        },
    }
    try:
        r = httpx.post(f"{base_url}/mcp", json=payload, timeout=30.0)
        r.raise_for_status()
        content = r.json().get("result", {}).get("content", [{}])
        return json.loads(content[0].get("text", "{}"))
    except Exception as e:
        return {"reachable": False, "error": str(e)}
