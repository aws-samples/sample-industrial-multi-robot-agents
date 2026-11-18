"""
Cell client — the orchestrator's link to the cell controller via MCP over HTTP.

The orchestrator uses this to hand the planned BehaviourTree to the cell for
DETERMINISTIC execution and to receive the structured result back — including,
on failure, the executor's failure detail and the LLM recovery layer's
recommendation. This is the channel that propagates execution outcomes (and
recovery advice) back up to the orchestrator so it can decide whether to retry,
re-plan, or abort.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx
from langchain_core.tools import tool

from agents.llm_model import current_llm_model
from agents.orchestrator.tools.plan_store import get_plan, latest_plan_id


def _cell_base_url() -> str:
    host = os.environ.get("CELL_CONTROLLER_HOST", "cell-controller")
    port = os.environ.get("CELL_MCP_PORT", "8892")
    return f"http://{host}:{port}"


@tool
def execute_behaviour_tree_on_cell(
    cell_id: str,
    behaviour_tree_ref: str,
    timeout_s: float = 1800.0,
) -> dict:
    """
    Run a planned BehaviourTree on the cell controller and return the outcome.

    Calls the cell controller's `execute_behaviour_tree` MCP tool. The cell runs
    the tree deterministically (dispatch + gates + collision monitor); no planner
    or LLM drives the motion.

    Args:
        cell_id:            Cell identifier (e.g. "welding_cell_2_small")
        behaviour_tree_ref: The `behaviour_tree_ref` string returned by
                            `call_trajectory_agent`. The full tree (with all joint
                            trajectories) is resolved from it server-side — you do
                            NOT pass the tree itself.
        timeout_s:          Request timeout in seconds.

    Returns:
        On success: {"success": True, "executed_nodes": int, "elapsed_s": float}
        On failure: {"success": False, "failure": {reason_code, reason, collision?},
                     "recovery": {recommended_action, rationale,
                                  propagate_to_orchestrator}}
        Use `recovery.recommended_action` to decide next steps: "retry",
        "replan"/"adjust_and_replan" (re-invoke the trajectory agent, e.g. with an
        approach strategy around `failure.collision.object_id`), or "abort".
    """
    # Resolve the reference to the real tree stashed by call_trajectory_agent.
    behaviour_tree = get_plan(behaviour_tree_ref)
    if behaviour_tree is None and behaviour_tree_ref:
        # Tolerate the LLM passing a slightly different ref: fall back to the
        # most recently planned tree if there is exactly one obvious candidate.
        fallback = latest_plan_id()
        if fallback is not None:
            behaviour_tree = get_plan(fallback)
    if behaviour_tree is None:
        return {
            "success": False,
            "error": (f"No planned BehaviourTree found for ref "
                      f"'{behaviour_tree_ref}'. Call call_trajectory_agent first "
                      f"and pass its behaviour_tree_ref."),
        }

    base_url = _cell_base_url()
    # Execution is deterministic; the model travels only so the cell's RECOVERY
    # reasoner diagnoses a failure with the model this request asked for. Read
    # from the request binding, not from an LLM-supplied argument.
    arguments = {"behaviour_tree": behaviour_tree, "replay_mode": False}
    bound_model = current_llm_model()
    if bound_model:
        arguments["model"] = bound_model
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "execute_behaviour_tree", "arguments": arguments},
    }
    try:
        response = httpx.post(
            f"{base_url}/mcp", json=payload, timeout=timeout_s,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        rpc_result = response.json()
        if "error" in rpc_result:
            return {"success": False, "error": rpc_result["error"].get("message", "cell MCP error")}
        content = rpc_result.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            return json.loads(content[0].get("text", "{}"))
        return {"success": False, "error": "Empty response from cell controller"}
    except httpx.ConnectError:
        return {
            "success": False,
            "error": f"Cannot connect to cell controller at {base_url}. "
                     "Is the cell-controller container running?",
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}
