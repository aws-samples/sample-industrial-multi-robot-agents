"""
Cell Agent.

Two clearly-separated layers:

  * **Execution (deterministic, no LLM).** ``execute_behaviour_tree`` builds the
    ROS cell runtime, starts the collision monitor, and runs the deterministic
    ``BehaviourTreeExecutor``. This is what actually moves robots and enforces
    timing/interlock/collision gates. It is fast and repeatable.

  * **Recovery (LLM, high level).** When deterministic execution FAILS, the
    structured failure is handed to an LLM (``reason_about_failure``) that
    diagnoses it and recommends an action (retry / replan / adjust / abort) plus
    a concise message to propagate back to the orchestrator. The LLM never drives
    the robots — it only reasons about failures.

``execute_behaviour_tree`` therefore returns::

    success: {"success": True, "executed_nodes": int, "elapsed_s": float}
    failure: {"success": False, "failure": {...}, "recovery": {...},
              "elapsed_s": float}
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agents.cell.tools.cell_state import get_all_robot_states, trigger_estop
from agents.cell_defaults import default_cell_id
from agents.llm_model import build_llm, effective_model_id

_REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_ID = default_cell_id()


# ─── LLM recovery/reasoning layer ─────────────────────────────────────────────

RECOVERY_SYSTEM_PROMPT = """\
You are the Cell Agent's RECOVERY layer for a multi-robot industrial cell.

You do NOT drive robots. Deterministic code already executed the BehaviourTree
and it FAILED. Your job is to diagnose the failure and recommend what the
orchestrator should do next.

You are given a structured failure from the executor. Key reason_code values:
  - collision            : a robot's live position breached the collision model.
                           The failure carries collision.object_id + collision.point.
                           → usually "replan": the planner must apply an approach
                             strategy (stage above & descend, intermediate waypoint,
                             reorient) around that object.
  - schedule_deviation   : a robot drifted from its planned position/time.
                           → usually "retry" (transient) or investigate hardware.
  - position_gate_timeout: an interlock/approach position was never reached.
                           → "retry" or "replan" depending on whether it recurs.
  - empty_trajectory / unknown_node / invalid_tree : a planning/assembly bug.
                           → "abort" and report; re-planning won't help as-is.

You may call get_all_robot_states to inspect current robot positions.

Respond with a SINGLE fenced ```json block:
{
  "recommended_action": "retry" | "replan" | "adjust_and_replan" | "abort",
  "rationale": "<one or two sentences>",
  "propagate_to_orchestrator": "<concise message the orchestrator should act on>"
}
"""

_RECOVERY_TOOLS = [get_all_robot_states, trigger_estop]


def create_cell_recovery_agent(model: str | None = None):
    """Create the LLM recovery/reasoning agent (does not drive robots).

    ``model`` is the per-request override; ``None`` uses whatever the caller bound
    for this operation, else the startup default. See :mod:`agents.llm_model`.
    """
    return create_react_agent(model=build_llm(model), tools=_RECOVERY_TOOLS,
                              prompt=RECOVERY_SYSTEM_PROMPT)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _fallback_recovery(failure: dict) -> dict:
    """Deterministic recovery recommendation when the LLM is unavailable."""
    code = failure.get("reason_code", "unknown")
    mapping = {
        "collision":             "replan",
        "schedule_deviation":    "retry",
        "position_gate_timeout": "retry",
        "empty_trajectory":      "abort",
        "unknown_node":          "abort",
        "invalid_tree":          "abort",
    }
    action = mapping.get(code, "abort")
    return {
        "recommended_action": action,
        "rationale": f"heuristic mapping for reason_code={code} (LLM reasoning unavailable)",
        "propagate_to_orchestrator": failure.get("reason", "execution failed"),
        "llm": False,
    }


def reason_about_failure(cell_id: str, behaviour_tree: dict, failure: dict,
                         model: str | None = None) -> dict:
    """Diagnose a deterministic-execution failure and recommend a next action.

    ``model`` selects the reasoning model for this one diagnosis; ``None`` uses
    the operation's bound model, else the startup default.

    Best-effort: if the LLM is unavailable (no credentials, offline), returns a
    deterministic heuristic recommendation so the orchestrator still gets a clear,
    actionable verdict.
    """
    try:
        agent = create_cell_recovery_agent(model)
        prompt = (
            f"Cell '{cell_id}' BehaviourTree execution failed.\n\n"
            f"Structured failure:\n{json.dumps(failure, indent=2)}\n\n"
            f"Tree summary: {json.dumps(behaviour_tree.get('metadata', {}))}\n\n"
            "Diagnose and respond with the json block."
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"configurable": {"thread_id": f"recovery_{behaviour_tree.get('tree_id', 'bt')}"}},
        )
        last = result["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        if isinstance(content, list):  # Bedrock block content
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            ) or str(content)
        parsed = _extract_json(content)
        if parsed:
            parsed.setdefault("llm", True)
            parsed.setdefault("model", effective_model_id(model))
            return parsed
        return {
            "recommended_action": _fallback_recovery(failure)["recommended_action"],
            "rationale": "LLM produced no parseable recommendation; used heuristic",
            "propagate_to_orchestrator": (content or failure.get("reason", "execution failed"))[:500],
            "llm": True,
            "model": effective_model_id(model),
        }
    except Exception as exc:  # noqa: BLE001 - degrade to heuristic
        rec = _fallback_recovery(failure)
        rec["note"] = f"recovery LLM error: {exc}"
        return rec


# ─── Live run handle (so an operator's Stop can reach the running tree) ───────
# The executor already knows how to stop itself mid-run (its `_abort` event is
# checked by every walker and by the dispatch sleep) — what was missing was a way
# for ANOTHER request to find the executor that is running right now. An
# execution occupies the process for its whole duration and the cell is
# single-writer by construction (the orchestrator's RunRegistry answers 409 to a
# second run), so one module-level slot is the whole mechanism.
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_EXECUTOR = None   # BehaviourTreeExecutor | None


def abort_active_execution(reason: str = "aborted by operator") -> dict:
    """Stop the behaviour tree currently executing in this process, if any.

    Software abort only: no further node is dispatched, and the trajectory a
    controller has already accepted finishes. It does NOT halt an arm and does
    NOT latch an e-stop — see :meth:`BehaviourTreeExecutor.abort`.

    Idempotent: with nothing running it reports ``aborted: False`` rather than
    failing, because Stop on an idle cell is a reasonable thing to press.
    """
    with _ACTIVE_LOCK:
        executor = _ACTIVE_EXECUTOR
    if executor is None:
        return {"aborted": False, "reason": "nothing is executing"}
    first = executor.abort(reason)
    return {"aborted": True, "was_first_failure": bool(first), "reason": reason}


def active_execution() -> bool:
    """True while a behaviour tree is executing in this process."""
    with _ACTIVE_LOCK:
        return _ACTIVE_EXECUTOR is not None


# ─── Deterministic execution entry point ──────────────────────────────────────

def _remember_weld_poses(cell_id: str, behaviour_tree: dict) -> None:
    """Keep the weld poses a plan carries (``metadata.weld_poses``) in this
    process. The arms may be stopped inside one of those seams — an e-stop, an
    abort — and the home reset plans their way out HERE, in the cell process,
    where nothing was planned: without the poses its lift-off cannot tell the arm
    is in a seam and would drag the tool sideways out of it. Best-effort (an
    older plan carries none); a tree without poses changes nothing."""
    poses = (behaviour_tree.get("metadata") or {}).get("weld_poses")
    if not poses:
        return
    try:
        from agents.trajectory.tools.trajectory_cache import load_weld_pose_record
        load_weld_pose_record(cell_id, poses)
    except Exception as exc:  # noqa: BLE001 — trajectory stack absent
        print(f"[cell] weld poses of {behaviour_tree.get('tree_id')} not kept: {exc}",
              flush=True)


def execute_behaviour_tree(behaviour_tree: dict, model: str | None = None) -> dict:
    """Execute a BehaviourTree deterministically; reason about any failure.

    Builds the ROS cell runtime + collision monitor, runs the deterministic
    ``BehaviourTreeExecutor``, and — on failure — invokes the LLM recovery layer.

    ``model`` reaches ONLY that recovery layer. Execution itself is deterministic
    and has no model to choose: the trajectories were planned and validated before
    this call, and no model may influence what the arms are commanded to do.
    """
    from agents.cell.bt_executor import BehaviourTreeExecutor, ExecConfig
    from agents.cell.collision_monitor import CollisionMonitor

    global _ACTIVE_EXECUTOR

    cell_id = behaviour_tree.get("cell_id") or CELL_ID
    t0 = time.monotonic()
    _remember_weld_poses(cell_id, behaviour_tree)

    # Build the live runtime; a missing ROS stack is an execution failure, not a
    # silent success (nothing would actually move).
    try:
        from agents.cell.cell_runtime import build_runtime
        runtime = build_runtime(cell_id)
    except RuntimeError as exc:
        return {
            "success": False,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "failure": {"reason_code": "runtime_unavailable", "reason": str(exc)},
            "recovery": {"recommended_action": "abort",
                         "rationale": "cell runtime (ROS) unavailable",
                         "propagate_to_orchestrator": str(exc), "llm": False},
        }

    monitor = CollisionMonitor(cell_id, runtime).start()
    try:
        # Clear any e-stop LATCHED by a prior aborted run. Otherwise the robot
        # agents reject every dispatch ("cell is E-STOPPED") and this run reports
        # a hollow success on the executor's timer while nothing actually moves.
        try:
            runtime.clear_estop("new execution start")
            time.sleep(0.5)   # let the clear reach the robot agents before dispatch
        except Exception:
            pass
        executor = BehaviourTreeExecutor(runtime, monitor, ExecConfig.for_cell(cell_id))
        # Publish the executor so abort_active_execution() can reach it, and
        # retract it in the same `finally` that tears the runtime down — a stale
        # handle would let a Stop "succeed" against a run that already ended.
        with _ACTIVE_LOCK:
            _ACTIVE_EXECUTOR = executor
        result = executor.execute(behaviour_tree)
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_EXECUTOR = None
        monitor.stop()
        runtime.shutdown()

    result["elapsed_s"] = round(time.monotonic() - t0, 2)

    if result.get("success"):
        return result

    # Deterministic execution failed → hand the structured failure to the LLM.
    failure = result.get("failure", {"reason_code": "unknown", "reason": "execution failed"})
    result["recovery"] = reason_about_failure(cell_id, behaviour_tree, failure, model)
    return result
