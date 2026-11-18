"""
Orchestrator Agent — entry point for A2A task requests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agents.llm_model import bind_model, build_llm, effective_model_id
from agents.orchestrator.tools.cell_spec import (
    read_cell_spec, read_task_program, publish_cell_description,
)
from agents.orchestrator.tools.trajectory_client import (
    call_trajectory_agent, check_robot_reachability,
)
from agents.orchestrator.tools.cell_client import execute_behaviour_tree_on_cell

_REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """\
You are the Orchestrator Agent for a multi-robot industrial cell system.

Your role: receive a task request (program name + cell ID), coordinate the
planning AND execution pipeline, handle failures, and report results.

STANDARD WORKFLOW:
1. Call `read_cell_spec` to read cell configuration and available programs
2. Call `read_task_program` to load the task program operations
3. Call `publish_cell_description` to confirm every arm's URDF is available (served over MCP)
4. Call `call_trajectory_agent` to plan all trajectories → it returns a
   `behaviour_tree_ref` (a string handle) plus a `behaviour_tree_summary`
5. Call `execute_behaviour_tree_on_cell` passing that `behaviour_tree_ref` to run it
6. Handle the execution result:
   - success → report the summary; the task is DONE
   - failure → read `recovery.recommended_action` and act on it:
       * "retry"             → call `execute_behaviour_tree_on_cell` again with
         the same `behaviour_tree_ref`
       * "replan"/"adjust_and_replan" → call `call_trajectory_agent` again,
         applying the recovery hint (e.g. an approach strategy around
         `failure.collision.object_id`), then execute the NEW `behaviour_tree_ref`
       * "abort"             → stop and report the failure clearly
   - Do not retry/replan more than twice; then abort and report.

NEVER attempt to construct, echo, or pass a raw BehaviourTree or trajectory
arrays yourself. The tree lives server-side; you only ever pass the
`behaviour_tree_ref` string.

PRINCIPLES:
- One program at a time per cell
- Always publish cell description before requesting planning
- Report trajectory PLANNING failures with the full error
- Report EXECUTION failures with the executor's reason + the recovery rationale
- A task is complete ONLY when execute_behaviour_tree_on_cell returns success
"""

_TOOLS = [
    read_cell_spec, read_task_program, publish_cell_description,
    call_trajectory_agent, check_robot_reachability,
    execute_behaviour_tree_on_cell,
]


def create_orchestrator_agent(model: str | None = None):
    """Create the orchestrator ReAct agent.

    ``model`` is the per-request override (an alias such as ``"opus-5"``, a full
    model id, or ``"provider:model"``); ``None`` uses the startup default. See
    :mod:`agents.llm_model`.
    """
    return create_react_agent(model=build_llm(model), tools=_TOOLS, prompt=SYSTEM_PROMPT)


def run_task(cell_id: str, program_name: str, model: str | None = None,
             run_id: str | None = None, backend: str | None = None,
             mode: str | None = None, backend_options: dict | None = None) -> dict:
    """Run a task program on a cell.

    ``model`` selects the LLM for THIS task only — an alias (``"opus-5"``), a full
    model id, or ``"provider:model"``. It is bound for the whole request rather
    than passed down argument by argument, so every LLM call the task triggers in
    this process (the recovery reasoner, the HOME staging proposer) runs on the
    same model, and the trajectory agent is told about it explicitly by
    ``call_trajectory_agent``. Omit it to use the startup default.

    ``run_id`` is the id the caller already handed out for this run (the A2A server
    mints one before dispatching, so a client can poll ``/a2a/result/<run_id>``).
    The run log is named after it, which is what keeps the id the client holds and
    the log on disk the same run. Omit it and one is minted when the log is written.

    Persists a timestamped run log (LLM reasoning + actions, trajectory result,
    and behaviour tree summary) for every run, whether it succeeds or fails.
    Returns the ``run_id`` used, so a synchronous caller can find the log too.
    """
    import time
    from agents.orchestrator.run_logger import new_run_id, save_run_log

    run_id = run_id or new_run_id(cell_id)

    # ``backend`` / ``mode``: the trajectory backend and planning mode for this
    # task, bound like the model so the planning call carries them whatever the
    # orchestrator LLM writes. Omitted: the planner's stack default.
    from agents.orchestrator.tools.trajectory_client import bind_plan_selection

    with bind_model(model), bind_plan_selection(backend, mode, backend_options):
        agent   = create_orchestrator_agent()
        used    = effective_model_id()
        t0      = time.monotonic()

        try:
            result = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Execute cell program '{program_name}' on cell '{cell_id}'."
                        )
                    ]
                },
                config={"configurable": {"thread_id": f"task_{cell_id}_{int(t0)}"}},
            )
        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            log_files = save_run_log(
                cell_id, program_name, messages=[], elapsed_s=elapsed,
                success=False, error=str(exc), run_id=run_id,
            )
            return {"success": False, "error": str(exc), "elapsed_s": elapsed,
                    "model": used, "run_id": run_id, "log_files": log_files}

        messages = result["messages"]
        last     = messages[-1]
        content  = last.content if hasattr(last, "content") else str(last)
        elapsed  = round(time.monotonic() - t0, 2)

        log_files = save_run_log(
            cell_id, program_name, messages=messages, elapsed_s=elapsed,
            success=True, run_id=run_id,
        )
        return {"success": True, "result": content, "elapsed_s": elapsed,
                "model": used, "run_id": run_id, "log_files": log_files}
