"""
Trajectory Agent MCP Server

Exposes the trajectory agent as MCP tools over HTTP/SSE so that any
MCP-compatible client can call it:
  - Dashboard chat  (via langchain-mcp-adapters)
  - IDE agents      (via the IDE's MCP client config)
  - Orchestrator    (HTTP A2A call)
  - Claude Desktop  (stdio or SSE)

Transport: SSE (default, port TRAJ_MCP_PORT=8890)
         or stdio (MCP_TRANSPORT=stdio, for IDE use)

Each tool is a clean Python function with typed inputs and structured JSON
output. The MCP SDK generates tool schemas automatically from type hints and
docstrings.

Usage:
    # Container (SSE)
    python3 /agents/trajectory/mcp_server.py

    # IDE / offline (stdio)
    MCP_TRANSPORT=stdio python3 /agents/trajectory/mcp_server.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP


def _quiet_noisy_loggers() -> None:
    """
    Silence chatty third-party INFO logs (e.g. the repeated
    'Using Bedrock Converse API to generate response') so the console shows
    the meaningful [plan] trajectory trace instead. Override with
    TRAJ_LOG_LEVEL=DEBUG to restore verbosity.
    """
    if os.environ.get("TRAJ_LOG_LEVEL", "").upper() == "DEBUG":
        return
    for name in ("langchain_aws", "langchain", "langchain_core", "botocore",
                 "boto3", "urllib3", "httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


_quiet_noisy_loggers()

# ─── Trajectory agent ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.cell_defaults import default_cell_id as _default_cell_id
from agents.llm_model import (
    MODEL_ALIASES as _MODEL_ALIASES,
    describe_models as _describe_models,
    normalize_model as _normalize_model,
    planning_model as _planning_model,
    resolve_model as _resolve_model,
)
from agents.trajectory.agent import plan as _agent_plan
from agents.trajectory.tools.pinocchio import run_ik as _run_ik, run_fk as _run_fk
from agents.trajectory.tools.planning_tools import run_trajectory_planner as _run_planner
from agents.trajectory.tools import plan_cancellation as _cancel
from agents.trajectory.tools.plan_cancellation import PlanCancelled as _PlanCancelled
from agents.trajectory.tools.knowledge_base import (
    get_warm_start as _get_warm_start,
    get_similar_trajectories as _get_similar,
)

# ─── Async planning jobs ──────────────────────────────────────────────────────
# plan_trajectories is long-running (minutes of LLM reasoning). Blocking the MCP
# call makes clients time out ("fetch failed"), so instead we ACCEPT the request,
# return the assigned tree_id immediately, and run the planner in the background.
# The client polls get_plan_status(tree_id); when status == "done" it fetches the
# tree with get_calculated_plan(cell_id, tree_id). plan_trajectories ALWAYS runs a
# fresh LLM plan — it never reuses a cached/earlier plan.
import asyncio
import threading
import time as _time

from utils import cell_layout, cell_store

# TOPP-RA logs six INFO lines per time-parameterised segment ("No gridpoint
# specified", "Select solver seidel", "Successfully parametrize path"), several
# per weld, and matplotlib announces its font cache — none of it actionable, and
# it buried the planner's own [plan] lines. Warnings and errors still show;
# PLANNER_LIBRARY_LOGS=1 brings the INFO lines back.
if os.environ.get("PLANNER_LIBRARY_LOGS", "0").strip() != "1":
    import logging as _logging
    for _name in ("toppra", "matplotlib"):
        _logging.getLogger(_name).setLevel(_logging.WARNING)

_PLAN_JOBS: dict[str, dict] = {}
_PLAN_JOBS_LOCK = threading.Lock()
_PLAN_TASKS: set = set()


#: Statuses of a run that still occupies this planner.
_ACTIVE_STATUSES = ("planning", "pausing", "cancelling")


def _active_job() -> Optional[dict]:
    """The run this planner is busy with, if any (callers hold no lock)."""
    with _PLAN_JOBS_LOCK:
        for tid, job in _PLAN_JOBS.items():
            if job.get("status") in _ACTIVE_STATUSES:
                return {"tree_id": tid, **{k: v for k, v in job.items()
                                           if k != "operation_ids"}}
    return None


def _busy_refusal(active: dict) -> dict:
    return {"accepted": False, "success": False, "error_kind": "planner_busy",
            "busy_with": {k: active.get(k) for k in ("tree_id", "cell_id", "status",
                                                     "kind", "submitted_s")},
            "error": (f"this planner is already planning {active.get('tree_id')} "
                      f"(cell {active.get('cell_id')}, {active.get('status')}); one planning "
                      "run per planner — wait for it, stop it (cancel_planning / "
                      "pause_planning), or submit to another planner"),
            "planner_id": _PLANNER_ID}


def _reserve(tree_id: str, **fields) -> Optional[dict]:
    """Claim THIS planner for ``tree_id`` — atomically — or return the run that
    already holds it. Check and claim under one lock, so two submissions that
    arrive together cannot both be accepted."""
    with _PLAN_JOBS_LOCK:
        for tid, job in _PLAN_JOBS.items():
            if job.get("status") in _ACTIVE_STATUSES and tid != tree_id:
                return {"tree_id": tid, **{k: v for k, v in job.items()
                                           if k != "operation_ids"}}
        job = _PLAN_JOBS.setdefault(tree_id, {})
        job.update(status="planning", **fields)
        snapshot = {k: v for k, v in job.items() if k != "operation_ids"}
    from agents.trajectory import plan_jobs
    plan_jobs.save(tree_id, snapshot)
    return None


def _release(tree_id: str, **fields) -> None:
    """End a reservation that never became a background run (it failed to start)."""
    _set_job(tree_id, **fields)


def _set_job(tree_id: str, **fields) -> None:
    with _PLAN_JOBS_LOCK:
        job = _PLAN_JOBS.setdefault(tree_id, {})
        job.update(fields)
        snapshot = {k: v for k, v in job.items() if k != "operation_ids"}
    # Also on the planner's own disk, one file per run (plan_jobs), so a restarted
    # planner still answers for this run — and can say it is resumable.
    from agents.trajectory import plan_jobs
    plan_jobs.save(tree_id, snapshot)


def _run_plan_job(cell_id: str, operations: list, constraints: Optional[dict],
                  tree_id: str, model: Optional[str] = None,
                  resume: bool = False) -> None:
    """Background worker: run the (blocking) agent plan and record the outcome.

    Categorises failures so an INFRASTRUCTURE problem (expired token, no
    connectivity) is never reported as if the robot could not reach the welds:
      failure_kind = "credentials" | "network" | "internal" | "planning".
    Per-weld reachability/collision issues are NOT here — they are in the run's
    tool failures with a target_diagnosis/suggestion.

    ``resume=True`` continues the run this ``tree_id`` already started, from its
    checkpoint, instead of planning from the first operation.
    """
    t0 = _time.monotonic()
    with _PLAN_JOBS_LOCK:
        cancelled_early = (_PLAN_JOBS.get(tree_id) or {}).get("cancel_requested")
    if cancelled_early:
        _set_job(tree_id, status="cancelled", success=False, failure_kind="cancelled",
                 error="cancelled before planning started",
                 note="Planning was stopped by an operator; no plan was stored.",
                 elapsed_s=0.0)
        return
    try:
        result = _agent_plan(cell_id, operations, constraints, tree_id=tree_id,
                             model=model, resume=resume)
        if result.get("success"):
            _set_job(tree_id, status="done", success=True,
                     failure_kind=None, error=None, note=None,
                     model=result.get("model"),
                     elapsed_s=round(_time.monotonic() - t0, 1))
        elif result.get("error_kind") == "paused":
            # Its own terminal status, and deliberately not "failed": nothing is
            # wrong, every solved weld is still on disk, and the ONLY correct next
            # action is resume_planning — which a "failed" job would not suggest.
            _set_job(tree_id, status="paused", success=False,
                     failure_kind=None, error=None,
                     paused=True, resumable=bool(result.get("resumable")),
                     checkpoint=result.get("checkpoint"),
                     resume_with=result.get("resume_with") or tree_id,
                     note=result.get("note"),
                     elapsed_s=round(_time.monotonic() - t0, 1))
        else:
            _set_job(tree_id, status="failed", success=False,
                     failure_kind=result.get("error_kind", "planning"),
                     error=result.get("error"), note=result.get("note"),
                     # A failed run that left a checkpoint can be continued rather
                     # than restarted, and the status is where an operator looks.
                     resumable=bool(result.get("resumable")),
                     checkpoint=result.get("checkpoint"),
                     resume_with=result.get("resume_with"),
                     elapsed_s=round(_time.monotonic() - t0, 1))
    except _PlanCancelled as exc:
        # The operator stopped this run. A distinct terminal status, not a
        # failure: nothing is wrong with the cell, the welds or the model, and no
        # partial plan was stored (the tree is only persisted on success).
        _set_job(tree_id, status="cancelled", success=False,
                 failure_kind="cancelled", error=str(exc),
                 note="Planning was stopped by an operator; no plan was stored.",
                 elapsed_s=round(_time.monotonic() - t0, 1))
    except Exception as exc:  # never let a job crash take down the loop
        from agents.trajectory.agent import _classify_error
        kind, msg = _classify_error(exc)
        note = ("Infrastructure/credentials failure — NOT a robot reachability "
                "result; the plan could not run. Refresh the token, recreate the "
                "agent container, and retry."
                if kind in ("credentials", "network")
                else "Internal error — not a robot reachability result.")
        # An expired token is the textbook resumable failure, so say so here too:
        # this handler catches what escapes the planner's own, and the retry advice
        # above is wrong if the run can be continued instead.
        from agents.trajectory import plan_checkpoint as _ckpt
        saved = _ckpt.describe(cell_id, tree_id) or {}
        if saved.get("resumable"):
            note += (f" This run checkpointed at step {saved.get('step')} — "
                     f"resume_planning('{tree_id}') continues from there instead "
                     "of replanning from the first operation.")
        _set_job(tree_id, status="failed", success=False, failure_kind=kind,
                 error=msg, note=note,
                 resumable=bool(saved.get("resumable")),
                 resume_with=tree_id if saved.get("resumable") else None,
                 elapsed_s=round(_time.monotonic() - t0, 1))


# ─── Transport ────────────────────────────────────────────────────────────────
_TRANSPORT = os.environ.get("MCP_TRANSPORT", "sse").lower()
_PORT      = int(os.environ.get("TRAJ_MCP_PORT", "8890"))
#: This planner's name among the planners the orchestrator distributes work to.
_PLANNER_ID = os.environ.get("PLANNER_ID") or __import__("socket").gethostname()

# ─── MCP server ───────────────────────────────────────────────────────────────
mcp = FastMCP(
    "trajectory-agent",
    instructions=(
        "Trajectory planning agent for multi-robot industrial cells.\n"
        "\n"
        "MAIN TOOL: plan_trajectories — give it a cell_id and operation list,\n"
        "get back a BehaviourTree JSON the cell agent can execute. Every plan is\n"
        "persisted automatically.\n"
        "\n"
        "REPLAY TOOLS: list_calculated_plans (see past plans for a cell) and\n"
        "get_calculated_plan (fetch a plan's BehaviourTree + trajectories by\n"
        "tree_id, or the latest) — pass the returned tree straight to the cell's\n"
        "execute_behaviour_tree to replay without re-planning.\n"
        "\n"
        "MODEL SELECTION: every plan_trajectories call may name the LLM to use\n"
        "with `model` (an alias such as 'opus-5', a full model id, or\n"
        "'provider:model'); list_models shows the options and the default.\n"
        "\n"
        "DIAGNOSTIC TOOLS: check_reachability, validate_trajectory_set (any robot\n"
        "count) / validate_trajectory_pair (two-robot shorthand), find_interlock_set\n"
        "(any robot count) / find_interlocks (two-robot shorthand),\n"
        "get_trajectory_knowledge.\n"
        "\n"
        "Operation types: weld, rivet, glue, laser_cut, laser_weld, mark, move\n"
    ),
)


@mcp.tool()
def planner_status() -> dict:
    """
    Whether THIS planner is free to take a planning run.

    A planner container runs one planning run at a time (plan, resume or re-plan).
    The orchestrator asks this before it submits, and each planner also heartbeats
    it to the orchestrator's registry (agents/trajectory/planner_registration.py).

    Returns {"planner_id", "busy": bool, "active": {tree_id, cell_id, status, kind,
    submitted_s} | null, "backends": [names], "registration": {...}}.
    """
    from agents.trajectory import planner_registration, plan_selection
    active = _active_job()
    return {"planner_id": _PLANNER_ID, "busy": active is not None,
            "active": ({k: active.get(k) for k in ("tree_id", "cell_id", "status", "kind",
                                                   "submitted_s")} if active else None),
            "backends": list(plan_selection.valid().get("backends") or []),
            "mcp_url": planner_registration.config()["mcp_url"] or None,
            "registration": dict(planner_registration.last_result)}


@mcp.tool()
def list_trajectory_backends() -> dict:
    """
    The motion-planning backends and planning modes a request may select.

    Built-in backends (agents/trajectory/backends/: ompl, ompl-compound, rrt),
    the bundled ``roboballet`` (a trained RoboBallet policy), plus any registered from another repository (entry point
    ``umarell.trajectory_backends`` or TRAJECTORY_BACKEND_PLUGINS). Pass a name
    as ``backend`` to plan_trajectories / replan_operation.

    Returns {"backends": [name], "modes": [name], "default_backend": str,
    "backend_details": [{name, conforming, home_planner, description}]}.
    ``conforming: false`` backends do not honour the full planning contract and
    their results are stamped ``non_conforming_backend``.
    """
    from agents.trajectory import plan_selection
    from agents.trajectory.tools.planner_backend import resolve_planning_backend
    try:
        default = resolve_planning_backend()
    except ValueError as exc:
        default = f"invalid: {exc}"
    return {**plan_selection.valid(), "default_backend": default}


@mcp.tool()
async def plan_trajectories(
    cell_id: str,
    operations: list[dict],
    constraints: Optional[dict] = None,
    model: Optional[str] = None,
    assignments: Optional[dict] = None,
    skill_priorities: Optional[dict] = None,
    instructions: Optional[str] = None,
    run_skills: Optional[list] = None,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    backend_options: Optional[dict] = None,
) -> dict:
    """
    Plan collision-free parallel trajectories for all operations in a cell.

    Runs the full trajectory planning pipeline:
      1. MRTA allocation + sequence optimisation (IK pre-filter, TSP 2-opt,
         thermal spread, least-movement pairing, cycle-time balance)
      2. Space-Time RRT path planning per step
      3. Interlock computation where robot paths collide
      4. Behaviour Tree assembly

    Args:
        cell_id:     Cell identifier, e.g. "welding_cell_2_small"
        operations:  List of operation points. Each entry:
                       {"id": str, "type": "weld"|"rivet"|...,
                        "position": [x, y, z],
                        "approach_vector": [ax, ay, az],
                        "face": str, "sequence_group": str, "priority": int}
        constraints: Optional planning constraints.
                     May also carry "trajectory_backend" / "planning_mode" (a task
                     program does); `backend` / `mode` below are the direct way.
        backend:     Trajectory backend for THIS plan: "ompl" | "ompl-compound" |
                     "rrt" | "roboballet" (trained policy, see the policy-trainer
                     service) | a plugin. Omitted → the stack default (make sim
                     TRAJECTORY_BACKEND=…), else "ompl".
        mode:        Planning mode for THIS plan: "classic" | "coordinated" |
                     "stepwise". Omitted → the stack default (make sim
                     PLANNING_MODE=…), else "classic". An unknown backend or mode
                     refuses the request.
        backend_options: Options for THIS plan's backend — the same field for
                     every backend; each declares the options it takes
                     (list_trajectory_backends → backend_details[].options).
                     E.g. backend="roboballet", backend_options={"policy":
                     "roboballet-test1"} plans with that trained policy instead
                     of the promoted "current" one. An option the backend does
                     not take, or a policy that is not installed, refuses the
                     request.
        model:       LLM for THIS plan only — an alias ("opus-5", "sonnet-5",
                     "haiku-4.5"), a full model id, or "provider:model". Omit to
                     use the model the stack was started with (Opus 5 by default).
                     It applies to the whole run, including the HOME staging
                     proposals made when the motion planner cannot find a return
                     path. It never affects which MOTION planner runs — that is
                     `constraints.trajectory_backend`, and no model can change it.

        assignments: OPERATOR robot assignments for THIS plan,
                     ``{"W05": "robot2", ...}``. ENFORCED, not suggested: each
                     becomes a hard lock — the allocator places the operation on
                     that robot, the motion planner refuses to plan it on any
                     other arm, and it cannot be reallocated. Overrides an
                     ``assigned_robot`` the task file gives the same operation.
        skill_priorities: Skill priorities for THIS plan only, layered over the
                     cell's skills.yaml (which is not changed):
                     ``{"find-interlocks": 90, "optimize-sequence":
                     {"priority": 10}, "thermal-spread": {"enabled": false}}``.
                     Names are the planning mode's skills (classic mode: the
                     trajectory-classic set — see list_skills).
        instructions: Free-text operator instructions for THIS plan. Put in
                     front of the planner as binding; it must report any it
                     could not meet. (Prose cannot be checked by code — use
                     ``assignments`` for anything that must be guaranteed.)
        run_skills:  Temporary skills for THIS plan only: ``[{"name",
                     "description", "content", "priority"?}]`` or
                     ``[{"skill_md": "<whole SKILL.md>"}]`` (see skill_template).
                     Listed in the planner's catalog and loadable; written
                     nowhere. A name an existing skill already has is refused.

    An unknown operation, robot or skill in these REFUSES the request
    (``accepted: false``) instead of being ignored.

    This call is ASYNCHRONOUS. Planning takes minutes, so the request is ACCEPTED
    and a `tree_id` (the id the finished BehaviourTree will carry) is returned
    immediately; the plan runs in the background. Poll `get_plan_status(tree_id)`
    and, once it reports `status == "done"`, fetch the result with
    `get_calculated_plan(cell_id, tree_id)`. Every call runs a fresh LLM plan.

    Returns (immediately):
        {"accepted": true, "tree_id": str, "status": "planning",
         "num_operations": int, "model": str, "backend": str, "mode": str,
         "selected_by": {"backend": "request"|"stack_default"|"code_default",
                         "mode": ...}}
    """
    # Reject an unusable model NOW, synchronously. Discovering it in the worker
    # thread would report a background "planning failure" minutes later for what
    # is really a rejected argument.
    try:
        requested = _normalize_model(model)
    except ValueError as exc:
        return {"accepted": False, "error": str(exc),
                "hint": f"valid aliases: {', '.join(sorted(_MODEL_ALIASES))}"}

    # Backend and mode: chosen by THIS request, else the stack's defaults, else
    # the code's — and written into the constraints plan() reads, so the run
    # cannot end up on another one than the reply names.
    from agents.trajectory import plan_selection
    try:
        selection = plan_selection.select(backend, mode, constraints,
                                          backend_options=backend_options)
    except plan_selection.SelectionError as exc:
        return {"accepted": False, "error": str(exc), "valid": plan_selection.valid()}
    constraints = plan_selection.apply(constraints, selection)

    # The operator's directives, validated NOW for the same reason as the model:
    # a typo'd weld id or skill name must refuse the request, not be dropped.
    directives = {k: v for k, v in (("assignments", assignments),
                                    ("skill_priorities", skill_priorities),
                                    ("instructions", instructions),
                                    ("run_skills", run_skills)) if v}
    if directives:
        from agents.trajectory import operator_directives
        from agents.trajectory.tools.planning_tools import _cell_robots
        try:
            normalised = operator_directives.normalise(
                directives, operations, list(_cell_robots(cell_id)),
                "trajectory-classic" if selection["mode"] == "classic" else "trajectory")
        except Exception as exc:  # noqa: BLE001 — DirectiveError, or the cell unreadable
            return {"accepted": False, "error": f"directives refused: {exc}"}
        constraints = {**(constraints or {}),
                       operator_directives.OPERATOR_KEY: normalised}

    # Assign the tree id up front so we can return it now and stamp the finished
    # tree with it. Run the blocking planner in a worker thread tracked as a
    # background task, so this call returns without waiting for planning.
    # Early, friendly refusal (plan() takes the cell atomically when it starts).
    from agents.trajectory import cell_lock
    from agents.trajectory.tools.trajectory_cache import new_tree_id
    busy = cell_lock.holder_of(cell_id)
    if busy:
        return {"accepted": False, "error_kind": "cell_busy",
                "error": f"cell '{cell_id}' is already being planned by {busy['holder']}"}
    # Unique even for two submissions in the same second (the id names the plan).
    tree_id = new_tree_id(cell_id)
    effective = _planning_model(requested)
    # ONE planning run per planner container: claimed atomically, refused when
    # busy (the orchestrator then picks another planner).
    holder = _reserve(tree_id, kind="plan", cell_id=cell_id, success=None,
             error=None, num_operations=len(operations), model=effective,
             backend=selection["backend"], mode=selection["mode"],
             backend_options=selection.get("backend_options") or {},
             operation_ids=[str(o.get("id") or o.get("operation_id"))
                            for o in operations if isinstance(o, dict)
                            and (o.get("id") or o.get("operation_id"))],
             submitted_s=_time.time(), planner_id=_PLANNER_ID)
    if holder is not None:
        return _busy_refusal(holder)
    task = asyncio.create_task(
        asyncio.to_thread(_run_plan_job, cell_id, operations, constraints,
                          tree_id, model)
    )
    _PLAN_TASKS.add(task)
    task.add_done_callback(_PLAN_TASKS.discard)
    return {
        "accepted": True,
        "tree_id": tree_id,
        "status": "planning",
        "num_operations": len(operations),
        "model": effective,
        "backend": selection["backend"],
        "mode": selection["mode"],
        "selected_by": {"backend": selection["backend_from"],
                        "mode": selection["mode_from"]},
        **({"backend_options": selection["backend_options"]}
           if selection.get("backend_options") else {}),
        **({"directives": constraints["operator"]}
           if constraints and "operator" in constraints else {}),
        "poll": ("get_plan_status(tree_id); when status=done, "
                 "get_calculated_plan(cell_id, tree_id)"),
    }


@mcp.tool()
def cancel_planning(tree_id: Optional[str] = None,
                    reason: str = "cancelled by operator") -> dict:
    """
    Stop an in-flight plan_trajectories job. Does NOT touch the robots.

    This is the "stop the system's activity" half of the operator's Stop button:
    it ends PLANNING. It cannot move, halt or e-stop an arm — planning is pure
    computation and commands nothing.

    Cancellation is COOPERATIVE, because a plan runs in a worker thread Python
    cannot kill: the run is flagged here and unwinds at its next checkpoint (the
    entry to a planning step, or the next motion request inside OMPL). So this
    call returns immediately and the job's status becomes "cancelled" shortly
    after — poll get_plan_status(tree_id) to see it land. The partial plan is
    discarded; a tree is only persisted on a successful run.

    Args:
        tree_id: The job to stop. Omit to stop EVERY job currently planning.
        reason:  Recorded on the job and shown in its status.

    Returns:
        {"success": true, "cancelled": [tree_id, ...], "still_planning": [...]}
        ``cancelled`` is empty when nothing was planning — pressing Stop twice,
        or with an idle stack, is not an error.
    """
    marked = _cancel.request(tree_id, reason=reason)
    # A run accepted but not started yet (its worker has not registered with the
    # cancellation registry): marked on the job, honoured before it plans at all.
    with _PLAN_JOBS_LOCK:
        pending = [tid for tid, job in _PLAN_JOBS.items()
                   if job.get("status") == "planning" and tid not in marked
                   and (tree_id is None or tid == tree_id)]
    for tid in pending:
        _set_job(tid, cancel_requested=True)
    marked = marked + pending
    for tid in marked:
        _set_job(tid, status="cancelling", cancel_reason=reason)
    return {
        "success": True,
        "cancelled": marked,
        "still_planning": _cancel.active(),
        "note": ("no planning job was running" if not marked else
                 "flagged; the run unwinds at its next checkpoint"),
    }


@mcp.tool()
def pause_planning(tree_id: Optional[str] = None,
                   reason: str = "paused by operator") -> dict:
    """
    Pause a planning run so it can be CONTINUED later, keeping all its work.

    Pause is not Stop. `cancel_planning` throws the run away — no trajectories, no
    conversation, and the next attempt starts at the first weld. Pause writes the
    run's place to disk (the model's conversation plus every joint trajectory
    already solved) and stops at the next clean step boundary. `resume_planning`
    then continues from there.

    Use it when the run is fine but its surroundings are not: an AWS token about to
    expire, a network link about to drop, a container that has to be recreated. A
    15-weld plan is minutes of LLM reasoning over a hundred motion solves; none of
    that has to be repeated because the credentials aged out.

    Pausing is COOPERATIVE and quiet — the planning tools see no exception. The run
    finishes the step it is on, checkpoints, and lands on `status == "paused"`
    (poll `get_plan_status(tree_id)`). At most that one in-flight step is replayed
    on resume, never the run.

    Args:
        tree_id: The run to pause. Omit to pause EVERY run currently planning.
        reason:  Recorded in the checkpoint and shown in the event feed.

    Returns:
        {"success": true, "paused": [tree_id, ...], "reason": str}
        ``paused`` is empty when nothing was planning — that is not an error.
    """
    from agents.trajectory import plan_checkpoint
    result = plan_checkpoint.request_pause(tree_id, reason=reason)
    for tid in result.get("paused") or []:
        _set_job(tid, status="pausing", pause_reason=reason)
    result["still_planning"] = _cancel.active()
    result["poll"] = ("get_plan_status(tree_id) until status=paused, then "
                      "resume_planning(tree_id) when you are ready")
    return result


@mcp.tool()
async def resume_planning(tree_id: str, cell_id: Optional[str] = None,
                          model: Optional[str] = None) -> dict:
    """
    Continue a paused or interrupted planning run from where it stopped.

    This is the recovery path for the failures that are not the cell's fault — an
    expired token, a dropped socket, a truncated reply, an exhausted step budget,
    an operator's Pause. The run keeps its conversation and every trajectory it had
    already solved, so it carries on at the next unplanned weld instead of the
    first one.

    Only runs that left a resumable checkpoint can be resumed; `list_checkpoints`
    shows which. A run that already produced a tree has none, by design — there is
    nothing left to do, and resuming into finished work is not a recovery.

    ASYNCHRONOUS, exactly like `plan_trajectories`: the request is accepted and the
    run continues in the background under the SAME tree_id, appending to the same
    event stream. Poll `get_plan_status(tree_id)`.

    Args:
        tree_id:  The run to continue.
        cell_id:  Optional. Found from the checkpoint if omitted, so a resume works
                  in a freshly recreated container that never saw the original job.
        model:    Optional different LLM for the remainder of the run.

    Returns:
        {"accepted": true, "tree_id": str, "status": "planning",
         "resumed_from_step": int, "trajectories_restored": int}
        or {"accepted": false, "error": ...} when there is nothing to resume.
    """
    from agents.trajectory import plan_checkpoint

    try:
        requested = _normalize_model(model)
    except ValueError as exc:
        return {"accepted": False, "error": str(exc),
                "hint": f"valid aliases: {', '.join(sorted(_MODEL_ALIASES))}"}

    if not plan_checkpoint.is_safe_id(tree_id):
        return {"accepted": False, "error": f"invalid tree_id: {tree_id!r}"}

    with _PLAN_JOBS_LOCK:
        job = dict(_PLAN_JOBS.get(tree_id) or {})
    if job.get("status") in ("planning", "pausing", "cancelling"):
        return {"accepted": False, "error": f"{tree_id} is already running",
                "status": job.get("status"),
                "note": "pause_planning(tree_id) first if you want it to stop"}

    resolved = cell_id or job.get("cell_id") or plan_checkpoint.find_cell(tree_id)
    if not resolved:
        return {"accepted": False, "error": f"no checkpoint found for {tree_id}",
                "hint": "list_checkpoints(cell_id) shows what can be resumed"}

    saved = plan_checkpoint.describe(resolved, tree_id)
    if not saved:
        return {"accepted": False, "error": f"no checkpoint for {tree_id} in {resolved}"}
    if not saved.get("resumable"):
        # Refuse rather than silently restart. "Resume" that quietly replans from
        # W01 is the exact surprise this whole mechanism exists to remove.
        return {"accepted": False, "error": f"the checkpoint for {tree_id} is not resumable",
                "checkpoint": saved,
                "note": ("Nothing usable was captured, so continuing is not possible. "
                         "Submit a new plan_trajectories run.")}

    # The checkpoint carries the operations, so the caller does not restate them —
    # and cannot accidentally resume a run against a different weld list.
    holder = _reserve(tree_id, kind="resume", cell_id=resolved, success=None,
                      error=None, note=None, resumed=True,
                      resumed_from_step=saved.get("step"),
                      model=_planning_model(requested or job.get("model")),
                      submitted_s=_time.time(), planner_id=_PLANNER_ID)
    if holder is not None:
        return _busy_refusal(holder)
    task = asyncio.create_task(
        # A resume continues on the model the run started on unless told otherwise.
        asyncio.to_thread(_run_plan_job, resolved, [], None, tree_id,
                          model or job.get("model"), True)
    )
    _PLAN_TASKS.add(task)
    task.add_done_callback(_PLAN_TASKS.discard)
    return {
        "accepted": True,
        "tree_id": tree_id,
        "cell_id": resolved,
        "status": "planning",
        "resumed_from_step": saved.get("step"),
        "trajectories_restored": saved.get("trajectories"),
        "operations": saved.get("operations"),
        "discarded_steps": saved.get("discarded"),
        "saved_at": saved.get("saved_at"),
        "poll": ("get_plan_status(tree_id); when status=done, "
                 "get_calculated_plan(cell_id, tree_id)"),
    }


@mcp.tool()
def list_checkpoints(cell_id: str) -> dict:
    """
    The planning runs that can be continued, newest first.

    A checkpoint exists only while a run has somewhere left to go: it is written
    when a run pauses or is interrupted, and deleted the moment the run produces a
    tree. So everything listed here with ``resumable: true`` is work that would
    otherwise have to be redone.

    Returns:
        {"cell_id": str, "count": int, "resumable": int, "checkpoints": [
            {"tree_id", "saved_at", "reason", "resumable", "step",
             "operations", "trajectories", "discarded", "backend", "model"}, ...]}
        ``trajectories`` is how many solved motions the resume would recover.
    """
    from agents.trajectory import plan_checkpoint
    items = plan_checkpoint.listing(cell_id)
    return {
        "cell_id": cell_id,
        "count": len(items),
        "resumable": sum(1 for i in items if i.get("resumable")),
        "checkpoints": items,
        "note": ("nothing to resume — every run either finished or left no "
                 "recoverable state" if not items else
                 "resume_planning(tree_id) continues one of these"),
    }


@mcp.tool()
def list_planning_jobs(cell_id: str, limit: int = 20) -> dict:
    """
    The planning runs of a cell, newest first — the ones in flight AND the ones a
    planner restart interrupted (each run has its own status file, so this
    survives restarts).

    Returns {"cell_id", "jobs": [{"tree_id", "status", "backend", "mode",
    "submitted_s", "resumable", "resume_with", ...}]}. ``status`` is
    "planning" | "pausing" | "cancelling" | "done" | "paused" | "failed" |
    "cancelled" | "interrupted". A run with ``resume_with`` set is continued by
    ``resume_planning(resume_with)``; one in flight is followed with
    ``get_plan_status(tree_id)``.
    """
    from agents.trajectory import plan_jobs
    with _PLAN_JOBS_LOCK:
        live = {tid: dict(j) for tid, j in _PLAN_JOBS.items() if j.get("cell_id") == cell_id}
    out = []
    for saved in plan_jobs.listing(cell_id, limit=limit):
        tid = saved.get("tree_id")
        job = live.pop(tid, None)
        rec = ({"tree_id": tid, **job} if job is not None
               else plan_jobs.as_seen_after_restart(saved))
        rec.pop("operation_ids", None)
        out.append(rec)
    for tid, job in live.items():                      # not persisted (yet): still live
        job.pop("operation_ids", None)
        out.append({"tree_id": tid, **job})
    out.sort(key=lambda r: r.get("submitted_s") or 0, reverse=True)
    return {"cell_id": cell_id, "jobs": out[:max(1, int(limit))]}


@mcp.tool()
def get_plan_events(cell_id: str, trace_id: Optional[str] = None,
                    since: int = 0, limit: int = 200) -> dict:
    """
    The live feed of a planning run: what it is doing right now, in order.

    Each event is one fact the planner knows — a weld started, a trajectory was
    computed, a validation passed or failed, the planner re-attempted an operation
    with a different standoff (a strategy change, with the changed values in the
    event), the run paused or finished. Reading them in order tells you what a run
    that takes minutes is spending them on, while it spends them.

    Poll it: pass the ``next_seq`` from the previous reply as ``since`` and you get
    only what has happened since, so the feed never re-sends the run.

    Args:
        cell_id:  Cell whose run to watch.
        trace_id: A specific run. Omit for the most recent one.
        since:    Return only events with ``seq`` greater than this. 0 = from the start.
        limit:    Maximum events to return; the NEWEST are kept when a poll falls
                  behind, so a slow reader sees the present, not a stale window.

    Returns:
        {"trace_id": str|null, "events": [{"seq","at","kind","text",...}],
         "next_seq": int, "dropped": int, "running": bool}
        ``running`` is false once a terminal event (finished/aborted/paused) is in
        the stream. ``events`` empty with the same ``next_seq`` means "nothing new".
    """
    from agents.trajectory import plan_events
    try:
        window = max(1, min(int(limit), 1000))
    except (TypeError, ValueError):
        window = 200
    try:
        start = max(0, int(since))
    except (TypeError, ValueError):
        start = 0
    return plan_events.read(cell_id, trace_id, since=start, limit=window)


@mcp.tool()
def list_models() -> dict:
    """
    The models this agent can plan with, and the one it uses by default.

    Pass any alias, full model id, or "provider:model" as the `model` argument of
    `plan_trajectories` to override the default for a single plan.
    """
    return _describe_models()


# ─── Robot descriptions ───────────────────────────────────────────────────────

@mcp.tool()
def get_robot_description(cell_id: str, robot_id: str) -> dict:
    """
    The robot description (URDF text) of one arm — the same document this agent
    plans with, so a viewer cannot show an arm the planner is not planning for.

    The RViz viewer fetches it here instead of listening for a DDS topic. Only the
    URDF TEXT travels; the meshes it references stay on the read-only mount.

    Args:
        cell_id:  Cell identifier
        robot_id: The arm, as named in robots.yaml (e.g. "robot1")

    Returns {"success", "found", "robot_id", "urdf", "urdf_path"}.
    """
    from utils import cell_layout
    try:
        path = cell_layout.robot_urdf(cell_id, robot_id)
        if path is None:
            return {"success": True, "found": False, "cell_id": cell_id,
                    "robot_id": robot_id, "urdf": None}
        return {"success": True, "found": True, "cell_id": cell_id,
                "robot_id": robot_id, "urdf": path.read_text(), "urdf_path": str(path)}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


# ─── Skills ───────────────────────────────────────────────────────────────────
# utils/skill_registry.py owns discovery and prioritisation; these tools are its
# MCP face, and the orchestrator's /a2a/skills routes proxy them.

@mcp.tool()
def list_skills(cell_id: Optional[str] = None, agent: Optional[str] = None,
                include_disabled: bool = True) -> dict:
    """
    The skills each agent can follow, with their EFFECTIVE priority for a cell.

    Args:
        cell_id:          Apply this cell's `skills.yaml` overrides. Omit for the
                          SKILL.md frontmatter defaults.
        agent:            "trajectory" | "trajectory-classic" | "cell" |
                          "orchestrator". Omit for all.
        include_disabled: Also list skills the cell disabled (flagged
                          `enabled: false`).

    Returns {"success", "skills": [{agent, name, description, priority,
    default_priority, enabled, required, overridden, category, version, path}]},
    highest priority first within each agent. Only the trajectory agent loads
    skills at run time; cell/orchestrator skills are reference procedures.
    """
    from utils import skill_registry
    try:
        skills = skill_registry.list_skills(cell_id, agent, include_disabled)
    except skill_registry.SkillError as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, "cell_id": cell_id, "count": len(skills),
            "skills": skills}


@mcp.tool()
def get_skill(agent: str, name: str, cell_id: Optional[str] = None) -> dict:
    """
    One skill's metadata and full SKILL.md text (frontmatter + instructions).

    Args:
        agent:   "trajectory" | "cell" | "orchestrator"
        name:    Skill name, e.g. "optimize-cycle-time"
        cell_id: Report the priority/enabled state for this cell.
    """
    from utils import skill_registry
    try:
        return {"success": True, **skill_registry.get_skill(agent, name, cell_id)}
    except skill_registry.SkillError as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def skill_template(agent: str = "trajectory") -> dict:
    """
    What a skill looks like and how to bring one to a single plan: a complete
    example SKILL.md, its fields and limits, the names already taken, and
    example plan requests (temporary skill, disabling / re-prioritising skills,
    prompt instructions).

    Args:
        agent: The skill set: "trajectory" (stepwise / coordinated modes) or
               "trajectory-classic" (classic mode).
    """
    from utils import skill_registry
    try:
        return {"success": True, **skill_registry.skill_template(agent)}
    except skill_registry.SkillError as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def validate_run_skills(run_skills: list, agent: str = "trajectory") -> dict:
    """
    Check temporary skills the way plan_trajectories(run_skills=...) will,
    without planning: returns their canonical form, or the reason one would be
    refused (bad/taken name, missing description or instructions, too long,
    `required`, too many).
    """
    from utils import skill_registry
    try:
        skills = skill_registry.normalise_run_skills(agent, run_skills)
    except skill_registry.SkillError as exc:
        return {"success": False, "valid": False, "error": str(exc)}
    return {"success": True, "valid": True, "agent": agent,
            "skills": [{k: v for k, v in s.items() if k != "content"}
                       | {"content_chars": len(s["content"])} for s in skills]}


@mcp.tool()
def set_skill_priority(cell_id: str, agent: str, name: str,
                       priority: Optional[int] = None,
                       enabled: Optional[bool] = None,
                       reset: bool = False) -> dict:
    """
    Set a cell's priority (0-100, higher = preferred) and/or enabled flag for one
    skill, stored in the cell's `config/skills.yaml`. `reset: true` drops the
    override so the SKILL.md default applies again.

    Takes effect at the NEXT planning run: the catalog is ordered when the
    planner's prompt is built. Skills marked `required` (collision validation)
    cannot be disabled.
    """
    from utils import skill_registry
    if priority is None and enabled is None and not reset:
        return {"success": False, "error": "give priority, enabled, or reset"}
    try:
        skill = skill_registry.set_skill_priority(
            cell_id, agent, name, priority=priority, enabled=enabled, reset=reset)
    except skill_registry.SkillError as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, "cell_id": cell_id, "skill": skill}


@mcp.tool()
def get_plan_status(tree_id: str) -> dict:
    """
    Status of an asynchronous plan_trajectories job, by its tree_id.

    Args:
        tree_id: The id returned by plan_trajectories.

    Returns:
        {"tree_id": str, "status": "planning"|"done"|"failed"|"unknown",
         "success": bool|None, "error": str|None, "elapsed_s": float,
         "num_operations": int,
         "progress": {"phase", "percent", "operations_solved",
                      "operations_total", "attempts", "failed_attempts",
                      "last_event", "last_event_at", "approximate"}}
        When status == "done", fetch the tree with
        get_calculated_plan(cell_id, tree_id).

        ``progress.percent`` is work done on the program (0-100), read off the
        run's event stream — see plan_events.progress. It is not a time estimate.
    """
    with _PLAN_JOBS_LOCK:
        job = _PLAN_JOBS.get(tree_id)
        job = dict(job) if job else None
    if job is None:
        # Not a run of THIS process: a run of one that restarted, if its status
        # file is there — reported as it stands now (interrupted, resumable?).
        from agents.trajectory import plan_jobs
        saved = plan_jobs.load(tree_id)
        if not saved:
            return {"tree_id": tree_id, "status": "unknown",
                    "error": "no such planning job"}
        job = plan_jobs.as_seen_after_restart(saved)
        job.pop("tree_id", None)
    from agents.trajectory import plan_events
    ids = job.pop("operation_ids", None)
    try:
        job["progress"] = plan_events.progress(job.get("cell_id"), tree_id, ids)
    except Exception as exc:  # noqa: BLE001 — a status answer never fails on this
        job["progress"] = {"error": f"progress unavailable: {exc}"}
    return {"tree_id": tree_id, **job}


@mcp.tool()
def list_calculated_plans(cell_id: str) -> dict:
    """
    List behaviour trees previously calculated by plan_trajectories for a cell.

    Every successful plan_trajectories run persists its assembled BehaviourTree
    (with all precomputed joint trajectories) under the cell's shared plans/
    directory. This tool lists them, newest first, so a caller can pick one to
    fetch with get_calculated_plan and hand to the cell's execute_behaviour_tree
    WITHOUT re-planning.

    Args:
        cell_id: Cell identifier, e.g. "welding_cell_2_small"

    FAILED runs are listed too, marked ``status: "failed"`` — a run that solved
    thirteen of fifteen welds before its token expired is worth reading, and worth
    resuming. They are NOT executable: get_calculated_plan will not return one
    without ``include_failed``, and "the latest plan" never resolves to one.

    Args:
        cell_id: Cell identifier, e.g. "welding_cell_2_small"

    Returns:
        {"cell_id": str, "count": int, "ok": int, "failed": int,
         "plans": [{"tree_id": str, "status": "ok"|"failed", "created_s": float,
                    "num_operations": int, "num_planned": int|None,
                    "error": str|None, "resumable": bool,
                    "size_bytes": int}, ...]}
    """
    from agents.trajectory.tools.plan_store import list_plans, STATUS_FAILED
    plans = list_plans(cell_id)
    # Drop the on-disk path from the client-facing view.
    for p in plans:
        p.pop("path", None)
    failed = sum(1 for p in plans if p.get("status") == STATUS_FAILED)
    return {"cell_id": cell_id, "count": len(plans),
            "ok": len(plans) - failed, "failed": failed, "plans": plans}


@mcp.tool()
def get_calculated_plan(cell_id: str, tree_id: Optional[str] = None,
                        include_failed: bool = False) -> dict:
    """
    Retrieve a previously calculated BehaviourTree (with its trajectories).

    Returns the full BehaviorTree.CPP v4 tree the planner produced. The
    ``ExecuteTrajectory`` leaves reference their waypoint arrays through the
    ``trajectory`` port as ``"{key}"`` into the tree's ``blackboard``, so the
    result is self-contained and can be passed straight to the cell MCP
    ``execute_behaviour_tree`` tool to replay the motion without re-planning.

    Args:
        cell_id: Cell identifier, e.g. "welding_cell_2_small"
        tree_id: Specific plan id (from list_calculated_plans). If omitted, the
                 most recently calculated plan for the cell is returned —
                 successful plans only, never a failed run.
        include_failed: Return the partial tree of a FAILED run. For inspection
                 only. A failed run did not assemble a plan the assembler would
                 accept, so what comes back is evidence, not something to execute.

    Returns:
        {"success": true, "tree_id": str, "num_operations": int,
         "status": "ok"|"failed", "executable": bool, "error": str|None,
         "error_kind": str|None, "resumable": bool,
         "behaviour_tree": {tree_id, cell_id, BTCPP_format, main_tree_to_execute,
                            root, blackboard, xml, metadata}}
        or {"success": false, "error": str}

    ``executable`` is false exactly when the tree came from a failed run via
    ``include_failed``. It is the caller's cue that this is evidence to read, not
    a plan to run.
    """
    from agents.trajectory.tools.plan_store import (STATUS_FAILED, get_failed_plan,
                                                    get_plan)
    bt = get_plan(cell_id, tree_id, include_failed=include_failed)
    if not bt:
        which = tree_id or "latest"
        # Distinguish "there is no such plan" from "there is, and it failed".
        # The second is a different instruction to the caller: read it, or resume
        # it — not plan again from the beginning.
        record = get_failed_plan(cell_id, tree_id) if tree_id else None
        if record:
            return {
                "success": False, "status": "failed", "tree_id": tree_id,
                "error": (f"plan '{tree_id}' is a FAILED run and is not "
                          "executable: " + str(record.get("error") or "no tree")),
                "error_kind": record.get("error_kind"),
                "num_planned": record.get("num_planned"),
                "planned": record.get("planned"),
                "failures": record.get("failures"),
                "resumable": bool(record.get("resumable")),
                "note": ("Its computed trajectories are kept. Pass "
                         "include_failed=true to inspect the partial tree, or "
                         "resume_planning(tree_id) to continue the run."),
            }
        return {"success": False,
                "error": f"no calculated plan found for cell '{cell_id}' ({which})"}

    def _count_exec(node) -> int:
        from agents.trajectory.tools.btcpp import node_kind
        if not isinstance(node, dict):
            return 0
        n = 1 if node_kind(node) == "ExecuteTrajectory" else 0
        for c in node.get("children", []) or []:
            n += _count_exec(c)
        return n

    # A tree returned under include_failed is a partial artefact from a run that
    # failed, and it looks exactly like a good one. Say which it is here, so a
    # caller that opted into reading failures cannot mistake the evidence for a
    # plan — the caller asked to *inspect*, and inspection is not permission to run.
    record = get_failed_plan(cell_id, bt.get("tree_id") or tree_id or "")
    return {
        "success":        True,
        "tree_id":        bt.get("tree_id"),
        "num_operations": _count_exec(bt.get("root")),
        "status":         STATUS_FAILED if record else "ok",
        "executable":     not record,
        "error":          (record or {}).get("error"),
        "error_kind":     (record or {}).get("error_kind"),
        "resumable":      bool((record or {}).get("resumable")),
        "behaviour_tree": bt,
    }


@mcp.tool()
def get_plan_xml(cell_id: str, tree_id: Optional[str] = None,
                 include_model: bool = True) -> dict:
    """
    Export a calculated plan as BehaviorTree.CPP v4 XML (loadable in Groot2).

    Serialises the stored tree to the standard ``<root BTCPP_format="4">`` form.
    The bulky waypoint arrays are NOT inlined: each ``ExecuteTrajectory`` keeps a
    ``trajectory="{key}"`` blackboard reference, and the referenced values are
    returned separately in ``blackboard`` so the XML stays human-readable.

    Args:
        cell_id:       Cell identifier, e.g. "welding_cell_2_small"
        tree_id:       Specific plan id (from list_calculated_plans), or None = latest.
        include_model: Emit a <TreeNodesModel> block describing the custom nodes
                       (Groot2 needs it to draw them). Default True.

    Returns:
        {"success": true, "tree_id": str, "xml": str, "blackboard": {...}}
        or {"success": false, "error": str}
    """
    from agents.trajectory.tools.plan_store import get_plan
    from agents.trajectory.tools.behaviour_tree import export_behaviour_tree_xml

    bt = get_plan(cell_id, tree_id)
    if not bt:
        which = tree_id or "latest"
        return {"success": False,
                "error": f"no calculated plan found for cell '{cell_id}' ({which})"}
    # Reuse the @tool so a plan saved in the legacy dialect is converted on the way
    # out rather than serialised as-is.
    result = export_behaviour_tree_xml.func(tree=bt, include_model=include_model)
    if result.get("success"):
        result["tree_id"] = bt.get("tree_id")
    return result


@mcp.tool()
def check_reachability(
    cell_id: str,
    robot_ns: str,
    x: float,
    y: float,
    z: float,
) -> dict:
    """
    Check if a Cartesian target is reachable by a robot via Pinocchio IK.

    Uses the robot's URDF from cells/<cell_id>/urdf/<robot_ns>/ with joint
    limits from robots.yaml. Tries 4 IK seeds with floor-safety check.

    Args:
        cell_id:  Cell identifier
        robot_ns: Robot namespace, e.g. "robot1"
        x, y, z:  Target position in world frame (metres)

    Returns:
        {"reachable": bool, "ik_solution": [j1..j6]}
        or {"reachable": false, "distance_from_base": float, "max_reach": float}
    """
    # _run_ik is a LangChain @tool (StructuredTool); call its underlying function.
    return _run_ik.func(cell_id=cell_id, robot_ns=robot_ns, x=x, y=y, z=z)


@mcp.tool()
def validate_trajectory_pair(
    cell_id: str,
    robot1_ns: str,
    trajectory1: list[dict],
    robot2_ns: str,
    trajectory2: list[dict],
) -> dict:
    """
    Validate two robot trajectories for inter-robot collision.

    Samples both trajectories at 20 time steps and checks sphere-sphere
    clearance (min 0.24m = 2 × 0.12m collision radius per link).

    Args:
        cell_id:       Cell identifier
        robot1_ns:     First robot namespace
        trajectory1:   List of {"positions": [j1..j6], "time_s": float}
        robot2_ns:     Second robot namespace
        trajectory2:   Same format as trajectory1

    Returns:
        {"collision_free": bool, "min_clearance_m": float}
        or {"collision_free": false, "collision_time_s": float,
            "min_clearance_m": float, "suggestion": str}
    """
    from agents.trajectory.tools.planning_tools import validate_trajectory_pair as _validate
    return _validate.func(
        cell_id=cell_id,
        robot1_ns=robot1_ns, trajectory1=trajectory1,
        robot2_ns=robot2_ns, trajectory2=trajectory2,
    )


@mcp.tool()
def validate_trajectory_set(
    cell_id: str,
    trajectories: dict,
    n_samples: int = 20,
    start_delays: Optional[dict] = None,
) -> dict:
    """
    Validate ANY number of robot trajectories for inter-robot collision.

    Use this for a PARALLEL step with two OR MORE arms. Checks every unordered
    robot pair at n_samples instants on a shared wall-clock timeline, so arms with
    different durations are compared at the same real time.

    Args:
        cell_id:       Cell identifier
        trajectories:  {robot_ns: [{"positions": [j1..j6], "time_s": float}, ...]}
        n_samples:     Number of time samples to check (default 20)
        start_delays:  {robot_ns: seconds} — check the staggered timeline. A
                       colliding set returns `delay_fix`, the smallest delays
                       that clear it.

    Returns:
        {"collision_free": bool, "min_clearance_m": float, "robots": [...],
         "pairs_checked": int, "worst_pair": [nsA, nsB],
         "collision_time_s": float?, "colliding_pairs": [...]?, "suggestion": str?}
    """
    from agents.trajectory.tools.planning_tools import validate_trajectory_set as _validate_set
    return _validate_set.func(
        cell_id=cell_id, trajectories=trajectories, n_samples=n_samples,
        start_delays=start_delays,
    )


@mcp.tool()
def find_interlock_set(
    cell_id: str,
    targets: list[dict],
    operation_type: str = "weld",
) -> dict:
    """
    Find safe staging positions enabling parallel motion for ANY number of robots.

    Use when validate_trajectory_set reports collision among 2+ arms, or when any
    pair of targets is closer than 0.4m. Accepts a strategy only when EVERY robot
    pair clears the margin — resolving two pairs of a 3-way convergence does not
    make the third pair safe.

    Args:
        cell_id:        Cell identifier
        targets:        [{"robot_ns": str, "x": float, "y": float, "z": float}, ...]
        operation_type: Affects the minimum clearance margin

    Returns:
        {"needs_interlock": bool, "strategy": str, "robots": [...],
         "min_clearance_m": float?, "interlocks": {ns: {...}}?, "execution_plan": [...]}
    """
    from agents.trajectory.tools.planning_tools import find_interlock_set as _find_set
    return _find_set.func(
        cell_id=cell_id, targets=targets, operation_type=operation_type,
    )


@mcp.tool()
def find_interlocks(
    cell_id: str,
    robot1_ns: str,
    target1_x: float,
    target1_y: float,
    target1_z: float,
    robot2_ns: str,
    target2_x: float,
    target2_y: float,
    target2_z: float,
    operation_type: str = "weld",
) -> dict:
    """
    Find safe intermediate positions enabling parallel robot motion without collision.

    Use when validate_trajectory_pair returns collision, or when targets are
    closer than 0.4m apart.

    Tries interlock strategies in priority order:
      raise_arm → rotate_outward → partial_retract → approach_from_above → home

    Args:
        cell_id:        Cell identifier
        robot1_ns:      First robot namespace
        target1_x/y/z:  First robot's final target (world frame, metres)
        robot2_ns:      Second robot namespace
        target2_x/y/z:  Second robot's final target
        operation_type: "weld" | "rivet" | "laser_cut" etc. (affects clearance margin)

    Returns:
        {"needs_interlock": false, "strategy": "direct_parallel", ...}
        or {"needs_interlock": true, "strategy": str,
            "robot1_interlock": {"ee_position": [...], "joint_positions": [...]},
            "robot2_interlock": {...},
            "execution_plan": [...]}
        or {"needs_interlock": true, "strategy": "sequential_fallback", ...}
    """
    from agents.trajectory.tools.planning_tools import find_interlocks as _find
    return _find.func(
        cell_id=cell_id,
        robot1_ns=robot1_ns, x1=target1_x, y1=target1_y, z1=target1_z,
        robot2_ns=robot2_ns, x2=target2_x, y2=target2_y, z2=target2_z,
        operation_type=operation_type,
    )


@mcp.tool()
def get_trajectory_knowledge(
    cell_id: str,
    operation_type: str,
    robot_ns: str,
    target: list[float],
) -> dict:
    """
    Get warm-start joint configuration from the trajectory knowledge base.

    Looks up a successful past trajectory for a similar target in the same
    cell to seed Pinocchio IK (reduces convergence from ~150 to ~20 iterations).

    Args:
        cell_id:        Cell identifier
        operation_type: "weld" | "rivet" | etc.
        robot_ns:       Robot namespace
        target:         [x, y, z] target position

    Returns:
        {"found": true, "joint_positions": [j1..j6], "position_error_m": float}
        or {"found": false}
    """
    return _get_warm_start.func(
        cell_id=cell_id,
        operation_type=operation_type,
        robot_ns=robot_ns,
        target=target,
    )


def _task_program_summary(cell_id: str) -> tuple[list, dict]:
    """The cell's task programs and their aggregate operation counts.

    Asked of the orchestrator, not of a mounted ``tasks/`` directory: the cell's
    task catalogue is cell information, and this planner is no longer one of its
    writers. Two callers share this because ``get_cell_info`` and
    ``list_task_programs`` differed only in which half of the answer they threw
    away — and drifted in how they counted.

    A program that cannot be parsed is skipped rather than fatal (a half-written
    catalogue should still list), but an unreachable store raises: reporting zero
    operations for a cell with forty is how an operator concludes the cell is
    empty.
    """
    entries = cell_store.list_entries(cell_id, cell_store.KIND_TASKS,
                                      suffix=".json")
    programs: list = []
    op_types: dict = {}
    total = 0
    for entry in entries:
        name = entry.get("name", "")
        programs.append(name[: -len(".json")] if name.endswith(".json") else name)
        try:
            data = cell_store.read_json(cell_id, cell_store.KIND_TASKS, name)
        except cell_store.CellStoreUnavailable:
            raise
        except Exception:  # noqa: BLE001
            continue
        for op in (data or {}).get("operations", []):
            kind = op.get("type", "weld")
            op_types[kind] = op_types.get(kind, 0) + 1
            total += 1
    return programs, {"total": total, "by_type": op_types}


@mcp.tool()
def get_cell_info(cell_id: str) -> dict:
    """
    Get cell configuration: robots, workpiece, operation points.

    Args:
        cell_id: Cell identifier

    Returns:
        {"cell_id": str, "num_robots": int, "robots": {...},
         "num_operations": int, "workpiece_bounds": [...]}

    ``robots`` is keyed by robot identity (robot1, robot2, …) — the same key
    ``check_reachability``/``find_interlocks`` take as ``robot_ns`` — so cells
    that share one ROS namespace ({arms: {robot1: .., robot2: ..}}) report every
    arm, not just the first.
    """
    from agents.cell.robots_util import iter_robots

    robots_info = {}
    for ns, rid, rcfg in iter_robots(cell_id):
        robots_info[rid] = {
            "namespace":        ns,
            "base_position":    rcfg.get("base_position"),
            # The yaw travels with the position because the two are one pose:
            # a consumer placing the arm in the world (forward kinematics off
            # joint states, for one) that got the translation but not the
            # rotation would report end-effector positions rotated about the
            # base — plausible numbers, wrong point, no error anywhere.
            "base_orientation_yaw": float(rcfg.get("base_orientation_yaw", 0.0)),
            "workspace_radius": rcfg.get("workspace_radius", 2.1),
            "end_effector":     rcfg.get("end_effector_link", "tool0"),
        }

    _programs, ops_info = _task_program_summary(cell_id)

    return {
        "cell_id":      cell_id,
        "num_robots":   len(robots_info),
        "robots":       robots_info,
        "operations":   ops_info,
    }


@mcp.tool()
def list_task_programs(cell_id: str) -> dict:
    """
    List the task programs defined for a cell (names + operation summary).

    A task program is a JSON file in the cell's tasks/ directory; its base name
    (without .json) is the ``program`` you pass to ``get_task_program`` /
    ``plan_trajectories``. Mirrors the orchestrator's ``GET /a2a/tasks`` so a
    native MCP client can browse the catalogue without the A2A dashboard.

    Returns:
        {"cell_id", "programs": [name, ...], "count",
         "operations": {"total": int, "by_type": {...}}}
    """
    programs, ops_info = _task_program_summary(cell_id)
    return {
        "cell_id":    cell_id,
        "programs":   programs,
        "count":      len(programs),
        "operations": ops_info,
    }


@mcp.tool()
def get_task_program(cell_id: str, program: str) -> dict:
    """
    Return one task program's full JSON.

    Its ``operations`` array is PLAN-READY — pass it straight to
    ``plan_trajectories`` (the orchestrator's ``/a2a/plan`` does exactly that, with
    no transform). Reads the cell's ``tasks/<program>.json``.

    Args:
        cell_id: Cell identifier
        program: Program name WITHOUT the .json extension
    """
    from agents.trajectory import plan_trace

    if not program or not plan_trace.is_safe_id(program):
        return {"success": False, "cell_id": cell_id,
                "error": "program must be a plain identifier (letters, digits, . _ -)"}
    try:
        data = cell_store.read_json(cell_id, cell_store.KIND_TASKS,
                                    f"{program}.json")
    except Exception as exc:  # noqa: BLE001
        # Unreachable is not "no such program": the caller would go and write one.
        return {"success": False, "cell_id": cell_id, "program": program,
                "error": f"could not read task program: {exc}"}
    if data is None:
        return {"success": False, "cell_id": cell_id, "program": program,
                "error": f"no task program '{program}' in cell '{cell_id}'"}
    from agents.trajectory.task_programs import program_warnings
    out = {"success": True, "cell_id": cell_id, "program": program, "task": data}
    warnings = program_warnings(cell_id, data)
    if warnings:
        out["warnings"] = warnings
    return out


@mcp.tool()
def save_task_program(cell_id: str, program: str, task: dict,
                      overwrite: bool = False) -> dict:
    """
    Write a task program to the cell's ``tasks/<program>.json``. Runs NOTHING.

    The write half of ``get_task_program``, and the only sanctioned way for a
    caller that is not the trajectory agent to author a program: the alternative
    every caller reached for otherwise was opening the shared volume itself, and
    two writers with no agreement on the schema is how the panel came to append
    ``tasks[].parameters.points`` to a format whose readers had long since moved
    to ``operations[]``.

    Args:
        cell_id:   Cell identifier.
        program:   Program name WITHOUT .json. Becomes the file name, so it is
                   restricted to letters, digits, '.', '_' and '-'.
        task:      The program, per skills/orchestrator/run-welding-program/
                   references/task-json-format.md. ``operations`` must be a
                   non-empty array; ``program`` and ``cell_id`` are set here to
                   match where the file lands, so a copied program cannot claim
                   to belong to another cell.
        overwrite: Required to replace an existing program. Without it an
                   existing name is REFUSED rather than replaced — a silent
                   overwrite destroys the only copy of a hand-authored program.

    Returns:
        {"success", "cell_id", "program", "path", "num_operations",
         "overwritten", "commanded": false}
        Planning and execution stay separate, explicit calls
        (``plan_trajectories``); saving a program moves nothing.
    """
    # The writing itself lives in agents.trajectory.task_programs, which imports
    # nothing from the LLM stack and can therefore be tested directly.
    from agents.trajectory import task_programs

    return task_programs.save_program(cell_id, program, task, overwrite)


@mcp.tool()
def list_plan_traces(cell_id: str, limit: int = 50) -> dict:
    """
    History of recorded planning-decision traces for a cell, newest first
    (metadata only). Each entry's ``trace_id`` feeds ``get_plan_trace``. Mirrors
    ``GET /a2a/plan_traces``.
    """
    from agents.trajectory import plan_trace

    limit = max(1, min(int(limit), 500))
    return {"cell_id": cell_id, "traces": plan_trace.listing(cell_id, limit=limit)}


@mcp.tool()
def get_plan_trace(cell_id: str, trace_id: str = "latest") -> dict:
    """
    One planning run's decisions, indexed per weld point.

    ``trace_id`` may be ``"latest"`` for the most recent recorded run. Returns
    ``points`` (each operation + the decisions that touched it and its verdict) and
    ``decisions`` (every tool call in order, with the reasoning that preceded it and
    the result that followed). Mirrors ``GET /a2a/plan_trace/<id>``.
    """
    from agents.trajectory import plan_trace

    wanted = trace_id
    if trace_id in ("latest", ""):
        history = plan_trace.listing(cell_id, limit=1)
        if not history:
            return {"success": False, "cell_id": cell_id, "trace_id": None,
                    "error": f"no planning traces recorded for '{cell_id}' yet"}
        wanted = history[0].get("trace_id")
    if not plan_trace.is_safe_id(wanted or ""):
        return {"success": False, "cell_id": cell_id,
                "error": "trace_id must be a plain identifier"}
    trace = plan_trace.read(cell_id, wanted)
    if trace is None:
        return {"success": False, "cell_id": cell_id, "trace_id": wanted,
                "error": f"no planning trace '{wanted}'"}
    return {"success": True, "cell_id": cell_id, **trace}


@mcp.tool()
def get_plan_quality(cell_id: str, tree_id: Optional[str] = None) -> dict:
    """
    How GOOD a calculated plan is — the trajectory-quality KPIs for one tree.

    A plan that executes is not the same as a plan worth executing: every motion
    here is sampled then shortcut, and the failure mode is silent — the path is
    valid, collision-free and four times longer than it needs to be. This is the
    report that says so, computed by ``tools/trajectory_quality.py`` (which owns
    the metric definitions and why each one was chosen).

    ``save_plan`` scores every plan as it stores it, so the normal answer is the
    stored report read back through the cell's owner (``source: "stored"``). A
    plan older than the metrics has none, so it is scored on demand from the
    stored tree instead (``source: "computed"``) — the same answer, just paid for
    now rather than at save time.

    Args:
        cell_id: Cell identifier, e.g. "welding_cell_2_small"
        tree_id: Plan to report on (from list_calculated_plans). Omitted or
                 "latest" resolves to the newest SUCCESSFUL plan.

    Returns:
        {"success": true, "cell_id": str, "tree_id": str,
         "source": "stored"|"computed",
         "efficiency_score": float, "verdict": "good"|"acceptable"|"poor",
         "score_components": {path, schedule, smoothness, dwell},
         "score_weights": {...},
         "summary": {makespan_s, motion_legs, joint_travel_rad,
                     direct_distance_rad, total_excess_travel_rad,
                     worst_detour_ratio, mean_detour_ratio,
                     total_interior_dwell_s, parallel_efficiency,
                     serial_fraction, load_balance},
         "schedule": {busy_s, idle_s, utilisation, unpaired_legs, ...},
         "continuity": {...}, "legs": [per-leg metrics], "safety": {...},
         "findings": [{severity, kind, robot_ns, node, detail}, ...]}
        or {"success": false, "error": str}

    ``findings`` is the ranked, actionable part — worst first, each naming the leg
    and by how much. ``safety`` is reported and deliberately excluded from every
    score: a score is a thing people optimise, and scoring clearance would turn a
    safety margin into a tunable.

    A FAILED run is refused rather than scored: its partial trajectories are worth
    reading, but scoring a fragment against a whole program compares two different
    things and the number would be quoted anyway.
    """
    from agents.trajectory.tools import trajectory_quality

    try:
        return trajectory_quality.report_for(cell_id, tree_id)
    except cell_store.CellStoreUnavailable as exc:
        # "The owner is unreachable" is not "this plan scores badly". Said as a
        # refusal so the panel shows a broken link rather than a bad plan. Caught
        # before CellStoreError below, which it subclasses.
        return {"success": False, "cell_id": cell_id, "tree_id": tree_id,
                "error": f"cell data unavailable: {exc}"}
    except cell_store.CellStoreError as exc:
        # A tree_id that is not a plain artefact name never reaches a path: the
        # store refuses it, and so does this.
        return {"success": False, "cell_id": cell_id, "tree_id": tree_id,
                "error": str(exc)}


@mcp.tool()
def suggest_plan_improvements(cell_id: str, tree_id: Optional[str] = None,
                              program: Optional[str] = None) -> dict:
    """
    What is wrong with a calculated plan, and what to try next. Changes NOTHING.

    Combines the plan's quality report (get_plan_quality), the planning run's
    trace (get_plan_trace — failed attempts, hot-spot operations, where the time
    went) and, when ``program`` is named, the program's cycle-time target, into
    one ranked list of problems. Each problem has the evidence and a concrete
    suggestion. Logic in ``agents/trajectory/plan_advice.py``.

    Args:
        cell_id: Cell identifier
        tree_id: Plan to examine. Omitted or "latest" = newest successful plan.
        program: Optional task program name, to compare the makespan with its
                 ``extensions.cycle_time`` target.

    Returns:
        {"success": true, "tree_id", "verdict", "efficiency_score",
         "summary": str,                       # one-line generic summary
         "problems": [{severity, source: "quality"|"planning"|"program", kind,
                       where, evidence, suggestion}, ...],   # worst first
         "time_breakdown": {elapsed_s, motion_planner_s, other_s, tool_calls,
                            step_budget, calls_by_tool} | null,
         "commanded": false}
        or {"success": false, "error": str}
    """
    from agents.trajectory import plan_advice

    try:
        return plan_advice.advice_for(cell_id, tree_id, program)
    except cell_store.CellStoreUnavailable as exc:
        return {"success": False, "cell_id": cell_id, "tree_id": tree_id,
                "error": f"cell data unavailable: {exc}"}
    except cell_store.CellStoreError as exc:
        return {"success": False, "cell_id": cell_id, "tree_id": tree_id,
                "error": str(exc)}


@mcp.tool()
async def replan_operation(
    cell_id: str,
    tree_id: str,
    operation_id: str,
    adjustments: Optional[dict] = None,
    operation: Optional[dict] = None,
    save: bool = True,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    backend_options: Optional[dict] = None,
) -> dict:
    """
    Re-plan ONE operation of an existing BehaviourTree and build a new tree —
    without re-planning the rest and without the LLM.

    ``backend`` decides how the legs are calculated, ``mode`` how the new tree
    is built — both like plan_trajectories (request, else the stack default,
    else ompl / classic):

    * ``mode`` "classic" | "stepwise" → steps (assemble_behaviour_tree); the
      source tree must be step-shaped. "coordinated" → one gated Sequence per arm
      (assemble_coordinated_tree), from a step-shaped or a coordinated source.
    * ``backend`` "ompl-compound" re-plans the operation's whole PARALLEL step in
      one compound query — every arm of that step, from where each arm's previous
      leg ends — so the step stays synchronised (and lockstep-tagged); later steps
      are re-planned the same way until every arm's chain closes.

    The operation is planned again with the motion request the original run used
    (from its plan trace; the task file is the fallback), with ``adjustments``
    applied, chained from where its arm's previous leg ends. If the new leg ends
    elsewhere, that arm's following legs are re-planned in turn until the chain
    closes. The edited plan then goes through the same assembler as every tree —
    inter-arm gate on every step, chain check, planned return home — and is saved
    as a NEW tree; the original is not modified.

    Args:
        cell_id:      Cell identifier.
        tree_id:      The plan to edit (as listed by list_calculated_plans).
        operation_id: The operation to re-plan, e.g. "W05".
        adjustments:  Fields of its motion request to change, any of:
                      target {x,y,z}, approach_vector [ax,ay,az], orientation
                      {x,y,z,w}, constrain_orientation, approach,
                      approach_offset_m, retreat_offset_m,
                      approach_velocity_scale, velocity_scale.
        operation:    The full motion request, when the run left no trace of it
                      and no task file defines the operation.
        save:         Save the new tree (default true).
        backend_options: Options for the backend, as in plan_trajectories (e.g.
                      {"policy": "<name>"} with backend="roboballet").

    Returns {"success", "tree_id" (the NEW tree), "derived_from", "robot_ns",
    "backend", "mode", "selected_by",
    "replanned": [{operation_id, robot_ns, waypoints, duration_s, end_shift_rad}],
    "step_gates", "serialised_steps", "planning_time_s"} — or {"success": false,
    "error", ...} with the planner's failures or the assembler's blocked steps.
    Refused while a plan_trajectories job is running (they share the planner's
    trajectory cache).
    """
    with _PLAN_JOBS_LOCK:
        busy = [tid for tid, job in _PLAN_JOBS.items() if job.get("status") == "planning"]
    if busy:
        return {"success": False, "error_kind": "planner_busy",
                "error": f"a planning job is running ({busy[0]}); "
                         "edit the plan once it has finished, or on another planner"}
    from agents.trajectory import plan_selection
    from agents.trajectory.tools import plan_edit
    # A re-plan is planning work too: it holds this planner for its duration, and
    # is refused while another run holds it (they share the trajectory cache).
    edit_id = f"replan_{tree_id}_{operation_id}_{int(_time.time())}"
    holder = _reserve(edit_id, kind="replan", cell_id=cell_id, derived_from=tree_id,
                      operation=operation_id, submitted_s=_time.time(),
                      planner_id=_PLANNER_ID)
    if holder is not None:
        return {**_busy_refusal(holder),
                "error": f"a planning job is running ({holder['tree_id']}); "
                         "edit the plan once it has finished, or on another planner"}
    try:
        return await _replan_operation(cell_id, tree_id, operation_id, adjustments,
                                       operation, save, backend, mode, backend_options)
    finally:
        _set_job(edit_id, status="done", ended_s=_time.time())


async def _replan_operation(cell_id, tree_id, operation_id, adjustments, operation, save,
                            backend, mode, backend_options) -> dict:
    from agents.trajectory import plan_selection
    from agents.trajectory.tools import plan_edit
    try:
        selection = plan_selection.select(backend, mode, backend_options=backend_options)
    except plan_selection.SelectionError as exc:
        return {"success": False, "error": str(exc), "valid": plan_selection.valid()}
    try:
        out = await asyncio.to_thread(plan_edit.replan_operation, cell_id, tree_id,
                                      operation_id, adjustments, operation, save,
                                      selection["backend"], selection["mode"],
                                      selection.get("backend_options"))
        return {**out, "backend": selection["backend"], "mode": selection["mode"],
                **({"backend_options": selection["backend_options"]}
                   if selection.get("backend_options") else {}),
                "selected_by": {"backend": selection["backend_from"],
                                "mode": selection["mode_from"]}}
    except plan_edit.EditError as exc:
        return {"success": False, "error": str(exc)}
    except cell_store.CellStoreError as exc:
        return {"success": False, "error": f"cell data unavailable: {exc}"}


# ─── Entry point ─────────────────────────────────────────────────────────────

def _check_cell_store_wiring() -> None:
    """Refuse to serve if this planner cannot reach the cell data it must read.

    In ``CELL_DATA_MODE=orchestrator`` every obstacle, robot config and task
    program comes over MCP. Booting without an endpoint would mean a server that
    accepts plan requests and fails each one deep inside collision checking — so
    it fails HERE, at boot, where the message is about wiring and a container
    restart loop is visible in ``docker compose ps``.

    Reachability is not checked: the orchestrator is what starts a planning run,
    so it is up by the time it asks, and probing at boot would only add a startup
    ordering constraint that does not exist.
    """
    print(f"[TrajectoryMCP] cell data: {cell_store.describe()}", flush=True)
    if not cell_store.is_remote():
        return
    try:
        cell_store.endpoint()
    except cell_store.CellStoreUnavailable as exc:
        raise SystemExit(
            f"[TrajectoryMCP] FATAL: CELL_DATA_MODE=orchestrator but no cell-data "
            f"endpoint ({exc}). Set ORCHESTRATOR_HOST (and ORCH_MCP_PORT), or "
            f"CELL_DATA_MCP_URL."
        ) from exc


if __name__ == "__main__":
    _check_cell_store_wiring()
    if _TRANSPORT == "stdio":
        print("[TrajectoryMCP] Starting stdio transport (offline/IDE mode)", flush=True)
        mcp.run(transport="stdio")
    else:
        import uvicorn
        from starlette.responses import JSONResponse
        print(
            f"[TrajectoryMCP] Starting SSE on 0.0.0.0:{_PORT} "
            f"(cell={_default_cell_id()}, llm={_resolve_model()}, "
            f"planning={_planning_model()})",
            flush=True,
        )
        from utils.mcp_http import add_jsonrpc_mcp_route

        app = mcp.sse_app()

        async def _health(_request):
            """Liveness probe used by the container healthcheck.

            Reports both seams this agent depends on but does not own: where cell
            data comes from, and whether the live progress topic is up. An operator
            debugging "the dashboard shows nothing" should not have to read the
            compose file to find out which half is missing.
            """
            from agents.trajectory import plan_progress

            active = _active_job()
            return JSONResponse({"status": "ok", "service": "trajectory-agent",
                                 "planner_id": _PLANNER_ID, "busy": active is not None,
                                 "active_tree_id": active.get("tree_id") if active else None,
                                 "cell_data": cell_store.describe(),
                                 "plan_progress": plan_progress.describe()})

        app.add_route("/health", _health, methods=["GET"])
        # Plain JSON-RPC /mcp endpoint for HTTP callers (orchestrator, Makefile).
        add_jsonrpc_mcp_route(app, mcp)

        # Announce this planner to the orchestrator and keep it informed (busy or
        # free), so the orchestrator can distribute runs across planners.
        from agents.trajectory import planner_registration

        def _status_for_registry() -> dict:
            s = planner_status()
            return {k: s[k] for k in ("planner_id", "busy", "active", "backends")}

        planner_registration.start(_status_for_registry)

        uvicorn.run(app, host="0.0.0.0", port=_PORT, log_level="warning")
