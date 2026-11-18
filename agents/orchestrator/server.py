"""
Orchestrator A2A HTTP server — :8889

Accepts task requests from external callers (dashboard, CI pipeline, other agents)
via a simple JSON API. Delegates to the orchestrator agent.

The contract is specified in ``docs/api/openapi.yaml``; ``tests/test_openapi_spec.py``
holds the spec and the routes below to each other, because an API description
nobody measures against the server is the thing that drifts.

Two conventions run through every endpoint:

* **Runs.** Anything that makes a robot move is a *run* with an id: task
  execution, plan replay, a manual jog. Pass ``"async": true`` and the call
  returns ``202`` with a run envelope immediately, then poll
  ``GET /a2a/result/<run_id>``; omit it and the call blocks until the run
  finishes. One run at a time per cell — a second motion command while one is
  executing is refused with ``409``, not interleaved
  (see :mod:`agents.orchestrator.run_registry`).
* **Models.** Every endpoint that reaches an LLM accepts an optional ``model`` —
  an alias (``"opus-5"``), a full model id, or ``"provider:model"`` — so one
  caller can run a hard cell on Opus 5 while another runs a routine replay on
  something cheaper, without restarting the stack. ``GET /a2a/models`` lists what
  is on offer. Omit it and the startup default applies (see
  :mod:`agents.llm_model`).

A run that *ran* but failed returns HTTP 200 with ``{"success": false, ...}``:
the request was served, the robot did not do the thing. HTTP status describes the
request; ``success`` describes the cell. 4xx/5xx mean the call never reached the
cell at all.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import httpx
from flask import Flask, request, jsonify

from agents.cell_defaults import default_cell_id
from agents.llm_model import describe_models
from agents.orchestrator import run_registry
from agents.orchestrator.agent import run_task
from agents.orchestrator.run_logger import (
    cell_id_of_run, list_run_logs, read_run_log,
)
from agents.orchestrator.run_registry import CellBusy
from agents.orchestrator.tools.cell_spec import read_cell_spec, read_task_program

from utils import cell_layout

# httpx logs every request at INFO, and the orchestrator polls the trajectory
# agent's MCP for plan status every few seconds — a "POST …/mcp 200 OK" line per
# poll buried the console. Warnings and errors still show; HTTP_REQUEST_LOGS=1
# brings the per-request lines back for debugging.
if os.environ.get("HTTP_REQUEST_LOGS", "0").strip() != "1":
    import logging as _logging
    for _name in ("httpx", "httpcore"):
        _logging.getLogger(_name).setLevel(_logging.WARNING)

app = Flask(__name__)
CELL_ID = default_cell_id()

# A program name becomes a file name under cells/<cell>/tasks/, so it is checked
# against a whitelist rather than sanitised: anything with a separator, a dot-dot,
# or a surprise character is refused outright. Rejecting is cheap; a submitted
# program that writes outside the cell's tasks/ directory is not recoverable.
_PROGRAM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Control fields a submitted task carries for the SERVER, which are not part of
# the task program format and must not be persisted into the program file.
_SUBMIT_CONTROL_FIELDS = ("model", "async", "overwrite", "run_id")

# How often a plan-only run asks the trajectory agent whether it has finished.
# Planning is minutes long, so a slow poll costs nothing; what it buys is that
# POST /a2a/abort is noticed within one interval.
_PLAN_POLL_S = float(os.environ.get("PLAN_POLL_INTERVAL_S", "3"))

# How long Reset waits for the run it just aborted to actually close before it
# tries to home the arms. An aborted executor returns within a tick, so this is
# slack for the thread to unwind — not a timeout anyone should hit.
_RESET_FREE_TIMEOUT_S = float(os.environ.get("RESET_FREE_TIMEOUT_S", "20"))


def _cell_mcp() -> str:
    host = os.environ.get("CELL_CONTROLLER_HOST", "cell-controller")
    port = os.environ.get("CELL_MCP_PORT", "8892")
    return f"http://{host}:{port}/mcp"


def _traj_mcp() -> str:
    """A planner for READS (plans, skills, reachability): a live, preferably free
    one from the registry. Planning runs go through planner_registry.submit, and
    everything about an existing run to the planner that owns it (_owner_mcp)."""
    from agents.orchestrator import planner_registry
    return planner_registry.any_url()


def _owner_mcp(tree_id: str) -> Optional[str]:
    """The MCP URL of the planner that owns run ``tree_id`` (None: no planner knows it)."""
    from agents.orchestrator import planner_registry
    p = planner_registry.owner_of(tree_id, _mcp_call)
    return p["mcp_url"] if p else None


def _planners_with_cell(cid: str) -> list:
    """``[(planner, status)]`` for every live planner currently busy with cell ``cid``."""
    from agents.orchestrator import planner_registry
    out = []
    for p in planner_registry.planners():
        if not p.get("live"):
            continue
        st = _mcp_call(p["mcp_url"], "planner_status", {}, timeout=5)
        active = st.get("active") if isinstance(st, dict) else None
        if active and active.get("cell_id") == cid:
            out.append((p, active))
    return out


def _cancel_cell_planning(cid: str, reason: str) -> dict:
    """Cancel the planning of cell ``cid`` on whichever planners are running it."""
    results = []
    for p, active in _planners_with_cell(cid):
        results.append({"planner_id": p["planner_id"], "tree_id": active.get("tree_id"),
                        **_mcp_call(p["mcp_url"], "cancel_planning",
                                    {"tree_id": active.get("tree_id"), "reason": reason},
                                    timeout=30)})
    return {"success": True, "cancelled": [r.get("tree_id") for r in results],
            "planners": results}


def _mcp_call(url: str, name: str, arguments: dict, timeout: float = 600.0) -> dict:
    """Call an MCP tool over the plain JSON-RPC /mcp endpoint and return the
    tool's parsed JSON result (or an {"error": ...} dict)."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}
    try:
        r = httpx.post(url, json=payload, timeout=timeout,
                       headers={"Content-Type": "application/json"})
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            return {"success": False, "error": body["error"].get("message", "mcp error")}
        content = body.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            txt = content[0].get("text", "{}")
            try:
                return json.loads(txt)
            except Exception:
                return {"result": txt}
        return {"success": False, "error": "empty MCP response"}
    except httpx.ConnectError:
        return {"success": False, "error": f"cannot connect to {url}"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}


# ─── Request helpers ─────────────────────────────────────────────────────────

def _body() -> dict:
    b = request.get_json(force=True, silent=True)
    return b if isinstance(b, dict) else {}


def _asked_async(b: dict) -> bool:
    """Did the caller ask for a background run?

    Defaults to FALSE: ``make task`` and the CI callers pipe the final result
    straight out of curl, and flipping the default would turn every one of those
    into a run envelope they cannot interpret. The dashboard opts in explicitly.
    """
    v = b.get("async", False)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _wait_for_cell_free(cell_id: str, timeout_s: float) -> float:
    """Block until no run is active on ``cell_id``, or ``timeout_s`` elapses.

    Returns the seconds waited. Deliberately does NOT force the registry entry
    closed on timeout: a run that will not end is a real condition an operator
    must see as a 409, not something to paper over by declaring the cell idle
    while something is still driving it.
    """
    t0 = time.monotonic()
    while run_registry.active(cell_id) is not None:
        if time.monotonic() - t0 >= max(0.0, timeout_s):
            break
        time.sleep(0.25)
    return round(time.monotonic() - t0, 2)


def _dispatch(kind: str, cell_id: str, work: Callable[[], dict], *,
              program: Optional[str] = None, asynchronous: bool = False):
    """Register a run, execute it (here or on a thread), and answer accordingly.

    Every motion endpoint funnels through this so the concurrency guard, the run
    id, and the envelope shape have exactly one implementation. A second copy is
    how ``/a2a/move`` would end up refusing a busy cell on a different rule than
    ``/a2a/run``.
    """
    try:
        envelope = run_registry.start(cell_id, kind, program=program)
    except CellBusy as busy:
        # Refusing is the safe answer: two schedulers driving one cell each
        # enforce their own gates and neither sees the other's motion.
        return jsonify({"success": False, "error": str(busy),
                        "active_run": busy.active}), 409

    run_id = envelope["run_id"]

    if asynchronous:
        run_registry.run_in_background(envelope, work)
        return jsonify(envelope), 202

    try:
        result = work()
    except Exception as exc:  # noqa: BLE001
        run_registry.fail(run_id, str(exc))
        return jsonify({"success": False, "error": str(exc), "run_id": run_id}), 500

    run_registry.finish(run_id, result)
    payload = dict(result) if isinstance(result, dict) else {"result": result}
    payload.setdefault("run_id", run_id)
    return jsonify(payload)


# ─── Health & capability ─────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cell_id": CELL_ID})


@app.route("/a2a/planners/register", methods=["POST"])
def a2a_planners_register():
    """
    A planner registers itself (and heartbeats) for planning distribution.

    Body: {"planner_id": str, "mcp_url": str, "busy": bool, "active": object?,
           "backends": [str]?, "heartbeat_s": number?}
    Returns: {"success": true, "planner": {...}, "planners": int}

    Sent by every trajectory-agent container at start-up and every
    ``PLANNER_HEARTBEAT_S`` (agents/trajectory/planner_registration.py). A planner
    that stops sending is no longer given new runs after a few missed beats.
    """
    from agents.orchestrator import planner_registry
    try:
        rec = planner_registry.register(_body())
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    return jsonify({"success": True, "planner": rec,
                    "planners": len(planner_registry.planners())})


@app.route("/a2a/planners", methods=["GET"])
def a2a_planners():
    """
    The planners runs are distributed to, and what each is doing.

    Returns: {"success": true, "planners": [{"planner_id", "mcp_url", "live",
              "busy", "active", "backends", "last_seen", "static"}], "free": int}

    One planning run per planner: a new run goes to a free one (POST /a2a/plan
    answers ``no_free_planner`` when there is none); a run's status, pause,
    resume and cancel go to the planner that owns it.
    """
    from agents.orchestrator import planner_registry
    ps = planner_registry.planners()
    return jsonify({"success": True, "planners": ps,
                    "free": sum(1 for p in ps if p.get("live") and not p.get("busy"))})


@app.route("/a2a/models", methods=["GET"])
def a2a_models():
    """What can be asked for in a `model` field, and what runs by default."""
    return jsonify(describe_models())


# ─── Skills ──────────────────────────────────────────────────────────────────
# Proxies of the trajectory MCP server's skill tools, which front
# utils/skill_registry.py — one owner, so A2A and MCP cannot disagree.

def _skill_status(result: dict) -> int:
    if result.get("success") is not False:
        return 200
    return 502 if str(result.get("error", "")).startswith("cannot connect") else 400


@app.route("/a2a/skills", methods=["GET"])
def a2a_skills():
    """List skills with their effective priority for a cell.

    Query: cell_id?, agent? (trajectory|cell|orchestrator), include_disabled?
    (default true).
    """
    args = {"cell_id": request.args.get("cell_id", CELL_ID),
            "include_disabled": request.args.get("include_disabled", "true").lower()
            not in ("0", "false", "no")}
    if request.args.get("agent"):
        args["agent"] = request.args["agent"]
    result = _mcp_call(_traj_mcp(), "list_skills", args, timeout=30)
    return jsonify(result), _skill_status(result)


@app.route("/a2a/skills/<agent>/<name>", methods=["GET"])
def a2a_skill(agent: str, name: str):
    """One skill's metadata and full SKILL.md text. Query: cell_id?"""
    result = _mcp_call(_traj_mcp(), "get_skill",
                       {"agent": agent, "name": name,
                        "cell_id": request.args.get("cell_id", CELL_ID)}, timeout=30)
    return jsonify(result), _skill_status(result)


@app.route("/a2a/skills/template", methods=["GET"])
def a2a_skill_template():
    """An example skill and how to bring one to a plan. Query: agent?
    ("trajectory" | "trajectory-classic", default "trajectory")."""
    result = _mcp_call(_traj_mcp(), "skill_template",
                       {"agent": request.args.get("agent", "trajectory")}, timeout=30)
    return jsonify(result), _skill_status(result)


@app.route("/a2a/skills/validate", methods=["POST"])
def a2a_skill_validate():
    """Check temporary skills before planning with them.

    Body: {"run_skills": [...], "agent": str?} — the same list POST /a2a/plan
    takes. Returns {"valid": bool, "skills"?, "error"?}; nothing is stored."""
    b = _body()
    result = _mcp_call(_traj_mcp(), "validate_run_skills",
                       {"run_skills": b.get("run_skills") or [],
                        "agent": b.get("agent", "trajectory")}, timeout=30)
    return jsonify(result), (200 if result.get("valid") else _skill_status(result))


@app.route("/a2a/skills/priority", methods=["POST"])
def a2a_skill_priority():
    """Set a skill's per-cell priority / enabled flag.

    Body: {"agent": str, "name": str, "cell_id": str?, "priority": int (0-100)?,
    "enabled": bool?, "reset": bool?}. Applies from the next planning run.
    Required skills (collision validation) cannot be disabled.
    """
    b = _body()
    args = {"cell_id": b.get("cell_id", CELL_ID), "agent": b.get("agent", ""),
            "name": b.get("name", "")}
    for key in ("priority", "enabled", "reset"):
        if key in b:
            args[key] = b[key]
    result = _mcp_call(_traj_mcp(), "set_skill_priority", args, timeout=30)
    return jsonify(result), _skill_status(result)


@app.route("/a2a/tasks", methods=["GET"])
def a2a_tasks():
    """List the task programs defined for a cell (`make tasks` calls this).

    Delegates to `read_cell_spec`, which already owns "what programs does this
    cell have?" — reading the tasks directory here as well is how the two answers
    would come to disagree.
    """
    cid = request.args.get("cell_id", CELL_ID)
    try:
        spec = read_cell_spec.invoke({"cell_id": cid})
    except FileNotFoundError as exc:
        return jsonify({"success": False, "cell_id": cid,
                        "error": f"cell not found or incomplete: {exc}"}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "cell_id": cid, "error": str(exc)}), 500
    programs = spec.get("programs", [])
    return jsonify({"success": True, "cell_id": cid, "programs": programs,
                    "count": len(programs), "operations": spec.get("operations")})


# ─── Task execution ──────────────────────────────────────────────────────────

@app.route("/a2a/run", methods=["POST"])
def a2a_run():
    """
    A2A task request endpoint — plan and execute a task program on a cell.

    Body: {"cell_id": str?, "program": str, "model": str?, "backend": str?,
           "mode": str?, "backend_options": object?, "async": bool?}
    ``backend_options`` are passed to the planner unchanged; it validates them
    against the backend (e.g. roboballet's {"policy": "<name>"}).
    Returns (sync): {"success": bool, "result": str, "elapsed_s": float,
                     "model": str, "run_id": str}
    Returns (async): 202 + a run envelope; poll GET /a2a/result/<run_id>.

    `model` is optional and applies to this task only: an alias ("opus-5"), a full
    model id, or "provider:model". See GET /a2a/models.
    """
    b = _body()
    cell_id      = b.get("cell_id", CELL_ID)
    program_name = b.get("program", "")
    model        = b.get("model") or None

    if not program_name:
        return jsonify({"success": False, "error": "Missing 'program' field"}), 400

    def work() -> dict:
        return run_task(cell_id, program_name, model=model, run_id=run_id,
                        **{k: b[k] for k in ("backend", "mode", "backend_options")
                           if b.get(k)})

    # run_task names its log after the run id, so the id has to exist before the
    # work function closes over it — hence the two-step dance with _dispatch.
    try:
        envelope = run_registry.start(cell_id, "task", program=program_name)
    except CellBusy as busy:
        return jsonify({"success": False, "error": str(busy),
                        "active_run": busy.active}), 409
    run_id = envelope["run_id"]

    if _asked_async(b):
        run_registry.run_in_background(envelope, work)
        return jsonify(envelope), 202

    try:
        result = work()
    except ValueError as exc:
        # An unknown provider/model is a caller error, not a 500 — say so, and
        # point at the endpoint that lists the valid answers.
        run_registry.fail(run_id, str(exc))
        return jsonify({"success": False, "error": str(exc), "run_id": run_id,
                        "hint": "GET /a2a/models lists the accepted values"}), 400
    except Exception as exc:  # noqa: BLE001
        run_registry.fail(run_id, str(exc))
        return jsonify({"success": False, "error": str(exc), "run_id": run_id}), 500

    run_registry.finish(run_id, result)
    return jsonify({**result, "run_id": run_id})


def _persist_submitted_program(b: dict):
    """Validate a submitted task program and write it to the cell's tasks dir.

    Returns ``(cell_id, program, target, None)`` on success, or
    ``(cell_id, program, None, (response, status))`` when the caller has to be
    told no. Shared by /a2a/task/save (write, run nothing) and /a2a/task/submit
    (write, then run) so the two cannot drift on what a legal program is or on
    where it lands — a save the operator can't then run would be a trap.
    """
    cell_id = b.get("cell_id") or CELL_ID
    program = (b.get("program") or "").strip()

    def no(payload: dict, status: int):
        return cell_id, program, None, (jsonify({"success": False, **payload}), status)

    if not program:
        return no({"error": "Missing 'program' field (the program name)"}, 400)
    if not _PROGRAM_NAME_RE.match(program):
        return no({"error": (
            f"invalid program name {program!r} — letters, digits, '.', '_' and '-' "
            f"only (it becomes a file name under cells/{cell_id}/tasks/)")}, 400)

    operations = b.get("operations")
    if not isinstance(operations, list) or not operations:
        return no({"error": (
            "'operations' must be a non-empty array — see "
            "skills/orchestrator/run-welding-program/references/task-json-format.md"
        )}, 400)

    tasks_dir = cell_layout.tasks_dir(cell_id)
    target = tasks_dir / f"{program}.json"
    if target.exists() and not b.get("overwrite"):
        return no({"error": (
            f"program '{program}' already exists in cell '{cell_id}'. Pass "
            f"\"overwrite\": true to replace it, or submit under another name."),
            "path": str(target)}, 409)

    program_json = {k: v for k, v in b.items() if k not in _SUBMIT_CONTROL_FIELDS}
    program_json["program"] = program
    program_json["cell_id"] = cell_id      # the file must name the cell it runs on

    try:
        tasks_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(program_json, indent=2) + "\n")
    except OSError as exc:
        return no({"error": f"could not write {target}: {exc}"}, 500)

    return cell_id, program, target, None


@app.route("/a2a/task/save", methods=["POST"])
def a2a_task_save():
    """
    SAVE a task program and run NOTHING — the authoring half of /a2a/task/submit.

    Body: identical to /a2a/task/submit (a task program per `skills/orchestrator/
    run-welding-program/references/task-json-format.md`, plus `overwrite`).
    Returns: {"success": true, "saved": path, "program": str, "cell_id": str,
              "num_operations": int, "commanded": false, "ran": false}

    This exists because writing a program and moving the arms are separate
    decisions. The task wizard in the panel authors a file here; planning it and
    running it stay explicit, later, separate calls (POST /a2a/plan, /a2a/run).
    No run is registered, so this never trips CellBusy and is safe while the cell
    is working.
    """
    b = _body()
    cell_id, program, target, err = _persist_submitted_program(b)
    if err:
        return err
    return jsonify({"success": True, "cell_id": cell_id, "program": program,
                    "saved": str(target),
                    "num_operations": len(b.get("operations") or []),
                    "commanded": False, "ran": False,
                    "hint": f"POST /a2a/plan with program '{program}' to calculate "
                            f"trajectories, or /a2a/run to plan and execute"})


@app.route("/a2a/task/submit", methods=["POST"])
def a2a_task_submit():
    """
    Submit a task program and run it — the authoring shortcut for a caller that
    holds the operations rather than a program name already on disk.

    Body: a task program per `skills/orchestrator/run-welding-program/references/
    task-json-format.md` — `program` and a non-empty `operations` array, plus the
    optional control fields `model`, `async` and `overwrite`.

    The program is PERSISTED to cells/<cell_id>/tasks/<program>.json and then run
    through the same path as GET-by-name, so what executed is on disk and can be
    re-run, diffed and replayed. An existing program of that name is NOT
    overwritten unless `overwrite: true` — a submit that silently replaced a
    hand-authored program would destroy the only copy of it.

    To author a program WITHOUT running it, POST /a2a/task/save instead.
    """
    b = _body()
    cell_id, program, target, err = _persist_submitted_program(b)
    if err:
        return err

    model = b.get("model") or None

    def work() -> dict:
        return run_task(cell_id, program, model=model, run_id=run_id)

    try:
        envelope = run_registry.start(cell_id, "task", program=program)
    except CellBusy as busy:
        # The program is already saved — say so, so the caller knows it only needs
        # to re-run it rather than re-submit it.
        return jsonify({"success": False, "error": str(busy),
                        "active_run": busy.active, "saved": str(target),
                        "hint": f"POST /a2a/run with program '{program}' once the "
                                f"cell is free"}), 409
    run_id = envelope["run_id"]
    envelope["saved"] = str(target)

    if _asked_async(b):
        run_registry.run_in_background(envelope, work)
        return jsonify(envelope), 202

    try:
        result = work()
    except Exception as exc:  # noqa: BLE001
        run_registry.fail(run_id, str(exc))
        return jsonify({"success": False, "error": str(exc), "run_id": run_id,
                        "saved": str(target)}), 500

    run_registry.finish(run_id, result)
    return jsonify({**result, "run_id": run_id, "saved": str(target)})


#: What a plan request may bring for that run only (validated by the planner).
PLAN_DIRECTIVE_FIELDS = ("run_skills", "skill_priorities", "instructions", "assignments")


@app.route("/a2a/plan", methods=["POST"])
def a2a_plan_task():   # not `a2a_plan`: GET /a2a/plans/<tree_id> owns that name,
                       # and Flask refuses to import a module with two of them
    """
    PLAN a task program — calculate trajectories, move nothing.

    Body: {"cell_id": str?, "program": str, "model": str?, "backend": str?,
           "mode": str?, "backend_options": object?, "async": bool?,
           "run_skills": list?, "skill_priorities": object?,
           "instructions": str?, "assignments": object?}
    ``backend_options`` are passed to the planner unchanged; it validates them
    against the backend (e.g. roboballet's {"policy": "<name>"}).

    For THIS plan only, to try a different strategy (nothing is stored on the
    cell; a resume keeps them):
      * ``run_skills`` — temporary skills, ``[{"name", "description", "content",
        "priority"?}]`` or ``[{"skill_md": "..."}]``; GET /a2a/skills/template
        shows one, POST /a2a/skills/validate checks them;
      * ``skill_priorities`` — ``{skill: priority | {"enabled": false}}``:
        re-order or switch off skills (required ones cannot be);
      * ``instructions`` — text put in front of the planner as binding;
      * ``assignments`` — ``{operation_id: robot}``, enforced as hard locks.
    A name or field the planner does not know refuses the request.
    ``backend`` / ``mode`` omitted → the planner's stack default, else ompl /
    classic.
    Returns: {"success": bool, "tree_id": str, "num_operations": int,
              "elapsed_s": float, "commanded": false, ...}

    This is the operator's Planning button, and it is the half of POST /a2a/run
    that produces a plan without dispatching it: the finished BehaviourTree is
    persisted for the cell, and running it is a separate, deliberate act (POST
    /a2a/replay). Nothing here reaches a robot.

    It still takes a run slot, because planning is long and the cell may only
    have one thing happening to it at a time — a second Planning press gets 409
    rather than two agents planning the same cell at once.

    Planning takes minutes: prefer `async: true` and poll GET
    /a2a/result/<run_id>, or stop it with POST /a2a/abort.
    """
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    program_name = (b.get("program") or "").strip()
    if not program_name:
        return jsonify({"success": False, "error": "Missing 'program' field",
                        "hint": "GET /a2a/tasks lists this cell's programs"}), 400
    if not _PROGRAM_NAME_RE.match(program_name):
        return jsonify({"success": False,
                        "error": f"invalid program name '{program_name}'"}), 400

    try:
        program = read_task_program.invoke({"cell_id": cid,
                                           "program_name": program_name})
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc),
                        "hint": "GET /a2a/tasks lists this cell's programs"}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": str(exc)}), 500

    operations = (program or {}).get("operations") or []
    if not operations:
        return jsonify({"success": False,
                        "error": f"program '{program_name}' has no operations"}), 400

    constraints = dict((program or {}).get("constraints") or {})
    # The operator's choice for THIS request beats the task file's constraints
    # (the planner reads both as the request); with neither, it applies the stack
    # default (make sim TRAJECTORY_BACKEND= / PLANNING_MODE=), else ompl / classic.
    if b.get("backend"):
        constraints["trajectory_backend"] = b["backend"]
    if b.get("mode"):
        constraints["planning_mode"] = b["mode"]
    if b.get("backend_options"):
        constraints["backend_options"] = b["backend_options"]

    def work() -> dict:
        args = {"cell_id": cid, "operations": operations}
        if constraints:
            args["constraints"] = constraints
        if b.get("model"):
            args["model"] = b["model"]
        for key in PLAN_DIRECTIVE_FIELDS:
            if b.get(key):
                args[key] = b[key]
        # To a FREE planner (one planning run per planner container); the planner
        # that accepts it owns the run, and is the one polled below.
        from agents.orchestrator import planner_registry
        accepted = planner_registry.submit(_mcp_call, "plan_trajectories", args, 120)
        if not accepted.get("accepted"):
            return {"success": False, "commanded": False,
                    "error_kind": accepted.get("error_kind"),
                    "error": accepted.get("error", "the trajectory agent "
                                                   "refused the plan request"),
                    "planners": accepted.get("planners"),
                    "program": program_name}
        tree_id = accepted["tree_id"]
        owner = planner_registry.url_of(accepted["planner_id"]) or _traj_mcp()
        # This work runs while its own run holds the cell, so the active run IS it.
        mine = (run_registry.active(cid) or {}).get("run_id")
        # Poll rather than block the MCP call: planning is minutes long and a
        # single long request is what used to time out as "fetch failed".
        t0 = time.monotonic()
        while True:
            status = _mcp_call(owner, "get_plan_status",
                              {"tree_id": tree_id}, timeout=30)
            state = status.get("status")
            if mine:
                # So GET /a2a/result/<run_id> answers "how far" while it runs.
                run_registry.note_progress(mine, tree_id=tree_id,
                                           progress=status.get("progress"))
            if state in ("done", "failed", "cancelled", "unknown", "interrupted", "paused"):
                return {"success": state == "done",
                        "status": state,
                        "tree_id": tree_id,
                        "planner_id": accepted.get("planner_id"),
                        "program": program_name,
                        "num_operations": status.get("num_operations",
                                                     len(operations)),
                        "model": status.get("model"),
                        "error": status.get("error"),
                        "note": status.get("note"),
                        "elapsed_s": round(time.monotonic() - t0, 1),
                        # Said out loud: a plan is not a motion. The tree is on
                        # disk; POST /a2a/replay is what drives the arms.
                        "commanded": False,
                        "next": f"POST /a2a/replay with tree_id={tree_id}"}
            time.sleep(_PLAN_POLL_S)

    return _dispatch("plan", cid, work, program=program_name,
                     asynchronous=_asked_async(b))


@app.route("/a2a/abort", methods=["POST"])
def a2a_abort():
    """
    Stop what the SYSTEM is doing — planning and/or an executing plan.

    Body: {"cell_id": str?, "reason": str?}
    Returns: {"success": true, "planning": {...}, "execution": {...}}

    This is the operator's Stop button, and it is a SOFTWARE stop:
      * a planning job is cancelled cooperatively (it unwinds at its next
        checkpoint; nothing was commanding a robot in the first place), and
      * an executing behaviour tree stops dispatching nodes.

    IT DOES NOT HALT AN ARM. A trajectory a controller has already accepted runs
    to its end, so an arm mid-motion finishes that motion — seconds — and then
    stops. To halt the arms where they are, use POST /a2a/stop, which trips the
    latched cell-wide e-stop.

    Both halves are attempted regardless of what is running, and neither is an
    error when idle: Stop pressed on an idle cell answers success with
    ``cancelled: []`` and ``aborted: false``.

    The run's own outcome is left to the executor, exactly as in /a2a/stop —
    reporting "aborted" from here would overwrite what actually happened.
    """
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    reason = b.get("reason") or "operator stop via A2A"
    # Only THIS cell's planning, on whichever planner runs it (several may run
    # other cells at the same time).
    planning = _cancel_cell_planning(cid, reason)
    execution = _mcp_call(_cell_mcp(), "abort_execution",
                          {"cell_id": cid, "reason": reason}, timeout=30)
    return jsonify({
        "success": True,
        "cell_id": cid,
        "reason": reason,
        "planning": planning,
        "execution": execution,
        "active_run": run_registry.active(cid),
        "estop": False,
        "note": ("activity stopped; an already-dispatched trajectory finishes. "
                 "POST /a2a/stop to halt the arms immediately (latching e-stop)."),
    })


@app.route("/a2a/reset", methods=["POST"])
def a2a_reset():
    """
    Stop the system's activity, then send the arms HOME. THIS MOVES THE ROBOTS.

    Body: {"cell_id": str?, "reason": str?, "async": bool?}
    Returns: {"success": bool, "abort": {...}, "home": {...}}

    Reset is Stop plus a reposition: /a2a/abort's software stop, then the cell's
    gated `home_robots` — the returns are planned for all arms together and
    measured, so arms that cannot go home clear of each other are homed one at a
    time, and an arm that cannot be cleared at all is LEFT WHERE IT STANDS and
    reported rather than driven blind.

    TWO CONSEQUENCES WORTH KNOWING BEFORE PRESSING IT:
      * it CLEARS A LATCHED E-STOP (execution must, or every dispatch would be
        rejected and this would report a hollow success), so do not use it to
        recover from an e-stop until the cause of that e-stop is gone, and
      * homing starts after the in-flight trajectory finishes, because the abort
        it performs is not a halt.
    """
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    reason = b.get("reason") or "operator reset via A2A"
    aborted = {
        "planning": _cancel_cell_planning(cid, reason),
        "execution": _mcp_call(_cell_mcp(), "abort_execution",
                               {"cell_id": cid, "reason": reason}, timeout=30),
    }

    # The run we just aborted still holds the cell's single run slot for the
    # moment its thread takes to unwind, and homing is itself a run. Without this
    # wait, Reset pressed during a run would abort it and then answer 409 — the
    # arms stopped where they were and the button looked broken.
    waited = _wait_for_cell_free(cid, _RESET_FREE_TIMEOUT_S)

    def work() -> dict:
        home = _mcp_call(_cell_mcp(), "home_robots", {"cell_id": cid}, timeout=600)
        return {"success": bool(home.get("success")),
                "abort": aborted, "home": home, "cleared_estop": True,
                "waited_for_run_s": waited,
                "note": "arms homed; a latched e-stop was cleared to do it"}

    return _dispatch("reset", cid, work, program="home",
                     asynchronous=_asked_async(b))


@app.route("/a2a/stop", methods=["POST"])
def a2a_stop():
    """
    Stop the cell: trip the cell-wide emergency stop.

    Body: {"cell_id": str?, "reason": str?}

    There is no gentler stop, and this endpoint deliberately does not invent one.
    The robot agents honour exactly one halt signal — the e-stop the cell
    controller publishes on ``/{cell_id}/estop`` — so this delegates to the cell's
    `trigger_estop` tool rather than adding a second, weaker notion of "stop" that
    some executors would observe and others would not.

    The run that was executing is NOT marked failed here: the executor reports its
    own outcome, and overwriting that from outside would replace what actually
    happened with what this endpoint assumed.
    """
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    reason = b.get("reason") or "operator stop via A2A"
    result = _mcp_call(_cell_mcp(), "trigger_estop",
                       {"cell_id": cid, "reason": reason}, timeout=30)
    stopped = run_registry.active(cid)
    return jsonify({**result, "cell_id": cid, "reason": reason,
                    "interrupted_run": stopped})


@app.route("/a2a/replay", methods=["POST"])
def a2a_replay():
    """Replay a previously calculated plan (by tree_id, or the latest) on the
    cell without re-planning. Body: {cell_id?, tree_id?, home_reset?, model?,
    async?}.

    `model` reaches only the recovery reasoner that runs if execution fails —
    replay itself is deterministic and consults no model."""
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    args = {"cell_id": cid,
            "tree_id": b.get("tree_id"),
            "home_reset": b.get("home_reset", True)}
    if b.get("model"):
        args["model"] = b["model"]

    return _dispatch("replay", cid,
                     lambda: _mcp_call(_cell_mcp(), "replay_plan", args, timeout=1800),
                     program=b.get("tree_id"),
                     asynchronous=_asked_async(b))


@app.route("/a2a/move", methods=["POST"])
def a2a_move():
    """Move a single robot — a manual jog, either to joint angles or to a point.

    Body: {cell_id?, robot_ns, duration_s?, async?} plus EITHER
          joint_positions: [6]   — the angles to drive to, or
          x, y, z                — a world-frame point, solved to joint angles by
                                   the trajectory agent's IK before dispatch.

    This is NOT a planned motion: it is a straight joint interpolation with no
    path planning and no inter-robot collision checking, the same as it has always
    been for `joint_positions`. Use it to jog one arm in a cell you can see is
    clear; use POST /a2a/run for anything that has to be collision-free. The
    response says so in `collision_checked`.

    Speed is `duration_s` (how long the move takes), because that is what the cell
    controller's `move_robot` actually implements — a velocity scale here would be
    a number this endpoint could not honour.
    """
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    robot_ns = b.get("robot_ns")
    if not robot_ns:
        return jsonify({"success": False, "error": "robot_ns is required"}), 400

    joints = b.get("joint_positions")
    has_point = all(k in b for k in ("x", "y", "z"))
    if joints is None and not has_point:
        return jsonify({"success": False, "error": (
            "provide either 'joint_positions' (6 angles) or 'x','y','z' "
            "(a world-frame point to solve by IK)")}), 400
    if joints is not None and has_point:
        return jsonify({"success": False, "error": (
            "provide 'joint_positions' OR 'x','y','z', not both — two targets in "
            "one request have no defined winner")}), 400

    try:
        duration_s = float(b.get("duration_s", 3.0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "duration_s must be a number"}), 400
    if duration_s <= 0:
        return jsonify({"success": False, "error": "duration_s must be > 0"}), 400

    if has_point:
        try:
            target = {k: float(b[k]) for k in ("x", "y", "z")}
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "x, y, z must be numbers"}), 400

        def work() -> dict:
            # IK first: an unreachable point must fail as a run, not silently
            # dispatch a partial move.
            ik = _mcp_call(_traj_mcp(), "check_reachability",
                           {"cell_id": cid, "robot_ns": robot_ns, **target},
                           timeout=60)
            if not ik.get("reachable") or not ik.get("ik_solution"):
                return {"success": False, "target": target,
                        "error": ik.get("error")
                        or f"{robot_ns} cannot reach {list(target.values())}",
                        "reachability": ik}
            result = _mcp_call(_cell_mcp(), "move_robot",
                               {"cell_id": cid, "robot_ns": robot_ns,
                                "joint_positions": ik["ik_solution"],
                                "duration_s": duration_s}, timeout=120)
            return {**result, "target": target,
                    "joint_positions": ik["ik_solution"],
                    "collision_checked": False}
    else:
        if not isinstance(joints, list) or not joints:
            return jsonify({"success": False,
                            "error": "joint_positions must be a non-empty array"}), 400

        def work() -> dict:
            result = _mcp_call(_cell_mcp(), "move_robot",
                               {"cell_id": cid, "robot_ns": robot_ns,
                                "joint_positions": joints,
                                "duration_s": duration_s}, timeout=120)
            return {**result, "collision_checked": False}

    return _dispatch("move", cid, work, program=robot_ns,
                     asynchronous=_asked_async(b))


# ─── Cell & plan state ───────────────────────────────────────────────────────

@app.route("/a2a/status", methods=["GET"])
def a2a_status():
    """Live per-robot state (joint + EE positions) from the cell controller."""
    cid = request.args.get("cell_id", CELL_ID)
    return jsonify(_mcp_call(_cell_mcp(), "get_cell_status", {"cell_id": cid}, timeout=30))


@app.route("/a2a/plans", methods=["GET"])
def a2a_plans():
    """List previously calculated plans (behaviour trees) for a cell."""
    cid = request.args.get("cell_id", CELL_ID)
    return jsonify(_mcp_call(_traj_mcp(), "list_calculated_plans", {"cell_id": cid}, timeout=30))


@app.route("/a2a/plans/<tree_id>", methods=["GET"])
def a2a_plan(tree_id: str):
    """One calculated plan's behaviour tree, WITHOUT its waypoint blackboard.

    For drawing the tree. A viewer needs the tree's shape and each leaf's ports;
    it does not need the joint samples, which are nearly all of the file (21 KB
    of the 24 KB a two-leaf plan takes) and grow with every weld point. The
    ``trajectory`` ports keep their ``"{key}"`` references and
    ``trajectory_sizes`` gives the waypoint count behind each key, so a leaf can
    still be labelled with the size of the motion it runs.

    ``tree_id`` may be ``latest`` for the most recently calculated plan — the
    same default ``get_calculated_plan`` applies when the id is omitted.
    Execution still goes through ``/a2a/replay``, which loads the full plan
    server-side; nothing round-trips the trajectories through a browser.

    **This is a reader, so it opts into failed runs** (``include_failed``). A
    failed run is listed by ``/a2a/plans`` on purpose — a token that expired at
    weld 13 says nothing about welds 1-12 — and a viewer that cannot open what the
    list shows is a dead end: it reported "could not load" for the one run the
    operator most needed to read. What comes back is marked ``status: "failed"``
    and ``executable: false``, with the failure beside it; when the run died before
    assembling anything there is no tree at all, and that is answered **200 with
    the evidence and a null tree**, not a 404. Refusing execution stays where it
    belongs: ``/a2a/replay`` loads through ``get_calculated_plan`` WITHOUT
    ``include_failed``, so nothing here widens what can reach a robot.
    """
    cid = request.args.get("cell_id", CELL_ID)
    wanted = None if tree_id in ("latest", "") else tree_id
    result = _mcp_call(_traj_mcp(), "get_calculated_plan",
                       {"cell_id": cid, "tree_id": wanted, "include_failed": True},
                       timeout=60)
    if not result.get("success"):
        # A failed run reports itself with status "failed"; that is a plan the
        # viewer can legitimately show, so it is not a 404. Anything else is.
        if str(result.get("status") or "") != "failed":
            return jsonify({"cell_id": cid, "tree_id": tree_id, **result}), 404
        return jsonify({"success": True, "cell_id": cid, "tree_id": tree_id,
                        "status": "failed", "executable": False,
                        "num_operations": result.get("num_planned") or 0,
                        "trajectory_sizes": {},
                        "behaviour_tree": None,
                        "error": result.get("error"),
                        "error_kind": result.get("error_kind"),
                        "resumable": bool(result.get("resumable")),
                        "failures": result.get("failures") or [],
                        "planned": result.get("planned") or [],
                        "note": result.get("note")})

    bt = dict(result.get("behaviour_tree") or {})
    blackboard = bt.pop("blackboard", {}) or {}
    bt.pop("xml", None)  # the same tree again, as text
    sizes = {k: len(v) for k, v in blackboard.items() if isinstance(v, list)}
    return jsonify({"success": True, "cell_id": cid,
                    "tree_id": result.get("tree_id"),
                    "num_operations": result.get("num_operations"),
                    "status": result.get("status") or "ok",
                    "executable": result.get("executable", True),
                    "error": result.get("error"),
                    "error_kind": result.get("error_kind"),
                    "resumable": bool(result.get("resumable")),
                    "trajectory_sizes": sizes,
                    "behaviour_tree": bt})


@app.route("/a2a/plan_traces", methods=["GET"])
def a2a_plan_traces():
    """The history: every planning run whose decision trace was recorded, newest first.

    Metadata only — a full trace is the whole decision history, which is not what a
    picker needs to draw its rows. Read straight off the cell's shared logs volume,
    so a trace the trajectory agent wrote is servable here and outlives the
    container that produced it (three recreations in one afternoon of debugging is
    what made "read it from `docker logs`" untenable).
    """
    cid = request.args.get("cell_id", CELL_ID)
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
    except ValueError:
        return jsonify({"success": False, "error": "limit must be an integer"}), 400
    from agents.trajectory import plan_trace
    return jsonify({"success": True, "cell_id": cid,
                    "traces": plan_trace.listing(cid, limit=limit)})


@app.route("/a2a/plan_trace/<trace_id>", methods=["GET"])
def a2a_plan_trace(trace_id: str):
    """One planning run's decisions, indexed per weld point.

    ``trace_id`` may be ``latest`` for the most recent recorded run — the same
    convenience ``/a2a/plans/<tree_id>`` offers, and the default a viewer opens on.

    The response carries both views of one history: ``points`` (each operation in
    the task, with the decisions that touched it and what became of it) and
    ``decisions`` (every tool call in order, with the reasoning that preceded it and
    the result that followed). Nothing is recomputed here — the trace is what the
    planner recorded as it ran.
    """
    cid = request.args.get("cell_id", CELL_ID)
    from agents.trajectory import plan_trace

    wanted = trace_id
    if trace_id in ("latest", ""):
        history = plan_trace.listing(cid, limit=1)
        if not history:
            return jsonify({"success": False, "cell_id": cid, "trace_id": None,
                            "error": f"no planning traces recorded for '{cid}' yet — "
                                     "they are written from the next planning run on"}), 404
        wanted = history[0].get("trace_id")

    if not plan_trace.is_safe_id(wanted or ""):
        return jsonify({"success": False, "cell_id": cid,
                        "error": "trace_id must be a plain identifier"}), 400

    trace = plan_trace.read(cid, wanted)
    if trace is None:
        return jsonify({"success": False, "cell_id": cid, "trace_id": wanted,
                        "error": f"no planning trace '{wanted}'"}), 404
    return jsonify({"success": True, "cell_id": cid, **trace})


@app.route("/a2a/plan_events", methods=["GET"])
def a2a_plan_events():
    """The live feed of a planning run — what it is doing, as it does it.

    Query: ``cell_id?``, ``trace_id?`` (default: the newest run), ``since?`` (the
    ``next_seq`` from the previous poll), ``limit?``.

    Read straight off the shared logs volume rather than asked of the trajectory
    agent, for the same reason the traces are: the run being watched is in another
    container, and the interesting runs are the ones whose container then died.
    Polling with ``since`` returns only what is new, so a dashboard refreshing every
    second re-reads nothing.
    """
    cid = request.args.get("cell_id", CELL_ID)
    try:
        since = max(0, int(request.args.get("since", 0)))
        limit = max(1, min(int(request.args.get("limit", 200)), 1000))
    except ValueError:
        return jsonify({"success": False,
                        "error": "since and limit must be integers"}), 400
    trace_id = request.args.get("trace_id") or None
    from agents.trajectory import plan_events
    if trace_id and not plan_events.is_safe_id(trace_id):
        return jsonify({"success": False, "cell_id": cid,
                        "error": "trace_id must be a plain identifier"}), 400
    feed = plan_events.read(cid, trace_id, since=since, limit=limit)
    return jsonify({"success": True, "cell_id": cid, **feed})


@app.route("/a2a/checkpoints", methods=["GET"])
def a2a_checkpoints():
    """Planning runs that can be CONTINUED instead of restarted, newest first.

    Query: ``cell_id?``, ``limit?``.

    A checkpoint exists only while its run has somewhere left to go — written when
    a run pauses or is interrupted, deleted the moment it produces a tree — so every
    entry here with ``resumable: true`` is work that would otherwise be redone.
    """
    cid = request.args.get("cell_id", CELL_ID)
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
    except ValueError:
        return jsonify({"success": False, "error": "limit must be an integer"}), 400
    # Ask the planner, because a checkpoint is the ONE artefact it still owns: an
    # unfinished run lives on its private scratch volume (see plan_checkpoint), so
    # this is the "ask the owning agent" direction, not a shortcut around it.
    # Reading the local volume is the fallback for a planner that is not up yet and
    # for a deployment where scratch and /cells are the same root — which is the
    # default, and what a test exercises.
    # Every planner keeps the checkpoints of ITS runs on its own scratch: ask each,
    # and say which planner holds which (only that one can resume it).
    from agents.orchestrator import planner_registry
    items: list = []
    answered = False
    for p in planner_registry.planners():
        if not p.get("live"):
            continue
        asked = _mcp_call(p["mcp_url"], "list_checkpoints", {"cell_id": cid}, timeout=10.0)
        if isinstance(asked.get("checkpoints"), list):
            answered = True
            for item in asked["checkpoints"]:
                planner_registry.remember(item.get("tree_id"), p["planner_id"])
                items.append({**item, "planner_id": p["planner_id"]})
    items = sorted(items, key=lambda i: str(i.get("saved_at") or ""), reverse=True)[:limit]
    if not answered:
        from agents.trajectory import plan_checkpoint
        items = plan_checkpoint.listing(cid, limit=limit)
    return jsonify({"success": True, "cell_id": cid,
                    "count": len(items),
                    "resumable": sum(1 for i in items if i.get("resumable")),
                    "checkpoints": items})


@app.route("/a2a/plan/pause", methods=["POST"])
def a2a_plan_pause():
    """
    Pause a planning run, keeping everything it has computed.

    Body: {"cell_id": str?, "tree_id": str?, "reason": str?}

    This is NOT /a2a/abort. Abort throws the run away and the next attempt starts at
    the first weld; pause writes the run's place to disk — the model's conversation
    and every trajectory already solved — and stops at the next clean step boundary,
    so POST /a2a/plan/resume continues from there. Use it before a token expires or
    a container has to be recreated.

    Pausing commands no robot: planning is computation. It also cannot halt an arm —
    POST /a2a/stop does that.

    The pause is cooperative, so this returns immediately and the job reaches
    ``status: paused`` a step later. Pausing an idle stack is not an error.
    """
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    reason = b.get("reason") or "paused by operator"
    args = {"reason": reason}
    if b.get("tree_id"):
        args["tree_id"] = b["tree_id"]
        url = _owner_mcp(b["tree_id"])
        if url is None:
            return jsonify({"success": False, "cell_id": cid, "tree_id": b["tree_id"],
                            "error": "no planner knows this run"}), 404
        targets = [(url, args)]
    else:
        # The cell's run(s), on whichever planner holds them.
        targets = [(p["mcp_url"], {**args, "tree_id": a.get("tree_id")})
                   for p, a in _planners_with_cell(cid)]
    results = [_mcp_call(u, "pause_planning", a, timeout=30) for u, a in targets]
    if len(results) == 1:
        return jsonify({**results[0], "cell_id": cid, "reason": reason})
    return jsonify({"success": True, "cell_id": cid, "reason": reason, "paused": results})


@app.route("/a2a/plan/resume", methods=["POST"])
def a2a_plan_resume():
    """
    Continue a paused or interrupted planning run from where it stopped.

    Body: {"cell_id": str?, "tree_id": str, "model": str?}

    The recovery for the failures that are not the cell's fault — an expired token,
    a dropped socket, a truncated reply, an exhausted step budget, an operator's
    pause. The run keeps its conversation and its solved trajectories and carries on
    at the next unplanned weld.

    Asynchronous, like POST /a2a/plan: the run continues in the BACKGROUND under the
    same tree_id, appending to the same event stream. Watch it with
    GET /a2a/plan_events and GET /a2a/plans/<tree_id> when it lands.

    A run with no resumable checkpoint is REFUSED (409) rather than quietly
    restarted — GET /a2a/checkpoints lists what can be resumed.
    """
    b = _body()
    cid = b.get("cell_id", CELL_ID)
    tree_id = b.get("tree_id")
    if not tree_id:
        return jsonify({"success": False, "cell_id": cid,
                        "error": "'tree_id' is required — GET /a2a/checkpoints "
                                 "lists the runs that can be resumed"}), 400
    args = {"tree_id": tree_id, "cell_id": cid}
    if b.get("model"):
        args["model"] = b["model"]
    # Only the planner that ran it holds its checkpoint (on its own scratch), so a
    # resume goes THERE; busy, it refuses (planner_busy) and this answers 409.
    url = _owner_mcp(tree_id)
    if url is None:
        return jsonify({"success": False, "cell_id": cid, "tree_id": tree_id,
                        "error": "no planner holds a checkpoint for this run — GET "
                                 "/a2a/checkpoints lists the runs that can be resumed"}), 409
    result = _mcp_call(url, "resume_planning", args, timeout=120)
    if not result.get("accepted"):
        return jsonify({"success": False, "cell_id": cid, **result}), 409
    return jsonify({"success": True, "cell_id": cid, **result})


def _find_bt_node(node: dict, node_id: str) -> dict | None:
    """Depth-first search for a node by its ``id``.

    Ids are optional in the format (``btcpp.to_btcpp`` only emits one if the
    source carried it) and do NOT survive an XML round-trip, so a miss is a
    normal answer here, not a corrupt plan.
    """
    if not isinstance(node, dict):
        return None
    if node.get("id") == node_id:
        return node
    for child in node.get("children") or []:
        found = _find_bt_node(child, node_id)
        if found is not None:
            return found
    return None


def _phase_segments(trajectory: list) -> list:
    """Split a trajectory into consecutive runs of the same ``phase``.

    ``phase`` (``"transit"`` / ``"approach"``, set by the OMPL planner) is
    metadata the EXECUTOR ignores entirely — it reads only ``positions`` and
    ``time_s``. It is, however, the only record of which part of a motion is the
    fast cross-cell move and which is the slow tool-oriented run-in, which is
    exactly the distinction someone stepping through a plan is looking for.

    Segments are contiguous index ranges so a caller can address one by slicing
    the same waypoint list it was handed.
    """
    segments: list[dict] = []
    for i, sample in enumerate(trajectory):
        if not isinstance(sample, dict):
            continue
        phase = sample.get("phase") or "transit"
        t = float(sample.get("time_s") or 0.0)
        if segments and segments[-1]["phase"] == phase:
            seg = segments[-1]
            seg["end_index"] = i
            seg["end_s"] = t
            seg["points"] += 1
        else:
            segments.append({"phase": phase, "start_index": i, "end_index": i,
                             "start_s": t, "end_s": t, "points": 1})
    return segments


@app.route("/a2a/plans/<tree_id>/nodes/<node_id>", methods=["GET"])
def a2a_plan_node(tree_id: str, node_id: str):
    """ONE node of a plan, with the waypoints the tree view cannot afford in bulk.

    ``GET /a2a/plans/<tree_id>`` deliberately strips every trajectory, because a
    whole plan's samples are nearly all of its bytes. But a leaf is opaque
    without them: "ExecuteTrajectory move_robot1_VIA_W06" says nothing about what
    the arm does. This serves one node's worth — a couple of hundred samples at
    most, whatever the size of the plan — so descending into a node costs the
    same on a 200-weld plan as on a two-leaf one.

    Returns the node's ports, its ``phase_segments`` (see ``_phase_segments``) and
    the waypoints themselves. A control node (Sequence/Parallel) has no
    trajectory: it answers with its children's ids instead, which is what a
    caller walking downwards needs next.

    Reads failed runs too, for the reason ``/a2a/plans/<tree_id>`` does: a partial
    tree the viewer can draw but not descend into is worse than one it refuses
    outright. Reading a node is not executing it.
    """
    cid = request.args.get("cell_id", CELL_ID)
    wanted = None if tree_id in ("latest", "") else tree_id
    result = _mcp_call(_traj_mcp(), "get_calculated_plan",
                       {"cell_id": cid, "tree_id": wanted, "include_failed": True},
                       timeout=60)
    if not result.get("success"):
        return jsonify({"cell_id": cid, "tree_id": tree_id, **result}), 404

    bt = result.get("behaviour_tree") or {}
    node = _find_bt_node(bt.get("root") or {}, node_id)
    if node is None:
        return jsonify({"success": False, "cell_id": cid,
                        "tree_id": result.get("tree_id"), "node_id": node_id,
                        "error": f"no node with id '{node_id}' in this tree"}), 404

    blackboard = bt.get("blackboard") or {}
    ports = dict(node.get("ports") or {})
    traj_ref = ports.get("trajectory")
    trajectory: list = []
    if isinstance(traj_ref, str):
        match = re.fullmatch(r"\{(.+)\}", traj_ref)
        key = match.group(1) if match else None
        if key is not None:
            resolved = blackboard.get(key)
            if isinstance(resolved, list):
                trajectory = resolved
    elif isinstance(traj_ref, list):
        # Inlined rather than referenced — the shape move_robot builds.
        trajectory = traj_ref

    return jsonify({
        "success": True, "cell_id": cid, "tree_id": result.get("tree_id"),
        "node_id": node_id, "type": node.get("type"), "name": node.get("name"),
        "ports": ports,
        "children": [c.get("id") for c in (node.get("children") or [])],
        "phase_segments": _phase_segments(trajectory),
        "duration_s": (float(trajectory[-1].get("time_s") or 0.0)
                       if trajectory else 0.0),
        "waypoints": trajectory,
    })


@app.route("/a2a/recordings", methods=["GET"])
def a2a_recordings():
    """List the .rrd execution recordings stored for a cell.

    The cell controller owns the recording store, so this asks it rather than
    reading the directory — the orchestrator's filesystem view and the cell's are
    only the same by accident of deployment.
    """
    cid = request.args.get("cell_id", CELL_ID)
    return jsonify({"cell_id": cid,
                    **_mcp_call(_cell_mcp(), "list_recordings",
                                {"cell_id": cid}, timeout=30)})


@app.route("/a2a/recordings/<tree_id>", methods=["GET"])
def a2a_recording(tree_id: str):
    """Metadata for one recording, by tree_id (a trailing `.rrd` is tolerated).

    The entry is looked up in the cell's own listing rather than assembled from a
    path built out of `tree_id`, which is what keeps a caller-supplied name from
    ever reaching the filesystem.
    """
    cid = request.args.get("cell_id", CELL_ID)
    wanted = tree_id[:-4] if tree_id.endswith(".rrd") else tree_id
    listing = _mcp_call(_cell_mcp(), "list_recordings", {"cell_id": cid}, timeout=30)
    for entry in listing.get("recordings", []):
        if entry.get("tree_id") == wanted:
            return jsonify({"success": True, "cell_id": cid, **entry})
    return jsonify({"success": False, "cell_id": cid, "tree_id": wanted,
                    "error": f"no recording '{wanted}' in cell '{cid}'",
                    **({"warning": listing["warning"]} if "warning" in listing else {})
                    }), 404


# ─── Runs ────────────────────────────────────────────────────────────────────

def _envelope_from_log(run_id: str, cell_id: Optional[str] = None) -> Optional[dict]:
    """Rebuild a run envelope from the on-disk log.

    This is what makes a run id outlive the process that issued it: the registry
    is in-memory, so after a restart the log is the only record. Status comes from
    the log's own `success` flag — a persisted log always describes a FINISHED run,
    because it is written after the run returns.
    """
    record = read_run_log(run_id, cell_id)
    if record is None:
        return None
    meta = record.get("run") or {}
    ok = bool(meta.get("success"))
    return {
        "run_id":      meta.get("run_id", run_id),
        "cell_id":     meta.get("cell_id", cell_id or cell_id_of_run(run_id)),
        "kind":        "task",
        "program":     meta.get("program"),
        "status":      run_registry.SUCCEEDED if ok else run_registry.FAILED,
        "done":        True,
        "started_at":  meta.get("timestamp"),
        "finished_at": meta.get("timestamp"),
        "elapsed_s":   meta.get("elapsed_s"),
        "result":      {"success": ok, "model": meta.get("llm_model")},
        "error":       meta.get("error") or None,
        "source":      "run_log",
    }


@app.route("/a2a/runs", methods=["GET"])
def a2a_runs():
    """Runs this orchestrator knows about, newest first.

    Merges the live registry with the on-disk logs so a restart does not make a
    cell's history disappear. Live entries win on run_id: they carry the running
    state and the full result, which a log cannot until the run ends.
    """
    cid = request.args.get("cell_id", CELL_ID)
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
    except ValueError:
        return jsonify({"success": False, "error": "limit must be an integer"}), 400

    runs = run_registry.recent(cid, limit=limit)
    seen = {r["run_id"] for r in runs}
    for meta in list_run_logs(cid, limit=limit):
        rid = meta.get("run_id")
        if rid and rid not in seen:
            envelope = _envelope_from_log(rid, cid)
            if envelope:
                runs.append(envelope)
                seen.add(rid)
    runs.sort(key=lambda r: (r.get("started_at") or ""), reverse=True)
    return jsonify({"success": True, "cell_id": cid, "runs": runs[:limit],
                    "count": min(len(runs), limit),
                    "active": run_registry.active(cid)})


@app.route("/a2a/result/<run_id>", methods=["GET"])
def a2a_result(run_id: str):
    """Poll one run: `{run_id, status, done, result, error, ...}`.

    `status` is "running" | "succeeded" | "failed" and `done` is the two terminal
    states collapsed, so a poller does not have to hardcode which ones they are.
    """
    envelope = run_registry.get(run_id) or _envelope_from_log(
        run_id, request.args.get("cell_id"))
    if envelope is None:
        return jsonify({"success": False, "run_id": run_id, "done": False,
                        "error": f"unknown run '{run_id}'"}), 404
    return jsonify(envelope)


@app.route("/a2a/reasoning", methods=["GET"])
def a2a_reasoning():
    """The LLM reasoning + action trace of a run.

    Query: cell_id?, run_id? (default: the newest run for the cell), limit?
    (default 200 — the LAST N entries, which is the end a reader wants).

    The trace comes from the persisted run log, so it is available only once the
    run has finished; a run still executing reports `status: "running"` with no
    entries rather than a partial trace this endpoint would have to guess at.
    """
    cid = request.args.get("cell_id", CELL_ID)
    run_id = request.args.get("run_id")
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 2000))
    except ValueError:
        return jsonify({"success": False, "error": "limit must be an integer"}), 400

    if not run_id:
        live = run_registry.recent(cid, limit=1)
        if live:
            run_id = live[0]["run_id"]
        else:
            logs = list_run_logs(cid, limit=1)
            if not logs:
                return jsonify({"success": False, "cell_id": cid, "entries": [],
                                "error": f"no runs recorded for cell '{cid}'"}), 404
            run_id = logs[0].get("run_id")

    record = read_run_log(run_id, cid)
    if record is None:
        envelope = run_registry.get(run_id)
        if envelope is not None and not envelope["done"]:
            return jsonify({"success": True, "cell_id": cid, "run_id": run_id,
                            "status": envelope["status"], "entries": [],
                            "note": "run still executing — the trace is written "
                                    "when it finishes"})
        return jsonify({"success": False, "cell_id": cid, "run_id": run_id,
                        "entries": [],
                        "error": f"no run log for '{run_id}'"}), 404

    meta = record.get("run") or {}
    trace = record.get("reasoning_trace") or []
    entries = []
    for seq, raw in enumerate(trace, start=1):
        entry = {"seq": seq, "role": raw.get("role", "?"),
                 "msg": raw.get("content", "")}
        if raw.get("tool_calls"):
            entry["tool_calls"] = raw["tool_calls"]
            if not entry["msg"]:
                # An assistant turn that only called tools has no text. Naming the
                # calls beats rendering a blank line in a viewer.
                entry["msg"] = "→ " + ", ".join(
                    tc.get("name", "?") for tc in raw["tool_calls"])
        if raw.get("tool_name"):
            entry["tool_name"] = raw["tool_name"]
        entries.append(entry)

    return jsonify({"success": True, "cell_id": meta.get("cell_id", cid),
                    "run_id": meta.get("run_id", run_id),
                    "program": meta.get("program"),
                    "status": (run_registry.SUCCEEDED if meta.get("success")
                               else run_registry.FAILED),
                    "model": meta.get("llm_model"),
                    "entries": entries[-limit:],
                    "total_entries": len(entries),
                    "truncated": len(entries) > limit})


if __name__ == "__main__":
    port = int(os.environ.get("ORCHESTRATOR_PORT", "8889"))

    # The cell-data MCP, in this process, because this is the process that holds
    # /cells read-write. The planner reads and writes every derived artefact of a
    # cell through it instead of mounting the volume a second time — see
    # agents/orchestrator/cell_data_mcp.py. Started BEFORE the Flask app, which
    # blocks; a planner asking early gets a connection refused and refuses to
    # plan, which is the intended failure.
    mcp_port = int(os.environ.get("ORCH_MCP_PORT", "8893"))
    try:
        from agents.orchestrator import cell_data_mcp
        cell_data_mcp.serve_in_thread(mcp_port)
        print(f"[Orchestrator] Cell-data MCP on 0.0.0.0:{mcp_port}", flush=True)
    except Exception as exc:  # noqa: BLE001
        # Availability, not safety: without this endpoint the planner refuses to
        # plan (it will not silently read the volume), and the orchestrator's own
        # HTTP surface still answers. Say so loudly and carry on.
        print(f"[Orchestrator] WARNING: cell-data MCP did not start ({exc}) — "
              f"the planner will refuse to plan", flush=True)

    print(f"[Orchestrator] Starting A2A HTTP on 0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
