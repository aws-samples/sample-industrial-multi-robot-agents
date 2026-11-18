#!/usr/bin/env python3
"""
Call the Cell Agent's MCP endpoint from the command line.

The cell controller speaks JSON-RPC 2.0 over HTTP at ``/mcp`` (see
``utils/mcp_http.py``). Every call is the same shape — wrap a tool name and its
arguments in an envelope, POST it, then unwrap ``result.content[0].text`` — so
this module owns that shape once and the Makefile targets stay one-liners.

Two agents split the work and neither does the other's half: the trajectory
agent (:8890) plans, the cell controller (:8892) executes. ``run-task`` therefore
spans both — plan, poll until the plan is done, then execute it.

Usage:
    python3 scripts/cell_mcp.py run-task    --cell welding_cell_2_small --task car_door_spot_welding
    python3 scripts/cell_mcp.py run-task    --cell welding_cell_2_small --task car_door_spot_welding --plan-only
    python3 scripts/cell_mcp.py run-task    --cell welding_cell_2_small --task car_door_spot_welding --model sonnet-5
    python3 scripts/cell_mcp.py call        --tool list_models
    python3 scripts/cell_mcp.py replay-plan --cell welding_cell_2_small --tree-id bt_1785860678_welding_cell_2_small
    python3 scripts/cell_mcp.py replay      --cell welding_cell_2_small --tree-id bt_1785860678_welding_cell_2_small
    python3 scripts/cell_mcp.py recordings  --cell welding_cell_2_small
    python3 scripts/cell_mcp.py call        --tool get_cell_status --args '{"cell_id": "welding_cell_2_small"}'

Exit status is 0 when the tool reports success and 1 otherwise, so
``make replay`` still fails a shell pipeline the way it always has.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_PORT = 8892
DEFAULT_TRAJ_PORT = 8890
DEFAULT_HOST = "localhost"

# Planning/execution calls run for minutes; introspection should not hang.
TASK_TIMEOUT_S = 600.0
QUERY_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 5.0


class CellMCPError(RuntimeError):
    """A call could not be completed (transport, protocol, or tool error)."""


def call_tool(
    tool: str,
    arguments: dict,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout_s: float = QUERY_TIMEOUT_S,
) -> dict:
    """Invoke one MCP tool and return its decoded result payload.

    Raises ``CellMCPError`` with an actionable message rather than letting an
    httpx traceback surface: the overwhelmingly common failure is simply that the
    stack is not running, and a stack trace buries that.
    """
    import httpx

    url = f"http://{host}:{port}/mcp"
    envelope = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }

    try:
        response = httpx.post(url, json=envelope, timeout=timeout_s)
    except httpx.ConnectError as exc:
        raise CellMCPError(
            f"cannot reach the cell controller at {url} — is the stack up? "
            f"Start it with 'make sim-detached' ({exc})"
        ) from exc
    except httpx.ReadTimeout as exc:
        raise CellMCPError(
            f"'{tool}' did not respond within {timeout_s:.0f}s. Long plans can "
            f"exceed this — check progress with 'make logs-cell' ({exc})"
        ) from exc

    if response.status_code != 200:
        raise CellMCPError(
            f"{url} returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise CellMCPError(f"non-JSON response from {url}: {response.text[:300]}") from exc

    if "error" in body:
        err = body["error"]
        raise CellMCPError(f"tool '{tool}' failed: {err.get('message', err)}")

    content = (body.get("result") or {}).get("content") or []
    if not content:
        # A result with no content blocks is still a valid response for some
        # tools; hand back the raw body rather than inventing a failure.
        return body

    text = content[0].get("text", "{}")
    try:
        # strict=False: tool payloads embed multi-line diagnostics, and raw
        # newlines inside a JSON string are rejected by the strict decoder.
        return json.loads(text, strict=False)
    except ValueError:
        return {"text": text}


def _succeeded(result: dict) -> bool:
    """Interpret a tool payload as success/failure.

    Tools that report a verdict carry ``success``; the rest (queries such as
    ``list_recordings``) are successful by virtue of having answered at all.
    """
    if isinstance(result, dict) and "success" in result:
        return bool(result["success"])
    return True


def _load_operations(cell: str, task: str) -> list:
    """Read a task program's operation points off disk.

    ``plan_trajectories`` takes the operations themselves, not a task name — the
    planner never reads the task file.
    """
    # Where the cell keeps its task programs (``<cell>/agentic_planner/tasks``
    # on a migrated cell): the one layout resolver, not a hard-coded path.
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils import cell_layout
    path = cell_layout.task_file(cell, task, str(repo / "cells"))
    if not path.exists():
        raise CellMCPError(
            f"no task program '{task}' for cell '{cell}' (looked in {path}). "
            f"List what exists with 'make tasks SIM={cell}'"
        )
    try:
        program = json.loads(path.read_text())
    except ValueError as exc:
        raise CellMCPError(f"{path} is not valid JSON: {exc}") from exc

    operations = program.get("operations") or []
    if not operations:
        raise CellMCPError(f"{path} declares no operations")
    return operations


def cmd_run_task(args) -> int:
    """Plan a task program, then execute the resulting tree.

    Two agents own the two halves and neither can do the other's: the trajectory
    agent (:8890) turns operation points into a BehaviourTree, and the cell
    controller (:8892) executes trees. So "run a task" is necessarily
    plan-then-execute, not one call.
    """
    operations = _load_operations(args.cell, args.task)
    model = getattr(args, "model", None)
    print(f"[1/3] planning {len(operations)} operations on {args.cell}"
          f"{f' with {model}' if model else ''} ...")

    plan_args = {"cell_id": args.cell, "operations": operations}
    # Backend and mode go in the MCP request; omitted, the planner applies the
    # stack default (make sim TRAJECTORY_BACKEND= / PLANNING_MODE=), else ompl /
    # classic. --model names the LLM, never the MOTION planner.
    for key in ("backend", "mode"):
        if getattr(args, key, None):
            plan_args[key] = getattr(args, key)
    if getattr(args, "backend_options", None):
        plan_args["backend_options"] = json.loads(args.backend_options)
    if model:
        plan_args["model"] = model
    accepted = call_tool(
        "plan_trajectories", plan_args,
        args.host, args.traj_port, QUERY_TIMEOUT_S,
    )
    tree_id = accepted.get("tree_id")
    if not tree_id:
        print(json.dumps(accepted, indent=2))
        raise CellMCPError("plan_trajectories did not return a tree_id")
    print(f"      accepted as {tree_id}; planning runs in the background")

    # Planning is asynchronous and takes minutes, so poll rather than block on
    # one long request that a proxy or client timeout could sever.
    deadline = time.monotonic() + args.timeout
    while True:
        if time.monotonic() > deadline:
            raise CellMCPError(
                f"'{tree_id}' still planning after {args.timeout:.0f}s. It may yet "
                f"finish — check with 'make logs-traj', then execute it with "
                f"'make replay-plan SIM={args.cell} TREE_ID={tree_id}'"
            )
        status = call_tool("get_plan_status", {"tree_id": tree_id},
                           args.host, args.traj_port, QUERY_TIMEOUT_S)
        state = status.get("status")
        if state == "done":
            break
        if state in ("failed", "unknown"):
            print(json.dumps(status, indent=2))
            raise CellMCPError(f"planning {state}: {status.get('error') or 'see logs-traj'}")
        time.sleep(args.poll_interval)

    print(f"[2/3] planned in {status.get('elapsed_s', '?')}s")
    if args.plan_only:
        plan = call_tool("get_calculated_plan", {"cell_id": args.cell, "tree_id": tree_id},
                         args.host, args.traj_port, QUERY_TIMEOUT_S)
        print(f"[3/3] plan-only: {plan.get('num_operations')} trajectories, not executed")
        print(f"      execute later with: make replay-plan SIM={args.cell} TREE_ID={tree_id}")
        return 0 if _succeeded(plan) else 1

    print(f"[3/3] executing {tree_id} on the cell ...")
    exec_args = {"cell_id": args.cell, "tree_id": tree_id}
    if model:
        exec_args["model"] = model     # reaches only the failure-recovery reasoner
    result = call_tool(
        "replay_plan", exec_args,
        args.host, args.port, args.timeout,
    )
    print(json.dumps(result, indent=2))
    return 0 if _succeeded(result) else 1


def cmd_replay_plan(args) -> int:
    """Execute an already-calculated plan without re-planning."""
    arguments = {"cell_id": args.cell}
    if args.tree_id:
        arguments["tree_id"] = args.tree_id
    if getattr(args, "model", None):
        # Replay is deterministic; the model is consulted only if it fails.
        arguments["model"] = args.model
    result = call_tool("replay_plan", arguments, args.host, args.port, args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if _succeeded(result) else 1


def cmd_replay(args) -> int:
    result = call_tool(
        "execute_behaviour_tree",
        {"behaviour_tree": {"tree_id": args.tree_id}, "replay_mode": True},
        args.host, args.port, args.timeout,
    )
    print(json.dumps(result, indent=2))
    return 0 if _succeeded(result) else 1


def cmd_recordings(args) -> int:
    result = call_tool(
        "list_recordings", {"cell_id": args.cell},
        args.host, args.port, args.timeout,
    )
    recordings = result.get("recordings") or []
    if not recordings:
        print("  No recordings found.")
        return 0
    for rec in recordings:
        tree_id = str(rec.get("tree_id", "?"))
        size = rec.get("size_bytes", 0)
        path = rec.get("path", "")
        print(f"  {tree_id:40s} {size:>10d} bytes  {path}")
    return 0


def cmd_call(args) -> int:
    """Escape hatch: invoke any cell tool with raw JSON arguments."""
    try:
        arguments = json.loads(args.args) if args.args else {}
    except ValueError as exc:
        print(f"ERROR: --args is not valid JSON: {exc}", file=sys.stderr)
        return 2
    result = call_tool(args.tool, arguments, args.host, args.port, args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if _succeeded(result) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cell_mcp.py",
        description="Call the Cell Agent's MCP endpoint (JSON-RPC over HTTP).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="agent host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="cell MCP port")
    parser.add_argument("--traj-port", type=int, default=DEFAULT_TRAJ_PORT,
                        help="trajectory agent MCP port")
    sub = parser.add_subparsers(dest="command", required=True)

    p_task = sub.add_parser("run-task", help="plan a task program, then execute it")
    p_task.add_argument("--cell", required=True)
    p_task.add_argument("--task", required=True)
    # No fixed list: every registered backend (built-in, bundled such as
    # roboballet, or a plugin) is selectable the same way, and the planner refuses
    # an unknown name with the valid ones (list_trajectory_backends).
    p_task.add_argument("--backend", default=None,
                        help="trajectory backend for this request (default: the stack's); "
                             "any registered name, e.g. ompl | ompl-compound | rrt | roboballet")
    p_task.add_argument("--backend-options", default=None,
                        help='options for the backend, as JSON, e.g. \'{"policy": "p1"}\' '
                             "for roboballet (list_trajectory_backends shows each "
                             "backend's options)")
    p_task.add_argument("--mode", default=None, choices=("classic", "coordinated", "stepwise"),
                        help="planning mode for this request (default: the stack's)")
    p_task.add_argument("--model", default=None,
                        help=("LLM for this task only: an alias (opus-5 | "
                              "opus-4.8 | sonnet-5 | haiku), a full model id, or "
                              "provider:model. Omit for the startup default. Does "
                              "not affect the motion planner — that is --backend"))
    p_task.add_argument("--plan-only", action="store_true",
                        help="stop after planning; do not move any robot")
    p_task.add_argument("--poll-interval", type=float, default=POLL_INTERVAL_S)
    p_task.add_argument("--timeout", type=float, default=TASK_TIMEOUT_S)
    p_task.set_defaults(func=cmd_run_task)

    p_rp = sub.add_parser("replay-plan", help="execute an already-planned tree (no re-planning)")
    p_rp.add_argument("--cell", required=True)
    p_rp.add_argument("--tree-id", help="plan id; omit for the latest plan")
    p_rp.add_argument("--model", default=None,
                      help=("LLM for the failure-recovery reasoner on this call "
                            "only; replay itself is deterministic"))
    p_rp.add_argument("--timeout", type=float, default=TASK_TIMEOUT_S)
    p_rp.set_defaults(func=cmd_replay_plan)

    p_replay = sub.add_parser("replay", help="replay a recorded BehaviourTree")
    p_replay.add_argument("--cell", required=False, help="accepted for symmetry; unused")
    p_replay.add_argument("--tree-id", required=True)
    p_replay.add_argument("--timeout", type=float, default=TASK_TIMEOUT_S)
    p_replay.set_defaults(func=cmd_replay)

    p_recs = sub.add_parser("recordings", help="list .rrd recordings for a cell")
    p_recs.add_argument("--cell", required=True)
    p_recs.add_argument("--timeout", type=float, default=QUERY_TIMEOUT_S)
    p_recs.set_defaults(func=cmd_recordings)

    p_call = sub.add_parser("call", help="invoke any cell tool with raw JSON args")
    p_call.add_argument("--tool", required=True)
    p_call.add_argument("--args", default="{}", help="tool arguments as a JSON object")
    p_call.add_argument("--timeout", type=float, default=TASK_TIMEOUT_S)
    p_call.set_defaults(func=cmd_call)

    args = parser.parse_args()
    try:
        return args.func(args)
    except CellMCPError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
