"""The orchestrator's cell-data MCP: one owner for everything derived in a cell.

The orchestrator is the process that holds ``/cells`` read-write. Until now it was
not the only one — the trajectory planner mounted the same volume and wrote plans,
traces, events, quality logs, the warm-start index and task programs straight to
it, and read obstacles.yaml and robots.yaml off it. Two writers on one tree is the
shape of the bug the dashboard already had to be moved off (see
``tests/test_dashboard_asks_mcp_not_the_volume.py``): nothing is wrong until the
two disagree, and then nothing says which one is right.

So the planner asks, and this is what it asks. Five tools over the fixed
``(kind, name)`` vocabulary in :mod:`utils.cell_store` — read, write, append,
list, delete — plus a health tool. Deliberately byte-level rather than one typed
tool per artefact family: the *schemas* (a BehaviourTree, a plan trace, an
obstacle model) belong to the modules that produce and consume them, and lifting
fifteen of them into this file would put two owners on every schema to end up with
one owner per file. What this owns is the tree: which names exist, where they
resolve, that a name cannot walk out of the cell, and that a write lands
atomically.

Not served here: the scene bundle (URDF/SRDF/meshes). It is shared immutable
*input*, its mesh URIs are resolved by processes outside this repo, and a cell is
33–973 MB — see :mod:`utils.cell_store` for why that stays a read-only mount.

Runs in the orchestrator's own container on ``ORCH_MCP_PORT`` (8893), started as a
thread by ``agents/orchestrator/server.py`` so the agent that owns the volume and
the endpoint that serves it cannot be scheduled apart or restarted separately.
"""
from __future__ import annotations

import threading

from mcp.server.fastmcp import FastMCP

from utils import cell_store

mcp = FastMCP(
    "cell-data",
    instructions=(
        "Derived data for a robot cell, owned by the orchestrator.\n\n"
        "KINDS: config (robots.yaml, obstacles.yaml, operations.yaml), manifest "
        "(cell.yaml), tasks (<program>.json), plans (<tree_id>.json), logs (run "
        "logs, plantrace_*.json, planevents_*.jsonl, <tree_id>.json quality "
        "reports), recordings (trajectory_index.json).\n\n"
        "cell_read/cell_write/cell_append/cell_list/cell_delete take a cell_id, a "
        "kind and a plain artefact name. Names are files, never paths: no "
        "separators, no '..'.\n\n"
        "The cell's scene bundle (URDF, SRDF, meshes, scene.yaml) is NOT here — it "
        "is a read-only mount shared with every service, because mesh URIs are "
        "baked into the published robot_description."
    ),
)


def _refuse(exc: Exception) -> dict:
    """A refusal that names itself, so a caller cannot read it as an empty answer."""
    return {"success": False, "error": str(exc)}


@mcp.tool()
def cell_read(cell_id: str, kind: str, name: str) -> dict:
    """Read one derived artefact of a cell.

    Returns ``{success, found, text}``. ``found=false`` means the artefact does
    not exist — which is a real answer a caller may degrade on. A transport or
    permission failure is an ERROR (``success=false``), never ``found=false``:
    "there are no obstacles" and "I could not read the obstacles" must not arrive
    as the same sentence.

    Args:
        cell_id: Cell identifier.
        kind: One of config, manifest, tasks, plans, logs, recordings.
        name: Plain file name within that kind (e.g. ``obstacles.yaml``).
    """
    try:
        text = _local_read(cell_id, kind, name)
    except Exception as exc:  # noqa: BLE001
        return _refuse(exc)
    return {"success": True, "cell_id": cell_id, "kind": kind, "name": name,
            "found": text is not None, "text": text}


def _local_read(cell_id: str, kind: str, name: str):
    """Read from this process's own disk, whatever CELL_DATA_MODE says.

    The server side is by definition the local side: if the orchestrator ever ran
    with ``CELL_DATA_MODE=orchestrator`` (a misconfiguration, or a second
    orchestrator pointed at this one) a mode-respecting read would call *itself*
    and the two would loop until the timeout. Resolving through cell_layout here
    makes that impossible.
    """
    path = cell_store.local_path(cell_id, kind, name)
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


@mcp.tool()
def cell_write(cell_id: str, kind: str, name: str, text: str) -> dict:
    """Replace one derived artefact of a cell, atomically.

    The write goes to a temp file and is renamed into place, so a reader polling
    while a plan is being saved sees either the old plan or the new one.

    Args:
        cell_id: Cell identifier.
        kind: One of config, manifest, tasks, plans, logs, recordings.
        name: Plain file name within that kind.
        text: Full new contents.
    """
    try:
        cell_store.write_text_local(cell_id, kind, name, text)
    except Exception as exc:  # noqa: BLE001
        return _refuse(exc)
    return {"success": True, "cell_id": cell_id, "kind": kind, "name": name,
            "written": True, "bytes": len(text)}


@mcp.tool()
def cell_append(cell_id: str, kind: str, name: str, text: str) -> dict:
    """Append to one derived artefact, creating it if absent.

    Its own verb because the plan-event stream is appended to line by line while a
    run is in flight; a read-modify-write would lose events to concurrent writers,
    and the live view would show a run with holes in it.

    Args:
        cell_id: Cell identifier.
        kind: One of config, manifest, tasks, plans, logs, recordings.
        name: Plain file name within that kind.
        text: Text to append — include the trailing newline you want.
    """
    try:
        cell_store.append_text_local(cell_id, kind, name, text)
    except Exception as exc:  # noqa: BLE001
        return _refuse(exc)
    return {"success": True, "cell_id": cell_id, "kind": kind, "name": name,
            "appended": True, "bytes": len(text)}


@mcp.tool()
def cell_list(cell_id: str, kind: str, prefix: str = "",
              suffix: str = "") -> dict:
    """List the artefacts of one kind: ``[{name, size_bytes, mtime_s, path}, …]``.

    Metadata only, name-sorted. A caller that wants contents reads them by name —
    listing 200 plans must not mean transporting 200 plans.

    Args:
        cell_id: Cell identifier.
        kind: tasks, plans, logs or recordings (config and manifest are single
            files with a read/write asymmetry, so they are not listable).
        prefix: Optional name prefix filter (e.g. ``planevents_``).
        suffix: Optional name suffix filter (e.g. ``.json``).
    """
    try:
        entries = cell_store.list_entries_local(cell_id, kind, prefix=prefix,
                                                suffix=suffix)
    except Exception as exc:  # noqa: BLE001
        return _refuse(exc)
    return {"success": True, "cell_id": cell_id, "kind": kind,
            "count": len(entries), "entries": entries}


@mcp.tool()
def cell_summarize(cell_id: str, kind: str, fields: list[str],
                   count_fields: list[str] | None = None,
                   prefix: str = "", suffix: str = ".json",
                   limit: int = 200) -> dict:
    """Pick a few fields out of every JSON artefact of one kind, server-side.

    The listings a picker renders (saved plans, recorded traces) each keep their
    metadata *inside* the artefact, and the artefacts are the expensive ones: a
    plan carries every precomputed waypoint. Transporting two hundred plans to
    read two hundred ``status`` fields would make the plan list cost more than
    planning. So the reading happens where the bytes already are, and only the
    named fields travel.

    Still schema-free: the CALLER names the fields, in its own vocabulary. This
    server knows that artefacts of these kinds are JSON objects and nothing else
    about them.

    Args:
        cell_id: Cell identifier.
        kind: tasks, plans, logs or recordings.
        fields: Top-level or dotted keys whose values to return (``run.trace_id``).
        count_fields: Keys whose *length* to return instead of their value, under
            ``<key>_count`` — for arrays a picker only counts (``points``).
        prefix: Optional name prefix filter (e.g. ``plantrace_``).
        suffix: Name suffix filter, ``.json`` by default.
        limit: Most recently modified artefacts to read, newest first.
    """
    try:
        rows = cell_store.summarize_local(
            cell_id, kind, list(fields or []),
            count_fields=list(count_fields or []),
            prefix=prefix, suffix=suffix, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return _refuse(exc)
    return {"success": True, "cell_id": cell_id, "kind": kind,
            "count": len(rows), "entries": rows}


@mcp.tool()
def cell_delete(cell_id: str, kind: str, name: str) -> dict:
    """Delete one derived artefact. ``deleted=false`` means it was not there.

    Args:
        cell_id: Cell identifier.
        kind: One of config, manifest, tasks, plans, logs, recordings.
        name: Plain file name within that kind.
    """
    try:
        gone = cell_store.delete_local(cell_id, kind, name)
    except Exception as exc:  # noqa: BLE001
        return _refuse(exc)
    return {"success": True, "cell_id": cell_id, "kind": kind, "name": name,
            "deleted": bool(gone)}


@mcp.tool()
def cell_data_health() -> dict:
    """Where this server keeps cell data, for a planner checking its wiring."""
    from utils import cell_layout
    return {"success": True, "service": "cell-data",
            "cells_root": str(cell_layout.cells_root()),
            "kinds": list(cell_store.KINDS)}


# ─── Serving ──────────────────────────────────────────────────────────────────

def build_app():
    """The Starlette app: MCP SSE plus the plain JSON-RPC ``/mcp`` route.

    Same two transports as the other agents expose, from the same helper, so a
    native MCP client and this repo's own one-POST callers reach the same tools.
    """
    from utils.mcp_http import add_jsonrpc_mcp_route

    app = mcp.sse_app()
    add_jsonrpc_mcp_route(app, mcp)
    return app


def serve_in_thread(port: int) -> threading.Thread:
    """Run the cell-data MCP beside the orchestrator's Flask app.

    A thread and not a service: this endpoint's whole job is to be the voice of
    the process that holds the volume. Split into its own container and there are
    two processes writing ``/cells`` again — which is the thing being fixed.

    Daemon, so it can never hold the orchestrator open on shutdown; failures are
    printed and the thread ends, because an orchestrator that stops answering HTTP
    because its MCP port is taken would be a worse outcome than a planner that
    refuses to plan (and says why).
    """
    import uvicorn

    def _run() -> None:
        try:
            uvicorn.run(build_app(), host="0.0.0.0", port=port,
                        log_level="warning")
        except Exception as exc:  # noqa: BLE001
            print(f"[cell-data] MCP server stopped: {exc}", flush=True)

    thread = threading.Thread(target=_run, name="cell-data-mcp", daemon=True)
    thread.start()
    return thread
