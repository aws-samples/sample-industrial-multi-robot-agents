"""
Minimal JSON-RPC-over-HTTP bridge for FastMCP servers.

FastMCP's ``sse_app()`` exposes the MCP SSE transport (``/sse`` + ``/messages``),
which requires a full initialize handshake. Several internal callers, however,
talk to the agents with a single plain JSON-RPC ``POST`` and expect a plain JSON
reply:

    request:  {"jsonrpc":"2.0","id":1,"method":"tools/call",
               "params":{"name":<tool>,"arguments":{...}}}
    response: {"jsonrpc":"2.0","id":1,
               "result":{"content":[{"type":"text","text":"<json>"}]}}

Callers using this contract:
  - orchestrator/tools/trajectory_client.py  -> trajectory-agent :8890/mcp
  - orchestrator cell client                  -> cell-controller :8892/mcp
  - Makefile `task-mcp` / `replay` targets    -> :8892/mcp

``add_jsonrpc_mcp_route`` attaches a ``/mcp`` endpoint implementing this contract
by dispatching through the FastMCP tool registry, so any registered tool is
callable without a hand-maintained dispatch table.
"""
from __future__ import annotations

import json
import uuid

from starlette.requests import Request
from starlette.responses import JSONResponse


def add_jsonrpc_mcp_route(app, mcp, path: str = "/mcp") -> None:
    """Attach a plain JSON-RPC ``tools/call`` endpoint to a Starlette ``app``."""

    async def _handle(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "Parse error"}},
                status_code=400,
            )

        req_id = body.get("id")
        method = body.get("method")
        params = body.get("params") or {}

        # ── Streamable-HTTP MCP handshake (so native MCP clients can use
        # this endpoint via `url` instead of SSE). Requests get application/json
        # responses (spec-compliant); notifications get 202 Accepted. ──────────
        if method == "initialize":
            client_proto = (params.get("protocolVersion")
                            or "2024-11-05")
            result = {
                "protocolVersion": client_proto,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": getattr(mcp, "name", "mcp-server"),
                               "version": "1.0.0"},
            }
            headers = {"Mcp-Session-Id": uuid.uuid4().hex}
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": result},
                headers=headers,
            )

        if method == "ping":
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})

        if method and method.startswith("notifications/"):
            # Notifications carry no id and expect no body — just acknowledge.
            return JSONResponse({}, status_code=202)

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            try:
                blocks = await mcp.call_tool(name, arguments)
            except Exception as exc:  # tool raised or unknown tool
                import traceback
                tb = traceback.format_exc()
                print(f"[mcp_http] tool '{name}' failed:\n{tb}", flush=True)
                return JSONResponse(
                    {"jsonrpc": "2.0", "id": req_id,
                     "error": {"code": -32603, "message": str(exc), "data": tb}},
                    status_code=500,
                )

            content = []
            for block in blocks:
                text = getattr(block, "text", None)
                content.append({
                    "type": "text",
                    "text": text if text is not None else json.dumps(block, default=str),
                })
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": {"content": content}}
            )

        if method == "tools/list":
            tools = await mcp.list_tools()
            listed = []
            for t in tools:
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
                entry = {"name": t.name, "description": getattr(t, "description", "")}
                if schema is not None:
                    entry["inputSchema"] = schema
                listed.append(entry)
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": {"tools": listed}}
            )

        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id,
             "error": {"code": -32601, "message": f"Method not found: {method}"}},
            status_code=404,
        )

    app.add_route(path, _handle, methods=["POST"])
