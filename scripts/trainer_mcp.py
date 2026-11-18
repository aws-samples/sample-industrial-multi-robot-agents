#!/usr/bin/env python3
"""
Drive the policy trainer's MCP endpoint (:8894/mcp) from the command line.

The ``make train*`` / ``make policy*`` targets are one-liners over this module.
It owns the JSON-RPC envelope (``utils/mcp_http.py``) once, like
``scripts/cell_mcp.py`` does for the cell controller. Standard library only.

Usage:
    python3 scripts/trainer_mcp.py status
    python3 scripts/trainer_mcp.py start   --train welding_cell_2_small --eval welding_cell_2_small \\
                                           --steps 2000 --job roboballet-test1 [--dry-run] [--watch]
    python3 scripts/trainer_mcp.py job     --job roboballet-test1
    python3 scripts/trainer_mcp.py watch   --job roboballet-test1
    python3 scripts/trainer_mcp.py logs    --job roboballet-test1 --lines 50
    python3 scripts/trainer_mcp.py stop    --job roboballet-test1
    python3 scripts/trainer_mcp.py resume  --job roboballet-test1
    python3 scripts/trainer_mcp.py jobs
    python3 scripts/trainer_mcp.py policies
    python3 scripts/trainer_mcp.py import  --job roboballet-test1 [--name p1] [--promote]
    python3 scripts/trainer_mcp.py evaluate --policy p1 --cell welding_cell_2_small [--mode assigned]
    python3 scripts/trainer_mcp.py promote --policy p1

Exit status is 0 when the tool reports ``ok`` (and, for ``watch``, the job
completed), 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PORT = 8894
ACTIVE = ("InProgress", "Stopping")


def call(host: str, port: int, tool: str, args: dict, timeout: float = 120.0) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": args}}).encode()
    req = urllib.request.Request(f"http://{host}:{port}/mcp", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        reply = json.loads(e.read() or b"{}")
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "error": f"policy trainer not reachable at {host}:{port} ({e}) "
                                      "— start it with `make trainer`"}
    if "error" in reply:
        return {"ok": False, "error": reply["error"].get("message", reply["error"])}
    return json.loads(reply["result"]["content"][0]["text"])


def _print(out: dict) -> None:
    print(json.dumps(out, indent=2, default=str))


def _cells(value: str) -> list[str]:
    return [c.strip() for c in (value or "").split(",") if c.strip()]


def watch(host: str, port: int, job: str, every: float) -> dict:
    """Poll until the job leaves InProgress/Stopping, printing one line per change."""
    last = None
    while True:
        out = call(host, port, "get_training_job", {"job_name": job})
        if not out.get("ok"):
            return out
        m = out.get("metrics") or {}
        line = (f"{out['status']:<10} {out.get('detail') or '':<14} "
                + "  ".join(f"{k}={v:.4g}" for k, v in sorted(m.items())
                            if isinstance(v, (int, float))))
        if line != last:
            print(time.strftime("%H:%M:%S"), line, flush=True)
            last = line
        if out["status"] not in ACTIVE:
            return out
        time.sleep(every)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("jobs")
    sub.add_parser("policies")
    s = sub.add_parser("start")
    s.add_argument("--train", required=True, help="comma-separated training cells")
    s.add_argument("--eval", default="", help="comma-separated evaluation cells")
    s.add_argument("--steps", type=int)
    s.add_argument("--hp", default="", help='extra hyperparameters as JSON, e.g. {"n_envs":4}')
    s.add_argument("--job")
    s.add_argument("--instance-type")
    s.add_argument("--max-runtime-s", type=int)
    s.add_argument("--no-auto-import", action="store_true")
    s.add_argument("--promote", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--watch", action="store_true")
    s.add_argument("--every", type=float, default=10.0)
    for name in ("job", "watch", "stop", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--job", required=True)
        p.add_argument("--every", type=float, default=10.0)
    p = sub.add_parser("logs")
    p.add_argument("--job", required=True)
    p.add_argument("--lines", type=int, default=50)
    p = sub.add_parser("import")
    p.add_argument("--job", required=True)
    p.add_argument("--name")
    p.add_argument("--promote", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--policy", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--mode", default="free", choices=("free", "assigned"))
    p.add_argument("--episodes", type=int, default=3)
    p = sub.add_parser("promote")
    p.add_argument("--policy", required=True)
    a = ap.parse_args(argv)
    h, port = a.host, a.port

    if a.cmd == "status":
        out = call(h, port, "trainer_status", {})
    elif a.cmd == "jobs":
        out = call(h, port, "list_training_jobs", {})
    elif a.cmd == "policies":
        out = call(h, port, "list_policies", {})
    elif a.cmd == "start":
        hp = json.loads(a.hp) if a.hp else {}
        if a.steps:
            hp["total_steps"] = a.steps
        args = {"train_cells": _cells(a.train), "eval_cells": _cells(a.eval),
                "hyperparameters": hp, "auto_import": not a.no_auto_import,
                "promote": a.promote, "dry_run": a.dry_run}
        for key, val in (("job_name", a.job), ("instance_type", a.instance_type),
                         ("max_runtime_s", a.max_runtime_s)):
            if val:
                args[key] = val
        out = call(h, port, "start_training", args, timeout=600.0)
        _print(out)
        if not out.get("ok") or a.dry_run or not a.watch:
            return 0 if out.get("ok") else 1
        out = watch(h, port, out["job_name"], a.every)
        _print(out)
        return 0 if out.get("status") == "Completed" else 1
    elif a.cmd == "job":
        out = call(h, port, "get_training_job", {"job_name": a.job})
    elif a.cmd == "watch":
        out = watch(h, port, a.job, a.every)
        _print(out)
        return 0 if out.get("status") == "Completed" else 1
    elif a.cmd == "stop":
        out = call(h, port, "stop_training_job", {"job_name": a.job})
    elif a.cmd == "resume":
        out = call(h, port, "resume_training_job", {"job_name": a.job}, timeout=600.0)
    elif a.cmd == "logs":
        out = call(h, port, "get_training_logs", {"job_name": a.job, "lines": a.lines})
        if out.get("ok"):
            print("\n".join(out["lines"]))
            return 0
    elif a.cmd == "import":
        args = {"job_name": a.job, "promote": a.promote}
        if a.name:
            args["name"] = a.name
        out = call(h, port, "import_policy", args, timeout=600.0)
    elif a.cmd == "evaluate":
        out = call(h, port, "evaluate_policy", {"name": a.policy, "cell_id": a.cell,
                                                "mode": a.mode, "episodes": a.episodes},
                   timeout=600.0)
    else:  # promote
        out = call(h, port, "promote_policy", {"name": a.policy})
    _print(out)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
