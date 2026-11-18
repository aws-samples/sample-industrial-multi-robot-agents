#!/usr/bin/env python3
"""Audit (and with --write, normalise) a cell's task programs. Any cell.

Reports, per ``agentic_planner/tasks/*.json``:
  * a ``cell_id`` that is not the cell the program is stored in;
  * placeholder operations — points that stand in for unmeasured geometry
    (``agents.trajectory.task_programs.is_placeholder``) — and marks them
    ``placeholder: true`` explicitly.

Dry-run by default: prints what would change. ``--write`` rewrites the files
(task programs are hand-authored; review the dry run first).

  python3 scripts/audit_task_programs.py --cell welding_cell_2_small [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.trajectory.task_programs import normalized_program  # noqa: E402
from utils import cell_layout  # noqa: E402


def minimal_rewrite(text: str, task: dict, fixed: dict) -> str:
    """``text`` with only the changed fields edited in place (programs are
    hand-formatted and diffed), or the whole program re-dumped when an in-place
    edit cannot be shown to produce exactly ``fixed``."""
    out = text
    if fixed.get("cell_id") != task.get("cell_id"):
        out = re.sub(r'("cell_id"\s*:\s*)"[^"]*"', lambda m: f'{m.group(1)}"{fixed["cell_id"]}"',
                     out, count=1)
    for old, new in zip(task.get("operations") or [], fixed.get("operations") or []):
        if new.get("placeholder") is True and "placeholder" not in old:
            pat = re.compile(r'("id"\s*:\s*"' + re.escape(str(old.get("id"))) + r'"\s*,)')
            out, n = pat.subn(r'\1 "placeholder": true,', out, count=1)
            if n != 1:
                break
    try:
        if json.loads(out) == fixed:
            return out
    except ValueError:
        pass
    return json.dumps(fixed, indent=2, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="cell id")
    ap.add_argument("--cell-base", default=None,
                    help="cells directory (default: $CELL_BASE_PATH, else ./cells)")
    ap.add_argument("--write", action="store_true", help="rewrite the programs")
    args = ap.parse_args(argv)
    base = args.cell_base or os.environ.get("CELL_BASE_PATH") or "cells"
    tasks = Path(cell_layout.tasks_dir(args.cell, base))
    if not tasks.is_dir():
        print(f"no tasks directory for {args.cell}: {tasks}")
        return 1
    total = 0
    unreadable = 0
    for path in sorted(tasks.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
            task = json.loads(text)
        except (OSError, ValueError) as exc:
            print(f"{path.name}: UNREADABLE ({exc})")
            unreadable += 1
            continue
        fixed, changes = normalized_program(args.cell, task)
        if not changes:
            print(f"{path.name}: ok")
            continue
        total += len(changes)
        print(f"{path.name}:")
        for c in changes:
            print(f"  {c}")
        if args.write:
            _write_atomic(path, minimal_rewrite(text, task, fixed))
            print("  written")
    if total and not args.write:
        print(f"{total} change(s); run again with --write to apply")
    # Non-zero while anything is pending or unreadable: usable as a CI gate.
    return 1 if unreadable or (total and not args.write) else 0


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and rename: a crash never leaves half a program."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
