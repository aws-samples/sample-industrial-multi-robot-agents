"""Reading and writing a cell's task programs (the ``tasks`` kind, ``<program>.json``).

Separate from ``mcp_server`` for the same reason ``list_recording_files`` is
separate from the rerun recorder: persisting a program is a filesystem question,
not an agent-framework one. Importing the MCP server drags in the LLM stack, so
logic that lives there can only be exercised with the whole agent mocked —
which is how a writer came to emit a schema no reader accepted.

The schema is ``skills/orchestrator/run-welding-program/references/
task-json-format.md``: ``{program, cell_id, operations: [{id, type, position,
approach_vector, ...}]}``.
"""
from __future__ import annotations

import json

from agents.trajectory import plan_trace
from utils import cell_layout, cell_store


def save_program(cell_id: str, program: str, task: dict,
                 overwrite: bool = False) -> dict:
    """Write ``tasks/<program>.json`` for one cell. Runs nothing, moves nothing.

    ``program`` and ``cell_id`` inside the file are set from the arguments, so a
    program copied from another cell cannot keep claiming the cell it came from
    — the planner reads those fields.

    Args:
        cell_id:   Cell identifier.
        program:   Program name WITHOUT .json. It becomes a file name, so it is
                   restricted to letters, digits, '.', '_' and '-'.
        task:      The program object. ``operations`` must be a non-empty list.
        overwrite: Required to replace an existing program. Without it, an
                   existing name is REFUSED — a silent overwrite can destroy the
                   only copy of a hand-authored program.

    Returns:
        {"success", "cell_id", "program", "path", "num_operations",
         "overwritten", "commanded": False} — or {"success": False, "error"}.
    """
    if not program or not plan_trace.is_safe_id(program):
        return {"success": False, "cell_id": cell_id, "program": program,
                "error": ("program must be a plain identifier (letters, digits, "
                          ". _ -) — it becomes a file name under "
                          f"cells/{cell_id}/tasks/")}
    if not isinstance(task, dict):
        return {"success": False, "cell_id": cell_id, "program": program,
                "error": "task must be the program object"}
    operations = task.get("operations")
    if not isinstance(operations, list) or not operations:
        return {"success": False, "cell_id": cell_id, "program": program,
                "error": ("task['operations'] must be a non-empty array — see "
                          "skills/orchestrator/run-welding-program/references/"
                          "task-json-format.md")}

    # The path is REPORTED, never used to write: a task program is cell data, so it
    # is written by the agent that owns the cell (cell_store), and this is the name
    # that artefact has on that agent's disk.
    path = cell_layout.task_file(cell_id, program)
    name = f"{program}.json"
    try:
        existed = cell_store.read_text(cell_id, cell_store.KIND_TASKS,
                                       name) is not None
    except Exception as exc:  # noqa: BLE001
        # Unknown is not "absent": answering "absent" here would let an overwrite
        # happen without the caller having asked for one.
        return {"success": False, "cell_id": cell_id, "program": program,
                "error": f"could not check for an existing program: {exc}"}
    if existed and not overwrite:
        return {"success": False, "cell_id": cell_id, "program": program,
                "path": str(path), "error": (
                    f"task program '{program}' already exists in cell "
                    f"'{cell_id}'. Pass overwrite=True to replace it, or save "
                    f"under another name.")}

    body = {**task, "program": program, "cell_id": cell_id}
    try:
        # Trailing newline kept deliberately: programs are hand-edited and diffed.
        # The write is staged and renamed by the store, so a half-written program
        # is never visible.
        cell_store.write_text(cell_id, cell_store.KIND_TASKS, name,
                              json.dumps(body, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "cell_id": cell_id, "program": program,
                "path": str(path), "error": f"could not write program: {exc}"}

    return {"success": True, "cell_id": cell_id, "program": program,
            "path": str(path), "num_operations": len(operations),
            "overwritten": existed, "commanded": False}


def program_warnings(cell_id: str, task: dict) -> list:
    """What a reader of a task program should be told before planning it.

    * the program names another cell (``cell_id``) — a copied program whose
      weld points were placed for a different layout;
    * placeholder operations (:func:`is_placeholder`) — points that stand in for
      geometry nobody has measured (CAD placeholders, points moved off a synthetic
      grid): planned and executed like any weld, so they are named, never
      silently trusted.

    Never raises; an unreadable program has no warnings to give.
    """
    out: list = []
    if not isinstance(task, dict):
        return out
    declared = task.get("cell_id")
    if declared and str(declared) != str(cell_id):
        out.append(f"the program declares cell_id '{declared}' but is stored in cell "
                   f"'{cell_id}': check its weld points against this cell's layout")
    placeholders = [str(op.get("id")) for op in task.get("operations") or []
                    if is_placeholder(op)]
    if placeholders:
        out.append(f"{len(placeholders)} operation(s) are placeholders, not measured "
                   f"geometry: {', '.join(placeholders)}")
    return out


def is_placeholder(op) -> bool:
    """Whether an operation's point is a stand-in rather than measured geometry:
    marked ``placeholder: true``, or carrying the trace of having been moved by
    hand (``original_position`` / ``position_note``). The rule, not a list of ids,
    so it holds for any cell's programs."""
    if not isinstance(op, dict):
        return False
    if op.get("placeholder") is not None:
        return bool(op["placeholder"])
    return "original_position" in op or "position_note" in op


def normalized_program(cell_id: str, task: dict) -> "tuple[dict, list]":
    """``(task, changes)``: the program with its declared ``cell_id`` set to the
    cell it is stored in and every placeholder operation (:func:`is_placeholder`)
    marked ``placeholder: true`` explicitly — what a reader that does not know the
    rule needs. Nothing else is touched; ``changes`` says what was."""
    import copy
    out = copy.deepcopy(task)
    changes: list = []
    if out.get("cell_id") is not None and str(out["cell_id"]) != str(cell_id):
        changes.append(f"cell_id: {out['cell_id']!r} -> {cell_id!r}")
        out["cell_id"] = cell_id
    for op in out.get("operations") or []:
        if isinstance(op, dict) and is_placeholder(op) and op.get("placeholder") is None:
            op["placeholder"] = True
            changes.append(f"{op.get('id')}: placeholder: true")
    return out, changes
