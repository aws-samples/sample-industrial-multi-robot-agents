"""What the operator asked of ONE planning run, and how it is enforced.

A ``plan_trajectories`` request may carry four directives, kept in the run's
``constraints["operator"]`` so a paused run resumes under the same ones:

* ``assignments`` — ``{operation_id: robot}``. ENFORCED by code, not by asking:
  every assignment becomes the operation's ``assigned_robot`` (a hard lock the
  allocator already honours), ``run_mrta_pipeline`` re-applies the locks to the
  operations it is handed (so the model cannot drop one), the motion planner
  refuses to plan a locked operation on any other arm (``assignment_violation``),
  and ``reallocate_operation`` refuses to move one.
* ``skill_priorities`` — ``{skill: priority | {"priority", "enabled"}}`` for this
  run only, layered over the cell's skills.yaml (utils/skill_registry.py). The
  prompt's catalog is ordered by them and ``load_skill`` refuses a skill this run
  switched off; the cell's file is never written.
* ``instructions`` — free text. Code cannot verify prose, so it is put in front
  of the model as a binding section of the task, and the model must say in its
  final message which instruction it could not meet and why.
* ``run_skills`` — temporary skills for this run only (utils/skill_registry.py
  ``normalise_run_skills``): in its catalog and loadable with ``load_skill``,
  written nowhere. Guidance, like any skill — the gates are code, and a
  temporary skill can neither replace an existing one nor be required.

Everything is validated when the request is SUBMITTED (:func:`normalise`): an
unknown operation, an unknown robot or an unknown skill refuses the request
rather than being silently ignored.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

OPERATOR_KEY = "operator"

# operation id -> robot, for the run in flight.
_LOCKS: ContextVar[dict] = ContextVar("operator_locks", default={})


class DirectiveError(ValueError):
    """A directive names something that does not exist, or is malformed."""


@contextmanager
def scope():
    """A fresh, empty lock table for one planning run."""
    token = _LOCKS.set({})
    try:
        yield
    finally:
        _LOCKS.reset(token)


def set_locks(locks: dict) -> None:
    _LOCKS.set({str(k): str(v) for k, v in (locks or {}).items()})


def current_locks() -> dict:
    return _LOCKS.get()


def _id_pattern(op_id: str):
    # "W01" also names the legs the agent derives from it — "R1_W01", "W01_VIA" —
    # but not "W010": the id must not continue into another letter or digit.
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(op_id)}(?![A-Za-z0-9])")


def locked_robot(operation_id: Optional[str]) -> Optional[tuple[str, str]]:
    """``(locked_op_id, robot)`` if ``operation_id`` is (derived from) a locked
    operation, else None."""
    if not operation_id:
        return None
    oid = str(operation_id)
    locks = current_locks()
    if oid in locks:
        return oid, locks[oid]
    for locked, robot in locks.items():
        if _id_pattern(locked).search(oid):
            return locked, robot
    return None


def apply_locks(operations: list) -> list:
    """Copies of ``operations`` with every run lock written as ``assigned_robot``."""
    locks = current_locks()
    if not locks:
        return operations
    out = []
    for op in operations or []:
        if isinstance(op, dict) and str(op.get("id")) in locks:
            op = {**op, "assigned_robot": locks[str(op.get("id"))]}
        out.append(op)
    return out


def violation(motion_request: dict) -> Optional[dict]:
    """The failure for a motion request that plans a locked operation on the
    wrong arm, or None when it is allowed."""
    hit = locked_robot(motion_request.get("operation_id"))
    if not hit:
        return None
    locked, robot = hit
    ns = motion_request.get("robot_ns")
    if ns == robot:
        return None
    return {
        "robot_ns": ns,
        "operation_id": motion_request.get("operation_id"),
        "reason_code": "assignment_violation",
        "error": (f"'{motion_request.get('operation_id')}' belongs to operation "
                  f"'{locked}', which the operator ASSIGNED to {robot}; it cannot be "
                  f"planned on {ns}. Plan it on {robot} (re-sequence, add a via "
                  f"point, or move the other arm out of the way) — do not reassign it."),
    }


def normalise(directives: Optional[dict], operations: list, robots: list,
              skill_agent: str) -> dict:
    """Validate a request's directives; return their canonical form.

    ``robots`` are the cell's robot ids, ``skill_agent`` the skill set the run's
    planning mode reads. Raises :class:`DirectiveError` on anything unknown.
    """
    from utils import skill_registry

    d = dict(directives or {})
    unknown = set(d) - {"assignments", "skill_priorities", "instructions", "run_skills"}
    if unknown:
        raise DirectiveError(f"unknown directive(s) {sorted(unknown)}")
    out: dict = {}

    assignments = d.get("assignments") or {}
    if not isinstance(assignments, dict):
        raise DirectiveError("assignments must be {operation_id: robot}")
    op_ids = {str(op.get("id")) for op in operations or [] if isinstance(op, dict)}
    robot_set = {str(r) for r in robots}
    clean: dict = {}
    for op_id, robot in assignments.items():
        if str(op_id) not in op_ids:
            raise DirectiveError(f"assignment for unknown operation {op_id!r} "
                                 f"(the request has {sorted(op_ids)})")
        if str(robot) not in robot_set:
            raise DirectiveError(f"{op_id!r} assigned to unknown robot {robot!r} "
                                 f"(the cell has {sorted(robot_set)})")
        clean[str(op_id)] = str(robot)
    if clean:
        out["assignments"] = clean

    try:
        temporary = skill_registry.normalise_run_skills(skill_agent, d.get("run_skills"))
    except skill_registry.SkillError as exc:
        raise DirectiveError(str(exc)) from None
    if temporary:
        out["run_skills"] = temporary

    priorities = d.get("skill_priorities") or {}
    if not isinstance(priorities, dict):
        raise DirectiveError("skill_priorities must be {skill: priority | "
                             "{priority, enabled}}")
    try:
        # A priority may name this run's own temporary skills too.
        with skill_registry.bind_run_skills({skill_agent: temporary}):
            runs = skill_registry.normalise_run_overrides(skill_agent, priorities)
    except skill_registry.SkillError as exc:
        raise DirectiveError(str(exc)) from None
    if runs:
        out["skill_priorities"] = runs.get(skill_agent, {})

    text = d.get("instructions")
    if text is not None and not isinstance(text, str):
        raise DirectiveError("instructions must be text")
    if text and text.strip():
        out["instructions"] = text.strip()
    return out


def merge_assignments(operations: list, directives: dict) -> list:
    """The request's operations with its assignments written in as hard locks."""
    assignments = (directives or {}).get("assignments") or {}
    if not assignments:
        return operations
    return [({**op, "assigned_robot": assignments[str(op.get("id"))]}
             if isinstance(op, dict) and str(op.get("id")) in assignments else op)
            for op in operations]


def all_locks(operations: list) -> dict:
    """Every hard lock in the run: the request's and the task file's own."""
    return {str(op["id"]): str(op["assigned_robot"]) for op in operations or []
            if isinstance(op, dict) and op.get("id") is not None
            and op.get("assigned_robot")}


def prompt_section(directives: dict, locks: dict) -> str:
    """The operator's directives as the model reads them, at the top of the task."""
    if not directives and not locks:
        return ""
    lines = ["OPERATOR DIRECTIVES FOR THIS RUN — binding. Do not trade them away for "
             "cycle time or convenience."]
    if locks:
        lines.append("- Robot assignments (ENFORCED: planning a locked operation on "
                     "another arm is refused, and so is reallocating it):")
        lines += [f"    {op} -> {robot}" for op, robot in sorted(locks.items())]
    sp = directives.get("skill_priorities") or {}
    if sp:
        lines.append("- Skill priorities set for this run (the catalog below is "
                     "already ordered by them; a disabled skill cannot be loaded):")
        for name, entry in sorted(sp.items()):
            bits = [f"priority {entry['priority']}"] if "priority" in entry else []
            if entry.get("enabled") is False:
                bits.append("DISABLED")
            elif entry.get("enabled") is True:
                bits.append("enabled")
            lines.append(f"    {name}: {', '.join(bits)}")
    if directives.get("run_skills"):
        lines.append("- Temporary skills the operator added for this run (in the "
                     "catalog below, marked temporary; load_skill them when they "
                     "apply). They are guidance: they do not relax any collision "
                     "check, required skill or assignment:")
        lines += [f"    {t['name']} (priority {t['priority']}): {t['description']}"
                  for t in directives["run_skills"]]
    if directives.get("instructions"):
        lines.append("- Instructions from the operator (follow every one; if one "
                     "cannot be met, say which and why in your final message):")
        lines += [f"    {ln}" for ln in directives["instructions"].splitlines()]
    return "\n".join(lines) + "\n\n"
