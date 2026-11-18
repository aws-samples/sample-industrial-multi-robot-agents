"""How many times one arm may fail one operation in a planning run — enforced.

The recovery skill says "~2 changes per weld on that arm, then hand it to another
arm". Prose cannot hold a model to it: on welding_cell_2_small (bt_1793277501) robot2
spent six attempts and ~2 minutes on one via point (``W18_VIA``) that never
planned, and the weld then went in without it. This module turns the skill's
cap into a refusal:

* every failure ``run_trajectory_planner`` returns for an operation THIS arm is
  responsible for is counted against ``(robot_ns, operation family)``;
* once the count reaches the cap, the next request for that pair is refused
  before any solve, with ``reason_code: "retry_cap_reached"`` — whose recovery
  is to stop re-trying it there: reallocate the weld to another arm, plan the
  weld without the helper waypoint, or record it unreachable.

An operation FAMILY groups the helper legs the model invents around one weld:
``W18_VIA``, ``W18_VIA2`` and ``W18_VIA_3`` are one family, so renaming a via
point does not reset its budget. Welds themselves are their own family.

Not counted — the arm is not the cause, or the attempt never ran: another arm's
failure (``step_refused``, ``blocked_by_parked_arm``), a batch that stopped
before the request (``not_planned``), the step gate (``step_gate_refused``), the
operator's lock, and infrastructure faults (which have their own stop rule).

Only active inside a planning run (:func:`scope`, bound by ``agent.plan``);
outside one — a direct tool call, a test — nothing is counted or refused.
``PLAN_RETRY_CAP`` sets the cap (default 3 failures; 0 disables it).
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

REASON_CODE = "retry_cap_reached"
DEFAULT_CAP = 3

#: Failures that do not count against the arm that was asked to move.
NOT_COUNTED = frozenset({
    REASON_CODE, "not_planned", "step_refused", "step_gate_refused",
    "blocked_by_parked_arm", "assignment_violation", "duplicate_robot_request",
    "self_collision_unavailable", "self_collision_check_error", "leg_check_error",
    "inter_arm_check_error", "backend_contract_violation", "tool_error", "planner_error",
})

_COUNTS: ContextVar[Optional[dict]] = ContextVar("plan_retry_counts", default=None)

_HELPER = re.compile(r"^(?P<base>.+?)_(?P<kind>VIA|SAFE|CLEAR|PARK|STAGE)(?:_?\d+)?$",
                     re.IGNORECASE)


@contextmanager
def scope():
    """A fresh failure count for one planning run."""
    token = _COUNTS.set({})
    try:
        yield
    finally:
        _COUNTS.reset(token)


def cap() -> int:
    try:
        return max(0, int(os.environ.get("PLAN_RETRY_CAP", DEFAULT_CAP)))
    except ValueError:
        return DEFAULT_CAP


def family(operation_id: Optional[str]) -> Optional[str]:
    """``W18_VIA2`` → ``W18_VIA``; a plain operation id is its own family."""
    if not operation_id:
        return None
    m = _HELPER.match(str(operation_id))
    if not m:
        return str(operation_id)
    return f"{m.group('base')}_{m.group('kind').upper()}"


def _key(robot_ns, operation_id):
    fam = family(operation_id)
    return (str(robot_ns), fam) if robot_ns and fam else None


def attempts(robot_ns: str, operation_id: str) -> int:
    counts = _COUNTS.get()
    k = _key(robot_ns, operation_id)
    return 0 if counts is None or k is None else int(counts.get(k, 0))


def refusal(request: dict) -> Optional[dict]:
    """The failure to return INSTEAD of planning ``request``, or None to plan it."""
    counts, limit = _COUNTS.get(), cap()
    if counts is None or not limit or not isinstance(request, dict):
        return None
    ns, op = request.get("robot_ns"), request.get("operation_id")
    k = _key(ns, op)
    if k is None or counts.get(k, 0) < limit:
        return None
    n = counts[k]
    helper = _HELPER.match(str(op)) is not None
    what = ("this helper waypoint (and every renamed copy of it)" if helper
            else "this operation")
    return {
        "robot_ns": ns, "operation_id": op, "reason_code": REASON_CODE,
        "attempts": n, "cap": limit,
        "error": (f"{ns} has already failed {what} {n} times in this run "
                  f"(cap {limit}): it is not planned again on {ns}. "
                  + ("Plan the weld it was meant to help WITHOUT it, or hand the weld "
                     "to another arm (reallocate_operation)." if helper else
                     "Hand it to another arm with reallocate_operation "
                     "(tried_robots must include " + str(ns) + "), or, when every arm "
                     "has failed it, record it unreachable and continue.")),
    }


def record(failures) -> None:
    """Count each failure the arm itself is responsible for."""
    counts = _COUNTS.get()
    if counts is None:
        return
    for f in failures or []:
        if not isinstance(f, dict) or f.get("reason_code") in NOT_COUNTED:
            continue
        k = _key(f.get("robot_ns"), f.get("operation_id"))
        if k is not None:
            counts[k] = counts.get(k, 0) + 1
