"""Which trajectory BACKEND and planning MODE a request runs with — one owner.

Both are chosen per request, over MCP (``plan_trajectories``, ``replan_operation``,
and the orchestrator's ``/a2a/plan`` / ``/a2a/run`` that call them). Precedence:

1. the request — its explicit ``backend`` / ``mode`` argument, else the same
   choice carried in its ``constraints`` (``trajectory_backend`` /
   ``planning_mode``, which is where a task program puts it);
2. the stack default — ``TRAJECTORY_BACKEND`` / ``PLANNING_MODE`` as the stack was
   started with (``make sim TRAJECTORY_BACKEND=… PLANNING_MODE=…``);
3. the code default — ``ompl`` / ``classic``.

Per-request **backend options** (``backend_options``, e.g. ``roboballet``'s
``{"policy": "<name>"}``) come only from the request — its argument, else
``constraints.backend_options`` — and are validated against the CHOSEN backend
(``planner_backend.validate_backend_options``): an option it does not take, or
a value its own check refuses, refuses the request.

A value the REQUEST names is validated and refused if unknown: a typo must not
quietly plan with something else. Every answer says where each value came from,
so a caller can see that a default was applied rather than assume its own.

Standard library only (plus ``planner_backend``, which is too), so the rule is
testable on a host with no LLM stack, no OMPL and no ROS.
"""
from __future__ import annotations

import os
from typing import Optional

from agents.trajectory.tools import planner_backend as _pb
from agents.trajectory.tools.planner_backend import DEFAULT_BACKEND, normalize_backend

#: The planning modes (the agent's strategy); ``agents.trajectory.agent`` reads
#: them from here. "classic": the end-of-August planner; "coordinated": per-arm
#: paths ordered by completion gates; "stepwise": composed parallel steps.
PLANNING_MODES: tuple[str, ...] = ("classic", "coordinated", "stepwise")
DEFAULT_PLANNING_MODE = "classic"

REQUEST, STACK, CODE = "request", "stack_default", "code_default"


class SelectionError(ValueError):
    """The request named a backend or mode that does not exist."""


def normalize_mode(value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    mode = str(value).strip().lower()
    if mode not in PLANNING_MODES:
        raise SelectionError(f"invalid planning mode {value!r} — valid: "
                             f"{', '.join(PLANNING_MODES)}")
    return mode


def _stack(name: str, normalise, env) -> Optional[str]:
    try:
        return normalise((env if env is not None else os.environ).get(name))
    except ValueError:
        # A bad stack default is the operator's startup setting, not this
        # request's fault; it is ignored and the code default applies.
        return None


def select(backend: Optional[str] = None, mode: Optional[str] = None,
           constraints: Optional[dict] = None, env=None,
           backend_options: Optional[dict] = None) -> dict:
    """``{"backend", "mode", "backend_from", "mode_from"[, "backend_options"]}``.

    Raises :class:`SelectionError` when the request (argument or constraints)
    names an unknown backend or mode, or names two different ones, or gives
    backend options the chosen backend refuses.
    """
    c = constraints or {}
    out: dict = {}
    for key, arg, cval, norm, env_name, default in (
        ("backend", backend, c.get("trajectory_backend"), normalize_backend,
         "TRAJECTORY_BACKEND", DEFAULT_BACKEND),
        ("mode", mode, c.get("planning_mode"), normalize_mode,
         "PLANNING_MODE", DEFAULT_PLANNING_MODE),
    ):
        try:
            a, b = norm(arg), norm(cval)
        except ValueError as exc:
            raise SelectionError(str(exc)) from None
        if a and b and a != b:
            raise SelectionError(f"the request names {key} {a!r} and its constraints "
                                 f"name {b!r} — give one")
        chosen, source = a or b, REQUEST
        if chosen is None:
            chosen, source = _stack(env_name, norm, env), STACK
        if chosen is None:
            chosen, source = default, CODE
        out[key], out[f"{key}_from"] = chosen, source
    from_c = c.get("backend_options")
    if backend_options and from_c and dict(backend_options) != dict(from_c):
        raise SelectionError("the request names backend_options and its constraints name "
                             "others — give one")
    try:
        opts = _pb.validate_backend_options(out["backend"], backend_options or from_c)
    except ValueError as exc:
        raise SelectionError(str(exc)) from None
    if opts:
        out["backend_options"] = opts
    return out


def apply(constraints: Optional[dict], selection: dict) -> dict:
    """``constraints`` with the selection written in, as ``plan()`` reads it."""
    c = dict(constraints or {})
    c["trajectory_backend"] = selection["backend"]
    c["planning_mode"] = selection["mode"]
    if selection.get("backend_options"):
        c["backend_options"] = dict(selection["backend_options"])
    else:
        c.pop("backend_options", None)
    return c


def valid() -> dict:
    # Read at call time: a plugin registered after import is selectable too.
    return {"backends": list(_pb.VALID_BACKENDS), "modes": list(PLANNING_MODES),
            "backend_details": _pb.available_backends()}
