"""The trajectory-backend contract — what a motion planner must provide.

A backend turns one STEP of motion requests (one per arm moving in the step)
into timed joint trajectories. Everything around that — allocation, the LLM's
strategy, the inter-arm gate, the chain check, the BehaviourTree, execution — is
owned by the trajectory layer and is the same for every backend. So a new
planner (e.g. a "robot ballet" planner living in another repository) implements
ONE function and declares itself with a :class:`BackendSpec`.

Planning entry point (``BackendSpec.entry``)::

    def plan(cell_id: str, motion_requests: list[dict], *,
             approach_offset_m: float = 0.1,
             velocity_scale: float | None = None,
             **kwargs) -> dict

``motion_requests`` — one dict per arm, all moving in the same step::

    {"robot_ns": "robot1", "operation_id": "W05", "operation_type": "weld",
     "target": {"x", "y", "z"},           # TCP goal, world frame (m)
     "approach_vector": [ax, ay, az],     # tool axis into the target
     "orientation": {"x", "y", "z", "w"}, "constrain_orientation": bool,
     "approach": bool,                    # final segment is the working stroke
     "approach_offset_m": float, "retreat_offset_m": float,
     "approach_velocity_scale": float,
     "current_joints": [..]}              # optional explicit start

Each arm starts where its previous leg ended unless ``current_joints`` is given:
read it with ``trajectory_cache.get_last_config(cell_id, robot_ns)`` (HOME when
``None``) and record the end with ``set_last_config`` — that is what keeps a
program continuous across calls.

Result — a dict with at least :data:`RESULT_KEYS`::

    {"success": bool,                                  # >= 1 trajectory produced
     "trajectories": {robot_ns: [{"positions": [q..], "time_s": t,
                                  "phase": "transit" | "approach"}, ...]},
     "failures": [{"robot_ns", "operation_id", "error", "reason_code"?}],
     "planning_time_s": float,
     "backend": str}

Trajectories must be time-monotonic, start at the arm's start configuration and
respect the cell's joint limits. They are checked after the backend returns —
inter-arm clearance by the step gate and the assembler, continuity by the chain
check — so a backend that skips a property is caught, never trusted.

Optional return-home entry points (``BackendSpec.home`` / ``home_single``),
same signatures as ``backends/ompl/planner.py``'s ``plan_home_trajectories`` /
``plan_home_trajectory``. A backend without them gets the default home planner
(see ``planner_backend``).

``options`` names the per-request options a backend takes. A planning request
carries them as ``backend_options`` (MCP ``plan_trajectories`` /
``replan_operation``, A2A, ``make plan BACKEND_OPTIONS=``), the facade validates
them against this list (and ``check_options``) before the run starts and hands
them to ``entry`` as keyword arguments — the same interface for every backend,
e.g. ``roboballet``'s ``{"policy": "<name>"}``.

``joint_step=True`` declares that the arms of a step are planned together (and
refused together), so ``replan_operation`` re-plans a PARALLEL step's legs as
one unit rather than one leg at a time.

``conforming=False`` declares that the backend does not honour the full planning
contract (inter-arm coupling, the orientation-constrained stroke, the two-phase
speed split); its results are stamped ``non_conforming_backend: true``. A backend
registered without a spec — the ``TRAJECTORY_BACKEND_PLUGINS`` shorthand — is
treated as non-conforming until it says otherwise.

Standard library only: nothing here imports a planner.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Optional

#: Keys every planning result carries (the facade's PLANNER_RESULT_CONTRACT).
RESULT_KEYS: tuple[str, ...] = (
    "success", "trajectories", "failures", "planning_time_s", "backend",
)


@dataclass(frozen=True)
class BackendSpec:
    """A backend's declaration. References are ``"package.module:function"``
    strings, imported only when the backend is actually used."""

    name: str
    entry: str
    home: Optional[str] = None          # plan_home_trajectories-compatible
    home_single: Optional[str] = None   # plan_home_trajectory-compatible
    conforming: bool = True
    description: str = ""
    #: The backend plans a PARALLEL step's arms as ONE motion, so an edit
    #: re-plans the whole step together (``tools/plan_edit.planning_units``).
    joint_step: bool = False
    #: Per-request options this backend accepts (``backend_options`` of a
    #: planning request), passed to ``entry`` as keyword arguments. Anything
    #: else a request names is refused — never silently ignored.
    options: tuple = ()
    #: Optional ``"module:function"`` checking a request's options before the
    #: run starts: ``check(options: dict) -> None``, raising ``ValueError``.
    check_options: Optional[str] = None


def load_ref(ref: str) -> Callable:
    """Import ``"package.module:attr"`` and return the attribute."""
    module, _, attr = ref.partition(":")
    if not module or not attr:
        raise ValueError(f"backend reference {ref!r} is not 'module:function'")
    return getattr(importlib.import_module(module), attr)


def missing_result_keys(result) -> list[str]:
    """The contract keys a result lacks (all of them if it is not a dict)."""
    if not isinstance(result, dict):
        return list(RESULT_KEYS)
    return [k for k in RESULT_KEYS if k not in result]
