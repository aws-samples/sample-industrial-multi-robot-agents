"""Trajectory backends — one package per motion planner.

    backends/
      contract.py        the interface every backend implements (BackendSpec)
      ompl/              production: OMPL per arm, swept-volume coupled
      ompl_compound/     all moving arms of a step in one compound state space
      rrt/               legacy Space-Time RRT (non-conforming, comparison only)
      roboballet/        BUNDLED: a trained RoboBallet GNN policy searches the
                         step's joint transit (agents/policy_trainer trains it)

Each package declares a ``SPEC`` (:class:`~.contract.BackendSpec`) in its
``__init__`` and keeps its implementation in ``planner.py``, imported only when
the backend is selected. Coordination, the LLM agent, the BehaviourTree and the
MCP interface stay in ``agents/trajectory`` and reach a backend only through
``agents/trajectory/tools/planner_backend.py``.

A backend from ANOTHER repository (e.g. a robot-ballet planner) is added without
editing this one:

* **Python entry point** in its own package metadata::

      [project.entry-points."umarell.trajectory_backends"]
      robot-ballet = "robot_ballet.umarell:SPEC"      # a BackendSpec

* **Environment** (quick wiring, e.g. a mounted checkout)::

      TRAJECTORY_BACKEND_PLUGINS="robot-ballet=robot_ballet.planner:plan"

  A backend given this way has no spec, so it is treated as non-conforming;
  point the variable at a ``BackendSpec`` instead to declare conformance.

A **bundled** backend lives in this repository but is registered like a
plugin rather than as a built-in: it is selectable by name, never the default,
and its absence of a prerequisite (``roboballet`` without a trained policy)
refuses its plans instead of affecting any other backend.

Either way the backend becomes selectable by name: ``TRAJECTORY_BACKEND=``,
``constraints.trajectory_backend``, or the ``backend`` of a planning request.
A plugin whose name collides with a built-in is refused, as is a malformed one.

Standard library only.
"""
from __future__ import annotations

import os
from typing import Callable, Union

from agents.trajectory.backends import ompl, ompl_compound, roboballet, rrt
from agents.trajectory.backends.contract import BackendSpec, load_ref

ENTRY_POINT_GROUP = "umarell.trajectory_backends"
PLUGIN_ENV = "TRAJECTORY_BACKEND_PLUGINS"

#: Built-in backends, in the order they are listed.
BUILTIN: dict[str, BackendSpec] = {
    s.name: s for s in (ompl.SPEC, rrt.SPEC, ompl_compound.SPEC)
}

#: In-repository backends registered like plugins: discovered, never default.
BUNDLED: dict[str, BackendSpec] = {s.name: s for s in (roboballet.SPEC,)}

#: A spec, or a zero-argument loader that returns one (plugins resolve lazily).
SpecSource = Union[BackendSpec, Callable[[], BackendSpec]]


def _spec_from_ref(name: str, ref: str) -> Callable[[], BackendSpec]:
    """Loader for ``name=module:attr``: a BackendSpec, or a bare plan function."""
    def load() -> BackendSpec:
        obj = load_ref(ref)
        if isinstance(obj, BackendSpec):
            return obj
        if callable(obj):
            return BackendSpec(name=name, entry=ref, conforming=False,
                               description=f"plugin {ref} (no spec: non-conforming)")
        raise TypeError(f"{ref} is neither a BackendSpec nor a function")
    return load


def plugins_from_env(value: str | None = None) -> dict[str, SpecSource]:
    """``"a=pkg.mod:fn,b=pkg2:SPEC"`` → ``{name: loader}``. Malformed entries raise."""
    raw = os.environ.get(PLUGIN_ENV, "") if value is None else value
    out: dict[str, SpecSource] = {}
    for item in (p.strip() for p in raw.split(",")):
        if not item:
            continue
        name, sep, ref = item.partition("=")
        name, ref = name.strip().lower(), ref.strip()
        if not sep or not name or ":" not in ref:
            raise ValueError(f"{PLUGIN_ENV}: {item!r} is not 'name=module:attr'")
        out[name] = _spec_from_ref(name, ref)
    return out


def plugins_from_entry_points() -> dict[str, SpecSource]:
    """Backends other installed packages declare under :data:`ENTRY_POINT_GROUP`."""
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — no metadata is no plugins
        return {}
    return {ep.name.strip().lower(): _spec_from_ref(ep.name.strip().lower(), ep.value)
            for ep in eps}


def discover() -> dict[str, SpecSource]:
    """Every selectable backend: built-ins, bundled, then entry points, then the env."""
    found: dict[str, SpecSource] = {**BUILTIN, **BUNDLED}
    for source in (plugins_from_entry_points(), plugins_from_env()):
        for name, spec in source.items():
            if name in BUILTIN or name in BUNDLED:
                raise ValueError(f"trajectory backend plugin {name!r} collides with "
                                 f"a {'built-in' if name in BUILTIN else 'bundled'} backend")
            found[name] = spec
    return found


def resolve_spec(source: SpecSource) -> BackendSpec:
    return source if isinstance(source, BackendSpec) else source()
