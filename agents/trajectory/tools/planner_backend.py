"""
Deterministic trajectory-planner backend selection **and dispatch**.

The submitted task decides which motion planner runs, and that decision must be
honoured regardless of what the LLM does. This module owns both halves of that
guarantee:

  * ``plan()`` (agents/trajectory/agent.py) resolves the backend ONCE from the
    task constraints (then the ``TRAJECTORY_BACKEND`` sim-launch default) and
    binds it into a ``ContextVar`` before the ReAct agent runs.
  * ``dispatch_plan`` reads that binding via ``resolve_planning_backend`` instead
    of trusting an argument the LLM might supply, then calls the selected
    backend through the :data:`PLANNER_RESULT_CONTRACT` interface below.

This is the facade. Each backend is its own package under
``agents/trajectory/backends/`` (``ompl/``, ``ompl_compound/``, ``rrt/``) that
declares a ``BackendSpec`` and implements the contract in
``backends/contract.py``. A backend from another repository registers through
the ``umarell.trajectory_backends`` entry point or ``TRAJECTORY_BACKEND_PLUGINS``
(see ``backends/__init__.py``) — never by editing a backend's module, and never
by editing this one.

There is NO runtime fallback between planners. If a task selects a backend and
that backend cannot plan, the request fails — it is never silently rerouted.

Import-safe: standard library only (``os`` + ``contextvars``). Backend modules
are imported lazily inside ``dispatch_plan``, so this module stays importable
(and unit-testable) on a host with no ROS, no OMPL, and no Pinocchio.
"""
from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import Callable, Optional

from agents.trajectory import backends as _backends
from agents.trajectory.backends.contract import RESULT_KEYS, BackendSpec, load_ref

# The planners the interface may dispatch to: the built-ins below, plus any
# plugin found at import (entry points / TRAJECTORY_BACKEND_PLUGINS).
#   ompl — direct OMPL geometric planners + PathSimplifier over the Pinocchio
#          collision model (agents/trajectory/backends/ompl/). The
#          production backend: two-phase clearance/speed, orientation-constrained
#          Cartesian descent, inter-robot swept-volume coupling, TOPP-RA timing.
#   rrt  — hand-rolled single-tree Space-Time RRT (backends/rrt/, formerly
#          tools/rrt_planner.py). LEGACY and
#          NON-CONFORMING: it does not couple robots to each other, drops the
#          tool-orientation constraint, and does not honour the two-phase
#          speed/clearance split. Selectable for comparison only; see
#          NON_CONFORMING_BACKENDS.
#   ompl-compound — the step's transit for ALL moving arms planned as one path in
#          an OMPL CompoundStateSpace, arm-vs-arm checked in every state and timed
#          on one clock (backends/ompl_compound/, docs/adr/0001). Lift-off and
#          descent strokes per arm as in ``ompl``.
_SPECS: dict = _backends.discover()
VALID_BACKENDS: tuple[str, ...] = tuple(_SPECS)

# Backends that do NOT implement the full planning contract. Selecting one is
# allowed (it is sometimes useful to compare against), but ``dispatch_plan``
# stamps the result so downstream consumers — and the audit log — can tell that
# the safety properties the OMPL backend guarantees were not in force.
NON_CONFORMING_BACKENDS: frozenset[str] = frozenset(
    name for name, spec in _backends.BUILTIN.items() if not spec.conforming)

# Used when nothing (task, env) selects a backend.
DEFAULT_BACKEND = "ompl"

# ─── Backend result contract ──────────────────────────────────────────────────
# Every ``plan_trajectories_*`` entry point takes
#   (cell_id, motion_requests, *, approach_offset_m, velocity_scale, **kw)
# and returns a dict with at least these keys:
#   success          bool  — at least one trajectory was produced
#   trajectories     {robot_ns: [{"positions": [...], "time_s": float,
#                                 "phase": "transit"|"approach"}, ...]}
#   failures         [{"robot_ns", "error", "reason_code"?, ...}]
#   planning_time_s  float
#   backend          str   — the backend that produced this result
# The full contract (motion requests, chaining, home) is backends/contract.py.
PLANNER_RESULT_CONTRACT: tuple[str, ...] = RESULT_KEYS

# ``None`` means "no task bound a backend in this context" — ``resolve`` then
# consults the explicit argument, then the environment, then the default.
_PLANNING_BACKEND: ContextVar[Optional[str]] = ContextVar(
    "planning_backend", default=None
)


# Per-request backend options (``backend_options``), validated against the
# selected backend's ``BackendSpec.options`` and bound for one planning run the
# same way the backend itself is. ``dispatch_plan`` passes them to the backend.
_BACKEND_OPTIONS: ContextVar[dict] = ContextVar("backend_options", default={})

#: Option values travel over MCP/JSON; a backend gets plain scalars only.
OPTION_VALUE_TYPES = (str, int, float, bool)


def normalize_backend(value: Optional[str]) -> Optional[str]:
    """Lower-case and validate a backend name.

    Returns ``None`` for ``None``/empty (meaning "unspecified"). Raises
    ``ValueError`` for a non-empty value that is not a recognised backend, so a
    typo fails loudly rather than silently selecting the wrong planner.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid trajectory backend {value!r}. "
            f"Valid options: {', '.join(VALID_BACKENDS)}."
        )
    return normalized


def set_planning_backend(value: Optional[str]) -> Token:
    """Bind the backend for the current context (one planning run / task).

    Pass a concrete backend (``"ompl"`` / ``"rrt"``) to guarantee it, or
    ``None`` to leave the context unbound (``resolve`` will use env/default).
    Returns a token that must be passed to :func:`reset_planning_backend`.

    Raises ``ValueError`` if ``value`` is not a recognised backend.
    """
    return _PLANNING_BACKEND.set(normalize_backend(value))


def reset_planning_backend(token: Token) -> None:
    """Undo a previous :func:`set_planning_backend`, restoring the prior binding."""
    _PLANNING_BACKEND.reset(token)


def validate_backend_options(backend: str, options: Optional[dict]) -> dict:
    """``options`` checked for ``backend``: only the names its spec declares,
    scalar values, and its own ``check_options`` (e.g. "that policy exists").

    Empty/None is ``{}``. Raises ``ValueError`` naming what the backend accepts —
    an option the selected backend does not take is a request mistake, and
    ignoring it would plan with something the caller did not ask for.
    """
    if not options:
        return {}
    if not isinstance(options, dict):
        raise ValueError(f"backend_options must be an object, got {type(options).__name__}")
    spec = backend_spec(normalize_backend(backend) or resolve_planning_backend())
    unknown = sorted(set(options) - set(spec.options))
    if unknown:
        raise ValueError(f"backend {spec.name!r} takes no option(s) {unknown}; it accepts "
                         f"{list(spec.options) if spec.options else 'none'}")
    bad = sorted(k for k, v in options.items() if not isinstance(v, OPTION_VALUE_TYPES))
    if bad:
        raise ValueError(f"backend option(s) {bad} must be a string, number or boolean")
    out = dict(options)
    if spec.check_options:
        _backends.load_ref(spec.check_options)(dict(out))
    return out


def set_backend_options(options: Optional[dict]) -> Token:
    """Bind already-validated ``backend_options`` for the current planning run."""
    return _BACKEND_OPTIONS.set(dict(options or {}))


def reset_backend_options(token: Token) -> None:
    _BACKEND_OPTIONS.reset(token)


def current_backend_options() -> dict:
    return dict(_BACKEND_OPTIONS.get())


def current_planning_backend() -> Optional[str]:
    """Return the context-bound backend, or ``None`` if no task bound one."""
    return _PLANNING_BACKEND.get()


def resolve_planning_backend(explicit: Optional[str] = None) -> str:
    """Resolve the effective backend in priority order.

    1. Context binding set by ``plan()`` from the submitted task (authoritative).
    2. An explicit argument — for direct/diagnostic callers and tests.
    3. The ``TRAJECTORY_BACKEND`` environment variable (sim-launch default).
    4. :data:`DEFAULT_BACKEND` (``"ompl"``).

    Raises ``ValueError`` if any consulted source holds an invalid value.
    """
    bound = _PLANNING_BACKEND.get()
    if bound is not None:
        return bound

    explicit_norm = normalize_backend(explicit)
    if explicit_norm is not None:
        return explicit_norm

    env_norm = normalize_backend(os.environ.get("TRAJECTORY_BACKEND"))
    if env_norm is not None:
        return env_norm

    return DEFAULT_BACKEND


# ─── Registry ─────────────────────────────────────────────────────────────────

def backend_spec(name: str) -> BackendSpec:
    """The declaration of a registered backend (a plugin's is loaded here)."""
    source = _SPECS.get(name)
    if source is None:
        raise ValueError(f"no trajectory backend {name!r}")
    spec = _backends.resolve_spec(source)
    _SPECS[name] = spec          # resolve a plugin once
    return spec


def _entry_loader(name: str) -> Callable[[], Callable]:
    # Lazy: importing this module never pulls in OMPL/Pinocchio/ROS, and the
    # legacy rrt planner (backends/rrt, formerly tools/rrt_planner.py) is loaded
    # only when a task selects it.
    return lambda: load_ref(backend_spec(name).entry)


# name -> zero-arg loader returning the backend's entry point.
_BACKENDS: dict[str, Callable[[], Callable]] = {n: _entry_loader(n) for n in _SPECS}


def register_backend(spec: BackendSpec) -> None:
    """Add a backend at run time (a plugin's own setup code, or a test).

    A name that is already registered is refused: replacing a planner under a
    name a task already selects would change what that task runs.
    """
    global VALID_BACKENDS
    name = spec.name.strip().lower()
    if name in _SPECS:
        raise ValueError(f"trajectory backend {name!r} is already registered")
    _SPECS[name] = spec
    _BACKENDS[name] = _entry_loader(name)
    VALID_BACKENDS = tuple(_SPECS)


def available_backends() -> list[dict]:
    """What can be selected — for discovery endpoints. Never imports a planner
    except a plugin's declaration."""
    out = []
    for name in VALID_BACKENDS:
        try:
            spec = backend_spec(name)
            out.append({"name": name, "conforming": spec.conforming,
                        "home_planner": bool(spec.home), "description": spec.description,
                        "options": list(spec.options)})
        except Exception as exc:  # noqa: BLE001 — a broken plugin is listed, not hidden
            out.append({"name": name, "error": str(exc)})
    return out


def _is_conforming(name: str) -> bool:
    if name in NON_CONFORMING_BACKENDS:
        return False
    try:
        return backend_spec(name).conforming
    except Exception:  # noqa: BLE001
        return False


# ─── Return home ──────────────────────────────────────────────────────────────
# The coordination layer plans every program's return to HOME (the assembler's
# home step, the coordinated tree, the cell's home reset). A backend may provide
# its own home planner; one that does not gets DEFAULT_HOME_BACKEND's — which is
# what every backend had before backends were packages. The result goes through
# the same inter-arm gate either way.
DEFAULT_HOME_BACKEND = "ompl"


def _home_provider(attr: str) -> Callable:
    try:
        spec = backend_spec(resolve_planning_backend())
    except ValueError:
        spec = None
    ref = getattr(spec, attr, None) if spec else None
    return load_ref(ref or getattr(backend_spec(DEFAULT_HOME_BACKEND), attr))


def plan_home_trajectories(cell_id: str, starts: dict, **kwargs) -> dict:
    """Every arm's return to HOME as one coordinated set (see backends/ompl).

    A leg that fails the platform's own leg check (:func:`_leg_refusal`) is
    dropped, and the refusal is RECORDED (:func:`home_refusal`): that arm has no
    planned return, and the blind straight line that would stand in for it must
    not run either."""
    from agents.trajectory.tools import trajectory_cache as tc
    for ns, q in (starts or {}).items():
        tc.set_home_refusal(cell_id, ns, None)
    out = _home_provider("home")(cell_id, starts, **kwargs)
    if not isinstance(out, dict):
        return out
    for ns in [ns for ns, leg in out.items() if leg]:
        why = _leg_refusal(cell_id, ns, out[ns])
        if why:
            print(f"[home-plan] {ns}: planned return refused ({why['error']})", flush=True)
            tc.set_home_refusal(cell_id, ns, {**why, "reason_code": "home_return_refused",
                                              "check": why.get("reason_code")},
                                start=(starts or {}).get(ns))
            del out[ns]
    return out


def plan_home_trajectory(cell_id: str, robot_ns: str, start_q, home_q, **kwargs):
    """One arm's return to HOME against static obstacles. ``None`` (no planned
    return) when the planned leg fails the platform's leg check — recorded, as in
    :func:`plan_home_trajectories`."""
    from agents.trajectory.tools import trajectory_cache as tc
    tc.set_home_refusal(cell_id, robot_ns, None)
    leg = _home_provider("home_single")(cell_id, robot_ns, start_q, home_q, **kwargs)
    why = _leg_refusal(cell_id, robot_ns, leg) if leg else None
    if why:
        print(f"[home-plan] {robot_ns}: planned return refused ({why['error']})", flush=True)
        tc.set_home_refusal(cell_id, robot_ns, {**why, "reason_code": "home_return_refused",
                                                "check": why.get("reason_code")},
                            start=start_q)
        return None
    return leg


def _leg_refusal(cell_id: str, robot_ns: str, leg) -> Optional[dict]:
    """Why the platform refuses one arm's leg, whichever backend planned it, or
    None. A backend's legs are checked here, never trusted (docs/creating-a-backend.md):
    the joint box (umarell issue 13) and the arm against ITSELF along the executed
    path, densely (issue 14) — :func:`leg_verdict.refusal`, the one leg check every
    gate shares, so a leg judged here is not re-computed downstream.

    Static obstacles are not re-judged here: each planner checks them at the
    margins it planned with, and the gates add them (``static=True``).

    A check that cannot run (no geometry stack / no model) refuses nothing; one
    that ran and raised refuses (``leg_check_error`` /
    ``self_collision_check_error``) — see :mod:`leg_verdict`.
    """
    from agents.trajectory.tools.leg_verdict import refusal
    return refusal(cell_id, robot_ns, leg)


def _drop_legs(cell_id: str, result: dict, motion_requests: list, before: dict,
               bad: dict, partner_entry) -> None:
    """Drop the legs in ``bad`` (``{ns: failure entry}``) from ``result`` — and a
    compound step's partners with them (ADR-0001: each arm's plan assumed the
    others moved), each told by ``partner_entry(culprits)``. The backend's
    chaining for a dropped leg is taken back, so the next leg does not start where
    this one would have ended."""
    trajectories = result.get("trajectories") or {}
    doomed = set(bad)
    for ns in list(bad):
        leg = trajectories.get(ns)
        head = leg[0] if isinstance(leg, list) and leg else {}
        doomed |= set(head.get("sync_members") or []) if isinstance(head, dict) else set()
    op_of = {r.get("robot_ns"): r.get("operation_id") for r in motion_requests}
    from agents.trajectory.tools import trajectory_cache as tc
    failures = list(result.get("failures") or [])
    for ns in sorted(doomed & set(trajectories)):
        trajectories.pop(ns, None)
        if ns in before:
            tc.restore_chain_state(cell_id, ns, before[ns])
        entry = dict(bad[ns]) if ns in bad else partner_entry(sorted(bad))
        entry.setdefault("robot_ns", ns)
        if op_of.get(ns) is not None:
            entry["operation_id"] = op_of[ns]
        failures.append(entry)
    result["failures"] = failures
    result["success"] = bool(trajectories)


def _limit_refusals(cell_id: str, result: dict, motion_requests: list,
                    before: dict) -> None:
    """Drop every leg of ``result`` that leaves its arm's joint box — the contract
    says trajectories are "within the joint limits in the cell's robots.yaml", and
    the platform checks it for every backend (umarell issue 13). Judged by
    ``leg_verdict`` (cached, one policy): a check that broke refuses the leg."""
    trajectories = result.get("trajectories") or {}
    if not trajectories:
        return
    try:
        from agents.trajectory.tools.joint_limits import describe
        from agents.trajectory.tools.leg_verdict import limits
    except ImportError:          # no numpy: nothing planned here anyway
        return
    bad: dict = {}
    for ns, traj in trajectories.items():
        status, v = limits(cell_id, ns, traj)
        if status == "error":
            bad[ns] = {"reason_code": "leg_check_error", "check": "joint limits",
                       "error": f"{ns}'s leg could not be checked (joint limits): {v}"}
        elif status == "hit" and v is not None:
            bad[ns] = {"reason_code": "joint_limit_violation",
                       "joint_limit_violation": v, "error": describe(v, ns)}
    if bad:
        _drop_legs(cell_id, result, motion_requests, before, bad, lambda who: {
            "reason_code": "joint_limit_violation",
            "error": (f"the compound step is refused as a whole: {', '.join(who)} "
                      "leaves its joint limits")})


#: How far (rad, any joint) a leg's first sample may be from where its arm stands.
CONTRACT_START_TOL_RAD = 5e-3


def _contract_refusals(cell_id: str, result: dict, motion_requests: list,
                       before: dict) -> None:
    """Hold every backend to the result contract (docs/creating-a-backend.md),
    whatever it is: each leg a list of samples with finite ``positions`` of one
    length and finite, non-decreasing ``time_s`` from >= 0 — and STARTING WHERE
    THE ARM STANDS (the request's ``current_joints``, else where its last leg
    ended, else its HOME). A leg from anywhere else is a jump the executor would
    make unchecked. Refused legs are ``backend_contract_violation``."""
    import math
    trajectories = result.get("trajectories") or {}
    if not trajectories:
        return
    req_of = {r.get("robot_ns"): r for r in motion_requests if isinstance(r, dict)}
    bad: dict = {}

    def violation(ns, traj) -> "Optional[str]":
        if not isinstance(traj, list):
            return "the leg is not a list of samples"
        dof, last_t = None, -math.inf
        for k, pt in enumerate(traj):
            if not isinstance(pt, dict):
                return f"sample {k} is not a dict"
            q = pt.get("positions")
            try:
                q = [float(v) for v in q]
                t = float(pt.get("time_s"))
            except (TypeError, ValueError):
                return f"sample {k} has non-numeric positions or time_s"
            if not q or not all(math.isfinite(v) for v in q) or not math.isfinite(t):
                return f"sample {k} has empty or non-finite values"
            if dof is None:
                dof = len(q)
            elif len(q) != dof:
                return f"sample {k} has {len(q)} joints, the leg started with {dof}"
            if t < 0.0 or t < last_t - 1e-9:
                return f"time_s goes backwards (or below 0) at sample {k}"
            last_t = t
        start = (req_of.get(ns) or {}).get("current_joints") \
            or (before.get(ns) or {}).get("last_config")
        if not start:
            try:
                from agents.home_pose import program_start_pose
                start = program_start_pose(cell_id, ns)
            except Exception:  # noqa: BLE001 — unknown: not measured against a guess
                start = None
        if start:
            first = [float(v) for v in traj[0]["positions"]]
            n = min(len(first), len(start))
            err = max((abs(first[i] - float(start[i])) for i in range(n)), default=0.0)
            if err > CONTRACT_START_TOL_RAD:
                return (f"the leg starts {err:.4f} rad from where the arm stands "
                        f"(tolerance {CONTRACT_START_TOL_RAD} rad)")
        return None

    for ns, traj in trajectories.items():
        if isinstance(traj, list) and not traj:
            continue          # no leg (the assembler reports it missing), not a bad one
        why = violation(ns, traj)
        if why:
            bad[ns] = {"reason_code": "backend_contract_violation",
                       "error": f"backend {result.get('backend')!r}: {ns}: {why}"}
    if bad:
        _drop_legs(cell_id, result, motion_requests, before, bad, lambda who: {
            "reason_code": "step_refused", "caused_by": who,
            "error": (f"the compound step is refused as a whole: {', '.join(who)}'s "
                      "leg breaks the backend contract")})


def _replan_start(cell_id: str, request: dict) -> dict:
    """Pin the start of a RE-PLANNED operation (umarell issue 16).

    Every backend chains a leg from the arm's last planned goal. When the agent
    re-plans the operation it planned LAST, that goal is the end of this very
    operation, so the new leg would start on its own target, and in the tree the
    arm would jump from wherever it really was (HOME for a first operation) to the
    weld. In that one case the request is pinned (``current_joints``) to where the
    previous plan of this operation started — where the arm was before it. Pinned,
    not re-chained, so the lift-off is decided by where that start really is
    (``seam_at_start``). Any other request is returned unchanged.
    """
    ns, op = request.get("robot_ns"), request.get("operation_id")
    if request.get("current_joints") or not ns or not op:
        return request
    try:
        from agents.trajectory.tools.trajectory_cache import (
            get_cached_trajectory, get_last_config,
        )
        prev = get_cached_trajectory(cell_id, ns, op)
        cur = get_last_config(cell_id, ns)
        if not prev or not cur:
            return request
        q0 = [float(v) for v in prev[0].get("positions") or []]
        q1 = [float(v) for v in prev[-1].get("positions") or []]
        cur = [float(v) for v in cur]
        if not q0 or len(q0) != len(cur) or len(q1) != len(cur):
            return request
        # The recorded goal and the leg's last waypoint can differ by the final
        # IK / interpolation step (measured up to 0.017 rad), hence 0.02 rad.
        at_own_end = max(abs(a - b) for a, b in zip(q1, cur)) <= 0.02
        moved = max(abs(a - b) for a, b in zip(q0, q1)) > 1e-3
    except Exception:  # noqa: BLE001 — no cache: chain as before
        return request
    if not (at_own_end and moved):
        return request
    return {**request, "current_joints": q0}


def home_refusal(cell_id: str, robot_ns: str, start=None) -> Optional[dict]:
    """Why the last planned return of ``robot_ns`` (from ``start``) was refused —
    no on-axis lift-off out of its seam, or a leg that failed a check — so no leg
    was planned and the blind home that would replace it must not run. None when
    nothing was refused (trajectory_cache.home_refusal)."""
    from agents.trajectory.tools.trajectory_cache import home_refusal as _why
    return _why(cell_id, robot_ns, start)


def home_config(cell_id: str, robot_ns: str, fallback) -> list:
    """The cell's declared HOME for an arm (``agents.home_pose`` owns it)."""
    from agents.home_pose import home_pose
    return list(home_pose(cell_id, robot_ns, fallback))


def apply_tool_spin(request: dict) -> dict:
    """``tool_spin_deg`` → the request's ``orientation`` turned about the TOOL AXIS.

    The tool axis is the TCP frame's -x (``pinocchio._tool_axis``), so a spin
    is a rotation about the local x axis: ``q' = q * Rx(spin)``. It keeps the lean
    ``validate_approach`` gates (the axis) and changes only the roll about the
    electrode — the freedom a spot weld has, and what turns a wrist out of a
    self-collision or out of the other arm's way. Done here, before dispatch, so
    every backend gets the same pose and the agent never edits a quaternion.
    """
    spin = request.get("tool_spin_deg")
    ori = request.get("orientation")
    if not spin or not isinstance(ori, dict):
        return request
    import math
    h = math.radians(float(spin)) / 2.0
    c, s_ = math.cos(h), math.sin(h)
    w, x, y, z = (float(ori.get(k, 0.0)) for k in ("w", "x", "y", "z"))
    out = {k: v for k, v in request.items() if k != "tool_spin_deg"}
    out["orientation"] = {"w": round(w * c - x * s_, 9), "x": round(x * c + w * s_, 9),
                          "y": round(y * c + z * s_, 9), "z": round(z * c - y * s_, 9)}
    out["applied_tool_spin_deg"] = float(spin)
    return out


def _with_stroke(request: dict, default_offset: float) -> dict:
    """The request with the stroke the backend will plan it with: its own
    ``approach_offset_m``, else the call's default."""
    if request.get("approach_offset_m"):
        return request
    return {**request, "approach_offset_m": float(default_offset)}


def _auto_weld_orientation(cell_id: str, request: dict, default_offset: float = 0.1
                           ) -> "tuple[dict, Optional[dict], Optional[dict]]":
    """The request with the weld pose ``weld_orientation.select`` picks when the
    specified one has no collision-free solution (umarell issue 12), and what was
    chosen — or the request unchanged and None. Same pose for every backend, judged
    at the stroke the backend will use."""
    try:
        from agents.trajectory.tools import weld_orientation
        from agents.trajectory.tools.trajectory_cache import get_last_config
        ns = request.get("robot_ns")
        start = (request.get("current_joints") or get_last_config(cell_id, ns)
                 or home_config(cell_id, ns, None))
        found = weld_orientation.search(cell_id, _with_stroke(request, default_offset),
                                        start)
    except Exception:  # noqa: BLE001 — no geometry stack: plan the request as given
        return request, None, None
    chosen = found.get("chosen")
    report = {"status": found.get("status"), "candidates": found.get("candidates", 0)}
    if not chosen:
        return request, None, report
    return ({**request, "orientation": chosen["orientation"],
             "applied_tool_spin_deg": chosen["spin_deg"]}, chosen, report)


#: Strokes tried, as fractions of the requested one, when no weld pose is usable.
_SHORTER_STROKES = (2.0 / 3.0, 0.5, 0.3)
_MIN_STROKE_M = 0.03


def _record_weld_poses(cell_id: str, motion_requests: list, result: dict,
                       before: dict, default_offset: float) -> None:
    """Every weld this result planned is a seam a later leg may start on, whichever
    backend planned it (a plugin need not know about ``set_weld_pose``). When the
    backend did not record the weld itself in this call, it is recorded here from
    the request that was ACTUALLY planned (spin applied, pose selected) and with the
    orientation the leg really ENDS in — what the next lift-off must hold."""
    trajectories = result.get("trajectories") or {}
    try:
        from agents.trajectory.tools import trajectory_cache as tc
        from agents.trajectory.tools.weld_orientation import tcp_orientation
    except ImportError:
        return
    for r in motion_requests:
        ns, op = r.get("robot_ns"), r.get("operation_id")
        traj = trajectories.get(ns)
        if not (traj and op and r.get("approach") and r.get("constrain_orientation")
                and r.get("orientation")):
            continue
        prior = (before.get(ns) or {}).get("weld_pose")
        now = tc.chain_state(cell_id, ns, op).get("weld_pose")
        if now is not None and now != prior:
            continue                       # the backend recorded it in this call
        t = r.get("target") or {}
        ori = tcp_orientation(cell_id, ns, (traj[-1] or {}).get("positions") or [])
        tc.set_weld_pose(cell_id, ns, op, [t.get("x"), t.get("y"), t.get("z")],
                         r.get("approach_vector") or [0.0, 0.0, -1.0],
                         ori or r["orientation"],
                         stroke_m=tc.stroke_of(_with_stroke(r, default_offset)))


def _stroke_hints(cell_id: str, motion_requests: list, result: dict,
                  default_offset: float = 0.1) -> None:
    """For a weld refused because the arm folds into itself, say which SHORTER
    stroke would have a usable pose. The stroke length is the request's choice
    (``approach_offset_m``), so it is suggested, never changed here: a shorter
    entry keeps the wrist out of its own forearm at the top of the stroke."""
    codes = {"goal_self_collision", "approach_self_collision", "approach_unreachable"}
    by_op = {r.get("operation_id"): _with_stroke(r, default_offset) for r in motion_requests}
    for f in result.get("failures") or []:
        req = by_op.get(f.get("operation_id"))
        if f.get("reason_code") not in codes or not req or "suggestion" in f:
            continue
        try:
            from agents.trajectory.tools import weld_orientation
            from agents.trajectory.tools.trajectory_cache import get_last_config
            ns = req.get("robot_ns")
            start = (req.get("current_joints") or get_last_config(cell_id, ns)
                     or home_config(cell_id, ns, None))
            offset = float(req.get("approach_offset_m") or 0.1)
            # The search is free to turn the tool here even when the agent fixed a
            # spin: the hint says which stroke AND pose would work.
            free = {k: v for k, v in req.items()
                    if k not in ("tool_spin_deg", "applied_tool_spin_deg")}
            for frac in _SHORTER_STROKES:
                off = round(max(offset * frac, _MIN_STROKE_M), 3)
                if off >= offset:
                    continue
                chosen = weld_orientation.select(
                    cell_id, {**free, "approach_offset_m": off}, start)
                if chosen:
                    f["suggestion"] = (
                        f"a {off} m stroke (approach_offset_m={off}) has a usable pose: "
                        f"spin {chosen['spin_deg']} deg, tilt {chosen['tilt_deg']} deg, "
                        f"{chosen['self_clearance_m']} m clear of itself — re-plan with it")
                    f["suggested_approach_offset_m"] = off
                    break
        except Exception:  # noqa: BLE001 — a hint, never a failure of its own
            continue


def _self_collision_refusals(cell_id: str, motion_requests: list) -> list:
    """Failures for the requests whose robot must not be planned (see above)."""
    try:
        from agents.trajectory.tools import self_collision
    except ImportError:          # no numpy: nothing can plan here anyway
        return []
    out = []
    for ns in sorted({r.get("robot_ns") for r in motion_requests if r.get("robot_ns")}):
        why = self_collision.refusal(cell_id, ns)
        if why:
            out += [{"robot_ns": ns, "operation_id": r.get("operation_id"),
                     "reason_code": "self_collision_unavailable", "error": why}
                    for r in motion_requests if r.get("robot_ns") == ns]
    return out


# ─── Dispatch (the facade) ────────────────────────────────────────────────────


def dispatch_plan(
    cell_id: str,
    motion_requests: list[dict],
    approach_offset_m: float = 0.1,
    velocity_scale: Optional[float] = None,
    **backend_kwargs,
) -> dict:
    """Plan via the context-bound backend and return its contract-shaped result.

    This is the single dispatch point for motion planning. It resolves the
    backend deterministically (never from an LLM-supplied argument), calls it,
    and stamps ``backend`` / ``backend_used`` on the way out. No fallback: a
    backend that fails to plan reports that failure verbatim.

    ``approach_offset_m`` and ``velocity_scale`` are *defaults* — a motion
    request may override either per-operation, which is what lets the LLM choose
    a standoff and a speed per weld. ``velocity_scale=None`` leaves the transit
    speed to the backend, which reads the cell's ``planning.max_velocity_scaling``.
    """
    backend = resolve_planning_backend()
    # One request per arm per call: a second one for the same arm would silently
    # replace the first in the result and be cached under the wrong operation.
    duplicates: list = []
    seen: set = set()
    unique: list = []
    for r in motion_requests:
        ns = r.get("robot_ns")
        if ns in seen:
            duplicates.append({"robot_ns": ns, "operation_id": r.get("operation_id"),
                               "reason_code": "duplicate_robot_request",
                               "error": (f"more than one motion request for {ns} in one "
                                         "call — each arm moves once per call; pass a "
                                         "sequence as `steps`")})
            continue
        seen.add(ns)
        unique.append(r)
    # The operation as SPECIFIED, for the assembler's approach check.
    try:
        from agents.trajectory.tools import trajectory_cache as _tcs
        for r in unique:
            _tcs.set_op_spec(cell_id, r.get("robot_ns"), r.get("operation_id"), r)
    except Exception:  # noqa: BLE001
        pass
    motion_requests = [_replan_start(cell_id, apply_tool_spin(r)) for r in unique]
    # The leg store (tools/leg_reuse.py): the same step, on the same cell, from the
    # same arm poses, planned before — reused after the platform re-checks it.
    from agents.trajectory.tools import leg_reuse
    try:
        _ldr = _BACKENDS.get(backend)
        try:
            _entry = _ldr() if _ldr is not None else None
        except Exception:  # noqa: BLE001 — unavailable: reported below, as before
            _entry = None
        leg_key = None if _entry is None else leg_reuse.step_key(
            cell_id, backend, motion_requests, approach_offset_m, velocity_scale,
            current_backend_options(), backend_kwargs,
            code=leg_reuse.code_fingerprint(_entry))
        reused = leg_reuse.lookup(cell_id, leg_key)
        if reused is not None and not leg_reuse.verified(cell_id, reused):
            leg_reuse.forget(cell_id, leg_key)
            reused = None
    except Exception:  # noqa: BLE001 — no reuse, plan as usual
        leg_key, reused = None, None
    chosen_poses: dict = {}
    searches: dict = {}
    if reused is not None:
        # The requests as they were dispatched then (pose search included).
        motion_requests = reused["requests"]
    for i, r in enumerate(motion_requests if reused is None else []):
        motion_requests[i], chosen, report = _auto_weld_orientation(
            cell_id, r, approach_offset_m)
        if chosen:
            chosen_poses[str(r.get("operation_id") or r.get("robot_ns"))] = chosen
        if report:
            searches[r.get("operation_id")] = report
    # A robot whose self-collision check is REQUIRED but unavailable is not
    # planned, by any backend: its poses would ship unchecked against the arm
    # itself (tools/self_collision.py). The others in the step still are.
    refused = _self_collision_refusals(cell_id, motion_requests)
    if refused:
        reused = None
        motion_requests = [r for r in motion_requests
                           if r.get("robot_ns") not in {f["robot_ns"] for f in refused}]
        if not motion_requests:
            return {"success": False, "trajectories": {}, "failures": refused + duplicates,
                    "planning_time_s": 0.0, "backend": backend}
    loader = _BACKENDS.get(backend)
    if loader is None:  # unreachable via normalize_backend; defensive
        return {
            "success": False,
            "trajectories": {},
            "failures": [{"error": f"No implementation registered for backend {backend!r}"}],
            "planning_time_s": 0.0,
            "backend": backend,
        }

    try:
        entry = loader()
    except Exception as e:
        return {
            "success": False,
            "trajectories": {},
            "failures": [{"error": f"Backend {backend!r} unavailable: {e}"}],
            "planning_time_s": 0.0,
            "backend": backend,
        }

    try:
        from agents.trajectory.tools import trajectory_cache as _tc
        before = {r["robot_ns"]: _tc.chain_state(cell_id, r["robot_ns"], r.get("operation_id"))
                  for r in motion_requests if r.get("robot_ns")}
    except Exception:  # noqa: BLE001
        before = {}
    # The run's backend_options (validated when the request was accepted); a
    # keyword the platform passes itself wins over an option of the same name.
    options = current_backend_options()
    after: dict = {}
    if reused is not None:
        # Commit what the backend committed then, as if it had just planned it.
        from agents.trajectory.tools import trajectory_cache as _tcr
        for ns, st in (reused.get("after") or {}).items():
            _tcr.restore_chain_state(cell_id, ns, st)
        result = reused["result"]
        result["reused_step"] = {
            "planning_time_s_saved": reused.get("planning_time_s"),
            "note": ("this step was planned before on the same cell from the same "
                     "arm poses; it was re-checked, not re-solved")}
        result["planning_time_s"] = 0.0
    else:
        result = entry(
            cell_id,
            motion_requests,
            approach_offset_m=approach_offset_m,
            velocity_scale=velocity_scale,
            **{**options, **backend_kwargs},
        )
        try:
            from agents.trajectory.tools import trajectory_cache as _tca
            after = {r["robot_ns"]: _tca.chain_state(cell_id, r["robot_ns"],
                                                     r.get("operation_id"))
                     for r in motion_requests if r.get("robot_ns")}
        except Exception:  # noqa: BLE001
            after = {}
    if not isinstance(result, dict):
        return {
            "success": False,
            "trajectories": {},
            "failures": [{"error": f"Backend {backend!r} returned {type(result).__name__}, not a dict"}],
            "planning_time_s": 0.0,
            "backend": backend,
        }

    result.setdefault("backend", backend)
    result["backend_used"] = backend
    if options:
        result["backend_options"] = options
    _contract_refusals(cell_id, result, motion_requests, before)
    _limit_refusals(cell_id, result, motion_requests, before)
    if chosen_poses:
        # The pose that was planned, when it is not the one the request specified.
        result["weld_orientations"] = chosen_poses
    _record_weld_poses(cell_id, motion_requests, result, before, approach_offset_m)
    _stroke_hints(cell_id, motion_requests, result, approach_offset_m)
    if refused or duplicates:
        result["failures"] = list(result.get("failures") or []) + refused + duplicates
    # Every failure names its operation (the agent's recovery acts per operation),
    # whichever backend wrote it — and, for a weld, what the pose search did.
    op_of = {r.get("robot_ns"): r.get("operation_id") for r in motion_requests}
    for f in result.get("failures") or []:
        if isinstance(f, dict):
            if f.get("operation_id") is None and op_of.get(f.get("robot_ns")) is not None:
                f["operation_id"] = op_of[f["robot_ns"]]
            oid = f.get("operation_id")
            if oid is not None and oid in searches:
                f.setdefault("orientation_search", searches[oid])
            # What can be done about it, from the one table (reason_codes.py).
            if f.get("reason_code") and "recovery" not in f:
                from agents.trajectory.reason_codes import recovery
                rec = recovery(f["reason_code"])
                if rec:
                    f["recovery"] = rec
    if not _is_conforming(backend):
        # Make it impossible to mistake a legacy-backend plan for a fully-gated
        # one: the two-phase split and inter-robot coupling were NOT applied.
        result["non_conforming_backend"] = True
    if reused is None and after:
        try:
            leg_reuse.remember(cell_id, leg_key, motion_requests, result, after)
        except Exception:  # noqa: BLE001 — not stored: costs reuse, not safety
            pass
    return result
