"""
OMPL trajectory-planning backend.

This is the ``ompl`` implementation behind the trajectory planner dispatcher
(``rrt_planner.run_trajectory_planner``). It is selected deterministically by the
submitted task via ``planner_backend`` — never by the LLM — and there is NO
fallback: if OMPL is unavailable or a plan fails, the affected request is
reported as a failure rather than being rerouted to the hand-rolled RRT planner.

Why this backend exists
-----------------------
The default ``rrt`` backend hand-rolls a single-tree RRT (``_rrt_single``) with a
linear-scan nearest-neighbour and a custom shortcut (``_shortcut_path``). This
backend swaps that hand-written machinery for OMPL's native, mature
implementations while keeping everything else identical:

  * **Global search:** OMPL ``RRTConnect`` (bidirectional, far better at the
    narrow-passage problem of threading the torch past the fixture) or any other
    OMPL geometric planner (``BITstar``, ``RRTstar``, ...). Replaces the custom
    RRT loop, sampling, goal-biasing and nearest-neighbour structures.
  * **Path cleanup:** OMPL ``PathSimplifier`` (collision-aware shortcutting +
    B-spline smoothing). Replaces the custom ``_shortcut_path``.

What is deliberately REUSED (not reimplemented):
  * **IK:** Pinocchio damped least-squares (``pinocchio.run_ik``).
  * **Collision:** the shared sphere-model predicate
    (``robot_kinematics._config_is_free``) is wrapped as OMPL's
    ``StateValidityChecker`` — so OMPL accepts/rejects exactly the same
    configurations the runtime collision monitor does. That module, not this
    one and not the legacy RRT backend, owns the collision model.
  * **Timing:** the shared time-parameterisation
    (``time_parameterization.parameterize`` — TOPP-RA, cubic-Hermite fallback)
    — OMPL geometric planners produce a *path*, not a time-stamped trajectory,
    so timing stays common to both backends.

Design notes
------------
* **Pure backend.** No LangChain ``@tool`` and no OMPL import at module load, so
  it imports cleanly on hosts without OMPL installed (dev boxes, unit tests).
  Call :func:`ompl_available` to check whether a live plan can be attempted; all
  OMPL symbols are imported lazily inside :func:`_import_ompl`, mirroring how the
  ROS-dependent code guards its import.
* **Output contract.** :func:`plan_trajectories_ompl` returns exactly the shape
  every backend returns::

      {"success": bool,
       "trajectories": {robot_ns: [{"positions": [j1..j6], "time_s": float}, ...]},
       "planning_time_s": float,
       "failures": [{"robot_ns": str, "error": str}, ...],
       "backend": "ompl"}

  so the behaviour-tree builder and ``validate_trajectory_pair`` are unchanged.

Tunables (environment variables)
---------------------------------
  OMPL_PLANNER              planner name (default ``RRTConnect``; also RRT,
                            RRTstar, BITstar, PRM, PRMstar, InformedRRTstar,
                            LBTRRT).
  OMPL_TIME_BUDGET_S        per-segment solve time budget (default ``5.0``).
  OMPL_RANGE                planner step/range in radians (default ``0`` →
                            let the planner pick its own).
  OMPL_VALIDITY_RESOLUTION  motion-check resolution as a fraction of the state
                            space extent (default ``0.005``; see
                            :func:`_search_resolution`).
"""
from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from agents.trajectory.tools import plan_cancellation


# ─── Lazy OMPL import ─────────────────────────────────────────────────────────
# _OMPL is: None (not yet attempted), False (attempted and unavailable), or a
# SimpleNamespace of the imported sub-modules (available).
_OMPL: Optional[object] = None
_OMPL_IMPORT_ERROR: Optional[str] = None


def _import_ompl():
    """Import OMPL sub-modules lazily; cache the result.

    Returns a ``SimpleNamespace(ob=..., og=...)`` when available, or ``None``
    when OMPL is not installed (message stored in ``_OMPL_IMPORT_ERROR``).
    """
    global _OMPL, _OMPL_IMPORT_ERROR
    if _OMPL is not None:
        return _OMPL or None  # False -> None
    try:
        from ompl import base as ob
        from ompl import geometric as og

        _OMPL = SimpleNamespace(ob=ob, og=og)
    except Exception as exc:  # ImportError on host, other errors in broken envs
        _OMPL_IMPORT_ERROR = repr(exc)
        _OMPL = False
        return None
    return _OMPL


def ompl_available() -> bool:
    """True if the ``ompl`` Python bindings can be imported in this process."""
    return _import_ompl() is not None


def ompl_import_error() -> Optional[str]:
    """The captured import error string, or None if OMPL imported fine."""
    _import_ompl()
    return _OMPL_IMPORT_ERROR


# ─── OMPL helpers ─────────────────────────────────────────────────────────────

def _make_planner(og, si, planner_name: str):
    """Instantiate an OMPL geometric planner by name (default RRTConnect).

    Unknown names fall back to RRTConnect. Names absent from the local OMPL
    build (some builds omit BITstar etc.) also fall back rather than raising.
    """
    aliases = {
        "rrtconnect": "RRTConnect",
        "rrt": "RRT",
        "rrtstar": "RRTstar",
        "bitstar": "BITstar",
        "abitstar": "ABITstar",
        "prm": "PRM",
        "prmstar": "PRMstar",
        "informedrrtstar": "InformedRRTstar",
        "lbtrrt": "LBTRRT",
        "est": "EST",
        "kpiece": "KPIECE1",
    }
    key = (planner_name or "RRTConnect").strip().lower()
    cls_name = aliases.get(key, "RRTConnect")
    cls = getattr(og, cls_name, None) or getattr(og, "RRTConnect")
    return cls(si)


def _make_validity_checker(ob, si, is_free_fn, nq: int):
    """Wrap the shared Pinocchio collision predicate as an OMPL checker.

    OMPL calls ``isValid`` for every sampled state and every interpolated motion
    segment. Each call reads the joint vector out of the OMPL state and defers to
    ``is_free_fn`` (``robot_kinematics._config_is_free``), so validity is identical
    to the runtime monitor's. This Python callback is the dominant per-check cost — the
    expected trade-off for reusing the exact sphere-model collision semantics.
    """

    class _PinocchioValidityChecker(ob.StateValidityChecker):
        def __init__(self, space_information):
            super().__init__(space_information)

        def isValid(self, state) -> bool:  # noqa: N802 (OMPL API name)
            q = np.empty(nq)
            for i in range(nq):
                q[i] = state[i]
            return bool(is_free_fn(q))

    return _PinocchioValidityChecker(si)


# Backend-neutral path metrics, shared with trajectory_quality (one owner).
from agents.trajectory.tools.path_metrics import (  # noqa: E402
    _path_directness, _path_excess_travel, _path_joint_length,
    _trajectory_excess_travel,
)




# ─── parallel candidate solves ────────────────────────────────────────────────
# The candidates are INDEPENDENT solves of the same problem, so they can run at
# once. Threads cannot: the validity checker is Python (Pinocchio collision
# queries), so the GIL serialises them. Forked processes can — each inherits the
# already-built problem (space, checker, model) and only ships its waypoints back.
# Measured cost per solve on welding_cell_2_small is 6-10 s and up to minutes, and a
# segment asks for up to OMPL_PATH_CANDIDATES (4) of them, so this is where most
# of a planning run's wall time goes. Opt-in: OMPL_PARALLEL_CANDIDATES=1 (off until benchmarked).

def _parallel_enabled() -> bool:
    if os.environ.get("OMPL_PARALLEL_CANDIDATES", "0").strip() in ("0", "false", "no"):
        return False
    if os.environ.get("_OMPL_CANDIDATE_CHILD"):
        return False                      # never fork from inside a candidate
    try:
        import multiprocessing as mp
        return "fork" in mp.get_all_start_methods()
    except Exception:  # noqa: BLE001
        return False


def _advance_rng(k: int) -> None:
    """Give this process its own random stream. Every forked child starts from
    the SAME RNG state, and identical streams mean identical paths — four equal
    candidates. OMPL seeds each new RNG from a shared seed generator, so drawing
    ``k`` throwaway RNGs moves this child to a different point of it."""
    try:
        from ompl import util as _ou
        for _ in range(int(k)):
            _ou.RNG()
    except Exception:  # noqa: BLE001
        pass


def _candidate_child(solve, conn, index: int) -> None:
    os.environ["_OMPL_CANDIDATE_CHILD"] = "1"
    try:
        _advance_rng(17 * (index + 1))
        waypoints, fail = solve()
        payload = ([[float(v) for v in q] for q in waypoints] if waypoints else None, fail)
    except BaseException as exc:  # noqa: BLE001
        payload = (None, {"reason_code": "planner_error",
                          "reason": f"candidate solve raised: {exc!r}"[:300]})
    try:
        conn.send(payload)
    finally:
        conn.close()
        os._exit(0)


def _parallel_solves(solve, n: int):
    """``n`` independent solves run at once, as ``[(waypoints, failure), ...]``,
    or ``None`` to fall back to solving in sequence (disabled, no fork, or any
    failure to set the batch up)."""
    if n < 2 or not _parallel_enabled():
        return None
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        jobs = []
        for i in range(n):
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_candidate_child, args=(solve, send, i), daemon=True)
            proc.start()
            send.close()
            jobs.append((proc, recv))
    except Exception:  # noqa: BLE001
        return None
    out = []
    for proc, recv in jobs:
        try:
            wps, fail = recv.recv()
            out.append(([np.asarray(q, float) for q in wps] if wps else None, fail))
        except (EOFError, OSError):
            out.append((None, {"reason_code": "planner_error",
                               "reason": "a parallel candidate solve exited without a result"}))
        finally:
            recv.close()
            proc.join(timeout=5)
    return out


def _pick_path_candidate(solve, attempts: int, directness_target: float,
                         score=None, parallel: bool = False):
    """The most direct of up to ``attempts`` independent solves.

    THE owner of "which path ships". A sampling planner returns whichever
    homotopy class its trees happened to meet in — over the top of the cell or
    around the side — and shortcutting cannot move a path between classes
    (:func:`_simplify_path`). Escaping a bad class takes another solve, so
    choosing among solves is the only place path height can be owned.

    Measured on ``welding_cell_2_small``, ten solves of ONE weld transit with byte-for-
    byte identical inputs produced tool rises from 0.027 m to 2.767 m (peak tool
    height 1.618 m to 4.358 m). Which one an operator watched was luck. That is
    what this removes: not the variance itself, which belongs to the planner, but
    the chance of shipping the worst draw.

    Stops early on the first candidate at or under ``directness_target`` rather
    than always spending ``attempts`` solves, because each solve costs ~6-10 s on
    this cell. Calibration for the 1.6 directness default, from those same ten
    candidates (lower bound 0.971 rad): ratios 1.05/1.55/1.59 rose
    0.027/0.182/0.453 m, while every candidate above 1.6 rose 0.6-2.8 m.

    ``score`` is the ranking function — LOWER IS BETTER, and ``directness_target``
    is the cutoff in whatever unit it returns. It defaults to
    :func:`_path_directness` (ratio, cutoff 1.6); production passes
    :func:`_path_tool_rise` (metres, cutoff from :func:`_path_rise_policy`). This
    function does not care which: it owns "which candidate ships", not what makes
    a candidate good.

    Returns ``(waypoints, None)`` or ``(None, failure)``, preserving the caller's
    contract exactly. A solve that fails does not sink the batch — a later one may
    succeed — and if every attempt fails the LAST failure is reported, so the
    actionable reason (``goal_in_collision`` and friends) still reaches the agent.
    Availability never regresses: with ``attempts=1`` this is the old behaviour,
    and a ``score`` that raises costs quality, never the path (see below).
    """
    score = _path_directness if score is None else score
    best = None
    best_ratio = None
    failure = None
    # ``parallel`` is opted into by the real OMPL call site only: a solve must be
    # a pure function of the problem for forked copies of it to be candidates.
    batch = (_parallel_solves(solve, int(attempts))
             if parallel and int(attempts) > 1 else None)
    results = iter(batch) if batch is not None else None
    for _ in range(max(1, int(attempts))):
        waypoints, fail = next(results) if results is not None else solve()
        if not waypoints:
            failure = fail or failure
            continue
        try:
            ratio = float(score(waypoints))
        except Exception:
            # Scoring is a quality mechanism. If FK or the metric blows up we
            # ship a valid path unranked rather than refuse a motion that plans.
            return waypoints, None
        if best_ratio is None or ratio < best_ratio:
            best, best_ratio = waypoints, ratio
        if best_ratio <= float(directness_target) and results is None:
            break
    if best is None:
        return None, failure
    return best, None


def _path_candidate_policy() -> "tuple[int, float]":
    """``(attempts, directness_target)`` for segment path selection.

    Env-tunable so a cell with a tighter time budget can spend fewer solves
    (``OMPL_PATH_CANDIDATES=1`` restores the pre-selection behaviour exactly)
    without a code change. A target below 1.0 is unreachable by construction and
    would burn every attempt for nothing, so it is floored at 1.0.
    """
    try:
        attempts = int(os.environ.get("OMPL_PATH_CANDIDATES", "4") or 4)
    except ValueError:
        attempts = 4
    try:
        target = float(os.environ.get("OMPL_PATH_DIRECTNESS", "1.6") or 1.6)
    except ValueError:
        target = 1.6
    return max(1, attempts), max(1.0, target)


from agents.trajectory.tools.path_metrics import _path_rise_policy  # noqa: E402,F401
from agents.trajectory.tools.leg_builder import (  # noqa: E402,F401 — the shared leg pieces
    path_tool_rise as _path_tool_rise,
    emitted_rise_limit as _emitted_rise_limit,
    trajectory_tool_rise as _trajectory_tool_rise,
    cartesian_descent as _cartesian_descent,
    descent_failure as _descent_failure,
    cartesian_retreat as _cartesian_retreat,
    accel_limits as _accel_limits,
    home_config as _home_config,
    trajectory_end as _trajectory_end,
)

# (moved to the backend-neutral path_metrics: the weld pose selection reads it too)


def _staging_extra_travel_limit() -> float:
    """Extra travel a STAGED route may add over the route it replaces, in radians.

    Staging trades a flatter path for a longer one, and the trade is only worth
    making while the extra travel stays within what a legitimate staged return
    costs. 1.20 rad is that cost, measured: it is the largest excess travel among
    the home legs on ``welding_cell_2_small`` that no operator objected to (robot2's
    return in ``bt_1787955158``). The staged leg that WAS reported spent 10.87.

    Floored at 0.0 — a negative allowance would forbid staging outright, including
    the case it exists for. ``OMPL_STAGING_EXTRA_TRAVEL_RAD`` tunes it for a cell
    whose fixtures make every direct return arc.
    """
    try:
        extra = float(
            os.environ.get("OMPL_STAGING_EXTRA_TRAVEL_RAD", "1.2") or 1.2)
    except ValueError:
        extra = 1.2
    return max(0.0, extra)






def _path_score_policy(model, data, base, yaw):
    """``(score_fn, target)`` for candidate selection — measured rise if we can.

    Prefers :func:`_path_tool_rise`, which measures the excursion an operator
    actually sees, and falls back to the :func:`_path_directness` proxy if FK is
    not usable for this model (no Pinocchio, no recognisable tool frame). The
    fallback is chosen HERE, once per segment, off a single probe — never
    per-candidate, because a score and its cutoff must be in the same unit for the
    stop rule to mean anything.
    """
    zero = np.zeros(model.nq)
    try:
        _path_tool_rise([zero, zero], model, data, base, yaw)
    except Exception:
        return _path_directness, _path_candidate_policy()[1]

    def score(waypoints):
        return _path_tool_rise(waypoints, model, data, base, yaw)

    return score, _path_rise_policy()


def _simplify_path(og, si, path) -> None:
    """Shortcut a solved path with OMPL's native ``PathSimplifier``.

    Replaces the hand-rolled ``_shortcut_path`` (collision-aware shortcutting +
    B-spline smoothing). Best-effort and silent: a raw valid path is still a valid
    path, and losing the cosmetics of a motion is never a reason to fail a plan.
    Validity is not this function's business — it is enforced by the validity
    checker bound into ``si``, which every state the shortcutter emits must pass.

    WHAT THIS CANNOT DO, measured, so nobody pays to learn it twice: it cannot
    take the height out of a tall transit. A solved RRTConnect path arrives with
    THREE states — start, the point where the two trees met, and goal — so the
    only shortcut available is start->goal, and next to a workpiece that shortcut
    is blocked. The random meeting point therefore survives verbatim, and if it
    sits high above the cell so does the motion the operator watches.

    Interpolating first, to give the shortcutter intermediate states to carve
    against, does NOT fix that: on ``welding_cell_2_small`` it took step 1 to 0.000 m
    of rise but left the W14 transit at 1.534 m (against 1.448 m without) and the
    robot2 home return at 1.582 m (against 0.715 m), while costing ~+50% segment
    planning time. ``simplifyMax`` collapses the interpolated path straight back
    to 3 states.

    The reason is that the arc is a HOMOTOPY choice, not a smoothing deficit —
    over the top versus around the side — and local shortcutting cannot leave the
    class the solve landed in. Escaping it takes another solve, which is why the
    candidate selection in :func:`_plan_ompl_segment` owns path height and this
    function does not.
    """
    try:
        og.PathSimplifier(si).simplifyMax(path)
    except Exception:
        # Simplification is best-effort — a raw valid path is still usable.
        pass


def _plan_ompl_segment(
    ompl,
    model,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    jl: np.ndarray,
    ju: np.ndarray,
    obstacle_spheres: list,
    base: np.ndarray,
    yaw: float,
    static_obstacles: list,
    planner_name: str,
    time_budget_s: float,
    step_size: Optional[float],
    validity_resolution: float,
    tool_cfg: "Optional[dict]" = None,
    sphere_radius: float = 0.12,
    sphere_radii=None,
    extra_margin: float = 0.0,
    limit_box=None,
    try_direct: bool = True,
) -> "tuple[Optional[list[np.ndarray]], Optional[dict]]":
    """Plan one collision-free joint-space segment with OMPL.

    ``limit_box`` is the arm's REAL joint box (``joint_limits.position_limits``,
    unbounded sides infinite): an endpoint outside it is refused; one merely
    outside the finite search bounds ``jl``/``ju`` is clipped, as before.
    ``try_direct=False`` skips the straight-line shortcut — for a re-solve after
    the straight line's emitted samples were refused, which it would return again.

    Returns ``(waypoints, None)`` on success (waypoints include the start and
    goal states), or ``(None, failure)`` on failure, where ``failure`` is a dict
    ``{"reason_code", "reason", "collision"?}``:

      * ``goal_in_collision`` / ``start_in_collision`` — the endpoint pose is
        itself invalid; ``collision`` names the object involved and the world
        collision point so the LLM can pick a recovery strategy (stage above and
        descend, add an intermediate waypoint, retract, or reorient).
      * ``no_path_found`` — endpoints are valid but no path within the budget.

    This mirrors the intent of ``rrt_planner._rrt_single`` while giving the agent
    a far clearer signal than a bare timeout.
    """
    from agents.trajectory.tools.robot_kinematics import (
        _config_is_free,
        _config_collision_info,
        _collision_failure,
    )

    ob, og = ompl.ob, ompl.og
    nq = model.nq

    # One reusable collision-data buffer for this segment's validity checks.
    col_data = model.createData()

    def is_free(q: np.ndarray) -> bool:
        return _config_is_free(
            model, col_data, q, base, yaw, obstacle_spheres, static_obstacles,
            sphere_radius=sphere_radius, tool_cfg=tool_cfg,
            sphere_radii=sphere_radii, extra_margin=extra_margin,
        )

    # State space = joint space with the robot's joint limits as bounds.
    space = ob.RealVectorStateSpace(nq)
    bounds = ob.RealVectorBounds(nq)
    for i in range(nq):
        bounds.setLow(i, float(jl[i]))
        bounds.setHigh(i, float(ju[i]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    # The checker is bound into `si` (C++ side) but we also hold this local
    # reference for the lifetime of the solve so it cannot be GC'd mid-plan.
    checker = _make_validity_checker(ob, si, is_free, nq)
    si.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(float(validity_resolution))
    si.setup()

    # An endpoint outside the joint box is refused, not clipped: a clipped start
    # is a path that does not begin where the arm is (umarell issue 13).
    from agents.trajectory.tools.joint_limits import describe, trajectory_limit_violation
    lo_box, hi_box = limit_box if limit_box is not None else (jl, ju)
    for which, q in (("start", start_q), ("goal", goal_q)):
        v = trajectory_limit_violation([np.asarray(q, dtype=float)[:nq]], lo_box, hi_box)
        if v is not None:
            return None, {"reason_code": f"{which}_outside_joint_limits",
                          "reason": f"the {which} configuration is outside the joint "
                                    f"limits: {describe(v)}",
                          "joint_limit_violation": v}
    start_c = np.clip(np.asarray(start_q, dtype=float)[:nq], jl[:nq], ju[:nq])
    goal_c = np.clip(np.asarray(goal_q, dtype=float)[:nq], jl[:nq], ju[:nq])

    # Collision-aware pre-checks so the failure is actionable for the LLM. A
    # goal/start that is itself in collision cannot be reached by ANY path — the
    # planner would just burn the whole budget — so fail fast with the object
    # involved + world collision point (shared with the RRT backend).
    ginfo = _config_collision_info(
        model, col_data, goal_c, base, yaw, obstacle_spheres, static_obstacles,
        sphere_radius=sphere_radius, sphere_radii=sphere_radii,
        extra_margin=extra_margin,
    )
    if ginfo is not None:
        return None, _collision_failure("goal", ginfo)
    sinfo = _config_collision_info(
        model, col_data, start_c, base, yaw, obstacle_spheres, static_obstacles,
        sphere_radius=sphere_radius, sphere_radii=sphere_radii,
        extra_margin=extra_margin,
    )
    if sinfo is not None:
        return None, _collision_failure("start", sinfo)

    # OMPL 2.x (nanobind) has no ScopedState/State constructor — allocate raw
    # states from the SpaceInformation. Their lifetime is managed by the binding
    # (Python GC), so we must NOT call si.freeState() on them: setStartAndGoalStates
    # copies them internally, and an explicit free here double-frees on GC.
    start = si.allocState()
    goal = si.allocState()
    for i in range(nq):
        start[i] = float(start_c[i])
        goal[i] = float(goal_c[i])

    # Nothing beats going straight there, so ask before searching. `checkMotion`
    # uses the validity checker and resolution already bound into `si`, so this is
    # the planner's own standard of "clear", not a second opinion — and it costs
    # one motion check instead of a multi-second solve. Guarded because the
    # nanobind bindings' surface varies by build; a missing checkMotion just means
    # we search as before.
    try:
        if try_direct and si.checkMotion(start, goal):
            return [np.asarray(start_c, float), np.asarray(goal_c, float)], None
    except Exception:
        pass

    def _solve():
        """One independent solve. Fresh pdef and planner every time.

        Reusing either would hand back the tree the last attempt already built,
        so every candidate would be the same path and selecting among them would
        be theatre.
        """
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)

        planner = _make_planner(og, si, planner_name)
        if step_size and hasattr(planner, "setRange"):
            planner.setRange(float(step_size))
        planner.setProblemDefinition(pdef)
        planner.setup()

        planner.solve(float(time_budget_s))

        # NOTE: in the nanobind bindings PlannerStatus is truthy even for an
        # APPROXIMATE solution, so we must NOT gate on bool(status). Gate on
        # hasExactSolution() so we only accept a fully-connected start→goal path.
        if not pdef.hasExactSolution():
            return None, {
                "reason_code": "no_path_found",
                "reason": (
                    f"no collision-free path found within {float(time_budget_s):.1f}s "
                    f"(planner={planner_name}) although the endpoints are valid: try an "
                    "intermediate waypoint or a larger time budget."
                ),
            }

        path = pdef.getSolutionPath()
        if path is None:
            return None, {"reason_code": "planner_error", "reason": "planner reported success but returned no path"}

        _simplify_path(og, si, path)

        n = path.getStateCount()
        if n == 0:
            return None, {"reason_code": "planner_error", "reason": "planner returned an empty path"}
        return [
            np.array([float(path.getState(k)[i]) for i in range(nq)])
            for k in range(n)
        ], None

    attempts, _ = _path_candidate_policy()
    # A data buffer of its own: `col_data` belongs to the validity checker, and
    # scoring runs between solves that are still using it.
    score, target = _path_score_policy(model, model.createData(), base, yaw)
    return _pick_path_candidate(_solve, attempts, target, score=score, parallel=True)








# ─── Two-phase time parameterisation ──────────────────────────────────────────

def _search_resolution() -> tuple[Optional[float], float]:
    """The planner's step/range and its motion-check resolution. ONE owner.

    ``setStateValidityCheckingResolution`` takes a FRACTION of the state space's
    maximum extent, not radians. This cell's joint space spans roughly 20 rad, so
    the old default of 0.01 left about 0.2 rad — 11 degrees — between consecutive
    checks on the straight line joining two knots. The welding torch sits 0.9 m past
    the flange, so it sweeps a long way in 11 degrees, and a path could dip into a
    fixture voxel and back out entirely inside one unchecked interval. It did:
    ``robot2``/``W13`` was certified by the search while sitting 6.5 mm inside
    FIXTURE_A_v143, and the runtime monitor e-stopped on it mid-replay.

    0.005 is not a proof, it is a MEASUREMENT: it is the coarsest resolution at
    which this cell's known-bad path is rejected and a clear one found instead.
    Measured cost on four consecutive welds, 0.01 → 0.005 → 0.002 → 0.001:
    60 s → 115 s → 253 s → 668 s, so this is roughly a doubling of planning time.
    Finer is safer and slower; the backstop for whatever a finite resolution still
    hides is ``robot_kinematics._trajectory_collision_info``, which re-checks the
    emitted samples and fails the request rather than shipping a path that e-stops.

    Read from the environment on every call so a cell can be planned more carefully
    without a code change. Returned together because a step size and the resolution
    that validates it are one decision; three separate copies of this read is how
    two of the call sites would drift.
    """
    step_size = float(os.environ.get("OMPL_RANGE", "0.0")) or None
    validity_resolution = float(os.environ.get("OMPL_VALIDITY_RESOLUTION", "0.005"))
    return step_size, validity_resolution


def _path_deviation(rcfg):
    """robots.yaml ``path_following.max_deviation_rad`` (robot_kinematics owns the read)."""
    from agents.trajectory.tools.robot_kinematics import path_deviation_rad
    return path_deviation_rad(rcfg)


def _real_box(rcfg, model=None):
    """The arm's REAL joint box — unbounded sides infinite (``position_limits``)."""
    from agents.trajectory.tools.joint_limits import position_limits
    return position_limits(rcfg, model)


def _joint_box(rcfg, model=None):
    """The arm's joint box (``joint_limits`` owns it: robots.yaml ∩ the model) —
    what the timing may not leave between the planner's waypoints, the same box
    the planner searched in and the platform checks."""
    from agents.trajectory.tools.joint_limits import planning_box
    return planning_box(rcfg, model)


def _parameterize_two_phase(
    waypoints: list,
    seam_index: int,
    vel_limits=None,
    accel_limits=None,
    transit_velocity_scale: float = 0.5,
    approach_velocity_scale: float = 0.2,
    max_deviation_rad=None,
    position_limits=None,
) -> list[dict]:
    """Time-parameterise a path whose two phases run at DIFFERENT speeds.

    ``waypoints[:seam_index + 1]`` is the transit phase (normal clearance, normal
    speed); ``waypoints[seam_index:]`` is the terminal working stroke (reduced
    clearance, reduced speed). The seam waypoint belongs to both — it is where
    the phases meet.

    Each phase is parameterised independently by :func:`parameterize` (so TOPP-RA
    honours each phase's own scaled velocity/acceleration limits), then the
    approach phase's timestamps are OFFSET by the transit duration and the two
    are concatenated. The seam sample is emitted once, so the returned time base
    is continuous and strictly monotonic — the BT chain validator sees no gap.

    Every waypoint carries a ``"phase"`` of ``"transit"`` or ``"approach"`` so
    downstream consumers (executor, visualiser, audit) can tell which safety
    envelope each sample was planned under.

    ``seam_index <= 0`` or a seam at the final waypoint means "single phase" —
    the whole path is parameterised once at the transit speed and tagged
    ``"transit"``.
    """
    from agents.trajectory.tools.time_parameterization import parameterize

    n = len(waypoints)
    if n == 0:
        return []

    single_phase = seam_index <= 0 or seam_index >= n - 1
    if single_phase:
        traj = parameterize(
            waypoints,
            vel_limits=vel_limits,
            accel_limits=accel_limits,
            max_deviation_rad=max_deviation_rad,
            position_limits=position_limits,
            velocity_scale=transit_velocity_scale,
        )
        for pt in traj:
            pt["phase"] = "transit"
        return traj

    transit = parameterize(
        waypoints[:seam_index + 1],
        vel_limits=vel_limits,
        accel_limits=accel_limits,
        max_deviation_rad=max_deviation_rad,
        position_limits=position_limits,
        velocity_scale=transit_velocity_scale,
    )
    approach = parameterize(
        waypoints[seam_index:],
        vel_limits=vel_limits,
        accel_limits=accel_limits,
        max_deviation_rad=max_deviation_rad,
        position_limits=position_limits,
        velocity_scale=approach_velocity_scale,
    )
    if not transit:
        for pt in approach:
            pt["phase"] = "approach"
        return approach
    if not approach:
        for pt in transit:
            pt["phase"] = "transit"
        return transit

    for pt in transit:
        pt["phase"] = "transit"
    t_end = float(transit[-1]["time_s"])

    # Drop the approach phase's first sample: it IS the seam, already emitted as
    # the transit phase's last sample. Re-emitting it would duplicate a timestamp
    # and stall the executor for one tick.
    out = list(transit)
    for pt in approach[1:]:
        out.append({
            "positions": pt["positions"],
            "time_s": round(t_end + float(pt["time_s"]), 4),
            "phase": "approach",
        })
    return out




def _home_velocity_scale(cell_id: str, requested):
    """Return-to-HOME speed: the request, else the cell's
    ``planning.home_velocity_scale`` — capped by ``max_velocity_scaling``."""
    from agents.trajectory.tools.robot_kinematics import resolve_home_velocity_scale
    return resolve_home_velocity_scale(cell_id, requested)


# ─── Public entry point ───────────────────────────────────────────────────────

def plan_trajectories_ompl(
    cell_id: str,
    motion_requests: list[dict],
    approach_offset_m: float = 0.1,
    velocity_scale: Optional[float] = None,
    planner_name: Optional[str] = None,
    time_budget_s: Optional[float] = None,
    max_iterations: Optional[int] = None,  # accepted for interface parity (unused)
    moving_obstacle_spheres: Optional[list] = None,
    straight_descent: bool = True,
    **_ignored,  # the facade may pass options another backend uses (contract)
) -> dict:
    """Plan trajectories for one or more robots with OMPL. No RRT fallback.

    Pipeline per request (mirrors the RRT backend, swapping the planner core):
      Pinocchio IK (approach + target) → OMPL global search (per segment) →
      OMPL PathSimplifier → shared cubic-Hermite time-parameterisation →
      velocity scaling.

    Args:
        cell_id:            Cell identifier.
        motion_requests:    Per-robot requests, same schema as the RRT backend:
                            ``{"robot_ns", "target": {"x","y","z"},
                            "approach_vector": [..], "operation_type",
                            "operation_id"?, "current_joints"?, "approach"?}``.
                            ``approach: true`` marks the terminal working stroke,
                            so its final descent uses the reduced
                            ``approach_link_radius`` clearance from obstacles.yaml
                            AND the reduced approach speed (transit stays at
                            ``robot_link_radius`` / ``velocity_scale``).
                            Two optional PER-OPERATION overrides let the LLM tune
                            each operation independently of the others in the same
                            call: ``approach_offset_m`` (the standoff at which
                            transit hands over to the orientation-enforced
                            approach) and ``approach_velocity_scale`` (the speed of
                            that terminal stroke).
        approach_offset_m:  DEFAULT approach-waypoint offset along the approach
                            vector, for requests that don't set their own.
        velocity_scale:     Transit-phase speed scaling; ``None`` = the cell's
                            ``planning.max_velocity_scaling`` (robots.yaml), which
                            also caps any value given. The approach phase runs
                            at ``approach_velocity_scale`` (per request) or
                            ``velocity_scale * approach_velocity_fraction`` (cell
                            default from obstacles.yaml, 0.4).
        planner_name:       OMPL planner (default env ``OMPL_PLANNER`` /
                            ``RRTConnect``).
        time_budget_s:      Per-segment solve budget (default env
                            ``OMPL_TIME_BUDGET_S`` / 5.0).
        max_iterations:     Accepted for parity with the RRT backend's
                            ``max_rrt_iterations``; OMPL is time-budget driven,
                            so this is ignored.

    Returns:
        The standard planner result dict with ``"backend": "ompl"``.
    """
    t0 = time.monotonic()
    trajectories: dict[str, list[dict]] = {}
    failures: list[dict] = []
    liftoff_warnings: list[dict] = []

    def _fail(req: dict, entry: dict) -> None:
        """Record one per-robot failure, identifying WHICH operation it was.

        The agent's recovery ladder acts per operation — add a via point, reorient,
        hand the weld to the other arm — so a failure that names only the robot
        makes it guess. Stamped in ONE place: every site that reports a failure
        would otherwise have to remember to do it, and they did not.
        """
        if req.get("operation_id") is not None:
            entry.setdefault("operation_id", req["operation_id"])
        failures.append(entry)

    ompl = _import_ompl()
    if ompl is None:
        # No fallback: OMPL was requested but is not available.
        return {
            "success": False,
            "trajectories": {},
            "planning_time_s": round(time.monotonic() - t0, 3),
            "failures": [
                {
                    "robot_ns": req.get("robot_ns", "?"),
                    "error": (
                        "OMPL backend unavailable "
                        f"(import failed: {ompl_import_error()}). "
                        "Install the 'ompl' Python bindings. No fallback to RRT."
                    ),
                }
                for req in motion_requests
            ],
            "backend": "ompl",
        }

    # Reuse the shared kinematics/collision math — no duplication of IK, model
    # loading, the sphere model or timing. robot_kinematics owns all of it.
    from agents.trajectory.tools.robot_kinematics import (
        _load_robot_config,
        _load_pinocchio_model,
        _cell_sphere_radii,
        _config_spheres,
        load_static_obstacles,
        _load_clearance_policy,
        _load_planning_config,
        resolve_velocity_scale,
        _speed_margin,
        _target_obstacle_suggestion,
        _trajectory_collision_info,
        _collision_failure,
        _self_violation,
        seam_at_start,
        liftoff_blocked_failure,
        inter_robot_clearance,
    )
    from agents.trajectory.tools.joint_limits import planning_box
    from agents.trajectory.tools.pinocchio import run_ik as _ik
    from agents.trajectory.tools.trajectory_cache import (
        clear_last_pose, get_last_config, get_last_pose, set_last_config, set_last_pose,
        set_weld_pose,
    )
    from agents.trajectory.tools.time_parameterization import parameterize

    # Inter-robot coupling within a multi-robot call: as each arm is planned, its
    # swept-volume spheres are accumulated here and fed as moving obstacles to the
    # arms planned AFTER it in the SAME call, so a PARALLEL step (both arms in one
    # run_trajectory_planner call) is routed with each arm aware of the others.
    #
    # It is a SEARCH HEURISTIC, not the safety criterion — it used to be described
    # as making the arms unable to collide "regardless of timing", and that is not
    # true in either direction. Measured on ``bt_1787955158``: a route this tube
    # accepted reached 0.1551 m of arm-to-arm clearance, inside the cell's 0.2 m
    # band, while routes it REJECTED clear at 0.2488 m. It is time-blind — it
    # forbids being anywhere the other arm will ever be, not being there at the
    # same instant — so it is both too strict and too lax.
    #
    # ``inter_robot_clearance`` is the criterion. The tube's job is only to make the
    # common case land inside it on the first solve; when it costs height or admits
    # nothing, the leg is replanned without it and the gate adjudicates.
    SWEPT_R = 0.14          # slightly > link radius, to cover gaps between samples
    SWEPT_SAMPLES = 12      # trajectory samples per arm added to the swept volume
    planned_swept: list = []

    def _swept_spheres(model_, base_, yaw_, tool_cfg_, traj_) -> list:
        if not traj_:
            return []
        n = len(traj_)
        if n > 1:
            idxs = sorted({int(round(i * (n - 1) / (SWEPT_SAMPLES - 1)))
                           for i in range(SWEPT_SAMPLES)})
        else:
            idxs = [0]
        out: list = []
        seen: set = set()
        data_ = model_.createData()
        for i in idxs:
            q_i = np.array(traj_[i]["positions"][:model_.nq])
            for p in _config_spheres(model_, data_, q_i, base_, yaw_, tool_cfg_):
                key = (round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2))
                if key not in seen:
                    seen.add(key)
                    out.append((p, SWEPT_R))
        return out

    planner_name = planner_name or os.environ.get("OMPL_PLANNER", "RRTConnect")
    if time_budget_s is None:
        time_budget_s = float(os.environ.get("OMPL_TIME_BUDGET_S", "5.0"))
    # 0 → let the planner choose its own range/step.
    step_size, validity_resolution = _search_resolution()

    # Static obstacles — one owner (robot_kinematics), asked of the orchestrator.
    static_obstacles: list[dict] = load_static_obstacles(cell_id)
    # Cell clearance policy — ONE reading, owned by robot_kinematics, identical to
    # the one the runtime collision monitor performs. Clearance vs a box is
    # ``that sphere's radius + per-box margin + the phase speed allowance``: the
    # radius is geometry (arm vs tool) and never varies with the motion, while the
    # phase-dependent part is the speed allowance, because what a fast body needs
    # and a slow one does not is stopping distance, not a thinner casting.
    policy = _load_clearance_policy(cell_id)
    normal_radius = policy["normal"]
    # Two-phase SPEED policy. The terminal working stroke runs slower than transit:
    # that is what earns it the tighter clearance. Expressed as a fraction OF the
    # transit scale, per cell, overridable per operation via a request's
    # "approach_velocity_scale".
    cell_approach_vs_frac = policy["approach_velocity_fraction"]
    # The cell's speed ceiling (robots.yaml ``planning.max_velocity_scaling``).
    # No phase may be timed faster than it, whoever called.
    max_vs = _load_planning_config(cell_id)["max_velocity_scaling"]
    velocity_scale = resolve_velocity_scale(cell_id, velocity_scale)
    # Allowance for the arm's own speed, from the SAME formula the monitor applies
    # to the speed it measures on the wire.
    transit_margin = _speed_margin(cell_id, velocity_scale)

    # Pre-load models/configs once per robot.
    models: dict[str, tuple] = {}
    configs: dict[str, dict] = {}
    for req in motion_requests:
        ns = req["robot_ns"]
        if ns not in models:
            try:
                models[ns] = _load_pinocchio_model(cell_id, ns)
                configs[ns] = _load_robot_config(cell_id, ns)
            except Exception as e:
                _fail(req, {"robot_ns": ns, "error": str(e)})

    # A WORKLIST rather than a plain iteration, because one request can need two
    # passes: the first plans against the swept tube of the arms already done, and
    # if that tube is what forces a needlessly tall route (or admits none at all)
    # the request is re-inserted at the same position to be replanned WITHOUT it,
    # this time adjudicated by ``inter_robot_clearance`` — the gate execution
    # enforces. Re-inserting *here* rather than appending matters: the tube, the
    # chained start pose and ``set_last_config`` are all order-dependent, so the
    # retry has to happen in the arm's own slot.
    #
    # Each entry is ``[req, relax_inter_arm, conservative_traj]``, where the third
    # element carries the tube-planned leg forward so a relaxation the gate refuses
    # can fall back to it instead of failing the operation.
    work: list = [[req, False, None] for req in motion_requests]
    wi = 0
    while wi < len(work):
        req, relax_inter_arm, conservative_traj = work[wi]
        wi += 1
        # The tightest cancellation checkpoint in the system: one motion request's
        # solve is the longest a Stop can be ignored for.
        plan_cancellation.raise_if_cancelled(f"ompl request {wi}/{len(work)}")
        ns = req["robot_ns"]
        if ns not in models:
            continue

        model, _data = models[ns]
        rcfg = configs[ns]
        base = np.array(rcfg["base_position"])
        yaw = float(rcfg.get("base_orientation_yaw", 0.0))
        tool_cfg = rcfg.get("tool_collision")
        # Arm spheres at the arm radius, torch spheres at the torch radius — the
        # same vector the monitor builds for this robot.
        sphere_radii = _cell_sphere_radii(cell_id, model, tool_cfg)
        jl, ju = planning_box(rcfg, model)
        # The cell's declared home, not a literal: this is the pose we ASSUME the
        # arm is at when nothing else tells us, so a wrong guess plans the first
        # leg from somewhere the arm is not.
        HOME = np.array(_home_config(cell_id, ns, HOME_JOINTS_DEFAULT), float)

        tgt = req["target"]
        x, y, z = float(tgt["x"]), float(tgt["y"]), float(tgt["z"])
        av = np.array(req.get("approach_vector", [0, 0, -1]), dtype=float)
        av /= (np.linalg.norm(av) + 1e-9)

        # PER-OPERATION standoff: the LLM chooses, for THIS operation, where
        # normal transit hands over to the orientation-enforced approach. Falls
        # back to the call-level default, so two arms in one PARALLEL step can
        # use different standoffs (a deep-pocket weld needs a shorter one than an
        # open seam).
        try:
            req_offset = float(req.get("approach_offset_m", approach_offset_m))
        except (TypeError, ValueError):
            req_offset = approach_offset_m
        if not (req_offset > 0.0):
            req_offset = approach_offset_m
        # How far the arm LEAVES its previous weld (on-axis lift-off) before this
        # motion: the request's own `retreat_offset_m`, else the approach standoff.
        try:
            leave_offset = float(req.get("retreat_offset_m") or req_offset)
        except (TypeError, ValueError):
            leave_offset = req_offset
        if not (leave_offset > 0.0):
            leave_offset = req_offset

        # Approach waypoint (offset back along the approach vector).
        approach_x = x - av[0] * req_offset
        approach_y = y - av[1] * req_offset
        approach_z = z - av[2] * req_offset

        # Chain from where this robot's PREVIOUS operation ended, so the arm
        # moves weld-to-weld instead of returning to HOME between every weld.
        # Priority: explicit current_joints > last planned goal > HOME.
        chained_start = (
            req.get("current_joints")
            or get_last_config(cell_id, ns)
            or HOME.tolist()
        )
        # Warm-start IK from the chained pose too, so the target IK solution is
        # kinematically close to where the arm already is (smoother motion).
        warm = chained_start

        # Optional tool-orientation constraint (perpendicular / top-down weld).
        # When the request carries an orientation quaternion + constrain flag we
        # solve full-pose IK so the tool points down the approach vector.
        ori = req.get("orientation") or {}
        ik_ori = {}
        if req.get("constrain_orientation") and ori:
            ik_ori = {
                "qx": float(ori.get("x", 0.0)), "qy": float(ori.get("y", 0.707)),
                "qz": float(ori.get("z", 0.0)), "qw": float(ori.get("w", 0.707)),
                "constrain_orientation": True,
            }

        ik_target = _ik.invoke({
            "cell_id": cell_id, "robot_ns": ns,
            "x": x, "y": y, "z": z,
            "warm_start_joints": warm,
            **ik_ori,
        })
        if not ik_target.get("success"):
            _fail(req, {"robot_ns": ns, "reason_code": "ik_failed", "error": f"IK failed for target: {ik_target.get('reason')}"})
            continue

        ik_approach = _ik.invoke({
            "cell_id": cell_id, "robot_ns": ns,
            "x": approach_x, "y": approach_y, "z": approach_z,
            "warm_start_joints": warm,
            **ik_ori,
        })

        goal_q = np.array(ik_target["joint_positions"][:model.nq])
        start_q = np.array(chained_start[:model.nq])
        # A weld pose in which the arm folds into itself cannot be reached by any
        # path: say so now, by name, instead of letting the descent fail as an
        # anonymous "goal in collision" (tools/self_collision.py).
        self_hit = _self_violation(model, goal_q)
        if self_hit is not None:
            why = _collision_failure("goal", self_hit)
            _fail(req, {"robot_ns": ns, "reason_code": why["reason_code"],
                        "collision": why["collision"],
                        "error": f"OMPL ({planner_name}): {why['reason']}"})
            continue
        from agents.trajectory.tools.leg_builder import (
            approach_ik, approach_waypoint_failure,
        )
        via = (ik_approach["joint_positions"] if ik_approach.get("success") else
               approach_ik(_ik, cell_id, ns, (approach_x, approach_y, approach_z), ik_ori,
                           [goal_q.tolist()]))
        via_q = np.array(via[:model.nq]) if via else None
        if via_q is None and ik_ori and straight_descent and req.get("approach"):
            # No straight descent without it, and a joint-space entry fails the
            # working-stroke check at assembly: refuse now, with a stroke hint.
            entry = approach_waypoint_failure(ns, req_offset)
            entry["error"] = f"OMPL ({planner_name}): {entry['error']}"
            _fail(req, entry)
            continue

        # Inter-robot avoidance: the other arm's occupied space is passed in as
        # (centre, radius) spheres so this arm plans around it. Empty when only
        # one robot is being planned in isolation.
        #
        # FIRST pass — static caller-supplied moving obstacles + the swept volume of
        # every arm already planned in THIS call. Cheap and deliberately
        # conservative: the later arm avoids the earlier arm's whole path, so most
        # legs are admissible without the gate ever being consulted.
        #
        # RELAXED pass — the tube dropped. It is a time-BLIND proxy: it forbids this
        # arm from every place the other arm will EVER be, while the gate only
        # forbids being there at the same instant. Measured on ``bt_1787955158``,
        # that gap is what shipped a 1.335 m arc where a 0.000 m route clears the
        # gate at 0.2488 m. So when the tube costs height, the leg is replanned
        # without it and admitted only if ``inter_robot_clearance`` passes.
        obstacle_spheres: list = list(moving_obstacle_spheres or [])
        if not relax_inter_arm:
            obstacle_spheres += list(planned_swept)

        # Terminal working stroke? The LLM sets "approach": true so the final
        # descent runs at the slower approach speed — and therefore needs a smaller
        # speed allowance than transit. The SPHERES are the same either way: this is
        # the same arm and the same torch on both phases.
        use_approach = bool(req.get("approach", False))
        try:
            req_app_vs = req.get("approach_velocity_scale")
            approach_velocity_scale = (
                float(req_app_vs) if req_app_vs is not None
                else velocity_scale * cell_approach_vs_frac
            )
        except (TypeError, ValueError):
            approach_velocity_scale = velocity_scale * cell_approach_vs_frac
        approach_velocity_scale = max(min(approach_velocity_scale, max_vs), 0.01)
        # Computed BEFORE planning, not just before timing: the speed a phase will
        # run at is now part of the gate that phase is planned against.
        approach_margin = _speed_margin(
            cell_id, approach_velocity_scale if use_approach else velocity_scale)

        waypoints = [start_q]
        lift_end = 0           # last waypoint of the on-axis lift-off (0: none)

        # ── On-axis LIFT-OFF from the previous weld ───────────────────────────
        # The previous operation ended ON its seam (dwell + validate_approach both
        # need that), so this arm is still in contact. Leave along the SAME axis
        # it entered on, at that weld's OWN orientation, before any transit —
        # otherwise the transit drags the tool sideways out of the seam. Skipped
        # when the start is on no recorded seam, or the seam was a plain move. A
        # pinned current_joints still lifts off when its TCP is on a seam — that
        # is what a chain repair or a coupled re-plan pins (seam_at_start).
        retreat_from = seam_at_start(cell_id, ns, model, start_q, base, yaw,
                                     pinned=bool(req.get("current_joints")))
        # Already at or above the standoff: nothing left to lift.
        if retreat_from and float(retreat_from.get("rise_m") or 0.0) >= leave_offset - 1e-3:
            retreat_from = None
        if retreat_from and retreat_from.get("orientation") and straight_descent:
            prev_av = np.asarray(retreat_from["approach_vector"], float)
            prev_av /= (np.linalg.norm(prev_av) + 1e-9)
            prev_tgt = np.asarray(retreat_from["target"], float)
            # Where the lift-off starts: the TCP's place on the axis (the seam, or
            # part-way up the stroke when the start was pinned there).
            lift_from = np.asarray(retreat_from.get("from_point") or prev_tgt, float)
            prev_ori = retreat_from["orientation"]
            prev_ik_ori = {
                "qx": float(prev_ori.get("x", 0.0)), "qy": float(prev_ori.get("y", 0.707)),
                "qz": float(prev_ori.get("z", 0.0)), "qw": float(prev_ori.get("w", 0.707)),
                "constrain_orientation": True,
            }
            retreat_wp = _cartesian_retreat(
                model, _ik, cell_id, ns,
                lift_from, prev_tgt - prev_av * leave_offset, prev_ik_ori,
                waypoints[-1], base, yaw, obstacle_spheres, static_obstacles,
                normal_radius, tool_cfg,
                sphere_radii=sphere_radii, extra_margin=approach_margin,
            )
            if retreat_wp:
                # The lift-off leaves the arm at the previous weld's standoff;
                # everything downstream now plans from THERE.
                waypoints.extend(retreat_wp)
                start_q = waypoints[-1]
                lift_end = len(waypoints) - 1
            else:
                # A blocked lift-off FAILS the leg (shared rule, every backend):
                # a joint-space transit from the seam drags the tool out of it.
                # OMPL_ALLOW_BLOCKED_LIFTOFF=1 restores the old warn-and-drag.
                refusal = liftoff_blocked_failure(ns, retreat_from)
                if refusal is not None and not relax_inter_arm and planned_swept:
                    # Blocked by the other arms' time-BLIND tube, perhaps — the
                    # proxy every other failure here doubts first: retry against
                    # the real gate before refusing.
                    work.insert(wi, [req, True, None])
                    continue
                if refusal is not None:
                    _fail(req, refusal)
                    continue
                liftoff_warnings.append({
                    "robot_ns": ns, "operation_id": req.get("operation_id"),
                    "from_operation_id": retreat_from.get("operation_id"),
                    "warning": ("the on-axis lift-off from the previous weld is blocked "
                                "or unreachable; this leg leaves the seam with a "
                                "joint-space transit")})

        if via_q is not None:
            seg1_wp, _seg1_fail = _plan_ompl_segment(
                ompl, model, start_q, via_q, jl, ju, obstacle_spheres, base, yaw,
                static_obstacles, planner_name, time_budget_s, step_size, validity_resolution,
                tool_cfg=tool_cfg, sphere_radius=normal_radius,
                sphere_radii=sphere_radii, extra_margin=transit_margin,
                limit_box=_real_box(rcfg, model),
            )
            # A failed approach via-point is non-fatal — fall through to a direct
            # plan from the current pose. The target segment carries the verdict.
            if seg1_wp:
                waypoints.extend(seg1_wp[1:])
        # Index of the LAST transit waypoint == the phase seam. Everything after
        # it belongs to the approach phase. Recorded here because the two lists
        # are about to be concatenated and the boundary would otherwise be lost —
        # which is what previously made a two-phase SPEED impossible.
        seam_index = len(waypoints) - 1

        # Terminal working stroke: a STRAIGHT Cartesian descent from the approach
        # waypoint into the point along the approach vector, so the tool enters
        # ON-AXIS (a joint-space OMPL segment arrives laterally and fails
        # validate_approach). Falls back to the OMPL segment only when the request
        # carries no orientation constraint (a plain move, where on-axis entry is
        # not required).
        seg2_wp = None
        seg2_fail = None
        if via_q is not None and ik_ori and straight_descent:
            descent_why: dict = {}
            seg2_wp = _cartesian_descent(
                model, _ik, cell_id, ns,
                (approach_x, approach_y, approach_z), (x, y, z), ik_ori,
                waypoints[-1], base, yaw, obstacle_spheres, static_obstacles,
                normal_radius, tool_cfg,
                sphere_radii=sphere_radii, extra_margin=approach_margin,
                why=descent_why,
            )
            if seg2_wp is None:
                seg2_fail = _descent_failure(descent_why)
        if seg2_wp is None and seg2_fail is None:
            seg2_wp, seg2_fail = _plan_ompl_segment(
                ompl, model, waypoints[-1], goal_q, jl, ju, obstacle_spheres, base, yaw,
                static_obstacles, planner_name, time_budget_s, step_size, validity_resolution,
                tool_cfg=tool_cfg, sphere_radius=normal_radius,
                sphere_radii=sphere_radii, extra_margin=approach_margin,
                limit_box=_real_box(rcfg, model),
            )
        if seg2_wp is None:
            # No route at all under the tube. The tube is a proxy, and a proxy that
            # admits nothing is the first thing to doubt — retry against the real
            # gate before reporting a failure the LLM would have to route around.
            if (not relax_inter_arm and planned_swept and not str(
                    (seg2_fail or {}).get("reason_code") or "").endswith("_outside_joint_limits")):
                work.insert(wi, [req, True, None])
                continue
            seg2_fail = seg2_fail or {"reason": "unknown planning failure"}
            entry = {
                "robot_ns": ns,
                "error": f"OMPL ({planner_name}): {seg2_fail.get('reason')}",
                "reason_code": seg2_fail.get("reason_code"),
            }
            if "collision" in seg2_fail:
                entry["collision"] = seg2_fail["collision"]
            # If the GOAL is in collision, diagnose the weld POINT vs the boxes:
            # is the target itself inside / too close to a fixture box? If so,
            # emit an actionable move-the-weld-out suggestion.
            if seg2_fail.get("reason_code") == "goal_in_collision":
                diag = _target_obstacle_suggestion(
                    [x, y, z], av, static_obstacles, policy["tool"] + 0.05,
                )
                if diag is not None:
                    entry["target_diagnosis"] = diag
                    entry["suggestion"] = diag["suggestion"]
                    entry["error"] = (f"OMPL ({planner_name}): goal in collision with "
                                      f"'{diag.get('object_id')}' — {diag['suggestion']}")
            _fail(req, entry)
            continue
        waypoints.extend(seg2_wp[1:])

        # ── Two-phase time parameterisation ──────────────────────────────────
        # OMPL PathSimplifier already simplified each segment. Time-parameterise
        # with TOPP-RA (velocity/acceleration-limited, time-optimal) when the
        # robot's limits are known; otherwise the shared Hermite fallback is used.
        #
        # Transit and approach are parameterised SEPARATELY so each runs at its
        # own speed — the counterpart of their separate clearance radii. Only the
        # terminal working stroke of a request marked "approach" is slowed; a pure
        # transit move is one phase at the transit speed throughout.
        traj = _parameterize_two_phase(
            waypoints[lift_end:],
            seam_index=(seam_index - lift_end) if use_approach else 0,
            vel_limits=rcfg.get("joint_velocity_limits"),
            accel_limits=_accel_limits(cell_id, rcfg),
            transit_velocity_scale=velocity_scale,
            approach_velocity_scale=approach_velocity_scale,
            max_deviation_rad=_path_deviation(rcfg),
            position_limits=_joint_box(rcfg, model),
        )
        if lift_end:
            # The lift-off is the on-axis stroke out of the seam: timed EXACTLY on
            # its own (no corner of it rounded off the axis), then the rest.
            lift = parameterize(
                waypoints[:lift_end + 1],
                vel_limits=rcfg.get("joint_velocity_limits"),
                accel_limits=_accel_limits(cell_id, rcfg),
                max_deviation_rad=None,
                position_limits=_joint_box(rcfg, model),
                velocity_scale=velocity_scale,
            )
            for pt in lift:
                pt["phase"] = "transit"
            traj = _concat_timed([lift, traj])

        # ── Validate what is EMITTED, not only what was searched ──────────────
        # OMPL certifies a path by sampling it: the knots, plus the straight-line
        # motion between them at ``validity_resolution``. That resolution is a
        # FRACTION of the state space extent, so on this arm's ~20 rad joint space
        # even 0.01 leaves ~0.2 rad between checks — and the torch tip, 0.9 m past
        # the flange, sweeps a long way in 0.2 rad. A dip into a fixture voxel that
        # falls between two checked samples is simply not seen.
        #
        # Measured here: ``robot2``/``W13``'s certified path enters voxel
        # FIXTURE_A_v143 by 6.5 mm, invisible to the search, emitted, and then
        # caught by the runtime collision monitor as an e-stop five seconds into the
        # replay. So the emitted samples get re-checked with the same predicate the
        # monitor applies, and a breach FAILS the request. An availability failure
        # the agent can act on beats a trajectory that stops the cell mid-motion.
        breach = _trajectory_collision_info(
            model, model.createData(), traj, base, yaw,
            obstacle_spheres, static_obstacles,
            sphere_radius=normal_radius, sphere_radii=sphere_radii,
            transit_margin=transit_margin, approach_margin=approach_margin,
        )
        if breach is not None:
            _fail(req, {
                "robot_ns": ns,
                "error": (
                    f"OMPL ({planner_name}): the emitted trajectory breaches "
                    f"clearance at sample {breach.get('sample_index')} of "
                    f"{len(traj)} ({breach.get('phase')} phase): link "
                    f"{breach.get('link_index')} vs {breach.get('type')} "
                    f"'{breach.get('object_id')}' (signed clearance "
                    f"{breach.get('clearance_m')} m). The search accepted this path "
                    f"because the breach falls between the samples it checked at "
                    f"validity resolution {validity_resolution}. Stage the arm further "
                    f"from the obstacle, add an intermediate waypoint that routes "
                    f"around it, or lower OMPL_VALIDITY_RESOLUTION so the search "
                    f"itself rejects paths this close."
                ),
                "reason_code": "trajectory_in_collision",
                "collision": breach,
            })
            continue

        # ── Inter-arm admissibility, and how tall the leg turned out ───────────
        # A relaxed leg was planned with the other arms' swept tube removed, so
        # nothing has yet checked it against them. THE gate checks it now — the
        # same function ``validate_trajectory_set`` exposes, so "the planner
        # accepted it" and "the gate accepts it" cannot mean different things.
        if relax_inter_arm and trajectories:
            try:
                verdict = inter_robot_clearance(
                    cell_id, {**trajectories, ns: traj})
            except Exception as e:
                # A gate that cannot run has not said yes. Fail-closed: the relaxed
                # leg is unverified, so it does not ship.
                verdict = {"collision_free": False, "min_clearance_m": None,
                           "error": str(e)}
            if not verdict.get("collision_free"):
                if conservative_traj is None:
                    _fail(req, {
                        "robot_ns": ns,
                        "error": (
                            f"OMPL ({planner_name}): no route for this operation clears "
                            f"the other arms. Planning against their swept volume found "
                            f"none, and the direct route reaches "
                            f"{verdict.get('min_clearance_m')} m of arm-to-arm clearance "
                            f"at t={verdict.get('collision_time_s')} s, at or inside the "
                            f"cell's {verdict.get('warning_distance_m')} m warning band — "
                            f"the runtime monitor would E-STOP it. Stage the arms apart "
                            f"with find_interlocks, or make this step SEQUENTIAL."
                        ),
                        "reason_code": "inter_arm_clearance",
                        "collision": verdict,
                    })
                    continue
                # The height was real safety work after all: keep the conservative
                # leg. It may still be too tall, which the rise check below reports.
                traj = conservative_traj

        # Rise on the EMITTED array — the one that lands on the blackboard and the
        # one an operator watches. Candidate selection already preferred the
        # flattest path it found, but a preference cannot report anything: if every
        # candidate was tall, selection ships the least-tall and says nothing. This
        # is where a leg that is tall anyway becomes a fact the agent can act on.
        rise_m = _trajectory_tool_rise(model, traj, base, yaw)
        rise_limit = _emitted_rise_limit()
        if rise_limit > 0.0 and rise_m > rise_limit:
            # Not yet a failure if the time-blind tube is a plausible cause — try
            # the relaxed pass first and let the gate decide.
            if not relax_inter_arm and planned_swept:
                work.insert(wi, [req, True, traj])
                continue
            _fail(req, {
                "robot_ns": ns,
                "error": (
                    f"OMPL ({planner_name}): the emitted trajectory lifts the tool "
                    f"{rise_m:.3f} m above both of its endpoints, over the "
                    f"{rise_limit:.2f} m limit — a visible arc over the workpiece, not "
                    f"transit. Every candidate the search found was this tall"
                    + (" and the direct route is blocked by the other arms"
                       if relax_inter_arm else "")
                    + f". Add an intermediate waypoint that routes AROUND the obstacle "
                    f"instead of over it, move the operation to the arm on that side of "
                    f"the cell, or raise OMPL_MAX_EMITTED_RISE_M if this cell genuinely "
                    f"requires arcing over a fixture."
                ),
                "reason_code": "excessive_tool_rise",
                "tool_rise_m": round(rise_m, 4),
                "tool_rise_limit_m": rise_limit,
            })
            continue

        trajectories[ns] = traj
        # Remember where this arm ENDED so the next weld starts from here — the
        # last EMITTED sample, not the IK goal. The terminal stroke is a Cartesian
        # descent whose own IK lands 0.01–0.03 rad from ``goal_q``; chaining from
        # ``goal_q`` left exactly that gap at every seam, chain_trajectories refused
        # it, and the agent re-planned the whole program in order to close it.
        set_last_config(cell_id, ns, _trajectory_end(traj, goal_q))
        # Remember the WORKING POSE it ended in, so the next operation lifts off
        # along this same axis/orientation instead of dragging the tool sideways.
        # Only an orientation-constrained stroke leaves the tool engaged in a seam;
        # a plain move has nothing to retract along, so clear it.
        if ik_ori:
            set_last_pose(cell_id, ns, [x, y, z], av.tolist(), ori)
            set_weld_pose(cell_id, ns, req.get("operation_id"), [x, y, z], av.tolist(), ori,
                          stroke_m=max(req_offset, leave_offset))
        else:
            clear_last_pose(cell_id, ns)
        # Add this arm's swept volume so any arm planned later in the SAME call
        # (a PARALLEL step) plans around it and cannot collide with it.
        try:
            planned_swept.extend(_swept_spheres(model, base, yaw, tool_cfg, traj))
        except Exception:
            pass  # coupling is best-effort; never fail a plan because of it

    return {
        "success": len(trajectories) > 0,
        "trajectories": trajectories,
        "planning_time_s": round(time.monotonic() - t0, 3),
        "failures": failures,
        **({"liftoff_warnings": liftoff_warnings} if liftoff_warnings else {}),
        "backend": "ompl",
        "planner": planner_name,
    }


def plan_joint_path(
    cell_id: str,
    robot_ns: str,
    start_q,
    goal_q,
    moving_obstacle_spheres: Optional[list] = None,
    velocity_scale: Optional[float] = None,
    try_direct: bool = True,
) -> Optional[list]:
    """Plan a collision-free joint-space path from ``start_q`` to an explicit
    ``goal_q`` (e.g. the HOME configuration) and return a timed trajectory
    ``[{"positions": [...], "time_s": float}, ...]``, or None if planning fails.

    Unlike ``plan_trajectories_ompl`` (which IK-solves a Cartesian target), this
    targets an exact joint configuration — used to plan and execute the return
    to HOME instead of a blind single-waypoint home command."""
    import numpy as np
    velocity_scale = _home_velocity_scale(cell_id, velocity_scale)
    ompl = _import_ompl()
    if ompl is None:
        return None
    from agents.trajectory.tools.robot_kinematics import _load_robot_config, _load_pinocchio_model
    from agents.trajectory.tools.time_parameterization import parameterize

    try:
        model, _ = _load_pinocchio_model(cell_id, robot_ns)
        rcfg = _load_robot_config(cell_id, robot_ns)
    except Exception:
        return None
    base = np.array(rcfg["base_position"])
    yaw = float(rcfg.get("base_orientation_yaw", 0.0))
    tool_cfg = rcfg.get("tool_collision")
    from agents.trajectory.tools.joint_limits import planning_box
    jl, ju = planning_box(rcfg, model)

    from agents.trajectory.tools.robot_kinematics import (
        _cell_sphere_radii, _load_clearance_policy, _speed_margin,
        load_static_obstacles,
    )
    normal_radius = _load_clearance_policy(cell_id)["normal"]
    sphere_radii = _cell_sphere_radii(cell_id, model, tool_cfg)
    static_obstacles: list = load_static_obstacles(cell_id)

    planner_name = os.environ.get("OMPL_PLANNER", "RRTConnect")
    time_budget_s = float(os.environ.get("OMPL_TIME_BUDGET_S", "5.0"))
    step_size, validity_resolution = _search_resolution()

    # Returning to HOME is pure transit — always the normal clearance radius.
    wp, _fail = _plan_ompl_segment(
        ompl, model,
        np.array(start_q[:model.nq]), np.array(goal_q[:model.nq]),
        jl, ju, list(moving_obstacle_spheres or []), base, yaw,
        static_obstacles, planner_name, time_budget_s, step_size, validity_resolution,
        tool_cfg=tool_cfg, sphere_radius=normal_radius,
        sphere_radii=sphere_radii, extra_margin=_speed_margin(cell_id, velocity_scale),
        limit_box=_real_box(rcfg, model), try_direct=try_direct,
    )
    if not wp:
        return None
    return parameterize(
        wp,
        vel_limits=rcfg.get("joint_velocity_limits"),
        accel_limits=_accel_limits(cell_id, rcfg),
        max_deviation_rad=_path_deviation(rcfg),
        position_limits=_joint_box(rcfg, model),
        velocity_scale=velocity_scale,
    )


def _home_liftoff(
    cell_id: str,
    robot_ns: str,
    start_q,
    velocity_scale: Optional[float] = None,
    retreat_offset_m: Optional[float] = None,
    obstacle_spheres: Optional[list] = None,
) -> Optional[list]:
    """Joint waypoints lifting the tool OUT of the seam the arm just welded.

    THE one place a return-to-home leaves a weld. A program's last operation ends
    ON its seam (that is where the dwell happens and what ``validate_approach``
    measures), so the first motion of the return must retract along the SAME
    approach vector at the SAME orientation — otherwise the joint-space transit
    drags the tool sideways out of the seam.

    Both returns go through here: the single-arm :func:`plan_home_trajectory` and
    the coordinated :func:`plan_home_trajectories`. The coordinated one — the one
    the behaviour tree and the e-stop reset use — skipped the lift-off entirely,
    so which of the two planned a return decided whether the tool was dragged out
    of the weld. That is one decision, so it has one owner.

    ``retreat_offset_m`` defaults to the cell's ``planning.home_liftoff_m``.

    Returns the waypoints AFTER ``start_q``; an empty list means the arm is not on
    a seam (the caller plans straight from where it is). ``None`` means the arm IS
    on a seam and no on-axis lift-off could be planned — not at the requested
    stroke, not ignoring the other arms' swept volume (the step gate serialises
    the returns anyway), not at the cell's ``planning.min_home_liftoff_m``.
    Planning the transit from the seam then would drag the tool out of it, so the
    callers plan no return; the reason is recorded for
    ``planner_backend.home_refusal``, which the assembler and the home reset use
    to refuse the blind home too. ``OMPL_ALLOW_BLOCKED_LIFTOFF=1`` restores ``[]``
    (warn-and-drag). An arm with recorded welds whose cell or model cannot be read
    is treated as blocked, not as "not on a seam": that is the fail-closed answer.
    """
    velocity_scale = _home_velocity_scale(cell_id, velocity_scale)
    from agents.trajectory.tools.trajectory_cache import set_home_refusal
    set_home_refusal(cell_id, robot_ns, None)
    from agents.trajectory.tools.robot_kinematics import (
        _cell_sphere_radii, _load_clearance_policy, _load_pinocchio_model,
        _load_robot_config, _speed_margin, load_static_obstacles,
    )
    from agents.trajectory.tools.trajectory_cache import get_last_pose

    # No recorded seam pose means the last motion was not an orientation-
    # constrained stroke: there is nothing to retract out of. Checked BEFORE the
    # IK import, which pulls the LangChain-decorated tool surface in behind it —
    # an arm that is not on a seam must not need the LLM stack to go home.
    from agents.trajectory.tools.trajectory_cache import weld_poses
    if not ((get_last_pose(cell_id, robot_ns) or {}).get("orientation")
            or weld_poses(cell_id, robot_ns)):
        return []

    from agents.trajectory.tools.pinocchio import run_ik as _ik
    from agents.trajectory.tools.robot_kinematics import planning_setting, seam_at_start

    try:
        model, _ = _load_pinocchio_model(cell_id, robot_ns)
        rcfg = _load_robot_config(cell_id, robot_ns)
        policy = _load_clearance_policy(cell_id)
        if retreat_offset_m is None:
            retreat_offset_m = planning_setting(cell_id, "home_liftoff_m")
        min_liftoff = planning_setting(cell_id, "min_home_liftoff_m")
    except Exception as exc:  # noqa: BLE001
        # The arm may be standing on a seam (welds are recorded) and nothing can
        # say whether, or how to leave it: not "no seam" — blocked.
        set_home_refusal(cell_id, robot_ns, {
            "robot_ns": robot_ns, "reason_code": "home_liftoff_blocked",
            "error": (f"the lift-off out of the last weld could not be planned: the "
                      f"cell or the arm's model cannot be read ({type(exc).__name__}: "
                      f"{exc}); a return planned from the seam would drag the tool")},
            start=start_q)
        return None

    base = np.array(rcfg["base_position"])
    yaw = float(rcfg.get("base_orientation_yaw", 0.0))
    tool_cfg = rcfg.get("tool_collision")
    # The seam the arm is ON at start_q — not merely the most recent one: a
    # return planned from a pinned pose leaves the weld it is actually at.
    pose = seam_at_start(cell_id, robot_ns, model, start_q, base, yaw, pinned=True)
    if not (pose and pose.get("orientation")) \
            or float(pose.get("rise_m") or 0.0) >= retreat_offset_m - 1e-3:
        return []

    static_obstacles: list = load_static_obstacles(cell_id)

    # The lift-off out of the seam is an approach-speed stroke; the transit home
    # is not. Same spheres, different speed allowance.
    retreat_margin = _speed_margin(
        cell_id, velocity_scale * policy["approach_velocity_fraction"])

    av = np.asarray(pose["approach_vector"], float)
    av /= (np.linalg.norm(av) + 1e-9)
    tgt = np.asarray(pose["target"], float)
    lift_from = np.asarray(pose.get("from_point") or tgt, float)
    o = pose["orientation"]
    ik_ori = {
        "qx": float(o.get("x", 0.0)), "qy": float(o.get("y", 0.707)),
        "qz": float(o.get("z", 0.0)), "qw": float(o.get("w", 0.707)),
        "constrain_orientation": True,
    }
    radii = _cell_sphere_radii(cell_id, model, tool_cfg)
    q0 = np.array([float(v) for v in start_q][:model.nq])
    # Rungs: as requested; then without the other arms' swept volume (the
    # coordinated return's proxy — the step gate checks the real concurrency and
    # serialises); then the shortest stroke the cell still accepts as leaving.
    rise = float(pose.get("rise_m") or 0.0)
    rungs = [(float(retreat_offset_m), list(obstacle_spheres or []))]
    if obstacle_spheres:
        rungs.append((float(retreat_offset_m), []))
    short = min(float(retreat_offset_m), float(min_liftoff))
    if rise + 1e-3 < short < float(retreat_offset_m) - 1e-6:
        rungs.append((short, []))
    for off, obst in rungs:
        wps = _cartesian_retreat(
            model, _ik, cell_id, robot_ns,
            lift_from, tgt - av * off, ik_ori, q0, base, yaw,
            obst, static_obstacles, policy["normal"], tool_cfg,
            sphere_radii=radii, extra_margin=retreat_margin,
        )
        if wps:
            return list(wps)
    set_home_refusal(cell_id, robot_ns, start=start_q, why={
        "robot_ns": robot_ns, "reason_code": "home_liftoff_blocked",
        "from_operation_id": pose.get("operation_id"),
        "from_target": [float(v) for v in tgt],
        "error": ("the on-axis lift-off out of the last weld is blocked or unreachable "
                  "(an obstacle or IK) at every stroke down to "
                  f"{short * 1000:.0f} mm; a return planned from the seam would drag "
                  "the tool sideways out of it"),
    })
    if os.environ.get("OMPL_ALLOW_BLOCKED_LIFTOFF", "0") == "1":
        return []
    return None


def _home_leg_breach(cell_id: str, robot_ns: str, traj: list) -> Optional[dict]:
    """The first emitted sample of a return-home leg that breaches the cell's
    static obstacles or the arm itself — the step gate's own rule
    (``robot_kinematics.leg_static_breach``) — or None. A check that cannot run
    breaches nothing here; the step gate still runs it."""
    try:
        from agents.trajectory.tools.robot_kinematics import leg_static_breach
        return leg_static_breach(cell_id, robot_ns, traj)
    except Exception:  # noqa: BLE001
        return None


def _home_emit_retries() -> int:
    """Fresh solves of a return whose emitted leg breached (OMPL_HOME_EMIT_RETRIES)."""
    try:
        return max(0, int(os.environ.get("OMPL_HOME_EMIT_RETRIES", "8")))
    except ValueError:
        return 8


def _timed_liftoff(cell_id: str, robot_ns: str, start_q, head_wps: list,
                   velocity_scale: float) -> Optional[list]:
    """The lift-off waypoints as a standalone timed leg, starting at ``start_q``.

    :func:`plan_home_trajectory` parameterises the lift-off together with the
    transit as one waypoint list, which is strictly better. The coordinated return
    cannot: its transit comes back from :func:`plan_joint_path` already timed, so
    the retract is timed on its own and joined with :func:`_concat_timed`.

    Returns None when there is nothing to lift off, or when the arm's limits
    cannot be read — the caller then ships the transit alone rather than losing
    the whole return.
    """
    if not head_wps:
        return None
    from agents.trajectory.tools.robot_kinematics import _load_robot_config
    from agents.trajectory.tools.time_parameterization import parameterize
    try:
        rcfg = _load_robot_config(cell_id, robot_ns)
        from agents.trajectory.tools.robot_kinematics import _load_pinocchio_model
        model, _ = _load_pinocchio_model(cell_id, robot_ns)
        n = len(head_wps[0])
        wps = ([np.asarray([float(v) for v in start_q][:n], float)]
               + [np.asarray(w, float)[:n] for w in head_wps])
        # Exact, as every lift-off: the on-axis stroke out of the seam.
        return parameterize(
            wps,
            vel_limits=rcfg.get("joint_velocity_limits"),
            accel_limits=_accel_limits(cell_id, rcfg),
            max_deviation_rad=None,
            position_limits=_joint_box(rcfg, model),
            velocity_scale=velocity_scale,
        )
    except Exception:
        return None


def plan_home_trajectory(
    cell_id: str,
    robot_ns: str,
    start_q,
    home_q,
    velocity_scale: Optional[float] = None,
    retreat_offset_m: Optional[float] = None,
) -> Optional[list]:
    """Plan the return to HOME: on-axis lift-off, then a collision-free path home.

    The final leg of a program is a real motion and needs real planning. The arm
    finishes ON its last seam, so this mirrors the weld-to-weld rule:

      1. If that last operation was an orientation-constrained stroke, retract
         straight back out along ITS approach vector at ITS orientation, so the
         tool leaves the seam the way it entered rather than being dragged out.
      2. OMPL-plan joint-space from there to ``home_q``, avoiding the fixtures,
         workpiece and the arm's own tool.

    Both phases are time-parameterised as one continuous trajectory. When step 2
    finds no direct path, the return is STAGED (``_plan_staged_home``) rather than
    abandoned. Returns ``None`` only when planning is unavailable or no route at
    all exists — the caller then falls back to the executor's blind
    single-waypoint home, so this can only improve on the previous behaviour,
    never break it.
    """
    velocity_scale = _home_velocity_scale(cell_id, velocity_scale)
    ompl = _import_ompl()
    if ompl is None:
        return None
    from agents.trajectory.tools.robot_kinematics import _load_robot_config, _load_pinocchio_model
    from agents.trajectory.tools.time_parameterization import parameterize
    from agents.trajectory.tools.trajectory_cache import clear_last_pose

    try:
        model, _ = _load_pinocchio_model(cell_id, robot_ns)
        rcfg = _load_robot_config(cell_id, robot_ns)
    except Exception:
        return None
    base = np.array(rcfg["base_position"])
    yaw = float(rcfg.get("base_orientation_yaw", 0.0))
    tool_cfg = rcfg.get("tool_collision")
    from agents.trajectory.tools.joint_limits import planning_box
    jl, ju = planning_box(rcfg, model)

    from agents.trajectory.tools.robot_kinematics import (
        _cell_sphere_radii, _load_clearance_policy, _speed_margin,
        load_static_obstacles,
    )
    _policy = _load_clearance_policy(cell_id)
    normal_radius = _policy["normal"]
    sphere_radii = _cell_sphere_radii(cell_id, model, tool_cfg)
    static_obstacles: list = load_static_obstacles(cell_id)
    transit_margin = _speed_margin(cell_id, velocity_scale)

    planner_name = os.environ.get("OMPL_PLANNER", "RRTConnect")
    time_budget_s = float(os.environ.get("OMPL_TIME_BUDGET_S", "5.0"))
    step_size, validity_resolution = _search_resolution()

    q0 = np.array([float(v) for v in start_q][:model.nq])
    waypoints: list = [q0]

    # ── 1. On-axis lift-off from the last weld ────────────────────────────────
    head = _home_liftoff(cell_id, robot_ns, q0, velocity_scale=velocity_scale,
                         retreat_offset_m=retreat_offset_m)
    if head is None:
        # On a seam with no on-axis way out: no planned drag (home_refusal).
        print(f"[home-plan] {robot_ns}: lift-off out of the last weld is blocked; "
              f"no planned return", flush=True)
        return None
    waypoints.extend(head)

    # ── 2. Collision-free joint-space path to HOME ────────────────────────────
    wp, _fail = _plan_ompl_segment(
        ompl, model,
        waypoints[-1], np.array([float(v) for v in home_q][:model.nq]),
        jl, ju, [], base, yaw,
        static_obstacles, planner_name, time_budget_s, step_size, validity_resolution,
        tool_cfg=tool_cfg, sphere_radius=normal_radius,
        sphere_radii=sphere_radii, extra_margin=transit_margin,
        limit_box=_real_box(rcfg, model),
    )
    if not wp:
        # No direct path home from the retreat pose. Before surrendering the last
        # motion of the program to the executor's blind straight line, try staging
        # it: same obstacles, same validity checker, different route. The retreat
        # already computed above is kept — it is why the tool left the seam
        # cleanly — and the staged legs are stitched onto it.
        head = parameterize(
            waypoints,
            vel_limits=rcfg.get("joint_velocity_limits"),
            accel_limits=_accel_limits(cell_id, rcfg),
            max_deviation_rad=_path_deviation(rcfg),
            position_limits=_joint_box(rcfg, model),
            velocity_scale=velocity_scale,
        ) if len(waypoints) > 1 else None
        staged, _strategy = _plan_staged_home(
            cell_id, robot_ns, waypoints[-1],
            [float(v) for v in home_q][:model.nq],
            [], [normal_radius], velocity_scale, sphere_radius=normal_radius,
        )
        if not staged:
            return None
        traj = _concat_timed([head, staged]) if head else staged
        for pt in traj:
            pt.setdefault("phase", "transit")
        clear_last_pose(cell_id, robot_ns)
        return traj
    seg_from = len(waypoints) - 1          # where the joint-space transit starts
    waypoints.extend(wp[1:])

    def _timed(path: list) -> list:
        """The lift-off timed EXACTLY on its own (it is the on-axis stroke: no
        corner of it may be rounded off the axis), then the transit with the
        cell's path-following tolerance, as one leg — the coordinated return's
        recipe (``_timed_liftoff`` + ``_concat_timed``)."""
        lim = dict(vel_limits=rcfg.get("joint_velocity_limits"),
                   accel_limits=_accel_limits(cell_id, rcfg),
                   position_limits=_joint_box(rcfg, model), velocity_scale=velocity_scale)
        transit = parameterize(path[seg_from:], max_deviation_rad=_path_deviation(rcfg), **lim)
        if seg_from < 1:
            return transit
        lift = parameterize(path[:seg_from + 1], max_deviation_rad=None, **lim)
        return _concat_timed([lift, transit])

    traj = _timed(waypoints)
    for pt in traj:
        pt.setdefault("phase", "transit")
    # Validate what is EMITTED (see plan_home_trajectories): a leg that dips into a
    # fixture between its checked states is re-solved, or not planned at all — the
    # step gate would refuse it anyway.
    breach = _home_leg_breach(cell_id, robot_ns, traj)
    for _ in range(_home_emit_retries() if breach is not None else 0):
        wp2, _f = _plan_ompl_segment(
            ompl, model, waypoints[seg_from],
            np.array([float(v) for v in home_q][:model.nq]),
            jl, ju, [], base, yaw,
            static_obstacles, planner_name, time_budget_s, step_size, validity_resolution,
            tool_cfg=tool_cfg, sphere_radius=normal_radius,
            sphere_radii=sphere_radii, extra_margin=transit_margin,
            limit_box=_real_box(rcfg, model), try_direct=False,
        )
        if not wp2:
            continue
        cand = _timed(list(waypoints[:seg_from + 1]) + list(wp2[1:]))
        for pt in cand:
            pt.setdefault("phase", "transit")
        if _home_leg_breach(cell_id, robot_ns, cand) is None:
            traj, breach = cand, None
            break
    if breach is not None:
        print(f"[home-plan] {robot_ns}: the planned return breaches "
              f"'{breach.get('object_id')}' at sample {breach.get('sample_index')}; "
              f"no planned return", flush=True)
        from agents.trajectory.tools.trajectory_cache import set_home_refusal
        set_home_refusal(cell_id, robot_ns, start=start_q, why={
            "robot_ns": robot_ns, "reason_code": "home_return_refused",
            "collision": breach,
            "error": (f"every planned return breaches '{breach.get('object_id')}'; the "
                      f"blind straight line home would cross the same obstacle")})
        return None
    # The arm is parked at HOME now, not on a seam — nothing left to retract from.
    clear_last_pose(cell_id, robot_ns)
    return traj


# ─── Coordinated multi-arm return to HOME ─────────────────────────────────────

# Used only when a cell declares no home pose of its own. Re-exported from
# agents.home_pose so the executor's blind fallback, the robot agents and the
# planner cannot drift apart: an undeclared HOME is the same pose whether it was
# planned or not, because it is now literally the same constant.
from agents.home_pose import HOME_JOINTS_FALLBACK as HOME_JOINTS_DEFAULT





def _concat_timed(legs: Sequence) -> list:
    """Join separately time-parameterised legs into one monotonic trajectory.

    Each leg comes back from ``plan_joint_path`` starting at t=0, so later legs
    are shifted by the running end time. The duplicated junction point is dropped:
    a leg starts where the previous one ended, and two samples at the same instant
    read as a zero-length step to the executor's schedule check.

    The arm comes to rest at each staging pose. That is honest — it is a stop, and
    every leg is dynamically valid on its own — and it is what makes the stitched
    path safe to hand over: no leg's timing was invented here.
    """
    out: list = [dict(pt) for pt in legs[0]]
    for leg in legs[1:]:
        t0 = float(out[-1]["time_s"]) if out else 0.0
        for pt in list(leg)[1:]:
            out.append({**pt, "time_s": round(float(pt["time_s"]) + t0, 4)})
    return out


def _plan_staged_home(
    cell_id: str,
    robot_ns: str,
    start_q,
    home_q,
    swept_centres: Sequence,
    swept_radii: Sequence[float],
    velocity_scale: float,
    others: Optional[dict] = None,
    sphere_radius: float = 0.12,
):
    """Get an arm home VIA staging poses when no direct path could be found.

    OMPL failing means the direct problem has no solution at any margin we asked
    for — not that the arm cannot get home. ``home_strategy`` proposes where to
    route (deterministically first, then via the LLM); this function is the part
    that decides whether a proposal is real: every leg is planned by OMPL against
    the same static obstacles and the same swept volumes of the arms already
    planned, at the same graded margins. A strategy is accepted only when EVERY
    leg plans, and the last leg's goal is the cell's HOME — so a staged return
    lands on exactly the pose a direct one would have.

    Returns ``(trajectory, strategy)`` or ``(None, None)``. Nothing is returned
    that OMPL did not produce.
    """
    from agents.trajectory.tools.home_strategy import propose_home_strategies
    from agents.trajectory.tools.robot_kinematics import _load_robot_config

    try:
        rcfg = _load_robot_config(cell_id, robot_ns)
        lower = rcfg.get("joint_limits_lower")
        upper = rcfg.get("joint_limits_upper")
    except Exception:
        lower = upper = None

    margins = [r - sphere_radius for r in swept_radii]
    try:
        strategies = propose_home_strategies(
            cell_id, robot_ns, start_q, home_q, others=others,
            attempted_margins_m=margins, lower=lower, upper=upper)
    except Exception as exc:  # noqa: BLE001 — a strategy source is best-effort
        print(f"[home-strategy] {robot_ns}: no strategies available ({exc})",
              flush=True)
        return None, None

    for strategy in strategies:
        legs: list = []
        q = list(start_q)
        for target in list(strategy["waypoints"]) + [list(home_q)]:
            leg = None
            for R in swept_radii:
                leg = plan_joint_path(
                    cell_id, robot_ns, q, target,
                    moving_obstacle_spheres=[(p, R) for p in swept_centres],
                    velocity_scale=velocity_scale,
                )
                if leg:
                    break
            if not leg:
                legs = []
                break
            legs.append(leg)
            q = leg[-1]["positions"]
        if legs:
            print(f"[home-strategy] {robot_ns}: no direct path home; staged via "
                  f"{len(strategy['waypoints'])} pose(s) "
                  f"[{strategy.get('source', '?')}] — "
                  f"{strategy.get('rationale', '')}", flush=True)
            return _concat_timed(legs), strategy

    return None, None


def _home_fastpath_enabled() -> bool:
    # Opt-in, like the candidate solves: the code default is the coupled planner
    # (whose wiring the tests pin); the stack turns this on (docker-compose.yml).
    return os.environ.get("HOME_PARALLEL_FASTPATH", "0").strip().lower() not in (
        "0", "false", "no", "off", "")


def _fork_child(fn, conn) -> None:
    os.environ["_OMPL_CANDIDATE_CHILD"] = "1"
    try:
        payload = ("ok", fn())
    except BaseException as exc:  # noqa: BLE001
        payload = ("err", f"{exc!r}"[:300])
    try:
        conn.send(payload)
    finally:
        conn.close()
        os._exit(0)


def _fork_map(fns: list, timeout_s: float = 600.0) -> Optional[list]:
    """Run each zero-argument callable in its own forked child, at once.

    ``[result | None, ...]`` in order (None for a child that raised, timed out or
    died), or None when no batch can be set up — the caller then runs them in
    sequence. The same mechanism and opt-in as the candidate solves
    (``OMPL_PARALLEL_CANDIDATES``): each child inherits the loaded models and
    ships back only its result."""
    if len(fns) < 2 or not _parallel_enabled():
        return None
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        jobs = []
        for fn in fns:
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_fork_child, args=(fn, send), daemon=True)
            proc.start()
            send.close()
            jobs.append((proc, recv))
    except Exception:  # noqa: BLE001
        return None
    deadline = time.monotonic() + float(timeout_s)
    out: list = []
    for proc, recv in jobs:
        try:
            if not recv.poll(max(0.0, deadline - time.monotonic())):
                raise TimeoutError
            tag, value = recv.recv()
            out.append(value if tag == "ok" else None)
        except Exception:  # noqa: BLE001 — EOF, timeout, unpickling: no result
            out.append(None)
        finally:
            recv.close()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
    return out


def _parallel_direct_home(cell_id: str, order: list, starts: dict,
                          home_by_robot: Optional[dict],
                          velocity_scale: float) -> Optional[dict]:
    """Every arm's DIRECT return planned at once, accepted only if the gate runs
    the whole set in PARALLEL — else None, and the caller plans them coupled.

    The coupled planner below plans the arms ONE AFTER ANOTHER, each against the
    swept volume of those before it, walking a ladder of margins, a gate-checked
    direct route and staging for each. On welding_cell_2_small (4 arms, bt_1793277501)
    that made the final assembly 539 s. Most returns need none of it: each arm's
    direct route home is clear of the others, which is exactly the rung the
    coupled planner already accepts on the gate's word ("a route the gate has
    CHECKED beats one a proxy merely tolerated").

    So the direct routes are planned first, all arms at once (forked, like the
    candidate solves), and the set is kept only when it passes everything the
    coupled path would hold it to: an on-axis lift-off out of every seam, no tall
    arc (the emitted-rise limit), no fixture breach in the emitted samples, and
    ``_gate_multi_arm_step`` — the assembler's own gate — clearing ALL arms moving
    TOGETHER. Anything less (a SEQUENTIAL verdict included, which would cost
    cycle time) falls back to the coupled planner unchanged. Fail-closed: this
    path can only return a set the gate has passed. Opt-in:
    ``HOME_PARALLEL_FASTPATH=1`` (set by the stack's compose file)."""
    if len(order) < 2 or not _home_fastpath_enabled():
        return None
    try:
        import numpy as np
        from agents.trajectory.tools.robot_kinematics import (
            _load_pinocchio_model, _load_robot_config,
        )
        legs_in: dict = {}
        for ns in order:
            home_q = (home_by_robot or {}).get(ns) or _home_config(
                cell_id, ns, HOME_JOINTS_DEFAULT)
            head = _home_liftoff(cell_id, ns, starts[ns], velocity_scale=velocity_scale,
                                 obstacle_spheres=[])
            if head is None:
                return None     # no on-axis way out: the coupled path reports it
            lifted = [float(v) for v in head[-1]] if head else list(starts[ns])
            legs_in[ns] = (head, lifted, home_q)

        def _job(_ns):
            return lambda: plan_joint_path(cell_id, _ns, legs_in[_ns][1], legs_in[_ns][2],
                                           moving_obstacle_spheres=[],
                                           velocity_scale=velocity_scale)

        jobs = [_job(ns) for ns in order]
        transits = _fork_map(jobs)
        if transits is None:
            transits = [j() for j in jobs]

        rise_limit = _emitted_rise_limit()
        out: dict = {}
        for ns, t in zip(order, transits):
            if not t:
                return None
            head = legs_in[ns][0]
            if head:
                h = _timed_liftoff(cell_id, ns, starts[ns], head, velocity_scale)
                t = _concat_timed([h, t]) if h else list(t)
            if rise_limit > 0.0:
                model_, _ = _load_pinocchio_model(cell_id, ns)
                rcfg_ = _load_robot_config(cell_id, ns)
                rise = _trajectory_tool_rise(model_, t, np.array(rcfg_["base_position"]),
                                             float(rcfg_.get("base_orientation_yaw", 0.0)))
                if rise > rise_limit:
                    return None
            if _home_leg_breach(cell_id, ns, t) is not None:
                return None
            for pt in t:
                pt.setdefault("phase", "transit")
            out[ns] = t

        from agents.trajectory.tools.behaviour_tree import _gate_multi_arm_step
        rep = _gate_multi_arm_step(cell_id, out)
        if rep.get("mode") != "PARALLEL":
            return None
    except Exception as exc:  # noqa: BLE001 — a fast path that cannot finish has not said yes
        print(f"[home-plan] parallel direct returns not used ({type(exc).__name__}: "
              f"{exc}); planning them coupled", flush=True)
        return None
    try:
        from agents.trajectory.tools.trajectory_cache import clear_last_pose
        for ns in out:
            clear_last_pose(cell_id, ns)
    except Exception:  # noqa: BLE001
        pass
    par = rep.get("parallel") or {}
    print(f"[home-plan] {', '.join(order)}: direct returns planned at once; the gate "
          f"runs them together (clearance {par.get('min_clearance_m')} m against a "
          f"{par.get('warning_distance_m')} m band)", flush=True)
    return out


def plan_home_trajectories(
    cell_id: str,
    starts: dict,
    home_by_robot: Optional[dict] = None,
    velocity_scale: Optional[float] = None,
) -> dict:
    """Plan every arm's return to HOME as ONE coordinated, collision-free set.

    ``replay_plan``'s home reset drives all arms home simultaneously, and
    ``plan_home_trajectory`` plans a single arm against static obstacles only —
    so two arms could each hold an individually valid path that intersects the
    other's. Planning them together closes that: each arm is planned against the
    accumulated swept volume of the arms already planned, exactly as a PARALLEL
    weld step is coupled in ``plan_trajectories_ompl``. The result is safe
    regardless of the relative timing of the two motions.

    Args:
        cell_id:        Cell identifier.
        starts:         ``{robot_ns: [j1..jn]}`` — where each arm is NOW.
        home_by_robot:  Optional explicit HOME per arm; otherwise taken from the
                        cell's ``home_pose_deg`` (falling back to the module
                        default).
        velocity_scale: Timing scale for the returns (deliberately gentle).

    An arm with no direct path at any margin is not given up: the return is
    STAGED through intermediate poses (``_plan_staged_home``), every leg planned
    and checked exactly as a direct path would be.

    Returns:
        ``{robot_ns: trajectory}`` for every arm that could be planned, directly
        or via staging. An arm missing from the result has NO planned path — the
        caller must fall back to its blind home for that arm, and should say so.
    """
    import numpy as np

    velocity_scale = _home_velocity_scale(cell_id, velocity_scale)
    from agents.trajectory.tools.robot_kinematics import (
        _config_spheres, _load_pinocchio_model, _load_robot_config,
    )

    ompl = _import_ompl()
    if ompl is None:
        return {}

    # Longest travel first: the arm that has to cross the most space gets the
    # emptiest cell, and the shorter movers plan around it.
    def _travel(ns: str) -> float:
        try:
            goal = np.asarray(_home_config(cell_id, ns, HOME_JOINTS_DEFAULT), float)
            cur = np.asarray(starts[ns], float)
            n = min(len(goal), len(cur))
            return float(np.abs(goal[:n] - cur[:n]).sum())
        except Exception:
            return 0.0

    order = sorted(starts.keys(), key=_travel, reverse=True)

    # The swept radius is sized against the gate EXECUTION applies, not merely
    # non-penetration. The validity checker rejects at
    # ``d < SWEPT_R + sphere_radius`` while the monitor measures clearance as
    # ``d - 2*sphere_radius`` and e-stops at ``clearance <= warning_distance``.
    # Equating them gives ``warning_distance + sphere_radius``, so a path that
    # satisfies the checker clears the monitor by construction.
    #
    # But that margin is not always satisfiable: when the arms START inside the
    # warning band — exactly the state a prior e-stop leaves them in — NO return
    # can hold it, and demanding it makes planning fail. A failed plan falls back
    # to the BLIND home, which is strictly more dangerous than a planned path
    # with a smaller margin. So the margin is relaxed in steps and the first
    # radius that plans every arm wins: fail-closed on *safety* means always
    # returning the most-separated plan available, never no plan at all.
    from agents.trajectory.tools.robot_kinematics import _load_collision_config
    SPHERE_R = 0.12
    _warn = _load_collision_config(cell_id)["warning_distance"]
    swept_radii = sorted({round(max(_warn * f + SPHERE_R, SPHERE_R + 0.02), 4)
                          for f in (1.0, 0.75, 0.5, 0.25, 0.0)}, reverse=True)

    SWEPT_SAMPLES = 12
    planned_swept: list = []
    out: dict = {}

    fast = _parallel_direct_home(cell_id, order, starts, home_by_robot, velocity_scale)
    if fast is not None:
        return fast

    for ns in order:
        home_q = (home_by_robot or {}).get(ns) or _home_config(
            cell_id, ns, HOME_JOINTS_DEFAULT)

        # Leave the seam the way the tool entered it, exactly as the single-arm
        # return does — a return home is an ordinary planned motion, and its first
        # leg is a retract. Checked against the arms already planned, so the
        # lift-off itself is not the motion that breaches. Empty when the arm is
        # not parked on a seam; the transit then starts from where it stands.
        head_wps = _home_liftoff(
            cell_id, ns, starts[ns],
            velocity_scale=velocity_scale,
            obstacle_spheres=[(p, swept_radii[0]) for p, _ in planned_swept],
        )
        if head_wps is None:
            # No on-axis way out of the seam: no planned return for this arm, and
            # the reason is recorded (home_refusal) so the blind home that
            # would replace it is refused too.
            print(f"[home-plan] {ns}: lift-off out of the last weld is blocked; "
                  f"no planned return for this arm", flush=True)
            continue
        lifted_q = [float(v) for v in head_wps[-1]] if head_wps else list(starts[ns])

        # The retract and the transit ship as ONE motion, so height and clearance
        # are only meaningful on the concatenated leg — that is the artefact the
        # executor streams and the monitor watches.
        def _with_head(t, _ns=ns):
            if not t:
                return None
            h = _timed_liftoff(cell_id, _ns, starts[_ns], head_wps, velocity_scale)
            return _concat_timed([h, t]) if h else list(t)

        def _leg_rise(t, _ns=ns):
            if not t:
                return 0.0
            try:
                model_, _ = _load_pinocchio_model(cell_id, _ns)
                rcfg_ = _load_robot_config(cell_id, _ns)
                return _trajectory_tool_rise(
                    model_, t, np.array(rcfg_["base_position"]),
                    float(rcfg_.get("base_orientation_yaw", 0.0)))
            except Exception:
                return 0.0

        # Widest margin first; shrink only as far as needed to find a path.
        traj = None
        for SWEPT_R in swept_radii:
            traj = plan_joint_path(
                cell_id, ns, lifted_q, home_q,
                moving_obstacle_spheres=[(p, SWEPT_R) for p, _ in planned_swept],
                velocity_scale=velocity_scale,
            )
            if traj:
                break

        # ── Gate-verified rung ────────────────────────────────────────────────
        # Every rung above narrows a time-BLIND proxy: it forbids this arm from
        # everywhere the other arms will ever be. Measured on ``bt_1787955158``,
        # ``robot2``'s return found NO path at the full margin and the ladder's
        # fallback shipped a 0.585 m arc, while the direct route rises 0.000 m and
        # clears the real gate at 0.5314 m against a 0.2 m band.
        #
        # So before shrinking margins buys an unverified path, try the direct route
        # and ask the gate — a route the gate has CHECKED beats one a proxy merely
        # tolerated. It is admitted only on a pass, so this widens what can be
        # planned without widening what is allowed to move.
        rise_limit = _emitted_rise_limit()
        full = _with_head(traj)
        if planned_swept and (
                full is None or (rise_limit > 0.0 and _leg_rise(full) > rise_limit)):
            cand = plan_joint_path(
                cell_id, ns, lifted_q, home_q,
                moving_obstacle_spheres=[],
                velocity_scale=velocity_scale,
            )
            cand_full = _with_head(cand)
            if cand_full is not None and (
                    full is None or _leg_rise(cand_full) < _leg_rise(full)):
                # ASK THE OWNER OF CONCURRENCY, not a parallel-only proxy.
                # ``inter_robot_clearance`` answers "are these legs clear if they
                # run AT THE SAME TIME?", and that was the wrong question: nothing
                # runs this set until ``_gate_multi_arm_step`` has decided whether
                # it runs together or one at a time, and both consumers of this
                # function (``behaviour_tree._home_step``,
                # ``home_reset.home_reset_tree``) put the emitted legs through
                # exactly that ladder and serialise when parallel is unsafe.
                #
                # Asking the narrow question made this planner refuse routes
                # execution would never have run in the arrangement the refusal
                # assumed. Measured on ``bt_1788022470``: robot2's flat direct
                # return (rise 0.000 m, 0.56 rad of excess travel) is NOT clear in
                # parallel with robot1's (-0.2133 m) but IS clear serialised
                # robot2 -> robot1 (0.7389 m and 1.2699 m against a 0.2 m band) —
                # which is the order the shipped tree runs them in anyway. Refusing
                # it sent the arm through the staging detour the operator reported:
                # x4.56 the straight-line distance, 36 direction reversals, 18 s
                # for a move the direct route makes in 3.9 s.
                #
                # Fail-closed is unchanged. ``mode=None`` is a refusal, an exception
                # is a refusal, and a set accepted here is accepted on the same
                # verdict the assembler will independently reach — one owner, so
                # the two cannot disagree.
                from agents.trajectory.tools.behaviour_tree import (
                    _gate_multi_arm_step)
                try:
                    rep = _gate_multi_arm_step(cell_id, {**out, ns: cand_full})
                except Exception as e:
                    # A gate that cannot run has not said yes.
                    rep = {"mode": None, "error": str(e)}
                if rep.get("mode") is not None:
                    par = rep.get("parallel") or {}
                    how = ("in parallel" if rep.get("mode") == "PARALLEL"
                           else "serialised " + " -> ".join(rep.get("order") or []))
                    print(f"[home-plan] {ns}: the swept-volume route "
                          + ("did not exist" if traj is None
                             else f"arced {_leg_rise(full):.2f} m")
                          + f"; using the direct route instead — the gate clears the "
                          f"set {how} (parallel clearance "
                          f"{par.get('min_clearance_m')} m against a "
                          f"{par.get('warning_distance_m')} m band)", flush=True)
                    traj, full = cand, cand_full
                    # For the arms planned after this one, this leg is an obstacle at
                    # the widest margin like any other; how it was routed is not
                    # their business.
                    SWEPT_R = swept_radii[0]

        def _staged(_ns=ns):
            return _plan_staged_home(
                cell_id, _ns, lifted_q, home_q,
                [p for p, _ in planned_swept], swept_radii, velocity_scale,
                others={o: starts[o] for o in starts if o != _ns},
                sphere_radius=SPHERE_R,
            )

        staged = None
        if not traj:
            # No direct path at any margin. Before giving this arm up to the
            # executor's blind straight line — the one motion nothing has checked —
            # try routing it through staging poses.
            traj, staged = _staged()
            full = _with_head(traj)
        elif rise_limit > 0.0 and _leg_rise(full) > rise_limit:
            # A route exists but it ARCS, and the direct route was either refused by
            # the gate or no flatter. Staging is the remaining option and the right
            # one: it routes AROUND the obstruction rather than over it — the same
            # advice ``excessive_tool_rise`` gives the agent for a weld leg. Measured
            # on these poses, the tube ladder's leg rose 0.585 m where the staged
            # route rises 0.000 m.
            #
            # Only taken if it is actually flatter. Staging costs a stop at each pose,
            # so it must buy something.
            #
            # AND only if it does not pay for the height with travel. This rung used
            # to compare rise alone, which is how it shipped the leg an operator
            # reported: it bought 0.70 m of rise for 9.7 rad of extra travel and
            # logged the trade as a win. A dimension nobody measures is the one that
            # drifts — the same lesson as the rise fix itself, one rung later.
            #
            # The tolerance is the measured cost of a legitimate staged route, not a
            # round number: on this cell the home legs nobody objected to spend up to
            # 1.20 rad beyond the straight line, so a staged route may add that much
            # and no more. The complained-about leg spent 10.87.
            st_traj, st = _staged()
            st_full = _with_head(st_traj)
            if st_full is not None and _leg_rise(st_full) < _leg_rise(full):
                excess = _trajectory_excess_travel(full)
                st_excess = _trajectory_excess_travel(st_full)
                if st_excess > excess + _staging_extra_travel_limit():
                    print(f"[home-plan] {ns}: the direct route arced "
                          f"{_leg_rise(full):.2f} m, but staging it would wander "
                          f"{st_excess:.2f} rad beyond the straight line against "
                          f"{excess:.2f} rad — keeping the arcing route, which is "
                          f"the lesser of two visible defects", flush=True)
                else:
                    print(f"[home-plan] {ns}: the direct route arced "
                          f"{_leg_rise(full):.2f} m; staged via "
                          f"{len((st or {}).get('waypoints') or [])} pose(s) instead, "
                          f"rising {_leg_rise(st_full):.2f} m and wandering "
                          f"{st_excess:.2f} rad against {excess:.2f} rad", flush=True)
                    traj, staged, full = st_traj, st, st_full
        if not traj:
            continue
        # ONE trajectory, so the executor dispatches the retract and the transit as
        # a single continuous motion starting where the arm IS. ``full`` already
        # holds it unless the staged fallback replaced ``traj`` after it was built.
        traj = full if full is not None else _with_head(traj)
        for pt in traj:
            pt.setdefault("phase", "transit")
        # Validate what is EMITTED, as every planned leg is: the search certified
        # the path at its resolution, and a dip into a fixture between two checked
        # states is what the step gate (and the monitor) then find. Re-route rather
        # than ship it: staging, then fresh solves of the direct route.
        breach = _home_leg_breach(cell_id, ns, traj)
        if breach is not None:
            fixed = None
            tries = [] if staged is not None else [lambda: _with_head(_staged()[0])]
            # Fresh SEARCHES: the straight-line shortcut would hand back the very
            # line whose samples were just refused.
            tries += [lambda: _with_head(plan_joint_path(
                cell_id, ns, lifted_q, home_q,
                moving_obstacle_spheres=[(p, SWEPT_R) for p, _ in planned_swept],
                velocity_scale=velocity_scale, try_direct=False))] * _home_emit_retries()
            base_excess = _trajectory_excess_travel(traj)

            def _acceptable(cand) -> bool:
                """The repair is held to what the leg it replaces was held to:
                no breach, no tall arc, no staging detour, and — with other arms
                planned — a set the step gate clears."""
                if not cand or _home_leg_breach(cell_id, ns, cand) is not None:
                    return False
                if rise_limit > 0.0 and _leg_rise(cand) > rise_limit:
                    return False
                if _trajectory_excess_travel(cand) > base_excess + _staging_extra_travel_limit():
                    return False
                if out:
                    from agents.trajectory.tools.behaviour_tree import _gate_multi_arm_step
                    try:
                        if _gate_multi_arm_step(cell_id, {**out, ns: cand}).get("mode") is None:
                            return False
                    except Exception:  # noqa: BLE001 — a gate that cannot run has not said yes
                        return False
                return True

            for attempt in tries:
                cand = attempt()
                if _acceptable(cand):
                    fixed = cand
                    break
            if fixed is None:
                print(f"[home-plan] {ns}: every planned return breaches "
                      f"'{breach.get('object_id')}' (sample {breach.get('sample_index')}, "
                      f"{breach.get('clearance_m')} m); no planned return for this arm",
                      flush=True)
                from agents.trajectory.tools.trajectory_cache import set_home_refusal
                set_home_refusal(cell_id, ns, start=starts[ns], why={
                    "robot_ns": ns, "reason_code": "home_return_refused",
                    "collision": breach,
                    "error": (f"every planned return breaches "
                              f"'{breach.get('object_id')}'; the blind straight line "
                              f"home would cross the same obstacle")})
                continue
            traj = fixed
            for pt in traj:
                pt.setdefault("phase", "transit")
        out[ns] = traj
        # Parked at HOME now, not on a seam — nothing left for the next motion to
        # retract from. Cleared here and not before the retry loop, which re-reads
        # the seam pose on every margin attempt.
        try:
            from agents.trajectory.tools.trajectory_cache import clear_last_pose
            clear_last_pose(cell_id, ns)
        except Exception:
            pass
        if staged is None and SWEPT_R < swept_radii[0]:
            print(f"[home-plan] {ns}: no path at the full "
                  f"{swept_radii[0] - SPHERE_R:.2f} m margin; planned at "
                  f"{SWEPT_R - SPHERE_R:.2f} m — the runtime monitor may warn on "
                  f"this return", flush=True)

        # Feed this arm's whole swept path forward as an obstacle for the rest.
        try:
            model, _ = _load_pinocchio_model(cell_id, ns)
            rcfg = _load_robot_config(cell_id, ns)
            base = np.array(rcfg["base_position"])
            yaw = float(rcfg.get("base_orientation_yaw", 0.0))
            tool_cfg = rcfg.get("tool_collision")
            data = model.createData()
            n = len(traj)
            idxs = (sorted({int(round(i * (n - 1) / (SWEPT_SAMPLES - 1)))
                            for i in range(SWEPT_SAMPLES)}) if n > 1 else [0])
            seen: set = set()
            for i in idxs:
                q_i = np.array(traj[i]["positions"][:model.nq])
                for p in _config_spheres(model, data, q_i, base, yaw, tool_cfg):
                    key = (round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2))
                    if key not in seen:
                        seen.add(key)
                        # Radius is applied at planning time (the retry loop
                        # re-sizes it), so only the centre is accumulated.
                        planned_swept.append((p, SWEPT_R))
        except Exception:
            # Without this arm's swept volume the remaining arms are planned
            # against less information — never against wrong information.
            pass

    return out
