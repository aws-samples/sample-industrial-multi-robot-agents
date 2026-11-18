"""
``ompl-compound`` — centralised multi-arm planning in ONE OMPL compound state space.

The ``ompl`` backend plans the arms of a PARALLEL step one after another: each
later arm avoids a time-blind swept tube of the earlier ones, and
``inter_robot_clearance`` adjudicates afterwards. That is a heuristic, and on
``welding_cell_2_small`` it is what most planning runs spend their failures on
(``REFUSED`` / ``SEQUENTIAL`` step gates, re-plans, exhausted step budgets).

This backend plans the step's TRANSIT for all moving arms at once instead:

1. **One space.** ``ob.CompoundStateSpace`` with one ``RealVectorStateSpace(nq)``
   subspace per moving arm (weight 1.0), bounded by each arm's joint limits.
2. **One validity checker** that unpacks the compound state, runs forward
   kinematics for every arm and rejects a state if any arm collides with the
   environment (the same ``_config_is_free`` predicate as ``ompl``) or any two
   arms come closer than the cell's ``collision.warning_distance`` — the same
   pair criterion the runtime monitor e-stops on.
3. **One time base.** The compound path is time-parameterised as ONE stacked
   vector, so every arm's transit samples share their timestamps: the arms are
   collision-free *at each instant*, not merely along their paths.

What stays per arm, exactly as ``ompl`` does it (ADR-0001): the on-axis Cartesian
lift-off from the previous weld and the orientation-constrained descent into the
next one. Those strokes are short and constrained; planning them in 12-D buys
nothing. The arms' lift-offs are padded to end together, the joint transit
starts at that instant, and the descents start together when it ends. The whole
timeline is then checked against every other arm on one clock with
``inter_robot_clearance`` — the gate execution enforces — and a step that fails
it is refused as a WHOLE: dropping one arm would invalidate the other arm's
guarantee, which assumed the first one moved.

Allocation, skills and the agent loop are untouched: this module only answers
``dispatch_plan`` for one step's ``motion_requests``.

Every emitted trajectory's first sample carries ``sync_group`` / ``sync_members``,
so the cell executor's optional lockstep mode can start and supervise the legs
of one compound step together (see ``agents/cell/bt_executor.py``).

Environment knobs (all optional):
  OMPL_PLANNER                      planner name, shared with ``ompl``
  OMPL_COMPOUND_TIME_BUDGET_S       seconds per solve (default 5 s x moving arms)
  OMPL_COMPOUND_ATTEMPTS            independent solves, best (flattest) kept
                                    (default OMPL_PATH_CANDIDATES, 4)
  OMPL_PARALLEL_CANDIDATES          shared with ``ompl``: 1 runs those solves at
                                    once in forked children (one budget of wall
                                    time instead of up to ATTEMPTS of them)
  OMPL_PATH_MAX_RISE_M              shared with ``ompl``: a candidate this flat
                                    ends the search early
  OMPL_COMPOUND_INTER_ARM_EXTRA_M   clearance kept beyond warning_distance in the
                                    search, absorbing interpolation (0.01 m)
  OMPL_VALIDITY_RESOLUTION          shared with ``ompl``; divided by the number of
                                    moving arms, because a compound space's extent
                                    is the sum of its subspaces'
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Callable, Optional, Sequence

import numpy as np

BACKEND_NAME = "ompl-compound"

#: Arm-vs-arm sphere radius. Uniform on purpose, like the gate and the monitor
#: (robot_kinematics.inter_robot_clearance): arm separation is one number.
PAIR_RADIUS_M = 0.12


# ─── pure helpers (no OMPL, no Pinocchio) ────────────────────────────────────

def split_requests(motion_requests: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """``(one request per robot, rejected duplicates)``.

    One compound query moves each arm once. Two requests for the same arm in one
    call are a SEQUENCE, which is what ``steps`` is for — and the cache keeps one
    operation per robot per call anyway, so a silent second leg would be lost.
    Every request for a duplicated robot is rejected, not just the later ones:
    which one the caller meant is not ours to guess.
    """
    counts: dict[str, int] = {}
    for req in motion_requests:
        ns = req.get("robot_ns")
        counts[ns] = counts.get(ns, 0) + 1
    keep = [r for r in motion_requests if counts.get(r.get("robot_ns")) == 1]
    dup = [r for r in motion_requests if counts.get(r.get("robot_ns"), 0) > 1]
    return keep, dup


def pair_clearance(spheres_a, spheres_b, radius: float = PAIR_RADIUS_M) -> float:
    """Minimum surface clearance between two sphere sets at one instant.

    The vectorised form of ``robot_kinematics._check_robot_pair_collision``:
    ``min ||a - b|| - 2 * radius``. Called for every arm pair in every state the
    search checks, so it must not be a Python double loop.
    """
    a = np.asarray(spheres_a, float).reshape(-1, 3)
    b = np.asarray(spheres_b, float).reshape(-1, 3)
    if not len(a) or not len(b):
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min()) - 2.0 * radius


def stack(configs: Sequence[Sequence[float]]) -> list[float]:
    return [float(v) for q in configs for v in q]


def unstack(vector: Sequence[float], sizes: Sequence[int]) -> list[list[float]]:
    out, i = [], 0
    for n in sizes:
        out.append([float(v) for v in vector[i:i + n]])
        i += n
    return out


def _shifted(samples: list[dict], offset: float, phase: str,
             drop_first: bool) -> list[dict]:
    pts = samples[1:] if drop_first else samples
    return [{"positions": list(p["positions"]),
             "time_s": round(offset + float(p["time_s"]), 4),
             "phase": phase} for p in pts]


def compose_timelines(retracts: dict, transit: list[dict], sizes: dict,
                      descents: dict, descent_phases: dict) -> dict:
    """Join per-arm lift-offs, the shared transit and per-arm descents.

    ``retracts`` / ``descents``: ``{ns: [timed samples from t=0]}`` (may be
    empty). ``transit``: timed samples of the STACKED vector, arms in ``sizes``
    order. The lift-offs are padded with a hold to end together, so the joint
    transit starts at one instant for every arm and its samples keep the same
    timestamps on every arm; the descents start together when it ends. An arm
    that finishes its descent first simply ends — the gate and the monitor both
    treat a finished arm as parked at its last sample.
    """
    names = list(sizes)
    t_lift = max([float(r[-1]["time_s"]) for r in retracts.values() if r] or [0.0])
    per_arm_transit = {ns: [] for ns in names}
    for pt in transit:
        for ns, q in zip(names, unstack(pt["positions"], [sizes[n] for n in names])):
            per_arm_transit[ns].append({"positions": q, "time_s": pt["time_s"]})
    t_transit = float(transit[-1]["time_s"]) if transit else 0.0

    out: dict = {}
    for ns in names:
        traj: list[dict] = []
        lift = retracts.get(ns) or []
        if lift:
            traj += _shifted(lift, 0.0, "transit", drop_first=False)
            if float(lift[-1]["time_s"]) < t_lift - 1e-6:
                # Hold the standoff until every arm has lifted off.
                traj.append({"positions": list(lift[-1]["positions"]),
                             "time_s": round(t_lift, 4), "phase": "transit"})
        traj += _shifted(per_arm_transit[ns], t_lift, "transit", drop_first=bool(traj))
        if t_lift > 0.0 and not lift and per_arm_transit[ns]:
            # No lift-off of its own: wait at the start while the others lift.
            traj.insert(0, {"positions": list(per_arm_transit[ns][0]["positions"]),
                            "time_s": 0.0, "phase": "transit"})
        descent = descents.get(ns) or []
        if descent:
            traj += _shifted(descent, t_lift + t_transit,
                             descent_phases.get(ns, "approach"), drop_first=True)
        out[ns] = traj
    return out


#: How far the leading arm gets before the next one starts, for staggered
#: candidates: 1.0 = strictly one after the other, 0.5 = half-way, ...
STAGGER_LEADS = (0.5, 0.75, 1.0)


def staggered_paths(starts: Sequence, goals: Sequence) -> list[list[list[float]]]:
    """Straight-line paths in which one arm leads the others.

    For each arm ``k`` and lead ``f``: arm ``k`` first covers fraction ``f`` of
    its own straight line while the others wait, then everyone finishes
    together. ``f = 1.0`` is "arm k goes first". Each candidate is
    ``[start, intermediate, goal]`` as stacked vectors; collision is the
    checker's business, not this function's.
    """
    s = [np.asarray(q, float) for q in starts]
    g = [np.asarray(q, float) for q in goals]
    out = []
    for k in range(len(s)):
        for f in STAGGER_LEADS:
            mid = [s[i] + (g[i] - s[i]) * f if i == k else s[i] for i in range(len(s))]
            out.append([stack(s), stack(mid), stack(g)])
    return out


def estimated_duration(path: Sequence[Sequence[float]], vel_limits: Sequence[float]) -> float:
    """Velocity-limited lower bound on a stacked path's duration, in seconds.

    Per segment, the slowest joint of any arm at its limit — what a synchronous
    time-parameterisation of the stacked vector cannot beat. Used only to rank
    candidates, never as a timing.
    """
    v = np.maximum(np.asarray(vel_limits, float), 1e-6)
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += float(np.max(np.abs(np.asarray(b, float) - np.asarray(a, float)) / v))
    return total


def tag_sync(trajectories: dict, group_id: Optional[str] = None) -> str:
    """Mark the legs of one compound step as one synchronised unit."""
    gid = group_id or f"cmp_{uuid.uuid4().hex[:10]}"
    members = sorted(trajectories)
    for traj in trajectories.values():
        if traj:
            traj[0]["sync_group"] = gid
            traj[0]["sync_members"] = members
    return gid


# ─── the backend ─────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _cell_robot_ids(cell_id: str) -> list[str]:
    try:
        from agents.trajectory.tools.planning_tools import _cell_robots
        return list(_cell_robots(cell_id))
    except Exception:  # noqa: BLE001 — no bystanders known is not a reason to fail
        return []


def _make_compound_checker(ob, si, arms: list, pair_min_m: float):
    """One ``StateValidityChecker`` for the compound state.

    ``arms`` carry ``nq``, ``is_free(q)`` (environment + bystanders) and
    ``spheres(q)`` (world spheres for the pair test). A state is valid only if
    every arm is clear of the environment and every pair of moving arms keeps
    ``pair_min_m`` of surface clearance.
    """

    class _CompoundValidityChecker(ob.StateValidityChecker):
        def __init__(self, space_information):
            super().__init__(space_information)

        def isValid(self, state) -> bool:  # noqa: N802 (OMPL API name)
            qs = []
            for k, arm in enumerate(arms):
                sub = state[k]
                q = np.fromiter((sub[i] for i in range(arm["nq"])), float, arm["nq"])
                if not arm["is_free"](q):
                    return False
                qs.append(q)
            spheres = [arm["spheres"](q) for arm, q in zip(arms, qs)]
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    if pair_clearance(spheres[i], spheres[j]) <= pair_min_m:
                        return False
            return True

    return _CompoundValidityChecker(si)


def _solve_compound(ompl, arms: list, starts: list, goals: list, *,
                    planner_name: str, time_budget_s: float, attempts: int,
                    pair_min_m: float, score: Optional[Callable[[list], float]] = None,
                    target: float = 0.0, cell_id: Optional[str] = None
                    ) -> tuple[Optional[list], Optional[dict]]:
    """A path through the compound space, as stacked vectors, or a failure.

    Up to ``attempts`` independent solves. With a ``score`` (lower is better —
    the worst arm's tool rise), every exact solution is scored and the best one
    kept, stopping early at the first that meets ``target``: a shortcut path
    keeps the homotopy class its solve landed in, so a tall arc is only escaped
    by solving again — the same selection the ``ompl`` backend makes per arm.
    """
    from agents.trajectory.tools import plan_cancellation
    from agents.trajectory.backends.ompl.planner import (
        _make_planner, _search_resolution, _simplify_path,
    )

    ob, og = ompl.ob, ompl.og
    space = ob.CompoundStateSpace()
    for arm in arms:
        sub = ob.RealVectorStateSpace(arm["nq"])
        bounds = ob.RealVectorBounds(arm["nq"])
        for i in range(arm["nq"]):
            bounds.setLow(i, float(arm["jl"][i]))
            bounds.setHigh(i, float(arm["ju"][i]))
        sub.setBounds(bounds)
        space.addSubspace(sub, 1.0)

    si = ob.SpaceInformation(space)
    checker = _make_compound_checker(ob, si, arms, pair_min_m)
    si.setStateValidityChecker(checker)
    step_size, resolution = _search_resolution()
    # The compound extent is the SUM of the subspaces' extents, so the same
    # fraction would check K times more coarsely in radians. Keep it per arm.
    si.setStateValidityCheckingResolution(float(resolution) / max(1, len(arms)))
    si.setup()

    def _alloc(configs):
        st = si.allocState()
        for k, (arm, q) in enumerate(zip(arms, configs)):
            for i in range(arm["nq"]):
                st[k][i] = float(np.clip(q[i], arm["jl"][i], arm["ju"][i]))
        return st

    # An endpoint outside an arm's joint box is refused, not clipped (umarell
    # issue 13): a clipped start is a path that does not begin where the arm is.
    from agents.trajectory.tools.joint_limits import describe, trajectory_limit_violation
    for label, configs in (("start", starts), ("goal", goals)):
        for arm, q in zip(arms, configs):
            v = trajectory_limit_violation([list(q)[:arm["nq"]]], arm["jl"], arm["ju"])
            if v is not None:
                return None, {"reason_code": f"{label}_outside_joint_limits",
                              "reason": (f"{arm['ns']}'s compound {label} is outside its "
                                         f"joint limits: {describe(v, arm['ns'])}"),
                              "joint_limit_violation": v}

    start, goal = _alloc(starts), _alloc(goals)
    for label, st in (("start", start), ("goal", goal)):
        if not checker.isValid(st):
            return None, {
                "reason_code": f"compound_{label}_invalid",
                "reason": (f"the joint {label} of the moving arms is itself in "
                           f"collision or closer than {pair_min_m:.3f} m arm-to-arm — "
                           "no compound path can begin/end there; stage one arm "
                           "elsewhere or pair a different weld (SEQUENTIAL is the last "
                           "resort)"),
            }

    def _clear(path_: list) -> bool:
        """Every segment of a stacked-vector path passes the compound checker."""
        try:
            sts = [_alloc(unstack(v, sizes)) for v in path_]
            return all(si.checkMotion(a, b) for a, b in zip(sts, sts[1:]))
        except Exception:  # noqa: BLE001 — a missing checkMotion: not proven clear
            return False

    sizes = [a["nq"] for a in arms]
    direct = [stack(starts), stack(goals)]
    if _clear(direct):
        return direct, None

    # Deterministic candidates first: the arms' own straight lines, staggered so
    # that one leads (or goes first). Measured on welding_cell_2_small W01/W13:
    # the synchronous straight line collides arm-to-arm at 75 % of the way, the
    # random search escapes only by lifting 0.4-0.5 m, while either arm leading
    # costs no height at all — each arm's straight line rises 0.0 m.
    candidates = [p for p in staggered_paths(starts, goals) if _clear(p)]
    vel = stack([a.get("vel") or [1.0] * a["nq"] for a in arms])

    def _key(path_):
        s = score(path_) if score is not None else 0.0
        # Under the rise target: the FASTEST wins (most overlap between arms).
        # Over it: the flattest wins, and the step is refused later if still tall.
        return (s > target, estimated_duration(path_, vel) if s <= target else s)

    # Experience (tools/path_experience.py): paths the search found before for
    # these arms, reconnected to this start and goal. Each is held to THIS
    # problem's checker like any candidate; one that also meets the rise target
    # makes the search unnecessary.
    arm_ids = [a.get("ns") for a in arms]
    experience: list = []
    if cell_id:
        try:
            from agents.trajectory.tools import path_experience
            experience = [p for p in path_experience.candidates(
                cell_id, arm_ids, stack(starts), stack(goals)) if _clear(p)]
        except Exception:  # noqa: BLE001 — no experience: search as usual
            experience = []
    if experience:
        best = min(experience + candidates, key=_key)
        if score is None or score(best) <= target:
            print(f"[ompl-compound] {', '.join(map(str, arm_ids))}: reused a stored "
                  f"path (checked against this step) instead of searching", flush=True)
            return best, None
        candidates = candidates + experience

    no_path = {"reason_code": "no_path_found",
               "reason": (f"no joint path for {len(arms)} arms within "
                          f"{time_budget_s:.1f} s x {attempts} attempts "
                          f"(planner={planner_name})")}

    def _one_solve():
        """One independent RRT solve: ``(stacked path | None, failure | None)``."""
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        planner = _make_planner(og, si, planner_name)
        if step_size and hasattr(planner, "setRange"):
            planner.setRange(float(step_size))
        planner.setProblemDefinition(pdef)
        planner.setup()
        planner.solve(float(time_budget_s))
        if not pdef.hasExactSolution():
            return None, no_path
        path = pdef.getSolutionPath()
        _simplify_path(og, si, path)
        out_ = []
        for s_ in range(path.getStateCount()):
            st = path.getState(s_)
            out_.append([float(st[k][i]) for k, n in enumerate(sizes) for i in range(n)])
        return (out_ or None), None

    last = None
    found = []
    n_attempts = max(1, int(attempts))
    # The attempts are independent solves of the SAME problem, so with
    # OMPL_PARALLEL_CANDIDATES=1 they run at once in forked children (the ompl
    # backend's mechanism, each child on its own RNG stream): the wall time of a
    # step is one budget instead of up to ``attempts`` of them — 4 x 5 s per
    # failed welding_cell_2_small step. Selection is unchanged (``_key`` over everything
    # found); only the early stop is lost, which costs CPU, not time.
    batch = None
    if n_attempts > 1:
        from agents.trajectory.backends.ompl.planner import _parallel_solves
        plan_cancellation.raise_if_cancelled("compound solve batch")
        batch = _parallel_solves(_one_solve, n_attempts)
    if batch is not None:
        for wps, fail in batch:
            if wps:
                found.append([[float(v) for v in q] for q in wps])
            elif fail is not None:
                last = fail
    else:
        for attempt in range(n_attempts):
            plan_cancellation.raise_if_cancelled(f"compound solve {attempt + 1}")
            out, fail = _one_solve()
            if out is None:
                last = fail or last
                continue
            found.append(out)
            if score is None or score(out) <= target:
                break       # a flat joint path: more solves only cost time
    pool = found + candidates
    if pool:
        chosen = min(pool, key=_key)
        if cell_id and any(chosen is f for f in found):
            try:
                from agents.trajectory.tools import path_experience
                path_experience.record(cell_id, arm_ids, stack(starts), stack(goals), chosen)
            except Exception:  # noqa: BLE001 — not kept: costs reuse, not safety
                pass
        return chosen, None
    return None, last


def plan_trajectories_ompl_compound(
    cell_id: str,
    motion_requests: list[dict],
    approach_offset_m: float = 0.1,
    velocity_scale: Optional[float] = None,
    planner_name: Optional[str] = None,
    time_budget_s: Optional[float] = None,
    max_iterations: Optional[int] = None,  # noqa: ARG001 — contract compatibility
    transit_solver: Optional[Callable] = None,
    backend_name: str = BACKEND_NAME,
    **_ignored,
) -> dict:
    """Plan one step's motion requests jointly. Contract: ``PLANNER_RESULT_CONTRACT``.

    ``transit_solver`` replaces the OMPL compound search and nothing else:
    ``solver(arms, starts, goals, *, pair_min_m, time_budget_s, cell_id) ->
    (stacked path | None, failure | None)``, with the same arms, the same
    endpoints and the same refusal shape as :func:`_solve_compound`. Lift-off,
    descent, the one time base, every emitted-leg check and the commit stay
    this function's, so a backend that only brings a different search (the
    ``roboballet`` policy) is held to exactly this backend's contract.
    ``backend_name`` is what its results and errors are signed with.
    """
    t0 = time.monotonic()
    failures: list[dict] = []
    liftoff_warnings: list[dict] = []
    trajectories: dict = {}

    def _fail(req: dict, entry: dict) -> None:
        if req.get("operation_id") is not None:
            entry.setdefault("operation_id", req["operation_id"])
        failures.append(entry)

    def _result(extra: Optional[dict] = None) -> dict:
        return {"success": len(trajectories) > 0, "trajectories": trajectories,
                "planning_time_s": round(time.monotonic() - t0, 3),
                "failures": failures, "backend": backend_name,
                "planner": planner_name,
                **({"liftoff_warnings": liftoff_warnings} if liftoff_warnings else {}),
                **(extra or {})}

    requests, dups = split_requests(motion_requests)
    for req in dups:
        _fail(req, {"robot_ns": req.get("robot_ns"), "reason_code": "duplicate_robot_request",
                    "error": (f"{backend_name}: more than one motion request for "
                              f"{req.get('robot_ns')} in one call — one compound step "
                              "moves each arm once; pass a sequence as `steps`")})

    from agents.trajectory.backends.ompl import planner as op
    from agents.trajectory.tools import leg_builder as lb
    from agents.trajectory.tools.path_metrics import _path_rise_policy
    ompl = op._import_ompl() if transit_solver is None else None
    if ompl is None and transit_solver is None:
        for req in requests:
            _fail(req, {"robot_ns": req.get("robot_ns"),
                        "error": f"{backend_name}: OMPL unavailable: {op.ompl_import_error()}"})
        return _result()

    from agents.trajectory.tools.robot_kinematics import (
        seam_at_start, liftoff_blocked_failure,
        _cell_sphere_radii, _config_is_free, _config_spheres, _load_clearance_policy,
        _load_collision_config, _load_pinocchio_model, _load_planning_config,
        _load_robot_config, _speed_margin, _target_obstacle_suggestion,
        _trajectory_collision_info, inter_robot_clearance, load_static_obstacles,
        path_deviation_rad, resolve_velocity_scale,
    )
    from agents.trajectory.tools.joint_limits import planning_box
    from agents.trajectory.tools.pinocchio import run_ik as _ik
    from agents.trajectory.tools.time_parameterization import parameterize
    from agents.trajectory.tools.trajectory_cache import (
        clear_last_pose, get_last_config, get_last_pose, set_last_config, set_last_pose,
        set_weld_pose,
    )

    planner_name = planner_name or os.environ.get("OMPL_PLANNER", "RRTConnect")
    static_obstacles = load_static_obstacles(cell_id)
    policy = _load_clearance_policy(cell_id)
    normal_radius = policy["normal"]
    max_vs = _load_planning_config(cell_id)["max_velocity_scaling"]
    velocity_scale = resolve_velocity_scale(cell_id, velocity_scale)
    transit_margin = _speed_margin(cell_id, velocity_scale)
    warning = float(_load_collision_config(cell_id).get("warning_distance", 0.20))
    pair_min = warning + _env_float("OMPL_COMPOUND_INTER_ARM_EXTRA_M", 0.01)

    def _robot(ns: str) -> dict:
        model, _ = _load_pinocchio_model(cell_id, ns)
        rcfg = _load_robot_config(cell_id, ns)
        tool_cfg = rcfg.get("tool_collision")
        return {"ns": ns, "model": model, "rcfg": rcfg, "nq": model.nq,
                "base": np.array(rcfg["base_position"]),
                "yaw": float(rcfg.get("base_orientation_yaw", 0.0)),
                "tool_cfg": tool_cfg,
                "radii": _cell_sphere_radii(cell_id, model, tool_cfg),
                **dict(zip(("jl", "ju"), planning_box(rcfg, model))),
                "home": np.array(lb.home_config(cell_id, ns, op.HOME_JOINTS_DEFAULT), float)}

    def _spheres_fn(r: dict) -> Callable:
        data = r["model"].createData()
        return lambda q: _config_spheres(r["model"], data, q, r["base"], r["yaw"],
                                         r["tool_cfg"])

    # ── bystanders: every arm of the cell not moving in this step, where it is ──
    moving_ns = {r.get("robot_ns") for r in requests}
    bystander_spheres: list = []
    bystanders_by_arm: dict = {}     # the same spheres, per arm: who blocks a stroke
    bystander_traj: dict = {}
    for ns in _cell_robot_ids(cell_id):
        if ns in moving_ns:
            continue
        try:
            b = _robot(ns)
            q = np.array((get_last_config(cell_id, ns) or b["home"].tolist())[:b["nq"]])
        except Exception:  # noqa: BLE001
            continue
        bystander_traj[ns] = [{"positions": q.tolist(), "time_s": 0.0}]
        # As moving obstacles for _config_is_free (dist < r_obs + r_link is a hit),
        # sized so that test IS the pair criterion: centre distance > 2r + pair_min.
        for p in _spheres_fn(b)(q):
            sph = (np.asarray(p, float), PAIR_RADIUS_M + pair_min)
            bystander_spheres.append(sph)
            bystanders_by_arm.setdefault(ns, []).append(sph)

    # ── per-arm preparation: IK, lift-off, descent (as the ompl backend does) ──
    arms: list[dict] = []
    blocked_liftoffs: list = []
    n_before_prep = len(failures)      # duplicates / no-OMPL are not prep failures
    for req in requests:
        ns = req["robot_ns"]
        try:
            r = _robot(ns)
        except Exception as e:  # noqa: BLE001
            _fail(req, {"robot_ns": ns, "error": str(e)})
            continue
        model, base, yaw = r["model"], r["base"], r["yaw"]
        tgt = req["target"]
        x, y, z = float(tgt["x"]), float(tgt["y"]), float(tgt["z"])
        av = np.array(req.get("approach_vector", [0, 0, -1]), float)
        av /= (np.linalg.norm(av) + 1e-9)
        try:
            offset = float(req.get("approach_offset_m", approach_offset_m))
        except (TypeError, ValueError):
            offset = approach_offset_m
        offset = offset if offset > 0.0 else approach_offset_m
        try:
            leave = float(req.get("retreat_offset_m") or offset)
        except (TypeError, ValueError):
            leave = offset
        leave = leave if leave > 0.0 else offset
        app_xyz = (x - av[0] * offset, y - av[1] * offset, z - av[2] * offset)

        chained = req.get("current_joints") or get_last_config(cell_id, ns) or r["home"].tolist()
        ori = req.get("orientation") or {}
        ik_ori = {}
        if req.get("constrain_orientation") and ori:
            ik_ori = {"qx": float(ori.get("x", 0.0)), "qy": float(ori.get("y", 0.707)),
                      "qz": float(ori.get("z", 0.0)), "qw": float(ori.get("w", 0.707)),
                      "constrain_orientation": True}
        ik_t = _ik.invoke({"cell_id": cell_id, "robot_ns": ns, "x": x, "y": y, "z": z,
                           "warm_start_joints": chained, **ik_ori})
        if not ik_t.get("success"):
            _fail(req, {"robot_ns": ns, "reason_code": "ik_failed", "error": f"IK failed for target: {ik_t.get('reason')}"})
            continue
        ik_a = _ik.invoke({"cell_id": cell_id, "robot_ns": ns, "x": app_xyz[0],
                           "y": app_xyz[1], "z": app_xyz[2],
                           "warm_start_joints": chained, **ik_ori})
        goal_q = np.array(ik_t["joint_positions"][:r["nq"]])
        via = (ik_a["joint_positions"] if ik_a.get("success") else
               lb.approach_ik(_ik, cell_id, ns, app_xyz, ik_ori, [goal_q.tolist()]))
        via_q = np.array(via[:r["nq"]]) if via else None
        if via_q is None and ik_ori and req.get("approach"):
            # No straight descent without it, and a joint-space entry fails the
            # working-stroke check at assembly: refuse now, with a stroke hint.
            _fail(req, lb.approach_waypoint_failure(ns, offset))
            continue

        use_approach = bool(req.get("approach", False))
        try:
            app_vs = (float(req["approach_velocity_scale"])
                      if req.get("approach_velocity_scale") is not None
                      else velocity_scale * policy["approach_velocity_fraction"])
        except (TypeError, ValueError):
            app_vs = velocity_scale * policy["approach_velocity_fraction"]
        app_vs = max(min(app_vs, max_vs), 0.01)
        app_margin = _speed_margin(cell_id, app_vs if use_approach else velocity_scale)

        start_q = np.array(chained[:r["nq"]])
        lift = [start_q]
        # The seam this arm is ON at its start — pinned or chained (see
        # robot_kinematics.seam_at_start): a pinned current_joints on a weld lifts off.
        prev = seam_at_start(cell_id, ns, model, start_q, base, yaw,
                             pinned=bool(req.get("current_joints")))
        if prev and prev.get("orientation") and float(prev.get("rise_m") or 0.0) < leave - 1e-3:
            pav = np.asarray(prev["approach_vector"], float)
            pav /= (np.linalg.norm(pav) + 1e-9)
            ptgt = np.asarray(prev["target"], float)
            pfrom = np.asarray(prev.get("from_point") or ptgt, float)
            po = prev["orientation"]
            wp = lb.cartesian_retreat(
                model, _ik, cell_id, ns, pfrom, ptgt - pav * leave,
                {"qx": float(po.get("x", 0.0)), "qy": float(po.get("y", 0.707)),
                 "qz": float(po.get("z", 0.0)), "qw": float(po.get("w", 0.707)),
                 "constrain_orientation": True},
                start_q, base, yaw, bystander_spheres, static_obstacles,
                normal_radius, r["tool_cfg"], sphere_radii=r["radii"],
                extra_margin=app_margin)
            if wp:
                lift.extend(wp)
            else:
                # Blocked: without the on-axis lift-off this arm would drag its tool
                # out of the seam (shared rule, as in ompl). The step is one motion
                # (ADR-0001), so the whole step is refused below, not just this arm.
                refusal = liftoff_blocked_failure(ns, prev)
                if refusal is not None:
                    blocked_liftoffs.append((req, refusal))
                    continue
                liftoff_warnings.append({
                    "robot_ns": ns, "operation_id": req.get("operation_id"),
                    "from_operation_id": prev.get("operation_id"),
                    "warning": ("the on-axis lift-off from the previous weld is blocked "
                                "or unreachable; this leg leaves the seam with a "
                                "joint-space transit")})

        # Transit ends at the standoff when there is a constrained descent to do,
        # else straight at the goal (a plain move), as in the ompl backend.
        descent = None
        if via_q is not None and ik_ori:
            why: dict = {}
            descent = lb.cartesian_descent(
                model, _ik, cell_id, ns, app_xyz, (x, y, z), ik_ori, via_q, base, yaw,
                bystander_spheres, static_obstacles, normal_radius, r["tool_cfg"],
                sphere_radii=r["radii"], extra_margin=app_margin, why=why)
            if descent is None and why.get("cause") == "collision" \
                    and (why.get("collision") or {}).get("type") != "self":
                # Is it a PARKED arm? Then say which: the cure is to move that arm,
                # not to give this weld away (the agent used to reallocate it).
                blockers = lb.blocking_arms(
                    lambda sph: lb.cartesian_descent(
                        model, _ik, cell_id, ns, app_xyz, (x, y, z), ik_ori, via_q, base,
                        yaw, sph, [], normal_radius, r["tool_cfg"], sphere_radii=r["radii"],
                        extra_margin=app_margin) is not None,
                    bystanders_by_arm)
                if blockers:
                    entry = lb.blocked_by_parked_failure(ns, blockers,
                                                         "straight descent into the weld")
                    entry["error"] = f"{backend_name} ({planner_name}): {entry['error']}"
                    _fail(req, entry)
                    continue
            if descent is None:
                # Named by what blocked it, as in ompl (self / IK / obstacle), so the
                # recovery — and the stroke hint — fit the cause.
                named = lb.descent_failure(why)
                entry = {"robot_ns": ns, "reason_code": named["reason_code"],
                         "error": f"{backend_name} ({planner_name}): {named['reason']}"}
                if named.get("collision"):
                    entry["collision"] = named["collision"]
                diag = _target_obstacle_suggestion([x, y, z], av, static_obstacles,
                                                   policy["tool"] + 0.05)
                if diag is not None:
                    entry.update(target_diagnosis=diag, suggestion=diag["suggestion"])
                _fail(req, entry)
                continue
        transit_goal = via_q if descent is not None else goal_q

        col = model.createData()
        arms.append({
            **r, "req": req, "goal_q": goal_q, "lift": lift,
            "transit_start": np.asarray(lift[-1], float), "transit_goal": transit_goal,
            "descent": descent, "use_approach": use_approach, "app_vs": app_vs,
            "stroke_m": max(offset, leave),
            "app_margin": app_margin, "target": [x, y, z], "av": av, "ori": ori,
            "ik_ori": ik_ori, "spheres": _spheres_fn(r),
            "vel": r["rcfg"].get("joint_velocity_limits"),
            "is_free": (lambda q, r=r, col=col: _config_is_free(
                r["model"], col, q, r["base"], r["yaw"], bystander_spheres,
                static_obstacles, sphere_radius=normal_radius, tool_cfg=r["tool_cfg"],
                sphere_radii=r["radii"], extra_margin=transit_margin)),
        })

    for req, refusal in blocked_liftoffs:
        _fail(req, {**refusal, "error": f"{backend_name}: {refusal['error']}"})
    prep_failed = {f.get("robot_ns") for f in failures[n_before_prep:] if f.get("robot_ns")}
    if prep_failed and arms:
        # An arm of the step could not be prepared (IK, the on-axis descent, the
        # lift-off out of its seam): no arm of the step ships. Each arm's plan
        # would assume the others move, and the failed arm would not even be a
        # bystander in it (ADR-0001: a step is refused as a whole). The agent
        # re-pairs or sequences the step.
        code = ("liftoff_blocked" if blocked_liftoffs
                else next((f.get("reason_code") for f in failures[n_before_prep:]
                           if f.get("reason_code")), "step_refused"))
        culprit = sorted(prep_failed)[0]
        for a in arms:
            if a["ns"] not in prep_failed:
                _fail(a["req"], {"robot_ns": a["ns"], "reason_code": "step_refused",
                                 "caused_by": {"robot_ns": culprit, "reason_code": code},
                                 "error": (f"{backend_name}: the step is refused as a whole: "
                                           f"{', '.join(sorted(prep_failed))} could not be "
                                           "prepared (see its failure)")})
        return _result({"compound": {"arms": sorted({a["ns"] for a in arms} | prep_failed),
                                     "refused": code}})

    if not arms:
        return _result()

    def _refuse(reason_code: str, error: str, culprit: Optional[str] = None,
                **detail) -> dict:
        # The step is ONE motion: no arm ships if the joint result does not hold.
        # The arm the fault is about keeps its reason code; the others are told
        # their step was refused because of it (``step_refused`` + ``caused_by``),
        # so a recovery never chases a weld that is fine.
        for arm in arms:
            if culprit is None or arm["ns"] == culprit:
                _fail(arm["req"], {"robot_ns": arm["ns"], "reason_code": reason_code,
                                   "error": f"{backend_name} ({planner_name}): {error}",
                                   **detail})
            else:
                _fail(arm["req"], {"robot_ns": arm["ns"], "reason_code": "step_refused",
                                   "caused_by": {"robot_ns": culprit,
                                                 "reason_code": reason_code},
                                   "error": (f"{backend_name}: the step is refused as a "
                                             f"whole because of {culprit} ({reason_code})")})
        trajectories.clear()
        return _result({"compound": {"arms": [a["ns"] for a in arms], "refused": reason_code}})

    budget = time_budget_s or _env_float("OMPL_COMPOUND_TIME_BUDGET_S", 5.0 * len(arms))
    rise_data = [a["model"].createData() for a in arms]

    def _worst_rise(path_: list) -> float:
        per_arm = list(zip(*[unstack(v, [a["nq"] for a in arms]) for v in path_]))
        return max(lb.path_tool_rise([np.asarray(q) for q in wps], a["model"], d,
                                      a["base"], a["yaw"])
                   for a, d, wps in zip(arms, rise_data, per_arm))

    if transit_solver is not None:
        path, fail = transit_solver(
            arms, [a["transit_start"] for a in arms], [a["transit_goal"] for a in arms],
            pair_min_m=pair_min, time_budget_s=float(budget), cell_id=cell_id)
    else:
        path, fail = _solve_compound(
            ompl, arms, [a["transit_start"] for a in arms], [a["transit_goal"] for a in arms],
            planner_name=planner_name, time_budget_s=float(budget),
            attempts=int(_env_float("OMPL_COMPOUND_ATTEMPTS",
                                    _env_float("OMPL_PATH_CANDIDATES", 4))),
            pair_min_m=pair_min, score=_worst_rise, target=_path_rise_policy(),
            cell_id=cell_id)
    if path is None:
        fail = fail or {}
        return _refuse(fail.get("reason_code", "no_path_found"), fail.get("reason", "no path"))

    # ── one time base ─────────────────────────────────────────────────────────
    vel = [a["rcfg"].get("joint_velocity_limits") for a in arms]
    acc = [lb.accel_limits(cell_id, a["rcfg"]) for a in arms]
    # The stacked path may leave the validated one by the TIGHTEST arm's bound.
    dev = [path_deviation_rad(a["rcfg"]) for a in arms]
    transit = parameterize(
        path,
        vel_limits=stack(vel) if all(v is not None for v in vel) else None,
        accel_limits=stack(acc) if all(v is not None for v in acc) else None,
        velocity_scale=velocity_scale,
        max_deviation_rad=min(dev) if all(d is not None for d in dev) else None,
        # The stacked joint box: every arm's own limits, side by side.
        position_limits=(stack([a["jl"] for a in arms]), stack([a["ju"] for a in arms])))
    retracts, descents, phases = {}, {}, {}
    for a in arms:
        lim = dict(vel_limits=a["rcfg"].get("joint_velocity_limits"),
                   accel_limits=lb.accel_limits(cell_id, a["rcfg"]),
                   max_deviation_rad=path_deviation_rad(a["rcfg"]),
                   position_limits=(a["jl"], a["ju"]))
        # The lift-off is the on-axis stroke out of the seam: timed exactly.
        retracts[a["ns"]] = (parameterize(a["lift"], velocity_scale=velocity_scale,
                                          **{**lim, "max_deviation_rad": None})
                             if len(a["lift"]) > 1 else [])
        if a["descent"]:
            vs = a["app_vs"] if a["use_approach"] else velocity_scale
            descents[a["ns"]] = parameterize([a["transit_goal"]] + list(a["descent"]),
                                             velocity_scale=vs, **lim)
            phases[a["ns"]] = "approach" if a["use_approach"] else "transit"
    joint = compose_timelines(retracts, transit, {a["ns"]: a["nq"] for a in arms},
                              descents, phases)

    # ── validate what is emitted ──────────────────────────────────────────────
    for a in arms:
        breach = _trajectory_collision_info(
            a["model"], a["model"].createData(), joint[a["ns"]], a["base"], a["yaw"],
            bystander_spheres, static_obstacles, sphere_radius=normal_radius,
            sphere_radii=a["radii"], transit_margin=transit_margin,
            approach_margin=a["app_margin"])
        if breach is not None:
            return _refuse("trajectory_in_collision", culprit=a["ns"], error=
                           f"{a['ns']}'s emitted trajectory breaches clearance at sample "
                           f"{breach.get('sample_index')} ({breach.get('phase')} phase) "
                           f"against '{breach.get('object_id')}'", collision=breach)
        rise, limit = lb.trajectory_tool_rise(a["model"], joint[a["ns"]], a["base"], a["yaw"]), \
            lb.emitted_rise_limit()
        if limit > 0.0 and rise > limit:
            return _refuse("excessive_tool_rise", culprit=a["ns"], error=
                           f"{a['ns']}'s joint path lifts the tool {rise:.3f} m above both "
                           f"endpoints, over the {limit:.2f} m limit",
                           tool_rise_m=round(rise, 4), tool_rise_limit_m=limit)
    try:
        verdict = inter_robot_clearance(cell_id, {**joint, **bystander_traj})
    except Exception as e:  # noqa: BLE001 — a gate that cannot run has not said yes
        verdict = {"collision_free": False, "error": str(e)}
    if not verdict.get("collision_free"):
        return _refuse("inter_arm_clearance",
                       f"the joint timeline reaches {verdict.get('min_clearance_m')} m "
                       f"arm-to-arm at t={verdict.get('collision_time_s')} s, inside the "
                       f"{verdict.get('warning_distance_m')} m band (during a lift-off or "
                       "descent, which are planned per arm) — stagger it (delay_fix), pair a "
                       "different weld or use find_interlock_set; SEQUENTIAL is the last "
                       "resort",
                       collision=verdict)

    # ── commit ────────────────────────────────────────────────────────────────
    gid = tag_sync(joint)
    for a in arms:
        traj = joint[a["ns"]]
        trajectories[a["ns"]] = traj
        set_last_config(cell_id, a["ns"], lb.trajectory_end(traj, a["goal_q"]))
        if a["ik_ori"]:
            set_last_pose(cell_id, a["ns"], a["target"], a["av"].tolist(), a["ori"])
            set_weld_pose(cell_id, a["ns"], a["req"].get("operation_id"), a["target"],
                          a["av"].tolist(), a["ori"], stroke_m=a["stroke_m"])
        else:
            clear_last_pose(cell_id, a["ns"])
    return _result({"compound": {
        "arms": [a["ns"] for a in arms], "dimension": sum(a["nq"] for a in arms),
        "sync_group": gid, "transit_s": float(transit[-1]["time_s"]) if transit else 0.0,
        "min_clearance_m": verdict.get("min_clearance_m"),
        "pair_min_m": round(pair_min, 4), "bystanders": sorted(bystander_traj)}})
