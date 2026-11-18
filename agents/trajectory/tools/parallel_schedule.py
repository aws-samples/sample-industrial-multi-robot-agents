"""
Compose PARALLEL steps for any number of arms — every arm working at once
wherever the geometry allows it.

A step is a set of motions that start together: each arm either moves from where
it stands to one of its pending operations, or stays parked where it is. The
composer builds the steps for ALL arms at once: at every step it picks, for each
arm, one operation from the front of that arm's own queue (``lookahead``) — or
nothing — so that the whole set is clear, and it prefers the set with the MOST
arms working, then the one that keeps each arm closest to its own optimised tour.

Why here and not later. Pairing welds by position is the wrong objective (weld-
point distance correlates at -0.607 with the arms' real clearance, see
``task_allocation._least_movement_schedule``), and pairing them blind leaves the
conflicts to be discovered one step at a time by the planner — which is how a
program ends up serial. The composer asks the right question instead: can these
ARMS be in these places, moving there together? It answers it with the arms' own
kinematics — an IK pose per (arm, operation) and a linear joint-space sweep from
each arm's previous pose — checked against the same warning band the runtime
monitor enforces.

That sweep is a PROXY for the planned path, not the path. The planner still plans
every step with all its arms together, ``validate_trajectory_set`` still checks
the real trajectories (and offers a start-delay fix), and the behaviour-tree gate
still has the final word. The composer's job is to hand them steps that are
parallel by construction, so the resolution ladder rarely has to run.

The core (:func:`compose_parallel_steps`) is pure Python with the geometry
injected as a pairwise predicate, so it is testable without Pinocchio.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Callable, Optional

HOME = "__home__"

#: Max arms for an exhaustive search of a step; above this, arms are added
#: greedily (busiest first), which keeps the cost linear in the arm count.
EXHAUSTIVE_MAX_ARMS = 4

#: Orientations tried per (arm, operation) for its standing pose: the specified one,
#: then the nearest spins/tilts the tool allows.
MAX_POSE_TRIES = 25


def compose_parallel_steps(
    sequences: dict[str, list[dict]],
    pair_clear: Callable[[str, str, str, str, str, str], bool],
    *,
    lookahead: Optional[int] = None,
    beam_width: int = 4,
    options_per_step: int = 3,
    allow_park: bool = True,
) -> list[dict]:
    """Interleave per-arm queues into steps where as many arms as possible work.

    Args:
        sequences:   ``{robot_id: [op, ...]}`` — each arm's ordered queue.
        pair_clear:  ``pair_clear(rid_a, from_a, to_a, rid_b, from_b, to_b)`` — may
                     arm A go from ``from_a`` to ``to_a`` while arm B goes from
                     ``from_b`` to ``to_b``, starting together? ``from == to``
                     means the arm stands still. Keys are operation ids or
                     :data:`HOME`.
        lookahead:   how deep into its own queue an arm may reach for an operation
                     that fits a step (``None`` = its whole queue; 1 = strict tour
                     order). Reaching deeper costs score, so the tour is kept
                     whenever it does not cost parallelism.
        beam_width:  programs kept per step of the search. The search is over the
                     WHOLE program, not step by step: sending an arm to the middle
                     of the part now can block every other arm later. 1 = greedy.
        options_per_step: step choices expanded from each kept program.
        allow_park:  an arm that is in the way may move back to HOME within a step
                     (a park) so the others can work. Parks cost score.

    Returns steps in the ``run_mrta_pipeline`` shape — ``{robot_id: op, ...}``
    with only the arms that WORK in the step — plus ``parallel_arms`` (how many
    work), ``waiting`` (arms with pending work held back by a conflict) and
    ``park`` (arms moved to HOME in the step to clear the way).

    TOTALITY: every operation appears in exactly one step. A step always makes
    progress: if no set of arms can move clear, the busiest arm's next operation
    runs alone and the step is marked ``conflict: true`` — the planner and the
    behaviour-tree gate then decide how to clear it.
    """
    rids = sorted(sequences)
    ops = {rid: list(seq) for rid, seq in sequences.items()}

    def op_id(op):
        return str(op.get("id"))

    def clear(at: dict, move: dict) -> bool:
        """``move``: {rid: target key}; arms not in it stand where they are."""
        for i, a in enumerate(rids):
            for b in rids[i + 1:]:
                fa, fb = at[a], at[b]
                ta, tb = move.get(a, fa), move.get(b, fb)
                if fa == ta and fb == tb:
                    continue
                if not pair_clear(a, fa, ta, b, fb, tb):
                    return False
        return True

    def options(at: dict, pending: dict) -> list:
        """The best few ways to compose the NEXT step from this state."""
        busy = sorted((r for r in rids if pending[r]), key=lambda r: (-len(pending[r]), r))
        depth = {r: pending[r] if lookahead is None else pending[r][:max(1, lookahead)]
                 for r in busy}
        choices = {}
        for r in rids:
            c = [None] + list(depth.get(r, []))
            if allow_park and at[r] != HOME:
                c.append(HOME)
            choices[r] = c
        found = []
        if len(rids) <= EXHAUSTIVE_MAX_ARMS:
            combos = (dict(zip(rids, combo)) for combo in product(*[choices[r] for r in rids]))
        else:
            greedy: dict = {}
            for r in busy:
                for op in depth[r]:
                    if clear(at, {**{k: op_id(v) for k, v in greedy.items()}, r: op_id(op)}):
                        greedy[r] = op
                        break
            combos = iter([greedy])
        for assign in combos:
            work = {r: o for r, o in assign.items() if isinstance(o, dict)}
            if not work:
                continue
            parks = [r for r, o in assign.items() if o == HOME]
            move = {r: op_id(o) for r, o in work.items()}
            move.update({r: HOME for r in parks})
            if not clear(at, move):
                continue
            reach = sum(depth[r].index(o) for r, o in work.items())
            score = (-len(work), len(parks), reach)
            found.append((score, work, parks))
        found.sort(key=lambda f: f[0])
        if not found and allow_park:
            # Nobody can work where the arms now stand: clear the way first — a
            # step in which only arms that are in the way move back to HOME.
            away = [r for r in rids if at[r] != HOME]
            for k in range(1, len(away) + 1):
                for subset in combinations(away, k):
                    if clear(at, {r: HOME for r in subset}):
                        found.append(((0, len(subset), 0), {}, list(subset)))
                if found:
                    break
        if not found:
            lead = busy[0]
            return [((0, 0, 0), {lead: pending[lead][0]}, [], True)]
        return [(sc, w, p, False) for sc, w, p in found[:options_per_step]]

    # A program: (cost, at, pending, steps). Cost ranks partial programs of the
    # same length: fewer arm-slots left idle, fewer parks, fewer conflicts.
    start_pending = {r: list(ops[r]) for r in rids}
    beam = [((0, 0, 0, 0), {r: HOME for r in rids}, start_pending, [])]
    while any(any(p.values()) for _, _, p, _ in beam):
        nxt = []
        for cost, at, pending, steps in beam:
            if not any(pending.values()):
                nxt.append((cost, at, pending, steps))
                continue
            for sc, work, parks, conflict in options(at, pending):
                at2 = dict(at)
                pend2 = {r: list(v) for r, v in pending.items()}
                step: dict = {}
                for r, op in work.items():
                    step[r] = op
                    pend2[r].remove(op)
                    at2[r] = op_id(op)
                for r in parks:
                    at2[r] = HOME
                had_work = [r for r in rids if pending[r]]
                step["parallel_arms"] = len(work)
                waiting = sorted(r for r in had_work if r not in work)
                if waiting:
                    step["waiting"] = waiting
                if parks:
                    step["park"] = sorted(parks)
                if conflict:
                    step["conflict"] = True
                idle = len(had_work) - len(work)
                c2 = (cost[0] + idle, cost[1] + len(parks), cost[2] + int(conflict),
                      cost[3] + sc[2])
                nxt.append((c2, at2, pend2, steps + [step]))
        nxt.sort(key=lambda b: b[0])
        beam = nxt[:max(1, beam_width)]
    best = min(beam, key=lambda b: (len(b[3]), b[0]))
    return best[3]


# ─── which way each arm walks its tour ────────────────────────────────────────
#
# An arm's tour is an OPEN path, and an open path costs the same walked either
# way. ``_tsp_2opt(start_near=base)`` picks the end nearest the arm's base — one
# fixed rule for every cell — and that rule is what serialises a cell whose arms
# share a boundary: two arms on the same side of a part each start at their outer
# end and both FINISH at the shared middle, at the same time, so every late step
# conflicts and an arm parked there blocks its neighbour. Measured on welding_cell_2_small
# (four arms, two per side of a car body): the composed program ran 4 steps with
# all four arms, then decayed to one arm per step over the B-pillar and roof bow.
#
# Nothing here is cell-specific: the direction of every arm is CHOSEN by composing
# the program both ways with the arms' own geometry (``pair_clear``) and keeping
# the most parallel one. Travel does not change; only the order does.

def schedule_score(steps: list[dict]) -> tuple:
    """Lower is better: fewer steps, fewer conflict steps, more arm-steps working,
    fewer parks."""
    return (len(steps),
            sum(1 for s in steps if s.get("conflict")),
            -sum(int(s.get("parallel_arms", 0)) for s in steps),
            sum(len(s.get("park") or ()) for s in steps))


def orient_for_parallelism(
    sequences: dict[str, list[dict]],
    pair_clear: Callable[[str, str, str, str, str, str], bool],
    *,
    compose: Callable = None,
    search_compose: Callable = None,
    exhaustive_max: int = EXHAUSTIVE_MAX_ARMS,
    max_rounds: int = 3,
) -> "tuple[dict, list, dict]":
    """``(sequences, steps, info)``: each arm's tour walked in the direction that
    makes the composed program most parallel (:func:`schedule_score`).

    Every combination of directions is composed for up to ``exhaustive_max`` arms
    (2^n programs); above that, one arm is flipped at a time while it helps
    (``max_rounds`` passes), so the cost stays linear in the arm count. Ties keep
    the given directions — the near-first walk is changed only when it costs
    parallelism. ``info``: ``reversed`` (arms walked the other way),
    ``baseline_steps`` / ``steps`` and ``programs_composed``.

    The candidates are ranked with ``search_compose`` (default: the greedy
    composer, one option per step — measured on welding_cell_2_small the full beam made 16
    programs take ~4 min); the winner and the given directions are then composed
    with ``compose`` and the better of the two is returned, so the search can never
    make the program worse than not searching.
    """
    compose = compose or compose_parallel_steps
    if search_compose is None:
        search_compose = (compose if compose is not compose_parallel_steps else
                          (lambda seqs, pc: compose_parallel_steps(
                              seqs, pc, beam_width=1, options_per_step=1)))
    flippable = sorted(r for r, s in sequences.items() if len(s) > 1)
    programs: dict = {}

    def walked(flips) -> dict:
        return {r: (list(reversed(s)) if r in flips else list(s))
                for r, s in sequences.items()}

    def evaluate(flips) -> tuple:
        key = frozenset(flips)
        if key not in programs:
            programs[key] = search_compose(walked(key), pair_clear)
        return schedule_score(programs[key])

    def rank(flips) -> tuple:
        return (evaluate(flips), len(flips), sorted(flips))

    best = frozenset()
    if len(flippable) <= exhaustive_max:
        for k in range(len(flippable) + 1):
            for subset in combinations(flippable, k):
                if rank(frozenset(subset)) < rank(best):
                    best = frozenset(subset)
    else:
        for _ in range(max(1, max_rounds)):
            improved = False
            for r in flippable:
                cand = best ^ {r}
                if rank(cand) < rank(best):
                    best, improved = cand, True
            if not improved:
                break
    searched = len(programs)
    if search_compose is not compose:
        final = {k: compose(walked(k), pair_clear) for k in {frozenset(), best}}
        if schedule_score(final[frozenset()]) <= schedule_score(final[best]):
            best = frozenset()
        programs = final
    info = {"reversed": sorted(best),
            "baseline_steps": len(programs[frozenset()]),
            "steps": len(programs[best]),
            "programs_composed": searched}
    return walked(best), programs[best], info


def standing_conflicts(sequences: dict[str, list[dict]],
                       pair_clear: Callable[[str, str, str, str, str, str], bool]
                       ) -> list[dict]:
    """Every pair of operations two arms can NOT work at the same time — not even
    standing still at them: the arms are inside each other's warning band there.

    No ordering makes such a pair parallel; it can only be sequenced apart, or the
    weld given to one arm. Reported so the planner (and the operator) is told the
    arms are too close there instead of discovering it one rejected step at a time.
    """
    out = []
    has_pose = getattr(pair_clear, "has_pose", None)
    rids = sorted(r for r, s in sequences.items() if s)
    for a, b in combinations(rids, 2):
        for x in sequences[a]:
            xa = str(x.get("id"))
            if has_pose is not None and not has_pose(a, xa):
                continue                  # reported by unposable_operations
            for y in sequences[b]:
                yb = str(y.get("id"))
                if has_pose is not None and not has_pose(b, yb):
                    continue
                if not pair_clear(a, xa, xa, b, yb, yb):
                    out.append({"robots": [a, b], "operations": [xa, yb]})
    return out


def unposable_operations(sequences: dict[str, list[dict]],
                         pair_clear: Callable) -> list[dict]:
    """``[{robot, operation}]``: operations the arm has no collision-free standing
    pose at (IK at every allowed orientation collides — the fixtures or the arm
    itself). Scheduled alone; the cause is the target or the obstacle model, not
    the other arms."""
    has_pose = getattr(pair_clear, "has_pose", None)
    if has_pose is None:
        return []
    return [{"robot": r, "operation": str(o.get("id"))}
            for r, s in sorted(sequences.items()) for o in s
            if not has_pose(r, str(o.get("id")))]


def contention_advice(conflicts: list[dict]) -> list[str]:
    """One line per pair of arms that are too close at some of their welds."""
    by_pair: dict = {}
    for c in conflicts:
        by_pair.setdefault(tuple(c["robots"]), []).append(c["operations"])
    lines = []
    for (a, b), pairs in sorted(by_pair.items()):
        ops_a = sorted({p[0] for p in pairs})
        ops_b = sorted({p[1] for p in pairs})
        shown = ", ".join(f"{p[0]}|{p[1]}" for p in pairs[:6]) + (" ..." if len(pairs) > 6 else "")
        lines.append(
            f"{a} and {b} are too close to work at the same time at {len(pairs)} weld "
            f"pair(s) ({shown}): the schedule keeps them in different steps. To get "
            f"the parallelism back, move the contested welds to one arm "
            f"({a}: {', '.join(ops_a)}; {b}: {', '.join(ops_b)}) with "
            f"reallocate_operation, or accept the stagger — do not drop either arm.")
    return lines


# ─── the geometric model ──────────────────────────────────────────────────────

def geometric_pair_model(cell_id: str, robots: dict,
                         operations: list[dict]) -> Optional[Callable]:
    """A ``pair_clear`` predicate from the arms' own kinematics, or ``None``.

    Each (arm, operation) gets its IK pose at the operation's working
    orientation; HOME comes from the cell. A move is a linear joint sweep timed
    at the cell's transit speed; standing is one pose. The pair check is
    ``inter_robot_clearance`` against the cell's warning band — the same
    function the planner and the runtime monitor use. Results are memoised.

    ``None`` when the geometry stack is missing (host), so the caller keeps the
    least-movement schedule. An operation an arm cannot stand at (IK fails) is
    treated as clear of nothing: it will be scheduled alone.
    """
    try:
        import numpy as np
        import pinocchio  # noqa: F401
        from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose
        from agents.trajectory.tools import weld_orientation as _wo
        from agents.trajectory.tools.pinocchio import run_ik
        from agents.trajectory.tools.robot_kinematics import (
            _load_collision_config, _load_planning_config, inter_robot_clearance,
        )
        _ik = getattr(run_ik, "func", run_ik)
        band = _load_collision_config(cell_id)["warning_distance"]
        vscale = _load_planning_config(cell_id)["default_velocity_scale"]
    except Exception:
        return None

    ops = {str(o.get("id")): o for o in operations}
    poses: dict = {}

    def pose(rid: str, key: str):
        if (rid, key) in poses:
            return poses[(rid, key)]
        q = None
        if key == HOME:
            q = list(home_pose(cell_id, rid, HOME_JOINTS_FALLBACK))
        elif key in ops:
            o = ops[key]
            ori = o.get("weld_orientation") or o.get("orientation") or {}
            x, y, z = [float(v) for v in o.get("position", [0, 0, 0])]
            # The pose the planner would stand at: the specified orientation, else
            # the freedom the tool allows (spin about its axis, then a small tilt —
            # weld_orientation.candidates, the planner's own search order). With one
            # fixed spin a weld whose IK folds the arm into itself counted as "clear
            # of nothing" here, and its arm was scheduled alone for no geometric
            # reason (welding_cell_2_small: robot2 at W23/W25, link_3 vs link_6).
            tries = [ori] if ori else [{}]
            if ori:
                try:
                    tries += [c[3] for c in _wo.candidates(
                        ori, _wo.policy((robots.get(rid) or {})))[1:MAX_POSE_TRIES]]
                except Exception:
                    pass
            for cand in tries:
                try:
                    r = _ik(cell_id=cell_id, robot_ns=rid, x=x, y=y, z=z,
                            qx=float(cand.get("x", 0.0)), qy=float(cand.get("y", 0.707)),
                            qz=float(cand.get("z", 0.0)), qw=float(cand.get("w", 0.707)),
                            constrain_orientation=bool(cand))
                except Exception:
                    continue
                if r.get("success") and r.get("goal_collision_free", True):
                    q = list(r["joint_positions"])
                    break
        poses[(rid, key)] = q
        return q

    vel = {}

    def sweep(rid: str, a: str, b: str):
        qa, qb = pose(rid, a), pose(rid, b)
        if qa is None or qb is None:
            return None
        if a == b:
            return [{"positions": qa, "time_s": 0.0}]
        if rid not in vel:
            cfg = robots.get(rid) or {}
            vel[rid] = np.asarray(cfg.get("joint_velocity_limits") or [1.0] * 6, float)
        dq = np.abs(np.asarray(qb) - np.asarray(qa))
        n = min(len(dq), len(vel[rid]))
        dur = max(0.5, float(np.max(dq[:n] / (vel[rid][:n] * vscale))))
        return [{"positions": qa, "time_s": 0.0}, {"positions": qb, "time_s": dur}]

    memo: dict = {}

    def pair_clear(ra, fa, ta, rb, fb, tb) -> bool:
        key = (ra, fa, ta, rb, fb, tb)
        if key not in memo:
            sa, sb = sweep(ra, fa, ta), sweep(rb, fb, tb)
            if sa is None or sb is None:
                memo[key] = False
            else:
                res = inter_robot_clearance(cell_id, {ra: sa, rb: sb},
                                            warning_distance_m=band)
                memo[key] = bool(res.get("collision_free"))
        return memo[key]

    # Whether the arm can stand at an operation at all — so a report can tell
    # "no collision-free pose" apart from "too close to the other arm".
    pair_clear.has_pose = lambda rid, key: pose(rid, key) is not None
    return pair_clear
