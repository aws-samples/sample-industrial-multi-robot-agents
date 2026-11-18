"""
Multi-Robot Task Allocation — who welds what, in which order.

Pure scheduling geometry: regional allocation of operations to arms, per-arm
sequencing (2-opt TSP), thermal spreading so consecutive welds are not adjacent,
and the interleaving that decides which operations may run in parallel.

Backend-neutral by construction — it decides the WORK, not the MOTION, so it is
independent of whichever motion planner turns an assignment into a trajectory.
It lived in ``rrt_planner`` for historical reasons only; nothing here is RRT.

WHAT THIS MODULE MAY AND MAY NOT DECIDE. It runs BEFORE any trajectory exists,
so it can see where the welds ARE but never how an arm gets to them. It therefore
owns COST — travel, load, thermal spacing — and it must not pretend to own SAFETY.
Whether two arms may move at once is decided where the trajectories exist, by the
geometric gate in ``behaviour_tree._gate_multi_arm_step``. Measured on
``welding_cell_2_small``, that division is not a nicety: across the six parallel pairs
of a shipped plan, the correlation between weld-point separation and the arms'
true swept clearance is **-0.607** — the wrong sign — and the pair whose GOAL
poses stand furthest apart (0.9212 m, the clearest of the six) is the one whose
transit closes to 0.1153 m and e-stops the cell. No positional proxy available at
this stage predicts arm clearance, so none is used as one.

Import-safe: numpy only.
"""
from __future__ import annotations

import numpy as np

# How decisive a robot's win must be for :func:`_regional_allocate` to take it at
# face value, as a fraction of the winning score. Below this the operation is a
# TIE and is placed by contiguity instead (see that function).
#
# 0.15 is measured, not chosen. On ``welding_cell_2_small``'s 15-weld door program the
# relative margins fall into two clusters with an empty band between them: the
# three welds of the contested middle row score 2.1%, 3.9% and 9.3%, and the next
# margin up is 21.0%, rising to 50.9%. Any threshold in (0.093, 0.210) separates
# "genuinely this arm's weld" from "coin-flip"; 0.15 sits in the middle of that
# band rather than on either edge of it.
_TIE_MARGIN_REL = 0.15

#: An arm standing on the FAR side of an operation's approach cannot bring the tool
#: in along it: below this horizontal alignment it is not a candidate at all. A
#: little negative, so an arm off to the side (perpendicular approach) still is.
_SIDE_MIN_ALIGN = -0.2

#: Rebalancing (see :func:`_rebalance`). It starts only when an arm that COULD do
#: work is starved — under this fraction of the mean load of the arms that can — so
#: an ordinary uneven split (6/9 on welding_cell_2_small, chosen for contiguity) is left
#: alone; and a weld moves only to an arm scoring at least this fraction of its
#: owner, i.e. one that is genuinely a good second choice for it.
_BALANCE_STARVED = 0.5
_BALANCE_MIN_REL = 0.6
_BALANCE_MIN_GAP = 2.0


# ─── MRTA pipeline helpers ────────────────────────────────────────────────────

def _robot_bases(robots: dict) -> dict[str, np.ndarray]:
    """``{robot_id: base_position}`` for the arms that declare one."""
    return {
        rid: np.array(rcfg["base_position"], dtype=float)
        for rid, rcfg in robots.items()
        if isinstance(rcfg, dict) and "base_position" in rcfg
    }


def _robot_scores(op: dict, robot_bases: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """Every robot RANKED for one operation, best first. THE allocation policy.

    The score prefers the arm whose inward-facing direction best aligns with the
    operation's approach vector and which stands closest to the point. Ties break
    on robot id so a ranking is reproducible across runs.

    This returns the whole ranking, not just the winner, because a first choice
    that cannot physically reach a point has to hand it to the SECOND choice — and
    that reassignment must consult the same policy the initial allocation used. Two
    functions that each decide "which robot" is the N-copies-of-one-decision bug
    this repo keeps paying for (the planner and the monitor's collision models drifted
    exactly that way); :func:`_regional_allocate` takes ``[0]`` of this list and
    :func:`_reallocation_candidates` takes the rest.
    """
    pos = np.array(op.get("position", [0, 0, 0]), dtype=float)
    scored: list[tuple[str, float]] = []
    for ns, base in robot_bases.items():
        dist  = float(np.linalg.norm(pos - base))
        score = (1.0 + _alignment(op, base)) / (dist + 0.1)
        scored.append((ns, score))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


def _alignment(op: dict, base: np.ndarray) -> float:
    """How well an arm at ``base`` can bring the tool in along the approach.

    ``approach_vector`` is the tool axis INTO the operation, so the arm has to stand
    BEHIND the point: base -> point should run along the approach. +1 is directly
    behind, -1 the far side (the tool would have to come back towards its own
    base), 0 off to the side. Horizontal only: a top-down weld (``[0, 0, -1]``)
    can be reached from any side, so it scores 0 here and distance alone decides.

    The sign used to be inverted (``-to_point``), which ranked the arm on the FAR
    side first. Invisible on top-down welds, where it reduced to "the weld is above
    the base" for every arm; on welding_cell_2_small's side-on welds it sent 24 of 27 to the
    wrong arm, 15 of them out of reach.
    """
    av = np.array(op.get("approach_vector", [0, 0, -1]), dtype=float)
    to_point = np.array(op.get("position", [0, 0, 0]), dtype=float) - base
    av_n = av / (np.linalg.norm(av) + 1e-12)
    horiz = np.linalg.norm(to_point[:2])
    if horiz < 1e-9:
        return 0.0
    return float(np.dot(av_n[:2], to_point[:2] / horiz))


def _eligible(op: dict, robots: dict, robot_bases: dict) -> list[str]:
    """The arms that can geometrically take ``op``: within their declared
    ``workspace_radius`` and not on the far side of its approach.

    No arm is assumed to reach every point of the cell. An arm with no declared
    radius is not known to be out of reach (the planner's IK is the authority).

    The radius is compared HORIZONTALLY: an arm's reach is centred on its shoulder,
    not on the floor its base stands on, so a 3D distance from the base reads a
    roof rail 1.4 m up as out of reach when the arm welds it comfortably
    (welding_cell_2_small: 3.0 m from the base, about 2.7 m from the shoulder). The
    horizontal envelope is the declared reach; IK decides the rest.
    """
    pos = np.array(op.get("position", [0, 0, 0]), dtype=float)
    out = []
    for ns, base in robot_bases.items():
        radius = (robots.get(ns) or {}).get("workspace_radius")
        if radius is not None and float(np.linalg.norm((pos - base)[:2])) > float(radius):
            continue
        if _alignment(op, base) < _SIDE_MIN_ALIGN:
            continue
        out.append(ns)
    return out


def _reallocation_candidates(
    op: dict,
    robots: dict,
    exclude: "set[str] | list[str]" = (),
) -> list[dict]:
    """The arms that could TAKE OVER an operation, best first.

    Used when the operation's current owner cannot do it — the goal pose collides,
    no path exists, or the leaned pose will not solve. Ranking comes from
    :func:`_robot_scores`, so a handover follows the same policy as the original
    allocation instead of a second, hand-rolled one.

    Each entry also carries the arm's straight-line distance to the point against
    its declared ``workspace_radius``. A candidate outside its own reach is
    reported with ``within_workspace: false`` rather than silently dropped: the
    caller then knows the operation has no taker at all, which is a different (and
    final) answer from "try this next arm".
    """
    excluded = set(exclude)
    bases = _robot_bases(robots)
    pos = np.array(op.get("position", [0, 0, 0]), dtype=float)

    out: list[dict] = []
    for rank, (ns, score) in enumerate(_robot_scores(op, bases)):
        if ns in excluded:
            continue
        dist = float(np.linalg.norm(pos - bases[ns]))
        radius = (robots.get(ns) or {}).get("workspace_radius")
        radius = float(radius) if radius is not None else None
        out.append({
            "robot_ns": ns,
            "rank": rank + 1,
            "score": round(score, 6),
            "distance_from_base_m": round(dist, 4),
            "workspace_radius_m": radius,
            # No declared radius means "not known to be out of reach" — the
            # planner's IK is the authority, this is only a cheap pre-filter.
            "within_workspace": True if radius is None else dist <= radius,
        })
    return out


def _regional_allocate(operations: list[dict], robots: dict,
                       can_service=None, unreachable: "list | None" = None
                       ) -> dict[str, list[dict]]:
    """Allocate operations to robots.

    An operation may pin itself to a robot with an optional ``assigned_robot``
    field (used for single-robot operations such as part handling or gluing);
    that operation is placed on the named robot unconditionally. Every other
    operation is allocated automatically by the approach-vector / proximity
    ranking in :func:`_robot_scores`, so allocation is a planning-time decision,
    not a property baked into the task — and a later handover can re-consult the
    same ranking.

    CONTIGUITY. Taking ``argmax`` of that ranking per operation, independently, is
    what let a single row of welds be split between two arms. On
    ``welding_cell_2_small``'s door program the middle row (three welds, all at
    y=-1.225) sits almost exactly on the bisector between the two bases, and the
    scores that decide its ownership differ by 2.1%, 3.9% and 9.3% — noise in the
    third decimal place. The row came out owned r2, r2, r1, so one arm had to reach
    ACROSS two of the other's welds to serve the third. Nothing in a per-operation
    argmax can see that, because it never looks at where its neighbours went.

    So a win is only taken at face value when it is DECISIVE (``_TIE_MARGIN_REL``).
    The rest are deferred, GROUPED, and placed as whole groups — one arm per group.
    The grouping is what buys contiguity, and it has to be the unit of decision:
    deciding each tied operation on its own, even by its nearest already-allocated
    neighbour, does NOT keep the boundary whole. On a regular weld grid the contested
    row is EXACTLY equidistant from the row above and the row below (0.287 m to each,
    to the last bit), so every tied operation is a second tie, broken by the same
    third-decimal score that caused the first — and the row came out r1, r2, r1 again.
    Deciding the row as one unit is what makes the cut a cut.

    A group is a set of tied operations that are closer to EACH OTHER than to any
    confidently-allocated operation — no radius to tune, and it scales with whatever
    pitch the weld grid happens to have. Each group goes to the arm with the highest
    TOTAL score across its members: the arm the boundary as a whole leans toward,
    rather than whichever member is visited first. A lone tie (no tied neighbours) is
    a group of one and lands on its own best arm, unchanged.

    ``can_service(robot_id, operation) -> bool`` is an optional predicate that says
    whether an arm can actually SERVICE an operation — reach the pose without a link
    inside the fixture. It is consulted for tie groups ONLY, and an arm that cannot
    service every member of a group is not eligible for it however well it scores.
    It exists because the score is a function of base position and approach vector
    alone and knows nothing about the fixtures, so a tie can be resolved toward an
    arm that physically cannot stand at those welds; without the predicate that
    surfaces as a planner ``goal_in_collision`` two stages later and has to be
    repaired by a handover. It is a filter on the IMPOSSIBLE, not a preference — on
    ``welding_cell_2_small`` today it vetoes nothing, because every arm has a
    collision-free configuration at every weld (see ``pinocchio.run_ik``, which
    ranks its IK branches free-branch-first; before it did, the contested row read
    as unreachable for robot2 with arm link 10 measured 0.0839 m inside
    ``FIXTURE_A_v102`` — a branch-selection artefact, not a reach limit).

    It stays a CALLBACK because this module is numpy-only and backend-neutral on
    purpose; the caller that owns a kinematic model supplies one (see
    ``planning_tools._servicability_check``).

    On this cell the grouping alone therefore decides the boundary, and it decides it
    as a unit: the contested row leans to robot2 on total score, giving robot1 the
    three rows nearest its base (W01-W06) and robot2 the rest (W07-W15). Neither arm
    reaches past the other. That the crossing-free cut is 6/9 rather than 8/7 is the
    intended trade — evening the split means handing a weld back across the boundary,
    which is the reach-across this exists to remove. Cycle time is set by the busier
    arm either way, and cycle time is the cheaper thing to lose.
    """
    robot_list = list(robots.keys())
    allocation: dict[str, list[dict]] = {r: [] for r in robot_list}

    if len(robot_list) < 2:
        if robot_list:
            allocation[robot_list[0]] = operations
        return allocation

    robot_bases = _robot_bases(robots)

    # Pass 1 — pinned operations, then every operation whose best arm wins
    # decisively. Ties are held back with their ranking. Only ELIGIBLE arms are
    # ranked (reach + approach side, see _eligible): no arm is assumed to reach
    # every point. An operation no arm can reach is reported in ``unreachable``
    # (when the caller asks) instead of being handed to an arm that cannot do it.
    deferred: list[tuple[dict, list[tuple[str, float]]]] = []
    rankings: dict[int, list[tuple[str, float]]] = {}
    pinned_ids: set = set()
    for op in operations:
        # Hard lock: honour an explicit per-operation robot assignment.
        pinned = op.get("assigned_robot") or op.get("robot_ns")
        if pinned in allocation:
            allocation[pinned].append(op)
            pinned_ids.add(id(op))
            continue

        ranked = _robot_scores(op, robot_bases)
        geo = set(_eligible(op, robots, robot_bases)) if robot_bases else set()
        if robot_bases and not geo and unreachable is not None:
            unreachable.append({
                "operation_id": op.get("id"),
                "reason": "no arm within its workspace_radius on the approach side",
                "candidates": [
                    {"robot_ns": ns, "distance_from_base_m": round(float(np.linalg.norm(
                        np.array(op.get("position", [0, 0, 0]), float) - robot_bases[ns])), 3),
                     "workspace_radius_m": (robots.get(ns) or {}).get("workspace_radius"),
                     "approach_alignment": round(_alignment(op, robot_bases[ns]), 3)}
                    for ns, _s in ranked]})
            continue
        if geo:
            ranked = [(ns, s) for ns, s in ranked if ns in geo]
        # The kinematic check, when the caller has one: an arm that cannot hold the
        # pose is not ranked first. Fail-open — if no eligible arm passes, the score
        # decides and the planner reports the pose it could not reach.
        if can_service is not None and len(ranked) > 1:
            serviceable = [(ns, s) for ns, s in ranked if can_service(ns, op)]
            if serviceable and serviceable[0][0] != ranked[0][0]:
                ranked = serviceable + [t for t in ranked if t not in serviceable]
        rankings[id(op)] = ranked
        if not ranked:
            allocation[robot_list[0]].append(op)
            continue
        if len(ranked) < 2:
            allocation[ranked[0][0]].append(op)
            continue

        best, second = ranked[0][1], ranked[1][1]
        margin = (best - second) / max(abs(best), abs(second), 1e-9)
        if margin >= _TIE_MARGIN_REL:
            allocation[ranked[0][0]].append(op)
        else:
            deferred.append((op, ranked))

    # Pass 2 — group the ties, then place each group as a unit. Everything is
    # measured against a SNAPSHOT of pass 1, so the outcome does not depend on the
    # order the operations happen to be visited in.
    if deferred:
        owned = [np.array(o.get("position", [0.0, 0.0, 0.0]), dtype=float)
                 for seq in allocation.values() for o in seq]
        pos = [np.array(op.get("position", [0.0, 0.0, 0.0]), dtype=float)
               for op, _r in deferred]
        # How far each tie is from the nearest CONFIDENTLY allocated operation. This
        # is the yardstick the grouping uses, so no distance threshold is invented.
        # With nothing allocated at all there is no yardstick and no grouping: every
        # tie is its own group, decided by its own ranking.
        reach = [min((float(np.linalg.norm(p - q)) for q in owned), default=0.0)
                 for p in pos]

        # Union-find over "closer to each other than to anyone already allocated".
        parent = list(range(len(deferred)))

        def _find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(deferred)):
            for j in range(i + 1, len(deferred)):
                if float(np.linalg.norm(pos[i] - pos[j])) < min(reach[i], reach[j]):
                    parent[_find(i)] = _find(j)

        groups: dict[int, list[int]] = {}
        for i in range(len(deferred)):
            groups.setdefault(_find(i), []).append(i)

        # Each group to the arm it leans toward as a WHOLE — among the arms that can
        # actually service every member of it. Summing the same score pass 1 used
        # keeps one policy; ties break on robot id, so it is determined.
        for _root, members in sorted(groups.items()):
            totals: dict[str, float] = {}
            for i in members:
                for rid, score in deferred[i][1]:
                    totals[rid] = totals.get(rid, 0.0) + score
            # Only an arm ranked for EVERY member can own the group.
            common = set.intersection(*[{rid for rid, _s in deferred[i][1]} for i in members])
            totals = {rid: t for rid, t in totals.items() if rid in common} or totals
            eligible = dict(totals)
            if can_service is not None:
                eligible = {
                    rid: t for rid, t in totals.items()
                    if all(can_service(rid, deferred[i][0]) for i in members)
                }
                # No arm can service the whole group: this is not the tie-breaker's
                # problem to solve. Score decides, and the planner reports the
                # operation it cannot reach so a handover can split the group.
                if not eligible:
                    eligible = totals
            winner = min(eligible.items(), key=lambda t: (-t[1], t[0]))[0]
            for i in members:
                allocation[winner].append(deferred[i][0])

    _rebalance(allocation, rankings, pinned_ids)
    return allocation


def _rebalance(allocation: dict[str, list[dict]], rankings: dict,
               pinned_ids: set) -> int:
    """Give a STARVED arm work it is a good second choice for; returns moves made.

    Pass 1 and 2 decide each operation by its own geometry, so nothing stops a cell
    from ending with an arm idle that could have worked: the cycle time is then the
    busiest arm's, and the idle arm is capacity thrown away. Here an arm that is
    eligible for operations but holds fewer than ``_BALANCE_STARVED`` of the mean
    load (over the arms that are eligible for anything) takes, one at a time, the
    operation it scores best on RELATIVE to its current owner — only from a busier
    arm, only if it scores at least ``_BALANCE_MIN_REL`` of that owner, and never
    a pinned one. An ordinary uneven split is left as it is: this repairs
    starvation, it does not trade contiguity for a perfect count.
    """
    able = {rid for ranked in rankings.values() for rid, _s in ranked}
    if len(able) < 2:
        return 0
    owner = {id(op): rid for rid, seq in allocation.items() for op in seq}
    by_id = {id(op): op for seq in allocation.values() for op in seq}
    moves = 0
    for _ in range(sum(len(s) for s in allocation.values())):
        loads = {rid: len(allocation[rid]) for rid in able}
        mean = sum(loads.values()) / len(loads)
        best = None
        for rid in sorted(able, key=lambda r: (loads[r], r)):
            # Starved = well under the mean AND by a real count: on a 5-weld cell a
            # 4/1 split is the grouping keeping a tie row whole, not idle capacity.
            if loads[rid] >= _BALANCE_STARVED * mean or mean - loads[rid] < _BALANCE_MIN_GAP:
                break
            for oid, ranked in rankings.items():
                cur = owner.get(oid)
                if cur is None or cur == rid or oid in pinned_ids \
                        or loads.get(cur, 0) <= loads[rid] + 1:
                    continue
                scores = dict(ranked)
                if rid not in scores or scores.get(cur, 0.0) <= 0.0:
                    continue
                rel = scores[rid] / scores[cur]
                if rel >= _BALANCE_MIN_REL and (best is None or rel > best[0]):
                    best = (rel, oid, cur, rid)
            if best:
                break
        if best is None:
            return moves
        _rel, oid, cur, rid = best
        op = by_id[oid]
        allocation[cur] = [o for o in allocation[cur] if o is not op]
        allocation[rid].append(op)
        owner[oid] = rid
        moves += 1
    return moves


def _tsp_2opt(points: list[dict], start_near=None) -> list[dict]:
    """Nearest-neighbour + 2-opt for one arm's sequence — an OPEN PATH.

    The arm welds its points and then goes home; it does NOT return to its first
    weld. So the cost is the sum of the hops BETWEEN consecutive points, with no
    closing hop — and optimising the closed cycle instead is not a harmless
    approximation. It penalises ending far from the start, which for a boustrophedon
    over a weld grid is exactly where you want to end. Measured on
    ``welding_cell_2_small``: robot2's nine welds came out 2.121 m under the cyclic
    objective against 1.768 m for the optimal open path (brute-forced) — 17% of that
    arm's travel spent returning toward a point it never visits again.

    Two consequences for the search, both from the same fact that the ends are now
    free rather than joined:

    * the starting point matters, so nearest-neighbour is run from EVERY point and
      the best result kept — ``n`` cheap constructions instead of one arbitrary one;
    * a 2-opt move may reverse a segment that includes the first point (``i`` from 0,
      not 1). Under a cyclic objective that move is a no-op, which is why the loop
      used to skip it; on an open path it is how the path picks its endpoints.

    Ties break on the lower starting index so the sequence is reproducible.

    What this does NOT know is where the arm currently STANDS — this stage runs
    before any joint state exists, so it minimises travel between welds, not the
    approach to the first one.

    ``start_near`` (the arm's base position) orients the open path: the same
    optimal path, walked from the end CLOSEST to the arm. Arms working a shared
    part from opposite sides then begin on their own side, far apart — which is
    what keeps their first steps parallel.
    """
    if len(points) <= 1:
        return points

    positions = [np.array(p["position"], dtype=float) for p in points]
    n = len(positions)
    dist = [[float(np.linalg.norm(positions[i] - positions[j])) for j in range(n)]
            for i in range(n)]

    def path_len(t: list[int]) -> float:
        return sum(dist[t[i]][t[i + 1]] for i in range(len(t) - 1))

    best_tour, best_len = None, float("inf")
    for start in range(n):
        # Nearest-neighbour construction from this start.
        visited = [False] * n
        tour = [start]
        visited[start] = True
        for _ in range(n - 1):
            last = tour[-1]
            nxt = min((j for j in range(n) if not visited[j]),
                      key=lambda j: (dist[last][j], j))
            tour.append(nxt)
            visited[nxt] = True

        # 2-opt on the OPEN path: reverse tour[i:j+1], i from 0 so the endpoints
        # are free to move.
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    if path_len(cand) < path_len(tour) - 1e-9:
                        tour = cand
                        improved = True

        if path_len(tour) < best_len - 1e-9:
            best_tour, best_len = tour, path_len(tour)

    if start_near is not None:
        base = np.asarray(start_near, dtype=float)[:2]
        first = float(np.linalg.norm(positions[best_tour[0]][:2] - base))
        last = float(np.linalg.norm(positions[best_tour[-1]][:2] - base))
        if last < first:
            best_tour = best_tour[::-1]
    return [points[i] for i in best_tour]


def _thermal_spread(seq: list[dict], thermal_gap: int = 3) -> list[dict]:
    """Reorder so same-face operations are spaced >= thermal_gap apart."""
    if len(seq) <= thermal_gap:
        return seq

    result: list[dict] = []
    remaining = seq.copy()

    while remaining:
        # Find next point that's not same face as the last `thermal_gap` results
        recent_faces = {r.get("face") for r in result[-thermal_gap:] if r.get("face")}
        placed = False
        for i, op in enumerate(remaining):
            if op.get("face") not in recent_faces:
                result.append(remaining.pop(i))
                placed = True
                break
        if not placed:
            result.append(remaining.pop(0))  # no choice

    return result


def _least_movement_schedule(sequences: dict[str, list[dict]]) -> list[dict]:
    """Interleave per-robot sequences into parallel steps, for ANY robot count.

    Each step draws at most one operation from every robot, so a step is a set of
    operations that MAY run simultaneously — whether they actually may is decided
    later, by geometry (see below). Every arm simply takes the next operation in
    its OWN sequence, so the objective is LEAST ARM MOVEMENT: the tour that
    :func:`_tsp_2opt` optimised and :func:`_thermal_spread` adjusted is the tour
    each arm actually drives.

    WHY NOT MAXIMISE THE DISTANCE BETWEEN THE PAIRED WELDS. This function used to,
    and it was measurably the wrong objective — twice over.

    It cost movement. Only the busiest arm kept its tour; every other arm re-picked
    from its whole pending queue to stand as far as possible from the ops already in
    the step, discarding its own optimised order. On ``welding_cell_2_small`` robot1 drove
    W03 -> W01 -> W02, 1.849 m, where its own tour was W01 -> W02 -> W03, 1.625 m.

    And it did not buy safety in exchange. Across the six parallel pairs of a shipped
    plan, weld-point separation correlates with the arms' true swept clearance at
    **-0.607** — the wrong sign. Two arms facing each other from opposite sides are
    furthest apart at the point where each is most EXTENDED toward the other, so
    maximising the gap between the weld points minimises the gap between the arms.
    The pair this heuristic liked best (1.201 m apart, by a wide margin the largest)
    is the one that closed to 0.1153 m against a 0.2 m warning distance and e-stopped
    the cell; the pair it liked least (0.688 m) had ~4x the margin.

    So the objective is cost and the constraint is geometry, each with one owner:
    this function orders the work for least movement, and
    ``behaviour_tree._gate_multi_arm_step`` validates the swept volumes of the
    resulting pair — reordering, serialising, or refusing the step. That gate is
    where the trajectories exist, which is the only place the question can be
    answered: for the failing pair, the GOAL poses stand 0.9212 m apart (the clearest
    of the six) and the START poses 0.6908 m; the breach is entirely in transit, so no
    pose-level check available at this stage could have caught it.

    ``min_inter_robot_dist_m`` is still reported per step, now strictly as
    telemetry. Read it as "how far apart the weld POINTS are", not as a clearance,
    and do not gate on it — that is the -0.607 mistake.

    TOTALITY CONTRACT: every operation in ``sequences`` appears in exactly one
    step. The number of steps is the longest single-robot sequence, so no robot's
    work can be truncated by another robot having less to do. ``run_mrta_pipeline``
    asserts this and refuses the plan if it is ever violated.

    Args:
        sequences: ``{robot_id: [op, ...]}`` — each already ordered by _tsp_2opt.

    Returns:
        ``[{robot_id: op, ..., "min_inter_robot_dist_m": float?}, ...]``
    """
    pending = {rid: list(seq) for rid, seq in sequences.items() if seq}
    steps: list[dict] = []

    while pending:
        # Busiest robot first: it has the least slack, so it should never be the
        # one left holding operations after the others have drained. With every arm
        # keeping its own order this no longer changes WHICH op each arm takes —
        # only the order the step's keys are built in, which keeps step composition
        # reproducible.
        order = sorted(pending, key=lambda r: (-len(pending[r]), r))
        step: dict = {}
        placed: list[np.ndarray] = []

        for rid in order:
            queue = pending[rid]
            op = queue.pop(0)                    # this arm's own next weld
            step[rid] = op
            placed.append(np.array(op.get("position", [0.0, 0.0, 0.0]), dtype=float))
            if not queue:
                del pending[rid]

        # Telemetry only — the closest pair of weld POINTS in this step. Not a
        # clearance and not a gate; see the docstring.
        if len(placed) > 1:
            step["min_inter_robot_dist_m"] = round(min(
                float(np.linalg.norm(placed[i] - placed[j]))
                for i in range(len(placed)) for j in range(i + 1, len(placed))
            ), 3)
        steps.append(step)

    return steps


# The name this function carried while it optimised weld-point separation. Kept so
# committed callers and tests keep importing something real, but it is a lie about
# the behaviour: the schedule is now least-movement and the distance it used to
# maximise is reported, not optimised. Prefer :func:`_least_movement_schedule`.
_max_distance_schedule = _least_movement_schedule


