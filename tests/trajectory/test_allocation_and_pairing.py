"""The arms must not cross, and each must drive its own shortest tour.

Three separate defects in the MRTA stages produced one visible symptom — a strange
weld distribution with the arms reaching across each other — and each stage is
pinned here.

**Allocation split a row.** Taking the best-scoring arm per operation independently
let a single row of welds be divided. On ``welding_cell_2_small``'s door program the
middle row (three welds, all at y=-1.225) sits almost on the bisector between the
two bases and the deciding scores differ by 2.1%, 3.9% and 9.3% — noise in the third
decimal. It came out owned r2, r2, r1, so one arm had to reach ACROSS two of the
other's welds. Deciding each tie by its nearest allocated neighbour does not fix it
either: on a regular grid the contested row is EXACTLY equidistant from the row above
and below (0.287 m to each), so every tie is a second tie broken by the same noise.
The row has to be decided as a UNIT.

**The tour optimised the wrong objective.** ``_tsp_2opt`` minimised the closed cycle,
but the arm welds and then goes home — it never returns to its first weld. The
closing hop it was paying for penalises ending far from the start, which for a
boustrophedon over a grid is exactly where you want to end. Measured: robot2's nine
welds came out 2.121 m cyclic against 1.768 m for the optimal open path.

**The pairing threw the tours away.** The scheduler paired the arms to MAXIMISE the
distance between the two weld points, so every arm but the busiest re-picked from its
whole pending queue and discarded its optimised order (robot1 drove 1.849 m where its
own tour was 1.625 m). It did not buy safety in exchange: across the six pairs of a
shipped plan, weld-point separation correlates with true swept clearance at
**-0.607** — the wrong sign, because two arms facing each other are furthest apart
where each is most EXTENDED toward the other. The pair this heuristic liked best
(1.201 m, by far the widest) is the one that closed to 0.1153 m and e-stopped the
cell.

Host-only: pure numpy, no ROS, no Pinocchio. Whether a pair may actually move at once
is not this stage's question — see ``test_weld_step_gate.py``.
"""
from __future__ import annotations

import itertools
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

if "langchain_core.tools" not in sys.modules:
    _lc = types.ModuleType("langchain_core")
    _lct = types.ModuleType("langchain_core.tools")
    _lct.tool = lambda f: f
    sys.modules.setdefault("langchain_core", _lc)
    sys.modules["langchain_core.tools"] = _lct

import numpy as np  # noqa: E402

from agents.trajectory.tools.task_allocation import (  # noqa: E402
    _least_movement_schedule,
    _max_distance_schedule,
    _regional_allocate,
    _tsp_2opt,
)

# Two arms facing the workpiece from opposite sides in y, as on welding_cell_2_small.
ROBOTS = {
    "robot1": {"base_position": [0.0, -3.3, 0.0]},
    "robot2": {"base_position": [-1.2, 1.0, 0.0]},
}

# Five rows of three, on a regular pitch: the shape that makes the middle row a tie.
GRID = [
    {"id": f"W{r * 3 + c + 1:02d}",
     "position": [-1.5 + 0.3 * c, -2.0 + 0.4 * r, 1.0],
     "approach_vector": [0, 0, -1]}
    for r in range(5) for c in range(3)
]


# Bases mirrored about y=0, so three welds ON the bisector score identically for
# both arms — a tie group with nothing left to break it but the group rule itself.
# The two outer welds are decisive, and they are also the yardstick the grouping
# measures against: the ties are 0.3 m from each other and 1.5 m from anything owned.
SYMMETRIC = {
    "robot1": {"base_position": [0.0, -3.0, 0.0]},
    "robot2": {"base_position": [0.0, 3.0, 0.0]},
}
BISECTOR = [
    {"id": "T1", "position": [-0.3, 0.0, 1.0], "approach_vector": [0, 0, -1]},
    {"id": "T2", "position": [0.0, 0.0, 1.0], "approach_vector": [0, 0, -1]},
    {"id": "T3", "position": [0.3, 0.0, 1.0], "approach_vector": [0, 0, -1]},
    {"id": "D1", "position": [0.0, -1.5, 1.0], "approach_vector": [0, 0, -1]},
    {"id": "D2", "position": [0.0, 1.5, 1.0], "approach_vector": [0, 0, -1]},
]
TIED = {"T1", "T2", "T3"}


def _rows(seq: list[dict]) -> list[float]:
    return sorted({round(op["position"][1], 4) for op in seq})


def _travel(seq: list[dict]) -> float:
    p = [np.array(op["position"], dtype=float) for op in seq]
    return sum(float(np.linalg.norm(p[i + 1] - p[i])) for i in range(len(p) - 1))


def _optimal_open_path(points: list[dict]) -> float:
    """Brute-force shortest OPEN path — the yardstick _tsp_2opt is measured against."""
    best = float("inf")
    for perm in itertools.permutations(range(len(points))):
        best = min(best, _travel([points[i] for i in perm]))
    return best


# ── allocation: the boundary is a cut, not a comb ─────────────────────────────

def test_no_arm_reaches_across_another_arms_welds():
    """Contiguity is the whole point: each arm owns a band of rows, not a comb.

    An arm owning a row on the far side of a row the other arm owns is the
    reach-across that made the arms cross.
    """
    alloc = _regional_allocate(GRID, ROBOTS)

    bands = {rid: _rows(seq) for rid, seq in alloc.items() if seq}
    assert len(bands) == 2, f"both arms should get work, got {bands}"
    lo, hi = sorted(bands.values(), key=lambda rs: rs[0])
    assert lo[-1] < hi[0], (
        f"the arms' rows interleave — one reaches across the other: {bands}")


def test_a_contested_row_is_not_split_between_the_arms():
    """The measured defect: the tied middle row came out r2, r2, r1."""
    alloc = _regional_allocate(GRID, ROBOTS)

    owner_of = {op["id"]: rid for rid, seq in alloc.items() for op in seq}
    for r in range(5):
        row = [f"W{r * 3 + c + 1:02d}" for c in range(3)]
        owners = {owner_of[i] for i in row}
        assert len(owners) == 1, f"row {row} was split across {owners}"


def test_the_allocation_does_not_depend_on_the_input_order():
    """Determinism was the ask. A tie decided by visit order is not deterministic.

    Pass 2 measures every tie against a SNAPSHOT of pass 1 for exactly this reason.
    """
    base = {rid: [op["id"] for op in seq]
            for rid, seq in _regional_allocate(GRID, ROBOTS).items()}

    for shuffle in (list(reversed(GRID)), GRID[7:] + GRID[:7]):
        again = {rid: sorted(op["id"] for op in seq)
                 for rid, seq in _regional_allocate(shuffle, ROBOTS).items()}
        assert again == {rid: sorted(ids) for rid, ids in base.items()}


def test_every_operation_is_allocated_exactly_once():
    """A dropped weld is never performed, and the loss is silent this early."""
    alloc = _regional_allocate(GRID, ROBOTS)
    placed = [op["id"] for seq in alloc.values() for op in seq]
    assert sorted(placed) == sorted(op["id"] for op in GRID)


def test_an_arm_that_cannot_service_a_group_does_not_win_it():
    """The score knows base positions and approach vectors — not the fixtures.

    So an arm can win a tie it physically cannot serve. The predicate is consulted
    for the GROUP: an arm that cannot reach every member is not eligible for any of
    it, because splitting the group is the reach-across being removed. (It is
    consulted for tie groups ONLY — a decisive win is not relitigated here; the
    planner reports that one, with the pose attached.)
    """
    plain = _regional_allocate(BISECTOR, SYMMETRIC)
    winner = next(rid for rid, seq in plain.items()
                  if TIED <= {op["id"] for op in seq})

    def cannot(rid, op):
        return not (rid == winner and op["id"] in TIED)

    alloc = _regional_allocate(BISECTOR, SYMMETRIC, can_service=cannot)

    owners = {rid for rid, seq in alloc.items() if TIED & {op["id"] for op in seq}}
    assert owners == {"robot1" if winner == "robot2" else "robot2"}, (
        f"{winner} cannot service the group, so it must not own any of it: {owners}")


def test_the_group_is_the_unit_of_the_servicability_decision():
    """One member being out of reach moves the WHOLE group, not just that member.

    Leaving the rest behind is exactly the split boundary this stage exists to
    prevent — and the arm that keeps the remainder is then reaching past the arm
    that took the one weld.
    """
    plain = _regional_allocate(BISECTOR, SYMMETRIC)
    winner = next(rid for rid, seq in plain.items()
                  if TIED <= {op["id"] for op in seq})
    one = sorted(TIED)[1]

    alloc = _regional_allocate(
        BISECTOR, SYMMETRIC,
        can_service=lambda rid, op: not (rid == winner and op["id"] == one))

    owners = {rid for rid, seq in alloc.items() if TIED & {op["id"] for op in seq}}
    assert len(owners) == 1 and winner not in owners, (
        f"one unreachable member must move all of {sorted(TIED)}: {owners}")


def test_a_group_no_arm_can_service_still_gets_allocated():
    """Fail-open on COST. An unreachable weld is the planner's report to make.

    Dropping it here would lose the operation silently; refusing the whole plan here
    would refuse it without ever naming which weld or which arm. Score decides, and
    the planner says ``goal_in_collision`` with the pose attached.
    """
    alloc = _regional_allocate(GRID, ROBOTS, can_service=lambda rid, op: False)

    placed = [op["id"] for seq in alloc.values() for op in seq]
    assert sorted(placed) == sorted(op["id"] for op in GRID)


def test_a_pinned_operation_still_overrides_everything():
    """``assigned_robot`` is a hard lock; the tie machinery must not relitigate it."""
    ops = [dict(op) for op in GRID]
    ops[0]["assigned_robot"] = "robot2"

    alloc = _regional_allocate(ops, ROBOTS)

    assert ops[0]["id"] in [op["id"] for op in alloc["robot2"]]


# ── the tour: an open path, because the arm goes home ─────────────────────────

def test_the_tour_is_the_shortest_open_path_not_the_shortest_cycle():
    """The arm never returns to its first weld, so the closing hop is not a cost.

    Brute-forced on a boustrophedon-shaped set, where the two objectives differ:
    the optimal open path ends at the far end, which the cyclic objective penalises.
    """
    points = [{"id": f"P{i}", "position": p} for i, p in enumerate(
        [[0.0, 0.0, 1.0], [0.3, 0.0, 1.0], [0.6, 0.0, 1.0],
         [0.6, 0.4, 1.0], [0.3, 0.4, 1.0], [0.0, 0.4, 1.0],
         [0.0, 0.8, 1.0], [0.3, 0.8, 1.0]])]

    tour = _tsp_2opt(points)

    assert sorted(op["id"] for op in tour) == sorted(op["id"] for op in points)
    assert _travel(tour) <= _optimal_open_path(points) + 1e-9, (
        f"{_travel(tour):.4f} m against optimal {_optimal_open_path(points):.4f} m")


def test_the_tour_starts_where_it_pays_to_start():
    """The start point is a free variable once the ends are not joined.

    Nearest-neighbour from an arbitrary first point is what made the cyclic tour
    look acceptable, so the construction is run from EVERY start and the best kept.
    """
    # A line with an outlier: starting at the outlier is 1.0 m, starting in the
    # middle is 1.5 m. A single arbitrary construction can pick either.
    points = [{"id": "A", "position": [0.0, 0.0, 0.0]},
              {"id": "B", "position": [0.5, 0.0, 0.0]},
              {"id": "C", "position": [1.0, 0.0, 0.0]}]

    tour = _tsp_2opt(list(reversed(points)))

    assert [op["id"] for op in tour] in (["A", "B", "C"], ["C", "B", "A"]), (
        "the path must run along the line, not double back through the middle")
    assert _travel(tour) == 1.0


def test_a_tour_of_one_or_none_is_returned_untouched():
    assert _tsp_2opt([]) == []
    one = [{"id": "A", "position": [0.0, 0.0, 0.0]}]
    assert _tsp_2opt(one) == one


# ── the pairing: least movement, and the distance is telemetry ────────────────

def test_every_arm_keeps_its_own_tour():
    """The defect: all but the busiest arm re-picked from its whole queue.

    Its optimised order was discarded to stand far from the other arm — paying real
    movement for a quantity that anti-correlates with clearance.
    """
    seqs = {
        "robot1": [{"id": f"A{i}", "position": [0.1 * i, 0.0, 1.0]} for i in range(4)],
        "robot2": [{"id": f"B{i}", "position": [0.1 * i, 1.0, 1.0]} for i in range(6)],
    }

    steps = _least_movement_schedule(seqs)

    for rid, seq in seqs.items():
        driven = [st[rid]["id"] for st in steps if rid in st]
        assert driven == [op["id"] for op in seq], (
            f"{rid} drove {driven}, not the tour it was given")


def test_every_operation_reaches_exactly_one_step():
    """TOTALITY: an allocated operation that never gets scheduled is never welded,
    and there is no cache entry for the assembler to notice is missing."""
    seqs = {
        "robot1": [{"id": f"A{i}", "position": [0.1 * i, 0.0, 1.0]} for i in range(4)],
        "robot2": [{"id": f"B{i}", "position": [0.1 * i, 1.0, 1.0]} for i in range(6)],
        "robot3": [{"id": "C0", "position": [0.5, 2.0, 1.0]}],
    }

    steps = _least_movement_schedule(seqs)

    scheduled = [op["id"] for st in steps
                 for rid, op in st.items() if isinstance(op, dict)]
    expected = [op["id"] for seq in seqs.values() for op in seq]
    assert sorted(scheduled) == sorted(expected)
    assert len(steps) == max(len(s) for s in seqs.values()), (
        "no arm's work may be truncated by another arm having less to do")


def test_the_weld_point_distance_is_reported_but_not_optimised():
    """It stays in the step as telemetry — read as "how far apart the POINTS are".

    Removing it would break every reader; gating on it is the -0.607 mistake. So it
    is reported, and the schedule is unaffected by it: here the arms' own tours put
    the CLOSEST points together and the schedule leaves them that way.
    """
    seqs = {
        "robot1": [{"id": "A0", "position": [0.0, 0.0, 1.0]},
                   {"id": "A1", "position": [1.0, 0.0, 1.0]}],
        "robot2": [{"id": "B0", "position": [0.0, 0.5, 1.0]},
                   {"id": "B1", "position": [1.0, 0.5, 1.0]}],
    }

    steps = _least_movement_schedule(seqs)

    assert [st["robot1"]["id"] for st in steps] == ["A0", "A1"]
    assert [st["robot2"]["id"] for st in steps] == ["B0", "B1"], (
        "pairing A0 with B1 would be 'further apart' and 0.618 m of extra travel")
    assert all(st["min_inter_robot_dist_m"] == 0.5 for st in steps)


def test_a_single_arm_step_carries_no_pair_distance():
    """There is no pair, so there is no distance. A 0.0 would read as a breach."""
    steps = _least_movement_schedule(
        {"robot1": [{"id": "A0", "position": [0.0, 0.0, 1.0]}]})

    assert "min_inter_robot_dist_m" not in steps[0]


def test_the_old_name_still_resolves_to_the_new_behaviour():
    """``_max_distance_schedule`` is imported by committed callers and tests.

    The name is now a lie about the objective, so it is an alias rather than a
    second implementation — two schedulers is how the two would drift apart.
    """
    assert _max_distance_schedule is _least_movement_schedule
