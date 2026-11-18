"""Which way each arm walks its tour is calculated; arms too close are reported.

Measured on welding_cell_2_small (four arms, two per side of a car body): every arm walked
its tour from the weld nearest its base, so the two arms of a side both FINISHED at
the shared B-pillar / roof bow at the same time — the late steps conflicted, an arm
parked there blocked its neighbour, and the plan came out serial. An open tour
costs the same walked either way, so the direction is a free choice:

* :func:`orient_for_parallelism` composes the program for the candidate
  directions with the arms' own geometry and keeps the most parallel one — for any
  cell and any arm count (exhaustive up to ``exhaustive_max`` arms, flip-one-at-a-
  time above). A tie keeps the given directions.
* :func:`standing_conflicts` / :func:`contention_advice` name the weld pairs two
  arms can never work at the same time — told, not discovered step by step — and
  :func:`unposable_operations` keeps "this arm has no pose there" apart from
  "the arms are too close".

The geometry is injected (``pair_clear``), as in test_parallel_schedule.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools.parallel_schedule import (  # noqa: E402
    HOME, compose_parallel_steps, contention_advice, orient_for_parallelism,
    schedule_score, standing_conflicts, unposable_operations,
)

# Two arms on one side of a part, working along y; their zones meet at y = 0.
Y = {**{f"A{i}": -1.0 + 0.25 * i for i in range(5)},       # A0 -1.0 ... A4 0.0
     **{f"B{i}": 1.2 - 0.25 * i for i in range(5)},        # B0 1.2 ... B4 0.2
     **{f"C{i}": 3.0 + 0.25 * i for i in range(5)},
     HOME: None}


def _seq(prefix):
    return [{"id": f"{prefix}{i}", "position": [0.0, Y[f"{prefix}{i}"], 1.0]} for i in range(5)]


def _too_close(sep=0.5):
    """Arms conflict when the welds they stand at / go to are within ``sep``."""
    def pair_clear(ra, fa, ta, rb, fb, tb):
        ya, yb = Y.get(ta), Y.get(tb)
        return ya is None or yb is None or abs(ya - yb) >= sep
    return pair_clear


def _strict(seqs, pc):
    return compose_parallel_steps(seqs, pc, lookahead=1, allow_park=False)


def test_arms_that_converge_on_a_shared_boundary_are_turned_to_sweep_together():
    seqs = {"r1": _seq("A"), "r2": _seq("B")}   # both walk TOWARD the middle
    baseline = _strict(seqs, _too_close())
    oriented, steps, info = orient_for_parallelism(seqs, _too_close(), compose=_strict)
    assert len(info["reversed"]) == 1                       # one arm turned round
    assert schedule_score(steps) < schedule_score(baseline)
    assert all(st["parallel_arms"] == 2 for st in steps)    # parallel throughout
    assert info["baseline_steps"] == len(baseline) and info["steps"] == len(steps)
    # The same welds, each arm's own tour — only the direction changed.
    for r in seqs:
        ids = [o["id"] for o in oriented[r]]
        assert ids in ([o["id"] for o in seqs[r]], [o["id"] for o in seqs[r]][::-1])


def test_a_tie_keeps_the_given_directions():
    seqs = {"r1": _seq("A"), "r3": _seq("C")}   # far apart: every direction works
    oriented, _steps, info = orient_for_parallelism(seqs, _too_close(), compose=_strict)
    assert info["reversed"] == []
    assert oriented == {r: list(s) for r, s in seqs.items()}


def test_many_arms_are_oriented_one_flip_at_a_time():
    seqs = {"r1": _seq("A"), "r2": _seq("B"), "r3": _seq("C")}
    _o, steps, info = orient_for_parallelism(seqs, _too_close(), compose=_strict,
                                             exhaustive_max=1)
    assert info["programs_composed"] < 2 ** 3              # not exhaustive
    assert all(st["parallel_arms"] == 3 for st in steps)


def test_weld_pairs_no_order_can_make_parallel_are_reported():
    seqs = {"r1": _seq("A"), "r2": _seq("B")}
    conflicts = standing_conflicts(seqs, _too_close())
    pairs = {tuple(c["operations"]) for c in conflicts}
    assert ("A4", "B4") in pairs and ("A3", "B4") in pairs  # 0.2 and 0.45 apart
    assert ("A0", "B0") not in pairs
    advice = contention_advice(conflicts)
    assert len(advice) == 1 and "r1 and r2" in advice[0] and "reallocate_operation" in advice[0]
    assert "drop" in advice[0]                              # ... "do not drop either arm"


def test_no_pose_is_not_reported_as_too_close():
    pc = _too_close()
    pc.has_pose = lambda rid, key: key != "A4"
    conflicts = standing_conflicts({"r1": _seq("A"), "r2": _seq("B")}, pc)
    assert not any("A4" in c["operations"] for c in conflicts)
    assert unposable_operations({"r1": _seq("A"), "r2": _seq("B")}, pc) == [
        {"robot": "r1", "operation": "A4"}]
    assert unposable_operations({"r1": _seq("A")}, _too_close()) == []   # no pose model
