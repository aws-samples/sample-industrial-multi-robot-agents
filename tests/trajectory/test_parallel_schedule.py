"""Steps are composed for ALL arms at once, for any number of arms.

The geometry is injected: ``pair_clear`` is a stand-in that declares which
(arm, operation) combinations conflict. What is pinned is the scheduling
contract: the most arms work in each step, an arm may reach a little into its own
queue to find an operation that fits, a parked arm counts, and no operation is
ever lost.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools.parallel_schedule import HOME, compose_parallel_steps  # noqa: E402


def _ops(prefix, n):
    return [{"id": f"{prefix}{i}", "position": [0, 0, 0]} for i in range(1, n + 1)]


def _conflicts(bad: set):
    """Arms collide when their TARGETS form a pair in ``bad`` (order-free)."""
    def pair_clear(ra, fa, ta, rb, fb, tb):
        return frozenset((ta, tb)) not in bad
    return pair_clear


def _ids(steps):
    return [{k: v["id"] for k, v in st.items() if isinstance(v, dict)} for st in steps]


def _all_ids(steps):
    return sorted(v["id"] for st in steps for v in st.values() if isinstance(v, dict))


def test_clear_arms_all_work_in_every_step():
    seqs = {"r1": _ops("A", 3), "r2": _ops("B", 3), "r3": _ops("C", 3)}
    steps = compose_parallel_steps(seqs, _conflicts(set()))
    assert len(steps) == 3
    assert all(st["parallel_arms"] == 3 for st in steps)
    assert _ids(steps)[0] == {"r1": "A1", "r2": "B1", "r3": "C1"}


def test_an_arm_reaches_into_its_queue_to_stay_parallel():
    # A1 conflicts with B1, but B2 fits beside A1: take B2 now, keep both arms busy.
    seqs = {"r1": _ops("A", 2), "r2": _ops("B", 2)}
    steps = compose_parallel_steps(seqs, _conflicts({frozenset(("A1", "B1"))}))
    assert _ids(steps)[0] == {"r1": "A1", "r2": "B2"}
    assert all(st["parallel_arms"] == 2 for st in steps)


def test_strict_tour_order_is_available():
    seqs = {"r1": _ops("A", 2), "r2": _ops("B", 2)}
    steps = compose_parallel_steps(seqs, _conflicts({frozenset(("A1", "B1"))}),
                                   lookahead=1)
    assert steps[0]["parallel_arms"] == 1
    assert steps[0]["waiting"]


def test_a_parked_arm_is_part_of_the_check():
    # r2 finished at B1; A2 conflicts with the arm STANDING at B1.
    def pair_clear(ra, fa, ta, rb, fb, tb):
        return not ({ta, tb} == {"A2", "B1"})
    seqs = {"r1": _ops("A", 2), "r2": _ops("B", 1)}
    steps = compose_parallel_steps(seqs, pair_clear, lookahead=1)
    assert _ids(steps)[0] == {"r1": "A1", "r2": "B1"}
    # r2 standing at B1 blocks A2, so r2 parks at HOME in the same step
    assert _ids(steps)[1] == {"r1": "A2"}
    assert steps[1]["park"] == ["r2"]
    assert not steps[1].get("conflict")


def test_without_parking_a_blocked_arm_is_flagged_not_dropped():
    def pair_clear(ra, fa, ta, rb, fb, tb):
        return not ({ta, tb} == {"A2", "B1"})
    seqs = {"r1": _ops("A", 2), "r2": _ops("B", 1)}
    steps = compose_parallel_steps(seqs, pair_clear, lookahead=1, allow_park=False)
    assert steps[1].get("conflict") is True
    assert _ids(steps)[1] == {"r1": "A2"}


def test_the_search_looks_ahead_across_steps():
    # Greedy would take A1+B1 now; but then B2 conflicts with A2 AND A3, so r2
    # idles later. Taking B2 first lets both arms stay busy every step.
    bad = {frozenset(("A2", "B2")), frozenset(("A3", "B2"))}
    seqs = {"r1": _ops("A", 3), "r2": _ops("B", 3)}
    steps = compose_parallel_steps(seqs, _conflicts(bad))
    assert len(steps) == 3 and all(st["parallel_arms"] == 2 for st in steps)


def test_the_most_arms_win_over_tour_order():
    seqs = {"r1": _ops("A", 2), "r2": _ops("B", 2), "r3": _ops("C", 2)}
    # A1 conflicts with everything first in line; A2 fits both.
    bad = {frozenset(("A1", "B1")), frozenset(("A1", "C1"))}
    steps = compose_parallel_steps(seqs, _conflicts(bad))
    assert steps[0]["parallel_arms"] == 3
    assert _ids(steps)[0] == {"r1": "A2", "r2": "B1", "r3": "C1"}


def test_many_arms_use_the_greedy_search_and_lose_nothing():
    seqs = {f"r{i}": _ops(f"X{i}_", 3) for i in range(1, 7)}
    bad = {frozenset(("X1_1", "X2_1")), frozenset(("X3_2", "X4_2"))}
    steps = compose_parallel_steps(seqs, _conflicts(bad))
    assert _all_ids(steps) == sorted(o["id"] for s in seqs.values() for o in s)
    assert max(st["parallel_arms"] for st in steps) >= 5


def test_every_operation_is_scheduled_exactly_once_even_when_all_conflict():
    seqs = {"r1": _ops("A", 3), "r2": _ops("B", 2)}
    steps = compose_parallel_steps(seqs, lambda *a: False)
    assert _all_ids(steps) == ["A1", "A2", "A3", "B1", "B2"]
    assert all(st["parallel_arms"] == 1 for st in steps)


def test_arms_start_at_home():
    seen = []

    def pair_clear(ra, fa, ta, rb, fb, tb):
        seen.append((fa, fb))
        return True
    compose_parallel_steps({"r1": _ops("A", 1), "r2": _ops("B", 1)}, pair_clear)
    assert seen[0] == (HOME, HOME)
