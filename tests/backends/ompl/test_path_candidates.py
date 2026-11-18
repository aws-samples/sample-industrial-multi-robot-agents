"""The shipped transit must be CHOSEN, not whichever one the RNG handed over.

A sampling planner returns whichever homotopy class its two trees happened to meet
in. Shortcutting cannot move a path between classes — `simplifyMax` collapses a
solved RRTConnect path straight back to its three states (start, meeting point,
goal) and the only shortcut available, start->goal, is blocked by the workpiece.
So the meeting point survives verbatim, and if it sits high above the cell then so
does the motion.

Measured on ``welding_cell_2_small``: ten solves of ONE weld transit, byte-for-byte
identical inputs, produced tool rises from 0.027 m to 2.767 m (peak tool height
1.618 m to 4.358 m). The operator watched whichever draw came up. Three legs of a
shipped 15-op plan arced 1.5-1.9 m over the cell to route around grazes of the car
body measuring 3 mm to 14 mm, and a flat path existed every time.

Directness — path length over ``||q_goal - q_start||`` — predicted the rise at
+0.878 and costs no collision checks and no FK, so it is the score. From those ten
candidates (lower bound 0.971 rad): ratios 1.05/1.55/1.59 rose 0.027/0.182/0.453 m
while everything above 1.6 rose 0.6-2.8 m.

What must NOT regress: a segment that used to plan must still plan. Selection is a
quality mechanism, and it may never turn an available path into a refusal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

ompl_planner = pytest.importorskip(
    "agents.trajectory.backends.ompl.planner", reason="needs the trajectory stack")

_path_joint_length = ompl_planner._path_joint_length
_path_directness = ompl_planner._path_directness
_pick_path_candidate = ompl_planner._pick_path_candidate
_path_candidate_policy = ompl_planner._path_candidate_policy


def _line(n: int, reach: float = 1.0) -> list:
    """A straight path from the origin to `reach` on joint 0, in n waypoints."""
    return [np.array([reach * i / (n - 1), 0.0, 0.0]) for i in range(n)]


def _detour(height: float) -> list:
    """Start and goal one radian apart, reached via an excursion on joint 1."""
    return [np.array([0.0, 0.0, 0.0]),
            np.array([0.5, height, 0.0]),
            np.array([1.0, 0.0, 0.0])]


def _solver(*batches):
    """A solve() that returns the given results in order, recording its calls."""
    calls = {"n": 0}

    def solve():
        i = calls["n"]
        calls["n"] += 1
        return batches[min(i, len(batches) - 1)]

    return solve, calls


# ── the score ────────────────────────────────────────────────────────────────

def test_a_straight_path_scores_one():
    """The lower bound is the straight line, so it must be the floor."""
    assert _path_directness(_line(2)) == pytest.approx(1.0)
    assert _path_directness(_line(9)) == pytest.approx(1.0)


def test_an_excursion_scores_above_one_in_proportion_to_its_size():
    """A bigger detour must score worse, or the score cannot rank candidates."""
    small = _path_directness(_detour(0.1))
    large = _path_directness(_detour(2.0))
    assert 1.0 < small < large
    # the 2.0-rad excursion is the dominant term: ~2*sqrt(0.5^2+2^2) over 1.0
    assert large == pytest.approx(4.123, abs=0.01)


def test_joint_length_is_the_sum_of_the_segments():
    assert _path_joint_length(_detour(2.0)) == pytest.approx(
        2 * float(np.hypot(0.5, 2.0)))
    assert _path_joint_length([np.array([1.0, 2.0, 3.0])]) == 0.0


def test_a_path_that_need_not_move_is_already_direct():
    """Zero travel must not divide by zero, and it has nothing to improve."""
    same = [np.array([0.4, 0.4, 0.4]), np.array([0.4, 0.4, 0.4])]
    assert _path_directness(same) == pytest.approx(1.0)
    assert _path_directness([np.array([0.1, 0.2, 0.3])]) == pytest.approx(1.0)


# ── the choice ───────────────────────────────────────────────────────────────

def test_the_most_direct_candidate_is_the_one_that_ships():
    """Shipping the first draw is what put a 1.8 m arc in front of the operator."""
    arc, flat = _detour(2.0), _detour(0.05)
    solve, calls = _solver((arc, None), (flat, None), (arc, None))

    # a target no candidate reaches, so selection cannot stop early and the
    # comparison across the whole batch is what is under test
    picked, failure = _pick_path_candidate(solve, attempts=3,
                                           directness_target=1.001)

    assert failure is None
    assert picked is flat, "the flattest candidate must win, not the first one"
    assert calls["n"] == 3, "no early stop is possible when nothing meets the target"


def test_a_candidate_that_meets_the_target_stops_the_search():
    """Each solve costs ~6-10 s on a real cell; a good path must not pay for more."""
    flat, arc = _detour(0.05), _detour(2.0)
    solve, calls = _solver((flat, None), (arc, None), (arc, None))

    picked, _ = _pick_path_candidate(solve, attempts=4, directness_target=1.6)

    assert picked is flat
    assert calls["n"] == 1, "the target was met on the first solve"


def test_the_target_admits_the_flat_draws_and_rejects_the_arcs():
    """Calibrated on the measured ten: <=1.6 rose <=0.453 m, >1.6 rose 0.6-2.8 m."""
    # a detour whose directness lands either side of the 1.6 cutoff
    accepted = _detour(0.35)   # ratio ~1.22
    rejected = _detour(1.20)   # ratio ~2.6
    assert _path_directness(accepted) < 1.6 < _path_directness(rejected)

    solve, calls = _solver((rejected, None), (accepted, None))
    picked, _ = _pick_path_candidate(solve, attempts=4, directness_target=1.6)

    assert picked is accepted
    assert calls["n"] == 2, "it must keep solving while only arcs have turned up"


def test_selection_never_turns_an_available_path_into_a_refusal():
    """A failed draw must not sink the batch — a later solve may still succeed."""
    flat = _detour(0.05)
    solve, calls = _solver((None, {"reason_code": "no_path_found"}), (flat, None))

    picked, failure = _pick_path_candidate(solve, attempts=3, directness_target=1.05)

    assert picked is flat, "one unlucky solve must not lose an available path"
    assert failure is None
    assert calls["n"] == 2


def test_when_every_attempt_fails_the_actionable_reason_survives():
    """`goal_in_collision` and friends are how the agent picks a recovery."""
    bad = {"reason_code": "goal_in_collision", "reason": "the fixture is in the way"}
    solve, calls = _solver((None, {"reason_code": "no_path_found"}), (None, bad))

    picked, failure = _pick_path_candidate(solve, attempts=2, directness_target=1.6)

    assert picked is None
    assert failure == bad, "the last, most specific failure must reach the caller"
    assert calls["n"] == 2


def test_a_single_attempt_is_exactly_the_old_behaviour():
    """OMPL_PATH_CANDIDATES=1 must be a true escape hatch, arc and all."""
    arc = _detour(2.0)
    solve, calls = _solver((arc, None))

    picked, failure = _pick_path_candidate(solve, attempts=1, directness_target=1.6)

    assert picked is arc and failure is None
    assert calls["n"] == 1


def test_a_nonsense_attempt_count_still_solves_once():
    """Availability first: a bad tuning value must not stop the arm from moving."""
    flat = _detour(0.05)
    for attempts in (0, -3):
        solve, calls = _solver((flat, None))
        picked, _ = _pick_path_candidate(solve, attempts=attempts,
                                         directness_target=1.6)
        assert picked is flat
        assert calls["n"] == 1


# ── the tuning ───────────────────────────────────────────────────────────────

def test_the_default_policy_spends_a_few_solves_at_the_measured_cutoff():
    monkey = pytest.MonkeyPatch()
    try:
        monkey.delenv("OMPL_PATH_CANDIDATES", raising=False)
        monkey.delenv("OMPL_PATH_DIRECTNESS", raising=False)
        attempts, target = _path_candidate_policy()
    finally:
        monkey.undo()
    assert attempts == 4
    assert target == pytest.approx(1.6)


def test_the_policy_is_tunable_without_a_code_change(monkeypatch):
    monkeypatch.setenv("OMPL_PATH_CANDIDATES", "7")
    monkeypatch.setenv("OMPL_PATH_DIRECTNESS", "2.5")
    assert _path_candidate_policy() == (7, pytest.approx(2.5))


def test_an_unreachable_target_is_floored_rather_than_burning_every_solve(
        monkeypatch):
    """Directness cannot go below 1.0, so a lower target would never stop early."""
    monkeypatch.setenv("OMPL_PATH_DIRECTNESS", "0.4")
    _, target = _path_candidate_policy()
    assert target == pytest.approx(1.0)


def test_unparseable_tuning_falls_back_to_the_defaults(monkeypatch):
    monkeypatch.setenv("OMPL_PATH_CANDIDATES", "lots")
    monkeypatch.setenv("OMPL_PATH_DIRECTNESS", "flat-ish")
    assert _path_candidate_policy() == (4, pytest.approx(1.6))
