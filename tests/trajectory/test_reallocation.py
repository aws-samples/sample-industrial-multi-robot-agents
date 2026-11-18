"""A weld one arm cannot reach is a REALLOCATION, not a verdict on the cell.

``run_mrta_pipeline`` distributes the work from geometry alone — approach-vector
alignment and distance from each base — and it runs BEFORE any IK or collision
check. So the allocation is a first guess. When the planner then reports that the
assigned arm cannot hold the leaned pose at that point without grazing a fixture,
the honest conclusion is that the guess was wrong for that weld, not that the weld
is impossible: a different base reaches the same point from a different direction.

This is not hypothetical. On ``welding_cell_2_small`` the two welds that still fail are
W07 and W08, both allocated to ``robot2``, both failing with an ARM link (not the
tool) against fixture boxes ``FIXTURE_A_v102``/``v104``. Both points sit
2.77 m from ``robot1``'s base, inside its 2.9 m ``workspace_radius`` — so there is
a second arm that could try, and before this the planner had no way to hand the
work over. The recovery ladder ended at "record it unreachable", which is a
weld the cell silently never performs.

``reallocate_operation`` closes that: it ranks the remaining arms with the SAME
policy that made the original allocation, refuses the cases where a handover
cannot help (a locked operation, a point outside every arm's reach, every arm
already tried), and appends the operation to the receiving arm's sequence so
nothing already planned and validated changes its start pose.

Host-only: pure scheduling geometry read from the real cell config. No ROS, no
Pinocchio, no OMPL.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# planning_tools decorates its LLM surface with langchain_core's @tool; the
# allocation math under test is pure numpy. Stub the decorator so this imports on
# a bare host without pulling in the LLM stack.
if "langchain_core.tools" not in sys.modules:
    _lc = types.ModuleType("langchain_core")
    _lct = types.ModuleType("langchain_core.tools")
    _lct.tool = lambda f: f
    sys.modules.setdefault("langchain_core", _lc)
    sys.modules["langchain_core.tools"] = _lct

from agents.trajectory.tools.planning_tools import (  # noqa: E402
    _cell_robots, reallocate_operation,
)
from utils import cell_layout  # noqa: E402
from agents.trajectory.tools.task_allocation import (  # noqa: E402
    _TIE_MARGIN_REL, _regional_allocate, _robot_bases, _robot_scores,
)

CELL = "welding_cell_2_small"
AGENT_SOURCE = PROJECT_ROOT / "agents" / "trajectory" / "agent.py"
TASK = cell_layout.task_file(CELL, "car_door_spot_welding",
                             cell_base=str(PROJECT_ROOT / "cells"))

# The real failing weld, copied from the task file (see the module docstring).
W07 = {"id": "W07", "position": [-1.15, -1.225, 1.35],
       "approach_vector": [0.0, 0.0, -1.0]}


@pytest.fixture(autouse=True)
def cells_on_the_host(monkeypatch):
    """Point the config loaders at the repo's cells/ instead of the container mount."""
    monkeypatch.setenv("CELL_BASE_PATH", str(PROJECT_ROOT / "cells"))


def _call(**kwargs) -> dict:
    """Invoke the tool whether or not langchain wrapped it (host stub vs container)."""
    fn = getattr(reallocate_operation, "func", reallocate_operation)
    return fn(**kwargs)


# ── one allocation policy, two callers ────────────────────────────────────────

def test_the_handover_and_the_original_allocation_share_one_policy():
    """A second "which robot" rule is how the distribution starts lying.

    The allocator picks a winner; the re-allocator picks the runner-up. If those
    came from two implementations they would eventually disagree, and a handover
    would move a weld to an arm the allocator would never have chosen — the same
    N-copies-of-one-decision failure that let the planner and the runtime monitor
    apply different collision models. Both read ``_robot_scores``, so this pins
    that the allocator's choice IS the top of the ranking for every real weld the
    ranking decides — and, where the ranking is a near-tie (inside
    ``_TIE_MARGIN_REL``), which the allocator deliberately settles per GROUP for a
    contiguous boundary, that the owner is still one of the tied arms of that same
    ranking, never an arm the policy ranks clearly worse.
    """
    robots = _cell_robots(CELL)
    bases = _robot_bases(robots)
    operations = json.loads(TASK.read_text())["operations"]

    allocation = _regional_allocate([dict(op) for op in operations], robots)
    owner = {op["id"]: ns for ns, ops in allocation.items() for op in ops}

    assert len(operations) == len(owner), "every operation is allocated exactly once"
    for op in operations:
        ranked = _robot_scores(op, bases)
        best = ranked[0][1]
        tied = {ns for ns, s in ranked
                if (best - s) / max(abs(best), abs(s), 1e-9) < _TIE_MARGIN_REL}
        assert owner[op["id"]] in tied, (
            f"{op['id']}: the allocator and the ranking disagree")
        if len(tied) == 1:
            assert owner[op["id"]] == ranked[0][0], (
                f"{op['id']}: a decisive ranking was overridden")
    # The near-ties are settled as UNITS (the contested row as a whole), not one
    # by one: tied welds of one row never end up split between arms.
    def is_tied(op):
        ranked = _robot_scores(op, bases)
        best, second = ranked[0][1], ranked[1][1]
        return (best - second) / max(abs(best), abs(second), 1e-9) < _TIE_MARGIN_REL
    rows: dict = {}
    for op in operations:
        if is_tied(op):
            rows.setdefault(round(float(op["position"][1]), 3), set()).add(owner[op["id"]])
    split = {y: arms for y, arms in rows.items() if len(arms) > 1}
    assert not split, f"a tied row was split between arms: {split}"
    # And a handover follows the same ranking the allocation used.
    from agents.trajectory.tools.task_allocation import _reallocation_candidates
    for op in operations:
        ranked = [ns for ns, _s in _robot_scores(op, bases)]
        handover = [c["robot_ns"] for c in
                    _reallocation_candidates(op, robots, exclude=[owner[op["id"]]])]
        assert handover == [ns for ns in ranked if ns != owner[op["id"]]], op["id"]


def test_the_ranking_covers_every_arm_not_just_the_winner():
    """Reallocation needs the runner-up, so the policy must rank, not argmax."""
    ranked = _robot_scores(W07, _robot_bases(_cell_robots(CELL)))

    assert [ns for ns, _ in ranked] == ["robot2", "robot1"], (
        "W07 is allocated to robot2 — which is exactly the arm that fails it"
    )
    assert ranked[0][1] > ranked[1][1], "ranked best-first"


# ── the handover ──────────────────────────────────────────────────────────────

def test_a_weld_its_own_arm_cannot_do_is_handed_to_the_other_arm():
    """The case the cell actually has: W07 fails on robot2, robot1 can try it."""
    out = _call(cell_id=CELL, operation=W07, tried_robots=["robot2"], robots=[])

    assert out["success"] is True, out.get("error")
    assert out["to_robot"] == "robot1"
    taker = next(c for c in out["candidates"] if c["robot_ns"] == "robot1")
    assert taker["within_workspace"] is True
    assert taker["distance_from_base_m"] < taker["workspace_radius_m"]


def test_an_arm_that_already_failed_is_never_proposed_again():
    """Otherwise the recovery loop hands the weld back to the arm that refused it."""
    out = _call(cell_id=CELL, operation=W07, tried_robots=["robot2"], robots=[])

    assert "robot2" not in [c["robot_ns"] for c in out["candidates"]]


def test_the_handover_appends_so_nothing_already_planned_is_invalidated():
    """Inserting mid-sequence would move the start pose of every later operation.

    Those trajectories are already planned, collision-checked and chained. An
    insertion silently invalidates all of them, and the chain gate would then
    refuse the tree for reasons that look unrelated to the handover. Appending
    re-plans exactly one operation — the one being moved.
    """
    order = [{"robot_ns": "robot1", "operation_ids": ["W01", "W02", "W03"]}]

    out = _call(cell_id=CELL, operation=W07, tried_robots=["robot2"], robots=order)

    assert out["insert_after_operation_id"] == "W03", "chains from robot1's LAST op"
    assert out["replan_operation_ids"] == ["W07"], "only the moved operation re-plans"


def test_a_handover_to_an_idle_arm_starts_from_home():
    """No predecessor to chain from is a real state, not a missing value."""
    out = _call(cell_id=CELL, operation=W07, tried_robots=["robot2"], robots=[])

    assert out["insert_after_operation_id"] is None
    assert "HOME" in out["instruction"]


# ── the three refusals ────────────────────────────────────────────────────────

def test_a_locked_operation_is_refused_not_reassigned():
    """``assigned_robot`` is a task requirement (part handling, gluing).

    Reallocating it would satisfy the planner by breaking the task. The refusal
    names the remaining honest options instead.
    """
    out = _call(cell_id=CELL, operation=dict(W07, assigned_robot="robot2"),
                tried_robots=["robot2"], robots=[])

    assert out["success"] is False
    assert out["reason_code"] == "robot_locked"
    assert out["assigned_robot"] == "robot2"


def test_unreachable_means_every_arm_has_actually_failed_it():
    """The only path to "unreachable" is exhaustion, and it must say so.

    This is the whole point: before, one arm's failure ended the weld. Now the
    verdict requires evidence from every arm in the cell.
    """
    out = _call(cell_id=CELL, operation=W07,
                tried_robots=["robot1", "robot2"], robots=[])

    assert out["success"] is False
    assert out["reason_code"] == "no_candidate_robot"
    assert out["robots_in_cell"] == ["robot1", "robot2"]
    assert "unreachable" in out["error"]


def test_an_arm_that_cannot_physically_reach_the_point_is_not_offered():
    """Cheap pre-filter: don't spend a planning attempt proving reach.

    The radius comes from each arm's ``workspace_radius`` in robots.yaml — not
    from a table copied into a skill, which is how a config value drifts from the
    prose that quotes it.
    """
    far = {"id": "FAR", "position": [20.0, 20.0, 1.0],
           "approach_vector": [0.0, 0.0, -1.0]}

    out = _call(cell_id=CELL, operation=far, tried_robots=["robot2"], robots=[])

    assert out["success"] is False
    assert out["reason_code"] == "out_of_reach_everywhere"
    assert all(c["within_workspace"] is False for c in out["candidates"])


# ── the capability is reachable from the agent ────────────────────────────────

def test_the_agent_exposes_the_tool_and_is_told_to_use_it():
    """A tool the LLM cannot call, or is not told to call, changes nothing.

    The planner had "try the other robot" as prose in two skills for a long time
    and still dropped W07/W08, because there was no tool behind the advice and the
    prompt's own recovery list ended at "EXCLUDE that weld". Both halves have to
    hold: the tool is in the toolset, and the prompt orders reallocation BEFORE
    exclusion. Read from source so this needs no LLM stack.
    """
    source = AGENT_SOURCE.read_text()
    prompt = source.split('SYSTEM_PROMPT = f"""', 1)[1].split('"""', 1)[0]

    assert "reallocate_operation," in source.split("_TOOLS = [", 1)[1], \
        "registered in the agent's toolset"
    assert "reallocate_operation" in prompt, "the prompt names the tool"
    assert prompt.index("REDISTRIBUTE") < prompt.index("UNREACHABLE"), \
        "redistribution is ordered before declaring a weld unreachable"
