"""Arms follow their own paths and are kept apart by ORDER (gates), not detours.

The conflict model is injected: a set of (arm, kind, op) item pairs that clash.
What is pinned is the coordination contract — arms run in parallel wherever
nothing conflicts, an arm that must wait holds where it stands, every clash is
ordered by a completion gate on the arm that goes second, an arm standing in the
way is named for a safe spot, and conflicting items never overlap in time.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools.coordination import HOME, schedule  # noqa: E402


def _prog(arm_ops: dict, dur: float = 2.0):
    return {a: (ops, {o: dur for o in ops}) for a, ops in arm_ops.items()}


def _clash(pairs):
    bad = {frozenset(p) for p in pairs}

    def conflict(x, y):
        return frozenset(((x.arm, x.kind, x.op_id), (y.arm, y.kind, y.op_id))) in bad
    return conflict


def test_arms_with_no_conflict_run_fully_parallel():
    res = schedule(_prog({"r1": ["A1", "A2"], "r2": ["B1", "B2"]}), _clash([]))
    assert res["deadlock"] is None
    assert res["makespan_s"] == 4.0 and res["parallel_s"] == 4.0
    assert all(not p.gates for p in res["arms"].values())


def test_conflicting_moves_are_ordered_by_a_completion_gate():
    res = schedule(_prog({"r1": ["A1", "A2"], "r2": ["B1", "B2"]}),
                   _clash([(("r1", "move", "A2"), ("r2", "move", "B2"))]))
    assert res["deadlock"] is None
    gated = {a: p.gates for a, p in res["arms"].items() if p.gates}
    assert len(gated) == 1
    (arm, gates), = gated.items()
    other = "r2" if arm == "r1" else "r1"
    op = "A2" if arm == "r1" else "B2"
    assert gates[op] == {other: ("B2" if other == "r2" else "A2")}
    assert res["makespan_s"] == 6.0          # one arm waited 2 s — not serialised
    assert res["parallel_s"] == 2.0


def test_an_arm_standing_in_the_way_is_waited_for_until_it_moves_on():
    # r1 standing at A1 blocks r2's move to B1; r2 waits until r1 has done A2.
    res = schedule(_prog({"r1": ["A1", "A2"], "r2": ["B1"]}),
                   _clash([(("r1", "stand", "A1"), ("r2", "move", "B1"))]))
    assert res["deadlock"] is None
    assert res["arms"]["r2"].gates["B1"] == {"r1": "A2"}
    assert res["arms"]["r2"].starts["B1"] >= 4.0


def test_an_arm_parked_for_good_in_the_way_needs_a_safe_spot():
    res = schedule(_prog({"r1": ["A1"], "r2": ["B1", "B2"]}),
                   _clash([(("r1", "stand", "A1"), ("r2", "move", "B2"))]))
    dl = res["deadlock"]
    assert dl is not None
    assert dl["arm"] == "r2" and dl["waiting_to_start"] == "B2"
    assert dl["needs_safe_spot"] == {"arm": "r1", "after_op": "A1"}


def test_three_arms_share_the_cell():
    res = schedule(_prog({"r1": ["A1", "A2"], "r2": ["B1", "B2"], "r3": ["C1", "C2"]}),
                   _clash([(("r1", "move", "A1"), ("r3", "move", "C1"))]))
    assert res["deadlock"] is None
    assert res["parallel_s"] >= 4.0
    assert sum(1 for p in res["arms"].values() if p.gates) == 1


def test_home_is_where_every_arm_starts():
    seen = []

    def conflict(x, y):
        seen.append((x.op_id, y.op_id))
        return False
    schedule(_prog({"r1": ["A1"], "r2": ["B1"]}), conflict)
    assert any(HOME in pair for pair in seen)


def _intervals(res, programs):
    """Simulated (start, end) of every move and stand, per (arm, kind, op)."""
    out = {}
    for arm, p in res["arms"].items():
        ops = programs[arm][0]
        prev_end = 0.0
        prev = HOME
        for op in ops:
            if op not in p.starts:
                break
            s = p.starts[op]
            e = s + programs[arm][1][op]
            out[(arm, "stand", prev)] = (prev_end, e)   # stand held until next move ends
            out[(arm, "move", op)] = (s, e)
            prev_end, prev = e, op
        out[(arm, "stand", prev)] = (prev_end, float("inf"))
    return out


def test_conflicting_items_never_overlap_in_time():
    rng = random.Random(7)
    for trial in range(60):
        arms = [f"r{i}" for i in range(rng.choice([2, 3]))]
        progs = {a: [f"{a}_{k}" for k in range(rng.randint(1, 4))] for a in arms}
        items = [(a, kind, op) for a in arms
                 for op in [HOME] + progs[a] for kind in ("move", "stand")
                 if not (op == HOME and kind == "move")]
        # HOME poses are clear of each other by definition: every arm starts there.
        pairs = [(x, y) for x in items for y in items
                 if x[0] < y[0] and not (x[2] == HOME and y[2] == HOME)
                 and rng.random() < 0.12]
        programs = _prog(progs, dur=rng.choice([1.0, 2.0, 3.0]))
        res = schedule(programs, _clash(pairs))
        if res["deadlock"] is not None:
            continue
        iv = _intervals(res, programs)
        for x, y in pairs:
            if x in iv and y in iv:
                (s1, e1), (s2, e2) = iv[x], iv[y]
                overlap = min(e1, e2) - max(s1, s2)
                assert overlap <= 1e-9, (trial, x, y, iv[x], iv[y])
