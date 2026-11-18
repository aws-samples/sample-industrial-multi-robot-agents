"""The home reset must clear the SAME gate every other multi-arm step clears.

``home_reset_tree`` planned the returns (``test_home_reset_planned``) and then
wrapped whatever it got in a ``Parallel`` — trusting the planner's coupling
instead of verifying the set. Two arms therefore moved at once on a decision
nobody made:

* An arm the planner could not route kept a bare ``MoveHome``, and the executor
  drives that as a single HOME waypoint — a straight line in joint space through
  whatever is in the way. Measured on ``welding_cell_2_small`` out of the pose a prior
  e-stop left the cell in (robot1 ``[-2.1956,-1.4599,1.3817,3.1797,-1.6873,
  -2.1900]``, robot2 ``[1.1381,-1.0962,0.7677,-1.9782,-0.8277,1.6803]``): robot1's
  blind return, with robot2 standing where it is, closes to **-0.1063 m** — the
  arms overlap. The same set one arm at a time, robot2 first, never drops below
  the 0.1385 m the arms already stand at, and reaches 1.4601 m once robot2 is
  home. A safe order existed; nothing looked for it.

* The pair was never measured at all, so a breach could only be discovered by the
  runtime monitor — i.e. by an e-stop, before the validated plan even started.

So the set goes through ``_gate_multi_arm_step``, the one owner of "may these arms
move at once?", and a blind arm is handed to it as the straight line the executor
will actually drive. Its verdict decides the node: clear ⇒ ``Parallel``; not clear
⇒ ``Sequence`` in an order where each mover clears the others where they stand; no
safe order ⇒ those arms do not move and the operator is told.

What must NOT change: a planned arm still carries its planned path, an unplanned
arm still carries no ``trajectory`` port (so the executor uses its documented
fallback), and a gate that cannot run must not refuse anything
(``test_home_reset_planned`` pins that shape).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"
ROBOTS = ["robot1", "robot2"]


def _traj(n: int, start: float = 0.0) -> list:
    return [{"positions": [start + i * 0.01] * 6, "time_s": i * 0.1} for i in range(n)]


@pytest.fixture
def home_reset():
    return pytest.importorskip(
        "agents.cell.home_reset", reason="needs the cell runtime")


@pytest.fixture
def harness(home_reset, monkeypatch):
    """Stub live state, the home planner, and the gate; record every gate call.

    The gate is stubbed rather than run for real so the ladder's behaviour is
    pinned independently of any one cell's geometry — the geometry itself is
    covered by ``test_home_planning_coupled``.
    """
    import agents.trajectory.tools.ompl_planner as ompl

    calls: list = []

    def install(states: dict, planned, verdict=None):
        monkeypatch.setattr(
            home_reset, "get_all_robot_states",
            SimpleNamespace(invoke=lambda _args: {"robots": states}))

        def fake_plan(cell_id, starts, home_by_robot=None, velocity_scale=0.4):
            return planned(starts) if callable(planned) else (planned or {})

        monkeypatch.setattr(ompl, "plan_home_trajectories", fake_plan)

        def fake_gate(cell_id, legs, *, allow_parallel=True, standing=None):
            calls.append({"cell_id": cell_id, "legs": legs,
                          "arms": sorted(legs), "allow_parallel": allow_parallel})
            report = {"arms": sorted(legs)}
            report.update(verdict(legs) if callable(verdict) else
                          (verdict or {"mode": "PARALLEL", "gated": True}))
            return report

        monkeypatch.setattr(home_reset, "_gate_multi_arm_step", fake_gate,
                            raising=False)
        return calls

    return install


def _children(tree: dict) -> list:
    return tree["root"]["children"]


def _by_arm(tree: dict) -> dict:
    return {c["ports"]["robot_ns"]: c for c in _children(tree)}


# ── the set is measured, not assumed ─────────────────────────────────────────

def test_the_pair_is_gated_before_the_arms_are_driven_home(home_reset, harness):
    """A reset that skips the gate is a two-arm motion nobody checked."""
    calls = harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
                    planned={r: _traj(10) for r in ROBOTS})

    home_reset.home_reset_tree(CELL, ROBOTS)

    assert len(calls) == 1, "the multi-arm set must be gated exactly once"
    assert calls[0]["arms"] == ROBOTS
    assert calls[0]["cell_id"] == CELL


def test_an_unplanned_arm_is_gated_as_the_line_the_executor_will_drive(
        home_reset, harness):
    """The blind fallback is a KNOWN motion — a straight line to HOME.

    Leaving it out of the gate is what let robot1's blind return reach -0.1063 m
    against a standing robot2. It is predictable, so it is checkable.
    """
    calls = harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
                    planned={"robot1": _traj(12)})

    home_reset.home_reset_tree(CELL, ROBOTS)

    assert calls, "the gate must still run when an arm could not be planned"
    legs = calls[0]["legs"]
    assert sorted(legs) == ROBOTS, (
        "an arm with no planned path still MOVES — it must be gated as the "
        "straight line the executor drives, not omitted from the check"
    )
    blind = legs["robot2"]
    assert len(blind) >= 2, "a straight line needs at least a start and an end"
    assert blind[-1]["time_s"] > blind[0]["time_s"]


def test_a_gate_leg_for_a_blind_arm_is_never_emitted_as_a_path(home_reset, harness):
    """Checking a motion must not silently turn it into a planned one.

    ``test_home_reset_planned`` pins that an unplanned arm carries no
    ``trajectory`` port, so the executor uses its documented fallback.
    """
    harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
            planned={"robot1": _traj(12)})

    nodes = _by_arm(home_reset.home_reset_tree(CELL, ROBOTS))

    assert nodes["robot1"]["ports"]["trajectory"]
    assert "trajectory" not in nodes["robot2"]["ports"]


# ── the verdict decides the node ──────────────────────────────────────────────

def test_a_clear_pair_still_goes_home_together(home_reset, harness):
    """Serialising a safe reset would cost cycle time for nothing."""
    harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
            planned={r: _traj(10) for r in ROBOTS},
            verdict={"mode": "PARALLEL", "gated": True})

    tree = home_reset.home_reset_tree(CELL, ROBOTS)

    assert tree["root"]["type"] == "Parallel"
    assert sorted(c["ports"]["robot_ns"] for c in _children(tree)) == ROBOTS


def test_a_breaching_pair_is_serialised_in_the_gates_order(home_reset, harness):
    """The order is the gate's answer — out of an e-stop pose robot2 must go first."""
    harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
            planned={r: _traj(10) for r in ROBOTS},
            verdict={"mode": "SEQUENTIAL", "gated": True,
                     "order": ["robot2", "robot1"]})

    tree = home_reset.home_reset_tree(CELL, ROBOTS)

    assert tree["root"]["type"] == "Sequence"
    assert [c["ports"]["robot_ns"] for c in _children(tree)] == ["robot2", "robot1"]


def test_an_arm_the_gate_refuses_does_not_move(home_reset, harness):
    """No safe order for an arm means the reset must not drive it anyway.

    Emitting it regardless is the fail-OPEN choice the blind fallback already
    made once; the monitor would e-stop, which is the failure being fixed.
    """
    harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
            planned={r: _traj(10) for r in ROBOTS},
            verdict={"mode": None, "gated": True, "order": ["robot2"],
                     "blocked_arms": ["robot1"]})

    tree = home_reset.home_reset_tree(CELL, ROBOTS)

    assert [c["ports"]["robot_ns"] for c in _children(tree)] == ["robot2"], (
        "only the arms the gate cleared may be driven"
    )


def test_a_refusal_is_reported_not_swallowed(home_reset, harness, capsys):
    """An operator has to know an arm was left where it stands."""
    harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
            planned={r: _traj(10) for r in ROBOTS},
            verdict={"mode": None, "gated": True, "order": [],
                     "blocked_arms": ROBOTS})

    tree = home_reset.home_reset_tree(CELL, ROBOTS)
    out = capsys.readouterr().out.lower()

    assert _children(tree) == []
    assert "robot1" in out and "robot2" in out
    assert "not" in out or "refus" in out or "blocked" in out


# ── degradation stays permitted ───────────────────────────────────────────────

def test_a_gate_that_cannot_run_refuses_nothing(home_reset, harness):
    """Without geometry the gate has no opinion — it must not block the reset.

    ``_gate_multi_arm_step`` reports ``gated: False`` with ``mode: PARALLEL``
    when it could not measure; the reset then behaves exactly as it did before
    the gate existed.
    """
    harness(states={r: {"joint_positions": [0.1] * 6} for r in ROBOTS},
            planned={r: _traj(10) for r in ROBOTS},
            verdict={"mode": "PARALLEL", "gated": False})

    tree = home_reset.home_reset_tree(CELL, ROBOTS)

    assert tree["root"]["type"] == "Parallel"
    assert len(_children(tree)) == len(ROBOTS)


def test_a_single_arm_reset_is_not_gated(home_reset, harness):
    """There is no pair to check, and the gate is expensive geometry."""
    calls = harness(states={"robot1": {"joint_positions": [0.1] * 6}},
                    planned={"robot1": _traj(10)})

    tree = home_reset.home_reset_tree(CELL, ["robot1"])

    assert calls == []
    assert [c["ports"]["robot_ns"] for c in _children(tree)] == ["robot1"]


def test_an_arm_with_no_live_pose_cannot_be_gated_or_blindly_driven(
        home_reset, harness):
    """Its blind line is unknown, so nothing about its motion can be checked.

    Planning already refuses to guess a start pose
    (``test_missing_live_state_does_not_invent_a_start_pose``); the gate must not
    invent one either, and an unmeasurable arm must not be paired with a mover.
    """
    calls = harness(states={"robot1": {"joint_positions": [0.2] * 6}, "robot2": {}},
                    planned=lambda starts: {ns: _traj(8) for ns in starts})

    tree = home_reset.home_reset_tree(CELL, ROBOTS)

    assert tree["root"]["type"] == "Sequence", (
        "an arm whose motion cannot be predicted must not move alongside another"
    )
    assert not any(c["ports"]["robot_ns"] == "robot2" and c["ports"].get("trajectory")
                   for c in _children(tree))
    for c in calls:
        assert "robot2" not in c["legs"], "a guessed leg is not a measurement"
