"""A coordinated return home must not arc over the cell to buy clearance it has.

The home ladder plans each arm against the SWEPT TUBE of the arms already planned:
every place they will ever be, with no reference to when. That is time-blind, while
the gate the runtime monitor implements is time-aware — so the ladder can refuse a
route the cell would happily have run.

Measured on ``bt_1787955158`` / ``welding_cell_2_small``, that is exactly what happened:
``robot2``'s return found NO path at the full margin, the ladder fell back through
shrinking margins, and the plan shipped a **0.585 m** arc (peak tool z 2.854 m) —
while the direct route rises **0.000 m** and clears the real gate at **0.5314 m**
against a 0.2 m warning band. The operator's complaint ("the return of one of the
arms is going up too much, without a clear reason") was correct, and the reason was
that the planner was solving a harder problem than the one safety requires.

These tests pin BOTH halves, which have to hold together:

  * the returned legs stay flat, and
  * they still clear the gate.

Either alone is trivially satisfiable and worthless — a flat leg that collides is
worse than an arc, so the height assertions are only meaningful next to the
clearance ones.

Real geometry, real collision model (link spheres plus the welding torch), real
``warning_distance``. Needs Pinocchio and OMPL, so it skips on a bare host and runs
in the container: the property is geometric, and a mocked geometry would pin
nothing. Planning is slow, so each scenario is planned once in a module-scoped
fixture. Marked ``slow`` — deselect with ``-m 'not slow'``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"

pytestmark = pytest.mark.slow

pytest.importorskip("pinocchio", reason="needs the ROS/Pinocchio runtime")
pytest.importorskip("numpy")

# The poses ``bt_1787955158`` left the arms in at its last weld — the exact state
# the 0.585 m arc was planned FROM. Read off the first point of each arm's MoveHome
# trajectory in that saved plan, not invented: the height depends on where the arms
# start, so a plausible-looking pose would pin a different scenario.
LAST_WELD_Q = {
    "robot1": [-2.371829, -0.9233, 0.918403, 0.611254, 2.119607, 0.952785],
    "robot2": [1.028917, -1.389619, 1.354952, 0.717472, 1.98089, -1.940387],
}


@pytest.fixture(scope="module")
def ompl_available():
    from agents.trajectory.backends.ompl.planner import _import_ompl
    if _import_ompl() is None:
        pytest.skip("OMPL not installed")


@pytest.fixture(scope="module")
def homed(ompl_available):
    from agents.trajectory.backends.ompl.planner import plan_home_trajectories
    return plan_home_trajectories(CELL, LAST_WELD_Q)


def _rise(ns: str, traj: list) -> float:
    import numpy as np
    from agents.trajectory.backends.ompl.planner import _trajectory_tool_rise
    from agents.trajectory.tools.robot_kinematics import (
        _load_pinocchio_model, _load_robot_config)
    model, _ = _load_pinocchio_model(CELL, ns)
    rcfg = _load_robot_config(CELL, ns)
    return _trajectory_tool_rise(
        model, traj, np.array(rcfg["base_position"]),
        float(rcfg.get("base_orientation_yaw", 0.0)))


def test_every_arm_gets_a_planned_return(homed):
    """A missing leg is handed to the executor's blind straight line — the one
    motion nothing checks. The relaxation rung must widen what can be planned."""
    assert set(homed) == set(LAST_WELD_Q)
    assert all(homed[ns] for ns in homed)


def test_no_return_leg_arcs_over_the_cell(homed):
    """The complaint, measured. 0.585 m shipped before; the limit refuses that."""
    from agents.trajectory.backends.ompl.planner import _emitted_rise_limit
    limit = _emitted_rise_limit()
    tall = {ns: round(_rise(ns, t), 3) for ns, t in homed.items()
            if _rise(ns, t) > limit}
    assert not tall, f"legs rising above {limit} m: {tall}"


def test_the_flat_returns_still_clear_the_gate(homed):
    """Flatness bought by moving the arms closer together would be a regression,
    not a fix. Judged by the gate the returns are EXECUTED through
    (``_gate_multi_arm_step``: in parallel when that is clear, else one arm after
    another in a proven order) — the planner admits a set on that verdict, so a
    parallel-only check would fail sets execution never runs in parallel."""
    from agents.trajectory.tools.behaviour_tree import _gate_multi_arm_step
    report = _gate_multi_arm_step(CELL, homed)
    par = report.get("parallel") or {}
    assert report.get("mode") is not None, (
        f"no order clears the returns: parallel clearance {par.get('min_clearance_m')} m, "
        f"band {par.get('warning_distance_m')} m")


def test_the_gate_is_checked_at_the_monitors_resolution(homed):
    """A coarse check aliases over the dip; the floor from the resolution fix holds."""
    from agents.trajectory.tools.robot_kinematics import (
        inter_robot_clearance, validation_sample_count)
    verdict = inter_robot_clearance(CELL, homed)
    horizon = max(float(t[-1]["time_s"]) for t in homed.values())
    assert verdict["samples_checked"] == validation_sample_count(horizon, floor=20)


def _force_the_rung(monkeypatch, verdict: dict) -> dict:
    """Make every arm look tall so the rung is reached, and fix the gate's answer.

    Forcing the rise rather than tuning the limit matters: RRTConnect is unseeded,
    so a run where the ladder happens to find a 0.000 m route would skip the rung
    entirely and the test would pass without exercising anything.

    The rung compares the two routes' heights, so a constant fake rise would make
    them tie and the comparison would reject the candidate for the wrong reason.
    Instead the relaxed calls — the ones with no inter-arm obstacles — are tagged as
    they come back, and the fake rise reads the tag: tube route "arcs", direct route
    is "flat". The tag is also what makes "did the direct route ship?" observable.
    """
    from agents.trajectory.backends.ompl import planner as op

    log: dict = {"gate_calls": 0, "relaxed": []}

    def refuse_or_allow(cell_id, trajectories, **kw):
        log["gate_calls"] += 1
        return dict(verdict)

    # The rung asks the step gate about ``{**already_planned, arm_under_test:
    # candidate}``, so the arm being adjudicated is the last key of THAT call (the
    # gate itself then tries the arms in other orders, which says nothing about
    # who is adjudicated). Recorded because the FIRST arm planned has nobody to
    # clear and never reaches the rung — an assertion that counted it would pass
    # without testing the relaxation.
    from agents.trajectory.tools import behaviour_tree as bt
    real_gate = bt._gate_multi_arm_step

    def adjudicate(cell_id, legs, *a, **kw):
        log.setdefault("gated", []).append(list(legs)[-1])
        return real_gate(cell_id, legs, *a, **kw)

    monkeypatch.setattr(bt, "_gate_multi_arm_step", adjudicate)

    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics.inter_robot_clearance",
        refuse_or_allow)
    # The rung asks the step gate (_gate_multi_arm_step -> validate_trajectory_set),
    # which holds its own reference to the owner.
    monkeypatch.setattr(
        "agents.trajectory.tools.planning_tools.inter_robot_clearance",
        refuse_or_allow)

    def fake_rise(model, traj, base, yaw):
        if any(p.get("_relaxed") for p in (traj or [])):
            return 0.0
        return 99.0

    monkeypatch.setattr(op, "_trajectory_tool_rise", fake_rise)

    real_plan = op.plan_joint_path

    def spy(cell_id, ns, start_q, goal_q, moving_obstacle_spheres=None, **kw):
        t = real_plan(cell_id, ns, start_q, goal_q,
                      moving_obstacle_spheres=moving_obstacle_spheres, **kw)
        if not moving_obstacle_spheres and t:
            t = [{**p, "_relaxed": True} for p in t]
            log["relaxed"].append((ns, [list(p["positions"]) for p in t]))
        return t

    monkeypatch.setattr(op, "plan_joint_path", spy)
    return log


def _ships(shipped: list, candidate_positions: list) -> bool:
    """Is ``candidate_positions`` the tail of ``shipped``?

    A shipped leg is ``retract + transit`` concatenated, and ``_concat_timed`` drops
    the duplicated junction point, so the transit appears as ``candidate[1:]``.
    Positions only — the concatenation legitimately rewrites ``time_s``.
    """
    tail = [list(p["positions"]) for p in shipped]
    want = [list(p) for p in candidate_positions][1:]
    return len(want) > 0 and tail[-len(want):] == want


def test_a_refused_relaxation_does_not_ship_but_still_returns_a_leg(
        monkeypatch, ompl_available):
    """Fail-closed: the relaxation widens what can be PLANNED, never what may move.

    With the verdict forced to refuse, the direct route must not appear in the
    result — and the arm must still come home by some other rung, because handing
    it to the executor's blind straight line would be strictly more dangerous.
    """
    from agents.trajectory.backends.ompl.planner import plan_home_trajectories

    log = _force_the_rung(monkeypatch, {
        "collision_free": False, "min_clearance_m": 0.0,
        "warning_distance_m": 0.2, "collision_time_s": 0.0})
    refused = plan_home_trajectories(CELL, LAST_WELD_Q)

    assert log["gate_calls"] >= 1, "the rung must consult the gate, not decide alone"
    assert refused, "a refused relaxation must fall back, not return nothing"
    gated = set(log.get("gated") or [])
    for ns, positions in log["relaxed"]:
        if ns in gated and ns in refused:
            assert not _ships(refused[ns], positions), (
                f"{ns} shipped the direct route the gate refused")


def test_an_accepted_relaxation_is_the_route_that_ships(monkeypatch, ompl_available):
    """The other half: a gate PASS is what lets the flat route through.

    Without this, the refusal test above would also pass on an implementation that
    ignored the relaxation entirely.
    """
    from agents.trajectory.backends.ompl.planner import plan_home_trajectories

    log = _force_the_rung(monkeypatch, {
        "collision_free": True, "min_clearance_m": 0.53,
        "warning_distance_m": 0.2})
    allowed = plan_home_trajectories(CELL, LAST_WELD_Q)

    assert log["gate_calls"] >= 1
    gated = set(log.get("gated") or [])
    shipped_direct = [ns for ns, positions in log["relaxed"]
                      if ns in gated and ns in allowed
                      and _ships(allowed[ns], positions)]
    assert shipped_direct, (
        "the gate said yes and no adjudicated arm used the direct route: "
        f"gated {sorted(gated)}, result {sorted(allowed)}")
