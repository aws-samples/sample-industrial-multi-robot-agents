"""Arms that conflict in a PARALLEL step are separated in TIME, not serialised.

Two things are pinned here:

  * ``find_start_delays`` — the smallest stagger that clears a set of legs, for
    two OR MORE arms, robust to a ±0.25 s start error, and ``None`` when no
    stagger short of "run afterwards" helps (a pairing problem, not a timing one).
  * the assembler's gate checks a step on the timeline it EXECUTES: an arm's
    ``start_delay_s`` shifts its leg. Checking a staggered step as if both arms
    started together passes, or refuses, a step the robots never run.

Host-only: the geometry is replaced by a time-overlap model — two arms "collide"
whenever both are moving at the same instant, and the moving window of each leg
is exactly its time span.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"


def _leg(t0: float, t1: float, q: float = 0.0) -> list[dict]:
    return [{"positions": [q] * 6, "time_s": t0},
            {"positions": [q + 0.1] * 6, "time_s": (t0 + t1) / 2},
            {"positions": [q + 0.2] * 6, "time_s": t1}]


def _window(traj: list[dict]) -> tuple[float, float]:
    return float(traj[0]["time_s"]), float(traj[-1]["time_s"])


def _overlap_clearance(conflict_s: float):
    """Pairs collide while both windows overlap by more than ``conflict_s``."""
    def clearance(cell_id, trajectories, **kw):
        names = sorted(trajectories)
        worst = 1.0
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                (a0, a1), (b0, b1) = _window(trajectories[a]), _window(trajectories[b])
                overlap = min(a1, b1) - max(a0, b0)
                if overlap > conflict_s:
                    worst = min(worst, 0.1)
        return {"collision_free": worst > 0.2, "min_clearance_m": worst,
                "warning_distance_m": 0.2, "robots": names}
    return clearance


@pytest.fixture
def pt(monkeypatch):
    def _tool(fn=None, **_kw):
        def _wrap(f):
            f.func = f
            f.invoke = lambda d: f(**d)
            return f
        return _wrap(fn) if fn is not None else _wrap

    cached = sys.modules.get("langchain_core.tools")
    if cached is None:
        try:
            import langchain_core.tools as cached  # noqa: F401
        except Exception:
            cached = None
    if getattr(cached, "__file__", None) is None:
        lc = sys.modules.get("langchain_core")
        if getattr(lc, "__file__", None) is None:
            lc = types.ModuleType("langchain_core")
            lc.__path__ = []
            monkeypatch.setitem(sys.modules, "langchain_core", lc)
        lct = types.ModuleType("langchain_core.tools")
        lct.tool = _tool
        monkeypatch.setattr(lc, "tools", lct, raising=False)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", lct)
    for name in [n for n in list(sys.modules) if n.startswith("agents.trajectory.tools")]:
        del sys.modules[name]
    import importlib
    module = importlib.import_module("agents.trajectory.tools.planning_tools")
    yield module
    for name in [n for n in list(sys.modules) if n.startswith("agents.trajectory.tools")]:
        del sys.modules[name]


def test_a_delay_shifts_the_leg_and_keeps_its_shape(pt):
    leg = _leg(0.0, 2.0)
    out = pt.delayed(leg, 1.5)
    assert [p["time_s"] for p in out] == [1.5, 2.5, 3.5]
    assert [p["positions"] for p in out] == [p["positions"] for p in leg]
    assert pt.delayed(leg, 0.0) is leg


def test_two_conflicting_arms_get_the_smallest_stagger(pt, monkeypatch):
    # Both legs span 0-4 s; allowed overlap 1 s ⇒ one arm must start ≥ 3 s later
    # (plus robustness: 3.25 s so that 3.0 s is also clear).
    monkeypatch.setattr(pt, "inter_robot_clearance", _overlap_clearance(1.0))
    legs = {"robot1": _leg(0.0, 4.0), "robot2": _leg(0.0, 4.0, 1.0)}
    fix = pt.find_start_delays(CELL, legs, warning_distance_m=0.2)
    assert fix is not None
    delays = fix["start_delays"]
    assert sorted(delays.values()) == [0.0, 3.25]
    assert fix["total_delay_s"] == 3.25


def test_the_shorter_leg_is_the_one_delayed_when_cheaper(pt, monkeypatch):
    monkeypatch.setattr(pt, "inter_robot_clearance", _overlap_clearance(0.5))
    legs = {"robot1": _leg(0.0, 6.0), "robot2": _leg(0.0, 1.0, 1.0)}
    fix = pt.find_start_delays(CELL, legs, warning_distance_m=0.2)
    # robot2 (1 s long) waiting for robot1 would cost ~5.75 s; robot1 waiting
    # for robot2 costs ~0.75 s — the search must take the cheaper order.
    assert fix["start_delays"]["robot2"] == 0.0
    assert fix["start_delays"]["robot1"] < 1.5


def test_three_arms_are_staggered_together(pt, monkeypatch):
    monkeypatch.setattr(pt, "inter_robot_clearance", _overlap_clearance(0.5))
    legs = {f"robot{i}": _leg(0.0, 2.0, float(i)) for i in (1, 2, 3)}
    fix = pt.find_start_delays(CELL, legs, warning_distance_m=0.2)
    assert fix is not None
    starts = sorted(fix["start_delays"].values())
    assert starts[0] == 0.0 and starts[1] >= 1.5 and starts[2] >= starts[1] + 1.5


def test_no_stagger_short_of_running_afterwards_returns_none(pt, monkeypatch):
    def always(cell_id, trajectories, **kw):
        return {"collision_free": False, "min_clearance_m": 0.05,
                "warning_distance_m": 0.2, "robots": sorted(trajectories)}
    monkeypatch.setattr(pt, "inter_robot_clearance", always)
    legs = {"robot1": _leg(0.0, 2.0), "robot2": _leg(0.0, 2.0, 1.0)}
    assert pt.find_start_delays(CELL, legs, warning_distance_m=0.2) is None


def test_a_colliding_set_is_offered_its_delay_fix(pt, monkeypatch):
    monkeypatch.setattr(pt, "inter_robot_clearance", _overlap_clearance(1.0))
    monkeypatch.setattr(pt, "_load_collision_config",
                        lambda c: {"warning_distance": 0.2, "min_clearance": 0.05})
    legs = {"robot1": _leg(0.0, 4.0), "robot2": _leg(0.0, 4.0, 1.0)}
    # The legs are synthetic joint values (inside the floor of the real cell, and
    # past its A2 limit): this tests the arm-vs-arm delay logic only, so the static
    # and joint-limit checks are off.
    out = pt.validate_trajectory_set.func(cell_id=CELL, trajectories=legs,
                                          check_static_obstacles=False,
                                          check_joint_limits=False)
    assert out["collision_free"] is False
    assert out["delay_fix"]["total_delay_s"] == 3.25
    assert "start_delay_s" in out["suggestion"]
    # and validating WITH those delays is clear
    again = pt.validate_trajectory_set.func(
        cell_id=CELL, trajectories=legs,
        start_delays=out["delay_fix"]["start_delays"], check_static_obstacles=False,
        check_joint_limits=False)
    assert again["collision_free"] is True


def test_the_gate_checks_the_step_on_its_staggered_timeline(pt, monkeypatch):
    import importlib
    bt = importlib.import_module("agents.trajectory.tools.behaviour_tree")
    seen = []

    def fake(cell_id, trajectories, **kw):
        seen.append({ns: tr[0]["time_s"] for ns, tr in trajectories.items()})
        return {"collision_free": True, "min_clearance_m": 0.5,
                "warning_distance_m": 0.2, "robots": sorted(trajectories)}
    monkeypatch.setattr(pt, "validate_trajectory_set", fake)
    legs = {"robot1": _leg(0.0, 2.0), "robot2": _leg(0.0, 2.0, 1.0)}
    report = bt._gate_multi_arm_step(CELL, legs, delays={"robot2": 1.5})
    assert report["mode"] == "PARALLEL"
    assert seen[0] == {"robot1": 0.0, "robot2": 1.5}
    assert report["start_delays"] == {"robot2": 1.5}
