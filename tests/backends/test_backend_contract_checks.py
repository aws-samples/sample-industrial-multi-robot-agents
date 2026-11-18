"""Every backend is held to the result contract by the facade, and builds its legs
from the shared ``leg_builder`` rather than another backend's internals."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def facade(monkeypatch):
    from agents.trajectory.tools import planner_backend as pb, trajectory_cache as tc
    tc.reset_robot_configs("c")
    monkeypatch.setattr(pb, "_self_collision_refusals", lambda c, r: [])

    def install(legs, commit=None):
        def backend(cell, reqs, **k):
            for ns, q in (commit or {}).items():
                tc.set_last_config(cell, ns, q)
            return {"success": True, "trajectories": legs, "failures": [],
                    "planning_time_s": 0.0}
        monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: backend)
        return pb, tc
    yield install
    tc.reset_robot_configs("c")


def _leg(start, end, t=(0.0, 1.0)):
    return [{"positions": list(start), "time_s": t[0]},
            {"positions": list(end), "time_s": t[1]}]


def test_a_leg_that_starts_where_the_arm_stands_ships(facade):
    pb, tc = facade({"robot1": _leg([0.2] * 6, [0.3] * 6)})
    tc.set_last_config("c", "robot1", [0.2] * 6)
    out = pb.dispatch_plan("c", [{"robot_ns": "robot1", "operation_id": "W1"}])
    assert "robot1" in out["trajectories"], out["failures"]


def test_a_leg_from_elsewhere_is_refused_and_its_chaining_taken_back(facade):
    pb, tc = facade({"robot1": _leg([0.9] * 6, [1.0] * 6)}, commit={"robot1": [1.0] * 6})
    tc.set_last_config("c", "robot1", [0.2] * 6)
    out = pb.dispatch_plan("c", [{"robot_ns": "robot1", "operation_id": "W1"}])
    assert out["trajectories"] == {}
    f, = [f for f in out["failures"] if f.get("operation_id") == "W1"]
    assert f["reason_code"] == "backend_contract_violation" and "starts" in f["error"]
    assert f["recovery"]["escalate"] == "stop"
    assert tc.get_last_config("c", "robot1") == [0.2] * 6


def test_current_joints_is_where_the_arm_stands(facade):
    pb, tc = facade({"robot1": _leg([0.5] * 6, [0.6] * 6)})
    tc.set_last_config("c", "robot1", [0.2] * 6)
    out = pb.dispatch_plan("c", [{"robot_ns": "robot1", "operation_id": "W1",
                                  "current_joints": [0.5] * 6}])
    assert "robot1" in out["trajectories"], out["failures"]


@pytest.mark.parametrize("leg, why", [
    ({"positions": [0.2] * 6}, "not a list"),
    ([{"positions": [0.2] * 6, "time_s": 1.0}, {"positions": [0.3] * 6, "time_s": 0.5}],
     "backwards"),
    ([{"positions": [0.2] * 6, "time_s": 0.0}, {"positions": [0.3] * 5, "time_s": 1.0}],
     "joints"),
    ([{"positions": [0.2] * 6, "time_s": 0.0}, {"positions": [float("nan")] * 6, "time_s": 1.0}],
     "non-finite"),
])
def test_a_malformed_leg_is_refused(facade, leg, why):
    pb, tc = facade({"robot1": leg})
    tc.set_last_config("c", "robot1", [0.2] * 6)
    out = pb.dispatch_plan("c", [{"robot_ns": "robot1", "operation_id": "W1"}])
    assert out["trajectories"] == {}
    f, = [f for f in out["failures"] if f.get("operation_id") == "W1"]
    assert f["reason_code"] == "backend_contract_violation" and why in f["error"]


def test_a_compound_partner_goes_with_the_refused_leg(facade):
    a = _leg([0.9] * 6, [1.0] * 6)
    b = _leg([0.2] * 6, [0.3] * 6)
    a[0]["sync_members"] = b[0]["sync_members"] = ["robot1", "robot2"]
    pb, tc = facade({"robot1": a, "robot2": b})
    tc.set_last_config("c", "robot1", [0.2] * 6)
    tc.set_last_config("c", "robot2", [0.2] * 6)
    out = pb.dispatch_plan("c", [{"robot_ns": "robot1", "operation_id": "W1"},
                                 {"robot_ns": "robot2", "operation_id": "W2"}])
    assert out["trajectories"] == {}
    by_op = {f["operation_id"]: f for f in out["failures"]}
    assert by_op["W2"]["reason_code"] == "step_refused"
    assert by_op["W2"]["caused_by"] == ["robot1"]


# ── shared leg pieces ────────────────────────────────────────────────────────

def test_no_backend_reaches_into_the_ompl_backends_leg_helpers():
    shared = ["cartesian_retreat", "cartesian_descent", "descent_failure",
              "trajectory_end", "accel_limits", "home_config", "emitted_rise_limit",
              "trajectory_tool_rise", "path_tool_rise"]
    pattern = re.compile(r"\b_?op\._(%s)\b" % "|".join(shared))
    for name in ("ompl_compound", "rrt"):
        text = (REPO / "agents/trajectory/backends" / name / "planner.py").read_text()
        assert not pattern.findall(text), name


def test_the_ompl_backend_uses_the_shared_pieces():
    pytest.importorskip("numpy")
    from agents.trajectory.backends.ompl import planner as op
    from agents.trajectory.tools import leg_builder as lb
    assert op._cartesian_retreat is lb.cartesian_retreat
    assert op._trajectory_tool_rise is lb.trajectory_tool_rise
    assert op._home_config is lb.home_config


def test_every_backend_accepts_options_it_does_not_use():
    import inspect
    pytest.importorskip("numpy")
    from agents.trajectory.backends.ompl.planner import plan_trajectories_ompl
    from agents.trajectory.backends.rrt.planner import plan_trajectories_rrt
    for fn in (plan_trajectories_ompl, plan_trajectories_rrt):
        kinds = {p.kind for p in inspect.signature(fn).parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD in kinds, fn.__name__
