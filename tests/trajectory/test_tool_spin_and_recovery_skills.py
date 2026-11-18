"""The agent can turn the tool about its axis, and the skills say how to recover.

* ``tool_spin_deg`` on a motion request turns the tool about its OWN axis (the TCP
  frame's -x): the lean ``validate_approach`` checks stays, the roll changes. It is
  applied by the backend facade, so every backend gets the same pose.
* A blocked straight descent names what blocked it (the arm itself, IK, or an
  obstacle) instead of always reading as ``goal_in_collision``.
* The skills and prompts agree with that: the lean is fixed, the spin is free;
  validators take operation ids; every new failure code has a recovery.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agents.trajectory.tools.planner_backend import apply_tool_spin  # noqa: E402


def _R(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _col(R, i):
    return [R[0][i], R[1][i], R[2][i]]


def _angle(a, b):
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    d = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


@pytest.mark.parametrize("spin", [30, -45, 90, 180])
def test_a_spin_keeps_the_lean_and_turns_the_roll(spin):
    q = {"x": 0.0, "y": 0.923877, "z": 0.0, "w": -0.382689}
    out = apply_tool_spin({"orientation": q, "tool_spin_deg": spin, "operation_id": "W1"})
    R0, R1 = _R(q), _R(out["orientation"])
    assert _angle(_col(R0, 0), _col(R1, 0)) < 0.01           # tool axis (-x) unchanged
    assert _angle(_col(R0, 1), _col(R1, 1)) == pytest.approx(abs(spin), abs=0.01)
    assert "tool_spin_deg" not in out and out["applied_tool_spin_deg"] == spin
    assert out["operation_id"] == "W1"


def test_no_spin_leaves_the_request_alone():
    r = {"orientation": {"w": 1, "x": 0, "y": 0, "z": 0}}
    assert apply_tool_spin(r) is r
    assert apply_tool_spin({"tool_spin_deg": 30}) == {"tool_spin_deg": 30}


def test_the_facade_applies_it_for_every_backend(monkeypatch):
    from agents.trajectory.tools import planner_backend as pb
    seen = []
    monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: (
        lambda c, reqs, **k: seen.extend(reqs) or {"success": True, "trajectories": {},
                                                  "failures": [], "planning_time_s": 0}))
    monkeypatch.setattr(pb, "_self_collision_refusals", lambda c, r: [])
    pb.dispatch_plan("c", [{"robot_ns": "r", "orientation": {"w": 1, "x": 0, "y": 0, "z": 0},
                            "tool_spin_deg": 90}])
    assert seen[0]["orientation"]["x"] == pytest.approx(math.sin(math.pi / 4))
    assert "tool_spin_deg" not in seen[0]


def test_a_blocked_descent_names_its_cause():
    op = pytest.importorskip("agents.trajectory.backends.ompl.planner")
    self_hit = op._descent_failure({"cause": "collision", "step": 1, "steps": 6,
                                    "collision": {"type": "self", "object_id": "l3 vs l6",
                                                  "clearance_m": -0.011}})
    assert self_hit["reason_code"] == "approach_self_collision" and "tool_spin_deg" in self_hit["reason"]
    assert op._descent_failure({"cause": "ik", "step": 3, "steps": 6})["reason_code"] == \
        "approach_unreachable"
    box = op._descent_failure({"cause": "collision", "step": 2, "steps": 6,
                               "collision": {"type": "box", "object_id": "fx"}})
    assert box["reason_code"] == "goal_in_collision" and box["collision"]["object_id"] == "fx"


# ── the skills and prompts agree with the code ───────────────────────────────

FLAVOURS = ("trajectory", "trajectory-classic")


@pytest.mark.parametrize("flav", FLAVOURS)
def test_the_recovery_skill_covers_every_new_failure_code(flav):
    text = (REPO / "skills" / flav / "recover-from-planner-failure/SKILL.md").read_text()
    for code in ("goal_self_collision", "approach_self_collision", "approach_unreachable",
                 "assignment_violation", "step_gate", "not_planned",
                 "self_collision_unavailable", "tool_spin_deg"):
        assert code in text, (flav, code)


@pytest.mark.parametrize("flav", FLAVOURS)
def test_the_lean_is_fixed_and_the_spin_is_free(flav):
    text = (REPO / "skills" / flav / "approach-strategy/SKILL.md").read_text()
    assert "never change the tool's LEAN" in text and "tool_spin_deg" in text
    assert "Never edit the orientation quaternion by hand" in text


@pytest.mark.parametrize("flav", FLAVOURS)
def test_validators_are_taught_operation_ids(flav):
    text = (REPO / "skills" / flav / "validate-collision/SKILL.md").read_text()
    assert 'trajectories={"robot1": "W01"' in text
    assert 'trajectories={"robot1": [...]' not in text


def test_the_prompts_agree():
    from agents.trajectory.classic_prompt import CLASSIC_PROMPT_TEMPLATE
    agent_src = (REPO / "agents/trajectory/agent.py").read_text()
    # agent.py keeps "{robot_ns: traj, ...}" (tests/trajectory/test_system_prompt_braces.py
    # pins that line); its EFFICIENT_TOOL_USE_PROMPT says to pass operation ids.
    assert "robot_ns: traj, ..." not in CLASSIC_PROMPT_TEMPLATE
    assert "{robot_ns: operation_id}" in agent_src
    for text in (CLASSIC_PROMPT_TEMPLATE, agent_src):
        assert "tool_spin_deg" in text
