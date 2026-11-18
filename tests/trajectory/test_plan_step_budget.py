"""The planner's step budget must scale with the work, and say so when it runs out.

A fixed budget of 100 was a silent cliff: a 15-weld cell burned through it after
63 tool calls of real planning and returned NO behaviour tree, reported as an
ordinary "planning" failure — indistinguishable from "the arm cannot reach that
weld".

These tests drive the REAL ``plan()`` with the LLM agent stubbed out, so they pin
what production actually computes rather than a copy of the formula: the
recursion_limit handed to LangGraph, and the error_kind on exhaustion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _ops(n: int) -> list:
    return [{"id": f"W{i:02d}", "type": "weld", "position": [0.0, 0.0, 1.0],
             "approach_vector": [0.0, 0.0, -1.0]} for i in range(n)]


class _FakeMessage:
    """Stands in for an AIMessage carrying the run's final text."""

    def __init__(self, content: str):
        self.content = content
        self.name = None


class _FakeAgent:
    """Captures the config plan() passes, and replays a scripted final message."""

    def __init__(self, final_text: str):
        self.final_text = final_text
        self.seen_config: dict = {}

    def stream(self, _inputs, config=None, stream_mode=None):
        self.seen_config = config or {}
        yield {"messages": [_FakeMessage(self.final_text)]}


@pytest.fixture
def plan_with(monkeypatch):
    """Run the real plan() against a stubbed agent; return (result, config)."""
    import agents.trajectory.agent as agent_mod

    def run(num_ops: int, final_text: str = "done", env: dict | None = None):
        fake = _FakeAgent(final_text)
        monkeypatch.setattr(agent_mod, "create_trajectory_agent", lambda **_kw: fake)
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        for key in ("TRAJ_RECURSION_LIMIT",):
            if not (env or {}).get(key):
                monkeypatch.delenv(key, raising=False)
        result = agent_mod.plan("welding_cell_2_small", _ops(num_ops), None)
        return result, fake.seen_config

    return run


# ── the budget scales with the work ───────────────────────────────────────────

def test_the_run_that_failed_would_now_fit(plan_with):
    """15 welds took 63 tool calls and died at 100; the budget must clear both."""
    _, config = plan_with(15)

    assert config["recursion_limit"] == 160
    assert config["recursion_limit"] > 63


def test_budget_never_drops_below_the_old_floor(plan_with):
    """A small cell keeps its former headroom — the change only ever adds room."""
    for num_ops in (0, 1, 7):
        _, config = plan_with(num_ops)
        assert config["recursion_limit"] == 100, num_ops


def test_budget_is_monotonic_in_operation_count(plan_with):
    """More welds must never buy fewer steps."""
    limits = [plan_with(n)[1]["recursion_limit"] for n in (0, 5, 10, 20, 40)]

    assert limits == sorted(limits)


def test_each_extra_weld_past_the_floor_buys_room(plan_with):
    _, small = plan_with(20)
    _, large = plan_with(21)

    assert large["recursion_limit"] - small["recursion_limit"] == 8


def test_the_ceiling_is_overridable(plan_with):
    """An operator must be able to widen or pin the budget without a code change."""
    _, config = plan_with(15, env={"TRAJ_RECURSION_LIMIT": "500"})

    assert config["recursion_limit"] == 500


# ── exhaustion is diagnosable ────────────────────────────────────────────────

def test_exhaustion_is_its_own_error_kind(plan_with):
    """The message LangGraph actually emitted must not read as a planning verdict."""
    result, _ = plan_with(15, final_text="Sorry, need more steps to process this request.")

    assert result["success"] is False
    assert result["error_kind"] == "step_budget"
    # The ceiling that was hit has to travel with the error, or raising it is guesswork.
    assert result["step_budget"] == 160
    assert "TRAJ_RECURSION_LIMIT" in result["note"]


def test_exhaustion_is_recognised_regardless_of_case(plan_with):
    result, _ = plan_with(15, final_text="NEED MORE STEPS")

    assert result["error_kind"] == "step_budget"


def test_an_ordinary_failure_is_not_relabelled_a_budget_problem(plan_with):
    """Raising the ceiling must not be suggested when the ceiling was not the issue."""
    result, _ = plan_with(15, final_text="Weld W07 is unreachable for robot2.")

    assert result["success"] is False
    assert result["error_kind"] == "planning"
    assert "step_budget" not in result
