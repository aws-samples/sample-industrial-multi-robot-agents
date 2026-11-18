"""The classic planning mode rebuilds the end-of-August 2026 planner (2f786cb,
29 Aug) on today's stack.

Pinned: it is the default; its prompt is the 29 Aug prompt with only the skill
catalog filled in; it offers exactly the 19 tools the planner had then; and its
skills are the 29 Aug texts, served from their own skill set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.classic_prompt import CLASSIC_PROMPT_TEMPLATE, classic_prompt  # noqa: E402
from utils import skill_registry as sr  # noqa: E402


def test_the_classic_prompt_is_the_29_aug_flow():
    p = classic_prompt("  - [priority 50] plan-trajectories: x")
    assert "<<SKILL_CATALOG>>" not in p and "plan-trajectories: x" in p
    for step in ("run_mrta_pipeline", "run_trajectory_planner", "validate_trajectory_pair",
                 "chain_trajectories", "assemble_behaviour_tree"):
        assert step in p
    # none of the later layers
    for later in ("coordinate_program", "delay_fix", "serial_steps", "SERIALISING IS NOT"):
        assert later not in CLASSIC_PROMPT_TEMPLATE


def test_the_classic_skills_are_their_own_set():
    names = {s["name"] for s in sr.list_skills(agent="trajectory-classic")}
    assert {"optimize-sequence", "plan-trajectories", "find-interlocks",
            "build-behaviour-tree", "validate-collision"} <= names
    assert "coordinate-arms" not in names and "optimize-cycle-time" not in names


@pytest.fixture
def agent_mod():
    try:
        import agents.trajectory.agent as A
    except Exception as exc:  # noqa: BLE001 — no LangChain on this host
        pytest.skip(f"agent not importable here: {exc}")
    return A


def test_classic_is_the_default_mode(agent_mod, monkeypatch):
    monkeypatch.delenv("PLANNING_MODE", raising=False)
    assert agent_mod.resolve_planning_mode(None) == "classic"
    assert agent_mod.resolve_planning_mode({"planning_mode": "coordinated"}) == "coordinated"
    assert agent_mod.resolve_planning_mode({"planning_mode": "stepwise"}) == "stepwise"


def test_classic_offers_exactly_the_29_aug_tools(agent_mod):
    names = {getattr(t, "name", "") for t in agent_mod._TOOLS
             if getattr(t, "name", "") in agent_mod.CLASSIC_TOOL_NAMES}
    assert names == set(agent_mod.CLASSIC_TOOL_NAMES)
    assert len(agent_mod.CLASSIC_TOOL_NAMES) == 19


def test_the_classic_prompt_lists_the_classic_skills(agent_mod):
    p = agent_mod.system_prompt_for(None, "classic")
    assert "coordinate-arms" not in p and "optimize-sequence" in p
