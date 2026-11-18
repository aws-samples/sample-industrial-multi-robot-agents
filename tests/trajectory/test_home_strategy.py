"""When OMPL cannot get an arm home, a strategy must be tried before a blind line.

OMPL answers one question — is there a path from A to B — and answering "no" is a
legitimate result: the arm finishes deep in a fixture, or the other arm is parked
across the corridor home. That "no" used to end the matter, and the executor drove
the arm home on an unchecked straight line, which is the most dangerous motion in
a replay taken at exactly the moment planning said the space is hard.

The staging layer closes that gap, and these tests hold its safety boundary:

  * the LLM proposes WHERE to route — it never produces a trajectory;
  * every leg between staging poses is planned by OMPL, against the same
    obstacles and the same swept volumes of the other arms;
  * a staged return ends on exactly the pose a direct return would have;
  * a strategy that cannot be planned in full is discarded, not partly executed;
  * when nothing plans, the result is NO trajectory — never an unchecked one.

Nothing here calls the real planner or a real model: the point is the contract
between the layers, and it must hold in seconds, offline, every time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"
NS = "robot1"
START = [1.2, -0.4, 0.9, 0.3, -0.5, 0.2]
HOME = [0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0]
LOWER = [-3.14] * 6
UPPER = [3.14] * 6


@pytest.fixture
def strategy():
    return pytest.importorskip("agents.trajectory.tools.home_strategy",
                               reason="needs the trajectory runtime")


@pytest.fixture
def no_llm(monkeypatch):
    """Deterministic strategies only — no network, no model."""
    monkeypatch.setenv("HOME_STRATEGY_LLM", "0")


@pytest.fixture
def fake_llm(strategy, monkeypatch):
    """Answer the strategy layer with a canned model reply; capture the prompt."""
    def install(reply_text, raises: bool = False):
        seen: dict = {}

        class _LLM:
            def invoke(self, messages):
                seen["system"] = messages[0].content
                seen["prompt"] = messages[-1].content
                if raises:
                    raise RuntimeError("no bedrock credentials")
                return type("R", (), {"content": reply_text})()

        import types
        mod = types.ModuleType("agents.trajectory.agent")
        mod._build_llm = lambda *a, **k: _LLM()
        monkeypatch.setitem(sys.modules, "agents.trajectory.agent", mod)

        # The message wrappers are LangChain's, and LangChain is a runtime
        # dependency of the agents — not of a strategy proposal. Stand them in
        # when absent so this contract is testable without the model stack.
        try:
            import langchain_core.messages  # noqa: F401
        except Exception:
            class _Msg:
                def __init__(self, content):
                    self.content = content
            lc = types.ModuleType("langchain_core")
            lc_msgs = types.ModuleType("langchain_core.messages")
            lc_msgs.HumanMessage = lc_msgs.SystemMessage = _Msg
            lc.messages = lc_msgs
            monkeypatch.setitem(sys.modules, "langchain_core", lc)
            monkeypatch.setitem(sys.modules, "langchain_core.messages", lc_msgs)

        monkeypatch.setenv("HOME_STRATEGY_LLM", "1")
        return seen

    return install


def _json_reply(*strategies) -> str:
    import json
    return "```json\n" + json.dumps({"strategies": list(strategies)}) + "\n```"


# ── the deterministic strategies are the operator's own moves ─────────────────

def test_the_first_strategy_folds_the_arm_without_moving_the_base(strategy):
    """An arm that cannot get home is extended; fold it before it swings.

    Keeping joint 1 at its current value while the rest go to HOME makes the arm
    compact IN PLACE. It then crosses the cell sweeping a small volume instead of
    dragging a fully extended arm through the other arm's space.
    """
    first = strategy.deterministic_strategies(START, HOME)[0]

    assert len(first["waypoints"]) == 1
    (wp,) = first["waypoints"]
    assert wp[0] == pytest.approx(START[0]), "the base must not move on the first leg"
    assert wp[1:] == pytest.approx(HOME[1:]), "everything else folds to HOME"


def test_no_strategy_merely_restates_the_start_or_home_pose(strategy):
    """A waypoint equal to an endpoint adds a leg and no information."""
    for s in strategy.deterministic_strategies(START, HOME):
        for wp in s["waypoints"]:
            assert wp != pytest.approx(START)
            assert wp != pytest.approx(HOME)


def test_a_short_arm_yields_no_nonsense_strategies(strategy):
    """Below two joints there is nothing to fold — say so rather than improvise."""
    assert strategy.deterministic_strategies([0.5], [0.0]) == []


# ── admissibility: cheap filter, not the safety check ────────────────────────

@pytest.mark.parametrize("bad, why", [
    ([0.0] * 5, "wrong arity"),
    ([0.0] * 7, "wrong arity"),
    ([9.9] + [0.0] * 5, "above the upper joint limit"),
    ([-9.9] + [0.0] * 5, "below the lower joint limit"),
    (["a"] + [0.0] * 5, "not numeric"),
    ([float("nan")] + [0.0] * 5, "not finite"),
    ([float("inf")] + [0.0] * 5, "not finite"),
])
def test_an_inadmissible_pose_is_rejected_before_the_planner(strategy, bad, why):
    """Spending a 5-second OMPL budget to discover a malformed pose is waste."""
    assert not strategy.admissible(bad, 6, LOWER, UPPER), why


def test_an_in_limit_pose_is_admissible(strategy):
    assert strategy.admissible(HOME, 6, LOWER, UPPER)


# ── the LLM is a proposal source, and an optional one ────────────────────────

def test_the_llm_is_not_asked_when_a_deterministic_pose_would_do(strategy, monkeypatch):
    """Cost and nondeterminism belong on the failure path only.

    ``propose_home_strategies`` is a generator precisely so the model is never
    consulted for an arm the free candidates rescue.
    """
    called = []
    monkeypatch.setattr(strategy, "llm_strategies",
                        lambda *a, **k: called.append(1) or [])

    gen = strategy.propose_home_strategies(CELL, NS, START, HOME,
                                           lower=LOWER, upper=UPPER)
    first = next(gen)

    assert first["source"] == "heuristic"
    assert called == [], "the LLM must not be called until the cheap moves run out"


def test_the_llm_is_asked_once_the_deterministic_moves_run_out(strategy, monkeypatch):
    monkeypatch.setattr(strategy, "llm_strategies",
                        lambda *a, **k: [{"waypoints": [[0.1] * 6],
                                          "source": "llm", "rationale": "x"}])

    sources = [s["source"] for s in strategy.propose_home_strategies(
        CELL, NS, START, HOME, lower=LOWER, upper=UPPER)]

    assert sources[-1] == "llm"
    assert "heuristic" in sources


def test_the_llm_can_be_switched_off(strategy, monkeypatch):
    """Offline runs and reproducible replays must not depend on a model."""
    monkeypatch.setenv("HOME_STRATEGY_LLM", "0")

    assert strategy.llm_strategies(CELL, NS, START, HOME) == []


def test_an_unavailable_llm_yields_no_strategies_rather_than_raising(
        strategy, fake_llm, capsys):
    """No credentials is a degradation: keep the deterministic path alive."""
    fake_llm("", raises=True)

    assert strategy.llm_strategies(CELL, NS, START, HOME) == []
    assert "unavailable" in capsys.readouterr().out.lower()


def test_a_malformed_reply_yields_no_strategies(strategy, fake_llm):
    """A model that answers in prose has proposed nothing."""
    fake_llm("I would suggest lifting the arm a bit and then going home.")

    assert strategy.llm_strategies(CELL, NS, START, HOME) == []


def test_a_proposed_pose_outside_the_joint_limits_is_discarded(
        strategy, fake_llm, capsys):
    """The model's arithmetic is not trusted — it is checked."""
    fake_llm(_json_reply(
        {"waypoints": [[9.9, 0, 0, 0, 0, 0]], "rationale": "lift high"},
        {"waypoints": [[0.5] * 6], "rationale": "stage mid-cell"},
    ))

    out = strategy.llm_strategies(CELL, NS, START, HOME,
                                  lower=LOWER, upper=UPPER)

    assert [s["waypoints"] for s in out] == [[[0.5] * 6]]
    assert "discarded" in capsys.readouterr().out.lower()


def test_a_proposal_with_the_wrong_joint_count_is_discarded(strategy, fake_llm):
    fake_llm(_json_reply({"waypoints": [[0.1, 0.2, 0.3]], "rationale": "short"}))

    assert strategy.llm_strategies(CELL, NS, START, HOME,
                                   lower=LOWER, upper=UPPER) == []


def test_at_most_three_proposals_are_taken(strategy, fake_llm):
    """A model asked for options can return many; each one costs an OMPL budget."""
    fake_llm(_json_reply(*[{"waypoints": [[0.1 * i] * 6], "rationale": str(i)}
                           for i in range(1, 8)]))

    assert len(strategy.llm_strategies(CELL, NS, START, HOME,
                                       lower=LOWER, upper=UPPER)) == 3


# ── what the model is actually told ──────────────────────────────────────────

def test_the_model_is_told_the_failure_it_has_to_solve(strategy, fake_llm):
    """Poses, limits, the other arms and the margins already attempted."""
    seen = fake_llm(_json_reply())

    strategy.llm_strategies(CELL, NS, START, HOME,
                            others={"robot2": [0.2] * 6},
                            attempted_margins_m=[0.20, 0.15],
                            lower=LOWER, upper=UPPER)
    prompt = seen["prompt"]

    assert NS in prompt and CELL in prompt
    assert "robot2" in prompt, "the arm it has to route around"
    assert "0.20" in prompt and "0.15" in prompt, "the margins already tried"
    assert "-3.14" in prompt, "the joint limits it must respect"


def test_the_model_is_told_which_strategies_already_failed(strategy, fake_llm):
    """Otherwise it proposes the fold that OMPL has just rejected."""
    seen = fake_llm(_json_reply())

    list(strategy.propose_home_strategies(CELL, NS, START, HOME,
                                          lower=LOWER, upper=UPPER))

    assert "fold" in seen["prompt"].lower()


def test_the_model_is_forbidden_from_producing_trajectories(strategy, fake_llm):
    """The boundary is stated in the prompt as well as enforced in code."""
    seen = fake_llm(_json_reply())

    strategy.llm_strategies(CELL, NS, START, HOME)

    system = seen["system"].lower()
    assert "do not produce trajectories" in system
    assert "collision-checked by ompl" in system


def test_fixtures_are_summarised_rather_than_pasted_voxel_by_voxel(strategy):
    """A voxel shell is hundreds of boxes for one physical object.

    Grouped by base id they become one bounding box per fixture — a scene a model
    can reason about, instead of a token bill.
    """
    summary = strategy._obstacle_summary(CELL)

    assert "_v000" not in summary and "_v001" not in summary
    assert summary.count("\n") < 20, "one line per fixture, not per voxel"


def test_a_cell_with_no_obstacle_model_still_produces_a_prompt(strategy):
    """Missing geometry must not raise on the failure path."""
    assert strategy._obstacle_summary("no_such_cell_at_all")
