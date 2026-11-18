"""The response-length ceiling is STATED, never inherited from the provider.

``ChatBedrockConverse`` leaves ``max_tokens=None`` unless told otherwise, which
omits ``inferenceConfig.maxTokens`` and lets Bedrock apply its own default. That
default was measured at **4096** output tokens on ``eu.anthropic.claude-opus-5``
(a 4000-line count came back with ``stopReason: max_tokens`` at exactly 4096),
against a model that allows 64k.

4096 is smaller than one step of a real planning run, and the way it fails is
silent: when the cut lands inside a ``toolUse`` block, the partial block is
unparseable, langchain drops it, and the AIMessage arrives with no text, no
tool_calls and no invalid_tool_calls. The ReAct loop has nothing to execute and
ends — a truncated sentence reported as "the agent stopped after 18 tool calls",
with an empty final message, indistinguishable from a planner that gave up.
Two planning runs on welding_cell_2_small died that way, at 18 and at 27 tool calls,
after every weld they had planned validated.

So two invariants live here: the ceiling is sent, and a truncation is never again
reported as a planning verdict.

Host-only: stdlib and source text. No langchain, no boto3, no network.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def lm():
    return pytest.importorskip("agents.llm_model")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """The ceiling must be decided by the test, not by the developer's shell."""
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)


# ── the ceiling ──────────────────────────────────────────────────────────────

def test_a_ceiling_is_stated_by_default(lm):
    assert lm.max_output_tokens() == lm._DEFAULT_MAX_OUTPUT_TOKENS


def test_the_default_ceiling_clears_bedrocks_inherited_4096(lm):
    """The number that has to be beaten is the measured provider default."""
    assert lm.max_output_tokens() > 4096


def test_an_operator_can_raise_or_lower_it(lm, monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "12345")
    assert lm.max_output_tokens() == 12345


def test_zero_means_let_the_provider_decide(lm, monkeypatch):
    """An explicit opt-out, for a model whose own ceiling is below ours."""
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    assert lm.max_output_tokens() is None


def test_a_junk_value_falls_back_to_the_provider_rather_than_crashing(lm, monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "lots")
    assert lm.max_output_tokens() is None


def test_it_is_read_per_call_not_frozen_at_import(lm, monkeypatch):
    """Containers are configured by environment; a cached read ignores it."""
    monkeypatch.setenv("LLM_MAX_TOKENS", "7000")
    assert lm.max_output_tokens() == 7000
    monkeypatch.setenv("LLM_MAX_TOKENS", "9000")
    assert lm.max_output_tokens() == 9000


def test_the_ceiling_is_reported_by_the_discovery_endpoint(lm):
    assert lm.describe_models()["max_output_tokens"] == lm.max_output_tokens()


# ── it reaches the constructed model ─────────────────────────────────────────

def test_every_hosted_provider_is_built_with_the_ceiling(lm):
    """A ceiling computed but not passed is the bug this test exists to catch."""
    import inspect

    src = inspect.getsource(lm.build_llm)

    def call_span(text: str, opening: str) -> str:
        """The constructor call, to its matching paren — nested calls included."""
        start = text.index(opening) + len(opening)
        depth = 1
        for i, ch in enumerate(text[start:], start):
            depth += (ch == "(") - (ch == ")")
            if depth == 0:
                return text[start:i]
        raise AssertionError(f"unbalanced parentheses after {opening}")

    for constructor in ("ChatBedrockConverse(", "ChatOpenAI("):
        assert "max_tokens=" in call_span(src, constructor), (
            f"{constructor} is built without a ceiling"
        )

    # The ceiling and the socket timeout are one decision: a generation long
    # enough to need 32k tokens outlives botocore's 60 s default read timeout,
    # and the failure mode swaps silent truncation for a read-timeout crash
    # 2000 s into a run. Passing one without the other is the bug.
    bedrock = call_span(src, "ChatBedrockConverse(")
    assert "config=" in bedrock, "ChatBedrockConverse is built without a boto config"
    assert "read_timeout" in src, "the read timeout is left at botocore's 60 s default"


# ── the socket outlives the generation ───────────────────────────────────────

def test_the_read_timeout_far_exceeds_botocores_default(lm):
    """60 s is botocore's default and is not enough for one planning step."""
    assert lm.read_timeout_s() > 60


def test_an_operator_can_raise_the_read_timeout(lm, monkeypatch):
    monkeypatch.setenv("LLM_READ_TIMEOUT_S", "1800")
    assert lm.read_timeout_s() == 1800


@pytest.mark.parametrize("bad", ["0", "-5", "soon", ""])
def test_a_useless_read_timeout_falls_back_rather_than_disabling_the_wait(lm, monkeypatch, bad):
    """A 0 or junk value must not become "give up immediately"."""
    monkeypatch.setenv("LLM_READ_TIMEOUT_S", bad)
    assert lm.read_timeout_s() > 60


def test_the_read_timeout_is_reported_by_the_discovery_endpoint(lm, monkeypatch):
    monkeypatch.setenv("LLM_READ_TIMEOUT_S", "1234")
    assert lm.describe_models()["read_timeout_s"] == 1234


# ── a truncation is not a planning verdict ───────────────────────────────────

def test_a_truncated_reply_is_reported_as_truncation_not_as_a_give_up():
    """``error_kind`` is what callers branch on, so the kinds must not collide."""
    src = (PROJECT_ROOT / "agents" / "trajectory" / "agent.py").read_text()
    assert '"truncated"' in src, "no error_kind distinguishes a cut-off reply"

    # The truncation branch must return BEFORE the generic "did not produce a
    # BehaviourTree" one, or the specific diagnosis is never reached.
    truncated_at = src.index('"error_kind": "truncated"')
    generic_at = src.index('"error_kind":     "planning"')
    assert truncated_at < generic_at


def test_the_stop_reason_is_read_in_every_spelling_a_provider_uses():
    """Bedrock says stopReason, OpenAI says finish_reason, others stop_reason."""
    src = (PROJECT_ROOT / "agents" / "trajectory" / "agent.py").read_text()
    window = src[src.index('"error_kind": "truncated"') - 3000:
                 src.index('"error_kind": "truncated"')]
    for spelling in ("stopReason", "stop_reason", "finish_reason"):
        assert spelling in window, f"{spelling} is never consulted"


def test_a_message_with_nothing_to_act_on_is_logged():
    """The loop ending on an empty message used to print nothing at all."""
    src = (PROJECT_ROOT / "agents" / "trajectory" / "agent.py").read_text()
    log = src[src.index("def _log_agent_message"):src.index("def _extract_behaviour_tree")]
    assert re.search(r"if not text and not tool_calls", log), (
        "an AIMessage with neither text nor tool calls still ends the run silently"
    )
    assert "stopReason" in log, "the silent-end log does not say why it stopped"
