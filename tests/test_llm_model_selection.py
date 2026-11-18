"""One place decides which model runs, and a caller may override it per operation.

Four modules each built their own chat model with their own hardcoded
``eu.anthropic.claude-sonnet-5``. Two consequences, both bad:

  * **The region prefix was a trap.** A Bedrock inference profile is region-coupled
    — an ``eu.`` profile is rejected outright in ``us-west-2`` — so the literal only
    worked from Ireland, and the failure was an opaque "invalid model identifier"
    rather than anything naming a region.
  * **Nothing could ask for a different model.** Neither A2A nor MCP had a way to
    say "plan this hard cell on Opus 5", so the only lever was a stack restart.

Now ``agents/llm_model.py`` owns the decision: aliases expand there, the profile
prefix is derived from the region there, and a per-request ContextVar binding lets
one operation run on a different model without touching the container default.

The binding matters more than it looks. The HOME staging proposer is reached from
deep inside OMPL, several frames below the request handler; no argument-threading
scheme reaches it, so the model has to be bound for the request instead of passed.

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
    """Resolution must be decided by the test, not by the developer's shell."""
    for var in ("LLM_MODEL", "LLM_PROVIDER", "BEDROCK_MODEL_ID", "OPENAI_MODEL",
                "OLLAMA_MODEL", "BEDROCK_REGION", "AWS_REGION",
                "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)


# ── the default ──────────────────────────────────────────────────────────────

def test_the_default_is_opus_5(lm):
    assert lm.DEFAULT_MODEL_ALIAS == "opus-5"
    assert lm.resolve_model() == "bedrock:eu.anthropic.claude-opus-5"


def test_the_default_is_not_the_sonnet_the_four_copies_pinned(lm):
    assert "sonnet" not in lm.resolve_model()


def test_a_model_id_carries_no_date_suffix(lm):
    """Claude 5 ids are undated; a date suffix is a 4.x-era habit that 404s."""
    for qualified in lm.describe_models()["aliases"].values():
        assert not re.search(r"-\d{8}(?:-v\d+:\d+)?$", qualified), qualified


# ── aliases ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alias,expected_core", [
    ("opus-5",    "anthropic.claude-opus-5"),
    ("opus5",     "anthropic.claude-opus-5"),
    ("opus",      "anthropic.claude-opus-5"),
    ("opus-4.8",  "anthropic.claude-opus-4-8"),
    ("sonnet-5",  "anthropic.claude-sonnet-5"),
    ("haiku-4.5", "anthropic.claude-haiku-4-5"),
])
def test_an_alias_expands_to_its_model(lm, alias, expected_core):
    assert lm.normalize_model(alias) == f"bedrock:eu.{expected_core}"


def test_an_alias_is_case_insensitive(lm):
    assert lm.normalize_model("Opus-5") == lm.normalize_model("opus-5")


def test_an_unspecified_model_means_unspecified_not_a_default(lm):
    """``None`` must fall through to the next source, not short-circuit it."""
    assert lm.normalize_model(None) is None
    assert lm.normalize_model("") is None
    assert lm.normalize_model("   ") is None


# ── region ↔ inference profile ────────────────────────────────────────────────

@pytest.mark.parametrize("region,prefix", [
    ("eu-west-1",      "eu."),
    ("eu-central-1",   "eu."),
    ("us-east-1",      "us."),
    ("us-west-2",      "us."),
    ("ap-northeast-1", "apac."),
])
def test_the_profile_prefix_follows_the_region(lm, region, prefix):
    """The bug the literal hid: a profile is only valid in its own region."""
    assert lm.region_prefix(region) == prefix
    assert lm.bedrock_model_id("anthropic.claude-opus-5", region) == \
        f"{prefix}anthropic.claude-opus-5"


def test_an_unknown_region_gets_the_global_profile(lm):
    """Better a cross-region profile than a prefix the region will reject."""
    assert lm.region_prefix("xx-nowhere-9") == "global."


def test_the_region_comes_from_the_environment(lm, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert lm.resolve_model() == "bedrock:us.anthropic.claude-opus-5"


def test_bedrock_region_beats_aws_region(lm, monkeypatch):
    """BEDROCK_REGION exists so the model pin survives an unrelated AWS_REGION."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_REGION", "eu-west-1")
    assert lm.resolve_model() == "bedrock:eu.anthropic.claude-opus-5"


def test_a_spelled_out_profile_is_left_alone(lm, monkeypatch):
    """An operator who writes the full profile id means it."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert lm.normalize_model("eu.anthropic.claude-opus-5") == \
        "bedrock:eu.anthropic.claude-opus-5"


def test_a_non_anthropic_id_is_not_decorated(lm):
    assert lm.bedrock_model_id("meta.llama3-70b-instruct-v1:0", "eu-west-1") == \
        "meta.llama3-70b-instruct-v1:0"


# ── resolution order ─────────────────────────────────────────────────────────

def test_llm_model_configures_the_startup_default(lm, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "sonnet-5")
    assert lm.resolve_model() == "bedrock:eu.anthropic.claude-sonnet-5"


def test_llm_model_beats_the_legacy_variable(lm, monkeypatch):
    """Both may be set during the migration; the new one has to win."""
    monkeypatch.setenv("LLM_MODEL", "opus-5")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-5")
    assert lm.resolve_model() == "bedrock:eu.anthropic.claude-opus-5"


def test_the_legacy_variable_still_works_alone(lm, monkeypatch):
    """An existing deployment that only sets BEDROCK_MODEL_ID must not change."""
    monkeypatch.setenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-5")
    assert lm.resolve_model() == "bedrock:eu.anthropic.claude-sonnet-5"


def test_an_explicit_argument_beats_the_environment(lm, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "sonnet-5")
    assert lm.resolve_model("opus-5") == "bedrock:eu.anthropic.claude-opus-5"


def test_a_binding_beats_everything(lm, monkeypatch):
    """The per-request override is the top of the order, by design."""
    monkeypatch.setenv("LLM_MODEL", "sonnet-5")
    with lm.bind_model("haiku"):
        assert lm.resolve_model("opus-5") == "bedrock:eu.anthropic.claude-haiku-4-5"


def test_a_binding_is_released_afterwards(lm):
    """A leaked binding would silently retarget every later request."""
    before = lm.resolve_model()
    with lm.bind_model("sonnet-5"):
        assert lm.resolve_model() != before
    assert lm.resolve_model() == before


def test_binding_nothing_leaves_the_default_in_place(lm):
    """Handlers bind unconditionally; ``model=None`` must mean "as configured"."""
    before = lm.resolve_model()
    with lm.bind_model(None):
        assert lm.resolve_model() == before
    assert lm.resolve_model() == before


def test_a_binding_survives_a_worker_thread(lm):
    """``asyncio.to_thread`` copies the context — the trajectory MCP relies on it.

    Planning runs in a worker thread, and the HOME staging proposer it may reach is
    several frames deeper. If the binding did not travel, an operation asked to run
    on one model would quietly plan on another.
    """
    import asyncio

    async def main():
        with lm.bind_model("sonnet-5"):
            return await asyncio.to_thread(lm.resolve_model)

    assert asyncio.run(main()) == "bedrock:eu.anthropic.claude-sonnet-5"


# ── providers ────────────────────────────────────────────────────────────────

def test_a_qualified_value_switches_provider(lm):
    """So one operation can be routed elsewhere without a restart."""
    assert lm.normalize_model("openai:gpt-4o") == "openai:gpt-4o"
    assert lm.normalize_model("ollama:llama3.1") == "ollama:llama3.1"


def test_each_provider_has_its_own_default(lm, monkeypatch):
    """A Claude alias means nothing to OpenAI, so it must not be the fallback."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert lm.resolve_model() == "openai:gpt-4o"


def test_an_unknown_provider_raises(lm, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrok")
    with pytest.raises(ValueError, match="bedrok"):
        lm.resolve_model()


def test_a_mistyped_provider_prefix_raises(lm):
    """"openia:gpt-4o" must not reach Bedrock as a model named "openia:gpt-4o"."""
    with pytest.raises(ValueError, match="openia"):
        lm.normalize_model("openia:gpt-4o")


def test_a_versioned_bedrock_id_is_not_mistaken_for_a_provider(lm):
    """Bedrock ids end in ``:0``; splitting on that colon would corrupt them."""
    vid = "anthropic.claude-3-sonnet-20240229-v1:0"
    assert lm.normalize_model(vid) == f"bedrock:eu.{vid}"


def test_an_ollama_tag_is_not_mistaken_for_a_provider(lm):
    """Ollama names models ``family:tag`` — that colon is part of the name."""
    assert lm.normalize_model("llama3.1:8b", "ollama") == "ollama:llama3.1:8b"


def test_split_model_round_trips(lm):
    for alias in ("opus-5", "sonnet-5"):
        qualified = lm.normalize_model(alias)
        provider, model_id = lm.split_model(qualified)
        assert f"{provider}:{model_id}" == qualified


# ── discovery ────────────────────────────────────────────────────────────────

def test_discovery_reports_what_can_be_asked_for(lm):
    """Backs GET /a2a/models and the two list_models MCP tools."""
    described = lm.describe_models()

    assert described["provider"] == "bedrock"
    assert described["default"] == "opus-5"
    assert described["effective"] == lm.resolve_model()
    assert described["region"] == lm.bedrock_region()
    for alias, qualified in described["aliases"].items():
        assert lm.normalize_model(alias) == qualified


# ── the fence: nobody carries their own model literal ────────────────────────

def test_no_module_hardcodes_a_claude_model_id():
    """``agents/llm_model.py`` owns the ids; everything else resolves through it.

    Tests are exempt — asserting on a literal id is the point of several above.
    """
    allowed = {Path("agents/llm_model.py")}
    roots = ("agents", "controller", "docker", "scripts", "utils", "dashboard",
             "cells")
    pattern = re.compile(r"claude-(?:opus|sonnet|haiku|instant|[0-9])")

    offenders = []
    searched = 0
    for root in roots:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            if rel in allowed:
                continue
            searched += 1
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i}")

    assert searched > 20, "the fence scanned almost nothing — check the roots"
    assert not offenders, (
        "a model id is hardcoded again in: " + ", ".join(offenders) +
        " — resolve it through agents.llm_model instead"
    )


def test_no_module_builds_its_own_chat_model():
    """One construction point, so region and provider are decided once.

    The dashboard is exempt for its Ollama branch alone, which needs a ``base_url``
    the shared builder does not take.
    """
    allowed = {Path("agents/llm_model.py"), Path("dashboard/web_control_panel.py")}
    roots = ("agents", "controller", "scripts", "utils", "dashboard")
    pattern = re.compile(r"\b(ChatBedrockConverse|ChatBedrock|ChatOpenAI)\s*\(")

    offenders = []
    for root in roots:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            if rel in allowed:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i}")

    assert not offenders, (
        "a chat model is constructed directly in: " + ", ".join(offenders) +
        " — call agents.llm_model.build_llm() instead"
    )


def test_the_claude_path_passes_no_sampling_parameters():
    """Claude 5 rejects these with an HTTP 400 — they are not soft options.

    ``temperature``, ``top_p``, ``top_k`` and ``budget_tokens`` are all refused
    outright; thinking is on by default, so a ``max_tokens`` cap now bounds thinking
    and text together and a tight one truncates the answer. Scoped to the Bedrock
    branch: Ollama still accepts a temperature, and the panel's Ollama branch sets
    one deliberately.
    """
    src = (PROJECT_ROOT / "agents" / "llm_model.py").read_text()
    bedrock_branch = src[src.index("def build_llm"):]
    bedrock_branch = bedrock_branch[:bedrock_branch.find("ChatOpenAI")
                                   if "ChatOpenAI" in bedrock_branch else None]
    rejected = re.compile(r"\b(temperature|top_p|top_k|budget_tokens)\s*=")

    found = rejected.findall(bedrock_branch)
    assert not found, (
        f"the Bedrock path passes {sorted(set(found))} — Claude 5 returns 400"
    )


# ── the startup layer agrees with the module ──────────────────────────────────

def test_the_makefile_default_agrees(lm):
    text = (PROJECT_ROOT / "Makefile").read_text()
    m = re.search(r"^LLM_MODEL\s*\?=\s*(\S+)", text, re.MULTILINE)
    assert m, "Makefile declares no LLM_MODEL default"
    assert m.group(1) == lm.DEFAULT_MODEL_ALIAS


def test_the_compose_default_agrees(lm):
    text = (PROJECT_ROOT / "docker-compose.yml").read_text()
    found = re.findall(r"\$\{LLM_MODEL:-([^}]+)\}", text)
    assert found, "docker-compose.yml declares no LLM_MODEL default"
    assert set(found) == {lm.DEFAULT_MODEL_ALIAS}


def _generate(*extra: str):
    """Run the compose generator with everything ambient pinned to an argument.

    Region, cell base and provider are all passed explicitly rather than inherited.
    A subprocess sees the whole process environment, and two of those values are
    read from it: the profile prefix follows the region, and the cell lookup follows
    ``CELL_BASE_PATH`` — which sibling tests set with a bare ``os.environ[...]``
    that outlives them. Pinned here, the assertion is about the generator rather
    than about whatever ran before it.
    """
    import subprocess

    return subprocess.run(
        [sys.executable, "scripts/generate_compose.py",
         "--cell", "welding_cell_2_small",
         "--cell-base", str(PROJECT_ROOT / "cells"),
         "--llm-provider", "bedrock",
         "--aws-region", "eu-west-1", *extra],
        cwd=PROJECT_ROOT, capture_output=True, text=True)


def test_the_generator_emits_the_default_model(lm):
    """``make sim`` with no arguments must start the stack on the default."""
    done = _generate()

    assert done.returncode == 0, done.stderr
    expected = lm.normalize_model(lm.DEFAULT_MODEL_ALIAS, "bedrock", "eu-west-1")
    assert f"LLM_MODEL: {expected}" in done.stdout


def test_the_generator_rejects_a_bad_model(lm):
    """Before writing a compose file, not after the containers fail to start."""
    done = _generate("--llm-model", "openia:gpt-4o")

    assert done.returncode != 0
    assert "openia" in done.stderr


# ── the model may never reach the motion planner ──────────────────────────────

def test_the_model_argument_is_not_the_planner_backend():
    """The safety boundary: an LLM proposes poses, OMPL validates them.

    ``constraints.trajectory_backend`` selects the MOTION planner and is read from
    the caller's binding, never from an LLM-supplied argument (see
    ``agents/trajectory/tools/planner_backend.py``). A ``model`` argument must not
    become a second, softer way to influence what the arms are commanded to do.
    """
    src = (PROJECT_ROOT / "agents" / "trajectory" / "agent.py").read_text()
    for line in src.splitlines():
        if "trajectory_backend" in line and "model" in line:
            pytest.fail(f"model and the motion backend are entangled: {line.strip()}")
