"""The one place that answers "which model runs this operation?".

Three modules used to answer it independently — the trajectory agent, the
orchestrator and the cell's recovery layer each carried their own ``_build_llm``
with its own hardcoded ``"eu.anthropic.claude-sonnet-5"`` default. Four copies of
a decision is how the four drift apart: pointing the stack at a different model
meant finding every copy, and missing one left that agent silently on the old
model while the logs reported the new one.

This module owns the decision end to end:

  * **Startup** picks the default — ``make sim LLM_MODEL=opus-5`` writes
    ``LLM_MODEL`` into every container's environment (scripts/generate_compose.py).
  * **Per operation** overrides it — the A2A endpoints and the MCP tools take a
    ``model`` argument, and :func:`bind_model` pins it for the duration of that
    one request so every nested LLM call in that run uses the same model
    (the planner's ReAct loop, the HOME staging proposer, the recovery reasoner).

Resolution order, highest first:

  1. The context binding set by :func:`bind_model` for the current request.
  2. An explicit argument (direct callers, tests).
  3. ``LLM_MODEL`` — the provider-neutral startup setting.
  4. The provider's own legacy variable (``BEDROCK_MODEL_ID`` / ``OPENAI_MODEL``
     / ``OLLAMA_MODEL``), so existing deployments keep working unchanged.
  5. The provider's default — Opus 5 on Bedrock.

Deliberately import-safe: standard library only. ``langchain_*`` is imported
lazily inside :func:`build_llm`, so a host with no model stack can still import
this module, read the aliases, and unit-test the resolution rules.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Optional

# ─── Model catalog ───────────────────────────────────────────────────────────
# Short names an operator can actually type, mapped to the vendor-neutral core
# model id. The provider decorates that core: Bedrock prefixes a cross-region
# inference profile (eu./us./apac./global.), the direct Anthropic API uses the
# bare id. Add a model by adding a row here — not by editing an agent.
MODEL_ALIASES: dict[str, str] = {
    "opus-5":    "anthropic.claude-opus-5",
    "opus5":     "anthropic.claude-opus-5",
    "opus":      "anthropic.claude-opus-5",
    "opus-4.8":  "anthropic.claude-opus-4-8",
    "sonnet-5":  "anthropic.claude-sonnet-5",
    "sonnet5":   "anthropic.claude-sonnet-5",
    "sonnet":    "anthropic.claude-sonnet-5",
    "haiku-4.5": "anthropic.claude-haiku-4-5",
    "haiku":     "anthropic.claude-haiku-4-5",
}

# What the stack runs when nothing says otherwise.
DEFAULT_MODEL_ALIAS = "opus-5"

# What TRAJECTORY PLANNING runs when the request names no model. Planning is a
# long ReAct loop of small, tool-driven decisions — 40+ sequential turns, measured
# at 92 % of a run's wall time on bt_1790857060 (1,124 s of 1,287 s waiting on the
# model, 45 s in OMPL) — so the per-turn latency of the model IS the planning
# time. Sonnet answers each turn several times faster; a hard cell can still ask
# for Opus per request (``model="opus-5"``) or for every run (PLANNING_LLM_MODEL).
DEFAULT_PLANNING_MODEL_ALIAS = "sonnet-5"

# Per-provider fallbacks for the non-Claude providers, which have no alias table.
PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o",
    "ollama": "llama3.1",
}

DEFAULT_PROVIDER = "bedrock"
VALID_PROVIDERS: tuple[str, ...] = ("bedrock", "openai", "ollama")

# Bedrock serves these models through cross-region inference profiles, and the
# profile prefix MUST match the region it is called in — a "us." profile is
# rejected outright in eu-west-1. So the prefix is derived from the region rather
# than baked into the model id, which is what made the old hardcoded default a
# trap for anyone running outside Ireland.
_REGION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("eu-", "eu."),
    ("us-", "us."),
    ("ap-", "apac."),
    ("ca-", "us."),      # Canada is served by the North America profile
    ("sa-", "us."),
)
_FALLBACK_PREFIX = "global."


def bedrock_region() -> str:
    """The region Bedrock is called in. ``BEDROCK_REGION`` pins it explicitly."""
    return (
        os.environ.get("BEDROCK_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "eu-west-1"
    )


def region_prefix(region: str) -> str:
    """The Bedrock inference-profile prefix that is valid in ``region``."""
    r = (region or "").strip().lower()
    for start, prefix in _REGION_PREFIXES:
        if r.startswith(start):
            return prefix
    return _FALLBACK_PREFIX


def bedrock_model_id(core: str, region: Optional[str] = None) -> str:
    """Decorate a core model id with the inference profile for its region.

    Already-prefixed ids (``eu.anthropic...``) and non-Anthropic ids are returned
    untouched — an operator who spells out a full profile id means it.
    """
    if not core.startswith("anthropic."):
        return core
    return f"{region_prefix(region or bedrock_region())}{core}"


def anthropic_model_id(core: str) -> str:
    """The direct-API spelling: the bare id, with no provider prefix.

    Kept alongside :func:`bedrock_model_id` so the two spellings of one model
    live in the same place. Used when talking to the Anthropic API directly
    rather than through Bedrock.
    """
    return core[len("anthropic."):] if core.startswith("anthropic.") else core


# ─── Normalisation ───────────────────────────────────────────────────────────

def normalize_model(value: Optional[str],
                    provider: Optional[str] = None,
                    region: Optional[str] = None) -> Optional[str]:
    """Turn what a caller typed into a canonical ``"provider:model_id"``.

    Accepts an alias (``"opus-5"``), a bare model id
    (``"eu.anthropic.claude-opus-5"``, ``"gpt-4o"``), or a fully-qualified
    ``"provider:model"`` — which also switches provider, so one operation can be
    routed to a local Ollama without restarting the stack.

    ``region`` picks the Bedrock inference-profile prefix. It defaults to this
    process's region, which is what an agent wants; pass it explicitly when
    resolving on behalf of somewhere else — ``scripts/generate_compose.py``
    resolves for the containers' region, not the region of the shell that ran it.

    Returns ``None`` for ``None``/empty, meaning "unspecified"; callers then fall
    through to the next source. Raises ``ValueError`` on an unknown provider so a
    typo fails loudly instead of quietly reaching the wrong endpoint.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    prov = (provider or current_provider()).strip().lower()
    if ":" in raw:
        head, tail = raw.split(":", 1)
        head = head.strip().lower()
        if head in VALID_PROVIDERS:
            prov, raw = head, tail.strip()
        elif prov == "bedrock" and "." not in head:
            # Not a provider, and not a Bedrock id either: every Bedrock id is
            # dotted (`anthropic.claude-…`, including versioned `…-v1:0`), so a
            # bare word before the colon is a mistyped provider. Fail here rather
            # than let "openia:gpt-4o" reach Bedrock as a model name.
            raise ValueError(
                f"Unknown LLM provider {head!r} in {raw!r}. Valid options: "
                f"{', '.join(VALID_PROVIDERS)}."
            )
        # Otherwise the ':' belongs to the model id itself — a Bedrock versioned
        # id (…-v1:0) or an Ollama tag (llama3.1:8b) — so leave prov and raw alone.

    if prov not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider {prov!r}. Valid options: "
            f"{', '.join(VALID_PROVIDERS)}."
        )
    if not raw:
        return None

    # Aliases are Claude short names, so they only mean anything on a provider
    # that serves Claude. Elsewhere the value passes through verbatim.
    core = MODEL_ALIASES.get(raw.lower())
    if prov == "bedrock":
        return f"bedrock:{bedrock_model_id(core or raw, region)}"
    return f"{prov}:{raw}"


def current_provider() -> str:
    """The configured provider (``LLM_PROVIDER``), lower-cased."""
    return (os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


def default_model_for(provider: str) -> str:
    """The model a provider falls back to when nothing was asked for.

    Opus 5 on Bedrock; a provider-appropriate default elsewhere, since a Claude
    alias means nothing to OpenAI or Ollama. Public because the startup layer
    (``scripts/generate_compose.py``) needs the same answer the agents will
    reach, resolved against the containers' region rather than its own.
    """
    return PROVIDER_DEFAULTS.get((provider or "").strip().lower(),
                                 DEFAULT_MODEL_ALIAS)


def split_model(qualified: str) -> tuple[str, str]:
    """Split a canonical ``"provider:model_id"`` into its two halves."""
    for prov in VALID_PROVIDERS:
        if qualified.startswith(f"{prov}:"):
            return prov, qualified[len(prov) + 1:]
    return DEFAULT_PROVIDER, qualified


# ─── Per-request binding ─────────────────────────────────────────────────────
# ``None`` means "no request bound a model" — resolve() then consults the
# explicit argument, then the environment, then the default.
_LLM_MODEL: ContextVar[Optional[str]] = ContextVar("llm_model", default=None)


def set_llm_model(value: Optional[str]) -> Token:
    """Pin the model for the current context (one A2A request / MCP call).

    Returns a token for :func:`reset_llm_model`. Prefer :func:`bind_model`, which
    cannot leak the binding into the next request on the same thread.
    """
    return _LLM_MODEL.set(normalize_model(value))


def reset_llm_model(token: Token) -> None:
    """Undo a previous :func:`set_llm_model`, restoring the prior binding."""
    _LLM_MODEL.reset(token)


@contextmanager
def bind_model(value: Optional[str]):
    """Pin ``value`` as the model for the duration of the block.

    A no-op for ``None``, so every call site can wrap unconditionally::

        with bind_model(model):
            run_the_operation()

    The binding is what makes a per-operation model reach the LLM calls nested
    inside that operation without threading an argument through every layer.
    """
    token = set_llm_model(value)
    try:
        yield current_llm_model()
    finally:
        reset_llm_model(token)


def current_llm_model() -> Optional[str]:
    """The context-bound model, or ``None`` if this request pinned none."""
    return _LLM_MODEL.get()


# ─── Resolution ──────────────────────────────────────────────────────────────

def resolve_model(explicit: Optional[str] = None) -> str:
    """The effective ``"provider:model_id"`` for right now.

    See the module docstring for the priority order. Never returns ``None``:
    something always runs, and on Bedrock that something is Opus 5.
    """
    bound = current_llm_model()
    if bound is not None:
        return bound

    from_arg = normalize_model(explicit)
    if from_arg is not None:
        return from_arg

    provider = current_provider()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. Valid options: "
            f"{', '.join(VALID_PROVIDERS)}."
        )

    from_env = normalize_model(os.environ.get("LLM_MODEL"), provider)
    if from_env is not None:
        return from_env

    legacy = {"bedrock": "BEDROCK_MODEL_ID", "openai": "OPENAI_MODEL",
              "ollama": "OLLAMA_MODEL"}[provider]
    from_legacy = normalize_model(os.environ.get(legacy), provider)
    if from_legacy is not None:
        return from_legacy

    resolved = normalize_model(default_model_for(provider), provider)
    assert resolved is not None  # the defaults above are never empty
    return resolved


def effective_model_id(explicit: Optional[str] = None) -> str:
    """Just the model id half of :func:`resolve_model` — for logs and reports."""
    return split_model(resolve_model(explicit))[1]


def planning_model(requested: Optional[str] = None) -> str:
    """The ``"provider:model_id"`` a trajectory PLANNING run uses.

    Resolution, highest first:

      1. ``requested`` — the ``model`` of this planning request (A2A / MCP);
      2. a model already bound for this context (a caller that bound one);
      3. ``PLANNING_LLM_MODEL`` — the startup setting for planning only;
      4. :data:`DEFAULT_PLANNING_MODEL_ALIAS` (Sonnet) on a provider that serves
         Claude; elsewhere the provider's own resolution (:func:`resolve_model`).

    ``LLM_MODEL`` is deliberately NOT consulted: it is the stack-wide default
    (orchestrator, recovery), and the compose file always sets it, so honouring it
    here would make the planning default unreachable.
    """
    explicit = normalize_model(requested)
    if explicit is not None:
        return explicit
    bound = current_llm_model()
    if bound is not None:
        return bound
    from_env = normalize_model(os.environ.get("PLANNING_LLM_MODEL"))
    if from_env is not None:
        return from_env
    if current_provider() == "bedrock":
        resolved = normalize_model(DEFAULT_PLANNING_MODEL_ALIAS, "bedrock")
        assert resolved is not None
        return resolved
    return resolve_model()


# ─── Prompt caching ──────────────────────────────────────────────────────────
# A ReAct turn re-sends the whole conversation: system prompt, tool specs, and
# every tool result so far. Without a cache point Bedrock re-reads all of it on
# every turn, so turn N costs N turns of input. With one, the shared prefix is
# read from cache and only the newest messages are processed.

def prompt_cache_enabled() -> bool:
    """``LLM_PROMPT_CACHE=0`` turns Bedrock prompt caching off (default on)."""
    return (os.environ.get("LLM_PROMPT_CACHE", "1").strip().lower()
            not in ("0", "false", "no", "off"))


def bind_tools_for_agent(llm, tools: list):
    """``llm.bind_tools(tools)`` — with a rolling Bedrock cache point when on.

    langchain-aws places the cache point after the system prompt, the tool specs
    and the latest message (``cache_control``), so each turn reuses the previous
    turn's prefix. The binding is done here, before ``create_react_agent``, which
    then uses it as is (it rebinds only a model with no tools bound).
    """
    from langchain_aws import ChatBedrockConverse
    if prompt_cache_enabled() and isinstance(llm, ChatBedrockConverse):
        return llm.bind_tools(tools, cache_control={"ttl": "5m"})
    return llm.bind_tools(tools)


# The response-length ceiling MUST be stated, not inherited.
#
# ``ChatBedrockConverse`` leaves ``max_tokens=None`` unless told otherwise, which
# omits ``inferenceConfig.maxTokens`` and lets Bedrock apply its own default of
# 4096 output tokens — measured, not assumed: a 4000-line count on
# eu.anthropic.claude-opus-5 came back with ``stopReason: max_tokens`` at exactly
# 4096, while the model itself allows 64k.
#
# 4096 is not enough for one step of a real planning run, and the way it fails is
# silent. When the truncation lands inside a ``toolUse`` block, Bedrock returns
# the partial block, langchain drops it (it is not parseable as a tool call), and
# the AIMessage arrives with NO text, NO tool_calls and NO invalid_tool_calls.
# The ReAct loop sees a message with nothing to execute and treats it as the
# agent's final answer, so a transport-level truncation was reported as
# "the agent stopped after 18 tool calls without assembling a tree" with an empty
# final message — indistinguishable, from the outside, from a planner that gave
# up. That is why runs died at 18 and at 27 steps: whenever the next payload
# (a multi-request ``run_trajectory_planner``, the tree assembly) crossed 4096.
_DEFAULT_MAX_OUTPUT_TOKENS = 32000


def max_output_tokens() -> Optional[int]:
    """The response-length ceiling to request. ``LLM_MAX_TOKENS`` overrides it.

    ``LLM_MAX_TOKENS=0`` (or an unparseable value) means "send nothing and let the
    provider decide", which restores the old inherited-default behaviour for a
    model whose ceiling this one exceeds.
    """
    raw = os.environ.get("LLM_MAX_TOKENS")
    if raw is None:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        return None
    return value if value > 0 else None


# The socket must outlive the generation the ceiling above now permits.
#
# botocore's default read timeout is 60 s, and it counts from the request, not from
# the last byte. A single planning step that actually uses the raised ceiling takes
# minutes to generate, so the two settings are coupled: raising maxTokens without
# raising this one converts silent truncation into
#
#     Read timeout on endpoint URL: ".../eu.anthropic.claude-opus-5/converse"
#
# which is what killed the 2026-09-08 run at 2025 s — the loop died on transport,
# not on reachability. Bedrock has no server-side cap that makes a long read wrong,
# so the timeout is set to the longest generation worth waiting for rather than the
# shortest one that usually works.
_DEFAULT_READ_TIMEOUT_S = 900
_DEFAULT_CONNECT_TIMEOUT_S = 30


def read_timeout_s() -> int:
    """Seconds to wait for the model's reply. ``LLM_READ_TIMEOUT_S`` overrides."""
    raw = os.environ.get("LLM_READ_TIMEOUT_S")
    if raw is None:
        return _DEFAULT_READ_TIMEOUT_S
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        return _DEFAULT_READ_TIMEOUT_S
    return value if value > 0 else _DEFAULT_READ_TIMEOUT_S


def describe_models() -> dict:
    """What a client can ask for. Backs the ``/a2a/models`` discovery endpoint."""
    provider = current_provider()
    return {
        "provider": provider,
        "default": DEFAULT_MODEL_ALIAS,
        "effective": resolve_model(),
        "planning_default": DEFAULT_PLANNING_MODEL_ALIAS,
        "planning_effective": planning_model(),
        "prompt_cache": prompt_cache_enabled(),
        "region": bedrock_region() if provider == "bedrock" else None,
        "max_output_tokens": max_output_tokens(),
        "read_timeout_s": read_timeout_s(),
        "aliases": {alias: normalize_model(alias, provider)
                    for alias in sorted(set(MODEL_ALIASES))},
        "note": ("Pass any alias, a full model id, or 'provider:model' as the "
                 "'model' argument of an A2A request or MCP tool call to "
                 "override this for one operation."),
    }


# ─── Construction ────────────────────────────────────────────────────────────

def build_llm(model: Optional[str] = None):
    """Build the LangChain chat model for this operation.

    The single construction point: every agent in the stack goes through here, so
    the model, the region and the provider are decided once.
    """
    provider, model_id = split_model(resolve_model(model))

    limit = max_output_tokens()

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse
        # ``config=`` rather than a bare region: the ceiling and the socket timeout
        # are one decision (see read_timeout_s). ``adaptive`` retries because a
        # throttle mid-run costs the whole plan, and a truncated ReAct loop cannot
        # be resumed from where it stopped.
        from botocore.config import Config as _BotoConfig
        return ChatBedrockConverse(
            model_id=model_id, region_name=bedrock_region(), max_tokens=limit,
            config=_BotoConfig(
                read_timeout=read_timeout_s(),
                connect_timeout=_DEFAULT_CONNECT_TIMEOUT_S,
                retries={"max_attempts": 4, "mode": "adaptive"},
            ),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, max_tokens=limit)
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_id)
    raise ValueError(f"Unknown LLM provider: {provider}")
