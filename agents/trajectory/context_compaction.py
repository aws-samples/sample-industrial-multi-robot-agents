"""Keep the planning model's context small: old turns are sent as digests.

The ReAct loop re-sends the WHOLE conversation on every turn. Measured on
welding_cell_2_small (bt_1793826556, 76 turns) the input grew ~5.2k tokens per turn, from
38k to 428k — every assembly result (~30k characters: the whole tree and every
step's gate report), every planner result's failure list, and the model's own
reasoning, all re-read on every later turn. A turn then takes 15-40 s.

The agent's ``prompt`` (:func:`compacting_prompt`) gives the model a COMPACTED copy
of the history, inside the model step (a ``pre_model_hook`` would be a graph node
of its own and cost each turn a third step of the recursion budget); the agent's
state keeps every message in full, so the tree
extraction, the trace and the live feed are unaffected. A message is compacted
once it is older than the last :data:`KEEP_RECENT` messages:

* a tool result keeps what a later decision can need — the tool, success,
  per-operation outcome and reason code, each arm's end joints (a re-plan from an
  explicit pose needs them), step-gate verdicts, the tree id, short errors and
  advice — and drops the rest (trees, blackboards, per-step reports, recovery
  tables, long texts);
* the model's own text is cut to :data:`AI_TEXT_CHARS`; its tool calls (names and
  arguments) are kept, so every tool result still answers a call;
* an assembly result is compacted as soon as a newer assembly exists, even inside
  the recent window (one is ~30k characters, and the newest says what matters).

The boundary moves in blocks of :data:`BLOCK` messages, not on every turn, so
the compacted prefix stays byte-identical between moves and the prompt cache
keeps hitting. ``CONTEXT_COMPACT=0`` sends the full history (the old behaviour).
"""
from __future__ import annotations

import json
import os
from typing import Any

KEEP_RECENT = 12        # messages at the end always sent in full
BLOCK = 8               # the compaction boundary moves this many messages at a time
AI_TEXT_CHARS = 300
TEXT_CHARS = 240

#: Tool-result fields kept in a digest (top level), besides the ones built below.
_KEEP_SCALARS = ("success", "complete", "ok", "valid", "chained", "collision_free",
                 "stopped_at", "planned_steps", "skipped_steps", "kept_steps_after_stop",
                 "min_clearance_m", "makespan_estimate_s", "cycle_time_balance",
                 "total_steps", "unreachable", "to_robot", "insert_after_operation_id",
                 "reason_code", "found", "cached", "operation_counts", "robots",
                 "serialised_steps", "serial_steps", "idle_arms")


def enabled() -> bool:
    return os.environ.get("CONTEXT_COMPACT", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _short(v: Any, n: int = TEXT_CHARS) -> Any:
    if isinstance(v, str) and len(v) > n:
        return v[:n] + "…"
    return v


def _failure(f: Any) -> Any:
    if not isinstance(f, dict):
        return _short(str(f))
    out = {k: f[k] for k in ("operation_id", "robot_ns", "reason_code") if f.get(k) is not None}
    if not out.get("reason_code") and f.get("error"):
        out["error"] = _short(f["error"], 120)
    return out


def _trajectories(t: Any) -> Any:
    if not isinstance(t, dict):
        return None
    out = {}
    for ns, leg in t.items():
        if isinstance(leg, dict) and ("end_joints" in leg or "waypoints" in leg):
            out[ns] = {k: leg[k] for k in ("end_joints", "duration_s") if k in leg}
    return out or None


def digest_result(data: Any) -> Any:
    """The parts of one tool result a LATER decision can still need."""
    if not isinstance(data, dict):
        return _short(data) if isinstance(data, str) else data
    out: dict = {}
    for k in _KEEP_SCALARS:
        if k in data and data[k] is not None:
            v = data[k]
            if isinstance(v, (list, dict)) and len(json.dumps(v, default=str)) > 400:
                v = f"<{type(v).__name__} of {len(v)}>"
            out[k] = v
    for k in ("error", "advice", "note", "warning", "serial_warning", "suggestion"):
        if data.get(k):
            out[k] = _short(data[k] if isinstance(data[k], str) else json.dumps(data[k], default=str))
    if isinstance(data.get("failures"), list) and data["failures"]:
        out["failures"] = [_failure(f) for f in data["failures"][:30]]
        if len(data["failures"]) > 30:
            out["failures_more"] = len(data["failures"]) - 30
    tr = _trajectories(data.get("trajectories"))
    if tr:
        out["trajectories"] = tr
    gate = data.get("step_gate")
    if isinstance(gate, dict):
        out["step_gate"] = {k: gate[k] for k in ("verdict", "order") if k in gate}
    if isinstance(data.get("steps"), list):
        steps = []
        for st in data["steps"]:
            if not isinstance(st, dict):
                continue
            s = {k: st[k] for k in ("step", "success") if k in st}
            if isinstance(st.get("step_gate"), dict):
                s["verdict"] = st["step_gate"].get("verdict")
            tr = _trajectories(st.get("trajectories"))
            if tr:
                s["trajectories"] = tr
            if st.get("failures"):
                s["failures"] = [_failure(f) for f in st["failures"][:10]]
            # An allocation step: which arm does which operation.
            for ns, op in st.items():
                if isinstance(op, dict) and op.get("id"):
                    s[ns] = op["id"]
            steps.append(s)
        out["steps"] = steps
    if isinstance(data.get("step_gates"), list):
        out["step_gates"] = [{k: r.get(k) for k in ("step", "mode") if k in r}
                             for r in data["step_gates"] if isinstance(r, dict)]
    tree = data.get("behaviour_tree")
    if isinstance(tree, dict):
        out["behaviour_tree"] = {"tree_id": tree.get("tree_id"),
                                 "note": "compacted in the model's view — the full tree "
                                         "is kept by the platform"}
    if isinstance(data.get("program_cost"), dict):
        out["program_cost"] = {k: data["program_cost"].get(k)
                               for k in ("cycle_time_s", "home_s", "mode")
                               if k in data["program_cost"]}
    if isinstance(data.get("barrier_free"), dict):
        out["barrier_free"] = {k: data["barrier_free"].get(k)
                               for k in ("used", "saved_s") if k in data["barrier_free"]}
    if data.get("content") and isinstance(data["content"], str):      # load_skill
        out["content"] = _short(data["content"], 160)
        out["skill"] = data.get("skill")
    out["compacted"] = True
    return out


def _compact_tool_content(content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return _short(text, 600)
    return json.dumps(digest_result(data), separators=(",", ":"), default=str)


def _compact_ai(msg):
    content = msg.content
    if isinstance(content, str):
        new = _short(content, AI_TEXT_CHARS)
    elif isinstance(content, list):
        # Bedrock / Anthropic content blocks: keep tool_use blocks, cut text, drop
        # reasoning blocks (their text is the bulk of the model's own output).
        new = []
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("text",):
                if b.get("text"):
                    new.append({**b, "text": _short(b["text"], AI_TEXT_CHARS)})
            elif isinstance(b, dict) and b.get("type") in ("reasoning_content", "thinking",
                                                           "redacted_thinking", "reasoning"):
                continue
            else:
                new.append(b)
        if not new:
            new = ""
    else:
        return msg
    if new == content:
        return msg
    return msg.model_copy(update={"content": new})


def boundary(n: int) -> int:
    """How many leading messages are compacted for a history of ``n``."""
    return max(0, (n - KEEP_RECENT) // BLOCK * BLOCK)


#: Tools whose newest result supersedes every earlier one (the latest assembly
#: says everything a refused earlier one did about the current program).
SUPERSEDED = ("assemble_behaviour_tree", "assemble_coordinated_tree")


def compact_messages(messages: list) -> list:
    """A copy of ``messages`` with everything before :func:`boundary` compacted —
    and, anywhere, every assembly result but the newest. The originals are not
    modified."""
    cut = boundary(len(messages))
    newest = {}
    for i, m in enumerate(messages):
        if getattr(m, "type", None) == "tool" and getattr(m, "name", None) in SUPERSEDED:
            newest[m.name] = i
    out = []
    for i, m in enumerate(messages):
        superseded = (getattr(m, "type", None) == "tool"
                      and getattr(m, "name", None) in SUPERSEDED and newest.get(m.name) != i)
        if i >= cut and not superseded:
            out.append(m)
            continue
        kind = getattr(m, "type", None)
        if kind == "tool":
            new = _compact_tool_content(m.content)
            out.append(m if new == m.content else m.model_copy(update={"content": new}))
        elif kind == "ai":
            out.append(_compact_ai(m))
        else:
            out.append(m)        # the task (human) message stays whole
    return out


def compact_for_model(state: dict) -> dict:
    """The history the MODEL sees, as ``{"llm_input_messages": [...]}``."""
    messages = list(state.get("messages") or [])
    if not enabled():
        return {"llm_input_messages": messages}
    return {"llm_input_messages": compact_messages(messages)}


def compacting_prompt(system_prompt: str):
    """A ``prompt`` callable for ``create_react_agent``: the system prompt, then the
    compacted history. It runs INSIDE the model step — a ``pre_model_hook`` is a
    graph node of its own and would cost every turn a third step of the recursion
    budget."""
    from langchain_core.messages import SystemMessage
    system = SystemMessage(content=system_prompt)

    def prompt(state) -> list:
        messages = state.get("messages") if isinstance(state, dict) else getattr(
            state, "messages", None)
        return [system, *compact_for_model({"messages": messages or []})["llm_input_messages"]]

    return prompt
