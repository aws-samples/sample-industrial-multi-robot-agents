"""The planning model sees old turns as digests; the agent keeps them whole.

Measured on welding_cell_2_small (bt_1793826556): the model's input grew ~5.2k tokens per
turn to 428k. ``context_compaction.compact_for_model`` (the agent's
``pre_model_hook``) compacts everything older than the last few messages, in
blocks so the prompt-cache prefix stays stable, without touching the state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

msgs = pytest.importorskip("langchain_core.messages")
cc = pytest.importorskip("agents.trajectory.context_compaction")

AIMessage, ToolMessage, HumanMessage = msgs.AIMessage, msgs.ToolMessage, msgs.HumanMessage

BIG_ASSEMBLY = {
    "success": False, "error": "these steps put two arms inside the warning band " * 20,
    "behaviour_tree": {"tree_id": "bt_1", "root": {"children": [{"x": "y" * 500}] * 40},
                       "blackboard": {f"traj_{i}": {"waypoints": 120} for i in range(40)},
                       "xml": "<root>" + "x" * 8000 + "</root>"},
    "step_gates": [{"step": i, "mode": "PARALLEL", "parallel": {"min_clearance_m": 0.3,
                                                                "pairs": ["p"] * 80}}
                   for i in range(12)],
    "program_cost": {"cycle_time_s": 104.4, "home_s": 8.9, "arms": {"robot1": {}}},
}
PLANNER = {
    "success": True, "stopped_at": 1,
    "steps": [{"step": 0, "success": True, "step_gate": {"verdict": "PARALLEL", "pairs": [1] * 50},
               "trajectories": {"robot1": {"waypoints": 120, "duration_s": 4.2,
                                           "start_joints": [0.0] * 6,
                                           "end_joints": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}}}],
    "failures": [{"operation_id": "W12", "robot_ns": "robot1", "reason_code": "no_path_found",
                  "error": "ompl-compound (RRTConnect): no joint path " * 10,
                  "recovery": {"levers": ["a"] * 10, "escalate": "reallocate"}}],
}


def _history(n_pairs: int, payload=PLANNER):
    out = [HumanMessage(content="plan these 27 welds")]
    for i in range(n_pairs):
        out.append(AIMessage(content="I will now plan step %d. " % i + "reasoning " * 200,
                             tool_calls=[{"name": "run_trajectory_planner", "args": {"i": i},
                                          "id": f"call_{i}"}]))
        out.append(ToolMessage(content=json.dumps(payload), tool_call_id=f"call_{i}",
                               name="run_trajectory_planner"))
    return out


def test_recent_messages_and_the_task_are_sent_whole():
    h = _history(20)
    out = cc.compact_messages(h)
    cut = cc.boundary(len(h))
    assert cut > 0 and len(out) == len(h)
    assert out[0].content == h[0].content, "the task message is never compacted"
    assert all(a is b for a, b in zip(out[cut:], h[cut:])), "recent messages untouched"
    assert len(h) - cut >= cc.KEEP_RECENT


def test_the_state_is_never_modified():
    h = _history(20)
    before = [m.content for m in h]
    cc.compact_for_model({"messages": h})
    assert [m.content for m in h] == before


def test_every_tool_result_still_answers_its_call():
    out = cc.compact_messages(_history(20))
    calls = {c["id"] for m in out if m.type == "ai" for c in m.tool_calls}
    answers = {m.tool_call_id for m in out if m.type == "tool"}
    assert calls == answers
    assert all(m.tool_calls for m in out if m.type == "ai"), "tool calls are kept"


def test_a_digest_keeps_what_a_later_decision_needs():
    d = json.loads(cc._compact_tool_content(json.dumps(PLANNER)))
    assert d["compacted"] is True and d["success"] is True and d["stopped_at"] == 1
    assert d["failures"] == [{"operation_id": "W12", "robot_ns": "robot1",
                              "reason_code": "no_path_found"}]
    assert d["steps"][0]["verdict"] == "PARALLEL"
    assert d["steps"][0]["trajectories"]["robot1"]["end_joints"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], \
        "end joints are what a re-plan from an explicit pose needs"
    a = json.loads(cc._compact_tool_content(json.dumps(BIG_ASSEMBLY)))
    assert a["behaviour_tree"]["tree_id"] == "bt_1" and "root" not in a["behaviour_tree"]
    assert a["step_gates"][3] == {"step": 3, "mode": "PARALLEL"}
    assert a["program_cost"]["cycle_time_s"] == 104.4
    assert len(json.dumps(a)) < len(json.dumps(BIG_ASSEMBLY)) / 20


def test_the_boundary_moves_in_blocks_so_the_cache_prefix_is_stable():
    seen = {cc.boundary(n) for n in range(cc.KEEP_RECENT, cc.KEEP_RECENT + cc.BLOCK)}
    assert seen == {0}, "a boundary must not move on every new message"
    a = cc.compact_messages(_history(30))
    b = cc.compact_messages(_history(30)[: len(_history(30)) - 1])
    cut = min(cc.boundary(len(a)), cc.boundary(len(b)))
    assert [m.content for m in a[:cut]] == [m.content for m in b[:cut]]


def test_old_reasoning_is_cut_and_reasoning_blocks_dropped():
    m = AIMessage(content=[{"type": "reasoning_content", "reasoning_content": {"text": "x" * 5000}},
                           {"type": "text", "text": "y" * 2000}],
                  tool_calls=[{"name": "t", "args": {}, "id": "c1"}])
    out = cc._compact_ai(m)
    assert [b["type"] for b in out.content] == ["text"]
    assert len(out.content[0]["text"]) <= cc.AI_TEXT_CHARS + 1
    assert out.tool_calls == m.tool_calls


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("CONTEXT_COMPACT", "0")
    h = _history(20)
    assert cc.compact_for_model({"messages": h})["llm_input_messages"] == h


def test_on_the_real_welding_cell_2_small_trace_the_input_shrinks():
    """Rebuild bt_1793826556's conversation from its trace and measure."""
    path = PROJECT_ROOT / "cells/welding_cell_2_small/agentic_planner/logs/plantrace_bt_1793826556_558e7c_welding_cell_2_small.json"
    if not path.exists():
        pytest.skip("trace not present")
    trace = json.loads(path.read_text())
    h = [HumanMessage(content="plan")]
    for i, x in enumerate(d for d in trace["decisions"] if "tool" in d):
        h.append(AIMessage(content=x.get("why") or "", tool_calls=[
            {"name": x["tool"], "args": x.get("args") or {}, "id": f"c{i}"}]))
        h.append(ToolMessage(content=json.dumps((x.get("result") or {}).get("payload")),
                             tool_call_id=f"c{i}", name=x["tool"]))
    # What the model is SENT, summed over every turn of the run (each turn
    # re-sends the history so far).
    full = small = 0
    for n in range(2, len(h) + 1, 2):
        part = h[:n]
        full += sum(len(str(m.content)) for m in part)
        small += sum(len(str(m.content)) for m in cc.compact_messages(part))
    assert small < full * 0.4, (full, small)


def test_only_the_newest_assembly_result_is_sent_whole():
    h = _history(2)
    for i in range(3):
        h.append(AIMessage(content="", tool_calls=[{"name": "assemble_behaviour_tree",
                                                    "args": {}, "id": f"a{i}"}]))
        h.append(ToolMessage(content=json.dumps(BIG_ASSEMBLY), tool_call_id=f"a{i}",
                             name="assemble_behaviour_tree"))
    out = cc.compact_messages(h)
    asm = [m for m in out if m.type == "tool" and m.name == "assemble_behaviour_tree"]
    assert [json.loads(m.content).get("compacted") for m in asm] == [True, True, None]


def test_the_prompt_callable_sends_the_system_prompt_then_the_compacted_history():
    prompt = cc.compacting_prompt("SYSTEM")
    h = _history(20)
    out = prompt({"messages": h})
    assert out[0].type == "system" and out[0].content == "SYSTEM"
    assert [m.content for m in out[1:]] == [m.content for m in cc.compact_messages(h)]


def test_compaction_adds_no_graph_step():
    """A pre_model_hook is its own node: 3 recursion steps per turn instead of 2,
    which stopped welding_cell_2_small at turn 53 of a 160-step budget."""
    src = (PROJECT_ROOT / "agents/trajectory/agent.py").read_text()
    assert "pre_model_hook=" not in src.split("def create_trajectory_agent", 1)[1].split("\ndef ", 1)[0]
    assert "prompt=compacting_prompt(" in src
