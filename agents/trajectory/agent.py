"""
Trajectory Agent — specialized LangGraph agent.

Responsibilities:
  1. Receives a planning request (cell_id + operation points)
  2. Activates skills progressively: optimize-sequence → plan-trajectories
     → find-interlocks (when needed) → build-behaviour-tree
  3. Returns a BehaviourTree JSON

Built as a proper LangGraph agent with skills loaded from skills/trajectory/.
Skills follow the Agent Skills specification (SKILL.md with YAML frontmatter).
The LLM decides WHICH skill to activate and WHEN — the scripts in each skill
directory execute the deterministic math.

Skills use progressive disclosure:
  - At startup: only name + description loaded into system prompt
  - On activation: full SKILL.md body loaded
  - On demand: scripts/references/assets loaded as needed
"""
from __future__ import annotations

import os
import sys
import time as _time
from datetime import datetime as _datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from agents.llm_model import (
    bind_model, bind_tools_for_agent, build_llm, effective_model_id, planning_model,
    resolve_model,
)
from agents.trajectory.tools.pinocchio import (
    run_ik, run_fk, check_self_collision, get_link_spheres, validate_approach,
)
from agents.trajectory.tools import plan_cancellation
from agents.trajectory.tools.program_eval import evaluate_program, find_clear_pose
from agents.trajectory.tools.planning_tools import (
    run_mrta_pipeline, run_trajectory_planner, run_path_shortcut,
    reallocate_operation,
    validate_trajectory_pair, validate_trajectory_set,
    find_interlocks, find_interlock_set,
)
from agents.trajectory.tools.behaviour_tree import (
    build_behaviour_tree_node, validate_behaviour_tree, emit_parallel_node,
    assemble_behaviour_tree, chain_trajectories, export_behaviour_tree_xml,
)
from agents.trajectory.tools.knowledge_base import (
    store_trajectory, get_warm_start, get_similar_trajectories,
)
from utils.llm_text import extract_text as _extract_text

from utils import skill_registry
from agents.trajectory.tools.coordinated_tree import (
    assemble_coordinated_tree, coordinate_program,
)


# ─── Reasoning / action trace logging ────────────────────────────────────────

def _compact(obj, max_list: int = 2, _depth: int = 0):
    """
    Shrink a value for logging: collapse long lists (e.g. trajectory point
    arrays with hundreds of entries) and round floats, keeping the meaningful
    structure (success flags, ids, counts, targets) readable.
    """
    if _depth > 4:
        return "…"
    if isinstance(obj, dict):
        return {k: _compact(v, max_list, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        if n > max_list:
            shown = [_compact(x, max_list, _depth + 1) for x in obj[:1]]
            return shown + [f"…+{n - 1} more"]
        return [_compact(x, max_list, _depth + 1) for x in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    return obj


def _brief(obj, limit: int = 300) -> str:
    """JSON-encode a compacted object and truncate to a log-friendly length."""
    import json
    try:
        s = json.dumps(_compact(obj), default=str, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def _summarize_tool_result(name: str, text: str) -> str:
    """Pull the headline fields out of a tool's JSON result for a concise log."""
    import json
    try:
        data = json.loads(text)
    except Exception:
        return text[:200] + ("…" if len(text) > 200 else "")
    if not isinstance(data, dict):
        return _brief(data, 200)

    # Highlight the fields operators care about, per tool.
    if "success" in data:
        head = "ok" if data.get("success") else f"FAILED: {data.get('error', '?')}"
        extra = {}
        fl = [f for f in (data.get("failures") or []) if isinstance(f, dict)]
        if fl:
            # What failed and why, by code: a planner result carries no top-level
            # `error`, so this line used to read "FAILED: ?" for every failure.
            extra["failures"] = [
                f"{f.get('operation_id') or f.get('robot_ns')}:{f.get('reason_code') or 'error'}"
                + (f"<-{','.join(f['blocked_by'])}" if f.get("blocked_by") else "")
                for f in fl[:6]] + ([f"+{len(fl) - 6} more"] if len(fl) > 6 else [])
            if not data.get("success") and not data.get("error"):
                head = "FAILED:"
        if "trajectories" in data and isinstance(data["trajectories"], dict):
            extra["robots"] = {
                r: f"{len(v) if isinstance(v, list) else v.get('waypoints')} pts"
                for r, v in data["trajectories"].items()
                if isinstance(v, (list, dict))
            }
        if isinstance(data.get("steps"), list):
            extra["steps"] = len(data["steps"])
            extra["stopped_at"] = data.get("stopped_at")
        if isinstance(data.get("step_gate"), dict):
            extra["gate"] = data["step_gate"].get("verdict")
        for k in ("planning_time_s", "position_error_m", "collision_free",
                  "min_clearance_m", "needs_interlock", "strategy"):
            if k in data:
                extra[k] = _compact(data[k])
        return head + (f" {_brief(extra, 200)}" if extra else "")
    for key in ("total_steps", "found", "reachable", "valid", "ee_position"):
        if key in data:
            return _brief({k: data[k] for k in data if k in
                           ("total_steps", "found", "reachable", "valid",
                            "ee_position", "joint_positions")}, 220)
    return _brief(data, 220)


def _log_agent_message(msg, counters: dict) -> None:
    """
    Print the agent's reasoning, the tool it decided to call (the 'why' + 'what'),
    and each tool result — so the ReAct loop is visible and meaningful in the
    container console via `make logs`.
    """
    kind = getattr(msg, "type", "") or msg.__class__.__name__.lower()

    if "ai" in kind:  # AIMessage — reasoning + tool decisions
        text = _extract_text(getattr(msg, "content", "")).strip()
        if text:
            print(f"[plan] think: {text}", flush=True)
        # Per-turn token use, so "where does the time go" is answerable from the
        # log: input size per turn and how much of it the prompt cache served.
        usage = getattr(msg, "usage_metadata", None) or {}
        if usage:
            details = usage.get("input_token_details") or {}
            print(f"[plan] llm: in={usage.get('input_tokens')} "
                  f"cache_read={details.get('cache_read', 0)} "
                  f"cache_write={details.get('cache_creation', 0)} "
                  f"out={usage.get('output_tokens')}", flush=True)
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            counters["step"] = counters.get("step", 0) + 1
            print(
                f"[plan] step {counters['step']:>2} -> {tc.get('name')} "
                f"{_brief(tc.get('args', {}), 240)}",
                flush=True,
            )
        # A message with neither text nor a tool call ENDS the ReAct loop, and it
        # used to print nothing at all — the run just stopped mid-plan with no
        # trace of why. Say so, with the provider's stop reason and the block kinds
        # that did arrive, because that is the whole diagnosis: stopReason
        # 'max_tokens' with only a reasoning_content block means the reply was cut
        # off inside a tool call, not that the agent decided it was finished.
        if not text and not tool_calls:
            meta = getattr(msg, "response_metadata", None) or {}
            content = getattr(msg, "content", "")
            blocks = [
                (b.get("type") or ",".join(b)) if isinstance(b, dict) else type(b).__name__
                for b in (content if isinstance(content, list) else [])
            ]
            print(
                "[plan] !! the model returned NOTHING to act on — the loop ends here. "
                f"stopReason={meta.get('stopReason') or meta.get('stop_reason') or '?'} "
                f"blocks={blocks or 'none'} "
                f"invalid_tool_calls={len(getattr(msg, 'invalid_tool_calls', None) or [])} "
                f"usage={_brief(getattr(msg, 'usage_metadata', None) or {}, 120)}",
                flush=True,
            )
    elif "tool" in kind:  # ToolMessage — result of a tool call
        name = getattr(msg, "name", "?")
        raw = _extract_text(getattr(msg, "content", "")).strip()
        print(f"[plan]         <- {name}: {_summarize_tool_result(name, raw)}", flush=True)


#: (trace_id, tool_call_id) -> (operation ids, robots) the call asked about, until
#: its result arrives.
_CALL_OPS: dict[tuple, tuple] = {}


def _publish_events(cell_id: str, trace_id: str | None, msg) -> None:
    """Mirror one agent message onto the run's live event stream.

    The console log above is for a human reading `make logs`; this is the same
    information as *data*, so the dashboard can show "W14 started", "W14 FAILED",
    "strategy change on W14: approach_offset_m 0.1 → 0.07" while the run is still
    going. Both are driven from the same message so they cannot disagree.

    Identity is read with the trace module's walkers rather than by knowing each
    tool's argument shape, because that shape differs per tool and a feed that
    only understands two of them goes quiet exactly when the run gets interesting.

    Never raises: a live feed is a convenience, and the planner is not.
    """
    if not trace_id:
        return
    try:
        from agents.trajectory import plan_events, plan_trace

        kind = getattr(msg, "type", "") or msg.__class__.__name__.lower()

        if "ai" in kind:
            text = _extract_text(getattr(msg, "content", "")).strip()
            if text:
                plan_events.thinking(cell_id, trace_id, text)
            for call in getattr(msg, "tool_calls", None) or []:
                args = call.get("args") or {}
                tool = call.get("name")
                robots = plan_trace.robot_names(args)
                ops = plan_trace.operation_ids(args)
                if call.get("id") and ops:
                    _CALL_OPS[(trace_id, call["id"])] = (ops, robots)
                for op in ops:
                    # Order matters: the strategy change is announced BEFORE the
                    # attempt it explains, so the feed reads as "changed the
                    # standoff, then retried W14" rather than the reverse.
                    plan_events.note_attempt(cell_id, trace_id, op, args,
                                             tool=tool,
                                             robot=robots[0] if robots else None)
                    plan_events.operation_started(
                        cell_id, trace_id, op, tool=tool,
                        robot=robots[0] if robots else None,
                        args={k: v for k, v in args.items()
                              if k in plan_events._STRATEGY_ARGS})
                if not ops:
                    plan_events.emit(cell_id, trace_id, "tool_started", tool=tool,
                                     robots=robots, text=f"{tool}")

        elif "tool" in kind:
            raw = _extract_text(getattr(msg, "content", "")).strip()
            payload = plan_trace._parse_result(raw)
            tool = getattr(msg, "name", None)
            ok = plan_trace._result_ok(payload)
            metrics = plan_trace._result_metrics(payload)
            robots = plan_trace.robot_names(payload)
            failed_ids = {
                f.get("operation_id"): (f.get("error") or f.get("reason"))
                for f in ((payload.get("failures") or []) if isinstance(payload, dict) else [])
                if isinstance(f, dict)
            }
            ops = plan_trace.operation_ids(payload)
            asked = _CALL_OPS.pop((trace_id, getattr(msg, "tool_call_id", None)), None)
            if asked and ok is not None:
                # A successful planner result names robots, not operations — only
                # its failures carry an operation_id — so without the call's own
                # ids a solved weld never reached the feed as "planned".
                ops = ops + [op for op in asked[0] if op not in ops]
                robots = robots or asked[1]
            for op in ops:
                op_ok = False if op in failed_ids else ok
                plan_events.operation_result(
                    cell_id, trace_id, op, ok=op_ok, tool=tool,
                    robot=robots[0] if robots else None, metrics=metrics,
                    error=failed_ids.get(op) or (
                        (payload.get("error") if isinstance(payload, dict) else None)
                        if op_ok is False else None),
                )
            if not ops and ok is False and isinstance(payload, dict):
                plan_events.emit(cell_id, trace_id, "tool_failed", tool=tool,
                                 error=payload.get("error"),
                                 text=f"{tool} failed: {payload.get('error')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[plan] live event skipped ({exc})", flush=True)


#: Tools whose result carries the plan's final tree.
_ASSEMBLERS = ("assemble_behaviour_tree", "assemble_coordinated_tree")


def _reassemble(messages, result_msg) -> dict | None:
    """Rebuild an accepted tree whose full version this process never held.

    A run resumed after it had assembled (a new process: the token expired, the
    container was recreated) has the tree only as the summary the model saw. The
    trajectories it was built from ARE restored with the checkpoint, so the same
    assembler call — the arguments are in the conversation — is run again; its
    result is used only if it is accepted again. Deterministic: the assembler
    plans nothing new except the return home, which it gates as before.
    """
    call_id = getattr(result_msg, "tool_call_id", None)
    name = getattr(result_msg, "name", None)
    for msg in reversed(list(messages)):
        for call in getattr(msg, "tool_calls", None) or []:
            if call.get("id") != call_id or call.get("name") != name:
                continue
            try:
                from agents.trajectory.tools.behaviour_tree import assemble_behaviour_tree
                from agents.trajectory.tools.coordinated_tree import assemble_coordinated_tree
                tool = {"assemble_behaviour_tree": assemble_behaviour_tree,
                        "assemble_coordinated_tree": assemble_coordinated_tree}[name]
                out = getattr(tool, "func", tool)(**(call.get("args") or {}))
            except Exception as exc:  # noqa: BLE001
                print(f"[plan] !! re-assembling the accepted tree failed: {exc}", flush=True)
                return None
            tree = out.get("behaviour_tree") if isinstance(out, dict) else None
            if out.get("success") is True and isinstance(tree, dict) and "root" in tree \
                    and not _has_summarised_legs(tree):
                print("[plan] re-assembled the accepted tree from the restored "
                      "trajectories (its full version was lost with the old process)",
                      flush=True)
                return tree
            print("[plan] !! re-assembly did not accept the tree again — not storing it",
                  flush=True)
            return None
    return None


def _extract_behaviour_tree(messages) -> dict | None:
    """
    Find the most recent ACCEPTED BehaviourTree in the message history.

    A tree is only harvested when the assembler that produced it reported
    ``success: true``. ``assemble_behaviour_tree`` returns the tree alongside
    ``success: false`` when it refuses one (a broken weld-to-weld chain, a
    duplicate operation_id, a failed structural validation), so accepting any dict
    carrying a "root" key would smuggle a rejected tree out as the plan and defeat
    the gap-refusal guarantee. Rejected trees are skipped here; an older accepted
    tree, or None, is returned instead.

    The hand-built fallback paths are only consulted when no assembler result
    exists at all, since a tree the LLM assembled by hand has passed no
    continuity check.
    """
    import json
    import re

    def _valid(obj) -> bool:
        return isinstance(obj, dict) and "root" in obj

    msgs = list(messages)

    # 0) Result of the one-shot assembler (the standard path). assemble_behaviour_tree
    #    returns {"success", "behaviour_tree": {tree_id, root, metadata}, ...} as a
    #    ToolMessage; that embedded tree is authoritative — but ONLY if accepted.
    saw_assembler = False
    for msg in reversed(msgs):
        if getattr(msg, "name", None) not in _ASSEMBLERS:
            continue
        text = _extract_text(getattr(msg, "content", "")).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            saw_assembler = True
            tree = obj.get("behaviour_tree")
            if _valid(tree) and obj.get("success") is True:
                # What the model saw is agent_view(tree): the real one, or None.
                full = _full_tree(tree)
                if full is None:
                    full = _reassemble(msgs, msg)
                return full
            # Rejected: keep scanning further back for an accepted assembly.

    # If the assembler ran and refused every tree, the run has NO valid plan.
    # Falling through to the hand-built paths would return unvalidated motion.
    if saw_assembler:
        return None

    # 1) A tree the agent hand-built and validated (fallback manual path, only
    #    reachable when the assembler never ran).
    for msg in reversed(msgs):
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") in ("validate_behaviour_tree", "build_behaviour_tree_node"):
                args = tc.get("args") or {}
                for cand in (args.get("tree"), args.get("behaviour_tree"), args):
                    if _valid(cand):
                        return cand

    # 2) Any tree_id JSON blob in message or tool-result content (most recent).
    #    Handles both a bare tree and an assembler-style wrapper carrying it.
    for msg in reversed(msgs):
        text = _extract_text(getattr(msg, "content", ""))
        if "tree_id" not in text:
            continue
        m = re.search(r'\{.*"tree_id".*\}', text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                # A wrapper that explicitly reports failure is a rejected tree.
                if obj.get("success") is False:
                    continue
                if _valid(obj) or "tree_id" in obj:
                    return obj
                inner = obj.get("behaviour_tree")
                if _valid(inner):
                    return inner
    return None

def _skill_catalog(cell_id: str | None = None) -> str:
    """
    The skill catalog (priority + name + description) for progressive disclosure:
    only this goes in the system prompt; the LLM calls load_skill(name) to read a
    skill's full instructions on demand. Ordered by the cell's effective priority
    (``utils/skill_registry.py``); skills the cell disabled are left out.
    """
    try:
        return skill_registry.catalog_text("trajectory", cell_id)
    except Exception:  # noqa: BLE001 — the prompt must still build
        return skill_registry.catalog_text("trajectory", None)


@tool
def load_skill(skill_name: str) -> dict:
    """
    Load the full instructions (the SKILL.md body) for a trajectory skill by
    name, so you can follow its steps. ACTIVATE a skill with this before acting
    on a decision it governs — e.g. load_skill('approach-strategy') when a target
    may be enclosed by a box obstacle, or load_skill('find-interlocks') for
    robot-vs-robot collisions. Returns the skill's markdown instructions.
    """
    cell_id = skill_registry.current_cell()
    agent = _skill_agent()
    try:
        skill = skill_registry.get_skill(agent, skill_name, cell_id)
    except skill_registry.SkillError:
        available = [s["name"] for s in skill_registry.list_skills(
            cell_id, agent, include_disabled=False)]
        return {"error": f"skill '{skill_name}' not found", "available": available}
    if not skill["enabled"]:
        return {"error": f"skill '{skill_name}' is disabled for cell {cell_id}"}
    return {"skill": skill["name"], "priority": skill["priority"],
            "content": skill["content"]}


SYSTEM_PROMPT = f"""\
You are the Trajectory Agent for a multi-robot industrial cell.

Your responsibility: given an operation list (welds, rivets, glue paths, laser
cuts, marks) and a cell configuration, produce a collision-free Behaviour Tree
that the cell agent can execute.

You work by ACTIVATING SKILLS. Each skill has detailed step-by-step
instructions. You are shown only skill names + descriptions below; call
`load_skill(name)` to read a skill's full instructions BEFORE following it.
Skills are listed highest priority first. When more than one skill applies to
a decision, follow the higher-priority one; a lower-priority skill never
overrides a safety gate (collision validation) that another skill requires.

AVAILABLE SKILLS:
{_skill_catalog()}

COMPLETION CONTRACT — read this first, it overrides any urge to over-analyse.
Your ONE deliverable is an assembled, validated BehaviourTree, and you MUST end
the run by calling `assemble_behaviour_tree` and return its tree. Ending without
a tree is a FAILURE, even when some welds are hard. The trajectories YOU plan are
FINAL — no pass runs after you to fix or re-chain them — so you gate them
collision-free AND chained (each operation continuous with the arm's previous
one) yourself, but you do it decisively, not by exhaustive probing:
  • Do NOT pre-screen welds with `run_ik` / `check_self_collision` before
    planning. `run_trajectory_planner` ALREADY runs IK + endpoint-collision + the
    approach-clearance check internally. Call it per step FIRST. Use
    `check_self_collision` only to DIAGNOSE a failure the planner reported.
  • The planner result carries per-robot `failures` with a `reason_code` and, for
    collisions, the object id + world point. FEED THAT BACK INTO YOUR NEXT MOVE:
    `load_skill('recover-from-planner-failure')` and apply the recovery for that
    reason_code (intermediate waypoint, wait / start_delay, move the other arm
    out, reorient, larger budget), then RE-CALL the planner. This is a closed
    loop — the planner reports why it failed, you decide the fix.
  • Spend at most ~2 recovery attempts on a problem weld WITH ITS CURRENT ROBOT.
    If it still cannot be planned, REDISTRIBUTE it: call `reallocate_operation`
    (cell_id, the operation, tried_robots=[every arm that has failed it], robots=
    your current per-robot execution order) and re-plan it on the arm it names.
    An operation is UNREACHABLE only once EVERY arm in the cell has actually
    failed it, or `reallocate_operation` refuses (`robot_locked` /
    `out_of_reach_everywhere`). Only then exclude it, note it in your final
    summary with the per-robot reasons, and move on — never let one corner weld
    consume the whole budget or stop you from assembling the rest.

CONTRACT WITH THE MOTION PLANNER (OMPL):
`run_trajectory_planner` is a point-to-point servo: from each arm's CURRENT
joints it reaches the requested target — optionally at a required tool
orientation — collision-free, or returns a structured failure. It makes NO task
decisions; YOU do (which welds, what order, which robot, the standoff distance,
the approach speed, and every recovery). When you submit several robots in ONE
call, the planner makes each later arm avoid the swept volume of the arms already
planned in that call, so a PARALLEL step comes back mutually collision-free.

STANDARD WORKFLOW:
1. `run_mrta_pipeline` → allocate operations to robots + compute optimal order.
   ROBOT ALLOCATION IS DECIDED HERE, NOT IN THE TASK. The task lists weld/other
   points but does not say which robot does each one. The pipeline distributes
   them. The only exception is an operation carrying an optional `assigned_robot`
   field: that operation is LOCKED to the named robot (single-robot operations
   like part handling or gluing). Never infer a robot from an operation's id or
   position — let the pipeline allocate the unassigned ones.
   That allocation is a FIRST GUESS from geometry alone — it runs before any IK or
   collision check. It is yours to REVISE: when an arm cannot actually reach one of
   its welds, hand that weld to another arm with `reallocate_operation` (step 4).
   Changing the distribution is a normal planning move, not a last resort.
2. FOR EACH STEP, call `run_trajectory_planner` ONCE with that step's motion
   requests. Put BOTH arms of a PARALLEL step in the SAME call — that is what
   makes the planner couple them (each avoids the other's swept volume).
   Each motion request has: `robot_ns`; a UNIQUE `operation_id`; `target` x/y/z;
   `approach_vector`; `operation_type`; `orientation` = the op's
   `weld_orientation` (else its `orientation`); `constrain_orientation` = true
   whenever an orientation is specified; `approach` = true for the terminal
   working stroke; and optional `current_joints` (omit it to chain automatically
   from the arm's previous goal).
   TWO PER-OPERATION parameters are yours to choose, set INSIDE each motion
   request (not on the call), so two arms in the same PARALLEL call may differ:
     • `approach_offset_m` — the standoff where normal transit hands over to the
       orientation-enforced approach. Shorter for a weld in a tight pocket,
       longer for an open seam where you want a straight run-in.
     • `approach_velocity_scale` — the speed of that terminal stroke alone.
   The call-level `velocity_scale` sets the TRANSIT speed. OMIT it: transit then
   runs at the cell's `planning.max_velocity_scaling`, which is also a ceiling
   (anything higher is clamped). Pass it only to go SLOWER than the cell. Omit
   `approach_velocity_scale` and the approach defaults to a fraction of transit.
   The planner INTERNALLY moves transit (normal clearance, transit speed) → an
   approach waypoint `approach_offset_m` back along `approach_vector` → the
   terminal descent into the point, straight along `approach_vector`, at the
   reduced approach speed and the required lean (the SAME collision clearance as
   transit — there is no reduced approach radius). The tool's spin about its own
   axis and a small tilt (the cell's `tool_orientation`) are searched for you when
   the specified pose folds the arm into itself; a failure then carries
   `orientation_search` and, when a shorter stroke works, `suggested_approach_offset_m`. So one call per weld covers the whole
   approach — you do NOT hand-build segments or probe IK yourself. Each returned
   waypoint is tagged `phase: "transit"|"approach"`. It caches each trajectory by
   `operation_id`.
   YOU choose, per weld (guided by `plan-trajectories` / `approach-strategy`):
     • `approach_offset_m` — where normal motion switches to the approach: larger
       when the point is open, smaller in a tight pocket.
     • `approach_velocity_scale` — slow the final stroke down for the working
       motion (transit speed is the cell's; leave it alone).
3. VERIFY each step before you move on (these gates are enforced):
   • PARALLEL step → its `step_gate` (returned with every planned step) IS the
     inter-arm check — the same one `assemble_behaviour_tree` applies — run against
     the step's arms AND the arms parked where they last stopped. PARALLEL: done,
     no separate call. SEQUENTIAL / REFUSED: resolve it now (below). Call
     `validate_trajectory_set` (`trajectories={{robot_ns: traj, ...}}`) yourself
     only for a set you built by hand (e.g. with `start_delays`), with EVERY arm
     in it.
     If it reports a collision, `load_skill('find-interlocks')` and use
     `find_interlock_set` with all the converging arms — never leave an arm out.
     Do not reason about clearance from the weld POSITIONS: two arms working opposite
     sides of the part are furthest apart exactly where each is most extended toward
     the other, so distance between weld points runs OPPOSITE to arm clearance
     (measured -0.607 on `welding_cell_2_small`). Only the swept-volume check answers it —
     the pair that e-stopped the cell was clear at both its start and goal poses.
   • Every weld's working stroke is checked by the PLATFORM: the assembler runs
     `validate_approach` on each weld leg against its own orientation, point and
     approach vector, and refuses a mismatch (`approach_mismatch` in
     `leg_refusals`). You do not need to call it per weld; use it to DIAGNOSE.
     NEVER drop or relax the target orientation to dodge a collision — change the
     PATH, not the working pose. The spin about the tool axis is the pose's one
     free parameter and the planner already searches it; set `tool_spin_deg` only
     to turn the tool clear of the OTHER arm, which that search does not consider.
4. ON ANY PLANNER FAILURE (a `failures` entry with a `reason_code`) →
   `load_skill('recover-from-planner-failure')` and apply the recovery for that
   reason_code (intermediate waypoint / wait / move the other arm out / reorient
   / larger budget), then re-call `run_trajectory_planner`. Cap at ~2 attempts on
   THAT arm, then REDISTRIBUTE with `reallocate_operation` and plan it on the arm
   it names — appended as that arm's last operation, so nothing already planned
   changes. Record the weld unreachable only when every arm has failed it or the
   reallocation is refused.
5. CHAIN THE PIECES — once every operation is planned, call
   `chain_trajectories(cell_id, robots)` with each robot's `operation_ids` IN
   EXECUTION ORDER. It does the chaining bookkeeping FOR you: it confirms every
   operation starts exactly where that arm's PREVIOUS operation ended (no
   teleport between welds) and, on a `break`, tells you the exact piece to fix.
   Re-plan the named `to_operation_id` with `current_joints` = the reported
   `end_joints` of `from_operation_id` (keep ops in execution order, one op per
   robot per call), then chain again. `assemble_behaviour_tree` runs the SAME
   check and REFUSES a tree with a gap, so resolve every break here.
6. `assemble_behaviour_tree(cell_id, plan)` → build the ENTIRE tree in ONE call.
   Pass only a compact decision `plan` — a list of `steps`, each with a `mode`
   (PARALLEL | SEQUENTIAL) and a `robots` list where each entry has `robot_ns`,
   `operation_id`, `operation_type`, plus optional gating fields (`start_delay_s`,
   `has_interlock`, `depends_on_robots`, `depends_on_positions`). Keep a step
   PARALLEL with both arms whenever their pair validated collision-free — that is
   how cycle time stays low. The tool pulls each cached trajectory by
   `operation_id`, builds the BehaviorTree.CPP nodes (`Parallel` / `Sequence` /
   `Delay` / `ExecuteTrajectory` / `WaitForRobot` / `MoveHome` / `CheckSafety`) and
   validates the tree. Do NOT pass trajectory arrays. Do NOT hand-assemble with
   `build_behaviour_tree_node` / `emit_parallel_node` unless the assembler reports
   a problem (e.g. a `missing` cache entry → re-run `run_trajectory_planner` with
   that exact `operation_id`).
   The assembler REFUSES a tree — `success: false` — when the plan reuses an
   `operation_id` (`duplicates`), when an op does not start where that arm's
   previous op ended (`breaks`), when a trajectory is not cached (`missing`), when a
   step lists the same arm twice (`repeated_arms` — split it into consecutive steps),
   or when a step's arms cannot move together AND no order moves them one at a time
   clear of each other (`blocked_steps`). A refused tree is NOT returned as the plan,
   so fix the cause and assemble again; never treat a refusal as done.
   Your step `mode` is a REQUEST. The assembler decides: it turns a step whose pair
   breaches into a `Sequence` (listed in `serialised_steps`) and reports every step's
   measured verdict in `step_gates`. For a `blocked_steps` entry, re-pair those
   operations with a DIFFERENT arm's operation, or give each its own step — do not
   re-pair by picking positions further apart, which is the -0.607 mistake above.

PRINCIPLES:
- Maximise parallel execution — both robots moving simultaneously reduces cycle time.
- Minimise interlocks — use temporal delays (start_delay_s) before spatial gates.
- Use warm-start joints from `get_warm_start` when available.
- All math is deterministic — use the tools, do not generate math yourself.

COMPLETION REQUIREMENT (do not stop early):
You are NOT finished until you have:
  (a) planned a trajectory (each with its own `operation_id`) for EVERY
      operation in the list,
  (a2) re-planned any weld the assembler reported in `leg_refusals`
      (`approach_mismatch`, `trajectory_self_collision`, `joint_limit_violation`)
      until it assembles,
  (a3) confirmed weld-to-weld continuity with `chain_trajectories` returning
      `chained=true` for every robot (re-planning any broken seam until it does),
  (a4) called `evaluate_program(cell_id, plan)` on the plan and acted on its
      `suggestions` where a faster VALID program is within reach (idle arms,
      imbalance, a return home that cannot run together) — the objective is
      the shortest cycle time, never at the cost of a gate. Every accepted
      `assemble_behaviour_tree` result also carries `program_cost` (cycle
      time, balance, tight steps, return home): read it, and re-plan and
      re-assemble once when its suggestions point to a clearly faster program,
  (b) called `assemble_behaviour_tree(cell_id, plan)` ONCE to build and validate
      the full tree, and
  (c) returned that complete BehaviourTree JSON — the `behaviour_tree` field of
      the assembler result — as your final message. It is a BehaviorTree.CPP v4
      tree: `tree_id`, `BTCPP_format: "4"`, `main_tree_to_execute`, `root`,
      `blackboard` (where the waypoint arrays live, referenced from node ports as
      `"{{key}}"`), `xml` (the Groot2-loadable serialisation) and `metadata`. Return
      it verbatim — never rewrite ports or inline the blackboard by hand.
      `export_behaviour_tree_xml(tree)` re-emits the XML if you need it alone.
THE ARMS PLAN TOGETHER. `run_mrta_pipeline` composes the steps for ALL arms at
once (any number): follow its steps as given — the arms listed in a step work
in it, `waiting` arms stand still, and `park` arms get a `move` to their
`home_tcp` in that same step (see optimize-sequence). Every step is a
collaboration of all its arms: plan them in ONE `run_trajectory_planner` call, then
`validate_trajectory_set` with every arm of the step. The steps are PARALLEL.
When the arms of a step conflict, resolve it INSIDE the step. Never run the
program as one arm's welds then the other's. A SEQUENTIAL step is the LAST resort,
only when 1–3 below cannot clear it — the assembler lists it in `serial_steps`,
and you must explain each one. Reason about the collision and pick the fix that
keeps the arms working together at the lowest execution time:
  1. DELAY — `validate_trajectory_set` returns `delay_fix.start_delays` when a
     short stagger clears it. Use those as `start_delay_s`; the step stays PARALLEL.
  2. CHANGE THE WELD POINT — decide whether a different assignment avoids the
     collision: give the conflicting arm another pending weld (prefer the one
     CLOSEST to its own base, on its own side of the part, so the arms work far
     apart), or hand the weld to the other arm with `reallocate_operation`.
     Re-plan the step with all its arms in one call and validate.
  3. INTERLOCK POSITION — `find_interlock_set`: an intermediate pose where one arm
     waits clear of the other's path, released by a position gate. Among the
     options choose the one that MINIMISES execution time (shortest wait, least
     extra travel), plan it, and put it in the step.
Compare 2 and 3 by the time they add, and take the cheaper one. If
`run_trajectory_planner` reports `inter_arm_clearance`, plan that arm in its own
call for the SAME step and apply 1–3. The assembler reports `serial_steps` —
steps where an arm with welds left stood idle — and every one must be removed
with 2 or 3 before you finish; explain any that geometrically cannot be.
Unreachable welds only remove THOSE welds; they never justify idling an arm.

Never end with only an explanation, a question, or a partial result. If a target
is hard (collision with a box, unreachable, robot-vs-robot conflict), activate the
relevant skill (`approach-strategy`, `find-interlocks`), REDISTRIBUTE it to another
arm with `reallocate_operation` when its own arm cannot do it, and KEEP GOING until
the tree is assembled and valid. Always produce the tree.
"""

# The catalog SYSTEM_PROMPT was built with — no cell, frontmatter priorities.
_DEFAULT_SKILL_CATALOG = _skill_catalog()


#: Planning modes. "coordinated" (default): each arm follows its own planned path
#: and the arms are ordered in time by completion gates (coordination.py).
#: "stepwise": the earlier flow — composed parallel steps, each arm pair resolved
#: inside its step. Selected per task (constraints.planning_mode) or per stack
#: (PLANNING_MODE).
#: "classic" rebuilds the end-of-August 2026 planner (2f786cb, 29 Aug): its
#: prompt, its 19 tools and its skills (skills/trajectory-classic), on today's
#: infrastructure. The list and the default live in plan_selection, which owns
#: how a request's mode is chosen.
from agents.trajectory.plan_selection import (  # noqa: E402
    DEFAULT_PLANNING_MODE, PLANNING_MODES,
)

#: The tools the classic (29 Aug, 2f786cb) planner had — nothing added since.
CLASSIC_TOOL_NAMES = (
    "load_skill", "run_mrta_pipeline", "run_trajectory_planner", "run_path_shortcut",
    "reallocate_operation", "validate_approach", "validate_trajectory_pair",
    "validate_trajectory_set", "find_interlocks", "find_interlock_set",
    "assemble_behaviour_tree", "chain_trajectories", "build_behaviour_tree_node",
    "validate_behaviour_tree", "emit_parallel_node", "export_behaviour_tree_xml",
    "store_trajectory", "get_warm_start", "get_similar_trajectories",
)


def _skill_agent() -> str:
    """Which skill set the running plan reads: the classic mode has its own."""
    from agents.trajectory.tools.coordination import current_planning_mode
    return "trajectory-classic" if current_planning_mode() == "classic" else "trajectory"


def resolve_planning_mode(constraints: dict | None) -> str:
    mode = str((constraints or {}).get("planning_mode")
               or os.environ.get("PLANNING_MODE") or DEFAULT_PLANNING_MODE).strip().lower()
    return mode if mode in PLANNING_MODES else DEFAULT_PLANNING_MODE


COORDINATED_MODE_PROMPT = """
PLANNING MODE: COORDINATED (this OVERRIDES the step-by-step flow above — the
"arms plan together" rules, step composition, parks, SEQUENTIAL steps and
`assemble_behaviour_tree` do not apply in this mode).

The arms each follow their OWN path and are kept apart in TIME, by the system.
You own the GEOMETRY; `coordinate_program` owns the TIMING. First
`load_skill('coordinate-arms')` and follow it. In short:
1. `run_mrta_pipeline` — use it for the ALLOCATION and each arm's ORDER only
   (read each arm's operations in step order); ignore its step pairing.
2. For EACH ARM, plan its operations IN ORDER with `run_trajectory_planner`,
   ONE ARM PER CALL (never couple arms in a call — that is what pushes paths
   away from the part). For every weld YOU choose the strategy and waypoints
   that satisfy it: `approach_offset_m` / `approach_velocity_scale` for the
   approach, `retreat_offset_m` for leaving the PREVIOUS weld, `orientation` +
   `constrain_orientation`, and a via-point move when the direct approach is
   blocked (`approach-strategy`). Keep every standoff as short as the process
   allows — the arms should stay close to the part. Validate each weld with
   `validate_approach`.
3. `coordinate_program(cell_id, sequences)` — it schedules all arms in parallel,
   an arm waiting WHERE IT STANDS when another must pass first. If it reports a
   `deadlock`, the arm named in `needs_safe_spot` is in the way: choose a SAFE
   SPOT for it — close to the part, on its own side, clear of the other arm's
   path (e.g. a short lift above its last weld, toward its own base) — plan it
   as a `move` chained after that operation, insert it into the sequence, and
   coordinate again. An arm that has FINISHED its work is sent home by the
   coordinator itself (`early_home`) — never plan a safe spot for it.
4. THE MOMENT `coordinate_program` returns success, call
   `assemble_coordinated_tree` with the `sequences` it returned (they include
   any early-home legs) and return its tree. Do not keep iterating to shave
   seconds — a valid coordinated program is the deliverable. It refuses a
   deadlocked program.
Never serialise the program and never idle an arm that has reachable welds.
"""


#: Appended to every mode's prompt. Measured on bt_1790857060: 92 % of a run is
#: the model's turns, so the fewer and smaller the turns, the faster the plan.
EFFICIENT_TOOL_USE_PROMPT = """
EFFICIENT TOOL USE (every tool call costs a model turn — keep them few):
- Plan SEVERAL consecutive steps in ONE `run_trajectory_planner` call:
  `steps=[[step-1 requests], [step-2 requests], ...]` in execution order. When a
  step fails or its gate refuses it (`stopped_at`), only the later steps that
  move one of ITS arms are skipped (`skipped_steps`); steps moving other arms are
  still planned and kept (`kept_steps_after_stop`). Fix the failed step, then
  send exactly the failed + `skipped_steps` again — never the kept ones.
- Every planned step returns `step_gate`, the assembler's inter-arm check run on
  the spot (the step's arms against each other AND against the arms parked where
  they last stopped). PARALLEL: no separate validate call is needed for it.
  REFUSED: fix it before planning further — the assembler would refuse it too.
- Trajectories are cached and come back as summaries. Refer to one by its
  operation_id — `validate_trajectory_set(trajectories={robot_ns: operation_id})`,
  `validate_trajectory_pair(..., operation_id1=..., operation_id2=...)`. Never
  copy waypoint arrays into a call.
- Each arm chains automatically from where its last planned leg ENDED: omit
  `current_joints` unless you deliberately re-plan from another pose.
- Make independent checks (e.g. `validate_approach` for several welds) as
  parallel tool calls in one turn.
"""


def system_prompt_for(cell_id: str | None, mode: str = DEFAULT_PLANNING_MODE) -> str:
    """SYSTEM_PROMPT with the catalog re-ordered/filtered for ``cell_id``'s
    ``skills.yaml`` overrides, and the planning mode's instructions."""
    if mode == "classic":
        from agents.trajectory.classic_prompt import classic_prompt
        try:
            catalog = skill_registry.catalog_text("trajectory-classic", cell_id)
        except Exception:  # noqa: BLE001
            catalog = skill_registry.catalog_text("trajectory-classic", None)
        return classic_prompt(catalog) + "\n" + EFFICIENT_TOOL_USE_PROMPT
    prompt = SYSTEM_PROMPT
    if cell_id:
        prompt = prompt.replace(_DEFAULT_SKILL_CATALOG, _skill_catalog(cell_id), 1)
    if mode == "coordinated":
        prompt = prompt + "\n" + COORDINATED_MODE_PROMPT
    return prompt + "\n" + EFFICIENT_TOOL_USE_PROMPT

# The LLM's toolset is deliberately HIGH-LEVEL. It has NO raw IK / FK /
# per-config collision tools (run_ik, run_fk, check_self_collision,
# get_link_spheres) — those exist only INSIDE run_trajectory_planner. Exposing
# them made the model rabbit-hole on manual pose-probing and never plan. The
# only way to get motion here is run_trajectory_planner; verification is done on
# the PLANNED result (validate_approach on the trajectory, validate_trajectory_pair
# for a PARALLEL pair); recovery uses find_interlocks + re-planning.
_TOOLS = [
    load_skill,
    run_mrta_pipeline, run_trajectory_planner, run_path_shortcut,
    reallocate_operation,
    validate_approach, validate_trajectory_pair, validate_trajectory_set,
    find_interlocks, find_interlock_set,
    assemble_behaviour_tree, chain_trajectories,
    build_behaviour_tree_node, validate_behaviour_tree, emit_parallel_node,
    export_behaviour_tree_xml,
    store_trajectory, get_warm_start, get_similar_trajectories,
    coordinate_program, assemble_coordinated_tree,
    evaluate_program, find_clear_pose,
]


def _resolve_model() -> str:
    """The effective ``"provider:model_id"`` for this run.

    Thin delegate: agents/llm_model.py owns the resolution order (per-request
    binding → argument → ``LLM_MODEL`` → legacy provider var → Opus 5) so the
    planner, the orchestrator and the cell's recovery layer cannot end up on
    three different models.
    """
    return resolve_model()


def _build_llm(model_str: str | None = None):
    """Build the LLM for this run. See :mod:`agents.llm_model`."""
    return build_llm(model_str)


def _is_waypoint_array(value) -> bool:
    return (isinstance(value, list) and len(value) > 1 and isinstance(value[0], dict)
            and "positions" in value[0])


def _round_joints(q) -> list:
    return [round(float(v), 4) for v in (q or [])]


def agent_view(value):
    """A tool result as the MODEL sees it: every waypoint array summarised.

    A planned leg is 70–160 timed samples; returned raw, one two-arm step is ~7k
    tokens, and the ReAct loop re-sends every one of them on every later turn —
    on bt_1790857060 the conversation carried ~190k tokens of waypoints by the
    end. The model never needs them: the planner caches each leg by operation_id
    and every consumer (validators, chain check, assembler) reads the cache. The
    joints an arm starts and ends at are kept, because those are what a re-plan
    from an explicit pose needs.
    """
    if _is_waypoint_array(value):
        last = value[-1]
        return {"waypoints": len(value),
                "duration_s": round(float(last.get("time_s") or 0.0), 3),
                "start_joints": _round_joints(value[0].get("positions")),
                "end_joints": _round_joints(last.get("positions"))}
    if isinstance(value, dict):
        return {k: agent_view(v) for k, v in value.items()}
    if isinstance(value, list):
        return [agent_view(v) for v in value]
    return value


def _agent_facing(t):
    """``t`` with its result passed through :func:`agent_view` — same name,
    description and argument schema, so the model sees the same tool."""
    from langchain_core.tools import StructuredTool
    inner = getattr(t, "func", None)
    if inner is None or not getattr(t, "args_schema", None):
        return t

    def run(**kwargs):
        raw = inner(**kwargs)
        if t.name in _ASSEMBLERS:
            _remember_full_tree(raw)
        return agent_view(raw)

    return StructuredTool(name=t.name, description=t.description,
                          args_schema=t.args_schema, func=run)


# The assembler's tree as it really is — every leg's waypoints — keyed by the
# tree's node ids. The model only ever sees the agent_view of it, and that view is
# what lands in the ToolMessage _extract_behaviour_tree reads, so without this the
# persisted plan carried leg SUMMARIES and replayed nothing (bt_1790951179: 19
# legs, 0 waypoints, success: true). Node ids are minted per assembly, so the key
# names exactly one assembler call and survives the view unchanged.
_FULL_TREES: dict[tuple, dict] = {}
_FULL_TREES_KEPT = 16


def _node_ids(tree: dict) -> tuple:
    ids: list = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("id") is not None:
                ids.append(str(node["id"]))
            for child in node.get("children") or []:
                walk(child)
            if isinstance(node.get("child"), dict):
                walk(node["child"])

    walk((tree or {}).get("root"))
    return (str((tree or {}).get("tree_id")), *ids)


def _remember_full_tree(result) -> None:
    tree = result.get("behaviour_tree") if isinstance(result, dict) else None
    if not isinstance(tree, dict) or "root" not in tree:
        return
    key = _node_ids(tree)
    if len(key) < 2:     # no node ids: nothing would identify it later
        return
    _FULL_TREES[key] = tree
    while len(_FULL_TREES) > _FULL_TREES_KEPT:
        _FULL_TREES.pop(next(iter(_FULL_TREES)))


def _has_summarised_legs(tree: dict) -> bool:
    """True when a blackboard trajectory is an agent_view summary, not waypoints."""
    bb = (tree or {}).get("blackboard") or {}
    return any(isinstance(v, dict) and isinstance(v.get("waypoints"), int)
               for v in bb.values())


def _full_tree(tree: dict) -> dict | None:
    """The assembler's real tree for one the model saw, or None if unknown.

    None for a summarised tree with no remembered original is deliberate: a tree
    whose legs are summaries cannot move an arm, and storing it as a successful
    plan is the failure this exists to prevent.
    """
    if not _has_summarised_legs(tree):
        return tree
    full = _FULL_TREES.get(_node_ids(tree))
    if full is None:
        print("[plan] !! the accepted tree carries trajectory summaries and its "
              "full version is not known (restarted mid-run?) — refusing to "
              "store it as a plan", flush=True)
    return full


def create_trajectory_agent(model: str | None = None, cell_id: str | None = None,
                            mode: str = DEFAULT_PLANNING_MODE):
    """Create the trajectory ReAct agent with all planning tools. ``cell_id``
    applies that cell's skill prioritisation to the prompt's catalog; ``mode`` is
    the planning mode (see PLANNING_MODES)."""
    llm = _build_llm(model)
    tools = ([t for t in _TOOLS if getattr(t, "name", "") in CLASSIC_TOOL_NAMES]
             if mode == "classic" else _TOOLS)
    tools = [_agent_facing(t) for t in tools]
    # Tools bound HERE, with a rolling prompt-cache point on Bedrock, so each turn
    # reuses the previous turn's prefix; create_react_agent keeps this binding.
    llm = bind_tools_for_agent(llm, tools)
    # The model sees old turns as digests (context_compaction): the input used to
    # grow ~5k tokens per turn to 400k+, and every turn re-read all of it. The
    # agent's state keeps the full history (tree extraction, trace, live feed).
    # Done in the PROMPT callable, not a pre_model_hook: a hook is a graph node of
    # its own, so every turn cost 3 steps of the recursion budget instead of 2 —
    # welding_cell_2_small (15 ops, budget 160) stopped at turn 53.
    from agents.trajectory.context_compaction import compacting_prompt
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=compacting_prompt(system_prompt_for(cell_id, mode)),
    )


def _classify_error(exc: Exception) -> tuple[str, str]:
    """Classify a planning-run exception so an INFRASTRUCTURE failure is never
    mistaken for a robot-reachability/collision result.

    Returns (kind, message) where kind is:
      "credentials" — expired/invalid AWS token or access denied (refresh + recreate),
      "network"     — cannot reach the model endpoint (DNS/TCP/timeout),
      "internal"    — anything else (a genuine code error, not the robot's fault).
    None of these mean a weld is unreachable — that is reported per-operation in
    the planner's ``failures`` (goal_in_collision / target_diagnosis / no_path).
    """
    s = str(exc)
    low = s.lower()
    code = ""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = ((resp.get("Error") or {}).get("Code") or "")
    cred_codes = {"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException",
                  "InvalidSignatureException", "AccessDeniedException", "InvalidClientTokenId"}
    cred_markers = ("expiredtoken", "security token included in the request is expired",
                    "unrecognizedclient", "invalidsignature", "accessdenied",
                    "invalidclienttokenid")
    net_markers = ("could not connect", "endpointconnectionerror", "name or service not known",
                   "failed to resolve", "temporary failure in name resolution",
                   "connection reset", "connection refused", "timed out", "read timeout")
    if code in cred_codes or any(m in low for m in cred_markers):
        return "credentials", f"AWS credentials/token error (refresh token + recreate the container): {s[:300]}"
    if any(m in low for m in net_markers):
        return "network", f"Network error reaching the model endpoint: {s[:300]}"
    return "internal", s[:400]


def plan(cell_id: str, operations: list[dict], constraints: dict | None = None,
         tree_id: str | None = None, model: str | None = None,
         resume: bool = False) -> dict:
    """
    Run the trajectory agent for a planning request.

    Args:
        cell_id:     Cell identifier (e.g. "welding_cell_2_small")
        operations:  List of operation point dicts
        constraints: Optional planning constraints
        tree_id:     Optional pre-assigned BehaviourTree id. When given, the
                     assembled tree is stamped with (and saved under) this id, so
                     an async caller that returned this id up front can fetch the
                     result with get_calculated_plan(cell_id, tree_id).
        model:       LLM for THIS plan only — an alias ("opus-5"), a full model
                     id, or "provider:model". Omit for the planning default
                     (PLANNING_LLM_MODEL, else Sonnet).
        resume:      Continue the run ``tree_id`` left off at, from its saved
                     checkpoint, instead of planning from the first weld. The
                     conversation and every trajectory already solved are restored
                     (see :mod:`agents.trajectory.plan_checkpoint`); at most the
                     one step that was in flight is repeated.

    Returns:
        dict with keys: success, behaviour_tree, planning_time_s

    The model is BOUND for the run rather than passed down, so every nested LLM
    call the plan makes — notably the HOME staging proposer in
    ``tools/home_strategy.py``, reached from deep inside OMPL — runs on the model
    the caller asked for instead of falling back to the container default.

    The run is also REGISTERED as cancellable for its duration (same per-run
    binding pattern as the model and the backend). That is what lets the
    operator's Stop reach a plan that is minutes into an unkillable worker
    thread: see :mod:`agents.trajectory.tools.plan_cancellation`.
    """
    # A run with no pre-assigned id still registers, under a synthetic one, so
    # "cancel everything planning" has no blind spot.
    cancel_id = tree_id or f"plan_{cell_id}_{int(_time.time())}"
    # One run per cell: every planning cache is keyed by cell, and a second run
    # would wipe this one's mid-flight (agents/trajectory/cell_lock.py).
    from agents.trajectory import cell_lock
    if not cell_lock.acquire(cell_id, cancel_id):
        busy = cell_lock.holder_of(cell_id) or {}
        return {"success": False, "error_kind": "cell_busy",
                "error": (f"cell '{cell_id}' is already being planned by "
                          f"{busy.get('holder')}; wait for it (or cancel it) and submit "
                          "again"),
                "planning_time_s": 0.0}
    plan_cancellation.begin(cancel_id)
    try:
        from agents.trajectory.tools.coordination import bind_planning_mode
        # Planning runs on the request's model, else the PLANNING default (Sonnet —
        # see agents/llm_model.planning_model), bound so every nested LLM call in
        # the run (the HOME staging proposer included) uses the same one.
        from agents.trajectory import operator_directives, retry_budget
        from agents.trajectory.tools import plan_memo
        with bind_model(planning_model(model)), skill_registry.bind_cell(cell_id), \
                bind_planning_mode(None), skill_registry.bind_run_overrides(None), \
                skill_registry.bind_run_skills(None), operator_directives.scope(), \
                plan_memo.run_scope(), retry_budget.scope():
            return _plan(cell_id, operations, constraints, tree_id, resume=resume)
    finally:
        plan_cancellation.end(cancel_id)
        cell_lock.release(cell_id, cancel_id)


def _plan(cell_id: str, operations: list[dict], constraints: dict | None = None,
          tree_id: str | None = None, resume: bool = False) -> dict:
    """The planning run itself. See :func:`plan`, which binds the model first."""
    import json, time
    from agents.trajectory.tools.planner_backend import (
        resolve_planning_backend, set_planning_backend, reset_planning_backend,
        reset_backend_options, set_backend_options, validate_backend_options,
    )

    # Fresh planning session: drop stale cached trajectories and per-robot
    # chaining configs for this cell so the first weld of each robot starts from
    # HOME and subsequent welds chain from the newly-planned goals.
    from agents.trajectory.tools.trajectory_cache import (
        clear_trajectory_cache, reset_robot_configs,
    )
    clear_trajectory_cache(cell_id)
    reset_robot_configs(cell_id)

    # A resume restores the previous attempt's conversation AND its solved
    # trajectories, so the run continues where it stopped instead of re-solving
    # welds it already had. The reset above still runs first: restoring onto a
    # cache holding another cell's leftovers is how a stale trajectory would get
    # chained into a new plan.
    from agents.trajectory import plan_checkpoint, plan_events
    resumed = plan_checkpoint.restore(cell_id, tree_id) if (resume and tree_id) else None
    if resume and not resumed:
        return {
            "success": False,
            "error_kind": "not_resumable",
            "error": (f"no resumable checkpoint for '{tree_id}' — nothing was saved, "
                      "or what was saved held no completed step"),
            "note": ("Plan from the start instead. A checkpoint is written when a "
                     "run is paused and when it aborts on a token/network failure."),
            "planning_time_s": 0.0,
        }
    if resumed:
        operations = resumed.get("operations") or operations
        constraints = resumed.get("constraints") if constraints is None else constraints
        print(f"[plan] RESUMING {tree_id} from step {resumed['step']} — "
              f"{len(resumed['messages'])} messages, "
              f"{resumed['cache_restored'].get('trajectories', 0)} trajectories restored, "
              f"{resumed['discarded']} in-flight message(s) dropped", flush=True)

    # Bind the planner backend from the submitted task BEFORE the agent runs —
    # and only now, after a resume has restored the run's own constraints, so a
    # resumed run keeps the backend it started on. run_trajectory_planner reads
    # this binding, so the planner swap is guaranteed by the task and cannot be
    # overridden by the LLM. No fallback: if the selected backend can't plan,
    # the result reports that failure.
    requested_backend = (constraints or {}).get("trajectory_backend")
    try:
        effective_backend = resolve_planning_backend(requested_backend)
        # The request's backend_options, checked again against the backend this
        # run really uses (a resumed run carries them in its own constraints).
        backend_options = validate_backend_options(
            effective_backend, (constraints or {}).get("backend_options"))
    except ValueError as exc:
        return {"success": False, "error": str(exc), "planning_time_s": 0.0}

    planning_mode = resolve_planning_mode(constraints)
    # Scoped by plan()'s bind_planning_mode(None): set here, where the mode is
    # known (a resumed run takes it from its checkpoint's constraints).
    from agents.trajectory.tools import coordination as _coord
    _coord._MODE.set(planning_mode)

    # The operator's directives (plan_trajectories assignments / skill_priorities /
    # instructions), validated at submission and carried in the constraints so a
    # resumed run keeps them. Applied BEFORE the agent is built: the catalog in its
    # prompt is ordered by this run's skill priorities.
    from agents.trajectory import operator_directives
    directives = (constraints or {}).get(operator_directives.OPERATOR_KEY) or {}
    operations = operator_directives.merge_assignments(operations, directives)
    locks = operator_directives.all_locks(operations)
    operator_directives.set_locks(locks)
    if directives.get("run_skills"):
        skill_registry.set_run_skills({_skill_agent(): directives["run_skills"]})
    if directives.get("skill_priorities"):
        skill_registry.set_run_overrides({_skill_agent(): directives["skill_priorities"]})
    agent = create_trajectory_agent(cell_id=cell_id, mode=planning_mode)
    t0    = time.monotonic()

    task = (
        operator_directives.prompt_section(directives, locks)
        + f"Plan trajectories for cell '{cell_id}'.\n\n"
        f"Operations ({len(operations)} points):\n"
        f"{json.dumps(operations, indent=2)}\n"
    )
    shown = {k: v for k, v in (constraints or {}).items()
             if k != operator_directives.OPERATOR_KEY}   # already stated above
    if shown:
        task += f"\nConstraints: {json.dumps(shown)}"

    op_types: dict[str, int] = {}
    for op in operations:
        t = op.get("type", "?")
        op_types[t] = op_types.get(t, 0) + 1
    print(
        f"[plan] ===== START cell='{cell_id}' "
        f"ops={len(operations)} ({_brief(op_types, 120)}) "
        f"backend={effective_backend} mode={planning_mode} "
        f"model={effective_model_id()} =====",
        flush=True,
    )

    # Scale the step budget with the work. A fixed 100 was a silent cliff: each
    # weld costs ~3 steps (plan, validate the approach, pair-validate) plus a
    # per-run overhead of skill loads, chaining and assembly, so a 15-weld cell
    # exhausted the budget mid-run and returned NO tree after ~13 minutes of real
    # planning. Budgeting per operation keeps the ceiling a runaway-loop backstop
    # rather than a limit real plans hit. Override with TRAJ_RECURSION_LIMIT.
    _STEPS_PER_OPERATION = 8
    _RUN_OVERHEAD_STEPS = 40
    recursion_limit = int(os.environ.get(
        "TRAJ_RECURSION_LIMIT",
        max(100, _RUN_OVERHEAD_STEPS + _STEPS_PER_OPERATION * len(operations)),
    ))

    config = {
        "configurable": {"thread_id": f"plan_{cell_id}_{int(time.time())}"},
        "recursion_limit": recursion_limit,
    }
    print(f"[plan] step budget: {recursion_limit} "
          f"({len(operations)} ops x {_STEPS_PER_OPERATION} + {_RUN_OVERHEAD_STEPS})",
          flush=True)

    # Stream the ReAct loop so reasoning + tool calls are logged live.
    # The backend binding is active only for the duration of this run, then
    # reset so it never leaks into a subsequent plan on the same thread.
    all_messages: list = resumed["messages"] if resumed else []
    logged = len(all_messages)
    counters: dict = {"step": resumed["step"] if resumed else 0}
    started_at = _datetime.now()
    paused_reason: str | None = None

    plan_events.run_started(
        cell_id, tree_id, operations=len(operations), backend=effective_backend,
        model=effective_model_id(), step_budget=recursion_limit,
        resumed_from_step=(resumed["step"] if resumed else None),
    )
    if resumed:
        plan_events.resumed(cell_id, tree_id, step=resumed["step"],
                            discarded=resumed["discarded"])

    def _checkpoint(reason: str, resumable: bool = True) -> tuple[str | None, bool]:
        """Write the run's place, and say whether continuing it would recover work.

        Both halves, because they differ: a run that died before completing a step
        still gets a file (it records what happened), but resuming it would replay
        the opening prompt and nothing else. Offering that as a resume is offering a
        restart under another name, so the caller is told which it has.
        """
        saved = plan_checkpoint.save(
            cell_id, tree_id, messages=all_messages, operations=operations,
            constraints=constraints, step=counters.get("step", 0),
            backend=effective_backend, model=effective_model_id(),
            step_budget=recursion_limit, reason=reason, resumable=resumable,
        )
        if saved:
            print(f"[plan] checkpoint saved ({reason}): {saved}", flush=True)
            plan_events.checkpointed(cell_id, tree_id,
                                     step=counters.get("step", 0), path=saved)
        stored = plan_checkpoint.describe(cell_id, tree_id) if tree_id else None
        return saved, bool(stored and stored.get("resumable"))

    def _save_failed_plan(result: dict) -> None:
        """Keep an unsuccessful run's computed work, marked failed.

        A failed run is usually a nearly-finished one. Discarding it — which is
        what "the tree is only persisted on success" meant in practice — is what
        made every retry re-solve welds that were already solved. So the partial
        work is stored beside the real plans with ``status: failed``, listed and
        readable, and NOT executable: plan_store.get_plan refuses to hand it to
        anything that would drive a robot.
        """
        from agents.trajectory.tools import plan_store, trajectory_cache
        snapshot = trajectory_cache.snapshot(cell_id)
        failures: list = []
        for record in (result.get("failures") or []):
            if isinstance(record, dict):
                failures.append(record)
        # Whether the failure is resumable is not a guess: it is whether a
        # checkpoint with a completed step is actually on disk right now.
        checkpoint = plan_checkpoint.describe(cell_id, tree_id) if tree_id else None
        plan_store.save_failed_plan(
            cell_id, tree_id,
            error=result.get("error"), error_kind=result.get("error_kind"),
            note=result.get("note"), failures=failures, operations=operations,
            trajectories=snapshot.get("trajectories") or [],
            behaviour_tree=result.get("behaviour_tree"),
            resumable=bool(checkpoint and checkpoint.get("resumable")),
        )

    def _finish(result: dict) -> dict:
        """Persist the run's decision trace, then return the result unchanged.

        Every exit path goes through here — the abort, the refusals, the success —
        because the runs worth reading are precisely the ones that failed, and a
        trace written only on success would miss all of them. The stdout log is not
        a substitute: it is unstructured, and it dies with the container.
        """
        from agents.trajectory import plan_trace
        saved = plan_trace.capture(
            cell_id, all_messages, operations, result,
            trace_id=tree_id, tree_id=tree_id, backend=effective_backend,
            model=effective_model_id(), started_at=started_at,
            step_budget=recursion_limit,
        )
        if saved:
            print(f"[plan] decision trace saved: {saved}", flush=True)
        if not result.get("success") and result.get("error_kind") != "paused":
            _save_failed_plan(result)
        plan_events.finished(
            cell_id, tree_id, success=bool(result.get("success")),
            tool_calls=counters.get("step", 0),
            elapsed_s=result.get("planning_time_s"), error=result.get("error"),
            tree_id=tree_id,
        )
        plan_events.forget(tree_id or "")
        return result

    _backend_token = set_planning_backend(effective_backend)
    _options_token = set_backend_options(backend_options)
    try:
        for state in agent.stream(
            {"messages": all_messages or [HumanMessage(content=task)]},
            config=config,
            stream_mode="values",
        ):
            all_messages = state["messages"]
            for msg in all_messages[logged:]:
                _log_agent_message(msg, counters)
                _publish_events(cell_id, tree_id, msg)
            logged = len(all_messages)

            # A state boundary is the only place a conversation is valid to
            # checkpoint from, so it is the only place pause can bite. Breaking
            # here costs at most the step that just finished — never the run.
            paused_reason = plan_checkpoint.pause_requested(tree_id)
            if paused_reason:
                plan_checkpoint.take_pause(tree_id)
                break
    except Exception as exc:
        # An LLM/transport error (expired token, no connectivity, ...) is an
        # INFRASTRUCTURE failure — report it as such so it is never confused with
        # a robot being unable to reach a weld. Per-weld reachability/collision
        # failures live in the planner result's `failures`, not here.
        kind, msg = _classify_error(exc)
        note = (
            "Infrastructure/credentials failure — NOT a robot reachability or "
            "collision result. No weld was found unreachable; the plan could not "
            "run. Refresh the AWS token, recreate the agent container, and retry."
            if kind in ("credentials", "network")
            else "Internal planner error — not a robot reachability result."
        )
        print(f"[plan] ABORTED ({kind}): {msg}", flush=True)
        # THE case this checkpoint exists for. An expired token or a dropped
        # socket says nothing about the welds; the run simply lost its place. So
        # the place is written down before the failure is reported, and the reply
        # says so — the recovery is `resume`, not fifteen minutes of re-solving.
        checkpoint, resumable = _checkpoint(f"aborted: {kind}")
        if resumable:
            note += (f" This run was CHECKPOINTED at step {counters.get('step', 0)}"
                     f"{f' with {len(all_messages)} messages' if all_messages else ''} — "
                     "resume it instead of replanning from the first weld.")
        plan_events.aborted(cell_id, tree_id, kind=kind, error=msg,
                            resumable=resumable)
        return _finish({
            "success":         False,
            "error_kind":      kind,
            "error":           msg,
            "note":            note,
            "resumable":       resumable,
            "checkpoint":      checkpoint,
            "resume_with":     (f"resume_planning('{tree_id}')" if resumable else None),
            "tool_calls":      counters.get("step", 0),
            "planning_time_s": round(time.monotonic() - t0, 3),
        })
    finally:
        reset_backend_options(_options_token)
        reset_planning_backend(_backend_token)

    # An operator's Pause. A distinct outcome, not a failure and not a
    # cancellation: nothing is wrong, everything is kept, and the run is waiting
    # to be continued. It deliberately does NOT store a failed plan — the run has
    # not failed, and marking it so would put a wreck in the plan list for what is
    # really a coffee break.
    if paused_reason:
        checkpoint, resumable = _checkpoint(f"paused: {paused_reason}")
        plan_events.paused(cell_id, tree_id, reason=paused_reason,
                           step=counters.get("step", 0), checkpoint=checkpoint)
        print(f"[plan] PAUSED after step {counters.get('step', 0)}: {paused_reason}",
              flush=True)
        return _finish({
            "success":         False,
            "error_kind":      "paused",
            "error":           f"planning paused after {counters.get('step', 0)} "
                               f"tool calls ({paused_reason})",
            "note":            ("Paused, not failed — no weld was found unplannable "
                                "and nothing was discarded. Resume to continue from "
                                "this point."),
            "paused":          True,
            "resumable":       resumable,
            "checkpoint":      checkpoint,
            "resume_with":     (f"resume_planning('{tree_id}')" if resumable else None),
            "tool_calls":      counters.get("step", 0),
            "planning_time_s": round(time.monotonic() - t0, 3),
        })

    # Extract the behaviour tree from the WHOLE history (not just the last
    # message) so we still capture it if the agent ended with a text summary.
    bt = _extract_behaviour_tree(all_messages)
    # The fallback paths read trees out of message text too, and all message text
    # is agent_view'd: a tree whose legs are summaries is not a plan.
    if bt is not None and _has_summarised_legs(bt):
        print("[plan] !! extracted tree has summarised legs — not a plan", flush=True)
        bt = None

    # No post-assembly finalizer. The per-target motion planner
    # (run_trajectory_planner) is the SOLE trajectory calculator: it chains each
    # operation from the arm's previous goal (trajectory_cache), so consecutive
    # operations are continuous BY CONSTRUCTION when planned in execution order.
    # The LLM verifies that continuity with the `chain_trajectories` tool and
    # re-plans any broken seam, and `assemble_behaviour_tree` re-checks it and
    # refuses a tree with a gap. The planner's cached trajectories go into the
    # tree VERBATIM — nothing here rewrites the motion.
    if bt is not None:
        # Stamp the tree with the caller's pre-assigned id (async submit path) so
        # it is saved under, and fetchable by, the id already returned to them.
        if tree_id:
            bt["tree_id"] = tree_id
        # Persist the finished tree (with precomputed trajectories) so it can be
        # retrieved/replayed later via the trajectory MCP get_plan/list_plans
        # tools and fed to the cell's execute_behaviour_tree — no re-planning.
        try:
            from agents.trajectory.tools.plan_store import save_plan
            saved = save_plan(cell_id, bt)
            if saved:
                print(f"[plan] saved plan '{saved}' for later replay", flush=True)
        except Exception as exc:  # persistence must never break planning
            print(f"[plan] plan-store save skipped ({exc})", flush=True)

    elapsed = round(time.monotonic() - t0, 3)

    print(
        f"[plan] ===== DONE cell='{cell_id}' "
        f"tool_calls={counters['step']} elapsed={elapsed}s "
        f"behaviour_tree={'yes' if bt else 'MISSING'} =====",
        flush=True,
    )

    if bt is not None:
        # The run arrived. Its checkpoint is now a stale conversation that would
        # resume into work already done, so it goes — a resumable checkpoint must
        # only ever exist for a run that has somewhere left to go.
        if tree_id:
            plan_checkpoint.discard(cell_id, tree_id)
        return _finish({
            "success":        True,
            "behaviour_tree": bt,
            "planning_time_s": elapsed,
            # Reported, not inferred: a plan is only reproducible if the model
            # that produced it is on the record next to it.
            "model":          effective_model_id(),
            # Likewise the operator's directives it was planned under.
            **({"operator_directives": directives} if directives else {}),
        })

    last = all_messages[-1] if all_messages else None
    content = _extract_text(getattr(last, "content", "")) if last else ""

    # If the assembler REFUSED a tree (broken chain / duplicate operation_id /
    # invalid structure), surface its reason: that is far more actionable than
    # "no tree was produced", and it distinguishes "the agent gave up" from
    # "the agent produced a tree that failed the consistency gate".
    rejection: dict | None = None
    for msg in reversed(all_messages or []):
        if getattr(msg, "name", None) not in _ASSEMBLERS:
            continue
        try:
            obj = json.loads(_extract_text(getattr(msg, "content", "")).strip())
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(obj, dict) and obj.get("success") is False:
            rejection = {
                k: obj[k] for k in
                ("error", "breaks", "duplicates", "missing", "validation")
                if k in obj
            }
            break

    if rejection:
        return _finish({
            "success":    False,
            "error_kind": "planning",
            "error": (
                "The assembled BehaviourTree was REFUSED by the consistency gate, so "
                "no plan is returned: " + str(rejection.get("error", "validation failed"))
            ),
            "rejection": rejection,
            "note": (
                "This is a fail-closed refusal, not an infrastructure failure. A tree "
                "was built but did not pass validation (trajectory continuity / unique "
                "operation ids / structure), and an unvalidated tree is never returned."
            ),
            "planning_time_s": elapsed,
        })

    # Distinguish a TRUNCATED response from "gave up". If the provider cut the
    # reply at its output-token ceiling in the middle of a tool_use block, the
    # partial block is unparseable and langchain drops it, so the AIMessage
    # arrives with no text, no tool_calls and no invalid_tool_calls. The ReAct
    # loop has nothing to execute and ends — looking exactly like a planner that
    # stopped on its own. It is not: no decision was made, the sentence was cut
    # off mid-word. Reporting it as a planning outcome would let a transport
    # failure masquerade as "these welds could not be planned", so it gets its own
    # kind. `agents.llm_model.max_output_tokens` is what keeps the ceiling clear
    # of a real step; this is the backstop that stays honest if it is ever hit.
    stop_reason = ""
    if last is not None:
        meta = getattr(last, "response_metadata", None) or {}
        stop_reason = str(
            meta.get("stopReason") or meta.get("stop_reason")
            or meta.get("finish_reason") or ""
        )
    if stop_reason == "max_tokens":
        from agents.llm_model import max_output_tokens
        # Resumable, and worth resuming: the conversation up to the cut is intact
        # and every weld solved so far is in the cache. The fix is a bigger ceiling,
        # not a replan, so leave a checkpoint for the retry to land on.
        checkpoint, resumable = _checkpoint("aborted: truncated")
        return _finish({
            "success":    False,
            "error_kind": "truncated",
            "error": (
                "The model's reply was CUT OFF at its output-token limit after "
                f"{counters['step']} tool calls, so the next tool call never arrived "
                "and no BehaviourTree was assembled (stopReason=max_tokens)."
            ),
            "note": (
                "NOT a reachability or collision result and not a planner give-up — "
                "no weld was found unplannable. The response ceiling is too low for "
                "one planning step: raise LLM_MAX_TOKENS (currently "
                f"{max_output_tokens()}) and retry."
                + (f" Resume with resume_planning('{tree_id}') to keep the "
                   f"{counters['step']} steps already done."
                   if resumable else "")
            ),
            "max_output_tokens": max_output_tokens(),
            "tool_calls":        counters["step"],
            "resumable":         resumable,
            "checkpoint":        checkpoint,
            "resume_with":       tree_id if resumable else None,
            "planning_time_s":   elapsed,
        })

    # Distinguish "ran out of steps" from "gave up". LangGraph reports budget
    # exhaustion as an ordinary final message, so without this the two are
    # indistinguishable and the actionable fix (raise the budget) is invisible.
    exhausted = "need more steps" in content.lower()
    if exhausted:
        # Same reasoning as truncated: the planner was mid-work at a clean state
        # boundary, so a resume with a bigger budget continues instead of restarting.
        checkpoint, resumable = _checkpoint("aborted: step_budget")
        return _finish({
            "success":    False,
            "error_kind": "step_budget",
            "error": (
                f"Planning exhausted its step budget of {recursion_limit} after "
                f"{counters['step']} tool calls, so no BehaviourTree was assembled. "
                f"Final message: {content[:200]}"
            ),
            "note": (
                "NOT an infrastructure or reachability failure — the planner was still "
                "working when it hit the ceiling. Raise it with TRAJ_RECURSION_LIMIT, "
                "or plan fewer operations per call."
                + (f" Resume with resume_planning('{tree_id}') to keep the "
                   f"{counters['step']} steps already done."
                   if resumable else "")
            ),
            "step_budget":     recursion_limit,
            "tool_calls":      counters["step"],
            "resumable":       resumable,
            "checkpoint":      checkpoint,
            "resume_with":     tree_id if resumable else None,
            "planning_time_s": elapsed,
        })

    return _finish({
        "success":        False,
        "error_kind":     "planning",
        "error": (
            "Planning did not produce a BehaviourTree. The agent stopped after "
            f"{counters['step']} tool calls without assembling/validating a tree. "
            f"Final message: {content[:400]}"
        ),
        "note": (
            "The planner ran (credentials/network were fine). This is a planning "
            "outcome, not an infrastructure failure. Any per-weld reachability or "
            "collision issues are listed in the run's tool failures with a "
            "target_diagnosis/suggestion."
        ),
        "planning_time_s": elapsed,
    })
