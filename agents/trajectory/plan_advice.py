"""What is wrong with a calculated plan, and what to try next.

``trajectory_quality`` measures a plan and ranks its findings; ``plan_trace``
records what the planner did to get there. Neither says what to *change*. This
module reads both (plus the task program's cycle-time target, when named) and
turns them into one list of problems, each with the evidence and a concrete next
step — the answer to "how do I make this plan better?" rather than "how good is
it?".

Advice, never action: nothing here re-plans, edits a program or moves an arm. A
suggestion is something an operator or the planning agent may choose to try.

Pure over its inputs (:func:`advise`), so it is testable without the LLM stack;
:func:`advice_for` is the loader the MCP tool calls.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

# What to try for each quality finding kind. Keyed by trajectory_quality's
# finding ``kind``; a kind not listed here is still reported, with no suggestion.
_FINDING_ADVICE = {
    "no_measurable_motion": (
        "The stored tree has no executable trajectory at all. Re-plan the program; "
        "do not replay this tree."),
    "unusable_leg": (
        "This leg carries no usable trajectory. Re-plan the operation — a leg "
        "with no waypoints cannot be executed."),
    "path_detour": (
        "Re-plan this leg alone (more planning time lets shortcutting converge), "
        "or add a via point at working height so the planner does not wander; if "
        "it keeps detouring, an obstacle sits on the direct path — check the "
        "approach side."),
    "interior_dwell": (
        "The arm stops mid-leg. Re-plan or re-time the leg; stalls usually come "
        "from duplicated waypoints left after chaining."),
    "slow_plan": (
        "Raise velocity_scale for transit legs; keep the working stroke slow "
        "through approach_velocity_fraction instead of slowing the whole plan."),
    "slow_leg": (
        "Raise velocity_scale on this leg — it is slow, not long."),
    "smoothness_outlier": (
        "Re-plan this leg; it is far jerkier than its peers, which usually means "
        "an unshortcut zig-zag."),
    "low_parallelism": (
        "Rebalance the weld allocation so both arms work at once: move operations "
        "off the busier arm, and give each arm its own half of the part so their "
        "swept volumes do not overlap (find_interlock_set checks a pairing)."),
    "unpaired_operations": (
        "These legs run with the other arm idle. Pair each with an operation on "
        "the other arm in a PARALLEL step, or move one to the idle arm."),
    "seam_discontinuity": (
        "An arm starts a leg away from where the previous one ended. Run "
        "chain_trajectories and re-plan the broken seam before replaying."),
}

# Planner failure messages, by what they mean. First match wins; order matters.
_FAILURE_PATTERNS = (
    ("tool_lift", re.compile(r"lifts the tool", re.I),
     "The route arcs high above both endpoints. Lower approach_offset_m, or add a "
     "via point at working height between the welds."),
    ("batch_stopped", re.compile(r"batch stopped at step", re.I),
     "A batch stopped after an earlier operation in it failed; the rest were "
     "never tried. Plan the failing operation alone first, then batch only what "
     "is known to be reachable."),
    ("arms_conflict", re.compile(r"clear of each other|clears the other arms|"
                                 r"swept volume|will REFUSE", re.I),
     "The two arms' paths overlap in this step. Stagger them (delay_fix), pair a "
     "different pending weld, or use find_interlock_set; a SEQUENTIAL step is the "
     "last resort."),
    ("unreachable", re.compile(r"unreachable|no IK|ik fail|out of reach|"
                               r"joint limit", re.I),
     "The target is outside what this arm can reach with that tool orientation "
     "(the planner already searched the tool's free spin and tilt). Give it to "
     "the other arm."),
    ("collision", re.compile(r"collision|collides|clearance", re.I),
     "The target or path collides with the cell. Check the obstacle model near "
     "it and the approach direction."),
)

#: An operation that took this many attempts is a hot spot worth naming.
HOT_SPOT_ATTEMPTS = 4
#: Share of the step budget past which a run is close to giving up.
STEP_BUDGET_WARN = 0.7

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _failure_class(error: str, reason_code: Optional[str] = None
                   ) -> tuple[str, Optional[str]]:
    """The failure's kind and advice: by its ``reason_code`` (the recovery table,
    reason_codes.py) when it has one, else by its text (traces from older runs)."""
    if reason_code:
        from agents.trajectory.reason_codes import recovery
        rec = recovery(reason_code)
        if rec:
            levers = "; ".join(rec["levers"]) or "none"
            return reason_code, (f"{levers}. When those are spent: {rec['escalate']}.")
    for kind, pattern, advice in _FAILURE_PATTERNS:
        if pattern.search(error or ""):
            return kind, advice
    return "other", None


def _from_quality(report: dict) -> list[dict]:
    out = []
    for f in report.get("findings") or []:
        where = " / ".join(str(x) for x in (f.get("robot_ns"), f.get("node")) if x)
        out.append({
            "severity": f.get("severity", "medium"),
            "source": "quality",
            "kind": f.get("kind"),
            "where": where or None,
            "evidence": f.get("detail"),
            "suggestion": _FINDING_ADVICE.get(f.get("kind")),
        })
    return out


def _from_trace(trace: dict) -> tuple[list[dict], dict]:
    """Problems the planner hit on the way, and where its time went."""
    out: list[dict] = []
    by_class: dict[str, dict] = defaultdict(lambda: {"count": 0, "ops": set(),
                                                     "example": None})
    for point in trace.get("points") or []:
        op = point.get("operation_id")
        for failure in point.get("failures") or []:
            error = str(failure.get("error") or failure.get("reason") or "")
            kind, advice = _failure_class(error, failure.get("reason_code"))
            bucket = by_class[kind]
            bucket["advice"] = advice
            bucket["count"] += 1
            if op:
                bucket["ops"].add(op)
            bucket["example"] = bucket["example"] or error[:240]
        attempts = int(point.get("attempts") or 0)
        if attempts >= HOT_SPOT_ATTEMPTS:
            out.append({
                "severity": "medium", "source": "planning", "kind": "hot_spot",
                "where": f"{point.get('robot') or '?'} / {op}",
                "evidence": (f"{attempts} attempts before it "
                             f"{'made it into the tree' if (point.get('flags') or {}).get('in_tree') else 'was given up'}"),
                "suggestion": ("Look at this operation first: re-check its "
                               "position, orientation and which arm owns it."),
            })
    for kind, bucket in by_class.items():
        advice = bucket.get("advice") or next(
            (a for k, _p, a in _FAILURE_PATTERNS if k == kind), None)
        out.append({
            "severity": "medium" if bucket["count"] >= 3 else "low",
            "source": "planning", "kind": f"planner_failure:{kind}",
            "where": ", ".join(sorted(bucket["ops"])[:8]) or None,
            "evidence": f"{bucket['count']} failed attempt(s), e.g. {bucket['example']}",
            "suggestion": advice,
        })

    run = trace.get("run") or {}
    elapsed = float(run.get("elapsed_s") or 0.0)
    planner_s = 0.0
    calls: dict[str, int] = defaultdict(int)
    for d in trace.get("decisions") or []:
        tool = d.get("tool")
        if not tool:
            continue
        calls[tool] += 1
        metrics = (d.get("result") or {}).get("metrics") or {}
        try:
            planner_s += float(metrics.get("planning_time_s") or 0.0)
        except (TypeError, ValueError):
            pass
    tool_calls = int(run.get("tool_calls") or sum(calls.values()))
    budget = int(run.get("step_budget") or 0)
    time = {
        "elapsed_s": round(elapsed, 1),
        "motion_planner_s": round(planner_s, 1),
        # Planner time is summed per call, and batched calls overlap, so it can
        # exceed the wall clock; the remainder is then not meaningful.
        "other_s": round(elapsed - planner_s, 1) if planner_s <= elapsed else None,
        "other_is": "LLM reasoning, validation and tool overhead",
        "tool_calls": tool_calls,
        "step_budget": budget or None,
        "calls_by_tool": dict(sorted(calls.items(), key=lambda kv: -kv[1])),
    }
    if elapsed > 0 and planner_s <= elapsed and (elapsed - planner_s) / elapsed > 0.6:
        out.append({
            "severity": "low", "source": "planning", "kind": "time_in_reasoning",
            "where": None,
            "evidence": (f"{elapsed - planner_s:.0f} s of {elapsed:.0f} s was spent "
                         f"outside the motion planner ({tool_calls} tool calls)"),
            "suggestion": ("Fewer, larger steps: batch operations known to be "
                           "reachable into one run_trajectory_planner call, and fix "
                           "the recurring planner failures above — each failure "
                           "costs a reasoning turn."),
        })
    if budget and tool_calls / budget > STEP_BUDGET_WARN:
        out.append({
            "severity": "medium", "source": "planning", "kind": "step_budget",
            "where": None,
            "evidence": f"{tool_calls} of {budget} steps used",
            "suggestion": "The run nearly ran out of steps; a harder program will not finish.",
        })
    return out, time


def _from_program(report: dict, program: dict) -> list[dict]:
    target = ((program.get("extensions") or {}).get("cycle_time") or {})
    makespan = (report.get("summary") or {}).get("makespan_s")
    if not makespan or not target:
        return []
    max_s, target_s = target.get("max_s"), target.get("target_s")
    if max_s and makespan > max_s:
        sev, word = "high", f"over the {max_s} s maximum"
    elif target_s and makespan > target_s:
        sev, word = "medium", f"over the {target_s} s target"
    else:
        return []
    return [{
        "severity": sev, "source": "program", "kind": "cycle_time",
        "where": None,
        "evidence": f"makespan {makespan:.1f} s is {word}",
        "suggestion": ("Fix parallelism first (it is usually the largest share), "
                       "then the longest detours, then speed."),
    }]


def advise(report: dict, trace: Optional[dict] = None,
           program: Optional[dict] = None) -> dict:
    """One plan's problems and suggested fixes, worst first."""
    problems = _from_quality(report or {})
    time = None
    if trace:
        more, time = _from_trace(trace)
        problems += more
    if program:
        problems += _from_program(report or {}, program)
    problems.sort(key=lambda p: _SEVERITY_RANK.get(p.get("severity"), 9))

    counts = defaultdict(int)
    for p in problems:
        counts[p["severity"]] += 1
    summary = report.get("summary") or {}
    parts = [f"Plan {report.get('tree_id')}: {report.get('verdict', '?')} "
             f"(efficiency {report.get('efficiency_score')})"]
    if summary.get("makespan_s") is not None:
        parts.append(f"makespan {summary['makespan_s']} s")
    parts.append(", ".join(f"{n} {s}" for s, n in sorted(
        counts.items(), key=lambda kv: _SEVERITY_RANK.get(kv[0], 9))) or "no problems found")
    if problems and problems[0].get("suggestion"):
        parts.append(f"first fix: {problems[0]['suggestion']}")
    return {
        "success": True,
        "cell_id": report.get("cell_id"),
        "tree_id": report.get("tree_id"),
        "verdict": report.get("verdict"),
        "efficiency_score": report.get("efficiency_score"),
        "summary": " — ".join(parts),
        "problems": problems,
        "time_breakdown": time,
        "commanded": False,
    }


def advice_for(cell_id: str, tree_id: Optional[str] = None,
               program: Optional[str] = None) -> dict:
    """Load the report, trace and program for one stored plan and advise on it.

    Lets ``cell_store.CellStoreUnavailable`` propagate, like ``report_for``: an
    unreachable owner is not a plan with no problems.
    """
    from agents.trajectory import plan_trace
    from agents.trajectory.tools import trajectory_quality
    from utils import cell_store

    report = trajectory_quality.report_for(cell_id, tree_id)
    if not report.get("success"):
        return report
    trace = plan_trace.read(cell_id, report["tree_id"])
    task = None
    if program:
        if not plan_trace.is_safe_id(program):
            return {"success": False, "cell_id": cell_id,
                    "error": "program must be a plain identifier"}
        task = cell_store.read_json(cell_id, cell_store.KIND_TASKS, f"{program}.json")
    out = advise(report, trace, task)
    out["trace_found"] = trace is not None
    out["quality_source"] = report.get("source")
    return out
