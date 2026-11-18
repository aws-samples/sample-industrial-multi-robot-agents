"""What the planning agent can do about each failure — one table, read by code.

Every ``reason_code`` the planning tools return (backends, the dispatch facade,
the step gate, the assembler) has an entry here, and the recovery skill
(``skills/*/recover-from-planner-failure/SKILL.md``) has a section for each
(``tests/trajectory/test_reason_codes.py`` holds both to it). The facade attaches
the entry to every failure (``recovery``), so the agent reads the levers next to
the failure instead of re-deriving them, and ``plan_advice`` keys on the code,
not on error text.

Fields:
  retry_same   re-sending the SAME request can succeed (a sampling planner may
               find a path next time); False means the same request is the same
               answer — change something first.
  levers       what to change, cheapest first (request fields / tools).
  escalate     when the levers are spent: ``reallocate`` (another arm),
               ``resequence`` (order / pairing / a reposition move),
               ``fix_cell`` (geometry or config — tell the operator), or
               ``stop`` (infrastructure: report, do not retry).
"""
from __future__ import annotations

from typing import Optional

RECOVERY: dict = {
    # ── the weld / target pose ───────────────────────────────────────────────
    "ik_failed": dict(
        retry_same=False,
        levers=["check_reachability", "another arm"],
        escalate="reallocate"),
    "goal_in_collision": dict(
        retry_same=False,
        levers=["the target_diagnosis suggestion (move the weld out of the box)",
                "shorter approach_offset_m", "via point / stage-above"],
        escalate="reallocate"),
    "start_in_collision": dict(
        retry_same=False,
        levers=["re-plan from HOME (omit current_joints)", "a retract move first"],
        escalate="resequence"),
    "goal_self_collision": dict(
        retry_same=False,
        levers=["suggested_approach_offset_m (the pose search already ran: "
                "orientation_search)", "another arm"],
        escalate="reallocate"),
    "start_self_collision": dict(
        retry_same=False,
        levers=["re-plan from HOME"],
        escalate="resequence"),
    "approach_self_collision": dict(
        retry_same=False,
        levers=["suggested_approach_offset_m", "another arm"],
        escalate="reallocate"),
    "approach_mismatch": dict(
        retry_same=False,
        levers=["re-plan the weld (its working stroke missed the axis, the point or "
                "a measurable entry: see approach_check)", "shorter approach_offset_m"],
        escalate="reallocate"),
    "approach_unreachable": dict(
        retry_same=False,
        levers=["suggested_approach_offset_m", "another arm"],
        escalate="reallocate"),
    # ── the path ─────────────────────────────────────────────────────────────
    "no_path_found": dict(
        retry_same=True,
        levers=["intermediate via point", "plan the step's arms one per call",
                "move the other arm out first"],
        escalate="reallocate"),
    "trajectory_in_collision": dict(
        retry_same=True,
        levers=["re-plan (sampling)", "via point away from the named object"],
        escalate="reallocate"),
    "trajectory_self_collision": dict(
        retry_same=True,
        levers=["re-plan (sampling)", "via point", "another arm"],
        escalate="reallocate"),
    "excessive_tool_rise": dict(
        retry_same=True,
        levers=["via point around (not over) the obstruction",
                "a staging move before the weld"],
        escalate="resequence"),
    "joint_limit_violation": dict(
        retry_same=True,
        levers=["re-plan (another IK branch)", "intermediate via point"],
        escalate="reallocate"),
    "start_outside_joint_limits": dict(
        retry_same=False,
        levers=["re-plan from HOME / a valid current_joints"],
        escalate="stop"),
    "goal_outside_joint_limits": dict(
        retry_same=False,
        levers=["another arm"],
        escalate="reallocate"),
    # ── the other arms ───────────────────────────────────────────────────────
    "inter_arm_clearance": dict(
        retry_same=False,
        levers=["delay_fix start_delays", "pair a different pending weld",
                "find_interlock_set"],
        escalate="resequence"),
    "liftoff_blocked": dict(
        retry_same=False,
        levers=["shorter retreat_offset_m", "plan this leg after the other arm moved",
                "park the other arm first"],
        escalate="resequence"),
    "blocked_by_parked_arm": dict(
        retry_same=False,
        levers=["a 'move' that takes the arm in blocked_by clear first (toward its own "
                "base, raised), then this weld", "move that arm in the same step"],
        escalate="resequence"),
    "reallocation_not_needed": dict(
        retry_same=False,
        levers=["find_clear_pose for the blocking arm, one move, then the weld again"],
        escalate="resequence"),
    "step_refused": dict(
        retry_same=False,
        levers=["fix the arm named in caused_by; this arm's weld is fine"],
        escalate="resequence"),
    "step_gate_refused": dict(
        retry_same=False,
        levers=["delay_fix start_delays", "pair a different pending weld",
                "find_interlock_set"],
        escalate="resequence"),
    "compound_start_invalid": dict(
        retry_same=False,
        levers=["the arms already conflict where they stand: move one first"],
        escalate="resequence"),
    "compound_goal_invalid": dict(
        retry_same=False,
        levers=["pair a different pending weld (the two goals conflict)"],
        escalate="resequence"),
    # ── the return home ──────────────────────────────────────────────────────
    "home_liftoff_blocked": dict(
        retry_same=False,
        levers=["end the arm's sequence with a 'move' away from its last seam",
                "re-order so its last weld is one it can leave"],
        escalate="resequence"),
    "home_return_refused": dict(
        retry_same=True,
        levers=["a reposition 'move' as the arm's last operation",
                "home_at_end: false and plan the return as an operation"],
        escalate="resequence"),
    # ── program structure ────────────────────────────────────────────────────
    "start_gap": dict(
        retry_same=False,
        levers=["re-plan the arm's first operation with current_joints = HOME"],
        escalate="resequence"),
    "chain_gap": dict(
        retry_same=False,
        levers=["re-plan to_operation_id from from_operation_id's end, in order"],
        escalate="resequence"),
    "missing_trajectory": dict(
        retry_same=False,
        levers=["plan the operation before assembling"],
        escalate="resequence"),
    "not_planned": dict(
        retry_same=True,
        levers=["fix the step the batch stopped at, then send it and the skipped_steps "
                "again (not the kept ones)"],
        escalate="resequence"),
    "duplicate_robot_request": dict(
        retry_same=False,
        levers=["one request per arm per call: pass a sequence as steps"],
        escalate="resequence"),
    "retry_cap_reached": dict(
        retry_same=False,
        levers=["hand the weld to another arm (reallocate_operation, tried_robots "
                "including this arm)", "for a helper waypoint: plan the weld without it"],
        escalate="reallocate"),
    # ── allocation / operator ────────────────────────────────────────────────
    "assignment_violation": dict(
        retry_same=False,
        levers=["plan the operation on the arm the operator locked it to"],
        escalate="stop"),
    "robot_locked": dict(
        retry_same=False,
        levers=["the operator locked this arm: use another"],
        escalate="reallocate"),
    "no_candidate_robot": dict(
        retry_same=False,
        levers=[],
        escalate="fix_cell"),
    "out_of_reach_everywhere": dict(
        retry_same=False,
        levers=[],
        escalate="fix_cell"),
    # ── infrastructure ───────────────────────────────────────────────────────
    "self_collision_unavailable": dict(
        retry_same=False,
        levers=[],
        escalate="stop"),
    "self_collision_check_error": dict(
        retry_same=True,
        levers=["retry once"],
        escalate="stop"),
    "leg_check_error": dict(
        retry_same=True,
        levers=["retry once"],
        escalate="stop"),
    "inter_arm_check_error": dict(
        retry_same=True,
        levers=["retry once"],
        escalate="stop"),
    "backend_contract_violation": dict(
        retry_same=False,
        levers=["plan with the default backend"],
        escalate="stop"),
    "tool_error": dict(
        retry_same=True,
        levers=["retry once"],
        escalate="stop"),
    "planner_error": dict(
        retry_same=True,
        levers=["retry once"],
        escalate="stop"),
}

#: Codes with a wildcard part emitted from an f-string, and what they expand to.
PATTERNS = {
    "{which}_in_collision": ("start_in_collision", "goal_in_collision"),
    "{which}_self_collision": ("start_self_collision", "goal_self_collision"),
    "{which}_outside_joint_limits": ("start_outside_joint_limits",
                                     "goal_outside_joint_limits"),
    "{label}_outside_joint_limits": ("start_outside_joint_limits",
                                     "goal_outside_joint_limits"),
    "compound_{label}_invalid": ("compound_start_invalid", "compound_goal_invalid"),
}


def recovery(code: Optional[str]) -> Optional[dict]:
    """The table entry for ``code`` (a copy), or None for an unknown code."""
    hit = RECOVERY.get(code or "")
    return None if hit is None else {**hit, "levers": list(hit["levers"])}
