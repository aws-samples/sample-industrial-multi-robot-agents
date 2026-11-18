"""Strategies for a HOME return that OMPL could not find on its own.

OMPL answers one question: *is there a path from A to B?* It does not invent a
different way to pose the problem. When the arm finishes deep inside a fixture,
or the other arm is parked across the only corridor home, RRTConnect can search
its whole time budget and legitimately report failure — the direct problem has no
solution at the margin we demand.

That failure used to end the matter: the arm kept a bare ``MoveHome`` node and the
executor drove it home on a blind straight line — the single most dangerous motion
in a replay, taken precisely when planning had just told us the space is hard.

This module supplies the missing step. It proposes STRATEGIES — fold the arm up
before swinging the base, retract along the tool axis first, stage through a
clear pose — as intermediate joint waypoints, and the LLM is asked for more when
the cheap deterministic ones do not work.

The safety boundary is absolute and it is the same one the recovery layer in
``agents/cell/agent.py`` holds: **the LLM never produces a trajectory.** It
proposes waypoints; every leg between them is still planned by OMPL and checked
by the same validity checker, against the same static obstacles and the same
swept volumes of the other arms. A proposal is a hypothesis about where to route,
never a motion. An inadmissible or unplannable proposal is discarded, and if
every strategy fails the caller still falls back to the blind home it would have
used anyway — so this can only add safety, never subtract it.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Optional, Sequence


STRATEGY_SYSTEM_PROMPT = """\
You are the trajectory planner's STRATEGY layer for a multi-robot welding cell.

OMPL (RRTConnect) has FAILED to find a collision-free joint-space path returning
one arm from its current pose to HOME. You do NOT produce trajectories and you do
NOT drive robots. You propose intermediate joint configurations ("staging poses")
that break the return into legs OMPL can each solve. Every leg you propose is
then planned and collision-checked by OMPL — a bad proposal is discarded, not
executed, so propose boldly but physically.

Why a direct plan fails, and what usually fixes it:
  - The arm is extended into a fixture or over the workpiece. → Fold it up FIRST
    (elbow/shoulder toward their HOME values) while leaving joint 1 where it is,
    so the arm becomes compact before it swings.
  - The base must rotate through the other arm's space. → Reach a compact pose
    first, then rotate; a folded arm sweeps a far smaller volume.
  - The tool is inside a cavity or under an overhang. → Retract along the tool
    axis (wrist joints unchanged, shoulder/elbow backed off) before anything else.
  - The corridor home is narrow. → Stage through a high, clear pose above the
    workpiece, then descend to HOME.

Rules:
  - Joint values are RADIANS, in joint order, exactly as many values as the arm
    has joints.
  - Do NOT repeat the start pose or HOME — those are added for you. Give only the
    poses IN BETWEEN, in the order they should be reached.
  - Stay inside the joint limits you are given.
  - 1 or 2 staging poses per strategy. More legs means more places to fail.
  - Order strategies best-first, at most 3.

Respond with a SINGLE fenced ```json block and nothing else:
{
  "strategies": [
    {"waypoints": [[j1, j2, ...], ...], "rationale": "<one sentence>"}
  ]
}
"""


# ─── Deterministic strategies ─────────────────────────────────────────────────

def deterministic_strategies(start_q: Sequence[float],
                             home_q: Sequence[float]) -> list[dict]:
    """Cheap staging poses to try before spending an LLM call.

    These encode the one thing that is nearly always true of an arm that cannot
    get home: it is EXTENDED. A 6-DOF arm folded into its home shape sweeps a
    small volume; the same arm reaching into a fixture sweeps most of the cell.
    So the first and best candidate keeps joint 1 (the base) exactly where it is
    and moves everything else to HOME — the arm folds in place, then swings home
    already compact. The reverse (swing first, unfold later) occasionally wins
    when the obstruction is behind the arm, so it is offered second.
    """
    s = [float(v) for v in start_q]
    h = [float(v) for v in home_q]
    n = min(len(s), len(h))
    if n < 2:
        return []
    fold_in_place = [s[0]] + h[1:n]
    rotate_first = [h[0]] + s[1:n]
    half_fold = [s[0]] + [(s[i] + h[i]) / 2.0 for i in range(1, n)]
    return [
        {"waypoints": [fold_in_place], "source": "heuristic",
         "rationale": "fold the arm into its home shape without moving the base, "
                      "then swing home compact"},
        {"waypoints": [half_fold, fold_in_place], "source": "heuristic",
         "rationale": "fold in two steps, in case the fully folded pose is not "
                      "directly reachable"},
        {"waypoints": [rotate_first], "source": "heuristic",
         "rationale": "swing the base clear first, then unfold — for an "
                      "obstruction behind the arm"},
    ]


# ─── Admissibility ────────────────────────────────────────────────────────────

def admissible(wp: Sequence[float], nq: int,
               lower: Optional[Sequence[float]] = None,
               upper: Optional[Sequence[float]] = None) -> bool:
    """Is a proposed staging pose worth handing to OMPL at all?

    Arity and joint limits are checked here rather than left to the planner: a
    pose outside the limits is not a hard problem, it is a malformed one, and
    burning a 5-second OMPL budget to discover that is waste. This is a cheap
    filter, NOT the safety check — collisions are still OMPL's call.
    """
    try:
        q = [float(v) for v in wp]
    except (TypeError, ValueError):
        return False
    if len(q) != nq or any(math.isnan(v) or math.isinf(v) for v in q):
        return False
    if lower is not None and any(q[i] < float(lower[i]) for i in range(min(nq, len(lower)))):
        return False
    if upper is not None and any(q[i] > float(upper[i]) for i in range(min(nq, len(upper)))):
        return False
    return True


# ─── Scene summary for the prompt ─────────────────────────────────────────────

def _obstacle_summary(cell_id: str, limit: int = 12) -> str:
    """A compact scene description: one line per fixture, not per voxel.

    Fixtures are stored as voxel shells — hundreds of ``<id>_vNNN`` boxes for a
    single physical object. Pasting those into a prompt buys nothing but tokens,
    so boxes are grouped back by their base id and reported as one bounding box.
    """
    from agents.trajectory.tools.robot_kinematics import load_static_obstacles
    try:
        obstacles = load_static_obstacles(cell_id)
    except Exception:
        # A prompt fragment, not a safety gate: the planner's own collision
        # checking reads the same obstacles through the same owner and refuses
        # there. Saying so beats inventing an empty cell for the LLM.
        return "(obstacle model unreadable)"
    if not obstacles:
        return "(no obstacle model available)"

    groups: dict = {}
    for o in obstacles:
        oid = re.sub(r"_v\d+$", "", str(o.get("id", "?")))
        c = o.get("center") or o.get("position")
        if o.get("type") == "plane" or not c:
            groups.setdefault(oid, {"n": 0, "plane": True})["n"] += 1
            continue
        he = o.get("half_extents") or [o.get("radius", 0.0)] * 3
        g = groups.setdefault(oid, {"n": 0, "lo": None, "hi": None})
        lo = [float(c[i]) - float(he[i]) for i in range(3)]
        hi = [float(c[i]) + float(he[i]) for i in range(3)]
        g["lo"] = lo if g.get("lo") is None else [min(a, b) for a, b in zip(g["lo"], lo)]
        g["hi"] = hi if g.get("hi") is None else [max(a, b) for a, b in zip(g["hi"], hi)]
        g["n"] += 1

    lines = []
    for oid, g in list(groups.items())[:limit]:
        if g.get("plane"):
            lines.append(f"  - {oid}: ground/plane constraint")
        else:
            lo, hi = g["lo"], g["hi"]
            lines.append(
                f"  - {oid}: x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
                f"z[{lo[2]:.2f},{hi[2]:.2f}] ({g['n']} boxes)")
    if len(groups) > limit:
        lines.append(f"  - ... and {len(groups) - limit} more fixtures")
    return "\n".join(lines) or "(no obstacles)"


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = (re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
         or re.search(r"(\{.*\})", text, re.DOTALL))
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


# ─── The LLM proposal ─────────────────────────────────────────────────────────

def llm_strategies(cell_id: str, robot_ns: str, start_q: Sequence[float],
                   home_q: Sequence[float], others: Optional[dict] = None,
                   attempted_margins_m: Optional[Sequence[float]] = None,
                   already_tried: Optional[Sequence[dict]] = None,
                   lower: Optional[Sequence[float]] = None,
                   upper: Optional[Sequence[float]] = None) -> list[dict]:
    """Ask the LLM for staging poses. Returns [] on any failure — never raises.

    Set ``HOME_STRATEGY_LLM=0`` to skip the call entirely (offline runs, tests,
    and anywhere a deterministic result matters more than a clever one).
    """
    if os.environ.get("HOME_STRATEGY_LLM", "1").lower() in ("0", "false", "no"):
        return []

    def _deg(q) -> str:
        return "[" + ", ".join(f"{math.degrees(float(v)):.0f}" for v in q) + "]"

    tried = "\n".join(
        f"  - {s.get('rationale', '?')}: waypoints {s.get('waypoints')}"
        for s in (already_tried or [])) or "  (none)"
    occupied = "\n".join(
        f"  - {ns} is at {list(map(lambda v: round(float(v), 3), q))} "
        f"({_deg(q)} deg) and is also returning home"
        for ns, q in (others or {}).items()) or "  (no other arms)"
    margins = ", ".join(f"{m:.2f}" for m in (attempted_margins_m or [])) or "n/a"

    prompt = (
        f"Cell '{cell_id}', arm '{robot_ns}' cannot reach HOME.\n\n"
        f"Current pose (rad): {[round(float(v), 4) for v in start_q]}\n"
        f"Current pose (deg): {_deg(start_q)}\n"
        f"HOME pose (rad):    {[round(float(v), 4) for v in home_q]}\n"
        f"HOME pose (deg):    {_deg(home_q)}\n"
        f"Joint limits lower: {[round(float(v), 3) for v in (lower or [])]}\n"
        f"Joint limits upper: {[round(float(v), 3) for v in (upper or [])]}\n\n"
        f"Other arms in the cell:\n{occupied}\n\n"
        f"Static fixtures (world-frame bounding boxes, metres):\n"
        f"{_obstacle_summary(cell_id)}\n\n"
        f"A direct plan was attempted and FAILED at every inter-arm clearance "
        f"margin: {margins} m.\n"
        f"These staging strategies were already tried and also failed to plan:\n"
        f"{tried}\n\n"
        "Propose different staging poses. Respond with the json block."
    )

    try:
        from agents.trajectory.agent import _build_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        reply = _build_llm().invoke([SystemMessage(content=STRATEGY_SYSTEM_PROMPT),
                                     HumanMessage(content=prompt)])
        content = getattr(reply, "content", reply)
        if isinstance(content, list):  # Bedrock returns content blocks
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict)) or str(content)
        parsed = _extract_json(str(content)) or {}
    except Exception as exc:  # noqa: BLE001 — no LLM is a degradation, not an error
        print(f"[home-strategy] {robot_ns}: LLM unavailable ({exc}) — "
              f"deterministic staging only", flush=True)
        return []

    nq = len(list(home_q))
    out: list[dict] = []
    for s in (parsed.get("strategies") or [])[:3]:
        wps = [list(w) for w in (s.get("waypoints") or [])]
        if not wps or not all(admissible(w, nq, lower, upper) for w in wps):
            print(f"[home-strategy] {robot_ns}: discarded an inadmissible "
                  f"proposal ({s.get('rationale', 'no rationale')})", flush=True)
            continue
        out.append({"waypoints": wps, "source": "llm",
                    "rationale": str(s.get("rationale", ""))[:200]})
    return out


def propose_home_strategies(cell_id: str, robot_ns: str, start_q: Sequence[float],
                            home_q: Sequence[float], others: Optional[dict] = None,
                            attempted_margins_m: Optional[Sequence[float]] = None,
                            lower: Optional[Sequence[float]] = None,
                            upper: Optional[Sequence[float]] = None):
    """Staging strategies to try, cheapest first.

    A GENERATOR, deliberately: the caller stops at the first strategy that plans,
    so the LLM is never called for an arm the deterministic candidates rescue.
    Those cost nothing, they are the moves an operator would make, and when one
    of them plans there is no reason to have asked a model. The LLM is paid for
    only where a cell-specific insight — this fixture, that parked arm — is what
    the deterministic moves cannot supply.

    Yields dicts ``{"waypoints": [[...], ...], "source", "rationale"}``. The
    caller must plan and validate every leg; nothing here has been checked for
    collisions.
    """
    nq = len(list(home_q))
    det = [s for s in deterministic_strategies(start_q, home_q)
           if all(admissible(w, nq, lower, upper) for w in s["waypoints"])]
    yield from det
    yield from llm_strategies(
        cell_id, robot_ns, start_q, home_q, others=others,
        attempted_margins_m=attempted_margins_m, already_tried=det,
        lower=lower, upper=upper)
