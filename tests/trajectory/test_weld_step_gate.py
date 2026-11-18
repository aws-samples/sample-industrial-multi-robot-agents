"""A weld step with two arms in it is a pair, and the assembler must check it.

``_gate_multi_arm_step`` exists because the answer to "may these arms move at once?"
had TWO owners. The planner's system prompt told the LLM that inter-robot clearance
is enforced on parallel steps; the assembler enforced it on the return home only.
Every weld step was wrapped in a ``Parallel`` node exactly as it was handed over, so
the enforcement the prompt promised did not exist anywhere.

Measured on ``welding_cell_2_small``, step 3 of a shipped plan — ``Parallel(robot1 W02,
robot2 W15)`` — closed to **0.1153 m** against the cell's 0.2 m ``warning_distance``
and e-stopped the replay six nodes in. Nothing had ever looked at that pair. And it
cannot be caught upstream: the same pair is clear at BOTH endpoints (goal poses
0.9212 m apart, the widest of the plan's six pairs; start poses 0.6908 m) and closes
only in transit. Only the swept volumes see it, so the check belongs where the
trajectories are.

What these tests pin, for weld steps specifically — the home step's own tests are in
``test_home_step_gate.py``, and the point here is that BOTH go through the one gate:

  * a multi-arm weld step is gated at all, and a clear one still runs in parallel;
  * a step that would enter the warning band is emitted as a ``Sequence``, in an
    order that is DISCOVERED, and is reported as having cost cycle time;
  * a step with no safe order REFUSES the tree, and every such step is reported in
    ONE pass so the caller re-plans once instead of once per replay;
  * a declared ``SEQUENTIAL`` still has its order verified — an arm moving past
    another that merely STANDS is a real pair;
  * a gate that cannot run degrades VISIBLY (``gated: False``) rather than passing
    the step off as checked.

Host-only: the geometry is stubbed. What is under test is the wiring — what gets
checked, what gets emitted, and what gets refused.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"
ARMS = ["robot1", "robot2"]


def _leg(ns: str, step: int) -> list[dict]:
    """Arm ``ns``'s leg for 1-based ``step``, chained onto the previous one.

    The assembler refuses a tree whose operations do not start where the arm's
    previous operation ended, so the legs have to chain or every test here would
    fail on that instead of on the gate.
    """
    sign = 1.0 if ns == "robot1" else -1.0
    a, b = sign * 0.1 * step, sign * 0.1 * (step + 1)
    return [{"positions": [a] * 6, "time_s": 0.0},
            {"positions": [(a + b) / 2] * 6, "time_s": 1.0},
            {"positions": [b] * 6, "time_s": 2.0}]


# ── the LangChain surface ─────────────────────────────────────────────────────

@pytest.fixture
def bt(monkeypatch):
    """The behaviour_tree module, with a LangChain stub that has ``.func``.

    Same reasoning as ``test_home_step_gate.py``: the module calls
    ``build_behaviour_tree_node.func(...)``, so a stub providing only ``.invoke``
    (as other modules install at import time) is not enough. The real package is
    used when installed; any stub already cached is replaced for the duration.
    """
    def _tool(fn=None, **_kw):
        def _wrap(f):
            f.func = f
            f.invoke = lambda d: f(**d)
            return f
        return _wrap(fn) if fn is not None else _wrap

    cached = sys.modules.get("langchain_core.tools")
    if cached is None:
        try:
            import langchain_core.tools as cached  # noqa: F401
        except Exception:
            cached = None

    if getattr(cached, "__file__", None) is None:
        lc = sys.modules.get("langchain_core")
        if getattr(lc, "__file__", None) is None:
            lc = types.ModuleType("langchain_core")
            lc.__path__ = []
            monkeypatch.setitem(sys.modules, "langchain_core", lc)
        lct = types.ModuleType("langchain_core.tools")
        lct.tool = _tool
        monkeypatch.setattr(lc, "tools", lct, raising=False)
        monkeypatch.setitem(sys.modules, "langchain_core.tools", lct)

    for name in [n for n in list(sys.modules) if n.startswith("agents.trajectory.tools")]:
        del sys.modules[name]
    import importlib
    module = importlib.import_module("agents.trajectory.tools.behaviour_tree")
    yield module
    for name in [n for n in list(sys.modules) if n.startswith("agents.trajectory.tools")]:
        del sys.modules[name]


@pytest.fixture
def gate(monkeypatch):
    """Install a stand-in for the inter-robot gate. Returns the call log.

    ``rule(moving, parked)`` decides, where ``parked`` maps each standing arm to the
    pose it holds — which is what makes "clear once the other arm has finished"
    expressible, the real question a serialised step asks.
    """
    calls: list = []

    def install(rule, error: str | None = None):
        import agents.trajectory.tools.planning_tools as pt

        def fake(cell_id, trajectories, **kw):
            moving = [ns for ns, t in trajectories.items() if len(t) > 1]
            parked = {ns: t[0]["positions"] for ns, t in trajectories.items()
                      if len(t) == 1}
            calls.append({"moving": sorted(moving), "parked": sorted(parked)})
            if error:
                return {"error": error}
            ok = rule(sorted(moving), parked)
            return {"collision_free": ok, "warning_distance_m": 0.2,
                    "min_clearance_m": 0.42 if ok else 0.1153,
                    "robots": sorted(trajectories)}

        monkeypatch.setattr(pt, "validate_trajectory_set", fake)
        return calls

    return install


@pytest.fixture
def welds(bt):
    """Cache ``n`` chained weld legs per arm; return the matching plan."""
    from agents.trajectory.tools.trajectory_cache import cache_trajectory

    def install(n: int = 1, mode: str = "PARALLEL", arms: list[str] | None = None,
                home: bool = False) -> dict:
        arms = arms or ARMS
        steps = []
        for k in range(1, n + 1):
            for ns in arms:
                cache_trajectory(CELL, ns, f"{ns}_W{k}", _leg(ns, k))
            steps.append({"mode": mode,
                          "robots": [{"robot_ns": ns, "operation_id": f"{ns}_W{k}",
                                      "operation_type": "weld"} for ns in arms]})
        return {"steps": steps, "home_at_end": home}

    return install


# ── reading the emitted tree ──────────────────────────────────────────────────

def _ns(node: dict) -> str | None:
    return (node.get("ports") or {}).get("robot_ns") or node.get("robot_ns")


def _is_exec(node: dict) -> bool:
    return (node.get("type") or "").endswith("ExecuteTrajectory")


def _emitted_steps(root: dict) -> list[tuple[str, list[str]]]:
    """Every container that directly holds welds, as (container type, arm order)."""
    found: list = []

    def walk(node):
        kids = node.get("children") or []
        if any(_is_exec(k) for k in kids):
            found.append((node.get("type") or "",
                          [_ns(k) for k in kids if _is_exec(k)]))
        for k in kids:
            walk(k)

    walk(root)
    return found


# ── a clear pair is still a parallel pair ─────────────────────────────────────

def test_a_multi_arm_weld_step_is_gated_at_all(bt, gate, welds):
    """The regression itself: weld steps used to reach the executor unexamined."""
    calls = gate(lambda moving, parked: True)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=2))

    assert out["success"] is True, out.get("error")
    assert [g["step"] for g in out["step_gates"]] == [1, 2]
    assert all(g["gated"] is True for g in out["step_gates"])
    assert [c["moving"] for c in calls] == [ARMS, ARMS], (
        "each step's pair must be validated as the SET, once per step")


def test_a_clear_pair_still_welds_in_parallel(bt, gate, welds):
    """Cycle time is not given away on steps that are actually clear."""
    gate(lambda moving, parked: True)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=1))

    assert [g["mode"] for g in out["step_gates"]] == ["PARALLEL"]
    assert "serialised_steps" not in out
    kind, order = _emitted_steps(out["behaviour_tree"]["root"])[0]
    assert "Parallel" in kind
    assert sorted(order) == ARMS


def test_a_single_arm_step_is_not_pretended_to_be_gated(bt, gate, welds):
    """One arm is not a pair. Saying it was checked would be a lie about it."""
    calls = gate(lambda moving, parked: True)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=1, arms=["robot1"]))

    assert out["success"] is True, out.get("error")
    assert calls == [], "nothing to check, so nothing is called"
    assert out.get("step_gates", []) == [], "and nothing is reported as gated"


# ── a breach is separated in time, in a DISCOVERED order ─────────────────────

def test_a_pair_that_breaches_the_warning_band_is_not_shipped_in_parallel(
        bt, gate, welds):
    """The measured case: 0.1153 m against a 0.2 m band, both legs valid alone.

    The step is emitted as a ``Sequence`` of the same, untouched legs — only their
    timing changes — and the cost is reported, because it is cycle time.
    """
    gate(lambda moving, parked: len(moving) == 1)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=1))

    assert out["success"] is True, out.get("error")
    assert out["step_gates"][0]["mode"] == "SEQUENTIAL"
    assert out["step_gates"][0]["parallel"]["collision_free"] is False
    assert out["step_gates"][0]["parallel"]["min_clearance_m"] == 0.1153
    assert out["serialised_steps"] == [1], "serialising costs cycle time — report it"
    kind, order = _emitted_steps(out["behaviour_tree"]["root"])[0]
    assert "Sequence" in kind
    assert sorted(order) == ARMS


def test_the_weld_order_is_discovered_not_assumed(bt, gate, welds):
    """Only one order may be clear, and it need not be the obvious one.

    Here ``robot1`` may move only once ``robot2`` stands at the END of its own leg.
    A serialisation that assumed an order (alphabetical, or the plan's order) would
    ship the one sequence that still e-stops, having "checked" it.
    """
    def rule(moving, parked):
        if len(moving) != 1:
            return False
        if moving[0] == "robot2":
            return True
        return parked.get("robot2") == _leg("robot2", 1)[-1]["positions"]

    gate(rule)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=1))

    assert out["step_gates"][0]["order"] == ["robot2", "robot1"]
    _kind, order = _emitted_steps(out["behaviour_tree"]["root"])[0]
    assert order == ["robot2", "robot1"], (
        "the tree must execute the welds in the order that was proven clear")


def test_a_declared_sequential_step_still_has_its_order_verified(bt, gate, welds):
    """An arm moving past another that merely STANDS is a real pair.

    ``SEQUENTIAL`` is the caller's request, so the together-check is skipped — but
    the order is checked, not trusted.
    """
    calls = gate(lambda moving, parked: moving == ["robot2"] or "robot2" in parked)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=1, mode="SEQUENTIAL"))

    assert out["success"] is True, out.get("error")
    assert [c["moving"] for c in calls] != [], "a declared SEQUENTIAL is still gated"
    assert all(len(c["moving"]) == 1 for c in calls), (
        "no point asking whether a step the caller will not run together is clear")
    assert out["step_gates"][0]["mode"] == "SEQUENTIAL"
    assert out["step_gates"][0]["gated"] is True
    assert "parallel" not in out["step_gates"][0]


# ── no safe order refuses the tree, and says so about EVERY step ─────────────

def test_a_pair_with_no_safe_order_refuses_the_tree(bt, gate, welds):
    """Fail closed. A tree with a step that e-stops the cell is not a plan."""
    gate(lambda moving, parked: False)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=1))

    assert out["success"] is False
    assert "behaviour_tree" not in out
    assert [b["step"] for b in out["blocked_steps"]] == [1]
    assert out["blocked_steps"][0]["blocked_arms"] == ARMS
    assert out["blocked_steps"][0]["serial_attempts"], (
        "the attempts are the evidence for the refusal, so report them")
    assert "SWEPT" in out["error"], (
        "the refusal must say why re-pairing on positions will not fix it")


def test_every_blocked_step_is_reported_in_one_pass(bt, gate, welds):
    """Reporting the first breach only makes the caller re-plan once per replay."""
    gate(lambda moving, parked: False)

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=3))

    assert out["success"] is False
    assert [b["step"] for b in out["blocked_steps"]] == [1, 2, 3]
    assert [g["step"] for g in out["step_gates"]] == [1, 2, 3], (
        "the verdict for every gated step, not just the refused ones")


def test_the_same_arm_twice_in_one_step_is_refused(bt, gate, welds):
    """One arm cannot run two motions at once, and the gate checks one leg per arm.

    So the second leg would go through unchecked — the very hole this gate closes.
    """
    gate(lambda moving, parked: True)
    plan = welds(n=1)
    plan["steps"][0]["robots"].append(
        {"robot_ns": "robot1", "operation_id": "robot1_W2", "operation_type": "weld"})

    out = bt.assemble_behaviour_tree.func(CELL, plan)

    assert out["success"] is False
    assert out["repeated_arms"] == [
        {"step": 1, "robot_ns": "robot1", "operation_id": "robot1_W2"}]


# ── a gate that cannot run says so ───────────────────────────────────────────

def test_a_gate_that_cannot_run_degrades_visibly(bt, gate, welds):
    """``error`` means the check did not happen — it must not read as a breach.

    The legs were planned by the same geometry stack, so this is only reachable in
    a stubbed environment; refusing the tree there would make the assembler
    unusable without Pinocchio. It ships, and it says ``gated: False``.
    """
    gate(lambda moving, parked: True, error="no pinocchio")

    out = bt.assemble_behaviour_tree.func(CELL, welds(n=1))

    assert out["success"] is True, out.get("error")
    assert out["step_gates"][0]["mode"] == "PARALLEL"
    assert out["step_gates"][0]["gated"] is False, (
        "an unchecked step must never be reported as checked")
