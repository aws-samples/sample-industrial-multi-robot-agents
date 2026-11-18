"""Going home is an ordinary motion: it leaves the seam, and it is verified.

The last step of every tree is the arms returning to HOME, and it was the one
step nothing inter-robot ever checked. Not because the check was skipped — because
of WHO makes that step. Every other PARALLEL step is planned and validated by the
caller (``run_trajectory_planner`` → ``validate_trajectory_set``) before the
assembler ever sees it. The home step is created BY ``assemble_behaviour_tree``,
so no caller had the trajectories to validate, and the assembler ran no check of
its own. Each arm's leg was planned alone, against static obstacles only.

Measured on ``welding_cell_2_small``, from the tree of a run whose every weld step
passed the gate: the two home legs pass **0.089 m** apart (n=400 samples; 0.092 m
at n=20) against the cell's 0.2 m ``warning_distance``. The replay halted 18 nodes
in — ``robot1 ↔ robot2 clearance 0.1855 m ≤ warning 0.2 m`` — with both arms
otherwise idle, on the final motion of the program. Two individually valid legs.

What must hold now, and what these tests pin:

  * the set is planned COUPLED — ONE call for all arms, each planned against the
    swept volume of the arms already planned — not one independent call per arm;
  * the arm LEAVES THE SEAM on-axis first, in the coupled path too: that lift-off
    lived only in the single-arm planner, so which function planned the return
    decided whether the tool was dragged sideways out of the weld;
  * the emitted set is put through the SAME gate a parallel weld step gets;
  * a set that would enter the warning band is not shipped as ``Parallel``: the
    arms are homed one after another, in an order that is DISCOVERED, not assumed.
    Measured in vivo on ``welding_cell_2_small`` — arms finishing on W02 and W09 stand
    0.3564 m apart, yet their coupled legs meet at **-0.0695 m**. The order was not
    the obvious one: ``robot1`` first is -0.1007 m, ``robot2`` first is +0.3063 m,
    and ``robot1`` after ``robot2`` is home has 1.685 m. Serialising in the
    discovered order ``[robot2, robot1]`` is what makes that pair executable;
  * when no order is clear either, the tree is REFUSED — a plan whose last motion
    e-stops the cell is not a plan;
  * an arm whose return could not be planned at all is NAMED, because the blind
    fallback home is a motion no gate can cover.

Host-only: the geometry is stubbed. The numbers above are measurements from the
container (see ``test_home_planning_coupled.py`` for the real-geometry pins); what
is under test here is the wiring — which planner is called, what is gated, and
what is emitted.
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

# Two legs, each three points: on the seam → lifted → HOME.
LEGS = {
    "robot1": [{"positions": [0.10] * 6, "time_s": 0.0},
               {"positions": [0.20] * 6, "time_s": 1.0},
               {"positions": [0.30] * 6, "time_s": 2.0}],
    "robot2": [{"positions": [-0.10] * 6, "time_s": 0.0},
               {"positions": [-0.20] * 6, "time_s": 1.0},
               {"positions": [-0.30] * 6, "time_s": 2.0}],
}


# ── the LangChain surface ─────────────────────────────────────────────────────

@pytest.fixture
def bt(monkeypatch):
    """The behaviour_tree module, with a LangChain stub that has ``.func``.

    ``behaviour_tree`` calls ``build_behaviour_tree_node.func(...)`` — the real
    decorator's escape hatch — so a stub that only provides ``.invoke`` (as other
    test modules install at import time) is not enough: run in file order, this
    module would decorate the tools with somebody else's ``tool`` and fail on a
    missing ``.func``. So the REAL package is used when it is installed, and any
    stub already cached is replaced for the duration — then the tools package is
    dropped and re-imported so it binds to whichever of the two is in force, and
    dropped again on the way out so the next module is not left holding ours.
    """
    def _tool(fn=None, **_kw):
        def _wrap(f):
            f.func = f
            f.invoke = lambda d: f(**d)
            return f
        return _wrap(fn) if fn is not None else _wrap

    # A stub has no ``__file__``; the installed package does. Presence in
    # ``sys.modules`` proves nothing about which one it is.
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
            lc.__path__ = []  # a package, so a missing submodule says so
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

    ``rule(moving_ns, parked)`` decides. ``parked`` maps each standing arm to its
    held pose, which is what makes "clear once the other arm is at HOME" express-
    ible — the real question a serialised home asks.

    A call with NO movers is the step asking whether the arms are clear WHERE THEY
    STAND, which is a different question from any about motion: it is answered by
    ``standing`` (clear by default) so that a rule about moving arms does not
    accidentally decide it too.
    """
    calls: list = []

    def install(rule, standing: bool = True):
        import agents.trajectory.tools.planning_tools as pt

        def fake(cell_id, trajectories, **kw):
            # A leg whose pose never changes is an arm STANDING, not a mover. The
            # cached weld legs are such legs (two samples, one pose): the
            # assembler's weld-step gate checks them before the home step exists,
            # and that step is not what these tests are about, so it is clear.
            def moves(t):
                return len(t) > 1 and any(p["positions"] != t[0]["positions"] for p in t)
            moving = [ns for ns, t in trajectories.items() if moves(t)]
            parked = {ns: t[0]["positions"] for ns, t in trajectories.items()
                      if not moves(t)}
            calls.append({"moving": sorted(moving), "parked": sorted(parked)})
            if moving:
                ok = rule(moving, parked)
            elif all(len(t) == 1 for t in trajectories.values()):
                ok = standing           # the home step: are they clear where they stand?
            else:
                ok = True               # the (stationary) weld step
            return {"collision_free": ok, "warning_distance_m": 0.2,
                    "min_clearance_m": 0.42 if ok else 0.089,
                    "robots": sorted(trajectories)}

        monkeypatch.setattr(pt, "validate_trajectory_set", fake)
        return calls

    return install


@pytest.fixture
def planned(bt, monkeypatch):
    """Cache one weld per arm and stub the home planners. Returns the call log."""
    from agents.trajectory.tools.trajectory_cache import cache_trajectory

    def install(coupled=None, single=None):
        log = {"coupled": [], "single": []}
        for ns in ARMS:
            cache_trajectory(CELL, ns, f"{ns}_W1",
                             [{"positions": LEGS[ns][0]["positions"], "time_s": 0.0},
                              {"positions": LEGS[ns][0]["positions"], "time_s": 1.0}])

        import agents.trajectory.tools.ompl_planner as op

        def fake_coupled(cell_id, starts, *a, **kw):
            log["coupled"].append(sorted(starts))
            return dict(coupled if coupled is not None else LEGS)

        def fake_single(cell_id, robot_ns, *a, **kw):
            log["single"].append(robot_ns)
            return (single or {}).get(robot_ns)

        monkeypatch.setattr(op, "plan_home_trajectories", fake_coupled, raising=False)
        monkeypatch.setattr(op, "plan_home_trajectory", fake_single, raising=False)
        return log

    return install


def _plan() -> dict:
    return {"steps": [{"mode": "PARALLEL",
                       "robots": [{"robot_ns": ns, "operation_id": f"{ns}_W1"}
                                  for ns in ARMS]}]}


def _home_children(tree_root: dict) -> list[dict]:
    """The MoveHome nodes of the assembled tree, in execution order."""
    def walk(node):
        if (node.get("type") or "").endswith("MoveHome"):
            yield node
        for child in node.get("children") or []:
            yield from walk(child)
    return list(walk(tree_root))


# ── coupled, not one-arm-at-a-time ────────────────────────────────────────────

def test_the_home_set_is_planned_in_one_coupled_call(bt, gate, planned):
    """Independent per-arm plans are exactly how 0.089 m got emitted.

    Each leg was valid against the fixtures; neither knew about the other arm.
    The coupled planner feeds each planned arm's swept volume forward, so this
    pins that the assembler asks for the SET, once, with every arm in it.
    """
    gate(lambda moving, parked: True)
    log = planned()

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert out["success"] is True, out.get("error")
    assert log["coupled"] == [ARMS], "one coupled call carrying every arm"
    assert log["single"] == [], "no arm was planned home on its own"


def test_the_gate_sees_exactly_the_legs_that_are_emitted(bt, gate, planned):
    """A gate on trajectories other than the shipped ones checks nothing."""
    calls = gate(lambda moving, parked: True)
    planned()

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    # The weld step's own gate runs first; the home step's calls are the last two.
    home_calls = calls[-2:]
    assert home_calls[0]["moving"] == [], "the standing pair is checked before planning"
    assert [c["moving"] for c in home_calls[1:]] == [ARMS], "then the legs, together"
    emitted = _home_children(out["behaviour_tree"]["root"])
    assert len(emitted) == 2
    assert out["home_step"]["gated"] is True


def test_a_clear_set_homes_the_arms_together(bt, gate, planned):
    """Parallelism is not given up when the set is actually clear."""
    gate(lambda moving, parked: True)
    planned()

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert out["home_step"]["mode"] == "PARALLEL"
    assert out["home_step"]["coupled_arms"] == ARMS


# ── a breach is separated in time, in a DISCOVERED order ─────────────────────

def test_a_set_that_breaches_the_warning_band_is_not_shipped_in_parallel(
        bt, gate, planned):
    """The measured case: both legs valid alone, 0.089 m together.

    Simultaneous homing is refused and the arms are homed one after another. The
    legs themselves are untouched — only their timing changes.
    """
    gate(lambda moving, parked: len(moving) == 1)
    planned()

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert out["success"] is True, out.get("error")
    assert out["home_step"]["mode"] == "SEQUENTIAL"
    assert out["home_step"]["parallel"]["collision_free"] is False
    assert sorted(out["home_step"]["order"]) == ARMS


def test_the_homing_order_is_discovered_not_assumed(bt, gate, planned):
    """Only one order may be clear, and it need not be the obvious one.

    Here ``robot1`` may only move once ``robot2`` is at HOME. A serialisation that
    assumed an order (alphabetical, or the planning order) would ship the one
    sequence that still e-stops, having "checked" it.
    """
    def rule(moving, parked):
        if len(moving) != 1:
            return False
        if moving[0] == "robot2":
            return True
        # robot1 moves only when robot2 stands at the END of its own leg.
        return parked.get("robot2") == LEGS["robot2"][-1]["positions"]

    gate(rule)
    planned()

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert out["home_step"]["order"] == ["robot2", "robot1"]
    emitted = _home_children(out["behaviour_tree"]["root"])
    assert [n.get("ports", {}).get("robot_ns") or n.get("robot_ns")
            for n in emitted] == ["robot2", "robot1"], (
        "the tree must execute the homing in the order that was proven clear")


def test_arms_already_in_the_band_are_told_that_and_not_told_to_re_order(
        bt, gate, planned):
    """Measured: arms finishing on W01/W04 of ``welding_cell_2_small`` stand 0.0367 m
    apart — inside the 0.2 m band with nothing moving.

    The monitor measures the PAIR, not the mover, so it e-stops whichever arm goes
    first; no ordering separates poses the arms are already in. Reporting that as
    "no safe order" sends the caller hunting for an order that cannot exist, so it
    is diagnosed on its own — and it is checked before planning, because planning a
    return that cannot execute is time spent for nothing.
    """
    gate(lambda moving, parked: True, standing=False)
    log = planned()

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert out["success"] is False
    assert out["home_step"]["standing"]["collision_free"] is False
    assert out["home_step"]["blocked_arms"] == ARMS
    assert log["coupled"] == [], "no point planning a return that cannot execute"
    assert "STAND" in out["error"]
    assert "no order" not in out["error"], "that is the other fault, with another fix"


def test_no_safe_order_refuses_the_tree(bt, gate, planned):
    """Fail closed on the motion. A tree whose last step e-stops is not a plan."""
    gate(lambda moving, parked: False)
    planned()

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert out["success"] is False
    assert "HOME" in out["error"]
    assert "home_at_end" in out["error"], "the refusal must name a way forward"
    assert out["home_step"]["blocked_arms"] == ARMS
    assert out["home_step"]["serial_attempts"], "the attempts are evidence, so report them"


# ── what cannot be gated is named ────────────────────────────────────────────

def test_an_unplannable_return_is_named_not_silently_blind(bt, gate, planned):
    """The executor's blind straight line home is a motion nothing checks.

    It is still better than losing the whole tree, so it is allowed — but it is
    reported, because "gated" would otherwise be a lie about that arm.
    """
    gate(lambda moving, parked: True)
    planned(coupled={"robot1": LEGS["robot1"]}, single={})

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert out["success"] is True, out.get("error")
    assert out["home_step"]["blind_arms"] == ["robot2"]
    assert out["home_step"]["gated"] is False
    assert out["home_step"]["coupled_arms"] == ["robot1"]


def test_the_single_arm_planner_is_the_fallback_not_the_default(bt, gate, planned):
    """A leg the coupled plan misses is still planned rather than left blind."""
    gate(lambda moving, parked: True)
    log = planned(coupled={"robot1": LEGS["robot1"]},
                  single={"robot2": LEGS["robot2"]})

    out = bt.assemble_behaviour_tree.func(CELL, _plan())

    assert log["single"] == ["robot2"], "only the arm the coupled plan missed"
    assert out["home_step"]["blind_arms"] == []
    assert out["home_step"]["gated"] is True, (
        "an uncoupled leg is exactly the one that must face the gate")


# ── leaving the seam is part of going home, in BOTH planners ──────────────────

@pytest.fixture
def ompl(monkeypatch):
    """The planner module with OMPL stood in for and the real cell config."""
    monkeypatch.setenv("CELL_BASE_PATH", str(PROJECT_ROOT / "cells"))
    op = pytest.importorskip("agents.trajectory.tools.ompl_planner")
    monkeypatch.setattr(op, "_import_ompl", lambda: object())
    # Synthetic legs in a real cell: the emitted-leg check (covered by
    # test_upstream_fixes_port) is not what these tests are about.
    monkeypatch.setattr(op, "_home_leg_breach", lambda *a, **k: None)
    return op


def test_the_coordinated_return_leaves_the_seam_on_axis(ompl, monkeypatch):
    """The lift-off used to exist only in the single-arm planner.

    The coordinated one — the one the behaviour tree and the e-stop reset use —
    planned joint-space straight off the seam, dragging the tool sideways across
    the workpiece. Same motion, same rule: retract along the approach vector
    first, then transit.
    """
    seam = [0.10] * 6
    lifted = [0.15] * 6
    seen: list = []

    monkeypatch.setattr(ompl, "_home_liftoff",
                        lambda cell_id, ns, start_q, **kw: [list(lifted)])

    def fake_transit(cell_id, robot_ns, start_q, goal_q, **kw):
        seen.append([round(float(v), 4) for v in start_q])
        return [{"positions": [float(v) for v in start_q], "time_s": 0.0},
                {"positions": [float(v) for v in goal_q], "time_s": 2.0}]

    monkeypatch.setattr(ompl, "plan_joint_path", fake_transit)

    out = ompl.plan_home_trajectories(CELL, {"robot1": seam})

    assert seen == [lifted], "the transit must start AFTER the retract"
    traj = out["robot1"]
    assert traj[0]["positions"] == pytest.approx(seam), (
        "the return must start where the arm actually is — nothing teleports")
    assert any(p["positions"] == pytest.approx(lifted) for p in traj), \
        "the retract itself has to be in the shipped trajectory"
    times = [p["time_s"] for p in traj]
    assert times == sorted(times), "time must be monotonic across the join"


def test_both_returns_leave_the_seam_through_the_same_function(ompl, monkeypatch):
    """Two lift-off implementations would drift, and one of them is the safe one.

    ``_home_liftoff`` owns "how a return leaves the weld". Pinned for both entry
    points, because the bug was precisely that only one of them did it.
    """
    import agents.trajectory.tools.robot_kinematics as rk

    asked: list = []
    monkeypatch.setattr(ompl, "_home_liftoff",
                        lambda cell_id, ns, start_q, **kw: asked.append(ns) or [])
    monkeypatch.setattr(ompl, "plan_joint_path",
                        lambda *a, **kw: [{"positions": [0.0] * 6, "time_s": 0.0}])
    monkeypatch.setattr(ompl, "_plan_ompl_segment",
                        lambda *a, **kw: ([[0.0] * 6, [0.1] * 6], None))
    # The single-arm planner loads the arm's kinematics before it retracts; on the
    # host there is no Pinocchio, and a model it cannot load makes it bail out
    # BEFORE the lift-off — which would let this test pass for the wrong reason.
    monkeypatch.setattr(rk, "_load_pinocchio_model",
                        lambda *a, **kw: (types.SimpleNamespace(nq=6, frames=[]), None))

    ompl.plan_home_trajectories(CELL, {"robot1": [0.1] * 6})
    ompl.plan_home_trajectory(CELL, "robot2", [0.1] * 6, [0.0] * 6)

    assert asked == ["robot1", "robot2"], (
        f"both returns must consult the lift-off owner, got {asked}")
