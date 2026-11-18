"""A staged HOME return is still OMPL's work; the strategy only says where to route.

``home_strategy`` proposes staging poses — deterministically, then via the LLM.
This is the other half: the planner decides whether a proposal is REAL. Every leg
between poses is planned by OMPL against the same static obstacles and the same
swept volumes of the arms already planned, and a strategy counts only when EVERY
leg plans and the last one lands on the cell's own HOME.

What must hold, whatever a model proposes:

  * staging is attempted only after a direct path has failed at every margin;
  * no leg is ever assumed — an unplannable leg discards the whole strategy;
  * a partially planned strategy is never returned, so the arm cannot be sent to
    a staging pose it has no planned way to leave;
  * a staged return ends on exactly the pose a direct return would have;
  * when nothing plans, the result is no trajectory — the caller's blind fallback,
    loudly announced, is still the floor.

The planner is stubbed throughout: this is the contract between the layers, and
it must hold offline in under a second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"
NS = "robot1"
START = [1.2, -0.4, 0.9, 0.3, -0.5, 0.2]
HOME = [0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0]
RADII = [0.32, 0.27, 0.22, 0.17, 0.14]


@pytest.fixture
def ompl():
    return pytest.importorskip("agents.trajectory.tools.ompl_planner",
                               reason="needs the trajectory runtime")


@pytest.fixture
def stub_legs(ompl, monkeypatch):
    """Replace OMPL with a recorded stub. Returns the call log.

    The stub's legs are synthetic joint values in a real cell, so the planner's
    emitted-leg check (which would rightly re-route them around the cell's real
    fixtures) is stood in for too: these tests are about the staging ladder.

    ``unplannable`` marks goals no leg may reach; ``unplannable_legs`` marks
    (start, goal) pairs, which is how a DIRECT path is failed without also
    failing a staged path's final leg to the same HOME.
    """
    monkeypatch.setattr(ompl, "_home_leg_breach", lambda *a, **k: None)

    def install(unplannable=(), unplannable_legs=(), plan_all: bool = False):
        calls: list = []
        blocked = [list(map(float, q)) for q in unplannable]
        blocked_legs = [(list(map(float, a)), list(map(float, b)))
                        for a, b in unplannable_legs]

        def fake_plan_joint_path(cell_id, robot_ns, start_q, goal_q,
                                 moving_obstacle_spheres=None, velocity_scale=0.4):
            calls.append({
                "robot_ns": robot_ns,
                "start": [float(v) for v in start_q],
                "goal": [float(v) for v in goal_q],
                "spheres": list(moving_obstacle_spheres or []),
            })
            leg = ([float(v) for v in start_q], [float(v) for v in goal_q])
            if not plan_all and (leg[1] in blocked or leg in blocked_legs):
                return None
            return [{"positions": [float(v) for v in start_q], "time_s": 0.0},
                    {"positions": [float(v) for v in start_q], "time_s": 0.5},
                    {"positions": [float(v) for v in goal_q], "time_s": 1.0}]

        monkeypatch.setattr(ompl, "plan_joint_path", fake_plan_joint_path)
        return calls

    return install


@pytest.fixture
def stub_strategies(monkeypatch):
    """Pin the strategy source, so these tests are about validation, not proposals."""
    def install(*strategies):
        mod = pytest.importorskip("agents.trajectory.tools.home_strategy")
        monkeypatch.setattr(mod, "propose_home_strategies",
                            lambda *a, **k: iter(strategies))
        return mod

    def one(waypoints, source="heuristic", rationale="stub"):
        return {"waypoints": waypoints, "source": source, "rationale": rationale}

    install.one = one
    return install


# ── a staged return is a real, whole, checked path ───────────────────────────

def test_a_staged_return_ends_exactly_at_the_cells_home(ompl, stub_legs, stub_strategies):
    """Staging changes the ROUTE, never the destination.

    An arm parked at a staging pose it reached but never left is not home, and the
    plan the replay is about to run was validated against HOME.
    """
    stub_strategies(stub_strategies.one([[0.5] * 6]))
    stub_legs()

    traj, strategy = ompl._plan_staged_home(
        CELL, NS, START, HOME, [], RADII, 0.4)

    assert traj[-1]["positions"] == pytest.approx(HOME)
    assert traj[0]["positions"] == pytest.approx(START)
    assert strategy["waypoints"] == [[0.5] * 6]


def test_every_leg_is_planned_by_ompl_in_order(ompl, stub_legs, stub_strategies):
    """The staging poses are hypotheses; only OMPL turns them into motion."""
    stub_strategies(stub_strategies.one([[0.5] * 6, [0.2] * 6]))
    calls = stub_legs()

    ompl._plan_staged_home(CELL, NS, START, HOME, [], RADII, 0.4)

    assert [c["goal"] for c in calls] == [[0.5] * 6, [0.2] * 6, HOME]
    assert [c["start"] for c in calls] == [START, [0.5] * 6, [0.2] * 6], (
        "each leg must start where the previous one actually ended"
    )


def test_a_strategy_is_abandoned_whole_when_any_leg_cannot_be_planned(
        ompl, stub_legs, stub_strategies):
    """Half a staged return is worse than none: the arm would strand mid-cell."""
    stub_strategies(
        stub_strategies.one([[0.5] * 6], rationale="first choice"),
        stub_strategies.one([[0.2] * 6], rationale="second choice"),
    )
    # The first strategy reaches its staging pose but cannot get home from there.
    calls = stub_legs(unplannable=[HOME])

    traj, strategy = ompl._plan_staged_home(CELL, NS, START, HOME, [], RADII, 0.4)

    assert traj is None and strategy is None, (
        "reaching the staging pose is not progress if the arm cannot leave it"
    )
    goals = [c["goal"] for c in calls]
    assert [0.5] * 6 in goals, "the first strategy's staging pose was attempted"
    assert [0.2] * 6 in goals, "and the second strategy was tried after it failed"


def test_the_next_strategy_is_tried_when_the_first_fails(
        ompl, stub_legs, stub_strategies):
    stub_strategies(
        stub_strategies.one([[0.5] * 6], rationale="first choice"),
        stub_strategies.one([[0.2] * 6], rationale="second choice"),
    )
    stub_legs(unplannable=[[0.5] * 6])

    traj, strategy = ompl._plan_staged_home(CELL, NS, START, HOME, [], RADII, 0.4)

    assert strategy["rationale"] == "second choice"
    assert traj[-1]["positions"] == pytest.approx(HOME)


def test_no_workable_strategy_yields_no_trajectory(ompl, stub_legs, stub_strategies):
    """Fail closed: the caller's blind fallback is the floor, not a stitched guess."""
    stub_strategies(stub_strategies.one([[0.5] * 6]))
    stub_legs(unplannable=[[0.5] * 6])

    assert ompl._plan_staged_home(CELL, NS, START, HOME, [], RADII, 0.4) == (None, None)


def test_a_strategy_source_that_raises_does_not_break_the_return(
        ompl, stub_legs, monkeypatch, capsys):
    """A broken proposal layer must degrade to the blind home, not to an exception."""
    mod = pytest.importorskip("agents.trajectory.tools.home_strategy")

    def _boom(*a, **k):
        raise RuntimeError("strategy layer exploded")

    monkeypatch.setattr(mod, "propose_home_strategies", _boom)
    stub_legs()

    assert ompl._plan_staged_home(CELL, NS, START, HOME, [], RADII, 0.4) == (None, None)
    assert "no strategies available" in capsys.readouterr().out


# ── the coupling that makes a simultaneous reset safe is not lost ────────────

def test_staged_legs_are_planned_against_the_other_arms_swept_volume(
        ompl, stub_legs, stub_strategies):
    """The whole point of coordinated homing must survive the staged path too."""
    stub_strategies(stub_strategies.one([[0.5] * 6]))
    calls = stub_legs()
    centres = [(1.0, 2.0, 3.0), (1.1, 2.0, 3.0)]

    ompl._plan_staged_home(CELL, NS, START, HOME, centres, RADII, 0.4)

    for c in calls:
        assert [p for p, _ in c["spheres"]] == centres, (
            "every leg must see the arms already planned, or a staged return can "
            "cross a path that was already accepted"
        )


def test_a_leg_relaxes_the_margin_the_same_way_a_direct_path_does(
        ompl, stub_legs, stub_strategies):
    """Graded margins are a property of the planner, not of the direct path."""
    stub_strategies(stub_strategies.one([[0.5] * 6]))
    calls = stub_legs(unplannable=[])
    ompl._plan_staged_home(CELL, NS, START, HOME, [((0.0, 0.0, 0.0), 0.0)], RADII, 0.4)

    first_leg = [c for c in calls if c["goal"] == [0.5] * 6]
    assert first_leg[0]["spheres"][0][1] == max(RADII), (
        "the widest margin must be attempted first"
    )


def test_the_strategy_layer_is_told_the_margins_that_already_failed(
        ompl, stub_legs, monkeypatch):
    """Otherwise the model proposes routes against a clearance it cannot know."""
    mod = pytest.importorskip("agents.trajectory.tools.home_strategy")
    seen: dict = {}

    def _spy(cell_id, robot_ns, start_q, home_q, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(mod, "propose_home_strategies", _spy)
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_robot_config",
        lambda *a, **k: {"joint_limits_lower": [-2.0] * 6,
                         "joint_limits_upper": [2.0] * 6})
    stub_legs()

    ompl._plan_staged_home(CELL, NS, START, HOME, [], RADII, 0.4,
                           others={"robot2": [0.1] * 6}, sphere_radius=0.12)

    assert seen["attempted_margins_m"] == pytest.approx([r - 0.12 for r in RADII])
    assert seen["others"] == {"robot2": [0.1] * 6}
    assert seen["lower"] == [-2.0] * 6, "the ARM's declared limits, not a guess"
    assert seen["upper"] == [2.0] * 6


def test_an_unreadable_robot_config_does_not_block_staging(
        ompl, stub_legs, stub_strategies, monkeypatch):
    """Limits are a filter on proposals, not a precondition for having any.

    Without them a proposal is simply not limit-checked here — OMPL still refuses
    to plan a leg it cannot reach, so the floor holds.
    """
    def _boom(*a, **k):
        raise FileNotFoundError("robots.yaml")

    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_robot_config", _boom)
    stub_strategies(stub_strategies.one([[0.5] * 6]))
    stub_legs()

    traj, _ = ompl._plan_staged_home(CELL, NS, START, HOME, [], RADII, 0.4)

    assert traj[-1]["positions"] == pytest.approx(HOME)


# ── the stitched trajectory is executable ────────────────────────────────────

def test_time_is_monotonic_across_the_join(ompl):
    """Later legs are shifted by the running clock; the executor reads one path."""
    legs = [
        [{"positions": [0.0] * 6, "time_s": 0.0}, {"positions": [0.5] * 6, "time_s": 2.0}],
        [{"positions": [0.5] * 6, "time_s": 0.0}, {"positions": [1.0] * 6, "time_s": 3.0}],
    ]

    out = ompl._concat_timed(legs)

    times = [p["time_s"] for p in out]
    assert times == sorted(times)
    assert times[-1] == pytest.approx(5.0)


def test_the_junction_sample_is_not_duplicated(ompl):
    """Two samples at one instant read as a zero-length step to the schedule check."""
    legs = [
        [{"positions": [0.0] * 6, "time_s": 0.0}, {"positions": [0.5] * 6, "time_s": 2.0}],
        [{"positions": [0.5] * 6, "time_s": 0.0}, {"positions": [1.0] * 6, "time_s": 3.0}],
    ]

    out = ompl._concat_timed(legs)

    assert len(out) == 3
    assert [p["time_s"] for p in out] == pytest.approx([0.0, 2.0, 5.0])


def test_a_single_leg_is_returned_unchanged(ompl):
    leg = [{"positions": [0.0] * 6, "time_s": 0.0}, {"positions": [0.5] * 6, "time_s": 1.5}]

    assert ompl._concat_timed([leg]) == leg


# ── staging is the failure path, and only the failure path ───────────────────

def test_staging_is_not_attempted_when_a_direct_path_exists(
        ompl, stub_legs, monkeypatch):
    """Cost, latency and nondeterminism stay off the happy path."""
    mod = pytest.importorskip("agents.trajectory.tools.home_strategy")
    asked = []
    monkeypatch.setattr(mod, "propose_home_strategies",
                        lambda *a, **k: asked.append(1) or iter(()))
    monkeypatch.setattr(ompl, "_import_ompl", lambda: object())
    stub_legs(plan_all=True)

    out = ompl.plan_home_trajectories(CELL, {NS: START})

    assert out[NS], "a direct path was available"
    assert asked == [], "no strategy may be proposed while OMPL is succeeding"


def test_an_arm_with_no_direct_path_is_rescued_by_staging(
        ompl, stub_legs, stub_strategies, monkeypatch, capsys):
    """The behaviour the whole layer exists for."""
    stub_strategies(stub_strategies.one([[0.5] * 6], source="llm",
                                        rationale="fold before swinging"))
    monkeypatch.setattr(ompl, "_import_ompl", lambda: object())
    home = ompl._home_config(CELL, NS, ompl.HOME_JOINTS_DEFAULT)
    # The DIRECT leg fails; the legs through the staging pose plan.
    stub_legs(unplannable_legs=[(START, home)])

    out = ompl.plan_home_trajectories(CELL, {NS: START})

    assert NS in out, "the arm must not be left to the blind home"
    assert out[NS][-1]["positions"] == pytest.approx(home)
    assert all(p.get("phase") == "transit" for p in out[NS])
    printed = capsys.readouterr().out
    assert "staged via" in printed and "fold before swinging" in printed, (
        "a staged return is not the plan that was asked for — say so"
    )


def test_an_arm_that_no_strategy_rescues_is_simply_absent(
        ompl, stub_legs, stub_strategies, monkeypatch):
    """Absent means the caller falls back to the blind home and reports it."""
    stub_strategies(stub_strategies.one([[0.5] * 6]))
    monkeypatch.setattr(ompl, "_import_ompl", lambda: object())

    def _never(*a, **k):
        return None

    monkeypatch.setattr(ompl, "plan_joint_path", _never)

    assert ompl.plan_home_trajectories(CELL, {NS: START}) == {}


# ── the end-of-program home is the same kind of motion ───────────────────────

@pytest.fixture
def langchain_present(monkeypatch):
    """Stand in for LangChain when it is absent, e.g. a run outside the container.

    ``pinocchio.py`` publishes its FK/IK as ``@tool`` functions, so importing the
    kinematics drags LangChain in behind them. The substitution is scoped: module
    stubs like these are order-sensitive — a leaked one breaks any suite that
    installs its own — so it is reverted, and so is the module it let through.
    """
    try:
        import langchain_core.tools  # noqa: F401
        yield
        return
    except Exception:
        pass

    import functools
    import types

    def _tool(fn=None, **_kw):
        def _wrap(f):
            @functools.wraps(f)
            def _inner(*a, **k):
                return f(*a, **k)
            _inner.invoke = lambda d: f(**d)
            return _inner
        return _wrap(fn) if fn is not None else _wrap

    class _Msg:
        def __init__(self, content):
            self.content = content

    lc = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.tool = _tool
    lc_msgs = types.ModuleType("langchain_core.messages")
    lc_msgs.HumanMessage = lc_msgs.SystemMessage = _Msg
    lc.tools, lc.messages = lc_tools, lc_msgs
    monkeypatch.setitem(sys.modules, "langchain_core", lc)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", lc_tools)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", lc_msgs)

    yield
    # Whatever imported under the stub must not be cached for anyone else.
    sys.modules.pop("agents.trajectory.tools.pinocchio", None)


@pytest.fixture
def single_arm_home(ompl, langchain_present, monkeypatch):
    """Drive ``plan_home_trajectory`` with the model and OMPL segment stubbed.

    That function is the END-of-program return — a different code path from the
    coordinated reset, and it fell back blind in exactly the same way. Only the
    staging wiring is under test here, so the kinematics are stood in for.
    """
    class _Model:
        nq = 6

        def createData(self):
            return object()

    def install(segment_succeeds: bool, staged=None):
        seen: dict = {}
        monkeypatch.setattr(ompl, "_import_ompl", lambda: object())
        monkeypatch.setattr(
            "agents.trajectory.tools.robot_kinematics._load_pinocchio_model",
            lambda *a, **k: (_Model(), None))
        # Stubbed rather than read from the cell: this test is about the staging
        # wiring, and a unit test that depends on real YAML and the pose cache is
        # a unit test that fails depending on what ran before it.
        monkeypatch.setattr(
            "agents.trajectory.tools.robot_kinematics._load_robot_config",
            lambda *a, **k: {"base_position": [0.0, 0.0, 0.0],
                             "base_orientation_yaw": 0.0,
                             "joint_limits_lower": [-3.14] * 6,
                             "joint_limits_upper": [3.14] * 6,
                             "joint_velocity_limits": [1.0] * 6,
                             "joint_acceleration_limits": [2.0] * 6})
        monkeypatch.setattr(
            "agents.trajectory.tools.trajectory_cache.get_last_pose",
            lambda *a, **k: None)
        monkeypatch.setattr(
            "agents.trajectory.tools.trajectory_cache.clear_last_pose",
            lambda *a, **k: None)
        monkeypatch.setattr(
            ompl, "_plan_ompl_segment",
            lambda *a, **k: (([[0.0] * 6, HOME], None) if segment_succeeds
                             else ([], {"reason": "no path"})))

        def fake_staged(cell_id, robot_ns, start_q, home_q, centres, radii,
                        vel, others=None, sphere_radius=0.12):
            seen["called"] = True
            seen["start"] = [float(v) for v in start_q]
            seen["home"] = [float(v) for v in home_q]
            return staged, {"waypoints": [[0.5] * 6], "source": "llm",
                            "rationale": "stub"}

        monkeypatch.setattr(ompl, "_plan_staged_home", fake_staged)
        return seen

    return install


def test_the_end_of_program_home_is_staged_rather_than_left_blind(
        ompl, single_arm_home):
    """The last motion of a program was the other blind straight line."""
    staged = [{"positions": [0.5] * 6, "time_s": 0.0},
              {"positions": HOME, "time_s": 2.0}]
    seen = single_arm_home(segment_succeeds=False, staged=staged)

    traj = ompl.plan_home_trajectory(CELL, NS, START, HOME)

    assert seen.get("called"), "a failed segment must reach the staging layer"
    assert traj[-1]["positions"] == pytest.approx(HOME)
    assert all(p.get("phase") == "transit" for p in traj)


def test_the_end_of_program_home_still_returns_none_when_staging_fails(
        ompl, single_arm_home):
    """Fail closed: no route at all means the caller's blind fallback, as before."""
    single_arm_home(segment_succeeds=False, staged=None)

    assert ompl.plan_home_trajectory(CELL, NS, START, HOME) is None


def test_a_direct_end_of_program_home_does_not_consult_the_staging_layer(
        ompl, single_arm_home):
    seen = single_arm_home(segment_succeeds=True)

    assert ompl.plan_home_trajectory(CELL, NS, START, HOME)
    assert not seen.get("called")


def test_no_ompl_means_no_staging_either(ompl, stub_strategies, monkeypatch):
    """Without a planner nothing can be validated, so nothing may be proposed.

    A staging pose is only safe because OMPL checks the legs. With no OMPL there
    is no check, and an unchecked staged path would be strictly worse than the
    blind home it replaced.
    """
    asked = []
    mod = pytest.importorskip("agents.trajectory.tools.home_strategy")
    monkeypatch.setattr(mod, "propose_home_strategies",
                        lambda *a, **k: asked.append(1) or iter(()))
    monkeypatch.setattr(ompl, "_import_ompl", lambda: None)

    assert ompl.plan_home_trajectories(CELL, {NS: START}) == {}
    assert asked == []
