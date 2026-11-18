"""HOME is one pose, declared by the cell, agreed on by everything that moves.

Six modules used to carry their own literal copy of
``[0, -1.57, 2.2, 0, -0.8, 0]`` while ``welding_cell_2_small`` declares
``home_pose_deg: [0, -90, 90, 0, 0, 0]``. Those are different poses — 36 deg
apart at the elbow, 46 deg at the wrist — and the split ran along the worst
possible seam: the PLANNER targeted the declared pose, so every coordinated
return was validated collision-free against it, while the executor's blind
fallback and the robot agents drove to the literal one.

So the pose an arm was cleared into was not the pose it was commanded to. On the
happy path nobody noticed, because a planned trajectory carries its own
waypoints. It surfaced exactly where it hurts: the unplanned home, taken after a
plan failed or an e-stop fired, aiming a third of a right angle away from the
only configuration anything had checked.

These tests hold the invariant rather than the implementation: whoever asks,
however they ask, HOME for a given arm is one pose, and it is the cell's.

Host-only: reads YAML, fakes the runtime. No ROS, no Pinocchio, no planner.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CELL = "welding_cell_2_small"
# What cells/welding_cell_2_small/config/robots.yaml actually declares.
DECLARED_DEG = [0, -90, 90, 0, 0, 0]
# The literal the six modules used to carry.
LEGACY = [0.0, -1.57, 2.2, 0.0, -0.8, 0.0]


@pytest.fixture(autouse=True)
def cells_on_disk(monkeypatch):
    """Point the config reader at this repo's cells, as the containers do."""
    monkeypatch.setenv("CELL_BASE_PATH", str(PROJECT_ROOT / "cells"))


@pytest.fixture
def hp():
    return pytest.importorskip("agents.home_pose")


# ── what the cell declares is what resolves ──────────────────────────────────

@pytest.mark.parametrize("ns", ["robot1", "robot2"])
def test_the_declared_home_is_what_resolves(hp, ns):
    """Degrees in the config, radians out — and not the legacy literal."""
    import math

    q = hp.home_pose(CELL, ns)

    assert q == pytest.approx([math.radians(v) for v in DECLARED_DEG])
    assert q != pytest.approx(LEGACY), (
        "resolving to the old literal means the cell config is being ignored"
    )


def test_a_cell_that_declares_nothing_falls_back_to_one_shared_constant(hp):
    """An undeclared HOME must still be ONE pose system-wide, not six."""
    assert hp.home_pose("no_such_cell_anywhere", "robot1") == pytest.approx(
        list(hp.HOME_JOINTS_FALLBACK))


def test_an_arm_absent_from_the_cell_falls_back(hp):
    assert hp.home_pose(CELL, "robot_that_does_not_exist") == pytest.approx(
        list(hp.HOME_JOINTS_FALLBACK))


def test_a_caller_may_supply_its_own_fallback(hp):
    """So a module with a stricter default does not silently lose it."""
    assert hp.home_pose("no_such_cell_anywhere", "robot1",
                        [0.1] * 6) == pytest.approx([0.1] * 6)


@pytest.mark.parametrize("key", ["home_pose", "home_joints"])
def test_a_radian_declaration_is_accepted_as_written(hp, monkeypatch, key):
    """Not every cell has to speak degrees."""
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_robot_config",
        lambda *a, **k: {key: [0.25] * 6})

    assert hp.home_pose(CELL, "robot1") == pytest.approx([0.25] * 6)


def test_degrees_win_when_a_cell_declares_both(hp, monkeypatch):
    """One ordering has to be defined; degrees are how the cells are written."""
    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_robot_config",
        lambda *a, **k: {"home_pose_deg": [0, 0, 0, 0, 0, 0],
                         "home_pose": [0.25] * 6})

    assert hp.home_pose(CELL, "robot1") == pytest.approx([0.0] * 6)


def test_an_unreadable_config_yields_the_fallback_rather_than_raising(hp, monkeypatch):
    """HOME is needed most on the failure path.

    A resolver that throws when the config is unreadable would take out the
    e-stop recovery it exists to serve.
    """
    def _boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(
        "agents.trajectory.tools.robot_kinematics._load_robot_config", _boom)

    assert hp.home_pose(CELL, "robot1") == pytest.approx(
        list(hp.HOME_JOINTS_FALLBACK))


# ── everyone who needs HOME agrees ───────────────────────────────────────────

def test_the_planner_resolves_the_same_home_as_the_shared_module(hp):
    """``_home_config`` is the planner's name for this; it must not be a copy."""
    op = pytest.importorskip("agents.trajectory.tools.ompl_planner")

    for ns in ("robot1", "robot2"):
        assert op._home_config(CELL, ns, op.HOME_JOINTS_DEFAULT) == pytest.approx(
            hp.home_pose(CELL, ns))


def test_the_planners_default_is_the_shared_fallback_object(hp):
    """Same constant, not an equal copy — equal copies drift, aliases cannot."""
    op = pytest.importorskip("agents.trajectory.tools.ompl_planner")

    assert op.HOME_JOINTS_DEFAULT is hp.HOME_JOINTS_FALLBACK


def test_no_module_carries_its_own_copy_of_the_home_pose():
    """The regression fence.

    The bug was not a wrong number, it was NINE copies of one. Another paste into
    a new module would reintroduce it silently and the tests above would all
    still pass, so the source itself is checked.

    Two files are allowed to name the value, and only as a declared-nothing
    fallback:

      * ``agents/home_pose.py`` — the single source.
      * ``docker/launch_viz.py`` — runs in the RViz image, which has neither the
        agents package nor numpy; it reads ``home_pose_deg`` out of robots.yaml
        itself and falls back to the literal only for a cell that declares none.

    Anything else must import it.
    """
    allowed = {Path("agents/home_pose.py"), Path("docker/launch_viz.py")}
    roots = ("agents", "controller", "docker", "scripts", "utils", "dashboard")
    pattern = re.compile(r"-1\.57\s*,\s*2\.2\s*,\s*0(?:\.0)?\s*,\s*-0\.8")

    offenders = []
    searched = 0
    for root in roots:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            if rel in allowed:
                continue
            searched += 1
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i}")

    assert searched > 20, "the fence scanned almost nothing — check the roots"
    assert not offenders, (
        "HOME is hardcoded again in: " + ", ".join(offenders) +
        " — import it from agents.home_pose instead"
    )


# ── the blind fallback, which is where the divergence actually bit ───────────

class _Runtime:
    """The executor's duck-typed runtime, recording what it was told to do."""

    def __init__(self):
        self.dispatches: list = []
        self.estops: list = []

    def now(self):
        import time
        return time.monotonic()

    def robots(self):
        return [("arms", "robot1")]

    def dispatch(self, robot_id, task):
        self.dispatches.append((robot_id, task))

    def latest_status(self, robot_id):
        return None

    def estop(self, reason):
        self.estops.append(reason)

    def clear_estop(self, reason="cleared"):
        pass

    def shutdown(self):
        pass


def _run_bare_home(tree_extra=None, **cfg_over):
    """Execute a HOME node with NO planned trajectory — the blind path."""
    from agents.cell.bt_executor import BehaviourTreeExecutor, ExecConfig

    cfg = ExecConfig(tick_hz=200.0, home_return_time_s=0.01,
                     completion_margin_s=0.01, position_gate_timeout_s=0.1,
                     **cfg_over)
    rt = _Runtime()
    tree = {"root": {"id": "h1", "type": "HOME", "robot_ns": "robot1"},
            "metadata": {}}
    tree.update(tree_extra or {})
    BehaviourTreeExecutor(rt, None, cfg).execute(tree)
    return rt


def test_the_blind_home_aims_at_the_pose_the_cell_declares(hp):
    """The motion nothing plans must still target the pose everything validates.

    This is the divergence that mattered: a bare HOME node drives one unchecked
    waypoint, and it used to be a waypoint no planner had ever agreed to.
    """
    rt = _run_bare_home({"cell_id": CELL})

    assert len(rt.dispatches) == 1
    _rid, task = rt.dispatches[0]
    commanded = task["trajectory"][-1]["positions"]

    assert commanded == pytest.approx(hp.home_pose(CELL, "robot1"))
    assert commanded != pytest.approx(LEGACY)


def test_the_blind_home_uses_the_cell_the_tree_was_planned_for(monkeypatch, hp):
    """The tree names its cell; that beats whatever the container is set to.

    A replay validated against one cell must not be homed against another just
    because the environment disagrees.
    """
    monkeypatch.setenv("CELL_ID", "welding_cell_2_small")

    rt = _run_bare_home({"cell_id": CELL})

    _rid, task = rt.dispatches[0]
    assert task["trajectory"][-1]["positions"] == pytest.approx(
        hp.home_pose(CELL, "robot1"))


def test_a_tree_with_no_cell_still_homes_somewhere_defined(hp):
    """Fail soft, not silent: an unidentifiable cell yields the shared fallback."""
    rt = _run_bare_home()

    _rid, task = rt.dispatches[0]
    assert task["trajectory"][-1]["positions"] == pytest.approx(
        list(hp.HOME_JOINTS_FALLBACK))


def test_a_planned_home_is_still_preferred_over_the_blind_one(hp):
    """Nothing here may promote the fallback over a checked path."""
    from agents.cell.bt_executor import BehaviourTreeExecutor, ExecConfig

    planned = [{"positions": [0.3] * 6, "time_s": 0.0},
               {"positions": hp.home_pose(CELL, "robot1"), "time_s": 0.01}]
    rt = _Runtime()
    BehaviourTreeExecutor(rt, None, ExecConfig(
        tick_hz=200.0, completion_margin_s=0.01)).execute({
            "cell_id": CELL,
            "root": {"id": "h1", "type": "HOME", "robot_ns": "robot1",
                     "precomputed_trajectory": planned},
            "metadata": {}})

    _rid, task = rt.dispatches[0]
    assert task["trajectory"] == planned, "the planned path must win"
