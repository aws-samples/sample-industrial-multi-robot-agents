"""Descending INTO a node: phase segments, node lookup, and the ghost's pose.

The tree view drew the plan's structure and stopped there — a leaf was a box with
some ports, and the motion inside it (the fast cross-cell transit, then the slow
tool-oriented run-in) was invisible even though every waypoint carries a ``phase``
label. Nothing consumed that label: the executor reads ``positions`` and ``time_s``
and nothing else. So the level below the leaf existed in the data and had no reader.

These are the three pure functions that turn it into something an operator can
step through:

  * ``_phase_segments`` (orchestrator) — waypoints → consecutive same-phase runs.
  * ``_find_bt_node`` (orchestrator) — one node out of a tree, by id.
  * ``interpolate_joints`` (viewer) — a pose at time t, to animate the ghost.

Plus the RViz config that gives every arm its translucent second copy.

Host-only. The orchestrator module needs flask, which the host does not ship, so
the two functions are lifted out of the source with ``ast`` and executed on their
own — both are self-contained by construction. ``docker/launch_viz.py`` imports
only yaml and ``utils``, so it is imported normally.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ORCH_SERVER = PROJECT_ROOT / "agents" / "orchestrator" / "server.py"
LAUNCHER = PROJECT_ROOT / "docker" / "launch_viz.py"


def _lift(path: Path, *names: str) -> dict:
    """Execute named top-level functions from ``path`` in a namespace of their own.

    Importing the orchestrator would pull in flask, httpx and a live MCP client for
    the sake of two functions that touch neither. Lifting them keeps this test on
    the host baseline — and fails loudly if either grows a dependency on module
    state, because the namespace it runs in holds nothing but itself.
    """
    tree = ast.parse(path.read_text())
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    if missing:
        pytest.fail(f"{path.name} no longer defines {sorted(missing)} at module level")
    module = ast.Module(body=wanted, type_ignores=[])
    ns: dict = {}
    exec(compile(module, str(path), "exec"), ns)  # noqa: S102
    return ns


_ORCH = _lift(ORCH_SERVER, "_phase_segments", "_find_bt_node")
phase_segments = _ORCH["_phase_segments"]
find_bt_node = _ORCH["_find_bt_node"]


def _launcher():
    spec = importlib.util.spec_from_file_location("launch_viz_under_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LV = _launcher()


def _wp(t: float, phase: str | None = None, positions=None) -> dict:
    w = {"positions": list(positions or [0.0] * 6), "time_s": t}
    if phase is not None:
        w["phase"] = phase
    return w


class TestAMotionSplitsIntoItsPhases:
    def test_consecutive_samples_of_one_phase_become_one_segment(self):
        segs = phase_segments([_wp(0.0, "transit"), _wp(0.5, "transit"),
                               _wp(1.0, "approach"), _wp(1.4, "approach")])
        assert [s["phase"] for s in segs] == ["transit", "approach"]
        assert [s["points"] for s in segs] == [2, 2]
        assert (segs[0]["start_s"], segs[0]["end_s"]) == (0.0, 0.5)
        assert (segs[1]["start_s"], segs[1]["end_s"]) == (1.0, 1.4)

    def test_a_phase_that_comes_back_is_a_second_segment(self):
        """Runs are consecutive, not grouped.

        A retreat is transit again, and merging it with the transit that led in
        would claim the arm crossed the cell once when it crossed twice.
        """
        segs = phase_segments([_wp(0.0, "transit"), _wp(1.0, "approach"),
                               _wp(2.0, "transit")])
        assert [s["phase"] for s in segs] == ["transit", "approach", "transit"]
        assert all(s["points"] == 1 for s in segs)

    def test_the_segments_tile_the_trajectory_exactly(self):
        """The invariant the panel slices on: ``waypoints[start:end + 1]``.

        Off-by-one here means a preview silently drops the first or last sample of
        a phase — which is the approach's contact pose, the one worth watching.
        """
        traj = ([_wp(i * 0.1, "transit") for i in range(7)]
                + [_wp(0.7 + i * 0.1, "approach") for i in range(4)])
        segs = phase_segments(traj)
        assert sum(s["points"] for s in segs) == len(traj)
        rebuilt: list = []
        for s in segs:
            rebuilt += traj[s["start_index"]:s["end_index"] + 1]
        assert rebuilt == traj

    def test_an_unlabelled_sample_counts_as_transit(self):
        """Only the planner's approach legs are labelled; plain moves carry no
        phase, and calling them 'unknown' would put a phantom class in the UI."""
        segs = phase_segments([_wp(0.0), _wp(1.0)])
        assert len(segs) == 1 and segs[0]["phase"] == "transit"

    def test_an_empty_trajectory_has_no_segments(self):
        """A control node reaches this with []; it must not be an error."""
        assert phase_segments([]) == []

    def test_junk_samples_are_skipped_not_fatal(self):
        segs = phase_segments([_wp(0.0, "transit"), None, "nope", _wp(1.0, "transit")])
        assert len(segs) == 1 and segs[0]["points"] == 2


class TestOneNodeOutOfATree:
    TREE = {
        "type": "Sequence", "id": "root", "children": [
            {"type": "Parallel", "id": "step1", "children": [
                {"type": "ExecuteTrajectory", "id": "leaf-a", "children": []},
                {"type": "ExecuteTrajectory", "id": "leaf-b", "children": []},
            ]},
            {"type": "MoveHome", "id": "home", "children": []},
        ],
    }

    def test_it_finds_a_leaf_two_levels_down(self):
        """The depth that matters: a real plan is Sequence → Parallel → leaf."""
        found = find_bt_node(self.TREE, "leaf-b")
        assert found is not None and found["type"] == "ExecuteTrajectory"

    def test_it_finds_the_root_itself(self):
        assert find_bt_node(self.TREE, "root") is self.TREE

    def test_a_miss_is_none_rather_than_an_error(self):
        """Ids are optional in the format and do not survive an XML round-trip,
        so 'no such node' is a normal answer that the route turns into a 404."""
        assert find_bt_node(self.TREE, "nope") is None

    def test_nodes_without_ids_do_not_stop_the_search(self):
        tree = {"type": "Sequence", "children": [
            {"type": "Sequence", "children": [{"type": "MoveHome", "id": "deep"}]}]}
        assert find_bt_node(tree, "deep") is not None

    def test_a_non_dict_is_none(self):
        assert find_bt_node(None, "x") is None
        assert find_bt_node("Sequence", "x") is None


class TestTheGhostPoseAtTimeT:
    WPS = [_wp(0.0, positions=[0.0, 0.0]), _wp(1.0, positions=[2.0, -2.0]),
           _wp(2.0, positions=[4.0, -4.0])]

    def test_it_lands_on_the_waypoints_themselves(self):
        assert LV.interpolate_joints(self.WPS, 0.0) == [0.0, 0.0]
        assert LV.interpolate_joints(self.WPS, 1.0) == [2.0, -2.0]
        assert LV.interpolate_joints(self.WPS, 2.0) == [4.0, -4.0]

    def test_it_interpolates_between_them(self):
        assert LV.interpolate_joints(self.WPS, 0.5) == pytest.approx([1.0, -1.0])
        assert LV.interpolate_joints(self.WPS, 1.75) == pytest.approx([3.5, -3.5])

    def test_it_clamps_outside_the_trajectory(self):
        """The timer can ask for t past the end (the last tick of a non-looping
        preview); a ghost that flew off past the final pose would be a lie."""
        assert LV.interpolate_joints(self.WPS, -5.0) == [0.0, 0.0]
        assert LV.interpolate_joints(self.WPS, 99.0) == [4.0, -4.0]

    def test_two_samples_at_the_same_time_do_not_divide_by_zero(self):
        """OMPL emits coincident times when a leg has no dwell."""
        wps = [_wp(0.0, positions=[1.0]), _wp(0.0, positions=[2.0]),
               _wp(1.0, positions=[3.0])]
        assert LV.interpolate_joints(wps, 0.0) == [1.0]
        assert LV.interpolate_joints(wps, 0.5) == pytest.approx([2.5])

    def test_no_waypoints_is_an_empty_pose_not_a_crash(self):
        assert LV.interpolate_joints([], 1.0) == []


class TestEveryArmHasAGhostToPreviewOn:
    """The ghost is what makes a preview possible without commanding anything.

    RViz can only draw what a topic tells it, and the repo publishes no
    ``DisplayTrajectory`` anywhere — so the alternative to a second robot model
    was moving the real arm. These pin the second model into the generated config.
    """

    ROBOTS = [{"robot_id": "robot1"}, {"robot_id": "robot2"}]

    def _displays(self, tmp_path) -> list[dict]:
        out = tmp_path / "rviz.rviz"
        LV.write_rviz_config(self.ROBOTS, str(out))
        cfg = yaml.safe_load(out.read_text())
        return cfg["Visualization Manager"]["Displays"]

    def test_one_ghost_robot_model_per_robot(self, tmp_path):
        displays = self._displays(tmp_path)
        models = [d for d in displays
                  if d["Class"].endswith("RobotModel")]
        ghosts = [d for d in models if "ghost" in d["Name"].lower()]
        assert len(ghosts) == len(self.ROBOTS), \
            f"expected a ghost per robot, got {[d['Name'] for d in models]}"

    def test_each_ghost_reads_its_own_description_topic(self, tmp_path):
        """A ghost pointed at the real ``robot_description`` would be a second
        copy of the arm in the real pose — visible, and useless for previews."""
        topics = {d["Description Topic"]["Value"]
                  for d in self._displays(tmp_path)
                  if d["Class"].endswith("RobotModel") and "ghost" in d["Name"].lower()}
        assert topics == {f"/viz/{LV.ghost_id(r['robot_id'])}/robot_description"
                          for r in self.ROBOTS}

    def test_the_ghost_is_translucent(self, tmp_path):
        """Opaque, it hides the arm it is meant to be compared against."""
        for d in self._displays(tmp_path):
            if d["Class"].endswith("RobotModel") and "ghost" in d["Name"].lower():
                assert 0.0 < d["Alpha"] < 1.0, \
                    f"ghost {d['Name']} has Alpha {d.get('Alpha')}"

    def test_the_ghost_name_cannot_collide_with_a_real_robot(self, tmp_path):
        """Both models publish into the one global /tf; identical link prefixes
        would fight over the same frames and the view would flicker."""
        real = {r["robot_id"] for r in self.ROBOTS}
        assert not real & {LV.ghost_id(rid) for rid in real}
        assert LV.GHOST_SUFFIX, "the suffix is what keeps the frames apart"
