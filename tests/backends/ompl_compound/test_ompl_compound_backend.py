"""The ``ompl-compound`` backend: selectable, and its arms share one clock.

Host-runnable: these cover the facade registration and the pure parts of
``ompl_compound_planner`` (request grouping, the pair clearance, the joint
timeline). The OMPL solve itself needs the container — see the last class.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.trajectory.backends.ompl_compound import planner as cp
from agents.trajectory.tools import planner_backend as pb


class TestSelectable:
    def test_registered_with_a_loader(self):
        assert "ompl-compound" in pb.VALID_BACKENDS
        assert "ompl-compound" in pb._BACKENDS

    def test_the_name_normalises_and_conforms(self):
        assert pb.normalize_backend(" OMPL-Compound ") == "ompl-compound"
        assert "ompl-compound" not in pb.NON_CONFORMING_BACKENDS

    def test_the_task_binding_wins(self, monkeypatch):
        monkeypatch.setenv("TRAJECTORY_BACKEND", "ompl")
        tok = pb.set_planning_backend("ompl-compound")
        try:
            assert pb.resolve_planning_backend() == "ompl-compound"
        finally:
            pb.reset_planning_backend(tok)

    def test_dispatch_reaches_the_compound_entry_point(self, monkeypatch):
        seen = {}

        def fake(cell_id, motion_requests, **kw):
            seen["called"] = (cell_id, len(motion_requests))
            return {"success": True, "trajectories": {}, "failures": [],
                    "planning_time_s": 0.0, "backend": cp.BACKEND_NAME}

        monkeypatch.setitem(pb._BACKENDS, "ompl-compound", lambda: fake)
        tok = pb.set_planning_backend("ompl-compound")
        try:
            out = pb.dispatch_plan("c", [{"robot_ns": "robot1"}])
        finally:
            pb.reset_planning_backend(tok)
        assert seen["called"] == ("c", 1)
        assert out["backend_used"] == "ompl-compound"
        assert "non_conforming_backend" not in out


class TestRequests:
    def test_one_request_per_arm_is_kept(self):
        keep, dup = cp.split_requests([{"robot_ns": "robot1"}, {"robot_ns": "robot2"}])
        assert len(keep) == 2 and dup == []

    def test_every_request_of_a_duplicated_arm_is_rejected(self):
        reqs = [{"robot_ns": "robot1", "operation_id": "a"},
                {"robot_ns": "robot1", "operation_id": "b"},
                {"robot_ns": "robot2", "operation_id": "c"}]
        keep, dup = cp.split_requests(reqs)
        assert [r["operation_id"] for r in keep] == ["c"]
        assert sorted(r["operation_id"] for r in dup) == ["a", "b"]


class TestPairClearance:
    def test_matches_the_gates_brute_force_definition(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(17, 3)), rng.normal(size=(11, 3)) + 2.0
        brute = min(np.linalg.norm(p - q) for p in a for q in b) - 2 * cp.PAIR_RADIUS_M
        assert cp.pair_clearance(a, b) == pytest.approx(brute)

    def test_an_empty_set_is_infinitely_clear(self):
        assert cp.pair_clearance([], [[0, 0, 0]]) == float("inf")


class TestCandidates:
    S, G = [[0.0, 0.0], [10.0, 10.0]], [[2.0, 2.0], [8.0, 8.0]]

    def test_one_arm_leads_while_the_others_wait(self):
        paths = cp.staggered_paths(self.S, self.G)
        assert len(paths) == 2 * len(cp.STAGGER_LEADS)
        first = paths[cp.STAGGER_LEADS.index(1.0)]          # arm 0 goes first
        assert first[0] == [0, 0, 10, 10] and first[-1] == [2, 2, 8, 8]
        assert first[1] == [2, 2, 10, 10]                   # arm 1 still at start

    def test_a_half_lead(self):
        half = cp.staggered_paths(self.S, self.G)[cp.STAGGER_LEADS.index(0.5)]
        assert half[1] == [1, 1, 10, 10]

    def test_duration_is_bounded_by_the_slowest_joint(self):
        path = [[0, 0, 0, 0], [2, 0, 0, 1]]
        assert cp.estimated_duration(path, [1, 1, 1, 1]) == pytest.approx(2.0)
        assert cp.estimated_duration(path, [4, 1, 1, 0.5]) == pytest.approx(2.0)

    def test_together_is_faster_than_one_after_the_other(self):
        v = [1.0] * 4
        together = [cp.stack(self.S), cp.stack(self.G)]
        sequential = cp.staggered_paths(self.S, self.G)[cp.STAGGER_LEADS.index(1.0)]
        assert cp.estimated_duration(together, v) < cp.estimated_duration(sequential, v)


def _timed(points, dt=1.0):
    return [{"positions": list(p), "time_s": i * dt} for i, p in enumerate(points)]


class TestJointTimeline:
    SIZES = {"robot1": 2, "robot2": 2}

    def build(self):
        retracts = {"robot1": _timed([[0, 0], [1, 1]], dt=2.0),   # lifts off in 2 s
                    "robot2": []}                                 # nothing to lift
        transit = _timed([[1, 1, 5, 5], [2, 2, 6, 6], [3, 3, 7, 7]], dt=1.5)
        descents = {"robot1": _timed([[3, 3], [4, 4]], dt=0.5),
                    "robot2": _timed([[7, 7], [8, 8], [9, 9]], dt=0.5)}
        phases = {"robot1": "approach", "robot2": "approach"}
        return cp.compose_timelines(retracts, transit, self.SIZES, descents, phases)

    def test_the_transit_samples_share_their_timestamps(self):
        out = self.build()
        t1 = [p["time_s"] for p in out["robot1"] if p["positions"] in ([1, 1], [2, 2], [3, 3])]
        t2 = [p["time_s"] for p in out["robot2"] if p["positions"] in ([5, 5], [6, 6], [7, 7])]
        # The shared transit starts at t=2 (the slowest lift-off) on BOTH arms.
        assert t1[-2:] == t2[-2:] == [3.5, 5.0]

    def test_an_arm_without_a_lift_off_waits_at_its_start(self):
        r2 = self.build()["robot2"]
        assert r2[0] == {"positions": [5, 5], "time_s": 0.0, "phase": "transit"}
        assert r2[1]["positions"] == [5, 5] and r2[1]["time_s"] == 2.0

    def test_descents_start_together_when_the_transit_ends(self):
        out = self.build()
        first = {ns: next(p for p in t if p["phase"] == "approach") for ns, t in out.items()}
        assert first["robot1"]["time_s"] == first["robot2"]["time_s"] == 5.5

    def test_every_timeline_is_strictly_increasing(self):
        for traj in self.build().values():
            times = [p["time_s"] for p in traj]
            assert all(b > a for a, b in zip(times, times[1:])), times

    def test_tag_sync_marks_one_group(self):
        out = self.build()
        gid = cp.tag_sync(out)
        assert {t[0]["sync_group"] for t in out.values()} == {gid}
        assert out["robot1"][0]["sync_members"] == ["robot1", "robot2"]


class TestPlansInTheContainer:
    """A real joint solve: two arms swapping sides must not meet halfway."""

    def test_two_arms_plan_in_one_compound_space(self):
        ompl = pytest.importorskip("ompl")  # noqa: F841 — container only
        from agents.trajectory.backends.ompl.planner import _import_ompl

        # Two point "arms" on a line: arm A goes 0 -> 2, arm B goes 2 -> 0. The
        # straight joint motion crosses at 1; the pair test must forbid that.
        arms = []
        for _ in range(2):
            arms.append({"nq": 2, "jl": np.array([-3.0, -3.0]), "ju": np.array([3.0, 3.0]),
                         "is_free": lambda q: True,
                         "spheres": lambda q: [np.array([q[0], q[1], 0.0])]})
        path, fail = cp._solve_compound(
            _import_ompl(), arms, [[0.0, 0.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, 0.0]],
            planner_name="RRTConnect", time_budget_s=5.0, attempts=3, pair_min_m=0.1)
        assert fail is None and path
        for v in path:
            a, b = np.array(v[:2]), np.array(v[2:])
            assert np.linalg.norm(a - b) - 2 * cp.PAIR_RADIUS_M > 0.1 - 1e-6
