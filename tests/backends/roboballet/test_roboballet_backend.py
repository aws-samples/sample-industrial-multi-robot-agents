"""The ``roboballet`` backend: the same interface as every other backend.

* It is selectable by name through the facade like any backend — bundled, not
  built-in, never the default — and declares that it plans a step jointly.
* No policy, or a policy the planner cannot run, refuses every request of the
  step with ``planner_error`` and names the operation. Nothing falls back.
* With a policy, it runs ompl-compound's step pipeline with the policy rollout
  as the transit search (``transit_solver``), signed ``roboballet``, and adds
  what the rollout did to the result.
* The rollout uses the platform's own predicate: its path starts at the
  transit start, ends at the exact goal, and every sample is free; a blocked
  arrival is ``no_path_found``; an invalid start is ``compound_start_invalid``.

Host-runnable (numpy); the compound pipeline's Pinocchio half is stubbed here and
exercised in the container like ompl-compound's.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.trajectory import backends
from agents.trajectory.backends.contract import RESULT_KEYS, missing_result_keys
from agents.trajectory.backends.roboballet import planner as rp
from agents.trajectory.backends.roboballet.core import graph as G
from agents.trajectory.backends.roboballet.core.chain import Chain
from agents.trajectory.backends.roboballet.core.geometry import BoxSet, pair_clearance
from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH, init_params
from agents.trajectory.backends.roboballet.core.policy_io import Policy, save
from agents.trajectory.backends.roboballet.rollout import make_transit_solver
from agents.trajectory.tools import planner_backend as pb
from tests.backends.roboballet.synthetic import pack

SMALL = {**DEFAULT_ARCH, "node": 8, "edge": 8, "global": 8, "depth": 2, "head": 8}
REQS = [{"robot_ns": "robot1", "operation_id": "W1"}, {"robot_ns": "robot2", "operation_id": "W2"}]


# ── selectable like any backend ──────────────────────────────────────────────

def test_it_is_bundled_selectable_and_never_the_default():
    assert "roboballet" not in backends.BUILTIN
    assert "roboballet" in backends.BUNDLED and "roboballet" in pb.VALID_BACKENDS
    spec = pb.backend_spec("roboballet")
    assert spec.joint_step and spec.conforming
    assert pb.backend_spec("ompl-compound").joint_step
    assert not pb.backend_spec("ompl").joint_step
    assert pb.DEFAULT_BACKEND == "ompl"


def test_a_plugin_cannot_take_the_bundled_name(monkeypatch):
    monkeypatch.setenv(backends.PLUGIN_ENV, "roboballet=json:dumps")
    with pytest.raises(ValueError, match="bundled"):
        backends.discover()


def test_a_replan_edit_replans_its_step_together():
    from agents.trajectory.tools.plan_edit import _plans_steps_jointly
    assert _plans_steps_jointly("roboballet") and _plans_steps_jointly("ompl-compound")
    assert not _plans_steps_jointly("ompl") and not _plans_steps_jointly("no-such")


# ── fail-closed on the policy ────────────────────────────────────────────────

def test_no_policy_refuses_every_request_and_names_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOBALLET_POLICY_DIR", str(tmp_path))
    out = rp.plan("c", REQS)
    assert not missing_result_keys(out) and out["success"] is False
    assert out["trajectories"] == {} and out["backend"] == "roboballet"
    assert {f["operation_id"] for f in out["failures"]} == {"W1", "W2"}
    assert {f["reason_code"] for f in out["failures"]} == {"planner_error"}
    assert "no policy 'current'" in out["failures"][0]["error"]


def test_a_policy_name_cannot_leave_the_policy_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOBALLET_POLICY_DIR", str(tmp_path))
    out = rp.plan("c", REQS, policy="../etc")
    assert "plain directory name" in out["failures"][0]["error"]


def test_the_step_pipeline_runs_with_the_policy_as_its_search(tmp_path, monkeypatch):
    save(tmp_path / "current", init_params(SMALL), {"arch": SMALL, "name": "current",
                                                    "training": {"job": "j1", "cells": ["a"]}})
    monkeypatch.setenv("ROBOBALLET_POLICY_DIR", str(tmp_path))
    monkeypatch.setattr(rp, "_cell_model", lambda cell, ids: ({}, {}, BoxSet([])))
    seen = {}

    def fake_compound(cell_id, motion_requests, **kw):
        seen.update(kw)
        return {"success": True, "trajectories": {}, "failures": [], "planning_time_s": 0.0,
                "backend": kw["backend_name"]}

    import agents.trajectory.backends.ompl_compound.planner as cp
    monkeypatch.setattr(cp, "plan_trajectories_ompl_compound", fake_compound)
    out = rp.plan("c", REQS, max_iterations=7)
    assert callable(seen["transit_solver"]) and seen["backend_name"] == "roboballet"
    assert seen["planner_name"] == "policy:current" and seen["max_iterations"] == 7
    assert out["policy"]["job"] == "j1" and out["policy"]["trained_on"] == ["a"]
    assert set(RESULT_KEYS) <= set(out)


def test_dispatch_through_the_facade_reaches_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOBALLET_POLICY_DIR", str(tmp_path))
    token = pb.set_planning_backend("roboballet")
    try:
        out = pb.dispatch_plan("c", [{"robot_ns": "robot1", "operation_id": "W1"}])
    finally:
        pb.reset_planning_backend(token)
    assert out["backend_used"] == "roboballet"
    assert out["failures"][0]["reason_code"] == "planner_error"


# ── the rollout as a transit search ──────────────────────────────────────────

class Steer(Policy):
    """A stand-in policy: drive each arm along its goal offset (graph slot 3)."""

    def __init__(self):
        self.card, self.name, self.arch, self.params = {"tol_m": 0.02}, "steer", {}, {}

    def act(self, g):
        d = g["robot"][:, 3 * G.MAX_DOF:4 * G.MAX_DOF] * np.pi
        return np.clip(d * 3.0, -1, 1)


def _arms(p, boxes, blocked=lambda i, q: False):
    chains = {r["robot_id"]: Chain(r["chain"]) for r in p["robots"]}
    arms = []
    for i, r in enumerate(p["robots"]):
        c = chains[r["robot_id"]]
        arms.append({"ns": r["robot_id"], "nq": 3, "vel": r["vel_limits"],
                     "is_free": (lambda q, c=c, i=i: not blocked(i, q)
                                 and not boxes.collides(c.points(q, moving_only=True), 0.08)),
                     "spheres": (lambda q, c=c: c.points(q))})
    return chains, arms


def _solve(p, starts, goals, blocked=lambda i, q: False):
    boxes = BoxSet(p["obstacles"])
    chains, arms = _arms(p, boxes, blocked)
    info = {}
    solver = make_transit_solver(Steer(), chains, boxes,
                                 accel_limits={r["robot_id"]: r["acc_limits"] for r in p["robots"]},
                                 info=info)
    return solver(arms, starts, goals, pair_min_m=0.02, time_budget_s=5.0, cell_id="syn"), arms, info


def test_the_rollout_starts_at_the_start_ends_at_the_goal_and_stays_free():
    p = pack(gap=2.4)
    starts = [np.array([0.0, -0.3, 0.6]), np.array([0.0, -0.3, 0.6])]
    goals = [np.array([0.5, -0.1, 0.9]), np.array([-0.4, -0.2, 0.8])]
    (path, fail), arms, info = _solve(p, starts, goals)
    assert fail is None, fail
    np.testing.assert_allclose(path[0], np.concatenate(starts))
    np.testing.assert_allclose(path[-1], np.concatenate(goals), atol=1e-9)
    for v in path:
        q1, q2 = np.asarray(v[:3]), np.asarray(v[3:])
        assert arms[0]["is_free"](q1) and arms[1]["is_free"](q2)
        assert pair_clearance(arms[0]["spheres"](q1), arms[1]["spheres"](q2), 0.12) > 0.02
    assert info["arrived"] == 2 and info["policy"] == "steer"


def test_an_arm_that_cannot_arrive_is_no_path_found():
    p = pack(gap=2.4)
    starts = [np.array([0.0, -0.3, 0.6])] * 2
    goals = [np.array([0.8, -0.1, 0.9]), np.array([0.0, -0.3, 0.6])]
    wall = lambda i, q: i == 0 and q[0] > 0.3                       # noqa: E731
    (path, fail), _, _ = _solve(p, starts, goals, blocked=wall)
    assert path is None and fail["reason_code"] in ("compound_goal_invalid", "no_path_found")
    (path, fail), _, _ = _solve(p, starts, [np.array([0.25, -0.1, 0.9]), goals[1]],
                                blocked=lambda i, q: i == 0 and 0.1 < q[0] < 0.2)
    assert path is None and fail["reason_code"] == "no_path_found"
    assert "robot1" in fail["reason"]


def test_an_invalid_start_is_refused_before_rolling_out():
    p = pack(gap=2.4)
    starts = [np.array([0.0, -0.3, 0.6])] * 2
    (path, fail), _, _ = _solve(p, starts, starts, blocked=lambda i, q: True)
    assert path is None and fail["reason_code"] == "compound_start_invalid"


def test_a_rollout_gets_its_own_time_budget_not_ompl_compounds(tmp_path, monkeypatch):
    save(tmp_path / "current", init_params(SMALL), {"arch": SMALL})
    monkeypatch.setenv("ROBOBALLET_POLICY_DIR", str(tmp_path))
    monkeypatch.delenv("ROBOBALLET_TIME_BUDGET_S", raising=False)
    monkeypatch.setattr(rp, "_cell_model", lambda cell, ids: ({}, {}, BoxSet([])))
    seen = {}
    import agents.trajectory.backends.ompl_compound.planner as cp
    monkeypatch.setattr(cp, "plan_trajectories_ompl_compound",
                        lambda c, r, **kw: seen.update(kw) or {"success": False,
                        "trajectories": {}, "failures": [], "planning_time_s": 0.0,
                        "backend": "roboballet"})
    rp.plan("c", REQS)
    assert seen["time_budget_s"] == rp.DEFAULT_TIME_BUDGET_S
    monkeypatch.setenv("ROBOBALLET_TIME_BUDGET_S", "7")
    rp.plan("c", REQS)
    assert seen["time_budget_s"] == 7.0
    rp.plan("c", REQS, time_budget_s=3)
    assert seen["time_budget_s"] == 3
