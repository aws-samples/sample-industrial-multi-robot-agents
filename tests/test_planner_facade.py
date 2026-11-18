"""Host unit tests for the planner-backend facade.

These run on the host with NO ROS, NO Pinocchio and NO OMPL: the facade imports
its backends lazily, which is precisely the property that makes it testable here.
They verify the guarantees the cell depends on — that the task's backend choice is
authoritative, that MoveIt is gone, that a legacy backend is stamped
non-conforming, and that an unavailable backend degrades to a structured failure
instead of raising.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools import planner_backend as pb  # noqa: E402


# ─── Backend registry ─────────────────────────────────────────────────────────

def test_moveit_is_not_a_valid_backend():
    """The MoveIt backend was removed; selecting it must fail loudly, not silently."""
    assert "moveit" not in pb.VALID_BACKENDS
    with pytest.raises(ValueError):
        pb.normalize_backend("moveit")


def test_every_valid_backend_has_an_implementation():
    """No backend may be selectable without a registered loader."""
    assert set(pb.VALID_BACKENDS) == set(pb._BACKENDS)


def test_default_backend_is_ompl_and_conforming():
    assert pb.DEFAULT_BACKEND == "ompl"
    assert "ompl" not in pb.NON_CONFORMING_BACKENDS


def test_legacy_rrt_is_marked_non_conforming():
    """RRT lacks inter-robot coupling / orientation / two-phase; say so in data."""
    assert "rrt" in pb.NON_CONFORMING_BACKENDS


# ─── Resolution priority ──────────────────────────────────────────────────────

def test_context_binding_beats_explicit_argument():
    """The task's binding is authoritative — an LLM-supplied argument cannot win."""
    token = pb.set_planning_backend("rrt")
    try:
        assert pb.resolve_planning_backend("ompl") == "rrt"
    finally:
        pb.reset_planning_backend(token)


def test_explicit_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("TRAJECTORY_BACKEND", "rrt")
    assert pb.resolve_planning_backend("ompl") == "ompl"


def test_environment_used_when_nothing_else_selects(monkeypatch):
    monkeypatch.setenv("TRAJECTORY_BACKEND", "rrt")
    assert pb.resolve_planning_backend() == "rrt"


def test_invalid_environment_value_raises(monkeypatch):
    monkeypatch.setenv("TRAJECTORY_BACKEND", "moveit")
    with pytest.raises(ValueError):
        pb.resolve_planning_backend()


# ─── Dispatch ─────────────────────────────────────────────────────────────────

def test_dispatch_routes_to_the_bound_backend(monkeypatch):
    calls = {}

    def fake_entry(cell_id, motion_requests, **kw):
        calls["cell_id"] = cell_id
        calls["kw"] = kw
        return {"success": True, "trajectories": {"r1": [{"positions": [0], "time_s": 0.0}]},
                "failures": [], "planning_time_s": 0.1}

    monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: fake_entry)
    token = pb.set_planning_backend("ompl")
    try:
        out = pb.dispatch_plan("cellA", [{"robot_ns": "r1"}],
                               approach_offset_m=0.07, velocity_scale=0.3)
    finally:
        pb.reset_planning_backend(token)

    assert calls["cell_id"] == "cellA"
    assert calls["kw"]["approach_offset_m"] == 0.07
    assert calls["kw"]["velocity_scale"] == 0.3
    assert out["backend"] == "ompl"
    assert out["backend_used"] == "ompl"
    assert "non_conforming_backend" not in out


def test_dispatch_stamps_non_conforming_for_legacy_rrt(monkeypatch):
    monkeypatch.setitem(
        pb._BACKENDS, "rrt",
        lambda: (lambda cell_id, reqs, **kw: {
            "success": True, "trajectories": {}, "failures": [], "planning_time_s": 0.0}),
    )
    token = pb.set_planning_backend("rrt")
    try:
        out = pb.dispatch_plan("cellA", [])
    finally:
        pb.reset_planning_backend(token)
    assert out["non_conforming_backend"] is True


def test_dispatch_reports_unavailable_backend_as_data(monkeypatch):
    """An uninstalled backend must degrade to a failure dict, never raise."""
    def boom():
        raise ImportError("no ompl wheel for this platform")

    monkeypatch.setitem(pb._BACKENDS, "ompl", boom)
    token = pb.set_planning_backend("ompl")
    try:
        out = pb.dispatch_plan("cellA", [])
    finally:
        pb.reset_planning_backend(token)

    assert out["success"] is False
    assert out["trajectories"] == {}
    assert "unavailable" in out["failures"][0]["error"]


def test_dispatch_rejects_a_non_dict_backend_result(monkeypatch):
    monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: (lambda c, r, **kw: None))
    token = pb.set_planning_backend("ompl")
    try:
        out = pb.dispatch_plan("cellA", [])
    finally:
        pb.reset_planning_backend(token)
    assert out["success"] is False
    assert "not a dict" in out["failures"][0]["error"]


def test_no_fallback_between_backends(monkeypatch):
    """A failing backend's verdict is returned as-is — never rerouted."""
    monkeypatch.setitem(
        pb._BACKENDS, "ompl",
        lambda: (lambda c, r, **kw: {
            "success": False, "trajectories": {}, "planning_time_s": 0.2,
            "failures": [{"robot_ns": "r1", "error": "goal in collision",
                          "reason_code": "goal_in_collision"}]}),
    )
    other_called = []
    monkeypatch.setitem(pb._BACKENDS, "rrt",
                        lambda: (lambda c, r, **kw: other_called.append(1) or {}))
    token = pb.set_planning_backend("ompl")
    try:
        out = pb.dispatch_plan("cellA", [{"robot_ns": "r1"}])
    finally:
        pb.reset_planning_backend(token)

    assert out["success"] is False
    assert out["failures"][0]["reason_code"] == "goal_in_collision"
    assert not other_called, "dispatch must not fall back to another backend"
