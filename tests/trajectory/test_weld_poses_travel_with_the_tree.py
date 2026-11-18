"""A plan carries its weld poses, so the cell's home reset can lift off on-axis.

The e-stop home reset plans in the CELL process, where nothing was planned: its
trajectory cache held no weld poses, ``seam_at_start`` could not tell an arm was
stopped inside a seam, and the return left the seam with a joint-space transit
— the sideways drag the on-axis lift-off exists to prevent. The assembler now
stamps every weld's pose into ``metadata.weld_poses`` and the cell keeps them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weld_step_gate import CELL, bt, gate  # noqa: E402,F401 (fixtures)



class _TC:
    """``trajectory_cache`` as currently imported: the ``bt`` fixture purges and
    re-imports ``agents.trajectory.tools``, so a module bound at import is stale."""

    def __getattr__(self, name):
        import importlib
        return getattr(importlib.import_module("agents.trajectory.tools.trajectory_cache"),
                       name)


tc = _TC()


@pytest.fixture
def fresh():
    tc.reset_robot_configs()
    yield
    tc.reset_robot_configs()


def _pose(z=1.0):
    return dict(target=[0.5, 0.1, z], approach_vector=[0, 0, -1],
                orientation={"x": 0, "y": 1, "z": 0, "w": 0}, stroke_m=0.08)


def test_the_record_round_trips_into_a_process_that_planned_nothing(fresh):
    p = _pose()
    tc.set_weld_pose("c", "robot1", "W1", p["target"], p["approach_vector"],
                     p["orientation"], stroke_m=p["stroke_m"])
    tc.set_weld_pose("c", "robot2", "W2", p["target"], p["approach_vector"], None)
    record = tc.weld_pose_record("c")
    assert set(record) == {"robot1", "robot2"}
    assert tc.weld_pose_record("c", ["robot1"]).keys() == {"robot1"}

    import json
    record = json.loads(json.dumps(record))        # what a saved plan holds
    tc.reset_robot_configs()
    assert tc.weld_poses("c", "robot1") == []
    assert tc.load_weld_pose_record("c", record) == 2
    (op, pose), = tc.weld_poses("c", "robot1")
    assert op == "W1" and pose["target"] == p["target"] and pose["stroke_m"] == 0.08


@pytest.mark.parametrize("junk", [None, [], "x", {"robot1": "x"},
                                  {"robot1": {"W1": {"target": None}}},
                                  {"robot1": {"W1": {"target": [1, 2]}}}])
def test_a_malformed_record_registers_nothing(fresh, junk):
    assert tc.load_weld_pose_record("c", junk) == 0
    assert tc.weld_poses("c", "robot1") == []


def test_the_assembled_tree_carries_the_weld_poses(bt, gate, fresh):
    gate(lambda moving, parked: True)
    home = list(bt._arm_home(CELL, "robot1"))
    tc.cache_trajectory(CELL, "robot1", "A1", [   # its first leg starts at HOME
        {"positions": home, "time_s": 0.0},
        {"positions": [v + 0.1 for v in home], "time_s": 1.0}])
    p = _pose()
    tc.set_weld_pose(CELL, "robot1", "A1", p["target"], p["approach_vector"],
                     p["orientation"], stroke_m=p["stroke_m"])
    out = bt.assemble_behaviour_tree.func(CELL, {
        "steps": [{"mode": "PARALLEL", "robots": [
            {"robot_ns": "robot1", "operation_id": "A1", "operation_type": "weld"}]}],
        "home_at_end": False})
    assert out["success"] is True, out
    poses = out["behaviour_tree"]["metadata"]["weld_poses"]
    assert poses["robot1"]["A1"]["target"] == p["target"]


def test_the_cell_keeps_the_poses_of_a_tree_it_executes(fresh):
    agent = pytest.importorskip("agents.cell.agent")
    p = _pose()
    agent._remember_weld_poses("c", {"tree_id": "bt_1", "metadata": {
        "weld_poses": {"robot1": {"W1": p}}}})
    assert [op for op, _ in tc.weld_poses("c", "robot1")] == ["W1"]
    agent._remember_weld_poses("c", {"tree_id": "home_reset_c", "root": {}})
    assert [op for op, _ in tc.weld_poses("c", "robot1")] == ["W1"]


def test_a_loaded_pose_lets_the_reset_see_the_seam(fresh, monkeypatch):
    """The point of it: an arm stopped in a seam is recognised as in it."""
    pytest.importorskip("pinocchio")
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("CELL_BASE_PATH", str(Path(__file__).resolve().parents[2] / "cells"))
    rk = pytest.importorskip("agents.trajectory.tools.robot_kinematics")
    cell = "welding_cell_2_small"
    try:
        model, _ = rk._load_pinocchio_model(cell, "robot1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{cell} model unavailable: {exc}")
    q = [0.1, -0.6, 0.9, 0.0, 0.8, 0.0]
    base, yaw = [0.0, 0.0, 0.0], 0.0
    tcp = rk._fk_ee(model, model.createData(), np.asarray(q), np.asarray(base), yaw)
    weld = [float(v) for v in tcp - np.array([0.0, 0.0, 0.03])]   # 3 cm up its stroke
    record = {"robot1": {"W7": dict(target=weld, approach_vector=[0, 0, -1],
                                    orientation={"x": 0, "y": 1, "z": 0, "w": 0},
                                    stroke_m=0.1)}}
    assert rk.seam_at_start(cell, "robot1", model, q, base, yaw, pinned=True) is None
    tc.load_weld_pose_record(cell, record)
    seam = rk.seam_at_start(cell, "robot1", model, q, base, yaw, pinned=True)
    assert seam and seam["operation_id"] == "W7"
    assert seam["rise_m"] == pytest.approx(0.03, abs=1e-3)
