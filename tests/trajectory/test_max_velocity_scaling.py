"""``planning.max_velocity_scaling`` in robots.yaml is honoured, not ignored.

It is the transit speed a plan runs at when the caller names none, and the
ceiling on any speed a caller does name. A cell that declares nothing keeps the
historic 0.5 default with no cap.
"""
import pytest

from agents.trajectory.tools import robot_kinematics as rk


def _cell(monkeypatch, cfg):
    monkeypatch.setattr(rk, "_load_robots_config", lambda cell_id: cfg)


def test_declared_scaling_is_the_default_transit_speed(monkeypatch):
    _cell(monkeypatch, {"planning": {"max_velocity_scaling": 0.7}})
    assert rk.resolve_velocity_scale("c", None) == pytest.approx(0.7)


def test_request_above_the_ceiling_is_clamped(monkeypatch):
    _cell(monkeypatch, {"planning": {"max_velocity_scaling": 0.7}})
    assert rk.resolve_velocity_scale("c", 1.0) == pytest.approx(0.7)


def test_slower_request_is_honoured(monkeypatch):
    _cell(monkeypatch, {"planning": {"max_velocity_scaling": 0.7}})
    assert rk.resolve_velocity_scale("c", 0.3) == pytest.approx(0.3)


@pytest.mark.parametrize("bad", [0, -1, "fast", None])
def test_nonsense_request_falls_back_to_the_cell_speed(monkeypatch, bad):
    _cell(monkeypatch, {"planning": {"max_velocity_scaling": 0.7}})
    assert rk.resolve_velocity_scale("c", bad) == pytest.approx(0.7)


def test_undeclared_cell_keeps_todays_default_and_no_cap(monkeypatch):
    _cell(monkeypatch, {})
    assert rk.resolve_velocity_scale("c", None) == pytest.approx(0.5)
    assert rk.resolve_velocity_scale("c", 0.9) == pytest.approx(0.9)


def test_ceiling_above_one_is_capped_at_full_speed(monkeypatch):
    _cell(monkeypatch, {"planning": {"max_velocity_scaling": 3.0}})
    assert rk.resolve_velocity_scale("c", None) == pytest.approx(1.0)


def test_home_speed_comes_from_the_cell_and_is_capped(monkeypatch):
    _cell(monkeypatch, {"planning": {"max_velocity_scaling": 0.3,
                                     "home_velocity_scale": 0.6}})
    assert rk.resolve_home_velocity_scale("c", None) == pytest.approx(0.3)
    _cell(monkeypatch, {"planning": {"max_velocity_scaling": 0.7,
                                     "home_velocity_scale": 0.5}})
    assert rk.resolve_home_velocity_scale("c", None) == pytest.approx(0.5)
    assert rk.resolve_home_velocity_scale("c", 0.9) == pytest.approx(0.7)


def test_undeclared_speed_keys_keep_todays_values(monkeypatch):
    _cell(monkeypatch, {})
    cfg = rk._load_planning_config("c")
    for key, value in rk.PLANNING_SPEED_DEFAULTS.items():
        assert cfg[key] == pytest.approx(value), key


def test_declared_acceleration_limits_win_over_the_factor(monkeypatch):
    _cell(monkeypatch, {"planning": {"acceleration_factor": 10.0}})
    rcfg = {"joint_velocity_limits": [1.0, 2.0],
            "joint_acceleration_limits": [3.0, 4.0]}
    assert rk.joint_acceleration_limits("c", rcfg) == [3.0, 4.0]


def test_missing_acceleration_limits_use_the_cell_factor(monkeypatch):
    _cell(monkeypatch, {"planning": {"acceleration_factor": 2.0}})
    rcfg = {"joint_velocity_limits": [1.0, 2.0]}
    assert rk.joint_acceleration_limits("c", rcfg) == [2.0, 4.0]


@pytest.mark.parametrize("cell", ["welding_cell_2_small", "welding_cell_2_small", "welding_cell_2_small"])
def test_shipped_cells_declare_every_speed_key(cell):
    import pathlib
    import yaml
    root = pathlib.Path(__file__).resolve().parents[2] / "cells" / cell
    path = next(p for p in (root / "config" / "robots.yaml",
                            root / "agentic_planner" / "config" / "robots.yaml")
                if p.exists())
    planning = yaml.safe_load(path.read_text())["planning"]
    for key in ("max_velocity_scaling", *rk.PLANNING_SPEED_DEFAULTS):
        assert key in planning, f"{path}: planning.{key} missing"


def test_store_outage_is_not_read_as_an_undeclared_cell(monkeypatch):
    def boom(cell_id):
        raise rk.cell_store.CellStoreUnavailable("down")
    monkeypatch.setattr(rk, "_load_robots_config", boom)
    with pytest.raises(rk.cell_store.CellStoreUnavailable):
        rk.resolve_velocity_scale("c", None)
