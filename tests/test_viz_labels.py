"""The viewer shows the labels the configuration gives — and only those.

* A weld point with a ``label`` in the task file carries a camera-facing TEXT
  marker with it, just above the spot. No label, no text.
* A robot with a ``label`` in robots.yaml gets a coloured disc at its base and the
  label beside it, on the cell side (toward the centroid of the bases) where the
  camera sees it. No label, neither.
* The labels are published on /viz/robot_labels, which the RViz config shows.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "docker" / "launch_viz.py"


@pytest.fixture(scope="module")
def lv():
    vm = pytest.importorskip("visualization_msgs.msg")
    # tests/test_dashboard_asks_mcp_not_the_volume.py registers MagicMock stand-ins
    # for the ROS message packages for the rest of the session; markers built from
    # those carry nothing to assert on. These tests need the real classes (the
    # RViz image has them).
    if not isinstance(getattr(vm, "Marker", None), type):
        pytest.skip("visualization_msgs is a stand-in in this session, not ROS")
    spec = importlib.util.spec_from_file_location("launch_viz_labels", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"launcher not importable here: {exc}")
    return mod


def _texts(markers):
    from visualization_msgs.msg import Marker
    return [m for m in markers.markers if m.type == Marker.TEXT_VIEW_FACING]


def test_a_weld_point_shows_its_configured_label(lv):
    points = [{"id": "W01", "position": [1.0, 2.0, 1.3], "group": "structural",
               "approach": [0, 0, -1], "label": "W01 sill"},
              {"id": "W02", "position": [0.5, 2.0, 1.3], "group": "secondary",
               "approach": [0, 0, -1], "label": "W02"},
              {"id": "W03", "position": [0.0, 2.0, 1.3], "group": "secondary",
               "approach": [0, 0, -1], "label": None}]
    texts = _texts(lv.build_weld_markers(points))
    assert [t.text for t in texts] == ["W01 sill", "W02"]
    t = texts[0]
    assert (t.pose.position.x, t.pose.position.y) == (1.0, 2.0)
    assert t.pose.position.z == pytest.approx(1.3 + lv.WELD_LABEL_LIFT_M)
    assert t.scale.z == lv.WELD_LABEL_HEIGHT_M and t.header.frame_id == "world"
    ids = [m.id for m in lv.build_weld_markers(points).markers]
    assert len(ids) == len(set(ids)), "marker ids must be unique or RViz drops some"


def test_a_robot_shows_its_configured_label_at_its_base(lv):
    from visualization_msgs.msg import Marker
    robots = [{"robot_id": "robot1", "base": [0.0, -3.0, 0.0], "label": "Robot 1"},
              {"robot_id": "robot2", "base": [0.0, 1.0, 0.0], "label": "Robot 2"}]
    markers = lv.build_robot_label_markers(robots)
    texts = _texts(markers)
    assert [t.text for t in texts] == ["Robot 1", "Robot 2"]
    # On the CELL side, toward the centroid (0, -1): robot1 at +y of its base,
    # robot2 at -y of its base — in front of each robot, where the camera sees it.
    assert texts[0].pose.position.y == pytest.approx(-3.0 + lv.ROBOT_LABEL_OFFSET_M)
    assert texts[1].pose.position.y == pytest.approx(1.0 - lv.ROBOT_LABEL_OFFSET_M)
    discs = [m for m in markers.markers if m.type == Marker.CYLINDER]
    assert [(d.pose.position.x, d.pose.position.y) for d in discs] == [(0.0, -3.0), (0.0, 1.0)]
    assert discs[0].color != discs[1].color, "the two robots must be told apart"
    assert texts[0].color == discs[0].color or texts[0].color.r == discs[0].color.r


def test_a_single_robot_still_gets_a_label(lv):
    texts = _texts(lv.build_robot_label_markers(
        [{"robot_id": "r", "base": [2, 3, 0], "label": "R"}]))
    assert texts[0].text == "R"
    assert math.hypot(texts[0].pose.position.x - 2, texts[0].pose.position.y - 3) \
        == pytest.approx(lv.ROBOT_LABEL_OFFSET_M)


def test_nothing_is_drawn_without_a_label(lv):
    robots = [{"robot_id": "robot1", "base": [0, -3, 0]},
              {"robot_id": "robot2", "base": [0, 1, 0], "label": "  "}]
    assert lv.build_robot_label_markers(robots).markers == []
    welds = [{"id": "W01", "position": [1, 2, 1.3], "group": "", "approach": [0, 0, -1]}]
    assert _texts(lv.build_weld_markers(welds)) == []
    assert lv._label(None) is None and lv._label(" ") is None and lv._label("A ") == "A"


def test_the_labels_are_read_from_the_configuration():
    src = LAUNCHER.read_text()
    assert '"label":    _label(op.get("label"))' in src          # task file
    assert '"label":     _label(rcfg.get("label"))' in src       # robots.yaml


def test_the_rviz_config_shows_the_robot_labels():
    src = LAUNCHER.read_text()
    assert '"Value": "/viz/robot_labels"' in src
    assert 'create_publisher(MarkerArray, "/viz/robot_labels"' in src
