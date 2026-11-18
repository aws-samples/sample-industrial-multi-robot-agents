"""A planning-only stack runs without ROS / DDS, and the dead topics are gone.

* ``make sim ONLY=trajectory`` starts the planner and the orchestrator only, with
  ``DDS_DISABLED=1``; RViz is refused in that scope (it needs the ROS graph).
* With ``DDS_DISABLED=1`` the planner's progress publisher stands down WITHOUT
  importing rclpy — the durable event stream (read over MCP) is the record.
* The orchestrator no longer publishes ``robot_description`` (nobody listened);
  the viewer fetches the URDF from the trajectory agent's MCP instead.
* Topics with a publisher and no subscriber — or the reverse — are removed:
  ``orchestrator/command``, ``cell_config/velocity_scale``, ``orchestrator/status``,
  ``llm_trace``, ``planning_summary``, ``planning_debug``, ``planning_sessions``,
  ``collaborative_plan_markers`` and the cell's ``behaviour_tree`` listener.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make(*args):
    if shutil.which("make") is None:
        pytest.skip("make not available")
    r = subprocess.run(["make", "-n", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_only_trajectory_starts_just_the_planner_and_the_orchestrator():
    code, out = _make("sim", "ONLY=trajectory", "SIM=welding_cell_2_small")
    assert code == 0, out
    up = [ln for ln in out.splitlines() if " up " in ln and "docker compose" in ln][-1]
    assert up.rstrip().endswith("up --no-build trajectory-agent orchestrator"), up
    for absent in ("web-panel", "rviz-web", "cell-controller", "robot-agent", "robot-controller"):
        assert absent not in up


def test_the_full_stack_is_unchanged_without_only():
    code, out = _make("sim", "SIM=welding_cell_2_small")
    assert code == 0, out
    up = [ln for ln in out.splitlines() if " up " in ln and "docker compose" in ln][-1]
    assert up.rstrip().endswith("up --no-build"), up


def test_rviz_is_refused_on_a_planning_only_stack():
    code, out = _make("sim", "ONLY=trajectory", "RVIZ=1")
    assert code != 0 and "drop RVIZ=1" in out


def test_with_dds_disabled_progress_stands_down_without_importing_rclpy(monkeypatch):
    from agents.trajectory import plan_progress
    monkeypatch.setenv("DDS_DISABLED", "1")
    monkeypatch.setattr(plan_progress, "_node", None)
    monkeypatch.setattr(plan_progress, "_publishers", {})
    monkeypatch.setattr(plan_progress, "_disabled_reason", None)
    monkeypatch.delitem(sys.modules, "rclpy", raising=False)
    assert plan_progress.publish("c1", "t1", {"seq": 1, "kind": "k"}) is False
    assert "rclpy" not in sys.modules
    assert "DDS_DISABLED" in (plan_progress._disabled_reason or "")


def test_the_compose_files_carry_the_switch_to_the_planner():
    import yaml
    env = yaml.safe_load((ROOT / "docker-compose.yml").read_text())[
        "services"]["trajectory-agent"]["environment"]
    assert env.get("DDS_DISABLED") == "${DDS_DISABLED:-0}"
    assert '"DDS_DISABLED": "${DDS_DISABLED:-0}"' in (ROOT / "scripts/generate_compose.py").read_text()


DASHBOARD = (ROOT / "dashboard/web_control_panel.py").read_text()


@pytest.mark.parametrize("topic", ["/orchestrator/command", "/cell_config/velocity_scale",
                                   "/orchestrator/status", "/llm_trace", "/planning_summary",
                                   "/planning_debug", "/planning_sessions",
                                   "/collaborative_plan_markers"])
def test_the_dashboard_neither_publishes_nor_listens_on_a_dead_topic(topic):
    assert f'cell_topic("{topic}")' not in DASHBOARD
    assert "cmd_pub" not in DASHBOARD and "speed_pub" not in DASHBOARD


def test_the_orchestrator_publishes_no_robot_description():
    src = (ROOT / "agents/orchestrator/tools/cell_spec.py").read_text()
    assert "rclpy" not in src and "create_publisher" not in src


def test_the_cell_no_longer_listens_for_behaviour_trees_on_dds():
    assert "behaviour_tree\"" not in (ROOT / "agents/cell/server.py").read_text().replace(
        "/{cell_id}/behaviour_tree``", "")
    assert "ros2_listener" not in (ROOT / "agents/cell/mcp_server.py").read_text()


def test_the_viewer_asks_the_planner_for_the_urdf():
    viz = (ROOT / "docker/launch_viz.py").read_text()
    assert '"get_robot_description"' in viz
    assert "def get_robot_description" in (ROOT / "agents/trajectory/mcp_server.py").read_text()
