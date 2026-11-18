"""``get_cell_info`` must report the arm's FULL base pose, not half of it.

A base pose is a translation AND a yaw. A consumer that placed the arm in the
world from ``base_position`` alone — the dashboard's forward kinematics off joint
states is exactly that consumer — would report end-effector positions rotated
about the base: plausible numbers, wrong point, no error anywhere to notice. The
dashboard used to avoid this by opening ``config/robots.yaml`` itself, which is
the coupling the MCP migration removed, so the tool has to carry the yaw.

The tool body is read from source: importing agents/trajectory/mcp_server.py
pulls in the LLM stack, which a plain checkout does not have. The yaml→robot
enumeration underneath it (``iter_robots``) is exercised directly.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.cell.robots_util import iter_robots  # noqa: E402

TRAJ_MCP = PROJECT_ROOT / "agents" / "trajectory" / "mcp_server.py"

ROTATED_ARMS = """
robots:
  arms:
    robot1:
      base_position: [0.07, -3.32, 0]
      base_orientation_yaw: 1.5707963
      workspace_radius: 2.9
      end_effector_link: welding_tcp
    robot2:
      base_position: [-1.18, 1.04, 0]
      base_orientation_yaw: -0.7853982
      workspace_radius: 2.9
      end_effector_link: welding_tcp
"""

NO_YAW = """
robots:
  arms:
    robot1:
      base_position: [1.0, 2.0, 0]
      end_effector_link: tool0
"""


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """Write robots.yaml into a scratch cell and return its id."""
    def _make(yaml_text: str, cell_id: str = "welding_cell_2_small") -> str:
        cfg = tmp_path / cell_id / "config"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "robots.yaml").write_text(yaml_text)
        monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
        return cell_id
    return _make


def _cell_info_source() -> str:
    tree = ast.parse(TRAJ_MCP.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "get_cell_info")
    return ast.get_source_segment(TRAJ_MCP.read_text(), fn) or ""


class TestTheToolReportsTheYaw:

    def test_get_cell_info_includes_base_orientation_yaw(self):
        assert "base_orientation_yaw" in _cell_info_source(), (
            "get_cell_info must report base_orientation_yaw — a consumer that "
            "gets the translation without the rotation misplaces the arm")

    def test_the_yaw_is_reported_alongside_the_position(self):
        """One pose, one entry: they must not be reachable through different tools."""
        body = _cell_info_source()
        assert "base_position" in body
        pos = body.index("base_position")
        yaw = body.index("base_orientation_yaw")
        assert abs(pos - yaw) < 600, "the yaw drifted away from the position it belongs to"

    def test_the_yaw_defaults_to_zero_rather_than_none(self):
        """An arm with no yaw configured is not rotated; None would break arithmetic."""
        assert 'rcfg.get("base_orientation_yaw", 0.0)' in _cell_info_source()

    def test_the_yaw_is_coerced_to_float(self):
        """YAML gives ints for `0`; a consumer doing trigonometry wants a float."""
        assert 'float(rcfg.get("base_orientation_yaw"' in _cell_info_source()


class TestTheUnderlyingConfigCarriesTheYaw:
    """``iter_robots`` is what the tool enumerates, so the yaw must survive it."""

    def test_each_arm_keeps_its_own_yaw(self, cell):
        cell_id = cell(ROTATED_ARMS)

        yaws = {rid: rcfg.get("base_orientation_yaw")
                for _ns, rid, rcfg in iter_robots(cell_id)}

        assert yaws == {"robot1": 1.5707963, "robot2": -0.7853982}

    def test_an_arm_without_a_yaw_simply_has_none_to_default(self, cell):
        cell_id = cell(NO_YAW)

        _ns, rid, rcfg = next(iter(iter_robots(cell_id)))

        assert rid == "robot1"
        assert rcfg.get("base_orientation_yaw") is None
        assert float(rcfg.get("base_orientation_yaw", 0.0)) == 0.0

    def test_every_arm_is_enumerated_under_a_shared_namespace(self, cell):
        """Regression guard shared with test_get_cell_info: not just the first arm."""
        cell_id = cell(ROTATED_ARMS)

        assert [rid for _ns, rid, _cfg in iter_robots(cell_id)] == ["robot1", "robot2"]
