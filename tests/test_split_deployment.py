"""Planning and execution can run on different machines.

* ``make sim ONLY=execution`` starts the cell controller (executor + collision
  supervisor) and every robot agent / robot controller — nothing else.
* Each side is told where the other is: ``CELL_CONTROLLER_HOST`` for the
  orchestrator (and dashboard), ``ORCHESTRATOR_HOST`` for the cell controller
  (and dashboard). Defaults are the compose names, so one machine is unchanged;
  the planner's own orchestrator address never follows them.
* The cell controller reads plans and cell config THROUGH the orchestrator's
  cell-data MCP (``CELL_DATA_MODE=orchestrator``), so a plan written on the
  planning machine is replayable on the execution machine, with the same "latest
  non-failed plan" rule as the local read.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make(*args):
    if shutil.which("make") is None:
        pytest.skip("make not available")
    r = subprocess.run(["make", "-n", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_only_execution_starts_the_cell_controller_and_the_robots():
    code, out = _make("sim", "ONLY=execution", "SIM=welding_cell_2_small",
                      "ORCHESTRATOR_HOST=10.0.0.4")
    assert code == 0, out
    up = [ln for ln in out.splitlines() if " up " in ln and "docker compose" in ln][-1]
    assert "grep -E '^(cell-controller|robot-agent-|robot-controller-)'" in up
    for absent in ("trajectory-agent", "orchestrator ", "web-panel"):
        assert absent not in up.split("up --no-build", 1)[1].split("$(", 1)[0]
    assert "orchestrator at 10.0.0.4" in out


def test_an_unknown_scope_names_both_valid_ones():
    code, out = _make("sim", "ONLY=bad")
    assert code != 0 and "ONLY=trajectory, ONLY=execution" in out


SERVICES = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
GENERATOR = (ROOT / "scripts/generate_compose.py").read_text()


def test_the_orchestrator_is_told_where_the_cell_controller_is():
    env = SERVICES["orchestrator"]["environment"]
    assert env["CELL_CONTROLLER_HOST"] == "${CELL_CONTROLLER_HOST:-cell-controller}"
    assert '"CELL_CONTROLLER_HOST": "${CELL_CONTROLLER_HOST:-cell-controller}"' in GENERATOR


def test_the_cell_controller_reads_through_the_orchestrator_it_is_told_about():
    env = SERVICES["cell-controller"]["environment"]
    assert env["CELL_DATA_MODE"] == "orchestrator"
    assert env["ORCHESTRATOR_HOST"] == "${ORCHESTRATOR_HOST:-orchestrator}"
    assert '"ORCHESTRATOR_HOST": "${ORCHESTRATOR_HOST:-orchestrator}"' in GENERATOR


def test_the_planner_keeps_its_local_orchestrator():
    assert SERVICES["trajectory-agent"]["environment"]["ORCHESTRATOR_HOST"] == "orchestrator"


def test_the_collision_supervisor_reads_config_through_the_owner():
    src = (ROOT / "agents/cell/collision_monitor.py").read_text()
    assert "cell_store.read_yaml" in src
    assert "with open(robots_yaml)" not in src and "with open(obs_path)" not in src


@pytest.fixture
def remote_plans(monkeypatch):
    """The cell controller in remote mode, with the owner's plan store faked."""
    sys.modules.pop("agents.cell.mcp_server", None)
    try:
        import agents.cell.mcp_server as mcp_mod
    except Exception as exc:  # noqa: BLE001 — no FastMCP / ROS on this host
        pytest.skip(f"cell MCP server not importable here: {exc}")
    from utils import cell_store
    plans = {}

    def entries(cell_id, kind, prefix="", suffix=""):
        return [{"name": n, "mtime_s": m} for n, (m, _) in plans.items()
                if n.startswith(prefix) and n.endswith(suffix)]

    monkeypatch.setattr(cell_store, "is_remote", lambda: True)
    monkeypatch.setattr(cell_store, "list_entries", entries)
    monkeypatch.setattr(cell_store, "read_json",
                        lambda cell_id, kind, name: plans.get(name, (0, None))[1])
    return mcp_mod, plans


def test_the_latest_non_failed_plan_is_found_through_the_owner(remote_plans):
    mcp_mod, plans = remote_plans
    good = {"tree_id": "bt_1_c", "status": "ok",
            "behaviour_tree": {"tree_id": "bt_1_c", "root": {"type": "Sequence"}}}
    bad = {"tree_id": "bt_2_c", "status": "failed", "behaviour_tree": None}
    plans["bt_1_c.json"] = (100.0, good)
    plans["bt_2_c.json"] = (200.0, bad)
    picked = mcp_mod._remote_plan("c", None, lambda r: r.get("status") == "failed")
    assert picked["tree_id"] == "bt_1_c"
    assert mcp_mod._remote_plan("c", "bt_2_c", lambda r: False)["status"] == "failed"
    assert mcp_mod._remote_plan("c", "bt_nope", lambda r: False) is None
