"""The execution side runs the dashboard too.

``make sim ONLY=execution`` starts the web panel next to the cell controller and
the robots — with ``--no-deps``, because the panel declares ``depends_on:
orchestrator`` and must not start a second orchestrator on the execution
machine. ``make down ONLY=execution`` takes it down with the rest of that side.

The panel (and the RViz viewer) reach the planner through TRAJECTORY_AGENT_HOST,
which defaults to ORCHESTRATOR_HOST — both run on the planning machine — and
follows the same loopback rule as the other two addresses.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _make(*args, n=True):
    if shutil.which("make") is None:
        pytest.skip("make not available")
    cmd = ["make", "-n", *args] if n else ["make", "-s", *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _up_line(out):
    return [ln for ln in out.splitlines() if " up " in ln and "docker compose" in ln][-1]


def test_execution_starts_the_dashboard_without_its_dependencies():
    code, out = _make("sim", "ONLY=execution", "SIM=welding_cell_2_small")
    assert code == 0, out
    up = _up_line(out)
    assert " --no-deps " in up and up.rstrip().endswith("web-panel")
    assert "dashboard" in out


def test_execution_down_takes_the_dashboard():
    code, out = _make("down", "ONLY=execution", "SIM=welding_cell_2_small")
    line = [ln for ln in out.splitlines() if ln.startswith("docker compose")][0]
    assert " web-panel " in line + " "


def test_trajectory_scope_still_has_no_dashboard():
    code, out = _make("sim", "ONLY=trajectory", "SIM=welding_cell_2_small")
    assert "web-panel" not in _up_line(out)


@pytest.mark.parametrize("service", ["web-panel", "rviz-web"])
def test_panel_and_viewer_are_told_where_the_planner_is(service):
    env = yaml.safe_load((ROOT / "docker-compose.yml").read_text())[
        "services"][service]["environment"]
    assert env["TRAJECTORY_AGENT_HOST"] == "${TRAJECTORY_AGENT_HOST:-trajectory-agent}"
    gen = (ROOT / "scripts/generate_compose.py").read_text()
    assert gen.count('"TRAJECTORY_AGENT_HOST": "${TRAJECTORY_AGENT_HOST:-trajectory-agent}"') == 2


def _env_seen(tmp_path, *args):
    probe = tmp_path / "probe.mk"
    probe.write_text(f"include {ROOT}/Makefile\n"
                     "probe:\n\t@echo \"T=[$$TRAJECTORY_AGENT_HOST]\"\n")
    code, out = _make("-f", str(probe), "probe", "SIM=welding_cell_2_small", *args, n=False)
    return [ln for ln in out.splitlines() if ln.startswith("T=")][-1]


def test_the_planner_address_follows_the_orchestrator(tmp_path):
    assert _env_seen(tmp_path, "ORCHESTRATOR_HOST=10.0.0.4") == "T=[10.0.0.4]"
    assert _env_seen(tmp_path, "ORCHESTRATOR_HOST=10.0.0.4",
                     "TRAJECTORY_AGENT_HOST=10.0.0.9") == "T=[10.0.0.9]"
    assert _env_seen(tmp_path, "TRAJECTORY_AGENT_HOST=127.0.0.1") == "T=[]"
    assert _env_seen(tmp_path) == "T=[]"
