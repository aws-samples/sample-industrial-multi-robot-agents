"""``make down`` can take down one side of a split stack and leave the other.

* ``make down`` — everything (incl. RViz), the network and orphans, as before.
* ``make down ONLY=execution`` — the cell controller, every robot agent and robot
  controller, and the RViz viewer.
* ``make down ONLY=trajectory`` — the planner and the orchestrator.

A one-sided down removes only that side's containers: the network stays, because
the other side may still be on it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make(*args):
    if shutil.which("make") is None:
        pytest.skip("make not available")
    r = subprocess.run(["make", "-n", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _compose_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("docker compose")]


def test_plain_down_takes_everything():
    code, out = _make("down", "SIM=welding_cell_2_small")
    assert code == 0, out
    assert _compose_lines(out) == [
        "docker compose -f .generated_compose.yml --profile rviz down --remove-orphans"]


def test_down_execution_takes_the_robots_and_the_viewer_only():
    code, out = _make("down", "SIM=welding_cell_2_small", "ONLY=execution")
    assert code == 0, out
    (line,) = _compose_lines(out)
    assert " rm --stop --force " in line and " down" not in line
    assert "grep -E '^(cell-controller|robot-agent-|robot-controller-)'" in line
    assert line.rstrip().endswith("rviz-web")
    assert "trajectory-agent" not in line and "orchestrator " not in line


def test_down_trajectory_takes_the_planner_and_the_orchestrator_only():
    code, out = _make("down", "SIM=welding_cell_2_small", "ONLY=trajectory")
    assert code == 0, out
    (line,) = _compose_lines(out)
    assert line.rstrip().endswith("rm --stop --force trajectory-agent orchestrator")


def test_an_unknown_scope_is_refused():
    code, out = _make("down", "ONLY=bad")
    assert code != 0 and "ONLY=trajectory, ONLY=execution" in out
