"""Agent containers are generic; a cell's assets resolve the same everywhere.

* Every agent container runs ONE image (``AGENT_IMAGE``) and only the
  trajectory-agent declares how to build it: a robot container is the generic
  robot container, made robot N by its name and ``ROBOT_ID``. A cell with 16
  arms gets 16 robot agents and 16 controllers from that one image (a
  per-service image left every arm beyond the last build with "No such image").
* ``make sim`` builds the image when it is missing.
* ``cell.yaml`` asset paths: relative to the cell unless they start with ``/``.
  An absolute path's directory is mounted read-only at the SAME path in every
  container that reads the cell, so it resolves there as it does on the host.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_IMAGE = "${AGENT_IMAGE:-multi-robots-agents/agent:latest}"


def _cell(base: Path, n_robots: int, urdf: str) -> None:
    ap = base / "big" / "agentic_planner"
    (ap / "config").mkdir(parents=True)
    robots = {f"robot{i}": {"base_position": [i, 0, 0], "base_orientation_yaw": 0.0,
                            "robot_model": "kr240r2900"} for i in range(1, n_robots + 1)}
    (ap / "config" / "robots.yaml").write_text(yaml.safe_dump({"robots": {"arms": robots}}))
    (ap / "cell.yaml").write_text(yaml.safe_dump({"cell_id": "big", "robots": [
        {"id": f"robot{i}", "namespace": "arms", "urdf": urdf,
         "srdf": "agentic_planner/srdf/arm.srdf"} for i in range(1, n_robots + 1)]}))


def _compose(base: Path) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "scripts/generate_compose.py"),
                          "--cell", "big", "--cell-base", str(base), "--aws-region", "eu-west-1"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return yaml.safe_load(out.stdout)["services"]


def test_sixteen_arms_from_one_generic_image(tmp_path):
    _cell(tmp_path, 16, "robots/arm/arm.urdf")
    s = _compose(tmp_path)
    agents = sorted(k for k in s if k.startswith("robot-agent-"))
    ctrls = sorted(k for k in s if k.startswith("robot-controller-"))
    assert len(agents) == 16 and len(ctrls) == 16
    assert {s[k]["environment"]["ROBOT_ID"] for k in agents} == {f"robot{i}" for i in range(1, 17)}
    agent_services = [k for k in s if k not in ("rviz-web", "policy-trainer")]
    assert {s[k]["image"] for k in agent_services} == {AGENT_IMAGE}
    assert [k for k in agent_services if "build" in s[k]] == ["trajectory-agent"]
    assert {s[k]["command"][0] for k in agents} == {s[agents[0]]["command"][0]}   # generic


def test_relative_paths_are_the_cells_and_mount_nothing(tmp_path):
    _cell(tmp_path, 2, "robots/arm/arm.urdf")
    s = _compose(tmp_path)
    assert not [v for v in s["trajectory-agent"]["volumes"] if "robots/arm" in v]


def test_an_absolute_path_is_mounted_at_the_same_path(tmp_path):
    assets = tmp_path / "elsewhere" / "arm"
    assets.mkdir(parents=True)
    _cell(tmp_path, 2, str(assets / "arm.urdf"))
    s = _compose(tmp_path)
    mount = f"{assets}:{assets}:ro"
    for name in ("trajectory-agent", "orchestrator", "cell-controller",
                 "robot-agent-robot1", "robot-controller-robot2", "rviz-web"):
        assert mount in s[name]["volumes"], name
    assert s["trajectory-agent"]["volumes"].count(mount) == 1      # once, not per robot


def test_the_resolver_reads_a_leading_slash_as_absolute(tmp_path):
    from utils import cell_layout
    assets = tmp_path / "abs"
    assets.mkdir()
    (assets / "arm.urdf").write_text("<robot/>")
    _cell(tmp_path, 1, str(assets / "arm.urdf"))
    assert cell_layout.robot_urdf("big", "robot1", str(tmp_path)) == assets / "arm.urdf"
    rel = tmp_path / "big" / "robots" / "arm"
    rel.mkdir(parents=True)
    (rel / "arm.urdf").write_text("<robot/>")
    (tmp_path / "big" / "agentic_planner" / "cell.yaml").write_text(yaml.safe_dump(
        {"robots": [{"id": "robot1", "urdf": "robots/arm/arm.urdf"}]}))
    assert cell_layout.robot_urdf("big", "robot1", str(tmp_path)) == rel / "arm.urdf"


def test_make_sim_builds_the_agent_image_when_it_is_missing():
    if shutil.which("make") is None:
        pytest.skip("make not available")
    r = subprocess.run(["make", "-n", "sim", "SIM=welding_cell_2_small", "ONLY=execution"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "docker image inspect multi-robots-agents/agent:latest" in r.stdout
    assert "build trajectory-agent" in r.stdout
