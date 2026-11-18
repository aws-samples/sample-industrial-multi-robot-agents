"""``make sim ONLY=execution`` starts the RViz viewer unless told not to.

The viewer belongs to the execution side, and it kept being missing because the
execution command was run without RVIZ=1. Empty RVIZ now means the scope's
default — on for execution, off for the full stack and for planning — and an
explicit RVIZ=0 / RVIZ=1 still decides.
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
    r = subprocess.run(["make", "-n", "sim", "SIM=welding_cell_2_small", *args],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("args, state", [
    (("ONLY=execution",), "rviz=on"),
    (("ONLY=execution", "RVIZ=0"), "rviz=off"),
    (("ONLY=execution", "RVIZ=1"), "rviz=on"),
    ((), "rviz=off"),
    (("RVIZ=1",), "rviz=on"),
    (("ONLY=trajectory",), "rviz=off"),
])
def test_the_viewer_follows_the_scope(args, state):
    code, out = _make(*args)
    assert code == 0, out
    assert state in out


def test_execution_brings_up_the_viewer_service():
    code, out = _make("ONLY=execution")
    up = [ln for ln in out.splitlines() if " up " in ln and "docker compose" in ln][-1]
    assert "--profile rviz" in up and " rviz-web " in up
