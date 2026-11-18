"""A loopback address for the other side means "this machine", never the container.

``make sim ONLY=execution ORCHESTRATOR_HOST=127.0.0.1`` used to hand 127.0.0.1 to
the cell controller, which then called its OWN :8893 and refused every read
("cell-data MCP http://127.0.0.1:8893/mcp is unreachable"). On one machine both
scopes share the compose network, so a loopback value now falls back to the
compose service name, and a real address passes through unchanged.
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


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.1.1", "localhost", "0.0.0.0"])
def test_a_loopback_orchestrator_is_the_compose_service(host):
    code, out = _make("sim", "ONLY=execution", "SIM=welding_cell_2_small",
                      f"ORCHESTRATOR_HOST={host}")
    assert code == 0, out
    assert "using the compose service 'orchestrator'" in out
    assert "orchestrator at orchestrator" in out


def test_a_loopback_cell_controller_is_the_compose_service():
    code, out = _make("sim", "ONLY=trajectory", "SIM=welding_cell_2_small",
                      "CELL_CONTROLLER_HOST=127.0.0.1")
    assert code == 0, out
    assert "using the compose service 'cell-controller'" in out


def test_a_real_address_is_kept():
    code, out = _make("sim", "ONLY=execution", "SIM=welding_cell_2_small",
                      "ORCHESTRATOR_HOST=10.0.0.4")
    assert code == 0, out
    assert "orchestrator at 10.0.0.4" in out and "compose service" not in out
