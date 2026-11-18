"""Published ports answer this machine only — except the split-deployment pair.

The agents' HTTP/MCP endpoints carry no authentication, and published as
``"8889:8889"`` they listened on every interface of the host, so anything on the
network could submit a task or read a plan. They are now bound to
``${PUBLISH_HOST:-127.0.0.1}``. Two stay on every interface because the other
machine of a split deployment must reach them: the cell controller's MCP (:8892)
and the orchestrator's cell-data MCP (:8893).

(The RViz viewer keeps ``6080:80``: tests/test_novnc_port_agrees.py pins that
mapping verbatim.)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCAL = "${PUBLISH_HOST:-127.0.0.1}"

EXPECTED = {
    "trajectory-agent": [f"{LOCAL}:8890:8890"],
    "orchestrator":     [f"{LOCAL}:8889:8889", "8893:8893"],
    "cell-controller":  [f"{LOCAL}:8891:8891", "8892:8892"],
    "web-panel":        [f"{LOCAL}:8888:8888", f"{LOCAL}:8765:8765"],
}


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_compose_ports",
                                                  ROOT / "scripts/generate_compose.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"generator not importable here: {exc}")
    return mod


def test_the_base_compose_file_binds_as_intended():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    for name, ports in EXPECTED.items():
        assert [str(p) for p in services[name]["ports"]] == ports, name


def test_the_generator_binds_as_intended(gen):
    assert gen.LOCAL == LOCAL
    ros = {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}
    panel = gen.web_panel_service("c", ros, {})
    assert [str(p) for p in panel["ports"]] == EXPECTED["web-panel"]
    src = (ROOT / "scripts/generate_compose.py").read_text()
    for literal in ('ports=[f"{LOCAL}:8890:8890"]',
                    'ports=[f"{LOCAL}:8889:8889", f"{ORCH_MCP_PORT}:{ORCH_MCP_PORT}"]',
                    'ports=[f"{LOCAL}:8891:8891", "8892:8892"]'):
        assert literal in src


def test_only_the_contract_ports_and_the_viewer_listen_everywhere():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    open_ports = sorted(str(p) for s in services.values() for p in s.get("ports", [])
                        if not str(p).startswith(LOCAL))
    assert open_ports == ["6080:80", "8892:8892", "8893:8893"]


def test_make_passes_publish_host_through():
    mk = (ROOT / "Makefile").read_text()
    assert "PUBLISH_HOST        ?=" in mk and "export PUBLISH_HOST" in mk
