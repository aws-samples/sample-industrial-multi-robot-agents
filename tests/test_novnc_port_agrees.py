"""The panel embeds the viewer by URL, so two services must agree on one port.

The left column of the dashboard is an ``<iframe>`` pointed at the noVNC server
inside the **rviz-web** container. Nothing connects those two services at
runtime: the browser resolves the URL itself, from a port the panel is told
about via ``NOVNC_HOST_PORT`` and a port rviz-web publishes in its own ``ports``
mapping. Two independent numbers for one address — and if they drift the panel
comes up perfectly, with a black rectangle and "unreachable" where the
simulation should be, which looks like a broken viewer rather than a broken
compose file.

So the number is declared once in ``scripts/generate_compose.py`` and this test
pins both uses to it. It is the reason ``NOVNC_HOST_PORT`` exists as a constant
instead of two literals.

Host-only: calls two pure functions in the generator. No docker, no network.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PANEL = PROJECT_ROOT / "dashboard" / "web_control_panel.py"


def _generate_compose():
    """Load the generator by path — ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "generate_compose_novnc", PROJECT_ROOT / "scripts" / "generate_compose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _generate_compose()
ROS_ENV = {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}
PANEL_SVC = GEN.web_panel_service("some_cell", ROS_ENV, {})
RVIZ_SVC = GEN.rviz_web_service(ROS_ENV)


class TestTheViewerPortIsDeclaredOnce:
    def test_the_constant_is_a_plausible_port(self):
        """Guard the guard: everything below is trivially true of a typo like 0."""
        assert isinstance(GEN.NOVNC_HOST_PORT, int)
        assert 1024 <= GEN.NOVNC_HOST_PORT <= 65535, \
            f"{GEN.NOVNC_HOST_PORT} is not a publishable unprivileged port"

    def test_rviz_web_publishes_it(self):
        """Host side of the mapping, container side is noVNC's own 80."""
        mappings = [str(p) for p in RVIZ_SVC["ports"]]
        assert f"{GEN.NOVNC_HOST_PORT}:80" in mappings, (
            f"rviz-web publishes {mappings}, so the viewer is not reachable on "
            f"{GEN.NOVNC_HOST_PORT} where the panel sends the browser")

    def test_the_panel_is_told_the_same_number(self):
        assert PANEL_SVC["environment"].get("NOVNC_HOST_PORT") \
            == str(GEN.NOVNC_HOST_PORT), (
            "web-panel's NOVNC_HOST_PORT does not match what rviz-web publishes; "
            "the iframe would point at a closed port")

    def test_no_other_service_claims_that_host_port(self):
        """A collision makes rviz-web fail to *start*, not just fail to be reached.

        The whole compose document is assembled inside ``main()``, which parses
        argv and writes a file, so the source's port mappings are scanned instead
        of building it: every ``"ports": [...]`` literal in the generator, only
        one of which may publish this host port.
        """
        source = (PROJECT_ROOT / "scripts" / "generate_compose.py").read_text()
        blocks = re.findall(r'"ports"\s*:\s*\[(.*?)\]', source, re.S)
        assert blocks, "the generator declares no port mappings — did it move?"
        port = str(GEN.NOVNC_HOST_PORT)
        claiming = [b.strip() for b in blocks
                    if re.search(rf'(^|[^\d]){port}\s*:', b) or "NOVNC_HOST_PORT" in b]
        assert len(claiming) == 1, (
            f"host port {port} is published by {len(claiming)} services "
            f"({claiming}); docker refuses the second one and rviz-web may lose")


class TestThePanelReadsItFromTheEnvironment:
    """A default baked into the panel must not silently replace the env value.

    ``NOVNC_PORT = os.environ.get("NOVNC_HOST_PORT", "6080")`` is fine; a bare
    literal in the template is not, because then the compose constant would move
    and the iframe would not.
    """

    def test_the_panel_takes_the_port_from_the_env_var_the_generator_sets(self):
        source = PANEL.read_text()
        assert re.search(r"""os\.environ\.get\(\s*["']NOVNC_HOST_PORT["']""", source), \
            "the panel no longer reads NOVNC_HOST_PORT; the compose value is dead"

    def test_the_template_injects_it_rather_than_hardcoding_a_port(self):
        source = PANEL.read_text()
        assert "novnc_port=NOVNC_PORT" in source.replace(" ", ""), \
            "the render call does not pass novnc_port into the template"
        assert '{{ novnc_port }}' in source, \
            "the script does not read the injected port"
        # The failure this catches: a literal :6080 left in the JS, which works
        # until the constant above changes and then silently does not.
        hardcoded = re.findall(r'":6080"|:6080/', source)
        assert not hardcoded, \
            f"the viewer port is hardcoded in the panel as well: {hardcoded}"


class TestTheFallbackDefaultAgreesToo:
    """The panel can run outside compose (``make dashboard``), so its default
    is a real code path — and a stale default is a silent misconfiguration."""

    def test_the_panels_default_matches_the_generator(self):
        default = re.search(
            r"""os\.environ\.get\(\s*["']NOVNC_HOST_PORT["']\s*,\s*["'](\d+)["']""",
            PANEL.read_text())
        assert default, "the panel's NOVNC_HOST_PORT lookup has no default to check"
        assert default.group(1) == str(GEN.NOVNC_HOST_PORT), (
            f"the panel defaults to {default.group(1)} but compose publishes "
            f"{GEN.NOVNC_HOST_PORT}")
