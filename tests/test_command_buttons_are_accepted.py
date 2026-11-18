"""Every control-panel button must send a command the server acts on.

``POST /api/command`` used to end with a bare ``return jsonify({"ok": True, ...})``
for anything it did not recognise. So a button whose command string had drifted —
renamed in the markup, or never implemented — answered ``ok: true``, the panel
printed a green tick, and nothing happened. That is precisely the failure the
front end was fixed for ("the commands seem not to work"), one layer down, and it
is invisible from either side alone: the button looks wired, the route looks
healthy.

Two halves, both needed:
  * the server rejects what it will not act on (so a drifted name is loud), and
  * every command the markup actually sends is in the accepted set (so the
    rejection above can never fire for a real button).

Host-only: the markup comes out of the source with ``ast`` and the dispatch is
read as text — the module builds a ROS node at import time and cannot be imported
here (see ``tests/test_dashboard_script_loads.py``).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PANEL = PROJECT_ROOT / "dashboard" / "web_control_panel.py"
SOURCE = PANEL.read_text()


def _assigned_tuple(name: str) -> tuple[str, ...]:
    """A module-level tuple/list of string literals, read without importing."""
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == name for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    pytest.fail(f"{name} is not assigned at module level in {PANEL.name}")


def _html_template() -> str:
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "HTML_TEMPLATE" for t in node.targets):
            return node.value.value
    pytest.fail("HTML_TEMPLATE not found")


KNOWN = _assigned_tuple("KNOWN_COMMANDS")
TEMPLATE = _html_template()


def _commands_the_page_sends() -> set[str]:
    """Both spellings: ``cmd('run')`` and a hand-rolled ``{command:'reset'}``."""
    return (set(re.findall(r"""\bcmd\(\s*['"]([a-z_]+)['"]\s*\)""", TEMPLATE))
            | set(re.findall(r"""command\s*:\s*['"]([a-z_]+)['"]""", TEMPLATE)))


class TestTheServerRejectsWhatItWillNotDo:
    def test_the_dispatch_has_a_rejecting_tail(self):
        """The fall-through must not be an unconditional ok."""
        tail = SOURCE.split("def send_command(")[1].split("\n@app.route")[0]
        assert "unknown command" in tail, \
            "send_command no longer names unknown commands; a drifted button " \
            "would be acked instead of reported"
        assert re.search(r'"ok":\s*False.*\n?.*400|400', tail), \
            "the unknown-command branch does not answer an error status"

    def test_every_known_command_has_a_branch_that_handles_it(self):
        """KNOWN_COMMANDS is the promise; the dispatch is the implementation."""
        tail = SOURCE.split("def send_command(")[1].split("\n@app.route")[0]
        for command in KNOWN:
            assert re.search(rf'''["']{command}["']''', tail), (
                f"'{command}' is advertised in KNOWN_COMMANDS but the dispatch "
                f"never mentions it, so it falls through to the 400")

    def test_status_is_still_acked(self):
        """It does nothing by design — the panel polls /api/state — but the
        button must prove the round trip, so it is a 200 and not a rejection."""
        tail = SOURCE.split("def send_command(")[1].split("\n@app.route")[0]
        assert 'if cmd == "status":' in tail, \
            "status must be handled before the rejecting tail"


class TestNoButtonSendsSomethingTheServerRejects:
    def test_the_page_sends_at_least_one_command(self):
        """Guard the guard: an empty set passes the check below trivially."""
        assert _commands_the_page_sends(), \
            "no command strings found in the template — did the buttons move?"

    @pytest.mark.parametrize("command", sorted(_commands_the_page_sends()))
    def test_the_command_is_in_the_accepted_set(self, command):
        assert command in KNOWN, (
            f"the panel sends command '{command}', which POST /api/command now "
            f"rejects with 400; accepted: {', '.join(KNOWN)}")

    def test_no_accepted_command_is_unreachable_from_the_ui(self):
        """A command the server handles that no control sends is dead code — or a
        control that was dropped from the markup, which is how this started."""
        sent = _commands_the_page_sends()
        orphans = sorted(c for c in KNOWN if c not in sent)
        assert not orphans, (
            f"{orphans} are accepted by /api/command but nothing in the panel "
            f"sends them; either wire a control or drop them from KNOWN_COMMANDS")
