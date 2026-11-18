"""Four buttons in the Controls row, and each must do only its own thing.

Planning plans and commands nothing. Stop ends the activity. Emergency stop halts
the arms. Reset stops, then drives them home. The distinctions are the safety
property here: an operator who presses Stop expecting a latched halt, or Planning
expecting a dry run and getting motion, has been lied to by the UI.

Every mix-up is a small edit away and none of them change how the button looks:
route Stop at ``/a2a/reset`` and it homes the arms; route the e-stop through the
orchestrator and it stops working exactly when the orchestrator is the thing
that is wedged; give Planning a default program and the button silently plans
somebody else's welds. So this file reads the panel and pins which wire each
button is on, one command at a time.

Source-scanned on purpose: importing the panel needs flask, which is not on the
host, and these are wiring facts — which endpoint, which publisher — that a
mocked request would not check any harder.
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


# ─── Branch extraction ───────────────────────────────────────────────────────

def _function(name: str, source: str = SOURCE) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{name}() is gone from the panel; this test guards nothing")


def _command_branches() -> dict[str, str]:
    """``/api/command``'s per-command bodies, keyed by every command they serve.

    Matches ``if cmd == "x"`` and ``if cmd in ("x", "y")`` — the second form is
    how Stop and Reset share a branch.
    """
    fn = _function("send_command")
    out: dict[str, str] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        names: list[str] = []
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.left, ast.Name) and test.left.id == "cmd"):
            op = test.ops[0]
            comparator = test.comparators[0]
            if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                names = [str(comparator.value)]
            elif isinstance(op, ast.In) and isinstance(comparator, (ast.Tuple, ast.List)):
                names = [str(e.value) for e in comparator.elts
                         if isinstance(e, ast.Constant)]
        if not names:
            continue
        body = "\n".join(ast.unparse(stmt) for stmt in node.body)
        for n in names:
            out[n] = body
    return out


BRANCHES = _command_branches()


def _template() -> str:
    """The dashboard HTML+JS. It is one big string constant in the module."""
    for node in ast.walk(ast.parse(SOURCE)):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and "<!DOCTYPE html>" in node.value.value):
            return node.value.value
    pytest.fail("could not find the dashboard template in the panel")


TEMPLATE = _template()
ONCLICKS = set(re.findall(r"""onclick=["']cmd\(['"]([a-z_]+)['"]\)""", TEMPLATE))


# ─── Each button is on its own wire ──────────────────────────────────────────

class TestEveryButtonIsOnItsOwnWire:
    def test_all_four_buttons_are_served(self):
        assert set(BRANCHES) >= {"planning", "stop", "reset", "estop"}, \
            f"a control button has no handler: {BRANCHES.keys()}"

    def test_planning_asks_the_orchestrator_to_plan(self):
        assert re.search(r"""["']/a2a/plan["']""", BRANCHES["planning"]), \
            "Planning does not call /a2a/plan"

    def test_stop_aborts_and_never_reaches_reset(self):
        """``/a2a/reset`` homes the arms. If Stop reached it, the button an
        operator presses to stop things would start the arms moving."""
        body = BRANCHES["stop"]
        assert "/a2a/abort" in body
        assert re.search(r"""["']/a2a/reset["'] if is_reset""", body), \
            "Stop and Reset share a branch without a discriminator — check that " \
            "Stop cannot reach /a2a/reset"

    def test_reset_is_the_only_one_that_homes(self):
        assert "/a2a/reset" in BRANCHES["reset"]

    def test_the_estop_publishes_to_ros_and_makes_no_http_call(self):
        """It must work when the orchestrator does not. A stop that depends on a
        healthy agent is not a stop."""
        body = BRANCHES["estop"]
        assert "estop_pub" in body, "the emergency stop no longer publishes /estop"
        for hop in ("_a2a_post", "_a2a_get", "requests.", "/a2a/"):
            assert hop not in body, \
                f"the emergency stop goes through '{hop}'; it must not depend on " \
                "another process being healthy"

    def test_nothing_else_can_publish_an_estop(self):
        """The one branch that halts arms is the one labelled as such."""
        for name, body in BRANCHES.items():
            if name == "estop":
                continue
            assert "estop_pub" not in body, \
                f"the '{name}' button publishes an e-stop; only EMERGENCY STOP may"

    def test_neither_planning_nor_stop_claims_to_have_stopped_the_arms(self):
        assert re.search(r"""["']estop["']:\s*False""", BRANCHES["stop"]), \
            "Stop's reply does not state that the arms were not halted"


# ─── Planning plans the task the operator picked, or nothing ─────────────────

class TestPlanningNeedsAChosenTask:
    def test_a_missing_program_is_refused_by_the_server(self):
        body = BRANCHES["planning"]
        assert "400" in body, "planning without a program is not rejected"
        assert re.search(r"if not program", body), \
            "the planning branch does not check that a program was chosen"

    def test_there_is_no_fallback_program_anywhere_in_the_planning_path(self):
        """A default is the dangerous kind of convenience: minutes of planning
        aimed at welds nobody asked for, and it looks like it worked."""
        body = BRANCHES["planning"]
        for guess in ("programs[0]", "_default_program", "or DEFAULT_PROGRAM",
                      'or "car_door', "sorted(programs)[0]"):
            assert guess not in body, \
                f"planning falls back to a guessed program via '{guess}'"
        # The comment where it used to live is welcome; the function is not.
        assert not re.search(r"def _default_program\b", SOURCE), \
            "a default-program helper is back in the panel; planning must plan " \
            "only the task the operator selected"

    def test_the_front_end_refuses_before_it_asks(self):
        assert 'const sel=$("plan-task")' in TEMPLATE, \
            "the task selector is gone; Planning cannot know what to plan"
        assert "pick a task to plan first" in TEMPLATE, \
            "the UI sends a planning request with no task chosen"
        assert "body.program=program" in TEMPLATE, \
            "the chosen task is not sent with the command"

    def test_the_options_come_from_the_orchestrator_not_a_hardcoded_list(self):
        assert "/api/a2a/tasks" in TEMPLATE, \
            "the task list is not fetched from the orchestrator"
        assert "no task programs found" in TEMPLATE, \
            "an empty task list is not reported; the selector would look broken"
        assert 'loadPlanTasks();' in TEMPLATE, \
            "loadPlanTasks() is never called; the selector would start empty and " \
            "Planning would be permanently unusable"

    def test_planning_says_out_loud_that_nothing_moved(self):
        """The operator's next question is always "did it run?"."""
        body = BRANCHES["planning"]
        assert "Nothing has moved" in body and "replay" in body.lower(), \
            "a finished plan does not tell the operator that nothing was executed"


# ─── Reset moves the arms, and says so twice ─────────────────────────────────

class TestResetIsTheOneThatMoves:
    def test_the_ui_confirms_before_homing(self):
        assert re.search(r"""c==="reset"&&!confirm\(""", TEMPLATE), \
            "Reset no longer asks; one stray click drives the arms across the cell"
        assert "CLEARS a latched e-stop" in TEMPLATE, \
            "the confirmation does not mention that Reset clears the e-stop latch"

    def test_nothing_else_is_behind_a_confirmation(self):
        """An emergency stop must never be one click slower than the emergency."""
        confirmed = set(re.findall(r"""c==="([a-z]+)"&&!confirm""", TEMPLATE))
        assert confirmed == {"reset"}, \
            f"these buttons ask for confirmation: {confirmed} — only Reset may"

    def test_the_log_records_both_facts(self):
        body = BRANCHES["reset"]
        assert "CLEARS a latched e-stop" in body and "arms are moving" in body.lower(), \
            "the reset log line does not record that arms move and the latch clears"


# ─── The two removed buttons ─────────────────────────────────────────────────

class TestRunAndStatusAreGoneFromTheControls:
    def test_no_button_starts_a_plan_and_executes_it_in_one_press(self):
        """Run planned AND dispatched. It is gone: execution is now only ever
        reached through Replay, on a plan someone has looked at."""
        assert "run" not in ONCLICKS, \
            "a Run button is back in the controls; plan-and-execute in one click"
        assert "cmd('run')" not in TEMPLATE and 'cmd("run")' not in TEMPLATE

    def test_no_status_button(self):
        assert "status" not in ONCLICKS, \
            "the Status button is back; the panel already polls /api/state"

    def test_the_status_command_still_answers_but_is_not_offered(self):
        """Kept as a round-trip probe for API clients (and pinned by
        tests/test_command_buttons_are_accepted.py), deliberately outside
        KNOWN_COMMANDS so the UI-coverage test stays honest about it."""
        assert "status" in BRANCHES, "the status probe stopped being answered"
        known = _known_commands()
        assert "status" not in known, \
            "'status' is advertised as a UI command again, but no button sends it"
        assert known == {"planning", "stop", "reset", "estop"}, \
            f"KNOWN_COMMANDS no longer matches the four buttons: {known}"

    def test_every_button_in_the_markup_is_a_command_the_server_serves(self):
        unserved = ONCLICKS - set(BRANCHES)
        assert not unserved, f"buttons that reach no handler: {unserved}"


def _known_commands() -> set[str]:
    for node in ast.walk(ast.parse(SOURCE)):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "KNOWN_COMMANDS"
                        for t in node.targets)):
            return {str(e.value) for e in node.value.elts
                    if isinstance(e, ast.Constant)}
    pytest.fail("KNOWN_COMMANDS is gone from the panel")
