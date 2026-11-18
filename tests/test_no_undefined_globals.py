"""Every global a function reaches for must exist in the module that reaches for it.

``run_mrta_pipeline`` called ``_tsp_2opt``, and nothing imported it. Three of the
four scheduling helpers came along when the code moved out of ``rrt_planner``; the
fourth was left behind. Python does not care until the line runs, and the line only
runs during a real planning job — where the pipeline's own ``except`` turned the
``NameError`` into ``{"error": "name '_tsp_2opt' is not defined"}``. The ReAct loop
read that as a transient fault and retried it, so the whole plan failed as a
sequencing problem rather than as the missing import it was.

Nothing caught it earlier because nothing could: the import-time tests import the
module fine (the name is only looked up on call), and the unit tests around
allocation call ``task_allocation`` directly, where the helper does exist.

This checks every module under ``agents/`` with ``symtable``: for each function
scope, a name read but never assigned there must resolve to a module-level name or
a builtin. Static and stdlib-only — no numpy, no langchain, no container. That is
the point: the bug class is "the name is not there", which does not need the
package's dependencies to detect, and this suite runs on a host that has neither.
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories of first-party source. Not ``tests`` — a test's fixtures are injected
# by pytest and read as undefined globals here.
ROOTS = ("agents", "controller", "dashboard", "scripts", "utils", "cells", "docker")

# The import machinery injects these; they are module globals that no statement in
# the module assigns, so symtable cannot see them.
MODULE_DUNDERS = frozenset({
    "__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
    "__builtins__", "__path__", "__debug__",
})


def _undefined_globals(source: str, filename: str) -> list[str]:
    """``["scope -> name", ...]`` for every unresolvable global read."""
    table = symtable.symtable(source, filename, "exec")
    module_level = {sym.get_name() for sym in table.get_symbols()} | MODULE_DUNDERS

    found: list[str] = []

    def walk(scope) -> None:
        for child in scope.get_children():
            for sym in child.get_symbols():
                name = sym.get_name()
                # is_global() means: read here, assigned in no enclosing scope.
                if (sym.is_global() and name not in module_level
                        and not hasattr(builtins, name)):
                    found.append(f"{child.get_name()} -> {name}")
            walk(child)

    walk(table)
    return found


# Offenders that predate this fence, recorded rather than hidden. Marked
# xfail(strict) so the suite stays green while the debt stays visible: fix the file
# and the XPASS fails the run, which is the reminder to delete the entry.
#
# Empty on purpose. The one entry was dashboard/web_control_panel.py, for two
# breaks with one cause: a lost `def _execute_chat_action(act)` header left that
# function's body sitting inside `set_loop` after its return — dead code reading an
# undefined `act` — while `chat()` called a name that existed nowhere, inside a
# `try/except Exception` that logged and carried on. So every chat-issued action
# (run, stop, reset, replay, move, create_weld) failed silently. The signature is
# back and the stale duplicate of do_replay's body next to it is gone, so the file
# is scanned for real now.
KNOWN_BROKEN: set = set()


def _sources() -> list:
    params = []
    for root in ROOTS:
        for path in sorted((PROJECT_ROOT / root).rglob("*.py")):
            rel = path.relative_to(PROJECT_ROOT)
            marks = ([pytest.mark.xfail(strict=True, reason=f"pre-existing: {rel}")]
                     if rel in KNOWN_BROKEN else [])
            params.append(pytest.param(path, marks=marks, id=str(rel)))
    return params


def test_the_scan_actually_covers_the_source():
    """A fence that walks an empty list passes forever."""
    assert len(_sources()) > 50


@pytest.mark.parametrize("path", _sources())
def test_no_function_reads_a_name_that_is_not_there(path):
    rel = path.relative_to(PROJECT_ROOT)
    offenders = _undefined_globals(path.read_text(), str(rel))

    assert not offenders, (
        f"{rel} reads globals that are defined nowhere: {', '.join(offenders)}"
        " — most likely a helper that moved modules without its import"
    )


def test_the_check_would_have_caught_the_missing_import():
    """The fence is only worth having if it fails on the bug it was written for."""
    source = (
        "from mod import _thermal_spread\n"
        "def run_mrta_pipeline(ops):\n"
        "    return _thermal_spread(_tsp_2opt(ops))\n"
    )

    assert _undefined_globals(source, "sample.py") == ["run_mrta_pipeline -> _tsp_2opt"]


def test_a_lazy_import_inside_a_function_is_not_an_offender():
    """Half the planner imports inside the function body to keep startup cheap."""
    source = (
        "def plan():\n"
        "    from agents.home_pose import home_pose\n"
        "    return home_pose('cell', 'robot1', None)\n"
    )

    assert _undefined_globals(source, "sample.py") == []
