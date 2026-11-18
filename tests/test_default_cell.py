"""The default cell is one name, declared once, agreed on by every launcher.

Six modules each carried their own ``os.environ.get("CELL_ID", ...)`` default, and
they had already drifted into three different answers: ``welding_cell_2_small``,
``welding_cell_2_small``, and — in the dashboard — ``welding_cell_2r``, a cell that does
not exist on disk at all. A default nobody can satisfy is worse than no default:
the panel resolved paths under a directory that was never there, so every lookup
it made silently found nothing.

The pattern is the same one that split HOME six ways (see
``test_home_pose_single_source.py``): a decision copied per module instead of
imported. These tests hold the invariant — whoever asks, however they ask, the
default cell is one name, it is ``welding_cell_2_small``, and it exists.

Host-only: reads source text and YAML. No ROS, no containers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED = "welding_cell_2_small"


@pytest.fixture
def cd():
    return pytest.importorskip("agents.cell_defaults")


# ── the declared default ─────────────────────────────────────────────────────

def test_the_default_is_the_2x_cell(cd):
    assert cd.DEFAULT_CELL_ID == EXPECTED


def test_the_default_cell_exists_on_disk(cd):
    """A default that names a missing directory is the bug this replaced."""
    cell = PROJECT_ROOT / "cells" / cd.DEFAULT_CELL_ID
    assert cell.is_dir(), f"{cell} does not exist"
    assert (cell / "config" / "robots.yaml").is_file()


def test_the_environment_wins_when_it_says_something(cd, monkeypatch):
    monkeypatch.setenv("CELL_ID", "welding_cell_2_small")
    assert cd.default_cell_id() == "welding_cell_2_small"


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_environment_value_is_not_a_cell_name(cd, monkeypatch, value):
    """An unset var and a var set to nothing must mean the same thing.

    Compose writes ``CELL_ID: ${SIM:-...}``; a caller exporting SIM= empty used to
    hand every path builder an empty cell name and produce ``/cells//plans``.
    """
    monkeypatch.setenv("CELL_ID", value)
    assert cd.default_cell_id() == EXPECTED


def test_the_value_is_read_per_call_not_captured_at_import(cd, monkeypatch):
    """A container whose env is set after import must not be stuck with the default."""
    monkeypatch.setenv("CELL_ID", "cell_a")
    assert cd.default_cell_id() == "cell_a"
    monkeypatch.setenv("CELL_ID", "cell_b")
    assert cd.default_cell_id() == "cell_b"


# ── the fence: nobody carries their own copy ──────────────────────────────────

def test_no_module_declares_its_own_default_cell(cd):
    """``agents/cell_defaults.py`` owns the value; everything else imports it.

    ``docker/launch_viz.py`` is exempt by design: it runs in the RViz image, which
    has no agents package on its path. Its literal is checked below instead.
    """
    allowed = {Path("agents/cell_defaults.py"), Path("docker/launch_viz.py")}
    roots = ("agents", "controller", "docker", "scripts", "utils", "dashboard",
             "cells")
    # A default supplied to a CELL_ID lookup — the shape that drifted.
    pattern = re.compile(r"""CELL_ID["']\s*,\s*["'][^"']""")

    offenders = []
    searched = 0
    for root in roots:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            if rel in allowed:
                continue
            searched += 1
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i}")

    assert searched > 20, "the fence scanned almost nothing — check the roots"
    assert not offenders, (
        "a default cell is hardcoded again in: " + ", ".join(offenders) +
        " — call agents.cell_defaults.default_cell_id() instead"
    )


def test_no_module_names_a_cell_that_does_not_exist():
    """The dashboard's ``welding_cell_2r`` passed every test that existed.

    Nothing checked that a cell named in code was a cell on disk, so a typo lived
    in two call sites. Any cell name spelled in source must resolve to a directory
    (test fixtures excluded — they are allowed to name absent cells deliberately).
    """
    on_disk = {p.name for p in (PROJECT_ROOT / "cells").iterdir() if p.is_dir()}
    roots = ("agents", "controller", "docker", "scripts", "utils", "dashboard",
             "cells")
    named = re.compile(r"welding_cell_\w+")

    offenders = []
    for root in roots:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            for i, line in enumerate(path.read_text().splitlines(), 1):
                for name in named.findall(line):
                    if name not in on_disk:
                        offenders.append(f"{rel}:{i} -> {name}")

    assert not offenders, (
        "these name cells that do not exist under cells/: " + ", ".join(offenders)
    )


# ── the launchers must agree with the module ──────────────────────────────────

def _literal_after(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.MULTILINE)
    assert m, f"pattern not found: {pattern}"
    return m.group(1)


def test_the_rviz_launchers_copy_agrees(cd):
    """The one sanctioned duplicate, kept in step by this assertion."""
    text = (PROJECT_ROOT / "docker" / "launch_viz.py").read_text()
    assert _literal_after(r"""CELL_ID["']\s*,\s*["']([^"']+)["']""",
                          text) == cd.DEFAULT_CELL_ID


def test_the_makefile_default_agrees(cd):
    text = (PROJECT_ROOT / "Makefile").read_text()
    assert _literal_after(r"^SIM\s*\?=\s*(\S+)", text) == cd.DEFAULT_CELL_ID


def test_the_compose_defaults_agree(cd):
    """Every ``${SIM:-...}`` in the base compose, not just the first."""
    text = (PROJECT_ROOT / "docker-compose.yml").read_text()
    found = re.findall(r"\$\{SIM:-([^}]+)\}", text)
    assert found, "no ${SIM:-...} default found in docker-compose.yml"
    assert set(found) == {cd.DEFAULT_CELL_ID}, f"compose disagrees: {sorted(set(found))}"
