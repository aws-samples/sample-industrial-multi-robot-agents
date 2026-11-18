"""The trajectory agent's SYSTEM_PROMPT is an f-string; its prose braces must be escaped.

``SYSTEM_PROMPT = f\"\"\"...\"\"\"`` documents tool call shapes in prose, e.g.
``trajectories={robot_ns: traj, ...}``. In an f-string a single brace is CODE, so
that line evaluated ``robot_ns`` at import and killed the whole agent:

    File "agents/trajectory/agent.py", line 386, in <module>
      (`trajectories={robot_ns: traj, ...}`); must be collision-free.
    NameError: name 'robot_ns' is not defined

The file parses cleanly — an f-string brace is a RUNTIME error — so no syntax
check catches it, and the container crash-looped on startup with the traceback
pointing at a docstring.

This walks the real f-string source and asserts every brace is either escaped or
a deliberate interpolation, so prose braces cannot silently become code again.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

AGENT_SOURCE = PROJECT_ROOT / "agents" / "trajectory" / "agent.py"

# The only interpolations the prompt is meant to perform.
INTENTIONAL = {"_skill_catalog()"}

# A brace group that is NOT doubled — i.e. one the f-string will evaluate.
UNESCAPED = re.compile(r"(?<!\{)\{([^{}]*)\}(?!\})")


def _system_prompt_lines() -> list[tuple[int, str]]:
    """The 1-indexed source lines of the SYSTEM_PROMPT f-string literal."""
    lines = AGENT_SOURCE.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("SYSTEM_PROMPT = f\"\"\""))
    end = next(i for i in range(start + 1, len(lines)) if '"""' in lines[i])
    return [(i + 1, lines[i]) for i in range(start, end + 1)]


def test_the_prompt_is_still_an_fstring():
    """If the prefix is ever dropped, the escaping rules below stop applying."""
    assert any(l.startswith("SYSTEM_PROMPT = f\"\"\"") for _, l in _system_prompt_lines()[:1])


def test_every_evaluated_brace_is_intentional():
    """Prose braces must be doubled; only known interpolations may evaluate."""
    offenders = [
        (num, m.group(1))
        for num, line in _system_prompt_lines()
        for m in UNESCAPED.finditer(line)
        if m.group(1) not in INTENTIONAL
    ]

    assert not offenders, (
        "unescaped brace(s) in the SYSTEM_PROMPT f-string — double them ({{...}}) "
        f"or they evaluate as code at import: {offenders}"
    )


def test_the_tool_shape_that_broke_import_stays_escaped():
    """Regression: the exact line that raised NameError at line 386."""
    source = AGENT_SOURCE.read_text()

    assert "trajectories={{robot_ns: traj, ...}}" in source
    assert "trajectories={robot_ns: traj, ...}" not in source.replace("{{", "\x00").replace("}}", "\x01")


def test_importing_the_agent_module_evaluates_the_prompt():
    """The real guard: importing must not raise, and the prose must survive verbatim."""
    from agents.trajectory.agent import SYSTEM_PROMPT

    # Rendered output keeps single braces — proving they were escaped, not evaluated.
    assert "trajectories={robot_ns: traj, ...}" in SYSTEM_PROMPT
    assert '"{key}"' in SYSTEM_PROMPT
    # And the deliberate interpolation actually ran.
    assert "{_skill_catalog()}" not in SYSTEM_PROMPT
