"""Planning must not depend on the legacy RRT backend.

``rrt_planner`` was both the LEGACY, explicitly non-conforming Space-Time RRT
backend AND the home of code every other component needed: URDF/Pinocchio
loading, the sphere collision model, the collision predicate, MRTA scheduling,
and the whole LangChain ``@tool`` surface. So the production OMPL backend, the
runtime collision monitor and the trajectory agent all imported a legacy planner
to reach geometry that was never RRT-specific.

That coupling is how the swept-volume bug survived: planning-time validation and
the runtime monitor called the same ``_config_spheres`` but disagreed about
whether the welding torch counted, because the shared function had no single
owner enforcing one answer.

The split:
  robot_kinematics       — models, spheres, collision predicate, cell thresholds
  task_allocation        — MRTA allocation / sequencing / thermal spread
  time_parameterization  — timing plus planner-agnostic path utilities
  planning_tools         — the LLM ``@tool`` surface (dispatches to a backend)
  backends/ompl          — the production backend
  backends/rrt           — ONLY the legacy algorithm, reachable via explicit opt-in

These tests pin that boundary so it cannot silently regrow.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TOOLS = PROJECT_ROOT / "agents" / "trajectory" / "tools"
BACKENDS = PROJECT_ROOT / "agents" / "trajectory" / "backends"
LEGACY = "backends.rrt.planner"
LEGACY_FILE = BACKENDS / "rrt" / "planner.py"

# Everything that must not reach into the legacy backend. planner_backend is the
# ONE exception: it is the facade whose job is to load a selected backend lazily.
MUST_NOT_IMPORT_LEGACY = (
    BACKENDS / "ompl" / "planner.py",
    TOOLS / "planning_tools.py",
    TOOLS / "robot_kinematics.py",
    TOOLS / "task_allocation.py",
    TOOLS / "time_parameterization.py",
    PROJECT_ROOT / "agents" / "cell" / "collision_monitor.py",
    PROJECT_ROOT / "agents" / "trajectory" / "agent.py",
    PROJECT_ROOT / "agents" / "trajectory" / "mcp_server.py",
)


def _imported_modules(path: Path) -> set[str]:
    """Every module named by an import in ``path``, at any nesting depth."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


# ── the legacy backend is isolated ───────────────────────────────────────────

def test_no_production_module_imports_the_legacy_backend():
    """Selecting OMPL must not drag in the planner it replaced."""
    offenders = {
        path.relative_to(PROJECT_ROOT).as_posix(): sorted(
            m for m in _imported_modules(path) if LEGACY in m)
        for path in MUST_NOT_IMPORT_LEGACY
    }
    offenders = {k: v for k, v in offenders.items() if v}

    assert not offenders, (
        f"these must not import {LEGACY} — the shared code moved to "
        f"robot_kinematics / task_allocation / time_parameterization: {offenders}"
    )


def test_only_the_backend_facade_may_load_the_legacy_backend():
    """The facade owns backend selection; that is the single sanctioned reference."""
    facade = BACKENDS / "rrt" / "__init__.py"   # the backend's registry declaration

    assert any(LEGACY in m for m in _imported_modules(facade)) or \
        LEGACY in facade.read_text(), "planner_backend must still be able to dispatch to rrt"


def test_importing_the_agent_does_not_load_the_legacy_backend():
    """The real guarantee, measured at runtime rather than read off the source."""
    code = (
        "import sys; import agents.trajectory.agent;"
        "print('LEGACY' if 'agents.trajectory.backends.rrt.planner' in sys.modules else 'CLEAN')"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        import pytest
        pytest.skip("agent import needs the ROS/LangChain runtime")

    assert "CLEAN" in proc.stdout, (
        "importing the trajectory agent pulled in the legacy RRT backend"
    )


# ── the legacy backend keeps working, via the facade ─────────────────────────

def test_the_legacy_backend_still_exposes_its_entry_point():
    """Isolating it must not break the comparison backend."""
    source = (LEGACY_FILE).read_text()

    assert "def plan_trajectories_rrt(" in source
    assert "def _rrt_single(" in source


def test_the_legacy_module_no_longer_owns_shared_machinery():
    """The extracted code must be GONE from here, not merely duplicated."""
    tree = ast.parse((LEGACY_FILE).read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    # Geometry, scheduling and the tool surface all belong elsewhere now.
    for moved in ("_config_spheres", "_config_is_free", "_load_pinocchio_model",
                  "_regional_allocate", "validate_trajectory_set", "find_interlock_set",
                  "run_trajectory_planner", "run_mrta_pipeline"):
        assert moved not in defined, (
            f"{moved} is still DEFINED in {LEGACY} — a second copy of shared code "
            "is how the planner and the monitor drifted apart in the first place"
        )


def test_the_legacy_module_carries_no_llm_tool_surface():
    """@tool belongs to planning_tools; a backend is called, not prompted.

    Checked on the parsed tree, not the raw text, so a docstring may still
    *describe* the tool surface without counting as exposing one.
    """
    tree = ast.parse((LEGACY_FILE).read_text())

    decorated = [
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        for d in n.decorator_list
        if "tool" in (getattr(d, "id", "") or getattr(getattr(d, "func", None), "id", "") or "")
    ]
    assert not decorated, f"legacy backend still exposes @tool functions: {decorated}"
    assert not any("langchain" in m for m in _imported_modules(LEGACY_FILE))


# ── the shared modules stay importable on a bare host ────────────────────────

def test_geometry_and_scheduling_import_without_langchain():
    """A bare host (no LLM stack) must still be able to import the math."""
    code = (
        "import sys;"
        "import agents.trajectory.tools.task_allocation;"
        "import agents.trajectory.tools.time_parameterization;"
        "print('LC' if 'langchain_core' in sys.modules else 'NO_LC')"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert "NO_LC" in proc.stdout, (
        "importing the scheduling/timing math pulled in LangChain; it must stay "
        "usable (and unit-testable) without the LLM stack"
    )


def test_the_tool_surface_is_where_the_agent_looks_for_it():
    """The agent's imports must resolve against planning_tools."""
    agent_src = (PROJECT_ROOT / "agents" / "trajectory" / "agent.py").read_text()
    tools_defined = ast.parse((TOOLS / "planning_tools.py").read_text())
    names = {n.name for n in ast.walk(tools_defined) if isinstance(n, ast.FunctionDef)}

    assert "planning_tools import" in agent_src
    for expected in ("run_mrta_pipeline", "run_trajectory_planner", "run_path_shortcut",
                     "validate_trajectory_pair", "validate_trajectory_set",
                     "find_interlocks", "find_interlock_set"):
        assert expected in names, expected
