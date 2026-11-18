"""Backends are packages behind one facade, and a new one plugs in from outside.

* Each built-in backend is a package under ``agents/trajectory/backends/`` that
  declares a ``BackendSpec``; its planner module is imported only when used.
* A backend from another repository registers by entry point or
  ``TRAJECTORY_BACKEND_PLUGINS`` and is selectable by name. Without a spec it is
  non-conforming (stamped); a name colliding with a built-in is refused.
* The coordination layer plans the return home through the facade, which uses
  the selected backend's home planner or the default one.
* The old module paths are the SAME modules, so nothing that imports or patches
  them changed behaviour.
* Nothing outside ``backends/`` reaches into a backend's planner module.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory import backends  # noqa: E402
from agents.trajectory.backends.contract import BackendSpec, missing_result_keys  # noqa: E402
from agents.trajectory.tools import planner_backend as pb  # noqa: E402


# ── built-ins ────────────────────────────────────────────────────────────────

def test_each_builtin_is_a_package_with_a_spec():
    assert set(backends.BUILTIN) == {"ompl", "rrt", "ompl-compound"}
    for name, spec in backends.BUILTIN.items():
        pkg = spec.entry.split(":")[0].rsplit(".", 1)[0]
        assert pkg.startswith("agents.trajectory.backends."), name
        assert (PROJECT_ROOT / pkg.replace(".", "/") / "planner.py").is_file(), name
    assert backends.BUILTIN["rrt"].conforming is False
    assert backends.BUILTIN["ompl"].home and backends.BUILTIN["ompl"].home_single


def test_the_registry_imports_no_planner():
    code = ("import sys; import agents.trajectory.tools.planner_backend as pb;"
            "pb.available_backends();"
            "bad=[m for m in sys.modules if m.endswith('.planner') and 'backends' in m]"
            "+[m for m in ('numpy','ompl','pinocchio') if m in sys.modules];"
            "print(bad)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


# ── plugins ──────────────────────────────────────────────────────────────────

def test_an_env_plugin_is_discovered_and_non_conforming():
    found = backends.plugins_from_env("robot-ballet=json:dumps, b2=json:loads")
    assert set(found) == {"robot-ballet", "b2"}
    spec = backends.resolve_spec(found["robot-ballet"])
    assert spec.entry == "json:dumps" and spec.conforming is False


def test_an_env_plugin_may_point_at_a_spec(tmp_path, monkeypatch):
    (tmp_path / "ballet_pkg.py").write_text(
        "from agents.trajectory.backends.contract import BackendSpec\n"
        "SPEC = BackendSpec(name='robot-ballet', entry='ballet_pkg:plan')\n"
        "def plan(cell_id, motion_requests, **kw): return {}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = backends.resolve_spec(
        backends.plugins_from_env("robot-ballet=ballet_pkg:SPEC")["robot-ballet"])
    assert spec.conforming is True and spec.entry == "ballet_pkg:plan"


@pytest.mark.parametrize("value", ["noequals", "x=nocolon", "=json:dumps"])
def test_a_malformed_env_plugin_is_refused(value):
    with pytest.raises(ValueError, match="name=module:attr"):
        backends.plugins_from_env(value)


def test_a_plugin_cannot_take_a_builtin_name(monkeypatch):
    monkeypatch.setenv(backends.PLUGIN_ENV, "ompl=json:dumps")
    with pytest.raises(ValueError, match="collides"):
        backends.discover()


def test_entry_point_plugins_are_discovered(monkeypatch):
    import importlib.metadata as md

    class EP:
        name, value = "Robot-Ballet", "json:dumps"

    monkeypatch.setattr(md, "entry_points",
                        lambda group=None: [EP()] if group == backends.ENTRY_POINT_GROUP else [])
    monkeypatch.delenv(backends.PLUGIN_ENV, raising=False)
    assert "robot-ballet" in backends.discover()


@pytest.fixture
def registry(monkeypatch):
    """The facade's registry, restored after the test."""
    monkeypatch.setattr(pb, "_SPECS", dict(pb._SPECS))
    monkeypatch.setattr(pb, "_BACKENDS", dict(pb._BACKENDS))
    monkeypatch.setattr(pb, "VALID_BACKENDS", tuple(pb.VALID_BACKENDS))
    return pb


def _fake_module(tmp_path, monkeypatch, body):
    (tmp_path / "fake_backend_mod.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fake_backend_mod", None)


def test_a_registered_backend_is_selected_and_dispatched(registry, tmp_path, monkeypatch):
    _fake_module(tmp_path, monkeypatch,
                 "def plan(cell_id, reqs, approach_offset_m=0.1, velocity_scale=None, **kw):\n"
                 "    return {'success': True, 'trajectories': {}, 'failures': [],\n"
                 "            'planning_time_s': 0.0, 'seen': [r['robot_ns'] for r in reqs]}\n")
    registry.register_backend(BackendSpec(name="robot-ballet", entry="fake_backend_mod:plan"))
    assert "robot-ballet" in registry.VALID_BACKENDS
    assert registry.normalize_backend("ROBOT-BALLET") == "robot-ballet"
    token = registry.set_planning_backend("robot-ballet")
    try:
        res = registry.dispatch_plan("c", [{"robot_ns": "robot1"}])
    finally:
        registry.reset_planning_backend(token)
    assert res["backend_used"] == "robot-ballet" and res["seen"] == ["robot1"]
    assert "non_conforming_backend" not in res and not missing_result_keys(res)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_backend(BackendSpec(name="robot-ballet", entry="x:y"))


def test_a_non_conforming_plugin_is_stamped(registry, tmp_path, monkeypatch):
    _fake_module(tmp_path, monkeypatch,
                 "def plan(cell_id, reqs, **kw):\n"
                 "    return {'success': False, 'trajectories': {}, 'failures': [],\n"
                 "            'planning_time_s': 0.0}\n")
    registry.register_backend(BackendSpec(name="quick", entry="fake_backend_mod:plan",
                                          conforming=False))
    token = registry.set_planning_backend("quick")
    try:
        res = registry.dispatch_plan("c", [])
    finally:
        registry.reset_planning_backend(token)
    assert res["non_conforming_backend"] is True


# ── return home through the facade ───────────────────────────────────────────

def test_home_uses_the_backends_own_planner_or_the_default(registry, tmp_path, monkeypatch):
    _fake_module(tmp_path, monkeypatch,
                 "def plan(*a, **k): return {}\n"
                 "def home(cell_id, starts, **kw): return {'by': 'ballet', 'arms': sorted(starts)}\n"
                 "def default_home(cell_id, starts, **kw): return {'by': 'default'}\n")
    registry.register_backend(BackendSpec(name="ballet", entry="fake_backend_mod:plan",
                                          home="fake_backend_mod:home"))
    registry.register_backend(BackendSpec(name="nohome", entry="fake_backend_mod:plan"))
    monkeypatch.setitem(registry._SPECS, "ompl", BackendSpec(
        name="ompl", entry="fake_backend_mod:plan", home="fake_backend_mod:default_home",
        home_single="fake_backend_mod:default_home"))
    for name, expected in (("ballet", "ballet"), ("nohome", "default"), ("rrt", "default")):
        token = registry.set_planning_backend(name)
        try:
            assert registry.plan_home_trajectories("c", {"robot1": [0] * 6})["by"] == expected
        finally:
            registry.reset_planning_backend(token)


def test_home_config_is_the_shared_home_pose():
    from agents.home_pose import HOME_JOINTS_FALLBACK
    assert pb.home_config("no_such_cell", "robot1", HOME_JOINTS_FALLBACK) == \
        pytest.approx(list(HOME_JOINTS_FALLBACK))


# ── the old paths are the same modules ───────────────────────────────────────

@pytest.mark.parametrize("old, new", [
    ("agents.trajectory.tools.ompl_planner", "agents.trajectory.backends.ompl.planner"),
    ("agents.trajectory.tools.ompl_compound_planner",
     "agents.trajectory.backends.ompl_compound.planner"),
    ("agents.trajectory.tools.rrt_planner", "agents.trajectory.backends.rrt.planner"),
])
def test_an_old_import_path_is_the_same_module(old, new):
    pytest.importorskip("numpy")
    import importlib
    assert importlib.import_module(old) is importlib.import_module(new)


# ── the boundary ─────────────────────────────────────────────────────────────

def test_nothing_outside_backends_imports_a_backend_planner():
    """Coordination, the agent, the MCP interface and the cell reach a backend
    only through tools/planner_backend.py (the aliases at the old paths aside)."""
    allowed = {Path("agents/trajectory/tools/planner_backend.py"),
               Path("agents/trajectory/tools/ompl_planner.py"),
               Path("agents/trajectory/tools/ompl_compound_planner.py"),
               Path("agents/trajectory/tools/rrt_planner.py")}
    offenders = []
    for root in ("agents", "utils", "scripts", "dashboard", "controller", "docker"):
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            if rel in allowed or Path("agents/trajectory/backends") in rel.parents:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                mods = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                for m in mods:
                    if (m.startswith("agents.trajectory.backends.") and m.endswith(".planner")) \
                            or m.split(".")[-1] in ("ompl_planner", "rrt_planner",
                                                    "ompl_compound_planner"):
                        offenders.append(f"{rel}:{node.lineno} {m}")
    assert not offenders, offenders
