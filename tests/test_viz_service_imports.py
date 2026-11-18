"""The rviz-web launcher is live-mounted, so its imports must be mounted too.

``docker/Dockerfile.viz`` bakes in no repo code: it installs rviz2 and the VNC
stack, and ``scripts/generate_compose.py`` mounts ``./docker`` at ``/viz`` so
``launch_viz.py`` can be edited without a rebuild. That is deliberate — but it
means the image has no idea what the launcher imports, and a new
``from utils import ...`` line is a packaging change, not just a code change.

That is exactly how rviz-web crash-looped: the launcher started reading the
cell's layout through ``utils.cell_layout`` (it had to — ``robots.yaml`` and
``obstacles.yaml`` moved into ``agentic_planner/``, so their location is probed,
not spelled out), while the service still mounted only ``./docker``. Nothing
failed at build time; the container came up, printed
``ModuleNotFoundError: No module named 'utils'``, and the supervisor loop
restarted it forever. Every other service imports fine because they run the
app image, which COPYs the repo — rviz-web is the one that does not.

Host-only: reads two source files and calls one pure function. No docker.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LAUNCHER = PROJECT_ROOT / "docker" / "launch_viz.py"


def _generate_compose():
    """Load the generator by path — ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "generate_compose_under_test", PROJECT_ROOT / "scripts" / "generate_compose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _generate_compose()
SERVICE = GEN.rviz_web_service({"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"})


def _mounts() -> dict[str, str]:
    """``{container_target: host_source}`` for the service's binds."""
    out = {}
    for vol in SERVICE["volumes"]:
        host, target = vol.split(":")[:2]
        out[target] = host
    return out


def _import_roots(source: Path) -> set[str]:
    """Top-level module names ``source`` imports, that are repo directories.

    A name is repo-local iff the repo root holds a package/module of that name;
    everything else (``yaml``, stdlib) comes from the image.
    """
    tree = ast.parse(source.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names
            if (PROJECT_ROOT / n).is_dir() or (PROJECT_ROOT / f"{n}.py").is_file()}


def _search_path() -> list[str]:
    """Where the launcher can import from: PYTHONPATH plus its own directory.

    ``/viz`` is on the list because ``python3 /viz/launch_viz.py`` puts the
    script's directory at ``sys.path[0]``.
    """
    pythonpath = SERVICE["environment"].get("PYTHONPATH", "")
    return [p for p in pythonpath.split(":") if p] + ["/viz"]


class TestTheLauncherCanImportWhatItImports:
    def test_it_has_a_repo_import_to_satisfy(self):
        """Guard the guard: if this ever empties, the tests below prove nothing."""
        assert _import_roots(LAUNCHER), \
            "launch_viz.py imports no repo package — did the launcher move?"

    @pytest.mark.parametrize("name", sorted(_import_roots(LAUNCHER)))
    def test_every_repo_import_is_mounted_on_the_search_path(self, name):
        """The assertion the crash-loop failed.

        Parametrised over what the launcher ACTUALLY imports, so adding a second
        repo import to it fails here instead of at container start.
        """
        wanted = {f"{root.rstrip('/')}/{name}" for root in _search_path()}
        assert wanted & set(_mounts()), (
            f"launch_viz.py does `import {name}` but rviz_web_service mounts it "
            f"nowhere on the search path {_search_path()}: {SERVICE['volumes']}")

    def test_the_mount_points_at_the_repo_package(self):
        """Mounted from the real directory, not an empty or renamed one."""
        for target, host in _mounts().items():
            if target.startswith("/pylib/"):
                assert (PROJECT_ROOT / host.lstrip("./")).is_dir(), \
                    f"{host} does not exist, so {target} would mount empty"

    def test_the_launcher_itself_is_reachable(self):
        """The entrypoint runs /viz/entrypoint_rviz.sh and finds its sibling there."""
        assert _mounts().get("/viz") == "./docker"
        assert SERVICE["command"] == ["bash", "/viz/entrypoint_rviz.sh"]


class TestNoMountNestsInsideAnother:
    """Docker cannot create a mountpoint inside a read-only bind.

    ``./utils:/viz/utils:ro`` would be the obvious way to make ``/viz`` a
    complete import root, and it does not work: with ``/viz`` mounted ``:ro``,
    container init fails outright with
    ``make mountpoint "/viz/utils": ... read-only file system``. The service
    never starts — a worse failure than the one being fixed, and one that only
    shows up at runtime. Hence a separate root plus PYTHONPATH.
    """

    def test_no_target_sits_under_a_read_only_target(self):
        ro = [v.split(":")[1] for v in SERVICE["volumes"] if v.endswith(":ro")]
        for inner in ro:
            for outer in ro:
                if inner != outer:
                    assert not inner.startswith(outer.rstrip("/") + "/"), (
                        f"{inner} nests inside the read-only mount {outer}; "
                        f"docker refuses to create the mountpoint")

    def test_the_import_root_is_not_the_launcher_directory(self):
        """Names the reason, so the two-root layout is not 'simplified' back."""
        assert GEN.VIZ_PYLIB != "/viz"
        assert not GEN.VIZ_PYLIB.startswith("/viz/")


class TestTheRepoRootIsNotHandedToTheViewer:
    """Mounting the whole repo would fix the import in one line. It must not.

    The repo root holds ``.env`` — real credentials — and the viewer needs a
    cell layout probe, not the repository. One mount per package it imports.
    """

    def test_no_mount_exposes_the_repo_root(self):
        for host in _mounts().values():
            assert host.rstrip("/") not in (".", "./"), \
                f"rviz-web mounts the repo root ({host}); .env would go with it"
