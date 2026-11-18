"""A cell's ``config/`` is agent-owned, and one probe decides where it is read.

Every file in a cell's config directory — ``robots.yaml``, ``cyclonedds.xml``,
``operations.yaml``, the Zenoh session config — is written by
``scripts/convert_isaac_scene.py`` from the scene. All of it is derived state, so
it belongs in the agent workspace next to the plans and the obstacle model, not
among the bundle files the cell's author shipped.

Cells written before that move keep a ``config/`` at the cell root, and
``welding_cell_2_small`` in this repo still does, so the fallback is live code rather
than legacy. That leaves exactly one thing that must not break: the DDS config is
read by the CycloneDDS **C runtime** from ``CYCLONEDDS_URI``, baked into each
container at create time. If the URI and the file disagreed, the whole cell would
come up on CycloneDDS's built-in defaults — a working-looking stack with no
per-cell tuning and no error anywhere. So the URI is computed from the same probe
the readers use, and these tests pin that they agree.

Host-only: path algebra plus tmp_path. No ROS, no cells fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.middleware import middleware_env  # noqa: E402
from utils import cell_layout  # noqa: E402

CELL = "some_cell"
AGENT = cell_layout.AGENT_ROOT_NAME


def _cell(tmp_path: Path, *, migrated: bool = True) -> Path:
    cell = tmp_path / CELL
    (cell / AGENT / "config").mkdir(parents=True) if migrated else cell.mkdir()
    return cell


def _write(cell: Path, rel: str, text: str = "x") -> Path:
    p = cell / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# ── where a config file is written ───────────────────────────────────────────

class TestTheWritePath:
    @pytest.mark.parametrize("name", cell_layout.CONFIG_FILES)
    def test_the_whole_directory_goes_to_the_workspace(self, tmp_path, name):
        """One destination for the config directory, not a rule per file."""
        cell = _cell(tmp_path)
        assert cell_layout.agent_config_file(CELL, name, tmp_path) == \
            cell / AGENT / "config" / name

    def test_it_does_not_depend_on_a_root_copy_being_there(self, tmp_path):
        """A stale file must not steer a writer into writing beside it."""
        cell = _cell(tmp_path)
        _write(cell, "config/robots.yaml")
        assert cell_layout.agent_config_file(CELL, "robots.yaml", tmp_path) == \
            cell / AGENT / "config" / "robots.yaml"


# ── where a config file is read from ─────────────────────────────────────────

class TestTheReadPath:
    def test_the_agent_copy_wins(self, tmp_path):
        cell = _cell(tmp_path)
        want = _write(cell, f"{AGENT}/config/robots.yaml")
        _write(cell, "config/robots.yaml")

        assert cell_layout.robots_yaml(CELL, tmp_path) == want

    def test_a_cell_written_before_the_move_still_resolves(self, tmp_path):
        """welding_cell_2_small ships a root config/ — the fallback is not legacy."""
        cell = _cell(tmp_path)
        want = _write(cell, "config/robots.yaml")

        assert cell_layout.robots_yaml(CELL, tmp_path) == want

    @pytest.mark.parametrize("name,fn", [
        ("robots.yaml",          cell_layout.robots_yaml),
        ("operations.yaml",      cell_layout.operations_yaml),
        ("cyclonedds.xml",       cell_layout.cyclonedds_xml),
        ("zenoh_session.json5",  cell_layout.zenoh_session_config),
    ])
    def test_every_config_file_resolves_the_same_way(self, tmp_path, name, fn):
        """One rule for the whole directory: a per-file rule is how they split."""
        cell = _cell(tmp_path)
        want = _write(cell, f"{AGENT}/config/{name}")

        assert fn(CELL, tmp_path) == want

    def test_an_unmigrated_cell_answers_with_its_only_config_dir(self, tmp_path):
        """No agent root means nothing to namespace under — both are <cell>/config."""
        cell = _cell(tmp_path, migrated=False)
        want = _write(cell, "config/robots.yaml")

        assert cell_layout.robots_yaml(CELL, tmp_path) == want
        assert cell_layout.agent_config_file(CELL, "robots.yaml", tmp_path) == want


# ── the manifest, which moved for the same reason ────────────────────────────

class TestTheCellManifest:
    """``cell.yaml`` is generated too, so it moved with ``config/``.

    The one thing that must NOT move with it: the paths recorded inside it stay
    relative to the cell directory. The manifest says where the bundle already put
    a file; it does not relocate the file.
    """

    def test_it_is_written_into_the_workspace(self, tmp_path):
        cell = _cell(tmp_path)
        assert cell_layout.agent_cell_yaml(CELL, tmp_path) == \
            cell / AGENT / "cell.yaml"

    def test_a_root_copy_does_not_steer_the_write(self, tmp_path):
        cell = _cell(tmp_path)
        _write(cell, "cell.yaml")
        assert cell_layout.agent_cell_yaml(CELL, tmp_path) == \
            cell / AGENT / "cell.yaml"

    def test_the_workspace_copy_wins_the_read(self, tmp_path):
        cell = _cell(tmp_path)
        want = _write(cell, f"{AGENT}/cell.yaml")
        _write(cell, "cell.yaml")
        assert cell_layout.cell_yaml(CELL, tmp_path) == want

    def test_a_cell_written_before_the_move_still_resolves(self, tmp_path):
        cell = _cell(tmp_path)
        want = _write(cell, "cell.yaml")
        assert cell_layout.cell_yaml(CELL, tmp_path) == want

    def test_a_cell_with_no_manifest_reports_the_root_path(self, tmp_path):
        """Callers read "not a file" as "no manifest" and fall back to urdf/."""
        cell = _cell(tmp_path)
        assert cell_layout.cell_yaml(CELL, tmp_path) == cell / "cell.yaml"
        assert cell_layout.read_manifest(CELL, tmp_path) == {}

    def test_an_unmigrated_cell_answers_flat_for_both(self, tmp_path):
        cell = _cell(tmp_path, migrated=False)
        want = _write(cell, "cell.yaml")
        assert cell_layout.cell_yaml(CELL, tmp_path) == want
        assert cell_layout.agent_cell_yaml(CELL, tmp_path) == want

    def test_the_manifest_is_read_from_wherever_it_resolved(self, tmp_path):
        cell = _cell(tmp_path)
        _write(cell, f"{AGENT}/cell.yaml", "robots:\n  - id: robot1\n")
        assert cell_layout.manifest_robot(CELL, "robot1", tmp_path)["id"] == "robot1"

    def test_an_asset_path_is_resolved_against_the_cell_not_the_manifest(self, tmp_path):
        """The invariant that let the manifest move at all.

        A workspace-relative reading would look for
        ``agentic_planner/robots/arm/arm.urdf``; the bundle's file is at
        ``robots/arm/arm.urdf`` and must keep resolving there.
        """
        cell = _cell(tmp_path)
        _write(cell, f"{AGENT}/cell.yaml",
               "robots:\n  - id: robot1\n    urdf: robots/arm/arm.urdf\n")
        want = _write(cell, "robots/arm/arm.urdf")

        assert cell_layout.robot_urdf(CELL, "robot1", tmp_path) == want

    def test_a_manifest_in_the_workspace_does_not_namespace_the_assets(self, tmp_path):
        """Same manifest, asset only under the workspace: that must NOT resolve."""
        cell = _cell(tmp_path)
        _write(cell, f"{AGENT}/cell.yaml",
               "robots:\n  - id: robot1\n    urdf: robots/arm/arm.urdf\n")
        _write(cell, f"{AGENT}/robots/arm/arm.urdf")

        # Falls through to the generated-layout glob, which has nothing either.
        assert cell_layout.robot_urdf(CELL, "robot1", tmp_path) is None


# ── the URI handed to peers outside this process ─────────────────────────────

class TestTheConfigUri:
    def test_it_names_the_file_the_readers_resolve(self, tmp_path):
        cell = _cell(tmp_path)
        _write(cell, f"{AGENT}/config/cyclonedds.xml")

        uri = cell_layout.config_uri(CELL, "cyclonedds.xml", tmp_path)
        assert uri == f"file:///cells/{CELL}/{AGENT}/config/cyclonedds.xml"

    def test_it_follows_a_cell_that_kept_its_config_at_the_root(self, tmp_path):
        cell = _cell(tmp_path)
        _write(cell, "config/cyclonedds.xml")

        assert cell_layout.config_uri(CELL, "cyclonedds.xml", tmp_path) == \
            f"file:///cells/{CELL}/config/cyclonedds.xml"

    def test_it_is_a_container_path_even_though_it_probed_the_host(self, tmp_path):
        """The generator resolves ./cells; the container reads /cells."""
        cell = _cell(tmp_path)
        _write(cell, f"{AGENT}/config/cyclonedds.xml")

        uri = cell_layout.config_uri(CELL, "cyclonedds.xml", tmp_path)
        assert str(tmp_path) not in uri
        assert uri.startswith("file:///cells/")

    def test_a_file_that_is_not_there_yields_no_uri(self, tmp_path):
        """A URI that resolves to nothing is ignored in silence by the runtime."""
        _cell(tmp_path)
        assert cell_layout.config_uri(CELL, "cyclonedds.xml", tmp_path) is None

    def test_the_uri_and_the_read_path_name_the_same_file(self, tmp_path):
        """The invariant the whole design exists for."""
        cell = _cell(tmp_path)
        _write(cell, f"{AGENT}/config/cyclonedds.xml")
        _write(cell, "config/cyclonedds.xml")

        uri = cell_layout.config_uri(CELL, "cyclonedds.xml", tmp_path)
        read = cell_layout.cyclonedds_xml(CELL, tmp_path)
        assert uri.endswith(read.relative_to(tmp_path / CELL).as_posix())


# ── what the containers are told ─────────────────────────────────────────────

class TestTheTransportEnv:
    def test_a_computed_uri_is_what_the_containers_receive(self):
        env = middleware_env(
            "cyclonedds", CELL,
            dds_config_uri=f"file:///cells/{CELL}/{AGENT}/config/cyclonedds.xml")
        assert env["CYCLONEDDS_URI"] == \
            f"file:///cells/{CELL}/{AGENT}/config/cyclonedds.xml"

    def test_without_one_it_falls_back_to_the_cell_root(self):
        """middleware.py is imported where no cells tree is mounted; it cannot probe."""
        env = middleware_env("cyclonedds", CELL)
        assert env["CYCLONEDDS_URI"] == f"file:///cells/{CELL}/config/cyclonedds.xml"

    def test_a_zenoh_stack_carries_no_dds_config_however_it_was_computed(self):
        env = middleware_env("zenoh", CELL, router="tcp/b.local:7447",
                             dds_config_uri=f"file:///cells/{CELL}/x.xml")
        assert "CYCLONEDDS_URI" not in env
