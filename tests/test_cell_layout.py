"""The path owner is now load-bearing for ~40 call sites, so its rules get pinned.

``utils.cell_layout`` exists because those call sites each used to compose
``f"{base}/{cell_id}/plans"`` themselves. Consolidating them means a mistake here
is no longer one wrong path — it is every path at once. Four things therefore have
to hold, and each has a real failure behind it:

1. **The two layers stay separate.** Agent-owned paths move under
   ``agentic_planner/``; cell-root paths never do. Namespacing a mesh URI or the
   DDS config would change what peers *outside this repo* resolve — a DDS
   subscriber reading published ``robot_description``, RViz, the CycloneDDS C
   runtime reading ``CYCLONEDDS_URI`` from an env var baked at container-create
   time. Those are not renames Python can make.
2. **The migration probe is per cell, not per directory.** ``welding_cell_2_small`` is
   kept as a flat read-only archive, so the fallback is live code, not legacy.
3. **A cell cannot end up half-migrated.** One probe answers for every agent
   directory, so ``plans/`` and ``logs/`` cannot land on opposite sides.
4. **Creating the layout flips the probe first.** Otherwise the subdirectories
   created next would resolve flat — the layout builder would scatter the very
   tree it is building.

Host-only: pure path algebra plus tmp_path. No ROS, no cell fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import cell_layout  # noqa: E402

CELL = "some_cell"

# Every accessor that answers with an agent-owned path, and the directory name it
# must land in. Table-driven so a new accessor cannot be added without deciding
# which layer it belongs to.
AGENT_PATHS = [
    ("agent_config_dir", lambda c, b: cell_layout.agent_config_dir(c, b), "config"),
    ("tasks_dir",        lambda c, b: cell_layout.tasks_dir(c, b),        "tasks"),
    ("plans_dir",        lambda c, b: cell_layout.plans_dir(c, b),        "plans"),
    ("logs_dir",         lambda c, b: cell_layout.logs_dir(c, b),         "logs"),
    ("recordings_dir",   lambda c, b: cell_layout.recordings_dir(c, b),   "recordings"),
    ("obstacles_yaml",   lambda c, b: cell_layout.obstacles_yaml(c, b),   "config"),
    ("task_file",        lambda c, b: cell_layout.task_file(c, "t", b),   "tasks"),
    ("plan_file",        lambda c, b: cell_layout.plan_file(c, "bt_1", b), "plans"),
    ("recording_file",   lambda c, b: cell_layout.recording_file(c, "bt_1", b),
                                                                          "recordings"),
    ("trajectory_index", lambda c, b: cell_layout.trajectory_index(c, b), "recordings"),
    ("run_log_file",     lambda c, b: cell_layout.run_log_file(c, "r1", b), "logs"),
]

# Every accessor that answers with a cell-root path. None of these may EVER gain
# the agent-root segment — see the module docstring, point 1.
ROOT_PATHS = [
    ("root_config_dir",       lambda c, b: cell_layout.root_config_dir(c, b)),
    ("robots_yaml",           lambda c, b: cell_layout.robots_yaml(c, b)),
    ("operations_yaml",       lambda c, b: cell_layout.operations_yaml(c, b)),
    ("cyclonedds_xml",        lambda c, b: cell_layout.cyclonedds_xml(c, b)),
    ("zenoh_session_config",  lambda c, b: cell_layout.zenoh_session_config(c, b)),
    ("urdf_dir",              lambda c, b: cell_layout.urdf_dir(c, cell_base=b)),
    ("urdf_dir(ns)",          lambda c, b: cell_layout.urdf_dir(c, "robot1", b)),
    ("srdf_dir",              lambda c, b: cell_layout.srdf_dir(c, cell_base=b)),
    ("mesh_dir",              lambda c, b: cell_layout.mesh_dir(c, b)),
    ("scene_dir",             lambda c, b: cell_layout.scene_dir(c, b)),
]


@pytest.fixture
def flat(tmp_path) -> Path:
    """A cell with no agent root — the pre-split layout, still on disk."""
    (tmp_path / CELL).mkdir()
    return tmp_path


@pytest.fixture
def migrated(tmp_path) -> Path:
    """A cell with the agent root present, which is the whole migration probe."""
    (tmp_path / CELL / cell_layout.AGENT_ROOT_NAME).mkdir(parents=True)
    return tmp_path


# ── the probe ────────────────────────────────────────────────────────────────

class TestTheMigrationProbe:
    def test_a_cell_is_migrated_exactly_when_the_agent_root_exists(self, tmp_path):
        """The directory IS the answer: no marker file to drift from the tree."""
        (tmp_path / CELL).mkdir()
        assert cell_layout.is_migrated(CELL, tmp_path) is False

        (tmp_path / CELL / cell_layout.AGENT_ROOT_NAME).mkdir()
        assert cell_layout.is_migrated(CELL, tmp_path) is True

    def test_a_file_where_the_agent_root_should_be_is_not_a_migration(self, tmp_path):
        """Fail closed: something that is not a directory cannot hold plans."""
        (tmp_path / CELL).mkdir()
        (tmp_path / CELL / cell_layout.AGENT_ROOT_NAME).write_text("not a dir")

        assert cell_layout.is_migrated(CELL, tmp_path) is False

    def test_a_cell_that_does_not_exist_resolves_flat(self, tmp_path):
        """Paths are answerable before the cell is created — the converter asks
        for them in order to create it."""
        assert cell_layout.is_migrated("never_made", tmp_path) is False
        assert cell_layout.plans_dir("never_made", tmp_path) == \
            tmp_path / "never_made" / "plans"


# ── the two layers ───────────────────────────────────────────────────────────

class TestAgentOwnedPaths:
    @pytest.mark.parametrize("label,fn,leaf", AGENT_PATHS,
                             ids=[p[0] for p in AGENT_PATHS])
    def test_they_move_under_the_agent_root_on_a_migrated_cell(
            self, migrated, label, fn, leaf):
        p = fn(CELL, migrated)
        rel = p.relative_to(migrated / CELL)

        assert rel.parts[0] == cell_layout.AGENT_ROOT_NAME, f"{label}: {rel}"
        assert leaf in rel.parts, f"{label} landed outside {leaf}/: {rel}"

    @pytest.mark.parametrize("label,fn,leaf", AGENT_PATHS,
                             ids=[p[0] for p in AGENT_PATHS])
    def test_they_stay_flat_on_a_cell_that_was_never_migrated(
            self, flat, label, fn, leaf):
        """welding_cell_2_small is a read-only archive; the fallback is live code."""
        rel = fn(CELL, flat).relative_to(flat / CELL)

        assert rel.parts[0] == leaf, f"{label}: {rel}"


class TestCellRootPaths:
    @pytest.mark.parametrize("label,fn", ROOT_PATHS, ids=[p[0] for p in ROOT_PATHS])
    def test_they_never_gain_the_agent_segment_even_on_a_migrated_cell(
            self, migrated, label, fn):
        """These are resolved by peers this repo does not control."""
        rel = fn(CELL, migrated).relative_to(migrated / CELL)

        assert cell_layout.AGENT_ROOT_NAME not in rel.parts, f"{label}: {rel}"

    @pytest.mark.parametrize("label,fn", ROOT_PATHS, ids=[p[0] for p in ROOT_PATHS])
    def test_the_layout_does_not_change_them_at_all(self, tmp_path, label, fn):
        """Same answer either side of the migration — that is what "shared input"
        means, and it is why converting a cell cannot break a running RViz."""
        (tmp_path / "flat_cell").mkdir()
        (tmp_path / "mig_cell" / cell_layout.AGENT_ROOT_NAME).mkdir(parents=True)

        a = fn("flat_cell", tmp_path).relative_to(tmp_path / "flat_cell")
        b = fn("mig_cell", tmp_path).relative_to(tmp_path / "mig_cell")

        assert a == b, f"{label}: {a} != {b}"

    def test_the_mesh_uri_is_a_container_path_without_the_agent_segment(self):
        """Baked into published robot_description. A host path or an extra segment
        here means every subscriber loses every mesh."""
        uri = cell_layout.mesh_uri_root(CELL)

        assert uri == f"file:///cells/{CELL}/urdf/meshes/"
        assert cell_layout.AGENT_ROOT_NAME not in uri

    def test_the_mesh_uri_ignores_the_host_side_migration_state(self, migrated):
        """It takes a container base, never CELL_BASE_PATH — the two differ, and
        resolving a host path on the wire is exactly the bug."""
        assert cell_layout.mesh_uri_root(CELL) == \
            f"file:///cells/{CELL}/urdf/meshes/"


# ── the invariant that matters most ──────────────────────────────────────────

class TestACellIsNeverHalfMigrated:
    def test_every_agent_directory_answers_from_the_same_probe(self, migrated):
        """One probe, so ``plans/`` cannot land under the agent root while
        ``logs/`` lands beside it — which is what a per-directory probe permits
        the first time someone creates one of them by hand."""
        roots = {
            fn(CELL, migrated).relative_to(migrated / CELL).parts[0]
            for _label, fn, _leaf in AGENT_PATHS
        }

        assert roots == {cell_layout.AGENT_ROOT_NAME}

    def test_creating_one_agent_directory_by_hand_does_not_split_the_cell(
            self, flat):
        """A stray flat ``plans/`` must not pull the other directories flat with
        it, nor be mistaken for a migration."""
        (flat / CELL / "plans").mkdir()

        assert cell_layout.is_migrated(CELL, flat) is False
        roots = {
            fn(CELL, flat).relative_to(flat / CELL).parts[0]
            for _label, fn, leaf in AGENT_PATHS
        }
        assert cell_layout.AGENT_ROOT_NAME not in roots


# ── building the layout ──────────────────────────────────────────────────────

class TestCreateAgentLayout:
    def test_it_creates_the_root_and_every_directory_the_agents_own(self, flat):
        made = cell_layout.create_agent_layout(CELL, flat)

        root = flat / CELL / cell_layout.AGENT_ROOT_NAME
        assert made[0] == root, "the root is created first"
        for name in cell_layout.AGENT_DIRS:
            assert (root / name).is_dir(), name

    def test_the_root_is_created_before_the_subdirectories(self, flat):
        """Not cosmetic: creating the root is what flips is_migrated(), so the
        subdirectories resolve to the new layout instead of being scattered flat.
        Reverse the order and this builder would write the old layout."""
        made = cell_layout.create_agent_layout(CELL, flat)

        root = flat / CELL / cell_layout.AGENT_ROOT_NAME
        assert all(p == root or p.parent == root for p in made), made
        assert not (flat / CELL / "plans").exists(), \
            "a subdirectory landed flat — the root was created too late"

    def test_it_leaves_the_cell_migrated(self, flat):
        assert cell_layout.is_migrated(CELL, flat) is False

        cell_layout.create_agent_layout(CELL, flat)

        assert cell_layout.is_migrated(CELL, flat) is True

    def test_it_is_idempotent_and_keeps_what_is_already_there(self, flat):
        """Conversion runs it on every re-convert, including --force."""
        cell_layout.create_agent_layout(CELL, flat)
        plan = cell_layout.plan_file(CELL, "bt_1", flat)
        plan.write_text('{"id": "bt_1"}')

        again = cell_layout.create_agent_layout(CELL, flat)

        assert plan.read_text() == '{"id": "bt_1"}', "an existing plan was clobbered"
        assert len(again) == 1 + len(cell_layout.AGENT_DIRS)

    def test_it_creates_nothing_at_the_cell_root(self, flat):
        """The cell's shared input is not this function's business."""
        cell_layout.create_agent_layout(CELL, flat)

        assert sorted(p.name for p in (flat / CELL).iterdir()) == \
            [cell_layout.AGENT_ROOT_NAME]


# ── the base of the tree ─────────────────────────────────────────────────────

class TestCellsRoot:
    def test_an_explicit_base_wins_over_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CELL_BASE_PATH", "/from/env")

        assert cell_layout.cells_root(tmp_path) == tmp_path

    def test_it_falls_back_to_the_environment_then_to_the_container_mount(
            self, monkeypatch):
        monkeypatch.setenv("CELL_BASE_PATH", "/from/env")
        assert cell_layout.cells_root() == Path("/from/env")

        monkeypatch.delenv("CELL_BASE_PATH")
        assert cell_layout.cells_root() == Path(cell_layout.DEFAULT_CELL_BASE)

    def test_an_empty_env_var_means_unset_not_the_filesystem_root(
            self, monkeypatch):
        """compose writes ``CELL_BASE_PATH: ${CELL_BASE_PATH}``; unset in the
        shell that is an empty string, which used to compose ``//plans``."""
        monkeypatch.setenv("CELL_BASE_PATH", "")

        assert cell_layout.cells_root() == Path(cell_layout.DEFAULT_CELL_BASE)

    def test_a_trailing_slash_does_not_double_up(self, monkeypatch):
        monkeypatch.setenv("CELL_BASE_PATH", "/cells/")

        assert str(cell_layout.plans_dir(CELL)) == f"/cells/{CELL}/plans"


# ── the relative form ────────────────────────────────────────────────────────

class TestAgentRelative:
    def test_it_carries_the_agent_segment_for_a_migrated_cell(self, migrated):
        assert cell_layout.agent_relative(
            CELL, "plans", cell_base=migrated) == "agentic_planner/plans"

    def test_it_omits_the_segment_for_a_flat_cell(self, flat):
        assert cell_layout.agent_relative(
            CELL, "plans", cell_base=flat) == "plans"

    def test_it_agrees_with_the_absolute_accessor(self, migrated):
        """Two answers to one question is the bug this module exists to remove,
        and this pair is the likeliest place for it: one probes on the host and
        emits a container path, the other resolves on the host."""
        rel = cell_layout.agent_relative(CELL, "plans", cell_base=migrated)
        absolute = cell_layout.plans_dir(CELL, migrated)

        assert absolute == (migrated / CELL / rel)
