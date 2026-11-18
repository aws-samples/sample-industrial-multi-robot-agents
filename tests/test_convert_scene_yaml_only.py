"""The converter references a scene bundle's assets; it does not copy them.

A scene bundle already contains the arms' dressed URDF/SRDF, their meshes and the
fixture CAD, and ``scene.yaml`` already says where each of those is. Copying them
into the cell would give the cell a SECOND answer to "what shape is this arm", and
the two answers drift. So the converter writes ``cell.yaml`` — the manifest that
names each asset in place — into its own workspace, and nothing else. These tests
pin that:

  * a bundle with no ``scene.usda`` converts from ``scene.yaml`` alone;
  * NOTHING is added to the cell root: everything generated — ``config/`` and
    the manifest included — lands under ``agentic_planner/``;
  * no ``urdf/``, ``srdf/``, ``scene/`` or ``robot_models/`` copy appears;
  * re-converting overwrites in place — no archived ``<cell>_1``, no manifest;
  * ``cell.yaml`` is shaped so the line-based scanners outside this repo read it;
  * fixture obstacles are voxelised from the fixture's own OBJ meshes, at the
    world position the URDF origins and the scene pose put them.

The converter needs only PyYAML, so this runs on the host — no ROS, no pinocchio.
"""
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONVERTER = REPO / "scripts" / "convert_isaac_scene.py"

sys.path.insert(0, str(REPO / "scripts"))
from convert_isaac_scene import (  # noqa: E402
    _cell_relative, _compose, _rpy_matrix, _yaw_matrix, aabb_of_triangles,
    find_srdf_beside, read_obj_triangles, resolve_scene_asset, scene_yaw,
    urdf_geometry_triangles, voxelise_triangles)

sys.path.insert(0, str(REPO))
from utils import cell_layout  # noqa: E402


TCP = "-0.9092 0.0005 -0.0056"

# Where the converter's config goes. Spelled through cell_layout rather than as a
# literal so this suite cannot disagree with the module that answers for readers.
AGENT_CONFIG = f"{cell_layout.AGENT_ROOT_NAME}/config"


def _cfg(cell: Path, name: str) -> Path:
    return cell / AGENT_CONFIG / name


def _manifest(cell: Path) -> Path:
    """The manifest as the converter writes it — inside the workspace."""
    return cell / cell_layout.AGENT_ROOT_NAME / cell_layout.CELL_MANIFEST


ARM_URDF = """\
<?xml version="1.0"?>
<robot name="arm__tool">
  <link name="World"/>
  <link name="base_link"/>
  <link name="link_1"/>
  <link name="link_2"/>
  <joint name="root_joint" type="fixed">
    <parent link="World"/><child link="base_link"/>
  </joint>
  <joint name="joint_a1" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <limit lower="-3.1" upper="3.1" velocity="2.5" effort="0"/>
  </joint>
  <joint name="joint_a2" type="revolute">
    <parent link="link_1"/><child link="link_2"/>
    <limit lower="-2.0" upper="0.6" velocity="1.5" effort="0"/>
  </joint>
</robot>
"""

# A 0.25 m cube at the mesh origin. Written with a mix of face spellings on
# purpose: OBJ in the wild carries v/vt/vn triples, and a reader that only
# handles bare indices silently produces an empty mesh — which reads as "this
# fixture has no geometry" rather than as a parse error.
CUBE_OBJ = """\
# a cube, 0..0.25 on every axis
v 0 0 0
v 0.25 0 0
v 0.25 0.25 0
v 0 0.25 0
v 0 0 0.25
v 0.25 0 0.25
v 0.25 0.25 0.25
v 0 0.25 0.25
f 1//1 2//1 3//1 4//1
f 5/1/2 6/2/2 7/3/2 8/4/2
f 1 2 6 5
f -5 -4 -1 -8
f 2 3 7 6
f 1 4 8 5
"""

FIXTURE_URDF = """\
<?xml version="1.0"?>
<robot name="">
  <link name="World">
    <visual>
      <origin xyz="0. 0. 0." rpy="0. 0. 0."/>
      <geometry><mesh filename="./meshes/cube.obj" scale="1. 1. 1."/></geometry>
    </visual>
    <collision>
      <origin xyz="0. 0. 0.5" rpy="0. 0. 0."/>
      <geometry><mesh filename="./meshes/cube.obj" scale="1. 1. 1."/></geometry>
    </collision>
  </link>
</robot>
"""


def _bundle(tmp_path: Path, *, fixture: bool = True) -> Path:
    """A minimal scene bundle handed over as a cell, with no ``scene.usda``.

    The arm's ``dressed_urdf`` is written the way a real export writes it —
    relative to the scene folder as it sat in the authoring workspace — so it
    dangles here and must be found by filename under the cell. The fixture's path
    is correct as written. Both forms occur in the same real bundle.
    """
    cell = tmp_path / "cells" / "tiny_cell"
    (cell / "robots" / "arm").mkdir(parents=True)
    (cell / "robots" / "arm" / "arm.urdf").write_text(ARM_URDF)
    (cell / "robots" / "arm" / "arm.srdf").write_text(
        '<robot name="arm__tool"/>\n')

    scene = {
        "name": "tiny_cell",
        "robots": [{
            "robot_id": "arm",
            "dressed_urdf": "../urdf-dressed-robots/arm/arm.urdf",
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"a": 90.0, "b": 0.0, "c": 0.0},
            "home_pose_deg": [0, -90],
        }],
    }
    if fixture:
        fx = cell / "fixtures" / "table"
        (fx / "meshes").mkdir(parents=True)
        (fx / "table.urdf").write_text(FIXTURE_URDF)
        (fx / "meshes" / "cube.obj").write_text(CUBE_OBJ)
        scene["fixtures"] = [{
            "id": "table",
            "urdf": "fixtures/table/table.urdf",
            "position": {"x": 5.0, "y": 0.0, "z": 0.0},
            "orientation": {"a": 0.0, "b": 0.0, "c": 0.0},
        }]

    (cell / "scene.yaml").write_text(yaml.safe_dump(scene, sort_keys=False))
    return cell


def _convert(cell: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the converter on a bundle. No --scene: there is no USD to give it."""
    proc = subprocess.run(
        [sys.executable, str(CONVERTER),
         "--scene-yaml", str(cell / "scene.yaml"),
         "--out-dir", str(cell),
         "--tcp-offset", TCP, *extra],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


@pytest.fixture
def converted(tmp_path):
    cell = _bundle(tmp_path)
    _convert(cell)
    return cell


# ── unit: OBJ reading ────────────────────────────────────────────────────────

class TestReadObjTriangles:
    def test_it_reads_every_face_spelling_as_triangles(self, tmp_path):
        obj = tmp_path / "cube.obj"
        obj.write_text(CUBE_OBJ)
        tris = list(read_obj_triangles(obj))
        # Six quads, fan-triangulated: two triangles each.
        assert len(tris) == 12
        assert all(len(t) == 3 for t in tris)

    def test_a_negative_index_counts_back_from_the_end(self, tmp_path):
        obj = tmp_path / "neg.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
        assert list(read_obj_triangles(obj)) == [
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]

    def test_a_polygon_is_fan_triangulated(self, tmp_path):
        obj = tmp_path / "pent.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 2 1 0\nv 1 2 0\nv 0 1 0\n"
                       "f 1 2 3 4 5\n")
        assert len(list(read_obj_triangles(obj))) == 3

    def test_a_missing_file_yields_nothing_rather_than_raising(self, tmp_path):
        # One unreadable mesh out of 900 must not abandon the other 899; the
        # converter warns and carries on.
        assert list(read_obj_triangles(tmp_path / "absent.obj")) == []


# ── unit: fixture geometry placement ─────────────────────────────────────────

class TestFixtureGeometryPlacement:
    def _cell(self, tmp_path):
        fx = tmp_path / "table"
        (fx / "meshes").mkdir(parents=True)
        (fx / "table.urdf").write_text(FIXTURE_URDF)
        (fx / "meshes" / "cube.obj").write_text(CUBE_OBJ)
        return fx / "table.urdf"

    def test_collision_geometry_wins_over_visual(self, tmp_path):
        """The <collision> entry is the geometry authored for collision.

        Here the two entries differ only in z (0 vs 0.5), so the resulting bounds
        say which one was used — and using both would double the geometry.
        """
        urdf = self._cell(tmp_path)
        bb = aabb_of_triangles(urdf_geometry_triangles(
            urdf, (_yaw_matrix(0.0), (0.0, 0.0, 0.0))))
        assert bb["min"] == pytest.approx((0.0, 0.0, 0.5))
        assert bb["max"] == pytest.approx((0.25, 0.25, 0.75))

    def test_the_entry_origin_and_the_scene_pose_are_both_applied(self, tmp_path):
        """A fixture's 900 mesh entries share one link and differ only by origin.

        Ignoring the per-entry <origin> collapses the whole assembly onto itself;
        ignoring the scene pose leaves it at the world origin. Both are silent.
        """
        urdf = self._cell(tmp_path)
        bb = aabb_of_triangles(urdf_geometry_triangles(
            urdf, (_yaw_matrix(0.0), (5.0, -1.0, 0.0))))
        assert bb["min"] == pytest.approx((5.0, -1.0, 0.5))
        assert bb["max"] == pytest.approx((5.25, -0.75, 0.75))

    def test_the_fixture_yaw_rotates_the_geometry(self, tmp_path):
        urdf = self._cell(tmp_path)
        bb = aabb_of_triangles(urdf_geometry_triangles(
            urdf, (_yaw_matrix(math.pi / 2), (0.0, 0.0, 0.0))))
        # +90° about Z: x spans 0..0.25 -> y, y spans 0..0.25 -> -x.
        assert bb["min"] == pytest.approx((-0.25, 0.0, 0.5), abs=1e-9)
        assert bb["max"] == pytest.approx((0.0, 0.25, 0.75), abs=1e-9)

    def test_the_voxel_shell_covers_the_geometry(self, tmp_path):
        urdf = self._cell(tmp_path)
        vox = voxelise_triangles(
            urdf_geometry_triangles(urdf, (_yaw_matrix(0.0), (5.0, 0.0, 0.0))),
            0.1)
        assert vox is not None and vox["voxels"]
        # Every voxel index is the floor of a coordinate inside the world bounds.
        for (i, j, k) in vox["voxels"]:
            assert 49 <= i <= 53, (i, j, k)
            assert -1 <= j <= 3, (i, j, k)
            assert 4 <= k <= 8, (i, j, k)


class TestSceneYaw:
    def test_the_a_angle_is_degrees(self):
        """Proved by the shipped manifests, whose values are 90, 164.4, 254.4."""
        assert scene_yaw({"orientation": {"a": 90.0}}, "r") == pytest.approx(
            math.pi / 2)

    def test_a_missing_orientation_is_zero(self):
        assert scene_yaw({}, "r") == 0.0

    def test_a_tilted_base_is_dropped_loudly(self, capsys):
        """robots.yaml holds one yaw; b/c cannot be folded into it silently."""
        assert scene_yaw({"orientation": {"a": 0.0, "c": 12.0}}, "r") == 0.0
        assert "ignored" in capsys.readouterr().err


class TestTransformHelpers:
    def test_rpy_is_the_fixed_axis_convention(self):
        flat = [v for row in _rpy_matrix(0.0, 0.0, math.pi / 2) for v in row]
        expect = [v for row in _yaw_matrix(math.pi / 2) for v in row]
        assert flat == pytest.approx(expect)

    def test_compose_applies_the_second_pose_first(self):
        outer = (_yaw_matrix(math.pi / 2), (1.0, 0.0, 0.0))
        inner = (_yaw_matrix(0.0), (1.0, 0.0, 0.0))
        rot, t = _compose(outer, inner)
        assert t == pytest.approx((1.0, 1.0, 0.0), abs=1e-9)


# ── unit: asset resolution and manifest paths ────────────────────────────────

class TestBasenameFallback:
    def test_a_dangling_path_is_matched_by_filename_under_the_cell(self, tmp_path):
        """The bundle was rearranged on the way out of the workspace.

        The 2t cell keeps the same assets under ``robots/``, so a path written
        against ``../urdf-dressed-robots/`` resolves nowhere. Matching the
        filename is what makes such a bundle convert at all — it is a guess, so
        the converter says so.
        """
        cell = tmp_path / "cell"
        asset = cell / "robots" / "arm" / "arm.urdf"
        asset.parent.mkdir(parents=True)
        asset.write_text("x")
        found, _ = resolve_scene_asset(
            "../urdf-dressed-robots/arm/arm.urdf", cell, None,
            basename_search=(cell,))
        assert found == asset.resolve()

    def test_it_is_the_last_resort_not_the_first_choice(self, tmp_path):
        """A path that resolves as written must never lose to a filename match."""
        cell = tmp_path / "cell"
        (cell / "robots" / "arm").mkdir(parents=True)
        (cell / "robots" / "arm" / "arm.urdf").write_text("wrong")
        right = cell / "as" / "written" / "arm.urdf"
        right.parent.mkdir(parents=True)
        right.write_text("right")
        found, _ = resolve_scene_asset(
            "as/written/arm.urdf", cell, None, basename_search=(cell,))
        assert found is not None and found.read_text() == "right"


class TestManifestPaths:
    def test_an_asset_inside_the_cell_is_recorded_relative(self, tmp_path):
        """Only a relative path means the same file on the host and in a container."""
        cell = tmp_path / "cells" / "c"
        (cell / "robots").mkdir(parents=True)
        asset = cell / "robots" / "a.urdf"
        asset.write_text("x")
        assert _cell_relative(asset, cell, "arm") == "robots/a.urdf"

    def test_an_asset_outside_the_cell_is_absolute_and_warned_about(
            self, tmp_path, capsys):
        cell = tmp_path / "cells" / "c"
        cell.mkdir(parents=True)
        asset = tmp_path / "elsewhere" / "a.urdf"
        asset.parent.mkdir(parents=True)
        asset.write_text("x")
        out = _cell_relative(asset, cell, "arm")
        assert Path(out).is_absolute()
        assert "not portable" in capsys.readouterr().err

    def test_the_sibling_srdf_is_found_beside_the_urdf(self, tmp_path):
        urdf = tmp_path / "a.urdf"
        urdf.write_text("x")
        (tmp_path / "a.srdf").write_text("y")
        assert find_srdf_beside(urdf) == tmp_path / "a.srdf"

    def test_no_srdf_is_reported_as_absent_rather_than_guessed(self, tmp_path):
        urdf = tmp_path / "a.urdf"
        urdf.write_text("x")
        assert find_srdf_beside(urdf) is None


# ── end to end: what a conversion writes, and what it does not ───────────────

class TestSceneYamlOnlyConversion:
    """A bundle with no scene.usda must convert from scene.yaml alone."""

    def test_it_writes_the_cell_manifest_and_the_config(self, converted):
        assert _manifest(converted).is_file()
        assert _cfg(converted, "robots.yaml").is_file()
        assert _cfg(converted, "cyclonedds.xml").is_file()

    def test_it_creates_the_whole_agent_workspace(self, converted):
        root = converted / cell_layout.AGENT_ROOT_NAME
        assert root.is_dir()
        for name in cell_layout.AGENT_DIRS:
            assert (root / name).is_dir(), name

    def test_the_obstacles_go_in_the_agent_workspace(self, converted):
        # They are derived state, re-tuned by whoever plans against the cell —
        # not part of the cell's shared description of itself.
        assert _cfg(converted, "obstacles.yaml").is_file()


class TestNothingLandsAtTheCellRoot:
    """The cell root is the bundle as its author shipped it. We add nothing to it.

    Every file this converter writes — the whole ``config/`` directory and the
    manifest — is computed from the scene, so all of it is derived state that
    belongs beside the plans and the logs. Writing any of it at the root put
    generated files among the author's own and made "which of these did we make?"
    unanswerable by looking at the tree.
    """

    def test_the_converter_creates_no_root_config_directory(self, converted):
        assert not (converted / "config").exists()

    def test_the_manifest_is_not_at_the_root_either(self, converted):
        assert not (converted / cell_layout.CELL_MANIFEST).exists()

    def test_the_converter_adds_no_file_at_all_to_the_root(self, converted):
        """The bundle's own files, unchanged — no cell.yaml, no config."""
        top = {p.name for p in converted.iterdir() if p.is_file()}
        assert top == {"scene.yaml"}

    def test_the_config_the_readers_resolve_is_the_one_written(self, converted):
        """cell_layout answers with the file the converter actually wrote."""
        assert cell_layout.robots_yaml("tiny_cell", converted.parent) == \
            _cfg(converted, "robots.yaml")
        assert cell_layout.cyclonedds_xml("tiny_cell", converted.parent) == \
            _cfg(converted, "cyclonedds.xml")

    def test_the_manifest_the_readers_resolve_is_the_one_written(self, converted):
        assert cell_layout.cell_yaml("tiny_cell", converted.parent) == \
            _manifest(converted)

    @pytest.mark.parametrize("rel,needle", [
        ("config/robots.yaml", "config/robots.yaml"),
        (cell_layout.CELL_MANIFEST, "cell.yaml"),
    ])
    def test_a_stale_root_copy_is_reported_not_silently_shadowed(
            self, tmp_path, rel, needle):
        """Both on disk means the root one is dead config that reads as live."""
        cell = _bundle(tmp_path)
        stale = cell / rel
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("robots: {}\n")

        err = _convert(cell).stderr
        assert needle in err
        assert "delete the old one" in err

    def test_a_stale_root_manifest_does_not_win_the_read(self, tmp_path):
        """The warning is advice; the resolution must be safe without acting on it."""
        cell = _bundle(tmp_path)
        (cell / cell_layout.CELL_MANIFEST).write_text("robots: []\n")
        _convert(cell)

        assert cell_layout.cell_yaml("tiny_cell", cell.parent) == _manifest(cell)
        assert cell_layout.manifest_robot("tiny_cell", "robot1", cell.parent) != {}

    def test_the_task_file_is_not_generated(self, converted):
        # The task describes the work, not the cell; it is authored separately.
        tasks = converted / cell_layout.AGENT_ROOT_NAME / "tasks"
        assert list(tasks.iterdir()) == []


class TestNothingIsCopied:
    def test_no_urdf_srdf_or_mesh_copy_appears(self, converted):
        for name in ("urdf", "srdf", "robot_models", "scene"):
            assert not (converted / name).exists(), name

    def test_the_manifest_points_at_the_bundles_own_files(self, converted):
        manifest = yaml.safe_load(_manifest(converted).read_text())
        entry = manifest["robots"][0]
        assert entry["urdf"] == "robots/arm/arm.urdf"
        assert entry["srdf"] == "robots/arm/arm.srdf"
        assert (converted / entry["urdf"]).read_text() == ARM_URDF

    def test_the_bundle_keeps_exactly_one_copy_of_each_asset(self, converted):
        assert len(list(converted.rglob("arm.urdf"))) == 1
        assert len(list(converted.rglob("cube.obj"))) == 1


class TestReconvertingOverwritesInPlace:
    def test_it_does_not_archive_the_previous_cell(self, tmp_path):
        cell = _bundle(tmp_path)
        _convert(cell)
        _convert(cell)
        siblings = sorted(p.name for p in cell.parent.iterdir())
        assert siblings == ["tiny_cell"]

    def test_it_writes_no_source_manifest(self, tmp_path):
        cell = _bundle(tmp_path)
        _convert(cell)
        assert list(cell.rglob("source_manifest.json")) == []

    def test_a_changed_scene_needs_no_force_flag(self, tmp_path):
        cell = _bundle(tmp_path)
        _convert(cell)
        scene = yaml.safe_load((cell / "scene.yaml").read_text())
        scene["robots"][0]["position"]["x"] = 7.5
        (cell / "scene.yaml").write_text(yaml.safe_dump(scene, sort_keys=False))
        _convert(cell)
        cfg = yaml.safe_load(_cfg(cell, "robots.yaml").read_text())
        assert cfg["robots"]["arms"]["robot1"]["base_position"][0] == pytest.approx(7.5)

    def test_work_under_the_agent_workspace_survives(self, tmp_path):
        """Overwriting derived files must not clear the agents' own directories."""
        cell = _bundle(tmp_path)
        _convert(cell)
        task = cell / cell_layout.AGENT_ROOT_NAME / "tasks" / "job.json"
        task.write_text("{}")
        _convert(cell)
        assert task.is_file()


class TestTheCellManifest:
    def test_it_bridges_the_cell_local_id_to_the_scene_id(self, converted):
        entry = yaml.safe_load(_manifest(converted).read_text())["robots"][0]
        assert entry["id"] == "robot1"
        assert entry["scene_id"] == "arm"

    def test_id_is_the_first_key_of_each_entry(self, converted):
        """Readers outside this repo scan cell.yaml line by line.

        ``trajectory_player.py`` finds ``robots:`` at column 0 and then takes
        ``- id: <value>`` as the first key of each item. Reordering the keys or
        indenting ``robots:`` gives it zero robots, silently.
        """
        lines = _manifest(converted).read_text().splitlines()
        assert "robots:" in lines
        idx = lines.index("robots:")
        items = [l for l in lines[idx + 1:] if l.lstrip().startswith("- ")]
        assert items and all(l.lstrip().startswith("- id: ") for l in items)

    def test_cell_layout_resolves_the_arm_through_it(self, converted, monkeypatch):
        monkeypatch.setenv("CELL_BASE_PATH", str(converted.parent))
        assert cell_layout.robot_urdf("tiny_cell", "robot1") == \
            converted / "robots" / "arm" / "arm.urdf"
        assert cell_layout.robot_srdf("tiny_cell", "robot1") == \
            converted / "robots" / "arm" / "arm.srdf"

    def test_an_arm_also_resolves_by_its_scene_id(self, converted, monkeypatch):
        monkeypatch.setenv("CELL_BASE_PATH", str(converted.parent))
        assert cell_layout.robot_urdf("tiny_cell", "arm") is not None

    def test_its_paths_are_relative_to_the_cell_not_to_itself(self, converted):
        """The manifest sits in the workspace; its targets do not move with it.

        If a path were read as manifest-relative it would resolve to
        ``agentic_planner/robots/arm/arm.urdf`` — nothing — and every arm would
        fall through to the guessed-``urdf/``-glob with no geometry at all. So the
        entries name the bundle's own layout, and the file they name is really
        there when joined to the CELL directory.
        """
        entry = yaml.safe_load(_manifest(converted).read_text())["robots"][0]
        agent_root = converted / cell_layout.AGENT_ROOT_NAME

        assert (converted / entry["urdf"]).is_file()
        assert not (agent_root / entry["urdf"]).exists()
        assert cell_layout.AGENT_ROOT_NAME not in entry["urdf"]

    def test_the_header_says_which_base_the_paths_use(self, converted):
        """A human editing it by hand is the reader most likely to guess wrong."""
        head = _manifest(converted).read_text()
        assert "relative to the CELL" in head


class TestTheToolFrameIsDeclaredNotBaked:
    """No URDF is written, so the TCP has to be declared for load time.

    ``utils.arm_model`` attaches ``end_effector_link`` to ``tip_link`` at
    ``tcp_offset``. Miss any of the three and the planner targets the flange —
    every weld off by the tool length, with no error anywhere.
    """

    def _arm(self, converted):
        cfg = yaml.safe_load(_cfg(converted, "robots.yaml").read_text())
        return cfg["robots"]["arms"]["robot1"]

    def test_it_names_the_frame_to_target(self, converted):
        assert self._arm(converted)["end_effector_link"] == "welding_tcp"

    def test_tip_link_is_the_last_movable_joints_child(self, converted):
        assert self._arm(converted)["tip_link"] == "link_2"

    def test_the_tcp_offset_is_recorded(self, converted):
        assert self._arm(converted)["tcp_offset"] == pytest.approx(
            [-0.9092, 0.0005, -0.0056])

    def test_a_zero_offset_is_warned_about(self, tmp_path):
        cell = _bundle(tmp_path)
        proc = subprocess.run(
            [sys.executable, str(CONVERTER),
             "--scene-yaml", str(cell / "scene.yaml"),
             "--out-dir", str(cell), "--tcp-offset", "0 0 0"],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode == 0, proc.stderr
        assert "flange" in proc.stderr

    def test_arm_model_reads_back_what_was_written(self, converted, monkeypatch):
        monkeypatch.setenv("CELL_BASE_PATH", str(converted.parent))
        from utils import arm_model
        spec = arm_model.tcp_spec("tiny_cell", "robot1")
        assert spec["ee_link"] == "welding_tcp"
        assert spec["tip_link"] == "link_2"
        assert spec["offset"] == pytest.approx((-0.9092, 0.0005, -0.0056))


class TestRobotsYamlFromTheSceneAlone:
    def _arm(self, converted):
        cfg = yaml.safe_load(_cfg(converted, "robots.yaml").read_text())
        return cfg["robots"]["arms"]["robot1"]

    def test_the_base_pose_comes_from_scene_yaml(self, converted):
        arm = self._arm(converted)
        assert arm["base_position"] == pytest.approx([1.0, 2.0, 0.0])
        assert arm["base_orientation_yaw"] == pytest.approx(math.pi / 2)

    def test_the_joint_limits_come_from_the_referenced_urdf(self, converted):
        arm = self._arm(converted)
        assert arm["joint_limits_lower"] == pytest.approx([-3.1, -2.0])
        assert arm["joint_limits_upper"] == pytest.approx([3.1, 0.6])
        assert arm["joint_velocity_limits"] == pytest.approx([2.5, 1.5])

    def test_the_fixed_root_joint_is_not_counted_as_an_axis(self, converted):
        assert len(self._arm(converted)["joint_limits_lower"]) == 2

    def test_the_home_pose_is_carried_over(self, converted):
        assert self._arm(converted)["home_pose_deg"] == [0, -90]


class TestObstaclesFromTheFixtureMeshes:
    def _obstacles(self, cell):
        return yaml.safe_load(
            (cell / cell_layout.AGENT_ROOT_NAME / "config" /
             "obstacles.yaml").read_text())

    def test_the_floor_is_always_there(self, converted):
        floor = [o for o in self._obstacles(converted)["obstacles"]
                 if o["id"] == "floor"]
        assert floor and floor[0]["type"] == "plane"

    def test_the_fixture_becomes_a_voxel_shell_at_its_world_pose(self, converted):
        boxes = [o for o in self._obstacles(converted)["obstacles"]
                 if o["type"] == "box"]
        assert boxes
        lo = [min(o["center"][i] - o["half_extents"][i] for o in boxes)
              for i in range(3)]
        hi = [max(o["center"][i] + o["half_extents"][i] for o in boxes)
              for i in range(3)]
        # The cube is 0.25 at the mesh origin, lifted 0.5 by the <collision>
        # origin and placed at x=5 by the scene. Voxel bounds are the enclosing
        # 0.1 m cells, so they may exceed the geometry by up to one voxel.
        assert lo[0] == pytest.approx(5.0, abs=0.1) and hi[0] >= 5.25
        assert lo[2] == pytest.approx(0.5, abs=0.1) and hi[2] >= 0.75

    def test_the_clearance_policy_is_written(self, converted):
        obs = self._obstacles(converted)
        assert obs["default_collision_margin"] > 0
        # Two link radii: transit motion is fat, the terminal approach stroke is
        # thin. A single radius makes every weld unreachable or every transit
        # unsafe.
        assert obs["robot_link_radius"] > obs["approach_link_radius"]

    def test_aabb_mode_emits_one_solid_box_per_fixture(self, tmp_path):
        cell = _bundle(tmp_path)
        _convert(cell, "--obstacle-mode", "aabb")
        boxes = [o for o in self._obstacles(cell)["obstacles"]
                 if o["type"] == "box"]
        assert len(boxes) == 1
        assert boxes[0]["id"] == "table"

    def test_a_fixture_with_no_readable_geometry_is_a_flagged_placeholder(
            self, tmp_path):
        """A silently absent obstacle is worse than a wrong one."""
        cell = _bundle(tmp_path)
        (cell / "fixtures" / "table" / "meshes" / "cube.obj").unlink()
        proc = _convert(cell)
        boxes = [o for o in self._obstacles(cell)["obstacles"]
                 if o["type"] == "box"]
        assert len(boxes) == 1 and "REVIEW" in boxes[0]["description"]
        assert "placeholder" in proc.stderr

    def test_a_cell_with_no_fixtures_still_gets_a_floor(self, tmp_path):
        cell = _bundle(tmp_path, fixture=False)
        _convert(cell)
        ids = [o["id"] for o in self._obstacles(cell)["obstacles"]]
        assert ids == ["floor"]


class TestUnresolvableInputsFail:
    def test_a_scene_whose_arm_urdf_is_nowhere_is_an_error(self, tmp_path):
        cell = _bundle(tmp_path)
        (cell / "robots" / "arm" / "arm.urdf").unlink()
        proc = subprocess.run(
            [sys.executable, str(CONVERTER),
             "--scene-yaml", str(cell / "scene.yaml"),
             "--out-dir", str(cell), "--tcp-offset", TCP],
            capture_output=True, text=True, cwd=str(REPO))
        # No robot model means no cell: writing a robots.yaml with no limits
        # would produce a planner that plans against nothing.
        assert proc.returncode != 0
        assert "no robot model resolved" in proc.stderr

    def test_a_missing_scene_usda_is_an_error_not_a_silent_downgrade(self, tmp_path):
        cell = _bundle(tmp_path)
        proc = subprocess.run(
            [sys.executable, str(CONVERTER),
             "--scene-yaml", str(cell / "scene.yaml"),
             "--scene", str(cell / "absent.usda"),
             "--out-dir", str(cell), "--tcp-offset", TCP],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode != 0
        assert "not found" in proc.stderr

    def test_a_malformed_tcp_offset_is_rejected(self, tmp_path):
        cell = _bundle(tmp_path)
        proc = subprocess.run(
            [sys.executable, str(CONVERTER),
             "--scene-yaml", str(cell / "scene.yaml"),
             "--out-dir", str(cell), "--tcp-offset", "0 0"],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode != 0
        assert "three numbers" in proc.stderr
