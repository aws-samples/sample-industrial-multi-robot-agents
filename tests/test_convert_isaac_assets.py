"""scene.yaml names its assets by path; the converter has to be able to find them.

The paths in an Isaac export (``dressed_urdf``, each fixture's ``urdf``,
``floor_plan.geometry_file``) are written relative to the scene folder *as it sits
inside the Isaac authoring workspace*, so they read ``../urdf-dressed-robots/…``.
Copy the export out on its own and all of them dangle — in the reference export
shipped with this repo, all three do.

``--isaacsim-root-cell-path`` supplies the workspace root that ".." was walking up
to. These tests pin the half of that which is still true: the paths resolve, and a
miss names every candidate it tried.

What the converter does with a resolved asset changed: it used to copy it into the
cell (``scene/assets/``, ``urdf/meshes/``, a baked ``urdf/<robot_id>/*.urdf``) and
version the old cell aside on a re-convert. It now REFERENCES each asset where the
bundle already put it and records the path in ``cell.yaml``, and a re-convert
overwrites in place. The tests for that behaviour — and for the copying and
versioning it replaced — live in ``test_convert_scene_yaml_only.py``.

The converter needs only PyYAML, so this runs on the host — no ROS, no pinocchio.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONVERTER = REPO / "scripts" / "convert_isaac_scene.py"

sys.path.insert(0, str(REPO / "scripts"))
from convert_isaac_scene import _strip_updirs, resolve_scene_asset  # noqa: E402


# ── unit: path resolution ────────────────────────────────────────────────────

class TestStripUpdirs:
    def test_it_drops_the_leading_updirs(self):
        assert _strip_updirs(Path("../urdf-dressed-robots/a/a.urdf")) == Path(
            "urdf-dressed-robots/a/a.urdf")

    def test_it_drops_several(self):
        assert _strip_updirs(Path("../../floor-plans/f.obj")) == Path(
            "floor-plans/f.obj")

    def test_a_plain_relative_path_is_untouched(self):
        assert _strip_updirs(Path("assets/f.obj")) == Path("assets/f.obj")

    def test_an_all_updirs_path_is_returned_rather_than_emptied(self):
        # Path(*[]) is ".", which would silently resolve to the scene dir and
        # copy the wrong thing. Returning the input keeps the failure visible.
        assert _strip_updirs(Path("../..")) == Path("../..")


class TestResolveSceneAsset:
    def test_the_scene_folder_wins_over_the_root(self, tmp_path):
        """The root must never shadow an asset that is where scene.yaml says.

        When the export still sits in its workspace, the path as written is
        already correct — preferring the root could silently pick up a stale
        copy of the same asset from elsewhere.
        """
        scene = tmp_path / "ws" / "scene-export"
        scene.mkdir(parents=True)
        near = tmp_path / "ws" / "models" / "a.urdf"
        near.parent.mkdir(parents=True)
        near.write_text("near")
        far = tmp_path / "root" / "models" / "a.urdf"
        far.parent.mkdir(parents=True)
        far.write_text("far")

        found, _ = resolve_scene_asset(
            "../models/a.urdf", scene, tmp_path / "root")
        assert found is not None and found.read_text() == "near"

    def test_the_root_resolves_a_path_the_scene_folder_cannot(self, tmp_path):
        """The case the option exists for: the export was copied out alone."""
        scene = tmp_path / "lonely"
        scene.mkdir()
        asset = tmp_path / "root" / "urdf-dressed-robots" / "a" / "a.urdf"
        asset.parent.mkdir(parents=True)
        asset.write_text("x")

        found, tried = resolve_scene_asset(
            "../urdf-dressed-robots/a/a.urdf", scene, tmp_path / "root")
        assert found == asset.resolve(), tried

    def test_a_root_that_is_the_scenes_parent_also_works(self, tmp_path):
        """The root may be either the workspace top or the scene's parent.

        Both are natural readings of "the Isaac root", so the path is tried with
        the ".." intact as well as peeled off.
        """
        scene = tmp_path / "root" / "scene-export"
        scene.mkdir(parents=True)
        asset = tmp_path / "models" / "a.urdf"
        asset.parent.mkdir(parents=True)
        asset.write_text("x")

        found, tried = resolve_scene_asset(
            "../models/a.urdf", scene, tmp_path / "root")
        assert found == asset.resolve(), tried

    def test_an_absolute_path_is_honoured_as_written(self, tmp_path):
        asset = tmp_path / "abs.urdf"
        asset.write_text("x")
        found, _ = resolve_scene_asset(str(asset), tmp_path / "nowhere", None)
        assert found == asset.resolve()

    def test_a_miss_reports_every_candidate_it_tried(self, tmp_path):
        """A bare "not found" against a path the user never typed is unactionable."""
        found, tried = resolve_scene_asset(
            "../models/a.urdf", tmp_path / "scene", tmp_path / "root")
        assert found is None
        assert len(tried) >= 2
        assert any("models/a.urdf" in t for t in tried)

    def test_without_a_root_only_the_scene_folder_is_tried(self, tmp_path):
        found, tried = resolve_scene_asset("../models/a.urdf", tmp_path, None)
        assert found is None
        assert len(tried) == 1
