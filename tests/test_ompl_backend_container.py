"""Container-based test for the OMPL trajectory backend.

Split architecture (as requested): the **client stays on the host** (this
pytest process — no Pinocchio/OMPL installed here) and the **backend runs in a
container** that carries Pinocchio + the ``ompl`` Python bindings. The host
merely shells into the container (``docker run``) and asserts on the JSON that
``tests/ompl_container_runner.py`` prints.

Build the backend image first (nothing is installed on the host):

    make test-ompl-container
    # or manually:
    docker build -f docker/Dockerfile.ompltest -t multi-robots-agents/ompltest:latest .
    python3 -m pytest tests/test_ompl_backend_container.py -v -s

The test SKIPS (does not fail) when Docker is unavailable or the image has not
been built, so it never blocks a host-only CI run. When the image IS present it
asserts that OMPL is importable in the container and that the backend planned a
collision-free trajectory on welding_cell_2_small (the welding_cell_2_small arms with the
FIXTURE_A fixture), and failed actionably on a target inside that fixture.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("OMPL_TEST_IMAGE", "multi-robots-agents/ompltest:latest")
CELL = os.environ.get("CELL_ID", "welding_cell_2_small")
SENTINEL = "OMPL_RESULT_JSON="

# In-container command: source ROS (so `import pinocchio` resolves) then run the
# runner. The repo + cells are bind-mounted; the runner emits the result JSON.
_CONTAINER_CMD = "source /opt/ros/jazzy/setup.bash && python3 tests/ompl_container_runner.py"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_present() -> bool:
    if not _docker_available():
        return False
    proc = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def _run_backend_in_container(planner: str = "RRTConnect") -> dict:
    """Run the OMPL backend inside the container and return the parsed JSON."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_ROOT}:/workspace/multi_robots_agents:ro",
        "-v", f"{PROJECT_ROOT / 'cells'}:/cells:ro",
        "-e", "CELL_BASE_PATH=/cells",
        "-e", f"CELL_ID={CELL}",
        "-e", f"OMPL_PLANNER={planner}",
        IMAGE,
        _CONTAINER_CMD,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # The runner prints the JSON on a sentinel-prefixed line; ROS/Pinocchio may
    # print unrelated noise around it, so scan every stdout line.
    for line in proc.stdout.splitlines():
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    raise AssertionError(
        "runner did not emit a result line.\n"
        f"--- exit {proc.returncode} ---\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# Run the container once and share the parsed JSON across the assertions below.
@pytest.fixture(scope="module")
def backend_out() -> dict:
    return _run_backend_in_container()


pytestmark = [
    pytest.mark.skipif(not _docker_available(), reason="docker not available on host"),
    pytest.mark.skipif(
        not _image_present(),
        reason=f"image '{IMAGE}' not built — run `make test-ompl-container`",
    ),
]


def test_ompl_importable_in_container(backend_out):
    """OMPL Python bindings must import inside the container image."""
    assert backend_out["ompl_available"] is True, (
        "OMPL bindings not importable in the container "
        f"(import_error={backend_out.get('import_error')!r}). "
        "The 'pip install ompl' layer likely produced no wheel for this platform."
    )


def test_ompl_plans_collision_free_motion(backend_out):
    """The tool plans a collision-free motion for both robots (mechanism works).

    This is also the shape of the retract/staging 'intermediate point' motions
    the LLM issues while recovering from a collision.
    """
    result = backend_out["scenarios"]["free"]
    assert result["backend"] == "ompl"
    assert result["success"] is True, f"planning failed: {result.get('failures')}"

    counts = result["trajectory_point_counts"]
    assert counts.get("robot1", 0) >= 2, counts
    assert counts.get("robot2", 0) >= 2, counts


def test_ompl_reports_actionable_collision_failure_on_enclosed_weld(backend_out):
    """A fixture-enclosed weld must fail with an actionable collision reason.

    This failure is the trigger for the LLM recovery loop (retract / stage above
    and descend / intermediate waypoint / reorient). It is the EXPECTED outcome
    for the direct weld pose under the sphere-collision model — not a tool bug.
    """
    if "weld" not in backend_out["scenarios"]:
        pytest.skip(f"{CELL} has no fixture box for an enclosed target")
    result = backend_out["scenarios"]["weld"]
    assert result["backend"] == "ompl"
    assert result["success"] is False, "enclosed weld should not plan directly"
    failures = result.get("failures") or []
    assert failures, "a failed weld must report a failure the LLM can act on"

    fail0 = failures[0]
    msg = str(fail0.get("error", "")).lower()
    assert "collision" in msg, f"failure must name the collision, got: {fail0}"
    assert fail0.get("reason_code") == "goal_in_collision", fail0

    # The failure must carry the object involved + the world collision point so
    # the LLM can decide a recovery strategy.
    col = fail0.get("collision") or {}
    assert col.get("object_id"), f"collision must name the object, got: {fail0}"
    point = col.get("point")
    assert isinstance(point, list) and len(point) == 3, f"collision must give a world point, got: {fail0}"
    # ...and the box the target was placed in is the one the diagnosis names.
    enclosed = backend_out.get("enclosed_target") or {}
    diag = fail0.get("target_diagnosis") or {}
    assert diag.get("object_id") == enclosed.get("object_id"), (diag, enclosed)


if __name__ == "__main__":
    # Allow a direct run for quick manual checks (bypasses pytest skips).
    import sys

    if not _image_present():
        print(f"image '{IMAGE}' not built — run `make test-ompl-container`")
        sys.exit(1)
    payload = _run_backend_in_container()
    print(json.dumps(payload, indent=2))
    sys.exit(0 if payload.get("scenarios", {}).get("free", {}).get("success") else 2)
