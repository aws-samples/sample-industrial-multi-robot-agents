"""A step-through preview must not be able to move the cell.

The debugging feature is "run this one node and watch it in RViz". Implemented as
real execution it would be a mutation with two nasty edges: the arm jumps from
wherever it is to the node's first waypoint (nodes are not independent — each
starts where the previous one ended), and pressing it mid-plan puts two writers on
one controller. So the preview commands nothing: the waypoints are published as
JSON on ``/<cell>/node_preview``, and the ONLY subscriber in the system is the
viewer, which animates a translucent second copy of the arm.

That guarantee is a property of the wiring, not of any one file, and it is silent
when it breaks — a preview that started dispatching would look like a feature that
finally works. So it gets tests:

  * the topic the panel publishes on is the topic the viewer listens to (two
    independent literals in two images — the drift ``test_novnc_port_agrees.py``
    guards, one topic over),
  * nothing that can drive a robot subscribes to it,
  * the preview route reaches no dispatch path and says so in its answer, and
  * the viewer's preview handling writes only to ghost topics.

Host-only: source-scans two files and imports the viewer (yaml + ``utils`` only).
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PANEL = PROJECT_ROOT / "dashboard" / "web_control_panel.py"
LAUNCHER = PROJECT_ROOT / "docker" / "launch_viz.py"
PANEL_SRC = PANEL.read_text()
LAUNCHER_SRC = LAUNCHER.read_text()

# Everything that can make a robot move, and the packages that own them. The
# viewer and the preview route must appear in neither.
COMMAND_TOPICS = ("dispatch_subtask", "follow_joint_trajectory",
                  "joint_trajectory_controller", "/command", "/estop")


def _launcher():
    spec = importlib.util.spec_from_file_location("launch_viz_preview_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LV = _launcher()


def _function_source(source: str, name: str) -> str:
    """One ``def`` as text, wherever it is nested.

    Located by AST (the viewer's preview handler and ghost timer are closures
    inside ``run_relay_and_rviz``, so a text scan would run to the end of the
    file) but returned as characters, because what these tests assert is that
    certain calls do not appear anywhere in it — a question about what a reviewer
    would read.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    pytest.fail(f"{name} is gone; this test guards nothing")


class TestOneTopicNameInTwoImages:
    """The panel and the viewer never import each other — different containers,
    and the RViz image has no ``agents`` package to share a constant through. So
    the topic is written twice, and a rename on one side is a silent no-op: the
    button reports success, the ghost never moves."""

    def test_the_panel_publishes_on_the_cells_preview_topic(self):
        match = re.search(
            r"preview_pub\s*=\s*self\.create_publisher\(\s*"
            r"String,\s*cell_topic\(\s*[\"']([^\"']+)[\"']\s*\)", PANEL_SRC)
        assert match, "the panel no longer builds preview_pub from cell_topic(...)"
        assert match.group(1) == "/node_preview", \
            f"the panel now previews on '{match.group(1)}'"

    def test_the_viewer_listens_on_the_same_string(self):
        assert LV.PREVIEW_TOPIC == f"/{LV.CELL_ID}/node_preview", (
            f"the viewer subscribes to {LV.PREVIEW_TOPIC}, but the panel publishes "
            f"cell_topic('/node_preview') == /<cell>/node_preview")

    def test_the_panel_scopes_it_to_the_cell_the_way_the_viewer_does(self):
        """``cell_topic`` must still be ``/<CELL_ID><suffix>``; if it grew a
        namespace the two strings would differ while both still 'look right'."""
        assert re.search(r'def cell_topic\(t: str\) -> str:\s*\n\s*'
                         r'return f"/\{CELL_ID\}\{t\}"', PANEL_SRC), \
            "cell_topic changed shape — recheck it against launch_viz's PREVIEW_TOPIC"

    def test_the_viewer_subscribes_rather_than_publishes_there(self):
        """Direction matters: the viewer is a listener on this topic. A publisher
        here would make the viewer a source of motion data for anyone else."""
        assert re.search(r"create_subscription\(\s*String,\s*PREVIEW_TOPIC", LAUNCHER_SRC)
        assert not re.search(r"create_publisher\([^)]*PREVIEW_TOPIC", LAUNCHER_SRC)


class TestNothingThatCanMoveARobotIsListening:
    def test_no_agent_or_controller_subscribes_to_the_preview_topic(self):
        """The whole safety argument in one assertion.

        If a robot agent ever subscribed to ``node_preview``, the preview button
        would become a motion command — from a route that answers
        ``"commanded": false``.
        """
        offenders = []
        for root in ("agents", "controller"):
            for path in sorted((PROJECT_ROOT / root).rglob("*.py")):
                if "node_preview" in path.read_text():
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
        assert not offenders, (
            f"{offenders} reference the preview topic; a preview is only safe "
            f"while the viewer is its sole subscriber")

    def test_the_viewer_cannot_command_anything_at_all(self):
        """It is a viewer: it reads state and draws. No controller topic anywhere
        in it, so even a bug in the preview handler cannot reach a robot."""
        for topic in COMMAND_TOPICS:
            assert topic not in LAUNCHER_SRC, \
                f"docker/launch_viz.py mentions '{topic}' — the viewer must only read"


class TestThePreviewRouteDispatchesNothing:
    ROUTE = _function_source(PANEL_SRC, "api_preview")

    def test_it_publishes_on_the_preview_publisher_only(self):
        published = set(re.findall(r"ros_node\.(\w+)\.publish", self.ROUTE))
        assert published == {"preview_pub"}, (
            f"POST /api/preview publishes via {sorted(published)}; only preview_pub "
            f"goes to the ghost — cmd_pub, speed_pub and estop_pub reach the cell")

    def test_it_calls_no_orchestrator_write(self):
        """``_a2a_get`` fetches the node. ``_a2a_post`` is how the panel starts a
        replay, and a replay is the whole plan running for real."""
        assert "_a2a_post" not in self.ROUTE, \
            "the preview route posts to the orchestrator; that is execution"
        for verb in ("replay", "dispatch", "execute"):
            assert not re.search(rf"\b{verb}\w*\(", self.ROUTE), \
                f"the preview route calls something named '{verb}...'"

    def test_it_reports_that_nothing_was_commanded(self):
        """The answer the panel shows the operator. Without it, a green tick from
        a preview is indistinguishable from a green tick from a run."""
        assert '"commanded": False' in self.ROUTE, \
            "the response no longer states that the cell was not commanded"

    def test_a_missing_ros_node_is_a_failure_and_not_a_silent_ok(self):
        """No ROS node ⇒ no one to draw the ghost. Answering 200 would report a
        preview that never happened."""
        assert re.search(r"if ros_node is None:\s*\n\s*return jsonify\(\{\"success\": False",
                         self.ROUTE), \
            "the preview route no longer refuses when it has no ROS node"


class TestThePreviewOnlyEverMovesTheGhost:
    def test_the_message_handler_publishes_nothing(self):
        """``_preview_cb`` records state; the ghost timer is what publishes. A
        publish inside the callback would send a pose at the sender's rate."""
        body = _function_source(LAUNCHER_SRC, "_preview_cb")
        assert ".publish(" not in body, \
            "_preview_cb publishes; it is meant to only record the active preview"

    def test_the_ghost_timer_publishes_to_the_ghost_publisher_only(self):
        body = _function_source(LAUNCHER_SRC, "make_ghost_timer")
        published = set(re.findall(r"(\w+)\.publish\(", body))
        assert published == {"pub_g"}, \
            f"the ghost timer publishes via {sorted(published)}, not just the ghost's"

    def test_the_ghost_publisher_is_a_viz_topic(self):
        """``/viz/...`` is the viewer's own namespace: the real arms' joint states
        are relayed INTO it, and nothing reads out of it but RViz."""
        match = re.search(
            r"ghost_pubs\[robot_id\] = node\.create_publisher\(\s*"
            r"JointState, f\"([^\"]+)\"", LAUNCHER_SRC)
        assert match, "the ghost publisher moved; recheck what topic it writes to"
        topic = match.group(1)
        assert topic.startswith("/viz/"), f"the ghost publishes on {topic}"
        assert "ghost_id(robot_id)" in topic, \
            f"the ghost publishes on the REAL arm's topic ({topic})"

    def test_an_unknown_robot_is_refused_rather_than_guessed(self):
        """``robot_ns`` arrives from a plan's ports. Falling back to 'the first
        ghost' would animate an arm that has nothing to do with the node."""
        body = _function_source(LAUNCHER_SRC, "_preview_cb")
        assert re.search(r"if rid not in ghost_pubs:", body), \
            "_preview_cb no longer checks the robot it was asked to preview"

    def test_a_zero_or_negative_speed_cannot_freeze_the_ghost(self):
        body = _function_source(LAUNCHER_SRC, "_preview_cb")
        assert "if speed > 0" in body, \
            "_preview_cb no longer clamps speed; 0 would freeze the ghost forever"


@pytest.mark.parametrize("robot_id", ["robot1", "arm_left", "r"])
def test_the_ghost_name_is_the_real_name_plus_the_suffix(robot_id):
    """Both sides derive frame prefixes from it — the viewer builds the ghost URDF
    with ``dict(r, robot_id=ghost_id(...))``, so this string is load-bearing."""
    assert LV.ghost_id(robot_id) == robot_id + LV.GHOST_SUFFIX
    assert LV.ghost_id(robot_id) != robot_id
