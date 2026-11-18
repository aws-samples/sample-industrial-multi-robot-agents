"""The transport is handed to the container at startup, and every topic is the cell's.

Two properties, one requirement: a planner in a plant that runs more than one cell
must not have its live progress read as another cell's.

**The DDS configuration comes from the outside.** Domain, RMW implementation and
the cell's ``cyclonedds.xml`` (or the Zenoh broker override) are written into the
planner's environment by the generator, at start-up, from the same ``ros_env`` every
other service in the stack receives. Nothing about the transport is left to the
image's compiled-in defaults, because a container that quietly falls back to the
default domain joins a graph nobody meant it to join — and looks perfectly healthy
doing it.

**The topic carries the cell name.** ``/<cell_id>/plan_progress``, one per cell. The
dashboard composes the same name from its own ``CELL_ID`` (``cell_topic``), so the
prefix is what keeps two cells' runs apart — and it must be the cell name
*verbatim*: a sanitised or defaulted prefix would still "look right" while the two
sides listened and spoke past each other. A cell id that cannot be a ROS name is
therefore refused rather than rewritten, and a missing one is refused rather than
published on a shared topic. Losing the live view for a badly named cell is an
availability failure; mixing two cells' runs into one feed is a correctness one.

Host-only: stdlib, source text and the generator's own functions. No ROS, no
network.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory import plan_events, plan_progress  # noqa: E402

CELL = "welding_cell_1x"


@pytest.fixture(autouse=True)
def _quiet_publisher(monkeypatch):
    """No publisher, no remembered failure, and no ambient cell identity."""
    monkeypatch.setattr(plan_progress, "_node", None)
    monkeypatch.setattr(plan_progress, "_publishers", {})
    monkeypatch.setattr(plan_progress, "_disabled_reason", None)
    monkeypatch.setattr(plan_progress, "_refused", {})
    monkeypatch.delenv("CELL_ID", raising=False)
    yield


class _Recorder:
    def __init__(self):
        self.messages: list = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))


@pytest.fixture
def wire(monkeypatch):
    """One recorder per topic, so a message on the wrong topic is visible."""
    by_topic: dict = {}

    def _publisher(cell_id):
        return by_topic.setdefault(plan_progress.topic_for(cell_id), _Recorder())

    monkeypatch.setattr(plan_progress, "_ensure_publisher", _publisher)
    if "std_msgs.msg" not in sys.modules:
        module = types.ModuleType("std_msgs")
        module.__path__ = []
        msg = types.ModuleType("std_msgs.msg")

        class String:
            def __init__(self):
                self.data = ""

        msg.String = String
        sys.modules.setdefault("std_msgs", module)
        sys.modules["std_msgs.msg"] = msg
    return by_topic


# ── the cell name is the prefix, verbatim ────────────────────────────────────

def test_the_topic_is_the_cell_name_then_the_suffix_and_nothing_else():
    """The shape the dashboard's ``cell_topic`` builds independently. Anything
    else here — a namespace, a normalised name — and the two sides drift while
    both still look right."""
    assert plan_progress.topic_for(CELL) == f"/{CELL}{plan_progress.TOPIC_SUFFIX}"
    panel = (PROJECT_ROOT / "dashboard" / "web_control_panel.py").read_text()
    assert 'return f"/{CELL_ID}{t}"' in panel, (
        "the panel composes the prefix differently now — recheck topic_for")


def test_two_cells_never_share_a_topic(wire):
    plan_progress.publish("welding_cell_1x", "t1", {"seq": 1, "kind": "k"})
    plan_progress.publish("welding_cell_2_small", "t2", {"seq": 1, "kind": "k"})
    assert sorted(wire) == ["/welding_cell_1x/plan_progress",
                            "/welding_cell_2_small/plan_progress"]
    for topic, recorder in wire.items():
        assert len(recorder.messages) == 1
        assert recorder.messages[0]["cell_id"] in topic


def test_the_payload_names_the_same_cell_the_topic_does(wire):
    plan_progress.publish(CELL, "t1", {"seq": 1, "kind": "k"})
    sent = wire[f"/{CELL}{plan_progress.TOPIC_SUFFIX}"].messages[0]
    assert sent["cell_id"] == CELL, (
        "a subscriber filters on the field as well as the topic")


@pytest.mark.parametrize("missing", ["", "   ", None])
def test_no_cell_means_no_topic_rather_than_a_shared_one(wire, missing):
    """``/plan_progress`` or ``//plan_progress`` is the one topic every cell in the
    plant would collide on — exactly what the prefix exists to prevent."""
    with pytest.raises(ValueError):
        plan_progress.topic_for(missing)
    assert plan_progress.publish(missing, "t1", {"seq": 1}) is False
    assert wire == {}


@pytest.mark.parametrize("hostile", ["weld-cell", "1cell", "a/b", "a b", "..",
                                     "cell$", "welding cell"])
def test_a_cell_id_that_cannot_be_a_ros_name_is_refused_not_rewritten(wire, hostile):
    """Rewriting it would publish on a name the dashboard never subscribes to,
    which is a live view that is silently always empty."""
    with pytest.raises(ValueError):
        plan_progress.topic_for(hostile)
    assert plan_progress.publish(hostile, "t1", {"seq": 1}) is False
    assert wire == {}


def test_the_cell_can_come_from_the_identity_the_container_was_started_with(
        wire, monkeypatch):
    """``CELL_ID`` is what compose passes in. A call site that does not carry the
    cell around is still scoped to the right one."""
    monkeypatch.setenv("CELL_ID", CELL)
    assert plan_progress.publish(None, "t1", {"seq": 1, "kind": "k"}) is True
    assert list(wire) == [f"/{CELL}{plan_progress.TOPIC_SUFFIX}"]
    assert wire[f"/{CELL}{plan_progress.TOPIC_SUFFIX}"].messages[0]["cell_id"] == CELL


def test_an_explicit_cell_beats_the_containers_own(wire, monkeypatch):
    """One planner may be asked to plan for another cell; the run belongs to the
    cell it is planning, not to the container's default."""
    monkeypatch.setenv("CELL_ID", "welding_cell_2_small")
    plan_progress.publish("welding_cell_1x", "t1", {"seq": 1, "kind": "k"})
    assert list(wire) == ["/welding_cell_1x/plan_progress"]


def test_a_refused_cell_still_gets_its_events_recorded(tmp_path, monkeypatch):
    """The durable stream is not scoped by a ROS name, so a cell the topic cannot
    carry must still have a readable run."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("CELL_DATA_MODE", raising=False)
    plan_events._SEQ.clear()
    assert plan_events.emit("weld-cell", "t1", "run_started")["kind"] == "run_started"
    from utils import cell_store
    assert "run_started" in cell_store.read_text("weld-cell", cell_store.KIND_LOGS,
                                                 "planevents_t1.jsonl")


def test_one_unpublishable_cell_does_not_darken_the_others(monkeypatch):
    """A per-cell name problem is per-cell. Latching it globally would take the
    live view down for every cell in the process."""
    monkeypatch.setattr(plan_progress, "_publishers", {})
    assert plan_progress.publish("weld-cell", "t1", {"seq": 1}) is False
    assert plan_progress._disabled_reason is None, (
        "a bad cell name is not a missing middleware")

    recorder = _Recorder()
    monkeypatch.setattr(plan_progress, "_ensure_publisher", lambda c: recorder)
    assert plan_progress.publish(CELL, "t1", {"seq": 1}) is True


def test_the_reason_a_cell_is_refused_is_said_once(monkeypatch, capsys):
    """Once per cell, not once per weld: the planner's log has to stay readable."""
    for _ in range(4):
        plan_progress.publish("weld-cell", "t1", {"seq": 1})
    printed = capsys.readouterr().out
    said = [line for line in printed.splitlines() if "weld-cell" in line]
    assert len(said) == 1, printed


# ── the transport arrives from outside the container ─────────────────────────

def test_the_planner_reports_the_transport_it_was_started_with(monkeypatch):
    """So ``/health`` and a puzzled operator can see which graph this planner
    joined. A domain mismatch is otherwise invisible: both sides are healthy and
    simply never hear each other."""
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    monkeypatch.setenv("CYCLONEDDS_URI",
                       f"file:///cells/{CELL}/config/cyclonedds.xml")
    monkeypatch.setenv("CELL_ID", CELL)
    transport = plan_progress.transport()
    assert transport["ROS_DOMAIN_ID"] == "42"
    assert transport["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert transport["CYCLONEDDS_URI"].endswith("cyclonedds.xml")
    assert plan_progress.describe()["transport"] == transport


def test_transport_reports_absence_rather_than_inventing_a_default(monkeypatch):
    """A default written here would be a second, competing configuration — and
    the symptom (nobody hears the planner) is a long way from the cause."""
    for var in ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI",
                "ZENOH_CONFIG_OVERRIDE", "ZENOH_SESSION_CONFIG_URI"):
        monkeypatch.delenv(var, raising=False)
    assert plan_progress.transport() == {}


def test_only_the_transport_that_is_configured_is_reported(monkeypatch):
    monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")
    monkeypatch.setenv("ZENOH_CONFIG_OVERRIDE",
                       'connect/endpoints=["tcp/broker.plant.local:7447"]')
    transport = plan_progress.transport()
    assert "CYCLONEDDS_URI" not in transport
    assert "broker.plant.local" in transport["ZENOH_CONFIG_OVERRIDE"]


# ── what the generator actually writes into the planner's container ──────────

@pytest.fixture
def gen():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    return pytest.importorskip("generate_compose")


def _planner(gen, middleware="cyclonedds", router=None):
    ros_env = gen._x_ros_env(CELL, middleware, router, None,
                             dds_config_uri=f"file:///cells/{CELL}/config/cyclonedds.xml")
    # The builders return the service body itself, not {name: body}.
    return ros_env, gen.trajectory_agent_service(CELL, "ompl", ros_env, {})


def test_the_planners_container_is_handed_the_whole_transport(gen):
    """Every transport variable in the stack's ros_env reaches the planner. It
    publishes now, so a planner without them is a planner nobody can watch."""
    ros_env, planner = _planner(gen)
    for key in ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI"):
        assert planner["environment"][key] == ros_env[key], key


def test_the_planner_is_told_which_cell_it_belongs_to(gen):
    """The topic prefix and the DDS config path are both derived from it."""
    _, planner = _planner(gen)
    assert planner["environment"]["CELL_ID"] == CELL


def test_the_dds_config_lands_inside_a_mount_the_planner_actually_has(gen):
    """``/cells`` is read-only in this container now. A URI pointing outside the
    mounts would leave CycloneDDS on its compiled-in defaults, silently."""
    _, planner = _planner(gen)
    uri = planner["environment"]["CYCLONEDDS_URI"]
    assert uri.startswith("file:///cells/")
    assert any(volume.split(":")[1] == "/cells"
               for volume in planner["volumes"] if ":" in volume), planner["volumes"]


def test_a_read_only_mount_is_enough_because_the_config_is_only_read(gen):
    _, planner = _planner(gen)
    assert "./cells:/cells:ro" in planner["volumes"]


def test_a_zenoh_stack_hands_the_planner_the_broker_and_no_dds_config(gen):
    _, planner = _planner(gen, "zenoh", "tcp/broker.plant.local:7447")
    env = planner["environment"]
    assert env["RMW_IMPLEMENTATION"] == "rmw_zenoh_cpp"
    assert "broker.plant.local" in env["ZENOH_CONFIG_OVERRIDE"]
    assert "CYCLONEDDS_URI" not in env, (
        "config for a middleware that is not running reads as live and is not")


def test_every_agent_in_the_stack_joins_the_same_graph(gen):
    """One domain and one RMW across the stack, from one ros_env. The planner
    publishing on a domain of its own would be invisible to the dashboard while
    both processes reported themselves healthy."""
    ros_env = gen._x_ros_env(CELL, "cyclonedds", None, None)
    services = [gen.trajectory_agent_service(CELL, "ompl", ros_env, {}),
                gen.orchestrator_service(CELL, ros_env, {}),
                gen.cell_controller_service(CELL, ros_env, {}),
                gen.web_panel_service(CELL, ros_env, {})]
    for service in services:
        env = service["environment"]
        assert env["ROS_DOMAIN_ID"] == ros_env["ROS_DOMAIN_ID"]
        assert env["RMW_IMPLEMENTATION"] == ros_env["RMW_IMPLEMENTATION"]
        assert env["CELL_ID"] == CELL


def test_the_hand_written_compose_file_also_passes_it_to_the_planner():
    """The checked-in file is a fallback, but it is the one an operator reads
    first; the planner's service there must carry the ros-env anchor too."""
    text = (PROJECT_ROOT / "docker-compose.yml").read_text()
    planner = text[text.index("  trajectory-agent:"):]
    planner = planner[:planner.index("\n  # ── Orchestrator")]
    assert "*ros-env" in planner
    assert "CELL_ID:" in planner
