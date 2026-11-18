"""Host unit tests for the BehaviorTree.CPP v4 format layer.

These run on the host with NO ROS, NO Pinocchio, NO OMPL and NO LangChain:
``agents.trajectory.tools.btcpp`` is a pure format module, which is exactly the
property that makes the tree format testable without a cell.

What they pin down is the contract every other consumer relies on:
  * the legacy dialect converts to schema-correct v4 (element names, port names,
    the Parallel defaults, start_delay_s → a Delay decorator),
  * the bulky waypoint arrays are NOT inlined in the XML but referenced as
    ``{key}`` into a side-car blackboard — the reason the format is readable,
  * conversion is idempotent, so no caller has to track which dialect it holds,
  * XML round-trips, so a tree edited in Groot2 can come back to the executor,
  * a malformed document fails loudly instead of yielding a half-tree.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools import btcpp  # noqa: E402


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _traj(n: int = 3, t0: float = 0.0) -> list[dict]:
    return [{"positions": [0.1 * i] * 6, "time_s": t0 + 0.5 * i} for i in range(n)]


def _legacy_tree() -> dict:
    """A tree in the old ad-hoc dialect, exercising every node kind."""
    return {
        "tree_id": "bt_test_legacy",
        "cell_id": "welding_cell_2_small",
        "root": {
            "id": "seq", "type": "SEQUENCE",
            "children": [
                {"id": "par", "type": "PARALLEL", "children": [
                    {"id": "e1", "type": "EXECUTE", "robot_ns": "robot1",
                     "operation_id": "W1", "operation_type": "weld",
                     "precomputed_trajectory": _traj(4),
                     "hold_duration_s": 0.5, "velocity_scale": 1.0,
                     "has_interlock": False},
                    {"id": "e2", "type": "EXECUTE", "robot_ns": "robot2",
                     "operation_id": "W2", "operation_type": "weld",
                     "precomputed_trajectory": _traj(3),
                     "start_delay_s": 1.5,
                     "has_interlock": True,
                     "interlock_target": {"ee_position": [1.0, 0.2, 1.1]}},
                ]},
                {"id": "iw", "type": "INTERLOCK_WAIT", "robot_ns": "robot1",
                 "depends_on_robots": ["robot2"],
                 "depends_on_positions": [{"robot": "robot2",
                                           "position": [1.0, 0.2, 1.1],
                                           "tolerance_m": 0.05}]},
                {"id": "sg", "type": "SAFETY_GATE", "condition": "estop_clear"},
                {"id": "h1", "type": "HOME", "robot_ns": "robot1",
                 "precomputed_trajectory": _traj(2)},
            ],
        },
    }


def _exec_nodes(node: dict) -> list[dict]:
    found = []
    if btcpp.node_kind(node) == "ExecuteTrajectory":
        found.append(node)
    for child in btcpp.children(node):
        found.extend(_exec_nodes(child))
    return found


# ─── Legacy → BT.CPP conversion ───────────────────────────────────────────────

def test_legacy_node_kinds_map_to_btcpp_registration_ids():
    """Every legacy type must resolve to the upstream node ID, not stay as-is."""
    tree = btcpp.to_btcpp(_legacy_tree())
    kinds = set()

    def walk(n):
        kinds.add(btcpp.node_kind(n))
        for c in btcpp.children(n):
            walk(c)

    walk(tree["root"])
    assert kinds == {"Sequence", "Parallel", "Delay", "ExecuteTrajectory",
                     "WaitForRobot", "CheckSafety", "MoveHome"}
    # No legacy spelling survives anywhere in the converted tree.
    assert not (kinds & set(btcpp.LEGACY_TO_BTCPP))


def test_converted_tree_declares_the_v4_envelope():
    tree = btcpp.to_btcpp(_legacy_tree())
    assert tree["BTCPP_format"] == "4"
    assert tree["main_tree_to_execute"] == "MainTree"
    # Identity fields are carried through, not dropped by the conversion.
    assert tree["tree_id"] == "bt_test_legacy"
    assert tree["cell_id"] == "welding_cell_2_small"


def test_parallel_defaults_to_all_children_must_succeed():
    """A synchronised multi-arm step needs every arm to finish and any failure to
    stop the step: success_count=-1, failure_count=1, stated explicitly."""
    tree = btcpp.to_btcpp(_legacy_tree())
    par = next(n for n in btcpp.children(tree["root"])
               if btcpp.node_kind(n) == "Parallel")
    assert btcpp.port(par, "success_count") == -1
    assert btcpp.port(par, "failure_count") == 1


def test_start_delay_becomes_a_delay_decorator_in_milliseconds():
    """BT.CPP already has a node for 'tick the child later'; keeping the delay as
    a leaf field would hide the timing from anything reading the tree."""
    tree = btcpp.to_btcpp(_legacy_tree())
    par = next(n for n in btcpp.children(tree["root"])
               if btcpp.node_kind(n) == "Parallel")
    delay = next(n for n in btcpp.children(par) if btcpp.node_kind(n) == "Delay")

    assert btcpp.port(delay, "delay_msec") == 1500
    kids = btcpp.children(delay)
    assert len(kids) == 1
    assert btcpp.node_kind(kids[0]) == "ExecuteTrajectory"
    assert btcpp.port(kids[0], "robot_ns") == "robot2"
    # The delay moved to the decorator; it is not left duplicated on the leaf.
    assert btcpp.port(kids[0], "start_delay_s") is None


def test_undelayed_leaf_is_not_wrapped():
    tree = btcpp.to_btcpp(_legacy_tree())
    par = next(n for n in btcpp.children(tree["root"])
               if btcpp.node_kind(n) == "Parallel")
    kinds = [btcpp.node_kind(c) for c in btcpp.children(par)]
    assert kinds.count("ExecuteTrajectory") == 1, "robot1 had no delay to wrap"


def test_leaf_ports_are_renamed_to_the_btcpp_names():
    tree = btcpp.to_btcpp(_legacy_tree())
    leaf = next(n for n in _exec_nodes(tree["root"])
                if btcpp.port(n, "operation_id") == "W1")
    ports = leaf["ports"]
    assert "precomputed_trajectory" not in ports
    assert "trajectory" in ports
    assert ports["robot_ns"] == "robot1"
    assert ports["operation_type"] == "weld"
    assert ports["hold_duration_s"] == 0.5


def test_conversion_is_idempotent():
    """Callers must never have to track which dialect they hold."""
    once = btcpp.to_btcpp(_legacy_tree())
    twice = btcpp.to_btcpp(once)
    assert twice["root"] == once["root"]
    assert twice["blackboard"] == once["blackboard"]
    assert twice["xml"] == once["xml"]


# ─── Blackboard ───────────────────────────────────────────────────────────────

def test_waypoint_arrays_live_on_the_blackboard_not_in_the_ports():
    """Inlining hundreds of waypoints would make the XML unreadable, which is the
    whole reason for adopting the format."""
    legacy = _legacy_tree()
    tree = btcpp.to_btcpp(legacy)
    leaf = next(n for n in _exec_nodes(tree["root"])
                if btcpp.port(n, "operation_id") == "W1")

    ref = btcpp.port(leaf, "trajectory")
    assert isinstance(ref, str) and ref.startswith("{") and ref.endswith("}")
    key = ref[1:-1]
    assert key in tree["blackboard"]
    assert tree["blackboard"][key] == _traj(4)


def test_blackboard_keys_are_readable_and_identify_the_node():
    tree = btcpp.to_btcpp(_legacy_tree())
    leaf = next(n for n in _exec_nodes(tree["root"])
                if btcpp.port(n, "operation_id") == "W1")
    key = btcpp.port(leaf, "trajectory")[1:-1]
    assert "robot1" in key and "W1" in key


def test_trajectory_of_follows_the_blackboard_reference():
    tree = btcpp.to_btcpp(_legacy_tree())
    leaf = next(n for n in _exec_nodes(tree["root"])
                if btcpp.port(n, "operation_id") == "W2")
    assert btcpp.trajectory_of(leaf, tree["blackboard"]) == _traj(3)


def test_trajectory_of_reports_a_dangling_reference_as_empty():
    """A missing entry must read as 'no trajectory' — the actionable failure the
    executor already checks for — rather than raising deep in a walker."""
    tree = btcpp.to_btcpp(_legacy_tree())
    leaf = next(iter(_exec_nodes(tree["root"])))
    assert btcpp.trajectory_of(leaf, {}) == []


def test_structured_gating_ports_also_go_to_the_blackboard():
    tree = btcpp.to_btcpp(_legacy_tree())
    wait = next(n for n in btcpp.children(tree["root"])
                if btcpp.node_kind(n) == "WaitForRobot")

    deps = btcpp.resolve(btcpp.port(wait, "depends_on_robots"), tree["blackboard"])
    positions = btcpp.resolve(btcpp.port(wait, "depends_on_positions"),
                              tree["blackboard"])
    assert deps == ["robot2"]
    assert positions[0]["robot"] == "robot2"
    assert positions[0]["tolerance_m"] == 0.05


def test_resolve_passes_plain_values_through():
    assert btcpp.resolve("estop_clear", {"estop_clear": "nope"}) == "estop_clear"
    assert btcpp.resolve(0.5, {}) == 0.5
    assert btcpp.resolve(None, {}) is None


# ─── XML emission ─────────────────────────────────────────────────────────────

def test_xml_matches_the_v4_schema():
    xml = btcpp.to_xml(btcpp.to_btcpp(_legacy_tree()))
    root = ET.fromstring(xml)

    assert root.tag == "root"
    assert root.get("BTCPP_format") == "4"
    assert root.get("main_tree_to_execute") == "MainTree"

    trees = root.findall("BehaviorTree")
    assert len(trees) == 1
    assert trees[0].get("ID") == "MainTree"
    assert len(list(trees[0])) == 1, "a BehaviorTree has exactly one root child"
    assert trees[0][0].tag == "Sequence"


def test_xml_element_names_are_the_registration_ids():
    xml = btcpp.to_xml(btcpp.to_btcpp(_legacy_tree()))
    root = ET.fromstring(xml)
    tags = {el.tag for el in root.find("BehaviorTree").iter()}
    assert {"Sequence", "Parallel", "Delay", "ExecuteTrajectory",
            "WaitForRobot", "CheckSafety", "MoveHome"} <= tags


def test_xml_does_not_inline_waypoints():
    xml = btcpp.to_xml(btcpp.to_btcpp(_legacy_tree()))
    # No waypoint payload anywhere — only the {key} references. ("positions" alone
    # would false-positive on the depends_on_positions port NAME.)
    assert '"positions"' not in xml
    assert "time_s" not in xml
    assert 'trajectory="{traj_robot1_W1}"' in xml
    assert 'interlock_target="{interlock_robot2_W2}"' in xml


def test_xml_renders_booleans_the_way_btcpp_parses_them():
    xml = btcpp.to_xml(btcpp.to_btcpp(_legacy_tree()))
    assert 'has_interlock="false"' in xml
    assert 'has_interlock="true"' in xml
    assert "has_interlock=\"False\"" not in xml


def test_tree_nodes_model_declares_every_custom_leaf():
    """Groot2 refuses to draw a node it has no model for."""
    xml = btcpp.to_xml(btcpp.to_btcpp(_legacy_tree()), include_model=True)
    model = ET.fromstring(xml).find("TreeNodesModel")
    assert model is not None

    declared = {el.get("ID"): el for el in model}
    assert set(declared) == set(btcpp.LEAF_PORTS)
    assert declared["CheckSafety"].tag == "Condition"
    assert declared["ExecuteTrajectory"].tag == "Action"

    ports = {p.get("name") for p in declared["ExecuteTrajectory"]}
    assert {"robot_ns", "operation_id", "trajectory", "velocity_scale"} <= ports


def test_model_can_be_omitted():
    xml = btcpp.to_xml(btcpp.to_btcpp(_legacy_tree()), include_model=False)
    assert ET.fromstring(xml).find("TreeNodesModel") is None


def test_converted_tree_carries_its_own_xml():
    """to_btcpp emits the XML itself, so no caller has to remember to serialise."""
    tree = btcpp.to_btcpp(_legacy_tree())
    head = tree["xml"].lstrip()
    assert head.startswith("<?xml") or head.startswith("<root")
    assert "BTCPP_format=\"4\"" in tree["xml"]


# ─── XML → JSON round-trip ────────────────────────────────────────────────────

def test_xml_round_trips_through_from_xml():
    original = btcpp.to_btcpp(_legacy_tree())
    parsed = btcpp.from_xml(original["xml"], blackboard=original["blackboard"])

    assert parsed["BTCPP_format"] == "4"
    assert parsed["main_tree_to_execute"] == "MainTree"
    assert btcpp.node_kind(parsed["root"]) == "Sequence"

    # Structure survives: same node kinds in the same order.
    def kinds(n):
        out = [btcpp.node_kind(n)]
        for c in btcpp.children(n):
            out.extend(kinds(c))
        return out

    assert kinds(parsed["root"]) == kinds(original["root"])


def test_round_trip_recovers_typed_ports():
    """XML attributes are text; ports are typed. Without recovery a round-tripped
    tree would differ from the original on "0.5" vs 0.5."""
    original = btcpp.to_btcpp(_legacy_tree())
    parsed = btcpp.from_xml(original["xml"], blackboard=original["blackboard"])

    leaf = next(n for n in _exec_nodes(parsed["root"])
                if btcpp.port(n, "operation_id") == "W1")
    assert btcpp.port(leaf, "hold_duration_s") == 0.5
    assert btcpp.port(leaf, "has_interlock") is False

    par = next(n for n in btcpp.children(parsed["root"])
               if btcpp.node_kind(n) == "Parallel")
    assert btcpp.port(par, "success_count") == -1


def test_round_trip_keeps_the_trajectory_resolvable():
    original = btcpp.to_btcpp(_legacy_tree())
    parsed = btcpp.from_xml(original["xml"], blackboard=original["blackboard"])
    leaf = next(n for n in _exec_nodes(parsed["root"])
                if btcpp.port(n, "operation_id") == "W1")
    assert btcpp.trajectory_of(leaf, parsed["blackboard"]) == _traj(4)


def test_reconverting_a_parsed_tree_reproduces_the_xml():
    original = btcpp.to_btcpp(_legacy_tree())
    parsed = btcpp.from_xml(original["xml"], blackboard=original["blackboard"])
    assert btcpp.to_btcpp(parsed)["xml"] == original["xml"]


# ─── Failure modes ────────────────────────────────────────────────────────────

def test_from_xml_rejects_a_non_bt_document():
    with pytest.raises(ValueError):
        btcpp.from_xml("<launch><node/></launch>")


def test_from_xml_rejects_malformed_xml():
    with pytest.raises(ValueError):
        btcpp.from_xml("<root BTCPP_format='4'><BehaviorTree>")


def test_from_xml_rejects_a_document_with_no_behaviortree():
    with pytest.raises(ValueError):
        btcpp.from_xml('<root BTCPP_format="4"><TreeNodesModel/></root>')


def test_from_xml_rejects_a_behaviortree_without_exactly_one_root_child():
    """BT.CPP allows exactly one root child; two would silently drop coordination."""
    two = ('<root BTCPP_format="4" main_tree_to_execute="MainTree">'
           '<BehaviorTree ID="MainTree"><Sequence/><Sequence/></BehaviorTree></root>')
    with pytest.raises(ValueError):
        btcpp.from_xml(two)

    none = ('<root BTCPP_format="4" main_tree_to_execute="MainTree">'
            '<BehaviorTree ID="MainTree"/></root>')
    with pytest.raises(ValueError):
        btcpp.from_xml(none)


def test_from_xml_rejects_a_missing_named_tree():
    xml = btcpp.to_btcpp(_legacy_tree())["xml"]
    with pytest.raises(ValueError):
        btcpp.from_xml(xml, main_tree_id="NoSuchTree")


def test_to_btcpp_rejects_a_tree_with_no_root():
    with pytest.raises(ValueError):
        btcpp.to_btcpp({"tree_id": "empty"})
    with pytest.raises(TypeError):
        btcpp.to_btcpp("not a tree")


# ─── Dialect-agnostic accessors ───────────────────────────────────────────────

def test_port_reads_both_dialects():
    legacy = {"type": "EXECUTE", "robot_ns": "robot1",
              "precomputed_trajectory": _traj(2)}
    modern = {"type": "ExecuteTrajectory",
              "ports": {"robot_ns": "robot1", "trajectory": "{k}"}}

    assert btcpp.port(legacy, "robot_ns") == "robot1"
    assert btcpp.port(modern, "robot_ns") == "robot1"
    # The legacy field name is reachable under the canonical port name, which is
    # what lets one walker serve a recorded tree and a new one.
    assert btcpp.port(legacy, "trajectory") == _traj(2)
    assert btcpp.port(modern, "trajectory") == "{k}"
    assert btcpp.port(modern, "missing", "fallback") == "fallback"


def test_node_kind_and_children_tolerate_non_nodes():
    assert btcpp.node_kind("nope") is None
    assert btcpp.node_kind({}) is None
    assert btcpp.children("nope") == []
    assert btcpp.children({"type": "Sequence"}) == []
