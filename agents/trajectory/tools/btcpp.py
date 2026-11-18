"""BehaviorTree.CPP v4 as the tree format — JSON mirror, XML emitter, XML parser.

The trees this repo produces used to be an ad-hoc JSON dialect (``type`` values
``SEQUENCE`` / ``PARALLEL`` / ``EXECUTE`` / ``INTERLOCK_WAIT`` / ``SAFETY_GATE`` /
``HOME``). That dialect was ours alone: no third-party tool could open it, and the
node semantics existed only in our executor. This module makes
**BehaviorTree.CPP v4** the canonical format instead
(https://www.behaviortree.dev, https://github.com/BehaviorTree/BehaviorTree.CPP),
so a tree can be opened in Groot2, executed by a C++ ``BehaviorTreeFactory``, or
diffed as XML.

Two representations, one meaning:

* **XML** — the interchange artifact, ``<root BTCPP_format="4">`` with one
  ``<BehaviorTree ID="MainTree">`` and a ``<TreeNodesModel>`` so Groot2 knows the
  ports of our custom nodes.
* **JSON mirror** — what travels over DDS/MCP and what the Python executor walks.
  Each node is ``{"type": <registration ID>, "name": <instance label>,
  "ports": {...}, "children": [...]}``. ``type`` is the XML element name, exactly
  as BT.CPP registers it; ``name`` is the optional instance label; ``ports`` are
  the XML attributes.

Vocabulary — built-in control nodes plus four custom leaves:

    Sequence            built-in   (was SEQUENCE)
    Parallel            built-in   (was PARALLEL) — success_count="-1" = all
    Delay               built-in   (was the EXECUTE start_delay_s field)
    ExecuteTrajectory   Action     (was EXECUTE)
    MoveHome            Action     (was HOME)
    WaitForRobot        Action     (was INTERLOCK_WAIT)
    CheckSafety         Condition  (was SAFETY_GATE)

**Why the joint arrays are not XML attributes.** A welding trajectory is hundreds
of waypoints; inlining it would make the XML unreadable and un-editable, which
defeats the point of adopting the format. BT.CPP's own answer is the blackboard:
a port whose value is ``{key}`` reads entry ``key`` at tick time. So bulky or
structured values (trajectory, interlock target, dependency lists) live in a
side-car ``blackboard`` dict and the XML carries only the reference. That is the
idiomatic BT.CPP split, and it keeps the XML something a human can actually read.

Nothing here plans or executes motion — it is a pure format layer.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any, Optional
from xml.dom import minidom

# ─── Vocabulary ───────────────────────────────────────────────────────────────

#: Legacy JSON dialect → BehaviorTree.CPP registration ID.
LEGACY_TO_BTCPP: dict[str, str] = {
    "SEQUENCE":       "Sequence",
    "PARALLEL":       "Parallel",
    "EXECUTE":        "ExecuteTrajectory",
    "HOME":           "MoveHome",
    "INTERLOCK_WAIT": "WaitForRobot",
    "SAFETY_GATE":    "CheckSafety",
}

#: BT.CPP registration ID → legacy name. Used by the executor's alias table so a
#: tree in either dialect runs, and by tooling that still reports legacy names.
BTCPP_TO_LEGACY: dict[str, str] = {v: k for k, v in LEGACY_TO_BTCPP.items()}

#: Control-flow nodes: they own ``children`` rather than ports.
CONTROL_NODES = frozenset({"Sequence", "Parallel", "Fallback", "ReactiveSequence",
                           "ReactiveFallback", "SequenceWithMemory"})

#: Decorators: exactly one child.
DECORATOR_NODES = frozenset({"Delay", "Timeout", "RetryUntilSuccessful",
                             "Inverter", "ForceSuccess", "ForceFailure"})

#: Our custom leaves and the ports they declare. Order is the declaration order
#: emitted into ``<TreeNodesModel>``; ``None`` default means the port is required.
LEAF_PORTS: dict[str, dict[str, Any]] = {
    "ExecuteTrajectory": {
        "_kind": "Action",
        "robot_ns":             None,
        "operation_id":         None,
        "operation_type":       "weld",
        "trajectory":           None,
        "hold_duration_s":      0.5,
        "velocity_scale":       1.0,
        "has_interlock":        False,
        "interlock_target":     "",
        "depends_on_robots":    "",
        "depends_on_positions": "",
    },
    "MoveHome": {
        "_kind": "Action",
        "robot_ns":   None,
        "trajectory": "",
    },
    "WaitForRobot": {
        "_kind": "Action",
        "robot_ns":             None,
        "depends_on_robots":    "",
        "depends_on_joints":    "",
        "depends_on_positions": "",
        "depends_on_ops":       "",
    },
    "CheckSafety": {
        "_kind": "Condition",
        "condition": "estop_clear",
    },
}

#: Ports whose value is a list/dict: they go to the blackboard, referenced as
#: ``{key}`` from the XML. Everything else is a plain scalar attribute.
_BLACKBOARD_PORTS = frozenset({"trajectory", "interlock_target", "depends_on_robots",
                               "depends_on_joints", "depends_on_positions"})

#: Legacy field name → BT.CPP port name (only where they differ).
_LEGACY_FIELD_TO_PORT = {"precomputed_trajectory": "trajectory"}

MAIN_TREE_ID = "MainTree"
BTCPP_FORMAT = "4"


# ─── Node access that works on either dialect ─────────────────────────────────

def node_kind(node: dict) -> Optional[str]:
    """Registration ID of ``node``, normalising a legacy ``type`` if present.

    The executor and every tree walker go through this, so a tree authored in
    either dialect reads the same. Returns None for a non-dict.
    """
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    if not t:
        return None
    return LEGACY_TO_BTCPP.get(str(t), str(t))


def port(node: dict, name: str, default: Any = None) -> Any:
    """Read port ``name`` from ``node``, whichever dialect it is in.

    A BT.CPP node keeps its ports under ``ports``; a legacy node kept them as
    top-level fields under different names. Checking both means one accessor
    serves both, instead of every caller carrying a compatibility branch.
    """
    if not isinstance(node, dict):
        return default
    ports = node.get("ports")
    if isinstance(ports, dict) and name in ports:
        return ports[name]
    if name in node:
        return node[name]
    for legacy, canonical in _LEGACY_FIELD_TO_PORT.items():
        if canonical == name and legacy in node:
            return node[legacy]
    return default


def children(node: dict) -> list:
    return node.get("children") or [] if isinstance(node, dict) else []


def resolve(value: Any, blackboard: Optional[dict]) -> Any:
    """Dereference a ``{key}`` blackboard pointer; pass anything else through.

    BT.CPP port syntax: the literal string ``"{traj_robot1_W1}"`` means "read
    blackboard entry traj_robot1_W1". A missing entry resolves to None rather
    than raising — the caller (e.g. the executor's empty-trajectory check)
    reports it as the actionable failure it is.
    """
    if isinstance(value, str) and len(value) > 2 and value[0] == "{" and value[-1] == "}":
        key = value[1:-1].strip()
        return (blackboard or {}).get(key)
    return value


def trajectory_of(node: dict, blackboard: Optional[dict] = None) -> list:
    """The waypoint list for a leaf, following a blackboard reference if needed."""
    traj = resolve(port(node, "trajectory"), blackboard)
    return traj if isinstance(traj, list) else []


# ─── Legacy JSON → BT.CPP JSON ────────────────────────────────────────────────

def _bb_key(kind: str, node: dict, ports_seen: dict, name: str) -> str:
    """Stable, readable blackboard key for one node's structured port."""
    rid = port(node, "robot_ns") or "cell"
    oid = port(node, "operation_id") or ports_seen.get("_seq") or "n"
    stem = {"trajectory": "traj", "interlock_target": "interlock",
            "depends_on_robots": "deps", "depends_on_joints": "waitjoints",
            "depends_on_positions": "waitpos"}.get(name, name)
    return f"{stem}_{rid}_{oid}"


def _instance_name(kind: str, node: dict) -> Optional[str]:
    """Readable Groot2/logging label for a node instance."""
    rid = port(node, "robot_ns")
    oid = port(node, "operation_id")
    if kind == "ExecuteTrajectory" and rid and oid:
        return f"{port(node, 'operation_type') or 'op'}_{rid}_{oid}"
    if kind in ("MoveHome", "WaitForRobot") and rid:
        return f"{'home' if kind == 'MoveHome' else 'wait'}_{rid}"
    return None


def to_btcpp(tree: dict) -> dict:
    """Convert a tree (legacy dialect or already-BT.CPP) to canonical BT.CPP JSON.

    Idempotent: converting an already-converted tree returns an equivalent tree,
    so callers never have to track which dialect they hold.

    Returns a new dict with ``BTCPP_format``, ``main_tree_to_execute``, ``root``
    (BT.CPP nodes), ``blackboard`` (the bulky values the ports point at) and
    ``xml`` (the interchange artifact). ``tree_id`` / ``cell_id`` / ``metadata``
    are preserved.
    """
    if not isinstance(tree, dict):
        raise TypeError("tree must be a dict")

    blackboard: dict[str, Any] = dict(tree.get("blackboard") or {})
    counter = {"_seq": 0}

    def convert(node: Any) -> Optional[dict]:
        if not isinstance(node, dict):
            return None
        kind = node_kind(node)
        if kind is None:
            return None
        counter["_seq"] += 1
        out: dict[str, Any] = {"type": kind}

        label = node.get("name") or _instance_name(kind, node)
        if label:
            out["name"] = label
        if node.get("id"):
            out["id"] = node["id"]

        if kind in CONTROL_NODES:
            out["ports"] = _control_ports(kind, node)
            out["children"] = [c for c in (convert(ch) for ch in children(node)) if c]
            return out
        if kind in DECORATOR_NODES:
            out["ports"] = dict(node.get("ports") or {})
            out["children"] = [c for c in (convert(ch) for ch in children(node)) if c]
            return out

        # Custom leaf: pull every declared port, sending structured values to the
        # blackboard so the XML stays legible.
        spec = LEAF_PORTS.get(kind, {})
        ports: dict[str, Any] = {}
        seq_ctx = {"_seq": f"n{counter['_seq']}"}
        for pname, pdefault in spec.items():
            if pname == "_kind":
                continue
            val = port(node, pname, pdefault)
            if pname in _BLACKBOARD_PORTS:
                if isinstance(val, (list, dict)) and val:
                    key = _bb_key(kind, node, seq_ctx, pname)
                    blackboard[key] = val
                    ports[pname] = "{" + key + "}"
                elif isinstance(val, str) and val:
                    ports[pname] = val          # already a {ref}
                continue
            if val is not None:
                ports[pname] = val
        out["ports"] = ports

        # start_delay_s becomes the built-in Delay decorator: BT.CPP already has
        # a node for "tick the child later", so expressing it as a leaf field
        # would hide timing from anything reading the tree.
        delay_s = float(port(node, "start_delay_s", 0.0) or 0.0)
        if kind == "ExecuteTrajectory" and delay_s > 0:
            return {"type": "Delay",
                    "name": f"delay_{label or 'exec'}",
                    "ports": {"delay_msec": int(round(delay_s * 1000))},
                    "children": [out]}
        return out

    src_root = tree.get("root")
    root = convert(src_root) if src_root is not None else None
    if root is None:
        raise ValueError("tree has no convertible 'root' node")

    out_tree: dict[str, Any] = {
        "tree_id":              tree.get("tree_id"),
        "cell_id":              tree.get("cell_id"),
        "BTCPP_format":         BTCPP_FORMAT,
        "main_tree_to_execute": tree.get("main_tree_to_execute") or MAIN_TREE_ID,
        "root":                 root,
        "blackboard":           blackboard,
        "metadata":             tree.get("metadata") or {},
    }
    out_tree["xml"] = to_xml(out_tree)
    return out_tree


def _control_ports(kind: str, node: dict) -> dict:
    """Ports for a control node, defaulting Parallel to all-must-succeed.

    BT.CPP's Parallel defaults to ``success_count=-1`` (all children) and
    ``failure_count=1`` (first failure fails the node), which is exactly the
    semantics a synchronised multi-arm step needs — both arms must finish, and
    one arm failing must stop the step. Stated explicitly so the XML does not
    depend on a library default that could change.
    """
    ports = dict(node.get("ports") or {})
    if kind == "Parallel":
        ports.setdefault("success_count", -1)
        ports.setdefault("failure_count", 1)
    return ports


# ─── BT.CPP JSON → XML ────────────────────────────────────────────────────────

def _attr(value: Any) -> str:
    """Render a port value as an XML attribute string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def _append_node(parent: ET.Element, node: dict) -> None:
    kind = node_kind(node)
    if kind is None:
        return
    el = ET.SubElement(parent, kind)
    if node.get("name"):
        el.set("name", str(node["name"]))
    for pname, pval in (node.get("ports") or {}).items():
        if pval is None or pval == "":
            continue
        el.set(pname, _attr(pval))
    for child in children(node):
        _append_node(el, child)


def tree_nodes_model() -> ET.Element:
    """``<TreeNodesModel>`` declaring our custom nodes' ports, for Groot2.

    Groot2 refuses to draw a node it has no model for, so a tree without this
    section opens as an error instead of a diagram. Built-in nodes are omitted —
    the library already knows them.
    """
    model = ET.Element("TreeNodesModel")
    for nid, spec in LEAF_PORTS.items():
        el = ET.SubElement(model, spec.get("_kind", "Action"), {"ID": nid})
        for pname, pdefault in spec.items():
            if pname == "_kind":
                continue
            attrs = {"name": pname}
            if pdefault is not None and pdefault != "":
                attrs["default"] = _attr(pdefault)
            ET.SubElement(el, "input_port", attrs)
    return model


def to_xml(tree: dict, include_model: bool = True) -> str:
    """Serialise a BT.CPP JSON tree as ``<root BTCPP_format="4">`` XML.

    The result is what BehaviorTree.CPP's ``createTreeFromText`` /
    ``createTreeFromFile`` consume and what Groot2 opens. Blackboard contents are
    NOT inlined — they are referenced as ``{key}`` ports and shipped alongside
    (see ``blackboard_json``), because the waypoint arrays would otherwise bury
    the structure the XML exists to show.
    """
    root = ET.Element("root", {"BTCPP_format": BTCPP_FORMAT})
    main_id = tree.get("main_tree_to_execute") or MAIN_TREE_ID
    root.set("main_tree_to_execute", main_id)

    bt_el = ET.SubElement(root, "BehaviorTree", {"ID": main_id})
    node = tree.get("root")
    if isinstance(node, dict):
        _append_node(bt_el, node)

    if include_model:
        root.append(tree_nodes_model())

    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    # minidom emits an XML declaration plus blank lines; keep the declaration
    # (log_test.xml in the upstream repo has one) and drop the empty lines.
    return "\n".join(line for line in pretty.splitlines() if line.strip()) + "\n"


def blackboard_json(tree: dict) -> str:
    """The side-car blackboard as JSON, to load before ticking the tree."""
    return json.dumps(tree.get("blackboard") or {}, indent=2)


# ─── XML → BT.CPP JSON ────────────────────────────────────────────────────────

def _coerce(value: str) -> Any:
    """Best-effort scalar recovery from an XML attribute string.

    XML attributes are always text; ports are typed. Recovering bool/int/float
    means a tree that round-trips through XML compares equal to the original
    instead of differing on ``"0.5"`` vs ``0.5``. Blackboard references and
    anything unparseable stay strings.
    """
    v = value.strip()
    if v in ("true", "false"):
        return v == "true"
    if v.startswith("{") and v.endswith("}"):
        return v                                  # blackboard reference
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return value


def _parse_node(el: ET.Element) -> dict:
    node: dict[str, Any] = {"type": el.tag}
    ports: dict[str, Any] = {}
    for k, v in el.attrib.items():
        if k == "name":
            node["name"] = v
        else:
            ports[k] = _coerce(v)
    node["ports"] = ports
    kids = [_parse_node(c) for c in el]
    if kids:
        node["children"] = kids
    return node


def from_xml(xml_text: str, blackboard: Optional[dict] = None,
             main_tree_id: Optional[str] = None) -> dict:
    """Parse BT.CPP v4 XML back into the JSON mirror.

    This is what makes the format two-way: a tree can be edited in Groot2 or by
    hand and fed back to the Python executor. Node ``id``s are not preserved —
    they are ours, not BT.CPP's, and carry no meaning in the XML.

    Raises ValueError if the document is not a BT.CPP tree or the requested
    ``BehaviorTree`` is absent.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"not well-formed XML: {exc}") from exc
    if root.tag != "root":
        raise ValueError(f"expected <root>, found <{root.tag}>")

    want = main_tree_id or root.get("main_tree_to_execute")
    trees = root.findall("BehaviorTree")
    if not trees:
        raise ValueError("no <BehaviorTree> element in the document")
    chosen = None
    if want:
        chosen = next((t for t in trees if t.get("ID") == want), None)
        if chosen is None:
            raise ValueError(f"no <BehaviorTree ID=\"{want}\"> in the document")
    else:
        chosen = trees[0]

    kids = list(chosen)
    if len(kids) != 1:
        raise ValueError(f"<BehaviorTree ID=\"{chosen.get('ID')}\"> must have exactly "
                         f"one root child, found {len(kids)}")

    return {
        "BTCPP_format":         root.get("BTCPP_format") or BTCPP_FORMAT,
        "main_tree_to_execute": chosen.get("ID") or MAIN_TREE_ID,
        "root":                 _parse_node(kids[0]),
        "blackboard":           dict(blackboard or {}),
    }
