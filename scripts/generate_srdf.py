#!/usr/bin/env python3
"""
Generate an SRDF from a URDF — the URDF is the master.

The semantic description is derived *entirely* from the URDF so it can never
drift from the kinematic model ROS tooling loads. For a serial arm it emits:

  * <robot name=...>          — copied verbatim from the URDF robot name
  * <group name=...>          — a base->tip kinematic chain over the movable joints
  * <group_state name="home"> — each group joint at 0 (clamped into its limits)
  * <end_effector ...>        — anchored at the chain tip link
  * <disable_collisions ...>  — every parent/child link pair where both links
                                carry collision geometry (reason="Adjacent")

Link and joint names are taken VERBATIM from the URDF (no prefixing): the cell
runtime loads the URDF unprefixed and applies the
robot namespace only as the ROS node namespace, so a prefixed SRDF would fail to
resolve against robot_description ("... not known to the URDF", "Group is empty").

Usage:
    # All robots in a cell (reads config/robots.yaml for robot ids + group name)
    python3 scripts/generate_srdf.py --cell welding_cell_2_small

    # A single URDF file
    python3 scripts/generate_srdf.py --urdf path/to/robot.urdf --out path/to/robot.srdf
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MOVABLE = {"revolute", "prismatic", "continuous"}


class Joint:
    __slots__ = ("name", "type", "parent", "child", "lower", "upper")

    def __init__(self, name, jtype, parent, child, lower, upper):
        self.name = name
        self.type = jtype
        self.parent = parent
        self.child = child
        self.lower = lower
        self.upper = upper


def parse_urdf(urdf_text: str):
    """Return (robot_name, {link_name: has_collision}, [Joint, ...])."""
    root = ET.fromstring(urdf_text)
    robot_name = root.get("name") or "robot"

    links: dict[str, bool] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name:
            links[name] = link.find("collision") is not None

    joints: list[Joint] = []
    for j in root.findall("joint"):
        parent_el = j.find("parent")
        child_el = j.find("child")
        lim = j.find("limit")
        lower = upper = None
        if lim is not None:
            lo, up = lim.get("lower"), lim.get("upper")
            lower = float(lo) if lo is not None else None
            upper = float(up) if up is not None else None
        joints.append(
            Joint(
                name=j.get("name"),
                jtype=j.get("type"),
                parent=parent_el.get("link") if parent_el is not None else None,
                child=child_el.get("link") if child_el is not None else None,
                lower=lower,
                upper=upper,
            )
        )
    return robot_name, links, joints


def build_chain(joints: list[Joint]) -> list[Joint]:
    """Order the movable joints into the serial chain base->tip."""
    movable = [j for j in joints if j.type in MOVABLE]
    if not movable:
        return []

    child_links = {j.child for j in movable}
    parent_to_joints: dict[str, list[Joint]] = {}
    for j in movable:
        parent_to_joints.setdefault(j.parent, []).append(j)

    # Chain root = a movable joint whose parent is never a movable child.
    roots = [j for j in movable if j.parent not in child_links]
    if not roots:
        return movable  # unexpected topology; fall back to document order

    chain: list[Joint] = []
    seen: set[str] = set()
    current = roots[0].parent
    while current in parent_to_joints:
        nxt = next((j for j in parent_to_joints[current] if j.name not in seen), None)
        if nxt is None:
            break
        chain.append(nxt)
        seen.add(nxt.name)
        current = nxt.child
    return chain


def adjacent_pairs(joints: list[Joint], links: dict[str, bool]) -> list[tuple[str, str]]:
    """Parent/child pairs where both links carry collision geometry."""
    pairs = []
    for j in joints:
        if j.parent and j.child and links.get(j.parent) and links.get(j.child):
            pairs.append((j.parent, j.child))
    return pairs


def home_value(j: Joint) -> str:
    lo = j.lower if j.lower is not None else -math.pi
    up = j.upper if j.upper is not None else math.pi
    val = 0.0 if lo <= 0.0 <= up else round((lo + up) / 2.0, 6)
    return f"{val:g}"


def build_srdf(urdf_text: str, group_name: str = "kuka_arm", ee_name: str = "tool_tip") -> str:
    robot_name, links, joints = parse_urdf(urdf_text)
    chain = build_chain(joints)
    if not chain:
        raise SystemExit("No movable joints found in URDF; cannot build a planning group.")

    base_link = chain[0].parent
    tip_link = chain[-1].child

    out: list[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append("<!--")
    out.append(f"  SRDF derived from URDF (URDF is master) for robot '{robot_name}'.")
    out.append("  Generated by scripts/generate_srdf.py — do not edit by hand.")
    out.append(f"  Chain: {base_link} -> {tip_link} over "
               f"{', '.join(j.name for j in chain)}.")
    out.append("  Link/joint names are verbatim from the URDF (unprefixed).")
    out.append("-->")
    out.append(f'<robot name="{robot_name}">')
    out.append("")
    out.append("  <!-- Planning group: full movable chain base -> tip -->")
    out.append(f'  <group name="{group_name}">')
    out.append(f'    <chain base_link="{base_link}" tip_link="{tip_link}"/>')
    out.append("  </group>")
    out.append("")
    out.append("  <!-- Named home state: each joint at 0 (clamped into its URDF limits) -->")
    out.append(f'  <group_state name="home" group="{group_name}">')
    for j in chain:
        out.append(f'    <joint name="{j.name}" value="{home_value(j)}"/>')
    out.append("  </group_state>")
    out.append("")
    out.append("  <!-- End-effector anchored at the chain tip (tool geometry is baked into it) -->")
    out.append(f'  <end_effector name="{ee_name}" parent_link="{tip_link}" group="{group_name}"/>')
    out.append("")
    out.append("  <!-- Adjacent link pairs: disable self-collision checks -->")
    for a, b in adjacent_pairs(joints, links):
        out.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="Adjacent"/>')
    out.append("")
    out.append("</robot>")
    out.append("")
    return "\n".join(out)


def _load_robots_yaml(cell_dir: Path) -> dict:
    import yaml

    with open(cell_dir / "config" / "robots.yaml") as f:
        return yaml.safe_load(f) or {}


def _iter_cell_robots(robots_cfg: dict):
    """Yield (robot_id, robot_cfg) for every robot in robots.yaml."""
    for _ns, ns_block in (robots_cfg.get("robots") or {}).items():
        if isinstance(ns_block, dict):
            for rid, rcfg in ns_block.items():
                if isinstance(rcfg, dict):
                    yield rid, rcfg


def _generate_for_cell(cell_id: str, cell_base: str) -> int:
    cell_dir = Path(cell_base) / cell_id
    robots_cfg = _load_robots_yaml(cell_dir)

    count = 0
    for robot_id, rcfg in _iter_cell_robots(robots_cfg):
        urdf_glob = sorted((cell_dir / "urdf" / robot_id).glob("*.urdf"))
        if not urdf_glob:
            print(f"[generate_srdf] {robot_id}: no URDF under urdf/{robot_id}/ — skipping",
                  file=sys.stderr)
            continue
        urdf_path = urdf_glob[0]
        group_name = rcfg.get("planning_group", "kuka_arm")
        # Keep the semantic end-effector label from robots.yaml; it is anchored at
        # the URDF tip link (the tool geometry is baked into that link).
        ee_name = rcfg.get("end_effector_link", "tool_tip")

        srdf_text = build_srdf(urdf_path.read_text(), group_name=group_name, ee_name=ee_name)

        srdf_dir = cell_dir / "srdf" / robot_id
        srdf_dir.mkdir(parents=True, exist_ok=True)
        srdf_path = srdf_dir / (urdf_path.stem + ".srdf")
        srdf_path.write_text(srdf_text)
        print(f"[generate_srdf] {robot_id}: {srdf_path} "
              f"(group={group_name}, ee={ee_name})")
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", help="cell id under the cells/ tree (batch mode)")
    ap.add_argument("--cell-base", default=os.environ.get("CELL_BASE_PATH", "./cells"),
                    help="root of the cells/ tree (default: $CELL_BASE_PATH or ./cells)")
    ap.add_argument("--urdf", help="single URDF file (single-file mode)")
    ap.add_argument("--out", help="output SRDF path (single-file mode)")
    ap.add_argument("--group", default="kuka_arm", help="planning group name (single-file mode)")
    ap.add_argument("--ee-name", default="tool_tip", help="end-effector name (single-file mode)")
    args = ap.parse_args()

    if args.urdf:
        urdf_text = Path(args.urdf).read_text()
        srdf_text = build_srdf(urdf_text, group_name=args.group, ee_name=args.ee_name)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(srdf_text)
            print(f"[generate_srdf] wrote {args.out}")
        else:
            sys.stdout.write(srdf_text)
        return

    if args.cell:
        n = _generate_for_cell(args.cell, args.cell_base)
        if n == 0:
            sys.exit("[generate_srdf] no SRDFs generated")
        return

    ap.error("provide either --cell CELL or --urdf FILE")


if __name__ == "__main__":
    main()
