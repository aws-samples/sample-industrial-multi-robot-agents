"""A synthetic cell for the RoboBallet tests: two 3-DoF arms, a box, a floor.

Host-runnable: no URDF on disk, no Pinocchio — the arms are a URDF string
parsed by the policy's own chain module.
"""
from __future__ import annotations

import numpy as np

from agents.trajectory.backends.roboballet.core.chain import parse_urdf

URDF = """<robot name="arm3">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="tool"/>
  <joint name="j1" type="revolute"><origin xyz="0 0 0.3" rpy="0 0 0"/>
    <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-3.1" upper="3.1" velocity="2"/></joint>
  <joint name="j2" type="revolute"><origin xyz="0 0 0.2" rpy="0 0 0"/>
    <parent link="l1"/><child link="l2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2"/></joint>
  <joint name="j3" type="revolute"><origin xyz="0.6 0 0" rpy="0 0 0"/>
    <parent link="l2"/><child link="l3"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="2"/></joint>
  <joint name="tcp" type="fixed"><origin xyz="0.5 0 0" rpy="0 0 0"/>
    <parent link="l3"/><child link="tool"/></joint>
</robot>"""


#: The same arm with a 3-axis wrist (roll-pitch-roll at the end of l3): 6 axes,
#: so standoff poses at a fixed tool orientation are reachable, as on a real arm.
URDF6 = URDF.replace(
    """  <joint name="tcp" type="fixed"><origin xyz="0.5 0 0" rpy="0 0 0"/>
    <parent link="l3"/><child link="tool"/></joint>""",
    """  <link name="w1"/><link name="w2"/><link name="w3"/>
  <joint name="j4" type="revolute"><origin xyz="0.35 0 0" rpy="0 0 0"/>
    <parent link="l3"/><child link="w1"/><axis xyz="1 0 0"/>
    <limit lower="-3.1" upper="3.1" velocity="3"/></joint>
  <joint name="j5" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="w1"/><child link="w2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="3"/></joint>
  <joint name="j6" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="w2"/><child link="w3"/><axis xyz="1 0 0"/>
    <limit lower="-3.1" upper="3.1" velocity="3"/></joint>
  <joint name="tcp" type="fixed"><origin xyz="0.15 0 0" rpy="0 0 0"/>
    <parent link="w3"/><child link="tool"/></joint>""")


def robot(rid: str, base, yaw: float = 0.0, wrist: bool = False) -> dict:
    dof = 6 if wrist else 3
    return {"robot_id": rid,
            "chain": {"joints": parse_urdf(URDF6 if wrist else URDF, "tool"), "base": list(base),
                      "yaw": yaw, "tcp_offset": [0.0, 0.0, 0.0], "link_spheres": {},
                      "sphere_radius": 0.08},
            "vel_limits": [2.0] * 3 + [3.0] * (dof - 3), "acc_limits": [8.0] * dof,
            "home": [0.0, -0.3, 0.6] + [0.0] * (dof - 3)}


def _reachable_op(r: dict, q, oid: str) -> dict:
    """A weld the arm can reach: the TCP pose at ``q``, so IK has an exact solution."""
    from agents.trajectory.backends.roboballet.core.chain import Chain
    t = Chain(r["chain"]).tip(q)
    return {"id": oid, "position": t[:3, 3].tolist(), "rotation": t[:3, :3].tolist(),
            "approach_vector": t[:3, 0].tolist()}


def pack(cell_id: str = "synthetic", gap: float = 2.0, wrist: bool = False) -> dict:
    """Two arms facing each other. ``wrist=True``: 6-axis arms, which training needs
    (its tasks are standoff poses at the weld's orientation)."""
    r1 = robot("robot1", (0.0, 0.0, 0.0), wrist=wrist)
    r2 = robot("robot2", (gap, 0.0, 0.0), np.pi, wrist=wrist)
    w = [0.3, -0.4, 0.2] if wrist else []
    from agents.trajectory.backends.roboballet.core.graph import SCHEMA_VERSION
    return {"schema_version": SCHEMA_VERSION, "cell_id": cell_id, "robots": [r1, r2],
            "obstacles": [
                {"id": "floor", "center": [gap / 2, 0, -0.05], "half_extents": [6, 6, 0.05],
                 "rotation": np.eye(3).tolist()},
                {"id": "block", "center": [gap / 2, 0.9, 0.4], "half_extents": [0.1, 0.1, 0.1],
                 "rotation": np.eye(3).tolist()}],
            # Reachable by construction (a 3-axis arm cannot take an arbitrary
            # orientation), one ahead of each arm and one beside it.
            "operations": [_reachable_op(r1, [0.4, 0.2, 0.5] + w, "P:W1"),
                           _reachable_op(r1, [-0.3, 0.1, 0.7] + w, "P:W2"),
                           _reachable_op(r2, [0.4, 0.2, 0.5] + w, "P:W3"),
                           _reachable_op(r2, [-0.3, 0.1, 0.7] + w, "P:W4")],
            "warning_distance_m": 0.1}
