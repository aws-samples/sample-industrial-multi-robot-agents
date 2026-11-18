"""Obstacle primitives as the policy sees them: oriented boxes.

RoboBallet gives the network one node per cuboid primitive (meshes are reduced
to oriented bounding boxes of a convex decomposition). A cell's
``obstacles.yaml`` already IS primitives — boxes, planes and spheres — so they
map one to one: a box stays a box, a sphere becomes its bounding cube, and a
plane becomes a thin slab under the workspace.

A box is ``{"center": [3], "half_extents": [3], "rotation": 3x3, "id"}``.

numpy + stdlib only.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .chain import quat_matrix

#: Half-thickness of the slab a plane becomes, and its half-width.
PLANE_SLAB_HALF_M = 0.05
PLANE_SLAB_SPAN_M = 6.0


def _frame_with_z(normal: Sequence[float]) -> np.ndarray:
    z = np.asarray(normal, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x) + 1e-12
    return np.stack([x, np.cross(z, x), z], axis=1)


def box_from_obstacle(ob: dict, workspace_center: Optional[Sequence[float]] = None) -> Optional[dict]:
    """One ``obstacles.yaml`` entry → an oriented box (``None`` for unknown types).

    The entry's ``collision_margin`` is added to the extents: the policy should
    keep the same air the cell's checker demands.
    """
    kind = str(ob.get("type", "box")).lower()
    margin = float(ob.get("collision_margin", 0.0) or 0.0)
    oid = str(ob.get("id", kind))
    if kind == "box":
        rot = quat_matrix(ob["orientation"]) if isinstance(ob.get("orientation"), dict) else np.eye(3)
        half = np.asarray(ob["half_extents"], float) + margin
        return {"id": oid, "center": [float(v) for v in ob["center"]],
                "half_extents": half.tolist(), "rotation": rot.tolist()}
    if kind == "sphere":
        r = float(ob["radius"]) + margin
        return {"id": oid, "center": [float(v) for v in ob["center"]],
                "half_extents": [r, r, r], "rotation": np.eye(3).tolist()}
    if kind == "plane":
        n = np.asarray(ob["normal"], float)
        n /= np.linalg.norm(n) + 1e-12
        d = float(ob.get("offset", 0.0))
        c = np.asarray(workspace_center if workspace_center is not None else (0, 0, 0), float)
        on_plane = c - (n @ c + d) * n
        center = on_plane - n * (PLANE_SLAB_HALF_M - margin)
        return {"id": oid, "center": center.tolist(),
                "half_extents": [PLANE_SLAB_SPAN_M, PLANE_SLAB_SPAN_M, PLANE_SLAB_HALF_M],
                "rotation": _frame_with_z(n).tolist()}
    return None


def boxes_from_obstacles(obstacles: Iterable[dict],
                         workspace_center: Optional[Sequence[float]] = None) -> list[dict]:
    return [b for b in (box_from_obstacle(o, workspace_center) for o in obstacles) if b]


class BoxSet:
    """Vectorised distances from many points to many oriented boxes."""

    def __init__(self, boxes: Sequence[dict]):
        self.boxes = list(boxes)
        n = len(self.boxes)
        self.centers = np.array([b["center"] for b in self.boxes], float).reshape(n, 3)
        self.half = np.array([b["half_extents"] for b in self.boxes], float).reshape(n, 3)
        self.rot = np.array([b["rotation"] for b in self.boxes], float).reshape(n, 3, 3)

    def __len__(self) -> int:
        return len(self.boxes)

    def distances(self, points: np.ndarray) -> np.ndarray:
        """``(P, B)`` distance from each point to each box surface (0 inside)."""
        if not len(self.boxes):
            return np.full((len(points), 0), np.inf)
        rel = points[:, None, :] - self.centers[None, :, :]                 # P,B,3
        local = np.einsum("bji,pbj->pbi", self.rot, rel)                    # R^T rel
        excess = np.maximum(np.abs(local) - self.half[None], 0.0)
        return np.linalg.norm(excess, axis=-1)

    def collides(self, points: np.ndarray, radius: float) -> bool:
        if not len(self.boxes) or not len(points):
            return False
        return bool((self.distances(points) < radius).any())

    def nearest(self, points: np.ndarray, k: int) -> np.ndarray:
        """Indices of the ``k`` boxes closest to any of ``points`` (fewer if fewer exist)."""
        if not len(self.boxes):
            return np.zeros(0, int)
        d = self.distances(points).min(axis=0)
        k = min(k, len(d))
        return np.argsort(d, kind="stable")[:k]


def pair_clearance(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    """Minimum surface clearance between two sphere sets of one radius."""
    if not len(a) or not len(b):
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min()) - 2.0 * radius
