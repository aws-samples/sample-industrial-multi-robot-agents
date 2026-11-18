"""A serial arm as the policy sees it: URDF chain, forward kinematics, IK.

The policy is trained in SageMaker, far from Pinocchio, and evaluated in the
trajectory agent next to it. Both sides compute the graph features with THIS
module so that a feature means the same thing at training and at planning time;
Pinocchio stays the authority for collision and validation, which the backend
leaves to the shared step pipeline.

numpy + stdlib only (the SageMaker bundle ships this package as-is).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Optional, Sequence

import numpy as np

#: Joint types that move. ``continuous`` is a revolute joint without limits.
MOVABLE = ("revolute", "continuous", "prismatic")


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), \
        math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(xyz: Sequence[float], rot: Optional[np.ndarray] = None) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = np.asarray(xyz, float)
    if rot is not None:
        t[:3, :3] = rot
    return t


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def base_transform(position: Sequence[float], yaw: float) -> np.ndarray:
    """The arm's base in the world: ``robots.yaml`` base_position + base_orientation_yaw."""
    return transform(position, rpy_matrix(0.0, 0.0, float(yaw)))


def quat_matrix(q: dict) -> np.ndarray:
    """``{"x","y","z","w"}`` → rotation matrix (normalised; identity when degenerate)."""
    x, y, z, w = (float(q.get(k, 0.0)) for k in ("x", "y", "z", "w"))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation_angle(ra: np.ndarray, rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, radians."""
    c = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def parse_urdf(text: str, tip_link: str = "") -> list[dict]:
    """Joints from the root to ``tip_link`` (the last movable joint's child when
    empty), each ``{name, type, origin: 4x4 list, axis, lower, upper, child}``.

    Fixed joints are kept: they place the next frame, and the link envelope is
    declared per link frame.
    """
    root = ET.fromstring(text)
    by_child: dict[str, ET.Element] = {}
    for j in root.findall("joint"):
        child = j.find("child")
        if child is not None:
            by_child[child.get("link")] = j
    if not tip_link:
        movable = [j for j in root.findall("joint") if j.get("type") in MOVABLE]
        if not movable:
            raise ValueError("URDF has no movable joint")
        tip_link = movable[-1].find("child").get("link")
    chain: list[dict] = []
    link = tip_link
    while link in by_child:
        j = by_child[link]
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = [float(v) for v in (ax.get("xyz", "1 0 0") if ax is not None else "1 0 0").split()]
        lim = j.find("limit")
        lower = float(lim.get("lower", -math.pi)) if lim is not None else -math.pi
        upper = float(lim.get("upper", math.pi)) if lim is not None else math.pi
        parent = j.find("parent").get("link")
        chain.append({"name": j.get("name"), "type": j.get("type"),
                      "origin": transform(xyz, rpy_matrix(*rpy)).tolist(), "axis": axis,
                      "lower": lower, "upper": upper, "child": link, "parent": parent})
        link = parent
    if not chain:
        raise ValueError(f"tip link {tip_link!r} is not the child of any joint")
    chain.reverse()
    return chain


class Chain:
    """Forward kinematics of one arm placed in the world.

    ``spec`` is the JSON-serialisable description a scene pack carries::

        {"joints": parse_urdf(...), "base": [x, y, z], "yaw": rad,
         "tcp_offset": [x, y, z], "link_spheres": {link: [[x, y, z], ...]},
         "sphere_radius": m}
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self.joints = spec["joints"]
        self._origins = [np.asarray(j["origin"], float) for j in self.joints]
        self._axes = [np.asarray(j["axis"], float) / (np.linalg.norm(j["axis"]) + 1e-12)
                      for j in self.joints]
        # Per joint, precomputed once: the origin's rotation and translation, and the
        # cross-product matrix of the axis (Rodrigues) — FK runs in every collision
        # check and every IK iteration, so it allocates as little as it can.
        self._o_rot = [o[:3, :3].copy() for o in self._origins]
        self._o_pos = [o[:3, 3].copy() for o in self._origins]
        self._kind = [0 if j["type"] not in MOVABLE else (2 if j["type"] == "prismatic" else 1)
                      for j in self.joints]
        self._k = [np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                   for a in self._axes]
        self._k2 = [k @ k for k in self._k]
        self.movable = [k for k, j in enumerate(self.joints) if j["type"] in MOVABLE]
        self.dof = len(self.movable)
        self.base = base_transform(spec.get("base", (0, 0, 0)), spec.get("yaw", 0.0))
        self.base_inv = np.linalg.inv(self.base)
        self.tcp = transform(spec.get("tcp_offset") or (0.0, 0.0, 0.0))
        self._tcp_pos = self.tcp[:3, 3].copy()
        self.lower = np.array([self.joints[k]["lower"] for k in self.movable], float)
        self.upper = np.array([self.joints[k]["upper"] for k in self.movable], float)
        self.sphere_radius = float(spec.get("sphere_radius", 0.12))
        self._link_pts = {name: np.asarray(pts, float).reshape(-1, 3)
                          for name, pts in (spec.get("link_spheres") or {}).items()}
        moving, self._moving_links = False, set()
        for j in self.joints:
            moving = moving or j["type"] in MOVABLE
            if moving:
                self._moving_links.add(j["child"])
        # Reach: the longest the chain can be (base origin to TCP), a cheap bound
        # that IK callers use to skip targets no configuration can touch.
        self.reach = float(sum(np.linalg.norm(p) for p in self._o_pos)
                           + np.linalg.norm(self._tcp_pos))

    def _walk(self, q) -> tuple[list, list, list]:
        """``(rotations, positions, joint_axes_world)`` of every child frame, base first."""
        q = np.asarray(q, float)
        r, p = self.base[:3, :3], self.base[:3, 3]
        rots, poss, axes = [r], [p], []
        k_q = 0
        for kind, ro, po, ax, k, k2 in zip(self._kind, self._o_rot, self._o_pos,
                                           self._axes, self._k, self._k2):
            p = p + r @ po
            r = r @ ro
            if kind == 1:
                axes.append((r @ ax, p))
                th = q[k_q]
                r = r @ (np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * k2)
                k_q += 1
            elif kind == 2:
                axes.append((r @ ax, None))
                p = p + r @ (ax * q[k_q])
                k_q += 1
            rots.append(r)
            poss.append(p)
        return rots, poss, axes

    def frames(self, q: Sequence[float]) -> dict[str, np.ndarray]:
        """World transform of every link of the chain (the root as ``"__base__"``)."""
        rots, poss, _ = self._walk(q)
        def tf(r, p):
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = r, p
            return t
        out = {"__base__": tf(rots[0], poss[0])}
        if self.joints[0].get("parent"):
            out[self.joints[0]["parent"]] = out["__base__"]
        for j, r, p in zip(self.joints, rots[1:], poss[1:]):
            out[j["child"]] = tf(r, p)
        return out

    def tip(self, q: Sequence[float]) -> np.ndarray:
        """World transform of the TCP (tip link + the declared ``tcp_offset``)."""
        rots, poss, _ = self._walk(q)
        t = np.eye(4)
        t[:3, :3] = rots[-1]
        t[:3, 3] = poss[-1] + rots[-1] @ self._tcp_pos
        return t

    @property
    def moving_links(self) -> set[str]:
        """Links carried by at least one movable joint (the base and its socle are not)."""
        return self._moving_links

    def points(self, q: Sequence[float], moving_only: bool = False) -> np.ndarray:
        """World centres of the arm's collision spheres, ``(N, 3)``.

        The declared link envelope when the cell has one; otherwise the joint
        frames and the TCP, interpolated so that no gap exceeds one radius.
        ``moving_only`` leaves out the links no joint moves: the base stands on
        the floor it would otherwise "collide" with.
        """
        fr = self.frames(q)
        if self._link_pts:
            keep = self.moving_links if moving_only else set(fr)
            pts = [(fr[name][:3, :3] @ p.T).T + fr[name][:3, 3]
                   for name, p in self._link_pts.items() if name in fr and name in keep]
            pts.append(self.tip(q)[None, :3, 3])
            return np.concatenate(pts, axis=0)
        anchors = [fr["__base__"][:3, 3]] + [fr[j["child"]][:3, 3] for j in self.joints] \
            + [self.tip(q)[:3, 3]]
        if moving_only:
            # from the first moving frame on (its own origin does not move either)
            first = next((k for k, j in enumerate(self.joints) if j["type"] in MOVABLE), 0)
            anchors = anchors[first + 1:]
        out = []
        for a, b in zip(anchors, anchors[1:]):
            n = max(1, int(math.ceil(np.linalg.norm(b - a) / self.sphere_radius)))
            out.extend(a + (b - a) * (i / n) for i in range(n))
        out.append(anchors[-1])
        return np.asarray(out, float)

    def clip(self, q: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(q, float), self.lower, self.upper)

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """Geometric 6 x dof Jacobian of the TCP (linear; angular), from ONE FK pass."""
        rots, poss, axes = self._walk(q)
        tip = poss[-1] + rots[-1] @ self._tcp_pos
        jac = np.zeros((6, self.dof))
        for i, (z, origin) in enumerate(axes):
            if origin is None:                       # prismatic
                jac[:3, i] = z
            else:
                jac[:3, i] = np.cross(z, tip - origin)
                jac[3:, i] = z
        return jac

    def _pose_error(self, q, target: np.ndarray, rotation: Optional[np.ndarray]) -> np.ndarray:
        t = self.tip(q)
        e = target - t[:3, 3]
        if rotation is None:
            return e
        dr = rotation @ t[:3, :3].T
        return np.concatenate([e, 0.5 * np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0],
                                                   dr[1, 0] - dr[0, 1]])])

    def ik(self, position: Sequence[float], rotation: Optional[np.ndarray], q0: Sequence[float],
           *, iters: int = 150, tol_m: float = 1e-3, tol_rad: float = 1e-2,
           damping: float = 0.1, max_step_rad: float = 0.5) -> Optional[np.ndarray]:
        """Levenberg-Marquardt IK from ``q0``; ``None`` when it does not converge.

        Damped least squares with ADAPTIVE damping: a step is kept only when it
        reduces the pose error (then the damping halves), otherwise it is
        rejected and the damping grows. A fixed-damping solver from HOME failed on
        almost every far target of a 6-axis arm; this one converges on most from
        HOME and on nearly all with a couple of seeds.

        Used to generate training goals, never to plan: the planner's IK is
        Pinocchio's (``tools/pinocchio.run_ik``).
        """
        target = np.asarray(position, float)
        if np.linalg.norm(target - self.base[:3, 3]) > self.reach + tol_m:
            return None                            # no configuration reaches that far
        eye = np.eye(3 if rotation is None else 6)
        q = self.clip(q0)
        err = self._pose_error(q, target, rotation)
        norm = float(np.linalg.norm(err))
        lam = damping
        for _ in range(iters):
            if np.linalg.norm(err[:3]) < tol_m and (rotation is None
                                                    or np.linalg.norm(err[3:]) < tol_rad):
                return q
            jac = self.jacobian(q)
            if rotation is None:
                jac = jac[:3]
            step = jac.T @ np.linalg.solve(jac @ jac.T + lam ** 2 * eye, err)
            big = float(np.max(np.abs(step)))
            if big > max_step_rad:
                step *= max_step_rad / big
            q_new = self.clip(q + step)
            err_new = self._pose_error(q_new, target, rotation)
            norm_new = float(np.linalg.norm(err_new))
            if norm_new < norm:
                q, err, norm = q_new, err_new, norm_new
                lam = max(lam * 0.5, 1e-3)
            else:
                lam = min(lam * 4.0, 10.0)
        return None
