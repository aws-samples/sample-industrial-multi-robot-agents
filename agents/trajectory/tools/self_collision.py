"""Self-collision of one arm — its links (and its tool) against each other.

The sphere model in :mod:`robot_kinematics` has no self term: one sphere per link
frame, and neighbouring links' spheres always overlap, so it cannot tell a wrist
folded into the forearm from a straight arm. OMPL knows no geometry at all (the
validity checker is ours), so nothing rejected a pose with the tool plate 51 mm
inside link_3 (umarell upstream issue 01, robot1 at W01 of welding_cell_2_small).

This module checks the arm's real collision geometry — the URDF ``<collision>``
meshes as convex hulls, the tool's meshes on the tip link included — with the
pairs that touch by design removed:

* ``self_collision.disabled_pairs`` from robots.yaml (link-name pairs, written by
  ``scripts/convert_isaac_scene.py``: adjacent links of the joint tree plus any the
  SRDF disables), else
* the SRDF beside the URDF, else
* the adjacent links of the model's own joint tree.

Pairs on one joint (a link and the tool meshes on it) are never checked.

Safety, not convenience, decides what an unavailable check means. A model that
cannot be built (no meshes, no coal) or that is in self-contact at HOME (a pair
that touches by design and is not disabled — the check would reject everything)
is UNAVAILABLE with a reason. When the robot's ``self_collision.required`` is true
(what the converter writes) planning refuses (``planner_backend.dispatch_plan``);
when it is false or the block is absent (cells converted before this existed) the
check is skipped and every report says so.

Hooked in through :func:`checker_for_model`: every arm model is loaded by
``utils.arm_model.load``, which registers it, so the shared validity predicate,
the IK branch choice, the emitted-trajectory backstop and the runtime monitor all
see the same check without each caller passing the cell and robot.

Dense checking (umarell issue 14). A leg is executed by interpolating linearly in
joint space between its samples, and the samples of a fast leg are far apart: a
home leg can move the wrist ~0.2 m between two 50 ms samples, so two clear
samples can have a self-contact between them.
:meth:`SelfCollisionChecker.trajectory_violation` walks the straight joint-space
segment between consecutive samples at a step bounded by joint motion: no point
of any link travels more than ``SELF_COLLISION_STEP_M`` (5 mm, env of the same
name) between two checks, and where the hulls are far apart the step grows by
conservative advancement (it can never skip over a contact). The per-joint bound
(:meth:`SelfCollisionChecker.lever_arms`) comes from the arm's own geometry, so
it holds for any arm.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

_LOCK = threading.Lock()
_CHECKERS: dict = {}
# Largest travel of any point of the arm between two checks of a leg (m).
SELF_COLLISION_STEP_M = 0.005
# Smallest step of the dense scan (m of point travel): contacts shallower than
# this at a grazing pass are below anything the hulls resolve.
_MIN_ADVANCE_M = 1e-4


def _step(override=None) -> float:
    """The dense-scan step: the argument, else ``SELF_COLLISION_STEP_M`` from the
    environment, else the default — a value that is not a positive finite number
    is the default (a zero step never advances, a bad string must not switch the
    check off)."""
    import math
    raw = override if override is not None else os.environ.get(
        "SELF_COLLISION_STEP_M", SELF_COLLISION_STEP_M)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return SELF_COLLISION_STEP_M
    return v if math.isfinite(v) and v > 0.0 else SELF_COLLISION_STEP_M
_TRAJ_CACHE_MAX = 1024


class SelfCollisionChecker:
    """Hull self-distance for one (cell, robot). Thread-safe."""

    def __init__(self, cell_id: str, robot_ns: str):
        self.cell_id, self.robot_ns = cell_id, robot_ns
        cfg = _robot_cfg(cell_id, robot_ns)
        sc = cfg.get("self_collision") or {}
        self.required = bool(sc.get("required", False))
        self.margin = float(sc.get("margin_m", 0.0) or 0.0)
        self.available = False
        self.reason = ""
        self.pair_source = None
        self._levers = None
        self._legs: "OrderedDict" = OrderedDict()      # verdicts per leg content
        # This checker's own Data/GeometryData: one lock per checker, so two arms
        # are checked concurrently (the module lock guards only the registry).
        self._lock = threading.Lock()
        try:
            import pinocchio as pin              # inside: no pinocchio = unavailable
            from utils import arm_model
            urdf = arm_model.urdf_path(cell_id, robot_ns)
            if urdf is None:
                raise FileNotFoundError(f"no URDF for {robot_ns} in {cell_id}")
            model = pin.buildModelFromUrdf(str(urdf))
            geom = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION,
                                         package_dirs=[str(urdf.parent)])
            if geom.ngeoms == 0:
                raise FileNotFoundError(f"{urdf.name} declares no <collision> geometry")
            for go in geom.geometryObjects:
                # Convex: GJK/EPA then returns a SIGNED distance (negative = depth).
                go.geometry.buildConvexRepresentation(False)
                go.geometry = go.geometry.convex
            geom.addAllCollisionPairs()            # pairs on different joints only
            self._filter_pairs(pin, model, geom, sc, urdf)
            self.model, self.geom = model, geom
            self.data, self.gdata = model.createData(), pin.GeometryData(geom)
            for req in self.gdata.collisionRequests:
                req.security_margin = self.margin
            self.links = [model.frames[go.parentFrame].name for go in geom.geometryObjects]
            self._home_must_be_clear()
            self.available = True
        except Exception as exc:  # noqa: BLE001 — reported, never silent
            self.reason = f"{type(exc).__name__}: {exc}"

    # ── pairs ────────────────────────────────────────────────────────────────
    def _filter_pairs(self, pin, model, geom, sc: dict, urdf) -> None:
        link_of = [model.frames[go.parentFrame].name for go in geom.geometryObjects]
        disabled = {frozenset(p) for p in (sc.get("disabled_pairs") or []) if len(p) == 2}
        if disabled:
            self.pair_source = "robots.yaml self_collision.disabled_pairs"
            drop = lambda cp: frozenset((link_of[cp.first], link_of[cp.second])) in disabled  # noqa: E731
        else:
            srdf = _srdf_beside(urdf)
            if srdf is not None:
                pin.removeCollisionPairs(model, geom, str(srdf))
                self.pair_source = f"SRDF {srdf.name}"
                return
            self.pair_source = "adjacent links of the joint tree"
            adjacent = {frozenset((model.parents[j], j)) for j in range(1, model.njoints)}
            drop = lambda cp: frozenset((geom.geometryObjects[cp.first].parentJoint,  # noqa: E731
                                         geom.geometryObjects[cp.second].parentJoint)) in adjacent
        keep = [cp for cp in geom.collisionPairs if not drop(cp)]
        geom.removeAllCollisionPairs()
        for cp in keep:
            geom.addCollisionPair(cp)

    def _home_must_be_clear(self) -> None:
        from agents.home_pose import home_pose
        q = home_pose(self.cell_id, self.robot_ns)
        hit = self.closest(q)
        if hit is not None and hit["clearance_m"] < self.margin:
            raise ValueError(
                f"{hit['pair'][0]} vs {hit['pair'][1]} is {hit['clearance_m']:.4f} m at HOME "
                f"(margin {self.margin} m): a pair that touches by design must be listed "
                f"in self_collision.disabled_pairs, or every pose is rejected")

    # ── queries ──────────────────────────────────────────────────────────────
    def _q(self, q) -> np.ndarray:
        qv = np.zeros(self.model.nq)
        qa = np.asarray(q, dtype=float)[: self.model.nq]
        qv[: len(qa)] = qa
        return qv

    def is_free(self, q) -> bool:
        """True when no checked pair is closer than the margin (fast path)."""
        import pinocchio as pin
        with self._lock:
            return not pin.computeCollisions(self.model, self.data, self.geom, self.gdata,
                                             self._q(q), True)

    def closest(self, q) -> Optional[dict]:
        """The closest checked pair and its signed distance."""
        import pinocchio as pin
        if not len(self.geom.collisionPairs):
            return None
        with self._lock:
            pin.computeDistances(self.model, self.data, self.geom, self.gdata, self._q(q))
            dists = [float(r.min_distance) for r in self.gdata.distanceResults]
        k = int(np.argmin(dists))
        cp = self.geom.collisionPairs[k]
        return {"pair": [self.links[cp.first], self.links[cp.second]],
                "clearance_m": round(dists[k], 4)}

    def distance(self, q) -> Optional[float]:
        """Signed distance of the closest checked pair at ``q`` (None: no pairs)."""
        hit = self.closest(q)
        return None if hit is None else float(hit["clearance_m"])

    def lever_arms(self) -> Optional[np.ndarray]:
        """Per joint, an upper bound (m) on the distance from that joint's axis to
        any point of the collision geometry it moves, valid in EVERY configuration.

        For joint j and a hull on joint k (k at or below j in the chain) the bound
        is the sum of the joint-origin offsets from j down to k plus the radius of
        the hull's bounding box about joint k's origin (triangle inequality). A
        joint-space step ``dq`` then moves no point of the arm by more than
        ``sum_j |dq_j| * R_j`` (revolute; a prismatic joint counts 1 m per unit).
        None when a joint is not 1-dof (not bounded here).
        """
        if self._levers is not None:
            return self._levers
        with self._lock:      # the AABB is computed on geometry other threads query
            return self._compute_levers()

    def _compute_levers(self) -> Optional[np.ndarray]:
        if self._levers is not None:
            return self._levers
        model, geom = self.model, self.geom
        radius_on: dict = {}
        for go in geom.geometryObjects:
            g = go.geometry
            g.computeLocalAABB()
            lo, hi = np.asarray(g.aabb_local.min_, float), np.asarray(g.aabb_local.max_, float)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                for z in (lo[2], hi[2])])
            R, t = np.asarray(go.placement.rotation), np.asarray(go.placement.translation)
            r = float(np.max(np.linalg.norm(corners @ R.T + t, axis=1)))
            k = int(go.parentJoint)
            radius_on[k] = max(radius_on.get(k, 0.0), r)
        levers = np.zeros(model.nq)
        for j in range(1, model.njoints):
            jm = model.joints[j]
            if jm.nq != 1:
                return None
            if "P" in jm.shortname().replace("JointModel", ""):
                levers[jm.idx_q] = 1.0
                continue
            best = 0.0
            for k, r in radius_on.items():
                chain, i = 0.0, k
                while i not in (j, 0):
                    chain += float(np.linalg.norm(model.jointPlacements[i].translation))
                    i = int(model.parents[i])
                if i == j:
                    best = max(best, chain + r)
            levers[jm.idx_q] = best
        self._levers = levers
        return levers

    def trajectory_violation(self, samples: list, dense: bool = True,
                             max_step_m: Optional[float] = None) -> Optional[dict]:
        """First pose along the executed leg closer than the margin, or None.

        ``dense`` (default) walks the straight joint-space segment between samples
        (see the module docstring); the step grows to ``(d - margin) / 2`` of point
        travel where the distance ``d`` allows it (two bodies close at most twice
        as fast as either moves), so far-apart stretches cost one check. The result
        is ``violation``-shaped plus ``sample_index`` (start of the segment),
        ``segment_fraction`` and the deepest point (``worst_*``). Verdicts are
        cached per leg content, so repeated gates on the same leg are free.
        """
        if not samples or not len(self.geom.collisionPairs):
            return None
        step = _step(max_step_m)
        rows = [(pt.get("positions") if isinstance(pt, dict) else pt) for pt in samples]
        n = min(len(r) for r in rows)
        Q = np.asarray([[float(v) for v in r[:n]] for r in rows], float)
        key = (bool(dense), step, Q.shape,
               hashlib.sha1(np.ascontiguousarray(Q).tobytes()).hexdigest())
        with self._lock:
            if key in self._legs:
                self._legs.move_to_end(key)
                hit = self._legs[key]
                return None if hit is None else dict(hit)
        info = self._scan(Q, dense, step)
        if info is not None:
            pt = samples[info["sample_index"]]
            info["phase"] = pt.get("phase", "transit") if isinstance(pt, dict) else None
        with self._lock:
            self._legs[key] = None if info is None else dict(info)
            while len(self._legs) > _TRAJ_CACHE_MAX:
                self._legs.popitem(last=False)
        return info

    def _scan(self, Q: np.ndarray, dense: bool, step: float) -> Optional[dict]:
        levers = self.lever_arms() if dense else None
        if dense and levers is None:
            dense = False                       # cannot bound motion: samples only
        thr = self.margin
        state = {"first": None, "worst": None, "checks": 0}

        def check(q, i, s):
            hit = self.closest(q)
            d, pair = float(hit["clearance_m"]), hit["pair"]
            state["checks"] += 1
            if d < thr:
                rec = (d, int(i), round(float(s), 4), pair, q)
                if state["first"] is None:
                    state["first"] = rec
                if state["worst"] is None or d < state["worst"][0]:
                    state["worst"] = rec
            return d

        d0 = check(Q[0], 0, 0.0)
        for i in range(len(Q) - 1):
            qa, qb = Q[i], Q[i + 1]
            if dense:
                n = min(len(levers), len(qa))
                travel = float(np.abs(qb[:n] - qa[:n]) @ levers[:n])   # m per unit s
                s, d = 0.0, d0
                while travel > 0.0:
                    # Conservative advancement: two bodies close at most twice as
                    # fast as either point moves, so a point travel of d/2 cannot
                    # reach a contact. Above the margin, step by what the margin
                    # allows (or `step`), but never more than d/2 — the margin is
                    # detected at `step` resolution, a CONTACT is never stepped
                    # over. The floor keeps a near-tangent pass finite.
                    if d > thr:
                        adv = max(0.5 * (d - thr), min(step, 0.5 * d))
                    else:
                        adv = step
                    s += max(adv, _MIN_ADVANCE_M) / travel
                    if s >= 1.0:
                        break
                    d = check(qa + s * (qb - qa), i, s)
            d0 = check(qb, i + 1, 0.0)
        if state["first"] is None:
            return None
        d, i, s, pair, q = state["first"]
        wd, wi, ws, wpair, _wq = state["worst"]
        return {"type": "self", "object_id": f"{pair[0]} vs {pair[1]}",
                "link_index": None, "clearance_m": round(d, 4), "margin_m": thr,
                "sample_index": i, "segment_fraction": s,
                "positions": [round(float(v), 6) for v in q],
                "worst_clearance_m": round(wd, 4), "worst_sample_index": wi,
                "worst_segment_fraction": ws, "worst_pair": wpair,
                "dense": bool(dense), "checks": state["checks"]}

    def violation(self, q) -> Optional[dict]:
        """``_config_collision_info``-shaped breach, or None when clear."""
        if self.is_free(q):
            return None
        hit = self.closest(q) or {"pair": ["?", "?"], "clearance_m": 0.0}
        return {"type": "self", "object_id": f"{hit['pair'][0]} vs {hit['pair'][1]}",
                "link_index": None, "clearance_m": hit["clearance_m"],
                "margin_m": self.margin}

    def status(self) -> dict:
        return {"robot_ns": self.robot_ns, "available": self.available,
                "required": self.required, "reason": self.reason or None,
                "margin_m": self.margin, "pairs": self.pair_source}


def _robot_cfg(cell_id: str, robot_ns: str) -> dict:
    try:
        from agents.trajectory.tools.robot_kinematics import _load_robot_config
        return _load_robot_config(cell_id, robot_ns) or {}
    except Exception:  # noqa: BLE001 — no config: defaults (not required)
        return {}


def _srdf_beside(urdf):
    found = sorted(urdf.parent.glob("*.srdf"))
    return found[0] if found else None


def checker(cell_id: str, robot_ns: str) -> SelfCollisionChecker:
    """The (cached) checker for one robot of a cell. Never raises.

    Keyed on the cell's files as well (``leg_verdict.cell_stamp``): a re-converted
    cell or an edited URDF gets a fresh checker instead of the hulls it replaced."""
    from agents.trajectory.tools.leg_verdict import cell_stamp
    name = (str(cell_id), str(robot_ns))
    key = (*name, cell_stamp(cell_id, robot_ns) or ())
    with _LOCK:
        hit = _CHECKERS.get(key)
    if hit is None:
        hit = SelfCollisionChecker(*name)
        with _LOCK:
            for stale in [k for k in _CHECKERS if k[:2] == name and k != key]:
                del _CHECKERS[stale]
            hit = _CHECKERS.setdefault(key, hit)
    return hit


def checker_for_model(model) -> Optional[SelfCollisionChecker]:
    """The usable checker of a model loaded by ``utils.arm_model.load``, or None
    (unregistered model, or the check is unavailable for that robot)."""
    from utils import arm_model
    owner = arm_model.owner_of(model)
    if owner is None:
        return None
    c = checker(*owner)
    return c if c.available else None


def refusal(cell_id: str, robot_ns: str) -> Optional[str]:
    """Why planning for this robot must be refused, or None.

    Only a REQUIRED check that is unavailable refuses: planning on without it
    would ship poses nobody checked against the arm itself.
    """
    c = checker(cell_id, robot_ns)
    if c.required and not c.available:
        return (f"self-collision check for {robot_ns} is required but unavailable "
                f"({c.reason}). Provide the arm's collision meshes, or set "
                f"self_collision.required: false in robots.yaml to plan without it.")
    return None


def clear_cache() -> None:
    with _LOCK:
        _CHECKERS.clear()
