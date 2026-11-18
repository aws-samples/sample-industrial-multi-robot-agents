"""Procedurally generated training episodes, seeded by real cells.

RoboBallet trains on random workcells: robots slid along rails, 30 random
cuboids, tasks sampled on the cuboids' faces, a task kept only if some arm has a
collision-free IK solution for it. Here the randomisation starts from the
TRAINING cells' scene packs instead of from nothing, so the policy learns the
arms, envelopes and fixtures it will meet:

* **robots** — the cell's arms, each base jittered in x/y (``base_jitter_m``)
  and yaw (``yaw_jitter_deg``); a placement whose start is in collision is
  redrawn;
* **obstacles** — the cell's own primitives, each kept with ``keep_obstacle_p``
  (the floor always), plus up to ``random_obstacles`` random cuboids in the
  shared workspace that do not hit an arm at its start;
* **tasks** — the cell's own operation points (as standoff poses, jittered),
  mixed with poses sampled on obstacle faces, the tool pointing into the face
  within 22.5° of its normal; kept only when an arm has a collision-free IK
  solution;
* **mode** — *assigned* (probability ``assigned_p``): one task per arm with its
  IK goal, as the backend asks a step; *free*: up to ``max_tasks`` tasks any arm
  may take, the paper's setting.

The evaluation cells are never generated from: an evaluation episode uses its
cell exactly as declared (``exact_episode``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from roboballet_core.chain import Chain, axis_angle
    from roboballet_core.geometry import BoxSet
    from roboballet_core.sim import ChainChecker
except ImportError:
    from agents.trajectory.backends.roboballet.core.chain import Chain, axis_angle
    from agents.trajectory.backends.roboballet.core.geometry import BoxSet
    from agents.trajectory.backends.roboballet.core.sim import ChainChecker


@dataclass
class ProceduralConfig:
    base_jitter_m: float = 0.25
    yaw_jitter_deg: float = 5.0
    keep_obstacle_p: float = 0.8
    random_obstacles: int = 8
    obstacle_size_m: tuple = (0.08, 0.4)
    max_tasks: int = 10
    cell_task_p: float = 0.6
    task_jitter_m: float = 0.05
    standoff_m: tuple = (0.05, 0.15)
    max_tilt_deg: float = 22.5
    assigned_p: float = 0.7
    random_start_p: float = 0.5
    ik_restarts: int = 3
    max_draws: int = 40
    pair_min_m: float = 0.05
    extra: dict = field(default_factory=dict)


@dataclass
class Episode:
    pack_id: str
    chains: list
    boxes: BoxSet
    vel: list
    acc: list
    q0: list
    tasks: list
    goal_q: Optional[list]
    mode: str


def _rot_to(axis_tcp: np.ndarray, direction: np.ndarray, spin: float) -> np.ndarray:
    """A rotation taking the TCP's tool axis onto ``direction``, spun about it."""
    a = axis_tcp / np.linalg.norm(axis_tcp)
    d = direction / np.linalg.norm(direction)
    v, c = np.cross(a, d), float(a @ d)
    if np.linalg.norm(v) < 1e-9:
        base = np.eye(3) if c > 0 else axis_angle(np.array([1.0, 0, 0]) if abs(a[0]) < 0.9
                                                   else np.array([0, 1.0, 0]), math.pi)
    else:
        base = axis_angle(v, math.atan2(np.linalg.norm(v), c))
    return axis_angle(d, spin) @ base


def tool_axis(pack: dict) -> np.ndarray:
    """The tool axis in the TCP frame, as the cell's own operations use it."""
    axes = []
    for op in pack.get("operations") or []:
        r = np.asarray(op["rotation"], float)
        av = np.asarray(op.get("approach_vector", (0, 0, -1)), float)
        axes.append(r.T @ (av / (np.linalg.norm(av) + 1e-12)))
    if not axes:
        return np.array([0.0, 0.0, 1.0])
    m = np.mean(axes, axis=0)
    return m / (np.linalg.norm(m) + 1e-12)


def _standoff(op: dict, offset: float) -> dict:
    av = np.asarray(op.get("approach_vector", (0, 0, -1)), float)
    av /= np.linalg.norm(av) + 1e-12
    return {"position": (np.asarray(op["position"], float) - av * offset).tolist(),
            "rotation": op["rotation"]}


class Generator:
    def __init__(self, packs: list[dict], cfg: Optional[ProceduralConfig] = None, seed: int = 0):
        if not packs:
            raise ValueError("no training scene packs")
        self.packs, self.cfg = packs, cfg or ProceduralConfig()
        self.rng = np.random.default_rng(seed)
        self.axes = {p["cell_id"]: tool_axis(p) for p in packs}

    # ── pieces ───────────────────────────────────────────────────────────────
    def _chains(self, pack: dict, jitter: bool) -> list[Chain]:
        out = []
        for r in pack["robots"]:
            spec = dict(r["chain"])
            if jitter:
                spec["base"] = (np.asarray(spec["base"], float) + np.r_[
                    self.rng.uniform(-1, 1, 2) * self.cfg.base_jitter_m, 0.0]).tolist()
                spec["yaw"] = float(spec["yaw"]) + math.radians(
                    self.rng.uniform(-1, 1) * self.cfg.yaw_jitter_deg)
            out.append(Chain(spec))
        return out

    def _random_box(self, centre: np.ndarray, reach: float) -> dict:
        lo, hi = self.cfg.obstacle_size_m
        r = self.rng.uniform(0.3, 1.0) * reach
        ang = self.rng.uniform(0, 2 * math.pi)
        c = centre + np.array([r * math.cos(ang), r * math.sin(ang), self.rng.uniform(0.3, 1.8)])
        yaw = self.rng.uniform(0, math.pi)
        return {"id": "rand", "center": c.tolist(),
                "half_extents": (self.rng.uniform(lo, hi, 3) / 2).tolist(),
                "rotation": axis_angle(np.array([0, 0, 1.0]), yaw).tolist()}

    def _ik(self, chain: Chain, task: dict, checker: ChainChecker, i: int,
            seeds: list) -> Optional[np.ndarray]:
        for s in seeds[: self.cfg.ik_restarts]:
            q = chain.ik(task["position"], np.asarray(task["rotation"]), s)
            if q is not None and checker.env_free(i, q):
                return q
        return None

    def _surface_task(self, boxes: list[dict], axis: np.ndarray) -> Optional[dict]:
        cand = [b for b in boxes if max(b["half_extents"]) < 3.0]    # not the floor slab
        if not cand:
            return None
        b = cand[self.rng.integers(len(cand))]
        rot, half = np.asarray(b["rotation"]), np.asarray(b["half_extents"])
        face = self.rng.integers(3)
        sign = self.rng.choice([-1.0, 1.0])
        local = self.rng.uniform(-1, 1, 3) * half
        local[face] = sign * half[face]
        normal = rot[:, face] * sign
        tilt = math.radians(self.rng.uniform(0, self.cfg.max_tilt_deg))
        perp = np.cross(normal, self.rng.normal(size=3))
        inward = axis_angle(perp, tilt) @ (-normal)
        pos = np.asarray(b["center"]) + rot @ local + normal * self.rng.uniform(*self.cfg.standoff_m)
        return {"position": pos.tolist(),
                "rotation": _rot_to(axis, inward, self.rng.uniform(0, 2 * math.pi)).tolist()}

    # ── episodes ─────────────────────────────────────────────────────────────
    def episode(self) -> Episode:
        """A training episode with at least one task.

        An episode with no task is "solved" at its first step and teaches nothing,
        and drawing a new one each step starves the learner — so a draw that finds
        no reachable task is redrawn (a different placement and task set), and a
        cell that never yields one is refused rather than trained on.
        """
        for _ in range(self.cfg.max_draws):
            ep = self._draw()
            if ep.tasks:
                return ep
        raise RuntimeError(f"no reachable task in {self.cfg.max_draws} episode draws — "
                           "check the cells' operations and arms (export_scene)")

    def _draw(self) -> Episode:
        cfg = self.cfg
        pack = self.packs[self.rng.integers(len(self.packs))]
        homes = [np.asarray(r["home"], float) for r in pack["robots"]]
        for _ in range(cfg.max_draws):
            chains = self._chains(pack, jitter=True)
            boxes = [b for b in pack["obstacles"]
                     if max(b["half_extents"]) >= 3.0 or self.rng.random() < cfg.keep_obstacle_p]
            checker = ChainChecker(chains, BoxSet(boxes), cfg.pair_min_m)
            if all(checker.env_free(i, h) for i, h in enumerate(homes)):
                break
        else:
            raise RuntimeError(f"{pack['cell_id']}: no collision-free placement in "
                               f"{cfg.max_draws} draws")
        centre = np.mean([c.base[:3, 3] for c in chains], axis=0)
        reach = max(1.0, float(np.mean([np.linalg.norm(c.tip(h)[:3, 3] - c.base[:3, 3])
                                        for c, h in zip(chains, homes)])))
        for _ in range(self.rng.integers(0, cfg.random_obstacles + 1)):
            b = self._random_box(centre, reach)
            trial = ChainChecker(chains, BoxSet([b]), cfg.pair_min_m)
            if all(trial.env_free(i, h) for i, h in enumerate(homes)):
                boxes.append(b)
        box_set = BoxSet(boxes)
        checker = ChainChecker(chains, box_set, cfg.pair_min_m)

        # tasks with their IK goals, per arm
        axis = self.axes[pack["cell_id"]]
        tasks, sols = [], []
        ops = pack.get("operations") or []
        for _ in range(cfg.max_tasks * 3):
            if len(tasks) >= cfg.max_tasks:
                break
            if ops and self.rng.random() < cfg.cell_task_p:
                op = ops[self.rng.integers(len(ops))]
                t = _standoff(op, self.rng.uniform(*cfg.standoff_m))
                t["position"] = (np.asarray(t["position"]) +
                                 self.rng.uniform(-1, 1, 3) * cfg.task_jitter_m).tolist()
            else:
                t = self._surface_task(boxes, axis)
                if t is None:
                    continue
            per_arm = {i: self._ik(c, t, checker, i, [homes[i], c.clip(homes[i] +
                                   self.rng.normal(0, 0.5, c.dof)), c.clip(
                                   self.rng.uniform(c.lower, c.upper))])
                       for i, c in enumerate(chains)}
            per_arm = {i: q for i, q in per_arm.items() if q is not None}
            if per_arm:
                tasks.append(t)
                sols.append(per_arm)

        # start: HOME, or a reachable task's configuration (mid-program steps)
        q0 = []
        for i, c in enumerate(chains):
            starts = [s[i] for s in sols if i in s]
            q0.append(starts[self.rng.integers(len(starts))] if starts and
                      self.rng.random() < cfg.random_start_p else homes[i].copy())
        if not checker.pairs_free(q0):
            q0 = [h.copy() for h in homes]

        if self.rng.random() < cfg.assigned_p and tasks:
            chosen, goal, used = [], [None] * len(chains), set()
            for i in self.rng.permutation(len(chains)):
                options = [k for k, s in enumerate(sols) if i in s and k not in used]
                if not options:
                    continue
                k = options[self.rng.integers(len(options))]
                used.add(k)
                chosen.append({**tasks[k], "robot": int(i)})
                goal[i] = sols[k][i]
            # An arm that starts at its own goal solves its task without moving: a
            # free reward that inflated the training score. Such an arm starts at
            # HOME instead, or at another task's configuration.
            for i, g in enumerate(goal):
                if g is None or np.max(np.abs(np.asarray(q0[i]) - g)) > 0.2:
                    continue
                others = [sl[i] for k2, sl in enumerate(sols) if i in sl
                          and np.max(np.abs(sl[i] - g)) > 0.2]
                q0[i] = (others[self.rng.integers(len(others))] if others
                         and self.rng.random() < cfg.random_start_p else homes[i].copy())
                if np.max(np.abs(np.asarray(q0[i]) - g)) <= 0.2:
                    goal[i] = None
                    chosen = [t for t in chosen if t.get("robot") != i]
            if not checker.pairs_free(q0):
                q0 = [h.copy() for h in homes]
            return Episode(pack["cell_id"], chains, box_set,
                           [r["vel_limits"] for r in pack["robots"]],
                           [r["acc_limits"] for r in pack["robots"]], q0, chosen, goal, "assigned")
        return Episode(pack["cell_id"], chains, box_set, [r["vel_limits"] for r in pack["robots"]],
                       [r["acc_limits"] for r in pack["robots"]], q0, tasks, None, "free")


def exact_episode(pack: dict, *, offset_m: float = 0.1, max_tasks: Optional[int] = None,
                  seed: int = 0) -> Episode:
    """The evaluation cell as declared: no jitter, its own obstacles, its own
    operation points (as standoffs, free allocation), every arm starting at HOME."""
    rng = np.random.default_rng(seed)
    chains = [Chain(r["chain"]) for r in pack["robots"]]
    ops = list(pack.get("operations") or [])
    if max_tasks is not None and len(ops) > max_tasks:
        ops = [ops[i] for i in sorted(rng.choice(len(ops), max_tasks, replace=False))]
    return Episode(pack["cell_id"], chains, BoxSet(pack["obstacles"]),
                   [r["vel_limits"] for r in pack["robots"]],
                   [r["acc_limits"] for r in pack["robots"]],
                   [np.asarray(r["home"], float) for r in pack["robots"]],
                   [_standoff(op, offset_m) for op in ops], None, "free")


def exact_assigned_episode(pack: dict, *, offset_m: float = 0.1, seed: int = 0) -> Optional[Episode]:
    """The evaluation cell as declared, used as the BACKEND uses a policy: one of
    the cell's own welds per arm (a random reachable pairing), its standoff as the
    task and its IK solution as the goal, every arm starting at HOME. ``None``
    when no arm reaches any weld."""
    rng = np.random.default_rng(seed)
    ep = exact_episode(pack, offset_m=offset_m)
    checker = ChainChecker(ep.chains, ep.boxes)
    ops = list(pack.get("operations") or [])
    rng.shuffle(ops)
    tasks, goal, used = [], [None] * len(ep.chains), set()
    for i, c in enumerate(ep.chains):
        for k, op in enumerate(ops):
            if k in used:
                continue
            t = _standoff(op, offset_m)
            q = c.ik(t["position"], np.asarray(t["rotation"]), ep.q0[i])
            if q is not None and checker.env_free(i, q):
                used.add(k)
                tasks.append({**t, "robot": i})
                goal[i] = q
                break
    if not tasks:
        return None
    return Episode(ep.pack_id, ep.chains, ep.boxes, ep.vel, ep.acc, ep.q0, tasks, goal,
                   "assigned")
