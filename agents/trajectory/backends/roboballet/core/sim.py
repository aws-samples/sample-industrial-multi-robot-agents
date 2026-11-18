"""The kinematic environment RoboBallet is trained in, and planned with.

Each step advances ``dt`` (0.1 s): the action is a commanded joint velocity for
every joint of every arm, clamped to the velocity and acceleration limits and
integrated. An arm whose motion would collide — with the environment, with
itself or with another arm — is held still for that step, and the step is
penalised per arm held. The environment never enters collision; the policy
learns not to ask it to.

A task is solved when a permitted arm's TCP is within ``tol_m`` / ``tol_rad`` of
its pose; that arm then dwells (frozen) for ``dwell_s``. ``freeze_on_solve``
keeps it there for good, which is how the backend uses a step: each arm reaches
its standoff and waits for the others.

Collision is a pluggable :class:`Checker`. Training uses
:class:`ChainChecker` (the envelope from the scene pack); the backend passes the
platform's own predicate, so the arm the policy is held to is the checker the
plan will be validated with.

numpy + stdlib only.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from . import graph as G
from .chain import Chain, rotation_angle
from .geometry import BoxSet, pair_clearance


class Checker:
    """What the environment asks about collision."""

    pair_min_m: float = 0.0

    def env_free(self, i: int, q: np.ndarray) -> bool:   # noqa: ARG002
        return True

    def points(self, i: int, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    radius: float = 0.12


class ChainChecker(Checker):
    """Envelope spheres of :class:`Chain` against oriented boxes and each other."""

    def __init__(self, chains: Sequence[Chain], boxes: BoxSet, pair_min_m: float = 0.05):
        self.chains, self.boxes, self.pair_min_m = list(chains), boxes, float(pair_min_m)
        self.radius = max(c.sphere_radius for c in self.chains) if self.chains else 0.12

    def points(self, i: int, q: np.ndarray) -> np.ndarray:
        return self.chains[i].points(q)

    def pairs_free(self, qs: Sequence[np.ndarray]) -> bool:
        """Every pair of arms keeps ``pair_min_m`` of surface clearance at ``qs``."""
        pts = [self.points(i, q) for i, q in enumerate(qs)]
        return all(pair_clearance(pts[i], pts[j], self.radius) > self.pair_min_m
                   for i in range(len(pts)) for j in range(i + 1, len(pts)))

    def env_free(self, i: int, q: np.ndarray) -> bool:
        # The base stands on whatever carries it: only the moving links are checked.
        pts = self.chains[i].points(q, moving_only=True)
        return not self.boxes.collides(pts, self.chains[i].sphere_radius)


def potential(q: Sequence[np.ndarray], tasks: Sequence[dict], done: Sequence[bool],
              goal_q: Optional[Sequence[Optional[Sequence[float]]]]) -> float:
    """Shaping potential: minus every assigned arm's normalised joint distance to
    its goal, for arms whose task is not solved yet (0 when nothing is assigned).

    Used as potential-based shaping, ``F = Phi(s') - Phi(s)`` (Ng, Harada &
    Russell 1999): it rewards progress towards the goal without changing which
    policy is optimal. Undiscounted on purpose: with ``Phi <= 0`` the discounted
    ``gamma * Phi(s') - Phi(s)`` pays ``(1 - gamma) * distance`` for standing still
    — a reward for exactly the immobility the learner must avoid.
    """
    if not goal_q:
        return 0.0
    total = 0.0
    for i, g in enumerate(goal_q):
        if g is None:
            continue
        mine = [k for k, t in enumerate(tasks) if t.get("robot") == i]
        if mine and all(done[k] for k in mine):
            continue
        g = np.asarray(g, float)
        total -= float(np.linalg.norm(g - np.asarray(q[i], float))) / (math.pi * math.sqrt(len(g)))
    return total


class Env:
    def __init__(self, chains: Sequence[Chain], boxes: BoxSet, *,
                 vel_limits: Sequence[Sequence[float]], acc_limits: Sequence[Sequence[float]],
                 checker: Optional[Checker] = None, dt: float = 0.1, horizon_s: float = 30.0,
                 tol_m: float = 0.025, tol_rad: float = math.radians(15.0),
                 dwell_s: float = 0.5, collision_penalty: float = 15.0,
                 freeze_on_solve: bool = False, check_step_rad: float = 0.05,
                 k_obstacles: int = G.K_OBSTACLES, shaping: float = 0.0, gamma: float = 0.94):
        self.chains, self.boxes = list(chains), boxes
        self.vmax = [np.asarray(v, float) for v in vel_limits]
        self.amax = [np.asarray(a, float) for a in acc_limits]
        self.checker = checker or ChainChecker(self.chains, boxes)
        self.dt, self.horizon = float(dt), float(horizon_s)
        self.tol_m, self.tol_rad, self.dwell_s = float(tol_m), float(tol_rad), float(dwell_s)
        self.penalty, self.freeze_on_solve = float(collision_penalty), bool(freeze_on_solve)
        self.check_step = float(check_step_rad)
        self.k_obstacles = int(k_obstacles)
        self.shaping, self.gamma = float(shaping), float(gamma)

    # ── episode ──────────────────────────────────────────────────────────────
    def reset(self, q0: Sequence[Sequence[float]], tasks: Sequence[dict],
              goal_q: Optional[Sequence[Optional[Sequence[float]]]] = None) -> dict:
        self.q = [np.asarray(q, float).copy() for q in q0]
        self.qd = [np.zeros(c.dof) for c in self.chains]
        self.dwell = np.zeros(len(self.chains))
        self.tasks = list(tasks)
        self.done_mask = np.zeros(len(self.tasks), bool)
        self.goal_q = (list(goal_q) if goal_q is not None else [None] * len(self.chains))
        self.t = 0.0
        self.history = [self._snapshot(0)]
        return self.observe()

    def _snapshot(self, n_col: int) -> dict:
        return {"q": [q.copy() for q in self.q], "qd": [v.copy() for v in self.qd],
                "dwell": self.dwell.copy(), "t": self.t, "n_colliding": n_col,
                "done": self.done_mask.copy()}

    def observe(self) -> dict:
        return G.build(self.chains, self.q, self.qd, self.dwell, self.tasks, self.done_mask,
                       self.boxes, t=self.t, horizon=self.horizon, goal_q=self.goal_q,
                       vel_limits=self.vmax, k_obstacles=self.k_obstacles)

    def score(self) -> float:
        return float(self.done_mask.mean()) if len(self.tasks) else 1.0

    @property
    def finished(self) -> bool:
        return bool(self.done_mask.all()) or self.t >= self.horizon - 1e-9

    # ── dynamics ─────────────────────────────────────────────────────────────
    def _motion_free(self, i: int, a: np.ndarray, b: np.ndarray) -> bool:
        n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / self.check_step)))
        return all(self.checker.env_free(i, a + (b - a) * (s / n)) for s in range(1, n + 1))

    def _pairs_free(self, i: int, qi: np.ndarray, others: Sequence[np.ndarray]) -> bool:
        pi = self.checker.points(i, qi)
        for j, qj in enumerate(others):
            if j != i and pair_clearance(pi, self.checker.points(j, qj),
                                         self.checker.radius) <= self.checker.pair_min_m:
                return False
        return True

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Apply ``(R, MAX_DOF)`` actions in [-1, 1] → ``(graph, reward, done, info)``."""
        before = self.score()
        phi = potential(self.q, self.tasks, self.done_mask, self.goal_q) if self.shaping else 0.0
        cand = []
        for i, c in enumerate(self.chains):
            if self.dwell[i] > 0.0:
                cand.append(self.q[i].copy())
                continue
            v = np.clip(np.asarray(action[i][:c.dof], float), -1, 1) * self.vmax[i]
            dv = np.clip(v - self.qd[i], -self.amax[i] * self.dt, self.amax[i] * self.dt)
            cand.append(c.clip(self.q[i] + (self.qd[i] + dv) * self.dt))
        moving = [i for i in range(len(self.chains)) if not np.allclose(cand[i], self.q[i])]
        held: set[int] = {i for i in moving if not self._motion_free(i, self.q[i], cand[i])}
        # Arm-to-arm: hold the colliding arm (the later one of a pair) until the set
        # is consistent. Held arms return to where they were, which was free.
        for _ in range(len(self.chains) + 1):
            pos = [self.q[i] if i in held else cand[i] for i in range(len(self.chains))]
            bad = {i for i in moving if i not in held and not self._pairs_free(i, pos[i], pos)}
            if not bad:
                break
            held |= {max(bad)} if len(bad) > 1 else bad
        n_col = len(held)
        for i in range(len(self.chains)):
            new = self.q[i] if i in held else cand[i]
            self.qd[i] = (new - self.q[i]) / self.dt
            self.q[i] = new
        self.t += self.dt
        self.dwell = np.maximum(self.dwell - self.dt, 0.0)

        solved = []
        for i, c in enumerate(self.chains):
            if self.dwell[i] > 0.0:
                continue
            tip = c.tip(self.q[i])
            for k, tk in enumerate(self.tasks):
                if self.done_mask[k] or tk.get("robot") not in (None, i):
                    continue
                if np.linalg.norm(tip[:3, 3] - np.asarray(tk["position"])) <= self.tol_m and \
                        rotation_angle(tip[:3, :3], np.asarray(tk.get("rotation", np.eye(3)))) \
                        <= self.tol_rad:
                    self.done_mask[k] = True
                    self.dwell[i] = self.horizon * 10 if self.freeze_on_solve else self.dwell_s
                    self.qd[i] = np.zeros(c.dof)
                    solved.append((i, k))
                    break
        # Paper Eqs. 2-4: score gain minus C_col per arm the environment had to hold.
        reward = (self.score() - before) - self.penalty * n_col
        if self.shaping:
            reward += self.shaping * (potential(self.q, self.tasks, self.done_mask,
                                                self.goal_q) - phi)
        self.history.append(self._snapshot(n_col))
        return self.observe(), reward, self.finished, {"n_colliding": n_col, "solved": solved}


def rollout(env: Env, act, q0, tasks, goal_q=None, max_steps: Optional[int] = None) -> dict:
    """Run a policy (``act(graph) -> (R, MAX_DOF)``) to the end of an episode."""
    obs = env.reset(q0, tasks, goal_q)
    steps = max_steps or int(math.ceil(env.horizon / env.dt))
    held = 0
    for _ in range(steps):
        obs, _, done, info = env.step(act(obs))
        held += info["n_colliding"]
        if done:
            break
    return {"solved": int(env.done_mask.sum()), "tasks": len(env.tasks), "time_s": env.t,
            "held_steps": held, "success": bool(env.done_mask.all()),
            "path": [h["q"] for h in env.history]}
