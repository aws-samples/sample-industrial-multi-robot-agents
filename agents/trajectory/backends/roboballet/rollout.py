"""The policy as a compound-step transit search.

``ompl-compound``'s pipeline hands a search the step's moving arms, where their
transit starts (the end of each lift-off) and where it must end (each arm's
standoff above its weld, in joint space). This module answers with a policy
rollout:

1. **The episode.** One task per arm — its standoff POSE, assigned to it — and
   the joint goal as the graph's goal offset (assigned mode, see
   ``core/graph.py``). The arms start at the transit start.
2. **The platform's checker.** The environment holds an arm whose next step
   would collide, and "collide" is the step pipeline's own predicate: each
   arm's ``is_free`` (static obstacles at the transit margin + the parked
   bystanders) and the arm-to-arm sphere clearance the compound search uses.
   The policy's own envelope only shapes its features.
3. **Freeze on arrival.** An arm within the policy's tolerance of its standoff
   (2.5 cm / 15° by default) stops there; the rollout ends when all have.
4. **Docking.** The paper reaches poses to a tolerance and post-processes;
   here the remaining joint offset to the exact goal is closed with a straight
   joint segment, checked like every other sample. A rollout that arrives in a
   different arm configuration than the descent was planned from
   (``ROBOBALLET_MAX_DOCK_RAD``) is a refusal, not a long docking move.

The result is a stacked path; the pipeline times it on one clock and checks it
again as emitted. A rollout that does not arrive is ``no_path_found`` and names
how far it got.
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional, Sequence

import numpy as np

from agents.trajectory.backends.roboballet.core.chain import Chain
from agents.trajectory.backends.roboballet.core.geometry import BoxSet, pair_clearance
from agents.trajectory.backends.roboballet.core.policy_io import Policy
from agents.trajectory.backends.roboballet.core.sim import Checker, Env

#: Arm-to-arm sphere radius, as ompl-compound's PAIR_RADIUS_M.
PAIR_RADIUS_M = 0.12


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class PlatformChecker(Checker):
    """The step pipeline's own collision predicate, in the environment's terms."""

    def __init__(self, arms: Sequence[dict], pair_min_m: float, radius: float = PAIR_RADIUS_M):
        self.arms, self.pair_min_m, self.radius = list(arms), float(pair_min_m), float(radius)

    def env_free(self, i: int, q: np.ndarray) -> bool:
        return bool(self.arms[i]["is_free"](np.asarray(q, float)))

    def points(self, i: int, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.arms[i]["spheres"](np.asarray(q, float)), float).reshape(-1, 3)

    def state_free(self, qs: Sequence[np.ndarray]) -> bool:
        if not all(self.env_free(i, q) for i, q in enumerate(qs)):
            return False
        pts = [self.points(i, q) for i, q in enumerate(qs)]
        return all(pair_clearance(pts[i], pts[j], self.radius) > self.pair_min_m
                   for i in range(len(pts)) for j in range(i + 1, len(pts)))


def _dedupe(path: list[list[float]]) -> list[list[float]]:
    out = [path[0]]
    for v in path[1:]:
        if np.max(np.abs(np.asarray(v) - np.asarray(out[-1]))) > 1e-9:
            out.append(v)
    return out


def make_transit_solver(policy: Policy, chains: dict[str, Chain], boxes: BoxSet, *,
                        accel_limits: dict[str, Sequence[float]], info: dict) -> Callable:
    """A ``transit_solver`` for ``plan_trajectories_ompl_compound``.

    ``chains`` / ``accel_limits`` are per robot namespace; ``info`` collects what
    the rollout did, for the planning result.
    """

    def solve(arms: list, starts: list, goals: list, *, pair_min_m: float,
              time_budget_s: float, cell_id: str = "") -> tuple[Optional[list], Optional[dict]]:
        names = [a["ns"] for a in arms]
        missing = [n for n in names if n not in chains]
        if missing:
            return None, {"reason_code": "planner_error",
                          "reason": f"no kinematic chain for {', '.join(missing)} in {cell_id}"}
        ch = [chains[n] for n in names]
        checker = PlatformChecker(arms, pair_min_m)
        starts = [np.asarray(s, float) for s in starts]
        goals = [np.asarray(g, float) for g in goals]
        for label, qs in (("start", starts), ("goal", goals)):
            if not checker.state_free(qs):
                return None, {"reason_code": f"compound_{label}_invalid",
                              "reason": (f"the joint {label} of the moving arms is itself in "
                                         f"collision or closer than {pair_min_m:.3f} m "
                                         "arm-to-arm — no rollout can begin/end there")}
        tasks = []
        for i, (c, g) in enumerate(zip(ch, goals)):
            tip = c.tip(g)
            tasks.append({"position": tip[:3, 3].tolist(), "rotation": tip[:3, :3].tolist(),
                          "robot": i})
        horizon = _env_float("ROBOBALLET_HORIZON_S", 30.0)
        env = Env(ch, boxes, vel_limits=[a.get("vel") or [1.0] * a["nq"] for a in arms],
                  acc_limits=[accel_limits.get(n) or [3.0] * a["nq"] for n, a in zip(names, arms)],
                  checker=checker, dt=policy.dt, horizon_s=horizon, tol_m=policy.tol_m,
                  tol_rad=policy.tol_rad, freeze_on_solve=True, collision_penalty=0.0,
                  k_obstacles=policy.k_obstacles)
        obs = env.reset(starts, tasks, goals)
        t0, held = time.monotonic(), 0
        deadline = t0 + max(float(time_budget_s), 1.0)
        while not env.finished and time.monotonic() < deadline:
            obs, _, _, step_info = env.step(policy.act(obs))
            held += step_info["n_colliding"]
        info.update({"policy": policy.name, "steps": len(env.history) - 1,
                     "rollout_s": round(env.t, 2), "held_steps": held,
                     "arrived": int(env.done_mask.sum()), "arms": len(arms),
                     "inference_s": round(time.monotonic() - t0, 3)})
        if not env.done_mask.all():
            late = [names[i] for i in range(len(arms)) if not env.done_mask[i]]
            why = "time budget" if time.monotonic() >= deadline else f"{horizon:.0f} s horizon"
            return None, {"reason_code": "no_path_found",
                          "reason": (f"policy '{policy.name}' did not bring {', '.join(late)} "
                                     f"to the standoff within the {why} "
                                     f"({int(env.done_mask.sum())}/{len(arms)} arrived, "
                                     f"{held} held steps)")}

        path = [np.concatenate(h["q"]).tolist() for h in env.history]
        # Docking: close the remaining joint offset to the exact goal.
        end = env.history[-1]["q"]
        gap = max(float(np.max(np.abs(g - q))) for g, q in zip(goals, end))
        limit = _env_float("ROBOBALLET_MAX_DOCK_RAD", 0.5)
        if gap > limit:
            info["docking_rad"] = round(gap, 4)
            return None, {"reason_code": "no_path_found",
                          "reason": (f"policy '{policy.name}' reached the standoff poses in an "
                                     f"arm configuration {gap:.2f} rad from the planned one "
                                     f"(limit {limit:.2f}) — the descent was planned from the "
                                     "other; re-plan with another backend or retrain")}
        n = max(1, int(math.ceil(gap / 0.02)))
        for s in range(1, n + 1):
            qs = [q + (g - q) * (s / n) for q, g in zip(end, goals)]
            if not checker.state_free(qs):
                return None, {"reason_code": "no_path_found",
                              "reason": (f"policy '{policy.name}' arrived, but the "
                                         f"{gap:.3f} rad docking move to the exact standoff "
                                         "configuration is blocked")}
            path.append(np.concatenate(qs).tolist())
        info["docking_rad"] = round(gap, 4)
        return _dedupe(path), None

    return solve
