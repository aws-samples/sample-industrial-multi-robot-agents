"""Offline evaluation of a policy on ANY cell — the evaluation layout.

Training cells and evaluation cells are chosen independently: a policy trained
on ``welding_cell_2_small`` can be scored here on ``welding_cell_2_small`` (or a cell
it never saw) in the kinematic environment it was trained in, with that cell's
own robots, envelope, obstacles and weld points as declared.

Two modes, as the policy is used:

* ``free`` — the paper's setting: every weld standoff is a task, any arm may
  take any, every arm starts at HOME;
* ``assigned`` — the backend's setting: one weld per arm, its IK goal given,
  repeated over ``episodes`` random pairings.

This is the policy's own simulator, a quick and cheap score. The authoritative
evaluation is planning the cell's program with ``backend="roboballet"``, which
holds every leg to the platform's checks before it becomes a BehaviourTree.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from agents.policy_trainer.sagemaker.procedural import exact_assigned_episode, exact_episode
from agents.trajectory.backends.roboballet.core.policy_io import Policy
from agents.trajectory.backends.roboballet.core.sim import ChainChecker, Env


def _env(ep, policy: Policy, horizon_s: float, freeze: bool) -> Env:
    return Env(ep.chains, ep.boxes, vel_limits=ep.vel, acc_limits=ep.acc,
               checker=ChainChecker(ep.chains, ep.boxes), dt=policy.dt, horizon_s=horizon_s,
               tol_m=policy.tol_m, tol_rad=policy.tol_rad, freeze_on_solve=freeze,
               k_obstacles=policy.k_obstacles)


def _run(env: Env, policy: Policy, q0, tasks, goal) -> dict:
    obs = env.reset(q0, tasks, goal)
    held = 0
    while not env.finished:
        obs, _, _, info = env.step(policy.act(obs))
        held += info["n_colliding"]
    by_arm = {}
    for h0, h1 in zip(env.history, env.history[1:]):
        for i in range(len(env.chains)):
            by_arm[i] = by_arm.get(i, 0) + int(not np.allclose(h0["q"][i], h1["q"][i]))
    return {"solved": int(env.done_mask.sum()), "tasks": len(tasks),
            "fraction_solved": float(env.done_mask.mean()) if tasks else 1.0,
            "time_s": round(env.t, 2), "held_steps": held,
            "moving_steps_per_arm": [by_arm.get(i, 0) for i in range(len(env.chains))]}


def evaluate(policy: Policy, pack: dict, *, mode: str = "free", episodes: int = 3,
             max_tasks: Optional[int] = 8, horizon_s: float = 30.0, seed: int = 0) -> dict:
    if mode not in ("free", "assigned"):
        raise ValueError("mode must be 'free' or 'assigned'")
    rng = np.random.default_rng(seed)
    runs = []
    for e in range(max(1, int(episodes))):
        ep = exact_episode(pack, max_tasks=max_tasks, seed=seed + e)
        if mode == "free":
            runs.append(_run(_env(ep, policy, horizon_s, False), policy, ep.q0, ep.tasks, None))
            continue
        aep = exact_assigned_episode(pack, seed=int(rng.integers(1 << 30)))
        if aep is None:
            continue
        runs.append(_run(_env(aep, policy, horizon_s, True), policy, aep.q0, aep.tasks,
                         aep.goal_q))
    if not runs:
        return {"cell_id": pack["cell_id"], "mode": mode, "episodes": 0,
                "error": "no operation of this cell is reachable by IK from HOME"}
    return {"cell_id": pack["cell_id"], "policy": policy.name, "mode": mode,
            "episodes": len(runs),
            "mean_fraction_solved": round(float(np.mean([r["fraction_solved"] for r in runs])), 4),
            "mean_time_s": round(float(np.mean([r["time_s"] for r in runs])), 2),
            "mean_held_steps": round(float(np.mean([r["held_steps"] for r in runs])), 2),
            "trained_on": (policy.card.get("training") or {}).get("cells"),
            "unseen_cell": pack["cell_id"] not in ((policy.card.get("training") or {})
                                                   .get("cells") or []),
            "tolerance": {"m": policy.tol_m, "deg": round(math.degrees(policy.tol_rad), 1)},
            "runs": runs}
