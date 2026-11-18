"""The scene graph the policy reads (RoboBallet, Lai et al., Sci. Robot. 2025).

Nodes are robots, tasks and obstacle primitives. Edges carry relative poses and
point INTO robots (task → robot, obstacle → robot), plus robot ↔ robot both ways
for coordination, so the actions are read out of the robot nodes:

==================  =========================================================
robot node          joint position (normalised to the joint box), joint
                    velocity (/ limit), remaining dwell, DoF mask, and — our
                    addition — the joint offset to the arm's assigned goal
                    with a has-goal flag (all zero in free-allocation mode)
task node           solved flag
robot → robot       sender BASE pose relative to the receiver TIP
task → robot        task pose relative to the receiver TIP
obstacle → robot    centroid relative to the receiver TIP + rotation with each
                    basis vector scaled by the primitive's span
global              episode time / horizon, fraction of tasks solved
==================  =========================================================

Poses are a translation and the 6D rotation representation (two columns).

Deviation from the paper, on purpose: in **assigned** mode (what the backend
uses — the platform's allocator has already decided which arm does which weld)
a task only has an edge to the robot it is assigned to, and the robot node
carries the joint-space offset to that task's IK goal. Free allocation (the
paper's setting) is the same graph with every task → robot edge present and no
goal offset; the trainer mixes both.

Obstacle edges are the ``K`` primitives closest to each arm rather than all of
them: a voxelised fixture is hundreds of boxes, and the distant ones carry no
information an arm can act on.

Arrays are unbatched here; :func:`stack` pads a list of graphs into a batch.
numpy + stdlib only.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .chain import Chain
from .geometry import BoxSet

#: v2: obstacle spans capped (SPAN_CAP_M) and mean aggregation in the network —
#: v1 policies saturated (every action at +-1) and are refused, not adapted.
SCHEMA_VERSION = 2
#: A primitive's span enters the features capped: the floor is a 12 m slab, and a
#: raw 12 next to unit-scale features saturated the first layers.
SPAN_CAP_M = 2.0
MAX_DOF = 7
K_OBSTACLES = 24

ROBOT_DIM = 4 * MAX_DOF + 2          # q, qd, mask, goal offset ; dwell, has_goal
TASK_DIM = 1
RR_DIM = 9
TR_DIM = 9
OR_DIM = 12
GLOBAL_DIM = 2

DIMS = {"robot": ROBOT_DIM, "task": TASK_DIM, "rr": RR_DIM, "tr": TR_DIM,
        "or": OR_DIM, "global": GLOBAL_DIM, "max_dof": MAX_DOF}


def _rel(tip: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """``pose`` in the ``tip`` frame as translation + 6D rotation (9 values)."""
    inv_r = tip[:3, :3].T
    p = inv_r @ (pose[:3, 3] - tip[:3, 3])
    r = inv_r @ pose[:3, :3]
    return np.concatenate([p, r[:, 0], r[:, 1]])


def _pad(v: Sequence[float], n: int = MAX_DOF) -> np.ndarray:
    out = np.zeros(n)
    v = np.asarray(v, float)[:n]
    out[:len(v)] = v
    return out


def task_pose(task: dict) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = np.asarray(task["position"], float)
    t[:3, :3] = np.asarray(task.get("rotation", np.eye(3)), float)
    return t


def build(chains: Sequence[Chain], q: Sequence[np.ndarray], qd: Sequence[np.ndarray],
          dwell: Sequence[float], tasks: Sequence[dict], task_done: Sequence[bool],
          boxes: BoxSet, *, t: float, horizon: float,
          goal_q: Optional[Sequence[Optional[np.ndarray]]] = None,
          vel_limits: Optional[Sequence[np.ndarray]] = None,
          k_obstacles: int = K_OBSTACLES) -> dict:
    """One state → one graph (dict of arrays, no batch dimension).

    ``tasks[i]["robot"]`` (an index into ``chains``) assigns the task; ``None``
    or absent leaves it to whichever arm takes it.
    """
    n_r, n_t = len(chains), len(tasks)
    tips = [c.tip(qi) for c, qi in zip(chains, q)]
    robot = np.zeros((n_r, ROBOT_DIM))
    for i, c in enumerate(chains):
        span = np.maximum(c.upper - c.lower, 1e-6)
        qn = 2.0 * (np.asarray(q[i]) - c.lower) / span - 1.0
        vmax = np.asarray(vel_limits[i], float) if vel_limits is not None else np.ones(c.dof)
        g = goal_q[i] if goal_q is not None else None
        robot[i] = np.concatenate([
            _pad(qn), _pad(np.asarray(qd[i]) / np.maximum(vmax, 1e-6)), _pad(np.ones(c.dof)),
            _pad((np.asarray(g) - np.asarray(q[i])) / np.pi if g is not None else np.zeros(c.dof)),
            [float(dwell[i]), 1.0 if g is not None else 0.0]])

    task = np.array([[1.0 if d else 0.0] for d in task_done], float).reshape(n_t, TASK_DIM)

    rr = np.zeros((n_r, n_r, RR_DIM))
    rr_mask = np.ones((n_r, n_r)) - np.eye(n_r)
    for r in range(n_r):
        for s in range(n_r):
            if r != s:
                rr[r, s] = _rel(tips[r], chains[s].base)

    tr = np.zeros((n_r, n_t, TR_DIM))
    tr_mask = np.zeros((n_r, n_t))
    for k, tk in enumerate(tasks):
        pose = task_pose(tk)
        owner = tk.get("robot")
        for r in range(n_r):
            if owner is None or owner == r:
                tr[r, k] = _rel(tips[r], pose)
                tr_mask[r, k] = 1.0

    or_ = np.zeros((n_r, k_obstacles, OR_DIM))
    or_mask = np.zeros((n_r, k_obstacles))
    for r, c in enumerate(chains):
        idx = boxes.nearest(c.points(q[r]), k_obstacles)
        inv_r = tips[r][:3, :3].T
        for slot, b in enumerate(idx):
            p = inv_r @ (boxes.centers[b] - tips[r][:3, 3])
            rot = inv_r @ boxes.rot[b] * np.minimum(2.0 * boxes.half[b], SPAN_CAP_M)[None, :]
            or_[r, slot] = np.concatenate([p, rot.reshape(-1)])
            or_mask[r, slot] = 1.0

    done = float(np.mean(task_done)) if n_t else 1.0
    glob = np.array([t / max(horizon, 1e-6), done])
    return {"robot": robot, "robot_mask": np.ones(n_r), "task": task, "task_mask": np.ones(n_t),
            "rr": rr, "rr_mask": rr_mask, "tr": tr, "tr_mask": tr_mask,
            "or": or_, "or_mask": or_mask, "global": glob}


def stack(graphs: Sequence[dict]) -> dict:
    """Pad graphs with different robot / task counts into one batch (masks mark the padding)."""
    n_r = max(g["robot"].shape[0] for g in graphs)
    n_t = max(max(g["task"].shape[0] for g in graphs), 1)
    k = max(g["or"].shape[1] for g in graphs)
    b = len(graphs)
    out = {"robot": np.zeros((b, n_r, ROBOT_DIM)), "robot_mask": np.zeros((b, n_r)),
           "task": np.zeros((b, n_t, TASK_DIM)), "task_mask": np.zeros((b, n_t)),
           "rr": np.zeros((b, n_r, n_r, RR_DIM)), "rr_mask": np.zeros((b, n_r, n_r)),
           "tr": np.zeros((b, n_r, n_t, TR_DIM)), "tr_mask": np.zeros((b, n_r, n_t)),
           "or": np.zeros((b, n_r, k, OR_DIM)), "or_mask": np.zeros((b, n_r, k)),
           "global": np.zeros((b, GLOBAL_DIM))}
    for i, g in enumerate(graphs):
        r, t, kk = g["robot"].shape[0], g["task"].shape[0], g["or"].shape[1]
        out["robot"][i, :r] = g["robot"]
        out["robot_mask"][i, :r] = g["robot_mask"]
        out["task"][i, :t] = g["task"]
        out["task_mask"][i, :t] = g["task_mask"]
        out["rr"][i, :r, :r] = g["rr"]
        out["rr_mask"][i, :r, :r] = g["rr_mask"]
        out["tr"][i, :r, :t] = g["tr"]
        out["tr_mask"][i, :r, :t] = g["tr_mask"]
        out["or"][i, :r, :kk] = g["or"]
        out["or_mask"][i, :r, :kk] = g["or_mask"]
        out["global"][i] = g["global"]
    return out
