"""The RoboBallet graph network, forward pass in numpy (inference only).

Training happens in PyTorch (``agents/policy_trainer/sagemaker/networks.py``),
which exports its weights under the parameter names used here; a parity test
holds the two forward passes to the same numbers. The planner therefore needs
no deep-learning framework, only numpy.

Architecture (paper Eqs. 7–13), one graph-network core applied ``rounds`` times:

* embeddings ``En`` (robot, task), ``Ee`` (per edge type), ``Eg``;
* edge update ``Ue``: sender ⊕ receiver ⊕ edge ⊕ global → edge, per edge type
  (obstacle nodes carry no features, so obstacle edges have no sender term);
* node update ``Un``: incoming edges ⊕ node ⊕ global → robot node;
* global update ``Ug``: robot nodes ⊕ edges ⊕ global → global;

  aggregated by MEAN per edge type (the paper sums): with 24 obstacle edges per
  arm a sum is 24 times the scale of a mean, and the first trained actors
  saturated on it (every action at +-1);
* policy head ``P_pi`` on every robot node → joint velocities in [-1, 1];
  critic head ``P_Q`` on the global → Q(s, a), the action concatenated to the
  robot node features before embedding.

All three updates are residual. Parameter names: ``"<module>.<layer>.w"``
(``in x out``) and ``"<module>.<layer>.b"``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .graph import DIMS

#: Default sizes. The paper's are larger (MLP(512, 7) node/global, MLP(256, 6)
#: edge, MLP(64, 2) heads); a job may ask for them through its hyperparameters.
DEFAULT_ARCH = {"node": 256, "edge": 128, "global": 128, "depth": 3, "head": 64,
                "head_depth": 2, "rounds": 2}


def module_shapes(arch: Mapping, critic: bool = False) -> dict[str, list[int]]:
    """``{module: [in, hidden..., out]}`` for the network ``arch`` describes."""
    dn, de, dg = int(arch["node"]), int(arch["edge"]), int(arch["global"])
    depth, head, head_depth = int(arch["depth"]), int(arch["head"]), int(arch["head_depth"])
    hid = lambda width: [width] * max(0, depth - 1)  # noqa: E731
    robot_in = DIMS["robot"] + (DIMS["max_dof"] if critic else 0)
    shapes = {
        "en_robot": [robot_in, *hid(dn), dn],
        "en_task": [DIMS["task"], *hid(dn), dn],
        "ee_rr": [DIMS["rr"], *hid(de), de],
        "ee_tr": [DIMS["tr"], *hid(de), de],
        "ee_or": [DIMS["or"], *hid(de), de],
        "eg": [DIMS["global"], *hid(dg), dg],
    }
    for k in range(int(arch["rounds"])):
        shapes[f"ue_rr{k}"] = [2 * dn + de + dg, *hid(de), de]
        shapes[f"ue_tr{k}"] = [2 * dn + de + dg, *hid(de), de]
        shapes[f"ue_or{k}"] = [dn + de + dg, *hid(de), de]
        shapes[f"un{k}"] = [de + dn + dg, *hid(dn), dn]
        shapes[f"ug{k}"] = [dn + de + dg, *hid(dg), dg]
    shapes["head"] = ([dg, *[head] * max(0, head_depth - 1), 1] if critic
                      else [dn, *[head] * max(0, head_depth - 1), DIMS["max_dof"]])
    return shapes


def init_params(arch: Mapping, critic: bool = False, seed: int = 0) -> dict[str, np.ndarray]:
    """Random parameters (tests and smoke runs; training initialises in torch)."""
    rng = np.random.default_rng(seed)
    params = {}
    for name, dims in module_shapes(arch, critic).items():
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            params[f"{name}.{i}.w"] = rng.normal(0, np.sqrt(2.0 / a), (a, b))
            params[f"{name}.{i}.b"] = np.zeros(b)
    return params


def _mlp(params: Mapping, name: str, x: np.ndarray) -> np.ndarray:
    i = 0
    while f"{name}.{i}.w" in params:
        x = x @ params[f"{name}.{i}.w"] + params[f"{name}.{i}.b"]
        if f"{name}.{i + 1}.w" in params:
            x = np.maximum(x, 0.0)
        i += 1
    if i == 0:
        raise KeyError(f"policy has no module {name!r}")
    return x


def _bcast(g: np.ndarray, *shape: int) -> np.ndarray:
    """Global ``(B, D)`` broadcast to ``(B, *shape, D)``."""
    return np.broadcast_to(g.reshape(g.shape[0], *([1] * len(shape)), g.shape[-1]),
                           (g.shape[0], *shape, g.shape[-1]))


def _mean(x: np.ndarray, mask: np.ndarray, axis) -> np.ndarray:
    """Masked mean of ``x`` (already zero where masked) over ``axis``."""
    return x.sum(axis) / np.maximum(mask.sum(axis), 1.0)[..., None]


def forward(params: Mapping, arch: Mapping, g: Mapping, action: np.ndarray | None = None):
    """Batched forward pass. Policy: ``(B, R, MAX_DOF)`` actions in [-1, 1].
    Critic (``action`` given): ``(B,)`` Q-values."""
    robot = g["robot"] if action is None else np.concatenate([g["robot"], action], -1)
    rm, tm = g["robot_mask"], g["task_mask"]
    rr_m, tr_m, or_m = g["rr_mask"], g["tr_mask"], g["or_mask"]
    b, n_r = robot.shape[:2]
    n_t, k = g["task"].shape[1], g["or"].shape[2]

    hn = _mlp(params, "en_robot", robot) * rm[..., None]
    ht = _mlp(params, "en_task", g["task"]) * tm[..., None]
    e_rr = _mlp(params, "ee_rr", g["rr"]) * rr_m[..., None]
    e_tr = _mlp(params, "ee_tr", g["tr"]) * tr_m[..., None]
    e_or = _mlp(params, "ee_or", g["or"]) * or_m[..., None]
    hg = _mlp(params, "eg", g["global"])

    for r in range(int(arch["rounds"])):
        # edges indexed [batch, receiver, sender]
        recv_rr = np.broadcast_to(hn[:, :, None, :], (b, n_r, n_r, hn.shape[-1]))
        send_rr = np.broadcast_to(hn[:, None, :, :], (b, n_r, n_r, hn.shape[-1]))
        e_rr = e_rr + _mlp(params, f"ue_rr{r}", np.concatenate(
            [send_rr, recv_rr, e_rr, _bcast(hg, n_r, n_r)], -1)) * rr_m[..., None]
        recv_tr = np.broadcast_to(hn[:, :, None, :], (b, n_r, n_t, hn.shape[-1]))
        send_tr = np.broadcast_to(ht[:, None, :, :], (b, n_r, n_t, ht.shape[-1]))
        e_tr = e_tr + _mlp(params, f"ue_tr{r}", np.concatenate(
            [send_tr, recv_tr, e_tr, _bcast(hg, n_r, n_t)], -1)) * tr_m[..., None]
        recv_or = np.broadcast_to(hn[:, :, None, :], (b, n_r, k, hn.shape[-1]))
        e_or = e_or + _mlp(params, f"ue_or{r}", np.concatenate(
            [recv_or, e_or, _bcast(hg, n_r, k)], -1)) * or_m[..., None]

        agg = _mean(e_rr, rr_m, 2) + _mean(e_tr, tr_m, 2) + _mean(e_or, or_m, 2)
        hn = hn + _mlp(params, f"un{r}", np.concatenate([agg, hn, _bcast(hg, n_r)], -1)) \
            * rm[..., None]
        edges = _mean(e_rr, rr_m, (1, 2)) + _mean(e_tr, tr_m, (1, 2)) + _mean(e_or, or_m, (1, 2))
        hg = hg + _mlp(params, f"ug{r}", np.concatenate([_mean(hn, rm, 1), edges, hg], -1))

    if action is None:
        return np.tanh(_mlp(params, "head", hn)) * rm[..., None]
    return _mlp(params, "head", hg)[..., 0]
