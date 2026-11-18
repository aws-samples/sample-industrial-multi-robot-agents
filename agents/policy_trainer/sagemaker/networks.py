"""The RoboBallet graph network in PyTorch — the training twin of ``core/gnn.py``.

Same modules, same parameter names, same arithmetic: :func:`export` turns a
trained network into the ``{"<module>.<layer>.w": in x out}`` dict the numpy
forward pass runs, and ``tests/policy_trainer/test_network_parity.py`` holds the
two to the same outputs. Change one, change both.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn

try:                                   # inside the SageMaker bundle
    from roboballet_core.gnn import module_shapes
except ImportError:                    # in the repository
    from agents.trajectory.backends.roboballet.core.gnn import module_shapes


def _mean(x: torch.Tensor, mask: torch.Tensor, dim) -> torch.Tensor:
    """Masked mean, as ``core/gnn.py``'s."""
    return x.sum(dim) / mask.sum(dim).clamp(min=1.0)[..., None]


class MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class GraphNet(nn.Module):
    """Policy (``critic=False``) or Q-function (``critic=True``)."""

    def __init__(self, arch: Mapping, critic: bool = False):
        super().__init__()
        self.arch, self.critic = dict(arch), critic
        self.mods = nn.ModuleDict({name: MLP(dims)
                                   for name, dims in module_shapes(arch, critic).items()})

    def forward(self, g: Mapping[str, torch.Tensor], action: torch.Tensor | None = None,
                pre_tanh: bool = False):
        """Actions in [-1, 1] (policy), Q-values (critic) — or, with ``pre_tanh``,
        the policy head before its squash (for the trainer's saturation penalty)."""
        m = self.mods
        robot = g["robot"] if action is None else torch.cat([g["robot"], action], -1)
        rm, tm = g["robot_mask"], g["task_mask"]
        rr_m, tr_m, or_m = g["rr_mask"], g["tr_mask"], g["or_mask"]
        b, n_r = robot.shape[:2]
        n_t, k = g["task"].shape[1], g["or"].shape[2]

        hn = m["en_robot"](robot) * rm[..., None]
        ht = m["en_task"](g["task"]) * tm[..., None]
        e_rr = m["ee_rr"](g["rr"]) * rr_m[..., None]
        e_tr = m["ee_tr"](g["tr"]) * tr_m[..., None]
        e_or = m["ee_or"](g["or"]) * or_m[..., None]
        hg = m["eg"](g["global"])

        def bc(x, *shape):
            return x.reshape(x.shape[0], *([1] * len(shape)), x.shape[-1]).expand(
                x.shape[0], *shape, x.shape[-1])

        for r in range(int(self.arch["rounds"])):
            dn = hn.shape[-1]
            recv = hn[:, :, None, :]
            e_rr = e_rr + m[f"ue_rr{r}"](torch.cat(
                [hn[:, None, :, :].expand(b, n_r, n_r, dn), recv.expand(b, n_r, n_r, dn),
                 e_rr, bc(hg, n_r, n_r)], -1)) * rr_m[..., None]
            e_tr = e_tr + m[f"ue_tr{r}"](torch.cat(
                [ht[:, None, :, :].expand(b, n_r, n_t, ht.shape[-1]), recv.expand(b, n_r, n_t, dn),
                 e_tr, bc(hg, n_r, n_t)], -1)) * tr_m[..., None]
            e_or = e_or + m[f"ue_or{r}"](torch.cat(
                [recv.expand(b, n_r, k, dn), e_or, bc(hg, n_r, k)], -1)) * or_m[..., None]
            agg = _mean(e_rr, rr_m, 2) + _mean(e_tr, tr_m, 2) + _mean(e_or, or_m, 2)
            hn = hn + m[f"un{r}"](torch.cat([agg, hn, bc(hg, n_r)], -1)) * rm[..., None]
            edges = (_mean(e_rr, rr_m, (1, 2)) + _mean(e_tr, tr_m, (1, 2))
                     + _mean(e_or, or_m, (1, 2)))
            hg = hg + m[f"ug{r}"](torch.cat([_mean(hn, rm, 1), edges, hg], -1))

        if action is None:
            pre = m["head"](hn)
            return pre * rm[..., None] if pre_tanh else torch.tanh(pre) * rm[..., None]
        return m["head"](hg)[..., 0]


def export(net: GraphNet) -> dict[str, np.ndarray]:
    """Weights under the numpy forward pass's names (``w`` is ``in x out``)."""
    out = {}
    for name, mlp in net.mods.items():
        for i, layer in enumerate(mlp.layers):
            out[f"{name}.{i}.w"] = layer.weight.detach().cpu().numpy().T.copy()
            out[f"{name}.{i}.b"] = layer.bias.detach().cpu().numpy().copy()
    return out


def to_tensors(batch: Mapping[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}
