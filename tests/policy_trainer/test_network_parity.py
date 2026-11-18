"""Training (PyTorch) and planning (numpy) run the SAME network.

The planner runs exported weights with ``core/gnn.py``; the trainer optimises
``sagemaker/networks.py``. Same inputs → same actions and Q-values, so what was
evaluated in training is what plans. Also: a tiny end-to-end training run on a
synthetic cell produces a policy the planner loads.

Needs torch (skipped without it): the trainer's image, or a dev machine.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.policy_trainer.sagemaker.networks import GraphNet, export, to_tensors  # noqa: E402
from agents.trajectory.backends.roboballet.core import graph as G  # noqa: E402
from agents.trajectory.backends.roboballet.core.chain import Chain  # noqa: E402
from agents.trajectory.backends.roboballet.core.geometry import BoxSet  # noqa: E402
from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH, forward  # noqa: E402
from agents.trajectory.backends.roboballet.core.policy_io import Policy  # noqa: E402
from tests.backends.roboballet.synthetic import pack  # noqa: E402

ARCH = {**DEFAULT_ARCH, "node": 16, "edge": 12, "global": 10, "depth": 3, "head": 8, "rounds": 2}


def _batch():
    p = pack()
    ch = [Chain(r["chain"]) for r in p["robots"]]
    rng = np.random.default_rng(0)
    graphs = []
    for n_tasks in (1, 3):
        q = [c.clip(rng.uniform(-1, 1, 3)) for c in ch]
        tasks = [{"position": rng.uniform(-1, 1, 3).tolist(), "rotation": np.eye(3).tolist(),
                  **({"robot": 0} if k == 0 else {})} for k in range(n_tasks)]
        graphs.append(G.build(ch, q, [rng.normal(size=3) for _ in ch], [0.0, 0.2], tasks,
                              [False] * n_tasks, BoxSet(p["obstacles"]), t=1.0, horizon=10.0,
                              goal_q=[np.zeros(3), None], k_obstacles=3))
    return G.stack(graphs)


@pytest.mark.parametrize("critic", [False, True])
def test_numpy_and_torch_agree(critic):
    torch.manual_seed(0)
    net = GraphNet(ARCH, critic=critic).double()
    g = _batch()
    act = np.random.default_rng(1).uniform(-1, 1, (2, g["robot"].shape[1], G.MAX_DOF))
    with torch.no_grad():
        t = net({k: torch.as_tensor(v, dtype=torch.float64) for k, v in g.items()},
                torch.as_tensor(act) if critic else None).numpy()
    n = forward(export(net), ARCH, g, act if critic else None)
    np.testing.assert_allclose(t, n, rtol=1e-9, atol=1e-9)


def test_a_tiny_training_run_produces_a_policy_the_planner_loads(tmp_path):
    from agents.policy_trainer.sagemaker import train
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    (scenes / "synthetic.json").write_text(json.dumps(pack(wrist=True)))
    rc = train.main(["--scenes", str(scenes), "--model-dir", str(tmp_path / "model"),
                     "--total-steps", "12", "--warmup", "8", "--n-envs", "1",
                     "--batch-size", "8", "--eval-every", "100", "--eval-episodes", "1",
                     "--eval-max-tasks", "1", "--horizon-s", "0.5", "--max-tasks", "2",
                     "--arch-node", "8", "--arch-edge", "8", "--arch-global", "8"])
    assert rc == 0
    pol = Policy.load(tmp_path / "model")
    assert pol.card["training"]["cells"] == ["synthetic"]
    g = _batch()
    assert pol.act({k: v[0] for k, v in g.items()}).shape[-1] == G.MAX_DOF
    assert (tmp_path / "model" / "best" / "policy.npz").is_file()
    _ = to_tensors  # exported for the trainer; keep the import honest
