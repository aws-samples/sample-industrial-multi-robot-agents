"""The training signal: the defects that kept the first policies from learning.

Each was measured on a real run (a policy that never solved a task, or every
action saturated at +-1); these hold the fixes in place, fast.

* IK converges from HOME on most reachable poses (it converged on none, so
  episodes had almost no tasks).
* An episode always has a task (empty ones "solved" instantly and starved the
  learner), and no arm starts at its own goal (a free solve inflated the score).
* Moving towards the goal is rewarded and standing still earns nothing
  (undiscounted potential-based shaping), and a held step is not punished out of scale with progress.
* Checkpoints are scored as the backend uses the policy: assigned, with goals.
* Saturated actions are penalised during training (``action_reg``).
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.policy_trainer.sagemaker.hyperparameters import DEFAULTS
from agents.policy_trainer.sagemaker.procedural import (
    Generator, ProceduralConfig, exact_assigned_episode,
)
from agents.trajectory.backends.roboballet.core import graph as G
from agents.trajectory.backends.roboballet.core.chain import Chain
from agents.trajectory.backends.roboballet.core.geometry import BoxSet
from agents.trajectory.backends.roboballet.core.sim import Env
from tests.backends.roboballet.synthetic import pack


def test_ik_converges_from_home_on_most_reachable_poses(monkeypatch):
    """On a real cell's arm (the KUKA of welding_cell_2_small, tracked in the repo)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "cells"
    if not (root / "welding_cell_2_small").is_dir():
        pytest.skip("welding_cell_2_small not in this checkout")
    monkeypatch.setenv("CELL_BASE_PATH", str(root))
    monkeypatch.setenv("CELL_DATA_MODE", "local")
    from agents.trajectory.backends.roboballet.cell_scene import scene_pack
    r = scene_pack("welding_cell_2_small", with_operations=False)["robots"][0]
    c = Chain(r["chain"])
    rng = np.random.default_rng(1)
    ok = 0
    for _ in range(20):
        t = c.tip(rng.uniform(c.lower, c.upper))
        seeds = [r["home"]] + [rng.uniform(c.lower, c.upper) for _ in range(2)]
        ok += any(c.ik(t[:3, 3], t[:3, :3], s) is not None for s in seeds)
    assert ok >= 16, ok          # it was 0 of 40 with the fixed-damping solver


def test_the_jacobian_is_the_derivative_of_fk():
    c = Chain(pack(wrist=True)["robots"][0]["chain"])
    q = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    jac, t0, eps = c.jacobian(q), c.tip(q), 1e-6
    for i in range(c.dof):
        dq = q.copy()
        dq[i] += eps
        np.testing.assert_allclose(jac[:3, i], (c.tip(dq)[:3, 3] - t0[:3, 3]) / eps, atol=1e-4)


def test_episodes_have_tasks_and_no_arm_starts_at_its_goal():
    gen = Generator([pack(gap=2.4, wrist=True)], ProceduralConfig(max_tasks=2, assigned_p=1.0),
                    seed=1)
    for _ in range(15):
        ep = gen.episode()
        assert ep.tasks
        for i, g in enumerate(ep.goal_q or []):
            if g is not None:
                assert np.max(np.abs(np.asarray(ep.q0[i]) - g)) > 0.2


def test_a_cell_with_no_reachable_task_is_refused_not_trained_on():
    p = pack(gap=2.4)                       # 3-axis arms: standoff poses unreachable
    with pytest.raises(RuntimeError, match="no reachable task"):
        Generator([p], ProceduralConfig(max_tasks=2, max_draws=3), seed=0).episode()


def _env(**kw):
    p = pack(gap=2.4, wrist=True)
    ch = [Chain(r["chain"]) for r in p["robots"]]
    return Env(ch, BoxSet(p["obstacles"]), vel_limits=[[2.0] * 6] * 2,
               acc_limits=[[50.0] * 6] * 2, horizon_s=3.0, **kw), ch


def test_progress_is_rewarded_and_standing_still_is_not():
    env, ch = _env(shaping=1.0, collision_penalty=DEFAULTS["collision_penalty"])
    q0 = np.zeros(6)
    goal = np.array([0.5, -0.3, 0.4, 0.0, 0.2, 0.0])
    tip = ch[0].tip(goal)
    task = [{"position": tip[:3, 3].tolist(), "rotation": tip[:3, :3].tolist(), "robot": 0}]
    toward = np.zeros((2, G.MAX_DOF))
    toward[0, :6] = np.sign(goal - q0) * 0.3
    env.reset([q0, q0.copy()], task, [goal, None])
    _, r_move, _, _ = env.step(toward)
    env.reset([q0, q0.copy()], task, [goal, None])
    _, r_still, _, _ = env.step(np.zeros((2, G.MAX_DOF)))
    assert r_move > 0.0
    assert r_still == pytest.approx(0.0)    # never rewarded for not moving


def test_a_held_step_is_not_punished_out_of_scale_with_progress():
    # The paper's 15 per held arm made "never move" the best policy here.
    assert DEFAULTS["collision_penalty"] <= 0.1
    assert DEFAULTS["shaping"] > 0 and DEFAULTS["action_reg"] > 0


def test_evaluation_is_assigned_like_the_backend():
    ep = exact_assigned_episode(pack(gap=2.4, wrist=True), seed=0)
    assert ep is not None and ep.mode == "assigned"
    assert all(t.get("robot") is not None for t in ep.tasks)
    assert sum(g is not None for g in ep.goal_q) == len(ep.tasks)


def test_the_saturation_penalty_reaches_the_actor():
    torch = pytest.importorskip("torch")
    from agents.policy_trainer.sagemaker.networks import GraphNet, to_tensors
    from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH
    arch = {**DEFAULT_ARCH, "node": 8, "edge": 8, "global": 8, "depth": 2, "head": 8}
    net = GraphNet(arch)
    env, _ = _env()
    obs = env.reset([np.zeros(6), np.zeros(6)], [])
    g = to_tensors(G.stack([obs]), "cpu")
    pre = net(g, pre_tanh=True)
    assert torch.allclose(torch.tanh(pre), net(g))
    (pre ** 2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
