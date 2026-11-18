"""The RoboBallet core: kinematics, primitives, the graph and the network.

* The chain's FK places the TCP where the URDF says, in the world frame of the
  arm's base pose; its IK reaches what FK says is reachable.
* Obstacles from obstacles.yaml become oriented boxes with the margin added.
* The graph has the paper's shape: edges INTO robots, tasks only to their owner
  in assigned mode, the K nearest primitives per arm.
* The network is permutation-equivariant over arms (renaming arms renames the
  actions), its actions are bounded, and padding is inert.
* A card for another graph schema is refused.

Host-runnable: numpy only.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.trajectory.backends.roboballet.core import graph as G
from agents.trajectory.backends.roboballet.core.chain import Chain
from agents.trajectory.backends.roboballet.core.geometry import BoxSet, box_from_obstacle
from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH, forward, init_params
from agents.trajectory.backends.roboballet.core.policy_io import Policy, PolicyError, save
from agents.trajectory.backends.roboballet.core.sim import ChainChecker, Env
from tests.backends.roboballet.synthetic import pack

SMALL = {**DEFAULT_ARCH, "node": 16, "edge": 16, "global": 16, "depth": 2, "head": 8}


def _chains(p):
    return [Chain(r["chain"]) for r in p["robots"]]


def test_fk_matches_the_urdf_and_the_base_pose():
    a, b = _chains(pack(gap=2.0))
    # straight arm: base 0.3 + 0.2 up, then 0.6 + 0.5 along x
    np.testing.assert_allclose(a.tip([0, 0, 0])[:3, 3], [1.1, 0, 0.5], atol=1e-9)
    # the second arm faces the first (yaw = pi) from x = 2
    np.testing.assert_allclose(b.tip([0, 0, 0])[:3, 3], [0.9, 0, 0.5], atol=1e-9)


def test_ik_reaches_what_fk_reached():
    (a, _) = _chains(pack())
    q = np.array([0.4, -0.5, 0.9])
    t = a.tip(q)
    sol = a.ik(t[:3, 3], None, [0.0, -0.3, 0.6])
    assert sol is not None
    np.testing.assert_allclose(a.tip(sol)[:3, 3], t[:3, 3], atol=2e-3)


def test_obstacles_become_boxes_with_their_margin():
    b = box_from_obstacle({"type": "box", "id": "x", "center": [0, 0, 1],
                           "half_extents": [0.1, 0.2, 0.3], "collision_margin": 0.05})
    assert b["half_extents"] == pytest.approx([0.15, 0.25, 0.35])
    s = box_from_obstacle({"type": "sphere", "id": "s", "center": [0, 0, 0], "radius": 0.2})
    assert s["half_extents"] == [0.2, 0.2, 0.2]
    f = box_from_obstacle({"type": "plane", "id": "floor", "normal": [0, 0, 1], "offset": 0.0})
    top = np.asarray(f["center"]) + np.asarray(f["rotation"]) @ [0, 0, f["half_extents"][2]]
    assert top[2] == pytest.approx(0.0)
    assert box_from_obstacle({"type": "mesh"}) is None


def test_box_distances():
    bs = BoxSet([{"id": "b", "center": [0, 0, 0], "half_extents": [1, 1, 1],
                  "rotation": np.eye(3).tolist()}])
    d = bs.distances(np.array([[0, 0, 0], [2, 0, 0], [2, 2, 1]], float))
    np.testing.assert_allclose(d[:, 0], [0, 1, np.sqrt(2)])


def _graph(p, tasks, goal=None):
    ch = _chains(p)
    q = [np.asarray(r["home"]) for r in p["robots"]]
    return G.build(ch, q, [np.zeros(3)] * 2, [0, 0], tasks, [False] * len(tasks),
                   BoxSet(p["obstacles"]), t=0.0, horizon=10.0, goal_q=goal, k_obstacles=4)


def test_the_graph_shape_follows_the_paper():
    p = pack()
    tasks = [{"position": [0.7, 0.3, 0.3], "rotation": np.eye(3).tolist(), "robot": 0},
             {"position": [1.3, -0.3, 0.3], "rotation": np.eye(3).tolist()}]
    g = _graph(p, tasks, goal=[np.zeros(3), None])
    assert g["robot"].shape == (2, G.ROBOT_DIM) and g["rr"].shape == (2, 2, G.RR_DIM)
    assert g["rr_mask"].tolist() == [[0, 1], [1, 0]]          # no self edge
    assert g["tr_mask"].tolist() == [[1, 1], [0, 1]]          # task 0 is robot 0's only
    assert g["or_mask"].sum(1).tolist() == [2, 2]             # 2 primitives, K = 4
    assert g["robot"][0, -1] == 1.0 and g["robot"][1, -1] == 0.0   # has-goal flag


def test_the_network_is_equivariant_over_arms_and_bounded():
    p = pack()
    tasks = [{"position": [0.7, 0.3, 0.3], "rotation": np.eye(3).tolist()}]
    g = G.stack([_graph(p, tasks)])
    params = init_params(SMALL, seed=3)
    a = forward(params, SMALL, g)
    perm = {k: (v[:, ::-1] if k in ("robot", "robot_mask", "tr", "tr_mask", "or", "or_mask")
                else v[:, ::-1, ::-1] if k in ("rr", "rr_mask") else v) for k, v in g.items()}
    b = forward(params, SMALL, perm)
    np.testing.assert_allclose(a[:, ::-1], b, atol=1e-9)
    assert np.all(np.abs(a) <= 1.0)
    q = forward(init_params(SMALL, critic=True, seed=4), SMALL, g, action=a)
    assert q.shape == (1,)


def test_padding_is_inert():
    p = pack()
    t1 = [{"position": [0.7, 0.3, 0.3], "rotation": np.eye(3).tolist()}]
    g1 = _graph(p, t1)
    g2 = _graph(p, t1 + t1 + t1)
    params = init_params(SMALL, seed=5)
    alone = forward(params, SMALL, G.stack([g1]))
    padded = forward(params, SMALL, G.stack([g1, g2]))[:1]
    np.testing.assert_allclose(alone, padded, atol=1e-9)


def test_a_policy_for_another_schema_is_refused(tmp_path):
    save(tmp_path / "ok", init_params(SMALL), {"arch": SMALL})
    assert Policy.load(tmp_path / "ok").act(_graph(pack(), [])).shape == (2, G.MAX_DOF)
    import json
    card = json.loads((tmp_path / "ok" / "policy_card.json").read_text())
    card["schema_version"] = 99
    (tmp_path / "ok" / "policy_card.json").write_text(json.dumps(card))
    with pytest.raises(PolicyError, match="schema"):
        Policy.load(tmp_path / "ok")


def test_the_environment_holds_an_arm_instead_of_colliding():
    p = pack(gap=1.6)                       # arms that can reach each other
    ch = _chains(p)
    boxes = BoxSet(p["obstacles"])
    env = Env(ch, boxes, vel_limits=[[2] * 3] * 2, acc_limits=[[50] * 3] * 2,
              checker=ChainChecker(ch, boxes, pair_min_m=0.02), horizon_s=3.0)
    folded = np.array([0.0, -1.0, 2.0])
    env.reset([folded, folded.copy()], [])
    assert env.checker.pairs_free(env.q)
    held = 0
    for _ in range(20):                     # both unfold, towards each other
        _, r, _, info = env.step(np.zeros((2, G.MAX_DOF)) + np.array([0, .4, -.6, 0, 0, 0, 0]))
        held += info["n_colliding"]
        assert env.checker.pairs_free(env.q)          # never in collision
        assert r == pytest.approx(-env.penalty * info["n_colliding"])
    assert held > 0                                   # ... because it held them
    assert not np.allclose(env.q[0], folded)          # ... after they moved
