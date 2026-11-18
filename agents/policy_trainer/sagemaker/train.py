"""SageMaker training entry point: RoboBallet (GNN policy, TD3 + HER).

Runs in a SageMaker PyTorch training container in script mode, or locally for a
smoke run::

    python -m agents.policy_trainer.sagemaker.train \\
        --scenes .tmp/scenes --model-dir .tmp/policy --total-steps 200

Inputs (SageMaker channels):
  ``train``  — scene packs (``*.json``) of the TRAINING cells: the generator
               randomises them (:mod:`procedural`);
  ``eval``   — optional scene packs of the EVALUATION cells, used exactly as
               declared to score checkpoints. These may be other layouts
               entirely: generalisation to an unseen cell is the point.

Checkpoints: every ``checkpoint_every`` steps the whole learner state — actor,
critics, their targets, both optimisers, the step, the RNG states — is written
atomically to ``checkpoint.pt`` in the checkpoint directory (``--checkpoint-dir``;
on SageMaker ``/opt/ml/checkpoints``, which SageMaker syncs to S3). ``--resume``
continues from it: the replay buffer is not saved (it is gigabytes), so a resumed
run refills it with the CURRENT policy before updating again.

Output: ``policy.npz`` + ``policy_card.json`` in ``SM_MODEL_DIR`` (SageMaker
tars it to ``model.tar.gz`` in S3), plus ``best/`` — the checkpoint with the
best evaluation score.

The algorithm follows the paper's Materials and Methods:

* kinematic environment, 0.1 s steps, arms held when they would collide
  (``core/sim.py``), reward = Δ fraction solved − C_col · arms held, plus
  potential-based shaping towards each assigned goal (``shaping``);
* TD3 with twin critics, target-policy smoothing, decoupled Polyak constants for
  actor and critics, no delayed actor update, exponentially decaying
  exploration noise and learning rates, and a small penalty on the actor's
  pre-tanh output against saturation (``action_reg``);

Where the defaults differ from the paper's, ``hyperparameters.py`` says why
(measured). ``action_abs_mean`` near 1.0 in the metrics is a saturated actor.
* hindsight experience replay: every episode that leaves tasks unsolved is also
  stored as an imaginary episode in which a random subset of those tasks sits on
  the arms' own trajectory (``her_relabel``).

Metrics are printed as ``[metric] name=value`` lines for the job's
``MetricDefinitions``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import roboballet_core  # noqa: F401 — present inside the SageMaker bundle
    from roboballet_core import graph as G, policy_io
    from roboballet_core.gnn import DEFAULT_ARCH
    from roboballet_core.sim import ChainChecker, Env, potential
    from roboballet_core.chain import rotation_angle
    from hyperparameters import DEFAULTS                          # type: ignore
    from procedural import (Episode, Generator, ProceduralConfig,  # type: ignore
                            exact_assigned_episode, exact_episode)
    _BUNDLE = True
except ImportError:
    from agents.trajectory.backends.roboballet.core import graph as G, policy_io
    from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH
    from agents.trajectory.backends.roboballet.core.sim import ChainChecker, Env, potential
    from agents.trajectory.backends.roboballet.core.chain import rotation_angle
    from agents.policy_trainer.sagemaker.hyperparameters import DEFAULTS
    from agents.policy_trainer.sagemaker.procedural import (
        Episode, Generator, ProceduralConfig, exact_assigned_episode, exact_episode,
    )
    _BUNDLE = False


def _networks():
    """The PyTorch half, imported only when training (not to read the defaults)."""
    if _BUNDLE:
        import networks                                            # type: ignore
        return networks
    from agents.policy_trainer.sagemaker import networks
    return networks


def log_metric(name: str, value: float, step: int) -> None:
    print(f"[metric] {name}={value:.6g} step={step}", flush=True)


def make_env(ep: Episode, hp: dict) -> Env:
    return Env(ep.chains, ep.boxes, vel_limits=ep.vel, acc_limits=ep.acc,
               checker=ChainChecker(ep.chains, ep.boxes), dt=hp["dt"],
               horizon_s=hp["horizon_s"], tol_m=hp["tol_m"], tol_rad=math.radians(hp["tol_deg"]),
               dwell_s=hp["dwell_s"], collision_penalty=hp["collision_penalty"],
               k_obstacles=hp["k_obstacles"], shaping=hp["shaping"], gamma=hp["gamma"])


# ─── hindsight experience replay ─────────────────────────────────────────────

def her_relabel(env: Env, ep: Episode, actions: list, rng: np.random.Generator) -> list:
    """An imaginary episode in which unsolved tasks lie on the arms' own paths.

    The recorded joint trajectory is replayed (not re-simulated), so the arms'
    dwell after a relabelled solve is not re-enacted — an approximation the
    paper's description leaves open. Assigned tasks are moved only along their
    own arm's path, and their goal becomes that configuration.
    """
    hist = env.history
    unsolved = [k for k in range(len(env.tasks)) if not env.done_mask[k]]
    if not unsolved or len(hist) < 3:
        return []
    pick = rng.choice(unsolved, size=rng.integers(1, len(unsolved) + 1), replace=False)
    tasks = [dict(t) for t in env.tasks]
    goal = list(env.goal_q)
    for k in pick:
        owner = tasks[k].get("robot")
        arm = int(owner) if owner is not None else int(rng.integers(len(env.chains)))
        s = int(rng.integers(1, len(hist)))
        tip = env.chains[arm].tip(hist[s]["q"][arm])
        tasks[k] = {**tasks[k], "position": tip[:3, 3].tolist(), "rotation": tip[:3, :3].tolist()}
        if owner is not None:
            goal[arm] = hist[s]["q"][arm].copy()
    done = np.zeros(len(tasks), bool)
    out, prev_graph, prev_score, prev_phi = [], None, None, 0.0
    for s, h in enumerate(hist):
        for i, c in enumerate(env.chains):
            tip = c.tip(h["q"][i])
            for k, t in enumerate(tasks):
                if not done[k] and t.get("robot") in (None, i) and \
                        np.linalg.norm(tip[:3, 3] - np.asarray(t["position"])) <= env.tol_m and \
                        rotation_angle(tip[:3, :3], np.asarray(t.get("rotation", np.eye(3)))) \
                        <= env.tol_rad:
                    done[k] = True
        gr = G.build(env.chains, h["q"], h["qd"], h["dwell"], tasks, done, env.boxes,
                     t=h["t"], horizon=env.horizon, goal_q=goal, vel_limits=env.vmax,
                     k_obstacles=env.k_obstacles)
        score = float(done.mean())
        phi = potential(h["q"], tasks, done, goal) if env.shaping else 0.0
        if prev_graph is not None:
            r = (score - prev_score) - env.penalty * h["n_colliding"]
            if env.shaping:
                r += env.shaping * (phi - prev_phi)
            out.append((prev_graph, actions[s - 1], r, gr, bool(done.all() or s == len(hist) - 1)))
        prev_graph, prev_score, prev_phi = gr, score, phi
        if done.all():
            break
    return out


# ─── TD3 ─────────────────────────────────────────────────────────────────────

class Replay:
    def __init__(self, size: int, seed: int):
        self.buf: deque = deque(maxlen=size)
        self.rng = random.Random(seed)

    def add(self, item) -> None:
        self.buf.append(item)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, n: int):
        items = self.rng.sample(self.buf, min(n, len(self.buf)))
        s, a, r, s2, d = zip(*items)
        n_r = max(x["robot"].shape[0] for x in s)
        act = np.zeros((len(a), n_r, G.MAX_DOF))
        for i, x in enumerate(a):
            act[i, :x.shape[0]] = x
        return G.stack(s), act, np.asarray(r, float), G.stack(s2), np.asarray(d, float)


CHECKPOINT = "checkpoint.pt"


def default_checkpoint_dir(model_dir: Path) -> Path:
    """SageMaker's synced checkpoint path when it exists, else beside the model."""
    sm = Path("/opt/ml/checkpoints")
    return sm if sm.is_dir() else Path(model_dir).parent / "checkpoint"


def train(hp: dict, train_packs: list[dict], eval_packs: list[dict], model_dir: Path,
          job: str = "", checkpoint_dir: Optional[Path] = None, resume: bool = False) -> dict:
    import torch
    threads = os.environ.get("TRAINER_THREADS")
    if threads:
        # Several local jobs share one machine: each gets its share of the cores.
        torch.set_num_threads(max(1, int(threads)))
    nets = _networks()
    GraphNet, export, to_tensors = nets.GraphNet, nets.export, nets.to_tensors

    torch.manual_seed(hp["seed"])
    rng = np.random.default_rng(hp["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = {k: int(hp.get(f"arch_{k}", v)) for k, v in DEFAULT_ARCH.items()}
    actor, actor_t = GraphNet(arch).to(dev), GraphNet(arch).to(dev)
    critics = [GraphNet(arch, critic=True).to(dev) for _ in range(2)]
    critics_t = [GraphNet(arch, critic=True).to(dev) for _ in range(2)]
    actor_t.load_state_dict(actor.state_dict())
    for c, ct in zip(critics, critics_t):
        ct.load_state_dict(c.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=hp["lr_policy"])
    opt_c = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=hp["lr_critic"])
    gen = Generator(train_packs, ProceduralConfig(max_tasks=int(hp["max_tasks"]),
                                                  assigned_p=float(hp["assigned_p"])),
                    seed=int(hp["seed"]))
    replay = Replay(int(hp["replay_size"]), int(hp["seed"]))
    # Checkpoints are scored the way the backend uses the policy — ASSIGNED, one
    # weld per arm with its goal (``eval_fraction_solved``) — and, when training
    # also covers free allocation, the paper's way too (``eval_free_fraction_solved``).
    packs_for_eval = eval_packs or train_packs
    eval_set = [e for e in (exact_assigned_episode(packs_for_eval[i % len(packs_for_eval)],
                                                   seed=i)
                            for i in range(int(hp["eval_episodes"]))) if e is not None]
    eval_free = ([exact_episode(p, max_tasks=int(hp["eval_max_tasks"]), seed=i)
                  for i, p in enumerate(packs_for_eval)] if hp["assigned_p"] < 1.0 else [])

    def card(extra: Optional[dict] = None) -> dict:
        return {"name": job or "local", "arch": arch, "dt": hp["dt"], "tol_m": hp["tol_m"],
                "tol_rad": math.radians(hp["tol_deg"]), "k_obstacles": int(hp["k_obstacles"]),
                "training": {"job": job, "cells": [p["cell_id"] for p in train_packs],
                             "eval_cells": [p["cell_id"] for p in eval_packs],
                             "hyperparameters": hp, **(extra or {})}}

    def act_np(graph: dict, noise: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            a = actor(to_tensors(G.stack([graph]), dev))[0].cpu().numpy()
        if noise > 0:
            a = np.clip(a + rng.normal(0, noise, a.shape), -1, 1)
        return a

    def run_eval(episodes: list) -> float:
        scores = []
        for ep in episodes:
            env = make_env(ep, hp)
            obs = env.reset(ep.q0, ep.tasks, ep.goal_q)
            while not env.finished:
                obs, _, _, _ = env.step(act_np(obs))
            scores.append(env.score())
        return float(np.mean(scores)) if scores else 0.0

    def evaluate(step: int) -> float:
        val = run_eval(eval_set)
        log_metric("eval_fraction_solved", val, step)
        if eval_free:
            log_metric("eval_free_fraction_solved", run_eval(eval_free), step)
        return val

    envs = []
    for _ in range(int(hp["n_envs"])):
        ep = gen.episode()
        env = make_env(ep, hp)
        envs.append({"ep": ep, "env": env, "obs": env.reset(ep.q0, ep.tasks, ep.goal_q),
                     "actions": []})

    updates, owed, best, t0 = 0, 0.0, -1.0, time.time()
    ep_scores: deque = deque(maxlen=50)
    total = int(hp["total_steps"])
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir(model_dir)
    nets = {"actor": actor, "actor_t": actor_t, "critic0": critics[0], "critic1": critics[1],
            "critic0_t": critics_t[0], "critic1_t": critics_t[1]}
    first, resumed = 1, False

    def save_checkpoint(step: int) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {"step": step, "updates": updates, "best": best, "owed": owed,
                 "nets": {k: n.state_dict() for k, n in nets.items()},
                 "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                 "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                 "gen_rng": gen.rng.bit_generator.state, "arch": arch,
                 "hyperparameters": hp, "job": job, "saved": time.time()}
        tmp = ckpt_dir / f".{CHECKPOINT}.tmp-{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, ckpt_dir / CHECKPOINT)        # never a half-written checkpoint
        log_metric("checkpoint_step", step, step)

    if resume:
        path = ckpt_dir / CHECKPOINT
        if not path.is_file():
            raise SystemExit(f"--resume: no {CHECKPOINT} in {ckpt_dir}")
        state = torch.load(path, map_location=dev, weights_only=False)
        if state.get("arch") != arch:
            raise SystemExit(f"--resume: checkpoint network {state.get('arch')} != this run's "
                             f"{arch} — resume with the job's own hyperparameters")
        for k, n in nets.items():
            n.load_state_dict(state["nets"][k])
        opt_a.load_state_dict(state["opt_a"])
        opt_c.load_state_dict(state["opt_c"])
        rng.bit_generator.state = state["np_rng"]
        gen.rng.bit_generator.state = state["gen_rng"]
        torch.set_rng_state(state["torch_rng"])
        updates, best, owed = int(state["updates"]), float(state["best"]), 0.0
        first, resumed = int(state["step"]) + 1, True
        print(f"[resume] from step {state['step']} ({updates} updates, best eval {best:.3f}) "
              f"in {ckpt_dir}", flush=True)
    for step in range(first, total + 1):
        sigma = hp["explore_sigma"] * hp["explore_decay"] ** (step / 100_000)
        for slot in envs:
            env = slot["env"]
            # Warm-up: random actions on a fresh run; a resumed run refills its
            # (unsaved) replay with the policy it already learned.
            a = act_np(slot["obs"], sigma) if (resumed or len(replay) >= hp["warmup"]) else \
                rng.uniform(-1, 1, (len(env.chains), G.MAX_DOF))
            obs2, r, done, _ = env.step(a)
            replay.add((slot["obs"], a, r, obs2, done))
            slot["actions"].append(a)
            slot["obs"] = obs2
            owed += hp["sample_to_insert"] / hp["batch_size"]
            if done:
                ep_scores.append(env.score())
                if rng.random() < hp["her_p"]:
                    for tr in her_relabel(env, slot["ep"], slot["actions"], rng):
                        replay.add(tr)
                ep = gen.episode()
                slot.update(ep=ep, env=make_env(ep, hp), actions=[])
                slot["obs"] = slot["env"].reset(ep.q0, ep.tasks, ep.goal_q)

        while owed >= 1.0 and len(replay) >= hp["warmup"]:
            owed -= 1.0
            updates += 1
            s, a, r, s2, d = replay.sample(int(hp["batch_size"]))
            s, s2 = to_tensors(s, dev), to_tensors(s2, dev)
            a, r, d = (torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (a, r, d))
            with torch.no_grad():
                a2 = actor_t(s2)
                a2 = (a2 + (torch.randn_like(a2) * hp["target_noise"]).clamp(
                    -hp["target_noise_clip"], hp["target_noise_clip"])).clamp(-1, 1)
                a2 = a2 * s2["robot_mask"][..., None]
                q_t = torch.min(critics_t[0](s2, a2), critics_t[1](s2, a2))
                y = r + hp["gamma"] * (1 - d) * q_t
            loss_c = sum(((c(s, a) - y) ** 2).mean() for c in critics)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            pre = actor(s, pre_tanh=True)
            # TD3 drives tanh actors into saturation (every joint at full speed);
            # a small penalty on the pre-squash output keeps them in range.
            loss_a = -critics[0](s, torch.tanh(pre) * s["robot_mask"][..., None]).mean() \
                + hp["action_reg"] * (pre ** 2).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            with torch.no_grad():
                for net, tgt, tau in [(actor, actor_t, hp["tau_policy"])] + \
                        [(c, ct, hp["tau_critic"]) for c, ct in zip(critics, critics_t)]:
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - tau).add_(p, alpha=tau)
            if updates % 1000 == 0:
                for g_, lr0, dec in ((opt_a, hp["lr_policy"], hp["lr_policy_decay"]),
                                     (opt_c, hp["lr_critic"], hp["lr_critic_decay"])):
                    for pg in g_.param_groups:
                        pg["lr"] = lr0 * dec ** (updates / 100_000)
                log_metric("critic_loss", float(loss_c), step)
                log_metric("actor_loss", float(loss_a), step)

        if step % 500 == 0 and ep_scores:
            log_metric("train_fraction_solved", float(np.mean(ep_scores)), step)
            # Saturation watch: the noise-free policy's mean |action| on the current
            # states — at 1.0 every joint is commanded at full speed (a broken actor).
            log_metric("action_abs_mean", float(np.mean([np.abs(act_np(sl["obs"])).mean()
                                                         for sl in envs])), step)
            log_metric("steps_per_s", step / max(time.time() - t0, 1e-6), step)
        if step % int(hp["eval_every"]) == 0 or step == total:
            score = evaluate(step)
            if score > best:
                best = score
                policy_io.save(model_dir / "best", export(actor),
                               card({"step": step, "eval_fraction_solved": score}))
        if step % int(hp["checkpoint_every"]) == 0:
            save_checkpoint(step)

    save_checkpoint(total)
    final = evaluate(total)
    policy_io.save(model_dir, export(actor), card({"step": total, "updates": updates,
                                                   "eval_fraction_solved": final,
                                                   "best_eval_fraction_solved": best}))
    return {"updates": updates, "eval_fraction_solved": final, "best": best}


# ─── entry point ─────────────────────────────────────────────────────────────

def _load_packs(directory: Optional[str]) -> list[dict]:
    if not directory or not Path(directory).is_dir():
        return []
    packs = []
    for f in sorted(Path(directory).glob("*.json")):
        pack = json.loads(f.read_text())
        if int(pack.get("schema_version", -1)) != G.SCHEMA_VERSION:
            raise SystemExit(f"{f.name}: scene schema v{pack.get('schema_version')} != "
                             f"v{G.SCHEMA_VERSION}")
        packs.append(pack)
    return packs


def parse_args(argv=None) -> tuple[dict, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default=os.environ.get("SM_CHANNEL_TRAIN"))
    ap.add_argument("--eval-scenes", default=os.environ.get("SM_CHANNEL_EVAL"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    ap.add_argument("--job-name", default=os.environ.get("TRAINING_JOB_NAME", ""))
    ap.add_argument("--checkpoint-dir", default=None,
                    help="where checkpoint.pt lives (default /opt/ml/checkpoints on "
                         "SageMaker, else <model-dir>/../checkpoint)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint instead of starting fresh")
    # Both spellings: SageMaker script mode passes hyperparameters as --<key> verbatim.
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", f"--{k}", dest=k, type=type(v), default=v)
    for k, v in DEFAULT_ARCH.items():
        ap.add_argument(f"--arch-{k}", f"--arch_{k}", dest=f"arch_{k}", type=int, default=v)
    ns, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"ignoring unknown arguments: {unknown}", file=sys.stderr)
    hp = {k: getattr(ns, k) for k in DEFAULTS} | {f"arch_{k}": getattr(ns, f"arch_{k}")
                                                  for k in DEFAULT_ARCH}
    return hp, ns


def main(argv=None) -> int:
    hp, ns = parse_args(argv)
    train_packs = _load_packs(ns.scenes)
    if not train_packs:
        raise SystemExit(f"no training scene packs in {ns.scenes!r}")
    eval_packs = _load_packs(ns.eval_scenes)
    out = train(hp, train_packs, eval_packs, Path(ns.model_dir), ns.job_name,
                checkpoint_dir=Path(ns.checkpoint_dir) if ns.checkpoint_dir else None,
                resume=ns.resume)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
