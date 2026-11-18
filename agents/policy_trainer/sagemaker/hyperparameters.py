"""Training hyperparameters and their defaults (paper Table S5, where this
implementation has the knob). Framework-free, so the trainer service can check
a request's keys before it starts anything — locally or on SageMaker — and a
typo is refused instead of silently trained with the default.
"""
from __future__ import annotations

try:                                   # inside the SageMaker bundle
    from roboballet_core.gnn import DEFAULT_ARCH
    from roboballet_core.graph import K_OBSTACLES
except ImportError:                    # in the repository
    from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH
    from agents.trajectory.backends.roboballet.core.graph import K_OBSTACLES

#: Defaults. Where they differ from the paper's Table S5 it is because the
#: paper's values did not learn in this implementation, measured on a two-arm
#: cell over 12k steps (eval = noise-free, assigned, from HOME on the declared cell):
#:   collision_penalty 15, sparse reward        -> solved 0 (the learner stands still)
#:   + shaping, penalty 0.5, tau 4e-5, lr 5e-5   -> solved 0 (actions saturated at +-1)
#:   + mean aggregation, action_reg 0.1,
#:     penalty 0.05, tau 5e-3, lr 3e-4           -> train 0.51, eval 0.375
#: The paper trained for 1e6+ steps on far larger batches; its values remain one
#: hyperparameter away (e.g. {"tau_policy": 4e-5, "collision_penalty": 15}).
DEFAULTS = {
    "total_steps": 1_000_000, "batch_size": 128, "lr_policy": 3e-4, "lr_critic": 3e-4,
    "lr_policy_decay": 0.98, "lr_critic_decay": 0.95,          # per 100 000 updates
    "target_noise": 0.1, "target_noise_clip": 0.2,
    "tau_policy": 5e-3, "tau_critic": 5e-3, "gamma": 0.94,
    "explore_sigma": 0.3, "explore_decay": 0.9,                # per 100 000 env steps
    "replay_size": 1_000_000, "sample_to_insert": 16.0, "warmup": 2_000,
    # shaping: potential-based progress reward towards the assigned goal (0 = the
    # paper's sparse reward); collision_penalty: per held arm and step.
    "dt": 0.1, "horizon_s": 20.0, "collision_penalty": 0.05, "shaping": 1.0, "dwell_s": 0.5,
    "tol_m": 0.025, "tol_deg": 15.0, "n_envs": 8, "her_p": 1.0,
    "eval_every": 20_000, "eval_episodes": 8, "eval_max_tasks": 6, "seed": 0,
    "k_obstacles": K_OBSTACLES, "max_tasks": 10, "assigned_p": 0.7,
    # Penalty on the actor's pre-tanh output: keeps actions out of saturation.
    "action_reg": 0.1,
    # Full learner state to checkpoint.pt every N steps (resume_training_job).
    "checkpoint_every": 5_000,
}

#: Every key a job may set: the above plus ``arch_<k>`` for the network sizes.
ALLOWED = set(DEFAULTS) | {f"arch_{k}" for k in DEFAULT_ARCH}


def check(hyperparameters: dict) -> dict:
    """The request's overrides, typed like their defaults; unknown keys refused."""
    unknown = sorted(set(hyperparameters) - ALLOWED)
    if unknown:
        raise ValueError(f"unknown hyperparameters {unknown}; allowed: {sorted(ALLOWED)}")
    out = {}
    for k, v in hyperparameters.items():
        default = DEFAULTS.get(k, DEFAULT_ARCH.get(k.removeprefix("arch_"), 0))
        try:
            out[k] = type(default)(v)
        except (TypeError, ValueError):
            raise ValueError(f"hyperparameter {k}={v!r} is not a {type(default).__name__}") from None
    return out
