# ADR-0002 — `roboballet`: a trained multi-arm policy as a backend, trained on SageMaker

- **Status:** proposed
- **Date:** 2026-10-24

## Context

RoboBallet (Lai et al., *Science Robotics* 2025, doi:10.1126/scirobotics.ads1204,
arXiv:2509.05397) trains a graph-neural-network policy with reinforcement
learning to plan multi-arm reaching:

- **Graph.** Robots, tasks and obstacle primitives are nodes. Edges carry
  relative poses: into robots from tasks and obstacles, and both ways between
  robots.
- **Actions.** Every 0.1 s the policy commands every joint of every arm.
- **Collisions.** A kinematic simulator holds any arm whose move would collide,
  and penalises it.
- **Training.** TD3 with hindsight experience replay, on procedurally
  generated cells.
- **Generalisation.** It transfers zero-shot to unseen robot placements,
  obstacles and task sets.

Two things were asked for: planning with such a policy, and a service, driven
over MCP, that trains it on SageMaker. Training and evaluation should be able
to use different cell layouts, both in the existing cell format.

## Decision

1. **A backend like any other.** `roboballet` implements the
   `BackendSpec`/`plan()` contract and is selected by name everywhere
   (`backend="roboballet"`). The orchestrator, the agent, the MCP interface and
   the BehaviourTree assembler do not know it exists. It is **bundled**: in this
   repository but registered like a plugin (`backends.BUNDLED`), never the
   default. The three built-ins are unchanged.

2. **The policy replaces the search, not the step.**
   `plan_trajectories_ompl_compound` gains a `transit_solver` argument. The
   policy rollout is that solver. Everything else is still `ompl-compound`'s and
   holds the policy to the same contract:
   - the on-axis lift-off out of the seam;
   - the straight descent into the weld;
   - one time base (TOPP-RA);
   - the emitted-leg checks and the inter-arm gate;
   - the commit.

   So `conforming=True` and `joint_step=True`.

3. **Assigned mode.** The platform's allocator has already decided which arm
   does which weld. Each arm therefore gets one task, its standoff pose, with an
   edge only to that arm. The robot node also carries the joint offset to that
   task's IK goal. The paper's free-allocation graph is the same graph with all
   task edges and no goal offset, and training mixes both modes. Allocation by
   the policy stays out of scope, because allocation is the planner's MRTA step.

4. **The platform's own collision predicate.** During a rollout an arm is held
   when the pipeline's `is_free` (static obstacles at the transit margin, plus
   the parked bystanders) or the arm-to-arm clearance says no. The policy's
   coarser envelope only shapes its features. Two cases are `no_path_found`:
   - an arm does not arrive within the horizon or the time budget;
   - an arm arrives in an arm configuration more than `ROBOBALLET_MAX_DOCK_RAD`
     from the one the descent was planned from.

   Otherwise the remaining offset is docked with a checked straight joint move,
   as the paper's post-processing does.

5. **Fail-closed policy loading.** With no policy, an unreadable one, or a
   graph-schema mismatch, every request of the step is refused with
   `planner_error`. Nothing falls back to another planner.

6. **One core, two runtimes.** Kinematics, primitives, the graph, the GNN
   forward pass and the simulator live in
   `agents/trajectory/backends/roboballet/core`, which uses numpy only. The
   planner runs them as they are. The SageMaker job ships them as
   `roboballet_core` with a PyTorch twin of the network, and a parity test holds
   the two forward passes to the same numbers. The planner image needs no
   deep-learning framework.

7. **The trainer is its own container, with its own image.**
   - **Startup.** `policy-trainer` is an MCP server on :8894, behind compose
     profile `training`. It starts with `make trainer`, or `make sim TRAINER=1`
     to start it with the stack.
   - **Image.** `docker/Dockerfile.trainer`: slim Python with PyTorch, no ROS,
     running as a non-root user.
   - **Role.** It is the only component that prepares, starts, monitors, stops
     and imports training.
   - **Targets.** One setting, `POLICY_TRAINING_TARGET`, decides where every job
     runs: `local` (the default, free) or `sagemaker`. Local jobs run the same
     `train.py` as a process of the trainer container.
   - **Tracking.** One tracker records both targets on the `trainer-state`
     volume. Its monitor thread refreshes status and metrics, auto-imports on
     completion when asked, and marks local jobs orphaned by a restart as Failed.
   - **Data.** It reads cells as the planner does and writes its own state plus
     the `policy-models` volume, which the planner mounts read-only.
   - **Jobs.** They run network-isolated, receive their code as an input
     channel, and upload with server-side encryption.
   - **Account pin.** A SageMaker job is refused unless the target is
     `sagemaker` and the role and the caller are both in the pinned
     `AWS_ACCOUNT_ID`.
   - **Dry run.** `dry_run=True` returns the exact `CreateTrainingJob` request.

8. **Training and evaluation layouts are independent.** `train_cells` are
   randomised by the procedural generator, seeded by the real cells:
   - base jitter;
   - obstacle subsets plus random cuboids;
   - the cell's welds plus face-sampled tasks, IK-filtered.

   `eval_cells` score checkpoints as declared, and the best checkpoint is kept.
   `evaluate_policy` scores any cell offline. Planning, the binding evaluation,
   works on any cell.

## Consequences

- A policy is as good as its training. The rollout may fail to arrive, and that
  failure is reported per operation like any other planner failure.
  `suggest_plan_improvements` and the recovery skill already act on
  `no_path_found`.
- `roboballet` imports `ompl_compound.planner`'s public entry point. The step
  pipeline (preparation, timing, validation) would be cleaner in `tools/`. That
  move is deferred: it touches the production compound backend and has no
  behaviour change to show for itself.
- Deviations from the paper, on purpose:
  - assigned-mode goal offsets (decision 3);
  - K-nearest obstacle edges per arm, because a voxelised fixture is hundreds of
    boxes;
  - HER replays the recorded path rather than re-simulating dwell;
  - the default network is smaller than Table S5. The paper's sizes are
    hyperparameters (`arch_node=512`, …).
- Not verified here: a real SageMaker job, and the backend inside the planner
  container against Pinocchio/OMPL. Both need the container and AWS
  credentials.
