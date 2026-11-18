# multi_robots_agents

Agentic, collision-free motion planning and execution for **multi-robot industrial
cells** (spot welding, riveting, gluing, laser). You describe *where* the work
points are; the system decides *which* robot does *what*, plans parallel
collision-free trajectories, and executes them on a deterministic runtime with a
live safety monitor.

This README is the entry point. For a deeper component/architecture reference see
[`ARCHITECTURE.md`](ARCHITECTURE.md); to add your own robot cell and jobs see
[`docs/creating-a-cell.md`](docs/creating-a-cell.md).

---

## What it does

Given a **cell** (robots + workpiece/fixtures + obstacles) and a **task** (a list
of operation points), the system:

1. **Allocates & sequences** the operations across the available robots (MRTA:
   reachability pre-filter, TSP 2-opt, thermal spread, cycle-time balance).
2. **Plans** collision-free, parallel joint trajectories per operation
   (Pinocchio IK, OMPL, TOPP-RA time-parameterization), inserting
   explicit interlocks where two arms would otherwise collide.
3. **Assembles** a **[BehaviorTree.CPP v4](https://www.behaviortree.dev/docs/intro)**
   tree (`Sequence` / `Parallel` / `Delay` / `ExecuteTrajectory` / `WaitForRobot` /
   `CheckSafety` / `MoveHome`, waypoints on the blackboard) and **persists** it as a
   replayable plan — as JSON and as Groot2-loadable XML.
4. **Executes** it deterministically on the cell controller, dispatching timed
   trajectories to per-robot agents, enforcing interlock/position/timing gates,
   and running a collision monitor that e-stops on an imminent hit.

Every planning run is saved to `cells/<cell>/agentic_planner/plans/*.json` and every
execution to `cells/<cell>/agentic_planner/recordings/*.rrd`, so plans can be
**replayed without re-planning**. A cell is layered: its root is the scene bundle
as its author shipped it (robot URDF/SRDF, meshes, fixtures, `scene.yaml`), which
any application reading the cell resolves, while `agentic_planner/` holds
everything these agents derive — `config/`, `cell.yaml`, tasks, plans, logs,
recordings and the calculated obstacle model. The converter adds **nothing** to the
root; `cell.yaml` is the manifest that *references* the bundle's own assets in
place, and its paths stay relative to the cell directory so moving the manifest
never changes what an entry resolves to. One module,
[`utils/cell_layout.py`](utils/cell_layout.py), answers every path inside a cell.

---

## Architecture at a glance

```
             external caller / dashboard / IDE
                          │  A2A HTTP / MCP
                          ▼
   ┌──────────────┐   plan_trajectories   ┌──────────────────┐
   │ orchestrator │ ────────────────────▶ │ trajectory-agent │  MCP  :8890
   │  A2A  :8889  │ ◀──── BehaviourTree ── │  (planning)      │
   └──────┬───────┘                        └──────────────────┘
          │ execute (ROS 2 / MCP)
          ▼
   ┌────────────────┐  dispatch trajectory   ┌─────────────────────┐
   │ cell-controller│ ─────────────────────▶ │ robot-agent-<id>    │  (one per robot)
   │  MCP  :8892    │                        │  + robot-controller │  (sim FJT motion)
   │  (execution +  │ ◀──── robot_status ─── │                     │
   │   collision    │                        └─────────────────────┘
   │   monitor)     │
   └────────────────┘
```

| Service | Port(s) | Role |
|---|---|---|
| `trajectory-agent` | 8890 (MCP SSE + `/mcp` JSON-RPC, `/health`) | Plans trajectories, persists BehaviourTree plans |
| `orchestrator` | 8889 (A2A HTTP), 8893 (cell-data MCP) | Entry point: reads a task, calls the planner, hands the plan to the cell. Owns the cell's data |
| `cell-controller` | 8891 (health), 8892 (MCP) | Deterministic executor, replay, collision monitor, e-stop |
| `robot-agent-<id>` | — (ROS 2) | One per robot; FollowJointTrajectory executor + status |
| `robot-controller-<id>` | — (ROS 2) | Per-robot motion backend (serves the FJT action). `ROBOT_CONTROLLER=sim` (default) Pinocchio sim · `isaac` Isaac Sim FJT bridge · `none` defer to an external sim |
| `web-panel` | 8888 (dashboard), 8765 (MCP) | Web control panel + chat |
| `policy-trainer` | 8894 (MCP SSE + `/mcp`, `/health`) | Optional (`make trainer`): trains RoboBallet policies on SageMaker for the `roboballet` backend |
| `rviz-web` | 6080 (noVNC) | Optional RViz web viewer (`RVIZ=1`) |

The planner and cell controller are **MCP servers**, so any MCP client (the
dashboard chat, an MCP-capable IDE, Claude Desktop, or another
agent) can call their tools directly.

### Several planners

A planner container runs **one planning run at a time** — a plan, a resume or a
re-plan; a second one is refused (`planner_busy`) with the run that holds it.
Stop it with `cancel_planning(tree_id)` (discards) or `pause_planning(tree_id)`
(checkpoints, `resume_planning` continues); a run accepted but not yet started is
cancellable too.

Start more planners to plan more cells at once:

```bash
make sim SIM=<cell> PLANNERS=3            # trajectory-agent, trajectory-agent-2, -3
```

Each planner registers its MCP endpoint with the orchestrator and heartbeats
whether it is busy (`POST /a2a/planners/register`; `GET /a2a/planners` lists them).
The orchestrator gives each new run to a **free** planner (asked, not assumed),
and sends a run's status, pause, resume and cancel to the planner that **owns**
it — the one whose scratch holds its checkpoint. With every planner busy,
`POST /a2a/plan` answers `no_free_planner`. Planner 1 keeps `:8890` on this
machine; the others are reachable on the stack's network only.

### Who owns the cell's data

One agent owns a cell's data: the **orchestrator**. Plans, task programs, plan
traces and event streams, quality logs, the warm-start index, `robots.yaml` and
`obstacles.yaml` are read and written through its cell-data MCP on `:8893` —
`utils/cell_store.py` is the single API, and the planner runs with
`CELL_DATA_MODE=orchestrator` and `/cells` mounted **read-only**. Ask the owning
agent, not the volume: two writers on one tree is two answers to the same
question. Reads fail closed, so an unreachable owner refuses rather than
reporting a cell with no obstacles.

Two things stay local. The **scene bundle** (URDF/SRDF/meshes) is a read-only
mount — it is large and changes at commissioning, not during a run. And the
planner's own `/scratch` volume holds **unfinished work**: the pause/resume
checkpoint of a run in flight plus temporary files, so a resume needs nobody
else.

### Watching a run

The planner publishes each plan event on its own DDS topic
`/{cell_id}/plan_progress`, and the dashboard subscribes (`TRANSIENT_LOCAL`, so
opening the page mid-run shows the run so far). The same events are also kept as
`logs/planevents_<trace_id>.jsonl` by the owner, which is what a reader who
arrives after the run gets: the browser prefers the DDS feed and falls back to
`/api/a2a/plan_events` whenever it is not available.

The cell name is the prefix and is used **verbatim** — publisher and dashboard
compose the topic independently, so a cell id that cannot be a ROS name is
refused rather than rewritten (a rewritten prefix is a live view that is silently
always empty), and one bad cell never affects the others. The transport itself is
a start-up decision: domain, RMW and the cell's `cyclonedds.xml` (or the Zenoh
override) are handed to every container by `make compose`, and the planner reports
what it was given on `/health` rather than defaulting.

### Seeing how good a plan is

A plan that executes is not the same as a plan worth executing — the paths are
sampled then shortcut, so a valid, collision-free trajectory can still be four
times longer than it needs to be. Every plan is scored as it is saved, and the
**Performance** card in the dashboard's BTree tab **publishes the metric**: the
efficiency score, the verdict, the makespan and leg count, and how much detail is
behind it (the finding count, worst first). That is all the card holds — one
number you can read at a glance from across the room.

**Click it and the full report opens in its own window**: the four components the
score is made of (path, schedule, smoothness, dwell) with their weights, the
numbers behind them — excess travel, worst detour, parallel efficiency, load
balance, interior dwell, per-arm utilisation — every finding naming its leg and by
how much, the worst legs in detail, and the achieved clearance. `Esc`, the close
button or the backdrop dismisses it. Opening it costs no round trip: the one fetch
that filled the card filled the window too, so the report you open is the plan the
card is measuring and not a newer one.

The panel computes none of it: it asks the trajectory agent over MCP
(`get_plan_quality`), which owns the metric definitions. Clearances appear only in
the window and deliberately in no score — a score is a thing people optimise, and
a scored safety margin is a tunable one. A failed run is refused rather than
scored, and a planner that cannot be reached reads as *not measured, and why*,
never as a plan that scores zero. Selecting another plan drops the published metric
and closes an open report immediately, before the new answer arrives, so no number
is ever shown under the wrong plan's name.

**What to change, not just how good.** `suggest_plan_improvements(cell_id,
tree_id?, program?)` (trajectory MCP, read-only) merges the quality findings with
the planning run's trace — recurring planner failures, operations that took many
attempts, where the wall-clock went (motion planner vs. LLM reasoning) — and, given
a `program`, its cycle-time target, into one ranked list of problems, each with its
evidence and a suggested fix, plus a one-line summary.

**How far a run is.** `get_plan_status` carries `progress` — `percent` (0-100),
`phase`, operations solved of total, attempts and failed attempts — read off the
run's event stream; the orchestrator copies it into `GET /a2a/result/<run_id>`
while a plan runs. It measures work done on the program, not time remaining.

---

## Prerequisites

- **Docker** + **Docker Compose v2** (`docker compose ...`).
- **`make`** and **`curl`**.
- **No host Python.** The generator scripts, the Makefile helpers and the test suite
  all run inside a small runner image built from `docker/Dockerfile.tools`, with the
  repo bind-mounted at its own path — so generated files land in the working tree as
  if a local interpreter had written them. The image builds itself on first use and
  is cached until its Dockerfile or `docker/requirements-tools.txt` changes
  (`make tools` to build it up front, `make tools-clean` to drop it).
- **LLM credentials** for planning. The default provider is **AWS Bedrock**
  (`eu.anthropic.claude-sonnet-5` in `eu-west-1`). Export valid credentials in the
  shell **before** starting the stack — they are interpolated into the generated
  compose file:

  ```bash
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_SESSION_TOKEN=...
  ```

  > Credentials are baked into the container environment at create time. If they
  > expire, refresh them and recreate the affected service
  > (`docker compose -f .generated_compose.yml up -d --force-recreate --no-deps trajectory-agent`).
  >
  > Replaying an already-saved plan is **deterministic and needs no LLM** — only
  > planning (`plan_trajectories`) and LLM-based recovery require credentials.

---

## Install & start

Everything runs through the `Makefile`, which generates a cell-specific compose
file (`.generated_compose.yml`) from the cell's `robots.yaml` and brings the
stack up.

```bash
# 1. Generate the cell-specific compose file (.generated_compose.yml)
make compose SIM=welding_cell_2_small

# 2. Build the images (first time, or after changing docker/ or requirements)
make build SIM=welding_cell_2_small

# 3. Start the full stack for a cell (attached; Ctrl-C to stop)
make sim SIM=welding_cell_2_small

#    …or detached (background):
make sim-detached SIM=welding_cell_2_small

#    …with the RViz web viewer (http://localhost:6080/vnc.html):
make sim SIM=welding_cell_2_small RVIZ=1

#    …trajectory planning only — planner + orchestrator, no ROS/DDS, no dashboard,
#    no robots. Plans are calculated and read over A2A/MCP, not executed:
make sim SIM=welding_cell_2_small ONLY=trajectory

#    …execution only — cell controller (executor + collision supervisor), every
#    robot agent / robot controller, and the dashboard on :8888 (add RVIZ=1 for
#    the viewer). The dashboard and viewer reach the planner at
#    TRAJECTORY_AGENT_HOST, which defaults to ORCHESTRATOR_HOST:
make sim SIM=welding_cell_2_small ONLY=execution

#    …planning and execution on two machines. Each side is told where the other is
#    (the orchestrator reaches the cell controller on :8892; the cell controller
#    reads plans and cell config from the orchestrator on :8893). Both machines need
#    the cell directory for the robot/fixture meshes:
#      planning machine (10.0.0.4):
make sim SIM=welding_cell_2_small ONLY=trajectory CELL_CONTROLLER_HOST=10.0.0.5
#      execution machine (10.0.0.5):
make sim SIM=welding_cell_2_small ONLY=execution  ORCHESTRATOR_HOST=10.0.0.4

# 4. Check everything is healthy
make health

# 5. Open the dashboard
#    http://localhost:8888
```

> `make sim` regenerates the compose file automatically, but `make build` needs
> it to already exist — so run `make compose` first on a fresh checkout or when
> switching `SIM` to a different cell.

Useful variables (see `make help` for the full list):

| Variable | Default | Meaning |
|---|---|---|
| `SIM` | `welding_cell_2_small` | Cell id (directory under `cells/`) |
| `RVIZ` | `0` | `1` to start the RViz web viewer on :6080 |
| `TRAJECTORY_BACKEND` | `ompl` | Planner's default backend for requests that name none: `ompl` \| `ompl-compound` \| `rrt` (legacy) |
| `PLANNING_MODE` | `classic` | Planner's default mode for requests that name none: `classic` \| `coordinated` \| `stepwise` |
| `BACKEND` / `MODE` | — | Backend / mode of one `make plan` / `task-mcp` / `task` request |
| `LOCKSTEP` | `0` | `1`: the cell executor runs `ompl-compound` steps in lockstep (barrier start, e-stop on drift) |
| `LLM_PROVIDER` | `bedrock` | `bedrock` \| `openai` \| `ollama` |
| `LLM_MODEL` | `opus-5` | Startup model of the orchestrator and the recovery layer |
| `PLANNING_LLM_MODEL` | `sonnet-5` | Model for trajectory planning; a request's own `model` (A2A / MCP, or `make task MODEL=opus-5`) still wins |
| `RMW` | `cyclonedds` | ROS 2 middleware: `cyclonedds` \| `zenoh` |
| `ZENOH_ROUTER` | — | Endpoint of the existing Zenoh broker; required when `RMW=zenoh` |

Stop everything with `make down`. On a split stack, `make down ONLY=execution` stops
the cell controller, the robots, the dashboard and RViz, and `make down ONLY=trajectory` the planner
and the orchestrator; the other side keeps running.

### Running on an external Zenoh broker

The transport is a **startup decision**, not a code change. CycloneDDS is
peer-to-peer and needs nothing else; Zenoh is brokered, so it needs an address:

```bash
# Every container joins the plant's existing Zenoh router
make sim SIM=welding_cell_2_small RMW=zenoh ZENOH_ROUTER=tcp/broker.plant.local:7447

# Several brokers (comma-separated, port defaults to 7447)
make sim SIM=welding_cell_2_small RMW=zenoh ZENOH_ROUTER=10.0.0.5,10.0.0.6:7448
```

This stack **runs no router of its own** — the broker is yours, already on your
network, and the agents are clients of it. `RMW=zenoh` without `ZENOH_ROUTER` is
therefore refused at generation time rather than defaulting: `rmw_zenoh`'s own
default is `tcp/localhost:7447`, which inside a container resolves to that
container, so the stack would come up looking healthy with every agent isolated
and no topic crossing between arms.

Changing `RMW` regenerates `.generated_compose.yml` (the settings the file was
generated from are recorded in `.compose_settings`), so the running stack always
matches the invocation that started it. `agents/middleware.py` owns the decision;
both RMWs are baked into the images, so switching needs no rebuild.

For anything the endpoint override cannot express — TLS material, scouting,
access control — drop a full session config at
`cells/<cell>/agentic_planner/config/zenoh_session.json5` and it is referenced
automatically (`ZENOH_SESSION_CONFIG_URI`) when present. A cell that predates the
config directory's move into the workspace is still read at
`cells/<cell>/config/`.

### Planning speed

A planning run is a loop of model turns around fast tools: on `bt_1790857060`
(15 welds) 92 % of the 1,287 s was the model, 45 s was OMPL. So planning is made
fast by making the turns few and small:

- **Sonnet plans by default** (`PLANNING_LLM_MODEL`); pass `model` with a request
  to plan one hard cell on Opus.
- **Bedrock prompt caching** re-uses the conversation prefix each turn
  (`LLM_PROMPT_CACHE=0` turns it off). Per-turn token use is logged as `[plan] llm:`.
- **The model sees trajectory summaries, not waypoint arrays** — legs are cached by
  operation id, and the validators take `{robot_ns: operation_id}`.
- **Several steps per planner call** (`run_trajectory_planner(steps=[...])`),
  stopping at the first step that fails.
- **Each step is gated as it is planned** (`step_gate`): the assembler's inter-arm
  check, against the arms parked where they last stopped (`PLAN_STEP_GATE=0` off).
- **Legs chain exactly** from where the previous one ended, so
  `chain_trajectories` no longer forces a re-plan of the program.

---

## Running work

```bash
# List the task programs defined for a cell
make tasks SIM=welding_cell_2_small

# Plan + execute a task through the orchestrator (A2A)
make task SIM=welding_cell_2_small TASK=car_door_spot_welding

# …or straight through the cell MCP (bypasses the orchestrator)
make task-mcp SIM=welding_cell_2_small TASK=car_door_spot_welding

# List saved recordings, then replay one without re-planning
make recordings SIM=welding_cell_2_small
make replay SIM=welding_cell_2_small TREE_ID=bt_1784731893_welding_cell_2_small
```

Logs and introspection:

```bash
make logs                 # all services   (or SERVICE=trajectory-agent)
make logs-traj            # trajectory-agent only
make list-tools           # MCP tools on the trajectory agent
make list-cell-tools      # MCP tools on the cell controller
```

---

## Repository layout

```
multi_robots_agents/
├── agents/
│   ├── orchestrator/     # A2A entry point: task → plan → execute
│   ├── trajectory/       # planning agent + MCP server (mcp_server.py)
│   │   └── tools/        # pinocchio, ompl/rrt, knowledge_base, plan_store …
│   ├── cell/             # deterministic executor + MCP server + collision monitor
│   └── robot/            # per-robot ROS 2 agent (trajectory executor)
├── controller/           # pinocchio_robot_controller.py — simulated FJT motion
├── dashboard/            # web_control_panel.py — Flask UI + MCP client
├── cells/                # cell definitions (see docs/creating-a-cell.md)
│   └── <cell>/           # root = the scene bundle, untouched (scene.yaml,
│       │                 #   robots/, fixtures/, floorplans/) — the converter
│       │                 #   writes NOTHING here
│       └── agentic_planner/  #   this app's workspace, all of it derived:
│                         #   cell.yaml (the manifest: REFERENCES the bundle's
│                         #   assets, copies nothing, paths relative to the CELL
│                         #   dir), config/ (robots.yaml, cyclonedds.xml, the
│                         #   calculated obstacles.yaml), tasks, plans, logs,
│                         #   recordings
├── utils/cell_layout.py  # the single owner of every path inside a cell
├── skills/               # SKILL.md capability packs for the agents
├── scripts/              # convert_isaac_scene.py (Isaac export → cell),
│                         #   generate_compose.py, generate_cell_urdfs.py, generate_srdf.py
├── docker/               # Dockerfile, Dockerfile.viz, requirements.txt, launchers
│                         #   Dockerfile.tools = the runner every make target uses
├── ros2_packages/        # agent_interfaces msgs/srv/action
├── tests/                # pytest suite
├── docker-compose.yml    # base compose (per-cell compose is generated)
├── Makefile              # entry point for build / run / tasks / replay
├── ARCHITECTURE.md       # deeper component reference
└── docs/
    └── creating-a-cell.md  # add a robot cell from a URDF + Isaac scene, author tasks
```

---

## Trajectory backends

Chosen **per planning request**, over MCP: `plan_trajectories(…, backend=, mode=)`,
`replan_operation(…, backend=, mode=)`, `"backend"` / `"mode"` on `/a2a/plan` and
`/a2a/run`, or `make plan|task-mcp|task … BACKEND= MODE=`. A request that names
none gets the stack default (`make sim TRAJECTORY_BACKEND=… PLANNING_MODE=…`), else
`ompl` / `classic`. The reply says which applied (`selected_by`). An unknown name
refuses the request, and there is **no silent fallback** between backends.

Each backend is its own package under `agents/trajectory/backends/` behind one
facade (`agents/trajectory/tools/planner_backend.py`); coordination, the agent and
the MCP interface are shared by all of them. A backend from **another repository**
(e.g. a robot-ballet planner) plugs in by entry point or
`TRAJECTORY_BACKEND_PLUGINS`, without editing this one — see
[`docs/creating-a-backend.md`](docs/creating-a-backend.md). `list_trajectory_backends`
(MCP) lists what can be selected.

- **`ompl`** (default) — OMPL RRTConnect, time-parameterized with TOPP-RA. No extra services.
- **`rrt`** — the legacy built-in Space-Time RRT planner. **Non-conforming**: no
  inter-robot coupling, no tool-orientation constraint, no two-phase speed/clearance
  split. Kept selectable for comparison only; results are stamped
  `non_conforming_backend: true`.
- **`ompl-compound`** — every arm moving in a step planned as ONE path in an OMPL
  `CompoundStateSpace`, arm-vs-arm checked in every state and timed on one clock;
  lift-off and descent strokes per arm as in `ompl`. A step that cannot hold is
  refused as a whole. Pair it with `make sim … LOCKSTEP=1` to have the executor
  start each step's legs together and halt the cell if they drift apart. See
  [ADR-0001](docs/adr/0001-ompl-compound-backend.md).
  `make plan SIM=welding_cell_2_small TASK=car_door_spot_welding TRAJECTORY_BACKEND=ompl-compound`
- **`roboballet`** (bundled) — a trained [RoboBallet](https://arxiv.org/abs/2509.05397)
  graph-network policy (RL: TD3 + hindsight replay) drives every arm of a step at
  once, holding an arm whenever its next move would collide; the policy's path
  replaces the compound OMPL search, and `ompl-compound`'s lift-off, descent,
  timing and checks wrap it, so its legs reach the BehaviourTree exactly like
  every other backend's. Same interface as any backend — nothing in the
  orchestrator knows about it. It needs a trained policy (no policy = a refused
  plan, never a fallback). See [ADR-0002](docs/adr/0002-roboballet-policy-backend.md)
  and [Training a policy](#training-a-roboballet-policy).
  `make plan SIM=welding_cell_2_small TASK=car_door_spot_welding BACKEND=roboballet`
  plans with the promoted policy; `POLICY=<name>` (or, over MCP/A2A,
  `backend_options={"policy": "<name>"}`) plans with another installed one —
  the generic per-request `backend_options` every backend can declare.

### Training a RoboBallet policy

The **policy trainer** is its own container with its own image
(`docker/Dockerfile.trainer`: slim Python + PyTorch, no ROS; compose profile
`training`), driven over MCP like the other agents and started from make. It is
the one place that prepares training (scene packs from the cells, the code
bundle), starts it, **monitors** it (a background thread refreshes every job's
status and metrics, and can import the policy when a job completes) and imports
the result. **One setting** decides where jobs run, `POLICY_TRAINING_TARGET`:

- `local` (default) — the **same** `train.py` as a process of the trainer
  container (CPU torch by default; `TRAINER_GPU=1 TRAINER_TORCH_INDEX=cu124` on a
  GPU host). No AWS settings needed; up to `LOCAL_TRAINING_MAX_JOBS` (2) in
  parallel, sharing the cores.

Every job keeps its own files, named after it (`<job>.status.json`, `<job>.log`,
`checkpoint/`, `model/` under `trainer-state:/state/jobs/<job>/`), and checkpoints
its whole learner state every `checkpoint_every` steps. A job that is interrupted
— the trainer restarted, the process was killed, it was stopped or failed —
resumes from that checkpoint with `make train-resume JOB=<job>` (MCP
`resume_training_job`): a local job continues in place; a SageMaker job continues
as `<job>-r<n>` from the same S3 checkpoint. `POLICY_TRAINING_AUTO_RESUME=1`
resumes interrupted local jobs on its own.
- `sagemaker` — a SageMaker PyTorch job (inputs uploaded to S3). Costs money,
  hence never the default.

```bash
make trainer                                   # local training (default)
make trainer POLICY_TRAINING_TARGET=sagemaker  # SageMaker training
make sim TRAINER=1                             # the trainer with the stack
```

| MCP tool (`:8894/mcp`) | What it does |
|---|---|
| `export_scene(cell_id)` | the cell, in the existing cell format, as the policy sees it |
| `start_training(train_cells, eval_cells, hyperparameters, auto_import, dry_run)` | starts a job on the configured target — local, or SageMaker (uploads scene packs + code) |
| `get_training_job` / `list_training_jobs` / `get_training_logs` / `stop_training_job` | the same for both targets: status, metrics (`eval_fraction_solved`, …), log tail (local log or CloudWatch), stop |
| `resume_training_job(job)` | continue an interrupted / stopped / failed job from its last checkpoint |
| `import_policy(job, name, promote)` | the job's model → the `policy-models` volume, validated against this build |
| `evaluate_policy(name, cell_id, mode)` | offline score on **any** cell (`unseen_cell` says if it was trained on) |
| `promote_policy(name)` / `list_policies` | which policy `roboballet` plans with (`current`, or `ROBOBALLET_POLICY=`) |

Training and evaluation use **different layouts if you want**: `train_cells` are
randomised by the procedural generator (robot placement, obstacle subsets plus
random cuboids, task sets), `eval_cells` are scored as declared, and planning is
any cell at all. Every cell is read from what it already has — `robots.yaml`, the
URDF, `obstacles.yaml`, task programs; nothing new is asked of a cell.

A SageMaker job is fail-closed: with `POLICY_TRAINING_TARGET=sagemaker` it still
needs `SAGEMAKER_ROLE_ARN`, `POLICY_S3_URI` and (recommended) `AWS_ACCOUNT_ID`,
which both the role and the caller must match — see `.env.example` and
`agents/policy_trainer/config.py`. Try `dry_run=True` first: it returns the exact
`CreateTrainingJob` request (or the local command line) and starts nothing.

---

## Documentation

- [`docs/creating-a-cell.md`](docs/creating-a-cell.md) — **create a new cell from a
  URDF + Isaac scene, and author a task to execute.**
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — components, agents, skills, message types.
- [`docs/creating-a-backend.md`](docs/creating-a-backend.md) — **add a trajectory
  backend** (in this repo or another) behind the shared coordination and MCP interface.
- [`docs/api/openapi.yaml`](docs/api/openapi.yaml) — the HTTP surface of the
  dashboard (`:8888`) and the orchestrator A2A (`:8889`), with the run-envelope
  and 409-busy shapes. `tests/test_openapi_spec.py` holds the spec and the
  routes served to each other.
- `skills/*/**/SKILL.md` — capability packs the agents load on demand.
- `skills/orchestrator/run-welding-program/references/task-json-format.md` — task JSON schema.
