# multi_robots_agents

## Design Philosophy

This repository is a clean rewrite — no code copied from legacy repos.

Every agent is built using **LangChain Deep Agents** (`deepagents`) with the
[Agent Skills specification](https://agentskills.io/specification). Skills are
`SKILL.md` files that package domain expertise as instructions the agent
discovers progressively — metadata at startup, full instructions on demand.
No Python skill class hierarchies. No monolithic skill modules.

The trajectory agent is a **specialized LangGraph agent** whose capabilities
are exposed externally via **MCP over HTTP/SSE** — any MCP-compatible client
(dashboard chat, an MCP-capable IDE, Claude Desktop, another agent) can call it directly.

---

## Container Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  orchestrator                     OWNS /cells (read-write)            │
│  agents/orchestrator/                                                 │
│  A2A HTTP :8889   ·   cell-data MCP :8893                             │
│  Deep Agent + skills/orchestrator/                                   │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ A2A HTTP / MCP  ·  cell data in BOTH directions
┌──────────────────────▼──────────────────────────────────────────────┐
│  trajectory-agent                 /cells read-only · /scratch its own │
│  agents/trajectory/                                                   │
│  MCP SSE :8890   ←── any MCP client can call this directly           │
│  LangGraph specialized agent + skills/trajectory/                    │
│                                                                       │
│  Skills (progressive disclosure):                                    │
│    optimize-sequence/    allocate + TSP + thermal + pairing          │
│    plan-trajectories/    IK + RRT + spline + collision avoidance     │
│    find-interlocks/      safe intermediate positions                 │
│    validate-collision/   trajectory pair validation                  │
│    check-reachability/   Pinocchio IK feasibility                    │
│    build-behaviour-tree/ assemble BT from planned batches            │
│    trajectory-knowledge/ warm-start from recordings KB               │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ ROS 2  /{cell_id}/behaviour_tree
┌──────────────────────▼──────────────────────────────────────────────┐
│  cell-controller                                                      │
│  agents/cell/                                                         │
│  Deep Agent + skills/cell/                                           │
│                                                                       │
│  Skills:                                                             │
│    dispatch/            time/position/safety gates                   │
│    collision-monitor/   250 Hz sphere-model collision check          │
│    cell-state/          publishes robot_description + scene          │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ ROS 2  /{cell_id}/{ns}/{id}/dispatch_subtask
┌──────────┐  ┌──────────┐  ┌──────────┐
│ robot-   │  │ robot-   │  │ robot-   │   one container per robot
│ agent-r1 │  │ agent-r2 │  │ agent-rN │   ROBOT_NAMESPACE env var
│          │  │          │  │          │
│  FJT +   │  │  FJT +   │  │  FJT +   │
│ RobotStatus  RobotStatus  RobotStatus
└──────────┘  └──────────┘  └──────────┘
```

---

## Cell data has one owner

A cell's data — plans, task programs, plan traces and event streams, quality
logs, the warm-start index, `robots.yaml`, `obstacles.yaml` — is **owned by the
orchestrator**. Every other agent asks it, over the cell-data MCP on `:8893`
(`agents/orchestrator/cell_data_mcp.py`), and nobody else writes to the volume.

*Ask the owning agent, not the volume.* Two processes mounting the same tree is
two answers to the same question the moment one of them is a container restart
behind, and a planner that writes a second copy of a plan makes the dashboard's
answer depend on which mount it happened to read.

`utils/cell_store.py` is the one API, with two modes behind it:

| `CELL_DATA_MODE` | Who resolves the bytes | Used by |
|---|---|---|
| `local` (default) | this process's own disk | orchestrator, host scripts, tests |
| `orchestrator` | one MCP call per operation | trajectory-agent |

Artefacts are addressed as `(kind, name)` — `config`, `manifest`, `tasks`,
`plans`, `logs`, `recordings` — never as a path; names must match
`^[A-Za-z0-9._-]+$`. `cell_summarize` picks named (optionally dotted) fields
server-side, so listing 200 plans does not ship 200 waypoint arrays.

Two deliberate exceptions:

* **The scene bundle** (URDF/SRDF/meshes/`scene.yaml`) stays a read-only mount.
  It is up to ~1 GB, it changes when a cell is re-commissioned rather than
  during a run, and mesh URIs are resolved by libraries outside this repo.
* **Unfinished work stays local.** The planner's own disk holds `/scratch` and
  nothing else: the pause/resume checkpoint of a run in flight plus temporary
  files, written on every step and meaningful only to the process that resumes
  them. A resume therefore needs no orchestrator.

Reads **fail closed**: an unreachable owner raises `CellStoreUnavailable` and
never degrades to "absent". "No obstacles" from a broken socket would route an
arm through a fixture; refusing to plan is the acceptable failure.

## Live progress goes out on DDS

Following a run is a question about *now*, so it does not travel through a file.
The planner publishes every plan event as `std_msgs/String` JSON on its own topic
`/{cell_id}/plan_progress` (`agents/trajectory/plan_progress.py`), and the
dashboard — already a ROS node in the same graph — subscribes with
`TRANSIENT_LOCAL` and a deep history, so a panel opened mid-run renders the run
instead of waiting for its next event.

The same event is also appended to `logs/planevents_<trace_id>.jsonl` through the
owner. The split is deliberate: DDS history is bounded and answers "what is
happening", the artefact is unbounded, survives the container, and answers "what
happened". Publishing is best effort in both directions — a dead middleware never
costs the durable record, and a failing write never darkens the live view.

The browser prefers DDS and falls back to HTTP in the same tick:
`GET /api/plan_progress` must answer `available: true` for its events to be used;
anything else (a panel built without `rclpy`, an older dashboard that does not
serve the route) falls through to `/api/a2a/plan_events`, because "I am not
listening" must never render as "no run in progress".

### The cell name is the prefix, verbatim

The topic is per cell, and the prefix is what keeps two cells in one plant from
reading as one run. Both ends compose it independently — the publisher from
`plan_progress.topic_for()`, the dashboard from `cell_topic()` (`/{CELL_ID}` +
suffix) — so the name is **validated, never normalised**: `cell_of()` accepts a
legal ROS name token (`^[A-Za-z_][A-Za-z0-9_]*$`) and refuses anything else
rather than rewriting it, because a sanitised prefix publishes on a name the
dashboard never subscribes to, which is a live view that is silently always
empty. A missing cell is refused too, rather than published on a shared
`/plan_progress`. The fallback for a call site that does not carry the cell is
`CELL_ID`, the identity compose hands the container.

The refusal is per cell (`_refused` in `describe()`), and said once per cell:
one badly named cell must not darken the live view for every other cell in the
process, and must not fill the planner's log. Its durable JSONL is unaffected —
that stream is not scoped by a ROS name.

### The transport arrives at start-up

`ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION` and the cell's `cyclonedds.xml`
(`CYCLONEDDS_URI`, a `file:///cells/...` URI inside the read-only `/cells`
mount) — or the Zenoh broker override — are written into **every** service's
environment by `scripts/generate_compose.py` (`_x_ros_env`), so the planner joins
the same graph as the dashboard by construction. `rclpy.init(args=None)` and the
RMW simply read them. Nothing here defaults: `plan_progress.transport()` reports
what it was given and reports absence as absence, surfaced in
`describe()["transport"]` and on `/health`, because a domain or RMW mismatch is
otherwise invisible — both sides are healthy and never hear each other.

---

## Repository Layout

```
multi_robots_agents/
│
├── agents/
│   ├── orchestrator/
│   │   ├── agent.py               # create_deep_agent with orchestrator skills
│   │   ├── server.py              # A2A HTTP :8889 + ROS 2 node
│   │   └── tools/                 # Python tools bound to the agent
│   │       ├── cell_spec.py       # read_cell_spec(), publish_to_ros()
│   │       └── trajectory_client.py  # call_trajectory_agent()
│   │
│   ├── trajectory/                # coordination + the external interface
│   │   ├── agent.py               # LangGraph planning agent (strategy, modes)
│   │   ├── mcp_server.py          # MCP :8890 — plan_trajectories, replan_operation, …
│   │   ├── tools/                 # backend-NEUTRAL: tool surface, gates, tree, cache
│   │   │   ├── planner_backend.py # THE facade: selection, dispatch, return home
│   │   │   ├── planning_tools.py  # LLM tools (MRTA, run_trajectory_planner, validators)
│   │   │   ├── behaviour_tree.py  # assembler + inter-arm step gate + home step
│   │   │   ├── coordination.py    # coordinated mode: completion gates, scheduling
│   │   │   ├── robot_kinematics.py, pinocchio.py, path_metrics.py, …  # shared geometry
│   │   │   └── ompl_planner.py, ompl_compound_planner.py, rrt_planner.py
│   │   │                          #   aliases of the moved backends (old import paths)
│   │   └── backends/              # ONE PACKAGE PER MOTION PLANNER
│   │       ├── contract.py        # BackendSpec + the planning/result contract
│   │       ├── __init__.py        # registry: built-ins + plugins (entry points / env)
│   │       ├── ompl/              # production backend + default return-home planner
│   │       ├── ompl_compound/     # all moving arms of a step in one state space
│   │       └── rrt/               # legacy Space-Time RRT (non-conforming)
│   │
│   ├── cell/
│   │   ├── agent.py               # create_deep_agent with cell skills
│   │   ├── server.py              # ROS 2 node + behaviour tree subscriber
│   │   └── tools/
│   │       ├── dispatch.py        # publish_task_assignment(), check_gates()
│   │       ├── cell_state.py      # publish_robot_description(), publish_scene()
│   │       └── emergency_stop.py  # trigger_estop()
│   │
│   └── robot/
│       ├── robot_agent.py         # ROS 2 node — FJT executor, one per robot
│       └── drivers/
│           ├── base_driver.py
│           ├── driver_factory.py
│           └── ros2_control_driver.py
│
├── skills/
│   │
│   ├── orchestrator/              # Skills for the orchestrator agent
│   │   ├── run-welding-program/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── task-json-format.md
│   │   ├── monitor-execution/
│   │   │   └── SKILL.md
│   │   └── handle-recovery/
│   │       └── SKILL.md
│   │
│   ├── trajectory/                # Skills for the trajectory agent
│   │   ├── optimize-sequence/
│   │   │   ├── SKILL.md           # MRTA 6-stage pipeline instructions
│   │   │   ├── scripts/
│   │   │   │   └── mrta_pipeline.py   # deterministic math (TSP, 2-opt, thermal)
│   │   │   └── references/
│   │   │       └── mrta-theory.md     # MRTA theory + bodyshop practices
│   │   │
│   │   ├── plan-trajectories/
│   │   │   ├── SKILL.md           # RRT + spline instructions
│   │   │   ├── scripts/
│   │   │   │   └── rrt_planner.py  # Space-Time RRT implementation
│   │   │   └── references/
│   │   │       ├── operation-params.md  # approach, hold, velocity per op type
│   │   │       └── joint-limits.md      # KR240, KR70 limits
│   │   │
│   │   ├── find-interlocks/
│   │   │   ├── SKILL.md           # interlock strategy instructions
│   │   │   ├── scripts/
│   │   │   │   └── interlock_optimizer.py  # candidate gen + greedy pairing
│   │   │   └── references/
│   │   │       └── interlock-strategies.md
│   │   │
│   │   ├── validate-collision/
│   │   │   ├── SKILL.md
│   │   │   └── scripts/
│   │   │       └── trajectory_validator.py
│   │   │
│   │   ├── check-reachability/
│   │   │   ├── SKILL.md
│   │   │   └── scripts/
│   │   │       └── ik_check.py
│   │   │
│   │   ├── build-behaviour-tree/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── bt-node-types.md
│   │   │
│   │   └── trajectory-knowledge/
│   │       ├── SKILL.md
│   │       └── scripts/
│   │           └── kb_query.py
│   │
│   └── cell/                      # Skills for the cell agent
│       ├── interpret-behaviour-tree/
│       │   └── SKILL.md
│       ├── manage-gates/
│       │   ├── SKILL.md
│       │   └── references/
│       │       └── gate-conditions.md
│       └── collision-response/
│           └── SKILL.md
│
├── cells/                         # Cell definitions — TWO LAYERS, see below
│   ├── <cell_id>/                  # ── the cell ROOT: the scene bundle, as its
│   │                               #      author shipped it. The converter writes
│   │                               #      NOTHING here — not one file.
│   │   ├── scene.yaml              # the scene export the cell was built from
│   │   ├── robots/<model>/         # the bundle's dressed URDF/SRDF + meshes,
│   │   │                           #   REFERENCED in place — one file per arm,
│   │   │                           #   loaded by the planners AND RViz
│   │   ├── fixtures/<id>/          # fixture URDF + CAD (voxelised → obstacles)
│   │   ├── floorplans/             # 2D CAD overlay: viewer only, not collision
│   │   ├── urdf/<ns>/<model>.urdf  # OLDER cells only: a baked copy. Still read
│   │   ├── srdf/<ns>/              #   (cell_layout.robot_urdf answers for both
│   │   │                           #   shapes), no longer produced.
│   │   ├── config/                 # OLDER cells only (welding_cell_2_small): the
│   │   │                           #   config dir before it moved below. Read as
│   │   │                           #   a fallback, never written to.
│   │   └── agentic_planner/        # ── this application's WORKSPACE: derived ──
│   │       ├── cell.yaml           # the MANIFEST: per arm, cell-local id ↔
│   │       │                       #   scene id + the urdf:/srdf: describing it.
│   │       │                       #   Written instead of copying anything. Its
│   │       │                       #   paths are relative to the CELL dir, not to
│   │       │                       #   this file, so they name the bundle's layout
│   │       ├── config/robots.yaml  # + the DECLARED tool frame (tip_link,
│   │       │                       #   tcp_offset) — no URDF is written, so
│   │       │                       #   utils/arm_model.py attaches it at load
│   │       ├── config/operations.yaml  # weld_points / rivet_points / glue_paths
│   │       ├── config/obstacles.yaml   # the CALCULATED obstacle model
│   │       ├── config/cyclonedds.xml   # read by the DDS C runtime via
│   │       │                       #   CYCLONEDDS_URI — the compose generator
│   │       │                       #   computes that URI with
│   │       │                       #   cell_layout.config_uri, so the env var
│   │       │                       #   follows the file
│   │       ├── tasks/<program>.json
│   │       ├── plans/<tree_id>.json
│   │       ├── logs/run_<run_id>.json
│   │       └── recordings/
│   │           ├── <tree_id>.rrd
│   │           └── trajectory_index.json  # warm-start KB index
│                                   # A re-convert OVERWRITES the derived files
│                                   #   above in place: no archive, no versioning.
│
├── ros2_packages/
│   └── agent_interfaces/
│       ├── msg/
│       │   ├── TaskAssignment.msg
│       │   ├── RobotStatus.msg
│       │   ├── CollisionAlert.msg
│       │   ├── CollisionPair.msg
│       │   ├── BehaviourNode.msg
│       │   └── StatusRequest.msg
│       ├── srv/
│       │   └── ValidateTrajectory.srv
│       └── action/
│           └── ExecuteOperation.action
│
├── controller/
│   └── pinocchio_robot_controller.py  # FJT action server (sim backend)
│
├── dashboard/
│   └── web_control_panel.py       # Flask :8888 + MCP client to trajectory-agent
│
├── utils/
│   ├── sim_config.py
│   └── cell_config.py
│
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.app
│   └── requirements.txt
│
├── scripts/
│   ├── generate_compose.py
│   └── generate_cell_urdfs.py
│
├── docker-compose.yml
└── Makefile
```

---

## Trajectory Agent — Specialized Deep Agent + MCP

The trajectory agent is the most important agent in the system. It is built as
a **LangGraph specialized agent** (not a general-purpose ReAct loop) with:

### Skills as `SKILL.md` directories

Each capability is a self-contained skill directory under `skills/trajectory/`.
The agent sees only the `name` and `description` at startup. When a task
matches, it reads the full `SKILL.md` and runs scripts from `scripts/` via
deterministic Python — not LLM-generated code.

```
skills/trajectory/optimize-sequence/SKILL.md

---
name: optimize-sequence
description: >-
  Allocate weld/rivet/glue operations to robots and compute the optimal
  execution sequence using MRTA best practices. Use when given a list of
  operation points that must be distributed across robots.
license: Apache-2.0
metadata:
  robot-type: any
  operations: weld, rivet, glue, laser_cut, mark
---

# optimize-sequence

## When to use
After receiving operation points from the orchestrator.
Before planning any trajectories.

## Steps

1. Run `scripts/mrta_pipeline.py` with the operation list.
   The script runs: IK pre-filter → regional allocation → TSP 2-opt →
   thermal spread → least-movement pairing → cycle-time balance.
2. Review the returned `steps[]` and `unreachable[]`.
3. If cycle_time_balance < 0.9, activate the `find-interlocks` skill
   to see if interlock optimisation can improve balance.

## Parameters
See `references/operation-params.md` for approach_offset, hold_duration,
velocity_scale, and thermal_gap per operation type.
```

### Exposed as MCP

`agents/trajectory/mcp_server.py` wraps the agent using FastMCP over SSE:

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("trajectory-agent", transport="sse")

@mcp.tool()
def plan_trajectories(cell_id: str, operations: list[dict]) -> dict:
    """Plan collision-free parallel trajectories for all operations in a cell.
    Returns a BehaviourTree JSON the cell agent can execute."""
    ...

@mcp.tool()
def check_reachability(cell_id: str, robot_ns: str, x: float, y: float, z: float) -> dict:
    """Check if a target position is reachable by a robot via Pinocchio IK."""
    ...

@mcp.tool()
def find_interlock_set(cell_id: str, targets: list[dict], operation_type: str = "weld") -> dict:
    """Find safe staging positions enabling parallel motion for ANY robot count.

    targets: [{"robot_ns", "x", "y", "z"}, ...]. Accepts a strategy only when every
    robot pair clears the margin. find_interlocks(robot1/robot2) is the 2-arm shim.
    """
    ...

@mcp.tool()
def validate_trajectory(cell_id: str, trajectories: dict) -> dict:
    """Validate trajectory pairs for inter-robot collision."""
    ...

@mcp.tool()
def get_trajectory_knowledge(cell_id: str, operation_type: str, robot_ns: str, target: list) -> dict:
    """Get warm-start joint configuration from trajectory knowledge base."""
    ...

# Task catalogue + planning history — so an MCP client (e.g. an IDE agent) can do everything
# the A2A dashboard does for planning, without the orchestrator's /a2a/* routes:
@mcp.tool()
def list_task_programs(cell_id: str) -> dict:
    """List the cell's task programs (names + op summary). Mirrors GET /a2a/tasks."""
    ...

@mcp.tool()
def get_task_program(cell_id: str, program: str) -> dict:
    """One task program's JSON; its `operations` array feeds plan_trajectories directly."""
    ...

@mcp.tool()
def save_task_program(cell_id: str, program: str, task: dict, overwrite: bool = False) -> dict:
    """Write tasks/<program>.json. Runs NOTHING. The only sanctioned way for a caller
    that is not this agent to author a program — the alternative every caller reached
    for was writing the shared volume itself, with its own idea of the schema.
    Refuses an existing name without overwrite=True."""
    ...

@mcp.tool()
def list_plan_traces(cell_id: str, limit: int = 50) -> dict:  # + get_plan_trace(cell_id, trace_id="latest")
    """Recorded planning-decision traces (per-weld decisions). Mirrors /a2a/plan_traces."""
    ...

@mcp.tool()
def get_plan_quality(cell_id: str, tree_id: Optional[str] = None) -> dict:
    """How GOOD a plan is: efficiency score, its four weighted components, the
    numbers behind them, per-leg metrics and the ranked findings. `source` says
    whether this is the report save_plan already wrote ("stored") or a fresh score
    of an older plan ("computed"). A FAILED run is refused, not scored."""
    ...
```

Any MCP client can call these tools:
- **IDE agents** — via the IDE's MCP client config; with the task-catalogue + planning-history
  tools above plus the cell-controller tools, the IDE's agent covers the full
  planning loop (list → plan → inspect → replay) that the A2A dashboard offered —
  the dashboard now only hosts the ROS/DDS runtime view (RViz, live feeds).
- **Dashboard chat** — `langchain-mcp-adapters` loads the tools into the LLM
- **Orchestrator** — calls `plan_trajectories` to get the BehaviourTree
- **Claude Desktop** — for offline planning and debugging

### How good the plan is, in the dashboard

A plan that executes is not the same as a plan worth executing: every motion is
sampled then shortcut, so the failure mode is silent — the path is valid,
collision-free, and four times longer than it needs to be. `save_plan` therefore
scores every plan as it stores it, into `logs/<tree_id>.json`.

The BTree pane **publishes the metric and nothing else**: the **Performance** card
above the tree drawing holds the weighted score, the verdict, the makespan and leg
count, and a line saying how much is behind it (the finding count, plus
"components, per-leg numbers and clearance"). A card that carried the whole report
made the pane a wall of numbers nobody scans, and the one number that should be
readable at a glance competed with thirty that should not.

**The detail opens on a click, in a dedicated window** (`#bt-kpi-modal`): the four
components the score is made of (`path` 0.40, `schedule` 0.35, `smoothness` 0.15,
`dwell` 0.10) with their weights, the numbers behind them, every finding worst
first naming its leg and by how much, the worst legs in detail, and the achieved
clearance. Three ways out — the close button, the backdrop, `Esc` (a named
`kpiKey` handler, so it is reachable from a test and not only from a listener).
The window is a **sibling of the layout**, not a child of the pane: `position:fixed`
inside a resizable, scrolling pane gets clipped by it.

Opening it issues **no request**. The single `get_plan_quality` answer fills the
card and the window in the same pass, so the report you open is the plan the card
is measuring — a fetch on open could answer for a plan the operator has since
changed away from. The converse matters as much: selecting another plan clears both
surfaces and closes the window **synchronously, at the top of `loadTree`**, before
the new answer is in flight. Otherwise, for the length of that fetch, one plan's
score is published under another plan's name, and if the newly selected run turns
out not to be scorable at all — a FAILED one — an open window would keep showing
the previous plan's report with a failed run behind it.

The panel computes none of it. It asks `GET /api/plans/<tree_id>/quality`, which
asks the trajectory agent's `get_plan_quality` over MCP — the same direction as
every other question the panel asks about a cell. The metric definitions live in
`agents/trajectory/tools/trajectory_quality.py` and the report belongs to the
agent that owns the cell's logs; a second implementation of "how good is this
plan" in the browser would be a second answer, and the two would drift the first
time a weight moved. The typed tool is on the **planner**, not on the cell-data
MCP: that one is byte-level on purpose and grows no per-artefact schemas.

Two things it must never do. **Clearances are reported and never scored** — a
score is a thing people optimise, and scoring a safety margin turns it into a
tunable; they appear in the window, alongside that sentence, and in no score block.
And an unreachable owner must read as *not measured, because X*, never as
a plan that scores zero: the route answers `200` with `success: false` and the
reason, and the card renders that reason in place of the numbers. Those are
opposite conclusions leading to opposite actions, and the wrong one is the one an
operator would act on.

### Tools are the math — LLM is the strategy

The LLM in the trajectory agent decides **what** to do: the allocation and order of
the operations, which arms work together in a step, per-weld standoffs and speeds,
via points and repositioning moves, and which recovery to try. The deterministic
tools (`agents/trajectory/tools/`, the backends) do the **math** and own every
decision that has one right answer:

* the weld pose's free spin and tilt (`weld_orientation.py`, run before dispatch);
* where a re-planned operation starts, and the on-axis lift-off out of a seam;
* every safety check, applied to every backend's output — the result contract
  (a leg starts where its arm stands), joint limits, the arm against itself along
  the executed path, static obstacles (one cached verdict per leg,
  `leg_verdict.py`, one fail-closed policy), the inter-arm gate (against every
  other arm where it stands, HOME included), the weld's working stroke
  (`validate_approach`, at assembly), the chain, and the return home;
* the pieces of a leg around the search — lift-off, descent, tool-rise cap,
  timing limits — shared by every backend (`leg_builder.py`);
* at execution: a leg is sent only from where the arm stands, is complete only
  when the arm reports its end, carries the planned velocities, and runs only
  while the collision monitor is watching (`bt_executor.py`,
  `collision_monitor.health`);
* what each failure means: every `reason_code` has one entry in
  `agents/trajectory/reason_codes.py` (retryable or not, the levers, where to
  escalate), attached to the failure as `recovery` and mirrored in the
  `recover-from-planner-failure` skill (a test keeps code, table and skill in
  step).

No LLM-generated math, and no safety property the LLM is trusted to check.

---

## Operation Types

Defined in `cells/<cell_id>/agentic_planner/config/operations.yaml`:

```yaml
operations:
  - id: "W001"
    type: weld               # weld | rivet | glue | laser_cut | laser_weld | mark | move
    position: [x, y, z]
    approach_vector: [ax, ay, az]
    face: front
    sequence_group: structural
    priority: 1
```

Default parameters per type live in `skills/trajectory/plan-trajectories/references/operation-params.md` — the agent reads this when activating the skill.

---

## Behaviour Tree Node Types

The format is standard **[BehaviorTree.CPP v4](https://www.behaviortree.dev/docs/intro)**
([repo](https://github.com/BehaviorTree/BehaviorTree.CPP)), so a plan opens in Groot2
and ticks in a C++ `BehaviorTreeFactory`. `agents/trajectory/tools/btcpp.py` is the
format layer (convert, emit XML, parse XML, dialect-agnostic accessors).

Built-in control nodes: `Sequence`, `Parallel` (`success_count="-1"` = all children),
`Delay` (`delay_msec`).

Custom leaves: `ExecuteTrajectory` and `MoveHome` and `WaitForRobot` (Actions),
`CheckSafety` (Condition).

Each `ExecuteTrajectory` node carries the ports:
- `robot_ns`, `operation_id`, `operation_type`
- `trajectory` — joint angles + timestamps
- `depends_on_robots` + `depends_on_positions` — position gate
- `has_interlock` + `interlock_target` — safe intermediate waypoint
- `hold_duration_s`, `velocity_scale`

**Blackboard.** A welding trajectory is hundreds of waypoints, so bulky ports
(`trajectory`, `interlock_target`, the `depends_on_*` lists) are not inlined: the port
value is `{key}` and the value lives in the tree's side-car `blackboard` dict — BT.CPP's
own idiom, and what keeps the XML readable. A tree must travel with its blackboard.

A timed stagger is a `Delay` decorator wrapping the leaf, not a field on it.

Trees recorded before the format change (`EXECUTE` / `PARALLEL` / `HOME` / …) still
replay, because every walker reads through `node_kind()` / `port()`.

Full port schema: [`skills/trajectory/build-behaviour-tree/references/bt-node-types.md`](skills/trajectory/build-behaviour-tree/references/bt-node-types.md).

---

## A2A and MCP Endpoints

The orchestrator's full A2A HTTP surface is specified in
[`docs/api/openapi.yaml`](docs/api/openapi.yaml); `tests/test_openapi_spec.py`
holds the spec and the routes actually served to each other. The table below
is the summary — the spec is the source of truth.

| Service | Endpoint | Protocol | Purpose |
|---------|----------|----------|---------|
| orchestrator | `:8889/health`, `/a2a/models`, `/a2a/tasks` | HTTP | Liveness, LLM catalogue, program list |
| orchestrator | `:8889/a2a/run`, `/a2a/task/submit` | HTTP | Dispatch a run (sync or `async: true` → poll `/a2a/result/{run_id}`) |
| orchestrator | `:8889/a2a/stop`, `/a2a/replay`, `/a2a/move` | HTTP | E-stop, replay a plan, jog a robot |
| orchestrator | `:8889/a2a/status`, `/a2a/plans`, `/a2a/recordings`, `/a2a/recordings/{id}` | HTTP | Live state, plan/recording indices |
| orchestrator | `:8889/a2a/runs`, `/a2a/result/{run_id}`, `/a2a/reasoning` | HTTP | Run history, envelope poll, LLM trace |
| orchestrator | `:8893/mcp` — `cell_read`, `cell_write`, `cell_append`, `cell_list`, `cell_summarize`, `cell_delete`, `cell_data_health` | MCP/HTTP | The cell's data, for the agents that do not own it |
| trajectory-agent | `/{cell_id}/plan_progress` | DDS | Live plan events, one topic per cell |
| trajectory-agent | `:8890/sse` | MCP/SSE | Tools callable by any MCP client |
| trajectory-agent | `:8890/mcp` | MCP/HTTP | Streamable HTTP alternative |
| cell-controller | ROS topic | DDS | Receives BehaviourTree |
| web-panel | `:8888` | HTTP | Dashboard UI + `/api/a2a/*` proxies (status is relayed) |
| web-panel | `:8765/sse` | MCP/SSE | MCP proxy for dashboard chat |

---

## Docker Services

| Service | Container | What runs |
|---------|-----------|-----------|
| `orchestrator` | `orchestrator` | `agents/orchestrator/server.py` A2A :8889 + cell-data MCP :8893 |
| `trajectory-agent` | `trajectory-agent` | `agents/trajectory/mcp_server.py` MCP :8890 |
| `cell-controller` | `cell-controller` | `agents/cell/server.py` |
| `collision-monitor` | `collision-monitor` | ROS 2 launch |
| `robot-agent-{ns}` | `robot-agent-{ns}` | `agents/robot/robot_agent.py` |
| `rviz-web` | `rviz-web` | RViz2 + noVNC :6080 (profile: rviz) |
| `web-panel` | `web-panel` | `dashboard/web_control_panel.py` :8888 + MCP :8765 |

---

## Why This Design is Clean

1. **No class inheritance for skills** — each skill is a `SKILL.md` directory,
   readable by humans, versionable by git, loadable by any Deep Agent

2. **Progressive disclosure** — the agent only loads what it needs; no 5000-line
   monolithic skill module loaded on every call

3. **Deterministic math in scripts** — Pinocchio IK, RRT, collision checks are
   Python scripts the agent runs; the LLM provides strategy, not math

4. **MCP exposure** — the trajectory agent is a first-class tool server;
   the dashboard, orchestrator, and developer tools all use the same interface

5. **Robot agent isolation** — one container per robot, clean restart,
   no coupling to the cell coordinator

6. **Operation-agnostic** — welding / riveting / gluing / laser use the same
   pipeline; only `operation-params.md` reference changes per type
