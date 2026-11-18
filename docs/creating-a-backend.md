# Creating a trajectory backend

A **backend** is the motion planner: it turns one step of motion requests (one
per arm moving in the step) into timed joint trajectories. Everything around it
is shared and stays in `agents/trajectory/`: allocation, the LLM's strategy and
planning modes, the inter-arm gate, the chain check, the BehaviourTree,
`replan_operation`, the MCP interface. A new planner, for example a robot-ballet
planner developed in another repository, implements one function and declares
itself. It does not edit this repository.

```
agents/trajectory/
  mcp_server.py, agent.py, …      coordination + external interface (unchanged by a backend)
  tools/planner_backend.py        THE facade: selection, dispatch, return home
  backends/
    contract.py                   BackendSpec + the contract below
    __init__.py                   registry: built-ins + plugins
    ompl/  ompl_compound/  rrt/   built-in backends (SPEC in __init__, code in planner.py)
plugins/                          out-of-repo backends, ignored by git
```

## 1. Implement the planning function

```python
def plan(cell_id: str, motion_requests: list[dict], *,
         approach_offset_m: float = 0.1,
         velocity_scale: float | None = None,
         **kwargs) -> dict:
```

**Input: one request per arm in the step**

```python
{"robot_ns": "robot1", "operation_id": "W05", "operation_type": "weld",
 "target": {"x": -0.8, "y": -1.51, "z": 1.35},      # TCP goal, world frame, m
 "approach_vector": [0, 0, -1],                     # tool axis into the target
 "orientation": {"x": 0, "y": 0.92, "z": 0, "w": -0.38}, "constrain_orientation": True,
 "approach": True,                                  # last segment = working stroke
 "approach_offset_m": 0.1, "retreat_offset_m": 0.1, "approach_velocity_scale": 0.25,
 "current_joints": [...]}                           # optional explicit start
```

**Output**

```python
{"success": True,                       # at least one trajectory
 "trajectories": {"robot1": [{"positions": [q1..q6], "time_s": 0.0, "phase": "transit"},
                             ...]},
 "failures": [{"robot_ns": "robot2", "operation_id": "W12",
               "error": "why", "reason_code": "no_path"}],
 "planning_time_s": 1.2,
 "backend": "robot-ballet"}
```

**What a backend does: the search, and only the search**

Your job is a collision-free joint path from where the arm stands to its
target, timed. Everything around that search is shared. Use it instead of
re-implementing it, so every backend leaves a seam, enters a weld and caps the
tool's rise in the same way:

| Piece | Where |
|---|---|
| The seam the arm starts in (where its TCP is, not how the start was given) | `robot_kinematics.seam_at_start` |
| The on-axis lift-off out of it, and the refusal when it is blocked | `leg_builder.cartesian_retreat`, `robot_kinematics.liftoff_blocked_failure` |
| The straight terminal descent into the weld, and its refusal | `leg_builder.cartesian_descent`, `leg_builder.descent_failure` |
| The tool-rise measure and the cap on an emitted leg | `leg_builder.trajectory_tool_rise`, `leg_builder.emitted_rise_limit`, `path_metrics._path_rise_policy` |
| Timing limits, HOME, where a leg ends | `leg_builder.accel_limits`, `leg_builder.home_config`, `leg_builder.trajectory_end`, `time_parameterization` |
| Models, spheres, obstacles, clearance band, speed margins | `robot_kinematics`, `pinocchio`, `time_parameterization` |

A backend never imports another backend's private helpers. If you need something
that only `backends/ompl/planner.py` has, it belongs in `tools/`. Move it there.

**Rules**

- **Chain every arm.** Start where the arm stands:
  - the request's `current_joints` when given;
  - else `trajectory_cache.get_last_config(cell_id, robot_ns)`;
  - else the arm's HOME.

  After planning, record the end with `set_last_config`. The facade refuses a leg
  that starts anywhere else (`backend_contract_violation`).
- **Honour each request's own fields.** These are `approach_offset_m`,
  `retreat_offset_m` and `approach_velocity_scale`. The call-level
  `approach_offset_m` is only the default.
- **Name the operation in every failure, with a `reason_code` from the vocabulary
  below.** The agent's recovery acts per operation and reads the code, not the
  error text. A code outside the vocabulary gets no guidance.
- **A step is refused as a whole** (ADR-0001) when your backend plans the step's
  arms together. Each partner gets `step_refused` with `caused_by`.

**What the platform does before and after you, whatever you return**

Before the call (`planner_backend.dispatch_plan`):

- It refuses two requests for one arm in one call (`duplicate_robot_request`).
- It searches the tool's free spin and tilt when the specified weld pose folds the arm into itself (`weld_orientation`). You receive the chosen pose.
- It refuses an arm whose required self-collision check is unavailable (`self_collision_unavailable`).

After the call, every returned leg is held to the contract and checked. A leg that fails is dropped, the backend's chaining for it is taken back, and a compound step's partners are dropped with it:

- **Shape:**
  - samples with finite `positions` of one length;
  - non-decreasing `time_s` from 0;
  - the first sample where the arm stands (`backend_contract_violation`).
- **Joint box:** every sample inside the arm's `robots.yaml` ∩ model limits (`joint_limit_violation`).
- **The arm against itself:** checked densely along the executed path (`trajectory_self_collision`).

  These leg checks are `tools/leg_verdict.py`. That is one cached verdict per leg, shared with every gate. A check that cannot run (no geometry stack) refuses nothing and says so. A check that ran and broke refuses the leg (`leg_check_error`).
- **Bookkeeping:**
  - each weld's pose is recorded for the next leg's lift-off and for the cell's e-stop home reset;
  - every failure is stamped with its `operation_id` and its `recovery` entry.

Later, in the planning tools and in the assembler:

- **Static obstacles** at the runtime monitor's margins, for every leg (`trajectory_in_collision`).
- **The inter-arm gate.** It is applied to every step against every other arm, where that arm stands: the end of its last leg, or HOME if it has not moved yet. The gate either runs the step together, serialises it, or refuses it. A leg that breaches on its own is reported as that leg's fault, not as a pairing problem. A gate that broke refuses (`inter_arm_check_error`).
- **The chain check:** each arm's first leg starts at HOME, and each leg starts where the previous one ended (`start_gap`, `chain_gap`).
- **The working stroke:** each weld leg is checked against its orientation, point and approach axis (`approach_mismatch`).
- **The gated return home** (`home_liftoff_blocked`, `home_return_refused`).

A backend that skips a property is caught, never trusted. Its plan is refused before it reaches a robot.

**Reason-code vocabulary.** These are the codes a backend emits. The full table, with what the agent does about each code, is `agents/trajectory/reason_codes.py`. `tests/trajectory/test_reason_codes.py` holds every emitted code to it and to the recovery skill.

| Code | Meaning |
|---|---|
| `ik_failed` | no joint solution for the target pose |
| `goal_in_collision` / `start_in_collision` | the endpoint breaches a static obstacle |
| `goal_self_collision` / `start_self_collision` / `approach_self_collision` | an endpoint, or the approach stroke, folds the arm into itself |
| `approach_unreachable` | the approach waypoint has no clear solution |
| `liftoff_blocked` | the on-axis lift-off out of the start seam is blocked |
| `no_path_found` | no path within the budget, although the endpoints are valid |
| `trajectory_in_collision` / `trajectory_self_collision` | the emitted leg breaches |
| `excessive_tool_rise` | the emitted leg arcs over the cap |
| `joint_limit_violation` | the emitted leg leaves the joint box |
| `inter_arm_clearance` | the arms of a step conflict (coupled backends) |
| `step_refused` (+ `caused_by`) | a partner of a refused compound step |
| `compound_start_invalid` / `compound_goal_invalid` | the joint start or goal of a compound step conflicts |
| `planner_error` | the planner itself failed |

## 2. Declare it

```python
# robot_ballet/umarell.py   (in YOUR repository)
from agents.trajectory.backends.contract import BackendSpec

SPEC = BackendSpec(
    name="robot-ballet",
    entry="robot_ballet.planner:plan",
    home="robot_ballet.planner:plan_home_trajectories",       # optional
    home_single="robot_ballet.planner:plan_home_trajectory",  # optional
    conforming=True,
    description="choreographed multi-arm timing",
)
```

- **`conforming=True`** is a claim that the backend honours the full contract:
  - inter-arm coupling within a step;
  - the orientation-constrained working stroke;
  - the two-phase transit and approach speed split.

  Say `False` until it does. Results are then stamped `non_conforming_backend: true`.
- **`options`** names the per-request options your backend takes, e.g.
  `options=("policy",)`. A request carries them as `backend_options` — the same
  field for every backend (MCP `plan_trajectories` / `replan_operation`, A2A,
  `make plan BACKEND_OPTIONS='{…}'`) — and your `plan()` receives them as keyword
  arguments. An option you do not declare refuses the request. `check_options`
  (`"module:function"`, raising `ValueError`) checks a value before the run
  starts, so a bad one is refused at once rather than in every step.
- **`joint_step=True`** declares that the arms of a step are planned (and refused)
  together, so `replan_operation` re-plans a PARALLEL step as one unit. `ompl-compound`
  and `roboballet` declare it.
- **`home` / `home_single`** are optional. Without them the default (OMPL) return-home planner is used.

## 3. Register it

Either way, the name becomes selectable everywhere.

- **Entry point.** Use this when the backend is an installed package:

  ```toml
  [project.entry-points."umarell.trajectory_backends"]
  robot-ballet = "robot_ballet.umarell:SPEC"
  ```

- **Environment.** No install needed: clone the backend into `plugins/` (the repo is mounted live into the trajectory agent) and run:

  ```bash
  make sim SIM=<cell> TRAJECTORY_BACKEND_PLUGINS="robot-ballet=plugins.robot_ballet.umarell:SPEC"
  ```

  A bare function (`…=plugins.robot_ballet.planner:plan`) also works, and is registered as non-conforming.

A **bundled** backend lives in this repository but registers like a plugin
(`backends.BUNDLED`): selectable by name, never the default. `roboballet` is one;
it also shows the smallest way to bring a new *search* while keeping the whole
step contract: it passes its search as `transit_solver` to
`plan_trajectories_ompl_compound` and inherits the strokes, timing and checks
(ADR-0002).

A plugin may not take a built-in or bundled name (`ompl`, `ompl-compound`, `rrt`, `roboballet`), and a malformed entry stops the agent at start-up instead of being dropped.

## 4. Select it

| Where | How |
|---|---|
| Stack default | `make sim TRAJECTORY_BACKEND=robot-ballet` |
| One plan | MCP `plan_trajectories(…, backend="robot-ballet")`, A2A `"backend": "robot-ballet"`, `make plan BACKEND=robot-ballet` |
| One weld edit | MCP `replan_operation(…, backend="robot-ballet")` |
| Discovery | MCP `list_trajectory_backends()` |

The planning modes (`classic`, `stepwise`, `coordinated`) work with any backend. They decide which requests share a call: a parallel step's arms together in `classic` and `stepwise`, one arm per call in `coordinated`.

## 5. Test it

Use the in-repo backends' tests as the model. The built-ins' tests are in `tests/backends/<name>/`; `tests/backends/test_backend_registry.py` shows how a backend is registered and dispatched in a test. As a minimum:

- the result carries every `RESULT_KEYS` key (`agents.trajectory.backends.contract.missing_result_keys`);
- each arm's trajectory starts where the arm stands (`current_joints`, else `get_last_config`, else HOME) and the end is recorded. The facade's contract check refuses it otherwise;
- a failure names its `operation_id` and carries a `reason_code` from the vocabulary;
- a two-arm step's legs pass `validate_trajectory_set` when they should, and fail when they must.
