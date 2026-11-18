# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project does not
publish versions yet, so changes collect under **Unreleased** until it does.

## [Unreleased]

### Added

#### Planning
- **One package per trajectory backend** (`agents/trajectory/backends/`).
  - **Layout:** `ompl/`, `ompl_compound/`, `rrt/` each declare a `BackendSpec` and keep
    their planner in `planner.py`; `contract.py` is the interface; the facade
    (`tools/planner_backend.py`) and the coordination layer are shared.
  - **Plugins:** a backend from another repository registers by the
    `umarell.trajectory_backends` entry point or `TRAJECTORY_BACKEND_PLUGINS`
    (code may live in `plugins/`). Without a spec it is non-conforming; a name
    colliding with a built-in is refused. Guide: `docs/creating-a-backend.md`.
  - **Return home** goes through the facade: the selected backend's home planner,
    else the default (OMPL) one, as before.
  - **MCP:** `list_trajectory_backends`.
  - **Compatibility:** `agents.trajectory.tools.{ompl_planner,ompl_compound_planner,
    rrt_planner}` are aliases of the moved modules (the same module objects).
    Path metrics moved to `tools/path_metrics.py`. Backend tests moved to
    `tests/backends/<name>/`.
- **Backend and mode chosen per request** (`agents/trajectory/plan_selection.py`).
  - **Where:** `plan_trajectories` and `replan_operation` take `backend` and
    `mode`; `/a2a/plan` and `/a2a/run` take `"backend"` / `"mode"`;
    `make plan|task-mcp|task` take `BACKEND=` / `MODE=`.
  - **Defaults:** a request that names neither gets the stack default (`make sim
    TRAJECTORY_BACKEND= PLANNING_MODE=`), else `ompl` / `classic`. Every reply
    says which source applied.
  - **Validation:** an unknown name refuses the request.
  - **`/a2a/run`:** the choice is bound for the run, so the orchestrator LLM
    cannot drop it.
- **`replan_operation` uses the request's backend and mode.**
  - **Backend:** with `ompl-compound`, the operation's whole PARALLEL step is
    re-planned jointly, so it stays synchronised; later steps follow until
    every arm's chain closes.
  - **Mode** decides the new tree's shape: `classic`/`stepwise` rebuild the
    steps; `coordinated` builds the per-arm gated tree, from a step tree or
    from a coordinated tree (now editable).
- **`ompl-compound` trajectory backend** (`agents/trajectory/tools/ompl_compound_planner.py`,
  [ADR-0001](docs/adr/0001-ompl-compound-backend.md)).
  - **Planning:** the transit of every arm moving in a step is one path in an OMPL
    `CompoundStateSpace`. Each state is checked arm-vs-environment and arm-vs-arm
    (`collision.warning_distance` + 0.01 m), and the path is timed as one stacked
    vector so all arms share timestamps.
  - **Approach strokes:** lift-off and descent stay per arm, as in `ompl`.
  - **Candidates:** the direct line, then staggered "one arm leads" lines, then
    RRT solves. The fastest candidate under the tool-rise target wins.
  - **Refusal:** a step that fails any check is refused as a whole.
  - **Selection:** `TRAJECTORY_BACKEND=ompl-compound`, `--backend ompl-compound`,
    `"backend"` on `/a2a/plan`, or the dashboard.
  - **Measured on `welding_cell_2_small`:** W03+W13 planned in 8 s and W02+W14 in
    7 s. `ompl` took 66 s and 316 s on those steps and refused robot2 both times.
- **Operator directives on a planning request** (`agents/trajectory/operator_directives.py`).
  - `assignments`: hard operation→robot locks that the allocator, motion planner
    and re-allocator all enforce.
  - `skill_priorities`: per-run skill overrides, applied on top of `skills.yaml`
    and never written to it.
  - `instructions`: shown to the model as a binding section.
  - Anything unknown refuses the request.
- **Re-plan one operation of an existing tree** (`agents/trajectory/tools/plan_edit.py`).
  The step plan is recovered from the tree, only the chosen leg is re-planned
  (plus its arm's following legs while the chain is broken), and the result is
  saved as a new tree that names its source.
- **`suggest_plan_improvements` MCP tool** (`agents/trajectory/plan_advice.py`). It
  returns a one-line summary and a ranked list of problems, each with its evidence
  and a suggested fix. Sources:
  - the quality findings;
  - the planning trace: recurring planner failures, operations that took many
    attempts, step-budget use, motion planner vs. reasoning time;
  - the program's cycle-time target, when a program is named.
- **Planning progress in percent.**
  - `get_plan_status` returns `progress`: phase, percent, operations solved/total,
    attempts and failed attempts, read off the run's event stream.
  - The orchestrator copies it into `GET /a2a/result/<run_id>` while a plan runs.
  - Renamed operations (`W03_fix`, `R2_W13`) count toward their weld.

- **Speed-dependent clearance.** robots.yaml `stopping` (`reaction_time_s`,
  `brake_decel_m_s2`; converter flags `--stop-reaction-time` / `--stop-decel`,
  `make convert STOP_REACTION_TIME= STOP_DECEL=`) gives each sphere a stopping
  distance `v * t_r + v^2 / (2 a)` at its own speed.
  - **Where:** the inter-arm gate (both arms), the emitted-trajectory backstop and
    the runtime monitor. The monitor measures each sphere's speed, and uses the
    joint-limit bound when it cannot measure.
  - **Default:** the converter writes `null` with a REVIEW note, and speed is then
    not accounted for. `min_geometric_clearance_m` reports the clearance without
    it.

#### Execution
- **Opt-in lockstep execution**, enabled with `make sim … LOCKSTEP=1` (default `0`).
  - **Start:** the cell executor holds the legs of one `ompl-compound` step at a
    barrier and dispatches them at the same instant.
  - **Drift:** it e-stops the cell if an arm drifts off the shared clock by more
    than 0.5 s for 0.5 s.
  - **Refusal, without dispatching anything:** when a partner leg is missing or
    the members' `velocity_scale` differ.

#### Deployment
- **Split deployment:** `make sim ONLY=execution` also starts the dashboard and,
  unless told otherwise, the RViz viewer.
- **Scoped teardown:** `make down ONLY=execution|trajectory` takes down one side
  and leaves the network.
- **RViz labels:** weld points with a `label` in the task file and robots with a
  `label` in `robots.yaml` are labelled.
- **Design records:** `docs/adr/`, starting with ADR-0001.

### Changed
- **Planning model:** planning defaults to Sonnet (`PLANNING_LLM_MODEL` overrides
  it; a request's `model` still wins).
- **Prompt caching:** Bedrock prompt caching is applied per ReAct turn.
- **Fewer, smaller model turns:**
  - Tool results reach the model with waypoint arrays summarised.
  - The planner chains from the last emitted sample.
  - The assembler's step gate runs as each step is planned.
- **Published ports:** agent ports bind to `${PUBLISH_HOST:-127.0.0.1}`. Only the
  split-deployment pair (:8892, :8893) stays on all interfaces.
- **Loopback addresses:** a loopback `ORCHESTRATOR_HOST` / `TRAJECTORY_AGENT_HOST`
  falls back to the compose service name, instead of the container calling itself.

### Fixed
- **The arm spheres were not an envelope of the arm** (umarell issue 04). They sat
  on the kinematic skeleton and covered 1-29 % of the KR240's hull vertices, so the
  inter-arm gate passed two arms whose castings interpenetrated by 300 mm.
  - **Converter:** `scripts/convert_isaac_scene.py` fits each link's `<collision>`
    hull (`link_collision` in robots.yaml: 662 spheres at 0.12 m, 100 % of the
    vertices covered).
  - **Checks:** the inter-arm gate, the static checks (OMPL validity, the
    emitted-trajectory backstop) and the runtime monitor check skeleton, tool and
    envelope.
  - **Cost:** a per-link bounding-sphere branch and bound for the pair check, and
    a broad phase for the box check (welding_cell_2_small: 1.2 ms per configuration
    with 752 spheres against 737 boxes).
- **The per-step gate was blind to the fixture** (umarell issue 06).
  `validate_trajectory_set` also checks every leg against the static obstacles
  and the arm itself. A breach sets `collision_free: false` and lists
  `static_collisions`; `check_static_obstacles=False` turns it off.
- **The timed path left the path the planner validated** (umarell issue 18).
  TOPP-RA was given a cubic spline at a uniform parameter, which overshot between
  unevenly spaced waypoints (up to 0.7 rad on corner-heavy transits).
  - **Timing:** `time_parameterization.parameterize` measures each candidate against
    the polyline and ships the fastest within the bound. Joint limits are checked
    on the timed path, not only at TOPP-RA's gridpoints.
  - **Candidates:** exact (each segment rest-to-rest) always; with a bound, also the
    spline through the waypoints and a corner-blended path.
  - **Bound:** the converter writes `path_following.max_deviation_rad` =
    `--path-tolerance` (default: the tool collision margin, 0.02 m) / the arm's lever
    arms from its URDF (`make convert PATH_TOLERANCE=`).
  - **Cells without it** follow the polyline exactly, stopping at every waypoint.
    That is slower (a 7-step descent takes 2.4x longer), so re-convert them.
- **robots.yaml was re-parsed on every check.** `cell_store.read_yaml` parses once
  per file content (libyaml when present) and hands each caller a copy.
- **Stored plans had no waypoints.** Since tool results were summarised for the
  model, the stored tree carried those summaries instead of waypoints. An affected
  plan replayed nothing yet reported `success: true` (`bt_1790951179`: 19 legs,
  0 waypoints). The plan is now taken from the assembler's full result, and a
  summarised tree is never stored.
- **Empty plans scored as acceptable.** A plan with no measurable motion scored
  0.85 "acceptable"; it now scores 0 with a `no_measurable_motion` finding.
- **Live planning feed missed successes.** A successful `run_trajectory_planner`
  result names robots, not operations, so solved welds never showed as `planned`.
- **Joint-state recorder stopped for good.** When the execution side started
  before the orchestrator, one failed read of the cell data ended the recorder;
  it now retries.

### Notes
- An edit of an `ompl-compound` step with a backend other than `ompl-compound`
  un-synchronises that step; with lockstep on, it is refused at the start barrier.
- Lockstep can only halt. No robot agent or controller supports a timed start,
  pause/resume or progress feedback yet.
