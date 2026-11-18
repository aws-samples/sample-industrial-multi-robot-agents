# Creating a new cell + authoring tasks

This guide covers the two things you do to make the system plan and execute work
for **your** robots:

1. **[Create a cell](#part-1--create-a-cell)** — turn a robot **URDF** and an
   **Isaac scene** into a `cells/<cell_id>/` definition the planner and executor
   understand.
2. **[Author a task](#part-2--author-a-task)** — describe the operation points
   (welds/rivets/…) to run, then plan and execute them.

The reference cell `cells/welding_cell_2_small/` (two KUKA KR240 arms spot-welding a
car-door fixture) is used as the worked example throughout — copy it as a
starting point.

---

## Part 1 — Create a cell

### What a cell is

A **cell** is a self-contained directory describing one physical work cell: the
robots and where they are bolted down, the workpiece/fixtures they operate on,
the static obstacles to avoid, and the jobs (tasks) to run. Everything the
planner and executor read for a cell lives under `cells/<cell_id>/`.

A cell is **two layers in one tree**, and the distinction is worth learning before
the layout below makes sense:

- The **cell root** is *input*, and it is **shared**: the scene bundle as its
  author shipped it — the robot URDF/SRDF, the meshes, the fixtures, `scene.yaml`.
  One cell may host several applications; they all read these files, so nothing at
  the root is namespaced to any one of them, and the converter adds **nothing** to
  it — not one file.
- **`agentic_planner/`** is *this* application's workspace — everything these
  agents author or compute: **`cell.yaml`**, **`config/`**, task programs, plans,
  run logs, recordings, and the calculated obstacle model. Isolating it means a
  second application dropped into the same cell cannot collide with our output.

**`config/` and `cell.yaml` are in the workspace, not at the root.** Both are
computed from the scene by the converter, so both are derived state like the plans
and the logs. Cells written before that move (`welding_cell_2_small`) keep them at the
cell root and are still read there — the agent copy wins when both exist, so
delete the old one rather than leaving config that reads as live and is not.

**The manifest moved; what is inside it did not.** Every path `cell.yaml` records
is relative to the **cell directory**, not to the manifest's own directory. Read
as manifest-relative, `robots/arm/arm.urdf` would mean
`agentic_planner/robots/arm/arm.urdf` — nothing — and each arm would silently fall
through to the guessed-`urdf/` layout with no geometry loaded.

One module, `utils/cell_layout.py`, answers every path inside a cell; nothing else
composes one by hand. A cell is on this layout **iff `agentic_planner/` exists** —
older cells (`welding_cell_2_small`, kept as a read-only archive) still resolve flat.

### Cell directory layout

A cell comes in one of **two shapes**, and one accessor
(`utils/cell_layout.py::robot_urdf`) answers for both, so no consumer needs to
know which it has:

- **Referenced** (what the converter writes today). The scene bundle's own
  `robots/`, `fixtures/` and `floorplans/` stay where they are, and `cell.yaml`
  names them. Nothing is copied, so there is exactly one file describing each arm.
- **Generated** (older cells, e.g. `welding_cell_2_small`). The converter had baked a
  dressed URDF per robot into `urdf/<robot_id>/` with the meshes copied alongside.
  Still supported; no longer produced.

```
cells/<cell_id>/                       ── the cell ROOT: shared INPUT ──
├── scene.yaml             #   the scene export the cell is built from: robot
│                          #     poses, fixtures, floor plan (INPUT, untouched)
├── robots/<model>/        #   the bundle's own dressed URDF + SRDF + meshes,
│                          #     REFERENCED in place by cell.yaml
├── fixtures/<id>/         #   the bundle's fixture URDF + CAD meshes
├── floorplans/            #   the 2D CAD overlay (viewer only, not collision)
└── agentic_planner/                   ── this app's WORKSPACE: DERIVED ──
    ├── cell.yaml              # GENERATED: the cell manifest — per arm, the
    │                          #   cell-local id ↔ the scene's own id, plus the
    │                          #   urdf:/srdf: that describe it. This is what
    │                          #   replaces copying. Its paths are relative to
    │                          #   the CELL dir, one level up, not to this file
    ├── config/
    │   ├── robots.yaml        # GENERATED: robots, base poses, joint limits,
    │   │                      #   tool frame (tip_link + tcp_offset), tool geometry
    │   ├── obstacles.yaml     # GENERATED: static collision obstacles
    │   │                      #   (voxelised fixtures + floor) — a CALCULATION
    │   └── cyclonedds.xml     # GENERATED: per-cell DDS config. Named to the
    │                          #   containers by cell_layout.config_uri, so the
    │                          #   CYCLONEDDS_URI env var follows the file
    ├── tasks/<program>.json   # ← YOU AUTHOR: the jobs to run  (see Part 2)
    ├── plans/                 #   auto: saved BehaviourTree plans (replayable)
    ├── recordings/            #   auto: .rrd recordings + trajectory_index.json (KB)
    └── logs/                  #   auto: run logs
```

You point the converter at `scene.yaml` (and, if the bundle's asset paths dangle,
at the workspace root they were written against — see
[`ISAAC_ROOT`](#isaac_root--resolving-the-paths-sceneyaml-names)); it **generates**
`agentic_planner/cell.yaml` plus `agentic_planner/config/{robots.yaml,`
`cyclonedds.xml,obstacles.yaml}`, and **creates the whole `agentic_planner/`
workspace**. You author `tasks/`; the rest is produced at runtime.

**It copies nothing.** A copied URDF is a second answer to "what shape is this
arm", and the two answers drift — the copy keeps the geometry the cell was built
with while the bundle moves on, and nothing reports the divergence. So the arm's
dressed URDF, its SRDF and every mesh stay where the bundle put them and
`cell.yaml` records the path.

The trade-off is explicit: a referenced cell is **not self-contained**. Its assets
must travel with it (they do when they sit under the cell directory, which is how
a handed-over bundle arrives) and must be mounted at the same relative path inside
a container. The converter records cell-relative paths for exactly that reason,
and warns when an asset resolves outside the cell.

> The **minimum** to bring a cell up is `agentic_planner/config/robots.yaml`,
> `agentic_planner/config/obstacles.yaml`, `agentic_planner/config/cyclonedds.xml`,
> and a way to reach each arm's URDF — `cell.yaml` naming it, or a copy under
> `urdf/<robot_id>/`. That is exactly the set the converter produces from
> `scene.yaml`.

### The fast path — convert the Isaac scene (recommended)

`scripts/convert_isaac_scene.py` builds a ready-to-run cell from a scene export.
It generates **every config file except the task**, and every one of them lands in
the workspace: `agentic_planner/cell.yaml`, then `robots.yaml`, `cyclonedds.xml`
and `obstacles.yaml` in `agentic_planner/config/`. The cell root is left exactly as
the bundle shipped it. It also creates the `agentic_planner/` workspace and its
subdirectories, so a freshly converted cell is ready to plan into.

**Inputs**

- `scene.yaml` — the only required input (see
  [Step 1](#step-1--bring-in-the-isaac-scene) for its shape). It names the robots,
  their poses, and the assets that describe them.
- `scene.usda` — **optional** (`SCENE=`). Used for two things only: the
  authoritative base-yaw quaternion, and fixture geometry as a fallback when a
  fixture's `urdf:` does not resolve. A bundle with no USD converts fine.
- A way to resolve the paths `scene.yaml` names, if they dangle: the **workspace
  root** they were written against (`ISAAC_ROOT=`) or a directory holding the arm
  models (`ROBOT_MODEL=`). Both only help *find* the assets — the cell references
  them where they are.

**Run it** — a bundle handed over as a cell, assets already beside `scene.yaml`:

```bash
make convert \
  OUT=cells/my_cell \
  SCENE_YAML=cells/my_cell/scene.yaml \
  TCP_OFFSET="-0.9092 0.0005 -0.0056"     # electrode tip offset in the flange frame
```

An Isaac export still in its authoring workspace, USD included:

```bash
make convert \
  OUT=cells/my_cell \
  SCENE_YAML=path/to/scene.yaml \
  SCENE=path/to/scene.usda \
  ISAAC_ROOT=path/to/isaac_workspace \
  TCP_OFFSET="-0.9092 0.0005 -0.0056"
```

Equivalent direct call:

```bash
python3 scripts/convert_isaac_scene.py \
  --scene-yaml path/to/scene.yaml \
  --isaacsim-root-cell-path path/to/isaac_workspace \
  --out-dir cells/my_cell \
  --tcp-offset "-0.9092 0.0005 -0.0056"
```

> **`TCP_OFFSET` is not optional in practice.** No URDF is written any more, so the
> tool frame is *declared* rather than baked: `robots.yaml` gets
> `end_effector_link: welding_tcp`, the `tip_link` read from the URDF, and this
> offset, and `utils/arm_model.py` attaches the frame when it loads the model.
> Leave it out and every weld is targeted at the flange — off by the whole tool
> length (~0.91 m on the reference arm), with no error anywhere. The converter
> warns; the warning is the only symptom.

#### `ISAAC_ROOT` — resolving the paths `scene.yaml` names

`scene.yaml` refers to its assets by **relative path**, written against the scene
folder *as it sits inside the Isaac authoring workspace*:

```yaml
dressed_urdf: ../urdf-dressed-robots/kr240r2900__sticktool/kr240r2900__sticktool.urdf
```

Hand the export over on its own and every one of those paths dangles — in the
reference export in this repo, all three do (the robot, the fixture URDF, and the
floor plan). `ISAAC_ROOT=` supplies the workspace root the `..` was walking up to.

With it, the converter **derives the robot model** from each robot's
`dressed_urdf`, so you do not name one yourself. Two robots sharing one dressed
model is handled — they get two `cell.yaml` entries pointing at the same file.

Resolution order per asset:

1. the scene folder, path as written — so a path that is already correct is never
   shadowed;
2. `ISAAC_ROOT` / `ROBOT_MODEL` / the cell directory, with the path as written and
   with the leading `..` peeled off;
3. as a **last resort**, any file of the same *name* under the cell directory.

Step 3 is what makes a rearranged bundle convert at all: a handed-over cell keeps
the same assets under `robots/`, `fixtures/`, `floorplans/`, so a path written
against `../urdf-dressed-robots/` resolves nowhere even though the file is right
there. It is a guess, so the converter says so out loud — check the match it
reports is the asset you meant.

An unresolved *fixture* is not fatal: its geometry falls back to `scene.usda`, or
to a placeholder box marked `REVIEW`. An unresolved *robot model* stops the run —
a `robots.yaml` with no joint limits would give you a planner planning against
nothing.

- **`OUT` is the target directory and is required.** Re-converting overwrites in
  place — see [below](#re-converting-an-existing-cell).
- For the cell to run it must ultimately live at `cells/<cell_id>/`. `<cell_id>`
  defaults to the `OUT` basename; override with `CELL_ID=` if you generate
  elsewhere first.
- One shared namespace `arms` is used by default (`--namespace` /
  `--namespace-per-robot` to change); robots are numbered `robot1..N` in scene
  order, and `cell.yaml` records each one's `scene_id` so the two names stay tied.

#### Re-converting an existing cell

Every file the converter writes is **derived** from `scene.yaml`, so a second run
simply recomputes it. Nothing is archived, nothing is renamed aside, and there is
no `FORCE=` flag:

```bash
make convert OUT=cells/my_cell SCENE_YAML=cells/my_cell/scene.yaml TCP_OFFSET="…"
```

Two things that survive a re-convert, and two that do not:

- **Survives:** everything under `agentic_planner/` that the converter does not
  write — your `tasks/*.json`, `plans/`, `logs/`, `recordings/`. The directories
  are created if missing and otherwise left alone.
- **Survives:** the bundle's own assets. They are inputs; the converter only reads
  them.
- **Overwritten:** hand-tuned `REVIEW` fields in
  `agentic_planner/config/robots.yaml`. Keep the
  values you settled on somewhere you can re-apply them.
- **Overwritten:** hand-refined boxes and any `is_workpiece: true` you set in
  `agentic_planner/config/obstacles.yaml`.

**What is derived vs what to review**

| Field | Source |
|---|---|
| `base_position` | `scene.yaml` `position` |
| `base_orientation_yaw` | the USD `xformOp:orient` when `SCENE=` is given, else `scene.yaml` `orientation.a` — the KUKA A angle about +Z, **in degrees**. A non-zero `b`/`c` is warned about and dropped: `robots.yaml` holds one yaw. |
| `joint_limits_*`, `joint_velocity_limits` | the referenced dressed URDF `<limit>` |
| `tip_link` | the child link of that URDF's last movable joint |
| `name`, `robot_model` | the resolved `dressed_urdf` filename |
| `cell.yaml` `urdf:` / `srdf:` | the resolved `dressed_urdf` and the `.srdf` beside it |
| fixture obstacle boxes | the fixture's own CAD meshes, reached through the `urdf:` in `scene.yaml`; the USD mesh data as fallback (see below) |
| `end_effector_link`, `tcp_offset` | `--ee-link` / `TCP_OFFSET=` — **not derivable; see the warning above** |
| `gripper`, `workspace_radius`, `planning_group`, `joint_acceleration_limits`, `rrt_*_range`, `tool_collision` | **defaults — marked `REVIEW`; tune to your robot/tool** |

**Obstacles — voxel occupancy by default.** Each fixture's own meshes are
rasterised into an occupancy grid and emitted as greedily-merged boxes covering
only where the fixture surface actually is, so cavities and openings stay clear and
a robot can still reach into them. The geometry is full-resolution CAD read through
the `urdf:` in `scene.yaml`, with each `<collision>` entry's `<origin>` and mesh
`scale` applied and the whole assembly placed at the fixture's world pose — the
reference fixture is 895 mesh entries and ~1.8 M triangles, so expect this step to
take a minute or two and print progress:

- `OBSTACLE_MODE=voxel` (default) with `VOXEL_SIZE=0.05` (m). Smaller = finer but
  more boxes; larger = fewer boxes but coarser (still safe — it over-approximates).
- `OBSTACLE_MODE=aabb` — one conservative solid bounding box per fixture instead.
- `is_workpiece` (the surface the tool tip may touch) is left `false`. Set it on
  the box(es) under your weld points — that is a semantic call best made with the
  task (see [Part 2](#part-2--author-a-task) and [Step 4](#step-4--write-configobstaclesyaml)).

- The **floor plan is not an obstacle.** It is a 2D CAD overlay for the viewer;
  cell walls are not modelled as collision geometry.

> **Do not run `make urdfs` on a converted cell.** A converted cell *references*
> its dressed URDF — that file is the bundle's input, not the cell's output, and
> `make urdfs` would rewrite it in place. The base pose is applied **at runtime**
> from `robots.yaml` (`base_position` / `base_orientation_yaw`) by the planners, so
> the URDF is meant to keep `base_link` at the origin; baking the pose in would
> double-apply it *and* corrupt the source bundle for every other cell using it.

Then skim the `REVIEW` fields (details in
[Step 3](#step-3--write-configrobotsyaml) / [Step 4](#step-4--write-configobstaclesyaml)),
and jump to [Step 7](#step-7--start-the-cell-and-sanity-check) to bring the cell
up, or [Part 2](#part-2--author-a-task) to author the task.

---

### Manual authoring & tuning

The steps below build the same files by hand. Use them to understand or tune
what the converter produced — adjusting `tool_collision`, refining obstacle
boxes, setting `is_workpiece` — or to create a cell without a scene export.

### Step 0 — Scaffold the directory

```bash
CELL=my_cell
# The cell root — shared input: the scene and the robot description.
mkdir -p cells/$CELL/{scene,urdf/meshes,srdf}
# This application's workspace — everything the agents author or compute,
# config/ included.
mkdir -p cells/$CELL/agentic_planner/{config,tasks,plans,logs,recordings}
# Reuse the reference DDS config verbatim:
cp cells/welding_cell_2_small/config/cyclonedds.xml \
   cells/$CELL/agentic_planner/config/cyclonedds.xml
```

> Creating `agentic_planner/` is what puts the cell on the two-layer layout — the
> directory's existence *is* the flag, so create it before writing anything into
> it. `make convert` does this for you; by hand, do it in Step 0 as above and every
> later step lands where the code will look for it.

> `cyclonedds.xml` is read only under the default `RMW=cyclonedds`. A cell started
> on `RMW=zenoh` ignores it and connects to the broker named by `ZENOH_ROUTER`
> instead (see the README) — optionally with a `zenoh_session.json5` beside it in
> the same `config/` for TLS or access control. Copy the DDS config anyway: the transport is chosen
> per startup, not per cell, so the same cell must work either way.

### Step 1 — Bring in the Isaac scene

Export your Isaac Sim layout. `make convert` reads it where it is and never copies
it, so the export stays the input it is — put it at the cell root (a handed-over
bundle already arrives that way) or leave it in the authoring workspace and point
`SCENE_YAML=` at it:

- `scene.usda` — the USD scene (optional; visual reference / re-import).
- `scene.yaml` — the structured export the cell is built from. It lists
  each robot's world **position** and **orientation**, the **fixtures**, and the
  **floor plan**. Example (from the reference cell):

  ```yaml
  robots:
    - robot_id: kr240r2900__sticktool
      dressed_urdf: ../urdf-dressed-robots/kr240r2900__sticktool/kr240r2900__sticktool.urdf
      position: { x: 0.069, y: -3.318, z: 0 }
      orientation: { a: 0, b: 0, c: 0 }      # KUKA A/B/C in DEGREES; a = yaw about Z
      home_pose_deg: [0, -90, 90, 0, 0, 0]
    - robot_id: kr240r2900__sticktool_2
      position: { x: -1.183, y: 1.042, z: 0 }
      orientation: { a: 0, b: 0, c: 0 }
      home_pose_deg: [0, -90, 90, 0, 0, 0]
  fixtures:
    - id: FIXTURE_A
      urdf: ../urdf-fixture-collection/FIXTURE_A/FIXTURE_A.urdf
      position: { x: -0.982, y: -1.239, z: 0 }
      orientation: { a: 0, b: 0, c: 0 }
  floor_plan:
    geometry_file: ../floor-plans/hall-a-floor-plan.obj
    scale: 1
  ```

`scene.yaml` is the bridge between Isaac and the planner: robot `position`/
`orientation` become `base_position`/`base_orientation_yaw` in `robots.yaml`
(Step 3), and the `fixtures` become the obstacles in `obstacles.yaml` (Step 4).

> Note the `../` in `dressed_urdf`, `fixtures[].urdf`, and
> `floor_plan.geometry_file`. Those paths are relative to the scene folder **as it
> sits inside the Isaac workspace**, so they dangle the moment the export is copied
> out on its own. Pass
> [`ISAAC_ROOT=`](#isaac_root--resolving-the-paths-sceneyaml-names) and the
> converter resolves them, copies the assets in, and rewrites this file to point at
> the local copies.

> The simulator itself is optional at runtime. The motion backend is selected
> per run with `ROBOT_CONTROLLER`:
>
> - `sim` (default) — one **Pinocchio FJT controller per robot** simulates the
>   motion and publishes `joint_states`.
> - `isaac` — one **Isaac Sim FJT bridge per robot**
>   (`controller/isaac_sim_controller.py`). It still serves the
>   `FollowJointTrajectory` action the robot-agent dispatches to, but converts
>   each goal into Isaac's `/{scene_root}/{prim}/joint_command` topic and relays
>   Isaac's `joint_states`/`contacts` back onto the cell contract. Isaac Sim is
>   the physics; the controller is *replaced*, not disabled.
> - `none` — no controller container. An **external** process must both serve
>   the FJT action **and** publish `joint_states`; otherwise the robot-agent's
>   dispatch finds no action server and the arms never move.
>
> The `isaac` bridge and `none` mode both require Isaac Sim, the robot-agent, and
> the bridge to share one `ROS_DOMAIN_ID` (a ROS 2 node lives on a single
> domain). The reference Isaac layout uses domain `0`.

### Step 2 — Add the robot model (URDF + meshes)

The cell holds **one** description of each arm's shape: a URDF per robot id, and a
single shared mesh directory beside them.

```
cells/my_cell/urdf/
├── meshes/                       # the arm/tool geometry, once for the cell
├── robot1/kr240r2900__sticktool.urdf
└── robot2/kr240r2900__sticktool.urdf
```

Copy the "master" arm URDF to `urdf/<robot_id>/<model>.urdf` for each robot id and
its meshes to `urdf/meshes/` — Step 6 patches those URDFs with the correct world
origin and rewrites their mesh references to `file:///cells/<cell>/urdf/meshes/…`.
There is deliberately no second, un-patched copy kept elsewhere in the cell: two
descriptions of one arm eventually disagree about what the planner is planning
around, and the planners, the collision monitor and RViz all load this same file.

**The tool is parameters, not geometry.** Do *not* bake the welding gun into the
tip link's collision mesh: the planner reads the tool's collision volume from
`robots.yaml` (`tool_collision`, Step 3) and the converter *creates* the
`welding_tcp` frame as a fixed child of the tip link at `--tcp-offset`. A baked-in
mesh would be checked twice, once as link geometry and once as `tool_collision`.
The reference cell's flange is bare — `link_6`'s collision hull spans just
`[0.036, 0.19, 0.19]` m — and its 0.92 m torch exists only in `tool_collision`.

This is also why `--tcp-offset` cannot be inferred from the export: neither
`scene.usda` nor the dressed URDF contains a tool link, a torch mesh, or a tip
frame. Measure the electrode tip in the flange frame and pass it.

### Step 3 — Write `agentic_planner/config/robots.yaml`

This is the core cell description. The structure is
`robots → <namespace> → <robot_id> → { config }`. A **namespace** is the shared
ROS namespace for a group of arms; each **robot_id** is a unique arm. Topics are
`/{cell}/{namespace}/{robot_id}/...`.

```yaml
robots:
  arms:                       # namespace (shared) → ROBOT_NAMESPACE
    robot1:                   # robot id (unique) → ROBOT_ID; topics /{cell}/arms/robot1/...
      name: kr240r2900__sticktool
      robot_model: kr240r2900           # → ROBOT_MODEL; also picks the container image config
      base_position: [0.069, -3.318, 0.0]   # from scene.yaml robot position
      base_orientation_yaw: 0.0              # from scene.yaml orientation (yaw about Z, radians)
      gripper: welding_gun
      workspace_radius: 2.9                  # reach (m) — reachability pre-filter
      planning_group: kuka_arm               # SRDF group name
      end_effector_link: welding_tcp         # TCP frame used for IK/FK targets
      # Joint limits (radians), taken from the URDF:
      joint_limits_lower: [-3.23, -2.71, -2.27, -6.11, -2.27, -6.11]
      joint_limits_upper: [ 3.23,  0.61,  2.69,  6.11,  2.27,  6.11]
      # Velocity (rad/s) + acceleration (rad/s²) — used by TOPP-RA time-parameterization:
      joint_velocity_limits:     [1.83, 1.75, 1.87, 3.05, 3.05, 4.01]
      joint_acceleration_limits: [3.0,  3.0,  3.0,  6.0,  6.0,  8.0]
      # RRT safe-sampling ranges keep the arm raised to avoid self-collision:
      rrt_j2_range: [-1.5, 0.0]
      rrt_j3_range: [0.3, 2.5]
      # Tool/plate collision geometry in the flange frame — kept in the inter-robot
      # clearance check so the torch + mounting plate are never ignored:
      tool_collision:
        mount_link: link_6                   # substring of the flange link name
        axis:                                # torch/electrode down the tool axis
          direction: [-1.0, 0.0, 0.0]
          start_m: 0.15
          end_m: 0.92
          step_m: 0.15
        plates:                              # flat mounting plate(s)
          - center: [-0.289, 0.0038, -0.0063]
            rpy: [1.5708, 1.5509, -1.5708]
            size: [0.30, 0.45]
            grid: [5, 5]
    robot2:                   # second arm in the same namespace
      name: kr240r2900__sticktool
      robot_model: kr240r2900
      base_position: [-1.183, 1.042, 0.0]    # from scene.yaml
      base_orientation_yaw: 0.0
      # … same fields as robot1 …

# Cell-wide policies:
collision:
  min_clearance: 0.05
  warning_distance: 0.20
  check_frequency: 100.0
planning:
  default_planner: RRTConnect
  planning_time: 5.0
  # Speed & acceleration — the ONE place these numbers live. Speeds are fractions
  # of each robot's joint_velocity_limits; acceleration scales by the square.
  max_velocity_scaling: 0.7    # transit speed AND the ceiling on any speed asked for
  home_velocity_scale: 0.4     # planned return to HOME + lift-off out of the last seam
  acceleration_factor: 2.5     # accel = factor x vel, for a robot with no joint_acceleration_limits
  interlock_move_time_s: 2.5   # unplanned single-point move to an interlock pose
  home_return_time_s: 7.0      # unplanned fallback return to HOME
```

Key points:

- **`base_position` / `base_orientation_yaw`** must match `scene.yaml` exactly —
  they place the robot in the world so IK, collision, and inter-robot clearance
  are correct.
- **`end_effector_link`** is the frame IK/FK targets are expressed against
  (e.g. `welding_tcp`); the planner looks for a frame whose name contains
  `welding_tcp` or `tool0`.
- **`tool_collision`** matters for two-arm cells: it is what stops the torches /
  mounting plates from clipping during coordinated motion.
- The number of robots here determines how many `robot-agent-<id>` (and
  `robot-controller-<id>`) containers the compose generator creates.

### Step 4 — Write `agentic_planner/config/obstacles.yaml`

Model the fixtures/workpiece and any static structure from `scene.yaml` as
collision primitives. This file sits in the agent workspace, not at the cell root:
it is a *calculated* model of the scene — the converter voxelises `scene.usda` to
produce it — and belongs to whoever computed it, not to the cell's own
description. Three shapes are supported:

| `type` | Fields |
|---|---|
| `box` | `center: [x,y,z]`, `half_extents: [hx,hy,hz]` |
| `plane` | `normal: [nx,ny,nz]`, `offset: d` (free where `n·p + d ≥ 0`) |
| `sphere` | `center: [x,y,z]`, `radius` |

```yaml
obstacles:
  - id: floor
    type: plane
    normal: [0.0, 0.0, 1.0]
    offset: -0.05                 # 5 cm tolerance below z=0
    collision_margin: 0.05

  - id: fixture_wall_south
    type: box
    center: [-0.975, -2.075, 0.60]
    half_extents: [0.575, 0.125, 0.60]
    collision_margin: 0.05

  - id: weld_surface_panel        # the surface the weld points sit ON
    type: box
    center: [-0.975, -1.225, 0.70]
    half_extents: [0.275, 0.675, 0.50]
    collision_margin: 0.03
    is_workpiece: true            # arm links skip it; the tool TIP may touch it

# Clearance policy (see "How clearance is computed" below)
default_collision_margin: 0.05    # fallback safety standoff when a box omits one
robot_link_radius: 0.12           # arm body radius, NORMAL transit  → 0.12 + margin
approach_link_radius: 0.05        # arm body radius, TERMINAL stroke → 0.05 + margin
```

Notes:

- Mark the weld/work surface with **`is_workpiece: true`** so arm links avoid it
  but the tool tip is allowed to make contact (weld points sit on it). Set it
  only on the actual workpiece panel, **never a fixture rail** — an
  `is_workpiece` box is skipped for all arm links, so marking a real obstacle
  would let the arm pass through it.
- Center your boxes on the fixture origin from `scene.yaml`; small offsets
  (< `collision_margin`) are fine.

#### How clearance is computed (`link_radius` vs `collision_margin`)

A robot link is modelled as **one sphere**; an obstacle box keeps its **real
size** (it is *not* inflated). A link is in collision when:

```
distance(link_sphere_center, box_surface)  <  link_radius + collision_margin
```

The two terms look similar but mean different things — they are **not two safety
buffers stacked together**:

| Term | What it is | Whose property |
|---|---|---|
| `link_radius` (`robot_link_radius` / `approach_link_radius`) | The arm's **physical half-thickness**. The sphere is the link's body envelope, so at `distance == link_radius` the metal is *touching* the box. It is geometry, not safety. | Robot (shared across all obstacles) |
| `collision_margin` | The **only safety standoff** — extra distance kept *beyond* physical contact, to absorb IK / trajectory-tracking / spline error. | Obstacle (per-box, `default_collision_margin` as fallback) |

So for the defaults, `0.17 m = 0.12 m arm body + 0.05 m safety`. Only the 5 cm is
"security"; the 0.12 m is how fat the arm is. Removing `link_radius` would model
the arm as an infinitely thin wire that could pass through a rail while the check
reports "clear"; removing `collision_margin` would plan trajectories that graze
the fixture with zero standoff. They are additive but not redundant.

Why 0.12 m looks large: it is **one sphere per link**, so a single radius must
bound the *thickest* section of that link — thin parts (wrist, tool shank) are
over-approximated. That is a modelling simplification, not extra safety.

**Two link radii, selected per motion segment.** Because the single-sphere model
over-approximates the thin tool near the work point, the cell provides a reduced
radius for the final stroke:

| Radius | Used for | Clearance (with 0.05 margin) |
|---|---|---|
| `robot_link_radius` (0.12) | Normal transit — moving between points | 0.17 m |
| `approach_link_radius` (0.05) | The **terminal working stroke** into the point | 0.10 m |

The planner applies `approach_link_radius` **only** to the final approach segment
(the trajectory tool sets `approach=true` for that stroke); every other motion
keeps the full `robot_link_radius` clearance. This lets a leaned tool reach a
point close to the fixture without the normal 0.17 m envelope rejecting it, while
transit stays conservative. You can tune either radius per cell, and override the
margin per obstacle.

### Step 5 — Generate per-robot URDFs

> If you used the converter, this is already done — **skip it.** The default
> `ompl` / `rrt` (Pinocchio) planners apply `base_position` /
> `base_orientation_yaw` from `robots.yaml` at runtime, so they do **not** need
> the pose baked into the URDF; the converter leaves `base_link` at the origin.

For the manual path, place a copy of the model URDF under `urdf/<robot_id>/`,
then bake the world→base origin from `robots.yaml`:

```bash
make urdfs SIM=my_cell
# → patches cells/my_cell/urdf/<robot_id>/<model>.urdf, setting the world→base_link
#   joint origin to base_position / base_orientation_yaw, and copies meshes/
```

(Equivalent: `python3 scripts/generate_cell_urdfs.py --cell my_cell`.) Do not run
this on a converted cell — it would bake a pose the runtime already applies.

### Step 6 — Generate SRDFs (optional)

Not needed by either planning backend — both plan in-process from the URDF via
Pinocchio. The converter no longer writes one: a scene bundle ships its own SRDF
beside each dressed URDF, and `cell.yaml` references that. This generator is for a
cell built by hand, or one whose bundle shipped none (without an SRDF a collision
checker retains every self-collision pair, so links that touch by design report as
colliding at every waypoint). The SRDF is derived from the URDF so it can't drift:

```bash
python3 scripts/generate_srdf.py --cell my_cell
# → cells/my_cell/srdf/<robot_id>/<model>.srdf  (planning group, home state,
#   end-effector, adjacent-link self-collision disables)
```

The `ompl` (default) and `rrt` backends need **no** SRDF and no extra services.

### Step 7 — Start the cell and sanity-check

```bash
make compose SIM=my_cell       # generate .generated_compose.yml for this cell
make build   SIM=my_cell       # first time (build needs the generated compose)
make sim-detached SIM=my_cell  # or: make sim SIM=my_cell   (attached)
make health

# Confirm the robots were detected (one robot-agent per robot):
make compose-show SIM=my_cell | grep container_name

# Check reachability of a candidate point before authoring a whole task:
curl -s -X POST http://localhost:8890/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":1,"method":"tools/call",
  "params":{"name":"check_reachability",
            "arguments":{"cell_id":"my_cell","robot_ns":"robot1",
                         "x":-0.98,"y":-1.2,"z":1.35}}}'
```

Regenerate the compose after changing `robots.yaml` (e.g. adding a robot) with
`make compose SIM=my_cell`, then `make sim`.

---

## Part 2 — Author a task

A **task** (aka program) lives at
`cells/<cell_id>/agentic_planner/tasks/<program>.json` and lists the operation
points to perform. You describe **where** the work is; the
trajectory agent decides **which** robot does each point (MRTA allocation),
balancing reachability and cycle time. The only exception is an explicit
`assigned_robot` lock.

### Task JSON schema

Top level:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `program` | string | yes | Program name (match the file stem). |
| `cell_id` | string | yes | Cell this runs on. |
| `version` | string | no | Program version. |
| `description` | string | no | Human summary. |
| `trajectory_backend` | `"ompl"` \| `"rrt"` | no | Planner to bind for this task. If omitted, the launch-time `TRAJECTORY_BACKEND` default is used. No fallback between planners. |
| `extensions` | object | no | Process metadata (strategy, cycle_time, weld_params). Passed through; not required for planning. |
| `operations` | array | yes | The operation points (non-empty). |

Each entry in `operations`:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique id. Keep it robot-neutral (e.g. `W01`) — do not encode a robot in the id. |
| `type` | string | yes | `weld` \| `rivet` \| `glue` \| `laser_cut` \| `mark` \| `move`. |
| `position` | `[x,y,z]` | yes | Target point in the world/cell frame (metres). |
| `approach_vector` | `[ax,ay,az]` | yes | Unit direction the tool travels **into** the point (e.g. `[0,0,-1]` = descend from above). |
| `orientation` | `{x,y,z,w}` | no | Target tool orientation (world-frame quaternion). If omitted, derived from `approach_vector`. |
| `weld_orientation` | `{x,y,z,w}` | no | Optional **leaned** tool orientation applied as a terminal wrist adjustment at the point (e.g. tilt over an edge), while travel between points stays vertical. |
| `sequence_group` | string | no | Ordering group (e.g. `structural`, `secondary`). Controls order, not robot allocation. |
| `priority` | int | no | Ordering hint within/among groups. |
| `assigned_robot` | string | no | **Robot lock.** If set (e.g. `"robot1"`), that operation is pinned to that robot and never reallocated. Use only when a single robot must perform it. |

### Example

```json
{
  "program": "my_spot_welding",
  "cell_id": "my_cell",
  "version": "1.0.0",
  "description": "12 spot welds on the panel; outer columns lean outward.",
  "trajectory_backend": "ompl",
  "extensions": {
    "strategy": { "parallel_within_group": true },
    "cycle_time": { "target_s": 20.0, "max_s": 30.0 },
    "weld_params": { "weld_type": "spot", "welding_duration_s": 2.0 }
  },
  "operations": [
    {
      "id": "W01",
      "type": "weld",
      "position": [-1.15, -1.8, 1.35],
      "approach_vector": [0.0, 0.0, -1.0],
      "orientation":      { "x": 0.0, "y": -0.7071,   "z": 0.0, "w": 0.7071 },
      "weld_orientation": { "x": 0.0, "y":  0.923877, "z": 0.0, "w": -0.382689 },
      "sequence_group": "structural",
      "priority": 1
    },
    {
      "id": "W02",
      "type": "weld",
      "position": [-0.975, -1.8, 1.35],
      "approach_vector": [0.0, 0.0, -1.0],
      "orientation": { "x": 0.0, "y": -0.7071, "z": 0.0, "w": 0.7071 },
      "sequence_group": "structural",
      "priority": 1
    },
    {
      "id": "HANDLING_01",
      "type": "move",
      "position": [-0.9, -1.2, 1.8],
      "approach_vector": [0.0, 0.0, -1.0],
      "assigned_robot": "robot1"
    }
  ]
}
```

Tips:

- **`orientation` vs `weld_orientation`**: keep `orientation` vertical (top-down)
  so travel between points is clean; put the required lean in `weld_orientation`
  and the planner applies it as a terminal wrist tilt at the point (and retracts
  back out along that same approach axis).
- Coordinates are in the **world/cell frame** — the same frame as
  `robots.yaml base_position` and `obstacles.yaml`.
- Validate individual points with `check_reachability` (Step 7) before running a
  full program.

### Plan & execute the task

```bash
# List tasks the cell defines
make tasks SIM=my_cell

# Plan + execute through the orchestrator (A2A)
make task SIM=my_cell TASK=my_spot_welding

# …or directly through the cell MCP (bypass the orchestrator)
make task-mcp SIM=my_cell TASK=my_spot_welding
```

Planning persists a BehaviourTree to
`cells/my_cell/agentic_planner/plans/bt_<timestamp>_my_cell.json` and execution
records a `.rrd` under `cells/my_cell/agentic_planner/recordings/`. Replay a saved
plan without re-planning (deterministic — no LLM needed):

```bash
make recordings SIM=my_cell
make replay SIM=my_cell TREE_ID=bt_<timestamp>_my_cell
```

### What planning produces

The trajectory agent turns the task into a **BehaviorTree.CPP v4** tree of node types
`Sequence`, `Parallel`, `Delay`, `ExecuteTrajectory`, `WaitForRobot`, `CheckSafety`,
`MoveHome`. Each `ExecuteTrajectory` node carries a precomputed joint trajectory
(referenced from the tree's blackboard as `{key}`) plus gating ports
(`depends_on_robots`, `has_interlock`, `hold_duration_s`, `velocity_scale`); a timed
stagger is a wrapping `Delay`. The plan is saved as JSON and as Groot2-loadable XML.
The cell controller walks this tree deterministically, enforcing the gates and
running the collision monitor. See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for
node-type details.

---

## Checklist

Creating a cell — recommended (converter):

- [ ] The scene export: `scene.yaml` (+ `scene.usda`, optional)
- [ ] Its assets reachable: beside `scene.yaml`, under the cell, or via the
      workspace root they were written against (`ISAAC_ROOT=` / `ROBOT_MODEL=`)
- [ ] `make convert OUT=cells/<cell> SCENE_YAML=… [SCENE=…] [ISAAC_ROOT=…] TCP_OFFSET="…"`
- [ ] **Pass `TCP_OFFSET=`** — without it every weld targets the flange, silently
- [ ] Check the summary says `assets: referenced in place, nothing copied` with no
      `(N unresolved)`, and check any `matched by FILENAME` warning names the asset
      you meant
- [ ] Check the summary lists the `agent workspace:` directories it created
- [ ] Re-converting? It overwrites in place — re-apply any hand-tuned `REVIEW`
      fields in `robots.yaml` and any `is_workpiece` you set in `obstacles.yaml`,
      and re-check `agentic_planner/tasks/*.json` against the new scene
- [ ] Review the `REVIEW` fields in the generated `robots.yaml` (tool_collision, workspace_radius, gripper)
- [ ] Set `is_workpiece: true` on the weld-surface box(es) in
      `agentic_planner/config/obstacles.yaml`
- [ ] `make sim SIM=<cell>` + `make health`, then `check_reachability`  (do **not** run `make urdfs`)

Creating a cell — manual:

- [ ] `cells/<cell>/agentic_planner/{config,tasks,plans,logs,recordings}/` — create
      the workspace first; its existence is what selects this layout
- [ ] `cells/<cell>/scene/scene.yaml` (+ `scene.usda`) from Isaac
- [ ] Robot URDF at `urdf/<robot_id>/<model>.urdf`, meshes in `urdf/meshes/`
      (one description per arm — no second copy)
- [ ] `agentic_planner/config/robots.yaml` — base poses (match scene), joint limits, tool geometry
- [ ] `agentic_planner/config/obstacles.yaml` — fixture/floor/walls, `is_workpiece`
      on the work surface
- [ ] `agentic_planner/config/cyclonedds.xml` — copied from the reference cell
- [ ] `make urdfs SIM=<cell>` (bake base origins)
- [ ] `make sim SIM=<cell>` + `make health`, then `check_reachability`

Authoring a task:

- [ ] `cells/<cell>/agentic_planner/tasks/<program>.json` with `operations[]`
- [ ] robot-neutral ids; world-frame `position`; sensible `approach_vector`
- [ ] `make task SIM=<cell> TASK=<program>` → plan saved → `make replay` to re-run
