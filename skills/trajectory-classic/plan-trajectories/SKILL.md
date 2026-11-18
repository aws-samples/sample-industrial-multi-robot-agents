---
name: plan-trajectories
description: >-
  Plan collision-free joint-space trajectories for one or more robots moving to
  operation targets, one sequence step at a time. The motion planner
  (run_trajectory_planner) is a point-to-point servo that does IK + collision +
  the approach split internally; this skill is the PLAN-FIRST loop around it:
  build the step request, plan, verify the planned result, recover on failure,
  assemble. Use after optimize-sequence has produced an ordered step table.
  Handles weld, rivet, glue, laser, mark, and move operations.
license: Apache-2.0
metadata:
  author: multi-robots-agents
  version: "2.0"
  category: motion-planning
  backend: ompl | rrt
---

# plan-trajectories

Plans collision-free trajectories per sequence step and feeds them to the
behaviour-tree builder.

## The one rule that matters

You have NO raw IK / FK / per-config collision tools. The ONLY way to produce
motion is `run_trajectory_planner`, which runs IK + endpoint-collision + the
approach split INTERNALLY. So never try to "find a good joint pose first" — you
cannot, and you do not need to. The flow is: **build the request → PLAN → verify
the planned RESULT → recover on failure** per step, then once all steps are
planned **→ chain → assemble.** Do it in that order.

## When to use

Once per step returned by `optimize-sequence`. A step has 1-2 robot targets.
Plan BOTH arms of a step in ONE `run_trajectory_planner` call so the planner
couples them (each arm avoids the swept volume of the arm already planned in that
call → the pair is collision-free by construction, and executes in parallel).

## Steps (in order)

### 1. Build the step's motion request(s)

One `run_trajectory_planner` call for the step; put both arms in the same call.
Each request:

```json
{
  "robot_ns": "robot1",
  "operation_id": "R1_W07",
  "target": {"x": ..., "y": ..., "z": ...},
  "approach_vector": [ax, ay, az],
  "operation_type": "weld",
  "orientation": {"x": ..., "y": ..., "z": ..., "w": ...},
  "constrain_orientation": true,
  "approach": true,
  "approach_offset_m": 0.10,
  "velocity_scale": 0.3
}
```

One call covers the WHOLE approach: the planner moves transit (normal clearance)
→ an approach waypoint `approach_offset_m` back along `approach_vector` → the
terminal descent into the point, straight along the approach vector, at the
approach speed and the required lean (same collision clearance as transit).

- **`operation_id`** must be UNIQUE — the trajectory is cached under it for
  `assemble_behaviour_tree`.
- **`orientation` + `constrain_orientation: true`** whenever the operation
  specifies a `weld_orientation` (preferred) or `orientation`. Copy that
  quaternion in; the planner solves full-pose IK so the tool ends at exactly that
  lean. A specified orientation is a HARD requirement of the final pose — never
  drop or relax it to dodge a collision; change the PATH instead (see
  `approach-strategy`). Omit it only for a plain `move` with no tool pose.
- **`approach: true`** for a working stroke → the final descent is planned as the
  working stroke: straight along the approach vector, at the lean, at the slower
  approach speed (and the speed allowance that earns, when the cell declares
  one). The collision radius is the SAME as transit. Leave it false/unset for a
  pure `move`.
- **`approach_offset_m`** = the distance where normal motion switches to the
  approach — YOU choose it per weld: larger (~0.12–0.15 m) at an open point,
  smaller (~0.06–0.08 m) in a tight pocket (see `approach-strategy`).
- **`velocity_scale`** ~0.2–0.3 to slow the working stroke.
- **`tool_spin_deg`** (optional) turns the tool about its own axis without changing
  the lean — for an arm that folds into itself or sits in the other arm's way.
- **`current_joints`** is optional — omit it and the arm chains automatically
  from where its previous leg ENDED (weld-to-weld), falling back to home
  `[0, -1.57, 2.2, 0, -0.8, 0]` for its first move.

### 2. Run the planner

Call `run_trajectory_planner(cell_id, motion_requests=[...])`. It returns
`{"success", "trajectories": {ns: [...]}, "failures": [...]}` and caches each
trajectory by `operation_id`. The backend (`ompl`/`rrt`) is bound by the
task — you do not pass or choose it, and there is no fallback. ALWAYS inspect
`failures`, even when `success` is true (some arms can fail while others succeed).

### 3. Verify the planned RESULT (not a raw pose)

- **Terminal stroke** — for every weld with a `weld_orientation`, call
  `validate_approach(cell_id, robot_ns, expected_orientation=<weld_orientation>,
  trajectory=<the PLANNED trajectory>, target_position=<point>,
  approach_vector=<approach_vector>)`. Pass the planned trajectory, NOT a joint
  config. `matches=false` → re-plan that weld (tighten the constraint / adjust
  `approach_offset_m` / change the entry via `approach-strategy`). Never assemble
  a weld whose `validate_approach` fails.
- **Parallel step** — call `validate_trajectory_set` on ALL arms of a
  PARALLEL step. If not collision-free, `load_skill('find-interlocks')`.

### 4. Recover on failure

If a `failures` entry carries a `reason_code` (`goal_in_collision`,
`start_in_collision`, `no_path_found`, `planner_error`, or an IK failure),
`load_skill('recover-from-planner-failure')` and apply the recovery for that
code, then re-call `run_trajectory_planner`. That skill is the single source of
truth for the recovery ladder (retry with `approach:true` / intermediate
via-point / wait + `start_delay` / move the other arm out / try the other robot /
larger budget), the "change ONE thing per re-plan" rule, and the ~2-attempts-
then-mark-unreachable cap. Do NOT hand-probe and do NOT switch backend.

### 5. Chain the pieces

Before assembling, call `chain_trajectories(cell_id, robots)` with each robot's
`operation_id`s IN EXECUTION ORDER:

```json
{ "robots": [
    {"robot_ns": "robot1", "operation_ids": ["R1_W02", "R1_W03", "R1_W06"]},
    {"robot_ns": "robot2", "operation_ids": ["R2_W15", "R2_W13"]}
] }
```

It does the chaining bookkeeping FOR you — it checks that each operation's
trajectory starts exactly where that arm's previous one ended (the planner
already chains from where each arm's previous leg ENDED, so this holds when you
plan IN ORDER — several steps per call with `steps=[...]` is fine). It never edits
motion. If it returns `chained=false`,
each `break` names a `to_operation_id` that jumps away from its
`from_operation_id`; re-plan that ONE operation with `current_joints` set to the
reported `end_joints`, then chain again. A `missing_trajectory` break means you
still owe a `run_trajectory_planner` call for that `operation_id`.

### 6. Assemble

When the chain is clean (or a hard weld was recorded unreachable), call
`assemble_behaviour_tree(cell_id, plan)` ONCE. Pass a compact plan of `steps`,
each `mode` PARALLEL|SEQUENTIAL with a `robots` list of `operation_id`s. Keep a
step PARALLEL with both arms whenever their pair validated collision-free — that
is what keeps cycle time low. `assemble_behaviour_tree` re-runs the chain check
and REFUSES a tree with a gap, returning the same `breaks` — so fix them in
step 5.

## Key parameters

- `velocity_scale`: from `operation-params.md` per operation type (slower for the
  working stroke).
- `approach_offset_m`: your per-weld choice (open point → larger; tight → smaller).
- `hold_duration_s`: from `operation-params.md`.

## Reference

See `references/joint-limits.md` and `references/operation-params.md`.
