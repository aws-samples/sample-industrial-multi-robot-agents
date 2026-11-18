---
name: recover-from-planner-failure
description: >-
  Decide the next move when run_trajectory_planner returns a failure for a robot.
  The motion planner is a point-to-point servo: it reports WHY it could not reach
  a target (reason_code + the object/world-point of a collision) and it is YOUR
  job to change the plan and re-call it. Maps each reason_code to a concrete
  recovery — retry, intermediate waypoint, wait/start_delay, move the other arm
  out, REDISTRIBUTE the operation to another arm with reallocate_operation, larger
  budget — in a cheap-to-expensive ladder, and says when a point is finally
  unreachable (only after every arm has failed it). Use whenever a planner result
  has a non-empty `failures` list, or a pair/approach gate fails.
license: Apache-2.0
metadata:
  author: multi-robots-agents
  version: "1.0"
  category: recovery
---

# recover-from-planner-failure

The planner never decides *what to do next* — it only reports what happened. You
close the loop: read the failure, change ONE thing, re-call the planner, repeat.

## The failure shape

`run_trajectory_planner` returns `{"success", "trajectories", "failures": [...]}`.
Each `failures` entry is for one robot and carries:

- `reason_code` — the machine-readable cause (below).
- `error` — a human sentence (often with a recovery hint).
- `collision` — for collisions: `{object_id, point:[x,y,z], link_index,
  clearance_m}` (negative clearance = penetration depth).
- `recovery` — what can be done, from the platform's one table:
  `{retry_same, levers, escalate}`. `retry_same: false` means re-sending the SAME
  request returns the same failure — change something first. Follow `levers` in
  order; when they are spent, `escalate` says what is left (`reallocate`,
  `resequence`, `fix_cell`, `stop`).
- `orientation_search` — for a weld: what the planner's own search of the tool's
  free spin and tilt did (`specified_ok`, `chosen`, `none_usable`,
  `not_applicable`). `none_usable` means no spin or tilt within the tool's
  tolerance works at this stroke: a hand-picked `tool_spin_deg` cannot either.
- `suggestion` / `suggested_approach_offset_m` — a shorter stroke that DOES have a
  usable pose, when one exists. Re-plan with it first.
- `caused_by` — on `step_refused` (ompl-compound): the arm and code that sank the
  step. THIS arm's weld is fine; fix the one named.

`success` can be `true` while `failures` is non-empty (some arms succeeded,
others didn't). ALWAYS inspect `failures`, not just `success`.

## The recovery ladder (cheapest first)

Try in order; stop as soon as the re-plan succeeds. Cap at ~2 changes per weld
**on that arm**, then hand it to another arm (below), and only mark it unreachable
once every arm has failed it.

## The last rung, before "unreachable": REDISTRIBUTE

`run_mrta_pipeline` allocated this weld from geometry alone — approach-vector
alignment and distance — *before* any IK or collision check ran. So "robot1 cannot
reach W07" is evidence the allocation was wrong for W07, not evidence that the cell
cannot weld it. Changing the distribution is a normal planning move.

    reallocate_operation(cell_id, operation, tried_robots=[...], robots=[...])

- `operation` — the failed operation dict (`id`, `position`, `approach_vector`).
- `tried_robots` — EVERY arm that has already failed it, including the current
  owner. The tool will not propose one of these again.
- `robots` — your current per-robot execution order, the same
  `[{"robot_ns", "operation_ids"}]` you pass to `chain_trajectories`.

It ranks the remaining arms with the SAME policy that made the original allocation,
pre-filters any arm whose `workspace_radius` cannot reach the point, and names
`to_robot` plus the `insert_after_operation_id` the handover chains from.

Then: re-plan that operation on `to_robot` as its **LAST** operation (omit
`current_joints` and it chains automatically), add the id to the END of that arm's
`operation_ids`, and put it in a SEQUENTIAL step under the new arm. Remove it from
the old arm's list — leaving it in both is how `chain_trajectories` reports a break
you cannot explain. The handover appends rather than inserts precisely so nothing
you have already planned and validated changes its start pose.

It REFUSES in three cases, each of which is a real answer, not an obstacle:
- `robot_locked` — the operation carries `assigned_robot`. Recover by changing the
  ENTRY on that arm, or record it unreachable. Never reassign a locked operation.
- `out_of_reach_everywhere` — outside every remaining arm's workspace radius.
- `no_candidate_robot` — every arm has now failed it. **This** is unreachable:
  record it with the per-robot reasons and continue.

### `goal_self_collision` — the arm folds into ITSELF at the weld pose

No obstacle is involved (the cell may have no fixtures): `collision.object_id`
names the two links of the SAME arm that meet, e.g. `link_3 vs link_6` — the tool
plate driven into the forearm. No path fixes it; the POSE has to change — and the
planner has ALREADY searched every spin of the tool about its axis and every tilt
the tool allows (`robots.yaml tool_orientation`), with its own IK and checks
(`orientation_search`). Do NOT re-send the weld with hand-picked `tool_spin_deg`
values: a spin you set turns that search off and tries one pose instead of all of
them.
1. If the failure carries `suggested_approach_offset_m`, re-plan with that
   `approach_offset_m`: a shorter stroke keeps the wrist out of its own forearm at
   the top of the stroke, and the suggestion is verified.
2. Otherwise reallocate the operation to another arm (`reallocate_operation`) —
   another base reaches the same point with a different configuration.
`tool_spin_deg` is for a different problem: turning the tool to keep it clear of
the OTHER arm (which the search does not consider). Never edit the quaternion.
`start_self_collision` means the arm is already in that state: re-plan from HOME.

### `approach_self_collision` / `approach_unreachable` — the stroke INTO the weld

The weld pose itself is fine, but the straight descent along the approach vector
is not: the arm folds into itself on the way in (`collision` names the two links
and the step), or no IK solution holds the lean at some step. Same ladder as
above: `suggested_approach_offset_m` (fewer steps of stroke) first, then another
arm. Measured on welding_cell_2_small: robot1's W07 has a usable pose only with a
3 cm stroke.

### `assignment_violation` — the operator locked this operation to another arm

The operation (or a leg derived from it, e.g. `W05_VIA`) was ASSIGNED to a robot by
the planning request. Plan it on that robot — re-sequence, add a via point, spin
the tool, or move the other arm out of the way. Never reassign it;
`reallocate_operation` refuses it too.

### `step_gate` on a planner result — clash found while the step is fresh

Every planned step carries `step_gate`: the assembler's inter-arm check against
the step's other arms AND the arms parked where they last stopped.
- `PARALLEL` — clear; no separate `validate_trajectory_set` call needed.
- `SEQUENTIAL` — clear only one at a time (`order`); the assembler will serialise
  it. To keep it parallel, pair a different operation or add a start delay.
- `REFUSED` (`step_gate_refused` in a batch) — no order is clear; the assembler
  would refuse the tree. Fix it NOW: pair a different operation, or first plan a
  `move` that retreats the parked arm (`blocked_by_parked`) clear, then re-plan.

### `not_planned` — a batch stopped before this step

When a step of a `steps=[...]` call fails or is refused (`stopped_at`), the later
steps that move one of ITS arms are not attempted (`skipped_steps`); later steps
that move only other arms ARE planned and kept (`kept_steps_after_stop`). Fix the
failed step, then send exactly the failed and `skipped_steps` again, in order —
never the kept ones (re-planning them would start from the wrong pose).

### `self_collision_unavailable` — infrastructure, not geometry

The robot's self-collision check is required but cannot run (e.g. the arm's
collision meshes are missing). No parameter fixes it and the robot cannot be
planned: report it with the `error` text and stop planning that arm. Never work
around it.

### `liftoff_blocked` — the arm cannot leave its previous seam on-axis

The leg starts ON a weld (the previous one, or the one a pinned `current_joints`
stands on — `from_operation_id`), and the straight lift-off along that weld's axis
is blocked: usually by the other arm's motion in the same call, else an obstacle or
IK. Without it the leg would drag the tool sideways out of the seam, so it is
refused. With `ompl-compound` the whole step is refused (one motion).
1. A shorter `retreat_offset_m` on THIS request: the lift-off then stops lower,
   under whatever blocks the top of the stroke. In coordinated mode (one arm per
   call) this is the main lever.
2. Plan this operation in its own call / step, after the other arm has moved.
3. Give the other arm a different operation in this step, or park it first (a
   `move`), then plan this leg.

### `home_return_refused` (assembler `home_step.home_refused`) — the planned return failed a check

The arm's planned return home was refused by the platform (`check` names why:
`trajectory_self_collision`, `joint_limit_violation`, or a fixture breach), and the
blind straight line that would replace it is the same unchecked motion, so the tree
is refused. Add a reposition `move` as the arm's last operation (from a pose the
return can leave), or set `"home_at_end": false` and plan the return as an ordinary
operation so it gets the normal recovery.

### `home_liftoff_blocked` (assembler `home_step.liftoff_blocked`) — no way out of the last seam

The arm's planned return home would start with the on-axis lift-off out of its last
weld, and that lift-off is blocked at every stroke down to the cell's
`planning.min_home_liftoff_m`. A blind home would drag the tool out of the seam, so
the tree is refused. End that arm's sequence with a `move` away from the seam (it
starts with the lift-off), or re-order so its last weld is one it can leave.

### `joint_limit_violation` / `start_outside_joint_limits` / `goal_outside_joint_limits`

A leg (or its start / goal) leaves the arm's joint limits (robots.yaml ∩ the URDF);
`joint_limit_violation` names the joint, the sample and by how much. The arm cannot
execute it, and no delay or order fixes it: re-plan the leg (the planner may take
another IK branch), add an intermediate waypoint, or use another arm.

### `trajectory_self_collision` (assembler `leg_refusals`) — the arm folds into itself mid-leg

Every sample is clear, but the straight motion BETWEEN two of them drives the arm
into its own collision hulls (`collision.object_id` names the pair,
`segment_fraction` where). Re-plan the operation (the planner samples a new path),
add a via pose, or use another arm. For a `home` leg, add a reposition as the arm's
last operation.

### `approach_mismatch` (assembler `leg_refusals`) — the working stroke is not the specified one

The assembler checks every weld leg with `validate_approach` against its
operation's own orientation, point and approach vector (`approach_check` gives the
numbers): the tool axis off the tolerance, the point missed, or no measurable
on-axis entry. Re-plan that weld (a shorter `approach_offset_m` if the entry was
the problem); never relax the orientation.

### `trajectory_in_collision` / `excessive_tool_rise` — what was emitted fails

The planner's own emitted samples breach an obstacle (`collision`) or arc the tool
over the cell (`tool_rise_m`). Retry once (the planner samples a new route), then
add a via point AROUND (not over) the obstruction, or a staging move before the
weld.

### `step_refused` — ompl-compound refused the whole step because of ANOTHER arm

`caused_by` names the arm and its code. This arm's weld is fine: fix the named
arm's failure (or re-pair the step), and re-plan the step.

### `compound_start_invalid` / `compound_goal_invalid` — the arms conflict where they stand / where they end

Two arms of the step are already too close at the start, or their two goal poses
are within the warning band of each other. No path fixes it: move one arm first
(start), or pair a different pending weld (goal).

A conflict between two arms in ONE step is about those two poses, not about the
cell: do not conclude that two arms "always conflict" and plan without one of
them. Before treating anything as a property of the layout, check it — e.g.
`validate_trajectory_set` with each arm at HOME (one-sample trajectories). On
welding_cell_2_small a model concluded "same-side arms conflict even at rest" from goal
poses an allocator bug had sent across the body; at HOME those arms are 2.76 m
apart. The fixes, in order: pair a different pending weld, split the step into
`steps=[[a], [b]]` (one after the other), stagger it — the number of arms in the
plan stays the same.

### `duplicate_robot_request` — two requests for one arm in one call

Each arm moves once per call. Pass a sequence as `steps=[[...], [...]]`.

### `self_collision_check_error` — the arm-vs-itself check ran and broke

Retry once; if it persists it is infrastructure — report it and stop planning
that arm (do not work around it).

### `leg_check_error` / `inter_arm_check_error` — a check ran and broke

A leg check (joint limits, static obstacles) or the inter-arm gate raised
instead of answering. The leg or step is refused, not waved through: a check
that broke has not said yes. Retry once; if it persists it is infrastructure —
report it and stop (do not re-pair or reorder to get around it).

### `backend_contract_violation` — the backend returned a malformed leg

The selected backend broke the result contract (a leg that does not start
where the arm stands, time running backwards, non-finite joints). It is a
backend bug, not a planning problem: report it; re-plan with the default
backend only if the task allows choosing one.

### `blocked_by_parked_arm` — another arm stands in the way of the stroke

The weld is reachable; the space above it is occupied by the arm(s) in
`blocked_by`, standing where their last leg left them. Do NOT reallocate the
weld: call `find_clear_pose(cell_id, <blocking arm>, clear_of=[{robot_ns:
<this arm>, operation_ids: [<this weld>, ...its next welds]}])` and plan its
`motion_request` as ONE move of the blocking arm — or give that arm its next
operation in the same step — then plan this weld again.

### `reallocation_not_needed` — the handover was refused

`reallocate_operation` was asked to move a weld whose failure was another arm
in the way. The weld stays on its arm: clear the blocking arm with
`find_clear_pose`, then plan the weld again.

### `chain_gap` / `missing_trajectory` — the program does not chain

From `chain_trajectories` / the assembler: an operation does not start where the
arm's previous one ended (`chain_gap`), or was never planned
(`missing_trajectory`). Re-plan in execution order, as each break's `fix` says.

### `start_gap` (from `chain_trajectories` / the assembler) — the first leg does not start at HOME

An arm's FIRST operation does not start where the arm stands when the program
runs (HOME). The planner already pins a re-plan of the last operation to where its
first plan started, so this now means a leg was planned from a pose the program
never reaches (a pinned `current_joints`, or an edit). Re-plan it with
`current_joints` set to HOME (`end_joints` in the break), then the arm's later
operations in order.

### `goal_in_collision` — the target pose itself collides

**FIRST check for a `target_diagnosis` field on the failure.** The planner adds
it when the WELD POINT itself is inside or too close to a fixture box:
`{target_status: "inside_box"|"too_close", object_id, target_clearance_m,
suggested_offset_m, suggested_position, suggestion}`.
- If `target_status == "inside_box"` (or `too_close` and re-planning already
  failed): the point is UNREACHABLE in the specified approach — no parameter fix
  exists. Do NOT loop, and do NOT reallocate: this is a property of the POINT
  (it is inside a fixture box), not of the arm, so no other arm can reach it
  either. Record the weld as unreachable in your final summary and quote the
  `suggestion` verbatim (move the weld out to `suggested_position`, or verify the
  box geometry). Then move on to the other welds.
- Only if there is NO `target_diagnosis` (the point itself is clear, so it is an
  arm-LINK graze) do you try the parameter recovery below.

The tool at the weld (usually leaned) grazes the named `object_id`.
1. The approach does NOT get a smaller collision radius (the old reduced approach
   radius is retired): the clearance at the weld is the body's radius plus the
   object's margin, plus a speed allowance only when the cell declares
   `speed_margin_full_speed_m`. Do not re-plan just to set `approach: true`.
2. `load_skill('approach-strategy')` and change the ENTRY, not the pose: raise
   `approach_offset_m` and come down a clear vertical corridor, or approach a
   box-enclosed point from the TOP. Re-plan.
3. If the only object in `collision` is the weld-surface panel the tool must
   actually touch (the real workpiece, NOT a fixture rail), that box should be
   `is_workpiece: true`; confirm and re-plan.
4. Still colliding after ~2 tries → this ARM cannot hold the leaned pose here
   without grazing `object_id`. That is arm-specific: a different base position
   reaches the same point from a different direction. REDISTRIBUTE — call
   `reallocate_operation` and re-plan on the arm it names. NEVER silently drop the
   required `weld_orientation` to make it fit.
5. Only when `reallocate_operation` reports `no_candidate_robot` (every arm has
   failed it) is the weld unreachable-with-lean. Record it with each arm's reason.

### `start_in_collision` — the arm's starting pose is invalid
The chained start (usually the previous weld's end pose) is in collision.
1. This almost always means the PREVIOUS operation left the arm in the fixture —
   fix that operation first (its own `goal_in_collision`), which fixes this.
2. Otherwise insert a REPOSITION step before this weld: give the robot an extra
   operation_id whose target is a clear pose above/outside the obstacle, so it
   moves out first, then plan the weld from there.

### `retry_cap_reached` — this arm has failed this operation too often

The planner refused the request without solving: in this run the arm has already
failed this operation (or this helper waypoint — `W18_VIA`, `W18_VIA2`, … count as
one) 3 times. Re-sending it, renamed or not, gets the same refusal. A weld: hand it
to another arm with `reallocate_operation` (`tried_robots` including this arm). A
helper waypoint: stop trying it; plan the weld without it, or reallocate the weld.

### `no_path_found` — endpoints valid, no collision-free path in budget
1. Add an intermediate VIA-POINT: insert a reposition step (own operation_id) to
   a clear waypoint (e.g. high above the target, or toward the robot's base),
   then plan the weld from there. Two short hops usually succeed where one long
   one fails.
2. If this is a PARALLEL step and the blocker is the OTHER arm's swept volume,
   prefer TEMPORAL separation over spatial: give this arm a `start_delay_s` so
   the other clears first, or make the step SEQUENTIAL. See below.
3. Retry once — RRTConnect/OMPL are randomised, a fresh attempt can find a path.
4. Still no path → REDISTRIBUTE with `reallocate_operation`. A corridor that is
   blocked from one base is often open from another.

### `inter_arm_clearance` / INTER-ROBOT collision (`validate_trajectory_set` not collision-free)
Two or more arms of a PARALLEL step conflict in space+time. `colliding_pairs` and
`worst_pair` name which.
1. `load_skill('find-interlocks')` → `find_interlock_set` with ALL the converging
   arms returns interlock poses / a `start_delay` / an execution plan. Apply the
   smallest fix that clears EVERY reported pair:
   - add `start_delay_s` to the later arm (temporal separation), else
   - add the interlock waypoint(s) it returns, else
   - pair a different pending weld, and only as the LAST resort make that step
     SEQUENTIAL (the assembler then lists it in `serial_steps`, which you must
     explain).

### `ik_failed` — IK failure (`error` contains "IK failed")
The target position + orientation is unreachable for that robot at the required
lean. You have NO IK tool — do not try to hand-solve it.
1. REDISTRIBUTE straight away — there is no parameter on this arm to change.
   `reallocate_operation` names the next-best arm and skips any whose
   `workspace_radius` cannot reach the point.
2. When it reports `no_candidate_robot` / `out_of_reach_everywhere` / `robot_locked`,
   the leaned pose is out of reach — record it unreachable and continue.

### `planner_error` / `tool_error` — internal/rare
Retry once. If `planner_error` persists, treat as `no_path_found` (add a
via-point); if `tool_error` persists, report it and stop using that tool.

## Rules
- Change ONE thing per re-plan so you know what worked.
- The `weld_orientation` and the approach switch are HARD requirements of the
  final stroke — recover by changing the PATH (waypoints, timing, which arm),
  never by relaxing the target pose.
- Keep the step PARALLEL if you can clear it with timing; only drop to SEQUENTIAL
  when spatial separation is impossible — parallelism is what keeps cycle time low.
- A weld is UNREACHABLE only when every arm has actually failed it. "robot1 cannot
  do W07" is a reallocation, not a verdict on the cell.
- Hard cap: ~2 recovery attempts per weld PER ARM, then reallocate; when no arm is
  left, mark it unreachable in your final summary and KEEP GOING. One corner weld
  must never block the whole tree.
