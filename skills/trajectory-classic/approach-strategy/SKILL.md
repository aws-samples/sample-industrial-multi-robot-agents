---
name: approach-strategy
description: >-
  Decide HOW a robot should approach an operation target when a straight approach
  collides with a box obstacle, WITHOUT changing the target's required tool
  orientation. You do NOT probe IK/collision by hand — you read the obstacle
  geometry and the planner's structured goal_in_collision failure, then choose
  the run_trajectory_planner parameters (approach flag, standoff distance, and
  whether an intermediate via-point is needed) that get the tool in. Use when
  run_trajectory_planner reports a target colliding with a box, or before
  planning a target that sits near/inside boxes.
license: Apache-2.0
metadata:
  author: multi-robots-agents
  version: "2.0"
  category: collision-avoidance
---

# approach-strategy

This is a REASONING skill: you choose planner parameters, you do not compute
poses. There are no IK or collision tools here — motion and its collision checks
happen inside `run_trajectory_planner`, which reports a structured failure you
interpret.

## Invariant: never change the tool's LEAN

The operation's `weld_orientation` (else `orientation`) fixes the tool AXIS at the
weld — the lean `validate_approach` checks (5°). That is a HARD requirement: this
skill only changes the PATH/parameters to reach it, and NEVER drops or relaxes the
lean to avoid a collision. Always keep `orientation` + `constrain_orientation:
true` on the request.

The spin ABOUT that axis is free (a spot-weld electrode is round): set
`tool_spin_deg` on the request to turn the wrist about the electrode without
changing the lean — the move for an arm that folds into itself
(`goal_self_collision`, `approach_self_collision`) or that sits in the other arm's
way. Never edit the orientation quaternion by hand for this.

## What the planner tells you

A colliding target comes back as a `failures` entry:
`{"reason_code": "goal_in_collision", "collision": {object_id, point:[x,y,z],
link_index, clearance_m}}` (negative `clearance_m` = penetration depth). Use the
`object_id` + `point` together with the cell's `obstacles.yaml` geometry to
classify the situation — do NOT guess.

Each box in `obstacles.yaml` has `center [cx,cy,cz]` and `half_extents
[hx,hy,hz]`: its top is `z_top = cz+hz`, bottom `z_bot = cz-hz`, and it covers
`(tx,ty)` when `|tx-cx|<=hx and |ty-cy|<=hy`.

## Classify, then choose parameters

### CLEAR — nothing within the clearance envelope
The planner already succeeded (no `goal_in_collision`). Use a **direct approach**:
`approach: true`, a moderate `approach_offset_m` (~0.12–0.15 m).

### TIGHT GAP — free point, but a nearby box grazes at transit clearance
The colliding `object_id` is a box BESIDE the point (its `z_bot` is at/below the
weld but no box sits directly above `(tx,ty)`), and the reported `clearance_m` is
only slightly short (roughly within 0.07 m of the point). This is the collision
inflation case: the arm is fine, the coarse transit radius just grazes the rail.
Fix with parameters, no detour (the approach gets NO smaller radius — the old
reduced approach clearance is retired — so the lever is geometry):
1. SHRINK `approach_offset_m` (~0.06–0.08 m) so only the thin tool tip enters the
   gap, keeping the fatter arm links out of it. Re-plan.
2. If it still grazes, raise the entry above the rail (see CORRIDOR below).
This resolves most `goal_in_collision` reports on the fixture rails.

### CORRIDOR — free point, but the straight lateral entry sweeps through a box
No box is directly above `(tx,ty)`, but the point sits deep and a side entry
would drag the arm across a rail even at approach clearance. Bring the tool DOWN
a clear vertical column instead of sideways:
1. Insert a REPOSITION via-point step BEFORE the weld (its own `operation_id`):
   target `[tx, ty, z_entry]` with a clear `z_entry` well above the boxes, a
   neutral top-down `orientation`, `approach: false`. Assemble it as a SEQUENTIAL
   step before the weld.
2. Then plan the weld from that entry with `approach: true` and the required
   `weld_orientation`; the descent is now a clean vertical column.

### ENCLOSED — a box sits directly ABOVE the point
Some box covers `(tx,ty)` with `z_bot > tz` (solid material above the weld). With
the task's `approach_vector` the tool would have to pass through it — there is no
parameter that fixes this. After confirming from the geometry, record the weld
**unreachable-with-this-approach** in your summary (do not loop). If the colliding
box is the weld-surface panel the tool must actually touch (the real workpiece,
NOT a fixture rail), that box should be `is_workpiece: true` — flag it.

## Rules
- Change ONE parameter per re-plan (`approach` flag, then `approach_offset_m`,
  then a via-point) so you know what worked.
- Cap at ~2 changes per weld, then mark it unreachable and move on — never let a
  corner weld block the rest of the tree.
- After a successful re-plan, still `validate_approach` on the planned trajectory
  (final lean + on-axis entry). The approach only changed the PATH; the final
  working pose must still match the spec.
