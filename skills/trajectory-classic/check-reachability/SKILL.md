---
name: check-reachability
description: >-
  Judge whether a Cartesian target is reachable by a robot WITHOUT a direct IK
  tool — by interpreting run_trajectory_planner's IK-failure / no_path_found
  result together with the robot's workspace radius and base position. Use when
  optimize-sequence flags a point unreachable, or when the planner returns an IK
  failure, to decide: try the other robot, or record the point unreachable.
license: Apache-2.0
metadata:
  author: multi-robots-agents
  version: "2.0"
  category: kinematics
---

# check-reachability

You have NO direct IK tool. Reachability is inferred from what
`run_trajectory_planner` reports plus simple geometry — you do not hand-solve IK.

## When to use

1. `optimize-sequence` returns points in `unreachable[]`.
2. `run_trajectory_planner` returns a `failures` entry whose `error` contains
   "IK failed" (the pose+orientation did not converge) or `no_path_found`.

## How to judge

1. **Geometry pre-check.** Compare the target's distance from the robot's base
   (`base_position` in `robots.yaml`) against the robot's `workspace_radius`.
   If `distance_from_base > workspace_radius`, the point is out of reach for that
   robot — go straight to the fallback.

   | Robot model | Workspace radius |
   |---|---|
   | KR240 R2900-2 | 2.90 m |
   | KR70 R2100-2  | 2.10 m |

2. **Within reach but IK failed** usually means the required tool orientation
   (the weld lean) is not achievable at that spot with that arm — the position is
   fine, the leaned pose is not.

## Fallback (recovery)

- **Re-assign the operation and re-plan it.** Call `reallocate_operation(cell_id,
  operation, tried_robots=[the arms that failed it], robots=[your current
  execution order])`. It applies the geometry check above for you — from each
  arm's `base_position` and `workspace_radius` in `robots.yaml`, so you do not
  need the table above to rule an arm out — ranks the remaining arms with the same
  policy that allocated the work originally, and names the arm to hand it to plus
  the operation the handover chains from. See
  `load_skill('recover-from-planner-failure')` for what to do with the result.
- An operation carrying `assigned_robot` is locked and is refused
  (`robot_locked`); a point outside every arm's radius is refused
  (`out_of_reach_everywhere`).
- Only when it reports `no_candidate_robot` — every arm has actually failed it —
  is the point unreachable. Record it in your final summary and continue. Do not
  loop, and do not try to hand-solve IK.
