---
name: validate-collision
description: >-
  Validate planned trajectory pairs for inter-robot collision. Use after
  plan-trajectories produces trajectories for two robots in the same step.
  Samples both trajectories at 20 time steps and checks sphere-sphere
  clearance. Returns collision time and the suggestion to call find-interlocks
  if clearance is insufficient.
license: Apache-2.0
metadata:
  author: multi-robots-agents
  version: "1.0"
  category: safety
---

# validate-collision

Checks whether two robots' planned trajectories collide at any point in time,
using the sphere model (0.12m radius per link).

## When to use

Call this skill immediately after `plan-trajectories` produces trajectories for
ALL robots in a parallel step. Do NOT skip this step — even trajectories that
look far apart in Cartesian space can collide in joint space at transition points.

## Steps

### 1. Call `validate_trajectory_set`

Pass EVERY arm in the step — two, three or more. The tool checks all robot pairs,
so a step with 3 arms is validated in one call (3 pairs), not two calls.

```
validate_trajectory_set(
    cell_id=...,
    trajectories={"robot1": "W01", "robot2": "W13", "robot3": "W20"}
)
```

Refer to each leg by the `operation_id` you planned it with — the planner caches
every leg, and its results show you summaries, not waypoint arrays, so there are
none to copy. `validate_trajectory_pair(cell_id, robot1_ns, robot2_ns=...,
operation_id1=..., operation_id2=...)` is the two-robot shorthand and delegates
here. A step whose planner result already says `step_gate.verdict: "PARALLEL"`
was checked this way when it was planned (against the parked arms too). Never validate a 3-arm step by
checking only two of the arms — the unchecked pair is exactly where a collision
hides.

The tool samples every trajectory at 20 instants on a shared wall-clock timeline
(so arms with different durations are compared at the same real time) and checks
sphere-sphere clearance. Minimum clearance values by operation:
- weld, rivet, glue, mark, move: **0.24m** (2 × sphere radius)
- laser_cut, laser_weld: **0.30m** (larger tool safety margin)

### 2. Interpret the result

**Case A — collision_free: true, min_clearance_m ≥ 0.24:**
Proceed to `build-behaviour-tree`. Emit a `Parallel` node with one
`ExecuteTrajectory` child per arm in the step.

**Case B — collision_free: false OR min_clearance_m < 0.24:**
Activate `find-interlocks` with the final targets of ALL arms in the step (see
`colliding_pairs` and `worst_pair` in the result to know which converge).
The interlock planner will find safe intermediate configurations.

**Case C — `static_collisions` is present:**
A leg breaches the floor, a fixture or the arm itself (the planner's
emitted-trajectory check, also run here). `collision_free` is false, and no start
delay or interlock fixes it: the leg itself is wrong. Re-plan that robot's leg
(`run_trajectory_planner` for the listed `robot_ns` and operation) and validate
again.
`static_obstacles_checked: false` means the static check could not run
(`static_check_error` says why) — report it, do not treat the step as checked.

Clearances also include each sphere's **stopping distance** at its speed when
the robot declares `stopping` data in robots.yaml (`min_geometric_clearance_m`
is the clearance without it): a faster leg needs more room. A step refused only
because of speed (geometric clearance above the band) can pass when the leg is
re-planned slower (lower `velocity_scale`).

### 3. Clearance margins

| Operation type | Min clearance | Sphere model |
|---|---|---|
| weld | 0.24m | 2 × 0.12m |
| rivet | 0.24m | 2 × 0.12m |
| glue | 0.24m | 2 × 0.12m |
| laser_cut | 0.30m | 2 × 0.12m + safety |
| laser_weld | 0.30m | 2 × 0.12m + safety |
| mark | 0.24m | 2 × 0.12m |
| move | 0.24m | 2 × 0.12m |

## Reference

See `scripts/trajectory_validator.py` for the validation implementation.
