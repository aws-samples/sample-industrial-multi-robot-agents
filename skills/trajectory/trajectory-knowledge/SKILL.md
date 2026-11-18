---
name: trajectory-knowledge
description: >-
  Query the trajectory knowledge base for warm-start joint configurations
  and similar past trajectories. Use at the start of each step to seed
  Pinocchio IK, reducing convergence iterations from ~150 to ~20. Also
  stores successful trajectories after each completed batch for future runs.
license: Apache-2.0
metadata:
  priority: 40
  author: multi-robots-agents
  version: "1.0"
  category: knowledge
---

# trajectory-knowledge

Provides warm-start joint configurations from previous successful trajectories
to accelerate Pinocchio IK convergence and reduce planning time.

## When to use

### Querying (before planning)

At the start of `plan-trajectories`, call `get_warm_start` for each robot
target. If a match is found within 0.05m, use those joints as the IK seed
instead of the default home config.

### Storing (after planning)

After each step's trajectories are executed successfully (confirmed by the
cell agent), call `store_trajectory` with the final joint positions to build
the knowledge base for future runs.

## Steps — Querying warm starts

### 1. Query the knowledge base

```
get_warm_start(
    cell_id="welding_cell_2_small",
    operation_type="weld",
    robot_ns="robot1",
    target=[x, y, z]
)
```

**Match types:**
- `exact`: Same target rounded to 2 decimal places — use directly
- `nearby`: Within 0.05m — use as IK seed (not guaranteed to converge)
- `found: false` — use default home config `[0, -1.57, 2.2, 0, -0.8, 0]`

### 2. Query similar trajectories for context

```
get_similar_trajectories(
    cell_id="welding_cell_2_small",
    operation_type="weld",
    robot_ns="robot1",
    target=[x, y, z],
    radius_m=0.2
)
```

Use the returned joint configs to understand the typical robot posture near
this workspace region. If the IK fails with the default seed, try joints from
the nearest similar trajectory.

## Steps — Storing successful trajectories

```
store_trajectory(
    cell_id="welding_cell_2_small",
    operation_type="weld",
    robot_ns="robot1",
    target=[x, y, z],
    final_joints=[j1, j2, j3, j4, j5, j6],
    position_error_m=0.003
)
```

Call this after receiving confirmation from the cell agent that a batch step
completed successfully. Do NOT store failed or aborted trajectories.

## Knowledge base storage

The KB is stored at:
```
cells/<cell_id>/recordings/trajectory_index.json
```

Keys are: `{operation_type}|{robot_ns}|{x:.2f}|{y:.2f}|{z:.2f}`

The KB persists across sessions. The more operations are executed, the faster
future planning runs become.

## Performance impact

| IK mode | Avg iterations | Success rate |
|---|---|---|
| Cold start (home seed) | ~120-150 | ~85% |
| Warm start (nearby KB) | ~15-25 | ~97% |
| Exact KB match | ~5-10 | ~99% |
