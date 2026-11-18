# Operation Parameters Reference

Default parameters per operation type for the trajectory planner.
These values are read by the `plan-trajectories` skill when building motion requests.

## Parameters

| Operation type | approach_offset_m | hold_duration_s | thermal_gap |
|---|---|---|---|
| weld           | 0.10 | 0.50 | 3 ops |
| rivet          | 0.08 | 0.30 | 2 ops |
| glue           | 0.05 | 0.00 | 1 op  |
| laser_cut      | 0.15 | 0.10 | 4 ops |
| laser_weld     | 0.12 | 0.20 | 4 ops |
| mark           | 0.05 | 0.05 | 1 op  |
| move           | 0.00 | 0.00 | 0 ops |

## Parameter definitions

- **approach_offset_m**: Distance from target along approach_vector to insert an approach waypoint.
  The robot moves to this waypoint first, then continues to the target.
  Prevents the robot from slamming into the surface.

- **hold_duration_s**: Time the robot holds at the target position after arriving.
  Used for welding arc dwell, rivet insertion, glue dispensing etc.

- **thermal_gap**: Minimum number of operations between same-face consecutive welds.
  Prevents thermal deformation from heat accumulation on one face.

Speed is not a per-operation parameter. Transit runs at the cell's
`planning.max_velocity_scaling` and the working stroke at its
`approach_velocity_fraction` of that (`robots.yaml` / `obstacles.yaml`); a motion
request may only slow its stroke further with `approach_velocity_scale`.

## Overrides

Parameters can be overridden per operation in the task program JSON:
```json
{
  "id": "W001",
  "type": "weld",
  "hold_duration_s": 0.8
}
```

Per-operation overrides take precedence over these defaults.
