# Operation Parameters Reference

Default parameters per operation type for the trajectory planner.
These values are read by the `plan-trajectories` skill when building motion requests.

## Parameters

| Operation type | approach_offset_m | hold_duration_s | velocity_scale | thermal_gap |
|---|---|---|---|---|
| weld           | 0.10 | 0.50 | 0.50 | 3 ops |
| rivet          | 0.08 | 0.30 | 0.60 | 2 ops |
| glue           | 0.05 | 0.00 | 0.40 | 1 op  |
| laser_cut      | 0.15 | 0.10 | 0.70 | 4 ops |
| laser_weld     | 0.12 | 0.20 | 0.60 | 4 ops |
| mark           | 0.05 | 0.05 | 0.80 | 1 op  |
| move           | 0.00 | 0.00 | 1.00 | 0 ops |

## Parameter definitions

- **approach_offset_m**: Distance from target along approach_vector to insert an approach waypoint.
  The robot moves to this waypoint first, then continues to the target.
  Prevents the robot from slamming into the surface.

- **hold_duration_s**: Time the robot holds at the target position after arriving.
  Used for welding arc dwell, rivet insertion, glue dispensing etc.

- **velocity_scale**: Speed scaling applied to trajectory timestamps.
  1.0 = max speed, 0.5 = half speed, 0.3 = slow for inspection.

- **thermal_gap**: Minimum number of operations between same-face consecutive welds.
  Prevents thermal deformation from heat accumulation on one face.

## Overrides

Parameters can be overridden per operation in the task program JSON:
```json
{
  "id": "W001",
  "type": "weld",
  "velocity_scale": 0.3,
  "hold_duration_s": 0.8
}
```

Per-operation overrides take precedence over these defaults.
