# Task program JSON format

A task program lives at `cells/<cell_id>/tasks/<program>.json` and describes an
industrial operation program (welding, riveting, gluing, laser, mark, move).

## Top-level fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `program` | string | yes | Program name (matches the file stem). |
| `cell_id` | string | yes | Cell this program runs on. |
| `version` | string | no | Program version. |
| `description` | string | no | Human-readable summary. |
| `trajectory_backend` | `"ompl"` \| `"rrt"` | no | Motion planner to bind. If omitted, the sim-launch `TRAJECTORY_BACKEND` default is used. There is NO fallback between planners. |
| `extensions` | object | no | Process metadata (strategy, cycle_time, weld_params, …). Passed through; not required for planning. |
| `operations` | array | yes | The operation points (see below). Must be non-empty. |

## Operation fields

Each entry in `operations`:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique operation id. Use neutral ids (e.g. `W01`); do NOT encode a robot in the id. |
| `type` | string | yes | `weld` \| `rivet` \| `glue` \| `laser_cut` \| `mark` \| `move`. |
| `position` | `[x, y, z]` | yes | Target point in the world/cell frame (metres). |
| `approach_vector` | `[ax, ay, az]` | yes | Unit direction the tool travels along to reach the point (e.g. `[0, 0, -1]` = descend from above). |
| `orientation` | `{x, y, z, w}` | no | Target tool (`welding_tcp`) orientation as a quaternion in the world frame. If omitted, the planner derives one from the approach vector. |
| `sequence_group` | string | no | Ordering group (e.g. `structural`, `secondary`). Controls execution order, NOT robot allocation. |
| `priority` | int | no | Ordering hint within/among groups. |
| `assigned_robot` | string | no | **Optional robot lock.** If set (e.g. `"robot1"`), this operation MUST be executed by that robot. Use only for operations that only one robot can perform (single-robot part handling, gluing, a moving part, etc.). |

## Robot allocation

Robot allocation is a **planning-time decision made by the trajectory agent**
(`run_mrta_pipeline`), not a property baked into the task. The task says WHERE
the operations are; the planner decides WHICH robot performs each one, balancing
reachability and cycle time.

The single exception is `assigned_robot`: an operation that sets it is pinned to
the named robot and never reallocated. Leave it out for every operation you want
the planner to distribute freely.

## Example

```json
{
  "program": "car_door_spot_welding",
  "cell_id": "welding_cell_2_small",
  "trajectory_backend": "ompl",
  "operations": [
    {
      "id": "W01",
      "type": "weld",
      "position": [-1.082, -1.489, 1.60],
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
