---
name: run-welding-program
description: >-
  Execute an industrial operation program (welding, riveting, gluing) on a
  cell. Reads the cell spec and task program, publishes cell description,
  requests trajectory planning from the trajectory agent, and delivers the
  BehaviourTree to the cell agent. Use when the orchestrator receives an A2A
  task request.
license: Apache-2.0
metadata:
  priority: 80
  author: multi-robots-agents
  version: "1.0"
  category: orchestration
  operations: weld, rivet, glue, laser_cut, mark
---

# run-welding-program

Coordinates the full pipeline from task receipt to BehaviourTree delivery.

## When to use

When an A2A request arrives at `:8889/a2a/run` with `cell_id` and `program`.

## Steps

### 1. Load cell specification

```
read_cell_spec(cell_id)
```

Read the cell configuration to understand robots, workspace, and available
programs. Log the robot count and operation count.

### 2. Load task program

```
read_task_program(cell_id, program_name)
```

The program JSON contains the operation list. Each operation:
```json
{
  "id": "W001",
  "type": "weld",
  "position": [x, y, z],
  "approach_vector": [ax, ay, az],
  "orientation": { "x": 0.0, "y": -0.7071, "z": 0.0, "w": 0.7071 },
  "sequence_group": "structural",
  "priority": 1,
  "assigned_robot": "robot1"
}
```

`assigned_robot` is OPTIONAL — include it only when an operation must run on a
specific robot (e.g. single-robot part handling or gluing). Otherwise robot
allocation is decided by the trajectory agent at planning time, not by the task.
Pass the operations through to the trajectory agent unchanged; do not assign
robots yourself.

If `operations` is empty, report an error — do not continue.

### 3. Publish cell description

```
publish_cell_description(cell_id)
```

This confirms every arm has a robot description (URDF). Nothing is published on
ROS 2: viewers fetch it from the trajectory agent's MCP (`get_robot_description`). If an
arm is listed under `missing`, report it — the viewer cannot show that arm.

### 4. Request trajectory planning

The motion planner is chosen by the **submitted task**, not by you. Pass the
task program's `trajectory_backend` straight through in the constraints. If the
program does not declare one, omit it (the trajectory agent then uses the
`TRAJECTORY_BACKEND` sim-launch default):

```
call_trajectory_agent(
    cell_id=cell_id,
    operations=program["operations"],
    constraints={"trajectory_backend": program.get("trajectory_backend")}  # "ompl" | "rrt" | omit
)
```

Do not substitute your own backend choice. The trajectory agent binds this value
before planning and there is NO fallback between planners: if the requested
backend cannot plan, the response reports failure — it is never silently
rerouted to another planner. Report such a failure with the full trajectory
agent response.

The trajectory agent returns a BehaviourTree JSON.
If planning fails, report the error with the full trajectory agent response.

### 5. Confirm BT received

Log:
- `tree_id`
- `planning_time_s`
- Number of `ExecuteTrajectory` nodes (= number of operation points planned)

The cell agent will now execute the BT automatically via the ROS 2 subscription.

### 6. Report success

Return a summary:
```
Program '{program_name}' planned for {n_ops} operations in {planning_time}s.
BT tree_id={tree_id} delivered to cell agent.
```

## Reference

See `references/task-json-format.md` for the full task program schema.
