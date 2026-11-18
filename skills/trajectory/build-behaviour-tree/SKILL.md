---
name: build-behaviour-tree
description: >-
  Assemble the complete Behaviour Tree in ONE call from a compact decision plan,
  after every operation has been planned with run_trajectory_planner. The
  assembler pulls cached trajectories by operation_id, so you never pass joint
  arrays. The resulting tree is the output sent to the cell agent for execution.
license: Apache-2.0
metadata:
  priority: 80
  author: multi-robots-agents
  version: "2.0"
  category: planning
---

# build-behaviour-tree

Builds a **BehaviorTree.CPP v4** tree that the cell agent interprets and executes.
The tree is assembled in a single tool call — you do NOT hand-build nodes one by
one, and you do NOT pass trajectory arrays.

## Prerequisite: plan with operation_ids

Before assembling, call `run_trajectory_planner` for every robot motion and give
each motion a unique `operation_id` in its request, e.g.:

```json
{"robot_ns": "robot1", "operation_id": "R1_W1", "operation_type": "weld",
 "target": {"x": ..., "y": ..., "z": ...}, "approach_vector": [0, 0, -1]}
```

The planner (chosen deterministically by the task, not by you)
caches the full joint trajectory server-side under
`(cell_id, robot_ns, operation_id)`. The assembler reads it back from there.

## One-shot assembly

Call `assemble_behaviour_tree(cell_id, plan)` exactly once. `plan` carries only
decisions, not trajectories:

```json
{
  "steps": [
    {
      "mode": "PARALLEL",
      "robots": [
        {"robot_ns": "robot1", "operation_id": "R1_W1", "operation_type": "weld"},
        {"robot_ns": "robot2", "operation_id": "R2_W1", "operation_type": "weld"}
      ]
    },
    {
      "mode": "SEQUENTIAL",
      "robots": [
        {"robot_ns": "robot1", "operation_id": "R1_W2", "operation_type": "weld",
         "start_delay_s": 0.0, "has_interlock": true,
         "interlock_target": {"ee_position": [x, y, z]},
         "depends_on_robots": ["robot2"],
         "depends_on_positions": [{"robot": "robot2", "position": [x, y, z],
                                   "tolerance_m": 0.05}]}
      ]
    }
  ],
  "home_at_end": true
}
```

Per-robot optional fields (all default sensibly): `start_delay_s`,
`hold_duration_s`, `velocity_scale`, `has_interlock`, `interlock_target`,
`depends_on_robots`, `depends_on_positions`.

## What the assembler produces

The output is a **BehaviorTree.CPP v4** tree (see `references/bt-node-types.md`),
not a repo-local dialect — it opens in Groot2 and ticks in a C++ factory.

- Each step becomes a `Parallel` node (both robots at once, `success_count="-1"` =
  all must succeed) or a `Sequence` node (`mode: SEQUENTIAL`); a single-robot step
  becomes a bare `ExecuteTrajectory` node.
- Every step's robot becomes an `ExecuteTrajectory` leaf whose `trajectory` port is
  a `{key}` reference into the tree's `blackboard`, filled from the cache by
  `operation_id`.
- A `start_delay_s` becomes a `Delay` decorator (`delay_msec`) wrapping that leaf,
  which is how BT.CPP expresses a timed gate.
- Steps run in order under a top-level `Sequence`.
- If `home_at_end` is true (default), a final home step (one `MoveHome` node per
  robot, wrapped in `Parallel` when there is more than one) is appended.
- The tree is validated automatically before it is returned.

Node vocabulary: `Sequence`, `Parallel`, `Delay`, `ExecuteTrajectory`,
`WaitForRobot`, `CheckSafety`, `MoveHome`. See `references/bt-node-types.md` for
the full port schema and the blackboard convention.

## Return shape

```json
{
  "success": true,
  "behaviour_tree": {
    "tree_id": "bt_<timestamp>_<cell_id>",
    "cell_id": "welding_cell_2_small",
    "BTCPP_format": "4",
    "main_tree_to_execute": "MainTree",
    "root": { "type": "Sequence", "children": [ ... ] },
    "blackboard": { "traj_robot1_R1_W1": [ {"positions": [...], "time_s": 0.0}, ... ] },
    "xml": "<root BTCPP_format=\"4\" main_tree_to_execute=\"MainTree\"> ... ",
    "metadata": {"steps": 2, "operations": 3}
  },
  "validation": { "valid": true, ... }
}
```

Return the `behaviour_tree` object as your final message — verbatim, including
`blackboard`. The waypoint arrays live there and the nodes only reference them, so a
tree without its blackboard cannot execute. Never rewrite ports or inline the
blackboard by hand.

`export_behaviour_tree_xml(tree)` re-emits the XML (and its blackboard) on demand,
e.g. after editing a tree or to render a legacy-dialect tree in the standard format.

On failure the assembler returns `{"success": false, "error": ..., "missing":
[{step, robot_ns, operation_id}, ...]}`. A `missing` list means those
trajectories were never cached — re-run `run_trajectory_planner` with each listed
`operation_id`, then call `assemble_behaviour_tree` again.

## Fallback (rare)

`build_behaviour_tree_node` / `emit_parallel_node` / `validate_behaviour_tree`
remain available for manual construction, but only use them to debug a case the
one-shot assembler cannot handle. The default path is `assemble_behaviour_tree`.
