---
name: interpret-behaviour-tree
description: >-
  Walk and execute a BehaviorTree.CPP v4 tree received from the orchestrator.
  Handles Sequence, Parallel, Delay, ExecuteTrajectory, WaitForRobot,
  CheckSafety, and MoveHome nodes. Dispatches trajectories to robot agents,
  manages position gates, and monitors for collision. Use when a BT arrives on
  the /{cell_id}/behaviour_tree topic.
license: Apache-2.0
metadata:
  priority: 80
  author: multi-robots-agents
  version: "2.0"
  category: execution
---

# interpret-behaviour-tree

Executes a BehaviorTree.CPP v4 tree by walking the node graph and dispatching to
robots.

## When to use

When a BehaviourTree message arrives at `/{cell_id}/behaviour_tree`.

## Format

The tree is standard **BehaviorTree.CPP v4** (see
`skills/trajectory/build-behaviour-tree/references/bt-node-types.md`). Each JSON node
is `{"type": <registration ID>, "name": <label>, "ports": {...}, "children": [...]}`.

**Read the blackboard first.** Bulky ports — `trajectory`, `interlock_target`,
`depends_on_robots`, `depends_on_joints`, `depends_on_positions` — are not inlined;
their value is the literal string `{key}` referring to an entry in the tree's
top-level `blackboard` dict. Resolve the reference before use; a `{key}` with no
blackboard entry is a missing trajectory, which is a hard failure, not an empty move.

Trees recorded in the old dialect (`EXECUTE` / `PARALLEL` / `HOME` / …, with
top-level fields instead of `ports`) still replay — every accessor normalises both —
but new trees are always v4.

## Node execution rules

### `Sequence` / `SequenceWithMemory`
Execute children in order. Stop on first failure.

### `Parallel`
Dispatch ALL children simultaneously — do NOT wait between dispatches. Wait for all
children before marking the node done. `success_count="-1"` (the assembler's default)
means every child must succeed; `failure_count="1"` means one failure fails the node.

### `Delay`
Wait `delay_msec` milliseconds, then tick its single child. This is how a timed gate
is expressed — there is no per-leaf delay field.

### `ExecuteTrajectory`
1. Resolve the `trajectory` port through the blackboard.
2. If `has_interlock = true`:
   - Dispatch to the resolved `interlock_target.joint_positions` (shortened trajectory)
   - Call `check_position_gate` for the interlock EE position
3. Dispatch the full trajectory via `publish_task_assignment`
4. Wait `hold_duration_s` at target

### `WaitForRobot`
Call `check_position_gate` for each entry in the resolved `depends_on_positions`
(or `depends_on_joints`). Block until ALL are satisfied.

### `CheckSafety`
Check the `condition` string:
- `estop_clear`: verify no active estop
- `cycle_ok`: verify previous operation completed

### `MoveHome`
Dispatch the resolved `trajectory` for that robot; if the port is empty, dispatch a
blind home move (6 joints → `[0, -1.57, 2.2, 0, -0.8, 0]`). One node per robot, so a
`Parallel` of `MoveHome` nodes homes the whole cell at once.

### Anything else
Fail loudly with "unregistered node". BT.CPP defines more built-ins (`Fallback`,
`Timeout`, `RetryUntilSuccessful`, …) than this executor implements; silently
skipping one would drop coordination the planner relied on.

## Monitoring during execution

After each `Parallel` node completes, call `get_all_robot_states` to verify all robots
are in expected positions. If any robot reports `status=error`, call `trigger_estop`
immediately.

## Completion

BT execution is complete when the root node returns SUCCESS.
Report final robot states to the orchestrator.
