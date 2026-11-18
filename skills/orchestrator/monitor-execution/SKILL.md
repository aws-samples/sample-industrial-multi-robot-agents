---
name: monitor-execution
description: >-
  Monitor cell execution after BehaviourTree delivery. Polls robot status
  and reports completion or errors. Use after run-welding-program delivers
  a BT to the cell agent.
license: Apache-2.0
metadata:
  priority: 70
  author: multi-robots-agents
  version: "1.0"
  category: monitoring
---

# monitor-execution

Polls robot states while the cell agent executes the BehaviourTree.

## When to use

After `run-welding-program` delivers the BT.
Poll until all robots return to idle or an error is detected.

## Steps

### 1. Poll robot states (every 2s)

No direct tool for streaming — use `get_all_robot_states` indirectly via
the cell agent's HTTP endpoint or status topic.

### 2. Report progress

Log a summary every ~10 operations:
```
[robot1] idle — last op W005
[robot2] busy — executing W009 (82% progress)
```

### 3. Detect completion

All robots `status == idle` with no `current_task` → execution complete.

### 4. Detect errors

Any robot `status == error` or a `CollisionAlert` received → activate
`handle-recovery` skill.

## Completion criteria

- All robots at `status=idle`
- No pending operations in the BT
- No active error conditions
