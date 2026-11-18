---
name: manage-gates
description: >-
  Manage position gates and timing constraints between parallel robot motions.
  Determines when WaitForRobot nodes should release, when CheckSafety nodes
  should pass, and when temporal stagger delays are needed. Use within
  interpret-behaviour-tree execution.
license: Apache-2.0
metadata:
  priority: 70
  author: multi-robots-agents
  version: "2.0"
  category: coordination
---

# manage-gates

Implements the gate management layer of the BehaviourTree executor.

## Gate types

### Position gates (`WaitForRobot`)
A robot reaches its interlock position before the other robot proceeds.
Check via `check_position_gate(robot_ns, expected_position, tolerance_m=0.05)`.
The `depends_on_*` ports are `{key}` blackboard references — resolve them first.

### Temporal gates (`Delay`)
Stagger robot starts to avoid joint-space conflicts near home position.
Implemented by sleeping the decorator's `delay_msec` before ticking its child, so the
stagger is a node in the tree rather than a field on the leaf.

### Safety gates (`CheckSafety`)
External conditions that must be true before a step proceeds.
Supported conditions:
- `estop_clear` — no active emergency stop
- `cycle_ok` — previous operation completed without error

## Gate timeout policy

All gates have a default 30s timeout. If a gate times out:
1. Log the timeout with robot_ns and expected position
2. Call `get_all_robot_states` to check actual positions
3. If robot is within 2× tolerance → release gate (sensor noise)
4. If robot is stuck → report to orchestrator → trigger estop

## Reference

See `references/gate-conditions.md` for timeout values per operation type.
