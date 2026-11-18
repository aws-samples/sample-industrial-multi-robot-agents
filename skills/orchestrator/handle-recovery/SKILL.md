---
name: handle-recovery
description: >-
  Handle errors during cell execution: estop activation, failed operations,
  timeout conditions. Decides whether to retry, skip, or abort the program.
  Use when monitor-execution detects an error or timeout.
license: Apache-2.0
metadata:
  priority: 75
  author: multi-robots-agents
  version: "1.0"
  category: recovery
---

# handle-recovery

Error recovery decision logic for the orchestrator agent.

## When to use

When `monitor-execution` reports:
- A robot in `status=error`
- A `CollisionAlert` with `severity=critical`
- FJT execution timeout (robot did not reach target)
- Planning failure from trajectory agent

## Decision logic

### Collision alert → ABORT

If a collision is detected:
1. The cell agent will have already triggered estop
2. Report the incident with collision point, robots involved, clearance
3. Do NOT retry automatically — require human confirmation

### FJT timeout (single robot) → RETRY once

If one robot timed out but no collision:
1. Re-plan the single failed operation via `call_trajectory_agent`
2. Retry dispatch once
3. If still failing → ABORT

### IK / planning failure → SKIP and continue

If a specific operation point is unreachable:
1. Log the unreachable operation ID
2. Continue with remaining operations
3. Report skipped ops in final summary

### Full planning failure → ABORT

If `call_trajectory_agent` returns `success=false`:
1. Log the full error
2. Report to operator
3. Stop program execution

## Output

Always report the recovery action taken:
```
RECOVERY: {action} for operation {op_id}
Reason: {error}
Robots affected: {robot_ns}
New status: {idle|retry|abort}
```
