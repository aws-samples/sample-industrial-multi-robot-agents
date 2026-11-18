---
name: collision-response
description: >-
  Respond to collision alerts during BehaviourTree execution. Triggers estop
  on critical collisions, logs warnings for marginal clearance. Use when
  get_all_robot_states or the collision monitor reports unsafe robot proximity.
license: Apache-2.0
metadata:
  priority: 95
  required: true
  author: multi-robots-agents
  version: "1.0"
  category: safety
---

# collision-response

Safety response layer — detects and responds to proximity violations.

## When to use

During BehaviourTree execution when:
1. `get_all_robot_states` shows unexpected robot positions
2. A CollisionAlert message arrives on `/{cell_id}/collision_alert`
3. A robot reports `status=error`

## Response levels

### Warning (clearance 0.15m – 0.24m)
- Log the warning with robot pair and clearance
- Continue execution but flag the step for review
- Store the near-miss for trajectory knowledge feedback

### Critical (clearance < 0.15m)
1. Call `trigger_estop(cell_id, reason="collision_detected")`
2. Stop all pending dispatches
3. Report to orchestrator with:
   - Robot pair
   - Collision point
   - Step/operation ID
   - Minimum clearance achieved

## Recovery (after estop)

After operator clears the estop:
1. Get all robot states
2. If robots are in unexpected positions → require manual reset
3. If robots are at known positions → orchestrator may re-plan from current step

## Sphere model parameters

Each robot link is modelled as a sphere of radius **0.12m**.
Critical threshold = 0.00m clearance (spheres overlap).
Warning threshold  = 0.24m - 0.15m = 0.09m margin from critical.
