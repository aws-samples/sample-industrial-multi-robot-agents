---
name: find-interlocks
description: >-
  Find safe intermediate positions (interlocks) that allow two robots to move
  simultaneously toward their targets without collision. Use when
  validate-collision returns a trajectory collision, or when targets are
  closer than 0.4m. Returns a 2-step execution plan the behaviour tree
  builder can use.
license: Apache-2.0
metadata:
  priority: 70
  author: multi-robots-agents
  version: "1.0"
  category: collision-avoidance
---

# find-interlocks

Computes minimum-cost intermediate joint configurations (one per robot, for ANY
number of robots) such that all robots can safely move to their interlocks
simultaneously, then all can safely move to their targets simultaneously.

## When to use

This is step 3 of resolving a conflicting PARALLEL step. Try a start delay
(`delay_fix` from `validate_trajectory_set`) and a different weld pairing first —
both keep the arms working at the same time.


1. `validate-collision` returns collision (clearance < 0.24m between trajectories)
2. `assemble_behaviour_tree` reports a step in `blocked_steps`, or serialises one
   you needed to stay parallel
3. Two or more robots must approach the same face of the workpiece

Use `find_interlock_set(cell_id, targets=[{robot_ns, x, y, z}, ...])` with EVERY
converging arm. A strategy is accepted only when all robot pairs clear the margin:
resolving two pairs of a 3-way convergence does not make the third pair safe, so
never call it with a subset of the arms. `find_interlocks(...robot1/robot2...)` is
the two-robot shorthand.

Do NOT decide this from how far apart the TARGETS are. Distance between weld
points runs opposite to arm clearance for arms facing each other (measured -0.607
on `welding_cell_2_small`): the pair 1.201 m apart is the one that e-stopped the cell,
and it was clear at both its start and goal poses — it breached only in transit.
Let `validate_trajectory_set` on the planned trajectories tell you.

## Steps

### 1. Check if interlock is needed

Call `check_interlock_needed(robot1_ns, target1, robot2_ns, target2)`.
If `needs_interlock=false`, proceed with direct parallel motion.

### 2. Generate interlock candidates

Call `generate_interlock_candidates(robot_ns, target_joints)` for each robot.
This produces joint configurations using these strategies (tried in order):
- `raise_arm`: shoulder up (j2-0.5 rad), elbow open (j3+0.4 rad) — minimal travel
- `rotate_outward`: rotate base away from workspace centre (j1±0.6 rad)
- `partial_retract`: 70% interpolation toward home
- `approach_from_above`: IK to target XY + 0.3m Z offset
- `home`: full home config — last resort

See `scripts/interlock_optimizer.py` for the implementation.

### 3. Select the best pair

Call `select_best_interlock_pair(candidates1, candidates2, min_clearance_m)`.

Scoring: `travel_distance(r1) + travel_distance(r2) + clearance_penalty`
where `clearance_penalty` penalises pairs closer than `min_clearance_m`.

`min_clearance_m` from `references/clearance-thresholds.md`:
- weld: 0.24m (sum of 2 × 0.12m sphere radii)
- laser: 0.30m (larger safety margin)

### 4. Build the execution plan

If a valid pair is found:
```json
{
  "needs_interlock": true,
  "strategy": "raise_arm + rotate_outward",
  "robot1_interlock": {"ee_position": [x,y,z], "joint_positions": [j1..j6]},
  "robot2_interlock": {"ee_position": [x,y,z], "joint_positions": [j1..j6]},
  "execution_plan": [
    {
      "step": 1, "action": "PARALLEL",
      "motions": [
        {"robot": "robot1", "target": interlock1_ee},
        {"robot": "robot2", "target": interlock2_ee}
      ],
      "depends_on": []
    },
    {
      "step": 2, "action": "PARALLEL",
      "motions": [
        {"robot": "robot1", "target": final_target1},
        {"robot": "robot2", "target": final_target2}
      ],
      "depends_on": [
        {"robot": "robot1", "position": interlock1_ee, "tolerance_m": 0.05},
        {"robot": "robot2", "position": interlock2_ee, "tolerance_m": 0.05}
      ]
    }
  ]
}
```

If no valid pair found (all combinations collide), do not run the arms one after
the other: change the weld pairing (step 5) instead. The legacy helper still
returns:
```json
{
  "needs_interlock": true,
  "strategy": "sequential_fallback",
  "execution_plan": [
    {"step": 1, "action": "SEQUENTIAL", "robot": "robot1", "target": target1},
    {"step": 2, "action": "SEQUENTIAL", "robot": "robot2", "target": target2}
  ]
}
```

Pass this execution plan to `build-behaviour-tree`.

### 5. Choose the interlock that costs the least time

Serialising the two arms is NOT a fallback. When several interlock poses clear
the conflict, cost each one — the waiting arm's detour plus the time it waits at
the gate — and use the cheapest. If none clears it, go back and change the weld
point: pair the arm with a different pending weld, closest to its own base.
An arm parked at its last weld can itself be in the way; its interlock pose is
then a short `move` clear of the other arm, planned in the SAME step.

## Reference

See `references/clearance-thresholds.md` for minimum clearance per operation.
See `scripts/interlock_optimizer.py` for candidate generation and scoring.
