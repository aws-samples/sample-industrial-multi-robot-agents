---
name: optimize-sequence
description: >-
  Allocate operation points (weld/rivet/glue/laser/mark) to robots and compute
  the optimal parallel execution sequence using MRTA best practices. Use when
  given a list of operation points that must be distributed across robots in a
  cell. Call this skill FIRST before planning any trajectories.
license: Apache-2.0
metadata:
  author: multi-robots-agents
  version: "1.0"
  category: planning
  operations: weld, rivet, glue, laser_cut, laser_weld, mark, move
---

# optimize-sequence

Computes the optimal execution sequence for all operation points using a
6-stage MRTA (Multi-Robot Task Allocation) pipeline based on automotive
bodyshop best practices.

## When to use

Call this skill when the orchestrator sends a task containing operation points
that must be distributed across multiple robots. This skill handles ALL
operation types — welding, riveting, gluing, laser cutting, marking. The
thermal gap and approach parameters vary by operation type (see
`references/operation-params.md`).

## Steps

### 1. Prepare the operation list

Extract operation points from the task into this format:
```json
[
  {
    "id": "W001",
    "operation_type": "weld",
    "position": [x, y, z],
    "approach_vector": [ax, ay, az],
    "sequence_group": "structural",
    "priority": 1,
    "assigned_robot": "robot1"
  }
]
```

**Robot allocation is decided here, not in the task.** The task lists WHERE the
operations are, not WHICH robot does them. `run_mrta_pipeline` distributes the
operations across the cell's robots (Stage 1). The only exception is the
optional `assigned_robot` field: if an operation sets it, that operation is
LOCKED to the named robot (used for single-robot operations such as part
handling or gluing where only one robot can perform it). Operations without
`assigned_robot` are allocated automatically — do not invent an assignment for
them from the operation id or any other field.

`run_mrta_pipeline` IGNORES an `assigned_robot` / `robot_ns` you add yourself
(only the operator's and the task file's locks bind; the result lists what it
ignored in `ignored_pins`). Re-running it gives the same allocation. Each weld
goes to an arm that can reach it from the approach side — the tool checks reach
and side, and balances work onto idle arms — and a weld no arm can take is
returned in `unreachable`, never forced onto an arm. To change who does ONE weld
after its arm failed it, use `reallocate_operation`. Never take arms out of the
plan to avoid a conflict: sequence or stagger the conflicting step instead.

### 2. Run the MRTA pipeline

Call the `run_mrta_pipeline` tool with the operation list and cell_id.

The pipeline runs deterministically:
- **Stage 1 — Reachability pre-filter**: Pinocchio IK validates each point
  against its preferred robot. Unreachable points go in `unreachable[]`.
- **Stage 2 — Regional allocation**: Points assigned by approach_vector
  dot-product (robot whose base faces the surface owns the face). Points the
  score cannot separate go to the arm owning the nearest already-assigned point,
  so a contested row is cut once instead of interleaved between the arms.
- **Stage 3 — TSP 2-opt sequence**: Nearest-neighbour then 2-opt improvement
  per robot to minimise air-cut travel.
- **Stage 4 — Thermal spread**: Same-face consecutive pairs reordered.
  Thermal gap from `references/operation-params.md`.
- **Stage 5 — Least-movement pairing**: Every arm takes the next point in its own
  Stage 3/4 tour, so the tour that was optimised is the tour that gets driven.
- **Stage 6 — Cycle-time balance**: Reports balance ratio (1.0 = perfect).

The pairing is a COST decision, not a safety one — the pipeline runs before any
trajectory exists, so it cannot see how an arm gets anywhere. Whether a step's
arms may actually move at once is decided by `assemble_behaviour_tree`, which
checks the swept volumes and serialises or refuses the step.

### 3. Review the output

The pipeline returns:
```json
{
  "total_steps": 13,
  "steps": [
    {
      "step": 1,
      "robot1": {"id": "W001", "x": ..., "y": ..., "z": ..., "approach_vector": [...]},
      "robot2": {"id": "W007", "x": ..., "y": ..., "z": ..., "approach_vector": [...]},
      "min_inter_robot_dist_m": 0.812
    }
  ],
  "cycle_time_balance": 0.94,
  "thermal_violations_fixed": 2,
  "unreachable": []
}
```

`min_inter_robot_dist_m` is the distance between the weld POINTS. Read it as
telemetry, never as a clearance and never as a gate: measured across the six
parallel pairs of a shipped `welding_cell_2_small` plan it correlates with the arms'
true swept clearance at **-0.607** — the wrong sign. Two arms working opposite
sides of the part stand furthest apart precisely where each is most extended
toward the other, so the pair with the largest value (1.201 m) was the one that
closed to 0.1153 m and e-stopped the cell, while the smallest (0.688 m) had four
times the margin. Clearance comes from `validate_trajectory_set` on the actual
trajectories, and from nothing else.

### 4. Handle unreachable points

For each point in `unreachable[]`, `load_skill('check-reachability')` to decide
whether to re-assign it to the other robot or record it unreachable. Skip points
that are truly outside the workspace.

### 5. If balance < 0.9

Note the imbalance in the result — the `build-behaviour-tree` skill will
handle timing compensation via start delays.

## Reference

See `references/operation-params.md` for thermal gap, approach offset,
hold duration, and velocity scale per operation type.
