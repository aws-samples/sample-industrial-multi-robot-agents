---
name: coordinate-arms
description: >-
  The default (coordinated) planning mode. Each arm follows its OWN path, planned
  tight to the part with the approach and leave strategy you choose, and
  coordinate_program keeps the arms apart in TIME — an arm waits where it stands
  until another has completed a conflicting operation. You own the geometry
  (approach / leave waypoints, and a safe spot when an arm is in another's way);
  the coordinator owns the timing. Load this first in coordinated mode.
license: Apache-2.0
metadata:
  priority: 92
  author: multi-robots-agents
  version: "1.0"
  category: planning
---

# coordinate-arms

Two arms working one part rarely collide because of WHERE they go — they
collide because they go there at the SAME TIME. So in this mode nobody makes
detours, parks at HOME or serialises the program. Each arm's path stays close
to the part, and the timing is solved deterministically:

- `coordinate_program` treats each arm's moves and standing poses as occupied
  space, finds every pair that could come inside the 0.20 m warning band, and
  starts every move as early as those conflicts allow. The arm that must wait
  holds WHERE IT STANDS — at its last weld, i.e. near the part.
- Each wait becomes a gate: "wait until arm X has COMPLETED operation Y". Gates
  are events, not clock times, so the separation holds whatever speed the arms
  really run at.

Your job is the geometry that makes each weld right and keeps the arms near the
part. You never decide the timing.

## Steps

### 1. Allocation and order

Call `run_mrta_pipeline`. Take each arm's operations in the order they appear in
the steps — that is the arm's route (near-first from its own base). Ignore the
step pairing: in this mode the coordinator decides what runs at the same time.

### 2. Plan every arm's route — one arm per call, in order

For each arm, plan its operations one by one with `run_trajectory_planner`,
ONE ARM PER CALL. Coupling arms in one call makes each avoid the other's whole
swept volume at every moment — the detours this mode exists to remove.

For every weld you decide the strategy and write it into the motion request:

| Decision | Field | Guidance |
|---|---|---|
| Approach standoff | `approach_offset_m` | shortest that still lets the tool enter on-axis — open point 0.08–0.12 m, tight pocket 0.05–0.07 m |
| Approach speed | `approach_velocity_scale` | only when the process needs a slower stroke |
| Leaving the previous weld | `retreat_offset_m` | the on-axis lift-off from the PREVIOUS weld before this motion; short (0.05–0.10 m) keeps the arm near the part |
| Tool pose | `orientation` + `constrain_orientation: true` | the weld's `weld_orientation`; never relax it |
| Blocked entry | a `move` via-point first | see `approach-strategy`; keep it close to the part |

Validate every weld with `validate_approach` and re-plan any that fails.

### 3. Coordinate

Call `coordinate_program(cell_id, sequences)` with `sequences = {robot_ns:
[operation_id, ...]}` in execution order. Read:

- `makespan_estimate_s` — the program's duration with all arms coordinated.
- `parallel_s` — seconds in which two or more arms move at once.
- `arms.<ns>.waits_s` / `gates` — where an arm waits and for what.

### 4. When it reports a deadlock — find a safe spot

A deadlock means an arm STANDS where another must pass, and cannot move on
until that other arm has passed. `needs_safe_spot` names the arm in the way and
the operation after which it must step aside. Choose its safe spot:

1. **Close to the part** — a short lift above its last weld (0.15–0.30 m along
   the weld's approach axis), not HOME.
2. **On its own side** — offset toward its own base, away from the other arm's
   path.
3. **Clear of the waiting arm** — check against the operation the waiting arm
   needs (`deadlock.waiting_to_start`).

Plan it as a `move` (`operation_type: "move"`, `approach: false`, id
`<ns>_SAFE_<n>`) chained after that operation, insert it into that arm's
sequence right after it, and call `coordinate_program` again.

An arm that has FINISHED all its work is never your problem: when it stands in
another arm's way, the coordinator plans its return home itself and reports it
in `early_home`. Safe spots are only for arms that still have work to do.

You may also resolve a deadlock by ORDER: moving a weld to the other arm
(`reallocate_operation`) or later in its own route — when that is shorter than a
safe-spot move, prefer it.

### 5. Assemble — immediately

As soon as `coordinate_program` returns `success: true`, call
`assemble_coordinated_tree(cell_id, sequences)` with the `sequences` it
returned (they include any early-home legs). Do not keep iterating to shave
seconds off the estimate: every extra round costs more than it saves. It builds one Sequence per arm,
all arms in a Parallel, a `WaitForRobot` gate wherever the coordinator ordered
arms, and the coupled return HOME. It refuses a deadlocked program.

## Never

- Never couple two arms in one `run_trajectory_planner` call in this mode.
- Never serialise the program or park an arm at HOME to avoid a conflict.
- Never pick a start delay yourself — the gates handle timing.
