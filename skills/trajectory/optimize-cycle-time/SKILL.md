---
name: optimize-cycle-time
description: >-
  Shorten the program's cycle time once a collision-free plan is within reach:
  keep steps PARALLEL, balance work between arms, avoid needless interlock waits
  and delays, and use the speed the cell allows instead of slowing moves down.
  Use after optimize-sequence has produced the steps and while planning each
  step with run_trajectory_planner. It never trades away a safety gate.
license: Apache-2.0
metadata:
  priority: 55
  author: multi-robots-agents
  version: "1.0"
  category: planning
---

# optimize-cycle-time

Cycle time is the wall-clock length of the whole program: the sum over steps of
the SLOWEST arm in each step, plus every wait, delay and hold. This skill is a
set of decisions to make while you plan, MEASURED with two tools (below). Every rule below
is subordinate to collision validation: a faster plan that fails
`validate_trajectory_set` is not a plan.

## Measure, don't guess

- `evaluate_program(cell_id, plan)` — the plan exactly as you would pass it to
  `assemble_behaviour_tree`. Returns `cycle_time_s`, each arm's `busy_s` /
  `idle_s`, `imbalance_s`, the steps where an arm with work ahead stands idle,
  `tight_steps` (clear of the warning band by < 3 cm) and `home` (can the arms
  return together from where they finish, and what it costs if not), plus
  `suggestions`. Call it before assembling, and on each alternative you weigh —
  keep the faster valid one.
- `find_clear_pose(cell_id, robot_ns, clear_of)` — ONE pose for an arm that is in
  the way (`blocked_by_parked_arm`, or idling where the other arm must work),
  clear of the listed legs with margin. Plan its `motion_request` as one move;
  never retreat in small guessed steps.
- A handover is a scheduling decision: pass `robots` (current orders) and the
  failure's `reason_code` to `reallocate_operation`. It refuses a handover when
  the cause is another arm in the way, and warns when it overloads an arm.

The assembler also merges back-to-back moves of one arm into one leg
(`merged_moves`) and staggers a return home that would otherwise run one arm at
a time (`home_step.staggered`).

## When to use

- After `run_mrta_pipeline` returns the step list, before planning step 1.
- Whenever you are about to serialise a step, add a delay, or add an interlock.
- Before calling `assemble_behaviour_tree`, as a final review.

## Where the time goes

| Source | What sets it | Your lever |
|---|---|---|
| Transit motion | cell `planning.max_velocity_scaling` (robots.yaml) | Omit call-level `velocity_scale` |
| Working stroke | cell `approach_velocity_fraction` (obstacles.yaml) | Only lower it, per weld, when the process needs it |
| Standoff length | `approach_offset_m` | Shortest standoff that still gets the tool in |
| Serial steps | `SEQUENTIAL` mode, `start_delay_s`, interlocks | Prefer PARALLEL; fix the conflict instead |
| Imbalance | allocation from `run_mrta_pipeline` | `reallocate_operation` from the busy arm |
| Holds | `hold_duration_s` from `operation-params.md` | Never shorten below the process value |

## Steps

### 1. Read the balance before planning

`run_mrta_pipeline` returns `cycle_time_balance` and `operation_counts`. A
balance well below 1.0 means one arm does most of the work and the other idles.
If one arm carries clearly more operations AND the other arm can reach some of
them (same face / approach side), plan the reachable ones on the idle arm — use
`reallocate_operation` to get the next-best arm rather than guessing.

### 2. Plan at the cell's speed

- Do NOT pass the call-level `velocity_scale` to `run_trajectory_planner`. It
  defaults to the cell's `max_velocity_scaling`, which is also the ceiling —
  anything higher is clamped, so passing it can only make the plan slower.
- Set `approach_velocity_scale` only when a process needs a slower stroke than
  the cell default (e.g. glue bead, contact move). Leave it out otherwise.
- Never put a `velocity_scale` in the `assemble_behaviour_tree` plan. The planned
  trajectory already carries its timing; the port is a playback stretch and must
  stay at its default.

### 3. Keep steps PARALLEL

A PARALLEL step costs `max(arm durations)`; the same step SEQUENTIAL costs the
sum. So when `validate_trajectory_set` reports a collision:

1. Apply `delay_fix.start_delays` from `validate_trajectory_set` as
   `start_delay_s` — a stagger of a second or two, and the step stays PARALLEL.
2. Change the weld point: give the conflicting arm a different pending weld —
   the one closest to its own base, so the arms work far apart — or hand the weld
   to the other arm (`reallocate_operation`). Re-plan the step with all arms in
   one call.
3. Interlock position (`find_interlock_set`): one arm waits at a pose clear of
   the other's path. Pick the candidate that adds the LEAST execution time.
4. Compare 2 and 3 by the time they add and take the cheaper. Serialising is
   not an option: every `serial_steps` entry the assembler reports is parallel
   time lost and must be removed with 2 or 3.

Never guess a `start_delay_s`: use the value `delay_fix` computed, which is
verified to clear at ±0.25 s too.

### 4. Compare arm durations inside a step

Each returned waypoint has `time_s`; the last one is that arm's duration. When
one arm in a PARALLEL step takes much longer than the other, the difference is
dead time. Pair the long operation with a longer one on the other arm in a later
step if the sequence allows (thermal gaps still apply), or give the short arm a
second operation via reallocation.

### 5. Review before assembling

Before `assemble_behaviour_tree`, check the plan you are about to submit:

- Every step that validated collision-free as a set is `PARALLEL`.
- No `start_delay_s`, interlock or `SEQUENTIAL` step exists without a collision
  result that required it.
- No call-level `velocity_scale` and no plan-level `velocity_scale` were set to
  save or spend time.
- `hold_duration_s` values match `operation-params.md` (holds are process time,
  not slack).

## Never

- Never skip, relax or bypass `validate_trajectory_set` / `validate-collision`
  to save time. It is a required gate.
- Never raise a speed above the cell's ceiling or ask for one — the ceiling is a
  cell safety declaration.
- Never drop an operation to shorten the program; reallocate it or report it
  unreachable per `check-reachability`.
