# ADR-0001 — `ompl-compound`: centralised multi-arm planning, optional lockstep execution

- **Status:** accepted
- **Date:** 2026-10-05

## Context

The `ompl` backend plans the arms of a PARALLEL step **one after another**. Each
later arm avoids a *time-blind* swept tube around the arms already planned, and
`inter_robot_clearance` checks the result on a shared clock afterwards. On
`welding_cell_2_small` (no fixture, 15 welds, two KR240 arms facing each other)
most planning runs fail this way: steps the gate marks `REFUSED`/`SEQUENTIAL`,
repeated re-plans, tall arcs and exhausted step budgets. Measured on the same
two-arm steps, both starting from HOME:

| step | `ompl` | `ompl-compound` |
|---|---|---|
| W03 + W13 | 65.6 s; robot2 refused (its path cut 0.012 m into robot1) | 8.1 s; both arms, 0.36 m min clearance |
| W02 + W14 | 316 s; robot2 refused (1.37 m tool arc) | 7.1 s; both arms, 0.27 m min clearance |

## Decision

1. **A new trajectory backend, `ompl-compound`**, in its own module
   (now `agents/trajectory/backends/ompl_compound/`) and registered in the
   backend facade (`planner_backend.py`). It is selected like any backend:
   `make plan … TRAJECTORY_BACKEND=ompl-compound`, `--backend ompl-compound`, or
   `"backend": "ompl-compound"` on `/a2a/plan`. The planning modes, the MRTA weld
   allocation and the skills are unchanged: only the motion calculation of each
   step differs.
2. **One compound query per step.** The transit of every arm moving in the step
   is planned as one path in an OMPL `CompoundStateSpace`, with one
   `RealVectorStateSpace(6)` per arm (weight 1.0). The validity checker unpacks
   the compound state and runs FK for every arm. It rejects robot-vs-environment
   contact using the same `_config_is_free` predicate as `ompl`. It rejects
   robot-vs-robot proximity when any two arms come within
   `collision.warning_distance` + `OMPL_COMPOUND_INTER_ARM_EXTRA_M` (0.01 m),
   which is the pair criterion the runtime monitor e-stops on. Arms not moving in
   the step are fixed obstacles, held where they last ended (or at HOME).
3. **The approach strokes stay per arm.** The on-axis Cartesian lift-off and
   the orientation-constrained descent are built exactly as in `ompl`. The
   lift-offs are padded to end together, the joint transit is timed as **one
   stacked vector** (TOPP-RA over all joints, so every arm's transit samples
   carry the same timestamps), and the descents start together when it ends.
   The whole timeline is then checked on one clock with `inter_robot_clearance`.
4. **Candidate selection.** The checker first tries the direct joint line, then
   *staggered* straight lines (one arm leads by 50 %, 75 % or 100 % of its path
   while the others wait), then up to `OMPL_COMPOUND_ATTEMPTS` RRT solves. Among
   the candidates whose worst-arm tool rise is under `OMPL_PATH_MAX_RISE_M`,
   the one with the shortest velocity-bounded duration wins. Otherwise the
   flattest wins and is still subject to `OMPL_MAX_EMITTED_RISE_M`.
5. **A step is refused as a whole.** If any check fails (no path, emitted
   clearance breach, tool rise, the shared-clock gate), no arm of the step ships.
   Each arm's guarantee assumed the others moved, so dropping one would make the
   rest unsafe. One request per arm per call is enforced; a sequence for one arm
   is what `steps` is for.
6. **Lockstep execution is opt-in: `make sim … LOCKSTEP=1`.** The planner tags
   each step's legs with `sync_group`/`sync_members` on their first sample.
   - **Off (default):** the tags are ignored and the legs run as plain `Parallel`
     children, as today.
   - **On:** the cell executor holds the legs at a barrier and dispatches them at
     one instant. While they run, it estimates each arm's time on its trajectory
     from joint feedback against that shared clock. If an arm drifts more than
     `lockstep_tolerance_s` (0.5 s) for `lockstep_grace_s` (0.5 s), it **e-stops
     the cell**. It also refuses to start, without dispatching anything, when a
     partner leg never arrives (for example, legs placed in a `Sequence`) or when
     the members run at different `velocity_scale`.

## Consequences

- **Halting is the only enforcement.** No robot agent or controller here
  accepts a start time, a time offset or pause/resume, or reports trajectory
  progress (FJT goals start "now"; feedback is ignored). Real lockstep (pausing
  the leader, re-timing the lagging arm) needs those controller features first.
  Until then, lockstep turns loss of sync into a stop instead of a near-miss.
- **Without lockstep, compound legs keep only the existing protections.** Those
  are the existing schedule-adherence check and the runtime collision monitor.
  If the assembler staggers them with `Delay`, its swept-volume gate re-checks
  the result.
- **Not every pairing can run in parallel.** Some steps are refused for real
  geometric reasons: the two goal standoffs are within the warning band
  (W04+W12, W06+W10), or a pairing whose only clear route is a tall arc
  (W01+W13). The agent then needs another pairing or a SEQUENTIAL step, which is
  the allocation's job, not the backend's.
- **12-D validity checks run in Python.** Each check covers every moving arm
  (FK plus the arm-pair test). On the measured steps the whole step still
  planned in under 10 s, against tens or hundreds of seconds for `ompl`.
- **Plan edits name their backend and mode** (amended 2026-10-11).
  `replan_operation` takes `backend` and `mode` like `plan_trajectories`. With
  `ompl-compound`, the edited operation's whole PARALLEL step is re-planned in
  one compound query, so the step keeps its joint timing and its `sync_group`.
  An edit with another backend re-plans the leg alone. That breaks the step's
  sync, and with lockstep on the untouched partner leg is refused at the start
  barrier.
