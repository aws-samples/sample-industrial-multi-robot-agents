"""
Deterministic BehaviourTree executor for the cell controller.

The tree format is **BehaviorTree.CPP v4** (https://www.behaviortree.dev): the
control nodes are ``Sequence`` / ``Parallel`` / ``Delay`` and the leaves are the
custom actions ``ExecuteTrajectory`` / ``MoveHome`` / ``WaitForRobot`` and the
condition ``CheckSafety``. Ports are read through ``btcpp.port()``, so a port
carrying a ``{key}`` blackboard reference resolves the same as an inline value.
The legacy JSON dialect (``EXECUTE``, ``PARALLEL``, …) still executes — the
handler table is keyed on the normalised kind — so a tree recorded before the
format change replays unchanged.

This is the runtime heart of the cell: it walks a BehaviourTree produced by the
trajectory agent and DISPATCHES precomputed trajectories to the robot agents in
the right order, enforcing timing/interlock gates and — critically — checking
two things continuously while robots move:

  1. **Collision** — the injected collision monitor's ``imminent()`` is polled
     every control tick; a non-None result aborts execution and (via the runtime)
     e-stops the cell.
  2. **Right-position-at-the-right-time** — for each EXECUTE node the executor
     interpolates the expected joint configuration from the node's trajectory at
     the elapsed (velocity-scaled) time and compares it to the robot's actual
     ``robot_status``. A sustained deviation aborts with a schedule-deviation
     failure.

There is NO planner and NO LLM here — planning happened offline, and recovery
reasoning happens ONE level up (the LLM cell agent inspects the structured
failure this returns). Execution itself must be deterministic and fast.

The executor talks to the world only through two duck-typed collaborators, so it
is fully unit-testable with fakes (no ROS / Pinocchio):

  runtime:  now() robots() latest_status(rid) dispatch(rid, task) estop(reason)
  monitor:  imminent() -> dict | None            (optional; may be None)

``execute(bt)`` returns::

    {"success": bool,
     "executed_nodes": int,
     "failure": {"failed_node_id", "node_type", "robot_ns"?, "reason_code",
                 "reason", "collision"?}   # present only when success is False
    }
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from agents.trajectory.tools.btcpp import (
    children as bt_children,
    node_kind,
    port as bt_port,
    resolve as bt_resolve,
    trajectory_of,
)

from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose

# Only the last resort, for a cell that declares no home of its own. The real
# value is resolved per robot from the cell config at dispatch time.
HOME_JOINTS = list(HOME_JOINTS_FALLBACK)


@dataclass
class ExecConfig:
    tick_hz: float = 20.0                 # control/monitor poll rate
    position_gate_tolerance_m: float = 0.05
    position_gate_timeout_s: float = 30.0
    schedule_deviation_rad: float = 0.7   # max |joint error| before it counts.
                                          # This is a schedule-ADHERENCE sanity
                                          # check (is the arm roughly following
                                          # its planned path?), not the collision
                                          # gate. The sim controller lags the
                                          # executor's wall-clock interpolation on
                                          # fast segments by ~0.4 rad, so a tight
                                          # 0.35 false-aborts correct motions;
                                          # 0.7 still catches gross divergence.
    schedule_deviation_grace_s: float = 1.5  # must persist this long to abort
    schedule_check_warmup_s: float = 3.0  # ignore deviation right after dispatch
                                          # (controller needs time to accept the
                                          # FJT goal and start moving)
    completion_margin_s: float = 1.0      # robots.yaml planning.completion_margin_s;
                                          # slack after trajectory end (the
                                          # trajectory's own TOPP-RA duration has
                                          # already elapsed and schedule adherence
                                          # is checked during motion, so this is
                                          # just settle slack — keep it small to
                                          # avoid idle "both arms waiting" gaps)
    home_time_s: float = 2.5              # robots.yaml planning.interlock_move_time_s
    op_gate_timeout_s: float = 900.0      # longest a coordinated arm waits for
                                          # another arm's operation to complete
    home_return_time_s: float = 7.0       # robots.yaml planning.home_return_time_s;
                                          # final return-to-HOME is slower/gentler
                                          # than intermediate interlock moves
    position_gate_feedback_grace_s: float = 3.0  # if no usable EE feedback within
                                                 # this window, degrade the gate to
                                                 # a settle wait instead of blocking
                                                 # (sim has no FK: ee_position=[0,0,0])
    cell_id: str = ""                     # which cell's robots.yaml declares HOME.
                                          # Normally taken from the tree being
                                          # executed; this is the override, and
                                          # $CELL_ID is the last resort.
    # ── lockstep (opt-in: LOCKSTEP=1 on the cell controller, docs/adr/0001) ──
    # Legs tagged with one ``sync_group`` (an ompl-compound step) were proven
    # collision-free AT EACH INSTANT on a shared clock. Off: they run as any
    # Parallel children do. On: they dispatch together off a barrier, and if any
    # arm's time on its trajectory drifts from the shared clock the whole cell is
    # e-stopped — nothing downstream can pause or re-time an arm, so a halt is
    # the only way to keep the plan's proof true.
    lockstep: bool = False
    lockstep_start_timeout_s: float = 30.0  # longest a leg waits for its partners
    lockstep_tolerance_s: float = 0.5       # max |lag| behind/ahead of the clock
    lockstep_grace_s: float = 0.5           # must persist this long to halt
    lockstep_warmup_s: float = 1.0          # ignore lag right after dispatch
    # ── position-confirmed motion (when the arm reports its joints) ──────────
    # A leg STARTS only where the arm stands: dispatching a trajectory from
    # elsewhere makes the controller jump to its first point, a motion nothing
    # validated. None disables the check.
    start_pose_tolerance_rad: "Optional[float]" = 0.1
    start_settle_s: float = 2.0             # the previous leg may still be settling
    # A leg is COMPLETE only when the arm is at its last point — not when the
    # clock says so. An arm that stalled, lagged or never moved used to be
    # reported complete, and the next node (or another arm's completion gate)
    # went on from a pose the arm was not in.
    arrival_tolerance_rad: float = 0.05
    arrival_timeout_s: float = 5.0          # past the leg's own end + margin
    monitor_ready_timeout_s: float = 2.0    # the monitor's first full check

    @classmethod
    def for_cell(cls, cell_id: str) -> "ExecConfig":
        """Defaults, with the move durations and the per-leaf completion margin
        taken from the cell's robots.yaml ``planning:`` block and lockstep from
        ``$LOCKSTEP``. Does NOT set ``cell_id`` — that stays the tree's."""
        from agents.trajectory.tools.robot_kinematics import (
            _load_planning_config, planning_setting,
        )
        lockstep = lockstep_from_env()
        try:
            plan = _load_planning_config(cell_id)
            margin = planning_setting(cell_id, "completion_margin_s")
        except Exception:  # noqa: BLE001 — the store is down: today's defaults
            return cls(lockstep=lockstep)
        return cls(home_time_s=plan["interlock_move_time_s"],
                   home_return_time_s=plan["home_return_time_s"],
                   completion_margin_s=margin,
                   lockstep=lockstep)


def lockstep_from_env(env=None) -> bool:
    """``LOCKSTEP`` as the cell controller was started with (make LOCKSTEP=1)."""
    value = (env if env is not None else os.environ).get("LOCKSTEP", "0")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def trajectory_lag_s(traj: list[dict], actual: list[float], clock_t: float,
                     back_s: float = 3.0, ahead_s: float = 1.5) -> Optional[float]:
    """How far behind (+) or ahead (-) of ``clock_t`` an arm is on ``traj``.

    The arm's time on its trajectory is the sample its measured joints match
    best, searched in a window around the clock. A hold (several samples at one
    pose) matches equally well everywhere, so ties go to the sample nearest the
    clock — an arm holding still where the plan holds still is on time.
    """
    if not traj or not actual:
        return None
    best_t, best_cost = None, float("inf")
    for p in traj:
        ts = float(p["time_s"])
        if ts < clock_t - back_s or ts > clock_t + ahead_s:
            continue
        cost = _max_joint_error(p["positions"], actual) + 0.01 * abs(ts - clock_t)
        if cost < best_cost:
            best_t, best_cost = ts, cost
    if best_t is None:
        return None
    return clock_t - best_t


class _SyncGroup:
    """The start barrier of one compound step's legs."""

    def __init__(self, members):
        self.members = set(members)
        self.arrived: dict = {}           # robot -> velocity_scale
        self.start_t: Optional[float] = None
        self.cv = threading.Condition()


def interpolate_trajectory(traj: list[dict], t: float) -> Optional[list[float]]:
    """Expected joint configuration at trajectory time ``t`` (linear)."""
    if not traj:
        return None
    if t <= float(traj[0]["time_s"]):
        return list(traj[0]["positions"])
    if t >= float(traj[-1]["time_s"]):
        return list(traj[-1]["positions"])
    for k in range(len(traj) - 1):
        t0 = float(traj[k]["time_s"])
        t1 = float(traj[k + 1]["time_s"])
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            p0 = traj[k]["positions"]
            p1 = traj[k + 1]["positions"]
            n = min(len(p0), len(p1))
            return [float(p0[i]) + a * (float(p1[i]) - float(p0[i])) for i in range(n)]
    return list(traj[-1]["positions"])


_TWO_PI = 2.0 * math.pi


def _angular_diff(x: float, y: float) -> float:
    """Smallest absolute difference between two joint angles, treating values
    that differ by a full turn as the SAME physical configuration. Weld poses
    put wrist joints near +/-pi, and the sim controller may resolve that to
    either sign; without wrapping, +pi vs -pi reads as a spurious 2*pi (6.283
    rad) deviation and false-aborts an otherwise correct motion."""
    d = abs(float(x) - float(y)) % _TWO_PI
    return min(d, _TWO_PI - d)


def _max_joint_error(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return max(_angular_diff(a[i], b[i]) for i in range(n))


class BehaviourTreeExecutor:
    def __init__(self, runtime, monitor=None, cfg: Optional[ExecConfig] = None):
        self._rt = runtime
        self._mon = monitor
        self._cfg = cfg or ExecConfig()
        self._abort = threading.Event()          # set when the whole run must stop
        self._abort_failure: Optional[dict] = None
        self._executed = 0
        self._tick = 1.0 / max(1.0, self._cfg.tick_hz)
        self._bb: dict = {}                      # BT.CPP blackboard for {key} ports
        # Operations that have COMPLETED in this run, as (robot_id, operation_id).
        # The coordinated planner orders conflicting arms by these events
        # (WaitForRobot depends_on_ops): an event, unlike a clock time, holds
        # whatever speed the arms actually run at.
        self._completed: set = set()
        self._completed_cv = threading.Condition()
        self._cell_id = self._cfg.cell_id or os.environ.get("CELL_ID", "")
        self._sync: dict[str, _SyncGroup] = {}   # lockstep barriers, per run
        self._sync_lock = threading.Lock()

    # ── public entry point ──────────────────────────────────────────────────────
    def execute(self, behaviour_tree: dict) -> dict:
        self._abort.clear()
        self._abort_failure = None
        self._executed = 0
        with self._completed_cv:
            self._completed = set()
        with self._sync_lock:
            self._sync = {}
        # Load the blackboard before the first tick: the trajectory arrays live
        # there and the XML/JSON ports only reference them by {key}.
        self._bb = dict(behaviour_tree.get("blackboard") or {})
        # The tree names the cell it was planned for. Trust that over the
        # environment: a blind HOME must aim at the pose THIS plan was validated
        # against, not at whatever cell the container happens to be configured for.
        self._cell_id = (behaviour_tree.get("cell_id")
                         or self._cfg.cell_id
                         or os.environ.get("CELL_ID", ""))

        root = behaviour_tree.get("root")
        if not isinstance(root, dict):
            return {"success": False, "executed_nodes": 0, "failure": {
                "node_type": "ROOT", "reason_code": "invalid_tree",
                "reason": "behaviour_tree has no 'root' node",
            }}

        # Nothing moves unless the collision monitor is watching (fail-closed).
        why = self._monitor_health(wait=True)
        if why:
            return {"success": False, "executed_nodes": 0, "failure": {
                "node_type": "ROOT", "reason_code": "monitor_fault",
                "reason": (f"collision monitor not watching the cell: {why} — nothing "
                           "was dispatched"),
            }}

        ok, failure = self._exec(root)
        result = {"success": bool(ok), "executed_nodes": self._executed}
        if not ok:
            result["failure"] = failure or self._abort_failure or {
                "reason_code": "unknown", "reason": "execution failed"
            }
        return result

    def abort(self, reason: str = "aborted by operator") -> bool:
        """Stop this run from OUTSIDE the executing thread. Commands nothing.

        The operator's Stop: no further node is dispatched and the run returns
        ``success: False`` with ``reason_code: "operator_abort"``. The trajectory
        already accepted by a controller RUNS TO ITS END — this is a clean end of
        activity, not a halt. What stops an arm mid-motion is the e-stop, which
        cancels the active FollowJointTrajectory goal; that is a different button
        for a different question, and it latches.

        Returns False if the run had already stopped for another reason (a
        collision, a node failure), because that first failure is the one worth
        reporting and ``_raise_abort`` deliberately keeps it.
        """
        already = self._abort.is_set()
        self._raise_abort({
            "node_type": "RUN", "reason_code": "operator_abort", "reason": reason,
        })
        return not already

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # ── collision / abort helpers ────────────────────────────────────────────────
    def _monitor_health(self, wait: bool = False) -> Optional[str]:
        """Why the monitor is not watching, or None. A monitor without a health
        report (a test double) is taken at its word; no monitor at all is the
        caller's choice (the cell agent always passes one)."""
        if self._mon is None:
            return None
        try:
            if wait and callable(getattr(self._mon, "wait_ready", None)):
                return self._mon.wait_ready(self._cfg.monitor_ready_timeout_s)
            health = getattr(self._mon, "health", None)
            return health() if callable(health) else None
        except Exception as exc:  # noqa: BLE001 — a monitor that broke is not watching
            return f"health check raised {type(exc).__name__}: {exc}"

    def _collision(self) -> Optional[dict]:
        """The monitor's imminent collision — or a ``monitor_fault`` when it cannot
        say: a monitor that raised, stopped or went stale has not said "clear"."""
        if self._mon is None:
            return None
        try:
            hit = self._mon.imminent()
        except Exception as exc:  # noqa: BLE001
            hit = {"kind": "monitor_fault",
                   "reason": f"collision monitor raised {type(exc).__name__}: {exc}"}
        if hit is not None:
            return hit
        why = self._monitor_health()
        if why:
            return {"kind": "monitor_fault",
                    "reason": f"collision monitor not watching the cell: {why}"}
        return None

    def _joints(self, robot_id) -> Optional[list]:
        status = self._rt.latest_status(robot_id)
        q = (status or {}).get("joint_positions")
        return list(q) if q else None

    def _await_pose(self, robot_id, target, tol: float, timeout_s: float, node,
                    reason_code: str, what: str) -> "tuple[bool, Optional[dict]]":
        """Wait until ``robot_id`` reports joints within ``tol`` of ``target``.

        No joint feedback at all → True (nothing to confirm with; the monitor's
        health already refuses a cell whose arms it cannot see). Collisions and
        aborts end the wait like any gate."""
        deadline = self._rt.now() + timeout_s
        err = None
        while True:
            if self._abort.is_set():
                return False, self._abort_failure
            col = self._collision()
            if col is not None:
                self._estop(col.get("reason", "imminent collision"))
                fail = _collision_failure(col, robot_id)
                self._raise_abort(fail)
                return False, fail
            actual = self._joints(robot_id)
            if actual is None:
                return True, None
            err = _max_joint_error(target, actual)
            if err <= tol:
                return True, None
            if self._rt.now() >= deadline:
                return False, {
                    "failed_node_id": node.get("id"), "node_type": node.get("type"),
                    "robot_ns": robot_id, "reason_code": reason_code,
                    "max_joint_error_rad": round(err, 4), "tolerance_rad": tol,
                    "reason": (f"{robot_id} is {err:.3f} rad from {what} "
                               f"(tolerance {tol} rad, waited {timeout_s:.1f} s)"),
                }
            time.sleep(self._tick)

    def _raise_abort(self, failure: dict) -> None:
        """Record the first failure and signal all siblings/threads to stop."""
        if self._abort_failure is None:
            self._abort_failure = failure
        self._abort.set()

    def _estop(self, reason: str) -> None:
        try:
            self._rt.estop(reason)
        except Exception:
            pass

    def _interruptible_sleep(self, duration_s: float) -> Optional[dict]:
        """Sleep in ticks, returning a collision/abort failure early if raised."""
        deadline = self._rt.now() + max(0.0, duration_s)
        while self._rt.now() < deadline:
            if self._abort.is_set():
                return self._abort_failure or {"reason_code": "aborted", "reason": "run aborted"}
            col = self._collision()
            if col is not None:
                self._estop(col.get("reason", "imminent collision"))
                fail = _collision_failure(col)
                self._raise_abort(fail)
                return fail
            time.sleep(min(self._tick, max(0.0, deadline - self._rt.now())))
        return None

    # ── node dispatch ─────────────────────────────────────────────────────────────
    def _exec(self, node: dict) -> "tuple[bool, Optional[dict]]":
        if self._abort.is_set():
            return False, self._abort_failure
        # node_kind normalises a legacy type ("EXECUTE") to its BT.CPP
        # registration ID ("ExecuteTrajectory"), so one table serves both dialects.
        kind = node_kind(node)
        handler = {
            "Sequence":          self._exec_sequence,
            "SequenceWithMemory": self._exec_sequence,
            "Parallel":          self._exec_parallel,
            "Delay":             self._exec_delay,
            "ExecuteTrajectory": self._exec_execute,
            "WaitForRobot":      self._exec_interlock_wait,
            "CheckSafety":       self._exec_safety_gate,
            "MoveHome":          self._exec_home,
        }.get(kind)
        if handler is None:
            return False, {
                "failed_node_id": node.get("id"), "node_type": kind or node.get("type"),
                "reason_code": "unknown_node",
                "reason": (f"unregistered node '{kind or node.get('type')}' — the "
                           f"executor implements Sequence, Parallel, Delay, "
                           f"ExecuteTrajectory, MoveHome, WaitForRobot, CheckSafety"),
            }
        return handler(node)

    def _exec_sequence(self, node: dict) -> "tuple[bool, Optional[dict]]":
        for child in bt_children(node):
            ok, failure = self._exec(child)
            if not ok:
                return False, failure
        return True, None

    def _exec_delay(self, node: dict) -> "tuple[bool, Optional[dict]]":
        """BT.CPP ``Delay``: wait ``delay_msec``, then tick the single child.

        This is where a staggered start lives now — expressing it as a decorator
        rather than a leaf field keeps the timing visible in the tree itself.
        """
        delay_s = float(bt_port(node, "delay_msec", 0) or 0) / 1000.0
        if delay_s > 0:
            fail = self._interruptible_sleep(delay_s)
            if fail:
                return False, fail
        for child in bt_children(node):
            ok, failure = self._exec(child)
            if not ok:
                return False, failure
        return True, None

    def _exec_parallel(self, node: dict) -> "tuple[bool, Optional[dict]]":
        children = bt_children(node)
        if not children:
            return True, None

        results: dict[int, tuple[bool, Optional[dict]]] = {}
        threads: list[threading.Thread] = []

        def run(idx: int, child: dict):
            results[idx] = self._exec(child)

        for i, child in enumerate(children):
            t = threading.Thread(target=run, args=(i, child), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        for i in range(len(children)):
            ok, failure = results.get(i, (False, {"reason_code": "no_result",
                                                  "reason": "parallel child produced no result"}))
            if not ok:
                return False, failure
        return True, None

    def _exec_execute(self, node: dict) -> "tuple[bool, Optional[dict]]":
        robot_id = bt_port(node, "robot_ns")
        op_id    = bt_port(node, "operation_id")
        traj     = trajectory_of(node, self._bb)
        vscale   = float(bt_port(node, "velocity_scale", 1.0)) or 1.0
        hold_s   = float(bt_port(node, "hold_duration_s", 0.5))
        # start_delay_s is now the Delay decorator, but a legacy tree still
        # carries the field — honour it so recorded trees replay identically.
        delay_s  = float(bt_port(node, "start_delay_s", 0.0) or 0.0)

        if not traj:
            return False, {
                "failed_node_id": node.get("id"), "node_type": "ExecuteTrajectory",
                "robot_ns": robot_id, "reason_code": "empty_trajectory",
                "reason": (f"ExecuteTrajectory {op_id} has no trajectory — the "
                           f"'trajectory' port is empty or its blackboard entry is missing"),
            }

        # 1. Start delay (interruptible).
        if delay_s > 0:
            fail = self._interruptible_sleep(delay_s)
            if fail:
                return False, fail

        # 2. Optional interlock: move to the safe intermediate first, gate on it.
        itgt = bt_resolve(bt_port(node, "interlock_target"), self._bb)
        if bt_port(node, "has_interlock") and isinstance(itgt, dict):
            il_joints = itgt.get("joint_positions")
            il_ee     = itgt.get("ee_position")
            if il_joints:
                self._rt.dispatch(robot_id, {
                    "operation_id": f"{op_id}:interlock", "operation_type": "move",
                    "trajectory": [{"positions": il_joints, "time_s": self._cfg.home_time_s}],
                    "hold_duration_s": 0.0, "velocity_scale": 1.0,
                })
            if il_ee:
                ok, failure = self._wait_position(robot_id, il_ee, node.get("id"),
                                                  "ExecuteTrajectory")
                if not ok:
                    return False, failure

        # 3. The leg starts where the arm stands, or it is not sent.
        if self._cfg.start_pose_tolerance_rad is not None and len(traj) > 1:
            ok, failure = self._await_pose(
                robot_id, traj[0]["positions"], float(self._cfg.start_pose_tolerance_rad),
                self._cfg.start_settle_s, node, "start_pose_mismatch",
                f"the start of {op_id} — nothing was dispatched")
            if not ok:
                self._raise_abort(failure)
                return False, failure

        # 3b. Lockstep: a compound step's legs leave together, or not at all.
        group = traj[0].get("sync_group") if self._cfg.lockstep else None
        t_sync = None
        if group:
            ok, failure, t_sync = self._sync_barrier(
                group, traj[0].get("sync_members") or [robot_id], robot_id, vscale, node)
            if not ok:
                return False, failure

        # 4. Dispatch the main trajectory (robot agent applies velocity_scale).
        self._rt.dispatch(robot_id, {
            "operation_id": op_id,
            "operation_type": bt_port(node, "operation_type", "weld"),
            "trajectory": traj, "hold_duration_s": hold_s, "velocity_scale": vscale,
        })
        self._executed += 1

        # 5. Monitor collision + schedule adherence until completion.
        traj_end = float(traj[-1]["time_s"])
        wall_duration = traj_end / vscale + hold_s + self._cfg.completion_margin_s
        # One clock for the whole group: the barrier's release, not this thread's
        # own dispatch instant, so every member measures lag against the same t=0.
        t0 = t_sync if t_sync is not None else self._rt.now()
        deviating_since: Optional[float] = None
        lagging_since: Optional[float] = None

        while True:
            now = self._rt.now()
            elapsed = now - t0

            if self._abort.is_set():
                return False, self._abort_failure

            col = self._collision()
            if col is not None:
                self._estop(col.get("reason", "imminent collision"))
                fail = _collision_failure(col, robot_id)
                self._raise_abort(fail)
                return False, fail

            # Right-position-at-the-right-time check (only when we have feedback
            # AND the controller has had time to pick up the goal — otherwise the
            # dispatch→accept latency reads as a large deviation and false-aborts).
            expected = interpolate_trajectory(traj, min(elapsed * vscale, traj_end))
            status = self._rt.latest_status(robot_id)
            actual = (status or {}).get("joint_positions")
            if expected is not None and actual and elapsed >= self._cfg.schedule_check_warmup_s:
                err = _max_joint_error(expected, actual)
                if err > self._cfg.schedule_deviation_rad:
                    deviating_since = deviating_since or now
                    if now - deviating_since >= self._cfg.schedule_deviation_grace_s:
                        self._estop(f"{robot_id} off-schedule during {op_id}")
                        fail = {
                            "failed_node_id": node.get("id"),
                            "node_type": "ExecuteTrajectory",
                            "robot_ns": robot_id, "reason_code": "schedule_deviation",
                            "reason": (
                                f"{robot_id} deviated {err:.3f} rad from the planned "
                                f"position for op {op_id} at t={elapsed:.1f}s "
                                f"(limit {self._cfg.schedule_deviation_rad} rad)"
                            ),
                        }
                        self._raise_abort(fail)
                        return False, fail
                else:
                    deviating_since = None

            # Lockstep: on time against the group's clock, within tolerance.
            if (group and actual and elapsed >= self._cfg.lockstep_warmup_s
                    and elapsed * vscale <= traj_end):
                lag = trajectory_lag_s(traj, actual, elapsed * vscale)
                if lag is not None and abs(lag) > self._cfg.lockstep_tolerance_s:
                    lagging_since = lagging_since or now
                    if now - lagging_since >= self._cfg.lockstep_grace_s:
                        # E-stop, not a software stop: the partner arms are still
                        # moving on a plan that is only safe in step with this one.
                        self._estop(f"lockstep lost: {robot_id} during {op_id}")
                        fail = {
                            "failed_node_id": node.get("id"),
                            "node_type": "ExecuteTrajectory",
                            "robot_ns": robot_id, "reason_code": "lockstep_lost",
                            "sync_group": group,
                            "reason": (
                                f"{robot_id} is {lag:+.2f} s off the shared clock of "
                                f"compound step {group} during op {op_id} (tolerance "
                                f"{self._cfg.lockstep_tolerance_s} s) — every arm halted"
                            ),
                        }
                        self._raise_abort(fail)
                        return False, fail
                else:
                    lagging_since = None

            if elapsed >= wall_duration:
                # Complete when the arm IS at the end of its leg, not when the
                # clock says so.
                ok, failure = self._await_pose(
                    robot_id, traj[-1]["positions"], self._cfg.arrival_tolerance_rad,
                    self._cfg.arrival_timeout_s, node, "arrival_timeout",
                    f"the end of {op_id}")
                if not ok:
                    self._raise_abort(failure)
                    return False, failure
                self._mark_completed(robot_id, op_id)
                return True, None
            time.sleep(self._tick)

    def _sync_barrier(self, group_id: str, members, robot_id, vscale: float,
                      node: dict) -> "tuple[bool, Optional[dict], Optional[float]]":
        """Wait until every member of ``group_id`` is ready, then release all at once.

        Returns ``(ok, failure, release_time)``. Fails — without dispatching — when
        a partner does not arrive in time (the tree does not run the legs side by
        side, e.g. put them in a Sequence), when the members would run at
        different ``velocity_scale`` (their clocks would diverge by construction),
        or when the run aborts while waiting.
        """
        with self._sync_lock:
            grp = self._sync.setdefault(group_id, _SyncGroup(members))
        base = {"failed_node_id": node.get("id"), "node_type": "ExecuteTrajectory",
                "robot_ns": robot_id, "sync_group": group_id}
        with grp.cv:
            grp.arrived[str(robot_id)] = vscale
            if len(set(grp.arrived.values())) > 1:
                fail = {**base, "reason_code": "lockstep_velocity_mismatch",
                        "reason": (f"compound step {group_id} members run at different "
                                   f"velocity_scale {grp.arrived} — they cannot stay in step")}
                self._raise_abort(fail)
                grp.cv.notify_all()
                return False, fail, None
            if grp.members <= set(grp.arrived):
                grp.start_t = self._rt.now()
                grp.cv.notify_all()
            deadline = self._rt.now() + self._cfg.lockstep_start_timeout_s
            while grp.start_t is None:
                if self._abort.is_set():
                    return False, self._abort_failure, None
                if self._rt.now() >= deadline:
                    missing = sorted(grp.members - set(grp.arrived))
                    fail = {**base, "reason_code": "lockstep_partner_missing",
                            "reason": (f"{robot_id} waited {self._cfg.lockstep_start_timeout_s} s "
                                       f"for {missing} to start compound step {group_id} "
                                       "with it — the tree does not run these legs side by "
                                       "side; nothing was dispatched")}
                    self._raise_abort(fail)
                    grp.cv.notify_all()
                    return False, fail, None
                grp.cv.wait(timeout=self._tick)
            return True, None, grp.start_t

    def _mark_completed(self, robot_id, op_id) -> None:
        with self._completed_cv:
            self._completed.add((str(robot_id), str(op_id)))
            self._completed_cv.notify_all()

    def _wait_ops(self, deps: list, node_id) -> "tuple[bool, Optional[dict]]":
        """Block until every ``{"robot", "operation_id"}`` in ``deps`` has
        COMPLETED in this run. Aborts and collisions end the wait like any gate."""
        need = {(str(d.get("robot")), str(d.get("operation_id"))) for d in deps
                if d.get("robot") and d.get("operation_id")}
        deadline = self._rt.now() + self._cfg.op_gate_timeout_s
        while True:
            if self._abort.is_set():
                return False, self._abort_failure
            col = self._collision()
            if col is not None:
                self._estop(col.get("reason", "imminent collision"))
                fail = _collision_failure(col, None)
                self._raise_abort(fail)
                return False, fail
            with self._completed_cv:
                if need <= self._completed:
                    return True, None
                self._completed_cv.wait(timeout=self._tick)
            if self._rt.now() >= deadline:
                missing = sorted(f"{r}:{o}" for r, o in need - self._completed)
                return False, {
                    "failed_node_id": node_id, "node_type": "WaitForRobot",
                    "reason_code": "gate_timeout",
                    "reason": (f"waited {self._cfg.op_gate_timeout_s:.0f}s for "
                               f"{', '.join(missing)} to complete"),
                }

    def _exec_interlock_wait(self, node: dict) -> "tuple[bool, Optional[dict]]":
        # Completion dependencies (coordinated plans): wait until other arms have
        # COMPLETED specific operations. Order by event — timing-independent.
        ops = bt_resolve(bt_port(node, "depends_on_ops"), self._bb) or []
        if ops:
            ok, failure = self._wait_ops(ops, node.get("id"))
            if not ok:
                return False, failure
        # Joint-space dependencies (preferred): wait until another robot reaches
        # a target joint configuration. Uses robot_status.joint_positions, which
        # is real feedback (unlike the stubbed ee_position), so these gates work
        # in this cell. Emitted by the trajectory agent's space-time coordinator
        # to serialize arms that cannot share the centre at the same time.
        for dep in (bt_resolve(bt_port(node, "depends_on_joints"), self._bb) or []):
            rid = dep.get("robot")
            joints = dep.get("joint_positions")
            tol = float(dep.get("tolerance_rad", 0.15))
            if rid and joints:
                ok, failure = self._wait_joints(rid, joints, node.get("id"), tol)
                if not ok:
                    return False, failure
        # Cartesian dependencies (legacy): gate on ee_position.
        for dep in (bt_resolve(bt_port(node, "depends_on_positions"), self._bb) or []):
            rid = dep.get("robot")
            pos = dep.get("position")
            tol = float(dep.get("tolerance_m", self._cfg.position_gate_tolerance_m))
            if rid and pos:
                ok, failure = self._wait_position(rid, pos, node.get("id"),
                                                  "WaitForRobot", tol)
                if not ok:
                    return False, failure
        return True, None

    def _wait_joints(self, robot_id, expected_joints, node_id,
                     tol: float = 0.15) -> "tuple[bool, Optional[dict]]":
        """Block until ``robot_id`` reaches ``expected_joints`` (max abs joint
        error <= tol). This is how one arm waits for another to clear a shared
        zone — a dependency/wait condition emitted into the tree."""
        exp = [float(v) for v in expected_joints]
        deadline = self._rt.now() + self._cfg.position_gate_timeout_s
        while self._rt.now() < deadline:
            if self._abort.is_set():
                return False, self._abort_failure
            col = self._collision()
            if col is not None:
                self._estop(col.get("reason", "imminent collision"))
                fail = _collision_failure(col, robot_id)
                self._raise_abort(fail)
                return False, fail
            status = self._rt.latest_status(robot_id)
            actual = (status or {}).get("joint_positions")
            if actual and _max_joint_error(exp, actual) <= tol:
                return True, None
            time.sleep(self._tick)
        return False, {
            "failed_node_id": node_id, "node_type": "WaitForRobot", "robot_ns": robot_id,
            "reason_code": "interlock_timeout",
            "reason": (f"{robot_id} did not reach the awaited joint configuration "
                       f"within {self._cfg.position_gate_timeout_s}s"),
        }

    def _exec_safety_gate(self, node: dict) -> "tuple[bool, Optional[dict]]":
        condition = bt_port(node, "condition", "estop_clear")
        col = self._collision()
        if condition == "estop_clear" and col is not None:
            fail = _collision_failure(col)
            fail.update({"failed_node_id": node.get("id"), "node_type": "CheckSafety"})
            self._raise_abort(fail)
            return False, fail
        return True, None

    def _exec_home(self, node: dict) -> "tuple[bool, Optional[dict]]":
        robot_id = bt_port(node, "robot_ns")
        # Prefer a PLANNED, collision-free return-to-HOME trajectory (attached by
        # the trajectory agent's re-chain). Fall back to a slow single-waypoint
        # home command when no planned path is present.
        traj = trajectory_of(node, self._bb)
        if traj:
            wait_s = float(traj[-1]["time_s"]) + self._cfg.completion_margin_s
            self._rt.dispatch(robot_id, {
                "operation_id": "home", "operation_type": "move",
                "trajectory": traj, "hold_duration_s": 0.0, "velocity_scale": 1.0,
            })
        else:
            t_home = self._cfg.home_return_time_s   # deliberately slow/gentle
            wait_s = t_home + self._cfg.completion_margin_s
            # This is the UNPLANNED home — a single waypoint the executor drives
            # blind. It must at least aim at the pose the cell declares and the
            # planner validates against, or the fallback disagrees with every
            # planned return by 36 deg at the elbow.
            blind_home = home_pose(self._cell_id, robot_id, HOME_JOINTS)
            self._rt.dispatch(robot_id, {
                "operation_id": "home", "operation_type": "move",
                "trajectory": [{"positions": list(blind_home), "time_s": t_home}],
                "hold_duration_s": 0.0, "velocity_scale": 1.0,
            })
        self._executed += 1
        fail = self._interruptible_sleep(wait_s)
        if fail:
            return False, fail
        # HOME is reached when the arm reports it, not when the timer runs out.
        goal = traj[-1]["positions"] if traj else list(blind_home)
        ok, failure = self._await_pose(robot_id, goal, self._cfg.arrival_tolerance_rad,
                                       self._cfg.arrival_timeout_s, node,
                                       "arrival_timeout", "HOME")
        if not ok:
            self._raise_abort(failure)
            return False, failure
        return True, None

    # ── position gate ─────────────────────────────────────────────────────────────
    def _wait_position(self, robot_id, expected_xyz, node_id, node_type,
                       tol: Optional[float] = None) -> "tuple[bool, Optional[dict]]":
        import math
        tol = float(tol if tol is not None else self._cfg.position_gate_tolerance_m)
        deadline = self._rt.now() + self._cfg.position_gate_timeout_s
        feedback_deadline = self._rt.now() + self._cfg.position_gate_feedback_grace_s
        exp = [float(v) for v in expected_xyz[:3]]
        saw_feedback = False

        while self._rt.now() < deadline:
            if self._abort.is_set():
                return False, self._abort_failure
            col = self._collision()
            if col is not None:
                self._estop(col.get("reason", "imminent collision"))
                fail = _collision_failure(col, robot_id)
                self._raise_abort(fail)
                return False, fail
            status = self._rt.latest_status(robot_id)
            ee = (status or {}).get("ee_position")
            # Usable EE feedback = present and not the all-zero stub the sim
            # publishes when forward kinematics is unavailable (no Pinocchio).
            if ee and len(ee) >= 3 and any(abs(float(v)) > 1e-6 for v in ee[:3]):
                saw_feedback = True
                d = math.sqrt(sum((float(ee[i]) - exp[i]) ** 2 for i in range(3)))
                if d <= tol:
                    return True, None
            elif not saw_feedback and self._rt.now() >= feedback_deadline:
                # No usable EE feedback in this cell → we cannot gate on Cartesian
                # position. Degrade to a brief settle wait and proceed rather than
                # block for the full timeout and fail the whole run.
                fail = self._interruptible_sleep(min(self._cfg.home_time_s, 1.5))
                return (False, fail) if fail else (True, None)
            time.sleep(self._tick)

        return False, {
            "failed_node_id": node_id, "node_type": node_type, "robot_ns": robot_id,
            "reason_code": "position_gate_timeout",
            "reason": (
                f"{robot_id} did not reach {exp} within {tol} m "
                f"before the {self._cfg.position_gate_timeout_s}s gate timeout"
            ),
        }


def _collision_failure(col: dict, robot_ns: Optional[str] = None) -> dict:
    """Normalise a collision-monitor detail into the executor failure shape."""
    return {
        "node_type": "ExecuteTrajectory",
        "robot_ns": robot_ns or col.get("robot_ns") or col.get("robot_a"),
        "reason_code": "monitor_fault" if col.get("kind") == "monitor_fault" else "collision",
        "reason": col.get("reason", "imminent collision detected"),
        "collision": col,
    }
