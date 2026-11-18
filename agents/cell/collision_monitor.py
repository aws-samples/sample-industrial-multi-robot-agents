"""
Runtime collision monitor for the cell controller.

While the deterministic BehaviourTree executor drives robots through their
PRECOMPUTED (already collision-checked) trajectories, this monitor is the
independent safety net: it watches the robots' ACTUAL positions and fires an
e-stop the moment the live sphere-model clearance drops below the warning
threshold — e.g. a robot drifting off schedule, a mis-execution, or an operator
poking the scene.

It deliberately reuses the SAME sphere-model collision math as the planner
(``robot_kinematics._gate_points`` / ``_check_robot_pair_collision`` /
``_config_collision_info``) so "planned safe" and "monitored safe" mean the same
thing. ``robot_kinematics`` is the single owner: neither this file nor a planner
may define a second copy, which is exactly how the two drifted apart before.

Thresholds come from the cell's ``robots.yaml`` ``collision`` block:
  warning_distance  — clearance (m) at/below which a hit is "imminent" → e-stop
  min_clearance     — reported but the warning band is what triggers

Design:
  * One background thread polls the runtime's cached ``robot_status`` at
    ``rate_hz`` and recomputes clearances.
  * ``imminent()`` returns the current collision detail dict (or None). The
    executor polls this each control tick and aborts on a non-None result.
  * On the transition into an imminent state the monitor calls
    ``runtime.estop(...)`` and publishes a JSON alert on
    ``/{cell}/collision_alerts_json`` for the dashboard.

If Pinocchio is unavailable the monitor degrades to disabled (``imminent()``
always None) with a logged warning — the executor still runs, just without the
live guard.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import numpy as np

# Import-safe (numpy only at its module scope; Pinocchio stays lazy) — see that
# module's docstring. The tick rate lives there, not here, so planning-time
# validation and this monitor cannot disagree about how finely a cell is watched.
from agents.trajectory.tools.robot_kinematics import MONITOR_RATE_HZ


class CollisionMonitor:
    def __init__(self, cell_id: str, runtime, rate_hz: "Optional[float]" = None):
        self._cell_id = cell_id
        self._runtime = runtime
        # None == "the shared rate". A caller may tick FASTER than planning-time
        # validation assumed; it must not silently tick slower than the planner
        # was told, so the shared constant is the default, not a local literal.
        self._rate_hz = max(1.0, float(MONITOR_RATE_HZ if rate_hz is None else rate_hz))

        self._lock = threading.Lock()
        self._imminent: Optional[dict] = None
        self._min_clearance: float = float("inf")

        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._enabled = False
        self._alert_pub = None

        # Per-robot kinematics, loaded lazily on start().
        self._models: dict[str, tuple] = {}     # robot_id -> (model, data)
        self._bases: dict[str, np.ndarray] = {}
        self._yaws: dict[str, float] = {}
        self._tool_cfgs: dict[str, dict] = {}   # robot_id -> tool_collision cfg
        self._static_obstacles: list[dict] = []
        self._sphere_radius = 0.12
        # Per-sphere radii (arm steel vs the welding torch), built by the SAME
        # helper the planners use — see _load_models.
        self._sphere_radii: dict[str, Optional[np.ndarray]] = {}
        self._vel_limits: dict[str, np.ndarray] = {}
        self._speed_margin_full = 0.0
        # Last accepted (timestamp, q) per robot, for the speed finite difference,
        # and the resulting speed scale. Seeded at FULL speed: until we have two
        # samples we assume the arm is moving as fast as it can, which demands the
        # WIDEST clearance. An unmeasured speed must never buy a tighter gate.
        self._last_sample: dict[str, tuple] = {}
        # Last (timestamp, sphere centres) per robot and sphere set, for each
        # sphere's own speed — what its stopping distance is computed from.
        self._last_points: dict[tuple, tuple] = {}
        self._speed_scale: dict[str, float] = {}
        self._warning_distance = 0.20
        self._min_clearance_cfg = 0.05
        # Health (see health()): why the monitor is not watching, when the last
        # full check completed, and which arms it could not see.
        self._disabled_reason: Optional[str] = None
        self._fault: Optional[str] = None
        self._last_ok: Optional[float] = None
        self._started_at: Optional[float] = None
        self._blind: list = []
        # A cell without arm kinematics (a bare sim) may run unwatched only when
        # its operator says so: COLLISION_MONITOR_REQUIRED=0.
        self._required = os.environ.get("COLLISION_MONITOR_REQUIRED", "1").strip().lower() \
            not in ("0", "false", "no", "off")

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> "CollisionMonitor":
        try:
            self._load_models()
        except Exception as exc:  # noqa: BLE001 - degrade to disabled, log why
            print(f"[CollisionMonitor] disabled — could not load kinematics: {exc}", flush=True)
            self._enabled = False
            self._disabled_reason = f"could not load kinematics: {exc}"
            return self

        self._enabled = True
        self._stop = False
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="collision-monitor")
        self._thread.start()
        print(
            f"[CollisionMonitor] running @ {self._rate_hz:.0f} Hz — "
            f"warning={self._warning_distance} m, {len(self._models)} robot(s)",
            flush=True,
        )
        return self

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── config / kinematics loading ────────────────────────────────────────────
    def _load_models(self) -> None:
        import os
        import yaml
        from agents.trajectory.tools.robot_kinematics import (
            _cell_sphere_radii, _load_clearance_policy, _load_pinocchio_model,
            _load_robot_config,
        )

        from utils import cell_layout, cell_store

        # Through the cell's owner, like the planner: on a cell controller that runs
        # on another machine (CELL_DATA_MODE=orchestrator) a local copy could be
        # stale, and a supervisor checking against different numbers than the plan
        # was validated with is the drift this module exists to prevent.
        cfg = cell_store.read_yaml(self._cell_id, cell_store.KIND_CONFIG,
                                   cell_layout.robots_yaml(self._cell_id).name) or {}
        coll = cfg.get("collision", {}) or {}
        self._warning_distance = float(coll.get("warning_distance", 0.20))
        self._min_clearance_cfg = float(coll.get("min_clearance", 0.05))

        obs = cell_store.read_yaml(self._cell_id, cell_store.KIND_CONFIG,
                                   cell_layout.obstacles_yaml(self._cell_id).name)
        if obs is not None:
            # Read as the planner reads them (is_workpiece = tool-only), so the
            # monitor and the planner judge the same configuration the same way.
            from agents.trajectory.tools.robot_kinematics import normalize_static_obstacles
            self._static_obstacles = normalize_static_obstacles(
                (obs or {}).get("obstacles", []) or [])

        # The cell's clearance policy, read through the same owner the planners
        # read. This is the whole point of the module: a configuration that the
        # planner called collision-free must not be a collision here, so neither
        # side may hold its own copy of the radii.
        policy = _load_clearance_policy(self._cell_id)
        self._sphere_radius = policy["normal"]
        self._speed_margin_full = policy["speed_margin_full_speed_m"]

        for _ns, rid in self._runtime.robots():
            model, _data = _load_pinocchio_model(self._cell_id, rid)
            rcfg = _load_robot_config(self._cell_id, rid)
            self._models[rid] = (model, model.createData())
            self._bases[rid] = np.array(rcfg["base_position"])
            self._yaws[rid] = float(rcfg.get("base_orientation_yaw", 0.0))
            self._tool_cfgs[rid] = rcfg.get("tool_collision")
            self._sphere_radii[rid] = _cell_sphere_radii(
                self._cell_id, model, self._tool_cfgs[rid])
            self._vel_limits[rid] = np.array(
                rcfg.get("joint_velocity_limits") or [], dtype=float)
            self._speed_scale[rid] = 1.0

    # Oldest usable finite difference. Beyond this the two samples say nothing
    # about the arm's speed NOW (it could have accelerated and stopped between
    # them), so the estimate is discarded rather than trusted.
    _MAX_SAMPLE_AGE_S = 0.5

    def _speed_scale_for(self, rid: str, qn: np.ndarray, status: dict) -> float:
        """Fraction of full commanded joint speed this arm is currently moving at.

        The planner is TOLD its ``velocity_scale``; the monitor has to measure it.
        ``robot_status`` publishes positions and a timestamp but no velocities, so
        this finite-differences consecutive published samples and normalises each
        joint by its URDF velocity limit — yielding exactly the quantity the
        planner passes to :func:`_speed_margin`, so both sides feed one formula.

        Fail-closed on every path where the measurement is not trustworthy: no
        limits declared, no previous sample, a stale pair, or a non-advancing
        timestamp all leave the last good estimate in place (seeded at 1.0 = full
        speed). Guessing "stopped" would license a tighter gate than the arm has
        earned; guessing "flat out" only ever asks for more room.
        """
        limits = self._vel_limits.get(rid)
        if limits is None or limits.size < qn.shape[0] or not np.all(limits > 0):
            return 1.0

        ts = status.get("timestamp")
        if not isinstance(ts, (int, float)):
            return self._speed_scale.get(rid, 1.0)

        prev = self._last_sample.get(rid)
        dt = 0.0 if prev is None else float(ts) - prev[0]
        if dt <= 0.0:
            # No previous sample, or the same one polled twice (the monitor runs
            # faster than the status publisher). Keep the older sample so the next
            # fresh reading still has something to difference against.
            if prev is None:
                self._last_sample[rid] = (float(ts), qn.copy())
            return self._speed_scale.get(rid, 1.0)

        self._last_sample[rid] = (float(ts), qn.copy())
        if dt > self._MAX_SAMPLE_AGE_S:
            return self._speed_scale.get(rid, 1.0)

        scale = float(np.max(np.abs(qn - prev[1]) / (limits[: qn.shape[0]] * dt)))
        self._speed_scale[rid] = min(max(scale, 0.0), 1.0)
        return self._speed_scale[rid]

    def _sphere_stops(self, rid: str, which: str, P: np.ndarray, status: dict) -> Optional[np.ndarray]:
        """Each sphere's stopping distance at its current speed (robots.yaml
        ``stopping``), or None when the robot declares no stopping data.

        The speed is measured — consecutive published samples, finite-differenced
        per sphere. Where the measurement is not trustworthy (no previous sample,
        a stale pair, no timestamp) it is the joint-limit upper bound instead
        (:func:`robot_kinematics.sphere_speed_bound`; FK must be at ``P``)."""
        from agents.trajectory.tools.robot_kinematics import (
            accounts_for_speed, sphere_speed_bound, stopping_distance,
        )
        model, data = self._models[rid]
        if not accounts_for_speed(model):
            return None
        ts = status.get("timestamp")
        prev = self._last_points.get((rid, which))
        speeds = None
        if isinstance(ts, (int, float)):
            if prev is not None and prev[1].shape == P.shape:
                dt = float(ts) - prev[0]
                if 0.0 < dt <= self._MAX_SAMPLE_AGE_S:
                    speeds = np.linalg.norm(P - prev[1], axis=1) / dt
                elif dt <= 0.0:
                    return prev[2]            # same sample polled again
            self._last_points[(rid, which)] = (float(ts), P.copy(), None)
        if speeds is None:
            speeds = sphere_speed_bound(model, data, self._bases[rid], self._yaws[rid], P,
                                        self._vel_limits.get(rid, np.zeros(0)))
        stops = stopping_distance(model, speeds)
        if isinstance(ts, (int, float)):
            self._last_points[(rid, which)] = (float(ts), P.copy(), stops)
        return stops

    def _ensure_alert_pub(self):
        if self._alert_pub is not None:
            return self._alert_pub
        try:
            node = getattr(self._runtime, "_node", None)
            String = getattr(self._runtime, "_String", None)
            if node is None or String is None:
                return None
            self._alert_pub = node.create_publisher(
                String, f"/{self._cell_id}/collision_alerts_json", 5
            )
        except Exception:
            self._alert_pub = None
        return self._alert_pub

    # ── monitor loop ────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        from agents.trajectory.tools.robot_kinematics import (
            _gate_points, _check_robot_pair_collision, _config_collision_info,
            _speed_margin_from,
        )
        period = 1.0 / self._rate_hz

        while not self._stop:
            t0 = time.monotonic()
            try:
                detail = self._check_once(
                    _gate_points, _check_robot_pair_collision, _config_collision_info,
                    _speed_margin_from,
                )
            except Exception as exc:  # noqa: BLE001 — a check that broke is not "clear"
                self._fault = f"check failed: {type(exc).__name__}: {exc}"
                detail = {"kind": "monitor_fault",
                          "reason": f"collision monitor {self._fault}"}
            else:
                self._fault = None
                self._last_ok = time.monotonic()

            with self._lock:
                was = self._imminent
                self._imminent = detail

            # Fire e-stop + alert on the transition into an imminent state.
            if detail is not None and was is None:
                reason = detail.get("reason", "imminent collision")
                try:
                    self._runtime.estop(f"collision monitor: {reason}")
                except Exception:
                    pass
                self._publish_alert(detail)
                print(f"[CollisionMonitor] E-STOP — {reason}", flush=True)

            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _check_once(self, _gate_points, _check_robot_pair_collision,
                    _config_collision_info, _speed_margin_from) -> Optional[dict]:
        from agents.trajectory.tools.robot_kinematics import _static_points
        # Gather current joint configs + sphere sets per robot: the same spheres
        # the planner's gate checks (skeleton, tool and the link envelope).
        spheres: dict[str, np.ndarray] = {}
        stops: dict[str, Optional[np.ndarray]] = {}
        blind: list = []
        for _ns, rid in self._runtime.robots():
            status = self._runtime.latest_status(rid)
            q = (status or {}).get("joint_positions")
            if not q or not _fresh(status):
                blind.append(rid)          # an arm the monitor cannot see
                continue
            model, data = self._models[rid]
            qn = np.array(q[: model.nq], dtype=float)
            if qn.shape[0] < model.nq:
                blind.append(rid)
                continue
            spheres[rid] = _gate_points(model, data, qn, self._bases[rid], self._yaws[rid],
                                        self._tool_cfgs.get(rid))
            stops[rid] = self._sphere_stops(rid, "gate", spheres[rid], status)
            static_pts = _static_points(model, data, qn, self._bases[rid], self._yaws[rid],
                                        self._sphere_radius, self._sphere_radii.get(rid))
            static_stops = self._sphere_stops(rid, "static", static_pts, status)

            # Static-obstacle penetration (drift into the fixture etc.), checked
            # against the cell's per-body radii plus the allowance for how fast
            # this arm is actually moving — the identical gate the planner applied
            # when it declared this configuration free.
            extra_margin = _speed_margin_from(
                self._speed_margin_full, self._speed_scale_for(rid, qn, status))
            info = _config_collision_info(
                model, data, qn, self._bases[rid], self._yaws[rid], [], self._static_obstacles,
                self._sphere_radius,
                sphere_radii=self._sphere_radii.get(rid),
                extra_margin=extra_margin,
                sphere_extra=static_stops,
            )
            if info is not None and info.get("type") == "self":
                # The arm against itself (tools/self_collision.py), checked by the
                # same predicate the planner applied.
                return {
                    "kind": "self",
                    "robot_ns": rid,
                    "object_id": info.get("object_id"),
                    "point": None,
                    "clearance_m": info.get("clearance_m"),
                    "reason": (f"{rid} self-collision: {info.get('object_id')} "
                               f"(signed distance {info.get('clearance_m')} m)"),
                }
            if info is not None:
                return {
                    "kind": "static",
                    "robot_ns": rid,
                    "object_id": info.get("object_id"),
                    "point": info.get("point"),
                    "clearance_m": info.get("clearance_m"),
                    "reason": (
                        f"{rid} link {info.get('link_index')} penetrates "
                        f"{info.get('type')} '{info.get('object_id')}' at {info.get('point')}"
                    ),
                }

        self._blind = blind

        # Inter-robot clearance across all pairs.
        rids = list(spheres.keys())
        global_min = float("inf")
        worst: Optional[dict] = None
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                a, b = rids[i], rids[j]
                # Deliberately UNIFORM, and deliberately the arm radius: unlike a
                # fixture, whose pose is exactly known, two moving arms each carry
                # tracking error and command latency, so the torch gets no
                # thinning credit against another robot.
                # Each sphere's stopping distance at its own speed, both arms.
                from agents.trajectory.tools.robot_kinematics import gate_groups
                _collision, clearance = _check_robot_pair_collision(
                    spheres[a], spheres[b], self._sphere_radius, stops.get(a), stops.get(b),
                    gate_groups(self._models[a][0], len(spheres[a])),
                    gate_groups(self._models[b][0], len(spheres[b])),
                )
                if clearance < global_min:
                    global_min = clearance
                    worst = {"robot_a": a, "robot_b": b, "clearance_m": round(float(clearance), 4)}

        with self._lock:
            self._min_clearance = global_min

        if worst is not None and global_min <= self._warning_distance:
            worst.update({
                "kind": "robot_pair",
                "reason": (
                    f"{worst['robot_a']} ↔ {worst['robot_b']} clearance "
                    f"{worst['clearance_m']} m ≤ warning {self._warning_distance} m"
                ),
            })
            return worst
        return None

    def _publish_alert(self, detail: dict) -> None:
        pub = self._ensure_alert_pub()
        if pub is None:
            return
        try:
            msg = self._runtime._String()
            msg.data = json.dumps({
                "collision_imminent": True,
                "detail": detail,
                "timestamp": time.time(),
            })
            pub.publish(msg)
        except Exception:
            pass

    # ── executor interface ──────────────────────────────────────────────────────
    def health(self) -> Optional[str]:
        """Why the monitor is NOT watching the cell right now, or None.

        Disabled (no kinematics), its thread gone, its last full check older than
        a few periods, or an arm whose feedback it cannot see. The executor treats
        any answer as a fault: motion runs only while something is watching.
        ``COLLISION_MONITOR_REQUIRED=0`` lets a cell without kinematics run
        unwatched (a disabled monitor is then healthy; the rest still counts).
        """
        if not self._enabled:
            if not self._required:
                return None
            return f"disabled ({self._disabled_reason or 'not started'})"
        if self._thread is not None and not self._thread.is_alive():
            return "its thread has stopped"
        if self._fault:
            return self._fault
        now = time.monotonic()
        stale_s = max(0.5, 5.0 / self._rate_hz)
        if self._last_ok is None:
            if now - (self._started_at or now) > max(2.0, stale_s):
                return "no check has completed since it started"
            return None
        if now - self._last_ok > stale_s:
            return f"its last check is {now - self._last_ok:.1f} s old"
        if self._blind:
            return f"no fresh feedback from {', '.join(sorted(self._blind))}"
        return None

    def wait_ready(self, timeout_s: float = 2.0) -> Optional[str]:
        """Block until the first full check completed (or it cannot); returns
        :meth:`health` — what the executor checks before the first dispatch."""
        deadline = time.monotonic() + timeout_s
        while self._enabled and self._last_ok is None and time.monotonic() < deadline:
            time.sleep(0.02)
        return self.health()

    def imminent(self) -> Optional[dict]:
        """Return the current imminent-collision detail, or None if clear."""
        with self._lock:
            return dict(self._imminent) if self._imminent is not None else None

    def min_clearance(self) -> float:
        with self._lock:
            return self._min_clearance


#: Feedback older than this is not where the arm is (robot agents publish at 10 Hz).
FEEDBACK_STALE_S = float(os.environ.get("COLLISION_MONITOR_FEEDBACK_STALE_S", "1.0"))


def _fresh(status: Optional[dict]) -> bool:
    """A status the monitor may judge from: received recently. A status without a
    receive time (an older runtime, a test double) counts as fresh."""
    t = (status or {}).get("_recv_t")
    if t is None:
        return True
    try:
        return time.time() - float(t) <= FEEDBACK_STALE_S
    except (TypeError, ValueError):
        return False
