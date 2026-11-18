#!/usr/bin/env python3
"""
Pinocchio Robot Controller — simulated robot "hardware" for one robot.

This is the FJT (FollowJointTrajectory) action server the robot agent dispatches
to. It closes the motion loop in simulation:

    robot_agent  --FJT goal-->  this controller  --joint_states-->  robot_agent
                                                  --joint_states-->  cell collision monitor
                                                  --joint_states-->  rviz relay (launch_viz)

Without it, the robot agent's FJT dispatch has no server, robots never move, and
the deterministic executor's position-vs-time check (correctly) flags the arms
as off-schedule. With it, the arms track their planned trajectories in sim.

Responsibilities:
  1. FollowJointTrajectory action server at
       /{cell}/{ns}/{rid}/joint_trajectory_controller/follow_joint_trajectory
  2. Trajectory enrichment (manifold-correct Pinocchio interpolation when the
     URDF loads, linear otherwise) + a short deceleration + hold at the end.
  3. ~120 Hz control loop stepping the joints toward the trajectory and
     publishing /{cell}/{ns}/{rid}/joint_states.

Ported and trimmed from multi-robot-agent/controller/pinocchio_robot_controller.py:
  - loads the URDF via utils.arm_model (cell.yaml reference or the generated
    cells/<cell>/urdf/<rid>/*.urdf) — no robot_description topic
  - no robot_state_publisher / static-TF / clock (RViz TF is owned by launch_viz)

Environment variables:
  CELL_ID           cell identifier (default: welding_cell_2_small)
  ROBOT_NAMESPACE   shared ROS namespace (default: arms)
  ROBOT_ID          unique robot identity (default: robot1)
  JOINT_COUNT       actuated joints (default: 6)
  CELL_BASE_PATH    cells mount (default: /cells)
  PUBLISH_RATE_HZ   control/joint_state rate (default: 120)
  TRAJ_INTERP_FACTOR / TRAJ_HOLD_TIME / TRAJ_DECEL_POINTS  enrichment tuning
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from agents.cell_defaults import default_cell_id  # noqa: E402

CELL_ID     = default_cell_id().strip("/")
ROBOT_NS    = os.environ.get("ROBOT_NAMESPACE", "arms")
ROBOT_ID    = os.environ.get("ROBOT_ID", "robot1")
JOINT_COUNT = int(os.environ.get("JOINT_COUNT", "6"))
CELL_BASE   = os.environ.get("CELL_BASE_PATH", "/cells").rstrip("/")
PUBLISH_RATE_HZ = float(os.environ.get("PUBLISH_RATE_HZ", "120"))

FULL_NS  = f"/{CELL_ID}/{ROBOT_NS}/{ROBOT_ID}"
FULL_PFX = f"{CELL_ID}/{ROBOT_NS}/{ROBOT_ID}"
# The pose this controller reports until it is commanded elsewhere, and the one
# the arm sits at when the cell comes up. It has to be the cell's declared home:
# the collision monitor and the planner both believe what this controller
# publishes, so a literal here makes every arm claim a pose it is not in.
from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose  # noqa: E402


HOME_JOINTS = home_pose(CELL_ID, ROBOT_ID, HOME_JOINTS_FALLBACK)


class PinocchioRobotController(Node):
    def __init__(self):
        super().__init__(f"pinocchio_robot_controller_{ROBOT_ID}")

        self._timer_group = ReentrantCallbackGroup()
        self._action_group = ReentrantCallbackGroup()

        self._current_joints = list(HOME_JOINTS[:JOINT_COUNT])
        self._target_joints = list(HOME_JOINTS[:JOINT_COUNT])
        self._joint_lock = threading.Lock()

        self._traj_points: list = []
        self._traj_start: float = 0.0
        self._traj_active: bool = False

        _v_scale = float(os.environ.get("ROBOT_VELOCITY_SCALE", "1.0"))
        self._interp_factor = int(os.environ.get("TRAJ_INTERP_FACTOR", "4"))
        self._hold_time = float(os.environ.get("TRAJ_HOLD_TIME", str(round(0.5 / max(0.5, _v_scale), 2))))
        self._decel_points = int(os.environ.get("TRAJ_DECEL_POINTS", "4"))

        self._pin_model = None  # (model, data) once the URDF loads
        self._pin_lock = threading.Lock()
        threading.Thread(target=self._load_pinocchio, daemon=True).start()

        self._joint_names = [f"{FULL_PFX}/joint_{i+1}" for i in range(JOINT_COUNT)]
        self._js_pub = self.create_publisher(JointState, f"{FULL_NS}/joint_states", 10)

        self._action_name = f"{FULL_NS}/joint_trajectory_controller/follow_joint_trajectory"
        self._action_server = ActionServer(
            self, FollowJointTrajectory, self._action_name,
            execute_callback=self._execute_goal,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=self._action_group,
        )

        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._tick, callback_group=self._timer_group)
        self.get_logger().info(
            f"[Controller] {FULL_NS} ready — FJT: {self._action_name} @ {PUBLISH_RATE_HZ:.0f}Hz"
        )

    # ── Pinocchio (optional, from file) ─────────────────────────────────────
    def _load_pinocchio(self):
        try:
            from utils import arm_model
            try:
                model, _data, _ee = arm_model.load(CELL_ID, ROBOT_ID)
            except FileNotFoundError as e:
                self.get_logger().warn(f"[Controller] {e} — using linear interpolation")
                return
            with self._pin_lock:
                self._pin_model = (model, model.createData())
            self.get_logger().info(f"[Controller] Pinocchio loaded (nq={model.nq})")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"[Controller] Pinocchio load failed: {e} — using linear interpolation")

    # ── FollowJointTrajectory action ────────────────────────────────────────
    def _execute_goal(self, goal_handle):
        traj = goal_handle.request.trajectory
        if not traj.points:
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            r.error_string = "Empty trajectory"
            return r

        raw = self._ros_to_internal(traj)

        # Wrist-flip guard relative to the live joint position.
        max_delta = float(os.environ.get("MAX_JOINT_DELTA", str(math.pi)))
        with self._joint_lock:
            start_joints = list(self._current_joints)
        raw = _validate_waypoints(
            [{"positions": start_joints, "velocities": [], "t": 0.0}] + raw, max_delta
        )[1:]

        if self._interp_factor > 1:
            raw = _enrich_interpolate(raw, self._interp_factor)
        if len(raw) >= 2:
            raw = _enrich_deceleration(raw, self._decel_points)
        if self._hold_time > 0:
            raw = _enrich_hold(raw, self._hold_time)
        if raw:
            raw[-1]["velocities"] = [0.0] * JOINT_COUNT

        with self._joint_lock:
            self._traj_points = raw
            self._traj_start = time.monotonic()
            self._traj_active = True

        total_t = (raw[-1]["t"] if raw else 0.0) + 0.5
        start_ts = time.monotonic()
        while time.monotonic() - start_ts < total_t:
            if not goal_handle.is_active:
                with self._joint_lock:
                    self._traj_active = False
                return FollowJointTrajectory.Result()
            if goal_handle.is_cancel_requested:
                with self._joint_lock:
                    self._traj_active = False
                    self._target_joints = list(self._current_joints)
                goal_handle.canceled()
                return FollowJointTrajectory.Result()
            with self._joint_lock:
                active = self._traj_active
            if not active and (time.monotonic() - start_ts) > (raw[-1]["t"] * 0.5 if raw else 0):
                break
            time.sleep(0.05)

        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        if goal_handle.is_active:
            try:
                goal_handle.succeed()
            except Exception:
                pass
        return result

    def _ros_to_internal(self, traj: JointTrajectory) -> list:
        out = []
        for pt in traj.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            out.append({
                "positions": list(pt.positions[:JOINT_COUNT]),
                "velocities": list(pt.velocities[:JOINT_COUNT]) if pt.velocities else [],
                "t": t,
            })
        return out

    # ── control loop ─────────────────────────────────────────────────────────
    def _tick(self):
        dt = 1.0 / PUBLISH_RATE_HZ
        with self._joint_lock:
            active = self._traj_active
            points = self._traj_points
            start = self._traj_start
            prev = list(self._current_joints)

        if active and points:
            elapsed = time.monotonic() - start
            new_pos, new_vel = _sample_trajectory(points, elapsed)
            if elapsed >= points[-1]["t"]:
                with self._joint_lock:
                    self._traj_active = False
                    self._target_joints = list(points[-1]["positions"])
        else:
            with self._joint_lock:
                target = list(self._target_joints)
            new_pos = self._interp_joints(prev, target, min(1.0, dt * 10))
            new_vel = [(new_pos[i] - prev[i]) / dt for i in range(JOINT_COUNT)]

        with self._joint_lock:
            self._current_joints = new_pos

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self._joint_names)
        js.position = new_pos
        js.velocity = new_vel
        self._js_pub.publish(js)

    def _interp_joints(self, q0: list, q1: list, alpha: float) -> list:
        with self._pin_lock:
            pin_data = self._pin_model
        if pin_data is not None:
            try:
                import pinocchio as pin
                import numpy as np
                model, _ = pin_data
                qv0 = np.zeros(model.nq); qv0[:len(q0)] = q0
                qv1 = np.zeros(model.nq); qv1[:len(q1)] = q1
                return list(pin.interpolate(model, qv0, qv1, alpha)[:JOINT_COUNT])
            except Exception:
                pass
        return [(1 - alpha) * q0[i] + alpha * q1[i] for i in range(JOINT_COUNT)]


# ── Wrist-flip guard + enrichment (pure Python, ported verbatim) ─────────────

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _shortest_angular_delta(a: float, b: float) -> float:
    return _wrap_angle(b - a)


def _validate_waypoints(points: list, max_delta: float = math.pi) -> list:
    if len(points) < 2:
        return points
    validated = [points[0]]
    for i in range(1, len(points)):
        prev_pos = validated[-1]["positions"]
        corrected = list(points[i]["positions"])
        for j in range(min(len(prev_pos), len(corrected))):
            if abs(corrected[j] - prev_pos[j]) > max_delta:
                corrected[j] = prev_pos[j] + _shortest_angular_delta(prev_pos[j], corrected[j])
        entry = dict(points[i]); entry["positions"] = corrected
        validated.append(entry)
    return validated


def _enrich_interpolate(points: list, factor: int) -> list:
    if factor <= 1 or len(points) < 2:
        return points
    out = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        out.append(p1)
        for j in range(1, factor):
            alpha = j / factor
            pos = [p1["positions"][k] + alpha * (p2["positions"][k] - p1["positions"][k])
                   for k in range(len(p1["positions"]))]
            if p1["velocities"] and p2["velocities"]:
                vel = [p1["velocities"][k] + alpha * (p2["velocities"][k] - p1["velocities"][k])
                       for k in range(len(p1["velocities"]))]
            else:
                vel = [0.0] * len(pos)
            out.append({"positions": pos, "velocities": vel,
                        "t": p1["t"] + j * (p2["t"] - p1["t"]) / factor})
    out.append(points[-1])
    return out


def _enrich_deceleration(points: list, n_decel: int) -> list:
    if n_decel < 1 or len(points) < 2:
        return points
    last, prev = points[-1], points[-2]
    peak_vel = max(abs(v) for v in prev["velocities"]) if prev["velocities"] else 0.0
    if peak_vel < 0.01:
        return points
    seg = last["t"] - prev["t"]
    decel_window = max(0.3, min(seg, peak_vel * 0.4))
    t_start = last["t"] - decel_window
    alpha_start = max(0.0, (t_start - prev["t"]) / seg if seg > 1e-9 else 0.0)
    pos_start = [prev["positions"][k] + alpha_start * (last["positions"][k] - prev["positions"][k])
                 for k in range(len(last["positions"]))]
    decel = []
    for i in range(1, n_decel + 1):
        alpha = i / (n_decel + 1)
        smooth = 3 * alpha ** 2 - 2 * alpha ** 3
        pos = [pos_start[k] + smooth * (last["positions"][k] - pos_start[k])
               for k in range(len(last["positions"]))]
        vel_s = max(0.0, 1.0 - smooth)
        vel = ([prev["velocities"][k] * vel_s for k in range(len(prev["velocities"]))]
               if prev["velocities"] else [0.0] * len(pos))
        decel.append({"positions": pos, "velocities": vel,
                      "t": t_start + i * decel_window / (n_decel + 1)})
    splice = len(points) - 1
    for idx in range(len(points) - 1, -1, -1):
        if points[idx]["t"] <= t_start:
            splice = idx
            break
    return points[:splice + 1] + decel + [last]


def _enrich_hold(points: list, hold_time: float, step: float = 0.1) -> list:
    if hold_time <= 0 or not points:
        return points
    last = points[-1]
    final_v = max(abs(v) for v in last["velocities"]) if last["velocities"] else 0.0
    if final_v < 0.005:
        return points
    peak = max((max(abs(v) for v in pt["velocities"]) for pt in points if pt["velocities"]), default=0.0)
    actual = min(hold_time * 3, hold_time + peak * 0.2)
    n = max(1, int(actual / step))
    dof = len(last["positions"])
    return points + [
        {"positions": list(last["positions"]), "velocities": [0.0] * dof, "t": last["t"] + (i + 1) * step}
        for i in range(n)
    ]


def _sample_trajectory(points: list, elapsed: float):
    if elapsed <= points[0]["t"]:
        p = points[0]
        return list(p["positions"]), (p["velocities"] or [0.0] * len(p["positions"]))
    if elapsed >= points[-1]["t"]:
        p = points[-1]
        return list(p["positions"]), [0.0] * len(p["positions"])
    for i in range(len(points) - 1):
        t0, t1 = points[i]["t"], points[i + 1]["t"]
        if t0 <= elapsed <= t1:
            d = t1 - t0
            alpha = (elapsed - t0) / d if d > 1e-9 else 1.0
            p0, p1 = points[i], points[i + 1]
            v0 = p0["velocities"] or [0.0] * len(p0["positions"])
            v1 = p1["velocities"] or [0.0] * len(p1["positions"])
            pos = [(1 - alpha) * p0["positions"][k] + alpha * p1["positions"][k]
                   for k in range(len(p0["positions"]))]
            vel = [(1 - alpha) * v0[k] + alpha * v1[k] for k in range(len(v0))]
            return pos, vel
    p = points[-1]
    return list(p["positions"]), [0.0] * len(p["positions"])


def main():
    rclpy.init()
    node = PinocchioRobotController()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    def _shutdown(_sig, _frame):
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
