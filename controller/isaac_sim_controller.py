#!/usr/bin/env python3
"""
Isaac Sim Controller — ROS 2 topic bridge for the NVIDIA Isaac Sim backend.

This is the Isaac-specific counterpart of ``pinocchio_robot_controller.py``. It
exposes the SAME FollowJointTrajectory (FJT) action server the robot agent
dispatches to, so the robot agent is unchanged — but instead of simulating the
motion itself, it converts each FJT goal into the joint-command topic Isaac Sim
listens on, and relays Isaac's joint_states / contacts back onto the cell's
topic contract.

    robot_agent --FJT goal-->  THIS BRIDGE  --/{scene}/{prim}/joint_command-->  Isaac Sim
                                            <--/{scene}/{prim}/joint_states---
    robot_agent / collision monitor / rviz  <--/{cell}/{ns}/{rid}/joint_states--

Use it via ``ROBOT_CONTROLLER=isaac`` (see scripts/generate_compose.py). That
runs one of these per robot INSTEAD of the pinocchio controller — the controller
is replaced, not disabled, so the FJT action server always exists and the arm
moves in Isaac.

Topic contract (welding_cell_2_small reference, scene root "Untitled"):
  Robot 1 articulation prim: kr240r2900__sticktool
  Robot 2 articulation prim: kr240r2900__sticktool_2
  Isaac joint names        : joint_a1 .. joint_a6

  Isaac Sim publishes:
    /{scene_root}/{prim}/joint_states     sensor_msgs/JointState
    /{scene_root}/{prim}/contacts/{link}  std_msgs/Bool
  Isaac Sim subscribes:
    /{scene_root}/{prim}/joint_command    sensor_msgs/JointState

  This bridge:
    FJT server: /{cell}/{ns}/{rid}/joint_trajectory_controller/follow_joint_trajectory
                -> streams JointState commands to /{scene_root}/{prim}/joint_command
    Publishes:  /{cell}/{ns}/{rid}/joint_states      (Isaac states, re-stamped)
    Publishes:  /{cell}/{ns}/{rid}/contacts/{link}   (Isaac Bool -> String JSON)

DDS domain
----------
A single ROS 2 node lives on one ROS_DOMAIN_ID. This bridge must therefore share
a domain with BOTH Isaac Sim and the robot agent. Run Isaac Sim, the robot
agent, and this bridge on the same ROS_DOMAIN_ID (the reference Isaac layout uses
0). If Isaac is on a different domain than the rest of the stack, put the
robot-agent + this bridge on Isaac's domain.

Environment variables
---------------------
  CELL_ID              e.g. welding_cell_2_small
  ROBOT_NAMESPACE      shared ROS namespace, e.g. "arms"
  ROBOT_ID             unique robot identity, e.g. "robot1"
  ROBOT_NAME           Isaac Sim articulation prim (default: derived from ROBOT_ID)
  JOINT_COUNT          actuated joints (default: 6)
  ISAAC_SCENE_ROOT     USD stage root (default: Untitled)
  ISAAC_JOINT_STATES_TOPIC   override full topic
  ISAAC_JOINT_COMMAND_TOPIC  override full topic
  ISAAC_CONTACT_PREFIX       override prefix
  ISAAC_JOINT_PREFIX   Isaac joint-name prefix (default: joint_a -> joint_a1..N)
  PUBLISH_RATE_HZ      command stream rate Hz (default: 50)
  POSITION_TOL         goal-reached tolerance rad (default: 0.02)
  GOAL_TIMEOUT_EXTRA   extra seconds after traj duration (default: 5.0)
  ROBOT_VELOCITY_SCALE scale factor for hold_time and timeouts (default: 1.0)
  TRAJ_INTERP_FACTOR   sub-points between waypoints (default: 4)
  TRAJ_HOLD_TIME       hold seconds at endpoint (default: auto = 0.5/scale)
  TRAJ_DECEL_POINTS    deceleration points (default: 4)
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import List, Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectoryPoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from agents.cell_defaults import default_cell_id  # noqa: E402

CELL_ID     = default_cell_id().strip("/")
ROBOT_NS    = os.environ.get("ROBOT_NAMESPACE", "arms")     # shared ROS namespace
ROBOT_ID    = os.environ.get("ROBOT_ID",        "robot1")   # unique robot identity
JOINT_COUNT = int(os.environ.get("JOINT_COUNT", "6"))

_v_scale           = float(os.environ.get("ROBOT_VELOCITY_SCALE", "1.0"))
PUBLISH_RATE_HZ    = float(os.environ.get("PUBLISH_RATE_HZ",    "50"))
POSITION_TOL       = float(os.environ.get("POSITION_TOL",       "0.02"))
GOAL_TIMEOUT_EXTRA = float(os.environ.get("GOAL_TIMEOUT_EXTRA", "5.0")) / max(0.5, _v_scale)

# Enrichment parameters — match pinocchio_robot_controller
INTERP_FACTOR = int(os.environ.get("TRAJ_INTERP_FACTOR", "4"))
HOLD_TIME     = float(os.environ.get("TRAJ_HOLD_TIME",   str(round(0.5 / max(0.5, _v_scale), 2))))
DECEL_POINTS  = int(os.environ.get("TRAJ_DECEL_POINTS",  "4"))

# Wrist-flip detection: reject waypoint-to-waypoint jumps > MAX_JOINT_DELTA rad.
# π (3.14) — any jump larger than half a revolution is treated as a flip.
MAX_JOINT_DELTA = float(os.environ.get("MAX_JOINT_DELTA", str(math.pi)))

ISAAC_SCENE_ROOT = os.environ.get("ISAAC_SCENE_ROOT", "Untitled")

# Isaac articulation prim, derived from the unique ROBOT_ID (robot1, robot2, …):
#   robot1  -> kr240r2900__sticktool
#   robot2  -> kr240r2900__sticktool_2
# matching the scene.yaml export. Override with ROBOT_NAME for other cells.
_robot_index = ROBOT_ID.replace("robot", "").strip()
if _robot_index in ("", "1"):
    _default_prim = "kr240r2900__sticktool"
else:
    _default_prim = f"kr240r2900__sticktool_{_robot_index}"

ROBOT_NAME = os.environ.get("ROBOT_NAME", _default_prim)

ISAAC_JOINT_STATES_TOPIC = os.environ.get(
    "ISAAC_JOINT_STATES_TOPIC", f"/{ISAAC_SCENE_ROOT}/{ROBOT_NAME}/joint_states",
)
ISAAC_JOINT_COMMAND_TOPIC = os.environ.get(
    "ISAAC_JOINT_COMMAND_TOPIC", f"/{ISAAC_SCENE_ROOT}/{ROBOT_NAME}/joint_command",
)
ISAAC_CONTACT_PREFIX = os.environ.get(
    "ISAAC_CONTACT_PREFIX", f"/{ISAAC_SCENE_ROOT}/{ROBOT_NAME}/contacts",
)

# Joint names Isaac Sim expects on the joint_command topic.
_ISAAC_JOINT_PREFIX = os.environ.get("ISAAC_JOINT_PREFIX", "joint_a")
ISAAC_JOINT_NAMES = [f"{_ISAAC_JOINT_PREFIX}{i + 1}" for i in range(JOINT_COUNT)]

FULL_NS    = f"/{CELL_ID}/{ROBOT_NS}/{ROBOT_ID}"
FULL_PFX   = f"{CELL_ID}/{ROBOT_NS}/{ROBOT_ID}"
FJT_ACTION = f"{FULL_NS}/joint_trajectory_controller/follow_joint_trajectory"

# Isaac contact link names -> the short names used on the cell contract.
ISAAC_CONTACT_LINKS = [
    "like_kr240r2900_base_link", "like_kr240r2900_link_1", "like_kr240r2900_link_2",
    "like_kr240r2900_link_3",    "like_kr240r2900_link_4", "like_kr240r2900_link_5",
    "like_kr240r2900_link_6",
]
_CONTACT_LINK_MAP = {
    "like_kr240r2900_base_link": "base_link",
    "like_kr240r2900_link_1":    "link_1",
    "like_kr240r2900_link_2":    "link_2",
    "like_kr240r2900_link_3":    "link_3",
    "like_kr240r2900_link_4":    "link_4",
    "like_kr240r2900_link_5":    "link_5",
    "like_kr240r2900_link_6":    "link_6",
}


# ---------------------------------------------------------------------------
# Angle utilities (wrap-aware, shared shape with pinocchio_robot_controller)
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _shortest_angular_delta(a: float, b: float) -> float:
    """Shortest signed angular difference from a to b, in [-π, π]."""
    return _wrap_angle(b - a)


def _angular_lerp(a: float, b: float, t: float) -> float:
    """Interpolate from angle a to b via the shortest path."""
    return a + t * _shortest_angular_delta(a, b)


def _validate_waypoints(points: list, max_delta: float = MAX_JOINT_DELTA) -> list:
    """Clamp large joint jumps between adjacent waypoints (wrist-flip guard).

    A jump > max_delta rad on any joint is almost certainly an IK branch flip;
    replace the target with the nearest wrap-equivalent angle so the arm takes
    the short way round instead of spinning ~360°.
    """
    if len(points) < 2:
        return points

    validated = [points[0]]
    for i in range(1, len(points)):
        prev_pos  = validated[-1]["positions"]
        corrected = list(points[i]["positions"])
        flipped   = []
        for j in range(min(len(prev_pos), len(corrected))):
            if abs(corrected[j] - prev_pos[j]) > max_delta:
                wrapped = prev_pos[j] + _shortest_angular_delta(prev_pos[j], corrected[j])
                flipped.append((j, prev_pos[j], corrected[j], wrapped))
                corrected[j] = wrapped
        if flipped:
            info = ", ".join(
                f"j{j + 1}: {math.degrees(old):.1f}°→{math.degrees(new):.1f}° "
                f"(was {math.degrees(raw):.1f}°)"
                for j, old, raw, new in flipped
            )
            print(f"  [IsaacCtrl] {ROBOT_ID}: wrist-flip clamped at waypoint {i}: {info}", flush=True)
        entry = dict(points[i])
        entry["positions"] = corrected
        validated.append(entry)
    return validated


def _interpolate_trajectory(start: List[float], traj_points: list, rate_hz: float) -> List[List[float]]:
    """Expand sparse JointTrajectoryPoints into dense position vectors at rate_hz.

    Uses shortest-path angular interpolation so the dense sampling itself cannot
    introduce a wrist flip.
    """
    result: List[List[float]] = []
    prev_pos  = list(start)
    prev_time = 0.0
    for pt in traj_points:
        t_sec  = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
        target = list(pt.positions[:JOINT_COUNT])
        if not target:
            continue
        duration = max(t_sec - prev_time, 1.0 / rate_hz)
        n_steps  = max(1, int(duration * rate_hz))
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            result.append([_angular_lerp(prev_pos[j], target[j], alpha) for j in range(len(target))])
        prev_pos  = target
        prev_time = t_sec
    return result


# ---------------------------------------------------------------------------
# Enrichment pipeline (matches pinocchio_robot_controller)
# ---------------------------------------------------------------------------

def _enrich_interpolate(points: list, factor: int) -> list:
    if factor <= 1 or len(points) < 2:
        return points
    out = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        out.append(p1)
        for j in range(1, factor):
            alpha = j / factor
            pos = [_angular_lerp(p1["positions"][k], p2["positions"][k], alpha)
                   for k in range(len(p1["positions"]))]
            if p1.get("velocities") and p2.get("velocities"):
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
    peak_vel = max(abs(v) for v in prev.get("velocities", [])) if prev.get("velocities") else 0.0
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
        alpha  = i / (n_decel + 1)
        smooth = 3 * alpha ** 2 - 2 * alpha ** 3
        pos    = [pos_start[k] + smooth * (last["positions"][k] - pos_start[k])
                  for k in range(len(last["positions"]))]
        vel_s  = max(0.0, 1.0 - smooth)
        vel    = ([prev["velocities"][k] * vel_s for k in range(len(prev["velocities"]))]
                  if prev.get("velocities") else [0.0] * len(pos))
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
    last    = points[-1]
    final_v = max(abs(v) for v in last.get("velocities", [])) if last.get("velocities") else 0.0
    if final_v < 0.005:
        return points
    peak = max((max(abs(v) for v in pt["velocities"]) for pt in points if pt.get("velocities")), default=0.0)
    actual = min(hold_time * 3, hold_time + peak * 0.2)
    n = max(1, int(actual / step))
    dof = len(last["positions"])
    return points + [
        {"positions": list(last["positions"]), "velocities": [0.0] * dof, "t": last["t"] + (i + 1) * step}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Controller node
# ---------------------------------------------------------------------------

class IsaacSimController(Node):
    """Bridges the cell's FJT action <-> Isaac Sim ROS 2 bridge topics."""

    def __init__(self):
        super().__init__(f"isaac_sim_controller_{ROBOT_ID}")
        cb = ReentrantCallbackGroup()

        self._latest_joints: Optional[List[float]] = None
        self._joint_lock = threading.Lock()

        qos_sensor = QoSPresetProfiles.SENSOR_DATA.value

        # ── Joint state bridge: Isaac -> our namespace ──────────────────────
        self._js_pub = self.create_publisher(JointState, f"{FULL_NS}/joint_states", 10)
        self.create_subscription(
            JointState, ISAAC_JOINT_STATES_TOPIC,
            self._on_isaac_joint_states, qos_sensor, callback_group=cb,
        )

        # ── Joint command publisher: us -> Isaac Sim ────────────────────────
        self._cmd_pub = self.create_publisher(JointState, ISAAC_JOINT_COMMAND_TOPIC, 10)

        # ── Contact bridge: Isaac Bool -> our String JSON ───────────────────
        self._contact_pubs: dict = {}
        for isaac_link in ISAAC_CONTACT_LINKS:
            short_link  = _CONTACT_LINK_MAP.get(isaac_link, isaac_link)
            isaac_topic = f"{ISAAC_CONTACT_PREFIX}/{isaac_link}"
            our_topic   = f"{FULL_NS}/contacts/{short_link}"
            self._contact_pubs[isaac_link] = self.create_publisher(String, our_topic, 10)
            self.create_subscription(
                Bool, isaac_topic,
                lambda msg, lnk=isaac_link: self._on_isaac_contact(msg, lnk),
                qos_sensor, callback_group=cb,
            )

        # ── FJT action server (same name the robot agent dispatches to) ─────
        self._action_server = ActionServer(
            self, FollowJointTrajectory, FJT_ACTION,
            execute_callback=self._execute_goal,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=cb,
        )

        self.get_logger().info(
            f"\n[IsaacSimController] {ROBOT_NS}/{ROBOT_ID} ready"
            f"\n  DDS domain           : ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '?')}"
            f"\n  Isaac prim           : {ROBOT_NAME}  (scene root: {ISAAC_SCENE_ROOT})"
            f"\n  Isaac joint states   : {ISAAC_JOINT_STATES_TOPIC}"
            f"\n  Isaac joint command  : {ISAAC_JOINT_COMMAND_TOPIC}"
            f"\n  Isaac joint names    : {ISAAC_JOINT_NAMES}"
            f"\n  Our joint states     : {FULL_NS}/joint_states"
            f"\n  FJT action           : {FJT_ACTION}"
            f"\n  Command rate         : {PUBLISH_RATE_HZ} Hz   pos tol: {POSITION_TOL} rad"
            f"\n  Velocity scale       : {_v_scale}  hold={HOLD_TIME:.2f}s"
            f"\n  Max joint delta      : {math.degrees(MAX_JOINT_DELTA):.0f}° (wrist-flip guard)"
        )

    # ── Joint state bridge ──────────────────────────────────────────────────
    def _on_isaac_joint_states(self, msg: JointState):
        """Cache Isaac's joint state and re-publish on the cell contract topic.

        Isaac emits NaN/Inf during collision recovery or joint-limit violations;
        reject those and republish the last known good state instead.
        """
        pos = list(msg.position[:JOINT_COUNT])
        if any(not math.isfinite(p) for p in pos):
            self.get_logger().warn(
                f"[IsaacSimController] {ROBOT_ID}: non-finite joint state from Isaac "
                f"({[round(p, 3) for p in pos]}) — republishing last good state"
            )
            with self._joint_lock:
                pos = list(self._latest_joints) if self._latest_joints else [0.0] * JOINT_COUNT
        else:
            with self._joint_lock:
                self._latest_joints = pos

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name     = [f"{FULL_PFX}/joint_{i + 1}" for i in range(JOINT_COUNT)]
        out.position = pos
        vel = list(msg.velocity[:JOINT_COUNT]) if msg.velocity else []
        out.velocity = vel if vel and all(math.isfinite(v) for v in vel) else []
        self._js_pub.publish(out)

    # ── Contact bridge ──────────────────────────────────────────────────────
    def _on_isaac_contact(self, msg: Bool, isaac_link: str):
        if msg.data and isaac_link in self._contact_pubs:
            short = _CONTACT_LINK_MAP.get(isaac_link, isaac_link)
            out = String()
            out.data = json.dumps({"link": short, "contact": True, "source": "isaac_sim"})
            self._contact_pubs[isaac_link].publish(out)

    # ── FJT action server ─────────────────────────────────────────────────────
    def _execute_goal(self, goal_handle):
        """Execute an FJT goal by streaming JointState commands to Isaac Sim."""
        trajectory = goal_handle.request.trajectory
        if not trajectory.points:
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code   = FollowJointTrajectory.Result.INVALID_GOAL
            r.error_string = "Empty trajectory"
            return r

        n_pts = len(trajectory.points)
        self.get_logger().info(
            f"[IsaacSimController] {ROBOT_ID}: executing {n_pts} waypoints "
            f"(interp×{INTERP_FACTOR}, decel={DECEL_POINTS}, hold={HOLD_TIME:.2f}s)"
        )

        with self._joint_lock:
            start = list(self._latest_joints) if self._latest_joints else [0.0] * JOINT_COUNT

        raw = []
        for pt in trajectory.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            raw.append({
                "positions":  list(pt.positions[:JOINT_COUNT]),
                "velocities": list(pt.velocities[:JOINT_COUNT]) if pt.velocities else [],
                "t": t,
            })

        # Wrist-flip validation relative to the live joint position.
        raw = _validate_waypoints([{"positions": start, "velocities": [], "t": 0.0}] + raw, MAX_JOINT_DELTA)[1:]

        # Enrichment pipeline (produces the deceleration/hold tail).
        if INTERP_FACTOR > 1:
            raw = _enrich_interpolate(raw, INTERP_FACTOR)
        if len(raw) >= 2:
            raw = _enrich_deceleration(raw, DECEL_POINTS)
        if HOLD_TIME > 0:
            raw = _enrich_hold(raw, HOLD_TIME)
        if raw:
            raw[-1]["velocities"] = [0.0] * JOINT_COUNT

        # Dense, time-proportional command stream to Isaac at PUBLISH_RATE_HZ.
        dense = _interpolate_trajectory(start, trajectory.points, PUBLISH_RATE_HZ)

        traj_end_t = (trajectory.points[-1].time_from_start.sec +
                      trajectory.points[-1].time_from_start.nanosec * 1e-9) if trajectory.points else 0.0
        hold_positions = [rp["positions"] for rp in raw if rp["t"] > traj_end_t + 0.01]

        dt = 1.0 / PUBLISH_RATE_HZ

        for positions in dense:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                r = FollowJointTrajectory.Result()
                r.error_code   = FollowJointTrajectory.Result.SUCCESSFUL
                r.error_string = "Cancelled"
                return r

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name     = ISAAC_JOINT_NAMES
            cmd.position = positions
            self._cmd_pub.publish(cmd)

            fb = FollowJointTrajectory.Feedback()
            fb.header.stamp = cmd.header.stamp
            des = JointTrajectoryPoint(); des.positions = positions
            fb.desired = des
            with self._joint_lock:
                act = JointTrajectoryPoint()
                act.positions = list(self._latest_joints) if self._latest_joints else positions
            fb.actual = act
            goal_handle.publish_feedback(fb)
            time.sleep(dt)

        # Deceleration/hold tail — smooth the stop.
        for positions in hold_positions:
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name     = ISAAC_JOINT_NAMES
            cmd.position = positions
            self._cmd_pub.publish(cmd)
            time.sleep(dt)

        # Wait for convergence (shortest-path error to handle wrap-around).
        goal_pos = list(trajectory.points[-1].positions[:JOINT_COUNT])
        deadline = time.monotonic() + GOAL_TIMEOUT_EXTRA
        reached  = False
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                break
            with self._joint_lock:
                cur = list(self._latest_joints) if self._latest_joints else None
            if cur:
                err = max(abs(_shortest_angular_delta(cur[i], goal_pos[i]))
                          for i in range(min(len(cur), len(goal_pos))))
                if err < POSITION_TOL:
                    reached = True
                    break
            time.sleep(0.05)

        result = FollowJointTrajectory.Result()
        if reached:
            self.get_logger().info(f"[IsaacSimController] {ROBOT_ID}: goal reached")
            goal_handle.succeed()
            result.error_code   = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = ""
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.error_code   = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = "Cancelled"
        else:
            self.get_logger().warn(
                f"[IsaacSimController] {ROBOT_ID}: goal timed out after {GOAL_TIMEOUT_EXTRA:.1f}s"
            )
            goal_handle.abort()
            result.error_code   = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = f"Did not converge within {GOAL_TIMEOUT_EXTRA:.1f}s (tol={POSITION_TOL} rad)"
        return result


def main():
    rclpy.init()
    node = IsaacSimController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
