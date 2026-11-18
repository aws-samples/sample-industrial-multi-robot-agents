"""
Robot Agent — one container per robot.

Executes precomputed joint trajectories received from the cell agent.
No planning, no LLM — pure execution.

Responsibilities:
  1. Subscribe to /{cell_id}/{ns}/dispatch_subtask  (receive trajectory JSON)
  2. Send FollowJointTrajectory action goals to the ros2_control controller
  3. Publish robot status (joint positions + EE pose) to /{cell_id}/{ns}/robot_status
  4. Report completion back to cell agent
  5. Honour cell-wide e-stop on /{cell_id}/estop — cancel the active FJT goal and
     reject new dispatches until the stop is cleared

Environment variables:
  CELL_ID          — cell identifier (e.g. "welding_cell_2_small")
  ROBOT_NAMESPACE  — robot namespace (e.g. "robot1")
  ROBOT_ID         — robot model ID (e.g. "kr240r2900")
  CONTROLLER_NAME  — FJT controller name (default: joint_trajectory_controller)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

from agents.cell_defaults import default_cell_id

CELL_ID      = default_cell_id()
ROBOT_NS     = os.environ.get("ROBOT_NAMESPACE", "arms")   # shared ROS namespace
ROBOT_ID     = os.environ.get("ROBOT_ID", "robot1")        # unique robot identity
ROBOT_MODEL  = os.environ.get("ROBOT_MODEL", "kr240r2900") # robot model
CTRL_NAME    = os.environ.get("CONTROLLER_NAME", "joint_trajectory_controller")

# All robot topics live under /{cell}/{namespace}/{robot_id}/...
ROBOT_ADDR   = f"{CELL_ID}/{ROBOT_NS}/{ROBOT_ID}"

JOINT_NAMES = [f"{ROBOT_ADDR}/joint_{i}" for i in range(1, 7)]

# Resolved from the cell that owns this arm. The agent reports this as its
# joint state until the controller sends real feedback, so a literal here makes
# the arm claim to be somewhere it is not — and the collision monitor believes it.
from agents.home_pose import HOME_JOINTS_FALLBACK as _HOME_FALLBACK, home_pose

from utils import cell_layout

HOME_JOINTS = home_pose(CELL_ID, ROBOT_ID, _HOME_FALLBACK)


def _build_fjt_goal(trajectory: list[dict]):
    """Build a FollowJointTrajectory goal from trajectory dicts."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import FollowJointTrajectory
    from builtin_interfaces.msg import Duration

    from agents.robot.fjt_profile import point_velocities

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES

    # The planned profile's velocities, not [0]*6 at every point: see fjt_profile.
    velocities = point_velocities(trajectory, 6)
    for pt, vel in zip(trajectory, velocities):
        tpt = JointTrajectoryPoint()
        pos = pt["positions"][:6]
        # Pad to 6 joints if URDF has fewer
        while len(pos) < 6:
            pos.append(0.0)
        tpt.positions  = pos
        tpt.velocities = [float(v) for v in vel]
        t_s = float(pt["time_s"])
        tpt.time_from_start = Duration(
            sec=int(t_s),
            nanosec=int((t_s - int(t_s)) * 1e9),
        )
        goal.trajectory.points.append(tpt)

    return goal


class RobotAgent:
    """ROS 2 node: subscribe → dispatch FJT → publish status."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from std_msgs.msg import String
        from control_msgs.action import FollowJointTrajectory
        from sensor_msgs.msg import JointState

        self._node = Node(f"robot_agent_{ROBOT_ID}", namespace=f"/{ROBOT_ADDR}")
        self._fjt_client = ActionClient(
            self._node,
            FollowJointTrajectory,
            f"/{ROBOT_ADDR}/{CTRL_NAME}/follow_joint_trajectory",
        )
        self._status_pub = self._node.create_publisher(
            String,
            f"/{ROBOT_ADDR}/robot_status",
            10,
        )
        self._joint_positions: list[float] = HOME_JOINTS.copy()
        self._current_task: str | None     = None

        # Subscribe to joint states for FK feedback
        self._node.create_subscription(
            JointState,
            f"/{ROBOT_ADDR}/joint_states",
            self._js_callback,
            10,
        )
        # Subscribe to dispatch topic
        self._node.create_subscription(
            String,
            f"/{ROBOT_ADDR}/dispatch_subtask",
            self._dispatch_callback,
            10,
        )
        # Cell-wide e-stop: a stop cancels the active FJT goal and blocks new
        # dispatches until cleared. This is what actually halts the robot when the
        # collision monitor or executor trips.
        self._estopped = False
        self._active_goal_handle = None
        self._node.create_subscription(
            String,
            f"/{CELL_ID}/estop",
            self._estop_callback,
            10,
        )
        # Periodic status publisher (10 Hz)
        self._status_timer = self._node.create_timer(0.1, self._publish_status)
        self._logger = self._node.get_logger()
        self._logger.info(f"[{ROBOT_ID}] Robot agent ready (ns={ROBOT_NS})")

    def _js_callback(self, msg):
        """Update joint positions from joint_states."""
        if len(msg.position) >= 6:
            self._joint_positions = list(msg.position[:6])

    def _estop_callback(self, msg):
        """Handle cell-wide e-stop.

        Message is JSON: ``{"reason": str}`` to stop, or ``{"clear": true}`` to
        release. A stop cancels any in-flight FJT goal and latches ``_estopped``
        so new dispatches are rejected until a clear arrives.
        """
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {}
        if data.get("clear"):
            self._estopped = False
            self._logger.info(f"[{ROBOT_ID}] E-STOP cleared")
            return
        self._estopped = True
        reason = data.get("reason", "unspecified")
        self._logger.error(f"[{ROBOT_ID}] E-STOP: {reason} — cancelling active motion")
        gh = self._active_goal_handle
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception:
                pass

    def _dispatch_callback(self, msg):
        """Receive a trajectory task from the cell agent."""
        import threading
        if self._estopped:
            self._logger.warning(f"[{ROBOT_ID}] Dispatch rejected — cell is E-STOPPED")
            return
        try:
            task = json.loads(msg.data)
        except Exception as e:
            self._logger.error(f"[{ROBOT_ID}] Bad task JSON: {e}")
            return

        op_id  = task.get("operation_id", "?")
        traj   = task.get("trajectory", [])
        hold_s = float(task.get("hold_duration_s", 0.5))
        vscale = float(task.get("velocity_scale", 1.0))

        self._logger.info(f"[{ROBOT_ID}] Dispatching op={op_id} pts={len(traj)}")

        if not traj:
            self._logger.warning(f"[{ROBOT_ID}] Empty trajectory for {op_id}")
            return

        # Apply velocity scale (stretch time)
        if vscale != 1.0 and vscale > 0:
            for pt in traj:
                pt["time_s"] = pt["time_s"] / vscale
                if pt.get("velocities"):            # slower clock, slower joints
                    pt["velocities"] = [float(v) * vscale for v in pt["velocities"]]

        threading.Thread(
            target=self._execute_trajectory,
            args=(traj, op_id, hold_s),
            daemon=True,
        ).start()

    def _execute_trajectory(self, trajectory: list[dict], op_id: str, hold_s: float):
        """Send FJT goal and wait for completion.

        Runs in a background thread. The node is already being spun by the main
        thread (``rclpy.spin`` in ``spin()``), which owns the global executor, so
        this thread MUST NOT spin the node itself — doing so raises
        ``RuntimeError('Executor is already spinning')`` and kills the thread
        before it can track the goal (leaving the arm "busy" forever and any
        dependent INTERLOCK_WAIT to time out). Instead we just poll the futures;
        the main executor services the action-client callbacks that complete
        them."""
        self._current_task = op_id

        if self._estopped:
            self._logger.warning(f"[{ROBOT_ID}] Skipping {op_id} — E-STOPPED")
            self._current_task = None
            return

        if not self._fjt_client.wait_for_server(timeout_sec=5.0):
            self._logger.error(f"[{ROBOT_ID}] FJT server not available for {op_id}")
            self._current_task = None
            return

        goal = _build_fjt_goal(trajectory)
        future = self._fjt_client.send_goal_async(goal)

        # Poll for goal acceptance (main executor services the response).
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)

        if not future.done() or not future.result().accepted:
            self._logger.error(f"[{ROBOT_ID}] Goal rejected for {op_id}")
            self._current_task = None
            return

        goal_handle = future.result()
        self._active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()

        # Wait for completion
        last_pts = len(trajectory)
        expected_s = float(trajectory[-1]["time_s"]) + hold_s + 5.0
        deadline = time.monotonic() + expected_s

        while not result_future.done() and time.monotonic() < deadline:
            if self._estopped:
                self._logger.warning(f"[{ROBOT_ID}] E-STOP during op={op_id} — cancelling goal")
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                break
            time.sleep(0.02)

        if result_future.done():
            result = result_future.result()
            ec     = result.result.error_code
            if ec == 0:
                self._logger.info(f"[{ROBOT_ID}] Trajectory complete op={op_id}")
            else:
                self._logger.warning(f"[{ROBOT_ID}] FJT error={ec} op={op_id}")
        else:
            self._logger.warning(f"[{ROBOT_ID}] Timeout waiting for op={op_id}")

        if hold_s > 0 and not self._estopped:
            time.sleep(hold_s)
        self._active_goal_handle = None
        self._current_task = None

    def _publish_status(self):
        """Publish robot status at 10 Hz."""
        from std_msgs.msg import String
        import numpy as np

        # Compute FK for EE position (simple FK using stored joints)
        ee_pos = self._compute_ee_position()

        msg = String()
        msg.data = json.dumps({
            "robot_ns":       ROBOT_ID,   # unique robot identity
            "namespace":      ROBOT_NS,
            "joint_positions": self._joint_positions,
            "ee_position":    ee_pos,
            "current_task":   self._current_task,
            "status":         "busy" if self._current_task else "idle",
            "timestamp":      time.time(),
        })
        self._status_pub.publish(msg)

    def _compute_ee_position(self) -> list[float]:
        """Simple FK via Pinocchio (non-blocking, best-effort)."""
        try:
            import pinocchio as pin
            import numpy as np

            from utils import arm_model

            # arm_model resolves the URDF (referenced in cell.yaml or baked under
            # urdf/) and attaches the tool frame this FK reports. Loading it here
            # by hand is how this status ended up reporting the flange position on
            # a cell whose URDF has no welding_tcp link.
            model, data, ee_id = arm_model.load(CELL_ID, ROBOT_ID)
            q     = np.zeros(model.nq)
            q[:min(6, model.nq)] = self._joint_positions[:model.nq]
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            p = data.oMf[ee_id].translation
            return [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
        except Exception:
            return [0.0, 0.0, 0.0]

    def spin(self):
        import rclpy
        rclpy.spin(self._node)

    def destroy(self):
        self._node.destroy_node()


def main():
    import rclpy
    rclpy.init()
    agent = RobotAgent()
    try:
        agent.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
