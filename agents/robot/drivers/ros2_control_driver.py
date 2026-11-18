"""
ROS 2 Control driver — sends FollowJointTrajectory goals to ros2_control.

Used by the robot agent for simulation (Isaac) and real hardware.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from agents.cell_defaults import default_cell_id
from agents.home_pose import HOME_JOINTS_FALLBACK, home_pose
from agents.robot.drivers.base_driver import BaseDriver


def _home_joints() -> list:
    """HOME for the arm this driver is configured to drive.

    Resolved per call rather than at import: the driver reads its identity from
    the environment, and this is the pose it will actually COMMAND the hardware
    to, so it has to be the one the cell declares and the planner validated.
    """
    return home_pose(default_cell_id(),
                     os.environ.get("ROBOT_ID", "robot1"),
                     HOME_JOINTS_FALLBACK)


class Ros2ControlDriver(BaseDriver):
    """Send FJT goals to a ros2_control FollowJointTrajectory action server."""

    def __init__(
        self,
        node,
        action_name: str,
        timeout_s: float = 60.0,
    ):
        """
        Args:
            node:        rclpy Node (shared with robot agent)
            action_name: Full action name e.g.
                         /welding_cell_2_small/robot1/joint_trajectory_controller/follow_joint_trajectory
            timeout_s:   Default execution timeout
        """
        import rclpy
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory

        self._node      = node
        self._timeout_s = timeout_s
        self._client    = ActionClient(node, FollowJointTrajectory, action_name)
        self._current_joints: list[float] = _home_joints()
        self._action_name = action_name

    @property
    def name(self) -> str:
        return f"ros2_control:{self._action_name}"

    def update_joints(self, joints: list[float]) -> None:
        """Called by robot agent's joint_states subscriber to keep joints current."""
        self._current_joints = list(joints[:6])

    def get_joint_positions(self) -> list[float]:
        return self._current_joints.copy()

    def execute_trajectory(
        self,
        trajectory: list[dict],
        joint_names: list[str],
        velocity_scale: float = 1.0,
    ) -> dict:
        import rclpy
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from control_msgs.action import FollowJointTrajectory
        from builtin_interfaces.msg import Duration

        if not self._client.wait_for_server(timeout_sec=5.0):
            return {"success": False, "elapsed_s": 0.0, "error": "Action server unavailable"}

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names

        from agents.robot.fjt_profile import point_velocities
        scale = max(velocity_scale, 0.01)
        timed = [{**pt, "time_s": float(pt["time_s"]) / scale,
                  **({"velocities": [float(v) * scale for v in pt["velocities"]]}
                     if pt.get("velocities") else {})} for pt in trajectory]
        # The planned profile's velocities, not [0]*6 at every point: fjt_profile.
        for pt, vel in zip(timed, point_velocities(timed, 6)):
            tpt = JointTrajectoryPoint()
            pos = list(pt["positions"][:6])
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

        t0     = time.monotonic()
        future = self._client.send_goal_async(goal)

        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)

        if not future.done() or not future.result().accepted:
            return {"success": False, "elapsed_s": round(time.monotonic() - t0, 2), "error": "Goal rejected"}

        result_future = future.result().get_result_async()
        expected_s    = float(trajectory[-1]["time_s"]) / max(velocity_scale, 0.01) + 5.0
        deadline      = time.monotonic() + expected_s

        while not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)

        elapsed = round(time.monotonic() - t0, 2)
        if not result_future.done():
            return {"success": False, "elapsed_s": elapsed, "error": "Execution timeout"}

        ec = result_future.result().result.error_code
        if ec == 0:
            return {"success": True, "elapsed_s": elapsed, "error": None}
        return {"success": False, "elapsed_s": elapsed, "error": f"FJT error_code={ec}"}

    def go_home(self) -> dict:
        # Use first joint_names from environment
        cell_id  = default_cell_id()
        robot_ns = os.environ.get("ROBOT_NAMESPACE", "arms")
        robot_id = os.environ.get("ROBOT_ID", "robot1")
        HOME_TRAJ = [
            {"positions": home_pose(cell_id, robot_id, HOME_JOINTS_FALLBACK),
             "time_s": 3.0},
        ]
        jnames   = [f"{cell_id}/{robot_ns}/{robot_id}/joint_{i}" for i in range(1, 7)]
        return self.execute_trajectory(HOME_TRAJ, jnames)
