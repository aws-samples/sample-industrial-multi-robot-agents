"""Driver factory — selects the execution backend from environment."""
from __future__ import annotations

import os


def create_driver(node, cell_id: str, robot_ns: str, robot_id: str, ctrl_name: str):
    """
    Create the appropriate execution driver.

    ROBOT_BACKEND env var:
      ros2_control (default) — FJT via ros2_control action server
      sim_only               — prints trajectory, no ROS (for unit tests)

    Args:
        node:      rclpy Node
        cell_id:   Cell identifier
        robot_ns:  Robot namespace
        robot_id:  Robot model ID
        ctrl_name: Controller name

    Returns:
        BaseDriver instance
    """
    backend = os.environ.get("ROBOT_BACKEND", "ros2_control").lower()

    if backend == "ros2_control":
        from agents.robot.drivers.ros2_control_driver import Ros2ControlDriver
        action_name = (
            f"/{cell_id}/{robot_ns}/{ctrl_name}/follow_joint_trajectory"
        )
        return Ros2ControlDriver(node, action_name)

    if backend == "sim_only":
        from agents.robot.drivers.sim_only_driver import SimOnlyDriver
        return SimOnlyDriver()

    raise ValueError(f"Unknown ROBOT_BACKEND: {backend!r}")
