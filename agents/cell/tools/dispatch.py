"""
Dispatch tools — assign tasks to robots and check execution gates.

The cell agent uses these to translate BehaviourTree EXECUTE nodes into
ROS 2 action goals dispatched to the robot agents.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from langchain_core.tools import tool

from agents.cell.rerun_recorder import get_recorder
from agents.cell.robots_util import robot_topic_prefix


@tool
def publish_task_assignment(
    cell_id: str,
    robot_ns: str,
    operation_id: str,
    operation_type: str,
    trajectory: list[dict],
    start_delay_s: float = 0.0,
    hold_duration_s: float = 0.5,
    velocity_scale: float = 1.0,
) -> dict:
    """
    Dispatch a trajectory to a robot agent via ROS 2 action.

    Publishes a TaskAssignment message to:
    /{cell_id}/{robot_ns}/dispatch_subtask

    Args:
        cell_id:         Cell identifier
        robot_ns:        Robot namespace (e.g. "robot1")
        operation_id:    Operation ID from operations.yaml
        operation_type:  "weld" | "rivet" | "glue" | "move" etc.
        trajectory:      [{"positions": [j1..j6], "time_s": float}]
        start_delay_s:   Seconds to wait before the robot starts
        hold_duration_s: Seconds to hold at target
        velocity_scale:  Speed scaling

    Returns:
        {"dispatched": bool, "robot_ns": str, "operation_id": str}
    """
    recorder = get_recorder()
    recorder.log_dispatch_event(
        operation_id=operation_id,
        robot_ns=robot_ns,
        operation_type=operation_type,
        trajectory=trajectory,
        phase="dispatched",
    )

    if start_delay_s > 0:
        time.sleep(start_delay_s)

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient

        if not rclpy.ok():
            rclpy.init()

        # Simplified dispatch — publish to action topic
        # Full implementation would use the ExecuteOperation action
        node = Node(f"cell_dispatch_{robot_ns}")
        topic = f"/{robot_topic_prefix(cell_id, robot_ns)}/dispatch_subtask"

        # Build a simple JSON task message over std_msgs/String
        from std_msgs.msg import String
        pub = node.create_publisher(String, topic, 1)
        msg = String()
        import json
        msg.data = json.dumps({
            "operation_id":   operation_id,
            "operation_type": operation_type,
            "trajectory":     trajectory,
            "hold_duration_s": hold_duration_s,
            "velocity_scale": velocity_scale,
        })
        # Small spin to let the publisher register
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.1)
        pub.publish(msg)
        node.destroy_node()
        result = {"dispatched": True, "robot_ns": robot_ns, "operation_id": operation_id}

        recorder.log_dispatch_event(
            operation_id=operation_id,
            robot_ns=robot_ns,
            operation_type=operation_type,
            trajectory=trajectory,
            phase="completed",
        )
        return result

    except ImportError:
        result = {
            "dispatched": True,
            "robot_ns":   robot_ns,
            "operation_id": operation_id,
            "note": "ROS 2 not available — simulated dispatch",
        }
        recorder.log_dispatch_event(
            operation_id=operation_id,
            robot_ns=robot_ns,
            operation_type=operation_type,
            trajectory=trajectory,
            phase="completed",
        )
        return result

    except Exception as e:
        recorder.log_dispatch_event(
            operation_id=operation_id,
            robot_ns=robot_ns,
            operation_type=operation_type,
            trajectory=trajectory,
            phase="failed",
            error=str(e),
        )
        raise


@tool
def check_position_gate(
    cell_id: str,
    robot_ns: str,
    expected_position: list[float],
    tolerance_m: float = 0.05,
    timeout_s: float = 30.0,
) -> dict:
    """
    Wait until a robot reaches an expected Cartesian position (interlock gate).

    Polls /{cell_id}/{robot_ns}/robot_status for EE position convergence.

    Args:
        cell_id:           Cell identifier
        robot_ns:          Robot namespace
        expected_position: [x, y, z] expected EE position
        tolerance_m:       Acceptable position error (metres)
        timeout_s:         Max wait time

    Returns:
        {"reached": bool, "elapsed_s": float, "actual_position": [...]}
    """
    import numpy as np

    t0  = time.monotonic()
    exp = np.array(expected_position[:3])

    try:
        import rclpy
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()

        node     = Node(f"gate_checker_{robot_ns}")
        last_pos = None

        from std_msgs.msg import String
        import json

        def cb(msg):
            nonlocal last_pos
            try:
                data = json.loads(msg.data)
                last_pos = data.get("ee_position")
            except Exception:
                pass

        node.create_subscription(
            String, f"/{robot_topic_prefix(cell_id, robot_ns)}/robot_status", cb, 10
        )

        while time.monotonic() - t0 < timeout_s:
            rclpy.spin_once(node, timeout_sec=0.2)
            if last_pos is not None:
                actual = np.array(last_pos[:3])
                if float(np.linalg.norm(actual - exp)) <= tolerance_m:
                    node.destroy_node()
                    return {
                        "reached":         True,
                        "elapsed_s":       round(time.monotonic() - t0, 2),
                        "actual_position": last_pos,
                    }

        node.destroy_node()
        return {
            "reached":   False,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "timeout":   True,
        }

    except ImportError:
        return {
            "reached":   True,
            "elapsed_s": 0.1,
            "note":      "ROS 2 not available — simulated gate pass",
        }
    except Exception as e:
        return {"reached": False, "error": str(e)}
