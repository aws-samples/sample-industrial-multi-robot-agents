"""
Cell state tools — publish cell configuration and monitor robot states.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.tools import tool

from agents.cell.robots_util import iter_robots


@tool
def get_all_robot_states(cell_id: str) -> dict:
    """
    Get current state (joint positions + EE position) for all robots in the cell.

    Reads latest status from /{cell_id}/{ns}/robot_status topics.

    Args:
        cell_id: Cell identifier

    Returns:
        {"robots": {ns: {"joint_positions": [...], "ee_position": [...], "status": str}}}
    """
    # (namespace, robot_id) pairs; states are keyed by robot_id.
    robots = [(ns, rid) for ns, rid, _cfg in iter_robots(cell_id)]

    states: dict = {}
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()

        node    = Node("cell_state_reader")
        pending = {rid for _ns, rid in robots}

        def make_cb(rid):
            def cb(msg):
                try:
                    states[rid] = json.loads(msg.data)
                    pending.discard(rid)
                except Exception:
                    pass
            return cb

        for ns, rid in robots:
            node.create_subscription(
                String, f"/{cell_id}/{ns}/{rid}/robot_status", make_cb(rid), 1
            )

        # Use a DEDICATED executor. The cell-controller process already spins the
        # global default executor in background daemon threads (ros2_listener,
        # ros2_joint_state_sub), so calling the global rclpy.spin_once() here
        # raises "Executor is already spinning". A private SingleThreadedExecutor
        # is fully independent and safe — same pattern as RosCellRuntime.
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            deadline = time.monotonic() + 2.0
            while pending and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.1)
        finally:
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()

    except ImportError:
        for _ns, rid in robots:
            states[rid] = {"status": "unknown", "note": "ROS 2 unavailable"}

    return {"robots": states}


@tool
def trigger_estop(cell_id: str, reason: str) -> dict:
    """
    Trigger emergency stop for all robots in the cell.

    Publishes an ESTOP command to /{cell_id}/estop.

    Args:
        cell_id: Cell identifier
        reason:  Human-readable reason for the estop

    Returns:
        {"triggered": bool, "cell_id": str, "reason": str}
    """
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()

        node = Node("cell_estop")
        pub  = node.create_publisher(String, f"/{cell_id}/estop", 1)
        msg  = String()
        msg.data = json.dumps({"reason": reason, "timestamp": time.time()})
        # Dedicated executor: the global default executor is already spinning in
        # the cell-controller's background threads, so rclpy.spin_once() on it
        # raises "Executor is already spinning". Spin a private executor briefly
        # to let the publisher establish connections before publishing.
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            for _ in range(5):
                executor.spin_once(timeout_sec=0.05)
            pub.publish(msg)
        finally:
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()
        return {"triggered": True, "cell_id": cell_id, "reason": reason}

    except ImportError:
        return {"triggered": True, "cell_id": cell_id, "reason": reason, "note": "ROS 2 unavailable"}
    except Exception as e:
        return {"triggered": False, "error": str(e)}
