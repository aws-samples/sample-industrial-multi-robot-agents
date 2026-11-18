"""
Cell Agent ROS 2 server.

A simple HTTP health/status endpoint, and the joint-state subscriber that feeds
the recorder. Behaviour trees arrive over the cell's MCP interface
(``execute_behaviour_tree``); the ``/{cell_id}/behaviour_tree`` topic this module
used to listen on had no publisher and is gone.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from agents.cell_defaults import default_cell_id

CELL_ID = default_cell_id()


def http_server():
    """Minimal HTTP health endpoint."""
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "cell_id": CELL_ID})

    port = int(os.environ.get("CELL_HTTP_PORT", "8891"))
    print(f"[CellAgent] HTTP health on 0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)


def ros2_joint_state_sub():
    """Subscribe to per-robot status topics and record joint states via Rerun.

    Reads robots.yaml to discover namespaces, then creates one ROS 2 subscriber
    per robot. Calls get_recorder().log_joint_state() for each incoming message.

    Designed to run as a daemon thread started by mcp_server.py.
    Returns silently if ROS 2 is unavailable (offline mode).
    """
    try:
        import json
        import yaml
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String

        from agents.cell.rerun_recorder import get_recorder
        from agents.cell.robots_util import iter_robots

        # The robot list comes from the cell-data owner (the orchestrator). With
        # planning and execution started separately, the execution side is often
        # up first: one failed read used to end this thread for good, so the
        # recorder stayed dead after the orchestrator arrived. Wait for it instead
        # — saying so once, then every minute.
        import time as _time
        from utils import cell_store
        waited_s = 0
        while True:
            try:
                robots = [(ns, rid) for ns, rid, _cfg in iter_robots(CELL_ID)]
                if waited_s:
                    print(f"[CellJointStateSub] cell data reachable after {waited_s} s",
                          flush=True)
                break
            except FileNotFoundError:
                print(f"[CellJointStateSub] robots.yaml not found for {CELL_ID}", flush=True)
                robots = []
                break
            except cell_store.CellStoreUnavailable as e:  # owner not up yet: retry
                if waited_s % 60 == 0:
                    print(f"[CellJointStateSub] waiting for the cell data ({e})", flush=True)
                _time.sleep(5)
                waited_s += 5

        if not robots:
            print("[CellJointStateSub] No robots found — subscriber not started", flush=True)
            return

        if not rclpy.ok():
            rclpy.init()

        node = Node("cell_joint_state_recorder")

        def make_cb(robot_id: str):
            def cb(msg):
                try:
                    data = json.loads(msg.data)
                    get_recorder().log_joint_state(
                        robot_ns=robot_id,
                        joint_positions=data.get("joint_positions", []),
                        ee_position=data.get("ee_position", [0.0, 0.0, 0.0]),
                        timestamp=data.get("timestamp"),
                    )
                except Exception:
                    pass
            return cb

        for ns, rid in robots:
            node.create_subscription(
                String,
                f"/{CELL_ID}/{ns}/{rid}/robot_status",
                make_cb(rid),
                10,
            )

        print(
            f"[CellJointStateSub] Listening on {len(robots)} robot_status topics "
            f"(cell={CELL_ID})",
            flush=True,
        )
        # Spin on a DEDICATED executor. ros2_listener() already spins the global
        # default executor (rclpy.spin), so calling rclpy.spin() here would raise
        # "Executor is already spinning" and kill this recorder thread — leaving
        # .rrd recordings without live joint-state data. A private
        # SingleThreadedExecutor is independent (same pattern as RosCellRuntime).
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()

    except ImportError:
        print("[CellJointStateSub] ROS 2 not available — joint state recording disabled", flush=True)
    except Exception as e:
        print(f"[CellJointStateSub] Error: {e}", flush=True)


if __name__ == "__main__":
    t1 = threading.Thread(target=ros2_listener, daemon=True)
    t1.start()
    http_server()
