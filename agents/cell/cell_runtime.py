"""
Cell runtime — the deterministic execution substrate for the cell controller.

This is the thin ROS 2 layer the deterministic BehaviourTree executor and the
collision monitor sit on top of. It owns ONE ROS node for the whole cell and:

  * caches the latest ``robot_status`` for every robot (joint + EE positions),
  * dispatches trajectories to robot agents (``dispatch_subtask``),
  * publishes cell-wide e-stop.

The executor and monitor talk to the runtime through a small duck-typed
interface (``now``, ``robots``, ``latest_status``, ``dispatch``, ``estop``,
``clear_estop``) so they can be unit-tested against a fake runtime with no ROS
or Pinocchio installed. ``RosCellRuntime`` is the live implementation;
``build_runtime`` returns it or raises a clear error when ROS is unavailable.

Topic conventions (all ``std_msgs/String`` carrying JSON, matching the existing
robot agent + dispatch tools):
  PUB /{cell}/{ns}/{rid}/dispatch_subtask   — trajectory task for a robot
  SUB /{cell}/{ns}/{rid}/robot_status       — {joint_positions, ee_position, ...}
  PUB /{cell}/estop                         — {reason} | {clear: true}
"""
from __future__ import annotations

import json
import threading
import time
from typing import Optional


class RosCellRuntime:
    """Live ROS 2 runtime shared by the executor and the collision monitor."""

    def __init__(self, cell_id: str):
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String

        from agents.cell.robots_util import iter_robots

        self._cell_id = cell_id
        self._String = String

        # (namespace, robot_id) pairs; everything is keyed by the unique robot_id.
        self._robots: list[tuple[str, str]] = [
            (ns, rid) for ns, rid, _cfg in iter_robots(cell_id)
        ]

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = Node("cell_runtime")

        self._status_lock = threading.Lock()
        self._status: dict[str, dict] = {}          # robot_id -> latest status dict
        self._dispatch_pubs: dict[str, object] = {}  # robot_id -> publisher

        for ns, rid in self._robots:
            addr = f"{cell_id}/{ns}/{rid}"
            self._dispatch_pubs[rid] = self._node.create_publisher(
                String, f"/{addr}/dispatch_subtask", 10
            )
            self._node.create_subscription(
                String, f"/{addr}/robot_status", self._make_status_cb(rid), 10
            )

        self._estop_pub = self._node.create_publisher(String, f"/{cell_id}/estop", 1)

        # Spin the node on a DEDICATED executor in a background thread. The cell
        # controller process already spins the global default executor elsewhere
        # (mcp_server's joint_state recorder + the cell agent listener), so
        # calling the global rclpy.spin_once() here raises
        # "Executor is already spinning" and kills this thread — which would stop
        # robot_status from ever reaching the executor and make every position
        # gate time out. Our own SingleThreadedExecutor is fully independent.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Spin the node in a background thread so status callbacks keep flowing
        # while the executor runs its (blocking) walk on the main thread.
        self._spin_stop = False
        self._spin_thread = threading.Thread(target=self._spin, daemon=True, name="cell-runtime-spin")
        self._spin_thread.start()

    # ── ROS plumbing ──────────────────────────────────────────────────────────
    def _make_status_cb(self, robot_id: str):
        def _cb(msg):
            try:
                data = json.loads(msg.data)
            except Exception:
                return
            data.setdefault("_recv_t", time.time())
            with self._status_lock:
                self._status[robot_id] = data
        return _cb

    def _spin(self):
        while not self._spin_stop and self._rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                # Never let a transient spin error kill the thread silently;
                # a dead spin thread stalls all robot_status updates.
                time.sleep(0.05)

    # ── Executor / monitor interface ───────────────────────────────────────────
    def now(self) -> float:
        return time.monotonic()

    def robots(self) -> list[tuple[str, str]]:
        return list(self._robots)

    def latest_status(self, robot_id: str) -> Optional[dict]:
        with self._status_lock:
            s = self._status.get(robot_id)
            return dict(s) if s is not None else None

    def dispatch(self, robot_id: str, task: dict) -> None:
        pub = self._dispatch_pubs.get(robot_id)
        if pub is None:
            raise KeyError(f"unknown robot_id '{robot_id}' (known: {list(self._dispatch_pubs)})")
        msg = self._String()
        msg.data = json.dumps(task)
        pub.publish(msg)

    def estop(self, reason: str) -> None:
        msg = self._String()
        msg.data = json.dumps({"reason": reason, "timestamp": time.time()})
        self._estop_pub.publish(msg)

    def clear_estop(self, reason: str = "cleared") -> None:
        msg = self._String()
        msg.data = json.dumps({"clear": True, "reason": reason, "timestamp": time.time()})
        self._estop_pub.publish(msg)

    def shutdown(self) -> None:
        self._spin_stop = True
        try:
            self._spin_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass


def build_runtime(cell_id: str):
    """Return a live ``RosCellRuntime`` or raise RuntimeError if ROS is missing.

    The deterministic executor requires a real runtime to move robots; callers
    should surface the RuntimeError as an execution failure rather than silently
    "succeeding" with no robots moving.
    """
    try:
        return RosCellRuntime(cell_id)
    except ImportError as exc:
        raise RuntimeError(
            f"ROS 2 not available — cannot build cell runtime for '{cell_id}': {exc}"
        ) from exc
