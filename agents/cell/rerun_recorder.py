"""Rerun.io recorder for the cell-controller container.

Captures joint-state samples and trajectory dispatch events into Rerun `.rrd`
files keyed by execution session, enabling post-execution review in the Rerun
viewer and replay mode (re-dispatching previously recorded trajectories without
invoking the trajectory planning agent).

Usage::

    from agents.cell.rerun_recorder import get_recorder, set_recorder
    recorder = RerunRecorder()          # or NoOpRecorder() when rerun-sdk absent
    set_recorder(recorder)

    recorder.start_session(tree_id="W001", cell_id="welding_cell_2_small")
    recorder.log_dispatch_event(...)
    recorder.end_session()

If ``rerun-sdk`` is not installed a ``RuntimeWarning`` is emitted at import
time and all recorder calls become no-ops via the ``NoOpRecorder`` shim.
"""

from __future__ import annotations

import json
import os
import re as _re
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from utils import cell_layout

try:
    import rerun as rr
    _RERUN_AVAILABLE = True
except ImportError:
    rr = None  # type: ignore
    _RERUN_AVAILABLE = False
    warnings.warn(
        "rerun-sdk not installed — recording disabled. "
        "Install rerun-sdk>=0.21.0 for recording support.",
        RuntimeWarning,
        stacklevel=2,
    )


def list_recording_files(cell_id: str) -> dict:
    """The cell's ``.rrd`` recordings, newest first.

    Module-level, and used by BOTH recorders, because listing what is on the
    volume is a filesystem question and not a rerun-sdk question. NoOpRecorder
    used to answer this with an empty list plus a "recording disabled" warning,
    which meant a cell controller running without rerun-sdk reported no
    recordings while its predecessor's files sat in the same directory — the
    caller cannot replay what it is told does not exist.

    ``modified_s`` is the mtime (epoch seconds) and ``timestamp`` its ISO-8601
    rendering: the ordering key travels WITH the data, so a consumer does not
    have to stat the paths itself to sort them (which is what the dashboard did).

    Returns:
        {"recordings": [{"tree_id", "path", "size_bytes", "modified_s",
                         "timestamp"}]}
    """
    from datetime import datetime

    recordings_dir = cell_layout.recordings_dir(cell_id)
    if not recordings_dir.exists():
        return {"recordings": []}

    recordings = []
    for rrd_file in recordings_dir.glob("*.rrd"):
        try:
            stat = rrd_file.stat()
        except OSError:
            continue          # deleted between the glob and the stat
        recordings.append({
            "tree_id":    rrd_file.stem,
            "path":       str(rrd_file),
            "size_bytes": stat.st_size,
            "modified_s": round(stat.st_mtime, 3),
            "timestamp":  datetime.fromtimestamp(
                stat.st_mtime).isoformat(timespec="seconds"),
        })
    recordings.sort(key=lambda r: r["modified_s"], reverse=True)
    return {"recordings": recordings}


@dataclass
class DispatchEvent:
    """A single trajectory dispatch record written to and read from `.rrd` files.

    Stored as a JSON string inside a ``rr.TextLog`` entry at entity path
    ``operations/<operation_id>``.
    """

    operation_id:   str
    robot_ns:       str
    operation_type: str
    trajectory:     list[dict]
    phase:          str           # "dispatched" | "completed" | "failed"
    timestamp:      float
    error:          str = field(default="")


def _sanitise_filename(name: str) -> str:
    """Replace non-alphanumeric chars (except _ - .) with underscore."""
    return _re.sub(r"[^A-Za-z0-9_\-.]", "_", name)


class RerunRecorder:
    """Wraps the Rerun SDK for per-session recording of cell execution data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._save_path: str | None = None

    def start_session(self, tree_id: str, cell_id: str) -> None:
        """Initialise a new Rerun recording for one execution session.

        Saves to the cell's agent-owned ``recordings/{tree_id}.rrd``.
        Creates the directory if absent. Warns and overwrites if file exists.
        """
        if not _RERUN_AVAILABLE:
            return

        safe_name = _sanitise_filename(tree_id or "unknown")
        save_dir = cell_layout.recordings_dir(cell_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"{safe_name}.rrd")

        if Path(save_path).exists():
            print(
                f"[RerunRecorder] WARNING: overwriting existing recording {save_path}",
                flush=True,
            )

        self._save_path = save_path
        # Recording is OPTIONAL — it must never block execution. Modern rerun
        # uses rr.init(app_id) + rr.save(path) (the old save_to= kwarg is gone).
        try:
            rr.init(safe_name)
            rr.save(save_path)
            self._active = True
            print(f"[RerunRecorder] Session started: {save_path}", flush=True)
        except Exception as e:  # noqa: BLE001 - degrade gracefully, never raise
            self._active = False
            print(f"[RerunRecorder] disabled — could not start recording: {e}", flush=True)

    def end_session(self) -> None:
        """Flush and close the current recording."""
        if not _RERUN_AVAILABLE or not self._active:
            return
        with self._lock:
            # rerun-sdk flushes automatically when the process exits or rr.disconnect() is called
            try:
                rr.disconnect()
            except Exception:
                pass
            self._active = False
            self._save_path = None
        print("[RerunRecorder] Session ended.", flush=True)

    def log_joint_state(
        self,
        robot_ns: str,
        joint_positions: list[float],
        ee_position: list[float],
        timestamp: float | None = None,
    ) -> None:
        """Record joint positions and EE position for one robot at one timestep.

        Thread-safe — called from the ROS 2 joint-state subscriber daemon thread
        concurrently with dispatch event logging from the MCP event loop.

        Entity paths:
          - ``robots/<robot_ns>/joint_positions``  → ``rr.SeriesLine``
          - ``robots/<robot_ns>/ee_position``       → ``rr.Points3D``

        Args:
            robot_ns:        Robot namespace (e.g. "robot1").
            joint_positions: 6 joint angles in radians.
            ee_position:     [x, y, z] end-effector position in world frame.
            timestamp:       Unix epoch seconds. Falls back to ``time.time()`` if None.
        """
        if not _RERUN_AVAILABLE or not self._active:
            return

        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            rr.set_time_seconds("execution", ts)
            rr.log(
                f"robots/{robot_ns}/joint_positions",
                rr.SeriesLine(values=joint_positions[:6], name=robot_ns),
            )
            if len(ee_position) >= 3:
                rr.log(
                    f"robots/{robot_ns}/ee_position",
                    rr.Points3D([ee_position[:3]]),
                )


    def log_dispatch_event(
        self,
        operation_id: str,
        robot_ns: str,
        operation_type: str,
        trajectory: list[dict],
        phase: str,
        error: str = "",
    ) -> None:
        """Record a trajectory dispatch event (dispatched/completed/failed).

        Thread-safe. Logs a TextLog entry at entity path
        ``operations/<operation_id>`` containing a JSON-serialised DispatchEvent.

        The trajectory waypoints are embedded verbatim in the JSON so they can
        be recovered losslessly for replay mode.

        Args:
            operation_id:   Operation point ID (e.g. "W001").
            robot_ns:       Robot namespace.
            operation_type: "weld" | "rivet" | "glue" | "laser_cut" | "mark" | "move".
            trajectory:     Full waypoint list [{"positions": [...], "time_s": float}].
            phase:          "dispatched" | "completed" | "failed".
            error:          Error detail (non-empty only when phase=="failed").
        """
        if not _RERUN_AVAILABLE or not self._active:
            return

        ts = time.time()
        body = json.dumps({
            "operation_id":   operation_id,
            "robot_ns":       robot_ns,
            "operation_type": operation_type,
            "phase":          phase,
            "timestamp":      ts,
            "error":          error,
            "trajectory":     trajectory,
        })

        with self._lock:
            rr.set_time_seconds("execution", ts)
            rr.log(f"operations/{operation_id}", rr.TextLog(body))

    def list_recordings(self, cell_id: str) -> dict:
        """List all .rrd files in the recording store for a cell.

        Args:
            cell_id: Cell identifier.

        Returns:
            {"recordings": [{"tree_id": str, "path": str, "size_bytes": int,
                             "modified_s": float, "timestamp": str}]}
        """
        return list_recording_files(cell_id)

    def replay_execution(self, tree_id: str, cell_id: str) -> dict:
        """Load a .rrd recording and re-dispatch all recorded trajectories.

        Extracts DispatchEvent records (phase=="dispatched") from the TextLog
        entries under ``operations/**``, sorts by timestamp, then dispatches
        each via publish_task_assignment.

        Does NOT call the trajectory planning agent or any IK solver.

        Args:
            tree_id: Recording session identifier.
            cell_id: Cell identifier.

        Returns:
            {"success": True, "replayed_operations": int, "elapsed_s": float}
            or {"success": False, "error": str, ...}
        """
        import time as _time

        # Locate the recording file
        safe_name = _sanitise_filename(tree_id)
        rrd_path  = cell_layout.recording_file(cell_id, safe_name)

        if not rrd_path.exists():
            return {"success": False, "error": "recording not found", "tree_id": tree_id}

        # Extract dispatch events from the .rrd file
        events: list[dict] = []
        try:
            if _RERUN_AVAILABLE and hasattr(rr, "dataframe"):
                recording = rr.dataframe.load_recording(str(rrd_path))
                view  = recording.view(index="execution", contents="operations/**")
                table = view.select().read_all()
                for row in table.to_pylist():
                    text = row.get("text") or row.get("body") or ""
                    if text:
                        try:
                            evt = json.loads(text)
                            if evt.get("phase") == "dispatched":
                                events.append(evt)
                        except (json.JSONDecodeError, ValueError):
                            pass
            else:
                return {
                    "success": False,
                    "error":   "rr.dataframe API not available — upgrade rerun-sdk>=0.21.0",
                }
        except Exception as exc:
            return {"success": False, "error": f"failed to load recording: {exc}"}

        # Sort by timestamp to preserve original dispatch order
        events.sort(key=lambda e: e.get("timestamp", 0.0))

        if not events:
            return {"success": False, "error": "no dispatched events found in recording", "tree_id": tree_id}

        # Re-dispatch each event
        from agents.cell.tools.dispatch import publish_task_assignment

        t0              = _time.monotonic()
        replayed_count  = 0

        for evt in events:
            op_id  = evt.get("operation_id", "?")
            try:
                result = publish_task_assignment.invoke({
                    "cell_id":        cell_id,
                    "robot_ns":       evt["robot_ns"],
                    "operation_id":   op_id,
                    "operation_type": evt["operation_type"],
                    "trajectory":     evt["trajectory"],
                    "hold_duration_s": 0.5,
                    "velocity_scale":  1.0,
                })
                if not result.get("dispatched", True):
                    return {
                        "success":              False,
                        "error":                str(result),
                        "failed_operation_id":  op_id,
                    }
                replayed_count += 1
            except Exception as exc:
                return {
                    "success":              False,
                    "error":                str(exc),
                    "failed_operation_id":  op_id,
                }

        return {
            "success":            True,
            "replayed_operations": replayed_count,
            "elapsed_s":          round(_time.monotonic() - t0, 2),
        }


class NoOpRecorder:
    """Drop-in replacement for RerunRecorder when rerun-sdk is not installed.

    All methods are no-ops that return immediately without side effects.
    The cell agent operates normally; only recording is disabled.
    """

    def start_session(self, tree_id: str, cell_id: str) -> None:
        pass

    def log_joint_state(
        self,
        robot_ns: str,
        joint_positions: list,
        ee_position: list,
        timestamp: float | None = None,
    ) -> None:
        pass

    def log_dispatch_event(
        self,
        operation_id: str,
        robot_ns: str,
        operation_type: str,
        trajectory: list,
        phase: str,
        error: str = "",
    ) -> None:
        pass

    def end_session(self) -> None:
        pass

    def replay_execution(self, tree_id: str, cell_id: str) -> dict:
        return {"success": False, "error": "recording disabled (rerun-sdk not installed)"}

    def list_recordings(self, cell_id: str) -> dict:
        """What is on the volume, plus a note that nothing NEW is being written.

        Not an empty list: the files an earlier container recorded are still
        there and still replayable, and answering "none" would hide them.
        """
        return {**list_recording_files(cell_id),
                "warning": "recording disabled (rerun-sdk not installed) — "
                           "listing what is already on the volume"}


# ─── Module-level singleton ───────────────────────────────────────────────────

_NOOP_RECORDER = NoOpRecorder()
_recorder: "RerunRecorder | NoOpRecorder | None" = None


def get_recorder() -> "RerunRecorder | NoOpRecorder":
    """Return the active recorder, falling back to the NoOp shim if not set."""
    return _recorder if _recorder is not None else _NOOP_RECORDER


def set_recorder(r: "RerunRecorder | NoOpRecorder") -> None:
    """Register the active recorder singleton."""
    global _recorder
    _recorder = r
