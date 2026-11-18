# Feature: cell-controller-mcp-rerun-recording
# Tests: Property 1 (session path) + Property 2 (joint state entity paths)
#
# **Validates: Requirements 4.1, 4.2** (Property 1)
# **Validates: Requirements 5.1, 5.2, 5.4** (Property 2)

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Patch rerun before importing the recorder so _RERUN_AVAILABLE = True
rr_mock = MagicMock()
sys.modules.setdefault("rerun", rr_mock)

# Must import AFTER patching
import importlib

import agents.cell.rerun_recorder as _mod

importlib.reload(_mod)  # reload with the mock in place
from agents.cell.rerun_recorder import RerunRecorder, _sanitise_filename


# ─── Strategy helpers ─────────────────────────────────────────────────────────

safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-.",
    ),
    min_size=1,
    max_size=32,
)

joint_positions_st = st.lists(
    st.floats(min_value=-4.0, max_value=4.0, allow_nan=False),
    min_size=6,
    max_size=6,
)
ee_position_st = st.lists(
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    min_size=3,
    max_size=3,
)


# ─── Property 1: Session path construction is deterministic ──────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    cell_id=safe_text,
    tree_id=safe_text,
)
def test_session_path_is_deterministic(cell_id: str, tree_id: str, tmp_path):
    """Property 1: start_session saves to the expected deterministic path.

    **Validates: Requirements 4.1, 4.2**

    For any cell_id and tree_id, the recording save path SHALL equal
    {CELL_BASE_PATH}/{cell_id}/recordings/{sanitise(tree_id)}.rrd.
    """
    os.environ["CELL_BASE_PATH"] = str(tmp_path)

    recorder = RerunRecorder()

    with patch.object(_mod, "_RERUN_AVAILABLE", True), \
         patch.object(_mod, "rr") as mock_rr:
        mock_rr.init = MagicMock()
        recorder.start_session(tree_id=tree_id, cell_id=cell_id)

    # Expected path: {tmp_path}/{cell_id}/recordings/{sanitised_tree_id}.rrd
    safe = _sanitise_filename(tree_id)
    expected = str(tmp_path / cell_id / "recordings" / f"{safe}.rrd")

    assert recorder._save_path == expected, (
        f"Expected _save_path={expected!r}, got {recorder._save_path!r}"
    )


# ─── Property 2: Joint state entity paths follow the robot namespace ──────────

@settings(max_examples=100)
@given(
    robot_ns=safe_text,
    joint_positions=joint_positions_st,
    ee_position=ee_position_st,
    timestamp=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1e10, allow_nan=False),
    ),
)
def test_joint_state_entity_paths(
    robot_ns: str,
    joint_positions: list[float],
    ee_position: list[float],
    timestamp: float | None,
):
    """Property 2: log_joint_state logs to exactly robots/{ns}/joint_positions
    and robots/{ns}/ee_position — no other entity paths.

    **Validates: Requirements 5.1, 5.2, 5.4**

    For any robot_ns and any joint_positions / ee_position values,
    log_joint_state(robot_ns, ...) SHALL log to exactly the entity paths
    robots/{robot_ns}/joint_positions and robots/{robot_ns}/ee_position
    and to no other paths.
    """
    recorder = RerunRecorder()
    recorder._active = True  # skip start_session

    with patch.object(_mod, "_RERUN_AVAILABLE", True), \
         patch.object(_mod, "rr") as mock_rr:
        mock_rr.set_time_seconds = MagicMock()
        mock_rr.SeriesLine = MagicMock(return_value=MagicMock())
        mock_rr.Points3D = MagicMock(return_value=MagicMock())
        mock_rr.log = MagicMock()

        recorder.log_joint_state(robot_ns, joint_positions, ee_position, timestamp)

    logged_paths = [c.args[0] for c in mock_rr.log.call_args_list]

    assert f"robots/{robot_ns}/joint_positions" in logged_paths, (
        f"Expected 'robots/{robot_ns}/joint_positions' in {logged_paths}"
    )
    assert f"robots/{robot_ns}/ee_position" in logged_paths, (
        f"Expected 'robots/{robot_ns}/ee_position' in {logged_paths}"
    )
    # Must log to EXACTLY these two paths and no others
    assert set(logged_paths) == {
        f"robots/{robot_ns}/joint_positions",
        f"robots/{robot_ns}/ee_position",
    }, f"Unexpected entity paths logged: {logged_paths}"


# ─── Property 3: Dispatch event entity paths and payload are complete ─────────

trajectory_st = st.lists(
    st.fixed_dictionaries({
        "positions": st.lists(st.floats(min_value=-4.0, max_value=4.0, allow_nan=False), min_size=6, max_size=6),
        "time_s": st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    }),
    min_size=1,
    max_size=5,
)

phase_st = st.sampled_from(["dispatched", "completed", "failed"])


@settings(max_examples=100)
@given(
    operation_id=safe_text,
    robot_ns=safe_text,
    operation_type=safe_text,
    trajectory=trajectory_st,
    phase=phase_st,
    error=st.text(max_size=64),
)
def test_dispatch_event_entity_path_and_payload(
    operation_id, robot_ns, operation_type, trajectory, phase, error
):
    """Property 3: log_dispatch_event logs at exactly operations/{op_id}
    with a JSON body containing all required fields.

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """
    recorder = RerunRecorder()
    recorder._active = True

    with patch.object(_mod, "_RERUN_AVAILABLE", True), \
         patch.object(_mod, "rr") as mock_rr:
        mock_rr.set_time_seconds = MagicMock()
        mock_rr.TextLog = MagicMock(side_effect=lambda body: body)
        mock_rr.log = MagicMock()

        recorder.log_dispatch_event(
            operation_id=operation_id,
            robot_ns=robot_ns,
            operation_type=operation_type,
            trajectory=trajectory,
            phase=phase,
            error=error,
        )

    assert mock_rr.log.call_count == 1
    entity_path = mock_rr.log.call_args.args[0]
    assert entity_path == f"operations/{operation_id}", (
        f"Expected entity path 'operations/{operation_id}', got {entity_path!r}"
    )

    # The TextLog body should be valid JSON with all required fields
    log_body = mock_rr.log.call_args.args[1]  # the TextLog mock returns the body string
    payload = json.loads(log_body)

    required_fields = {"operation_id", "robot_ns", "operation_type", "phase", "timestamp", "error", "trajectory"}
    assert required_fields <= set(payload.keys()), (
        f"Missing fields: {required_fields - set(payload.keys())}"
    )
    assert payload["operation_id"]   == operation_id
    assert payload["robot_ns"]       == robot_ns
    assert payload["operation_type"] == operation_type
    assert payload["phase"]          == phase
    assert payload["trajectory"]     == trajectory


# ─── Property 4: Dispatch event recording round-trip ─────────────────────────

import json as _json_mod

@st.composite
def dispatch_event_strategy(draw):
    return {
        "operation_id":   draw(safe_text),
        "robot_ns":       draw(safe_text),
        "operation_type": draw(st.sampled_from(["weld", "rivet", "glue", "move"])),
        "trajectory":     draw(trajectory_st),
        "phase":          "dispatched",
        "timestamp":      draw(st.floats(min_value=1e9, max_value=2e9, allow_nan=False)),
        "error":          "",
    }


@settings(max_examples=100)
@given(events=st.lists(dispatch_event_strategy(), min_size=1, max_size=10, unique_by=lambda e: e["operation_id"]))
def test_dispatch_event_round_trip(events):
    """Property 4: DispatchEvents written via log_dispatch_event are fully
    recoverable with correct field values.

    Uses an in-memory list mock (not real .rrd) to test the
    serialisation/deserialisation round-trip in isolation.

    **Validates: Requirements 8.1, 8.2, 10.1, 10.2, 10.3**
    """
    # Capture what log_dispatch_event would write
    captured_bodies = []

    recorder = RerunRecorder()
    recorder._active = True

    with patch.object(_mod, "_RERUN_AVAILABLE", True), \
         patch.object(_mod, "rr") as mock_rr:
        mock_rr.set_time_seconds = MagicMock()
        mock_rr.TextLog = MagicMock(side_effect=lambda body: body)
        mock_rr.log = MagicMock(side_effect=lambda path, body: captured_bodies.append(body) if path.startswith("operations/") else None)

        for evt in events:
            recorder.log_dispatch_event(
                operation_id=evt["operation_id"],
                robot_ns=evt["robot_ns"],
                operation_type=evt["operation_type"],
                trajectory=evt["trajectory"],
                phase=evt["phase"],
                error=evt["error"],
            )

    # Deserialise (mimics what replay_execution does)
    recovered = [_json_mod.loads(body) for body in captured_bodies if body]
    dispatched = [r for r in recovered if r.get("phase") == "dispatched"]

    assert len(dispatched) == len(events), (
        f"Expected {len(events)} dispatched events, recovered {len(dispatched)}"
    )

    # Match recovered events to originals by operation_id
    # (log_dispatch_event assigns its own wall-clock timestamp, so we can't
    #  rely on timestamp order to correlate input <-> output)
    recovered_by_id = {r["operation_id"]: r for r in dispatched}

    for orig in events:
        op_id = orig["operation_id"]
        assert op_id in recovered_by_id, f"operation_id {op_id!r} not found in recovered events"
        rec = recovered_by_id[op_id]
        assert rec["robot_ns"]       == orig["robot_ns"]
        assert rec["operation_type"] == orig["operation_type"]
        assert rec["trajectory"]     == orig["trajectory"]


# ─── Property 5: list_recordings completeness ────────────────────────────────

@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n_files=st.integers(min_value=0, max_value=10),
    cell_id=safe_text,
)
def test_list_recordings_completeness(n_files: int, cell_id: str, tmp_path):
    """Property 5: list_recordings returns metadata for every .rrd file present.

    **Validates: Requirements 9.3**

    For any set of .rrd files present in the recording store, list_recordings
    SHALL return a list with the same count where each entry has non-empty
    tree_id, path, and non-negative size_bytes.
    """
    import tempfile

    # Use a fresh temp dir per hypothesis example to avoid cross-example pollution
    # (tmp_path is reused across examples when function_scoped_fixture is suppressed)
    with tempfile.TemporaryDirectory() as fresh_base:
        os.environ["CELL_BASE_PATH"] = fresh_base

        # Create N .rrd files in the fresh directory
        recordings_dir = Path(fresh_base) / cell_id / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        created_names = set()
        for i in range(n_files):
            name = f"session_{i}"
            (recordings_dir / f"{name}.rrd").write_bytes(b"RRD_FAKE_CONTENT")
            created_names.add(name)

        recorder = RerunRecorder()
        result = recorder.list_recordings(cell_id)

    assert "recordings" in result
    entries = result["recordings"]
    assert len(entries) == n_files, (
        f"Expected {n_files} recordings, got {len(entries)}"
    )

    for entry in entries:
        assert entry["tree_id"], "tree_id must be non-empty"
        assert entry["path"], "path must be non-empty"
        assert entry["size_bytes"] >= 0, "size_bytes must be non-negative"
        assert entry["tree_id"] in created_names, (
            f"Unexpected tree_id {entry['tree_id']!r}"
        )
