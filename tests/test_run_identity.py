"""Round-trip properties of the run-id helpers in ``run_logger``.

The id is the anchor: the client polls it, the log is named after it, and the
concurrency registry keys on it. Every ``run_logger`` helper below is what lets
a poller find its run without being told which cell it belonged to — the tests
here fix that contract in place.

Host-runnable (no flask, no ROS; only ``json``/``datetime``/``pathlib``).
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from agents.orchestrator import run_logger


CELL = "welding_cell_2_small"


def test_new_run_id_shape() -> None:
    when = datetime(2026, 9, 4, 14, 12, 33)
    rid = run_logger.new_run_id(CELL, when=when)
    assert rid == "welding_cell_2_small_20260904_141233"


def test_new_run_id_defaults_to_now() -> None:
    # Only the shape is testable here, not the value.
    rid = run_logger.new_run_id(CELL)
    prefix, date, time_part = rid.rsplit("_", 2)
    assert prefix == CELL
    assert len(date) == 8 and date.isdigit()
    assert len(time_part) == 6 and time_part.isdigit()


@pytest.mark.parametrize("cell_id", [
    "welding_cell_2_small",
    "welding_cell_2_small",
    "simple",
    "a_b_c_d_e_f_g",       # many underscores in the cell id itself
])
def test_cell_id_of_run_recovers_the_cell(cell_id: str) -> None:
    """The whole point of the format: the id alone locates its own cell.
    Cell ids contain underscores, so splitting from the LEFT would silently
    keep only the first field and route a poll to the wrong cell."""
    rid = run_logger.new_run_id(cell_id)
    assert run_logger.cell_id_of_run(rid) == cell_id


def test_cell_id_of_run_returns_input_for_a_shape_it_cannot_parse() -> None:
    # A future format might not follow the two-suffix convention. Rather than
    # raise, the helper hands the input back so a downstream lookup still gets a
    # useful thing to try, and the caller can decide what to do with the miss.
    assert run_logger.cell_id_of_run("nope") == "nope"


def test_run_log_path_uses_the_cells_log_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    rid = "welding_cell_2_small_20260904_120000"
    path = run_logger.run_log_path(rid)
    assert path == tmp_path / "welding_cell_2_small" / "logs" / f"run_{rid}.json"


def test_read_run_log_returns_none_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    assert run_logger.read_run_log("welding_cell_2_small_20260904_120000") is None


def test_read_run_log_returns_none_for_a_truncated_log(tmp_path, monkeypatch) -> None:
    """A truncated log (a run killed mid-write) proves nothing about the run,
    so the honest answer is not-found rather than a half-parsed record."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    rid = "welding_cell_2_small_20260904_120000"
    path = run_logger.run_log_path(rid)
    path.parent.mkdir(parents=True)
    path.write_text('{"run": {"cell_id": "wel')       # invalid JSON
    assert run_logger.read_run_log(rid) is None


def test_read_run_log_round_trips_a_persisted_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    rid = "welding_cell_2_small_20260904_120000"
    path = run_logger.run_log_path(rid)
    path.parent.mkdir(parents=True)
    payload = {"run": {"run_id": rid, "cell_id": CELL, "success": True}}
    path.write_text(json.dumps(payload))
    assert run_logger.read_run_log(rid) == payload


def test_list_run_logs_returns_newest_first_with_metadata_only(
        tmp_path, monkeypatch) -> None:
    """The listing is a live view for a UI, not an archive dump: returning the
    full log per row would be megabytes per cell."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    log_dir = tmp_path / CELL / "logs"
    log_dir.mkdir(parents=True)
    for rid in (
        "welding_cell_2_small_20260904_120000",
        "welding_cell_2_small_20260904_130000",
        "welding_cell_2_small_20260904_140000",
    ):
        (log_dir / f"run_{rid}.json").write_text(json.dumps({
            "run": {"run_id": rid, "cell_id": CELL, "success": True},
            "reasoning_trace": [{"role": "ai", "content": "x" * 1000}],   # bulky
        }))
    listing = run_logger.list_run_logs(CELL)
    assert [row["run_id"] for row in listing] == [
        "welding_cell_2_small_20260904_140000",
        "welding_cell_2_small_20260904_130000",
        "welding_cell_2_small_20260904_120000",
    ]
    # metadata only — no reasoning trace fields leak in
    for row in listing:
        assert "reasoning_trace" not in row
        assert row["log_json"].endswith(".json")


def test_list_run_logs_empty_for_missing_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    assert run_logger.list_run_logs(CELL) == []


def test_list_run_logs_respects_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("RUN_LOG_DIR", raising=False)
    log_dir = tmp_path / CELL / "logs"
    log_dir.mkdir(parents=True)
    for hour in range(6):
        rid = f"welding_cell_2_small_20260904_{hour:02d}0000"
        (log_dir / f"run_{rid}.json").write_text(json.dumps(
            {"run": {"run_id": rid, "cell_id": CELL, "success": True}}))
    assert len(run_logger.list_run_logs(CELL, limit=3)) == 3
