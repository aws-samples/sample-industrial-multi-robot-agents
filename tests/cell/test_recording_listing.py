"""Listing ``.rrd`` recordings is a filesystem question, not a rerun-sdk one.

``list_recording_files`` is module-level in agents/cell/rerun_recorder.py and is
used by BOTH recorders on purpose. ``NoOpRecorder`` used to answer with an empty
list plus a "recording disabled" warning, so a cell controller running without
rerun-sdk reported no recordings while its predecessor's files sat in the same
directory — and a caller cannot replay what it is told does not exist. The
dashboard now gets its recording list from this, through the cell MCP server,
instead of globbing the volume itself, so the ordering and the metadata have to
come WITH the answer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# rerun is optional in this repo; the listing must not depend on it either way.
sys.modules.setdefault("rerun", MagicMock())

from agents.cell.rerun_recorder import (  # noqa: E402
    NoOpRecorder,
    RerunRecorder,
    list_recording_files,
)

CELL = "welding_cell_2_small"


@pytest.fixture
def recordings(tmp_path, monkeypatch):
    """Create ``.rrd`` files with controlled mtimes; return the directory."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    d = tmp_path / CELL / "recordings"
    d.mkdir(parents=True)

    def _make(name: str, body: bytes = b"x", mtime: float | None = None) -> Path:
        p = d / name
        p.write_bytes(body)
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    _make.dir = d          # type: ignore[attr-defined]
    return _make


# ── what the listing contains ────────────────────────────────────────────────

class TestItListsWhatIsOnTheVolume:

    def test_a_missing_directory_is_an_empty_list_not_an_error(self, tmp_path, monkeypatch):
        """A cell that has never executed has no recordings — that is not a fault."""
        monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))

        assert list_recording_files(CELL) == {"recordings": []}

    def test_every_rrd_file_is_reported(self, recordings):
        recordings("bt_a.rrd")
        recordings("bt_b.rrd")
        recordings("bt_c.rrd")

        got = list_recording_files(CELL)["recordings"]

        assert {r["tree_id"] for r in got} == {"bt_a", "bt_b", "bt_c"}

    def test_the_tree_id_is_the_stem_and_the_path_is_absolute(self, recordings):
        p = recordings("bt_1788893504_welding_cell_2_small.rrd")

        entry = list_recording_files(CELL)["recordings"][0]

        assert entry["tree_id"] == "bt_1788893504_welding_cell_2_small"
        assert entry["path"] == str(p)
        assert Path(entry["path"]).is_absolute()

    def test_non_rrd_files_are_ignored(self, recordings):
        """The plans directory used to be globbed for `plan_*.json` by mistake."""
        recordings("bt_a.rrd")
        recordings("notes.txt")
        recordings("plan_123.json")

        got = list_recording_files(CELL)["recordings"]

        assert [r["tree_id"] for r in got] == ["bt_a"]

    def test_the_size_is_the_real_byte_count(self, recordings):
        recordings("bt_a.rrd", body=b"0123456789")

        assert list_recording_files(CELL)["recordings"][0]["size_bytes"] == 10

    def test_an_empty_recording_is_still_listed(self, recordings):
        """A zero-byte .rrd means a session that started and wrote nothing.

        Hiding it would make an aborted run look like a run that never happened.
        """
        recordings("bt_aborted.rrd", body=b"")

        got = list_recording_files(CELL)["recordings"]

        assert [r["tree_id"] for r in got] == ["bt_aborted"]
        assert got[0]["size_bytes"] == 0


class TestTheOrderingKeyTravelsWithTheData:
    """The consumer must not have to stat the paths to sort them — the dashboard
    did exactly that, from a container that may not even see the same mounts."""

    def test_recordings_are_newest_first(self, recordings):
        recordings("old.rrd",    mtime=1_000_000.0)
        recordings("newest.rrd", mtime=3_000_000.0)
        recordings("middle.rrd", mtime=2_000_000.0)

        got = list_recording_files(CELL)["recordings"]

        assert [r["tree_id"] for r in got] == ["newest", "middle", "old"]

    def test_modified_s_is_the_mtime(self, recordings):
        recordings("bt_a.rrd", mtime=1_700_000_000.0)

        assert list_recording_files(CELL)["recordings"][0]["modified_s"] == 1_700_000_000.0

    def test_timestamp_is_an_iso_rendering_of_the_same_instant(self, recordings):
        from datetime import datetime

        recordings("bt_a.rrd", mtime=1_700_000_000.0)

        entry = list_recording_files(CELL)["recordings"][0]
        assert entry["timestamp"] == datetime.fromtimestamp(
            entry["modified_s"]).isoformat(timespec="seconds")

    def test_every_entry_carries_the_full_metadata_set(self, recordings):
        recordings("bt_a.rrd")
        recordings("bt_b.rrd")

        for entry in list_recording_files(CELL)["recordings"]:
            assert set(entry) >= {"tree_id", "path", "size_bytes",
                                  "modified_s", "timestamp"}


# ── both recorders answer the same question ──────────────────────────────────

class TestBothRecordersReportTheSameFiles:

    def test_the_noop_recorder_lists_files_that_exist(self, recordings):
        """The regression this guards: "no recordings" with recordings on disk."""
        recordings("bt_a.rrd")
        recordings("bt_b.rrd")

        got = NoOpRecorder().list_recordings(CELL)["recordings"]

        assert {r["tree_id"] for r in got} == {"bt_a", "bt_b"}

    def test_the_noop_recorder_still_says_nothing_new_is_being_written(self, recordings):
        recordings("bt_a.rrd")

        result = NoOpRecorder().list_recordings(CELL)

        assert "recording disabled" in result["warning"]

    def test_the_two_recorders_agree_file_for_file(self, recordings):
        recordings("bt_a.rrd", mtime=2_000_000.0)
        recordings("bt_b.rrd", mtime=1_000_000.0)

        real = RerunRecorder().list_recordings(CELL)["recordings"]
        noop = NoOpRecorder().list_recordings(CELL)["recordings"]

        assert real == noop

    def test_the_warning_is_the_only_difference(self, recordings):
        recordings("bt_a.rrd")

        real = RerunRecorder().list_recordings(CELL)
        noop = NoOpRecorder().list_recordings(CELL)

        assert set(noop) - set(real) == {"warning"}
