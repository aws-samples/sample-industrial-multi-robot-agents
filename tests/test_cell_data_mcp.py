"""The owner's side of the seam: the orchestrator's cell-data MCP.

The planner asks; this answers. What matters about the answers is not that they
carry the right bytes — :mod:`utils.cell_store` is the same code on both sides for
that — but that they carry the right *distinctions* over a wire:

* ``found=false`` (the artefact is not there) and ``success=false`` (I would not,
  or could not, look) must never arrive as the same sentence. The caller degrades
  on the first and refuses on the second, and the file most often asked for is
  ``obstacles.yaml``.
* A refusal names itself. An empty dict would read as an empty cell.
* Reads resolve on **this** process's disk regardless of ``CELL_DATA_MODE``: an
  orchestrator that respected the mode while misconfigured to ``orchestrator``
  would call itself.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agents.orchestrator import cell_data_mcp as api
from utils import cell_layout, cell_store


@pytest.fixture(autouse=True)
def _owned_cells(tmp_path, monkeypatch):
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("CELL_DATA_MODE", raising=False)
    cell_store.clear_cache()
    yield
    cell_store.clear_cache()


def _tools() -> dict:
    """The tools as the MCP registry advertises them, keyed by name."""
    loop = asyncio.new_event_loop()
    try:
        return {t.name: t for t in loop.run_until_complete(api.mcp.list_tools())}
    finally:
        loop.close()


# ── the surface ──────────────────────────────────────────────────────────────

def test_the_tools_the_planner_calls_are_all_registered():
    """cell_store names these five verbs plus the summariser; if one is missing
    from the registry the planner's call fails at runtime, not at import."""
    assert set(_tools()) == {"cell_read", "cell_write", "cell_append", "cell_list",
                             "cell_summarize", "cell_delete", "cell_data_health"}


def test_every_tool_documents_itself_for_a_model_that_has_to_choose():
    for name, tool in _tools().items():
        assert (tool.description or "").strip(), f"{name} has no description"


def test_the_scene_bundle_is_not_reachable_through_any_tool():
    """URDF/SRDF/meshes stay a read-only mount — mesh URIs are resolved outside
    this repo and a cell is up to ~1 GB."""
    assert api.cell_read("c1", "scene", "robot.urdf")["success"] is False
    assert api.cell_read("c1", "meshes", "base.stl")["success"] is False


# ── read: absent, present, refused ───────────────────────────────────────────

def test_reading_an_absent_artefact_is_a_success_that_says_not_found():
    answer = api.cell_read("c1", "plans", "bt_nope.json")
    assert answer["success"] is True
    assert answer["found"] is False
    assert answer["text"] is None


def test_reading_an_existing_artefact_returns_its_bytes_verbatim():
    cell_store.write_text_local("c1", "config", "obstacles.yaml",
                                "obstacles:\n  - id: table\n")
    answer = api.cell_read("c1", "config", "obstacles.yaml")
    assert answer["found"] is True
    assert answer["text"] == "obstacles:\n  - id: table\n"


@pytest.mark.parametrize("name", ["../../etc/passwd", "a/b.yaml", "..", ""])
def test_a_traversing_name_is_refused_and_the_refusal_is_not_a_not_found(name):
    """The dangerous confusion: a refused request that answered ``found=false``
    would tell the planner the cell has no obstacle model."""
    answer = api.cell_read("c1", "config", name)
    assert answer["success"] is False
    assert answer.get("error")
    assert "found" not in answer


def test_the_manifest_kind_holds_only_the_manifest():
    assert api.cell_read("c1", "manifest", cell_layout.CELL_MANIFEST)["success"]
    assert api.cell_read("c1", "manifest", "robots.yaml")["success"] is False


def test_a_read_uses_this_processs_disk_even_if_the_mode_says_orchestrator(
        monkeypatch):
    """Otherwise a misconfigured orchestrator — or a second one pointed at this
    one — calls itself and both wait for the timeout."""
    cell_store.write_text_local("c1", "plans", "bt_1_c1.json", '{"ok":true}')
    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://127.0.0.1:1/mcp")
    answer = api.cell_read("c1", "plans", "bt_1_c1.json")
    assert answer["found"] is True and answer["text"] == '{"ok":true}'


# ── write and append ─────────────────────────────────────────────────────────

def test_a_write_lands_where_the_layout_says_and_reads_back():
    answer = api.cell_write("c1", "plans", "bt_1_c1.json", '{"tree_id":"bt_1_c1"}')
    assert answer["written"] is True and answer["bytes"] == len('{"tree_id":"bt_1_c1"}')
    assert cell_layout.plan_file("c1", "bt_1_c1").is_file()
    assert api.cell_read("c1", "plans", "bt_1_c1.json")["text"] == \
        '{"tree_id":"bt_1_c1"}'


def test_a_write_leaves_no_temp_file_behind():
    api.cell_write("c1", "plans", "bt_1_c1.json", "{}")
    names = [e["name"] for e in api.cell_list("c1", "plans")["entries"]]
    assert names == ["bt_1_c1.json"], names


def test_a_refused_write_writes_nothing():
    answer = api.cell_write("c1", "plans", "../escape.json", "{}")
    assert answer["success"] is False
    assert not (cell_layout.cells_root().parent / "escape.json").exists()


def test_append_builds_the_event_stream_line_by_line():
    """The plan-event stream is appended to while the run is in flight; a
    read-modify-write would lose events to a concurrent writer."""
    for seq in range(3):
        assert api.cell_append("c1", "logs", "planevents_t1.jsonl",
                               json.dumps({"seq": seq}) + "\n")["appended"]
    text = api.cell_read("c1", "logs", "planevents_t1.jsonl")["text"]
    assert [json.loads(line)["seq"] for line in text.splitlines()] == [0, 1, 2]


def test_append_creates_the_artefact_when_it_is_the_first_event():
    api.cell_append("c1", "logs", "planevents_t2.jsonl", "{}\n")
    assert api.cell_read("c1", "logs", "planevents_t2.jsonl")["found"] is True


# ── list and summarize ───────────────────────────────────────────────────────

def test_a_listing_is_metadata_only():
    api.cell_write("c1", "plans", "bt_1_c1.json",
                   json.dumps({"waypoints": list(range(2000))}))
    entry = api.cell_list("c1", "plans")["entries"][0]
    assert set(entry) == {"name", "size_bytes", "mtime_s", "path"}


def test_a_listing_can_be_filtered_by_prefix_and_suffix():
    api.cell_write("c1", "logs", "plantrace_a.json", "{}")
    api.cell_write("c1", "logs", "planevents_a.jsonl", "{}\n")
    traces = api.cell_list("c1", "logs", prefix="plantrace_", suffix=".json")
    assert [e["name"] for e in traces["entries"]] == ["plantrace_a.json"]
    assert traces["count"] == 1


def test_listing_a_config_kind_is_refused_rather_than_answered_wrongly():
    """config and manifest have a read/write asymmetry, so a directory listing
    would report copies a reader would never resolve to."""
    assert api.cell_list("c1", "config")["success"] is False
    assert api.cell_list("c1", "manifest")["success"] is False


def test_listing_a_cell_that_has_nothing_yet_is_empty_and_successful():
    answer = api.cell_list("c1", "plans")
    assert answer["success"] is True and answer["entries"] == []


def test_summarize_transports_the_named_fields_and_not_the_artefact():
    """The reason this tool exists: a plan carries every waypoint it computed, and
    a plan list wants none of them."""
    api.cell_write("c1", "plans", "bt_1_c1.json", json.dumps(
        {"tree_id": "bt_1_c1", "status": "ok", "waypoints": list(range(400))}))
    row = api.cell_summarize("c1", "plans", ["tree_id", "status"],
                             count_fields=["waypoints"])["entries"][0]
    assert row["tree_id"] == "bt_1_c1" and row["status"] == "ok"
    assert row["waypoints_count"] == 400
    assert "waypoints" not in row


def test_summarize_reaches_a_nested_field_the_caller_names():
    api.cell_write("c1", "logs", "plantrace_t1.json",
                   json.dumps({"run": {"trace_id": "t1", "success": False}}))
    row = api.cell_summarize("c1", "logs", ["run.trace_id", "run.success"],
                             prefix="plantrace_")["entries"][0]
    assert row["run.trace_id"] == "t1" and row["run.success"] is False


def test_summarize_skips_an_unreadable_artefact_instead_of_failing_the_listing():
    """One corrupt trace must not make the run history unopenable."""
    api.cell_write("c1", "logs", "plantrace_good.json", json.dumps({"run": "good"}))
    api.cell_write("c1", "logs", "plantrace_bad.json", "{ truncated")
    rows = api.cell_summarize("c1", "logs", ["run"], prefix="plantrace_")
    assert rows["success"] is True
    assert [r["run"] for r in rows["entries"]] == ["good"]


def test_summarize_asks_for_no_fields_and_still_answers_with_metadata():
    api.cell_write("c1", "plans", "bt_1_c1.json", "{}")
    row = api.cell_summarize("c1", "plans", [])["entries"][0]
    assert row["name"] == "bt_1_c1.json"


# ── delete ───────────────────────────────────────────────────────────────────

def test_delete_removes_it_and_says_so():
    api.cell_write("c1", "plans", "bt_1_c1.json", "{}")
    assert api.cell_delete("c1", "plans", "bt_1_c1.json")["deleted"] is True
    assert api.cell_read("c1", "plans", "bt_1_c1.json")["found"] is False


def test_deleting_what_is_not_there_is_a_success_that_deleted_nothing():
    answer = api.cell_delete("c1", "plans", "bt_gone.json")
    assert answer["success"] is True and answer["deleted"] is False


def test_delete_refuses_a_traversing_name():
    assert api.cell_delete("c1", "plans", "../../robots.yaml")["success"] is False


# ── health ───────────────────────────────────────────────────────────────────

def test_health_tells_a_puzzled_planner_where_this_owner_keeps_cell_data():
    answer = api.cell_data_health()
    assert answer["success"] is True
    assert answer["service"] == "cell-data"
    assert answer["cells_root"] == str(cell_layout.cells_root())
    assert answer["kinds"] == list(cell_store.KINDS)


# ── the two sides of the wire agree ──────────────────────────────────────────

def test_the_client_and_the_server_speak_the_same_field_names(monkeypatch):
    """cell_store reads ``found``/``text``/``written``/``appended``/``entries``
    off these replies. Wire them together in-process: a rename on either side
    would otherwise show up only in a container."""
    def _dispatch(request, timeout=None):
        payload = json.loads(request.data.decode())
        tool = getattr(api, payload["params"]["name"])
        result = tool(**payload["params"]["arguments"])

        class _Response:
            def read(self):
                return json.dumps({"result": {"content": [
                    {"type": "text", "text": json.dumps(result)}]}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Response()

    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://orchestrator:8893/mcp")
    monkeypatch.setattr(cell_store, "_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(cell_store.urllib.request, "urlopen", _dispatch)

    assert cell_store.write_text("c1", "plans", "bt_1_c1.json", '{"a":1}') is True
    assert cell_store.read_text("c1", "plans", "bt_1_c1.json") == '{"a":1}'
    assert cell_store.read_text("c1", "plans", "bt_none.json") is None
    assert cell_store.append_text("c1", "logs", "planevents_t1.jsonl", "x\n") is True
    assert [e["name"] for e in cell_store.list_entries("c1", "plans")] == \
        ["bt_1_c1.json"]
    assert cell_store.summarize("c1", "plans", ["a"])[0]["a"] == 1
    assert cell_store.delete("c1", "plans", "bt_1_c1.json") is True
