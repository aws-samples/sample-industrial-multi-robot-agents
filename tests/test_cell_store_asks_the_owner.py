"""The planner's one door to cell data: it asks the owner, and it fails closed.

``dashboard/web_control_panel.py`` was moved off the shared ``/cells`` mount for a
reason that is pinned in ``tests/test_dashboard_asks_mcp_not_the_volume.py``: a
second independent reader of a tree it does not own is how two readers come to
disagree about the same cell. The trajectory planner was the *other* reader — and
also a writer, of plans, traces, events, quality logs, task programs and the
warm-start index.

:mod:`utils.cell_store` is the seam that ends that. What is pinned here is not
"the planner uses a helper", it is the three properties that make the seam worth
having:

* **One vocabulary, no paths.** A caller names a ``(kind, name)`` pair. Nothing
  the planner can say resolves outside the cell.
* **Fail-closed.** In ``orchestrator`` mode an unreachable owner raises. It never
  degrades into "the file is absent" — because the file most often asked for is
  ``obstacles.yaml``, and a planner told "no obstacles" by a broken socket routes
  arms through the fixtures it cannot see. Availability failures are acceptable;
  that one is not.
* **The same answer either way.** Local and remote must read and write the same
  bytes at the same names, or the seam is a second source of truth rather than a
  door to the first.
"""
from __future__ import annotations

import json

import pytest

from utils import cell_layout, cell_store


@pytest.fixture(autouse=True)
def _local_cells(tmp_path, monkeypatch):
    """A cell tree of our own, and default (local) mode unless a test says else."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("CELL_DATA_MODE", raising=False)
    monkeypatch.delenv("CELL_DATA_MCP_URL", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_HOST", raising=False)
    cell_store.clear_cache()
    yield
    cell_store.clear_cache()


# ── the vocabulary is closed ──────────────────────────────────────────────────

def test_the_default_mode_is_local_so_nothing_changes_by_importing_this():
    """Every existing caller — the orchestrator, a test, a host script — is local."""
    assert cell_store.mode() == cell_store.MODE_LOCAL
    assert cell_store.is_remote() is False


@pytest.mark.parametrize("name", [
    "../../etc/passwd", "a/b.json", "..", "", "a b.json", "x\x00y",
    "/absolute.json", "sub/../escape.json",
])
def test_a_name_that_is_not_a_plain_file_is_refused(name):
    """Refused at the door, in BOTH modes: the check is on the request, not on
    what the filesystem happens to allow."""
    with pytest.raises(cell_store.CellStoreError):
        cell_store.read_text("c1", cell_store.KIND_LOGS, name)
    with pytest.raises(cell_store.CellStoreError):
        cell_store.write_text("c1", cell_store.KIND_LOGS, name, "x")


def test_an_unknown_kind_is_refused_rather_than_guessed():
    with pytest.raises(cell_store.CellStoreError):
        cell_store.read_text("c1", "scene", "robot.urdf")


def test_the_scene_bundle_is_not_one_of_the_kinds():
    """It stays a read-only mount: mesh URIs in the published robot_description
    are resolved by processes outside this repo, and a cell is up to ~1 GB."""
    assert "scene" not in cell_store.KINDS
    assert "meshes" not in cell_store.KINDS
    assert "urdf" not in cell_store.KINDS


def test_a_cell_id_cannot_walk_out_of_the_cells_root():
    with pytest.raises(cell_store.CellStoreError):
        cell_store.local_path("../..", cell_store.KIND_LOGS, "x.json")


# ── local mode reads and writes the tree the orchestrator owns ────────────────

def test_a_write_then_a_read_is_the_same_bytes():
    cell_store.write_text("c1", cell_store.KIND_PLANS, "bt_1_c1.json", '{"a":1}')
    assert cell_store.read_text("c1", cell_store.KIND_PLANS,
                               "bt_1_c1.json") == '{"a":1}'


def test_an_absent_artefact_reads_as_none_and_is_not_an_error():
    """The one degradation a caller may make: absence is a real answer."""
    assert cell_store.read_text("c1", cell_store.KIND_PLANS, "nope.json") is None


def test_a_write_lands_where_cell_layout_says_it_does():
    """The seam must not become a second opinion on the cell's layout."""
    cell_store.write_text("c1", cell_store.KIND_PLANS, "bt_2_c1.json", "{}")
    assert cell_layout.plan_file("c1", "bt_2_c1").is_file()


def test_append_adds_a_line_without_rewriting_the_stream():
    for i in range(3):
        cell_store.append_text("c1", cell_store.KIND_LOGS, "planevents_t1.jsonl",
                               json.dumps({"seq": i}) + "\n")
    text = cell_store.read_text("c1", cell_store.KIND_LOGS, "planevents_t1.jsonl")
    assert [json.loads(line)["seq"] for line in text.splitlines()] == [0, 1, 2]


def test_config_resolves_through_the_read_write_asymmetry_cell_layout_owns():
    """A writer must not be steered by a stale copy at the cell root, and a reader
    must still find one. cell_layout owns that rule; this only checks it is used."""
    read = cell_store.local_path("c1", cell_store.KIND_CONFIG, "obstacles.yaml")
    write = cell_store.local_path("c1", cell_store.KIND_CONFIG, "obstacles.yaml",
                                  for_write=True)
    assert read == cell_layout.config_file("c1", "obstacles.yaml")
    assert write == cell_layout.agent_config_file("c1", "obstacles.yaml")


# ── listings are metadata, and newest-first means newest-first ───────────────

def test_a_listing_carries_no_contents():
    cell_store.write_text("c1", cell_store.KIND_PLANS, "bt_1_c1.json",
                          json.dumps({"waypoints": list(range(1000))}))
    entries = cell_store.list_entries("c1", cell_store.KIND_PLANS, suffix=".json")
    assert [e["name"] for e in entries] == ["bt_1_c1.json"]
    assert "text" not in entries[0] and "waypoints" not in entries[0]


def test_summarize_returns_only_the_named_fields():
    """Why this exists: a plan carries every precomputed waypoint, and a plan LIST
    wants none of them. The reading happens where the bytes already are."""
    cell_store.write_json("c1", cell_store.KIND_PLANS, "bt_1_c1.json",
                          {"tree_id": "bt_1_c1", "status": "ok",
                           "waypoints": list(range(500))})
    rows = cell_store.summarize("c1", cell_store.KIND_PLANS,
                                ["tree_id", "status"], count_fields=["waypoints"])
    assert rows[0]["tree_id"] == "bt_1_c1" and rows[0]["status"] == "ok"
    assert rows[0]["waypoints_count"] == 500
    assert "waypoints" not in rows[0]


def test_two_artefacts_written_in_the_same_instant_still_list_newest_first():
    """The mtime tie is the common case in a fast run, and inside the tie the
    fallback used to reverse the order — silently listing oldest first."""
    for name in ("plantrace_c1_20260101_000000.json",
                 "plantrace_c1_20260102_000000.json"):
        cell_store.write_json("c1", cell_store.KIND_LOGS, name, {"run": name})
    rows = cell_store.summarize("c1", cell_store.KIND_LOGS, ["run"],
                                prefix="plantrace_")
    assert [r["name"] for r in rows] == ["plantrace_c1_20260102_000000.json",
                                         "plantrace_c1_20260101_000000.json"]


def test_summarize_can_reach_a_nested_field_by_dotted_name():
    """The caller names the fields in its own vocabulary; the store knows no schema."""
    cell_store.write_json("c1", cell_store.KIND_LOGS, "plantrace_t1.json",
                          {"run": {"trace_id": "t1", "success": True}})
    rows = cell_store.summarize("c1", cell_store.KIND_LOGS, ["run.trace_id"],
                                prefix="plantrace_")
    assert rows[0]["run.trace_id"] == "t1"


# ── the two parsers disagree on purpose ──────────────────────────────────────

def test_unparseable_json_reads_as_absent():
    """Every caller's fallback for half a JSON artefact is what it does for a
    missing one, and there is nothing safer to do with half a file."""
    cell_store.write_text("c1", cell_store.KIND_PLANS, "bt_1_c1.json", "{not json")
    assert cell_store.read_json("c1", cell_store.KIND_PLANS, "bt_1_c1.json") is None


def test_unparseable_yaml_raises_because_it_is_a_collision_input():
    """obstacles.yaml and robots.yaml decide where an arm may go. Half of one must
    stop the run, not read as an empty cell."""
    cell_store.write_text("c1", cell_store.KIND_CONFIG, "obstacles.yaml",
                          "obstacles: [\n  - id: bad\n :::")
    with pytest.raises(Exception):
        cell_store.read_yaml("c1", cell_store.KIND_CONFIG, "obstacles.yaml")


# ── remote mode: it asks, and it refuses when it cannot ──────────────────────

def test_remote_mode_without_an_endpoint_refuses_instead_of_reading_the_mount(
        monkeypatch):
    """The whole point. "Route cell data through the owner" must not degrade
    silently into "read the volume" — that is the drift being fixed."""
    cell_store.write_text("c1", cell_store.KIND_CONFIG, "obstacles.yaml",
                          "obstacles: []\n")
    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    with pytest.raises(cell_store.CellStoreUnavailable):
        cell_store.endpoint()
    with pytest.raises(cell_store.CellStoreUnavailable):
        cell_store.read_text("c1", cell_store.KIND_CONFIG, "obstacles.yaml")


def test_an_unreachable_owner_is_unknown_and_never_absent(monkeypatch):
    """``None`` means "the cell declares no fixtures". A dead socket must not be
    able to say that."""
    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    # Port 1 on the loopback: nothing listens, connection refused immediately.
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://127.0.0.1:1/mcp")
    monkeypatch.setenv("CELL_DATA_CACHE_TTL_S", "0")
    with pytest.raises(cell_store.CellStoreUnavailable):
        cell_store.read_text("c1", cell_store.KIND_CONFIG, "obstacles.yaml")


def test_a_bad_mode_is_an_error_not_a_default(monkeypatch):
    monkeypatch.setenv("CELL_DATA_MODE", "whatever")
    with pytest.raises(cell_store.CellStoreError):
        cell_store.mode()


def test_the_endpoint_is_built_from_the_same_vars_the_compose_file_sets(
        monkeypatch):
    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("ORCHESTRATOR_HOST", "orchestrator")
    monkeypatch.setenv("ORCH_MCP_PORT", "8893")
    assert cell_store.endpoint() == "http://orchestrator:8893/mcp"
    assert cell_store.describe()["endpoint"] == "http://orchestrator:8893/mcp"


def test_remote_mode_speaks_the_repos_own_json_rpc_contract(monkeypatch):
    """One POST, one JSON reply, the tool's return value as JSON text in
    ``result.content[0].text`` — the same contract utils/mcp_http.py serves."""
    seen: dict = {}

    class _Response:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data.decode())
        tool_result = {"success": True, "found": True, "text": "obstacles: []\n"}
        return _Response(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "result": {"content": [{"type": "text",
                                    "text": json.dumps(tool_result)}]}}).encode())

    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://orchestrator:8893/mcp")
    monkeypatch.setenv("CELL_DATA_CACHE_TTL_S", "0")
    monkeypatch.setattr(cell_store.urllib.request, "urlopen", _fake_urlopen)

    assert cell_store.read_text("c1", cell_store.KIND_CONFIG,
                                "obstacles.yaml") == "obstacles: []\n"
    assert seen["url"] == "http://orchestrator:8893/mcp"
    assert seen["payload"]["method"] == "tools/call"
    assert seen["payload"]["params"]["name"] == "cell_read"
    assert seen["payload"]["params"]["arguments"] == {
        "cell_id": "c1", "kind": "config", "name": "obstacles.yaml"}


def test_a_remote_found_false_is_absence_and_a_refusal_is_not(monkeypatch):
    """Both come back over the same wire, and they must not arrive as the same
    sentence: one is "there is no such file", the other is "I would not look"."""
    replies: list = []

    class _Response:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(request, timeout=None):
        return _Response(json.dumps({
            "result": {"content": [{"type": "text",
                                    "text": json.dumps(replies.pop(0))}]}}).encode())

    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://orchestrator:8893/mcp")
    monkeypatch.setenv("CELL_DATA_CACHE_TTL_S", "0")
    monkeypatch.setattr(cell_store.urllib.request, "urlopen", _fake_urlopen)

    replies.append({"success": True, "found": False, "text": None})
    assert cell_store.read_text("c1", cell_store.KIND_PLANS, "gone.json") is None

    replies.append({"success": False, "error": "unknown cell-data kind"})
    with pytest.raises(cell_store.CellStoreError):
        cell_store.read_text("c1", cell_store.KIND_PLANS, "gone.json")


def test_config_reads_are_cached_only_when_they_cost_a_round_trip(monkeypatch):
    """Assembling one plan asks for obstacles.yaml several times over. Caching it
    locally, though, would make a test that writes then reads see its own write
    five seconds late."""
    calls = {"n": 0}

    class _Response:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(request, timeout=None):
        calls["n"] += 1
        return _Response(json.dumps({"result": {"content": [{
            "type": "text",
            "text": json.dumps({"success": True, "found": True,
                                "text": "obstacles: []\n"})}]}}).encode())

    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://orchestrator:8893/mcp")
    monkeypatch.setattr(cell_store, "_CACHE_TTL_S", 60.0)
    monkeypatch.setattr(cell_store.urllib.request, "urlopen", _fake_urlopen)
    cell_store.clear_cache()

    for _ in range(4):
        cell_store.read_text("c1", cell_store.KIND_CONFIG, "obstacles.yaml")
    assert calls["n"] == 1, "four config reads cost more than one round trip"

    # And the local side is never cached: a writer must see its own write.
    monkeypatch.setenv("CELL_DATA_MODE", "local")
    cell_store.write_text("c1", cell_store.KIND_CONFIG, "obstacles.yaml", "a: 1\n")
    cell_store.write_text("c1", cell_store.KIND_CONFIG, "obstacles.yaml", "a: 2\n")
    assert cell_store.read_text("c1", cell_store.KIND_CONFIG,
                                "obstacles.yaml") == "a: 2\n"


# ── an explicit cell_base is the one documented way past the seam ────────────

def test_an_explicit_cell_base_stays_local_even_in_remote_mode(monkeypatch,
                                                               tmp_path):
    """A caller that has already named a directory on this machine (a host script,
    a test fixture) is not asking an owner anything — and there is nothing to ask,
    because that path does not exist on the owner's disk."""
    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://127.0.0.1:1/mcp")
    where = tmp_path / "elsewhere"
    cell_store.write_text("c1", cell_store.KIND_LOGS, "bt_9_c1.json", "{}",
                          cell_base=str(where))
    assert cell_store.read_text("c1", cell_store.KIND_LOGS, "bt_9_c1.json",
                                cell_base=str(where)) == "{}"
