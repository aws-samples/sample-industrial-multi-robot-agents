"""In ``orchestrator`` mode the planner's own disk holds scratch and nothing else.

The requirement, stated plainly: everything the planner knows about a cell — plans,
traces, events, obstacle and robot models, task programs, quality reports, the
warm-start index — is read and written through the orchestrator's MCP. What stays
on the planner's disk is the work that is not finished yet: the pause/resume
checkpoint of a run in flight, and temporary files.

Tested by running the planner's real write paths against a fake owner, with the
planner's ``CELL_BASE_PATH`` pointed at a directory of its own. The assertion is
the same for every one of them: the artefact appears in the *owner's* tree, and
the planner's tree is still empty afterwards. A call site that was missed in the
rewiring writes to its own volume and fails here — which is the whole point, since
in production that second copy of the truth would just sit there being wrong.

The complement is in :mod:`tests.test_cell_store_asks_the_owner`: this file proves
the traffic goes through the door, that one proves the door refuses to become a
fallback to the mount.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

# knowledge_base exposes its store functions as @tool for the planning agent, and
# the checkpoint encodes LangChain messages; what is under test here is where the
# bytes go. Stub both so this file runs on a bare host without dragging in the LLM
# stack (the decorator stub is the one from tests/test_multi_robot_arity.py). The
# message lists here are empty, so nothing but the import is exercised.
#
# Additive rather than all-or-nothing: another test file may have registered its
# own partial stub before this one is imported, and an ``if not in sys.modules``
# guard would then leave ``langchain_core.load`` missing — which the checkpoint
# needs, so every save would degrade and this file's failures would depend on
# collection order. A real installation (one with a ``__file__``) is left alone.
_lc = sys.modules.get("langchain_core")
if _lc is None or getattr(_lc, "__file__", None) is None:
    if _lc is None:
        _lc = types.ModuleType("langchain_core")
        sys.modules["langchain_core"] = _lc
    if not hasattr(_lc, "__path__"):
        _lc.__path__ = []  # a package, so submodule imports resolve
    if "langchain_core.tools" not in sys.modules:
        _lct = types.ModuleType("langchain_core.tools")
        _lct.tool = lambda f: f
        sys.modules["langchain_core.tools"] = _lct
        _lc.tools = _lct
    if "langchain_core.load" not in sys.modules:
        _lcl = types.ModuleType("langchain_core.load")
        _lcl.dumpd = lambda message: message
        _lcl.load = lambda record, **kw: record
        sys.modules["langchain_core.load"] = _lcl
        _lc.load = _lcl

from agents.trajectory import (  # noqa: E402
    plan_checkpoint, plan_events, plan_trace, task_programs)
from agents.trajectory.tools import (  # noqa: E402
    knowledge_base, plan_store, trajectory_quality)
from agents.orchestrator import cell_data_mcp  # noqa: E402
from utils import cell_layout, cell_store  # noqa: E402

CELL = "welding_cell_1x"


@pytest.fixture
def owner(tmp_path, monkeypatch):
    """The planner in remote mode, wired in-process to a real owner.

    Two cell trees on purpose: ``owner`` is the orchestrator's ``/cells``, ``mine``
    is what the planner would write to if any call site still went to the volume.
    The transport is the orchestrator's own MCP functions, so this exercises the
    real client, the real request shape and the real server-side resolution — only
    the socket is missing.
    """
    owner_root = tmp_path / "owner-cells"
    planner_root = tmp_path / "planner-cells"
    scratch = tmp_path / "planner-scratch"
    owner_root.mkdir()
    planner_root.mkdir()

    def _dispatch(request, timeout=None):
        payload = json.loads(request.data.decode())
        tool = getattr(cell_data_mcp, payload["params"]["name"])
        # The server resolves against the tree IT owns, whatever the client's env
        # says — the same asymmetry the two containers have.
        import os
        was = os.environ.get("CELL_BASE_PATH")
        os.environ["CELL_BASE_PATH"] = str(owner_root)
        try:
            result = tool(**payload["params"]["arguments"])
        finally:
            if was is None:
                os.environ.pop("CELL_BASE_PATH", None)
            else:
                os.environ["CELL_BASE_PATH"] = was

        class _Response:
            def read(self):
                return json.dumps({"result": {"content": [
                    {"type": "text", "text": json.dumps(result)}]}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Response()

    monkeypatch.setenv("CELL_BASE_PATH", str(planner_root))
    monkeypatch.setenv("PLANNER_SCRATCH_PATH", str(scratch))
    monkeypatch.setenv("CELL_DATA_MODE", "orchestrator")
    monkeypatch.setenv("CELL_DATA_MCP_URL", "http://orchestrator:8893/mcp")
    monkeypatch.setattr(cell_store, "_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(cell_store.urllib.request, "urlopen", _dispatch)
    cell_store.clear_cache()
    plan_events._SEQ.clear()
    yield {"owner": owner_root, "mine": planner_root, "scratch": scratch}
    cell_store.clear_cache()
    plan_events._SEQ.clear()


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


# ── each artefact family, one at a time ──────────────────────────────────────

def test_an_assembled_plan_is_written_by_the_owner(owner):
    tree_id = plan_store.save_plan(CELL, {"tree_id": "bt_1_" + CELL,
                                          "root": {"type": "Sequence"}})
    assert tree_id == "bt_1_" + CELL
    assert _files(owner["mine"]) == []
    assert json.loads(cell_store.read_text(CELL, cell_store.KIND_PLANS,
                                           f"{tree_id}.json"))["tree_id"] == tree_id


def test_a_plan_reads_back_through_the_owner_and_not_off_a_mount(owner):
    plan_store.save_plan(CELL, {"tree_id": "bt_2", "root": {"type": "Sequence"}})
    assert (plan_store.get_plan(CELL, "bt_2") or {}).get("tree_id") == "bt_2"


def test_the_plan_listing_is_summarised_by_the_owner(owner):
    """The listing has to cross the wire without the plans crossing with it: each
    tree carries every waypoint it precomputed."""
    for i in (1, 2):
        plan_store.save_plan(CELL, {"tree_id": f"bt_l{i}",
                                    "root": {"type": "Sequence"}})
    listed = plan_store.list_plans(CELL)
    assert sorted(p["tree_id"] for p in listed) == ["bt_l1", "bt_l2"]


def test_a_plan_trace_is_written_by_the_owner_but_named_in_the_owners_tree(owner):
    """The trace has to travel, and the *path* still has to be reportable: the
    dashboard links to it and an operator is told where it is."""
    where = plan_trace.save(CELL, {"run": {"trace_id": "t1", "success": True}})
    assert _files(owner["mine"]) == []
    assert where and where.endswith("plantrace_t1.json")
    assert cell_store.read_json(CELL, cell_store.KIND_LOGS,
                                "plantrace_t1.json") is not None


def test_the_plan_event_stream_is_appended_to_by_the_owner(owner):
    for kind in ("run_started", "operation_planned", "finished"):
        plan_events.emit(CELL, "t1", kind)
    assert _files(owner["mine"]) == []
    text = cell_store.read_text(CELL, cell_store.KIND_LOGS, "planevents_t1.jsonl")
    assert [json.loads(line)["kind"] for line in text.splitlines()] == \
        ["run_started", "operation_planned", "finished"]


def test_the_event_sequence_continues_from_the_owners_copy(owner):
    """A resumed run appends to the stream its earlier attempt began. The length
    it counts from now lives on the other side of the wire."""
    plan_events.emit(CELL, "t9", "run_started")
    plan_events.emit(CELL, "t9", "operation_planned")
    plan_events._SEQ.clear()          # a fresh writer, as after a restart
    third = plan_events.emit(CELL, "t9", "finished")
    assert third["seq"] == 3


def test_a_task_program_is_written_by_the_owner(owner):
    answer = task_programs.save_program(CELL, "weld_door", {
        "operations": [{"type": "weld", "robot": "kr240r2900_1"}]})
    assert answer["success"] is True
    assert _files(owner["mine"]) == []
    assert cell_store.read_json(CELL, cell_store.KIND_TASKS,
                                "weld_door.json")["program"] == "weld_door"


def test_the_quality_report_is_written_by_the_owner(owner):
    path = trajectory_quality.write_quality_log(CELL, {
        "tree_id": "bt_3", "root": {"type": "Sequence"}})
    assert _files(owner["mine"]) == []
    assert path is not None
    assert cell_store.read_json(CELL, cell_store.KIND_LOGS, "bt_3.json") is not None


def test_the_warm_start_index_is_written_by_the_owner(owner):
    knowledge_base._save_index(CELL, {"trajectories": {"bt_4": {"points": 3}}})
    assert _files(owner["mine"]) == []
    assert knowledge_base._load_index(CELL)["trajectories"]["bt_4"]["points"] == 3


def test_the_obstacle_model_is_read_from_the_owner(owner):
    """The single most consequential read in the planner: where an arm may go."""
    from agents.trajectory.tools import robot_kinematics
    cell_store.write_text_local(CELL, cell_store.KIND_CONFIG, "obstacles.yaml",
                                "obstacles:\n  - id: table\n    type: box\n",
                                cell_base=str(owner["owner"]))
    assert [o["id"] for o in robot_kinematics.load_static_obstacles(CELL)] == \
        ["table"]


def test_an_unreachable_owner_never_reads_as_a_cell_with_no_obstacles(owner,
                                                                     monkeypatch):
    """Fail-closed. "No obstacles" from a broken socket routes an arm through a
    fixture; refusing to plan is the acceptable failure."""
    from agents.trajectory.tools import robot_kinematics

    def _dead(request, timeout=None):
        raise cell_store.urllib.error.URLError("connection refused")

    monkeypatch.setattr(cell_store.urllib.request, "urlopen", _dead)
    with pytest.raises(cell_store.CellStoreUnavailable):
        robot_kinematics.load_static_obstacles(CELL)


def test_the_robot_model_is_read_from_the_owner(owner):
    from agents.trajectory.tools import robot_kinematics
    cell_store.write_text_local(
        CELL, cell_store.KIND_CONFIG, "robots.yaml",
        "robots:\n  kr240r2900_1:\n    model: kr240r2900\n",
        cell_base=str(owner["owner"]))
    assert "kr240r2900_1" in robot_kinematics._load_robots_config(CELL)["robots"]


def test_a_cell_with_no_robots_yaml_stops_rather_than_planning_for_nobody(owner):
    from agents.trajectory.tools import robot_kinematics
    with pytest.raises(FileNotFoundError):
        robot_kinematics._load_robots_config(CELL)


# ── what the planner does keep ───────────────────────────────────────────────

def test_the_checkpoint_of_a_run_in_flight_stays_on_the_planners_own_disk(owner):
    """This is the exception, and it is the right one: a checkpoint is the state of
    work that is NOT finished, it is only meaningful to the process that resumes
    it, and it is written on every step of a long run."""
    where = plan_checkpoint.save(CELL, "bt_5", messages=[], operations=[],
                                 constraints=None, step=3)
    assert where is not None
    assert Path(where).is_file()
    assert str(owner["scratch"]) in where
    assert _files(owner["mine"]) == []
    assert _files(owner["owner"]) == [], "a checkpoint must not travel to the owner"


def test_the_checkpoint_root_is_not_inside_the_cells_tree(owner):
    """Otherwise "keep scratch locally" and "the volume is read-only" collide the
    first time a run pauses."""
    scratch = cell_layout.scratch_root()
    assert cell_layout.cells_root() not in scratch.parents
    assert scratch != cell_layout.cells_root()


def test_a_resumed_run_finds_its_checkpoint_without_asking_the_owner(owner,
                                                                    monkeypatch):
    plan_checkpoint.save(CELL, "bt_6", messages=[], operations=[],
                         constraints=None, step=2)

    def _dead(request, timeout=None):
        raise AssertionError("resume must not need the owner")

    monkeypatch.setattr(cell_store.urllib.request, "urlopen", _dead)
    assert plan_checkpoint.load_checkpoint(CELL, "bt_6")["step"] == 2
    assert plan_checkpoint.find_cell("bt_6") == CELL


# ── the whole run, as one ────────────────────────────────────────────────────

def test_a_run_that_plans_pauses_and_records_leaves_one_file_of_its_own(owner):
    """Everything at once, because the property is about the tree and not about any
    single call: after a run, the planner's cell tree is untouched and the owner's
    holds every artefact."""
    plan_events.emit(CELL, "t7", "run_started")
    plan_store.save_plan(CELL, {"tree_id": "bt_7", "root": {"type": "Sequence"}})
    trajectory_quality.write_quality_log(CELL, {"tree_id": "bt_7",
                                                "root": {"type": "Sequence"}})
    knowledge_base._save_index(CELL, {"trajectories": {"bt_7": {}}})
    plan_checkpoint.save(CELL, "bt_7", messages=[], operations=[],
                         constraints=None, step=1)
    plan_trace.save(CELL, {"run": {"trace_id": "t7", "success": True}})
    plan_events.emit(CELL, "t7", "finished")

    assert _files(owner["mine"]) == [], "the planner wrote cell data to its own disk"
    landed = _files(owner["owner"])
    assert any(f.endswith("plans/bt_7.json") for f in landed), landed
    assert any(f.endswith("logs/planevents_t7.jsonl") for f in landed), landed
    assert any(f.endswith("logs/plantrace_t7.json") for f in landed), landed
    assert any(f.endswith("logs/bt_7.json") for f in landed), landed
    assert any(f.endswith("recordings/trajectory_index.json") for f in landed), landed
    assert [f for f in _files(owner["scratch"]) if "bt_7" in f], "no checkpoint kept"
