"""Planning artefacts under concurrency, and a run's status across a restart.

* Every writer of a cell artefact uses its OWN temp file (``.<name>.<pid>.<tid>.tmp``,
  hidden from listings, removed on failure). With the shared ``<name>.tmp``, two
  writers of one file — e.g. two runs storing to a cell's knowledge base — could
  rename each other's half-written text into place, or fail outright.
* Storing to the trajectory knowledge base no longer reads, edits and rewrites
  the whole ``trajectory_index.json``: concurrent stores each append one line, so
  none is lost; the old index stays readable and a later line wins for its key.
* A planning run's status has its own file (``planjob_<tree_id>.json``) on the
  planner's scratch; after a restart the run is found by its id and reported
  ``interrupted`` — ``resumable`` with ``resume_with`` when it left a checkpoint.

Host-runnable (the LangChain decorator is stubbed, as in
tests/test_planner_keeps_nothing_but_scratch.py).
"""
from __future__ import annotations

import json
import sys
import threading
import types

import pytest

_lc = sys.modules.get("langchain_core")
if _lc is None or getattr(_lc, "__file__", None) is None:
    if _lc is None:
        _lc = types.ModuleType("langchain_core")
        sys.modules["langchain_core"] = _lc
    if not hasattr(_lc, "__path__"):
        _lc.__path__ = []
    if "langchain_core.tools" not in sys.modules:
        _lct = types.ModuleType("langchain_core.tools")
        _lct.tool = lambda f: f
        sys.modules["langchain_core.tools"] = _lct
        _lc.tools = _lct

from agents.trajectory import plan_checkpoint, plan_jobs  # noqa: E402
from agents.trajectory.tools import knowledge_base as kb  # noqa: E402
from utils import cell_store  # noqa: E402

CELL = "cellA"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path / "cells"))
    monkeypatch.setenv("PLANNER_SCRATCH_PATH", str(tmp_path / "scratch"))
    monkeypatch.setenv("CELL_DATA_MODE", "local")
    (tmp_path / "cells" / CELL / "agentic_planner").mkdir(parents=True)
    cell_store.clear_cache()
    return tmp_path


def _fn(f):
    return getattr(f, "func", f)


def test_concurrent_writers_of_one_artefact_never_tear_it(store):
    texts = [json.dumps({"writer": w, "pad": "x" * 20000}) for w in range(8)]
    errors, seen = [], set()

    def writer(w):
        try:
            for _ in range(30):
                cell_store.write_text(CELL, cell_store.KIND_LOGS, "shared.json", texts[w])
                seen.add(cell_store.read_text(CELL, cell_store.KIND_LOGS, "shared.json"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert seen <= set(texts)                         # only ever a complete text
    names = [e["name"] for e in cell_store.list_entries(CELL, cell_store.KIND_LOGS)]
    assert names == ["shared.json"]                   # no temp file listed or left
    logs = cell_store.local_path(CELL, cell_store.KIND_LOGS, "shared.json").parent
    assert not [p for p in logs.iterdir() if p.name.endswith(".tmp")]


def test_concurrent_knowledge_base_stores_lose_nothing(store):
    def store_some(arm):
        for i in range(25):
            _fn(kb.store_trajectory)(CELL, "weld", arm, [i * 0.1, 0.0, 1.0],
                                     [float(i)] * 6, 0.001)

    threads = [threading.Thread(target=store_some, args=(f"robot{n}",)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    index = kb._load_index(CELL)
    assert len(index) == 100                         # 4 arms x 25 targets, none lost
    hit = _fn(kb.get_warm_start)(CELL, "weld", "robot2", [0.5, 0.0, 1.0])
    assert hit["found"] and hit["joint_positions"] == [5.0] * 6


def test_the_old_index_is_still_read_and_a_later_entry_wins(store):
    kb._save_index(CELL, {"weld|robot1|0.0|0.0|1.0": {"final_joints": [1] * 6,
                                                      "position_error_m": 0.0}})
    assert _fn(kb.get_warm_start)(CELL, "weld", "robot1", [0, 0, 1])["joint_positions"] == [1] * 6
    _fn(kb.store_trajectory)(CELL, "weld", "robot1", [0, 0, 1], [2] * 6, 0.0)
    assert _fn(kb.get_warm_start)(CELL, "weld", "robot1", [0, 0, 1])["joint_positions"] == [2] * 6


def test_a_runs_status_survives_a_restart_and_says_it_can_resume(store, monkeypatch):
    plan_jobs.save("bt_1_aa_cellA", {"cell_id": CELL, "status": "planning", "backend": "ompl",
                                     "submitted_s": 1.0})
    plan_jobs.save("bt_2_bb_cellA", {"cell_id": CELL, "status": "done", "success": True,
                                     "submitted_s": 2.0})
    path = plan_jobs.path_for(CELL, "bt_1_aa_cellA")
    assert path.name == "planjob_bt_1_aa_cellA.json" and path.is_file()
    assert "scratch" in str(path)                     # the planner's own disk, not the cell's

    monkeypatch.setattr(plan_checkpoint, "describe",
                        lambda c, t: {"resumable": True, "step": 17} if t == "bt_1_aa_cellA" else None)
    saved = plan_jobs.load("bt_1_aa_cellA")             # found by id alone (cell looked up)
    seen = plan_jobs.as_seen_after_restart(saved)
    assert seen["status"] == "interrupted" and seen["resumable"]
    assert seen["resume_with"] == "bt_1_aa_cellA" and seen["checkpoint_step"] == 17
    done = plan_jobs.as_seen_after_restart(plan_jobs.load("bt_2_bb_cellA"))
    assert done["status"] == "done"
    assert [j["tree_id"] for j in plan_jobs.listing(CELL)] == ["bt_2_bb_cellA", "bt_1_aa_cellA"] \
        or {j["tree_id"] for j in plan_jobs.listing(CELL)} == {"bt_1_aa_cellA", "bt_2_bb_cellA"}


def test_a_run_without_a_checkpoint_is_interrupted_not_resumable(store, monkeypatch):
    plan_jobs.save("bt_3_cc_cellA", {"cell_id": CELL, "status": "planning"})
    monkeypatch.setattr(plan_checkpoint, "describe", lambda c, t: None)
    seen = plan_jobs.as_seen_after_restart(plan_jobs.load("bt_3_cc_cellA"))
    assert seen["status"] == "interrupted" and not seen["resumable"]
    assert seen["resume_with"] is None and "plan it again" in seen["note"]


def test_an_unsafe_run_id_is_never_a_path(store):
    assert plan_jobs.save("../x", {"cell_id": CELL, "status": "planning"}) is None
    assert plan_jobs.load("../x") is None
