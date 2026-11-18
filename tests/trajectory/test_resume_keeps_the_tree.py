"""A run resumed after it had assembled still stores its tree: the full trees are
checkpointed, and a summary-only tree is re-assembled from the restored legs."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

agent = pytest.importorskip("agents.trajectory.agent")

FULL = {"tree_id": "bt_1", "root": {"type": "Sequence", "id": "n1", "children": []},
        "blackboard": {"traj_1": [{"positions": [0.0] * 6, "time_s": 0.0}]}}
SUMMARY = {**FULL, "blackboard": {"traj_1": {"waypoints": 1, "duration_s": 0.0}}}


def _history():
    call = SimpleNamespace(tool_calls=[{"id": "c1", "name": "assemble_behaviour_tree",
                                        "args": {"cell_id": "c", "plan": {"steps": []}}}])
    result = SimpleNamespace(name="assemble_behaviour_tree", tool_call_id="c1",
                             content=json.dumps({"success": True, "behaviour_tree": SUMMARY}))
    return [call, result]


@pytest.fixture(autouse=True)
def _forget():
    agent._FULL_TREES.clear()
    yield
    agent._FULL_TREES.clear()


def test_a_summary_tree_is_reassembled_from_its_call(monkeypatch):
    from agents.trajectory.tools import behaviour_tree
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"success": True, "behaviour_tree": FULL}
    monkeypatch.setattr(behaviour_tree.assemble_behaviour_tree, "func", fake, raising=False)
    assert agent._extract_behaviour_tree(_history()) == FULL
    assert seen == {"cell_id": "c", "plan": {"steps": []}}


def test_a_reassembly_that_is_refused_stores_nothing(monkeypatch):
    from agents.trajectory.tools import behaviour_tree
    monkeypatch.setattr(behaviour_tree.assemble_behaviour_tree, "func",
                        lambda **kw: {"success": False, "behaviour_tree": FULL}, raising=False)
    assert agent._extract_behaviour_tree(_history()) is None


def test_the_full_trees_travel_with_the_checkpoint():
    from agents.trajectory import plan_checkpoint
    agent._remember_full_tree({"behaviour_tree": FULL})
    saved = plan_checkpoint._full_trees()
    agent._FULL_TREES.clear()
    assert plan_checkpoint._restore_full_trees(json.loads(json.dumps(saved))) == 1
    assert agent._full_tree(SUMMARY) == FULL
