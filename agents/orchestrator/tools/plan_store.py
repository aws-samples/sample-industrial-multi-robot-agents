"""
In-process store for planned BehaviourTrees.

The orchestrator is a ReAct agent: it calls ``call_trajectory_agent`` (which
returns the planned BehaviourTree) and then ``execute_behaviour_tree_on_cell``
(which sends a tree to the cell). If the full tree flowed through the LLM
between those two tool calls, the model would have to re-emit thousands of joint
trajectory floats as a tool argument — which it cannot do faithfully, and the
Bedrock/LangChain tool-use serialization collapses the nested float arrays into
strings (the executor then sees ``precomputed_trajectory`` as a str and fails).

To avoid that, the planning tool stashes the real tree here keyed by its
``tree_id`` and hands the LLM only a lightweight reference; the execution tool
resolves the reference back to the real tree. Both tools run in the SAME
orchestrator process, so a module-level dict is sufficient and nothing large
ever passes through the model.
"""
from __future__ import annotations

import threading
from typing import Optional

_LOCK = threading.Lock()
_STORE: "dict[str, dict]" = {}


def put_plan(tree_id: str, behaviour_tree: dict) -> None:
    """Stash a planned BehaviourTree under its tree_id."""
    with _LOCK:
        _STORE[tree_id] = behaviour_tree


def get_plan(tree_id: str) -> Optional[dict]:
    """Resolve a stashed BehaviourTree by tree_id (None if unknown)."""
    with _LOCK:
        return _STORE.get(tree_id)


def latest_plan_id() -> Optional[str]:
    """The most recently stashed tree_id (dicts keep insertion order)."""
    with _LOCK:
        if not _STORE:
            return None
        return next(reversed(_STORE))
