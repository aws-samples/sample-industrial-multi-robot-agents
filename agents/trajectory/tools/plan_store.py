"""
Plan store — persist assembled BehaviourTrees (with their precomputed
trajectories) so a plan calculated by the trajectory agent can be retrieved and
replayed later by the cell agent WITHOUT re-planning.

Each plan is one JSON artefact named ``{tree_id}.json`` in the cell's ``plans``
kind, written through :mod:`utils.cell_store` — the orchestrator's cell-data MCP
when this process is the planner, the local ``plans/`` directory when it is the
orchestrator or a test. The planner holds no plan of its own on disk: a plan is
cell data, and cell data has one owner.

where the payload is exactly the BehaviourTree dict the planner produced
({"tree_id", "cell_id", "root", "metadata"}) — the EXECUTE leaves carry their
``precomputed_trajectory`` arrays, so the file is self-contained and can be fed
straight to the cell MCP ``execute_behaviour_tree`` tool.

The store lives on the orchestrator's /cells mount, so the cell-controller can
read a plan the trajectory agent computed without either of them writing the
other's tree.
"""
from __future__ import annotations

import time
from typing import Optional

from agents.trajectory.tools.btcpp import node_kind
from utils import cell_store


# A stored plan is either usable or a record of a run that did not produce one.
# Both live in plans/ — the failed one because the trajectories it DID compute are
# hours of work worth keeping and reading — so the difference has to be explicit
# and it has to be checked before anything is handed to a robot.
STATUS_OK = "ok"
STATUS_FAILED = "failed"


# The fields a plan LIST needs, all of them at the top level of the artefact. Named
# here so the pick happens where the bytes are: a plan carries every waypoint it
# computed, and reading two hundred of them to show two hundred status chips is the
# difference between a listing and a transfer.
_LIST_FIELDS = ("tree_id", "status", "error", "error_kind", "resumable",
                "num_planned", "created_s", "num_operations")


def _name(tree_id: str) -> str:
    return f"{tree_id}.json"


def _count_execute(node: dict) -> int:
    """Count ExecuteTrajectory leaves in a tree (number of operations/welds).

    node_kind normalises a legacy ``EXECUTE`` to the BT.CPP name, so plans saved
    in either dialect count the same.
    """
    if not isinstance(node, dict):
        return 0
    n = 1 if node_kind(node) == "ExecuteTrajectory" else 0
    for c in node.get("children", []) or []:
        n += _count_execute(c)
    if isinstance(node.get("root"), dict):
        n += _count_execute(node["root"])
    return n


def save_plan(cell_id: str, behaviour_tree: dict) -> Optional[str]:
    """Persist ``behaviour_tree`` under the cell's plans/ dir keyed by tree_id.

    Returns the tree_id on success, or None if the tree has no tree_id / could
    not be written. Never raises — persistence must not break planning.
    """
    try:
        tree_id = behaviour_tree.get("tree_id")
        if not tree_id:
            return None
        payload = {
            "tree_id":     tree_id,
            "cell_id":     cell_id,
            "created_s":   time.time(),
            # Stated, never implied. Failed runs are stored in this same directory
            # (see save_failed_plan), so "no status" must not be the way a reader
            # tells a usable plan from a wreck.
            "status":      STATUS_OK,
            "num_operations": _count_execute(behaviour_tree),
            "behaviour_tree": behaviour_tree,
        }
        # Atomic wherever it lands: cell_store renames a temp file into place, so a
        # reader polling the plan list never sees half a tree.
        cell_store.write_json(cell_id, cell_store.KIND_PLANS, _name(tree_id),
                              payload)
        _write_quality_log(cell_id, payload)
        return tree_id
    except Exception as exc:  # noqa: BLE001
        print(f"[plan_store] WARNING: could not save plan: {exc}", flush=True)
        return None


def _write_quality_log(cell_id: str, payload: dict) -> None:
    """Score the plan we just stored and log the result beside it.

    Here because this is the one place every usable plan passes through, so the
    quality report cannot be forgotten by a new caller. It runs AFTER the atomic
    rename and swallows everything: a plan that took minutes to compute is never
    lost because measuring it went wrong.
    """
    try:
        from agents.trajectory.tools import trajectory_quality
        trajectory_quality.write_quality_log(cell_id, payload)
    except Exception as exc:  # noqa: BLE001
        print(f"[plan_store] WARNING: quality log skipped: {exc}", flush=True)


def save_failed_plan(cell_id: str, tree_id: Optional[str], *,
                     error: Optional[str] = None,
                     error_kind: Optional[str] = None,
                     note: Optional[str] = None,
                     failures: Optional[list] = None,
                     operations: Optional[list] = None,
                     trajectories: Optional[list] = None,
                     behaviour_tree: Optional[dict] = None,
                     resumable: bool = False) -> Optional[str]:
    """Persist a run that produced NO usable tree, marked ``failed``.

    A failed run is not nothing. By the time it fails it has typically solved most
    of its welds — the 2026-09-08 run had thirteen trajectories and a validated
    approach on each before the socket dropped — and throwing that away is what
    made every retry start from W01. So the same directory keeps it, in the same
    shape, with the same reader, and the failure written down beside it: which
    welds are in it, what went wrong, and whether the run can be resumed.

    ``trajectories`` is the trajectory-cache snapshot, so a reader can see exactly
    which operations DID plan. The tree, if the assembler produced one and it was
    then rejected, is kept too — it is the artefact under discussion.

    It is stored, listed and readable. It is NOT executable: ``get_plan`` refuses
    to hand it out without an explicit opt-in, because "replay the latest plan"
    must never quietly mean "replay the broken one".
    """
    try:
        if not tree_id:
            return None
        planned = [
            {"robot_ns": r.get("robot_ns"), "operation_id": r.get("operation_id"),
             "waypoints": len(r.get("trajectory") or [])}
            for r in (trajectories or []) if isinstance(r, dict)
        ]
        payload = {
            "tree_id":        tree_id,
            "cell_id":        cell_id,
            "created_s":      time.time(),
            "status":         STATUS_FAILED,
            "error":          error,
            "error_kind":     error_kind,
            "note":           note,
            "resumable":      bool(resumable),
            "failures":       failures or [],
            "num_operations": len(operations or []),
            "num_planned":    len(planned),
            "planned":        planned,
            "trajectories":   trajectories or [],
            "behaviour_tree": behaviour_tree,
        }
        cell_store.write_json(cell_id, cell_store.KIND_PLANS, _name(tree_id),
                              payload)
        return tree_id
    except Exception as exc:  # noqa: BLE001
        print(f"[plan_store] WARNING: could not save failed plan: {exc}", flush=True)
        return None


def list_plans(cell_id: str) -> list[dict]:
    """List saved plans for a cell, newest first.

    Returns [{tree_id, status, created_s, num_operations, size_bytes, path}, ...].
    ``status`` is ``ok`` or ``failed``; a plan written before the status field
    existed reads as ``ok``, which is what it was — the store only ever held
    successfully assembled trees at that point.
    """
    out: list[dict] = []
    try:
        for row in cell_store.summarize(cell_id, cell_store.KIND_PLANS,
                                        list(_LIST_FIELDS), suffix=".json"):
            out.append({
                "tree_id":        row.get("tree_id") or row["name"][:-len(".json")],
                "status":         row.get("status") or STATUS_OK,
                "error":          row.get("error"),
                "error_kind":     row.get("error_kind"),
                "resumable":      bool(row.get("resumable")),
                "num_planned":    row.get("num_planned"),
                "created_s":      row.get("created_s"),
                "num_operations": row.get("num_operations"),
                "size_bytes":     row.get("size_bytes"),
                "path":           row.get("path"),
            })
    except Exception as exc:  # noqa: BLE001
        print(f"[plan_store] WARNING: could not list plans: {exc}", flush=True)
    out.sort(key=lambda m: (m.get("created_s") or 0), reverse=True)
    return out


def get_failed_plan(cell_id: str, tree_id: str) -> Optional[dict]:
    """The full record of a failed run — for reading, not for executing."""
    try:
        meta = cell_store.read_json(cell_id, cell_store.KIND_PLANS,
                                    _name(tree_id))
    except Exception:  # noqa: BLE001
        return None
    if not meta or (meta.get("status") or STATUS_OK) != STATUS_FAILED:
        return None
    return meta


def get_plan(cell_id: str, tree_id: Optional[str] = None,
             include_failed: bool = False) -> Optional[dict]:
    """Load a saved plan's BehaviourTree (with precomputed trajectories).

    If ``tree_id`` is None, returns the most recently created plan for the cell.
    Returns the BehaviourTree dict ({"tree_id","cell_id","root","metadata"}) or
    None if not found.

    **Failed plans are excluded unless ``include_failed`` is set.** This is the
    fail-closed half of storing them at all: the trees this returns get fed
    straight to ``execute_behaviour_tree``, and "the newest plan" silently
    resolving to a run that never assembled a valid one is how a partial plan
    reaches a robot. Read a failed run with :func:`get_failed_plan`.
    """
    try:
        if tree_id:
            meta = cell_store.read_json(cell_id, cell_store.KIND_PLANS,
                                        _name(tree_id))
            if not meta:
                return None
            if (meta.get("status") or STATUS_OK) == STATUS_FAILED and not include_failed:
                return None
            return meta.get("behaviour_tree")
        plans = [p for p in list_plans(cell_id)
                 if include_failed or p.get("status") != STATUS_FAILED]
        if not plans:
            return None
        # By tree_id, not by path: the newest plan is identified from the listing,
        # then fetched from its owner by name. A path would only be resolvable by
        # whoever holds the volume.
        meta = cell_store.read_json(cell_id, cell_store.KIND_PLANS,
                                    _name(plans[0]["tree_id"]))
        return (meta or {}).get("behaviour_tree")
    except Exception as exc:  # noqa: BLE001
        print(f"[plan_store] WARNING: could not load plan: {exc}", flush=True)
        return None
