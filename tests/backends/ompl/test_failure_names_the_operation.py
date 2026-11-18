"""A planner failure says WHICH operation failed, not just which arm.

The agent's recovery ladder is per operation: add an intermediate waypoint to
*this weld*, reorient *this* stroke, hand *this* one to the other arm via
``reallocate_operation``. A failure that names only ``robot_ns`` makes the agent
match it back to a request by position, and a step that submits several requests
for one arm has no position to match.

Measured on a replay of ``bt_1787955158``: three legs were correctly refused with
``reason_code: excessive_tool_rise`` and every one of them reported
``operation_id: None`` — the refusal was right and unactionable at the same time.

The stamping lives in ONE place (``_fail``) because six separate sites report
failures in this backend and all six had forgotten it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.backends.ompl.planner import (  # noqa: E402
    ompl_available, plan_trajectories_ompl)

pytestmark = pytest.mark.skipif(
    not ompl_available(), reason="needs the OMPL runtime")


def _request(**over) -> dict:
    req = {
        "robot_ns": "robot1",
        "operation_id": "W42",
        "operation_type": "weld",
        "target": {"x": 0.0, "y": 0.0, "z": 1.0},
        "approach_vector": [0.0, 0.0, -1.0],
    }
    req.update(over)
    return req


def test_a_failure_carries_the_operation_it_was_planning():
    """An unloadable cell fails before any solving, so this pins the stamping
    itself rather than a particular geometric outcome."""
    res = plan_trajectories_ompl("no-such-cell-exists", [_request()])

    assert res["success"] is False
    assert res["failures"], "an unloadable cell must be reported, not swallowed"
    assert all(f.get("operation_id") == "W42" for f in res["failures"]), (
        f"failures did not name the operation: {res['failures']}")


def test_the_arm_is_still_named_alongside_the_operation():
    """Both identities are needed: the operation to fix, the arm that failed it."""
    res = plan_trajectories_ompl("no-such-cell-exists", [_request()])
    assert all(f.get("robot_ns") == "robot1" for f in res["failures"])


def test_a_request_without_an_operation_id_is_not_given_a_fake_one():
    """Via points and bare moves have no operation. Inventing an id would send the
    agent looking for a weld that is not in the task."""
    res = plan_trajectories_ompl(
        "no-such-cell-exists", [_request(operation_id=None)])
    assert res["failures"]
    assert all("operation_id" not in f or f["operation_id"] is None
               for f in res["failures"])


def test_two_arms_failing_are_told_apart_by_operation():
    """The case ``robot_ns`` alone cannot express: one step, several operations."""
    res = plan_trajectories_ompl("no-such-cell-exists", [
        _request(operation_id="W01"),
        _request(robot_ns="robot2", operation_id="W07"),
    ])
    assert {f.get("operation_id") for f in res["failures"]} == {"W01", "W07"}
