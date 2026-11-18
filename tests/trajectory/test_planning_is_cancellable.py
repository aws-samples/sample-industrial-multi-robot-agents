"""Stopping a plan that is already running.

Planning is minutes of LLM reasoning around OMPL solves, launched with
``asyncio.to_thread`` — a worker thread CPython cannot kill. So the operator's
Stop button cannot be a kill, and "stop the planning" is a cooperative protocol
with exactly three moving parts: the run registers, Stop flags it, the run
notices at a checkpoint and unwinds.

Every one of those parts is silent when it breaks. A flag nobody checks is a
Stop button that reports success while the plan keeps burning tokens; a
checkpoint that only exists at the top of a run stops nothing after the first
second; and — the sharp one — a cancel raised as an ``Exception`` is CAUGHT by
LangGraph's ``ToolNode``, handed back to the model as a tool result, and the
agent reasons about it and carries on planning. That last failure looks like a
working Stop from the outside: the HTTP call returns 200 and the job keeps going.

So the protocol is tested directly, and the two things that make it work at all
— ``PlanCancelled`` deriving from ``BaseException``, and the checkpoints being
where the time actually goes — are pinned by source assertions, because neither
can be observed from the module's own behaviour.

Host-only: ``plan_cancellation`` imports nothing but ``threading``; the planner
and the MCP server are read as text.
"""
from __future__ import annotations

import ast
import re
import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.tools import plan_cancellation as pc  # noqa: E402

PLANNING_TOOLS = PROJECT_ROOT / "agents" / "trajectory" / "tools" / "planning_tools.py"
OMPL = PROJECT_ROOT / "agents" / "trajectory" / "backends" / "ompl" / "planner.py"
TRAJ_MCP = PROJECT_ROOT / "agents" / "trajectory" / "mcp_server.py"
TRAJ_AGENT = PROJECT_ROOT / "agents" / "trajectory" / "agent.py"


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-wide; a leftover run would cancel the next test."""
    yield
    for tid in pc.active():
        pc.end(tid)
    pc.end(getattr(pc._CURRENT, "tree_id", None) or "")


class TestTheProtocol:
    def test_a_registered_run_is_not_cancelled_until_asked(self):
        pc.begin("bt_1")
        assert pc.active() == ["bt_1"]
        assert pc.is_cancelled("bt_1") is None
        pc.raise_if_cancelled()   # must not raise

    def test_requesting_a_run_makes_its_next_checkpoint_unwind(self):
        pc.begin("bt_1")
        assert pc.request("bt_1") == ["bt_1"]
        with pytest.raises(pc.PlanCancelled) as caught:
            pc.raise_if_cancelled("somewhere")
        assert caught.value.tree_id == "bt_1"
        assert "somewhere" in str(caught.value)

    def test_stop_with_no_tree_id_cancels_everything_planning(self):
        """What the Stop button sends: it knows a cell is busy, not which id."""
        pc.begin("bt_1")
        assert pc.request() == ["bt_1"]
        assert pc.is_cancelled("bt_1")

    def test_stopping_an_idle_stack_is_not_an_error(self):
        """Stop is pressed when nothing is running, and pressed twice. Neither is
        a fault: an exception here would surface as a failed Stop button."""
        assert pc.request() == []
        assert pc.request("no-such-run") == []
        pc.begin("bt_1")
        assert pc.request("bt_1") == ["bt_1"]
        assert pc.request("bt_1") == ["bt_1"]   # idempotent, still reports it

    def test_a_finished_run_cannot_be_cancelled_afterwards(self):
        """Otherwise a Stop pressed just after a plan finished would flag an id
        that the NEXT run might reuse, and cancel a plan nobody stopped."""
        pc.begin("bt_1")
        pc.end("bt_1")
        assert pc.request("bt_1") == []
        assert pc.active() == []
        pc.raise_if_cancelled()   # must not raise

    def test_beginning_a_run_clears_a_stale_flag(self):
        pc.begin("bt_1")
        pc.request("bt_1")
        pc.end("bt_1")
        pc.begin("bt_1")
        assert pc.is_cancelled("bt_1") is None

    def test_a_checkpoint_outside_any_run_does_nothing(self):
        """Tests, scripts and the sync planner path all reach the checkpoint with
        no run bound. It must be a no-op, not a crash."""
        pc.raise_if_cancelled("no run here")

    def test_one_runs_cancel_does_not_unwind_another(self):
        """Two cells, one process. Cancelling one plan must not stop the other —
        which is why the flag is per tree_id and the binding is per thread."""
        pc.begin("bt_other")
        pc.request("bt_mine")   # not registered: no-op
        pc.raise_if_cancelled()
        results: dict = {}

        def worker():
            pc.begin("bt_mine")
            pc.request("bt_mine")
            try:
                pc.raise_if_cancelled()
                results["mine"] = "not cancelled"
            except pc.PlanCancelled:
                results["mine"] = "cancelled"

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert results["mine"] == "cancelled"
        pc.raise_if_cancelled()   # this thread's run is untouched


class TestWhyItIsABaseException:
    """The single most load-bearing line in the feature, and invisible: as an
    ``Exception`` this would be described to the LLM as a tool result and the
    agent would keep planning."""

    def test_plan_cancelled_is_not_an_exception(self):
        assert issubclass(pc.PlanCancelled, BaseException)
        assert not issubclass(pc.PlanCancelled, Exception), (
            "PlanCancelled derives from Exception — LangGraph's ToolNode catches "
            "Exception from a tool and returns it to the model, so Stop would "
            "make the agent reason about the cancel and continue planning")

    def test_it_escapes_the_planner_tools_own_error_handling(self):
        """``run_trajectory_planner`` wraps its dispatch in ``except Exception``
        and turns failures into a result dict for the LLM. A cancel that landed
        there would be reported as a planning failure and the run would go on."""
        source = PLANNING_TOOLS.read_text()
        body = _function_source(source, "run_trajectory_planner")
        assert "except Exception" in body, \
            "the guard this test reasons about is gone; re-check the cancel path"
        try:
            raise pc.PlanCancelled("bt_1")
        except Exception:  # noqa: BLE001 — exactly the guard in the planner
            pytest.fail("a PlanCancelled was caught by `except Exception`")
        except pc.PlanCancelled:
            pass

    def test_the_reason_is_documented_where_someone_would_change_it(self):
        """A future reader "tidying" the base class needs the reason in front of
        them; it cannot be inferred from the class."""
        source = (PROJECT_ROOT / "agents" / "trajectory" / "tools"
                  / "plan_cancellation.py").read_text()
        assert "ToolNode" in source, \
            "nothing in plan_cancellation.py explains why BaseException is required"


class TestTheCheckpointsAreWhereTheTimeGoes:
    """A checkpoint only at the start of a run is a Stop that works for one
    second and then does nothing for twenty minutes."""

    def test_the_planner_step_checks_before_it_solves(self):
        body = _function_source(PLANNING_TOOLS.read_text(), "run_trajectory_planner")
        assert "raise_if_cancelled" in body, \
            "a planning step no longer checks for cancellation; Stop would be " \
            "ignored for the rest of the run"
        assert body.index("raise_if_cancelled") < body.index("dispatch_plan"), \
            "the check happens after the solve, so Stop costs a full solve extra"

    def test_ompl_checks_between_motion_requests(self):
        """The tightest bound available: one request's solve, not one whole LLM
        step, is the longest a Stop can be ignored for."""
        source = OMPL.read_text()
        assert re.search(r"while wi < len\(work\):[\s\S]{0,600}?raise_if_cancelled",
                         source), \
            "the OMPL per-request loop no longer checks for cancellation"

    def test_the_run_registers_itself_or_no_checkpoint_can_fire(self):
        """``raise_if_cancelled`` is a no-op with no run bound to the thread — so
        forgetting ``begin`` disables every checkpoint silently."""
        body = _function_source(TRAJ_AGENT.read_text(), "plan")
        assert "plan_cancellation.begin(" in body, \
            "plan() no longer registers the run; every checkpoint becomes a no-op"
        assert "finally:" in body and "plan_cancellation.end(" in body, \
            "plan() does not deregister in a finally; a crashed run would stay " \
            "registered and the next Stop would 'cancel' a run that is gone"


class TestTheJobSaysItWasCancelledAndNotThatItFailed:
    def test_the_worker_maps_a_cancel_to_its_own_status(self):
        """"failed" would send an operator looking for a broken cell or an
        unreachable weld; they pressed Stop."""
        body = _function_source(TRAJ_MCP.read_text(), "_run_plan_job")
        assert "except _PlanCancelled" in body, \
            "the planning job no longer handles PlanCancelled — it would escape " \
            "asyncio.to_thread and be logged as an unhandled task exception"
        assert 'status="cancelled"' in body
        assert body.index("except _PlanCancelled") < body.index("except Exception"), \
            "the Exception handler comes first; ordering decides which one wins"

    def test_cancel_planning_reports_what_it_actually_stopped(self):
        """An empty list is the honest answer for an idle stack, and the tool must
        still succeed — the Stop button calls it unconditionally."""
        body = _function_source(TRAJ_MCP.read_text(), "cancel_planning")
        assert '"cancelled": marked' in body
        assert '"success": True' in body

    def test_cancel_planning_touches_no_robot(self):
        """It cancels COMPUTATION. If this ever grew an e-stop or a dispatch, the
        Stop button would silently become a motion command."""
        body = _function_source(TRAJ_MCP.read_text(), "cancel_planning")
        for forbidden in ("estop", "dispatch", "move_robot", "publish"):
            assert forbidden not in body, \
                f"cancel_planning mentions '{forbidden}'; it must only stop planning"


def _function_source(source: str, name: str) -> str:
    """One ``def`` as text, wherever it is nested."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    pytest.fail(f"{name} is gone; this test guards nothing")
