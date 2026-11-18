"""The trajectory KPIs reach the dashboard, and they come over MCP.

"How good was the plan?" had three answers before this and none of them was in
the UI: a JSON report written beside the plan, a CLI that resolved cell paths on
whatever machine it ran on, and a person reading waypoints. So the Performance
card in the BTree pane shows the score, its four components, the numbers behind
them and the ranked findings — and it gets every one of those from the planner
over MCP, because the panel computing a score would be a second answer to the
question and the two would drift the first time a weight moved.

Three seams, three kinds of test:

  * :func:`trajectory_quality.report_for` — the policy. Which plan, the stored
    report or a fresh score, and what a missing one means. Real store, real
    plans, a tmp cell tree.
  * ``get_plan_quality`` on the planner's MCP — plumbing, read as source text:
    ``agents/trajectory/mcp_server.py`` cannot be imported without pinocchio and
    the LLM stack, which is exactly why the policy is not in it.
  * the dashboard route and the browser — the route asks the trajectory agent and
    renders whatever comes back.

The failure these are mostly about is one shape: an unreachable owner rendering
as a plan that scores badly. "Not measured, because X" and "measured, and it is
poor" are opposite conclusions that lead to opposite actions, and the second is
the one an operator acts on.
"""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest

from agents.trajectory.tools import plan_store, trajectory_quality as tq
from utils import cell_layout, cell_store

from tests.test_dashboard_asks_mcp_not_the_volume import (  # noqa: F401
    PANEL_CODE, PANEL_SRC, calls, panel)
from tests.test_dashboard_script_loads import (
    NODE, PLAN_BODY, _render, _run_probe)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_SRC = (PROJECT_ROOT / "agents" / "trajectory" / "mcp_server.py").read_text()


# ─── plans to score ──────────────────────────────────────────────────────────

def wp(positions, t, phase="transit"):
    return {"positions": list(positions), "time_s": float(t), "phase": phase}


def straight_leg(n=11, span=1.0, duration=1.0):
    return [wp([span * i / (n - 1), 0.0], duration * i / (n - 1))
            for i in range(n)]


def exec_node(name, robot_ns, key):
    return {"type": "ExecuteTrajectory", "name": name,
            "ports": {"robot_ns": robot_ns, "operation_id": name,
                      "trajectory": "{%s}" % key}}


def a_tree(tree_id, cell_id="kpi_cell"):
    return {"tree_id": tree_id, "cell_id": cell_id,
            "root": exec_node("A", "robot1", "a"),
            "blackboard": {"a": straight_leg(duration=2.0)}}


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """A cell tree of our own, read and written locally."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    monkeypatch.delenv("CELL_DATA_MODE", raising=False)
    cell_store.clear_cache()
    yield "kpi_cell"
    cell_store.clear_cache()


# ─── the policy: which plan, and where the numbers came from ─────────────────

class TestReportForServesTheStoredReport:

    def test_a_saved_plan_answers_from_the_report_save_plan_already_wrote(self, cell):
        """The normal path. ``save_plan`` pays for the score once, at save time."""
        assert plan_store.save_plan(cell, a_tree("bt_1_kpi")) == "bt_1_kpi"
        assert cell_layout.quality_log_file(cell, "bt_1_kpi").is_file()

        r = tq.report_for(cell, "bt_1_kpi")
        assert r["success"] is True
        assert r["source"] == "stored", \
            "a plan saved with the metrics in place was re-scored on read"
        assert r["tree_id"] == "bt_1_kpi"
        assert "efficiency_score" in r and "findings" in r

    def test_a_plan_older_than_the_metrics_is_scored_on_demand(self, cell):
        """No stored report: score the stored tree now rather than say nothing.

        This is every plan in every cell that predates the quality log, and
        "unknown" for those would make the card useless on arrival.
        """
        assert plan_store.save_plan(cell, a_tree("bt_2_kpi")) == "bt_2_kpi"
        cell_layout.quality_log_file(cell, "bt_2_kpi").unlink()
        cell_store.clear_cache()

        r = tq.report_for(cell, "bt_2_kpi")
        assert r["success"] is True
        assert r["source"] == "computed"
        assert isinstance(r["efficiency_score"], float)
        assert "findings" in r and "legs" in r

    def test_a_truncated_report_is_recomputed_rather_than_served(self, cell):
        """A log without a score is not a score. Half a report reads as a report."""
        assert plan_store.save_plan(cell, a_tree("bt_3_kpi")) == "bt_3_kpi"
        cell_layout.quality_log_file(cell, "bt_3_kpi").write_text(
            json.dumps({"tree_id": "bt_3_kpi", "note": "written by an older writer"}))
        cell_store.clear_cache()

        r = tq.report_for(cell, "bt_3_kpi")
        assert r["success"] is True
        assert r["source"] == "computed"
        assert "efficiency_score" in r


class TestLatestNeverResolvesToAFailedRun:

    def test_omitting_the_tree_id_takes_the_newest_successful_plan(self, cell):
        plan_store.save_plan(cell, a_tree("bt_ok_kpi"))
        for wanted in (None, "latest"):
            r = tq.report_for(cell, wanted)
            assert r["success"] is True, r
            assert r["tree_id"] == "bt_ok_kpi"

    def test_a_failed_run_is_never_what_latest_finds(self, cell):
        """A fragment scored against a whole program flatters it for stopping."""
        plan_store.save_plan(cell, a_tree("bt_ok_kpi"))
        plan_store.save_failed_plan(cell, "bt_bad_kpi",
                                    error="ExpiredTokenException",
                                    error_kind="credentials", resumable=True)
        assert {p["tree_id"] for p in plan_store.list_plans(cell)} == \
            {"bt_ok_kpi", "bt_bad_kpi"}, "the failed run was not even stored"

        assert tq.report_for(cell, "latest")["tree_id"] == "bt_ok_kpi"

    def test_asking_for_a_failed_run_by_name_is_refused_not_scored(self, cell):
        plan_store.save_failed_plan(cell, "bt_bad_kpi", error="socket closed")
        r = tq.report_for(cell, "bt_bad_kpi")
        assert r["success"] is False
        assert "FAILED" in r["error"], r["error"]
        assert "efficiency_score" not in r, \
            "a fragment was scored — the number would be quoted anyway"

    def test_a_cell_with_no_plans_says_so_and_names_the_cell(self, cell):
        r = tq.report_for(cell, "latest")
        assert r["success"] is False
        assert cell in r["error"]
        assert "efficiency_score" not in r


class TestAnUnreachableOwnerIsNotABadPlan:
    """Fail-closed. The whole point of the ``success`` flag on this report."""

    def _unreachable(self, monkeypatch, cell):
        def boom(*a, **k):
            raise cell_store.CellStoreUnavailable(
                "cell-data MCP at http://orchestrator:8893/mcp is not answering")
        monkeypatch.setattr(cell_store, "read_json", boom)
        monkeypatch.setattr(cell_store, "read_text", boom)
        monkeypatch.setattr(cell_store, "list_entries", boom)
        monkeypatch.setattr(cell_store, "summarize", boom)

    def test_it_propagates_instead_of_answering_no_findings(self, cell, monkeypatch):
        self._unreachable(monkeypatch, cell)
        with pytest.raises(cell_store.CellStoreUnavailable):
            tq.report_for(cell, "bt_1_kpi")

    def test_resolving_latest_does_not_swallow_it_into_an_unplanned_cell(
            self, cell, monkeypatch):
        """``plan_store.list_plans`` answers ``[]`` for every failure it meets.

        So "this cell has never been planned" and "the owner is down" arrive at
        the same branch, and the refusal that branch writes is the wrong one —
        it sends an operator to re-plan a cell whose plans are fine.
        """
        self._unreachable(monkeypatch, cell)
        with pytest.raises(cell_store.CellStoreUnavailable):
            tq.report_for(cell, "latest")

    def test_an_unsafe_tree_id_never_reaches_a_path(self, cell):
        with pytest.raises(cell_store.CellStoreError):
            tq.report_for(cell, "../../../etc/passwd")


class TestSafetyIsReportedAndNeverScored:

    def test_the_report_carries_clearance_and_no_score_weighs_it(self, cell):
        plan_store.save_plan(cell, a_tree("bt_1_kpi"))
        r = tq.report_for(cell, "bt_1_kpi")
        assert "safety" in r, "the card has nothing to show for clearance"
        assert set(r["score_weights"]) == {"path", "schedule", "smoothness", "dwell"}, \
            ("a score is a thing people optimise; a scored clearance is a "
             "safety margin turned into a tunable")
        assert "safety" not in r["score_components"]


# ─── the MCP tool is plumbing ────────────────────────────────────────────────

class TestTheMcpToolDelegatesAndComputesNothing:
    """Source text: the module pulls pinocchio and the LLM stack at import."""

    def _body(self) -> str:
        start = MCP_SRC.index("def get_plan_quality(")
        rest = MCP_SRC[start:]
        end = rest.index("\n@mcp.tool()") if "\n@mcp.tool()" in rest else len(rest)
        return rest[:end]

    def test_the_planner_owns_the_tool_because_it_owns_the_metrics(self):
        assert "def get_plan_quality(" in MCP_SRC, (
            "the KPI tool is not on the trajectory agent — the cell-data MCP is "
            "byte-level on purpose and grows no typed artefact tools")

    def test_it_calls_report_for_and_scores_nothing_itself(self):
        body = self._body()
        assert "trajectory_quality.report_for(" in body
        for own in ("score_plan(", "leg_metrics(", "WEIGHTS", "efficiency_score ="):
            assert own not in body, \
                f"{own!r} in the tool: the policy moved back out of the module tests can import"

    def test_unavailable_is_caught_before_the_error_it_subclasses(self):
        """Order matters: ``CellStoreUnavailable`` IS a ``CellStoreError``.

        Caught the other way round, an unreachable owner comes back as a plain
        refusal with no way for the card to tell it from a bad tree_id.
        """
        body = self._body()
        assert "CellStoreUnavailable" in body and "CellStoreError" in body
        assert body.index("CellStoreUnavailable") < body.index("except cell_store.CellStoreError")

    def test_neither_failure_answers_success_true(self):
        body = self._body()
        after = body[body.index("except"):]
        assert '"success": True' not in after and "'success': True" not in after


# ─── the dashboard route ─────────────────────────────────────────────────────

QUALITY_ANSWER = {
    "success": True, "cell_id": "some_cell", "tree_id": "bt_1_some_cell",
    "source": "stored", "efficiency_score": 0.72, "verdict": "acceptable",
    "score_components": {"path": 0.61, "schedule": 0.68, "smoothness": 0.93,
                         "dwell": 0.99},
    "score_weights": {"path": 0.40, "schedule": 0.35, "smoothness": 0.15,
                      "dwell": 0.10},
    "summary": {"makespan_s": 58.24, "motion_legs": 4, "unusable_legs": 0,
                "joint_travel_rad": 41.2, "direct_distance_rad": 25.1,
                "total_excess_travel_rad": 16.1, "worst_excess_travel_rad": 10.87,
                "worst_detour_ratio": 3.4, "mean_detour_ratio": 1.6,
                "total_interior_dwell_s": 0.4, "parallel_efficiency": 0.679,
                "serial_fraction": 0.32, "load_balance": 0.49},
    "schedule": {"available": True, "makespan_s": 58.24,
                 "utilisation": {"robot1": 0.49, "robot2": 0.98}},
    "continuity": {},
    "legs": [
        {"usable": True, "node": "weld_robot2_W14", "robot_ns": "robot2",
         "detour_ratio": 3.4, "excess_travel_rad": 2.35, "waypoints": 95,
         "duration_s": 4.1},
        {"usable": True, "node": "home_robot2", "robot_ns": "robot2",
         "detour_ratio": None, "excess_travel_rad": 10.87, "waypoints": 84,
         "duration_s": 6.2},
        {"usable": False, "node": "no_trajectory", "robot_ns": "robot1",
         "excess_travel_rad": 99.0},
    ],
    "safety": {"note": "reported, never scored", "achieved_clearance_m": 0.031},
    "findings": [
        {"severity": "high", "kind": "excess_travel", "robot_ns": "robot2",
         "node": "home_robot2", "detail": "10.87 rad of excess travel"},
        {"severity": "medium", "kind": "idle_arm", "robot_ns": "robot1",
         "node": None, "detail": "robot1 idle 29.43 s (50.5 % of the program)"},
    ],
}


def _body(answer):
    """The route returns ``(payload, status)``; the stub jsonify is identity."""
    payload, status = answer
    return payload, status


class TestTheRouteAsksThePlannerNotTheVolume:

    def test_it_asks_the_trajectory_agent_for_get_plan_quality(self, panel, calls):
        calls.answers["get_plan_quality"] = QUALITY_ANSWER
        payload, status = _body(panel.api_plan_quality("bt_1_some_cell"))

        assert status == 200
        assert payload["efficiency_score"] == 0.72
        assert calls.tools() == ["get_plan_quality"]
        sent = calls.recorded[0]
        assert sent["url"] == panel._traj_mcp(), \
            "the KPIs were asked of the wrong agent"
        assert sent["arguments"] == {"cell_id": panel.CELL_ID,
                                     "tree_id": "bt_1_some_cell"}

    def test_a_refusal_comes_back_200_with_its_reason_intact(self, panel, calls):
        """The card has to render the reason. A status code cannot carry it there.

        Same convention as ``/api/plan_progress``: an MCP-backed read answers 200
        with ``success: false``, and only input validation is a 4xx.
        """
        calls.answers["get_plan_quality"] = {
            "success": False, "error": "cell data unavailable: orchestrator refused"}
        payload, status = _body(panel.api_plan_quality("bt_1_some_cell"))

        assert status == 200
        assert payload["success"] is False
        assert "unavailable" in payload["error"]
        assert "efficiency_score" not in payload, \
            "an unmeasured plan was given a score"

    @pytest.mark.parametrize("hostile", ["../../../etc/passwd", "bt 1", "a" * 400,
                                         "bt_1;rm -rf /", ""])
    def test_a_tree_id_that_is_not_an_id_is_refused_before_any_call(
            self, panel, calls, hostile):
        payload, status = _body(panel.api_plan_quality(hostile))
        assert status == 400
        assert payload["success"] is False
        assert calls.recorded == [], \
            "an unvalidated id was forwarded to the planner"

    def test_the_panel_holds_no_second_implementation_of_the_score(self):
        """Read as imports, not as a grep.

        The card's own comment names ``tools/trajectory_quality.py`` — that is the
        record of where the numbers come from, and it lives in a JS comment inside
        a Python string, so a text search cannot tell it from a dependency. What
        must not exist is the import.
        """
        imported = {
            (node.module or "") + "." + alias.name
            for node in ast.walk(ast.parse(PANEL_SRC))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        } | {
            alias.name
            for node in ast.walk(ast.parse(PANEL_SRC))
            if isinstance(node, ast.Import) for alias in node.names
        }
        # The planner's own package, precisely: ``trajectory_msgs`` is a ROS
        # message package, and ``agents.cell_defaults`` / ``agents.llm_model`` are
        # shared and hold no metric.
        planner_side = sorted(n for n in imported
                              if n.startswith("agents.trajectory"))
        assert planner_side == [], (
            f"the panel imports the planner's modules ({planner_side}): two "
            "answers to 'how good is this plan' drift the first time a weight moves")
        for own in ("score_plan(", "leg_metrics(", "quality_log_file("):
            assert own not in PANEL_CODE, f"{own!r} is called in the panel"


class TestTheRouteIsInTheSpec:
    """``tests/test_openapi_spec.py`` pins both directions; this names the path."""

    def test_the_quality_path_is_documented(self):
        import yaml
        spec = yaml.safe_load(
            (PROJECT_ROOT / "docs" / "api" / "openapi.yaml").read_text())
        path = spec["paths"]["/api/plans/{tree_id}/quality"]
        assert "get" in path
        assert "get_plan_quality" in json.dumps(path), \
            "the spec does not say where the numbers come from"


# ─── the browser ─────────────────────────────────────────────────────────────

QUALITY_URL = "/api/plans/bt_1_some_cell/quality"


KPI_READ = textwrap.dedent("""
    function kpiRead() {
      return {
        status: (REG["bt-kpi-status"] || {}).textContent || "",
        hidden: (REG["bt-kpi-status"] || {style:{}}).style.display || "",
        source: (REG["bt-kpi-source"] || {}).textContent || "",
        score: HTML["bt-kpi-score"] || "",
        open: HTML["bt-kpi-open"] || "",
        modal: (REG["bt-kpi-modal"] || {style:{}}).style.display || "",
        modalScore: HTML["bt-kpi-modal-score"] || "",
        tree: (REG["bt-kpi-modal-tree"] || {}).textContent || "",
        bars: HTML["bt-kpi-bars"] || "",
        grid: HTML["bt-kpi-grid"] || "",
        findings: HTML["bt-kpi-findings"] || "",
        legs: HTML["bt-kpi-legs"] || "",
        safety: HTML["bt-kpi-safety"] || ""};
    }
""")


def _kpi_probe(extra_routes: dict, call: str, then: str = "") -> str:
    """Boot, run *call*, settle, run *then*, print the card and window state.

    ``then`` is where the interaction goes: the answer has arrived and both
    surfaces are filled by the time it runs, so a click in it sees what an
    operator's click would.
    """
    routes = "".join(f"ROUTES[{json.dumps(k)}] = {json.dumps(v)};\n"
                     for k, v in extra_routes.items())
    return KPI_READ + textwrap.dedent("""
        %s
        %s
        setImmediate(() => setImmediate(() => setImmediate(() => {
          %s
          console.log("OUT=" + JSON.stringify(kpiRead()));
        })));
    """) % (routes, call, then or "")


def _out(stdout: str) -> dict:
    line = next((l for l in stdout.splitlines() if l.startswith("OUT=")), None)
    assert line, f"the probe printed no OUT= line:\n{stdout}"
    return json.loads(line[len("OUT="):])


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheCardRendersWhatMcpReturned:

    def test_selecting_a_plan_shows_its_score_and_verdict(self, tmp_path):
        """The whole request: the numbers are visible in the dashboard."""
        probe = _kpi_probe(
            {"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: QUALITY_ANSWER},
            "loadTree('bt_1_some_cell');")
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)

        assert "0.72" in out["score"], f"no score rendered: {out}"
        assert "acceptable" in out["score"].lower(), out["score"]
        assert out["hidden"] == "none", "the placeholder stayed over the numbers"
        # Which of the four the score is made of — a bare number says nothing
        # about what to fix.
        for component in ("path", "schedule", "smoothness", "dwell"):
            assert component in out["bars"], f"{component} bar missing: {out['bars']}"
        assert "16.10 rad" in out["grid"], f"excess travel missing: {out['grid']}"
        assert "68%" in out["grid"], f"parallel efficiency missing: {out['grid']}"

    def test_the_findings_are_shown_worst_first_and_name_the_leg(self, tmp_path):
        probe = _kpi_probe(
            {"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: QUALITY_ANSWER},
            "loadTree('bt_1_some_cell');")
        out = _out(_run_probe(tmp_path, probe).stdout)
        f = out["findings"]
        assert "excess_travel" in f and "home_robot2" in f, f
        assert f.index("excess_travel") < f.index("idle_arm"), \
            "the ranking the report produced was not preserved"
        assert "10.87 rad" in f

    def test_the_worst_legs_are_ranked_by_radians_not_by_ratio(self, tmp_path):
        """A short 3.4x transit must not outrank a home return wandering 10 rad."""
        out = _out(_run_probe(tmp_path, _kpi_probe(
            {"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: QUALITY_ANSWER},
            "loadTree('bt_1_some_cell');")).stdout)
        legs = out["legs"]
        assert legs.index("home_robot2") < legs.index("weld_robot2_W14"), legs
        assert "no_trajectory" not in legs, \
            "a leg with no trajectory was ranked among the worst by its excess"

    def test_a_plan_that_was_scored_on_demand_says_so(self, tmp_path):
        answer = dict(QUALITY_ANSWER, source="computed")
        out = _out(_run_probe(tmp_path, _kpi_probe(
            {"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: answer},
            "loadTree('bt_1_some_cell');")).stdout)
        assert "on demand" in out["source"], out["source"]

    def test_a_refusal_renders_its_reason_and_never_a_score(self, tmp_path):
        """The failure this card exists to avoid becoming."""
        answer = {"success": False,
                  "error": "cell data unavailable: orchestrator is not answering"}
        out = _out(_run_probe(tmp_path, _kpi_probe(
            {"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: answer},
            "loadTree('bt_1_some_cell');")).stdout)

        assert "not answering" in out["status"], out
        assert out["score"] == "", "an unreachable planner rendered as a score"
        assert out["bars"] == "" and out["grid"] == ""
        assert "poor" not in out["status"].lower()

    def test_a_metric_the_report_could_not_measure_prints_a_dash(self, tmp_path):
        """None is not zero. A zero there reads as a measured perfect score."""
        answer = json.loads(json.dumps(QUALITY_ANSWER))
        answer["summary"]["load_balance"] = None
        answer["summary"]["total_excess_travel_rad"] = None
        answer["schedule"]["utilisation"] = {}
        out = _out(_run_probe(tmp_path, _kpi_probe(
            {"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: answer},
            "loadTree('bt_1_some_cell');")).stdout)

        assert "—" in out["grid"], f"a missing metric was invented: {out['grid']}"
        assert "0.00 rad" not in out["grid"]
        assert "0%" not in out["grid"]

    def test_a_failed_run_is_told_it_is_not_scored(self, tmp_path):
        failed = {"success": True, "tree_id": "bt_bad_some_cell", "status": "failed",
                  "executable": False, "num_operations": 0, "trajectory_sizes": {},
                  "behaviour_tree": None, "error": "ExpiredTokenException"}
        out = _out(_run_probe(tmp_path, _kpi_probe(
            {"/api/plans/bt_bad_some_cell": failed},
            "loadTree('bt_bad_some_cell');")).stdout)

        assert "not scored" in out["status"], out["status"]
        assert out["score"] == "", "a fragment was given a score"

    def test_refresh_re_asks_about_the_tree_on_screen(self, tmp_path):
        """Not the selector: it moves before ``loadTree`` resolves 'latest'."""
        probe = _kpi_probe(
            {"/api/plans/latest": PLAN_BODY,
             "/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: QUALITY_ANSWER},
            textwrap.dedent("""
                loadTree('latest');
                setImmediate(() => setImmediate(() => {
                  const seen = [];
                  const real = globalThis.fetch;
                  globalThis.fetch = (u, o) => { seen.push(String(u)); return real(u, o); };
                  document.getElementById('bt-select').value = 'bt_other';
                  kpiRefresh();
                  setImmediate(() => console.log("ASKED=" + JSON.stringify(seen)));
                }));
            """))
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        asked = json.loads(
            next(l for l in done.stdout.splitlines()
                 if l.startswith("ASKED="))[len("ASKED="):])
        assert asked == [QUALITY_URL], (
            f"refresh asked about a plan the pane is not showing: {asked}")


# ─── the metric is published; the report is a window ─────────────────────────

MANY_FINDINGS = json.loads(json.dumps(QUALITY_ANSWER))
MANY_FINDINGS["findings"] = [
    {"severity": "low", "kind": f"kind_{n}", "robot_ns": "robot1",
     "node": f"leg_{n}", "detail": f"finding number {n}"}
    for n in range(11)
]


class TestTheCardPublishesOnlyTheMetric:
    """Static, on the markup: WHERE each block lives decides what the pane shows.

    The pane's job is the tree. The report inline was thirty numbers between the
    plan selector and the drawing, so a reader who came for the tree scrolled past
    a page of metrics they had not asked for on every load.
    """

    def _page(self) -> str:
        return _render(["robot1", "robot2"])

    def _modal(self, html: str) -> str:
        start = html.index('<div id="bt-kpi-modal"')
        return html[start:html.index("<script>", start)]

    def _card(self, html: str) -> str:
        start = html.index('<div id="bt-kpi"')
        return html[start:html.index('<div id="bt-canvas"', start)]

    @pytest.mark.parametrize("block", ["bt-kpi-bars", "bt-kpi-grid",
                                       "bt-kpi-findings", "bt-kpi-legs",
                                       "bt-kpi-safety"])
    def test_every_detail_block_lives_in_the_window_not_the_card(self, block):
        html = self._page()
        assert f'id="{block}"' in self._modal(html), \
            f"{block} is not in the dedicated window"
        assert f'id="{block}"' not in self._card(html), \
            f"{block} is still inline in the pane — the detail was not moved out"

    def test_the_card_keeps_the_score_and_the_way_into_the_report(self):
        card = self._card(self._page())
        assert 'id="bt-kpi-score"' in card, "the metric is not published at all"
        assert 'id="bt-kpi-open"' in card, \
            "nothing in the card opens the report; a hidden report is not a report"
        assert "kpiOpen()" in card

    def test_the_window_starts_hidden_and_is_not_nested_in_a_pane(self):
        html = self._page()
        modal = self._modal(html)
        assert "display:none" in modal.split(">")[0], \
            "the report window is on screen before anything is measured"
        # position:fixed inside a scrolling, resizable pane is clipped by it, which
        # turns the report into a sliver of the report.
        assert "position:fixed" in modal.split(">")[0]
        assert html.index('<div id="bt-kpi-modal"') > html.index("/chat-panel"), \
            "the window is inside a pane; it must be a sibling of the layout"

    def test_the_window_can_be_dismissed_three_ways(self):
        modal = self._modal(self._page())
        assert "kpiClose()" in modal, "no close button"
        assert "kpiBackdrop(event)" in modal, "the backdrop does not dismiss"
        from tests.test_dashboard_asks_mcp_not_the_volume import PANEL_SRC as src
        assert "document.addEventListener('keydown',kpiKey)" in src, \
            "Escape does not close it — a mouse-only dialog stays open over the tree"


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheDetailOpensOnlyOnAClick:

    def _probe(self, then="", answer=QUALITY_ANSWER, routes=None):
        return _kpi_probe(
            dict(routes or {},
                 **{"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: answer}),
            "loadTree('bt_1_some_cell');", then)

    def test_the_window_stays_shut_until_it_is_asked_for(self, tmp_path):
        out = _out(_run_probe(tmp_path, self._probe()).stdout)
        assert out["modal"] == "none", \
            "the report opened itself over the tree the operator selected"
        assert "0.72" in out["score"], "the metric was not published in the card"

    def test_clicking_the_metric_opens_the_report(self, tmp_path):
        out = _out(_run_probe(tmp_path, self._probe("kpiOpen();")).stdout)
        assert out["modal"] == "flex", "the click did not open the window"
        assert "0.72" in out["modalScore"], \
            "the window does not repeat the number it is explaining"
        assert "bt_1_some_cell" in out["tree"], \
            "the window does not say which plan it is about"
        for block in ("bars", "grid", "findings", "legs", "safety"):
            assert out[block], f"the window opened without its {block}"

    def test_opening_it_asks_the_planner_nothing_a_second_time(self, tmp_path):
        """Both surfaces are filled by the one fetch; the click is presentation.

        A second round trip on open would also mean the window could disagree with
        the card it was opened from.
        """
        probe = self._probe(textwrap.dedent("""
            const seen = [];
            const real = globalThis.fetch;
            globalThis.fetch = (u, o) => { seen.push(String(u)); return real(u, o); };
            kpiOpen();
            console.log("ASKED=" + JSON.stringify(seen));
        """))
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        asked = json.loads(next(l for l in done.stdout.splitlines()
                                if l.startswith("ASKED="))[len("ASKED="):])
        assert asked == [], f"opening the report re-fetched: {asked}"

    def test_the_close_button_shuts_it(self, tmp_path):
        out = _out(_run_probe(tmp_path, self._probe("kpiOpen();kpiClose();")).stdout)
        assert out["modal"] == "none"

    def test_escape_shuts_it(self, tmp_path):
        out = _out(_run_probe(
            tmp_path, self._probe("kpiOpen();kpiKey({key:'Escape'});")).stdout)
        assert out["modal"] == "none"

    def test_a_click_on_the_backdrop_shuts_it_and_one_on_the_report_does_not(
            self, tmp_path):
        """Selecting a leg name to copy it must not dismiss what is being read."""
        inside = _out(_run_probe(tmp_path, self._probe(
            "kpiOpen();kpiBackdrop({target:{id:'bt-kpi-sheet'}});")).stdout)
        assert inside["modal"] == "flex", \
            "a click inside the report dismissed it"
        outside = _out(_run_probe(tmp_path, self._probe(
            "kpiOpen();kpiBackdrop({target:{id:'bt-kpi-modal'}});")).stdout)
        assert outside["modal"] == "none"

    def test_a_plan_that_is_not_measured_has_no_report_to_open(self, tmp_path):
        """kpiOpen must refuse, not present an empty window over the reason."""
        answer = {"success": False, "error": "cell data unavailable: no answer"}
        out = _out(_run_probe(tmp_path, self._probe("kpiOpen();",
                                                    answer=answer)).stdout)
        assert out["modal"] == "none", \
            "an empty report window covered the explanation of why there is none"
        assert "no answer" in out["status"]

    def test_selecting_another_plan_drops_the_report_at_once(self, tmp_path):
        """The stickiest bug in a dialog: the numbers outliving their plan.

        Read synchronously, before the new plan's answer can arrive. For the length
        of that fetch the card would otherwise publish one plan's score under
        another plan's name, and if the newly selected run turns out not to be
        scorable at all — a FAILED one — the open window would keep showing the
        previous plan's report with a failed run behind it.
        """
        failed = {"success": True, "tree_id": "bt_bad_some_cell", "status": "failed",
                  "executable": False, "num_operations": 0, "trajectory_sizes": {},
                  "behaviour_tree": None, "error": "ExpiredTokenException"}
        probe = _kpi_probe(
            {"/api/plans/bt_1_some_cell": PLAN_BODY, QUALITY_URL: QUALITY_ANSWER,
             "/api/plans/bt_bad_some_cell": failed},
            "loadTree('bt_1_some_cell');",
            "kpiOpen();loadTree('bt_bad_some_cell');")
        out = _out(_run_probe(tmp_path, probe).stdout)
        assert out["modal"] == "none", \
            "the previous plan's report stayed open over another plan"
        assert out["score"] == "" and out["open"] == "", \
            "the card still published the previous plan's metric"
        assert "bt_bad_some_cell" in out["status"], out["status"]

    def test_the_card_says_how_much_is_behind_the_click(self, tmp_path):
        """"Details" alone gives a reader no idea whether it is worth opening."""
        out = _out(_run_probe(tmp_path, self._probe()).stdout)
        assert "2 findings" in out["open"], out["open"]
        clean = json.loads(json.dumps(QUALITY_ANSWER))
        clean["findings"] = []
        out = _out(_run_probe(tmp_path, self._probe(answer=clean)).stdout)
        assert "no findings" in out["open"], out["open"]

    def test_the_window_lists_every_finding_not_the_first_few(self, tmp_path):
        """The cap existed because the pane had no room. The window scrolls.

        The tail is what a reader opens the report to hunt for — a truncated list
        hides exactly the low-severity findings that are not in the headline.
        """
        out = _out(_run_probe(tmp_path,
                              self._probe(answer=MANY_FINDINGS)).stdout)
        for n in range(11):
            assert f"kind_{n}" in out["findings"], \
                f"finding {n} of 11 was dropped: {out['findings'][:400]}"
        assert "more in the quality log" not in out["findings"]
        assert "11 findings" in out["open"]

    def test_clearance_is_in_the_window_and_in_no_score(self, tmp_path):
        out = _out(_run_probe(tmp_path, self._probe("kpiOpen();")).stdout)
        assert "0.031 m" in out["safety"], f"clearance not reported: {out['safety']}"
        assert "never scored" in out["safety"], \
            "the exclusion is not stated, so the margin reads as a KPI to push"
        for scored in (out["bars"], out["grid"], out["score"], out["modalScore"]):
            assert "clearance" not in scored.lower(), \
                "a clearance was rendered inside a score block"
