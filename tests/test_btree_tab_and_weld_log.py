"""The BTree tab (tree + Replay) and the Log tab's per-weld story.

Five reported defects, each with a test that would have caught it:

1. **Replay threw the operator off the page.** It ended with ``switchTab('chat')``,
   so pressing it in the recordings browser jumped panes and the thing you had just
   launched vanished. Replay now reports in place and the tab does not move.
2. **The replayable list was the wrong list.** ``/a2a/replay`` takes a *tree_id* and
   executes a CALCULATED plan, but the button lived next to a list of *recordings* —
   i.e. only plans that had already run at least once. Every plan calculated and
   never executed was invisible. The button now hangs off the plan selector.
3. **A Sequence looked exactly like a Parallel.** Every parent laid its children out
   side by side, so the drawing could not distinguish "weld, then move, then weld"
   from "weld while moving" — different cell behaviours, different cycle times, same
   picture. Ordered children now carry their step number and an arrow to the next.
4. **The log could not answer "was this weld allocated, and how did it get there?"**
   The points column showed a one-word status; the failures and the reasoning behind
   the attempt that finally worked were not reachable from a weld at all.
5. **The viewer could not open a failed run.** Selecting the run whose token expired
   gave "Could not load … is a FAILED run and is not executable". The refusal was
   right and in the wrong place: failed runs are listed on purpose, so refusing to
   *read* one is a dead end at the run most worth reading. Reading is now allowed and
   marked; the refusal moved to the execute path, where the cell's own `replay_plan`
   enforces it rather than trusting a disabled button.

The JS-engine half reuses the harness in :mod:`test_dashboard_script_loads` rather
than reinventing a DOM stub — one harness, so a stub artefact cannot pass here and
fail there.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

from tests.test_dashboard_script_loads import (  # the served page, and how to boot it
    NODE,
    PLAN_BODY,
    PLAN_LIST,
    _run_probe,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = (ROOT / "dashboard" / "web_control_panel.py").read_text()


def _pane(name: str) -> str:
    """The markup of one tab pane, so 'is it in THIS tab' is answerable."""
    start = PANEL.index(f'id="pane-{name}"')
    rest = PANEL[start:]
    nxt = [i for i in (rest.find('id="pane-'), rest.find('id="tip"')) if i > 0]
    return rest[:min(nxt)] if nxt else rest


def _fn(name: str) -> str:
    """One JS function's body, by brace matching. Used to assert what it does NOT do."""
    start = PANEL.index(f"function {name}(")
    depth, i = 0, PANEL.index("{", start)
    for j in range(i, len(PANEL)):
        if PANEL[j] == "{":
            depth += 1
        elif PANEL[j] == "}":
            depth -= 1
            if depth == 0:
                return PANEL[start:j + 1]
    pytest.fail(f"{name} has unbalanced braces")


# ── the tabs themselves ──────────────────────────────────────────────────────

def test_the_rec_tab_is_gone_entirely() -> None:
    """Removed, not hidden: a dead pane is a place bugs keep living."""
    for gone in ('id="tab-rec"', 'id="pane-rec"', 'id="rec-list"', 'id="rec-speed2"',
                 "function loadRecTab", "function selectRec", "function replaySelTab",
                 "_selectedRec"):
        assert gone not in PANEL, gone


def test_the_tab_list_no_longer_carries_rec() -> None:
    """TABS drives show/hide; a stale name there means switchTab looks up a null."""
    tabs = re.search(r"const TABS=\[(.+?)\];", PANEL)
    assert tabs and "'rec'" not in tabs.group(1)


def test_the_tabs_are_named_btree_and_log() -> None:
    tree = PANEL[PANEL.index('id="tab-tree"'):]
    assert "BTree" in tree[:tree.index("</button>")]
    plan = PANEL[PANEL.index('id="tab-plan"'):]
    label = plan[:plan.index("</button>")]
    assert "Log" in label and "Plan Log" not in label


# ── replay, in the tree tab ──────────────────────────────────────────────────

def test_replay_lives_in_the_btree_tab() -> None:
    pane = _pane("tree")
    for element in ('id="bt-replay"', 'id="bt-speed"', 'id="bt-replay-status"'):
        assert element in pane, element
    assert "btReplay()" in pane


def test_replay_does_not_change_tab() -> None:
    """The reported defect: pressing Replay navigated away from the tree."""
    body = _fn("btReplay")
    assert "switchTab" not in body
    assert "/api/a2a/replay" in body
    assert "bt-replay-status" in body


def test_replay_is_driven_by_the_plan_selector_not_a_recording_list() -> None:
    """Every CALCULATED plan must be replayable — that is the list /api/plans returns."""
    body = _fn("btReplay")
    assert "bt-select" in body
    assert "recordings" not in body


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_pressing_replay_posts_the_selected_tree_and_stays_put(tmp_path) -> None:
    probe = textwrap.dedent("""
        const CALLS = [];
        globalThis.fetch = (url, opt) => {
          CALLS.push([String(url), (opt && opt.body) || null]);
          return Promise.resolve({ ok: true, status: 200,
            json: () => Promise.resolve({ run_id: 'run-7' }) });
        };
        _activeTab = 'tree';
        document.getElementById('bt-select').value = 'bt_9_some_cell';
        document.getElementById('bt-speed').value = '2';
        btReplay();
        setImmediate(() => setImmediate(() => setImmediate(() => {
          console.log('CALLS=' + JSON.stringify(CALLS));
          console.log('tab=' + _activeTab);
          console.log('status=' + REG['bt-replay-status'].textContent);
        })));
    """)
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    calls = json.loads(re.search(r"CALLS=(\[.*\])", done.stdout).group(1))
    urls = [c[0] for c in calls]
    assert "/api/a2a/replay" in urls, urls
    body = json.loads([c[1] for c in calls if c[0] == "/api/a2a/replay"][0])
    assert body["tree_id"] == "bt_9_some_cell"
    # The speed is the cell's velocity scale, set on its own endpoint.
    assert "/api/speed" in urls
    # The defect, pinned: the operator is still looking at the tree they launched.
    assert "tab=tree" in done.stdout, done.stdout
    assert "run-7" in done.stdout, done.stdout


# ── the sequence is drawn as a sequence ──────────────────────────────────────

PARALLEL_BODY = json.loads(json.dumps(PLAN_BODY))
PARALLEL_BODY["behaviour_tree"]["root"]["type"] = "Parallel"

_COUNT_PROBE = """
    setImmediate(() => setImmediate(() => {
      const svg = REG['bt-svg'];
      const lines = svg.children.filter(c => c.id === 'el:line').length;
      const dashed = svg.children.filter(c => c.id === 'el:path'
                                        && c.attrs['stroke-dasharray']).length;
      const slots = [];
      svg.children.filter(c => c.id === 'el:g').forEach(g =>
        (g.children || []).forEach(ch => {
          if (ch.id === 'el:text' && /^(\\d+|\\u2225)$/.test(ch.textContent))
            slots.push(ch.textContent);
        }));
      console.log('lines=' + lines + ' dashed=' + dashed
                  + ' slots=' + JSON.stringify(slots));
    }));
"""


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_a_sequence_gets_numbered_children_and_an_order_arrow(tmp_path) -> None:
    """The fix for "the blocks under the sequence all seem parallel"."""
    probe = (f"ROUTES['/api/plans'] = {json.dumps(PLAN_LIST)};\n"
             f"ROUTES['/api/plans/bt_1_some_cell'] = {json.dumps(PLAN_BODY)};\n"
             "loadPlanList();\n" + _COUNT_PROBE)
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    # Two children run one after the other: one arrow between them, and each box
    # says which step it is.
    assert "lines=1" in done.stdout, done.stdout
    assert 'slots=["1","2"]' in done.stdout, done.stdout
    # Ordered edges are solid — dashing is reserved for concurrency.
    assert "dashed=0" in done.stdout, done.stdout


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_a_parallel_gets_no_order_arrows_and_says_so(tmp_path) -> None:
    """Same two children, genuinely concurrent — and drawn differently."""
    probe = (f"ROUTES['/api/plans'] = {json.dumps(PLAN_LIST)};\n"
             f"ROUTES['/api/plans/bt_1_some_cell'] = {json.dumps(PARALLEL_BODY)};\n"
             "loadPlanList();\n" + _COUNT_PROBE)
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    assert "lines=0" in done.stdout, done.stdout
    assert "dashed=2" in done.stdout, done.stdout
    assert "slots=" in done.stdout and '"1"' not in done.stdout.split("slots=")[1], \
        f"a Parallel's children must not be numbered: {done.stdout}"


def test_the_notation_is_explained_on_the_page() -> None:
    """A numbered box means nothing if the page never says what the number is."""
    pane = _pane("tree")
    assert "runs in order" in pane and "same time" in pane


# ── one weld's story ─────────────────────────────────────────────────────────

def test_the_log_tab_serves_a_per_weld_detail_block() -> None:
    assert 'id="plan-point-detail"' in _pane("plan")
    assert "function planRenderPointDetail" in PANEL


def test_allocation_is_read_from_the_flags_not_from_the_last_success() -> None:
    """"Planned" is not "allocated", and conflating them ships a plan with a hole.

    A trajectory can exist for a weld no tree node ever runs. Only ``in_tree``
    means a robot will actually perform it, so only ``in_tree`` may say allocated.
    """
    table = re.search(r"const PLAN_ALLOC=\{(.+?)\n\};", PANEL, re.S)
    assert table, "the allocation verdicts are not a table any more"
    body = table.group(1)
    in_tree = re.search(r"in_tree:\s*\['([^']+)'", body)
    assert in_tree and "allocated" in in_tree.group(1)
    for status in ("planned", "validated", "failed", "pending"):
        line = re.search(status + r":\s*\['([^']+)'", body)
        assert line, status
        assert "✓" not in line.group(1), \
            f"{status} claims allocation: {line.group(1)}"


TRACE_LIST = {"cell_id": "some_cell",
              "traces": [{"trace_id": "t1", "started_at": "2026-09-08T10:00:00",
                          "tool_calls": 3, "success": True}]}
# A weld that failed twice and then worked — the case the log exists for.
TRACE = {
    "success": True, "trace_id": "t1",
    "run": {"trace_id": "t1", "success": True, "tool_calls": 3, "step_budget": 100,
            "model": "opus-5", "backend": "ompl", "points_by_status": {"in_tree": 1}},
    "points": [{
        "operation_id": "W12", "type": "WELDING", "robot": "robot1",
        "position": [1.0, 2.0, 3.0], "attempts": 3, "decisions": [1, 2, 3],
        "flags": {"failed": True, "planned": True, "in_tree": True, "last": "in_tree"},
        "metrics": {}, "status": "in_tree",
        "failures": [{"operation_id": "W12", "error": "clamp in the approach cone"}],
    }],
    "decisions": [
        {"seq": 1, "kind": "decision", "step": 1, "tool": "plan_weld",
         "operations": ["W12"], "robots": ["robot1"], "why": "standard 0.10 m standoff",
         "args": {"approach_offset_m": 0.10},
         "result": {"ok": False, "failures": [
             {"operation_id": "W12", "error": "clamp in the approach cone"}]}},
        {"seq": 2, "kind": "decision", "step": 2, "tool": "plan_weld",
         "operations": ["W12"], "robots": ["robot1"], "why": "still colliding",
         "args": {"approach_offset_m": 0.09},
         "result": {"ok": False, "failures": [
             {"operation_id": "W12", "error": "no IK solution at that angle"}]}},
        {"seq": 3, "kind": "decision", "step": 3, "tool": "plan_weld",
         "operations": ["W12"], "robots": ["robot1"],
         "why": "shrink the standoff to 0.07 m so the tool clears the clamp",
         "args": {"approach_offset_m": 0.07},
         "result": {"ok": True, "metrics": {"waypoints": 95}}},
    ],
}

_DOSSIER_PROBE = textwrap.dedent("""
    const txt = el => {
      let s = String(el.textContent || '');
      (el.children || []).forEach(c => { s += ' ' + txt(c); });
      return s;
    };
    planLoadHistory();
    setImmediate(() => setImmediate(() => setImmediate(() => {
      planSelectPoint('W12');
      console.log('DOSSIER=' + JSON.stringify(txt(REG['plan-point-detail'])));
      console.log('POINTS=' + JSON.stringify(txt(REG['plan-points'])));
    })));
""")


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_a_weld_shows_its_verdict_its_failures_and_the_winning_reasoning(tmp_path) -> None:
    """The request, in one test: allocated or not, the failure history, the why."""
    probe = (f"ROUTES['/api/a2a/plan_traces'] = {json.dumps(TRACE_LIST)};\n"
             f"ROUTES['/api/a2a/plan_trace/t1'] = {json.dumps(TRACE)};\n"
             + _DOSSIER_PROBE)
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    dossier = json.loads(re.search(r"DOSSIER=(\".*\")", done.stdout).group(1))
    assert "W12" in dossier
    assert "allocated" in dossier and "robot1" in dossier
    # Both failures, with the errors that ended them — in order, not summarised away.
    assert "clamp in the approach cone" in dossier
    assert "no IK solution at that angle" in dossier
    assert "2 failed" in dossier and "1 succeeded" in dossier
    # And the reasoning behind the attempt that worked, which is the point.
    assert "shrink the standoff to 0.07" in dossier
    assert "led to success" in dossier


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_the_points_column_marks_allocation_and_failure_count(tmp_path) -> None:
    """Scannable at a glance: which welds a robot will actually perform."""
    probe = (f"ROUTES['/api/a2a/plan_traces'] = {json.dumps(TRACE_LIST)};\n"
             f"ROUTES['/api/a2a/plan_trace/t1'] = {json.dumps(TRACE)};\n"
             + _DOSSIER_PROBE)
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    points = json.loads(re.search(r"POINTS=(\".*\")", done.stdout).group(1))
    assert "W12" in points and "in_tree" in points
    assert "1 failure" in points, points
    assert "3 ×" in points, points


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_a_weld_nothing_touched_is_reported_as_never_attempted(tmp_path) -> None:
    """The quiet failure mode: a weld the planner simply never got to.

    It must not read as a success and must not read as a failure — neither is what
    happened, and both send the reader looking for the wrong thing.
    """
    trace = json.loads(json.dumps(TRACE))
    trace["points"].append({
        "operation_id": "W15", "type": "WELDING", "robot": None, "position": None,
        "attempts": 0, "decisions": [], "flags": {}, "metrics": {},
        "failures": [], "status": "pending"})
    probe = (f"ROUTES['/api/a2a/plan_traces'] = {json.dumps(TRACE_LIST)};\n"
             f"ROUTES['/api/a2a/plan_trace/t1'] = {json.dumps(trace)};\n"
             + _DOSSIER_PROBE.replace("planSelectPoint('W12')", "planSelectPoint('W15')")
                             .replace("W12", "W15"))
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    dossier = json.loads(re.search(r"DOSSIER=(\".*\")", done.stdout).group(1))
    assert "never attempted" in dossier, dossier
    assert "not assigned" in dossier, dossier
    assert "No decision named this weld" in dossier, dossier


def test_the_weld_dossier_goes_in_via_textcontent() -> None:
    """Weld ids, error strings and model prose. Never innerHTML."""
    body = _fn("planRenderPointDetail")
    assert "innerHTML" not in body


# ── a FAILED run reads, but never runs ───────────────────────────────────────
# Reported live, on the BTree tab: selecting the run whose token expired gave
#   "Could not load bt_1788893504_welding_cell_2_small: plan ... is a FAILED run and is
#    not executable: ExpiredTokenException ..."
# The refusal was right and in the wrong place. Failed runs are LISTED on purpose —
# a token that expired at weld 13 says nothing about welds 1-12 — so a viewer that
# cannot open what the list shows is a dead end at exactly the run the operator most
# needs to read. Reading is not executing; the refusal belongs on the execute path.

FAILED_PLAN_LIST = {"cell_id": "some_cell", "count": 2, "ok": 1, "failed": 1,
                    "plans": [
                        {"tree_id": "bt_bad_some_cell", "status": "failed",
                         "error": "ExpiredTokenException", "error_kind": "credentials",
                         "num_planned": 13, "num_operations": 15, "resumable": True},
                        {"tree_id": "bt_1_some_cell", "status": "ok",
                         "num_operations": 2, "resumable": False}]}


def test_the_viewer_endpoint_opts_into_failed_runs() -> None:
    """It is a reader. The execute path is /a2a/replay, and it does NOT opt in."""
    src = (ROOT / "agents" / "orchestrator" / "server.py").read_text()
    viewer = src[src.index("def a2a_plan(tree_id"):src.index("def a2a_plan_traces")]
    assert '"include_failed": True' in viewer
    replay = src[src.index("def a2a_replay("):src.index("def a2a_move(")]
    assert "include_failed" not in replay


def test_the_executor_refuses_a_failed_record_itself() -> None:
    """Defence where it counts: the MCP tool, not the page that offered the button."""
    src = (ROOT / "agents" / "cell" / "mcp_server.py").read_text()
    fn = src[src.index("def replay_plan("):src.index("def list_models(")]
    # Refused before the tree is taken out of the record, so the
    # `behaviour_tree or plan` fallback can never pass a failure report on.
    assert fn.index('== "failed"') < fn.index('bt = plan.get("behaviour_tree")')


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_a_failed_run_is_labelled_in_the_plan_selector(tmp_path) -> None:
    """The refusal has to be visible BEFORE the click, not after it."""
    probe = (f"ROUTES['/api/plans'] = {json.dumps(FAILED_PLAN_LIST)};\n"
             f"ROUTES['/api/plans/bt_bad_some_cell'] = " + json.dumps({
                 "success": True, "tree_id": "bt_bad_some_cell", "status": "failed",
                 "executable": False, "num_operations": 13, "trajectory_sizes": {},
                 "behaviour_tree": None, "error": "ExpiredTokenException",
                 "error_kind": "credentials", "resumable": True}) + ";\n"
             "_btWant='bt_bad_some_cell';\n"
             "loadPlanList();\n"
             "setImmediate(() => setImmediate(() => setImmediate(() => {\n"
             "  console.log('OPTIONS=' + JSON.stringify(HTML['bt-select'] || ''));\n"
             "  console.log('META=' + JSON.stringify(REG['bt-meta'].textContent));\n"
             "  console.log('SAY=' + JSON.stringify(REG['bt-replay-status'].textContent));\n"
             "})));\n")
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    options = json.loads(re.search(r"OPTIONS=(\".*\")", done.stdout).group(1))
    assert "FAILED" in options, options
    assert "13 solved" in options, options
    meta = json.loads(re.search(r"META=(\".*\")", done.stdout).group(1))
    # The reported message said "Could not load". It loads; it just cannot run.
    assert "Could not load" not in meta, meta
    assert "FAILED" in meta and "ExpiredTokenException" in meta, meta
    assert "resumable" in meta, meta
    say = json.loads(re.search(r"SAY=(\".*\")", done.stdout).group(1))
    assert "FAILED" in say or "before it built a tree" in say, say


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_replay_refuses_a_failed_run_without_calling_the_cell(tmp_path) -> None:
    """The arms must not move through a tree the assembler rejected."""
    probe = (f"ROUTES['/api/plans'] = {json.dumps(FAILED_PLAN_LIST)};\n"
             f"ROUTES['/api/plans/bt_bad_some_cell'] = " + json.dumps({
                 "success": True, "tree_id": "bt_bad_some_cell", "status": "failed",
                 "executable": False, "num_operations": 13, "trajectory_sizes": {},
                 "behaviour_tree": None, "error": "ExpiredTokenException",
                 "resumable": True}) + ";\n"
             "_btWant='bt_bad_some_cell';\n"
             "loadPlanList();\n"
             "setImmediate(() => setImmediate(() => setImmediate(() => {\n"
             "  const CALLS = [];\n"
             "  globalThis.fetch = (url, opt) => { CALLS.push(String(url));\n"
             "    return Promise.resolve({ok: true, status: 200,\n"
             "      json: () => Promise.resolve({run_id: 'nope'})}); };\n"
             "  document.getElementById('bt-select').value = 'bt_bad_some_cell';\n"
             "  btReplay();\n"
             "  setImmediate(() => {\n"
             "    console.log('CALLS=' + JSON.stringify(CALLS));\n"
             "    console.log('SAY=' + JSON.stringify(REG['bt-replay-status'].textContent));\n"
             "  });\n"
             "})));\n")
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    calls = json.loads(re.search(r"CALLS=(\[.*\])", done.stdout).group(1))
    assert calls == [], f"a failed run reached the cell: {calls}"
    say = json.loads(re.search(r"SAY=(\".*\")", done.stdout).group(1))
    assert "FAILED" in say and "not executable" in say, say
    # It points at the one action that IS available for this run.
    assert "Resume" in say, say


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_selecting_a_good_plan_after_a_failed_one_re_enables_replay(tmp_path) -> None:
    """The stickiest bug in a guard like this: the refusal outliving its plan."""
    probe = (f"ROUTES['/api/plans'] = {json.dumps(FAILED_PLAN_LIST)};\n"
             f"ROUTES['/api/plans/bt_bad_some_cell'] = " + json.dumps({
                 "success": True, "tree_id": "bt_bad_some_cell", "status": "failed",
                 "executable": False, "num_operations": 0, "trajectory_sizes": {},
                 "behaviour_tree": None, "error": "ExpiredTokenException"}) + ";\n"
             f"ROUTES['/api/plans/bt_1_some_cell'] = {json.dumps(PLAN_BODY)};\n"
             "_btWant='bt_bad_some_cell';\n"
             "loadPlanList();\n"
             "setImmediate(() => setImmediate(() => setImmediate(() => {\n"
             "  loadTree('bt_1_some_cell');\n"
             "  setImmediate(() => setImmediate(() => {\n"
             "    const CALLS = [];\n"
             "    globalThis.fetch = (url, opt) => { CALLS.push(String(url));\n"
             "      return Promise.resolve({ok: true, status: 200,\n"
             "        json: () => Promise.resolve({run_id: 'r1'})}); };\n"
             "    document.getElementById('bt-select').value = 'bt_1_some_cell';\n"
             "    btReplay();\n"
             "    setImmediate(() => setImmediate(() =>\n"
             "      console.log('CALLS=' + JSON.stringify(CALLS))));\n"
             "  }));\n"
             "})));\n")
    done = _run_probe(tmp_path, probe)
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
    calls = json.loads(re.search(r"CALLS=(\[.*\])", done.stdout).group(1))
    assert "/api/a2a/replay" in calls, f"the good plan was refused too: {calls}"


# ── the endpoint the viewer calls ────────────────────────────────────────────

try:
    import flask  # noqa: F401

    from agents.orchestrator import server as srv
except Exception:  # pragma: no cover — depends on the host's packages
    srv = None


@pytest.fixture
def client():
    if srv is None:
        pytest.skip("flask is not installed on this host")
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as c:
        yield c


def test_a_failed_run_answers_200_with_its_evidence_not_404(client, monkeypatch) -> None:
    """404 said "there is no such plan". There is one; it failed. Different answer.

    The reported message came from this path: the viewer showed "Could not load"
    for the very run the operator opened it to read.
    """
    seen: dict = {}

    def fake(url, name, args, timeout=600.0):
        seen.update(args)
        return {"success": False, "status": "failed", "tree_id": args["tree_id"],
                "error": "plan is a FAILED run and is not executable: ExpiredToken",
                "error_kind": "credentials", "num_planned": 13, "resumable": True,
                "failures": [{"operation_id": "W14", "error": "ExpiredToken"}],
                "note": "resume_planning(tree_id) to continue the run."}

    monkeypatch.setattr(srv, "_mcp_call", fake)
    resp = client.get("/a2a/plans/bt_bad")
    assert resp.status_code == 200
    body = resp.get_json()
    assert seen["include_failed"] is True
    assert body["success"] is True          # the READ succeeded
    assert body["status"] == "failed"
    assert body["executable"] is False      # …and it is not runnable
    assert body["behaviour_tree"] is None
    assert body["resumable"] is True
    assert "ExpiredToken" in body["error"]
    assert body["failures"][0]["operation_id"] == "W14"


def test_a_plan_that_does_not_exist_is_still_a_404(client, monkeypatch) -> None:
    """Opening failed runs must not turn a typo into a blank tree."""
    monkeypatch.setattr(srv, "_mcp_call",
                        lambda url, name, args, timeout=600.0: {
                            "success": False,
                            "error": "no calculated plan found for cell 'c' (bt_nope)"})
    assert client.get("/a2a/plans/bt_nope").status_code == 404


def test_a_good_plan_is_still_marked_executable(client, monkeypatch) -> None:
    monkeypatch.setattr(srv, "_mcp_call",
                        lambda url, name, args, timeout=600.0: {
                            "success": True, "tree_id": "bt_ok", "num_operations": 2,
                            "status": "ok", "executable": True,
                            "behaviour_tree": {"tree_id": "bt_ok", "root": {},
                                               "blackboard": {"k": [1, 2, 3]}}})
    body = client.get("/a2a/plans/bt_ok").get_json()
    assert body["status"] == "ok" and body["executable"] is True
    assert body["trajectory_sizes"] == {"k": 3}
