"""The Operator tab's task wizard: questions in, a task FILE out, nothing run.

"add next to the template the button wizard, that open a new window and drive the
user to create the task via set of question to fill the template and write the
file after that in the task".

Three properties are worth holding, and each of them has already been got wrong
somewhere in this panel:

1. **The wizard writes a program the planner can actually read.** The old
   ``Template`` button emitted ``step_number`` / ``operation_type`` /
   ``parameters.points`` — a shape no tool in this repo reads. A wizard that
   walked the operator through fifteen welds and then produced an unplannable
   file would be worse than no wizard, so the built JSON is asserted against the
   real schema (``skills/orchestrator/run-welding-program/references/
   task-json-format.md``: ``id``, ``type``, ``position``, ``approach_vector``).

2. **Writing the file commands nothing.** ``Submit & Run`` saves *and* executes;
   the wizard's endpoint is the file. So it posts to ``/api/a2a/task/save`` and
   to nothing else — no ``/task/submit``, no ``/api/command``, no ``/api/replay``.
   That is checked by watching every fetch it makes, not by reading the source.

3. **The wizard is its own page.** The panel is one inline ``<script>``, where one
   uncaught error blanks the whole UI (see ``test_dashboard_script_loads.py``).
   The wizard therefore lives in ``WIZARD_TEMPLATE`` behind ``GET /wizard``, and
   is booted here in node under the same fake DOM, so its own script is held to
   the same "reaches the end with no uncaught error" bar.

No Python import of the dashboard: it builds a ROS node at import time.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from tests.test_dashboard_script_loads import (NODE, _dom_harness, _render,
                                               _script, _served_ids)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "dashboard" / "web_control_panel.py"


def _literal(name: str) -> str:
    """A module-level string constant out of the panel source, without importing."""
    tree = ast.parse(PANEL.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == name for t in node.targets):
            assert isinstance(node.value, ast.Constant), \
                f"{name} stopped being a plain string literal"
            return node.value.value
    pytest.fail(f"{name} not found in dashboard/web_control_panel.py")


def _wizard_html(rids=("robot1", "robot2")) -> str:
    from jinja2 import Template
    return Template(_literal("WIZARD_TEMPLATE")).render(
        cell_id="some_cell",
        robot_options="".join(f'<option value="{r}">{r}</option>' for r in rids),
    )


PANEL_HTML = _render(["robot1", "robot2"])
PANEL_JS = _script(PANEL_HTML)
WIZARD_HTML = _wizard_html()
WIZARD_JS = _script(WIZARD_HTML)


def _fn(script: str, name: str) -> str:
    """One JS function body by brace matching, for asserting what it does NOT do."""
    start = script.index(f"function {name}(")
    depth, i = 0, script.index("{", start)
    for j in range(i, len(script)):
        depth += (script[j] == "{") - (script[j] == "}")
        if depth == 0:
            return script[start:j + 1]
    pytest.fail(f"function {name} is not brace-balanced")


# ── the button, in the card the operator was told to look in ─────────────────

class TestTheButtonIsWhereItWasAsked:

    def test_the_submit_task_card_offers_template_wizard_and_submit(self):
        """"next to the template" — same button row, same card, in that order."""
        card = re.search(r"Submit Task JSON.*?</div>\s*</div>", PANEL_HTML, re.S)
        assert card, "the Submit Task JSON card is gone"
        row = card.group(0)
        order = [m for m in re.findall(r"on[cC]lick=\"(op\w+)\(\)\"", row)]
        assert order == ["opLoadTemplate", "opTaskWizard", "opSubmitTask"], (
            f"the wizard must sit between Template and Submit & Run: {order}")
        assert "Wizard" in row, "the button does not say what it is"

    def test_the_button_opens_the_wizard_in_a_new_window(self):
        """A new window, not a tab-swap in the panel: the operator keeps the cell
        view on screen while authoring."""
        body = _fn(PANEL_JS, "opTaskWizard")
        assert "window.open" in body, \
            "opTaskWizard does not open a window — it was asked to"
        assert "'/wizard'" in body, f"it opens something other than /wizard: {body}"
        # A blocked pop-up must be reported, or the button looks broken.
        assert "blocked" in body.lower(), \
            "a pop-up blocker would make this button silently do nothing"

    def test_the_template_button_now_emits_the_schema_the_planner_reads(self):
        """The wizard fills a template; a template that cannot be planned is a
        trap. The legacy shape (`step_number`/`parameters.points`) is read by
        nothing in this repo."""
        body = _fn(PANEL_JS, "opLoadTemplate")
        for key in ('"id"', '"type"', '"position"', '"approach_vector"'):
            assert key in body, f"the template has no {key}: {body[:400]}"
        for legacy in ("step_number", "operation_type", "parameters"):
            assert legacy not in body, (
                f"the template still emits the legacy {legacy!r} shape, which no "
                f"tool in this repo reads")


# ── the wizard page itself ───────────────────────────────────────────────────

class TestTheWizardPageIsWellFormed:

    def test_it_is_a_separate_document_behind_its_own_route(self):
        assert '@app.route("/wizard")' in PANEL.read_text(), \
            "nothing serves the wizard page"
        assert "<!DOCTYPE html>" in WIZARD_HTML and "</html>" in WIZARD_HTML

    def test_every_element_it_looks_up_is_one_it_serves(self):
        """The failure mode that blanked this panel twice: getElementById on an id
        the markup lost, then a property access on null."""
        served = _served_ids(WIZARD_HTML)
        looked_up = set(re.findall(
            r"""(?:getElementById|\$)\(\s*["']([^"']+)["']\s*\)""", WIZARD_JS))
        assert looked_up, "no element lookups found — did the wizard script move?"
        missing = sorted(i for i in looked_up if i not in served)
        assert not missing, (
            f"the wizard looks up {missing}, which its page never renders")

    def test_the_step_panes_and_chips_it_toggles_all_exist(self):
        """``wzShow`` walks 1..LAST building ids by concatenation; a pane it can
        reach but the page does not serve is a null it then styles."""
        served = _served_ids(WIZARD_HTML)
        last = int(re.search(r"const LAST\s*=\s*(\d+)", WIZARD_JS).group(1))
        for i in range(1, last + 1):
            assert f"wz-p{i}" in served, f"step pane wz-p{i} is not served"
            assert f"wz-s{i}" in served, f"step chip wz-s{i} is not served"

    def test_it_asks_for_every_field_the_task_schema_needs(self):
        """A question per required field, in the operator's language."""
        for probe in ("Program name", "Position", "approach", "Point id",
                      "Sequence group"):
            assert probe.lower() in WIZARD_HTML.lower(), \
                f"the wizard never asks about {probe!r}"
        # The legal operation types, not a free-text box that fails at plan time.
        for op_type in ("weld", "rivet", "glue", "laser_cut", "mark", "move"):
            assert f'value="{op_type}"' in WIZARD_HTML, \
                f"operation type {op_type!r} cannot be chosen"

    def test_the_defaults_the_markup_offers(self):
        """What a browser applies before the operator types anything.

        These are the values ``WIZARD_HARNESS`` seeds into the fake DOM (which
        does not parse markup), so pin them here: a changed default that the
        harness did not follow would make every executed test below exercise a
        form no operator ever sees.
        """
        # The first <option> of a <select> is the selected one.
        assert re.search(r'id="wz-op-type">\s*<option value="weld"', WIZARD_HTML), \
            "the operation type no longer defaults to weld"
        assert re.search(r'id="wz-approach"[^>]*>\s*<option value="0,0,-1"',
                         WIZARD_HTML), "the approach no longer defaults to straight down"
        assert re.search(r'id="wz-backend">\s*<option value=""', WIZARD_HTML), \
            "the planner no longer defaults to the cell default"
        assert re.search(r'id="wz-robot"[^>]*>\s*<option value=""', WIZARD_HTML), \
            "the robot question no longer defaults to 'let the planner allocate'"
        assert 'id="wz-group" value="structural"' in WIZARD_HTML
        assert 'id="wz-prio" value="1"' in WIZARD_HTML
        assert ('id="wz-ax" value="0"' in WIZARD_HTML
                and 'id="wz-az" value="-1"' in WIZARD_HTML)

    def test_it_says_that_writing_the_file_runs_nothing(self):
        """The operator must not read "Written" as "the cell is welding"."""
        assert "no robot" in WIZARD_HTML.lower() or "nothing is planned" in \
            WIZARD_HTML.lower(), "the page never says it commands no motion"

    def test_the_source_carries_no_html_it_would_inject_unescaped(self):
        """Point ids reach the summary table through innerHTML; they are operator
        input, so they go through esc()."""
        body = _fn(WIZARD_JS, "wzRenderPoints")
        assert "innerHTML" in body, "the point list is rendered some other way now"
        assert "esc(o.id)" in body, \
            "the point id is interpolated into innerHTML unescaped"


# ── executed ─────────────────────────────────────────────────────────────────

# The wizard's own fake DOM: the panel's harness, plus what a form page needs
# (checkable inputs, window.opener, window.close) and a fetch recorder.
#
# MARKUP_DEFAULTS is what a browser would have applied from the markup — the
# selected <option> of a <select>, the value="" of an <input>. The fake DOM does
# not parse the HTML, so a test that skipped this would exercise a form where
# every field is the empty string, which no operator ever sees. The values are
# pinned against the markup by
# ``TestTheWizardPageIsWellFormed.test_the_defaults_the_markup_offers``, so a
# changed default cannot quietly diverge from what is seeded here.
WIZARD_HARNESS = """
const MARKUP_DEFAULTS = {'wz-op-type': 'weld', 'wz-approach': '0,0,-1',
                         'wz-backend': '', 'wz-robot': '', 'wz-group': 'structural',
                         'wz-prio': '1', 'wz-ax': '0', 'wz-ay': '0', 'wz-az': '-1'};
Object.entries(MARKUP_DEFAULTS).forEach(([id, v]) => {
  document.getElementById(id).value = v; });
const CALLS = [];
const _fetch0 = globalThis.fetch;
globalThis.fetch = (url, opt) => { CALLS.push({url: String(url), opt: opt || null});
                                   return _fetch0(url, opt); };
const OPENER_CALLS = [];
globalThis.window.opener = { opWizardSaved: (...a) => OPENER_CALLS.push(a) };
globalThis.window.close = () => {};
function set(id, v) { document.getElementById(id).value = String(v); }
function check(id, v) { document.getElementById(id).checked = !!v; }
function addPoint(id, x, y, z, extra) {
  set('wz-op-id', id); set('wz-x', x); set('wz-y', y); set('wz-z', z);
  if (extra) Object.entries(extra).forEach(([k, v]) => set(k, v));
  wzAddPoint();
}
"""


def _run(tmp_path, probe: str):
    js = tmp_path / "wizard.js"
    js.write_text(_dom_harness(WIZARD_HTML) + WIZARD_JS + WIZARD_HARNESS + probe)
    return subprocess.run([NODE, str(js)], capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheWizardRunsInAJsEngine:

    def test_it_parses(self, tmp_path):
        js = tmp_path / "wizard.js"
        js.write_text(WIZARD_JS)
        done = subprocess.run([NODE, "--check", str(js)],
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr[:1500]

    def test_it_reaches_the_end_with_no_uncaught_error(self, tmp_path):
        """Boot it as a browser would. It ends by calling wzShow(1)."""
        done = _run(tmp_path, "console.log('step=' + STEP);")
        assert done.returncode == 0, \
            f"the wizard script threw at load:\n{done.stderr[:1500]}"
        assert "step=1" in done.stdout, done.stdout


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheQuestionsFillTheTemplate:
    """The answers must become the task program, in the planner's own schema."""

    def test_answering_the_questions_builds_a_plannable_program(self, tmp_path):
        probe = textwrap.dedent("""
            set('wz-program', 'door_frame_welding');
            set('wz-desc', 'two structural welds');
            wzNext();                                   // -> planner
            set('wz-backend', 'ompl');
            wzNext();                                   // -> points
            addPoint('W01', -1.082, -1.489, 1.60);
            addPoint('W02', -0.982, -1.239, 1.10, {'wz-prio': 2});
            wzNext();                                   // -> review
            console.log('step=' + STEP);
            console.log('json=' + JSON.stringify($('wz-json').value));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "step=4" in done.stdout, f"the wizard did not advance: {done.stdout}"
        built = json.loads(json.loads(
            re.search(r"json=(\".*\")", done.stdout).group(1)))
        assert built["program"] == "door_frame_welding"
        assert built["cell_id"] == "some_cell", \
            "the program does not name the cell it runs on"
        assert built["description"] == "two structural welds"
        assert built["trajectory_backend"] == "ompl"
        assert len(built["operations"]) == 2, built
        first = built["operations"][0]
        # The real schema, field for field.
        assert first == {"id": "W01", "type": "weld",
                         "position": [-1.082, -1.489, 1.60],
                         "approach_vector": [0, 0, -1],
                         "sequence_group": "structural", "priority": 1}, first
        assert built["operations"][1]["priority"] == 2
        # Nothing from the legacy shape leaks in.
        for legacy in ("step_number", "operation_type", "parameters", "points"):
            assert legacy not in first, f"{legacy!r} is not part of the task schema"

    def test_a_point_can_be_pinned_to_one_robot_but_is_not_by_default(self, tmp_path):
        """Allocation is the planner's decision; pinning is the exception."""
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            addPoint('W01', 0, 0, 1);
            addPoint('HANDLING_01', 0.1, 0.2, 1.3, {'wz-robot': 'robot2'});
            wzNext();
            const ops = JSON.parse($('wz-json').value).operations;
            console.log('pin0=' + (ops[0].assigned_robot || 'none')
                        + ' pin1=' + (ops[1].assigned_robot || 'none'));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "pin0=none" in done.stdout, (
            "an unanswered robot question must leave the point free for the "
            f"planner: {done.stdout}")
        assert "pin1=robot2" in done.stdout, done.stdout

    def test_a_custom_approach_direction_is_asked_for_and_used(self, tmp_path):
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            set('wz-approach', 'custom'); wzApproachChanged();
            console.log('shown=' + $('wz-custom-av').style.display);
            set('wz-ax', 0.5); set('wz-ay', 0); set('wz-az', -0.866);
            addPoint('W01', 0, 0, 1);
            wzNext();
            console.log('av=' + JSON.stringify(
              JSON.parse($('wz-json').value).operations[0].approach_vector));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "shown=block" in done.stdout, \
            f"choosing a custom direction never revealed its boxes: {done.stdout}"
        assert "av=[0.5,0,-0.866]" in done.stdout, done.stdout

    def test_the_next_point_id_is_suggested_from_the_last_one(self, tmp_path):
        """W01 -> W02. Fifteen welds typed by hand is where a typo'd duplicate id
        comes from."""
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            console.log('first=' + $('wz-op-id').value);
            addPoint('W09', 0, 0, 1);
            console.log('next=' + $('wz-op-id').value);
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "first=W01" in done.stdout, done.stdout
        assert "next=W10" in done.stdout, \
            f"the id series was not continued: {done.stdout}"

    def test_a_point_can_be_removed_again(self, tmp_path):
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            addPoint('W01', 0, 0, 1); addPoint('W02', 0, 0.1, 1);
            wzDelPoint(0);
            wzNext();
            const ops = JSON.parse($('wz-json').value).operations;
            console.log('ids=' + ops.map(o => o.id).join(',') + ' count=' + $('wz-count').textContent);
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "ids=W02" in done.stdout and "count=1" in done.stdout, done.stdout


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheWizardRefusesAnswersThatWouldNotPlan:
    """Every one of these is a failure the operator would otherwise meet minutes
    later, as a planning error against a file already on disk."""

    def test_it_will_not_leave_the_first_question_without_a_program_name(self, tmp_path):
        done = _run(tmp_path, "wzNext(); console.log('step=' + STEP "
                              "+ ' say=' + $('wz-status').textContent);")
        assert done.returncode == 0, done.stderr[:1500]
        assert "step=1" in done.stdout, \
            f"an unnamed program advanced: {done.stdout}"
        assert "name" in done.stdout.lower(), done.stdout

    def test_it_rejects_a_program_name_that_is_not_a_file_name(self, tmp_path):
        """The same rule the orchestrator enforces — it becomes a path segment."""
        probe = textwrap.dedent("""
            const out = [];
            for (const bad of ['../etc/passwd', 'a b', 'p/x', 'a;b']) {
              set('wz-program', bad); wzNext(); out.push(bad + '=>' + STEP);
            }
            console.log('tries=' + out.join(' '));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, done.stderr[:1500]
        assert "=>2" not in done.stdout, \
            f"an unusable program name was accepted: {done.stdout}"

    def test_it_will_not_review_a_program_with_no_operations(self, tmp_path):
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            wzNext();
            console.log('step=' + STEP + ' say=' + $('wz-status').textContent);
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, done.stderr[:1500]
        assert "step=3" in done.stdout, \
            f"an empty program reached the review step: {done.stdout}"
        assert "at least one point" in done.stdout, done.stdout

    def test_a_point_with_a_non_numeric_coordinate_is_refused(self, tmp_path):
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            addPoint('W01', '', 0, 1);
            console.log('count=' + $('wz-count').textContent
                        + ' say=' + $('wz-status').textContent);
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, done.stderr[:1500]
        assert "count=0" in done.stdout, \
            f"a point with no x was added: {done.stdout}"
        assert "numbers" in done.stdout, done.stdout

    def test_a_duplicate_point_id_is_refused(self, tmp_path):
        """Operation ids must be unique — the planner keys allocations by them."""
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            addPoint('W01', 0, 0, 1); addPoint('W01', 0, 0.2, 1);
            console.log('count=' + $('wz-count').textContent
                        + ' say=' + $('wz-status').textContent);
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, done.stderr[:1500]
        assert "count=1" in done.stdout, \
            f"a duplicate operation id was accepted: {done.stdout}"
        assert "already a point" in done.stdout, done.stdout

    def test_hand_edited_json_that_does_not_parse_is_not_posted(self, tmp_path):
        """The review box is editable, so it is also a way to break the file."""
        probe = textwrap.dedent("""
            set('wz-program', 'p'); wzNext(); wzNext();
            addPoint('W01', 0, 0, 1); wzNext();
            set('wz-json', '{not json');
            wzWrite();
            console.log('posts=' + CALLS.length + ' say=' + $('wz-status').textContent);
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, done.stderr[:1500]
        assert "posts=0" in done.stdout, \
            f"invalid JSON was posted to the server: {done.stdout}"
        assert "not valid JSON" in done.stdout, done.stdout


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestWritingTheFileCommandsNothing:
    """The wizard's endpoint is a file on disk. Not a run."""

    AUTHORED = """
        ROUTES['/api/a2a/task/save'] = { success: true, program: 'door_frame_welding',
          saved: '/cells/some_cell/tasks/door_frame_welding.json',
          num_operations: 1, commanded: false, ran: false };
        set('wz-program', 'door_frame_welding'); wzNext(); wzNext();
        addPoint('W01', -1.082, -1.489, 1.60); wzNext();
    """

    def test_it_posts_the_program_to_the_save_endpoint(self, tmp_path):
        probe = textwrap.dedent(self.AUTHORED + """
            wzWrite();
            const post = CALLS.filter(c => c.url === '/api/a2a/task/save').pop();
            console.log('method=' + (post && post.opt && post.opt.method));
            console.log('body=' + (post && post.opt && post.opt.body));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "method=POST" in done.stdout, \
            f"the task file was never posted: {done.stdout}"
        body = json.loads(re.search(r"body=(\{.*\})", done.stdout).group(1))
        assert body["program"] == "door_frame_welding"
        assert body["cell_id"] == "some_cell"
        assert body["operations"][0]["id"] == "W01"
        assert "overwrite" not in body, \
            "an unticked overwrite box must not send overwrite: true"

    def test_it_never_touches_an_endpoint_that_moves_a_robot(self, tmp_path):
        """Watched, not read from the source: any fetch to a run/replay/command
        route would mean the wizard executes what it authored."""
        probe = textwrap.dedent(self.AUTHORED + """
            wzWrite();
            setImmediate(() => setImmediate(() =>
              console.log('urls=' + JSON.stringify(CALLS.map(c => c.url)))));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        urls = json.loads(re.search(r"urls=(\[.*\])", done.stdout).group(1))
        assert urls == ["/api/a2a/task/save"], \
            f"the wizard called more than the save endpoint: {urls}"
        for forbidden in ("/api/a2a/task/submit", "/api/a2a/run", "/api/command",
                          "/api/a2a/replay", "/api/replay", "/api/a2a/move"):
            assert forbidden not in urls, (
                f"the wizard reached {forbidden} — writing a task file must not "
                f"start anything")

    def test_a_written_file_is_reported_as_written_and_not_as_run(self, tmp_path):
        probe = textwrap.dedent(self.AUTHORED + """
            wzWrite();
            setImmediate(() => setImmediate(() => {
              console.log('step=' + STEP);
              console.log('result=' + JSON.stringify(HTML['wz-result'] || ''));
            }));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "step=5" in done.stdout, \
            f"a successful write did not reach the final step: {done.stdout}"
        out = done.stdout
        assert "door_frame_welding.json" in out, \
            f"the operator is not told where the file went: {out}"
        assert "no robot has moved" in out, \
            f"'Written' must not read as 'the cell is welding': {out}"

    def test_the_panel_that_opened_it_is_handed_the_result(self, tmp_path):
        """So Submit & Run is one click away and the Planning list refreshes."""
        probe = textwrap.dedent(self.AUTHORED + """
            wzWrite();
            setImmediate(() => setImmediate(() =>
              console.log('opener=' + JSON.stringify(OPENER_CALLS))));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        calls = json.loads(re.search(r"opener=(\[.*\])", done.stdout).group(1))
        assert len(calls) == 1, f"the opener was not notified: {calls}"
        program, path, text = calls[0]
        assert program == "door_frame_welding"
        assert path.endswith("door_frame_welding.json")
        assert json.loads(text)["operations"][0]["id"] == "W01", \
            "the panel was handed something other than what was written"

    def test_opening_the_wizard_directly_still_works(self, tmp_path):
        """No opener (someone bookmarked /wizard) must not throw at the last step."""
        probe = textwrap.dedent("""
            globalThis.window.opener = null;
        """ + self.AUTHORED + """
            wzWrite();
            setImmediate(() => setImmediate(() => console.log('step=' + STEP)));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, \
            f"a wizard with no opener threw:\n{done.stdout}\n{done.stderr[:1500]}"
        assert "step=5" in done.stdout, done.stdout

    def test_a_refused_write_says_why_and_how_to_get_past_it(self, tmp_path):
        """409: that program exists. The wizard has a box for exactly this."""
        probe = textwrap.dedent("""
            ROUTES['/api/a2a/task/save'] = { success: false,
              error: "program 'p' already exists in cell 'some_cell'." };
            set('wz-program', 'p'); wzNext(); wzNext();
            addPoint('W01', 0, 0, 1); wzNext();
            wzWrite();
            setImmediate(() => setImmediate(() => {
              console.log('step=' + STEP + ' say=' + $('wz-status').textContent);
            }));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "step=4" in done.stdout, \
            f"a refused write moved on as if it had succeeded: {done.stdout}"
        assert "already exists" in done.stdout and "Overwrite" in done.stdout, \
            f"the operator is not told how to proceed: {done.stdout}"

    def test_ticking_overwrite_sends_it(self, tmp_path):
        probe = textwrap.dedent(self.AUTHORED + """
            check('wz-overwrite', true);
            wzWrite();
            const post = CALLS.filter(c => c.url === '/api/a2a/task/save').pop();
            console.log('body=' + (post && post.opt && post.opt.body));
        """)
        done = _run(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        body = json.loads(re.search(r"body=(\{.*\})", done.stdout).group(1))
        assert body.get("overwrite") is True, body


# ── the panel side of the handshake ──────────────────────────────────────────

@pytest.mark.skipif(not NODE, reason="node not installed")
class TestThePanelAcceptsWhatTheWizardWrote:

    def test_opWizardSaved_fills_the_textarea_and_says_nothing_ran(self, tmp_path):
        from tests.test_dashboard_script_loads import _run_probe
        probe = textwrap.dedent("""
            opWizardSaved('door_frame_welding',
                          '/cells/some_cell/tasks/door_frame_welding.json',
                          '{"program":"door_frame_welding"}');
            console.log('ta=' + REG['op-task-json'].value);
            console.log('st=' + REG['op-task-status'].textContent);
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert 'ta={"program":"door_frame_welding"}' in done.stdout, \
            f"the panel textarea was not filled: {done.stdout}"
        assert "door_frame_welding.json" in done.stdout, done.stdout
        assert "nothing has run yet" in done.stdout, (
            "the panel must not imply the program was executed: " + done.stdout)
