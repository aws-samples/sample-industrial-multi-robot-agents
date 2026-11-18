"""A planning run in flight has to be visible from across the cell.

The reported defect: "make visible in the dashboard if the planning is running".
Planning takes MINUTES, dispatches asynchronously and moves nothing — and the
panel looked identical whether a run was in flight or not:

* ``updateUI`` wrote ``data.status`` into ``class="sdot dot-planning"`` and **no CSS
  rule matched that class**, so the one long-lived state was drawn in the same grey
  as idle.
* The button that starts the run stayed exactly as green after the click as before
  it, so the only feedback was a line of 10px text under the controls. It got
  pressed twice.
* The Agents tab — the pane that reports what the agents are doing — said nothing
  about the cell's overall state, so a reader on another tab had to switch just to
  find out whether anything was happening at all.

What is pinned here is therefore the *visibility*, not the plumbing: a status of
``planning`` must change the button's colour AND its label, blink a message in the
header bar, and show the same overall status on the Agents tab. Plus the two ways
this could lie — a stale blink after the run ends, and two indicators disagreeing.

The JS-engine half reuses the harness in :mod:`test_dashboard_script_loads` (one
DOM stub for the suite, so a stub artefact cannot pass here and fail there).
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

from tests.test_dashboard_script_loads import NODE, _run_probe

ROOT = Path(__file__).resolve().parents[1]
PANEL = (ROOT / "dashboard" / "web_control_panel.py").read_text()

# The panel's own state machine (dashboard status field) — every value it can hold.
PANEL_STATUSES = ("idle", "planning", "homing", "estopped")


def _fn(name: str) -> str:
    """One JS function's body, by brace matching."""
    start = PANEL.index(f"function {name}(")
    depth = 0
    for j in range(PANEL.index("{", start), len(PANEL)):
        if PANEL[j] == "{":
            depth += 1
        elif PANEL[j] == "}":
            depth -= 1
            if depth == 0:
                return PANEL[start:j + 1]
    pytest.fail(f"{name} has unbalanced braces")


def _out(stdout: str) -> dict:
    """The probe's ``OUT={...}`` line."""
    line = re.search(r"^OUT=(\{.*\})$", stdout, re.M)
    assert line, f"the probe printed no OUT= line:\n{stdout}"
    return json.loads(line.group(1))


def _state(status: str) -> dict:
    """The /api/state shape for a given status, with every field updateUI reads."""
    return {"status": status, "velocity_scale": 1.0, "loop_active": False,
            "agent_activity": {}, "collision_alerts": [], "current_plan_info": {},
            "recordings": [], "robot_positions": {}, "robot_states": {},
            "planning_log": []}


# ── the styling exists at all ────────────────────────────────────────────────

@pytest.mark.parametrize("status", PANEL_STATUSES)
def test_every_status_the_panel_can_hold_has_a_dot_colour(status) -> None:
    """The original bug, generalised: a state with no rule renders as idle grey."""
    assert re.search(rf"^\.dot-{status}\{{", PANEL, re.M), (
        f".dot-{status} has no CSS rule, so status '{status}' is drawn in the "
        f"same colour as an idle cell")


def test_the_blink_is_a_real_animation_and_costs_no_layout() -> None:
    """Opacity only: a blinking element must not shove its neighbours around."""
    frames = re.search(r"@keyframes blink-fade\{(.+?)\}\n", PANEL)
    assert frames, "the blink keyframes are gone; .blink would do nothing"
    assert re.search(r"^\.blink\{animation:blink-fade", PANEL, re.M)
    assert "opacity" in frames.group(1)
    for forbidden in ("width", "height", "margin", "padding", "display"):
        assert forbidden not in frames.group(1), forbidden


def test_the_running_planning_button_is_a_different_colour() -> None:
    """Not just a label: the request was "change the color of the planning button"."""
    idle = re.search(r"\.btn-ag\{background:(#[0-9a-fA-F]+)", PANEL)
    busy = re.search(r"\.btn-ag\.btn-busy\{background:(#[0-9a-fA-F]+)", PANEL)
    assert idle and busy, "the in-flight Planning button has no colour of its own"
    assert idle.group(1).lower() != busy.group(1).lower()


@pytest.mark.parametrize("element_id", [
    "status-msg", "btn-planning", "tab-agents-dot", "agents-overall",
])
def test_every_element_the_renderer_writes_is_served(element_id) -> None:
    assert f'id="{element_id}"' in PANEL


def test_the_blinking_message_lives_in_the_header_bar() -> None:
    """"blink a message in the bar" — the bar being the header, not a toast."""
    header = PANEL[PANEL.index('<div class="header">'):]
    header = header[:header.index("</div>")]
    assert 'id="status-msg"' in header


def test_the_overall_status_is_on_the_agents_tab_too() -> None:
    """Both halves of "the same for the agent tab": the tab button and the pane."""
    tab = PANEL[PANEL.index('id="tab-agents"'):]
    assert 'id="tab-agents-dot"' in tab[:tab.index("</button>")]
    pane = PANEL[PANEL.index('id="pane-agents"'):]
    pane = pane[:pane.index('id="tip"')]
    assert 'id="agents-overall"' in pane


def test_one_renderer_owns_every_indicator() -> None:
    """Two writers is how two indicators come to disagree about the same run."""
    owned = ("status-dot", "status-text", "status-msg", "btn-planning",
             "tab-agents-dot", "agents-overall")
    pattern = r'\$\("(' + "|".join(owned) + r')"\)'
    assert sorted(re.findall(pattern, PANEL)) == \
        sorted(re.findall(pattern, _fn("renderRunState"))), \
        "something outside renderRunState() writes a status indicator"


def test_leaving_the_agents_tab_drops_what_only_that_poll_knew() -> None:
    """Otherwise the header blinks PLANNING for the rest of the session."""
    assert "_planRunning=false" in _fn("liveStop")


# ── the same thing, executed ─────────────────────────────────────────────────

_REPORT = """
    console.log("OUT=" + JSON.stringify({
      dot: REG["status-dot"].className,
      text: REG["status-text"].textContent,
      msg: REG["status-msg"].textContent,
      msgClass: REG["status-msg"].className,
      msgColor: REG["status-msg"].style.color,
      btn: REG["btn-planning"].className,
      label: REG["btn-planning"].textContent,
      title: REG["btn-planning"].title,
      tabDot: REG["tab-agents-dot"].className,
      overall: REG["agents-overall"].textContent,
      overallClass: REG["agents-overall"].className,
    }));
"""


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestAPlanningRunIsVisibleWithoutReadingText:
    def test_a_planning_status_lights_up_all_four_places(self, tmp_path) -> None:
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(_state("planning"))};
            updateUI();
            setImmediate(() => {{{_REPORT}}});
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        # The header: an amber dot that blinks, next to a message that blinks.
        assert out["dot"] == "sdot dot-planning blink", out
        assert out["text"] == "PLANNING"
        assert "PLANNING IN PROGRESS" in out["msg"], out
        assert "blink" in out["msgClass"], out
        # "nothing moves" is the part an operator acts on: a planning run is not
        # motion, and reading it as motion is the dangerous direction of wrong.
        assert "nothing moves" in out["msg"], out
        # The button: different class (so different colour) AND a changed label.
        assert "btn-busy" in out["btn"] and "blink" in out["btn"], out
        assert out["label"].endswith("…"), out
        assert "SECOND run" in out["title"], out
        # The Agents tab, for a reader sitting on another one.
        assert out["tabDot"] == "sdot dot-planning blink", out
        assert "PLANNING" in out["overall"] and "blink" in out["overallClass"], out

    def test_an_idle_cell_is_quiet_everywhere(self, tmp_path) -> None:
        """A panel that always blinks says nothing. Asserted AFTER a run, so a
        one-way transition (the easy bug) fails here."""
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(_state("planning"))};
            updateUI();
            setImmediate(() => {{
              ROUTES["/api/state"] = {json.dumps(_state("idle"))};
              updateUI();
              setImmediate(() => {{{_REPORT}}});
            }});
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert out["dot"] == "sdot dot-idle", out
        assert out["msg"] == "" and "blink" not in out["msgClass"], out
        assert out["btn"] == "btn btn-ag", out
        assert not out["label"].endswith("…"), out
        assert "blink" not in out["tabDot"] and "blink" not in out["overallClass"], out
        assert "IDLE" in out["overall"], out

    def test_a_latched_estop_is_the_loudest_thing_on_the_bar(self, tmp_path) -> None:
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(_state("estopped"))};
            updateUI();
            setImmediate(() => {{{_REPORT}}});
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert out["dot"] == "sdot dot-estopped blink", out
        assert "E-STOP" in out["msg"] and "Reset" in out["msg"], out
        # Red, not the amber of a planning run: the two must not read alike.
        assert out["msgColor"] == "#f85149", out
        # An e-stop is not a planning run, so the Planning button stays itself.
        assert out["btn"] == "btn btn-ag", out

    def test_pressing_planning_reports_before_the_next_poll(self, tmp_path) -> None:
        """The 1.5 s gap between the click and the colour change is exactly when
        the button gets pressed a second time."""
        probe = textwrap.dedent("""
            ROUTES["/api/command"] = { ok: true, command: "planning",
                                       program: "car_door_spot_welding", dispatched: true };
            document.getElementById("plan-task").value = "car_door_spot_welding";
            cmd("planning");
            setImmediate(() => setImmediate(() => {
              console.log("OUT=" + JSON.stringify({
                btn: REG["btn-planning"].className,
                label: REG["btn-planning"].textContent,
                say: REG["cmd-status"].textContent }));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert "btn-busy" in out["btn"], (
            f"the button did not change on dispatch, only on the next poll: {out}")
        assert out["label"].endswith("…"), out
        assert out["say"].startswith("✓ planning dispatched"), out

    def test_a_refused_planning_command_leaves_the_button_alone(self, tmp_path) -> None:
        """No task selected is a refusal, not a run. Showing it as one would blink
        PLANNING at nothing until the next poll corrected it."""
        probe = textwrap.dedent("""
            cmd("planning");
            setImmediate(() => setImmediate(() => {
              console.log("OUT=" + JSON.stringify({
                btn: REG["btn-planning"].className,
                say: REG["cmd-status"].textContent }));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert out["btn"] == "btn btn-ag", out
        assert "pick a task" in out["say"], out


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestARunThisPanelDidNotStartIsStillVisible:
    """``/api/state`` only knows about runs launched from this panel's own button.

    A plan submitted from the chat, from an MCP client or by ``make task`` leaves it
    at ``idle`` — and that is the run an operator walking up to the panel most needs
    to know about. The planner's own event feed answers ``running``, so it may ADD
    planning to an idle status; it must never take one away.
    """

    def _events(self, running: bool) -> str:
        return json.dumps({"success": True, "trace_id": "t1", "events": [],
                           "next_seq": 0, "running": running})

    def test_the_feeds_verdict_reaches_the_header(self, tmp_path) -> None:
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(_state("idle"))};
            ROUTES["/api/a2a/plan_events"] = {self._events(True)};
            updateUI();
            setImmediate(() => {{
              livePoll();
              setImmediate(() => {{{_REPORT}}});
            }});
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert out["dot"] == "sdot dot-planning blink", (
            f"a live run the panel did not start is invisible: {out}")
        assert "btn-busy" in out["btn"], out

    def test_a_finished_run_goes_quiet_again(self, tmp_path) -> None:
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(_state("idle"))};
            ROUTES["/api/a2a/plan_events"] = {self._events(True)};
            updateUI();
            setImmediate(() => {{
              livePoll();
              setImmediate(() => {{
                ROUTES["/api/a2a/plan_events"] = {self._events(False)};
                livePoll();
                setImmediate(() => {{{_REPORT}}});
              }});
            }});
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert out["dot"] == "sdot dot-idle", out
        assert out["btn"] == "btn btn-ag", out

    def test_leaving_the_tab_hands_the_status_back_to_api_state(self, tmp_path) -> None:
        """The poll stops with the tab, so its answer must stop counting too."""
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(_state("idle"))};
            ROUTES["/api/a2a/plan_events"] = {self._events(True)};
            ROUTES["/api/a2a/checkpoints"] = {{ checkpoints: [] }};
            updateUI();
            setImmediate(() => {{
              livePoll();
              setImmediate(() => {{
                liveStop();
                {_REPORT}
              }});
            }});
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert out["dot"] == "sdot dot-idle", (
            f"the blink outlived the poll that justified it: {out}")
        assert out["msg"] == "", out

    def test_the_panels_own_status_wins_over_the_feed(self, tmp_path) -> None:
        """A homing cell is MOVING. A stale 'running' from the feed must not
        relabel that as a planning run, which moves nothing."""
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(_state("homing"))};
            ROUTES["/api/a2a/plan_events"] = {self._events(True)};
            updateUI();
            setImmediate(() => {{
              livePoll();
              setImmediate(() => {{{_REPORT}}});
            }});
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = _out(done.stdout)
        assert out["dot"] == "sdot dot-homing blink", out
        assert "arms are moving" in out["msg"], out
