"""The dashboard is one ``<script>``, so one undefined name blanks the whole UI.

``dashboard/web_control_panel.py`` serves its entire front end as a single inline
script inside ``HTML_TEMPLATE``. Everything lives there: ``buildDiagram()`` for
the agent graph, the ``setInterval(updateUI, 1500)`` that polls ``/api/state``
for the log panel, the chat wiring, the column resizers. A browser abandons a
script block at the first uncaught error, so a mistake in ONE top-level
statement takes the diagram, the logs, the chat and the controls with it — and
the page still returns 200 with all the markup, so the server looks healthy.

That happened: a top-level ``Object.assign(NODE_POS, ...)`` merged the
server-injected robot nodes into a table nothing declared (the table is called
``LAYOUT``). ``ReferenceError: NODE_POS is not defined``, thrown before a single
function was reachable. Every endpoint the panel reads was fine — ``/api/topology``
returned all 8 agents — and nothing rendered.

No Python import of the dashboard here: it constructs a ROS node at import time,
which does not exist on the host. The template comes out of the source with
``ast`` and is rendered with Jinja, exactly as the route does it.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL = PROJECT_ROOT / "dashboard" / "web_control_panel.py"

def _html_template() -> str:
    """``HTML_TEMPLATE`` straight out of the source, without importing it."""
    tree = ast.parse(PANEL.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "HTML_TEMPLATE" for t in node.targets):
            assert isinstance(node.value, ast.Constant), \
                "HTML_TEMPLATE stopped being a plain string literal"
            return node.value.value
    pytest.fail("HTML_TEMPLATE not found in dashboard/web_control_panel.py")


def _robot_graph_js(rids: list[str]) -> str:
    """What ``robot_graph_js()`` injects, for an arbitrary robot count.

    Mirrors the server rather than importing it (see the module docstring). Only
    the shape matters here: three ``const`` declarations the template merges in.
    """
    n = max(len(rids), 1)
    positions, edges = {}, []
    for i, rid in enumerate(rids):
        node = f"robot-agent-{rid}"
        positions[node] = {"cx": 500 if n == 1 else int(200 + 600 * i / (n - 1)),
                           "cy": 495}
        edges += [["rai-orchestrator", node, "Dispatch task"],
                  [node, "rviz-web", "Joint states"]]
    return (f"const ROBOT_NODE_POS = {json.dumps(positions)};\n"
            f"const ROBOT_EDGES = {json.dumps(edges)};\n"
            f"const ROBOT_IDS = {json.dumps(rids)};\n")


def _render(rids: list[str]) -> str:
    from jinja2 import Template
    return Template(_html_template()).render(
        cell_id="some_cell",
        novnc_port="6080",
        robot_graph_js=_robot_graph_js(rids),
        robot_options="".join(f'<option value="{r}">{r}</option>' for r in rids),
    )


def _served_ids(html: str) -> set[str]:
    """Element ids the page actually contains."""
    return set(re.findall(r'id="([^"]+)"', html))


# Ids built at runtime by concatenation (``"actor-"+agent.id``); the literal
# prefix is what the source shows, and the element exists only once drawn.
DYNAMIC_ID_PREFIXES = ("actor-", "arrow-", "bt-", "pane-", "tab-")


def _script(html: str) -> str:
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert blocks, "the page serves no inline script"
    return "\n".join(blocks)


SCRIPT_2 = _script(_render(["robot1", "robot2"]))


# ── the bug class, checked without a JS engine ───────────────────────────────

class TestEveryTableTheScriptReadsIsDeclared:
    """Pure-Python guards, so these run even where node is not installed.

    Deliberately narrow — matched against specific constructs rather than every
    upper-case word in the file. A broad "is this identifier declared" scan reads
    names out of comments and regex literals (``/COLLISION.*ALERT/i``) and fails
    on prose; that is a linter's job, not this suite's, and a test that cries
    wolf gets deleted. What is pinned below is the exact wiring that broke.
    """

    @pytest.mark.parametrize("rids", [["robot1"], ["robot1", "robot2"],
                                      ["robot1", "robot2", "robot3"]])
    def test_every_top_level_object_assign_targets_a_declared_table(self, rids):
        """``Object.assign`` into an undeclared name is this bug, exactly."""
        script = _script(_render(rids))
        declared = set(re.findall(
            r"\b(?:const|let|var|function)\s+([A-Za-z_][A-Za-z0-9_]*)\b", script))
        targets = re.findall(r"Object\.assign\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", script)
        assert targets, "no Object.assign left — did the injection wiring change?"
        missing = sorted(t for t in set(targets) if t not in declared)
        assert not missing, (
            f"Object.assign merges into {missing}, which nothing declares; a "
            f"browser aborts the whole <script> there and the page renders blank")

    def test_the_injected_robot_tables_are_merged_into_the_real_one(self):
        """The specific line that broke: the merge target must be the read table."""
        merge = re.search(r"Object\.assign\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
                          r"\(typeof ROBOT_NODE_POS", SCRIPT_2)
        assert merge, "the per-robot node positions are no longer merged in"
        target = merge.group(1)
        assert re.search(rf"\bconst\s+{target}\s*=", SCRIPT_2), \
            f"Object.assign merges into {target}, which is never declared"
        assert re.search(rf"\b{target}\[\s*agent\.id\s*\]", SCRIPT_2), \
            f"{target} is not the table buildDiagram() reads to place a node"

    def test_a_robot_node_actually_lands_in_that_table(self):
        """Injection and merge agree on the key, so arms get drawn."""
        assert '"robot-agent-robot2": {"cx": 800, "cy": 495}' in SCRIPT_2


class TestEveryElementTheScriptLooksUpIsServed:
    """The second way this page went blank: a lookup of an id the markup lost.

    ``getElementById`` returns null, and the very next ``.value =`` throws. One of
    those (``#speed-slider``, left behind when the speed control was dropped from
    the markup) sat in ``updateUI`` ABOVE the log render, so every 1.5 s poll died
    before writing the log panel — with ``/api/state`` answering correctly the
    whole time. Same blank-panel symptom as the undeclared table, different cause,
    so it gets its own guard.

    Element ids are compared, not guessed: a lookup either names an id the page
    serves, or one of the documented runtime-built prefixes.
    """

    @pytest.mark.parametrize("rids", [["robot1"], ["robot1", "robot2"],
                                      ["robot1", "robot2", "robot3"]])
    def test_no_lookup_names_an_id_the_page_does_not_have(self, rids):
        html = _render(rids)
        served = _served_ids(html)
        # Both spellings: the direct DOM call and the $() helper that wraps it.
        looked_up = set(re.findall(
            r"""(?:getElementById|\$)\(\s*["']([^"']+)["']\s*\)""", _script(html)))
        assert looked_up, "no element lookups found — did the script move?"
        missing = sorted(i for i in looked_up
                         if i not in served
                         and not i.startswith(DYNAMIC_ID_PREFIXES))
        assert not missing, (
            f"the script looks up {missing}, which the page never renders; "
            f"getElementById returns null and the next property access throws, "
            f"taking everything after it in that handler with it")


# ── the same thing, executed ─────────────────────────────────────────────────

NODE = shutil.which("node")

# Any property access or call yields another proxy, so a gap in the DOM stub
# cannot masquerade as the error under test.
HARNESS = textwrap.dedent("""
    const mk = () => new Proxy(function(){}, {
      get: (t, p) => (p === 'then' ? undefined : mk()),
      apply: () => mk(), set: () => true, has: () => true });
    globalThis.document = mk();
    globalThis.window = mk();
    globalThis.navigator = mk();
    // Real strings: the noVNC iframe src is built from these, and a proxy here
    // fails with "Cannot convert object to primitive value".
    globalThis.location = { protocol: 'http:', hostname: 'localhost', href: 'http://localhost:8888/' };
    globalThis.fetch = () => ({ then: () => ({ then: () => ({ catch(){} }), catch(){} }), catch(){} });
    globalThis.setInterval = () => 0;
    globalThis.setTimeout = () => 0;
""")

# For the drawing test the proxy is not enough: buildDiagram does arithmetic on
# clientWidth/clientHeight, and a proxy there fails with "Cannot convert object
# to primitive value" — a stub artefact that would masquerade as a real defect.
# This DOM records what gets appended so the test can count the actors drawn.
DOM_HARNESS = textwrap.dedent("""
    const APPENDED = {}, HTML = {};
    function mkEl(id) {
      return {
        id, dataset: { cell: 'some_cell' }, style: {}, textContent: '', value: '',
        clientWidth: 600, clientHeight: 760, children: [], options: [],
        add(){}, src: '', href: '', attrs: {},
        classList: { add(){}, remove(){}, toggle(){}, contains: () => false },
        setAttribute(k, v) { this.attrs[k] = v; },
        removeAttribute(k) { delete this.attrs[k]; }, addEventListener(){},
        appendChild(c) { APPENDED[id] = (APPENDED[id] || 0) + 1; this.children.push(c); },
        removeChild(){}, remove(){}, insertAdjacentHTML(){},
        getBoundingClientRect: () => ({ width: 600, height: 760, left: 0, top: 0 }),
        querySelector: () => null, querySelectorAll: () => [],
        get innerHTML() { return HTML[id] || ''; },
        // Record it: the log panel is written by assigning innerHTML, so a test
        // that only counted appendChild would call an empty log panel healthy.
        set innerHTML(v) { HTML[id] = v; APPENDED[id] = 0; this.children = []; },
      };
    }
    const REG = {};
    // Only ids the page really serves resolve. A harness that invents an element
    // for any id asked of it is exactly what let the #speed-slider lookup — the
    // one that broke the log panel in a browser — pass this suite.
    // SERVED_IDS/DYNAMIC_PREFIXES are injected by the test from the rendered HTML.
    const isReal = id => SERVED_IDS.includes(id)
                      || DYNAMIC_PREFIXES.some(p => id.startsWith(p));
    globalThis.document = {
      getElementById: id => isReal(id) ? (REG[id] = REG[id] || mkEl(id)) : null,
      createElementNS: (ns, tag) => mkEl('el:' + tag),
      createElement: tag => mkEl('el:' + tag),
      querySelector: () => null, querySelectorAll: () => [],
      addEventListener(){}, body: mkEl('body'), documentElement: mkEl('html'),
    };
    globalThis.window = { addEventListener(){}, innerWidth: 1400, innerHeight: 900,
      getComputedStyle: () => ({ getPropertyValue: () => '600px' }) };
    globalThis.navigator = {};
    globalThis.location = { protocol: 'http:', hostname: 'localhost', href: 'http://localhost:8888/' };
    globalThis.Option = function(text, value) { return { text, value }; };
    // Real promises keyed by URL, so updateUI's /api/state chain actually runs.
    // A test overrides a route to feed it data; the default is the shape the
    // real /api/state always returns for an idle cell, because the boot code
    // calls updateUI() immediately and every field it reads must be there.
    globalThis.ROUTES = {
      "/api/state": { status: "idle", velocity_scale: 1.0, loop_active: false,
        agent_activity: {}, collision_alerts: [], current_plan_info: {},
        recordings: [], robot_positions: {}, robot_states: {}, planning_log: [] },
      "/api/topology": [],
    };
    globalThis.fetch = (url) => Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve(ROUTES[String(url).split('?')[0]] ?? {}),
      text: () => Promise.resolve(JSON.stringify(ROUTES[String(url).split('?')[0]] ?? {})),
    });
    globalThis.setInterval = () => 0;
    globalThis.setTimeout = (fn) => 0;
""")


def _dom_harness(html: str) -> str:
    """``DOM_HARNESS`` with the served ids of *this* page injected."""
    return ("const SERVED_IDS = "
            + json.dumps(sorted(_served_ids(html))) + ";\n"
            + "const DYNAMIC_PREFIXES = "
            + json.dumps(list(DYNAMIC_ID_PREFIXES)) + ";\n"
            + DOM_HARNESS)


@pytest.mark.skipif(not NODE, reason="node not installed; the static test above covers it")
class TestTheScriptRunsInAJsEngine:
    @pytest.mark.parametrize("rids", [["robot1"], ["robot1", "robot2"],
                                      ["robot1", "robot2", "robot3"]])
    def test_it_reaches_the_end_with_no_uncaught_error(self, tmp_path, rids):
        """Boot the served script for real. This is what a browser does."""
        js = tmp_path / "panel.js"
        js.write_text(HARNESS + _script(_render(rids)))
        done = subprocess.run([NODE, str(js)], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (
            f"the dashboard script threw at load for {len(rids)} robot(s), so "
            f"nothing after that line runs:\n{done.stderr[:1500]}")

    def test_it_parses(self, tmp_path):
        """A syntax error would be a different failure with the same blank page."""
        js = tmp_path / "panel.js"
        js.write_text(SCRIPT_2)
        done = subprocess.run([NODE, "--check", str(js)],
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr[:1500]

    def test_the_diagram_draws_one_actor_per_placed_agent(self, tmp_path):
        """Not just 'no error': buildDiagram must actually emit the nodes.

        Runs inside the script's own scope, because ``TOPOLOGY`` and ``LAYOUT``
        are block-scoped consts — assigning them from outside would leave
        ``buildDiagram`` looking at its own empty table and returning early,
        which is how a broken diagram could still pass.
        """
        topology = [{"id": i, "label": i, "icon": "x", "description": "d",
                     "topics_in": {}, "topics_out": {}}
                    for i in ("rai-orchestrator", "collaborative-planner",
                              "collision-monitor", "cell-controller",
                              "robot-agent-robot1", "robot-agent-robot2",
                              "rviz-web", "web-panel")]
        probe = textwrap.dedent(f"""
            TOPOLOGY = {json.dumps(topology)};
            buildDiagram();
            const placed = TOPOLOGY.filter(a => LAYOUT[a.id]).map(a => a.id);
            if (!LAYOUT["robot-agent-robot2"]) {{
              console.error("the injected arm never reached LAYOUT");
              process.exit(4);
            }}
            console.log("placed=" + placed.length + " actors=" + (APPENDED["actors-g"] || 0));
        """)
        js = tmp_path / "panel.js"
        # TOPOLOGY is declared with let/var in the template; the probe runs in the
        # same file so it shares that scope.
        html = _render(["robot1", "robot2"])
        js.write_text(_dom_harness(html) + _script(html) + probe)
        done = subprocess.run([NODE, str(js)], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, \
            f"buildDiagram did not place the agents:\n{done.stdout}\n{done.stderr[:1500]}"
        # 8 agents, 7 placed: web-panel has no LAYOUT entry, and buildDiagram
        # guards for that rather than throwing — the panel does not draw itself.
        assert "placed=7" in done.stdout, done.stdout
        # Each actor is a <g> appended to #actors-g: one per placed agent.
        assert "actors=7" in done.stdout, \
            f"agents were placed but not drawn: {done.stdout}"

    def test_the_log_panel_renders_an_entry_it_is_given(self, tmp_path):
        """The other half of the report: 'the dashboard doesn't show any logs'.

        The panel was empty for the same reason the diagram was — the script
        aborted before ``updateUI`` existed, so ``/api/state`` was never polled.
        Feeding it one entry proves the render path itself works, so an empty
        panel now means an idle cell rather than a broken page.
        """
        state = {
            "status": "idle", "velocity_scale": 1.0, "loop_active": False,
            "agent_activity": {}, "collision_alerts": [], "current_plan_info": {},
            "recordings": [], "robot_positions": {}, "robot_states": {},
            "planning_log": [
                {"time": "22:50:20", "level": "info",
                 "msg": "Velocity scale set to 1.0x"},
                {"time": "22:50:21", "level": "plan",
                 "msg": "Dispatched batch to robot1"},
            ],
        }
        probe = textwrap.dedent(f"""
            ROUTES["/api/state"] = {json.dumps(state)};
            updateUI();
            // Let the fetch promise chain settle before inspecting the DOM.
            setImmediate(() => {{
              const html = HTML["log-container"] || "";
              if (!html.includes("Velocity scale set to")) {{
                console.error("log-container was never written: " + JSON.stringify(html).slice(0, 300));
                process.exit(5);
              }}
              const rows = (html.match(/class='le le-/g) || []).length;
              console.log("logrows=" + rows);
            }});
        """)
        js = tmp_path / "panel.js"
        html = _render(["robot1", "robot2"])
        js.write_text(_dom_harness(html) + _script(html) + probe)
        done = subprocess.run([NODE, str(js)], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, \
            f"updateUI did not render the log:\n{done.stdout}\n{done.stderr[:1500]}"
        assert "logrows=2" in done.stdout, \
            f"expected both log entries rendered, got: {done.stdout}"


def _run_probe(tmp_path, probe, rids=("robot1", "robot2")):
    """Boot the real page script under the DOM harness, then run ``probe``.

    Module-level on purpose: as a base-class method it would drag the whole of
    ``TestTheScriptRunsInAJsEngine`` into every subclass and re-run it.
    """
    html = _render(list(rids))
    js = tmp_path / "panel.js"
    js.write_text(_dom_harness(html) + _script(html) + probe)
    return subprocess.run([NODE, str(js)], capture_output=True, text=True, timeout=60)


# ── the behaviour-tree view ──────────────────────────────────────────────────

# The shape GET /api/plans/<tree_id> returns: the tree without its blackboard,
# plus the waypoint count per referenced key. Trimmed from a real plan file
# (cells/*/agentic_planner/plans/bt_*.json) so the renderer is exercised against
# the format the planner writes, ports and "{key}" references included.
PLAN_BODY = {
    "success": True, "cell_id": "some_cell", "tree_id": "bt_1_some_cell",
    "num_operations": 1,
    "trajectory_sizes": {"traj_robot1_VIA_W06": 95, "traj_robot1_n3": 84},
    "behaviour_tree": {
        "tree_id": "bt_1_some_cell", "cell_id": "some_cell", "BTCPP_format": "4",
        "main_tree_to_execute": "MainTree", "metadata": {"steps": 1, "operations": 1},
        "root": {
            "type": "Sequence", "id": "aaa", "name": None, "ports": {}, "children": [
                {"type": "ExecuteTrajectory", "id": "bbb", "name": "move_robot1_VIA_W06",
                 "ports": {"robot_ns": "robot1", "operation_id": "VIA_W06",
                           "trajectory": "{traj_robot1_VIA_W06}",
                           "velocity_scale": "1.0"}, "children": []},
                {"type": "MoveHome", "id": "ccc", "name": "home_robot1",
                 "ports": {"robot_ns": "robot1", "trajectory": "{traj_robot1_n3}"},
                 "children": []},
            ],
        },
    },
}
PLAN_LIST = {"cell_id": "some_cell", "count": 1,
             "plans": [{"tree_id": "bt_1_some_cell", "created_s": 1.0,
                        "num_operations": 1, "size_bytes": 21240}]}


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheBehaviourTreeIsDrawn:
    """"visualize the tree in a graphical way" — so assert on the graphics.

    A tree view that fetches correctly and draws nothing looks identical to a
    broken endpoint from the operator's chair, so these count the SVG elements
    that come out of ``renderTree`` rather than checking the fetch happened.
    """

    def test_a_selected_plan_becomes_one_box_per_node(self, tmp_path):
        probe = textwrap.dedent(f"""
            ROUTES["/api/plans"] = {json.dumps(PLAN_LIST)};
            ROUTES["/api/plans/bt_1_some_cell"] = {json.dumps(PLAN_BODY)};
            loadPlanList();
            setImmediate(() => setImmediate(() => {{
              const svg = REG["bt-svg"];
              const boxes = svg.children.filter(c => c.id === "el:g").length;
              const edges = svg.children.filter(c => c.id === "el:path").length;
              console.log("boxes=" + boxes + " edges=" + edges
                          + " w=" + svg.attrs.width + " meta=" + REG["bt-meta"].textContent);
            }}));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        # Sequence + two leaves, and one edge per parent→child link.
        assert "boxes=3" in done.stdout, f"the tree was not drawn: {done.stdout}"
        assert "edges=2" in done.stdout, f"the links were not drawn: {done.stdout}"
        # A canvas with no width is an invisible tree.
        width = int(re.search(r"w=(\d+)", done.stdout).group(1))
        assert width > 100, done.stdout
        # The summary must count the operations, not the bytes.
        assert "1 executable operation(s)" in done.stdout, done.stdout
        # Waypoints are reported but never shipped: 95 + 84.
        assert "179 waypoints" in done.stdout, done.stdout

    def test_clicking_a_leaf_shows_its_ports(self, tmp_path):
        probe = textwrap.dedent(f"""
            ROUTES["/api/plans"] = {json.dumps(PLAN_LIST)};
            ROUTES["/api/plans/bt_1_some_cell"] = {json.dumps(PLAN_BODY)};
            loadPlanList();
            setImmediate(() => setImmediate(() => {{
              btSelect("bbb");
              const d = HTML["bt-detail"] || "";
              if (!d.includes("robot_ns") || !d.includes("ExecuteTrajectory")) {{
                console.error("port detail missing: " + JSON.stringify(d).slice(0, 300));
                process.exit(6);
              }}
              console.log("waypointnote=" + d.includes("95 waypoints"));
            }}));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "waypointnote=true" in done.stdout, \
            f"the trajectory port did not resolve to its size: {done.stdout}"

    def test_a_plan_with_no_tree_does_not_throw(self, tmp_path):
        """An empty store is the normal state of a fresh cell."""
        probe = textwrap.dedent("""
            ROUTES["/api/plans"] = { cell_id: "some_cell", count: 0, plans: [] };
            loadPlanList();
            setImmediate(() => console.log("meta=" + REG["bt-meta"].textContent));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "make plan" in done.stdout, \
            f"an empty plan store should say how to make one: {done.stdout}"


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheOperatorSeesWhatACommandDid:
    """``cmd()`` used to be fire-and-forget: no .then, no .catch.

    That is why "all the commands seem not to work" was unfalsifiable from the
    UI — a 500, an unreachable orchestrator and a successful dispatch all looked
    the same (nothing).
    """

    def test_a_dispatched_command_is_confirmed(self, tmp_path):
        probe = textwrap.dedent("""
            ROUTES["/api/command"] = { ok: true, command: "run",
                                       program: "car_door_spot_welding", dispatched: true };
            cmd("run");
            setImmediate(() => console.log("st=" + REG["cmd-status"].textContent));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "st=✓ run dispatched" in done.stdout, done.stdout
        assert "car_door_spot_welding" in done.stdout, done.stdout

    def test_a_rejected_command_says_so(self, tmp_path):
        probe = textwrap.dedent("""
            globalThis.fetch = (url) => Promise.resolve({
              ok: false, status: 500,
              text: () => Promise.resolve("<html>Internal Server Error</html>"),
              json: () => Promise.resolve({}) });
            cmd("agentic");
            setImmediate(() => console.log("st=" + REG["cmd-status"].textContent));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "st=✗ agentic failed" in done.stdout and "HTTP 500" in done.stdout, \
            f"a failed command must be visible, got: {done.stdout}"


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestTheSimulationViewIsTheLeftColumn:
    """"replace the agent diagram on the left with the vnc to the rviz".

    The viewer is served by the rviz-web container, not by this panel, so the
    iframe src has to be built from the hostname the operator actually used —
    hardcoding localhost works only on the machine running docker.
    """

    def test_the_iframe_points_at_the_novnc_container_on_the_page_host(self, tmp_path):
        probe = textwrap.dedent("""
            location.hostname = "cell-gateway.plant.local";
            reloadVnc();
            console.log("src=" + REG["vnc-frame"].src);
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "src=http://cell-gateway.plant.local:6080/vnc.html" in done.stdout, \
            f"the viewer is not pointed at the rviz-web host: {done.stdout}"
        assert "autoconnect=1" in done.stdout, \
            "the viewer should connect without a click"

    def test_the_left_column_is_the_viewer_and_the_diagram_moved_to_a_tab(self):
        html = _render(["robot1", "robot2"])
        left = re.search(r'<div id="canvas-wrap">(.*?)<!-- DIVIDER 1', html, re.S)
        assert left, "the left column is gone entirely"
        assert 'id="vnc-frame"' in left.group(1), \
            "the left column does not hold the simulation view"
        assert 'id="arch-svg"' not in left.group(1), \
            "the agent diagram is still in the left column"
        # Moved, not deleted: it is still reachable, and still drawable.
        assert 'id="pane-agents"' in html and 'id="arch-svg"' in html


# GET /api/plans/<tree_id>/nodes/<node_id> for the ExecuteTrajectory leaf above:
# the level the tree view could not show. Waypoints come per node and never with
# the plan, so this shape is the whole cost of descending one leaf.
NODE_BODY = {
    "success": True, "cell_id": "some_cell", "tree_id": "bt_1_some_cell",
    "node_id": "bbb", "type": "ExecuteTrajectory", "name": "move_robot1_VIA_W06",
    "ports": {"robot_ns": "robot1", "operation_id": "VIA_W06"},
    "children": [],
    "phase_segments": [
        {"phase": "transit", "start_index": 0, "end_index": 1,
         "start_s": 0.0, "end_s": 1.0, "points": 2},
        {"phase": "approach", "start_index": 2, "end_index": 3,
         "start_s": 1.4, "end_s": 2.0, "points": 2},
    ],
    "duration_s": 2.0,
    "waypoints": [
        {"positions": [0.0] * 6, "time_s": 0.0, "phase": "transit"},
        {"positions": [0.1] * 6, "time_s": 1.0, "phase": "transit"},
        {"positions": [0.2] * 6, "time_s": 1.4, "phase": "approach"},
        {"positions": [0.25] * 6, "time_s": 2.0, "phase": "approach"},
    ],
}
# The root Sequence: a control node has no motion of its own.
CONTROL_NODE_BODY = {
    "success": True, "cell_id": "some_cell", "tree_id": "bt_1_some_cell",
    "node_id": "aaa", "type": "Sequence", "name": None, "ports": {},
    "children": ["bbb", "ccc"], "phase_segments": [], "duration_s": 0.0,
    "waypoints": [],
}

# Boot the tree tab, then hand control to the probe. Every test below needs the
# plan list loaded and a tree selected first, because _btTree comes from the
# tree RESPONSE — "latest" has to resolve to a real id before any node fetch.
_TREE_LOADED = f"""
    ROUTES["/api/plans"] = {json.dumps(PLAN_LIST)};
    ROUTES["/api/plans/bt_1_some_cell"] = {json.dumps(PLAN_BODY)};
    ROUTES["/api/plans/bt_1_some_cell/nodes/bbb"] = {json.dumps(NODE_BODY)};
    ROUTES["/api/plans/bt_1_some_cell/nodes/aaa"] = {json.dumps(CONTROL_NODE_BODY)};
    ROUTES["/api/preview"] = {{ success: true, label: "ExecuteTrajectory move_robot1_VIA_W06",
      robot_ns: "robot1", waypoints: 4, duration_s: 2.0, commanded: false }};
    const CALLS = [];
    const _fetch = globalThis.fetch;
    globalThis.fetch = (url, opt) => {{ CALLS.push({{url: String(url), opt: opt || null}});
                                       return _fetch(url, opt); }};
    loadPlanList();
"""


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestClickingANodeDescendsIntoIt:
    """"doesn't go down in the single node" — so assert on what is under a leaf.

    The renderer was never truncating the tree (a synthetic three-level tree draws
    all ten boxes); what was missing is the level BELOW a leaf: the phases of the
    motion the leaf carries, and the waypoints inside each phase.
    """

    def test_a_leaf_lists_one_row_per_phase_of_its_motion(self, tmp_path):
        probe = textwrap.dedent(_TREE_LOADED + """
            setImmediate(() => setImmediate(() => {
              btSelect("bbb");
              setImmediate(() => setImmediate(() => {
                const h = HTML["bt-steps"] || "";
                console.log("steps=" + JSON.stringify(h));
              }));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        out = done.stdout
        # Both phases named, with their sample counts and their span in seconds:
        # which part is the fast cross-cell move and which the tool run-in.
        assert "transit" in out and "approach" in out, \
            f"the phases of the motion are not shown: {out[:600]}"
        assert "2 phase segment(s)" in out and "4 waypoint(s)" in out, out[:600]
        assert "2 pts" in out, f"the per-phase sample count is missing: {out[:600]}"
        assert "1.40" in out and "2.00" in out, \
            f"the phase timings are missing: {out[:600]}"

    def test_expanding_a_phase_lists_its_waypoints(self, tmp_path):
        """The finest grain: one row per sample, with its joints. Indices are the
        node's own, so they line up with what a preview is asked to hold."""
        probe = textwrap.dedent(_TREE_LOADED + """
            setImmediate(() => setImmediate(() => {
              btSelect("bbb");
              setImmediate(() => setImmediate(() => {
                btTogglePhase("bbb", 1);          // the approach
                const h = HTML["bt-steps"] || "";
                console.log("rows=" + (h.match(/t=/g) || []).length
                            + " idx2=" + h.includes("#2") + " idx3=" + h.includes("#3")
                            + " idx0=" + h.includes("#0")
                            + " joints=" + h.includes("0.200"));
              }));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "rows=2" in done.stdout, \
            f"the expanded phase does not list its two samples: {done.stdout}"
        # Its own waypoints and not the whole trajectory's.
        assert "idx2=true" in done.stdout and "idx3=true" in done.stdout, done.stdout
        assert "idx0=false" in done.stdout, \
            f"expanding the approach listed the transit's samples too: {done.stdout}"
        assert "joints=true" in done.stdout, \
            f"the joint values are not shown: {done.stdout}"

    def test_a_control_node_says_it_runs_its_children(self, tmp_path):
        """A Sequence has no motion. Showing "no waypoints" would read as a bug in
        the plan rather than as the shape of a control node."""
        probe = textwrap.dedent(_TREE_LOADED + """
            setImmediate(() => setImmediate(() => {
              btSelect("aaa");
              setImmediate(() => setImmediate(() =>
                console.log("steps=" + JSON.stringify(HTML["bt-steps"] || ""))));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "Control node" in done.stdout and "2 child node(s)" in done.stdout, \
            f"a control node's detail is wrong: {done.stdout[:600]}"

    def test_the_waypoints_are_fetched_per_node_and_not_with_the_plan(self, tmp_path):
        """The reason this is affordable on a 200-weld plan: the plan-level fetch
        never carries joint samples, so descending costs one node's worth."""
        probe = textwrap.dedent(_TREE_LOADED + """
            setImmediate(() => setImmediate(() => {
              btSelect("bbb");
              setImmediate(() => setImmediate(() =>
                console.log("urls=" + JSON.stringify(CALLS.map(c => c.url)))));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "/api/plans/bt_1_some_cell/nodes/bbb" in done.stdout, \
            f"selecting a node fetched no node detail: {done.stdout}"


@pytest.mark.skipif(not NODE, reason="node not installed")
class TestPreviewingANodeSaysWhatItDid:
    """A preview draws the motion on a ghost arm and commands nothing.

    Which makes honest reporting the whole feature: an operator who reads "▶" as
    "the cell is moving" is worse off than before. The route answers
    ``commanded: false`` (see ``tests/test_preview_never_commands.py``); these
    check the panel sends the right request and shows the right answer.
    """

    def test_previewing_a_phase_posts_that_phase(self, tmp_path):
        probe = textwrap.dedent(_TREE_LOADED + """
            setImmediate(() => setImmediate(() => {
              btSelect("bbb");
              setImmediate(() => setImmediate(() => {
                btPreview("bbb", {phase_index: 1});
                const post = CALLS.filter(c => c.url === "/api/preview").pop();
                console.log("method=" + (post && post.opt && post.opt.method)
                            + " body=" + (post && post.opt && post.opt.body));
              }));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "method=POST" in done.stdout, \
            f"the preview was not posted: {done.stdout}"
        body = json.loads(re.search(r"body=(\{.*\})", done.stdout).group(1))
        # The tree id from the response, not the "latest" the selector held.
        assert body == {"tree_id": "bt_1_some_cell", "node_id": "bbb",
                        "phase_index": 1}, body

    def test_a_successful_preview_says_the_cell_did_not_move(self, tmp_path):
        probe = textwrap.dedent(_TREE_LOADED + """
            setImmediate(() => setImmediate(() => {
              btSelect("bbb");
              setImmediate(() => setImmediate(() => {
                btPreview("bbb", {});
                setImmediate(() => setImmediate(() =>
                  console.log("say=" + REG["bt-preview-status"].textContent)));
              }));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "cell not moved" in done.stdout, (
            f"a preview must not be reportable as motion: {done.stdout}")
        assert "robot1" in done.stdout and "4 pt" in done.stdout, done.stdout

    def test_a_refused_preview_is_reported_as_a_failure(self, tmp_path):
        """The 400 the route answers for a control node, and the 503 it answers
        with no ROS node. Either one silently swallowed leaves the operator
        waiting for a ghost that will never move."""
        probe = textwrap.dedent(_TREE_LOADED + """
            setImmediate(() => setImmediate(() => {
              btSelect("bbb");
              setImmediate(() => setImmediate(() => {
                const inner = globalThis.fetch;
                globalThis.fetch = (url, opt) => (String(url) === "/api/preview"
                  ? Promise.resolve({ ok: false, status: 400,
                      json: () => Promise.resolve({ success: false,
                        error: "carries no trajectory to preview" }) })
                  : inner(url, opt));
                btPreview("bbb", {});
                setImmediate(() => setImmediate(() =>
                  console.log("say=" + REG["bt-preview-status"].textContent)));
              }));
            }));
        """)
        done = _run_probe(tmp_path, probe)
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr[:1500]}"
        assert "✗" in done.stdout and "carries no trajectory" in done.stdout, \
            f"a refused preview looked like a success: {done.stdout}"
