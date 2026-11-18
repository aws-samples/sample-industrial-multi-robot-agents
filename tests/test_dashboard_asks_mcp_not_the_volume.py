"""The panel asks the owning agent; it does not read the shared volume.

``dashboard/web_control_panel.py`` mounts /cells alongside the agents, and it used
to answer several questions by reading that mount itself: robots.yaml for the arm
list and each arm's base pose, tasks/*.json for the chat context, and
recordings/plans for the recording list. Every one of those files belongs to an
agent that already publishes it as an MCP tool, and the second independent reader
is exactly how the two drifted:

  * it globbed ``plan_*.json`` for recordings long after the recorder had moved to
    ``<tree_id>.rrd`` — "no recordings" with recordings on disk;
  * it summarised task programs as ``tasks[].parameters.points`` on a hardcoded
    ``welding_task.json``, a shape no writer has produced since the format became
    ``operations[]`` — so the chat LLM was told nothing about the cell's programs,
    silently, because the failure was swallowed.

These tests pin the direction of the dependency (MCP, not open()), the failure
behaviour (degrade one panel, never blank the page), and the ONE deliberate
exception: the arm list falls back to the file so the panel can boot while the
trajectory agent is still starting.

The module is imported with ROS and Flask stubbed — it needs neither to answer
these questions, and neither exists on a plain checkout. The Flask stub is a real
(tiny) object rather than a MagicMock so that route functions keep their bodies.
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PANEL_SRC = (PROJECT_ROOT / "dashboard" / "web_control_panel.py").read_text()


def _code_only(source: str) -> str:
    """*source* with comments and docstrings removed, everything else intact.

    The comments and docstrings in the panel NAME the dead schemas and the removed
    globs on purpose — that is the record of why they went. A grep for "what is
    gone" has to look at code only, or the explanation of the fix reads as the
    bug. Other string literals stay, because a dead schema shows up precisely as
    one (``data["tasks"]``, ``"operation_type"``).
    """
    import io
    import tokenize

    tree = ast.parse(source)
    prose_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        for child in body if isinstance(body, list) else []:
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) \
                    and isinstance(child.value.value, str):
                prose_lines.update(range(child.lineno, (child.end_lineno or child.lineno) + 1))

    comment_lines: set[int] = set()
    stripped: dict[int, str] = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])
            stripped[tok.start[0]] = tok.line[:tok.start[1]]

    out = []
    for n, line in enumerate(source.splitlines(), start=1):
        if n in prose_lines:
            continue
        out.append(stripped[n] if n in comment_lines else line)
    return "\n".join(out)


PANEL_CODE = _code_only(PANEL_SRC)


# ─── import harness ───────────────────────────────────────────────────────────

def _install_stubs() -> None:
    """Register the ROS/Flask stand-ins the panel imports at module level."""
    for name in ("rclpy", "rclpy.node", "rclpy.qos", "std_msgs", "std_msgs.msg",
                 "visualization_msgs", "visualization_msgs.msg"):
        sys.modules.setdefault(name, MagicMock())

    if "flask" in sys.modules and getattr(sys.modules["flask"], "_panel_stub", False):
        return
    if "flask" in sys.modules:
        return                       # a real Flask is installed; use it

    flask = types.ModuleType("flask")
    flask._panel_stub = True         # type: ignore[attr-defined]

    class _App:
        """Just enough Flask to let @app.route keep the function it decorates."""

        def __init__(self, name):
            self.name = name
            self.routes: dict = {}

        def route(self, rule, **_kw):
            def deco(fn):
                self.routes[rule] = fn
                return fn            # NOT a mock: the body must stay callable
            return deco

        def run(self, *a, **k):       # pragma: no cover - never started here
            raise AssertionError("the test suite must not start the web server")

    flask.Flask = _App                                        # type: ignore[attr-defined]
    flask.jsonify = lambda *a, **k: (a[0] if a else k)         # type: ignore[attr-defined]
    flask.render_template_string = lambda *a, **k: ""          # type: ignore[attr-defined]
    flask.request = MagicMock()                               # type: ignore[attr-defined]
    sys.modules["flask"] = flask


@pytest.fixture(scope="module")
def panel():
    """The imported dashboard module."""
    _install_stubs()
    mod = importlib.import_module("dashboard.web_control_panel")
    return mod


@pytest.fixture
def calls(panel, monkeypatch):
    """Record every MCP call and answer it from a scripted table.

    Keyed by tool name; the value is the parsed result the panel would have got
    back. An unscripted tool answers with a failure, so a test that forgets to
    script a call sees the degraded path rather than a MagicMock.
    """
    recorded: list = []
    answers: dict = {}

    def _fake_mcp_call(url, tool, arguments, timeout=None):
        recorded.append({"url": url, "tool": tool, "arguments": arguments})
        if tool not in answers:
            return {"success": False, "error": f"{tool}: not scripted"}
        answer = answers[tool]
        return answer(arguments) if callable(answer) else answer

    monkeypatch.setattr(panel, "_mcp_call", _fake_mcp_call)
    # Every cache in the module is per-process; clear it so ordering cannot make
    # one test answer another test's question.
    panel._robots_cache.update(pairs=[], at=0.0)
    panel._base_pose_cache.update(poses={}, at=0.0)

    return types.SimpleNamespace(recorded=recorded, answers=answers,
                                 tools=lambda: [c["tool"] for c in recorded])


# ─── the volume is not read any more ─────────────────────────────────────────

class TestNothingOpensTheVolumeExceptTheArmListFallback:

    def test_the_only_open_call_left_is_the_robots_yaml_fallback(self):
        """Source-level, because the point is that the CAPABILITY is gone."""
        tree = ast.parse(PANEL_SRC)
        openers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
                fn = next((f.name for f in ast.walk(tree)
                           if isinstance(f, ast.FunctionDef)
                           and f.lineno <= node.lineno <= (f.end_lineno or f.lineno)),
                          "<module>")
                openers.append(fn)
        assert openers == ["_robots_from_volume"], (
            f"these still read the shared volume directly: {openers}")

    def test_no_task_file_or_plan_globbing_remains(self):
        for dead in ("cell_layout.task_file(", "cell_layout.plans_dir(",
                     "cell_layout.recordings_dir(", "plan_*.json", "*.rrd",
                     "/sim_recordings"):
            assert dead not in PANEL_CODE, (
                f"{dead!r} is back in the panel — that question belongs to an agent")

    def test_the_legacy_task_schema_is_gone_from_the_panel(self):
        """`tasks[].parameters.points` with operation_type WELDING: the dead shape."""
        for dead in ('"operation_type"', 'operation_type ==', '["tasks"]',
                     'parameters"]["points"'):
            assert dead not in PANEL_CODE


# ─── which arms exist ────────────────────────────────────────────────────────

class TestTheArmListComesFromTheTrajectoryAgent:

    CELL_INFO = {"robots": {
        "robot1": {"namespace": "arms", "base_position": [0.07, -3.32, 0],
                   "base_orientation_yaw": 1.5707963},
        "robot2": {"namespace": "arms", "base_position": [-1.18, 1.04, 0],
                   "base_orientation_yaw": 0.0},
    }}

    def test_it_asks_get_cell_info(self, panel, calls):
        calls.answers["get_cell_info"] = self.CELL_INFO

        pairs = panel._discover_robots()

        assert "get_cell_info" in calls.tools()
        assert pairs == [("arms", "robot1"), ("arms", "robot2")]

    def test_it_asks_the_trajectory_agent_not_the_cell_controller(self, panel, calls):
        calls.answers["get_cell_info"] = self.CELL_INFO

        panel._discover_robots()

        url = next(c["url"] for c in calls.recorded if c["tool"] == "get_cell_info")
        assert url == panel._traj_mcp()

    def test_the_answer_is_cached_so_a_page_refresh_is_not_a_round_trip(self, panel, calls):
        calls.answers["get_cell_info"] = self.CELL_INFO

        panel._discover_robots()
        panel._discover_robots()
        panel._discover_robots()

        assert calls.tools().count("get_cell_info") == 1

    def test_an_arm_with_no_namespace_is_skipped(self, panel, calls):
        """The namespace is half the topic name; a guess would subscribe nowhere."""
        calls.answers["get_cell_info"] = {"robots": {
            "robot1": {"namespace": "arms"},
            "robot2": {"base_position": [0, 0, 0]},
        }}

        assert panel._discover_robots() == [("arms", "robot1")]

    def test_it_falls_back_to_the_file_when_the_agent_is_unreachable(self, panel, calls,
                                                                    monkeypatch):
        """Deliberate: a panel that booted first must still find its arms.

        This list decides which ROS topics get subscribed. An empty answer would
        leave the page subscribed to no arm at all, with nothing on screen to say
        why — so the fallback reads the same file the agent reads.
        """
        monkeypatch.setattr(panel, "_robots_from_volume",
                            lambda: [("arms", "robotA"), ("arms", "robotB")])

        pairs = panel._discover_robots()          # get_cell_info is unscripted → fails

        assert pairs == [("arms", "robotA"), ("arms", "robotB")]

    def test_the_fallback_is_not_cached_as_the_truth(self, panel, calls, monkeypatch):
        """Once the agent answers, its answer wins — the file was the stand-in."""
        monkeypatch.setattr(panel, "_robots_from_volume", lambda: [])

        panel._discover_robots()
        assert panel._robots_cache["pairs"] == []


class TestTheBasePoseComesFromTheTrajectoryAgent:
    """FK needs the arm's world placement. Reading robots.yaml here is how the
    panel and the planner could disagree about where an arm stands."""

    CELL_INFO = {"robots": {
        "robot1": {"namespace": "arms", "base_position": [0.07, -3.32, 0.0],
                   "base_orientation_yaw": 1.5707963},
    }}

    def test_it_returns_the_configured_position_and_yaw(self, panel, calls):
        calls.answers["get_cell_info"] = self.CELL_INFO

        pose = panel._robot_base_pose("robot1")

        assert pose["base_position"] == [0.07, -3.32, 0.0]
        assert pose["base_orientation_yaw"] == pytest.approx(1.5707963)

    def test_it_does_not_call_mcp_once_per_joint_sample(self, panel, calls):
        """joint_states arrives at 10 Hz per arm; FK runs on every one of them."""
        calls.answers["get_cell_info"] = self.CELL_INFO

        for _ in range(50):
            panel._robot_base_pose("robot1")

        assert calls.tools().count("get_cell_info") == 1

    def test_an_unreachable_agent_costs_one_call_per_ttl_not_one_per_sample(self, panel,
                                                                          calls):
        """The negative case is the hot one: a blocked call per sample would stall."""
        for _ in range(50):
            panel._robot_base_pose("robot1")

        assert calls.tools().count("get_cell_info") == 1

    def test_an_unknown_arm_gets_the_identity_pose_not_an_exception(self, panel, calls):
        """FK then reports in the arm's own base frame: wrong by an offset, bounded."""
        calls.answers["get_cell_info"] = self.CELL_INFO

        pose = panel._robot_base_pose("robot_that_is_not_configured")

        assert pose == {"base_position": [0.0, 0.0, 0.0], "base_orientation_yaw": 0.0}

    def test_a_missing_yaw_is_zero_and_a_float(self, panel, calls):
        calls.answers["get_cell_info"] = {
            "robots": {"robot1": {"base_position": [1.0, 2.0, 0.0]}}}

        pose = panel._robot_base_pose("robot1")

        assert pose["base_orientation_yaw"] == 0.0
        assert isinstance(pose["base_orientation_yaw"], float)

    def test_fk_looks_the_pose_up_per_call_not_at_model_load(self):
        """An arm whose pose was unavailable when Pinocchio loaded must not stay
        at the origin for the life of the process."""
        tree = ast.parse(PANEL_SRC)
        fk = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_compute_fk")
        body = _code_only(textwrap.dedent(ast.get_source_segment(PANEL_SRC, fk) or ""))
        assert "_robot_base_pose(ns)" in body
        assert "robots_yaml" not in body


# ─── the recording list ──────────────────────────────────────────────────────

class TestTheRecordingListIsJoinedFromTwoAgents:

    LISTING = {"recordings": [
        {"tree_id": "bt_new", "path": "/cells/c/recordings/bt_new.rrd",
         "size_bytes": 2048, "modified_s": 3.0, "timestamp": "2026-01-02T03:04:05"},
        {"tree_id": "bt_old", "path": "/cells/c/recordings/bt_old.rrd",
         "size_bytes": 1024, "modified_s": 1.0, "timestamp": "2026-01-01T03:04:05"},
    ]}
    PLANS = {"plans": [{"tree_id": "bt_new", "num_operations": 17}]}

    def test_the_cell_controller_is_asked_for_the_recordings(self, panel, calls):
        calls.answers["list_recordings"] = self.LISTING

        panel._list_recordings()

        url = next(c["url"] for c in calls.recorded if c["tool"] == "list_recordings")
        assert url == panel._cell_mcp()

    def test_the_trajectory_agent_is_asked_for_the_operation_counts(self, panel, calls):
        calls.answers["list_recordings"] = self.LISTING
        calls.answers["list_calculated_plans"] = self.PLANS

        recs = panel._list_recordings()

        by_tree = {r["tree_id"]: r["operations"] for r in recs}
        assert by_tree["bt_new"] == 17

    def test_a_recording_with_no_plan_is_still_listed(self, panel, calls):
        """It can outlive its plan, or be a bare home_reset_* — and it is replayable."""
        calls.answers["list_recordings"] = self.LISTING
        calls.answers["list_calculated_plans"] = self.PLANS

        recs = panel._list_recordings()

        assert [r["tree_id"] for r in recs] == ["bt_new", "bt_old"]
        assert next(r for r in recs if r["tree_id"] == "bt_old")["operations"] is None

    def test_the_order_the_agent_gave_is_preserved(self, panel, calls):
        """The agent sorted by mtime; re-sorting here would need the panel to stat
        paths in a container that may not even see the same mount."""
        calls.answers["list_recordings"] = self.LISTING

        assert [r["tree_id"] for r in panel._list_recordings()] == ["bt_new", "bt_old"]

    def test_the_name_is_derived_from_the_path_the_agent_reported(self, panel, calls):
        calls.answers["list_recordings"] = self.LISTING

        assert panel._list_recordings()[0]["name"] == "bt_new.rrd"

    def test_a_recording_without_a_path_still_gets_a_name(self, panel, calls):
        calls.answers["list_recordings"] = {"recordings": [{"tree_id": "bt_x"}]}

        assert panel._list_recordings()[0]["name"] == "bt_x.rrd"

    def test_size_and_timestamp_are_passed_through(self, panel, calls):
        calls.answers["list_recordings"] = self.LISTING

        first = panel._list_recordings()[0]
        assert first["size_bytes"] == 2048
        assert first["timestamp"] == "2026-01-02T03:04:05"

    def test_an_unreachable_cell_controller_yields_an_empty_list(self, panel, calls):
        """One empty panel, not a 500 that takes the whole page with it."""
        assert panel._list_recordings() == []

    def test_a_plan_listing_that_fails_does_not_lose_the_recordings(self, panel, calls):
        calls.answers["list_recordings"] = self.LISTING
        # list_calculated_plans unscripted → fails

        recs = panel._list_recordings()

        assert [r["tree_id"] for r in recs] == ["bt_new", "bt_old"]
        assert all(r["operations"] is None for r in recs)

    def test_garbage_entries_are_skipped_not_crashed_on(self, panel, calls):
        calls.answers["list_recordings"] = {"recordings": ["not-a-dict", None,
                                                          {"tree_id": "bt_ok"}]}

        assert [r["tree_id"] for r in panel._list_recordings()] == ["bt_ok"]

    def test_the_list_is_capped_so_one_long_lived_cell_cannot_flood_the_panel(self,
                                                                             panel, calls):
        calls.answers["list_recordings"] = {
            "recordings": [{"tree_id": f"bt_{i}"} for i in range(100)]}

        assert len(panel._list_recordings()) == 20


# ─── the MCP client itself ───────────────────────────────────────────────────

class TestTheMcpClientDegradesInsteadOfRaising:
    """One panel of the page may go blank; the page may not."""

    def test_a_transport_failure_becomes_a_failure_result(self, panel):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = panel._mcp_call("http://nowhere:1/mcp", "get_cell_info", {})

        assert result["success"] is False
        assert "get_cell_info" in result["error"]

    def test_a_jsonrpc_error_becomes_a_failure_result(self, panel):
        body = json.dumps({"jsonrpc": "2.0", "id": 1,
                           "error": {"code": -32602, "message": "no such cell"}}).encode()
        with patch("urllib.request.urlopen", _fake_response(body)):
            result = panel._mcp_call("http://x/mcp", "get_cell_info", {})

        assert result["success"] is False
        assert "no such cell" in result["error"]

    def test_prose_instead_of_json_becomes_a_failure_result(self, panel):
        body = json.dumps({"result": {"content": [{"text": "I could not do that"}]}}).encode()
        with patch("urllib.request.urlopen", _fake_response(body)):
            result = panel._mcp_call("http://x/mcp", "get_cell_info", {})

        assert result["success"] is False
        assert "not JSON" in result["error"]

    def test_an_empty_content_list_becomes_a_failure_result(self, panel):
        body = json.dumps({"result": {"content": []}}).encode()
        with patch("urllib.request.urlopen", _fake_response(body)):
            result = panel._mcp_call("http://x/mcp", "list_recordings", {})

        assert result["success"] is False

    def test_a_tool_result_is_parsed_out_of_the_content_block(self, panel):
        payload = {"recordings": [{"tree_id": "bt_a"}]}
        body = json.dumps({"result": {"content": [{"text": json.dumps(payload)}]}}).encode()
        with patch("urllib.request.urlopen", _fake_response(body)):
            result = panel._mcp_call("http://x/mcp", "list_recordings", {})

        assert result == payload

    def test_the_request_is_a_jsonrpc_tools_call(self, panel):
        """The panel speaks the same wire protocol as utils/mcp_http.py."""
        seen: dict = {}
        body = json.dumps({"result": {"content": [{"text": "{}"}]}}).encode()

        def _capture(req, timeout=None):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data)
            return _reader(body)

        with patch("urllib.request.urlopen", _capture):
            panel._mcp_call("http://traj:8890/mcp", "get_cell_info", {"cell_id": "c"})

        assert seen["url"] == "http://traj:8890/mcp"
        assert seen["body"]["method"] == "tools/call"
        assert seen["body"]["params"] == {"name": "get_cell_info",
                                         "arguments": {"cell_id": "c"}}

    def test_a_missing_success_flag_is_not_treated_as_failure(self, panel):
        """get_cell_info just returns the cell; only an explicit false is a refusal."""
        assert panel._mcp_failed({"robots": {}}) == ""
        assert panel._mcp_failed({"success": False, "error": "nope"}) == "nope"
        assert panel._mcp_failed({"error": "boom"}) == "boom"
        assert panel._mcp_failed("not a dict") != ""


# ─── chat context and chat actions ───────────────────────────────────────────

class TestTheChatContextAsksForTheProgramCatalogue:
    """The LLM used to be told nothing about the cell's programs, because the
    panel counted a schema nobody writes in a file no cell defines, and the
    failure was swallowed at DEBUG."""

    def test_the_catalogue_call_replaced_the_hardcoded_program(self):
        tree = ast.parse(PANEL_SRC)
        chat = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "chat")
        body = _code_only(textwrap.dedent(ast.get_source_segment(PANEL_SRC, chat) or ""))
        assert '"list_task_programs"' in body
        assert "welding_task" not in body

    def test_an_unavailable_catalogue_is_stated_not_hidden(self):
        """"unavailable (reason)" beats an empty section the LLM reads as "none"."""
        assert "Task programs: unavailable" in PANEL_SRC

    def test_a_cell_with_no_programs_says_so_explicitly(self):
        assert "Task programs: none defined for this cell yet" in PANEL_SRC


class TestCreateWeldWritesThroughTheAgent:

    PROGRAM = {"success": True, "task": {
        "description": "door welds",
        "operations": [{"id": "W001", "type": "weld", "position": [1.0, 0.0, 0.5],
                        "approach_vector": [0, 0, -1]}]}}

    def _saved(self, calls) -> dict:
        return next(c["arguments"] for c in calls.recorded
                    if c["tool"] == "save_task_program")

    def test_it_reads_then_writes_through_mcp(self, panel, calls):
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": True, "num_operations": 2}

        reply = panel._execute_chat_action(
            {"action": "create_weld", "program": "door_welds",
             "x": 1.2, "y": -0.4, "z": 0.9})

        assert calls.tools() == ["get_task_program", "save_task_program"]
        assert "✓" in reply

    def test_the_new_operation_uses_the_schema_the_planner_reads(self, panel, calls):
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": True, "num_operations": 2}

        panel._execute_chat_action({"action": "create_weld", "program": "door_welds",
                                    "x": 1.2, "y": -0.4, "z": 0.9})

        op = self._saved(calls)["task"]["operations"][-1]
        assert op["type"] == "weld"
        assert op["position"] == [1.2, -0.4, 0.9]
        assert op["approach_vector"] == [0.0, 0.0, -1.0]
        assert op["id"]

    def test_the_existing_operations_are_kept(self, panel, calls):
        """Appending must not be replacing: the other welds are somebody's work."""
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": True, "num_operations": 2}

        panel._execute_chat_action({"action": "create_weld", "program": "door_welds",
                                    "x": 0.0, "y": 0.0, "z": 0.0})

        ops = self._saved(calls)["task"]["operations"]
        assert ops[0] == self.PROGRAM["task"]["operations"][0]
        assert len(ops) == 2

    def test_the_rest_of_the_program_survives_the_round_trip(self, panel, calls):
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": True, "num_operations": 2}

        panel._execute_chat_action({"action": "create_weld", "program": "door_welds",
                                    "x": 0.0, "y": 0.0, "z": 0.0})

        assert self._saved(calls)["task"]["description"] == "door welds"

    def test_it_overwrites_deliberately_because_it_is_editing_that_program(self, panel,
                                                                          calls):
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": True, "num_operations": 2}

        panel._execute_chat_action({"action": "create_weld", "program": "door_welds",
                                    "x": 0.0, "y": 0.0, "z": 0.0})

        assert self._saved(calls)["overwrite"] is True

    def test_with_exactly_one_program_it_does_not_need_to_be_told_which(self, panel,
                                                                       calls):
        calls.answers["list_task_programs"] = {"programs": ["door_welds"], "count": 1}
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": True, "num_operations": 2}

        panel._execute_chat_action({"action": "create_weld", "x": 0.0, "y": 0.0, "z": 0.0})

        assert self._saved(calls)["program"] == "door_welds"

    def test_with_several_programs_it_asks_rather_than_guesses(self, panel, calls):
        """A weld in the wrong program looks exactly like a weld in the right one."""
        calls.answers["list_task_programs"] = {"programs": ["door_welds", "sill_welds"],
                                               "count": 2}

        reply = panel._execute_chat_action({"action": "create_weld",
                                           "x": 0.0, "y": 0.0, "z": 0.0})

        assert "door_welds" in reply and "sill_welds" in reply
        assert "save_task_program" not in calls.tools()

    def test_with_no_programs_it_points_at_the_wizard(self, panel, calls):
        calls.answers["list_task_programs"] = {"programs": [], "count": 0}

        reply = panel._execute_chat_action({"action": "create_weld",
                                           "x": 0.0, "y": 0.0, "z": 0.0})

        assert "Wizard" in reply
        assert "save_task_program" not in calls.tools()

    def test_a_read_that_fails_does_not_write(self, panel, calls):
        """Writing over a program you could not read is how one gets truncated."""
        calls.answers["save_task_program"] = {"success": True}

        reply = panel._execute_chat_action({"action": "create_weld",
                                            "program": "door_welds",
                                            "x": 0.0, "y": 0.0, "z": 0.0})

        assert "save_task_program" not in calls.tools()
        assert "Could not read" in reply

    def test_a_refused_write_is_reported_not_claimed_as_success(self, panel, calls):
        """The old code reported "✓ Weld point added" from inside a bare except."""
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": False,
                                              "error": "already exists"}

        reply = panel._execute_chat_action({"action": "create_weld",
                                            "program": "door_welds",
                                            "x": 0.0, "y": 0.0, "z": 0.0})

        assert "✓" not in reply
        assert "already exists" in reply

    def test_adding_a_weld_plans_nothing_and_moves_nothing(self, panel, calls):
        calls.answers["get_task_program"] = self.PROGRAM
        calls.answers["save_task_program"] = {"success": True, "num_operations": 2}

        reply = panel._execute_chat_action({"action": "create_weld",
                                            "program": "door_welds",
                                            "x": 0.0, "y": 0.0, "z": 0.0})

        assert not ({"plan_trajectories", "execute_behaviour_tree", "move_robot",
                     "replay_plan"} & set(calls.tools()))
        assert "Nothing has been planned or moved" in reply


class TestTheChatActionExecutorIsReachable:
    """It lost its ``def`` line upstream: the body sat unreachable after
    ``set_loop``'s return, so every action ``chat()`` extracted raised NameError
    and was swallowed — run, stop, reset, replay, move, create_weld, all of them."""

    def test_chat_calls_a_function_that_exists(self, panel):
        assert callable(getattr(panel, "_execute_chat_action", None))

    def test_no_function_in_the_panel_has_unreachable_statements(self):
        """The shape of the original bug: a body with no signature of its own."""
        tree = ast.parse(PANEL_SRC)
        stray = []
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for st in fn.body[1:]:
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
                        and isinstance(st.value.value, str):
                    stray.append((fn.name, st.lineno))
        assert stray == [], f"a docstring mid-body means a lost def: {stray}"

    def test_an_unknown_action_is_ignored_quietly(self, panel, calls):
        assert panel._execute_chat_action({"action": "sudo_make_me_a_sandwich"}) == ""

    def test_an_action_dict_with_no_action_is_ignored_quietly(self, panel, calls):
        assert panel._execute_chat_action({}) == ""


# ─── helpers ──────────────────────────────────────────────────────────────────

def _reader(body: bytes):
    """A context-manager response object with .read()."""
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body
    return _Resp()


def _fake_response(body: bytes):
    def _urlopen(req, timeout=None):
        return _reader(body)
    return _urlopen
