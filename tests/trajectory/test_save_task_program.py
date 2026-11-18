"""The task-program WRITER: agents/trajectory/task_programs.save_program.

Exposed as the ``save_task_program`` MCP tool. It exists so that a caller which
is not the trajectory agent — the dashboard, the orchestrator, a native MCP
client — can author a program without opening the shared volume itself. Two
writers with no agreement on the schema is how the panel came to append
``tasks[].parameters.points`` to a format whose readers had long since moved to
``operations[]``, and report success while doing it.

So these tests pin three things: what a legal program is, that an existing one
is never replaced by accident, and that saving COMMANDS NOTHING.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.trajectory.task_programs import save_program  # noqa: E402
from utils import cell_layout  # noqa: E402

CELL = "welding_cell_2_small"

PROGRAM = {
    "description": "two spot welds on the inner sill",
    "operations": [
        {"id": "W001", "type": "weld", "position": [1.2, -0.4, 0.9],
         "approach_vector": [0, 0, -1]},
        {"id": "W002", "type": "weld", "position": [1.3, -0.4, 0.9],
         "approach_vector": [0, 0, -1]},
    ],
}


@pytest.fixture
def cell_base(tmp_path, monkeypatch):
    """Point cell_layout at a scratch volume for the duration of one test."""
    monkeypatch.setenv("CELL_BASE_PATH", str(tmp_path))
    return tmp_path


def _written(cell_base) -> dict:
    return json.loads((cell_base / CELL / "tasks" / "door_welds.json").read_text())


# ── what it writes ───────────────────────────────────────────────────────────

class TestItWritesThePlannerReadableSchema:

    def test_it_creates_the_tasks_directory_and_the_file(self, cell_base):
        result = save_program(CELL, "door_welds", PROGRAM)

        assert result["success"] is True
        assert Path(result["path"]) == cell_layout.task_file(CELL, "door_welds")
        assert Path(result["path"]).exists()

    def test_operations_survive_verbatim(self, cell_base):
        """The array the planner consumes is the array the caller handed over.

        ``plan_trajectories`` takes ``operations`` with no transform, so any
        normalisation here would be a silent divergence from what was authored.
        """
        save_program(CELL, "door_welds", PROGRAM)

        assert _written(cell_base)["operations"] == PROGRAM["operations"]

    def test_program_and_cell_id_are_set_from_where_the_file_lands(self, cell_base):
        """A program copied out of another cell must not keep claiming it."""
        borrowed = {**PROGRAM, "program": "somebody_elses", "cell_id": "other_cell"}

        save_program(CELL, "door_welds", borrowed)

        body = _written(cell_base)
        assert body["program"] == "door_welds"
        assert body["cell_id"] == CELL

    def test_unrelated_top_level_fields_are_preserved(self, cell_base):
        """trajectory_backend / extensions / description belong to the caller."""
        save_program(CELL, "door_welds",
                     {**PROGRAM, "trajectory_backend": "ompl",
                      "extensions": {"tool": "torch_A"}})

        body = _written(cell_base)
        assert body["trajectory_backend"] == "ompl"
        assert body["extensions"] == {"tool": "torch_A"}
        assert body["description"] == PROGRAM["description"]

    def test_it_reports_the_operation_count_it_wrote(self, cell_base):
        result = save_program(CELL, "door_welds", PROGRAM)

        assert result["num_operations"] == len(PROGRAM["operations"])

    def test_the_file_is_valid_json_with_a_trailing_newline(self, cell_base):
        """Programs are hand-edited and diffed; a no-newline file is a nuisance."""
        save_program(CELL, "door_welds", PROGRAM)

        text = cell_layout.task_file(CELL, "door_welds").read_text()
        assert text.endswith("\n")
        json.loads(text)          # raises if the write was not complete JSON

    def test_no_temp_file_is_left_behind(self, cell_base):
        """The write is staged then renamed; the staging file must not survive."""
        save_program(CELL, "door_welds", PROGRAM)

        leftovers = list((cell_base / CELL / "tasks").glob("*.tmp"))
        assert leftovers == []


# ── what it refuses ──────────────────────────────────────────────────────────

class TestItRefusesWhatWouldNotPlan:

    def test_a_program_with_no_operations_is_refused(self, cell_base):
        result = save_program(CELL, "empty", {"operations": []})

        assert result["success"] is False
        assert "operations" in result["error"]
        assert not cell_layout.task_file(CELL, "empty").exists()

    def test_a_program_missing_operations_entirely_is_refused(self, cell_base):
        result = save_program(CELL, "shapeless", {"description": "nothing here"})

        assert result["success"] is False
        assert not cell_layout.task_file(CELL, "shapeless").exists()

    def test_the_legacy_tasks_shape_is_refused(self, cell_base):
        """``tasks[].parameters.points`` is the schema this tool exists to stop.

        It is not a program the planner can read, so accepting it would recreate
        the bug: a file on the volume, a success reported, and nothing plannable.
        """
        legacy = {"program_name": "welding_task",
                  "tasks": [{"step_number": 1, "operation_type": "WELDING",
                             "parameters": {"points": [{"x": 1, "y": 2, "z": 3}]}}]}

        result = save_program(CELL, "legacy", legacy)

        assert result["success"] is False
        assert not cell_layout.task_file(CELL, "legacy").exists()

    def test_a_non_dict_task_is_refused(self, cell_base):
        result = save_program(CELL, "listy", [{"id": "W001"}])   # type: ignore[arg-type]

        assert result["success"] is False

    @pytest.mark.parametrize("program", [
        "../escape", "sub/dir", "", "with space", "semi;colon", "..",
    ])
    def test_a_name_that_is_not_a_plain_identifier_is_refused(self, cell_base, program):
        """The name becomes a file name — a path in it would escape the cell."""
        result = save_program(CELL, program, PROGRAM)

        assert result["success"] is False
        assert "identifier" in result["error"]

    def test_nothing_is_written_outside_the_cell_for_a_traversing_name(self, cell_base):
        save_program(CELL, "../../escaped", PROGRAM)

        assert list(cell_base.rglob("escaped.json")) == []


class TestExistingProgramsAreNotDestroyedByAccident:

    def test_an_existing_name_is_refused_without_overwrite(self, cell_base):
        save_program(CELL, "door_welds", PROGRAM)

        result = save_program(CELL, "door_welds",
                             {"operations": [{"id": "X", "type": "weld",
                                              "position": [0, 0, 0],
                                              "approach_vector": [0, 0, -1]}]})

        assert result["success"] is False
        assert "overwrite" in result["error"]

    def test_the_refusal_leaves_the_original_content_intact(self, cell_base):
        save_program(CELL, "door_welds", PROGRAM)

        save_program(CELL, "door_welds", {"operations": [{"id": "X", "type": "weld",
                                                         "position": [0, 0, 0],
                                                         "approach_vector": [0, 0, -1]}]})

        assert _written(cell_base)["operations"] == PROGRAM["operations"]

    def test_overwrite_true_replaces_and_says_so(self, cell_base):
        save_program(CELL, "door_welds", PROGRAM)
        replacement = {"operations": [{"id": "W009", "type": "rivet",
                                       "position": [2.0, 0.1, 0.5],
                                       "approach_vector": [0, -1, 0]}]}

        result = save_program(CELL, "door_welds", replacement, overwrite=True)

        assert result["success"] is True
        assert result["overwritten"] is True
        assert _written(cell_base)["operations"] == replacement["operations"]

    def test_a_first_save_does_not_claim_to_have_overwritten(self, cell_base):
        result = save_program(CELL, "door_welds", PROGRAM)

        assert result["overwritten"] is False


# ── it authors, it does not run ───────────────────────────────────────────────

class TestSavingCommandsNothing:

    def test_the_result_says_nothing_was_commanded(self, cell_base):
        result = save_program(CELL, "door_welds", PROGRAM)

        assert result["commanded"] is False

    def test_it_writes_only_into_the_cells_tasks_directory(self, cell_base):
        """No plan, no recording, no log — authoring touches one directory."""
        save_program(CELL, "door_welds", PROGRAM)

        touched = {p.relative_to(cell_base).parts[1]
                   for p in cell_base.rglob("*") if p.is_file()}
        assert touched == {"tasks"}

    def test_the_module_cannot_reach_dispatch_planning_or_ros(self):
        """A writer that could move an arm is a writer nobody can safely call.

        Checked on the import graph rather than by calling it, because the point
        is that the capability is absent, not merely unused on the happy path.
        """
        source = (PROJECT_ROOT / "agents" / "trajectory" / "task_programs.py").read_text()
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        forbidden = [m for m in imported if any(
            bad in m for bad in ("rclpy", "dispatch", "planning_tools", "ompl",
                                 "behaviour", "mcp", "trajectory.agent"))]
        assert forbidden == [], f"the task-program writer must not import {forbidden}"


class TestTheMcpToolIsTheThinLayer:
    """``save_task_program`` must delegate, so the tool and the writer agree.

    Read from source: importing agents/trajectory/mcp_server.py pulls in the LLM
    stack, which is not installed on a plain checkout.
    """

    SOURCE = (PROJECT_ROOT / "agents" / "trajectory" / "mcp_server.py").read_text()

    def test_the_tool_exists_and_is_registered(self):
        tree = ast.parse(self.SOURCE)
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "save_task_program"),
                  None)
        assert fn is not None, "save_task_program is missing from the trajectory MCP server"
        assert any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool"
                   for d in fn.decorator_list), "save_task_program is not an @mcp.tool()"

    def test_the_tool_delegates_to_the_writer(self):
        assert "task_programs.save_program(" in self.SOURCE

    def test_the_tool_does_not_write_the_file_itself(self):
        """Two writers is the failure mode; one of them has to be a pass-through."""
        tree = ast.parse(self.SOURCE)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "save_task_program")
        body = ast.get_source_segment(self.SOURCE, fn) or ""
        assert "json.dump" not in body
        assert "open(" not in body
