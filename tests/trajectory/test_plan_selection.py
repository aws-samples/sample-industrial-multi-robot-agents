"""Backend and mode are chosen per request; the stack only supplies defaults.

Precedence (agents/trajectory/plan_selection.py): the request's argument or its
constraints, else TRAJECTORY_BACKEND / PLANNING_MODE as the stack was started
with, else ompl / classic. A name the REQUEST gives is validated. Host-runnable.
"""
from __future__ import annotations

import pytest

from agents.trajectory import plan_selection as ps

NONE: dict = {}


class TestPrecedence:
    def test_code_defaults(self):
        s = ps.select(env=NONE)
        assert (s["backend"], s["mode"]) == ("ompl", "classic")
        assert (s["backend_from"], s["mode_from"]) == (ps.CODE, ps.CODE)

    def test_the_stack_default_beats_the_code(self):
        s = ps.select(env={"TRAJECTORY_BACKEND": "ompl-compound", "PLANNING_MODE": "coordinated"})
        assert (s["backend"], s["mode"]) == ("ompl-compound", "coordinated")
        assert (s["backend_from"], s["mode_from"]) == (ps.STACK, ps.STACK)

    def test_the_request_beats_the_stack(self):
        env = {"TRAJECTORY_BACKEND": "rrt", "PLANNING_MODE": "stepwise"}
        s = ps.select("ompl-compound", "coordinated", env=env)
        assert (s["backend"], s["mode"]) == ("ompl-compound", "coordinated")
        assert s["backend_from"] == s["mode_from"] == ps.REQUEST

    def test_constraints_are_part_of_the_request(self):
        s = ps.select(constraints={"trajectory_backend": "rrt", "planning_mode": "stepwise"},
                      env={"TRAJECTORY_BACKEND": "ompl-compound"})
        assert (s["backend"], s["mode"], s["backend_from"]) == ("rrt", "stepwise", ps.REQUEST)

    def test_an_empty_stack_value_means_unset(self):
        assert ps.select(env={"PLANNING_MODE": ""})["mode_from"] == ps.CODE

    def test_one_value_can_come_from_each_source(self):
        s = ps.select(mode="stepwise", env={"TRAJECTORY_BACKEND": "ompl-compound"})
        assert (s["backend_from"], s["mode_from"]) == (ps.STACK, ps.REQUEST)


class TestValidation:
    @pytest.mark.parametrize("kw", [{"backend": "moveit"}, {"mode": "fastest"},
                                    {"constraints": {"planning_mode": "x"}}])
    def test_an_unknown_name_in_the_request_is_refused(self, kw):
        with pytest.raises(ps.SelectionError):
            ps.select(env=NONE, **kw)

    def test_two_different_names_in_one_request_are_refused(self):
        with pytest.raises(ps.SelectionError, match="give one"):
            ps.select("ompl", constraints={"trajectory_backend": "rrt"}, env=NONE)

    def test_the_same_name_twice_is_fine(self):
        assert ps.select("rrt", constraints={"trajectory_backend": "RRT"},
                         env=NONE)["backend"] == "rrt"

    def test_a_bad_stack_default_falls_back_to_the_code(self):
        s = ps.select(env={"TRAJECTORY_BACKEND": "moveit", "PLANNING_MODE": "x"})
        assert (s["backend"], s["mode"]) == ("ompl", "classic")

    def test_names_are_normalised(self):
        assert ps.select(" OMPL-Compound ", " Coordinated ", env=NONE)["mode"] == "coordinated"


def test_apply_writes_what_plan_reads():
    s = ps.select("ompl-compound", "coordinated", env=NONE)
    c = ps.apply({"max_parallel": 2}, s)
    assert c == {"max_parallel": 2, "trajectory_backend": "ompl-compound",
                 "planning_mode": "coordinated"}


def test_the_mode_list_has_one_owner():
    src = (ps.__file__.rsplit("/agents/", 1)[0] + "/agents/trajectory/agent.py")
    text = open(src).read()
    assert "PLANNING_MODES = (" not in text
    assert "from agents.trajectory.plan_selection import" in text


def test_the_cli_sends_backend_and_mode_only_when_given():
    import os
    root = ps.__file__.rsplit("/agents/", 1)[0]
    cli = open(os.path.join(root, "scripts/cell_mcp.py")).read()
    assert '"--backend", default=None' in cli and '"--mode", default=None' in cli
    make = open(os.path.join(root, "Makefile")).read()
    assert "--backend $(TRAJECTORY_BACKEND)" not in make
    assert "export PLANNING_MODE" in make
