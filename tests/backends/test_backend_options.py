"""Per-request backend options — one field for every backend.

* A backend declares the options it takes (``BackendSpec.options``); a request
  carries them as ``backend_options``. An option the chosen backend does not
  declare, a non-scalar value, or a value its own check refuses (``roboballet``:
  a policy that is not installed) refuses the REQUEST — nothing is ignored.
* Selection takes them from the request argument or its constraints (not both,
  differently), validated against the backend that was chosen, and writes them
  into the constraints the run reads.
* The facade hands a run's options to the backend as keyword arguments and
  stamps them on the result; a keyword the platform passes itself wins.
* ``roboballet`` plans with the policy the request names, not ``current``.
* The planning CLI (``make plan BACKEND_OPTIONS=`` / ``POLICY=``) sends them.

Host-runnable.
"""
from __future__ import annotations

import sys

import pytest

from agents.trajectory import plan_selection as ps
from agents.trajectory.backends.contract import BackendSpec
from agents.trajectory.backends.roboballet import planner as rp
from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH, init_params
from agents.trajectory.backends.roboballet.core.policy_io import save
from agents.trajectory.backends.roboballet.core.geometry import BoxSet
from agents.trajectory.tools import planner_backend as pb

SMALL = {**DEFAULT_ARCH, "node": 8, "edge": 8, "global": 8, "depth": 2, "head": 8}
NONE = {}


@pytest.fixture
def policies(tmp_path, monkeypatch):
    for name in ("current", "p2"):
        save(tmp_path / name, init_params(SMALL), {"arch": SMALL, "name": name,
                                                   "training": {"job": f"job-{name}"}})
    monkeypatch.setenv("ROBOBALLET_POLICY_DIR", str(tmp_path))
    return tmp_path


def test_a_backend_lists_the_options_it_takes():
    details = {d["name"]: d for d in pb.available_backends()}
    assert details["roboballet"]["options"] == ["policy"]
    assert details["ompl"]["options"] == []


@pytest.mark.parametrize("backend,options,why", [
    ("ompl", {"policy": "p2"}, "takes no option"),
    ("roboballet", {"polcy": "p2"}, r"it accepts \['policy'\]"),
    ("roboballet", {"policy": ["p2"]}, "string, number or boolean"),
    ("roboballet", {"policy": "missing"}, "no policy 'missing'"),
])
def test_what_a_backend_does_not_take_refuses(policies, backend, options, why):
    with pytest.raises(ValueError, match=why):
        pb.validate_backend_options(backend, options)


def test_an_installed_policy_is_accepted(policies):
    assert pb.validate_backend_options("roboballet", {"policy": "p2"}) == {"policy": "p2"}
    assert pb.validate_backend_options("roboballet", None) == {}


def test_selection_takes_them_from_the_request_and_writes_them_into_the_run(policies):
    s = ps.select("roboballet", None, None, env=NONE, backend_options={"policy": "p2"})
    assert s["backend_options"] == {"policy": "p2"}
    c = ps.apply({"max_parallel": 2}, s)
    assert c["backend_options"] == {"policy": "p2"} and c["trajectory_backend"] == "roboballet"
    # from the constraints (a task program), and the same again is not a conflict
    s2 = ps.select(None, None, {"trajectory_backend": "roboballet",
                                "backend_options": {"policy": "p2"}}, env=NONE,
                   backend_options={"policy": "p2"})
    assert s2["backend_options"] == {"policy": "p2"}
    with pytest.raises(ps.SelectionError, match="give one"):
        ps.select("roboballet", None, {"backend_options": {"policy": "current"}}, env=NONE,
                  backend_options={"policy": "p2"})
    with pytest.raises(ps.SelectionError, match="takes no option"):
        ps.select("ompl", None, None, env=NONE, backend_options={"policy": "p2"})
    plain = ps.select("ompl", None, None, env=NONE)
    assert "backend_options" not in plain
    assert "backend_options" not in ps.apply({"backend_options": {}}, plain)


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(pb, "_SPECS", dict(pb._SPECS))
    monkeypatch.setattr(pb, "_BACKENDS", dict(pb._BACKENDS))
    monkeypatch.setattr(pb, "VALID_BACKENDS", tuple(pb.VALID_BACKENDS))
    return pb


def test_the_facade_passes_a_runs_options_to_the_backend(registry, tmp_path, monkeypatch):
    (tmp_path / "opt_backend_mod.py").write_text(
        "SEEN = {}\n"
        "def plan(cell_id, reqs, approach_offset_m=0.1, velocity_scale=None, **kw):\n"
        "    SEEN.update(kw)\n"
        "    return {'success': True, 'trajectories': {}, 'failures': [],\n"
        "            'planning_time_s': 0.0, 'backend': 'knobbed'}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("opt_backend_mod", None)
    registry.register_backend(BackendSpec(name="knobbed", entry="opt_backend_mod:plan",
                                          options=("knob", "max_iterations")))
    opts = registry.validate_backend_options("knobbed", {"knob": 3, "max_iterations": 1})
    b_tok, o_tok = registry.set_planning_backend("knobbed"), registry.set_backend_options(opts)
    try:
        out = registry.dispatch_plan("c", [{"robot_ns": "robot1"}], max_iterations=99)
    finally:
        registry.reset_backend_options(o_tok)
        registry.reset_planning_backend(b_tok)
    import opt_backend_mod
    assert opt_backend_mod.SEEN["knob"] == 3
    assert opt_backend_mod.SEEN["max_iterations"] == 99           # the platform's wins
    assert out["backend_options"] == {"knob": 3, "max_iterations": 1}
    assert registry.current_backend_options() == {}               # unbound after the run


def test_roboballet_plans_with_the_requested_policy(policies, monkeypatch):
    monkeypatch.setattr(rp, "_cell_model", lambda cell, ids: ({}, {}, BoxSet([])))
    import agents.trajectory.backends.ompl_compound.planner as cp
    monkeypatch.setattr(cp, "plan_trajectories_ompl_compound",
                        lambda cell_id, reqs, **kw: {"success": True, "trajectories": {},
                                                     "failures": [], "planning_time_s": 0.0,
                                                     "backend": kw["backend_name"],
                                                     "planner": kw["planner_name"]})
    token = pb.set_planning_backend("roboballet")
    o_tok = pb.set_backend_options({"policy": "p2"})
    try:
        out = pb.dispatch_plan("c", [{"robot_ns": "robot1", "operation_id": "W1"}])
    finally:
        pb.reset_backend_options(o_tok)
        pb.reset_planning_backend(token)
    assert out["policy"]["name"] == "p2" and out["policy"]["job"] == "job-p2"
    assert out["planner"] == "policy:p2" and out["backend_options"] == {"policy": "p2"}
    assert rp.plan("c", [{"robot_ns": "robot1"}])["policy"]["name"] == "current"


def test_the_planning_cli_sends_them(monkeypatch):
    sys.path.insert(0, "scripts")
    import cell_mcp
    sent = {}

    def fake_call(name, args, *a, **k):
        sent[name] = args
        return {"accepted": False, "error": "stop here"}
    monkeypatch.setattr(cell_mcp, "call_tool", fake_call)
    monkeypatch.setattr(cell_mcp, "_load_operations", lambda cell, task: [{"id": "W1"}])
    monkeypatch.setattr(sys, "argv", ["cell_mcp.py", "run-task", "--cell", "c", "--task", "t",
                                      "--backend", "roboballet",
                                      "--backend-options", '{"policy": "p2"}', "--plan-only"])
    assert cell_mcp.main() == 1                       # refused by the fake planner
    assert sent["plan_trajectories"]["backend"] == "roboballet"
    assert sent["plan_trajectories"]["backend_options"] == {"policy": "p2"}
