"""The policy trainer is its own container, enabled from make.

* The generated compose file has a ``policy-trainer`` service behind the
  ``training`` profile — a plain ``make sim`` does not start it — published on
  this machine only, with training switched OFF unless the operator says so.
* It writes one thing, the ``policy-models`` volume, which the planner mounts
  read-only; its cells mount is read-only and its cell data comes from the
  orchestrator, like the planner's.
* ``make trainer`` starts it alone; ``make sim TRAINER=1`` (full stack or
  ``ONLY=trajectory``) starts it with the stack; the execution side refuses it.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = "${PUBLISH_HOST:-127.0.0.1}"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_compose_trainer",
                                                  ROOT / "scripts/generate_compose.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"generator not importable here: {exc}")
    return mod


def test_the_trainer_is_profile_gated_local_and_switched_off(gen):
    svc = gen.policy_trainer_service("c1", {"CELL_BASE_PATH": "/cells"})
    assert svc["profiles"] == ["training"]
    assert svc["ports"] == [f"{LOCAL}:8894:8894"]
    env = svc["environment"]
    assert env["POLICY_TRAINING_TARGET"] == "${POLICY_TRAINING_TARGET:-local}"   # never paid by default
    assert not {"POLICY_TRAINING_SAGEMAKER", "POLICY_TRAINING_LOCAL"} & set(env)
    assert env["CELL_DATA_MODE"] == "orchestrator"
    assert "agents/policy_trainer/mcp_server.py" in svc["command"][0]
    assert "./cells:/cells:ro" in svc["volumes"] and "policy-models:/models" in svc["volumes"]
    assert not any(v.startswith("./cells:/cells") and not v.endswith(":ro") for v in svc["volumes"])


def test_the_planner_reads_the_policies_read_only(gen):
    svc = gen.trajectory_agent_service("c1", "ompl", {}, {})
    assert "policy-models:/models:ro" in svc["volumes"]
    assert svc["environment"]["ROBOBALLET_POLICY_DIR"] == "/models"


def _make(*args):
    if shutil.which("make") is None:
        pytest.skip("make not available")
    r = subprocess.run(["make", "-n", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _up(out):
    return [ln for ln in out.splitlines() if ln.startswith("docker compose") and " up " in ln]


def test_make_trainer_starts_it_alone():
    code, out = _make("trainer", "SIM=welding_cell_2_small")
    assert code == 0, out
    (line,) = _up(out)
    assert "--profile training" in line and line.rstrip().endswith("policy-trainer")


def test_a_plain_sim_leaves_it_off_and_trainer_1_turns_it_on():
    code, out = _make("sim", "SIM=welding_cell_2_small")
    assert code == 0 and "--profile training" not in _up(out)[0]
    code, out = _make("sim", "SIM=welding_cell_2_small", "TRAINER=1")
    assert code == 0 and "--profile training" in _up(out)[0]
    code, out = _make("sim", "SIM=welding_cell_2_small", "ONLY=trajectory", "TRAINER=1")
    assert code == 0
    assert _up(out)[0].rstrip().endswith("trajectory-agent orchestrator policy-trainer")


def test_the_execution_side_refuses_it():
    code, out = _make("sim", "SIM=welding_cell_2_small", "ONLY=execution", "TRAINER=1")
    assert code != 0 and "planning side" in out


def test_down_removes_it_with_the_stack():
    code, out = _make("down", "SIM=welding_cell_2_small")
    assert code == 0 and "docker rm -f policy-trainer" in out


def test_the_trainer_has_its_own_image_and_state(gen):
    svc = gen.policy_trainer_service("c1", {"CELL_BASE_PATH": "/cells"})
    assert svc["build"] == {"context": "./docker", "dockerfile": "Dockerfile.trainer",
                            "args": {"TORCH_INDEX": "${TRAINER_TORCH_INDEX:-cpu}"}}
    assert svc["image"] == "${TRAINER_IMAGE:-multi-robots-agents/policy-trainer:latest}"
    assert "trainer-state:/state" in svc["volumes"]
    df = (ROOT / "docker/Dockerfile.trainer").read_text()
    assert "torch" in df and "FROM ros" not in df and "USER trainer" in df
    assert "curl" not in " ".join(svc["healthcheck"]["test"])      # slim image: no curl


def test_the_trainer_image_builds_without_a_cell():
    code, out = _make("trainer-build")
    assert code == 0, out
    (line,) = [ln for ln in out.splitlines() if ln.startswith("docker build")]
    assert "-f docker/Dockerfile.trainer" in line and line.rstrip().endswith("docker/")
    assert "-t multi-robots-agents/policy-trainer:latest" in line
    assert "generate_compose" not in out                      # no compose file involved


def test_training_is_driven_from_make():
    code, out = _make("train", "SIM=welding_cell_2_small", "EVAL_CELLS=welding_cell_2_small",
                      "STEPS=2000", "JOB=roboballet-t1", "WATCH=1")
    assert code == 0, out
    assert "scripts/trainer_mcp.py" in out
    for part in ('start --train "welding_cell_2_small"', '--eval "welding_cell_2_small"',
                 "--steps 2000", "--job roboballet-t1", "--watch"):
        assert part in out, part


@pytest.mark.parametrize("target,needs", [("train-watch", "JOB"), ("train-logs", "JOB"),
                                          ("train-stop", "JOB"), ("policy-import", "JOB"),
                                          ("policy-eval", "POLICY"), ("policy-promote", "POLICY")])
def test_job_and_policy_targets_say_what_they_need(target, needs):
    if shutil.which("make") is None:
        pytest.skip("make not available")
    r = subprocess.run(["make", target, "TOOLS="], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and f"{needs}= is required" in r.stdout + r.stderr


def test_planners_scale_each_with_its_own_name_scratch_and_registration(gen):
    one = gen.trajectory_agent_service("c1", "ompl", {}, {})
    three = gen.trajectory_agent_service("c1", "ompl", {}, {}, index=3)
    assert one["container_name"] == "trajectory-agent" and one["ports"]
    assert three["container_name"] == "trajectory-agent-3" and "ports" not in three
    assert "planner-scratch-3:/scratch" in three["volumes"]
    env = three["environment"]
    assert env["PLANNER_ID"] == "trajectory-agent-3"
    assert env["PLANNER_ADVERTISE_URL"] == "http://trajectory-agent-3:8890/mcp"
    assert env["ORCHESTRATOR_URL"] == "http://orchestrator:8889"


def test_make_starts_the_extra_planners_with_the_planning_side():
    code, out = _make("sim", "SIM=welding_cell_2_small", "ONLY=trajectory", "PLANNERS=3")
    assert code == 0
    assert _up(out)[0].rstrip().endswith(
        "up --no-build trajectory-agent trajectory-agent-2 trajectory-agent-3 orchestrator")
