"""One training interface, two targets, one monitor.

* With ``POLICY_TRAINING_TARGET=local``, ``start_training`` runs the SAME train.py as a process of the
  trainer; the tracker follows it to Completed, parses its metrics, serves its
  log, and ``import_policy`` installs its policy — the planner loads it.
* SageMaker jobs are tracked the same way: the monitor refreshes
  status and metrics from DescribeTrainingJob and, with ``auto_import``,
  installs the policy when the job completes.
* A local job is stoppable; a local job that was running when the trainer
  restarted is marked Failed, not left "InProgress" forever.
* Unknown hyperparameters refuse — for both targets — before anything runs.
* ``POLICY_TRAINING_TARGET`` alone decides where; the dry run shows the local command.
"""
from __future__ import annotations

import json
import sys
import time

import pytest

from agents.policy_trainer.sagemaker.hyperparameters import check as hp_check
from agents.policy_trainer.tracker import Tracker, parse_metrics
from tests.backends.roboballet.synthetic import pack

TINY = {"total_steps": 12, "warmup": 8, "n_envs": 1, "batch_size": 8, "eval_every": 100,
        "eval_episodes": 1, "eval_max_tasks": 1, "horizon_s": 0.5, "max_tasks": 2,
        "arch_node": 8, "arch_edge": 8, "arch_global": 8}


@pytest.fixture
def server(monkeypatch, tmp_path):
    mcp_server = pytest.importorskip("agents.policy_trainer.mcp_server")
    monkeypatch.setenv("POLICY_TRAINER_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("ROBOBALLET_POLICY_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("POLICY_TRAINING_TARGET", "local")
    monkeypatch.setattr(mcp_server, "_TRACKER", None)
    monkeypatch.setattr(mcp_server, "_pack", lambda c: pack(c, wrist=True))
    return mcp_server


def _wait(server, job, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = server.get_training_job(job)
        if rec["status"] not in ("InProgress", "Stopping"):
            return rec
        time.sleep(0.5)
    pytest.fail(f"{job} still {rec['status']} after {timeout}s")


def test_hyperparameters_are_checked_and_typed():
    assert hp_check({"total_steps": "10", "arch_node": 16.0}) == {"total_steps": 10, "arch_node": 16}
    with pytest.raises(ValueError, match="unknown"):
        hp_check({"total_step": 10})
    with pytest.raises(ValueError, match="not a int"):
        hp_check({"total_steps": "many"})


def test_metrics_are_read_from_the_log():
    m, step = parse_metrics("x\n[metric] eval_fraction_solved=0.25 step=10\n"
                            "[metric] eval_fraction_solved=0.5 step=20\n[metric] critic_loss=1e-3 step=20")
    assert m == {"eval_fraction_solved": 0.5, "critic_loss": 0.001} and step == 20


def test_a_local_job_trains_is_monitored_and_imports(server):
    pytest.importorskip("torch")
    out = server.start_training(["train_cell"], ["eval_cell"], TINY,
                                job_name="roboballet-local1")
    assert out["ok"] and out["target"] == "local", out
    rec = _wait(server, "roboballet-local1")
    assert rec["status"] == "Completed", rec
    assert "eval_fraction_solved" in rec["metrics"]
    logs = server.get_training_logs("roboballet-local1", 5)
    assert logs["ok"] and any("eval_fraction_solved" in ln or "updates" in ln for ln in logs["lines"])
    listed = server.list_training_jobs()["jobs"]
    assert listed[0]["job_name"] == "roboballet-local1" and listed[0]["target"] == "local"
    imp = server.import_policy("roboballet-local1", "p-local", promote=True)
    assert imp["ok"], imp
    assert imp["policy"]["provenance"]["target"] == "local"
    from agents.trajectory.backends.roboballet.planner import load_policy
    import os
    os.environ["ROBOBALLET_POLICY_DIR"] = server._cfg().policy_dir
    assert load_policy("current").card["promoted_from"] == "p-local"


def test_a_local_job_can_be_stopped(server):
    pytest.importorskip("torch")
    out = server.start_training(["c"], None, {**TINY, "total_steps": 100000},
                                job_name="roboballet-long")
    assert out["ok"], out
    time.sleep(1.0)
    assert server.stop_training_job("roboballet-long")["status"] == "Stopping"
    assert _wait(server, "roboballet-long")["status"] == "Stopped"


def test_local_jobs_are_capped(server, monkeypatch):
    monkeypatch.setenv("LOCAL_TRAINING_MAX_JOBS", "2")
    t = server._tracker()
    monkeypatch.setattr(t, "running_local", lambda: 2)
    out = server.start_training(["c"], None, TINY, job_name="roboballet-2nd")
    assert not out["ok"] and "already running" in out["error"]


def test_the_dry_run_shows_the_local_command(server):
    dry = server.start_training(["c"], ["e"], TINY, dry_run=True, job_name="roboballet-dry")
    assert dry["ok"] and dry["target"] == "local"
    assert "agents.policy_trainer.sagemaker.train" in dry["command"]
    assert "--eval-scenes" in dry["command"] and "--total_steps" in dry["command"]


def test_unknown_hyperparameters_refuse_for_both_targets(server, monkeypatch):
    for target in ("local", "sagemaker"):
        monkeypatch.setenv("POLICY_TRAINING_TARGET", target)
        out = server.start_training(["c"], None, {"totl_steps": 1}, dry_run=True)
        assert not out["ok"] and "unknown hyperparameters" in out["error"], target


class FakeSageMaker:
    def __init__(self):
        self.status, self.stopped = "InProgress", False

    def describe(self, job):
        return {"TrainingJobName": job, "TrainingJobStatus": self.status,
                "SecondaryStatus": "Training" if self.status == "InProgress" else self.status,
                "FinalMetricDataList": [{"MetricName": "eval_fraction_solved", "Value": 0.75}],
                "ModelArtifacts": {"S3ModelArtifacts": "s3://b/p/model.tar.gz"}}

    def stop(self, job):
        self.stopped = True

    def logs(self, job, n):
        return [f"{job} line"] * n


def test_the_monitor_follows_a_sagemaker_job_and_auto_imports(tmp_path):
    sm, imported = FakeSageMaker(), []
    t = Tracker(tmp_path, sagemaker=lambda: sm,
                importer=lambda rec: imported.append(rec["job_name"]) or {"name": rec["job_name"]})
    t.track_sagemaker("roboballet-sm", train_cells=["a"], eval_cells=["b"],
                      hyperparameters={}, auto_import=True)
    t.poll_once()
    assert t.get("roboballet-sm")["status"] == "InProgress" and not imported
    sm.status = "Completed"
    t.start_monitor(interval_s=0.05)
    for _ in range(100):
        if imported:
            break
        time.sleep(0.05)
    t.shutdown()
    rec = t.get("roboballet-sm")
    assert rec["status"] == "Completed" and rec["metrics"]["eval_fraction_solved"] == 0.75
    assert imported == ["roboballet-sm"] and rec["imported"] == {"name": "roboballet-sm"}
    assert t.logs("roboballet-sm", 2) == ["roboballet-sm line"] * 2
    # persisted: a new tracker on the same state knows the job
    assert Tracker(tmp_path).get("roboballet-sm")["status"] == "Completed"


def test_an_unreachable_api_is_not_a_failed_job(tmp_path):
    def broken():
        raise ConnectionError("no route")
    t = Tracker(tmp_path, sagemaker=broken)
    t.track_sagemaker("roboballet-x", train_cells=["a"], eval_cells=[], hyperparameters={})
    rec = t.refresh("roboballet-x")
    assert rec["status"] == "InProgress" and "no route" in rec["monitor_error"]


def test_a_local_job_orphaned_by_a_restart_is_failed(tmp_path):
    d = tmp_path / "jobs" / "roboballet-orphan"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"job_name": "roboballet-orphan", "target": "local",
                                            "status": "InProgress", "created": "x"}))
    rec = Tracker(tmp_path).get("roboballet-orphan")
    assert rec["status"] == "Failed" and "restarted" in rec["detail"]


def test_the_local_command_is_the_sagemaker_entry_point(tmp_path):
    cmd = Tracker(tmp_path).local_command("roboballet-c", {"total_steps": 5}, has_eval=False)
    assert cmd[0] == sys.executable and cmd[3] == "agents.policy_trainer.sagemaker.train"
    assert "--eval-scenes" not in cmd and cmd[-2:] == ["--total_steps", "5"]
