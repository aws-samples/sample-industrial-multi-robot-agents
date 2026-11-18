"""Every job has its own files, jobs run side by side, an interrupted job resumes.

* A job's status and log are named after it (``<job>.status.json``,
  ``<job>.log``) — no shared ``job.json`` — and written atomically by a
  per-writer temp file.
* Two local jobs run at once, each to Completed, each with its own files.
* A local job whose process is killed — or whose trainer restarts — is
  ``Interrupted`` and ``resumable`` once it has a checkpoint; resuming continues
  THAT job from its checkpoint (``[resume] from step``), in its own directory and
  log, to Completed.
* A SageMaker job resumes as ``<job>-r<n>`` from the same S3 checkpoint prefix,
  with its request otherwise unchanged; every new request carries that prefix.
* Records written before per-job names (``job.json`` / ``train.log``) are
  migrated, not lost.

The local cases run real (tiny) training processes: they need torch.
"""
from __future__ import annotations

import json
import os
import signal
import time

import pytest

from agents.policy_trainer import sagemaker_jobs as sj
from agents.policy_trainer.config import TrainerConfig
from agents.policy_trainer.tracker import Tracker
from tests.backends.roboballet.synthetic import pack

HP = {"warmup": 16, "n_envs": 1, "batch_size": 8, "eval_every": 10_000, "eval_episodes": 1,
      "horizon_s": 1.0, "max_tasks": 2, "arch_node": 8, "arch_edge": 8, "arch_global": 8}


def _wait(t: Tracker, job: str, until, timeout: float = 180.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = t.refresh(job)
        if until(rec):
            return rec
        time.sleep(0.3)
    pytest.fail(f"{job}: {t.get(job)}")


def _start(t: Tracker, job: str, **hp) -> dict:
    p = pack(wrist=True)
    return t.start_local(job, {"syn": p}, {}, {**HP, **hp})


def test_a_jobs_files_are_named_after_it_and_jobs_run_in_parallel(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("LOCAL_TRAINING_MAX_JOBS", "2")
    t = Tracker(tmp_path)
    _start(t, "roboballet-a", total_steps=30)
    _start(t, "roboballet-b", total_steps=30)
    assert t.running_local() == 2
    for job in ("roboballet-a", "roboballet-b"):
        rec = _wait(t, job, lambda r: r["status"] not in ("InProgress", "Stopping"))
        assert rec["status"] == "Completed", rec
        d = t.job_dir(job)
        assert (d / f"{job}.status.json").is_file() and (d / f"{job}.log").is_file()
        assert not (d / "job.json").exists() and not list(d.glob("*.tmp"))
        assert (d / "checkpoint" / "checkpoint.pt").is_file()
        assert json.loads((d / f"{job}.status.json").read_text())["job_name"] == job


def test_a_killed_job_is_interrupted_and_resumes_from_its_checkpoint(tmp_path):
    pytest.importorskip("torch")
    t = Tracker(tmp_path)
    _start(t, "roboballet-k", total_steps=4000, checkpoint_every=20)
    rec = _wait(t, "roboballet-k", lambda r: (r.get("checkpoint_step") or 0) >= 20)
    os.killpg(rec["pid"], signal.SIGKILL)
    rec = _wait(t, "roboballet-k", lambda r: r["status"] != "InProgress")
    assert rec["status"] == "Interrupted" and rec["resumable"], rec
    step = rec["checkpoint_step"]

    # ... and the trainer itself restarts before anyone resumes it
    t2 = Tracker(tmp_path)
    assert t2.get("roboballet-k")["status"] == "Interrupted"
    t2.resume("roboballet-k")
    rec = _wait(t2, "roboballet-k", lambda r: (r.get("checkpoint_step") or 0) > step)
    log = t2.log_file("roboballet-k").read_text()
    assert f"[resume] from step {step}" in log and "===== resumed" in log
    assert rec["resumes"][-1]["checkpoint_step"] == step
    t2.stop("roboballet-k")
    assert _wait(t2, "roboballet-k", lambda r: r["status"] != "Stopping")["status"] == "Stopped"


def test_a_running_local_job_lost_by_a_restart_is_interrupted(tmp_path):
    d = tmp_path / "jobs" / "roboballet-x"
    (d / "checkpoint").mkdir(parents=True)
    (d / "checkpoint" / "checkpoint.pt").write_bytes(b"x")
    (d / "roboballet-x.status.json").write_text(json.dumps(
        {"job_name": "roboballet-x", "target": "local", "status": "InProgress",
         "created": "x", "checkpoint_step": 500}))
    rec = Tracker(tmp_path).get("roboballet-x")
    assert rec["status"] == "Interrupted" and rec["resumable"] and "500" in rec["detail"]


def test_only_resumable_jobs_resume(tmp_path):
    t = Tracker(tmp_path)
    t.track_sagemaker("roboballet-s", train_cells=["a"], eval_cells=[], hyperparameters={},
                      request={"TrainingJobName": "roboballet-s"})
    with pytest.raises(ValueError, match="only"):
        t.resume("roboballet-s")                       # still InProgress
    t.records["roboballet-s"]["status"] = "Stopped"
    with pytest.raises(ValueError, match="no checkpoint"):
        t.resume("roboballet-s", create_sagemaker=lambda r: {})


def test_a_sagemaker_job_resumes_as_a_new_job_on_the_same_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/r")
    monkeypatch.setenv("POLICY_S3_URI", "s3://bkt/robo")
    monkeypatch.setenv("POLICY_TRAINING_TARGET", "sagemaker")
    req = sj.training_request(TrainerConfig.from_env(), "roboballet-s", hyperparameters={},
                              train_cells=["a"], eval_cells=[])
    assert req["CheckpointConfig"] == {"S3Uri": "s3://bkt/robo/jobs/roboballet-s/checkpoints/",
                                       "LocalPath": "/opt/ml/checkpoints"}
    t = Tracker(tmp_path)
    t.track_sagemaker("roboballet-s", train_cells=["a"], eval_cells=[], hyperparameters={},
                      request=req)
    t.records["roboballet-s"]["status"] = "Stopped"
    sent = []
    r1 = t.resume("roboballet-s", create_sagemaker=lambda r: sent.append(r) or
                  {"TrainingJobArn": "arn:1"})
    assert r1["job_name"] == "roboballet-s-r1" and r1["resumed_from"] == "roboballet-s"
    assert sent[0]["CheckpointConfig"] == req["CheckpointConfig"]
    assert sent[0]["HyperParameters"]["resume"] == "1"
    assert sent[0]["InputDataConfig"] == req["InputDataConfig"]
    t.records["roboballet-s-r1"]["status"] = "Failed"
    r2 = t.resume("roboballet-s-r1", create_sagemaker=lambda r: {"TrainingJobArn": "arn:2"})
    assert r2["job_name"] == "roboballet-s-r2" and r2["resumed_from"] == "roboballet-s"


def test_records_from_before_per_job_names_are_migrated(tmp_path):
    d = tmp_path / "jobs" / "roboballet-old"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"job_name": "roboballet-old", "target": "local",
                                            "status": "Completed", "created": "x"}))
    (d / "train.log").write_text("[metric] eval_fraction_solved=0.5 step=10\n")
    t = Tracker(tmp_path)
    assert t.get("roboballet-old")["status"] == "Completed"
    assert (d / "roboballet-old.status.json").is_file() and not (d / "job.json").exists()
    assert t.logs("roboballet-old", 5) == ["[metric] eval_fraction_solved=0.5 step=10"]
