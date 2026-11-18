"""Training jobs this service started — local or on SageMaker — and their monitor.

One record per job, whatever the target, persisted under the trainer's state
directory (``POLICY_TRAINER_STATE``, the ``trainer-state`` volume) so a restarted
container still knows its jobs. Every file is NAMED after its job, so jobs run
side by side without sharing a file, and any one of them can be found, followed
and resumed on its own::

    <state>/jobs/<job>/<job>.status.json    the record (below), written atomically
    <state>/jobs/<job>/<job>.log            local: the training process's output
    <state>/jobs/<job>/scenes/{train,eval}/<cell>.json   local: its inputs
    <state>/jobs/<job>/checkpoint/checkpoint.pt          local: the learner state
    <state>/jobs/<job>/model/               local: its output (policy + best/)

Record::

    {"job_name", "target": "local" | "sagemaker", "status", "detail",
     "created", "updated", "ended", "train_cells", "eval_cells",
     "hyperparameters", "metrics": {name: last value}, "metric_step",
     "checkpoint_step", "resumable", "resumes": [...], "resumed_from",
     "auto_import", "promote", "imported": {...} | None, "error"}

``status`` uses SageMaker's words for both targets — ``InProgress``,
``Completed``, ``Failed``, ``Stopping``, ``Stopped`` — plus ``Interrupted``: a
local job whose process died under it (the trainer restarted, the process was
killed). A job that stopped before completing is ``resumable`` when it has a
checkpoint; :meth:`Tracker.resume` continues THAT job from its last checkpoint:

* local — the same job, the same directory, ``train.py --resume`` appending to the
  same log;
* SageMaker — a new SageMaker job ``<job>-r<n>`` (SageMaker names are single-use)
  reading the same S3 checkpoint prefix, tracked under its own name with
  ``resumed_from`` pointing back.

**The monitor** (:meth:`Tracker.start_monitor`) is one daemon thread that, every
``POLICY_TRAINER_POLL_S`` seconds, refreshes every unfinished job — local: has
the process exited, the latest ``[metric]`` lines; SageMaker:
``DescribeTrainingJob`` — and, when a job with ``auto_import`` completes,
installs its policy (and promotes it when asked). With
``POLICY_TRAINING_AUTO_RESUME=1`` it also resumes interrupted local jobs.

Local jobs run the SAME ``train.py`` as SageMaker, as subprocesses of this
container, at most ``LOCAL_TRAINING_MAX_JOBS`` (default 2) at a time; each gets
its share of the cores (``TRAINER_THREADS``).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ACTIVE = ("InProgress", "Stopping")
RESUMABLE = ("Interrupted", "Stopped", "Failed")
METRIC_RE = re.compile(r"\[metric\] (\w+)=([-+0-9.eE]+|nan|inf) step=(\d+)")
REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = "checkpoint.pt"
DEFAULT_MAX_LOCAL_JOBS = 2


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def parse_metrics(text: str) -> tuple[dict, int]:
    """Last value of every ``[metric]`` line, and the last step seen."""
    metrics, step = {}, 0
    for m in METRIC_RE.finditer(text):
        metrics[m.group(1)] = float(m.group(2))
        step = max(step, int(m.group(3)))
    return metrics, step


def _tail(path: Path, max_bytes: int = 256 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", "replace")


def max_local_jobs() -> int:
    return max(1, int(_env_float("LOCAL_TRAINING_MAX_JOBS", DEFAULT_MAX_LOCAL_JOBS)))


class Tracker:
    def __init__(self, state_dir: str | Path, *, sagemaker: Optional[Callable] = None,
                 importer: Optional[Callable] = None):
        """``sagemaker()`` → a :class:`~agents.policy_trainer.sagemaker_jobs.Jobs`
        (lazily, only when a SageMaker job needs it); ``importer(record)`` installs
        a completed job's policy and returns its summary."""
        self.root = Path(state_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._sagemaker, self._importer = sagemaker, importer
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._monitor: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.records: dict[str, dict] = {}
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            f = self.status_file(d.name)
            legacy = d / "job.json"                      # records written before per-job names
            src = f if f.is_file() else legacy if legacy.is_file() else None
            if src is None:
                continue
            try:
                rec = json.loads(src.read_text())
            except ValueError:
                continue
            if rec.get("target") == "local" and rec.get("status") in ACTIVE:
                # Its process died with the old container. With a checkpoint, the
                # work up to it is not lost: the job waits to be resumed.
                self._mark_interrupted(rec, "the trainer restarted; the local training "
                                            "process did not survive it")
            self.records[rec["job_name"]] = rec
            if src is legacy:
                old_log = d / "train.log"
                if old_log.is_file() and not self.log_file(d.name).exists():
                    old_log.rename(self.log_file(d.name))
                self._write(rec)
                legacy.unlink(missing_ok=True)

    # ── files ────────────────────────────────────────────────────────────────
    def job_dir(self, job: str) -> Path:
        return self.root / job

    def status_file(self, job: str) -> Path:
        return self.job_dir(job) / f"{job}.status.json"

    def log_file(self, job: str) -> Path:
        return self.job_dir(job) / f"{job}.log"

    def checkpoint_dir(self, job: str) -> Path:
        return self.job_dir(job) / "checkpoint"

    def _write(self, rec: dict) -> None:
        rec["updated"] = _now()
        job = rec["job_name"]
        d = self.job_dir(job)
        d.mkdir(parents=True, exist_ok=True)
        # Unique per writer, so two threads (or two trainers on one volume) never
        # rename each other's half-written file into place.
        tmp = d / f".{job}.status.json.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(rec, indent=2, default=str))
        tmp.replace(self.status_file(job))

    # ── records ──────────────────────────────────────────────────────────────
    def get(self, job: str) -> Optional[dict]:
        with self._lock:
            rec = self.records.get(job)
            return dict(rec) if rec else None

    def list(self) -> list[dict]:
        with self._lock:
            return sorted((dict(r) for r in self.records.values()),
                          key=lambda r: r.get("created", ""), reverse=True)

    def _new(self, job: str, target: str, **fields) -> dict:
        if job in self.records:
            raise ValueError(f"job {job!r} already exists")
        rec = {"job_name": job, "target": target, "status": "InProgress", "detail": "Starting",
               "created": _now(), "ended": None, "metrics": {}, "metric_step": 0,
               "checkpoint_step": None, "resumable": False, "resumes": [],
               "imported": None, "error": None, **fields}
        self.records[job] = rec
        self._write(rec)
        return rec

    def _has_checkpoint(self, rec: dict) -> bool:
        if rec["target"] == "local":
            return (self.checkpoint_dir(rec["job_name"]) / CHECKPOINT).is_file()
        return bool((rec.get("request") or {}).get("CheckpointConfig"))

    def _mark_interrupted(self, rec: dict, why: str) -> None:
        resumable = self._has_checkpoint(rec)
        rec.update(status="Interrupted" if resumable else "Failed", ended=_now(),
                   resumable=resumable,
                   detail=(f"{why}; resume_training_job continues it from step "
                           f"{rec.get('checkpoint_step') or 'its last checkpoint'}"
                           if resumable else f"{why}; no checkpoint was written yet"))

    # ── local ────────────────────────────────────────────────────────────────
    def running_local(self, excluding: Optional[str] = None) -> int:
        return sum(1 for r in self.records.values()
                   if r["target"] == "local" and r["status"] in ACTIVE
                   and r["job_name"] != excluding)

    def local_command(self, job: str, hyperparameters: dict, has_eval: bool,
                      resume: bool = False) -> list[str]:
        d = self.job_dir(job)
        cmd = [sys.executable, "-u", "-m", "agents.policy_trainer.sagemaker.train",
               "--scenes", str(d / "scenes" / "train"), "--model-dir", str(d / "model"),
               "--checkpoint-dir", str(self.checkpoint_dir(job)), "--job-name", job]
        if has_eval:
            cmd += ["--eval-scenes", str(d / "scenes" / "eval")]
        for k, v in sorted(hyperparameters.items()):
            cmd += [f"--{k}", str(v)]
        if resume:
            cmd.append("--resume")
        return cmd

    def _spawn(self, rec: dict, limit: Optional[int], resume: bool) -> None:
        job = rec["job_name"]
        cap = int(limit if limit is not None else max_local_jobs())
        others = self.running_local(excluding=job)      # its own record is not "another job"
        if others >= cap:
            raise RuntimeError(f"{others} local training job(s) already running "
                               f"(LOCAL_TRAINING_MAX_JOBS={cap}) — wait or stop one")
        cmd = self.local_command(job, rec.get("hyperparameters") or {},
                                 bool(rec.get("eval_cells")), resume=resume)
        threads = str(max(1, (os.cpu_count() or 2) // cap))
        env = {**os.environ, "TRAINER_THREADS": threads, "OMP_NUM_THREADS": threads,
               "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        with self.log_file(job).open("ab") as log:
            if resume:
                log.write(f"\n===== resumed {_now()} =====\n".encode())
                log.flush()
            self._procs[job] = subprocess.Popen(cmd, cwd=str(REPO), stdout=log,
                                                stderr=subprocess.STDOUT, env=env,
                                                start_new_session=True)
        rec.update(status="InProgress", detail="Resuming" if resume else "Training",
                   pid=self._procs[job].pid, command=cmd[2:], ended=None, resumable=False,
                   error=None)

    def start_local(self, job: str, train: dict, evals: dict, hyperparameters: dict, *,
                    auto_import: bool = False, promote: bool = False,
                    max_jobs: Optional[int] = None) -> dict:
        with self._lock:
            cap = int(max_jobs if max_jobs is not None else max_local_jobs())
            if self.running_local() >= cap:
                raise RuntimeError(f"{self.running_local()} local training job(s) already "
                                   f"running (LOCAL_TRAINING_MAX_JOBS={cap}) — wait or stop one")
            d = self.job_dir(job)
            for kind, packs in (("train", train), ("eval", evals)):
                (d / "scenes" / kind).mkdir(parents=True, exist_ok=True)
                for cell, pack in packs.items():
                    (d / "scenes" / kind / f"{cell}.json").write_text(json.dumps(pack))
            rec = self._new(job, "local", train_cells=list(train), eval_cells=list(evals),
                            hyperparameters=hyperparameters, auto_import=auto_import,
                            promote=promote, model_dir=str(d / "model"),
                            checkpoint_dir=str(self.checkpoint_dir(job)),
                            log_file=str(self.log_file(job)))
            self._spawn(rec, cap, resume=False)
            self._write(rec)
            return dict(rec)

    def _refresh_local(self, rec: dict) -> None:
        job = rec["job_name"]
        metrics, step = parse_metrics(_tail(self.log_file(job)))
        if metrics:
            rec.update(metrics={**rec.get("metrics", {}), **metrics}, metric_step=step)
            if "checkpoint_step" in metrics:
                rec["checkpoint_step"] = int(metrics["checkpoint_step"])
        proc = self._procs.get(job)
        if proc is None:
            if rec["status"] in ACTIVE:
                # Tracked as running, but no process of ours: it died under us.
                self._mark_interrupted(rec, "its training process is gone")
            return
        code = proc.poll()
        if code is None:
            return
        self._procs.pop(job, None)
        model = Path(rec["model_dir"])
        if rec["status"] == "Stopping":
            rec.update(status="Stopped", detail=f"stopped (exit {code})", ended=_now(),
                       resumable=self._has_checkpoint(rec))
        elif code == 0 and (model / "policy.npz").is_file():
            rec.update(status="Completed", detail="Completed", ended=_now(), resumable=False)
        elif code < 0:
            self._mark_interrupted(rec, f"its training process was killed (signal {-code})")
        else:
            lines = _tail(self.log_file(job), 4096).strip().splitlines()
            rec.update(status="Failed", detail=f"exit code {code}", ended=_now(),
                       error=lines[-1] if lines else None,
                       resumable=self._has_checkpoint(rec))

    def logs(self, job: str, lines: int = 100) -> list[str]:
        rec = self.get(job)
        if rec is None:
            raise KeyError(job)
        n = max(1, min(int(lines), 2000))
        if rec["target"] == "local":
            return _tail(self.log_file(job)).splitlines()[-n:]
        return self._sagemaker().logs(job, n)

    # ── SageMaker ────────────────────────────────────────────────────────────
    def track_sagemaker(self, job: str, *, train_cells: list, eval_cells: list,
                        hyperparameters: dict, auto_import: bool = False,
                        promote: bool = False, arn: Optional[str] = None,
                        request: Optional[dict] = None,
                        resumed_from: Optional[str] = None) -> dict:
        with self._lock:
            return dict(self._new(job, "sagemaker", train_cells=train_cells,
                                  eval_cells=eval_cells, hyperparameters=hyperparameters,
                                  auto_import=auto_import, promote=promote,
                                  training_job_arn=arn, request=request,
                                  resumed_from=resumed_from))

    def _refresh_sagemaker(self, rec: dict) -> None:
        from agents.policy_trainer.sagemaker_jobs import summarize
        s = summarize(self._sagemaker().describe(rec["job_name"]))
        rec.update(status=s["status"] or rec["status"], detail=s["secondary_status"],
                   error=s["failure_reason"], billable_s=s["billable_s"],
                   model_artifact=s["model_artifact"])
        if s["metrics"]:
            rec["metrics"] = s["metrics"]
            if "checkpoint_step" in s["metrics"]:
                rec["checkpoint_step"] = int(s["metrics"]["checkpoint_step"])
        if rec["status"] not in ACTIVE and not rec.get("ended"):
            rec["ended"] = s["ended"] or _now()
        rec["resumable"] = rec["status"] in RESUMABLE and self._has_checkpoint(rec)

    # ── resume ───────────────────────────────────────────────────────────────
    def resume(self, job: str, *, max_jobs: Optional[int] = None,
               create_sagemaker: Optional[Callable[[dict], dict]] = None) -> dict:
        """Continue ``job`` from its last checkpoint (see the module docstring).

        ``create_sagemaker(request) -> CreateTrainingJob reply`` starts the new
        SageMaker job (the caller owns the AWS checks). Returns the record of the
        job that is now running: the same one (local) or ``<job>-r<n>``.
        """
        with self._lock:
            rec = self.records.get(job)
            if rec is None:
                raise KeyError(job)
            if rec["status"] not in RESUMABLE:
                raise ValueError(f"job {job} is {rec['status']} — only "
                                 f"{', '.join(RESUMABLE)} jobs can be resumed")
            if not self._has_checkpoint(rec):
                raise ValueError(f"job {job} has no checkpoint to resume from")
            if rec["target"] == "local":
                prev = {"at": _now(), "from_status": rec["status"],
                        "checkpoint_step": rec.get("checkpoint_step")}
                self._spawn(rec, max_jobs, resume=True)
                rec["resumes"] = list(rec.get("resumes") or []) + [prev]
                self._write(rec)
                return dict(rec)
            if create_sagemaker is None:
                raise RuntimeError("resuming a SageMaker job needs the SageMaker client")
            base = rec.get("resumed_from") or job
            n = 1 + sum(1 for r in self.records.values() if r.get("resumed_from") == base)
            new_job = f"{base}-r{n}"
            req = json.loads(json.dumps(rec["request"]))
            req["TrainingJobName"] = new_job
            req["HyperParameters"] = {**req.get("HyperParameters", {}), "resume": "1",
                                      "job-name": new_job}
            reply = create_sagemaker(req)
            rec["resumes"] = list(rec.get("resumes") or []) + [{"at": _now(), "as": new_job}]
            rec["resumable"] = False
            self._write(rec)
            return self.track_sagemaker(
                new_job, train_cells=rec["train_cells"], eval_cells=rec["eval_cells"],
                hyperparameters=rec["hyperparameters"], auto_import=rec.get("auto_import"),
                promote=rec.get("promote"), arn=reply.get("TrainingJobArn"), request=req,
                resumed_from=base)

    # ── both ─────────────────────────────────────────────────────────────────
    def refresh(self, job: str) -> dict:
        """Bring one record up to date now (and auto-import when it just completed)."""
        with self._lock:
            rec = self.records.get(job)
            if rec is None:
                raise KeyError(job)
            if rec["status"] in ACTIVE:
                try:
                    (self._refresh_local if rec["target"] == "local"
                     else self._refresh_sagemaker)(rec)
                except Exception as e:  # noqa: BLE001 — an unreachable API is not a failed job
                    rec["monitor_error"] = f"{type(e).__name__}: {e}"
                else:
                    rec.pop("monitor_error", None)
            elif rec["target"] == "local":
                self._refresh_local(rec)                 # late metrics after exit
            if rec["status"] == "Completed" and rec.get("auto_import") and not rec.get("imported") \
                    and self._importer is not None:
                try:
                    rec["imported"] = self._importer(rec)
                except Exception as e:  # noqa: BLE001
                    rec["imported"] = {"error": f"{type(e).__name__}: {e}"}
            self._write(rec)
            return dict(rec)

    def stop(self, job: str) -> dict:
        with self._lock:
            rec = self.records.get(job)
            if rec is None:
                raise KeyError(job)
            if rec["status"] not in ACTIVE:
                raise ValueError(f"job {job} is {rec['status']}")
            if rec["target"] == "local":
                proc = self._procs.get(job)
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            else:
                self._sagemaker().stop(job)
            rec.update(status="Stopping", detail="Stopping")
            self._write(rec)
            return dict(rec)

    def poll_once(self) -> None:
        for job in [j for j, r in list(self.records.items()) if r["status"] in ACTIVE
                    or (r["status"] == "Completed" and r.get("auto_import") and not r.get("imported"))]:
            self.refresh(job)
        if os.environ.get("POLICY_TRAINING_AUTO_RESUME", "").strip() in ("1", "true", "yes"):
            for job, r in list(self.records.items()):
                if r["target"] == "local" and r["status"] == "Interrupted" \
                        and self.running_local() < max_local_jobs():
                    try:
                        self.resume(job)
                    except Exception as e:  # noqa: BLE001 — try again next round
                        print(f"[PolicyTrainer] auto-resume {job}: {e}", flush=True)

    def start_monitor(self, interval_s: Optional[float] = None) -> None:
        if self._monitor and self._monitor.is_alive():
            return
        every = float(interval_s or _env_float("POLICY_TRAINER_POLL_S", 30.0))

        def loop():
            while not self._stop.wait(every):
                try:
                    self.poll_once()
                except Exception as e:  # noqa: BLE001 — the monitor never dies
                    print(f"[PolicyTrainer] monitor: {e}", flush=True)

        self._monitor = threading.Thread(target=loop, name="training-monitor", daemon=True)
        self._monitor.start()

    def shutdown(self) -> None:
        self._stop.set()
