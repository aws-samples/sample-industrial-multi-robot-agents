"""SageMaker training jobs for RoboBallet policies.

Split in two on purpose:

* **pure** — :func:`source_bundle` (the code the job runs, as a tarball) and
  :func:`training_request` (the exact ``CreateTrainingJob`` request). Testable
  without AWS, and what ``start_training(dry_run=True)`` returns;
* **AWS** — :class:`Jobs`, a thin boto3 wrapper that uploads the inputs, creates,
  describes, lists and stops jobs, and downloads the model artefact.

S3 layout under ``POLICY_S3_URI``::

    jobs/<job>/source/sourcedir.tar.gz     the training code (SageMaker script mode)
    jobs/<job>/scenes/train/<cell>.json    scene packs of the training cells
    jobs/<job>/scenes/eval/<cell>.json     scene packs of the evaluation cells
    jobs/<job>/output/<job>/output/model.tar.gz   written by SageMaker
    jobs/<job>/checkpoints/checkpoint.pt   the learner state, synced by SageMaker
                                           from /opt/ml/checkpoints — a resumed job
                                           (<job>-r<n>) reads the SAME prefix

Every upload is server-side encrypted (KMS when ``SAGEMAKER_KMS_KEY_ID`` is
set, else SSE-S3); nothing is ever given a public ACL. The bucket itself is the
operator's and should keep S3 Block Public Access on.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Optional

from agents.policy_trainer.config import NAME_RE, TrainerConfig

REPO = Path(__file__).resolve().parents[2]
CORE_DIR = REPO / "agents/trajectory/backends/roboballet/core"
ENTRY_DIR = REPO / "agents/policy_trainer/sagemaker"
ENTRY_FILES = ("train.py", "networks.py", "procedural.py", "hyperparameters.py")
JOB_PREFIX = "roboballet"

#: Lines train.py prints, as SageMaker metrics.
METRICS = [{"Name": n, "Regex": rf"\[metric\] {n}=([-+0-9.eE]+)"}
           for n in ("eval_fraction_solved", "eval_free_fraction_solved",
                     "train_fraction_solved", "action_abs_mean", "checkpoint_step",
                     "critic_loss", "actor_loss", "steps_per_s")]


def new_job_name(stamp: Optional[float] = None) -> str:
    return f"{JOB_PREFIX}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime(stamp or time.time()))}"


def source_bundle() -> tuple[bytes, str]:
    """``(sourcedir.tar.gz bytes, sha256)``: the entry point at the root and the
    shared core as the top-level package ``roboballet_core``.

    Deterministic (sorted, fixed mtimes and owners), so the digest identifies
    the code a job ran.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        def add(path: Path, arcname: str) -> None:
            data = path.read_bytes()
            info = tarfile.TarInfo(arcname)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
        for name in ENTRY_FILES:
            add(ENTRY_DIR / name, name)
        for path in sorted(CORE_DIR.glob("*.py")):
            add(path, f"roboballet_core/{path.name}")
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def _key(prefix: str, *parts: str) -> str:
    return "/".join(p.strip("/") for p in (prefix, *parts) if p)


def job_uris(cfg: TrainerConfig, job: str) -> dict:
    bucket, prefix = cfg.bucket_and_prefix
    base = _key(prefix, "jobs", job)
    return {"bucket": bucket, "base": base,
            "source_key": _key(base, "source", "sourcedir.tar.gz"),
            "train_prefix": _key(base, "scenes", "train") + "/",
            "eval_prefix": _key(base, "scenes", "eval") + "/",
            "output": f"s3://{bucket}/{_key(base, 'output')}",
            "checkpoints": f"s3://{bucket}/{_key(base, 'checkpoints')}/"}


def training_request(cfg: TrainerConfig, job: str, *, hyperparameters: dict,
                     train_cells: list[str], eval_cells: list[str],
                     instance_type: Optional[str] = None,
                     max_runtime_s: Optional[int] = None, source_sha256: str = "") -> dict:
    """The ``CreateTrainingJob`` request (pure; nothing is sent)."""
    if not NAME_RE.match(job):
        raise ValueError(f"job name {job!r} must match {NAME_RE.pattern}")
    uris = job_uris(cfg, job)
    hp = {str(k): json.dumps(v) if not isinstance(v, str) else v
          for k, v in sorted(hyperparameters.items())}
    # The job runs network-isolated, so the code arrives as an input channel and
    # script mode is pointed at the local copy (as the SageMaker SDK does).
    hp.update({"sagemaker_program": "train.py",
               "sagemaker_submit_directory": "/opt/ml/input/data/code/sourcedir.tar.gz",
               "sagemaker_region": cfg.region, "job-name": job})

    def channel(name: str, prefix: str) -> dict:
        return {"ChannelName": name, "InputMode": "File",
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                                "S3Uri": f"s3://{uris['bucket']}/{prefix}",
                                                "S3DataDistributionType": "FullyReplicated"}}}

    runtime = int(max_runtime_s or cfg.max_runtime_s)
    req = {
        "TrainingJobName": job,
        "RoleArn": cfg.role_arn,
        "AlgorithmSpecification": {"TrainingImage": cfg.image_uri, "TrainingInputMode": "File",
                                   "MetricDefinitions": METRICS},
        "HyperParameters": hp,
        "InputDataConfig": [channel("code", uris["source_key"]),
                            channel("train", uris["train_prefix"])] +
                           ([channel("eval", uris["eval_prefix"])] if eval_cells else []),
        "OutputDataConfig": {"S3OutputPath": uris["output"],
                             **({"KmsKeyId": cfg.kms_key_id} if cfg.kms_key_id else {})},
        "ResourceConfig": {"InstanceType": instance_type or cfg.instance_type,
                           "InstanceCount": 1, "VolumeSizeInGB": 50,
                           **({"VolumeKmsKeyId": cfg.kms_key_id} if cfg.kms_key_id else {})},
        "StoppingCondition": {"MaxRuntimeInSeconds": runtime,
                              **({"MaxWaitTimeInSeconds": runtime * 2} if cfg.use_spot else {})},
        # Checkpoints survive the job: a stopped, failed or spot-interrupted job
        # is resumed by a new one reading this prefix (resume_training_job).
        "CheckpointConfig": {"S3Uri": uris["checkpoints"], "LocalPath": "/opt/ml/checkpoints"},
        "EnableNetworkIsolation": True,
        "EnableInterContainerTrafficEncryption": True,
        "Tags": [{"Key": "app", "Value": "multi_robots_agents"},
                 {"Key": "kind", "Value": "roboballet-policy"},
                 {"Key": "train_cells", "Value": ",".join(train_cells)[:256]},
                 {"Key": "eval_cells", "Value": ",".join(eval_cells)[:256] or "-"},
                 {"Key": "source_sha256", "Value": source_sha256[:64] or "-"}],
    }
    if cfg.use_spot:
        req["EnableManagedSpotTraining"] = True
    return req


class Jobs:
    """boto3 calls, in the configured region. Credentials: the default chain."""

    def __init__(self, cfg: TrainerConfig, session=None):
        import boto3
        self.cfg = cfg
        self.session = session or boto3.session.Session(region_name=cfg.region)
        self.sm = self.session.client("sagemaker")
        self.s3 = self.session.client("s3")

    def check_account(self) -> str:
        """The caller's account; refuses one that is not the pinned ``AWS_ACCOUNT_ID``."""
        account = self.session.client("sts").get_caller_identity()["Account"]
        if self.cfg.account_id and account != self.cfg.account_id:
            raise PermissionError("the credentials in this container belong to another AWS "
                                  "account than the pinned AWS_ACCOUNT_ID — refusing")
        return account

    def _put(self, key: str, body: bytes, content_type: str) -> None:
        extra = ({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self.cfg.kms_key_id}
                 if self.cfg.kms_key_id else {"ServerSideEncryption": "AES256"})
        self.s3.put_object(Bucket=self.cfg.bucket_and_prefix[0], Key=key, Body=body,
                           ContentType=content_type, **extra)

    def upload_inputs(self, job: str, source: bytes, train: dict[str, dict],
                      evals: dict[str, dict]) -> None:
        uris = job_uris(self.cfg, job)
        self._put(uris["source_key"], source, "application/gzip")
        for prefix, packs in ((uris["train_prefix"], train), (uris["eval_prefix"], evals)):
            for cell, pack in packs.items():
                self._put(prefix + f"{cell}.json", json.dumps(pack).encode(), "application/json")

    def create(self, request: dict) -> dict:
        return self.sm.create_training_job(**request)

    def describe(self, job: str) -> dict:
        return self.sm.describe_training_job(TrainingJobName=job)

    def list(self, max_results: int = 20) -> list[dict]:
        out = self.sm.list_training_jobs(NameContains=JOB_PREFIX, SortBy="CreationTime",
                                         SortOrder="Descending", MaxResults=int(max_results))
        return out.get("TrainingJobSummaries", [])

    def stop(self, job: str) -> None:
        self.sm.stop_training_job(TrainingJobName=job)

    def logs(self, job: str, lines: int = 100) -> list[str]:
        """The last ``lines`` lines of the job's CloudWatch log (its training container)."""
        logs = self.session.client("logs")
        group = "/aws/sagemaker/TrainingJobs"
        streams = logs.describe_log_streams(logGroupName=group, logStreamNamePrefix=f"{job}/",
                                            limit=5).get("logStreams", [])
        out: list[str] = []
        for st in streams:
            ev = logs.get_log_events(logGroupName=group, logStreamName=st["logStreamName"],
                                     limit=int(lines), startFromHead=False)
            out += [e["message"] for e in ev.get("events", [])]
        return out[-int(lines):]

    def download_model(self, job: str, max_bytes: int = 512 * 1024 * 1024) -> bytes:
        desc = self.describe(job)
        if desc.get("TrainingJobStatus") != "Completed":
            raise RuntimeError(f"job {job} is {desc.get('TrainingJobStatus')}, not Completed")
        uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
        bucket, _, key = uri[5:].partition("/")
        if bucket != self.cfg.bucket_and_prefix[0]:
            raise PermissionError(f"model artefact {uri} is outside POLICY_S3_URI's bucket")
        head = self.s3.head_object(Bucket=bucket, Key=key)
        if int(head["ContentLength"]) > max_bytes:
            raise ValueError(f"model artefact is {head['ContentLength']} bytes (> {max_bytes})")
        return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def summarize(desc: dict) -> dict:
    """A ``DescribeTrainingJob`` reply, reduced to what an operator acts on."""
    metrics = {m["MetricName"]: m["Value"] for m in desc.get("FinalMetricDataList") or []}
    def ts(k):
        v = desc.get(k)
        return v.isoformat() if hasattr(v, "isoformat") else v
    return {"job_name": desc.get("TrainingJobName"), "status": desc.get("TrainingJobStatus"),
            "secondary_status": desc.get("SecondaryStatus"),
            "failure_reason": desc.get("FailureReason"),
            "created": ts("CreationTime"), "started": ts("TrainingStartTime"),
            "ended": ts("TrainingEndTime"),
            "billable_s": desc.get("BillableTimeInSeconds"),
            "instance_type": (desc.get("ResourceConfig") or {}).get("InstanceType"),
            "model_artifact": (desc.get("ModelArtifacts") or {}).get("S3ModelArtifacts"),
            "metrics": metrics}
