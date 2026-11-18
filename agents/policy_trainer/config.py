"""The trainer's configuration — read from the environment, checked before use.

ONE switch decides where training runs: ``POLICY_TRAINING_TARGET``.

* ``local`` (the default) — the same ``train.py`` as a process of this
  container (CPU torch by default, see ``docker/Dockerfile.trainer``). No money,
  no AWS account.
* ``sagemaker`` — a SageMaker PyTorch job. It costs money and writes to an AWS
  account, so choosing it is the operator's explicit opt-in, and a job still
  starts only when ``SAGEMAKER_ROLE_ARN`` and ``POLICY_S3_URI`` are set and, when
  ``AWS_ACCOUNT_ID`` pins an account, both the role ARN and the caller's identity
  (STS) are in it — a job never lands in an account nobody chose.

Any other value is refused. Everything that only reads (status, listing, logs,
evaluation, export) works whatever the target, and so does stopping a job this
trainer already started.

==========================  =========================================  ==========
variable                    meaning                                    default
==========================  =========================================  ==========
POLICY_TRAINING_TARGET      where jobs run: ``local`` | ``sagemaker``  local
SAGEMAKER_ROLE_ARN          execution role the job runs as             —
POLICY_S3_URI               ``s3://bucket/prefix`` for inputs/outputs  —
AWS_REGION                  region of the jobs and the bucket          eu-west-1
AWS_ACCOUNT_ID              pinned account (optional, recommended)     —
SAGEMAKER_IMAGE_URI         PyTorch training image                     AWS DLC
SAGEMAKER_INSTANCE_TYPE     training instance                          ml.g5.2xlarge
SAGEMAKER_MAX_RUNTIME_S     stopping condition                         86400
SAGEMAKER_KMS_KEY_ID        KMS key for outputs and uploads            SSE-S3
SAGEMAKER_USE_SPOT          ``1`` = managed spot training              off
ROBOBALLET_POLICY_DIR       where imported policies are kept           /models
POLICY_TRAINER_STATE        job records, local logs and outputs        /state
LOCAL_TRAINING_MAX_JOBS     local jobs at a time (cores shared)        2
POLICY_TRAINING_AUTO_RESUME ``1``: resume interrupted local jobs       off
POLICY_TRAINER_POLL_S       monitor period, seconds                    30
==========================  =========================================  ==========
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

#: AWS Deep Learning Container for PyTorch training (script mode). The account
#: hosting DLC images is the same in most commercial regions; override with
#: SAGEMAKER_IMAGE_URI where it is not, or to pin another framework version.
DLC_TEMPLATE = ("763104351884.dkr.ecr.{region}.amazonaws.com/"
                "pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker")

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")       # SageMaker job names
POLICY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ARN_ACCOUNT_RE = re.compile(r"^arn:aws[a-z-]*:iam::(\d{12}):role/.+$")


TARGETS = ("local", "sagemaker")
DEFAULT_TARGET = "local"


class TrainerConfigError(RuntimeError):
    """Training is not configured (or not allowed) — says which part, never a secret."""


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class TrainerConfig:
    target: str
    region: str
    role_arn: str
    s3_uri: str
    account_id: str
    image_uri: str
    instance_type: str
    max_runtime_s: int
    kms_key_id: str
    use_spot: bool
    policy_dir: str
    state_dir: str = "/state"

    @classmethod
    def from_env(cls) -> "TrainerConfig":
        region = os.environ.get("SAGEMAKER_REGION") or os.environ.get("AWS_REGION") or "eu-west-1"
        try:
            runtime = int(os.environ.get("SAGEMAKER_MAX_RUNTIME_S", "86400"))
        except ValueError:
            runtime = 86400
        target = (os.environ.get("POLICY_TRAINING_TARGET") or DEFAULT_TARGET).strip().lower()
        return cls(target=target, region=region,
                   role_arn=os.environ.get("SAGEMAKER_ROLE_ARN", "").strip(),
                   s3_uri=os.environ.get("POLICY_S3_URI", "").strip().rstrip("/"),
                   account_id=os.environ.get("AWS_ACCOUNT_ID", "").strip(),
                   image_uri=(os.environ.get("SAGEMAKER_IMAGE_URI", "").strip()
                              or DLC_TEMPLATE.format(region=region)),
                   instance_type=os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.g5.2xlarge"),
                   max_runtime_s=runtime,
                   kms_key_id=os.environ.get("SAGEMAKER_KMS_KEY_ID", "").strip(),
                   use_spot=_flag("SAGEMAKER_USE_SPOT"),
                   policy_dir=os.environ.get("ROBOBALLET_POLICY_DIR", "/models"),
                   state_dir=os.environ.get("POLICY_TRAINER_STATE", "/state"))

    @property
    def bucket_and_prefix(self) -> tuple[str, str]:
        if not self.s3_uri.startswith("s3://"):
            raise TrainerConfigError(f"POLICY_S3_URI must be s3://bucket/prefix, got {self.s3_uri!r}")
        bucket, _, prefix = self.s3_uri[5:].partition("/")
        if not bucket:
            raise TrainerConfigError("POLICY_S3_URI names no bucket")
        return bucket, prefix.strip("/")

    def require_target(self) -> str:
        """The configured target, or a refusal naming the valid ones."""
        if self.target not in TARGETS:
            raise TrainerConfigError(f"POLICY_TRAINING_TARGET={self.target!r} — use one of "
                                     f"{', '.join(TARGETS)}")
        return self.target

    def problems(self, *, mutating: bool) -> list[str]:
        """What stands between this configuration and a SageMaker call (empty = ready).

        ``mutating``: starting a job, which also needs the target to be ``sagemaker``.
        """
        out = []
        if self.target not in TARGETS:
            out.append(f"POLICY_TRAINING_TARGET={self.target!r} is not one of {', '.join(TARGETS)}")
        elif mutating and self.target != "sagemaker":
            out.append(f"POLICY_TRAINING_TARGET is {self.target!r} — set it to 'sagemaker' "
                       "to train on SageMaker")
        if not self.role_arn:
            out.append("SAGEMAKER_ROLE_ARN is not set")
        elif not ARN_ACCOUNT_RE.match(self.role_arn):
            out.append("SAGEMAKER_ROLE_ARN is not an IAM role ARN")
        elif self.account_id and ARN_ACCOUNT_RE.match(self.role_arn).group(1) != self.account_id:
            out.append("SAGEMAKER_ROLE_ARN is not in the pinned AWS_ACCOUNT_ID")
        if not self.s3_uri:
            out.append("POLICY_S3_URI is not set")
        else:
            try:
                self.bucket_and_prefix
            except TrainerConfigError as e:
                out.append(str(e))
        if self.account_id and not re.fullmatch(r"\d{12}", self.account_id):
            out.append("AWS_ACCOUNT_ID is not a 12-digit account id")
        return out

    def require(self, *, mutating: bool) -> None:
        problems = self.problems(mutating=mutating)
        if problems:
            raise TrainerConfigError("; ".join(problems))

    def describe(self) -> dict:
        """For status replies: what is configured, never a credential."""
        return {"target": self.target, "region": self.region, "role_arn": self.role_arn or None,
                "s3_uri": self.s3_uri or None, "account_id": self.account_id or None,
                "image_uri": self.image_uri, "instance_type": self.instance_type,
                "max_runtime_s": self.max_runtime_s, "kms": bool(self.kms_key_id),
                "spot": self.use_spot, "policy_dir": self.policy_dir,
                "state_dir": self.state_dir,
                "sagemaker_problems": (self.problems(mutating=True)
                                       if self.target == "sagemaker" else None)}


def valid_policy_name(name: Optional[str]) -> str:
    if not name or not POLICY_NAME_RE.match(name) or name in ("current",):
        raise ValueError(f"policy name {name!r} must match {POLICY_NAME_RE.pattern} "
                         "and not be 'current'")
    return name
