"""The policy trainer: what it sends to SageMaker, and what it refuses to send.

* The ``CreateTrainingJob`` request runs the job network-isolated with the code
  as an input channel, encrypts traffic, tags the cells, and keeps training
  and evaluation cells as separate channels.
* Nothing starts unless training is switched on and fully configured; a role
  outside the pinned account is refused.
* The source bundle is the entry point plus the planner's own core, and is
  deterministic (its digest identifies the code a job ran).
* ``start_training(dry_run=True)`` builds the request and starts nothing.
* A model archive yields only its two policy files — never a path outside the
  registry — and only a policy for this graph schema is installed.

Host-runnable: no AWS, no torch.
"""
from __future__ import annotations

import io
import json
import tarfile

import pytest

from agents.policy_trainer import registry
from agents.policy_trainer import sagemaker_jobs as sj
from agents.policy_trainer.config import TrainerConfig, TrainerConfigError
from agents.trajectory.backends.roboballet.core.gnn import DEFAULT_ARCH, init_params
from agents.trajectory.backends.roboballet.core.policy_io import save
from tests.backends.roboballet.synthetic import pack

ROLE = "arn:aws:iam::123456789012:role/sm-train"


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k, v in {"POLICY_TRAINING_TARGET": "sagemaker", "SAGEMAKER_ROLE_ARN": ROLE,
                 "POLICY_S3_URI": "s3://bkt/robo", "AWS_REGION": "eu-west-1",
                 "AWS_ACCOUNT_ID": "123456789012", "ROBOBALLET_POLICY_DIR": str(tmp_path),
                 "POLICY_TRAINER_STATE": str(tmp_path / "state")}.items():
        monkeypatch.setenv(k, v)
    try:
        import agents.policy_trainer.mcp_server as mcp_server
        monkeypatch.setattr(mcp_server, "_TRACKER", None)       # a fresh tracker per test
    except ImportError:
        pass
    for k in ("SAGEMAKER_REGION", "SAGEMAKER_KMS_KEY_ID", "SAGEMAKER_USE_SPOT",
              "SAGEMAKER_IMAGE_URI"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_the_request_is_isolated_encrypted_and_names_its_cells(env):
    cfg = TrainerConfig.from_env()
    req = sj.training_request(cfg, "roboballet-x", hyperparameters={"total_steps": 10},
                              train_cells=["a"], eval_cells=["b"], source_sha256="f" * 64)
    assert req["EnableNetworkIsolation"] and req["EnableInterContainerTrafficEncryption"]
    chans = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"] for c in req["InputDataConfig"]}
    assert chans == {"code": "s3://bkt/robo/jobs/roboballet-x/source/sourcedir.tar.gz",
                     "train": "s3://bkt/robo/jobs/roboballet-x/scenes/train/",
                     "eval": "s3://bkt/robo/jobs/roboballet-x/scenes/eval/"}
    hp = req["HyperParameters"]
    assert hp["sagemaker_program"] == "train.py" and hp["total_steps"] == "10"
    assert hp["sagemaker_submit_directory"].startswith("/opt/ml/input/data/code/")
    assert req["OutputDataConfig"]["S3OutputPath"] == "s3://bkt/robo/jobs/roboballet-x/output"
    assert "763104351884.dkr.ecr.eu-west-1" in req["AlgorithmSpecification"]["TrainingImage"]
    tags = {t["Key"]: t["Value"] for t in req["Tags"]}
    assert tags["train_cells"] == "a" and tags["eval_cells"] == "b"


def test_no_eval_cells_no_eval_channel(env):
    req = sj.training_request(TrainerConfig.from_env(), "roboballet-y", hyperparameters={},
                              train_cells=["a"], eval_cells=[])
    assert [c["ChannelName"] for c in req["InputDataConfig"]] == ["code", "train"]


@pytest.mark.parametrize("var,value,why", [
    ("POLICY_TRAINING_TARGET", "local", "set it to 'sagemaker'"),
    ("POLICY_TRAINING_TARGET", "cloud", "not one of"),
    ("SAGEMAKER_ROLE_ARN", "", "SAGEMAKER_ROLE_ARN is not set"),
    ("SAGEMAKER_ROLE_ARN", "arn:aws:iam::999999999999:role/x", "pinned AWS_ACCOUNT_ID"),
    ("POLICY_S3_URI", "bkt/robo", "s3://"),
])
def test_nothing_starts_unless_switched_on_and_configured(env, monkeypatch, var, value, why):
    monkeypatch.setenv(var, value)
    with pytest.raises(TrainerConfigError, match=why):
        TrainerConfig.from_env().require(mutating=True)


def test_reading_needs_no_switch(env, monkeypatch):
    monkeypatch.setenv("POLICY_TRAINING_TARGET", "local")
    TrainerConfig.from_env().require(mutating=False)


def test_the_bundle_is_the_entry_point_and_the_shared_core_and_deterministic():
    data, digest = sj.source_bundle()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        names = set(tar.getnames())
    assert {"train.py", "networks.py", "procedural.py", "roboballet_core/graph.py",
            "roboballet_core/gnn.py", "roboballet_core/sim.py"} <= names
    assert sj.source_bundle()[1] == digest


def test_a_dry_run_builds_the_request_and_starts_nothing(env, monkeypatch):
    mcp_server = pytest.importorskip("agents.policy_trainer.mcp_server")
    monkeypatch.setattr(mcp_server, "_pack", lambda c: pack(c))
    monkeypatch.setattr(mcp_server, "_jobs", lambda cfg: pytest.fail("AWS was called"))
    out = mcp_server.start_training(["train_cell"], ["eval_cell"], {"total_steps": 5},
                                    job_name="roboballet-dry", dry_run=True)
    assert out["ok"] and out["dry_run"] and out["request"]["TrainingJobName"] == "roboballet-dry"
    assert set(out["train_cells"]) == {"train_cell"} and set(out["eval_cells"]) == {"eval_cell"}


def test_the_target_is_the_stacks_not_the_requests(env, monkeypatch):
    """With the target local, a request runs locally — it cannot ask for SageMaker."""
    mcp_server = pytest.importorskip("agents.policy_trainer.mcp_server")
    monkeypatch.setenv("POLICY_TRAINING_TARGET", "local")
    monkeypatch.setattr(mcp_server, "_pack", lambda c: pack(c))
    monkeypatch.setattr(mcp_server, "_jobs", lambda cfg: pytest.fail("AWS was called"))
    out = mcp_server.start_training(["a"], dry_run=True)
    assert out["ok"] and out["target"] == "local" and "command" in out
    with pytest.raises(TypeError):
        mcp_server.start_training(["a"], target="sagemaker")


def test_the_default_target_is_local_and_a_bad_one_refuses(env, monkeypatch):
    mcp_server = pytest.importorskip("agents.policy_trainer.mcp_server")
    monkeypatch.delenv("POLICY_TRAINING_TARGET")
    assert TrainerConfig.from_env().target == "local"
    monkeypatch.setenv("POLICY_TRAINING_TARGET", "cloud")
    monkeypatch.setattr(mcp_server, "_pack", lambda c: pack(c))
    out = mcp_server.start_training(["a"], dry_run=True)
    assert not out["ok"] and "local, sagemaker" in out["error"]


def _model_tar(tmp_path, extra: dict | None = None) -> bytes:
    arch = {**DEFAULT_ARCH, "node": 8, "edge": 8, "global": 8, "depth": 2, "head": 8}
    save(tmp_path / "m", init_params(arch), {"arch": arch, "training": {"cells": ["a"]}})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in ("policy.npz", "policy_card.json"):
            tar.add(tmp_path / "m" / f, arcname=f)
            tar.add(tmp_path / "m" / f, arcname=f"best/{f}")
        for name, data in (extra or {}).items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_an_archive_yields_only_its_policy_files(tmp_path):
    data = _model_tar(tmp_path, {"../../evil.py": b"x", "/etc/passwd": b"x"})
    files = registry.extract_policy(data)
    assert set(files) == {"policy.npz", "policy_card.json", "_from"} and files["_from"] == b"best/"
    assert not (tmp_path.parent / "evil.py").exists()


def test_install_validates_and_promote_makes_it_current(tmp_path):
    files = registry.extract_policy(_model_tar(tmp_path))
    files.pop("_from")
    root = tmp_path / "models"
    out = registry.install(str(root), "p1", files, provenance={"job": "j"})
    assert out["name"] == "p1" and out["trained_on"] == ["a"]
    assert registry.promote(str(root), "p1")["promoted_from"] == "p1"
    assert [p["name"] for p in registry.list_policies(str(root))] == ["current", "p1"]
    bad = dict(files)
    card = json.loads(bad["policy_card.json"])
    card["schema_version"] = 99
    bad["policy_card.json"] = json.dumps(card).encode()
    with pytest.raises(ValueError):
        registry.install(str(root), "p2", bad)
    assert not (root / "p2").exists()
    with pytest.raises(ValueError):
        registry.install(str(root), "current", files)
