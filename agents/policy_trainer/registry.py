"""Trained policies on the shared ``/models`` volume.

The trainer is the only writer (``policy-models`` is mounted read-write here
and read-only in the trajectory agent). Layout::

    /models/<name>/policy.npz + policy_card.json    an imported policy
    /models/current/                                 the promoted one (a copy)

A policy enters only through :func:`install`, which validates its card against
this build's graph schema, so the planner never finds a file it cannot run.
"""
from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

from agents.policy_trainer.config import valid_policy_name
from agents.trajectory.backends.roboballet.core.policy_io import CARD, WEIGHTS, Policy

CURRENT = "current"
MAX_MEMBER_BYTES = 256 * 1024 * 1024


def _root(policy_dir: str) -> Path:
    return Path(policy_dir)


def extract_policy(model_tar_gz: bytes, *, use_best: bool = True) -> dict[str, bytes]:
    """``{policy.npz, policy_card.json}`` from a SageMaker ``model.tar.gz``.

    Only those two regular files are read — from ``best/`` when asked and
    present, else the root — and nothing is ever extracted to disk by path,
    so a crafted archive cannot write outside the registry.
    """
    wanted = {}
    with tarfile.open(fileobj=io.BytesIO(model_tar_gz), mode="r:gz") as tar:
        members = {m.name.lstrip("./"): m for m in tar.getmembers() if m.isfile()}
        for base in (("best/", "") if use_best else ("",)):
            if all(base + f in members for f in (WEIGHTS, CARD)):
                for f in (WEIGHTS, CARD):
                    m = members[base + f]
                    if m.size > MAX_MEMBER_BYTES:
                        raise ValueError(f"{m.name} is {m.size} bytes")
                    wanted[f] = tar.extractfile(m).read()
                wanted["_from"] = (base or "final").encode()
                break
    if WEIGHTS not in wanted:
        raise ValueError(f"model archive has no {WEIGHTS} + {CARD}")
    return wanted


def read_policy_dir(model_dir: Path, *, use_best: bool = True) -> dict[str, bytes]:
    """``{policy.npz, policy_card.json}`` from a local job's output directory."""
    for sub, label in ((("best", "best/"),) if use_best else ()) + ((".", "final"),):
        d = model_dir / sub
        if (d / WEIGHTS).is_file() and (d / CARD).is_file():
            return {WEIGHTS: (d / WEIGHTS).read_bytes(), CARD: (d / CARD).read_bytes(),
                    "_from": label.encode()}
    raise ValueError(f"{model_dir} holds no {WEIGHTS} + {CARD}")


def install(policy_dir: str, name: str, files: dict[str, bytes], *,
            provenance: Optional[dict] = None) -> dict:
    """Validate and store a policy as ``<policy_dir>/<name>``; returns its summary."""
    name = valid_policy_name(name)
    root = _root(policy_dir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".incoming-") as tmp:
        t = Path(tmp)
        (t / WEIGHTS).write_bytes(files[WEIGHTS])
        card = json.loads(files[CARD])
        card["name"] = name
        if provenance:
            card["provenance"] = provenance
        (t / CARD).write_text(json.dumps(card, indent=2, default=str))
        Policy.load(t, name)                          # refuses a wrong schema / missing modules
        dest = root / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(t, dest)
    return describe(policy_dir, name)


def promote(policy_dir: str, name: str) -> dict:
    """Make ``name`` the policy the planner uses by default (``current``)."""
    name = valid_policy_name(name)
    root = _root(policy_dir)
    src = root / name
    Policy.load(src, name)
    staging = root / f".{CURRENT}.new"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(src, staging)
    card = json.loads((staging / CARD).read_text())
    card["promoted_from"] = name
    (staging / CARD).write_text(json.dumps(card, indent=2, default=str))
    old = root / f".{CURRENT}.old"
    if old.exists():
        shutil.rmtree(old)
    if (root / CURRENT).exists():
        (root / CURRENT).rename(old)
    staging.rename(root / CURRENT)
    if old.exists():
        shutil.rmtree(old)
    return describe(policy_dir, CURRENT)


def describe(policy_dir: str, name: str) -> dict:
    d = _root(policy_dir) / name
    card = json.loads((d / CARD).read_text())
    training = card.get("training") or {}
    return {"name": name, "promoted_from": card.get("promoted_from"),
            "job": training.get("job"), "trained_on": training.get("cells"),
            "evaluated_on": training.get("eval_cells"),
            "eval_fraction_solved": training.get("eval_fraction_solved"),
            "step": training.get("step"), "schema_version": card.get("schema_version"),
            "arch": card.get("arch"), "provenance": card.get("provenance"),
            "size_bytes": (d / WEIGHTS).stat().st_size}


def list_policies(policy_dir: str) -> list[dict]:
    root = _root(policy_dir)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        try:
            out.append(describe(policy_dir, d.name))
        except Exception as e:  # noqa: BLE001 — list what is there, say what is broken
            out.append({"name": d.name, "error": str(e)})
    return out
