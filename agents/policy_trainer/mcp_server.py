"""
Policy Trainer MCP Server — trains RoboBallet policies on SageMaker.

The ``roboballet`` trajectory backend plans with a trained graph-network
policy; this service is where such a policy comes from. It is driven entirely
over MCP, like the other agents:

  export_scene        a cell (existing cell format) → the scene pack training uses
  start_training      TRAINING cells (+ optional EVALUATION cells) → a job where
                      POLICY_TRAINING_TARGET says: "local" (the same train.py,
                      run by this container) or "sagemaker" (a SageMaker job)
  get_training_job / list_training_jobs / get_training_logs / stop_training_job /
  resume_training_job (continue one interrupted job from its checkpoint)
                      one interface for both targets; a monitor thread keeps
                      every job's status and metrics current (tracker.py)
  import_policy       a finished job's model → /models/<name> (validated);
                      or auto_import=True on start_training
  promote_policy      make it the one the planner uses (/models/current)
  list_policies
  evaluate_policy     score a policy offline on ANY cell — the evaluation layout

Training and evaluation cells are independent lists: train on one layout,
evaluate on another. Planning with the policy is the trajectory agent's
``plan_trajectories(..., backend="roboballet")``; nothing here plans.

Transport: SSE + plain JSON-RPC ``/mcp`` + ``/health`` on POLICY_TRAINER_PORT
(8894), or stdio (MCP_TRANSPORT=stdio). The container publishes the port on
127.0.0.1 only (PUBLISH_HOST): these endpoints have no authentication of
their own, and ``start_training`` spends money.

Where a job runs is the stack's choice, not the request's: POLICY_TRAINING_TARGET
= ``local`` (default) or ``sagemaker``. A SageMaker job also needs a complete
configuration (see ``agents/policy_trainer/config.py``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.policy_trainer import registry  # noqa: E402
from agents.policy_trainer.config import TrainerConfig, TrainerConfigError, valid_policy_name  # noqa: E402
from agents.policy_trainer import sagemaker_jobs as sj  # noqa: E402
from agents.policy_trainer.sagemaker.hyperparameters import check as hp_check  # noqa: E402
from agents.policy_trainer.tracker import Tracker  # noqa: E402

_TRANSPORT = os.environ.get("MCP_TRANSPORT", "sse").lower()
_PORT = int(os.environ.get("POLICY_TRAINER_PORT", "8894"))

mcp = FastMCP(
    "policy-trainer",
    instructions=(
        "Trains RoboBallet multi-robot policies (graph neural network, TD3 + hindsight\n"
        "replay) on AWS SageMaker from cells in the existing cell format, and installs\n"
        "them for the trajectory agent's `roboballet` backend.\n"
        "\n"
        "FLOW: export_scene(cell) to check a cell → start_training(train_cells,\n"
        "eval_cells) (try dry_run=True first; it runs where trainer_status's\n"
        "target says: local or sagemaker) →\n"
        "get_training_job(job) / get_training_logs(job) until Completed →\n"
        "import_policy(job, name) (or auto_import=True) → evaluate_policy(name, cell)\n"
        "on any layout → promote_policy(name). Then plan with backend='roboballet'.\n"
    ),
)


def _cfg() -> TrainerConfig:
    return TrainerConfig.from_env()


def _jobs(cfg: TrainerConfig) -> sj.Jobs:
    return sj.Jobs(cfg)


def _err(e: Exception) -> dict:
    return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _pack(cell_id: str) -> dict:
    from agents.trajectory.backends.roboballet.cell_scene import scene_pack
    return scene_pack(cell_id)


@mcp.tool()
def trainer_status() -> dict:
    """The trainer's configuration (no credentials), what stops a SageMaker job,
    whether local training can run here (torch, GPU), and the jobs in flight."""
    cfg = _cfg()
    return {"ok": True, "service": "policy-trainer", "config": cfg.describe(),
            "local_runtime": _local_runtime(),
            "active_jobs": [r["job_name"] for r in _tracker().list()
                            if r["status"] in ("InProgress", "Stopping")],
            "policies": [p["name"] for p in registry.list_policies(cfg.policy_dir)]}


def _local_runtime() -> dict:
    try:
        import torch
        return {"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()),
                "threads": torch.get_num_threads()}
    except Exception as e:  # noqa: BLE001
        return {"torch": None, "error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def export_scene(cell_id: str) -> dict:
    """Describe a cell the way the policy sees it: arms (DoF, base, envelope size),
    obstacle primitives and weld points. Fails when the cell cannot be described
    (no robots.yaml / URDF), which is also why it would fail to train or plan."""
    try:
        p = _pack(cell_id)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return {"ok": True, "cell_id": cell_id, "schema_version": p["schema_version"],
            "robots": [{"robot_id": r["robot_id"], "dof": len(r["home"]),
                        "base": r["chain"]["base"], "yaw": r["chain"]["yaw"],
                        "envelope_spheres": sum(len(v) for v in r["chain"]["link_spheres"].values())}
                       for r in p["robots"]],
            "obstacles": len(p["obstacles"]), "operations": len(p["operations"]),
            "warning_distance_m": p["warning_distance_m"]}


def _install_from(cfg: TrainerConfig, rec_or_job, name: Optional[str], use_best: bool) -> dict:
    """A finished job's policy → the registry, from wherever that job left it."""
    rec = rec_or_job if isinstance(rec_or_job, dict) else (_tracker().get(rec_or_job) or
                                                           {"job_name": rec_or_job,
                                                            "target": "sagemaker"})
    job = rec["job_name"]
    name = valid_policy_name(name or job)
    if rec["target"] == "local":
        if rec.get("status") != "Completed":
            raise RuntimeError(f"local job {job} is {rec.get('status')}, not Completed")
        files = registry.read_policy_dir(Path(rec["model_dir"]), use_best=use_best)
    else:
        cfg.require(mutating=False)
        files = registry.extract_policy(_jobs(cfg).download_model(job), use_best=use_best)
    picked = files.pop("_from").decode()
    return registry.install(cfg.policy_dir, name, files,
                            provenance={"job": job, "target": rec["target"], "checkpoint": picked})


_TRACKER: Optional[Tracker] = None


def _tracker() -> Tracker:
    """The job tracker (created on first use; the server starts its monitor)."""
    global _TRACKER
    if _TRACKER is None:
        cfg = _cfg()

        def importer(rec: dict) -> dict:
            out = _install_from(_cfg(), rec, None, True)
            if rec.get("promote"):
                registry.promote(_cfg().policy_dir, out["name"])
                out["promoted"] = True
            return out

        _TRACKER = Tracker(cfg.state_dir, sagemaker=lambda: _jobs(_cfg()), importer=importer)
    return _TRACKER


@mcp.tool()
def start_training(train_cells: list[str], eval_cells: Optional[list[str]] = None,
                   hyperparameters: Optional[dict] = None,
                   instance_type: Optional[str] = None, max_runtime_s: Optional[int] = None,
                   job_name: Optional[str] = None, auto_import: bool = False,
                   promote: bool = False, dry_run: bool = False) -> dict:
    """
    Train a RoboBallet policy where this stack trains: POLICY_TRAINING_TARGET =
    "local" (the same train.py as a process of this container, CPU unless the
    image has a GPU torch) or "sagemaker" (a SageMaker PyTorch job). The reply's
    ``target`` says which; trainer_status shows it before you start.

    Args:
        train_cells:     cells the procedural generator randomises (robot placement,
                         obstacle subsets + random cuboids, task sets).
        eval_cells:      cells used AS DECLARED to score checkpoints — may be other
                         layouts; the best-scoring checkpoint is kept as ``best/``.
        hyperparameters: overrides of the defaults (paper Table S5), e.g.
                         {"total_steps": 2000000, "arch_node": 512}. Unknown keys refuse.
        instance_type / max_runtime_s: SageMaker only.
        job_name:        default roboballet-<UTC timestamp>.
        auto_import:     the monitor installs the policy when the job completes
                         (as /models/<job_name>); ``promote`` also makes it current.
        dry_run:         return what would run (the CreateTrainingJob request, or the
                         local command line); start nothing.

    The job is tracked whatever the target: get_training_job, list_training_jobs,
    get_training_logs, stop_training_job.
    """
    cfg = _cfg()
    eval_cells = list(eval_cells or [])
    job = job_name or sj.new_job_name()
    try:
        target = cfg.require_target()
        if not train_cells:
            raise ValueError("train_cells is empty")
        if _tracker().get(job):
            raise ValueError(f"job {job!r} already exists")
        hp = hp_check(dict(hyperparameters or {}))
        train = {c: _pack(c) for c in train_cells}
        evals = {c: _pack(c) for c in eval_cells}
    except Exception as e:  # noqa: BLE001
        return _err(e)
    summary = {"job_name": job, "target": target,
               "train_cells": {c: {"robots": len(p["robots"]), "obstacles": len(p["obstacles"]),
                                   "operations": len(p["operations"])} for c, p in train.items()},
               "eval_cells": {c: {"robots": len(p["robots"]), "operations": len(p["operations"])}
                              for c, p in evals.items()}}

    if target == "local":
        if dry_run:
            return {"ok": True, "dry_run": True, **summary,
                    "command": _tracker().local_command(job, hp, bool(evals))[2:]}
        try:
            rec = _tracker().start_local(job, train, evals, hp, auto_import=auto_import,
                                         promote=promote)
        except Exception as e:  # noqa: BLE001
            return _err(e)
        return {"ok": True, "dry_run": False, "status": rec["status"], **summary}

    try:
        source, digest = sj.source_bundle()
        if not dry_run:
            cfg.require(mutating=True)
        request = sj.training_request(cfg, job, hyperparameters=hp, train_cells=list(train),
                                      eval_cells=list(evals), instance_type=instance_type,
                                      max_runtime_s=max_runtime_s, source_sha256=digest)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    summary["source_sha256"] = digest
    if dry_run:
        return {"ok": True, "dry_run": True, "request": request,
                "config_problems": cfg.problems(mutating=True), **summary}
    try:
        jobs = _jobs(cfg)
        account = jobs.check_account()
        jobs.upload_inputs(job, source, train, evals)
        arn = jobs.create(request).get("TrainingJobArn")
        _tracker().track_sagemaker(job, train_cells=list(train), eval_cells=list(evals),
                                   hyperparameters=hp, auto_import=auto_import,
                                   promote=promote, arn=arn, request=request)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return {"ok": True, "dry_run": False, "training_job_arn": arn, "account": account, **summary}


@mcp.tool()
def get_training_job(job_name: str) -> dict:
    """One job, refreshed now: status, latest metrics (eval_fraction_solved,
    train_fraction_solved, losses, steps_per_s), failure reason, import result.
    A SageMaker job started elsewhere is described directly."""
    rec = _tracker().get(job_name)
    try:
        if rec is not None:
            return {"ok": True, **_tracker().refresh(job_name)}
        cfg = _cfg()
        cfg.require(mutating=False)
        return {"ok": True, "target": "sagemaker", "tracked": False,
                **sj.summarize(_jobs(cfg).describe(job_name))}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def list_training_jobs(max_results: int = 20, include_untracked: bool = False) -> dict:
    """The jobs this trainer started (local and SageMaker), newest first, as the
    monitor last saw them. ``include_untracked`` also lists roboballet-* SageMaker
    jobs it does not track (started from another trainer or the console)."""
    jobs = [{k: r.get(k) for k in ("job_name", "target", "status", "detail", "created",
                                   "ended", "metrics", "checkpoint_step", "resumable",
                                   "resumed_from", "imported")}
            for r in _tracker().list()[: max(1, min(int(max_results), 200))]]
    out = {"ok": True, "jobs": jobs}
    if include_untracked:
        cfg = _cfg()
        try:
            cfg.require(mutating=False)
            known = {j["job_name"] for j in jobs}
            out["untracked"] = [{"job_name": j.get("TrainingJobName"),
                                 "status": j.get("TrainingJobStatus"),
                                 "created": str(j.get("CreationTime"))}
                                for j in _jobs(cfg).list(min(int(max_results), 100))
                                if j.get("TrainingJobName") not in known]
        except Exception as e:  # noqa: BLE001
            out["untracked_error"] = f"{type(e).__name__}: {e}"
    return out


@mcp.tool()
def get_training_logs(job_name: str, lines: int = 100) -> dict:
    """The last ``lines`` lines of a job's training output — the local process's
    log, or the SageMaker job's CloudWatch stream."""
    try:
        return {"ok": True, "job_name": job_name, "lines": _tracker().logs(job_name, lines)}
    except KeyError:
        return {"ok": False, "error": f"job {job_name!r} is not tracked by this trainer"}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def resume_training_job(job_name: str) -> dict:
    """
    Continue an interrupted, stopped or failed job from its last checkpoint.

    * local — the SAME job resumes in this container (``train.py --resume``),
      appending to its own log; the replay buffer is refilled with the policy it
      had learned, so nothing before the checkpoint is trained again.
    * SageMaker — a new job ``<job_name>-r<n>`` starts from the job's S3
      checkpoint (needs POLICY_TRAINING_TARGET=sagemaker, like any new job);
      follow it under that name.

    A trainer that restarts marks its running local jobs ``Interrupted`` (with
    ``resumable: true`` once a checkpoint was written — every
    ``checkpoint_every`` steps); set POLICY_TRAINING_AUTO_RESUME=1 to have the
    monitor resume them by itself.
    """
    cfg = _cfg()
    try:
        rec = _tracker().get(job_name)
        if rec is None:
            raise ValueError(f"job {job_name!r} is not tracked by this trainer")
        create = None
        if rec["target"] == "sagemaker":
            cfg.require(mutating=True)
            jobs = _jobs(cfg)
            jobs.check_account()
            create = jobs.create
        out = _tracker().resume(job_name, create_sagemaker=create)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return {"ok": True, "resumed": job_name, "running_as": out["job_name"],
            "status": out["status"], "target": out["target"],
            "from_checkpoint_step": (out.get("resumes") or [{}])[-1].get("checkpoint_step")
            if out["job_name"] == job_name else None}


@mcp.tool()
def stop_training_job(job_name: str) -> dict:
    """Stop a running job: terminate the local process, or StopTrainingJob on
    SageMaker (billing stops). Only jobs this trainer started."""
    cfg = _cfg()
    try:
        rec = _tracker().get(job_name)
        if rec is None:
            raise ValueError(f"job {job_name!r} is not tracked by this trainer")
        if rec["target"] == "sagemaker":
            # Stopping only ever saves money: allowed whatever the target is now.
            cfg.require(mutating=False)
            _jobs(cfg).check_account()
        return {"ok": True, **_tracker().stop(job_name)}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def import_policy(job_name: str, name: Optional[str] = None, use_best: bool = True,
                  promote: bool = False) -> dict:
    """
    Install a completed job's policy as /models/<name> (default: the job name) —
    from the local job's output, or the SageMaker job's model artefact.

    ``use_best`` takes the best-evaluated checkpoint when the job kept one, else
    the final weights. The card is validated against this build's graph schema;
    a mismatched policy is refused. ``promote`` also makes it ``current``.
    """
    cfg = _cfg()
    try:
        out = _install_from(cfg, job_name, name, use_best)
        if promote:
            registry.promote(cfg.policy_dir, out["name"])
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return {"ok": True, "policy": out, "promoted": bool(promote)}


@mcp.tool()
def list_policies() -> dict:
    """Installed policies, with what they were trained and evaluated on."""
    return {"ok": True, "policies": registry.list_policies(_cfg().policy_dir)}


@mcp.tool()
def promote_policy(name: str) -> dict:
    """Make an installed policy the planner's default (``ROBOBALLET_POLICY=current``)."""
    try:
        return {"ok": True, "current": registry.promote(_cfg().policy_dir, name)}
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def evaluate_policy(name: str, cell_id: str, mode: str = "free", episodes: int = 3,
                    max_tasks: int = 8, horizon_s: float = 30.0, seed: int = 0) -> dict:
    """
    Score a policy on a cell in its own kinematic simulator — any cell, trained on
    or not (``unseen_cell`` says which). ``mode``: "free" (any arm takes any weld,
    the paper's setting) or "assigned" (one weld per arm, as the backend asks).
    Returns mean fraction of welds reached, time, and how often an arm was held to
    avoid a collision. Planning with backend="roboballet" is the binding check.
    """
    from agents.policy_trainer.evaluate import evaluate
    from agents.trajectory.backends.roboballet.core.policy_io import Policy
    try:
        pol = Policy.load(Path(_cfg().policy_dir) / (name if name == registry.CURRENT
                                                     else valid_policy_name(name)), name)
        return {"ok": True, **evaluate(pol, _pack(cell_id), mode=mode, episodes=episodes,
                                       max_tasks=max_tasks, horizon_s=horizon_s, seed=seed)}
    except Exception as e:  # noqa: BLE001
        return _err(e)


if __name__ == "__main__":
    if _TRANSPORT == "stdio":
        _tracker().start_monitor()
        mcp.run(transport="stdio")
    else:
        import uvicorn
        from starlette.responses import JSONResponse

        from utils.mcp_http import add_jsonrpc_mcp_route

        app = mcp.sse_app()

        async def _health(_request):
            cfg = _cfg()
            return JSONResponse({"status": "ok", "service": "policy-trainer",
                                 "training_target": cfg.target,
                                 "sagemaker_problems": (cfg.problems(mutating=True)
                                                        if cfg.target == "sagemaker" else None)})

        app.add_route("/health", _health, methods=["GET"])
        add_jsonrpc_mcp_route(app, mcp)
        _tracker().start_monitor()
        # 0.0.0.0 INSIDE the container so the compose network reaches it; the host
        # publish is 127.0.0.1 by default (scripts/generate_compose.py, LOCAL).
        host = os.environ.get("POLICY_TRAINER_BIND", "0.0.0.0")
        print(f"[PolicyTrainerMCP] Starting SSE on {host}:{_PORT}", flush=True)
        uvicorn.run(app, host=host, port=_PORT, log_level="warning")
