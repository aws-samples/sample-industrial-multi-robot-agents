#!/usr/bin/env python3
"""
Generate a complete docker-compose file for a specific cell.

Robot-agent services are derived from the cell's robots.yaml — one container
per robot namespace. The output is a self-contained compose file that replaces
the base docker-compose.yml for that cell.

Usage:
    python3 scripts/generate_compose.py --cell welding_cell_2_small
    python3 scripts/generate_compose.py --cell welding_cell_2_small \
        --trajectory-backend ompl \
                --llm-provider bedrock \
                --llm-model opus-5 \
        > docker-compose.welding_cell_2_small.yml

    # Middleware is a startup decision (agents/middleware.py owns it):
    #   --middleware cyclonedds                       peer-to-peer (default)
    #   --middleware zenoh --zenoh-router tcp/broker.plant.local:7447
    #                                                 connect to an existing
    #                                                 broker; no router is run

    # Called automatically by `make sim SIM=<cell>` via:
    #   python3 scripts/generate_compose.py --cell <cell> [options] > .generated_compose.yml
    #   docker compose -f .generated_compose.yml up
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# agents/llm_model.py is the single source of truth for what "opus-5" expands to
# and which inference-profile prefix a region accepts. Importing it here — rather
# than repeating a model literal in the generator — is why the startup default and
# the agents' runtime default cannot drift apart. The module is stdlib-only, so it
# imports fine in a bare shell with no model stack installed.
from agents.llm_model import (  # noqa: E402
    default_model_for,
    normalize_model,
    split_model,
)

# agents/middleware.py owns which transport the cell runs on and, for a brokered
# transport, which existing broker it connects to. Imported for the same reason
# as llm_model above: a middleware literal repeated here is how the generator and
# the Dockerfile drift apart. Stdlib-only, so it imports with no ROS present.
from agents.middleware import (  # noqa: E402
    DEFAULT_ZENOH_PORT,
    describe as describe_middleware,
    middleware_env,
    normalize_router_endpoint,
    resolve_middleware,
    uses_broker,
)

# utils/cell_layout.py owns where a cell keeps its config. This generator probes
# the HOST tree (./cells) but writes paths a CONTAINER will read (/cells), and the
# config directory moved into agentic_planner/ — so the URIs baked into the env
# here are computed, never spelled out. Stdlib-only apart from a lazy yaml import.
from utils import cell_layout  # noqa: E402

# Import root for the repo packages the live-mounted rviz launcher needs. Its own
# directory (/viz) is a read-only bind and Docker refuses to create a mountpoint
# inside one, so the packages cannot be mounted under it — see rviz_web_service.
VIZ_PYLIB = "/pylib"

# Host port for rviz-web's noVNC. Declared once: the dashboard embeds that viewer
# in an iframe, so it is given the same number (NOVNC_HOST_PORT) rather than
# repeating it — a viewer that silently points at the wrong port is a black
# rectangle with no error. tests/test_novnc_port_agrees.py pins the two together.
NOVNC_HOST_PORT = 6080


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _x_ros_env(cell_id: str, middleware: str, router: str | None,
               session_config_uri: str | None,
               dds_config_uri: str | None = None) -> dict:
    """ROS env shared by every container: identity, then transport.

    The transport half is delegated so that only the selected middleware's
    variables appear — a Zenoh stack carries no CycloneDDS config to be mistaken
    for something the system reads.

    Both config URIs are passed IN rather than composed by ``middleware_env``:
    which directory a cell keeps its config in is a question only a filesystem
    probe answers, and this generator is the layer that can see the tree.
    """
    return {
        "ROS_DOMAIN_ID": "42",
        **middleware_env(middleware, cell_id, router=router,
                         session_config_uri=session_config_uri,
                         dds_config_uri=dds_config_uri),
        "CELL_BASE_PATH": "/cells",
        "CELL_ID": cell_id,
    }


def _x_llm_env(llm_provider: str, llm_model: str, bedrock_model: str,
               bedrock_region: str) -> dict:
    """LLM configuration (non-secret). Credentials live in _x_secret_env().

    ``LLM_MODEL`` is the startup default every agent resolves against, and it is
    written as a fully-qualified ``provider:model_id`` — resolved HERE, against
    the containers' region, so what runs is visible in the generated file rather
    than re-derived per container from whatever region each one happens to see.
    """
    env = {
        "LLM_PROVIDER": llm_provider,
        "LLM_MODEL": llm_model,
        # BEDROCK_REGION pins Bedrock to Ireland regardless of the caller's
        # ambient AWS_REGION; the agents prefer BEDROCK_REGION over AWS_REGION.
        "BEDROCK_REGION": bedrock_region,
        "AWS_REGION": bedrock_region,
    }
    # Only emitted when the operator actually set it. Written unconditionally it
    # would be a second, competing default — the divergence this consolidation
    # removed (four modules each pinning their own Sonnet 5).
    if bedrock_model:
        env["BEDROCK_MODEL_ID"] = bedrock_model
    return env


def _x_secret_env() -> dict:
    """Cloud / LLM credentials, passed via environment variables.

    Each value is interpolated by ``docker compose`` from the host environment
    at parse time — export the variables in the shell that runs ``make sim`` (or
    put them in a ``.env`` file next to the compose file). These are injected
    into EVERY service so any container can authenticate. Unset variables
    resolve to an empty string (``:-``), which simply leaves that credential
    absent rather than failing compose interpolation."""
    return {
        # AWS (Bedrock). AWS_SESSION_TOKEN is required for temporary/STS creds.
        "AWS_ACCESS_KEY_ID": "${AWS_ACCESS_KEY_ID:-}",
        "AWS_SECRET_ACCESS_KEY": "${AWS_SECRET_ACCESS_KEY:-}",
        "AWS_SESSION_TOKEN": "${AWS_SESSION_TOKEN:-}",
        # OpenAI provider.
        "OPENAI_API_KEY": "${OPENAI_API_KEY:-}",
    }


def _app_service(
    name: str,
    container_name: str,
    command: str,
    environment: dict,
    ports: list[str] | None = None,
    volumes: list[str] | None = None,
    networks: list[str] | None = None,
    depends_on: dict | None = None,
    healthcheck: dict | None = None,
    profiles: list[str] | None = None,
    builds: bool = False,
) -> dict:
    """One agent container. Every agent runs the SAME image (AGENT_IMAGE): the
    container is generic and what it is — which robot, which planner — comes from
    its name and environment. ``builds`` marks the one service that declares how
    to build that image; a per-service image (compose's default without
    ``image:``) left every robot beyond those of the last build with "No such
    image", so a cell with more arms could not start."""
    svc: dict = {
        "image": AGENT_IMAGE,
        "pull_policy": "never",          # local-only — never pull from registry
        "container_name": container_name,
        "command": [command],
        "environment": environment,
        "volumes": volumes or [
            ".:/workspace/multi_robots_agents:ro",   # live code mount — no rebuild needed
            "./cells:/cells:ro",                      # cell definitions
        ],
        "networks": networks or ["agent-net"],
    }
    if builds:
        svc["build"] = {"context": ".", "dockerfile": "docker/Dockerfile"}
    if ports:
        svc["ports"] = ports
    if depends_on:
        svc["depends_on"] = depends_on
    if healthcheck:
        svc["healthcheck"] = healthcheck
    if profiles:
        svc["profiles"] = profiles
    return svc


#: The one image every agent container runs (planners, orchestrator, cell
#: controller, robot agents and controllers, dashboard). Built by `make build`
#: through the trajectory-agent service, which is in every generated stack.
AGENT_IMAGE = "${AGENT_IMAGE:-multi-robots-agents/agent:latest}"


def manifest_absolute_dirs(cell_id: str, cell_base: str | None) -> list[str]:
    """Host directories of the cell's ABSOLUTE asset paths (``cell.yaml``).

    A manifest path is relative to the cell unless it starts with ``/``
    (``cell_layout._manifest_asset``). A relative path means the same file on the
    host and at ``/cells/<cell>`` in a container; an absolute one is a host path,
    so every container that reads the cell gets its directory mounted read-only
    at the SAME path — then it means the same file there too. The directory, not
    the file: a URDF's meshes sit beside it.
    """
    dirs = set()
    for robot in (cell_layout.read_manifest(cell_id, cell_base).get("robots") or []):
        if not isinstance(robot, dict):
            continue
        for key in ("urdf", "srdf"):
            value = str(robot.get(key) or "")
            if value.startswith("/"):
                dirs.add(os.path.dirname(os.path.normpath(value)))
    return sorted(d for d in dirs if d and d != "/")


# The orchestrator's cell-data MCP. In its own process rather than its own
# service: its whole job is to be the voice of the agent that holds /cells, and
# split apart there would be two processes writing that tree again.
ORCH_MCP_PORT = "8893"

# Where published ports listen on the host. The agents' HTTP/MCP endpoints have no
# authentication, so by default they answer this machine only; PUBLISH_HOST=0.0.0.0
# opens them to the network. Two ports are the split deployment's contract and
# always listen on every interface: the cell controller's MCP (:8892, reached by
# the orchestrator) and the orchestrator's cell-data MCP (:8893, reached by the
# cell controller).
LOCAL = "${PUBLISH_HOST:-127.0.0.1}"

# Trained RoboBallet policies. The policy trainer is the one writer; the planner
# mounts it read-only (agents/trajectory/backends/roboballet). Not cell data:
# a policy is trained across cells and used on any of them.
POLICY_MODELS_VOLUME = "policy-models"
POLICY_TRAINER_PORT = "8894"
# The trainer's own state: job records, local training logs, inputs and outputs.
# Nobody else reads it.
TRAINER_STATE_VOLUME = "trainer-state"
TRAINER_IMAGE = "${TRAINER_IMAGE:-multi-robots-agents/policy-trainer:latest}"

# The planner's private scratch. Named, not a bind mount: an unfinished run is the
# worker's own business, and nothing outside that container reads it.
PLANNER_SCRATCH_VOLUME = "planner-scratch"


# ─── Service builders ─────────────────────────────────────────────────────────

def planner_name(index: int) -> str:
    """Planner 1 keeps the historical name; the others are numbered."""
    return "trajectory-agent" if index == 1 else f"trajectory-agent-{index}"


def planner_scratch_volume(index: int) -> str:
    return PLANNER_SCRATCH_VOLUME if index == 1 else f"{PLANNER_SCRATCH_VOLUME}-{index}"


def trajectory_agent_service(cell_id: str, traj_backend: str, ros_env: dict, llm_env: dict,
                             index: int = 1) -> dict:
    """The planner. Reads the cell's scene bundle; asks the orchestrator for the rest.

    ``/cells`` is READ-ONLY here, and that is the point. The planner used to hold
    the same tree read-write as the orchestrator and write plans, traces, events,
    quality logs and task programs straight to it — two writers on one tree, which
    is the shape of the bug the dashboard already had to be moved off. Everything
    derived now goes through the orchestrator's cell-data MCP
    (``CELL_DATA_MODE=orchestrator``), which is the process that owns the volume.

    What stays mounted is the scene bundle — URDF/SRDF/meshes/scene.yaml — because
    its mesh URIs are baked into the published ``robot_description`` and resolved
    by RViz and other subscribers, and a cell is 33–973 MB: shared immutable input,
    not derived data.

    What stays on the planner's OWN disk is ``/scratch``: unfinished work. A paused
    run's checkpoint (the model's conversation plus trajectories solved but not yet
    committed) belongs to the worker that is still working on it, and a named
    volume rather than a bind mount because it is nobody else's to read.

    ``index`` > 1 is an ADDITIONAL planner (``make sim PLANNERS=N``): its own name,
    its own scratch volume, no published port (the orchestrator reaches it over
    the stack's network). Every planner registers with the orchestrator, which
    gives each new planning run to a free one — one run per planner.
    """
    name = planner_name(index)
    return _app_service(
        name=name,
        container_name=name,
        builds=index == 1,
        command="source /opt/ros/jazzy/setup.bash && python3 agents/trajectory/mcp_server.py",
        environment={**ros_env, **llm_env, "TRAJ_MCP_PORT": "8890",
                     "TRAJECTORY_BACKEND": traj_backend,
                     # Out-of-repo backends, "name=module:attr,..." (see
                     # agents/trajectory/backends/__init__.py; code in plugins/).
                     "TRAJECTORY_BACKEND_PLUGINS": "${TRAJECTORY_BACKEND_PLUGINS:-}",
                     # Every read and write of derived cell data goes to the owner.
                     "CELL_DATA_MODE": "orchestrator",
                     "ORCHESTRATOR_HOST": "orchestrator",
                     "ORCH_MCP_PORT": ORCH_MCP_PORT,
                     # Retry a REFUSED connection while the orchestrator starts.
                     "CELL_STORE_CONNECT_RETRY_S": "20",
                     # 1 on a planning-only stack (make sim ONLY=trajectory).
                     "DDS_DISABLED": "${DDS_DISABLED:-0}",
                     # Planning-only model; empty = the code default (Sonnet),
                     # see agents/llm_model.planning_model.
                     "PLANNING_LLM_MODEL": "${PLANNING_LLM_MODEL:-}",
                     # Default planning mode for requests that name none;
                     # empty = the code default (classic), see plan_selection.
                     "PLANNING_MODE": "${PLANNING_MODE:-}",
                     # Independent OMPL solves of one problem run at once in
                     # forked children (ompl + ompl-compound); 0 = one after another.
                     "OMPL_PARALLEL_CANDIDATES": "${OMPL_PARALLEL_CANDIDATES:-1}",
                     # The arms' returns HOME planned at once, kept only when the
                     # gate runs them together; 0 = the coupled planner only.
                     "HOME_PARALLEL_FASTPATH": "${HOME_PARALLEL_FASTPATH:-1}",
                     # Reuse an unchanged task's allocation across runs.
                     "PLAN_MEMO_PROCESS": "${PLAN_MEMO_PROCESS:-1}",
                     # Across runs (tools/experience_store.py): reuse a re-checked
                     # planned step; offer past search paths as candidates.
                     "PLAN_LEG_STORE": "${PLAN_LEG_STORE:-1}",
                     "PLAN_EXPERIENCE": "${PLAN_EXPERIENCE:-1}",
                     # Compiled collision kernels (tools/fastgeom.py), 0 = NumPy;
                     # machine code cached on the persistent scratch volume.
                     "NUMBA_GEOM": "${NUMBA_GEOM:-1}",
                     "NUMBA_CACHE_DIR": "/scratch/numba_cache",
                     # Unfinished runs and temporary work only.
                     "PLANNER_SCRATCH_PATH": "/scratch",
                     # Registration for planning distribution (planner_registration):
                     # who this planner is, where the orchestrator reaches it.
                     "PLANNER_ID": name,
                     "PLANNER_ADVERTISE_URL": f"http://{name}:8890/mcp",
                     "ORCHESTRATOR_URL": "http://orchestrator:8889",
                     "PLANNER_HEARTBEAT_S": "${PLANNER_HEARTBEAT_S:-10}",
                     # The roboballet backend's trained policies (read-only here)
                     # and which one it uses; empty = "current", the promoted one.
                     "ROBOBALLET_POLICY_DIR": "/models",
                     "ROBOBALLET_POLICY": "${ROBOBALLET_POLICY:-}"},
        ports=[f"{LOCAL}:8890:8890"] if index == 1 else None,
        volumes=[
            ".:/workspace/multi_robots_agents:ro",   # live code mount
            "./cells:/cells:ro",                      # scene bundle only — read-only on purpose
            f"{planner_scratch_volume(index)}:/scratch",  # unfinished trajectories + temp work
            f"{POLICY_MODELS_VOLUME}:/models:ro",     # trained policies — the trainer writes them
        ],
        healthcheck={
            "test": ["CMD", "curl", "-f", "http://localhost:8890/health"],
            "interval": "10s",
            "retries": 5,
        },
    )


def orchestrator_service(cell_id: str, ros_env: dict, llm_env: dict) -> dict:
    return _app_service(
        name="orchestrator",
        container_name="orchestrator",
        command="source /opt/ros/jazzy/setup.bash && python3 agents/orchestrator/server.py",
        environment={**ros_env, **llm_env, "ORCHESTRATOR_PORT": "8889",
                     "TRAJECTORY_AGENT_HOST": "trajectory-agent", "TRAJECTORY_AGENT_PORT": "8890",
                     # Which planner URLs may register (planner_registry); empty =
                     # this stack's trajectory-agent[-N] containers only.
                     "PLANNER_ALLOWED_URLS": "${PLANNER_ALLOWED_URLS:-}",
                     # Where the cell controller runs: its compose name on one
                     # machine, its IP/hostname when execution is deployed apart
                     # (make sim ONLY=trajectory CELL_CONTROLLER_HOST=<ip>).
                     "CELL_CONTROLLER_HOST": "${CELL_CONTROLLER_HOST:-cell-controller}",
                     "CELL_MCP_PORT": "8892",
                     # The cell-data MCP, served in-process by this container
                     # because this is the container that holds /cells read-write.
                     "ORCH_MCP_PORT": ORCH_MCP_PORT},
        ports=[f"{LOCAL}:8889:8889", f"{ORCH_MCP_PORT}:{ORCH_MCP_PORT}"],
        depends_on={"trajectory-agent": {"condition": "service_healthy"}},
        volumes=[
            ".:/workspace/multi_robots_agents:ro",   # live code mount
            "./cells:/cells",                         # read-write: run logs → cells/<cell>/logs
        ],
    )


def policy_trainer_service(cell_id: str, ros_env: dict) -> dict:
    """Trains RoboBallet policies — on SageMaker or locally (optional, profile: training).

    Its OWN image (docker/Dockerfile.trainer: slim Python + PyTorch, no ROS) and
    its own container, started by ``make trainer`` (or ``make sim TRAINER=1``).
    The one place that talks to SageMaker: it prepares the inputs (scene packs
    from the cells, the training code bundle), starts, monitors and stops jobs,
    and imports finished policies. ``target="local"`` runs the same training
    code as a process of this container instead.

    It reads cells as the planner does — scene bundle from the read-only mount,
    derived data through the orchestrator that owns it. It writes its own
    ``trainer-state`` and the ``policy-models`` volume the planner reads. Its port
    is published on this machine only (LOCAL): the endpoint has no
    authentication and can start paid jobs. Jobs run where POLICY_TRAINING_TARGET
    says (local by default); "sagemaker" also needs the settings below, from the host's
    environment / .env; credentials come in like every service's.
    """
    svc = _app_service(
        name="policy-trainer",
        container_name="policy-trainer",
        command="python3 agents/policy_trainer/mcp_server.py",
        environment={"CELL_BASE_PATH": ros_env.get("CELL_BASE_PATH", "/cells"),
                     "CELL_ID": cell_id,
                     "POLICY_TRAINER_PORT": POLICY_TRAINER_PORT,
                     "CELL_DATA_MODE": "orchestrator",
                     "ORCHESTRATOR_HOST": "${ORCHESTRATOR_HOST:-orchestrator}",
                     "ORCH_MCP_PORT": ORCH_MCP_PORT,
                     "CELL_STORE_CONNECT_RETRY_S": "20",
                     "ROBOBALLET_POLICY_DIR": "/models",
                     "POLICY_TRAINER_STATE": "/state",
                     # THE switch: where jobs run. "local" unless the operator
                     # chooses "sagemaker" — a SageMaker job costs money.
                     "POLICY_TRAINING_TARGET": "${POLICY_TRAINING_TARGET:-local}",
                     "LOCAL_TRAINING_MAX_JOBS": "${LOCAL_TRAINING_MAX_JOBS:-2}",
                     # 1: the monitor resumes local jobs a restart interrupted.
                     "POLICY_TRAINING_AUTO_RESUME": "${POLICY_TRAINING_AUTO_RESUME:-0}",
                     "POLICY_TRAINER_POLL_S": "${POLICY_TRAINER_POLL_S:-30}",
                     "SAGEMAKER_ROLE_ARN": "${SAGEMAKER_ROLE_ARN:-}",
                     "POLICY_S3_URI": "${POLICY_S3_URI:-}",
                     "SAGEMAKER_REGION": "${SAGEMAKER_REGION:-${AWS_REGION:-}}",
                     "AWS_ACCOUNT_ID": "${AWS_ACCOUNT_ID:-}",
                     "SAGEMAKER_IMAGE_URI": "${SAGEMAKER_IMAGE_URI:-}",
                     "SAGEMAKER_INSTANCE_TYPE": "${SAGEMAKER_INSTANCE_TYPE:-ml.g5.2xlarge}",
                     "SAGEMAKER_MAX_RUNTIME_S": "${SAGEMAKER_MAX_RUNTIME_S:-86400}",
                     "SAGEMAKER_KMS_KEY_ID": "${SAGEMAKER_KMS_KEY_ID:-}",
                     "SAGEMAKER_USE_SPOT": "${SAGEMAKER_USE_SPOT:-0}"},
        ports=[f"{LOCAL}:{POLICY_TRAINER_PORT}:{POLICY_TRAINER_PORT}"],
        volumes=[
            ".:/workspace/multi_robots_agents:ro",   # live code mount
            "./cells:/cells:ro",                      # scene bundle (URDFs) — read-only
            f"{POLICY_MODELS_VOLUME}:/models",        # trained policies, for the planner
            f"{TRAINER_STATE_VOLUME}:/state",         # job records, local logs + outputs
        ],
        depends_on={"orchestrator": {"condition": "service_started"}},
        healthcheck={
            # The slim image has no curl.
            "test": ["CMD", "python3", "-c",
                     "import urllib.request; urllib.request.urlopen("
                     f"'http://localhost:{POLICY_TRAINER_PORT}/health', timeout=5)"],
            "interval": "10s",
            "retries": 5,
        },
        profiles=["training"],
    )
    # A fixed tag, so `make trainer-build` can build it without a compose file
    # (the image does not depend on the cell) and `up --no-build` finds it.
    svc["image"] = TRAINER_IMAGE
    svc["build"] = {"context": "./docker", "dockerfile": "Dockerfile.trainer",
                    "args": {"TORCH_INDEX": "${TRAINER_TORCH_INDEX:-cpu}"}}
    if os.environ.get("TRAINER_GPU", "").strip() in ("1", "true", "yes"):
        # A GPU host: local training on CUDA (build with TRAINER_TORCH_INDEX=cu124).
        svc["deploy"] = {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}}
    return svc


def cell_controller_service(cell_id: str, ros_env: dict, llm_env: dict) -> dict:
    return _app_service(
        name="cell-controller",
        container_name="cell-controller",
        command="source /opt/ros/jazzy/setup.bash && python3 agents/cell/mcp_server.py",
        environment={**ros_env, **llm_env, "CELL_HTTP_PORT": "8891", "CELL_MCP_PORT": "8892",
                     # Plans and cell config are read through the orchestrator's
                     # cell-data MCP — on one machine or across two
                     # (make sim ONLY=execution ORCHESTRATOR_HOST=<ip>).
                     "CELL_DATA_MODE": "orchestrator",
                     "ORCHESTRATOR_HOST": "${ORCHESTRATOR_HOST:-orchestrator}",
                     "ORCH_MCP_PORT": ORCH_MCP_PORT,
                     "CELL_STORE_CONNECT_RETRY_S": "20",
                     # Opt-in lockstep for ompl-compound steps (make LOCKSTEP=1),
                     # interpolated at `up` time like DDS_DISABLED.
                     "LOCKSTEP": "${LOCKSTEP:-0}"},
        ports=[f"{LOCAL}:8891:8891", "8892:8892"],
        volumes=[
            ".:/workspace/multi_robots_agents:ro",   # live code mount
            "./cells:/cells",                         # read-write for .rrd recordings
        ],
    )


def robot_agent_service(cell_id: str, namespace: str, robot_id: str,
                        robot_model: str, ros_env: dict) -> dict:
    """One robot-agent per robot. namespace is the shared ROS namespace,
    robot_id is the unique robot identity (topics: /{cell}/{ns}/{robot_id}/...)."""
    return _app_service(
        name=f"robot-agent-{robot_id}",
        container_name=f"robot-agent-{robot_id}",
        command="source /opt/ros/jazzy/setup.bash && python3 agents/robot/robot_agent.py",
        environment={
            **ros_env,
            "ROBOT_NAMESPACE": namespace,
            "ROBOT_ID": robot_id,
            "ROBOT_MODEL": robot_model,
            "CONTROLLER_NAME": "joint_trajectory_controller",
        },
    )


def robot_controller_service(cell_id: str, namespace: str, robot_id: str,
                             ros_env: dict) -> dict:
    """Simulated FJT 'hardware' controller for one robot (sim motion backend).

    Provides the FollowJointTrajectory action server the robot-agent dispatches
    to and publishes joint_states at ~120 Hz, so the arm actually moves in sim.
    One container per robot. There is no Isaac backend in this repo, so
    this is the default motion source; set ROBOT_CONTROLLER=none to disable it
    when an external simulator (e.g. Isaac Sim) publishes joint_states instead."""
    return _app_service(
        name=f"robot-controller-{robot_id}",
        container_name=f"robot-controller-{robot_id}",
        command="source /opt/ros/jazzy/setup.bash && python3 controller/pinocchio_robot_controller.py",
        environment={
            **ros_env,
            "ROBOT_NAMESPACE": namespace,
            "ROBOT_ID": robot_id,
            "JOINT_COUNT": "6",
        },
    )


def isaac_controller_service(cell_id: str, namespace: str, robot_id: str,
                             robot_cfg: dict, ros_env: dict) -> dict:
    """Isaac Sim FJT bridge for one robot (isaac motion backend).

    Runs INSTEAD of the pinocchio controller when ROBOT_CONTROLLER=isaac. It
    exposes the same FollowJointTrajectory action server the robot-agent
    dispatches to, converts each goal into Isaac's joint_command topic, and
    relays Isaac's joint_states/contacts back onto the cell contract. The
    controller is replaced, not disabled, so the arm still moves — Isaac Sim
    provides the physics.

    NOTE: a ROS 2 node lives on a single ROS_DOMAIN_ID, so Isaac Sim, the
    robot-agent, and this bridge must all share the same domain (the reference
    Isaac layout uses 0). Override the domain via ROS_DOMAIN_ID if Isaac is not
    on 42."""
    # Isaac articulation prim, derived from the robot index to match the
    # scene.yaml export (robot1 -> kr240r2900__sticktool, robot2 -> ..._2).
    # Override per robot with ROBOT_NAME / ISAAC_*_TOPIC env if your scene
    # differs.
    idx = robot_id.replace("robot", "").strip()
    robot_name = "kr240r2900__sticktool" if idx in ("", "1") else f"kr240r2900__sticktool_{idx}"
    return _app_service(
        name=f"robot-controller-{robot_id}",
        container_name=f"robot-controller-{robot_id}",
        command="source /opt/ros/jazzy/setup.bash && python3 controller/isaac_sim_controller.py",
        environment={
            **ros_env,
            "ROBOT_NAMESPACE": namespace,
            "ROBOT_ID": robot_id,
            "ROBOT_NAME": robot_name,
            "JOINT_COUNT": "6",
            "ISAAC_SCENE_ROOT": os.environ.get("ISAAC_SCENE_ROOT", "Untitled"),
        },
    )


def web_panel_service(cell_id: str, ros_env: dict, llm_env: dict) -> dict:
    return _app_service(
        name="web-panel",
        container_name="web-panel",
        command="source /opt/ros/jazzy/setup.bash && python3 dashboard/web_control_panel.py",
        environment={
            **ros_env, **llm_env,
            # Where the planner runs: its compose name on one machine, the planning
            # machine when the panel runs with the execution side (ONLY=execution).
            "TRAJECTORY_AGENT_HOST": "${TRAJECTORY_AGENT_HOST:-trajectory-agent}",
            "TRAJECTORY_AGENT_PORT": "8890",
            "ORCHESTRATOR_HOST": "${ORCHESTRATOR_HOST:-orchestrator}", "ORCHESTRATOR_PORT": "8889",
            "CELL_CONTROLLER_HOST": "${CELL_CONTROLLER_HOST:-cell-controller}",
            # For the noVNC iframe in the panel's left column. A host port, not a
            # service name: the browser connects to rviz-web directly, so this
            # must be reachable from the operator's machine, not from this network.
            "NOVNC_HOST_PORT": str(NOVNC_HOST_PORT),
        },
        ports=[f"{LOCAL}:8888:8888", f"{LOCAL}:8765:8765"],
        depends_on={"orchestrator": {"condition": "service_started"}},
    )


def rviz_web_service(ros_env: dict) -> dict:
    """RViz web viewer (optional, profile: rviz) — built locally.

    ``launch_viz.py`` is live-mounted rather than baked in, so anything it
    imports has to be mounted too. It imports ``utils.cell_layout`` to find the
    cell's ``robots.yaml`` / ``obstacles.yaml`` / URDFs, which no longer sit at a
    fixed place in the tree (they moved under ``agentic_planner/``), so the
    viewer has to run the same probe as the planners instead of guessing.

    ``utils`` gets its own mount root, NOT ``/viz/utils``: ``/viz`` is a
    read-only bind, and Docker cannot create a mountpoint inside one — nesting
    fails at container init with "read-only file system", before any of this
    runs. Hence ``/pylib`` plus PYTHONPATH.
    """
    return {
        "build": {"context": ".", "dockerfile": "docker/Dockerfile.viz"},
        "pull_policy": "never",
        "container_name": "rviz-web",
        "profiles": ["rviz"],
        "environment": {
            **{k: v for k, v in ros_env.items()},
            # /viz is already sys.path[0] for the launcher; this is for its imports.
            "PYTHONPATH": VIZ_PYLIB,
            # The robot descriptions come from the trajectory agent's MCP — on
            # the planning machine when the viewer runs with the execution side.
            "TRAJECTORY_AGENT_HOST": "${TRAJECTORY_AGENT_HOST:-trajectory-agent}",
            "TRAJECTORY_AGENT_PORT": "8890",
        },
        "volumes": [
            "./cells:/cells:ro",   # cyclonedds.xml + robot URDFs/meshes
            "./docker:/viz:ro",    # entrypoint_rviz.sh + launch_viz.py (live-mounted)
            # The launcher's repo imports. Keep in step with launch_viz.py —
            # tests/test_viz_service_imports.py fails if it grows another one.
            f"./utils:{VIZ_PYLIB}/utils:ro",
        ],
        # Run the live-mounted entrypoint so the visualization launcher can be
        # iterated without rebuilding the image.
        "command": ["bash", "/viz/entrypoint_rviz.sh"],
        "ports": [f"{NOVNC_HOST_PORT}:80"],
        "networks": ["agent-net"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a complete docker-compose for a cell (robots derived from robots.yaml)."
    )
    parser.add_argument("--cell",               required=True,
                        help="Cell ID (directory under cells/)")
    parser.add_argument("--cell-base",          default=None,
                        help="Base path for cells/ (default: ./cells or CELL_BASE_PATH env)")
    parser.add_argument("--trajectory-backend", default=None,
                        help="ompl | rrt | ompl-compound  (default: ompl)")
    parser.add_argument("--middleware",         default=None,
                        help=("ROS 2 middleware: cyclonedds | zenoh  "
                              "(default: cyclonedds; env RMW)"))
    parser.add_argument("--zenoh-router",       default=None,
                        help=("Endpoint of the EXISTING Zenoh broker to connect to, "
                              f"e.g. tcp/broker.plant.local:{DEFAULT_ZENOH_PORT} "
                              "(comma-separated for several). No router is ever "
                              "started by this stack, so this is REQUIRED with "
                              "--middleware zenoh; env ZENOH_ROUTER"))
    parser.add_argument("--llm-provider",       default=None,
                        help="bedrock | openai | ollama  (default: bedrock)")
    parser.add_argument("--llm-model",          default=None,
                        help=("Startup LLM for every agent: an alias "
                              "(opus-5 | opus-4.8 | sonnet-5 | haiku), a full "
                              "model id, or provider:model  (default: opus-5)"))
    parser.add_argument("--bedrock-model",      default=None,
                        help="Bedrock model ID override [legacy; prefer --llm-model]")
    parser.add_argument("--aws-region",         default=None,
                        help="AWS region (default: us-east-1)")
    parser.add_argument("--robot-id",           default=None,
                        help="Robot model ID override (default: from robots.yaml or kr240r2900)")
    parser.add_argument("--planners",           type=int, default=None,
                        help="planner containers (env PLANNERS, default 1): each runs one "
                             "planning run at a time; the orchestrator distributes runs")
    args = parser.parse_args()
    planners = max(1, int(args.planners if args.planners is not None
                          else os.environ.get("PLANNERS") or 1))

    # ── Resolve arguments / env-var fallbacks ────────────────────────────────
    cell_id      = args.cell
    cell_base    = (args.cell_base
                    or os.environ.get("CELL_BASE_PATH", "").rstrip("/")
                    or "./cells")
    traj_backend = args.trajectory_backend or os.environ.get("TRAJECTORY_BACKEND", "ompl")
    # Motion backend (per robot, one controller container each):
    #   "sim"   (default) — pinocchio FJT controller simulates the motion
    #   "isaac"           — Isaac Sim FJT bridge: serves the FJT action and
    #                       converts goals to Isaac's joint_command topic while
    #                       relaying Isaac's joint_states back (Isaac is the sim)
    #   "none"            — no controller; an external process must serve the
    #                       FJT action AND publish joint_states, or arms won't move
    robot_controller = os.environ.get("ROBOT_CONTROLLER", "sim").strip().lower()
    # Middleware: which transport, and (for a brokered one) which existing broker.
    # Both are resolved HERE so a bad value fails before a compose file exists —
    # the same reason --llm-model is validated at generation time rather than in
    # the container. A stack that refuses to start beats one that comes up on a
    # transport the operator did not choose, or pointed at nothing.
    try:
        middleware = resolve_middleware(args.middleware)
        zenoh_router = normalize_router_endpoint(
            args.zenoh_router if args.zenoh_router is not None
            else os.environ.get("ZENOH_ROUTER"))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    # An endpoint given for a peer-to-peer transport is a contradiction, not a
    # harmless extra: the operator asked for a broker and would get none.
    if zenoh_router and not uses_broker(middleware):
        print(f"ERROR: --zenoh-router requires --middleware zenoh "
              f"(got {middleware!r})", file=sys.stderr)
        sys.exit(2)
    # A brokered transport with no broker named is the failure worth catching
    # loudest: rmw_zenoh would default to localhost inside each container, so the
    # stack would come up "healthy" with every agent talking only to itself.
    if uses_broker(middleware) and not zenoh_router:
        print(f"ERROR: --middleware {middleware} requires --zenoh-router "
              f"(this stack runs no router; point it at your broker, e.g. "
              f"--zenoh-router tcp/broker.plant.local:{DEFAULT_ZENOH_PORT})",
              file=sys.stderr)
        sys.exit(2)
    llm_provider = args.llm_provider       or os.environ.get("LLM_PROVIDER", "bedrock")
    # Legacy override only — no default. A default here would shadow --llm-model.
    bedrock_model = args.bedrock_model or os.environ.get("BEDROCK_MODEL_ID", "")
    # Bedrock region is pinned to Ireland by default. Deliberately NOT sourced
    # from the ambient AWS_REGION (which varies per shell and caused an eu.
    # model to be paired with a us-east-1 region → invalid model identifier).
    bedrock_region = (args.aws_region
                      or os.environ.get("BEDROCK_REGION")
                      or "eu-west-1")
    # Resolve the alias against the CONTAINERS' region, not this shell's: the
    # inference-profile prefix must match the region the agents will call from.
    # A bad --llm-model raises here, before a compose file is written.
    try:
        llm_model = (normalize_model(args.llm_model, llm_provider, bedrock_region)
                     or normalize_model(os.environ.get("LLM_MODEL"), llm_provider,
                                        bedrock_region)
                     or normalize_model(bedrock_model, llm_provider, bedrock_region)
                     or normalize_model(default_model_for(llm_provider),
                                        llm_provider, bedrock_region))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    llm_model_id = split_model(llm_model)[1]
    robot_id_override = args.robot_id or os.environ.get("ROBOT_ID", "")

    # ── Read robots.yaml ─────────────────────────────────────────────────────
    robots_path = cell_layout.robots_yaml(cell_id, cell_base)
    if not robots_path.exists():
        print(f"ERROR: robots.yaml not found at {robots_path}", file=sys.stderr)
        print(f"       Convert the cell first (make convert), or write "
              f"{cell_layout.agent_config_file(cell_id, 'robots.yaml', cell_base)} "
              f"by hand.", file=sys.stderr)
        sys.exit(1)

    with open(robots_path) as f:
        robots_cfg = yaml.safe_load(f)

    # Flatten robots.yaml to (namespace, robot_id, config). Works for both the
    # legacy layout {robot1: {robot1: {..}}} and the shared-namespace layout
    # {arms: {robot1: {..}, robot2: {..}}}.
    robot_entries: list[tuple[str, str, dict]] = []
    for ns, ns_block in (robots_cfg.get("robots") or {}).items():
        if isinstance(ns_block, dict):
            for rid, rcfg in ns_block.items():
                if isinstance(rcfg, dict):
                    robot_entries.append((ns, rid, rcfg))
    robot_ids = [rid for _, rid, _ in robot_entries]
    if not robot_entries:
        print(f"WARNING: No robots found in {robots_path}", file=sys.stderr)

    # ── Shared env blocks ────────────────────────────────────────────────────
    # A cell MAY ship a full Zenoh session config for anything the inline
    # override cannot express (TLS material, scouting, access control). Pointed at
    # only when the file is actually there, so the variable never names a path
    # that does not exist inside the container.
    session_config_uri = (
        cell_layout.config_uri(cell_id, "zenoh_session.json5", cell_base)
        if middleware == "zenoh" else None)

    # The DDS config is read by the CycloneDDS C runtime from CYCLONEDDS_URI, so the
    # URI has to name wherever the cell actually keeps the file — the workspace's
    # config/ for a converted cell, the cell root for one written before it moved.
    # config_uri() answers from the same probe the readers use, and None when the
    # cell ships no DDS config at all: an unresolvable URI is ignored in silence by
    # the runtime, which then runs on its own defaults across the whole cell.
    dds_config_uri = (cell_layout.config_uri(cell_id, "cyclonedds.xml", cell_base)
                      if middleware == "cyclonedds" else None)
    if middleware == "cyclonedds" and dds_config_uri is None:
        print(f"WARNING: no cyclonedds.xml in cell '{cell_id}' — containers will "
              f"use the CycloneDDS built-in defaults.", file=sys.stderr)

    ros_env = _x_ros_env(cell_id, middleware, zenoh_router, session_config_uri,
                         dds_config_uri)
    llm_env = _x_llm_env(llm_provider, llm_model, bedrock_model, bedrock_region)

    # ── Build compose dict ───────────────────────────────────────────────────
    header = (
        f"## Auto-generated by scripts/generate_compose.py\n"
        f"## Cell: {cell_id}   Robots: {', '.join(robot_ids) or '(none)'}\n"
        f"## Trajectory backend: {traj_backend}   LLM provider: {llm_provider}\n"
        f"## LLM (startup default): {llm_model_id}   region: {bedrock_region}\n"
        f"##   Override per operation with the `model` field of an A2A request or\n"
        f"##   MCP tool call — no restart needed. GET /a2a/models lists the options.\n"
        f"## Motion backend (ROBOT_CONTROLLER): {robot_controller}\n"
        f"## Middleware (RMW): {describe_middleware(middleware, zenoh_router)}\n"
        f"##\n"
        f"## Credentials: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN\n"
        f"##   and OPENAI_API_KEY are read from the host env (or a .env file) and\n"
        f"##   injected into EVERY container. Export them before `make sim`.\n"
        f"##\n"
        f"## Regenerate:  make compose SIM={cell_id}\n"
        f"## Start:       make sim SIM={cell_id}\n"
    )

    services: dict = {}

    services["trajectory-agent"]  = trajectory_agent_service(cell_id, traj_backend, ros_env, llm_env)
    # Additional planners (make sim PLANNERS=N): one planning run each, at the same time.
    for i in range(2, planners + 1):
        services[planner_name(i)] = trajectory_agent_service(cell_id, traj_backend, ros_env,
                                                             llm_env, index=i)
    services["orchestrator"]       = orchestrator_service(cell_id, ros_env, llm_env)
    services["cell-controller"]    = cell_controller_service(cell_id, ros_env, llm_env)

    # ── One robot-agent per robot (all sharing their namespace) ───────────────
    for ns, robot_id, rcfg in robot_entries:
        # Robot model: CLI/env override > robots.yaml robot_model > default
        robot_model = robot_id_override or rcfg.get("robot_model", "") or "kr240r2900"
        services[f"robot-agent-{robot_id}"] = robot_agent_service(
            cell_id, ns, robot_id, robot_model, ros_env,
        )
        # Motion backend: one controller container per robot so the arm moves
        # and joint_states flow. "sim" uses the pinocchio controller; "isaac"
        # swaps in the Isaac Sim FJT bridge (controller replaced, not disabled);
        # "none" omits it entirely (external process must serve FJT + states).
        if robot_controller == "sim":
            services[f"robot-controller-{robot_id}"] = robot_controller_service(
                cell_id, ns, robot_id, ros_env,
            )
        elif robot_controller == "isaac":
            services[f"robot-controller-{robot_id}"] = isaac_controller_service(
                cell_id, ns, robot_id, rcfg, ros_env,
            )

    # Both backends (ompl, rrt) plan in-process in the trajectory agent — no
    # per-robot planning server container is needed.
    services["web-panel"] = web_panel_service(cell_id, ros_env, llm_env)
    services["rviz-web"]  = rviz_web_service(ros_env)
    services["policy-trainer"] = policy_trainer_service(cell_id, ros_env)

    # ── Redirect the cells bind-mount to the resolved host cell_base ──────────
    # Cell definitions can live ANYWHERE on the host (--cell-base / CELL_BASE_PATH,
    # default ./cells); the container path stays /cells. The service builders emit
    # the mount with the literal "./cells" host source, so rewrite just that source
    # here — every mount that targets /cells (rw and :ro), and nothing else.
    if cell_base != "./cells":
        for svc in services.values():
            vols = svc.get("volumes")
            if not vols:
                continue
            svc["volumes"] = [
                (f"{cell_base}:{v.split(':', 1)[1]}" if v.startswith("./cells:") else v)
                for v in vols
            ]

    # ── Absolute asset paths in cell.yaml: the same path in every container ──
    for host_dir in manifest_absolute_dirs(cell_id, cell_base):
        if not os.path.isdir(host_dir):
            print(f"[generate_compose] WARNING: cell.yaml names {host_dir}, which is not "
                  "a directory here — containers will not find those assets",
                  file=sys.stderr)
        for svc in services.values():
            vols = svc.get("volumes") or []
            if any(v.split(":")[1:2] == ["/cells"] for v in vols if ":" in v):
                svc["volumes"] = vols + [f"{host_dir}:{host_dir}:ro"]

    # No broker service is ever generated. Under a brokered middleware the router
    # is the operator's, already running on their network; this stack is a client
    # of it. That is also why there is no depends_on for it — compose can only
    # gate on containers it starts, so an unreachable broker has to surface as a
    # session that never connects rather than as a failed dependency.

    # ── Credentials + LLM config for ALL containers ──────────────────────────
    # Every service gets the host-interpolated credentials (and LLM config) so
    # any container — not just the four LLM agents — can authenticate. Values
    # already set on a service are preserved (setdefault); only missing keys are
    # added.
    common_env = {**llm_env, **_x_secret_env()}
    for svc in services.values():
        env = svc.setdefault("environment", {})
        for k, v in common_env.items():
            env.setdefault(k, v)

    compose = {
        "services": services,
        "networks": {"agent-net": {"driver": "bridge"}},
        "volumes": {**{planner_scratch_volume(i): {} for i in range(1, planners + 1)},
                    POLICY_MODELS_VOLUME: {}, TRAINER_STATE_VOLUME: {}},
    }

    # ── Emit YAML with header comment ────────────────────────────────────────
    output = header + yaml.dump(compose, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(output)


if __name__ == "__main__":
    main()
