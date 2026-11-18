## multi_robots_agents Makefile
##
## ─── Requirements ─────────────────────────────────────────────────────────────
##
##   Docker, make, curl. That is all — there is NO host Python requirement.
##   Every python3 and pytest invocation below runs inside the small runner image
##   built from docker/Dockerfile.tools, with the repo bind-mounted, so generated
##   files appear in the working tree as if a local interpreter had written them.
##   The image builds itself on first use (`make sim`, `make test`, ...) and is
##   cached until docker/Dockerfile.tools or docker/requirements-tools.txt change.
##   Build it ahead of time with `make tools`; drop it with `make tools-clean`.
##
## ─── Quick start ──────────────────────────────────────────────────────────────
##
##   # Start without any visualiser (headless / CI)
##   make sim SIM=welding_cell_2_small
##
##   # Start with the RViz web viewer (http://localhost:6080/vnc.html)
##   make sim SIM=welding_cell_2_small RVIZ=1
##
##   # Attach RViz to an already-running stack
##   make rviz SIM=welding_cell_2_small
##
##   # List tasks available in a cell (requires running stack)
##   make tasks SIM=welding_cell_2_small
##
##   # Run a task via A2A orchestrator
##   make task SIM=welding_cell_2_small TASK=car_door_spot_welding
##
##   # Run one task on a different LLM without restarting the stack
##   make task SIM=welding_cell_2_small TASK=car_door_spot_welding MODEL=sonnet-5
##
##   # Start the stack on a different default LLM
##   make sim SIM=welding_cell_2_small LLM_MODEL=opus-4.8
##
##   # Start on Zenoh, connecting to an existing broker (none is started here)
##   make sim SIM=welding_cell_2_small RMW=zenoh ZENOH_ROUTER=tcp/broker.plant.local:7447
##
##   # Run a task directly via Cell MCP (bypass orchestrator)
##   make task-mcp SIM=welding_cell_2_small TASK=car_door_spot_welding
##
##   # Replay a recorded session
##   make replay SIM=welding_cell_2_small TREE_ID=bt_20250101_001
##
## ─── Variables ────────────────────────────────────────────────────────────────
##
##   SIM                  Cell ID — directory under cells/      (default: welding_cell_2_small)
##   RVIZ                 RViz web viewer on :6080: 1 / 0        (default: on with
##                          ONLY=execution, off otherwise)
##                          1 | on | true | yes — start RViz
##                          0 | off             — headless (agents only)
##   SIM_BACKEND          [deprecated] use RVIZ instead;        (default: headless)
##                          SIM_BACKEND=rviz still enables RViz
##   TRAJECTORY_BACKEND   Planner's DEFAULT backend for requests  (default: ompl)
##                          naming none: ompl | rrt | ompl-compound
##   PLANNING_MODE        Planner's DEFAULT mode for requests     (default: classic)
##                          naming none: classic | coordinated | stepwise
##   BACKEND / MODE       Backend / mode of ONE request (plan,   (no default)
##                          task-mcp, task); beats the defaults
##   LOCKSTEP             1 = run ompl-compound steps in lockstep (default: 0)
##                          — barrier start, e-stop on drift
##   LLM_PROVIDER         bedrock | openai | ollama             (default: bedrock)
##   LLM_MODEL            Startup LLM for every agent           (default: opus-5)
##                          alias (opus-5 | opus-4.8 | sonnet-5 | haiku),
##                          a full model id, or provider:model
##   PLANNING_LLM_MODEL   LLM for trajectory planning only      (default: sonnet-5)
##                          — a request's own model still wins
##   MODEL                Per-call LLM override for task /       (no default)
##                          task-mcp / plan / replay-plan; beats
##                          LLM_MODEL for that one invocation
##   ROBOT_ID             Robot model ID                        (default: kr240r2900)
##   RMW                  ROS 2 middleware for the whole stack  (default: cyclonedds)
##                          cyclonedds — peer-to-peer, no broker
##                          zenoh      — brokered; requires ZENOH_ROUTER
##   ZENOH_ROUTER         Endpoint of the EXISTING Zenoh broker  (no default)
##                          to connect to, e.g.
##                          tcp/broker.plant.local:7447
##                          (comma-separate for several).
##                          No broker is ever started by this stack,
##                          so this is REQUIRED when RMW=zenoh.
##   TASK                 Task/program name to execute          (no default)
##   TREE_ID              Recording session ID for replay       (no default)
##   TOOLS_IMAGE          Tag of the Python runner image        (default: multi-robots-agents/tools:latest)
##                          Built from docker/Dockerfile.tools;
##                          every python3/pytest below runs in it.
##   MCP_HOST             Host the MCP client dials from inside (default: host.docker.internal)
##                          the runner. The stack publishes its
##                          agent ports on the host, so the
##                          client reaches them from outside the
##                          compose network. Override when the
##                          stack runs on another machine.

SIM                 ?= welding_cell_2_small
CELL_BASE           ?= ./cells
# Empty = the scope's default: ON for ONLY=execution (the viewer belongs to the
# execution side), OFF otherwise. RVIZ=1 / RVIZ=0 decide explicitly.
RVIZ                ?=
# ONLY=trajectory starts just what trajectory PLANNING needs, with no ROS / DDS
# at all: the planner (trajectory-agent, DDS_DISABLED=1) and the orchestrator
# (cell-data owner + A2A/MCP entry). No dashboard, no RViz, no robot agents,
# robot controllers or cell controller — plans are calculated and read over
# A2A/MCP, not executed.
# ONLY=execution starts what EXECUTION needs: the cell controller (executor +
# collision supervisor), every robot agent and robot controller of the cell, and
# the dashboard (web-panel, :8888) for the operator at the cell (+ rviz-web with
# RVIZ=1). No planner or orchestrator: the dashboard reaches them on the
# planning machine (ORCHESTRATOR_HOST / TRAJECTORY_AGENT_HOST).
# Empty = the full stack.
ONLY                ?=

# Deploying planning and execution on different machines: each side is told
# where the other one is. Defaults are the compose service names (one machine).
#   planning host:   make sim ONLY=trajectory CELL_CONTROLLER_HOST=<execution ip>
#   execution host:  make sim ONLY=execution  ORCHESTRATOR_HOST=<planning ip>
# The orchestrator reaches the cell controller on <ip>:8892 (MCP); the cell
# controller reads plans and cell config from the orchestrator on <ip>:8893.
CELL_CONTROLLER_HOST ?=
ORCHESTRATOR_HOST    ?=
# A loopback address means "this machine" to the operator, but inside a
# container 127.0.0.1 is the container itself: the cell controller would call
# its own :8893 and refuse every read. On one machine both scopes share the
# compose network, so a loopback value is read as "same machine" — the service
# name — and said so. Empty is exported as empty, which the compose file's
# `${VAR:-<service>}` treats as unset.
LOOPBACK_HOSTS := 127.% localhost 0.0.0.0 ::1
ifneq ($(filter $(LOOPBACK_HOSTS),$(CELL_CONTROLLER_HOST)),)
  $(info [sim] CELL_CONTROLLER_HOST=$(CELL_CONTROLLER_HOST) is this machine — using the compose service 'cell-controller')
  override CELL_CONTROLLER_HOST :=
endif
ifneq ($(filter $(LOOPBACK_HOSTS),$(ORCHESTRATOR_HOST)),)
  $(info [sim] ORCHESTRATOR_HOST=$(ORCHESTRATOR_HOST) is this machine — using the compose service 'orchestrator')
  override ORCHESTRATOR_HOST :=
endif
# The planner runs next to the orchestrator, so the execution side's dashboard
# and viewer look for it there unless told otherwise.
TRAJECTORY_AGENT_HOST ?= $(ORCHESTRATOR_HOST)
ifneq ($(filter $(LOOPBACK_HOSTS),$(TRAJECTORY_AGENT_HOST)),)
  $(info [sim] TRAJECTORY_AGENT_HOST=$(TRAJECTORY_AGENT_HOST) is this machine — using the compose service 'trajectory-agent')
  override TRAJECTORY_AGENT_HOST :=
endif
ifneq ($(TRAJECTORY_AGENT_HOST),)
  export TRAJECTORY_AGENT_HOST
endif
ifneq ($(CELL_CONTROLLER_HOST),)
  export CELL_CONTROLLER_HOST
endif
ifneq ($(ORCHESTRATOR_HOST),)
  export ORCHESTRATOR_HOST
endif

# The ONE image every agent container runs (planners, orchestrator, cell controller,
# every robot agent and controller, dashboard): containers are generic, the robot
# or planner they serve comes from their name and environment. Built once, by the
# trajectory-agent service; `make sim` builds it when it is missing.
AGENT_IMAGE         ?= multi-robots-agents/agent:latest
export AGENT_IMAGE
AGENT_ENSURE_IMAGE   = docker image inspect $(AGENT_IMAGE) >/dev/null 2>&1 || $(COMPOSE) build trajectory-agent
# Planner containers: each runs ONE planning run at a time; the orchestrator gives
# each new run to a free one (GET /a2a/planners). Planner 1 is trajectory-agent
# (:8890 on this machine); the others are trajectory-agent-2.. on the stack network.
PLANNERS            ?= 1
EXTRA_PLANNERS       = $(if $(filter-out 0 1,$(PLANNERS)),$(foreach i,$(shell seq 2 $(PLANNERS)),trajectory-agent-$(i)),)
SIM_BACKEND         ?= headless
TRAJECTORY_BACKEND  ?= ompl
# Out-of-repo trajectory backends: "name=module:attr,..." (docs/creating-a-backend.md).
# Exported so the compose file interpolates it at `up` time.
TRAJECTORY_BACKEND_PLUGINS ?=
export TRAJECTORY_BACKEND_PLUGINS
LLM_PROVIDER        ?= bedrock
LLM_MODEL           ?= opus-5
# Empty = the planner's own default (sonnet-5, agents/llm_model.py). Exported so
# the compose file interpolates it at `up` time, like DDS_DISABLED.
PLANNING_LLM_MODEL  ?=
export PLANNING_LLM_MODEL
# The planner's DEFAULT planning mode, for requests that name none (empty = the
# code default, classic). Interpolated at `up` time, like PLANNING_LLM_MODEL.
PLANNING_MODE       ?=
export PLANNING_MODE
# Execute ompl-compound steps in lockstep (1) or as plain Parallel legs (0).
# Read by the cell controller; exported so compose interpolates it at `up` time.
LOCKSTEP            ?= 0
export LOCKSTEP
# The policy trainer (agents/policy_trainer, profile "training"): TRAINER=1 starts it
# with `make sim` (full stack or ONLY=trajectory); `make trainer` starts it alone.
# Where its jobs run: POLICY_TRAINING_TARGET=local (default) | sagemaker; the latter
# also needs the SAGEMAKER_* / POLICY_S3_URI settings (agents/policy_trainer/config.py).
TRAINER             ?=
ifneq ($(filter 1 on true yes,$(TRAINER)),)
  TRAINER_PROFILE = --profile training
endif
# Which trained policy the roboballet backend uses (empty = "current").
ROBOBALLET_POLICY   ?=
export ROBOBALLET_POLICY
# Where training jobs run: local (in the trainer container, default) | sagemaker.
POLICY_TRAINING_TARGET ?= local
export POLICY_TRAINING_TARGET
# Local training on a GPU host: TRAINER_GPU=1 TRAINER_TORCH_INDEX=cu124 (rebuild).
TRAINER_GPU         ?=
export TRAINER_GPU
TRAINER_TORCH_INDEX ?= cpu
export TRAINER_TORCH_INDEX
# Host address the agents' published ports listen on (default 127.0.0.1: this
# machine only). :8892 and :8893 always listen on every interface.
PUBLISH_HOST        ?=
ifneq ($(PUBLISH_HOST),)
  export PUBLISH_HOST
endif
ROBOT_ID            ?= kr240r2900
RMW                 ?= cyclonedds
ZENOH_ROUTER        ?=

# ─── Python runner ────────────────────────────────────────────────────────────
#
# NOTHING here runs on the host interpreter. Every python3 / pytest invocation in
# this file goes through $(PY), which is a `docker run` into the small image built
# from docker/Dockerfile.tools. A clone plus Docker is the entire prerequisite
# list: no host Python, no venv, no pip, no uv.
#
# The repo is mounted at ITS OWN host path and -w is set to match, so relative and
# absolute paths behave inside exactly as they do outside — `scripts/x.py`,
# `./cells`, `> .generated_compose.yml` and an absolute SCENE_YAML= all resolve
# without rewriting. Anything the scripts create is written straight into the
# bind-mounted working tree, not into the container.
TOOLS_IMAGE         ?= multi-robots-agents/tools:latest
TOOLS_DOCKERFILE     = docker/Dockerfile.tools
TOOLS_REQUIREMENTS   = docker/requirements-tools.txt
TOOLS_STAMP          = .tools_image

# Host reachable from inside the runner. The stack publishes its agent ports on the
# host (8888-8892), so the MCP client dials back out rather than joining the compose
# network — that keeps `make task-mcp` working against a stack started any way at
# all, including one already running before this container existed.
MCP_HOST            ?= host.docker.internal

# -u: files land owned by the invoking user instead of root. --add-host: makes
# MCP_HOST resolve on native Linux (a no-op on Docker Desktop, which already
# provides it). No -t: a TTY would put CR into the compose file that `make sim`
# redirects to disk.
DOCKER_RUN           = docker run --rm \
                         -u $(shell id -u):$(shell id -g) \
                         -e PYTHONDONTWRITEBYTECODE=1 \
                         --add-host host.docker.internal:host-gateway \
                         -v $(CURDIR):$(CURDIR) \
                         -w $(CURDIR)

PY                   = $(DOCKER_RUN) $(TOOLS_IMAGE) python3
# -i for the targets that pipe curl output in (json.tool).
PY_STDIN             = $(DOCKER_RUN) -i $(TOOLS_IMAGE) python3

# The runner for anything that READS THE CELLS TREE off the host.
#
# CELL_BASE may point anywhere on the host, but $(CURDIR) above is the runner's
# only bind mount — so an out-of-tree cells tree is simply ABSENT inside the
# container. A script that probes it then gets "no such directory" for a cell that
# is sitting right there on the host, and because cell_layout.is_migrated() IS that
# probe (`<cell>/agentic_planner`.is_dir(), utils/cell_layout.py), the answer is not
# an error but a WRONG LAYOUT: every path resolves flat, and `make sim` reports
# "robots.yaml not found at <cell>/config/robots.yaml — convert the cell first" for a
# cell `make convert` converted correctly. Path.is_dir() swallows OSError, so "not
# mounted" and "not converted" are indistinguishable to the reader.
#
# So mount it, at its own absolute path, exactly as the `convert` target already
# does for the out-of-tree paths IT is handed (see MOUNTS there). Empty when the
# tree already lives under the repo mount, which leaves the in-tree default —
# CELL_BASE=./cells — invoking byte-identical docker commands.
#
# $(abspath) resolves a relative CELL_BASE against $(CURDIR) and normalizes `..`
# without touching the filesystem. It does NOT resolve symlinks, so a symlinked
# ./cells pointing out of tree is still unmounted — use CELL_BASE for that.
CELL_BASE_ABS       := $(abspath $(CELL_BASE))
CELL_MOUNT          := $(if $(filter $(CURDIR)/%,$(CELL_BASE_ABS)/),,-v $(CELL_BASE_ABS):$(CELL_BASE_ABS))
PY_CELL              = $(DOCKER_RUN) $(CELL_MOUNT) $(TOOLS_IMAGE) python3
# Socket passthrough, for the ONE target whose test drives docker itself. Scoped
# deliberately: a container with the docker socket is effectively root on the host,
# so it is not part of the default runner.
#
# --group-add 0: the socket is root:root mode 660, and the -u above is an unprivileged
# host uid with no matching group inside, so without this the CLI fails with
# "permission denied while trying to connect to the Docker daemon socket" — and the
# test READS THAT AS "no docker here" and skips, reporting success while testing
# nothing. Group membership only; the uid is unchanged, so output stays user-owned.
PY_DOCKER            = $(DOCKER_RUN) --group-add 0 \
                         -v /var/run/docker.sock:/var/run/docker.sock \
                         $(TOOLS_IMAGE) python3

# Build the runner once, then never again unless its inputs change. Same
# delete-to-invalidate idiom as compose-guard below, and for the same make 3.81
# mtime-granularity reason: the guard removes the stamp when the image is gone
# (pruned, or a fresh clone), rather than relying on file ordering.
.PHONY: tools-guard
tools-guard:
	@docker image inspect $(TOOLS_IMAGE) >/dev/null 2>&1 || rm -f $(TOOLS_STAMP)

$(TOOLS_STAMP): $(TOOLS_DOCKERFILE) $(TOOLS_REQUIREMENTS)
	@echo "[tools] building $(TOOLS_IMAGE) — one-off, cached afterwards"
	@docker build -f $(TOOLS_DOCKERFILE) -t $(TOOLS_IMAGE) $(dir $(TOOLS_DOCKERFILE))
	@date > $@

# Prerequisite list for every target that runs Python. Ordered first so the image
# exists before any recipe tries to use it.
TOOLS                = tools-guard $(TOOLS_STAMP)

.PHONY: tools
tools: $(TOOLS) ## Build the Python runner image (done automatically on first use)
	@echo "$(TOOLS_IMAGE) ready"

.PHONY: tools-clean
tools-clean: ## Remove the Python runner image
	@docker rmi -f $(TOOLS_IMAGE) 2>/dev/null; true
	@rm -f $(TOOLS_STAMP)

GENERATED_COMPOSE   = .generated_compose.yml
COMPOSE             = docker compose -f $(GENERATED_COMPOSE)

# The compose file is produced by a shell redirect, which creates the file before
# the generator can reject its inputs. Without this, a rejected run (unknown RMW,
# RMW=zenoh with no broker) would leave a truncated .generated_compose.yml behind
# for the next `docker compose -f` to act on. Fail closed: delete it.
.DELETE_ON_ERROR:

# Per-call model override, rendered for the two calling conventions. A literal
# comma cannot appear inside $(if ...) — make reads it as an argument separator —
# so the JSON separator is spelled as a variable.
COMMA               := ,
MODEL_JSON           = $(if $(MODEL),$(COMMA)"model":"$(MODEL)",)
MODEL_ARG            = $(if $(MODEL),--model $(MODEL),)
# Backend / mode of ONE planning request (plan, task-mcp, task): BACKEND= / MODE=,
# or TRAJECTORY_BACKEND= / PLANNING_MODE= given on this command line. Neither:
# nothing is sent and the planner applies its stack default, else ompl / classic.
REQ_BACKEND          = $(or $(BACKEND),$(if $(filter command line,$(origin TRAJECTORY_BACKEND)),$(TRAJECTORY_BACKEND)))
REQ_MODE             = $(or $(MODE),$(if $(filter command line,$(origin PLANNING_MODE)),$(PLANNING_MODE)))
# Backend options for the request, as JSON (BACKEND_OPTIONS='{"policy":"p1"}'); with
# BACKEND=roboballet, POLICY=<name> is the shorthand for {"policy": "<name>"}.
REQ_OPTIONS          = $(or $(BACKEND_OPTIONS),$(if $(and $(POLICY),$(filter roboballet,$(REQ_BACKEND))),{"policy":"$(POLICY)"},))
REQ_ARGS             = $(if $(REQ_BACKEND),--backend $(REQ_BACKEND),) $(if $(REQ_MODE),--mode $(REQ_MODE),) $(if $(REQ_OPTIONS),--backend-options '$(REQ_OPTIONS)',)
REQ_JSON             = $(if $(REQ_BACKEND),$(COMMA)"backend":"$(REQ_BACKEND)",)$(if $(REQ_MODE),$(COMMA)"mode":"$(REQ_MODE)",)$(if $(REQ_OPTIONS),$(COMMA)"backend_options":$(REQ_OPTIONS),)

# RViz is optional. Enable it with the dedicated RVIZ flag (RVIZ=1); the legacy
# SIM_BACKEND=rviz is still honoured for backward compatibility.
ifeq ($(RVIZ)$(ONLY),execution)
  RVIZ_ON = 1
else ifneq ($(filter 1 on true yes rviz,$(RVIZ) $(SIM_BACKEND)),)
  RVIZ_ON = 1
endif
ifneq ($(RVIZ_ON),)
  PROFILES  = --profile rviz
  RVIZ_STATE = on  (http://localhost:6080/vnc.html)
else
  PROFILES  =
  RVIZ_STATE = off
endif

ifeq ($(ONLY),trajectory)
  ifneq ($(PROFILES),)
    $(error ONLY=trajectory runs without RViz (no ROS graph) — drop RVIZ=1)
  endif
  SERVICES   = trajectory-agent$(if $(EXTRA_PLANNERS), $(EXTRA_PLANNERS),) orchestrator
  DOWN_SERVICES = $(SERVICES)
  TRAINER_SERVICE = $(if $(TRAINER_PROFILE),policy-trainer,)
  SCOPE_NOTE = trajectory planning only (no DDS, no dashboard, no robots)
  export DDS_DISABLED = 1
else ifeq ($(ONLY),execution)
  ifneq ($(TRAINER_PROFILE),)
    $(error TRAINER=1 belongs to the planning side — use it with ONLY=trajectory or the full stack)
  endif
  # The robot services are named after the cell's arms — read them from the
  # generated compose file rather than repeating the robot list here.
  # --no-deps: the dashboard declares depends_on orchestrator, which would
  # otherwise start a second orchestrator on the execution side. (Flags travel
  # in SERVICES so the other scopes' `up` lines stay exactly as they were.)
  SERVICES   = --no-deps $$($(COMPOSE) config --services | grep -E '^(cell-controller|robot-agent-|robot-controller-)' | tr '\n' ' ') $(if $(PROFILES),rviz-web,) web-panel
  # Down always takes the viewer with it: RViz belongs to the execution side.
  DOWN_SERVICES = $$($(COMPOSE) config --services | grep -E '^(cell-controller|robot-agent-|robot-controller-)' | tr '\n' ' ') web-panel rviz-web
  SCOPE_NOTE = execution only (cell controller + robots + dashboard; orchestrator at $(or $(ORCHESTRATOR_HOST),orchestrator))
  export DDS_DISABLED = 0
else ifeq ($(ONLY),)
  SERVICES   =
  SCOPE_NOTE = full stack
  export DDS_DISABLED = 0
else
  $(error ONLY=$(ONLY) is not a known scope — use ONLY=trajectory, ONLY=execution, or leave it empty)
endif

# Everything that changes the shape of the generated stack. A variable is not a
# file, so make cannot see that `make sim RMW=zenoh ...` differs from the plain
# `make sim` that produced the existing compose file — without this the stale file
# counts as up to date and the stack silently comes up on the previous transport.
COMPOSE_INPUTS = cell=$(SIM) cellbase=$(CELL_BASE) traj=$(TRAJECTORY_BACKEND) \
                 provider=$(LLM_PROVIDER) llm=$(LLM_MODEL) robot=$(ROBOT_ID) \
                 rmw=$(RMW) router=$(ZENOH_ROUTER) controller=$(ROBOT_CONTROLLER) \
                 planners=$(PLANNERS)
COMPOSE_STAMP  = .compose_settings

# Invalidation is by DELETING the generated file, not by listing the stamp as a
# prerequisite: make 3.81 (what macOS ships) compares mtimes at 1-second
# granularity, so a stamp rewritten in the same second as the compose file is not
# "newer" and the change is missed. Removing the file is unambiguous at any clock
# resolution. Ordered first among the prerequisites of the targets that start the
# stack, so the deletion happens before the compose file is considered.
.PHONY: compose-guard
compose-guard:
	@printf '%s\n' "$(COMPOSE_INPUTS)" | cmp -s - $(COMPOSE_STAMP) 2>/dev/null || { \
	  test ! -f $(GENERATED_COMPOSE) \
	    || echo "[generate] startup settings changed — regenerating compose"; \
	  rm -f $(GENERATED_COMPOSE); \
	  printf '%s\n' "$(COMPOSE_INPUTS)" > $(COMPOSE_STAMP); }

# Generate the cell-specific compose file.
# Runs when it is missing (see compose-guard) or the cell's robots.yaml changes.
# Everything that shapes the generated file: the generator and the modules it
# imports, and the cell's own robots.yaml / cell.yaml in BOTH layouts (pre-migration
# config/ and agentic_planner/). Without the generator here, a fix to it never
# reached a stack that already had a compose file; without agentic_planner/, a
# migrated cell gaining an arm kept the old robot services.
COMPOSE_DEPS = scripts/generate_compose.py utils/cell_layout.py agents/middleware.py \
               agents/llm_model.py \
               $(wildcard $(CELL_BASE)/$(SIM)/config/robots.yaml \
                          $(CELL_BASE)/$(SIM)/agentic_planner/config/robots.yaml \
                          $(CELL_BASE)/$(SIM)/agentic_planner/cell.yaml \
                          $(CELL_BASE)/$(SIM)/cell.yaml)
# Starting the stack regenerates it EVERY time, so a start always runs on the cell
# and the code as they are now (a few seconds; containers whose definition did not
# change are left running). Other targets reuse the file.
ifneq ($(filter sim sim-detached trainer build build-no-cache,$(MAKECMDGOALS)),)
  COMPOSE_DEPS += FORCE
endif
.PHONY: FORCE
FORCE:
$(GENERATED_COMPOSE): $(COMPOSE_DEPS) | $(TOOLS)
	@echo "[generate] Creating $(GENERATED_COMPOSE) for cell=$(SIM) base=$(CELL_BASE) llm=$(LLM_MODEL) rmw=$(RMW) ..."
	@# PY_CELL, not PY: the generator READS the cell off the host, so the tree has to
	@# be visible inside the runner or it resolves the pre-migration layout. See
	@# CELL_MOUNT above.
	@$(PY_CELL) scripts/generate_compose.py \
		--cell $(SIM) \
		--cell-base $(CELL_BASE) \
		--trajectory-backend $(TRAJECTORY_BACKEND) \
		--planners $(PLANNERS) \
		--llm-provider $(LLM_PROVIDER) \
		--llm-model $(LLM_MODEL) \
		--middleware $(RMW) \
		$(if $(ZENOH_ROUTER),--zenoh-router $(ZENOH_ROUTER),) \
		$(if $(ROBOT_ID),--robot-id $(ROBOT_ID),) \
		> $(GENERATED_COMPOSE)
	@echo "[generate] Robot namespaces:"
	@grep "container_name: robot-agent-" $(GENERATED_COMPOSE) \
		| awk '{print "  " $$2}' || true

# ─── Build ────────────────────────────────────────────────────────────────────

# IMAGES — build them BEFORE `make sim` / `make sim-detached`.
#
# `sim`/`sim-detached` run `up --no-build`: they NEVER build on the fly, so an
# unbuilt image fails fast (e.g. "No such image: <project>-robot-agent-robot1")
# instead of silently kicking off a multi-minute build. Run `make build` first.
#
# No service declares an explicit `image:`, so docker compose auto-names each
# one `<project>-<service>` from the shared docker/Dockerfile (rviz-web uses
# docker/Dockerfile.viz). Consequence to know about:
#   - The fixed services (trajectory-agent, orchestrator, cell-controller,
#     web-panel) keep their image across runs and cells.
#   - robot-agent-<id> / robot-controller-<id> are named PER ROBOT, and the
#     robot ids come from the cell's robots.yaml. A cell whose robot ids you
#     have never built lands on brand-new image names — so re-run `make build`
#     after switching to a cell with a new robot set (a plain `up` would report
#     "No such image" for the missing per-robot images). `docker compose build`
#     is layer-cached against the shared Dockerfile, so this is fast after the
#     first full build.
# `--profile rviz` so the OPTIONAL rviz-web image is built too. A plain
# `docker compose build` SKIPS profile-gated services, so without this the
# viewer image never exists and `make rviz` / `make sim RVIZ=1` fail later with
# "No such image: <project>-rviz-web" (rviz targets run `up --no-build`).
.PHONY: build
build: compose-guard $(GENERATED_COMPOSE) ## Build all Docker images (incl. the optional RViz viewer) for the current cell — run before `make sim`. SIM= CELL_BASE=
	$(COMPOSE) --profile rviz --profile training build

.PHONY: build-no-cache
build-no-cache: compose-guard $(GENERATED_COMPOSE) ## Build images without layer cache (incl. RViz viewer). SIM= CELL_BASE=
	$(COMPOSE) --profile rviz --profile training build --no-cache

# ─── Simulation ───────────────────────────────────────────────────────────────

.PHONY: sim
sim: compose-guard $(GENERATED_COMPOSE) ## Start simulation stack (attached). SIM= RVIZ=1 TRAJECTORY_BACKEND= RMW= ONLY=trajectory|execution CELL_CONTROLLER_HOST= ORCHESTRATOR_HOST=
	@echo "Starting multi_robots_agents — cell=$(SIM) scope=$(SCOPE_NOTE) rviz=$(RVIZ_STATE) trajectory=$(TRAJECTORY_BACKEND) rmw=$(RMW)$(if $(ZENOH_ROUTER), broker=$(ZENOH_ROUTER),)"
	@$(AGENT_ENSURE_IMAGE)
	$(if $(TRAINER_PROFILE),$(TRAINER_ENSURE_IMAGE))
	$(COMPOSE) $(PROFILES) $(TRAINER_PROFILE) up --no-build $(SERVICES) $(TRAINER_SERVICE)

.PHONY: sim-detached
sim-detached: compose-guard $(GENERATED_COMPOSE) ## Start simulation stack detached (background). SIM= RVIZ=1 ONLY=trajectory|execution CELL_CONTROLLER_HOST= ORCHESTRATOR_HOST=
	@echo "Starting multi_robots_agents (detached) — cell=$(SIM) scope=$(SCOPE_NOTE) rviz=$(RVIZ_STATE)"
	@$(AGENT_ENSURE_IMAGE)
	$(if $(TRAINER_PROFILE),$(TRAINER_ENSURE_IMAGE))
	$(COMPOSE) $(PROFILES) $(TRAINER_PROFILE) up -d --no-build $(SERVICES) $(TRAINER_SERVICE)
	@echo ""
	@echo "Stack is running. Use 'make logs' or 'make health' to inspect."
	@echo "Submit a task with:  make task SIM=$(SIM) TASK=<program_name>"

.PHONY: rviz
rviz: $(GENERATED_COMPOSE) ## Start/attach the RViz web viewer to a running stack (:6080). SIM=
	$(COMPOSE) --profile rviz up -d --no-build rviz-web
	@echo "RViz web viewer available at http://localhost:6080/vnc.html"

.PHONY: trainer
trainer: compose-guard $(GENERATED_COMPOSE) ## Start the RoboBallet policy trainer (MCP :8894). POLICY_TRAINING_TARGET=local|sagemaker
	$(TRAINER_ENSURE_IMAGE)
	$(COMPOSE) --profile training up -d --no-build policy-trainer
	@echo "Policy trainer: http://localhost:8894/health   MCP: http://localhost:8894/mcp"
	@echo "Training target: $(POLICY_TRAINING_TARGET)$(if $(filter sagemaker,$(POLICY_TRAINING_TARGET)), (SageMaker jobs: needs SAGEMAKER_ROLE_ARN + POLICY_S3_URI),)"

# The trainer image does not depend on the cell, so this needs no SIM and no
# compose file: a plain build, tagged as the compose service expects it.
TRAINER_IMAGE       ?= multi-robots-agents/policy-trainer:latest
export TRAINER_IMAGE
TRAINER_BUILD       = docker build -f docker/Dockerfile.trainer --build-arg TORCH_INDEX=$(TRAINER_TORCH_INDEX) -t $(TRAINER_IMAGE) docker/
# The stack starts with `up --no-build`; the trainer's image is built here, once,
# when it is missing (an explicit `make trainer-build` rebuilds it).
TRAINER_ENSURE_IMAGE = docker image inspect $(TRAINER_IMAGE) >/dev/null 2>&1 || $(TRAINER_BUILD)

.PHONY: trainer-build
trainer-build: ## Build only the trainer image (no SIM needed). TRAINER_TORCH_INDEX=cpu|cu124
	$(TRAINER_BUILD)

.PHONY: trainer-down
trainer-down: ## Stop and remove the policy trainer (policies and job records stay in their volumes)
	docker rm -f policy-trainer >/dev/null 2>&1 || true

.PHONY: logs-trainer
logs-trainer: ## Tail policy-trainer logs
	docker logs -f policy-trainer

# ─── Training (the policy trainer's MCP, :8894) ─────────────────────────────────
# Jobs run where the trainer's POLICY_TRAINING_TARGET says (local | sagemaker).
TRAIN_CELLS         ?= $(SIM)
EVAL_CELLS          ?=
TRAINER_MCP          = $(PY) scripts/trainer_mcp.py --host $(MCP_HOST)

.PHONY: train
train: $(TOOLS) ## Start a training job. TRAIN_CELLS=a,b (default SIM) EVAL_CELLS=c [STEPS=] [JOB=] [HP='{"n_envs":4}'] [DRY_RUN=1] [WATCH=1] [PROMOTE=1]
	@$(TRAINER_MCP) start --train "$(TRAIN_CELLS)" $(if $(EVAL_CELLS),--eval "$(EVAL_CELLS)",) \
		$(if $(STEPS),--steps $(STEPS),) $(if $(JOB),--job $(JOB),) $(if $(HP),--hp '$(HP)',) \
		$(if $(DRY_RUN),--dry-run,) $(if $(WATCH),--watch,) $(if $(PROMOTE),--promote,)

.PHONY: train-status
train-status: $(TOOLS) ## The trainer (target, config) or one job with JOB= (status, metrics)
	@$(TRAINER_MCP) $(if $(JOB),job --job $(JOB),status)

.PHONY: train-jobs
train-jobs: $(TOOLS) ## List the training jobs (local and SageMaker)
	@$(TRAINER_MCP) jobs

.PHONY: train-watch
train-watch: $(TOOLS) ## Follow a job until it ends. JOB=
	@test -n "$(JOB)" || (echo "ERROR: JOB= is required (see make train-jobs)" && exit 1)
	@$(TRAINER_MCP) watch --job $(JOB)

.PHONY: train-logs
train-logs: $(TOOLS) ## A job's training log (local log or CloudWatch). JOB= [LINES=50]
	@test -n "$(JOB)" || (echo "ERROR: JOB= is required (see make train-jobs)" && exit 1)
	@$(TRAINER_MCP) logs --job $(JOB) --lines $(or $(LINES),50)

.PHONY: train-stop
train-stop: $(TOOLS) ## Stop a running job. JOB=
	@test -n "$(JOB)" || (echo "ERROR: JOB= is required (see make train-jobs)" && exit 1)
	@$(TRAINER_MCP) stop --job $(JOB)

.PHONY: train-resume
train-resume: $(TOOLS) ## Continue an interrupted/stopped job from its checkpoint. JOB= (SageMaker: runs as JOB-r<n>)
	@test -n "$(JOB)" || (echo "ERROR: JOB= is required (see make train-jobs)" && exit 1)
	@$(TRAINER_MCP) resume --job $(JOB)

.PHONY: policies
policies: $(TOOLS) ## List the trained policies (and which is current)
	@$(TRAINER_MCP) policies

.PHONY: policy-import
policy-import: $(TOOLS) ## Install a completed job's policy. JOB= [POLICY=name] [PROMOTE=1]
	@test -n "$(JOB)" || (echo "ERROR: JOB= is required (see make train-jobs)" && exit 1)
	@$(TRAINER_MCP) import --job $(JOB) $(if $(POLICY),--name $(POLICY),) $(if $(PROMOTE),--promote,)

.PHONY: policy-eval
policy-eval: $(TOOLS) ## Score a policy offline on a cell. POLICY= [CELL=SIM] [EVAL_MODE=free|assigned]
	@test -n "$(POLICY)" || (echo "ERROR: POLICY= is required (see make policies)" && exit 1)
	@$(TRAINER_MCP) evaluate --policy $(POLICY) --cell $(or $(CELL),$(SIM)) --mode $(or $(EVAL_MODE),free)

.PHONY: policy-promote
policy-promote: $(TOOLS) ## Make a policy the one the roboballet backend plans with. POLICY=
	@test -n "$(POLICY)" || (echo "ERROR: POLICY= is required (see make policies)" && exit 1)
	@$(TRAINER_MCP) promote --policy $(POLICY)

.PHONY: down
down: ## Stop and remove project containers. Empty ONLY = ALL (incl. RViz, networks, orphans); ONLY=execution | ONLY=trajectory = that side only
ifeq ($(ONLY),)
	# The trainer is profile-gated too ("training") and would otherwise hold the
	# network; it is removed by name, before the stack.
	docker rm -f policy-trainer >/dev/null 2>&1 || true
	# --profile rviz ensures profile-gated containers are torn down too, otherwise
	# the rviz-web container lingers and blocks removal of the shared network.
	$(COMPOSE) --profile rviz down --remove-orphans
else
	# One side only: stop and remove its containers and leave the network, which
	# the other side may still be using.
	@echo "Stopping $(ONLY) side: $(SCOPE_NOTE)"
	$(COMPOSE) --profile rviz rm --stop --force $(DOWN_SERVICES)
endif

.PHONY: stop
stop: down ## Alias for 'down' — stop and remove all containers

.PHONY: restart
restart: down sim ## Stop and restart the full stack

# ─── Logs ─────────────────────────────────────────────────────────────────────

.PHONY: logs
logs: ## Tail logs for all services (or SERVICE=<name> for one)
	$(COMPOSE) logs -f $(SERVICE)

.PHONY: logs-traj
logs-traj: ## Tail trajectory-agent logs
	$(COMPOSE) logs -f trajectory-agent

.PHONY: logs-orch
logs-orch: ## Tail orchestrator logs
	$(COMPOSE) logs -f orchestrator

.PHONY: logs-cell
logs-cell: ## Tail cell-controller logs
	$(COMPOSE) logs -f cell-controller

.PHONY: logs-robots
logs-robots: $(GENERATED_COMPOSE) ## Tail all robot-agent logs (auto-detected from cell)
	$(COMPOSE) logs -f $$(grep "container_name: robot-agent-" $(GENERATED_COMPOSE) \
		| awk '{print $$2}' | tr '\n' ' ')

# ─── Health ───────────────────────────────────────────────────────────────────

.PHONY: health
health: $(TOOLS) ## Check health of all agent services
	@echo "=== Trajectory Agent  (MCP :8890) ===" \
	  && curl -sf http://localhost:8890/health | $(PY_STDIN) -m json.tool || echo "  OFFLINE"
	@echo "=== Orchestrator      (A2A :8889) ===" \
	  && curl -sf http://localhost:8889/health | $(PY_STDIN) -m json.tool || echo "  OFFLINE"
	@echo "=== Cell Controller   (MCP :8892) ===" \
	  && curl -sf http://localhost:8891/health | $(PY_STDIN) -m json.tool || echo "  OFFLINE"
	@echo "=== Dashboard         (HTTP :8888) ===" \
	  && curl -sf http://localhost:8888/health | $(PY_STDIN) -m json.tool || echo "  OFFLINE"
	@echo "=== Policy Trainer    (MCP :8894, optional) ===" \
	  && curl -sf http://localhost:8894/health | $(PY_STDIN) -m json.tool || echo "  not running (make trainer)"

# ─── Task management ──────────────────────────────────────────────────────────

.PHONY: tasks
tasks: $(TOOLS) ## List all task programs available in a cell. SIM=<cell_id>
	@echo "=== Tasks in cell '$(SIM)' ==="
	@curl -sf http://localhost:8889/a2a/tasks?cell_id=$(SIM) | $(PY_STDIN) -m json.tool \
	  || (echo "  Orchestrator offline — listing from filesystem:" && \
	      ls $(CELL_BASE)/$(SIM)/tasks/*.json 2>/dev/null | xargs -I{} basename {} .json || \
	      echo "  No tasks found in $(CELL_BASE)/$(SIM)/tasks/")

.PHONY: task
task: $(TOOLS) ## Run a task via A2A orchestrator. SIM= TASK=<program_name> [MODEL=]
	@test -n "$(TASK)" || (echo "ERROR: TASK= is required. Example: make task SIM=$(SIM) TASK=car_door_spot_welding" && exit 1)
	@echo "=== Submitting task '$(TASK)' to cell '$(SIM)' via A2A$(if $(MODEL), on $(MODEL),) ==="
	@curl -sf -X POST http://localhost:8889/a2a/run \
		-H "Content-Type: application/json" \
		-d '{"cell_id":"$(SIM)","program":"$(TASK)"$(MODEL_JSON)$(REQ_JSON)}' | $(PY_STDIN) -m json.tool

.PHONY: task-mcp
task-mcp: $(TOOLS) ## Plan then execute a task via MCP, bypassing the orchestrator. SIM= TASK= [PLAN_ONLY=1] [MODEL=] [BACKEND=] [MODE=]
	@test -n "$(TASK)" || (echo "ERROR: TASK= is required. Example: make task-mcp SIM=$(SIM) TASK=car_door_spot_welding" && exit 1)
	@echo "=== Submitting task '$(TASK)' via MCP (cell=$(SIM), backend=$(or $(REQ_BACKEND),planner default), mode=$(or $(REQ_MODE),planner default)$(if $(MODEL), model=$(MODEL),)) ==="
	@$(PY) scripts/cell_mcp.py --host $(MCP_HOST) \
		run-task --cell $(SIM) --task $(TASK) \
		$(REQ_ARGS) $(MODEL_ARG) $(if $(PLAN_ONLY),--plan-only,)

.PHONY: plan
plan: $(TOOLS) ## Plan a task WITHOUT moving any robot (read-only). SIM= TASK= [MODEL=] [BACKEND=] [MODE=] [POLICY=] [BACKEND_OPTIONS=]
	@test -n "$(TASK)" || (echo "ERROR: TASK= is required. Example: make plan SIM=$(SIM) TASK=car_door_spot_welding" && exit 1)
	@echo "=== Planning '$(TASK)' on cell '$(SIM)' (no execution$(if $(MODEL), model=$(MODEL),)) ==="
	@$(PY) scripts/cell_mcp.py --host $(MCP_HOST) \
		run-task --cell $(SIM) --task $(TASK) \
		$(REQ_ARGS) $(MODEL_ARG) --plan-only

.PHONY: replay-plan
replay-plan: $(TOOLS) ## Execute an already-calculated plan, no re-planning. SIM= [TREE_ID=] [MODEL=]
	@echo "=== Executing plan '$(if $(TREE_ID),$(TREE_ID),latest)' on cell '$(SIM)' ==="
	@$(PY) scripts/cell_mcp.py --host $(MCP_HOST) \
		replay-plan --cell $(SIM) $(if $(TREE_ID),--tree-id $(TREE_ID),) $(MODEL_ARG)

.PHONY: replay
replay: $(TOOLS) ## Replay a recorded session. SIM= TREE_ID=<recording_id>
	@test -n "$(TREE_ID)" || (echo "ERROR: TREE_ID= is required. Example: make replay SIM=$(SIM) TREE_ID=bt_20250101_001" && exit 1)
	@echo "=== Replaying recording '$(TREE_ID)' on cell '$(SIM)' ==="
	@$(PY) scripts/cell_mcp.py --host $(MCP_HOST) replay --cell $(SIM) --tree-id $(TREE_ID)

.PHONY: recordings
recordings: $(TOOLS) ## List all .rrd recordings for a cell. SIM=
	@echo "=== Recordings for cell '$(SIM)' ==="
	@$(PY) scripts/cell_mcp.py --host $(MCP_HOST) recordings --cell $(SIM)

# ─── MCP introspection ────────────────────────────────────────────────────────

.PHONY: list-tools
list-tools: $(TOOLS) ## List MCP tools on the trajectory agent
	@curl -sf -X POST http://localhost:8890/mcp \
		-H "Content-Type: application/json" \
		-d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | $(PY_STDIN) -m json.tool

.PHONY: list-cell-tools
list-cell-tools: $(TOOLS) ## List MCP tools on the cell controller
	@curl -sf -X POST http://localhost:8892/mcp \
		-H "Content-Type: application/json" \
		-d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | $(PY_STDIN) -m json.tool

# ─── Cell management ──────────────────────────────────────────────────────────

.PHONY: convert
convert: $(TOOLS) ## Convert a scene export to a cell (all config except tasks). Assets are REFERENCED, never copied; re-converting overwrites in place. OUT= SCENE_YAML= [SCENE=scene.usda] [ISAAC_ROOT=|ROBOT_MODEL=] [CELL_ID=] [TCP_OFFSET=] [OBSTACLE_MODE=voxel|aabb] [VOXEL_SIZE=0.05] [THERMAL_SPREAD=on|off] [STOP_REACTION_TIME=s STOP_DECEL=m/s2] [PATH_TOLERANCE=m]
	@test -n "$(OUT)"         || (echo "ERROR: OUT= (target cell directory, e.g. OUT=cells/my_cell) is required" && exit 1)
	@test -n "$(SCENE_YAML)"  || (echo "ERROR: SCENE_YAML= (path to scene.yaml) is required" && exit 1)
	@# The only target whose inputs routinely live OUTSIDE the repo: an Isaac export
	@# is wherever the user unpacked it, and OUT= may be out of tree too. The repo
	@# mount does not cover those, so each such path is mounted at its own absolute
	@# location — which is why the argument as typed still resolves inside.
	@MOUNTS=""; \
	for p in "$(SCENE_YAML)" "$(ISAAC_ROOT)" "$(OUT)"; do \
	  [ -n "$$p" ] || continue; \
	  if [ -d "$$p" ]; then d=$$(cd "$$p" && pwd); \
	  else d=$$(cd "$$(dirname "$$p")" 2>/dev/null && pwd) || continue; fi; \
	  case "$$d/" in $(CURDIR)/*) continue ;; esac; \
	  case " $$MOUNTS " in *" -v $$d:$$d "*) continue ;; esac; \
	  echo "[convert] mounting out-of-tree path $$d"; \
	  MOUNTS="$$MOUNTS -v $$d:$$d"; \
	done; \
	$(DOCKER_RUN) $$MOUNTS $(TOOLS_IMAGE) python3 scripts/convert_isaac_scene.py \
		--scene-yaml "$(SCENE_YAML)" \
		$(if $(SCENE),--scene "$(SCENE)",) \
		$(if $(ROBOT_MODEL),--robot-model "$(ROBOT_MODEL)",) \
		$(if $(ISAAC_ROOT),--isaacsim-root-cell-path "$(ISAAC_ROOT)",) \
		--out-dir "$(OUT)" \
		$(if $(CELL_ID),--cell-id $(CELL_ID),) \
		$(if $(TCP_OFFSET),--tcp-offset "$(TCP_OFFSET)",) \
		$(if $(OBSTACLE_MODE),--obstacle-mode $(OBSTACLE_MODE),) \
		$(if $(VOXEL_SIZE),--voxel-size $(VOXEL_SIZE),) \
		$(if $(THERMAL_SPREAD),--thermal-spread $(THERMAL_SPREAD),) \
		$(if $(STOP_REACTION_TIME),--stop-reaction-time $(STOP_REACTION_TIME),) \
		$(if $(STOP_DECEL),--stop-decel $(STOP_DECEL),) \
		$(if $(PATH_TOLERANCE),--path-tolerance $(PATH_TOLERANCE),)

.PHONY: urdfs
urdfs: $(TOOLS) ## Generate URDF files for a cell. SIM=
	$(PY) scripts/generate_cell_urdfs.py --cell $(SIM)

.PHONY: compose
compose: compose-guard ## Generate docker-compose for a cell (auto-called by make sim). SIM= LLM_MODEL= RMW=
	@rm -f $(GENERATED_COMPOSE)
	$(MAKE) $(GENERATED_COMPOSE) SIM=$(SIM) TRAJECTORY_BACKEND=$(TRAJECTORY_BACKEND) \
		LLM_PROVIDER=$(LLM_PROVIDER) LLM_MODEL=$(LLM_MODEL) ROBOT_ID=$(ROBOT_ID) \
		RMW=$(RMW) ZENOH_ROUTER=$(ZENOH_ROUTER)
	@echo "Generated $(GENERATED_COMPOSE) for cell=$(SIM)"
	@echo "Robots detected:"
	@grep "container_name: robot-agent-" $(GENERATED_COMPOSE) \
		| awk '{print "  " $$2}' || echo "  (none)"

.PHONY: compose-show
compose-show: $(GENERATED_COMPOSE) ## Print the generated compose file. SIM=
	@cat $(GENERATED_COMPOSE)

# ─── Tests ────────────────────────────────────────────────────────────────────

.PHONY: test
test: $(TOOLS) ## Run all tests
	$(PY) -m pytest tests/ -v --tb=short

.PHONY: test-trajectory
test-trajectory: $(TOOLS) ## Run trajectory tool tests only
	$(PY) -m pytest tests/test_trajectory_tools.py -v --tb=short

.PHONY: test-cell
test-cell: $(TOOLS) ## Run cell controller tests only
	$(PY) -m pytest tests/cell/ -v --tb=short

.PHONY: test-ompl-container
test-ompl-container: $(TOOLS) ## Build the focused backend image and run the OMPL container test (backend in its own container, client in the runner)
	docker build -f docker/Dockerfile.ompltest -t multi-robots-agents/ompltest:latest .
	@# The only target given the docker socket: this test drives `docker run` itself,
	@# and skips silently when it cannot find a docker CLI or daemon. See PY_DOCKER.
	$(PY_DOCKER) -m pytest tests/test_ompl_backend_container.py -v -s --tb=short

# ─── Clean ────────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Stop containers and remove cache/bytecode
	$(COMPOSE) down -v --remove-orphans
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'
