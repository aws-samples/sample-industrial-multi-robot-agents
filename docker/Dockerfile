## multi_robots_agents — single-stage image (local build only)
## ROS 2 Jazzy LTS (Ubuntu 24.04 Noble), ARM64 + AMD64
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CELL_BASE_PATH=/cells \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# System deps — Ubuntu 24.04 + the RMWs + control_msgs for ROS 2
# BOTH middlewares are installed because the transport is a startup decision
# (RMW=cyclonedds | zenoh, see agents/middleware.py), not a build-time one — the
# image must be able to honour either without a rebuild. rmw_zenoh_cpp is the
# client side only; the router it connects to is the operator's, running outside
# this stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-numpy \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rmw-zenoh-cpp \
    ros-jazzy-control-msgs \
    ros-jazzy-trajectory-msgs \
    ros-jazzy-pinocchio \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Python packages baked in — never reinstalled at runtime
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --ignore-installed -r /tmp/requirements.txt

# ROS support packages. Kept in a SEPARATE layer after the pip install so
# editing this list does not invalidate the (slow) pip layer.
#   ompl                              OMPL C++ core (standalone, no planning framework)
#   robot-state-publisher / joint-state-publisher  TF + current-state source
#   xacro                             URDF preprocessing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ompl \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# OMPL (Open Motion Planning Library) — Python bindings for the 'ompl'
# trajectory backend (agents/trajectory/backends/ompl/): native
# RRTConnect/BIT* planners + PathSimplifier over the Pinocchio collision model.
# The C++ core ships with ros-jazzy-ompl above; this adds the standalone
# nanobind-based Python module published on PyPI.
# Kept RESILIENT and in its own layer: if no wheel is available for this
# platform/Python, the image still builds and the 'ompl' backend reports a clear
# 'unavailable' error at runtime (the legacy 'rrt' backend is unaffected).
RUN pip3 install --no-cache-dir ompl \
    || echo "[build] WARNING: 'ompl' Python wheel unavailable for this platform; \
the ompl trajectory backend will be disabled at runtime (legacy rrt still works)."

# Working directory — agent code volume-mounted here at runtime
WORKDIR /workspace/multi_robots_agents
ENV PYTHONPATH=/workspace/multi_robots_agents:$PYTHONPATH

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'Specify CMD in docker-compose service'"]
