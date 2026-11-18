#!/bin/bash
# RViz-in-the-browser entrypoint.
#
# Stack:  Xvfb (virtual display) -> x11vnc (VNC on :5900) -> websockify/noVNC
#         (web on :80) ; launch_viz.py runs rviz2 + the joint/marker relay.
#
# Deliberately NO `set -e`: this is a supervisor. A single failed command
# (or a component dying) must NOT kill the container — we detect and restart.
#
# The previous version started everything once with `&` and never recovered.
# On a `docker restart` the container filesystem is reused, so a stale
# /tmp/.X0-lock from the prior run made Xvfb die with "Server is already
# active for display 0"; x11vnc + rviz2 then had no display, so noVNC showed
# a black screen with "Connection refused" to :5900. This version cleans the
# stale lock and keeps every component alive.

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPY="${DISPLAY_NUM:-0}"
export DISPLAY=":${DPY}"

source /opt/ros/jazzy/setup.bash 2>/dev/null || true

XVFB_PID=""

ensure_x() {
  # Healthy if the X socket exists and our Xvfb is still alive.
  if [ -S "/tmp/.X11-unix/X${DPY}" ] && [ -n "${XVFB_PID}" ] && kill -0 "${XVFB_PID}" 2>/dev/null; then
    return 0
  fi
  # Remove stale lock/socket left by a previous run (the docker-restart bug).
  rm -f "/tmp/.X${DPY}-lock" "/tmp/.X11-unix/X${DPY}" 2>/dev/null || true
  echo "[RVizWeb] starting Xvfb on :${DPY}"
  Xvfb ":${DPY}" -screen 0 "${RESOLUTION:-1280x900x24}" -nolisten tcp &
  XVFB_PID=$!
  # Wait until the display is actually accepting connections.
  for _ in $(seq 1 40); do
    [ -S "/tmp/.X11-unix/X${DPY}" ] && { echo "[RVizWeb] Xvfb ready on :${DPY}"; return 0; }
    sleep 0.25
  done
  echo "[RVizWeb] WARNING: Xvfb did not become ready on :${DPY}"
  return 1
}

ensure_x11vnc() {
  pgrep -f "x11vnc.*rfbport 5900" >/dev/null 2>&1 && return 0
  echo "[RVizWeb] starting x11vnc on :5900"
  x11vnc -display ":${DPY}" -nopw -forever -shared -rfbport 5900 -bg -o /tmp/x11vnc.log >/dev/null 2>&1
}

ensure_websockify() {
  pgrep -f "websockify" >/dev/null 2>&1 && return 0
  echo "[RVizWeb] starting noVNC (websockify :${NOVNC_PORT:-80} -> :5900)"
  websockify --web /usr/share/novnc "${NOVNC_PORT:-80}" localhost:5900 &
}

ensure_viz() {
  pgrep -f "launch_viz.py" >/dev/null 2>&1 && return 0
  local launcher="${SELF_DIR}/launch_viz.py"
  if [ -f "${launcher}" ]; then
    echo "[RVizWeb] starting viz launcher"
    python3 "${launcher}" &
  else
    echo "[RVizWeb] launch_viz.py not found at ${launcher}; starting bare rviz2"
    rviz2 &
  fi
}

echo "[RVizWeb] noVNC available at http://localhost:${NOVNC_PORT:-80}/vnc.html"

# Supervisor loop: keep the display, VNC bridge, web proxy and viz alive.
while true; do
  ensure_x
  ensure_x11vnc
  ensure_websockify
  ensure_viz
  sleep 5
done
