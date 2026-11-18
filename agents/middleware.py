"""The one place that answers "which middleware carries this cell's traffic?".

The stack was wired to Eclipse Cyclone DDS in three places at once — the base
``docker-compose.yml``, the generator that supersedes it, and the Dockerfile that
installs the RMW — each spelling ``rmw_cyclonedds_cpp`` as a literal. Nothing was
wrong with that choice; what was wrong is that it was not a choice. A cell on a
plant network where multicast is filtered, or one that has to federate with a
site-wide broker, could not be started differently without editing the generator.

This module makes the transport a **startup decision**, resolved once and written
into every container's environment, exactly as ``LLM_MODEL`` is:

    make sim SIM=welding_cell_2_small                          # CycloneDDS (unchanged)
    make sim SIM=welding_cell_2_small RMW=zenoh \
             ZENOH_ROUTER=tcp/broker.plant.local:7447     # Zenoh, external broker

Why a broker is a different shape of thing
------------------------------------------
CycloneDDS is peer-to-peer: every node discovers every other node and there is no
component to point at. Zenoh is *brokered* — sessions rendezvous at a router — so
selecting it raises a second question that DDS never asks: **which broker?**

The answer is always the operator's. This stack does **not** run a router: it is a
client of a broker that already exists on the plant network. So ``ZENOH_ROUTER``
is required whenever ``RMW=zenoh``, and no router service is ever generated.

``connect/endpoints`` is therefore always set explicitly. The rmw_zenoh default is
``tcp/localhost:7447``, which is correct on one host and silently wrong from inside
a container — every agent would sit in its own island and the cell would look up
while no topic crossed between arms. An unset endpoint is not a usable default
here, so there isn't one.

Fail-closed, per the repo's governing principle: an unrecognised middleware, a
malformed endpoint, or Zenoh with no broker named raises before a compose file is
written, rather than producing a stack that comes up on the wrong transport or
talking to nothing. An availability failure (the stack refuses to start) is
acceptable; a cell whose arms cannot hear each other is not.

Note on what is NOT emitted: in Zenoh mode ``CYCLONEDDS_URI`` is left out
entirely. Emitting it would create configuration that looks live and is not —
the same defect as a cell declaring ``check_frequency: 100.0`` at a monitor that
reads a literal 25 Hz.

Deliberately dependency-light: standard library only, so the compose generator
can import it in a bare shell with no ROS on the PATH.
"""
from __future__ import annotations

import os
import re
from typing import Optional

# ─── The catalog ──────────────────────────────────────────────────────────────
# Canonical names an operator can type, mapped to the RMW implementation that
# ROS 2 loads. Add a middleware by adding a row here and installing its RMW in
# docker/Dockerfile — not by editing the generator.
RMW_IMPLEMENTATIONS: dict[str, str] = {
    "cyclonedds": "rmw_cyclonedds_cpp",
    "zenoh":      "rmw_zenoh_cpp",
}

VALID_MIDDLEWARES: tuple[str, ...] = tuple(RMW_IMPLEMENTATIONS)

DEFAULT_MIDDLEWARE = "cyclonedds"

# Spellings accepted for each canonical name, so `RMW=rmw_zenoh_cpp` and
# `RMW=Zenoh` both work and neither silently falls through to the default.
MIDDLEWARE_ALIASES: dict[str, str] = {
    "cyclonedds":        "cyclonedds",
    "cyclone":           "cyclonedds",
    "cyclone_dds":       "cyclonedds",
    "dds":               "cyclonedds",
    "rmw_cyclonedds_cpp": "cyclonedds",
    "zenoh":             "zenoh",
    "rmw_zenoh_cpp":     "zenoh",
}

# Middlewares that rendezvous at a router rather than discovering peer-to-peer.
# Membership here is what makes a broker endpoint mandatory at all.
BROKERED_MIDDLEWARES = frozenset({"zenoh"})

# The port a ROS 2 Zenoh router listens on, appended when an operator gives a
# bare host. Only ever used to complete an endpoint they supplied.
DEFAULT_ZENOH_PORT = 7447

# Zenoh transport prefixes. An endpoint that already names one is passed through
# as the operator wrote it; a bare host gets tcp/, which is what a ROS 2 router
# listens on by default.
_ZENOH_PROTOCOLS = ("tcp", "udp", "tls", "quic", "ws", "wss", "unixsock-stream",
                    "serial", "vsock")

# An endpoint is interpolated into a shell-quoted compose value and into a JSON
# array, so it is restricted to what a host/port can legitimately contain. This
# is the injection boundary, not a cosmetic check.
_ENDPOINT_SAFE = re.compile(r"^[A-Za-z0-9_.:/\[\]%-]+$")


# ─── Resolution ───────────────────────────────────────────────────────────────

def normalize_middleware(value: Optional[str]) -> Optional[str]:
    """Canonical middleware name, or ``None`` when nothing was specified.

    Raises ``ValueError`` on an unrecognised name. Returning the default here
    instead would mean a typo (``RMW=zenow``) started the stack on CycloneDDS
    while the operator believed it was on Zenoh — the failure would surface as
    "the external broker sees no traffic", a long way from its cause.
    """
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    try:
        return MIDDLEWARE_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown middleware {value!r}; expected one of "
            f"{', '.join(VALID_MIDDLEWARES)}"
        ) from None


def resolve_middleware(explicit: Optional[str] = None) -> str:
    """Which middleware to start on, highest precedence first.

    1. ``explicit`` — the ``--middleware`` flag (i.e. ``make sim RMW=...``).
    2. ``RMW`` — the environment variable an operator exports.
    3. ``RMW_IMPLEMENTATION`` — honoured so a shell already configured for ROS
       (where this variable is the conventional one) does not get overridden by
       a default that contradicts it.
    4. CycloneDDS, the historical behaviour.
    """
    return (normalize_middleware(explicit)
            or normalize_middleware(os.environ.get("RMW"))
            or normalize_middleware(os.environ.get("RMW_IMPLEMENTATION"))
            or DEFAULT_MIDDLEWARE)


def rmw_implementation(middleware: str) -> str:
    """The ``RMW_IMPLEMENTATION`` value ROS 2 loads for this middleware."""
    return RMW_IMPLEMENTATIONS[normalize_middleware(middleware)]


def uses_broker(middleware: str) -> bool:
    """True when this middleware rendezvous at a router."""
    return normalize_middleware(middleware) in BROKERED_MIDDLEWARES


# ─── Router endpoints ─────────────────────────────────────────────────────────

def normalize_router_endpoint(value: Optional[str]) -> Optional[str]:
    """One or more Zenoh endpoints in canonical ``proto/host:port`` form.

    Accepts what an operator would plausibly type — ``broker.plant.local``,
    ``10.0.0.5:7447``, ``tcp/10.0.0.5:7447`` — and a comma-separated list of
    them, because a production broker is usually more than one address and a
    session may list several to connect to.

    Returns ``None`` for "nothing specified"; under Zenoh that is a fatal
    condition, caught by :func:`router_endpoints`. Raises ``ValueError`` on
    anything that is not a plausible endpoint, rather than passing it through
    to be interpolated into a config.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    out: list[str] = []
    for part in raw.split(","):
        ep = part.strip()
        if not ep:
            continue
        if not _ENDPOINT_SAFE.match(ep):
            raise ValueError(
                f"malformed Zenoh endpoint {ep!r}; expected proto/host:port "
                f"(e.g. tcp/broker.plant.local:{DEFAULT_ZENOH_PORT})"
            )
        proto, sep, rest = ep.partition("/")
        if sep and proto.lower() in _ZENOH_PROTOCOLS:
            # Operator named the transport; only supply the port if absent.
            ep = f"{proto.lower()}/{_with_default_port(rest)}"
        else:
            ep = f"tcp/{_with_default_port(ep)}"
        out.append(ep)

    if not out:
        return None
    return ",".join(out)


def _with_default_port(host: str) -> str:
    """``host`` unchanged if it already carries a port, else with the default.

    Bracketed IPv6 (``[::1]``) is handled by looking for the colon AFTER the
    closing bracket — a bare ``::1`` has colons that are not port separators.
    """
    tail = host[host.rindex("]") + 1:] if "]" in host else host
    if ":" in tail:
        return host
    return f"{host}:{DEFAULT_ZENOH_PORT}"


def router_endpoints(middleware: str, router: Optional[str] = None) -> Optional[str]:
    """The broker endpoints agents connect to, or ``None`` for a peer-to-peer RMW.

    Raises ``ValueError`` when a brokered middleware was selected without an
    endpoint. There is deliberately no fallback: rmw_zenoh's own default of
    ``tcp/localhost:7447`` resolves inside each container to itself, so a stack
    started that way comes up looking healthy with every agent isolated. Better
    to refuse to start than to hide that.
    """
    if not uses_broker(middleware):
        return None
    endpoints = normalize_router_endpoint(
        router if router is not None else os.environ.get("ZENOH_ROUTER"))
    if not endpoints:
        raise ValueError(
            f"middleware {normalize_middleware(middleware)!r} is brokered and this "
            f"stack does not run a router — name the existing broker with "
            f"ZENOH_ROUTER (e.g. ZENOH_ROUTER=tcp/broker.plant.local:"
            f"{DEFAULT_ZENOH_PORT})"
        )
    return endpoints


# ─── Environment ──────────────────────────────────────────────────────────────

def middleware_env(middleware: str, cell_id: str, *,
                   router: Optional[str] = None,
                   session_config_uri: Optional[str] = None,
                   dds_config_uri: Optional[str] = None) -> dict:
    """The transport environment every container in the cell receives.

    Only the selected middleware's variables are emitted. Configuration for a
    middleware that is not running would read as live and would not be — the
    reason ``CYCLONEDDS_URI`` is absent under Zenoh.

    ``dds_config_uri`` names the cell's ``cyclonedds.xml``. Pass it: a converted
    cell keeps its config under ``agentic_planner/config/`` and only a filesystem
    probe can tell (``cell_layout.config_uri``), which this module deliberately
    does not do — it is imported by containers that have no cells tree mounted.
    The cell-root path below is the fallback for callers that cannot probe.
    """
    mw = normalize_middleware(middleware)
    if mw is None:
        mw = DEFAULT_MIDDLEWARE
    env = {"RMW_IMPLEMENTATION": RMW_IMPLEMENTATIONS[mw]}

    if mw == "cyclonedds":
        # Per-cell DDS config, resolved from the cell name under CELL_BASE_PATH.
        env["CYCLONEDDS_URI"] = (
            dds_config_uri or f"file:///cells/{cell_id}/config/cyclonedds.xml")
        return env

    if mw == "zenoh":
        # Raises if no broker was named — see router_endpoints().
        eps = router_endpoints(mw, router)
        # JSON array, the form rmw_zenoh's override parser expects. Always set:
        # see the module docstring on why localhost is not a usable default
        # from inside a container.
        env["ZENOH_CONFIG_OVERRIDE"] = (
            "connect/endpoints=[" + ",".join(f'"{e}"' for e in eps.split(",")) + "]"
        )
        # A full session config is the escape hatch for anything the override
        # cannot express (TLS material, scouting, access control). Emitted only
        # when the cell actually ships one, so it never points at a missing file.
        if session_config_uri:
            env["ZENOH_SESSION_CONFIG_URI"] = session_config_uri
        return env

    return env


def describe(middleware: str, router: Optional[str] = None) -> str:
    """One line for the generated compose header and the Makefile banner."""
    mw = normalize_middleware(middleware) or DEFAULT_MIDDLEWARE
    if not uses_broker(mw):
        return f"{mw} ({RMW_IMPLEMENTATIONS[mw]}, peer-to-peer — no broker)"
    eps = router_endpoints(mw, router)
    return f"{mw} ({RMW_IMPLEMENTATIONS[mw]}) via external broker at {eps}"
