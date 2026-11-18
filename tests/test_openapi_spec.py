"""Hold ``docs/api/openapi.yaml`` and the two servers that implement it together.

A dimension nobody measures is the one that drifts, so the spec has one job here:
every route the orchestrator and the dashboard actually serve appears in the
spec, and every path in the spec is served by exactly one of them. When someone
adds a route without a matching entry (or renames one and forgets), this test
names both the file and the mismatch.

Extraction is by regex, not by importing the server: the host has no flask, and
requiring it would move the baseline from 14-failed/14-errors into new failures
whenever a machine ran the test outside the container.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "docs" / "api" / "openapi.yaml"

ORCH_SERVER = REPO_ROOT / "agents" / "orchestrator" / "server.py"
DASH_SERVER = REPO_ROOT / "dashboard" / "web_control_panel.py"

_ROUTE_RE = re.compile(r'@app\.route\(\s*[\'"]([^\'"]+)[\'"]')


def _routes(path: Path) -> set[str]:
    """Every `@app.route("...")` declaration in a Flask server file.

    Regex, not import: the goal is to compare the shape the operator sees to
    the shape the spec claims — which is a static-file property, not a runtime
    one. Importing would drag flask into a host that doesn't ship it.
    """
    return set(_ROUTE_RE.findall(path.read_text()))


def _flask_to_openapi(route: str) -> str:
    """Flask ``<name>`` / ``<int:name>`` → OpenAPI ``{name}``."""
    return re.sub(r"<(?:[^:>]+:)?([^>]+)>", r"{\1}", route)


def _spec() -> dict:
    doc = yaml.safe_load(SPEC_PATH.read_text())
    assert isinstance(doc, dict)
    return doc


def test_spec_file_exists() -> None:
    assert SPEC_PATH.is_file(), (
        f"OpenAPI spec is missing at {SPEC_PATH.relative_to(REPO_ROOT)}")


def test_spec_declares_openapi_31_and_two_servers() -> None:
    doc = _spec()
    assert doc.get("openapi", "").startswith("3."), \
        f"expected OpenAPI 3.x, got {doc.get('openapi')!r}"
    servers = doc.get("servers") or []
    assert len(servers) >= 2, \
        "spec must document both the dashboard (:8888) and the orchestrator (:8889)"


def test_every_orchestrator_route_is_in_the_spec() -> None:
    served = {_flask_to_openapi(r) for r in _routes(ORCH_SERVER)}
    documented = set((_spec().get("paths") or {}).keys())
    missing = served - documented
    assert not missing, (
        f"orchestrator serves routes not documented in openapi.yaml: {sorted(missing)}")


def test_every_dashboard_route_is_in_the_spec() -> None:
    served = {_flask_to_openapi(r) for r in _routes(DASH_SERVER)}
    documented = set((_spec().get("paths") or {}).keys())
    missing = served - documented
    assert not missing, (
        f"dashboard serves routes not documented in openapi.yaml: {sorted(missing)}")


def test_every_spec_path_is_served_by_one_of_the_two_servers() -> None:
    served = {_flask_to_openapi(r) for r in _routes(ORCH_SERVER) | _routes(DASH_SERVER)}
    documented = set((_spec().get("paths") or {}).keys())
    orphaned = documented - served
    assert not orphaned, (
        "openapi.yaml documents paths no server serves — either the routes were "
        f"removed or the spec is out of date: {sorted(orphaned)}")


def test_spec_declares_the_run_envelope_shape() -> None:
    """`RunEnvelope` is the shared shape both async dispatch and result polls
    return; a caller writing to the spec relies on it being defined."""
    schemas = ((_spec().get("components") or {}).get("schemas") or {})
    env = schemas.get("RunEnvelope")
    assert env is not None, "components.schemas.RunEnvelope missing"
    props = env.get("properties") or {}
    for key in ("run_id", "cell_id", "status", "done"):
        assert key in props, f"RunEnvelope missing required field {key!r}"
    status = props["status"]
    assert set(status.get("enum") or []) == {"running", "succeeded", "failed"}, \
        f"RunEnvelope.status enum drift: {status.get('enum')}"


def test_spec_documents_409_for_cell_busy_on_every_motion_endpoint() -> None:
    """Refused-because-busy is a *safety* answer, not availability — every
    endpoint that starts a run must say so, so a caller cannot handle "cell
    busy" for one route and swallow it silently for another."""
    doc = _spec()
    motion_paths = ["/a2a/run", "/a2a/replay", "/a2a/move", "/a2a/task/submit"]
    for path in motion_paths:
        item = (doc.get("paths") or {}).get(path)
        assert item is not None, f"spec is missing path {path}"
        responses = (item.get("post") or {}).get("responses") or {}
        assert "409" in responses, f"{path} must document a 409 (CellBusy)"
