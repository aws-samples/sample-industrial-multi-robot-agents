"""The one door to a cell's *derived* data — local disk, or the orchestrator's MCP.

Ask the owning agent, not the volume. ``dashboard/web_control_panel.py`` already
learned that lesson: a second independent reader of the shared ``/cells`` mount is
how two readers come to disagree about the same cell (a recording list globbing a
filename pattern no writer had produced for months, a task summary in a shape no
writer had produced since the format changed). The dashboard was moved onto MCP;
the trajectory planner was still the *other* writer on that mount.

So this module is the planner's door, and it has two implementations behind one
API:

  * ``local``        — resolve through :mod:`utils.cell_layout` and use the
    filesystem. This is what the **orchestrator** does: it owns ``/cells``, and
    every reader below it (its own ``/a2a/*`` routes, the MCP tools in
    :mod:`agents.orchestrator.cell_data_mcp`) is that one owner reading its own
    disk. It is also what a test or a host script does, so nothing needs a running
    service to exercise a store function.
  * ``orchestrator`` — a JSON-RPC ``tools/call`` to the orchestrator's cell-data
    MCP endpoint. This is what the **planner** does: it holds no cell data of its
    own, so a plan, a trace, an obstacle model or a task program is read from and
    written to the agent that owns them.

Selected by ``CELL_DATA_MODE`` (default ``local``, i.e. today's behaviour), so
importing this module changes nothing until a service opts in. The planner's
container sets ``CELL_DATA_MODE=orchestrator`` and mounts ``/cells`` read-only —
belt and braces: the mode routes the traffic, the mount makes a missed call site
fail loudly with ``EROFS`` instead of silently writing a second copy of the truth.

**Fail-closed.** In ``orchestrator`` mode a transport failure raises
:class:`CellStoreUnavailable`, never "the file is absent". The difference matters
most for exactly one file: an unreachable orchestrator that reported "no
obstacles" would have the planner route arms through the fixtures it cannot see.
An availability failure (no plan) is acceptable; that one is not.

What is deliberately NOT here: the scene bundle. URDF, SRDF, meshes and
``scene.yaml`` stay a read-only mount, because mesh URIs are baked into the
published ``robot_description`` and resolved by every DDS subscriber and by RViz
(see :mod:`utils.cell_layout`), and a cell is 33–973 MB of geometry. Shared,
immutable *input* is a mount; derived, mutable *output* goes through its owner.

Stdlib only, like :mod:`utils.cell_layout` — this is on the import path of
scripts that run before dependencies are installed, so the transport is
``urllib`` and YAML/JSON parsing stays with the caller (except for the two
convenience helpers at the bottom, which import lazily).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from utils import cell_layout

# ── the namespace of derived cell data ───────────────────────────────────────
#
# A fixed vocabulary of (kind, name) pairs rather than free paths: the planner
# never names a filesystem location, so no call can walk out of the cell, and the
# orchestrator can validate every request against this table alone.
KIND_CONFIG = "config"          # robots.yaml, obstacles.yaml, operations.yaml …
KIND_MANIFEST = "manifest"      # cell.yaml
KIND_TASKS = "tasks"            # <program>.json
KIND_PLANS = "plans"            # <tree_id>.json  (assembled BehaviourTrees)
KIND_LOGS = "logs"              # run logs, plan traces, plan events, quality logs
KIND_RECORDINGS = "recordings"  # trajectory_index.json (the .rrd files are the cell agent's)

KINDS = (KIND_CONFIG, KIND_MANIFEST, KIND_TASKS, KIND_PLANS, KIND_LOGS,
         KIND_RECORDINGS)

MODE_LOCAL = "local"
MODE_ORCHESTRATOR = "orchestrator"

# One name, one file, no traversal. Same expression as plan_trace.is_safe_id and
# plan_checkpoint.is_safe_id, applied one layer earlier so every artefact family
# inherits it instead of re-deriving it.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

_MCP_TIMEOUT_S = float(os.environ.get("CELL_DATA_TIMEOUT_S", "30") or 30)

# Config reads are the hot path: assembling one plan asks for obstacles.yaml and
# robots.yaml several times over (four call sites in the OMPL backend alone), and
# over MCP each of those is a round trip. A few seconds of cache collapses them
# without letting a re-converted scene stay invisible for longer than the setup
# phase of a single planning call. Local mode is NEVER cached — a disk read is
# already cheap, and a test that writes a file and reads it back must see it.
_CACHE_TTL_S = float(os.environ.get("CELL_DATA_CACHE_TTL_S", "5") or 5)
_CACHED_KINDS = (KIND_CONFIG, KIND_MANIFEST)
_cache: dict[tuple, tuple[float, Optional[str]]] = {}
_cache_lock = threading.Lock()


class CellStoreError(RuntimeError):
    """A request this store refuses to make (bad kind, unsafe name)."""


class CellStoreUnavailable(CellStoreError):
    """The owning agent could not be reached, so the answer is unknown.

    Distinct from "absent" on purpose, and never swallowed into a default: a
    planner that reads "no obstacles" from a broken socket plans through the
    fixtures. Callers may degrade on absence; they must refuse on this.
    """


# ── mode ─────────────────────────────────────────────────────────────────────

def mode() -> str:
    """``local`` (default) or ``orchestrator``, from ``CELL_DATA_MODE``.

    Read per call rather than at import: a test flips it with monkeypatch, and a
    module-level snapshot would make the first import win for the whole session.
    """
    value = (os.environ.get("CELL_DATA_MODE") or MODE_LOCAL).strip().lower()
    if value not in (MODE_LOCAL, MODE_ORCHESTRATOR):
        raise CellStoreError(
            f"CELL_DATA_MODE={value!r} is not one of {MODE_LOCAL!r}, "
            f"{MODE_ORCHESTRATOR!r}")
    return value


def is_remote() -> bool:
    return mode() == MODE_ORCHESTRATOR


def endpoint() -> str:
    """The orchestrator's cell-data MCP URL.

    Built from the same env vars the orchestrator's own service definition sets,
    so the two cannot name different ports. Missing configuration is an error and
    not a fallback to the volume: "route cell data through the owner" degrading
    silently into "read the mount" is the drift this module exists to end.
    """
    explicit = (os.environ.get("CELL_DATA_MCP_URL") or "").strip()
    if explicit:
        return explicit
    host = (os.environ.get("ORCHESTRATOR_HOST") or "").strip()
    port = (os.environ.get("ORCH_MCP_PORT") or "8893").strip()
    if not host:
        raise CellStoreUnavailable(
            "CELL_DATA_MODE=orchestrator but neither CELL_DATA_MCP_URL nor "
            "ORCHESTRATOR_HOST is set — refusing to fall back to the /cells "
            "volume")
    return f"http://{host}:{port}/mcp"


def describe() -> dict:
    """Where this process reads cell data from — for /health and for humans."""
    info: dict = {"mode": mode()}
    if info["mode"] == MODE_ORCHESTRATOR:
        try:
            info["endpoint"] = endpoint()
        except CellStoreError as exc:
            info["error"] = str(exc)
    else:
        info["cells_root"] = str(cell_layout.cells_root())
    return info


# ── validation and local resolution ──────────────────────────────────────────

def _check(kind: str, name: str) -> None:
    if kind not in KINDS:
        raise CellStoreError(f"unknown cell-data kind {kind!r} (want one of "
                             f"{', '.join(KINDS)})")
    if not name or not _SAFE_NAME.match(name) or ".." in name:
        raise CellStoreError(
            f"{name!r} is not a plain artefact name (letters, digits, . _ -)")


def local_path(cell_id: str, kind: str, name: str, *,
               for_write: bool = False,
               cell_base: Optional[str] = None) -> Path:
    """Resolve one artefact on disk. LOCAL side only — never called remotely.

    Every branch delegates to :mod:`utils.cell_layout`, which owns the read/write
    asymmetry for ``config/`` and the manifest (a writer must not be steered by a
    stale copy at the cell root, a reader must still find one). Re-deriving that
    here is how the two would drift.
    """
    _check(kind, name)
    if not cell_id or not _SAFE_NAME.match(cell_id):
        raise CellStoreError(f"{cell_id!r} is not a plain cell id")
    if kind == KIND_CONFIG:
        return (cell_layout.agent_config_file(cell_id, name, cell_base)
                if for_write else
                cell_layout.config_file(cell_id, name, cell_base))
    if kind == KIND_MANIFEST:
        if name != cell_layout.CELL_MANIFEST:
            raise CellStoreError(
                f"the manifest kind holds only {cell_layout.CELL_MANIFEST!r}")
        return (cell_layout.agent_cell_yaml(cell_id, cell_base) if for_write
                else cell_layout.cell_yaml(cell_id, cell_base))
    directory = {
        KIND_TASKS: cell_layout.tasks_dir,
        KIND_PLANS: cell_layout.plans_dir,
        KIND_LOGS: cell_layout.logs_dir,
        KIND_RECORDINGS: cell_layout.recordings_dir,
    }[kind](cell_id, cell_base)
    return directory / name


# ── the transport ────────────────────────────────────────────────────────────

#: How long a call keeps retrying while the owner is not listening yet — the
#: containers start together and the first reader (the web panel asking for the
#: cell info) routinely beat the orchestrator's cell-data MCP by a few seconds,
#: logging a traceback on every stack start. Only CONNECTION REFUSED is retried:
#: the request never reached the server, so a retry cannot duplicate anything.
#: A timeout or an HTTP error is an answer and is reported at once, as before.
def _connect_retry_s() -> float:
    """Seconds to keep retrying a refused connection: ``CELL_STORE_CONNECT_RETRY_S``
    if set, else 5 s when reading through the owner (the containers, where the
    start-up race happens) and 0 locally."""
    raw = os.environ.get("CELL_STORE_CONNECT_RETRY_S")
    if raw is not None and raw.strip() != "":
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    try:
        return 5.0 if is_remote() else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _refused(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ConnectionRefusedError) or (
        isinstance(reason, OSError) and getattr(reason, "errno", None) == 111)


def _post_with_startup_retry(request) -> dict:
    deadline = time.monotonic() + _connect_retry_s()
    delay = 0.5
    while True:
        try:
            with urllib.request.urlopen(request, timeout=_MCP_TIMEOUT_S) as response:
                return json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as exc:
            if not _refused(exc) or time.monotonic() + delay > deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def _call(tool: str, arguments: dict) -> dict:
    """One JSON-RPC ``tools/call`` against the orchestrator's cell-data MCP.

    Same contract as :mod:`utils.mcp_http` serves and the orchestrator's own
    ``_mcp_call`` speaks: a single POST, a single JSON reply, the tool's return
    value as JSON text in ``result.content[0].text``.
    """
    url = endpoint()
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": tool, "arguments": arguments}})
    request = urllib.request.Request(
        url, data=payload.encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        body = _post_with_startup_retry(request)
    except urllib.error.HTTPError as exc:  # the tool raised, or the route is gone
        detail = ""
        try:
            detail = (exc.read() or b"").decode()[:400]
        except Exception:  # noqa: BLE001
            pass
        raise CellStoreUnavailable(
            f"cell-data MCP {url} returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CellStoreUnavailable(
            f"cell-data MCP {url} is unreachable: {exc}") from exc

    if body.get("error"):
        raise CellStoreUnavailable(
            f"cell-data MCP {url} refused {tool}: "
            f"{body['error'].get('message')}")
    blocks = ((body.get("result") or {}).get("content") or [])
    if not blocks:
        raise CellStoreUnavailable(f"cell-data MCP {url} returned no content "
                                   f"for {tool}")
    try:
        result = json.loads(blocks[0].get("text") or "{}")
    except ValueError as exc:
        raise CellStoreUnavailable(
            f"cell-data MCP {url} returned non-JSON for {tool}") from exc
    if not isinstance(result, dict):
        raise CellStoreUnavailable(
            f"cell-data MCP {url} returned {type(result).__name__} for {tool}")
    if result.get("error") and not result.get("success", True):
        # A refusal by the owner (unknown kind, unsafe name) is a defect in the
        # caller, not an outage — but it is still never an empty answer.
        raise CellStoreError(f"{tool} refused: {result['error']}")
    return result


# ── the API ──────────────────────────────────────────────────────────────────

def read_text(cell_id: str, kind: str, name: str, *,
              cell_base: Optional[str] = None) -> Optional[str]:
    """The artefact's contents, or ``None`` when it does not exist.

    Raises :class:`CellStoreUnavailable` in remote mode when the owner cannot be
    reached — "absent" and "unknown" are different answers.

    ``cell_base`` names a cell tree on THIS machine and therefore forces the local
    path: a caller that has already decided which directory it means (a host
    script, a test fixture) is not asking an owner anything. It is the one
    documented way past the seam, and it is not available over the wire.
    """
    _check(kind, name)
    if cell_base is not None:
        return _read_text_local(cell_id, kind, name, cell_base=cell_base)
    key = (cell_id, kind, name)
    if is_remote() and kind in _CACHED_KINDS and _CACHE_TTL_S > 0:
        now = time.monotonic()
        with _cache_lock:
            hit = _cache.get(key)
            if hit and now - hit[0] < _CACHE_TTL_S:
                return hit[1]
        text = _read_text_uncached(cell_id, kind, name)
        with _cache_lock:
            _cache[key] = (now, text)
        return text
    return _read_text_uncached(cell_id, kind, name)


def _read_text_uncached(cell_id: str, kind: str, name: str) -> Optional[str]:
    if is_remote():
        result = _call("cell_read", {"cell_id": cell_id, "kind": kind,
                                     "name": name})
        return result.get("text") if result.get("found") else None
    return _read_text_local(cell_id, kind, name)


def _read_text_local(cell_id: str, kind: str, name: str, *,
                     cell_base: Optional[str] = None) -> Optional[str]:
    path = local_path(cell_id, kind, name, cell_base=cell_base)
    try:
        return path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CellStoreUnavailable(f"could not read {path}: {exc}") from exc


def write_text(cell_id: str, kind: str, name: str, text: str, *,
               cell_base: Optional[str] = None) -> bool:
    """Replace the artefact, atomically. ``True`` when it landed.

    Atomic in both modes: the local writer renames a temp file into place, and
    the remote one is a single call to that same writer, so no reader ever sees a
    half-written plan.

    ``cell_base`` forces the local path — see :func:`read_text`.
    """
    _check(kind, name)
    _invalidate(cell_id, kind, name)
    if cell_base is not None:
        return write_text_local(cell_id, kind, name, text, cell_base=cell_base)
    if is_remote():
        return bool(_call("cell_write", {"cell_id": cell_id, "kind": kind,
                                         "name": name, "text": text}
                          ).get("written"))
    return write_text_local(cell_id, kind, name, text)


def temp_path(path: Path) -> Path:
    """A temp file beside ``path`` that belongs to THIS writer only.

    ``<name>.tmp`` was shared by every writer of the artefact: two concurrent
    writes of one file (two planners storing to the same cell's knowledge base)
    truncated each other's temp file, so one could rename the other's half-written
    text into place, or fail because its temp file had already been moved. Hidden
    (leading dot) so listers skip it.
    """
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def write_text_local(cell_id: str, kind: str, name: str, text: str, *,
                     cell_base: Optional[str] = None) -> bool:
    """The local writer, used directly by the orchestrator's MCP tool."""
    path = local_path(cell_id, kind, name, for_write=True, cell_base=cell_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(path)
    try:
        tmp.write_text(text)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)          # nothing left behind if the write failed
    return True


def append_text(cell_id: str, kind: str, name: str, text: str) -> bool:
    """Append to an artefact, creating it if needed.

    Its own verb rather than read-modify-write: the event stream is appended to
    from several threads and must not lose a line to a lost update.
    """
    _check(kind, name)
    if is_remote():
        return bool(_call("cell_append", {"cell_id": cell_id, "kind": kind,
                                          "name": name, "text": text}
                          ).get("appended"))
    return append_text_local(cell_id, kind, name, text)


def append_text_local(cell_id: str, kind: str, name: str, text: str) -> bool:
    """The local appender, used directly by the orchestrator's MCP tool.

    ``flush`` + ``fsync`` because the reader is another process (the dashboard
    polls while the run is in flight), and a buffered line is a line the live view
    does not have.
    """
    path = local_path(cell_id, kind, name, for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def list_entries(cell_id: str, kind: str, *, prefix: str = "",
                 suffix: str = "") -> list[dict]:
    """``[{name, size_bytes, mtime_s, path}, …]`` for one kind, name-sorted.

    Metadata only. A caller that wants contents asks for them by name — listing a
    directory of 200 plans must not mean transporting 200 plans.
    """
    if kind not in KINDS:
        raise CellStoreError(f"unknown cell-data kind {kind!r}")
    if is_remote():
        result = _call("cell_list", {"cell_id": cell_id, "kind": kind,
                                     "prefix": prefix, "suffix": suffix})
        entries = result.get("entries") or []
        return [e for e in entries if isinstance(e, dict)]
    return list_entries_local(cell_id, kind, prefix=prefix, suffix=suffix)


def list_entries_local(cell_id: str, kind: str, *, prefix: str = "",
                       suffix: str = "",
                       cell_base: Optional[str] = None) -> list[dict]:
    """The local lister, used directly by the orchestrator's MCP tool."""
    if kind in (KIND_CONFIG, KIND_MANIFEST):
        # Both are single-file kinds with a read/write asymmetry; listing a
        # directory would report copies a reader would never resolve to.
        raise CellStoreError(f"kind {kind!r} is not listable — read it by name")
    directory = {
        KIND_TASKS: cell_layout.tasks_dir,
        KIND_PLANS: cell_layout.plans_dir,
        KIND_LOGS: cell_layout.logs_dir,
        KIND_RECORDINGS: cell_layout.recordings_dir,
    }[kind](cell_id, cell_base)
    out: list[dict] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue                         # hidden: another writer's temp file
        if prefix and not path.name.startswith(prefix):
            continue
        if suffix and not path.name.endswith(suffix):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append({"name": path.name, "size_bytes": stat.st_size,
                    "mtime_s": stat.st_mtime, "path": str(path)})
    return out


def summarize(cell_id: str, kind: str, fields: list[str], *,
              count_fields: Optional[list[str]] = None,
              prefix: str = "", suffix: str = ".json",
              limit: int = 200) -> list[dict]:
    """A few named fields out of every JSON artefact of one kind, newest first.

    Each row is ``{name, size_bytes, mtime_s, path, <field>…, <count>_count}``.
    The reading happens wherever the bytes are: locally that is this process, and
    remotely it is the orchestrator — which is the point, because a plan carries
    every waypoint it computed and a plan LIST wants none of them.
    """
    if kind not in KINDS:
        raise CellStoreError(f"unknown cell-data kind {kind!r}")
    if is_remote():
        result = _call("cell_summarize", {
            "cell_id": cell_id, "kind": kind, "fields": list(fields),
            "count_fields": list(count_fields or []), "prefix": prefix,
            "suffix": suffix, "limit": limit})
        rows = result.get("entries") or []
        return [r for r in rows if isinstance(r, dict)]
    return summarize_local(cell_id, kind, fields, count_fields=count_fields,
                           prefix=prefix, suffix=suffix, limit=limit)


def summarize_local(cell_id: str, kind: str, fields: list[str], *,
                    count_fields: Optional[list[str]] = None,
                    prefix: str = "", suffix: str = ".json",
                    limit: int = 200,
                    cell_base: Optional[str] = None) -> list[dict]:
    """The local summariser. Shared with the orchestrator's MCP tool via that
    tool calling the same primitives, so both modes read the same fields the same
    way — a listing must not depend on who is doing the listing."""
    entries = list_entries_local(cell_id, kind, prefix=prefix, suffix=suffix,
                                 cell_base=cell_base)
    # Name breaks the mtime tie, descending. Two artefacts written in the same
    # filesystem timestamp are common (a test, a fast run) and the names in this
    # repo carry the timestamp, so falling back to the name keeps "newest first"
    # true instead of silently reversing to oldest-first inside the tie.
    entries.sort(key=lambda e: (e.get("mtime_s") or 0, e.get("name") or ""),
                 reverse=True)
    out: list[dict] = []
    for entry in entries[:max(1, int(limit))]:
        try:
            data = json.loads(local_path(cell_id, kind, entry["name"],
                                         cell_base=cell_base).read_text())
        except Exception:  # noqa: BLE001 - an unreadable artefact is skipped
            continue
        if not isinstance(data, dict):
            continue
        row = {"name": entry["name"], "size_bytes": entry.get("size_bytes"),
               "mtime_s": entry.get("mtime_s"), "path": entry.get("path")}
        for field in fields or []:
            row[field] = dig(data, field)
        for field in count_fields or []:
            value = dig(data, field)
            row[f"{field}_count"] = len(value) if isinstance(
                value, (list, dict, str)) else 0
        out.append(row)
    return out


def dig(data: dict, dotted: str):
    """``a.b.c`` through nested dicts; ``None`` at the first thing that is not one."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def delete(cell_id: str, kind: str, name: str) -> bool:
    """Remove an artefact. ``True`` when it existed and is gone."""
    _check(kind, name)
    _invalidate(cell_id, kind, name)
    if is_remote():
        return bool(_call("cell_delete", {"cell_id": cell_id, "kind": kind,
                                          "name": name}).get("deleted"))
    return delete_local(cell_id, kind, name)


def delete_local(cell_id: str, kind: str, name: str) -> bool:
    """The local deleter, used directly by the orchestrator's MCP tool."""
    path = local_path(cell_id, kind, name, for_write=True)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CellStoreUnavailable(f"could not delete {path}: {exc}") from exc


def _invalidate(cell_id: str, kind: str, name: str) -> None:
    with _cache_lock:
        _cache.pop((cell_id, kind, name), None)


def clear_cache() -> None:
    """Drop the config read cache — for tests and for an explicit re-read."""
    with _cache_lock:
        _cache.clear()


# ── convenience for the two formats every artefact is in ─────────────────────

def read_json(cell_id: str, kind: str, name: str, *,
              cell_base: Optional[str] = None) -> Optional[dict]:
    """Parsed JSON, or ``None`` when absent or unparseable.

    Unparseable reads as absent because every caller's fallback for a corrupt
    artefact is the same as for a missing one, and there is nothing safer to do
    with half a file.
    """
    text = read_text(cell_id, kind, name, cell_base=cell_base)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def write_json(cell_id: str, kind: str, name: str, data, *,
               indent: Optional[int] = None,
               cell_base: Optional[str] = None) -> bool:
    return write_text(cell_id, kind, name,
                      json.dumps(data, indent=indent, default=str),
                      cell_base=cell_base)


def read_yaml(cell_id: str, kind: str, name: str, *,
              cell_base: Optional[str] = None) -> Optional[dict]:
    """Parsed YAML, or ``None`` when absent. Unparseable YAML raises.

    Asymmetric with :func:`read_json` on purpose: the YAML artefacts are the
    *inputs to collision checking* (obstacles.yaml, robots.yaml), and a
    half-parsed obstacle model must stop a planning run, not read as an empty
    cell.
    """
    key, hit, blob = _parsed_yaml(cell_id, kind, name, cell_base)
    if not isinstance(hit, dict):
        return None
    # Each caller gets its OWN copy, so none can edit another's. marshal (C) makes
    # that copy ~20x faster than copy.deepcopy: on welding_cell_2_small a planning step read
    # obstacles.yaml (2,678 boxes) and robots.yaml 362 times, and the deep copies
    # took 12.2 s of a 16.8 s step. Data marshal cannot carry (a YAML timestamp)
    # keeps the deep copy.
    if blob is not None:
        import marshal  # noqa: PLC0415
        return marshal.loads(blob)
    import copy  # noqa: PLC0415
    return copy.deepcopy(hit)


def _parsed_yaml(cell_id: str, kind: str, name: str, cell_base: Optional[str]):
    """``(content key, parsed object, marshal blob | None)`` — parsed once per
    CONTENT (the same text is the same data; a changed file is a new key), or
    ``(None, None, None)`` when the artefact is absent. The parsed object is
    SHARED: callers must not mutate it (``read_yaml`` hands out copies)."""
    text = read_text(cell_id, kind, name, cell_base=cell_base)
    if text is None:
        return None, None, None
    import hashlib  # noqa: PLC0415
    key = hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()
    with _YAML_LOCK:
        entry = _YAML_PARSED.get(key)
    if entry is None:
        import marshal  # noqa: PLC0415
        import yaml  # noqa: PLC0415 - lazy, so this module keeps importing on stdlib
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        hit = yaml.load(text, Loader=loader)  # noqa: S506 - a SafeLoader
        try:
            blob = marshal.dumps(hit)
            if marshal.loads(blob) != hit:      # a type marshal does not round-trip
                blob = None
        except ValueError:
            blob = None
        entry = (hit, blob)
        with _YAML_LOCK:
            _YAML_PARSED[key] = entry
            while len(_YAML_PARSED) > _YAML_KEPT:
                _YAML_PARSED.pop(next(iter(_YAML_PARSED)))
    return (key, *entry)



_YAML_PARSED: dict = {}
_YAML_KEPT = 32
_YAML_LOCK = __import__("threading").Lock()
