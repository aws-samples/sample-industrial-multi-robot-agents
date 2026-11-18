"""The single owner of "where does this live inside a cell directory?".

A cell directory is now **two things layered in one tree**, and the split is the
whole point of this module:

* The **cell root** is *input*, and it is shared. It is the scene bundle as the
  cell's author shipped it: ``scene.yaml``, the robots' dressed URDF/SRDF and
  meshes, the fixtures, the floor plans. A cell hosts several applications; they
  all read these, so nothing here is namespaced to any one of them, and nothing
  we generate is written among them.

* ``<cell>/agentic_planner/`` is *this* application's workspace — everything the
  agents author or compute: ``config/`` (``robots.yaml``, ``operations.yaml``, the
  transport config, the calculated ``obstacles.yaml``), task programs,
  behaviour-tree plans, run logs, recordings. Isolating it means another
  application dropped into the same cell cannot collide with our output, and a
  reader can tell input from derived state by looking at the path.

**``config/`` and ``cell.yaml`` belong to the agent workspace.** Both are
something the converter *writes* from the scene, so both are derived state and
live with the rest of our output. The converter adds **nothing** to the cell root.
Cells written before that decision keep them at the root, so the accessors below
resolve the agent copy first and fall back to the root one — see
:func:`_config_file` and :func:`cell_yaml`. A cell that predates the split
entirely (``welding_cell_2_small``) resolves both to the same place, because an
unmigrated cell has no agent root to namespace under.

The manifest moving does **not** change what is inside it: every path it records
stays relative to the *cell directory*, not to the manifest's own directory, so
``robots/<model>/<model>.urdf`` means the same file before and after the move (see
:func:`_manifest_asset`). Relative-to-the-cell is what makes one manifest readable
both on the host (``./cells/<cell>/…``) and in a container (``/cells/<cell>/…``);
re-anchoring it on the manifest would have broken that and every existing cell.

Why one module. Before this file, roughly forty call sites each composed
``f"{base}/{cell_id}/plans"`` themselves — forty copies of one decision, the
failure mode this codebase keeps paying for (see ``tests/test_default_cell.py``
for the same bug in the default cell *name*). A layout change then means forty
edits and one of them is always missed. Every path inside a cell is answered
here, once.

**The URDF and the meshes are deliberately NOT namespaced.** Mesh URIs are baked
into published ``robot_description`` and resolved by every DDS subscriber and by
RViz. Adding a path segment there changes what peers *outside this repo* resolve,
so it is not a rename we can make from Python. The accessors for those live in the
*cell-root* section below and must stay there.

The DDS config is the one generated file with a consumer outside this repo: the
CycloneDDS C runtime reads it via ``CYCLONEDDS_URI``, baked into the container at
create time. Moving the file is safe only because the URI is computed from the
same probe — :func:`config_uri` — so the path in the env var and the file on disk
cannot disagree. Never hand-write that URI.

A cell root comes in two shapes, and only one accessor should know it. A
**generated** cell has the converter's baked ``urdf/<robot_id>/<model>.urdf``. A
**referenced** cell (``cell.yaml`` present) leaves the robot's dressed URDF/SRDF
and meshes where the source bundle already put them and names those paths in the
manifest, so converting it copies nothing and there is still exactly one document
per arm. :func:`robot_urdf` and :func:`robot_srdf` answer for both; no consumer
should glob for a URDF itself.

Migration state is per cell, decided by a single probe: a cell is on the new
layout **iff** ``<cell>/agentic_planner/`` exists. One directory answers it, so a
writer can never leave a cell half-migrated — it cannot write ``plans/`` under
the agent root while writing ``logs/`` beside it. Cells that predate the split
(``welding_cell_2_small`` is kept as a read-only archive) keep resolving flat.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

# The one spelling of the agent workspace. Underscored, not hyphenated: this name
# appears in paths that get interpolated into shell, YAML and Python module-ish
# contexts, and a hyphen is the kind of thing that only breaks in the one place
# nobody tested.
AGENT_ROOT_NAME = "agentic_planner"

# Directories the agents own. Each is *derived* state — deleting the agent root
# loses work, but never the cell's description of itself.
AGENT_DIRS: tuple[str, ...] = ("config", "tasks", "plans", "logs", "recordings")

# The files that live in a cell's config directory. All of them are written by the
# converter from the scene, so all of them belong to the agent workspace; older
# cells keep them at the cell root and still resolve (see _config_file).
CONFIG_FILES: tuple[str, ...] = (
    "robots.yaml",
    "operations.yaml",
    "cyclonedds.xml",
    "zenoh_session.json5",
)

DEFAULT_CELL_BASE = "/cells"

# The cell manifest. It answers "which files describe robot <id>?" for a cell
# whose assets are REFERENCED where they already sit rather than copied into a
# generated ``urdf/<robot_id>/`` tree. A cell without one is not broken — the
# accessors below fall back to the generated layout — so both kinds of cell
# resolve through the same call. Agent-owned: the converter writes it, so it lives
# in the workspace (see cell_yaml), and its contents stay cell-relative.
CELL_MANIFEST = "cell.yaml"


# ── the base of the cells tree ───────────────────────────────────────────────

def cells_root(cell_base: Optional[str] = None) -> Path:
    """Root of the cells tree: the explicit override, else ``CELL_BASE_PATH``.

    An env var set to empty means unset — compose writes ``CELL_BASE_PATH:
    ${...}`` and an empty value used to yield ``//plans``.
    """
    base = cell_base or os.environ.get("CELL_BASE_PATH") or DEFAULT_CELL_BASE
    return Path(str(base).rstrip("/") or "/")


def cell_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """The cell's own directory. Everything below is relative to this."""
    return cells_root(cell_base) / cell_id


def scratch_root(scratch_base: Optional[str] = None) -> Path:
    """Where a planner keeps its OWN unfinished work: ``PLANNER_SCRATCH_PATH``.

    Not cell data. A checkpoint of a run that has not finished, and the temporary
    files of a run in flight, belong to the process doing the run — nobody else
    reads them, and a resume is only meaningful in the planner that took them.
    Everything a cell *keeps* (plans, traces, events, obstacle models, task
    programs) goes to its owner through :mod:`utils.cell_store` instead.

    Defaults to :func:`cells_root`, so a process that sets nothing behaves exactly
    as it did before this existed — a test, a host script, or the orchestrator
    reading a checkpoint on the volume it owns. The planner's container sets it to
    a private volume, which is what makes "the planner keeps only its unfinished
    work" a property of the deployment and not a promise in a docstring.

    The layout below it is the ordinary cell layout (``<cell>/logs/…``), so the
    same accessors resolve in both roots and nothing needs a second set of paths.
    """
    base = scratch_base or os.environ.get("PLANNER_SCRATCH_PATH")
    return cells_root(base) if base else cells_root()


# ── which layout is this cell on? ────────────────────────────────────────────

def agent_root(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """``<cell>/agentic_planner`` — whether or not it exists yet."""
    return cell_dir(cell_id, cell_base) / AGENT_ROOT_NAME


def is_migrated(cell_id: str, cell_base: Optional[str] = None) -> bool:
    """True when this cell keeps agent state under ``agentic_planner/``.

    The presence of the directory IS the answer; there is no marker file to get
    out of sync with the tree it describes.
    """
    return agent_root(cell_id, cell_base).is_dir()


def _agent_dir(cell_id: str, name: str, cell_base: Optional[str] = None) -> Path:
    """One agent-owned directory, under the agent root or flat.

    Resolved through :func:`is_migrated` rather than by probing for ``name``
    itself: probing per-directory is what lets a cell end up with ``plans/`` in
    one place and ``logs/`` in the other.
    """
    root = cell_dir(cell_id, cell_base)
    return (root / AGENT_ROOT_NAME / name) if is_migrated(cell_id, cell_base) else (root / name)


def agent_relative(cell_id: str, *parts: str,
                   cell_base: Optional[str] = None) -> str:
    """An agent-owned path *relative to the cell directory*, as a POSIX string.

    For callers that must probe on the host but emit a **container** path — the
    compose generator resolving ``./cells`` and writing ``/cells/...``. Composing
    that by hand is how the two drift apart.
    """
    prefix = (AGENT_ROOT_NAME + "/") if is_migrated(cell_id, cell_base) else ""
    return prefix + "/".join(parts)


# ── agent-owned directories ──────────────────────────────────────────────────

def agent_config_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """Config the agents *compute*, not config that describes the cell.

    Distinct from the cell's own ``config/`` (see :func:`root_config_dir`): the
    obstacle model in here is an output of the scene converter, so it belongs to
    whoever produced it.
    """
    return _agent_dir(cell_id, "config", cell_base)


def tasks_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _agent_dir(cell_id, "tasks", cell_base)


def plans_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _agent_dir(cell_id, "plans", cell_base)


def logs_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _agent_dir(cell_id, "logs", cell_base)


def recordings_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _agent_dir(cell_id, "recordings", cell_base)


# ── agent-owned files ────────────────────────────────────────────────────────

def obstacles_yaml(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """The calculated obstacle model (voxel or AABB), an output of conversion."""
    return agent_config_dir(cell_id, cell_base) / "obstacles.yaml"


# ── the config directory ─────────────────────────────────────────────────────
#
# ``config/`` is agent-owned: the converter writes every file in it. Two places
# have to be answered separately, because a writer and a reader want different
# things from a cell that has not been converted yet.

def agent_config_file(cell_id: str, name: str,
                      cell_base: Optional[str] = None) -> Path:
    """Where a config file is **written**: the workspace's own ``config/``.

    Unlike :func:`_config_file` this never answers with a root copy that happens
    to exist — a writer must not be steered by a stale file. It does still go through
    :func:`is_migrated`, so writing config to a cell that predates the split leaves
    it flat instead of creating an agent root and stranding that cell's existing
    ``plans/`` and ``logs/`` outside it. Call this AFTER
    :func:`create_agent_layout`, as the converter does, and the two agree.
    """
    return agent_config_dir(cell_id, cell_base) / name


def _config_file(cell_id: str, name: str,
                 cell_base: Optional[str] = None) -> Path:
    """Where a config file is **read** from: the agent copy, else the cell root.

    The agent copy wins when it exists. The root is the fallback for cells written
    before ``config/`` moved into the workspace (``welding_cell_2_small`` still ships
    one), and it is also the answer when neither exists — so a cell that has not
    been converted yet reports the path it has historically reported rather than a
    path invented by this function.

    A cell can therefore briefly hold both. That is a hazard, not a feature — a
    stale root copy reads as live config to anyone looking at the tree — so the
    converter warns when it writes the agent copy beside an existing root one.
    """
    agent = agent_config_file(cell_id, name, cell_base)
    return agent if agent.is_file() else root_config_dir(cell_id, cell_base) / name


def config_file(cell_id: str, name: str,
                cell_base: Optional[str] = None) -> Path:
    """Public reader for one config file *by name*, with the same resolution as
    :func:`robots_yaml` and friends — the agent copy, else the cell root.

    The named accessors below cover the files this repo knows about. This exists
    for the one caller that is handed a name at runtime rather than choosing it:
    :mod:`utils.cell_store`, which serves ``(kind, name)`` requests on behalf of
    an agent that no longer reads the volume itself. It delegates rather than
    re-deriving, so a caller holding a name and a caller holding an accessor
    cannot resolve to different files.
    """
    return _config_file(cell_id, name, cell_base)


def config_uri(cell_id: str, name: str, cell_base: Optional[str] = None,
               container_base: str = DEFAULT_CELL_BASE) -> Optional[str]:
    """``file:///cells/<cell>/…/config/<name>`` for a file that is actually there.

    Probed on the **host** (``cell_base``), emitted as a **container** path — the
    compose generator's problem: it resolves ``./cells`` but writes an env var a
    container will read. ``None`` when the cell ships no such file, so a caller
    never bakes a URI that resolves to nothing (a missing ``CYCLONEDDS_URI`` target
    is silently ignored by the DDS runtime, which then runs on its own defaults).

    This is the only sanctioned way to name one of these files to a peer outside
    this process: it and :func:`_config_file` read the same probe, so the env var
    and the file cannot end up naming different directories.
    """
    host = _config_file(cell_id, name, cell_base)
    if not host.is_file():
        return None
    rel = host.relative_to(cell_dir(cell_id, cell_base)).as_posix()
    return f"file://{str(container_base).rstrip('/')}/{cell_id}/{rel}"


def robots_yaml(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _config_file(cell_id, "robots.yaml", cell_base)


def operations_yaml(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _config_file(cell_id, "operations.yaml", cell_base)


def cyclonedds_xml(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _config_file(cell_id, "cyclonedds.xml", cell_base)


def zenoh_session_config(cell_id: str, cell_base: Optional[str] = None) -> Path:
    return _config_file(cell_id, "zenoh_session.json5", cell_base)


# ── the cell manifest ────────────────────────────────────────────────────────
#
# Same write/read split as ``config/``, and for the same reason: the converter
# writes the manifest, so it is derived state that belongs in the workspace, but a
# cell written before the move keeps it at the root and must still resolve.

def agent_cell_yaml(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """Where the manifest is **written**: the workspace's own ``cell.yaml``.

    Like :func:`agent_config_file`, this never answers with a root copy that
    happens to exist, so a stale manifest cannot steer the converter into writing
    beside it. Call it AFTER :func:`create_agent_layout`.
    """
    return agent_root(cell_id, cell_base) / CELL_MANIFEST \
        if is_migrated(cell_id, cell_base) \
        else cell_dir(cell_id, cell_base) / CELL_MANIFEST


def cell_yaml(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """Where the manifest is **read** from: the agent copy, else the cell root.

    The agent copy wins when it exists; the root is the fallback for cells written
    before the manifest moved into the workspace, and the answer when neither
    exists. Callers treat "not a file" as "this cell has no manifest" and fall back
    to the generated ``urdf/`` layout (:func:`robot_urdf`), so an unconverted cell
    reports the path it has always reported rather than one invented here.
    """
    agent = agent_cell_yaml(cell_id, cell_base)
    return agent if agent.is_file() else cell_dir(cell_id, cell_base) / CELL_MANIFEST


_PARSED: dict = {}          # sha1(text) -> (marshal blob | None, parsed)
_PARSED_KEPT = 32


def load_yaml_text(text: str):
    """``yaml`` of ``text``, parsed ONCE per content with the C loader when the
    build has it, returned as a fresh copy (callers may mutate it).

    The manifest and robots.yaml were parsed with the pure-Python ``safe_load`` on
    every call — ~75 times each per assembler gate on welding_cell_2_small. Same text, same
    data; a changed file is a new key, so nothing goes stale. Parse errors raise,
    exactly as ``yaml.safe_load`` does."""
    import hashlib  # noqa: PLC0415
    import marshal  # noqa: PLC0415
    key = hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()
    hit = _PARSED.get(key)
    if hit is None:
        import yaml  # noqa: PLC0415 - lazy on purpose (stdlib-only import path)
        data = yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))  # noqa: S506
        try:
            blob = marshal.dumps(data)
            if marshal.loads(blob) != data:
                blob = None
        except ValueError:
            blob = None
        hit = (blob, data)
        _PARSED[key] = hit
        while len(_PARSED) > _PARSED_KEPT:
            _PARSED.pop(next(iter(_PARSED)))
    blob, data = hit
    if blob is not None:
        return marshal.loads(blob)
    import copy  # noqa: PLC0415
    return copy.deepcopy(data)


def read_manifest(cell_id: str, cell_base: Optional[str] = None) -> dict:
    """The cell manifest as a dict, or ``{}`` when there is none.

    Unreadable is treated as absent: every caller has a fallback to the generated
    layout, and that fallback is the safe branch. YAML is imported lazily so this
    module keeps importing with nothing but the stdlib — it is on the import path
    of scripts that run before dependencies are installed.
    """
    p = cell_yaml(cell_id, cell_base)
    if not p.is_file():
        return {}
    try:
        data = load_yaml_text(p.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def manifest_robot(cell_id: str, robot_ns: str,
                   cell_base: Optional[str] = None) -> dict:
    """The manifest entry for one robot, matched on ``id``, ``scene_id``, ``namespace``.

    ``robot_ns`` is the cell-local id the planner and the ROS graph use
    (``robot1``); ``scene_id`` is what the scene export called the same arm
    (``kr240r2900__sticktool``). Accepting either means a caller holding whichever
    id it happens to have still resolves the arm.

    ``namespace`` is matched LAST and only as a fallback, because it is not unique:
    arms in a shared namespace all carry the same one, so a caller that holds only
    a namespace (it read the top level of ``robots.yaml``) gets the first arm in
    it. That is right when the arms share a model — the usual reason to share a
    namespace — and is still better than no answer, but a caller that can name the
    arm should pass its ``id``.
    """
    entries = [e for e in (read_manifest(cell_id, cell_base).get("robots") or [])
               if isinstance(e, dict)]
    for entry in entries:
        if robot_ns in (entry.get("id"), entry.get("scene_id")):
            return entry
    for entry in entries:
        if robot_ns == entry.get("namespace"):
            return entry
    return {}


def task_file(cell_id: str, program: str, cell_base: Optional[str] = None) -> Path:
    return tasks_dir(cell_id, cell_base) / f"{program}.json"


def plan_file(cell_id: str, tree_id: str, cell_base: Optional[str] = None) -> Path:
    return plans_dir(cell_id, cell_base) / f"{tree_id}.json"


def recording_file(cell_id: str, tree_id: str,
                   cell_base: Optional[str] = None) -> Path:
    return recordings_dir(cell_id, cell_base) / f"{tree_id}.rrd"


def trajectory_index(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """Warm-start knowledge-base index, kept beside the recordings it indexes."""
    return recordings_dir(cell_id, cell_base) / "trajectory_index.json"


def run_log_file(cell_id: str, run_id: str,
                 cell_base: Optional[str] = None, suffix: str = "json") -> Path:
    return logs_dir(cell_id, cell_base) / f"run_{run_id}.{suffix}"


def quality_log_file(cell_id: str, tree_id: str,
                     cell_base: Optional[str] = None) -> Path:
    """Trajectory-quality report for one plan, named for the plan it describes.

    Deliberately the SAME basename as ``plan_file`` and ``recording_file``, so a
    reader holding a tree_id finds the plan, its replay and its quality report
    without a lookup table. It lives in logs/ rather than beside the plan because
    it is a derived observation, not part of what gets executed — and ``run_``
    prefixes every run log, so a tree_id can never collide with one.
    """
    return logs_dir(cell_id, cell_base) / f"{tree_id}.json"


# ── cell-root paths: shared input, NEVER namespaced ──────────────────────────
#
# Everything below ANSWERS with a path under the cell directory itself. These are
# read by peers we do not control — DDS subscribers resolving mesh URIs, RViz — so
# the answer never carries the agent-root segment. (robot_urdf and robot_srdf read
# the agent-owned manifest to find them, but what they return is still a root path:
# the manifest says where the bundle put a file, it does not move it.)

def root_config_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """``<cell>/config`` — where a cell written before the move keeps its config.

    The fallback arm of :func:`_config_file`, and the answer for an unmigrated cell
    (there, it is the same directory as :func:`agent_config_dir`). Nothing writes
    here any more: the converter writes to the agent workspace.
    """
    return cell_dir(cell_id, cell_base) / "config"


def _manifest_asset(cell_id: str, robot_ns: str, key: str,
                    cell_base: Optional[str] = None) -> Optional[Path]:
    """One asset path from the manifest, resolved against the CELL directory.

    Manifest paths are cell-relative on purpose: the same manifest is read on the
    host (``./cells/<cell>/...``) and inside a container (``/cells/<cell>/...``),
    and only a relative path means the same file in both.

    Cell-relative, **not** manifest-relative — and now that the manifest itself
    lives in ``agentic_planner/`` those are different directories. Anchoring on the
    manifest would have reinterpreted every ``urdf:`` in every existing cell as
    ``agentic_planner/robots/...``, silently resolving nothing and sending each arm
    down the guessed-``urdf/``-glob fallback. The base stays :func:`cell_dir` so
    moving the manifest changes where the file is, never what it means.
    """
    rel = manifest_robot(cell_id, robot_ns, cell_base).get(key)
    if not rel:
        return None
    p = Path(str(rel))
    return p if p.is_absolute() else cell_dir(cell_id, cell_base) / p


def robot_urdf(cell_id: str, robot_ns: str,
               cell_base: Optional[str] = None) -> Optional[Path]:
    """The URDF describing one arm, or None when the cell has none.

    Two kinds of cell answer this differently and every consumer needs the same
    answer, which is why the choice lives here rather than at six call sites:

    * a REFERENCED cell names the file in ``cell.yaml`` where it already sits
      (``robots/<model>/<model>.urdf``) — nothing was copied to make the cell;
    * a GENERATED cell has the converter's baked ``urdf/<robot_id>/*.urdf``.

    The manifest wins when it resolves. A manifest entry pointing at a file that
    is not there falls through to the glob rather than failing: half a manifest
    should degrade to the old layout, not break the cell.
    """
    p = _manifest_asset(cell_id, robot_ns, "urdf", cell_base)
    if p is not None and p.is_file():
        return p
    matches = sorted(urdf_dir(cell_id, robot_ns, cell_base).glob("*.urdf"))
    return matches[0] if matches else None


def robot_srdf(cell_id: str, robot_ns: str,
               cell_base: Optional[str] = None) -> Optional[Path]:
    """The SRDF beside the arm's URDF, or None. See :func:`robot_urdf`."""
    p = _manifest_asset(cell_id, robot_ns, "srdf", cell_base)
    if p is not None and p.is_file():
        return p
    matches = sorted(srdf_dir(cell_id, robot_ns, cell_base).glob("*.srdf"))
    return matches[0] if matches else None


def urdf_dir(cell_id: str, robot_ns: Optional[str] = None,
             cell_base: Optional[str] = None) -> Path:
    d = cell_dir(cell_id, cell_base) / "urdf"
    return (d / robot_ns) if robot_ns else d


def srdf_dir(cell_id: str, robot_ns: Optional[str] = None,
             cell_base: Optional[str] = None) -> Path:
    d = cell_dir(cell_id, cell_base) / "srdf"
    return (d / robot_ns) if robot_ns else d


def mesh_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """Where the ``filename="…"`` in every baked URDF points."""
    return urdf_dir(cell_id, cell_base=cell_base) / "meshes"


def scene_dir(cell_id: str, cell_base: Optional[str] = None) -> Path:
    """The Isaac export the cell was built from, plus its source manifest."""
    return cell_dir(cell_id, cell_base) / "scene"


def mesh_uri_root(cell_id: str, container_base: str = DEFAULT_CELL_BASE) -> str:
    """``file:///cells/<cell>/urdf/meshes/`` — the URI baked into the URDF.

    A **container** path, not a host one: it is resolved by whoever loads the
    published ``robot_description``. Never carries the agent-root segment.
    """
    return f"file://{str(container_base).rstrip('/')}/{cell_id}/urdf/meshes/"


def urdf_glob(cell_id: str, robot_ns: str,
              cell_base: Optional[str] = None) -> str:
    """Glob pattern for a robot's baked URDF. One robot, one file, by convention."""
    return str(urdf_dir(cell_id, robot_ns, cell_base) / "*.urdf")


# ── creating the layout ──────────────────────────────────────────────────────

def create_agent_layout(cell_id: str, cell_base: Optional[str] = None,
                        dirs: Iterable[str] = AGENT_DIRS) -> list[Path]:
    """Create the agent root and every subdirectory it owns.

    Idempotent, and it creates the agent root **first** — which is what flips
    :func:`is_migrated` for this cell, so the subdirectories that follow resolve
    to the new layout rather than being scattered flat.

    Returns the directories that exist afterwards, in creation order, so a caller
    can report what it made.
    """
    root = agent_root(cell_id, cell_base)
    root.mkdir(parents=True, exist_ok=True)
    made = [root]
    for name in dirs:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        made.append(d)
    return made
