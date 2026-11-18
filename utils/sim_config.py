"""
Simulation configuration helpers — resolve cell paths and validate layout.

Path composition itself lives in :mod:`utils.cell_layout`, which owns the split
between the shared cell root (the scene bundle: URDF, meshes, fixtures) and the
agent workspace under ``agentic_planner/`` (``config/``, tasks, plans, logs,
recordings). This module is the thin validation surface on top of it.
"""
from __future__ import annotations

from pathlib import Path

from utils import cell_layout


def cell_path(cell_id: str) -> Path:
    return cell_layout.cell_dir(cell_id)


def urdf_path(cell_id: str, robot_ns: str) -> Path | None:
    """The URDF describing one arm, whether referenced or generated."""
    return cell_layout.robot_urdf(cell_id, robot_ns)


def _rel(path: Path, base: Path) -> str:
    """``path`` as a label relative to the cell directory, absolute if outside."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def validate_cell_layout(cell_id: str) -> dict:
    """
    Validate that a cell directory has the required structure.

    Paths are resolved through :mod:`utils.cell_layout`, so a cell on either
    layout — agent workspace or the flat pre-split form — validates against the
    directories it actually uses rather than against one hardcoded shape.

    Returns:
        {"valid": bool, "missing": [...], "found": [...]}
    """
    base = cell_path(cell_id)

    # (label, path). Required is the cell's description of itself; a cell with no
    # robots.yaml is not a cell. Everything optional is either derived state or
    # generated, so its absence means "nothing has run yet", not "malformed".
    # The label is the path as RESOLVED, not a fixed literal: config/ and cell.yaml
    # live under agentic_planner/ for a converted cell and at the cell root for an
    # older one, and a report that names the wrong one sends the reader to the wrong
    # directory.
    robots = cell_layout.robots_yaml(cell_id)
    manifest = cell_layout.cell_yaml(cell_id)
    required = [(_rel(robots, base), robots)]
    optional = [
        ("tasks/",       cell_layout.tasks_dir(cell_id)),
        # Either of these answers "where is this arm's URDF": cell.yaml names it
        # where the scene bundle already put it, urdf/ holds a baked copy. A cell
        # needs one of the two, but which one is not a validity question.
        (_rel(manifest, base), manifest),
        ("urdf/",        cell_layout.urdf_dir(cell_id)),
        ("recordings/trajectory_index.json",
                         cell_layout.trajectory_index(cell_id)),
    ]

    found:   list[str] = []
    missing: list[str] = []

    for label, p in required:
        (found if p.exists() else missing).append(label)
    for label, p in optional:
        if p.exists():
            found.append(label)

    return {
        "valid":    len(missing) == 0,
        "cell_id":  cell_id,
        "base":     str(base),
        "found":    found,
        "missing":  missing,
    }
