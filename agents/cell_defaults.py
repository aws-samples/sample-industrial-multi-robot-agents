"""The cell the stack runs when nothing names one.

Six modules used to carry their own ``os.environ.get("CELL_ID",
"welding_cell_2_small")``. A default duplicated six times is a default nobody owns:
changing it meant finding every copy, and a module that kept the old string would
quietly load a *different cell's* robots.yaml, obstacles and home poses than the
one everything else was working on — the same failure mode as the nine hardcoded
HOME poses (see :mod:`agents.home_pose`).

``welding_cell_2_small`` is the reference cell: two arms in a shared namespace, a
declared ``home_pose_deg``, and the task programs the golden path is exercised
with. ``welding_cell_2_small`` remains on disk and is still selectable with
``make sim SIM=welding_cell_2_small``.

Import-safe: no dependencies at all. ``docker/launch_viz.py`` runs in the RViz
image, which has no agents package, so it repeats the literal; the fence test in
``tests/test_default_cell.py`` keeps that copy honest.
"""
from __future__ import annotations

import os

DEFAULT_CELL_ID = "welding_cell_2_small"


def default_cell_id() -> str:
    """The cell this process should act on: ``$CELL_ID``, else the default.

    Read per call rather than captured at import so a test (or a container whose
    environment is set after import) is not stuck with the value that happened to
    be present when the module first loaded.
    """
    return (os.environ.get("CELL_ID") or "").strip() or DEFAULT_CELL_ID
