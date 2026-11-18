"""The one place an arm's Pinocchio model is built, TCP frame included.

Why this module exists
──────────────────────
The planner targets a **tool centre point** — the electrode tip — not the robot's
flange. That frame is not in the robot's dressed URDF: the vendor model ends at
``link_6``. It used to be added by the scene converter, which wrote a *copy* of
the URDF with a ``welding_tcp`` link appended. A cell that REFERENCES its URDF
where the source bundle put it (``cell.yaml`` → ``robots/<model>/<model>.urdf``)
has no such copy to bake into, so the frame has to be added at load time instead,
from the offset the cell declares in ``config/robots.yaml``:

    end_effector_link: welding_tcp     # the frame to target
    tip_link:          link_6          # its parent, from the URDF
    tcp_offset:        [x, y, z]       # metres, in the tip-link frame

The dangerous version of this is per-consumer: four processes load this model —
the trajectory agent's IK, its collision geometry, the per-robot agent's status
FK, and the simulated controller — and each used to pick an end-effector frame by
searching ``model.frames`` for a name containing "welding_tcp" or "tool0", with
``model.nframes - 1`` as the fallback. On a referenced cell that fallback silently
resolves to whatever frame happens to be last, so every weld would be targeted at
the wrong point by the tool's length with nothing reporting an error. One loader
means one answer: :func:`ee_frame_id` never guesses.

Pinocchio and NumPy are imported lazily. This module is on the import path of
``utils`` consumers that run on the host, where neither is installed.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from utils import cell_layout

# Frame names that have historically meant "the tool tip". Consulted only when
# the cell does not declare end_effector_link, so a hand-written cell still works.
_LEGACY_EE_NAMES = ("welding_tcp", "tool0")


def _robot_config(cell_id: str, robot_ns: str,
                  cell_base: Optional[str] = None) -> dict:
    """The ``config/robots.yaml`` block for one robot, matched on the inner key."""
    path = cell_layout.robots_yaml(cell_id, cell_base)
    try:
        text = path.read_text()
    except OSError:
        return {}
    # Parsed once per content (cell_layout.load_yaml_text), not on every call.
    cfg = cell_layout.load_yaml_text(text) or {}
    for ns_block in (cfg.get("robots") or {}).values():
        if not isinstance(ns_block, dict):
            continue
        rcfg = ns_block.get(robot_ns)
        if isinstance(rcfg, dict):
            return rcfg
    return {}


def tcp_spec(cell_id: str, robot_ns: str,
             cell_base: Optional[str] = None) -> dict:
    """What the cell says its tool frame is: ``{ee_link, tip_link, offset}``.

    ``offset`` is always a 3-tuple of floats, so a caller never has to defend
    against a half-written declaration. A missing offset is ``(0, 0, 0)`` — the
    flange itself — which is the same thing the converter warns about when
    ``--tcp-offset`` is not given.
    """
    rcfg = _robot_config(cell_id, robot_ns, cell_base)
    raw = rcfg.get("tcp_offset") or (0.0, 0.0, 0.0)
    try:
        offset = tuple(float(v) for v in raw)[:3]
    except (TypeError, ValueError):
        offset = (0.0, 0.0, 0.0)
    while len(offset) < 3:
        offset = offset + (0.0,)
    return {
        "ee_link":  rcfg.get("end_effector_link") or "welding_tcp",
        "tip_link": rcfg.get("tip_link") or "",
        "offset":   offset,
    }


def urdf_path(cell_id: str, robot_ns: str,
              cell_base: Optional[str] = None) -> Optional[Path]:
    """The arm's URDF — referenced or generated. See :func:`cell_layout.robot_urdf`."""
    return cell_layout.robot_urdf(cell_id, robot_ns, cell_base)


def add_tcp_frame(model, spec: dict) -> bool:
    """Attach the cell's tool frame to a loaded model. True if it was added.

    Idempotent: a generated cell's URDF already carries the frame, so this is a
    no-op there and the same call site serves both kinds of cell.

    The new frame hangs off the tip link's *joint*, with the tip link's own
    placement composed in — not off the model root — so it moves with the arm.
    """
    import numpy as np
    import pinocchio as pin

    ee = spec["ee_link"]
    if model.existFrame(ee):
        return False

    tip = spec["tip_link"]
    if not tip or not model.existFrame(tip):
        # Without a declared parent there is no honest place to put the frame.
        # Returning False lets ee_frame_id fall back and say so out loud.
        return False

    parent_id = model.getFrameId(tip)
    parent = model.frames[parent_id]
    placement = parent.placement * pin.SE3(
        np.eye(3), np.asarray(spec["offset"], dtype=float))
    model.addFrame(
        pin.Frame(ee, frame_parent_joint(parent), parent_id, placement,
                  pin.FrameType.OP_FRAME))
    return True


def frame_parent_joint(frame) -> int:
    """The joint a frame hangs on, under any Pinocchio.

    Pinocchio >= 3 calls it ``parentJoint``; 2.x (what ``pip install pin``
    resolves to next to numpy<2) calls it ``parent``.
    """
    joint = getattr(frame, "parentJoint", None)
    return int(frame.parent if joint is None else joint)


def ee_frame_id(model, spec: Optional[dict] = None) -> int:
    """The frame id the planner targets. Never the "last frame" by luck.

    Order: the cell's declared ``end_effector_link``, then the legacy names an
    older hand-written cell may use, and only then the last frame — which is
    logged as a guess by the caller rather than quietly trusted.
    """
    if spec and model.existFrame(spec["ee_link"]):
        return model.getFrameId(spec["ee_link"])
    for name in _LEGACY_EE_NAMES:
        if model.existFrame(name):
            return model.getFrameId(name)
    return model.nframes - 1


def load(cell_id: str, robot_ns: str, cell_base: Optional[str] = None):
    """``(model, data, ee_id)`` for one arm, with the TCP frame guaranteed.

    Raises ``FileNotFoundError`` when the cell describes no URDF for this robot —
    an unavailability the caller can report, rather than a model that silently
    plans for the wrong geometry.
    """
    import pinocchio as pin

    path = urdf_path(cell_id, robot_ns, cell_base)
    if path is None:
        raise FileNotFoundError(
            f"no URDF for robot '{robot_ns}' in cell '{cell_id}': neither "
            f"{cell_layout.cell_yaml(cell_id, cell_base)} nor "
            f"{cell_layout.urdf_glob(cell_id, robot_ns, cell_base)} resolves one")

    spec = tcp_spec(cell_id, robot_ns, cell_base)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = None
    key = (str(cell_id), str(robot_ns), str(cell_base), str(path), stamp,
           spec["ee_link"], spec["tip_link"], spec["offset"])
    with _LOCK:
        model = _MODELS.get(key)
    if model is None:
        # One model per arm and declaration, shared: parsing the URDF on every
        # call was the hot path's largest cost, and a fresh model per call is
        # what pushed in-use models out of the owner registry (below), switching
        # their self-collision check off. Callers get their own Data.
        model = pin.buildModelFromUrdf(str(path))
        add_tcp_frame(model, spec)
        with _LOCK:
            model = _MODELS.setdefault(key, model)
            _OWNERS[id(model)] = (model, str(cell_id), str(robot_ns))
    return model, model.createData(), ee_frame_id(model, spec)


# The loaded models, one per (cell, robot, URDF file + mtime, declared TCP): a
# re-converted cell or an edited URDF loads anew. Never evicted while the process
# runs — a cell has a handful of arms.
_MODELS: dict = {}
_LOCK = threading.Lock()

# Which (cell, robot) each loaded model belongs to, so checks that receive only a
# model (the shared collision predicates) can find the robot's self-collision
# model. Holds the model itself, so an id is never reused while it is listed, and
# is never evicted: an unlisted model has no self-collision check.
_OWNERS: dict = {}


def _register(model, cell_id: str, robot_ns: str) -> None:
    with _LOCK:
        _OWNERS[id(model)] = (model, str(cell_id), str(robot_ns))


def owner_of(model) -> Optional[tuple]:
    """``(cell_id, robot_ns)`` of a model this module loaded, else None."""
    with _LOCK:
        hit = _OWNERS.get(id(model))
    return (hit[1], hit[2]) if hit and hit[0] is model else None
