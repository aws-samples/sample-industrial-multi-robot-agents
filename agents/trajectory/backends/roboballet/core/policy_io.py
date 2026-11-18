"""A trained policy on disk: ``policy.npz`` (weights) + ``policy_card.json``.

The card is what makes a file a policy this code may run::

    {"schema_version": 1, "kind": "roboballet-policy", "arch": {...},
     "dims": graph.DIMS, "dt": 0.1, "tol_m": 0.025, "tol_rad": 0.26,
     "k_obstacles": 24, "training": {"job": ..., "cells": [...], "steps": ...}}

A card that does not match this build's graph schema is refused, never adapted:
a feature in the wrong slot is a policy steering by the wrong numbers.

numpy + stdlib only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from . import graph as G
from .gnn import forward, module_shapes

KIND = "roboballet-policy"
WEIGHTS = "policy.npz"
CARD = "policy_card.json"


class PolicyError(ValueError):
    """A policy that cannot be used, and why."""


def save(directory: str | Path, params: Mapping[str, np.ndarray], card: dict) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / WEIGHTS, **{k: np.asarray(v, np.float32) for k, v in params.items()})
    full = {"kind": KIND, "schema_version": G.SCHEMA_VERSION, "dims": G.DIMS, **card}
    (d / CARD).write_text(json.dumps(full, indent=2, default=str))
    return d


def check_card(card: Mapping) -> None:
    if card.get("kind") != KIND:
        raise PolicyError(f"not a {KIND} card (kind={card.get('kind')!r})")
    if int(card.get("schema_version", -1)) != G.SCHEMA_VERSION:
        raise PolicyError(f"policy graph schema v{card.get('schema_version')} does not match "
                          f"this planner's v{G.SCHEMA_VERSION} — retrain or use a matching build")
    if dict(card.get("dims") or {}) != G.DIMS:
        raise PolicyError(f"policy feature dims {card.get('dims')} != planner's {G.DIMS}")


class Policy:
    """A loaded, validated policy: ``act(graph) -> (R, MAX_DOF)`` in [-1, 1]."""

    def __init__(self, params: Mapping[str, np.ndarray], card: Mapping, name: str = ""):
        check_card(card)
        self.card, self.name = dict(card), name or str(card.get("name", ""))
        self.arch = dict(card["arch"])
        missing = [f"{m}.0.w" for m in module_shapes(self.arch) if f"{m}.0.w" not in params]
        if missing:
            raise PolicyError(f"policy weights lack {missing[:3]}{'…' if len(missing) > 3 else ''}")
        self.params = {k: np.asarray(v, np.float64) for k, v in params.items()}

    @classmethod
    def load(cls, directory: str | Path, name: Optional[str] = None) -> "Policy":
        d = Path(directory)
        if not (d / CARD).is_file() or not (d / WEIGHTS).is_file():
            raise PolicyError(f"{d} holds no policy ({CARD} + {WEIGHTS})")
        card = json.loads((d / CARD).read_text())
        with np.load(d / WEIGHTS) as z:
            params = {k: z[k] for k in z.files}
        return cls(params, card, name or d.name)

    def act(self, g: Mapping) -> np.ndarray:
        batched = G.stack([g])
        return forward(self.params, self.arch, batched)[0]

    @property
    def tol_m(self) -> float:
        return float(self.card.get("tol_m", 0.025))

    @property
    def tol_rad(self) -> float:
        return float(self.card.get("tol_rad", np.radians(15.0)))

    @property
    def dt(self) -> float:
        return float(self.card.get("dt", 0.1))

    @property
    def k_obstacles(self) -> int:
        return int(self.card.get("k_obstacles", G.K_OBSTACLES))
