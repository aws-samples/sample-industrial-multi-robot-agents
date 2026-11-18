"""Compatibility alias — this backend now lives in ``agents/trajectory/backends/ompl_compound/``.

The old import path keeps working AS THE SAME MODULE (not a copy), so code and
tests that import, or monkeypatch, ``agents.trajectory.tools.ompl_compound_planner`` act on
the real backend. New code imports ``agents.trajectory.backends.ompl_compound.planner``.
"""
import sys

from agents.trajectory.backends.ompl_compound import planner as _planner

sys.modules[__name__] = _planner
