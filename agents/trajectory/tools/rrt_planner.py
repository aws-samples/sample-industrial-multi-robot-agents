"""Compatibility alias — this backend now lives in ``agents/trajectory/backends/rrt/``.

The old import path keeps working AS THE SAME MODULE (not a copy), so code and
tests that import, or monkeypatch, ``agents.trajectory.tools.rrt_planner`` act on
the real backend. New code imports ``agents.trajectory.backends.rrt.planner``.
"""
import sys

from agents.trajectory.backends.rrt import planner as _planner

sys.modules[__name__] = _planner
