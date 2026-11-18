"""``rrt`` — the legacy hand-rolled Space-Time RRT. NON-CONFORMING.

It does not couple the arms to each other, drops the tool-orientation constraint
and does not honour the two-phase speed/clearance split; results are stamped
``non_conforming_backend``. Kept selectable for comparison only, and imported
only when selected. Return home: the default planner.
"""
from agents.trajectory.backends.contract import BackendSpec

SPEC = BackendSpec(
    name="rrt",
    entry="agents.trajectory.backends.rrt.planner:plan_trajectories_rrt",
    conforming=False,
    description="legacy Space-Time RRT, no inter-arm coupling (comparison only)",
)
