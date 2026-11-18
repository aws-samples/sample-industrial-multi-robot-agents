"""``ompl`` — the production backend.

Pinocchio IK, OMPL geometric planners + PathSimplifier over the Pinocchio
collision model, orientation-constrained Cartesian descent, inter-robot
swept-volume coupling within a step, two-phase TOPP-RA timing. It also provides
the default return-home planner (``plan_home_trajectories``). Implementation in
:mod:`.planner`; imported only when the backend is used.
"""
from agents.trajectory.backends.contract import BackendSpec

SPEC = BackendSpec(
    name="ompl",
    entry="agents.trajectory.backends.ompl.planner:plan_trajectories_ompl",
    home="agents.trajectory.backends.ompl.planner:plan_home_trajectories",
    home_single="agents.trajectory.backends.ompl.planner:plan_home_trajectory",
    description="OMPL per arm, coupled through the swept volume of arms already planned",
)
