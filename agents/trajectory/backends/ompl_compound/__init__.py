"""``ompl-compound`` — every arm moving in a step planned as ONE path.

The step's transit for all moving arms is a single path in an OMPL
CompoundStateSpace, arm-vs-arm checked in every state and timed on one clock
(docs/adr/0001-ompl-compound-backend.md). Lift-off and descent strokes per arm
reuse the ``ompl`` backend's, so this package builds on :mod:`..ompl.planner`.
Return home: the default planner.
"""
from agents.trajectory.backends.contract import BackendSpec

SPEC = BackendSpec(
    name="ompl-compound",
    entry="agents.trajectory.backends.ompl_compound.planner:plan_trajectories_ompl_compound",
    description="all moving arms of a step in one OMPL compound state space",
    joint_step=True,
)
