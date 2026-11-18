"""``roboballet`` — a trained graph-network policy plans the step's joint transit.

RoboBallet (Lai et al., *Science Robotics* 2025, doi:10.1126/scirobotics.ads1204)
trains a GNN policy with reinforcement learning to move many arms at once: every
0.1 s it commands every joint of every arm, and an arm whose motion would collide
is held still. Here that policy replaces the compound OMPL search of a PARALLEL
step; everything around the search is ``ompl-compound``'s step pipeline
(lift-off out of the seam, straight descent into the weld, one time base, every
emitted-leg check, the commit), so the legs are held to the same contract and
reach the BehaviourTree the same way (docs/adr/0002-roboballet-policy-backend.md).

The policy is trained by the ``policy-trainer`` service on SageMaker
(``agents/policy_trainer``) and mounted read-only at ``ROBOBALLET_POLICY_DIR``
(``/models``). Which one plans: the request's ``backend_options={"policy":
"<name>"}``, else ``ROBOBALLET_POLICY``, else ``current`` (the promoted one). A
named policy that is not installed refuses the request before it runs.
No policy is a refused plan, never a fallback to another planner.

A BUNDLED backend: in this repository, registered like a plugin, never the
default. Implementation in :mod:`.planner`, imported only when selected; this
module is stdlib-only so the registry stays light.
"""
from agents.trajectory.backends.contract import BackendSpec

SPEC = BackendSpec(
    name="roboballet",
    entry="agents.trajectory.backends.roboballet.planner:plan",
    description="trained RoboBallet GNN policy (RL, SageMaker) plans the step's joint "
                "transit; ompl-compound's strokes, timing and checks around it",
    joint_step=True,
    options=("policy",),
    check_options="agents.trajectory.backends.roboballet.planner:check_options",
)
