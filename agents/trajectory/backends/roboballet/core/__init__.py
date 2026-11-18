"""RoboBallet core: chain kinematics, obstacle primitives, the scene graph, the
graph network's forward pass, the kinematic environment and the policy file.

Shared VERBATIM by the planner (``..planner``) and the SageMaker training job
(``agents/policy_trainer/sagemaker``), which ships this directory in its source
bundle as the top-level package ``roboballet_core``. Hence: numpy + stdlib only,
and relative imports only.
"""
