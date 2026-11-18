"""Fixtures shared by the trajectory tests.

The upstream-fix port added the assembler's first-piece check (``behaviour_tree._chain_breaks``,
``start_gap``): an arm's first operation must start at the arm's HOME.

The modules below cache SYNTHETIC legs (``[0.1] * 6`` ...) in a real cell: legs
that chain onto each other but are not anchored at the cell's HOME, because what
they test is the gate / tree logic, not the chain's anchor. For them only, the
check is switched off here — in the test, never by a switch in the production
code. The check itself is covered by ``test_upstream_fixes_port.py``.
"""
import pytest

_SYNTHETIC_CHAIN_MODULES = {
    "test_coordinated_tree",
    "test_home_step_gate",
    "test_parked_arm_gate",
    "test_weld_step_gate",
}


@pytest.fixture(autouse=True)
def _synthetic_legs(request, monkeypatch):
    module = request.module.__name__.rsplit(".", 1)[-1]
    if module in _SYNTHETIC_CHAIN_MODULES:
        # Patched in agents.home_pose, which these modules' fixtures do not purge
        # and re-import (they do that to agents.trajectory.tools.*).
        import agents.home_pose
        monkeypatch.setattr(agents.home_pose, "program_start_pose",
                            lambda cell_id, robot_ns: None)
    yield
