"""One recovery table, held to the code and to the skill.

Every ``reason_code`` the planning tools can return has an entry in
``agents.trajectory.reason_codes.RECOVERY`` (what the agent can do about it), and
every entry has a section in the recovery skill — in both skill trees. A new code
without guidance is how the agent ends up re-sending the same failing request.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agents.trajectory import reason_codes  # noqa: E402

_EMITTED = re.compile(r'"reason_code":\s*"([a-z_]+)"|reason_code="([a-z_]+)"|'
                      r'_refuse\("([a-z_]+)"|f"(\{which\}_[a-z_]+|compound_\{label\}_invalid|'
                      r'\{label\}_outside_joint_limits)"')


def _emitted_codes() -> set:
    out: set = set()
    for path in (REPO / "agents" / "trajectory").rglob("*.py"):
        if path.name == "reason_codes.py":
            continue
        for m in _EMITTED.finditer(path.read_text()):
            code = next(g for g in m.groups() if g)
            out.update(reason_codes.PATTERNS.get(code, (code,)))
    return out


def test_every_emitted_code_has_a_recovery_entry():
    missing = sorted(_emitted_codes() - set(reason_codes.RECOVERY))
    assert not missing, f"codes the tools emit with no recovery entry: {missing}"


@pytest.mark.parametrize("tree", ["trajectory", "trajectory-classic"])
def test_every_recovery_entry_has_a_skill_section(tree):
    text = (REPO / "skills" / tree / "recover-from-planner-failure" / "SKILL.md").read_text()
    missing = sorted(c for c in reason_codes.RECOVERY if f"`{c}`" not in text)
    assert not missing, f"{tree}: no skill guidance for {missing}"


def test_entries_are_well_formed():
    for code, e in reason_codes.RECOVERY.items():
        assert set(e) == {"retry_same", "levers", "escalate"}, code
        assert e["escalate"] in {"reallocate", "resequence", "fix_cell", "stop"}, code


def test_the_skill_no_longer_offers_retired_or_self_defeating_levers():
    for tree in ("trajectory", "trajectory-classic"):
        text = (REPO / "skills" / tree / "recover-from-planner-failure" / "SKILL.md").read_text()
        assert "(try 30" not in text, "the hand-spin ladder turns the search off"
        assert "reduced\n   `approach_link_radius`" not in text
        assert "0.10 m approach clearance" not in text


def test_every_failure_carries_its_recovery(monkeypatch):
    from agents.trajectory.tools import planner_backend as pb
    monkeypatch.setitem(pb._BACKENDS, "ompl", lambda: (
        lambda c, reqs, **k: {"success": False, "trajectories": {}, "planning_time_s": 0,
                              "failures": [{"robot_ns": "r", "reason_code": "no_path_found",
                                            "error": "x"}]}))
    monkeypatch.setattr(pb, "_self_collision_refusals", lambda c, r: [])
    out = pb.dispatch_plan("c", [{"robot_ns": "r", "operation_id": "W1"},
                                 {"robot_ns": "r", "operation_id": "W2"}])
    by_op = {f["operation_id"]: f for f in out["failures"]}
    assert by_op["W1"]["recovery"]["retry_same"] is True        # operation_id stamped
    assert by_op["W2"]["reason_code"] == "duplicate_robot_request"
