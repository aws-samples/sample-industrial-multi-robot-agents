"""The skill registry: discovery, prioritisation, per-cell overrides.

Overrides go through the cell store; here it is replaced by an in-memory dict so
the test needs no cell on disk.
"""
import pytest

from utils import skill_registry as sr


@pytest.fixture
def store(monkeypatch):
    files: dict = {}

    def read_yaml(cell_id, kind, name, **_):
        import yaml
        text = files.get((cell_id, kind, name))
        return yaml.safe_load(text) if text is not None else None

    def write_text(cell_id, kind, name, text, **_):
        files[(cell_id, kind, name)] = text
        return True

    monkeypatch.setattr(sr.cell_store, "read_yaml", read_yaml)
    monkeypatch.setattr(sr.cell_store, "write_text", write_text)
    return files


def _names(skills):
    return [s["name"] for s in skills]


def test_every_agent_is_listed_with_its_skills(store):
    agents = {s["agent"] for s in sr.list_skills()}
    assert agents == set(sr.AGENTS)


def test_the_new_planning_skill_is_discovered(store):
    names = _names(sr.list_skills(agent="trajectory"))
    assert "optimize-cycle-time" in names


def test_catalog_is_sorted_by_priority_within_an_agent(store):
    prios = [s["priority"] for s in sr.list_skills(agent="trajectory")]
    assert prios == sorted(prios, reverse=True)


def test_frontmatter_priority_is_the_default(store):
    skill = sr.get_skill("trajectory", "optimize-cycle-time")
    assert skill["priority"] == skill["default_priority"] == 55
    assert "optimize-cycle-time" in skill["content"]


def test_a_cell_override_reorders_the_catalog(store):
    sr.set_skill_priority("cellA", "trajectory", "optimize-cycle-time", priority=100)
    first = sr.list_skills("cellA", "trajectory")[0]
    assert first["name"] == "optimize-cycle-time" and first["overridden"]
    # other cells are untouched
    assert sr.get_skill("trajectory", "optimize-cycle-time", "cellB")["priority"] == 55


def test_a_disabled_skill_leaves_the_catalog(store):
    sr.set_skill_priority("cellA", "trajectory", "trajectory-knowledge", enabled=False)
    assert "trajectory-knowledge" not in _names(
        sr.list_skills("cellA", "trajectory", include_disabled=False))
    assert "trajectory-knowledge" not in sr.catalog_text("trajectory", "cellA")


def test_reset_restores_the_default(store):
    sr.set_skill_priority("cellA", "trajectory", "optimize-cycle-time", priority=5)
    sr.set_skill_priority("cellA", "trajectory", "optimize-cycle-time", reset=True)
    skill = sr.get_skill("trajectory", "optimize-cycle-time", "cellA")
    assert skill["priority"] == 55 and not skill["overridden"]


@pytest.mark.parametrize("agent,name", [("trajectory", "validate-collision"),
                                        ("cell", "collision-response")])
def test_required_safety_skills_cannot_be_disabled(store, agent, name):
    with pytest.raises(sr.SkillError):
        sr.set_skill_priority("cellA", agent, name, enabled=False)


def test_a_hand_edited_override_cannot_disable_a_required_skill(store):
    store[("cellA", sr.cell_store.KIND_CONFIG, sr.OVERRIDES_NAME)] = (
        "skills:\n  trajectory:\n    validate-collision: {enabled: false}\n")
    assert sr.get_skill("trajectory", "validate-collision", "cellA")["enabled"]


@pytest.mark.parametrize("bad", [-1, 101, "high"])
def test_out_of_range_priority_is_refused(store, bad):
    with pytest.raises(sr.SkillError):
        sr.set_skill_priority("cellA", "trajectory", "optimize-cycle-time", priority=bad)


def test_unknown_skill_or_agent_is_refused(store):
    with pytest.raises(sr.SkillError):
        sr.get_skill("trajectory", "no-such-skill")
    with pytest.raises(sr.SkillError):
        sr.list_skills(agent="robot")


def test_a_bound_cell_is_visible_to_the_run(store):
    with sr.bind_cell("cellA"):
        assert sr.current_cell() == "cellA"
    assert sr.current_cell() is None
