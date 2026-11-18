"""
The skill registry — one owner for "which skills exist, and in what order".

Skills live under ``skills/<agent>/<name>/SKILL.md`` (Agent Skills format: YAML
frontmatter + markdown body). Every reader goes through here: the trajectory
agent's catalog and ``load_skill``, the trajectory MCP server's skill tools, and
through those the orchestrator's ``/a2a/skills`` routes.

Prioritisation, in two layers:

  * **Default** — ``metadata.priority`` in the skill's own frontmatter
    (0–100, higher = preferred; :data:`DEFAULT_PRIORITY` when absent).
  * **Per-cell override** — ``skills.yaml`` in the cell's config, owned by the
    cell store like every other config file::

        skills:
          trajectory:
            optimize-cycle-time: {priority: 80}
            recover-from-planner-failure: {enabled: false}

Catalogs are sorted by effective priority, so the planner's LLM sees (and is told
to prefer) the higher-priority skill when several apply. A disabled skill is left
out of the catalog and refused by ``load_skill``.

A skill whose frontmatter says ``metadata.required: true`` — the collision gates —
cannot be disabled. Priorities only reorder guidance; they must never be a way to
switch a safety check off.

Import-safe: stdlib + a lazy ``yaml`` import, no ROS/LangChain.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from utils import cell_store

SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"

#: The agents that own skills, in display order.
AGENTS: tuple[str, ...] = ("trajectory", "trajectory-classic", "cell", "orchestrator")

DEFAULT_PRIORITY = 50
MIN_PRIORITY, MAX_PRIORITY = 0, 100

#: Per-cell override file, in the cell's config (``cell_store.KIND_CONFIG``).
OVERRIDES_NAME = "skills.yaml"


class SkillError(ValueError):
    """A request named a skill/agent that does not exist, or asked for an
    override the registry refuses (e.g. disabling a required skill)."""


# ── the cell a planning run is for ───────────────────────────────────────────
# ``load_skill`` is an LLM tool with no cell argument; the run binds its cell
# here (same per-run ContextVar pattern as the model and the planner backend) so
# a skill the cell disabled cannot be loaded mid-run.
_ACTIVE_CELL: ContextVar[Optional[str]] = ContextVar("skill_cell", default=None)


@contextmanager
def bind_cell(cell_id: Optional[str]):
    token = _ACTIVE_CELL.set(cell_id)
    try:
        yield cell_id
    finally:
        _ACTIVE_CELL.reset(token)


def current_cell() -> Optional[str]:
    return _ACTIVE_CELL.get()


# ── per-RUN overrides ────────────────────────────────────────────────────────
# A planning request may re-prioritise (or switch off) skills for that one run
# — plan_trajectories(skill_priorities=...). They sit ON TOP of the cell's
# skills.yaml for every read made during the run (the prompt catalog, load_skill)
# and are never written to it.
_RUN_OVERRIDES: ContextVar[dict] = ContextVar("skill_run_overrides", default={})


@contextmanager
def bind_run_overrides(overrides: Optional[dict]):
    """``{agent: {skill: {"priority"?: int, "enabled"?: bool}}}`` for this run."""
    token = _RUN_OVERRIDES.set(dict(overrides or {}))
    try:
        yield
    finally:
        _RUN_OVERRIDES.reset(token)


def set_run_overrides(overrides: Optional[dict]) -> None:
    """Set this run's overrides inside a :func:`bind_run_overrides` scope."""
    _RUN_OVERRIDES.set(dict(overrides or {}))


def current_run_overrides() -> dict:
    return _RUN_OVERRIDES.get()


# ── per-RUN temporary skills ─────────────────────────────────────────────────
# A planning request may also bring skills of its own — plan_trajectories(
# run_skills=[...]) / POST /a2a/plan — to try a different strategy without
# shipping a SKILL.md. They exist for that one run only (and its resume): listed
# in its catalog, loadable with load_skill, never written anywhere. They are
# GUIDANCE: every collision gate is code and a required skill cannot be
# disabled, so no temporary skill can loosen what the planner must check. They
# may not reuse an existing skill's name — a request cannot shadow, say,
# validate-collision with a text of its own.
_RUN_SKILLS: ContextVar[dict] = ContextVar("skill_run_skills", default={})

#: Limits on what one request may bring.
RUN_SKILLS_MAX = 5
RUN_SKILL_NAME_RE = r"^[a-z0-9][a-z0-9-]{1,47}$"
RUN_SKILL_DESCRIPTION_MAX = 600
RUN_SKILL_CONTENT_MAX = 12000
DEFAULT_RUN_SKILL_PRIORITY = 60


@contextmanager
def bind_run_skills(skills: Optional[dict]):
    """``{agent: [skill, ...]}`` (as :func:`normalise_run_skills` returns) for this run."""
    token = _RUN_SKILLS.set(dict(skills or {}))
    try:
        yield
    finally:
        _RUN_SKILLS.reset(token)


def set_run_skills(skills: Optional[dict]) -> None:
    """Set this run's temporary skills inside a :func:`bind_run_skills` scope."""
    _RUN_SKILLS.set(dict(skills or {}))


def current_run_skills() -> dict:
    return _RUN_SKILLS.get()


def normalise_run_skills(agent: str, requested) -> list:
    """Validate a request's temporary skills; return their canonical form.

    Each entry is ``{"name", "description", "content", "priority"?}`` — or
    ``{"skill_md": "<a whole SKILL.md>"}``, frontmatter and all (as
    :func:`skill_template` shows). Raises :class:`SkillError` for anything the
    registry would not load: a bad or taken name, a missing description or body,
    over-long text, ``required: true``, or more than :data:`RUN_SKILLS_MAX`.
    """
    import re  # noqa: PLC0415
    _check_agent(agent)
    if requested in (None, [], {}):
        return []
    if not isinstance(requested, list):
        raise SkillError("run_skills must be a list of skills")
    if len(requested) > RUN_SKILLS_MAX:
        raise SkillError(f"at most {RUN_SKILLS_MAX} temporary skills per run, "
                         f"got {len(requested)}")
    taken = {s["name"] for s in _discover(agent)} | {s["dir"] for s in _discover(agent)}
    out, seen = [], set()
    for i, entry in enumerate(requested):
        if not isinstance(entry, dict):
            raise SkillError(f"run_skills[{i}] must be an object")
        meta: dict = {}
        if entry.get("skill_md") is not None:
            if not isinstance(entry["skill_md"], str):
                raise SkillError(f"run_skills[{i}].skill_md must be text")
            meta, body = _split_frontmatter(entry["skill_md"])
            if not meta:
                raise SkillError(f"run_skills[{i}].skill_md has no YAML frontmatter "
                                 f"(--- name/description --- then the body)")
        else:
            unknown = set(entry) - {"name", "description", "content", "priority"}
            if unknown:
                raise SkillError(f"run_skills[{i}]: unknown field(s) {sorted(unknown)}")
            body = entry.get("content")
        md = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        name = str(entry.get("name") or meta.get("name") or "").strip()
        desc = " ".join(str(entry.get("description") or meta.get("description") or "").split())
        prio = entry.get("priority", md.get("priority", DEFAULT_RUN_SKILL_PRIORITY))
        if not re.match(RUN_SKILL_NAME_RE, name):
            raise SkillError(f"run_skills[{i}]: name {name!r} must match "
                             f"{RUN_SKILL_NAME_RE} (lower-case, digits, '-')")
        if name in taken:
            raise SkillError(f"run_skills[{i}]: {agent}/{name} already exists — a "
                             f"temporary skill cannot replace one; pick a new name "
                             f"(and disable the old one with skill_priorities)")
        if name in seen:
            raise SkillError(f"run_skills[{i}]: {name} given twice")
        if not desc:
            raise SkillError(f"run_skills[{i}] ({name}): a description is required — "
                             f"it is what the planner reads to decide when to load it")
        if len(desc) > RUN_SKILL_DESCRIPTION_MAX:
            raise SkillError(f"run_skills[{i}] ({name}): description over "
                             f"{RUN_SKILL_DESCRIPTION_MAX} characters")
        if not isinstance(body, str) or not body.strip():
            raise SkillError(f"run_skills[{i}] ({name}): the skill has no instructions")
        if len(body) > RUN_SKILL_CONTENT_MAX:
            raise SkillError(f"run_skills[{i}] ({name}): instructions over "
                             f"{RUN_SKILL_CONTENT_MAX} characters")
        if md.get("required"):
            raise SkillError(f"run_skills[{i}] ({name}): a temporary skill cannot be "
                             f"required")
        try:
            p = int(prio)
        except (TypeError, ValueError):
            raise SkillError(f"run_skills[{i}] ({name}): priority must be an integer") from None
        if not MIN_PRIORITY <= p <= MAX_PRIORITY:
            raise SkillError(f"run_skills[{i}] ({name}): priority must be "
                             f"{MIN_PRIORITY}-{MAX_PRIORITY}, got {p}")
        seen.add(name)
        out.append({"name": name, "description": desc, "priority": p,
                    "content": body.strip() + "\n"})
    return out


def normalise_run_overrides(agent: str, requested: Optional[dict]) -> dict:
    """Validate a request's ``{skill: priority | {"priority", "enabled"}}``.

    Returns ``{agent: {skill_name: {...}}}``. Raises :class:`SkillError` for an
    unknown skill, a priority outside 0-100, or disabling a required skill — a
    request that names something that does not exist is refused, not ignored.
    """
    entries: dict = {}
    for name, value in (requested or {}).items():
        skill = _find(agent, str(name))
        spec = value if isinstance(value, dict) else {"priority": value}
        unknown = set(spec) - {"priority", "enabled"}
        if unknown:
            raise SkillError(f"{name}: unknown field(s) {sorted(unknown)} "
                             f"(want priority and/or enabled)")
        entry: dict = {}
        if spec.get("priority") is not None:
            try:
                p = int(spec["priority"])
            except (TypeError, ValueError):
                raise SkillError(f"{name}: priority must be an integer, "
                                 f"got {spec['priority']!r}") from None
            if not MIN_PRIORITY <= p <= MAX_PRIORITY:
                raise SkillError(f"{name}: priority must be {MIN_PRIORITY}-"
                                 f"{MAX_PRIORITY}, got {p}")
            entry["priority"] = p
        if "enabled" in spec:
            if spec["enabled"] is False and skill["required"]:
                raise SkillError(f"{agent}/{skill['name']} is a required skill and "
                                 f"cannot be disabled")
            entry["enabled"] = bool(spec["enabled"])
        if entry:
            entries[skill["name"]] = entry
    return {agent: entries} if entries else {}


# ── discovery ────────────────────────────────────────────────────────────────

def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    import yaml  # noqa: PLC0415
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), parts[2].lstrip("\n")


def _clamp_priority(value, fallback: int = DEFAULT_PRIORITY) -> int:
    try:
        p = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(MIN_PRIORITY, min(MAX_PRIORITY, p))


def _check_agent(agent: str) -> str:
    if agent not in AGENTS:
        raise SkillError(f"unknown agent {agent!r} (want one of {', '.join(AGENTS)})")
    return agent


def _discover(agent: Optional[str] = None) -> list[dict]:
    """Every skill on disk with its frontmatter defaults. No overrides applied."""
    agents = (_check_agent(agent),) if agent else AGENTS
    out: list[dict] = []
    for ag in agents:
        for path in sorted((SKILLS_ROOT / ag).glob("*/SKILL.md")):
            try:
                meta, _ = _split_frontmatter(path.read_text())
            except OSError:
                continue
            md = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
            out.append({
                "agent": ag,
                "name": str(meta.get("name") or path.parent.name),
                "dir": path.parent.name,
                "description": " ".join(str(meta.get("description") or "").split()),
                "category": md.get("category"),
                "version": md.get("version"),
                "default_priority": _clamp_priority(md.get("priority")),
                "required": bool(md.get("required", False)),
                "path": path.relative_to(SKILLS_ROOT.parent).as_posix(),
                "temporary": False,
            })
        for t in current_run_skills().get(ag) or []:
            out.append({"agent": ag, "name": t["name"], "dir": t["name"],
                        "description": t["description"], "category": "temporary",
                        "version": None, "default_priority": t["priority"],
                        "required": False, "path": None, "temporary": True,
                        "_content": t["content"]})
    return out


def _find(agent: str, name: str) -> dict:
    _check_agent(agent)
    for s in _discover(agent):
        if name in (s["name"], s["dir"]):
            return s
    raise SkillError(f"no {agent} skill named {name!r}")


# ── per-cell overrides ───────────────────────────────────────────────────────

def _read_overrides(cell_id: Optional[str]) -> dict:
    """``{agent: {skill: {"priority"?: int, "enabled"?: bool}}}`` for the cell.

    An absent file is "no overrides". A store outage propagates — "I could not ask"
    must not read as "the cell reorders nothing".
    """
    if not cell_id:
        return {}
    data = cell_store.read_yaml(cell_id, cell_store.KIND_CONFIG, OVERRIDES_NAME)
    skills = (data or {}).get("skills") if isinstance(data, dict) else None
    if not isinstance(skills, dict):
        return {}
    return {ag: {k: v for k, v in (entries or {}).items() if isinstance(v, dict)}
            for ag, entries in skills.items()
            if ag in AGENTS and isinstance(entries, dict)}


def _write_overrides(cell_id: str, overrides: dict) -> bool:
    import yaml  # noqa: PLC0415
    body = {"skills": {ag: dict(sorted(entries.items()))
                       for ag, entries in sorted(overrides.items()) if entries}}
    text = ("# Per-cell skill prioritisation — see utils/skill_registry.py.\n"
            "# priority: 0-100, higher = preferred. enabled: false hides a skill\n"
            "# from the planner (skills marked required cannot be disabled).\n"
            + yaml.safe_dump(body, sort_keys=False))
    return cell_store.write_text(cell_id, cell_store.KIND_CONFIG, OVERRIDES_NAME, text)


def _overrides_for_reading(cell_id: Optional[str]) -> dict:
    """The cell's overrides with this run's on top (never written back)."""
    merged = {ag: {k: dict(v) for k, v in entries.items()}
              for ag, entries in _read_overrides(cell_id).items()}
    for ag, entries in current_run_overrides().items():
        for name, entry in (entries or {}).items():
            merged.setdefault(ag, {}).setdefault(name, {}).update(entry)
            merged[ag][name]["run"] = True
    return merged


def _effective(skill: dict, overrides: dict) -> dict:
    o = (overrides.get(skill["agent"]) or {}).get(skill["name"]) or {}
    priority = _clamp_priority(o.get("priority"), skill["default_priority"])
    enabled = True if skill["required"] else bool(o.get("enabled", True))
    return {**skill, "priority": priority, "enabled": enabled,
            "overridden": bool(o), "run_override": bool(o.get("run"))}


# ── public API ───────────────────────────────────────────────────────────────

def list_skills(cell_id: Optional[str] = None, agent: Optional[str] = None,
                include_disabled: bool = True) -> list[dict]:
    """Skills with their EFFECTIVE priority for ``cell_id`` (and the running
    plan's own overrides, if any), highest first within each agent."""
    overrides = _overrides_for_reading(cell_id)
    skills = [_effective(s, overrides) for s in _discover(agent)]
    if not include_disabled:
        skills = [s for s in skills if s["enabled"]]
    skills = [{k: v for k, v in s.items() if k != "_content"} for s in skills]
    order = {ag: i for i, ag in enumerate(AGENTS)}
    skills.sort(key=lambda s: (order[s["agent"]], -s["priority"], s["name"]))
    return skills


def get_skill(agent: str, name: str, cell_id: Optional[str] = None) -> dict:
    """One skill's metadata plus its full SKILL.md text."""
    skill = _effective(_find(agent, name), _overrides_for_reading(cell_id))
    if skill.get("temporary"):
        skill["content"] = skill.pop("_content")
    else:
        skill["content"] = (SKILLS_ROOT.parent / skill["path"]).read_text()
    return skill


def set_skill_priority(cell_id: str, agent: str, name: str, *,
                       priority: Optional[int] = None,
                       enabled: Optional[bool] = None,
                       reset: bool = False) -> dict:
    """Set (or ``reset`` to the frontmatter default) one skill's override for a
    cell. Returns the skill as it now stands."""
    if not cell_id or not isinstance(cell_id, str):
        raise SkillError("cell_id is required")
    skill = _find(agent, name)
    if skill.get("temporary"):
        raise SkillError(f"{agent}/{name} is a temporary skill of the running plan; "
                         f"it has no cell setting")
    if priority is not None:
        try:
            p = int(priority)
        except (TypeError, ValueError):
            raise SkillError(f"priority must be an integer, got {priority!r}") from None
        if not MIN_PRIORITY <= p <= MAX_PRIORITY:
            raise SkillError(f"priority must be {MIN_PRIORITY}-{MAX_PRIORITY}, got {p}")
        priority = p
    if enabled is False and skill["required"]:
        raise SkillError(f"{agent}/{skill['name']} is a required skill and cannot "
                         f"be disabled")

    overrides = _read_overrides(cell_id)
    entries = overrides.setdefault(agent, {})
    if reset:
        entries.pop(skill["name"], None)
    else:
        entry = dict(entries.get(skill["name"]) or {})
        if priority is not None:
            entry["priority"] = priority
        if enabled is not None:
            entry["enabled"] = bool(enabled)
        if entry:
            entries[skill["name"]] = entry
    if not _write_overrides(cell_id, overrides):
        raise SkillError(f"could not write {OVERRIDES_NAME} for cell {cell_id!r}")
    return _effective(skill, _read_overrides(cell_id))


def catalog_text(agent: str, cell_id: Optional[str] = None) -> str:
    """The prompt catalog: enabled skills, highest priority first."""
    lines = [f"  - [priority {s['priority']}"
             + (", temporary: supplied by the operator for this run" if s.get("temporary") else "")
             + f"] {s['name']}: {s['description']}"
             for s in list_skills(cell_id, agent, include_disabled=False)]
    return "\n".join(lines) or "  (no skills found)"


# ── what a skill looks like ──────────────────────────────────────────────────

_TEMPLATE_SKILL = """---
name: stagger-shared-boundary
description: >-
  Use when two arms on the same side of the part both have welds near their
  shared boundary (run_mrta_pipeline reports them in `contention`). Keeps
  those welds in different steps and fills each step with far-away work.
metadata:
  priority: 70
  category: optimization
---

# stagger-shared-boundary

Two neighbouring arms that reach their shared boundary at the same time block
each other: one must park while the other welds, and the program goes serial.

## When to use

After `run_mrta_pipeline`, when `contention` lists weld pairs of two arms on
the same side (e.g. robot1|robot4 at the B-pillar).

## Steps

1. Keep the pipeline's step composition; never plan an arm against a parked
   neighbour.
2. For each contended pair, check that its two welds are in different steps.
   If a step has both, move ONE of them to the nearest step whose other arms
   are far from it, and plan the step with all its arms in one call.
3. If a step still fails `validate_trajectory_set`, apply its `delay_fix`
   before splitting the step.
4. Report the steps you changed and the cycle time before/after
   (`program_cost`).

## Never

- Never drop an arm, disable a collision check or reassign a locked weld.
"""


def skill_template(agent: str = "trajectory") -> dict:
    """What a skill looks like, and what a request may bring: a complete example
    SKILL.md, the fields, the limits, and request bodies that use it."""
    _check_agent(agent)
    existing = [s["name"] for s in _discover(agent) if not s.get("temporary")]
    return {
        "agent": agent,
        "format": ("Agent Skills: YAML frontmatter (name, description, "
                   "metadata.priority/category) between '---' lines, then markdown "
                   "instructions. The planner sees only `[priority] name: "
                   "description` in its catalog and calls load_skill(name) for the "
                   "body — so the description must say WHEN to use it."),
        "fields": {
            "name": f"required; {RUN_SKILL_NAME_RE}; must not be an existing skill",
            "description": f"required; <= {RUN_SKILL_DESCRIPTION_MAX} characters; "
                           f"when to use it",
            "content": f"required; the markdown instructions; <= "
                       f"{RUN_SKILL_CONTENT_MAX} characters",
            "priority": f"optional; {MIN_PRIORITY}-{MAX_PRIORITY}, default "
                        f"{DEFAULT_RUN_SKILL_PRIORITY}; higher = preferred",
            "skill_md": "alternative to the fields above: a whole SKILL.md text",
        },
        "limits": {"max_skills_per_run": RUN_SKILLS_MAX,
                   "lifetime": "the one planning run (and its resume); nothing is "
                               "written to the cell or the repository",
                   "safety": "guidance only — collision gates are code and "
                             "required skills cannot be disabled"},
        "existing_skills": existing,
        "example_skill_md": _TEMPLATE_SKILL,
        "example_requests": {
            "temporary_skill": {"program": "<task program>", "mode": "stepwise",
                                "run_skills": [{"skill_md": "<example_skill_md>"}]},
            "temporary_skill_fields": {
                "program": "<task program>",
                "run_skills": [{"name": "prefer-short-transits",
                                "description": "Use when choosing between via points: "
                                               "prefer the one with the shorter transit.",
                                "content": "## Steps\n1. ...", "priority": 65}]},
            "disable_and_reprioritise": {
                "program": "<task program>",
                "skill_priorities": {"optimize-cycle-time": 90,
                                     "trajectory-knowledge": {"enabled": False}}},
            "adjust_the_prompt": {
                "program": "<task program>",
                "instructions": "Minimise cycle time; keep every step with at least "
                                "three arms working where the composition allows."},
        },
    }
