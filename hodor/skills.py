"""Skill discovery and loading for repository-specific review guidelines.

This module uses the OpenHands SDK's skill system with Hodor-specific extensions.
Skills provide project-specific context for PR reviews.
"""

import logging
from pathlib import Path

from openhands.sdk.context import Skill, load_project_skills

logger = logging.getLogger(__name__)


def discover_skills(workspace: Path) -> list[Skill]:
    """Discover and load skills from workspace.

    Uses the OpenHands SDK's load_project_skills() for standard files:
    - .cursorrules, agents.md, claude.md, gemini.md

    Additionally loads Hodor-specific skills from:
    - .hodor/skills/*.md

    Args:
        workspace: Path to the repository workspace

    Returns:
        List of Skill objects. Empty list if no skills found.
    """
    seen_names: set[str] = set()
    all_skills: list[Skill] = []

    # 1. Load SDK-supported skills (third-party files: .cursorrules, agents.md, etc.)
    try:
        sdk_skills = load_project_skills(workspace)
        for skill in sdk_skills:
            if skill.name not in seen_names:
                all_skills.append(skill)
                seen_names.add(skill.name)
                logger.info(f"Found skill: {skill.name} (via SDK)")
    except Exception as e:
        logger.warning(f"Failed to load SDK skills: {e}")

    # 2. Load Hodor-specific skills from .hodor/skills/
    skills_dir = workspace / ".hodor" / "skills"
    if skills_dir.exists() and skills_dir.is_dir():
        skill_files = sorted(skills_dir.glob("*.md"))
        for skill_file in skill_files:
            skill_name = f".hodor/skills/{skill_file.name}"
            if skill_name in seen_names:
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                skill = Skill(name=skill_name, content=content, trigger=None)
                all_skills.append(skill)
                seen_names.add(skill_name)
                logger.info(f"Found skill: {skill_name}")
            except Exception as e:
                logger.warning(f"Failed to read {skill_file}: {e}")

    if all_skills:
        logger.info(f"Loaded {len(all_skills)} skill(s) from workspace")
    else:
        logger.debug("No skills found in workspace")

    return all_skills
