from pathlib import Path
from langchain_core.tools import tool

_SKILLS_DIR = Path(__file__).parent.parent.parent.parent / "skills"


def _list_skills() -> str:
    lines = []
    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        # Parse description from frontmatter
        content = skill_md.read_text()
        description = ""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                for line in content[3:end].splitlines():
                    if line.startswith("description:"):
                        description = line.split(":", 1)[1].strip()
                        break
        lines.append(f"- {skill_dir.name}: {description}")
    return "\n".join(lines) if lines else "No skills available."


@tool
def read_skill(name: str) -> str:
    """Read the full instructions for a skill by name.
    Call list_skills() first if you don't know available skill names.
    Available skills: python-dev, data-analysis, web-research.
    """
    if name == "__list__":
        return _list_skills()

    skill_dir = _SKILLS_DIR / name
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        available = _list_skills()
        return f"Skill '{name}' not found.\n\nAvailable skills:\n{available}"

    return skill_md.read_text()
