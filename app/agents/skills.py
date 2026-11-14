"""Skills: instructions the agent loads only when a task calls for them.

A skill is a directory under `app/skills/` (or SKILLS_DIR) holding a SKILL.md:

    ---
    name: research-report
    description: One sentence on what it is for and when to use it.
    ---
    The instructions, in Markdown.

Only the name and description go into the system prompt — that is the whole
cost of having a skill. The body is read into the conversation by the
`load_skill` tool when the model decides the task matches, so a long how-to
costs nothing on the turns that do not need it.

The file format is the same SKILL.md as Claude Code's and deepagents', so a
skill written for either drops in. deepagents' own SkillsMiddleware is not
used: it expects the model to open the file through a filesystem backend, and
Jarvis's `workspace_read_file` reads the user's uploaded workspace, not the server.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml
from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import tool

from app.agents.token_estimate import EXTRA_KEY, tokens

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?(.*)\Z", re.S)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def parse_skill(path: Path) -> Skill | None:
    """One SKILL.md, or None (with a warning saying why) if it is unusable."""
    try:
        m = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
        if not m:
            raise ValueError("no `---` frontmatter block at the top")
        meta = yaml.safe_load(m.group(1)) or {}
        name, description = str(meta.get("name") or "").strip(), str(meta.get("description") or "").strip()
        if not _NAME.match(name) or len(name) > 64:
            raise ValueError(f"name {name!r} must be lowercase letters, digits and hyphens, at most 64 characters")
        if name != path.parent.name:
            raise ValueError(f"name {name!r} must match its directory {path.parent.name!r}")
        if not description or len(description) > 1024:
            raise ValueError("description is required and at most 1024 characters")
        body = m.group(2).strip()
        if not body:
            raise ValueError("the instructions are empty")
        return Skill(name, description, body)
    except Exception as e:  # one bad skill must not take the app down
        logger.warning("skipping skill %s: %s", path, e)
        return None


def load_skills(root: Path) -> dict[str, Skill]:
    """Every valid `<root>/<name>/SKILL.md`, by name."""
    skills: dict[str, Skill] = {}
    if not root.is_dir():
        return skills
    for path in sorted(root.glob("*/SKILL.md")):
        if skill := parse_skill(path):
            skills[skill.name] = skill
    return skills


class SkillsMiddleware(AgentMiddleware):
    """Lists the skills in the system prompt and provides `load_skill`.

    Left out of the graph altogether when there are no skills (graph.py), so
    an install without any pays nothing — no prompt text, no tool schema.
    """

    def __init__(self, skills: dict[str, Skill]):
        self.skills = skills
        self.tools = [self._make_tool(skills)]

    @staticmethod
    def _make_tool(skills: dict[str, Skill]):
        @tool
        def load_skill(name: str) -> str:
            """Load the full instructions of one of the skills listed under
            "Skills" in your system prompt. Call it BEFORE starting a task a
            skill covers, then follow what it says.

            Args:
                name: The skill's name exactly as listed.
            """
            skill = skills.get(name.strip())
            if skill is None:
                return f"There is no skill named {name!r}. Available: {', '.join(sorted(skills))}."
            return f"# Skill: {skill.name}\n\n{skill.body}"

        return load_skill

    def listing(self) -> str:
        rows = "\n".join(f"- `{s.name}`: {s.description}" for s in self.skills.values())
        return (
            "## Skills\n"
            "A skill is a set of instructions for one kind of task. When the task matches one "
            "below, call `load_skill` with its name first and follow what it returns; "
            "otherwise ignore this section.\n" + rows
        )

    def _apply(self, request: ModelRequest) -> ModelRequest:
        text = self.listing()
        extra = {**(request.state.get(EXTRA_KEY) or {}), "skills": tokens(text)}
        return request.override(
            system_message=append_to_system_message(request.system_message, text),
            state={**request.state, EXTRA_KEY: extra},
        )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        return await handler(self._apply(request))
