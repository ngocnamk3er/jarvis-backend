"""Whose files a user can reach: their own, and their departments'.

A department is a Keycloak group directly under settings.DEPARTMENTS_GROUP ("/departments/Kế toán"). Its members can
read its files; the members of its editors subgroup ("/departments/Kế toán/editors") can also change them. Membership
comes from the token's `groups` claim, so it is whatever Keycloak says at the user's last token refresh.

The file service knows nothing of departments: it keeps every file under an owner id, and a department's files are
simply owned by "dept:<slug of its name>". Every check of who may do what is made here, in the backend, before the
file service is called with that owner id.
"""

import re
import unicodedata
from dataclasses import dataclass

from app.core.config import settings

PERSONAL = "personal"
_DEPT_PREFIX = "dept:"


@dataclass(frozen=True)
class Space:
    id: str  # "personal" or "dept:<slug>"; what the API and the agent's citations carry
    name: str
    kind: str  # "personal" | "department"
    owner: str  # the file service's user_id for this space
    can_edit: bool

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "can_edit": self.can_edit}


class SpaceError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def slug(name: str) -> str:
    """"Kế toán" -> "ke-toan": an owner id that stays plain ASCII whatever the group is called."""
    folded = unicodedata.normalize("NFKD", name.replace("đ", "d").replace("Đ", "D"))
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    out = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return out or "x" + name.encode().hex()[:24]


def departments(groups: tuple[str, ...] | list[str]) -> list[tuple[str, bool]]:
    """[(department name, can edit)] from the token's group paths, by name. Being in the editors subgroup makes one a
    member too; any other subgroup of a department makes one a plain member."""
    root = settings.DEPARTMENTS_GROUP.rstrip("/") + "/"
    found: dict[str, bool] = {}
    for path in groups:
        if not path.startswith(root):
            continue
        parts = [p for p in path[len(root):].split("/") if p]
        if not parts:
            continue
        name = parts[0]
        editor = len(parts) == 2 and parts[1] == settings.DEPARTMENT_EDITORS_GROUP
        found[name] = found.get(name, False) or editor
    return sorted(found.items(), key=lambda kv: kv[0].casefold())


def department_spaces(groups) -> list[Space]:
    spaces: dict[str, Space] = {}
    for name, can_edit in departments(groups):
        sid = _DEPT_PREFIX + slug(name)
        prior = spaces.get(sid)
        # Two groups whose names fold to the same slug are one space; either one's edit right is enough.
        spaces[sid] = Space(sid, prior.name if prior else name, "department", sid, can_edit or bool(prior and prior.can_edit))
    return list(spaces.values())


def spaces_for(user_sub: str, groups) -> list[Space]:
    return [Space(PERSONAL, "My files", "personal", user_sub, True), *department_spaces(groups)]


def resolve(user_sub: str, groups, space_id: str | None, *, write: bool = False) -> Space:
    """The space `space_id` names, if this user may use it that way. None or "" is the user's own."""
    if not space_id or space_id == PERSONAL:
        return spaces_for(user_sub, ())[0]
    for space in department_spaces(groups):
        if space.id == space_id:
            if write and not space.can_edit:
                raise SpaceError(f"You can read {space.name}'s documents but not change them.", 403)
            return space
    # The same answer for a space that does not exist and one the user is not in: neither should be told apart.
    raise SpaceError("No such document space.", 404)


def for_agent(groups) -> list[dict]:
    """The department spaces as the agent's tools get them, through the run's configurable (key "file_spaces")."""
    return [{"id": s.id, "name": s.name, "owner": s.owner, "can_edit": s.can_edit} for s in department_spaces(groups)]
