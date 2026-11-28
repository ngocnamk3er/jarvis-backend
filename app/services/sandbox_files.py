"""The files in a conversation's sandbox, for the user to see and handle (the chat's Sandbox panel).

Paths are relative to the sandbox's working directory, where the agent's files and the user's attachments go. That
directory is the runtime image's own (/app on the stock python runtime) and also holds the runtime's server files
(SANDBOX_HIDDEN_ENTRIES): those are never listed, and no path may start with one, so they cannot be deleted, renamed
or overwritten from here. No path may be absolute or climb out with `..`.

Looking never creates or wakes a sandbox: a conversation without one says so, and a paused one (its pod gone, its
files kept) is listed only when asked to wake it. Changing something (delete, rename, save) needs the files, so it
wakes a paused sandbox; making a folder may start one, as attaching a file does.
"""

import json
import mimetypes
import shlex
from datetime import UTC, datetime

import httpx

from app.agents.tools import sandbox_manager
from app.clients import file_client
from app.core.config import settings

MAX_ENTRIES = 500
MAX_SAVE_BYTES = 50 << 20        # what one file copied to the user's Files may weigh (the attachment limit)
MAX_NAME = 255

# Lists one folder as JSON: [name, "dir" | "file" | "link", size, mtime]. Python, not `ls`/`find` output, so a name
# with spaces, tabs or newlines comes through intact.
_LIST = """
import json, os, sys
out = []
with os.scandir(sys.argv[1]) as it:
    for e in it:
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        kind = "dir" if e.is_dir(follow_symlinks=False) else "link" if e.is_symlink() else "file"
        out.append([e.name, kind, st.st_size, st.st_mtime])
        if len(out) > int(sys.argv[2]):
            break
print(json.dumps(out))
"""


class SandboxPathError(ValueError):
    """A path or name this API does not accept."""


class SandboxFileError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.message = message
        self.status = status


def hidden() -> frozenset[str]:
    return frozenset(n.strip() for n in settings.SANDBOX_HIDDEN_ENTRIES.split(",") if n.strip())


def clean(path: str, *, allow_root: bool = False) -> str:
    """`path` as a relative path inside the working directory ("" is the directory itself)."""
    raw = (path or "").strip()
    p = sandbox_manager.normalize_workspace_path(raw).strip()
    if p.startswith("/"):
        raise SandboxPathError("Paths are relative to the sandbox's working directory.")
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if any(x == ".." for x in parts) or "\0" in p:
        raise SandboxPathError("A path cannot go up out of the sandbox's working directory.")
    if not parts:
        if allow_root:
            return ""
        raise SandboxPathError("Name a file or folder.")
    if parts[0] in hidden():
        raise SandboxPathError(f"'{parts[0]}' belongs to the sandbox itself.")
    return "/".join(parts)


def valid_name(name: str) -> str:
    n = (name or "").strip()
    if not n or n in (".", "..") or "/" in n or "\0" in n or len(n) > MAX_NAME:
        raise SandboxPathError("A name is 1 to 255 characters, without '/'.")
    return n


def _join(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


async def _run(thread_id: str, command: str) -> dict:
    r = await sandbox_manager.exec_bash(thread_id, command)
    if r.get("error") or r.get("timed_out"):
        raise SandboxFileError("The sandbox is not reachable right now.", 503)
    return r


async def _must_have_sandbox(thread_id: str) -> None:
    if await sandbox_manager.sandbox_state(thread_id) == "none":
        raise SandboxFileError("This conversation has no sandbox.", 404)


async def list_dir(thread_id: str, path: str = "", wake: bool = False) -> dict:
    """{state, path, entries, truncated}. entries: folders first, then files, by name."""
    path = clean(path, allow_root=True)
    state = await sandbox_manager.sandbox_state(thread_id)
    if state == "none" or (state == "paused" and not wake):
        return {"state": state, "path": path, "entries": [], "truncated": False}
    r = await _run(thread_id, f"python3 -c {shlex.quote(_LIST)} {shlex.quote(path or '.')} {MAX_ENTRIES}")
    if r.get("exit_code") != 0:
        raise SandboxFileError("There is no such folder in the sandbox.", 404)
    rows = json.loads(r.get("stdout") or "[]")
    skip = hidden() if not path else frozenset()
    entries = [
        {
            "name": name,
            "path": _join(path, name),
            "type": kind,
            "size": None if kind == "dir" else size,
            "modified": datetime.fromtimestamp(mtime, UTC).isoformat(timespec="seconds"),
        }
        for name, kind, size, mtime in rows
        if name not in skip
    ]
    entries.sort(key=lambda e: (e["type"] != "dir", e["name"].casefold()))
    return {"state": "running", "path": path, "entries": entries[:MAX_ENTRIES], "truncated": len(rows) > MAX_ENTRIES}


async def delete(thread_id: str, path: str) -> None:
    p = shlex.quote(clean(path))
    await _must_have_sandbox(thread_id)
    r = await _run(thread_id, f"if [ ! -e {p} ] && [ ! -L {p} ]; then exit 3; fi; rm -rf -- {p}")
    if r.get("exit_code") == 3:
        raise SandboxFileError("It is not in the sandbox any more.", 404)
    if r.get("exit_code") != 0:
        raise SandboxFileError((r.get("stderr") or "Could not delete it.").strip()[:300], 500)


async def rename(thread_id: str, path: str, new_name: str) -> str:
    """Rename in place (same folder); the new path."""
    src = clean(path)
    folder = src.rsplit("/", 1)[0] if "/" in src else ""
    dst = clean(_join(folder, valid_name(new_name)))
    if dst == src:
        return dst
    await _must_have_sandbox(thread_id)
    s, d = shlex.quote(src), shlex.quote(dst)
    r = await _run(thread_id, f"if [ ! -e {s} ] && [ ! -L {s} ]; then exit 3; fi; if [ -e {d} ]; then exit 4; fi; "
                              f"mv -T -- {s} {d}")
    if r.get("exit_code") == 3:
        raise SandboxFileError("It is not in the sandbox any more.", 404)
    if r.get("exit_code") == 4:
        raise SandboxFileError("Something with that name is already there.", 409)
    if r.get("exit_code") != 0:
        raise SandboxFileError((r.get("stderr") or "Could not rename it.").strip()[:300], 500)
    return dst


async def make_folder(thread_id: str, parent: str, name: str) -> str:
    """Make a folder (and the sandbox, if the conversation has none yet); its path."""
    dst = clean(_join(clean(parent, allow_root=True), valid_name(name)))
    d = shlex.quote(dst)
    r = await _run(thread_id, f"if [ -e {d} ]; then exit 4; fi; mkdir -- {d}")
    if r.get("exit_code") == 4:
        raise SandboxFileError("Something with that name is already there.", 409)
    if r.get("exit_code") != 0:
        raise SandboxFileError("There is no such folder in the sandbox.", 404)
    return dst


async def save_to_files(thread_id: str, path: str, user_id: str) -> dict:
    """Copy a sandbox file into the top of the user's Files, where it outlives the conversation. A name that is taken
    there gets " (2)", " (3)"... The saved {name, path}."""
    src = clean(path)
    await _must_have_sandbox(thread_id)
    q = shlex.quote(src)
    r = await _run(thread_id, f"if [ ! -f {q} ]; then exit 3; fi; stat -c %s -- {q}")
    if r.get("exit_code") == 3:
        raise SandboxFileError("Only a file can be saved, and this one is not in the sandbox any more.", 404)
    if int((r.get("stdout") or "0").strip() or 0) > MAX_SAVE_BYTES:
        raise SandboxFileError(f"The file is over the {MAX_SAVE_BYTES >> 20} MB limit.", 413)
    content, mime, filename = await sandbox_manager.read_file(thread_id, src)
    mime = mimetypes.guess_type(filename)[0] or mime
    stem, dot, ext = filename.rpartition(".")
    if not dot or not stem:
        stem, ext = filename, ""
    for n in range(1, 50):
        name = filename if n == 1 else f"{stem} ({n}){'.' + ext if ext else ''}"
        try:
            node = await file_client.upload_file(user_id, "/", name, content, mime)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                continue
            raise
        return {"name": node.get("name", name), "path": node.get("path") or f"/{name}"}
    raise SandboxFileError("Too many files with that name in your Files.", 409)
