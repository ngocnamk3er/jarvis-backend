"""Save a tool's raw output into the conversation's sandbox workspace instead
of returning it straight into the model's context.

`web_search`, `web_fetch`, and `read_file` are the tools that can each
return thousands of characters in one shot — all of it landing in context
on the same turn if returned directly. Writing the raw result to a file and
handing back a short stub means the model pulls out only what it needs,
using `bash`, instead of the whole thing landing in context up front.

That only holds if the follow-up `bash` read is itself narrow — `bash.py`
caps its own output the same way (see `_cap_output` there) so a blind `cat`
of the saved file can't undo this by dumping it straight back into context.
This is the only tool-output size control in the agent; there's no separate
aging-based offload behind it, so what a tool returns here is what the
model's context actually holds.
"""

import base64
import shlex

import httpx

from app.agents.tools.sandbox_manager import exec_bash

_PREVIEW_CHARS = 600

# Linux caps any single execve() argv/envp string at MAX_ARG_STRLEN — 32 pages,
# 128 KiB (131,072 bytes) on a standard 4 KiB-page kernel — independent of the
# much larger total ARG_MAX. The sandbox agent runs the write via
# `asyncio.create_subprocess_exec("bash", "-c", command, ...)`, so `command`
# itself is one argv string: a base64 blob past that length makes execve()
# fail with `OSError: [Errno 7] Argument list too long` (E2BIG) before the
# process even starts — verified live against the sandbox agent's own
# traceback. Chunk size is base64 chars (a multiple of 4, so each chunk
# decodes independently to the right bytes) chosen with wide headroom under
# the 131,072-byte limit once wrapped in `printf '%s' '...' | base64 -d >> f`.
_CHUNK_B64_CHARS = 60_000


async def save_and_stub(
    thread_id: str, filename: str, content: str, *, kind: str, min_chars: int = 2000
) -> str:
    """Return `content` unchanged if it's short enough to not matter. Above
    `min_chars`, write it to `/workspace/<filename>` in the sandbox and
    return a short stub instead. Degrades to returning `content` directly if
    the sandbox write fails for any reason — never silently drops data.
    """
    n = len(content)
    if n <= min_chars:
        return content

    # Content travels base64-piped through `bash -c` — the base64 alphabet
    # has no shell metacharacters, so this is safe regardless of what's
    # inside (quotes, backticks, `$`, binary-ish bytes from a mis-decoded
    # page) without needing to escape the content itself. Written in
    # _CHUNK_B64_CHARS-sized pieces (first truncates, rest append) rather
    # than one shot — see _CHUNK_B64_CHARS above for why.
    b64 = base64.b64encode(content.encode("utf-8", errors="replace")).decode()
    quoted_name = shlex.quote(filename)
    try:
        for i in range(0, len(b64), _CHUNK_B64_CHARS):
            chunk = b64[i : i + _CHUNK_B64_CHARS]
            redirect = ">" if i == 0 else ">>"
            command = f"printf '%s' '{chunk}' | base64 -d {redirect} {quoted_name}"
            result = await exec_bash(thread_id, command)
            if result.get("timed_out") or result.get("exit_code") not in (0, None):
                return content  # write failed partway — fall back, don't leave a partial file
    except httpx.HTTPError:
        return content  # sandbox unreachable — fall back rather than lose the result

    preview = content[:_PREVIEW_CHARS].rstrip()
    omitted = n - _PREVIEW_CHARS
    more = f"\n... [{omitted:,} more chars in the file] ..." if omitted > 0 else ""
    return (
        f"[{kind} — {n:,} chars, saved to /workspace/{filename}]\n"
        f"--- preview ---\n{preview}{more}\n--- end preview ---\n"
        "Use bash to pull out what you need (grep / head / a short Python "
        "read) instead of fetching or searching this again."
    )
