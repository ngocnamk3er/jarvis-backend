"""Save a tool's raw output into the conversation's sandbox workspace instead
of returning it straight into the model's context.

`web_search` and `web_fetch` are the primary source of tool-output spill —
both are told to fan out in parallel and both can each return thousands of
characters of scraped/searched text in one shot, all of it landing in
context on the same turn (the `ToolOutputOffloadMiddleware` stub-and-recall
mechanism in middleware.py only helps *after* a result is a few turns old —
it doesn't stop the initial spike). Writing the raw result to a file and
handing back a short stub means the model pulls out only what it needs,
using `bash`, instead of the whole thing landing in context up front.

That only holds if the follow-up `bash` read is itself narrow — `bash.py`
caps its own output the same way (see `_cap_output` there) so a blind `cat`
of the saved file can't undo this by dumping it straight back into context.
"""

import base64
import shlex

import httpx

from app.agents.tools.sandbox_manager import exec_bash

_PREVIEW_CHARS = 600


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

    try:
        # Content travels base64-piped through `bash -c` — the base64
        # alphabet has no shell metacharacters, so this is safe regardless of
        # what's inside (quotes, backticks, `$`, binary-ish bytes from a
        # mis-decoded page) without needing to escape the content itself.
        b64 = base64.b64encode(content.encode("utf-8", errors="replace")).decode()
        command = f"printf '%s' '{b64}' | base64 -d > {shlex.quote(filename)}"
        result = await exec_bash(thread_id, command)
    except httpx.HTTPError:
        return content  # sandbox unreachable — fall back rather than lose the result

    if result.get("timed_out") or result.get("exit_code") not in (0, None):
        return content  # write failed for some other reason — same fallback

    preview = content[:_PREVIEW_CHARS].rstrip()
    omitted = n - _PREVIEW_CHARS
    more = f"\n... [{omitted:,} more chars in the file] ..." if omitted > 0 else ""
    return (
        f"[{kind} — {n:,} chars, saved to /workspace/{filename}]\n"
        f"--- preview ---\n{preview}{more}\n--- end preview ---\n"
        "Use bash to pull out what you need (grep / head / a short Python "
        "read) instead of fetching or searching this again."
    )
