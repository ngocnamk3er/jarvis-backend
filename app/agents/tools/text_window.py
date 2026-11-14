"""A numbered window onto a long text, as DeepSeek Harness's `read` returns it.

Lines come back as `<n>: <text>` and the last line says where to continue, so a
long document is read in steps (offset=, limit=) rather than all at once. The
caps are theirs: 2,000 lines a call, 2,000 characters a line, 51,200 bytes a call.
"""

READ_LIMIT = 2_000
MAX_LINE_CHARS = 2_000
MAX_BYTES = 51_200


class WindowError(ValueError):
    """Bad arguments; the message is written for the model."""


def read_window(text: str, offset: int | None = None, limit: int | None = None) -> str:
    offset = 1 if offset is None else offset
    limit = READ_LIMIT if limit is None else limit
    if offset < 1:
        raise WindowError("invalid arguments: offset must be at least 1 (it counts lines from 1)")
    if limit < 1:
        raise WindowError("invalid arguments: limit must be at least 1")
    if limit > READ_LIMIT:
        raise WindowError(f"invalid arguments: limit must be less than or equal to {READ_LIMIT}")

    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return "(End of file - total 0 lines)"
    if offset > total:
        raise WindowError(f"offset {offset} is past the end of the file, which has {total} lines")

    out: list[str] = []
    used = 0
    capped = False
    last = offset - 1
    for n in range(offset, min(offset + limit - 1, total) + 1):
        line = lines[n - 1]
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + "..."
        rendered = f"{n}: {line}"
        if used + len(rendered.encode("utf-8")) + 1 > MAX_BYTES and out:
            capped = True
            break
        out.append(rendered)
        used += len(rendered.encode("utf-8")) + 1
        last = n

    if capped:
        footer = f"(Output capped. Showing lines {offset}-{last}. Use offset={last + 1} to continue.)"
    elif last < total:
        footer = f"(Showing lines {offset}-{last} of {total}. Use offset={last + 1} to continue.)"
    else:
        footer = f"(End of file - total {total} lines)"
    return "\n".join(out) + "\n\n" + footer
