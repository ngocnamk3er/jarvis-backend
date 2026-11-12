def tool_output_text(content) -> str:
    """A tool result as plain text.

    Ordinary tools return a string, but an MCP tool returns a list of content
    blocks (`[{"type": "text", "text": "42"}, {"type": "image", ...}]`). The
    chat UI and the saved history both treat a result as a string, so blocks
    are flattened: text kept, anything else shown as a `[type]` placeholder.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif isinstance(b, dict):
                parts.append(f"[{b.get('type', 'content')}]")
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return str(content)
