import json
from langchain_core.tools import tool

from app.agents.tools.viz_validate import validate_svg

# Matches the sandbox flags svg-diagram.tsx sets on the real iframe.
_SVG_SANDBOX = "allow-scripts"


@tool
async def generate_visualization_svg(code: str, label: str, title: str = "") -> str:
    """Render a custom SVG graphic — the tool for any static or interactive visual.

    Use this for bar charts, pie charts, line graphs, illustrations, icons,
    custom layouts, flowcharts, sequence diagrams, mind maps, or any visual
    that requires precise positioning and styling. Write complete,
    self-contained SVG markup.

    The SVG renders inside a sandboxed iframe, so it can be interactive: CSS
    `:hover`/`:active` states, `<a>` links, SMIL `<animate>`, and inline
    `<script>` with event handlers (onclick, onmouseover, etc.) all work.
    Keep everything self-contained — no external URLs.

    On success this already renders as a visual block in the chat, so you
    don't need to repeat the code to show it to the user. If you do include
    the SVG source in your reply (e.g. because the user asked to see or copy
    it), always wrap it in a fenced code block with the `svg` language tag
    (```svg ... ```) — raw unfenced markup renders as broken/ugly text since
    the chat view doesn't render raw inline HTML.

    Args:
        code: Complete SVG markup starting with <svg ...> (include viewBox)
        label: Brief human-readable description shown to the user (e.g. "Drawing revenue chart").
        title: Short title shown above the graphic (optional)
    """
    error = await validate_svg(code, sandbox=_SVG_SANDBOX)
    if error:
        return f"Error: invalid SVG — {error}"
    return json.dumps({"__viz__": "svg", "code": code, "title": title})
