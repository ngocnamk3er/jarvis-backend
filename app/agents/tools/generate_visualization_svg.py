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

    Args:
        code: Complete SVG markup starting with <svg ...> (include viewBox)
        title: Short title shown above the graphic (optional)
    """
    error = await validate_svg(code, sandbox=_SVG_SANDBOX)
    if error:
        return f"Error: invalid SVG — {error}"
    return json.dumps({"__viz__": "svg", "code": code, "title": title})
