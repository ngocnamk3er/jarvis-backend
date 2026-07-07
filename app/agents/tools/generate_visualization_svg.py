import json
from langchain_core.tools import tool

from app.agents.tools.viz_validate import validate_svg


@tool
async def generate_visualization_svg(code: str, label: str, title: str = "") -> str:
    """Render a custom SVG graphic for anything that is not flow-based.

    Use this for bar charts, pie charts, line graphs, illustrations, icons,
    custom layouts, or any visual that requires precise positioning and styling
    beyond what Mermaid supports. Write complete, self-contained SVG markup.

    Args:
        code: Complete SVG markup starting with <svg ...> (include viewBox)
        title: Short title shown above the graphic (optional)
    """
    error = await validate_svg(code)
    if error:
        return f"Error: invalid SVG — {error}"
    return json.dumps({"__viz__": "svg", "code": code, "title": title})
