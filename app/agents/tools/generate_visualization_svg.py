import json
from langchain_core.tools import tool


@tool
def generate_visualization_svg(code: str, title: str = "") -> str:
    """Render a custom SVG graphic for anything that is not flow-based.

    Use this for bar charts, pie charts, line graphs, illustrations, icons,
    custom layouts, or any visual that requires precise positioning and styling
    beyond what Mermaid supports. Write complete, self-contained SVG markup.

    Args:
        code: Complete SVG markup starting with <svg ...> (include viewBox)
        title: Short title shown above the graphic (optional)
    """
    return json.dumps({"__viz__": "svg", "code": code, "title": title})
