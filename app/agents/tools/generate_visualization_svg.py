import json
from langchain_core.tools import tool


@tool
def generate_visualization_svg(code: str, title: str = "") -> str:
    """Render a custom SVG illustration to visually explain something to the user.

    Use this when you need a precise custom graphic — charts, diagrams, icons,
    data visualizations, or anything that requires exact positioning and styling
    that Mermaid cannot express. Write complete, self-contained SVG markup
    (include width, height, and viewBox attributes).

    Args:
        code: Complete SVG markup starting with <svg ...>
        title: Short title shown above the graphic (optional)
    """
    return json.dumps({"__viz__": "svg", "code": code, "title": title})
