import json
from langchain_core.tools import tool


@tool
def generate_visualization_mermaid(code: str, title: str = "") -> str:
    """Render a Mermaid diagram to visually explain something to the user.

    Use this whenever a visual would help — flowcharts, sequence diagrams,
    class diagrams, state machines, pie charts, mindmaps, timelines, etc.
    Write valid Mermaid syntax in `code`. Keep diagrams focused and readable.

    Args:
        code: Valid Mermaid diagram syntax (e.g. "flowchart TD\\n  A --> B")
        title: Short title shown above the diagram (optional)
    """
    return json.dumps({"__viz__": "mermaid", "code": code, "title": title})
