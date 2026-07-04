import json
from langchain_core.tools import tool


@tool
def generate_visualization_mermaid(code: str, label: str, title: str = "") -> str:
    """Render a Mermaid diagram when the user needs a flow-based visual.

    Use this for anything flow-related: flowcharts, sequence diagrams, state
    machines, class diagrams, mindmaps, timelines, entity relationships, etc.
    Write valid Mermaid syntax in `code`.

    Args:
        code: Valid Mermaid diagram syntax (e.g. "flowchart TD\\n  A --> B")
        title: Short title shown above the diagram (optional)
    """
    return json.dumps({"__viz__": "mermaid", "code": code, "title": title})
