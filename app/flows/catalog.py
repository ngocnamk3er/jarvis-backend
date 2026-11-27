"""The tools a Tool node may call, and what the editor and validation need to know about them.

Version 1: reading the web and the user's own workspace files. Each of these takes the run's config (the user's id, and
`sandbox: False`, which makes web_search/web_fetch return a long result cut inline instead of saving it to a sandbox
file the flow could not read back). Tools that act — bash, writing files — wait for the Code node and for approvals.
"""

from functools import cache

from langchain_core.tools import BaseTool

from app.flows.schema import ToolSpec

# Arguments the flow fills in itself; the user does not set them.
AUTO_ARGS = {"label"}


@cache
def flow_tools() -> dict[str, BaseTool]:
    from app.agents.tools.files import list_files, read_file, search_files
    from app.agents.tools.web_fetch import web_fetch
    from app.agents.tools.web_search import web_search

    return {t.name: t for t in (web_search, web_fetch, search_files, read_file, list_files)}


def _schema(tool: BaseTool) -> dict:
    return tool.tool_call_schema.model_json_schema()


def tool_specs(tools: dict[str, BaseTool] | None = None) -> dict[str, ToolSpec]:
    specs = {}
    for name, tool in (tools or flow_tools()).items():
        schema = _schema(tool)
        props = {k: v for k, v in (schema.get("properties") or {}).items() if k not in AUTO_ARGS}
        specs[name] = ToolSpec(name=name, args=props, required=set(schema.get("required") or ()) - AUTO_ARGS)
    return specs


def describe(tools: dict[str, BaseTool] | None = None) -> list[dict]:
    """The tools as the editor shows them: name, what it does, its arguments."""
    out = []
    for name, spec in tool_specs(tools).items():
        tool = (tools or flow_tools())[name]
        out.append({
            "name": name,
            "description": (tool.description or "").strip().split("\n\n")[0],
            "args": [{"name": arg, "type": _type(s), "required": arg in spec.required,
                      "description": s.get("description", "")} for arg, s in spec.args.items()],
        })
    return out


def _type(schema: dict) -> str:
    if "type" in schema:
        return schema["type"]
    types = [s.get("type") for s in schema.get("anyOf", []) if s.get("type") != "null"]
    return types[0] if types else "string"
