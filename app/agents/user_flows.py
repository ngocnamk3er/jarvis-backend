"""The user's own flows (app/flows/), as tools of their conversations.

Each switched-on flow is a tool named `flow_<slug>`, its arguments the flow's Input fields, its description the flow's.
Like the user's MCP tools (user_mcp.py) they differ per user, so UserFlowsMiddleware adds them to what the model is
offered on every model call and runs them when called; the agent's tool node has never heard of them.

A flow called from chat runs inside the tool call: its node events (`flow_node`) and its models' tokens reach the
turn's stream under the tool's run id, which chat_service nests under the flow's badge, the way it does a subagent's.
Its tools get the turn's config (the user, the switched-off tools), with `sandbox: False` so a long web result comes
back inline into the flow rather than as a sandbox file it could not read.
"""

import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import patch_config
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from app.core.config import settings
from app.db import flow_repository
from app.flows.catalog import tool_specs
from app.flows.compiler import FlowNodeError
from app.flows.runner import FlowInputError, run_flow
from app.flows.schema import ParsedFlow, validate_flow

logger = logging.getLogger(__name__)

FLOW_TOOL_PREFIX = "flow_"
_JSON_TYPES = {"string": "string", "number": "number", "boolean": "boolean"}

# (flow id, version) -> the validated flow, or None when the saved one no longer validates (a tool it calls was
# removed). A save bumps the version, so an entry is never stale; the oldest are dropped past the limit.
_parsed: OrderedDict[tuple[str, int], ParsedFlow | None] = OrderedDict()
_PARSED_MAX = 512


def is_flow_tool(name: str | None) -> bool:
    return bool(name) and name.startswith(FLOW_TOOL_PREFIX)


def _parse(row: dict) -> ParsedFlow | None:
    key = (row["id"], row["version"])
    if key in _parsed:
        _parsed.move_to_end(key)
        return _parsed[key]
    parsed, problems = validate_flow(row["definition"], tool_specs())
    if problems:
        logger.warning("flow %s (v%s) no longer validates, not offered: %s", row["id"], row["version"],
                       [p.as_dict() for p in problems[:3]])
    _parsed[key] = parsed
    while len(_parsed) > _PARSED_MAX:
        _parsed.popitem(last=False)
    return parsed


def _args_schema(parsed: ParsedFlow) -> dict:
    fields = parsed.input_fields
    return {
        "type": "object",
        "properties": {f.name: {"type": _JSON_TYPES[f.type], **({"description": f.description} if f.description else {})}
                       for f in fields},
        "required": [f.name for f in fields if f.required],
    }


def make_tool(row: dict, parsed: ParsedFlow) -> BaseTool:
    """The flow as a tool. Its failures come back to the model as an error result naming the failed step."""
    uses = {n.config.tool for n in parsed.nodes.values() if n.type == "tool"}
    name = row["name"]

    async def run(config: RunnableConfig, **inputs: Any) -> str:
        switched_off = sorted(uses & set((config.get("configurable") or {}).get("disabled_tools") or []))
        if switched_off:
            raise ToolException(f"This flow uses {', '.join(switched_off)}, which is turned off in this conversation.")
        try:
            result = await run_flow(parsed, inputs, patch_config(config, configurable={"sandbox": False}))
        except FlowInputError as e:
            raise ToolException(str(e)) from None
        except FlowNodeError as e:
            raise ToolException(f"The flow stopped at step '{e.node_id}': {e.message}") from None
        except TimeoutError:
            raise ToolException(f"The flow ran for more than {settings.FLOW_RUN_TIMEOUT_SECONDS} s and was stopped.") \
                from None
        return result.text or "(the flow finished without output)"

    description = (row.get("description") or "").strip() or name
    return StructuredTool(
        name=FLOW_TOOL_PREFIX + row["slug"],
        description=f"{description}\n\n(A flow the user built: \"{name}\". It runs its steps and returns the result.)",
        args_schema=_args_schema(parsed),
        coroutine=run,
        handle_tool_error=True,
        metadata={"flow_id": row["id"], "flow_name": name},
    )


async def tools_for_user(user_id: str | None) -> dict[str, BaseTool]:
    if not (settings.FLOWS_ENABLED and user_id):
        return {}
    try:
        rows = await flow_repository.list_enabled(user_id)
    except Exception:
        logger.warning("could not load the user's flows", exc_info=True)
        return {}
    out: dict[str, BaseTool] = {}
    for row in rows:
        parsed = _parse(row)
        if parsed is not None:
            tool = make_tool(row, parsed)
            out[tool.name] = tool
    return out


def _user_id() -> str | None:
    try:
        from langgraph.config import get_config

        return get_config().get("configurable", {}).get("user_id")
    except RuntimeError:
        return None


class UserFlowsMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        flows = await tools_for_user(_user_id())
        if flows:
            request = request.override(tools=[*request.tools, *flows.values()])
        return await handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        name = request.tool_call["name"]
        if request.tool is None and is_flow_tool(name):
            tool = (await tools_for_user(_user_id())).get(name)
            if tool is not None:
                request = request.override(tool=tool)
        return await handler(request)
