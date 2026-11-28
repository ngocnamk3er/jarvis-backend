"""Running a flow: to its result (run_flow), or as the events of its nodes while it runs (stream_flow)."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.tools import BaseTool

from app.core.config import settings
from app.flows.compiler import FLOW_NODE_EVENT, FlowNodeError, compile_flow, content_text
from app.flows.schema import InputField, ParsedFlow


class FlowInputError(ValueError):
    """The inputs do not fit the flow's Input node."""


@dataclass
class FlowResult:
    text: str
    outputs: dict


def coerce_inputs(fields: list[InputField], inputs: dict) -> dict:
    """The inputs, typed as the Input node declares them. Unknown names are an error, so a typo is not silently lost."""
    unknown = set(inputs) - {f.name for f in fields}
    if unknown:
        raise FlowInputError(f"The flow has no input {', '.join(sorted(unknown))}.")
    values = {}
    for f in fields:
        raw = inputs.get(f.name)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            if f.required:
                raise FlowInputError(f"The input '{f.name}' is required.")
            values[f.name] = "" if f.type == "string" else None
            continue
        try:
            if f.type == "number":
                values[f.name] = raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else float(raw)
            elif f.type == "boolean":
                values[f.name] = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("true", "1", "yes")
            else:
                values[f.name] = raw if isinstance(raw, str) else str(raw)
        except (TypeError, ValueError):
            raise FlowInputError(f"The input '{f.name}' must be a {f.type}.") from None
    return values


def _result(flow: ParsedFlow, state: dict) -> FlowResult:
    outputs = state.get("outputs") or {}
    texts = [outputs[o]["text"] for o in flow.order if o in flow.output_ids and o in outputs]
    return FlowResult(text="\n\n".join(texts), outputs=outputs)


def _initial(flow: ParsedFlow, inputs: dict) -> dict:
    return {"inputs": coerce_inputs(flow.input_fields, inputs), "outputs": {}}


def _with_room(flow: ParsedFlow, config: dict) -> dict:
    """The config with a recursion limit the flow fits in. LangGraph counts steps, and a chat turn's limit (25 by
    default) is too low for a long chain of nodes: 30 nodes in a row are 30 steps."""
    config = ensure_config(config)
    return patch_config(config, recursion_limit=max(config.get("recursion_limit", 25), len(flow.order) + 10))


async def run_flow(flow: ParsedFlow, inputs: dict, config: dict, *, llm=None,
                   tools: dict[str, BaseTool] | None = None) -> FlowResult:
    """Run to the end. Raises FlowInputError, FlowNodeError (naming the node), TimeoutError."""
    initial = _initial(flow, inputs)
    graph = compile_flow(flow, llm=llm, tools=tools)
    async with asyncio.timeout(settings.FLOW_RUN_TIMEOUT_SECONDS):
        state = await graph.ainvoke(initial, config=_with_room(flow, config))
    return _result(flow, state)


async def stream_flow(flow: ParsedFlow, inputs: dict, config: dict, *, llm=None,
                      tools: dict[str, BaseTool] | None = None) -> AsyncIterator[dict]:
    """Run, yielding what happens:
        {"type": "node", "node", "node_type", "phase": "start" | "end", "output"?}  a node starting / done (its output)
        {"type": "token", "node", "content"}                                     a model token, and whose
        {"type": "result", "text"}  or  {"type": "error", "node", "message"}     last
    Inputs are checked before anything runs: a FlowInputError is raised, not yielded."""
    initial = _initial(flow, inputs)
    graph = compile_flow(flow, llm=llm, tools=tools)
    final = None
    try:
        async with asyncio.timeout(settings.FLOW_RUN_TIMEOUT_SECONDS):
            async for ev in graph.astream_events(initial, config=_with_room(flow, config), version="v2"):
                kind = ev["event"]
                if kind == "on_custom_event" and ev["name"] == FLOW_NODE_EVENT:
                    data = dict(ev["data"])
                    yield {"type": "node", "node": data.pop("node"), "node_type": data.pop("type"), **data}
                elif kind == "on_chat_model_stream":
                    content = content_text(ev["data"]["chunk"].content)
                    if content:
                        yield {"type": "token", "node": ev.get("metadata", {}).get("langgraph_node"), "content": content}
                elif kind == "on_chain_end" and not ev.get("parent_ids"):
                    final = ev["data"].get("output")
    except FlowNodeError as e:
        yield {"type": "error", "node": e.node_id, "message": e.message}
        return
    except TimeoutError:
        yield {"type": "error", "node": None,
               "message": f"The flow ran for more than {settings.FLOW_RUN_TIMEOUT_SECONDS} s and was stopped."}
        return
    yield {"type": "result", "text": _result(flow, final or {}).text}
