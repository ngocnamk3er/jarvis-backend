"""A valid flow (schema.ParsedFlow) -> a compiled LangGraph StateGraph.

The state is the run's inputs and, per node, what that node produced. Each flow node is one graph node with the same
id; an edge is an edge, and a node with several incoming edges is joined on all of them (LangGraph runs it once every
one of them is done), so parallel branches meet the way they look in the editor.

Every node announces itself (a `flow_node` custom event, start and end) and its model's tokens stream as LangChain
events. Run inside a chat turn, all of it reaches the turn's astream_events with the calling tool's run id among the
parent ids; run on its own, runner.stream_flow reads the same events.
"""

import asyncio
import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import patch_config
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from app.flows.catalog import flow_tools, tool_specs
from app.flows.schema import DEFAULT_FIELD, REF, ParsedFlow, ParsedNode

FLOW_NODE_EVENT = "flow_node"
PREVIEW_CHARS = 4_000

# What a user's own model brings into a run's configurable (user_models.info). A node that names another model must not
# carry them over: they describe the run's model, not the node's.
_MODEL_KEYS = ("model_vision", "model_context_window", "no_fallback")


class FlowNodeError(Exception):
    """A node failed; the flow stops there."""

    def __init__(self, node_id: str, message: str):
        super().__init__(f"{node_id}: {message}")
        self.node_id = node_id
        self.message = message


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class FlowState(TypedDict):
    inputs: dict
    outputs: Annotated[dict, _merge]     # node id -> what it produced; parallel nodes write different keys


def render(text: str, outputs: dict, types: dict[str, str]) -> str:
    """Replace each {{node}} / {{node.field}} with what that node produced. Validation has made sure every reference
    is to a node before this one, and to a field it has."""
    def value(m: re.Match) -> str:
        ref = m.group(1)
        fld = m.group(2) or DEFAULT_FIELD.get(types[ref], "")
        v = (outputs.get(ref) or {}).get(fld, "")
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return REF.sub(value, text)


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in content if isinstance(p, (str, dict)))
    return str(content)


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def _parse_json(text: str) -> Any:
    """The JSON a model answered with, also when it wrapped it in prose or a ``` fence."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text, i)[0]
            except ValueError:
                continue
    raise ValueError("no JSON in the answer")


def _coerce_arg(value: str, schema: dict, name: str) -> Any:
    types = [schema["type"]] if "type" in schema else [s.get("type") for s in schema.get("anyOf", [])]
    if "null" in types and value.strip() == "":
        return None
    if "integer" in types:
        try:
            return int(value.strip())
        except ValueError:
            raise ValueError(f"argument '{name}' must be a whole number, got {value!r}") from None
    if "number" in types:
        try:
            return float(value.strip())
        except ValueError:
            raise ValueError(f"argument '{name}' must be a number, got {value!r}") from None
    if "boolean" in types:
        return value.strip().lower() in ("true", "1", "yes")
    return value


_LIMIT_REACHED = ("Not run: this step's limit of tool calls is reached. Answer now, with what you have, and say what "
                  "you could not find out.")


async def _call_tool(node: ParsedNode, tool: BaseTool | None, call: dict, config: RunnableConfig) -> ToolMessage:
    """One tool call the model made. A failure goes back to the model as an error result, for it to work around."""
    def reply(content: str, error: bool = False) -> ToolMessage:
        return ToolMessage(content, tool_call_id=call["id"], name=call["name"], status="error" if error else "success")

    if tool is None:
        return reply(f"There is no tool '{call['name']}' in this step.", error=True)
    args = dict(call.get("args") or {})
    if "label" in (tool.tool_call_schema.model_json_schema().get("properties") or {}) and not args.get("label"):
        args["label"] = f"Step {node.id} · {tool.name}"
    try:
        result = await tool.ainvoke(args, config=config)
    except Exception as e:  # noqa: BLE001
        return reply(f"{type(e).__name__}: {e}", error=True)
    return reply(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False))


async def _use_tools(node: ParsedNode, ask, llm, messages: list[BaseMessage], tool_list: list[BaseTool], budget: int,
                     config: RunnableConfig) -> tuple[AIMessage, list[dict]]:
    """An LLM step with tools: the model calls them as it decides, up to `budget` calls in all, then answers. At the
    limit it gets one last turn in which it may not call tools (tool_choice="none"), the way SoftHardToolCallLimit-
    Middleware ends a chat turn, so the step ends with an answer rather than with nothing."""
    by_name = {t.name: t for t in tool_list}
    free = llm.bind_tools(tool_list)
    msgs = list(messages)
    done: list[dict] = []
    while True:
        last_turn = len(done) >= budget
        reply = await ask(msgs, llm.bind_tools(tool_list, tool_choice="none") if last_turn else free)
        if not reply.tool_calls:
            return reply, done
        if last_turn:
            raise FlowNodeError(node.id, f"The model kept calling tools after its limit of {budget} calls.")
        msgs.append(reply)
        allowed = reply.tool_calls[: budget - len(done)]
        # `label` only names the call's badge in the chat; it is not part of what the tool was asked
        done += [{"name": tc["name"], "args": {k: v for k, v in (tc.get("args") or {}).items() if k != "label"}}
                 for tc in allowed]
        results = await asyncio.gather(*(_call_tool(node, by_name.get(tc["name"]), tc, config) for tc in allowed))
        msgs += results
        msgs += [ToolMessage(_LIMIT_REACHED, tool_call_id=tc["id"], name=tc["name"], status="error")
                 for tc in reply.tool_calls[len(allowed):]]


def _preview(value: dict) -> dict:
    return {k: (v[:PREVIEW_CHARS] + "…" if isinstance(v, str) and len(v) > PREVIEW_CHARS else v) for k, v in value.items()}


def compile_flow(flow: ParsedFlow, *, llm=None, tools: dict[str, BaseTool] | None = None):
    """The flow as a compiled graph. `llm` and `tools` are for tests; by default the app's model chain (which follows
    the run's `model`) and the catalog's tools."""
    if llm is None:
        from app.agents.llm import build_llm_with_fallback
        llm = build_llm_with_fallback()
    tools = tools if tools is not None else flow_tools()
    specs = tool_specs(tools)
    types = {i: n.type for i, n in flow.nodes.items()}

    def node_fn(node: ParsedNode):
        async def produce(state: FlowState, config: RunnableConfig) -> dict:
            outputs = state["outputs"]
            c = node.config
            if node.type == "input":
                return dict(state["inputs"])
            if node.type == "output":
                return {"text": render(c.template, outputs, types)}
            if node.type == "tool":
                tool = tools[c.tool]
                spec = specs[c.tool]
                args = {k: _coerce_arg(render(v, outputs, types), spec.args[k], k) for k, v in c.args.items()}
                if "label" in (tool.tool_call_schema.model_json_schema().get("properties") or {}):
                    args["label"] = node.label or f"Step {node.id} · {c.tool}"
                result = await tool.ainvoke(args, config=config)
                return {"result": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)}
            # llm
            system = render(c.system, outputs, types)
            if c.output == "json":
                system = (system + "\n\n" if system else "") + "Answer with one JSON value only, and nothing else."
            messages = ([SystemMessage(system)] if system else []) + [HumanMessage(render(c.prompt, outputs, types))]
            call_config = config
            if c.model:
                # Set to None, not left out: configurables are merged (patch_config, then the RunnableLambda's
                # ensure_config), so a key that is only missing comes back from the node's config.
                call_config = patch_config(config, configurable={**dict.fromkeys(_MODEL_KEYS), "model": c.model})

            async def ask(msgs: list[BaseMessage], model) -> AIMessage:
                async def call(m):
                    # Inside a RunnableLambda, so that the model reads `call_config`'s configurable (ThinkingChatOpenAI
                    # takes the model to use from there) rather than the node's.
                    return await model.ainvoke(m)
                return await RunnableLambda(call, name=f"{node.id}.model").ainvoke(msgs, config=call_config)

            if c.tools:
                reply, calls = await _use_tools(node, ask, llm, messages, [tools[t] for t in c.tools], c.max_tool_calls,
                                                config)
            else:
                reply, calls = await ask(messages, llm), None
            text = _THINK.sub("", content_text(reply.content)).strip()
            out: dict[str, Any] = {"text": text}
            if calls is not None:
                out["tool_calls"] = calls          # shown with the step; not a field other steps can reference
            if c.output == "json":
                try:
                    out["json"] = _parse_json(text)
                except ValueError:
                    raise FlowNodeError(node.id, f"The model was asked for JSON and answered: {text[:300]!r}") from None
            return out

        async def run(state: FlowState, config: RunnableConfig) -> dict:
            await adispatch_custom_event(FLOW_NODE_EVENT, {"node": node.id, "type": node.type, "phase": "start"},
                                         config=config)
            try:
                value = await produce(state, config)
            except FlowNodeError:
                raise
            except Exception as e:  # noqa: BLE001 — a node's failure is the flow's result, named after the node
                raise FlowNodeError(node.id, f"{type(e).__name__}: {e}") from e
            await adispatch_custom_event(FLOW_NODE_EVENT, {"node": node.id, "type": node.type, "phase": "end",
                                                           "output": _preview(value)}, config=config)
            return {"outputs": {node.id: value}}

        return run

    g = StateGraph(FlowState)
    for i in flow.order:
        g.add_node(i, node_fn(flow.nodes[i]))
    g.add_edge(START, flow.input_id)
    for i in flow.order:
        preds = flow.preds[i]
        if len(preds) == 1:
            g.add_edge(preds[0], i)
        elif preds:
            g.add_edge(preds, i)        # waits for all of them
    for o in flow.output_ids:
        g.add_edge(o, END)
    # checkpointer=False, not left unset: a graph run inside another graph's node (a chat turn's tool) otherwise
    # inherits that graph's checkpointer as a subgraph, and every flow step was checkpointed into the conversation's
    # thread (6 checkpoints for a 4-node flow, seen in a test). A flow keeps nothing between runs and has nothing to
    # resume in version 1.
    return g.compile(name="flow", checkpointer=False)
