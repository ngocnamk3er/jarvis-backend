"""The flow format, and what makes a flow valid.

A flow is JSON: nodes and the edges between them. The editor saves it as is (node positions included, which nothing
here reads). A node reads what nodes before it produced through `{{node_id.field}}` in its text: a plain substitution,
never an expression or a template language, so a flow cannot run code through it.

    {"schema_version": 1,
     "nodes": [{"id": "in",  "type": "input",  "config": {"fields": [{"name": "topic"}]}},
               {"id": "s",   "type": "tool",   "config": {"tool": "web_search", "args": {"query": "{{in.topic}}"}}},
               {"id": "sum", "type": "llm",    "config": {"prompt": "Summarise:\\n{{s.result}}"}},
               {"id": "out", "type": "output", "config": {"template": "{{sum.text}}"}}],
     "edges": [{"source": "in", "target": "s"}, {"source": "s", "target": "sum"}, {"source": "sum", "target": "out"}]}

validate_flow() checks everything that can be checked before running, and says which node each problem is in, so the
editor can point at it. Version 1 has no branches and no loops: the graph is acyclic, and a node with several incoming
edges runs once all of them are done.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

SCHEMA_VERSION = 1
MAX_NODES = 50
MAX_EDGES = 100
MAX_TEXT = 20_000          # a prompt, a template, a tool argument
MAX_FIELDS = 20

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
RESERVED_IDS = {"inputs", "outputs"}   # the flow state's keys (compiler.FlowState); a node cannot share a name with one
# {{node}} or {{node.field}}, spaces allowed inside the braces
REF = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\s*\}\}")

NodeType = Literal["input", "llm", "tool", "output"]


class InputField(BaseModel):
    name: str
    type: Literal["string", "number", "boolean"] = "string"
    description: str = Field(default="", max_length=500)
    required: bool = True


class InputConfig(BaseModel):
    fields: list[InputField] = Field(default_factory=list)


class LLMConfig(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_TEXT)
    system: str = Field(default="", max_length=MAX_TEXT)
    model: str | None = None          # None: the model of the run (the conversation's)
    output: Literal["text", "json"] = "text"


class ToolConfig(BaseModel):
    tool: str
    args: dict[str, str] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    template: str = Field(min_length=1, max_length=MAX_TEXT)


CONFIGS: dict[str, type[BaseModel]] = {
    "input": InputConfig, "llm": LLMConfig, "tool": ToolConfig, "output": OutputConfig,
}


class Node(BaseModel):
    id: str
    type: NodeType
    label: str = Field(default="", max_length=80)
    config: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] | None = None   # the editor's, not read here


class Edge(BaseModel):
    source: str
    target: str


class FlowDefinition(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)


@dataclass
class Problem:
    message: str
    node: str | None = None

    def as_dict(self) -> dict:
        return {"node": self.node, "message": self.message}


@dataclass
class ParsedNode:
    id: str
    type: str
    label: str
    config: Any  # the type's *Config model


@dataclass
class ParsedFlow:
    """A flow that passed validate_flow, in the shape the compiler wants."""
    nodes: dict[str, ParsedNode]
    order: list[str]                                   # topological
    preds: dict[str, list[str]] = field(default_factory=dict)
    input_id: str = ""
    output_ids: list[str] = field(default_factory=list)

    @property
    def input_fields(self) -> list[InputField]:
        return self.nodes[self.input_id].config.fields


@dataclass
class ToolSpec:
    """What validation needs to know about a tool a Tool node may call (see catalog.py)."""
    name: str
    args: dict[str, dict]          # arg name -> JSON schema of it
    required: set[str]


def fields_of(node: ParsedNode) -> set[str]:
    """What `{{node.<field>}}` may name."""
    if node.type == "input":
        return {f.name for f in node.config.fields}
    if node.type == "llm":
        return {"text", "json"} if node.config.output == "json" else {"text"}
    if node.type == "tool":
        return {"result"}
    return {"text"}


DEFAULT_FIELD = {"llm": "text", "tool": "result", "output": "text"}  # what `{{node}}` alone means; an input has none


def texts_of(node: ParsedNode) -> list[tuple[str, str]]:
    """(where, text) of every text of a node that may hold {{refs}}."""
    c = node.config
    if node.type == "llm":
        return [("prompt", c.prompt), ("system", c.system)]
    if node.type == "tool":
        return [(f"argument '{k}'", v) for k, v in c.args.items()]
    if node.type == "output":
        return [("template", c.template)]
    return []


def validate_flow(
    definition: dict | FlowDefinition, tools: dict[str, ToolSpec], models: set[str] | None = None
) -> tuple[ParsedFlow | None, list[Problem]]:
    """(the flow ready to compile, []) or (None, what is wrong with it). `tools` are those a Tool node may call;
    `models`, if given, the models an LLM node may name."""
    try:
        flow = definition if isinstance(definition, FlowDefinition) else FlowDefinition.model_validate(definition)
    except ValidationError as e:
        return None, [Problem(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}") for err in e.errors()[:20]]

    problems: list[Problem] = []
    if len(flow.nodes) > MAX_NODES:
        problems.append(Problem(f"A flow can have at most {MAX_NODES} nodes."))
    if len(flow.edges) > MAX_EDGES:
        problems.append(Problem(f"A flow can have at most {MAX_EDGES} connections."))
    if problems:
        return None, problems

    nodes: dict[str, ParsedNode] = {}
    for n in flow.nodes:
        if not IDENT.match(n.id):
            problems.append(Problem("A node id is a letter or _, then letters, digits or _ (32 at most).", n.id))
            continue
        if n.id.startswith("__") or n.id in RESERVED_IDS:
            # LangGraph's own names (__start__, __end__...) and the keys of the flow's state
            problems.append(Problem(f"'{n.id}' is a reserved name; pick another id.", n.id))
            continue
        if n.id in nodes:
            problems.append(Problem("Two nodes have this id.", n.id))
            continue
        try:
            config = CONFIGS[n.type].model_validate(n.config)
        except ValidationError as e:
            for err in e.errors()[:5]:
                problems.append(Problem(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}", n.id))
            continue
        nodes[n.id] = ParsedNode(n.id, n.type, n.label, config)

    inputs = [i for i, n in nodes.items() if n.type == "input"]
    outputs = [i for i, n in nodes.items() if n.type == "output"]
    if len(inputs) != 1:
        problems.append(Problem("A flow has exactly one Input node."))
    if not outputs:
        problems.append(Problem("A flow needs at least one Output node."))

    preds: dict[str, list[str]] = {i: [] for i in nodes}
    succs: dict[str, list[str]] = {i: [] for i in nodes}
    seen_edges: set[tuple[str, str]] = set()
    for e in flow.edges:
        bad = [end for end in (e.source, e.target) if end not in nodes]
        if bad:
            problems.append(Problem(f"A connection goes to a node that does not exist: {', '.join(bad)}."))
            continue
        if e.source == e.target:
            problems.append(Problem("A node is connected to itself.", e.source))
            continue
        if (e.source, e.target) in seen_edges:
            continue
        seen_edges.add((e.source, e.target))
        preds[e.target].append(e.source)
        succs[e.source].append(e.target)

    for i, n in nodes.items():
        if n.type == "input" and preds[i]:
            problems.append(Problem("Nothing can lead into the Input node.", i))
        if n.type == "output" and succs[i]:
            problems.append(Problem("Nothing can follow an Output node.", i))

    order = _topological(nodes, preds, succs)
    if order is None:
        problems.append(Problem("The connections form a loop; version 1 flows cannot loop."))
        return None, problems

    ancestors: dict[str, set[str]] = {}
    for i in order:
        ancestors[i] = set().union(*([{p} | ancestors[p] for p in preds[i]] or [set()]))

    if len(inputs) == 1:
        start = inputs[0]
        for i in nodes:
            if i != start and start not in ancestors[i]:
                problems.append(Problem("This node is not reached from the Input node.", i))
        leads_to_output = {o for o in outputs} | {a for o in outputs for a in ancestors[o]}
        for i, n in nodes.items():
            if n.type != "output" and i not in leads_to_output:
                problems.append(Problem("This node does not lead to an Output node, so its work would be lost.", i))

    for i, n in nodes.items():
        problems += _check_node(n, nodes, ancestors[i], tools, models)

    if problems:
        return None, problems
    return ParsedFlow(nodes=nodes, order=order, preds=preds, input_id=inputs[0], output_ids=outputs), []


def _topological(nodes, preds, succs) -> list[str] | None:
    remaining = {i: len(preds[i]) for i in nodes}
    ready = [i for i in nodes if remaining[i] == 0]
    order = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for s in succs[i]:
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(s)
    return order if len(order) == len(nodes) else None


def _check_node(n: ParsedNode, nodes, ancestors: set[str], tools, models) -> list[Problem]:
    problems = []
    c = n.config
    if n.type == "input":
        if len(c.fields) > MAX_FIELDS:
            problems.append(Problem(f"An Input node has at most {MAX_FIELDS} fields.", n.id))
        names = [f.name for f in c.fields]
        for name in names:
            if not IDENT.match(name):
                problems.append(Problem(f"Field name '{name}': a letter or _, then letters, digits or _.", n.id))
        if len(set(names)) != len(names):
            problems.append(Problem("Two fields have the same name.", n.id))
    if n.type == "llm" and c.model and models is not None and c.model not in models:
        problems.append(Problem(f"Unknown model '{c.model}'.", n.id))
    if n.type == "tool":
        spec = tools.get(c.tool)
        if spec is None:
            problems.append(Problem(f"'{c.tool}' is not a tool a flow can use.", n.id))
        else:
            for arg in c.args:
                if arg not in spec.args:
                    problems.append(Problem(f"'{c.tool}' has no argument '{arg}'.", n.id))
            for arg in sorted(spec.required - {a for a, v in c.args.items() if v.strip()}):
                problems.append(Problem(f"'{c.tool}' needs the argument '{arg}'.", n.id))
            for arg, value in c.args.items():
                if len(value) > MAX_TEXT:
                    problems.append(Problem(f"Argument '{arg}' is too long.", n.id))
    for where, text in texts_of(n):
        for m in REF.finditer(text):
            ref, fld = m.group(1), m.group(2)
            if ref not in nodes:
                problems.append(Problem(f"The {where} uses {m.group(0)}, but there is no node '{ref}'.", n.id))
            elif ref not in ancestors:
                problems.append(Problem(
                    f"The {where} uses {m.group(0)}, but '{ref}' does not come before this node.", n.id))
            elif fld is None and nodes[ref].type not in DEFAULT_FIELD:
                problems.append(Problem(
                    f"The {where} uses {m.group(0)}: name the field, e.g. {{{{{ref}.{_any_field(nodes[ref])}}}}}.",
                    n.id))
            elif fld is not None and fld not in fields_of(nodes[ref]):
                problems.append(Problem(f"The {where} uses {m.group(0)}, but '{ref}' has no field '{fld}'.", n.id))
    return problems


def _any_field(node: ParsedNode) -> str:
    return next(iter(sorted(fields_of(node))), "field")
