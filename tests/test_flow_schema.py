"""validate_flow: what makes a flow valid, and that each problem names its node so the editor can point at it."""

import pytest
from app.flows.catalog import tool_specs
from app.flows.schema import MAX_NODES, validate_flow

from tests.flow_fakes import TOOLS, flow, news_flow, node

SPECS = tool_specs(TOOLS)


def problems(definition, models=None):
    parsed, found = validate_flow(definition, SPECS, models)
    assert (parsed is None) == bool(found)
    return [(p.node, p.message) for p in found]


def test_a_valid_flow_is_parsed_in_running_order():
    parsed, found = validate_flow(news_flow(), SPECS)
    assert found == []
    assert parsed.order == ["in", "s", "sum", "out"]
    assert parsed.input_id == "in" and parsed.output_ids == ["out"]
    assert [f.name for f in parsed.input_fields] == ["topic"]


def test_parallel_branches_that_meet_again_are_valid():
    d = flow([node("in", "input", fields=[{"name": "q"}]),
              node("a", "llm", prompt="A {{in.q}}"), node("b", "llm", prompt="B {{in.q}}"),
              node("out", "output", template="{{a}} / {{b}}")],
             [("in", "a"), ("in", "b"), ("a", "out"), ("b", "out")])
    assert problems(d) == []


@pytest.mark.parametrize("change, node_id, says", [
    (lambda d: d["nodes"].append(node("in", "llm", prompt="x")), "in", "Two nodes have this id"),
    (lambda d: d["nodes"][2].update(id="2bad"), "2bad", "node id is a letter"),
    (lambda d: d["nodes"][2].update(id="__end__"), "__end__", "reserved name"),
    (lambda d: d["nodes"][2].update(id="outputs"), "outputs", "reserved name"),
    (lambda d: d["nodes"][2]["config"].update(prompt=""), "sum", "prompt"),
    (lambda d: d["nodes"][1]["config"].update(tool="rm_rf"), "s", "not a tool a flow can use"),
    (lambda d: d["nodes"][1]["config"].update(args={}), "s", "needs the argument 'query'"),
    (lambda d: d["nodes"][1]["config"]["args"].update(page="2"), "s", "has no argument 'page'"),
    (lambda d: d["nodes"][2]["config"].update(prompt="{{nope.text}}"), "sum", "there is no node 'nope'"),
    (lambda d: d["nodes"][2]["config"].update(prompt="{{out.text}}"), "sum", "does not come before this node"),
    (lambda d: d["nodes"][2]["config"].update(prompt="{{s.text}}"), "sum", "has no field 'text'"),
    (lambda d: d["nodes"][2]["config"].update(prompt="{{in}}"), "sum", "name the field"),
    (lambda d: d["nodes"][0]["config"].update(fields=[{"name": "a"}, {"name": "a"}]), "in", "same name"),
    (lambda d: d["nodes"][0]["config"].update(fields=[{"name": "a b"}]), "in", "Field name 'a b'"),
])
def test_each_problem_names_its_node(change, node_id, says):
    d = news_flow()
    change(d)
    found = problems(d)
    assert any(n == node_id and says in m for n, m in found), found


def test_a_flow_needs_one_input_and_an_output():
    d = flow([node("a", "llm", prompt="hi")], [])
    found = [m for _, m in problems(d)]
    assert any("exactly one Input" in m for m in found) and any("at least one Output" in m for m in found)


def test_a_loop_is_refused():
    d = news_flow()
    d["nodes"].insert(3, node("again", "llm", prompt="more"))
    d["edges"] += [{"source": "sum", "target": "again"}, {"source": "again", "target": "s"}]
    assert any("loop" in m for _, m in problems(d))


def test_a_node_cut_off_from_the_input_or_from_every_output_is_pointed_at():
    d = news_flow()
    d["nodes"] += [node("orphan", "llm", prompt="alone"), node("dead_end", "llm", prompt="{{in.topic}}")]
    d["edges"].append({"source": "in", "target": "dead_end"})
    found = problems(d)
    assert ("orphan", "This node is not reached from the Input node.") in found
    assert any(n == "dead_end" and "does not lead to an Output" in m for n, m in found)


def test_connections_into_the_input_or_out_of_an_output_are_refused():
    d = news_flow()
    d["edges"].append({"source": "out", "target": "in"})
    found = problems(d)
    assert ("in", "Nothing can lead into the Input node.") in found
    assert ("out", "Nothing can follow an Output node.") in found


def test_an_edge_to_a_missing_node_is_refused():
    d = news_flow()
    d["edges"].append({"source": "sum", "target": "ghost"})
    assert any("does not exist: ghost" in m for _, m in problems(d))


def test_a_model_an_llm_node_names_must_be_one_on_offer():
    d = news_flow()
    d["nodes"][2]["config"]["model"] = "made/up"
    assert any(n == "sum" and "Unknown model" in m for n, m in problems(d, models={"deepseek/deepseek-v4-flash"}))


def test_size_limits():
    d = flow([node(f"n{i}", "llm", prompt="x") for i in range(MAX_NODES + 1)], [])
    assert any(f"at most {MAX_NODES} nodes" in m for _, m in problems(d))


def test_a_malformed_definition_is_reported_not_raised():
    assert problems({"nodes": "not a list"})
    assert problems({"schema_version": 2, "nodes": []})
