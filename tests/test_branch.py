"""Branching: a new conversation that starts as a copy of the first N turns of another."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.clients import conversation_client
from app.services import conversation_service as cs


def convo():
    """Two turns: the first with a tool call, the second a plain answer."""
    return [
        HumanMessage(content="q1", id="h1"),
        AIMessage(content="", id="a1", tool_calls=[{"name": "web_search", "args": {}, "id": "c1"}]),
        ToolMessage(content="result", tool_call_id="c1", id="t1"),
        AIMessage(content="answer 1", id="a2"),
        HumanMessage(content="q2", id="h2"),
        AIMessage(content="answer 2", id="a3"),
    ]


# ---- where a conversation can be cut ------------------------------------------------------------------

def test_keeping_the_first_turn_cuts_before_the_second_question():
    assert [m.id for m in cs.messages_for_branch(convo(), 1)] == ["h1", "a1", "t1", "a2"]


def test_keeping_every_turn_copies_everything():
    assert len(cs.messages_for_branch(convo(), 2)) == 6


def test_keeping_no_turns_gives_an_empty_conversation():
    assert cs.messages_for_branch(convo(), 0) == []


def test_more_turns_than_the_conversation_has_is_refused():
    with pytest.raises(HTTPException) as e:
        cs.messages_for_branch(convo(), 3)
    assert e.value.status_code == 400


def test_a_cut_after_an_unanswered_tool_call_is_refused():
    msgs = convo()[:2]                      # human, AI with a tool call, no result yet
    with pytest.raises(HTTPException):
        cs.messages_for_branch(msgs, 1)


def test_a_cut_ending_on_a_stopped_answer_is_allowed():
    msgs = [HumanMessage(content="q", id="h"), AIMessage(content="par", id="a", additional_kwargs={"interrupted": True})]
    assert len(cs.messages_for_branch(msgs, 1)) == 2


def test_the_summary_message_is_not_counted_as_a_turn():
    summary = HumanMessage(content=cs.SUMMARY_PREFIX + " earlier stuff", id="s")
    msgs = [summary, AIMessage(content="ok", id="a0"), *convo()]
    kept = cs.messages_for_branch(msgs, 1)
    assert [m.id for m in kept] == ["s", "a0", "h1", "a1", "t1", "a2"]


def test_a_negative_count_is_refused():
    with pytest.raises(HTTPException):
        cs.messages_for_branch(convo(), -1)


# ---- the whole fork, against the real graph with an in-memory checkpointer ----------------------------------

class FakeConversations:
    def __init__(self, conv):
        self.conv = conv
        self.created, self.deleted, self.tokens, self.touched = [], [], {}, []

    async def get_conversation(self, thread_id):
        return self.conv if thread_id == self.conv.id else None

    async def create_conversation(self, title, user_id):
        self.created.append((title, user_id))
        return {"id": "new-thread", "title": title, "last_model": None, "last_subagent_model": None,
                "context_tokens": 0, "created_at": "2026-11-15T00:00:00Z", "updated_at": "2026-11-15T00:00:00Z"}

    async def set_context_tokens(self, thread_id, tokens):
        self.tokens[thread_id] = tokens

    async def touch_conversation(self, thread_id, model=None, subagent_model=None):
        self.touched.append((thread_id, model, subagent_model))

    async def delete_conversation(self, thread_id):
        self.deleted.append(thread_id)


@pytest.fixture
def setup(monkeypatch):
    conv = SimpleNamespace(id="old", user_id="u1", title="Trip planning", last_model="m1", last_subagent_model=None)
    fake = FakeConversations(conv)
    for name in ("get_conversation", "create_conversation", "set_context_tokens", "touch_conversation", "delete_conversation"):
        monkeypatch.setattr(conversation_client, name, getattr(fake, name))
    graph = build_graph(checkpointer=InMemorySaver(), store=None, include_tools=False)
    return graph, fake


async def seed(graph, thread="old", messages=None):
    await graph.aupdate_state({"configurable": {"thread_id": thread}}, {"messages": messages or convo()},
                              as_node=cs._node_that_ends_the_run(graph))


async def test_a_branch_holds_the_first_turns_and_the_original_is_untouched(setup):
    graph, fake = setup
    await seed(graph)
    new = await cs.fork_conversation(graph, "old", "u1", 1)
    st = await graph.aget_state({"configurable": {"thread_id": "new-thread"}})
    assert [m.id for m in st.values["messages"]] == ["h1", "a1", "t1", "a2"]
    assert st.next == ()                                      # not shown as an interrupted run
    old = await graph.aget_state({"configurable": {"thread_id": "old"}})
    assert len(old.values["messages"]) == 6
    assert new["title"] == "Trip planning (branch)" and fake.created == [("Trip planning (branch)", "u1")]
    assert new["last_model"] == "m1" and fake.touched == [("new-thread", "m1", None)]
    assert fake.tokens["new-thread"] > 0 and new["context_tokens"] == fake.tokens["new-thread"]


async def test_the_branch_can_be_continued_with_its_own_messages(setup):
    graph, _ = setup
    await seed(graph)
    await cs.fork_conversation(graph, "old", "u1", 1)
    cfg = {"configurable": {"thread_id": "new-thread"}}
    # what the start of a new turn does: the user's message goes in and the model is next
    await graph.aupdate_state(cfg, {"messages": [HumanMessage(content="different q2", id="h9")]}, as_node="__start__")
    st = await graph.aget_state(cfg)
    assert [m.id for m in st.values["messages"]][-1] == "h9" and len(st.values["messages"]) == 5
    assert st.next                                           # the model is due to answer it


async def test_branching_before_the_first_question_makes_an_empty_conversation(setup):
    graph, fake = setup
    await seed(graph)
    await cs.fork_conversation(graph, "old", "u1", 0)
    st = await graph.aget_state({"configurable": {"thread_id": "new-thread"}})
    assert not st.values.get("messages") and "new-thread" not in fake.tokens


async def test_someone_elses_conversation_is_not_found(setup):
    graph, _ = setup
    await seed(graph)
    with pytest.raises(HTTPException) as e:
        await cs.fork_conversation(graph, "old", "someone-else", 1)
    assert e.value.status_code == 404


async def test_a_refused_cut_creates_nothing(setup):
    graph, fake = setup
    await seed(graph)
    with pytest.raises(HTTPException):
        await cs.fork_conversation(graph, "old", "u1", 5)
    assert fake.created == []


async def test_a_failure_while_copying_removes_the_empty_conversation(setup, monkeypatch):
    graph, fake = setup
    await seed(graph)

    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(graph, "aupdate_state", boom)
    with pytest.raises(RuntimeError):
        await cs.fork_conversation(graph, "old", "u1", 1)
    assert fake.deleted == ["new-thread"]
