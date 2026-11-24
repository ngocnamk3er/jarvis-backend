"""An answer the user stops part-way is kept: on the page after a reload, and in what the model reads next turn."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.services import chat_service as cs
from app.services.conversation_service import serialize_messages
from app.services.interrupted import INTERRUPTED_NO_TEXT, INTERRUPTED_NOTE


class FakeGraph:
    """Streams the given events, then waits as a model does while it is still writing."""

    def __init__(self, events, messages=None):
        self.events = events
        self.messages = messages if messages is not None else [HumanMessage(content="hãy viết dài")]
        self.updates = []
        self.streamed = asyncio.Event()

    async def astream_events(self, graph_input, config=None, version="v2", **kw):
        for ev in self.events:
            yield ev
        self.streamed.set()
        await asyncio.sleep(60)

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": self.messages}, interrupts=(), next=("model",))

    async def aupdate_state(self, config, values, **kw):
        self.updates.append((values, kw))


def ev(kind, content=None, **chunk_kw):
    data = {"chunk": AIMessageChunk(content=content or "", **chunk_kw)} if kind == "on_chat_model_stream" else {}
    return {"event": kind, "run_id": "r1", "parent_ids": [], "name": "model", "data": data}


async def run_and_stop(graph, monkeypatch):
    async def no_reset(thread_id):
        pass

    monkeypatch.setattr(cs, "reset_sandbox", no_reset)
    service = cs.ChatService()
    lines = []

    async def consume():
        async for line in service._run_graph({"messages": []}, {"configurable": {"thread_id": "t"}}, graph):
            lines.append(line)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(graph.streamed.wait(), 5)
    assert await service.stop("t") is True
    await asyncio.wait_for(task, 5)
    return lines


# ---- stopping a run -------------------------------------------------------------------------------------

async def test_the_words_already_streamed_are_kept_when_the_user_stops(monkeypatch):
    graph = FakeGraph([ev("on_chat_model_start"), ev("on_chat_model_stream", "Đà Nẵng nằm ở "), ev("on_chat_model_stream", "miền Trung, bên")])
    lines = await run_and_stop(graph, monkeypatch)
    assert any('"type": "stopped"' in l for l in lines) and any('"type": "done"' in l for l in lines)
    (values, kw), = graph.updates
    msg = values["messages"][0]
    assert isinstance(msg, AIMessage) and msg.content == "Đà Nẵng nằm ở miền Trung, bên" + INTERRUPTED_NOTE
    assert msg.additional_kwargs["interrupted"] is True and "as_node" not in kw


async def test_the_reasoning_streamed_so_far_is_kept_with_it(monkeypatch):
    graph = FakeGraph([ev("on_chat_model_start"), ev("on_chat_model_stream", "", additional_kwargs={"reasoning": "đang cân nhắc "}),
                       ev("on_chat_model_stream", "Câu trả lời")])
    await run_and_stop(graph, monkeypatch)
    msg = graph.updates[0][0]["messages"][0]
    assert msg.additional_kwargs["reasoning"] == "đang cân nhắc " and msg.content.startswith("Câu trả lời")


async def test_a_run_stopped_while_it_is_only_thinking_keeps_the_thinking(monkeypatch):
    graph = FakeGraph([ev("on_chat_model_start"), ev("on_chat_model_stream", "", additional_kwargs={"reasoning": "hmm, cân nhắc "}),
                       ev("on_chat_model_stream", "", additional_kwargs={"reasoning": "các phương án"})])
    await run_and_stop(graph, monkeypatch)
    msg = graph.updates[0][0]["messages"][0]
    assert msg.additional_kwargs == {"interrupted": True, "reasoning": "hmm, cân nhắc các phương án"}
    assert msg.content == INTERRUPTED_NO_TEXT                   # never an empty message: some providers refuse one


async def test_a_run_stopped_before_anything_streamed_saves_nothing(monkeypatch):
    graph = FakeGraph([ev("on_chat_model_start")])
    await run_and_stop(graph, monkeypatch)
    assert graph.updates == []


async def test_a_finished_model_call_is_not_saved_a_second_time(monkeypatch):
    # the first call completed (it is in the checkpoint already); the run is stopped before the next one writes anything
    graph = FakeGraph([ev("on_chat_model_start"), ev("on_chat_model_stream", "một câu đã xong"), ev("on_chat_model_end"), ev("on_chat_model_start")])
    await run_and_stop(graph, monkeypatch)
    assert graph.updates == []


async def test_a_stop_that_comes_just_after_the_answer_finished_adds_nothing(monkeypatch):
    graph = FakeGraph([ev("on_chat_model_start"), ev("on_chat_model_stream", "xong rồi")],
                      messages=[HumanMessage(content="hi"), AIMessage(content="xong rồi")])
    await run_and_stop(graph, monkeypatch)
    assert graph.updates == []


async def test_a_failure_to_save_does_not_break_the_stop(monkeypatch):
    graph = FakeGraph([ev("on_chat_model_start"), ev("on_chat_model_stream", "dở dang")])

    async def boom(config, values, **kw):
        raise RuntimeError("db down")

    graph.aupdate_state = boom
    lines = await run_and_stop(graph, monkeypatch)
    assert any('"type": "stopped"' in l for l in lines)


async def test_an_answer_written_after_tool_calls_is_kept_after_them(monkeypatch):
    from langchain_core.messages import ToolMessage

    history = [HumanMessage(content="q"), AIMessage(content="", tool_calls=[{"name": "bash", "id": "1", "args": {}}]),
               ToolMessage(content="ok", tool_call_id="1")]
    graph = FakeGraph([ev("on_chat_model_start"), ev("on_chat_model_stream", "Kết quả là")], messages=history)
    await run_and_stop(graph, monkeypatch)
    assert graph.updates[0][0]["messages"][0].content.startswith("Kết quả là")


# ---- showing it again and handing it to the model -------------------------------------------------------

def test_a_stopped_answer_shows_without_the_note_but_marked():
    messages = [HumanMessage(content="hãy viết dài"),
                AIMessage(content="Đà Nẵng nằm ở" + INTERRUPTED_NOTE, additional_kwargs={"interrupted": True, "reasoning": "nghĩ"})]
    parts = serialize_messages(messages)[1]["parts"]
    assert parts == [{"type": "thinking", "content": "nghĩ", "isStreaming": False, "interrupted": True},
                     {"type": "text", "content": "Đà Nẵng nằm ở", "interrupted": True}]


def test_a_stopped_answer_with_only_thinking_shows_the_thinking_and_no_text():
    messages = [HumanMessage(content="hãy viết dài"),
                AIMessage(content=INTERRUPTED_NO_TEXT, additional_kwargs={"interrupted": True, "reasoning": "đang nghĩ"})]
    parts = serialize_messages(messages)[1]["parts"]
    assert parts == [{"type": "thinking", "content": "đang nghĩ", "isStreaming": False, "interrupted": True}]


def test_an_ordinary_answer_is_untouched():
    parts = serialize_messages([HumanMessage(content="hi"), AIMessage(content="xin chào")])[1]["parts"]
    assert parts == [{"type": "text", "content": "xin chào"}]


def test_the_model_reads_the_note_so_it_knows_the_answer_was_cut_off():
    msg = AIMessage(content="Đà Nẵng nằm ở" + INTERRUPTED_NOTE)
    assert "stopped this response" in msg.content and "incomplete" in msg.content
