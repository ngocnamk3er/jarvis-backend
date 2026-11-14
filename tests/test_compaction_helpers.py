"""Pruning of bulk tool output before summarizing, the compaction trigger, and the shipped PDF skill."""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents import pruning
from app.agents.skills import load_skills
from app.core.config import settings
from app.services.chat_service import compact_threshold


def test_a_long_tool_result_keeps_its_ends_and_says_what_was_cut():
    text = "H" * 5000 + "M" * 20000 + "T" * 3000
    (out,) = pruning.prune_tool_results([ToolMessage(content=text, tool_call_id="1")])
    assert out.content.startswith("H" * pruning.HEAD_CHARS) and out.content.endswith("T" * pruning.TAIL_CHARS)
    assert "characters pruned from the middle" in out.content and "M" * 100 not in out.content
    assert len(out.content) < pruning.THRESHOLD_CHARS


def test_short_results_and_other_messages_are_left_alone():
    msgs = [HumanMessage("x" * 50_000), AIMessage("y" * 50_000), ToolMessage(content="short", tool_call_id="1")]
    assert pruning.prune_tool_results(msgs) == msgs


def test_pruning_does_not_touch_the_original_messages():
    original = ToolMessage(content="z" * 20_000, tool_call_id="1")
    pruning.prune_tool_results([original])
    assert len(original.content) == 20_000


def test_text_blocks_are_pruned_and_other_blocks_kept():
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    (out,) = pruning.prune_tool_results(
        [ToolMessage(content=[{"type": "text", "text": "q" * 20_000}, img], tool_call_id="1")])
    assert "pruned" in out.content[0]["text"] and out.content[1] == img


def test_a_big_window_compacts_at_80_percent_and_a_small_one_leaves_room():
    assert compact_threshold(1_000_000) == 800_000
    assert compact_threshold(262_000) == 262_000 - 16_384 - 65_536    # the reserve wins
    assert compact_threshold(100_000) == 50_000                         # never below half the window


def test_the_pdf_skill_ships_and_its_tools_are_loaded_on_demand():
    skills = load_skills(Path(__file__).resolve().parent.parent / "app" / "skills")
    assert {"pdf-and-images", "office-documents"} <= set(skills)
    body = skills["pdf-and-images"].body
    for tool in ("check_pdf", "read_pdf", "view_image", "tool_search"):
        assert tool in body
    deferred = {n.strip() for n in settings.DEFERRED_TOOLS.split(",")}
    assert {"check_pdf", "read_pdf", "office_convert", "office_recalculate"} <= deferred
    assert not {"view_image", "ocr_image", "draw_boxes"} & deferred          # the image tools are always in front of the model


async def test_compaction_sends_the_summarizer_pruned_tool_output(monkeypatch):
    from types import SimpleNamespace

    from app.services import chat_service

    sent = {}

    class FakeLLM:
        async def ainvoke(self, request):
            sent["request"] = request
            return AIMessage("## Primary Request and Intent\n- x")

    class FakeGraph:
        async def aget_state(self, config):
            msgs = [HumanMessage("hi", id="1"), AIMessage("", id="2", tool_calls=[{"name": "t", "id": "c", "args": {}}]),
                    ToolMessage(content="B" * 30_000, tool_call_id="c", id="3")]
            return SimpleNamespace(values={"messages": msgs}, tasks=(), next=())

        async def aupdate_state(self, config, update):
            sent["update"] = update

    monkeypatch.setattr(chat_service, "build_llm_with_fallback", lambda model=None: FakeLLM())
    monkeypatch.setattr(chat_service, "_extract_hitl_events", lambda state: [])
    await chat_service._compact_conversation(FakeGraph(), {"configurable": {"thread_id": "t"}}, None)
    tool_in_request = next(m for m in sent["request"] if isinstance(m, ToolMessage))
    assert len(tool_in_request.content) < 8_192 and "pruned" in tool_in_request.content


# ---- keeping the newest turns verbatim when condensing ---------------------------------

from app.services.chat_service import split_for_compaction  # noqa: E402


def turn(n, size=400):
    """One user turn: a question, a tool call, its result and an answer."""
    return [HumanMessage(f"question {n} " + "q" * size, id=f"h{n}"),
            AIMessage("", id=f"a{n}", tool_calls=[{"name": "t", "id": f"c{n}", "args": {}}]),
            ToolMessage(content="r" * size, tool_call_id=f"c{n}", id=f"t{n}"),
            AIMessage(f"answer {n}", id=f"f{n}")]


def test_the_newest_turns_that_fit_the_budget_are_kept():
    msgs = [m for n in range(1, 7) for m in turn(n)]
    from langchain_core.messages.utils import count_tokens_approximately as count

    one_turn = count(turn(1))
    head, tail = split_for_compaction(msgs, int(one_turn * 2.5))     # room for two turns
    assert [m.id for m in tail] == [m.id for n in (5, 6) for m in turn(n)]
    assert len(head) + len(tail) == len(msgs) and isinstance(tail[0], HumanMessage)


def test_a_turn_is_never_split_between_a_tool_call_and_its_result():
    msgs = [m for n in range(1, 5) for m in turn(n)]
    for budget in range(100, 6000, 150):
        _, tail = split_for_compaction(msgs, budget)
        if tail:
            assert isinstance(tail[0], HumanMessage)


def test_when_even_the_last_turn_is_too_big_everything_is_condensed():
    msgs = [m for n in range(1, 4) for m in turn(n)]
    head, tail = split_for_compaction(msgs, 10)
    assert tail == [] and head == msgs


async def test_compaction_keeps_the_tail_after_the_summary_and_summarizes_only_the_head(monkeypatch):
    from types import SimpleNamespace

    from app.services import chat_service

    seen = {}

    class FakeLLM:
        async def ainvoke(self, request):
            seen["request"] = request
            return AIMessage("## Primary Request and Intent\n- x")

    msgs = [m for n in range(1, 8) for m in turn(n, size=2000)]

    class FakeGraph:
        async def aget_state(self, config):
            return SimpleNamespace(values={"messages": msgs}, tasks=(), next=())

        async def aupdate_state(self, config, update):
            seen["update"] = update["messages"]

    monkeypatch.setattr(chat_service, "build_llm_with_fallback", lambda model=None: FakeLLM())
    monkeypatch.setattr(chat_service, "_extract_hitl_events", lambda state: [])
    monkeypatch.setitem(chat_service._MODEL_CONTEXT_WINDOW, "test-model", 100_000)
    monkeypatch.setattr(chat_service, "RESERVED_OUTPUT_TOKENS", 0)
    monkeypatch.setattr(chat_service, "COMPACT_RETAIN_RATIO", 0.06)            # ~6,000 tokens kept
    await chat_service._compact_conversation(FakeGraph(), {"configurable": {"thread_id": "t"}}, "test-model")

    update = seen["update"]
    from langchain_core.messages import RemoveMessage

    removed = [m.id for m in update if isinstance(m, RemoveMessage)]
    after = [m for m in update if not isinstance(m, RemoveMessage)]
    assert set(removed) == {m.id for m in msgs}                                 # all old ids removed
    assert "compacted-summary" in after[0].content                              # summary comes first
    kept = after[1:]
    assert kept and isinstance(kept[0], HumanMessage) and all(m.id is None for m in kept)
    assert [m.content for m in kept] == [m.content for m in msgs[-len(kept):]]  # the newest messages, unchanged
    sent_ids = {getattr(m, "id", None) for m in seen["request"]}
    assert not (sent_ids & {m.id for m in msgs[-len(kept):]})                   # the kept part was not summarized
