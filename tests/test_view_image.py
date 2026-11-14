"""view_image: what the model is sent, and what is (not) saved in the thread.
A scripted model records its requests; the sandbox is a dict of bytes."""

import io

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from app.agents import vision
import importlib

from app.agents.tools import sandbox_manager
from app.agents.tools.view_image import view_image

# `app.agents.tools.view_image` as an attribute is the tool; the module is under sys.modules.
view_image_module = importlib.import_module("app.agents.tools.view_image")

VISION = "qwen/qwen3.7-plus"
TEXT_ONLY = "deepseek/deepseek-v4-flash"


def png(size=(40, 30), color=(200, 30, 30), fmt="PNG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, fmt)
    return out.getvalue()


SEEN: list = []   # every request the model got, in order


class Looker(BaseChatModel):
    """Opens photo.png on the first turn's first step, then answers. Keeps every request."""

    @property
    def _llm_type(self):
        return "looker"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        SEEN.append(list(messages))
        if isinstance(messages[-1], HumanMessage) and not any(isinstance(m, ToolMessage) for m in messages[-3:]) \
                and messages[-1].content == "what is in photo.png?":
            msg = AIMessage("", tool_calls=[{"name": "view_image", "id": "v1",
                                             "args": {"path": "photo.png", "label": "Looking"}}])
        else:
            msg = AIMessage("a red rectangle")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def sandbox(monkeypatch):
    files = {"photo.png": png()}
    reads = []

    async def read_file(thread_id, name):
        reads.append(name)
        if name not in files:
            raise FileNotFoundError(name)
        return files[name], "image/png", name

    monkeypatch.setattr(view_image_module, "read_file", read_file)
    monkeypatch.setattr(sandbox_manager, "read_file", read_file)
    vision.cache._items.clear()
    files["_reads"] = reads
    return files


def graph():
    SEEN.clear()
    return create_agent(model=Looker(), tools=[view_image], middleware=[vision.ViewedImagesMiddleware()],
                        checkpointer=InMemorySaver())


def image_parts(messages):
    return [b for m in messages if isinstance(m.content, list) for b in m.content
            if isinstance(b, dict) and b.get("type") == "image_url"]


async def run(g, model, text="what is in photo.png?", thread="t1"):
    return await g.ainvoke({"messages": [HumanMessage(text)]},
                           config={"configurable": {"thread_id": thread, "model": model}})


async def test_a_vision_model_is_sent_the_image_after_the_tool_result(sandbox):
    g = graph()
    await run(g, VISION)
    second_call = SEEN[1]
    assert isinstance(second_call[-2], ToolMessage) and second_call[-2].name == "view_image"
    last = second_call[-1]
    assert isinstance(last, HumanMessage)
    (img,) = image_parts([last])
    assert img["image_url"]["url"].startswith("data:image/png;base64,")


async def test_the_saved_thread_holds_no_image_bytes(sandbox):
    g = graph()
    state = await run(g, VISION)
    for m in state["messages"]:
        assert "base64" not in str(m.content), m
        assert "base64" not in str(getattr(m, "artifact", ""))


async def test_a_text_only_model_gets_an_explanation_and_no_image(sandbox):
    g = graph()
    state = await run(g, TEXT_ONLY)
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "[IMAGE_UNSUPPORTED_MODEL]" in tool_msg.content and "cannot see images" in tool_msg.content
    assert all(not image_parts(call) for call in SEEN)


async def test_a_later_turn_still_sees_the_image(sandbox):
    g = graph()
    await run(g, VISION)
    SEEN.clear()
    await run(g, VISION, text="thanks")                    # a follow-up that opens nothing
    assert SEEN and all(len(image_parts(call)) == 1 for call in SEEN)


async def test_the_image_is_dropped_if_the_file_changed_since_it_was_viewed(sandbox):
    g = graph()
    await run(g, VISION)
    sandbox["photo.png"] = png(color=(0, 0, 200))          # overwritten after the model looked
    vision.cache._items.clear()                            # and the cached copy is gone (a restart)
    SEEN.clear()
    await run(g, VISION, text="thanks")
    assert SEEN and all(not image_parts(call) for call in SEEN)


async def test_a_lost_cache_entry_is_reread_from_the_sandbox(sandbox, monkeypatch):
    g = graph()
    real_put = vision.cache.put
    monkeypatch.setattr(vision.cache, "put", lambda *a, **k: None)  # as after a restart
    await run(g, VISION)
    assert sandbox["_reads"].count("photo.png") == 2   # once for the tool, once for the model
    assert image_parts(SEEN[1])
    monkeypatch.setattr(vision.cache, "put", real_put)


async def test_a_missing_file_is_an_error_message_not_a_crash(sandbox):
    del sandbox["photo.png"]
    g = graph()
    state = await run(g, VISION)
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "photo.png is not in the sandbox" in tool_msg.content and "workspace_copy_to_sandbox" in tool_msg.content
    assert all(not image_parts(call) for call in SEEN)


def test_a_large_image_is_shrunk():
    data, mime, size = vision.prepare_image(png((4000, 2000)), "big.png")
    assert max(size) == vision.MAX_SIDE and mime == "image/png"
    assert Image.open(io.BytesIO(data)).size == size


def test_a_format_the_apis_do_not_take_is_converted():
    data, mime, _ = vision.prepare_image(png(fmt="BMP"), "scan.bmp")
    assert mime in ("image/png", "image/jpeg") and Image.open(io.BytesIO(data)).format in ("PNG", "JPEG")


def test_a_non_image_is_refused_with_a_reason():
    with pytest.raises(vision.ImageError, match="could not be opened"):
        vision.prepare_image(b"just some text, not a picture", "notes.png")


def test_which_models_can_see():
    assert vision.model_supports_vision(VISION)
    assert vision.model_supports_vision("anthropic/claude-opus-4.8")
    assert not vision.model_supports_vision(TEXT_ONLY)
    assert not vision.model_supports_vision(None)   # the default model is text-only


# ---- PDF pages and the image budget ---------------------------------------------------

from tests.test_check_pdf import build_pdf  # noqa: E402


class PdfLooker(BaseChatModel):
    """Asks view_image for the pages given in CALLS ([(call_id, pages), ...]) in one step, then answers."""

    @property
    def _llm_type(self):
        return "pdf-looker"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        SEEN.append(list(messages))
        if not any(isinstance(m, ToolMessage) for m in messages):
            calls = [{"name": "view_image", "id": cid, "args": {"path": "doc.pdf", "label": "Looking", **({"pages": pg} if pg is not None else {})}}
                     for cid, pg in PDF_CALLS]
            return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=calls))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])


PDF_CALLS: list = []


async def run_pdf(calls, pdf=None, model=VISION):
    PDF_CALLS[:] = calls
    SEEN.clear()
    g = create_agent(model=PdfLooker(), tools=[view_image], middleware=[vision.ViewedImagesMiddleware()],
                     checkpointer=InMemorySaver())
    return await g.ainvoke({"messages": [HumanMessage("look at the pdf")]},
                           config={"configurable": {"thread_id": "p1", "model": model}})


def tool_texts(state):
    return [m.content for m in state["messages"] if isinstance(m, ToolMessage)]


async def test_a_pdf_page_is_rendered_and_the_page_count_is_reported(sandbox):
    sandbox["doc.pdf"] = build_pdf([("text", 30), ("image", 0), ("text", 30)])
    state = await run_pdf([("c1", [2])])
    out = tool_texts(state)[0]
    assert "Rendered 1 page of doc.pdf (3 in the document) at 150 DPI. Page 2:" in out and " px" in out
    last = SEEN[1][-1]
    assert isinstance(last, HumanMessage) and "doc.pdf page 2" in str(last.content)
    assert len(image_parts([last])) == 1


async def test_asking_for_too_many_pages_is_refused_not_cut(sandbox):
    sandbox["doc.pdf"] = build_pdf([("text", 30)] * 8)
    state = await run_pdf([("c1", list(range(1, 7)))])
    (text,) = tool_texts(state)
    assert text.startswith("Error [TOO_MANY_PAGES]: 6 pages asked for; at most 5 per call")
    assert all(not image_parts(call) for call in SEEN)


async def test_a_pdf_needs_pages_and_pages_need_to_exist(sandbox):
    sandbox["doc.pdf"] = build_pdf([("text", 30), ("text", 30)])
    assert "say which pages" in tool_texts(await run_pdf([("c1", None)]))[0]
    assert "[PAGE_OUT_OF_RANGE]" in tool_texts(await run_pdf([("c1", [9])]))[0] and "has 2 pages; page 9 does not exist" in tool_texts(await run_pdf([("c1", [9])]))[0]


async def test_pages_on_a_plain_image_is_an_error(sandbox):
    sandbox["doc.pdf"] = png()      # a png under a .pdf name is still read as a pdf and refused as one
    assert tool_texts(await run_pdf([("c1", [1])]))[0].startswith("Error [DOCUMENT_UNREADABLE]:")


async def test_the_image_budget_keeps_the_newest_and_says_what_it_left_out(sandbox, monkeypatch):
    sandbox["doc.pdf"] = build_pdf([("text", 30), ("image", 0), ("image", 0)])
    monkeypatch.setattr(vision, "image_tokens", lambda size: 100)
    monkeypatch.setattr(vision, "IMAGE_TOKEN_BUDGET", 150)
    await run_pdf([("c1", [2]), ("c2", [3])])
    last = SEEN[1][-1]
    assert len(image_parts([last])) == 1
    assert "doc.pdf page 3" in str(last.content) and "doc.pdf page 2" not in str(last.content)
    assert "1 more image(s)" in str(last.content)


# ---- a provider that refuses the images ------------------------------------------------

class Rejected(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class Picky(BaseChatModel):
    """Looks at photo.png, then fails with REJECT_STATUS whenever a request carries an image."""

    @property
    def _llm_type(self):
        return "picky"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        SEEN.append(list(messages))
        if image_parts(messages):
            raise Rejected(REJECT_STATUS[0])
        if not any(isinstance(m, ToolMessage) for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[
                {"name": "view_image", "id": "v1", "args": {"path": "photo.png", "label": "Looking"}}]))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage("could not see it"))])


REJECT_STATUS = [400]


async def run_picky(status):
    REJECT_STATUS[0] = status
    SEEN.clear()
    g = create_agent(model=Picky(), tools=[view_image], middleware=[vision.ViewedImagesMiddleware()],
                     checkpointer=InMemorySaver())
    return await g.ainvoke({"messages": [HumanMessage("look")]},
                           config={"configurable": {"thread_id": "k", "model": VISION}})


async def test_a_provider_that_refuses_the_images_gets_the_request_again_without_them(sandbox):
    state = await run_picky(413)
    assert state["messages"][-1].content == "could not see it"
    retry = SEEN[-1]
    assert not image_parts(retry)
    assert "could not be sent" in str(retry[-1].content)


async def test_an_error_that_is_not_about_the_request_is_not_retried(sandbox):
    with pytest.raises(Rejected):
        await run_picky(500)


class TurnLooker(BaseChatModel):
    """"look N" opens page N of doc.pdf; anything else just gets an answer."""

    @property
    def _llm_type(self):
        return "turn-looker"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        SEEN.append(list(messages))
        last = messages[-1]
        text = last.content if isinstance(last, HumanMessage) and isinstance(last.content, str) else ""
        if text.startswith("look "):
            call = {"name": "view_image", "id": f"call-{text.split()[1]}",
                    "args": {"path": "doc.pdf", "label": "Looking", "pages": [int(text.split()[1])]}}
            return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])


async def test_the_budget_keeps_the_newest_images_across_turns(sandbox, monkeypatch):
    sandbox["doc.pdf"] = build_pdf([("image", 0)] * 3)
    monkeypatch.setattr(vision, "image_tokens", lambda size: 100)
    monkeypatch.setattr(vision, "IMAGE_TOKEN_BUDGET", 250)           # room for two
    g = create_agent(model=TurnLooker(), tools=[view_image], middleware=[vision.ViewedImagesMiddleware()],
                     checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "tl", "model": VISION}}
    for n in (1, 2, 3):
        await g.ainvoke({"messages": [HumanMessage(f"look {n}")]}, config=cfg)
    SEEN.clear()
    await g.ainvoke({"messages": [HumanMessage("thanks")]}, config=cfg)
    final = SEEN[-1]
    shown = " ".join(str(m.content) for m in final if isinstance(m, HumanMessage) and isinstance(m.content, list))
    assert len([b for m in final if isinstance(m.content, list) for b in m.content if isinstance(b, dict) and b.get("type") == "image_url"]) == 2
    assert "doc.pdf page 3" in shown and "doc.pdf page 2" in shown and "doc.pdf page 1" not in shown
    assert "1 more image(s)" in shown


# ---- the attachment store --------------------------------------------------------------

class FakeStore:
    """In-memory stand-in for the Postgres attachment store: content-addressed, per user."""

    def __init__(self):
        self.items: dict[str, tuple[str, bytes, str]] = {}

    async def save(self, user_id, thread_id, data, mime):
        import hashlib

        att_id = hashlib.sha256(f"{user_id}/{thread_id}".encode() + data).hexdigest()[:32]
        self.items[att_id] = (user_id, data, mime)
        return att_id

    async def get(self, attachment_id, user_id):
        owner, data, mime = self.items.get(attachment_id, (None, None, None))
        return (data, mime) if owner == user_id and data else None


@pytest.fixture
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr(vision, "attachment_store", s)
    vision.blob_cache._items.clear()
    return s


async def test_a_viewed_image_survives_the_sandbox_file_and_the_cache_being_gone(sandbox, store):
    g = graph()
    await run(g, VISION)
    del sandbox["photo.png"]                       # the file is deleted from the sandbox
    vision.cache._items.clear(); vision.blob_cache._items.clear()   # and a restart emptied the caches
    SEEN.clear()
    await run(g, VISION, text="thanks")
    assert SEEN and all(len(image_parts(call)) == 1 for call in SEEN)


async def test_the_tool_result_lists_the_stored_images_for_the_chat(sandbox, store):
    g = graph()
    state = await run(g, VISION)
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    (img,) = tool_msg.artifact["images"]
    assert img["id"] in store.items and img["label"] == "photo.png"
    assert "base64" not in str(tool_msg.artifact)


async def test_an_image_stored_for_one_user_is_not_shown_for_another(sandbox, store):
    g = graph()
    await g.ainvoke({"messages": [HumanMessage("what is in photo.png?")]},
                    config={"configurable": {"thread_id": "t1", "model": VISION, "user_id": "alice"}})
    del sandbox["photo.png"]
    vision.cache._items.clear(); vision.blob_cache._items.clear()
    SEEN.clear()
    await g.ainvoke({"messages": [HumanMessage("thanks")]},
                    config={"configurable": {"thread_id": "t1", "model": VISION, "user_id": "mallory"}})
    assert SEEN and all(not image_parts(call) for call in SEEN)


# ---- a model that cannot see images ----------------------------------------------------

class Recorder(BaseChatModel):
    """Answers at once and records the system prompt each request carried."""

    @property
    def _llm_type(self):
        return "recorder"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        from langchain_core.messages import SystemMessage

        def text(c):   # the system prompt is a list of text blocks once a middleware appends to it
            return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))

        SYSTEMS.append("".join(text(m.content) for m in messages if isinstance(m, SystemMessage)))
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])


SYSTEMS: list = []


async def ask_with_notice(model):
    SYSTEMS.clear()
    g = create_agent(model=Recorder(), tools=[], system_prompt="You are Jarvis.",
                     middleware=[vision.VisionNoticeMiddleware()], checkpointer=InMemorySaver())
    await g.ainvoke({"messages": [HumanMessage("what is in this image?")]},
                    config={"configurable": {"thread_id": "n1", "model": model}})
    return SYSTEMS[0]


async def test_a_model_without_vision_is_told_so_up_front():
    system = await ask_with_notice(TEXT_ONLY)
    assert system.startswith("You are Jarvis.") and "cannot see images" in system
    assert "do not guess its content from the file name" in system


async def test_a_model_with_vision_gets_no_such_note():
    assert "cannot see images" not in await ask_with_notice(VISION)


async def test_the_refusal_tells_the_agent_to_stop_not_to_look_for_a_workaround(sandbox):
    state = await run(graph(), TEXT_ONLY)
    text = next(m for m in state["messages"] if isinstance(m, ToolMessage)).content
    assert "Do not try other ways to look at it" in text and "Do not guess" in text
    assert "work from the file's metadata" not in text           # the old wording that invited workarounds


def test_the_bash_description_matches_what_the_sandbox_image_installs():
    """The description is what tells the model which libraries exist. It must name every package the
    image installs (sandbox/requirements*.txt) and no library the image does not have."""
    import re
    from pathlib import Path

    from app.agents.tools.bash import bash

    root = Path(__file__).resolve().parent.parent
    text = bash.description.lower()
    packages = []
    for f in ("requirements.txt", "requirements-nodeps.txt"):
        for line in (root / "sandbox" / f).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                packages.append(re.split(r"[<>=!~\[ ]", line)[0].lower())
    named = {"opencv-python-headless": "opencv"}              # how the description names a package whose pip name differs
    small = {"tqdm", "pytz", "regex", "tabulate"}             # installed, too minor to list one by one
    for pkg in packages:
        assert pkg in small or named.get(pkg, pkg) in text, f"{pkg} is in the image but not in the description"

    dockerfile = (root / "Dockerfile.sandbox").read_text()   # command-line tools it names must be installed
    apt = {"file": "file", "curl": "curl", "unzip": "unzip", "zip": "zip", "jq": "jq", "pandoc": "pandoc",
           "pdftotext": "poppler-utils", "pdftoppm": "poppler-utils", "tesseract": "tesseract-ocr"}
    for tool, pkg in apt.items():
        assert tool in text and pkg in dockerfile, f"{tool}: not named in the description, or missing from the Dockerfile"

    for never_installed in ("nltk", "kaleido", "yfinance", "pandoc for generating"):
        assert never_installed not in text, f"the description names {never_installed}, which the image does not have"
    assert "view_image" in text
