"""Letting the agent look at an image that sits in its sandbox.

`view_image` (tools/view_image.py) reads the file and checks it is something a
model can be shown. The picture itself does not travel in the tool result:
OpenAI-style chat APIs only take text in a `tool` message, so an image there
works on some providers and is silently dropped on others. Instead
`ViewedImagesMiddleware` adds the image to the request as an ordinary user
message right after the tool results, for every model call of the turn.

Nothing here is saved to the conversation state. The tool result stays a short
text note; the bytes live in a small in-process cache (and are re-read from the
sandbox if the cache lost them), so a checkpoint never carries base64. The
images stay attached on later turns, so a follow-up question can still be
answered from the picture, but only up to IMAGE_TOKEN_BUDGET per request, newest
first: older images drop out instead of the cost growing with every turn. If the
file has changed since it was viewed, the image is left out rather than shown
as something it no longer is.
"""

import asyncio
import base64
import hashlib
import io
import threading
import logging
from collections import OrderedDict
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage

from app.schemas.chat import AVAILABLE_MODELS, DEFAULT_MODEL

logger = logging.getLogger(__name__)

TOOL_NAME = "view_image"
ARTIFACT_KEY = "view_image"

MAX_INPUT_BYTES = 50 << 20   # refuse anything bigger than this before decoding (DeepSeek Harness's source cap)
MAX_SEND_BYTES = 4 << 20     # what is actually sent to the model, after downscaling
MAX_SIDE = 1568              # longest side in pixels; larger images are shrunk
IMAGE_TOKEN_BUDGET = 16_000  # estimated tokens of images shown per model call (newest first)
MAX_PDF_PAGES_PER_CALL = 5   # pages one view_image call may ask for; more is refused, not cut
# pdfium (behind pypdfium2) is not safe to use from several threads at once: concurrent renders in the
# server's worker threads crashed the whole process (a segfault, exit 139). Everything that touches it
# holds this lock.
PDFIUM_LOCK = threading.RLock()

PDF_RENDER_SCALE = 150 / 72  # ~150 DPI: legible body text without a huge bitmap

# Types a model API accepts as-is. Anything else Pillow can open is converted.
_PASSTHROUGH = {"image/png", "image/jpeg", "image/webp", "image/gif"}

_VISION_MODELS = frozenset(m["id"] for m in AVAILABLE_MODELS if m.get("vision"))


def model_supports_vision(model_id: str | None) -> bool:
    return (model_id or DEFAULT_MODEL) in _VISION_MODELS


class ImageError(Exception):
    """The file cannot be shown; the message is written for the model.

    `code` is a stable name for the refusal, shown to the model in square
    brackets (as DeepSeek Harness's IMAGE_UNSUPPORTED_* codes are) so it can tell
    kinds of refusal apart without parsing prose."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


def prepare_image(data: bytes, name: str) -> tuple[bytes, str, tuple[int, int] | None]:
    """(bytes, mime type, (width, height) or None) ready to send to a model.

    Shrinks anything over MAX_SIDE and converts formats the APIs do not take
    (bmp, tiff, ...). Without Pillow, only a png/jpeg/webp/gif already under the
    size cap can be shown, as it is.
    """
    if len(data) > MAX_INPUT_BYTES:
        raise ImageError(f"{name} is {len(data) >> 20} MB, over the {MAX_INPUT_BYTES >> 20} MB limit.", "IMAGE_TOO_LARGE")
    try:
        from PIL import Image, ImageOps
    except ImportError:
        mime = _sniff(data)
        if mime not in _PASSTHROUGH:
            raise ImageError(f"{name} is not a png, jpeg, webp or gif, and this server cannot convert it.", "IMAGE_UNSUPPORTED_FORMAT")
        if len(data) > MAX_SEND_BYTES:
            raise ImageError(f"{name} is too large to show and this server cannot shrink it.", "IMAGE_TOO_LARGE")
        return data, mime, None

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            fmt = (img.format or "").upper()
            img = ImageOps.exif_transpose(img)   # phone photos: honour the rotation tag
            size = img.size
            shrink = max(size) > MAX_SIDE
            if not shrink and f"image/{fmt.lower()}" in _PASSTHROUGH and len(data) <= MAX_SEND_BYTES:
                return data, f"image/{fmt.lower()}", size
            if shrink:
                img.thumbnail((MAX_SIDE, MAX_SIDE))
            # Screenshots and scans are PNG for a reason (sharp text); keep them
            # lossless when the result still fits, and fall back to JPEG if not.
            lossless = fmt == "PNG" or img.mode in ("RGBA", "LA", "P")
            attempts = ([("PNG", {"optimize": True})] if lossless else []) + [
                ("JPEG", {"quality": 88}), ("JPEG", {"quality": 70})]
            for kind, opts in attempts:
                out = io.BytesIO()
                img.convert("RGBA" if kind == "PNG" else "RGB").save(out, kind, **opts)
                if len(out.getvalue()) <= MAX_SEND_BYTES:
                    return out.getvalue(), f"image/{kind.lower()}", img.size
            raise ImageError(f"{name} is still too large to show after shrinking it.", "IMAGE_TOO_LARGE")
    except ImageError:
        raise
    except Exception as e:  # noqa: BLE001 — Pillow raises many types for a bad file
        raise ImageError(f"{name} could not be opened as an image ({type(e).__name__}).", "IMAGE_UNSUPPORTED_FORMAT") from e


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


def image_tokens(size: tuple[int, int] | None) -> int:
    """Rough token cost of an image (width * height / 750, the usual estimate).
    Unknown dimensions count as a full-size image."""
    if not size:
        return 2_500
    return max(1, size[0] * size[1] // 750)


def render_pdf_pages(data: bytes, name: str, pages: list[int]) -> tuple[int, list[tuple[int, bytes, str, tuple[int, int] | None]]]:
    """(total pages, [(page number, image bytes, mime, size)]) for 1-based `pages`.

    Raises ImageError for a page out of range or a file that cannot be opened."""
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        return _render_pdf_pages(pdfium, data, name, pages)


def _render_pdf_pages(pdfium, data: bytes, name: str, pages: list[int]):
    try:
        doc = pdfium.PdfDocument(data)
    except Exception as e:  # noqa: BLE001 — encrypted or damaged
        raise ImageError(f"{name} could not be opened as a PDF ({type(e).__name__}); it may be encrypted or damaged.", "DOCUMENT_UNREADABLE") from e
    try:
        total = len(doc)
        bad = [p for p in pages if p < 1 or p > total]
        if bad:
            raise ImageError(f"{name} has {total} page{'s' if total != 1 else ''}; page {bad[0]} does not exist.", "PAGE_OUT_OF_RANGE")
        out = []
        for p in pages:
            buf = io.BytesIO()
            page = doc[p - 1]
            bitmap = page.render(scale=PDF_RENDER_SCALE)
            bitmap.to_pil().save(buf, "PNG")
            bitmap.close()                 # freed under the lock, not later by the garbage collector in another thread
            page.close()
            img, mime, size = prepare_image(buf.getvalue(), f"{name} page {p}")
            out.append((p, img, mime, size))
        return total, out
    finally:
        doc.close()


class _Cache:
    """A small in-process cache, newest kept. A miss is never an error: the caller
    goes back to where the data really lives."""

    def __init__(self, limit: int = 24):
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._limit = limit

    def put(self, key: str, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._limit:
            self._items.popitem(last=False)

    def get(self, key: str) -> Any:
        return self._items.get(key)


cache = _Cache()                  # legacy path: prepared images by tool_call_id
blob_cache = _Cache(limit=48)     # (bytes, mime) by attachment id, saves a database read per model call


class _PostgresStore:
    """The attachment store (DeepSeek Harness's term): where a viewed image lives
    for good. Tests swap this for an in-memory one."""

    async def save(self, user_id: str, thread_id: str, data: bytes, mime: str) -> str:
        from app.db import attachment_repository

        return await attachment_repository.save(user_id, thread_id, data, mime)

    async def get(self, attachment_id: str, user_id: str) -> tuple[bytes, str] | None:
        from app.db import attachment_repository

        return await attachment_repository.get(attachment_id, user_id)


attachment_store: Any = _PostgresStore()


def _image_refs(message: Any) -> dict | None:
    if not isinstance(message, ToolMessage) or message.name != TOOL_NAME:
        return None
    art = message.artifact
    if isinstance(art, dict) and isinstance(art.get(ARTIFACT_KEY), dict):
        return art[ARTIFACT_KEY]
    return None


def is_office(path: str) -> bool:
    from app.agents import office

    return PurePosixPath(path.lower()).suffix in office.OFFICE_EXTENSIONS


def load_images(raw: bytes, path: str, pages: list[int] | None) -> tuple[int | None, list[tuple[str, bytes, str, int]]]:
    """What view_image shows for `path`: (total pages or None for a plain image,
    [(label, bytes, mime, estimated tokens)]). Shared by the tool and by the
    middleware's reload after a cache miss, so both produce the same thing."""
    if pages is None:
        data, mime, size = prepare_image(raw, path)
        return None, [(path, data, mime, image_tokens(size))]
    if is_office(path):
        # Word/Excel/PowerPoint are drawn by LibreOffice: convert to a PDF, then render its pages.
        from app.agents import office

        try:
            raw = office.pdf_cache.get_or_convert(raw, path)
        except office.OfficeError as e:
            raise ImageError(str(e), "DOCUMENT_UNREADABLE") from e
    total, rendered = render_pdf_pages(raw, path, pages)
    return total, [(f"{path} page {p}", data, mime, image_tokens(size)) for p, data, mime, size in rendered]


class ViewedImagesMiddleware(AgentMiddleware):
    """Shows the model the images `view_image` opened in this conversation, newest
    first, until IMAGE_TOKEN_BUDGET is spent."""

    async def _from_store(self, ref: dict, user_id: str) -> list[tuple[str, bytes, str, int]] | None:
        out = []
        for a in ref["attachments"]:
            hit = blob_cache.get(a["id"])
            if hit is None:
                hit = await attachment_store.get(a["id"], user_id)
                if hit is None:
                    return None
                blob_cache.put(a["id"], hit)
            out.append((a["label"], hit[0], hit[1], a["tokens"]))
        return out

    async def _load(self, ref: dict, tool_call_id: str, thread_id: str, user_id: str = "") -> list[tuple[str, bytes, str, int]] | None:
        if ref.get("attachments") and all(a.get("id") for a in ref["attachments"]):
            try:
                stored = await self._from_store(ref, user_id)
            except Exception:  # noqa: BLE001 — fall back to the sandbox rather than end the turn
                logger.warning("could not read attachments for %s", ref.get("path"), exc_info=True)
                stored = None
            if stored is not None:
                return stored
        hit = cache.get(tool_call_id)
        if hit:
            return hit
        from app.agents.tools import sandbox_manager

        try:
            raw, _, _ = await sandbox_manager.read_file(thread_id, ref["path"])
            if ref.get("sha") and hashlib.sha256(raw).hexdigest() != ref["sha"]:
                logger.info("%s changed since it was viewed; leaving the image out", ref["path"])
                return None
            _, images = await asyncio.to_thread(load_images, raw, ref["path"], ref.get("pages"))
        except Exception:  # noqa: BLE001 — a missing file must not end the turn
            logger.warning("could not reload %s for the model", ref.get("path"), exc_info=True)
            return None
        cache.put(tool_call_id, images)
        return images

    async def _apply(self, request: ModelRequest) -> ModelRequest:
        config: dict = {}
        try:
            from langgraph.config import get_config

            config = get_config().get("configurable", {})
        except RuntimeError:
            return request
        if not model_supports_vision(config.get("model")):
            return request

        messages = list(request.messages)
        wanted = [(i, m) for i, m in enumerate(messages) if _image_refs(m) is not None]
        if not wanted:
            return request

        thread_id = config.get("thread_id", "default")
        user_id = config.get("user_id", "")
        budget = IMAGE_TOKEN_BUDGET
        left_out = 0
        by_block_end: dict[int, list[tuple[str, bytes, str]]] = {}
        for i, m in reversed(wanted):                       # newest first
            images = await self._load(_image_refs(m), m.tool_call_id, thread_id, user_id)
            if images is None:
                continue
            kept = []
            for label, data, mime, tokens in images:
                if tokens > budget:
                    left_out += 1
                    continue
                budget -= tokens
                kept.append((label, data, mime))
            if not kept:
                continue
            end = i
            while end + 1 < len(messages) and isinstance(messages[end + 1], ToolMessage):
                end += 1                       # keep the tool results of one step together
            by_block_end.setdefault(end, []).extend(kept)
        if not by_block_end:
            return request

        out: list = []
        for i, m in enumerate(messages):
            out.append(m)
            group = by_block_end.get(i)
            if not group:
                continue
            content: list[dict] = []
            for label, data, mime in group:
                content.append({"type": "text", "text": f"Image from view_image: {label}"})
                b64 = base64.b64encode(data).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            if left_out and i == max(by_block_end):
                content.append({"type": "text", "text": f"({left_out} more image(s) from earlier in this turn were left "
                                                        f"out to stay within the image budget; open them again if needed.)"})
            out.append(HumanMessage(content=content))
        return request.override(messages=out)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        raise NotImplementedError("ViewedImagesMiddleware needs the async graph (it reads the sandbox)")

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        with_images = await self._apply(request)
        if with_images is request:
            return await handler(request)
        try:
            return await handler(with_images)
        except Exception as e:  # noqa: BLE001 — see _rejected_request
            if not _rejected_request(e):
                raise
            # The provider refused the request, and the images are the likeliest
            # reason (too big, or a model that takes none). Try once more without
            # them and say so, rather than ending the turn on an error.
            logger.warning("model request with images was rejected (%s); retrying without them", e)
            note = HumanMessage(
                "The images you opened with view_image could not be sent: the model provider "
                "rejected the request. Carry on without seeing them, and tell the user that "
                "you could not view the image."
            )
            return await handler(request.override(messages=[*request.messages, note]))


def _rejected_request(e: Exception) -> bool:
    """A provider refusing the request itself (400 bad request, 413 too large,
    422 unprocessable) as opposed to being down or rate limited, which a retry
    without images would not fix."""
    return getattr(e, "status_code", None) in (400, 413, 422)


_NO_VISION_NOTE = (
    "\n\n## Images with this model\n"
    "The model you are running as cannot see images. If the user gives you an image "
    "(or asks what is in one), say so in your first reply and suggest switching to a "
    "vision-capable model (Qwen 3.7+, Qwen 3.5 9B or Claude Opus 4.8). Do not go looking "
    "for another way to see it, and do not guess its content from the file name. You can "
    "still help with the rest of the request, and with PDFs that have a text layer or "
    "Office files."
)


class VisionNoticeMiddleware(AgentMiddleware):
    """Adds a short note to the system prompt when the selected model cannot see
    images, so an agent given a picture says so at once instead of discovering it
    through a failed `view_image` and then trying everything else it can think of."""

    def _apply(self, request: ModelRequest) -> ModelRequest:
        try:
            from langgraph.config import get_config

            model = get_config().get("configurable", {}).get("model")
        except RuntimeError:
            return request
        if model_supports_vision(model):
            return request
        from deepagents.middleware._utils import append_to_system_message

        return request.override(system_message=append_to_system_message(request.system_message, _NO_VISION_NOTE))

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        return await handler(self._apply(request))
