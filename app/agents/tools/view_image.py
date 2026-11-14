import asyncio
import hashlib
import io
import logging
from pathlib import PurePosixPath
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from PIL import Image
from langchain_core.tools import InjectedToolCallId, tool

from app.agents import vision
from app.agents.tools.sandbox_errors import read_error
from app.agents.tools.sandbox_manager import get_thread_id, normalize_workspace_path, read_file


logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def view_image(
    path: str,
    label: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    pages: list[int] | None = None,
) -> tuple[str, dict | None]:
    """Look at an image file, or pages of a PDF or Office document, in your bash sandbox.

    After this call the picture is shown to you with the next step of this
    turn, so you can describe it, read what it says, or answer questions about
    it. Use it for a photo, screenshot, scan, chart or diagram — a file the
    user uploaded (copy it in with `workspace_copy_to_sandbox` first) or one you made with
    bash. The picture stays attached on later turns, up to a limit on how many
    images are shown at once (the newest are kept), so a follow-up question can
    be answered from it. If it is a picture you need again later and many have
    been opened since, open it again.

    For a PDF, pass `pages` (1-based, at most 5 per call) to see those pages as
    images — the way to read a scanned page, or to see a table or figure. A Word,
    Excel or PowerPoint file (.docx .xlsx .pptx and the older and OpenDocument
    forms) works the same way: its pages or slides are drawn for you, which is how
    to check layout, charts and formatting.
    `check_pdf` tells you which pages are scans; `read_pdf` gets the text of the
    others. Asking for more pages than allowed is refused, not cut short, and
    the result always states how many pages the document has.

    It does not run OCR or analyse pixels; you see the image yourself. For
    exact measurements or colours, compute them in bash instead.

    Args:
        path: The file's path in your working directory, e.g. "photo.jpg" or "scan.pdf".
        pages: PDF only: the page numbers to look at, e.g. [3] or [1, 2, 3].
        label: Brief human-readable description shown to the user (e.g. "Looking at the screenshot").
    """
    model = config.get("configurable", {}).get("model")
    if not vision.model_supports_vision(model):
        return (
            "Error [IMAGE_UNSUPPORTED_MODEL]: the model selected for this conversation cannot see images, "
            "and bash cannot see a picture either. "
            "Do not try other ways to look at it. Stop, tell the user plainly that you cannot see the image "
            "with this model, and suggest switching to a vision-capable one (Qwen 3.7+, Qwen 3.5 9B or "
            "Claude Opus 4.8). Do not guess what the picture shows from its file name.",
            None,
        )

    name = normalize_workspace_path(path)
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        return "Error: path must be inside your working directory (no '..', no absolute path).", None
    paged = name.lower().endswith(".pdf") or vision.is_office(name)
    if pages is not None and not paged:
        return "Error: `pages` only applies to a PDF or an Office document.", None
    if paged:
        if not pages:
            return f"Error: {name} is a document; say which pages to look at with `pages`, e.g. [1]. For a PDF, check_pdf tells you which pages are scans.", None
        pages = sorted(set(pages))
        if len(pages) > vision.MAX_PDF_PAGES_PER_CALL:
            return (f"Error [TOO_MANY_PAGES]: {len(pages)} pages asked for; at most {vision.MAX_PDF_PAGES_PER_CALL} per call. "
                    f"Ask for fewer pages at a time."), None
    try:
        raw, _, filename = await read_file(get_thread_id(config), name)
    except Exception as e:  # noqa: BLE001 — a missing file or a sandbox hiccup, same to the model
        return read_error(name, e), None

    try:
        # Decoding and rendering are CPU-bound; keep them off the event loop.
        total, images = await asyncio.to_thread(vision.load_images, raw, filename, pages)
    except vision.ImageError as e:
        return f"Error [{e.code}]: {e}" if e.code else f"Error: {e}", None

    vision.cache.put(tool_call_id, images)
    # Save each picture to the attachment store and refer to it by id. If the store
    # is down the tool still works: the model is shown the images this turn, and the
    # sha + path let a later turn reload them from the sandbox instead.
    cfg = config.get("configurable", {})
    attachments = []
    for label, data, mime, tokens in images:
        try:
            att_id = await vision.attachment_store.save(cfg.get("user_id", ""), get_thread_id(config), data, mime)
        except Exception:  # noqa: BLE001
            logger.warning("could not save %s to the attachment store", label, exc_info=True)
            att_id = None
        attachments.append({"id": att_id, "label": label, "mime": mime, "tokens": tokens})
    artifact = {
        vision.ARTIFACT_KEY: {"path": name, "pages": pages, "sha": hashlib.sha256(raw).hexdigest(),
                              "attachments": attachments},
        "images": [{"id": a["id"], "label": a["label"]} for a in attachments if a["id"]],   # what the chat shows
    }
    if total is None:
        _, data, mime, tokens = images[0]
        return (f"Opened {filename} ({len(data) >> 10} KB, {mime}). The image is attached "
                f"to the next message — look at it there."), artifact
    sizes = []
    for (label, data, _mime, _tokens), page in zip(images, pages):
        with Image.open(io.BytesIO(data)) as im:
            sizes.append(f"Page {page}: {im.width}x{im.height} px")
    return (f"Rendered {len(images)} page{'s' if len(images) != 1 else ''} of {filename} "
            f"({total} in the document) at {round(vision.PDF_RENDER_SCALE * 72)} DPI. " + "; ".join(sizes) +
            ". The images are attached to the next message — look at them there."), artifact
