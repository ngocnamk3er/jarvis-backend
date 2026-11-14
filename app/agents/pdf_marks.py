"""Pages of a PDF as images, and boxes laid back over the PDF itself.

OCR and drawing work on pictures, so a page is rendered to a PNG first (`render_page`); the boxes
found there are in that PNG's pixels. To mark the PDF, the boxes are drawn on a transparent
picture of the same size and stretched over the page (`annotate`), which leaves the PDF's own text
and scans untouched underneath.
"""

import io

from PIL import Image

from app.agents import ocr
from app.agents.vision import PDFIUM_LOCK

RENDER_DPI = 200                       # a page of body text reads cleanly in OCR at this size
MAX_PAGE_PIXELS = 24_000_000           # a poster-sized page is scaled down rather than refused
MAX_PAGES = 2000


class PdfError(Exception):
    pass


def _open(data: bytes, name: str):
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(data)
    except Exception as e:  # noqa: BLE001 — encrypted or damaged
        raise PdfError(f"{name} could not be opened as a PDF ({type(e).__name__}); it may be encrypted or damaged.") from e


def render_page(data: bytes, name: str, page: int) -> tuple[bytes, int, tuple[int, int]]:
    """(PNG of the page, number of pages, (width, height) in pixels) for the 1-based `page`."""
    with PDFIUM_LOCK:
        return _render_page(data, name, page)


def _render_page(data: bytes, name: str, page: int) -> tuple[bytes, int, tuple[int, int]]:
    doc = _open(data, name)
    try:
        total = len(doc)
        if page < 1 or page > total:
            raise PdfError(f"{name} has {total} page{'s' if total != 1 else ''}; page {page} does not exist.")
        pg = doc[page - 1]
        w, h = pg.get_size()
        scale = RENDER_DPI / 72
        if w * h * scale * scale > MAX_PAGE_PIXELS:
            scale = (MAX_PAGE_PIXELS / (w * h)) ** 0.5
        bitmap = pg.render(scale=scale)
        img = bitmap.to_pil().convert("RGB")
        bitmap.close()                       # freed here, under the lock, not later by the garbage collector in some other thread
        pg.close()
        out = io.BytesIO()
        img.save(out, "PNG")
        return out.getvalue(), total, img.size
    finally:
        doc.close()


def annotate(data: bytes, name: str, pages: dict[int, dict]) -> bytes:
    """The PDF with boxes over the pages in `pages`.

    `pages` maps a 1-based page number to {"size": [w, h] of the image the boxes are in,
    "boxes": [(label, value box, key box or None), ...] (a pair without the key box is accepted),
    "obstacles": [boxes to keep the labels off]}."""
    with PDFIUM_LOCK:
        return _annotate(data, name, pages)


def _annotate(data: bytes, name: str, pages: dict[int, dict]) -> bytes:
    import pypdfium2 as pdfium

    doc = _open(data, name)
    try:
        total = len(doc)
        for number in sorted(pages):
            info = pages[number]
            if number < 1 or number > total or not info["boxes"]:
                continue
            pg = doc[number - 1]
            if pg.get_rotation():
                raise PdfError(f"page {number} of {name} is rotated, which is not supported when marking the PDF itself; "
                               f"the marked PNG of that page is still available.")
            left, bottom, right, top = pg.get_mediabox()
            size = (int(info["size"][0]), int(info["size"][1]))
            fields = [(label, tuple(box), tuple(rest[0]) if rest and rest[0] else None) for label, box, *rest in info["boxes"]]
            layer = ocr.overlay(size, fields, [tuple(o) for o in info.get("obstacles", [])])
            _put_image(pdfium, doc, pg, layer, (right - left, top - bottom, left, bottom))
            pg.close()
        out = io.BytesIO()
        doc.save(out)
        return out.getvalue()
    finally:
        doc.close()


def _put_image(pdfium, doc, page, img: Image.Image, area: tuple[float, float, float, float]) -> None:
    """Lay `img` over `page` in the rectangle (width, height, left, bottom) of its own coordinates. Every
    pdfium object made here is closed here, while the caller holds the lock."""
    width, height, left, bottom = area
    bitmap = pdfium.PdfBitmap.from_pil(img)
    obj = pdfium.PdfImage.new(doc)
    obj.set_bitmap(bitmap)
    bitmap.close()
    obj.set_matrix(pdfium.PdfMatrix(width, 0, 0, height, left, bottom))
    page.insert_obj(obj)
    page.gen_content()
