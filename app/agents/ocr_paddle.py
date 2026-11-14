"""PaddleOCR as the engine behind ocr.run_ocr: text detection and recognition in deep-learning
models, much better than tesseract on small, noisy, tilted or photographed text.

What it gives that the rest of ocr.py needs is the position of each WORD, not only of each
line: with `return_word_box` it reports a box for every piece of a recognised line, which are
put back together into words here (a token that starts with a space begins a word, one that ends
with a space ends it; Chinese and Japanese come out a character at a time, which is how they are
matched anyway).

The engines are created once per language and kept; they are not thread-safe, so a lock guards each.
"""

import functools
import os
import threading

import numpy as np
from PIL import Image

MIN_SCORE = float(os.environ.get("OCR_PADDLE_MIN_SCORE", "0.3"))   # a line recognised less surely than this is dropped

# tesseract language code (what ocr_image's `lang` takes) -> PaddleOCR language
LANGS = {"vie": "vi", "eng": "en", "fra": "fr", "deu": "german", "spa": "es", "jpn": "japan",
         "chi_sim": "ch", "chi_tra": "chinese_cht", "kor": "korean", "rus": "ru", "tha": "th", "ara": "ar"}

_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def available() -> bool:
    try:
        import paddleocr  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def paddle_lang(lang: str) -> str:
    """The one PaddleOCR language for a tesseract-style "vie+eng". A model reads one script family plus
    English, so the first language that is not English decides; English alone is "en"."""
    parts = [p for p in lang.replace(",", "+").split("+") if p.strip()]
    for p in parts:
        if p != "eng" and p in LANGS:
            return LANGS[p]
    return "en"


# What a language is read with, beyond the defaults (the PP-OCRv6 models); OCR_PADDLE_OPTIONS, a JSON object of the same
# shape, overrides it. {"vi": {"ocr_version": "PP-OCRv3"}} makes Vietnamese several times faster and keeps more
# of its letters, but measured on the evaluation set it finds fewer things in noisy pictures (12 of 18 against 16 of 18),
# so v6 is the default.
OPTIONS: dict[str, dict] = {}


def _options(plang: str) -> dict:
    import json

    try:
        override = json.loads(os.environ.get("OCR_PADDLE_OPTIONS") or "{}")
    except ValueError:
        override = {}
    return {**OPTIONS.get(plang, {}), **override.get(plang, {})}


@functools.lru_cache(maxsize=2)       # each language's models take a few hundred MB of memory
def _engine(plang: str):
    from paddleocr import PaddleOCR

    # The detector looks at a copy of the page no larger than 1280 px on its long side: a full A4 page at 200 dpi
    # takes ~24 s at its own size and ~9 s at this one, and the text is still found (it is read from the original).
    return PaddleOCR(lang=plang, use_textline_orientation=False, use_doc_orientation_classify=False,
                     use_doc_unwarping=False, enable_mkldnn=False,
                     text_det_limit_type="max", text_det_limit_side_len=1280, **_options(plang))


def _words(tokens: list[str], regions: list) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Words and their boxes from the tokens PaddleOCR splits a line into."""
    words: list[tuple[str, tuple[int, int, int, int]]] = []
    cur = ""
    boxes: list[tuple[float, float, float, float]] = []

    def flush() -> None:
        nonlocal cur, boxes
        if cur.strip() and boxes:
            words.append((cur.strip(), (round(min(b[0] for b in boxes)), round(min(b[1] for b in boxes)),
                                        round(max(b[2] for b in boxes)), round(max(b[3] for b in boxes)))))
        cur, boxes = "", []

    for tok, quad in zip(tokens, regions):
        pts = np.asarray(quad).reshape(-1, 2)
        box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
        if tok.strip() == "":                                  # a space on its own: a word ends here
            flush()
            continue
        if tok[0].isspace():                                   # " (" : a new word begins
            flush()
        cur += tok
        boxes.append(box)
        if tok[-1].isspace():
            flush()
    flush()
    return words


def _edge_y(a: np.ndarray, b: np.ndarray, x: float) -> float:
    """y of the straight edge from point a to point b at `x` (the nearer end's y beyond its ends)."""
    if b[0] == a[0]:
        return float(a[1])
    t = min(1.0, max(0.0, (x - a[0]) / (b[0] - a[0])))
    return float(a[1] + t * (b[1] - a[1]))


def follow_slant(box: tuple[int, int, int, int], poly) -> tuple[int, int, int, int]:
    """`box` with its height taken from the tilted line it belongs to. PaddleOCR gives a tilted line as a
    parallelogram, but the boxes of its words come back as horizontal strips between the line's two edges
    where they overlap: on a line tilted by a few degrees that is a sliver a few pixels high, across the middle
    of the letters. A word's real extent is the line's top and bottom edges at the word's own left and right ends.
    A line that is not four points, or whose long edges are not roughly horizontal, is left alone."""
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    if pts.shape[0] != 4:
        return box
    tl, tr, br, bl = pts
    if abs(tr[0] - tl[0]) < 2 * abs(tr[1] - tl[1]) or abs(br[0] - bl[0]) < 2 * abs(br[1] - bl[1]):
        return box
    x0, x1 = box[0], box[2]
    top = min(_edge_y(tl, tr, x0), _edge_y(tl, tr, x1))
    bottom = max(_edge_y(bl, br, x0), _edge_y(bl, br, x1))
    return (box[0], min(box[1], round(top)), box[2], max(box[3], round(bottom)))


def tighten(gray: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    """`box` shrunk to the ink inside it. The boxes the detector draws are loose around the text,
    a few pixels more above and below than the letters need, and a box meant to sit on a piece of text
    should hug it. Ink is whatever differs clearly from the background of the box (so light text on a
    dark ground counts too); a box with no clear ink is returned as it is."""
    h, w = gray.shape
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return box
    crop = gray[y0:y1, x0:x1].astype(np.int16)
    ink = np.abs(crop - np.median(crop)) > max(35, 0.25 * (crop.max() - crop.min()))
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if rows.size == 0 or ink.mean() < 0.01:
        return box
    return (x0 + int(cols[0]) - pad, y0 + int(rows[0]) - pad, x0 + int(cols[-1]) + 1 + pad, y0 + int(rows[-1]) + 1 + pad)


def read(img: Image.Image, lang: str) -> list[tuple[str, float, list[tuple[str, tuple[int, int, int, int]]]]]:
    """[(line text, score 0..1, [(word, box), ...])] for the image, in its own pixels."""
    plang = paddle_lang(lang)
    engine = _engine(plang)
    with _guard:
        lock = _locks.setdefault(plang, threading.Lock())
    rgb = np.array(img.convert("RGB"))
    gray = np.array(img.convert("L"))
    with lock:
        result = engine.predict(rgb, return_word_box=True)
    if not result:
        return []
    r = result[0]
    texts, scores = r.get("rec_texts") or [], r.get("rec_scores") or []
    polys = r.get("rec_polys")
    if polys is None:
        polys = r.get("dt_polys") or []
    token_lists, region_lists = r.get("text_word") or [], r.get("text_word_region") or []
    out = []
    for i, text in enumerate(texts):
        score = float(scores[i]) if i < len(scores) else 1.0
        if not text.strip() or score < MIN_SCORE:
            continue
        words = _words(token_lists[i], region_lists[i]) if i < len(token_lists) and i < len(region_lists) else []
        if not words:                                           # no word boxes: the line as one word
            pts = np.asarray(polys[i]).reshape(-1, 2)
            words = [(text.strip(), (round(pts[:, 0].min()), round(pts[:, 1].min()), round(pts[:, 0].max()), round(pts[:, 1].max())))]
        line_poly = polys[i] if i < len(polys) else None
        if line_poly is not None:
            words = [(t, follow_slant(b, line_poly)) for t, b in words]
        out.append((text.strip(), score, [(t, tighten(gray, b)) for t, b in words]))
    return out
