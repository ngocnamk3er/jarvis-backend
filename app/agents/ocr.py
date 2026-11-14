"""Reading the text in an image together with where each line sits, and drawing boxes on
it from that.

Why this exists: asked to mark things on an image, a model that can see it writes pixel
coordinates by eye, and they come out tens of pixels off. Here the positions come from
OCR instead, the model only says WHICH recognised line holds a thing, and the drawing
tool accepts line ids, never coordinates (see tools/ocr_tools.py).

Only the standard library and Pillow are used, so this module can be run anywhere
tesseract is installed. tesseract itself is run as a command, once, on a copy of the image
enlarged 2x with page-segmentation mode 4 (a column of text of varying sizes): on an
invoice with small mixed Vietnamese/English text that found all of the 15 fields tried,
against 10 to 14 for the other modes and scales compared.
"""

import csv
import difflib
import functools
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

LANG = "vie+eng"
PSM = "4"
MAX_OCR_SIDE = 4000          # the enlarged copy's longest side is kept under this
TIMEOUT_SECONDS = 120
CELL_SHORT_CONF = 85        # ... and a cell with only a character or two must be recognised at least this surely
CELL_MIN_CONF = 55          # a word read inside a table cell must be recognised at least this surely
MATCH_THRESHOLD = 0.72       # how alike (0..1) wanted text and the recognised words must be (digits must match exactly regardless)

Box = tuple[int, int, int, int]


class OcrError(Exception):
    """OCR could not be done; the message is written for the model."""


@dataclass
class Word:
    text: str
    box: Box
    conf: float = 100.0       # how surely tesseract recognised it (0 to 100); not kept in the saved result


@dataclass
class Line:
    id: int
    text: str
    box: Box
    words: list[Word] = field(default_factory=list)


@dataclass
class Cell:
    r: int                    # row and column, counted from 1
    c: int
    box: Box                  # the cell's interior as ruled
    words: list[Word] = field(default_factory=list)
    rowspan: int = 1
    colspan: int = 1

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: (round((w.box[1] + w.box[3]) / 2 / 8), w.box[0])))

    @property
    def ink(self) -> Box:
        """Where the cell's text is (the cell's own box if it has none)."""
        return union([w.box for w in self.words]) if self.words else self.box


@dataclass
class Table:
    id: int
    box: Box
    rows: int
    cols: int
    cells: list[Cell]


def available() -> bool:
    return shutil.which("tesseract") is not None


@functools.lru_cache(maxsize=1)
def installed_langs() -> tuple[str, ...]:
    """The OCR languages installed on this machine (tesseract codes), without the script detector."""
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return ()
    return tuple(l.strip() for l in out.splitlines()[1:] if l.strip() and l.strip() != "osd")


def check_lang(lang: str) -> str:
    """`lang` ("jpn", "chi_sim+eng"...) if every part of it is installed, else an OcrError saying what is."""
    have = installed_langs()
    parts = [p for p in re.split(r"[+,\s]+", lang.strip()) if p]
    bad = [p for p in parts if have and p not in have]
    if not parts or bad:
        raise OcrError(f"OCR language {', '.join(bad) or repr(lang)} is not installed. Installed: {', '.join(have)}.")
    return "+".join(parts)


# what tesseract's script detector reports, and the language to read it with
_SCRIPT_LANG = {"Han": "chi_sim (or jpn for Japanese, chi_tra for traditional)", "Japanese": "jpn", "Hangul": "kor", "Korean": "kor",
                "Cyrillic": "rus", "Thai": "tha", "Arabic": "ara"}


def looks_unreliable(lines: list[Line]) -> bool:
    """Whether what was read looks like noise: almost nothing, or low confidence throughout."""
    words = [w for l in lines for w in l.words]
    return len(words) < 3 or sum(w.conf for w in words) / len(words) < 60


def script_hint(data: bytes) -> str | None:
    """If the picture is in a script other than Latin, the language to read it with; else None. Uses
    tesseract's script detection, which needs a few lines of text to say anything."""
    try:
        with tempfile.TemporaryDirectory(prefix="ocrosd-") as tmp:
            src = os.path.join(tmp, "in.png")
            Image.open(io.BytesIO(data)).convert("RGB").save(src)
            out = subprocess.run(["tesseract", src, "stdout", "--psm", "0"], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"Script:\s*(\w+)", out)
    return _SCRIPT_LANG.get(m.group(1)) if m else None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- reading --------------------------------------------------------------------------------

Keyed = list[tuple[tuple[int, int, int], Word]]


def read_words(tsv: str, scale: float, min_conf: float = 0) -> Keyed:
    """Every recognised word with tesseract's (block, paragraph, line) key, boxes divided by
    `scale` to be in the original image's pixels. Words recognised with less confidence than
    `min_conf` (0 to 100) are left out."""
    out: Keyed = []
    for r in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (r.get("text") or "").strip()
        if not text or float(r.get("conf") or -1) < max(0, min_conf):
            continue
        x0, y0 = int(r["left"]) / scale, int(r["top"]) / scale
        x1, y1 = x0 + int(r["width"]) / scale, y0 + int(r["height"]) / scale
        out.append(((int(r["block_num"]), int(r["par_num"]), int(r["line_num"])),
                    Word(text, (round(x0), round(y0), round(x1), round(y1)), float(r.get("conf") or 0))))
    return out


def make_lines(keyed: Keyed) -> list[Line]:
    """Lines (top to bottom, then left to right, numbered from 1) from keyed words."""
    groups: dict[tuple[int, int, int], list[Word]] = {}
    for key, word in keyed:
        groups.setdefault(key, []).append(word)
    raw = []
    for words in groups.values():
        words.sort(key=lambda w: w.box[0])
        raw.append((union([w.box for w in words]), " ".join(w.text for w in words), words))
    raw.sort(key=lambda t: (t[0][1], t[0][0]))
    return [Line(i, text, box, words) for i, (box, text, words) in enumerate(raw, 1)]


def parse_tsv(tsv: str, scale: float) -> list[Line]:
    return make_lines(read_words(tsv, scale))


def _inside(box: Box, p: tuple[float, float]) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def build_tables(grids: list, keyed: Keyed, read_cell=None) -> tuple[list[Table], Keyed]:
    """(the tables with their words in cells, the words left outside every table).

    Words whose centre is inside a table's box are taken out of the page's lines. Each cell's
    words then come from `read_cell(box)` (its own OCR, on the cell cropped out: much better than
    reading across the cells) when that is given, else from the page-wide words by the cell their
    centre is in. Tables that end up with no text are dropped."""
    from app.agents.tables import TOL, anchors

    tables: list[Table] = []
    taken: set[int] = set()
    for grid in grids:
        xs, ys = anchors(grid.cells)

        def index(v: int, at: list[int]) -> int:
            return min(range(len(at)), key=lambda i: abs(at[i] - v))

        cells = []
        for box in grid.cells:
            rowspan = sum(1 for y in ys if box[1] - TOL <= y <= box[3] - TOL)
            colspan = sum(1 for x in xs if box[0] - TOL <= x <= box[2] - TOL)
            cells.append(Cell(index(box[1], ys) + 1, index(box[0], xs) + 1, box, [], max(1, rowspan), max(1, colspan)))
        inside = [n for n, (_, w) in enumerate(keyed) if _inside(grid.box, _centre(w.box))]
        taken.update(inside)
        if read_cell is not None:
            for cell in cells:
                cell.words = read_cell(cell.box)
        else:
            for n in inside:
                word = keyed[n][1]
                px, py = _centre(word.box)
                where = [c for c in cells if _inside(c.box, (px, py))] or \
                        [min(cells, key=lambda c: abs(_centre(c.box)[0] - px) + abs(_centre(c.box)[1] - py))]
                where[0].words.append(word)
        if any(c.words for c in cells):
            tables.append(Table(0, grid.box, len(ys), len(xs), sorted(cells, key=lambda c: (c.r, c.c))))
    tables.sort(key=lambda t: (t.box[1], t.box[0]))
    for i, t in enumerate(tables, 1):
        t.id = i
    return tables, [kw for n, kw in enumerate(keyed) if n not in taken]


def _read_cell(img: Image.Image, box: Box, lang: str = LANG) -> list[Word]:
    """The words in one table cell, OCR'd on their own: the cell is cropped (inside its rules),
    enlarged and read as a single block. Empty-looking cells are not sent to tesseract."""
    import numpy as np

    x0, y0, x1, y1 = box
    inset = 3
    crop = img.crop((x0 + inset, y0 + inset, x1 - inset, y1 - inset)).convert("L")
    if crop.width < 6 or crop.height < 6:
        return []
    if float((np.array(crop) < 140).mean()) < 0.004:           # (almost) no ink: an empty cell
        return []
    scale = max(2, min(4, -(-70 // crop.height)))
    big = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    with tempfile.TemporaryDirectory(prefix="ocrcell-") as tmp:
        src = os.path.join(tmp, "c.png")
        big.save(src)
        found: list[Word] = []
        for psm in ("6", "7", "10"):          # a block; then one line; then one character (a lone "1" in a narrow cell)
            try:
                proc = subprocess.run(["tesseract", src, "stdout", "-l", lang, "--psm", psm, "tsv"], capture_output=True,
                                      text=True, timeout=TIMEOUT_SECONDS, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
            except subprocess.TimeoutExpired:
                return []
            if proc.returncode != 0:
                continue
            # a cell holding only a faint watermark or a stray rule comes out as low-confidence noise
            found = [w for _, w in read_words(proc.stdout, scale, min_conf=CELL_MIN_CONF) if re.search(r"\w", w.text)]
            # one or two characters is what a stray stroke of a watermark looks like too: believe it only if sure
            if found and sum(len(re.findall(r"\w", w.text)) for w in found) <= 2:
                found = [w for w in found if w.conf >= CELL_SHORT_CONF]
            if found:
                break
    return [Word(w.text, (w.box[0] + x0 + inset, w.box[1] + y0 + inset, w.box[2] + x0 + inset, w.box[3] + y0 + inset))
            for w in found]


def engine_name(engine: str | None = None) -> str:
    """"paddle" or "tesseract": what was asked for (argument or OCR_ENGINE), else PaddleOCR when it is installed."""
    from app.agents import ocr_paddle

    want = (engine or os.environ.get("OCR_ENGINE") or "auto").lower()
    if want == "tesseract":
        return "tesseract"
    if want in ("paddle", "auto") and ocr_paddle.available():
        return "paddle"
    return "tesseract"


def _open_rgb(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001 — Pillow raises many types for a bad file
        raise OcrError(f"could not be opened as an image ({type(e).__name__}).") from e
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.split()[3])
        return flat
    return img.convert("RGB")


def run_ocr(data: bytes, lang: str | None = None, psm: str | None = None, scale: int | None = None,
            engine: str | None = None) -> tuple[tuple[int, int], list[Line], list[Table]]:
    """((width, height) of the image, the lines of text outside tables, the tables). Blocking:
    call from a thread. PaddleOCR reads it when installed (see ocr_paddle.py), else tesseract; if
    PaddleOCR fails on a picture, tesseract is tried."""
    if engine_name(engine) == "paddle":
        try:
            return _run_paddle(data, lang)
        except OcrError:
            raise
        except Exception:  # noqa: BLE001 — a model that would not load or run: fall back
            if not available():
                raise OcrError("OCR failed (PaddleOCR could not read this picture, and tesseract is not installed).") from None
    return _run_tesseract(data, lang, psm, scale)


def _run_paddle(data: bytes, lang: str | None) -> tuple[tuple[int, int], list[Line], list[Table]]:
    from app.agents import ocr_paddle

    lang = (lang or LANG).strip()
    bad = [p for p in re.split(r"[+,\s]+", lang) if p and p not in ocr_paddle.LANGS]
    if bad:
        raise OcrError(f"OCR language {', '.join(bad)} is not supported. Supported: {', '.join(ocr_paddle.LANGS)}.")
    img = _open_rgb(data)
    size = img.size
    keyed: Keyed = []
    for n, (_, score, words) in enumerate(ocr_paddle.read(img, lang)):
        for text, box in words:
            keyed.append(((n, 0, 0), Word(text, box, score * 100)))
    try:
        from app.agents.tables import find_grids

        grids = find_grids(img)
        tables, keyed = build_tables(grids, keyed) if grids else ([], keyed)      # words by the cell they sit in
    except ImportError:
        tables = []
    return size, make_lines(keyed), tables


def _run_tesseract(data: bytes, lang: str | None, psm: str | None, scale: int | None) -> tuple[tuple[int, int], list[Line], list[Table]]:
    if not available():
        raise OcrError("OCR (tesseract) is not installed on the server.")
    lang = check_lang(lang or LANG)
    psm = psm or PSM
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001 — Pillow raises many types for a bad file
        raise OcrError(f"could not be opened as an image ({type(e).__name__}).") from e
    size = img.size
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.split()[3])
        img = flat
    else:
        img = img.convert("RGB")
    scale = scale or 2
    while scale > 1 and max(size) * scale > MAX_OCR_SIDE:
        scale -= 1
    big = img.resize((size[0] * scale, size[1] * scale), Image.LANCZOS) if scale > 1 else img
    with tempfile.TemporaryDirectory(prefix="ocr-") as tmp:
        src = os.path.join(tmp, "in.png")
        big.save(src)
        try:
            proc = subprocess.run(["tesseract", src, "stdout", "-l", lang, "--psm", psm, "tsv"],
                                  capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                                  env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        except subprocess.TimeoutExpired:
            raise OcrError(f"OCR took longer than {TIMEOUT_SECONDS} seconds and was stopped.") from None
    if proc.returncode != 0:
        raise OcrError(f"OCR failed: {proc.stderr.strip()[:200] or 'tesseract exited with an error'}")
    keyed = read_words(proc.stdout, scale)
    try:
        from app.agents.tables import find_grids

        from concurrent.futures import ThreadPoolExecutor

        grids = find_grids(img)
        with ThreadPoolExecutor(max_workers=4) as pool:                # one tesseract per cell, a few at a time
            cache: dict[Box, list[Word]] = {}
            for grid in grids:
                for box, words in zip(grid.cells, pool.map(lambda b: _read_cell(img, b, lang), grid.cells)):
                    cache[box] = words
        tables, keyed = build_tables(grids, keyed, read_cell=lambda box: cache.get(box, []))
    except ImportError:        # OpenCV not installed here: read everything as lines
        tables = []
    return size, make_lines(keyed), tables


# ---- matching text to words -------------------------------------------------------------------

def _ascii_digits(s: str) -> str:
    """Digits of any script (Arabic-Indic, Thai, full-width) as 0-9, so numbers compare across scripts."""
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in s)


def norm(s: str) -> str:
    """The form in which OCR output and wanted text are compared: case folded, digits as 0-9, letters
    and digits of every script kept, accents dropped from Latin letters only (OCR most often gets those
    wrong, and "Hà Nội" must match "Ha Noi"), everything else a space. Accents on other scripts (Thai
    vowel marks, Arabic) are part of the letter and stay."""
    s = _ascii_digits(unicodedata.normalize("NFKC", s).casefold().replace("đ", "d"))
    out: list[str] = []
    prev_latin = False
    for ch in unicodedata.normalize("NFD", s):
        cat = unicodedata.category(ch)
        if cat == "Mn":
            if not prev_latin:
                out.append(ch)
            continue
        if cat[0] in "LN" or cat == "Mc":
            out.append(ch)
            prev_latin = cat[0] == "L" and unicodedata.name(ch, "").startswith("LATIN")
        else:
            out.append(" ")
            prev_latin = False
    return re.sub(r"\s+", " ", "".join(out)).strip()


def union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


# In a word that is mostly digits, the letters OCR mistakes for them are put back.
_CONFUSABLE = str.maketrans({"o": "0", "d": "0", "q": "0", "l": "1", "i": "1", "s": "5", "b": "8", "z": "2", "g": "9"})


def _read_digits(word: str) -> str:
    """Put back the digits OCR read as look-alike letters, but only in a word made of nothing else than
    digits and those look-alikes ("1O5" is 105; "INV-4714" has other letters and is left alone)."""
    alnum = [c for c in word if c.isalnum()]
    digits = sum(c.isdecimal() for c in alnum)
    if digits >= 2 and digits / len(alnum) >= 0.5 and all(c.isdecimal() or ord(c) in _CONFUSABLE for c in alnum):
        return word.translate(_CONFUSABLE)
    return word


def _digits(s: str) -> str:
    return "".join(c for c in s if c in "0123456789")


def locate(line: Line, text: str) -> tuple[Box, str, float] | None:
    """(box, the recognised words, how alike 0..1) for the stretch of this line's words that matches
    `text`, or None.

    Digits are never matched loosely: if the wanted text has digits, the stretch must have exactly
    those digits in that order, so "1.230.000" is not taken for "1.320.000" and a wrong number is not
    boxed. A wanted text that is mostly digits (a date as 02/04/2026, an account number) is found as the
    shortest run of words whose digits add up to it, so printed words between them ("02 tháng 04 năm
    2026", "1404 1485 10 25563") do not matter. Anything else is matched as approximate text, by
    characters rather than words, so it works in scripts that do not put spaces between words, and
    extra words inside the stretch (a bilingual caption such as "(month)") are tolerated."""
    want = norm(text).replace(" ", "")
    if not want or not line.words:
        return None
    hay_words = [_read_digits(norm(w.text).replace(" ", "")) for w in line.words]       # accents off first: "0Š" is "0s", then 05
    wd = _digits(want)
    if len(wd) >= 3 and len(wd) >= 0.5 * len(want):
        return _locate_number(line, hay_words, wd)
    return _locate_text(line, hay_words, want, wd)


def _locate_number(line: Line, hay_words: list[str], wd: str) -> tuple[Box, str, float] | None:
    best: tuple[float, int, int] | None = None
    for i, w in enumerate(hay_words):
        if not _digits(w):
            continue
        acc = ""
        for j in range(i, min(len(hay_words), i + 14)):
            acc += _digits(hay_words[j])
            if acc == wd:
                score = max(0.8, 1.0 - 0.03 * (j - i))
                if best is None or score > best[0]:
                    best = (score, i, j + 1)
                break
            if not wd.startswith(acc):
                break
    if best is None:
        return None
    score, i, j = best
    words = line.words[i:j]
    return union([w.box for w in words]), " ".join(w.text for w in words), round(score, 2)


def _locate_text(line: Line, hay_words: list[str], want: str, wd: str) -> tuple[Box, str, float] | None:
    owner = [wi for wi, hw in enumerate(hay_words) for _ in hw]
    hay = "".join(hay_words)
    if not hay:
        return None
    n = len(want)
    best = (0.0, 0, 0)

    def ratio(i: int, k: int) -> float:
        sub = hay[i:i + k]
        if wd and _digits(sub) != wd:
            return 0.0
        return difflib.SequenceMatcher(None, sub, want, autojunk=False).ratio()

    for k in range(max(1, n - max(1, n // 6)), min(len(hay), n + max(6, n // 2)) + 1):
        for i in range(0, len(hay) - k + 1):
            sm = difflib.SequenceMatcher(None, hay[i:i + k], want, autojunk=False)
            if sm.real_quick_ratio() <= best[0] or sm.quick_ratio() <= best[0]:
                continue
            r = ratio(i, k)
            if r > best[0] + 1e-9:
                best = (r, i, k)
    score, i, k = best
    if score < MATCH_THRESHOLD:
        return None
    improved = True                       # trim characters at either end that do not help the match
    while improved and k > 1:
        improved = False
        if ratio(i + 1, k - 1) >= score:
            i, k, improved = i + 1, k - 1, True
            score = ratio(i, k)
        elif ratio(i, k - 1) >= score:
            k, improved = k - 1, True
            score = ratio(i, k)
    covered: dict[int, int] = {}
    for pos in range(i, i + k):
        covered[owner[pos]] = covered.get(owner[pos], 0) + 1
    chosen = [wi for wi, c in covered.items() if c >= 0.5 * max(1, len(hay_words[wi]))] or [max(covered, key=covered.get)]
    first, last = min(chosen), max(chosen)
    words = line.words[first:last + 1]
    return union([w.box for w in words]), " ".join(w.text for w in words), round(score, 2)


# ---- drawing --------------------------------------------------------------------------------

def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _tag_spot(box: Box, size: tuple[int, int], obstacles: list[Box], bounds: tuple[int, int]) -> tuple[int, int]:
    """Top-left corner for a tag of `size` beside `box`: the first of a few spots (above-left, left,
    below-left, above-right, right) that stays inside the picture and covers no text. If every spot
    covers some, the one covering the least."""
    tw, th = size
    x0, y0, x1, y1 = box
    spots = [(x0, y0 - th), (x0 - tw, y0), (x0, y1), (x1 - tw, y0 - th), (x1, y0), (x0 - tw, y0 - th), (x1, y1)]
    best, best_cost = (max(0, x0), max(0, y0 - th)), None
    for sx, sy in spots:
        if sx < 0 or sy < 0 or sx + tw > bounds[0] or sy + th > bounds[1]:
            continue
        tag = (sx, sy, sx + tw, sy + th)
        cost = sum(1 for o in obstacles if _overlap(tag, o) and not _overlap(box, o))
        if cost == 0:
            return sx, sy
        if best_cost is None or cost < best_cost:
            best, best_cost = (sx, sy), cost
    return best



_PALETTE = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00897B", "#6D4C41", "#D81B60", "#3949AB", "#7CB342"]
_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]


def _font(size: int):
    for path in _FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


Field = tuple[str, Box, "Box | None"]       # (label, the value's box, the key's box if it was found)


def _paint(img: Image.Image, fields: list[Field], obstacles: list[Box] | None) -> None:
    """Draw the frames onto `img` (RGB or RGBA) in place: one colour per field, a heavy frame on its value and
    a lighter one on its key. A field whose key was not boxed has its label written beside the value instead,
    since the key is then nowhere on the picture to say what the value is."""
    w, h = img.size
    thickness = max(2, round(min(w, h) / 250))
    light = max(1, thickness - 1)
    tag = _font(max(11, round(min(w, h) / 55)))
    draw = ImageDraw.Draw(img)
    pad = max(1, thickness)
    for n, (label, box, key_box) in enumerate(fields):
        colour = _PALETTE[n % len(_PALETTE)]
        if key_box:
            kx0, ky0, kx1, ky1 = key_box
            draw.rectangle((kx0 - pad, ky0 - pad, kx1 + pad, ky1 + pad), outline=colour, width=light)
        x0, y0, x1, y1 = box
        draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=colour, width=thickness)
        if key_box:
            continue
        text = label if len(label) <= 28 else label[:27] + "…"
        tw, th = draw.textbbox((0, 0), text, font=tag)[2:]
        tx, ty = _tag_spot((x0 - pad, y0 - pad, x1 + pad, y1 + pad), (tw + 6, th + 4), obstacles or [], (w, h))
        draw.rectangle((tx, ty, tx + tw + 6, ty + th + 4), fill=colour)
        draw.text((tx + 3, ty + 1), text, fill="white", font=tag)


def draw_boxes(data: bytes, fields: list[Field], obstacles: list[Box] | None = None) -> bytes:
    """The image as a PNG, the same size, with a coloured frame on each field's value and, lighter and in the
    same colour, on its key. Nothing is added around the picture. Where a label has to be written (a field
    with no key frame) it is put beside its box where it covers none of `obstacles` (the boxes of all the
    recognised words), so it does not hide text."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    _paint(img, fields, obstacles)
    return _png(img)


def overlay(size: tuple[int, int], fields: list[Field], obstacles: list[Box] | None = None) -> Image.Image:
    """Only the frames and labels, on a transparent image of `size` (to lay over a PDF page)."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    _paint(img, fields, obstacles)
    return img


def _png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


# ---- the saved OCR result (kept in the sandbox beside the image) ------------------------------------

def to_json(name: str, digest: str, size: tuple[int, int], lines: list[Line], tables: list[Table] | None = None,
            source: dict | None = None) -> dict:
    """`source`, when the image is a page rendered from a PDF: {"pdf": its name, "pdf_sha256", "page", "pages"}."""
    return {"image": name, "sha256": digest, "size": list(size), **({"source": source} if source else {}),
            "lines": [{"id": l.id, "text": l.text, "box": list(l.box),
                       "words": [{"text": w.text, "box": list(w.box)} for w in l.words]} for l in lines],
            "tables": [{"id": t.id, "box": list(t.box), "rows": t.rows, "cols": t.cols,
                        "cells": [{"r": c.r, "c": c.c, "rowspan": c.rowspan, "colspan": c.colspan, "box": list(c.box), "text": c.text,
                                   "words": [{"text": w.text, "box": list(w.box)} for w in c.words]} for c in t.cells]}
                       for t in (tables or [])]}


def from_json(doc: dict) -> tuple[str, tuple[int, int], list[Line], list[Table]]:
    lines = [Line(l["id"], l["text"], tuple(l["box"]), [Word(w["text"], tuple(w["box"])) for w in l["words"]])
             for l in doc["lines"]]
    tables = [Table(t["id"], tuple(t["box"]), t["rows"], t["cols"],
                    [Cell(c["r"], c["c"], tuple(c["box"]), [Word(w["text"], tuple(w["box"])) for w in c["words"]],
                          c.get("rowspan", 1), c.get("colspan", 1)) for c in t["cells"]])
              for t in doc.get("tables", [])]
    return doc["sha256"], tuple(doc["size"]), lines, tables


# ---- naming a table, a row, a column or a cell ----------------------------------------------------------

_REF = re.compile(r"^T(\d+)(?:\.R(\d+))?(?:\.C(\d+))?$", re.I)


def resolve_cells(tables: list[Table], ref: str) -> list[Cell] | None:
    """The cells a reference like "T1", "T1.R2", "T1.C3" or "T1.R2.C3" names, or None if it names
    nothing (bad syntax, no such table, no such row or column)."""
    m = _REF.match(ref.strip())
    if not m:
        return None
    table = next((t for t in tables if t.id == int(m.group(1))), None)
    if table is None:
        return None
    r = int(m.group(2)) if m.group(2) else None
    c = int(m.group(3)) if m.group(3) else None
    cells = [x for x in table.cells if (r is None or x.r <= r < x.r + x.rowspan) and (c is None or x.c <= c < x.c + x.colspan)]
    return cells or None


def cell_line(cells: list[Cell]) -> Line:
    """The words of these cells as one Line, to search for a part of them with locate()."""
    words = [w for c in cells for w in c.words]
    box = union([w.box for w in words]) if words else union([c.box for c in cells])
    return Line(0, " ".join(w.text for w in words), box, words)
