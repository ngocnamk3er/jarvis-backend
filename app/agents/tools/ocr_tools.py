"""ocr_image and draw_boxes: marking things on an image from where OCR found the text.

The model that can see an image estimates pixel coordinates by eye and gets them wrong by tens
of pixels, so the two steps are tied together:

1. `ocr_image` reads the text and where each line sits, and saves that beside the image. Ruled
   tables are taken out first and read cell by cell, because reading across a table glues the
   cells of a row into one line and the columns get mixed up; they come back as separate tables
   (and as CSV files), and what is outside them as lines.
2. `draw_boxes` draws boxes, and only takes the ids of lines or table cells from step 1 (optionally
   with the part wanted). It has no argument for coordinates, and it refuses an image that has not
   been through `ocr_image`. A box is therefore always where the text really is.

The work runs on the server (app/agents/ocr.py); the sandbox only holds the files.
"""

import asyncio
import io
import json
import re
from collections import Counter
from pathlib import PurePosixPath

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel, Field

from app.agents import ocr, pdf_marks, vision
from app.agents.tools.sandbox_errors import read_error
from app.agents.tools.sandbox_manager import get_thread_id, normalize_workspace_path, read_file, write_file

MAX_LINES_SHOWN = 250
MAX_LINE_CHARS = 160
MAX_ITEMS = 60


class BoxItem(BaseModel):
    label: str = Field(description="The NAME of the field, e.g. 'Invoice number' or 'Total' (it is written on the picture only "
                                    "when the key is not boxed). Not its value: a value typed here by hand "
                                    "ends up on the picture, wrong if you mistyped it.")
    lines: list[int] = Field(default_factory=list,
                             description="Ids (from ocr_image) of text lines outside tables that hold it. "
                                         "More than one id boxes them together as a single block.")
    cells: list[str] = Field(default_factory=list,
                             description="Table references from ocr_image: a cell 'T1.R3.C2', a whole row 'T1.R3', "
                                         "a whole column 'T1.C2', or the whole table 'T1'.")
    text: str | None = Field(default=None, description="Box only this part of those lines/cells, as printed. For a field with "
                                                         "a key and a value on one line, pass the VALUE here (e.g. '4821') "
                                                         "so the box is on the value and not on the whole line. Diacritics and "
                                                         "OCR slips do not matter. Omit it to box everything in them.")
    key: str | None = Field(default=None, description="The key (the printed name of the field) as it appears on the picture, "
                                                       "e.g. 'Invoice No.:'. It gets a box of its own, separate from the value's and "
                                                       "in the same colour. It is looked for in those lines/cells first and then "
                                                       "anywhere on the picture, nearest to the value. Give it whenever the field "
                                                       "has a printed name.")


def _path(path: str) -> tuple[str | None, str | None]:
    name = normalize_workspace_path(path)
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        return None, "Error: path must be inside your working directory (no '..', no absolute path)."
    return name, None


def _ocr_name(name: str) -> str:
    return f"{name}.ocr.json"


def _csv_name(name: str, table_id: int) -> str:
    return f"{name}.T{table_id}.csv"


def _csv(table: ocr.Table) -> bytes:
    """The table as CSV (UTF-8 with a BOM so Excel opens it right), a merged cell's text in its top-left cell."""
    import csv

    grid = [["" for _ in range(table.cols)] for _ in range(table.rows)]
    for c in table.cells:
        if 1 <= c.r <= table.rows and 1 <= c.c <= table.cols:
            grid[c.r - 1][c.c - 1] = c.text
    buf = io.StringIO()
    csv.writer(buf).writerows(grid)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def _line(l: ocr.Line) -> str:
    text = l.text if len(l.text) <= MAX_LINE_CHARS else l.text[:MAX_LINE_CHARS] + "…"
    return f"L{l.id} [{l.box[0]},{l.box[1]},{l.box[2]},{l.box[3]}] {text}"


@tool
async def ocr_image(path: str, label: str, config: RunnableConfig, lang: str | None = None,
                    page: int | None = None) -> str:
    """Read the text in an image in your bash sandbox, with where each line sits (pixel boxes).

    Do this BEFORE marking, boxing or locating anything on an image: positions you estimate by
    eye are wrong by tens of pixels, OCR positions are exact. You get the lines as
    `L<id> [x0,y0,x1,y1] text`; then call `draw_boxes` with the ids of the lines you want
    marked. Ruled tables are read cell by cell and come back as separate tables. The full
    result, with a box for every word, is also saved as `<name>.ocr.json` beside the image.

    Use this for POSITIONS. The recognised text has mistakes (accents dropped, a digit lost,
    small print misread), so read the values you will tell the user from the picture itself with
    `view_image`, quote them as printed, and do not expand abbreviations.

    Also works on ONE PAGE of a PDF (a scan or a text PDF alike): pass the PDF's path and `page`.
    The page is rendered to `<name>.p<page>.png` beside the PDF and read from there; use THAT png's
    path with `draw_boxes`, which then also marks the PDF itself (`<name>_boxes.pdf`). Read a page
    at a time; to find out which page holds something, use `read_pdf` or `check_pdf`.

    The language to read defaults to Vietnamese + English. For text in another language pass
    `lang` as codes joined by "+" (for example "jpn", "chi_sim", "chi_tra", "kor", "rus", "tha", "ara",
    "fra", "deu", "spa"; the error message lists what is supported). One script family is read at a
    time, so for a document in two (Japanese and Russian, say) call it once for each. If the result
    looks like noise the reply says which language the picture seems to be in.

    Args:
        path: The image's path in your working directory, e.g. "invoice.png", or a PDF's, e.g. "report.pdf".
        label: Brief human-readable description shown to the user (e.g. "Reading the invoice's text").
        page: For a PDF: the page to read (1 = the first). Default 1. Not for images.
        lang: Language(s) of the text, e.g. "jpn" or "rus+eng". Default "vie+eng".
    """
    name, err = _path(path)
    if err:
        return err
    thread_id = get_thread_id(config)
    try:
        data, _, filename = await read_file(thread_id, name)
    except Exception as e:  # noqa: BLE001
        return read_error(name, e)
    if len(data) > vision.MAX_INPUT_BYTES:
        return f"Error: {filename} is {len(data) >> 20} MB, over the {vision.MAX_INPUT_BYTES >> 20} MB limit."
    source = None
    page_note = ""
    if name.lower().endswith(".pdf"):
        number = page or 1
        try:
            png, total, _ = await asyncio.to_thread(pdf_marks.render_page, data, filename, number)
        except pdf_marks.PdfError as e:
            return f"Error: {e}"
        pdf_name = name
        name = f"{str(PurePosixPath(pdf_name).with_suffix(''))}.p{number}.png"
        source = {"pdf": pdf_name, "pdf_sha256": ocr.sha256(data), "page": number, "pages": total}
        try:
            await write_file(thread_id, name, png)
        except Exception as e:  # noqa: BLE001
            return f"Error: could not save the rendered page into the sandbox ({type(e).__name__}: {e})."
        data, filename = png, f"{filename} page {number}"
        page_note = f" Page {number} of {total} was rendered to {name}: use {name} (not the PDF) as the path in draw_boxes."
    elif page is not None:
        return "Error: `page` is only for PDFs."
    try:
        size, lines, tables = await asyncio.to_thread(ocr.run_ocr, data, lang)
    except ocr.OcrError as e:
        return f"Error: {filename} {e}" if "opened" in str(e) else f"Error: {e}"
    if not lines and not tables:
        return f"No text was recognised in {filename} ({size[0]}x{size[1]}). It may contain no text, or too little contrast."
    doc = ocr.to_json(name, ocr.sha256(data), size, lines, tables, source)
    try:
        await write_file(thread_id, _ocr_name(name), json.dumps(doc, ensure_ascii=False).encode("utf-8"))
        for t in tables:
            await write_file(thread_id, _csv_name(name, t.id), _csv(t))
    except Exception as e:  # noqa: BLE001
        return f"Error: could not save the OCR result into the sandbox ({type(e).__name__}: {e})."
    hint = None
    if ocr.looks_unreliable(lines):
        hint = await asyncio.to_thread(ocr.script_hint, data)
    shown = lines[:MAX_LINES_SHOWN]
    head = f"OCR of {filename} ({size[0]}x{size[1]} px): {len(lines)} line{'s' if len(lines) != 1 else ''} of text"
    head += f" outside tables and {len(tables)} table{'s' if len(tables) != 1 else ''}." if tables else "."
    out = [head + f" Boxes are in this image's pixels.{page_note} Full data with word boxes: {_ocr_name(name)}.",
           "Text outside tables, as L<id> [x0,y0,x1,y1] text:" if tables else "Lines, as L<id> [x0,y0,x1,y1] text:",
           *[_line(l) for l in shown]]
    if len(lines) > len(shown):
        out.append(f"... {len(lines) - len(shown)} more lines are in {_ocr_name(name)}.")
    for t in tables:
        out.append("")
        out.append(f"Table T{t.id} [{t.box[0]},{t.box[1]},{t.box[2]},{t.box[3]}], {t.rows} rows x {t.cols} columns, read cell by cell "
                   f"(cells are T{t.id}.R<row>.C<col>; a * marks a merged cell; empty cells are left out). Also saved as {_csv_name(name, t.id)}.")
        for r in range(1, t.rows + 1):
            cells = [c for c in t.cells if c.r == r and c.text]
            if cells:
                out.append(f"  R{r}: " + " | ".join(f"C{c.c}{'*' if c.rowspan > 1 or c.colspan > 1 else ''} {c.text[:MAX_LINE_CHARS]!r}" for c in cells))
    if hint:
        out.append("")
        out.append(f"Note: this picture looks like it is in another script, and what is above was read as {lang or ocr.LANG}, so it is unreliable. "
                   f"Call ocr_image again with lang={hint!r}.")
    out.append("")
    out.append("This text is for finding POSITIONS: it has mistakes (dropped diacritics, lost digits). Take the values you report from the "
               "picture itself (view_image) and quote them as printed. To mark things, call draw_boxes with line ids and/or table cell "
               "references from this list; pass `text` (the value) to box just the value. Do not write coordinates yourself. Table cells "
               "keep their column: read a row's values from the table, not from the lines.")
    return "\n".join(out)


_PDF_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}      # one per (conversation, pdf)


def _pdf_out_name(pdf: str) -> str:
    return f"{str(PurePosixPath(pdf).with_suffix(''))}_boxes.pdf"


async def _mark_pdf(thread_id: str, source: dict, size: list[int], boxes: list[dict], obstacles: list[ocr.Box]) -> tuple[bool, str]:
    """Lay the boxes of this page over the PDF and save `<pdf>_boxes.pdf`, keeping the boxes of the other
    pages marked before. Returns (done, a line for the reply); it never raises: the PNG is already saved."""
    pdf, number = source["pdf"], source["page"]
    out_pdf = _pdf_out_name(pdf)
    state_name = str(PurePosixPath(out_pdf).with_suffix(".json"))
    # A model often marks several pages in parallel calls; each reads the pages marked so far,
    # adds its own and writes the PDF back, so without this one call's page would overwrite another's.
    async with _PDF_LOCKS.setdefault((thread_id, pdf), asyncio.Lock()):
        return await _mark_pdf_locked(thread_id, source, size, boxes, obstacles, pdf, number, out_pdf, state_name)


async def _mark_pdf_locked(thread_id: str, source: dict, size: list[int], boxes: list[dict], obstacles: list[ocr.Box],
                           pdf: str, number: int, out_pdf: str, state_name: str) -> tuple[bool, str]:
    try:
        original, _, _ = await read_file(thread_id, pdf)
        if ocr.sha256(original) != source["pdf_sha256"]:
            return False, f"The PDF {pdf} has changed since this page was read, so it was not marked; run ocr_image on the page again."
        pages: dict[str, dict] = {}
        try:
            raw, _, _ = await read_file(thread_id, state_name)
            prev = json.loads(raw)
            if prev.get("pdf_sha256") == source["pdf_sha256"]:
                pages = prev.get("pages", {})
        except Exception:  # noqa: BLE001 — nothing marked yet
            pages = {}
        pages[str(number)] = {"size": size, "boxes": [[r["label"], r["box"], r.get("key_box")] for r in boxes],
                              "obstacles": [list(o) for o in obstacles]}
        marked = await asyncio.to_thread(pdf_marks.annotate, original, pdf, {int(k): v for k, v in pages.items()})
        await write_file(thread_id, out_pdf, marked)
        await write_file(thread_id, state_name, json.dumps({"pdf_sha256": source["pdf_sha256"], "pages": pages}).encode("utf-8"))
    except pdf_marks.PdfError as e:
        return False, f"The PDF itself could not be marked: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"The PDF itself could not be marked ({type(e).__name__}: {e}); the page PNG is fine."
    marked_pages = ", ".join(str(k) for k in sorted(int(k) for k in pages))
    return True, f"Also marked on the PDF: {out_pdf} (pages marked so far: {marked_pages})."


@tool(response_format="content_and_artifact")
async def draw_boxes(path: str, items: list[BoxItem], label: str, config: RunnableConfig,
                     output: str | None = None, replace: bool = False) -> tuple[str, dict | None]:
    """Draw boxes on an image around text that `ocr_image` has located.

    Give each field a `label`, the OCR line ids and/or table cells that hold it, the `text` of its VALUE
    (to box just that part of them) and the `key` (its printed name). The value and the key each get a
    box of their own, in the same colour: heavy on the value, light on the key. There is deliberately
    no way to pass coordinates: the boxes come from the OCR result, so they sit exactly on the text.
    The image must have been through `ocr_image` first. The result is a new image of the same size
    (the original is untouched; nothing is added around it), plus a `.json` file listing every box.

    Boxes ADD UP: a later call on the same image keeps the boxes of the earlier calls (a box with
    the same label replaces its earlier one), so to fix or add a few boxes, send just those. Pass
    `replace=true` to start the image over.

    Anything that could not be placed (an unknown line id, text not found in that line) is
    listed in the reply as not drawn: tell the user about those, do not leave them out silently.
    For a page of a PDF (the `<name>.p<page>.png` that `ocr_image` made), the boxes are also laid
    over the PDF itself, in `<name>_boxes.pdf`: the whole document, the pages you marked carrying
    their boxes. Marking more pages adds to the same PDF.

    The finished image (the PDF, when the user gave you a PDF) is shown to the user under this
    step by itself: do not `present_file` it. Check it with `view_image` once if it matters.

    Args:
        path: The image's path in your working directory, e.g. "invoice.png".
        items: The boxes to draw: [{"label": ..., "lines": [ids], "cells": ["T1.R3.C2"], "text": optional part}].
        output: Where to save the result; default `<name>_boxes.png` beside the image.
        replace: Start the image over instead of adding to the boxes already drawn on it.
        label: Brief human-readable description shown to the user (e.g. "Marking the key fields").
    """
    shown: list[str] = []
    text = await _draw_boxes(path, items, config, output, replace, shown)
    return text, await _files_artifact(get_thread_id(config), shown)


def _field(r: dict) -> ocr.Field:
    return (r["label"], tuple(r["box"]), tuple(r["key_box"]) if r.get("key_box") else None)


def _gap(a: ocr.Box, b: ocr.Box) -> float:
    """Distance between the centres of two boxes."""
    (ax, ay), (bx, by) = ocr._centre(a), ocr._centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _find_key(key: str, value: ocr.Box, lines: list[ocr.Line], tables: list[ocr.Table]) -> tuple[ocr.Box, str, float] | None:
    """Where the key text sits on the picture, as (box, what OCR read, match 0..1), or None. A key such as
    'Company name' can be printed more than once (seller, buyer): of the places that match about as well as the
    best one, the one nearest to the value is taken. Places overlapping the value itself are not the key."""
    want = ocr.norm(key).replace(" ", "")
    if not want:
        return None
    spans = list(lines) + [ocr.cell_line([c]) for t in tables for c in t.cells]
    found: list[tuple[ocr.Box, str, float]] = []
    for line in spans:
        hay = "".join(ocr.norm(w.text).replace(" ", "") for w in line.words)
        if sum((Counter(want) & Counter(hay)).values()) < 0.6 * len(want):          # cheap check before the matching
            continue
        hit = ocr.locate(line, key)
        if not hit:
            continue
        box = hit[0]
        ix = min(box[2], value[2]) - max(box[0], value[0])
        iy = min(box[3], value[3]) - max(box[1], value[1])
        if ix > 0 and iy > 0 and ix * iy > 0.5 * min((box[2] - box[0]) * (box[3] - box[1]), (value[2] - value[0]) * (value[3] - value[1])):
            continue
        found.append(hit)
    if not found:
        return None
    top = max(f[2] for f in found)
    return min((f for f in found if f[2] >= top - 0.1), key=lambda f: _gap(f[0], value))


async def _files_artifact(thread_id: str, names: list[str]) -> dict | None:
    """The files to show the user under the step (the same description `present_file` gives), or None.
    The chat turns these into download chips, so the result is on screen even if the model never says so."""
    files = []
    for name in names:
        try:
            content, mime, filename = await read_file(thread_id, name)
        except Exception:  # noqa: BLE001 — the text result still tells the model what happened
            continue
        files.append({"name": filename, "mime": mime, "size": len(content), "path": name, "thread_id": thread_id})
    return {"files": files} if files else None


async def _draw_boxes(path: str, items: list[BoxItem], config: RunnableConfig, output: str | None,
                      replace: bool, shown: list[str]) -> str:
    """The work of `draw_boxes`; appends the name of the file to show the user to `shown`."""
    name, err = _path(path)
    if err:
        return err
    if not items:
        return "Error: no boxes asked for."
    if len(items) > MAX_ITEMS:
        return f"Error: {len(items)} boxes asked for; at most {MAX_ITEMS} at a time."
    stem = str(PurePosixPath(name).with_suffix(""))
    out_name = normalize_workspace_path(output) if output else f"{stem}_boxes.png"
    if out_name.startswith("/") or ".." in PurePosixPath(out_name).parts:
        return "Error: output must be inside your working directory (no '..', no absolute path)."
    if not out_name.lower().endswith(".png"):
        out_name += ".png"
    if out_name == name:
        return "Error: output must be a different file from the image, which is left untouched."

    thread_id = get_thread_id(config)
    try:
        data, _, filename = await read_file(thread_id, name)
    except Exception as e:  # noqa: BLE001
        return read_error(name, e)
    try:
        raw, _, _ = await read_file(thread_id, _ocr_name(name))
        ocr_doc = json.loads(raw)
        digest, _, lines, tables = ocr.from_json(ocr_doc)
    except Exception:  # noqa: BLE001 — missing, or not something we wrote
        return (f"Error: there is no OCR result for {filename}. Call ocr_image on it first: boxes can only be "
                f"drawn on text that OCR has located, never at coordinates you choose.")
    if digest != ocr.sha256(data):
        return f"Error: {filename} has changed since it was read by OCR. Call ocr_image on it again."
    by_id = {l.id: l for l in lines}

    drawn: list[tuple[str, ocr.Box]] = []
    record: list[dict] = []
    not_drawn: list[str] = []
    notes: list[str] = []
    for item in items:
        if not item.lines and not item.cells:
            not_drawn.append(f"{item.label}: no line ids or table cells given")
            continue
        missing = [f"L{i}" for i in item.lines if i not in by_id]
        for ref in item.cells:
            if ocr.resolve_cells(tables, ref) is None:
                missing.append(ref)
        if missing:
            not_drawn.append(f"{item.label}: no OCR {'line or cell' if item.cells else 'line'} {', '.join(missing)}")
            continue
        picked = [by_id[i] for i in item.lines]
        # what can be searched for `text` (a Line to search), where each part is, and what it reads
        spans = [(l, l.box, l.text) for l in picked]
        for ref in item.cells:
            cells = ocr.resolve_cells(tables, ref) or []
            if re.fullmatch(r"T\d+", ref.strip(), re.I):          # the whole table: its own outline, not the text inside
                table = next(t for t in tables if t.id == int(ref.strip()[1:]))
                spans.append((ocr.cell_line(cells), table.box, ""))
            else:
                spans += [(ocr.cell_line([c]), c.ink, c.text) for c in cells]
        if item.text:
            found = [(ocr.locate(line, item.text), line) for line, _, _ in spans]
            found = [(f, l) for f, l in found if f]
            if not found and item.cells and not item.lines and all(not t for _, _, t in spans):
                # a cell named outright that OCR read nothing in (a lone digit, say): box the cell itself
                box, matched, score = ocr.union([b for _, b, _ in spans]), "", 0.0
                notes.append(f"{item.label}: OCR read no text in {', '.join(item.cells)}, so the cell itself is boxed")
            elif not found:
                reads = "; ".join(f"{t[:70]!r}" for _, _, t in spans)
                not_drawn.append(f"{item.label}: the text {item.text!r} was not found in those lines/cells (they read {reads})")
                continue
            else:
                (box, matched, score), _line_found = max(found, key=lambda t: t[0][2])
        else:
            box, matched, score = ocr.union([b for _, b, _ in spans]), " ".join(t for _, _, t in spans if t), 1.0
        key_box, key_read = None, ""
        if item.key:
            hit = _find_key(item.key, box, lines, tables)
            if hit:
                key_box, key_read, _ = hit
            else:
                notes.append(f"{item.label}: the key {item.key!r} was not found on the picture, so only the value is boxed")
        drawn.append((item.label, box))
        record.append({"label": item.label, "box": list(box), "ocr_text": matched, "match": score,
                       "key_box": list(key_box) if key_box else None, "key_ocr_text": key_read,
                       "lines": list(item.lines), "cells": list(item.cells)})
    if not drawn:
        return "Error: nothing could be drawn.\nNot drawn:\n" + "\n".join(f"- {n}" for n in not_drawn)

    json_name = str(PurePosixPath(out_name).with_suffix(".json"))
    previous: list[dict] = []
    if not replace:                                   # boxes add up across calls on the same, unchanged image
        try:
            prev_raw, _, _ = await read_file(thread_id, json_name)
            prev = json.loads(prev_raw)
            if prev.get("sha256") == digest and prev.get("image") == name:
                previous = prev.get("boxes", [])
        except Exception:  # noqa: BLE001 — no earlier result: start fresh
            previous = []
    new_labels = {r["label"] for r in record}
    kept = [r for r in previous if r["label"] not in new_labels]
    merged = kept + record
    try:
        obstacles = [w.box for l in lines for w in l.words] + [w.box for t in tables for c in t.cells for w in c.words]   # keep the number tags off the text
        png = await asyncio.to_thread(ocr.draw_boxes, data, [_field(r) for r in merged], obstacles)
        await write_file(thread_id, out_name, png)
        await write_file(thread_id, json_name, json.dumps({"image": name, "sha256": digest, "output": out_name, "boxes": merged},
                                                          ensure_ascii=False, indent=1).encode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"Error: could not write the result into the sandbox ({type(e).__name__}: {e})."
    pdf_note, pdf_done = None, False
    source = ocr_doc.get("source")
    if source:
        pdf_done, pdf_note = await _mark_pdf(thread_id, source, ocr_doc["size"], merged, obstacles)
    with Image.open(io.BytesIO(png)) as im:
        dims = f"{im.width}x{im.height}"
    head = f"Drew {len(drawn)} box{'es' if len(drawn) != 1 else ''} on {filename}"
    if kept:
        head += f", added to the {len(kept)} already there ({len(merged)} in all)"
    lines_out = [f"{head}; saved as {out_name} ({dims}) and listed in {json_name}."]
    for r in record:
        box = ",".join(str(v) for v in r["box"])
        key = f"; key [{','.join(str(v) for v in r['key_box'])}] reads {r['key_ocr_text'][:40]!r}" if r.get("key_box") else ""
        lines_out.append(f"- {r['label']}: value [{box}] reads {r['ocr_text'][:60]!r}{key}")
    if notes:
        lines_out += ["Note:", *[f"- {n}" for n in notes]]
    if not_drawn:
        lines_out += ["Not drawn:", *[f"- {n}" for n in not_drawn],
                      "If a text that is in the picture was not found, it may have been read in the wrong language: run ocr_image again with the right "
                      "`lang`, then draw those boxes again (the boxes already drawn are kept)."]
    if pdf_note:
        lines_out.append(pdf_note)
    shown_file = _pdf_out_name(source["pdf"]) if pdf_done else out_name
    shown.append(shown_file)
    lines_out.append(f"{shown_file} is shown to the user under this step; do not present_file it"
                     + (", and tell them which boxes could not be drawn." if not_drawn else "."))
    return "\n".join(lines_out)
