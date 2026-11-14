"""check_pdf: telling text pages from scanned ones. The PDFs are built by hand in
the test, so nothing depends on a sample file."""

import importlib

import pytest

from app.agents.tools.check_pdf import check_pdf

mod = importlib.import_module("app.agents.tools.check_pdf")  # the attribute above is the tool

WORDS = ("Invoice number INV-2026-0417 total due 1,250.00 USD payment due by 30 November 2026 "
         "customer Acme Trading Company Limited thank you for your business").split()


def build_pdf(pages: list[tuple[str, int]]) -> bytes:
    """pages: ("text", n_words) draws n words; ("image", n_words) draws a full-page
    picture and, on top, n_words of text; ("empty", 0) is a blank page."""
    objs: list[str | bytes] = [None, None]  # 1 catalog, 2 pages (filled in below)
    kids = []
    for kind, n in pages:
        page_no = len(objs) + 1
        content_no, font_no = page_no + 1, page_no + 2
        text = " ".join((WORDS * 5)[:n])
        draw = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET" if n else ""
        resources = f"/Font << /F1 {font_no} 0 R >>"
        if kind == "image":
            img_no = page_no + 3
            resources += f" /XObject << /Im1 {img_no} 0 R >>"
            draw = "q 612 0 0 792 0 0 cm /Im1 Do Q " + draw
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_no} 0 R "
                    f"/Resources << {resources} >> >>")
        objs.append(f"<< /Length {len(draw)} >>\nstream\n{draw}\nendstream")
        objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        if kind == "image":
            raw = bytes([200, 200, 200]) * 16
            objs.append(b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceRGB "
                        b"/BitsPerComponent 8 /Length 48 >>\nstream\n" + raw + b"\nendstream")
        kids.append(f"{page_no} 0 R")
    objs[0] = "<< /Type /Catalog /Pages 2 0 R >>"
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + (o if isinstance(o, bytes) else o.encode()) + b"\nendobj\n"
    x = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode()


def kinds(pages):
    total, infos = mod.analyze(build_pdf(pages))
    assert total == len(pages)
    return [p.kind for p in infos]


def test_a_text_pdf_is_text():
    assert kinds([("text", 30), ("text", 25)]) == ["text", "text"]


def test_a_picture_only_page_is_a_scan():
    assert kinds([("image", 0)]) == ["scan"]


def test_a_stray_page_number_on_a_scan_is_still_a_scan():
    assert kinds([("image", 2)]) == ["scan"]   # under the word threshold


def test_a_picture_with_a_short_header_is_flagged_as_probably_scanned():
    assert kinds([("image", 12)]) == ["scan-with-text"]


def test_a_full_page_picture_under_a_long_text_layer_is_text():
    # an OCR'd scan keeps its picture and gains a real text layer: it can be read as text
    assert kinds([("image", 60)]) == ["text"]


def test_a_blank_page_counts_as_no_text():
    assert kinds([("empty", 0)]) == ["scan"]


def test_a_mixed_document_is_told_apart_page_by_page():
    assert kinds([("text", 30), ("image", 0), ("image", 0), ("text", 30)]) == ["text", "scan", "scan", "text"]


def test_unreadable_glyphs_make_a_text_layer_garbled():
    junk = "(cid:12)(cid:7)(cid:93) " * 30
    assert mod.bad_ratio(junk) > mod.GARBLED_RATIO
    assert mod.classify(words=30, cover=0.0, bad=mod.bad_ratio(junk)) == "garbled"
    assert mod.bad_ratio("An ordinary sentence with a café and 12.5% in it.") == 0


def test_the_description_names_page_ranges():
    total, infos = mod.analyze(build_pdf([("text", 30), ("image", 0), ("image", 0), ("text", 30)]))
    out = mod.describe("a.pdf", 1234, total, infos)
    assert "Verdict: mixed" in out
    assert "Pages 1, 4 have a real text layer" in out
    assert "Pages 2-3 have no text layer" in out


@pytest.fixture
def sandbox(monkeypatch):
    files: dict[str, bytes] = {}

    async def read_file(thread_id, name):
        if name not in files:
            raise FileNotFoundError(name)
        return files[name], "application/pdf", name

    monkeypatch.setattr(mod, "read_file", read_file)
    return files


async def run(path):
    return await check_pdf.ainvoke({"path": path, "label": "x"}, config={"configurable": {"thread_id": "t"}})


async def test_the_tool_reports_a_scanned_pdf(sandbox):
    sandbox["scan.pdf"] = build_pdf([("image", 0), ("image", 0)])
    out = await run("scan.pdf")
    assert "Verdict: scanned PDF" in out and "Pages 1-2 have no text layer" in out


async def test_the_tool_refuses_what_is_not_a_pdf(sandbox):
    sandbox["notes.pdf"] = b"just some text"
    assert "is not a PDF" in await run("notes.pdf")


async def test_the_tool_survives_a_damaged_pdf(sandbox):
    sandbox["bad.pdf"] = b"%PDF-1.4\n garbage garbage"
    assert "could not be read as a PDF" in await run("bad.pdf")


async def test_the_tool_reports_a_missing_file_and_a_bad_path(sandbox):
    assert "nope.pdf is not in the sandbox" in await run("nope.pdf")
    assert "inside your working directory" in await run("../etc/x.pdf")
