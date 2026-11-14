"""read_pdf: the text lands in a file beside the PDF; the model gets a summary."""

import importlib

import pytest

from app.agents.tools.read_pdf import read_pdf
from tests.test_check_pdf import build_pdf

mod = importlib.import_module("app.agents.tools.read_pdf")


@pytest.fixture
def sandbox(monkeypatch):
    files: dict[str, bytes] = {}

    async def read_file(thread_id, name):
        if name not in files:
            raise FileNotFoundError(name)
        return files[name], "application/pdf", name

    async def write_file(thread_id, name, content):
        files[name] = content

    monkeypatch.setattr(mod, "read_file", read_file)
    monkeypatch.setattr(mod, "write_file", write_file)
    return files


async def run(path):
    return await read_pdf.ainvoke({"path": path, "label": "x"}, config={"configurable": {"thread_id": "t"}})


async def test_the_text_goes_to_a_file_and_the_model_gets_a_summary(sandbox):
    sandbox["docs/report.pdf"] = build_pdf([("text", 30), ("text", 25)])
    out = await run("docs/report.pdf")
    written = sandbox["docs/report.txt"].decode()
    assert "=== page 1 ===" in written and "=== page 2 ===" in written and "Invoice number" in written
    assert "Text of 2 pages written to docs/report.txt" in out
    assert "Not read" not in out
    assert "grep -n" in out


async def test_a_big_pdf_costs_the_model_a_preview_not_the_whole_text(sandbox):
    sandbox["big.pdf"] = build_pdf([("text", 140)] * 40)
    out = await run("big.pdf")
    assert len(sandbox["big.txt"]) > 10_000
    assert len(out) < 3_000 and "(rest is in the file)" in out


async def test_scanned_pages_are_marked_and_listed(sandbox):
    sandbox["doc.pdf"] = build_pdf([("text", 30), ("image", 0), ("image", 0), ("text", 30)])
    out = await run("doc.pdf")
    written = sandbox["doc.txt"].decode()
    assert "=== page 2 === [scanned page, no text]" in written
    assert "=== page 3 === [scanned page, no text]" in written
    assert "Text of 2 pages written" in out and "pages 2-3" in out


async def test_a_fully_scanned_pdf_says_nothing_was_read(sandbox):
    sandbox["scan.pdf"] = build_pdf([("image", 0), ("image", 0)])
    out = await run("scan.pdf")
    assert "Text of 0 pages" in out and "pages 1-2" in out


async def test_a_name_without_pdf_extension_still_gets_a_txt_beside_it(sandbox):
    sandbox["upload"] = build_pdf([("text", 30)])
    await run("upload")
    assert "upload.txt" in sandbox


async def test_errors_are_messages_not_crashes(sandbox):
    sandbox["notes.pdf"] = b"plain text"
    sandbox["bad.pdf"] = b"%PDF-1.4\n garbage garbage"
    assert "is not a PDF" in await run("notes.pdf")
    assert "could not be read as a PDF" in await run("bad.pdf")
    assert "nope.pdf is not in the sandbox" in await run("nope.pdf")
    assert "inside your working directory" in await run("/etc/x.pdf")
