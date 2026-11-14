"""Office files through LibreOffice. These run the real converter, so they are
skipped on a machine without `soffice`."""

import importlib
import io
import zipfile

import pytest

from app.agents import office, vision
from app.agents.tools.office_tools import office_convert, office_recalculate

tools_mod = importlib.import_module("app.agents.tools.office_tools")
needs_soffice = pytest.mark.skipif(not office.available(), reason="LibreOffice is not installed")

RTF = rb"{\rtf1\ansi Quarterly review: revenue grew by twelve percent.\par Second paragraph about hiring.\par}"


def workbook(sheets: dict[str, list[tuple[str, str | None]]]) -> bytes:
    """A minimal .xlsx. Each cell is (formula, cached value) as a script would leave it: a wrong cache on purpose."""
    names = list(sheets)
    ct = ('<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                    for i, _ in enumerate(names, 1)) + "</Types>")
    wb = ('<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + "".join(f'<sheet name="{n}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(names, 1)) + "</sheets></workbook>")
    rels = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                      for i, _ in enumerate(names, 1)) + "</Relationships>")
    root = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", root)
        z.writestr("xl/workbook.xml", wb); z.writestr("xl/_rels/workbook.xml.rels", rels)
        for i, n in enumerate(names, 1):
            cells = "".join(f'<row r="{r}"><c r="A{r}"><v>1</v></c><c r="B{r}"><v>2</v></c>'
                            f'<c r="C{r}"><f>{f}</f><v>{v}</v></c></row>' for r, (f, v) in enumerate(sheets[n], 1))
            z.writestr(f"xl/worksheets/sheet{i}.xml",
                       f'<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{cells}</sheetData></worksheet>')
    return buf.getvalue()


STALE = {"Sheet1": [("A1+B1", "999")]}   # really 3, but the file says 999


@needs_soffice
def test_a_document_converts_to_text_and_to_other_word_formats():
    ((name, txt),) = office.convert(RTF, "review.rtf", "txt")
    assert name == "review.txt" and "revenue grew by twelve percent" in txt.decode("utf-8-sig", "replace")
    ((_, docx),) = office.convert(RTF, "review.rtf", "docx")
    assert zipfile.is_zipfile(io.BytesIO(docx))


@needs_soffice
def test_a_workbook_exports_one_csv_per_sheet_or_just_the_one_asked_for():
    book = workbook({"First": [("A1+B1", "3")], "Second": [("A1*B1", "2")]})
    both = dict(office.convert(book, "book.xlsx", "csv"))
    assert sorted(both) == ["book-First.csv", "book-Second.csv"]
    ((name, one),) = office.convert(book, "book.xlsx", "csv", sheet=2)
    assert name == "book-Second.csv" and one.decode().strip().endswith(",2")


@needs_soffice
def test_recalculating_replaces_a_stale_cached_value_and_plain_conversion_does_not():
    book = workbook(STALE)
    ((_, plain),) = office.convert(book, "m.xlsx", "csv", sheet=1)
    assert plain.decode().strip() == "1,2,999"                      # trusts what the file says
    ((_, fresh_xlsx),) = office.convert(book, "m.xlsx", "xlsx", recalculate=True)
    ((_, fresh),) = office.convert(fresh_xlsx, "m.xlsx", "csv", sheet=1)
    assert fresh.decode().strip() == "1,2,3"


@needs_soffice
def test_pages_of_a_document_are_drawn_as_images():
    total, images = vision.load_images(RTF, "review.rtf", [1])
    assert total == 1 and len(images) == 1 and images[0][2] == "image/png" and "review.rtf page 1" in images[0][0]


@needs_soffice
def test_a_damaged_file_is_reported_not_raised_as_a_crash():
    with pytest.raises(office.OfficeError, match="not a valid .docx"):
        office.convert(b"this is not a docx", "bad.docx", "txt")
    with pytest.raises(office.OfficeError, match="could not be converted|not a valid"):
        office.convert(b"PK\x03\x04 truncated", "cut.xlsx", "csv")


def test_the_formats_a_file_can_become_are_checked_before_any_work():
    with pytest.raises(office.OfficeError, match="can be converted to: .*csv"):
        office.convert(b"x", "book.xlsx", "txt")
    with pytest.raises(office.OfficeError, match="not a Word, Excel or PowerPoint"):
        office.convert(b"x", "photo.png", "pdf")


def test_which_files_count_as_office():
    assert vision.is_office("a/Deck.PPTX") and vision.is_office("x.ods")
    assert not vision.is_office("scan.pdf") and not vision.is_office("photo.png") and not vision.is_office("noext")
    assert office.family("a.docx") == "word" and office.family("a.xlsx") == "sheet" and office.family("a.pptx") == "slide"


# ---- the tools, against a fake sandbox ------------------------------------------------

@pytest.fixture
def sandbox(monkeypatch):
    files: dict[str, bytes] = {}

    async def read_file(thread_id, name):
        if name not in files:
            raise FileNotFoundError(name)
        return files[name], "application/octet-stream", name

    async def write_file(thread_id, name, content):
        files[name] = content

    monkeypatch.setattr(tools_mod, "read_file", read_file)
    monkeypatch.setattr(tools_mod, "write_file", write_file)
    return files


async def call(tool, **args):
    return await tool.ainvoke({"label": "x", **args}, config={"configurable": {"thread_id": "t"}})


@needs_soffice
async def test_office_convert_writes_the_result_beside_the_original(sandbox):
    sandbox["docs/review.rtf"] = RTF
    out = await call(office_convert, path="docs/review.rtf", format="txt")
    assert "docs/review.txt" in out and b"twelve percent" in sandbox["docs/review.txt"]


@needs_soffice
async def test_office_recalculate_saves_a_new_file_and_keeps_the_original(sandbox):
    sandbox["m.xlsx"] = original = workbook(STALE)
    out = await call(office_recalculate, path="m.xlsx")
    assert "m-recalculated.xlsx" in out and sandbox["m.xlsx"] == original
    ((_, csv),) = office.convert(sandbox["m-recalculated.xlsx"], "r.xlsx", "csv", sheet=1)
    assert csv.decode().strip() == "1,2,3"


async def test_the_tools_say_what_is_wrong_in_words(sandbox):
    sandbox["a.rtf"] = RTF
    assert "not a Word, Excel or PowerPoint file" in await call(office_convert, path="x.png", format="pdf")
    assert "can be converted to" in await call(office_convert, path="a.rtf", format="csv")
    assert "not a spreadsheet" in await call(office_recalculate, path="a.rtf")
    assert "nope.docx is not in the sandbox" in await call(office_convert, path="nope.docx", format="txt")
    assert "inside your working directory" in await call(office_convert, path="../a.docx", format="txt")


def test_a_404_from_the_sandbox_says_to_fetch_the_file_first():
    from types import SimpleNamespace

    from app.agents.tools.sandbox_errors import read_error

    not_there = RuntimeError("Failed to communicate with the sandbox at http://router/download/status.docx")
    not_there.response = SimpleNamespace(status_code=404)      # what SandboxRequestError carries
    msg = read_error("status.docx", not_there)
    assert "not in the sandbox" in msg and "workspace_copy_to_sandbox" in msg and "http://" not in msg
    assert "could not read" in read_error("status.docx", RuntimeError("boom"))   # other failures keep their detail
