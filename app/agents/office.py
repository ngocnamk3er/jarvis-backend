"""Word, Excel and PowerPoint files, converted with LibreOffice running headless.

The sandbox has no library for these formats, so the backend does the work: a
file is read out of the sandbox, converted here, and the result goes back in as
a new file (office_convert, office_recalculate) or is drawn as page images
(view_image, via a PDF). It is the same idea as DeepSeek Harness's Office kit
(office_render / office_convert / office_recalculate).

Every call gets its own throw-away LibreOffice profile in a temp directory:
LibreOffice locks its profile, so sharing one would make concurrent
conversions fail, and a profile is also where "recalculate on load" is set.
"""

import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 120
MAX_INPUT_BYTES = 50 << 20
_slots = threading.Semaphore(2)   # LibreOffice is heavy; at most two conversions at once

FAMILIES = {
    "word": {".doc", ".docx", ".odt", ".rtf"},
    "sheet": {".xls", ".xlsx", ".ods", ".csv"},
    "slide": {".ppt", ".pptx", ".odp"},
}
TARGETS = {
    "word": {"pdf", "docx", "odt", "txt"},
    "sheet": {"pdf", "xlsx", "ods", "csv"},
    "slide": {"pdf", "pptx", "odp"},
}
OFFICE_EXTENSIONS = frozenset().union(*FAMILIES.values())
_ZIP_FORMATS = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}


class OfficeError(Exception):
    """The conversion cannot be done; the message is written for the model."""


def family(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    return next((f for f, exts in FAMILIES.items() if ext in exts), None)


def available() -> bool:
    return shutil.which("soffice") is not None


# Make LibreOffice recalculate every formula when it opens an OOXML or ODF
# workbook (the default is to trust the values saved in the file, which a file
# written by a script does not have).
_RECALC_CONFIG = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0</value></prop></item>
<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop></item>
</oor:items>
"""


def _csv_filter(sheet: int | None) -> str:
    # token 12: 0 = first sheet only, -1 = every sheet to its own file, N = the Nth sheet
    which = -1 if sheet is None else sheet
    return f'csv:Text - txt - csv (StarCalc):44,34,76,1,,0,false,true,false,false,false,{which}'


def _filter(target: str, sheet: int | None) -> str:
    if target == "csv":
        return _csv_filter(sheet)
    if target == "txt":
        return "txt:Text (encoded):UTF8"
    return target


def convert(data: bytes, filename: str, target: str, *, sheet: int | None = None,
            recalculate: bool = False) -> list[tuple[str, bytes]]:
    """[(output file name, bytes)] — one file, or one per sheet for csv of a whole
    workbook. Blocking; call it from a thread."""
    fam = family(filename)
    if fam is None:
        raise OfficeError(f"{filename} is not a Word, Excel or PowerPoint file.")
    target = target.lower().lstrip(".")
    if target not in TARGETS[fam]:
        raise OfficeError(f"A {fam} file can be converted to: {', '.join(sorted(TARGETS[fam]))}.")
    if len(data) > MAX_INPUT_BYTES:
        raise OfficeError(f"{filename} is {len(data) >> 20} MB, over the {MAX_INPUT_BYTES >> 20} MB limit.")
    # LibreOffice opens a file it cannot recognise as plain text and "converts" it,
    # so garbage under a .docx name would come back as a successful, meaningless
    # result. These formats are zip containers; a file that is not one is damaged
    # (or is not really that format).
    if Path(filename).suffix.lower() in _ZIP_FORMATS and not zipfile.is_zipfile(io.BytesIO(data)):
        raise OfficeError(f"{filename} is not a valid {Path(filename).suffix.lower()} file: it may be damaged, or be another type of file renamed.")
    if not available():
        raise OfficeError("The document converter (LibreOffice) is not installed on the server.")

    with _slots, tempfile.TemporaryDirectory(prefix="office-") as tmp:
        root = Path(tmp)
        src = root / f"in{Path(filename).suffix.lower()}"      # a plain name: no quoting, no traversal
        src.write_bytes(data)
        out_dir = root / "out"
        profile = root / "profile"
        if recalculate:
            (profile / "user").mkdir(parents=True)
            (profile / "user" / "registrymodifications.xcu").write_text(_RECALC_CONFIG)
        cmd = ["soffice", "--headless", "--norestore", "--nologo", "--nodefault",
               f"-env:UserInstallation=file://{profile}",
               "--convert-to", _filter(target, sheet), "--outdir", str(out_dir), str(src)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_SECONDS, env={**os.environ, "HOME": tmp})
        except subprocess.TimeoutExpired:
            raise OfficeError(f"Converting {filename} took longer than {TIMEOUT_SECONDS} seconds and was stopped.") from None
        produced = sorted(out_dir.glob("*")) if out_dir.exists() else []
        if proc.returncode != 0 or not produced:
            logger.warning("soffice failed (%s): %s", proc.returncode, proc.stderr[-300:])
            raise OfficeError(f"{filename} could not be converted; it may be damaged or password-protected.")
        stem = Path(filename).stem
        results = []
        for f in produced:
            # in.csv -> in-Sheet1.csv for a multi-sheet export; keep the sheet part
            suffix = f.stem[len("in"):] if f.stem.startswith("in") else ""
            results.append((f"{stem}{suffix}{f.suffix}", f.read_bytes()))
        return results


class _PdfCache:
    """Converted PDFs by content hash: looking at several pages of one document
    should convert it once, not once per call."""

    def __init__(self, limit: int = 4):
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._limit = limit

    def get_or_convert(self, data: bytes, filename: str) -> bytes:
        key = hashlib.sha1(data).hexdigest() + Path(filename).suffix.lower()
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        ((_, pdf),) = convert(data, filename, "pdf")
        self._items[key] = pdf
        while len(self._items) > self._limit:
            self._items.popitem(last=False)
        return pdf


pdf_cache = _PdfCache()
