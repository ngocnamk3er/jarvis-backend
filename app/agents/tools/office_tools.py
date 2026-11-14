"""office_convert and office_recalculate: Word, Excel and PowerPoint files in the
sandbox, which has no library for them. The conversion runs on the server
(app/agents/office.py) and the result is written back into the sandbox."""

import asyncio
from pathlib import PurePosixPath

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents import office
from app.agents.tools.sandbox_errors import read_error
from app.agents.tools.sandbox_manager import (
    get_thread_id,
    normalize_workspace_path,
    read_file,
    write_file,
)


def _check_path(path: str) -> tuple[str | None, str | None]:
    """(normalised name, error)"""
    name = normalize_workspace_path(path)
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        return None, "Error: path must be inside your working directory (no '..', no absolute path)."
    if office.family(name) is None:
        return None, f"Error: {name} is not a Word, Excel or PowerPoint file."
    return name, None


async def _run(name: str, target: str, config: RunnableConfig, *, sheet: int | None = None,
               recalculate: bool = False, suffix: str = "") -> str:
    thread_id = get_thread_id(config)
    try:
        data, _, filename = await read_file(thread_id, name)
    except Exception as e:  # noqa: BLE001 — a missing file or a sandbox hiccup, same to the model
        return read_error(name, e)
    try:
        outputs = await asyncio.to_thread(office.convert, data, filename, target, sheet=sheet, recalculate=recalculate)
    except office.OfficeError as e:
        return f"Error: {e}"
    folder = str(PurePosixPath(name).parent)
    written = []
    for out_name, content in outputs:
        stem, dot, ext = out_name.rpartition(".")
        out_name = f"{stem}{suffix}.{ext}"
        dest = out_name if folder == "." else f"{folder}/{out_name}"
        try:
            await write_file(thread_id, dest, content)
        except Exception as e:  # noqa: BLE001
            return f"Error: could not write {dest} into the sandbox ({type(e).__name__}: {e})."
        written.append(f"{dest} ({len(content) >> 10} KB)")
    return f"Converted {filename} to {target}: " + ", ".join(written) + "."


@tool
async def office_convert(path: str, format: str, label: str, config: RunnableConfig, sheet: int | None = None) -> str:
    """Convert a Word, Excel or PowerPoint file in your bash sandbox to another format.

    The result is written beside the original with the new extension. This is
    how to read these files, since the sandbox has no library for them.
    Word (.doc .docx .odt .rtf) -> pdf, docx, odt, txt.
    Excel (.xls .xlsx .ods .csv) -> pdf, xlsx, ods, csv.
    PowerPoint (.ppt .pptx .odp) -> pdf, pptx, odp.
    Use `txt` to get a document's words and `csv` to get a workbook's cells,
    then read them with pandas or bash. A workbook converted to csv is written
    as one file per sheet, named for the sheet (`book-Sheet1.csv`); pass `sheet`
    (1-based) to export just that one. To see how a document looks, use `view_image` instead.

    Args:
        path: The file's path in your working directory, e.g. "report.docx".
        format: The format to convert to, e.g. "txt", "csv" or "pdf".
        sheet: Excel to csv only: the sheet to export, counting from 1.
        label: Brief human-readable description shown to the user (e.g. "Converting the workbook to CSV").
    """
    name, err = _check_path(path)
    if err:
        return err
    return await _run(name, format, config, sheet=sheet)


@tool
async def office_recalculate(path: str, label: str, config: RunnableConfig) -> str:
    """Recalculate every formula in an Excel workbook in your bash sandbox.

    A workbook written by a script (openpyxl, xlsxwriter, pandas) holds formulas
    but no computed values until a spreadsheet program opens it. This opens it,
    recalculates, and saves the result as a new file beside the original,
    `<name>-recalculated.xlsx`, leaving the original untouched. Then read the
    values with `office_convert` to csv.

    Args:
        path: The workbook's path in your working directory, e.g. "model.xlsx".
        label: Brief human-readable description shown to the user (e.g. "Recalculating the formulas").
    """
    name, err = _check_path(path)
    if err:
        return err
    if office.family(name) != "sheet":
        return f"Error: {name} is not a spreadsheet."
    return await _run(name, "xlsx", config, recalculate=True, suffix="-recalculated")
