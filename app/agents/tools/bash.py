from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agents.tools.sandbox_manager import exec_bash, get_thread_id

# Much higher than save_and_stub()'s threshold (10,000 chars, for
# web_search/web_fetch/read_file) — bash is how the model extracts what it
# needs from a saved file, so capping it that tight just pushes the "can't
# see the row I'm looking for" problem down a level: a mid-sized excerpt
# (e.g. a 40-line slice of a Wikipedia infobox) can easily clear 10-20k
# chars while the one field being searched for lands in the omitted middle,
# forcing round after round of blind narrowing to find it. Verified live:
# a real conversation needed 13 bash round-trips to locate two infobox
# fields under the old 2,000-char cap. 20,000 gives a wide excerpt (or
# several targeted grep hits) a real chance of landing whole.
# Head+tail together cover 80% of the threshold (split evenly) — high
# enough that crossing _MAX_INLINE_CHARS by a few chars doesn't fall off a
# cliff into a tiny preview; still leaves a real gap so capping means
# something. Must stay under _MAX_INLINE_CHARS or `omitted` below goes
# negative.
_MAX_INLINE_CHARS = 20_000
_PREVIEW_HEAD = 8_000
_PREVIEW_TAIL = 8_000


def _cap_output(output: str) -> str:
    """Truncate to a head+tail preview above `_MAX_INLINE_CHARS`.

    Unlike web_search/web_fetch results, bash output is cheap to regenerate —
    same sandbox, no network, fully deterministic — so there's nothing to
    save to a file or recall later. If the middle mattered, re-run a
    narrower command instead (grep / head / tail / sed -n / a smaller
    Python snippet) rather than reading the whole thing again.
    """
    n = len(output)
    if n <= _MAX_INLINE_CHARS:
        return output
    head = output[:_PREVIEW_HEAD].rstrip()
    tail = output[-_PREVIEW_TAIL:].lstrip()
    omitted = n - _PREVIEW_HEAD - _PREVIEW_TAIL
    # No "re-run a narrower command" hint here — that guidance lives once,
    # in the system prompt's "Large tool outputs" section.
    return (
        f"[bash output — {n:,} chars, showing head+tail]\n"
        f"--- head ---\n{head}\n"
        f"... [{omitted:,} chars omitted] ...\n"
        f"--- tail ---\n{tail}"
    )


@tool
async def bash(command: str, label: str, config: RunnableConfig) -> str:
    """Execute a bash command inside the sandbox and return stdout.

    Each call runs in a private working directory for this conversation —
    nothing else is in there and no other conversation can see it. Files you
    write with a plain relative path (`report.docx`, `out/chart.png`) persist
    across bash calls in this conversation. Don't assume a specific absolute
    path for the working directory — skip `mkdir`/`ls` on a literal
    `/workspace`, just use relative paths (run `pwd` first if you ever need
    to confirm where you are). A fresh shell each call, so chain steps with
    `&&` or write a script and run it.

    Output over ~20,000 chars comes back as a head+tail preview, not the
    whole thing — e.g. `cat`-ing a very large file just shows you the ends
    of it. Pipe through `grep`/`head`/`tail`/`sed -n` or a short Python
    snippet to pull out the specific part you need instead of dumping a
    whole file.

    The sandbox is a plain Python 3.14 environment. Installed: numpy, pandas,
    scipy, scikit-learn, lightgbm, xgboost, statsmodels, pyarrow, matplotlib,
    seaborn, plotly, pillow, opencv (cv2), pytesseract, openpyxl, xlrd,
    xlsxwriter, python-docx, python-pptx, reportlab, fpdf2, pypdf, pdfplumber,
    beautifulsoup4, lxml, requests, jinja2, pyyaml and svgwrite; and the
    command-line tools file, curl, unzip, zip, jq, pandoc, pdftotext, pdftoppm
    and tesseract (languages: eng, vie, jpn, chi_sim, chi_tra, kor, rus, tha, ara, fra, deu, spa). Use these rather than guessing at what
    exists. Do not count on `pip install` or on the network.

    Never estimate pixel coordinates by eye: to mark or locate things on an image
    or on a page of a PDF, use `ocr_image` and then `draw_boxes`.

    You cannot SEE an image from bash: to look at a picture or a page, use
    `view_image`. For PDFs use `check_pdf` and `read_pdf`; for Word, Excel and
    PowerPoint files `office_convert` and `office_recalculate` give you their
    content as plain files. The libraries above are for making and changing
    files: draw on an image with pillow, build a workbook with openpyxl, write a
    report with python-docx or reportlab, then pass its path to `present_file` to
    hand it to the user: `present_file("report.docx", ...)`.

    Common uses:
        bash("ls -la")
        bash("python analyze.py")
        bash("python -c 'import pandas as pd; print(pd.read_csv(\\"data.csv\\").describe())'")

    IMPORTANT — timeouts:
        A command past the 300s limit is killed (its process tree too). If a
        command times out, retry it; if it keeps timing out, split the work.

    Args:
        command: Bash command to execute.
        label: Brief human-readable description shown to the user (e.g. "Running the analysis script", "Rendering the chart").
    """
    thread_id = get_thread_id(config)

    result = await exec_bash(thread_id, command)

    if result.get("error"):
        return f"Error: no sandbox available. {result['error']}"
    if result.get("timed_out"):
        return "Error: command timed out (300s limit) and was killed."

    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    exit_code = result.get("exit_code")

    output = stdout
    if exit_code not in (0, None) and stderr:
        output += f"\nStderr:\n{stderr}" if output else f"Stderr:\n{stderr}"
    if exit_code not in (0, None) and not output:
        output = f"Command exited with code {exit_code}"

    return _cap_output(output) if output else "(no output)"
