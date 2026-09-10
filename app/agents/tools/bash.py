import httpx
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.tools.sandbox_manager import exec_bash, get_thread_id


@tool
async def bash(command: str, label: str, config: RunnableConfig) -> str:
    """Execute a bash command inside the sandbox and return stdout.

    Each call starts in `/workspace`, a private directory for this
    conversation — nothing else is in there and no other conversation can see
    it. Files you write with a plain relative path (`report.docx`,
    `out/chart.png`) or an absolute one under `/workspace` persist across bash
    calls in this conversation. A fresh shell each call, so chain steps with
    `&&` or write a script and run it.

    The environment is FIXED and OFFLINE — you cannot install packages
    (`pip`/`uv` are removed, the filesystem is read-only, there is no network)
    and cannot fetch anything from a URL. Work with what's provided. Available:
    Python 3.11 with numpy, pandas, scipy, statsmodels, pyarrow, scikit-learn,
    xgboost, lightgbm, matplotlib, seaborn, plotly, nltk (corpora bundled),
    beautifulsoup4/lxml, python-docx, python-pptx, openpyxl, xlsxwriter,
    reportlab, fpdf2, pillow, jinja2, and pandoc for generating
    .docx/.pptx/.xlsx/.pdf files. After creating a file, pass its path to
    `present_file` to hand it to the user — a relative name or the
    `/workspace/...` form both work: `present_file("report.docx", ...)`.

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

    try:
        result = await exec_bash(thread_id, command)
    except httpx.HTTPError as e:
        return f"Error: sandbox unavailable ({e})."

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

    return output or "(no output)"
