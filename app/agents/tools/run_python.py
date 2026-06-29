import subprocess

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.messages import RunPythonMsg
from app.agents.tools.sandbox_manager import (
    ensure_container_running,
    exec_in_sandbox,
    mask_real_paths,
)


@tool
def run_python(code: str, config: RunnableConfig) -> str:
    """Execute Python code in the conversation's sandbox and return stdout.

    Three persistent directories are available:
    - /workspace  : working directory, files here survive across all calls in this conversation
    - /output     : save files here, then call represent_file("/output/<filename>") to show them
    - /upload     : user-uploaded files available for reading

    Use print() to produce output.

    Args:
        code: Python code to execute.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    try:
        ensure_container_running()
        result = exec_in_sandbox(thread_id, code)
    except FileNotFoundError:
        return RunPythonMsg.DOCKER_NOT_AVAILABLE
    except subprocess.TimeoutExpired:
        return RunPythonMsg.TIMEOUT.format(timeout=300)

    if result.returncode != 0:
        stderr = mask_real_paths(result.stderr.strip(), thread_id)
        stdout = mask_real_paths(result.stdout.strip(), thread_id)
        if result.returncode == 137 and not stderr:
            hint = "Process was killed (OOM) — likely exceeded memory limit. Use smaller data or batches."
            return f"Error (OOM kill):\n{hint}" + (f"\n\nPartial output:\n{stdout}" if stdout else "")
        msg = stderr or f"Process exited with code {result.returncode}"
        return RunPythonMsg.EXEC_ERROR.format(stderr=msg) + (f"\n\nPartial output:\n{stdout}" if stdout else "")

    stdout = mask_real_paths(result.stdout.strip(), thread_id)
    return stdout or RunPythonMsg.NO_OUTPUT
