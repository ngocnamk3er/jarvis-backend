import subprocess

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.tools.sandbox_manager import (
    ensure_container_running,
    exec_bash_in_sandbox,
    get_thread_id,
    mask_real_paths,
)


@tool
def bash(command: str, config: RunnableConfig) -> str:
    """Execute a bash command inside the sandbox and return stdout.

    Three persistent directories are available:
    - /workspace  : working directory (default cwd)
    - /output     : save files here to show them to the user
    - /upload     : user-uploaded files available for reading

    Common uses:
        bash("ls /workspace")
        bash("pip install pandas -q")
        bash("cp /upload/data.csv /workspace/data.csv")
        bash("wc -l /workspace/*.csv")

    Args:
        command: Bash command to execute.
    """
    thread_id = get_thread_id(config)

    try:
        ensure_container_running()
        result = exec_bash_in_sandbox(thread_id, command)
    except FileNotFoundError:
        return "Error: Docker is not available on this system."
    except subprocess.TimeoutExpired:
        return "Error: command timed out (60s limit)."

    output = ""
    if result.stdout.strip():
        output += mask_real_paths(result.stdout.strip(), thread_id)
    if result.returncode != 0 and result.stderr.strip():
        stderr = mask_real_paths(result.stderr.strip(), thread_id)
        output += f"\nStderr:\n{stderr}" if output else f"Stderr:\n{stderr}"
    if result.returncode != 0 and not output:
        output = f"Command exited with code {result.returncode}"

    return output or "(no output)"
