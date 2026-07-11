import subprocess

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.tools.sandbox_manager import (
    exec_bash_in_sandbox,
    get_thread_id,
    mask_real_paths,
)


@tool
def bash(command: str, label: str, config: RunnableConfig) -> str:
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

    IMPORTANT — package installation timeouts:
        When apt-get or pip install times out, the process continues running
        in the background inside the container. Do NOT immediately retry with
        a different package name — you will hit a dpkg/pip lock conflict.
        Instead, wait then check: `which java`, `dpkg -l | grep openjdk`,
        `pip show pandas`, etc. Only retry if the package is truly absent.

    Args:
        command: Bash command to execute.
        label: Brief human-readable description shown to the user (e.g. "Running fibonacci script", "Installing pandas").
    """
    thread_id = get_thread_id(config)

    try:
        result = exec_bash_in_sandbox(thread_id, command)
    except FileNotFoundError:
        return "Error: Docker is not available on this system."
    except subprocess.TimeoutExpired:
        return (
            "Error: command timed out (300s limit). "
            "If this was a package install (apt-get/pip), the process may still be running "
            "in the background. Check first with 'which <binary>' or 'dpkg -l | grep <pkg>' "
            "before retrying — do NOT retry with a different package name immediately."
        )

    output = ""
    if result.stdout.strip():
        output += mask_real_paths(result.stdout.strip(), thread_id)
    if result.returncode != 0 and result.stderr.strip():
        stderr = mask_real_paths(result.stderr.strip(), thread_id)
        output += f"\nStderr:\n{stderr}" if output else f"Stderr:\n{stderr}"
    if result.returncode != 0 and not output:
        output = f"Command exited with code {result.returncode}"

    return output or "(no output)"
