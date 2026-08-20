from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from opensandbox.exceptions.sandbox import SandboxException

from app.agents.tools.sandbox_manager import exec_bash_in_sandbox, get_thread_id


@tool
async def bash(command: str, label: str, config: RunnableConfig) -> str:
    """Execute a bash command inside the sandbox and return stdout.

    One persistent directory is available:
    - /workspace  : working directory (default cwd), persists across calls
                    within the same conversation

    Common uses:
        bash("ls /workspace")
        bash("pip install pandas -q")
        bash("wc -l /workspace/*.csv")

    IMPORTANT — package installation timeouts:
        When apt-get or pip install runs past the 300s command timeout, it is
        killed — the process does NOT keep running in the background (unlike
        a plain subprocess). If a command times out, just retry it; if it
        keeps timing out, break the work into smaller steps.

    Args:
        command: Bash command to execute.
        label: Brief human-readable description shown to the user (e.g. "Running fibonacci script", "Installing pandas").
    """
    thread_id = get_thread_id(config)

    try:
        execution = await exec_bash_in_sandbox(thread_id, command)
    except SandboxException as e:
        return f"Error: sandbox unavailable ({e})."

    stdout = "\n".join(m.text for m in execution.logs.stdout).strip()
    stderr = "\n".join(m.text for m in execution.logs.stderr).strip()

    output = ""
    if stdout:
        output += stdout
    if execution.exit_code and execution.exit_code != 0 and stderr:
        output += f"\nStderr:\n{stderr}" if output else f"Stderr:\n{stderr}"
    if execution.exit_code and execution.exit_code != 0 and not output:
        if execution.error and "killed" in " ".join(execution.error.traceback or []):
            output = "Error: command timed out (300s limit) and was killed."
        else:
            output = f"Command exited with code {execution.exit_code}"

    return output or "(no output)"
