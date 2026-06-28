import subprocess
from langchain_core.tools import tool

from app.agents.messages import RunPythonMsg

_DOCKER_IMAGE = "jarvis-sandbox"
_PIP_CACHE_VOLUME = "jarvis-pip-cache"


@tool
def run_python(code: str) -> str:
    """Execute Python code in an isolated Docker container and return stdout.

    Each call starts fresh — no state is shared between calls.
    Use print() to produce output.

    Args:
        code: Python code to execute.
    """
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--memory", "512m",
                "--cpus", "1.0",
                "--security-opt", "no-new-privileges:true",
                "--tmpfs", "/tmp:size=512m,exec",
                "-v", f"{_PIP_CACHE_VOLUME}:/root/.cache/pip",
                "-i", _DOCKER_IMAGE,
                "python", "-",
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        return RunPythonMsg.DOCKER_NOT_AVAILABLE
    except subprocess.TimeoutExpired:
        return RunPythonMsg.TIMEOUT.format(timeout=300)

    if result.returncode != 0:
        return RunPythonMsg.EXEC_ERROR.format(stderr=result.stderr.strip())
    return result.stdout.strip() or RunPythonMsg.NO_OUTPUT
