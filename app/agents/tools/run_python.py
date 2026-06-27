import subprocess
from langchain_core.tools import tool

from app.agents.tools.messages import RunPythonMsg

_DOCKER_IMAGE = "python:3.12-slim"
_TIMEOUT_SECONDS = 15


@tool
def run_python(code: str) -> str:
    """Execute Python code in an isolated Docker container and return the output.
    The container has no network access, 128 MB memory limit, and a 15-second timeout.
    """
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "128m",
                "--cpus", "0.5",
                "--security-opt", "no-new-privileges:true",
                "--read-only",
                "--tmpfs", "/tmp:size=32m",
                "-i",
                _DOCKER_IMAGE,
                "python", "-",
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return RunPythonMsg.DOCKER_NOT_AVAILABLE
    except subprocess.TimeoutExpired:
        return RunPythonMsg.TIMEOUT.format(timeout=_TIMEOUT_SECONDS)

    if result.returncode != 0:
        return RunPythonMsg.EXEC_ERROR.format(stderr=result.stderr.strip())
    return result.stdout.strip() or RunPythonMsg.NO_OUTPUT
