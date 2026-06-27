class RunPythonMsg:
    DOCKER_NOT_AVAILABLE = "Error: Docker is not available on this system."
    TIMEOUT = "Error: execution timed out ({timeout}s limit)."
    EXEC_ERROR = "Error:\n{stderr}"
    NO_OUTPUT = "(no output)"


class CalculatorMsg:
    INVALID_CHARS = "Error: only basic arithmetic operators are allowed."
    EVAL_ERROR = "Error: {error}"


class WebSearchMsg:
    NO_RESULTS = "No results found."


class CreateFileMsg:
    DOCKER_NOT_AVAILABLE = "Error: Docker is not available on this system."
    TIMEOUT = "Error: execution timed out ({timeout}s limit)."
    EXEC_ERROR = "Error:\n{stderr}"
    FILE_NOT_CREATED = "Error: code ran but no file was written to OUTPUT_PATH."
    SUCCESS = "✓ **{filename}** created.\n\n[⬇ Download {filename}]({url})"
