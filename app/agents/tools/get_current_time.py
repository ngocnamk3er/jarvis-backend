from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time(label: str) -> str:
    """Return the current date and time.

    Args:
        label: Brief human-readable description shown to the user (e.g. "Getting current time").
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
